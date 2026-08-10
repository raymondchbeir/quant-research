from __future__ import annotations

import json, os, threading, traceback, urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests

FROZEN_SERIES={"KXBNB15M","KXDOGE15M","KXETH15M","KXHYPE15M","KXNEAR15M","KXSOL15M","KXXRP15M","KXZEC15M"}
QTY=3.0; DEPTH_C=3.0; LIFETIME_SEC=15.0; MAX_SPREAD_C=2.0; EXIT_PRICE=0.10
COINBASE_CANDLES_URL="https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60"
COINBASE_TICKER_URL="https://api.exchange.coinbase.com/products/BTC-USD/ticker"; COINBASE_WS_URL="wss://ws-feed.exchange.coinbase.com"
KALSHI_REST_BASE="https://external-api.kalshi.com/trade-api/v2"

def _ts(x):return pd.to_datetime(x,utc=True,errors="coerce")
def _num(x,default=np.nan):
    try:
        y=float(x);return y if np.isfinite(y) else default
    except Exception:return default
def _price(x):
    p=_num(x)
    if np.isfinite(p) and p>1.5:p/=100.0
    return p
def _read_recent_jsonl(path,max_bytes):
    if not path.exists() or path.stat().st_size==0:return []
    with path.open("rb") as f:
        size=f.seek(0,os.SEEK_END);start=max(0,size-max_bytes);f.seek(start)
        if start:f.readline()
        raw=f.read().decode("utf-8",errors="ignore")
    out=[]
    for line in raw.splitlines():
        try:out.append(json.loads(line))
        except Exception:pass
    return out

class ShadowTrader:
    def __init__(self,session_dir):
        self.session_dir=Path(session_dir);self.ticker_file=self.session_dir/"ticker_updates.jsonl";self.trades_file=self.session_dir/"trades.jsonl";self.book_file=self.session_dir/"full_books.jsonl"
        for p in (self.ticker_file,self.trades_file,self.book_file):
            if not p.exists():raise FileNotFoundError(f"Missing recorder file: {p}")
        self.out_dir=self.session_dir/"SHADOW_15S_3C_V1";self.out_dir.mkdir(parents=True,exist_ok=True);self.event_file=self.out_dir/"shadow_events.jsonl";self.state_file=self.out_dir/"shadow_records.csv"
        self.lock,self.log_lock,self.stop_event=threading.RLock(),threading.Lock(),threading.Event();self.started_at=pd.Timestamp.now(tz="UTC")
        self.quotes=defaultdict(lambda:deque(maxlen=10000));self.books=defaultdict(lambda:deque(maxlen=1200));self.trades=defaultdict(lambda:deque(maxlen=50000));self.btc=deque(maxlen=100000)
        self.close_times={};self.decisions_done=set();self.entry_orders={};self.exit_orders={};self.positions={};self.results={};self.records={};self.threads=[];self._rest=requests.Session()
    def emit(self,event,ticker=None,detail=None,**extra):
        row={"time":pd.Timestamp.now(tz="UTC").isoformat(),"event":event,"ticker":ticker,"detail":detail,**extra}
        with self.log_lock:
            with self.event_file.open("a",encoding="utf-8") as f:f.write(json.dumps(row,default=str,separators=(",",":"))+"\n")
        if event in {"SIGNAL","ENTRY_POST","ENTRY_FILL","ENTRY_CANCEL","EXIT_TRIGGER","EXIT_POST","EXIT_FILL","SETTLED","ERROR"}:
            prefix=pd.Timestamp.now(tz="UTC").strftime("[%H:%M:%S UTC]");print(f"{prefix} {event}"+(f" | {ticker}" if ticker else "")+(f" | {detail}" if detail else ""))
    def _record(self,ticker):
        if ticker not in self.records:self.records[ticker]={"ticker":ticker,"series":ticker.split("-")[0],"decision_time":pd.NaT,"direction":None,"midpoint":np.nan,"spread_c":np.nan,"btc_return":np.nan,"entry_price":np.nan,"entry_queue":np.nan,"entry_fill_qty":0.0,"entry_first_fill":pd.NaT,"entry_last_fill":pd.NaT,"exit_trigger":pd.NaT,"exit_post":pd.NaT,"exit_queue":np.nan,"exit_fill_qty":0.0,"result":None,"realized_pnl":0.0,"status":None}
        return self.records[ticker]
    def _save_state(self):
        with self.lock:df=pd.DataFrame(list(self.records.values()))
        if len(df):
            tmp=self.state_file.with_suffix(".tmp");df.sort_values("decision_time",na_position="last").to_csv(tmp,index=False);tmp.replace(self.state_file)
    @staticmethod
    def _held_quote(direction,yes_bid,yes_ask):return (yes_bid,yes_ask) if direction=="YES" else (1.0-yes_ask,1.0-yes_bid)
    def _quote_at(self,ticker,when):
        for row in reversed(self.quotes.get(ticker,())):
            if row[0]<=when:return row
        return None
    def _book_at(self,ticker,when):
        for row in reversed(self.books.get(ticker,())):
            if row[0]<=when:return row
        return None
    def _btc_append(self,ts,price):
        t,p=_ts(ts),_num(price)
        if pd.isna(t) or not np.isfinite(p) or p<=0:return
        with self.lock:self.btc.append((t,float(p)))
    def _btc_price_at(self,when):
        for ts,price in reversed(self.btc):
            if ts<=when:return ts,price
        return None
    def _btc_return_15m(self,when):
        a,b=self._btc_price_at(when),self._btc_price_at(when-pd.Timedelta(minutes=15))
        return None if a is None or b is None else (a[1]/b[1]-1.0,a,b)
    def _seed_coinbase(self):
        req=urllib.request.Request(COINBASE_CANDLES_URL,headers={"User-Agent":"kalshi-shadow-research/1.0"})
        with urllib.request.urlopen(req,timeout=10) as r:candles=json.loads(r.read().decode("utf-8"))
        seed=[]
        for row in candles:
            if len(row)>=5:seed.append((pd.to_datetime(row[0],unit="s",utc=True)+pd.Timedelta(minutes=1),float(row[4])))
        seed.sort(key=lambda x:x[0])
        with self.lock:
            for item in seed:self.btc.append(item)
        print(f"Coinbase BTC history seeded: {len(seed)} 1m candles")
    def _coinbase_loop(self):
        try:
            import websocket
            while not self.stop_event.is_set():
                try:
                    def on_open(ws):ws.send(json.dumps({"type":"subscribe","product_ids":["BTC-USD"],"channels":["ticker"]}))
                    def on_message(ws,message):
                        try:
                            x=json.loads(message)
                            if x.get("type")=="ticker" and x.get("product_id")=="BTC-USD":self._btc_append(x.get("time"),x.get("price"))
                        except Exception:pass
                    ws=websocket.WebSocketApp(COINBASE_WS_URL,on_open=on_open,on_message=on_message);ws.run_forever(ping_interval=20,ping_timeout=10)
                except Exception as exc:self.emit("ERROR",detail=f"Coinbase websocket: {exc!r}")
                if not self.stop_event.is_set():self.stop_event.wait(2)
        except Exception:
            while not self.stop_event.is_set():
                try:
                    req=urllib.request.Request(COINBASE_TICKER_URL,headers={"Cache-Control":"no-cache","User-Agent":"kalshi-shadow-research/1.0"})
                    with urllib.request.urlopen(req,timeout=5) as r:x=json.loads(r.read().decode("utf-8"))
                    self._btc_append(x.get("time"),x.get("price"))
                except Exception as exc:self.emit("ERROR",detail=f"Coinbase REST: {exc!r}")
                self.stop_event.wait(1)
    def _on_ticker(self,obj):
        raw=obj.get("raw_msg") or {};ticker=obj.get("ticker") or obj.get("market_ticker") or raw.get("market_ticker")
        if not ticker:return
        ts=_ts(obj.get("time") or obj.get("timestamp"));bid,ask=_price(obj.get("yes_bid_dollars")),_price(obj.get("yes_ask_dollars"));close_time=_ts(obj.get("market_close"))
        if not pd.isna(close_time):self.close_times[ticker]=close_time
        result=str(obj.get("result") or raw.get("result") or "").upper()
        with self.lock:
            if np.isfinite(bid) and np.isfinite(ask) and 0<=bid<=ask<=1 and not pd.isna(ts):
                self.quotes[ticker].append((ts,bid,ask));pos=self.positions.get(ticker)
                if pos and not pos["settled"] and pos["open_qty"]>0 and ticker not in self.exit_orders and pd.isna(pos["trigger_time"]):
                    _,held_ask=self._held_quote(pos["direction"],bid,ask)
                    if held_ask<=EXIT_PRICE+1e-12:
                        pos["trigger_time"]=ts;pos["confirm_time"]=ts.floor("min")+pd.Timedelta(minutes=1);self._record(ticker)["exit_trigger"]=ts;self.emit("EXIT_TRIGGER",ticker,f"held ask={100*held_ask:.1f}c | confirm={pos['confirm_time']}")
            if result in {"YES","NO"}:self.results[ticker]=result;self._try_settle(ticker)
    def _on_book(self,obj):
        ticker=obj.get("ticker");ts=_ts(obj.get("time"))
        if not ticker or pd.isna(ts):return
        yes_bids,yes_asks={},{}
        for level in obj.get("yes_bids") or []:
            if isinstance(level,(list,tuple)) and len(level)>=2:
                p,q=_price(level[0]),_num(level[1])
                if np.isfinite(p) and np.isfinite(q) and q>0:yes_bids[round(p,3)]=q
        for level in obj.get("yes_asks") or []:
            if isinstance(level,(list,tuple)) and len(level)>=2:
                p,q=_price(level[0]),_num(level[1])
                if np.isfinite(p) and np.isfinite(q) and q>0:yes_asks[round(p,3)]=q
        no_bids={round(1.0-p,3):q for p,q in yes_asks.items() if 0<=1.0-p<=1.0}
        with self.lock:self.books[ticker].append((ts,yes_bids,no_bids))
    def _extract_trade(self,obj):
        raw=obj.get("raw_msg") or {};ticker=obj.get("ticker") or raw.get("market_ticker")
        if not ticker:return None
        p=_price(obj.get("yes_price") if obj.get("yes_price") is not None else raw.get("yes_price_dollars"));qty=_num(obj.get("qty") if obj.get("qty") is not None else raw.get("count_fp") if raw.get("count_fp") is not None else obj.get("volume"));side=str(obj.get("taker_book_side") or raw.get("taker_book_side") or "").lower();ts=_ts(obj.get("time"));trade_id=obj.get("trade_id") or raw.get("trade_id")
        if not np.isfinite(p) or not np.isfinite(qty) or qty<=0 or side not in {"bid","ask"} or pd.isna(ts):return None
        key=str(trade_id) if trade_id is not None else f"{ticker}|{ts}|{p:.4f}|{qty:.8f}|{side}";return {"ticker":ticker,"ts":ts,"yes_price":p,"qty":qty,"taker_book_side":side,"key":key}
    def _on_trade(self,obj):
        trade=self._extract_trade(obj)
        if trade is None:return
        ticker=trade["ticker"]
        with self.lock:
            self.trades[ticker].append(trade)
            if ticker in self.entry_orders:self._apply_entry_trade(self.entry_orders[ticker],trade)
            if ticker in self.exit_orders:self._apply_exit_trade(self.exit_orders[ticker],trade)
    def _entry_queue(self,direction,entry_price,book):
        if book is None:return np.nan
        _,yes_bids,no_bids=book;levels=yes_bids if direction=="YES" else no_bids;return float(levels.get(round(entry_price,2),0.0))
    def _exit_queue(self,direction,book):
        if book is None:return np.nan
        _,yes_bids,no_bids=book;return float(no_bids.get(0.90,0.0) if direction=="YES" else yes_bids.get(0.90,0.0))
    def _apply_entry_trade(self,order,trade):
        if order["status"]!="OPEN" or trade["key"] in order["seen"]:return
        order["seen"].add(trade["key"]);ts=trade["ts"]
        if ts<order["order_time"] or ts>order["cancel_time"]:return
        q,p,qty,side=order["price"],trade["yes_price"],trade["qty"],trade["taker_book_side"];exact=through=False
        if order["direction"]=="YES":
            if side!="ask":return
            through,exact=p<q-1e-9,abs(p-q)<1e-9
        else:
            yes_equiv=1.0-q
            if side!="bid":return
            through,exact=p>yes_equiv+1e-9,abs(p-yes_equiv)<1e-9
        fill_qty=order["remaining"] if through else 0.0
        if exact:
            queue=order["queue_ahead"]
            if not np.isfinite(queue):return
            if qty<=queue:order["queue_ahead"]=queue-qty;return
            order["queue_ahead"]=0.0;fill_qty=min(order["remaining"],qty-queue)
        if fill_qty<=0:return
        order["remaining"]-=fill_qty;order["filled_qty"]+=fill_qty;rec=self._record(order["ticker"]);rec["entry_fill_qty"]+=fill_qty
        if pd.isna(rec["entry_first_fill"]):rec["entry_first_fill"]=ts
        rec["entry_last_fill"]=ts;ticker=order["ticker"]
        if ticker not in self.positions:self.positions[ticker]={"ticker":ticker,"direction":order["direction"],"entry_price":order["price"],"qty":0.0,"open_qty":0.0,"realized_pnl":0.0,"trigger_time":pd.NaT,"confirm_time":pd.NaT,"settled":False}
        pos=self.positions[ticker];pos["qty"]+=fill_qty;pos["open_qty"]+=fill_qty;reason="TRADED_THROUGH" if through else "QUEUE_CONSUMED";self.emit("ENTRY_FILL",ticker,f"{fill_qty:.2f}ctrs @ {order['price']:.2f} | {reason} | total={order['filled_qty']:.2f}/{QTY:g}",fill_qty=fill_qty,entry_price=order["price"],reason=reason)
        if order["remaining"]<=1e-12:order["status"]="FILLED"
        self._save_state()
    def _apply_exit_trade(self,order,trade):
        if order["status"]!="OPEN" or trade["key"] in order["seen"]:return
        order["seen"].add(trade["key"])
        if trade["ts"]<order["order_time"]:return
        ticker=order["ticker"];pos=self.positions.get(ticker)
        if pos is None or pos["open_qty"]<=0:return
        p,qty,side=trade["yes_price"],trade["qty"],trade["taker_book_side"];exact=through=False
        if order["direction"]=="YES":
            if side!="bid":return
            through,exact=p>0.10+1e-9,abs(p-0.10)<1e-9
        else:
            if side!="ask":return
            through,exact=p<0.90-1e-9,abs(p-0.90)<1e-9
        fill_qty=order["remaining"] if through else 0.0
        if exact:
            queue=order["queue_ahead"]
            if not np.isfinite(queue):return
            if qty<=queue:order["queue_ahead"]=queue-qty;return
            order["queue_ahead"]=0.0;fill_qty=min(order["remaining"],qty-queue)
        if fill_qty<=0:return
        fill_qty=min(fill_qty,pos["open_qty"]);order["remaining"]-=fill_qty;order["filled_qty"]+=fill_qty;pos["open_qty"]-=fill_qty;pnl=fill_qty*(EXIT_PRICE-pos["entry_price"]);pos["realized_pnl"]+=pnl;rec=self._record(ticker);rec["exit_fill_qty"]+=fill_qty;rec["realized_pnl"]=pos["realized_pnl"];reason="TRADED_THROUGH" if through else "QUEUE_CONSUMED";self.emit("EXIT_FILL",ticker,f"{fill_qty:.2f}ctrs @ 0.10 | {reason} | pnl_increment=${pnl:+.4f}",fill_qty=fill_qty,pnl_increment=pnl,reason=reason)
        if order["remaining"]<=1e-12 or pos["open_qty"]<=1e-12:order["status"]="FILLED"
        self._save_state()
    def _try_settle(self,ticker):
        pos=self.positions.get(ticker);result=self.results.get(ticker)
        if pos is None or pos["settled"] or result not in {"YES","NO"}:return
        remaining=pos["open_qty"];settlement_value=1.0 if result==pos["direction"] else 0.0;pos["realized_pnl"]+=remaining*(settlement_value-pos["entry_price"]);pos["open_qty"]=0.0;pos["settled"]=True
        if ticker in self.exit_orders:self.exit_orders[ticker]["status"]="SETTLED"
        rec=self._record(ticker);rec["result"]=result;rec["realized_pnl"]=pos["realized_pnl"];rec["status"]="SETTLED";self.emit("SETTLED",ticker,f"result={result} | held={pos['direction']} | qty={pos['qty']:.2f} | final pnl=${pos['realized_pnl']:+.4f}",result=result,final_pnl=pos["realized_pnl"]);self._save_state()
    def _process_decision(self,ticker,decision_time):
        if ticker.split("-")[0] not in FROZEN_SERIES:return
        self.decisions_done.add(ticker);q=self._quote_at(ticker,decision_time)
        if q is None:self.emit("SKIP",ticker,"no causal quote at decision");return
        _,yes_bid,yes_ask=q;spread=yes_ask-yes_bid;midpoint=(yes_bid+yes_ask)/2.0
        if spread>MAX_SPREAD_C/100.0+1e-12:self.emit("SKIP",ticker,f"spread={100*spread:.1f}c >2c");return
        if abs(midpoint-.50)<1e-12:self.emit("SKIP",ticker,"midpoint exactly 50c");return
        direction="YES" if midpoint>.50 else "NO";sign=1 if direction=="YES" else -1;btc=self._btc_return_15m(decision_time)
        if btc is None:self.emit("SKIP",ticker,"missing causal Coinbase BTC history");return
        btc_ret,_,_=btc
        if btc_ret*sign>=0:self.emit("SKIP",ticker,f"BTC not opposition | signal={direction} | btc15={10000*btc_ret:+.2f}bp");return
        held_bid,held_ask=self._held_quote(direction,yes_bid,yes_ask);entry_price=round(held_bid-DEPTH_C/100.0,2)
        if entry_price<.01 or entry_price>.99 or entry_price>=held_ask:self.emit("SKIP",ticker,f"illegal/non-passive -3c quote | bid={held_bid:.2f} ask={held_ask:.2f} entry={entry_price:.2f}");return
        queue=self._entry_queue(direction,entry_price,self._book_at(ticker,decision_time));cancel_time=decision_time+pd.Timedelta(seconds=LIFETIME_SEC);rec=self._record(ticker);rec.update({"decision_time":decision_time,"direction":direction,"midpoint":midpoint,"spread_c":100*spread,"btc_return":btc_ret,"entry_price":entry_price,"entry_queue":queue,"status":"ENTRY_OPEN"});self.entry_orders[ticker]={"ticker":ticker,"direction":direction,"order_time":decision_time,"cancel_time":cancel_time,"price":entry_price,"qty":QTY,"remaining":QTY,"filled_qty":0.0,"queue_ahead":queue,"status":"OPEN","seen":set()}
        self.emit("SIGNAL",ticker,f"{direction} | mid={100*midpoint:.1f}c | spread={100*spread:.1f}c | BTC15={10000*btc_ret:+.2f}bp");self.emit("ENTRY_POST",ticker,f"{QTY:g}ctrs @ {100*entry_price:.0f}c | queue={queue:.2f} | cancel in {LIFETIME_SEC:.0f}s")
        for trade in list(self.trades.get(ticker,())):
            if trade["ts"]>=decision_time:self._apply_entry_trade(self.entry_orders[ticker],trade)
        self._save_state()
    def _check_exit_confirmation(self,ticker,pos,now):
        if pd.isna(pos["confirm_time"]) or now<pos["confirm_time"]:return
        confirm_time=pos["confirm_time"];q=self._quote_at(ticker,confirm_time)
        if q is None:return
        _,yes_bid,yes_ask=q;held_bid,held_ask=self._held_quote(pos["direction"],yes_bid,yes_ask)
        if held_ask>EXIT_PRICE+1e-12:pos["trigger_time"]=pd.NaT;pos["confirm_time"]=pd.NaT;self.emit("SKIP",ticker,f"10c candle rejected | close ask={100*held_ask:.1f}c");return
        if EXIT_PRICE<=held_bid+1e-12:pos["trigger_time"]=pd.NaT;pos["confirm_time"]=pd.NaT;self.emit("SKIP",ticker,f"10c exit would be marketable | held bid={100*held_bid:.1f}c");return
        queue=self._exit_queue(pos["direction"],self._book_at(ticker,confirm_time));self.exit_orders[ticker]={"ticker":ticker,"direction":pos["direction"],"order_time":confirm_time,"price":EXIT_PRICE,"qty":float(pos["open_qty"]),"remaining":float(pos["open_qty"]),"filled_qty":0.0,"queue_ahead":queue,"status":"OPEN","seen":set()};rec=self._record(ticker);rec["exit_post"]=confirm_time;rec["exit_queue"]=queue;rec["status"]="EXIT_OPEN";self.emit("EXIT_POST",ticker,f"{pos['open_qty']:.2f}ctrs @10c | queue={queue:.2f}")
        for trade in list(self.trades.get(ticker,())):
            if trade["ts"]>=confirm_time:self._apply_exit_trade(self.exit_orders[ticker],trade)
        self._save_state()
    def _scheduler(self):
        while not self.stop_event.is_set():
            try:
                now=pd.Timestamp.now(tz="UTC")
                with self.lock:
                    for ticker in list(self.quotes.keys()):
                        if ticker in self.decisions_done or ticker.split("-")[0] not in FROZEN_SERIES:continue
                        close_time=self.close_times.get(ticker)
                        if close_time is None or pd.isna(close_time):continue
                        decision_time=close_time-pd.Timedelta(minutes=10)
                        if decision_time<self.started_at:
                            if now>=decision_time:self.decisions_done.add(ticker)
                            continue
                        if now>=decision_time:
                            lateness=(now-decision_time).total_seconds()
                            if lateness>5:self.decisions_done.add(ticker);self.emit("SKIP",ticker,f"shadow scheduler late by {lateness:.2f}s");continue
                            self._process_decision(ticker,decision_time)
                    for ticker,order in list(self.entry_orders.items()):
                        if order["status"]=="OPEN" and now>=order["cancel_time"]:
                            order["status"]="CANCELLED";rec=self._record(ticker);rec["status"]="POSITION_OPEN" if order["filled_qty"]>0 else "NO_FILL";self.emit("ENTRY_CANCEL",ticker,f"filled={order['filled_qty']:.2f}/{QTY:g} | cancelled={order['remaining']:.2f}");self._save_state()
                    for ticker,pos in list(self.positions.items()):
                        if not pos["settled"] and pos["open_qty"]>0 and ticker not in self.exit_orders:self._check_exit_confirmation(ticker,pos,now)
                    for ticker in list(self.positions):self._try_settle(ticker)
            except Exception as exc:self.emit("ERROR",detail=f"scheduler: {exc!r}");traceback.print_exc()
            self.stop_event.wait(.10)
    def _follow_file(self,path,handler,label):
        while not path.exists() and not self.stop_event.is_set():self.stop_event.wait(.25)
        if self.stop_event.is_set():return
        with path.open("r",encoding="utf-8",errors="ignore") as f:
            f.seek(0,os.SEEK_END)
            while not self.stop_event.is_set():
                line=f.readline()
                if not line:self.stop_event.wait(.02);continue
                try:handler(json.loads(line))
                except Exception as exc:self.emit("ERROR",detail=f"{label}: {exc!r}")
    def _settlement_loop(self):
        while not self.stop_event.is_set():
            try:
                now=pd.Timestamp.now(tz="UTC")
                with self.lock:pending=[t for t,p in self.positions.items() if not p["settled"] and self.close_times.get(t) is not None and now>=self.close_times[t]]
                for ticker in pending:
                    try:
                        r=self._rest.get(f"{KALSHI_REST_BASE}/markets/{ticker}",timeout=8);r.raise_for_status();market=r.json().get("market") or r.json();result=str(market.get("result") or "").upper()
                        if result in {"YES","NO"}:
                            with self.lock:self.results[ticker]=result;self._try_settle(ticker)
                    except Exception as exc:self.emit("ERROR",ticker,f"settlement lookup: {exc!r}")
            except Exception as exc:self.emit("ERROR",detail=f"settlement loop: {exc!r}")
            self.stop_event.wait(10)
    def start(self):
        if any(t.is_alive() for t in self.threads):print("Shadow trader is already running.");return self
        print(f"Shadow session: {self.session_dir}\nShadow output: {self.out_dir}")
        for obj in _read_recent_jsonl(self.ticker_file,16*1024*1024):self._on_ticker(obj)
        for obj in _read_recent_jsonl(self.book_file,64*1024*1024):self._on_book(obj)
        for obj in _read_recent_jsonl(self.trades_file,8*1024*1024):
            trade=self._extract_trade(obj)
            if trade is not None:self.trades[trade["ticker"]].append(trade)
        self._seed_coinbase();specs=[(self._follow_file,(self.ticker_file,self._on_ticker,"ticker"),"shadow-ticker"),(self._follow_file,(self.book_file,self._on_book,"book"),"shadow-book"),(self._follow_file,(self.trades_file,self._on_trade,"trades"),"shadow-trades"),(self._scheduler,(),"shadow-scheduler"),(self._coinbase_loop,(),"shadow-coinbase"),(self._settlement_loop,(),"shadow-settlement")]
        for target,args,name in specs:
            th=threading.Thread(target=target,args=args,name=name,daemon=True);th.start();self.threads.append(th)
        print(f"Shadow trader running: -{DEPTH_C:.0f}c / {LIFETIME_SEC:.0f}s / {QTY:.0f} contracts | READ-ONLY");return self
    def stop(self):
        self.stop_event.set()
        for th in self.threads:th.join(timeout=3)
        self._save_state();print(f"Shadow trader stopped. Saved to: {self.out_dir}");return self.out_dir
    def status(self):
        with self.lock:df=pd.DataFrame(list(self.records.values()));thread_state={t.name:t.is_alive() for t in self.threads}
        if len(df):
            df=df.sort_values("decision_time").reset_index(drop=True);signals=len(df);filled=df["entry_fill_qty"]>0;full=df["entry_fill_qty"]>=QTY-1e-12;settled=df["status"]=="SETTLED";print(f"Signals: {signals} | any fills: {int(filled.sum())}/{signals} ({100*filled.mean():.2f}%) | full fills: {int(full.sum())} | settled: {int(settled.sum())} | realized PnL: ${df['realized_pnl'].sum():+.4f}");print(f"10c exit orders: {df['exit_post'].notna().sum()} | 10c exit fills: {(df['exit_fill_qty']>0).sum()}");cols=["ticker","series","decision_time","direction","midpoint","spread_c","btc_return","entry_price","entry_queue","entry_fill_qty","entry_first_fill","entry_last_fill","exit_trigger","exit_post","exit_queue","exit_fill_qty","result","realized_pnl","status"]
            try:
                from IPython.display import display;display(df[cols])
            except Exception:print(df[cols].to_string(index=False))
        else:print("No eligible shadow signals yet.")
        print("Threads:",thread_state);return df

_SHADOW=None
def start_shadow_trader(session_dir):
    global _SHADOW
    if _SHADOW is not None and any(t.is_alive() for t in _SHADOW.threads):print("Shadow trader is already running.");return _SHADOW
    _SHADOW=ShadowTrader(session_dir).start();return _SHADOW
def stop_shadow_trader():
    if _SHADOW is None:print("Shadow trader is not running.");return None
    return _SHADOW.stop()
def shadow_status():
    if _SHADOW is None:print("Shadow trader is not running.");return pd.DataFrame()
    return _SHADOW.status()
