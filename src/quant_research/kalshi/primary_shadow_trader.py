from __future__ import annotations
import json, os, threading, traceback, urllib.request
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SERIES={"KXBNB15M","KXDOGE15M","KXETH15M","KXHYPE15M","KXNEAR15M","KXSOL15M","KXXRP15M","KXZEC15M"}
QTY=3.0; DEPTH=0.03; LIFE=15.0; MAX_SPREAD=0.02
CB_CANDLES="https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60"
CB_WS="wss://ws-feed.exchange.coinbase.com"
KALSHI="https://external-api.kalshi.com/trade-api/v2"

def _ts(x):return pd.to_datetime(x,utc=True,errors="coerce")
def _f(x,d=np.nan):
    try:
        y=float(x);return y if np.isfinite(y) else d
    except Exception:return d
def _p(x):
    x=_f(x)
    if np.isfinite(x) and x>1.5:x/=100
    return x
def _recent(path,n):
    if not path.exists() or path.stat().st_size==0:return []
    with path.open("rb") as f:
        size=f.seek(0,2);f.seek(max(0,size-n))
        if size>n:f.readline()
        raw=f.read().decode("utf-8",errors="ignore")
    out=[]
    for s in raw.splitlines():
        try:out.append(json.loads(s))
        except Exception:pass
    return out

class PrimaryShadow:
    def __init__(self,session_dir):
        self.session_dir=Path(session_dir);self.ticker=self.session_dir/"ticker_updates.jsonl";self.book=self.session_dir/"full_books.jsonl";self.trades=self.session_dir/"trades.jsonl"
        for p in (self.ticker,self.book,self.trades):
            if not p.exists():raise FileNotFoundError(p)
        self.out=self.session_dir/"SHADOW_PRIMARY_M5_MINUS3C_15S_3CT_HOLD_V1";self.out.mkdir(parents=True,exist_ok=True)
        self.events=self.out/"events.jsonl";self.csv=self.out/"records.csv";self.started=pd.Timestamp.now(tz="UTC")
        (self.out/"strategy_manifest.json").write_text(json.dumps({"strategy":"PRIMARY_M5_MINUS3C_15S_3CT_HOLD","started_at":self.started.isoformat(),"series":sorted(SERIES),"decision_elapsed_minute":5,"spread_max_c":2,"btc_signal":"strict causal trailing-15m opposition using 1m Coinbase closes","entry":"held bid - 3c, round to cents","qty":3,"lifetime_sec":15,"fill_model":"queue ahead + exact-price aggressive volume; trade-through fills; no cancellation credit","exit":"hold to settlement","read_only":True},indent=2))
        self.lock=threading.RLock();self.log_lock=threading.Lock();self.stop_event=threading.Event();self.threads=[];self.cbws=None
        self.quotes=defaultdict(lambda:deque(maxlen=10000));self.books=defaultdict(lambda:deque(maxlen=1200));self.tape=defaultdict(lambda:deque(maxlen=50000));self.btc=deque(maxlen=100000)
        self.closes={};self.done=set();self.orders={};self.records={};self.results={};self.rest=requests.Session()

    def emit(self,event,ticker=None,detail=None,**x):
        row={"time":pd.Timestamp.now(tz="UTC").isoformat(),"event":event,"ticker":ticker,"detail":detail,**x}
        with self.log_lock:
            with self.events.open("a") as f:f.write(json.dumps(row,default=str,separators=(",",":"))+"\n")
        if event in {"SIGNAL","POST","FILL","CANCEL","SETTLED","FINALIZED_NO_FILL","ERROR"}:
            print(f"{pd.Timestamp.now(tz='UTC').strftime('[%H:%M:%S UTC]')} {event}"+(f" | {ticker}" if ticker else "")+(f" | {detail}" if detail else ""))

    def rec(self,t):
        if t not in self.records:self.records[t]={"ticker":t,"series":t.split("-")[0],"decision_time":pd.NaT,"direction":None,"midpoint":np.nan,"spread_c":np.nan,"btc_return":np.nan,"entry_price":np.nan,"entry_queue":np.nan,"entry_fill_qty":0.0,"entry_first_fill":pd.NaT,"entry_last_fill":pd.NaT,"result":None,"correct":np.nan,"realized_pnl":0.0,"status":None}
        return self.records[t]

    def save(self):
        with self.lock:df=pd.DataFrame(self.records.values())
        if len(df):
            tmp=self.csv.with_suffix(".tmp");df.sort_values("decision_time",na_position="last").to_csv(tmp,index=False);tmp.replace(self.csv)

    @staticmethod
    def held(d,b,a):return (b,a) if d=="YES" else (1-a,1-b)

    def last(self,d,t,when):
        for x in reversed(d.get(t,())):
            if x[0]<=when:return x
        return None

    def btc_add(self,t,p):
        t,p=_ts(t),_f(p)
        if pd.isna(t) or not np.isfinite(p) or p<=0:return
        with self.lock:
            if self.btc and t<self.btc[-1][0]:return
            if self.btc and t==self.btc[-1][0]:self.btc[-1]=(t,p)
            else:self.btc.append((t,p))

    def btc_at(self,t):
        for x in reversed(self.btc):
            if x[0]<=t:return x
        return None

    def btc15(self,t):
        a,b=self.btc_at(t),self.btc_at(t-pd.Timedelta(minutes=15))
        return None if a is None or b is None else a[1]/b[1]-1

    def seed_btc(self):
        req=urllib.request.Request(CB_CANDLES,headers={"User-Agent":"kalshi-shadow-primary/1.0"})
        with urllib.request.urlopen(req,timeout=10) as r:rows=json.loads(r.read())
        now=pd.Timestamp.now(tz="UTC");seed=[]
        for x in rows:
            if len(x)>=5:
                t=pd.to_datetime(x[0],unit="s",utc=True)+pd.Timedelta(minutes=1)
                if t<=now:seed.append((t,float(x[4])))
        for t,p in sorted(seed):self.btc_add(t,p)
        print(f"Coinbase BTC history seeded: {len(seed)} causal 1m closes")

    def cb_loop(self):
        import websocket
        s={"bucket":None,"last":None}
        def op(ws):ws.send(json.dumps({"type":"subscribe","product_ids":["BTC-USD"],"channels":["ticker"]}))
        def msg(ws,m):
            try:
                x=json.loads(m)
                if x.get("type")!="ticker" or x.get("product_id")!="BTC-USD":return
                t,p=_ts(x.get("time")),_f(x.get("price"))
                if pd.isna(t) or not np.isfinite(p):return
                b=t.floor("min")
                if s["bucket"] is None:s.update(bucket=b,last=p);return
                if b>s["bucket"]:self.btc_add(s["bucket"]+pd.Timedelta(minutes=1),s["last"]);s["bucket"]=b
                s["last"]=p
            except Exception as e:self.emit("ERROR",detail=f"Coinbase: {e!r}")
        while not self.stop_event.is_set():
            try:self.cbws=websocket.WebSocketApp(CB_WS,on_open=op,on_message=msg);self.cbws.run_forever(ping_interval=20,ping_timeout=10)
            except Exception as e:self.emit("ERROR",detail=f"Coinbase WS: {e!r}")
            finally:self.cbws=None
            if not self.stop_event.is_set():self.stop_event.wait(2)

    def on_ticker(self,o):
        raw=o.get("raw_msg") or {};t=o.get("ticker") or raw.get("market_ticker")
        if not t:return
        ts=_ts(o.get("time"));b,a=_p(o.get("yes_bid_dollars")),_p(o.get("yes_ask_dollars"));c=_ts(o.get("market_close"))
        if not pd.isna(c):self.closes[t]=c
        r=str(o.get("result") or raw.get("result") or "").upper()
        with self.lock:
            if np.isfinite(b) and np.isfinite(a) and 0<=b<=a<=1 and not pd.isna(ts):self.quotes[t].append((ts,b,a))
            if r in {"YES","NO"}:self.results[t]=r

    def on_book(self,o):
        t=o.get("ticker");ts=_ts(o.get("time"))
        if not t or pd.isna(ts):return
        y,n={},{}
        for z in o.get("yes_bids") or []:
            if isinstance(z,(list,tuple)) and len(z)>=2:
                p,q=_p(z[0]),_f(z[1])
                if np.isfinite(p) and np.isfinite(q) and q>0:y[round(p,4)]=q
        for z in o.get("yes_asks") or []:
            if isinstance(z,(list,tuple)) and len(z)>=2:
                p,q=_p(z[0]),_f(z[1])
                if np.isfinite(p) and np.isfinite(q) and q>0:n[round(p,4)]=q
        with self.lock:self.books[t].append((ts,y,n))

    def trade(self,o):
        raw=o.get("raw_msg") or {};t=o.get("ticker") or raw.get("market_ticker")
        if not t:return None
        p=_p(o.get("yes_price") if o.get("yes_price") is not None else raw.get("yes_price_dollars"));q=_f(o.get("qty") if o.get("qty") is not None else raw.get("count_fp"));side=str(o.get("taker_book_side") or raw.get("taker_book_side") or "").lower();ts=_ts(o.get("time"));i=o.get("trade_id") or raw.get("trade_id")
        if not np.isfinite(p) or not np.isfinite(q) or q<=0 or side not in {"bid","ask"} or pd.isna(ts):return None
        return {"ticker":t,"ts":ts,"yes_price":p,"qty":q,"side":side,"key":str(i) if i is not None else f"{t}|{ts}|{p:.4f}|{q:.8f}|{side}"}

    def on_trade(self,o):
        x=self.trade(o)
        if x is None:return
        with self.lock:
            self.tape[x["ticker"]].append(x)
            if x["ticker"] in self.orders:self.apply(self.orders[x["ticker"]],x)

    def queue(self,d,p,b):
        if b is None:return np.nan
        _,y,n=b;return float((y if d=="YES" else n).get(round(p,4),0.0))

    def apply(self,o,x):
        if o["status"]!="OPEN" or x["key"] in o["seen"]:return
        o["seen"].add(x["key"])
        if x["ts"]<o["time"] or x["ts"]>o["cancel"]:return
        q,p,qty,side=o["price"],x["yes_price"],x["qty"],x["side"];exact=through=False
        if o["direction"]=="YES":
            if side!="ask":return
            exact=abs(p-q)<=1e-9;through=p<q-1e-9
        else:
            yp=1-q
            if side!="bid":return
            exact=abs(p-yp)<=1e-9;through=p>yp+1e-9
        fill=o["remaining"] if through else 0.0
        if exact:
            qa=o["queue"]
            if not np.isfinite(qa):return
            if qty<=qa+1e-12:o["queue"]=max(0,qa-qty);return
            o["queue"]=0;fill=min(o["remaining"],qty-qa)
        if fill<=1e-12:return
        o["remaining"]-=fill;o["filled"]+=fill;r=self.rec(o["ticker"]);r["entry_fill_qty"]+=fill
        if pd.isna(r["entry_first_fill"]):r["entry_first_fill"]=x["ts"]
        r["entry_last_fill"]=x["ts"];reason="TRADED_THROUGH" if through else "QUEUE_CONSUMED";self.emit("FILL",o["ticker"],f"{fill:.2f}ctrs @ {100*q:.1f}c | {reason} | total={o['filled']:.2f}/3")
        if o["remaining"]<=1e-12:o["status"]="FILLED";r["status"]="POSITION_OPEN"
        self.save()

    def decide(self,t,dt,final=False):
        q=self.last(self.quotes,t,dt);b=self.last(self.books,t,dt);br=self.btc15(dt)
        if q is None or b is None or br is None:
            if final:self.done.add(t);self.emit("SKIP",t,"missing causal quote/book/BTC")
            return final
        self.done.add(t);_,yb,ya=q;sp=ya-yb;mid=(yb+ya)/2
        if sp>MAX_SPREAD+1e-12:self.emit("SKIP",t,f"spread={100*sp:.2f}c >2c");return True
        if abs(mid-.5)<1e-12:self.emit("SKIP",t,"midpoint=50c");return True
        d="YES" if mid>.5 else "NO";sgn=1 if d=="YES" else -1
        if br*sgn>=0:self.emit("SKIP",t,f"BTC not opposition | {d} | BTC15={10000*br:+.2f}bp");return True
        hb,ha=self.held(d,yb,ya);ep=round(hb-DEPTH,2)
        if ep<.01 or ep>.99 or ep>=ha-1e-12:self.emit("SKIP",t,f"non-passive -3c quote | bid={hb:.4f} ask={ha:.4f} quote={ep:.2f}");return True
        qa=self.queue(d,ep,b);r=self.rec(t);r.update(decision_time=dt,direction=d,midpoint=mid,spread_c=100*sp,btc_return=br,entry_price=ep,entry_queue=qa,status="ENTRY_OPEN")
        self.orders[t]={"ticker":t,"direction":d,"time":dt,"cancel":dt+pd.Timedelta(seconds=LIFE),"price":ep,"queue":qa,"remaining":QTY,"filled":0.0,"status":"OPEN","seen":set()}
        self.emit("SIGNAL",t,f"{d} | mid={100*mid:.2f}c | spread={100*sp:.2f}c | BTC15={10000*br:+.2f}bp");self.emit("POST",t,f"3ctrs @ {100*ep:.1f}c | queue={qa:.2f} | cancel in 15s")
        for x in list(self.tape.get(t,())):
            if x["ts"]>=dt:self.apply(self.orders[t],x)
        self.save();return True

    def settle(self,t,result):
        r=self.records.get(t)
        if r is None or r.get("status") in {"SETTLED","FINALIZED_NO_FILL"}:return
        result=str(result).upper()
        if result not in {"YES","NO"}:return
        r["result"]=result;r["correct"]=float(result==r["direction"])
        if r["entry_fill_qty"]>0:
            sv=1.0 if result==r["direction"] else 0.0;r["realized_pnl"]=r["entry_fill_qty"]*(sv-r["entry_price"]);r["status"]="SETTLED";self.emit("SETTLED",t,f"result={result} | held={r['direction']} | qty={r['entry_fill_qty']:.2f} | pnl=${r['realized_pnl']:+.4f}")
        else:r["status"]="FINALIZED_NO_FILL";self.emit("FINALIZED_NO_FILL",t,f"result={result} | signal={r['direction']}")
        self.save()

    def scheduler(self):
        while not self.stop_event.is_set():
            try:
                now=pd.Timestamp.now(tz="UTC")
                with self.lock:
                    for t in list(self.quotes):
                        if t in self.done or t.split("-")[0] not in SERIES:continue
                        c=self.closes.get(t)
                        if c is None or pd.isna(c):continue
                        dt=c-pd.Timedelta(minutes=10)
                        if dt<self.started:
                            if now>=dt:self.done.add(t)
                            continue
                        if now>=dt:self.decide(t,dt,(now-dt).total_seconds()>=5)
                    for t,o in list(self.orders.items()):
                        if o["status"]=="OPEN" and now>=o["cancel"]:
                            o["status"]="CANCELLED";r=self.rec(t);r["status"]="POSITION_OPEN" if o["filled"]>0 else "NO_FILL";self.emit("CANCEL",t,f"filled={o['filled']:.2f}/3 | cancelled={o['remaining']:.2f}");self.save()
            except Exception as e:self.emit("ERROR",detail=f"scheduler: {e!r}");traceback.print_exc()
            self.stop_event.wait(.05)

    def follow(self,path,handler,label):
        while not path.exists() and not self.stop_event.is_set():self.stop_event.wait(.25)
        if self.stop_event.is_set():return
        with path.open("r",errors="ignore") as f:
            f.seek(0,2)
            while not self.stop_event.is_set():
                s=f.readline()
                if not s:self.stop_event.wait(.02);continue
                try:handler(json.loads(s))
                except Exception as e:self.emit("ERROR",detail=f"{label}: {e!r}")

    def settlement_loop(self):
        while not self.stop_event.is_set():
            try:
                now=pd.Timestamp.now(tz="UTC")
                with self.lock:pending=[t for t,r in self.records.items() if r.get("status") not in {"SETTLED","FINALIZED_NO_FILL"} and t in self.closes and now>=self.closes[t]]
                for t in pending:
                    res=self.results.get(t)
                    if res not in {"YES","NO"}:
                        try:
                            x=self.rest.get(f"{KALSHI}/markets/{t}",timeout=8);x.raise_for_status();res=str((x.json().get("market") or x.json()).get("result") or "").upper()
                            if res in {"YES","NO"}:self.results[t]=res
                        except Exception as e:self.emit("ERROR",t,f"settlement: {e!r}")
                    if res in {"YES","NO"}:
                        with self.lock:self.settle(t,res)
            except Exception as e:self.emit("ERROR",detail=f"settlement loop: {e!r}")
            self.stop_event.wait(10)

    def start(self):
        print(f"Primary shadow session: {self.session_dir}\nOutput: {self.out}")
        for x in _recent(self.ticker,16*1024*1024):self.on_ticker(x)
        for x in _recent(self.book,64*1024*1024):self.on_book(x)
        self.seed_btc()
        specs=[(self.follow,(self.ticker,self.on_ticker,"ticker"),"shadow-ticker"),(self.follow,(self.book,self.on_book,"book"),"shadow-book"),(self.follow,(self.trades,self.on_trade,"trades"),"shadow-trades"),(self.scheduler,(),"shadow-scheduler"),(self.cb_loop,(),"shadow-coinbase"),(self.settlement_loop,(),"shadow-settlement")]
        for fn,args,name in specs:
            th=threading.Thread(target=fn,args=args,name=name,daemon=True);th.start();self.threads.append(th)
        print("PRIMARY SHADOW: M5 | BTC opposition | spread<=2c | -3c | 15s | 3ct | HOLD TO SETTLEMENT | READ-ONLY");return self

    def stop(self):
        self.stop_event.set()
        try:
            if self.cbws:self.cbws.close()
        except Exception:pass
        for th in self.threads:th.join(timeout=3)
        self.save();print(f"Primary shadow stopped. Saved to: {self.out}");return self.out

    def status(self):
        with self.lock:df=pd.DataFrame(self.records.values());threads={t.name:t.is_alive() for t in self.threads}
        if len(df):
            df=df.sort_values("decision_time").reset_index(drop=True);f=df.entry_fill_qty>0;s=df.status=="SETTLED";full=df.entry_fill_qty>=QTY-1e-12;acc=df.loc[s,"correct"].mean() if s.any() else np.nan
            z=f"Signals: {len(df)} | fills: {int(f.sum())}/{len(df)} ({100*f.mean():.2f}%) | full 3ct: {int(full.sum())} | settled fills: {int(s.sum())} | PnL: ${df.realized_pnl.sum():+.4f}"
            if np.isfinite(acc):z+=f" | filled accuracy: {100*acc:.2f}%"
            print(z);print("Policy: M5 / BTC opposition / <=2c / -3c / 15s / 3ct / hold")
            cols=["ticker","series","decision_time","direction","midpoint","spread_c","btc_return","entry_price","entry_queue","entry_fill_qty","entry_first_fill","entry_last_fill","result","correct","realized_pnl","status"]
            try:
                from IPython.display import display;display(df[cols])
            except Exception:print(df[cols].to_string(index=False))
        else:print("No eligible primary shadow signals yet.")
        print("Threads:",threads);return df

_SHADOW=None
def start_shadow_trader(session_dir):
    global _SHADOW
    session_dir=Path(session_dir)
    if _SHADOW is not None and any(t.is_alive() for t in _SHADOW.threads):
        if _SHADOW.session_dir.resolve()==session_dir.resolve():print("Primary shadow already running on this session.");return _SHADOW
        _SHADOW.stop()
    _SHADOW=PrimaryShadow(session_dir).start();return _SHADOW
def stop_shadow_trader():
    global _SHADOW
    if _SHADOW is None:print("Primary shadow is not running.");return None
    out=_SHADOW.stop();_SHADOW=None;return out
def shadow_status():
    if _SHADOW is None:print("Primary shadow is not running.");return pd.DataFrame()
    return _SHADOW.status()
