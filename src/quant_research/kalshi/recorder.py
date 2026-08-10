from __future__ import annotations
import asyncio,json,traceback
from . import recorder_core as C

_TASK=None; _STOP=None; _SESSION=None; _STATE=None
async def _send(ws,lock,obj):
    async with lock:await ws.send(json.dumps(obj))
async def _subscribe(ws,lock,state,ch,tickers):
    if not tickers or ch in state["pending"]:return
    i=state["next_id"];state["next_id"]+=1;p={"channels":[ch],"market_tickers":sorted(tickers)}
    if ch=="orderbook_delta":p["use_yes_price"]=True
    state["pending"].add(ch);state["pending_ids"][i]=ch;await _send(ws,lock,{"id":i,"cmd":"subscribe","params":p})
async def _update(ws,lock,state,ch,action,tickers):
    if not tickers or state["sids"].get(ch) is None:return
    i=state["next_id"];state["next_id"]+=1;await _send(ws,lock,{"id":i,"cmd":"update_subscription","params":{"sid":state["sids"][ch],"market_tickers":sorted(tickers),"action":action}})
async def _snapshots(ws,lock,state,tickers):
    sid=state["sids"].get("orderbook_delta")
    if not tickers or sid is None:return
    i=state["next_id"];state["next_id"]+=1;await _send(ws,lock,{"id":i,"cmd":"update_subscription","params":{"sid":sid,"market_tickers":sorted(tickers),"action":"get_snapshot"}})
async def _supervisor(ws,lock,state,files):
    latest=[];last=0.0
    while not _STOP.is_set():
        import time
        nowm=time.monotonic()
        if not latest or nowm-last>=C.MARKET_RESCAN_SECONDS:
            try:latest=await C.scan_open_15m_markets();last=time.monotonic()
            except Exception as e:C.write_jsonl(files["connection_events"],{"time":C.iso_utc(),"type":"discovery_error","error":repr(e)});await asyncio.sleep(C.SUPERVISOR_INTERVAL_SECONDS);continue
        now=C.utc_now();meta={m["ticker"]:m for m in latest if m.get("close_time") is None or m["close_time"]>now};desired,current=set(meta),set(state["markets"]);add,delete=desired-current,current-desired;state["meta"].update(meta)
        if desired:
            for ch in ("orderbook_delta","trade","ticker"):
                if ch not in state["sids"] and ch not in state["pending"]:await _subscribe(ws,lock,state,ch,desired)
        if add:
            for ch in ("orderbook_delta","trade","ticker"):await _update(ws,lock,state,ch,"add_markets",add)
            await _snapshots(ws,lock,state,add)
        if delete:
            for ch in ("orderbook_delta","trade","ticker"):await _update(ws,lock,state,ch,"delete_markets",delete)
            for t in delete:state["books"].pop(t,None)
        if add or delete:
            state["markets"]=desired;C.write_jsonl(files["market_rotations"],{"time":C.iso_utc(now),"active_count":len(desired),"added":sorted(add),"removed":sorted(delete),"active":[{"ticker":x["ticker"],"series_ticker":x.get("series_ticker"),"series_title":x.get("series_title"),"close_time":C.iso_utc(x.get("close_time")) if x.get("close_time") else None} for x in meta.values()]});print(f"[{now:%H:%M:%S} UTC] 15m active={len(desired)} added={len(add)} removed={len(delete)}")
            for t in sorted(add):print(f" + {t} | {meta[t].get('series_title','')}")
            for t in sorted(delete):print(f" - {t}")
        await asyncio.sleep(C.SUPERVISOR_INTERVAL_SECONDS)
async def _book_sampler(state,files):
    while not _STOP.is_set():
        now=C.utc_now()
        for t in list(state["markets"]):
            b,m=state["books"].get(t),state["meta"].get(t)
            if b is None or m is None or (m.get("close_time") is not None and now>=m["close_time"]):continue
            C.write_jsonl(files["full_books"],C.full_book_row(t,b,m))
        await asyncio.sleep(C.FULL_BOOK_INTERVAL_SECONDS)
async def _regime_sampler(state,files):
    while not _STOP.is_set():
        now=C.utc_now()
        for t in sorted(state["markets"]):
            m=state["meta"].get(t)
            if not m:continue
            ct=m.get("close_time");r=C.seconds_until(ct,now) if ct else None;C.write_jsonl(files["regime_snapshots"],{"time":C.iso_utc(now),"ticker":t,"event_ticker":m.get("event_ticker"),"series_ticker":m.get("series_ticker"),"series_title":m.get("series_title"),"series_category":m.get("series_category"),"series_frequency":m.get("series_frequency"),"market_title":m.get("market_title"),"market_open":C.iso_utc(m.get("open_time")) if m.get("open_time") else None,"market_close":C.iso_utc(ct) if ct else None,"seconds_to_close":r,"minutes_to_close":r/60 if r is not None else None,"regime":C.regime_from_close(ct,now)})
        await asyncio.sleep(C.REGIME_INTERVAL_SECONDS)
async def _consumer(ws,lock,state,files):
    async for raw in ws:
        if _STOP.is_set():break
        try:d=json.loads(raw)
        except Exception:continue
        typ,msg,sid,seq=d.get("type"),d.get("msg",{}),d.get("sid"),d.get("seq")
        if typ=="subscribed":
            ch,rs=msg.get("channel"),msg.get("sid",sid)
            if ch and rs is not None:
                state["sids"][ch]=rs;state["pending"].discard(ch)
                for i,x in list(state["pending_ids"].items()):
                    if x==ch:state["pending_ids"].pop(i,None)
                C.write_jsonl(files["connection_events"],{"time":C.iso_utc(),"type":"subscribed","channel":ch,"sid":rs})
            continue
        if typ=="error":C.write_jsonl(files["connection_events"],{"time":C.iso_utc(),"type":"ws_error","payload":d});continue
        if typ in {"orderbook_snapshot","orderbook_delta"} and seq is not None:
            last=state.get("seq")
            if last is not None and typ=="orderbook_delta" and seq!=last+1:C.write_jsonl(files["connection_events"],{"time":C.iso_utc(),"type":"sequence_gap","last_seq":last,"new_seq":seq});state["books"].clear();await _snapshots(ws,lock,state,state["markets"]);state["seq"]=seq;continue
            state["seq"]=seq
        if typ=="orderbook_snapshot":C.apply_snapshot(state["books"],msg);continue
        t=msg.get("market_ticker");m=state["meta"].get(t,{}) if t else {};ct=m.get("close_time")
        if typ=="orderbook_delta":C.write_jsonl(files["book_deltas"],{"time":C.iso_utc(),"ticker":t,"event_ticker":m.get("event_ticker"),"series_ticker":m.get("series_ticker"),"market_close":C.iso_utc(ct) if ct else None,"sid":sid,"seq":seq,"side":msg.get("side"),"price_dollars":msg.get("price_dollars"),"delta_fp":msg.get("delta_fp"),"ts_ms":msg.get("ts_ms"),"raw_msg":msg});C.apply_delta(state["books"],msg);continue
        if typ=="trade":
            oside=str(msg.get("taker_outcome_side") or msg.get("taker_side") or "").lower();qty=msg.get("count_fp") or msg.get("count");price=msg.get("yes_price_dollars") or msg.get("price_dollars");C.write_jsonl(files["trades"],{"time":C.iso_utc(),"ticker":t,"event_ticker":m.get("event_ticker"),"series_ticker":m.get("series_ticker"),"series_title":m.get("series_title"),"series_category":m.get("series_category"),"market_close":C.iso_utc(ct) if ct else None,"seconds_to_close":C.seconds_until(ct) if ct else None,"regime":C.regime_from_close(ct),"yes_price":C.safe_float(price,None),"qty":C.safe_float(qty,None),"volume":C.safe_float(qty,None),"action":"BUY" if oside=="yes" else "SELL" if oside=="no" else "UNKNOWN","taker_side":oside,"taker_book_side":msg.get("taker_book_side"),"trade_id":msg.get("trade_id"),"ts_ms":msg.get("ts_ms"),"raw_msg":msg});continue
        if typ=="ticker":C.write_jsonl(files["ticker_updates"],{"time":C.iso_utc(),"ticker":t,"event_ticker":m.get("event_ticker"),"series_ticker":m.get("series_ticker"),"market_close":C.iso_utc(ct) if ct else None,"seconds_to_close":C.seconds_until(ct) if ct else None,"regime":C.regime_from_close(ct),"yes_bid_dollars":msg.get("yes_bid_dollars"),"yes_ask_dollars":msg.get("yes_ask_dollars"),"yes_bid_size_fp":msg.get("yes_bid_size_fp"),"yes_ask_size_fp":msg.get("yes_ask_size_fp"),"price_dollars":msg.get("price_dollars"),"volume_fp":msg.get("volume_fp"),"result":msg.get("result"),"ts_ms":msg.get("ts_ms"),"raw_msg":msg})
async def _connection(files,key_id,key):
    global _STATE
    ws=await C.open_ws(key_id,key);lock=asyncio.Lock();state={"books":{},"meta":{},"markets":set(),"sids":{},"pending":set(),"pending_ids":{},"next_id":1,"seq":None};_STATE=state;C.write_jsonl(files["connection_events"],{"time":C.iso_utc(),"type":"connected"});print("Kalshi WS connected.")
    tasks=[asyncio.create_task(_supervisor(ws,lock,state,files)),asyncio.create_task(_book_sampler(state,files)),asyncio.create_task(_regime_sampler(state,files)),asyncio.create_task(_consumer(ws,lock,state,files))]
    try:
        done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
        for x in done:
            if x.exception() is not None:raise x.exception()
    finally:
        for x in tasks:
            if not x.done():x.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        try:await ws.close()
        except Exception:pass
async def run_recorder(duration_minutes=None,key_id=None,private_key_path=None):
    global _SESSION
    kid,key=C.load_auth(key_id,private_key_path);d,files=C.make_session();_SESSION=d;started=C.utc_now()
    try:series=await C.discover_15m_series(True)
    except Exception:series=[]
    manifest={"version":"V3.5_DYNAMIC_15M_MODULE","started_at":C.iso_utc(started),"ended_at":None,"duration_minutes_requested":duration_minutes,"market_rescan_seconds":C.MARKET_RESCAN_SECONDS,"series_rescan_seconds":C.SERIES_RESCAN_SECONDS,"full_book_interval_seconds":C.FULL_BOOK_INTERVAL_SECONDS,"dynamic_rotation":True,"use_yes_price":True,"initial_discovered_series":[{"ticker":s.get("ticker"),"title":s.get("title"),"frequency":s.get("frequency"),"category":s.get("category")} for s in series]};mp=d/"session_manifest.json";mp.write_text(json.dumps(manifest,indent=2,default=str));print(f"Recorder session: {d}\n15m series discovered: {len(series)}")
    async def timer():
        if duration_minutes is None:return
        await asyncio.sleep(duration_minutes*60);_STOP.set()
    tm=asyncio.create_task(timer())
    try:
        while not _STOP.is_set():
            try:await _connection(files,kid,key)
            except asyncio.CancelledError:raise
            except Exception as e:C.write_jsonl(files["connection_events"],{"time":C.iso_utc(),"type":"connection_exception","error":repr(e),"traceback":traceback.format_exc()})
            if not _STOP.is_set():await asyncio.sleep(C.RECONNECT_DELAY_SECONDS)
    finally:
        tm.cancel();await asyncio.gather(tm,return_exceptions=True);ended=C.utc_now();manifest["ended_at"]=C.iso_utc(ended);manifest["actual_duration_minutes"]=(ended-started).total_seconds()/60;mp.write_text(json.dumps(manifest,indent=2,default=str));C.close_files(files);print(f"15-minute recorder stopped. Saved to: {d}")
    return d
async def start_recorder(duration_minutes=None,key_id=None,private_key_path=None):
    global _TASK,_STOP,_SESSION
    if _TASK is not None and not _TASK.done():print("Recorder is already running.");return _TASK
    _STOP=asyncio.Event();_SESSION=None;_TASK=asyncio.create_task(run_recorder(duration_minutes,key_id,private_key_path))
    for _ in range(100):
        await asyncio.sleep(.05)
        if _SESSION is not None:break
        if _TASK.done():await _TASK
    return _TASK
async def stop_recorder():
    if _TASK is None or _TASK.done():print(f"Recorder is not running. Last session: {_SESSION}");return _SESSION
    _STOP.set();await _TASK;print(f"Saved to: {_SESSION}");return _SESSION
def current_session_dir():return _SESSION
def recorder_status():
    s=_STATE or {};out={"running":_TASK is not None and not _TASK.done(),"session_dir":str(_SESSION) if _SESSION else None,"active_markets":len(s.get("markets",[])),"books_in_memory":len(s.get("books",{})),"channels":sorted(s.get("sids",{}))};print(out);return out
preview_15m_markets=C.preview_15m_markets
