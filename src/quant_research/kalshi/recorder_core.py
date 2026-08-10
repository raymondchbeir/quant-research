from __future__ import annotations
import base64, json, os, re, time
from datetime import datetime, timezone
from pathlib import Path
import requests, websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

REST_BASE="https://external-api.kalshi.com/trade-api/v2"; WS_URL="wss://external-api-ws.kalshi.com/trade-api/ws/v2"; WS_PATH="/trade-api/ws/v2"
MARKET_RESCAN_SECONDS=15; SERIES_RESCAN_SECONDS=1800; SUPERVISOR_INTERVAL_SECONDS=2; MAX_ACTIVE_MARKETS=100
FULL_BOOK_INTERVAL_SECONDS=1.0; REGIME_INTERVAL_SECONDS=15.0; RECONNECT_DELAY_SECONDS=3; HTTP_TIMEOUT_SECONDS=12
PROJECT_ROOT=Path(__file__).resolve().parents[3]; DATA_ROOT=PROJECT_ROOT/"data"/"kalshi_15m"; DEFAULT_PRIVATE_KEY_PATH=PROJECT_ROOT/"API Keys"/"research.txt"; DATA_ROOT.mkdir(parents=True,exist_ok=True)
_HTTP=requests.Session(); _SERIES_CACHE={"updated_monotonic":0.0,"series":[]}

def utc_now(): return datetime.now(timezone.utc)
def iso_utc(dt=None):
    dt=dt or utc_now(); dt=dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt; return dt.astimezone(timezone.utc).isoformat()
def parse_time(x):
    if not x:return None
    try:
        dt=datetime.fromisoformat(str(x).replace("Z","+00:00")); dt=dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt; return dt.astimezone(timezone.utc)
    except Exception:return None
def safe_float(x,default=0.0):
    try:return float(x)
    except Exception:return default
def seconds_until(dt,now=None): return None if dt is None else (dt-(now or utc_now())).total_seconds()
def regime_from_close(close_time,now=None):
    if close_time is None:return "LIVE_15M_UNKNOWN_CLOSE"
    r=seconds_until(close_time,now); return "LIVE_15M_UNKNOWN_CLOSE" if r is None else "CLOSED" if r<=0 else "LIVE_15M"

def load_auth(key_id=None,private_key_path=None):
    if not key_id:key_id=os.environ.get("KALSHI_KEY_ID") or os.environ.get("KALSHI_API_KEY_ID")
    if not key_id:
        for p in (PROJECT_ROOT/"API Keys"/"kalshi_key_id.txt",PROJECT_ROOT/"API Keys"/"key_id.txt",PROJECT_ROOT/"API Keys"/"research_key_id.txt"):
            if p.exists() and p.read_text().strip():key_id=p.read_text().strip();break
    if not key_id:raise RuntimeError(f"Kalshi key ID not found. Export KALSHI_KEY_ID or save it in {PROJECT_ROOT/'API Keys'/'kalshi_key_id.txt'}")
    p=Path(private_key_path) if private_key_path else DEFAULT_PRIVATE_KEY_PATH
    if not p.exists():raise FileNotFoundError(f"Kalshi private key not found: {p}")
    with p.open("rb") as f:key=serialization.load_pem_private_key(f.read(),password=None)
    return str(key_id).strip(),key

def ws_headers(key_id,private_key):
    ts=str(int(time.time()*1000)); msg=(ts+"GET"+WS_PATH).encode(); sig=private_key.sign(msg,padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=padding.PSS.DIGEST_LENGTH),hashes.SHA256())
    return {"KALSHI-ACCESS-KEY":key_id,"KALSHI-ACCESS-TIMESTAMP":ts,"KALSHI-ACCESS-SIGNATURE":base64.b64encode(sig).decode()}
async def open_ws(key_id,private_key):
    h=ws_headers(key_id,private_key); kw=dict(ping_interval=20,ping_timeout=20,close_timeout=10,max_size=None)
    try:return await websockets.connect(WS_URL,additional_headers=h,**kw)
    except TypeError:return await websockets.connect(WS_URL,extra_headers=h,**kw)
def rest_get(path,params=None):
    r=_HTTP.get(REST_BASE+path,params=params,timeout=HTTP_TIMEOUT_SECONDS); r.raise_for_status(); return r.json()

def looks_like_15m_series(s):
    text=" ".join([str(s.get("frequency","")),str(s.get("title",""))," ".join(str(x) for x in (s.get("tags") or [])),str(s.get("product_metadata",""))]).lower()
    return any(re.search(p,text) for p in (r"\b15\s*min\b",r"\b15\s*mins\b",r"\b15\s*minute\b",r"\b15\s*minutes\b",r"\b15m\b",r"\bquarter[\s\-]?hour\b"))
def discover_15m_series_sync(force=False):
    now=time.monotonic(); cached=_SERIES_CACHE["series"]
    if cached and not force and now-_SERIES_CACHE["updated_monotonic"]<SERIES_RESCAN_SECONDS:return cached
    fifteen=[s for s in (rest_get("/series",{"include_volume":"true"}).get("series") or []) if looks_like_15m_series(s)]; fifteen.sort(key=lambda s:safe_float(s.get("volume_fp"),0),reverse=True)
    _SERIES_CACHE.update(updated_monotonic=now,series=fifteen); return fifteen
async def discover_15m_series(force=False):
    import asyncio; return await asyncio.to_thread(discover_15m_series_sync,force)
def market_volume(m):
    for f in ("volume_fp","volume_24h_fp","volume","volume_24h"):
        try:
            if m.get(f) is not None:return float(m[f])
        except Exception:pass
    return 0.0
def scan_open_15m_markets_sync():
    out=[]
    for s in discover_15m_series_sync():
        st=str(s.get("ticker",''))
        if not st:continue
        try:markets=rest_get("/markets",{"series_ticker":st,"status":"open","limit":1000}).get("markets") or []
        except Exception:continue
        for m in markets:
            if str(m.get("status",'')).lower()!="open" or str(m.get("market_type","binary")).lower()!="binary" or not m.get("ticker"):continue
            ot,ct=parse_time(m.get("open_time")),parse_time(m.get("close_time"))
            if ct is not None and ct<=utc_now():continue
            out.append({"ticker":m["ticker"],"event_ticker":m.get("event_ticker"),"series_ticker":st,"series_title":s.get("title",''),"series_frequency":s.get("frequency",''),"series_category":s.get("category",''),"series_tags":s.get("tags",[]),"market_title":m.get("title") or m.get("yes_sub_title") or "","open_time":ot,"close_time":ct,"volume":market_volume(m),"yes_bid_dollars":m.get("yes_bid_dollars"),"yes_ask_dollars":m.get("yes_ask_dollars"),"status":m.get("status")})
    out=list({x["ticker"]:x for x in out}.values()); out.sort(key=lambda x:x["volume"],reverse=True); return out[:MAX_ACTIVE_MARKETS]
async def scan_open_15m_markets():
    import asyncio; return await asyncio.to_thread(scan_open_15m_markets_sync)
async def preview_15m_markets():
    s=await discover_15m_series(True); m=await scan_open_15m_markets(); print(f"15m series: {len(s)} | open 15m contracts: {len(m)}")
    for x in sorted(m,key=lambda z:(z.get("close_time") or utc_now(),z["ticker"])):print(f"{x['ticker']} | {x.get('series_title','')} | close={x.get('close_time')}")
    return m

def empty_book():return {"yes":{},"no":{}}
def apply_snapshot(books,msg):
    t=msg.get("market_ticker")
    if not t:return
    b=empty_book()
    for side,field in (("yes","yes_dollars_fp"),("no","no_dollars_fp")):
        for level in msg.get(field) or []:
            if isinstance(level,(list,tuple)) and len(level)>=2:
                p,q=safe_float(level[0]),safe_float(level[1])
                if q>0:b[side][p]=q
    books[t]=b
def apply_delta(books,msg):
    t,side=msg.get("market_ticker"),str(msg.get("side",'')).lower()
    if not t or side not in {"yes","no"} or t not in books:return
    p,d=safe_float(msg.get("price_dollars")),safe_float(msg.get("delta_fp")); new=books[t][side].get(p,0)+d
    if new<=0:books[t][side].pop(p,None)
    else:books[t][side][p]=new
def full_book_row(ticker,book,meta):
    now=utc_now(); ct,ot=meta.get("close_time"),meta.get("open_time"); rem=seconds_until(ct,now) if ct else None
    bids=sorted([[float(p),float(q)] for p,q in book["yes"].items() if q>0],reverse=True); asks=sorted([[float(p),float(q)] for p,q in book["no"].items() if q>0])
    return {"time":iso_utc(now),"ticker":ticker,"event_ticker":meta.get("event_ticker"),"series_ticker":meta.get("series_ticker"),"series_title":meta.get("series_title"),"series_category":meta.get("series_category"),"market_title":meta.get("market_title"),"market_open":iso_utc(ot) if ot else None,"market_close":iso_utc(ct) if ct else None,"seconds_to_close":rem,"minutes_to_close":rem/60 if rem is not None else None,"regime":regime_from_close(ct,now),"yes_bids":bids,"yes_asks":asks}
def write_jsonl(h,obj):h.write(json.dumps(obj,default=str,separators=(",",":"))+"\n")
def make_session():
    d=DATA_ROOT/utc_now().strftime("%Y%m%d_%H%M%S"); d.mkdir(parents=True,exist_ok=False); names={"full_books":"full_books.jsonl","book_deltas":"book_deltas.jsonl","trades":"trades.jsonl","ticker_updates":"ticker_updates.jsonl","regime_snapshots":"regime_snapshots.jsonl","connection_events":"connection_events.jsonl","market_rotations":"fifteen_min_market_rotations.jsonl"}; return d,{k:open(d/v,"a",buffering=1,encoding="utf-8") for k,v in names.items()}
def close_files(files):
    for f in files.values():
        try:f.flush();f.close()
        except Exception:pass
