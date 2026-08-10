from __future__ import annotations
import asyncio
from . import recorder_core as C

C.MARKET_RESCAN_SECONDS=5

def scan_open_15m_markets_sync():
    out=[]
    for s in C.discover_15m_series_sync():
        st=str(s.get("ticker",''))
        if not st:continue
        try:markets=C.rest_get("/markets",{"series_ticker":st,"status":"open","limit":1000}).get("markets") or []
        except Exception:continue
        for m in markets:
            status=str(m.get("status",'')).lower()
            if status not in {"open","active"} or str(m.get("market_type","binary")).lower()!="binary" or not m.get("ticker"):continue
            ot,ct=C.parse_time(m.get("open_time")),C.parse_time(m.get("close_time"))
            if ct is not None and ct<=C.utc_now():continue
            out.append({"ticker":m["ticker"],"event_ticker":m.get("event_ticker"),"series_ticker":st,"series_title":s.get("title",''),"series_frequency":s.get("frequency",''),"series_category":s.get("category",''),"series_tags":s.get("tags",[]),"market_title":m.get("title") or m.get("yes_sub_title") or "","open_time":ot,"close_time":ct,"volume":C.market_volume(m),"yes_bid_dollars":m.get("yes_bid_dollars"),"yes_ask_dollars":m.get("yes_ask_dollars"),"status":m.get("status")})
    out=list({x["ticker"]:x for x in out}.values());out.sort(key=lambda x:x["volume"],reverse=True);return out[:C.MAX_ACTIVE_MARKETS]

async def scan_open_15m_markets():return await asyncio.to_thread(scan_open_15m_markets_sync)

C.scan_open_15m_markets_sync=scan_open_15m_markets_sync
C.scan_open_15m_markets=scan_open_15m_markets
