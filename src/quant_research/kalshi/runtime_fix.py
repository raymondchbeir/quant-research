from __future__ import annotations

import asyncio
import concurrent.futures
import time

from . import recorder_core as C

# Faster rotation for 15-minute markets and a bounded REST wait.
C.MARKET_RESCAN_SECONDS = 5
C.HTTP_TIMEOUT_SECONDS = 6

# Keep last successful per-series result briefly so one transient REST failure
# does not make us delete a live subscription.
_SERIES_MARKET_CACHE = {}
_CACHE_TTL_SECONDS = 45.0
_MAX_SCAN_WORKERS = 8
_SCAN_WALL_TIMEOUT_SECONDS = 8.0


def _scan_one_series(series):
    st = str(series.get("ticker", ""))
    if not st:
        return st, [], False
    try:
        payload = C.rest_get(
            "/markets",
            {"series_ticker": st, "status": "open", "limit": 1000},
        )
        markets = payload.get("markets") or []
        return st, markets, True
    except Exception:
        return st, [], False


def scan_open_15m_markets_sync():
    series = list(C.discover_15m_series_sync())
    now_mono = time.monotonic()
    by_series = {}

    if series:
        workers = max(1, min(_MAX_SCAN_WORKERS, len(series)))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = {pool.submit(_scan_one_series, s): str(s.get("ticker", "")) for s in series}
        done, pending = concurrent.futures.wait(
            futures,
            timeout=_SCAN_WALL_TIMEOUT_SECONDS,
            return_when=concurrent.futures.ALL_COMPLETED,
        )

        for future in done:
            try:
                st, markets, ok = future.result()
            except Exception:
                st, markets, ok = futures.get(future, ""), [], False
            if not st:
                continue
            if ok:
                _SERIES_MARKET_CACHE[st] = (now_mono, markets)
                by_series[st] = markets
            else:
                cached = _SERIES_MARKET_CACHE.get(st)
                by_series[st] = cached[1] if cached and now_mono - cached[0] <= _CACHE_TTL_SECONDS else []

        for future in pending:
            st = futures.get(future, "")
            cached = _SERIES_MARKET_CACHE.get(st)
            by_series[st] = cached[1] if cached and now_mono - cached[0] <= _CACHE_TTL_SECONDS else []
            future.cancel()

        # Do not let slow individual series block the scanner wall clock.
        pool.shutdown(wait=False, cancel_futures=True)

    series_meta = {str(s.get("ticker", "")): s for s in series}
    out = []
    now = C.utc_now()

    for st, markets in by_series.items():
        s = series_meta.get(st, {})
        for m in markets:
            status = str(m.get("status", "")).lower()
            if (
                status not in {"open", "active"}
                or str(m.get("market_type", "binary")).lower() != "binary"
                or not m.get("ticker")
            ):
                continue

            ot = C.parse_time(m.get("open_time"))
            ct = C.parse_time(m.get("close_time"))
            if ct is not None and ct <= now:
                continue

            out.append(
                {
                    "ticker": m["ticker"],
                    "event_ticker": m.get("event_ticker"),
                    "series_ticker": st,
                    "series_title": s.get("title", ""),
                    "series_frequency": s.get("frequency", ""),
                    "series_category": s.get("category", ""),
                    "series_tags": s.get("tags", []),
                    "market_title": m.get("title") or m.get("yes_sub_title") or "",
                    "open_time": ot,
                    "close_time": ct,
                    "volume": C.market_volume(m),
                    "yes_bid_dollars": m.get("yes_bid_dollars"),
                    "yes_ask_dollars": m.get("yes_ask_dollars"),
                    "status": m.get("status"),
                }
            )

    out = list({x["ticker"]: x for x in out}.values())
    out.sort(key=lambda x: x["volume"], reverse=True)
    return out[: C.MAX_ACTIVE_MARKETS]


async def scan_open_15m_markets():
    return await asyncio.to_thread(scan_open_15m_markets_sync)


C.scan_open_15m_markets_sync = scan_open_15m_markets_sync
C.scan_open_15m_markets = scan_open_15m_markets
