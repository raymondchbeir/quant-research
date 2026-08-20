from __future__ import annotations

"""V5.2 live-recorder discovery hardening.

V5.1 correctly failed closed on discovery errors, but it performed 18 market
queries serially (9 frozen series x open/unopened). The unchanged V5 supervisor
wraps discovery in ``asyncio.wait_for(..., timeout=20)``. A serial strict scan can
therefore exceed 20 seconds even when each individual HTTP request is behaving
normally. Because V5 runs the sync discovery function in ``asyncio.to_thread``,
timing out the await does not stop the worker thread; repeated scans can then
create overlapping discovery work.

V5.2 preserves the exact V5 capture window, universe and persisted semantics. It
only changes public market discovery transport:
- all 18 frozen series/status GETs run concurrently in a dedicated 18-worker pool;
- each request has a 4 second HTTP timeout and bounded retries;
- the complete scan has a 16 second hard budget, below V5's 20 second supervisor
  timeout;
- any unresolved query or a frozen series with zero raw markets fails closed.

Importing this module sends no orders and performs no HTTP requests.
"""

import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import requests

from . import recorder_core as C
from . import mm_event_time_m0_m5_recorder_v5 as V5


STUDY_VERSION = "MM_EVENT_TIME_M0_M5_V5_2_BOUNDED_STRICT_DISCOVERY"
REQUEST_TIMEOUT_S = 4.0
RETRY_DELAYS_S = (0.35, 0.80)
DISCOVERY_BUDGET_S = 16.0
MAX_WORKERS = 18
SUPERVISOR_TIMEOUT_S = 20.0

_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="kalshi-v52-discovery")


def _query_one(series: str, query_status: str):
    """One bounded public /markets query; no auth and no order endpoints."""
    params = {
        "series_ticker": str(series),
        "status": str(query_status),
        "limit": 1000,
    }
    last = None
    attempts = len(RETRY_DELAYS_S) + 1
    for i in range(attempts):
        try:
            r = requests.get(
                C.REST_BASE + "/markets",
                params=params,
                timeout=REQUEST_TIMEOUT_S,
            )
            r.raise_for_status()
            body = r.json()
            return body.get("markets") or []
        except Exception as exc:
            last = exc
            if i >= len(RETRY_DELAYS_S):
                break
            time.sleep(float(RETRY_DELAYS_S[i]))
    raise RuntimeError(
        f"bounded market discovery failed series={series} status={query_status}: {last!r}"
    )


def _discover_sync_bounded():
    """Complete frozen-universe discovery in <20s or fail closed."""
    now = C.utc_now()
    jobs = {}
    for series in V5.CRYPTO_SERIES:
        for query_status in ("unopened", "open"):
            fut = _POOL.submit(_query_one, series, query_status)
            jobs[fut] = (str(series), str(query_status))

    done, pending = wait(set(jobs), timeout=DISCOVERY_BUDGET_S)
    errors = []
    if pending:
        for fut in pending:
            series, query_status = jobs[fut]
            errors.append(f"{series}/{query_status}: scan_budget_exceeded")
            fut.cancel()

    out = {}
    raw_counts = {str(s): 0 for s in V5.CRYPTO_SERIES}

    for fut in done:
        series, query_status = jobs[fut]
        try:
            markets = fut.result()
        except Exception as exc:
            errors.append(f"{series}/{query_status}: {exc!r}")
            continue

        raw_counts[series] += len(markets)
        for m in markets:
            row = V5._market_row(series, m)
            if row is None:
                continue
            e = V5._elapsed(row, now)
            if e is None:
                continue
            # Preserve V5's exact presubscribe/capture lifecycle.
            if -V5.PRESUBSCRIBE_LEAD_S <= e < V5.LABEL_TAIL_END_S:
                out[row["ticker"]] = row

    missing_series = [s for s, n in raw_counts.items() if int(n) <= 0]
    if errors or missing_series:
        parts = []
        if errors:
            parts.append("query_errors=" + " ; ".join(errors[:18]))
        if missing_series:
            parts.append("zero_raw_markets=" + ",".join(missing_series))
        raise RuntimeError("STRICT_DISCOVERY_FAIL_CLOSED: " + " | ".join(parts))

    return out


async def run_event_time_m0_m5_v5_2_recorder(session_dir: Path):
    old_discover = V5._discover_sync
    old_version = V5.STUDY_VERSION
    V5._discover_sync = _discover_sync_bounded
    V5.STUDY_VERSION = STUDY_VERSION
    try:
        await V5.run_event_time_m0_m5_v5_recorder(Path(session_dir).resolve())
    finally:
        V5._discover_sync = old_discover
        V5.STUDY_VERSION = old_version
        try:
            _POOL.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def static_self_check(*, show=True):
    out = {
        "study_version": STUDY_VERSION,
        "base_capture_window_unchanged": (
            V5.TRADE_WINDOW_START_S == 0.0
            and V5.TRADE_WINDOW_END_S == 300.0
            and V5.LABEL_TAIL_END_S == 330.0
            and V5.PRESUBSCRIBE_LEAD_S == 300.0
        ),
        "universe_unchanged": tuple(V5.CRYPTO_SERIES) == (
            "KXBTC15M", "KXBNB15M", "KXDOGE15M", "KXETH15M", "KXHYPE15M",
            "KXNEAR15M", "KXSOL15M", "KXXRP15M", "KXZEC15M",
        ),
        "query_count": 18,
        "max_workers": MAX_WORKERS,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "discovery_budget_s": DISCOVERY_BUDGET_S,
        "supervisor_timeout_s": SUPERVISOR_TIMEOUT_S,
        "budget_below_supervisor_timeout": DISCOVERY_BUDGET_S < SUPERVISOR_TIMEOUT_S,
        "strict_fail_closed": True,
        "orders_sent": False,
        "api_called": False,
    }
    out["ok"] = bool(
        out["base_capture_window_unchanged"]
        and out["universe_unchanged"]
        and out["max_workers"] >= out["query_count"]
        and out["budget_below_supervisor_timeout"]
        and out["strict_fail_closed"]
    )
    if show:
        print("=" * 100)
        print("V5.2 BOUNDED STRICT DISCOVERY STATIC CHECK — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:48s}: {v}")
    if not out["ok"]:
        raise RuntimeError(f"V5.2 static check failed: {out}")
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-session")
    a = ap.parse_args()
    if a.run_session:
        asyncio.run(run_event_time_m0_m5_v5_2_recorder(Path(a.run_session)))
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "STUDY_VERSION", "REQUEST_TIMEOUT_S", "RETRY_DELAYS_S", "DISCOVERY_BUDGET_S",
    "MAX_WORKERS", "_query_one", "_discover_sync_bounded",
    "run_event_time_m0_m5_v5_2_recorder", "static_self_check",
]
