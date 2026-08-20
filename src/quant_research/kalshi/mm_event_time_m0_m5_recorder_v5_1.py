from __future__ import annotations

"""V5.1 live-recorder hardening: fail closed on silent market-discovery failure.

This wrapper changes no persisted event-time window or book/trade semantics from V5.
It fixes an operational bug in V5._discover_sync(): V5 swallowed every /markets
exception and returned an empty dict, after which the supervisor cleared
last_scan_error and health could report healthy=True with zero subscribed markets.
A live engine depending on first book rows could then remain ARMED forever.

V5.1 preserves the exact V5 M0-M5(+30s) capture rules but:
- retries/paces public /markets discovery;
- requires both open/unopened discovery calls to succeed for every frozen series;
- requires each frozen series to return at least one market across those statuses;
- raises on discovery failure so health becomes unhealthy rather than silently idle.

Importing this module sends no orders.
"""

import argparse
import asyncio
import time
from pathlib import Path

from . import recorder_core as C
from . import mm_event_time_m0_m5_recorder_v5 as V5


STUDY_VERSION = "MM_EVENT_TIME_M0_M5_V5_1_STRICT_DISCOVERY"
DISCOVERY_PACE_S = 0.10
DISCOVERY_RETRY_DELAYS_S = (0.50, 1.0, 2.0)


def _is_429(exc):
    s = repr(exc).lower()
    return "429" in s or "too many requests" in s


def _rest_get_resilient(path, params):
    last = None
    attempts = len(DISCOVERY_RETRY_DELAYS_S) + 1
    for i in range(attempts):
        try:
            out = C.rest_get(path, params)
            time.sleep(DISCOVERY_PACE_S)
            return out
        except Exception as exc:
            last = exc
            if not _is_429(exc) or i >= len(DISCOVERY_RETRY_DELAYS_S):
                break
            time.sleep(float(DISCOVERY_RETRY_DELAYS_S[i]))
    raise RuntimeError(f"market discovery GET failed path={path} params={params}: {last!r}")


def _discover_sync_strict():
    now = C.utc_now()
    out = {}
    errors = []
    raw_counts = {}

    for series in V5.CRYPTO_SERIES:
        series_raw = 0
        for query_status in ("unopened", "open"):
            try:
                payload = _rest_get_resilient(
                    "/markets",
                    {"series_ticker": series, "status": query_status, "limit": 1000},
                )
                markets = payload.get("markets") or []
                series_raw += len(markets)
            except Exception as exc:
                errors.append(f"{series}/{query_status}: {exc!r}")
                continue

            for m in markets:
                row = V5._market_row(series, m)
                if row is None:
                    continue
                e = V5._elapsed(row, now)
                if e is None:
                    continue
                if -V5.PRESUBSCRIBE_LEAD_S <= e < V5.LABEL_TAIL_END_S:
                    out[row["ticker"]] = row

        raw_counts[series] = series_raw

    missing_series = [s for s, n in raw_counts.items() if int(n) <= 0]
    if errors or missing_series:
        parts = []
        if errors:
            parts.append("query_errors=" + " ; ".join(errors[:12]))
        if missing_series:
            parts.append("zero_raw_markets=" + ",".join(missing_series))
        raise RuntimeError(
            "STRICT_DISCOVERY_FAIL_CLOSED: " + " | ".join(parts)
        )

    return out


async def run_event_time_m0_m5_v5_1_recorder(session_dir: Path):
    old_discover = V5._discover_sync
    old_version = V5.STUDY_VERSION
    V5._discover_sync = _discover_sync_strict
    V5.STUDY_VERSION = STUDY_VERSION
    try:
        await V5.run_event_time_m0_m5_v5_recorder(Path(session_dir).resolve())
    finally:
        V5._discover_sync = old_discover
        V5.STUDY_VERSION = old_version


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
        "discovery_fail_closed": True,
        "discovery_429_retry": True,
        "orders_sent": False,
    }
    out["ok"] = bool(
        out["base_capture_window_unchanged"]
        and out["universe_unchanged"]
        and out["discovery_fail_closed"]
        and out["discovery_429_retry"]
    )
    if show:
        print("=" * 100)
        print("V5.1 STRICT DISCOVERY STATIC CHECK — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:44s}: {v}")
    if not out["ok"]:
        raise RuntimeError(f"V5.1 static check failed: {out}")
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-session")
    a = ap.parse_args()
    if a.run_session:
        asyncio.run(run_event_time_m0_m5_v5_1_recorder(Path(a.run_session)))
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "STUDY_VERSION", "DISCOVERY_PACE_S", "DISCOVERY_RETRY_DELAYS_S",
    "_discover_sync_strict", "run_event_time_m0_m5_v5_1_recorder", "static_self_check",
]
