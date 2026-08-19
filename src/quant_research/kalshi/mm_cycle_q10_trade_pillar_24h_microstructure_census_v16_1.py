from __future__ import annotations

"""Compatibility wrapper for V16 full-24h pillar census.

Fixes a pandas timezone dtype bug in V15._attach_start_bbo used by V16:
`bbo_receipt_time` was initialized as tz-naive datetime64[ns], then assigned
UTC-aware timestamps. This wrapper replaces only that helper with a UTC-aware
implementation and delegates the rest of V16 unchanged.

READ ONLY. NO API. NO ORDERS.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from . import mm_cycle_q10_trade_pillar_24h_census_v15 as V15
from . import mm_cycle_q10_trade_pillar_24h_microstructure_census_v16 as V16

VERSION = "MM_CYCLE_Q10_TRADE_PILLAR_24H_MICROSTRUCTURE_CENSUS_V16_1"
HARD_BOUND_SESSION = V16.HARD_BOUND_SESSION


def _attach_start_bbo_utc(source, pillars: pd.DataFrame, *, show=True):
    if pillars.empty:
        return pillars

    out = pillars.copy().reset_index(drop=True)

    # Critical fix: explicitly initialize as UTC-aware datetime dtype.
    out["bbo_receipt_time"] = pd.Series(
        pd.array([pd.NaT] * len(out), dtype="datetime64[ns, UTC]"),
        index=out.index,
    )
    out["bbo_age_ms"] = np.nan
    out["bid_at_start"] = np.nan
    out["ask_at_start"] = np.nan
    out["spread_at_start_c"] = np.nan

    targets = defaultdict(list)
    for idx, row in out.iterrows():
        targets[str(row["ticker"])].append((row["receipt_start"], int(idx)))
    for ticker in targets:
        targets[ticker].sort(key=lambda z: z[0])

    ptr = defaultdict(int)
    latest = {}
    scanned = 0

    def assign_until(ticker, cutoff):
        arr = targets.get(ticker)
        if not arr:
            return
        i = ptr[ticker]
        last = latest.get(ticker)
        while i < len(arr) and arr[i][0] < cutoff:
            target_ts, idx = arr[i]
            if last is not None:
                bt, bid, ask = last
                age_ms = (target_ts - bt).total_seconds() * 1000.0
                if -V15.EPS <= age_ms <= V15.BBO_MAX_AGE_S * 1000.0 + V15.EPS:
                    out.at[idx, "bbo_receipt_time"] = bt
                    out.at[idx, "bbo_age_ms"] = age_ms
                    out.at[idx, "bid_at_start"] = bid
                    out.at[idx, "ask_at_start"] = ask
                    out.at[idx, "spread_at_start_c"] = 100.0 * (ask - bid)
            i += 1
        ptr[ticker] = i

    for r in V15._iter_jsonl(source / "book_top3_events.jsonl"):
        scanned += 1
        ticker = str(r.get("ticker") or "")
        if ticker not in targets:
            continue
        rt = V15._ts(r.get("receipt_time"))
        bid = V15._f(r.get("yes_bid"))
        ask = V15._f(r.get("yes_ask"))
        if pd.isna(rt) or not (np.isfinite(bid) and np.isfinite(ask) and 0 <= bid < ask <= 1):
            continue
        assign_until(ticker, rt)
        latest[ticker] = (rt, float(bid), float(ask))

        if show and scanned % 1_000_000 == 0:
            done = sum(ptr.values())
            print(f"book scan: {scanned:,} rows | pillar BBOs assigned/advanced={done:,}/{len(out):,}")

    for ticker, arr in targets.items():
        i = ptr[ticker]
        last = latest.get(ticker)
        while i < len(arr):
            target_ts, idx = arr[i]
            if last is not None:
                bt, bid, ask = last
                age_ms = (target_ts - bt).total_seconds() * 1000.0
                if -V15.EPS <= age_ms <= V15.BBO_MAX_AGE_S * 1000.0 + V15.EPS:
                    out.at[idx, "bbo_receipt_time"] = bt
                    out.at[idx, "bbo_age_ms"] = age_ms
                    out.at[idx, "bid_at_start"] = bid
                    out.at[idx, "ask_at_start"] = ask
                    out.at[idx, "spread_at_start_c"] = 100.0 * (ask - bid)
            i += 1
        ptr[ticker] = i

    out["wide_spread_at_start"] = (
        pd.to_numeric(out["spread_at_start_c"], errors="coerce") > V15.WIDE_SPREAD_C
    )
    out["bbo_start_available"] = pd.to_numeric(
        out["spread_at_start_c"], errors="coerce"
    ).notna()
    return out


def run_trade_pillar_24h_microstructure_census(source_session, *, hard_bind=True, show=True):
    original = V15._attach_start_bbo
    V15._attach_start_bbo = _attach_start_bbo_utc
    try:
        result = V16.run_trade_pillar_24h_microstructure_census(
            source_session,
            hard_bind=hard_bind,
            show=show,
        )
        result["version"] = VERSION
        return result
    finally:
        V15._attach_start_bbo = original


__all__ = ["VERSION", "HARD_BOUND_SESSION", "run_trade_pillar_24h_microstructure_census"]
