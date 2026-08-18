from __future__ import annotations

"""V7.1 compatibility wrapper for the Q5 live-vs-shadow order-path forensic.

V7 called V6._load_all_strategy_fills() with a two-value unpack, but the V6
helper returns three values: (rows, time_sources, fee_keys).  This wrapper fixes
only that loader contract and delegates the complete forensic to V7 unchanged.

READ ONLY. NO API. NO ORDERS. Source session is not modified.
"""

from collections import defaultdict

from . import mm_cycle_q5_live_shadow_order_path_forensic_v7 as V7
from . import mm_cycle_q5_tp_fn_economics_v6 as V6

VERSION = "MM_CYCLE_Q5_LIVE_SHADOW_ORDER_PATH_FORENSIC_V7_1"
OUTPUT_ROOT = V7.OUTPUT_ROOT


def _live_fill_catalog_fixed(source):
    rows, _time_sources, _fee_keys = V6._load_all_strategy_fills(source)
    by_oid = defaultdict(list)
    by_ticker_entry_times = defaultdict(list)

    for r in rows:
        if r["role"] == "ENTRY":
            by_oid[str(r["order_id"])].append(r)
            by_ticker_entry_times[str(r["ticker"])].append(float(r["fill_s"]))

    out = {}
    for oid, xs in by_oid.items():
        xs.sort(key=lambda z: z["fill_s"])
        out[oid] = {
            "actual_first_fill_s": float(xs[0]["fill_s"]),
            "actual_last_fill_s": float(xs[-1]["fill_s"]),
            "actual_entry_fill_qty": float(sum(x["qty"] for x in xs)),
            "actual_entry_fill_rows": len(xs),
        }

    for ticker in by_ticker_entry_times:
        by_ticker_entry_times[ticker].sort()

    return out, dict(by_ticker_entry_times)


def run_q5_live_shadow_order_path_forensic(source_session, *, show=True):
    original = V7._live_fill_catalog
    V7._live_fill_catalog = _live_fill_catalog_fixed
    try:
        return V7.run_q5_live_shadow_order_path_forensic(source_session, show=show)
    finally:
        V7._live_fill_catalog = original


__all__ = ["run_q5_live_shadow_order_path_forensic", "VERSION", "OUTPUT_ROOT"]
