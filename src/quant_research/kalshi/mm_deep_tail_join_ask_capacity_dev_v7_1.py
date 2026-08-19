from __future__ import annotations

"""V7.1 wrapper: same V7 capacity replay, corrected predeclared capacity gate.

V7 reported both conditional M5-residual coverage and overall terminal exit coverage.
For a strategy that exits much of its inventory passively before M5, gating on
``M5_exit_qty / M5_required_qty`` alone can be misleading: one unexecutable contract
out of 40 M5-required contracts looks like 97.5% even though 164/165 total entered
contracts were executable across passive + M5 exits (99.39%).

V7.1 therefore changes NO strategy/execution result.  It only defines the capacity gate
on the economically relevant overall terminal coverage:

    (passive_exit_qty + M5_exit_qty) / entry_filled_qty >= 99%

plus positive net PnL and no missing decision snapshot for a fully-filled entry.

Development only. No API. No orders. No validation data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from . import mm_deep_tail_join_ask_capacity_dev_v7 as V7
from . import mm_cycle_q10_oos_stack_v1 as OOS

VERSION = "MM_DEEP_TAIL_JOIN_ASK_CAPACITY_DEV_V7_1_TERMINAL_GATE"
EPS = V7.EPS


def run_join_ask_capacity_dev(source_session, *, hard_bind=True, show=True):
    # Run the exact V7 economics first.  V7 already uses only compact cached artifacts.
    res = V7.run_join_ask_capacity_dev(
        source_session,
        hard_bind=hard_bind,
        show=False,
    )

    surface = res["capacity_curve"].copy()
    detail = res["detail"].copy()
    by_asset = res["by_asset"].copy()

    surface["capacity_gate_positive_99pct_terminal_no_missing_decision"] = (
        (pd.to_numeric(surface["join_ask_net_pnl_rounding_bound"], errors="coerce") > 0.0)
        & (pd.to_numeric(surface["terminal_exit_fraction_of_entry_qty"], errors="coerce") >= 0.99 - EPS)
        & (pd.to_numeric(surface["decision_snapshot_missing_full_entries"], errors="coerce").fillna(0) == 0)
    )

    feasible = surface[
        surface["capacity_gate_positive_99pct_terminal_no_missing_decision"]
    ].copy()
    largest_feasible = (
        feasible.sort_values("requested_qty", ascending=False).iloc[0].to_dict()
        if len(feasible) else {}
    )
    max_pnl = (
        surface.sort_values(
            ["join_ask_net_pnl_rounding_bound", "requested_qty"],
            ascending=False,
        ).iloc[0].to_dict()
        if len(surface) else {}
    )

    out = Path(res["output_dir"])
    surface.to_csv(out / "join_ask_capacity_curve_terminal_gate.csv", index=False)

    summary = dict(res["summary"])
    summary.update({
        "version": VERSION,
        "capacity_gate": (
            "net PnL > 0 AND overall terminal exit coverage "
            "(passive + M5) / entered qty >= 99% AND no missing decision snapshot for full entries"
        ),
        "largest_positive_qty_with_99pct_terminal_coverage_and_no_missing_decision": largest_feasible,
        "highest_development_pnl_row": max_pnl,
        "note": (
            "V7.1 changes only the capacity-selection statistic. Strategy and all execution/PnL rows are identical to V7."
        ),
    })
    OOS._atomic_json(out / "summary_v7_1.json", summary)

    if show:
        print("=" * 170)
        print("DEEP-TAIL IMMEDIATE JOIN_ASK CAPACITY DEV V7.1")
        print("=" * 170)
        print("Strategy/execution: identical to V7")
        print("Capacity gate: positive PnL + >=99% OVERALL terminal coverage + zero missing full-entry decisions")
        print("DEVELOPMENT ONLY — 15h sample is NOT read")
        print()
        cols = [
            "requested_qty",
            "entry_fill_events",
            "full_entry_fill_orders",
            "partial_entry_fill_orders",
            "entry_filled_qty",
            "decision_snapshot_missing_full_entries",
            "median_join_ask_quote_c",
            "passive_exit_qty",
            "passive_exit_fraction_of_entry_qty",
            "full_passive_exit_positions",
            "m5_required_qty",
            "m5_exit_qty",
            "m5_exit_fraction_of_required_qty",
            "terminal_exit_fraction_of_entry_qty",
            "m5_residual_zero_valued",
            "m5_taker_fees",
            "rounding_drag",
            "m5_only_baseline_net",
            "join_ask_net_pnl_rounding_bound",
            "incremental_vs_m5_only",
            "net_pnl_per_filled_contract",
            "max_drawdown_rounding_bound",
            "top1_share_of_net",
            "top5_share_of_net",
            "capacity_gate_positive_99pct_terminal_no_missing_decision",
        ]
        print(surface[cols].to_string(index=False))
        print()
        print("Largest quantity passing the predeclared capacity gate:")
        print(largest_feasible)
        print()
        print("Highest development PnL row (diagnostic only; NOT automatic selection):")
        print(max_pnl)
        print()
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "capacity_curve": surface,
        "detail": detail,
        "by_asset": by_asset,
        "output_dir": str(out),
    }
