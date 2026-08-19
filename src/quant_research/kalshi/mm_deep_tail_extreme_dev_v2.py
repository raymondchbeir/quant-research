from __future__ import annotations

"""Extreme deep-tail development sweep built on the conservative V1 replay.

Purpose
-------
Extend the already-inspected 24h DEVELOPMENT sample below the original 10c tail
entry and determine whether the apparent extreme-tail edge strengthens at 1-7c.
This remains DEVELOPMENT ONLY.  Nothing from the separate ~15h validation sample
is read here.

The primary execution model is inherited unchanged from
``mm_deep_tail_passive_feasibility_dev_v1``:
- Q1 symmetric BUY YES / BUY NO tail orders;
- intended from M1, active after 100ms;
- V11 causal economic trade clock;
- strict trade-through entry confirmation only (exact-price touches do not fill);
- strict trade-through passive target exits;
- otherwise executable M5 fallback with stored quadratic taker fee;
- separate $0.0099/cross conservative rounding-drag upper bound;
- deep FIFO queue is never invented from missing book depth.

Exploratory grid
----------------
Entry cents: 1, 2, 3, 5, 7, 10, 15
Target cents: 10, 15, 20, 25, 30, 35, 40, 45, 50 (only targets > entry)

V2 also computes a pure M5-only rule for every entry level, plus distribution,
asset, window, concentration, drawdown, and leave-one-asset-out diagnostics.  The
point is to look for a broad robust region, not to freeze the single largest cell.

NO API calls.  NO orders.  Source capture is read-only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1

VERSION = "MM_DEEP_TAIL_EXTREME_DEV_V2"
HARD_BOUND_SESSION = V1.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_extreme_dev_v2"

ENTRY_LEVELS_C = (1, 2, 3, 5, 7, 10, 15)
EXIT_TARGETS_C = (10, 15, 20, 25, 30, 35, 40, 45, 50)
QTY = 1.0
EPS = 1e-12


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _safe_float(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _m5_detail(orders: pd.DataFrame, fee_mult: dict[str, float]) -> pd.DataFrame:
    x = orders.copy()
    x["m5_only_fee"] = 0.0
    x["m5_only_pnl"] = 0.0
    x["m5_only_pnl_rounding_upper_bound"] = 0.0
    x["m5_only_exit_known"] = True

    for idx, r in x.iterrows():
        if not bool(r.get("coverage_eligible", False)):
            x.at[idx, "m5_only_pnl"] = np.nan
            x.at[idx, "m5_only_pnl_rounding_upper_bound"] = np.nan
            x.at[idx, "m5_only_exit_known"] = False
            continue
        if not bool(r.get("conservative_entry_fill", False)):
            continue

        bid = _safe_float(r.get("m5_outcome_bid"))
        entry = _safe_float(r.get("entry_c")) / 100.0
        mult = _safe_float(fee_mult.get(str(r.get("series") or "")))
        if not (np.isfinite(bid) and np.isfinite(entry) and np.isfinite(mult)):
            x.at[idx, "m5_only_fee"] = np.nan
            x.at[idx, "m5_only_pnl"] = np.nan
            x.at[idx, "m5_only_pnl_rounding_upper_bound"] = np.nan
            x.at[idx, "m5_only_exit_known"] = False
            continue

        fee = OOS._quadratic_taker_fee(QTY, bid, mult)
        pnl = (bid - entry) * QTY - fee
        pnl_round = pnl - V1.BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS
        x.at[idx, "m5_only_fee"] = float(fee)
        x.at[idx, "m5_only_pnl"] = float(pnl)
        x.at[idx, "m5_only_pnl_rounding_upper_bound"] = float(pnl_round)

    return x


def _drawdown(values) -> float:
    a = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return np.nan
    equity = np.cumsum(a)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    dd = np.r_[0.0, equity] - peak
    return float(np.min(dd))


def _concentration(pnl: pd.Series):
    a = pd.to_numeric(pnl, errors="coerce").dropna().sort_values(ascending=False)
    total = float(a.sum()) if len(a) else np.nan
    out = {}
    for k in (1, 5, 10):
        top = float(a.head(k).sum()) if len(a) else np.nan
        out[f"top_{k}_pnl"] = top
        out[f"top_{k}_share_of_net"] = (top / total) if len(a) and abs(total) > EPS else np.nan
    return out


def _aggregate_m5(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = detail[detail["coverage_eligible"]].copy()
    for tail_key, q0 in list(eligible.groupby("tail", sort=True)) + [("BOTH", eligible)]:
        for entry_c, g in q0.groupby("entry_c", sort=True):
            fills = g[g["conservative_entry_fill"]].copy()
            known = fills[fills["m5_only_exit_known"]].copy()
            p = pd.to_numeric(known["m5_only_pnl_rounding_upper_bound"], errors="coerce").dropna()
            p_raw = pd.to_numeric(known["m5_only_pnl"], errors="coerce").dropna()
            chronological = known.sort_values("fill_exec_s")
            cp = pd.to_numeric(chronological["m5_only_pnl_rounding_upper_bound"], errors="coerce")
            c = _concentration(p)
            rows.append({
                "tail": str(tail_key),
                "entry_c": int(entry_c),
                "eligible_posted_orders": int(len(g)),
                "entry_fills": int(len(fills)),
                "entry_fill_rate": float(len(fills) / len(g)) if len(g) else np.nan,
                "known_m5_exits": int(len(known)),
                "missing_m5_exits": int(len(fills) - len(known)),
                "total_m5_net_pnl_q1": float(p_raw.sum()) if len(p_raw) else 0.0,
                "total_m5_net_pnl_q1_rounding_upper_bound": float(p.sum()) if len(p) else 0.0,
                "m5_rounding_pnl_per_posted_order": float(p.sum() / len(g)) if len(g) else np.nan,
                "m5_rounding_pnl_per_fill": float(p.mean()) if len(p) else np.nan,
                "m5_rounding_median_pnl_per_fill": float(p.median()) if len(p) else np.nan,
                "m5_rounding_win_rate_per_fill": float((p > 0).mean()) if len(p) else np.nan,
                "m5_rounding_q10_pnl_per_fill": float(p.quantile(.10)) if len(p) else np.nan,
                "m5_rounding_q25_pnl_per_fill": float(p.quantile(.25)) if len(p) else np.nan,
                "m5_rounding_q75_pnl_per_fill": float(p.quantile(.75)) if len(p) else np.nan,
                "m5_rounding_q90_pnl_per_fill": float(p.quantile(.90)) if len(p) else np.nan,
                "m5_rounding_worst_fill": float(p.min()) if len(p) else np.nan,
                "m5_rounding_best_fill": float(p.max()) if len(p) else np.nan,
                "m5_rounding_max_drawdown": _drawdown(cp.dropna().tolist()),
                **c,
            })
    return pd.DataFrame(rows).sort_values(["entry_c", "tail"]).reset_index(drop=True)


def _m5_by_asset(detail: pd.DataFrame) -> pd.DataFrame:
    q = detail[
        detail["coverage_eligible"]
        & detail["conservative_entry_fill"]
        & detail["m5_only_exit_known"]
    ].copy()
    if q.empty:
        return pd.DataFrame()
    rows = []
    for (entry_c, series), g in q.groupby(["entry_c", "series"], sort=True):
        p = pd.to_numeric(g["m5_only_pnl_rounding_upper_bound"], errors="coerce").dropna()
        rows.append({
            "entry_c": int(entry_c),
            "series": str(series),
            "fills": int(len(g)),
            "total_rounding_bound_pnl": float(p.sum()),
            "mean_pnl_per_fill": float(p.mean()) if len(p) else np.nan,
            "median_pnl_per_fill": float(p.median()) if len(p) else np.nan,
            "win_rate": float((p > 0).mean()) if len(p) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["entry_c", "total_rounding_bound_pnl"], ascending=[True, False])


def _m5_by_window(detail: pd.DataFrame) -> pd.DataFrame:
    q = detail[detail["coverage_eligible"]].copy()
    rows = []
    for (entry_c, close), g in q.groupby(["entry_c", "close_time"], sort=True):
        fills = g[g["conservative_entry_fill"] & g["m5_only_exit_known"]]
        p = pd.to_numeric(fills["m5_only_pnl_rounding_upper_bound"], errors="coerce").dropna()
        rows.append({
            "entry_c": int(entry_c),
            "close_time": str(close),
            "posted_orders": int(len(g)),
            "fills": int(g["conservative_entry_fill"].sum()),
            "known_m5_exits": int(len(fills)),
            "rounding_bound_pnl": float(p.sum()),
        })
    return pd.DataFrame(rows).sort_values(["entry_c", "close_time"]).reset_index(drop=True)


def _leave_one_asset_out(detail: pd.DataFrame) -> pd.DataFrame:
    q = detail[
        detail["coverage_eligible"]
        & detail["conservative_entry_fill"]
        & detail["m5_only_exit_known"]
    ].copy()
    if q.empty:
        return pd.DataFrame()
    rows = []
    for entry_c, g in q.groupby("entry_c", sort=True):
        total = float(pd.to_numeric(g["m5_only_pnl_rounding_upper_bound"], errors="coerce").fillna(0).sum())
        by_asset = g.groupby("series")["m5_only_pnl_rounding_upper_bound"].sum()
        for series, asset_pnl in by_asset.items():
            rows.append({
                "entry_c": int(entry_c),
                "excluded_series": str(series),
                "full_pnl": total,
                "excluded_asset_pnl": float(asset_pnl),
                "pnl_without_asset": float(total - asset_pnl),
            })
    return pd.DataFrame(rows).sort_values(["entry_c", "pnl_without_asset"]).reset_index(drop=True)


def _strategy_comparison(surface: pd.DataFrame, m5_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    s = surface[surface["tail"].eq("BOTH")].copy()
    for _, r in s.iterrows():
        rows.append({
            "entry_c": int(r["entry_c"]),
            "exit_rule": f"TARGET_{int(r['target_c'])}C",
            "target_c": int(r["target_c"]),
            "eligible_posted_orders": int(r["coverage_eligible_posted_orders"]),
            "entry_fills": int(r["entry_fills"]),
            "entry_fill_rate": float(r["entry_fill_rate"]),
            "target_exit_rate_given_fill": float(r["p_conservative_target_exit_given_fill"]),
            "total_rounding_bound_pnl": float(r["total_net_pnl_q1_rounding_upper_bound"]),
            "rounding_bound_pnl_per_posted_order": float(r["rounding_upper_bound_pnl_per_eligible_posted_order"]),
        })
    m = m5_summary[m5_summary["tail"].eq("BOTH")]
    for _, r in m.iterrows():
        rows.append({
            "entry_c": int(r["entry_c"]),
            "exit_rule": "M5_ONLY",
            "target_c": np.nan,
            "eligible_posted_orders": int(r["eligible_posted_orders"]),
            "entry_fills": int(r["entry_fills"]),
            "entry_fill_rate": float(r["entry_fill_rate"]),
            "target_exit_rate_given_fill": np.nan,
            "total_rounding_bound_pnl": float(r["total_m5_net_pnl_q1_rounding_upper_bound"]),
            "rounding_bound_pnl_per_posted_order": float(r["m5_rounding_pnl_per_posted_order"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["rounding_bound_pnl_per_posted_order", "entry_fills"], ascending=False
    ).reset_index(drop=True)


def run_extreme_deep_tail_dev(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected development source {HARD_BOUND_SESSION}, got {source.name}")
    if hard_bind and "mm_event_m0_m5_oos_cycle_q10_v1" not in str(source.parent):
        raise RuntimeError("Expected source under mm_event_m0_m5_oos_cycle_q10_v1")

    if show:
        print("=" * 150)
        print("EXTREME DEEP-TAIL DEVELOPMENT V2 — READ ONLY")
        print("=" * 150)
        print("Entry grid:", ENTRY_LEVELS_C)
        print("Target grid:", EXIT_TARGETS_C)
        print("Primary fills: strict trade-through only")
        print("Q1 | M1-M5 | V11 causal trade clock | 100ms activation")
        print()

    # Reuse V1 mechanics exactly, changing only the exploratory development grid
    # and result destination/version.  Restore globals even if the run errors.
    old = {
        "ENTRY_LEVELS_C": V1.ENTRY_LEVELS_C,
        "EXIT_TARGETS_C": V1.EXIT_TARGETS_C,
        "OUTPUT_ROOT": V1.OUTPUT_ROOT,
        "VERSION": V1.VERSION,
    }
    try:
        V1.ENTRY_LEVELS_C = ENTRY_LEVELS_C
        V1.EXIT_TARGETS_C = EXIT_TARGETS_C
        V1.OUTPUT_ROOT = OUTPUT_ROOT
        V1.VERSION = VERSION
        base = V1.run_deep_tail_passive_feasibility(source, hard_bind=hard_bind, show=show)
    finally:
        V1.ENTRY_LEVELS_C = old["ENTRY_LEVELS_C"]
        V1.EXIT_TARGETS_C = old["EXIT_TARGETS_C"]
        V1.OUTPUT_ROOT = old["OUTPUT_ROOT"]
        V1.VERSION = old["VERSION"]

    out = Path(base["output_dir"]).resolve()
    orders = base["orders"].copy()
    surface = base["surface"].copy()

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    m5_detail = _m5_detail(orders, fee_mult)
    m5_summary = _aggregate_m5(m5_detail)
    by_asset = _m5_by_asset(m5_detail)
    by_window = _m5_by_window(m5_detail)
    loo_asset = _leave_one_asset_out(m5_detail)
    strategies = _strategy_comparison(surface, m5_summary)

    m5_detail.to_csv(out / "m5_only_order_detail.csv", index=False)
    m5_summary.to_csv(out / "m5_only_summary.csv", index=False)
    by_asset.to_csv(out / "m5_only_by_asset.csv", index=False)
    by_window.to_csv(out / "m5_only_by_window.csv", index=False)
    loo_asset.to_csv(out / "m5_only_leave_one_asset_out.csv", index=False)
    strategies.to_csv(out / "strategy_comparison_all_targets_and_m5.csv", index=False)

    best = strategies.iloc[0].to_dict() if len(strategies) else {}
    positive = strategies[strategies["rounding_bound_pnl_per_posted_order"] > 0].copy()

    summary_v2 = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "research_stage": "DEVELOPMENT_DISCOVERY_ONLY",
        "entry_levels_c": list(ENTRY_LEVELS_C),
        "exit_targets_c": list(EXIT_TARGETS_C),
        "qty": QTY,
        "primary_fill_rule": base["summary"].get("primary_fill_rule"),
        "clock_policy": base["summary"].get("clock_policy"),
        "queue_policy": base["summary"].get("queue_policy"),
        "best_exploratory_strategy_row": best,
        "positive_strategy_cells": int(len(positive)),
        "total_strategy_cells": int(len(strategies)),
        "guardrail": (
            "Development only.  The lower-tail grid was chosen after seeing the original 10c result. "
            "Do not treat the best cell as validation.  Freeze a simple robust rule before opening the ~15h validation sample."
        ),
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
    }
    _atomic_json(out / "summary_v2.json", summary_v2)

    if show:
        print("\n" + "=" * 150)
        print("V2 EXTREME-TAIL SUMMARY")
        print("=" * 150)
        print("\nM5-ONLY — BOTH TAILS COMBINED")
        cols = [
            "entry_c", "eligible_posted_orders", "entry_fills", "entry_fill_rate",
            "total_m5_net_pnl_q1_rounding_upper_bound", "m5_rounding_pnl_per_posted_order",
            "m5_rounding_pnl_per_fill", "m5_rounding_median_pnl_per_fill",
            "m5_rounding_win_rate_per_fill", "m5_rounding_worst_fill", "m5_rounding_best_fill",
            "m5_rounding_max_drawdown", "top_1_share_of_net", "top_5_share_of_net", "top_10_share_of_net",
        ]
        print(m5_summary[m5_summary["tail"].eq("BOTH")][cols].to_string(index=False))

        print("\nTOP 30 STRATEGY CELLS — TARGETS + M5 ONLY")
        cols2 = [
            "entry_c", "exit_rule", "eligible_posted_orders", "entry_fills", "entry_fill_rate",
            "target_exit_rate_given_fill", "total_rounding_bound_pnl", "rounding_bound_pnl_per_posted_order",
        ]
        print(strategies[cols2].head(30).to_string(index=False))

        print("\nM5-ONLY BY ASSET")
        print(by_asset.to_string(index=False))

        print("\nLEAVE-ONE-ASSET-OUT — WORST REMOVALS FIRST")
        print(loo_asset.to_string(index=False))

        print("\nBest exploratory row:")
        print(best)
        print("\nPositive strategy cells:", len(positive), "/", len(strategies))
        print("\nIMPORTANT: lower levels were added after seeing 10c. This is still development, not confirmation.")
        print("Do not open the 15h validation set until we choose and freeze one simple rule.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        **base,
        "summary_v2": summary_v2,
        "m5_detail": m5_detail,
        "m5_summary": m5_summary,
        "m5_by_asset": by_asset,
        "m5_by_window": by_window,
        "leave_one_asset_out": loo_asset,
        "strategy_comparison": strategies,
        "output_dir": str(out),
    }
