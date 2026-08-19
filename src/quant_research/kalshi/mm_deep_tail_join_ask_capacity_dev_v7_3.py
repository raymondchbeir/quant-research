from __future__ import annotations

"""Extended development-only capacity diagnostic for frozen 5c + immediate JOIN_ASK.

This module changes only requested quantity.  It extends the corrected V7.2 capacity
machinery beyond the old V3 Q1000 grid by reconstructing strict-through entry quantity
directly from the compact M1-M5 trade cache.  No held-out or 15h data are read.

Scientific status: development capacity diagnostic only.  Results above the already-
frozen Q10 are not validation and must not be treated as deployable size without fresh
forward/live evidence.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_capacity_dev_v7 as V7
from . import mm_deep_tail_join_ask_capacity_dev_v7_2 as V72

VERSION = "MM_DEEP_TAIL_JOIN_ASK_CAPACITY_DEV_V7_3_Q10_TO_Q1500"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_join_ask_capacity_dev_v7_3"

Q_GRID = (10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1250, 1500)
EPS = V7.EPS
ENTRY = V7.ENTRY
M1_S = V7.M1_S
ACTIVATION_LATENCY_MS = V7.ACTIVATION_LATENCY_MS
ROUNDING_DRAG = V7.ROUNDING_DRAG


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _zero_m5():
    return {
        "exit_qty": 0.0,
        "residual_qty_zero_valued": 0.0,
        "m5_exit_proceeds": 0.0,
        "m5_taker_fee": 0.0,
        "m5_slippage_vs_best_bid": 0.0,
        "m5_cross_cost_vs_mid": 0.0,
    }


def _m5_result(tail: str, qty: float, snap: dict, mult: float):
    if qty <= EPS:
        return _zero_m5()
    return V7.V3._consume_m5_depth(tail, qty, snap, mult)


def _m5_net(entry_qty: float, tail: str, snap: dict, mult: float):
    ex = _m5_result(tail, entry_qty, snap, mult)
    rounding = ROUNDING_DRAG if float(ex["exit_qty"]) > EPS else 0.0
    net = (
        float(ex["m5_exit_proceeds"])
        - ENTRY * float(entry_qty)
        - float(ex["m5_taker_fee"])
        - rounding
    )
    return float(net), ex, float(rounding)


def _candidate_positions(v3: pd.DataFrame):
    q5 = v3[
        np.isclose(pd.to_numeric(v3["requested_qty"], errors="coerce"), 5.0)
    ].copy()
    q5 = q5[
        q5["coverage_eligible"].astype(bool)
        & (pd.to_numeric(q5["entry_filled_qty"], errors="coerce").fillna(0.0) > EPS)
    ].copy()
    if q5.empty:
        raise RuntimeError("No positive Q5 candidate positions in prior V3 capacity detail")
    q5 = q5.drop_duplicates(["ticker", "tail"], keep="last")
    return q5[["ticker", "series", "tail"]].copy()


def _evaluate_grid(source: Path, meta: dict, candidates: pd.DataFrame, bbo: pd.DataFrame,
                   m5: dict, trades_df: pd.DataFrame, fee_mult: dict):
    bbo_by_ticker = {
        str(t): g.sort_values("receipt_s", kind="mergesort").reset_index(drop=True)
        for t, g in bbo.groupby("ticker", sort=False)
    }
    trades_by_ticker = {
        str(t): g.to_dict("records")
        for t, g in trades_df.groupby("ticker", sort=False)
    } if len(trades_df) else {}

    rows = []
    for requested in Q_GRID:
        requested = float(requested)
        for _, c in candidates.iterrows():
            ticker = str(c["ticker"])
            tail = str(c["tail"])
            series = str(c["series"])
            if ticker not in meta or ticker not in m5:
                raise RuntimeError(f"Missing metadata/M5 cache for candidate {ticker} {tail}")
            mult = float(fee_mult.get(series, np.nan))
            if not np.isfinite(mult) or mult <= 0:
                raise RuntimeError(f"Missing fee multiplier for {series}")

            active_s = float(meta[ticker]["window_start_s"]) + M1_S + ACTIVATION_LATENCY_MS / 1000.0
            clk = V7._full_fill_clock(
                trades_by_ticker.get(ticker, []), tail, requested, active_s
            )
            entry_qty = float(clk["reconstructed_fill_qty"])
            if entry_qty <= EPS:
                continue

            full_entry = bool(clk["reconstructed_full"])
            partial_entry = bool(not full_entry and entry_qty > EPS)
            first_exec = float(clk["first_entry_exec_s_reconstructed"])
            full_obs = float(clk["full_entry_observed_s_reconstructed"]) if full_entry else np.nan
            snap_m5 = m5[ticker]

            baseline_net, baseline_ex, baseline_rounding = _m5_net(
                entry_qty, tail, snap_m5, mult
            )

            passive_qty = 0.0
            passive_full = False
            quote_px = np.nan
            queue0 = np.nan
            seconds_to_full_exit = np.nan
            decision_available = False
            missing_decision = False

            if full_entry:
                g = bbo_by_ticker.get(ticker)
                snap = V7._latest_known_bbo(g, full_obs) if g is not None and len(g) else None
                if snap is None:
                    missing_decision = True
                else:
                    decision_available = True
                    ob = V7._outcome_bbo_and_queue(snap, tail)
                    quote_px = float(ob["ask"])
                    queue0 = float(ob["ask_queue"])
                    exit_active_s = full_obs + ACTIVATION_LATENCY_MS / 1000.0
                    pex = V7.V4._simulate_passive_exit(
                        trades_by_ticker.get(ticker, []),
                        tail,
                        {"quote_price": quote_px, "queue_ahead_initial": queue0},
                        exit_active_s,
                        entry_qty,
                    )
                    passive_qty = float(pex["passive_exit_qty"])
                    passive_full = bool(pex["passive_exit_full"])
                    z = V7._f(pex.get("full_passive_exit_exec_s"))
                    if passive_full and np.isfinite(z):
                        seconds_to_full_exit = float(z - full_obs)

            residual = max(0.0, entry_qty - passive_qty)
            m5ex = _m5_result(tail, residual, snap_m5, mult)
            rounding = ROUNDING_DRAG if float(m5ex["exit_qty"]) > EPS else 0.0
            passive_proceeds = passive_qty * (quote_px if np.isfinite(quote_px) else 0.0)
            net = (
                passive_proceeds
                + float(m5ex["m5_exit_proceeds"])
                - ENTRY * entry_qty
                - float(m5ex["m5_taker_fee"])
                - rounding
            )

            rows.append({
                "requested_qty": requested,
                "ticker": ticker,
                "series": series,
                "tail": tail,
                "entry_filled_qty": entry_qty,
                "full_entry_fill": full_entry,
                "partial_entry_fill": partial_entry,
                "decision_snapshot_available": decision_available,
                "decision_snapshot_missing_full_entry": bool(missing_decision),
                "quote_c": 100.0 * quote_px if np.isfinite(quote_px) else np.nan,
                "queue_ahead_initial": queue0,
                "passive_exit_qty": passive_qty,
                "passive_exit_full": passive_full,
                "m5_required_qty": residual,
                "m5_exit_qty": float(m5ex["exit_qty"]),
                "m5_residual_zero_valued": float(m5ex["residual_qty_zero_valued"]),
                "m5_taker_fee": float(m5ex["m5_taker_fee"]),
                "rounding_drag": float(rounding),
                "net_pnl_rounding_bound": float(net),
                "m5_only_baseline_net": float(baseline_net),
                "incremental_vs_m5_only": float(net - baseline_net),
                "first_entry_exec_s": first_exec,
                "full_entry_observed_s": full_obs,
                "seconds_full_fill_to_passive_full": seconds_to_full_exit,
                "strict_through_qty_seen": float(clk["strict_through_qty_seen_until_full_or_end"]),
                "baseline_m5_exit_qty": float(baseline_ex["exit_qty"]),
                "baseline_m5_residual_zero_valued": float(baseline_ex["residual_qty_zero_valued"]),
                "baseline_rounding_drag": baseline_rounding,
            })

    return pd.DataFrame(rows)


def _max_drawdown(g: pd.DataFrame):
    if g.empty:
        return 0.0
    q = g.sort_values(["first_entry_exec_s", "ticker", "tail"], kind="mergesort")
    pnl = pd.to_numeric(q["net_pnl_rounding_bound"], errors="coerce").fillna(0.0).to_numpy(float)
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    return float((np.r_[0.0, eq] - peak).min())


def _surface(detail: pd.DataFrame):
    rows = []
    for requested in Q_GRID:
        g = detail[np.isclose(detail["requested_qty"].astype(float), float(requested))].copy()
        entry = float(g["entry_filled_qty"].sum()) if len(g) else 0.0
        pqty = float(g["passive_exit_qty"].sum()) if len(g) else 0.0
        m5req = float(g["m5_required_qty"].sum()) if len(g) else 0.0
        m5exit = float(g["m5_exit_qty"].sum()) if len(g) else 0.0
        resid = float(g["m5_residual_zero_valued"].sum()) if len(g) else 0.0
        net = float(g["net_pnl_rounding_bound"].sum()) if len(g) else 0.0
        baseline = float(g["m5_only_baseline_net"].sum()) if len(g) else 0.0
        perpos = pd.to_numeric(g["net_pnl_rounding_bound"], errors="coerce").fillna(0.0).sort_values(ascending=False)
        top1 = float(perpos.head(1).sum()) if len(perpos) else 0.0
        top5 = float(perpos.head(5).sum()) if len(perpos) else 0.0
        terminal = (pqty + m5exit) / entry if entry > EPS else np.nan
        passive = pqty / entry if entry > EPS else np.nan
        missing = int(g["decision_snapshot_missing_full_entry"].sum()) if len(g) else 0
        rows.append({
            "requested_qty": float(requested),
            "entry_fill_events": int(len(g)),
            "full_entry_fill_orders": int(g["full_entry_fill"].sum()) if len(g) else 0,
            "partial_entry_fill_orders": int(g["partial_entry_fill"].sum()) if len(g) else 0,
            "entry_filled_qty": entry,
            "decision_snapshot_missing_full_entries": missing,
            "passive_exit_qty": pqty,
            "passive_exit_fraction_of_entry_qty": passive,
            "m5_required_qty": m5req,
            "m5_exit_qty": m5exit,
            "terminal_exit_fraction_of_entry_qty": terminal,
            "m5_residual_zero_valued": resid,
            "join_ask_net_pnl_rounding_bound": net,
            "m5_only_baseline_net": baseline,
            "incremental_vs_m5_only": net - baseline,
            "net_pnl_per_filled_contract": net / entry if entry > EPS else np.nan,
            "max_drawdown_rounding_bound": _max_drawdown(g),
            "top1_share_of_net": top1 / net if net > EPS else np.nan,
            "top5_share_of_net": top5 / net if net > EPS else np.nan,
            "coverage_ge_99": bool(np.isfinite(terminal) and terminal >= 0.99 - EPS),
            "coverage_ge_95": bool(np.isfinite(terminal) and terminal >= 0.95 - EPS),
            "coverage_ge_90": bool(np.isfinite(terminal) and terminal >= 0.90 - EPS),
        })
    return pd.DataFrame(rows)


def _by_asset(detail: pd.DataFrame):
    rows = []
    for (requested, series), g in detail.groupby(["requested_qty", "series"], sort=True):
        entry = float(g["entry_filled_qty"].sum())
        pqty = float(g["passive_exit_qty"].sum())
        m5q = float(g["m5_exit_qty"].sum())
        rows.append({
            "requested_qty": float(requested),
            "series": str(series),
            "positions": int(len(g)),
            "entry_filled_qty": entry,
            "passive_exit_qty": pqty,
            "passive_exit_fraction": pqty / entry if entry > EPS else np.nan,
            "m5_exit_qty": m5q,
            "m5_residual_zero_valued": float(g["m5_residual_zero_valued"].sum()),
            "net_pnl_rounding_bound": float(g["net_pnl_rounding_bound"].sum()),
        })
    return pd.DataFrame(rows)


def run_extended_capacity_dev(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != V7.HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected development source {V7.HARD_BOUND_SESSION}, got {source.name}")
    if hard_bind and "mm_event_m0_m5_oos_cycle_q10_v1" not in str(source.parent):
        raise RuntimeError("Expected source under mm_event_m0_m5_oos_cycle_q10_v1")

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored development fee preflight is not PASS")
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    meta = V7.V3.V1._metadata(source)
    v3, _, v3_path, _ = V7._load_required_prior_results(source)
    candidates = _candidate_positions(v3)

    base_bbo, m5, base_bbo_path, m5_path = V7._load_book_cache(source)
    anchors, anchor_source = V72._load_q5_anchors(source)
    prelude, prelude_path, _ = V72._build_prelude(source, meta, anchors, show=show)
    bbo = V72._augment_bbo(base_bbo, prelude)

    tickers = set(candidates["ticker"].astype(str))
    trades_df, trade_filter = V7._load_m1_trade_cache(source, meta, tickers, show=show)

    if show:
        print("=" * 118)
        print("EXTENDED JOIN_ASK CAPACITY DIAGNOSTIC — Q10 TO Q1500")
        print("=" * 118)
        print("Source:", source)
        print("Grid:", Q_GRID)
        print("Only requested quantity changes. Entry/exit/latency/queue/M5 mechanics stay frozen.")
        print("DEVELOPMENT CAPACITY DIAGNOSTIC ONLY — 15h and held-out sessions are NOT read.")

    detail = _evaluate_grid(source, meta, candidates, bbo, m5, trades_df, fee_mult)
    surface = _surface(detail)
    by_asset = _by_asset(detail)

    out = _new_output(source.name)
    detail.to_csv(out / "extended_capacity_detail.csv", index=False)
    surface.to_csv(out / "extended_capacity_curve.csv", index=False)
    by_asset.to_csv(out / "extended_capacity_by_asset.csv", index=False)

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "requested_qty_grid": list(Q_GRID),
        "candidate_positions": int(len(candidates)),
        "strategy": "5c strict-through entry; full fill -> immediate fixed JOIN_ASK +100ms; partial -> M5-only; M5 top3 residual",
        "scientific_label": "DEVELOPMENT_CAPACITY_DIAGNOSTIC_ONLY",
        "prior_v3_detail": v3_path,
        "anchor_source": anchor_source,
        "book_cache": base_bbo_path,
        "prelude_cache": prelude_path,
        "m5_cache": m5_path,
        "trade_filter": trade_filter,
        "output_dir": str(out),
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
    }
    OOS._atomic_json(out / "summary.json", summary)

    if show:
        print()
        for _, r in surface.iterrows():
            print(
                f"Q{int(r['requested_qty']):4d} | "
                f"fills {int(r['entry_fill_events']):2d} "
                f"(full {int(r['full_entry_fill_orders']):2d}/partial {int(r['partial_entry_fill_orders']):2d}) | "
                f"entered {r['entry_filled_qty']:8.1f} | "
                f"passive {100*r['passive_exit_fraction_of_entry_qty']:6.1f}% | "
                f"terminal {100*r['terminal_exit_fraction_of_entry_qty']:6.1f}% | "
                f"resid {r['m5_residual_zero_valued']:7.1f} | "
                f"PnL ${r['join_ask_net_pnl_rounding_bound']:+8.2f} | "
                f"edge {100*r['net_pnl_per_filled_contract']:+6.2f}c | "
                f"DD ${r['max_drawdown_rounding_bound']:+7.2f}"
            )
        print()
        print("No quantity above Q10 is promoted by this diagnostic. Fresh forward/live evidence is still required.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "capacity_curve": surface,
        "detail": detail,
        "by_asset": by_asset,
        "output_dir": str(out),
    }
