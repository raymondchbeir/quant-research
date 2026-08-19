from __future__ import annotations

"""Frozen Q5 holdout validation for the 5c deep-tail / M5-only strategy.

Scientific role
---------------
The deep-tail hypothesis, 5c entry, M5-only exit, all-9-series universe, and Q5 size
were selected using the separate 24h development realization ``20260817_064143``.
This module is hard-bound to the earlier ~15h event-time capture ``20260816_070627``
and evaluates ONE frozen specification only.  It does not scan alternative prices,
quantities, exits, assets, or filters.

Frozen strategy
---------------
- Universe: every recorded crypto series in the validation capture (expected frozen 9).
- From M1: rest BUY YES Q5 @ 5c and BUY NO Q5 @ 5c.
- Entry activation: M1 + 100ms.
- Entry capacity: cumulative same-outcome aggressive seller quantity trading STRICTLY
  THROUGH 5c while the order is active.  Exact-price 5c prints are excluded because
  deep FIFO queue ahead is unobserved.  Partial fills are allowed.
- Never reprice/cancel before M5.
- M5: cross recorded top-3 outcome bid depth L1->L2->L3.
- Residual beyond observed top-3 receives ZERO value in primary PnL.
- Entry maker fee: zero under the stored quadratic fee schedule.
- M5 taker fees: quadratic fee per consumed level.
- Balance rounding: subtract additional $0.0099 per nonzero M5 exit.
- Spread crossing and book-walk slippage are embedded in actual exit proceeds and
  reported as diagnostics only (never double-subtracted).

Validation gate (declared before opening this sample for this strategy)
-----------------------------------------------------------------------
PASS requires all of:
1. rounding-bound net PnL > 0;
2. >= 10 entry fill events (otherwise INCONCLUSIVE_SAMPLE_SIZE);
3. >= 95% of filled quantity exits inside recorded M5 top-3.

If >=10 fills but either economics or exit coverage fail, validation FAILS.  No retuning
is permitted after seeing the validation output.  A pass advances the exact unchanged
specification to the separate final test sample.

Fee provenance
--------------
If the validation session contains its own PASS ``fee_preflight.json``, it is used.
Some early V5 development captures predate that artifact.  In that case this script uses
the frozen formal-OOS PASS fee preflight from ``20260817_064143`` strictly as a cost-model
reference and labels that provenance explicitly.  This fallback changes no trading rule.

READ ONLY.  NO API calls.  NO orders.  Source capture is never modified.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_capacity_dev_v3 as V3

VERSION = "MM_DEEP_TAIL_Q5_VALIDATION_V1"
HARD_BOUND_SESSION = "20260816_070627"
EXPECTED_PARENT = "mm_event_m0_m5_v5_dev"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_q5_validation_v1"

FROZEN_ENTRY_C = 5
FROZEN_QTY = 5.0
FROZEN_EXIT = "M5_ONLY"
FROZEN_UNIVERSE = tuple(OOS.SERIES)
DEV_SOURCE_SESSION = "20260817_064143"
DEV_Q5_ROUNDING_BOUND_PNL = 8.04614
DEV_Q5_EXIT_COVERAGE = 0.9939393939393939

MIN_FILL_EVENTS_FOR_DECISION = 10
MIN_M5_EXIT_COVERAGE = 0.95
EPS = 1e-10

FALLBACK_FEE_PREFLIGHT = (
    C.DATA_ROOT
    / "mm_event_m0_m5_oos_cycle_q10_v1"
    / DEV_SOURCE_SESSION
    / "fee_preflight.json"
)


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _resolve_fee_preflight(source: Path):
    local = source / "fee_preflight.json"
    if local.exists():
        fee = OOS._read_json(local, {}) or {}
        if fee.get("ok"):
            return fee, local.resolve(), "VALIDATION_SESSION_LOCAL"
        raise RuntimeError("Validation-local fee_preflight.json exists but is not PASS.")

    ref = FALLBACK_FEE_PREFLIGHT.resolve()
    if not ref.exists():
        raise FileNotFoundError(
            "Validation session has no fee_preflight.json and frozen fallback fee artifact "
            f"is missing: {ref}"
        )
    fee = OOS._read_json(ref, {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Frozen fallback formal-OOS fee_preflight.json is not PASS.")
    return fee, ref, "FROZEN_LATER_FORMAL_OOS_COST_REFERENCE"


def _window_summary(detail: pd.DataFrame):
    q = detail[detail["coverage_eligible"]].copy()
    rows = []
    for close, g in q.groupby("close_time", sort=True):
        filled = g[pd.to_numeric(g["entry_filled_qty"], errors="coerce") > EPS].copy()
        pnl = pd.to_numeric(filled["net_pnl_rounding_bound"], errors="coerce").fillna(0.0)
        rows.append({
            "close_time": str(close),
            "posted_orders": int(len(g)),
            "fill_events": int(len(filled)),
            "entry_filled_qty": float(pd.to_numeric(filled["entry_filled_qty"], errors="coerce").fillna(0).sum()),
            "m5_exit_qty": float(pd.to_numeric(filled["exit_qty"], errors="coerce").fillna(0).sum()),
            "residual_qty_zero_valued": float(pd.to_numeric(filled["residual_qty_zero_valued"], errors="coerce").fillna(0).sum()),
            "net_pnl_rounding_bound": float(pnl.sum()),
        })
    return pd.DataFrame(rows)


def _validation_status(curve_row: dict):
    fills = int(curve_row.get("entry_fill_events", 0) or 0)
    pnl = float(curve_row.get("net_pnl_rounding_bound", np.nan))
    exit_cov = float(curve_row.get("m5_exit_fraction_of_filled_qty", np.nan))

    if fills < MIN_FILL_EVENTS_FOR_DECISION:
        return "INCONCLUSIVE_SAMPLE_SIZE"
    if not np.isfinite(pnl) or pnl <= 0:
        return "FAIL_NONPOSITIVE_PNL"
    if not np.isfinite(exit_cov) or exit_cov < MIN_M5_EXIT_COVERAGE:
        return "FAIL_EXIT_CAPACITY"
    return "PASS"


def run_deep_tail_q5_validation(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected frozen validation source {HARD_BOUND_SESSION}, got {source.name}")
    if hard_bind and EXPECTED_PARENT not in str(source.parent):
        raise RuntimeError(f"Expected validation source under {EXPECTED_PARENT}")

    required = [
        source / "book_top3_events.jsonl",
        source / "trades_event_time.jsonl",
        source / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing validation artifacts: " + " | ".join(missing))

    fee, fee_path, fee_provenance = _resolve_fee_preflight(source)
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    meta = V1._metadata(source)
    if not meta:
        raise RuntimeError("No valid validation market metadata found.")

    observed_series = sorted({str(x.get("series") or "") for x in meta.values() if str(x.get("series") or "")})
    missing_fee_series = sorted(s for s in observed_series if s not in fee_mult)
    if missing_fee_series:
        raise RuntimeError(f"Missing frozen fee multipliers for validation series: {missing_fee_series}")

    if show:
        print("=" * 150)
        print("FROZEN DEEP-TAIL Q5 VALIDATION V1 — READ ONLY")
        print("=" * 150)
        print("Source:", source)
        print("Frozen entry:", f"{FROZEN_ENTRY_C}c")
        print("Frozen quantity:", f"Q{int(FROZEN_QTY)} per tail")
        print("Frozen exit:", FROZEN_EXIT)
        print("No parameter sweep. No asset filter. No retuning.")
        print("Fee provenance:", fee_provenance)
        print("Fee artifact:", fee_path)
        print()

    if show:
        print("PASS 1/3 — loading validation M1-M5 trades with frozen V11 causal clock...")
    trades, trade_clock_stats = V1._load_trades(source, meta, show=show)

    if show:
        print("PASS 2/3 — scanning validation true-M5 top-3 depth and coverage...")
    books = V3._scan_m5_books(source, meta, show=show)

    if show:
        print("PASS 3/3 — replaying ONE frozen Q5 specification...")

    old_qtys = V3.REQUESTED_QTYS
    try:
        V3.REQUESTED_QTYS = (FROZEN_QTY,)
        detail = V3._evaluate(source, meta, trades, books, fee_mult, show=show)
    finally:
        V3.REQUESTED_QTYS = old_qtys

    # Defensive invariants: validation must contain exactly one tested size and the frozen entry.
    tested_q = sorted(set(pd.to_numeric(detail["requested_qty"], errors="coerce").dropna().tolist()))
    if tested_q != [FROZEN_QTY]:
        raise RuntimeError(f"Validation contamination: expected only Q5, got {tested_q}")
    if abs(float(V3.ENTRY_C) - FROZEN_ENTRY_C) > EPS:
        raise RuntimeError("Underlying capacity engine entry price no longer equals frozen 5c.")

    curve = V3._aggregate_curve(detail)
    asset = V3._aggregate_asset(detail)
    windows = _window_summary(detail)

    if len(curve) != 1:
        raise RuntimeError(f"Expected one frozen validation curve row, got {len(curve)}")
    row = curve.iloc[0].to_dict()
    status = _validation_status(row)

    # Context only, never used to alter the frozen specification.
    pnl = float(row.get("net_pnl_rounding_bound", np.nan))
    fill_qty = float(row.get("entry_filled_qty", 0.0))
    exit_cov = float(row.get("m5_exit_fraction_of_filled_qty", np.nan))
    fill_events = int(row.get("entry_fill_events", 0))

    closes = pd.to_datetime(windows.get("close_time", pd.Series(dtype=str)), utc=True, errors="coerce").dropna()
    span_h = float((closes.max() - closes.min()).total_seconds() / 3600.0 + 0.25) if len(closes) >= 2 else np.nan
    pnl_per_24h = pnl * 24.0 / span_h if np.isfinite(span_h) and span_h > 0 and np.isfinite(pnl) else np.nan

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "research_stage": "HOLDOUT_VALIDATION_DEEP_TAIL_V1",
        "source_session": str(source),
        "hard_bound": bool(hard_bind),
        "frozen_before_validation": True,
        "development_source_session": DEV_SOURCE_SESSION,
        "development_q5_rounding_bound_pnl": DEV_Q5_ROUNDING_BOUND_PNL,
        "development_q5_exit_coverage": DEV_Q5_EXIT_COVERAGE,
        "frozen_spec": {
            "entry_c": FROZEN_ENTRY_C,
            "qty_per_tail": FROZEN_QTY,
            "sides": ["BUY_YES", "BUY_NO"],
            "posting_window": "M1_TO_M5",
            "activation_latency_ms": V3.ACTIVATION_LATENCY_MS,
            "entry_fill_rule": "strict trade-through seller flow; exact-price flow excluded; partial fills allowed",
            "exit_rule": "M5 consume recorded top-3 outcome bids L1->L2->L3; deeper residual valued zero",
            "asset_filters": None,
            "pillar_filter": None,
            "volatility_filter": None,
            "repricing": False,
        },
        "validation_gate_predeclared": {
            "minimum_fill_events": MIN_FILL_EVENTS_FOR_DECISION,
            "net_pnl_rounding_bound_must_be_positive": True,
            "minimum_m5_exit_fraction_of_filled_qty": MIN_M5_EXIT_COVERAGE,
        },
        "validation_status": status,
        "metadata_tickers": int(len(meta)),
        "observed_series": observed_series,
        "trade_clock_stats": trade_clock_stats,
        "fee_provenance": fee_provenance,
        "fee_artifact": str(fee_path),
        "entry_fill_events": fill_events,
        "entry_filled_qty": fill_qty,
        "m5_exit_qty": float(row.get("m5_exit_qty", 0.0)),
        "m5_exit_fraction_of_filled_qty": exit_cov,
        "residual_qty_zero_valued": float(row.get("residual_qty_zero_valued", 0.0)),
        "m5_taker_fees": float(row.get("m5_taker_fees", 0.0)),
        "m5_cross_spread_cost_vs_mid_embedded": float(row.get("m5_cross_spread_cost_vs_mid_embedded", 0.0)),
        "m5_top3_slippage_vs_best_bid_embedded": float(row.get("m5_top3_slippage_vs_best_bid_embedded", 0.0)),
        "balance_rounding_upper_bound_drag": float(row.get("balance_rounding_upper_bound_drag", 0.0)),
        "net_pnl_before_rounding_bound": float(row.get("net_pnl_before_rounding_bound", np.nan)),
        "net_pnl_rounding_bound": pnl,
        "max_drawdown_rounding_bound": float(row.get("max_drawdown_rounding_bound", np.nan)),
        "validation_span_hours_from_close_times": span_h,
        "normalized_net_pnl_per_24h_diagnostic": pnl_per_24h,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "guardrail": (
            "This output evaluates one frozen Q5/5c/M5-only specification. Do not change entry, size, exit, asset universe, or filters based on validation results."
        ),
    }

    out = _new_output(source.name)
    summary["output_dir"] = str(out)
    _atomic_json(out / "summary.json", summary)
    detail.to_csv(out / "q5_validation_detail.csv", index=False)
    curve.to_csv(out / "q5_validation_curve.csv", index=False)
    asset.to_csv(out / "q5_validation_by_asset.csv", index=False)
    windows.to_csv(out / "q5_validation_by_window.csv", index=False)

    if show:
        print("=" * 150)
        print("Q5 VALIDATION RESULT")
        print("=" * 150)
        print("Status:", status)
        print("Fill events:", fill_events)
        print("Entry filled qty:", f"{fill_qty:.2f}")
        print("M5 exit qty:", f"{float(row.get('m5_exit_qty', 0.0)):.2f}")
        print("M5 exit coverage:", f"{100.0 * exit_cov:.2f}%" if np.isfinite(exit_cov) else "NA")
        print("Residual beyond top-3 (zero-valued):", f"{float(row.get('residual_qty_zero_valued', 0.0)):.2f}")
        print("M5 taker fees:", f"${float(row.get('m5_taker_fees', 0.0)):.4f}")
        print("Embedded spread-cross cost vs mid:", f"${float(row.get('m5_cross_spread_cost_vs_mid_embedded', 0.0)):.4f}")
        print("Embedded top-3 slippage vs best bid:", f"${float(row.get('m5_top3_slippage_vs_best_bid_embedded', 0.0)):.4f}")
        print("Rounding upper-bound drag:", f"${float(row.get('balance_rounding_upper_bound_drag', 0.0)):.4f}")
        print("Net before rounding bound:", f"${float(row.get('net_pnl_before_rounding_bound', np.nan)):+.4f}")
        print("PRIMARY rounding-bound net:", f"${pnl:+.4f}")
        print("Max drawdown:", f"${float(row.get('max_drawdown_rounding_bound', np.nan)):+.4f}")
        if np.isfinite(span_h):
            print("Observed close-time span:", f"{span_h:.3f} h")
            print("Normalized /24h diagnostic:", f"${pnl_per_24h:+.4f}")
        print()
        print("PREDECLARED GATE")
        print("  fills >=", MIN_FILL_EVENTS_FOR_DECISION, "->", fill_events >= MIN_FILL_EVENTS_FOR_DECISION)
        print("  net > 0 ->", bool(np.isfinite(pnl) and pnl > 0))
        print("  M5 exit coverage >= 95% ->", bool(np.isfinite(exit_cov) and exit_cov >= MIN_M5_EXIT_COVERAGE))
        print()
        print("Fee provenance:", fee_provenance)
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")
        print("=" * 150)

    return {
        "summary": summary,
        "curve": curve,
        "detail": detail,
        "by_asset": asset,
        "by_window": windows,
        "output_dir": out,
    }
