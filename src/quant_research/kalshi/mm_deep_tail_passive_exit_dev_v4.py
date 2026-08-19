from __future__ import annotations

"""Development-only passive-exit study for the frozen 5c/Q5 deep-tail entry.

This script is intentionally hard-bound to the already-inspected 24h DEVELOPMENT
sample.  It must NOT be pointed at the ~15h validation sample, because the passive-exit
hypothesis was proposed only after that validation result had already been observed.

Entry is held fixed:
- rest BUY YES Q5 @ 5c and BUY NO Q5 @ 5c from M1;
- activation M1 + 100ms;
- strict trade-through entry capacity only; exact 5c prints excluded;
- all 9 recorded crypto series; no filters.

After a FULL Q5 entry fill is locally observable, compare three fixed passive exit rules:
1) JOIN_ASK: sell the outcome at the contemporaneous outcome best ask, behind displayed L1.
2) IMPROVE_1C: if spread >=2c, sell 1c inside the ask (new best ask, queue ahead=0),
   otherwise join the ask.
3) TIGHTEN_TO_1C: if spread >=2c, sell at outcome bid+1c (collapse spread to 1c,
   queue ahead=0), otherwise join the ask.

The chosen exit quote is fixed after placement: no repricing, pegging, or cancellation.
Passive exits use public trade flow causally:
- trade-through beyond our sell quote fills all remaining quantity;
- exact-price aggressive buy flow first burns queue ahead, then fills us;
- inside-spread quotes start with zero displayed queue ahead.

Any unfilled residual at M5 is liquidated against recorded top-3 outcome bids L1->L3,
with the stored quadratic taker fee and the same $0.0099/cross conservative rounding drag.
Passive entry/exit maker fees are $0 under the stored historical quadratic fee assumption.
Spread crossing and book-walking are embedded in the actual M5 execution prices and are
reported diagnostically, not double-subtracted.

Partial entry fills that never reach the full frozen Q5 are NOT given a newly invented
passive-exit rule in this development experiment; they remain M5-only.  On the hard-bound
24h development sample, the prior Q5 capacity study found every qualifying Q5 entry fully
filled, so this does not alter the observed development comparison.

NO API calls.  NO orders.  Source files read-only.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_capacity_dev_v3 as V3

VERSION = "MM_DEEP_TAIL_PASSIVE_EXIT_DEV_V4"
HARD_BOUND_SESSION = "20260817_064143"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_passive_exit_dev_v4"

ENTRY_C = 5
ENTRY = ENTRY_C / 100.0
QTY = 5.0
M1_S = 60.0
M5_S = 300.0
WINDOW_S = M5_S - M1_S
ACTIVATION_LATENCY_MS = 100.0
BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS = 0.0099
MIN_BOOK_COVERAGE_FRAC = 0.95
EPS = 1e-10

VARIANTS = ("M5_ONLY", "JOIN_ASK", "IMPROVE_1C", "TIGHTEN_TO_1C")


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _outcome_price(tail: str, yes_price: float) -> float:
    return float(yes_price) if tail == "YES" else 1.0 - float(yes_price)


def _entry_seller_side(tail: str) -> str:
    return "ask" if tail == "YES" else "bid"


def _exit_buyer_side(tail: str) -> str:
    return "bid" if tail == "YES" else "ask"


def _entry_fill_q5(trades: list[dict], tail: str, active_s: float) -> dict:
    seller = _entry_seller_side(tail)
    remaining = QTY
    filled = 0.0
    first_exec = np.nan
    last_exec = np.nan
    last_obs = np.nan
    exact_qty_excluded = 0.0
    exact_rows_excluded = 0

    for tr in trades:
        if tr["receipt_s"] + EPS < active_s or tr["exec_s"] + EPS < active_s:
            continue
        if tr["taker_book_side"] != seller:
            continue
        opx = _outcome_price(tail, tr["yes_price"])
        if abs(opx - ENTRY) <= EPS:
            exact_qty_excluded += float(tr["qty"])
            exact_rows_excluded += 1
            continue
        if opx >= ENTRY - EPS:
            continue

        take = min(remaining, float(tr["qty"]))
        if take <= EPS:
            continue
        if not np.isfinite(first_exec):
            first_exec = float(tr["exec_s"])
        filled += take
        remaining -= take
        last_exec = float(tr["exec_s"])
        last_obs = float(max(tr["exec_s"], tr["receipt_s"]))
        if remaining <= EPS:
            break

    return {
        "entry_filled_qty": float(filled),
        "entry_full_q5": bool(filled >= QTY - EPS),
        "first_entry_exec_s": first_exec,
        "full_entry_exec_s": last_exec if filled >= QTY - EPS else np.nan,
        "full_entry_observed_s": last_obs if filled >= QTY - EPS else np.nan,
        "exact_entry_rows_excluded": int(exact_rows_excluded),
        "exact_entry_qty_excluded": float(exact_qty_excluded),
    }


def _snapshot_from_cur(cur: dict, receipt_s: float, elapsed_s: float) -> dict:
    return {
        "receipt_s": float(receipt_s),
        "elapsed_s": float(elapsed_s),
        "yes_bid": float(cur["bid"]),
        "yes_ask": float(cur["ask"]),
        "yes_bid_q1": float(cur["bid_q1"]),
        "yes_ask_q1": float(cur["ask_q1"]),
        "yes_mid": float(cur["mid"]),
        "bid_levels": [(float(p), float(q)) for p, q in cur["bid_levels"]],
        "ask_levels": [(float(p), float(q)) for p, q in cur["ask_levels"]],
    }


def _scan_books(source: Path, meta: dict, positions: dict, *, show=True):
    """One pass: latest actionable BBO at fill observation + true M5 top-3 + coverage."""
    pos_by_ticker = defaultdict(list)
    for key, p in positions.items():
        if p.get("entry_full_q5") and np.isfinite(_f(p.get("full_entry_observed_s"))):
            pos_by_ticker[key[0]].append(key)

    decision_snapshot = {}
    coverage = defaultdict(float)
    last = {}
    m5 = {}
    finalized = set()

    def apply_interval(ticker: str, state: dict, end_elapsed: float):
        if state is None:
            return
        start = max(M1_S, float(state["elapsed_s"]))
        end = min(M5_S, float(end_elapsed))
        dt = max(0.0, end - start)
        if dt > 0:
            coverage[ticker] += dt

    for n, r in enumerate(_iter_jsonl(source / "book_top3_events.jsonl"), start=1):
        ticker = str(r.get("ticker") or "")
        if ticker not in meta or ticker in finalized:
            continue
        e = _f(r.get("elapsed_s"))
        rt = OOS._ts(r.get("receipt_time"))
        cur = OOS._top_state(r)
        if not (np.isfinite(e) and np.isfinite(rt) and cur is not None):
            continue

        prev = last.get(ticker)
        if prev is not None:
            apply_interval(ticker, prev, e)

        # Keep the latest valid local BBO known no later than fill observation.
        for key in pos_by_ticker.get(ticker, []):
            d = float(positions[key]["full_entry_observed_s"])
            if rt <= d + EPS:
                decision_snapshot[key] = _snapshot_from_cur(cur, rt, e)

        if e >= M5_S:
            if prev is not None:
                m5[ticker] = dict(prev)
                m5[ticker]["true_m5_finalized"] = True
            finalized.add(ticker)
            last.pop(ticker, None)
        else:
            last[ticker] = _snapshot_from_cur(cur, rt, e)

        if show and n % 1_000_000 == 0:
            print(
                f"book scan: {n:,} rows | decision snaps={len(decision_snapshot):,} | "
                f"true-M5={len(m5):,}"
            )

    out = {}
    for ticker in meta:
        cov = min(WINDOW_S, float(coverage.get(ticker, 0.0)))
        snap = m5.get(ticker)
        out[ticker] = {
            "book_covered_seconds": cov,
            "book_coverage_fraction": cov / WINDOW_S,
            "coverage_eligible": bool(
                cov >= MIN_BOOK_COVERAGE_FRAC * WINDOW_S - EPS
                and snap is not None
                and bool(snap.get("true_m5_finalized"))
            ),
            "m5": snap,
        }
    return decision_snapshot, out


def _outcome_bbo(tail: str, snap: dict):
    if tail == "YES":
        return {
            "bid": float(snap["yes_bid"]),
            "ask": float(snap["yes_ask"]),
            "ask_queue": max(0.0, float(snap["yes_ask_q1"])),
            "mid": float(snap["yes_mid"]),
        }
    return {
        "bid": 1.0 - float(snap["yes_ask"]),
        "ask": 1.0 - float(snap["yes_bid"]),
        "ask_queue": max(0.0, float(snap["yes_bid_q1"])),
        "mid": 1.0 - float(snap["yes_mid"]),
    }


def _quote_for_variant(variant: str, tail: str, snap: dict):
    bbo = _outcome_bbo(tail, snap)
    bid = float(bbo["bid"])
    ask = float(bbo["ask"])
    spread = ask - bid

    if variant == "JOIN_ASK":
        px = ask
        queue = float(bbo["ask_queue"])
        mode = "JOIN_VISIBLE_ASK"
    elif variant == "IMPROVE_1C" and spread >= 0.02 - EPS:
        px = ask - 0.01
        queue = 0.0
        mode = "NEW_BEST_ASK_MINUS_1C"
    elif variant == "TIGHTEN_TO_1C" and spread >= 0.02 - EPS:
        px = bid + 0.01
        queue = 0.0
        mode = "NEW_BEST_ASK_BID_PLUS_1C"
    else:
        px = ask
        queue = float(bbo["ask_queue"])
        mode = "JOIN_VISIBLE_ASK_FALLBACK"

    # Defensive passive-only guard.
    if px <= bid + EPS:
        px = ask
        queue = float(bbo["ask_queue"])
        mode = "JOIN_VISIBLE_ASK_PASSIVE_GUARD"

    return {
        "quote_price": float(px),
        "queue_ahead_initial": float(queue),
        "outcome_bid_at_decision": bid,
        "outcome_ask_at_decision": ask,
        "outcome_mid_at_decision": float(bbo["mid"]),
        "spread_c_at_decision": 100.0 * spread,
        "ask_improvement_c": 100.0 * (ask - px),
        "quote_mode": mode,
    }


def _simulate_passive_exit(trades: list[dict], tail: str, quote: dict, active_s: float, qty: float):
    buyer = _exit_buyer_side(tail)
    qpx = float(quote["quote_price"])
    queue = float(quote["queue_ahead_initial"])
    remaining = float(qty)
    filled = 0.0
    exact_qty = 0.0
    trade_through = False
    first_fill_exec = np.nan
    full_fill_exec = np.nan

    for tr in trades:
        if tr["exec_s"] + EPS < active_s or tr["receipt_s"] + EPS < active_s:
            continue
        if tr["taker_book_side"] != buyer:
            continue
        opx = _outcome_price(tail, tr["yes_price"])

        if opx > qpx + EPS:
            # Trade beyond our active sell quote proves all remaining liquidity at our
            # better price was consumed before the market traded higher.
            if not np.isfinite(first_fill_exec):
                first_fill_exec = float(tr["exec_s"])
            filled += remaining
            remaining = 0.0
            trade_through = True
            full_fill_exec = float(tr["exec_s"])
            break

        if abs(opx - qpx) <= EPS:
            trade_qty = float(tr["qty"])
            exact_qty += trade_qty
            burn = min(queue, trade_qty)
            queue -= burn
            available = max(0.0, trade_qty - burn)
            take = min(remaining, available)
            if take > EPS:
                if not np.isfinite(first_fill_exec):
                    first_fill_exec = float(tr["exec_s"])
                filled += take
                remaining -= take
                if remaining <= EPS:
                    remaining = 0.0
                    full_fill_exec = float(tr["exec_s"])
                    break

    return {
        "passive_exit_qty": float(filled),
        "passive_exit_residual_qty": float(remaining),
        "passive_exit_full": bool(remaining <= EPS),
        "queue_ahead_remaining": float(queue),
        "exact_price_buyer_qty_seen": float(exact_qty),
        "trade_through_exit": bool(trade_through),
        "first_passive_exit_exec_s": first_fill_exec,
        "full_passive_exit_exec_s": full_fill_exec,
    }


def _m5_snapshot_for_v3(snap: dict):
    if not snap:
        return None
    return {
        "yes_bid": float(snap["yes_bid"]),
        "yes_ask": float(snap["yes_ask"]),
        "yes_mid": float(snap["yes_mid"]),
        "bid_levels": list(snap["bid_levels"]),
        "ask_levels": list(snap["ask_levels"]),
        "snapshot_elapsed_s": float(snap["elapsed_s"]),
        "true_m5_finalized": bool(snap.get("true_m5_finalized", False)),
    }


def _evaluate(meta: dict, trades: dict, positions: dict, decision_snaps: dict, books: dict, fee_mult: dict):
    rows = []

    for (ticker, tail), p in sorted(positions.items()):
        m = meta[ticker]
        bd = books[ticker]
        mult = _f(fee_mult.get(str(m.get("series") or "")))
        eligible = bool(bd.get("coverage_eligible") and np.isfinite(mult) and mult > 0)
        entry_qty = float(p["entry_filled_qty"]) if eligible else 0.0
        full_q5 = bool(p["entry_full_q5"] and eligible)
        m5snap = _m5_snapshot_for_v3(bd.get("m5"))

        # Baseline M5-only economics for the actually filled quantity.
        if entry_qty > EPS:
            base_m5 = V3._consume_m5_depth(tail, entry_qty, m5snap, mult)
        else:
            base_m5 = {
                "exit_qty": 0.0,
                "residual_qty_zero_valued": 0.0,
                "m5_exit_proceeds": 0.0,
                "m5_taker_fee": 0.0,
                "m5_slippage_vs_best_bid": 0.0,
                "m5_cross_cost_vs_mid": 0.0,
            }
        base_round = BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS if base_m5["exit_qty"] > EPS else 0.0
        base_net = (
            float(base_m5["m5_exit_proceeds"])
            - ENTRY * entry_qty
            - float(base_m5["m5_taker_fee"])
            - base_round
        )

        base_common = {
            "ticker": ticker,
            "series": str(m.get("series") or ""),
            "close_time": str(m.get("close_time") or ""),
            "tail": tail,
            "entry_c": ENTRY_C,
            "requested_qty": QTY,
            "coverage_eligible": eligible,
            "entry_filled_qty": entry_qty,
            "entry_full_q5": full_q5,
            "first_entry_exec_s": p.get("first_entry_exec_s"),
            "full_entry_exec_s": p.get("full_entry_exec_s"),
            "full_entry_observed_s": p.get("full_entry_observed_s"),
            "exact_entry_qty_excluded": p.get("exact_entry_qty_excluded"),
        }

        rows.append({
            **base_common,
            "variant": "M5_ONLY",
            "decision_snapshot_available": False,
            "quote_price": np.nan,
            "queue_ahead_initial": np.nan,
            "spread_c_at_decision": np.nan,
            "ask_improvement_c": np.nan,
            "quote_mode": "NONE",
            "passive_exit_qty": 0.0,
            "passive_exit_full": False,
            "passive_exit_proceeds": 0.0,
            "passive_exit_seconds_to_full": np.nan,
            "m5_exit_qty": float(base_m5["exit_qty"]),
            "m5_residual_zero_valued": float(base_m5["residual_qty_zero_valued"]),
            "m5_taker_fee": float(base_m5["m5_taker_fee"]),
            "m5_cross_cost_vs_mid_embedded": float(base_m5.get("m5_cross_cost_vs_mid", 0.0) or 0.0),
            "m5_slippage_vs_best_bid_embedded": float(base_m5.get("m5_slippage_vs_best_bid", 0.0) or 0.0),
            "rounding_drag": float(base_round),
            "net_pnl_rounding_bound": float(base_net),
            "incremental_vs_m5_only": 0.0,
        })

        for variant in ("JOIN_ASK", "IMPROVE_1C", "TIGHTEN_TO_1C"):
            snap = decision_snaps.get((ticker, tail))
            if not (full_q5 and snap is not None):
                # Partial entry or missing decision BBO: retain baseline M5 handling.
                rows.append({
                    **base_common,
                    "variant": variant,
                    "decision_snapshot_available": bool(snap is not None),
                    "quote_price": np.nan,
                    "queue_ahead_initial": np.nan,
                    "spread_c_at_decision": np.nan,
                    "ask_improvement_c": np.nan,
                    "quote_mode": "NO_PASSIVE_EXIT_PARTIAL_OR_MISSING_SNAPSHOT",
                    "passive_exit_qty": 0.0,
                    "passive_exit_full": False,
                    "passive_exit_proceeds": 0.0,
                    "passive_exit_seconds_to_full": np.nan,
                    "m5_exit_qty": float(base_m5["exit_qty"]),
                    "m5_residual_zero_valued": float(base_m5["residual_qty_zero_valued"]),
                    "m5_taker_fee": float(base_m5["m5_taker_fee"]),
                    "m5_cross_cost_vs_mid_embedded": float(base_m5.get("m5_cross_cost_vs_mid", 0.0) or 0.0),
                    "m5_slippage_vs_best_bid_embedded": float(base_m5.get("m5_slippage_vs_best_bid", 0.0) or 0.0),
                    "rounding_drag": float(base_round),
                    "net_pnl_rounding_bound": float(base_net),
                    "incremental_vs_m5_only": 0.0,
                })
                continue

            quote = _quote_for_variant(variant, tail, snap)
            exit_active_s = float(p["full_entry_observed_s"]) + ACTIVATION_LATENCY_MS / 1000.0
            ex_passive = _simulate_passive_exit(
                trades.get(ticker, []), tail, quote, exit_active_s, entry_qty
            )
            passive_qty = float(ex_passive["passive_exit_qty"])
            passive_proceeds = passive_qty * float(quote["quote_price"])
            residual = float(ex_passive["passive_exit_residual_qty"])

            if residual > EPS:
                ex_m5 = V3._consume_m5_depth(tail, residual, m5snap, mult)
            else:
                ex_m5 = {
                    "exit_qty": 0.0,
                    "residual_qty_zero_valued": 0.0,
                    "m5_exit_proceeds": 0.0,
                    "m5_taker_fee": 0.0,
                    "m5_slippage_vs_best_bid": 0.0,
                    "m5_cross_cost_vs_mid": 0.0,
                }
            rounding = BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS if ex_m5["exit_qty"] > EPS else 0.0
            net = (
                passive_proceeds
                + float(ex_m5["m5_exit_proceeds"])
                - ENTRY * entry_qty
                - float(ex_m5["m5_taker_fee"])
                - rounding
            )
            seconds_to_full = (
                float(ex_passive["full_passive_exit_exec_s"] - p["full_entry_observed_s"])
                if ex_passive["passive_exit_full"] and np.isfinite(_f(ex_passive["full_passive_exit_exec_s"]))
                else np.nan
            )

            rows.append({
                **base_common,
                "variant": variant,
                "decision_snapshot_available": True,
                "quote_price": float(quote["quote_price"]),
                "queue_ahead_initial": float(quote["queue_ahead_initial"]),
                "spread_c_at_decision": float(quote["spread_c_at_decision"]),
                "ask_improvement_c": float(quote["ask_improvement_c"]),
                "quote_mode": quote["quote_mode"],
                "passive_exit_qty": passive_qty,
                "passive_exit_full": bool(ex_passive["passive_exit_full"]),
                "passive_exit_proceeds": float(passive_proceeds),
                "passive_exit_seconds_to_full": seconds_to_full,
                "m5_exit_qty": float(ex_m5["exit_qty"]),
                "m5_residual_zero_valued": float(ex_m5["residual_qty_zero_valued"]),
                "m5_taker_fee": float(ex_m5["m5_taker_fee"]),
                "m5_cross_cost_vs_mid_embedded": float(ex_m5.get("m5_cross_cost_vs_mid", 0.0) or 0.0),
                "m5_slippage_vs_best_bid_embedded": float(ex_m5.get("m5_slippage_vs_best_bid", 0.0) or 0.0),
                "rounding_drag": float(rounding),
                "net_pnl_rounding_bound": float(net),
                "incremental_vs_m5_only": float(net - base_net),
            })

    return pd.DataFrame(rows)


def _aggregate(detail: pd.DataFrame):
    rows = []
    for variant, g0 in detail.groupby("variant", sort=False):
        g = g0[g0["coverage_eligible"]].copy()
        fills = g[g["entry_filled_qty"] > EPS].copy()
        p = pd.to_numeric(g["net_pnl_rounding_bound"], errors="coerce").fillna(0.0)
        d = pd.to_numeric(g["incremental_vs_m5_only"], errors="coerce").fillna(0.0)
        passive = pd.to_numeric(fills["passive_exit_qty"], errors="coerce").fillna(0.0)
        full = fills["passive_exit_full"] if len(fills) else pd.Series(dtype=bool)
        secs = pd.to_numeric(fills["passive_exit_seconds_to_full"], errors="coerce").dropna()
        rows.append({
            "variant": variant,
            "eligible_positions": int(len(g)),
            "entry_fill_positions": int(len(fills)),
            "entry_filled_qty": float(pd.to_numeric(fills["entry_filled_qty"], errors="coerce").fillna(0).sum()),
            "passive_exit_qty": float(passive.sum()),
            "passive_exit_fraction_of_entry_qty": float(passive.sum() / pd.to_numeric(fills["entry_filled_qty"], errors="coerce").fillna(0).sum()) if len(fills) and pd.to_numeric(fills["entry_filled_qty"], errors="coerce").fillna(0).sum() > EPS else np.nan,
            "passive_exit_full_positions": int(full.sum()) if len(full) else 0,
            "passive_exit_full_rate": float(full.mean()) if len(full) else np.nan,
            "median_seconds_to_full_passive_exit": float(secs.median()) if len(secs) else np.nan,
            "m5_exit_qty": float(pd.to_numeric(fills["m5_exit_qty"], errors="coerce").fillna(0).sum()),
            "m5_residual_zero_valued": float(pd.to_numeric(fills["m5_residual_zero_valued"], errors="coerce").fillna(0).sum()),
            "m5_taker_fees": float(pd.to_numeric(fills["m5_taker_fee"], errors="coerce").fillna(0).sum()),
            "rounding_drag": float(pd.to_numeric(fills["rounding_drag"], errors="coerce").fillna(0).sum()),
            "total_net_pnl_rounding_bound": float(p.sum()),
            "incremental_pnl_vs_m5_only": float(d.sum()),
            "mean_incremental_pnl_per_filled_position": float(d.loc[fills.index].mean()) if len(fills) else np.nan,
            "positive_incremental_positions": int((d.loc[fills.index] > 0).sum()) if len(fills) else 0,
            "negative_incremental_positions": int((d.loc[fills.index] < 0).sum()) if len(fills) else 0,
        })
    return pd.DataFrame(rows)


def _by_asset(detail: pd.DataFrame):
    q = detail[detail["coverage_eligible"]].copy()
    rows = []
    for (variant, series), g in q.groupby(["variant", "series"], sort=True):
        fills = g[g["entry_filled_qty"] > EPS]
        rows.append({
            "variant": variant,
            "series": series,
            "fill_positions": int(len(fills)),
            "entry_filled_qty": float(pd.to_numeric(fills["entry_filled_qty"], errors="coerce").fillna(0).sum()),
            "passive_exit_qty": float(pd.to_numeric(fills["passive_exit_qty"], errors="coerce").fillna(0).sum()),
            "net_pnl_rounding_bound": float(pd.to_numeric(g["net_pnl_rounding_bound"], errors="coerce").fillna(0).sum()),
            "incremental_vs_m5_only": float(pd.to_numeric(g["incremental_vs_m5_only"], errors="coerce").fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def run_deep_tail_passive_exit_dev(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected 24h development source {HARD_BOUND_SESSION}, got {source.name}")
    if hard_bind and "mm_event_m0_m5_oos_cycle_q10_v1" not in str(source.parent):
        raise RuntimeError("This V4 study is development-only and hard-bound to the 24h formal capture root.")

    required = [
        source / "book_top3_events.jsonl",
        source / "trades_event_time.jsonl",
        source / "market_metadata.jsonl",
        source / "fee_preflight.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + " | ".join(missing))

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored development fee preflight is not PASS.")
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    meta = V1._metadata(source)
    if show:
        print("=" * 150)
        print("DEEP-TAIL PASSIVE EXIT DEV V4 — 5c/Q5 FIXED ENTRY")
        print("=" * 150)
        print("Source:", source)
        print("Variants:", VARIANTS)
        print("IMPORTANT: development-only. The 15h validation sample is NOT read here.")
        print()
        print("PASS 1/3 — loading M1-M5 public trades with frozen causal clock...")
    trades, trade_stats = V1._load_trades(source, meta, show=show)

    positions = {}
    for ticker, m in meta.items():
        active_s = float(m["window_start_s"]) + M1_S + ACTIVATION_LATENCY_MS / 1000.0
        tr = trades.get(ticker, [])
        for tail in ("YES", "NO"):
            positions[(ticker, tail)] = _entry_fill_q5(tr, tail, active_s)

    if show:
        full_n = sum(int(p["entry_full_q5"]) for p in positions.values())
        print(f"full Q5 entry positions found: {full_n:,}")
        print("PASS 2/3 — scanning decision-time BBOs and true M5 top-3...")
    decision_snaps, books = _scan_books(source, meta, positions, show=show)

    if show:
        print("PASS 3/3 — simulating fixed passive exit variants...")
    detail = _evaluate(meta, trades, positions, decision_snaps, books, fee_mult)
    summary_df = _aggregate(detail)
    by_asset = _by_asset(detail)

    out = _new_output(source.name)
    detail.to_csv(out / "passive_exit_detail.csv", index=False)
    summary_df.to_csv(out / "passive_exit_variant_summary.csv", index=False)
    by_asset.to_csv(out / "passive_exit_by_asset.csv", index=False)

    baseline = summary_df[summary_df["variant"] == "M5_ONLY"]
    base_net = float(baseline.iloc[0]["total_net_pnl_rounding_bound"]) if len(baseline) else np.nan
    ranked = summary_df.sort_values("total_net_pnl_rounding_bound", ascending=False)
    best = ranked.iloc[0].to_dict() if len(ranked) else {}

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "research_stage": "DEVELOPMENT_ONLY_NEW_EXIT_HYPOTHESIS",
        "entry_c": ENTRY_C,
        "qty": QTY,
        "entry_activation_latency_ms": ACTIVATION_LATENCY_MS,
        "exit_activation_latency_ms": ACTIVATION_LATENCY_MS,
        "variants": list(VARIANTS),
        "entry_full_q5_positions": int(sum(int(p["entry_full_q5"]) for p in positions.values())),
        "decision_snapshots": int(len(decision_snaps)),
        "trade_clock_stats": trade_stats,
        "baseline_m5_only_net": base_net,
        "best_development_variant": best,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "guardrail": (
            "The passive-exit hypothesis was proposed after the 15h Q5 validation result was observed. "
            "Therefore that 15h sample is NOT valid independent validation for any exit variant selected here."
        ),
    }
    OOS._atomic_json(out / "summary.json", summary)

    if show:
        print("=" * 150)
        print("PASSIVE EXIT VARIANT SUMMARY")
        print("=" * 150)
        cols = [
            "variant", "entry_fill_positions", "entry_filled_qty", "passive_exit_qty",
            "passive_exit_fraction_of_entry_qty", "passive_exit_full_positions",
            "passive_exit_full_rate", "median_seconds_to_full_passive_exit",
            "m5_exit_qty", "m5_residual_zero_valued", "m5_taker_fees",
            "total_net_pnl_rounding_bound", "incremental_pnl_vs_m5_only",
            "positive_incremental_positions", "negative_incremental_positions",
        ]
        print(summary_df[cols].to_string(index=False))
        print()
        print("Best development variant:")
        print(best)
        print()
        print("GUARDRAIL:")
        print(summary["guardrail"])
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "variant_summary": summary_df,
        "detail": detail,
        "by_asset": by_asset,
        "output_dir": str(out),
    }
