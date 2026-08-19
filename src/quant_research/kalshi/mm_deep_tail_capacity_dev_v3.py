from __future__ import annotations

"""Development-only capacity study for the 5c deep-tail / M5-only candidate.

Scientific purpose
------------------
V2 found a broad positive extreme-tail region on the already-inspected 24h development
sample and identified 5c / M5-only as a deliberately non-optimal, higher-count candidate.
This module does NOT retune the entry price or exit rule.  It freezes those mechanics for
capacity exploration and varies only requested quantity.

Candidate held fixed here
-------------------------
- Universe: all recorded 15m crypto series in the hard-bound 24h development capture.
- From M1: rest BUY YES @ 5c and BUY NO @ 5c.
- Activation: M1 + 100ms, matching the prior deep-tail development studies.
- Entry fee: $0 maker fee under the stored PASS quadratic fee preflight.
- Never reprice/cancel before M5.
- M5: cross the executable book to exit.

Capacity model
--------------
Deep FIFO at 5c is not fully observable because the recorder persisted only top 3 levels.
For capacity we therefore use a stricter flow-capped rule than the original Q1 feasibility
study:

    confirmed_entry_capacity = cumulative SAME-OUTCOME aggressive seller quantity that
                               trades STRICTLY THROUGH 5c while our order is active.

Exact-price 5c trades are excluded from confirmed capacity because queue ahead is unknown.
Requested quantity may therefore fill partially.  This is intentionally conservative.

M5 execution is depth-aware.  We consume the recorded top-3 outcome bids level by level.
For YES positions this is the recorded YES bid ladder.  For NO positions it is the YES ask
ladder converted to NO bids (NO bid = 1 - YES ask).  Any position beyond recorded top-3
M5 depth receives ZERO terminal value in the primary capacity PnL.  This prevents us from
inventing deeper executable liquidity.

Costs
-----
- Entry maker fee: $0 under stored historical fee preflight.
- M5 exit: actual top-3 bid prices, so crossing the spread and walking the book are already
  embedded in proceeds.
- M5 taker fee: stored quadratic fee, applied conservatively per consumed book level.
- Balance rounding: additional $0.0099 upper-bound drag once per non-zero M5 cross.
- Spread-cross cost vs contemporaneous outcome midpoint and top-3 slippage vs best bid are
  reported as diagnostics only; they are NOT subtracted twice.

Quantity grid
-------------
Q1, Q2, Q5, Q10, Q20, Q50, Q100, Q200, Q300, Q500, Q750, Q1000 per tail/order.

This remains DEVELOPMENT ONLY.  Quantity chosen here must be frozen before opening the
separate validation sample.  No API calls.  No orders.  Source capture is read-only.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1

VERSION = "MM_DEEP_TAIL_CAPACITY_DEV_V3"
HARD_BOUND_SESSION = V1.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_capacity_dev_v3"

ENTRY_C = 5
ENTRY = ENTRY_C / 100.0
REQUESTED_QTYS = (1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000)
M1_S = V1.M1_S
M5_S = V1.M5_S
WINDOW_S = V1.WINDOW_S
ACTIVATION_LATENCY_MS = V1.ACTIVATION_LATENCY_MS
MIN_BOOK_COVERAGE_FRAC = V1.MIN_BOOK_COVERAGE_FRAC
BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS = V1.BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS
EPS = 1e-10


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict):
                yield r


def _outcome_price(tail: str, yes_price: float) -> float:
    return float(yes_price) if tail == "YES" else 1.0 - float(yes_price)


def _entry_seller_side(tail: str) -> str:
    return "ask" if tail == "YES" else "bid"


def _scan_m5_books(source: Path, meta: dict, *, show=True):
    """Single receipt-clock pass for M1-M5 coverage and last top-3 state before true M5."""
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
        cur = OOS._top_state(r)
        if not np.isfinite(e) or cur is None:
            continue

        prev = last.get(ticker)
        if prev is not None:
            apply_interval(ticker, prev, e)

        if e >= M5_S:
            if prev is not None:
                m5[ticker] = {
                    "yes_bid": float(prev["bid"]),
                    "yes_ask": float(prev["ask"]),
                    "yes_mid": float(prev["mid"]),
                    "bid_levels": [(float(p), float(q)) for p, q in prev["bid_levels"]],
                    "ask_levels": [(float(p), float(q)) for p, q in prev["ask_levels"]],
                    "snapshot_elapsed_s": float(prev["elapsed_s"]),
                    "true_m5_finalized": True,
                }
            finalized.add(ticker)
            last.pop(ticker, None)
        else:
            last[ticker] = {
                "elapsed_s": float(e),
                "bid": float(cur["bid"]),
                "ask": float(cur["ask"]),
                "mid": float(cur["mid"]),
                "bid_levels": list(cur["bid_levels"]),
                "ask_levels": list(cur["ask_levels"]),
            }

        if show and n % 1_000_000 == 0:
            print(
                f"book capacity scan: {n:,} rows | active={len(last):,} | "
                f"true-M5 finalized={len(finalized):,}"
            )

    # Do NOT manufacture a true M5 snapshot for terminal partial markets.
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
    return out


def _confirmed_entry_flow(trades: list[dict], tail: str, active_s: float):
    """Strict-through seller flow only; exact-price flow is diagnostic and excluded."""
    seller = _entry_seller_side(tail)
    through = []
    exact_qty = 0.0
    exact_rows = 0

    for tr in trades:
        if tr["receipt_s"] + EPS < active_s or tr["exec_s"] + EPS < active_s:
            continue
        if tr["taker_book_side"] != seller:
            continue
        opx = _outcome_price(tail, tr["yes_price"])
        if abs(opx - ENTRY) <= EPS:
            exact_qty += float(tr["qty"])
            exact_rows += 1
        elif opx < ENTRY - EPS:
            through.append(tr)

    through_qty = float(sum(float(tr["qty"]) for tr in through))
    first_exec = float(through[0]["exec_s"]) if through else np.nan
    last_exec = float(through[-1]["exec_s"]) if through else np.nan
    return {
        "strict_through_rows": int(len(through)),
        "strict_through_qty": through_qty,
        "exact_entry_rows_excluded": int(exact_rows),
        "exact_entry_qty_excluded": float(exact_qty),
        "first_through_exec_s": first_exec,
        "last_through_exec_s": last_exec,
    }


def _outcome_exit_levels(tail: str, m5: dict):
    if not m5:
        return [], np.nan, np.nan

    if tail == "YES":
        levels = [(float(p), max(0.0, float(q))) for p, q in (m5.get("bid_levels") or [])]
        levels = [(p, q) for p, q in levels if np.isfinite(p) and np.isfinite(q) and q > EPS]
        levels.sort(key=lambda z: z[0], reverse=True)
        mid = _f(m5.get("yes_mid"))
    else:
        levels = [(1.0 - float(p), max(0.0, float(q))) for p, q in (m5.get("ask_levels") or [])]
        levels = [(p, q) for p, q in levels if np.isfinite(p) and np.isfinite(q) and q > EPS]
        levels.sort(key=lambda z: z[0], reverse=True)
        ym = _f(m5.get("yes_mid"))
        mid = 1.0 - ym if np.isfinite(ym) else np.nan

    best = levels[0][0] if levels else np.nan
    return levels, float(best) if np.isfinite(best) else np.nan, float(mid) if np.isfinite(mid) else np.nan


def _consume_m5_depth(tail: str, qty: float, m5: dict, multiplier: float):
    levels, best_bid, outcome_mid = _outcome_exit_levels(tail, m5)
    rem = max(0.0, float(qty))
    fills = []
    fee = 0.0

    for level_idx, (px, avail) in enumerate(levels, start=1):
        if rem <= EPS:
            break
        take = min(rem, float(avail))
        if take <= EPS:
            continue
        level_fee = OOS._quadratic_taker_fee(take, px, multiplier)
        fills.append({
            "level": int(level_idx),
            "price": float(px),
            "qty": float(take),
            "fee": float(level_fee),
        })
        fee += float(level_fee)
        rem -= take

    exited = float(sum(x["qty"] for x in fills))
    proceeds = float(sum(x["qty"] * x["price"] for x in fills))
    avg_px = proceeds / exited if exited > EPS else np.nan
    top3_depth = float(sum(q for _, q in levels))
    slippage_vs_best = (
        float(best_bid * exited - proceeds)
        if exited > EPS and np.isfinite(best_bid)
        else 0.0
    )
    cross_cost_vs_mid = (
        float(outcome_mid * exited - proceeds)
        if exited > EPS and np.isfinite(outcome_mid)
        else np.nan
    )

    return {
        "exit_qty": exited,
        "residual_qty_zero_valued": max(0.0, float(qty) - exited),
        "m5_top3_depth": top3_depth,
        "m5_best_outcome_bid": best_bid,
        "m5_outcome_mid": outcome_mid,
        "m5_avg_exit_price": avg_px,
        "m5_exit_proceeds": proceeds,
        "m5_taker_fee": float(fee),
        "m5_levels_consumed": int(len(fills)),
        "m5_slippage_vs_best_bid": max(0.0, slippage_vs_best),
        "m5_cross_cost_vs_mid": (
            max(0.0, cross_cost_vs_mid) if np.isfinite(cross_cost_vs_mid) else np.nan
        ),
        "m5_level_fills": fills,
    }


def _max_drawdown(detail: pd.DataFrame) -> float:
    if detail.empty:
        return np.nan
    q = detail[detail["entry_filled_qty"] > EPS].copy()
    if q.empty:
        return 0.0
    q = q.sort_values(["first_through_exec_s", "ticker", "tail"])
    pnl = pd.to_numeric(q["net_pnl_rounding_bound"], errors="coerce").fillna(0.0).to_numpy(float)
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    dd = np.r_[0.0, eq] - peak
    return float(dd.min())


def _evaluate(source: Path, meta: dict, trades: dict, books: dict, fee_mult: dict, *, show=True):
    rows = []
    tickers = sorted(meta)

    for j, ticker in enumerate(tickers, start=1):
        m = meta[ticker]
        bd = books[ticker]
        tr = trades.get(ticker, [])
        active_s = float(m["window_start_s"]) + M1_S + ACTIVATION_LATENCY_MS / 1000.0

        for tail in ("YES", "NO"):
            flow = _confirmed_entry_flow(tr, tail, active_s)
            mult = _f(fee_mult.get(str(m.get("series") or "")))

            for requested_qty in REQUESTED_QTYS:
                requested_qty = float(requested_qty)
                eligible = bool(bd["coverage_eligible"] and np.isfinite(mult) and mult > 0)
                entry_capacity = float(flow["strict_through_qty"]) if eligible else 0.0
                entry_fill = min(requested_qty, entry_capacity) if eligible else 0.0
                full_entry = bool(entry_fill >= requested_qty - EPS) if eligible else False
                partial_entry = bool(EPS < entry_fill < requested_qty - EPS) if eligible else False

                if entry_fill > EPS:
                    ex = _consume_m5_depth(tail, entry_fill, bd.get("m5"), mult)
                else:
                    ex = {
                        "exit_qty": 0.0,
                        "residual_qty_zero_valued": 0.0,
                        "m5_top3_depth": 0.0,
                        "m5_best_outcome_bid": np.nan,
                        "m5_outcome_mid": np.nan,
                        "m5_avg_exit_price": np.nan,
                        "m5_exit_proceeds": 0.0,
                        "m5_taker_fee": 0.0,
                        "m5_levels_consumed": 0,
                        "m5_slippage_vs_best_bid": 0.0,
                        "m5_cross_cost_vs_mid": np.nan,
                        "m5_level_fills": [],
                    }

                entry_cost = ENTRY * entry_fill
                maker_fee = 0.0
                rounding_drag = (
                    BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS if ex["exit_qty"] > EPS else 0.0
                )
                net_before_rounding = ex["m5_exit_proceeds"] - entry_cost - maker_fee - ex["m5_taker_fee"]
                net_round = net_before_rounding - rounding_drag
                full_exit = bool(ex["exit_qty"] >= entry_fill - EPS) if entry_fill > EPS else True

                rows.append({
                    "ticker": ticker,
                    "series": str(m.get("series") or ""),
                    "close_time": str(m.get("close_time") or ""),
                    "tail": tail,
                    "entry_c": ENTRY_C,
                    "requested_qty": requested_qty,
                    "coverage_eligible": eligible,
                    "book_coverage_fraction": float(bd["book_coverage_fraction"]),
                    **flow,
                    "entry_confirmed_capacity_qty": entry_capacity,
                    "entry_filled_qty": float(entry_fill),
                    "entry_fill_fraction_of_request": float(entry_fill / requested_qty) if requested_qty > 0 else np.nan,
                    "full_entry_fill": full_entry,
                    "partial_entry_fill": partial_entry,
                    "entry_price": ENTRY,
                    "entry_cost": float(entry_cost),
                    "entry_maker_fee": maker_fee,
                    **{k: v for k, v in ex.items() if k != "m5_level_fills"},
                    "m5_level_fills_json": json.dumps(ex["m5_level_fills"], separators=(",", ":")),
                    "full_m5_exit_within_top3": full_exit,
                    "balance_rounding_upper_bound_drag": float(rounding_drag),
                    "net_pnl_before_rounding_bound": float(net_before_rounding),
                    "net_pnl_rounding_bound": float(net_round),
                })

        if show and j % 100 == 0:
            print(f"capacity evaluated {j:,}/{len(tickers):,} tickers | rows={len(rows):,}")

    return pd.DataFrame(rows)


def _aggregate_curve(detail: pd.DataFrame):
    rows = []
    for requested_qty, g0 in detail.groupby("requested_qty", sort=True):
        g = g0[g0["coverage_eligible"]].copy()
        filled = g[pd.to_numeric(g["entry_filled_qty"], errors="coerce") > EPS].copy()

        total_requested = float(len(g) * requested_qty)
        entry_qty = float(pd.to_numeric(g["entry_filled_qty"], errors="coerce").fillna(0).sum())
        exit_qty = float(pd.to_numeric(g["exit_qty"], errors="coerce").fillna(0).sum())
        residual = float(pd.to_numeric(g["residual_qty_zero_valued"], errors="coerce").fillna(0).sum())
        pnl = float(pd.to_numeric(g["net_pnl_rounding_bound"], errors="coerce").fillna(0).sum())
        pnl_raw = float(pd.to_numeric(g["net_pnl_before_rounding_bound"], errors="coerce").fillna(0).sum())
        fees = float(pd.to_numeric(g["m5_taker_fee"], errors="coerce").fillna(0).sum())
        rounding = float(pd.to_numeric(g["balance_rounding_upper_bound_drag"], errors="coerce").fillna(0).sum())
        cross_cost = float(pd.to_numeric(g["m5_cross_cost_vs_mid"], errors="coerce").fillna(0).sum())
        slippage = float(pd.to_numeric(g["m5_slippage_vs_best_bid"], errors="coerce").fillna(0).sum())

        rows.append({
            "requested_qty": float(requested_qty),
            "eligible_posted_orders": int(len(g)),
            "total_requested_contracts": total_requested,
            "entry_fill_events": int(len(filled)),
            "full_entry_fill_orders": int(g["full_entry_fill"].sum()),
            "partial_entry_fill_orders": int(g["partial_entry_fill"].sum()),
            "entry_filled_qty": entry_qty,
            "entry_fill_fraction_of_requested_contracts": entry_qty / total_requested if total_requested > EPS else np.nan,
            "m5_exit_qty": exit_qty,
            "m5_exit_fraction_of_filled_qty": exit_qty / entry_qty if entry_qty > EPS else np.nan,
            "positions_fully_exited_top3": int(filled["full_m5_exit_within_top3"].sum()) if len(filled) else 0,
            "positions_with_m5_residual": int((filled["residual_qty_zero_valued"] > EPS).sum()) if len(filled) else 0,
            "residual_qty_zero_valued": residual,
            "m5_taker_fees": fees,
            "balance_rounding_upper_bound_drag": rounding,
            "m5_cross_spread_cost_vs_mid_embedded": cross_cost,
            "m5_top3_slippage_vs_best_bid_embedded": slippage,
            "net_pnl_before_rounding_bound": pnl_raw,
            "net_pnl_rounding_bound": pnl,
            "net_pnl_per_requested_contract": pnl / total_requested if total_requested > EPS else np.nan,
            "net_pnl_per_filled_contract": pnl / entry_qty if entry_qty > EPS else np.nan,
            "max_drawdown_rounding_bound": _max_drawdown(g),
        })

    out = pd.DataFrame(rows).sort_values("requested_qty").reset_index(drop=True)
    out["incremental_requested_qty"] = out["requested_qty"].diff()
    out["incremental_net_pnl"] = out["net_pnl_rounding_bound"].diff()
    out["marginal_pnl_per_incremental_requested_contract_per_order"] = (
        out["incremental_net_pnl"]
        / (
            out["incremental_requested_qty"]
            * out["eligible_posted_orders"].replace(0, np.nan)
        )
    )
    return out


def _aggregate_asset(detail: pd.DataFrame):
    q = detail[detail["coverage_eligible"]].copy()
    rows = []
    for (requested_qty, series), g in q.groupby(["requested_qty", "series"], sort=True):
        entry_qty = float(pd.to_numeric(g["entry_filled_qty"], errors="coerce").fillna(0).sum())
        exit_qty = float(pd.to_numeric(g["exit_qty"], errors="coerce").fillna(0).sum())
        pnl = float(pd.to_numeric(g["net_pnl_rounding_bound"], errors="coerce").fillna(0).sum())
        rows.append({
            "requested_qty": float(requested_qty),
            "series": str(series),
            "entry_fill_events": int((g["entry_filled_qty"] > EPS).sum()),
            "entry_filled_qty": entry_qty,
            "m5_exit_qty": exit_qty,
            "exit_fraction": exit_qty / entry_qty if entry_qty > EPS else np.nan,
            "net_pnl_rounding_bound": pnl,
            "taker_fees": float(pd.to_numeric(g["m5_taker_fee"], errors="coerce").fillna(0).sum()),
            "residual_qty_zero_valued": float(pd.to_numeric(g["residual_qty_zero_valued"], errors="coerce").fillna(0).sum()),
        })
    return pd.DataFrame(rows).sort_values(["requested_qty", "net_pnl_rounding_bound"], ascending=[True, False])


def run_deep_tail_capacity_dev(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected development source {HARD_BOUND_SESSION}, got {source.name}")
    if hard_bind and "mm_event_m0_m5_oos_cycle_q10_v1" not in str(source.parent):
        raise RuntimeError("Expected source under mm_event_m0_m5_oos_cycle_q10_v1")

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
        raise RuntimeError("Stored fee preflight is not PASS; refusing fee-adjusted capacity economics.")
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    meta = V1._metadata(source)
    if not meta:
        raise RuntimeError("No valid market metadata")

    if show:
        print("=" * 150)
        print("DEEP-TAIL 5C CAPACITY DEV V3 — FLOW-CAPPED ENTRY / TOP-3 M5 EXIT")
        print("=" * 150)
        print("Source:", source)
        print("Fixed entry:", f"{ENTRY_C}c")
        print("Fixed exit: M5 marketable exit")
        print("Quantity grid:", REQUESTED_QTYS)
        print("Entry maker fee: stored historical quadratic maker = $0")
        print("M5: consume actual top-3 outcome bids + quadratic taker fee + rounding bound")
        print("Spread/slippage: embedded in actual exit proceeds; diagnostics reported separately")
        print()
        print("PASS 1/3 — loading M1-M5 public trades with V11 causal execution clock...")

    trades, trade_clock_stats = V1._load_trades(source, meta, show=show)

    if show:
        print("PASS 2/3 — scanning true M5 top-3 depth and M1-M5 book coverage...")
    books = _scan_m5_books(source, meta, show=show)

    if show:
        print("PASS 3/3 — replaying requested quantity grid...")
    detail = _evaluate(source, meta, trades, books, fee_mult, show=show)
    curve = _aggregate_curve(detail)
    by_asset = _aggregate_asset(detail)

    out = _new_output(source.name)
    detail.to_csv(out / "capacity_order_detail.csv", index=False)
    curve.to_csv(out / "capacity_curve.csv", index=False)
    by_asset.to_csv(out / "capacity_by_asset.csv", index=False)

    feasible = curve[
        (curve["net_pnl_rounding_bound"] > 0)
        & (curve["m5_exit_fraction_of_filled_qty"] >= 0.99 - EPS)
    ].copy()
    best_feasible = (
        feasible.sort_values(["net_pnl_rounding_bound", "requested_qty"], ascending=False).iloc[0].to_dict()
        if len(feasible) else {}
    )

    max_total_pnl = (
        curve.sort_values(["net_pnl_rounding_bound", "requested_qty"], ascending=False).iloc[0].to_dict()
        if len(curve) else {}
    )

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "research_stage": "DEVELOPMENT_CAPACITY_ONLY",
        "hard_bound": bool(hard_bind),
        "candidate": {
            "entry_c": ENTRY_C,
            "entry_window": "M1 to M5",
            "exit_rule": "M5_ONLY",
            "tails": ["YES", "NO"],
            "activation_latency_ms": ACTIVATION_LATENCY_MS,
            "requested_qty_grid": list(REQUESTED_QTYS),
        },
        "trade_clock_stats": trade_clock_stats,
        "eligible_tickers": int(sum(bool(v["coverage_eligible"]) for v in books.values())),
        "best_positive_qty_with_99pct_m5_exit_coverage": best_feasible,
        "max_total_pnl_row_any_exit_coverage": max_total_pnl,
        "entry_capacity_policy": (
            "cumulative strict-through seller trade quantity only; exact-price flow excluded; partial fills supported"
        ),
        "m5_capacity_policy": (
            "consume recorded top-3 outcome bids level by level; residual beyond top-3 valued at zero"
        ),
        "cost_policy": (
            "entry maker fee $0 from stored PASS preflight; M5 actual bid ladder embeds spread/slippage; "
            "quadratic taker fee per consumed level; additional $0.0099 rounding upper-bound once per nonzero exit"
        ),
        "guardrail": (
            "Development only. Quantity selected from this curve must be frozen before validation; do not use validation to retune size."
        ),
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
    }
    OOS._atomic_json(out / "summary.json", summary)

    if show:
        print("=" * 150)
        print("CAPACITY CURVE — PRIMARY CONSERVATIVE ECONOMICS")
        print("=" * 150)
        cols = [
            "requested_qty", "eligible_posted_orders", "entry_fill_events",
            "full_entry_fill_orders", "partial_entry_fill_orders", "entry_filled_qty",
            "entry_fill_fraction_of_requested_contracts", "m5_exit_qty",
            "m5_exit_fraction_of_filled_qty", "positions_with_m5_residual",
            "residual_qty_zero_valued", "m5_taker_fees",
            "m5_cross_spread_cost_vs_mid_embedded", "m5_top3_slippage_vs_best_bid_embedded",
            "balance_rounding_upper_bound_drag", "net_pnl_rounding_bound",
            "net_pnl_per_requested_contract", "net_pnl_per_filled_contract",
            "incremental_net_pnl", "marginal_pnl_per_incremental_requested_contract_per_order",
            "max_drawdown_rounding_bound",
        ]
        print(curve[cols].to_string(index=False))
        print()
        print("Best positive row with >=99% M5 top-3 exit coverage:")
        print(best_feasible)
        print()
        print("Highest total development PnL row regardless of exit coverage:")
        print(max_total_pnl)
        print()
        print("IMPORTANT:")
        print("- Spread crossing and top-3 slippage are ALREADY embedded in exit proceeds.")
        print("- The displayed spread/slippage columns are diagnostics and are not double-subtracted.")
        print("- Any M5 residual beyond recorded top-3 depth is valued at ZERO in net PnL.")
        print("- Exact 5c entry prints are excluded from capacity because deep queue ahead is unknown.")
        print("- This is development only. Do not freeze the scalar maximum automatically.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "capacity_curve": curve,
        "detail": detail,
        "by_asset": by_asset,
        "books": books,
        "output_dir": out,
    }
