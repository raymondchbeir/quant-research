from __future__ import annotations

"""Development-only feasibility study for deep-tail passive liquidity on Kalshi 15m crypto.

Motivation
----------
The formal 24h event-time capture contains frequent large-range public-trade clusters.
Rather than treating the cluster classifier itself as a trading signal, this study asks
a simpler causal question:

    If a Q1 order had already been resting from M1 at a fixed cheap outcome price,
    would public aggressive flow have traded strictly THROUGH that price, and what
    happened to the outcome price afterward?

The study posts symmetric hypothetical BUY orders from M1 to M5:
- BUY YES at 10/15/20/25/30 cents; and
- BUY NO  at 10/15/20/25/30 cents.

On the recorder's unified YES-price scale, BUY NO @ p is the opposite tail, equivalent
to a resting YES ask at 1-p.

Primary conservative fill rule
------------------------------
The V5 recorder persists only top-3 book levels, so deep FIFO queue ahead is not always
observable.  We therefore do NOT claim a fill merely because our level trades exactly.
A primary entry is counted only when same-outcome aggressive selling trades strictly
THROUGH our resting price after the order is active:

- BUY YES @ p: taker_book_side='ask' and YES trade price < p.
- BUY NO  @ p: taker_book_side='bid' and NO trade price < p
               (equivalently YES trade price > 1-p).

Exact-price trade quantity is retained only as an optimistic/touch diagnostic.

Causal clocks / activation
--------------------------
- Entry intent is fixed at M1.
- Hypothetical order activation is M1 + 100ms (explicit conservative engineering
  assumption; NOT claimed to be measured Kalshi latency).
- Trade economic time uses V11's corrected causal execution clock: public exchange_time
  unless that timestamp is locally impossible (after receipt), in which case it clamps
  to receipt_time.
- Fill knowledge uses max(public receipt_time, economic execution time).
- A passive exit target is not active until fill knowledge + 100ms.
- Conservative target exits require aggressive buying to trade strictly THROUGH the
  target after the exit is active.  Exact target touches do not count as fills.

Book diagnostics
----------------
A second pass measures how long each exact candidate level is visible inside the
persisted top-3 book during M1-M5, the number of visible visits, time-weighted displayed
quantity, valid-BBO coverage, and the last executable outcome bid before M5.

Economics
---------
Q1 only.  Entry and confirmed target exits are passive and use the stored formal-OOS
zero-maker-fee assumption.  If a filled position does not achieve a conservative target
exit, the study liquidates Q1 at the last observed executable outcome bid before M5 and
applies the stored quadratic taker fee.  A separate PnL column subtracts an additional
$0.0099 per M5 liquidation as the same conservative balance-rounding upper bound used by
the frozen OOS stack.

Scientific status
-----------------
DEVELOPMENT / DISCOVERY ONLY.  This 24h realization has already been inspected heavily.
Any candidate selected from these surfaces must be frozen before the separate validation
sample is opened for this strategy.  No API calls.  No orders.  Source files read-only.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q10_oos_dual_clock_causal_replay_v11 as V11

VERSION = "MM_DEEP_TAIL_PASSIVE_FEASIBILITY_DEV_V1"
HARD_BOUND_SESSION = "20260817_064143"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_passive_feasibility_dev_v1"

M1_S = 60.0
M5_S = 300.0
WINDOW_S = M5_S - M1_S
ENTRY_LEVELS_C = (10, 15, 20, 25, 30)
EXIT_TARGETS_C = (30, 35, 40, 45, 50)
QTY = 1.0
ACTIVATION_LATENCY_MS = 100.0
MIN_BOOK_COVERAGE_FRAC = 0.95
BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS = 0.0099
EPS = 1e-10


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


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _ts(x):
    z = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(z):
        return np.nan
    return float(z.timestamp())


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _metadata(source: Path):
    by_ticker = {}
    for r in _iter_jsonl(source / "market_metadata.jsonl"):
        ticker = str(r.get("ticker") or "")
        close = pd.to_datetime(r.get("close_time"), utc=True, errors="coerce")
        if not ticker or pd.isna(close):
            continue
        by_ticker[ticker] = {
            "ticker": ticker,
            "series": str(r.get("series_ticker") or ""),
            "close_time": close.isoformat(),
            "window_start_s": float((close - pd.Timedelta(minutes=15)).timestamp()),
        }
    return by_ticker


def _load_trades(source: Path, meta, *, show=True):
    """Load only M1-M5 trades, with V11 causal economic time, grouped by ticker."""
    rows = defaultdict(list)
    c = defaultdict(int)
    for n, r in enumerate(_iter_jsonl(source / "trades_event_time.jsonl"), start=1):
        ticker = str(r.get("ticker") or "")
        if ticker not in meta:
            continue
        e = _f(r.get("elapsed_s"))
        if not (np.isfinite(e) and M1_S <= e < M5_S):
            continue
        yes = _f(r.get("yes_price"))
        qty = _f(r.get("qty"))
        side = str(r.get("taker_book_side") or "").lower()
        rt = V11._receipt_s(r)
        if not (
            np.isfinite(yes) and 0.0 <= yes <= 1.0
            and np.isfinite(qty) and qty > 0
            and side in {"bid", "ask"}
            and np.isfinite(rt)
        ):
            c["invalid_selected_trade"] += 1
            continue
        exec_s, exec_source = V11._causal_exec_s(r, rt)
        if exec_source == "CLAMP_EXCHANGE_AFTER_RECEIPT":
            c["exchange_after_receipt_clamped"] += 1
        elif exec_source == "RECEIPT_FALLBACK":
            c["exchange_missing_receipt_fallback"] += 1
        else:
            c["exchange_time_used"] += 1
        rows[ticker].append({
            "exec_s": float(exec_s),
            "receipt_s": float(rt),
            "obs_s": float(max(exec_s, rt)),
            "receipt_elapsed_s": float(e),
            "yes_price": float(yes),
            "qty": float(qty),
            "taker_book_side": side,
            "trade_id": str(r.get("trade_id") or ""),
        })
        c["selected_trade_rows"] += 1
        if show and n % 250_000 == 0:
            print(
                f"trade load: read {n:,} rows | selected={c['selected_trade_rows']:,} | "
                f"tickers={len(rows):,}"
            )
    for ticker in rows:
        rows[ticker].sort(key=lambda z: (z["exec_s"], z["receipt_s"], z["trade_id"]))
    return rows, dict(c)


def _level_qty(levels, price):
    for x in levels or []:
        try:
            p, q = float(x[0]), float(x[1])
        except Exception:
            continue
        if abs(p - price) <= EPS:
            return max(0.0, q)
    return 0.0


def _book_diagnostics(source: Path, meta, *, show=True):
    """Receipt-clock top-3 occupancy + M5 executable outcome bid."""
    level_keys = [(tail, c / 100.0) for tail in ("YES", "NO") for c in ENTRY_LEVELS_C]
    stats = defaultdict(lambda: {
        "visible_seconds": 0.0,
        "visible_qty_time": 0.0,
        "visible_visits": 0,
        "last_visible": False,
    })
    coverage = defaultdict(float)
    last = {}
    m5 = {}
    finalized = set()

    def apply_interval(ticker, state, end_e):
        if state is None:
            return
        start = max(M1_S, float(state["elapsed_s"]))
        end = min(M5_S, float(end_e))
        dt = max(0.0, end - start)
        if dt <= 0:
            return
        coverage[ticker] += dt
        bid_levels = state["bid_levels"]
        ask_levels = state["ask_levels"]
        for tail, p in level_keys:
            if tail == "YES":
                px = p
                qty = _level_qty(bid_levels, px)
            else:
                px = 1.0 - p
                qty = _level_qty(ask_levels, px)
            key = (ticker, tail, int(round(p * 100)))
            z = stats[key]
            visible = qty > EPS
            if visible:
                z["visible_seconds"] += dt
                z["visible_qty_time"] += qty * dt
            if visible and not z["last_visible"]:
                z["visible_visits"] += 1
            z["last_visible"] = bool(visible)

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
                }
            finalized.add(ticker)
            last.pop(ticker, None)
        else:
            last[ticker] = {
                "elapsed_s": float(e),
                "bid": float(cur["bid"]),
                "ask": float(cur["ask"]),
                "bid_levels": list(cur["bid_levels"]),
                "ask_levels": list(cur["ask_levels"]),
            }
        if show and n % 1_000_000 == 0:
            print(f"book scan: {n:,} rows | active={len(last):,} | finalized={len(finalized):,}")

    for ticker, prev in list(last.items()):
        apply_interval(ticker, prev, M5_S)
        m5[ticker] = {"yes_bid": float(prev["bid"]), "yes_ask": float(prev["ask"])}

    out = {}
    for ticker in meta:
        out[ticker] = {
            "book_covered_seconds": min(WINDOW_S, float(coverage.get(ticker, 0.0))),
            "m5_yes_bid": _f((m5.get(ticker) or {}).get("yes_bid")),
            "m5_yes_ask": _f((m5.get(ticker) or {}).get("yes_ask")),
            "levels": {},
        }
        for tail, p in level_keys:
            key = (ticker, tail, int(round(p * 100)))
            z = stats[key]
            vis_s = float(z["visible_seconds"])
            out[ticker]["levels"][(tail, int(round(p * 100)))] = {
                "visible_seconds": vis_s,
                "visible_fraction_of_m1_m5": vis_s / WINDOW_S,
                "visible_visits": int(z["visible_visits"]),
                "time_weighted_visible_qty": (
                    float(z["visible_qty_time"]) / vis_s if vis_s > EPS else np.nan
                ),
            }
    return out


def _outcome_price(tail, yes_price):
    return float(yes_price) if tail == "YES" else 1.0 - float(yes_price)


def _entry_seller_side(tail):
    return "ask" if tail == "YES" else "bid"


def _exit_buyer_side(tail):
    return "bid" if tail == "YES" else "ask"


def _m5_outcome_bid(tail, book_diag):
    if tail == "YES":
        return _f(book_diag.get("m5_yes_bid"))
    ask = _f(book_diag.get("m5_yes_ask"))
    return 1.0 - ask if np.isfinite(ask) else np.nan


def _evaluate_order(ticker, tail, entry_c, trades, meta_row, book_diag, fee_mult):
    entry = float(entry_c) / 100.0
    active_s = float(meta_row["window_start_s"]) + M1_S + ACTIVATION_LATENCY_MS / 1000.0
    seller_side = _entry_seller_side(tail)
    buyer_side = _exit_buyer_side(tail)

    exact_touch_trades = 0
    exact_touch_qty = 0.0
    first_touch_exec_s = np.nan
    fill_idx = None

    for i, tr in enumerate(trades):
        # Both clocks must be after local activation.  This is intentionally conservative.
        if tr["receipt_s"] + EPS < active_s or tr["exec_s"] + EPS < active_s:
            continue
        if tr["taker_book_side"] != seller_side:
            continue
        opx = _outcome_price(tail, tr["yes_price"])
        if abs(opx - entry) <= EPS:
            exact_touch_trades += 1
            exact_touch_qty += float(tr["qty"])
            if not np.isfinite(first_touch_exec_s):
                first_touch_exec_s = float(tr["exec_s"])
        elif opx < entry - EPS:
            fill_idx = i
            break

    lev = book_diag["levels"][(tail, int(entry_c))]
    base = {
        "ticker": ticker,
        "series": meta_row["series"],
        "close_time": meta_row["close_time"],
        "tail": tail,
        "entry_c": int(entry_c),
        "qty": QTY,
        "order_active_elapsed_s": M1_S + ACTIVATION_LATENCY_MS / 1000.0,
        "book_covered_seconds": float(book_diag["book_covered_seconds"]),
        "book_coverage_fraction": float(book_diag["book_covered_seconds"]) / WINDOW_S,
        "coverage_eligible": bool(float(book_diag["book_covered_seconds"]) >= MIN_BOOK_COVERAGE_FRAC * WINDOW_S - EPS),
        "top3_level_visible_seconds": float(lev["visible_seconds"]),
        "top3_level_visible_fraction": float(lev["visible_fraction_of_m1_m5"]),
        "top3_level_visible_visits": int(lev["visible_visits"]),
        "top3_level_time_weighted_qty": _f(lev["time_weighted_visible_qty"]),
        "exact_price_seller_trades_before_through": int(exact_touch_trades),
        "exact_price_seller_qty_before_through": float(exact_touch_qty),
        "first_exact_touch_exec_s": first_touch_exec_s,
        "conservative_entry_fill": bool(fill_idx is not None),
        "fill_exec_s": np.nan,
        "fill_observed_s": np.nan,
        "fill_trade_yes_price": np.nan,
        "fill_trade_outcome_price": np.nan,
        "fill_trade_qty": np.nan,
        "max_outcome_trade_after_fill": np.nan,
        "min_outcome_trade_after_fill": np.nan,
        "m5_outcome_bid": _m5_outcome_bid(tail, book_diag),
    }
    target_rows = []

    if fill_idx is None:
        for target_c in EXIT_TARGETS_C:
            if int(target_c) <= int(entry_c):
                continue
            target_rows.append({
                **base,
                "target_c": int(target_c),
                "target_trade_reached": False,
                "target_reach_exec_s": np.nan,
                "seconds_fill_to_target_trade": np.nan,
                "conservative_target_exit_fill": False,
                "target_exit_exec_s": np.nan,
                "seconds_fill_obs_to_exit": np.nan,
                "used_m5_fallback": False,
                "m5_taker_fee": 0.0,
                "net_pnl_q1": 0.0,
                "net_pnl_q1_rounding_upper_bound": 0.0,
            })
        return base, target_rows

    fill = trades[fill_idx]
    fill_exec = float(fill["exec_s"])
    fill_obs = float(max(fill["obs_s"], fill_exec))
    fill_opx = _outcome_price(tail, fill["yes_price"])
    after = trades[fill_idx + 1:]
    after_prices = [_outcome_price(tail, tr["yes_price"]) for tr in after]
    base.update({
        "fill_exec_s": fill_exec,
        "fill_observed_s": fill_obs,
        "fill_trade_yes_price": float(fill["yes_price"]),
        "fill_trade_outcome_price": float(fill_opx),
        "fill_trade_qty": float(fill["qty"]),
        "max_outcome_trade_after_fill": float(max(after_prices)) if after_prices else np.nan,
        "min_outcome_trade_after_fill": float(min(after_prices)) if after_prices else np.nan,
    })

    exit_active_s = fill_obs + ACTIVATION_LATENCY_MS / 1000.0
    m5_bid = _f(base["m5_outcome_bid"])
    mult = _f(fee_mult.get(meta_row["series"]))

    for target_c in EXIT_TARGETS_C:
        if int(target_c) <= int(entry_c):
            continue
        target = float(target_c) / 100.0
        reach = None
        exit_fill = None
        for tr in after:
            opx = _outcome_price(tail, tr["yes_price"])
            if reach is None and opx >= target - EPS:
                reach = tr
            if (
                exit_fill is None
                and tr["exec_s"] + EPS >= exit_active_s
                and tr["receipt_s"] + EPS >= exit_active_s
                and tr["taker_book_side"] == buyer_side
                and opx > target + EPS
            ):
                exit_fill = tr
            if reach is not None and exit_fill is not None:
                break

        used_m5 = exit_fill is None and np.isfinite(m5_bid)
        if exit_fill is not None:
            fee = 0.0
            pnl = (target - entry) * QTY
            pnl_round = pnl
        elif used_m5:
            fee = OOS._quadratic_taker_fee(QTY, m5_bid, mult) if np.isfinite(mult) else np.nan
            pnl = (m5_bid - entry) * QTY - fee if np.isfinite(fee) else np.nan
            pnl_round = pnl - BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS if np.isfinite(pnl) else np.nan
        else:
            fee = np.nan
            pnl = np.nan
            pnl_round = np.nan

        target_rows.append({
            **base,
            "target_c": int(target_c),
            "target_trade_reached": bool(reach is not None),
            "target_reach_exec_s": float(reach["exec_s"]) if reach is not None else np.nan,
            "seconds_fill_to_target_trade": float(reach["exec_s"] - fill_exec) if reach is not None else np.nan,
            "conservative_target_exit_fill": bool(exit_fill is not None),
            "target_exit_exec_s": float(exit_fill["exec_s"]) if exit_fill is not None else np.nan,
            "seconds_fill_obs_to_exit": float(exit_fill["exec_s"] - fill_obs) if exit_fill is not None else np.nan,
            "used_m5_fallback": bool(used_m5),
            "m5_taker_fee": float(fee) if np.isfinite(fee) else np.nan,
            "net_pnl_q1": float(pnl) if np.isfinite(pnl) else np.nan,
            "net_pnl_q1_rounding_upper_bound": float(pnl_round) if np.isfinite(pnl_round) else np.nan,
        })
    return base, target_rows


def _aggregate_levels(order_df: pd.DataFrame):
    rows = []
    for (tail, entry_c), g in order_df.groupby(["tail", "entry_c"], sort=True):
        elig = g[g["coverage_eligible"]]
        fills = elig[elig["conservative_entry_fill"]]
        vis = pd.to_numeric(elig["top3_level_visible_seconds"], errors="coerce")
        rows.append({
            "tail": tail,
            "entry_c": int(entry_c),
            "posted_orders_all_metadata": int(len(g)),
            "coverage_eligible_orders": int(len(elig)),
            "conservative_price_through_fills": int(len(fills)),
            "fill_rate_per_eligible_order": float(len(fills) / len(elig)) if len(elig) else np.nan,
            "orders_with_exact_price_trade_before_through": int((elig["exact_price_seller_trades_before_through"] > 0).sum()),
            "exact_price_seller_qty_before_through": float(pd.to_numeric(elig["exact_price_seller_qty_before_through"], errors="coerce").fillna(0).sum()),
            "orders_level_visible_top3": int((vis > EPS).sum()),
            "mean_top3_visible_seconds_per_eligible_order": float(vis.mean()) if len(vis) else np.nan,
            "median_top3_visible_seconds_per_eligible_order": float(vis.median()) if len(vis) else np.nan,
            "mean_top3_visible_seconds_given_visible": float(vis[vis > EPS].mean()) if (vis > EPS).any() else np.nan,
            "mean_visible_visits": float(pd.to_numeric(elig["top3_level_visible_visits"], errors="coerce").mean()) if len(elig) else np.nan,
            "mean_time_weighted_visible_qty_given_visible": float(pd.to_numeric(elig.loc[vis > EPS, "top3_level_time_weighted_qty"], errors="coerce").mean()) if (vis > EPS).any() else np.nan,
            "mean_book_coverage_fraction": float(pd.to_numeric(elig["book_coverage_fraction"], errors="coerce").mean()) if len(elig) else np.nan,
        })
    both = pd.DataFrame(rows)
    combined = []
    for entry_c, g in both.groupby("entry_c", sort=True):
        # Combine from order detail for exact denominators rather than averaging tail rates.
        q = order_df[(order_df["entry_c"] == entry_c) & order_df["coverage_eligible"]]
        fills = q[q["conservative_entry_fill"]]
        vis = pd.to_numeric(q["top3_level_visible_seconds"], errors="coerce")
        combined.append({
            "tail": "BOTH",
            "entry_c": int(entry_c),
            "posted_orders_all_metadata": int((order_df["entry_c"] == entry_c).sum()),
            "coverage_eligible_orders": int(len(q)),
            "conservative_price_through_fills": int(len(fills)),
            "fill_rate_per_eligible_order": float(len(fills) / len(q)) if len(q) else np.nan,
            "orders_with_exact_price_trade_before_through": int((q["exact_price_seller_trades_before_through"] > 0).sum()),
            "exact_price_seller_qty_before_through": float(pd.to_numeric(q["exact_price_seller_qty_before_through"], errors="coerce").fillna(0).sum()),
            "orders_level_visible_top3": int((vis > EPS).sum()),
            "mean_top3_visible_seconds_per_eligible_order": float(vis.mean()) if len(vis) else np.nan,
            "median_top3_visible_seconds_per_eligible_order": float(vis.median()) if len(vis) else np.nan,
            "mean_top3_visible_seconds_given_visible": float(vis[vis > EPS].mean()) if (vis > EPS).any() else np.nan,
            "mean_visible_visits": float(pd.to_numeric(q["top3_level_visible_visits"], errors="coerce").mean()) if len(q) else np.nan,
            "mean_time_weighted_visible_qty_given_visible": float(pd.to_numeric(q.loc[vis > EPS, "top3_level_time_weighted_qty"], errors="coerce").mean()) if (vis > EPS).any() else np.nan,
            "mean_book_coverage_fraction": float(pd.to_numeric(q["book_coverage_fraction"], errors="coerce").mean()) if len(q) else np.nan,
        })
    return pd.concat([both, pd.DataFrame(combined)], ignore_index=True).sort_values(["entry_c", "tail"])


def _aggregate_surface(target_df: pd.DataFrame):
    rows = []
    for (tail, entry_c, target_c), g in target_df.groupby(["tail", "entry_c", "target_c"], sort=True):
        elig = g[g["coverage_eligible"]]
        fills = elig[elig["conservative_entry_fill"]]
        exited = fills[fills["conservative_target_exit_fill"]]
        reached = fills[fills["target_trade_reached"]]
        pnl = pd.to_numeric(elig["net_pnl_q1"], errors="coerce")
        pnl_round = pd.to_numeric(elig["net_pnl_q1_rounding_upper_bound"], errors="coerce")
        tf = pd.to_numeric(reached["seconds_fill_to_target_trade"], errors="coerce").dropna()
        rows.append({
            "tail": tail,
            "entry_c": int(entry_c),
            "target_c": int(target_c),
            "coverage_eligible_posted_orders": int(len(elig)),
            "entry_fills": int(len(fills)),
            "entry_fill_rate": float(len(fills) / len(elig)) if len(elig) else np.nan,
            "target_trade_reached": int(len(reached)),
            "p_target_trade_reached_given_fill": float(len(reached) / len(fills)) if len(fills) else np.nan,
            "conservative_target_exit_fills": int(len(exited)),
            "p_conservative_target_exit_given_fill": float(len(exited) / len(fills)) if len(fills) else np.nan,
            "m5_fallbacks": int(fills["used_m5_fallback"].sum()),
            "median_seconds_fill_to_target_trade": float(tf.median()) if len(tf) else np.nan,
            "total_net_pnl_q1": float(pnl.fillna(0.0).sum()),
            "net_pnl_per_eligible_posted_order": float(pnl.fillna(0.0).sum() / len(elig)) if len(elig) else np.nan,
            "net_pnl_per_entry_fill": float(pd.to_numeric(fills["net_pnl_q1"], errors="coerce").mean()) if len(fills) else np.nan,
            "total_net_pnl_q1_rounding_upper_bound": float(pnl_round.fillna(0.0).sum()),
            "rounding_upper_bound_pnl_per_eligible_posted_order": float(pnl_round.fillna(0.0).sum() / len(elig)) if len(elig) else np.nan,
        })
    base = pd.DataFrame(rows)
    combined = []
    for (entry_c, target_c), g in target_df.groupby(["entry_c", "target_c"], sort=True):
        elig = g[g["coverage_eligible"]]
        fills = elig[elig["conservative_entry_fill"]]
        reached = fills[fills["target_trade_reached"]]
        exited = fills[fills["conservative_target_exit_fill"]]
        pnl = pd.to_numeric(elig["net_pnl_q1"], errors="coerce")
        pnl_round = pd.to_numeric(elig["net_pnl_q1_rounding_upper_bound"], errors="coerce")
        tf = pd.to_numeric(reached["seconds_fill_to_target_trade"], errors="coerce").dropna()
        combined.append({
            "tail": "BOTH",
            "entry_c": int(entry_c),
            "target_c": int(target_c),
            "coverage_eligible_posted_orders": int(len(elig)),
            "entry_fills": int(len(fills)),
            "entry_fill_rate": float(len(fills) / len(elig)) if len(elig) else np.nan,
            "target_trade_reached": int(len(reached)),
            "p_target_trade_reached_given_fill": float(len(reached) / len(fills)) if len(fills) else np.nan,
            "conservative_target_exit_fills": int(len(exited)),
            "p_conservative_target_exit_given_fill": float(len(exited) / len(fills)) if len(fills) else np.nan,
            "m5_fallbacks": int(fills["used_m5_fallback"].sum()),
            "median_seconds_fill_to_target_trade": float(tf.median()) if len(tf) else np.nan,
            "total_net_pnl_q1": float(pnl.fillna(0.0).sum()),
            "net_pnl_per_eligible_posted_order": float(pnl.fillna(0.0).sum() / len(elig)) if len(elig) else np.nan,
            "net_pnl_per_entry_fill": float(pd.to_numeric(fills["net_pnl_q1"], errors="coerce").mean()) if len(fills) else np.nan,
            "total_net_pnl_q1_rounding_upper_bound": float(pnl_round.fillna(0.0).sum()),
            "rounding_upper_bound_pnl_per_eligible_posted_order": float(pnl_round.fillna(0.0).sum() / len(elig)) if len(elig) else np.nan,
        })
    return pd.concat([base, pd.DataFrame(combined)], ignore_index=True).sort_values(["entry_c", "target_c", "tail"])


def _both_tail_fills(order_df: pd.DataFrame):
    rows = []
    q = order_df[order_df["coverage_eligible"]].copy()
    for (ticker, entry_c), g in q.groupby(["ticker", "entry_c"], sort=True):
        yes = g[g["tail"] == "YES"]
        no = g[g["tail"] == "NO"]
        if yes.empty or no.empty:
            continue
        yf = bool(yes.iloc[0]["conservative_entry_fill"])
        nf = bool(no.iloc[0]["conservative_entry_fill"])
        rows.append({
            "ticker": ticker,
            "series": str(g.iloc[0]["series"]),
            "close_time": str(g.iloc[0]["close_time"]),
            "entry_c": int(entry_c),
            "yes_filled": yf,
            "no_filled": nf,
            "both_filled": bool(yf and nf),
            "theoretical_locked_settlement_gross_if_both_retained": (1.0 - 2.0 * int(entry_c) / 100.0) if yf and nf else 0.0,
        })
    return pd.DataFrame(rows)


def run_deep_tail_passive_feasibility(source_session, *, hard_bind=True, show=True):
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
        raise RuntimeError("Stored fee preflight is not PASS; refusing fee-adjusted feasibility economics.")
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    meta = _metadata(source)
    if not meta:
        raise RuntimeError("No valid market metadata")

    if show:
        print("PASS 1/3 — loading M1-M5 public trades with V11 causal execution clock...")
    trades, trade_clock_stats = _load_trades(source, meta, show=show)

    if show:
        print("PASS 2/3 — measuring top-3 level occupancy and M5 executable outcome bids...")
    books = _book_diagnostics(source, meta, show=show)

    if show:
        print("PASS 3/3 — evaluating symmetric Q1 tail orders and passive target exits...")
    order_rows = []
    target_rows = []
    tickers = sorted(meta)
    for j, ticker in enumerate(tickers, start=1):
        tr = trades.get(ticker, [])
        for tail in ("YES", "NO"):
            for entry_c in ENTRY_LEVELS_C:
                base, targets = _evaluate_order(
                    ticker, tail, entry_c, tr, meta[ticker], books[ticker], fee_mult
                )
                order_rows.append(base)
                target_rows.extend(targets)
        if show and j % 100 == 0:
            print(f"evaluated {j:,}/{len(tickers):,} tickers | orders={len(order_rows):,}")

    order_df = pd.DataFrame(order_rows)
    target_df = pd.DataFrame(target_rows)
    level_df = _aggregate_levels(order_df)
    surface_df = _aggregate_surface(target_df)
    both_df = _both_tail_fills(order_df)

    # Window-level fill counts, useful for checking whether the edge is broad or clustered.
    by_window = (
        order_df[order_df["coverage_eligible"]]
        .groupby(["close_time", "entry_c"], as_index=False)
        .agg(
            posted_orders=("ticker", "size"),
            conservative_entry_fills=("conservative_entry_fill", "sum"),
            markets=("ticker", "nunique"),
        )
    )
    by_window["fill_rate"] = by_window["conservative_entry_fills"] / by_window["posted_orders"].replace(0, np.nan)

    out = _new_output(source.name)
    order_df.to_csv(out / "order_level_detail.csv", index=False)
    target_df.to_csv(out / "target_exit_detail.csv", index=False)
    level_df.to_csv(out / "tail_level_feasibility.csv", index=False)
    surface_df.to_csv(out / "target_strategy_surface.csv", index=False)
    both_df.to_csv(out / "both_tail_fill_detail.csv", index=False)
    by_window.to_csv(out / "by_window_entry_fill.csv", index=False)

    elig_orders = order_df[order_df["coverage_eligible"]]
    both_agg = (
        both_df.groupby("entry_c", as_index=False)
        .agg(
            market_windows=("ticker", "size"),
            both_filled=("both_filled", "sum"),
            theoretical_locked_settlement_gross=("theoretical_locked_settlement_gross_if_both_retained", "sum"),
        )
        if not both_df.empty else pd.DataFrame()
    )
    if not both_agg.empty:
        both_agg["both_fill_rate"] = both_agg["both_filled"] / both_agg["market_windows"].replace(0, np.nan)
        both_agg.to_csv(out / "both_tail_fill_summary.csv", index=False)

    best = surface_df[surface_df["tail"] == "BOTH"].sort_values(
        ["rounding_upper_bound_pnl_per_eligible_posted_order", "entry_fills"],
        ascending=False,
    )
    best_row = best.iloc[0].to_dict() if len(best) else {}

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "research_stage": "DEVELOPMENT_DISCOVERY_ONLY",
        "hard_bound": bool(hard_bind),
        "metadata_tickers": int(len(meta)),
        "entry_levels_c": list(ENTRY_LEVELS_C),
        "exit_targets_c": list(EXIT_TARGETS_C),
        "qty": QTY,
        "m1_s": M1_S,
        "m5_s": M5_S,
        "activation_latency_ms": ACTIVATION_LATENCY_MS,
        "minimum_book_coverage_fraction": MIN_BOOK_COVERAGE_FRAC,
        "trade_clock_stats": trade_clock_stats,
        "coverage_eligible_order_rows": int(len(elig_orders)),
        "conservative_entry_fills": int(elig_orders["conservative_entry_fill"].sum()),
        "best_combined_surface_row_by_rounding_upper_bound_pnl_per_posted_order": best_row,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "primary_fill_rule": "strict price-through only; exact-price touch is diagnostic and never counted as a primary fill",
        "clock_policy": "V11 causal execution clock; fill observation=max(receipt,execution); entry/exit activation +100ms; both clocks must be after activation",
        "queue_policy": "deep queue is not invented; strict trade-through is treated as conservative confirmation; top-3 visibility/qty reported separately",
        "fee_policy": "passive maker fills $0 under stored PASS fee preflight; M5 fallback pays frozen quadratic taker fee; separate $0.0099/cross rounding upper-bound PnL",
        "guardrail": (
            "This realization is development only.  Do not use the 15h validation sample to tune parameters after choosing a candidate from these surfaces."
        ),
    }
    OOS._atomic_json(out / "summary.json", summary)

    if show:
        print("=" * 150)
        print("DEEP-TAIL PASSIVE FEASIBILITY DEV V1 — Q1 / CONSERVATIVE PRICE-THROUGH")
        print("=" * 150)
        print("Source:", source)
        print("Metadata tickers:", len(meta))
        print("Trade rows selected:", f"{trade_clock_stats.get('selected_trade_rows', 0):,}")
        print("Exchange-after-receipt rows clamped:", f"{trade_clock_stats.get('exchange_after_receipt_clamped', 0):,}")
        print("Activation latency assumption:", f"{ACTIVATION_LATENCY_MS:.0f} ms")
        print()
        print("TAIL-LEVEL FEASIBILITY")
        cols = [
            "tail", "entry_c", "coverage_eligible_orders", "conservative_price_through_fills",
            "fill_rate_per_eligible_order", "orders_with_exact_price_trade_before_through",
            "orders_level_visible_top3", "mean_top3_visible_seconds_given_visible",
            "mean_time_weighted_visible_qty_given_visible",
        ]
        print(level_df[cols].to_string(index=False))
        print()
        print("COMBINED YES+NO TARGET SURFACE")
        s = surface_df[surface_df["tail"] == "BOTH"].copy()
        cols2 = [
            "entry_c", "target_c", "coverage_eligible_posted_orders", "entry_fills", "entry_fill_rate",
            "p_target_trade_reached_given_fill", "p_conservative_target_exit_given_fill", "m5_fallbacks",
            "net_pnl_per_eligible_posted_order", "rounding_upper_bound_pnl_per_eligible_posted_order",
            "total_net_pnl_q1_rounding_upper_bound",
        ]
        print(s[cols2].to_string(index=False))
        if not both_agg.empty:
            print()
            print("BOTH-TAIL ENTRY FILLS (same ticker + same entry level)")
            print(both_agg.to_string(index=False))
        print()
        print("Best development surface row by conservative rounding-upper-bound PnL/post:")
        print(best_row)
        print()
        print("IMPORTANT: best row is DEVELOPMENT ONLY — do not freeze automatically from one scalar maximum.")
        print("Inspect robustness across neighboring entry/target cells and by-window concentration first.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 150)

    return {
        "summary": summary,
        "orders": order_df,
        "targets": target_df,
        "levels": level_df,
        "surface": surface_df,
        "both_tail": both_df,
        "both_tail_summary": both_agg,
        "by_window": by_window,
        "output_dir": out,
        "version": VERSION,
    }


__all__ = ["VERSION", "run_deep_tail_passive_feasibility"]
