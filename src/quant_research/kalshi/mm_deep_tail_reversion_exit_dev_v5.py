from __future__ import annotations

"""Development-only post-fill reversion study for the 5c/Q5 deep-tail strategy.

Purpose
-------
After the 5c/Q5 extreme-tail entry was developed, this module asks a different question:

    Once the frozen 5c passive entry is economically filled, how far does the outcome
    price typically rebound before M5, and can a passive exit posted at a selected
    rebound level actually fill before the existing M5 fallback is needed?

This is deliberately run ONLY on the already-inspected 24h DEVELOPMENT sample.  The
~15h validation sample is not read because the passive-exit hypothesis was proposed
after that validation result was observed.

Entry mechanics held fixed
--------------------------
- BUY YES Q5 @ 5c and BUY NO Q5 @ 5c from M1.
- 100ms activation latency.
- V11 causal public-trade execution clock via V1._load_trades.
- Confirmed entry capacity comes only from same-outcome seller trades strictly THROUGH
  5c. Exact 5c prints are excluded because deep FIFO queue ahead is not fully observed.
- Partial entry fills are supported.

Post-fill event study
---------------------
The event anchor is the observation time of the final strict-through trade contributing
to the Q5 entry fill (or the final contributing trade for a partial fill), not a raw
"pillar" classifier.  This avoids using the V14/V16 book/trade mismatch classifier as a
strategy signal and measures the economic event we actually care about: our own fill.

For each filled position the study measures, from fill-observation + 100ms until M5:
- first actionable outcome BBO;
- maximum executable outcome bid and maximum midpoint;
- maximum bid within 1/5/15/30/60 seconds and through M5;
- time to first reach fixed outcome-price levels.

Passive target replay
---------------------
Candidate outcome sell targets: 6,7,8,10,12,15,20,25,30,35,40,45,50 cents.

At fill-observation + 100ms:
- If the target is above the current bid, post at the target.
- If the target is already at/below the current bid, remain passive by posting at
  min(current ask, current bid + 1c).  When this improves the spread, displayed queue
  ahead is zero.
- If the chosen quote equals a recorded ask level, join behind displayed quantity at
  that exact price.

The quote is then left fixed until M5.  Exact-price aggressive buyer flow burns queue
ahead first; buyer flow strictly through the quote can fill the residual conservatively
up to observed trade quantity.  Any remaining position at M5 crosses recorded top-3
outcome bids using V3's depth-aware liquidation and historical quadratic taker fees.
The same $0.0099 conservative balance-rounding drag is applied once if an M5 cross occurs.

This is DEVELOPMENT / DISCOVERY ONLY.  No API calls.  No orders.  Source data read-only.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_capacity_dev_v3 as V3

VERSION = "MM_DEEP_TAIL_REVERSION_EXIT_DEV_V5"
HARD_BOUND_SESSION = V1.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_reversion_exit_dev_v5"

ENTRY_C = 5
ENTRY = 0.05
QTY = 5.0
TARGETS_C = (6, 7, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50)
HORIZONS_S = (1, 5, 15, 30, 60)
M1_S = V1.M1_S
M5_S = V1.M5_S
WINDOW_S = V1.WINDOW_S
ACTIVATION_LATENCY_MS = V1.ACTIVATION_LATENCY_MS
MIN_BOOK_COVERAGE_FRAC = V1.MIN_BOOK_COVERAGE_FRAC
ROUNDING_DRAG = V1.BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS
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
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _outcome_price(tail: str, yes_price: float) -> float:
    return float(yes_price) if tail == "YES" else 1.0 - float(yes_price)


def _entry_seller_side(tail: str) -> str:
    return "ask" if tail == "YES" else "bid"


def _exit_buyer_side(tail: str) -> str:
    return "bid" if tail == "YES" else "ask"


def _entry_anchors(meta: dict, trades: dict):
    rows = []
    for ticker, m in meta.items():
        active_s = float(m["window_start_s"]) + M1_S + ACTIVATION_LATENCY_MS / 1000.0
        tr = trades.get(ticker, [])
        for tail in ("YES", "NO"):
            seller = _entry_seller_side(tail)
            rem = QTY
            first_exec = np.nan
            last_exec = np.nan
            last_obs = np.nan
            contributing_rows = 0
            strict_qty_seen = 0.0
            exact_qty_excluded = 0.0
            exact_rows_excluded = 0

            for z in tr:
                if z["receipt_s"] + EPS < active_s or z["exec_s"] + EPS < active_s:
                    continue
                if z["taker_book_side"] != seller:
                    continue
                opx = _outcome_price(tail, z["yes_price"])
                if abs(opx - ENTRY) <= EPS:
                    exact_qty_excluded += float(z["qty"])
                    exact_rows_excluded += 1
                    continue
                if opx >= ENTRY - EPS:
                    continue

                strict_qty_seen += float(z["qty"])
                if rem <= EPS:
                    continue
                take = min(rem, float(z["qty"]))
                if take <= EPS:
                    continue
                contributing_rows += 1
                if not np.isfinite(first_exec):
                    first_exec = float(z["exec_s"])
                last_exec = float(z["exec_s"])
                last_obs = float(max(z["obs_s"], z["exec_s"]))
                rem -= take

            filled = max(0.0, QTY - rem)
            if filled <= EPS:
                continue
            rows.append({
                "ticker": ticker,
                "series": str(m.get("series") or ""),
                "close_time": str(m.get("close_time") or ""),
                "tail": tail,
                "requested_qty": QTY,
                "entry_filled_qty": float(filled),
                "entry_full_fill": bool(filled >= QTY - EPS),
                "entry_strict_through_qty_seen": float(strict_qty_seen),
                "entry_contributing_trade_rows": int(contributing_rows),
                "exact_entry_rows_excluded": int(exact_rows_excluded),
                "exact_entry_qty_excluded": float(exact_qty_excluded),
                "first_fill_exec_s": float(first_exec),
                "last_fill_exec_s": float(last_exec),
                "fill_observed_s": float(last_obs),
                "exit_active_s": float(last_obs + ACTIVATION_LATENCY_MS / 1000.0),
            })
    return pd.DataFrame(rows)


def _scan_books(source: Path, meta: dict, relevant_tickers: set[str], *, show=True):
    coverage = defaultdict(float)
    last = {}
    m5 = {}
    finalized = set()
    paths = defaultdict(list)

    def apply_interval(ticker: str, state: dict, end_e: float):
        if state is None:
            return
        start = max(M1_S, float(state["elapsed_s"]))
        end = min(M5_S, float(end_e))
        if end > start:
            coverage[ticker] += end - start

    for n, r in enumerate(_iter_jsonl(source / "book_top3_events.jsonl"), start=1):
        ticker = str(r.get("ticker") or "")
        if ticker not in meta or ticker in finalized:
            continue
        e = _f(r.get("elapsed_s"))
        cur = V1.OOS._top_state(r)
        rt = V1._ts(r.get("receipt_time"))
        if not np.isfinite(e) or cur is None or not np.isfinite(rt):
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
                    "bid_levels": list(prev["bid_levels"]),
                    "ask_levels": list(prev["ask_levels"]),
                    "snapshot_elapsed_s": float(prev["elapsed_s"]),
                    "receipt_s": float(prev["receipt_s"]),
                    "true_m5_finalized": True,
                }
            finalized.add(ticker)
            last.pop(ticker, None)
        else:
            state = {
                "receipt_s": float(rt),
                "elapsed_s": float(e),
                "bid": float(cur["bid"]),
                "ask": float(cur["ask"]),
                "mid": float(cur["mid"]),
                "bid_levels": [(float(p), float(q)) for p, q in cur["bid_levels"]],
                "ask_levels": [(float(p), float(q)) for p, q in cur["ask_levels"]],
            }
            last[ticker] = state
            if ticker in relevant_tickers and M1_S <= e < M5_S:
                paths[ticker].append(state)

        if show and n % 1_000_000 == 0:
            print(
                f"book scan: {n:,} rows | relevant_paths={sum(len(v) for v in paths.values()):,} "
                f"| true-M5 finalized={len(finalized):,}"
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
            "path": sorted(paths.get(ticker, []), key=lambda z: z["receipt_s"]),
        }
    return out


def _outcome_book(tail: str, state: dict):
    if tail == "YES":
        bid = float(state["bid"])
        ask = float(state["ask"])
        mid = float(state["mid"])
        bid_levels = [(float(p), float(q)) for p, q in state["bid_levels"]]
        ask_levels = [(float(p), float(q)) for p, q in state["ask_levels"]]
    else:
        bid = 1.0 - float(state["ask"])
        ask = 1.0 - float(state["bid"])
        mid = 1.0 - float(state["mid"])
        bid_levels = [(1.0 - float(p), float(q)) for p, q in state["ask_levels"]]
        ask_levels = [(1.0 - float(p), float(q)) for p, q in state["bid_levels"]]
    bid_levels = sorted(bid_levels, key=lambda z: z[0], reverse=True)
    ask_levels = sorted(ask_levels, key=lambda z: z[0])
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "bid_levels": bid_levels,
        "ask_levels": ask_levels,
    }


def _queue_at_price(levels, price):
    for p, q in levels or []:
        if abs(float(p) - float(price)) <= EPS:
            return max(0.0, float(q))
    return 0.0


def _first_actionable_state(path: list[dict], active_s: float):
    for s in path:
        if float(s["receipt_s"]) + EPS >= float(active_s):
            return s
    return None


def _reversion_detail(anchors: pd.DataFrame, books: dict):
    rows = []
    for _, a in anchors.iterrows():
        ticker = str(a["ticker"])
        tail = str(a["tail"])
        bd = books[ticker]
        active_s = float(a["exit_active_s"])
        path = [s for s in bd["path"] if float(s["receipt_s"]) + EPS >= active_s]
        first = _first_actionable_state(path, active_s)
        if first is None:
            continue
        ob0 = _outcome_book(tail, first)
        obs = [(_outcome_book(tail, s), s) for s in path]
        bids = np.asarray([x[0]["bid"] for x in obs], dtype=float)
        mids = np.asarray([x[0]["mid"] for x in obs], dtype=float)
        times = np.asarray([x[1]["receipt_s"] for x in obs], dtype=float)
        max_i = int(np.nanargmax(bids)) if len(bids) else 0
        row = {
            **a.to_dict(),
            "coverage_eligible": bool(bd["coverage_eligible"]),
            "first_actionable_receipt_s": float(first["receipt_s"]),
            "first_bid_c": 100.0 * ob0["bid"],
            "first_ask_c": 100.0 * ob0["ask"],
            "first_mid_c": 100.0 * ob0["mid"],
            "max_bid_c_to_m5": 100.0 * float(np.nanmax(bids)) if len(bids) else np.nan,
            "max_mid_c_to_m5": 100.0 * float(np.nanmax(mids)) if len(mids) else np.nan,
            "max_bid_reversion_from_entry_c": 100.0 * float(np.nanmax(bids)) - ENTRY_C if len(bids) else np.nan,
            "seconds_to_max_bid": float(times[max_i] - active_s) if len(times) else np.nan,
        }
        for h in HORIZONS_S:
            mask = times <= active_s + float(h) + EPS
            row[f"max_bid_c_{h}s"] = 100.0 * float(np.nanmax(bids[mask])) if mask.any() else np.nan
        for tc in TARGETS_C:
            hit = np.where(bids >= float(tc) / 100.0 - EPS)[0]
            row[f"reached_{tc}c"] = bool(len(hit))
            row[f"seconds_to_{tc}c"] = float(times[hit[0]] - active_s) if len(hit) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _simulate_target(a: pd.Series, target_c: int, trades: list[dict], bd: dict):
    ticker = str(a["ticker"])
    tail = str(a["tail"])
    active_s = float(a["exit_active_s"])
    qty = float(a["entry_filled_qty"])
    path = bd["path"]
    first = _first_actionable_state(path, active_s)
    if first is None or qty <= EPS:
        return None

    ob = _outcome_book(tail, first)
    target = float(target_c) / 100.0
    bid = float(ob["bid"])
    ask = float(ob["ask"])

    if target <= bid + EPS:
        quote = min(ask, bid + 0.01)
    else:
        quote = target
    quote = min(0.99, max(0.01, quote))

    inside = bool(bid + EPS < quote < ask - EPS)
    queue0 = 0.0 if inside else _queue_at_price(ob["ask_levels"], quote)
    queue = float(queue0)
    rem = float(qty)
    passive_qty = 0.0
    buyer_side = _exit_buyer_side(tail)
    first_passive_exec = np.nan
    full_passive_exec = np.nan

    for tr in trades:
        if rem <= EPS:
            break
        if tr["exec_s"] + EPS < active_s or tr["receipt_s"] + EPS < active_s:
            continue
        if tr["taker_book_side"] != buyer_side:
            continue
        opx = _outcome_price(tail, tr["yes_price"])
        if opx < quote - EPS:
            continue
        available = 0.0
        if abs(opx - quote) <= EPS:
            burn = min(queue, float(tr["qty"]))
            queue -= burn
            available = max(0.0, float(tr["qty"]) - burn)
        elif opx > quote + EPS:
            available = float(tr["qty"])
        if available <= EPS:
            continue
        take = min(rem, available)
        if take <= EPS:
            continue
        if not np.isfinite(first_passive_exec):
            first_passive_exec = float(tr["exec_s"])
        passive_qty += take
        rem -= take
        if rem <= EPS:
            full_passive_exec = float(tr["exec_s"])

    mult = _f(a.get("fee_multiplier"))
    if rem > EPS:
        m5 = V3._consume_m5_depth(tail, rem, bd.get("m5"), mult)
    else:
        m5 = {
            "exit_qty": 0.0,
            "residual_qty_zero_valued": 0.0,
            "m5_exit_proceeds": 0.0,
            "m5_taker_fee": 0.0,
            "m5_slippage_vs_best_bid": 0.0,
            "m5_cross_cost_vs_mid": 0.0,
        }

    entry_cost = ENTRY * qty
    passive_proceeds = quote * passive_qty
    rounding = ROUNDING_DRAG if m5["exit_qty"] > EPS else 0.0
    net = (
        passive_proceeds
        + float(m5["m5_exit_proceeds"])
        - entry_cost
        - float(m5["m5_taker_fee"])
        - rounding
    )

    base = V3._consume_m5_depth(tail, qty, bd.get("m5"), mult)
    base_round = ROUNDING_DRAG if base["exit_qty"] > EPS else 0.0
    baseline = float(base["m5_exit_proceeds"]) - entry_cost - float(base["m5_taker_fee"]) - base_round

    return {
        "ticker": ticker,
        "series": str(a["series"]),
        "close_time": str(a["close_time"]),
        "tail": tail,
        "entry_filled_qty": qty,
        "target_c": int(target_c),
        "current_bid_c": 100.0 * bid,
        "current_ask_c": 100.0 * ask,
        "current_spread_c": 100.0 * (ask - bid),
        "quote_c": 100.0 * quote,
        "quote_inside_spread": inside,
        "queue_ahead_initial": float(queue0),
        "passive_exit_qty": float(passive_qty),
        "passive_exit_fraction": float(passive_qty / qty) if qty > EPS else np.nan,
        "passive_exit_full": bool(rem <= EPS),
        "seconds_to_first_passive_fill": float(first_passive_exec - active_s) if np.isfinite(first_passive_exec) else np.nan,
        "seconds_to_full_passive_exit": float(full_passive_exec - active_s) if np.isfinite(full_passive_exec) else np.nan,
        "m5_exit_qty": float(m5["exit_qty"]),
        "m5_residual_zero_valued": float(m5["residual_qty_zero_valued"]),
        "m5_taker_fee": float(m5["m5_taker_fee"]),
        "rounding_drag": float(rounding),
        "net_pnl_rounding_bound": float(net),
        "baseline_m5_only_net": float(baseline),
        "incremental_vs_m5_only": float(net - baseline),
    }


def _aggregate_targets(detail: pd.DataFrame):
    rows = []
    for target_c, g in detail.groupby("target_c", sort=True):
        pnl = pd.to_numeric(g["net_pnl_rounding_bound"], errors="coerce").fillna(0.0)
        inc = pd.to_numeric(g["incremental_vs_m5_only"], errors="coerce").fillna(0.0)
        qty = float(pd.to_numeric(g["entry_filled_qty"], errors="coerce").fillna(0.0).sum())
        pqty = float(pd.to_numeric(g["passive_exit_qty"], errors="coerce").fillna(0.0).sum())
        rows.append({
            "target_c": int(target_c),
            "positions": int(len(g)),
            "entry_filled_qty": qty,
            "passive_exit_qty": pqty,
            "passive_exit_fraction_of_entry_qty": pqty / qty if qty > EPS else np.nan,
            "full_passive_exit_positions": int(g["passive_exit_full"].sum()),
            "full_passive_exit_rate": float(g["passive_exit_full"].mean()) if len(g) else np.nan,
            "inside_spread_quote_rate": float(g["quote_inside_spread"].mean()) if len(g) else np.nan,
            "median_initial_queue": float(pd.to_numeric(g["queue_ahead_initial"], errors="coerce").median()),
            "median_quote_c": float(pd.to_numeric(g["quote_c"], errors="coerce").median()),
            "median_seconds_to_full_passive_exit": float(pd.to_numeric(g["seconds_to_full_passive_exit"], errors="coerce").dropna().median()) if g["passive_exit_full"].any() else np.nan,
            "m5_exit_qty": float(pd.to_numeric(g["m5_exit_qty"], errors="coerce").fillna(0.0).sum()),
            "m5_residual_zero_valued": float(pd.to_numeric(g["m5_residual_zero_valued"], errors="coerce").fillna(0.0).sum()),
            "m5_taker_fees": float(pd.to_numeric(g["m5_taker_fee"], errors="coerce").fillna(0.0).sum()),
            "rounding_drag": float(pd.to_numeric(g["rounding_drag"], errors="coerce").fillna(0.0).sum()),
            "total_net_pnl_rounding_bound": float(pnl.sum()),
            "total_incremental_vs_m5_only": float(inc.sum()),
            "positive_incremental_positions": int((inc > 0).sum()),
            "negative_incremental_positions": int((inc < 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("total_net_pnl_rounding_bound", ascending=False).reset_index(drop=True)


def _reversion_summary(rev: pd.DataFrame):
    if rev.empty:
        return pd.DataFrame()
    rows = []
    for col in ["max_bid_reversion_from_entry_c", "max_bid_c_to_m5", "max_mid_c_to_m5", "seconds_to_max_bid"]:
        x = pd.to_numeric(rev[col], errors="coerce").dropna()
        rows.append({
            "metric": col,
            "n": int(len(x)),
            "mean": float(x.mean()) if len(x) else np.nan,
            "q10": float(x.quantile(.10)) if len(x) else np.nan,
            "q25": float(x.quantile(.25)) if len(x) else np.nan,
            "median": float(x.median()) if len(x) else np.nan,
            "q75": float(x.quantile(.75)) if len(x) else np.nan,
            "q90": float(x.quantile(.90)) if len(x) else np.nan,
            "max": float(x.max()) if len(x) else np.nan,
        })
    return pd.DataFrame(rows)


def _target_reach_summary(rev: pd.DataFrame):
    rows = []
    for tc in TARGETS_C:
        c = f"reached_{tc}c"
        t = f"seconds_to_{tc}c"
        if c not in rev.columns:
            continue
        hit = rev[c].astype(bool)
        times = pd.to_numeric(rev.loc[hit, t], errors="coerce").dropna()
        rows.append({
            "target_c": int(tc),
            "positions": int(len(rev)),
            "reached": int(hit.sum()),
            "reach_rate": float(hit.mean()) if len(rev) else np.nan,
            "median_seconds_to_reach": float(times.median()) if len(times) else np.nan,
            "q25_seconds_to_reach": float(times.quantile(.25)) if len(times) else np.nan,
            "q75_seconds_to_reach": float(times.quantile(.75)) if len(times) else np.nan,
        })
    return pd.DataFrame(rows)


def run_deep_tail_reversion_exit_dev(source_session, *, hard_bind=True, show=True):
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

    fee = V1.OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored development fee preflight is not PASS")
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    meta = V1._metadata(source)
    if show:
        print("=" * 150)
        print("DEEP-TAIL POST-FILL REVERSION + PASSIVE TARGET DEV V5")
        print("=" * 150)
        print("Source:", source)
        print("Entry: 5c | Q5 | YES+NO | M1-M5")
        print("Target grid:", TARGETS_C)
        print("DEVELOPMENT ONLY — validation sample is not read")
        print()
        print("PASS 1/4 — loading M1-M5 causal public trades...")

    trades, trade_clock_stats = V1._load_trades(source, meta, show=show)
    anchors = _entry_anchors(meta, trades)
    if anchors.empty:
        raise RuntimeError("No 5c/Q5 strict-through entry fills found")
    anchors["fee_multiplier"] = anchors["series"].map(fee_mult)
    relevant = set(anchors["ticker"].astype(str))

    if show:
        print(f"Entry anchors: {len(anchors):,} filled side-positions | relevant tickers={len(relevant):,}")
        print("PASS 2/4 — scanning post-fill book paths and true-M5 top-3...")
    books = _scan_books(source, meta, relevant, show=show)

    anchors["coverage_eligible"] = anchors["ticker"].map(lambda t: bool(books[str(t)]["coverage_eligible"]))
    anchors = anchors[anchors["coverage_eligible"]].copy().reset_index(drop=True)

    if show:
        print(f"Coverage-eligible filled positions: {len(anchors):,}")
        print("PASS 3/4 — measuring post-fill reversion distribution...")
    rev = _reversion_detail(anchors, books)
    rev_summary = _reversion_summary(rev)
    reach = _target_reach_summary(rev)

    if show:
        print("PASS 4/4 — replaying fixed passive target quotes + M5 residual fallback...")
    detail_rows = []
    for _, a in anchors.iterrows():
        tr = trades.get(str(a["ticker"]), [])
        bd = books[str(a["ticker"])]
        for tc in TARGETS_C:
            z = _simulate_target(a, int(tc), tr, bd)
            if z is not None:
                detail_rows.append(z)
    detail = pd.DataFrame(detail_rows)
    surface = _aggregate_targets(detail)

    out = _new_output(source.name)
    anchors.to_csv(out / "entry_fill_anchors.csv", index=False)
    rev.to_csv(out / "post_fill_reversion_detail.csv", index=False)
    rev_summary.to_csv(out / "post_fill_reversion_distribution.csv", index=False)
    reach.to_csv(out / "target_reach_distribution.csv", index=False)
    detail.to_csv(out / "passive_target_replay_detail.csv", index=False)
    surface.to_csv(out / "passive_target_surface.csv", index=False)

    best = surface.iloc[0].to_dict() if len(surface) else {}
    baseline = float(detail.drop_duplicates(["ticker", "tail"])["baseline_m5_only_net"].sum()) if len(detail) else np.nan
    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "research_stage": "DEVELOPMENT_DISCOVERY_ONLY",
        "entry_c": ENTRY_C,
        "requested_qty": QTY,
        "target_grid_c": list(TARGETS_C),
        "entry_filled_positions": int(len(anchors)),
        "entry_filled_qty": float(pd.to_numeric(anchors["entry_filled_qty"], errors="coerce").sum()),
        "baseline_m5_only_net": baseline,
        "best_passive_target_row": best,
        "trade_clock_stats": trade_clock_stats,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "guardrail": (
            "This exit hypothesis was created after the 15h validation result was opened. "
            "The 15h sample is therefore not independent validation for any target selected here."
        ),
    }
    V1.OOS._atomic_json(out / "summary.json", summary)

    if show:
        print("=" * 150)
        print("POST-FILL REVERSION DISTRIBUTION")
        print("=" * 150)
        print(rev_summary.to_string(index=False))
        print()
        print("TARGET REACH RATES — BOOK BID, NOT FILL ASSUMPTION")
        print(reach.to_string(index=False))
        print()
        print("=" * 150)
        print("PASSIVE TARGET EXECUTION SURFACE")
        print("=" * 150)
        cols = [
            "target_c", "positions", "entry_filled_qty", "passive_exit_qty",
            "passive_exit_fraction_of_entry_qty", "full_passive_exit_rate",
            "inside_spread_quote_rate", "median_initial_queue", "median_quote_c",
            "median_seconds_to_full_passive_exit", "m5_exit_qty",
            "m5_residual_zero_valued", "m5_taker_fees", "rounding_drag",
            "total_net_pnl_rounding_bound", "total_incremental_vs_m5_only",
        ]
        print(surface[cols].to_string(index=False))
        print()
        print("M5-only baseline net:", f"{baseline:+.4f}")
        print("Best DEVELOPMENT target row:", best)
        print()
        print("IMPORTANT: target selection here is development only; do not reuse the 15h sample as validation.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "anchors": anchors,
        "reversion": rev,
        "reversion_summary": rev_summary,
        "target_reach": reach,
        "detail": detail,
        "target_surface": surface,
        "output_dir": out,
    }
