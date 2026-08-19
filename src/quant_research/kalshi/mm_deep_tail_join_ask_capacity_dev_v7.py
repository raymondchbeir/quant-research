from __future__ import annotations

"""Development-only capacity scaling for the 5c deep-tail + immediate JOIN_ASK exit.

This study is intentionally narrow.  The immediate JOIN_ASK exit already won the
24h development comparison at Q5.  V7 changes only requested quantity and asks how
that exact mechanic scales upward.

Fixed strategy
--------------
- From M1: rest BUY YES and BUY NO at 5c.
- Conservative entry capacity: same-outcome aggressive seller quantity trading
  STRICTLY THROUGH 5c; exact 5c prints remain excluded.
- Quantity grid: Q5, Q10, Q20, Q50, Q100, Q200, Q300, Q500, Q750, Q1000.
- If the requested quantity fully fills, once that full fill is locally observable,
  take the latest locally-known outcome BBO and post one fixed passive SELL for the
  full position at the contemporaneous outcome ask.
- Same 100ms exit activation assumption as V4.
- Same V4 queue model: join behind displayed L1; exact-price buyer flow burns queue
  first; strict trade-through fills residual; no cancellation-ahead credit.
- If an entry is only partial by M5, do NOT invent a new partial-exit mechanic: it
  remains M5-only, matching V4's development rule.
- Any passive residual at M5 consumes recorded top-3 outcome bids using V3; deeper
  residual is valued at zero; stored quadratic taker fee + $0.0099 rounding bound.

Speed / cache discipline
------------------------
No multi-GiB raw book scan is needed.  V7 reuses:
- V3 capacity_order_detail.csv for the already-computed strict-through capacity facts;
- V6.3 compact 32-ticker BBO/M5 cache;
- V6.4's already-materialized 14-15 MiB relevant trade filter, reparsed from M1 in
  parallel so full-fill times for larger Q can be reconstructed correctly.

Q5 is taken directly from the prior V4 JOIN_ASK detail, so the scaling curve anchors
exactly to the already-observed +$10.69570 development result rather than subtly
changing the Q5 implementation.

Scientific status
-----------------
DEVELOPMENT ONLY.  This is quantity tuning after the immediate JOIN_ASK hypothesis was
selected on the same 24h realization.  The previously opened ~15h sample cannot be
called independent validation for any size selected here.  No API calls. No orders.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_capacity_dev_v3 as V3
from . import mm_deep_tail_passive_exit_dev_v4 as V4
from . import mm_deep_tail_trailing_passive_exit_dev_v6 as V6
from . import mm_deep_tail_trailing_passive_exit_dev_v6_2 as V62
from . import mm_deep_tail_trailing_passive_exit_dev_v6_3 as V63
from . import mm_deep_tail_trailing_passive_exit_dev_v6_4 as V64

VERSION = "MM_DEEP_TAIL_JOIN_ASK_CAPACITY_DEV_V7"
HARD_BOUND_SESSION = V3.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_join_ask_capacity_dev_v7"

ENTRY = 0.05
ENTRY_C = 5.0
M1_S = V3.M1_S
M5_S = V3.M5_S
ACTIVATION_LATENCY_MS = V3.ACTIVATION_LATENCY_MS
ROUNDING_DRAG = V3.BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS
EPS = V3.EPS

REQUESTED_QTYS = (5, 10, 20, 50, 100, 200, 300, 500, 750, 1000)


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


def _latest(root: Path, session: str, filename: str):
    return V6._latest_result_file(Path(root), session, filename)


def _load_required_prior_results(source: Path):
    v3_detail_path = _latest(
        C.PROJECT_ROOT / "results" / "kalshi_deep_tail_capacity_dev_v3",
        source.name,
        "capacity_order_detail.csv",
    )
    v4_detail_path = _latest(
        C.PROJECT_ROOT / "results" / "kalshi_deep_tail_passive_exit_dev_v4",
        source.name,
        "passive_exit_detail.csv",
    )
    if v3_detail_path is None:
        raise FileNotFoundError("Need prior V3 capacity_order_detail.csv; rerun V3 only if it was deleted.")
    if v4_detail_path is None:
        raise FileNotFoundError("Need prior V4 passive_exit_detail.csv; rerun V4 only if it was deleted.")

    v3 = pd.read_csv(v3_detail_path)
    v4 = pd.read_csv(v4_detail_path)
    v4 = v4[v4["variant"].astype(str).eq("JOIN_ASK")].copy()
    if v4.empty:
        raise RuntimeError("Prior V4 detail has no JOIN_ASK rows")
    return v3, v4, str(v3_detail_path), str(v4_detail_path)


def _load_book_cache(source: Path):
    d = V63.CACHE_ROOT / source.name
    bbo_path = d / "relevant_post_fill_bbo.pkl"
    m5_path = d / "m5_top3.json"
    if not bbo_path.exists() or not m5_path.exists():
        raise FileNotFoundError(
            "V6.3 compact book cache is required. It was already built in this development workflow; "
            "refusing a new 5.67 GiB book scan from V7."
        )
    bbo = pd.read_pickle(bbo_path)
    m5 = V6._read_json(m5_path, {}) or {}
    return bbo, m5, str(bbo_path), str(m5_path)


def _load_m1_trade_cache(source: Path, meta: dict, tickers: set[str], *, show=True):
    """Reparse V6.4's small materialized ticker filter from M1, not from Q5 fill time."""
    cache_dir, _, _, filtered_path, _ = V64._trade_cache_paths(source)
    if not filtered_path.exists() or filtered_path.stat().st_size <= 0:
        raise FileNotFoundError(
            "V6.4 materialized relevant trade filter is missing. Run V6.4 once; "
            "V7 intentionally refuses another full raw trade scan."
        )

    window_start = {t: float(meta[t]["window_start_s"]) for t in tickers if t in meta}
    min_active = {
        t: float(meta[t]["window_start_s"]) + M1_S + ACTIVATION_LATENCY_MS / 1000.0
        for t in tickers if t in meta
    }
    workers = V62._workers_default()
    if show:
        print(
            f"FAST PATH: reparsing existing {filtered_path.stat().st_size/(1024**2):.1f} MiB "
            f"trade filter from M1 with {workers} workers..."
        )
    results = V62._parallel_parse(
        filtered_path,
        V62._parse_trade_range,
        window_start,
        min_active,
        workers=workers,
        show=show,
        label="M1-trade",
    )

    rows = []
    for worker_rows, _, _, _, _ in results:
        rows.extend(worker_rows)
    cols = [
        "ticker", "exec_s", "receipt_s", "obs_s", "yes_price", "qty",
        "taker_book_side", "trade_id",
    ]
    df = pd.DataFrame.from_records(rows, columns=cols)
    if len(df):
        df.sort_values(
            ["ticker", "exec_s", "receipt_s", "trade_id"],
            kind="mergesort",
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)
    if show:
        print(f"FAST PATH: M1-M5 relevant trade rows loaded={len(df):,}")
    return df, str(filtered_path)


def _outcome_price(tail: str, yes_price: float) -> float:
    return float(yes_price) if tail == "YES" else 1.0 - float(yes_price)


def _seller_side(tail: str) -> str:
    return "ask" if tail == "YES" else "bid"


def _full_fill_clock(trades: list[dict], tail: str, qty: float, active_s: float):
    """Strict-through entry flow, returning exact full-fill economic/observation clocks."""
    seller = _seller_side(tail)
    rem = float(qty)
    filled = 0.0
    first_exec = np.nan
    full_exec = np.nan
    full_obs = np.nan
    through_qty = 0.0

    for tr in trades:
        if tr["exec_s"] + EPS < active_s or tr["receipt_s"] + EPS < active_s:
            continue
        if tr["taker_book_side"] != seller:
            continue
        opx = _outcome_price(tail, tr["yes_price"])
        if opx >= ENTRY - EPS:  # exact 5c excluded as well as prices above 5c
            continue
        tq = max(0.0, float(tr["qty"]))
        through_qty += tq
        if rem <= EPS:
            continue
        take = min(rem, tq)
        if take <= EPS:
            continue
        if not np.isfinite(first_exec):
            first_exec = float(tr["exec_s"])
        filled += take
        rem -= take
        if rem <= EPS:
            full_exec = float(tr["exec_s"])
            full_obs = float(max(tr["exec_s"], tr["receipt_s"]))
            break

    return {
        "reconstructed_fill_qty": float(filled),
        "reconstructed_full": bool(rem <= EPS),
        "first_entry_exec_s_reconstructed": first_exec,
        "full_entry_exec_s_reconstructed": full_exec,
        "full_entry_observed_s_reconstructed": full_obs,
        "strict_through_qty_seen_until_full_or_end": float(through_qty),
    }


def _outcome_bbo_and_queue(row: pd.Series, tail: str):
    if tail == "YES":
        return {
            "bid": float(row["yes_bid"]),
            "ask": float(row["yes_ask"]),
            "ask_queue": max(0.0, float(row["ask_q1"])),
        }
    return {
        "bid": 1.0 - float(row["yes_ask"]),
        "ask": 1.0 - float(row["yes_bid"]),
        "ask_queue": max(0.0, float(row["bid_q1"])),
    }


def _latest_known_bbo(g: pd.DataFrame, decision_s: float):
    times = g["receipt_s"].to_numpy(float)
    i = int(np.searchsorted(times, float(decision_s) + EPS, side="right") - 1)
    if i < 0:
        return None
    return g.iloc[i]


def _zero_m5():
    return {
        "exit_qty": 0.0,
        "residual_qty_zero_valued": 0.0,
        "m5_exit_proceeds": 0.0,
        "m5_taker_fee": 0.0,
        "m5_slippage_vs_best_bid": 0.0,
        "m5_cross_cost_vs_mid": 0.0,
    }


def _evaluate_q_gt5(
    source: Path,
    v3: pd.DataFrame,
    bbo: pd.DataFrame,
    m5: dict,
    trades_df: pd.DataFrame,
    fee_mult: dict,
):
    bbo_by_ticker = {
        str(t): g.sort_values("receipt_s", kind="mergesort").reset_index(drop=True)
        for t, g in bbo.groupby("ticker", sort=False)
    }
    trades_by_ticker = {
        str(t): g.to_dict("records")
        for t, g in trades_df.groupby("ticker", sort=False)
    } if len(trades_df) else {}

    rows = []
    q = v3[
        v3["requested_qty"].astype(float).isin([float(x) for x in REQUESTED_QTYS if x > 5])
    ].copy()

    for _, r in q.iterrows():
        requested = float(r["requested_qty"])
        ticker = str(r["ticker"])
        tail = str(r["tail"])
        series = str(r.get("series") or "")
        eligible = bool(r.get("coverage_eligible", False))
        entry_qty = float(_f(r.get("entry_filled_qty"), 0.0)) if eligible else 0.0
        full_entry = bool(r.get("full_entry_fill", False)) if eligible else False
        partial_entry = bool(r.get("partial_entry_fill", False)) if eligible else False
        baseline = float(_f(r.get("net_pnl_rounding_bound"), 0.0))

        if entry_qty <= EPS:
            rows.append({
                "requested_qty": requested,
                "ticker": ticker,
                "series": series,
                "tail": tail,
                "coverage_eligible": eligible,
                "entry_filled_qty": 0.0,
                "full_entry_fill": False,
                "partial_entry_fill": False,
                "decision_snapshot_available": False,
                "decision_snapshot_missing_full_entry": False,
                "quote_c": np.nan,
                "queue_ahead_initial": np.nan,
                "passive_exit_qty": 0.0,
                "passive_exit_full": False,
                "m5_required_qty": 0.0,
                "m5_exit_qty": 0.0,
                "m5_residual_zero_valued": 0.0,
                "m5_taker_fee": 0.0,
                "rounding_drag": 0.0,
                "net_pnl_rounding_bound": 0.0,
                "m5_only_baseline_net": baseline,
                "incremental_vs_m5_only": 0.0,
                "first_entry_exec_s": _f(r.get("first_through_exec_s")),
                "full_entry_observed_s": np.nan,
                "seconds_full_fill_to_passive_full": np.nan,
            })
            continue

        mult = _f(fee_mult.get(series))
        m5snap = m5.get(ticker)
        if not (np.isfinite(mult) and mult > 0 and m5snap):
            raise RuntimeError(f"Missing fee/M5 cache for filled position {ticker} {tail}")

        passive_qty = 0.0
        quote_px = np.nan
        queue0 = np.nan
        passive_full = False
        full_obs = np.nan
        first_exec = _f(r.get("first_through_exec_s"))
        seconds_to_full_exit = np.nan
        decision_available = False
        missing_full_decision = False

        if full_entry:
            meta_active = float(source_meta[ticker]["window_start_s"]) + M1_S + ACTIVATION_LATENCY_MS / 1000.0
            clk = _full_fill_clock(
                trades_by_ticker.get(ticker, []), tail, requested, meta_active
            )
            if not clk["reconstructed_full"]:
                raise RuntimeError(
                    f"V3 says full entry but M1 trade cache cannot reconstruct it: {ticker} {tail} Q{requested:g}"
                )
            full_obs = float(clk["full_entry_observed_s_reconstructed"])
            first_exec = float(clk["first_entry_exec_s_reconstructed"])
            g = bbo_by_ticker.get(ticker)
            snap = _latest_known_bbo(g, full_obs) if g is not None and len(g) else None
            if snap is None:
                missing_full_decision = True
            else:
                decision_available = True
                ob = _outcome_bbo_and_queue(snap, tail)
                quote_px = float(ob["ask"])
                queue0 = float(ob["ask_queue"])
                exit_active_s = full_obs + ACTIVATION_LATENCY_MS / 1000.0
                ex_passive = V4._simulate_passive_exit(
                    trades_by_ticker.get(ticker, []),
                    tail,
                    {"quote_price": quote_px, "queue_ahead_initial": queue0},
                    exit_active_s,
                    entry_qty,
                )
                passive_qty = float(ex_passive["passive_exit_qty"])
                passive_full = bool(ex_passive["passive_exit_full"])
                z = _f(ex_passive.get("full_passive_exit_exec_s"))
                if passive_full and np.isfinite(z):
                    seconds_to_full_exit = float(z - full_obs)

        # Partial entries and full entries with an unavailable decision BBO remain M5-only.
        residual = max(0.0, entry_qty - passive_qty)
        m5ex = V3._consume_m5_depth(tail, residual, m5snap, mult) if residual > EPS else _zero_m5()
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
            "coverage_eligible": eligible,
            "entry_filled_qty": entry_qty,
            "full_entry_fill": full_entry,
            "partial_entry_fill": partial_entry,
            "decision_snapshot_available": decision_available,
            "decision_snapshot_missing_full_entry": bool(missing_full_decision),
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
            "m5_only_baseline_net": baseline,
            "incremental_vs_m5_only": float(net - baseline),
            "first_entry_exec_s": first_exec,
            "full_entry_observed_s": full_obs,
            "seconds_full_fill_to_passive_full": seconds_to_full_exit,
        })

    return pd.DataFrame(rows)


def _q5_reference(v3: pd.DataFrame, v4: pd.DataFrame):
    q3 = v3[v3["requested_qty"].astype(float).eq(5.0)].copy()
    q4 = v4[pd.to_numeric(v4["entry_filled_qty"], errors="coerce").fillna(0) > EPS].copy()
    base_by_key = {
        (str(r["ticker"]), str(r["tail"])): float(_f(r.get("net_pnl_rounding_bound"), 0.0))
        for _, r in q3.iterrows()
    }
    rows = []
    for _, r in q4.iterrows():
        ticker = str(r["ticker"]); tail = str(r["tail"])
        entry_qty = float(_f(r.get("entry_filled_qty"), 0.0))
        pqty = float(_f(r.get("passive_exit_qty"), 0.0))
        m5q = float(_f(r.get("m5_exit_qty"), 0.0))
        residual = max(0.0, entry_qty - pqty)
        base = base_by_key.get((ticker, tail), 0.0)
        rows.append({
            "requested_qty": 5.0,
            "ticker": ticker,
            "series": str(r.get("series") or ""),
            "tail": tail,
            "coverage_eligible": bool(r.get("coverage_eligible", True)),
            "entry_filled_qty": entry_qty,
            "full_entry_fill": bool(r.get("entry_full_q5", True)),
            "partial_entry_fill": False,
            "decision_snapshot_available": bool(r.get("decision_snapshot_available", True)),
            "decision_snapshot_missing_full_entry": False,
            "quote_c": 100.0 * _f(r.get("quote_price")),
            "queue_ahead_initial": _f(r.get("queue_ahead_initial")),
            "passive_exit_qty": pqty,
            "passive_exit_full": bool(r.get("passive_exit_full", False)),
            "m5_required_qty": residual,
            "m5_exit_qty": m5q,
            "m5_residual_zero_valued": float(_f(r.get("m5_residual_zero_valued"), 0.0)),
            "m5_taker_fee": float(_f(r.get("m5_taker_fee"), 0.0)),
            "rounding_drag": float(_f(r.get("rounding_drag"), 0.0)),
            "net_pnl_rounding_bound": float(_f(r.get("net_pnl_rounding_bound"), 0.0)),
            "m5_only_baseline_net": base,
            "incremental_vs_m5_only": float(_f(r.get("net_pnl_rounding_bound"), 0.0) - base),
            "first_entry_exec_s": _f(r.get("first_entry_exec_s")),
            "full_entry_observed_s": _f(r.get("full_entry_observed_s")),
            "seconds_full_fill_to_passive_full": _f(r.get("passive_exit_seconds_to_full")),
        })
    return pd.DataFrame(rows)


def _max_drawdown(g: pd.DataFrame):
    q = g[g["entry_filled_qty"] > EPS].copy()
    if q.empty:
        return 0.0
    q.sort_values(["first_entry_exec_s", "ticker", "tail"], kind="mergesort", inplace=True)
    p = pd.to_numeric(q["net_pnl_rounding_bound"], errors="coerce").fillna(0).to_numpy(float)
    eq = np.cumsum(p)
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    dd = np.r_[0.0, eq] - peak
    return float(dd.min())


def _surface(detail: pd.DataFrame, v3: pd.DataFrame):
    rows = []
    for requested, g in detail.groupby("requested_qty", sort=True):
        requested = float(requested)
        gv3 = v3[v3["requested_qty"].astype(float).eq(requested)]
        filled = g[g["entry_filled_qty"] > EPS].copy()
        entry_qty = float(filled["entry_filled_qty"].sum())
        pqty = float(filled["passive_exit_qty"].sum())
        m5_required = float(filled["m5_required_qty"].sum())
        m5_exit = float(filled["m5_exit_qty"].sum())
        residual = float(filled["m5_residual_zero_valued"].sum())
        net = float(filled["net_pnl_rounding_bound"].sum())
        baseline = float(filled["m5_only_baseline_net"].sum())
        perpos = pd.to_numeric(filled["net_pnl_rounding_bound"], errors="coerce").fillna(0).sort_values(ascending=False)
        top1 = float(perpos.head(1).sum()) if len(perpos) else 0.0
        top5 = float(perpos.head(5).sum()) if len(perpos) else 0.0
        full_pass = filled[filled["passive_exit_full"]]
        secs = pd.to_numeric(full_pass["seconds_full_fill_to_passive_full"], errors="coerce").dropna()
        quotes = pd.to_numeric(filled["quote_c"], errors="coerce").dropna()
        queues = pd.to_numeric(filled["queue_ahead_initial"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        m5_cov = (m5_exit / m5_required) if m5_required > EPS else 1.0
        terminal_cov = ((pqty + m5_exit) / entry_qty) if entry_qty > EPS else np.nan
        missing = int(filled["decision_snapshot_missing_full_entry"].sum())
        positive = bool(net > 0.0)
        gate = bool(positive and m5_cov >= 0.99 - EPS and missing == 0)

        rows.append({
            "requested_qty": requested,
            "eligible_posted_orders": int(gv3["coverage_eligible"].sum()) if len(gv3) else np.nan,
            "entry_fill_events": int(len(filled)),
            "full_entry_fill_orders": int(filled["full_entry_fill"].sum()),
            "partial_entry_fill_orders": int(filled["partial_entry_fill"].sum()),
            "entry_filled_qty": entry_qty,
            "decision_snapshot_missing_full_entries": missing,
            "median_join_ask_quote_c": float(quotes.median()) if len(quotes) else np.nan,
            "median_initial_queue_ahead": float(queues.median()) if len(queues) else np.nan,
            "passive_exit_qty": pqty,
            "passive_exit_fraction_of_entry_qty": pqty / entry_qty if entry_qty > EPS else np.nan,
            "full_passive_exit_positions": int(filled["passive_exit_full"].sum()),
            "full_passive_exit_rate_of_filled_positions": float(filled["passive_exit_full"].mean()) if len(filled) else np.nan,
            "median_seconds_full_fill_to_full_passive_exit": float(secs.median()) if len(secs) else np.nan,
            "m5_required_qty": m5_required,
            "m5_exit_qty": m5_exit,
            "m5_exit_fraction_of_required_qty": m5_cov,
            "terminal_exit_fraction_of_entry_qty": terminal_cov,
            "m5_residual_zero_valued": residual,
            "m5_taker_fees": float(filled["m5_taker_fee"].sum()),
            "rounding_drag": float(filled["rounding_drag"].sum()),
            "m5_only_baseline_net": baseline,
            "join_ask_net_pnl_rounding_bound": net,
            "incremental_vs_m5_only": net - baseline,
            "net_pnl_per_filled_contract": net / entry_qty if entry_qty > EPS else np.nan,
            "max_drawdown_rounding_bound": _max_drawdown(filled),
            "top1_positive_pnl": top1,
            "top5_positive_pnl": top5,
            "top1_share_of_net": top1 / net if net > EPS else np.nan,
            "top5_share_of_net": top5 / net if net > EPS else np.nan,
            "capacity_gate_positive_99pct_m5_no_missing_decision": gate,
        })
    return pd.DataFrame(rows).sort_values("requested_qty").reset_index(drop=True)


def _by_asset(detail: pd.DataFrame):
    rows = []
    for (requested, series), g in detail[detail["entry_filled_qty"] > EPS].groupby(["requested_qty", "series"], sort=True):
        entry = float(g["entry_filled_qty"].sum())
        pqty = float(g["passive_exit_qty"].sum())
        rows.append({
            "requested_qty": float(requested),
            "series": str(series),
            "positions": int(len(g)),
            "entry_filled_qty": entry,
            "passive_exit_qty": pqty,
            "passive_exit_fraction": pqty / entry if entry > EPS else np.nan,
            "m5_exit_qty": float(g["m5_exit_qty"].sum()),
            "m5_residual_zero_valued": float(g["m5_residual_zero_valued"].sum()),
            "net_pnl_rounding_bound": float(g["net_pnl_rounding_bound"].sum()),
            "incremental_vs_m5_only": float(g["incremental_vs_m5_only"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["requested_qty", "net_pnl_rounding_bound"], ascending=[True, False])


# module-level metadata used only inside one run; set immediately before evaluation
source_meta = {}


def run_join_ask_capacity_dev(source_session, *, hard_bind=True, show=True):
    global source_meta
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected development source {HARD_BOUND_SESSION}, got {source.name}")
    if hard_bind and "mm_event_m0_m5_oos_cycle_q10_v1" not in str(source.parent):
        raise RuntimeError("Expected source under mm_event_m0_m5_oos_cycle_q10_v1")

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored development fee preflight is not PASS")
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    source_meta = V3.V1._metadata(source)
    v3, v4, v3_path, v4_path = _load_required_prior_results(source)
    bbo, m5, bbo_path, m5_path = _load_book_cache(source)

    # All positive Q1/5c capacity in V3 came from the same 33 side-positions that fully
    # filled Q5, so the 32-ticker V6.3/V6.4 cache is complete for scaling upward from Q5.
    q5_v3 = v3[v3["requested_qty"].astype(float).eq(5.0)]
    positive_q5 = set(q5_v3.loc[pd.to_numeric(q5_v3["entry_filled_qty"], errors="coerce").fillna(0) > EPS, "ticker"].astype(str))
    cache_tickers = set(bbo["ticker"].astype(str))
    if not positive_q5.issubset(cache_tickers):
        raise RuntimeError("V6.3 book cache does not cover every Q5-positive ticker; refusing biased capacity scaling")

    if show:
        print("=" * 160)
        print("DEEP-TAIL IMMEDIATE JOIN_ASK CAPACITY DEV V7")
        print("=" * 160)
        print("Source:", source)
        print("Fixed entry: 5c strict-through | M1-M5 | YES+NO")
        print("Fixed exit: immediate JOIN_ASK only after FULL requested fill; fixed quote; 100ms activation")
        print("Partial entries: M5-only (same V4 rule; no new partial-exit mechanic)")
        print("Quantity grid:", REQUESTED_QTYS)
        print("Q5 anchor: exact prior V4 JOIN_ASK result")
        print("DEVELOPMENT ONLY — 15h sample is NOT read")
        print()
        print(f"Reusing V3 capacity detail: {v3_path}")
        print(f"Reusing V4 Q5 JOIN_ASK detail: {v4_path}")
        print(f"Reusing V6.3 compact book cache: {bbo_path}")

    tickers = set(bbo["ticker"].astype(str))
    trades_df, filtered_path = _load_m1_trade_cache(source, source_meta, tickers, show=show)

    q5 = _q5_reference(v3, v4)
    qgt5 = _evaluate_q_gt5(source, v3, bbo, m5, trades_df, fee_mult)
    detail = pd.concat([q5, qgt5], ignore_index=True, sort=False)
    surface = _surface(detail, v3)
    by_asset = _by_asset(detail)

    feasible = surface[surface["capacity_gate_positive_99pct_m5_no_missing_decision"]].copy()
    largest_feasible = (
        feasible.sort_values("requested_qty", ascending=False).iloc[0].to_dict()
        if len(feasible) else {}
    )
    max_pnl = (
        surface.sort_values(["join_ask_net_pnl_rounding_bound", "requested_qty"], ascending=False).iloc[0].to_dict()
        if len(surface) else {}
    )

    out = _new_output(source.name)
    detail.to_csv(out / "join_ask_capacity_order_detail.csv", index=False)
    surface.to_csv(out / "join_ask_capacity_curve.csv", index=False)
    by_asset.to_csv(out / "join_ask_capacity_by_asset.csv", index=False)

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "research_stage": "DEVELOPMENT_CAPACITY_ONLY_AFTER_JOIN_ASK_SELECTION",
        "entry_c": ENTRY_C,
        "requested_qty_grid": list(REQUESTED_QTYS),
        "exit_rule": "FULL requested fill -> latest known ask -> fixed JOIN_ASK passive exit; partial entry -> M5-only",
        "activation_latency_ms": ACTIVATION_LATENCY_MS,
        "largest_positive_qty_with_99pct_m5_residual_coverage_and_no_missing_decision": largest_feasible,
        "highest_development_pnl_row": max_pnl,
        "prior_v3_detail": v3_path,
        "prior_v4_detail": v4_path,
        "book_cache": bbo_path,
        "m5_cache": m5_path,
        "materialized_trade_filter": filtered_path,
        "guardrail": (
            "Quantity selection is development tuning on the same 24h realization. The previously opened ~15h sample "
            "cannot serve as independent validation for the selected JOIN_ASK quantity."
        ),
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
    }
    OOS._atomic_json(out / "summary.json", summary)

    if show:
        print("=" * 160)
        print("IMMEDIATE JOIN_ASK CAPACITY CURVE — PRIMARY")
        print("=" * 160)
        cols = [
            "requested_qty", "entry_fill_events", "full_entry_fill_orders", "partial_entry_fill_orders",
            "entry_filled_qty", "decision_snapshot_missing_full_entries", "median_join_ask_quote_c",
            "passive_exit_qty", "passive_exit_fraction_of_entry_qty", "full_passive_exit_positions",
            "m5_required_qty", "m5_exit_qty", "m5_exit_fraction_of_required_qty",
            "terminal_exit_fraction_of_entry_qty", "m5_residual_zero_valued", "m5_taker_fees", "rounding_drag",
            "m5_only_baseline_net", "join_ask_net_pnl_rounding_bound", "incremental_vs_m5_only",
            "net_pnl_per_filled_contract", "max_drawdown_rounding_bound", "top1_share_of_net", "top5_share_of_net",
            "capacity_gate_positive_99pct_m5_no_missing_decision",
        ]
        print(surface[cols].to_string(index=False))
        print()
        print("Largest quantity passing predeclared capacity gate:")
        print(largest_feasible)
        print()
        print("Highest development PnL row (diagnostic only; NOT automatic selection):")
        print(max_pnl)
        print()
        print("GUARDRAIL:", summary["guardrail"])
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "capacity_curve": surface,
        "detail": detail,
        "by_asset": by_asset,
        "output_dir": str(out),
    }
