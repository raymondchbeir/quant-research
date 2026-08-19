from __future__ import annotations

"""Secondary historical robustness replay for frozen 5c/Q10 + immediate JOIN_ASK.

IMPORTANT SCIENTIFIC STATUS
---------------------------
This is NOT independent validation.  The ~15h source session was already opened for the
older 5c/Q5/M5-only strategy before the immediate JOIN_ASK hypothesis and Q10 size were
selected on the separate 24h development realization.  This module therefore answers a
narrow robustness question only:

    Does the now-frozen Q10 JOIN_ASK implementation retain sensible economics and
    execution behavior on a different historical realization?

No parameter sweep, asset filtering, threshold search, exit-rule variation, or size
comparison is performed here.  Results must not be used to retune the strategy.

Frozen strategy
---------------
- Universe: all recorded crypto series in the hard-bound ~15h source.
- M1: rest BUY YES Q10 @ 5c and BUY NO Q10 @ 5c.
- Entry activation: M1 + 100ms.
- Entry evidence: same-outcome aggressive seller flow STRICTLY THROUGH 5c only; exact
  5c prints excluded because deep queue ahead is not observed.
- Partial entry fills are allowed.
- If FULL Q10 is reached, after the full fill is locally observable, use the latest
  locally-known outcome BBO and post one fixed passive SELL Q10 at the current best ask.
- Exit activation: +100ms after fill observation.
- Passive queue model is exactly the development V4/V7 model: displayed L1 queue ahead;
  exact-price aggressive buyer flow burns queue first; strict trade-through fills all
  remaining quantity; no cancellation-ahead credit; no repricing.
- If entry is only partial by M5, no new partial-exit mechanic is invented: M5-only.
- Any passive residual at M5 consumes recorded top-3 outcome bids L1->L3.  Deeper
  residual is valued at zero.
- Entry/passive maker fees are zero under the stored fee model; M5 taker fee uses the
  stored quadratic schedule; subtract $0.0099 rounding upper bound once per M5 cross.

Speed
-----
Trades are read once with the existing causal V11 loader.  Only tickers that actually
receive Q10 entry flow are used to build a receipt-time mmap slice of the large book file,
so the entire multi-GiB book does not need to be parsed.

READ ONLY.  NO API.  NO ORDERS.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_capacity_dev_v3 as V3
from . import mm_deep_tail_passive_exit_dev_v4 as V4
from . import mm_deep_tail_q5_validation_v1 as OLDVAL
from . import mm_deep_tail_trailing_passive_exit_dev_v6_3 as V63

VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q10_SECONDARY_VALIDATION_V1"
HARD_BOUND_SESSION = "20260816_070627"
EXPECTED_PARENT = "mm_event_m0_m5_v5_dev"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_join_ask_q10_secondary_validation_v1"

ENTRY = 0.05
ENTRY_C = 5.0
QTY = 10.0
M1_S = 60.0
M5_S = 300.0
WINDOW_S = M5_S - M1_S
ACTIVATION_LATENCY_MS = 100.0
ROUNDING_DRAG_PER_M5_CROSS = 0.0099
MIN_BOOK_COVERAGE_FRAC = 0.95
EPS = 1e-10
TIME_PAD_S = 2.0

# Context only; never used to change the secondary-check strategy.
DEV_SOURCE_SESSION = "20260817_064143"
DEV_Q10_NET = 21.08520
DEV_Q10_TERMINAL_COVERAGE = 0.9815384615384616
DEV_Q10_PASSIVE_FRACTION = 0.7692307692307693
DEV_Q10_MAX_DRAWDOWN = -0.29950


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


def _outcome_price(tail: str, yes_price: float) -> float:
    return float(yes_price) if tail == "YES" else 1.0 - float(yes_price)


def _entry_seller_side(tail: str) -> str:
    return "ask" if tail == "YES" else "bid"


def _entry_fill_q10(trades: list[dict], tail: str, active_s: float) -> dict:
    seller = _entry_seller_side(tail)
    remaining = QTY
    filled = 0.0
    first_exec = np.nan
    last_exec = np.nan
    last_obs = np.nan
    exact_qty_excluded = 0.0
    exact_rows_excluded = 0
    strict_through_qty = 0.0

    for tr in trades:
        if float(tr["receipt_s"]) + EPS < active_s or float(tr["exec_s"]) + EPS < active_s:
            continue
        if str(tr["taker_book_side"]) != seller:
            continue
        opx = _outcome_price(tail, float(tr["yes_price"]))
        tq = max(0.0, float(tr["qty"]))
        if abs(opx - ENTRY) <= EPS:
            exact_qty_excluded += tq
            exact_rows_excluded += 1
            continue
        if opx >= ENTRY - EPS:
            continue

        strict_through_qty += tq
        if remaining <= EPS:
            continue
        take = min(remaining, tq)
        if take <= EPS:
            continue
        if not np.isfinite(first_exec):
            first_exec = float(tr["exec_s"])
        filled += take
        remaining -= take
        last_exec = float(tr["exec_s"])
        last_obs = float(max(tr["exec_s"], tr["receipt_s"]))
        if remaining <= EPS:
            remaining = 0.0
            break

    return {
        "entry_filled_qty": float(filled),
        "entry_full_q10": bool(filled >= QTY - EPS),
        "entry_partial_q10": bool(EPS < filled < QTY - EPS),
        "first_entry_exec_s": first_exec,
        "full_entry_exec_s": last_exec if filled >= QTY - EPS else np.nan,
        "full_entry_observed_s": last_obs if filled >= QTY - EPS else np.nan,
        "strict_through_qty_seen": float(strict_through_qty),
        "exact_entry_rows_excluded": int(exact_rows_excluded),
        "exact_entry_qty_excluded": float(exact_qty_excluded),
    }


def _merge_intervals(intervals):
    q = sorted((float(a), float(b)) for a, b in intervals if np.isfinite(a) and np.isfinite(b) and b > a)
    if not q:
        return []
    out = [[q[0][0], q[0][1]]]
    for a, b in q[1:]:
        if a <= out[-1][1] + TIME_PAD_S:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(float(a), float(b)) for a, b in out]


def _book_intervals(meta: dict, tickers: set[str]):
    intervals = []
    for ticker in sorted(tickers):
        m = meta.get(ticker)
        if not m:
            continue
        ws = float(m["window_start_s"])
        intervals.append((ws + M1_S - TIME_PAD_S, ws + M5_S + TIME_PAD_S))
    return _merge_intervals(intervals)


def _parse_book_state(r: dict):
    if not bool(r.get("valid_bbo")):
        return None
    bid = _f(r.get("yes_bid"))
    ask = _f(r.get("yes_ask"))
    if not (np.isfinite(bid) and np.isfinite(ask) and 0.0 <= bid < ask <= 1.0):
        return None
    bp, bids = V63._levels_fixed(r.get("bid_levels"))
    ap, asks = V63._levels_fixed(r.get("ask_levels"))
    if not bids or not asks:
        return None
    mid = _f(r.get("mid"), 0.5 * (bid + ask))
    return {
        "yes_bid": float(bid),
        "yes_ask": float(ask),
        "yes_mid": float(mid),
        "bid_p1": bp[0], "bid_q1": bp[1],
        "bid_p2": bp[2], "bid_q2": bp[3],
        "bid_p3": bp[4], "bid_q3": bp[5],
        "ask_p1": ap[0], "ask_q1": ap[1],
        "ask_p2": ap[2], "ask_q2": ap[3],
        "ask_p3": ap[4], "ask_q3": ap[5],
        "bid_levels": [[float(p), float(q)] for p, q in bids],
        "ask_levels": [[float(p), float(q)] for p, q in asks],
    }


def _load_relevant_books(source: Path, meta: dict, tickers: set[str], *, show=True):
    raw_path = source / "book_top3_events.jsonl"
    intervals = _book_intervals(meta, tickers)
    if not intervals:
        raise RuntimeError("No relevant validation book intervals")

    index = V63._build_coarse_index(raw_path, show=show)
    ranges = V63._intervals_to_byte_ranges(index, intervals)
    selected_bytes = int(sum(b - a for a, b in ranges))

    if show:
        print(
            f"BOOK FAST PATH: {len(tickers)} filled tickers | {len(intervals)} merged windows | "
            f"read={selected_bytes/(1024**2):.1f} MiB ({100.0*selected_bytes/index['size']:.2f}% of raw book)"
        )

    rows = []
    events = defaultdict(list)
    parsed = 0
    relevant = 0
    audit_n = 0
    audit_max = 0.0

    for raw in V63._iter_byte_ranges(raw_path, ranges):
        parsed += 1
        try:
            r = V63._loads(raw)
        except Exception:
            continue
        ticker = str(r.get("ticker") or "")
        if ticker not in tickers:
            continue
        e = _f(r.get("elapsed_s"))
        if not np.isfinite(e) or e < M1_S - TIME_PAD_S or e > M5_S + TIME_PAD_S:
            continue
        relevant += 1
        ws = float(meta[ticker]["window_start_s"])
        rt = ws + float(e)

        if audit_n < 100:
            rs = OOS._ts(r.get("receipt_time"))
            if np.isfinite(rs):
                audit_n += 1
                audit_max = max(audit_max, abs(float(rs) - rt))

        state = _parse_book_state(r)
        events[ticker].append((float(e), float(rt), state))
        if state is not None and M1_S <= e < M5_S:
            rows.append((
                ticker, float(rt), float(e),
                state["yes_bid"], state["yes_ask"], state["yes_mid"],
                state["bid_p1"], state["bid_q1"], state["bid_p2"], state["bid_q2"], state["bid_p3"], state["bid_q3"],
                state["ask_p1"], state["ask_q1"], state["ask_p2"], state["ask_q2"], state["ask_p3"], state["ask_q3"],
            ))

    if audit_n and audit_max > 0.001 + EPS:
        raise RuntimeError(f"Validation book receipt-clock audit failed: max error={audit_max:.6f}s")

    cols = [
        "ticker", "receipt_s", "elapsed_s", "yes_bid", "yes_ask", "yes_mid",
        "bid_p1", "bid_q1", "bid_p2", "bid_q2", "bid_p3", "bid_q3",
        "ask_p1", "ask_q1", "ask_p2", "ask_q2", "ask_p3", "ask_q3",
    ]
    bbo = pd.DataFrame.from_records(rows, columns=cols)
    if len(bbo):
        bbo.sort_values(["ticker", "receipt_s"], kind="mergesort", inplace=True)
        bbo.reset_index(drop=True, inplace=True)

    books = {}
    for ticker in sorted(tickers):
        ev = sorted(events.get(ticker, []), key=lambda z: (z[0], z[1]))
        coverage = 0.0
        last_state = None
        last_e = None
        m5_state = None

        for e, rt, state in ev:
            if last_e is not None and last_e < M5_S:
                a = max(M1_S, float(last_e))
                b = min(M5_S, float(e))
                if b > a and last_state is not None:
                    coverage += b - a
            if e >= M5_S:
                if last_state is not None:
                    m5_state = dict(last_state)
                    m5_state["snapshot_elapsed_s"] = float(last_e)
                    m5_state["true_m5_finalized"] = True
                break
            last_e = float(e)
            last_state = state

        if m5_state is None and last_e is not None and last_e < M5_S:
            if last_state is not None:
                a = max(M1_S, float(last_e))
                if M5_S > a:
                    coverage += M5_S - a
                m5_state = dict(last_state)
                m5_state["snapshot_elapsed_s"] = float(last_e)
                m5_state["true_m5_finalized"] = True

        cov = min(WINDOW_S, max(0.0, float(coverage)))
        books[ticker] = {
            "book_covered_seconds": cov,
            "book_coverage_fraction": cov / WINDOW_S,
            "coverage_eligible": bool(
                cov >= MIN_BOOK_COVERAGE_FRAC * WINDOW_S - EPS
                and m5_state is not None
            ),
            "m5": m5_state,
        }

    if show:
        eligible = sum(bool(v["coverage_eligible"]) for v in books.values())
        print(
            f"BOOK FAST PATH DONE: parsed={parsed:,} | relevant={relevant:,} | "
            f"BBO states={len(bbo):,} | eligible filled tickers={eligible}/{len(tickers)} | "
            f"clock audit max={audit_max:.9f}s"
        )
    return bbo, books, {
        "raw_size_bytes": int(index["size"]),
        "selected_bytes": selected_bytes,
        "parsed_rows": int(parsed),
        "relevant_rows": int(relevant),
        "receipt_clock_audit_samples": int(audit_n),
        "receipt_clock_audit_max_abs_error_s": float(audit_max),
    }


def _latest_known_bbo(g: pd.DataFrame, decision_s: float):
    if g is None or g.empty:
        return None
    times = g["receipt_s"].to_numpy(float)
    i = int(np.searchsorted(times, float(decision_s) + EPS, side="right") - 1)
    if i < 0:
        return None
    return g.iloc[i]


def _outcome_quote(row: pd.Series, tail: str):
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


def _m5_for_v3(raw: dict | None):
    if not raw:
        return None
    return {
        "yes_bid": float(raw["yes_bid"]),
        "yes_ask": float(raw["yes_ask"]),
        "yes_mid": float(raw["yes_mid"]),
        "bid_levels": [(float(p), float(q)) for p, q in raw.get("bid_levels", [])],
        "ask_levels": [(float(p), float(q)) for p, q in raw.get("ask_levels", [])],
        "snapshot_elapsed_s": float(raw.get("snapshot_elapsed_s", np.nan)),
        "true_m5_finalized": bool(raw.get("true_m5_finalized", False)),
    }


def _evaluate(source: Path, meta: dict, trades: dict, entry_positions: dict, bbo: pd.DataFrame, books: dict, fee_mult: dict):
    bbo_by_ticker = {
        str(t): g.sort_values("receipt_s", kind="mergesort").reset_index(drop=True)
        for t, g in bbo.groupby("ticker", sort=False)
    } if len(bbo) else {}

    rows = []
    for (ticker, tail), ent in sorted(entry_positions.items()):
        entry_qty = float(ent["entry_filled_qty"])
        if entry_qty <= EPS:
            continue
        series = str(meta[ticker].get("series") or "")
        mult = _f(fee_mult.get(series))
        bd = books.get(ticker) or {}
        eligible = bool(bd.get("coverage_eligible") and np.isfinite(mult) and mult > 0)
        m5snap = _m5_for_v3(bd.get("m5")) if eligible else None

        passive_qty = 0.0
        quote_px = np.nan
        queue0 = np.nan
        decision_available = False
        passive_full = False
        full_exit_exec = np.nan

        if bool(ent["entry_full_q10"]) and eligible:
            decision_s = float(ent["full_entry_observed_s"])
            snap = _latest_known_bbo(bbo_by_ticker.get(ticker), decision_s)
            if snap is not None:
                decision_available = True
                ob = _outcome_quote(snap, tail)
                quote_px = float(ob["ask"])
                queue0 = float(ob["ask_queue"])
                quote = {
                    "quote_price": quote_px,
                    "queue_ahead_initial": queue0,
                }
                exit_active_s = decision_s + ACTIVATION_LATENCY_MS / 1000.0
                pex = V4._simulate_passive_exit(
                    trades.get(ticker, []), tail, quote, exit_active_s, entry_qty
                )
                passive_qty = float(pex["passive_exit_qty"])
                passive_full = bool(pex["passive_exit_full"])
                full_exit_exec = _f(pex.get("full_passive_exit_exec_s"))

        residual = max(0.0, entry_qty - passive_qty)
        if residual > EPS and m5snap is not None:
            m5ex = V3._consume_m5_depth(tail, residual, m5snap, mult)
        else:
            m5ex = {
                "exit_qty": 0.0,
                "residual_qty_zero_valued": residual if residual > EPS and m5snap is None else 0.0,
                "m5_exit_proceeds": 0.0,
                "m5_taker_fee": 0.0,
                "m5_slippage_vs_best_bid": 0.0,
                "m5_cross_cost_vs_mid": np.nan,
            }

        rounding = ROUNDING_DRAG_PER_M5_CROSS if float(m5ex["exit_qty"]) > EPS else 0.0
        passive_proceeds = passive_qty * (quote_px if np.isfinite(quote_px) else 0.0)
        net = (
            passive_proceeds
            + float(m5ex["m5_exit_proceeds"])
            - ENTRY * entry_qty
            - float(m5ex["m5_taker_fee"])
            - rounding
        )

        rows.append({
            "ticker": ticker,
            "series": series,
            "close_time": str(meta[ticker].get("close_time") or ""),
            "tail": tail,
            "coverage_eligible": eligible,
            **ent,
            "decision_snapshot_available": bool(decision_available),
            "quote_c": 100.0 * quote_px if np.isfinite(quote_px) else np.nan,
            "queue_ahead_initial": queue0,
            "passive_exit_qty": passive_qty,
            "passive_exit_full": bool(passive_full),
            "seconds_full_fill_to_full_passive_exit": (
                full_exit_exec - float(ent["full_entry_observed_s"])
                if np.isfinite(full_exit_exec) and np.isfinite(_f(ent.get("full_entry_observed_s")))
                else np.nan
            ),
            "m5_required_qty": residual,
            "m5_exit_qty": float(m5ex["exit_qty"]),
            "m5_residual_zero_valued": float(m5ex["residual_qty_zero_valued"]),
            "m5_taker_fee": float(m5ex["m5_taker_fee"]),
            "m5_slippage_vs_best_bid": float(m5ex.get("m5_slippage_vs_best_bid", 0.0)),
            "m5_cross_cost_vs_mid": _f(m5ex.get("m5_cross_cost_vs_mid")),
            "rounding_drag": float(rounding),
            "net_pnl_rounding_bound": float(net),
        })

    return pd.DataFrame(rows)


def _max_drawdown(detail: pd.DataFrame) -> float:
    if detail.empty:
        return 0.0
    q = detail.sort_values(["first_entry_exec_s", "ticker", "tail"], kind="mergesort")
    pnl = pd.to_numeric(q["net_pnl_rounding_bound"], errors="coerce").fillna(0.0).to_numpy(float)
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    dd = np.r_[0.0, eq] - peak
    return float(dd.min())


def _by_asset(detail: pd.DataFrame):
    rows = []
    for series, g in detail.groupby("series", sort=True):
        entry = float(g["entry_filled_qty"].sum())
        pqty = float(g["passive_exit_qty"].sum())
        m5q = float(g["m5_exit_qty"].sum())
        resid = float(g["m5_residual_zero_valued"].sum())
        rows.append({
            "series": str(series),
            "positions": int(len(g)),
            "entry_filled_qty": entry,
            "full_q10_positions": int(g["entry_full_q10"].sum()),
            "partial_q10_positions": int(g["entry_partial_q10"].sum()),
            "passive_exit_qty": pqty,
            "passive_exit_fraction": pqty / entry if entry > EPS else np.nan,
            "m5_exit_qty": m5q,
            "terminal_exit_fraction": (pqty + m5q) / entry if entry > EPS else np.nan,
            "residual_qty_zero_valued": resid,
            "net_pnl_rounding_bound": float(g["net_pnl_rounding_bound"].sum()),
        })
    return pd.DataFrame(rows).sort_values("net_pnl_rounding_bound", ascending=False)


def run_q10_join_ask_secondary_validation(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected secondary-validation source {HARD_BOUND_SESSION}, got {source.name}")
    if hard_bind and EXPECTED_PARENT not in str(source.parent):
        raise RuntimeError(f"Expected source under {EXPECTED_PARENT}")

    required = [
        source / "book_top3_events.jsonl",
        source / "trades_event_time.jsonl",
        source / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source artifacts: " + " | ".join(missing))

    fee, fee_path, fee_provenance = OLDVAL._resolve_fee_preflight(source)
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}
    meta = V1._metadata(source)
    if not meta:
        raise RuntimeError("No valid source metadata")

    if show:
        print("=" * 160)
        print("Q10 + IMMEDIATE JOIN_ASK — SECONDARY HISTORICAL ROBUSTNESS CHECK")
        print("=" * 160)
        print("Source:", source)
        print("Frozen for this check: Q10 @ 5c, YES+NO, M1-M5, immediate fixed JOIN_ASK after full Q10 fill")
        print("Partial Q10 entry -> M5-only | 100ms entry/exit action latency | no filters | no repricing")
        print("SCIENTIFIC LABEL: secondary historical check, NOT independent validation")
        print("Fee provenance:", fee_provenance)
        print()
        print("PASS 1/3 — loading validation trades once with V11 causal clock...")

    trades, trade_clock_stats = V1._load_trades(source, meta, show=show)

    entry_positions = {}
    for ticker, m in meta.items():
        active_s = float(m["window_start_s"]) + M1_S + ACTIVATION_LATENCY_MS / 1000.0
        tr = trades.get(ticker, [])
        for tail in ("YES", "NO"):
            ent = _entry_fill_q10(tr, tail, active_s)
            if float(ent["entry_filled_qty"]) > EPS:
                entry_positions[(ticker, tail)] = ent

    filled_tickers = {k[0] for k in entry_positions}
    if not filled_tickers:
        raise RuntimeError("No Q10 entry fills on secondary-validation source")

    if show:
        full_n = sum(bool(v["entry_full_q10"]) for v in entry_positions.values())
        partial_n = sum(bool(v["entry_partial_q10"]) for v in entry_positions.values())
        print(
            f"ENTRY PRECHECK: fill events={len(entry_positions)} | full Q10={full_n} | "
            f"partial Q10={partial_n} | filled tickers={len(filled_tickers)}"
        )
        print("PASS 2/3 — mmap slicing book only around Q10-filled market windows...")

    bbo, books, book_stats = _load_relevant_books(
        source, meta, filled_tickers, show=show
    )

    if show:
        print("PASS 3/3 — replaying exactly one frozen Q10 JOIN_ASK specification...")

    detail = _evaluate(source, meta, trades, entry_positions, bbo, books, fee_mult)
    if detail.empty:
        raise RuntimeError("Secondary validation produced no position detail")

    entry_qty = float(detail["entry_filled_qty"].sum())
    passive_qty = float(detail["passive_exit_qty"].sum())
    m5_exit_qty = float(detail["m5_exit_qty"].sum())
    residual = float(detail["m5_residual_zero_valued"].sum())
    net = float(detail["net_pnl_rounding_bound"].sum())
    terminal_cov = (passive_qty + m5_exit_qty) / entry_qty if entry_qty > EPS else np.nan
    passive_frac = passive_qty / entry_qty if entry_qty > EPS else np.nan
    full_q10 = int(detail["entry_full_q10"].sum())
    partial_q10 = int(detail["entry_partial_q10"].sum())
    decision_missing = int((detail["entry_full_q10"] & ~detail["decision_snapshot_available"]).sum())
    passive_full_positions = int(detail["passive_exit_full"].sum())
    ppc = net / entry_qty if entry_qty > EPS else np.nan
    max_dd = _max_drawdown(detail)

    secs = pd.to_numeric(
        detail.loc[detail["passive_exit_full"], "seconds_full_fill_to_full_passive_exit"],
        errors="coerce",
    ).dropna()
    median_exit_s = float(secs.median()) if len(secs) else np.nan

    closes = pd.to_datetime(detail["close_time"], utc=True, errors="coerce").dropna()
    span_h = float((closes.max() - closes.min()).total_seconds() / 3600.0 + 0.25) if len(closes) >= 2 else np.nan
    pnl_per_24h = net * 24.0 / span_h if np.isfinite(span_h) and span_h > 0 else np.nan

    perpos = pd.to_numeric(detail["net_pnl_rounding_bound"], errors="coerce").fillna(0.0).sort_values(ascending=False)
    top1 = float(perpos.head(1).sum()) if len(perpos) else 0.0
    top5 = float(perpos.head(5).sum()) if len(perpos) else 0.0
    top1_share = top1 / net if net > EPS else np.nan
    top5_share = top5 / net if net > EPS else np.nan

    by_asset = _by_asset(detail)

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "research_stage": "SECONDARY_HISTORICAL_ROBUSTNESS_NOT_INDEPENDENT_VALIDATION",
        "source_session": str(source),
        "scientific_label": (
            "NOT independent validation: this 15h realization was opened before JOIN_ASK/Q10 were selected on the 24h development sample"
        ),
        "frozen_for_this_check": {
            "entry_c": ENTRY_C,
            "qty_per_tail": QTY,
            "sides": ["BUY_YES", "BUY_NO"],
            "posting_window": "M1_TO_M5",
            "activation_latency_ms": ACTIVATION_LATENCY_MS,
            "entry_fill_rule": "strict-through seller flow only; exact 5c excluded; partial fills allowed",
            "full_entry_exit_rule": "latest known outcome ask at full Q10 fill observation; fixed passive JOIN_ASK; +100ms activation; V4 queue model",
            "partial_entry_exit_rule": "M5_ONLY",
            "m5_rule": "consume recorded top-3 outcome bids; deeper residual valued zero",
            "asset_filters": None,
            "repricing": False,
        },
        "development_context": {
            "source_session": DEV_SOURCE_SESSION,
            "q10_net_pnl": DEV_Q10_NET,
            "q10_terminal_coverage": DEV_Q10_TERMINAL_COVERAGE,
            "q10_passive_exit_fraction": DEV_Q10_PASSIVE_FRACTION,
            "q10_max_drawdown": DEV_Q10_MAX_DRAWDOWN,
        },
        "fee_provenance": fee_provenance,
        "fee_artifact": str(fee_path),
        "trade_clock_stats": trade_clock_stats,
        "book_fast_path": book_stats,
        "entry_fill_events": int(len(detail)),
        "full_q10_entry_positions": full_q10,
        "partial_q10_entry_positions": partial_q10,
        "entry_filled_qty": entry_qty,
        "decision_snapshot_missing_full_entries": decision_missing,
        "passive_exit_qty": passive_qty,
        "passive_exit_fraction_of_entry_qty": passive_frac,
        "full_passive_exit_positions": passive_full_positions,
        "median_seconds_full_fill_to_full_passive_exit": median_exit_s,
        "m5_exit_qty": m5_exit_qty,
        "terminal_exit_fraction_of_entry_qty": terminal_cov,
        "residual_qty_zero_valued": residual,
        "m5_taker_fees": float(detail["m5_taker_fee"].sum()),
        "rounding_drag": float(detail["rounding_drag"].sum()),
        "net_pnl_rounding_bound": net,
        "net_pnl_per_filled_contract": ppc,
        "max_drawdown_rounding_bound": max_dd,
        "span_hours_from_filled_close_times": span_h,
        "normalized_net_pnl_per_24h_diagnostic": pnl_per_24h,
        "top1_positive_pnl": top1,
        "top5_positive_pnl": top5,
        "top1_share_of_net": top1_share,
        "top5_share_of_net": top5_share,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "guardrail": (
            "Secondary historical check only. Do not alter entry, Q10 size, assets, latency, or exit mechanics based on this output. Fresh forward/live resting-order evidence is still required."
        ),
    }

    out = _new_output(source.name)
    summary["output_dir"] = str(out)
    OOS._atomic_json(out / "summary.json", summary)
    detail.to_csv(out / "q10_join_ask_secondary_detail.csv", index=False)
    by_asset.to_csv(out / "q10_join_ask_secondary_by_asset.csv", index=False)

    if show:
        print("=" * 160)
        print("Q10 JOIN_ASK — SECONDARY CHECK RESULT")
        print("=" * 160)
        print(f"Fill events:                 {len(detail)}")
        print(f"Full Q10 / partial:          {full_q10} / {partial_q10}")
        print(f"Entry filled qty:            {entry_qty:.2f}")
        print(f"Missing full-fill decisions: {decision_missing}")
        print(f"Passive exit qty:            {passive_qty:.2f} ({100.0*passive_frac:.2f}%)")
        print(f"Full passive exits:          {passive_full_positions}")
        print(f"Median passive full-exit:    {median_exit_s:.2f}s" if np.isfinite(median_exit_s) else "Median passive full-exit:    NA")
        print(f"M5 exit qty:                 {m5_exit_qty:.2f}")
        print(f"Terminal execution coverage: {100.0*terminal_cov:.2f}%")
        print(f"Unproven residual:           {residual:.2f}")
        print(f"Net PnL:                     ${net:+.5f}")
        print(f"PnL / filled contract:       ${ppc:+.5f}")
        print(f"Max drawdown:                ${max_dd:+.5f}")
        print(f"Observed span:               {span_h:.2f}h" if np.isfinite(span_h) else "Observed span:               NA")
        print(f"24h-normalized diagnostic:   ${pnl_per_24h:+.5f}" if np.isfinite(pnl_per_24h) else "24h-normalized diagnostic:   NA")
        print(f"Top-1 share of net:          {100.0*top1_share:.1f}%" if np.isfinite(top1_share) else "Top-1 share of net:          NA")
        print(f"Top-5 share of net:          {100.0*top5_share:.1f}%" if np.isfinite(top5_share) else "Top-5 share of net:          NA")
        print()
        print("DEVELOPMENT Q10 CONTEXT (not a target):")
        print(f"  net=${DEV_Q10_NET:+.5f} | terminal={100.0*DEV_Q10_TERMINAL_COVERAGE:.2f}% | passive={100.0*DEV_Q10_PASSIVE_FRACTION:.2f}% | DD=${DEV_Q10_MAX_DRAWDOWN:+.5f}")
        print()
        print("SCIENTIFIC LABEL: SECONDARY HISTORICAL CHECK — NOT INDEPENDENT VALIDATION")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "detail": detail,
        "by_asset": by_asset,
        "output_dir": str(out),
    }
