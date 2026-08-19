from __future__ import annotations

"""Held-out historical replay suite for frozen 5c/Q10 + immediate JOIN_ASK.

Scientific role
---------------
The deep-tail entry, immediate JOIN_ASK exit and Q10 working size were selected using
other realizations.  This module is hard-bound to older live-recorder raw captures that
were not used to tune this deep-tail strategy.  They are therefore useful held-out
historical replays for the hypothesis.

This is still NOT live execution validation: the deep-tail 5c orders were not actually
resting in these captures.  Entry and exit fills are reconstructed conservatively from
public trade flow and recorded top-3 books.

Frozen strategy
---------------
- M1: rest BUY YES Q10 @ 5c and BUY NO Q10 @ 5c.
- Entry activation M1 + 100ms.
- Entry evidence: same-outcome aggressive seller flow STRICTLY THROUGH 5c only;
  exact 5c prints excluded because deep queue ahead is unobserved.
- Full Q10: once locally observable, latest locally-known outcome BBO -> fixed passive
  SELL Q10 at current best ask, active 100ms later, no repricing.
- Passive exit queue model: displayed L1 queue ahead; exact-price aggressive buyer flow
  burns queue first; strict trade-through fills all residual; no cancellation-ahead credit.
- Partial Q10: M5-only.
- M5 residual: consume recorded top-3 outcome bids. Deeper residual gets ZERO value.
- No asset filters, threshold sweeps, quantity sweeps, or parameter changes.

Data-quality policy
-------------------
The live raw files can be receipt-order disordered.  Book selection therefore NEVER uses
receipt-time byte slicing.  After entry precheck identifies filled tickers, native rg/grep
scans the whole raw book once for those exact ticker strings; Python parses only the small
filtered file.  Within each ticker, states are sorted by recorder elapsed_s.

A ticker is replay-eligible only if:
- >=95% M1-M5 valid-book coverage; and
- a genuine valid book event at/after M5 exists to finalize the last valid pre-M5 state.

Unlike the earlier convenience replay, this module NEVER manufactures an M5 snapshot for
an interrupted terminal window.  Full-entry positions with no actionable decision BBO are
also data-quality excluded from primary economics and counted explicitly.

Decision label
--------------
Per realization, >=10 replay-eligible fill events are required for a directional verdict,
matching the earlier validation sample-size convention.  With fewer than 10, status is
INCONCLUSIVE_LOW_COUNT regardless of PnL.  Otherwise positive PnL and >=95% terminal
execution coverage are required for PASS_HISTORICAL_REPLAY.

READ ONLY source. NO API. NO ORDERS.
"""

from collections import defaultdict
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_passive_exit_dev_v4 as V4
from . import mm_deep_tail_capacity_dev_v3 as V3
from . import mm_deep_tail_trailing_passive_exit_dev_v6_2 as V62
from . import mm_deep_tail_trailing_passive_exit_dev_v6_3 as V63
from . import mm_deep_tail_join_ask_q10_secondary_validation_v1 as BASE

VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q10_HELDOUT_SUITE_V1"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_join_ask_q10_heldout_suite_v1"

ENTRY = 0.05
QTY = 10.0
M1_S = 60.0
M5_S = 300.0
WINDOW_S = M5_S - M1_S
ACTIVATION_LATENCY_MS = 100.0
MIN_BOOK_COVERAGE_FRAC = 0.95
MIN_FILL_EVENTS_FOR_DIRECTIONAL_VERDICT = 10
MIN_TERMINAL_COVERAGE_FOR_PASS = 0.95
EPS = 1e-10

# Historical raw captures held out with respect to the later deep-tail/JOIN_ASK selection.
# V11 is the primary held-out realization.  V12.2 is a much shorter supplemental capture.
HARD_BOUND = {
    "V11_PRIMARY": {
        "session": "20260818_092428_live_q10_24h_v11",
        "parent": "live_cycle_q10_v1",
    },
    "V12_2_SUPPLEMENT": {
        "session": "20260818_212037_live_q5_1h_v12_2",
        "parent": "live_cycle_q10_v1",
    },
}


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


def _fee_from_live_session(session: Path):
    p = Path(session) / "fee_preflight.json"
    if not p.exists():
        raise FileNotFoundError(f"Held-out live fee_preflight missing: {p}")
    fee = OOS._read_json(p, {}) or {}
    if not fee.get("ok"):
        raise RuntimeError(f"Held-out live fee_preflight is not PASS: {p}")
    return fee, p.resolve()


def _entry_positions(meta: dict, trades: dict):
    out = {}
    for ticker, m in meta.items():
        active_s = float(m["window_start_s"]) + M1_S + ACTIVATION_LATENCY_MS / 1000.0
        tr = trades.get(ticker, [])
        for tail in ("YES", "NO"):
            ent = BASE._entry_fill_q10(tr, tail, active_s)
            if float(ent["entry_filled_qty"]) > EPS:
                out[(ticker, tail)] = ent
    return out


def _book_filter_path(out: Path, session_name: str):
    d = out / "cache" / session_name
    d.mkdir(parents=True, exist_ok=True)
    return d / "book_filled_tickers.filtered.jsonl"


def _load_books_disorder_safe(raw: Path, meta: dict, tickers: set[str], out: Path, *, show=True):
    """Full-file native ticker filter, then exact per-ticker replay. No time-slice assumption."""
    raw_path = Path(raw) / "book_top3_events.jsonl"
    filtered = _book_filter_path(out, raw.parent.name)

    if not filtered.exists() or filtered.stat().st_size <= 0:
        if show:
            print(
                f"BOOK SAFE FILTER: full raw scan by native rg/grep | "
                f"raw={raw_path.stat().st_size/(1024**3):.2f} GiB | tickers={len(tickers)}"
            )
        V62._native_filter(raw_path, filtered, set(tickers), show=show)
    elif show:
        print(
            f"BOOK SAFE FILTER: reusing materialized filter | "
            f"{filtered.stat().st_size/(1024**2):.1f} MiB"
        )

    valid_events = defaultdict(list)
    bbo_rows = []
    parsed = 0
    relevant = 0
    audit_n = 0
    audit_max = 0.0

    with filtered.open("rb", buffering=8 * 1024 * 1024) as fh:
        for raw_line in fh:
            parsed += 1
            try:
                r = V63._loads(raw_line)
            except Exception:
                continue
            ticker = str(r.get("ticker") or "")
            if ticker not in tickers or ticker not in meta:
                continue
            relevant += 1
            e = _f(r.get("elapsed_s"))
            if not np.isfinite(e):
                continue
            ws = float(meta[ticker]["window_start_s"])
            rt = ws + float(e)

            if audit_n < 200:
                rs = OOS._ts(r.get("receipt_time"))
                if np.isfinite(rs):
                    audit_n += 1
                    audit_max = max(audit_max, abs(float(rs) - rt))

            state = BASE._parse_book_state(r)
            if state is None:
                continue
            valid_events[ticker].append((float(e), float(rt), state))

            # Include M0-M5 so a full fill immediately after M1 can use the latest
            # locally-known BBO even if the last update arrived just before M1.
            if 0.0 <= e < M5_S:
                bbo_rows.append((
                    ticker, float(rt), float(e),
                    state["yes_bid"], state["yes_ask"], state["yes_mid"],
                    state["bid_p1"], state["bid_q1"], state["bid_p2"], state["bid_q2"], state["bid_p3"], state["bid_q3"],
                    state["ask_p1"], state["ask_q1"], state["ask_p2"], state["ask_q2"], state["ask_p3"], state["ask_q3"],
                ))

    if audit_n and audit_max > 0.001 + EPS:
        raise RuntimeError(f"Held-out book receipt-clock audit failed: max error={audit_max:.6f}s")

    cols = [
        "ticker", "receipt_s", "elapsed_s", "yes_bid", "yes_ask", "yes_mid",
        "bid_p1", "bid_q1", "bid_p2", "bid_q2", "bid_p3", "bid_q3",
        "ask_p1", "ask_q1", "ask_p2", "ask_q2", "ask_p3", "ask_q3",
    ]
    bbo = pd.DataFrame.from_records(bbo_rows, columns=cols)
    if len(bbo):
        bbo.sort_values(["ticker", "receipt_s"], kind="mergesort", inplace=True)
        bbo.reset_index(drop=True, inplace=True)

    books = {}
    for ticker in sorted(tickers):
        ev = sorted(valid_events.get(ticker, []), key=lambda z: (z[0], z[1]))
        coverage = 0.0
        last_state = None
        last_e = None
        m5_state = None
        finalized = False

        for e, rt, state in ev:
            if last_e is not None and last_e < M5_S:
                a = max(M1_S, float(last_e))
                b = min(M5_S, float(e))
                if b > a and last_state is not None:
                    coverage += b - a

            if e >= M5_S:
                # Genuine post-M5 valid event proves the preceding state is the final
                # recorded executable state before M5. Never synthesize this on EOF.
                if last_state is not None and last_e is not None and last_e < M5_S:
                    m5_state = dict(last_state)
                    m5_state["snapshot_elapsed_s"] = float(last_e)
                    m5_state["true_m5_finalized"] = True
                    finalized = True
                break

            last_e = float(e)
            last_state = state

        cov = min(WINDOW_S, max(0.0, float(coverage)))
        books[ticker] = {
            "book_covered_seconds": cov,
            "book_coverage_fraction": cov / WINDOW_S,
            "coverage_eligible": bool(
                finalized
                and m5_state is not None
                and cov >= MIN_BOOK_COVERAGE_FRAC * WINDOW_S - EPS
            ),
            "m5": m5_state,
            "true_m5_finalized": bool(finalized),
        }

    if show:
        eligible = sum(bool(v["coverage_eligible"]) for v in books.values())
        incomplete = sum(not bool(v["true_m5_finalized"]) for v in books.values())
        print(
            f"BOOK SAFE PARSE DONE | filtered rows={parsed:,} | relevant={relevant:,} | "
            f"BBO={len(bbo):,} | eligible={eligible}/{len(tickers)} | "
            f"no-true-M5={incomplete} | audit={audit_max:.9f}s"
        )

    return bbo, books, {
        "raw_book_size_bytes": int(raw_path.stat().st_size),
        "filtered_book_size_bytes": int(filtered.stat().st_size),
        "filtered_rows": int(parsed),
        "relevant_rows": int(relevant),
        "eligible_filled_tickers": int(sum(bool(v["coverage_eligible"]) for v in books.values())),
        "filled_tickers_without_true_m5": int(sum(not bool(v["true_m5_finalized"]) for v in books.values())),
        "receipt_clock_audit_samples": int(audit_n),
        "receipt_clock_audit_max_abs_error_s": float(audit_max),
        "selection": "full raw book native ticker filter; no receipt-order slicing",
    }


def _position_decision_available(ticker: str, ent: dict, bbo_by_ticker: dict):
    if not bool(ent.get("entry_full_q10")):
        return True
    d = _f(ent.get("full_entry_observed_s"))
    if not np.isfinite(d):
        return False
    g = bbo_by_ticker.get(ticker)
    return BASE._latest_known_bbo(g, d) is not None if g is not None else False


def _status(fill_events: int, net: float, terminal_cov: float):
    if int(fill_events) < MIN_FILL_EVENTS_FOR_DIRECTIONAL_VERDICT:
        return "INCONCLUSIVE_LOW_COUNT"
    if not np.isfinite(net) or net <= 0.0:
        return "FAIL_NONPOSITIVE_PNL"
    if not np.isfinite(terminal_cov) or terminal_cov < MIN_TERMINAL_COVERAGE_FOR_PASS - EPS:
        return "FAIL_TERMINAL_EXECUTION_COVERAGE"
    return "PASS_HISTORICAL_REPLAY"


def _summarize_detail(detail: pd.DataFrame):
    if detail.empty:
        return {
            "fill_events": 0,
            "entry_filled_qty": 0.0,
            "passive_exit_qty": 0.0,
            "passive_exit_fraction": np.nan,
            "m5_exit_qty": 0.0,
            "terminal_coverage": np.nan,
            "residual_qty_zero_valued": 0.0,
            "net_pnl": 0.0,
            "pnl_per_filled_contract": np.nan,
            "max_drawdown": 0.0,
            "full_passive_exit_positions": 0,
            "median_passive_exit_s": np.nan,
            "top1_share": np.nan,
            "top5_share": np.nan,
            "span_hours": np.nan,
            "normalized_24h": np.nan,
        }

    entry = float(detail["entry_filled_qty"].sum())
    passive = float(detail["passive_exit_qty"].sum())
    m5 = float(detail["m5_exit_qty"].sum())
    residual = float(detail["m5_residual_zero_valued"].sum())
    net = float(detail["net_pnl_rounding_bound"].sum())
    terminal = (passive + m5) / entry if entry > EPS else np.nan
    passive_frac = passive / entry if entry > EPS else np.nan
    ppc = net / entry if entry > EPS else np.nan
    dd = BASE._max_drawdown(detail)

    secs = pd.to_numeric(
        detail.loc[detail["passive_exit_full"], "seconds_full_fill_to_full_passive_exit"],
        errors="coerce",
    ).dropna()
    median_s = float(secs.median()) if len(secs) else np.nan

    p = pd.to_numeric(detail["net_pnl_rounding_bound"], errors="coerce").fillna(0.0).sort_values(ascending=False)
    top1 = float(p.head(1).sum()) if len(p) else 0.0
    top5 = float(p.head(5).sum()) if len(p) else 0.0

    closes = pd.to_datetime(detail["close_time"], utc=True, errors="coerce").dropna()
    span_h = float((closes.max() - closes.min()).total_seconds() / 3600.0 + 0.25) if len(closes) >= 2 else np.nan
    norm = net * 24.0 / span_h if np.isfinite(span_h) and span_h > 0 else np.nan

    return {
        "fill_events": int(len(detail)),
        "entry_filled_qty": entry,
        "passive_exit_qty": passive,
        "passive_exit_fraction": passive_frac,
        "m5_exit_qty": m5,
        "terminal_coverage": terminal,
        "residual_qty_zero_valued": residual,
        "net_pnl": net,
        "pnl_per_filled_contract": ppc,
        "max_drawdown": float(dd),
        "full_passive_exit_positions": int(detail["passive_exit_full"].sum()),
        "median_passive_exit_s": median_s,
        "top1_share": top1 / net if net > EPS else np.nan,
        "top5_share": top5 / net if net > EPS else np.nan,
        "span_hours": span_h,
        "normalized_24h": norm,
    }


def _by_asset(detail: pd.DataFrame):
    if detail.empty:
        return pd.DataFrame()
    return BASE._by_asset(detail)


def run_one_heldout(source_session, *, label="HELDOUT", hard_bind=True, output_root=None, show=True):
    session = Path(source_session).resolve()
    raw = session / "raw_capture"

    if hard_bind:
        allowed = {v["session"] for v in HARD_BOUND.values()}
        if session.name not in allowed:
            raise RuntimeError(f"Held-out session is not hard-bound: {session.name}")
        cfg = next(v for v in HARD_BOUND.values() if v["session"] == session.name)
        if cfg["parent"] not in str(session.parent):
            raise RuntimeError(f"Expected held-out session under {cfg['parent']}")

    required = [
        raw / "book_top3_events.jsonl",
        raw / "trades_event_time.jsonl",
        raw / "market_metadata.jsonl",
        session / "fee_preflight.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing held-out artifacts: " + " | ".join(missing))

    fee, fee_path = _fee_from_live_session(session)
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}
    meta = V1._metadata(raw)
    if not meta:
        raise RuntimeError(f"No held-out raw metadata: {raw}")

    if show:
        print("=" * 150)
        print(f"HELD-OUT Q10 JOIN_ASK REPLAY — {label}")
        print("=" * 150)
        print("Live session:", session)
        print("Raw source:", raw)
        print("Frozen: Q10 @ 5c both tails | M1-M5 | full fill -> immediate fixed JOIN_ASK")
        print("No parameter/asset/size sweep. Exact 5c entry prints excluded.")
        print("Book path: disorder-safe full-file ticker filter; incomplete M5 windows are NOT manufactured.")
        print("SCIENTIFIC LABEL: held-out historical replay for deep-tail selection; NOT actual deep-tail live fill validation")
        print()
        print("PASS 1/3 — loading raw M1-M5 trades with frozen causal clock...")

    trades, trade_stats = V1._load_trades(raw, meta, show=show)
    entries = _entry_positions(meta, trades)
    filled_tickers = {k[0] for k in entries}

    if show:
        print(
            f"ENTRY PRECHECK: fill events={len(entries)} | "
            f"full Q10={sum(bool(x['entry_full_q10']) for x in entries.values())} | "
            f"partial={sum(bool(x['entry_partial_q10']) for x in entries.values())} | "
            f"tickers={len(filled_tickers)}"
        )

    if not entries:
        return {
            "label": label,
            "session": str(session),
            "status": "INCONCLUSIVE_NO_FILLS",
            "summary": _summarize_detail(pd.DataFrame()),
            "detail": pd.DataFrame(),
            "by_asset": pd.DataFrame(),
            "dq": {"precheck_fill_events": 0},
        }

    out_base = Path(output_root) if output_root is not None else _new_output(session.name)
    out_base.mkdir(parents=True, exist_ok=True)

    if show:
        print("PASS 2/3 — extracting books for filled tickers with disorder-safe native filter...")
    bbo, books, book_stats = _load_books_disorder_safe(raw, meta, filled_tickers, out_base, show=show)

    bbo_by_ticker = {
        str(t): g.sort_values("receipt_s", kind="mergesort").reset_index(drop=True)
        for t, g in bbo.groupby("ticker", sort=False)
    } if len(bbo) else {}

    eligible_entries = {}
    excluded_no_m5 = []
    excluded_low_coverage = []
    excluded_no_decision = []

    for key, ent in entries.items():
        ticker, tail = key
        bd = books.get(ticker) or {}
        if not bool(bd.get("true_m5_finalized")):
            excluded_no_m5.append(key)
            continue
        if not bool(bd.get("coverage_eligible")):
            excluded_low_coverage.append(key)
            continue
        if not _position_decision_available(ticker, ent, bbo_by_ticker):
            excluded_no_decision.append(key)
            continue
        eligible_entries[key] = ent

    if show:
        print(
            f"DQ GATE: replay-eligible positions={len(eligible_entries)}/{len(entries)} | "
            f"no true M5={len(excluded_no_m5)} | low coverage={len(excluded_low_coverage)} | "
            f"missing full-fill decision={len(excluded_no_decision)}"
        )
        print("PASS 3/3 — replaying the one frozen Q10 JOIN_ASK specification...")

    if eligible_entries:
        detail = BASE._evaluate(raw, meta, trades, eligible_entries, bbo, books, fee_mult)
    else:
        detail = pd.DataFrame()

    if len(detail):
        detail.insert(0, "heldout_label", str(label))
        detail.insert(1, "source_session", session.name)

    sm = _summarize_detail(detail)
    status = _status(sm["fill_events"], sm["net_pnl"], sm["terminal_coverage"])
    by_asset = _by_asset(detail)

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "heldout_label": label,
        "source_session": str(session),
        "raw_source": str(raw),
        "scientific_label": "HELDOUT_HISTORICAL_REPLAY_FOR_DEEP_TAIL_SELECTION_NOT_LIVE_FILL_VALIDATION",
        "frozen_spec": {
            "entry_c": 5.0,
            "qty_per_tail": QTY,
            "posting_window": "M1_TO_M5",
            "entry_activation_latency_ms": ACTIVATION_LATENCY_MS,
            "entry_fill": "strict-through seller flow only; exact 5c excluded",
            "full_entry_exit": "latest locally-known ask -> fixed JOIN_ASK; +100ms activation; V4 queue model",
            "partial_entry_exit": "M5_ONLY",
            "m5": "recorded top3 only; deeper residual zero",
            "asset_filters": None,
            "repricing": False,
        },
        "status": status,
        "minimum_fill_events_for_directional_verdict": MIN_FILL_EVENTS_FOR_DIRECTIONAL_VERDICT,
        "minimum_terminal_coverage_for_pass": MIN_TERMINAL_COVERAGE_FOR_PASS,
        "precheck_entry_fill_events": int(len(entries)),
        "precheck_full_q10": int(sum(bool(x["entry_full_q10"]) for x in entries.values())),
        "precheck_partial_q10": int(sum(bool(x["entry_partial_q10"]) for x in entries.values())),
        "dq_excluded_no_true_m5_positions": int(len(excluded_no_m5)),
        "dq_excluded_low_book_coverage_positions": int(len(excluded_low_coverage)),
        "dq_excluded_missing_decision_positions": int(len(excluded_no_decision)),
        "replay_eligible_fill_events": int(sm["fill_events"]),
        "entry_filled_qty": sm["entry_filled_qty"],
        "passive_exit_qty": sm["passive_exit_qty"],
        "passive_exit_fraction_of_entry_qty": sm["passive_exit_fraction"],
        "m5_exit_qty": sm["m5_exit_qty"],
        "terminal_exit_fraction_of_entry_qty": sm["terminal_coverage"],
        "residual_qty_zero_valued": sm["residual_qty_zero_valued"],
        "net_pnl_rounding_bound": sm["net_pnl"],
        "net_pnl_per_filled_contract": sm["pnl_per_filled_contract"],
        "max_drawdown_rounding_bound": sm["max_drawdown"],
        "full_passive_exit_positions": sm["full_passive_exit_positions"],
        "median_seconds_full_fill_to_full_passive_exit": sm["median_passive_exit_s"],
        "top1_share_of_net": sm["top1_share"],
        "top5_share_of_net": sm["top5_share"],
        "span_hours_from_filled_close_times": sm["span_hours"],
        "normalized_net_pnl_per_24h_diagnostic": sm["normalized_24h"],
        "trade_clock_stats": trade_stats,
        "book_stats": book_stats,
        "fee_artifact": str(fee_path),
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "guardrail": "Do not retune Q10, 5c entry, assets, latency, or JOIN_ASK mechanics from this held-out result.",
    }

    OOS._atomic_json(out_base / f"summary_{label}.json", summary)
    if len(detail):
        detail.to_csv(out_base / f"detail_{label}.csv", index=False)
    if len(by_asset):
        by_asset.to_csv(out_base / f"by_asset_{label}.csv", index=False)

    if show:
        print("-" * 90)
        print(f"{label} RESULT: {status}")
        print(f"Replay-eligible fills:       {sm['fill_events']}  (precheck {len(entries)})")
        print(f"Entered contracts:           {sm['entry_filled_qty']:.2f}")
        print(f"Passive exits:               {sm['passive_exit_qty']:.2f} ({100*sm['passive_exit_fraction']:.2f}%)" if np.isfinite(sm["passive_exit_fraction"]) else "Passive exits:               NA")
        print(f"Terminal coverage:           {100*sm['terminal_coverage']:.2f}%" if np.isfinite(sm["terminal_coverage"]) else "Terminal coverage:           NA")
        print(f"Unproven residual:           {sm['residual_qty_zero_valued']:.2f}")
        print(f"Net PnL:                     ${sm['net_pnl']:+.5f}")
        print(f"PnL / filled contract:       ${sm['pnl_per_filled_contract']:+.5f}" if np.isfinite(sm["pnl_per_filled_contract"]) else "PnL / filled contract:       NA")
        print(f"Max drawdown:                ${sm['max_drawdown']:+.5f}")
        print(f"24h-normalized diagnostic:   ${sm['normalized_24h']:+.5f}" if np.isfinite(sm["normalized_24h"]) else "24h-normalized diagnostic:   NA")
        print(f"Top-1 share of net:          {100*sm['top1_share']:.1f}%" if np.isfinite(sm["top1_share"]) else "Top-1 share of net:          NA")
        print("-" * 90)

    return {
        "label": label,
        "session": str(session),
        "status": status,
        "summary": summary,
        "detail": detail,
        "by_asset": by_asset,
        "output_dir": str(out_base),
    }


def _pooled_summary(results):
    frames = [r["detail"] for r in results if isinstance(r.get("detail"), pd.DataFrame) and len(r["detail"])]
    if not frames:
        return {}, pd.DataFrame(), pd.DataFrame()
    detail = pd.concat(frames, ignore_index=True, sort=False)
    sm = _summarize_detail(detail)
    status = _status(sm["fill_events"], sm["net_pnl"], sm["terminal_coverage"])
    by_asset = _by_asset(detail)
    return {
        "status": status,
        "fill_events": sm["fill_events"],
        "entry_filled_qty": sm["entry_filled_qty"],
        "passive_exit_qty": sm["passive_exit_qty"],
        "passive_exit_fraction": sm["passive_exit_fraction"],
        "terminal_coverage": sm["terminal_coverage"],
        "residual_qty_zero_valued": sm["residual_qty_zero_valued"],
        "net_pnl": sm["net_pnl"],
        "pnl_per_filled_contract": sm["pnl_per_filled_contract"],
        "max_drawdown": sm["max_drawdown"],
        "top1_share": sm["top1_share"],
        "top5_share": sm["top5_share"],
        "note": "Pooled historical replay diagnostic across hard-bound held-out raw captures; not live execution validation.",
    }, detail, by_asset


def run_heldout_suite(primary_session, supplemental_session=None, *, hard_bind=True, show=True):
    suite_out = _new_output("heldout_suite")
    results = []

    results.append(run_one_heldout(
        primary_session,
        label="V11_PRIMARY",
        hard_bind=hard_bind,
        output_root=suite_out,
        show=show,
    ))

    if supplemental_session is not None and Path(supplemental_session).exists():
        results.append(run_one_heldout(
            supplemental_session,
            label="V12_2_SUPPLEMENT",
            hard_bind=hard_bind,
            output_root=suite_out,
            show=show,
        ))

    pooled, pooled_detail, pooled_asset = _pooled_summary(results)
    if pooled:
        OOS._atomic_json(suite_out / "pooled_summary.json", pooled)
        pooled_detail.to_csv(suite_out / "pooled_detail.csv", index=False)
        if len(pooled_asset):
            pooled_asset.to_csv(suite_out / "pooled_by_asset.csv", index=False)

    if show:
        print("=" * 100)
        print("HELD-OUT SUITE — CLEAN SUMMARY")
        print("=" * 100)
        for r in results:
            s = r.get("summary") or {}
            if "net_pnl_rounding_bound" not in s:
                print(f"{r['label']}: {r['status']}")
                continue
            print(
                f"{r['label']:18s} | {r['status']:28s} | fills {int(s['replay_eligible_fill_events']):2d} | "
                f"PnL ${float(s['net_pnl_rounding_bound']):+7.3f} | "
                f"edge {100*float(s['net_pnl_per_filled_contract']):+5.2f}c | "
                f"passive {100*float(s['passive_exit_fraction_of_entry_qty']):5.1f}% | "
                f"terminal {100*float(s['terminal_exit_fraction_of_entry_qty']):5.1f}% | "
                f"DD ${float(s['max_drawdown_rounding_bound']):+6.3f}"
            )
        if pooled:
            print("-" * 100)
            print(
                f"POOLED             | {pooled['status']:28s} | fills {pooled['fill_events']:2d} | "
                f"PnL ${pooled['net_pnl']:+7.3f} | edge {100*pooled['pnl_per_filled_contract']:+5.2f}c | "
                f"passive {100*pooled['passive_exit_fraction']:5.1f}% | terminal {100*pooled['terminal_coverage']:5.1f}% | "
                f"DD ${pooled['max_drawdown']:+6.3f}"
            )
        print()
        print("Interpretation guardrail: held-out historical public-data replay, not proof of actual resting-order fills.")
        print("NO API CALLED | NO ORDERS SENT | NO RETUNING")
        print("Output:", suite_out)

    return {
        "version": VERSION,
        "results": results,
        "pooled_summary": pooled,
        "pooled_detail": pooled_detail,
        "pooled_by_asset": pooled_asset,
        "output_dir": str(suite_out),
    }
