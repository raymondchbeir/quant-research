from __future__ import annotations

"""Streaming dual-clock causality-correct replay for the formal Q10 OOS session.

This is the Q10 / 24-hour analogue of ``mm_cycle_q5_dual_clock_causal_replay_v10``.
It is deliberately hard-bound to the untouched formal OOS capture
``20260817_064143`` and changes NO Candidate-C strategy parameter.

Clocks
------
- BOOK strategy decisions: local receipt_time (when information was actionable).
- PASSIVE execution: public trade exchange_time (when execution became economic).
- FILL observation: public trade receipt_time (proxy for when the strategy learns).

The frozen displayed-L1 FIFO model, Q10 size, Candidate-C spread/depth rule,
inventory exit rule, M5 liquidation and fee model are unchanged.

Scientific status
-----------------
This execution-model correction was motivated by later live forensics, so the old
24h session is no longer independent validation for the corrected simulator.  It
is a read-only historical robustness/forensic replay.  A fresh forward OOS sample
is required if the corrected model remains promising.

Implementation note
-------------------
The raw OOS capture contains millions of book rows.  To avoid materializing the
whole event stream in RAM, book receipts are streamed.  Trade rows are also
streamed with a bounded look-ahead derived from a first-pass measurement of the
maximum exchange->receipt lag plus receipt-file disorder.  Each trade row remains
the same Python object from economic execution through local observation, which
preserves V10's hidden-fill accounting exactly.

NO exchange/API calls. NO real orders. Source session is never modified.
"""

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q5_dual_clock_causal_replay_v10 as V10

VERSION = "MM_CYCLE_Q10_OOS_DUAL_CLOCK_CAUSAL_REPLAY_V11"
HARD_BOUND_OOS_SESSION = "20260817_064143"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q10_oos_dual_clock_causal_replay_v11"
QTY = float(OOS.QUOTE_SIZE)
EPS = 1e-10
LOOKAHEAD_PAD_S = 0.002


def _f(x, default=np.nan):
    return OOS._f(x, default)


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


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _receipt_s(row):
    t = OOS._ts((row or {}).get("receipt_time"))
    return float(t) if np.isfinite(t) else np.nan


def _raw_exchange_s(row):
    t = OOS._ts((row or {}).get("exchange_time"))
    if np.isfinite(t):
        return float(t)
    z = _f((row or {}).get("ts_ms"))
    if np.isfinite(z):
        return float(z) / 1000.0
    return np.nan


def _causal_exec_s(row, receipt_s):
    """Use exchange time unless it is locally impossible (after receipt).

    A tiny positive exchange-after-receipt discrepancy is clock/schema skew, not
    evidence that an event was observed before it existed.  Such rows are clamped
    to receipt time and counted explicitly in the scan diagnostics.
    """
    x = _raw_exchange_s(row)
    if not np.isfinite(x):
        return float(receipt_s), "RECEIPT_FALLBACK"
    if x > float(receipt_s) + EPS:
        return float(receipt_s), "CLAMP_EXCHANGE_AFTER_RECEIPT"
    return float(x), "EXCHANGE_TIME"


def _load_metadata(source: Path):
    rows = []
    by_ticker = {}
    for row in _iter_jsonl(source / "market_metadata.jsonl"):
        ticker = str(row.get("ticker") or "")
        if not ticker:
            continue
        rows.append(row)
        by_ticker[ticker] = row
    return rows, by_ticker


def _scan_trade_timing(path: Path, selected_tickers: set[str], *, show=True):
    stats = Counter()
    lags_ms = []
    max_lag_s = 0.0
    max_receipt_seen = -np.inf
    max_receipt_disorder_s = 0.0

    for n, row in enumerate(_iter_jsonl(path), start=1):
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        rt = _receipt_s(row)
        if not np.isfinite(rt):
            stats["missing_receipt"] += 1
            continue

        if np.isfinite(max_receipt_seen):
            max_receipt_disorder_s = max(
                max_receipt_disorder_s,
                max(0.0, float(max_receipt_seen) - float(rt)),
            )
        max_receipt_seen = max(float(max_receipt_seen), float(rt))

        raw_x = _raw_exchange_s(row)
        exec_s, source = _causal_exec_s(row, rt)
        stats["trade_rows"] += 1
        stats[f"execution_source_{source}"] += 1
        if np.isfinite(raw_x):
            stats["with_exchange_time"] += 1
            raw_lag = float(rt) - float(raw_x)
            if raw_lag >= -EPS:
                stats["exchange_at_or_before_receipt"] += 1
                max_lag_s = max(max_lag_s, max(0.0, raw_lag))
                lags_ms.append(1000.0 * raw_lag)
            else:
                stats["exchange_after_receipt_clamped"] += 1
        else:
            stats["missing_exchange_time"] += 1

        if show and n % 1_000_000 == 0:
            print(f"trade timing scan: read {n:,} raw lines | selected={stats['trade_rows']:,}")

    lookahead_s = max_lag_s + max_receipt_disorder_s + LOOKAHEAD_PAD_S
    a = np.asarray([x for x in lags_ms if np.isfinite(x)], dtype=float)
    return {
        "stats": dict(stats),
        "max_exchange_to_receipt_lag_s": float(max_lag_s),
        "max_receipt_file_disorder_s": float(max_receipt_disorder_s),
        "safe_stream_lookahead_s": float(lookahead_s),
        "lag_ms_median": float(np.median(a)) if len(a) else np.nan,
        "lag_ms_p95": float(np.percentile(a, 95)) if len(a) else np.nan,
        "lag_ms_max": float(np.max(a)) if len(a) else np.nan,
    }


def _next_book(it, selected_tickers):
    for row in it:
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        t = _receipt_s(row)
        if np.isfinite(t):
            return float(t), row
    return None


def _next_trade(it, selected_tickers):
    for row in it:
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        rt = _receipt_s(row)
        if not np.isfinite(rt):
            continue
        exec_s, exec_source = _causal_exec_s(row, rt)
        return {
            "receipt_s": float(rt),
            "exec_s": float(exec_s),
            "exec_source": exec_source,
            "row": row,
        }
    return None


class Q10DualClockShadow(V10.QuietDualClockQ5Shadow):
    """V10 dual-clock state machine with the frozen formal-OOS Q10 size."""

    def _desired_quote(self, ticker, cur, elapsed):
        if cur is None or not (0.0 <= elapsed < 300.0) or ticker in self.finalized:
            return None
        inv = float(self.known_inventory[ticker])
        if abs(inv) <= OOS.EPS:
            side = OOS._entry_side(cur)
            if side is None:
                return None
            return {
                "role": "ENTRY",
                "side": side,
                "price": cur["bid"] if side == "BID" else cur["ask"],
                "qty": QTY,
                "queue_ahead": cur["bid_q1"] if side == "BID" else cur["ask_q1"],
            }
        side = "ASK" if inv > 0 else "BID"
        return {
            "role": "EXIT",
            "side": side,
            "price": cur["ask"] if side == "ASK" else cur["bid"],
            "qty": abs(inv),
            "queue_ahead": cur["ask_q1"] if side == "ASK" else cur["bid_q1"],
        }


def _baseline_net(summary):
    for key in (
        "realized_net_trade_fee_only",
        "realized_net_pnl",
        "net_pnl",
    ):
        z = _f((summary or {}).get(key))
        if np.isfinite(z):
            return float(z), key
    p = _f((summary or {}).get("passive_matched_pnl"), 0.0)
    g = _f((summary or {}).get("forced_liq_gross_pnl"), 0.0)
    f = _f((summary or {}).get("taker_trade_fees"), 0.0)
    return float(p + g - f), "RECONSTRUCTED_PASSIVE_PLUS_M5_MINUS_FEES"


def _window_completeness(meta_rows):
    by_close = defaultdict(set)
    for r in meta_rows:
        close = str(r.get("close_time") or "")
        series = str(r.get("series_ticker") or "")
        if close and series:
            by_close[close].add(series)
    frozen = set(OOS.SERIES)
    complete = sum(1 for s in by_close.values() if frozen.issubset(s))
    return len(by_close), int(complete)


def run_q10_oos_dual_clock_causal_replay(
    source_session,
    *,
    hard_bind=True,
    show=True,
):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_OOS_SESSION:
        raise RuntimeError(
            f"Replay is hard-bound to formal OOS session {HARD_BOUND_OOS_SESSION}; got {source.name}."
        )
    if hard_bind and "mm_event_m0_m5_oos_cycle_q10_v1" not in str(source.parent):
        raise RuntimeError(
            "Expected formal frozen OOS root mm_event_m0_m5_oos_cycle_q10_v1."
        )

    original_shadow = source / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1" / "shadow_summary.json"
    required = [
        source / "book_top3_events.jsonl",
        source / "trades_event_time.jsonl",
        source / "market_metadata.jsonl",
        source / "fee_preflight.json",
        original_shadow,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required formal OOS artifacts: " + " | ".join(missing))

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored formal-OOS fee preflight was not PASS.")

    baseline = OOS._read_json(original_shadow, {}) or {}
    baseline_net, baseline_net_field = _baseline_net(baseline)
    baseline_passive = _f(baseline.get("passive_matched_pnl"))
    baseline_forced = _f(baseline.get("forced_liq_gross_pnl"))
    baseline_fees = _f(baseline.get("taker_trade_fees"))
    baseline_fill_qty = _f(baseline.get("fill_qty"))
    baseline_fill_events = _f(baseline.get("fill_events"))
    baseline_cycles_started = _f(baseline.get("cycles_started"))
    baseline_cycles_completed = _f(baseline.get("cycles_completed"))

    meta_rows, meta_by_ticker = _load_metadata(source)
    selected_tickers = set(meta_by_ticker)
    if not selected_tickers:
        raise RuntimeError("No market metadata tickers found in formal OOS session.")
    metadata_windows, complete_windows = _window_completeness(meta_rows)

    if show:
        print("Scanning public-trade timing to establish a memory-bounded causal look-ahead...")
    timing_scan = _scan_trade_timing(
        source / "trades_event_time.jsonl",
        selected_tickers,
        show=show,
    )
    lookahead_s = float(timing_scan["safe_stream_lookahead_s"])

    out = _new_output(source.name)
    workspace = out / "dual_clock_workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    shadow = Q10DualClockShadow(workspace, fee)

    for row in meta_rows:
        ticker = str(row.get("ticker") or "")
        shadow.meta[ticker] = row
        shadow.series_by_ticker[ticker] = str(row.get("series_ticker") or "")
        shadow.close_by_ticker[ticker] = str(row.get("close_time") or "")

    book_it = iter(_iter_jsonl(source / "book_top3_events.jsonl"))
    trade_it = iter(_iter_jsonl(source / "trades_event_time.jsonl"))
    b = _next_book(book_it, selected_tickers)
    tr_next = _next_trade(trade_it, selected_tickers)

    trade_heap = []
    trade_seq = 0
    loaded_trades = 0
    processed_books = 0
    processed_exec = 0
    processed_obs = 0
    processed_events = 0
    first_event_s = np.inf
    last_event_s = -np.inf
    max_trade_heap = 0

    def load_one_trade():
        nonlocal tr_next, trade_seq, loaded_trades, max_trade_heap
        if tr_next is None:
            return False
        z = tr_next
        row = z["row"]
        trade_seq += 1
        # Same row object is used for economic execution and later observation.
        heapq.heappush(
            trade_heap,
            (float(z["exec_s"]), 0, trade_seq, "TRADE_EXEC", row, float(z["receipt_s"])),
        )
        heapq.heappush(
            trade_heap,
            (float(z["receipt_s"]), 2, trade_seq, "TRADE_OBS", row, float(z["receipt_s"])),
        )
        loaded_trades += 1
        max_trade_heap = max(max_trade_heap, len(trade_heap))
        tr_next = _next_trade(trade_it, selected_tickers)
        return True

    def load_through_receipt(cutoff_s):
        while tr_next is not None and float(tr_next["receipt_s"]) <= float(cutoff_s) + EPS:
            if not load_one_trade():
                break

    # Seed enough trades to establish the first causal candidate.
    if b is None and tr_next is not None:
        load_one_trade()

    if b is None and not trade_heap and tr_next is None:
        raise RuntimeError("No timed raw events found in formal OOS session.")

    while b is not None or trade_heap or tr_next is not None:
        # Provisional earliest known event.  Read sufficiently far ahead in the
        # receipt-ordered trade file that no unseen trade execution can precede it.
        candidates = []
        if b is not None:
            candidates.append(float(b[0]))
        if trade_heap:
            candidates.append(float(trade_heap[0][0]))
        if not candidates and tr_next is not None:
            candidates.append(min(float(tr_next["exec_s"]), float(tr_next["receipt_s"])))
        if not candidates:
            break

        horizon = min(candidates)
        load_through_receipt(horizon + lookahead_s)

        # If heap is still empty but trades remain, load the next row explicitly.
        if not trade_heap and tr_next is not None:
            load_one_trade()

        book_key = (float(b[0]), 1, -1) if b is not None else (np.inf, 1, -1)
        trade_key = trade_heap[0][:3] if trade_heap else (np.inf, 9, -1)

        if trade_key < book_key:
            t, _priority, _seq, typ, row, receipt_t = heapq.heappop(trade_heap)
            if typ == "TRADE_EXEC":
                shadow.on_trade_execution(float(t), float(receipt_t), row)
                processed_exec += 1
            else:
                shadow.on_trade_observation(float(t), row)
                processed_obs += 1
        else:
            t, row = b
            shadow.on_book_receipt(float(t), row)
            processed_books += 1
            b = _next_book(book_it, selected_tickers)

        first_event_s = min(first_event_s, float(t))
        last_event_s = max(last_event_s, float(t))
        processed_events += 1
        shadow._update_drawdown()

        if show and processed_events % 1_000_000 == 0:
            print(
                f"processed {processed_events:,} causal events | books={processed_books:,} "
                f"trade_exec={processed_exec:,} trade_obs={processed_obs:,} "
                f"fills={int(shadow.c['fill_events']):,} heap={len(trade_heap):,}"
            )

    shadow.thread_alive = False

    unfinalized = sorted(t for t in selected_tickers if t not in shadow.finalized)
    detail, by_window, by_asset = V10._economics(shadow)
    fills = pd.DataFrame(shadow.fills)

    passive = float(shadow.passive_matched_pnl)
    forced = float(shadow.forced_liq_gross_pnl)
    fees = float(shadow.taker_trade_fees)
    causal_net = passive + forced - fees

    event_span_h = (
        (last_event_s - first_event_s) / 3600.0
        if np.isfinite(first_event_s) and np.isfinite(last_event_s)
        else np.nan
    )
    runtime_h = _f(baseline.get("runtime_hours"), event_span_h)
    if not np.isfinite(runtime_h) or runtime_h <= 0:
        runtime_h = event_span_h
    baseline_per_day = baseline_net * 24.0 / runtime_h if np.isfinite(runtime_h) and runtime_h > 0 else np.nan
    causal_per_day = causal_net * 24.0 / runtime_h if np.isfinite(runtime_h) and runtime_h > 0 else np.nan

    obs_delays = (
        pd.to_numeric(fills.get("observation_delay_ms"), errors="coerce").dropna()
        if not fills.empty and "observation_delay_ms" in fills
        else pd.Series(dtype=float)
    )

    win_net = (
        pd.to_numeric(by_window["net_pnl"], errors="coerce").dropna()
        if not by_window.empty else pd.Series(dtype=float)
    )
    window_stats = {
        "windows_with_economics": int(len(win_net)),
        "positive_windows": int((win_net > 0).sum()) if len(win_net) else 0,
        "negative_windows": int((win_net < 0).sum()) if len(win_net) else 0,
        "zero_windows": int((win_net == 0).sum()) if len(win_net) else 0,
        "positive_rate": float((win_net > 0).mean()) if len(win_net) else np.nan,
        "mean_net": float(win_net.mean()) if len(win_net) else np.nan,
        "median_net": float(win_net.median()) if len(win_net) else np.nan,
        "worst_net": float(win_net.min()) if len(win_net) else np.nan,
        "best_net": float(win_net.max()) if len(win_net) else np.nan,
    }

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "hard_bound": bool(hard_bind),
        "same_historical_oos_source": True,
        "independent_validation_for_corrected_model": False,
        "posthoc_execution_model_correction": True,
        "quote_size": QTY,
        "selected_tickers": len(selected_tickers),
        "metadata_windows": metadata_windows,
        "complete_9_series_windows": complete_windows,
        "trade_timing_scan": timing_scan,
        "processed_book_receipts": processed_books,
        "processed_trade_executions": processed_exec,
        "processed_trade_observations": processed_obs,
        "total_causal_events": processed_events,
        "max_trade_event_heap": int(max_trade_heap),
        "event_span_hours": event_span_h,
        "comparison_runtime_hours": runtime_h,
        "baseline_net_field": baseline_net_field,
        "baseline_receipt_shadow_passive_pnl": baseline_passive,
        "baseline_receipt_shadow_forced_gross": baseline_forced,
        "baseline_receipt_shadow_fees": baseline_fees,
        "baseline_receipt_shadow_net_pnl": baseline_net,
        "baseline_receipt_shadow_per_day": baseline_per_day,
        "baseline_fill_events": baseline_fill_events,
        "baseline_fill_qty": baseline_fill_qty,
        "baseline_cycles_started": baseline_cycles_started,
        "baseline_cycles_completed": baseline_cycles_completed,
        "causal_passive_matched_pnl": passive,
        "causal_forced_liq_gross_pnl": forced,
        "causal_taker_trade_fees": fees,
        "causal_net_pnl": causal_net,
        "causal_per_day": causal_per_day,
        "causal_minus_baseline_net_pnl": causal_net - baseline_net,
        "causal_fill_events": int(shadow.c["fill_events"]),
        "causal_fill_qty": shadow.c["fill_qty_x1000"] / 1000.0,
        "causal_cycles_started": int(shadow.c["cycles_started"]),
        "causal_cycles_completed": int(shadow.c["cycles_completed"]),
        "causal_forced_liquidations": int(shadow.c["forced_liquidations"]),
        "causal_forced_liq_qty": shadow.c["forced_liq_qty_x1000"] / 1000.0,
        "causal_max_drawdown_online": float(shadow.max_drawdown),
        "economic_fill_observation_delay_ms_median": float(obs_delays.median()) if len(obs_delays) else np.nan,
        "cancel_after_hidden_execution": int(shadow.dual["cancel_after_hidden_execution"]),
        "creates_while_fill_hidden": int(shadow.dual["creates_while_fill_hidden"]),
        "newer_quote_cancelled_on_late_fill_observation": int(
            shadow.dual["newer_quote_cancelled_on_late_fill_observation"]
        ),
        "fill_observations": int(shadow.dual["fill_observations"]),
        "fill_observed_after_m5_finalize": int(shadow.dual["fill_observed_after_m5_finalize"]),
        "unfinalized_tickers": unfinalized,
        "window_stats": window_stats,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "model_limitations": {
            "book_decision_clock": "local receipt_time",
            "passive_execution_clock": "public trade exchange_time; impossible exchange-after-receipt rows clamp to receipt",
            "fill_observation_clock": "public trade receipt_time proxy",
            "create_cancel_latency": "idealized zero at local book receipt",
            "queue_model": "frozen displayed-L1 FIFO; no cancellation-ahead credit",
            "tie_policy": "trade execution before book receipt before trade observation",
        },
        "interpretation_guardrail": (
            "This reruns the old formal OOS realization with a posthoc causality correction discovered after live testing. "
            "It can invalidate the old +PnL evidence, but a positive corrected result would still require fresh forward OOS validation."
        ),
    }

    OOS._atomic_json(out / "summary.json", summary)
    detail.to_csv(out / "causal_by_contract.csv", index=False)
    by_window.to_csv(out / "causal_by_window.csv", index=False)
    by_asset.to_csv(out / "causal_by_asset.csv", index=False)
    fills.to_csv(out / "causal_passive_fills.csv", index=False)

    if show:
        print("=" * 146)
        print("FORMAL Q10 OOS — DUAL-CLOCK CAUSAL REPLAY V11 / READ ONLY")
        print("=" * 146)
        print("Source:", source)
        print("Tickers:", len(selected_tickers), "| metadata windows:", metadata_windows, "| complete 9-series windows:", complete_windows)
        print("Causal events processed:", f"{processed_events:,}")
        print("  books:", f"{processed_books:,}", "| trade exec:", f"{processed_exec:,}", "| trade obs:", f"{processed_obs:,}")
        print("Trade timing scan:")
        print("  exchange coverage:", timing_scan["stats"].get("with_exchange_time", 0), "/", timing_scan["stats"].get("trade_rows", 0))
        print("  exchange->receipt median/p95/max ms:", f"{timing_scan['lag_ms_median']:.3f}", "/", f"{timing_scan['lag_ms_p95']:.3f}", "/", f"{timing_scan['lag_ms_max']:.3f}")
        print("  safe streaming lookahead ms:", f"{1000.0 * lookahead_s:.3f}")
        print("  exchange-after-receipt rows clamped:", timing_scan["stats"].get("exchange_after_receipt_clamped", 0))
        print("  max trade-event heap:", f"{max_trade_heap:,}")
        print()
        print("CORE ECONOMICS")
        print(f"  ORIGINAL RECEIPT-TIME OOS NET:    {baseline_net:+.4f}")
        print(f"  original normalized / day:       {baseline_per_day:+.4f}")
        print(f"  causal dual-clock passive PnL:    {passive:+.4f}")
        print(f"  causal dual-clock M5 gross:       {forced:+.4f}")
        print(f"  causal dual-clock taker fees:     {fees:.4f}")
        print(f"  CAUSAL DUAL-CLOCK NET:            {causal_net:+.4f}")
        print(f"  causal normalized / day:         {causal_per_day:+.4f}")
        print(f"  CAUSAL - ORIGINAL:                {causal_net - baseline_net:+.4f}")
        print()
        print("FILL / PATH COMPARISON")
        print("  original fill events / qty:", baseline_fill_events, "/", baseline_fill_qty)
        print("  causal fill events / qty:  ", int(shadow.c["fill_events"]), "/", f"{summary['causal_fill_qty']:.4f}")
        print("  original cycles start/complete:", baseline_cycles_started, "/", baseline_cycles_completed)
        print("  causal cycles start/complete:  ", int(shadow.c["cycles_started"]), "/", int(shadow.c["cycles_completed"]))
        print("  cancel after hidden execution:", int(shadow.dual["cancel_after_hidden_execution"]))
        print("  creates while fill hidden:", int(shadow.dual["creates_while_fill_hidden"]))
        print("  newer quote cancelled on fill observation:", int(shadow.dual["newer_quote_cancelled_on_late_fill_observation"]))
        print("  median economic fill -> observation ms:", f"{summary['economic_fill_observation_delay_ms_median']:.3f}")
        print("  unfinalized tickers:", unfinalized)
        print()
        print("WINDOW DISTRIBUTION")
        print(window_stats)
        if not by_window.empty:
            print("\nWORST 10 WINDOWS")
            print(by_window.nsmallest(10, "net_pnl").to_string(index=False))
            print("\nBEST 5 WINDOWS")
            print(by_window.nlargest(5, "net_pnl").to_string(index=False))
        print()
        print("BY ASSET")
        print(by_asset.to_string(index=False) if not by_asset.empty else "  none")
        print()
        print("SCIENTIFIC STATUS")
        print("  - Candidate-C, Q10, spread/depth rule, exit rule and frozen FIFO assumptions are unchanged.")
        print("  - The execution/observation clock model is corrected posthoc using knowledge from later live forensics.")
        print("  - Therefore this old 24h sample is NOT independent OOS validation for the corrected simulator.")
        print("  - If corrected economics survive, the next profitability test must be a fresh forward OOS capture.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 146)

    return {
        "summary": summary,
        "detail": detail,
        "by_window": by_window,
        "by_asset": by_asset,
        "fills": fills,
        "output_dir": out,
    }


__all__ = [
    "VERSION",
    "HARD_BOUND_OOS_SESSION",
    "run_q10_oos_dual_clock_causal_replay",
]
