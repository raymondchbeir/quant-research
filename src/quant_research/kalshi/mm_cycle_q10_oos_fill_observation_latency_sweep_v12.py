from __future__ import annotations

"""Fill-observation latency sensitivity for the formal Q10 OOS realization.

Purpose
-------
V11 corrected the frozen Candidate-C simulator so that passive execution occurs at
public-trade exchange time while fill knowledge arrives at public-trade receipt
 time.  On the old formal OOS realization this changed economics from strongly
positive to strongly negative, but also created thousands of quotes while an
execution was economically real but not yet known to the strategy.

This module brackets ONLY that fill-observation assumption.  Passive execution,
book-decision timing, Candidate-C, Q10 size, displayed-L1 FIFO queue assumptions,
exit mechanics, M5 liquidation, fees, assets and every strategy threshold remain
unchanged.

Observation scenarios
---------------------
- 0 ms after economic execution
- 5 ms after economic execution
- 10 ms after economic execution
- 20 ms after economic execution
- 30 ms after economic execution
- actual public-trade receipt time (the V11 model)

The fixed-delay scenarios are hypothetical private-fill-notification latencies.
They are NOT claims about Kalshi's actual private fill feed.  Their purpose is to
answer whether Candidate-C remains negative even under extremely fast fill
awareness, or whether the catastrophic V11 result depends mainly on hidden-fill
latency.

Scientific guardrails
---------------------
- SAME historical realization; posthoc sensitivity only.
- NOT fresh OOS validation for the corrected simulator.
- NO exchange/API calls and NO orders.
- Source session is read-only.
- Hard-bound to formal OOS session 20260817_064143 by default.
- V11's causal execution clock is reused exactly, including its explicit clamp of
  impossible exchange-after-receipt timestamps.
- Tie policy is kept identical to V11: trade execution, then book receipt, then
  fill observation at an exact timestamp tie.  Therefore the PUBLIC_RECEIPT
  scenario should reconcile V11 exactly.
- The six scenarios are replayed in one streaming pass to avoid six full reads of
  the ~10M-row book capture.
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
from . import mm_cycle_q10_oos_dual_clock_causal_replay_v11 as V11

VERSION = "MM_CYCLE_Q10_OOS_FILL_OBSERVATION_LATENCY_SWEEP_V12"
HARD_BOUND_OOS_SESSION = V11.HARD_BOUND_OOS_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q10_oos_fill_observation_latency_sweep_v12"
EPS = 1e-10

SCENARIOS = (
    {"key": "OBS_0MS", "mode": "FIXED", "delay_ms": 0.0},
    {"key": "OBS_5MS", "mode": "FIXED", "delay_ms": 5.0},
    {"key": "OBS_10MS", "mode": "FIXED", "delay_ms": 10.0},
    {"key": "OBS_20MS", "mode": "FIXED", "delay_ms": 20.0},
    {"key": "OBS_30MS", "mode": "FIXED", "delay_ms": 30.0},
    {"key": "OBS_PUBLIC_RECEIPT", "mode": "PUBLIC_RECEIPT", "delay_ms": None},
)


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


def _observation_s(spec, exec_s: float, receipt_s: float) -> float:
    if str(spec["mode"]) == "PUBLIC_RECEIPT":
        return float(receipt_s)
    return float(exec_s) + float(spec["delay_ms"]) / 1000.0


def _find_prior_v11(source: Path):
    root = Path(V11.OUTPUT_ROOT)
    candidates = []
    if root.exists():
        for d in root.glob(source.name + "*"):
            sp = d / "summary.json"
            if not sp.exists():
                continue
            obj = OOS._read_json(sp, {}) or {}
            try:
                same = Path(obj.get("source_session", "")).resolve() == source.resolve()
            except Exception:
                same = False
            if same:
                candidates.append((sp.stat().st_mtime, d.resolve(), obj))
    if not candidates:
        return None, None
    _, d, obj = max(candidates, key=lambda z: z[0])
    return d, obj


class SweepQ10Shadow(V11.Q10DualClockShadow):
    """V11 state machine with in-memory fill output for a multi-scenario sweep."""

    def __init__(self, workspace, fee_preflight_result, scenario_key):
        super().__init__(workspace, fee_preflight_result)
        self.scenario_key = str(scenario_key)

    def _write_fill(self, f):
        # ``self.fills`` is already authoritative in the base mechanics.  Avoid
        # six simultaneous JSONL fill streams during the sensitivity sweep.
        return None


def _window_stats(by_window: pd.DataFrame):
    x = (
        pd.to_numeric(by_window["net_pnl"], errors="coerce").dropna()
        if not by_window.empty
        else pd.Series(dtype=float)
    )
    return {
        "windows_with_economics": int(len(x)),
        "positive_windows": int((x > 0).sum()) if len(x) else 0,
        "negative_windows": int((x < 0).sum()) if len(x) else 0,
        "zero_windows": int((x == 0).sum()) if len(x) else 0,
        "positive_rate": float((x > 0).mean()) if len(x) else np.nan,
        "mean_net": float(x.mean()) if len(x) else np.nan,
        "median_net": float(x.median()) if len(x) else np.nan,
        "worst_net": float(x.min()) if len(x) else np.nan,
        "best_net": float(x.max()) if len(x) else np.nan,
    }


def _scenario_result(
    key,
    spec,
    shadow,
    selected_tickers,
    runtime_h,
    original_net,
):
    detail, by_window, by_asset = V10._economics(shadow)
    fills = pd.DataFrame(shadow.fills)

    passive = float(shadow.passive_matched_pnl)
    forced = float(shadow.forced_liq_gross_pnl)
    fees = float(shadow.taker_trade_fees)
    net = passive + forced - fees
    per_day = net * 24.0 / runtime_h if np.isfinite(runtime_h) and runtime_h > 0 else np.nan

    obs_delays = (
        pd.to_numeric(fills.get("observation_delay_ms"), errors="coerce").dropna()
        if not fills.empty and "observation_delay_ms" in fills
        else pd.Series(dtype=float)
    )
    unfinalized = sorted(t for t in selected_tickers if t not in shadow.finalized)

    summary = {
        "scenario": key,
        "mode": spec["mode"],
        "fixed_observation_delay_ms": spec["delay_ms"],
        "passive_matched_pnl": passive,
        "forced_liq_gross_pnl": forced,
        "taker_trade_fees": fees,
        "net_pnl": net,
        "normalized_per_day": per_day,
        "minus_original_receipt_shadow_net": net - original_net,
        "fill_events": int(shadow.c["fill_events"]),
        "fill_qty": shadow.c["fill_qty_x1000"] / 1000.0,
        "cycles_started": int(shadow.c["cycles_started"]),
        "cycles_completed": int(shadow.c["cycles_completed"]),
        "forced_liquidations": int(shadow.c["forced_liquidations"]),
        "forced_liq_qty": shadow.c["forced_liq_qty_x1000"] / 1000.0,
        "max_drawdown_online": float(shadow.max_drawdown),
        "median_realized_observation_delay_ms": (
            float(obs_delays.median()) if len(obs_delays) else np.nan
        ),
        "cancel_after_hidden_execution": int(shadow.dual["cancel_after_hidden_execution"]),
        "creates_while_fill_hidden": int(shadow.dual["creates_while_fill_hidden"]),
        "newer_quote_cancelled_on_late_fill_observation": int(
            shadow.dual["newer_quote_cancelled_on_late_fill_observation"]
        ),
        "fill_observations": int(shadow.dual["fill_observations"]),
        "fill_observed_after_m5_finalize": int(shadow.dual["fill_observed_after_m5_finalize"]),
        "unfinalized_tickers": unfinalized,
        "window_stats": _window_stats(by_window),
    }
    return summary, detail, by_window, by_asset, fills


def run_q10_oos_fill_observation_latency_sweep(
    source_session,
    *,
    hard_bind=True,
    show=True,
):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_OOS_SESSION:
        raise RuntimeError(
            f"Sweep is hard-bound to formal OOS session {HARD_BOUND_OOS_SESSION}; got {source.name}."
        )
    if hard_bind and "mm_event_m0_m5_oos_cycle_q10_v1" not in str(source.parent):
        raise RuntimeError("Expected formal OOS root mm_event_m0_m5_oos_cycle_q10_v1.")

    original_shadow_path = (
        source / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1" / "shadow_summary.json"
    )
    required = [
        source / "book_top3_events.jsonl",
        source / "trades_event_time.jsonl",
        source / "market_metadata.jsonl",
        source / "fee_preflight.json",
        original_shadow_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required formal OOS artifacts: " + " | ".join(missing))

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored formal-OOS fee preflight was not PASS.")

    original = OOS._read_json(original_shadow_path, {}) or {}
    original_net, original_net_field = V11._baseline_net(original)
    original_runtime_h = _f(original.get("runtime_hours"))

    meta_rows, meta_by_ticker = V11._load_metadata(source)
    selected_tickers = set(meta_by_ticker)
    if not selected_tickers:
        raise RuntimeError("No market metadata tickers found.")
    metadata_windows, complete_windows = V11._window_completeness(meta_rows)

    if show:
        print("Scanning public-trade timing once for the shared streaming look-ahead...")
    timing_scan = V11._scan_trade_timing(
        source / "trades_event_time.jsonl",
        selected_tickers,
        show=show,
    )
    lookahead_s = float(timing_scan["safe_stream_lookahead_s"])

    out = _new_output(source.name)
    shadows = {}
    specs = {str(x["key"]): dict(x) for x in SCENARIOS}

    for spec in SCENARIOS:
        key = str(spec["key"])
        workspace = out / key.lower()
        workspace.mkdir(parents=True, exist_ok=False)
        shadow = SweepQ10Shadow(workspace, fee, key)
        for row in meta_rows:
            ticker = str(row.get("ticker") or "")
            shadow.meta[ticker] = row
            shadow.series_by_ticker[ticker] = str(row.get("series_ticker") or "")
            shadow.close_by_ticker[ticker] = str(row.get("close_time") or "")
        shadow.thread_alive = True
        shadows[key] = shadow

    book_it = iter(_iter_jsonl(source / "book_top3_events.jsonl"))
    trade_it = iter(_iter_jsonl(source / "trades_event_time.jsonl"))
    b = V11._next_book(book_it, selected_tickers)
    tr_next = V11._next_trade(trade_it, selected_tickers)

    # Heap entries:
    #   (event_time, priority, seq, type, scenario_key, row, public_receipt_s)
    # Priority intentionally matches V11: execution=0, book=1, observation=2.
    trade_heap = []
    trade_seq = 0
    loaded_trades = 0
    max_trade_heap = 0

    processed_books = 0
    processed_exec = 0
    processed_obs = Counter()
    processed_events = 0
    first_event_s = np.inf
    last_event_s = -np.inf

    def load_one_trade():
        nonlocal tr_next, trade_seq, loaded_trades, max_trade_heap
        if tr_next is None:
            return False
        z = tr_next
        row = z["row"]
        exec_s = float(z["exec_s"])
        receipt_s = float(z["receipt_s"])
        trade_seq += 1
        seq = trade_seq

        heapq.heappush(
            trade_heap,
            (exec_s, 0, seq, "TRADE_EXEC", None, row, receipt_s),
        )
        for j, spec in enumerate(SCENARIOS):
            key = str(spec["key"])
            obs_s = _observation_s(spec, exec_s, receipt_s)
            # Preserve deterministic scenario ordering without changing the V11
            # priority relative to book receipts.
            obs_seq = seq * 16 + j
            heapq.heappush(
                trade_heap,
                (float(obs_s), 2, obs_seq, "TRADE_OBS", key, row, receipt_s),
            )

        loaded_trades += 1
        max_trade_heap = max(max_trade_heap, len(trade_heap))
        tr_next = V11._next_trade(trade_it, selected_tickers)
        return True

    def load_through_receipt(cutoff_s):
        while tr_next is not None and float(tr_next["receipt_s"]) <= float(cutoff_s) + EPS:
            if not load_one_trade():
                break

    if b is None and tr_next is not None:
        load_one_trade()
    if b is None and not trade_heap and tr_next is None:
        raise RuntimeError("No timed raw events found.")

    while b is not None or trade_heap or tr_next is not None:
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
        if not trade_heap and tr_next is not None:
            load_one_trade()

        book_key = (float(b[0]), 1, -1) if b is not None else (np.inf, 1, -1)
        heap_key = trade_heap[0][:3] if trade_heap else (np.inf, 9, -1)

        if heap_key < book_key:
            t, _priority, _seq, typ, scenario_key, row, receipt_s = heapq.heappop(trade_heap)
            if typ == "TRADE_EXEC":
                for spec in SCENARIOS:
                    key = str(spec["key"])
                    obs_s = _observation_s(spec, float(t), float(receipt_s))
                    sh = shadows[key]
                    sh.on_trade_execution(float(t), float(obs_s), row)
                    sh._update_drawdown()
                processed_exec += 1
            else:
                sh = shadows[str(scenario_key)]
                sh.on_trade_observation(float(t), row)
                sh._update_drawdown()
                processed_obs[str(scenario_key)] += 1
        else:
            t, row = b
            for sh in shadows.values():
                sh.on_book_receipt(float(t), row)
                sh._update_drawdown()
            processed_books += 1
            b = V11._next_book(book_it, selected_tickers)

        first_event_s = min(first_event_s, float(t))
        last_event_s = max(last_event_s, float(t))
        processed_events += 1

        if show and processed_events % 2_000_000 == 0:
            fill_bits = " | ".join(
                f"{k}={int(shadows[k].c['fill_events'])}"
                for k in ("OBS_0MS", "OBS_5MS", "OBS_10MS", "OBS_20MS", "OBS_30MS", "OBS_PUBLIC_RECEIPT")
            )
            print(
                f"processed {processed_events:,} sweep events | books={processed_books:,} "
                f"trade_exec={processed_exec:,} heap={len(trade_heap):,} | fills {fill_bits}"
            )

    for sh in shadows.values():
        sh.thread_alive = False

    event_span_h = (
        (last_event_s - first_event_s) / 3600.0
        if np.isfinite(first_event_s) and np.isfinite(last_event_s)
        else np.nan
    )
    runtime_h = original_runtime_h
    if not np.isfinite(runtime_h) or runtime_h <= 0:
        runtime_h = event_span_h

    scenario_summaries = {}
    details = {}
    windows = {}
    assets = {}
    fills = {}

    for spec in SCENARIOS:
        key = str(spec["key"])
        res = _scenario_result(
            key,
            spec,
            shadows[key],
            selected_tickers,
            runtime_h,
            original_net,
        )
        s, d, w, a, f = res
        scenario_summaries[key] = s
        details[key] = d
        windows[key] = w
        assets[key] = a
        fills[key] = f

    rows = []
    for spec in SCENARIOS:
        key = str(spec["key"])
        s = scenario_summaries[key]
        ws = s["window_stats"]
        rows.append(
            {
                "scenario": key,
                "observation_model": s["mode"],
                "fixed_delay_ms": s["fixed_observation_delay_ms"],
                "net_pnl": s["net_pnl"],
                "normalized_per_day": s["normalized_per_day"],
                "passive_pnl": s["passive_matched_pnl"],
                "m5_gross": s["forced_liq_gross_pnl"],
                "taker_fees": s["taker_trade_fees"],
                "fill_events": s["fill_events"],
                "fill_qty": s["fill_qty"],
                "cycles_started": s["cycles_started"],
                "cycles_completed": s["cycles_completed"],
                "creates_while_fill_hidden": s["creates_while_fill_hidden"],
                "cancel_after_hidden_execution": s["cancel_after_hidden_execution"],
                "newer_quote_cancelled_on_late_fill_observation": s[
                    "newer_quote_cancelled_on_late_fill_observation"
                ],
                "median_realized_observation_delay_ms": s[
                    "median_realized_observation_delay_ms"
                ],
                "positive_windows": ws["positive_windows"],
                "negative_windows": ws["negative_windows"],
                "median_window_net": ws["median_net"],
                "worst_window_net": ws["worst_net"],
                "best_window_net": ws["best_net"],
                "unfinalized_tickers": len(s["unfinalized_tickers"]),
            }
        )
    sweep = pd.DataFrame(rows)

    prior_v11_dir, prior_v11 = _find_prior_v11(source)
    public = scenario_summaries["OBS_PUBLIC_RECEIPT"]
    v11_recon = {
        "prior_v11_source": str(prior_v11_dir) if prior_v11_dir else None,
        "prior_v11_found": bool(prior_v11),
        "net_delta": np.nan,
        "fill_events_delta": np.nan,
        "fill_qty_delta": np.nan,
        "cycles_started_delta": np.nan,
        "cycles_completed_delta": np.nan,
        "pass": None,
    }
    if prior_v11:
        v11_recon["net_delta"] = public["net_pnl"] - _f(prior_v11.get("causal_net_pnl"))
        v11_recon["fill_events_delta"] = public["fill_events"] - _f(prior_v11.get("causal_fill_events"))
        v11_recon["fill_qty_delta"] = public["fill_qty"] - _f(prior_v11.get("causal_fill_qty"))
        v11_recon["cycles_started_delta"] = public["cycles_started"] - _f(prior_v11.get("causal_cycles_started"))
        v11_recon["cycles_completed_delta"] = public["cycles_completed"] - _f(prior_v11.get("causal_cycles_completed"))
        v11_recon["pass"] = bool(
            abs(float(v11_recon["net_delta"])) <= 1e-9
            and abs(float(v11_recon["fill_events_delta"])) <= 1e-12
            and abs(float(v11_recon["fill_qty_delta"])) <= 1e-9
            and abs(float(v11_recon["cycles_started_delta"])) <= 1e-12
            and abs(float(v11_recon["cycles_completed_delta"])) <= 1e-12
        )

    low_latency_nets = {
        key: scenario_summaries[key]["net_pnl"]
        for key in ("OBS_0MS", "OBS_5MS", "OBS_10MS")
    }
    if all(float(v) < 0.0 for v in low_latency_nets.values()):
        diagnostic_verdict = "NEGATIVE_EVEN_AT_0_5_10MS_ON_THIS_HISTORICAL_SAMPLE"
    elif scenario_summaries["OBS_0MS"]["net_pnl"] > 0 and public["net_pnl"] < 0:
        diagnostic_verdict = "STRONG_FILL_OBSERVATION_LATENCY_SENSITIVITY"
    else:
        diagnostic_verdict = "MIXED_LATENCY_SENSITIVITY"

    overall = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "hard_bound": bool(hard_bind),
        "same_historical_oos_source": True,
        "independent_validation": False,
        "posthoc_sensitivity": True,
        "strategy_parameters_changed": False,
        "quote_size": float(OOS.QUOTE_SIZE),
        "selected_tickers": len(selected_tickers),
        "metadata_windows": metadata_windows,
        "complete_9_series_windows": complete_windows,
        "trade_timing_scan": timing_scan,
        "processed_book_receipts": processed_books,
        "processed_trade_executions": processed_exec,
        "processed_trade_observations_by_scenario": dict(processed_obs),
        "total_sweep_events": processed_events,
        "max_trade_event_heap": int(max_trade_heap),
        "event_span_hours": event_span_h,
        "comparison_runtime_hours": runtime_h,
        "original_receipt_shadow_net_field": original_net_field,
        "original_receipt_shadow_net_pnl": original_net,
        "original_receipt_shadow_per_day": (
            original_net * 24.0 / runtime_h
            if np.isfinite(runtime_h) and runtime_h > 0
            else np.nan
        ),
        "scenarios": scenario_summaries,
        "public_receipt_reconciliation_to_prior_v11": v11_recon,
        "diagnostic_verdict": diagnostic_verdict,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "model_constants": {
            "book_decision_clock": "local receipt_time",
            "passive_execution_clock": "V11 causal exchange clock with exchange-after-receipt clamp",
            "queue_model": "frozen displayed-L1 FIFO; no cancellation-ahead credit",
            "create_cancel_latency": "idealized zero at local book receipt",
            "tie_policy": "trade execution before book receipt before fill observation",
            "observation_scenarios": list(SCENARIOS),
        },
        "interpretation_guardrail": (
            "This sensitivity sweep uses the same old historical realization after the execution-clock flaw was discovered. "
            "It can test robustness to assumed fill-notification latency but cannot restore independent OOS status."
        ),
    }

    OOS._atomic_json(out / "summary.json", overall)
    sweep.to_csv(out / "latency_sweep_summary.csv", index=False)

    for spec in SCENARIOS:
        key = str(spec["key"])
        d = out / key.lower()
        d.mkdir(parents=True, exist_ok=True)
        OOS._atomic_json(d / "scenario_summary.json", scenario_summaries[key])
        details[key].to_csv(d / "by_contract.csv", index=False)
        windows[key].to_csv(d / "by_window.csv", index=False)
        assets[key].to_csv(d / "by_asset.csv", index=False)
        fills[key].to_csv(d / "passive_fills.csv", index=False)

    if show:
        print("=" * 154)
        print("FORMAL Q10 OOS — FILL-OBSERVATION LATENCY SWEEP V12 / READ ONLY")
        print("=" * 154)
        print("Source:", source)
        print("Candidate-C / Q10 / FIFO / exit / fees: UNCHANGED")
        print("Tickers:", len(selected_tickers), "| metadata windows:", metadata_windows, "| complete 9-series windows:", complete_windows)
        print("Shared trade timing scan:")
        print("  exchange coverage:", timing_scan["stats"].get("with_exchange_time", 0), "/", timing_scan["stats"].get("trade_rows", 0))
        print("  exchange->receipt median/p95/max ms:", f"{timing_scan['lag_ms_median']:.3f}", "/", f"{timing_scan['lag_ms_p95']:.3f}", "/", f"{timing_scan['lag_ms_max']:.3f}")
        print("  exchange-after-receipt rows clamped:", timing_scan["stats"].get("exchange_after_receipt_clamped", 0))
        print("  max shared heap:", f"{max_trade_heap:,}")
        print()
        print("ORIGINAL RECEIPT-TIME SHADOW")
        print(f"  old net:     {original_net:+.4f}")
        print(f"  old / day:   {overall['original_receipt_shadow_per_day']:+.4f}")
        print()
        print("LATENCY SWEEP")
        display_cols = [
            "scenario", "fixed_delay_ms", "net_pnl", "normalized_per_day",
            "passive_pnl", "m5_gross", "taker_fees", "fill_events", "fill_qty",
            "cycles_started", "cycles_completed", "creates_while_fill_hidden",
            "cancel_after_hidden_execution", "positive_windows", "negative_windows",
            "median_window_net", "worst_window_net", "best_window_net",
        ]
        print(sweep[display_cols].to_string(index=False))
        print()
        print("PUBLIC-RECEIPT SCENARIO vs PRIOR V11")
        if prior_v11:
            print("  prior V11:", prior_v11_dir)
            print("  net delta:", v11_recon["net_delta"])
            print("  fill-events delta:", v11_recon["fill_events_delta"])
            print("  fill-qty delta:", v11_recon["fill_qty_delta"])
            print("  cycles started/completed delta:", v11_recon["cycles_started_delta"], "/", v11_recon["cycles_completed_delta"])
            print("  RECONCILIATION:", "PASS" if v11_recon["pass"] else "FAIL")
        else:
            print("  prior V11 summary not found; no cross-run reconciliation available")
        print()
        print("DIAGNOSTIC VERDICT:", diagnostic_verdict)
        print()
        print("Interpretation:")
        print("  - If 0ms and 5ms are still materially negative, hidden-fill notification latency is not the main reason Candidate-C fails on this realization.")
        print("  - If 0ms is positive but losses grow rapidly with delay, the strategy is highly fill-notification-latency sensitive.")
        print("  - PUBLIC_RECEIPT should reproduce V11. A reconciliation FAIL means do not interpret the sweep until the implementation mismatch is fixed.")
        print("  - Same old realization / posthoc sensitivity only; no fresh profitability claim is allowed.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 154)

    return {
        "summary": overall,
        "sweep": sweep,
        "scenario_summaries": scenario_summaries,
        "details": details,
        "by_window": windows,
        "by_asset": assets,
        "fills": fills,
        "output_dir": out,
    }


__all__ = [
    "VERSION",
    "HARD_BOUND_OOS_SESSION",
    "SCENARIOS",
    "run_q10_oos_fill_observation_latency_sweep",
]
