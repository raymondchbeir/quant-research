from __future__ import annotations

"""Asymmetric ENTRY-vs-EXIT FIFO queue stress for frozen CYCLE_ALWAYS_EXIT Q10.

DEVELOPMENT DIAGNOSTIC ONLY. This does not alter the frozen OOS strategy/session.
It replays the clean V5 development session once across a fixed grid of extra
contracts ahead, allowing ENTRY and EXIT queue-placement error to differ.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .mm_event_time_c_inventory_cycle_dev_v1 import (
    CycleSim,
    EXPECTED_SESSION_NAME,
    _contract_result,
    _top_state_ext,
)
from .mm_event_time_abcd_capacity_dev_v1 import (
    EPS,
    MARKOUTS_S,
    _f,
    _load_inputs,
    _load_meta,
    _load_research_trades,
    _resolve_pending,
    _ts,
    _wavg,
)
from .mm_cycle_q10_queue_position_stress_v1 import (
    _fee_adjust_contract,
    DEFAULT_TAKER_MULTIPLIER,
)

STUDY_VERSION = "MM_CYCLE_Q10_ENTRY_EXIT_QUEUE_STRESS_V1"

DEFAULT_SCENARIOS = (
    ("BASE", 0, 0),
    ("ENTRY_5", 5, 0),
    ("ENTRY_10", 10, 0),
    ("ENTRY_20", 20, 0),
    ("ENTRY_30", 30, 0),
    ("EXIT_5", 0, 5),
    ("EXIT_10", 0, 10),
    ("EXIT_20", 0, 20),
    ("EXIT_30", 0, 30),
    ("ENTRY5_EXIT10", 5, 10),
    ("ENTRY10_EXIT5", 10, 5),
    ("ENTRY10_EXIT10", 10, 10),
    ("ENTRY10_EXIT20", 10, 20),
    ("ENTRY20_EXIT10", 20, 10),
    ("ENTRY20_EXIT20", 20, 20),
)


class AsymmetricQueueStressSim(CycleSim):
    __slots__ = ("scenario", "entry_extra_queue", "exit_extra_queue")

    def __init__(self, ticker, meta, scenario, entry_extra_queue, exit_extra_queue):
        super().__init__("CYCLE_ALWAYS_EXIT", ticker, meta)
        self.scenario = str(scenario)
        self.entry_extra_queue = float(entry_extra_queue)
        self.exit_extra_queue = float(exit_extra_queue)

    def _open(self, side: str, t: float, cur: dict, role: str, qty: float):
        super()._open(side, t, cur, role, qty)
        ep = self.active.get(side)
        if ep is None:
            return
        extra = self.entry_extra_queue if role == "ENTRY" else self.exit_extra_queue
        if extra > 0:
            ep["queue_ahead"] = float(ep["queue_ahead"]) + extra
            ep["queue_ahead_initial"] = float(ep["queue_ahead_initial"]) + extra


def run_entry_exit_queue_stress(
    session_dir,
    scenarios=DEFAULT_SCENARIOS,
    *,
    taker_multiplier=DEFAULT_TAKER_MULTIPLIER,
    output_dir=None,
    show=True,
):
    session = Path(session_dir).resolve()
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"Stress is hard-bound to V5 development session {EXPECTED_SESSION_NAME}; got {session.name}."
        )
    if not session.exists():
        raise FileNotFoundError(session)

    parsed = []
    seen = set()
    for name, entry_q, exit_q in scenarios:
        key = str(name)
        if key in seen:
            raise ValueError(f"Duplicate scenario name: {key}")
        seen.add(key)
        entry_q, exit_q = float(entry_q), float(exit_q)
        if entry_q < 0 or exit_q < 0:
            raise ValueError("Queue penalties must be non-negative")
        parsed.append((key, entry_q, exit_q))
    if not any(e == 0 and x == 0 for _, e, x in parsed):
        parsed.insert(0, ("BASE", 0.0, 0.0))

    _, _, _, quality, eligible = _load_inputs(session)
    meta = _load_meta(session, eligible)
    close_windows = sorted({meta[t]["close_ts"] for t in eligible})
    exposure_hours = 0.25 * len(close_windows)
    day_scale = 24.0 / exposure_hours

    if show:
        print(f"Eligible contracts={len(eligible)} | windows={len(close_windows)} | exposure={exposure_hours:.2f}h")
        print("Frozen strategy: Candidate C | CYCLE_ALWAYS_EXIT | Q10")
        print("Scenarios:")
        for name, e, x in parsed:
            print(f"  {name:18s} entry+{e:g}  exit+{x:g}")
        print("Loading aggressive trades...")

    trades = _load_research_trades(session, eligible, meta)

    sims = {
        (ticker, name): AsymmetricQueueStressSim(ticker, meta[ticker], name, e, x)
        for ticker in eligible
        for name, e, x in parsed
    }
    sim_groups = {ticker: [sims[(ticker, name)] for name, _, _ in parsed] for ticker in eligible}
    scenario_names = [name for name, _, _ in parsed]

    trade_i = defaultdict(int)
    pending = {(ticker, name): [] for ticker in eligible for name in scenario_names}
    pending_counter = 0
    current_mid = defaultdict(lambda: np.nan)

    if show:
        print("Streaming book once for all asymmetric queue scenarios...")

    with (session / "book_top3_events.jsonl").open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ticker = str(r.get("ticker") or "")
            if ticker not in eligible:
                continue
            t = _ts(r.get("receipt_time"))
            if not np.isfinite(t):
                continue
            t = float(t)

            arr = trades.get(ticker, [])
            j = trade_i[ticker]
            while j < len(arr) and arr[j][0] < t - EPS:
                tr_t, tr_px, tr_qty, tr_side = arr[j]
                if tr_t < meta[ticker]["m5_ts"] - EPS:
                    for name, sim in zip(scenario_names, sim_groups[ticker]):
                        pending_counter = sim.apply_trade(
                            tr_t, tr_px, tr_qty, tr_side, current_mid[ticker],
                            pending[(ticker, name)], pending_counter,
                        )
                j += 1
            trade_i[ticker] = j

            cur = _top_state_ext(r)
            elapsed = _f(r.get("elapsed_s"))
            event_type = str(r.get("event_type") or "")
            in_research = bool(np.isfinite(elapsed) and 0.0 <= elapsed < 300.0)

            if cur is None:
                if in_research:
                    for sim in sim_groups[ticker]:
                        sim.cancel_all("INVALID_BOOK")
                continue

            current_mid[ticker] = float(cur["mid"])
            for name in scenario_names:
                _resolve_pending(pending[(ticker, name)], t, float(cur["mid"]))

            if event_type == "trade_window_end" or (np.isfinite(elapsed) and 299.5 <= elapsed <= 302.0):
                for sim in sim_groups[ticker]:
                    sim.set_m5_state(t, cur)

            for sim in sim_groups[ticker]:
                sim.on_book(t, cur, in_research)

            if show and n % 1_000_000 == 0:
                print(f"  streamed {n:,} book rows")

    for ticker in sorted(eligible):
        arr = trades.get(ticker, [])
        j = trade_i[ticker]
        while j < len(arr):
            tr_t, tr_px, tr_qty, tr_side = arr[j]
            if tr_t < meta[ticker]["m5_ts"] - EPS:
                for name, sim in zip(scenario_names, sim_groups[ticker]):
                    pending_counter = sim.apply_trade(
                        tr_t, tr_px, tr_qty, tr_side, current_mid[ticker],
                        pending[(ticker, name)], pending_counter,
                    )
            j += 1
        for sim in sim_groups[ticker]:
            sim.cancel_all("FILE_END")

    scenario_map = {name: (e, x) for name, e, x in parsed}
    contract_rows, fill_rows, cycle_rows = [], [], []
    for ticker in sorted(eligible, key=lambda z: (meta[z]["close_ts"], z)):
        for name in scenario_names:
            sim = sims[(ticker, name)]
            cr = _fee_adjust_contract(_contract_result(sim), taker_multiplier)
            e, x = scenario_map[name]
            cr.update({"scenario": name, "entry_extra_queue": e, "exit_extra_queue": x})
            contract_rows.append(cr)
            for f in sim.fills:
                z = dict(f)
                z.update({"scenario": name, "entry_extra_queue": e, "exit_extra_queue": x})
                fill_rows.append(z)
            for c in sim.cycle_rows():
                z = dict(c)
                z.update({"scenario": name, "entry_extra_queue": e, "exit_extra_queue": x})
                cycle_rows.append(z)

    contracts = pd.DataFrame(contract_rows)
    fills = pd.DataFrame(fill_rows)
    cycles = pd.DataFrame(cycle_rows)

    rows = []
    for name, entry_q, exit_q in parsed:
        cdf = contracts[contracts.scenario == name].copy()
        fdf = fills[fills.scenario == name].copy() if len(fills) else pd.DataFrame()
        prefee = float(pd.to_numeric(cdf.bbo_liquidated_net_pnl, errors="coerce").sum())
        fees = float(pd.to_numeric(cdf.m5_taker_fee, errors="coerce").sum())
        net = float(pd.to_numeric(cdf.net_after_taker_fee, errors="coerce").sum())
        matched = float(pd.to_numeric(cdf.matched_roundtrip_pnl, errors="coerce").sum())
        residual = float(pd.to_numeric(cdf.bbo_residual_liquidation_pnl, errors="coerce").sum())
        residual_qty = float(pd.to_numeric(cdf.residual_qty_m5, errors="coerce").sum())
        starts = int(pd.to_numeric(cdf.cycles_started, errors="coerce").sum())
        completed = int(pd.to_numeric(cdf.cycles_completed, errors="coerce").sum())
        fill_qty = float(pd.to_numeric(fdf.qty, errors="coerce").sum()) if len(fdf) else 0.0
        entry_f = fdf[fdf.role == "ENTRY"] if len(fdf) else fdf
        exit_f = fdf[fdf.role == "EXIT"] if len(fdf) else fdf
        forced = int((pd.to_numeric(cdf.residual_qty_m5, errors="coerce") > EPS).sum())

        row = {
            "scenario": name,
            "entry_extra_queue": entry_q,
            "exit_extra_queue": exit_q,
            "fill_events": len(fdf),
            "entry_fill_events": len(entry_f),
            "exit_fill_events": len(exit_f),
            "fill_qty": fill_qty,
            "fill_qty_per_day": fill_qty * day_scale,
            "cycles_started": starts,
            "cycles_completed": completed,
            "cycle_completion_pct": 100.0 * completed / starts if starts else np.nan,
            "forced_m5_contracts": forced,
            "residual_qty_m5": residual_qty,
            "prefee_net_per_day": prefee * day_scale,
            "matched_pnl_per_day": matched * day_scale,
            "residual_gross_pnl_per_day": residual * day_scale,
            "m5_taker_fee_per_day": fees * day_scale,
            "fee_adjusted_net_per_day": net * day_scale,
            "net_cents_per_filled_contract_after_fee": 100.0 * net / fill_qty if fill_qty > EPS else np.nan,
            "entry_queue_p50": pd.to_numeric(entry_f.queue_ahead_initial, errors="coerce").median() if len(entry_f) else np.nan,
            "exit_queue_p50": pd.to_numeric(exit_f.queue_ahead_initial, errors="coerce").median() if len(exit_f) else np.nan,
        }
        for h in MARKOUTS_S:
            tag = f"{int(h)}s"
            row[f"qw_markout_{tag}_c"] = _wavg(fdf, f"markout_{tag}_c") if len(fdf) else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    base = summary[(summary.entry_extra_queue == 0) & (summary.exit_extra_queue == 0)].iloc[0]
    base_net = float(base.fee_adjusted_net_per_day)
    base_fill = float(base.fill_qty_per_day)
    summary["pnl_loss_vs_base_per_day"] = base_net - summary.fee_adjusted_net_per_day
    summary["pnl_retention_vs_base_pct"] = 100.0 * summary.fee_adjusted_net_per_day / base_net if abs(base_net) > EPS else np.nan
    summary["fill_qty_retention_vs_base_pct"] = 100.0 * summary.fill_qty_per_day / base_fill if base_fill > EPS else np.nan

    order = {name: i for i, name in enumerate(scenario_names)}
    summary["_order"] = summary.scenario.map(order)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    if output_dir is None:
        output_dir = Path(session).parents[3] / "results" / "kalshi_mm_cycle_q10_entry_exit_queue_stress" / session.name
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "entry_exit_queue_stress_summary.csv", index=False)
    contracts.to_csv(out / "entry_exit_queue_stress_contracts.csv", index=False)
    fills.to_csv(out / "entry_exit_queue_stress_fills.csv", index=False)
    cycles.to_csv(out / "entry_exit_queue_stress_cycles.csv", index=False)

    spec = {
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "strategy": "frozen Candidate C + CYCLE_ALWAYS_EXIT + Q10",
        "scenarios": [{"name": n, "entry_extra_queue": e, "exit_extra_queue": x} for n, e, x in parsed],
        "stress": "fixed extra FIFO contracts ahead, independently for ENTRY and EXIT quotes",
        "taker_multiplier": float(taker_multiplier),
        "maker_fee": 0.0,
        "warning": "Queue-position sensitivity only; does not model stale BBO/post-only rejection/network latency directly.",
        "oos_strategy_changed": False,
    }
    (out / "study_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    if show:
        print("\n" + "=" * 170)
        print("FROZEN CYCLE_ALWAYS_EXIT Q10 — ENTRY VS EXIT QUEUE STRESS")
        print("=" * 170)
        cols = [
            "scenario", "entry_extra_queue", "exit_extra_queue",
            "entry_fill_events", "exit_fill_events", "fill_qty_retention_vs_base_pct",
            "cycle_completion_pct", "forced_m5_contracts", "residual_qty_m5",
            "matched_pnl_per_day", "m5_taker_fee_per_day", "fee_adjusted_net_per_day",
            "pnl_loss_vs_base_per_day", "pnl_retention_vs_base_pct",
            "qw_markout_5s_c", "qw_markout_15s_c", "qw_markout_30s_c",
        ]
        print(summary[cols].round(4).to_string(index=False))
        print("\nENTRY-only rows isolate worse arrival priority when flat.")
        print("EXIT-only rows isolate worse priority while carrying inventory.")
        print("Mixed rows show realistic asymmetric combinations.")
        print("Outputs:", out)
        print("=" * 170)

    return {
        "summary": summary,
        "contracts": contracts,
        "fills": fills,
        "cycles": cycles,
        "output_dir": out,
        "spec": spec,
    }
