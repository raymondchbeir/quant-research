from __future__ import annotations

"""FIFO queue-position stress for frozen CYCLE_ALWAYS_EXIT Q10.

DEVELOPMENT DIAGNOSTIC ONLY. This does not change the frozen strategy or current
OOS session. It replays the clean V5 development session once while adding a
fixed number of hypothetical contracts ahead of every newly posted passive
ENTRY and EXIT quote.

The purpose is to quantify sensitivity to live order-arrival / queue-position
error. It does NOT model every latency effect (for example BBO movement between
market-data receipt and exchange order arrival); it isolates FIFO queue loss.
"""

import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .mm_event_time_c_inventory_cycle_dev_v1 import (
    CycleSim,
    EXPECTED_SESSION_NAME,
    QUOTE_SIZE,
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

STUDY_VERSION = "MM_CYCLE_Q10_QUEUE_POSITION_STRESS_V1"
DEFAULT_EXTRA_QUEUE = (0, 5, 10, 20, 30, 50, 75, 100, 150, 250)
DEFAULT_TAKER_MULTIPLIER = 1.0


class QueueStressSim(CycleSim):
    __slots__ = ("extra_queue",)

    def __init__(self, ticker: str, meta: dict, extra_queue: float):
        super().__init__("CYCLE_ALWAYS_EXIT", ticker, meta)
        self.extra_queue = float(extra_queue)

    def _open(self, side: str, t: float, cur: dict, role: str, qty: float):
        super()._open(side, t, cur, role, qty)
        ep = self.active.get(side)
        if ep is not None and self.extra_queue > 0:
            ep["queue_ahead"] = float(ep["queue_ahead"]) + self.extra_queue
            ep["queue_ahead_initial"] = float(ep["queue_ahead_initial"]) + self.extra_queue


def _quadratic_taker_fee(qty: float, price: float, multiplier: float) -> float:
    qty = float(qty)
    price = float(price)
    multiplier = float(multiplier)
    if qty <= EPS or not np.isfinite(price):
        return 0.0
    raw = 0.07 * multiplier * qty * price * (1.0 - price)
    return math.ceil(max(0.0, raw) * 10000.0 - 1e-12) / 10000.0


def _fee_adjust_contract(row: dict, multiplier: float) -> dict:
    out = dict(row)
    inv = float(out.get("ending_inventory_yes_equiv", 0.0) or 0.0)
    qty = float(out.get("residual_qty_m5", 0.0) or 0.0)
    if qty <= EPS or abs(inv) <= EPS:
        px = np.nan
        fee = 0.0
    else:
        px = float(out["m5_bid"]) if inv > 0 else float(out["m5_ask"])
        fee = _quadratic_taker_fee(qty, px, multiplier)
    out["m5_liquidation_price"] = px
    out["m5_taker_fee"] = fee
    out["net_after_taker_fee"] = float(out["bbo_liquidated_net_pnl"]) - fee
    return out


def run_queue_position_stress(
    session_dir,
    extra_queue=DEFAULT_EXTRA_QUEUE,
    *,
    taker_multiplier=DEFAULT_TAKER_MULTIPLIER,
    output_dir=None,
    show=True,
):
    session = Path(session_dir).resolve()
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"Queue stress is hard-bound to clean V5 development session "
            f"{EXPECTED_SESSION_NAME}; got {session.name}."
        )
    if not session.exists():
        raise FileNotFoundError(session)

    penalties = tuple(sorted({float(x) for x in extra_queue if float(x) >= 0.0}))
    if 0.0 not in penalties:
        penalties = (0.0,) + penalties

    _, _, _, quality, eligible = _load_inputs(session)
    meta = _load_meta(session, eligible)
    close_windows = sorted({meta[t]["close_ts"] for t in eligible})
    exposure_hours = 0.25 * len(close_windows)
    day_scale = 24.0 / exposure_hours

    if show:
        print(
            f"Eligible contracts={len(eligible)} | windows={len(close_windows)} | "
            f"exposure={exposure_hours:.2f}h"
        )
        print("Frozen strategy: CYCLE_ALWAYS_EXIT | Candidate C entry | Q10")
        print("Extra FIFO contracts ahead:", penalties)
        print("Loading aggressive trades...")

    trades = _load_research_trades(session, eligible, meta)

    sims = {
        (ticker, penalty): QueueStressSim(ticker, meta[ticker], penalty)
        for ticker in eligible
        for penalty in penalties
    }
    sim_groups = {
        ticker: [sims[(ticker, penalty)] for penalty in penalties]
        for ticker in eligible
    }

    trade_i = defaultdict(int)
    pending = {(ticker, penalty): [] for ticker in eligible for penalty in penalties}
    pending_counter = 0
    current_mid = defaultdict(lambda: np.nan)

    if show:
        print("Streaming book once for all queue-stress scenarios...")

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
                    for penalty, sim in zip(penalties, sim_groups[ticker]):
                        pending_counter = sim.apply_trade(
                            tr_t,
                            tr_px,
                            tr_qty,
                            tr_side,
                            current_mid[ticker],
                            pending[(ticker, penalty)],
                            pending_counter,
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
            for penalty in penalties:
                _resolve_pending(
                    pending[(ticker, penalty)],
                    t,
                    float(cur["mid"]),
                )

            if event_type == "trade_window_end" or (
                np.isfinite(elapsed) and 299.5 <= elapsed <= 302.0
            ):
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
                for penalty, sim in zip(penalties, sim_groups[ticker]):
                    pending_counter = sim.apply_trade(
                        tr_t,
                        tr_px,
                        tr_qty,
                        tr_side,
                        current_mid[ticker],
                        pending[(ticker, penalty)],
                        pending_counter,
                    )
            j += 1
        for sim in sim_groups[ticker]:
            sim.cancel_all("FILE_END")

    contract_rows = []
    fill_rows = []
    cycle_rows = []
    for ticker in sorted(eligible, key=lambda x: (meta[x]["close_ts"], x)):
        for penalty in penalties:
            sim = sims[(ticker, penalty)]
            cr = _fee_adjust_contract(_contract_result(sim), taker_multiplier)
            cr["extra_queue_ahead"] = penalty
            contract_rows.append(cr)
            for f in sim.fills:
                z = dict(f)
                z["extra_queue_ahead"] = penalty
                fill_rows.append(z)
            for c in sim.cycle_rows():
                z = dict(c)
                z["extra_queue_ahead"] = penalty
                cycle_rows.append(z)

    contracts = pd.DataFrame(contract_rows)
    fills = pd.DataFrame(fill_rows)
    cycles = pd.DataFrame(cycle_rows)

    rows = []
    for penalty in penalties:
        cdf = contracts[contracts.extra_queue_ahead == penalty].copy()
        fdf = fills[fills.extra_queue_ahead == penalty].copy() if len(fills) else pd.DataFrame()

        prefee = float(pd.to_numeric(cdf.bbo_liquidated_net_pnl, errors="coerce").sum())
        fees = float(pd.to_numeric(cdf.m5_taker_fee, errors="coerce").sum())
        net = float(pd.to_numeric(cdf.net_after_taker_fee, errors="coerce").sum())
        matched = float(pd.to_numeric(cdf.matched_roundtrip_pnl, errors="coerce").sum())
        residual = float(pd.to_numeric(cdf.bbo_residual_liquidation_pnl, errors="coerce").sum())
        residual_qty = float(pd.to_numeric(cdf.residual_qty_m5, errors="coerce").sum())
        starts = int(pd.to_numeric(cdf.cycles_started, errors="coerce").sum())
        completed = int(pd.to_numeric(cdf.cycles_completed, errors="coerce").sum())
        fill_qty = float(pd.to_numeric(fdf.qty, errors="coerce").sum()) if len(fdf) else 0.0
        forced = int((pd.to_numeric(cdf.residual_qty_m5, errors="coerce") > EPS).sum())

        row = {
            "extra_queue_ahead": penalty,
            "fill_events": len(fdf),
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
        }
        for h in MARKOUTS_S:
            tag = f"{int(h)}s"
            row[f"qw_markout_{tag}_c"] = _wavg(fdf, f"markout_{tag}_c") if len(fdf) else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("extra_queue_ahead").reset_index(drop=True)
    base = summary.loc[summary.extra_queue_ahead == 0.0].iloc[0]
    base_net = float(base.fee_adjusted_net_per_day)
    base_fill = float(base.fill_qty_per_day)
    summary["pnl_loss_vs_q0_per_day"] = base_net - summary.fee_adjusted_net_per_day
    summary["pnl_retention_vs_q0_pct"] = (
        100.0 * summary.fee_adjusted_net_per_day / base_net if abs(base_net) > EPS else np.nan
    )
    summary["fill_qty_retention_vs_q0_pct"] = (
        100.0 * summary.fill_qty_per_day / base_fill if base_fill > EPS else np.nan
    )

    if output_dir is None:
        output_dir = (
            Path(session).parents[3]
            / "results"
            / "kalshi_mm_cycle_q10_queue_position_stress"
            / session.name
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "queue_stress_summary.csv", index=False)
    contracts.to_csv(out / "queue_stress_contracts.csv", index=False)
    fills.to_csv(out / "queue_stress_fills.csv", index=False)
    cycles.to_csv(out / "queue_stress_cycles.csv", index=False)

    spec = {
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "strategy": "frozen CYCLE_ALWAYS_EXIT / Candidate C entry / Q10",
        "stress": "add fixed contracts ahead to every new passive entry and exit quote",
        "extra_queue_grid": list(penalties),
        "taker_multiplier": float(taker_multiplier),
        "maker_fee": 0.0,
        "warning": "Queue-position sensitivity only; does not model BBO changes/rejections/network outages caused by latency.",
        "oos_strategy_changed": False,
    }
    (out / "study_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    if show:
        print("\n" + "=" * 150)
        print("FROZEN CYCLE_ALWAYS_EXIT Q10 — FIFO QUEUE-POSITION STRESS")
        print("=" * 150)
        cols = [
            "extra_queue_ahead",
            "fill_events",
            "fill_qty_per_day",
            "fill_qty_retention_vs_q0_pct",
            "cycle_completion_pct",
            "forced_m5_contracts",
            "residual_qty_m5",
            "matched_pnl_per_day",
            "m5_taker_fee_per_day",
            "fee_adjusted_net_per_day",
            "pnl_loss_vs_q0_per_day",
            "pnl_retention_vs_q0_pct",
            "qw_markout_5s_c",
            "qw_markout_15s_c",
            "qw_markout_30s_c",
        ]
        print(summary[cols].round(4).to_string(index=False))
        print("\nInterpretation: +N means every new passive quote lands N contracts farther back than the historical displayed-L1 assumption.")
        print("This is a queue-only latency stress; it does not model stale-price/post-only rejection effects.")
        print("Outputs:", out)
        print("=" * 150)

    return {
        "summary": summary,
        "contracts": contracts,
        "fills": fills,
        "cycles": cycles,
        "output_dir": out,
        "spec": spec,
    }
