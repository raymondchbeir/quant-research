from __future__ import annotations

"""Candidate C inventory-cycle development replay.

DEVELOPMENT ONLY -- hard-bound to V5 development session 20260816_070627.

Why this exists
---------------
The pre-registered A/B/C/D development family found that Candidate C
(L3 side support + natural spread >= 2c) had positive matched economics but
failed the >=$100/day requirement after executable M5 liquidation because too
much value depended on residual inventory.

This script changes ZERO entry-signal parameters. It tests one structural idea:
after the first passive fill, stop adding inventory and work only the opposite
side until flat.

The three exit mechanics are pre-registered here BEFORE viewing results:

    CYCLE_ALWAYS_EXIT
        Flat: enter using Candidate C.
        Inventory: quote ONLY the opposite BBO until flat, regardless of L3.

    CYCLE_L3_EXIT
        Flat: enter using Candidate C.
        Inventory: quote ONLY the opposite BBO when opposite-side L3 support
        is present. No spread condition is imposed on the exit.

    CYCLE_C_EXIT
        Flat: enter using Candidate C.
        Inventory: quote ONLY the opposite BBO when the full opposite-side
        Candidate C condition is present (L3 support + natural spread >=2c).

All variants use Q10 only. This is not a size sweep.

Execution convention
--------------------
- Public YES BBO, Q10 entry.
- Exit quote quantity equals current absolute inventory (never adds risk).
- Join behind displayed L1 quantity.
- Exact-price opposing aggressive trades consume queue ahead first.
- Trade-through fills the remaining hypothetical quote.
- No cancellation-ahead credit.
- Any fill cancels residual same-side quote; state is re-evaluated causally on
  the next book event.
- No fees.
- Any inventory still open at M5 is liquidated at the observed executable M5
  BBO for the primary PnL estimate. Recorded top-3 depth coverage is reported;
  deeper liquidity is never fabricated.

This remains counterfactual historical-flow research. Passing development gates
would only justify freezing ONE exact variant and collecting fresh OOS data.
"""

import heapq
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from .mm_event_time_abcd_capacity_dev_v1 import (
    BTC_SERIES,
    EPS,
    MARKOUTS_S,
    _f,
    _iso,
    _load_inputs,
    _load_meta,
    _load_research_trades,
    _resolve_pending,
    _top_state,
    _ts,
    _wavg,
)

STUDY_VERSION = "MM_EVENT_TIME_C_INVENTORY_CYCLE_DEV_V1"
EXPECTED_SESSION_NAME = "20260816_070627"
QUOTE_SIZE = 10.0
SPREAD_FLOOR_C = 2.0  # inherited Candidate C rule; NOT re-tuned here
ECONOMIC_TARGET_USD_PER_DAY = 100.0

VARIANTS = (
    "CYCLE_ALWAYS_EXIT",
    "CYCLE_L3_EXIT",
    "CYCLE_C_EXIT",
)

# Freeze-review gates are declared before results are read.
MIN_MATCHED_SHARE_OF_POSITIVE_NET = 0.50
MIN_CYCLE_COMPLETION_PCT = 90.0
MIN_TOP3_RESIDUAL_QTY_COVERAGE_PCT = 95.0


def _candidate_c_allows(side: str, cur: dict) -> bool:
    if side == "BID":
        l3 = cur["bid_depth3"] > cur["ask_depth3"] + EPS
    elif side == "ASK":
        l3 = cur["ask_depth3"] > cur["bid_depth3"] + EPS
    else:
        return False
    return bool(l3 and cur["spread_c"] + 1e-9 >= SPREAD_FLOOR_C)


def _l3_allows(side: str, cur: dict) -> bool:
    if side == "BID":
        return bool(cur["bid_depth3"] > cur["ask_depth3"] + EPS)
    if side == "ASK":
        return bool(cur["ask_depth3"] > cur["bid_depth3"] + EPS)
    return False


def _exit_allows(variant: str, side: str, cur: dict) -> bool:
    if variant == "CYCLE_ALWAYS_EXIT":
        return True
    if variant == "CYCLE_L3_EXIT":
        return _l3_allows(side, cur)
    if variant == "CYCLE_C_EXIT":
        return _candidate_c_allows(side, cur)
    raise RuntimeError(f"Unknown variant: {variant}")


def _top_state_ext(row):
    cur = _top_state(row)
    if cur is None:
        return None
    try:
        bids = [
            (float(x[0]), max(0.0, float(x[1])))
            for x in (row.get("bid_levels") or [])[:3]
            if len(x) >= 2
        ]
        asks = [
            (float(x[0]), max(0.0, float(x[1])))
            for x in (row.get("ask_levels") or [])[:3]
            if len(x) >= 2
        ]
    except Exception:
        return None
    if not bids or not asks:
        return None
    cur = dict(cur)
    cur["bid_levels"] = bids
    cur["ask_levels"] = asks
    return cur


class CycleSim:
    __slots__ = (
        "variant", "ticker", "meta", "active", "fills", "counts", "inventory",
        "cash", "max_abs_inventory", "last_research_state", "last_research_t",
        "m5_state", "m5_state_t", "episode_no", "cycle_no", "cycle_open",
        "cycles",
    )

    def __init__(self, variant: str, ticker: str, meta: dict):
        self.variant = variant
        self.ticker = ticker
        self.meta = meta
        self.active = {"BID": None, "ASK": None}
        self.fills = []
        self.counts = Counter()
        self.inventory = 0.0
        self.cash = 0.0
        self.max_abs_inventory = 0.0
        self.last_research_state = None
        self.last_research_t = np.nan
        self.m5_state = None
        self.m5_state_t = np.nan
        self.episode_no = 0
        self.cycle_no = 0
        self.cycle_open = None
        self.cycles = []

    def cancel(self, side: str, reason: str):
        if self.active[side] is not None:
            self.active[side] = None
            self.counts[f"{side}_CANCEL_{reason}"] += 1

    def cancel_all(self, reason: str):
        self.cancel("BID", reason)
        self.cancel("ASK", reason)

    def _open(self, side: str, t: float, cur: dict, role: str, qty: float):
        qty = float(qty)
        if qty <= EPS:
            return
        px = cur["bid"] if side == "BID" else cur["ask"]
        qahead = cur["bid_q1"] if side == "BID" else cur["ask_q1"]
        self.episode_no += 1
        self.active[side] = {
            "role": role,
            "price": float(px),
            "queue_ahead": float(qahead),
            "queue_ahead_initial": float(qahead),
            "remaining_qty": qty,
            "join_ts": float(t),
            "spread_c_at_join": float(cur["spread_c"]),
            "l3_imbalance_at_join": (
                (cur["bid_depth3"] - cur["ask_depth3"])
                / (cur["bid_depth3"] + cur["ask_depth3"])
                if cur["bid_depth3"] + cur["ask_depth3"] > EPS else 0.0
            ),
        }
        self.counts[f"{role}_{side}_OPEN"] += 1

    def _maintain(self, side: str, t: float, cur: dict, *, allow: bool, role: str, qty: float):
        desired_px = cur["bid"] if side == "BID" else cur["ask"]
        ep = self.active[side]
        if ep is not None:
            stale = (
                ep.get("role") != role
                or abs(float(ep["price"]) - float(desired_px)) > EPS
                or abs(float(ep["remaining_qty"]) - float(qty)) > 1e-9
            )
            if stale:
                self.cancel(side, "REPRICE_OR_ROLE_OR_QTY")
        if self.active[side] is not None and not allow:
            self.cancel(side, "STATE_FALSE")
        if self.active[side] is None and allow and qty > EPS:
            self._open(side, t, cur, role, qty)

    def set_m5_state(self, t: float, cur: dict):
        self.m5_state = dict(cur)
        self.m5_state_t = float(t)

    def on_book(self, t: float, cur: dict, in_research: bool):
        if in_research:
            self.last_research_state = dict(cur)
            self.last_research_t = float(t)
        else:
            self.cancel_all("M5_END")
            return

        inv = float(self.inventory)
        if abs(inv) <= 1e-9:
            self.inventory = 0.0
            # Flat: Candidate C controls entry. Because L3 support is strict,
            # at most one side can be supported at a time.
            for side in ("BID", "ASK"):
                self._maintain(
                    side,
                    t,
                    cur,
                    allow=_candidate_c_allows(side, cur),
                    role="ENTRY",
                    qty=QUOTE_SIZE,
                )
            return

        # Inventory: NEVER quote the risk-increasing side.
        if inv > 0.0:
            self.cancel("BID", "RISK_INCREASING_BLOCKED")
            exit_side = "ASK"
        else:
            self.cancel("ASK", "RISK_INCREASING_BLOCKED")
            exit_side = "BID"

        self._maintain(
            exit_side,
            t,
            cur,
            allow=_exit_allows(self.variant, exit_side, cur),
            role="EXIT",
            qty=abs(inv),
        )

    def _start_cycle(self, t: float, side: str, qty: float, px: float):
        if self.cycle_open is not None:
            raise RuntimeError("Entry fill occurred while a cycle was already open")
        self.cycle_no += 1
        self.cycle_open = {
            "cycle_id": self.cycle_no,
            "start_ts": float(t),
            "entry_side": side,
            "entry_qty": float(qty),
            "entry_px": float(px),
        }
        self.counts["CYCLES_STARTED"] += 1

    def _maybe_complete_cycle(self, t: float):
        if abs(self.inventory) > 1e-9 or self.cycle_open is None:
            return
        c = dict(self.cycle_open)
        c.update({
            "variant": self.variant,
            "ticker": self.ticker,
            "series": self.meta["series"],
            "close_ts": self.meta["close_ts"],
            "close_time": _iso(self.meta["close_ts"]),
            "completed": True,
            "end_ts": float(t),
            "duration_s": float(t) - float(c["start_ts"]),
        })
        self.cycles.append(c)
        self.counts["CYCLES_COMPLETED"] += 1
        self.cycle_open = None
        self.inventory = 0.0

    def apply_trade(
        self,
        tr_t: float,
        trade_px: float,
        trade_qty: float,
        taker_side: str,
        mid_at_fill: float,
        pending_heap,
        pending_counter: int,
    ) -> int:
        side = "BID" if taker_side == "ask" else "ASK" if taker_side == "bid" else None
        if side is None:
            return pending_counter
        ep = self.active[side]
        if ep is None:
            return pending_counter

        qpx = float(ep["price"])
        role = str(ep["role"])
        trade_through = False
        aggressive_available = 0.0

        if side == "BID":
            if trade_px < qpx - EPS:
                trade_through = True
                aggressive_available = float(ep["remaining_qty"])
            elif abs(trade_px - qpx) <= EPS:
                ahead = float(ep["queue_ahead"])
                used = min(ahead, trade_qty)
                ep["queue_ahead"] = ahead - used
                aggressive_available = max(0.0, trade_qty - used)
            else:
                return pending_counter
        else:
            if trade_px > qpx + EPS:
                trade_through = True
                aggressive_available = float(ep["remaining_qty"])
            elif abs(trade_px - qpx) <= EPS:
                ahead = float(ep["queue_ahead"])
                used = min(ahead, trade_qty)
                ep["queue_ahead"] = ahead - used
                aggressive_available = max(0.0, trade_qty - used)
            else:
                return pending_counter

        if aggressive_available <= EPS:
            return pending_counter
        qty = min(float(ep["remaining_qty"]), float(aggressive_available))
        if qty <= EPS:
            return pending_counter

        inv_before = float(self.inventory)
        sign = 1.0 if side == "BID" else -1.0

        if role == "ENTRY":
            if abs(inv_before) > 1e-9:
                raise RuntimeError("ENTRY fill while non-flat")
            self._start_cycle(tr_t, side, qty, qpx)
        elif role == "EXIT":
            # Exit must reduce risk. Guard against accidental overshoot/addition.
            if inv_before > 0 and side != "ASK":
                raise RuntimeError("Long inventory received non-ASK exit fill")
            if inv_before < 0 and side != "BID":
                raise RuntimeError("Short inventory received non-BID exit fill")
            qty = min(qty, abs(inv_before))
            if qty <= EPS:
                self.cancel(side, "ZERO_EXIT_QTY")
                return pending_counter
        else:
            raise RuntimeError(f"Unknown quote role: {role}")

        self.inventory += sign * qty
        if abs(self.inventory) <= 1e-9:
            self.inventory = 0.0
        self.cash += (-qpx * qty) if side == "BID" else (qpx * qty)
        self.max_abs_inventory = max(self.max_abs_inventory, abs(self.inventory), abs(inv_before))

        elapsed = float(tr_t) - float(self.meta["m0_ts"])
        fill = {
            "variant": self.variant,
            "quote_size": QUOTE_SIZE,
            "ticker": self.ticker,
            "series": self.meta["series"],
            "close_ts": self.meta["close_ts"],
            "close_time": _iso(self.meta["close_ts"]),
            "cycle_id": self.cycle_open["cycle_id"] if self.cycle_open else self.cycle_no,
            "role": role,
            "side": side,
            "fill_ts": float(tr_t),
            "fill_time": _iso(tr_t),
            "elapsed_s": elapsed,
            "qty": float(qty),
            "price": qpx,
            "mid_at_fill": float(mid_at_fill) if np.isfinite(mid_at_fill) else np.nan,
            "gross_edge_at_fill_c": (
                sign * (float(mid_at_fill) - qpx) * 100.0
                if np.isfinite(mid_at_fill) else np.nan
            ),
            "trade_through": bool(trade_through),
            "queue_ahead_initial": float(ep["queue_ahead_initial"]),
            "spread_c_at_join": float(ep["spread_c_at_join"]),
            "l3_imbalance_at_join": float(ep["l3_imbalance_at_join"]),
            "inventory_before_fill": inv_before,
            "inventory_after_fill": float(self.inventory),
            "historical_trade_price": float(trade_px),
            "historical_trade_qty": float(trade_qty),
        }
        for h in MARKOUTS_S:
            tag = f"{int(h)}s"
            fill[f"future_mid_{tag}"] = np.nan
            fill[f"markout_{tag}_c"] = np.nan
            fill[f"post_mid_move_{tag}_c"] = np.nan
            pending_counter += 1
            heapq.heappush(
                pending_heap,
                (float(tr_t) + h, pending_counter, fill, h, sign),
            )

        self.fills.append(fill)
        self.counts[f"{role}_FILL_EVENTS"] += 1
        self.counts[f"{role}_FILL_QTY_X1000"] += int(round(qty * 1000.0))
        if trade_through:
            self.counts[f"{role}_TRADE_THROUGH_FILL"] += 1
        else:
            self.counts[f"{role}_EXACT_PRICE_FILL"] += 1

        # Conservative replay convention: after any fill, cancel residual and
        # re-evaluate state on the next book event.
        self.cancel_all(f"{role}_FILL_STATE_SWITCH")
        if role == "EXIT":
            self._maybe_complete_cycle(tr_t)
        return pending_counter

    def cycle_rows(self):
        rows = list(self.cycles)
        if self.cycle_open is not None:
            c = dict(self.cycle_open)
            end_ts = self.meta["m5_ts"]
            c.update({
                "variant": self.variant,
                "ticker": self.ticker,
                "series": self.meta["series"],
                "close_ts": self.meta["close_ts"],
                "close_time": _iso(self.meta["close_ts"]),
                "completed": False,
                "end_ts": end_ts,
                "duration_s": max(0.0, end_ts - float(c["start_ts"])),
            })
            rows.append(c)
        return rows


def _fifo_open_lots(fills):
    long_lots = deque()
    short_lots = deque()
    matched = 0.0
    for f in sorted(fills, key=lambda x: (float(x["fill_ts"]), 0 if x["side"] == "BID" else 1)):
        qty = float(f["qty"])
        px = float(f["price"])
        if f["side"] == "BID":
            rem = qty
            while rem > EPS and short_lots:
                sq, spx = short_lots[0]
                m = min(rem, sq)
                matched += (spx - px) * m
                rem -= m
                sq -= m
                if sq <= EPS:
                    short_lots.popleft()
                else:
                    short_lots[0] = (sq, spx)
            if rem > EPS:
                long_lots.append((rem, px))
        else:
            rem = qty
            while rem > EPS and long_lots:
                lq, lpx = long_lots[0]
                m = min(rem, lq)
                matched += (px - lpx) * m
                rem -= m
                lq -= m
                if lq <= EPS:
                    long_lots.popleft()
                else:
                    long_lots[0] = (lq, lpx)
            if rem > EPS:
                short_lots.append((rem, px))
    return float(matched), list(long_lots), list(short_lots)


def _liquidate_residual(long_lots, short_lots, state):
    residual_qty = sum(q for q, _ in long_lots) + sum(q for q, _ in short_lots)
    if residual_qty <= EPS:
        return {
            "residual_qty": 0.0,
            "bbo_residual_pnl": 0.0,
            "top3_known_residual_pnl": 0.0,
            "top3_covered_qty": 0.0,
            "top3_coverage_pct": 100.0,
            "top3_full_cover": True,
        }
    if state is None:
        return {
            "residual_qty": residual_qty,
            "bbo_residual_pnl": np.nan,
            "top3_known_residual_pnl": np.nan,
            "top3_covered_qty": 0.0,
            "top3_coverage_pct": 0.0,
            "top3_full_cover": False,
        }

    bids = [(float(p), float(q)) for p, q in state.get("bid_levels", []) if q > EPS]
    asks = [(float(p), float(q)) for p, q in state.get("ask_levels", []) if q > EPS]
    if not bids or not asks:
        return {
            "residual_qty": residual_qty,
            "bbo_residual_pnl": np.nan,
            "top3_known_residual_pnl": np.nan,
            "top3_covered_qty": 0.0,
            "top3_coverage_pct": 0.0,
            "top3_full_cover": False,
        }

    bbo_pnl = 0.0
    for q, entry_px in long_lots:
        bbo_pnl += (bids[0][0] - entry_px) * q
    for q, entry_px in short_lots:
        bbo_pnl += (entry_px - asks[0][0]) * q

    known = 0.0
    covered = 0.0

    bid_depth = [[p, q] for p, q in bids]
    for lot_q, entry_px in long_lots:
        rem = lot_q
        for level in bid_depth:
            if rem <= EPS:
                break
            take = min(rem, level[1])
            if take > EPS:
                known += (level[0] - entry_px) * take
                covered += take
                rem -= take
                level[1] -= take

    ask_depth = [[p, q] for p, q in asks]
    for lot_q, entry_px in short_lots:
        rem = lot_q
        for level in ask_depth:
            if rem <= EPS:
                break
            take = min(rem, level[1])
            if take > EPS:
                known += (entry_px - level[0]) * take
                covered += take
                rem -= take
                level[1] -= take

    pct = 100.0 * covered / residual_qty if residual_qty > EPS else 100.0
    return {
        "residual_qty": float(residual_qty),
        "bbo_residual_pnl": float(bbo_pnl),
        "top3_known_residual_pnl": float(known),
        "top3_covered_qty": float(covered),
        "top3_coverage_pct": float(pct),
        "top3_full_cover": bool(covered >= residual_qty - 1e-9),
    }


def _contract_result(sim: CycleSim):
    matched, long_lots, short_lots = _fifo_open_lots(sim.fills)
    final_state = sim.m5_state or sim.last_research_state
    liq = _liquidate_residual(long_lots, short_lots, final_state)
    bbo_net = matched + liq["bbo_residual_pnl"] if np.isfinite(liq["bbo_residual_pnl"]) else np.nan

    expected_bbo_cash = np.nan
    if final_state is not None and np.isfinite(sim.cash):
        inv = float(sim.inventory)
        if abs(inv) <= 1e-9:
            expected_bbo_cash = sim.cash
        elif inv > 0:
            expected_bbo_cash = sim.cash + inv * float(final_state["bid"])
        else:
            expected_bbo_cash = sim.cash + inv * float(final_state["ask"])

    return {
        "variant": sim.variant,
        "quote_size": QUOTE_SIZE,
        "ticker": sim.ticker,
        "series": sim.meta["series"],
        "close_ts": sim.meta["close_ts"],
        "close_time": _iso(sim.meta["close_ts"]),
        "fill_events": len(sim.fills),
        "fill_qty": sum(float(f["qty"]) for f in sim.fills),
        "entry_fill_events": sum(1 for f in sim.fills if f["role"] == "ENTRY"),
        "entry_fill_qty": sum(float(f["qty"]) for f in sim.fills if f["role"] == "ENTRY"),
        "exit_fill_events": sum(1 for f in sim.fills if f["role"] == "EXIT"),
        "exit_fill_qty": sum(float(f["qty"]) for f in sim.fills if f["role"] == "EXIT"),
        "cycles_started": int(sim.counts["CYCLES_STARTED"]),
        "cycles_completed": int(sim.counts["CYCLES_COMPLETED"]),
        "ending_inventory_yes_equiv": float(sim.inventory),
        "max_abs_inventory": float(sim.max_abs_inventory),
        "cash_before_m5_liquidation": float(sim.cash),
        "matched_roundtrip_pnl": float(matched),
        "residual_qty_m5": liq["residual_qty"],
        "bbo_residual_liquidation_pnl": liq["bbo_residual_pnl"],
        "top3_known_residual_liquidation_pnl": liq["top3_known_residual_pnl"],
        "top3_residual_qty_covered": liq["top3_covered_qty"],
        "top3_residual_qty_coverage_pct": liq["top3_coverage_pct"],
        "top3_full_cover": liq["top3_full_cover"],
        "bbo_liquidated_net_pnl": bbo_net,
        "bbo_cash_crosscheck": expected_bbo_cash,
        "bbo_reconstruction_error": (
            bbo_net - expected_bbo_cash
            if np.isfinite(bbo_net) and np.isfinite(expected_bbo_cash) else np.nan
        ),
        "m5_bid": float(final_state["bid"]) if final_state is not None else np.nan,
        "m5_ask": float(final_state["ask"]) if final_state is not None else np.nan,
        "m5_state_age_s": (
            sim.meta["m5_ts"] - sim.m5_state_t
            if np.isfinite(sim.m5_state_t) else
            sim.meta["m5_ts"] - sim.last_research_t
            if np.isfinite(sim.last_research_t) else np.nan
        ),
    }


def _window_table(cdf):
    rows = []
    for close, z in cdf.groupby("close_ts", sort=True):
        rows.append({
            "close_ts": float(close),
            "close_time": _iso(close),
            "contracts": len(z),
            "bbo_liquidated_net_pnl": pd.to_numeric(z.bbo_liquidated_net_pnl, errors="coerce").sum(),
            "matched_roundtrip_pnl": pd.to_numeric(z.matched_roundtrip_pnl, errors="coerce").sum(),
            "bbo_residual_liquidation_pnl": pd.to_numeric(z.bbo_residual_liquidation_pnl, errors="coerce").sum(),
            "cycles_started": pd.to_numeric(z.cycles_started, errors="coerce").sum(),
            "cycles_completed": pd.to_numeric(z.cycles_completed, errors="coerce").sum(),
        })
    w = pd.DataFrame(rows).sort_values("close_ts").reset_index(drop=True)
    if len(w):
        w["cum_pnl"] = w.bbo_liquidated_net_pnl.cumsum()
        w["running_peak"] = np.maximum(0.0, w.cum_pnl.cummax())
        w["drawdown"] = w.cum_pnl - w.running_peak
    return w


def _aggregate(scope: str, cdf, fdf, cycledf, exposure_hours: float):
    w = _window_table(cdf)
    scale = 24.0 / exposure_hours
    net = float(pd.to_numeric(cdf.bbo_liquidated_net_pnl, errors="coerce").sum())
    matched = float(pd.to_numeric(cdf.matched_roundtrip_pnl, errors="coerce").sum())
    residual = float(pd.to_numeric(cdf.bbo_residual_liquidation_pnl, errors="coerce").sum())
    qty = float(pd.to_numeric(fdf.qty, errors="coerce").sum()) if len(fdf) else 0.0
    starts = int(pd.to_numeric(cdf.cycles_started, errors="coerce").sum())
    completes = int(pd.to_numeric(cdf.cycles_completed, errors="coerce").sum())
    residual_qty = float(pd.to_numeric(cdf.residual_qty_m5, errors="coerce").sum())
    covered_qty = float(pd.to_numeric(cdf.top3_residual_qty_covered, errors="coerce").sum())
    residual_contracts = cdf[pd.to_numeric(cdf.residual_qty_m5, errors="coerce") > EPS]
    full_cover_pct = (
        100.0 * residual_contracts.top3_full_cover.astype(bool).mean()
        if len(residual_contracts) else 100.0
    )
    coverage_pct = 100.0 * covered_qty / residual_qty if residual_qty > EPS else 100.0
    matched_share = matched / net if net > EPS else np.nan

    completed_cycles = cycledf[cycledf.completed.astype(bool)] if len(cycledf) else cycledf
    durations = pd.to_numeric(completed_cycles.duration_s, errors="coerce").dropna() if len(completed_cycles) else pd.Series(dtype=float)

    row = {
        "scope": scope,
        "contracts": len(cdf),
        "windows": len(w),
        "exposure_hours": exposure_hours,
        "fill_events": len(fdf),
        "fill_qty": qty,
        "fill_qty_per_day": qty * scale,
        "bbo_liquidated_net_pnl": net,
        "bbo_liquidated_net_pnl_per_day": net * scale,
        "matched_roundtrip_pnl": matched,
        "matched_roundtrip_pnl_per_day": matched * scale,
        "bbo_residual_liquidation_pnl": residual,
        "bbo_residual_liquidation_pnl_per_day": residual * scale,
        "matched_share_of_positive_net": matched_share,
        "net_cents_per_filled_contract": 100.0 * net / qty if qty > EPS else np.nan,
        "cycles_started": starts,
        "cycles_completed": completes,
        "cycle_completion_pct": 100.0 * completes / starts if starts else np.nan,
        "completed_cycle_duration_p50_s": durations.quantile(0.50) if len(durations) else np.nan,
        "completed_cycle_duration_p90_s": durations.quantile(0.90) if len(durations) else np.nan,
        "completed_cycle_duration_p95_s": durations.quantile(0.95) if len(durations) else np.nan,
        "residual_contracts": len(residual_contracts),
        "residual_qty_m5": residual_qty,
        "top3_residual_qty_coverage_pct": coverage_pct,
        "top3_full_cover_contract_pct": full_cover_pct,
        "median_abs_ending_inventory_contract": pd.to_numeric(cdf.ending_inventory_yes_equiv, errors="coerce").abs().median(),
        "p95_max_abs_inventory_contract": pd.to_numeric(cdf.max_abs_inventory, errors="coerce").quantile(0.95),
        "positive_window_pct": 100.0 * (w.bbo_liquidated_net_pnl > 0).mean() if len(w) else np.nan,
        "median_window_pnl": w.bbo_liquidated_net_pnl.median() if len(w) else np.nan,
        "worst_window": w.bbo_liquidated_net_pnl.min() if len(w) else np.nan,
        "max_drawdown": w.drawdown.min() if len(w) else np.nan,
    }
    for h in MARKOUTS_S:
        tag = f"{int(h)}s"
        row[f"qw_markout_{tag}_c"] = _wavg(fdf, f"markout_{tag}_c") if len(fdf) else np.nan
    return row, w


def _chronology(cdf, exposure_hours):
    w = _window_table(cdf)
    if w.empty:
        return pd.DataFrame()
    cut = len(w) // 2
    rows = []
    for label, z in (("EARLY_HALF", w.iloc[:cut]), ("LATE_HALF", w.iloc[cut:])):
        if z.empty:
            continue
        hours = 0.25 * len(z)
        net = float(z.bbo_liquidated_net_pnl.sum())
        rows.append({
            "half": label,
            "windows": len(z),
            "hours": hours,
            "bbo_liquidated_net_pnl": net,
            "bbo_liquidated_pnl_per_day": net * 24.0 / hours if hours > 0 else np.nan,
            "matched_pnl": float(z.matched_roundtrip_pnl.sum()),
            "residual_pnl": float(z.bbo_residual_liquidation_pnl.sum()),
            "positive_window_pct": 100.0 * (z.bbo_liquidated_net_pnl > 0).mean(),
            "median_window_pnl": z.bbo_liquidated_net_pnl.median(),
            "worst_window": z.bbo_liquidated_net_pnl.min(),
        })
    return pd.DataFrame(rows)


def run_c_inventory_cycle_development(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"This development replay is hard-bound to {EXPECTED_SESSION_NAME}; got {session.name}."
        )
    if not session.exists():
        raise FileNotFoundError(session)

    manifest, plan, audit_summary, quality, eligible = _load_inputs(session)
    meta = _load_meta(session, eligible)
    close_windows = sorted({meta[t]["close_ts"] for t in eligible})
    exposure_hours = 0.25 * len(close_windows)

    print(
        f"Eligible complete contracts: {len(eligible)} | series={quality.series.nunique()} | "
        f"windows={len(close_windows)} | exposure={exposure_hours:.2f}h"
    )
    print("ENTRY FIXED: Candidate C = L3 support + natural spread >=2c | SIZE FIXED: Q10")
    print("Loading M0-M5 aggressive trades...")
    trades = _load_research_trades(session, eligible, meta)

    sims = {
        (ticker, variant): CycleSim(variant, ticker, meta[ticker])
        for ticker in eligible for variant in VARIANTS
    }
    sim_groups = {
        ticker: [sims[(ticker, v)] for v in VARIANTS]
        for ticker in eligible
    }
    trade_i = defaultdict(int)
    pending = {ticker: [] for ticker in eligible}
    pending_counter = 0
    current_mid = defaultdict(lambda: np.nan)

    print("Streaming V5 book once: inventory-cycle causal replay + markouts...")
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

            # Trades before this book event see the prior book/quotes.
            arr = trades.get(ticker, [])
            j = trade_i[ticker]
            while j < len(arr) and arr[j][0] < t - EPS:
                tr_t, tr_px, tr_qty, tr_side = arr[j]
                if tr_t < meta[ticker]["m5_ts"] - EPS:
                    for sim in sim_groups[ticker]:
                        pending_counter = sim.apply_trade(
                            tr_t,
                            tr_px,
                            tr_qty,
                            tr_side,
                            current_mid[ticker],
                            pending[ticker],
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
            _resolve_pending(pending[ticker], t, float(cur["mid"]))

            if event_type == "trade_window_end" or (
                np.isfinite(elapsed) and 299.5 <= elapsed <= 302.0
            ):
                for sim in sim_groups[ticker]:
                    sim.set_m5_state(t, cur)

            for sim in sim_groups[ticker]:
                sim.on_book(t, cur, in_research)

            if show and n % 1_000_000 == 0:
                print(f"  streamed {n:,} / book rows...")

    # Flush any remaining M0-M5 trades against the final pre-M5 quote state.
    for ticker in sorted(eligible):
        arr = trades.get(ticker, [])
        j = trade_i[ticker]
        while j < len(arr):
            tr_t, tr_px, tr_qty, tr_side = arr[j]
            if tr_t < meta[ticker]["m5_ts"] - EPS:
                for sim in sim_groups[ticker]:
                    pending_counter = sim.apply_trade(
                        tr_t,
                        tr_px,
                        tr_qty,
                        tr_side,
                        current_mid[ticker],
                        pending[ticker],
                        pending_counter,
                    )
            j += 1
        for sim in sim_groups[ticker]:
            sim.cancel_all("FILE_END")

    contract_rows = []
    fill_rows = []
    cycle_rows = []
    count_rows = []
    for ticker in sorted(eligible, key=lambda x: (meta[x]["close_ts"], x)):
        for variant in VARIANTS:
            sim = sims[(ticker, variant)]
            contract_rows.append(_contract_result(sim))
            fill_rows.extend(sim.fills)
            cycle_rows.extend(sim.cycle_rows())
            count_rows.extend(
                {
                    "ticker": ticker,
                    "series": meta[ticker]["series"],
                    "variant": variant,
                    "reason": k,
                    "count": v,
                }
                for k, v in sim.counts.items()
            )

    contracts = pd.DataFrame(contract_rows)
    fills = pd.DataFrame(fill_rows)
    cycles = pd.DataFrame(cycle_rows)
    counts = pd.DataFrame(count_rows)

    summary_rows = []
    chronology_parts = []
    series_rows = []
    role_rows = []
    windows_parts = []

    for variant in VARIANTS:
        c0 = contracts[contracts.variant == variant].copy()
        f0 = fills[fills.variant == variant].copy() if len(fills) else pd.DataFrame()
        cy0 = cycles[cycles.variant == variant].copy() if len(cycles) else pd.DataFrame()

        scopes = {
            "ALL_9": (c0, f0, cy0),
            "NON_BTC_8": (
                c0[c0.series != BTC_SERIES],
                f0[f0.series != BTC_SERIES] if len(f0) else f0,
                cy0[cy0.series != BTC_SERIES] if len(cy0) else cy0,
            ),
        }
        for scope, (cdf, fdf, cydf) in scopes.items():
            row, w = _aggregate(scope, cdf, fdf, cydf, exposure_hours)
            row["variant"] = variant
            row["quote_size"] = QUOTE_SIZE
            summary_rows.append(row)
            if len(w):
                w["variant"] = variant
                w["scope"] = scope
                windows_parts.append(w)

        ch = _chronology(c0, exposure_hours)
        if len(ch):
            ch["variant"] = variant
            chronology_parts.append(ch)

        for series, cdf in c0.groupby("series", sort=True):
            fdf = f0[f0.series == series] if len(f0) else f0
            cydf = cy0[cy0.series == series] if len(cy0) else cy0
            row, _ = _aggregate(str(series), cdf, fdf, cydf, exposure_hours)
            row.update({"variant": variant, "series": series})
            series_rows.append(row)

        if len(f0):
            for role, z in f0.groupby("role", sort=True):
                role_rows.append({
                    "variant": variant,
                    "role": role,
                    "fill_events": len(z),
                    "fill_qty": z.qty.sum(),
                    "gross_edge_c": _wavg(z, "gross_edge_at_fill_c"),
                    "markout_5s_c": _wavg(z, "markout_5s_c"),
                    "markout_15s_c": _wavg(z, "markout_15s_c"),
                    "markout_30s_c": _wavg(z, "markout_30s_c"),
                    "queue_ahead_p50": pd.to_numeric(z.queue_ahead_initial, errors="coerce").median(),
                    "trade_through_fill_pct": 100.0 * z.trade_through.astype(bool).mean(),
                })

    summary = pd.DataFrame(summary_rows)
    chronology = pd.concat(chronology_parts, ignore_index=True) if chronology_parts else pd.DataFrame()
    by_series = pd.DataFrame(series_rows)
    by_role = pd.DataFrame(role_rows)
    windows = pd.concat(windows_parts, ignore_index=True) if windows_parts else pd.DataFrame()

    # Concentration diagnostic -- not used as an optimization target.
    concentration_rows = []
    for variant in VARIANTS:
        z = by_series[by_series.variant == variant].copy()
        if z.empty:
            continue
        total = float(z.bbo_liquidated_net_pnl.sum())
        z = z.sort_values("bbo_liquidated_net_pnl", ascending=False)
        top = z.iloc[0]
        concentration_rows.append({
            "variant": variant,
            "total_net_pnl": total,
            "largest_positive_series": top["series"],
            "largest_series_net_pnl": top["bbo_liquidated_net_pnl"],
            "largest_series_share_of_total_net": (
                float(top["bbo_liquidated_net_pnl"]) / total if total > EPS else np.nan
            ),
        })
    concentration = pd.DataFrame(concentration_rows)

    # Freeze-review gates. These decide whether an exact variant is even
    # eligible to be frozen for a fresh OOS recording; they do not prove it.
    gate_rows = []
    for variant in VARIANTS:
        s = summary[(summary.variant == variant) & (summary.scope == "ALL_9")].iloc[0]
        ch = chronology[chronology.variant == variant]
        both_halves_positive = bool(len(ch) == 2 and (ch.bbo_liquidated_net_pnl > 0).all())
        gates = {
            "net_rate_ge_100_day": bool(s.bbo_liquidated_net_pnl_per_day >= ECONOMIC_TARGET_USD_PER_DAY),
            "matched_pnl_positive": bool(s.matched_roundtrip_pnl > 0),
            "matched_share_ge_50pct_of_positive_net": bool(
                np.isfinite(s.matched_share_of_positive_net)
                and s.matched_share_of_positive_net >= MIN_MATCHED_SHARE_OF_POSITIVE_NET
            ),
            "cycle_completion_ge_90pct": bool(
                np.isfinite(s.cycle_completion_pct)
                and s.cycle_completion_pct >= MIN_CYCLE_COMPLETION_PCT
            ),
            "top3_residual_qty_coverage_ge_95pct": bool(
                s.top3_residual_qty_coverage_pct >= MIN_TOP3_RESIDUAL_QTY_COVERAGE_PCT
            ),
            "both_chronological_halves_positive": both_halves_positive,
        }
        for gate, passed in gates.items():
            gate_rows.append({"variant": variant, "gate": gate, "pass": bool(passed)})
    gates = pd.DataFrame(gate_rows)
    pass_by_variant = gates.groupby("variant").pass.all().to_dict() if len(gates) else {}
    freeze_eligible = [v for v in VARIANTS if pass_by_variant.get(v, False)]

    max_recon_error = pd.to_numeric(contracts.bbo_reconstruction_error, errors="coerce").abs().max()

    if output_dir is None:
        output_dir = (
            C.PROJECT_ROOT
            / "results"
            / "kalshi_mm_event_c_inventory_cycle_dev"
            / session.name
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    contracts.to_csv(out / "contract_results.csv", index=False)
    fills.to_csv(out / "fills.csv", index=False)
    cycles.to_csv(out / "cycles.csv", index=False)
    counts.to_csv(out / "policy_counts.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    chronology.to_csv(out / "chronology.csv", index=False)
    by_series.to_csv(out / "by_series.csv", index=False)
    by_role.to_csv(out / "by_role.csv", index=False)
    windows.to_csv(out / "window_results.csv", index=False)
    concentration.to_csv(out / "concentration.csv", index=False)
    gates.to_csv(out / "freeze_review_gates.csv", index=False)

    spec = {
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "hard_bound_session_name": EXPECTED_SESSION_NAME,
        "entry_signal": "Candidate C: same-side L3 depth > opposite-side L3 depth AND natural spread >=2c",
        "entry_signal_changed": False,
        "quote_size": QUOTE_SIZE,
        "size_sweep": False,
        "variants": {
            "CYCLE_ALWAYS_EXIT": "after entry fill, only quote opposite BBO until flat; no exit-state filter",
            "CYCLE_L3_EXIT": "after entry fill, only quote opposite BBO while opposite L3 support is true",
            "CYCLE_C_EXIT": "after entry fill, only quote opposite BBO while full opposite Candidate C condition is true",
        },
        "fees": 0.0,
        "economic_target_usd_per_day": ECONOMIC_TARGET_USD_PER_DAY,
        "freeze_review_gates": {
            "net_rate_ge_100_day": ECONOMIC_TARGET_USD_PER_DAY,
            "matched_share_of_positive_net_min": MIN_MATCHED_SHARE_OF_POSITIVE_NET,
            "cycle_completion_pct_min": MIN_CYCLE_COMPLETION_PCT,
            "top3_residual_qty_coverage_pct_min": MIN_TOP3_RESIDUAL_QTY_COVERAGE_PCT,
            "both_chronological_halves_positive": True,
        },
        "asset_filter": False,
        "minute_filter": False,
        "side_filter": False,
        "new_numeric_threshold_sweep": False,
        "execution": "public BBO; back of displayed L1; exact-price trades burn queue; trade-through fills; no cancellation-ahead credit; any fill cancels residual and state re-evaluates next book event",
        "m5_liquidation": "residual marked by executable observed M5 BBO; top3 depth coverage separately audited; deeper depth never fabricated",
        "guardrail": "passing development gates only permits freezing exact variant for fresh OOS; it is not OOS evidence",
        "max_bbo_reconstruction_error": max_recon_error,
    }
    (out / "study_spec.json").write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")

    if show:
        print("\n" + "=" * 176)
        print("CANDIDATE C INVENTORY-CYCLE DEVELOPMENT — ENTRY FIXED, Q10 FIXED")
        print("=" * 176)
        print(
            f"session={session.name} | contracts={len(eligible)} | windows={len(close_windows)} | "
            f"exposure={exposure_hours:.2f}h"
        )
        print("ENTRY: L3 support + natural spread >=2c | Q10 | fees=0")
        print("ONLY CHANGE: post-fill inventory mechanics")
        print(f"BBO FIFO reconstruction max abs error: ${max_recon_error:.12f}")

        print("\nECONOMIC SUMMARY — ALL 9 + NON-BTC 8")
        cols = [
            "variant", "scope", "fill_qty", "fill_qty_per_day",
            "bbo_liquidated_net_pnl_per_day", "matched_roundtrip_pnl_per_day",
            "bbo_residual_liquidation_pnl_per_day", "matched_share_of_positive_net",
            "net_cents_per_filled_contract", "cycles_started", "cycles_completed",
            "cycle_completion_pct", "completed_cycle_duration_p50_s",
            "completed_cycle_duration_p95_s", "residual_contracts", "residual_qty_m5",
            "top3_residual_qty_coverage_pct", "top3_full_cover_contract_pct",
            "qw_markout_5s_c", "qw_markout_15s_c", "qw_markout_30s_c",
            "worst_window", "max_drawdown",
        ]
        print(summary[cols].round(4).to_string(index=False))

        print("\nCHRONOLOGY — ALL 9")
        print(chronology.round(4).to_string(index=False) if len(chronology) else "none")

        print("\nBY SERIES — ALL 9")
        scols = [
            "variant", "series", "fill_qty", "bbo_liquidated_net_pnl_per_day",
            "matched_roundtrip_pnl_per_day", "bbo_residual_liquidation_pnl_per_day",
            "cycle_completion_pct", "residual_qty_m5",
        ]
        print(by_series[scols].round(4).to_string(index=False) if len(by_series) else "none")

        print("\nENTRY VS EXIT FILL ECONOMICS")
        print(by_role.round(4).to_string(index=False) if len(by_role) else "none")

        print("\nCONCENTRATION DIAGNOSTIC")
        print(concentration.round(4).to_string(index=False) if len(concentration) else "none")

        print("\nFREEZE-REVIEW GATES")
        for variant in VARIANTS:
            print(f"\n{variant}")
            z = gates[gates.variant == variant]
            for _, r in z.iterrows():
                print(f"  {'PASS' if bool(r['pass']) else 'FAIL':4s}  {r['gate']}")

        if freeze_eligible:
            print("\nDEVELOPMENT STATUS: FREEZE_ELIGIBLE_VARIANTS_PRESENT")
            print("Freeze-eligible exact variants: " + ", ".join(freeze_eligible))
            print("DO NOT tune them further. Choose/freeze one exact rule before collecting fresh OOS.")
        else:
            print("\nDEVELOPMENT STATUS: NO_VARIANT_MEETS_FREEZE_GATES")
            print("Do not collect OOS for this family merely to rescue development economics.")

        print("\nGUARDRAILS")
        print("  - Same burned V5 development data; no OOS claim is allowed.")
        print("  - Candidate C entry and 2c threshold were not changed.")
        print("  - Q10 was fixed before this replay; no Q1/Q2/Q5/Q10 sweep here.")
        print("  - No asset, minute, side, or new spread/imbalance filter is allowed after this output.")
        print("  - Passing only authorizes an exact freeze followed by a brand-new OOS recording.")
        print(f"\nOUTPUTS: {out}")
        print("=" * 176)

    return {
        "output_dir": out,
        "summary": summary,
        "chronology": chronology,
        "by_series": by_series,
        "by_role": by_role,
        "concentration": concentration,
        "freeze_review_gates": gates,
        "freeze_eligible_variants": freeze_eligible,
        "contracts": contracts,
        "fills": fills,
        "cycles": cycles,
        "counts": counts,
        "spec": spec,
    }
