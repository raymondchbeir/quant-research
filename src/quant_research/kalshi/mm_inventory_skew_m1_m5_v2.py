from __future__ import annotations

"""Inventory-skewed defensive M1-M5 market-making study.

Exploratory same-session hypothesis. This is NOT OOS validation.

Frozen structural rule set:
- same validated reconstructed-book universe as the Q1 baseline and Defensive V1;
- when flat, use Defensive V1's spread/momentum/flow protection;
- once inventory is non-zero, stop quoting the side that would increase that inventory;
- quote only the inventory-reducing side until flat;
- if the reducing-side signal is not toxic and the spread is at least 2c,
  improve that exit quote by exactly one 1c tick inside the spread;
- otherwise reduce at the current BBO (never cross during the main M1-M5 loop);
- reducing orders are capped at the absolute inventory so they cannot flip the book;
- a 3s cooldown applies when a fill creates a position or returns inventory to flat;
- during the final 10s before M5, stop market making and cross the displayed BBO
  only to flatten residual inventory, capped by displayed L1 size;
- passive FIFO mechanics and markouts match the reconstructed Q1 baseline.

Fees are excluded. Late taker unwind quantity and spread cost are reported separately.
"""

import argparse
import bisect
import json
import math
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_reconstructed_m1_m5_backtest as _bt

STUDY_VERSION = "M1_M5_INVENTORY_SKEW_MM_V2"
EPS = 1e-9
DEFAULT_MARKOUT_SECONDS = (5, 15, 30, 60)


def _f(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _flow_imbalance(flow):
    buys = sum(float(tr.qty) for tr in flow if tr.taker_book_side == "bid")
    sells = sum(float(tr.qty) for tr in flow if tr.taker_book_side == "ask")
    den = buys + sells
    return (buys - sells) / den if den > EPS else 0.0


def _momentum_c(mids, now, lookback_s):
    if not mids:
        return 0.0
    target = now - lookback_s
    ref = mids[0][1]
    for t, mid in mids:
        if t <= target + EPS:
            ref = mid
        else:
            break
    return 100.0 * (mids[-1][1] - ref)


def _load_headline(path):
    p = Path(path) / "headline_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.iloc[0] if len(df) else None


def _compare_table(base, v1, v2):
    metrics = [
        "fill_qty", "avg_gross_edge_at_fill_c", "avg_markout_5s_c", "avg_markout_15s_c",
        "avg_markout_30s_c", "avg_markout_60s_c", "gross_spread_capture_dollars",
        "adverse_selection_to_m5_dollars", "net_mtm_pnl_before_fees", "pnl_per_window",
        "avg_max_abs_inventory", "p95_max_abs_inventory", "worst_window_pnl", "max_drawdown",
    ]
    rows = []
    for m in metrics:
        rows.append({
            "metric": m,
            "baseline": _f(base.get(m)) if base is not None else np.nan,
            "defensive_v1": _f(v1.get(m)) if v1 is not None else np.nan,
            "inventory_skew_v2": _f(v2.get(m)),
        })
    out = pd.DataFrame(rows)
    out["vs_baseline"] = out["inventory_skew_v2"] - out["baseline"]
    out["vs_v1"] = out["inventory_skew_v2"] - out["defensive_v1"]
    return out


def _chronological(window_df):
    if window_df.empty:
        return pd.DataFrame()
    w = window_df.sort_values("close_ts").reset_index(drop=True)
    n = len(w)
    specs = [
        ("FIRST_HALF", 0, n // 2), ("SECOND_HALF", n // 2, n),
        ("QUARTILE_1", 0, n // 4), ("QUARTILE_2", n // 4, n // 2),
        ("QUARTILE_3", n // 2, 3 * n // 4), ("QUARTILE_4", 3 * n // 4, n),
    ]
    rows = []
    for name, a, b in specs:
        g = w.iloc[a:b].copy()
        if g.empty:
            continue
        pnl = g["net_mtm_pnl_before_fees"].astype(float)
        cum = pnl.cumsum()
        peak = cum.cummax()
        dd = cum - peak
        rows.append({
            "split": name,
            "windows": len(g),
            "fill_qty": g["fill_qty"].sum(),
            "net_pnl": pnl.sum(),
            "pnl_per_window": pnl.mean(),
            "median_window_pnl": pnl.median(),
            "positive_window_pct": 100.0 * (pnl > 0).mean(),
            "worst_window": pnl.min(),
            "best_window": pnl.max(),
            "max_drawdown_within_split": dd.min(),
        })
    return pd.DataFrame(rows)


def _simulate_contract(
    ticker, meta, samples, trades, quote_qty, markouts, max_markout_lag_s,
    min_spread_c, momentum_lookback_s, max_adverse_momentum_c,
    flow_lookback_s, max_adverse_flow_imbalance, cooldown_s,
    improve_ticks, tick_c, flatten_before_end_s, policy_counts,
):
    close = meta["close_ts"]
    wstart = close - 900.0 + 60.0
    wend = close - 900.0 + 300.0
    series = meta["series"]
    samples = [s for s in samples if wstart <= s.t < wend]
    if not samples:
        return [], [], [], None
    samples.sort(key=lambda z: z.t)
    times = [s.t for s in samples]

    episodes, fills, unwinds = [], [], []
    active = {"BID": None, "ASK": None}
    remaining = {"BID": 0.0, "ASK": 0.0}
    cooldown_until = {"BID": -np.inf, "ASK": -np.inf}
    mids = deque()
    flow = deque()
    episode_id = 0
    inventory = 0.0
    cash = 0.0
    max_abs_inventory = 0.0
    last_valid_mid = np.nan
    current_sample = None
    late_flatten = False

    def close_episode(ep, t, reason):
        if ep is None:
            return
        if not np.isfinite(_f(ep.get("end_ts"))):
            ep["end_ts"] = float(t)
            ep["end_time"] = _bt._iso(t)
            ep["end_reason"] = reason
        ep["duration_s"] = max(0.0, float(ep["end_ts"]) - float(ep["join_ts"]))
        ep["filled_any"] = bool(ep["fill_qty"] > EPS)

    def cancel(side, t, reason):
        ep = active[side]
        if ep is not None:
            close_episode(ep, t, reason)
            policy_counts[f"{side}_CANCEL_{reason}"] += 1
        active[side] = None
        remaining[side] = 0.0

    def prune(now):
        while flow and flow[0].t < now - flow_lookback_s - EPS:
            flow.popleft()
        while len(mids) > 1 and mids[1][0] <= now - momentum_lookback_s + EPS:
            mids.popleft()

    def signal_state(now):
        prune(now)
        mom = _momentum_c(mids, now, momentum_lookback_s)
        fim = _flow_imbalance(flow)
        return mom, fim

    def desired(side, s, now):
        if late_flatten or not _bt._valid_sample(s):
            return None
        mom, fim = signal_state(now)
        spread = float(s.spread_c)
        long_inv = inventory > EPS
        short_inv = inventory < -EPS

        # Inventory state dominates: never add to an existing position.
        if long_inv:
            if side == "BID":
                policy_counts["BID_BLOCK_LONG_INVENTORY"] += 1
                return None
            qty = min(float(quote_qty), abs(inventory))
            toxic = mom >= max_adverse_momentum_c - EPS or fim >= max_adverse_flow_imbalance - EPS
            px = float(s.ask1)
            mode = "EXIT_BBO"
            queue = float(s.ask1_qty)
            if (not toxic) and spread >= min_spread_c - EPS:
                candidate = s.ask1 - improve_ticks * tick_c / 100.0
                if candidate > s.bid1 + EPS:
                    px, queue, mode = float(candidate), 0.0, "EXIT_IMPROVED"
            return px, qty, queue, mode, mom, fim

        if short_inv:
            if side == "ASK":
                policy_counts["ASK_BLOCK_SHORT_INVENTORY"] += 1
                return None
            qty = min(float(quote_qty), abs(inventory))
            toxic = mom <= -max_adverse_momentum_c + EPS or fim <= -max_adverse_flow_imbalance + EPS
            px = float(s.bid1)
            mode = "EXIT_BBO"
            queue = float(s.bid1_qty)
            if (not toxic) and spread >= min_spread_c - EPS:
                candidate = s.bid1 + improve_ticks * tick_c / 100.0
                if candidate < s.ask1 - EPS:
                    px, queue, mode = float(candidate), 0.0, "EXIT_IMPROVED"
            return px, qty, queue, mode, mom, fim

        # Flat: same defensive entry logic as V1.
        if now < cooldown_until[side] - EPS:
            policy_counts[f"{side}_BLOCK_COOLDOWN"] += 1
            return None
        if spread < min_spread_c - EPS:
            policy_counts[f"{side}_BLOCK_SPREAD"] += 1
            return None
        if side == "BID":
            if mom <= -max_adverse_momentum_c + EPS:
                policy_counts["BID_BLOCK_MOMENTUM"] += 1
                return None
            if fim <= -max_adverse_flow_imbalance + EPS:
                policy_counts["BID_BLOCK_FLOW"] += 1
                return None
            return float(s.bid1), float(quote_qty), float(s.bid1_qty), "ENTRY_BBO", mom, fim
        if mom >= max_adverse_momentum_c - EPS:
            policy_counts["ASK_BLOCK_MOMENTUM"] += 1
            return None
        if fim >= max_adverse_flow_imbalance - EPS:
            policy_counts["ASK_BLOCK_FLOW"] += 1
            return None
        return float(s.ask1), float(quote_qty), float(s.ask1_qty), "ENTRY_BBO", mom, fim

    def open_order(side, s, now, spec):
        nonlocal episode_id
        px, qty, qa, mode, mom, fim = spec
        if qty <= EPS:
            return
        episode_id += 1
        ep = {
            "ticker": ticker, "series": series,
            "episode_id": f"{ticker}:{side}:{episode_id}",
            "side": side, "quote_mode": mode,
            "join_ts": float(now), "join_time": _bt._iso(now),
            "join_minute": (now - (close - 900.0)) / 60.0,
            "price": float(px), "order_qty": float(qty),
            "queue_ahead_initial": float(qa), "queue_ahead_final": float(qa),
            "spread_c_at_join": float(s.spread_c), "mid_at_join": float(s.mid),
            "momentum_c_at_join": float(mom), "flow_imbalance_at_join": float(fim),
            "inventory_at_join": float(inventory),
            "fill_qty": 0.0, "first_fill_ts": np.nan, "last_fill_ts": np.nan,
            "fill_latency_s": np.nan, "end_ts": np.nan, "end_time": None, "end_reason": None,
        }
        episodes.append(ep)
        active[side] = ep
        remaining[side] = float(qty)
        policy_counts[f"{side}_OPEN_{mode}"] += 1

    def refresh_side(side, s, now):
        spec = desired(side, s, now)
        ep = active[side]
        if spec is None:
            if ep is not None:
                cancel(side, now, "POLICY")
            return
        px, qty, qa, mode, mom, fim = spec
        if ep is None:
            open_order(side, s, now, spec)
            return
        needs = (
            abs(float(ep["price"]) - float(px)) > EPS
            or str(ep.get("quote_mode")) != mode
            or abs(float(remaining[side]) - float(qty)) > 1e-6
        )
        if needs:
            cancel(side, now, "REPRICE_OR_SKEW")
            open_order(side, s, now, spec)

    def refresh_all(now):
        if current_sample is None or not _bt._valid_sample(current_sample):
            cancel("BID", now, "INVALID_BOOK")
            cancel("ASK", now, "INVALID_BOOK")
            return
        refresh_side("BID", current_sample, now)
        refresh_side("ASK", current_sample, now)

    def record_passive_fill(side, tr, current_mid):
        nonlocal inventory, cash, max_abs_inventory
        ep = active[side]
        if ep is None or not np.isfinite(current_mid):
            return False
        qpx = float(ep["price"])
        if side == "BID":
            if tr.taker_book_side != "ask":
                return False
            exact = abs(tr.yes_price - qpx) <= EPS
            through = tr.yes_price < qpx - EPS
        else:
            if tr.taker_book_side != "bid":
                return False
            exact = abs(tr.yes_price - qpx) <= EPS
            through = tr.yes_price > qpx + EPS
        if not (exact or through):
            return False

        if through:
            qty = remaining[side]
        else:
            qa = max(0.0, float(ep["queue_ahead_final"]))
            if tr.qty <= qa + EPS:
                ep["queue_ahead_final"] = max(0.0, qa - tr.qty)
                return False
            ep["queue_ahead_final"] = 0.0
            qty = min(remaining[side], tr.qty - qa)
        if qty <= EPS:
            return False

        inv_before = inventory
        ep["fill_qty"] += qty
        if not np.isfinite(_f(ep["first_fill_ts"])):
            ep["first_fill_ts"] = tr.t
            ep["fill_latency_s"] = tr.t - ep["join_ts"]
        ep["last_fill_ts"] = tr.t
        remaining[side] -= qty

        sign = 1.0 if side == "BID" else -1.0
        fill = {
            "ticker": ticker, "series": series, "episode_id": ep["episode_id"],
            "side": side, "quote_mode": ep["quote_mode"],
            "fill_ts": tr.t, "fill_time": _bt._iso(tr.t),
            "fill_minute": (tr.t - (close - 900.0)) / 60.0,
            "qty": float(qty), "price": qpx, "mid_at_fill": current_mid,
            "gross_edge_at_fill_c": sign * (current_mid - qpx) * 100.0,
            "queue_ahead_initial": ep["queue_ahead_initial"],
            "fill_latency_s": ep["fill_latency_s"],
            "spread_c_at_join": ep["spread_c_at_join"],
            "inventory_before_fill": inv_before,
        }
        for h in markouts:
            fs = _bt._future_valid_sample(samples, times, tr.t + h, max_lag_s=max_markout_lag_s)
            if fs is None:
                fill[f"future_mid_{h}s"] = np.nan
                fill[f"post_mid_move_{h}s_c"] = np.nan
                fill[f"markout_{h}s_c"] = np.nan
            else:
                fill[f"future_mid_{h}s"] = fs.mid
                fill[f"post_mid_move_{h}s_c"] = sign * (fs.mid - current_mid) * 100.0
                fill[f"markout_{h}s_c"] = sign * (fs.mid - qpx) * 100.0

        if side == "BID":
            inventory += qty
            cash -= qpx * qty
        else:
            inventory -= qty
            cash += qpx * qty
        fill["inventory_after_fill"] = inventory
        fills.append(fill)
        policy_counts[f"{side}_FILL_{ep['quote_mode']}"] += 1
        max_abs_inventory = max(max_abs_inventory, abs(inventory))

        # Cool down when a fill creates risk or returns us to flat.
        if abs(inventory) > abs(inv_before) + EPS or abs(inventory) <= EPS:
            cooldown_until[side] = max(cooldown_until[side], tr.t + cooldown_s)

        if remaining[side] <= EPS:
            close_episode(ep, tr.t, "FILLED")
            active[side] = None
            remaining[side] = 0.0
        return True

    def taker_flatten(s, now):
        nonlocal inventory, cash, max_abs_inventory
        if abs(inventory) <= EPS or not _bt._valid_sample(s):
            return
        if inventory > 0:
            available = max(0.0, float(s.bid1_qty))
            qty = min(inventory, available)
            if qty <= EPS:
                return
            px = float(s.bid1)
            cash += px * qty
            inventory -= qty
            side = "SELL_YES"
            cost_c = (float(s.mid) - px) * 100.0
        else:
            available = max(0.0, float(s.ask1_qty))
            qty = min(-inventory, available)
            if qty <= EPS:
                return
            px = float(s.ask1)
            cash -= px * qty
            inventory += qty
            side = "BUY_YES"
            cost_c = (px - float(s.mid)) * 100.0
        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        unwinds.append({
            "ticker": ticker, "series": series, "time": _bt._iso(now), "ts": now,
            "side": side, "qty": qty, "price": px, "mid": float(s.mid),
            "half_spread_cost_c_per_ct": cost_c, "inventory_after": inventory,
        })
        policy_counts["LATE_TAKER_UNWIND"] += 1

    trade_times = [tr.t for tr in trades]
    tr_idx = bisect.bisect_left(trade_times, samples[0].t)
    last_sample_t = samples[0].t

    for i, s in enumerate(samples):
        if i > 0:
            while tr_idx < len(trades) and trades[tr_idx].t <= s.t + EPS:
                tr = trades[tr_idx]
                if tr.t > last_sample_t + EPS:
                    current_mid = current_sample.mid if current_sample is not None and _bt._valid_sample(current_sample) else np.nan
                    got_bid = record_passive_fill("BID", tr, current_mid)
                    got_ask = record_passive_fill("ASK", tr, current_mid)
                    flow.append(tr)
                    prune(tr.t)
                    if (got_bid or got_ask) and not late_flatten:
                        refresh_all(tr.t)
                    elif not late_flatten:
                        # Current trade can make an existing quote toxic for subsequent flow.
                        refresh_all(tr.t)
                tr_idx += 1
            last_sample_t = s.t

        current_sample = s
        if _bt._valid_sample(s):
            last_valid_mid = s.mid
            mids.append((s.t, s.mid))
            prune(s.t)
        else:
            cancel("BID", s.t, "INVALID_BOOK")
            cancel("ASK", s.t, "INVALID_BOOK")
            continue

        if s.t >= wend - flatten_before_end_s - EPS:
            if not late_flatten:
                late_flatten = True
                cancel("BID", s.t, "LATE_FLATTEN")
                cancel("ASK", s.t, "LATE_FLATTEN")
            taker_flatten(s, s.t)
        else:
            refresh_all(s.t)

    end_t = min(wend, samples[-1].t)
    cancel("BID", end_t, "M5_END")
    cancel("ASK", end_t, "M5_END")

    if not np.isfinite(last_valid_mid):
        return episodes, fills, unwinds, None
    final_mid = last_valid_mid
    net_mtm = cash + inventory * final_mid
    gross_capture = sum((f["gross_edge_at_fill_c"] / 100.0) * f["qty"] for f in fills)
    adverse_to_m5 = net_mtm - gross_capture
    matched_qty, matched_pnl = _bt._match_roundtrips(fills)
    bid_qty = sum(f["qty"] for f in fills if f["side"] == "BID")
    ask_qty = sum(f["qty"] for f in fills if f["side"] == "ASK")
    unwind_qty = sum(x["qty"] for x in unwinds)
    unwind_cost = sum((x["half_spread_cost_c_per_ct"] / 100.0) * x["qty"] for x in unwinds)

    contract = {
        "ticker": ticker, "series": series, "close_time": _bt._iso(close), "close_ts": close,
        "reconstructed_coverage_pct": meta.get("reconstructed_coverage_pct"),
        "sample_rows": len(samples), "valid_sample_rows": sum(_bt._valid_sample(s) for s in samples),
        "bid_quote_episodes": sum(e["side"] == "BID" for e in episodes),
        "ask_quote_episodes": sum(e["side"] == "ASK" for e in episodes),
        "bid_filled_episodes": sum(e["side"] == "BID" and e.get("filled_any") for e in episodes),
        "ask_filled_episodes": sum(e["side"] == "ASK" and e.get("filled_any") for e in episodes),
        "bid_fill_qty": bid_qty, "ask_fill_qty": ask_qty, "fill_qty": bid_qty + ask_qty,
        "both_sides_filled": bid_qty > EPS and ask_qty > EPS,
        "one_sided_fill": (bid_qty > EPS) ^ (ask_qty > EPS),
        "no_fill": bid_qty <= EPS and ask_qty <= EPS,
        "max_abs_inventory": max_abs_inventory, "ending_inventory_yes_equiv": inventory,
        "final_mid_m5": final_mid, "cash": cash,
        "gross_spread_capture_dollars": gross_capture,
        "adverse_selection_to_m5_dollars": adverse_to_m5,
        "net_mtm_pnl_before_fees": net_mtm,
        "matched_roundtrip_qty": matched_qty, "matched_roundtrip_pnl": matched_pnl,
        "late_taker_unwind_qty": unwind_qty,
        "late_taker_half_spread_cost_dollars": unwind_cost,
        "late_taker_events": len(unwinds),
    }
    for h in markouts:
        vals = [f[f"markout_{h}s_c"] for f in fills if np.isfinite(_f(f.get(f"markout_{h}s_c")))]
        moves = [f[f"post_mid_move_{h}s_c"] for f in fills if np.isfinite(_f(f.get(f"post_mid_move_{h}s_c")))]
        contract[f"mean_markout_{h}s_c"] = float(np.mean(vals)) if vals else np.nan
        contract[f"mean_post_mid_move_{h}s_c"] = float(np.mean(moves)) if moves else np.nan
    return episodes, fills, unwinds, contract


def _print_report(headline, side_df, asset_df, window_df, chrono_df, compare_df, contracts, fills, unwinds, counts, out, markouts):
    r = headline.iloc[0]
    uqty = sum(x["qty"] for x in unwinds)
    ucost = sum((x["half_spread_cost_c_per_ct"] / 100.0) * x["qty"] for x in unwinds)
    residual = contracts["ending_inventory_yes_equiv"].abs().sum()
    improved = sum(1 for f in fills if str(f.get("quote_mode")) == "EXIT_IMPROVED")
    exitfills = sum(1 for f in fills if str(f.get("quote_mode", "")).startswith("EXIT_"))

    print("\n" + "=" * 116)
    print("M1-M5 INVENTORY-SKEW MM V2 — STRUCTURAL INVENTORY CONTROL / BEFORE FEES")
    print("=" * 116)
    print("FROZEN STRUCTURE")
    print("  Flat: Defensive V1 entry rules. Non-flat: never add inventory; quote only the reducing side.")
    print("  Safe reducing side: improve exactly 1c inside spread. Toxic reducing side: stay at BBO.")
    print("  Last 10s before M5: stop MM and cross displayed L1 only to flatten residual inventory.")
    print("  No asset/minute filter and no V1 threshold retuning.")

    print("\nCORE RESULT")
    print(f"  Eligible contracts:       {int(r['eligible_contracts']):,}")
    print(f"  Independent windows:      {int(r['independent_windows']):,}")
    print(f"  Contracts with fill:      {int(r['contracts_with_fill']):,} ({r['contract_fill_pct']:.2f}%)")
    print(f"  Passive fill events / qty:{int(r['fill_events']):,} / {r['fill_qty']:.2f}")
    print(f"  Gross edge at fill:       {r.get('avg_gross_edge_at_fill_c', np.nan):+.3f} c/ct")
    for h in markouts:
        print(f"  {h:>2}s markout:              {r.get(f'avg_markout_{h}s_c', np.nan):+.3f} c/ct | adverse={r.get(f'adverse_markout_{h}s_pct', np.nan):.2f}%")

    print("\nINVENTORY / EXIT MECHANICS")
    print(f"  Avg max |inventory|:      {r['avg_max_abs_inventory']:.3f} ct")
    print(f"  P95 max |inventory|:      {r['p95_max_abs_inventory']:.3f} ct")
    print(f"  One-sided / stuck:        {r['one_sided_contract_pct']:.2f}%")
    print(f"  Exit fills:               {exitfills:,} | improved-exit fills={improved:,}")
    print(f"  Late taker unwind qty:    {uqty:.2f} ct")
    print(f"  Late unwind half-spread:  ${ucost:+.4f} before taker fees")
    print(f"  Residual |inventory| M5:  {residual:.4f} ct total")

    print("\nECONOMICS — BEFORE ALL FEES")
    print(f"  Gross passive capture:    ${r['gross_spread_capture_dollars']:+.4f}")
    print(f"  Adverse/exit selection:   ${r['adverse_selection_to_m5_dollars']:+.4f}")
    print(f"  Net M1-M5 MTM:            ${r['net_mtm_pnl_before_fees']:+.4f}")
    print(f"  Matched passive RT PnL:   ${r['matched_roundtrip_pnl']:+.4f} on {r['matched_roundtrip_qty']:.2f} ct")
    print(f"  PnL / 15m window:         ${r['pnl_per_window']:+.5f}")
    print(f"  Passive break-even fee:   {r['break_even_fee_c_per_filled_qty']:+.3f} c/fill-qty")

    print("\nTAIL / RISK")
    print(f"  Window median:            ${r['window_median_pnl']:+.4f}")
    print(f"  Worst window:             ${r['worst_window_pnl']:+.4f}")
    print(f"  Best window:              ${r['best_window_pnl']:+.4f}")
    print(f"  Max drawdown:             ${r['max_drawdown']:+.4f}")

    print("\nVERSUS BASELINE AND DEFENSIVE V1")
    print(compare_df.round(4).to_string(index=False))
    print("\nCHRONOLOGICAL ROBUSTNESS")
    print(chrono_df.round(4).to_string(index=False))
    print("\nBY ASSET")
    cols = ["series", "eligible_contracts", "contracts_with_fill", "fill_qty", "avg_max_abs_inventory",
            "gross_spread_capture_dollars", "adverse_selection_to_m5_dollars", "net_mtm_pnl_before_fees",
            "pnl_per_eligible_contract", "matched_roundtrip_pnl"]
    print(asset_df[cols].round(4).to_string(index=False))
    print("\nTOP POLICY COUNTS")
    print(pd.DataFrame([{"reason": k, "count": v} for k, v in counts.most_common(25)]).to_string(index=False))
    print("\nInterpretation: same-session exploratory test of one structural inventory-skew hypothesis.")
    print("Late taker fees are NOT included. A small positive before-fee result is not deployment-ready.")
    print("Outputs:", out)
    print("=" * 116)


def run_inventory_skew_m1_m5_v2(
    session_dir,
    reconstruction_dir,
    baseline_dir,
    defensive_v1_dir,
    output_dir=None,
    *,
    quote_qty=1.0,
    min_spread_c=2.0,
    momentum_lookback_s=3.0,
    max_adverse_momentum_c=1.0,
    flow_lookback_s=5.0,
    max_adverse_flow_imbalance=0.60,
    cooldown_s=3.0,
    improve_ticks=1,
    tick_c=1.0,
    flatten_before_end_s=10.0,
    markout_seconds=DEFAULT_MARKOUT_SECONDS,
    max_markout_lag_s=2.0,
    show=True,
):
    session = Path(session_dir)
    recon = Path(reconstruction_dir)
    if not session.exists():
        raise FileNotFoundError(session)
    if not recon.exists():
        raise FileNotFoundError(recon)
    if quote_qty <= 0 or improve_ticks < 0 or tick_c <= 0:
        raise ValueError("Invalid quote/tick parameters")
    markouts = tuple(sorted({int(x) for x in markout_seconds if int(x) > 0}))

    quality_df, meta = _bt._load_quality_contracts(recon)
    eligible = set(meta)
    print(f"Validated reconstruction contracts: {len(eligible):,}")
    print("Loading validated 1 Hz reconstructed books...")
    samples, sample_stats = _bt._load_reconstructed_samples(recon, eligible)
    print(f"Streaming trades for {len(eligible):,} eligible contracts...")
    trades, trade_stats = _bt._scan_trades(session, meta)

    if output_dir is None:
        root = session.resolve().parents[2] if len(session.resolve().parents) >= 3 else Path.cwd()
        output_dir = root / "results" / "kalshi_mm_m1_m5_inventory_skew_v2" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_episodes, all_fills, all_unwinds, contracts = [], [], [], []
    policy_counts = Counter()
    targets = sorted(eligible, key=lambda x: (meta[x]["close_ts"], x))
    t0 = time.time()
    print(f"Replaying Inventory-Skew MM V2 on {len(targets):,} contracts...")
    for i, ticker in enumerate(targets, 1):
        eps, fills, unwinds, contract = _simulate_contract(
            ticker, meta[ticker], samples[ticker], trades.get(ticker, []),
            float(quote_qty), markouts, float(max_markout_lag_s),
            float(min_spread_c), float(momentum_lookback_s), float(max_adverse_momentum_c),
            float(flow_lookback_s), float(max_adverse_flow_imbalance), float(cooldown_s),
            int(improve_ticks), float(tick_c), float(flatten_before_end_s), policy_counts,
        )
        all_episodes.extend(eps)
        all_fills.extend(fills)
        all_unwinds.extend(unwinds)
        if contract is not None:
            contracts.append(contract)
        if i % 100 == 0 or i == len(targets):
            print(f"  replayed {i:,}/{len(targets):,} | passive fills={len(all_fills):,} | unwinds={len(all_unwinds):,} | {time.time()-t0:.1f}s")

    contract_df = pd.DataFrame(contracts)
    episodes_df = pd.DataFrame(all_episodes)
    fills_df = pd.DataFrame(all_fills)
    unwind_df = pd.DataFrame(all_unwinds)
    side_df = _bt._side_summary(all_episodes, all_fills, markouts)
    asset_df = _bt._asset_summary(contract_df)
    minute_df = _bt._minute_summary(all_fills, markouts)
    window_df = _bt._window_summary(contract_df)
    headline = _bt._headline(contract_df, side_df, all_fills, window_df, markouts)
    chrono_df = _chronological(window_df)

    base = _load_headline(baseline_dir)
    v1 = _load_headline(defensive_v1_dir)
    compare_df = _compare_table(base, v1, headline.iloc[0])

    episodes_df.to_csv(out / "quote_episodes.csv", index=False)
    fills_df.to_csv(out / "fills.csv", index=False)
    unwind_df.to_csv(out / "late_taker_unwinds.csv", index=False)
    contract_df.to_csv(out / "contract_summary.csv", index=False)
    side_df.to_csv(out / "side_summary.csv", index=False)
    asset_df.to_csv(out / "asset_summary.csv", index=False)
    minute_df.to_csv(out / "minute_fill_summary.csv", index=False)
    window_df.to_csv(out / "window_summary.csv", index=False)
    headline.to_csv(out / "headline_summary.csv", index=False)
    chrono_df.to_csv(out / "chronological_robustness.csv", index=False)
    compare_df.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame([{"reason": k, "count": v} for k, v in policy_counts.most_common()]).to_csv(out / "policy_counts.csv", index=False)
    pd.DataFrame([{**sample_stats, **trade_stats}]).to_csv(out / "scan_stats.csv", index=False)

    config = {
        "study_version": STUDY_VERSION,
        "session": str(session.resolve()), "reconstruction_dir": str(recon.resolve()),
        "baseline_dir": str(Path(baseline_dir).resolve()), "defensive_v1_dir": str(Path(defensive_v1_dir).resolve()),
        "quote_qty": quote_qty, "min_spread_c": min_spread_c,
        "momentum_lookback_s": momentum_lookback_s, "max_adverse_momentum_c": max_adverse_momentum_c,
        "flow_lookback_s": flow_lookback_s, "max_adverse_flow_imbalance": max_adverse_flow_imbalance,
        "cooldown_s": cooldown_s, "improve_ticks": improve_ticks, "tick_c": tick_c,
        "flatten_before_end_s": flatten_before_end_s, "markout_seconds": list(markouts),
        "inventory_rule": "non-flat blocks risk-increasing side; reducing order capped at abs inventory",
        "exit_rule": "safe reducing quote improves 1 tick inside spread; toxic reducing quote stays at BBO",
        "late_rule": "last 10s stop passive MM and taker-flatten against displayed L1; taker fees excluded",
        "fees": "all fees excluded",
        "purpose": "same-session exploratory structural inventory-skew test; not OOS",
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if show:
        _print_report(headline, side_df, asset_df, window_df, chrono_df, compare_df,
                      contract_df, all_fills, all_unwinds, policy_counts, out, markouts)
        try:
            from IPython.display import display
            print("\nSIDE SUMMARY")
            display(side_df.round(4))
            print("\nWORST 15 WINDOWS")
            display(window_df.nsmallest(15, "net_mtm_pnl_before_fees").round(4))
            print("\nBEST 15 WINDOWS")
            display(window_df.nlargest(15, "net_mtm_pnl_before_fees").round(4))
        except Exception:
            pass

    return {
        "output_dir": out, "headline": headline, "contracts": contract_df, "fills": fills_df,
        "quote_episodes": episodes_df, "late_taker_unwinds": unwind_df, "side_summary": side_df,
        "asset_summary": asset_df, "minute_summary": minute_df, "window_summary": window_df,
        "chronological_robustness": chrono_df, "comparison": compare_df,
    }


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--session", required=True)
    p.add_argument("--reconstruction-dir", required=True)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--defensive-v1-dir", required=True)
    args = p.parse_args()
    run_inventory_skew_m1_m5_v2(
        args.session, args.reconstruction_dir, args.baseline_dir, args.defensive_v1_dir, show=True,
    )


if __name__ == "__main__":
    _main()
