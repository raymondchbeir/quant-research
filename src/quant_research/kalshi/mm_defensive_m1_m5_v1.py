from __future__ import annotations

"""Fixed defensive M1-M5 market-making strategy on validated reconstructed books.

This is an exploratory strategy test, not OOS validation.

The rule set is frozen before conditional results are inspected:
- all reconstructed quality-gate contracts remain in the universe;
- quote Q1 at the current reconstructed BBO;
- require spread >= 2.0c;
- BID is suppressed when 3s midpoint momentum <= -1.0c;
- ASK is suppressed when 3s midpoint momentum >= +1.0c;
- BID is suppressed when prior-5s aggressive-flow imbalance <= -0.60;
- ASK is suppressed when prior-5s aggressive-flow imbalance >= +0.60;
- soft inventory skew starts at |inventory| >= 2 contracts;
- hard projected inventory cap is 3 contracts;
- after any fill, cancel residual same-side quantity and impose a 3s cooldown;
- flow/inventory toxicity can cancel an active quote immediately on trade events;
- BBO repricing still occurs on the validated 1 Hz reconstructed book;
- FIFO assumptions match the unconditional Q1 baseline.

No asset/minute/volatility filters are selected.
Fees are excluded; break-even fee per filled quantity is reported.
"""

import argparse
import json
import math
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_reconstructed_m1_m5_backtest as B

STUDY_VERSION = "M1_M5_DEFENSIVE_MM_V1_FIXED"
EPS = 1e-9
DEFAULT_MARKOUT_SECONDS = (5, 15, 30, 60)

DEFAULT_POLICY = {
    "min_spread_c": 2.0,
    "momentum_lookback_s": 3.0,
    "max_adverse_momentum_c": 1.0,
    "flow_lookback_s": 5.0,
    "max_adverse_flow_imbalance": 0.60,
    "inventory_soft_limit": 2.0,
    "inventory_hard_limit": 3.0,
    "cooldown_s": 3.0,
}


def _flow_imbalance(flow_hist):
    if not flow_hist:
        return 0.0
    signed = sum(x[1] for x in flow_hist)
    total = sum(abs(x[1]) for x in flow_hist)
    return signed / total if total > EPS else 0.0


def _momentum_c(mid_hist):
    if len(mid_hist) < 2:
        return 0.0
    return 100.0 * (mid_hist[-1][1] - mid_hist[0][1])


def _purge_mid_history(dq, now_t, lookback_s):
    cutoff = now_t - lookback_s
    # Keep the latest observation at-or-before the cutoff as the momentum anchor.
    while len(dq) > 1 and dq[1][0] <= cutoff + EPS:
        dq.popleft()


def _purge_flow_history(dq, now_t, lookback_s):
    cutoff = now_t - lookback_s
    # Flow must contain ONLY events inside the lookback window.
    while dq and dq[0][0] < cutoff - EPS:
        dq.popleft()


def _policy_reasons(
    side,
    s,
    inventory,
    quote_qty,
    last_fill_ts,
    now_t,
    momentum_c,
    flow_imb,
    policy,
):
    reasons = []

    if s.spread_c < policy["min_spread_c"] - EPS:
        reasons.append("SPREAD")

    if side == "BID":
        if momentum_c <= -policy["max_adverse_momentum_c"] + EPS:
            reasons.append("MOMENTUM")
        if flow_imb <= -policy["max_adverse_flow_imbalance"] + EPS:
            reasons.append("FLOW")
        if inventory >= policy["inventory_soft_limit"] - EPS:
            reasons.append("INVENTORY_SOFT")
        if inventory + quote_qty > policy["inventory_hard_limit"] + EPS:
            reasons.append("INVENTORY_HARD")
    else:
        if momentum_c >= policy["max_adverse_momentum_c"] - EPS:
            reasons.append("MOMENTUM")
        if flow_imb >= policy["max_adverse_flow_imbalance"] - EPS:
            reasons.append("FLOW")
        if inventory <= -policy["inventory_soft_limit"] + EPS:
            reasons.append("INVENTORY_SOFT")
        if inventory - quote_qty < -policy["inventory_hard_limit"] - EPS:
            reasons.append("INVENTORY_HARD")

    lf = last_fill_ts.get(side, -np.inf)
    if now_t < lf + policy["cooldown_s"] - EPS:
        reasons.append("COOLDOWN")

    return reasons


def _simulate_contract_defensive(
    ticker,
    meta,
    samples,
    trades,
    quote_qty,
    markouts,
    max_markout_lag_s,
    policy,
):
    close = meta["close_ts"]
    wstart = close - 900.0 + 60.0
    wend = close - 900.0 + 300.0
    series = meta["series"]

    samples = [s for s in samples if wstart <= s.t < wend]
    if not samples:
        return [], [], None, Counter()
    samples.sort(key=lambda z: z.t)
    times = [s.t for s in samples]

    episodes = []
    fills = []
    active = {"BID": None, "ASK": None}
    remaining = {"BID": 0.0, "ASK": 0.0}
    last_fill_ts = {"BID": -np.inf, "ASK": -np.inf}
    episode_id = 0

    inventory = 0.0
    cash = 0.0
    max_abs_inventory = 0.0
    last_valid_mid = np.nan
    current_sample = None
    current_mid = np.nan

    mid_hist = deque()
    flow_hist = deque()
    policy_counts = Counter()

    last_sample_t = samples[0].t
    trade_times = [tr.t for tr in trades]
    tr_idx = B.bisect.bisect_left(trade_times, last_sample_t)

    def features(now_t):
        _purge_flow_history(flow_hist, now_t, policy["flow_lookback_s"])
        return _momentum_c(mid_hist), _flow_imbalance(flow_hist)

    def open_order(side, s, momentum_c, flow_imb):
        nonlocal episode_id
        episode_id += 1
        if side == "BID":
            px, qa = s.bid1, s.bid1_qty
        else:
            px, qa = s.ask1, s.ask1_qty
        ep = {
            "ticker": ticker,
            "series": series,
            "episode_id": f"{ticker}:{side}:{episode_id}",
            "side": side,
            "join_ts": s.t,
            "join_time": B._iso(s.t),
            "join_minute": s.minute,
            "price": px,
            "queue_ahead_initial": qa,
            "queue_ahead_final": qa,
            "spread_c_at_join": s.spread_c,
            "mid_at_join": s.mid,
            "momentum_3s_c_at_join": momentum_c,
            "flow_imbalance_5s_at_join": flow_imb,
            "inventory_at_join": inventory,
            "fill_qty": 0.0,
            "first_fill_ts": np.nan,
            "last_fill_ts": np.nan,
            "fill_latency_s": np.nan,
            "end_ts": np.nan,
            "end_time": None,
            "end_reason": None,
        }
        episodes.append(ep)
        active[side] = ep
        remaining[side] = float(quote_qty)
        policy_counts[f"{side}_OPEN"] += 1

    def cancel(side, t, reason):
        ep = active[side]
        if ep is not None:
            B._close_episode(ep, t, reason)
            policy_counts[f"{side}_CANCEL_{reason}"] += 1
        active[side] = None
        remaining[side] = 0.0

    def enforce_policy_on_active(now_t, s, allow_open=False):
        if s is None or not B._valid_sample(s):
            cancel("BID", now_t, "INVALID_BOOK")
            cancel("ASK", now_t, "INVALID_BOOK")
            return

        mom, flow = features(now_t)
        for side in ("BID", "ASK"):
            reasons = _policy_reasons(
                side, s, inventory, quote_qty, last_fill_ts, now_t,
                mom, flow, policy,
            )
            if reasons:
                for r in reasons:
                    policy_counts[f"{side}_BLOCK_{r}"] += 1
                if active[side] is not None:
                    cancel(side, now_t, "POLICY_" + reasons[0])
            elif allow_open and active[side] is None:
                open_order(side, s, mom, flow)

    def fill_order(side, tr, fill_mid):
        nonlocal inventory, cash, max_abs_inventory
        ep = active[side]
        if ep is None or not np.isfinite(fill_mid):
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

        qty = 0.0
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

        ep["fill_qty"] += qty
        if not np.isfinite(B._f(ep.get("first_fill_ts"))):
            ep["first_fill_ts"] = tr.t
            ep["fill_latency_s"] = tr.t - ep["join_ts"]
        ep["last_fill_ts"] = tr.t
        remaining[side] -= qty

        sign = 1.0 if side == "BID" else -1.0
        gross_edge_c = sign * (fill_mid - qpx) * 100.0
        fill = {
            "ticker": ticker,
            "series": series,
            "episode_id": ep["episode_id"],
            "side": side,
            "fill_ts": tr.t,
            "fill_time": B._iso(tr.t),
            "fill_minute": (tr.t - (close - 900.0)) / 60.0,
            "qty": qty,
            "price": qpx,
            "mid_at_fill": fill_mid,
            "gross_edge_at_fill_c": gross_edge_c,
            "queue_ahead_initial": ep["queue_ahead_initial"],
            "fill_latency_s": ep["fill_latency_s"],
            "spread_c_at_join": ep["spread_c_at_join"],
            "momentum_3s_c_at_join": ep["momentum_3s_c_at_join"],
            "flow_imbalance_5s_at_join": ep["flow_imbalance_5s_at_join"],
            "inventory_at_join": ep["inventory_at_join"],
        }
        for h in markouts:
            fs = B._future_valid_sample(samples, times, tr.t + h, max_lag_s=max_markout_lag_s)
            if fs is None:
                fill[f"future_mid_{h}s"] = np.nan
                fill[f"post_mid_move_{h}s_c"] = np.nan
                fill[f"markout_{h}s_c"] = np.nan
            else:
                fill[f"future_mid_{h}s"] = fs.mid
                fill[f"post_mid_move_{h}s_c"] = sign * (fs.mid - fill_mid) * 100.0
                fill[f"markout_{h}s_c"] = sign * (fs.mid - qpx) * 100.0
        fills.append(fill)

        if side == "BID":
            inventory += qty
            cash -= qpx * qty
        else:
            inventory -= qty
            cash += qpx * qty
        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        last_fill_ts[side] = tr.t
        policy_counts[f"{side}_FILL_EVENT"] += 1

        # Defensive rule: after ANY fill, cancel any residual same-side size
        # and do not rejoin that side for cooldown_s.
        B._close_episode(ep, tr.t, "FILLED_COOLDOWN")
        active[side] = None
        remaining[side] = 0.0
        return True

    for i, s in enumerate(samples):
        if i == 0:
            last_sample_t = s.t
        else:
            while tr_idx < len(trades) and trades[tr_idx].t <= s.t + EPS:
                tr = trades[tr_idx]
                if tr.t > last_sample_t + EPS:
                    fill_order("BID", tr, current_mid)
                    fill_order("ASK", tr, current_mid)

                    signed = 0.0
                    if tr.taker_book_side == "bid":
                        signed = float(tr.qty)   # aggressive YES buy
                    elif tr.taker_book_side == "ask":
                        signed = -float(tr.qty)  # aggressive YES sell
                    if abs(signed) > EPS:
                        flow_hist.append((tr.t, signed))
                        _purge_flow_history(flow_hist, tr.t, policy["flow_lookback_s"])

                    # Flow/inventory/cooldown may pull live quotes immediately.
                    enforce_policy_on_active(tr.t, current_sample, allow_open=False)
                tr_idx += 1
            last_sample_t = s.t

        if not B._valid_sample(s):
            current_sample = None
            current_mid = np.nan
            mid_hist.clear()
            cancel("BID", s.t, "INVALID_BOOK")
            cancel("ASK", s.t, "INVALID_BOOK")
            continue

        current_sample = s
        current_mid = s.mid
        last_valid_mid = s.mid

        mid_hist.append((s.t, s.mid))
        _purge_mid_history(mid_hist, s.t, policy["momentum_lookback_s"])

        # First enforce BBO price changes.
        for side, px in (("BID", s.bid1), ("ASK", s.ask1)):
            ep = active[side]
            if ep is not None and abs(float(ep["price"]) - px) > EPS:
                cancel(side, s.t, "BBO_REPRICE")

        # Then apply the fixed defensive quoting policy and open if eligible.
        enforce_policy_on_active(s.t, s, allow_open=True)

    end_t = min(wend, samples[-1].t)
    cancel("BID", end_t, "M5_END")
    cancel("ASK", end_t, "M5_END")

    if not np.isfinite(last_valid_mid):
        return episodes, fills, None, policy_counts

    final_mid = last_valid_mid
    net_mtm = cash + inventory * final_mid
    gross_capture = sum((f["gross_edge_at_fill_c"] / 100.0) * f["qty"] for f in fills)
    adverse_to_m5 = net_mtm - gross_capture
    matched_qty, matched_pnl = B._match_roundtrips(fills)
    bid_qty = sum(f["qty"] for f in fills if f["side"] == "BID")
    ask_qty = sum(f["qty"] for f in fills if f["side"] == "ASK")
    both = bid_qty > EPS and ask_qty > EPS
    one = (bid_qty > EPS) ^ (ask_qty > EPS)

    contract = {
        "ticker": ticker,
        "series": series,
        "close_time": B._iso(close),
        "close_ts": close,
        "reconstructed_coverage_pct": meta.get("reconstructed_coverage_pct"),
        "sample_rows": len(samples),
        "valid_sample_rows": sum(B._valid_sample(s) for s in samples),
        "bid_quote_episodes": sum(e["side"] == "BID" for e in episodes),
        "ask_quote_episodes": sum(e["side"] == "ASK" for e in episodes),
        "bid_filled_episodes": sum(e["side"] == "BID" and e.get("filled_any") for e in episodes),
        "ask_filled_episodes": sum(e["side"] == "ASK" and e.get("filled_any") for e in episodes),
        "bid_fill_qty": bid_qty,
        "ask_fill_qty": ask_qty,
        "fill_qty": bid_qty + ask_qty,
        "both_sides_filled": both,
        "one_sided_fill": one,
        "no_fill": bid_qty <= EPS and ask_qty <= EPS,
        "max_abs_inventory": max_abs_inventory,
        "ending_inventory_yes_equiv": inventory,
        "final_mid_m5": final_mid,
        "cash": cash,
        "gross_spread_capture_dollars": gross_capture,
        "adverse_selection_to_m5_dollars": adverse_to_m5,
        "net_mtm_pnl_before_fees": net_mtm,
        "matched_roundtrip_qty": matched_qty,
        "matched_roundtrip_pnl": matched_pnl,
    }
    for h in markouts:
        vals = [f[f"markout_{h}s_c"] for f in fills if np.isfinite(B._f(f.get(f"markout_{h}s_c")))]
        moves = [f[f"post_mid_move_{h}s_c"] for f in fills if np.isfinite(B._f(f.get(f"post_mid_move_{h}s_c")))]
        contract[f"mean_markout_{h}s_c"] = float(np.mean(vals)) if vals else np.nan
        contract[f"mean_post_mid_move_{h}s_c"] = float(np.mean(moves)) if moves else np.nan

    return episodes, fills, contract, policy_counts


def _chronological_splits(window_df):
    if window_df.empty:
        return pd.DataFrame()
    w = window_df.sort_values("close_ts").reset_index(drop=True).copy()
    n = len(w)
    groups = []

    def add(name, z):
        if z.empty:
            return
        pnl = z["net_mtm_pnl_before_fees"].astype(float)
        cum = pnl.cumsum()
        peak = cum.cummax()
        dd = cum - peak
        groups.append({
            "split": name,
            "windows": len(z),
            "fill_qty": z["fill_qty"].sum(),
            "net_pnl": pnl.sum(),
            "pnl_per_window": pnl.mean(),
            "median_window_pnl": pnl.median(),
            "positive_window_pct": 100.0 * (pnl > 0).mean(),
            "worst_window": pnl.min(),
            "best_window": pnl.max(),
            "max_drawdown_within_split": dd.min(),
        })

    cut = n // 2
    add("FIRST_HALF", w.iloc[:cut])
    add("SECOND_HALF", w.iloc[cut:])

    for q in range(4):
        lo = round(q * n / 4)
        hi = round((q + 1) * n / 4)
        add(f"QUARTILE_{q+1}", w.iloc[lo:hi])

    return pd.DataFrame(groups)


def _baseline_comparison(baseline_dir, headline):
    if baseline_dir is None:
        return pd.DataFrame()
    p = Path(baseline_dir) / "headline_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    base = pd.read_csv(p)
    if base.empty:
        raise RuntimeError(f"Empty baseline summary: {p}")
    b = base.iloc[0]
    s = headline.iloc[0]
    fields = [
        "fill_qty",
        "avg_gross_edge_at_fill_c",
        "avg_markout_5s_c",
        "avg_markout_15s_c",
        "avg_markout_30s_c",
        "avg_markout_60s_c",
        "gross_spread_capture_dollars",
        "adverse_selection_to_m5_dollars",
        "net_mtm_pnl_before_fees",
        "pnl_per_window",
        "avg_max_abs_inventory",
        "p95_max_abs_inventory",
        "worst_window_pnl",
        "max_drawdown",
    ]
    rows = []
    for f in fields:
        bv = B._f(b.get(f))
        sv = B._f(s.get(f))
        rows.append({
            "metric": f,
            "baseline": bv,
            "defensive_v1": sv,
            "difference": sv - bv if np.isfinite(bv) and np.isfinite(sv) else np.nan,
        })
    return pd.DataFrame(rows)


def _print_defensive_report(headline, side_df, asset_df, split_df, comparison_df, counts_df, out, markouts, policy):
    r = headline.iloc[0]
    print("\n" + "=" * 112)
    print("M1-M5 DEFENSIVE MM V1 — FIXED PRE-SPECIFIED RULES / VALIDATED RECONSTRUCTED BOOKS / BEFORE FEES")
    print("=" * 112)
    print("FIXED POLICY")
    print(
        f"  Q1 | spread >= {policy['min_spread_c']:.1f}c | "
        f"3s adverse momentum < {policy['max_adverse_momentum_c']:.1f}c | "
        f"5s adverse flow imbalance < {policy['max_adverse_flow_imbalance']:.2f}"
    )
    print(
        f"  inventory soft/hard = {policy['inventory_soft_limit']:.1f}/{policy['inventory_hard_limit']:.1f} ct | "
        f"same-side cooldown = {policy['cooldown_s']:.1f}s after any fill"
    )
    print("  No asset/minute/volatility filter. Thresholds were frozen before this conditional run.")

    print("\nCORE RESULT")
    print(f"  Eligible contracts:       {int(r['eligible_contracts']):,}")
    print(f"  Independent windows:      {int(r['independent_windows']):,}")
    print(f"  Contracts with fill:      {int(r['contracts_with_fill']):,} ({r['contract_fill_pct']:.2f}%)")
    print(f"  Fill events / qty:        {int(r['fill_events']):,} / {r['fill_qty']:.2f}")
    print(f"  Gross edge at fill:       {r.get('avg_gross_edge_at_fill_c', np.nan):+.3f} c/ct")
    for h in markouts:
        print(
            f"  {h:>2}s markout:              {r.get(f'avg_markout_{h}s_c', np.nan):+.3f} c/ct | "
            f"adverse={r.get(f'adverse_markout_{h}s_pct', np.nan):.2f}%"
        )

    print("\nINVENTORY")
    print(f"  Avg max |inventory|:      {r['avg_max_abs_inventory']:.3f} ct")
    print(f"  P95 max |inventory|:      {r['p95_max_abs_inventory']:.3f} ct")
    print(f"  One-sided / stuck:        {r['one_sided_contract_pct']:.2f}%")

    print("\nECONOMICS — BEFORE FEES")
    print(f"  Gross capture:            ${r['gross_spread_capture_dollars']:+.4f}")
    print(f"  Adverse selection to M5:  ${r['adverse_selection_to_m5_dollars']:+.4f}")
    print(f"  Net M1-M5 MTM:            ${r['net_mtm_pnl_before_fees']:+.4f}")
    print(f"  Matched round-trip PnL:   ${r['matched_roundtrip_pnl']:+.4f} on {r['matched_roundtrip_qty']:.2f} ct")
    print(f"  PnL / 15m window:         ${r['pnl_per_window']:+.5f}")
    print(f"  Break-even fee / fill qty:{r['break_even_fee_c_per_filled_qty']:+.3f} c")

    print("\nTAIL / RISK")
    print(f"  Window median:            ${r['window_median_pnl']:+.4f}")
    print(f"  Worst window:             ${r['worst_window_pnl']:+.4f}")
    print(f"  Best window:              ${r['best_window_pnl']:+.4f}")
    print(f"  Max drawdown:             ${r['max_drawdown']:+.4f}")

    if not comparison_df.empty:
        print("\nVERSUS UNCONDITIONAL Q1 BASELINE")
        print(comparison_df.round(4).to_string(index=False))

    print("\nCHRONOLOGICAL ROBUSTNESS")
    if not split_df.empty:
        print(split_df.round(4).to_string(index=False))

    print("\nBY ASSET")
    if not asset_df.empty:
        cols = [
            "series", "eligible_contracts", "contracts_with_fill", "fill_qty",
            "avg_max_abs_inventory", "gross_spread_capture_dollars",
            "adverse_selection_to_m5_dollars", "net_mtm_pnl_before_fees",
            "pnl_per_eligible_contract", "matched_roundtrip_pnl",
        ]
        print(asset_df[cols].round(4).to_string(index=False))

    print("\nTOP POLICY BLOCK COUNTS")
    if not counts_df.empty:
        print(counts_df.head(20).to_string(index=False))

    print("\nInterpretation: this is an exploratory same-session test of ONE frozen defensive rule set.")
    print("Do not retune these thresholds from this output. If economically promising and broad across time/assets, freeze it for OOS.")
    print("Outputs:", out)
    print("=" * 112)


def run_defensive_m1_m5_mm_v1(
    session_dir,
    reconstruction_dir,
    baseline_dir=None,
    output_dir=None,
    *,
    quote_qty=1.0,
    markout_seconds=DEFAULT_MARKOUT_SECONDS,
    max_markout_lag_s=2.0,
    min_spread_c=2.0,
    momentum_lookback_s=3.0,
    max_adverse_momentum_c=1.0,
    flow_lookback_s=5.0,
    max_adverse_flow_imbalance=0.60,
    inventory_soft_limit=2.0,
    inventory_hard_limit=3.0,
    cooldown_s=3.0,
    show=True,
):
    session = Path(session_dir)
    recon = Path(reconstruction_dir)
    if not session.exists():
        raise FileNotFoundError(session)
    if not recon.exists():
        raise FileNotFoundError(recon)
    if quote_qty <= 0:
        raise ValueError("quote_qty must be positive")
    if inventory_soft_limit <= 0 or inventory_hard_limit < inventory_soft_limit:
        raise ValueError("Require 0 < inventory_soft_limit <= inventory_hard_limit")

    policy = {
        "min_spread_c": float(min_spread_c),
        "momentum_lookback_s": float(momentum_lookback_s),
        "max_adverse_momentum_c": float(max_adverse_momentum_c),
        "flow_lookback_s": float(flow_lookback_s),
        "max_adverse_flow_imbalance": float(max_adverse_flow_imbalance),
        "inventory_soft_limit": float(inventory_soft_limit),
        "inventory_hard_limit": float(inventory_hard_limit),
        "cooldown_s": float(cooldown_s),
    }
    markouts = tuple(sorted({int(x) for x in markout_seconds if int(x) > 0}))

    quality_df, meta = B._load_quality_contracts(recon)
    eligible = set(meta)
    print(f"Validated reconstruction contracts: {len(eligible):,}")
    print("Loading validated 1 Hz reconstructed books...")
    samples, sample_stats = B._load_reconstructed_samples(recon, eligible)
    missing = sorted(eligible - set(samples))
    if missing:
        raise RuntimeError(f"{len(missing)} eligible contracts missing reconstructed samples; first={missing[:3]}")

    print(f"Streaming trades for {len(eligible):,} eligible contracts...")
    trades, trade_stats = B._scan_trades(session, meta)

    if output_dir is None:
        root = session.resolve().parents[2] if len(session.resolve().parents) >= 3 else Path.cwd()
        output_dir = root / "results" / "kalshi_mm_m1_m5_defensive_v1" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_episodes, all_fills, contracts = [], [], []
    policy_counts = Counter()
    targets = sorted(eligible, key=lambda x: (meta[x]["close_ts"], x))
    t0 = time.time()
    print(f"Replaying fixed Defensive MM V1 on {len(targets):,} contracts...")
    for i, ticker in enumerate(targets, 1):
        eps, fills, contract, counts = _simulate_contract_defensive(
            ticker,
            meta[ticker],
            samples[ticker],
            trades.get(ticker, []),
            float(quote_qty),
            markouts,
            float(max_markout_lag_s),
            policy,
        )
        all_episodes.extend(eps)
        all_fills.extend(fills)
        if contract is not None:
            contracts.append(contract)
        policy_counts.update(counts)
        if i % 100 == 0 or i == len(targets):
            print(f"  replayed {i:,}/{len(targets):,} | fills={len(all_fills):,} | {time.time()-t0:.1f}s")

    contract_df = pd.DataFrame(contracts)
    if len(contract_df) != len(targets):
        raise RuntimeError(f"Only {len(contract_df)}/{len(targets)} eligible contracts produced summaries")

    episodes_df = pd.DataFrame(all_episodes)
    fills_df = pd.DataFrame(all_fills)
    side_df = B._side_summary(all_episodes, all_fills, markouts)
    asset_df = B._asset_summary(contract_df)
    minute_df = B._minute_summary(all_fills, markouts)
    window_df = B._window_summary(contract_df)
    headline = B._headline(contract_df, side_df, all_fills, window_df, markouts)
    split_df = _chronological_splits(window_df)
    comparison_df = _baseline_comparison(baseline_dir, headline)
    counts_df = pd.DataFrame(
        [{"reason": k, "count": v} for k, v in policy_counts.most_common()]
    )

    episodes_df.to_csv(out / "quote_episodes.csv", index=False)
    fills_df.to_csv(out / "fills.csv", index=False)
    contract_df.to_csv(out / "contract_summary.csv", index=False)
    side_df.to_csv(out / "side_summary.csv", index=False)
    asset_df.to_csv(out / "asset_summary.csv", index=False)
    minute_df.to_csv(out / "minute_fill_summary.csv", index=False)
    window_df.to_csv(out / "window_summary.csv", index=False)
    headline.to_csv(out / "headline_summary.csv", index=False)
    split_df.to_csv(out / "chronological_split_summary.csv", index=False)
    comparison_df.to_csv(out / "baseline_comparison.csv", index=False)
    counts_df.to_csv(out / "policy_block_counts.csv", index=False)
    pd.DataFrame([{**sample_stats, **trade_stats}]).to_csv(out / "scan_stats.csv", index=False)

    config = {
        "study_version": STUDY_VERSION,
        "session": str(session.resolve()),
        "reconstruction_dir": str(recon.resolve()),
        "baseline_dir": str(Path(baseline_dir).resolve()) if baseline_dir is not None else None,
        "eligible_contracts": len(targets),
        "quote_qty": quote_qty,
        "quote_window": "M1-M5",
        "policy": policy,
        "policy_status": "FROZEN BEFORE CONDITIONAL OUTPUT; exploratory same-session test, not OOS",
        "universe": "all contracts passing the existing reconstructed-book quality gate; no asset filter",
        "quote_refresh": "1 Hz reconstructed BBO; active orders may be canceled immediately after trade-flow/inventory events",
        "queue_model": "same FIFO model as unconditional Q1 baseline: join displayed L1 back, exact aggressive flow depletes queue, no cancellation credit, trade-through fills",
        "post_fill": "cancel any residual same-side quote immediately and impose same-side cooldown",
        "markout_seconds": list(markouts),
        "max_markout_lag_s": max_markout_lag_s,
        "fees": "excluded; break-even fee per filled qty reported",
        "purpose": "test one economically motivated defensive MM rule set without threshold search",
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if show:
        _print_defensive_report(
            headline, side_df, asset_df, split_df, comparison_df,
            counts_df, out, markouts, policy,
        )
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
        "output_dir": out,
        "headline": headline,
        "contracts": contract_df,
        "fills": fills_df,
        "quote_episodes": episodes_df,
        "side_summary": side_df,
        "asset_summary": asset_df,
        "minute_summary": minute_df,
        "window_summary": window_df,
        "chronological_splits": split_df,
        "baseline_comparison": comparison_df,
        "policy_counts": counts_df,
    }


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--session", required=True)
    p.add_argument("--reconstruction-dir", required=True)
    p.add_argument("--baseline-dir", default=None)
    args = p.parse_args()
    run_defensive_m1_m5_mm_v1(
        args.session,
        args.reconstruction_dir,
        baseline_dir=args.baseline_dir,
        show=True,
    )


if __name__ == "__main__":
    _main()
