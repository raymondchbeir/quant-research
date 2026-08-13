from __future__ import annotations

"""NAT4->2 inside-spread MM with inventory-targeted sizing.

Development mechanism test on the compact 1 Hz recording.

Fixed hypothesis
----------------
- M1-M5 only.
- Quote only when the public/natural YES spread is exactly 4c (+/- 0.05c).
- Improve each side by 1c: BID = public bid + 1c, ASK = public ask - 1c.
- Resulting hypothetical spread is 2c.
- Maximum displayed quantity is 100 contracts per side.
- Inventory target is 0 YES-equivalent contracts.
- Soft inventory threshold is |I| = 125.
- Absolute hard inventory cap is |I| = 200.
- Risk-reducing side remains Q100.
- Risk-increasing side is clipped to remaining hard-cap headroom:
      BID qty = min(100, 200-I) when I >= 0, else 100
      ASK qty = min(100, 200+I) when I <= 0, else 100
  Thus at +125 inventory BID/ASK = 75/100; +150 = 50/100;
  +175 = 25/100; +200 = 0/100, mirrored when short.
- For 100 < |I| < 125, the same clipping is only hard-cap protection;
  |I| >= 125 is reported as the soft-skew regime.
- Same 3s momentum, prior-5s aggressive-flow, and 3s same-side cooldown rules.
- Queue ahead is zero because our hypothetical quote improves BBO.
- ONLY recorded aggressive trades can fill. BBO changes never manufacture fills.
- Any fill cancels residual same-side quantity. If inventory changes make the
  opposite quote size stale, it is canceled immediately and may reopen only at
  the next 1 Hz book sample.
- No forced flatten at M5; remaining inventory is marked to the last valid mid.
- Fees excluded from replay.

Q100 inside the spread can alter future market behavior, so this remains a
counterfactual development diagnostic rather than executable/OOS PnL.
"""

import json
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_reconstructed_m1_m5_backtest as B
from . import mm_defensive_m1_m5_v1 as D
from . import mm_exact_quote_lifetime_m1_m5_v1 as L
from . import mm_exact_min_spread_m1_m5_v1 as S
from . import mm_oos_4c_audit_replay as O
from . import mm_oos_4c_compact_recorder_v2 as R
from . import mm_inside_spread_q100_dev_v1 as Q

STUDY_VERSION = "M1_M5_NAT4_TO_2_INVENTORY_TARGET_Q100_DEV_V1"
EPS = 1e-9
NATURAL_SPREAD_C = 4.0
OUR_SPREAD_C = 2.0
IMPROVE_C = 1.0
MAX_QUOTE_QTY = 100.0
SOFT_INVENTORY = 125.0
HARD_INVENTORY = 200.0
TARGET_INVENTORY = 0.0
MARKOUTS = (5, 15, 30, 60)


def _desired_qty(side: str, inventory: float) -> tuple[float, str]:
    """Inventory-targeted displayed size while guaranteeing |I| <= 200."""
    inv = float(inventory)
    if side == "BID":
        risk_increasing = inv >= TARGET_INVENTORY - EPS
        if not risk_increasing:
            return MAX_QUOTE_QTY, "RISK_REDUCING"
        qty = min(MAX_QUOTE_QTY, max(0.0, HARD_INVENTORY - inv))
    else:
        risk_increasing = inv <= TARGET_INVENTORY + EPS
        if not risk_increasing:
            return MAX_QUOTE_QTY, "RISK_REDUCING"
        qty = min(MAX_QUOTE_QTY, max(0.0, HARD_INVENTORY + inv))

    if qty <= EPS:
        return 0.0, "HARD_LIMIT"
    if abs(inv) >= SOFT_INVENTORY - EPS:
        return qty, "SOFT_SKEW"
    if qty < MAX_QUOTE_QTY - EPS:
        return qty, "HARD_CAP_PROTECTION"
    return qty, "SYMMETRIC"


def _base_reasons(side, s, last_fill_ts, now_t, mom, flow):
    reasons = []
    if not Q._match_spread(s, NATURAL_SPREAD_C):
        reasons.append("NATURAL_SPREAD")
    if side == "BID":
        if mom <= -D.DEFAULT_POLICY["max_adverse_momentum_c"] + EPS:
            reasons.append("MOMENTUM")
        if flow <= -D.DEFAULT_POLICY["max_adverse_flow_imbalance"] + EPS:
            reasons.append("FLOW")
    else:
        if mom >= D.DEFAULT_POLICY["max_adverse_momentum_c"] - EPS:
            reasons.append("MOMENTUM")
        if flow >= D.DEFAULT_POLICY["max_adverse_flow_imbalance"] - EPS:
            reasons.append("FLOW")
    if now_t < last_fill_ts.get(side, -np.inf) + D.DEFAULT_POLICY["cooldown_s"] - EPS:
        reasons.append("COOLDOWN")
    return reasons


def _simulate_contract(ticker, meta, samples, trades):
    close = float(meta["close_ts"])
    wstart, wend = close - 840.0, close - 600.0
    series = meta["series"]
    samples = sorted([s for s in samples if wstart <= s.t < wend], key=lambda z: z.t)
    if not samples:
        return [], [], None, Counter()
    times = [s.t for s in samples]
    trade_times = [tr.t for tr in trades]
    tr_idx = B.bisect.bisect_left(trade_times, samples[0].t)

    active = {"BID": None, "ASK": None}
    remaining = {"BID": 0.0, "ASK": 0.0}
    last_fill_ts = {"BID": -np.inf, "ASK": -np.inf}
    episodes, fills = [], []
    counts = Counter()
    mid_hist, flow_hist = deque(), deque()
    inventory = cash = max_abs_inventory = 0.0
    current_sample = None
    current_mid = np.nan
    last_valid_mid = np.nan
    episode_id = 0
    last_sample_t = samples[0].t

    def features(now_t):
        D._purge_flow_history(flow_hist, now_t, D.DEFAULT_POLICY["flow_lookback_s"])
        return D._momentum_c(mid_hist), D._flow_imbalance(flow_hist)

    def cancel(side, t, reason):
        ep = active[side]
        if ep is not None:
            B._close_episode(ep, t, reason)
            counts[f"{side}_CANCEL_{reason}"] += 1
        active[side] = None
        remaining[side] = 0.0

    def open_order(side, s, mom, flow, desired_qty, size_regime):
        nonlocal episode_id
        if desired_qty <= EPS:
            counts[f"{side}_BLOCK_HARD_LIMIT"] += 1
            return
        px = Q._quote_price(side, s)
        other = Q._quote_price("ASK" if side == "BID" else "BID", s)
        if (side == "BID" and px >= other - EPS) or (side == "ASK" and other >= px - EPS):
            counts[f"{side}_BLOCK_CROSSED_OURS"] += 1
            return
        episode_id += 1
        ep = {
            "ticker": ticker, "series": series,
            "episode_id": f"{ticker}:NAT4_TO_2_INV:{side}:{episode_id}",
            "scenario": "NAT4_TO_2_INV_TARGET",
            "side": side, "join_ts": s.t, "join_time": B._iso(s.t), "join_minute": s.minute,
            "price": px, "queue_ahead_initial": 0.0, "queue_ahead_final": 0.0,
            "natural_bid_at_join": s.bid1, "natural_ask_at_join": s.ask1,
            "natural_spread_c_at_join": s.spread_c, "our_spread_c": OUR_SPREAD_C,
            "mid_at_join": s.mid, "momentum_3s_c_at_join": mom,
            "flow_imbalance_5s_at_join": flow, "inventory_at_join": inventory,
            "quote_qty_initial": float(desired_qty), "size_regime": size_regime,
            "soft_skew_active": abs(inventory) >= SOFT_INVENTORY - EPS,
            "fill_qty": 0.0, "first_fill_ts": np.nan, "last_fill_ts": np.nan,
            "fill_latency_s": np.nan, "end_ts": np.nan, "end_time": None, "end_reason": None,
        }
        episodes.append(ep)
        active[side] = ep
        remaining[side] = float(desired_qty)
        counts[f"{side}_OPEN"] += 1
        counts[f"{side}_OPEN_{size_regime}"] += 1

    def enforce(now_t, s, allow_open):
        if s is None or not B._valid_sample(s):
            cancel("BID", now_t, "INVALID_BOOK")
            cancel("ASK", now_t, "INVALID_BOOK")
            return
        mom, flow = features(now_t)
        for side in ("BID", "ASK"):
            reasons = _base_reasons(side, s, last_fill_ts, now_t, mom, flow)
            desired_qty, size_regime = _desired_qty(side, inventory)
            if desired_qty <= EPS:
                reasons.append("HARD_LIMIT")
            if reasons:
                for reason in reasons:
                    counts[f"{side}_BLOCK_{reason}"] += 1
                if active[side] is not None:
                    cancel(side, now_t, "POLICY_" + reasons[0])
                continue

            ep = active[side]
            if ep is not None:
                old_qty = float(ep.get("quote_qty_initial", 0.0))
                if abs(old_qty - desired_qty) > EPS:
                    cancel(side, now_t, "INVENTORY_RESIZE")
                    ep = None
            if allow_open and ep is None:
                open_order(side, s, mom, flow, desired_qty, size_regime)

    def fill_order(side, tr, fill_mid):
        nonlocal inventory, cash, max_abs_inventory
        ep = active[side]
        if ep is None or not np.isfinite(fill_mid):
            return False
        qpx = float(ep["price"])
        if side == "BID":
            crossed = tr.taker_book_side == "ask" and tr.yes_price <= qpx + EPS
        else:
            crossed = tr.taker_book_side == "bid" and tr.yes_price >= qpx - EPS
        if not crossed:
            return False
        qty = min(float(remaining[side]), float(tr.qty))
        if qty <= EPS:
            return False

        # Hard-cap invariant. This should be redundant with _desired_qty, but is
        # enforced here as a replay safety check.
        max_safe = HARD_INVENTORY - inventory if side == "BID" else HARD_INVENTORY + inventory
        qty = min(qty, max(0.0, max_safe))
        if qty <= EPS:
            counts[f"{side}_FILL_BLOCK_HARD_LIMIT"] += 1
            return False

        ep["fill_qty"] += qty
        if not np.isfinite(B._f(ep.get("first_fill_ts"))):
            ep["first_fill_ts"] = tr.t
            ep["fill_latency_s"] = tr.t - ep["join_ts"]
        ep["last_fill_ts"] = tr.t
        remaining[side] -= qty

        sign = 1.0 if side == "BID" else -1.0
        inv_before = inventory
        inv_after = inventory + sign * qty
        fill = {
            "ticker": ticker, "series": series, "scenario": ep["scenario"],
            "episode_id": ep["episode_id"], "side": side,
            "fill_ts": tr.t, "fill_time": B._iso(tr.t),
            "fill_minute": (tr.t - (close - 900.0)) / 60.0,
            "qty": qty, "price": qpx, "mid_at_fill": fill_mid,
            "gross_edge_at_fill_c": sign * (fill_mid - qpx) * 100.0,
            "fill_latency_s": ep["fill_latency_s"],
            "natural_spread_c_at_join": ep["natural_spread_c_at_join"],
            "our_spread_c": OUR_SPREAD_C,
            "quote_qty_initial": ep["quote_qty_initial"], "size_regime": ep["size_regime"],
            "soft_skew_active": ep["soft_skew_active"],
            "inventory_before_fill": inv_before, "inventory_after_fill": inv_after,
            "momentum_3s_c_at_join": ep["momentum_3s_c_at_join"],
            "flow_imbalance_5s_at_join": ep["flow_imbalance_5s_at_join"],
            "historical_trade_price": tr.yes_price, "historical_trade_qty": tr.qty,
            "historical_trade_participation_pct": 100.0 * qty / tr.qty if tr.qty > EPS else np.nan,
        }
        for h in MARKOUTS:
            fs = B._future_valid_sample(samples, times, tr.t + h, max_lag_s=2.0)
            if fs is None:
                fill[f"future_mid_{h}s"] = np.nan
                fill[f"markout_{h}s_c"] = np.nan
                fill[f"post_mid_move_{h}s_c"] = np.nan
            else:
                fill[f"future_mid_{h}s"] = fs.mid
                fill[f"markout_{h}s_c"] = sign * (fs.mid - qpx) * 100.0
                fill[f"post_mid_move_{h}s_c"] = sign * (fs.mid - fill_mid) * 100.0
        fills.append(fill)

        inventory = inv_after
        if side == "BID":
            cash -= qpx * qty
        else:
            cash += qpx * qty
        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        if abs(inventory) > HARD_INVENTORY + 1e-7:
            raise RuntimeError(f"hard inventory cap violated: {ticker} inventory={inventory}")
        last_fill_ts[side] = tr.t
        counts[f"{side}_FILL_EVENT"] += 1

        # Any fill cancels residual same-side order and starts same-side cooldown.
        B._close_episode(ep, tr.t, "FILLED_COOLDOWN")
        active[side] = None
        remaining[side] = 0.0

        # Inventory changed. If the opposite live quote now has stale size,
        # cancel it immediately; conservative replay waits until next 1 Hz sample
        # before reopening at the new size.
        other = "ASK" if side == "BID" else "BID"
        other_ep = active[other]
        if other_ep is not None:
            new_qty, _ = _desired_qty(other, inventory)
            if abs(float(other_ep.get("quote_qty_initial", 0.0)) - new_qty) > EPS:
                cancel(other, tr.t, "INVENTORY_RESIZE_AFTER_FILL")
        return True

    for i, s in enumerate(samples):
        if i > 0:
            while tr_idx < len(trades) and trades[tr_idx].t <= s.t + EPS:
                tr = trades[tr_idx]
                if tr.t > last_sample_t + EPS:
                    fill_order("BID", tr, current_mid)
                    fill_order("ASK", tr, current_mid)
                    signed = tr.qty if tr.taker_book_side == "bid" else -tr.qty if tr.taker_book_side == "ask" else 0.0
                    if abs(signed) > EPS:
                        flow_hist.append((tr.t, float(signed)))
                        D._purge_flow_history(flow_hist, tr.t, D.DEFAULT_POLICY["flow_lookback_s"])
                    enforce(tr.t, current_sample, False)
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
        D._purge_mid_history(mid_hist, s.t, D.DEFAULT_POLICY["momentum_lookback_s"])

        # Public BBO remains exogenous. Reprice/cancel only; never infer fills.
        for side in ("BID", "ASK"):
            ep = active[side]
            if ep is not None:
                desired_px = Q._quote_price(side, s) if Q._match_spread(s, NATURAL_SPREAD_C) else np.nan
                if not np.isfinite(desired_px) or abs(float(ep["price"]) - desired_px) > EPS:
                    cancel(side, s.t, "NATURAL_BBO_REPRICE")
        enforce(s.t, s, True)

    end_t = min(wend, samples[-1].t)
    cancel("BID", end_t, "M5_END")
    cancel("ASK", end_t, "M5_END")
    if not np.isfinite(last_valid_mid):
        return episodes, fills, None, counts

    final_mid = last_valid_mid
    net = cash + inventory * final_mid
    gross = sum((f["gross_edge_at_fill_c"] / 100.0) * f["qty"] for f in fills)
    matched_qty, matched_pnl = B._match_roundtrips(fills)
    bid_qty = sum(f["qty"] for f in fills if f["side"] == "BID")
    ask_qty = sum(f["qty"] for f in fills if f["side"] == "ASK")
    contract = {
        "ticker": ticker, "series": series, "close_time": B._iso(close), "close_ts": close,
        "scenario": "NAT4_TO_2_INV_TARGET",
        "natural_spread_c": NATURAL_SPREAD_C, "our_spread_c": OUR_SPREAD_C,
        "sample_rows": len(samples), "valid_sample_rows": sum(B._valid_sample(x) for x in samples),
        "bid_fill_qty": bid_qty, "ask_fill_qty": ask_qty, "fill_qty": bid_qty + ask_qty,
        "both_sides_filled": bid_qty > EPS and ask_qty > EPS,
        "one_sided_fill": (bid_qty > EPS) ^ (ask_qty > EPS),
        "max_abs_inventory": max_abs_inventory, "ending_inventory_yes_equiv": inventory,
        "final_mid_m5": final_mid, "cash": cash,
        "gross_spread_capture_dollars": gross,
        "adverse_selection_to_m5_dollars": net - gross,
        "net_mtm_pnl_before_fees": net,
        "matched_roundtrip_qty": matched_qty, "matched_roundtrip_pnl": matched_pnl,
    }
    return episodes, fills, contract, counts


def _headline(contract_df, fills_df, windows, episodes_df):
    qty = fills_df.qty.sum() if len(fills_df) else 0.0
    pnl = contract_df.net_mtm_pnl_before_fees.sum() if len(contract_df) else 0.0
    row = {
        "scenario": "NAT4_TO_2_INV_TARGET",
        "eligible_contracts": len(contract_df), "independent_windows": len(windows),
        "contracts_with_fill": int((contract_df.fill_qty > EPS).sum()),
        "fill_events": len(fills_df), "fill_qty": qty,
        "net_mtm_pnl_before_fees": pnl, "pnl_per_window": pnl / len(windows) if len(windows) else np.nan,
        "gross_capture_dollars": contract_df.gross_spread_capture_dollars.sum(),
        "adverse_selection_to_m5_dollars": contract_df.adverse_selection_to_m5_dollars.sum(),
        "matched_roundtrip_pnl": contract_df.matched_roundtrip_pnl.sum(),
        "qty_weighted_gross_edge_c": Q._wavg(fills_df, "gross_edge_at_fill_c"),
        "avg_fill_qty": fills_df.qty.mean() if len(fills_df) else np.nan,
        "p95_fill_qty": fills_df.qty.quantile(.95) if len(fills_df) else np.nan,
        "avg_max_abs_inventory": contract_df.max_abs_inventory.mean(),
        "p95_max_abs_inventory": contract_df.max_abs_inventory.quantile(.95),
        "max_abs_inventory_observed": contract_df.max_abs_inventory.max(),
        "avg_abs_ending_inventory": contract_df.ending_inventory_yes_equiv.abs().mean(),
        "p95_abs_ending_inventory": contract_df.ending_inventory_yes_equiv.abs().quantile(.95),
        "worst_window_pnl": windows.net_mtm_pnl_before_fees.min(),
        "best_window_pnl": windows.net_mtm_pnl_before_fees.max(),
        "max_drawdown": windows.drawdown.min(),
        "break_even_fee_c_per_fill_qty": 100.0 * pnl / qty if qty > EPS else np.nan,
        "avg_historical_trade_participation_pct": fills_df.historical_trade_participation_pct.mean() if len(fills_df) else np.nan,
        "soft_skew_fill_qty_pct": 100.0 * fills_df.loc[fills_df.soft_skew_active.astype(bool), "qty"].sum() / qty if qty > EPS and len(fills_df) else 0.0,
        "soft_skew_open_pct": 100.0 * episodes_df.soft_skew_active.astype(bool).mean() if len(episodes_df) else 0.0,
        "avg_displayed_quote_qty": episodes_df.quote_qty_initial.mean() if len(episodes_df) else np.nan,
    }
    for h in MARKOUTS:
        row[f"qty_weighted_markout_{h}s_c"] = Q._wavg(fills_df, f"markout_{h}s_c")
    return pd.DataFrame([row])


def _inventory_bucket(fills_df):
    if fills_df.empty:
        return pd.DataFrame()
    z = fills_df.copy()
    a = z.inventory_before_fill.abs()
    z["inventory_bucket"] = pd.cut(a, [-EPS, 50, 100, 125, 150, 175, 200 + EPS], labels=["0-50", "50-100", "100-125", "125-150", "150-175", "175-200"], include_lowest=True)
    rows = []
    for bucket, g in z.groupby("inventory_bucket", observed=True):
        r = {"inventory_bucket": str(bucket), "fill_events": len(g), "fill_qty": g.qty.sum(),
             "avg_fill_qty": g.qty.mean(), "qty_weighted_gross_edge_c": Q._wavg(g, "gross_edge_at_fill_c")}
        for h in MARKOUTS:
            r[f"qty_weighted_markout_{h}s_c"] = Q._wavg(g, f"markout_{h}s_c")
        rows.append(r)
    return pd.DataFrame(rows)


def _comparison(baseline_dir, headline):
    if baseline_dir is None:
        return pd.DataFrame()
    p = Path(baseline_dir) / "headline_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    bdf = pd.read_csv(p)
    b = bdf[bdf.scenario == "NAT4_TO_2"].iloc[0]
    h = headline.iloc[0]
    fields = [
        "fill_qty", "net_mtm_pnl_before_fees", "pnl_per_window",
        "qty_weighted_gross_edge_c", "qty_weighted_markout_5s_c",
        "qty_weighted_markout_15s_c", "qty_weighted_markout_30s_c",
        "qty_weighted_markout_60s_c", "p95_max_abs_inventory",
        "worst_window_pnl", "max_drawdown", "break_even_fee_c_per_fill_qty",
    ]
    return pd.DataFrame([{
        "metric": f, "fixed_q100_nat4_to_2": B._f(b.get(f)),
        "inventory_targeted": B._f(h.get(f)),
        "difference": B._f(h.get(f)) - B._f(b.get(f)),
    } for f in fields])


def run_nat4_to_2_inventory_target_development(session_dir, baseline_q100_dir=None, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    _, _, duration_h = O._verify(session)
    meta = O._metadata(session)
    samples, info, duplicates, _ = O._bbo(session, meta)
    trades, _ = O._trades(session, meta)
    audit = O._audit(meta, samples, info, duplicates, trades)
    good = audit[audit.quality_ok].copy()
    if good.empty:
        raise RuntimeError("No contracts pass the existing 80% quality gate")
    sim_meta = {str(r.ticker): {"ticker": str(r.ticker), "series": str(r.series), "close_ts": float(r.close_ts)} for r in good.itertuples(index=False)}
    targets = sorted(sim_meta, key=lambda t: (sim_meta[t]["close_ts"], t))

    print(f"NAT4->2 inventory-target development | {len(targets)} quality contracts | {good.close_ts.nunique()} windows")
    print(f"Qmax={MAX_QUOTE_QTY:g}/side | target=0 | soft=+/-{SOFT_INVENTORY:g} | hard=+/-{HARD_INVENTORY:g}")
    contracts, fills, episodes, counts = [], [], [], Counter()
    for i, ticker in enumerate(targets, 1):
        e, f, c, k = _simulate_contract(ticker, sim_meta[ticker], samples.get(ticker, []), trades.get(ticker, []))
        episodes.extend(e); fills.extend(f); counts.update(k)
        if c is not None:
            contracts.append(c)
        if i % 100 == 0 or i == len(targets):
            print(f"  replay {i}/{len(targets)} | fills={len(fills)} | qty={sum(x['qty'] for x in fills):.1f}")

    cdf, fdf, edf = pd.DataFrame(contracts), pd.DataFrame(fills), pd.DataFrame(episodes)
    wdf = L._window_summary(cdf)
    chrono = S._chronological_detail(wdf, "NAT4_TO_2_INV_TARGET")
    side = Q._side("NAT4_TO_2_INV_TARGET", fdf)
    inventory = _inventory_bucket(fdf)
    headline = _headline(cdf, fdf, wdf, edf)
    compare = _comparison(baseline_q100_dir, headline)
    count_df = pd.DataFrame([{"reason": k, "count": v} for k, v in counts.most_common()])

    out = Path(output_dir) if output_dir else R.PROJECT_ROOT / "results" / "kalshi_nat4_to_2_inventory_target_dev" / f"{session.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in {
        "headline_summary": headline, "chronological_robustness": chrono,
        "side_summary": side, "inventory_bucket_summary": inventory,
        "contract_summary": cdf, "fills": fdf, "quote_episodes": edf,
        "policy_counts": count_df, "data_quality": audit,
        "fixed_q100_comparison": compare,
    }.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "study_config.json").write_text(json.dumps({
        "study_version": STUDY_VERSION, "session": str(session), "session_duration_hours": duration_h,
        "natural_spread_c": NATURAL_SPREAD_C, "improve_each_side_c": IMPROVE_C,
        "our_spread_c": OUR_SPREAD_C, "max_quote_qty": MAX_QUOTE_QTY,
        "inventory_target": TARGET_INVENTORY, "soft_inventory": SOFT_INVENTORY,
        "hard_inventory": HARD_INVENTORY,
        "sizing": "risk-reducing side Q100; risk-increasing side min(100, remaining hard-cap headroom); |I|>=125 labeled soft-skew",
        "fill_rule": "recorded aggressive trades only; no fills from BBO moves",
        "opposite_resize": "cancel immediately after inventory-changing fill; reopen next 1Hz sample",
        "fees": "excluded", "market_feedback": "ignored; public 1Hz BBO treated as exogenous",
        "status": "development mechanism diagnostic, not OOS",
    }, indent=2), encoding="utf-8")

    if show:
        h = headline.iloc[0]
        print("\n" + "=" * 126)
        print("NAT4->2 INSIDE-SPREAD Q100 — INVENTORY-TARGETED SIZING DEVELOPMENT REPLAY")
        print("=" * 126)
        print(f"fills={int(h.fill_events)} qty={h.fill_qty:.2f} | net=${h.net_mtm_pnl_before_fees:+.4f} | pnl/window=${h.pnl_per_window:+.4f}")
        print(f"gross edge={h.qty_weighted_gross_edge_c:+.3f}c | DD=${h.max_drawdown:+.4f} | worst=${h.worst_window_pnl:+.4f}")
        print(f"inventory: avg max={h.avg_max_abs_inventory:.2f} | p95 max={h.p95_max_abs_inventory:.2f} | observed max={h.max_abs_inventory_observed:.2f} | p95 ending |I|={h.p95_abs_ending_inventory:.2f}")
        print(f"sizing: avg displayed={h.avg_displayed_quote_qty:.2f} | soft-skew opens={h.soft_skew_open_pct:.2f}% | soft-skew fill qty={h.soft_skew_fill_qty_pct:.2f}%")
        print("markouts:", " | ".join(f"{x}s={h[f'qty_weighted_markout_{x}s_c']:+.3f}c" for x in MARKOUTS))
        print("\nCHRONOLOGY\n", chrono.round(4).to_string(index=False))
        print("\nSIDE\n", side.round(4).to_string(index=False) if len(side) else "no fills")
        print("\nINVENTORY BUCKETS\n", inventory.round(4).to_string(index=False) if len(inventory) else "no fills")
        if len(compare):
            print("\nVERSUS FIXED Q100 NAT4->2\n", compare.round(4).to_string(index=False))
        print("\nCAVEAT: improved Q100 quotes can alter future behavior; public 1Hz BBO is treated as exogenous.")
        print("Outputs:", out)
        print("=" * 126)

    return {
        "output_dir": out, "headline": headline, "chronology": chrono,
        "side": side, "inventory": inventory, "contracts": cdf,
        "fills": fdf, "episodes": edf, "comparison": compare, "audit": audit,
    }
