from __future__ import annotations

"""Exact path-dependent quote-lifetime replay for Defensive M1-M5 MM V1.

This study changes ONE mechanism only: maximum resting quote lifetime.

For each scenario (unlimited, 1s, 2s, 3s, 5s):
- reload the same validated reconstructed 1 Hz books and recorded trades;
- keep every Defensive V1 rule unchanged;
- if a live quote reaches its lifetime between book samples, cancel it at the
  exact deadline before any later trade can fill it;
- after an age expiry, that side cannot re-enter from the stale state; it may
  re-enter only when the next valid 1 Hz reconstructed book sample arrives and
  the full Defensive V1 policy is re-evaluated;
- queue priority is reset on every re-entry;
- all PnL/inventory paths are recomputed from scratch.

The unlimited scenario is required to reproduce the saved Defensive V1 result.
No asset/minute/asymmetric/spread retuning is performed.
Fees are excluded.
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

from . import mm_reconstructed_m1_m5_backtest as B
from . import mm_defensive_m1_m5_v1 as D

STUDY_VERSION = "M1_M5_DEFENSIVE_V1_EXACT_QUOTE_LIFETIME_V1"
EPS = 1e-9
DEFAULT_LIFETIMES = (None, 1.0, 2.0, 3.0, 5.0)
DEFAULT_MARKOUT_SECONDS = (5, 15, 30, 60)


def _scenario_name(max_age_s):
    return "UNLIMITED" if max_age_s is None else f"MAX_AGE_{float(max_age_s):g}S"


def _simulate_contract(
    ticker,
    meta,
    samples,
    trades,
    quote_qty,
    markouts,
    max_markout_lag_s,
    policy,
    max_age_s,
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
    trade_times = [tr.t for tr in trades]

    episodes = []
    fills = []
    active = {"BID": None, "ASK": None}
    remaining = {"BID": 0.0, "ASK": 0.0}
    last_fill_ts = {"BID": -np.inf, "ASK": -np.inf}
    expired_waiting_for_next_sample = {"BID": False, "ASK": False}
    episode_id = 0

    inventory = 0.0
    cash = 0.0
    max_abs_inventory = 0.0
    last_valid_mid = np.nan
    current_sample = None
    current_mid = np.nan
    mid_hist = deque()
    flow_hist = deque()
    counts = Counter()

    last_sample_t = samples[0].t
    tr_idx = bisect.bisect_left(trade_times, last_sample_t)

    def features(now_t):
        D._purge_flow_history(flow_hist, now_t, policy["flow_lookback_s"])
        return D._momentum_c(mid_hist), D._flow_imbalance(flow_hist)

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
            "join_ts": float(s.t),
            "join_time": B._iso(s.t),
            "join_minute": s.minute,
            "price": float(px),
            "queue_ahead_initial": float(qa),
            "queue_ahead_final": float(qa),
            "spread_c_at_join": s.spread_c,
            "mid_at_join": s.mid,
            "momentum_3s_c_at_join": momentum_c,
            "flow_imbalance_5s_at_join": flow_imb,
            "inventory_at_join": inventory,
            "max_age_s": np.nan if max_age_s is None else float(max_age_s),
            "expiry_ts": np.nan if max_age_s is None else float(s.t + max_age_s),
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
        counts[f"{side}_OPEN"] += 1

    def cancel(side, t, reason, *, wait_for_sample=False):
        ep = active[side]
        if ep is not None:
            B._close_episode(ep, t, reason)
            counts[f"{side}_CANCEL_{reason}"] += 1
        active[side] = None
        remaining[side] = 0.0
        if wait_for_sample:
            expired_waiting_for_next_sample[side] = True

    def expire_due(now_t):
        if max_age_s is None:
            return
        for side in ("BID", "ASK"):
            ep = active[side]
            if ep is None:
                continue
            deadline = float(ep["join_ts"]) + float(max_age_s)
            if deadline <= now_t + EPS:
                cancel(side, deadline, "MAX_AGE", wait_for_sample=True)

    def enforce_policy(now_t, s, allow_open):
        if s is None or not B._valid_sample(s):
            cancel("BID", now_t, "INVALID_BOOK")
            cancel("ASK", now_t, "INVALID_BOOK")
            return

        mom, flow = features(now_t)
        for side in ("BID", "ASK"):
            reasons = D._policy_reasons(
                side,
                s,
                inventory,
                quote_qty,
                last_fill_ts,
                now_t,
                mom,
                flow,
                policy,
            )
            if reasons:
                for reason in reasons:
                    counts[f"{side}_BLOCK_{reason}"] += 1
                if active[side] is not None:
                    cancel(side, now_t, "POLICY_" + reasons[0])
            elif allow_open and active[side] is None and not expired_waiting_for_next_sample[side]:
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
            "quote_age_s": tr.t - ep["join_ts"],
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
        counts[f"{side}_FILL_EVENT"] += 1

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
                    expire_due(tr.t)
                    fill_order("BID", tr, current_mid)
                    fill_order("ASK", tr, current_mid)

                    signed = 0.0
                    if tr.taker_book_side == "bid":
                        signed = float(tr.qty)
                    elif tr.taker_book_side == "ask":
                        signed = -float(tr.qty)
                    if abs(signed) > EPS:
                        flow_hist.append((tr.t, signed))
                        D._purge_flow_history(flow_hist, tr.t, policy["flow_lookback_s"])

                    enforce_policy(tr.t, current_sample, allow_open=False)
                tr_idx += 1
            last_sample_t = s.t

        expire_due(s.t)

        for side in ("BID", "ASK"):
            expired_waiting_for_next_sample[side] = False

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
        D._purge_mid_history(mid_hist, s.t, policy["momentum_lookback_s"])

        for side, px in (("BID", s.bid1), ("ASK", s.ask1)):
            ep = active[side]
            if ep is not None and abs(float(ep["price"]) - px) > EPS:
                cancel(side, s.t, "BBO_REPRICE")

        enforce_policy(s.t, s, allow_open=True)

    end_t = min(wend, samples[-1].t)
    expire_due(end_t)
    cancel("BID", end_t, "M5_END")
    cancel("ASK", end_t, "M5_END")

    if not np.isfinite(last_valid_mid):
        return episodes, fills, None, counts

    final_mid = last_valid_mid
    net_mtm = cash + inventory * final_mid
    gross_capture = sum((f["gross_edge_at_fill_c"] / 100.0) * f["qty"] for f in fills)
    adverse_to_m5 = net_mtm - gross_capture
    matched_qty, matched_pnl = B._match_roundtrips(fills)
    bid_qty = sum(f["qty"] for f in fills if f["side"] == "BID")
    ask_qty = sum(f["qty"] for f in fills if f["side"] == "ASK")

    contract = {
        "ticker": ticker,
        "series": series,
        "close_time": B._iso(close),
        "close_ts": close,
        "reconstructed_coverage_pct": meta.get("reconstructed_coverage_pct"),
        "sample_rows": len(samples),
        "valid_sample_rows": sum(B._valid_sample(s) for s in samples),
        "bid_fill_qty": bid_qty,
        "ask_fill_qty": ask_qty,
        "fill_qty": bid_qty + ask_qty,
        "both_sides_filled": bid_qty > EPS and ask_qty > EPS,
        "one_sided_fill": (bid_qty > EPS) ^ (ask_qty > EPS),
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
    return episodes, fills, contract, counts


def _window_summary(contract_df):
    if contract_df.empty:
        return pd.DataFrame()
    rows = []
    for close_time, g in contract_df.groupby("close_time", sort=True):
        rows.append({
            "close_time": close_time,
            "close_ts": g["close_ts"].iloc[0],
            "eligible_assets": len(g),
            "filled_assets": int((g["fill_qty"] > EPS).sum()),
            "fill_qty": g["fill_qty"].sum(),
            "net_mtm_pnl_before_fees": g["net_mtm_pnl_before_fees"].sum(),
        })
    w = pd.DataFrame(rows).sort_values("close_ts").reset_index(drop=True)
    w["cumulative_pnl"] = w["net_mtm_pnl_before_fees"].cumsum()
    w["running_peak"] = w["cumulative_pnl"].cummax()
    w["drawdown"] = w["cumulative_pnl"] - w["running_peak"]
    return w


def _chronology(window_df):
    if window_df.empty:
        return {}
    w = window_df.sort_values("close_ts").reset_index(drop=True)
    n = len(w)
    cut = n // 2
    out = {}
    for name, z in (("FIRST_HALF", w.iloc[:cut]), ("SECOND_HALF", w.iloc[cut:])):
        pnl = z["net_mtm_pnl_before_fees"].astype(float)
        out[f"{name.lower()}_pnl"] = pnl.sum()
        out[f"{name.lower()}_pnl_per_window"] = pnl.mean()
        out[f"{name.lower()}_positive_window_pct"] = 100.0 * (pnl > 0).mean()
    return out


def _scenario_summary(name, max_age_s, contract_df, fills_df, window_df, counts, markouts):
    total_pnl = contract_df["net_mtm_pnl_before_fees"].sum()
    fill_qty = fills_df["qty"].sum() if len(fills_df) else 0.0
    matched_qty = contract_df["matched_roundtrip_qty"].sum()
    matched_pnl = contract_df["matched_roundtrip_pnl"].sum()
    row = {
        "scenario": name,
        "max_age_s": np.nan if max_age_s is None else max_age_s,
        "eligible_contracts": len(contract_df),
        "independent_windows": len(window_df),
        "contracts_with_fill": int((contract_df["fill_qty"] > EPS).sum()),
        "fill_events": len(fills_df),
        "fill_qty": fill_qty,
        "gross_capture_dollars": contract_df["gross_spread_capture_dollars"].sum(),
        "adverse_selection_to_m5_dollars": contract_df["adverse_selection_to_m5_dollars"].sum(),
        "net_mtm_pnl_before_fees": total_pnl,
        "pnl_per_window": total_pnl / len(window_df) if len(window_df) else np.nan,
        "matched_roundtrip_qty": matched_qty,
        "matched_roundtrip_pnl": matched_pnl,
        "avg_max_abs_inventory": contract_df["max_abs_inventory"].mean(),
        "p95_max_abs_inventory": contract_df["max_abs_inventory"].quantile(0.95),
        "window_median_pnl": window_df["net_mtm_pnl_before_fees"].median() if len(window_df) else np.nan,
        "positive_window_pct": 100.0 * (window_df["net_mtm_pnl_before_fees"] > 0).mean() if len(window_df) else np.nan,
        "worst_window_pnl": window_df["net_mtm_pnl_before_fees"].min() if len(window_df) else np.nan,
        "best_window_pnl": window_df["net_mtm_pnl_before_fees"].max() if len(window_df) else np.nan,
        "max_drawdown": window_df["drawdown"].min() if len(window_df) else np.nan,
        "age_expiry_cancels": counts.get("BID_CANCEL_MAX_AGE", 0) + counts.get("ASK_CANCEL_MAX_AGE", 0),
        "avg_quote_age_at_fill_s": fills_df["quote_age_s"].mean() if len(fills_df) else np.nan,
        "avg_gross_edge_at_fill_c": fills_df["gross_edge_at_fill_c"].mean() if len(fills_df) else np.nan,
        "break_even_fee_c_per_fill_qty": (100.0 * total_pnl / fill_qty) if fill_qty > EPS else np.nan,
    }
    for h in markouts:
        row[f"avg_markout_{h}s_c"] = pd.to_numeric(
            fills_df.get(f"markout_{h}s_c"), errors="coerce"
        ).mean() if len(fills_df) else np.nan
    row.update(_chronology(window_df))
    return row


def _asset_summary(scenario, contract_df):
    rows = []
    for series, g in contract_df.groupby("series"):
        rows.append({
            "scenario": scenario,
            "series": series,
            "eligible_contracts": len(g),
            "contracts_with_fill": int((g["fill_qty"] > EPS).sum()),
            "fill_qty": g["fill_qty"].sum(),
            "net_mtm_pnl_before_fees": g["net_mtm_pnl_before_fees"].sum(),
            "pnl_per_eligible_contract": g["net_mtm_pnl_before_fees"].mean(),
            "avg_max_abs_inventory": g["max_abs_inventory"].mean(),
        })
    return pd.DataFrame(rows)


def _load_v1_reference(defensive_v1_dir):
    p = Path(defensive_v1_dir) / "headline_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError(f"Empty V1 headline: {p}")
    return df.iloc[0]


def _print_report(summary_df, v1_ref, out):
    print("\n" + "=" * 126)
    print("EXACT PATH-DEPENDENT QUOTE-LIFETIME REPLAY — DEFENSIVE V1 OTHERWISE UNCHANGED")
    print("=" * 126)
    cols = [
        "scenario", "fill_qty", "net_mtm_pnl_before_fees", "pnl_per_window",
        "matched_roundtrip_pnl", "avg_gross_edge_at_fill_c",
        "avg_markout_5s_c", "avg_markout_15s_c", "avg_markout_30s_c", "avg_markout_60s_c",
        "avg_quote_age_at_fill_s", "p95_max_abs_inventory",
        "worst_window_pnl", "max_drawdown",
        "first_half_pnl", "second_half_pnl", "positive_window_pct", "age_expiry_cancels",
    ]
    print(summary_df[cols].round(4).to_string(index=False))

    u = summary_df[summary_df["scenario"] == "UNLIMITED"].iloc[0]
    v1_pnl = B._f(v1_ref.get("net_mtm_pnl_before_fees"))
    diff = float(u["net_mtm_pnl_before_fees"]) - v1_pnl
    print("\nREPRODUCTION CHECK")
    print(f"  Saved Defensive V1 PnL: ${v1_pnl:+.6f}")
    print(f"  Unlimited replay PnL:   ${u['net_mtm_pnl_before_fees']:+.6f}")
    print(f"  Difference:             ${diff:+.8f}")

    print("\nINTERPRETATION RULE")
    print("  Unlimited must reproduce V1 before finite-lifetime results are trusted.")
    print("  This is same-session exploratory mechanism testing, not OOS validation.")
    print("  Do not combine the best lifetime with new spread/side filters from this recording.")
    print("Outputs:", out)
    print("=" * 126)


def run_exact_quote_lifetime_replay(
    session_dir,
    reconstruction_dir,
    defensive_v1_dir,
    output_dir=None,
    *,
    quote_lifetimes_s=DEFAULT_LIFETIMES,
    quote_qty=1.0,
    markout_seconds=DEFAULT_MARKOUT_SECONDS,
    max_markout_lag_s=2.0,
    reproduction_tolerance_dollars=0.02,
    show=True,
):
    session = Path(session_dir)
    recon = Path(reconstruction_dir)
    v1_dir = Path(defensive_v1_dir)
    for p in (session, recon, v1_dir):
        if not p.exists():
            raise FileNotFoundError(p)

    lifetimes = []
    for x in quote_lifetimes_s:
        if x is None:
            lifetimes.append(None)
        else:
            x = float(x)
            if x <= 0:
                raise ValueError("Finite quote lifetimes must be > 0")
            lifetimes.append(x)
    if None not in lifetimes:
        lifetimes = [None] + lifetimes
    seen = set()
    normalized = []
    for x in lifetimes:
        key = "NONE" if x is None else float(x)
        if key not in seen:
            seen.add(key)
            normalized.append(x)
    lifetimes = normalized

    markouts = tuple(sorted({int(x) for x in markout_seconds if int(x) > 0}))
    policy = dict(D.DEFAULT_POLICY)

    quality_df, meta = B._load_quality_contracts(recon)
    eligible = set(meta)
    print(f"Validated reconstruction contracts: {len(eligible):,}")
    print("Loading validated 1 Hz reconstructed books once...")
    samples, sample_stats = B._load_reconstructed_samples(recon, eligible)
    missing = sorted(eligible - set(samples))
    if missing:
        raise RuntimeError(f"{len(missing)} eligible contracts missing samples; first={missing[:3]}")

    print("Streaming trades once...")
    trades, trade_stats = B._scan_trades(session, meta)

    if output_dir is None:
        root = session.resolve().parents[2] if len(session.resolve().parents) >= 3 else Path.cwd()
        output_dir = (
            root / "results" / "kalshi_mm_m1_m5_exact_quote_lifetime"
            / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targets = sorted(eligible, key=lambda x: (meta[x]["close_ts"], x))
    all_summaries = []
    all_asset = []
    scenario_returns = {}

    for max_age_s in lifetimes:
        name = _scenario_name(max_age_s)
        print(f"\nReplaying {name} on {len(targets):,} contracts...")
        t0 = time.time()
        episodes_all = []
        fills_all = []
        contracts = []
        counts = Counter()

        for i, ticker in enumerate(targets, 1):
            eps, fills, contract, c = _simulate_contract(
                ticker,
                meta[ticker],
                samples[ticker],
                trades.get(ticker, []),
                float(quote_qty),
                markouts,
                float(max_markout_lag_s),
                policy,
                max_age_s,
            )
            episodes_all.extend(eps)
            fills_all.extend(fills)
            counts.update(c)
            if contract is not None:
                contracts.append(contract)
            if i % 100 == 0 or i == len(targets):
                print(
                    f"  {name}: {i:,}/{len(targets):,} | "
                    f"fills={len(fills_all):,} | expiries="
                    f"{counts.get('BID_CANCEL_MAX_AGE',0)+counts.get('ASK_CANCEL_MAX_AGE',0):,} | "
                    f"{time.time()-t0:.1f}s"
                )

        contract_df = pd.DataFrame(contracts)
        fills_df = pd.DataFrame(fills_all)
        episodes_df = pd.DataFrame(episodes_all)
        window_df = _window_summary(contract_df)
        summary = _scenario_summary(name, max_age_s, contract_df, fills_df, window_df, counts, markouts)
        all_summaries.append(summary)
        all_asset.append(_asset_summary(name, contract_df))
        scenario_returns[name] = {
            "contracts": contract_df,
            "fills": fills_df,
            "episodes": episodes_df,
            "windows": window_df,
        }

        stem = name.lower()
        contract_df.to_csv(out / f"{stem}_contract_summary.csv", index=False)
        fills_df.to_csv(out / f"{stem}_fills.csv", index=False)
        episodes_df.to_csv(out / f"{stem}_quote_episodes.csv", index=False)
        window_df.to_csv(out / f"{stem}_window_summary.csv", index=False)
        pd.DataFrame(
            [{"reason": k, "count": v} for k, v in counts.most_common()]
        ).to_csv(out / f"{stem}_policy_counts.csv", index=False)

    summary_df = pd.DataFrame(all_summaries)
    unlimited_pnl = float(
        summary_df.loc[summary_df["scenario"] == "UNLIMITED", "net_mtm_pnl_before_fees"].iloc[0]
    )
    summary_df["pnl_change_vs_unlimited"] = summary_df["net_mtm_pnl_before_fees"] - unlimited_pnl
    summary_df["pnl_per_window_change_vs_unlimited"] = (
        summary_df["pnl_per_window"]
        - float(summary_df.loc[summary_df["scenario"] == "UNLIMITED", "pnl_per_window"].iloc[0])
    )
    asset_df = pd.concat(all_asset, ignore_index=True) if all_asset else pd.DataFrame()

    v1_ref = _load_v1_reference(v1_dir)
    v1_pnl = B._f(v1_ref.get("net_mtm_pnl_before_fees"))
    reproduction_diff = unlimited_pnl - v1_pnl
    reproduction_ok = abs(reproduction_diff) <= float(reproduction_tolerance_dollars)

    summary_df.to_csv(out / "scenario_summary.csv", index=False)
    asset_df.to_csv(out / "asset_by_scenario.csv", index=False)
    pd.DataFrame([{**sample_stats, **trade_stats}]).to_csv(out / "scan_stats.csv", index=False)

    config = {
        "study_version": STUDY_VERSION,
        "session": str(session.resolve()),
        "reconstruction_dir": str(recon.resolve()),
        "defensive_v1_dir": str(v1_dir.resolve()),
        "eligible_contracts": len(targets),
        "quote_lifetimes_s": ["UNLIMITED" if x is None else x for x in lifetimes],
        "quote_qty": quote_qty,
        "policy": policy,
        "mechanism_changed": "maximum resting quote lifetime only",
        "expiry_semantics": (
            "cancel at exact join_ts + max_age before any later trade; "
            "re-entry only at next valid 1 Hz book sample after full V1 policy re-evaluation"
        ),
        "queue_semantics": "re-entry joins back of current displayed L1 queue; no cancellation-ahead credit",
        "fees": "excluded",
        "saved_v1_pnl": v1_pnl,
        "unlimited_replay_pnl": unlimited_pnl,
        "reproduction_difference_dollars": reproduction_diff,
        "reproduction_tolerance_dollars": reproduction_tolerance_dollars,
        "reproduction_ok": reproduction_ok,
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if show:
        _print_report(summary_df, v1_ref, out)
        try:
            from IPython.display import display
            print("\nSCENARIO SUMMARY")
            display(summary_df.round(4))
            print("\nBY ASSET / SCENARIO")
            display(asset_df.round(4))
        except Exception:
            pass

    if not reproduction_ok:
        raise RuntimeError(
            f"Unlimited replay failed V1 reproduction: diff=${reproduction_diff:+.6f}, "
            f"tolerance=${reproduction_tolerance_dollars:.6f}. "
            "Do not interpret finite-lifetime scenarios."
        )

    return {
        "output_dir": out,
        "scenario_summary": summary_df,
        "asset_summary": asset_df,
        "scenarios": scenario_returns,
        "reproduction_ok": reproduction_ok,
        "reproduction_difference_dollars": reproduction_diff,
    }


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--session", required=True)
    p.add_argument("--reconstruction-dir", required=True)
    p.add_argument("--defensive-v1-dir", required=True)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    run_exact_quote_lifetime_replay(
        args.session,
        args.reconstruction_dir,
        args.defensive_v1_dir,
        output_dir=args.output_dir,
        show=True,
    )


if __name__ == "__main__":
    _main()
