from __future__ import annotations

"""Winner/loser diagnostics for NAT4->2 inventory-targeted MM development.

This module is descriptive only. It does NOT alter the strategy, search for a
threshold, select assets/sides, or produce a new trading policy.

It separates three questions:
1) Mechanical explanation: what actually made winning and losing windows differ?
2) Pre-window state: were M0-M1 conditions different before M1 quoting began?
3) Quote/fill-time observable state: did pre-fill momentum/flow/depth context differ?

Post-fill markouts and full-window M1-M5 regime variables are intentionally kept
out of the pre-window predictor table to avoid look-ahead leakage.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_reconstructed_m1_m5_backtest as B
from . import mm_oos_4c_audit_replay as O
from . import mm_oos_4c_compact_recorder_v2 as R

STUDY_VERSION = "NAT4_TO_2_WINDOW_DIAGNOSTICS_V1"
EPS = 1e-9
MARKOUTS = (5, 15, 30, 60)
EXACT4_TOL_C = 0.05


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _wavg(df, col, weight="qty"):
    if df.empty or col not in df.columns or weight not in df.columns:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    w = pd.to_numeric(df[weight], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def _mean(df, col):
    if df.empty or col not in df.columns:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce")
    return float(x.mean()) if x.notna().any() else np.nan


def _load_study(study_dir):
    study = Path(study_dir).resolve()
    required = ["contract_summary.csv", "fills.csv", "quote_episodes.csv", "study_config.json"]
    missing = [x for x in required if not (study / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing study files in {study}: {missing}")

    config = json.loads((study / "study_config.json").read_text(encoding="utf-8"))
    expected = {
        "natural_spread_c": 4.0,
        "our_spread_c": 2.0,
        "max_quote_qty": 100.0,
        "inventory_target": 0.0,
        "soft_inventory": 125.0,
        "hard_inventory": 200.0,
    }
    bad = []
    for k, v in expected.items():
        if abs(_f(config.get(k)) - v) > EPS:
            bad.append(f"{k}={config.get(k)!r}, expected {v}")
    if bad:
        raise RuntimeError("Study verification failed:\n - " + "\n - ".join(bad))

    cdf = pd.read_csv(study / "contract_summary.csv")
    fdf = pd.read_csv(study / "fills.csv")
    edf = pd.read_csv(study / "quote_episodes.csv")
    if cdf.empty:
        raise RuntimeError("contract_summary.csv is empty")
    cdf["close_ts"] = pd.to_numeric(cdf["close_ts"], errors="coerce")
    cdf = cdf[np.isfinite(cdf["close_ts"])].copy()
    return study, config, cdf, fdf, edf


def _winner_label(pnl):
    if pnl > EPS:
        return "WINNER"
    if pnl < -EPS:
        return "LOSER"
    return "FLAT"


def _exact4(s):
    return B._valid_sample(s) and abs(float(s.spread_c) - 4.0) <= EXACT4_TOL_C


def _interval_contract_features(ticker, close_ts, samples, trades, start_offset_s, end_offset_s):
    window_start = float(close_ts) - 900.0
    a = window_start + float(start_offset_s)
    b = window_start + float(end_offset_s)
    ss = sorted([s for s in samples if a <= s.t < b and B._valid_sample(s)], key=lambda z: z.t)
    tr = [x for x in trades if a <= x.t < b]

    mids = np.asarray([s.mid for s in ss], dtype=float)
    spreads = np.asarray([s.spread_c for s in ss], dtype=float)
    depth_imb = []
    exact4 = []
    diff_c = []
    run_count = 0
    longest_run = 0.0
    current_run = 0.0
    prev = None
    prev_exact = False

    for s in ss:
        denom = float(s.bid1_qty + s.ask1_qty)
        if denom > EPS:
            depth_imb.append((float(s.bid1_qty) - float(s.ask1_qty)) / denom)
        is4 = _exact4(s)
        exact4.append(is4)
        if is4:
            if prev is not None and prev_exact and s.t - prev.t <= 1.5:
                current_run += max(0.0, s.t - prev.t)
            else:
                run_count += 1
                current_run = 1.0
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0.0
        if prev is not None and s.t - prev.t <= 1.5:
            diff_c.append(100.0 * (float(s.mid) - float(prev.mid)))
        prev = s
        prev_exact = is4

    buy_qty = sum(float(x.qty) for x in tr if x.taker_book_side == "bid")
    sell_qty = sum(float(x.qty) for x in tr if x.taker_book_side == "ask")
    total_qty = buy_qty + sell_qty
    flow_imb = (buy_qty - sell_qty) / total_qty if total_qty > EPS else 0.0
    d = np.asarray(diff_c, dtype=float)

    return {
        "ticker": ticker,
        "close_ts": float(close_ts),
        "valid_seconds": len(ss),
        "exact4_seconds": int(sum(exact4)),
        "exact4_run_count": run_count,
        "longest_exact4_run_s": longest_run,
        "mean_spread_c": float(np.mean(spreads)) if len(spreads) else np.nan,
        "mean_depth_imbalance": float(np.mean(depth_imb)) if depth_imb else np.nan,
        "mid_move_c": 100.0 * (mids[-1] - mids[0]) if len(mids) >= 2 else np.nan,
        "mid_range_c": 100.0 * (np.max(mids) - np.min(mids)) if len(mids) else np.nan,
        "mid_diff_sq_sum_c2": float(np.sum(d * d)) if len(d) else 0.0,
        "mid_diff_abs_sum_c": float(np.sum(np.abs(d))) if len(d) else 0.0,
        "mid_diff_count": len(d),
        "trade_count": len(tr),
        "aggressive_buy_qty": buy_qty,
        "aggressive_sell_qty": sell_qty,
        "aggressive_total_qty": total_qty,
        "aggressive_flow_imbalance": flow_imb,
    }


def _aggregate_interval(contract_interval_df, prefix):
    rows = []
    for close_ts, z in contract_interval_df.groupby("close_ts", sort=True):
        valid_seconds = float(z["valid_seconds"].sum())
        exact4_seconds = float(z["exact4_seconds"].sum())
        diff_n = float(z["mid_diff_count"].sum())
        buy = float(z["aggressive_buy_qty"].sum())
        sell = float(z["aggressive_sell_qty"].sum())
        total = buy + sell
        row = {
            "close_ts": float(close_ts),
            f"{prefix}_contracts": len(z),
            f"{prefix}_valid_seconds": valid_seconds,
            f"{prefix}_exact4_pct": 100.0 * exact4_seconds / valid_seconds if valid_seconds > 0 else np.nan,
            f"{prefix}_exact4_runs": float(z["exact4_run_count"].sum()),
            f"{prefix}_longest_exact4_run_s": float(z["longest_exact4_run_s"].max()),
            f"{prefix}_mean_spread_c": np.average(z["mean_spread_c"], weights=z["valid_seconds"]) if valid_seconds > 0 else np.nan,
            f"{prefix}_mean_depth_imbalance": np.average(
                z["mean_depth_imbalance"].fillna(0.0), weights=z["valid_seconds"]
            ) if valid_seconds > 0 else np.nan,
            f"{prefix}_mean_mid_move_c": float(z["mid_move_c"].mean()),
            f"{prefix}_mean_abs_mid_move_c": float(z["mid_move_c"].abs().mean()),
            f"{prefix}_mean_mid_range_c": float(z["mid_range_c"].mean()),
            f"{prefix}_max_mid_range_c": float(z["mid_range_c"].max()),
            f"{prefix}_rms_1s_mid_move_c": np.sqrt(float(z["mid_diff_sq_sum_c2"].sum()) / diff_n) if diff_n > 0 else np.nan,
            f"{prefix}_mean_abs_1s_mid_move_c": float(z["mid_diff_abs_sum_c"].sum()) / diff_n if diff_n > 0 else np.nan,
            f"{prefix}_trade_count": int(z["trade_count"].sum()),
            f"{prefix}_aggressive_total_qty": total,
            f"{prefix}_aggressive_flow_imbalance": (buy - sell) / total if total > EPS else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _run_age_maps(samples_by_ticker):
    maps = {}
    for ticker, ss0 in samples_by_ticker.items():
        ss = sorted(ss0, key=lambda z: z.t)
        out = {}
        prev_t = None
        prev_exact = False
        age = 0.0
        for s in ss:
            is4 = _exact4(s)
            if is4:
                if prev_t is not None and prev_exact and s.t - prev_t <= 1.5:
                    age += max(0.0, s.t - prev_t)
                else:
                    age = 1.0
                out[float(s.t)] = age
            else:
                age = 0.0
            prev_t = s.t
            prev_exact = is4
        maps[ticker] = out
    return maps


def _enrich_episodes(edf, ticker_close, samples_by_ticker, run_age_maps):
    if edf.empty:
        return edf.copy()
    x = edf.copy()
    x["close_ts"] = x["ticker"].map(ticker_close)
    bidq, askq, depth, runage = [], [], [], []
    sample_maps = {k: {float(s.t): s for s in v} for k, v in samples_by_ticker.items()}
    for r in x.itertuples(index=False):
        t = _f(getattr(r, "join_ts", np.nan))
        ticker = str(getattr(r, "ticker"))
        s = sample_maps.get(ticker, {}).get(float(t)) if np.isfinite(t) else None
        if s is None:
            bidq.append(np.nan); askq.append(np.nan); depth.append(np.nan); runage.append(np.nan)
            continue
        bq, aq = float(s.bid1_qty), float(s.ask1_qty)
        den = bq + aq
        bidq.append(bq); askq.append(aq)
        depth.append((bq - aq) / den if den > EPS else np.nan)
        runage.append(run_age_maps.get(ticker, {}).get(float(t), np.nan))
    x["natural_bid_qty_at_join"] = bidq
    x["natural_ask_qty_at_join"] = askq
    x["natural_depth_imbalance_at_join"] = depth
    x["exact4_run_age_s_at_join"] = runage
    return x


def _fill_window_features(fdf, ticker_close, enriched_episodes):
    if fdf.empty:
        return pd.DataFrame(columns=["close_ts"])
    x = fdf.copy()
    x["close_ts"] = x["ticker"].map(ticker_close)
    epcols = [
        "episode_id", "natural_bid_qty_at_join", "natural_ask_qty_at_join",
        "natural_depth_imbalance_at_join", "exact4_run_age_s_at_join", "quote_qty_initial",
    ]
    avail = [c for c in epcols if c in enriched_episodes.columns]
    if "episode_id" in avail:
        x = x.merge(enriched_episodes[avail].drop_duplicates("episode_id"), on="episode_id", how="left")

    rows = []
    for close_ts, z in x.groupby("close_ts", sort=True):
        bid = z[z["side"] == "BID"]
        ask = z[z["side"] == "ASK"]
        qty = float(z["qty"].sum())
        bidq = float(bid["qty"].sum()) if len(bid) else 0.0
        askq = float(ask["qty"].sum()) if len(ask) else 0.0
        r = {
            "close_ts": float(close_ts),
            "fill_events": len(z), "fill_qty": qty,
            "bid_fill_qty": bidq, "ask_fill_qty": askq,
            "fill_side_imbalance": (bidq - askq) / qty if qty > EPS else 0.0,
            "avg_fill_qty": float(z["qty"].mean()),
            "qw_join_momentum_3s_c": _wavg(z, "momentum_3s_c_at_join"),
            "qw_abs_join_momentum_3s_c": _wavg(z.assign(_abs=pd.to_numeric(z["momentum_3s_c_at_join"], errors="coerce").abs()), "_abs"),
            "qw_join_flow_imbalance_5s": _wavg(z, "flow_imbalance_5s_at_join"),
            "qw_abs_inventory_at_join": _wavg(z.assign(_abs=pd.to_numeric(z["inventory_at_join"], errors="coerce").abs()), "_abs"),
            "qw_fill_latency_s": _wavg(z, "fill_latency_s"),
            "qw_historical_trade_qty": _wavg(z, "historical_trade_qty"),
            "qw_historical_trade_participation_pct": _wavg(z, "historical_trade_participation_pct"),
            "qw_natural_depth_imbalance_at_join": _wavg(z, "natural_depth_imbalance_at_join"),
            "qw_exact4_run_age_s_at_join": _wavg(z, "exact4_run_age_s_at_join"),
        }
        for h in MARKOUTS:
            r[f"qw_markout_{h}s_c"] = _wavg(z, f"markout_{h}s_c")
            r[f"bid_qw_markout_{h}s_c"] = _wavg(bid, f"markout_{h}s_c")
            r[f"ask_qw_markout_{h}s_c"] = _wavg(ask, f"markout_{h}s_c")
        rows.append(r)
    return pd.DataFrame(rows)


def _quote_window_features(edf):
    if edf.empty or "close_ts" not in edf.columns:
        return pd.DataFrame(columns=["close_ts"])
    rows = []
    for close_ts, z in edf.groupby("close_ts", sort=True):
        r = {
            "close_ts": float(close_ts),
            "quote_episodes": len(z),
            "quote_bid_pct": 100.0 * (z["side"] == "BID").mean(),
            "quote_mean_momentum_3s_c": _mean(z, "momentum_3s_c_at_join"),
            "quote_mean_abs_momentum_3s_c": pd.to_numeric(z["momentum_3s_c_at_join"], errors="coerce").abs().mean(),
            "quote_mean_flow_imbalance_5s": _mean(z, "flow_imbalance_5s_at_join"),
            "quote_mean_abs_inventory": pd.to_numeric(z["inventory_at_join"], errors="coerce").abs().mean(),
            "quote_mean_displayed_qty": _mean(z, "quote_qty_initial"),
            "quote_mean_natural_depth_imbalance": _mean(z, "natural_depth_imbalance_at_join"),
            "quote_mean_exact4_run_age_s": _mean(z, "exact4_run_age_s_at_join"),
        }
        rows.append(r)
    return pd.DataFrame(rows)


def _mechanical_window_base(cdf):
    x = cdf.copy()
    x["residual_inventory_mtm_pnl"] = (
        pd.to_numeric(x["net_mtm_pnl_before_fees"], errors="coerce")
        - pd.to_numeric(x["matched_roundtrip_pnl"], errors="coerce")
    )
    rows = []
    for close_ts, z in x.groupby("close_ts", sort=True):
        pnl = float(z["net_mtm_pnl_before_fees"].sum())
        rows.append({
            "close_ts": float(close_ts),
            "close_time": B._iso(float(close_ts)),
            "window_class": _winner_label(pnl),
            "net_pnl": pnl,
            "gross_capture": float(z["gross_spread_capture_dollars"].sum()),
            "adverse_selection_to_m5": float(z["adverse_selection_to_m5_dollars"].sum()),
            "matched_roundtrip_pnl": float(z["matched_roundtrip_pnl"].sum()),
            "residual_inventory_mtm_pnl": float(z["residual_inventory_mtm_pnl"].sum()),
            "matched_roundtrip_qty": float(z["matched_roundtrip_qty"].sum()),
            "contract_fill_qty": float(z["fill_qty"].sum()),
            "contracts_with_fill": int((pd.to_numeric(z["fill_qty"], errors="coerce") > EPS).sum()),
            "max_contract_abs_inventory": float(z["max_abs_inventory"].max()),
            "sum_abs_ending_inventory": float(pd.to_numeric(z["ending_inventory_yes_equiv"], errors="coerce").abs().sum()),
            "net_ending_inventory": float(pd.to_numeric(z["ending_inventory_yes_equiv"], errors="coerce").sum()),
        })
    return pd.DataFrame(rows)


def _comparison_table(windows, features, category):
    rows = []
    win = windows[windows["window_class"] == "WINNER"]
    lose = windows[windows["window_class"] == "LOSER"]
    for feature in features:
        if feature not in windows.columns:
            continue
        a = pd.to_numeric(win[feature], errors="coerce").dropna()
        b = pd.to_numeric(lose[feature], errors="coerce").dropna()
        allx = pd.to_numeric(windows[feature], errors="coerce")
        pnl = pd.to_numeric(windows["net_pnl"], errors="coerce")
        ok = np.isfinite(allx) & np.isfinite(pnl)
        pooled = np.nan
        if len(a) >= 2 and len(b) >= 2:
            va, vb = a.var(ddof=1), b.var(ddof=1)
            denom = np.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / max(1, len(a)+len(b)-2))
            pooled = (a.mean() - b.mean()) / denom if denom > EPS else np.nan
        pearson = windows.loc[ok, [feature, "net_pnl"]].corr(method="pearson").iloc[0, 1] if ok.sum() >= 3 else np.nan
        spearman = windows.loc[ok, [feature, "net_pnl"]].corr(method="spearman").iloc[0, 1] if ok.sum() >= 3 else np.nan
        rows.append({
            "category": category, "feature": feature,
            "winner_n": len(a), "loser_n": len(b),
            "winner_mean": a.mean() if len(a) else np.nan,
            "loser_mean": b.mean() if len(b) else np.nan,
            "mean_difference": a.mean() - b.mean() if len(a) and len(b) else np.nan,
            "winner_median": a.median() if len(a) else np.nan,
            "loser_median": b.median() if len(b) else np.nan,
            "standardized_mean_difference": pooled,
            "pearson_with_window_pnl": pearson,
            "spearman_with_window_pnl": spearman,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["abs_standardized_difference"] = out["standardized_mean_difference"].abs()
        out = out.sort_values(["abs_standardized_difference", "feature"], ascending=[False, True])
    return out


def run_nat4_to_2_window_diagnostics(session_dir, strategy_result_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    study, config, cdf, fdf, edf = _load_study(strategy_result_dir)

    meta = O._metadata(session)
    samples, info, duplicates, _ = O._bbo(session, meta)
    trades, _ = O._trades(session, meta)

    ticker_close = dict(zip(cdf["ticker"].astype(str), cdf["close_ts"].astype(float)))
    ticker_series = dict(zip(cdf["ticker"].astype(str), cdf["series"].astype(str)))
    targets = sorted(ticker_close)

    pre_rows, live_rows = [], []
    for ticker in targets:
        close = ticker_close[ticker]
        pre = _interval_contract_features(ticker, close, samples.get(ticker, []), trades.get(ticker, []), 0.0, 60.0)
        live = _interval_contract_features(ticker, close, samples.get(ticker, []), trades.get(ticker, []), 60.0, 300.0)
        pre["series"] = ticker_series.get(ticker); live["series"] = ticker_series.get(ticker)
        pre_rows.append(pre); live_rows.append(live)

    pre_contract = pd.DataFrame(pre_rows)
    live_contract = pd.DataFrame(live_rows)
    pre_window = _aggregate_interval(pre_contract, "pre_m0_m1")
    live_window = _aggregate_interval(live_contract, "live_m1_m5")

    run_age_maps = _run_age_maps(samples)
    enriched_ep = _enrich_episodes(edf, ticker_close, samples, run_age_maps)
    quote_window = _quote_window_features(enriched_ep)
    fill_window = _fill_window_features(fdf, ticker_close, enriched_ep)
    windows = _mechanical_window_base(cdf)

    for extra in (fill_window, quote_window, pre_window, live_window):
        windows = windows.merge(extra, on="close_ts", how="left")

    windows = windows.sort_values("close_ts").reset_index(drop=True)
    windows["chronological_index"] = np.arange(1, len(windows) + 1)
    windows["matched_fill_pct"] = np.where(
        pd.to_numeric(windows.get("fill_qty", 0), errors="coerce") > EPS,
        100.0 * pd.to_numeric(windows["matched_roundtrip_qty"], errors="coerce") /
        pd.to_numeric(windows.get("fill_qty", np.nan), errors="coerce"),
        np.nan,
    )

    if not fdf.empty:
        af = fdf.copy()
        af["close_ts"] = af["ticker"].map(ticker_close)
        asset_fill = af.pivot_table(index="close_ts", columns="series", values="qty", aggfunc="sum", fill_value=0.0).reset_index()
    else:
        asset_fill = pd.DataFrame(columns=["close_ts"])

    mechanical_features = [
        "gross_capture", "adverse_selection_to_m5", "matched_roundtrip_pnl", "residual_inventory_mtm_pnl",
        "fill_qty", "bid_fill_qty", "ask_fill_qty", "fill_side_imbalance", "avg_fill_qty",
        "matched_fill_pct", "max_contract_abs_inventory", "sum_abs_ending_inventory", "net_ending_inventory",
        "qw_markout_5s_c", "qw_markout_15s_c", "qw_markout_30s_c", "qw_markout_60s_c",
        "bid_qw_markout_5s_c", "bid_qw_markout_15s_c", "bid_qw_markout_30s_c", "bid_qw_markout_60s_c",
        "ask_qw_markout_5s_c", "ask_qw_markout_15s_c", "ask_qw_markout_30s_c", "ask_qw_markout_60s_c",
        "live_m1_m5_exact4_pct", "live_m1_m5_longest_exact4_run_s", "live_m1_m5_mean_mid_move_c",
        "live_m1_m5_mean_abs_mid_move_c", "live_m1_m5_mean_mid_range_c", "live_m1_m5_rms_1s_mid_move_c",
        "live_m1_m5_aggressive_total_qty", "live_m1_m5_aggressive_flow_imbalance",
    ]
    prewindow_features = [
        "pre_m0_m1_exact4_pct", "pre_m0_m1_exact4_runs", "pre_m0_m1_longest_exact4_run_s",
        "pre_m0_m1_mean_spread_c", "pre_m0_m1_mean_depth_imbalance", "pre_m0_m1_mean_mid_move_c",
        "pre_m0_m1_mean_abs_mid_move_c", "pre_m0_m1_mean_mid_range_c", "pre_m0_m1_max_mid_range_c",
        "pre_m0_m1_rms_1s_mid_move_c", "pre_m0_m1_mean_abs_1s_mid_move_c",
        "pre_m0_m1_trade_count", "pre_m0_m1_aggressive_total_qty", "pre_m0_m1_aggressive_flow_imbalance",
    ]
    quote_state_features = [
        "quote_mean_momentum_3s_c", "quote_mean_abs_momentum_3s_c", "quote_mean_flow_imbalance_5s",
        "quote_mean_abs_inventory", "quote_mean_displayed_qty", "quote_mean_natural_depth_imbalance",
        "quote_mean_exact4_run_age_s", "qw_join_momentum_3s_c", "qw_abs_join_momentum_3s_c",
        "qw_join_flow_imbalance_5s", "qw_abs_inventory_at_join", "qw_natural_depth_imbalance_at_join",
        "qw_exact4_run_age_s_at_join", "qw_historical_trade_qty",
    ]

    mech_cmp = _comparison_table(windows, mechanical_features, "MECHANICAL_POST_OUTCOME")
    pre_cmp = _comparison_table(windows, prewindow_features, "PRE_WINDOW_M0_M1")
    quote_cmp = _comparison_table(windows, quote_state_features, "QUOTE_FILL_TIME_OBSERVABLE")

    out = Path(output_dir) if output_dir else R.PROJECT_ROOT / "results" / "kalshi_nat4_to_2_window_diagnostics" / f"{session.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in {
        "window_diagnostics": windows,
        "winner_loser_mechanical": mech_cmp,
        "winner_loser_prewindow": pre_cmp,
        "winner_loser_quote_state": quote_cmp,
        "asset_fill_by_window": asset_fill,
        "contract_prewindow_regime": pre_contract,
        "contract_m1_m5_regime": live_contract,
        "enriched_quote_episodes": enriched_ep,
    }.items():
        df.to_csv(out / f"{name}.csv", index=False)

    (out / "study_config.json").write_text(json.dumps({
        "study_version": STUDY_VERSION,
        "session": str(session),
        "strategy_result_dir": str(study),
        "verified_strategy_config": config,
        "purpose": "descriptive winner/loser decomposition and predictor discovery only",
        "no_strategy_changes": True,
        "prewindow_definition": "M0 <= t < M1; available before M1 quoting begins",
        "live_window_definition": "M1 <= t < M5; explanatory only, not ex-ante",
        "warning": "25 windows is a small development sample; correlations are descriptive and not validation",
    }, indent=2, default=str), encoding="utf-8")

    if show:
        winners = int((windows.window_class == "WINNER").sum())
        losers = int((windows.window_class == "LOSER").sum())
        flats = int((windows.window_class == "FLAT").sum())
        print("\n" + "=" * 150)
        print("NAT4->2 WINDOW DIAGNOSTICS — WHY DO SOME 15-MINUTE WINDOWS WIN AND OTHERS LOSE?")
        print("=" * 150)
        print(f"windows={len(windows)} | winners={winners} | losers={losers} | flat={flats} | total pnl=${windows.net_pnl.sum():+.4f}")

        show_cols = [
            "chronological_index", "close_time", "window_class", "net_pnl", "gross_capture",
            "adverse_selection_to_m5", "matched_roundtrip_pnl", "residual_inventory_mtm_pnl",
            "fill_qty", "bid_fill_qty", "ask_fill_qty", "qw_markout_60s_c",
            "pre_m0_m1_rms_1s_mid_move_c", "pre_m0_m1_aggressive_flow_imbalance",
        ]
        show_cols = [c for c in show_cols if c in windows.columns]
        print("\nWINDOW-BY-WINDOW")
        print(windows[show_cols].round(4).to_string(index=False))

        def top(df, title):
            print(f"\n{title}")
            if df.empty:
                print("no comparable features")
                return
            cols = ["feature", "winner_mean", "loser_mean", "mean_difference", "standardized_mean_difference", "spearman_with_window_pnl"]
            print(df[cols].head(12).round(4).to_string(index=False))

        top(mech_cmp, "TOP REALIZED / MECHANICAL DIFFERENCES — explanation, NOT predictors")
        top(pre_cmp, "TOP PRE-WINDOW M0-M1 DIFFERENCES — genuinely available before quoting")
        top(quote_cmp, "TOP QUOTE/FILL-TIME OBSERVABLE DIFFERENCES — available during execution")

        print("\nINTERPRETATION GUARDRAIL")
        print("  Post-fill markouts and full M1-M5 regime variables explain outcomes but cannot be used as ex-ante filters.")
        print("  M0-M1 variables are the cleanest prospective candidates because they exist before M1 quoting starts.")
        print("  Quote/fill-time variables may support dynamic quote cancellation/sizing hypotheses, but this run does not choose thresholds.")
        print("  With only 25 development windows, treat every correlation as descriptive until it survives another recording.")
        print("\nOutputs:", out)
        print("=" * 150)

    return {
        "output_dir": out,
        "windows": windows,
        "mechanical_comparison": mech_cmp,
        "prewindow_comparison": pre_cmp,
        "quote_state_comparison": quote_cmp,
        "asset_fill_by_window": asset_fill,
        "prewindow_contracts": pre_contract,
        "live_contracts": live_contract,
        "enriched_episodes": enriched_ep,
    }
