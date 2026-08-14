from __future__ import annotations

"""State-aware / toxicity-aware market-making research suite.

DEVELOPMENT ONLY. Hard-bound to the already-burned compact session
20260813_190334. It must never read the newer pre-open V3 recording, which is
reserved for later validation of one frozen candidate.

What this tests on the old development session
-----------------------------------------------
1. Side-specific BBO dynamics and spread origin.
2. Quote-side age / staleness and L1 depth changes.
3. Aggressive-flow level and acceleration from event-time trades.
4. Cross-asset crypto-factor moves, BTC moves, and own residual moves.
5. A direct toxicity model for the 30-second future midpoint move.
6. Economic quote gating from the predicted fair-value shift:
      BID edge ~= 1c + predicted future midpoint move
      ASK edge ~= 1c - predicted future midpoint move
   on NAT4->2 opportunities. No threshold sweep: quote iff predicted edge > 0.
7. Fill-score diagnostics and post-fill toxicity concentration.
8. Portfolio-level aggregate YES-equivalent inventory diagnostics.

What this CANNOT test from this recording
-----------------------------------------
- true sub-second/event-time BBO reaction (BBO is only persisted at 1 Hz);
- actual market impact of displaying Q100 inside the public spread;
- real live maker fills / exchange queue behavior after changing the public BBO;
- a true execution replay of continuously price-skewed quotes away from the
  NAT4->2 inside-spread price. We report the implied reservation-price shift
  and use side suppression as the executable historical proxy.

The newest V3 recording is intentionally never opened by this module.
"""

import bisect
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_m1_m5_feasibility as F
from . import mm_reconstructed_m1_m5_backtest as B
from . import mm_oos_4c_audit_replay as O
from . import mm_nat4_to_2_inventory_target_dev_v1 as INV
from . import mm_exact_quote_lifetime_m1_m5_v1 as L
from . import mm_inside_spread_q100_dev_v1 as Q
from . import mm_oos_4c_compact_recorder_v2 as R

STUDY_VERSION = "STATE_AWARE_TOXICITY_MM_DEV_V1"
EXPECTED_SESSION_NAME = "20260813_190334"
RESERVED_VALIDATION_SESSION_NAME = "20260814_185216"
EPS = 1e-9
EXACT4_TOL_C = 0.05
TRAIN_WINDOW_FRACTION = 0.67
RIDGE_ALPHA = 1.0
TARGET_HORIZON_S = 30
MARKOUTS = (5, 15, 30, 60)
LAGS = (1, 3, 5, 10)
FLOW_WINDOWS = (1, 3, 5, 10)


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _iso(ts):
    return B._iso(float(ts))


def _exact4(s):
    return B._valid_sample(s) and abs(float(s.spread_c) - 4.0) <= EXACT4_TOL_C


def _past_sample(samples, times, now_t, lag_s, max_lag_s=2.0):
    target = float(now_t) - float(lag_s)
    j = bisect.bisect_right(times, target) - 1
    if j < 0:
        return None
    s = samples[j]
    if target - float(s.t) > max_lag_s + EPS:
        return None
    return s if B._valid_sample(s) else None


def _age_maps(samples):
    out = {}
    for ticker, seq0 in samples.items():
        seq = sorted(seq0, key=lambda z: z.t)
        bid_age = ask_age = spread_age = 0.0
        prev = None
        m = {}
        for s in seq:
            if not B._valid_sample(s):
                prev = None
                bid_age = ask_age = spread_age = 0.0
                continue
            if prev is None or s.t - prev.t > 2.5:
                bid_age = ask_age = spread_age = 0.0
            else:
                dt = max(0.0, float(s.t - prev.t))
                bid_age = bid_age + dt if abs(float(s.bid1) - float(prev.bid1)) <= EPS else 0.0
                ask_age = ask_age + dt if abs(float(s.ask1) - float(prev.ask1)) <= EPS else 0.0
                spread_age = spread_age + dt if abs(float(s.spread_c) - float(prev.spread_c)) <= EPS else 0.0
            m[float(s.t)] = (bid_age, ask_age, spread_age)
            prev = s
        out[ticker] = m
    return out


def _trade_cums(trades):
    out = {}
    for ticker, tr0 in trades.items():
        tr = sorted(tr0, key=lambda z: z.t)
        tt = np.asarray([float(x.t) for x in tr], dtype=float)
        buy = np.asarray([float(x.qty) if x.taker_book_side == "bid" else 0.0 for x in tr], dtype=float)
        sell = np.asarray([float(x.qty) if x.taker_book_side == "ask" else 0.0 for x in tr], dtype=float)
        out[ticker] = (tt, np.r_[0.0, np.cumsum(buy)], np.r_[0.0, np.cumsum(sell)])
    return out


def _flow_at(cum, now_t, window_s):
    if cum is None:
        return 0.0, 0.0, 0.0, 0.0
    tt, cb, cs = cum
    hi = bisect.bisect_right(tt, float(now_t))
    lo = bisect.bisect_left(tt, float(now_t) - float(window_s))
    buy = float(cb[hi] - cb[lo])
    sell = float(cs[hi] - cs[lo])
    total = buy + sell
    imb = (buy - sell) / total if total > EPS else 0.0
    return buy, sell, total, imb


def _pre_range_by_window(audit, samples):
    rows = []
    good = audit[audit["quality_ok"].astype(bool)].copy()
    for r in good.itertuples(index=False):
        close = float(r.close_ts)
        start = close - 900.0
        ss = [s for s in samples.get(str(r.ticker), []) if start <= s.t < start + 60.0 and B._valid_sample(s)]
        mids = np.asarray([float(s.mid) for s in ss], dtype=float)
        rng = 100.0 * (float(mids.max()) - float(mids.min())) if len(mids) else np.nan
        rows.append({"ticker": str(r.ticker), "close_ts": close, "pre_range_c": rng})
    c = pd.DataFrame(rows)
    if c.empty:
        return {}, c
    w = c.groupby("close_ts", as_index=False)["pre_range_c"].max()
    return dict(zip(w.close_ts.astype(float), w.pre_range_c.astype(float))), c


def _window_tickers(audit):
    good = audit[audit["quality_ok"].astype(bool)].copy()
    out = defaultdict(list)
    series_map = {}
    for r in good.itertuples(index=False):
        out[float(r.close_ts)].append(str(r.ticker))
        series_map[str(r.ticker)] = str(r.series)
    return out, series_map


def _origin_category(dbid3, dask3, dspread3):
    if np.isfinite(dbid3) and np.isfinite(dask3):
        if dbid3 <= -0.5 and dask3 >= 0.5:
            return "BOTH_WIDEN"
        if dbid3 <= -0.5 and abs(dask3) < 0.5:
            return "BID_DOWN"
        if dask3 >= 0.5 and abs(dbid3) < 0.5:
            return "ASK_UP"
    if np.isfinite(dspread3):
        if dspread3 >= 0.5:
            return "OTHER_WIDEN"
        if dspread3 <= -0.5:
            return "CONTRACTING"
    return "STABLE"


def _build_opportunities(audit, samples, info, trades):
    good = audit[audit["quality_ok"].astype(bool)].copy()
    meta = {
        str(r.ticker): {"close_ts": float(r.close_ts), "series": str(r.series)}
        for r in good.itertuples(index=False)
    }
    win_tickers, series_map = _window_tickers(audit)
    pre_range, pre_contract = _pre_range_by_window(audit, samples)
    ages = _age_maps(samples)
    tcum = _trade_cums(trades)
    sample_maps = {t: {float(s.t): s for s in seq if B._valid_sample(s)} for t, seq in samples.items()}
    time_arrays = {t: [float(s.t) for s in seq] for t, seq in samples.items()}

    btc_by_close = {}
    for close, tickers in win_tickers.items():
        for t in tickers:
            if series_map.get(t) == "KXBTC15M":
                btc_by_close[close] = t
                break

    origin_names = ["BID_DOWN", "ASK_UP", "BOTH_WIDEN", "OTHER_WIDEN", "CONTRACTING", "STABLE"]
    rows = []

    for ticker, m in sorted(meta.items(), key=lambda kv: (kv[1]["close_ts"], kv[0])):
        close = float(m["close_ts"])
        start, end = close - 840.0, close - 600.0
        seq = sorted([s for s in samples.get(ticker, []) if start <= s.t < end and _exact4(s)], key=lambda z: z.t)
        if not seq:
            continue
        all_seq = samples.get(ticker, [])
        all_times = time_arrays.get(ticker, [])
        other_tickers = [x for x in win_tickers.get(close, []) if x != ticker]
        btc_ticker = btc_by_close.get(close)

        for s in seq:
            row = {
                "ticker": ticker,
                "series": m["series"],
                "close_ts": close,
                "close_time": _iso(close),
                "ts": float(s.t),
                "time": _iso(s.t),
                "minute": float(s.minute),
                "mid": float(s.mid),
                "bid": float(s.bid1),
                "ask": float(s.ask1),
                "bid_size": float(s.bid1_qty),
                "ask_size": float(s.ask1_qty),
                "depth_imbalance": (
                    (float(s.bid1_qty) - float(s.ask1_qty)) / (float(s.bid1_qty) + float(s.ask1_qty))
                    if float(s.bid1_qty) + float(s.ask1_qty) > EPS else 0.0
                ),
                "log_total_depth": np.log1p(max(0.0, float(s.bid1_qty) + float(s.ask1_qty))),
                "log_bid_size": np.log1p(max(0.0, float(s.bid1_qty))),
                "log_ask_size": np.log1p(max(0.0, float(s.ask1_qty))),
                "pre_m0_m1_max_mid_range_c": _f(pre_range.get(close)),
                "source_age_s": _f(info.get(ticker, {}).get(float(s.t), {}).get("source_age_ms"), 0.0) / 1000.0,
            }
            ba, aa, sa = ages.get(ticker, {}).get(float(s.t), (np.nan, np.nan, np.nan))
            row["bid_age_s"] = ba
            row["ask_age_s"] = aa
            row["spread_age_s"] = sa

            own_moves = {}
            for lag in LAGS:
                p = _past_sample(all_seq, all_times, s.t, lag)
                if p is None:
                    db = da = dm = ds = np.nan
                else:
                    db = 100.0 * (float(s.bid1) - float(p.bid1))
                    da = 100.0 * (float(s.ask1) - float(p.ask1))
                    dm = 100.0 * (float(s.mid) - float(p.mid))
                    ds = float(s.spread_c) - float(p.spread_c)
                row[f"bid_move_{lag}s_c"] = db
                row[f"ask_move_{lag}s_c"] = da
                row[f"mid_move_{lag}s_c"] = dm
                row[f"spread_move_{lag}s_c"] = ds
                row[f"abs_mid_move_{lag}s_c"] = abs(dm) if np.isfinite(dm) else np.nan
                own_moves[lag] = dm

            row["spread_origin"] = _origin_category(
                row.get("bid_move_3s_c", np.nan),
                row.get("ask_move_3s_c", np.nan),
                row.get("spread_move_3s_c", np.nan),
            )
            for name in origin_names:
                row[f"origin_{name.lower()}"] = 1.0 if row["spread_origin"] == name else 0.0

            for w in FLOW_WINDOWS:
                buy, sell, total, imb = _flow_at(tcum.get(ticker), s.t, w)
                row[f"flow_buy_qty_{w}s"] = buy
                row[f"flow_sell_qty_{w}s"] = sell
                row[f"flow_total_log_{w}s"] = np.log1p(total)
                row[f"flow_imbalance_{w}s"] = imb
            row["flow_accel_3v10"] = row["flow_imbalance_3s"] - row["flow_imbalance_10s"]
            row["flow_qty_accel_5v10"] = np.expm1(row["flow_total_log_5s"]) - 0.5 * np.expm1(row["flow_total_log_10s"])

            for lag in LAGS:
                common = []
                for ot in other_tickers:
                    cur = sample_maps.get(ot, {}).get(float(s.t))
                    if cur is None:
                        continue
                    oseq = samples.get(ot, [])
                    otimes = time_arrays.get(ot, [])
                    pp = _past_sample(oseq, otimes, s.t, lag)
                    if pp is None:
                        continue
                    common.append(100.0 * (float(cur.mid) - float(pp.mid)))
                cm = float(np.median(common)) if common else np.nan
                row[f"common_move_{lag}s_c"] = cm
                row[f"residual_move_{lag}s_c"] = own_moves.get(lag, np.nan) - cm if np.isfinite(cm) and np.isfinite(own_moves.get(lag, np.nan)) else np.nan

                bm = np.nan
                if btc_ticker is not None:
                    cur = sample_maps.get(btc_ticker, {}).get(float(s.t))
                    if cur is not None:
                        pp = _past_sample(samples.get(btc_ticker, []), time_arrays.get(btc_ticker, []), s.t, lag)
                        if pp is not None:
                            bm = 100.0 * (float(cur.mid) - float(pp.mid))
                row[f"btc_move_{lag}s_c"] = bm

            for h in MARKOUTS:
                fs = B._future_valid_sample(all_seq, all_times, float(s.t) + h, max_lag_s=2.0)
                row[f"future_mid_move_{h}s_c"] = (
                    100.0 * (float(fs.mid) - float(s.mid)) if fs is not None else np.nan
                )

            rows.append(row)

    opp = pd.DataFrame(rows)
    return opp, pre_contract


def _feature_columns(df):
    cols = [
        "minute", "depth_imbalance", "log_total_depth", "log_bid_size", "log_ask_size",
        "pre_m0_m1_max_mid_range_c", "source_age_s", "bid_age_s", "ask_age_s", "spread_age_s",
        "flow_accel_3v10", "flow_qty_accel_5v10",
    ]
    for lag in LAGS:
        cols += [
            f"bid_move_{lag}s_c", f"ask_move_{lag}s_c", f"mid_move_{lag}s_c",
            f"spread_move_{lag}s_c", f"abs_mid_move_{lag}s_c",
            f"common_move_{lag}s_c", f"btc_move_{lag}s_c", f"residual_move_{lag}s_c",
        ]
    for w in FLOW_WINDOWS:
        cols += [f"flow_total_log_{w}s", f"flow_imbalance_{w}s"]
    cols += [c for c in df.columns if c.startswith("origin_")]
    return [c for c in cols if c in df.columns]


def _fit_ridge(train, features, target):
    X = train[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    oky = np.isfinite(y)
    X, y = X[oky], y[oky]
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    Z = (X - mean) / std
    ymean = float(y.mean())
    yc = y - ymean
    gram = Z.T @ Z
    coef = np.linalg.solve(gram + RIDGE_ALPHA * np.eye(gram.shape[0]), Z.T @ yc)
    return {"features": features, "median": med, "mean": mean, "std": std, "coef": coef, "intercept": ymean}


def _predict(model, df):
    X = df[model["features"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X = np.where(np.isfinite(X), X, model["median"])
    Z = (X - model["mean"]) / model["std"]
    return model["intercept"] + Z @ model["coef"]


def _model_performance(df, target, pred_col, split):
    y = pd.to_numeric(df[target], errors="coerce")
    p = pd.to_numeric(df[pred_col], errors="coerce")
    ok = np.isfinite(y) & np.isfinite(p)
    z = pd.DataFrame({"y": y[ok], "p": p[ok]})
    if z.empty:
        return {"split": split, "rows": 0}
    err = z.p - z.y
    pear = z.corr(method="pearson").iloc[0, 1] if len(z) >= 3 else np.nan
    spear = z.corr(method="spearman").iloc[0, 1] if len(z) >= 3 else np.nan
    large = z.y.abs() >= 1.0
    return {
        "split": split,
        "rows": len(z),
        "target_mean_c": z.y.mean(),
        "pred_mean_c": z.p.mean(),
        "mae_c": err.abs().mean(),
        "rmse_c": float(np.sqrt(np.mean(err * err))),
        "pearson": pear,
        "spearman": spear,
        "directional_accuracy_pct": 100.0 * (np.sign(z.y) == np.sign(z.p)).mean(),
        "large_move_rows": int(large.sum()),
        "large_move_directional_accuracy_pct": 100.0 * (np.sign(z.loc[large, "y"]) == np.sign(z.loc[large, "p"])).mean() if large.any() else np.nan,
        "actual_bid_toxic_pct": 100.0 * (z.y < -1.0).mean(),
        "actual_ask_toxic_pct": 100.0 * (z.y > 1.0).mean(),
        "pred_bid_suppressed_pct": 100.0 * (z.p <= -1.0).mean(),
        "pred_ask_suppressed_pct": 100.0 * (z.p >= 1.0).mean(),
    }


def _feature_screen(train, features):
    rows = []
    for f in features:
        x = pd.to_numeric(train[f], errors="coerce")
        for h in MARKOUTS:
            y = pd.to_numeric(train[f"future_mid_move_{h}s_c"], errors="coerce")
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 30:
                continue
            z = pd.DataFrame({"x": x[ok], "y": y[ok]})
            rows.append({
                "feature": f,
                "horizon_s": h,
                "n": len(z),
                "pearson": z.corr(method="pearson").iloc[0, 1],
                "spearman": z.corr(method="spearman").iloc[0, 1],
                "mean_abs_feature": z.x.abs().mean(),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["abs_spearman"] = out.spearman.abs()
        out = out.sort_values(["horizon_s", "abs_spearman"], ascending=[True, False])
    return out


def _origin_summary(opp, split_col="split"):
    rows = []
    for (split, origin), z in opp.groupby([split_col, "spread_origin"], sort=True):
        y30 = pd.to_numeric(z["future_mid_move_30s_c"], errors="coerce")
        y5 = pd.to_numeric(z["future_mid_move_5s_c"], errors="coerce")
        rows.append({
            "split": split, "spread_origin": origin, "rows": len(z),
            "mean_future_move_5s_c": y5.mean(), "mean_future_move_30s_c": y30.mean(),
            "mean_abs_future_move_30s_c": y30.abs().mean(),
            "bid_toxic_pct": 100.0 * (y30 < -1.0).mean(),
            "ask_toxic_pct": 100.0 * (y30 > 1.0).mean(),
        })
    return pd.DataFrame(rows)


def _allowed_map(opp_test, pred_col, oracle=False):
    allowed = {}
    for r in opp_test.itertuples(index=False):
        move = _f(getattr(r, "future_mid_move_30s_c")) if oracle else _f(getattr(r, pred_col))
        if not np.isfinite(move):
            continue
        key = (str(r.ticker), float(r.ts))
        allowed[(key[0], key[1], "BID")] = bool(1.0 + move > 0.0 + EPS)
        allowed[(key[0], key[1], "ASK")] = bool(1.0 - move > 0.0 + EPS)
    return allowed


def _simulate_policy(name, targets, sim_meta, samples, trades, allowed=None):
    original = INV._base_reasons
    holder = {"ticker": None}

    if allowed is not None:
        def gated(side, s, last_fill_ts, now_t, mom, flow):
            reasons = list(original(side, s, last_fill_ts, now_t, mom, flow))
            if not allowed.get((holder["ticker"], float(s.t), side), False):
                reasons.append("TOXICITY_MODEL")
            return reasons
        INV._base_reasons = gated

    episodes, fills, contracts, counts = [], [], [], Counter()
    try:
        for i, ticker in enumerate(targets, 1):
            holder["ticker"] = ticker
            e, f, c, k = INV._simulate_contract(
                ticker, sim_meta[ticker], samples.get(ticker, []), trades.get(ticker, [])
            )
            for x in e:
                x["policy_name"] = name
            for x in f:
                x["policy_name"] = name
            if c is not None:
                c["policy_name"] = name
                contracts.append(c)
            episodes.extend(e); fills.extend(f); counts.update(k)
            if i % 100 == 0 or i == len(targets):
                print(f"  {name}: {i}/{len(targets)} | fills={len(fills)} | qty={sum(float(x['qty']) for x in fills):.2f}")
    finally:
        INV._base_reasons = original

    return pd.DataFrame(contracts), pd.DataFrame(fills), pd.DataFrame(episodes), pd.DataFrame(
        [{"policy_name": name, "reason": k, "count": v} for k, v in counts.most_common()]
    )


def _policy_summary(name, cdf, fdf, wdf):
    qty = pd.to_numeric(fdf.get("qty", pd.Series(dtype=float)), errors="coerce").sum() if len(fdf) else 0.0
    pnl = pd.to_numeric(cdf.get("net_mtm_pnl_before_fees", pd.Series(dtype=float)), errors="coerce").sum() if len(cdf) else 0.0
    matched = pd.to_numeric(cdf.get("matched_roundtrip_pnl", pd.Series(dtype=float)), errors="coerce").sum() if len(cdf) else 0.0
    gross = pd.to_numeric(cdf.get("gross_spread_capture_dollars", pd.Series(dtype=float)), errors="coerce").sum() if len(cdf) else 0.0
    adv = pd.to_numeric(cdf.get("adverse_selection_to_m5_dollars", pd.Series(dtype=float)), errors="coerce").sum() if len(cdf) else 0.0
    r = {
        "policy": name, "windows": len(wdf), "contracts": len(cdf), "fill_events": len(fdf), "fill_qty": qty,
        "net_pnl": pnl, "pnl_per_window": pnl / len(wdf) if len(wdf) else np.nan,
        "gross_capture": gross, "adverse_selection_to_m5": adv,
        "matched_roundtrip_pnl": matched, "residual_inventory_mtm_pnl": pnl - matched,
        "worst_window": pd.to_numeric(wdf.get("net_mtm_pnl_before_fees", pd.Series(dtype=float)), errors="coerce").min() if len(wdf) else np.nan,
        "max_drawdown": pd.to_numeric(wdf.get("drawdown", pd.Series(dtype=float)), errors="coerce").min() if len(wdf) else np.nan,
    }
    for h in MARKOUTS:
        col = f"markout_{h}s_c"
        if len(fdf) and col in fdf.columns:
            x = pd.to_numeric(fdf[col], errors="coerce")
            w = pd.to_numeric(fdf["qty"], errors="coerce")
            ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
            r[f"qw_markout_{h}s_c"] = float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan
        else:
            r[f"qw_markout_{h}s_c"] = np.nan
    return r


def _portfolio_inventory(policy, fdf, close_map, wdf):
    if fdf.empty:
        return pd.DataFrame()
    x = fdf.copy()
    x["close_ts"] = x["ticker"].map(close_map)
    x["signed_qty"] = np.where(x["side"].astype(str) == "BID", x["qty"].astype(float), -x["qty"].astype(float))
    rows = []
    pnl_map = dict(zip(wdf.close_ts.astype(float), wdf.net_mtm_pnl_before_fees.astype(float))) if len(wdf) else {}
    for close, z in x.groupby("close_ts", sort=True):
        z = z.sort_values("fill_ts")
        inv = z.signed_qty.cumsum()
        rows.append({
            "policy": policy, "close_ts": float(close), "close_time": _iso(close),
            "fill_qty": float(z.qty.sum()), "max_abs_portfolio_inventory": float(inv.abs().max()),
            "ending_portfolio_inventory": float(inv.iloc[-1]), "window_pnl": _f(pnl_map.get(float(close))),
        })
    return pd.DataFrame(rows)


def _fill_score_buckets(fdf, edf, opp, pred_col):
    if fdf.empty or edf.empty:
        return pd.DataFrame()
    ep = edf[["episode_id", "ticker", "join_ts"]].copy()
    p = opp[["ticker", "ts", pred_col, "future_mid_move_30s_c"]].copy()
    p = p.rename(columns={"ts": "join_ts"})
    ep = ep.merge(p, on=["ticker", "join_ts"], how="left")
    x = fdf.merge(ep[["episode_id", pred_col, "future_mid_move_30s_c"]], on="episode_id", how="left")
    sign = np.where(x.side.astype(str) == "BID", 1.0, -1.0)
    x["predicted_edge_30s_c"] = 1.0 + sign * pd.to_numeric(x[pred_col], errors="coerce")
    x["actual_edge_30s_proxy_c"] = 1.0 + sign * pd.to_numeric(x["future_mid_move_30s_c"], errors="coerce")
    good = np.isfinite(x.predicted_edge_30s_c)
    x = x[good].copy()
    if len(x) < 10:
        return pd.DataFrame()
    try:
        x["score_bucket"] = pd.qcut(x.predicted_edge_30s_c, 5, duplicates="drop")
    except Exception:
        x["score_bucket"] = "ALL"
    rows = []
    for b, z in x.groupby("score_bucket", observed=True, sort=True):
        w = pd.to_numeric(z.qty, errors="coerce")
        a = pd.to_numeric(z.actual_edge_30s_proxy_c, errors="coerce")
        ok = np.isfinite(w) & np.isfinite(a) & (w > 0)
        rows.append({
            "score_bucket": str(b), "fill_events": len(z), "fill_qty": w.sum(),
            "predicted_edge_mean_c": pd.to_numeric(z.predicted_edge_30s_c, errors="coerce").mean(),
            "actual_edge_30s_qw_c": float(np.average(a[ok], weights=w[ok])) if ok.any() else np.nan,
            "toxic_fill_pct": 100.0 * (a < 0).mean(),
        })
    return pd.DataFrame(rows)


def _print_report(audit, split_windows, perf, coeff, origin, policy, portfolio, score_buckets, out):
    print("\n" + "=" * 150)
    print("STATE-AWARE / TOXICITY-AWARE MM — DEVELOPMENT SUITE")
    print("=" * 150)
    print(f"session={EXPECTED_SESSION_NAME} (DEVELOPMENT ONLY) | reserved validation={RESERVED_VALIDATION_SESSION_NAME} NOT READ")
    print(f"M1-M5 quality contracts={int(audit.quality_ok.sum())}/{len(audit)} | windows={audit.loc[audit.quality_ok, 'close_ts'].nunique()}")
    print(f"chronological split: train windows={len(split_windows['train'])} | internal test windows={len(split_windows['test'])}")
    print("\nMODEL PERFORMANCE — predicts raw 30s future midpoint move")
    print(perf.round(4).to_string(index=False))
    print("\nTOP STANDARDIZED RIDGE COEFFICIENTS")
    print(coeff.head(20).round(4).to_string(index=False))
    print("\nSPREAD-ORIGIN DIAGNOSTIC")
    print(origin.round(4).to_string(index=False))
    print("\nINTERNAL HOLDOUT POLICY REPLAY")
    print(policy.round(4).to_string(index=False))
    if len(score_buckets):
        print("\nBASELINE FILL TOXICITY BY MODEL SCORE")
        print(score_buckets.round(4).to_string(index=False))
    if len(portfolio):
        ps = portfolio.groupby("policy", as_index=False).agg(
            windows=("close_ts", "count"),
            median_max_abs_portfolio_inventory=("max_abs_portfolio_inventory", "median"),
            p95_max_abs_portfolio_inventory=("max_abs_portfolio_inventory", lambda x: x.quantile(.95)),
            median_abs_ending_portfolio_inventory=("ending_portfolio_inventory", lambda x: x.abs().median()),
        )
        print("\nPORTFOLIO-INVENTORY DIAGNOSTIC")
        print(ps.round(3).to_string(index=False))
    print("\nLIMITATIONS")
    print("  - BBO is 1 Hz: true sub-second/event-time BBO awareness is NOT testable here.")
    print("  - Q100 inside-spread flow remains counterfactual; historical flow is treated as exogenous.")
    print("  - Price skew is represented by predicted fair-value shift + toxic-side suppression; continuously re-priced skew is not replayable from this archive.")
    print("  - Actual live maker / market-impact validation is NOT testable on historical data.")
    print("  - No threshold sweep: economic side gate is fixed at predicted expected edge > 0.")
    print("Outputs:", out)
    print("=" * 150)


def run_state_aware_toxicity_research(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"Development suite is hard-bound to {EXPECTED_SESSION_NAME}; got {session.name}. "
            f"Reserved validation session {RESERVED_VALIDATION_SESSION_NAME} must not be read."
        )
    if not session.exists():
        raise FileNotFoundError(session)

    meta = O._metadata(session)
    samples, info, duplicates, bbo_stats = O._bbo(session, meta)
    trades, trade_stats = O._trades(session, meta)
    audit = O._audit(meta, samples, info, duplicates, trades)
    if audit.empty:
        raise RuntimeError("No compact-session contracts found")
    good = audit[audit.quality_ok].copy()
    if good.empty:
        raise RuntimeError("No contracts pass the unchanged 80% M1-M5 quality gate")

    print(f"Quality pass: {len(good)}/{len(audit)} contracts | {good.close_ts.nunique()} windows")
    print("Building exact-4c opportunity/state table...")
    opp, pre_contract = _build_opportunities(audit, samples, info, trades)
    if opp.empty:
        raise RuntimeError("No exact-4c NAT4->2 opportunities found")

    windows = np.asarray(sorted(opp.close_ts.unique()), dtype=float)
    cut = max(1, min(len(windows) - 1, int(np.floor(len(windows) * TRAIN_WINDOW_FRACTION))))
    train_windows = set(windows[:cut])
    test_windows = set(windows[cut:])
    opp["split"] = np.where(opp.close_ts.isin(train_windows), "TRAIN", "INTERNAL_TEST")

    features = _feature_columns(opp)
    target = f"future_mid_move_{TARGET_HORIZON_S}s_c"
    train = opp[(opp.split == "TRAIN") & np.isfinite(pd.to_numeric(opp[target], errors="coerce"))].copy()
    test = opp[(opp.split == "INTERNAL_TEST") & np.isfinite(pd.to_numeric(opp[target], errors="coerce"))].copy()
    if len(train) < 1000 or len(test) < 500:
        raise RuntimeError(f"Not enough finite opportunities: train={len(train)} test={len(test)}")

    model = _fit_ridge(train, features, target)
    opp["pred_future_mid_move_30s_c"] = _predict(model, opp)
    opp["pred_bid_edge_30s_c"] = 1.0 + opp["pred_future_mid_move_30s_c"]
    opp["pred_ask_edge_30s_c"] = 1.0 - opp["pred_future_mid_move_30s_c"]
    opp["recommended_quote_state"] = np.select(
        [opp.pred_future_mid_move_30s_c >= 1.0, opp.pred_future_mid_move_30s_c <= -1.0],
        ["BID_ONLY", "ASK_ONLY"],
        default="TWO_SIDED",
    )
    opp["implied_reservation_shift_c"] = opp["pred_future_mid_move_30s_c"]

    perf = pd.DataFrame([
        _model_performance(opp[opp.split == "TRAIN"], target, "pred_future_mid_move_30s_c", "TRAIN"),
        _model_performance(opp[opp.split == "INTERNAL_TEST"], target, "pred_future_mid_move_30s_c", "INTERNAL_TEST"),
    ])
    coeff = pd.DataFrame({"feature": model["features"], "standardized_coef": model["coef"]})
    coeff["abs_coef"] = coeff.standardized_coef.abs()
    coeff = coeff.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    feature_screen = _feature_screen(train, features)
    origin = _origin_summary(opp)

    test_good = good[good.close_ts.astype(float).isin(test_windows)].copy()
    sim_meta = {
        str(r.ticker): {"ticker": str(r.ticker), "series": str(r.series), "close_ts": float(r.close_ts)}
        for r in test_good.itertuples(index=False)
    }
    targets = sorted(sim_meta, key=lambda t: (sim_meta[t]["close_ts"], t))
    print(f"Internal test replay: {len(test_windows)} windows | {len(targets)} quality contracts")

    baseline_c, baseline_f, baseline_e, baseline_counts = _simulate_policy(
        "BASELINE_NAT4_TO_2", targets, sim_meta, samples, trades, allowed=None
    )
    allowed = _allowed_map(opp[opp.split == "INTERNAL_TEST"], "pred_future_mid_move_30s_c", oracle=False)
    aware_c, aware_f, aware_e, aware_counts = _simulate_policy(
        "TOXICITY_AWARE_EDGE_GT0", targets, sim_meta, samples, trades, allowed=allowed
    )
    oracle_allowed = _allowed_map(opp[opp.split == "INTERNAL_TEST"], "pred_future_mid_move_30s_c", oracle=True)
    oracle_c, oracle_f, oracle_e, oracle_counts = _simulate_policy(
        "ORACLE_30S_LOOKAHEAD_CEILING", targets, sim_meta, samples, trades, allowed=oracle_allowed
    )

    policy_rows, windows_all, portfolio_all = [], [], []
    close_map = {t: float(m["close_ts"]) for t, m in sim_meta.items()}
    for name, cdf, fdf, edf in [
        ("BASELINE_NAT4_TO_2", baseline_c, baseline_f, baseline_e),
        ("TOXICITY_AWARE_EDGE_GT0", aware_c, aware_f, aware_e),
        ("ORACLE_30S_LOOKAHEAD_CEILING", oracle_c, oracle_f, oracle_e),
    ]:
        wdf = L._window_summary(cdf) if len(cdf) else pd.DataFrame()
        if len(wdf):
            wdf["policy"] = name
            windows_all.append(wdf)
        policy_rows.append(_policy_summary(name, cdf, fdf, wdf))
        p = _portfolio_inventory(name, fdf, close_map, wdf)
        if len(p):
            portfolio_all.append(p)

    policy = pd.DataFrame(policy_rows)
    policy_windows = pd.concat(windows_all, ignore_index=True) if windows_all else pd.DataFrame()
    portfolio = pd.concat(portfolio_all, ignore_index=True) if portfolio_all else pd.DataFrame()
    score_buckets = _fill_score_buckets(
        baseline_f, baseline_e, opp[opp.split == "INTERNAL_TEST"], "pred_future_mid_move_30s_c"
    )

    if output_dir is None:
        output_dir = (
            R.PROJECT_ROOT / "results" / "kalshi_state_aware_toxicity_mm_dev"
            / f"{session.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, df in {
        "contract_quality.csv": audit,
        "pre_m1_contract_range.csv": pre_contract,
        "opportunity_state_features.csv": opp,
        "feature_screen_train.csv": feature_screen,
        "model_coefficients.csv": coeff,
        "model_performance.csv": perf,
        "spread_origin_summary.csv": origin,
        "policy_summary_internal_test.csv": policy,
        "policy_windows_internal_test.csv": policy_windows,
        "baseline_fills_internal_test.csv": baseline_f,
        "aware_fills_internal_test.csv": aware_f,
        "oracle_fills_internal_test.csv": oracle_f,
        "baseline_quote_episodes_internal_test.csv": baseline_e,
        "aware_quote_episodes_internal_test.csv": aware_e,
        "fill_score_buckets.csv": score_buckets,
        "portfolio_inventory_diagnostics.csv": portfolio,
        "policy_counts.csv": pd.concat([baseline_counts, aware_counts, oracle_counts], ignore_index=True),
    }.items():
        df.to_csv(out / name, index=False)

    config = {
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "hard_bound_session_name": EXPECTED_SESSION_NAME,
        "reserved_validation_session_name": RESERVED_VALIDATION_SESSION_NAME,
        "reserved_validation_accessed": False,
        "m1_m5_quality_gate_pct": 80.0,
        "opportunity": "natural spread exactly 4c (+/-0.05c), NAT4->2 inside-spread mechanism",
        "model_target": "raw future 30s midpoint move in cents",
        "model": "standardized linear ridge",
        "ridge_alpha": RIDGE_ALPHA,
        "train_window_fraction": TRAIN_WINDOW_FRACTION,
        "split_unit": "15-minute close_ts window, chronological",
        "economic_gate": "BID iff 1c + predicted_move > 0; ASK iff 1c - predicted_move > 0",
        "threshold_sweep": False,
        "asset_one_hot_features": False,
        "features": features,
        "limitations": [
            "1Hz BBO only; no sub-second event-time BBO test",
            "Q100 inside-spread historical flow is counterfactual/exogenous",
            "continuous price-skew execution not replayable; side suppression is used as proxy",
            "live maker / market impact cannot be tested on historical archive",
        ],
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    if show:
        _print_report(
            audit,
            {"train": train_windows, "test": test_windows},
            perf, coeff, origin, policy, portfolio, score_buckets, out,
        )

    return {
        "output_dir": out,
        "opportunities": opp,
        "feature_screen": feature_screen,
        "model_coefficients": coeff,
        "model_performance": perf,
        "spread_origin": origin,
        "policy_summary": policy,
        "policy_windows": policy_windows,
        "portfolio_inventory": portfolio,
        "fill_score_buckets": score_buckets,
        "baseline_fills": baseline_f,
        "aware_fills": aware_f,
        "oracle_fills": oracle_f,
    }
