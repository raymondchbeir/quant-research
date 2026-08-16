from __future__ import annotations

"""Event-time passive-fill toxicity diagnostic for V4 M0-M5 data.

DEVELOPMENT DIAGNOSTIC ONLY -- no strategy replay and no PnL optimization.

Scientific scope
----------------
- Uses only contracts that passed the prior V4 full-boundary audit.
- Excludes KXBTC15M as a quoted/passive-fill sample because the V4 forensic
  audit found all locked/crossed reconstructed-book states were isolated to BTC.
- BTC ticker data remains allowed as an exogenous causal factor input because
  it is an independent stream from the corrupted reconstructed BTC depth.
- Uses real public aggressive trades as passive-fill events at the public book.
- Compares pre-trade event-time microstructure over 100/250/500/1000 ms with
  signed post-fill midpoint movement at 5/15/30 seconds.
- No policy threshold, quote-size rule, asset ranking, or PnL is evaluated.

The purpose is to answer one question before building another maker strategy:
    Do event-time depletion/replenishment/trade-burst features distinguish
    favorable passive fills from toxic passive fills in a chronologically
    stable way?
"""

import bisect
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C

STUDY_VERSION = "MM_EVENT_TIME_FILL_TOXICITY_DIAGNOSTIC_V1"
EXPECTED_SESSION_NAME = "20260815_043130"
BTC_SERIES = "KXBTC15M"
WINDOWS_S = (0.10, 0.25, 0.50, 1.00)
HORIZONS_S = (5.0, 15.0, 30.0)
MAX_PREHISTORY_S = 1.05
EPS = 1e-12


def _ts(x):
    if x is None:
        return np.nan
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except Exception:
        return np.nan


def _f(x):
    try:
        z = float(x)
        return z if np.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _latest_at_or_before(times: np.ndarray, vals: np.ndarray, t: float):
    if len(times) == 0 or not np.isfinite(t):
        return np.nan
    j = int(np.searchsorted(times, float(t), side="right") - 1)
    if j < 0:
        return np.nan
    return float(vals[j])


def _top_fields(row):
    bids = row.get("bid_levels") or []
    asks = row.get("ask_levels") or []
    if not row.get("valid_bbo") or not bids or not asks:
        return None
    try:
        bid = float(row["yes_bid"]); ask = float(row["yes_ask"])
        bq = float(row["yes_bid_size"]); aq = float(row["yes_ask_size"])
    except Exception:
        return None
    if not (0.0 <= bid < ask <= 1.0):
        return None
    bdepth3 = float(sum(float(x[1]) for x in bids[:3]))
    adepth3 = float(sum(float(x[1]) for x in asks[:3]))
    return {
        "bid": bid, "ask": ask, "mid": 0.5 * (bid + ask),
        "spread_c": 100.0 * (ask - bid),
        "bid_q1": max(0.0, bq), "ask_q1": max(0.0, aq),
        "bid_depth3": max(0.0, bdepth3), "ask_depth3": max(0.0, adepth3),
    }


def _past_state(hist, target_t):
    # hist is tiny (<=1.05s of top3-changing events) for non-BTC series.
    for e in reversed(hist):
        if e["t"] <= target_t + EPS:
            return e
    return None


def _book_features(hist, trade_t, last_bid_change, last_ask_change):
    if not hist:
        return None
    cur = hist[-1]
    if cur["t"] > trade_t + EPS:
        return None
    out = {
        "book_age_ms": 1000.0 * max(0.0, trade_t - cur["t"]),
        "bid": cur["bid"], "ask": cur["ask"], "mid_pre": cur["mid"],
        "spread_c": cur["spread_c"],
        "bid_q1": cur["bid_q1"], "ask_q1": cur["ask_q1"],
        "bid_depth3": cur["bid_depth3"], "ask_depth3": cur["ask_depth3"],
        "l1_depth_imbalance": (
            (cur["bid_q1"] - cur["ask_q1"]) / (cur["bid_q1"] + cur["ask_q1"])
            if cur["bid_q1"] + cur["ask_q1"] > EPS else 0.0
        ),
        "l3_depth_imbalance": (
            (cur["bid_depth3"] - cur["ask_depth3"]) / (cur["bid_depth3"] + cur["ask_depth3"])
            if cur["bid_depth3"] + cur["ask_depth3"] > EPS else 0.0
        ),
        "bid_age_ms": 1000.0 * max(0.0, trade_t - last_bid_change) if np.isfinite(last_bid_change) else np.nan,
        "ask_age_ms": 1000.0 * max(0.0, trade_t - last_ask_change) if np.isfinite(last_ask_change) else np.nan,
    }

    for w in WINDOWS_S:
        tag = f"{int(round(w * 1000))}ms"
        past = _past_state(hist, trade_t - w)
        if past is None:
            for name in (
                "mid_move_c", "bid_move_c", "ask_move_c", "spread_move_c",
                "bid_q1_change", "ask_q1_change", "bid_depth3_change", "ask_depth3_change",
            ):
                out[f"{name}_{tag}"] = np.nan
        else:
            out[f"mid_move_c_{tag}"] = 100.0 * (cur["mid"] - past["mid"])
            out[f"bid_move_c_{tag}"] = 100.0 * (cur["bid"] - past["bid"])
            out[f"ask_move_c_{tag}"] = 100.0 * (cur["ask"] - past["ask"])
            out[f"spread_move_c_{tag}"] = cur["spread_c"] - past["spread_c"]
            out[f"bid_q1_change_{tag}"] = cur["bid_q1"] - past["bid_q1"]
            out[f"ask_q1_change_{tag}"] = cur["ask_q1"] - past["ask_q1"]
            out[f"bid_depth3_change_{tag}"] = cur["bid_depth3"] - past["bid_depth3"]
            out[f"ask_depth3_change_{tag}"] = cur["ask_depth3"] - past["ask_depth3"]

        ev = [e for e in hist if e["t"] >= trade_t - w - EPS and e["t"] < trade_t + EPS]
        out[f"book_event_count_{tag}"] = float(len(ev))
        out[f"bid_top_change_count_{tag}"] = float(sum(e["bid_changed"] for e in ev))
        out[f"ask_top_change_count_{tag}"] = float(sum(e["ask_changed"] for e in ev))
        out[f"bid_add_qty_{tag}"] = float(sum(max(0.0, e["delta_qty"]) for e in ev if e["delta_side"] == "yes"))
        out[f"bid_remove_qty_{tag}"] = float(sum(max(0.0, -e["delta_qty"]) for e in ev if e["delta_side"] == "yes"))
        out[f"ask_add_qty_{tag}"] = float(sum(max(0.0, e["delta_qty"]) for e in ev if e["delta_side"] == "no"))
        out[f"ask_remove_qty_{tag}"] = float(sum(max(0.0, -e["delta_qty"]) for e in ev if e["delta_side"] == "no"))
    return out


def _trade_flow_features(target_df, all_trades):
    if target_df.empty:
        return target_df
    out = target_df.copy()
    for w in WINDOWS_S:
        tag = f"{int(round(w * 1000))}ms"
        for c in ("aggr_buy_qty", "aggr_sell_qty", "aggr_total_qty", "aggr_imbalance", "trade_count"):
            out[f"{c}_{tag}"] = 0.0

    by_ticker = {}
    for ticker, z in all_trades.groupby("ticker", sort=False):
        z = z.sort_values("receipt_ts")
        tt = z.receipt_ts.to_numpy(float)
        qty = z.qty.to_numpy(float)
        side = z.taker_book_side.astype(str).to_numpy()
        buy = np.where(side == "bid", qty, 0.0)
        sell = np.where(side == "ask", qty, 0.0)
        by_ticker[ticker] = (tt, np.r_[0.0, np.cumsum(buy)], np.r_[0.0, np.cumsum(sell)])

    for idx, r in out.iterrows():
        pack = by_ticker.get(r.ticker)
        if pack is None:
            continue
        tt, cb, cs = pack
        hi = int(np.searchsorted(tt, float(r.receipt_ts), side="left"))  # strictly before current trade
        for w in WINDOWS_S:
            tag = f"{int(round(w * 1000))}ms"
            lo = int(np.searchsorted(tt, float(r.receipt_ts) - w, side="left"))
            b = float(cb[hi] - cb[lo]); s = float(cs[hi] - cs[lo]); tot = b + s
            out.at[idx, f"aggr_buy_qty_{tag}"] = b
            out.at[idx, f"aggr_sell_qty_{tag}"] = s
            out.at[idx, f"aggr_total_qty_{tag}"] = tot
            out.at[idx, f"aggr_imbalance_{tag}"] = (b - s) / tot if tot > EPS else 0.0
            out.at[idx, f"trade_count_{tag}"] = float(hi - lo)
    return out


def _spearman(x, y):
    z = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(z) < 30 or z.x.nunique() < 2 or z.y.nunique() < 2:
        return np.nan
    return float(z.corr(method="spearman").iloc[0, 1])


def _feature_summary(df, features):
    rows = []
    if df.empty:
        return pd.DataFrame()
    closes = np.asarray(sorted(df.close_ts.dropna().unique()), dtype=float)
    cut = closes[len(closes) // 2] if len(closes) else np.nan
    parts = {
        "FULL": df,
        "EARLY_HALF": df[df.close_ts < cut] if np.isfinite(cut) else df.iloc[0:0],
        "LATE_HALF": df[df.close_ts >= cut] if np.isfinite(cut) else df.iloc[0:0],
    }
    for side in ("BID", "ASK"):
        for split, z0 in parts.items():
            z = z0[z0.passive_side == side]
            y = pd.to_numeric(z.post_mid_move_30s_c, errors="coerce")
            for f in features:
                x = pd.to_numeric(z[f], errors="coerce")
                ok = np.isfinite(x) & np.isfinite(y)
                zz = pd.DataFrame({"x": x[ok], "y": y[ok]})
                if len(zz) < 30 or zz.x.nunique() < 2:
                    continue
                good = zz[zz.y > 0]; toxic = zz[zz.y < 0]
                pooled = float(np.sqrt(0.5 * (good.x.var(ddof=1) + toxic.x.var(ddof=1)))) if len(good) > 1 and len(toxic) > 1 else np.nan
                smd = (good.x.mean() - toxic.x.mean()) / pooled if np.isfinite(pooled) and pooled > EPS else np.nan
                rows.append({
                    "side": side, "split": split, "feature": f, "n": len(zz),
                    "spearman_vs_30s_post_move": _spearman(zz.x, zz.y),
                    "good_n": len(good), "toxic_n": len(toxic),
                    "good_median": good.x.median() if len(good) else np.nan,
                    "toxic_median": toxic.x.median() if len(toxic) else np.nan,
                    "good_minus_toxic_smd": smd,
                })
    return pd.DataFrame(rows)


def _stable_ranking(summary):
    rows = []
    if summary.empty:
        return pd.DataFrame()
    for (side, feature), z in summary.groupby(["side", "feature"], sort=False):
        vals = {r.split: _f(r.spearman_vs_30s_post_move) for r in z.itertuples(index=False)}
        full, early, late = vals.get("FULL", np.nan), vals.get("EARLY_HALF", np.nan), vals.get("LATE_HALF", np.nan)
        finite = all(np.isfinite(v) for v in (full, early, late))
        same = finite and np.sign(full) == np.sign(early) == np.sign(late) and abs(full) > 0
        rows.append({
            "side": side, "feature": feature,
            "full_spearman": full, "early_spearman": early, "late_spearman": late,
            "same_sign_both_halves": bool(same),
            "min_abs_half_spearman": min(abs(early), abs(late)) if finite else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["same_sign_both_halves", "min_abs_half_spearman"], ascending=[False, False])
    return out


def _quantile_shapes(df, ranking, max_features_per_side=8):
    rows = []
    if df.empty or ranking.empty:
        return pd.DataFrame()
    for side in ("BID", "ASK"):
        chosen = ranking[(ranking.side == side) & ranking.same_sign_both_halves].head(max_features_per_side)
        zside = df[df.passive_side == side].copy()
        for f in chosen.feature:
            x = pd.to_numeric(zside[f], errors="coerce")
            y = pd.to_numeric(zside.post_mid_move_30s_c, errors="coerce")
            ok = np.isfinite(x) & np.isfinite(y)
            z = pd.DataFrame({"x": x[ok], "y": y[ok], "close_ts": zside.loc[ok, "close_ts"]})
            if len(z) < 100 or z.x.nunique() < 5:
                continue
            try:
                z["bucket"] = pd.qcut(z.x, 5, duplicates="drop")
            except Exception:
                continue
            for rank, (bucket, b) in enumerate(z.groupby("bucket", observed=True, sort=True), 1):
                # Window-balanced target: average within window, then across windows.
                wb = b.groupby("close_ts").y.mean()
                rows.append({
                    "side": side, "feature": f, "bucket_rank": rank, "bucket": str(bucket),
                    "fills": len(b), "windows": wb.size,
                    "feature_median": b.x.median(),
                    "fill_mean_30s_post_move_c": b.y.mean(),
                    "window_balanced_mean_30s_post_move_c": wb.mean(),
                    "positive_fill_pct": 100.0 * (b.y > 0).mean(),
                })
    return pd.DataFrame(rows)


def run_event_time_fill_toxicity_diagnostic(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(f"V1 diagnostic is hard-bound to development session {EXPECTED_SESSION_NAME}; got {session.name}")

    project_root = C.PROJECT_ROOT
    audit_csv = project_root / "results" / "kalshi_mm_event_m0_m5_v4_audit" / session.name / "contract_event_time_quality.csv"
    if not audit_csv.exists():
        raise FileNotFoundError(f"Run mm_event_time_m0_m5_audit_v1 first; missing {audit_csv}")
    quality = pd.read_csv(audit_csv)
    eligible = quality[(quality.full_boundary_capture.astype(bool)) & (quality.series.astype(str) != BTC_SERIES)].copy()
    eligible_tickers = set(eligible.ticker.astype(str))
    if not eligible_tickers:
        raise RuntimeError("No full-boundary non-BTC contracts available")

    # Metadata gives the BTC contract corresponding to each close window.
    meta = []
    with (session / "market_metadata.jsonl").open() as fh:
        for line in fh:
            try: x = json.loads(line)
            except Exception: continue
            meta.append(x)
    mdf = pd.DataFrame(meta).drop_duplicates("ticker", keep="first") if meta else pd.DataFrame()
    btc_by_close = {}
    if len(mdf):
        for r in mdf[mdf.series_ticker.astype(str) == BTC_SERIES].itertuples(index=False):
            btc_by_close[str(r.close_time)] = str(r.ticker)

    print(f"Eligible passive-fill contracts: {len(eligible_tickers)} across {eligible.series.nunique()} non-BTC series")
    print("Loading event-time trades...")
    trade_rows = []
    with (session / "trades_event_time.jsonl").open() as fh:
        for n, line in enumerate(fh, 1):
            try: x = json.loads(line)
            except Exception: continue
            ticker = str(x.get("ticker") or "")
            series = str(x.get("series_ticker") or "")
            if ticker not in eligible_tickers and series != BTC_SERIES:
                continue
            rt = _ts(x.get("receipt_time")); qty = _f(x.get("qty")); px = _f(x.get("yes_price")); e = _f(x.get("elapsed_s"))
            side = str(x.get("taker_book_side") or "").lower()
            if not np.isfinite(rt) or not np.isfinite(qty) or qty <= 0 or not np.isfinite(px) or side not in {"bid", "ask"}:
                continue
            trade_rows.append({
                "trade_id": x.get("trade_id"), "ticker": ticker, "series": series,
                "close_time": str(x.get("close_time")), "receipt_ts": rt,
                "elapsed_s": e, "qty": qty, "yes_price": px, "taker_book_side": side,
            })
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        raise RuntimeError("No eligible trades")
    trades["close_ts"] = trades.close_time.map(_ts)

    # Ticker stream supplies future midpoint labels and BTC factor moves.
    print("Loading ticker state stream...")
    tick_rows = defaultdict(list)
    with (session / "ticker_event_time.jsonl").open() as fh:
        for line in fh:
            try: x = json.loads(line)
            except Exception: continue
            ticker = str(x.get("ticker") or "")
            series = str(x.get("series_ticker") or "")
            if ticker not in eligible_tickers and series != BTC_SERIES:
                continue
            t = _ts(x.get("receipt_time")); b = _f(x.get("yes_bid")); a = _f(x.get("yes_ask"))
            if np.isfinite(t) and np.isfinite(b) and np.isfinite(a) and 0 <= b < a <= 1:
                tick_rows[ticker].append((t, 0.5 * (b + a)))
    ticker_arrays = {}
    for ticker, seq in tick_rows.items():
        seq.sort()
        ticker_arrays[ticker] = (
            np.asarray([x[0] for x in seq], dtype=float),
            np.asarray([x[1] for x in seq], dtype=float),
        )

    targets = trades[(trades.ticker.isin(eligible_tickers)) & (trades.elapsed_s >= 1.0) & (trades.elapsed_s <= 270.0)].copy()
    targets = targets.sort_values(["ticker", "receipt_ts"]).reset_index(drop=True)
    targets["passive_side"] = np.where(targets.taker_book_side == "ask", "BID", "ASK")

    # Future labels from same-contract ticker state; carry-forward is causal market state.
    for h in HORIZONS_S:
        targets[f"future_mid_{int(h)}s"] = np.nan
    targets["btc_move_c_100ms"] = np.nan
    targets["btc_move_c_250ms"] = np.nan
    targets["btc_move_c_500ms"] = np.nan
    targets["btc_move_c_1000ms"] = np.nan
    for i, r in targets.iterrows():
        arr = ticker_arrays.get(r.ticker)
        if arr is not None:
            tt, mm = arr
            for h in HORIZONS_S:
                targets.at[i, f"future_mid_{int(h)}s"] = _latest_at_or_before(tt, mm, r.receipt_ts + h)
        btc_ticker = btc_by_close.get(str(r.close_time))
        barr = ticker_arrays.get(btc_ticker) if btc_ticker else None
        if barr is not None:
            bt, bm = barr
            cur = _latest_at_or_before(bt, bm, r.receipt_ts)
            for w in WINDOWS_S:
                past = _latest_at_or_before(bt, bm, r.receipt_ts - w)
                targets.at[i, f"btc_move_c_{int(round(w*1000))}ms"] = 100.0 * (cur - past) if np.isfinite(cur) and np.isfinite(past) else np.nan

    # Index target rows by ticker; book stream remains on disk.
    target_indices = defaultdict(list)
    for idx, r in targets.iterrows():
        target_indices[str(r.ticker)].append((float(r.receipt_ts), int(idx)))
    ptr = {t: 0 for t in target_indices}
    hist = {t: deque() for t in target_indices}
    prev_state = {}
    last_bid_change = defaultdict(lambda: np.nan)
    last_ask_change = defaultdict(lambda: np.nan)
    feat_rows = {}

    def emit_until(ticker, now_t):
        arr = target_indices.get(ticker, [])
        p = ptr.get(ticker, 0)
        h = hist.get(ticker)
        while p < len(arr) and arr[p][0] < now_t - EPS:
            trade_t, idx = arr[p]
            f = _book_features(h, trade_t, last_bid_change[ticker], last_ask_change[ticker]) if h else None
            if f is not None:
                feat_rows[idx] = f
            p += 1
        ptr[ticker] = p

    print("Streaming 3GB top-3 event book and extracting pre-trade features...")
    with (session / "book_top3_events.jsonl").open() as fh:
        for n, line in enumerate(fh, 1):
            try: x = json.loads(line)
            except Exception: continue
            ticker = str(x.get("ticker") or "")
            if ticker not in target_indices:
                continue
            t = _ts(x.get("receipt_time"))
            if not np.isfinite(t):
                continue
            emit_until(ticker, t)
            top = _top_fields(x)
            if top is None:
                continue
            prev = prev_state.get(ticker)
            bid_changed = bool(prev is None or abs(top["bid"] - prev["bid"]) > EPS or abs(top["bid_q1"] - prev["bid_q1"]) > EPS)
            ask_changed = bool(prev is None or abs(top["ask"] - prev["ask"]) > EPS or abs(top["ask_q1"] - prev["ask_q1"]) > EPS)
            if bid_changed: last_bid_change[ticker] = t
            if ask_changed: last_ask_change[ticker] = t
            ev = {
                "t": t, **top,
                "bid_changed": bid_changed, "ask_changed": ask_changed,
                "delta_side": str(x.get("delta_side") or "").lower(),
                "delta_qty": _f(x.get("delta_qty")) if np.isfinite(_f(x.get("delta_qty"))) else 0.0,
            }
            h = hist[ticker]
            h.append(ev)
            cutoff = t - MAX_PREHISTORY_S
            while h and h[0]["t"] < cutoff:
                h.popleft()
            prev_state[ticker] = top
            if n % 1_000_000 == 0:
                print(f"  streamed {n:,} book rows...")

    # Flush trades that occur before the last known state time.
    for ticker, arr in target_indices.items():
        h = hist[ticker]
        if h:
            emit_until(ticker, h[-1]["t"] + 1e-6)

    feature_df = pd.DataFrame.from_dict(feat_rows, orient="index")
    targets = targets.join(feature_df, how="left")
    targets = targets[np.isfinite(pd.to_numeric(targets.mid_pre, errors="coerce"))].copy()

    # Add strictly-prior aggressive-flow features.
    targets = _trade_flow_features(targets, trades)

    sign = np.where(targets.passive_side == "BID", 1.0, -1.0)
    for h in HORIZONS_S:
        hh = int(h)
        fut = pd.to_numeric(targets[f"future_mid_{hh}s"], errors="coerce").to_numpy(float)
        mid0 = pd.to_numeric(targets.mid_pre, errors="coerce").to_numpy(float)
        px = pd.to_numeric(targets.yes_price, errors="coerce").to_numpy(float)
        targets[f"post_mid_move_{hh}s_c"] = sign * 100.0 * (fut - mid0)
        targets[f"public_fill_markout_{hh}s_c"] = sign * 100.0 * (fut - px)

    target30 = pd.to_numeric(targets.post_mid_move_30s_c, errors="coerce")
    targets = targets[np.isfinite(target30)].copy()
    targets["toxicity_label_30s"] = np.select(
        [targets.post_mid_move_30s_c < 0, targets.post_mid_move_30s_c > 0],
        ["TOXIC", "FAVORABLE"], default="FLAT"
    )

    excluded = {
        "trade_id", "ticker", "series", "close_time", "receipt_ts", "elapsed_s", "qty", "yes_price",
        "taker_book_side", "close_ts", "passive_side", "toxicity_label_30s",
        "future_mid_5s", "future_mid_15s", "future_mid_30s",
        "post_mid_move_5s_c", "post_mid_move_15s_c", "post_mid_move_30s_c",
        "public_fill_markout_5s_c", "public_fill_markout_15s_c", "public_fill_markout_30s_c",
        "bid", "ask", "mid_pre",
    }
    features = [c for c in targets.columns if c not in excluded and pd.api.types.is_numeric_dtype(targets[c])]
    feature_summary = _feature_summary(targets, features)
    ranking = _stable_ranking(feature_summary)
    shapes = _quantile_shapes(targets, ranking)

    side_summary = targets.groupby("passive_side", as_index=False).agg(
        fills=("trade_id", "count"), windows=("close_ts", "nunique"),
        qty=("qty", "sum"),
        mean_post_5s_c=("post_mid_move_5s_c", "mean"),
        mean_post_15s_c=("post_mid_move_15s_c", "mean"),
        mean_post_30s_c=("post_mid_move_30s_c", "mean"),
        median_post_30s_c=("post_mid_move_30s_c", "median"),
        favorable_pct=("post_mid_move_30s_c", lambda x: 100.0 * (x > 0).mean()),
    )

    if output_dir is None:
        output_dir = project_root / "results" / "kalshi_mm_event_fill_toxicity_diag" / session.name
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    targets.to_csv(out / "fill_event_features.csv.gz", index=False, compression="gzip")
    side_summary.to_csv(out / "side_summary.csv", index=False)
    feature_summary.to_csv(out / "feature_good_vs_toxic.csv", index=False)
    ranking.to_csv(out / "feature_stability_ranking.csv", index=False)
    shapes.to_csv(out / "top_feature_quantile_shapes.csv", index=False)
    eligible.to_csv(out / "eligible_contracts.csv", index=False)
    (out / "study_config.json").write_text(json.dumps({
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "quoted_sample_exclusion": "KXBTC15M excluded because forensic audit found all reconstructed crossed/locked books isolated to BTC before any strategy/PnL test",
        "btc_ticker_factor_allowed": True,
        "contract_gate": "prior audit full_boundary_capture == True",
        "pretrade_windows_seconds": list(WINDOWS_S),
        "target_horizons_seconds": list(HORIZONS_S),
        "primary_target": "signed passive-side post-fill midpoint move at 30s",
        "strategy_replay": False,
        "pnl_evaluated": False,
    }, indent=2), encoding="utf-8")

    if show:
        print("\n" + "=" * 150)
        print("EVENT-TIME PASSIVE-FILL TOXICITY DIAGNOSTIC — DEVELOPMENT ONLY / NO STRATEGY / NO PNL")
        print("=" * 150)
        print(f"session={session.name} | eligible contracts={len(eligible_tickers)} | quoted series={eligible.series.nunique()}")
        print("BTC reconstructed book excluded by pre-PnL data-quality decision; BTC ticker retained only as factor input.")
        print("\nPASSIVE FILL SAMPLE")
        print(side_summary.round(4).to_string(index=False))
        print("\nTOP CHRONOLOGICALLY STABLE FEATURES — BID")
        print(ranking[(ranking.side == "BID") & ranking.same_sign_both_halves].head(20).round(4).to_string(index=False))
        print("\nTOP CHRONOLOGICALLY STABLE FEATURES — ASK")
        print(ranking[(ranking.side == "ASK") & ranking.same_sign_both_halves].head(20).round(4).to_string(index=False))
        print("\nTOP FEATURE QUANTILE SHAPES")
        if len(shapes):
            print(shapes.round(4).to_string(index=False))
        else:
            print("No stable feature had enough unique values for quintile shapes.")
        print("\nINTERPRETATION RULE")
        print("  Continue only if multiple causal microstructure features have the same Spearman sign in both chronological halves")
        print("  AND show coherent/roughly monotone window-balanced 30s toxicity shapes. Do not select a strategy threshold here.")
        print("\nOUTPUTS:", out)
        print("=" * 150)

    return {
        "output_dir": out,
        "fill_events": targets,
        "side_summary": side_summary,
        "feature_summary": feature_summary,
        "feature_stability_ranking": ranking,
        "quantile_shapes": shapes,
        "eligible_contracts": eligible,
    }
