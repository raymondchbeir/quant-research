from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .window_toxicity_history import PROJECT_ROOT

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

FEATURE_FAMILIES = {
    "signals": "BREADTH",
    "conf_mean_c": "KALSHI_CONFIDENCE",
    "conf_min_c": "KALSHI_CONFIDENCE",
    "conf_max_c": "KALSHI_CONFIDENCE",
    "conf_std_c": "KALSHI_CONFIDENCE",
    "entry_mean_c": "ENTRY_PRICE",
    "entry_min_c": "ENTRY_PRICE",
    "entry_max_c": "ENTRY_PRICE",
    "entry_std_c": "ENTRY_PRICE",
    "spread_mean_c": "SPREAD",
    "spread_max_c": "SPREAD",
    "share_conf_ge20": "KALSHI_CONFIDENCE",
    "share_conf_ge30": "KALSHI_CONFIDENCE",
    "share_entry_ge70": "ENTRY_PRICE",
    "share_entry_ge80": "ENTRY_PRICE",
    "btc_ret_5m_bp": "BTC_MOMENTUM",
    "btc_ret_15m_bp": "BTC_MOMENTUM",
    "btc_ret_30m_bp": "BTC_MOMENTUM",
    "btc_ret_60m_bp": "BTC_MOMENTUM",
    "btc_ret_120m_bp": "BTC_MOMENTUM",
    "btc_abs_15m_bp": "BTC_MOMENTUM",
    "btc_rv_60m_bp": "BTC_VOL",
    "btc_rv_6h_bp": "BTC_VOL",
    "btc_rv_24h_bp": "BTC_VOL",
    "btc_trend_alignment": "BTC_TREND",
    "btc_5v60_reversal": "BTC_TREND",
    "btc_dist_ma60_bp": "BTC_TREND",
}


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _normalize_btc_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns or "close" not in df.columns:
        raise ValueError("BTC frame requires time and close columns")
    z = df.copy()
    z["bucket_start"] = pd.to_datetime(z["time"], utc=True, errors="coerce")
    z["close"] = _safe_numeric(z["close"])
    z = z.dropna(subset=["bucket_start", "close"]).sort_values("bucket_start")
    z["available_time"] = z["bucket_start"] + pd.Timedelta(minutes=1)
    return z[["available_time", "close"]].drop_duplicates("available_time", keep="last")


def _fetch_coinbase_1m(start, end, cache_path: Path | None = None, pause_s: float = 0.12) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    if cache_path is not None and cache_path.exists():
        return _normalize_btc_frame(pd.read_csv(cache_path, low_memory=False))

    rows = []
    cursor = start
    session = requests.Session()

    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(minutes=250), end)
        params = {
            "granularity": 60,
            "start": cursor.isoformat(),
            "end": chunk_end.isoformat(),
        }
        r = session.get(COINBASE_CANDLES_URL, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        for row in payload:
            if isinstance(row, (list, tuple)) and len(row) >= 5:
                rows.append({"time": row[0], "close": row[4]})
        cursor = chunk_end
        if cursor < end:
            time.sleep(pause_s)

    if not rows:
        return pd.DataFrame(columns=["available_time", "close"])

    raw = pd.DataFrame(rows)
    raw["time"] = pd.to_datetime(raw["time"], unit="s", utc=True, errors="coerce")
    raw = raw.dropna(subset=["time"]).drop_duplicates("time", keep="last").sort_values("time")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cache_path, index=False)

    return _normalize_btc_frame(raw)


def _combined_btc_history(signals: pd.DataFrame, project_root: Path, show: bool = True) -> pd.DataFrame:
    frames = []
    loaded = set()

    if "btc_source_file" in signals.columns:
        for value in signals["btc_source_file"].dropna().astype(str).unique():
            path = Path(value)
            if path.exists() and path.is_file() and str(path.resolve()) not in loaded:
                try:
                    frames.append(_normalize_btc_frame(pd.read_csv(path, low_memory=False)))
                    loaded.add(str(path.resolve()))
                except Exception:
                    pass

    live = signals[signals["period"].eq("AUG_PROSPECTIVE")].copy() if "period" in signals.columns else pd.DataFrame()
    if not live.empty:
        lo = live["decision_time"].min() - pd.Timedelta(hours=25)
        hi = live["decision_time"].max() + pd.Timedelta(minutes=2)
        tag = f"{lo.strftime('%Y%m%d_%H%M')}_{hi.strftime('%Y%m%d_%H%M')}"
        cache = project_root / "data" / "kalshi_historical_15m" / "analysis_cache" / f"coinbase_BTCUSD_1m_{tag}.csv"
        try:
            live_btc = _fetch_coinbase_1m(lo, hi, cache_path=cache)
            if not live_btc.empty:
                frames.append(live_btc)
                if show:
                    print(f"Live BTC cache: {cache}")
        except Exception as exc:
            if show:
                print(f"WARNING: live BTC feature fetch failed: {exc!r}")

    if not frames:
        raise RuntimeError("No usable Coinbase 1m history found for detector features.")

    btc = pd.concat(frames, ignore_index=True)
    btc = btc.dropna(subset=["available_time", "close"]).sort_values("available_time")
    btc = btc.drop_duplicates("available_time", keep="last").reset_index(drop=True)

    logret_bp = 10000.0 * np.log(btc["close"] / btc["close"].shift(1))
    btc["btc_rv_60m_bp"] = np.sqrt(logret_bp.pow(2).rolling(60, min_periods=45).sum())
    btc["btc_rv_6h_bp"] = np.sqrt(logret_bp.pow(2).rolling(360, min_periods=240).sum())
    btc["btc_rv_24h_bp"] = np.sqrt(logret_bp.pow(2).rolling(1440, min_periods=900).sum())
    btc["ma60"] = btc["close"].rolling(60, min_periods=45).mean()
    btc["btc_dist_ma60_bp"] = 10000.0 * (btc["close"] / btc["ma60"] - 1.0)
    return btc


def _asof_close(base: pd.DataFrame, btc: pd.DataFrame, minutes: int, name: str) -> pd.DataFrame:
    left = base[["decision_time", "_row"]].copy()
    left["target"] = left["decision_time"] - pd.Timedelta(minutes=minutes)
    right = btc[["available_time", "close"]].rename(columns={"available_time": f"{name}_time", "close": name})
    out = pd.merge_asof(
        left.sort_values("target"),
        right.sort_values(f"{name}_time"),
        left_on="target",
        right_on=f"{name}_time",
        direction="backward",
    )
    return out[["_row", name]]


def _btc_features(decision_times: pd.Series, btc: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"decision_time": pd.to_datetime(decision_times, utc=True, errors="coerce")})
    base["_row"] = np.arange(len(base))

    now_right = btc[[
        "available_time", "close", "btc_rv_60m_bp", "btc_rv_6h_bp", "btc_rv_24h_bp", "btc_dist_ma60_bp"
    ]].rename(columns={"available_time": "btc_now_time", "close": "btc_now"})
    now = pd.merge_asof(
        base.sort_values("decision_time"),
        now_right.sort_values("btc_now_time"),
        left_on="decision_time",
        right_on="btc_now_time",
        direction="backward",
    )

    z = now[[
        "_row", "btc_now", "btc_rv_60m_bp", "btc_rv_6h_bp", "btc_rv_24h_bp", "btc_dist_ma60_bp"
    ]].copy()

    for minutes in (5, 15, 30, 60, 120):
        z = z.merge(_asof_close(base, btc, minutes, f"btc_{minutes}m_ago"), on="_row", how="left")
        z[f"btc_ret_{minutes}m_bp"] = 10000.0 * (z["btc_now"] / z[f"btc_{minutes}m_ago"] - 1.0)

    z["btc_abs_15m_bp"] = z["btc_ret_15m_bp"].abs()
    signs = np.sign(z[["btc_ret_5m_bp", "btc_ret_15m_bp", "btc_ret_30m_bp", "btc_ret_60m_bp"]])
    anchor = np.sign(z["btc_ret_15m_bp"]).to_numpy()[:, None]
    z["btc_trend_alignment"] = (signs.to_numpy() == anchor).sum(axis=1)
    z["btc_5v60_reversal"] = (np.sign(z["btc_ret_5m_bp"]) != np.sign(z["btc_ret_60m_bp"])).astype(float)
    z = z.sort_values("_row").reset_index(drop=True)
    return z.drop(columns=[c for c in z.columns if c.endswith("_ago") or c == "btc_now"])


def _build_exante_windows(study: dict, project_root: Path, show: bool = True) -> pd.DataFrame:
    signals = study["signals"].copy()
    outcomes = study["windows"].copy()

    signals["decision_time"] = pd.to_datetime(signals["decision_time"], utc=True, errors="coerce").dt.floor("min")
    signals["midpoint"] = _safe_numeric(signals["midpoint"])
    signals["entry_price"] = _safe_numeric(signals["entry_price"])
    signals["spread_c"] = _safe_numeric(signals["spread_c"])
    signals["conf_c"] = 100.0 * (signals["midpoint"] - 0.50).abs()
    signals["entry_c"] = 100.0 * signals["entry_price"]

    rows = []
    for (period, dt), g in signals.groupby(["period", "decision_time"], dropna=False):
        row = {
            "period": period,
            "decision_time": dt,
            "signals": len(g),
            "conf_mean_c": g["conf_c"].mean(),
            "conf_min_c": g["conf_c"].min(),
            "conf_max_c": g["conf_c"].max(),
            "conf_std_c": g["conf_c"].std(ddof=0),
            "entry_mean_c": g["entry_c"].mean(),
            "entry_min_c": g["entry_c"].min(),
            "entry_max_c": g["entry_c"].max(),
            "entry_std_c": g["entry_c"].std(ddof=0),
            "spread_mean_c": g["spread_c"].mean(),
            "spread_max_c": g["spread_c"].max(),
            "share_conf_ge20": (g["conf_c"] >= 20.0).mean(),
            "share_conf_ge30": (g["conf_c"] >= 30.0).mean(),
            "share_entry_ge70": (g["entry_c"] >= 70.0).mean(),
            "share_entry_ge80": (g["entry_c"] >= 80.0).mean(),
        }
        for series in sorted(signals["series"].dropna().astype(str).unique()):
            row[f"asset_{series}"] = float((g["series"].astype(str) == series).any())
        rows.append(row)

    ex = pd.DataFrame(rows)
    out_cols = [
        "period", "decision_time", "signal_edge", "signal_pnl_3ct", "accuracy",
        "fills", "fill_rate", "filled_edge", "actual_pnl", "avg_mid_distance_c",
        "mid_dispersion_c", "avg_entry_c", "btc_abs_bp",
    ]
    out_cols = [c for c in out_cols if c in outcomes.columns]
    ex = ex.merge(outcomes[out_cols], on=["period", "decision_time"], how="left")

    btc = _combined_btc_history(signals, project_root, show=show)
    bf = _btc_features(ex["decision_time"], btc)
    ex = pd.concat([ex.reset_index(drop=True), bf.drop(columns=["_row"]).reset_index(drop=True)], axis=1)
    return ex.sort_values("decision_time").reset_index(drop=True)


def _smd(a, b):
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    denom = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return (b.mean() - a.mean()) / denom if denom > 1e-12 else np.nan


def _edge_summary(g: pd.DataFrame, mask: pd.Series):
    flagged = g[mask]
    unflagged = g[~mask]
    return {
        "flagged_windows": len(flagged),
        "unflagged_windows": len(unflagged),
        "flagged_edge_c": 100.0 * flagged["signal_edge"].mean() if len(flagged) else np.nan,
        "unflagged_edge_c": 100.0 * unflagged["signal_edge"].mean() if len(unflagged) else np.nan,
        "flagged_filled_edge_c": 100.0 * flagged["filled_edge"].mean() if "filled_edge" in flagged and flagged["filled_edge"].notna().any() else np.nan,
        "unflagged_filled_edge_c": 100.0 * unflagged["filled_edge"].mean() if "filled_edge" in unflagged and unflagged["filled_edge"].notna().any() else np.nan,
    }


def _candidate_scan(ex: pd.DataFrame, ref: pd.DataFrame, toxic: pd.DataFrame, aug: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in FEATURE_FAMILIES if c in ex.columns]
    rows = []

    ref_h = ref[ref["signals"] >= 3].copy()
    tox_h = toxic[toxic["signals"] >= 3].copy()
    aug_h = aug[aug["signals"] >= 3].copy()

    for feature in features:
        r = ref_h[feature].dropna().astype(float)
        t = tox_h[feature].dropna().astype(float)
        if len(r) < 30 or len(t) < 10 or r.nunique() < 2:
            continue

        shift = _smd(r, t)
        if not np.isfinite(shift):
            continue

        direction = "HIGH" if t.median() >= r.median() else "LOW"
        threshold = r.quantile(0.80 if direction == "HIGH" else 0.20)

        def flag(g):
            if direction == "HIGH":
                return g[feature] >= threshold
            return g[feature] <= threshold

        ref_mask = flag(ref_h).fillna(False)
        tox_mask = flag(tox_h).fillna(False)
        aug_mask = flag(aug_h).fillna(False) if len(aug_h) else pd.Series(dtype=bool)

        ref_rate = ref_mask.mean() if len(ref_mask) else np.nan
        tox_rate = tox_mask.mean() if len(tox_mask) else np.nan
        aug_rate = aug_mask.mean() if len(aug_mask) else np.nan
        enrichment = tox_rate / ref_rate if np.isfinite(ref_rate) and ref_rate > 1e-12 else np.nan

        row = {
            "feature": feature,
            "family": FEATURE_FAMILIES.get(feature, "OTHER"),
            "direction": direction,
            "threshold": float(threshold),
            "ref_median": float(r.median()),
            "toxic_median": float(t.median()),
            "smd_toxic_vs_ref": float(shift),
            "ref_flag_rate_pct": 100.0 * ref_rate,
            "toxic_flag_rate_pct": 100.0 * tox_rate,
            "aug_flag_rate_pct": 100.0 * aug_rate if np.isfinite(aug_rate) else np.nan,
            "toxic_enrichment": enrichment,
        }
        row.update({f"ref_{k}": v for k, v in _edge_summary(ref_h, ref_mask).items()})
        row.update({f"toxic_{k}": v for k, v in _edge_summary(tox_h, tox_mask).items()})
        if len(aug_h):
            row.update({f"aug_{k}": v for k, v in _edge_summary(aug_h, aug_mask).items()})

        score = abs(shift) * math.log1p(enrichment) if np.isfinite(enrichment) and enrichment > 0 else abs(shift)
        row["historical_rank_score"] = score
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("historical_rank_score", ascending=False).reset_index(drop=True)


def _select_rules(scan: pd.DataFrame, max_rules: int = 4) -> pd.DataFrame:
    if scan.empty:
        return scan.copy()
    chosen = []
    used_families = set()
    for _, row in scan.iterrows():
        family = row["family"]
        if family in used_families:
            continue
        if row["toxic_flagged_windows"] < 8:
            continue
        if not (5.0 <= row["ref_flag_rate_pct"] <= 40.0):
            continue
        chosen.append(row)
        used_families.add(family)
        if len(chosen) >= max_rules:
            break
    return pd.DataFrame(chosen).reset_index(drop=True)


def _apply_rules(ex: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    z = ex.copy()
    z["risk_score"] = 0
    if rules.empty:
        z["risk_flag"] = False
        return z

    for i, row in rules.iterrows():
        feature = row["feature"]
        threshold = float(row["threshold"])
        if row["direction"] == "HIGH":
            hit = z[feature] >= threshold
        else:
            hit = z[feature] <= threshold
        z[f"risk_rule_{i+1}"] = hit.fillna(False)
        z["risk_score"] += z[f"risk_rule_{i+1}"].astype(int)

    trigger = max(1, math.ceil(len(rules) / 2))
    z["risk_flag"] = (z["signals"] >= 3) & (z["risk_score"] >= trigger)
    return z


def _prevention_table(z: pd.DataFrame) -> pd.DataFrame:
    segments = {
        "REFERENCE_APR_TO_JUN28": (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-06-29", tz="UTC")),
        "TOXIC_JUN29_TO_JUL3": (pd.Timestamp("2026-06-29", tz="UTC"), pd.Timestamp("2026-07-04", tz="UTC")),
        "AUG10_LIVE": (pd.Timestamp("2026-08-10", tz="UTC"), pd.Timestamp("2026-08-11", tz="UTC")),
    }
    rows = []
    for name, (start, end) in segments.items():
        g = z[(z["decision_time"] >= start) & (z["decision_time"] < end)].copy()
        if g.empty:
            continue
        flagged = g[g["risk_flag"]]
        baseline_signal_pnl = g["signal_pnl_3ct"].sum()
        skip_signal_pnl = g.loc[~g["risk_flag"], "signal_pnl_3ct"].sum()
        baseline_actual = g["actual_pnl"].sum(min_count=1) if "actual_pnl" in g.columns else np.nan
        skip_actual = g.loc[~g["risk_flag"], "actual_pnl"].sum(min_count=1) if "actual_pnl" in g.columns else np.nan
        rows.append({
            "segment": name,
            "windows": len(g),
            "high_breadth_windows": int((g["signals"] >= 3).sum()),
            "risk_flagged_windows": len(flagged),
            "risk_flagged_share_pct": 100.0 * len(flagged) / len(g),
            "baseline_window_edge_c": 100.0 * g["signal_edge"].mean(),
            "flagged_window_edge_c": 100.0 * flagged["signal_edge"].mean() if len(flagged) else np.nan,
            "unflagged_window_edge_c": 100.0 * g.loc[~g["risk_flag"], "signal_edge"].mean(),
            "baseline_signal_pnl_3ct": baseline_signal_pnl,
            "skip_flagged_signal_pnl_3ct": skip_signal_pnl,
            "signal_pnl_change_if_skip": skip_signal_pnl - baseline_signal_pnl,
            "baseline_actual_pnl": baseline_actual,
            "actual_pnl_if_skip_flagged": skip_actual,
            "actual_pnl_change_if_skip": skip_actual - baseline_actual if np.isfinite(baseline_actual) and np.isfinite(skip_actual) else np.nan,
        })
    return pd.DataFrame(rows)


def run_toxic_window_detector(
    study: dict,
    project_root=None,
    show: bool = True,
    max_rules: int = 4,
):
    """Exploratory ex-ante detector study.

    Development protocol is fixed inside this function:
      - normal reference: Apr 1 through Jun 28, 2026
      - historical toxic episode: Jun 29 through Jul 3, 2026
      - Aug 10 live session: validation/display only, never used to choose thresholds or rules

    The detector is not approved for deployment. It is intended to identify candidate state
    variables and estimate the opportunity cost of skipping flagged high-breadth windows.
    """
    project_root = Path(project_root or PROJECT_ROOT)

    print("=" * 118)
    print("EX-ANTE TOXIC WINDOW DETECTOR — HISTORICAL DEVELOPMENT ONLY")
    print("=" * 118)
    print("Reference: 2026-04-01 through 2026-06-28")
    print("Toxic development episode: 2026-06-29 through 2026-07-03")
    print("August: validation/display only; never used to select rules")

    ex = _build_exante_windows(study, project_root=project_root, show=show)

    ref = ex[(ex["decision_time"] >= pd.Timestamp("2026-04-01", tz="UTC")) & (ex["decision_time"] < pd.Timestamp("2026-06-29", tz="UTC"))].copy()
    toxic = ex[(ex["decision_time"] >= pd.Timestamp("2026-06-29", tz="UTC")) & (ex["decision_time"] < pd.Timestamp("2026-07-04", tz="UTC"))].copy()
    aug = ex[(ex["decision_time"] >= pd.Timestamp("2026-08-10", tz="UTC")) & (ex["decision_time"] < pd.Timestamp("2026-08-11", tz="UTC"))].copy()

    coverage = pd.DataFrame([
        {"segment": "REFERENCE_APR_TO_JUN28", "windows": len(ref), "high_breadth": int((ref["signals"] >= 3).sum())},
        {"segment": "TOXIC_JUN29_TO_JUL3", "windows": len(toxic), "high_breadth": int((toxic["signals"] >= 3).sum())},
        {"segment": "AUG10_LIVE", "windows": len(aug), "high_breadth": int((aug["signals"] >= 3).sum())},
    ])

    scan = _candidate_scan(ex, ref, toxic, aug)
    rules = _select_rules(scan, max_rules=max_rules)
    scored = _apply_rules(ex, rules)
    prevention = _prevention_table(scored)

    high_breadth = scored[scored["signals"] >= 3].copy()
    risk_buckets = high_breadth.groupby(["period", "risk_score"]).agg(
        windows=("decision_time", "size"),
        signal_edge=("signal_edge", "mean"),
        filled_edge=("filled_edge", "mean"),
        fill_rate=("fill_rate", "mean"),
        actual_pnl=("actual_pnl", "sum"),
    ).reset_index()
    risk_buckets["signal_edge_c"] = 100.0 * risk_buckets["signal_edge"]
    risk_buckets["filled_edge_c"] = 100.0 * risk_buckets["filled_edge"]
    risk_buckets["fill_rate_pct"] = 100.0 * risk_buckets["fill_rate"]

    if show:
        print("\n" + "=" * 118 + "\nCOVERAGE\n" + "=" * 118)
        _display(coverage)

        print("\n" + "=" * 118 + "\nRANKED EX-ANTE FEATURE SHIFTS — HIGH-BREADTH WINDOWS ONLY\n" + "=" * 118)
        cols = [
            "feature", "family", "direction", "threshold", "ref_median", "toxic_median",
            "smd_toxic_vs_ref", "ref_flag_rate_pct", "toxic_flag_rate_pct", "aug_flag_rate_pct",
            "toxic_enrichment", "ref_flagged_edge_c", "toxic_flagged_edge_c", "aug_flagged_edge_c",
            "historical_rank_score",
        ]
        cols = [c for c in cols if c in scan.columns]
        _display(scan[cols].head(20).round(3) if not scan.empty else scan)

        print("\n" + "=" * 118 + "\nHISTORICAL-ONLY CANDIDATE RULES\n" + "=" * 118)
        if rules.empty:
            print("No candidate rules met the minimum support requirements.")
        else:
            _display(rules[[
                "feature", "family", "direction", "threshold", "ref_flag_rate_pct",
                "toxic_flag_rate_pct", "aug_flag_rate_pct", "toxic_enrichment",
                "ref_flagged_edge_c", "toxic_flagged_edge_c", "aug_flagged_edge_c",
            ]].round(3))
            print(f"Composite trigger: at least {max(1, math.ceil(len(rules) / 2))} of {len(rules)} historical-only rules AND signals>=3")

        print("\n" + "=" * 118 + "\nHYPOTHETICAL PREVENTION TEST — SKIP FLAGGED HIGH-BREADTH WINDOWS\n" + "=" * 118)
        _display(prevention.round(3))
        print("\nWARNING: prevention table is exploratory. Do not change the frozen live strategy from this sample.")

        print("\n" + "=" * 118 + "\nRISK SCORE DETAIL — HIGH-BREADTH WINDOWS\n" + "=" * 118)
        _display(risk_buckets[[
            "period", "risk_score", "windows", "signal_edge_c", "fill_rate_pct", "filled_edge_c", "actual_pnl"
        ]].round(3))

        print("\nNext interpretation rule:")
        print("  Useful detector candidate = enriched in Jun29-Jul3, costly/rare in reference, and same direction on Aug10.")
        print("  If August disagrees, reject it rather than retuning the threshold on August.")

    return {
        "exante_windows": ex,
        "coverage": coverage,
        "feature_scan": scan,
        "candidate_rules": rules,
        "scored_windows": scored,
        "prevention": prevention,
        "risk_buckets": risk_buckets,
    }
