from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .window_toxicity_history import (
    PROJECT_ROOT,
    FROZEN_SERIES,
    PERIODS,
    QTY,
    _build_windows,
    _display,
    _feature_table,
    _fill_intensity_test,
    _load_historical_observations,
    _period_for_time,
    find_historical_window_sources,
)

KALSHI_REST_BASE = "https://external-api.kalshi.com/trade-api/v2"
PRIMARY_DIR = "PRIMARY_SHADOW_M5_MINUS3C_15S_3CT_HOLD_V1"


def _clean_result(x):
    x = str(x or "").strip().upper()
    return x if x in {"YES", "NO"} else None


def _rest_result(ticker: str, timeout: float = 12.0):
    try:
        r = requests.get(f"{KALSHI_REST_BASE}/markets/{ticker}", timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        market = payload.get("market", payload) if isinstance(payload, dict) else {}
        return _clean_result(market.get("result"))
    except Exception:
        return None


def load_live_primary_signals(
    session_dir,
    settle_missing: bool = True,
    timeout: float = 12.0,
    show: bool = True,
):
    """Load all valid primary-shadow signals, including unfilled signals.

    shadow_records.csv only receives settlement on filled positions. For unfilled signals,
    recover the final market result from a small analysis cache first, then the public
    Kalshi market endpoint when requested. Raw recorder/shadow files are not modified.
    """
    session_dir = Path(session_dir)
    out_dir = session_dir / PRIMARY_DIR
    shadow = out_dir / "shadow_records.csv"
    cache_path = out_dir / "analysis_settlement_cache.csv"
    if not shadow.exists():
        raise FileNotFoundError(f"Missing primary shadow records: {shadow}")

    raw = pd.read_csv(shadow, low_memory=False)
    z = pd.DataFrame(index=raw.index)
    z["ticker"] = raw["ticker"].astype(str)
    z["series"] = raw["series"].astype(str) if "series" in raw.columns else z["ticker"].str.split("-").str[0]
    z["decision_time"] = pd.to_datetime(raw["decision_time"], utc=True, errors="coerce").dt.floor("min")
    z["direction"] = raw["direction"].astype(str).str.upper().str.strip()
    z["midpoint"] = pd.to_numeric(raw["midpoint"], errors="coerce")
    z["spread_c"] = pd.to_numeric(raw["spread_c"], errors="coerce")
    z["btc_return"] = pd.to_numeric(raw["btc_return"], errors="coerce")
    z["entry_price"] = pd.to_numeric(raw["entry_price"], errors="coerce")
    z["entry_fill_qty"] = pd.to_numeric(raw["entry_fill_qty"], errors="coerce").fillna(0.0)
    z["entry_queue"] = pd.to_numeric(raw["entry_queue"], errors="coerce")
    z["shadow_result"] = raw["result"].map(_clean_result) if "result" in raw.columns else None
    status = raw["status"].astype(str) if "status" in raw.columns else pd.Series("", index=raw.index)

    # Records are created for valid signals and DATA_INVALID decisions. A valid signal has
    # a frozen direction + legal -3c entry price; skips never become a shadow record.
    eligible = z[
        z["series"].isin(FROZEN_SERIES)
        & z["decision_time"].notna()
        & z["direction"].isin(["YES", "NO"])
        & z["entry_price"].between(0.01, 0.99, inclusive="both")
        & ~status.eq("DATA_INVALID")
    ].copy()

    cache = {}
    if cache_path.exists():
        try:
            c = pd.read_csv(cache_path, low_memory=False)
            if {"ticker", "result"}.issubset(c.columns):
                cache = {
                    str(t): r
                    for t, r in zip(c["ticker"], c["result"].map(_clean_result))
                    if r in {"YES", "NO"}
                }
        except Exception:
            cache = {}

    results = []
    sources = []
    new_cache = dict(cache)
    unresolved = []

    for _, row in eligible.iterrows():
        ticker = row["ticker"]
        result = row["shadow_result"]
        source = "shadow_record"

        if result not in {"YES", "NO"} and ticker in cache:
            result = cache[ticker]
            source = "analysis_cache"

        if result not in {"YES", "NO"} and settle_missing:
            result = _rest_result(ticker, timeout=timeout)
            if result in {"YES", "NO"}:
                source = "kalshi_rest"
                new_cache[ticker] = result

        if result not in {"YES", "NO"}:
            unresolved.append(ticker)
            source = "unresolved"

        results.append(result)
        sources.append(source)

    eligible["result_final"] = results
    eligible["settlement_source"] = sources

    if new_cache != cache:
        pd.DataFrame(
            sorted(new_cache.items()), columns=["ticker", "result"]
        ).assign(updated_at=pd.Timestamp.now(tz="UTC").isoformat()).to_csv(cache_path, index=False)

    total = len(eligible)
    settled = int(eligible["result_final"].isin(["YES", "NO"]).sum())
    live_coverage = pd.DataFrame([{
        "session": str(session_dir),
        "eligible_signals": total,
        "settled_signals": settled,
        "settlement_coverage_pct": 100.0 * settled / total if total else np.nan,
        "unresolved": total - settled,
    }])

    if show:
        print("Live primary settlement coverage:")
        _display(live_coverage.round(3))
        if unresolved:
            print(f"Unresolved current/recent markets: {len(unresolved)} (excluded until settled)")

    eligible = eligible[eligible["result_final"].isin(["YES", "NO"])].copy()
    eligible["filled"] = eligible["entry_fill_qty"] > 1e-12
    eligible["btc_opposition"] = True
    eligible["correct"] = eligible["direction"].eq(eligible["result_final"])
    eligible["signal_edge"] = np.where(
        eligible["correct"], 1.0 - eligible["entry_price"], -eligible["entry_price"]
    )
    eligible["hypo_pnl_3ct"] = QTY * eligible["signal_edge"]
    eligible["actual_pnl"] = eligible["entry_fill_qty"] * eligible["signal_edge"]
    eligible["mid_distance_50_c"] = 100.0 * (eligible["midpoint"] - 0.50).abs()
    eligible["btc_abs_bp"] = 10000.0 * eligible["btc_return"].abs()
    eligible["source_file"] = str(shadow)
    eligible["btc_source_file"] = "live completed-1m shadow pipeline"
    eligible["period"] = eligible["decision_time"].map(_period_for_time)

    keep = [
        "ticker", "series", "decision_time", "direction", "midpoint", "spread_c",
        "btc_return", "entry_price", "entry_fill_qty", "entry_queue", "result_final",
        "filled", "btc_opposition", "correct", "signal_edge", "hypo_pnl_3ct",
        "actual_pnl", "mid_distance_50_c", "btc_abs_bp", "source_file",
        "btc_source_file", "period", "settlement_source",
    ]
    return eligible[keep].reset_index(drop=True), live_coverage


def _bootstrap_breadth_diff(g: pd.DataFrame, n: int, seed: int):
    low = g.loc[g["signals"] <= 2, "signal_edge"].dropna().to_numpy(float)
    high = g.loc[g["signals"] >= 3, "signal_edge"].dropna().to_numpy(float)
    if len(low) < 2 or len(high) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    sims = np.empty(n, dtype=float)
    for i in range(n):
        l = rng.choice(low, size=len(low), replace=True).mean()
        h = rng.choice(high, size=len(high), replace=True).mean()
        sims[i] = h - l
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return 100.0 * lo, 100.0 * hi


def _regime_table(windows: pd.DataFrame, freq: str, bootstrap_n: int, seed: int):
    z = windows.copy()
    day = z["decision_time"].dt.floor("D")
    if freq == "monthly":
        z["slice"] = z["decision_time"].dt.strftime("%Y-%m")
        z["slice_start"] = z["decision_time"].dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    elif freq == "weekly":
        z["slice_start"] = day - pd.to_timedelta(day.dt.weekday, unit="D")
        z["slice"] = z["slice_start"].dt.strftime("%Y-%m-%d")
    else:
        raise ValueError("freq must be 'monthly' or 'weekly'")

    rows = []
    for j, (label, g) in enumerate(z.groupby("slice", sort=True)):
        low = g[g["signals"] <= 2]
        high = g[g["signals"] >= 3]
        low_edge = low["signal_edge"].mean() if len(low) else np.nan
        high_edge = high["signal_edge"].mean() if len(high) else np.nan
        diff = high_edge - low_edge if np.isfinite(low_edge) and np.isfinite(high_edge) else np.nan
        ci_lo, ci_hi = _bootstrap_breadth_diff(g, bootstrap_n, seed + j)

        rows.append({
            "slice": label,
            "slice_start": g["slice_start"].iloc[0],
            "first": g["decision_time"].min(),
            "last": g["decision_time"].max(),
            "windows": len(g),
            "signals": int(g["signals"].sum()),
            "low_1_2_windows": len(low),
            "high_3plus_windows": len(high),
            "high_3plus_share_pct": 100.0 * len(high) / len(g) if len(g) else np.nan,
            "all_window_edge_c": 100.0 * g["signal_edge"].mean(),
            "low_1_2_edge_c": 100.0 * low_edge if np.isfinite(low_edge) else np.nan,
            "high_3plus_edge_c": 100.0 * high_edge if np.isfinite(high_edge) else np.nan,
            "high_minus_low_c": 100.0 * diff if np.isfinite(diff) else np.nan,
            "high_minus_low_ci_lo_c": ci_lo,
            "high_minus_low_ci_hi_c": ci_hi,
            "all_accuracy_pct": 100.0 * g["accuracy"].mean(),
            "avg_btc_abs_bp": g["btc_abs_bp"].mean(),
            "avg_mid_distance_c": g["avg_mid_distance_c"].mean(),
            "avg_signals_per_window": g["signals"].mean(),
            "low_fill_rate_pct": 100.0 * low["fill_rate"].mean() if low["fill_rate"].notna().any() else np.nan,
            "high_fill_rate_pct": 100.0 * high["fill_rate"].mean() if high["fill_rate"].notna().any() else np.nan,
        })

    return pd.DataFrame(rows).sort_values("slice_start").reset_index(drop=True)


def run_window_regime_study(
    project_root=None,
    live_sessions=None,
    permutation_n: int = 20000,
    bootstrap_n: int = 10000,
    settle_missing_live: bool = True,
    show: bool = True,
):
    project_root = Path(project_root or PROJECT_ROOT)
    historical_root = project_root / "data" / "kalshi_historical_15m"

    print("=" * 110)
    print("WINDOW REGIME STUDY — APRIL -> JULY HISTORY + CLEAN LIVE AUGUST")
    print("=" * 110)

    source_df = find_historical_window_sources(project_root=project_root, show=show)
    if source_df.empty:
        raise RuntimeError(f"No historical minute5 files found under {historical_root}")

    frames = []
    for _, row in source_df.dropna(subset=["btc_1m"]).iterrows():
        obs_path = Path(row["observations"])
        btc_path = Path(row["btc_1m"])
        print(f"Loading {obs_path.parent.name}: {int(row['rows']):,} raw rows")
        h = _load_historical_observations(obs_path, btc_path)
        print(f"  frozen eligible signals: {len(h):,}")
        frames.append(h)

    live_coverages = []
    for session in live_sessions or []:
        print("Loading clean live primary shadow:", session)
        live, cov = load_live_primary_signals(
            session, settle_missing=settle_missing_live, show=show
        )
        frames.append(live)
        live_coverages.append(cov)

    signals = pd.concat(frames, ignore_index=True)
    signals = signals[signals["period"].notna()].copy()
    signals = signals.sort_values(["decision_time", "ticker", "source_file"]).drop_duplicates(
        ["ticker", "decision_time"], keep="last"
    ).reset_index(drop=True)
    windows = _build_windows(signals)

    coverage = signals.groupby("period").agg(
        signals=("ticker", "size"),
        windows=("decision_time", "nunique"),
        first=("decision_time", "min"),
        last=("decision_time", "max"),
        days=("decision_time", lambda x: x.dt.floor("D").nunique()),
        accuracy=("correct", "mean"),
        signal_edge=("signal_edge", "mean"),
    )
    coverage["accuracy_pct"] = 100.0 * coverage["accuracy"]
    coverage["signal_edge_c"] = 100.0 * coverage["signal_edge"]

    breadth = windows.groupby(["period", "signals"]).agg(
        windows=("decision_time", "size"),
        accuracy=("accuracy", "mean"),
        signal_edge=("signal_edge", "mean"),
        fill_rate=("fill_rate", "mean"),
        filled_edge=("filled_edge", "mean"),
    ).reset_index()
    breadth["accuracy_pct"] = 100.0 * breadth["accuracy"]
    breadth["signal_edge_c"] = 100.0 * breadth["signal_edge"]
    breadth["fill_rate_pct"] = 100.0 * breadth["fill_rate"]
    breadth["filled_edge_c"] = 100.0 * breadth["filled_edge"]

    studies = {
        "SIGNAL_BREADTH": _feature_table(windows, "signals", [0, 1, 2, 4, 8], ["1", "2", "3-4", "5+"]),
        "BTC_ABS": _feature_table(windows, "btc_abs_bp", [-1e-9, 5, 10, 15, 25, np.inf], ["0-5bp", "5-10bp", "10-15bp", "15-25bp", "25bp+"]),
        "MIDPOINT_EXTREMITY": _feature_table(windows, "avg_mid_distance_c", [-1e-9, 5, 10, 20, 30, 50], ["0-5c", "5-10c", "10-20c", "20-30c", "30c+"]),
    }

    consistency_rows = []
    required = ["APR_DISCOVERY", "MAY_JUN_VALID", "JUL_HOLDOUT"]
    for feature, table in studies.items():
        for bucket in table["bucket"].unique():
            row = {"feature": feature, "bucket": bucket}
            signs = []
            for period in required:
                b = table[(table["period"] == period) & (table["bucket"] == bucket)]
                base = windows[windows["period"] == period]
                if b.empty or base.empty:
                    delta, nwin = np.nan, 0
                else:
                    delta = float(b["avg_signal_edge"].iloc[0]) - float(base["signal_edge"].mean())
                    nwin = int(b["windows"].iloc[0])
                row[f"{period}_delta_c"] = 100.0 * delta if np.isfinite(delta) else np.nan
                row[f"{period}_windows"] = nwin
                if np.isfinite(delta) and delta != 0:
                    signs.append(np.sign(delta))
            row["same_direction_all_3"] = len(signs) == 3 and len(set(signs)) == 1
            row["consistently_toxic"] = row["same_direction_all_3"] and signs[0] < 0
            consistency_rows.append(row)
    consistency = pd.DataFrame(consistency_rows)

    fill_rows = []
    for period, g in windows.groupby("period"):
        res = _fill_intensity_test(g, n=permutation_n)
        if res is not None:
            fill_rows.append({"period": period, **res})
    fill_tests = pd.DataFrame(fill_rows).set_index("period") if fill_rows else pd.DataFrame()

    monthly_regime = _regime_table(windows, "monthly", bootstrap_n, seed=20260810)
    weekly_regime = _regime_table(windows, "weekly", bootstrap_n, seed=20260811)
    live_coverage = pd.concat(live_coverages, ignore_index=True) if live_coverages else pd.DataFrame()

    if show:
        print("\n" + "=" * 110 + "\nDATASET COVERAGE\n" + "=" * 110)
        _display(coverage[["signals", "windows", "days", "first", "last", "accuracy_pct", "signal_edge_c"]])

        print("\n" + "=" * 110 + "\nMONTHLY REGIME — 3+ SIGNAL WINDOWS VS 1-2\n" + "=" * 110)
        _display(monthly_regime[[
            "slice", "windows", "low_1_2_windows", "high_3plus_windows",
            "all_window_edge_c", "low_1_2_edge_c", "high_3plus_edge_c",
            "high_minus_low_c", "high_minus_low_ci_lo_c", "high_minus_low_ci_hi_c",
            "avg_btc_abs_bp", "avg_mid_distance_c"
        ]].round(3))

        print("\n" + "=" * 110 + "\nWEEKLY REGIME — 3+ SIGNAL WINDOWS VS 1-2\n" + "=" * 110)
        _display(weekly_regime[[
            "slice", "windows", "low_1_2_windows", "high_3plus_windows",
            "all_window_edge_c", "low_1_2_edge_c", "high_3plus_edge_c",
            "high_minus_low_c", "high_minus_low_ci_lo_c", "high_minus_low_ci_hi_c",
            "avg_btc_abs_bp", "avg_mid_distance_c"
        ]].round(3))

        print("\n" + "=" * 110 + "\nAUGUST / LIVE FILL INTENSITY TEST\n" + "=" * 110)
        _display(fill_tests.round(4) if not fill_tests.empty else pd.DataFrame({"note": ["No settled live execution rows"]}))

        print("\n" + "=" * 110 + "\nBREADTH DETAIL\n" + "=" * 110)
        _display(breadth[[
            "period", "signals", "windows", "accuracy_pct", "signal_edge_c",
            "fill_rate_pct", "filled_edge_c"
        ]].round(3))

    return {
        "signals": signals,
        "windows": windows,
        "coverage": coverage,
        "breadth": breadth,
        "studies": studies,
        "consistency": consistency,
        "fill_tests": fill_tests,
        "monthly_regime": monthly_regime,
        "weekly_regime": weekly_regime,
        "live_coverage": live_coverage,
        "sources": source_df,
    }
