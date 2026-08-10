from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FROZEN_SERIES = {
    "KXBNB15M", "KXDOGE15M", "KXETH15M", "KXHYPE15M",
    "KXNEAR15M", "KXSOL15M", "KXXRP15M", "KXZEC15M",
}

PERIODS = {
    "APR_DISCOVERY": ("2026-04-01", "2026-05-01"),
    "MAY_JUN_VALID": ("2026-05-01", "2026-07-01"),
    "JUL_HOLDOUT": ("2026-07-01", "2026-08-01"),
    "AUG_PROSPECTIVE": ("2026-08-01", "2026-09-01"),
}

DEPTH_C = 3.0
MAX_SPREAD_C = 2.0
QTY = 3.0


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _period_for_time(t):
    if pd.isna(t):
        return None
    for label, (start, end) in PERIODS.items():
        if pd.Timestamp(start, tz="UTC") <= t < pd.Timestamp(end, tz="UTC"):
            return label
    return None


def _load_btc(path: Path) -> pd.DataFrame:
    btc = pd.read_csv(path, low_memory=False)
    if "time" not in btc.columns or "close" not in btc.columns:
        raise ValueError(f"BTC file missing time/close columns: {path}")
    btc = btc[["time", "close"]].copy()
    btc["bucket_start"] = pd.to_datetime(btc["time"], utc=True, errors="coerce")
    btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
    btc = btc.dropna(subset=["bucket_start", "close"]).sort_values("bucket_start")
    # Coinbase 1m timestamps are bucket START; close is causal at bucket END.
    btc["available_time"] = btc["bucket_start"] + pd.Timedelta(minutes=1)
    return btc[["available_time", "close"]].drop_duplicates("available_time", keep="last")


def _btc_return_15m(decision_times: pd.Series, btc: pd.DataFrame) -> pd.Series:
    base = pd.DataFrame({"decision_time": pd.to_datetime(decision_times, utc=True, errors="coerce")})
    base["_row"] = np.arange(len(base))

    now = pd.merge_asof(
        base.sort_values("decision_time"),
        btc.rename(columns={"available_time": "btc_now_time", "close": "btc_now_close"}).sort_values("btc_now_time"),
        left_on="decision_time",
        right_on="btc_now_time",
        direction="backward",
    )

    past = base.copy()
    past["past_target"] = past["decision_time"] - pd.Timedelta(minutes=15)
    past = pd.merge_asof(
        past.sort_values("past_target"),
        btc.rename(columns={"available_time": "btc_past_time", "close": "btc_past_close"}).sort_values("btc_past_time"),
        left_on="past_target",
        right_on="btc_past_time",
        direction="backward",
    )

    z = now[["_row", "btc_now_close"]].merge(
        past[["_row", "btc_past_close"]],
        on="_row",
        how="left",
    ).sort_values("_row")

    out = z["btc_now_close"] / z["btc_past_close"] - 1.0
    out.index = decision_times.index
    return out


def _best_btc_file(obs_path: Path, historical_root: Path) -> Path:
    local = obs_path.parent / "coinbase_BTCUSD_1m.csv"
    if local.exists():
        return local

    candidates = sorted(historical_root.glob("coinbase_BTCUSD_1m*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No Coinbase BTC 1m file found for {obs_path}")

    obs = pd.read_csv(obs_path, usecols=lambda c: c == "decision_utc", low_memory=False)
    t = pd.to_datetime(obs["decision_utc"], utc=True, errors="coerce").dropna()
    if t.empty:
        raise ValueError(f"No valid decision_utc in {obs_path}")

    best = None
    best_overlap = pd.Timedelta(0)
    for path in candidates:
        try:
            sample = pd.read_csv(path, usecols=["time"], low_memory=False)
            bt = pd.to_datetime(sample["time"], utc=True, errors="coerce").dropna()
            if bt.empty:
                continue
            lo, hi = bt.min(), bt.max() + pd.Timedelta(minutes=1)
            overlap = max(pd.Timedelta(0), min(t.max(), hi) - max(t.min(), lo))
            if best is None or overlap > best_overlap:
                best = path
                best_overlap = overlap
        except Exception:
            continue

    if best is None:
        raise FileNotFoundError(f"No usable Coinbase file found for {obs_path}")
    return best


def _load_historical_observations(obs_path: Path, btc_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(obs_path, low_memory=False)
    needed = {
        "ticker", "series_ticker", "decision_utc", "yes_bid", "yes_ask",
        "mid", "spread_c", "outcome_yes",
    }
    missing = sorted(needed - set(raw.columns))
    if missing:
        raise ValueError(f"{obs_path} missing columns: {missing}")

    out = pd.DataFrame()
    out["ticker"] = raw["ticker"].astype(str)
    out["series"] = raw["series_ticker"].astype(str)
    out["decision_time"] = pd.to_datetime(raw["decision_utc"], utc=True, errors="coerce").dt.floor("min")
    out["yes_bid"] = pd.to_numeric(raw["yes_bid"], errors="coerce")
    out["yes_ask"] = pd.to_numeric(raw["yes_ask"], errors="coerce")
    out["midpoint"] = pd.to_numeric(raw["mid"], errors="coerce")
    out["spread_c"] = pd.to_numeric(raw["spread_c"], errors="coerce")
    out["outcome_yes"] = pd.to_numeric(raw["outcome_yes"], errors="coerce")
    out["error"] = raw["error"] if "error" in raw.columns else np.nan

    out = out[
        out["series"].isin(FROZEN_SERIES)
        & out["decision_time"].notna()
        & out["yes_bid"].notna()
        & out["yes_ask"].notna()
        & out["midpoint"].notna()
        & out["outcome_yes"].isin([0, 1])
    ].copy()

    if "error" in out.columns:
        out = out[out["error"].isna() | out["error"].astype(str).isin(["", "nan", "None"])].copy()

    out = out[(out["spread_c"] <= MAX_SPREAD_C + 1e-12) & (out["midpoint"] != 0.50)].copy()
    out["direction"] = np.where(out["midpoint"] > 0.50, "YES", "NO")
    out["result_final"] = np.where(out["outcome_yes"] == 1, "YES", "NO")

    out["held_bid"] = np.where(out["direction"].eq("YES"), out["yes_bid"], 1.0 - out["yes_ask"])
    out["held_ask"] = np.where(out["direction"].eq("YES"), out["yes_ask"], 1.0 - out["yes_bid"])
    # Match frozen primary: round(held_bid - 3c, 2).
    out["entry_price"] = np.round(out["held_bid"] - DEPTH_C / 100.0, 2)
    out = out[
        out["entry_price"].between(0.01, 0.99, inclusive="both")
        & (out["entry_price"] < out["held_ask"])
    ].copy()

    btc = _load_btc(btc_path)
    out["btc_return"] = _btc_return_15m(out["decision_time"], btc)
    sign = np.where(out["direction"].eq("YES"), 1.0, -1.0)
    out["btc_opposition"] = out["btc_return"] * sign < 0
    out = out[out["btc_return"].notna() & out["btc_opposition"]].copy()

    out["correct"] = out["direction"].eq(out["result_final"])
    out["signal_edge"] = np.where(out["correct"], 1.0 - out["entry_price"], -out["entry_price"])
    out["hypo_pnl_3ct"] = QTY * out["signal_edge"]
    out["mid_distance_50_c"] = 100.0 * (out["midpoint"] - 0.50).abs()
    out["btc_abs_bp"] = 10000.0 * out["btc_return"].abs()

    # Historical M5 files do not contain 15s order-book/trade replay.
    out["filled"] = np.nan
    out["entry_fill_qty"] = np.nan
    out["entry_queue"] = np.nan
    out["actual_pnl"] = np.nan
    out["source_file"] = str(obs_path)
    out["btc_source_file"] = str(btc_path)
    out["period"] = out["decision_time"].map(_period_for_time)
    return out


def _load_live_shadow(session_dir: Path) -> pd.DataFrame:
    session_dir = Path(session_dir)
    shadow = session_dir / "PRIMARY_SHADOW_M5_MINUS3C_15S_3CT_HOLD_V1" / "shadow_records.csv"
    if not shadow.exists():
        raise FileNotFoundError(f"Missing primary shadow records: {shadow}")

    raw = pd.read_csv(shadow, low_memory=False)
    out = pd.DataFrame()
    out["ticker"] = raw["ticker"].astype(str)
    out["series"] = raw["series"].astype(str) if "series" in raw.columns else out["ticker"].str.split("-").str[0]
    out["decision_time"] = pd.to_datetime(raw["decision_time"], utc=True, errors="coerce").dt.floor("min")
    out["direction"] = raw["direction"].astype(str).str.upper()
    out["midpoint"] = pd.to_numeric(raw["midpoint"], errors="coerce")
    out["spread_c"] = pd.to_numeric(raw["spread_c"], errors="coerce")
    out["btc_return"] = pd.to_numeric(raw["btc_return"], errors="coerce")
    out["entry_price"] = pd.to_numeric(raw["entry_price"], errors="coerce")
    out["entry_fill_qty"] = pd.to_numeric(raw["entry_fill_qty"], errors="coerce").fillna(0)
    out["entry_queue"] = pd.to_numeric(raw["entry_queue"], errors="coerce")
    out["actual_pnl"] = pd.to_numeric(raw["realized_pnl"], errors="coerce").fillna(0)
    out["result_final"] = raw["result"].astype(str).str.upper()
    status = raw["status"].astype(str) if "status" in raw.columns else pd.Series("", index=raw.index)

    out = out[
        out["series"].isin(FROZEN_SERIES)
        & out["direction"].isin(["YES", "NO"])
        & out["result_final"].isin(["YES", "NO"])
        & out["entry_price"].notna()
        & ~status.eq("DATA_INVALID")
    ].copy()

    out["filled"] = out["entry_fill_qty"] > 1e-12
    out["btc_opposition"] = True
    out["correct"] = out["direction"].eq(out["result_final"])
    out["signal_edge"] = np.where(out["correct"], 1.0 - out["entry_price"], -out["entry_price"])
    out["hypo_pnl_3ct"] = QTY * out["signal_edge"]
    out["mid_distance_50_c"] = 100.0 * (out["midpoint"] - 0.50).abs()
    out["btc_abs_bp"] = 10000.0 * out["btc_return"].abs()
    out["source_file"] = str(shadow)
    out["btc_source_file"] = "live completed-1m shadow pipeline"
    out["period"] = out["decision_time"].map(_period_for_time)
    return out


def find_historical_window_sources(project_root=None, show=True):
    project_root = Path(project_root or PROJECT_ROOT)
    historical_root = project_root / "data" / "kalshi_historical_15m"
    paths = sorted(historical_root.rglob("minute5_contract_observations.csv")) if historical_root.exists() else []

    rows = []
    for obs_path in paths:
        try:
            raw = pd.read_csv(obs_path, usecols=lambda c: c == "decision_utc", low_memory=False)
            t = pd.to_datetime(raw["decision_utc"], utc=True, errors="coerce").dropna()
            btc_path = _best_btc_file(obs_path, historical_root)
            rows.append({
                "observations": str(obs_path),
                "btc_1m": str(btc_path),
                "rows": len(raw),
                "start": t.min() if len(t) else pd.NaT,
                "end": t.max() if len(t) else pd.NaT,
            })
        except Exception as exc:
            rows.append({
                "observations": str(obs_path),
                "btc_1m": None,
                "rows": np.nan,
                "start": pd.NaT,
                "end": pd.NaT,
                "error": repr(exc),
            })

    df = pd.DataFrame(rows)
    if show:
        print(f"Historical minute-5 observation files: {len(df)}")
        _display(df)
    return df


def _build_windows(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, decision_time), g in signals.groupby(["period", "decision_time"], dropna=False):
        counts = g["direction"].value_counts()
        same_n = int(counts.max()) if len(counts) else 0
        fill_known = g["filled"].notna().any()
        fill_mask = g["filled"].fillna(False).astype(bool) if fill_known else pd.Series(False, index=g.index)
        fills = int(fill_mask.sum()) if fill_known else np.nan

        rows.append({
            "period": period,
            "decision_time": decision_time,
            "signals": len(g),
            "same_direction_n": same_n,
            "same_direction_share": same_n / len(g),
            "unanimous_direction": len(counts) == 1,
            "accuracy": g["correct"].mean(),
            "signal_edge": g["signal_edge"].mean(),
            "signal_pnl_3ct": g["hypo_pnl_3ct"].sum(),
            "avg_mid_distance_c": g["mid_distance_50_c"].mean(),
            "mid_dispersion_c": 100.0 * g["midpoint"].std(ddof=0) if g["midpoint"].notna().any() else np.nan,
            "avg_entry_c": 100.0 * g["entry_price"].mean(),
            "btc_abs_bp": g["btc_abs_bp"].mean(),
            "avg_queue": g["entry_queue"].mean(),
            "zero_queue_share": ((g["entry_queue"].fillna(np.inf) <= 1e-12).mean() if g["entry_queue"].notna().any() else np.nan),
            "fills": fills,
            "fill_rate": fills / len(g) if fill_known else np.nan,
            "filled_edge": g.loc[fill_mask, "signal_edge"].mean() if fill_known and fills else np.nan,
            "actual_pnl": g.loc[fill_mask, "actual_pnl"].sum() if fill_known else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["period", "decision_time"]).reset_index(drop=True)


def _feature_table(windows, feature, bins=None, labels=None, categorical=False):
    z = windows.copy()
    z["bucket"] = z[feature].astype(str) if categorical else pd.cut(
        z[feature], bins=bins, labels=labels, include_lowest=True
    )
    out = z.dropna(subset=["bucket"]).groupby(["period", "bucket"], observed=True).agg(
        windows=("decision_time", "size"),
        signals=("signals", "sum"),
        avg_accuracy=("accuracy", "mean"),
        avg_signal_edge=("signal_edge", "mean"),
        avg_fill_rate=("fill_rate", "mean"),
        avg_filled_edge=("filled_edge", "mean"),
        actual_pnl=("actual_pnl", "sum"),
    ).reset_index()
    out["accuracy_pct"] = 100 * out["avg_accuracy"]
    out["signal_edge_c"] = 100 * out["avg_signal_edge"]
    out["fill_rate_pct"] = 100 * out["avg_fill_rate"]
    out["filled_edge_c"] = 100 * out["avg_filled_edge"]
    return out


def _fill_intensity_test(z, n=20000, seed=20260810):
    z = z[z["fills"].notna() & z["signal_edge"].notna()].copy()
    if len(z) < 5 or z["fills"].sum() <= 0:
        return None

    fills = z["fills"].to_numpy(float)
    edges = z["signal_edge"].to_numpy(float)
    actual = np.average(edges, weights=fills)
    rng = np.random.default_rng(seed)
    sims = np.empty(n)
    for i in range(n):
        sims[i] = np.average(edges, weights=rng.permutation(fills))

    return {
        "windows": len(z),
        "actual_fill_weighted_edge_c": 100 * actual,
        "random_fill_edge_c": 100 * sims.mean(),
        "window_selection_penalty_c": 100 * (actual - sims.mean()),
        "p_value": (1 + np.sum(sims <= actual)) / (n + 1),
    }


def run_historical_window_toxicity_study(
    project_root=None,
    live_sessions=None,
    permutation_n=20000,
    show=True,
):
    project_root = Path(project_root or PROJECT_ROOT)
    historical_root = project_root / "data" / "kalshi_historical_15m"

    print("=" * 110)
    print("HISTORICAL WINDOW TOXICITY STUDY — RAW M5 SOURCES")
    print("=" * 110)
    print("Historical root:", historical_root)

    source_df = find_historical_window_sources(project_root=project_root, show=show)
    if source_df.empty:
        raise RuntimeError(f"No minute5_contract_observations.csv files found under {historical_root}")

    frames = []
    for _, row in source_df.dropna(subset=["btc_1m"]).iterrows():
        obs_path = Path(row["observations"])
        btc_path = Path(row["btc_1m"])
        print(f"Loading {obs_path.parent.name}: {int(row['rows']):,} rows")
        z = _load_historical_observations(obs_path, btc_path)
        print(f"  frozen M5 + spread<=2c + BTC opposition: {len(z):,}")
        frames.append(z)

    for session in live_sessions or []:
        print("Loading clean live primary shadow:", session)
        frames.append(_load_live_shadow(Path(session)))

    if not frames:
        raise RuntimeError("No usable historical/live signals loaded.")

    signals = pd.concat(frames, ignore_index=True)
    signals = signals[signals["period"].notna()].copy()
    signals = signals.sort_values(["decision_time", "ticker", "source_file"]).drop_duplicates(
        ["ticker", "decision_time"], keep="first"
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
    coverage["accuracy_pct"] = 100 * coverage["accuracy"]
    coverage["signal_edge_c"] = 100 * coverage["signal_edge"]
    baseline = coverage[["signals", "windows", "days", "accuracy_pct", "signal_edge_c"]].copy()

    studies = {
        "SIGNAL_BREADTH": _feature_table(windows, "signals", [0, 1, 2, 4, 8], ["1", "2", "3-4", "5+"]),
        "SAME_DIRECTION_N": _feature_table(windows, "same_direction_n", [0, 1, 2, 4, 8], ["1", "2", "3-4", "5+"]),
        "UNANIMOUS": _feature_table(windows, "unanimous_direction", categorical=True),
        "BTC_ABS": _feature_table(windows, "btc_abs_bp", [-1e-9, 5, 10, 15, 25, np.inf], ["0-5bp", "5-10bp", "10-15bp", "15-25bp", "25bp+"]),
        "MIDPOINT_EXTREMITY": _feature_table(windows, "avg_mid_distance_c", [-1e-9, 5, 10, 20, 30, 50], ["0-5c", "5-10c", "10-20c", "20-30c", "30c+"]),
    }

    required = ["APR_DISCOVERY", "MAY_JUN_VALID", "JUL_HOLDOUT"]
    consistency_rows = []
    for feature, table in studies.items():
        for bucket in table["bucket"].unique():
            row = {"feature": feature, "bucket": bucket}
            signs = []
            for period in required:
                bucket_row = table[(table["period"] == period) & (table["bucket"] == bucket)]
                base = windows[windows["period"] == period]
                if bucket_row.empty or base.empty:
                    delta, nwin = np.nan, 0
                else:
                    delta = float(bucket_row["avg_signal_edge"].iloc[0]) - float(base["signal_edge"].mean())
                    nwin = int(bucket_row["windows"].iloc[0])
                row[f"{period}_delta_c"] = 100 * delta if np.isfinite(delta) else np.nan
                row[f"{period}_windows"] = nwin
                if np.isfinite(delta) and delta != 0:
                    signs.append(np.sign(delta))
            row["same_direction_all_3"] = len(signs) == 3 and len(set(signs)) == 1
            row["consistently_toxic"] = row["same_direction_all_3"] and signs[0] < 0
            consistency_rows.append(row)
    consistency = pd.DataFrame(consistency_rows)

    perm_rows = []
    for period, z in windows.groupby("period"):
        res = _fill_intensity_test(z, n=permutation_n)
        if res is not None:
            perm_rows.append({"period": period, **res})
    fill_tests = pd.DataFrame(perm_rows).set_index("period") if perm_rows else pd.DataFrame()

    breadth = windows.groupby(["period", "signals"]).agg(
        windows=("decision_time", "size"),
        accuracy=("accuracy", "mean"),
        signal_edge=("signal_edge", "mean"),
        fill_rate=("fill_rate", "mean"),
        filled_edge=("filled_edge", "mean"),
    ).reset_index()
    breadth["accuracy_pct"] = 100 * breadth["accuracy"]
    breadth["signal_edge_c"] = 100 * breadth["signal_edge"]
    breadth["fill_rate_pct"] = 100 * breadth["fill_rate"]
    breadth["filled_edge_c"] = 100 * breadth["filled_edge"]

    if show:
        print("\n" + "=" * 110 + "\nDATASET COVERAGE\n" + "=" * 110)
        _display(coverage[["signals", "windows", "days", "first", "last", "accuracy_pct", "signal_edge_c"]].round(3))

        if "JUL_HOLDOUT" not in coverage.index:
            print("\nWARNING: no July Kalshi M5 observations were found.")
        elif int(coverage.loc["JUL_HOLDOUT", "days"]) < 20:
            july = coverage.loc["JUL_HOLDOUT"]
            print(
                f"\nWARNING: July holdout has only {int(july['days'])} calendar days of M5 data "
                f"({july['first']} through {july['last']}). Treat July conclusions as partial."
            )

        print("\n" + "=" * 110 + "\nBASELINE BY PERIOD\n" + "=" * 110)
        _display(baseline.round(3))

        print("\n" + "=" * 110 + "\nCROSS-PERIOD CONSISTENCY\n" + "=" * 110)
        _display(consistency.sort_values(["consistently_toxic", "same_direction_all_3"], ascending=False).round(3))

        print("\n" + "=" * 110 + "\nWINDOW-LEVEL FILL INTENSITY TEST\n" + "=" * 110)
        if fill_tests.empty:
            print(
                "No historical fill test: April-July minute5 files contain M5 quotes/outcomes, not 15-second "
                "order-book/trade replay. Pass live_sessions=[...] for clean prospective execution data."
            )
        else:
            _display(fill_tests.round(4))

        print("\n" + "=" * 110 + "\nCORRELATED EXPOSURE / BREADTH\n" + "=" * 110)
        _display(breadth[[
            "period", "signals", "windows", "accuracy_pct", "signal_edge_c", "fill_rate_pct", "filled_edge_c"
        ]].round(3))

    return {
        "signals": signals,
        "windows": windows,
        "coverage": coverage,
        "baseline": baseline,
        "studies": studies,
        "consistency": consistency,
        "fill_tests": fill_tests,
        "breadth": breadth,
        "sources": source_df,
    }
