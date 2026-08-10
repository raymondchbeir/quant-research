from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_SESSION_RE = re.compile(r"^\d{8}_\d{6}$")

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

ALIASES = {
    "ticker": ["ticker", "market_ticker"],
    "decision_time": ["decision_time", "signal_time", "m5_time", "entry_decision_time", "timestamp", "time"],
    "direction": ["direction", "side", "held_side", "signal_direction", "prediction"],
    "result": ["result", "settlement", "settled_result", "outcome", "winner", "market_result"],
    "correct": ["correct", "is_correct", "won", "win", "signal_correct"],
    "midpoint": ["midpoint", "mid", "m5_midpoint", "yes_mid", "yes_midpoint"],
    "spread_c": ["spread_c", "spread_cent", "spread_cents", "yes_spread_c", "quoted_spread_c"],
    "entry_price": ["entry_price", "maker_price", "quote_price", "passive_price", "held_entry"],
    "btc_return": ["btc_return", "btc_return_15m", "btc15", "btc_ret_15m", "btc_15m_return", "btc15_return"],
    "btc_opposition": ["btc_opposition", "opposition", "btc_opp", "is_opposition"],
    "entry_fill_qty": ["entry_fill_qty", "fill_qty", "filled_qty", "maker_fill_qty", "qty_filled"],
    "filled": ["filled", "did_fill", "maker_filled", "any_fill"],
    "entry_queue": ["entry_queue", "queue", "queue_ahead", "initial_queue"],
    "actual_pnl": ["realized_pnl", "actual_pnl", "pnl", "maker_pnl"],
    "depth_c": ["depth_c", "depth", "entry_depth_c", "offset_c", "improvement_c"],
    "lifetime_sec": ["lifetime_sec", "entry_lifetime_sec", "cancel_seconds", "lifetime_seconds"],
}

NAME_HINTS = (
    "histor", "replay", "depth", "decision", "shadow", "signal", "maker",
    "trade", "result", "exact_clock", "sweep", "study", "market_level",
)
VALID_EXT = {".csv", ".parquet", ".feather", ".pkl", ".pickle"}
SUMMARY_PENALTIES = ("summary", "bootstrap", "curve", "breakdown", "daily", "consistency", "status", "cash_event", "skipped")
DETAIL_BONUSES = ("signal", "market_level", "replay", "detail", "accepted")


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _norm_name(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def _first_col(df, names):
    cmap = {_norm_name(c): c for c in df.columns}
    for name in names:
        if _norm_name(name) in cmap:
            return cmap[_norm_name(name)]
    return None


def _resolve_columns(df):
    return {key: _first_col(df, aliases) for key, aliases in ALIASES.items()}


def _read_table(path: Path, nrows=None):
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if ext == ".parquet":
        df = pd.read_parquet(path)
        return df if nrows is None else df.head(nrows)
    if ext == ".feather":
        df = pd.read_feather(path)
        return df if nrows is None else df.head(nrows)
    df = pd.read_pickle(path)
    return df if nrows is None else df.head(nrows)


def _as_num(s):
    return pd.to_numeric(s, errors="coerce")


def _as_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def _boolish(s):
    if s.dtype == bool:
        return s.fillna(False)
    x = s.astype(str).str.strip().str.lower()
    return x.isin({"true", "1", "yes", "y", "filled", "win", "won"})


def _period_for_time(t):
    if pd.isna(t):
        return None
    for label, (start, end) in PERIODS.items():
        if pd.Timestamp(start, tz="UTC") <= t < pd.Timestamp(end, tz="UTC"):
            return label
    return None


def _discover_candidate_paths(search_roots):
    out = []
    seen = set()
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not RAW_SESSION_RE.match(d)]
            current = Path(current_root)
            for filename in files:
                p = current / filename
                if p.suffix.lower() not in VALID_EXT:
                    continue
                text = str(p).lower()
                if not any(h in text for h in NAME_HINTS):
                    continue
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)
    return out


def _profile_candidate(path: Path):
    try:
        sample = _read_table(path, nrows=2500)
    except Exception:
        return None
    cols = _resolve_columns(sample)
    required = ["ticker", "decision_time", "direction", "entry_price"]
    core = sum(cols[k] is not None for k in required)
    outcome_ok = cols["result"] is not None or cols["correct"] is not None
    if core < 4 or not outcome_ok:
        return None
    bonus_keys = ["midpoint", "spread_c", "btc_return", "btc_opposition", "entry_fill_qty", "filled", "entry_queue", "actual_pnl", "depth_c", "lifetime_sec"]
    bonus = sum(cols[k] is not None for k in bonus_keys)
    times = _as_utc(sample[cols["decision_time"]]).dropna()
    periods = sorted({x for x in times.map(_period_for_time).dropna().unique()})
    text = str(path).lower()
    path_score = sum(2 for x in DETAIL_BONUSES if x in text) - sum(3 for x in SUMMARY_PENALTIES if x in text)
    return {
        "path": str(path),
        "core_score": core + int(outcome_ok),
        "bonus_score": bonus,
        "path_score": path_score,
        "sample_rows": len(sample),
        "sample_start": times.min() if len(times) else pd.NaT,
        "sample_end": times.max() if len(times) else pd.NaT,
        "sample_periods": ",".join(periods),
        "columns": cols,
    }


def find_window_toxicity_sources(project_root=None, show=True):
    project_root = Path(project_root or PROJECT_ROOT)
    search_roots = [project_root / "data", project_root / "results"]
    print("Searching research outputs...")
    print("Roots:")
    for root in search_roots:
        print(" ", root)
    paths = _discover_candidate_paths(search_roots)
    print(f"Candidate tabular files: {len(paths)}")
    profiles = []
    for i, path in enumerate(paths, 1):
        prof = _profile_candidate(path)
        if prof is not None:
            profiles.append(prof)
        if i % 25 == 0:
            print(f"  profiled {i}/{len(paths)}")
    if not profiles:
        return pd.DataFrame()
    df = pd.DataFrame(profiles)
    df["rank"] = df["core_score"] * 100 + df["bonus_score"] * 10 + df["path_score"]
    df = df.sort_values(["rank", "sample_rows"], ascending=False).reset_index(drop=True)
    if show:
        _display(df[["path", "core_score", "bonus_score", "path_score", "sample_rows", "sample_start", "sample_end", "sample_periods"]].head(30))
    return df


def _normalize_source(path: Path):
    raw = _read_table(path)
    cols = _resolve_columns(raw)
    out = pd.DataFrame(index=raw.index)
    out["ticker"] = raw[cols["ticker"]].astype(str)
    out["decision_time_raw"] = _as_utc(raw[cols["decision_time"]])
    # Critical correction: contracts evaluated in the same exact M5 minute are one independent window.
    out["decision_time"] = out["decision_time_raw"].dt.floor("min")
    out["direction"] = raw[cols["direction"]].astype(str).str.upper().str.strip()
    out["entry_price"] = _as_num(raw[cols["entry_price"]])
    med = out["entry_price"].median()
    if np.isfinite(med) and med > 1.5:
        out["entry_price"] /= 100.0

    if cols["result"] is not None:
        r = raw[cols["result"]].astype(str).str.upper().str.strip()
        out["result_final"] = np.where(r.isin(["YES", "Y", "1", "TRUE"]), "YES", np.where(r.isin(["NO", "N", "0", "FALSE"]), "NO", None))
    elif cols["correct"] is not None:
        correct = _boolish(raw[cols["correct"]])
        out["result_final"] = np.where(correct, out["direction"], np.where(out["direction"].eq("YES"), "NO", "YES"))
    else:
        out["result_final"] = None

    for key in ["midpoint", "spread_c", "btc_return", "entry_queue", "entry_fill_qty", "actual_pnl", "depth_c", "lifetime_sec"]:
        out[key] = _as_num(raw[cols[key]]) if cols[key] is not None else np.nan
    if out["midpoint"].notna().any() and out["midpoint"].median() > 1.5:
        out["midpoint"] /= 100.0
    if out["btc_return"].notna().any() and out["btc_return"].abs().median() > 0.1:
        out["btc_return"] /= 10000.0

    if cols["btc_opposition"] is not None:
        out["btc_opposition"] = _boolish(raw[cols["btc_opposition"]])
    else:
        out["btc_opposition"] = np.nan
    if cols["entry_fill_qty"] is not None:
        out["filled"] = out["entry_fill_qty"].fillna(0) > 1e-12
    elif cols["filled"] is not None:
        out["filled"] = _boolish(raw[cols["filled"]])
        out["entry_fill_qty"] = np.where(out["filled"], 3.0, 0.0)
    else:
        out["filled"] = np.nan
    out["source_file"] = str(path)
    return out


def _select_sources(profiles, explicit_sources=None):
    if explicit_sources:
        paths = [Path(x) for x in explicit_sources]
        return {label: paths for label in PERIODS}
    if profiles.empty:
        raise RuntimeError("No usable signal-level historical tables found.")

    # Profile dates from full selected-looking tables so an August-only table cannot masquerade as history.
    candidates = []
    for _, row in profiles.head(20).iterrows():
        path = Path(row["path"])
        try:
            z = _normalize_source(path)
        except Exception:
            continue
        present = set(z["decision_time"].map(_period_for_time).dropna())
        candidates.append((path, present, float(row["rank"]), len(z)))

    selected = {}
    for period in PERIODS:
        eligible = [c for c in candidates if period in c[1]]
        if eligible:
            eligible.sort(key=lambda x: (len(x[1]), x[2], x[3]), reverse=True)
            selected[period] = [eligible[0][0]]
        else:
            selected[period] = []
    return selected


def _prepare_signals(selected_sources, strict_history=True):
    frames = []
    selection_rows = []
    for period, paths in selected_sources.items():
        if not paths:
            continue
        start, end = PERIODS[period]
        for path in paths:
            z = _normalize_source(Path(path))
            z = z[(z["decision_time"] >= pd.Timestamp(start, tz="UTC")) & (z["decision_time"] < pd.Timestamp(end, tz="UTC"))].copy()
            if z.empty:
                continue
            z["period"] = period
            selection_rows.append({"period": period, "source": str(path), "rows_before_filters": len(z)})
            frames.append(z)
    if not frames:
        raise RuntimeError("Selected source tables contained no Apr-Aug rows.")
    hist = pd.concat(frames, ignore_index=True)
    hist["series"] = hist["ticker"].str.split("-").str[0]
    hist = hist[hist["series"].isin(FROZEN_SERIES)].copy()
    hist = hist[
        hist["decision_time"].notna()
        & hist["direction"].isin(["YES", "NO"])
        & hist["result_final"].isin(["YES", "NO"])
        & hist["entry_price"].between(0.01, 0.99, inclusive="both")
    ].copy()

    # If a source is a depth/lifetime sweep, isolate the frozen -3c / 15s branch.
    if hist["depth_c"].notna().any():
        hist = hist[np.isclose(hist["depth_c"], 3.0, atol=1e-9) | hist["depth_c"].isna()].copy()
    if hist["lifetime_sec"].notna().any():
        hist = hist[np.isclose(hist["lifetime_sec"], 15.0, atol=1e-9) | hist["lifetime_sec"].isna()].copy()
    if hist["spread_c"].notna().any():
        hist = hist[(hist["spread_c"] <= 2.0 + 1e-12) | hist["spread_c"].isna()].copy()

    opp_known = hist["btc_opposition"].notna()
    if opp_known.any():
        hist = hist[(~opp_known) | hist["btc_opposition"].astype(bool)].copy()
    elif hist["btc_return"].notna().any():
        sign = np.where(hist["direction"].eq("YES"), 1.0, -1.0)
        valid = hist["btc_return"].isna() | (hist["btc_return"] * sign < 0)
        hist = hist[valid].copy()

    hist = hist.sort_values(["period", "decision_time", "ticker"]).drop_duplicates(
        ["period", "decision_time", "ticker", "direction", "entry_price"], keep="first"
    ).reset_index(drop=True)

    required_history = {"APR_DISCOVERY", "MAY_JUN_VALID", "JUL_HOLDOUT"}
    present = set(hist["period"].dropna())
    missing = sorted(required_history - present)
    if strict_history and missing:
        raise RuntimeError(
            "Historical window study is missing required pre-August periods: " + ", ".join(missing) + ".\n"
            "The script is refusing to label an August-only table as an April-July study.\n"
            "Run find_window_toxicity_sources() and pass explicit source paths if your historical table lives elsewhere."
        )
    return hist, pd.DataFrame(selection_rows)


def _build_windows(hist):
    hist = hist.copy()
    hist["correct"] = hist["direction"].eq(hist["result_final"])
    hist["signal_edge"] = np.where(hist["correct"], 1.0 - hist["entry_price"], -hist["entry_price"])
    hist["hypo_pnl_3ct"] = 3.0 * hist["signal_edge"]
    hist["mid_distance_50_c"] = 100.0 * (hist["midpoint"] - 0.5).abs()
    hist["btc_abs_bp"] = 10000.0 * hist["btc_return"].abs()

    rows = []
    for (period, dt), g in hist.groupby(["period", "decision_time"], dropna=False):
        counts = g["direction"].value_counts()
        same_n = int(counts.max()) if len(counts) else 0
        fill_known = g["filled"].notna().any()
        if fill_known:
            fm = g["filled"].fillna(False).astype(bool)
            fills = int(fm.sum())
            fill_rate = fills / len(g)
            filled_edge = g.loc[fm, "signal_edge"].mean() if fills else np.nan
            actual_pnl = g.loc[fm, "actual_pnl"].fillna(0).sum()
        else:
            fills = fill_rate = filled_edge = actual_pnl = np.nan
        rows.append({
            "period": period, "decision_time": dt, "signals": len(g), "same_direction_n": same_n,
            "same_direction_share": same_n / len(g), "unanimous_direction": len(counts) == 1,
            "accuracy": g["correct"].mean(), "signal_edge": g["signal_edge"].mean(),
            "signal_pnl_3ct": g["hypo_pnl_3ct"].sum(),
            "avg_mid_distance_c": g["mid_distance_50_c"].mean(),
            "mid_dispersion_c": 100.0 * g["midpoint"].std(ddof=0) if g["midpoint"].notna().any() else np.nan,
            "avg_spread_c": g["spread_c"].mean(), "avg_entry_c": 100.0 * g["entry_price"].mean(),
            "btc_abs_bp": g["btc_abs_bp"].mean(), "avg_queue": g["entry_queue"].mean(),
            "zero_queue_share": (g["entry_queue"].fillna(np.inf) <= 1e-12).mean() if g["entry_queue"].notna().any() else np.nan,
            "fills": fills, "fill_rate": fill_rate, "filled_edge": filled_edge, "actual_pnl": actual_pnl,
        })
    return hist, pd.DataFrame(rows).sort_values(["period", "decision_time"]).reset_index(drop=True)


def _feature_table(windows, feature, bins=None, labels=None, categorical=False):
    z = windows.copy()
    z["bucket"] = z[feature].astype(str) if categorical else pd.cut(z[feature], bins=bins, labels=labels, include_lowest=True)
    out = z.dropna(subset=["bucket"]).groupby(["period", "bucket"], observed=True).agg(
        windows=("decision_time", "size"), signals=("signals", "sum"), avg_accuracy=("accuracy", "mean"),
        avg_signal_edge=("signal_edge", "mean"), total_signal_pnl=("signal_pnl_3ct", "sum"),
        avg_fill_rate=("fill_rate", "mean"), avg_filled_edge=("filled_edge", "mean"), actual_pnl=("actual_pnl", "sum"),
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
    p = (1 + np.sum(sims <= actual)) / (n + 1)
    return {
        "windows": len(z), "actual_fill_weighted_edge_c": 100 * actual,
        "random_fill_edge_c": 100 * sims.mean(), "window_selection_penalty_c": 100 * (actual - sims.mean()),
        "p_value": p,
    }


def run_window_toxicity_study(project_root=None, sources=None, strict_history=True, permutation_n=20000, show=True):
    project_root = Path(project_root or PROJECT_ROOT)
    profiles = find_window_toxicity_sources(project_root=project_root, show=show)
    selected = _select_sources(profiles, explicit_sources=sources)

    print("\nSelected source by period:")
    for period, paths in selected.items():
        print(f"  {period:<16} {', '.join(str(p) for p in paths) if paths else 'MISSING'}")

    hist, source_selection = _prepare_signals(selected, strict_history=strict_history)
    hist, windows = _build_windows(hist)

    coverage = hist.groupby("period").agg(
        signals=("ticker", "size"), windows=("decision_time", "nunique"), accuracy=("correct", "mean"), signal_edge=("signal_edge", "mean")
    )
    coverage["accuracy_pct"] = 100 * coverage["accuracy"]
    coverage["signal_edge_c"] = 100 * coverage["signal_edge"]

    baseline_rows = []
    for period, g in hist.groupby("period"):
        row = {
            "period": period, "signals": len(g), "windows": g["decision_time"].nunique(),
            "accuracy_pct": 100 * g["correct"].mean(), "signal_edge_c": 100 * g["signal_edge"].mean(),
            "hypo_all_pnl_3ct": g["hypo_pnl_3ct"].sum(),
        }
        if g["filled"].notna().any():
            f = g[g["filled"].fillna(False)]
            u = g[~g["filled"].fillna(False)]
            row.update({
                "fill_rate_pct": 100 * len(f) / len(g),
                "filled_accuracy_pct": 100 * f["correct"].mean() if len(f) else np.nan,
                "unfilled_accuracy_pct": 100 * u["correct"].mean() if len(u) else np.nan,
                "filled_edge_c": 100 * f["signal_edge"].mean() if len(f) else np.nan,
                "actual_pnl": f["actual_pnl"].fillna(0).sum(),
            })
        baseline_rows.append(row)
    baseline = pd.DataFrame(baseline_rows).set_index("period")

    studies = {
        "SIGNAL_BREADTH": _feature_table(windows, "signals", [0, 1, 2, 4, 8], ["1", "2", "3-4", "5+"]),
        "SAME_DIRECTION_N": _feature_table(windows, "same_direction_n", [0, 1, 2, 4, 8], ["1", "2", "3-4", "5+"]),
        "UNANIMOUS": _feature_table(windows, "unanimous_direction", categorical=True),
    }
    if windows["btc_abs_bp"].notna().any():
        studies["BTC_ABS"] = _feature_table(windows, "btc_abs_bp", [-1e-9, 5, 10, 15, 25, np.inf], ["0-5bp", "5-10bp", "10-15bp", "15-25bp", "25bp+"])
    if windows["avg_mid_distance_c"].notna().any():
        studies["MIDPOINT_EXTREMITY"] = _feature_table(windows, "avg_mid_distance_c", [-1e-9, 5, 10, 20, 30, 50], ["0-5c", "5-10c", "10-20c", "20-30c", "30c+"])
    if windows["zero_queue_share"].notna().any():
        studies["ZERO_QUEUE_SHARE"] = _feature_table(windows, "zero_queue_share", [-1e-9, .001, .25, .50, .75, 1.001], ["0%", "0-25%", "25-50%", "50-75%", "75-100%"])

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
        print(f"Running fill-intensity permutation: {period} ({len(z)} windows)...")
        res = _fill_intensity_test(z, n=permutation_n)
        if res is not None:
            perm_rows.append({"period": period, **res})
    fill_tests = pd.DataFrame(perm_rows).set_index("period") if perm_rows else pd.DataFrame()

    breadth = windows.groupby(["period", "signals"]).agg(
        windows=("decision_time", "size"), accuracy=("accuracy", "mean"), signal_edge=("signal_edge", "mean"),
        fill_rate=("fill_rate", "mean"), filled_edge=("filled_edge", "mean")
    ).reset_index()
    breadth["accuracy_pct"] = 100 * breadth["accuracy"]
    breadth["signal_edge_c"] = 100 * breadth["signal_edge"]
    breadth["fill_rate_pct"] = 100 * breadth["fill_rate"]
    breadth["filled_edge_c"] = 100 * breadth["filled_edge"]

    if show:
        print("\n" + "=" * 110 + "\nDATASET COVERAGE\n" + "=" * 110)
        _display(coverage[["signals", "windows", "accuracy_pct", "signal_edge_c"]].round(3))
        print("\n" + "=" * 110 + "\nBASELINE BY PERIOD\n" + "=" * 110)
        _display(baseline.round(3))
        print("\n" + "=" * 110 + "\nCROSS-PERIOD CONSISTENCY\n" + "=" * 110)
        _display(consistency.sort_values(["consistently_toxic", "same_direction_all_3"], ascending=False).round(3))
        print("\n" + "=" * 110 + "\nWINDOW-LEVEL FILL INTENSITY TEST BY PERIOD\n" + "=" * 110)
        _display(fill_tests.round(4) if not fill_tests.empty else pd.DataFrame({"note": ["No historical fill fields available"]}))
        print("\n" + "=" * 110 + "\nCORRELATED EXPOSURE / BREADTH\n" + "=" * 110)
        _display(breadth[["period", "signals", "windows", "accuracy_pct", "signal_edge_c", "fill_rate_pct", "filled_edge_c"]].round(3))

    return {
        "signals": hist, "windows": windows, "source_profiles": profiles,
        "source_selection": source_selection, "coverage": coverage, "baseline": baseline,
        "feature_studies": studies, "consistency": consistency, "fill_tests": fill_tests,
        "breadth": breadth,
    }
