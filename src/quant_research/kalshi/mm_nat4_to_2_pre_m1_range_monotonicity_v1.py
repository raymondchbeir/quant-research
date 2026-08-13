from __future__ import annotations

"""Single-feature development diagnostic for NAT4->2 pre-M1 max midpoint range.

No strategy replay and no threshold selection. Reads window_diagnostics.csv and
checks whether PnL deteriorates as pre_m0_m1_max_mid_range_c increases.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_oos_4c_compact_recorder_v2 as R

STUDY_VERSION = "NAT4_TO_2_PRE_M1_RANGE_MONOTONICITY_V1"
FEATURE = "pre_m0_m1_max_mid_range_c"
PNL = "net_pnl"
SEED = 20260813
N_PERM = 20000
EPS = 1e-12


def _wavg(df, col):
    if col not in df.columns or "fill_qty" not in df.columns or df.empty:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce")
    w = pd.to_numeric(df["fill_qty"], errors="coerce")
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def _dd(pnl):
    x = pd.Series(pd.to_numeric(pnl, errors="coerce")).dropna().reset_index(drop=True)
    if x.empty:
        return np.nan
    c = x.cumsum()
    peak = np.maximum.accumulate(np.r_[0.0, c.to_numpy()])[:-1]
    return float(np.min(c.to_numpy() - peak))


def _summary(z, label_col, label):
    pnl = pd.to_numeric(z[PNL], errors="coerce")
    return {
        label_col: label,
        "windows": len(z),
        "range_min_c": z[FEATURE].min(),
        "range_mean_c": z[FEATURE].mean(),
        "range_max_c": z[FEATURE].max(),
        "net_pnl": pnl.sum(),
        "pnl_per_window": pnl.mean(),
        "median_window_pnl": pnl.median(),
        "positive_window_pct": 100.0 * (pnl > 0).mean(),
        "worst_window_pnl": pnl.min(),
        "best_window_pnl": pnl.max(),
        "max_drawdown_chronological": _dd(pnl),
        "fill_qty": pd.to_numeric(z.get("fill_qty", np.nan), errors="coerce").sum(),
        "gross_capture": pd.to_numeric(z.get("gross_capture", np.nan), errors="coerce").sum(),
        "adverse_selection_to_m5": pd.to_numeric(z.get("adverse_selection_to_m5", np.nan), errors="coerce").sum(),
        "matched_roundtrip_pnl": pd.to_numeric(z.get("matched_roundtrip_pnl", np.nan), errors="coerce").sum(),
        "residual_inventory_mtm_pnl": pd.to_numeric(z.get("residual_inventory_mtm_pnl", np.nan), errors="coerce").sum(),
        "qw_markout_60s_c": _wavg(z, "qw_markout_60s_c"),
    }


def _perm_p(x, observed):
    rng = np.random.default_rng(SEED)
    xr = pd.Series(x[FEATURE]).rank().to_numpy(float)
    yr = pd.Series(x[PNL]).rank().to_numpy(float)
    xr -= xr.mean()
    yr -= yr.mean()
    denom_x = np.sqrt(np.sum(xr * xr))
    exceed = 0
    for _ in range(N_PERM):
        yp = rng.permutation(yr)
        den = denom_x * np.sqrt(np.sum(yp * yp))
        rho = np.sum(xr * yp) / den if den > EPS else np.nan
        if np.isfinite(rho) and abs(rho) >= abs(observed) - EPS:
            exceed += 1
    return (exceed + 1.0) / (N_PERM + 1.0)


def run_pre_m1_range_monotonicity(diagnostic_result_dir, output_dir=None, *, show=True):
    src = Path(diagnostic_result_dir).resolve()
    p = src / "window_diagnostics.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    w = pd.read_csv(p)
    if FEATURE not in w.columns or PNL not in w.columns:
        raise KeyError(f"Need columns {FEATURE!r} and {PNL!r}")

    w[FEATURE] = pd.to_numeric(w[FEATURE], errors="coerce")
    w[PNL] = pd.to_numeric(w[PNL], errors="coerce")
    good = np.isfinite(w[FEATURE]) & np.isfinite(w[PNL])
    missing = w[~good].copy()
    x = w[good].copy().sort_values("close_ts").reset_index(drop=True)
    if len(x) < 8:
        raise RuntimeError(f"Only {len(x)} usable windows")

    labels = ["Q1_CALMEST", "Q2", "Q3", "Q4_MOST_VOLATILE"]
    x["range_quartile"] = pd.qcut(x[FEATURE], 4, labels=labels)

    quartiles = pd.DataFrame([
        _summary(z.sort_values("close_ts"), "range_quartile", str(q))
        for q, z in x.groupby("range_quartile", observed=True, sort=True)
    ])

    y = x.sort_values(FEATURE).reset_index(drop=True)
    cut = len(y) // 2
    halves = pd.DataFrame([
        _summary(y.iloc[:cut].sort_values("close_ts"), "group", "CALMER_HALF"),
        _summary(y.iloc[cut:].sort_values("close_ts"), "group", "MORE_VOLATILE_HALF"),
    ])

    rho = float(x[[FEATURE, PNL]].corr(method="spearman").iloc[0, 1])
    pearson = float(x[[FEATURE, PNL]].corr(method="pearson").iloc[0, 1])
    p_perm = _perm_p(x, rho)

    loo_rows = []
    for i in x.index:
        z = x.drop(index=i)
        rr = float(z[[FEATURE, PNL]].corr(method="spearman").iloc[0, 1])
        r = x.loc[i]
        loo_rows.append({
            "removed_chronological_index": r.get("chronological_index", np.nan),
            "removed_close_time": r.get("close_time"),
            "removed_range_c": r[FEATURE],
            "removed_pnl": r[PNL],
            "spearman_after_removal": rr,
        })
    loo = pd.DataFrame(loo_rows)
    lv = pd.to_numeric(loo.spearman_after_removal, errors="coerce")

    qmeans = pd.to_numeric(quartiles.pnl_per_window, errors="coerce").to_numpy(float)
    monotone = bool(np.all(np.diff(qmeans) <= EPS))
    qrank = float(pd.DataFrame({"q": np.arange(len(qmeans)), "pnl": qmeans}).corr(method="spearman").iloc[0, 1])

    robustness = pd.DataFrame([{
        "usable_windows": len(x),
        "excluded_missing_windows": len(missing),
        "spearman_range_vs_pnl": rho,
        "pearson_range_vs_pnl": pearson,
        "permutation_two_sided_p": p_perm,
        "quartile_pnl_monotone_decreasing": monotone,
        "quartile_index_spearman": qrank,
        "loo_spearman_min": lv.min(),
        "loo_spearman_median": lv.median(),
        "loo_spearman_max": lv.max(),
        "loo_negative_pct": 100.0 * (lv < 0).mean(),
        "usable_total_pnl": x[PNL].sum(),
    }])

    cols = [
        "chronological_index", "close_time", "window_class", FEATURE, PNL,
        "gross_capture", "adverse_selection_to_m5", "matched_roundtrip_pnl",
        "residual_inventory_mtm_pnl", "fill_qty", "bid_fill_qty", "ask_fill_qty",
        "qw_markout_5s_c", "qw_markout_15s_c", "qw_markout_30s_c", "qw_markout_60s_c",
        "range_quartile",
    ]
    cols = [c for c in cols if c in y.columns]
    sorted_windows = y[cols].copy()

    out = Path(output_dir) if output_dir else R.PROJECT_ROOT / "results" / "kalshi_nat4_to_2_pre_m1_range_monotonicity" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    quartiles.to_csv(out / "range_quartile_summary.csv", index=False)
    halves.to_csv(out / "range_half_summary.csv", index=False)
    sorted_windows.to_csv(out / "windows_sorted_by_pre_m1_range.csv", index=False)
    loo.to_csv(out / "leave_one_out_spearman.csv", index=False)
    robustness.to_csv(out / "robustness_summary.csv", index=False)
    missing.to_csv(out / "excluded_missing_windows.csv", index=False)
    (out / "study_config.json").write_text(json.dumps({
        "study_version": STUDY_VERSION,
        "source_diagnostic_dir": str(src),
        "feature": FEATURE,
        "permutations": N_PERM,
        "seed": SEED,
        "strategy_changes": "none",
        "threshold_selection": "none",
        "status": "development diagnostic only",
    }, indent=2), encoding="utf-8")

    if show:
        r = robustness.iloc[0]
        print("\n" + "=" * 132)
        print("NAT4->2 PRE-M1 MAX-RANGE MONOTONICITY — SINGLE-FEATURE DEVELOPMENT DIAGNOSTIC")
        print("=" * 132)
        print(f"usable={int(r.usable_windows)} | missing={int(r.excluded_missing_windows)} | usable pnl=${r.usable_total_pnl:+.4f}")
        print(f"range vs pnl: Spearman={r.spearman_range_vs_pnl:+.4f} | Pearson={r.pearson_range_vs_pnl:+.4f} | permutation p={r.permutation_two_sided_p:.4f}")
        print(f"quartile PnL/window monotone decreasing={bool(r.quartile_pnl_monotone_decreasing)} | quartile-rank rho={r.quartile_index_spearman:+.4f}")
        print(f"LOO Spearman min/median/max={r.loo_spearman_min:+.4f}/{r.loo_spearman_median:+.4f}/{r.loo_spearman_max:+.4f} | negative={r.loo_negative_pct:.1f}%")
        print("\nRANGE QUARTILES — CALMEST -> MOST VOLATILE")
        print(quartiles.round(4).to_string(index=False))
        print("\nCALMER HALF VS MORE-VOLATILE HALF")
        print(halves.round(4).to_string(index=False))
        print("\nEVERY USABLE WINDOW SORTED BY PRE-M1 MAX RANGE")
        print(sorted_windows.round(4).to_string(index=False))
        if len(missing):
            mc = [c for c in ["chronological_index", "close_time", PNL, FEATURE] if c in missing.columns]
            print("\nEXCLUDED FOR MISSING PRE-M1 RANGE")
            print(missing[mc].round(4).to_string(index=False))
        print("\nGUARDRAIL: this describes shape only; it does not choose a cutoff or change NAT4->2.")
        print("Outputs:", out)
        print("=" * 132)

    return {
        "output_dir": out,
        "robustness": robustness,
        "quartiles": quartiles,
        "halves": halves,
        "sorted_windows": sorted_windows,
        "leave_one_out": loo,
        "excluded_missing": missing,
    }
