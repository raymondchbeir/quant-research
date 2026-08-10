from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .pre_m5_path_study import run_pre_m5_path_study

STUDY_VERSION = "PRE_M5_PNL_GRID_SEARCH_V1"

# We intentionally grid only the three continuous pre-M5 risk features that showed
# the clearest separation in the Aug-10 forensic study. Dominant-move share is kept
# as a descriptive diagnostic because, with small window breadth, it is highly discrete.
GRID_FEATURES = (
    "max_mid_path_length_c",
    "max_mid_range_c",
    "max_mid_rv_c",
)

PAIR_LABELS = {
    ("max_mid_path_length_c", "max_mid_range_c"): ("Max path length (c)", "Max range (c)"),
    ("max_mid_path_length_c", "max_mid_rv_c"): ("Max path length (c)", "Max RV (c)"),
    ("max_mid_range_c", "max_mid_rv_c"): ("Max range (c)", "Max RV (c)"),
}


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _max_drawdown(window_pnl: pd.Series) -> float:
    x = pd.to_numeric(window_pnl, errors="coerce").fillna(0.0).to_numpy(float)
    if len(x) == 0:
        return np.nan
    equity = np.cumsum(x)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    dd = np.r_[0.0, equity] - peak
    return float(dd.min())


def _threshold_grid(x: pd.Series, points=13):
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.array([], dtype=float)
    # Quantile-spaced thresholds give roughly even occupancy rather than wasting
    # cells in empty tails. Still fully in-sample: this is diagnostic, not validation.
    q = np.linspace(0.05, 0.95, int(points))
    vals = np.quantile(x, q)
    vals = np.unique(np.round(vals, 6))
    return vals.astype(float)


def _prepare_sample(study, min_path_coverage_pct=75.0, cutoff_utc=None):
    windows = study["window_paths"].copy()
    contracts = study["contract_paths"].copy()

    windows["decision_time"] = pd.to_datetime(windows["decision_time"], utc=True, errors="coerce")
    contracts["decision_time"] = pd.to_datetime(contracts["decision_time"], utc=True, errors="coerce")

    if cutoff_utc is not None:
        cutoff = pd.to_datetime(cutoff_utc, utc=True)
        windows = windows[windows["decision_time"] <= cutoff].copy()
        contracts = contracts[contracts["decision_time"] <= cutoff].copy()

    windows = windows[
        windows["decision_time"].notna()
        & windows["execution_complete"].fillna(False)
        & (pd.to_numeric(windows["filled_assets"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(windows["path_complete_share_pct"], errors="coerce") >= float(min_path_coverage_pct))
    ].copy()

    for feature in GRID_FEATURES + ("m1_m5_dominant_move_share",):
        windows[feature] = pd.to_numeric(windows[feature], errors="coerce")
    windows["signals"] = pd.to_numeric(windows["signals"], errors="coerce")
    windows["actual_pnl"] = pd.to_numeric(windows["actual_pnl"], errors="coerce").fillna(0.0)

    keys = windows[["session", "decision_time"]].drop_duplicates()
    contracts = contracts.merge(keys, on=["session", "decision_time"], how="inner")
    contracts["entry_fill_qty"] = pd.to_numeric(contracts["entry_fill_qty"], errors="coerce").fillna(0.0)
    contracts["actual_pnl"] = pd.to_numeric(contracts["actual_pnl"], errors="coerce").fillna(0.0)
    contracts["pnl_per_contract"] = np.where(
        contracts["entry_fill_qty"] > 1e-12,
        contracts["actual_pnl"] / contracts["entry_fill_qty"],
        0.0,
    )

    return (
        windows.sort_values(["decision_time", "session"]).reset_index(drop=True),
        contracts.sort_values(["decision_time", "session", "ticker"]).reset_index(drop=True),
    )


def _score_rule(windows, contracts, feature_x, threshold_x, feature_y, threshold_y,
                reduced_qty=1.0, high_breadth_min=3):
    w = windows.copy()
    w["flagged"] = (
        (w["signals"] >= int(high_breadth_min))
        & (w[feature_x] >= float(threshold_x))
        & (w[feature_y] >= float(threshold_y))
    )

    c = contracts.merge(
        w[["session", "decision_time", "flagged"]],
        on=["session", "decision_time"],
        how="left",
    )
    c["flagged"] = c["flagged"].fillna(False)
    c["target_qty"] = np.where(c["flagged"], float(reduced_qty), 3.0)
    c["accepted_qty"] = np.minimum(c["entry_fill_qty"], c["target_qty"])
    c["strategy_pnl"] = c["accepted_qty"] * c["pnl_per_contract"]

    sw = c.groupby(["session", "decision_time"], as_index=False).agg(
        strategy_pnl=("strategy_pnl", "sum"),
        strategy_contracts=("accepted_qty", "sum"),
    )
    w = w.merge(sw, on=["session", "decision_time"], how="left")
    w["strategy_pnl"] = w["strategy_pnl"].fillna(0.0)
    w["strategy_contracts"] = w["strategy_contracts"].fillna(0.0)
    w["delta_vs_q3"] = w["strategy_pnl"] - w["actual_pnl"]

    by_session = w.groupby("session").agg(
        q3_pnl=("actual_pnl", "sum"),
        strategy_pnl=("strategy_pnl", "sum"),
        delta=("delta_vs_q3", "sum"),
    )

    deltas = by_session["delta"].to_numpy(float) if len(by_session) else np.array([], dtype=float)
    return {
        "strategy_pnl": float(w["strategy_pnl"].sum()),
        "q3_pnl": float(w["actual_pnl"].sum()),
        "delta_vs_q3": float(w["delta_vs_q3"].sum()),
        "max_drawdown": _max_drawdown(w.sort_values(["decision_time", "session"])["strategy_pnl"]),
        "q3_max_drawdown": _max_drawdown(w.sort_values(["decision_time", "session"])["actual_pnl"]),
        "worst_window_pnl": float(w["strategy_pnl"].min()),
        "flagged_windows": int(w["flagged"].sum()),
        "flagged_filled_assets": int(w.loc[w["flagged"], "filled_assets"].sum()),
        "strategy_contracts": float(w["strategy_contracts"].sum()),
        "q3_contracts": float(pd.to_numeric(contracts["entry_fill_qty"], errors="coerce").fillna(0.0).sum()),
        "sessions": int(len(by_session)),
        "positive_sessions": int((by_session["delta"] > 1e-12).sum()) if len(by_session) else 0,
        "nonnegative_sessions": int((by_session["delta"] >= -1e-12).sum()) if len(by_session) else 0,
        "worst_session_delta": float(np.min(deltas)) if len(deltas) else np.nan,
        "best_session_delta": float(np.max(deltas)) if len(deltas) else np.nan,
    }


def _surface(windows, contracts, feature_x, feature_y, x_grid, y_grid,
             reduced_qty=1.0, high_breadth_min=3):
    rows = []
    for y in y_grid:
        for x in x_grid:
            score = _score_rule(
                windows, contracts,
                feature_x, x, feature_y, y,
                reduced_qty=reduced_qty,
                high_breadth_min=high_breadth_min,
            )
            rows.append({
                "feature_x": feature_x,
                "feature_y": feature_y,
                "threshold_x": float(x),
                "threshold_y": float(y),
                "reduced_qty": float(reduced_qty),
                **score,
            })
    return pd.DataFrame(rows)


def _surface_diagnostics(surface: pd.DataFrame):
    if surface.empty:
        return {}
    best = surface.sort_values(
        ["delta_vs_q3", "worst_session_delta", "max_drawdown"],
        ascending=[False, False, False],
    ).iloc[0]

    total = len(surface)
    positive = int((surface["delta_vs_q3"] > 0).sum())
    all_sessions_positive = int(
        ((surface["sessions"] > 0) & (surface["positive_sessions"] == surface["sessions"])).sum()
    )

    xs = np.sort(surface["threshold_x"].unique())
    ys = np.sort(surface["threshold_y"].unique())
    ix = int(np.where(np.isclose(xs, best["threshold_x"]))[0][0])
    iy = int(np.where(np.isclose(ys, best["threshold_y"]))[0][0])
    x_near = xs[max(0, ix - 1): min(len(xs), ix + 2)]
    y_near = ys[max(0, iy - 1): min(len(ys), iy + 2)]
    neigh = surface[
        surface["threshold_x"].isin(x_near)
        & surface["threshold_y"].isin(y_near)
    ]

    return {
        "best_threshold_x": float(best["threshold_x"]),
        "best_threshold_y": float(best["threshold_y"]),
        "best_delta_vs_q3": float(best["delta_vs_q3"]),
        "best_strategy_pnl": float(best["strategy_pnl"]),
        "best_flagged_windows": int(best["flagged_windows"]),
        "best_worst_session_delta": float(best["worst_session_delta"]),
        "positive_cells_pct": 100.0 * positive / total if total else np.nan,
        "all_sessions_positive_cells_pct": 100.0 * all_sessions_positive / total if total else np.nan,
        "best_neighborhood_cells": int(len(neigh)),
        "best_neighborhood_positive_pct": 100.0 * (neigh["delta_vs_q3"] > 0).mean() if len(neigh) else np.nan,
        "best_neighborhood_median_delta": float(neigh["delta_vs_q3"].median()) if len(neigh) else np.nan,
        "best_neighborhood_min_delta": float(neigh["delta_vs_q3"].min()) if len(neigh) else np.nan,
    }


def _plot_surface(surface: pd.DataFrame, value_col="delta_vs_q3", title=None):
    import matplotlib.pyplot as plt

    p = surface.pivot(index="threshold_y", columns="threshold_x", values=value_col).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    im = ax.imshow(p.to_numpy(float), aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(p.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in p.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(p.index)))
    ax.set_yticklabels([f"{y:.1f}" for y in p.index])
    fx = surface["feature_x"].iloc[0]
    fy = surface["feature_y"].iloc[0]
    labels = PAIR_LABELS.get((fx, fy), (fx, fy))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title or f"{value_col}: {fx} vs {fy}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    plt.show()


def run_pre_m5_pnl_grid_search(
    session_dirs=None,
    study=None,
    reduced_qtys=(1.0, 2.0),
    high_breadth_min=3,
    grid_points=13,
    min_path_coverage_pct=75.0,
    cutoff_utc=None,
    plot=True,
    show=True,
):
    """Development-only PnL stability search for pre-M5 risk thresholds.

    Rule family for each 2D surface:
        if signals >= high_breadth_min AND feature_x >= X AND feature_y >= Y:
            accept at most reduced_qty contracts per filled asset;
        else:
            keep the observed Q3 fill quantity (up to 3).

    This is an in-sample search over August recorder windows. Its purpose is to inspect
    topology/stability, not to validate a trading rule. A broad plateau that also helps
    each recorded session is more credible than an isolated maximum, but only future
    frozen data can validate a selected rule.
    """
    if study is None:
        if session_dirs is None:
            raise ValueError("Pass either study=<run_pre_m5_path_study result> or session_dirs=[...].")
        study = run_pre_m5_path_study(
            session_dirs=session_dirs,
            source="auto",
            full_book_fallback=True,
            max_anchor_age_sec=90.0,
            min_window_path_coverage_pct=min_path_coverage_pct,
            settle_missing=True,
            show=False,
        )

    if cutoff_utc is None:
        cutoff_utc = pd.Timestamp.now(tz="UTC")
    else:
        cutoff_utc = pd.to_datetime(cutoff_utc, utc=True)

    windows, contracts = _prepare_sample(
        study,
        min_path_coverage_pct=min_path_coverage_pct,
        cutoff_utc=cutoff_utc,
    )
    if windows.empty:
        raise RuntimeError("No execution-complete filled windows with usable M1-M5 paths.")

    hb = windows[windows["signals"] >= int(high_breadth_min)].copy()
    if len(hb) < 4:
        raise RuntimeError(f"Only {len(hb)} high-breadth windows; too few for a useful grid.")

    grids = {f: _threshold_grid(hb[f], points=grid_points) for f in GRID_FEATURES}
    surfaces = {}
    diagnostics = []

    for qty in reduced_qtys:
        qty = float(qty)
        if qty <= 0 or qty > 3:
            raise ValueError("reduced_qtys must be >0 and <=3.")
        for fx, fy in combinations(GRID_FEATURES, 2):
            key = f"Q{qty:g}_{fx}__{fy}"
            s = _surface(
                windows, contracts,
                fx, fy,
                grids[fx], grids[fy],
                reduced_qty=qty,
                high_breadth_min=high_breadth_min,
            )
            surfaces[key] = s
            d = _surface_diagnostics(s)
            diagnostics.append({
                "surface": key,
                "reduced_qty": qty,
                "feature_x": fx,
                "feature_y": fy,
                **d,
            })

    diagnostics = pd.DataFrame(diagnostics).sort_values(
        ["all_sessions_positive_cells_pct", "positive_cells_pct", "best_neighborhood_min_delta", "best_delta_vs_q3"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    baseline = pd.DataFrame([{
        "analysis_cutoff_utc": cutoff_utc,
        "sessions": windows["session"].nunique(),
        "filled_windows": len(windows),
        "high_breadth_windows": len(hb),
        "filled_assets": int(windows["filled_assets"].sum()),
        "q3_realized_pnl": float(windows["actual_pnl"].sum()),
        "q3_max_drawdown": _max_drawdown(windows.sort_values(["decision_time", "session"])["actual_pnl"]),
        "q3_worst_window": float(windows["actual_pnl"].min()),
    }])

    by_session = windows.groupby("session", as_index=False).agg(
        filled_windows=("decision_time", "size"),
        high_breadth_windows=("signals", lambda x: int((x >= high_breadth_min).sum())),
        q3_pnl=("actual_pnl", "sum"),
        worst_window=("actual_pnl", "min"),
    )

    dominance = hb.groupby("m1_m5_dominant_move_share", dropna=False).agg(
        windows=("decision_time", "size"),
        mean_q3_pnl=("actual_pnl", "mean"),
        total_q3_pnl=("actual_pnl", "sum"),
        negative_window_pct=("actual_pnl", lambda x: 100.0 * (x < 0).mean()),
    ).reset_index().sort_values("m1_m5_dominant_move_share")

    if show:
        print("=" * 118)
        print("M1->M5 DEVELOPMENT PNL GRID SEARCH — THRESHOLD STABILITY, NOT VALIDATION")
        print("=" * 118)
        print("Rule family: high-breadth window + BOTH plotted features above thresholds => reduce each filled asset from Q3 to Q1/Q2.")
        print("A broad positive plateau is encouraging; an isolated optimum is an overfit warning.")
        print("The same data chooses and scores thresholds, so even a beautiful plateau still needs future frozen OOS validation.")
        print("\nBASELINE SAMPLE")
        _display(baseline.round(4))
        print("\nBY SESSION")
        _display(by_session.round(4))
        print("\nSURFACE ROBUSTNESS SUMMARY")
        _display(diagnostics.round(4))
        print("\nDOMINANT-MOVE SHARE (DESCRIPTIVE; NOT GRID-OPTIMIZED)")
        _display(dominance.round(4))

    if plot:
        for _, row in diagnostics.iterrows():
            key = row["surface"]
            s = surfaces[key]
            qty = row["reduced_qty"]
            fx, fy = row["feature_x"], row["feature_y"]
            title = (
                f"Q3 -> Q{qty:g} in flagged high-breadth windows | Δ realized PnL vs Q3\n"
                f"{PAIR_LABELS.get((fx, fy), (fx, fy))[0]} × {PAIR_LABELS.get((fx, fy), (fx, fy))[1]}"
            )
            _plot_surface(s, value_col="delta_vs_q3", title=title)

    return {
        "version": STUDY_VERSION,
        "analysis_cutoff_utc": cutoff_utc,
        "baseline": baseline,
        "by_session": by_session,
        "dominance": dominance,
        "diagnostics": diagnostics,
        "surfaces": surfaces,
        "windows": windows,
        "contracts": contracts,
        "source_study": study,
    }
