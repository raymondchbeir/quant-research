from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .pre_m5_pnl_grid_search import PAIR_LABELS, _score_rule

STUDY_VERSION = "PRE_M5_GRID_ROBUSTNESS_V1"


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _window_key_frame(windows: pd.DataFrame) -> pd.DataFrame:
    return windows[["session", "decision_time"]].drop_duplicates().copy()


def _subset_contracts(contracts: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    keys = _window_key_frame(windows)
    return contracts.merge(keys, on=["session", "decision_time"], how="inner")


def _flagged_key_tuple(windows: pd.DataFrame, feature_x, threshold_x, feature_y, threshold_y,
                       high_breadth_min=3):
    m = (
        (pd.to_numeric(windows["signals"], errors="coerce") >= int(high_breadth_min))
        & (pd.to_numeric(windows[feature_x], errors="coerce") >= float(threshold_x))
        & (pd.to_numeric(windows[feature_y], errors="coerce") >= float(threshold_y))
    )
    g = windows.loc[m, ["session", "decision_time"]].copy()
    if g.empty:
        return tuple()
    g["decision_time"] = pd.to_datetime(g["decision_time"], utc=True, errors="coerce")
    g = g.sort_values(["decision_time", "session"])
    return tuple(
        f"{r.session}|{pd.Timestamp(r.decision_time).isoformat()}"
        for r in g.itertuples(index=False)
    )


def _score_surface(surface: pd.DataFrame, windows: pd.DataFrame, contracts: pd.DataFrame,
                   high_breadth_min=3) -> pd.DataFrame:
    if windows.empty:
        out = surface[["threshold_x", "threshold_y"]].copy()
        out["delta"] = np.nan
        return out

    fx = surface["feature_x"].iloc[0]
    fy = surface["feature_y"].iloc[0]
    qty = float(surface["reduced_qty"].iloc[0])
    rows = []
    for r in surface.itertuples(index=False):
        s = _score_rule(
            windows,
            contracts,
            fx,
            float(r.threshold_x),
            fy,
            float(r.threshold_y),
            reduced_qty=qty,
            high_breadth_min=high_breadth_min,
        )
        rows.append({
            "threshold_x": float(r.threshold_x),
            "threshold_y": float(r.threshold_y),
            "delta": float(s["delta_vs_q3"]),
            "strategy_pnl": float(s["strategy_pnl"]),
            "flagged_windows": int(s["flagged_windows"]),
        })
    return pd.DataFrame(rows)


def _plot_heatmap(surface: pd.DataFrame, value_col: str, title: str, label: str):
    import matplotlib.pyplot as plt

    p = surface.pivot(index="threshold_y", columns="threshold_x", values=value_col).sort_index()
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
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    plt.show()


def run_pre_m5_grid_robustness(
    grid_result,
    high_breadth_min=3,
    plot=True,
    plot_session_min=True,
    show=True,
):
    """Stress-test an already-computed pre-M5 PnL threshold grid.

    Uses the exact full-sample threshold lattice from ``grid_result``; it does not
    re-select quantile thresholds inside folds. Three diagnostics are produced:

    1) Leave-one-high-breadth-window-out (LOO): each high-breadth window is removed
       and every original grid cell is rescored.
    2) Session robustness: every original grid cell is rescored separately inside
       each recorder session.
    3) Unique flagged-set analysis: counts how many visually different threshold
       cells actually select the same set of high-breadth windows.

    This is still development/in-sample analysis. It measures stability and influence;
    it is not prospective validation.
    """
    if not isinstance(grid_result, dict):
        raise TypeError("grid_result must be the dict returned by run_pre_m5_pnl_grid_search().")
    surfaces = grid_result.get("surfaces")
    windows = grid_result.get("windows")
    contracts = grid_result.get("contracts")
    if not isinstance(surfaces, dict) or not surfaces:
        raise ValueError("grid_result has no surfaces.")
    if not isinstance(windows, pd.DataFrame) or windows.empty:
        raise ValueError("grid_result has no usable windows.")
    if not isinstance(contracts, pd.DataFrame) or contracts.empty:
        raise ValueError("grid_result has no usable contracts.")

    windows = windows.copy()
    contracts = contracts.copy()
    windows["decision_time"] = pd.to_datetime(windows["decision_time"], utc=True, errors="coerce")
    contracts["decision_time"] = pd.to_datetime(contracts["decision_time"], utc=True, errors="coerce")
    hb = windows[pd.to_numeric(windows["signals"], errors="coerce") >= int(high_breadth_min)].copy()
    hb = hb.sort_values(["decision_time", "session"]).reset_index(drop=True)
    sessions = sorted(windows["session"].astype(str).unique())

    if len(hb) < 4:
        raise RuntimeError(f"Only {len(hb)} high-breadth windows; too few for useful LOO diagnostics.")

    if show:
        print("=" * 122, flush=True)
        print("PRE-M5 GRID ROBUSTNESS — LOO WINDOWS + SESSION SPLITS + UNIQUE FLAGGED SETS", flush=True)
        print("=" * 122, flush=True)
        print(f"Filled windows: {len(windows)} | high-breadth windows: {len(hb)} | sessions: {len(sessions)}", flush=True)
        print("Using the ORIGINAL threshold lattice. No thresholds are re-fit inside folds.", flush=True)
        print("This is development stability analysis, not OOS validation.\n", flush=True)

    robustness_rows = []
    loo_rows = []
    session_rows = []
    unique_rows = []
    enriched_surfaces = {}

    for surface_i, (surface_name, surface) in enumerate(surfaces.items(), 1):
        surface = surface.copy().reset_index(drop=True)
        fx = str(surface["feature_x"].iloc[0])
        fy = str(surface["feature_y"].iloc[0])
        qty = float(surface["reduced_qty"].iloc[0])
        n_cells = len(surface)

        if show:
            print(f"[{surface_i}/{len(surfaces)}] {surface_name}: {n_cells} cells ...", flush=True)

        best = surface.sort_values(
            ["delta_vs_q3", "worst_session_delta", "max_drawdown"],
            ascending=[False, False, False],
        ).iloc[0]
        best_x = float(best["threshold_x"])
        best_y = float(best["threshold_y"])
        best_idx_arr = np.where(
            np.isclose(surface["threshold_x"].to_numpy(float), best_x)
            & np.isclose(surface["threshold_y"].to_numpy(float), best_y)
        )[0]
        if len(best_idx_arr) != 1:
            raise RuntimeError(f"Could not identify unique full-sample best cell for {surface_name}.")
        best_cell_idx = int(best_idx_arr[0])

        # 1) Leave one high-breadth window out.
        loo_delta_by_cell = [[] for _ in range(n_cells)]
        fold_positive_cell_pct = []
        best_rule_fold_delta = []

        for omitted in hb.itertuples(index=False):
            keep = ~(
                windows["session"].astype(str).eq(str(omitted.session))
                & windows["decision_time"].eq(pd.Timestamp(omitted.decision_time))
            )
            w_fold = windows.loc[keep].copy()
            c_fold = _subset_contracts(contracts, w_fold)
            scored = _score_surface(surface, w_fold, c_fold, high_breadth_min=high_breadth_min)
            deltas = scored["delta"].to_numpy(float)
            for j, d in enumerate(deltas):
                loo_delta_by_cell[j].append(float(d))

            fold_positive = 100.0 * float(np.mean(deltas > 1e-12))
            fold_positive_cell_pct.append(fold_positive)
            brd = float(deltas[best_cell_idx])
            best_rule_fold_delta.append(brd)

            loo_rows.append({
                "surface": surface_name,
                "omitted_session": str(omitted.session),
                "omitted_decision_time": pd.Timestamp(omitted.decision_time),
                "omitted_q3_pnl": float(omitted.actual_pnl),
                "positive_cells_pct_after_omit": fold_positive,
                "best_delta_available_after_omit": float(np.nanmax(deltas)),
                "full_sample_best_rule_delta_after_omit": brd,
            })

        loo_positive_pct = np.array([
            100.0 * np.mean(np.asarray(v, dtype=float) > 1e-12) for v in loo_delta_by_cell
        ])
        loo_min_delta = np.array([np.nanmin(v) for v in loo_delta_by_cell], dtype=float)
        loo_median_delta = np.array([np.nanmedian(v) for v in loo_delta_by_cell], dtype=float)

        # 2) Score every cell separately in each session.
        session_delta_by_cell = [[] for _ in range(n_cells)]
        for session in sessions:
            w_s = windows[windows["session"].astype(str).eq(str(session))].copy()
            c_s = _subset_contracts(contracts, w_s)
            scored = _score_surface(surface, w_s, c_s, high_breadth_min=high_breadth_min)
            deltas = scored["delta"].to_numpy(float)
            for j, d in enumerate(deltas):
                session_delta_by_cell[j].append(float(d))

            session_rows.append({
                "surface": surface_name,
                "session": str(session),
                "filled_windows": int(len(w_s)),
                "high_breadth_windows": int((pd.to_numeric(w_s["signals"], errors="coerce") >= int(high_breadth_min)).sum()),
                "q3_pnl": float(pd.to_numeric(w_s["actual_pnl"], errors="coerce").fillna(0.0).sum()),
                "positive_cells_pct": 100.0 * float(np.mean(deltas > 1e-12)),
                "best_delta_in_session": float(np.nanmax(deltas)),
                "full_sample_best_rule_delta": float(deltas[best_cell_idx]),
            })

        session_min_delta = np.array([np.nanmin(v) for v in session_delta_by_cell], dtype=float)
        session_all_positive = np.array([
            all(d > 1e-12 for d in v) for v in session_delta_by_cell
        ], dtype=bool)
        best_rule_worst_session_delta = float(np.nanmin(session_delta_by_cell[best_cell_idx]))

        # 3) How many heatmap cells are really the same flagged-window subset?
        set_groups = defaultdict(list)
        for j, r in surface.iterrows():
            flagged_set = _flagged_key_tuple(
                windows,
                fx, r["threshold_x"],
                fy, r["threshold_y"],
                high_breadth_min=high_breadth_min,
            )
            set_groups[flagged_set].append(j)

        for flagged_set, idxs in set_groups.items():
            sub = surface.loc[idxs]
            unique_rows.append({
                "surface": surface_name,
                "flagged_windows": int(len(flagged_set)),
                "cells": int(len(idxs)),
                "cell_pct": 100.0 * len(idxs) / n_cells,
                "delta_vs_q3": float(sub["delta_vs_q3"].median()),
                "delta_min": float(sub["delta_vs_q3"].min()),
                "delta_max": float(sub["delta_vs_q3"].max()),
                "threshold_x_min": float(sub["threshold_x"].min()),
                "threshold_x_max": float(sub["threshold_x"].max()),
                "threshold_y_min": float(sub["threshold_y"].min()),
                "threshold_y_max": float(sub["threshold_y"].max()),
                "flagged_set": "; ".join(flagged_set) if flagged_set else "NONE",
            })

        set_sizes = np.array([len(v) for v in set_groups.values()], dtype=float)

        enriched = surface.copy()
        enriched["loo_positive_pct"] = loo_positive_pct
        enriched["loo_min_delta"] = loo_min_delta
        enriched["loo_median_delta"] = loo_median_delta
        enriched["session_min_delta"] = session_min_delta
        enriched["all_sessions_positive"] = session_all_positive
        enriched_surfaces[surface_name] = enriched

        robustness_rows.append({
            "surface": surface_name,
            "reduced_qty": qty,
            "feature_x": fx,
            "feature_y": fy,
            "cells": n_cells,
            "full_positive_cells_pct": 100.0 * float((surface["delta_vs_q3"] > 1e-12).mean()),
            "full_best_delta": float(best["delta_vs_q3"]),
            "full_best_threshold_x": best_x,
            "full_best_threshold_y": best_y,
            "full_best_flagged_windows": int(best["flagged_windows"]),
            "loo_folds": int(len(hb)),
            "loo_all_positive_cells_pct": 100.0 * float(np.mean(loo_positive_pct >= 100.0 - 1e-9)),
            "loo_80pct_positive_cells_pct": 100.0 * float(np.mean(loo_positive_pct >= 80.0 - 1e-9)),
            "loo_min_fold_positive_cells_pct": float(np.nanmin(fold_positive_cell_pct)),
            "loo_median_fold_positive_cells_pct": float(np.nanmedian(fold_positive_cell_pct)),
            "best_rule_loo_min_delta": float(np.nanmin(best_rule_fold_delta)),
            "best_rule_loo_median_delta": float(np.nanmedian(best_rule_fold_delta)),
            "all_sessions_positive_cells_pct": 100.0 * float(np.mean(session_all_positive)),
            "best_rule_worst_session_delta": best_rule_worst_session_delta,
            "unique_flagged_sets": int(len(set_groups)),
            "cells_per_unique_set": float(n_cells / len(set_groups)) if set_groups else np.nan,
            "largest_identical_set_cells_pct": 100.0 * float(np.max(set_sizes)) / n_cells if len(set_sizes) else np.nan,
        })

    robustness = pd.DataFrame(robustness_rows)
    loo_influence = pd.DataFrame(loo_rows)
    by_session = pd.DataFrame(session_rows)
    unique_sets = pd.DataFrame(unique_rows)

    robustness = robustness.sort_values(
        [
            "loo_all_positive_cells_pct",
            "loo_min_fold_positive_cells_pct",
            "all_sessions_positive_cells_pct",
            "best_rule_loo_min_delta",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    loo_influence = loo_influence.sort_values(
        ["surface", "full_sample_best_rule_delta_after_omit", "omitted_decision_time"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    by_session = by_session.sort_values(["surface", "session"]).reset_index(drop=True)
    unique_sets = unique_sets.sort_values(
        ["surface", "cells", "delta_vs_q3"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    if show:
        print("\nROBUSTNESS SUMMARY", flush=True)
        _display(robustness.round(4))

        print("\nLEAVE-ONE-HIGH-BREADTH-WINDOW-OUT — MOST INFLUENTIAL OMISSIONS", flush=True)
        influential = loo_influence.groupby("surface", group_keys=False).head(4)
        _display(influential.round(4))

        print("\nSESSION-BY-SESSION GRID ROBUSTNESS", flush=True)
        _display(by_session.round(4))

        print("\nUNIQUE FLAGGED-WINDOW SETS — LARGEST THRESHOLD REGIONS", flush=True)
        top_sets = unique_sets.groupby("surface", group_keys=False).head(4)
        cols = [
            "surface", "flagged_windows", "cells", "cell_pct", "delta_vs_q3",
            "threshold_x_min", "threshold_x_max", "threshold_y_min", "threshold_y_max",
            "flagged_set",
        ]
        _display(top_sets[cols].round(4))

    if plot:
        for _, r in robustness.iterrows():
            key = r["surface"]
            s = enriched_surfaces[key]
            qty = float(r["reduced_qty"])
            fx, fy = r["feature_x"], r["feature_y"]
            labels = PAIR_LABELS.get((fx, fy), (fx, fy))
            _plot_heatmap(
                s,
                "loo_positive_pct",
                f"Q3 -> Q{qty:g} | LOO robustness: % omitted-window folds with positive delta vs Q3\n{labels[0]} x {labels[1]}",
                "% LOO folds positive",
            )
            if plot_session_min:
                _plot_heatmap(
                    s,
                    "session_min_delta",
                    f"Q3 -> Q{qty:g} | Worst recorder-session delta vs Q3\n{labels[0]} x {labels[1]}",
                    "worst-session delta vs Q3",
                )

    return {
        "version": STUDY_VERSION,
        "robustness": robustness,
        "loo_influence": loo_influence,
        "by_session": by_session,
        "unique_sets": unique_sets,
        "enriched_surfaces": enriched_surfaces,
        "high_breadth_windows": hb,
        "source_grid": grid_result,
    }
