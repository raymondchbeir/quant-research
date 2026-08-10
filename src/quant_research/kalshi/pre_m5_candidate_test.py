from __future__ import annotations

import numpy as np
import pandas as pd

from .pre_m5_pnl_grid_search import _score_rule

STUDY_VERSION = "PRE_M5_CANDIDATE_TEST_V1"
DEFAULT_SURFACE = "Q1_max_mid_path_length_c__max_mid_range_c"


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _subset_contracts(contracts: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    keys = windows[["session", "decision_time"]].drop_duplicates()
    return contracts.merge(keys, on=["session", "decision_time"], how="inner")


def _candidate_rows(grid_result, robustness_result, surface_name, min_flagged_windows=3, max_flagged_windows=6):
    surface = grid_result["surfaces"][surface_name].copy().reset_index(drop=True)
    enriched = robustness_result["enriched_surfaces"][surface_name].copy().reset_index(drop=True)

    join_cols = ["threshold_x", "threshold_y"]
    extra = [c for c in ["loo_positive_pct", "loo_min_delta", "loo_median_delta", "session_min_delta", "all_sessions_positive"] if c in enriched.columns]
    z = surface.merge(enriched[join_cols + extra], on=join_cols, how="left", suffixes=("", "_rob"))

    z["robust_candidate"] = (
        (pd.to_numeric(z["delta_vs_q3"], errors="coerce") > 0)
        & (pd.to_numeric(z["loo_min_delta"], errors="coerce") > 0)
        & z["all_sessions_positive"].fillna(False).astype(bool)
        & (pd.to_numeric(z["flagged_windows"], errors="coerce") >= int(min_flagged_windows))
        & (pd.to_numeric(z["flagged_windows"], errors="coerce") <= int(max_flagged_windows))
    )
    candidates = z[z["robust_candidate"]].copy()
    if candidates.empty:
        # Relax only the flagged-window-count restriction; keep true robustness requirements.
        candidates = z[
            (pd.to_numeric(z["delta_vs_q3"], errors="coerce") > 0)
            & (pd.to_numeric(z["loo_min_delta"], errors="coerce") > 0)
            & z["all_sessions_positive"].fillna(False).astype(bool)
        ].copy()
    if candidates.empty:
        raise RuntimeError("No cell is positive in full sample, every LOO fold, and both sessions.")

    # Choose a geometric interior/medoid cell among robust cells, NOT the max-PnL cell.
    # Rank-space makes path/range axes comparable despite different units.
    xs = np.sort(candidates["threshold_x"].unique())
    ys = np.sort(candidates["threshold_y"].unique())
    x_rank = {v: i for i, v in enumerate(xs)}
    y_rank = {v: i for i, v in enumerate(ys)}
    pts = np.array([[x_rank[x], y_rank[y]] for x, y in zip(candidates["threshold_x"], candidates["threshold_y"])], dtype=float)
    dist = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=2)
    candidates["medoid_distance"] = dist.sum(axis=1)
    candidates = candidates.sort_values(
        ["medoid_distance", "flagged_windows", "loo_min_delta", "delta_vs_q3"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return z, candidates


def _window_contributions(windows, contracts, fx, tx, fy, ty, reduced_qty=1.0, high_breadth_min=3):
    w = windows.copy()
    c = contracts.copy()
    w["decision_time"] = pd.to_datetime(w["decision_time"], utc=True, errors="coerce")
    c["decision_time"] = pd.to_datetime(c["decision_time"], utc=True, errors="coerce")

    w["flagged"] = (
        (pd.to_numeric(w["signals"], errors="coerce") >= int(high_breadth_min))
        & (pd.to_numeric(w[fx], errors="coerce") >= float(tx))
        & (pd.to_numeric(w[fy], errors="coerce") >= float(ty))
    )

    c = c.merge(w[["session", "decision_time", "flagged"]], on=["session", "decision_time"], how="left")
    c["flagged"] = c["flagged"].fillna(False)
    c["entry_fill_qty"] = pd.to_numeric(c["entry_fill_qty"], errors="coerce").fillna(0.0)
    if "pnl_per_contract" not in c.columns:
        c["actual_pnl"] = pd.to_numeric(c["actual_pnl"], errors="coerce").fillna(0.0)
        c["pnl_per_contract"] = np.where(c["entry_fill_qty"] > 1e-12, c["actual_pnl"] / c["entry_fill_qty"], 0.0)
    c["target_qty"] = np.where(c["flagged"], float(reduced_qty), 3.0)
    c["accepted_qty"] = np.minimum(c["entry_fill_qty"], c["target_qty"])
    c["candidate_pnl"] = c["accepted_qty"] * pd.to_numeric(c["pnl_per_contract"], errors="coerce").fillna(0.0)

    agg = c.groupby(["session", "decision_time"], as_index=False).agg(
        candidate_pnl=("candidate_pnl", "sum"),
        candidate_contracts=("accepted_qty", "sum"),
        q3_contracts=("entry_fill_qty", "sum"),
    )
    out = w.merge(agg, on=["session", "decision_time"], how="left")
    out["candidate_pnl"] = out["candidate_pnl"].fillna(0.0)
    out["candidate_contracts"] = out["candidate_contracts"].fillna(0.0)
    out["q3_contracts"] = out["q3_contracts"].fillna(0.0)
    out["delta_vs_q3"] = out["candidate_pnl"] - pd.to_numeric(out["actual_pnl"], errors="coerce").fillna(0.0)
    return out.sort_values(["decision_time", "session"]).reset_index(drop=True)


def run_pre_m5_candidate_test(
    grid_result,
    robustness_result,
    surface_name=DEFAULT_SURFACE,
    min_flagged_windows=3,
    max_flagged_windows=6,
    high_breadth_min=3,
    show=True,
):
    """Inspect and stress-test one robust interior pre-M5 Q1 candidate.

    Candidate selection deliberately avoids choosing the highest-PnL cell. Among cells
    that are positive on the full sample, positive in every leave-one-HB-window-out fold,
    and positive in both recorder sessions, it selects a medoid/interior threshold cell.

    Development-only. This does not alter or start any live strategy.
    """
    if not isinstance(grid_result, dict) or not isinstance(robustness_result, dict):
        raise TypeError("Pass the dicts returned by the grid search and robustness study.")
    if surface_name not in grid_result.get("surfaces", {}):
        raise KeyError(f"Surface not found: {surface_name}")
    if surface_name not in robustness_result.get("enriched_surfaces", {}):
        raise KeyError(f"Robustness surface not found: {surface_name}")

    windows = grid_result["windows"].copy()
    contracts = grid_result["contracts"].copy()
    surface = grid_result["surfaces"][surface_name].copy()
    fx = str(surface["feature_x"].iloc[0])
    fy = str(surface["feature_y"].iloc[0])
    reduced_qty = float(surface["reduced_qty"].iloc[0])

    all_cells, candidates = _candidate_rows(
        grid_result,
        robustness_result,
        surface_name,
        min_flagged_windows=min_flagged_windows,
        max_flagged_windows=max_flagged_windows,
    )
    chosen = candidates.iloc[0]
    tx, ty = float(chosen["threshold_x"]), float(chosen["threshold_y"])

    detail = _window_contributions(
        windows, contracts, fx, tx, fy, ty,
        reduced_qty=reduced_qty,
        high_breadth_min=high_breadth_min,
    )
    flagged = detail[detail["flagged"]].copy()

    # Session split for this exact fixed candidate.
    session_rows = []
    for session, g in detail.groupby("session", sort=True):
        session_rows.append({
            "session": session,
            "filled_windows": len(g),
            "flagged_windows": int(g["flagged"].sum()),
            "q3_pnl": float(pd.to_numeric(g["actual_pnl"], errors="coerce").fillna(0.0).sum()),
            "candidate_pnl": float(g["candidate_pnl"].sum()),
            "delta_vs_q3": float(g["delta_vs_q3"].sum()),
        })
    by_session = pd.DataFrame(session_rows)

    # Exact leave-one-high-breadth-window-out test for this fixed candidate.
    hb = detail[pd.to_numeric(detail["signals"], errors="coerce") >= int(high_breadth_min)].copy()
    loo_rows = []
    for omitted in hb.itertuples(index=False):
        keep = ~(
            windows["session"].astype(str).eq(str(omitted.session))
            & pd.to_datetime(windows["decision_time"], utc=True, errors="coerce").eq(pd.Timestamp(omitted.decision_time))
        )
        wf = windows.loc[keep].copy()
        cf = _subset_contracts(contracts, wf)
        score = _score_rule(
            wf, cf, fx, tx, fy, ty,
            reduced_qty=reduced_qty,
            high_breadth_min=high_breadth_min,
        )
        loo_rows.append({
            "omitted_session": omitted.session,
            "omitted_decision_time": pd.Timestamp(omitted.decision_time),
            "omitted_q3_pnl": float(omitted.actual_pnl),
            "delta_vs_q3_after_omit": float(score["delta_vs_q3"]),
            "candidate_pnl_after_omit": float(score["strategy_pnl"]),
        })
    loo = pd.DataFrame(loo_rows).sort_values("delta_vs_q3_after_omit").reset_index(drop=True)

    # Local threshold neighborhood around the selected medoid cell.
    xs = np.sort(surface["threshold_x"].unique())
    ys = np.sort(surface["threshold_y"].unique())
    ix = int(np.where(np.isclose(xs, tx))[0][0])
    iy = int(np.where(np.isclose(ys, ty))[0][0])
    xnear = xs[max(0, ix - 1): min(len(xs), ix + 2)]
    ynear = ys[max(0, iy - 1): min(len(ys), iy + 2)]
    neighborhood = all_cells[
        all_cells["threshold_x"].isin(xnear)
        & all_cells["threshold_y"].isin(ynear)
    ].copy().sort_values(["threshold_y", "threshold_x"])

    summary = pd.DataFrame([{
        "version": STUDY_VERSION,
        "surface": surface_name,
        "selection_method": "robust-cell medoid; not max PnL",
        "feature_x": fx,
        "threshold_x": tx,
        "feature_y": fy,
        "threshold_y": ty,
        "normal_qty": 3.0,
        "flagged_qty": reduced_qty,
        "high_breadth_min": int(high_breadth_min),
        "flagged_windows": int(flagged.shape[0]),
        "q3_pnl": float(pd.to_numeric(detail["actual_pnl"], errors="coerce").fillna(0.0).sum()),
        "candidate_pnl": float(detail["candidate_pnl"].sum()),
        "delta_vs_q3": float(detail["delta_vs_q3"].sum()),
        "loo_min_delta": float(loo["delta_vs_q3_after_omit"].min()) if len(loo) else np.nan,
        "loo_median_delta": float(loo["delta_vs_q3_after_omit"].median()) if len(loo) else np.nan,
        "worst_session_delta": float(by_session["delta_vs_q3"].min()) if len(by_session) else np.nan,
        "best_session_delta": float(by_session["delta_vs_q3"].max()) if len(by_session) else np.nan,
    }])

    if show:
        print("=" * 118)
        print("PRE-M5 ROBUST INTERIOR CANDIDATE TEST — DEVELOPMENT ONLY")
        print("=" * 118)
        print("Candidate is chosen from the robust region by geometric interior/medoid, NOT by highest PnL.")
        print("No live strategy is changed or started by this function.\n")
        print("SELECTED CANDIDATE")
        _display(summary.round(4))

        print("\nTOP ROBUST INTERIOR CELLS")
        cols = [
            "threshold_x", "threshold_y", "flagged_windows", "delta_vs_q3",
            "loo_positive_pct", "loo_min_delta", "loo_median_delta",
            "session_min_delta", "medoid_distance",
        ]
        _display(candidates[[c for c in cols if c in candidates.columns]].head(12).round(4))

        print("\nEXACT WINDOWS FLAGGED BY SELECTED CANDIDATE")
        fcols = [
            "session", "decision_time", "signals", fx, fy, "filled_assets",
            "q3_contracts", "candidate_contracts", "actual_pnl", "candidate_pnl", "delta_vs_q3",
        ]
        _display(flagged[[c for c in fcols if c in flagged.columns]].round(4))

        print("\nSESSION SPLIT — FIXED CANDIDATE")
        _display(by_session.round(4))

        print("\nLEAVE-ONE-HIGH-BREADTH-WINDOW-OUT — FIXED CANDIDATE")
        _display(loo.round(4))

        print("\nLOCAL 3x3 THRESHOLD NEIGHBORHOOD")
        ncols = [
            "threshold_x", "threshold_y", "flagged_windows", "delta_vs_q3",
            "loo_positive_pct", "loo_min_delta", "session_min_delta", "all_sessions_positive",
        ]
        _display(neighborhood[[c for c in ncols if c in neighborhood.columns]].round(4))

    return {
        "version": STUDY_VERSION,
        "summary": summary,
        "candidate": chosen.to_dict(),
        "robust_candidates": candidates,
        "flagged_windows": flagged,
        "window_detail": detail,
        "by_session": by_session,
        "loo": loo,
        "neighborhood": neighborhood,
    }
