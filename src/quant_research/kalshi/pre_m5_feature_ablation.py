from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STUDY_VERSION = "PRE_M5_FEATURE_ABLATION_V1"

RANGE = "max_mid_range_c"
PATH = "max_mid_path_length_c"
RV = "max_mid_rv_c"

PAIR_SURFACES = {
    "RANGE_PATH": "Q1_max_mid_path_length_c__max_mid_range_c",
    "RANGE_RV": "Q1_max_mid_range_c__max_mid_rv_c",
}


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _prepare_window_effects(grid_result):
    windows = grid_result["windows"].copy()
    contracts = grid_result["contracts"].copy()

    windows["decision_time"] = pd.to_datetime(windows["decision_time"], utc=True, errors="coerce")
    contracts["decision_time"] = pd.to_datetime(contracts["decision_time"], utc=True, errors="coerce")

    windows["signals"] = _num(windows["signals"]).fillna(0)
    windows["actual_pnl"] = _num(windows["actual_pnl"]).fillna(0.0)
    for c in (RANGE, PATH, RV):
        windows[c] = _num(windows[c])

    contracts["entry_fill_qty"] = _num(contracts["entry_fill_qty"]).fillna(0.0)
    contracts["actual_pnl"] = _num(contracts["actual_pnl"]).fillna(0.0)
    contracts["pnl_per_contract"] = np.where(
        contracts["entry_fill_qty"] > 1e-12,
        contracts["actual_pnl"] / contracts["entry_fill_qty"],
        0.0,
    )
    contracts["q1_qty"] = np.minimum(contracts["entry_fill_qty"], 1.0)
    contracts["q1_pnl"] = contracts["q1_qty"] * contracts["pnl_per_contract"]

    effects = contracts.groupby(["session", "decision_time"], as_index=False).agg(
        q3_contract_pnl=("actual_pnl", "sum"),
        q1_if_flagged_pnl=("q1_pnl", "sum"),
        q3_contracts=("entry_fill_qty", "sum"),
        q1_contracts=("q1_qty", "sum"),
    )
    effects["q1_delta_if_flagged"] = effects["q1_if_flagged_pnl"] - effects["q3_contract_pnl"]

    z = windows.merge(effects, on=["session", "decision_time"], how="left")
    for c in ("q3_contract_pnl", "q1_if_flagged_pnl", "q3_contracts", "q1_contracts", "q1_delta_if_flagged"):
        z[c] = _num(z[c]).fillna(0.0)
    z["q3_match_error"] = z["q3_contract_pnl"] - z["actual_pnl"]
    return z.sort_values(["decision_time", "session"]).reset_index(drop=True), contracts


def _single_thresholds(grid_result, feature):
    surfaces = grid_result["surfaces"]
    vals = []
    for s in surfaces.values():
        if float(s["reduced_qty"].iloc[0]) != 1.0:
            continue
        fx = str(s["feature_x"].iloc[0])
        fy = str(s["feature_y"].iloc[0])
        if fx == feature:
            vals.extend(_num(s["threshold_x"]).dropna().tolist())
        if fy == feature:
            vals.extend(_num(s["threshold_y"]).dropna().tolist())
    if not vals:
        raise ValueError(f"Could not recover original threshold lattice for {feature}.")
    return np.array(sorted(set(float(x) for x in vals)), dtype=float)


def _rule_mask(windows, model, tx, ty=None, high_breadth_min=3):
    hb = windows["signals"] >= int(high_breadth_min)
    if model == "RANGE_ONLY":
        return hb & (windows[RANGE] >= float(tx))
    if model == "PATH_ONLY":
        return hb & (windows[PATH] >= float(tx))
    if model == "RV_ONLY":
        return hb & (windows[RV] >= float(tx))
    if model == "RANGE_PATH":
        # tx=path threshold, ty=range threshold, matching the original surface axes.
        return hb & (windows[PATH] >= float(tx)) & (windows[RANGE] >= float(ty))
    if model == "RANGE_RV":
        # tx=range threshold, ty=RV threshold.
        return hb & (windows[RANGE] >= float(tx)) & (windows[RV] >= float(ty))
    raise ValueError(f"Unknown model: {model}")


def _score_mask(windows, mask, high_breadth_min=3):
    mask = pd.Series(mask, index=windows.index).fillna(False).astype(bool)
    delta_contrib = windows["q1_delta_if_flagged"].where(mask, 0.0)
    full_delta = float(delta_contrib.sum())
    q3_pnl = float(windows["actual_pnl"].sum())

    sessions = sorted(windows["session"].astype(str).unique())
    session_deltas = []
    for session in sessions:
        sm = windows["session"].astype(str).eq(session)
        session_deltas.append(float(delta_contrib[sm].sum()))

    hb = windows[windows["signals"] >= int(high_breadth_min)]
    loo = []
    for r in hb.itertuples(index=True):
        contribution = float(delta_contrib.loc[r.Index]) if bool(mask.loc[r.Index]) else 0.0
        loo.append(full_delta - contribution)

    flagged = windows.loc[mask].copy()
    return {
        "flagged_windows": int(mask.sum()),
        "q3_pnl": q3_pnl,
        "candidate_pnl": q3_pnl + full_delta,
        "delta_vs_q3": full_delta,
        "loo_folds": int(len(loo)),
        "loo_positive_pct": 100.0 * float(np.mean(np.asarray(loo) > 1e-12)) if loo else np.nan,
        "loo_min_delta": float(np.min(loo)) if loo else np.nan,
        "loo_median_delta": float(np.median(loo)) if loo else np.nan,
        "session_min_delta": float(np.min(session_deltas)) if session_deltas else np.nan,
        "session_max_delta": float(np.max(session_deltas)) if session_deltas else np.nan,
        "all_sessions_positive": bool(all(x > 1e-12 for x in session_deltas)) if session_deltas else False,
        "flagged_keys": tuple(
            f"{r.session}|{pd.Timestamp(r.decision_time).isoformat()}"
            for r in flagged.sort_values(["decision_time", "session"]).itertuples(index=False)
        ),
    }


def _build_single_surface(grid_result, windows, model, feature, high_breadth_min=3):
    rows = []
    for t in _single_thresholds(grid_result, feature):
        mask = _rule_mask(windows, model, t, high_breadth_min=high_breadth_min)
        rows.append({
            "model": model,
            "dimensions": 1,
            "feature_x": feature,
            "threshold_x": float(t),
            "feature_y": None,
            "threshold_y": np.nan,
            **_score_mask(windows, mask, high_breadth_min=high_breadth_min),
        })
    return pd.DataFrame(rows)


def _build_pair_surface(grid_result, windows, model, surface_name, high_breadth_min=3):
    src = grid_result["surfaces"][surface_name].copy()
    src = src[np.isclose(_num(src["reduced_qty"]), 1.0)].copy()
    rows = []
    for r in src.itertuples(index=False):
        mask = _rule_mask(
            windows,
            model,
            float(r.threshold_x),
            float(r.threshold_y),
            high_breadth_min=high_breadth_min,
        )
        rows.append({
            "model": model,
            "dimensions": 2,
            "feature_x": str(r.feature_x),
            "threshold_x": float(r.threshold_x),
            "feature_y": str(r.feature_y),
            "threshold_y": float(r.threshold_y),
            **_score_mask(windows, mask, high_breadth_min=high_breadth_min),
        })
    return pd.DataFrame(rows)


def _medoid_row(cells, min_flagged_windows=3, max_flagged_windows=6):
    robust = cells[
        (cells["delta_vs_q3"] > 1e-12)
        & (cells["loo_min_delta"] > 1e-12)
        & cells["all_sessions_positive"].fillna(False)
    ].copy()
    if robust.empty:
        return None, robust, "no robust cell"

    preferred = robust[
        (robust["flagged_windows"] >= int(min_flagged_windows))
        & (robust["flagged_windows"] <= int(max_flagged_windows))
    ].copy()
    pool = preferred if len(preferred) else robust
    reason = "robust medoid with flagged-window guard" if len(preferred) else "robust medoid (guard had no cells)"

    tx_vals = np.sort(cells["threshold_x"].dropna().unique())
    x_rank = {float(v): i for i, v in enumerate(tx_vals)}
    denom_x = max(1, len(tx_vals) - 1)
    pts = []
    for r in pool.itertuples(index=False):
        x = x_rank[float(r.threshold_x)] / denom_x
        if int(r.dimensions) == 1:
            pts.append([x])
        else:
            ty_vals = np.sort(cells["threshold_y"].dropna().unique())
            y_rank = {float(v): i for i, v in enumerate(ty_vals)}
            denom_y = max(1, len(ty_vals) - 1)
            y = y_rank[float(r.threshold_y)] / denom_y
            pts.append([x, y])
    pts = np.asarray(pts, dtype=float)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)).sum(axis=1)
    chosen = pool.iloc[int(np.argmin(d))].copy()
    return chosen, robust, reason


def _selected_flagged_table(windows, selected_rows, high_breadth_min=3):
    out = []
    for model, row in selected_rows.items():
        if row is None:
            continue
        mask = _rule_mask(
            windows,
            model,
            row["threshold_x"],
            row.get("threshold_y", np.nan),
            high_breadth_min=high_breadth_min,
        )
        g = windows.loc[mask].copy()
        for r in g.itertuples(index=False):
            out.append({
                "model": model,
                "session": r.session,
                "decision_time": r.decision_time,
                "signals": int(r.signals),
                "max_range_c": float(getattr(r, RANGE)),
                "max_path_c": float(getattr(r, PATH)),
                "max_rv_c": float(getattr(r, RV)),
                "q3_pnl": float(r.actual_pnl),
                "q1_delta_if_flagged": float(r.q1_delta_if_flagged),
                "q1_pnl_if_flagged": float(r.actual_pnl + r.q1_delta_if_flagged),
            })
    return pd.DataFrame(out).sort_values(["model", "decision_time", "session"]).reset_index(drop=True) if out else pd.DataFrame()


def _plot_single_curves(surfaces):
    import matplotlib.pyplot as plt
    for model in ("RANGE_ONLY", "PATH_ONLY", "RV_ONLY"):
        s = surfaces[model].sort_values("threshold_x")
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.plot(s["threshold_x"], s["delta_vs_q3"], marker="o", label="full-sample delta")
        ax.plot(s["threshold_x"], s["loo_min_delta"], marker="o", linestyle="--", label="worst LOO delta")
        ax.plot(s["threshold_x"], s["session_min_delta"], marker="o", linestyle=":", label="worst-session delta")
        ax.axhline(0.0, linewidth=1)
        ax.set_title(f"{model}: Q3 -> Q1 threshold ablation")
        ax.set_xlabel(s["feature_x"].iloc[0])
        ax.set_ylabel("PnL improvement vs Q3 ($)")
        ax.legend()
        fig.tight_layout()
        plt.show()


def run_pre_m5_feature_ablation(
    grid_result,
    high_breadth_min=3,
    min_flagged_windows=3,
    max_flagged_windows=6,
    plot=True,
    show=True,
):
    """Ablate pre-M5 risk features on the exact already-built Q3 execution sample.

    Models compared, all using Q3 -> Q1 only in flagged high-breadth windows:
      RANGE_ONLY, PATH_ONLY, RV_ONLY, RANGE_PATH, RANGE_RV, plus Q3 baseline.

    Single-feature thresholds are recovered from the ORIGINAL Q1 grid lattice. Pair
    models reuse the ORIGINAL 2D Q1 lattice. No recorder files are rescanned and no
    live monitor/shadow state is touched.

    A robust cell must beat Q3 on the full sample, remain positive after omitting each
    high-breadth window one at a time, and improve each recorder session separately.
    The reported selected rule is the geometric medoid of that robust region, not the
    highest-PnL cell.
    """
    if not isinstance(grid_result, dict):
        raise TypeError("grid_result must be the dict returned by run_pre_m5_pnl_grid_search().")
    if "windows" not in grid_result or "contracts" not in grid_result or "surfaces" not in grid_result:
        raise ValueError("grid_result is missing windows/contracts/surfaces.")

    windows, contracts = _prepare_window_effects(grid_result)
    if windows.empty:
        raise RuntimeError("No windows available in grid_result.")

    max_match_error = float(windows["q3_match_error"].abs().max())
    if max_match_error > 1e-8:
        raise RuntimeError(f"Contract-level Q3 PnL does not reconcile to window PnL; max error={max_match_error:.8f}")

    surfaces = {
        "RANGE_ONLY": _build_single_surface(grid_result, windows, "RANGE_ONLY", RANGE, high_breadth_min),
        "PATH_ONLY": _build_single_surface(grid_result, windows, "PATH_ONLY", PATH, high_breadth_min),
        "RV_ONLY": _build_single_surface(grid_result, windows, "RV_ONLY", RV, high_breadth_min),
        "RANGE_PATH": _build_pair_surface(grid_result, windows, "RANGE_PATH", PAIR_SURFACES["RANGE_PATH"], high_breadth_min),
        "RANGE_RV": _build_pair_surface(grid_result, windows, "RANGE_RV", PAIR_SURFACES["RANGE_RV"], high_breadth_min),
    }

    summary_rows = []
    selected_rows = {}
    for model, cells in surfaces.items():
        chosen, robust, selection_reason = _medoid_row(
            cells,
            min_flagged_windows=min_flagged_windows,
            max_flagged_windows=max_flagged_windows,
        )
        selected_rows[model] = chosen
        best = cells.sort_values(["delta_vs_q3", "loo_min_delta", "session_min_delta"], ascending=False).iloc[0]
        row = {
            "model": model,
            "dimensions": int(cells["dimensions"].iloc[0]),
            "cells": len(cells),
            "positive_cells_pct": 100.0 * float((cells["delta_vs_q3"] > 1e-12).mean()),
            "robust_cells": int(len(robust)),
            "robust_cells_pct": 100.0 * len(robust) / len(cells),
            "best_delta_vs_q3": float(best["delta_vs_q3"]),
            "selection": selection_reason,
        }
        if chosen is not None:
            row.update({
                "selected_threshold_x": float(chosen["threshold_x"]),
                "selected_threshold_y": float(chosen["threshold_y"]) if pd.notna(chosen["threshold_y"]) else np.nan,
                "selected_flagged_windows": int(chosen["flagged_windows"]),
                "selected_delta_vs_q3": float(chosen["delta_vs_q3"]),
                "selected_candidate_pnl": float(chosen["candidate_pnl"]),
                "selected_loo_min_delta": float(chosen["loo_min_delta"]),
                "selected_loo_median_delta": float(chosen["loo_median_delta"]),
                "selected_worst_session_delta": float(chosen["session_min_delta"]),
            })
        summary_rows.append(row)

    q3_pnl = float(windows["actual_pnl"].sum())
    summary = pd.DataFrame(summary_rows)
    summary["incremental_selected_delta_vs_range_only"] = np.nan
    range_sel = summary.loc[summary["model"] == "RANGE_ONLY", "selected_delta_vs_q3"]
    if len(range_sel) and pd.notna(range_sel.iloc[0]):
        summary["incremental_selected_delta_vs_range_only"] = summary["selected_delta_vs_q3"] - float(range_sel.iloc[0])

    selected = []
    for model, r in selected_rows.items():
        if r is None:
            continue
        selected.append({
            "model": model,
            "feature_x": r["feature_x"],
            "threshold_x": float(r["threshold_x"]),
            "feature_y": r["feature_y"],
            "threshold_y": float(r["threshold_y"]) if pd.notna(r["threshold_y"]) else np.nan,
            "flagged_windows": int(r["flagged_windows"]),
            "q3_pnl": q3_pnl,
            "candidate_pnl": float(r["candidate_pnl"]),
            "delta_vs_q3": float(r["delta_vs_q3"]),
            "loo_min_delta": float(r["loo_min_delta"]),
            "loo_median_delta": float(r["loo_median_delta"]),
            "worst_session_delta": float(r["session_min_delta"]),
            "all_sessions_positive": bool(r["all_sessions_positive"]),
        })
    selected = pd.DataFrame(selected)
    flagged = _selected_flagged_table(windows, selected_rows, high_breadth_min=high_breadth_min)

    # Single-feature threshold detail is useful for seeing whether one feature has a plateau.
    single_detail = pd.concat(
        [surfaces[m] for m in ("RANGE_ONLY", "PATH_ONLY", "RV_ONLY")],
        ignore_index=True,
    )

    if show:
        print("=" * 122, flush=True)
        print("PRE-M5 FEATURE ABLATION — DOES RANGE NEED PATH/RV?", flush=True)
        print("=" * 122, flush=True)
        print(f"Q3 realized PnL on exact sample: ${q3_pnl:+.4f}", flush=True)
        print(f"Filled windows: {len(windows)} | high-breadth windows: {int((windows['signals'] >= high_breadth_min).sum())}", flush=True)
        print("All candidate rules are Q3 -> Q1 only when flagged. Thresholds reuse the ORIGINAL grid lattice.", flush=True)
        print("Selected rules are robust-region medoids, NOT maximum-PnL cells.\n", flush=True)

        print("ABLATION ROBUSTNESS SUMMARY", flush=True)
        cols = [
            "model", "dimensions", "cells", "positive_cells_pct", "robust_cells", "robust_cells_pct",
            "best_delta_vs_q3", "selected_threshold_x", "selected_threshold_y",
            "selected_flagged_windows", "selected_delta_vs_q3", "selected_candidate_pnl",
            "selected_loo_min_delta", "selected_worst_session_delta",
            "incremental_selected_delta_vs_range_only",
        ]
        _display(summary[[c for c in cols if c in summary.columns]].round(4))

        print("\nSELECTED ROBUST MEDOID RULES", flush=True)
        _display(selected.round(4))

        print("\nSINGLE-FEATURE THRESHOLD CURVES", flush=True)
        detail_cols = [
            "model", "threshold_x", "flagged_windows", "delta_vs_q3",
            "loo_min_delta", "loo_positive_pct", "session_min_delta", "all_sessions_positive",
        ]
        _display(single_detail[detail_cols].round(4))

        print("\nEXACT WINDOWS FLAGGED BY EACH SELECTED RULE", flush=True)
        _display(flagged.round(4) if len(flagged) else flagged)

        if "RANGE_ONLY" in selected_rows and selected_rows["RANGE_ONLY"] is not None:
            r = summary[summary["model"] == "RANGE_ONLY"].iloc[0]
            print("\nKEY ABLATION QUESTION", flush=True)
            print(
                f"Range-only robust-medoid delta: ${float(r['selected_delta_vs_q3']):+.4f}. "
                "Compare RANGE_PATH / RANGE_RV incremental_selected_delta_vs_range_only above."
            )
            print(
                "If the pair adds little or nothing while range-only has a broad robust threshold plateau, "
                "prefer the simpler range-only prospective rule."
            )

    if plot:
        _plot_single_curves(surfaces)

    return {
        "version": STUDY_VERSION,
        "q3_pnl": q3_pnl,
        "summary": summary,
        "selected": selected,
        "single_detail": single_detail,
        "flagged_windows": flagged,
        "surfaces": surfaces,
        "windows": windows,
        "contracts": contracts,
    }
