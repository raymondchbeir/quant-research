from __future__ import annotations

"""Offline NFL-only robustness study for the sports gap/reversion research.

Reads an already-completed sports run. No API calls, no account/order endpoints,
no live-Q50 imports, no orders.

Outputs:
- nfl_exact_reconciliation.csv/json
- nfl_robustness_phase_summary.csv
- nfl_robustness_conditionals.csv
- nfl_transition_timing.csv
- nfl_robustness_headline.json
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

VERSION = "KALSHI_SPORTS_NFL_ROBUSTNESS_V1_OFFLINE"
SERIES = "KXNFLGAME"
SPREAD_CAPS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
DEADBANDS = [0.005, 0.01, 0.02, 0.03]
FAV_BINS = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.000001]
FAV_LABELS = ["50-55", "55-60", "60-70", "70-80", "80-90", "90-100"]
TARGET_FAVS = [0.80, 0.70, 0.60, 0.55]


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def state_array(mid: np.ndarray, deadband: float) -> np.ndarray:
    out = np.zeros(len(mid), dtype=np.int8)
    out[mid >= 0.5 + deadband] = 1
    out[mid <= 0.5 - deadband] = -1
    return out


def cross_count(mid: np.ndarray, deadband: float) -> int:
    seq: list[int] = []
    for s in state_array(mid, deadband):
        v = int(s)
        if not v:
            continue
        if not seq or seq[-1] != v:
            seq.append(v)
    return max(0, len(seq) - 1)


def first_opposite_delay(times: np.ndarray, mids: np.ndarray, start_i: int,
                         initial_sign: int, deadband: float) -> float | None:
    if initial_sign == 0:
        return None
    future = state_array(mids[start_i:], deadband)
    idx = np.flatnonzero(future == -initial_sign)
    if not len(idx):
        return None
    j = start_i + int(idx[0])
    return float(times[j] - times[start_i]) / 60.0


def filter_paths(paths: pd.DataFrame, spread_cap: float) -> pd.DataFrame:
    p = paths[paths["series_ticker"].astype(str) == SERIES].copy()
    for c in ("yes_mid", "quote_spread", "end_period_ts", "elapsed_from_start_s"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[
        p["yes_mid"].between(0, 1, inclusive="both")
        & p["quote_spread"].between(0, spread_cap, inclusive="both")
    ].copy()
    p = p[
        ((p["phase"] == "pregame") & p["elapsed_from_start_s"].between(-86400, 0, inclusive="left"))
        | ((p["phase"] == "in_game") & (p["elapsed_from_start_s"] >= 0))
    ].copy()
    return p


def phase_rows(p: pd.DataFrame, spread_cap: float, deadband: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (ticker, phase), g in p.groupby(["ticker", "phase"], sort=False):
        if len(g) < 3:
            continue
        g = g.sort_values("end_period_ts")
        mids = g["yes_mid"].to_numpy(float)
        cc = cross_count(mids, deadband)
        rows.append({
            "spread_cap": spread_cap,
            "deadband": deadband,
            "ticker": ticker,
            "event_ticker": str(g["event_ticker"].iloc[0]),
            "phase": phase,
            "rows": len(g),
            "cross_count": cc,
            "any_flip": cc > 0,
            "multi_flip": cc >= 2,
            "median_spread": float(np.nanmedian(g["quote_spread"].to_numpy(float))),
            "max_spread": float(np.nanmax(g["quote_spread"].to_numpy(float))),
        })
    return rows


def conditional_rows(p: pd.DataFrame, spread_cap: float, deadband: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (ticker, phase), g in p.groupby(["ticker", "phase"], sort=False):
        if len(g) < 3:
            continue
        g = g.sort_values("end_period_ts").reset_index(drop=True)
        mids = g["yes_mid"].to_numpy(float)
        times = g["end_period_ts"].to_numpy(float)
        fav = np.maximum(mids, 1 - mids)
        labels = np.asarray(pd.cut(
            fav, bins=FAV_BINS, labels=FAV_LABELS,
            right=False, include_lowest=True,
        ).astype(str))

        for label in FAV_LABELS:
            idx = np.flatnonzero(labels == label)
            if not len(idx):
                continue
            i = int(idx[0])
            start_mid = float(mids[i])
            start_abs = abs(start_mid - 0.5)
            initial_sign = 1 if start_mid >= 0.5 + deadband else (-1 if start_mid <= 0.5 - deadband else 0)
            future_mid = mids[i:]
            future_abs = np.abs(future_mid - 0.5)
            states = state_array(future_mid, deadband)
            closest = float(np.min(future_abs))
            closure = max(0.0, min(1.0, (start_abs - closest) / start_abs)) if start_abs > 1e-12 else 0.0
            rows.append({
                "spread_cap": spread_cap,
                "deadband": deadband,
                "ticker": ticker,
                "event_ticker": str(g["event_ticker"].iloc[0]),
                "phase": phase,
                "start_band": label,
                "start_ts": int(times[i]),
                "start_mid": start_mid,
                "start_favorite_prob": max(start_mid, 1 - start_mid),
                "hit_60_40": bool(np.any(future_abs <= 0.10)),
                "hit_55_45": bool(np.any(future_abs <= 0.05)),
                "touch_50_band": bool(np.any(future_abs <= deadband)),
                "favorite_flip": bool(np.any(states == -initial_sign) if initial_sign else False),
                "first_flip_delay_min": first_opposite_delay(times, mids, i, initial_sign, deadband),
                "max_gap_closure_fraction": closure,
            })
    return rows


def summarize_phase(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for keys, g in df.groupby(["spread_cap", "deadband", "phase"], dropna=False):
        spread_cap, deadband, phase = keys
        n = len(g)
        k = int(g["any_flip"].sum())
        lo, hi = wilson(k, n)
        out.append({
            "spread_cap": spread_cap,
            "deadband": deadband,
            "phase": phase,
            "n_games": n,
            "any_flip_n": k,
            "any_flip_rate": k / n if n else np.nan,
            "any_flip_wilson_lo": lo,
            "any_flip_wilson_hi": hi,
            "multi_flip_n": int(g["multi_flip"].sum()),
            "multi_flip_rate": float(g["multi_flip"].mean()),
            "median_cross_count": float(g["cross_count"].median()),
        })
    return pd.DataFrame(out)


def summarize_conditionals(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    groups = ["spread_cap", "deadband", "phase", "start_band"]
    for keys, g in df.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        n = len(g)
        row["n_games"] = n
        for col in ("hit_60_40", "hit_55_45", "touch_50_band", "favorite_flip"):
            k = int(g[col].astype(bool).sum())
            lo, hi = wilson(k, n)
            row[f"{col}_n"] = k
            row[f"{col}_rate"] = k / n if n else np.nan
            row[f"{col}_wilson_lo"] = lo
            row[f"{col}_wilson_hi"] = hi
        row["median_max_gap_closure_fraction"] = float(g["max_gap_closure_fraction"].median())
        d = g["first_flip_delay_min"].dropna()
        row["median_first_flip_delay_min"] = float(d.median()) if len(d) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def transition_rows(p: pd.DataFrame, spread_cap: float, deadband: float = 0.005) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    q = p[p["phase"] == "in_game"].copy()
    for ticker, g in q.groupby("ticker", sort=False):
        if len(g) < 3:
            continue
        g = g.sort_values("end_period_ts").reset_index(drop=True)
        mids = g["yes_mid"].to_numpy(float)
        times = g["end_period_ts"].to_numpy(float)
        fav = np.maximum(mids, 1 - mids)
        labels = np.asarray(pd.cut(
            fav, bins=FAV_BINS, labels=FAV_LABELS,
            right=False, include_lowest=True,
        ).astype(str))

        for label in FAV_LABELS:
            idx = np.flatnonzero(labels == label)
            if not len(idx):
                continue
            i = int(idx[0])
            start_fav = float(fav[i])
            future_fav = fav[i:]
            future_times = times[i:]
            for target in TARGET_FAVS:
                if target >= start_fav - 1e-12:
                    continue
                hit_idx = np.flatnonzero(future_fav <= target)
                delay = float(future_times[int(hit_idx[0])] - times[i]) / 60.0 if len(hit_idx) else None
                rows.append({
                    "spread_cap": spread_cap,
                    "deadband": deadband,
                    "ticker": ticker,
                    "event_ticker": str(g["event_ticker"].iloc[0]),
                    "start_band": label,
                    "start_favorite_prob": start_fav,
                    "target_favorite_prob": target,
                    "hit_target": bool(len(hit_idx)),
                    "first_hit_delay_min": delay,
                })
    return rows


def summarize_transitions(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    groups = ["spread_cap", "deadband", "start_band", "target_favorite_prob"]
    for keys, g in df.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        n = len(g); k = int(g["hit_target"].sum()); lo, hi = wilson(k, n)
        row.update({
            "n_games": n,
            "hit_n": k,
            "hit_rate": k / n if n else np.nan,
            "hit_wilson_lo": lo,
            "hit_wilson_hi": hi,
        })
        d = g.loc[g["hit_target"].astype(bool), "first_hit_delay_min"].dropna()
        row["delay_p25_min"] = float(d.quantile(0.25)) if len(d) else np.nan
        row["delay_median_min"] = float(d.median()) if len(d) else np.nan
        row["delay_p75_min"] = float(d.quantile(0.75)) if len(d) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def exact_reconciliation(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = root / "exact_trade_crossing_windows.csv"
    if not path.exists():
        return pd.DataFrame(), {"available": False}
    ex = pd.read_csv(path)
    ex = ex[ex["series_ticker"].astype(str) == SERIES].copy()
    if ex.empty:
        return pd.DataFrame(), {"available": True, "nfl_rows": 0}
    ex["trade_verified_flip"] = ex["trade_verified_flip"].astype(bool)
    out = ex.groupby("phase", dropna=False).agg(
        quote_cross_events=("quote_cross_id", "count"),
        verified_trade_flips=("trade_verified_flip", "sum"),
        crossing_markets=("ticker", "nunique"),
        windows_with_trades=("trade_count", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
    ).reset_index()
    out["verified_fraction"] = out["verified_trade_flips"] / out["quote_cross_events"]
    headline = {
        "available": True,
        "nfl_quote_cross_events": int(len(ex)),
        "nfl_crossing_markets": int(ex["ticker"].nunique()),
        "nfl_verified_trade_flips": int(ex["trade_verified_flip"].sum()),
        "nfl_verified_fraction": float(ex["trade_verified_flip"].mean()),
        "nfl_windows_with_trades": int((pd.to_numeric(ex["trade_count"], errors="coerce") > 0).sum()),
    }
    return out, headline


def static_self_check(show: bool = True) -> dict[str, Any]:
    out = {
        "version": VERSION,
        "offline_only": True,
        "series_filter": SERIES,
        "api_called": False,
        "orders_sent": False,
        "live_q50_modules_imported": False,
        "spread_caps": SPREAD_CAPS,
        "deadbands": DEADBANDS,
        "one_first_entry_per_game_phase_band": True,
        "wilson_95_intervals": True,
        "ok": True,
    }
    if show:
        print("=" * 128)
        print("NFL SPORTS GAP ROBUSTNESS — STATIC CHECK")
        print("=" * 128)
        for k, v in out.items():
            print(f"{k:45s}: {v}")
    return out


def run(run_dir: str | Path, show: bool = True) -> dict[str, Any]:
    static_self_check(show=show)
    root = Path(run_dir).expanduser().resolve()
    paths = pd.read_csv(root / "minute_paths.csv.gz", compression="gzip")
    nfl_raw = paths[paths["series_ticker"].astype(str) == SERIES].copy()
    if nfl_raw.empty:
        raise RuntimeError(f"No {SERIES} rows in minute_paths.csv.gz")

    exact_df, exact_headline = exact_reconciliation(root)
    exact_df.to_csv(root / "nfl_exact_reconciliation.csv", index=False)
    (root / "nfl_exact_reconciliation.json").write_text(json.dumps(exact_headline, indent=2))

    phase_all: list[dict[str, Any]] = []
    cond_all: list[dict[str, Any]] = []
    transition_all: list[dict[str, Any]] = []

    for spread in SPREAD_CAPS:
        p = filter_paths(paths, spread)
        for deadband in DEADBANDS:
            phase_all.extend(phase_rows(p, spread, deadband))
            cond_all.extend(conditional_rows(p, spread, deadband))
        transition_all.extend(transition_rows(p, spread, deadband=0.005))

    phase_detail = pd.DataFrame(phase_all)
    cond_detail = pd.DataFrame(cond_all)
    trans_detail = pd.DataFrame(transition_all)

    phase_summary = summarize_phase(phase_detail)
    cond_summary = summarize_conditionals(cond_detail)
    trans_summary = summarize_transitions(trans_detail)

    phase_summary.to_csv(root / "nfl_robustness_phase_summary.csv", index=False)
    cond_summary.to_csv(root / "nfl_robustness_conditionals.csv", index=False)
    trans_summary.to_csv(root / "nfl_transition_timing.csv", index=False)

    reference = cond_summary[
        (cond_summary["spread_cap"] == 0.02)
        & (cond_summary["deadband"] == 0.005)
        & (cond_summary["phase"] == "in_game")
    ].copy()

    headline = {
        "version": VERSION,
        "run_dir": str(root),
        "series": SERIES,
        "nfl_markets_raw": int(nfl_raw["ticker"].nunique()),
        "nfl_minute_rows_raw": int(len(nfl_raw)),
        "spread_caps": SPREAD_CAPS,
        "deadbands": DEADBANDS,
        "exact_reconciliation": exact_headline,
        "reference_slice": "in_game spread<=2c deadband=0.5c",
        "reference_band_rows": int(len(reference)),
        "api_called": False,
        "orders_sent": False,
        "scientific_note": (
            "This is robustness/transition analysis, not an execution backtest. "
            "A spread cap filters observed quote states but does not prove a fill at midpoint."
        ),
    }
    (root / "nfl_robustness_headline.json").write_text(json.dumps(headline, indent=2, default=str))

    if show:
        print("\n" + "=" * 128)
        print("NFL EXACT-vs-MINUTE RECONCILIATION")
        print("=" * 128)
        print(json.dumps(exact_headline, indent=2))
        if not exact_df.empty:
            print("\n" + exact_df.to_string(index=False))

        print("\n" + "=" * 128)
        print("IN-GAME FLIP ROBUSTNESS — SPREAD x DEADBAND")
        print("=" * 128)
        grid = phase_summary[phase_summary["phase"] == "in_game"][
            ["spread_cap", "deadband", "n_games", "any_flip_rate", "any_flip_wilson_lo", "any_flip_wilson_hi", "multi_flip_rate"]
        ]
        print(grid.to_string(index=False))

        print("\n" + "=" * 128)
        print("REFERENCE CONDITIONALS — IN-GAME, SPREAD <= 2c, DEADBAND = 0.5c")
        print("=" * 128)
        cols = [
            "start_band", "n_games", "hit_60_40_rate", "hit_55_45_rate",
            "touch_50_band_rate", "favorite_flip_rate",
            "favorite_flip_wilson_lo", "favorite_flip_wilson_hi",
            "median_max_gap_closure_fraction", "median_first_flip_delay_min",
        ]
        print(reference[cols].to_string(index=False) if not reference.empty else "No rows at <=2c reference slice")

        print("\n" + "=" * 128)
        print("TRANSITION/TIMING — DEADBAND 0.5c")
        print("=" * 128)
        show_t = trans_summary[trans_summary["spread_cap"].isin([0.02, 0.05])]
        print(show_t.to_string(index=False))

        print("\nOutputs:")
        for name in (
            "nfl_exact_reconciliation.csv", "nfl_robustness_phase_summary.csv",
            "nfl_robustness_conditionals.csv", "nfl_transition_timing.csv",
            "nfl_robustness_headline.json",
        ):
            print(" -", root / name)
        print("\nIMPORTANT: robustness does not establish executable alpha.")

    return headline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run(a.run_dir, show=True)


if __name__ == "__main__":
    main()
