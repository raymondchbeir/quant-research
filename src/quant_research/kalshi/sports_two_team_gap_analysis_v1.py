from __future__ import annotations

"""Analyze broad two-team Kalshi sports gap closure and favorite flips.

Input is the output of sports_two_team_gap_backfill_v1. The primary path uses
YES quote midpoint, not last trade, to avoid interpreting sparse prints as the
current market state.

Cross counts are minute-resolution lower bounds. They do not count multiple
within-minute flips; exact trade/orderbook refinement belongs in a second-stage
study after the broad phenomenon is established.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

VERSION = "KALSHI_SPORTS_TWO_TEAM_GAP_ANALYSIS_V1"

FAV_BINS = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.000001]
FAV_LABELS = ["50-55", "55-60", "60-70", "70-80", "80-90", "90-100"]


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def _hysteresis_states(mid: np.ndarray, deadband: float = 0.005) -> np.ndarray:
    out = np.zeros(len(mid), dtype=np.int8)
    out[mid >= 0.5 + deadband] = 1
    out[mid <= 0.5 - deadband] = -1
    return out


def _compressed_nonzero(states: np.ndarray) -> list[int]:
    seq: list[int] = []
    for s in states:
        v = int(s)
        if v == 0:
            continue
        if not seq or seq[-1] != v:
            seq.append(v)
    return seq


def _cross_count(mid: np.ndarray, deadband: float = 0.005) -> int:
    seq = _compressed_nonzero(_hysteresis_states(mid, deadband))
    return max(0, len(seq) - 1)


def _first_opposite_delay_minutes(times: np.ndarray, mid: np.ndarray,
                                  start_idx: int, initial_sign: int,
                                  deadband: float = 0.005) -> float | None:
    if initial_sign == 0:
        return None
    future = _hysteresis_states(mid[start_idx:], deadband)
    idx = np.flatnonzero(future == -initial_sign)
    if not len(idx):
        return None
    j = start_idx + int(idx[0])
    return float(times[j] - times[start_idx]) / 60.0


def _event_phase_summary(g: pd.DataFrame, deadband: float) -> dict[str, Any]:
    g = g.sort_values("end_period_ts")
    mid = g["yes_mid"].to_numpy(float)
    times = g["end_period_ts"].to_numpy(float)
    abs_from_half = np.abs(mid - 0.5)
    return {
        "rows": len(g),
        "first_ts": int(times[0]),
        "last_ts": int(times[-1]),
        "start_mid": float(mid[0]),
        "end_mid": float(mid[-1]),
        "max_favorite_prob": float(np.max(np.maximum(mid, 1 - mid))),
        "min_abs_from_50": float(np.min(abs_from_half)),
        "cross_count": _cross_count(mid, deadband),
        "ever_inside_60_40": bool(np.any(abs_from_half <= 0.10)),
        "ever_inside_55_45": bool(np.any(abs_from_half <= 0.05)),
        "ever_touch_50_band": bool(np.any(abs_from_half <= deadband)),
        "max_spread": float(np.nanmax(g["quote_spread"].to_numpy(float))),
        "median_spread": float(np.nanmedian(g["quote_spread"].to_numpy(float))),
    }


def _conditional_rows(g: pd.DataFrame, phase: str, deadband: float) -> list[dict[str, Any]]:
    g = g.sort_values("end_period_ts").reset_index(drop=True)
    mids = g["yes_mid"].to_numpy(float)
    times = g["end_period_ts"].to_numpy(float)
    fav = np.maximum(mids, 1 - mids)
    labels = pd.cut(
        fav,
        bins=FAV_BINS,
        labels=FAV_LABELS,
        right=False,
        include_lowest=True,
    )
    label_array = np.asarray(labels.astype(str))

    out: list[dict[str, Any]] = []
    for label in FAV_LABELS:
        idx = np.flatnonzero(label_array == label)
        if not len(idx):
            continue
        i = int(idx[0])  # one first-entry observation per game/phase/band
        start_mid = float(mids[i])
        start_abs = abs(start_mid - 0.5)
        initial_sign = 1 if start_mid >= 0.5 + deadband else (-1 if start_mid <= 0.5 - deadband else 0)
        future_mid = mids[i:]
        future_abs = np.abs(future_mid - 0.5)
        closest = float(np.min(future_abs))
        closure_frac = (
            max(0.0, min(1.0, (start_abs - closest) / start_abs))
            if start_abs > 1e-12 else 0.0
        )
        states = _hysteresis_states(future_mid, deadband)
        out.append({
            "phase": phase,
            "start_band": label,
            "start_ts": int(times[i]),
            "start_mid": start_mid,
            "start_favorite_prob": max(start_mid, 1 - start_mid),
            "start_gap_c": 200.0 * start_abs,
            "future_min_gap_c": 200.0 * closest,
            "max_gap_closure_fraction": closure_frac,
            "hit_60_40": bool(np.any(future_abs <= 0.10)),
            "hit_55_45": bool(np.any(future_abs <= 0.05)),
            "touch_50_band": bool(np.any(future_abs <= deadband)),
            "favorite_flip": bool(np.any(states == -initial_sign) if initial_sign else False),
            "first_flip_delay_min": _first_opposite_delay_minutes(
                times, mids, i, initial_sign, deadband
            ),
        })
    return out


def _summarize_conditionals(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_games"] = int(len(g))
        for col in ("hit_60_40", "hit_55_45", "touch_50_band", "favorite_flip"):
            k = int(g[col].astype(bool).sum())
            n = int(len(g))
            lo, hi = wilson(k, n)
            row[f"{col}_n"] = k
            row[f"{col}_rate"] = k / n if n else float("nan")
            row[f"{col}_wilson_lo"] = lo
            row[f"{col}_wilson_hi"] = hi
        row["median_max_gap_closure_fraction"] = float(g["max_gap_closure_fraction"].median())
        flips = g["first_flip_delay_min"].dropna()
        row["median_first_flip_delay_min"] = float(flips.median()) if len(flips) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def static_self_check(show: bool = True) -> dict[str, Any]:
    out = {
        "version": VERSION,
        "api_called": False,
        "orders_sent": False,
        "quote_mid_primary": True,
        "one_first_entry_observation_per_game_phase_band": True,
        "wilson_95_intervals": True,
        "minute_cross_count_is_lower_bound": True,
        "ok": True,
    }
    if show:
        print("=" * 112)
        print("SPORTS TWO-TEAM GAP ANALYSIS STATIC CHECK — OFFLINE")
        print("=" * 112)
        for k, v in out.items():
            print(f"{k:55s}: {v}")
    return out


def run(*, run_dir: str | Path, max_quote_spread: float = 0.20,
        cross_deadband: float = 0.005, show: bool = True) -> dict[str, Any]:
    static_self_check(show=show)
    root = Path(run_dir).expanduser().resolve()
    markets = pd.read_csv(root / "markets.csv")
    paths = pd.read_csv(root / "minute_paths.csv.gz", compression="gzip")

    for c in ("yes_mid", "quote_spread", "end_period_ts", "elapsed_from_start_s"):
        paths[c] = pd.to_numeric(paths[c], errors="coerce")
    paths = paths[
        paths["yes_mid"].between(0, 1, inclusive="both")
        & paths["quote_spread"].between(0, max_quote_spread, inclusive="both")
    ].copy()

    paths = paths[
        ((paths["phase"] == "pregame") & paths["elapsed_from_start_s"].between(-24 * 3600, 0, inclusive="left"))
        | ((paths["phase"] == "in_game") & (paths["elapsed_from_start_s"] >= 0))
    ].copy()

    event_phase_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []

    for (ticker, phase), g in paths.groupby(["ticker", "phase"], sort=False):
        if len(g) < 3:
            continue
        base_meta = {
            "ticker": ticker,
            "event_ticker": str(g["event_ticker"].iloc[0]),
            "series_ticker": str(g["series_ticker"].iloc[0]),
            "series_title": str(g["series_title"].iloc[0]),
            "sport": str(g["sport"].iloc[0]),
            "phase": phase,
        }
        event_phase_rows.append({**base_meta, **_event_phase_summary(g, cross_deadband)})
        for r in _conditional_rows(g, phase, cross_deadband):
            conditional_rows.append({**base_meta, **r})

    event_phase = pd.DataFrame(event_phase_rows)
    conditional = pd.DataFrame(conditional_rows)

    event_phase.to_csv(root / "game_phase_crossing_summary.csv", index=False)
    conditional.to_csv(root / "conditional_start_observations.csv", index=False)

    overall = _summarize_conditionals(conditional, ["phase", "start_band"])
    by_sport = _summarize_conditionals(conditional, ["sport", "phase", "start_band"])
    by_series = _summarize_conditionals(conditional, ["series_ticker", "phase", "start_band"])
    overall.to_csv(root / "conditional_reversion_overall.csv", index=False)
    by_sport.to_csv(root / "conditional_reversion_by_sport.csv", index=False)
    by_series.to_csv(root / "conditional_reversion_by_series.csv", index=False)

    cross_dist = (
        event_phase.groupby(["phase", "cross_count"]).size().rename("games").reset_index()
        if not event_phase.empty else pd.DataFrame(columns=["phase", "cross_count", "games"])
    )
    cross_dist.to_csv(root / "cross_count_distribution.csv", index=False)

    headline: dict[str, Any] = {
        "version": VERSION,
        "run_dir": str(root),
        "markets_in_manifest": int(len(markets)),
        "markets_with_valid_paths": int(paths["ticker"].nunique()),
        "events_with_valid_paths": int(paths["event_ticker"].nunique()),
        "minute_rows_after_quote_filter": int(len(paths)),
        "max_quote_spread": float(max_quote_spread),
        "cross_deadband": float(cross_deadband),
        "pregame_games": int((event_phase["phase"] == "pregame").sum()) if not event_phase.empty else 0,
        "in_game_games": int((event_phase["phase"] == "in_game").sum()) if not event_phase.empty else 0,
        "pregame_any_flip_rate": (
            float((event_phase.loc[event_phase["phase"] == "pregame", "cross_count"] > 0).mean())
            if (not event_phase.empty and (event_phase["phase"] == "pregame").any()) else None
        ),
        "in_game_any_flip_rate": (
            float((event_phase.loc[event_phase["phase"] == "in_game", "cross_count"] > 0).mean())
            if (not event_phase.empty and (event_phase["phase"] == "in_game").any()) else None
        ),
        "pregame_multi_flip_rate": (
            float((event_phase.loc[event_phase["phase"] == "pregame", "cross_count"] >= 2).mean())
            if (not event_phase.empty and (event_phase["phase"] == "pregame").any()) else None
        ),
        "in_game_multi_flip_rate": (
            float((event_phase.loc[event_phase["phase"] == "in_game", "cross_count"] >= 2).mean())
            if (not event_phase.empty and (event_phase["phase"] == "in_game").any()) else None
        ),
        "orders_sent": False,
        "api_called": False,
        "scientific_note": (
            "Minute-resolution quote-mid crossings are a lower bound. Conditional rows use "
            "one first entry per game/phase/band to avoid overweighting long games. "
            "No result can establish a guaranteed reversion."
        ),
    }
    (root / "analysis_headline.json").write_text(json.dumps(headline, indent=2, default=str), encoding="utf-8")

    if show:
        print("\n" + "=" * 128)
        print("SPORTS TWO-TEAM GAP / FAVORITE-FLIP ANALYSIS")
        print("=" * 128)
        for k, v in headline.items():
            if k != "scientific_note":
                print(f"{k:40s}: {v}")
        print("\nCONDITIONAL GAP CLOSURE / CROSSING — OVERALL")
        if overall.empty:
            print("No conditional rows.")
        else:
            show_cols = [
                "phase", "start_band", "n_games",
                "hit_60_40_rate", "hit_55_45_rate",
                "touch_50_band_rate", "favorite_flip_rate",
                "favorite_flip_wilson_lo", "favorite_flip_wilson_hi",
                "median_max_gap_closure_fraction", "median_first_flip_delay_min",
            ]
            print(overall[show_cols].to_string(index=False))
        print("\nREPEATED CROSS DISTRIBUTION")
        print(cross_dist.to_string(index=False) if not cross_dist.empty else "none")
        print("\nIMPORTANT: these are empirical frequencies, not guarantees.")
        print("Output:", root)

    return headline


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-quote-spread", type=float, default=0.20)
    ap.add_argument("--cross-deadband", type=float, default=0.005)
    a = ap.parse_args()
    run(
        run_dir=a.run_dir,
        max_quote_spread=a.max_quote_spread,
        cross_deadband=a.cross_deadband,
        show=True,
    )


if __name__ == "__main__":
    _main()

__all__ = ["VERSION", "wilson", "static_self_check", "run"]
