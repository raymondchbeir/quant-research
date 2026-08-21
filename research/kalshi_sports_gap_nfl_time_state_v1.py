from __future__ import annotations

"""Offline NFL time-of-game x probability-state reversion study.

Reads an already completed sports gap run. No API calls, account/order endpoints,
live-Q50 imports, or orders.

Scientific unit: one first entry per game x elapsed-time bucket x favorite-probability
band. The future path tracks the SAME side that was favorite at entry, so a later
favorite identity change cannot create a false reversion.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

VERSION = "KALSHI_SPORTS_NFL_TIME_STATE_V1_OFFLINE"
SERIES = "KXNFLGAME"
SPREAD_CAPS = [0.01, 0.02]
TIME_BINS_MIN = [0, 30, 60, 90, 120, 150, 180, np.inf]
TIME_LABELS = ["0-30", "30-60", "60-90", "90-120", "120-150", "150-180", "180+"]
FAV_BINS = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.000001]
FAV_LABELS = ["50-55", "55-60", "60-70", "70-80", "80-90", "90-100"]
DROP_POINTS = [0.05, 0.10, 0.20]
TARGETS = [0.80, 0.70, 0.60, 0.55, 0.50]


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def season_regime_from_ts(ts: Any) -> str:
    t = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
    if pd.isna(t):
        return "unknown"
    month = int(t.month)
    if month in (7, 8):
        return "preseason"
    if month in (9, 10, 11, 12, 1, 2):
        return "regular_or_postseason"
    return "unknown"


def first_hit_delay(times: np.ndarray, values: np.ndarray, threshold: float) -> tuple[bool, float | None]:
    idx = np.flatnonzero(values <= threshold + 1e-12)
    if not len(idx):
        return False, None
    j = int(idx[0])
    return True, float(times[j] - times[0]) / 60.0


def prepare_paths(root: Path, spread_cap: float) -> pd.DataFrame:
    paths = pd.read_csv(root / "minute_paths.csv.gz", compression="gzip")
    paths = paths[paths["series_ticker"].astype(str) == SERIES].copy()
    for c in ("yes_mid", "quote_spread", "end_period_ts", "elapsed_from_start_s"):
        paths[c] = pd.to_numeric(paths[c], errors="coerce")
    paths = paths[
        (paths["phase"].astype(str) == "in_game")
        & (paths["elapsed_from_start_s"] >= 0)
        & paths["yes_mid"].between(0, 1, inclusive="both")
        & paths["quote_spread"].between(0, spread_cap, inclusive="both")
    ].copy()
    paths["elapsed_min"] = paths["elapsed_from_start_s"] / 60.0
    paths["time_bucket"] = pd.cut(
        paths["elapsed_min"], bins=TIME_BINS_MIN, labels=TIME_LABELS,
        right=False, include_lowest=True,
    ).astype(str)

    sampled = root / "sampled_events.csv"
    if sampled.exists():
        meta = pd.read_csv(sampled)
        keep = [c for c in ["ticker", "game_start_ts", "game_start_iso"] if c in meta.columns]
        if "ticker" in keep:
            meta = meta[keep].drop_duplicates("ticker")
            paths = paths.merge(meta, on="ticker", how="left", suffixes=("", "_sample"))
    if "game_start_ts" not in paths.columns:
        paths["game_start_ts"] = paths["end_period_ts"] - paths["elapsed_from_start_s"]
    paths["game_start_ts"] = pd.to_numeric(paths["game_start_ts"], errors="coerce")
    paths["season_regime"] = paths["game_start_ts"].map(season_regime_from_ts)
    return paths


def observation_rows(p: pd.DataFrame, spread_cap: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker, g0 in p.groupby("ticker", sort=False):
        g0 = g0.sort_values("end_period_ts").reset_index(drop=True)
        if len(g0) < 3:
            continue
        mids = g0["yes_mid"].to_numpy(float)
        times = g0["end_period_ts"].to_numpy(float)
        elapsed = g0["elapsed_min"].to_numpy(float)
        spreads = g0["quote_spread"].to_numpy(float)
        fav_now = np.maximum(mids, 1 - mids)
        band_labels = np.asarray(pd.cut(
            fav_now, bins=FAV_BINS, labels=FAV_LABELS,
            right=False, include_lowest=True,
        ).astype(str))
        time_labels = g0["time_bucket"].astype(str).to_numpy()

        for tb in TIME_LABELS:
            for band in FAV_LABELS:
                idx = np.flatnonzero((time_labels == tb) & (band_labels == band))
                if not len(idx):
                    continue
                i = int(idx[0])  # one first entry per game x time bucket x band
                start_mid = float(mids[i])
                side = 1 if start_mid >= 0.5 else -1
                start_side_prob = start_mid if side == 1 else 1.0 - start_mid
                future_mid = mids[i:]
                future_side_prob = future_mid if side == 1 else 1.0 - future_mid
                future_times = times[i:]

                row: dict[str, Any] = {
                    "spread_cap": spread_cap,
                    "ticker": str(ticker),
                    "event_ticker": str(g0["event_ticker"].iloc[i]),
                    "season_regime": str(g0["season_regime"].iloc[i]),
                    "time_bucket": tb,
                    "start_band": band,
                    "entry_ts": int(times[i]),
                    "entry_elapsed_min": float(elapsed[i]),
                    "entry_yes_mid": start_mid,
                    "entry_favorite_side": "YES" if side == 1 else "NO",
                    "entry_favorite_prob": float(start_side_prob),
                    "entry_spread": float(spreads[i]),
                    "future_min_entry_side_prob": float(np.min(future_side_prob)),
                    "max_drop_points": float(start_side_prob - np.min(future_side_prob)),
                }

                for drop in DROP_POINTS:
                    hit, delay = first_hit_delay(future_times, future_side_prob, start_side_prob - drop)
                    tag = f"drop_{int(round(drop * 100))}"
                    row[f"{tag}_hit"] = bool(hit)
                    row[f"{tag}_delay_min"] = delay

                for target in TARGETS:
                    tag = f"target_{int(round(target * 100))}"
                    eligible = bool(start_side_prob > target + 1e-12)
                    row[f"{tag}_eligible"] = eligible
                    if eligible:
                        hit, delay = first_hit_delay(future_times, future_side_prob, target)
                        row[f"{tag}_hit"] = bool(hit)
                        row[f"{tag}_delay_min"] = delay
                    else:
                        row[f"{tag}_hit"] = False
                        row[f"{tag}_delay_min"] = None
                rows.append(row)
    return rows


def _delay_stats(g: pd.DataFrame, hit_col: str, delay_col: str) -> dict[str, float]:
    d = pd.to_numeric(g.loc[g[hit_col].astype(bool), delay_col], errors="coerce").dropna()
    return {
        "delay_p25_min": float(d.quantile(0.25)) if len(d) else np.nan,
        "delay_median_min": float(d.median()) if len(d) else np.nan,
        "delay_p75_min": float(d.quantile(0.75)) if len(d) else np.nan,
    }


def summarize_drops(obs: pd.DataFrame) -> pd.DataFrame:
    work = pd.concat([obs, obs.assign(season_regime="all_nfl")], ignore_index=True)
    out: list[dict[str, Any]] = []
    groups = ["spread_cap", "season_regime", "time_bucket", "start_band"]
    for keys, g in work.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        n = int(g["ticker"].nunique())
        row["n_games"] = n
        row["median_entry_favorite_prob"] = float(g["entry_favorite_prob"].median())
        row["median_entry_spread"] = float(g["entry_spread"].median())
        row["median_max_drop_points"] = float(g["max_drop_points"].median())
        for drop in DROP_POINTS:
            tag = f"drop_{int(round(drop * 100))}"
            k = int(g[f"{tag}_hit"].astype(bool).sum())
            lo, hi = wilson(k, n)
            row[f"{tag}_n"] = k
            row[f"{tag}_rate"] = k / n if n else np.nan
            row[f"{tag}_wilson_lo"] = lo
            row[f"{tag}_wilson_hi"] = hi
            row.update({f"{tag}_{k2}": v for k2, v in _delay_stats(g, f"{tag}_hit", f"{tag}_delay_min").items()})
        out.append(row)
    return pd.DataFrame(out)


def summarize_targets(obs: pd.DataFrame) -> pd.DataFrame:
    work = pd.concat([obs, obs.assign(season_regime="all_nfl")], ignore_index=True)
    out: list[dict[str, Any]] = []
    groups = ["spread_cap", "season_regime", "time_bucket", "start_band"]
    for keys, g in work.groupby(groups, dropna=False):
        base = dict(zip(groups, keys))
        for target in TARGETS:
            tag = f"target_{int(round(target * 100))}"
            eg = g[g[f"{tag}_eligible"].astype(bool)].copy()
            n = int(eg["ticker"].nunique())
            if n == 0:
                continue
            k = int(eg[f"{tag}_hit"].astype(bool).sum())
            lo, hi = wilson(k, n)
            row = dict(base)
            row.update({
                "target_entry_side_prob": target,
                "n_games": n,
                "hit_n": k,
                "hit_rate": k / n,
                "hit_wilson_lo": lo,
                "hit_wilson_hi": hi,
            })
            row.update(_delay_stats(eg, f"{tag}_hit", f"{tag}_delay_min"))
            out.append(row)
    return pd.DataFrame(out)


def candidate_table(drop_summary: pd.DataFrame) -> pd.DataFrame:
    q = drop_summary[
        (drop_summary["spread_cap"] == 0.02)
        & (drop_summary["season_regime"] == "all_nfl")
        & (drop_summary["n_games"] >= 20)
    ].copy()
    if q.empty:
        return q
    cols = [
        "time_bucket", "start_band", "n_games", "median_entry_favorite_prob",
        "drop_5_rate", "drop_5_wilson_lo", "drop_5_delay_median_min",
        "drop_10_rate", "drop_10_wilson_lo", "drop_10_delay_median_min",
        "drop_20_rate", "drop_20_wilson_lo", "drop_20_delay_median_min",
        "median_max_drop_points",
    ]
    q = q.sort_values(["drop_10_wilson_lo", "drop_10_rate", "n_games"], ascending=[False, False, False])
    return q[cols].reset_index(drop=True)


def static_self_check(show: bool = True) -> dict[str, Any]:
    out = {
        "version": VERSION,
        "offline_only": True,
        "series_filter": SERIES,
        "api_called": False,
        "orders_sent": False,
        "live_q50_modules_imported": False,
        "spread_caps": SPREAD_CAPS,
        "time_buckets_min": TIME_LABELS,
        "favorite_bands": FAV_LABELS,
        "tracks_same_entry_favorite_side": True,
        "one_first_entry_per_game_time_bucket_band": True,
        "season_regime_rule": "Jul-Aug preseason; Sep-Feb regular_or_postseason",
        "wilson_95_intervals": True,
        "ok": True,
    }
    if show:
        print("=" * 128)
        print("NFL TIME-STATE REVERSION — STATIC CHECK")
        print("=" * 128)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    return out


def run(run_dir: str | Path, show: bool = True) -> dict[str, Any]:
    static_self_check(show=show)
    root = Path(run_dir).expanduser().resolve()
    all_obs: list[dict[str, Any]] = []
    for spread_cap in SPREAD_CAPS:
        p = prepare_paths(root, spread_cap)
        if p.empty:
            continue
        all_obs.extend(observation_rows(p, spread_cap))
    obs = pd.DataFrame(all_obs)
    if obs.empty:
        raise RuntimeError("No NFL in-game time-state observations were produced")

    drops = summarize_drops(obs)
    targets = summarize_targets(obs)
    candidates = candidate_table(drops)

    obs.to_csv(root / "nfl_time_state_observations.csv", index=False)
    drops.to_csv(root / "nfl_time_state_drop_summary.csv", index=False)
    targets.to_csv(root / "nfl_time_state_target_summary.csv", index=False)
    candidates.to_csv(root / "nfl_time_state_candidate_rank.csv", index=False)

    headline = {
        "version": VERSION,
        "nfl_markets": int(obs["ticker"].nunique()),
        "observation_rows": int(len(obs)),
        "primary_spread_cap": 0.02,
        "robustness_spread_cap": 0.01,
        "time_buckets": TIME_LABELS,
        "favorite_bands": FAV_LABELS,
        "drop_points": DROP_POINTS,
        "targets": TARGETS,
        "season_regimes": sorted(obs["season_regime"].dropna().astype(str).unique().tolist()),
        "api_called": False,
        "orders_sent": False,
        "scientific_note": (
            "Each cell uses one first entry per game x time bucket x probability band. "
            "The future path follows the same side that was favorite at entry. A game may "
            "appear in multiple time buckets, so cells are not mutually independent. "
            "This is still quote-mid research, not an execution or PnL backtest."
        ),
    }
    (root / "nfl_time_state_headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")

    if show:
        print("\n" + "=" * 128)
        print("PRIMARY — ALL NFL, SPREAD <= 2c")
        print("=" * 128)
        primary = drops[(drops.spread_cap == 0.02) & (drops.season_regime == "all_nfl")].copy()
        cols = [
            "time_bucket", "start_band", "n_games",
            "drop_5_rate", "drop_5_wilson_lo", "drop_5_delay_median_min",
            "drop_10_rate", "drop_10_wilson_lo", "drop_10_delay_median_min",
            "drop_20_rate", "drop_20_wilson_lo", "drop_20_delay_median_min",
        ]
        print(primary[cols].to_string(index=False))

        print("\n" + "=" * 128)
        print("PRIMARY TARGETS — SAME ENTRY FAVORITE SIDE, SPREAD <= 2c")
        print("=" * 128)
        tq = targets[(targets.spread_cap == 0.02) & (targets.season_regime == "all_nfl")].copy()
        print(tq[["time_bucket", "start_band", "target_entry_side_prob", "n_games", "hit_rate", "hit_wilson_lo", "hit_wilson_hi", "delay_median_min"]].to_string(index=False))

        print("\n" + "=" * 128)
        print("SEASON REGIME COMPARISON — SPREAD <= 2c, CELLS WITH >= 10 GAMES")
        print("=" * 128)
        rq = drops[(drops.spread_cap == 0.02) & (drops.season_regime != "all_nfl") & (drops.n_games >= 10)].copy()
        print(rq[["season_regime", "time_bucket", "start_band", "n_games", "drop_10_rate", "drop_10_wilson_lo", "drop_10_delay_median_min", "drop_20_rate"]].to_string(index=False))

        print("\n" + "=" * 128)
        print("CANDIDATE CELLS — RANKED BY 10-POINT REVERSION WILSON LOWER BOUND")
        print("=" * 128)
        print(candidates.head(30).to_string(index=False) if not candidates.empty else "none")
        print("\nIMPORTANT: this does not establish executable alpha.")
        print("Output:", root)
    return headline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run(a.run_dir, show=True)


if __name__ == "__main__":
    main()
