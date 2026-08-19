from __future__ import annotations

"""Development-only full post-peak BBO distribution study for the 5c/Q5 deep-tail strategy.

Scientific question
-------------------
V5.1 measured post-peak extrema (peak bid, later minimum bid, M5 bid).  This module keeps
the exact same 5c/Q5 fill event and executable-bid peak definition, but measures the FULL
outcome BBO path from that peak until true M5:

- bid, ask, midpoint, and spread distributions;
- both raw book-update distributions and time-weighted distributions;
- per-position time-weighted BBO summaries;
- time-since-peak bucket distributions, with state durations split correctly across bins;
- occupancy fractions at economically relevant bid/ask levels.

Why time weighting matters
--------------------------
Raw book updates overweight periods with rapid quote churn.  The primary distribution in
this study is therefore TIME-WEIGHTED: each observed BBO state receives weight equal to the
seconds for which it remained the last observed state.  The raw-update distribution is
reported only as a diagnostic.

Peak definition
---------------
The peak is the same operational peak used by V5.1: the maximum executable outcome bid
observed after the conservative 5c/Q5 entry fill becomes actionable and before M5.  This is
NOT the older V14/V16 visual pillar classifier.

This reads ONLY the already-inspected 24h DEVELOPMENT source.  No validation data, no API
calls, no orders, and no source modification.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_reversion_exit_dev_v5 as V5

VERSION = "MM_DEEP_TAIL_POST_PEAK_BOOK_DISTRIBUTION_DEV_V5_2"
HARD_BOUND_SESSION = V1.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_post_peak_book_distribution_dev_v5_2"

EPS = 1e-10
METRICS = ("bid_c", "ask_c", "mid_c", "spread_c")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
TIME_BUCKETS_S = (
    (0.0, 0.25, "0-250ms"),
    (0.25, 1.0, "250ms-1s"),
    (1.0, 5.0, "1-5s"),
    (5.0, 15.0, "5-15s"),
    (15.0, 30.0, "15-30s"),
    (30.0, 60.0, "30-60s"),
    (60.0, 120.0, "60-120s"),
    (120.0, np.inf, "120s-M5"),
)
BID_LEVELS_C = (5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50)
ASK_LEVELS_C = (5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50)


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _atomic_json(path: Path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _weighted_quantile(values, weights, quantiles=QUANTILES):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[mask]
    w = w[mask]
    if len(v) == 0:
        return {q: np.nan for q in quantiles}
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    out = {}
    for q in quantiles:
        target = float(q) * total
        i = int(np.searchsorted(cw, target, side="left"))
        i = min(max(i, 0), len(v) - 1)
        out[q] = float(v[i])
    return out


def _summary_row(metric: str, values, weights=None, *, label="ALL"):
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    x = x[finite]
    if len(x) == 0:
        return {
            "group": label,
            "metric": metric,
            "n_states": 0,
            "weight_seconds": 0.0,
            "mean": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    if weights is None:
        q = np.quantile(x, QUANTILES)
        return {
            "group": label,
            "metric": metric,
            "n_states": int(len(x)),
            "weight_seconds": np.nan,
            "mean": float(np.mean(x)),
            "q10": float(q[0]),
            "q25": float(q[1]),
            "median": float(q[2]),
            "q75": float(q[3]),
            "q90": float(q[4]),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    w0 = np.asarray(weights, dtype=float)[finite]
    good = np.isfinite(w0) & (w0 > 0)
    x = x[good]
    w = w0[good]
    if len(x) == 0:
        return _summary_row(metric, [], None, label=label)
    qq = _weighted_quantile(x, w)
    return {
        "group": label,
        "metric": metric,
        "n_states": int(len(x)),
        "weight_seconds": float(np.sum(w)),
        "mean": float(np.average(x, weights=w)),
        "q10": qq[0.10],
        "q25": qq[0.25],
        "median": qq[0.50],
        "q75": qq[0.75],
        "q90": qq[0.90],
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _post_peak_states(anchors: pd.DataFrame, books: dict):
    rows = []
    position_rows = []

    for _, a in anchors.iterrows():
        ticker = str(a["ticker"])
        tail = str(a["tail"])
        bd = books.get(ticker) or {}
        if not bool(bd.get("coverage_eligible", False)):
            continue

        active_s = float(a["exit_active_s"])
        path = [
            s for s in (bd.get("path") or [])
            if float(s["receipt_s"]) + EPS >= active_s
        ]
        if not path:
            continue

        obs = [(V5._outcome_book(tail, s), s) for s in path]
        bids = np.asarray([float(x[0]["bid"]) for x in obs], dtype=float)
        if len(bids) == 0 or not np.isfinite(bids).any():
            continue

        peak_i = int(np.nanargmax(bids))
        peak_book, peak_state = obs[peak_i]
        peak_s = float(peak_state["receipt_s"])
        m5 = bd.get("m5") or {}
        m5_s = float(m5.get("receipt_s", np.nan))
        if not np.isfinite(m5_s) or m5_s + EPS < peak_s:
            continue

        post = obs[peak_i:]
        position_id = f"{ticker}|{tail}"
        total_duration = max(0.0, m5_s - peak_s)

        position_rows.append({
            "position_id": position_id,
            "ticker": ticker,
            "series": str(a.get("series") or ""),
            "close_time": str(a.get("close_time") or ""),
            "tail": tail,
            "entry_filled_qty": float(a["entry_filled_qty"]),
            "peak_receipt_s": peak_s,
            "peak_bid_c": 100.0 * float(peak_book["bid"]),
            "peak_ask_c": 100.0 * float(peak_book["ask"]),
            "peak_mid_c": 100.0 * float(peak_book["mid"]),
            "peak_spread_c": 100.0 * (float(peak_book["ask"]) - float(peak_book["bid"])),
            "post_peak_duration_s": total_duration,
            "post_peak_book_updates": int(len(post)),
        })

        for j, (ob, state) in enumerate(post):
            t0 = float(state["receipt_s"])
            if t0 > m5_s + EPS:
                break
            if j + 1 < len(post):
                t1 = min(float(post[j + 1][1]["receipt_s"]), m5_s)
            else:
                t1 = m5_s
            dt = max(0.0, t1 - t0)
            bid = 100.0 * float(ob["bid"])
            ask = 100.0 * float(ob["ask"])
            mid = 100.0 * float(ob["mid"])
            rows.append({
                "position_id": position_id,
                "ticker": ticker,
                "series": str(a.get("series") or ""),
                "close_time": str(a.get("close_time") or ""),
                "tail": tail,
                "entry_filled_qty": float(a["entry_filled_qty"]),
                "peak_bid_c": 100.0 * float(peak_book["bid"]),
                "peak_receipt_s": peak_s,
                "receipt_s": t0,
                "seconds_since_peak": max(0.0, t0 - peak_s),
                "interval_end_s": t1,
                "interval_end_since_peak": max(0.0, t1 - peak_s),
                "dt_s": dt,
                "bid_c": bid,
                "ask_c": ask,
                "mid_c": mid,
                "spread_c": ask - bid,
                "is_peak_state": bool(j == 0),
            })

    return pd.DataFrame(rows), pd.DataFrame(position_rows)


def _pooled_distributions(states: pd.DataFrame):
    tw_rows = []
    raw_rows = []
    groups = [("ALL", states)]
    for tail, g in states.groupby("tail", sort=True):
        groups.append((str(tail), g))

    for label, g in groups:
        for metric in METRICS:
            tw_rows.append(_summary_row(metric, g[metric].to_numpy(), g["dt_s"].to_numpy(), label=label))
            raw_rows.append(_summary_row(metric, g[metric].to_numpy(), None, label=label))
    return pd.DataFrame(tw_rows), pd.DataFrame(raw_rows)


def _per_position(states: pd.DataFrame, positions: pd.DataFrame):
    rows = []
    pos_meta = positions.set_index("position_id") if len(positions) else pd.DataFrame()
    for pid, g in states.groupby("position_id", sort=True):
        w = pd.to_numeric(g["dt_s"], errors="coerce").fillna(0).to_numpy(float)
        row = dict(pos_meta.loc[pid].to_dict()) if len(pos_meta) and pid in pos_meta.index else {"position_id": pid}
        row["position_id"] = pid
        row["observed_state_seconds"] = float(w.sum())
        row["states"] = int(len(g))
        for metric in METRICS:
            x = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
            qq = _weighted_quantile(x, w)
            good = np.isfinite(x) & np.isfinite(w) & (w > 0)
            if good.any():
                row[f"tw_mean_{metric}"] = float(np.average(x[good], weights=w[good]))
                row[f"tw_q25_{metric}"] = qq[0.25]
                row[f"tw_median_{metric}"] = qq[0.50]
                row[f"tw_q75_{metric}"] = qq[0.75]
                row[f"min_{metric}"] = float(np.nanmin(x))
                row[f"max_{metric}"] = float(np.nanmax(x))
            else:
                for k in ("tw_mean", "tw_q25", "tw_median", "tw_q75", "min", "max"):
                    row[f"{k}_{metric}"] = np.nan

        total = float(w.sum())
        bid = pd.to_numeric(g["bid_c"], errors="coerce").to_numpy(float)
        ask = pd.to_numeric(g["ask_c"], errors="coerce").to_numpy(float)
        for level in BID_LEVELS_C:
            occ = float(w[(np.isfinite(bid)) & (bid >= level - EPS)].sum()) if total > EPS else np.nan
            row[f"frac_time_bid_ge_{level}c"] = occ / total if total > EPS else np.nan
        for level in ASK_LEVELS_C:
            occ = float(w[(np.isfinite(ask)) & (ask <= level + EPS)].sum()) if total > EPS else np.nan
            row[f"frac_time_ask_le_{level}c"] = occ / total if total > EPS else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _time_bucket_rows(states: pd.DataFrame):
    pieces = []
    for _, r in states.iterrows():
        s0 = float(r["seconds_since_peak"])
        s1 = float(r["interval_end_since_peak"])
        if s1 <= s0 + EPS:
            continue
        for lo, hi, label in TIME_BUCKETS_S:
            left = max(s0, float(lo))
            right = min(s1, float(hi)) if np.isfinite(hi) else s1
            overlap = max(0.0, right - left)
            if overlap <= EPS:
                continue
            z = r.to_dict()
            z["bucket"] = label
            z["bucket_weight_s"] = overlap
            pieces.append(z)
    if not pieces:
        return pd.DataFrame(), pd.DataFrame()

    p = pd.DataFrame(pieces)
    rows = []
    for bucket, g in p.groupby("bucket", sort=False):
        positions = int(g["position_id"].nunique())
        weight = float(pd.to_numeric(g["bucket_weight_s"], errors="coerce").fillna(0).sum())
        for metric in METRICS:
            x = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
            w = pd.to_numeric(g["bucket_weight_s"], errors="coerce").to_numpy(float)
            z = _summary_row(metric, x, w, label=bucket)
            z["positions_contributing"] = positions
            z["bucket_weight_seconds"] = weight
            rows.append(z)
    return pd.DataFrame(rows), p


def _occupancy(states: pd.DataFrame):
    w = pd.to_numeric(states["dt_s"], errors="coerce").fillna(0).to_numpy(float)
    bid = pd.to_numeric(states["bid_c"], errors="coerce").to_numpy(float)
    ask = pd.to_numeric(states["ask_c"], errors="coerce").to_numpy(float)
    total = float(w.sum())
    rows = []
    for level in BID_LEVELS_C:
        good = np.isfinite(bid) & (bid >= float(level) - EPS)
        sec = float(w[good].sum())
        rows.append({
            "side_metric": "BID_GE",
            "level_c": int(level),
            "time_seconds": sec,
            "fraction_of_post_peak_time": sec / total if total > EPS else np.nan,
        })
    for level in ASK_LEVELS_C:
        good = np.isfinite(ask) & (ask <= float(level) + EPS)
        sec = float(w[good].sum())
        rows.append({
            "side_metric": "ASK_LE",
            "level_c": int(level),
            "time_seconds": sec,
            "fraction_of_post_peak_time": sec / total if total > EPS else np.nan,
        })
    return pd.DataFrame(rows)


def run_post_peak_book_distribution_dev(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected development source {HARD_BOUND_SESSION}, got {source.name}")
    if hard_bind and "mm_event_m0_m5_oos_cycle_q10_v1" not in str(source.parent):
        raise RuntimeError("Expected source under mm_event_m0_m5_oos_cycle_q10_v1")

    required = [
        source / "book_top3_events.jsonl",
        source / "trades_event_time.jsonl",
        source / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + " | ".join(missing))

    if show:
        print("=" * 150)
        print("DEEP-TAIL FULL POST-PEAK BID/ASK DISTRIBUTION DEV V5.2")
        print("=" * 150)
        print("Source:", source)
        print("Peak: same max executable outcome bid used by V5.1")
        print("Primary distribution: TIME-WEIGHTED BBO states from peak until true M5")
        print("Also reports raw-update distribution and time-since-peak buckets")
        print("DEVELOPMENT ONLY — no validation data")
        print()

    meta = V1._metadata(source)
    trades, trade_stats = V1._load_trades(source, meta, show=show)
    anchors = V5._entry_anchors(meta, trades)
    relevant = set(anchors["ticker"].astype(str)) if len(anchors) else set()

    if show:
        print(f"Entry anchors: {len(anchors):,} filled side-positions | relevant tickers={len(relevant):,}")
        print("Scanning full post-fill book paths and true M5 snapshots...")

    books = V5._scan_books(source, meta, relevant, show=show)
    states, positions = _post_peak_states(anchors, books)
    if states.empty:
        raise RuntimeError("No eligible post-peak states found.")

    tw, raw = _pooled_distributions(states)
    per_position = _per_position(states, positions)
    bucket_dist, bucket_pieces = _time_bucket_rows(states)
    occupancy = _occupancy(states)

    out = _new_output(source.name)
    states.to_csv(out / "post_peak_bbo_states.csv", index=False)
    positions.to_csv(out / "post_peak_positions.csv", index=False)
    tw.to_csv(out / "time_weighted_distribution.csv", index=False)
    raw.to_csv(out / "raw_update_distribution.csv", index=False)
    per_position.to_csv(out / "per_position_time_weighted.csv", index=False)
    bucket_dist.to_csv(out / "time_since_peak_bucket_distribution.csv", index=False)
    occupancy.to_csv(out / "post_peak_level_occupancy.csv", index=False)

    summary = {
        "version": VERSION,
        "source": str(source),
        "entry_anchors": int(len(anchors)),
        "eligible_positions": int(positions["position_id"].nunique()) if len(positions) else 0,
        "post_peak_states": int(len(states)),
        "total_time_weight_seconds": float(pd.to_numeric(states["dt_s"], errors="coerce").fillna(0).sum()),
        "trade_stats": trade_stats,
        "scientific_status": "DEVELOPMENT_ONLY",
        "primary_weighting": "time-weighted last-observed BBO state",
        "peak_definition": "maximum executable outcome bid after actionable 5c/Q5 fill and before M5",
    }
    _atomic_json(out / "summary.json", summary)

    if show:
        print("=" * 150)
        print("TIME-WEIGHTED POST-PEAK BBO DISTRIBUTION — ALL POSITIONS")
        print("=" * 150)
        print(tw[tw["group"].eq("ALL")].to_string(index=False))
        print()
        print("=" * 150)
        print("TIME-SINCE-PEAK BBO DISTRIBUTION")
        print("=" * 150)
        print(bucket_dist.to_string(index=False))
        print()
        print("=" * 150)
        print("POST-PEAK LEVEL OCCUPANCY")
        print("=" * 150)
        print(occupancy.to_string(index=False))
        print()
        print("IMPORTANT: time-weighted table is primary; raw update counts can overweight quote-churn periods.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "states": states,
        "positions": positions,
        "time_weighted_distribution": tw,
        "raw_update_distribution": raw,
        "per_position": per_position,
        "time_bucket_distribution": bucket_dist,
        "occupancy": occupancy,
        "output_dir": str(out),
    }
