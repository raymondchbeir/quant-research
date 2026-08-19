from __future__ import annotations

"""Development-only post-peak retracement study for the 5c/Q5 deep-tail strategy.

This extends V5's post-fill reversion analysis.  V5 measured how HIGH the executable
outcome bid rebounds after a conservative 5c/Q5 fill.  This module instead asks what
happens AFTER that peak and BEFORE M5:

- peak executable bid and peak time;
- minimum executable bid observed after the peak;
- executable M5 bid;
- drawdown from peak to post-peak minimum;
- drawdown from peak to M5;
- fraction of the 5c->peak rebound retained at M5;
- conditional post-peak hit rates for fixed absolute levels;
- conditional retracement hit rates of 1/2/3/5/10/15/20 cents from the peak;
- conditional rebound-retention thresholds (75%, 50%, 25%, 0% of the rebound).

The event anchor and book mechanics are inherited from V5 unchanged.  This reads ONLY
the already-inspected 24h DEVELOPMENT source.  No API calls.  No orders.  Source data
are read-only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_reversion_exit_dev_v5 as V5

VERSION = "MM_DEEP_TAIL_POST_PEAK_DEV_V5_1"
HARD_BOUND_SESSION = V1.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_post_peak_dev_v5_1"

ENTRY_C = 5.0
EPS = 1e-10
ABS_LEVELS_C = (5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50)
PEAK_DRAWDOWNS_C = (1, 2, 3, 5, 10, 15, 20)
REBOUND_RETAIN_FRACS = (0.75, 0.50, 0.25, 0.00)


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _m5_outcome_bid(tail: str, snap: dict | None) -> float:
    if not snap:
        return np.nan
    if tail == "YES":
        return float(snap["yes_bid"])
    return 1.0 - float(snap["yes_ask"])


def _detail(anchors: pd.DataFrame, books: dict) -> pd.DataFrame:
    rows = []

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
        asks = np.asarray([float(x[0]["ask"]) for x in obs], dtype=float)
        mids = np.asarray([float(x[0]["mid"]) for x in obs], dtype=float)
        times = np.asarray([float(x[1]["receipt_s"]) for x in obs], dtype=float)

        if len(bids) == 0 or not np.isfinite(bids).any():
            continue

        peak_i = int(np.nanargmax(bids))
        peak_bid = float(bids[peak_i])
        peak_ask = float(asks[peak_i])
        peak_mid = float(mids[peak_i])
        peak_s = float(times[peak_i])

        post_bids = bids[peak_i:]
        post_asks = asks[peak_i:]
        post_mids = mids[peak_i:]
        post_times = times[peak_i:]

        post_min_i_local = int(np.nanargmin(post_bids)) if len(post_bids) else 0
        post_min_bid = float(post_bids[post_min_i_local]) if len(post_bids) else np.nan
        post_min_s = float(post_times[post_min_i_local]) if len(post_times) else np.nan

        m5_bid = _m5_outcome_bid(tail, bd.get("m5"))
        peak_c = 100.0 * peak_bid
        post_min_c = 100.0 * post_min_bid
        m5_c = 100.0 * m5_bid if np.isfinite(m5_bid) else np.nan

        rebound_c = peak_c - ENTRY_C
        retained = (
            (m5_c - ENTRY_C) / rebound_c
            if np.isfinite(m5_c) and rebound_c > EPS
            else np.nan
        )

        row = {
            **a.to_dict(),
            "peak_bid_c": peak_c,
            "peak_ask_c": 100.0 * peak_ask,
            "peak_mid_c": 100.0 * peak_mid,
            "seconds_fill_to_peak": peak_s - active_s,
            "seconds_peak_to_m5_snapshot": (
                float(bd["m5"]["receipt_s"]) - peak_s
                if bd.get("m5") and np.isfinite(float(bd["m5"].get("receipt_s", np.nan)))
                else np.nan
            ),
            "post_peak_min_bid_c": post_min_c,
            "seconds_peak_to_post_min": post_min_s - peak_s,
            "m5_bid_c": m5_c,
            "peak_to_post_min_drawdown_c": peak_c - post_min_c,
            "peak_to_m5_drawdown_c": peak_c - m5_c if np.isfinite(m5_c) else np.nan,
            "m5_minus_entry_c": m5_c - ENTRY_C if np.isfinite(m5_c) else np.nan,
            "peak_rebound_from_entry_c": rebound_c,
            "rebound_fraction_retained_at_m5": retained,
            "post_peak_bid_observations": int(len(post_bids)),
        }

        # Absolute levels: conditional on the peak having reached the level, did the bid
        # later trade/book back DOWN to or through that level before M5?
        for level in ABS_LEVELS_C:
            eligible = peak_c >= float(level) - EPS
            hit = bool(np.any(post_bids * 100.0 <= float(level) + EPS)) if eligible else False
            row[f"post_peak_down_hit_{level}c_eligible"] = eligible
            row[f"post_peak_down_hit_{level}c"] = hit
            if eligible and hit:
                idx = int(np.where(post_bids * 100.0 <= float(level) + EPS)[0][0])
                row[f"seconds_peak_to_down_hit_{level}c"] = float(post_times[idx] - peak_s)
            else:
                row[f"seconds_peak_to_down_hit_{level}c"] = np.nan

        # Relative drawdowns from each position's own peak.
        for dd in PEAK_DRAWDOWNS_C:
            threshold = peak_c - float(dd)
            eligible = threshold >= 0.0
            hit = bool(np.any(post_bids * 100.0 <= threshold + EPS)) if eligible else False
            row[f"drawdown_{dd}c_hit"] = hit
            if hit:
                idx = int(np.where(post_bids * 100.0 <= threshold + EPS)[0][0])
                row[f"seconds_to_drawdown_{dd}c"] = float(post_times[idx] - peak_s)
            else:
                row[f"seconds_to_drawdown_{dd}c"] = np.nan

        # Retention thresholds relative to the 5c->peak rebound.  Example: 50% retention
        # threshold = 5c + .5 * (peak-5c).  'hit' means the executable bid subsequently
        # fell to or below that threshold before M5.
        if rebound_c > EPS:
            for frac in REBOUND_RETAIN_FRACS:
                threshold = ENTRY_C + float(frac) * rebound_c
                hit = bool(np.any(post_bids * 100.0 <= threshold + EPS))
                tag = int(round(100 * frac))
                row[f"rebound_retention_{tag}pct_threshold_c"] = threshold
                row[f"fell_to_{tag}pct_rebound_retention"] = hit
                if hit:
                    idx = int(np.where(post_bids * 100.0 <= threshold + EPS)[0][0])
                    row[f"seconds_to_{tag}pct_rebound_retention"] = float(post_times[idx] - peak_s)
                else:
                    row[f"seconds_to_{tag}pct_rebound_retention"] = np.nan
        else:
            for frac in REBOUND_RETAIN_FRACS:
                tag = int(round(100 * frac))
                row[f"rebound_retention_{tag}pct_threshold_c"] = np.nan
                row[f"fell_to_{tag}pct_rebound_retention"] = False
                row[f"seconds_to_{tag}pct_rebound_retention"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _distribution(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "peak_bid_c",
        "post_peak_min_bid_c",
        "m5_bid_c",
        "peak_to_post_min_drawdown_c",
        "peak_to_m5_drawdown_c",
        "peak_rebound_from_entry_c",
        "m5_minus_entry_c",
        "rebound_fraction_retained_at_m5",
        "seconds_fill_to_peak",
        "seconds_peak_to_post_min",
    ]
    rows = []
    for metric in metrics:
        x = pd.to_numeric(detail.get(metric), errors="coerce").dropna()
        rows.append({
            "metric": metric,
            "n": int(len(x)),
            "mean": float(x.mean()) if len(x) else np.nan,
            "q10": float(x.quantile(.10)) if len(x) else np.nan,
            "q25": float(x.quantile(.25)) if len(x) else np.nan,
            "median": float(x.median()) if len(x) else np.nan,
            "q75": float(x.quantile(.75)) if len(x) else np.nan,
            "q90": float(x.quantile(.90)) if len(x) else np.nan,
            "min": float(x.min()) if len(x) else np.nan,
            "max": float(x.max()) if len(x) else np.nan,
        })
    return pd.DataFrame(rows)


def _absolute_level_hits(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in ABS_LEVELS_C:
        elig_col = f"post_peak_down_hit_{level}c_eligible"
        hit_col = f"post_peak_down_hit_{level}c"
        sec_col = f"seconds_peak_to_down_hit_{level}c"
        elig = detail[detail[elig_col].astype(bool)].copy()
        hit = elig[elig[hit_col].astype(bool)].copy()
        secs = pd.to_numeric(hit[sec_col], errors="coerce").dropna()
        rows.append({
            "level_c": int(level),
            "peaks_at_or_above_level": int(len(elig)),
            "fell_back_to_or_below_level": int(len(hit)),
            "conditional_fallback_rate": float(len(hit) / len(elig)) if len(elig) else np.nan,
            "median_seconds_peak_to_level": float(secs.median()) if len(secs) else np.nan,
            "q25_seconds_peak_to_level": float(secs.quantile(.25)) if len(secs) else np.nan,
            "q75_seconds_peak_to_level": float(secs.quantile(.75)) if len(secs) else np.nan,
        })
    return pd.DataFrame(rows)


def _drawdown_hits(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dd in PEAK_DRAWDOWNS_C:
        hit_col = f"drawdown_{dd}c_hit"
        sec_col = f"seconds_to_drawdown_{dd}c"
        h = detail[detail[hit_col].astype(bool)].copy()
        secs = pd.to_numeric(h[sec_col], errors="coerce").dropna()
        rows.append({
            "drawdown_from_peak_c": int(dd),
            "positions": int(len(detail)),
            "hit": int(len(h)),
            "hit_rate": float(len(h) / len(detail)) if len(detail) else np.nan,
            "median_seconds_after_peak": float(secs.median()) if len(secs) else np.nan,
            "q25_seconds_after_peak": float(secs.quantile(.25)) if len(secs) else np.nan,
            "q75_seconds_after_peak": float(secs.quantile(.75)) if len(secs) else np.nan,
        })
    return pd.DataFrame(rows)


def _retention_hits(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = detail[pd.to_numeric(detail["peak_rebound_from_entry_c"], errors="coerce") > EPS].copy()
    for frac in REBOUND_RETAIN_FRACS:
        tag = int(round(100 * frac))
        hit_col = f"fell_to_{tag}pct_rebound_retention"
        sec_col = f"seconds_to_{tag}pct_rebound_retention"
        h = q[q[hit_col].astype(bool)].copy()
        secs = pd.to_numeric(h[sec_col], errors="coerce").dropna()
        rows.append({
            "rebound_retention_threshold_pct": tag,
            "positions_with_positive_rebound": int(len(q)),
            "fell_to_or_below_threshold": int(len(h)),
            "hit_rate": float(len(h) / len(q)) if len(q) else np.nan,
            "median_seconds_after_peak": float(secs.median()) if len(secs) else np.nan,
            "q25_seconds_after_peak": float(secs.quantile(.25)) if len(secs) else np.nan,
            "q75_seconds_after_peak": float(secs.quantile(.75)) if len(secs) else np.nan,
        })
    return pd.DataFrame(rows)


def run_post_peak_dev(source_session, *, hard_bind=True, show=True):
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
        print("DEEP-TAIL POST-PEAK -> M5 DISTRIBUTION DEV V5.1")
        print("=" * 150)
        print("Source:", source)
        print("Entry event: conservative 5c/Q5 fill, same as V5")
        print("Question: AFTER each position's executable-bid peak, what does price do before M5?")
        print("DEVELOPMENT ONLY — no validation data")
        print()

    meta = V1._metadata(source)
    trades, trade_stats = V1._load_trades(source, meta, show=show)
    anchors = V5._entry_anchors(meta, trades)
    relevant = set(anchors["ticker"].astype(str)) if len(anchors) else set()

    if show:
        print(f"Entry anchors: {len(anchors):,} filled side-positions | relevant tickers={len(relevant):,}")

    books = V5._scan_books(source, meta, relevant, show=show)
    detail = _detail(anchors, books)
    dist = _distribution(detail)
    abs_hits = _absolute_level_hits(detail)
    dd_hits = _drawdown_hits(detail)
    retention = _retention_hits(detail)

    out = _new_output(source.name)
    detail.to_csv(out / "post_peak_detail.csv", index=False)
    dist.to_csv(out / "post_peak_distribution.csv", index=False)
    abs_hits.to_csv(out / "post_peak_absolute_level_hits.csv", index=False)
    dd_hits.to_csv(out / "post_peak_drawdown_hits.csv", index=False)
    retention.to_csv(out / "post_peak_rebound_retention_hits.csv", index=False)

    summary = {
        "version": VERSION,
        "source_session": str(source),
        "filled_positions": int(len(detail)),
        "entry_c": ENTRY_C,
        "trade_rows_selected": int(trade_stats.get("selected", 0)) if isinstance(trade_stats, dict) else None,
        "output_dir": str(out),
        "scientific_status": "DEVELOPMENT_ONLY",
    }
    pd.Series(summary).to_json(out / "summary.json", indent=2)

    if show:
        print("=" * 150)
        print("POST-PEAK DISTRIBUTION")
        print("=" * 150)
        print(dist.to_string(index=False))
        print()
        print("CONDITIONAL ABSOLUTE LEVEL FALLBACKS AFTER PEAK")
        print(abs_hits.to_string(index=False))
        print()
        print("RELATIVE DRAWDOWN HITS AFTER PEAK")
        print(dd_hits.to_string(index=False))
        print()
        print("REBOUND RETENTION AFTER PEAK")
        print(retention.to_string(index=False))
        print()
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "anchors": anchors,
        "detail": detail,
        "distribution": dist,
        "absolute_level_hits": abs_hits,
        "drawdown_hits": dd_hits,
        "retention_hits": retention,
        "output_dir": out,
    }
