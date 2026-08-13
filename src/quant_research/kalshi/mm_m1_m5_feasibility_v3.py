from __future__ import annotations

"""Schema-safe driver for the M1-M5 market-making feasibility replay.

Kalshi orderbook snapshots may encode the NO-side levels on the legacy NO-leg
price scale or on the unified YES-leg price scale. This driver probes each
recorded session first, selects one convention for that entire session, then
runs the unchanged V1 FIFO replay through the robust V2 summary layer.

No strategy threshold, queue rule, fill rule, markout horizon, or quality gate
is changed here.
"""

import json
import time
from pathlib import Path

import numpy as np

from . import mm_m1_m5_feasibility as _v1
from . import mm_m1_m5_feasibility_v2 as _v2

STUDY_VERSION = "M1_M5_MM_FEASIBILITY_V3_SCHEMA_SAFE"

_ORIG_SCAN_BOOKS = _v1._scan_books
_ACTIVE_ORIENTATION = None
_SCHEMA_RESULTS = {}


def _candidate_parts(row):
    yes_bids = sorted(_v1._levels(row.get("yes_bids") or []), key=lambda z: z[0], reverse=True)
    raw_other = _v1._levels(row.get("yes_asks") or [])
    if not yes_bids or not raw_other:
        return None
    direct_asks = sorted(raw_other, key=lambda z: z[0])
    complement_asks = sorted([(1.0 - p, q) for p, q in raw_other], key=lambda z: z[0])
    return yes_bids, direct_asks, complement_asks


def _spread(yes_bids, asks):
    if not yes_bids or not asks:
        return np.nan
    bid = float(yes_bids[0][0])
    ask = float(asks[0][0])
    if not (0.0 <= bid < ask <= 1.0):
        return np.nan
    return ask - bid


def _probe_session_orientation(session_dir, series, max_relevant_rows=50000):
    path = Path(session_dir) / "full_books.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    prefix_re = _v1._prefix_regex(series)
    n = vd = vc = both = neither = 0
    sd_all, sc_all = [], []
    t0 = time.time()

    with path.open("rb") as f:
        for raw in f:
            if prefix_re is not None and prefix_re.search(raw) is None:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue

            ticker = _v1._get_ticker(row)
            if not ticker or ticker.split("-")[0] not in series:
                continue

            parts = _candidate_parts(row)
            if parts is None:
                continue

            n += 1
            bids, direct, comp = parts
            sd, sc = _spread(bids, direct), _spread(bids, comp)
            okd, okc = np.isfinite(sd), np.isfinite(sc)

            if okd:
                vd += 1
                sd_all.append(float(sd))
            if okc:
                vc += 1
                sc_all.append(float(sc))
            if okd and okc:
                both += 1
            elif not okd and not okc:
                neither += 1

            if n >= int(max_relevant_rows):
                break

    if n == 0:
        raise RuntimeError(f"No relevant crypto full-book rows found in {path}")

    rd, rc = vd / n, vc / n
    md = float(np.median(sd_all)) if sd_all else np.nan
    mc = float(np.median(sc_all)) if sc_all else np.nan

    if rd > rc + 0.05:
        orientation, reason = "UNIFIED_YES_PRICE", "higher valid-book rate"
    elif rc > rd + 0.05:
        orientation, reason = "LEGACY_NO_PRICE", "higher valid-book rate"
    elif np.isfinite(md) and np.isfinite(mc):
        orientation = "UNIFIED_YES_PRICE" if md <= mc else "LEGACY_NO_PRICE"
        reason = "similar validity; tighter median spread"
    elif np.isfinite(md):
        orientation, reason = "UNIFIED_YES_PRICE", "only direct convention produced valid spreads"
    elif np.isfinite(mc):
        orientation, reason = "LEGACY_NO_PRICE", "only complement convention produced valid spreads"
    else:
        raise RuntimeError(
            f"Could not infer book price convention: direct={vd}, complement={vc}, rows={n}"
        )

    return {
        "session": Path(session_dir).name,
        "orientation": orientation,
        "reason": reason,
        "probe_rows": n,
        "direct_valid_rows": vd,
        "direct_valid_pct": 100.0 * rd,
        "direct_median_spread_c": 100.0 * md if np.isfinite(md) else np.nan,
        "complement_valid_rows": vc,
        "complement_valid_pct": 100.0 * rc,
        "complement_median_spread_c": 100.0 * mc if np.isfinite(mc) else np.nan,
        "both_valid_rows": both,
        "neither_valid_rows": neither,
        "probe_seconds": time.time() - t0,
    }


def _normalize_book_schema_safe(row, close_ts, start_minute):
    global _ACTIVE_ORIENTATION

    t = _v1._event_ts(row)
    if not np.isfinite(t):
        return None

    parts = _candidate_parts(row)
    if parts is None:
        return None
    yes_bids, direct_asks, complement_asks = parts

    if _ACTIVE_ORIENTATION == "UNIFIED_YES_PRICE":
        yes_asks = direct_asks
    elif _ACTIVE_ORIENTATION == "LEGACY_NO_PRICE":
        yes_asks = complement_asks
    else:
        raise RuntimeError("Book orientation was not probed before normalization")

    bid1, bid1q = yes_bids[0]
    ask1, ask1q = yes_asks[0]
    if not (0.0 <= bid1 < ask1 <= 1.0):
        return None

    bid2, bid2q = yes_bids[1] if len(yes_bids) > 1 else (np.nan, 0.0)
    ask2, ask2q = yes_asks[1] if len(yes_asks) > 1 else (np.nan, 0.0)
    mid = 0.5 * (bid1 + ask1)
    spread = ask1 - bid1
    den = bid1q + ask1q
    imbalance = (bid1q - ask1q) / den if den > 0 else np.nan
    minute = (t - (close_ts - 900.0)) / 60.0

    return _v1.BookSnap(
        t=float(t),
        minute=float(minute),
        bid1=float(bid1),
        bid1_qty=float(bid1q),
        bid2=float(bid2) if np.isfinite(bid2) else np.nan,
        bid2_qty=float(bid2q),
        ask1=float(ask1),
        ask1_qty=float(ask1q),
        ask2=float(ask2) if np.isfinite(ask2) else np.nan,
        ask2_qty=float(ask2q),
        mid=float(mid),
        spread=float(spread),
        imbalance=float(imbalance),
    )


def _scan_books_schema_safe(
    session_dir,
    series,
    start_minute,
    end_minute,
    max_markout_s,
    min_book_coverage_pct,
    max_edge_gap_s,
    book_writer=None,
):
    global _ACTIVE_ORIENTATION

    probe = _probe_session_orientation(session_dir, series)
    _SCHEMA_RESULTS[str(Path(session_dir).resolve())] = probe
    _ACTIVE_ORIENTATION = probe["orientation"]

    print(
        f"[{Path(session_dir).name}] book schema={probe['orientation']} | "
        f"direct valid={probe['direct_valid_pct']:.2f}% "
        f"(median {probe['direct_median_spread_c']:.3f}c) | "
        f"complement valid={probe['complement_valid_pct']:.2f}% "
        f"(median {probe['complement_median_spread_c']:.3f}c)"
    )

    return _ORIG_SCAN_BOOKS(
        session_dir,
        series,
        start_minute,
        end_minute,
        max_markout_s,
        min_book_coverage_pct,
        max_edge_gap_s,
        book_writer=book_writer,
    )


def run_m1_m5_mm_feasibility(*args, **kwargs):
    old_norm = _v1._normalize_book
    old_scan = _v1._scan_books
    _SCHEMA_RESULTS.clear()

    _v1._normalize_book = _normalize_book_schema_safe
    _v1._scan_books = _scan_books_schema_safe
    try:
        result = _v2.run_m1_m5_mm_feasibility(*args, **kwargs)
    finally:
        _v1._normalize_book = old_norm
        _v1._scan_books = old_scan

    result["book_schema_probe"] = dict(_SCHEMA_RESULTS)

    if kwargs.get("show", True):
        print("\nBOOK SCHEMA PROBE")
        for p in _SCHEMA_RESULTS.values():
            print(
                f"  {p['session']}: {p['orientation']} ({p['reason']}); "
                f"probe_rows={p['probe_rows']:,}"
            )
        print(
            "\nOnly recorder price normalization changed. "
            "M1-M5 timing, 1ct quotes, FIFO/trade-through rules, markouts, "
            "and the 80%/5s quality gate are unchanged."
        )

    return result


__all__ = ["STUDY_VERSION", "run_m1_m5_mm_feasibility"]
