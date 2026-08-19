from __future__ import annotations

"""V6.3 DEVELOPMENT replay: unchanged V6 strategy with mmap time-sliced raw extraction.

Why this exists
---------------
V6.2 still paid the cost of scanning the entire multi-GiB book JSONL once with grep/rg.
That is unnecessary because the recorder file is written in receipt-time order and the
5c/Q5 entry anchors already tell us the small set of absolute time windows that can
possibly matter.

V6.3 therefore:
1. memory-maps the raw JSONL file (macOS page cache / unified memory is used on demand);
2. builds a tiny coarse byte-offset -> receipt-time index by sampling one line every
   few MiB, without scanning the whole file;
3. converts the relevant filled-ticker time windows into byte ranges;
4. reads/parses ONLY those byte ranges;
5. persists the same compact BBO/trade cache consumed by the unchanged V6 strategy.

The GPU is intentionally not used: this workload is dominated by text I/O, delimiter
search, and JSON parsing rather than dense numerical arithmetic.  mmap + avoiding most
of the 5.7 GiB read is the high-leverage optimization on Apple silicon.

Strategy and execution semantics are unchanged from V6.  DEVELOPMENT source only.
No validation data. No API calls. No orders.
"""

import bisect
import json
import mmap
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_trailing_passive_exit_dev_v6 as V6

VERSION = "MM_DEEP_TAIL_TRAILING_PASSIVE_EXIT_DEV_V6_3_MMAP_TIME_SLICE"
CACHE_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_fast_cache_v4"
EPS = V6.EPS
M1_S = V6.M1_S
M5_S = V6.M5_S

INDEX_STRIDE_BYTES = 2 * 1024 * 1024
TIME_PAD_S = 2.0
MAX_COARSE_DISORDER_S = 2.0

try:
    import orjson as _orjson  # type: ignore
except Exception:  # pragma: no cover
    _orjson = None


def _loads(raw: bytes):
    if _orjson is not None:
        return _orjson.loads(raw)
    return json.loads(raw)


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _fast_iso_s(x):
    if x is None:
        return np.nan
    try:
        s = str(x)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return float(datetime.fromisoformat(s).timestamp())
    except Exception:
        return np.nan


def _receipt_from_raw_line(raw: bytes):
    """Extract receipt_time from one JSONL row with minimal work.

    Recorder JSON is compact, but this handles optional whitespace as a fallback by
    decoding the full JSON only when the direct byte search fails.
    """
    key = b'"receipt_time"'
    i = raw.find(key)
    if i >= 0:
        colon = raw.find(b":", i + len(key))
        if colon >= 0:
            q1 = raw.find(b'"', colon + 1)
            if q1 >= 0:
                q2 = raw.find(b'"', q1 + 1)
                if q2 > q1:
                    try:
                        s = raw[q1 + 1:q2].decode("ascii")
                        if s.endswith("Z"):
                            s = s[:-1] + "+00:00"
                        return float(datetime.fromisoformat(s).timestamp())
                    except Exception:
                        pass
    try:
        r = _loads(raw)
        return _fast_iso_s(r.get("receipt_time"))
    except Exception:
        return np.nan


def _line_at_or_after(mm: mmap.mmap, pos: int):
    size = len(mm)
    if size <= 0 or pos >= size:
        return None
    if pos <= 0:
        start = 0
    else:
        nl = mm.find(b"\n", pos)
        if nl < 0 or nl + 1 >= size:
            return None
        start = nl + 1
    end = mm.find(b"\n", start)
    if end < 0:
        end = size
    if end <= start:
        return None
    return start, end, mm[start:end]


def _sample_valid_time(mm: mmap.mmap, pos: int, attempts: int = 8):
    cur = max(0, int(pos))
    for _ in range(attempts):
        z = _line_at_or_after(mm, cur)
        if z is None:
            return None
        start, end, raw = z
        ts = _receipt_from_raw_line(raw)
        if np.isfinite(ts):
            return int(start), float(ts)
        cur = int(end + 1)
    return None


def _build_coarse_index(path: Path, *, show=True):
    path = Path(path)
    size = int(path.stat().st_size)
    if size <= 0:
        raise RuntimeError(f"Empty raw file: {path}")

    offsets = []
    times = []
    with path.open("rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for off in range(0, size, INDEX_STRIDE_BYTES):
                z = _sample_valid_time(mm, off)
                if z is None:
                    continue
                start, ts = z
                if offsets and start == offsets[-1]:
                    continue
                offsets.append(start)
                times.append(ts)

            z = _sample_valid_time(mm, max(0, size - INDEX_STRIDE_BYTES))
            if z is not None and (not offsets or z[0] != offsets[-1]):
                offsets.append(z[0])
                times.append(z[1])
        finally:
            mm.close()

    if len(offsets) < 2:
        raise RuntimeError(f"Could not build coarse receipt-time index for {path}")

    # The recorder is receipt-ordered.  Allow tiny scheduler/interleave noise but refuse
    # the optimization if coarse samples reveal material time reversal.
    diffs = np.diff(np.asarray(times, dtype=float))
    worst_back = float(np.min(diffs)) if len(diffs) else 0.0
    if worst_back < -MAX_COARSE_DISORDER_S:
        raise RuntimeError(
            f"Raw file is not sufficiently receipt-time ordered for mmap slicing: "
            f"worst coarse reversal={worst_back:.3f}s in {path.name}"
        )

    # Clamp tiny local reversals to preserve bisectability.  Exact ticker/elapsed filters
    # later prevent the padding from changing strategy semantics.
    monotone_times = np.maximum.accumulate(np.asarray(times, dtype=float)).tolist()

    if show:
        print(
            f"MMAP INDEX: {path.name} | size={size/(1024**3):.2f} GiB | "
            f"samples={len(offsets):,} | stride={INDEX_STRIDE_BYTES/(1024**2):.1f} MiB"
        )
    return {
        "path": str(path),
        "size": size,
        "offsets": [int(x) for x in offsets],
        "times": [float(x) for x in monotone_times],
        "worst_coarse_time_step_s": worst_back,
    }


def _merge_time_intervals(intervals):
    q = sorted((float(a), float(b)) for a, b in intervals if np.isfinite(a) and np.isfinite(b) and b > a)
    if not q:
        return []
    out = [[q[0][0], q[0][1]]]
    for a, b in q[1:]:
        if a <= out[-1][1] + TIME_PAD_S:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(float(a), float(b)) for a, b in out]


def _relevant_time_intervals(meta: dict, anchors: pd.DataFrame):
    min_active = anchors.groupby("ticker")["exit_active_s"].min().astype(float).to_dict()
    out = []
    for ticker in sorted(set(anchors["ticker"].astype(str))):
        m = meta.get(ticker)
        if not m:
            continue
        start = float(min_active[ticker]) - TIME_PAD_S
        end = float(m["window_start_s"]) + M5_S + TIME_PAD_S
        if end > start:
            out.append((start, end))
    return _merge_time_intervals(out), min_active


def _intervals_to_byte_ranges(index: dict, intervals):
    offsets = index["offsets"]
    times = index["times"]
    size = int(index["size"])
    ranges = []

    for start_s, end_s in intervals:
        # Pick a sample before the padded start and a sample after the padded end.
        i = bisect.bisect_right(times, float(start_s)) - 1
        j = bisect.bisect_left(times, float(end_s))
        i = max(0, i - 1)
        j = min(len(offsets) - 1, j + 1)
        byte_start = 0 if i <= 0 else int(offsets[i])
        byte_end = size if j >= len(offsets) - 1 else int(offsets[j])
        if byte_end <= byte_start:
            byte_end = min(size, byte_start + 2 * INDEX_STRIDE_BYTES)
        ranges.append((byte_start, byte_end))

    # Merge overlapping byte ranges.
    ranges.sort()
    merged = []
    for a, b in ranges:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(int(a), int(b)) for a, b in merged]


def _iter_byte_ranges(path: Path, ranges):
    with Path(path).open("rb") as fh:
        for start, end in ranges:
            fh.seek(int(start))
            if start > 0:
                fh.readline()  # discard possible partial line
            while fh.tell() < int(end):
                raw = fh.readline()
                if not raw:
                    break
                yield raw


def _levels_fixed(raw_levels):
    vals = []
    levels = []
    for x in (raw_levels or [])[:3]:
        try:
            p = float(x[0])
            q = max(0.0, float(x[1]))
        except Exception:
            continue
        if np.isfinite(p) and np.isfinite(q):
            levels.append((p, q))
    for i in range(3):
        if i < len(levels):
            vals.extend([levels[i][0], levels[i][1]])
        else:
            vals.extend([np.nan, np.nan])
    return vals, levels


def _build_book_cache_sliced(source: Path, anchors: pd.DataFrame, meta: dict, intervals, min_active, cache_dir: Path, *, show=True):
    raw_path = source / "book_top3_events.jsonl"
    index = _build_coarse_index(raw_path, show=show)
    ranges = _intervals_to_byte_ranges(index, intervals)
    selected_bytes = sum(b - a for a, b in ranges)
    if show:
        print(
            f"MMAP SLICE: book ranges={len(ranges)} | read={selected_bytes/(1024**2):.1f} MiB "
            f"({100.0*selected_bytes/index['size']:.2f}% of raw file)"
        )

    tickers = set(anchors["ticker"].astype(str))
    rows = []
    last_by_ticker = {}
    parsed_lines = 0
    relevant_lines = 0

    for raw in _iter_byte_ranges(raw_path, ranges):
        parsed_lines += 1
        try:
            r = _loads(raw)
        except Exception:
            continue
        ticker = str(r.get("ticker") or "")
        if ticker not in tickers:
            continue
        relevant_lines += 1
        e = _f(r.get("elapsed_s"))
        if not (np.isfinite(e) and M1_S <= e < M5_S):
            continue
        if not bool(r.get("valid_bbo")):
            continue
        bid = _f(r.get("yes_bid"))
        ask = _f(r.get("yes_ask"))
        if not (np.isfinite(bid) and np.isfinite(ask) and 0.0 <= bid < ask <= 1.0):
            continue
        bp, bid_levels = _levels_fixed(r.get("bid_levels"))
        ap, ask_levels = _levels_fixed(r.get("ask_levels"))
        if not bid_levels or not ask_levels:
            continue

        ws = float(meta[ticker]["window_start_s"])
        rt = ws + float(e)
        mid = _f(r.get("mid"), 0.5 * (bid + ask))
        m5state = {
            "yes_bid": float(bid),
            "yes_ask": float(ask),
            "yes_mid": float(mid),
            "bid_levels": [[float(p), float(q)] for p, q in bid_levels],
            "ask_levels": [[float(p), float(q)] for p, q in ask_levels],
            "receipt_s": float(rt),
            "snapshot_elapsed_s": float(e),
            "true_m5_finalized": True,
        }
        last_by_ticker[ticker] = m5state

        if rt + EPS < float(min_active.get(ticker, -np.inf)):
            continue
        rows.append((
            ticker, float(rt), float(e), float(bid), float(ask), float(mid),
            bp[0], bp[1], bp[2], bp[3], bp[4], bp[5],
            ap[0], ap[1], ap[2], ap[3], ap[4], ap[5],
        ))

    if not rows:
        raise RuntimeError("MMAP time-sliced book cache returned no relevant states")

    cols = [
        "ticker", "receipt_s", "elapsed_s", "yes_bid", "yes_ask", "yes_mid",
        "bid_p1", "bid_q1", "bid_p2", "bid_q2", "bid_p3", "bid_q3",
        "ask_p1", "ask_q1", "ask_p2", "ask_q2", "ask_p3", "ask_q3",
    ]
    df = pd.DataFrame.from_records(rows, columns=cols)
    df.sort_values(["ticker", "receipt_s"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Every relevant ticker must have a last valid pre-M5 state.  If not, refuse rather
    # than silently changing economics.
    missing_m5 = sorted(tickers - set(last_by_ticker))
    if missing_m5:
        raise RuntimeError(f"Missing pre-M5 book state for relevant tickers: {missing_m5}")

    df.to_pickle(cache_dir / "relevant_post_fill_bbo.pkl")
    V6._atomic_json(cache_dir / "m5_top3.json", last_by_ticker)
    V6._atomic_json(cache_dir / "book_slice_index.json", {
        "index": index,
        "time_intervals": intervals,
        "byte_ranges": ranges,
        "selected_bytes": int(selected_bytes),
        "parsed_lines": int(parsed_lines),
        "relevant_lines": int(relevant_lines),
    })

    if show:
        print(
            f"MMAP SLICE: book DONE | parsed={parsed_lines:,} | relevant={relevant_lines:,} | "
            f"cached={len(df):,} | M5={len(last_by_ticker):,}"
        )
    return df, last_by_ticker, index, ranges


def _raw_exchange_s(r):
    z = _f(r.get("ts_ms"))
    if np.isfinite(z):
        return float(z) / 1000.0
    return _fast_iso_s(r.get("exchange_time"))


def _build_trade_cache_sliced(source: Path, anchors: pd.DataFrame, meta: dict, intervals, min_active, cache_dir: Path, *, show=True):
    raw_path = source / "trades_event_time.jsonl"
    index = _build_coarse_index(raw_path, show=show)
    ranges = _intervals_to_byte_ranges(index, intervals)
    selected_bytes = sum(b - a for a, b in ranges)
    if show:
        print(
            f"MMAP SLICE: trade ranges={len(ranges)} | read={selected_bytes/(1024**2):.1f} MiB "
            f"({100.0*selected_bytes/index['size']:.2f}% of raw file)"
        )

    tickers = set(anchors["ticker"].astype(str))
    rows = []
    stats = Counter()
    parsed_lines = 0
    relevant_lines = 0

    for raw in _iter_byte_ranges(raw_path, ranges):
        parsed_lines += 1
        try:
            r = _loads(raw)
        except Exception:
            continue
        ticker = str(r.get("ticker") or "")
        if ticker not in tickers:
            continue
        relevant_lines += 1
        e = _f(r.get("elapsed_s"))
        if not (np.isfinite(e) and M1_S <= e < M5_S):
            continue
        yes = _f(r.get("yes_price"))
        qty = _f(r.get("qty"))
        side = str(r.get("taker_book_side") or "").lower()
        if not (
            np.isfinite(yes) and 0.0 <= yes <= 1.0
            and np.isfinite(qty) and qty > 0.0
            and side in {"bid", "ask"}
        ):
            continue
        rt = float(meta[ticker]["window_start_s"]) + float(e)
        x = _raw_exchange_s(r)
        if not np.isfinite(x):
            exec_s = rt
            source_name = "RECEIPT_FALLBACK"
        elif x > rt + EPS:
            exec_s = rt
            source_name = "CLAMP_EXCHANGE_AFTER_RECEIPT"
        else:
            exec_s = float(x)
            source_name = "EXCHANGE_TIME"
        obs_s = max(float(exec_s), rt)
        if obs_s + EPS < float(min_active.get(ticker, -np.inf)):
            continue
        rows.append((
            ticker, float(exec_s), float(rt), float(obs_s), float(yes), float(qty),
            side, str(r.get("trade_id") or ""),
        ))
        stats[source_name] += 1

    cols = ["ticker", "exec_s", "receipt_s", "obs_s", "yes_price", "qty", "taker_book_side", "trade_id"]
    df = pd.DataFrame.from_records(rows, columns=cols)
    if len(df):
        df.sort_values(["ticker", "exec_s", "receipt_s", "trade_id"], kind="mergesort", inplace=True)
        df.reset_index(drop=True, inplace=True)
    df.to_pickle(cache_dir / "relevant_post_fill_trades.pkl")
    V6._atomic_json(cache_dir / "trade_clock_stats.json", dict(stats))
    V6._atomic_json(cache_dir / "trade_slice_index.json", {
        "index": index,
        "time_intervals": intervals,
        "byte_ranges": ranges,
        "selected_bytes": int(selected_bytes),
        "parsed_lines": int(parsed_lines),
        "relevant_lines": int(relevant_lines),
    })

    if show:
        print(
            f"MMAP SLICE: trades DONE | parsed={parsed_lines:,} | relevant={relevant_lines:,} | cached={len(df):,}"
        )
    return df, dict(stats), index, ranges


def _load_or_build_mmap_cache(source: Path, anchors: pd.DataFrame, *, rebuild=False, show=True):
    cache_dir = CACHE_ROOT / source.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    bbo_path = cache_dir / "relevant_post_fill_bbo.pkl"
    trade_path = cache_dir / "relevant_post_fill_trades.pkl"
    m5_path = cache_dir / "m5_top3.json"
    stats_path = cache_dir / "trade_clock_stats.json"
    manifest_path = cache_dir / "manifest.json"

    if not rebuild and all(p.exists() for p in (bbo_path, trade_path, m5_path, stats_path, manifest_path)):
        manifest = V6._read_json(manifest_path, {}) or {}
        if manifest.get("cache_version") == VERSION:
            if show:
                print("MMAP TIME-SLICE CACHE HIT:", cache_dir)
            return (
                pd.read_pickle(bbo_path),
                pd.read_pickle(trade_path),
                V6._read_json(m5_path, {}) or {},
                V6._read_json(stats_path, {}) or {},
                "MMAP_TIME_SLICE_CACHE_HIT",
            )

    meta = V1._metadata(source)
    intervals, min_active = _relevant_time_intervals(meta, anchors)
    if not intervals:
        raise RuntimeError("No relevant absolute time intervals for 5c/Q5 anchors")

    if show:
        total_s = sum(b - a for a, b in intervals)
        print(
            f"MMAP TIME-SLICE CACHE MISS — {len(intervals)} merged time intervals | "
            f"total wall-clock span read target={total_s/60.0:.1f} min"
        )
        print("Using mmap/page cache (unified memory) and skipping irrelevant hours entirely.")

    for p in (bbo_path, trade_path, m5_path, stats_path, manifest_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    bbo, m5, book_index, book_ranges = _build_book_cache_sliced(
        source, anchors, meta, intervals, min_active, cache_dir, show=show
    )
    trades, stats, trade_index, trade_ranges = _build_trade_cache_sliced(
        source, anchors, meta, intervals, min_active, cache_dir, show=show
    )

    manifest = {
        "cache_version": VERSION,
        "strategy_version": V6.VERSION,
        "source": str(source),
        "source_session": source.name,
        "anchors": int(len(anchors)),
        "tickers": sorted(set(anchors["ticker"].astype(str))),
        "merged_time_intervals": intervals,
        "book_bbo_rows": int(len(bbo)),
        "trade_rows": int(len(trades)),
        "book_raw_size_bytes": int(book_index["size"]),
        "book_selected_bytes": int(sum(b - a for a, b in book_ranges)),
        "trade_raw_size_bytes": int(trade_index["size"]),
        "trade_selected_bytes": int(sum(b - a for a, b in trade_ranges)),
        "json_parser": "orjson" if _orjson is not None else "stdlib-json",
        "cache_method": "mmap coarse receipt-time index + relevant absolute-time byte slicing",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    V6._atomic_json(manifest_path, manifest)
    return bbo, trades, m5, stats, "MMAP_TIME_SLICE_CACHE_BUILT"


def run_trailing_passive_exit_dev(source_session, *, hard_bind=True, rebuild_cache=False, show=True):
    """Run unchanged V6 strategy with the V6.3 mmap time-slice cache layer."""
    old_loader = V6._load_or_build_fast_cache
    old_version = V6.VERSION
    try:
        V6._load_or_build_fast_cache = _load_or_build_mmap_cache
        V6.VERSION = VERSION
        return V6.run_trailing_passive_exit_dev(
            source_session,
            hard_bind=hard_bind,
            rebuild_cache=rebuild_cache,
            show=show,
        )
    finally:
        V6._load_or_build_fast_cache = old_loader
        V6.VERSION = old_version
