from __future__ import annotations

"""V6.4 DEVELOPMENT replay: unchanged V6 strategy with hybrid fast cache.

V6.3 successfully reduced the 5.67 GiB BOOK file to the relevant 10% using mmap
receipt-time slicing, but the TRADE file contains material receipt-time disorder
(~111s coarse reversal), so time slicing that file is not scientifically safe.

V6.4 therefore uses the fastest safe split:

BOOKS
-----
Reuse the already-built V6.3 compact book cache when present.  If missing, rebuild
with V6.3 mmap time slicing.  This avoids rescanning the 5.67 GiB book file after the
successful V6.3 extraction.

TRADES
------
Do NOT time-slice by receipt time.  Native rg/grep scans the trade file once for the
~32 exact relevant ticker strings and materializes only matching rows.  The filtered
trade file is then parsed across multiple CPU processes with the same causal execution
clock used by V6.2/V6.3.  File disorder cannot drop a relevant trade because selection
is ticker-based over the entire trade file.

The resulting compact trade cache is persistent.  Strategy/execution semantics are
unchanged from V6.  DEVELOPMENT source only; no validation data, API calls, or orders.
"""

from collections import Counter
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_trailing_passive_exit_dev_v6 as V6
from . import mm_deep_tail_trailing_passive_exit_dev_v6_2 as V62
from . import mm_deep_tail_trailing_passive_exit_dev_v6_3 as V63

VERSION = "MM_DEEP_TAIL_TRAILING_PASSIVE_EXIT_DEV_V6_4_HYBRID_CACHE"
CACHE_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_fast_cache_v5"
EPS = V6.EPS


def _book_cache_paths(source: Path):
    d = V63.CACHE_ROOT / source.name
    return d, d / "relevant_post_fill_bbo.pkl", d / "m5_top3.json"


def _load_or_rebuild_book_cache(source: Path, anchors: pd.DataFrame, meta: dict, *, show=True):
    """Reuse V6.3's successfully materialized book cache whenever possible."""
    d, bbo_path, m5_path = _book_cache_paths(source)
    required_tickers = set(anchors["ticker"].astype(str))

    if bbo_path.exists() and m5_path.exists():
        try:
            bbo = pd.read_pickle(bbo_path)
            m5 = V6._read_json(m5_path, {}) or {}
            have_bbo = set(bbo["ticker"].astype(str)) if len(bbo) else set()
            have_m5 = set(str(x) for x in m5)
            if required_tickers.issubset(have_bbo) and required_tickers.issubset(have_m5):
                if show:
                    print(
                        f"HYBRID CACHE: REUSING V6.3 BOOK CACHE | rows={len(bbo):,} | "
                        f"tickers={len(required_tickers)}"
                    )
                    print("  No 5.67 GiB book rescan required.")
                return bbo, m5, "REUSED_V6_3_BOOK_CACHE"
        except Exception as exc:
            if show:
                print("V6.3 book cache exists but could not be reused:", repr(exc))

    # Safe fallback: rebuild only the BOOK cache with V6.3's time-slice logic.
    intervals, min_active = V63._relevant_time_intervals(meta, anchors)
    if show:
        print("HYBRID CACHE: V6.3 book cache missing; rebuilding book slice once...")
    d.mkdir(parents=True, exist_ok=True)
    bbo, m5, _, _ = V63._build_book_cache_sliced(
        source, anchors, meta, intervals, min_active, d, show=show
    )
    return bbo, m5, "REBUILT_V6_3_BOOK_CACHE"


def _trade_cache_paths(source: Path):
    d = CACHE_ROOT / source.name
    return (
        d,
        d / "relevant_post_fill_trades.pkl",
        d / "trade_clock_stats.json",
        d / "trades_relevant.filtered.jsonl",
        d / "trade_manifest.json",
    )


def _build_trade_cache_disorder_safe(
    source: Path,
    anchors: pd.DataFrame,
    meta: dict,
    *,
    rebuild=False,
    show=True,
):
    """Full-file ticker filtering + parallel parse. Safe under arbitrary file disorder."""
    cache_dir, trade_path, stats_path, filtered_path, manifest_path = _trade_cache_paths(source)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not rebuild and trade_path.exists() and stats_path.exists() and manifest_path.exists():
        manifest = V6._read_json(manifest_path, {}) or {}
        if manifest.get("cache_version") == VERSION:
            if show:
                print("HYBRID CACHE: TRADE CACHE HIT |", trade_path)
            return (
                pd.read_pickle(trade_path),
                V6._read_json(stats_path, {}) or {},
                "TRADE_CACHE_HIT",
            )

    tickers = set(anchors["ticker"].astype(str))
    window_start = {t: float(meta[t]["window_start_s"]) for t in tickers if t in meta}
    min_active = anchors.groupby("ticker")["exit_active_s"].min().astype(float).to_dict()
    raw_trade = source / "trades_event_time.jsonl"

    if show:
        raw_gib = raw_trade.stat().st_size / (1024 ** 3)
        print(
            f"HYBRID CACHE: disorder-safe trade extraction | raw={raw_gib:.2f} GiB | "
            f"tickers={len(tickers)}"
        )
        print("  Trade file will be scanned ONCE by native rg/grep; no Python in that scan.")

    # Native filter is restartable.  If it finished in an interrupted attempt, reuse it.
    if rebuild:
        for p in (trade_path, stats_path, filtered_path, manifest_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    if not filtered_path.exists() or filtered_path.stat().st_size == 0:
        V62._native_filter(raw_trade, filtered_path, tickers, show=show)
    elif show:
        print(
            f"HYBRID CACHE: reusing materialized trade filter | "
            f"size={filtered_path.stat().st_size/(1024**2):.1f} MiB"
        )

    workers = V62._workers_default()
    results = V62._parallel_parse(
        filtered_path,
        V62._parse_trade_range,
        window_start,
        min_active,
        workers=workers,
        show=show,
        label="trade",
    )

    rows = []
    stats = Counter()
    matched = 0
    audit_n = 0
    audit_max = 0.0
    for worker_rows, worker_stats, worker_matched, worker_audit_n, worker_audit_max in results:
        rows.extend(worker_rows)
        stats.update(worker_stats)
        matched += int(worker_matched)
        audit_n += int(worker_audit_n)
        audit_max = max(audit_max, float(worker_audit_max))

    if audit_n and audit_max > 0.001 + EPS:
        raise RuntimeError(
            f"Trade receipt clock audit failed: max error={audit_max:.6f}s"
        )

    cols = [
        "ticker", "exec_s", "receipt_s", "obs_s", "yes_price", "qty",
        "taker_book_side", "trade_id",
    ]
    trades = pd.DataFrame.from_records(rows, columns=cols)
    if len(trades):
        trades.sort_values(
            ["ticker", "exec_s", "receipt_s", "trade_id"],
            kind="mergesort",
            inplace=True,
        )
        trades.reset_index(drop=True, inplace=True)

    trades.to_pickle(trade_path)
    V6._atomic_json(stats_path, dict(stats))
    V6._atomic_json(
        manifest_path,
        {
            "cache_version": VERSION,
            "source": str(source),
            "source_session": source.name,
            "raw_trade_size_bytes": int(raw_trade.stat().st_size),
            "filtered_trade_size_bytes": int(filtered_path.stat().st_size),
            "tickers": sorted(tickers),
            "workers": int(workers),
            "matched_filtered_rows": int(matched),
            "cached_trade_rows": int(len(trades)),
            "receipt_clock_audit_samples": int(audit_n),
            "receipt_clock_audit_max_abs_error_s": float(audit_max),
            "selection_method": "full raw trade file native ticker filter; safe under arbitrary receipt-time disorder",
            "native_filter": "ripgrep" if shutil.which("rg") else "grep",
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )

    if show:
        print(
            f"HYBRID CACHE: trades DONE | filtered rows={matched:,} | "
            f"cached={len(trades):,} | audit max={audit_max:.9f}s"
        )
    return trades, dict(stats), "TRADE_CACHE_BUILT"


def _load_or_build_hybrid_cache(source: Path, anchors: pd.DataFrame, *, rebuild=False, show=True):
    source = Path(source).resolve()
    meta = V1._metadata(source)

    bbo, m5, book_mode = _load_or_rebuild_book_cache(
        source, anchors, meta, show=show
    )
    trades, stats, trade_mode = _build_trade_cache_disorder_safe(
        source, anchors, meta, rebuild=rebuild, show=show
    )

    mode = f"HYBRID[{book_mode}|{trade_mode}]"
    return bbo, trades, m5, stats, mode


def run_trailing_passive_exit_dev(source_session, *, hard_bind=True, rebuild_cache=False, show=True):
    """Run unchanged V6 strategy with V6.4 hybrid cache."""
    old_loader = V6._load_or_build_fast_cache
    old_version = V6.VERSION
    try:
        V6._load_or_build_fast_cache = _load_or_build_hybrid_cache
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
