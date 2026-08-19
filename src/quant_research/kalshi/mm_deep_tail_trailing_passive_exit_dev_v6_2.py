from __future__ import annotations

"""V6.2 DEVELOPMENT replay: unchanged V6 strategy, much faster one-time raw cache build.

Why V6.1 could still be slow
----------------------------
V6.1 streamed native grep output line-by-line through one Python process.  Even after
removing pandas timestamp parsing, that serial JSON loop can become the bottleneck and
also back-pressure grep.

V6.2 separates the work into two stages:
1. native ripgrep/grep writes all matching rows directly to a temporary filtered file;
   Python is not in the pipe, so the native scan can run at disk speed;
2. the filtered file is split by BYTE RANGES and parsed in parallel by multiple worker
   processes.  Workers return only the compact columns needed by V6.

The resulting compact pickle/JSON cache is persistent.  Future development sweeps use a
cache hit and never touch the multi-million-row raw files again.

Strategy/execution semantics are unchanged from V6.  Development source only; no API,
no orders, no validation data.
"""

import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_trailing_passive_exit_dev_v6 as V6

VERSION = "MM_DEEP_TAIL_TRAILING_PASSIVE_EXIT_DEV_V6_2_PARALLEL_CACHE"
CACHE_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_fast_cache_v3"
EPS = V6.EPS
M1_S = V6.M1_S
M5_S = V6.M5_S

try:
    import orjson as _orjson  # type: ignore
except Exception:  # optional
    _orjson = None

# Worker globals, populated by initializer under macOS spawn as well as fork.
_W_WINDOW_START = {}
_W_MIN_ACTIVE = {}


def _loads(raw: bytes):
    if _orjson is not None:
        return _orjson.loads(raw)
    return json.loads(raw)


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


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _worker_init(window_start, min_active):
    global _W_WINDOW_START, _W_MIN_ACTIVE
    _W_WINDOW_START = dict(window_start)
    _W_MIN_ACTIVE = dict(min_active)


def _native_filter(src: Path, dst: Path, tickers: set[str], *, show=True):
    """Materialize relevant ticker lines with no Python process in the stdout pipe."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    pattern = dst.with_suffix(dst.suffix + ".patterns")
    pattern.write_text("\n".join(sorted(tickers)) + "\n", encoding="utf-8")

    rg = shutil.which("rg")
    grep = shutil.which("grep")
    if rg:
        cmd = [rg, "--text", "--no-line-number", "--no-filename", "-F", "-f", str(pattern), str(src)]
        engine = "ripgrep"
    elif grep:
        cmd = [grep, "-F", "-f", str(pattern), str(src)]
        engine = "grep"
    else:
        raise RuntimeError("V6.2 fast cache requires rg or grep; neither was found")

    if show:
        gb = src.stat().st_size / (1024 ** 3)
        print(f"PARALLEL CACHE: native {engine} filter | source={gb:.2f} GiB")

    t0 = time.perf_counter()
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        with tmp.open("wb", buffering=8 * 1024 * 1024) as out:
            p = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, env=env, check=False)
        if p.returncode not in (0, 1):
            err = (p.stderr or b"")[:1000].decode("utf-8", errors="ignore")
            raise RuntimeError(f"{engine} failed rc={p.returncode}: {err}")
        tmp.replace(dst)
    finally:
        try:
            pattern.unlink()
        except FileNotFoundError:
            pass
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass

    if show:
        mb = dst.stat().st_size / (1024 ** 2)
        print(f"  native filter DONE in {time.perf_counter()-t0:.1f}s | filtered={mb:.1f} MiB")


def _byte_ranges(path: Path, workers: int):
    size = int(path.stat().st_size)
    if size <= 0:
        return []
    workers = max(1, min(int(workers), size))
    step = int(math.ceil(size / workers))
    out = []
    start = 0
    while start < size:
        end = min(size, start + step)
        out.append((str(path), start, end, size))
        start = end
    return out


def _iter_range(path_s: str, start: int, end: int, size: int):
    """Yield complete JSONL rows in [start,end), with boundaries aligned to newlines."""
    with open(path_s, "rb", buffering=4 * 1024 * 1024) as fh:
        if start > 0:
            fh.seek(start - 1)
            prev = fh.read(1)
            if prev != b"\n":
                fh.readline()  # discard partial line
        else:
            fh.seek(0)
        while True:
            pos = fh.tell()
            if pos >= end and start > 0:
                break
            raw = fh.readline()
            if not raw:
                break
            yield raw
            if fh.tell() >= end:
                break


def _levels_fixed(raw_levels):
    vals = []
    levels = []
    for x in (raw_levels or [])[:3]:
        try:
            p = float(x[0]); q = max(0.0, float(x[1]))
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


def _parse_book_range(args):
    path_s, start, end, size = args
    rows = []
    audit_max = 0.0
    audit_n = 0
    matched = 0
    for raw in _iter_range(path_s, start, end, size):
        matched += 1
        try:
            r = _loads(raw)
        except Exception:
            continue
        ticker = str(r.get("ticker") or "")
        ws = _W_WINDOW_START.get(ticker)
        if ws is None:
            continue
        e = _f(r.get("elapsed_s"))
        if not (np.isfinite(e) and M1_S <= e < M5_S):
            continue
        if not bool(r.get("valid_bbo")):
            continue
        bid = _f(r.get("yes_bid")); ask = _f(r.get("yes_ask"))
        if not (np.isfinite(bid) and np.isfinite(ask) and 0 <= bid < ask <= 1):
            continue
        bp, bid_levels = _levels_fixed(r.get("bid_levels"))
        ap, ask_levels = _levels_fixed(r.get("ask_levels"))
        if not bid_levels or not ask_levels:
            continue
        mid = _f(r.get("mid"), 0.5 * (bid + ask))
        rt = float(ws) + float(e)

        # Tiny sampled audit only; no expensive timestamp parsing in the hot loop.
        if audit_n < 3:
            rs = _fast_iso_s(r.get("receipt_time"))
            if np.isfinite(rs):
                audit_max = max(audit_max, abs(float(rs) - rt)); audit_n += 1

        rows.append((
            ticker, rt, float(e), float(bid), float(ask), float(mid),
            bp[0], bp[1], bp[2], bp[3], bp[4], bp[5],
            ap[0], ap[1], ap[2], ap[3], ap[4], ap[5],
        ))
    return rows, matched, audit_n, audit_max


def _raw_exchange_s(r):
    z = _f(r.get("ts_ms"))
    if np.isfinite(z):
        return float(z) / 1000.0
    return _fast_iso_s(r.get("exchange_time"))


def _parse_trade_range(args):
    path_s, start, end, size = args
    rows = []
    stats = Counter()
    audit_max = 0.0
    audit_n = 0
    matched = 0
    for raw in _iter_range(path_s, start, end, size):
        matched += 1
        try:
            r = _loads(raw)
        except Exception:
            continue
        ticker = str(r.get("ticker") or "")
        ws = _W_WINDOW_START.get(ticker)
        if ws is None:
            continue
        e = _f(r.get("elapsed_s"))
        if not (np.isfinite(e) and M1_S <= e < M5_S):
            continue
        yes = _f(r.get("yes_price")); qty = _f(r.get("qty"))
        side = str(r.get("taker_book_side") or "").lower()
        if not (np.isfinite(yes) and 0 <= yes <= 1 and np.isfinite(qty) and qty > 0 and side in {"bid", "ask"}):
            continue
        rt = float(ws) + float(e)
        if audit_n < 3:
            rs = _fast_iso_s(r.get("receipt_time"))
            if np.isfinite(rs):
                audit_max = max(audit_max, abs(float(rs) - rt)); audit_n += 1
        x = _raw_exchange_s(r)
        if not np.isfinite(x):
            exec_s = rt; src = "RECEIPT_FALLBACK"
        elif x > rt + EPS:
            exec_s = rt; src = "CLAMP_EXCHANGE_AFTER_RECEIPT"
        else:
            exec_s = float(x); src = "EXCHANGE_TIME"
        obs_s = max(float(exec_s), rt)
        if obs_s + EPS < float(_W_MIN_ACTIVE.get(ticker, -np.inf)):
            continue
        rows.append((ticker, float(exec_s), rt, obs_s, float(yes), float(qty), side, str(r.get("trade_id") or "")))
        stats[src] += 1
    return rows, dict(stats), matched, audit_n, audit_max


def _workers_default():
    # Leave some cores for the notebook/OS. M4 Pro commonly has enough cores for 6-8 workers.
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu - 2 if cpu > 4 else cpu))


def _parallel_parse(path: Path, worker_fn, window_start, min_active, *, workers, show, label):
    ranges = _byte_ranges(path, workers)
    if not ranges:
        return []
    if show:
        print(f"PARALLEL CACHE: parsing {label} with {len(ranges)} worker processes...")
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(
        processes=len(ranges),
        initializer=_worker_init,
        initargs=(window_start, min_active),
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_fn, ranges), start=1):
            results.append(result)
            if show:
                print(f"  {label} worker {i}/{len(ranges)} finished")
    if show:
        print(f"  parallel {label} parse DONE in {time.perf_counter()-t0:.1f}s")
    return results


def _build_cache_parallel(source: Path, anchors: pd.DataFrame, *, rebuild=False, show=True):
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
                print("PARALLEL CACHE HIT:", cache_dir)
            return pd.read_pickle(bbo_path), pd.read_pickle(trade_path), V6._read_json(m5_path, {}) or {}, V6._read_json(stats_path, {}) or {}, "PARALLEL_CACHE_HIT"

    meta = V1._metadata(source)
    tickers = set(anchors["ticker"].astype(str))
    window_start = {t: float(meta[t]["window_start_s"]) for t in tickers if t in meta}
    min_active = anchors.groupby("ticker")["exit_active_s"].min().astype(float).to_dict()
    workers = _workers_default()

    if show:
        print(f"PARALLEL CACHE MISS — workers={workers}; native-filter then multi-core parse")

    book_filtered = cache_dir / "book_relevant.filtered.jsonl"
    trade_filtered = cache_dir / "trades_relevant.filtered.jsonl"

    # Materialization is restartable: if a prior run already completed native filtering,
    # keep the file and skip that expensive raw scan.
    if not book_filtered.exists() or book_filtered.stat().st_size == 0:
        _native_filter(source / "book_top3_events.jsonl", book_filtered, tickers, show=show)
    elif show:
        print(f"PARALLEL CACHE: reusing materialized book filter ({book_filtered.stat().st_size/(1024**2):.1f} MiB)")

    book_results = _parallel_parse(book_filtered, _parse_book_range, window_start, min_active, workers=workers, show=show, label="book")
    book_rows = []
    audit_n = 0; audit_max = 0.0; matched = 0
    for rows, m, an, am in book_results:
        book_rows.extend(rows); matched += m; audit_n += an; audit_max = max(audit_max, am)
    if audit_n and audit_max > 0.001 + EPS:
        raise RuntimeError(f"receipt clock audit failed for book: max error={audit_max:.6f}s")
    if not book_rows:
        raise RuntimeError("parallel book parse returned no rows")

    cols = [
        "ticker", "receipt_s", "elapsed_s", "yes_bid", "yes_ask", "yes_mid",
        "bid_p1", "bid_q1", "bid_p2", "bid_q2", "bid_p3", "bid_q3",
        "ask_p1", "ask_q1", "ask_p2", "ask_q2", "ask_p3", "ask_q3",
    ]
    all_bbo = pd.DataFrame.from_records(book_rows, columns=cols)
    all_bbo.sort_values(["ticker", "receipt_s"], kind="mergesort", inplace=True)

    # True M5 snapshot = last valid BBO strictly before M5 for each ticker.
    m5 = {}
    for ticker, g in all_bbo.groupby("ticker", sort=False):
        r = g.iloc[-1]
        bids = [(float(r[f"bid_p{i}"]), float(r[f"bid_q{i}"])) for i in (1,2,3) if np.isfinite(r[f"bid_p{i}"]) and np.isfinite(r[f"bid_q{i}"])]
        asks = [(float(r[f"ask_p{i}"]), float(r[f"ask_q{i}"])) for i in (1,2,3) if np.isfinite(r[f"ask_p{i}"]) and np.isfinite(r[f"ask_q{i}"])]
        m5[str(ticker)] = {
            "yes_bid": float(r["yes_bid"]), "yes_ask": float(r["yes_ask"]), "yes_mid": float(r["yes_mid"]),
            "bid_levels": bids, "ask_levels": asks,
            "receipt_s": float(r["receipt_s"]), "snapshot_elapsed_s": float(r["elapsed_s"]), "true_m5_finalized": True,
        }

    # Strategy path needs only states actionable after each ticker's earliest fill.
    mins = all_bbo["ticker"].map(min_active).astype(float)
    bbo = all_bbo[all_bbo["receipt_s"] + EPS >= mins].copy().reset_index(drop=True)
    bbo.to_pickle(bbo_path)
    V6._atomic_json(m5_path, m5)
    del all_bbo, book_rows, book_results

    if show:
        print(f"PARALLEL CACHE: compact book states={len(bbo):,} | M5 snapshots={len(m5):,}")

    if not trade_filtered.exists() or trade_filtered.stat().st_size == 0:
        _native_filter(source / "trades_event_time.jsonl", trade_filtered, tickers, show=show)
    elif show:
        print(f"PARALLEL CACHE: reusing materialized trade filter ({trade_filtered.stat().st_size/(1024**2):.1f} MiB)")

    trade_results = _parallel_parse(trade_filtered, _parse_trade_range, window_start, min_active, workers=workers, show=show, label="trade")
    trade_rows = []
    stats = Counter(); audit_n_t = 0; audit_max_t = 0.0; matched_t = 0
    for rows, st, m, an, am in trade_results:
        trade_rows.extend(rows); stats.update(st); matched_t += m; audit_n_t += an; audit_max_t = max(audit_max_t, am)
    if audit_n_t and audit_max_t > 0.001 + EPS:
        raise RuntimeError(f"receipt clock audit failed for trades: max error={audit_max_t:.6f}s")
    tcols = ["ticker", "exec_s", "receipt_s", "obs_s", "yes_price", "qty", "taker_book_side", "trade_id"]
    trades = pd.DataFrame.from_records(trade_rows, columns=tcols)
    if len(trades):
        trades.sort_values(["ticker", "exec_s", "receipt_s", "trade_id"], kind="mergesort", inplace=True)
        trades.reset_index(drop=True, inplace=True)
    trades.to_pickle(trade_path)
    V6._atomic_json(stats_path, dict(stats))

    manifest = {
        "cache_version": VERSION,
        "strategy_version": V6.VERSION,
        "source": str(source),
        "source_session": source.name,
        "workers": workers,
        "json_parser": "orjson" if _orjson is not None else "stdlib-json",
        "tickers": sorted(tickers),
        "anchors": int(len(anchors)),
        "book_filtered_bytes": int(book_filtered.stat().st_size),
        "trade_filtered_bytes": int(trade_filtered.stat().st_size),
        "book_matching_rows": int(matched),
        "trade_matching_rows": int(matched_t),
        "bbo_rows": int(len(bbo)),
        "trade_rows": int(len(trades)),
        "book_clock_audit_samples": int(audit_n),
        "book_clock_audit_max_error_s": float(audit_max),
        "trade_clock_audit_samples": int(audit_n_t),
        "trade_clock_audit_max_error_s": float(audit_max_t),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    V6._atomic_json(manifest_path, manifest)

    # Final cache is complete; large filtered intermediates are no longer needed.
    for p in (book_filtered, trade_filtered):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    if show:
        print(f"PARALLEL CACHE BUILT | BBO={len(bbo):,} | trades={len(trades):,}")
    return bbo, trades, m5, dict(stats), "PARALLEL_CACHE_BUILT"


def run_trailing_passive_exit_dev(source_session, *, hard_bind=True, rebuild_cache=False, show=True):
    """Run unchanged V6 strategy with V6.2 parallel cache layer."""
    old_loader = V6._load_or_build_fast_cache
    old_version = V6.VERSION
    try:
        V6._load_or_build_fast_cache = _build_cache_parallel
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
