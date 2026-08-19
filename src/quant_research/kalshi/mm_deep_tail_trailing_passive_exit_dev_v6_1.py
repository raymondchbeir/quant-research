from __future__ import annotations

"""Ultrafast cache front-end for the V6 deep-tail trailing passive-exit development replay.

This module changes NO strategy or execution rule from V6.  It only replaces the one-time
raw-cache builder, which was slow because it called pandas timestamp parsing and the generic
book-state normalizer hundreds of thousands of times.

Speed changes
-------------
- Native ripgrep/grep prefilters to the ~32 relevant filled tickers.
- The native subprocess is consumed as BYTES; no per-line UTF-8 decode/encode round trip.
- ``orjson`` is used when installed, otherwise stdlib ``json`` is used.
- Book rows already contain the persisted top-3 BBO, so they are decoded directly instead
  of re-running the generic ``_top_state`` adapter.
- ``receipt_s`` is reconstructed as ``window_start_s + elapsed_s``.  This is exactly how
  the V5 recorder created ``elapsed_s`` from the same receipt timestamp.  A sample audit
  compares this reconstruction with the persisted ``receipt_time`` and aborts if they
  disagree by more than 1 ms.
- Trade receipt time uses the same reconstruction; exchange time is parsed with the fast
  stdlib datetime parser (or ``ts_ms`` when available), then the same V11 causal clamp is
  applied.
- A new cache namespace is used so an interrupted/older V6 cache cannot be mistaken for
  the optimized cache.

Scientific status
-----------------
DEVELOPMENT ONLY.  The strategy logic remains V6.  The 15h validation sample is not read.
No API calls.  No orders.  Source data are read-only.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_trailing_passive_exit_dev_v6 as V6

VERSION = "MM_DEEP_TAIL_TRAILING_PASSIVE_EXIT_DEV_V6_1_ULTRAFAST_CACHE"
CACHE_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_fast_cache_v2"
EPS = V6.EPS
M1_S = V6.M1_S
M5_S = V6.M5_S

try:
    import orjson as _orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _orjson = None


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


def _raw_exchange_s(row):
    z = V6._f((row or {}).get("ts_ms"))
    if np.isfinite(z):
        return float(z) / 1000.0
    return _fast_iso_s((row or {}).get("exchange_time"))


def _causal_exec_fast(row, receipt_s):
    x = _raw_exchange_s(row)
    if not np.isfinite(x):
        return float(receipt_s), "RECEIPT_FALLBACK"
    if x > float(receipt_s) + EPS:
        return float(receipt_s), "CLAMP_EXCHANGE_AFTER_RECEIPT"
    return float(x), "EXCHANGE_TIME"


def _binary_prefilter(path: Path, tickers: set[str]):
    """Yield matching JSONL rows as bytes using the fastest native matcher available."""
    tickers = {str(t) for t in tickers if str(t)}
    if not tickers:
        return

    with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
        pattern_path = Path(fh.name)
        for ticker in sorted(tickers):
            fh.write(ticker.encode("utf-8") + b"\n")

    env = dict(os.environ)
    env["LC_ALL"] = "C"

    # ripgrep is normally materially faster on large files; fall back to grep.
    rg = shutil.which("rg")
    grep = shutil.which("grep")
    if rg:
        cmd = [rg, "--text", "--no-line-number", "--no-filename", "-F", "-f", str(pattern_path), str(path)]
        engine = "ripgrep"
    elif grep:
        cmd = [grep, "-F", "-f", str(pattern_path), str(path)]
        engine = "grep"
    else:
        cmd = None
        engine = "python-bytes-fallback"

    try:
        if cmd is not None:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=1024 * 1024,
                env=env,
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                if raw.strip():
                    yield raw
            stderr = proc.stderr.read() if proc.stderr is not None else b""
            rc = proc.wait()
            if rc not in (0, 1):
                raise RuntimeError(
                    f"{engine} failed rc={rc}: {stderr[:500].decode('utf-8', errors='ignore')}"
                )
            return

        needles = [t.encode("utf-8") for t in sorted(tickers)]
        with Path(path).open("rb", buffering=1024 * 1024) as src:
            for raw in src:
                if any(n in raw for n in needles):
                    yield raw
    finally:
        try:
            pattern_path.unlink()
        except Exception:
            pass


def _levels3(raw_levels):
    out = []
    for x in (raw_levels or [])[:3]:
        try:
            p = float(x[0])
            q = max(0.0, float(x[1]))
        except Exception:
            continue
        if np.isfinite(p) and np.isfinite(q):
            out.append((p, q))
    return out


def _fixed_levels(levels):
    out = []
    for i in range(3):
        if i < len(levels):
            out.extend([float(levels[i][0]), float(levels[i][1])])
        else:
            out.extend([np.nan, np.nan])
    return out


def _audit_receipt_clock(row, derived_s, audit_state, *, max_samples=200):
    if audit_state["samples"] >= max_samples:
        return
    raw_s = _fast_iso_s(row.get("receipt_time"))
    if not np.isfinite(raw_s):
        return
    err = abs(float(raw_s) - float(derived_s))
    audit_state["samples"] += 1
    audit_state["max_abs_error_s"] = max(audit_state["max_abs_error_s"], err)
    if err > 0.001 + EPS:
        raise RuntimeError(
            "Fast receipt-clock reconstruction disagrees with persisted receipt_time: "
            f"error={err:.6f}s. Refusing cache build."
        )


def _build_book_cache_fast(source: Path, anchors: pd.DataFrame, cache_dir: Path, *, show=True):
    meta = V1._metadata(source)
    tickers = set(anchors["ticker"].astype(str))
    min_active = anchors.groupby("ticker")["exit_active_s"].min().astype(float).to_dict()
    window_start = {t: float(meta[t]["window_start_s"]) for t in tickers if t in meta}

    rows = []
    last = {}
    m5 = {}
    finalized = set()
    matched = 0
    valid = 0
    audit = {"samples": 0, "max_abs_error_s": 0.0}

    if show:
        engine = "ripgrep" if shutil.which("rg") else ("grep" if shutil.which("grep") else "python")
        parser = "orjson" if _orjson is not None else "stdlib-json"
        print(
            f"ULTRAFAST CACHE: books via {engine} + {parser}; "
            f"no pandas timestamp parsing; {len(tickers)} tickers"
        )

    for raw in _binary_prefilter(source / "book_top3_events.jsonl", tickers):
        matched += 1
        try:
            r = _loads(raw)
        except Exception:
            continue

        ticker = str(r.get("ticker") or "")
        if ticker not in tickers or ticker in finalized or ticker not in window_start:
            continue

        e = V6._f(r.get("elapsed_s"))
        if not np.isfinite(e):
            continue

        # Same clock as persisted receipt_time by recorder construction.
        rt = float(window_start[ticker]) + float(e)
        _audit_receipt_clock(r, rt, audit)

        if e >= M5_S:
            prev = last.get(ticker)
            if prev is not None:
                m5[ticker] = prev
            finalized.add(ticker)
            last.pop(ticker, None)
            continue
        if e < M1_S:
            continue
        if not bool(r.get("valid_bbo")):
            continue

        try:
            bid = float(r.get("yes_bid"))
            ask = float(r.get("yes_ask"))
            bid_levels = _levels3(r.get("bid_levels"))
            ask_levels = _levels3(r.get("ask_levels"))
        except Exception:
            continue
        if not (np.isfinite(bid) and np.isfinite(ask) and 0.0 <= bid < ask <= 1.0):
            continue
        if not bid_levels or not ask_levels:
            continue

        mid_raw = V6._f(r.get("mid"))
        mid = float(mid_raw) if np.isfinite(mid_raw) else 0.5 * (bid + ask)
        valid += 1

        m5state = {
            "yes_bid": bid,
            "yes_ask": ask,
            "yes_mid": mid,
            "bid_levels": [[float(p), float(q)] for p, q in bid_levels],
            "ask_levels": [[float(p), float(q)] for p, q in ask_levels],
            "receipt_s": rt,
            "snapshot_elapsed_s": float(e),
            "true_m5_finalized": True,
        }
        last[ticker] = m5state

        if rt + EPS < float(min_active.get(ticker, -np.inf)):
            continue

        bp = _fixed_levels(bid_levels)
        ap = _fixed_levels(ask_levels)
        rows.append({
            "ticker": ticker,
            "receipt_s": rt,
            "elapsed_s": float(e),
            "yes_bid": bid,
            "yes_ask": ask,
            "yes_mid": mid,
            "bid_p1": bp[0], "bid_q1": bp[1],
            "bid_p2": bp[2], "bid_q2": bp[3],
            "bid_p3": bp[4], "bid_q3": bp[5],
            "ask_p1": ap[0], "ask_q1": ap[1],
            "ask_p2": ap[2], "ask_q2": ap[3],
            "ask_p3": ap[4], "ask_q3": ap[5],
        })

        if show and matched % 50_000 == 0:
            print(
                f"  book matches={matched:,} | valid={valid:,} | cached={len(rows):,} | "
                f"M5={len(m5):,}"
            )

    if not rows:
        raise RuntimeError("Ultrafast book cache found no relevant BBO states")

    df = pd.DataFrame.from_records(rows)
    df.sort_values(["ticker", "receipt_s"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(cache_dir / "relevant_post_fill_bbo.pkl")
    V6._atomic_json(cache_dir / "m5_top3.json", m5)
    V6._atomic_json(cache_dir / "clock_audit.json", audit)

    if show:
        print(
            f"ULTRAFAST CACHE: book done | matches={matched:,} | cached={len(df):,} | "
            f"M5={len(m5):,} | receipt audit max error={audit['max_abs_error_s']:.9f}s"
        )
    return df, m5


def _build_trade_cache_fast(source: Path, anchors: pd.DataFrame, cache_dir: Path, *, show=True):
    meta = V1._metadata(source)
    tickers = set(anchors["ticker"].astype(str))
    min_active = anchors.groupby("ticker")["exit_active_s"].min().astype(float).to_dict()
    window_start = {t: float(meta[t]["window_start_s"]) for t in tickers if t in meta}

    rows = []
    stats = defaultdict(int)
    matched = 0
    audit = {"samples": 0, "max_abs_error_s": 0.0}

    if show:
        print("ULTRAFAST CACHE: trade tape extraction...")

    for raw in _binary_prefilter(source / "trades_event_time.jsonl", tickers):
        matched += 1
        try:
            r = _loads(raw)
        except Exception:
            continue

        ticker = str(r.get("ticker") or "")
        if ticker not in tickers or ticker not in window_start:
            continue
        e = V6._f(r.get("elapsed_s"))
        if not (np.isfinite(e) and M1_S <= e < M5_S):
            continue

        rt = float(window_start[ticker]) + float(e)
        _audit_receipt_clock(r, rt, audit)

        yes = V6._f(r.get("yes_price"))
        qty = V6._f(r.get("qty"))
        side = str(r.get("taker_book_side") or "").lower()
        if not (
            np.isfinite(yes) and 0.0 <= yes <= 1.0
            and np.isfinite(qty) and qty > 0.0
            and side in {"bid", "ask"}
        ):
            continue

        exec_s, source_name = _causal_exec_fast(r, rt)
        obs_s = float(max(exec_s, rt))
        if obs_s + EPS < float(min_active.get(ticker, -np.inf)):
            continue

        rows.append({
            "ticker": ticker,
            "exec_s": float(exec_s),
            "receipt_s": rt,
            "obs_s": obs_s,
            "yes_price": float(yes),
            "qty": float(qty),
            "taker_book_side": side,
            "trade_id": str(r.get("trade_id") or ""),
        })
        stats[str(source_name)] += 1

        if show and matched % 50_000 == 0:
            print(f"  trade matches={matched:,} | cached={len(rows):,}")

    df = pd.DataFrame.from_records(rows)
    if len(df):
        df.sort_values(["ticker", "exec_s", "receipt_s", "trade_id"], kind="mergesort", inplace=True)
        df.reset_index(drop=True, inplace=True)
    df.to_pickle(cache_dir / "relevant_post_fill_trades.pkl")
    V6._atomic_json(cache_dir / "trade_clock_stats.json", dict(stats))
    V6._atomic_json(cache_dir / "trade_clock_audit.json", audit)

    if show:
        print(
            f"ULTRAFAST CACHE: trades done | matches={matched:,} | cached={len(df):,} | "
            f"receipt audit max error={audit['max_abs_error_s']:.9f}s"
        )
    return df, dict(stats)


def _load_or_build_ultrafast_cache(source: Path, anchors: pd.DataFrame, *, rebuild=False, show=True):
    cache_dir = CACHE_ROOT / source.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    bbo_path = cache_dir / "relevant_post_fill_bbo.pkl"
    trades_path = cache_dir / "relevant_post_fill_trades.pkl"
    m5_path = cache_dir / "m5_top3.json"
    stats_path = cache_dir / "trade_clock_stats.json"
    manifest_path = cache_dir / "manifest.json"

    if not rebuild and bbo_path.exists() and trades_path.exists() and m5_path.exists() and manifest_path.exists():
        manifest = V6._read_json(manifest_path, {}) or {}
        if manifest.get("cache_version") == VERSION:
            if show:
                print("ULTRAFAST CACHE HIT:", cache_dir)
            return (
                pd.read_pickle(bbo_path),
                pd.read_pickle(trades_path),
                V6._read_json(m5_path, {}) or {},
                V6._read_json(stats_path, {}) or {},
                "ULTRAFAST_CACHE_HIT",
            )

    if show:
        print("ULTRAFAST CACHE MISS — optimized one-time extraction starting...")

    # Write into temp names first.  If interrupted, a future run never treats partial data as valid.
    for p in (bbo_path, trades_path, m5_path, stats_path, manifest_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    bbo, m5 = _build_book_cache_fast(source, anchors, cache_dir, show=show)
    trades, stats = _build_trade_cache_fast(source, anchors, cache_dir, show=show)

    manifest = {
        "cache_version": VERSION,
        "strategy_version": V6.VERSION,
        "source": str(source),
        "source_session": source.name,
        "tickers": sorted(set(anchors["ticker"].astype(str))),
        "anchors": int(len(anchors)),
        "bbo_rows": int(len(bbo)),
        "trade_rows": int(len(trades)),
        "receipt_clock": "window_start_s + persisted elapsed_s; audited vs receipt_time <=1ms",
        "json_parser": "orjson" if _orjson is not None else "stdlib-json",
        "native_filter": "ripgrep" if shutil.which("rg") else ("grep" if shutil.which("grep") else "python"),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    V6._atomic_json(manifest_path, manifest)
    return bbo, trades, m5, stats, "ULTRAFAST_CACHE_BUILT"


def run_trailing_passive_exit_dev(source_session, *, hard_bind=True, rebuild_cache=False, show=True):
    """Run the unchanged V6 strategy with the optimized cache layer."""
    old_loader = V6._load_or_build_fast_cache
    old_version = V6.VERSION
    try:
        V6._load_or_build_fast_cache = _load_or_build_ultrafast_cache
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
