from __future__ import annotations

"""Independent streaming raw-event replay for frozen Candidate-C Q10 OOS.

This module is deliberately separate from the live shadow thread. It reads the
authoritative completed V5 raw JSONL files directly, merges BOOK and TRADE events
by receipt timestamp (book before trade on exact ties, matching the frozen heap
priority), calls the frozen mechanics, and writes all replay outputs to a new
results workspace so the original OOS session is never modified.

No exchange API calls. No real orders. No parameter tuning.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS

REPLAY_VERSION = "MM_CYCLE_Q10_INDEPENDENT_RAW_REPLAY_V1"
HARD_BOUND_OOS_SESSION = "20260817_064143"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q10_independent_raw_replay"
EPS = 1e-10


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _next_timed(it):
    for r in it:
        t = OOS._ts(r.get("receipt_time"))
        ticker = str(r.get("ticker") or "")
        if np.isfinite(t) and ticker:
            return float(t), r
    return None


def _load_metadata(shadow, path: Path):
    n = 0
    for r in _iter_jsonl(path):
        ticker = str(r.get("ticker") or "")
        if not ticker:
            continue
        series = str(r.get("series_ticker") or "")
        shadow.meta[ticker] = r
        shadow.series_by_ticker[ticker] = series
        shadow.close_by_ticker[ticker] = str(r.get("close_time") or "")
        n += 1
    return n


def _field_delta(a, b):
    try:
        x, y = float(a), float(b)
        if np.isfinite(x) and np.isfinite(y):
            return x - y
    except Exception:
        pass
    return None


def run_independent_raw_replay(
    source_session,
    *,
    hard_bind=True,
    show=True,
):
    """Replay completed raw V5 data. READ-ONLY source; NO EXCHANGE API; NO ORDERS."""
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_OOS_SESSION:
        raise RuntimeError(
            f"Replay is hard-bound to formal OOS session {HARD_BOUND_OOS_SESSION}; got {source.name}."
        )

    required = [
        source / "book_top3_events.jsonl",
        source / "trades_event_time.jsonl",
        source / "market_metadata.jsonl",
        source / "fee_preflight.json",
        source / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1" / "shadow_summary.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required formal OOS artifacts: " + " | ".join(missing))

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored OOS fee preflight was not PASS; refusing replay.")

    out = (OUTPUT_ROOT / source.name).resolve()
    if out.exists():
        raise FileExistsError(
            f"Independent replay output already exists: {out}. Preserve it; do not overwrite."
        )
    out.mkdir(parents=True, exist_ok=False)

    # FrozenCycleShadow writes only inside the supplied workspace. It does not call Kalshi.
    # Give it a copy of recorder health for summary context, but never alter source files.
    src_health = source / "health.json"
    if src_health.exists():
        try:
            (out / "health.json").write_text(src_health.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    shadow = OOS.FrozenCycleShadow(out, fee)
    meta_rows = _load_metadata(shadow, source / "market_metadata.jsonl")

    book_it = _iter_jsonl(source / "book_top3_events.jsonl")
    trade_it = _iter_jsonl(source / "trades_event_time.jsonl")
    b = _next_timed(book_it)
    tr = _next_timed(trade_it)

    if b is None and tr is None:
        raise RuntimeError("No timed raw events found.")

    first_ts = min(x[0] for x in (b, tr) if x is not None)
    last_ts = first_ts
    shadow.started_at = pd.Timestamp(first_ts, unit="s", tz="UTC")
    shadow.thread_alive = True
    shadow.emit("INDEPENDENT_REPLAY_START", detail=str(source))

    book_rows = 0
    trade_rows = 0
    events = 0

    while b is not None or tr is not None:
        choose_book = False
        if tr is None:
            choose_book = True
        elif b is None:
            choose_book = False
        elif b[0] < tr[0] - EPS:
            choose_book = True
        elif tr[0] < b[0] - EPS:
            choose_book = False
        else:
            # Exact/near-exact timestamp tie: frozen event heap gives book priority 0, trade 1.
            choose_book = True

        if choose_book:
            t, r = b
            shadow._on_book(t, r)
            book_rows += 1
            b = _next_timed(book_it)
        else:
            t, r = tr
            shadow._on_trade(t, r)
            trade_rows += 1
            tr = _next_timed(trade_it)

        last_ts = max(last_ts, float(t))
        events += 1
        shadow._update_drawdown()

        if show and events % 1_000_000 == 0:
            print(
                f"replayed {events:,} events | books={book_rows:,} trades={trade_rows:,} "
                f"fills={int(shadow.c['fill_events']):,}"
            )

    shadow.thread_alive = False
    shadow.emit("INDEPENDENT_REPLAY_STOP", events=events)
    shadow._save()

    replay = OOS._read_json(shadow.summary_path, {}) or {}
    original = OOS._read_json(
        source / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1" / "shadow_summary.json", {}
    ) or {}

    compare_fields = [
        "passive_matched_pnl",
        "forced_liq_gross_pnl",
        "taker_trade_fees",
        "realized_net_trade_fee_only",
        "fill_events",
        "fill_qty",
        "cycles_started",
        "cycles_completed",
        "forced_liquidations",
        "forced_liq_qty",
        "max_abs_inventory",
    ]
    comparison = {}
    for k in compare_fields:
        comparison[k] = {
            "independent_replay": replay.get(k),
            "original_shadow": original.get(k),
            "delta": _field_delta(replay.get(k), original.get(k)),
        }

    exact_numeric = True
    for k, v in comparison.items():
        d = v.get("delta")
        if d is None:
            exact_numeric = False
            continue
        tol = 1e-9 if k not in {"fill_events", "cycles_started", "cycles_completed", "forced_liquidations"} else 1e-12
        if abs(float(d)) > tol:
            exact_numeric = False

    source_duration_h = (last_ts - first_ts) / 3600.0
    out_summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "replay_version": REPLAY_VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "hard_bound": bool(hard_bind),
        "metadata_rows_loaded": meta_rows,
        "book_rows_replayed": book_rows,
        "trade_rows_replayed": trade_rows,
        "total_events_replayed": events,
        "first_receipt_ts": first_ts,
        "last_receipt_ts": last_ts,
        "source_event_span_hours": source_duration_h,
        "comparison": comparison,
        "exact_core_reconciliation": exact_numeric,
        "orders_sent": False,
        "exchange_api_called": False,
        "source_modified": False,
        "note": (
            "Independent replay reads authoritative raw book/trade JSONL directly and merges by receipt time. "
            "It writes only under results/kalshi_q10_independent_raw_replay."
        ),
    }
    OOS._atomic_json(out / "independent_replay_comparison.json", out_summary)

    if show:
        print("=" * 108)
        print("INDEPENDENT FROZEN Q10 RAW REPLAY — NO ORDERS / NO EXCHANGE API")
        print("=" * 108)
        print("Source:", source)
        print(f"Book rows:  {book_rows:,}")
        print(f"Trade rows: {trade_rows:,}")
        print(f"Events:     {events:,}")
        print(f"Span:       {source_duration_h:.4f} h")
        print()
        for k, v in comparison.items():
            print(
                f"{k:30s} replay={v['independent_replay']} | shadow={v['original_shadow']} | delta={v['delta']}"
            )
        print()
        print("EXACT CORE RECONCILIATION:", "PASS" if exact_numeric else "FAIL")
        print("ORDERS SENT:              NO")
        print("EXCHANGE API CALLED:      NO")
        print("SOURCE MODIFIED:           NO")

    return out_summary


__all__ = [
    "REPLAY_VERSION",
    "HARD_BOUND_OOS_SESSION",
    "run_independent_raw_replay",
]
