from __future__ import annotations

"""V7.2 DEVELOPMENT capacity replay: fix the pre-Q5 BBO cache gap for Q>5.

Why V7/V7.1 need this correction
---------------------------------
V7 reused the V6.3 compact BBO cache.  That cache intentionally stored rows only from
Q5 ``exit_active_s`` onward because it was built for the trailing-exit study.  Larger
entry quantities can fully fill during the ~100ms between the Q5 fill observation and
Q5 exit activation.  In that case V7 asks for the latest BBO at the larger-Q full-fill
observation, but the compact cache has not started yet, producing an artificial
``decision_snapshot_missing_full_entry``.

The pattern is visible in V7.1 (many missing snapshots at Q10/Q20, fewer at larger Q as
full fills occur later).  Those rows must not be used to choose capacity.

V7.2 changes NO strategy rule.  It augments the existing V6.3 post-Q5 BBO cache with a
small M0->Q5-activation prelude for the same ~32 relevant tickers, using V6.3's already
materialized coarse byte/time index.  It therefore reads only the relevant time slices,
not the full 5.67 GiB book file.  The prelude is cached permanently.

Strategy/execution and the V7.1 terminal-capacity gate are unchanged.  Development only;
15h sample is not read.  No API calls.  No orders.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_trailing_passive_exit_dev_v6 as V6
from . import mm_deep_tail_trailing_passive_exit_dev_v6_3 as V63
from . import mm_deep_tail_join_ask_capacity_dev_v7 as V7
from . import mm_deep_tail_join_ask_capacity_dev_v7_1 as V71

VERSION = "MM_DEEP_TAIL_JOIN_ASK_CAPACITY_DEV_V7_2_PRE_Q5_BBO_FIX"
CACHE_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_join_ask_capacity_cache_v7_2"
EPS = V7.EPS
M5_S = V7.M5_S

BBO_COLS = [
    "ticker", "receipt_s", "elapsed_s", "yes_bid", "yes_ask", "yes_mid",
    "bid_p1", "bid_q1", "bid_p2", "bid_q2", "bid_p3", "bid_q3",
    "ask_p1", "ask_q1", "ask_p2", "ask_q2", "ask_p3", "ask_q3",
]


def _load_q5_anchors(source: Path):
    p = V6._latest_result_file(
        C.PROJECT_ROOT / "results" / "kalshi_deep_tail_reversion_exit_dev_v5",
        source.name,
        "entry_fill_anchors.csv",
    )
    if p is None:
        raise FileNotFoundError("Need V5 entry_fill_anchors.csv to reconstruct the pre-Q5 BBO prelude")
    a = pd.read_csv(p)
    needed = {"ticker", "exit_active_s"}
    if not needed.issubset(a.columns):
        raise RuntimeError(f"V5 anchor cache missing columns: {sorted(needed - set(a.columns))}")
    return a.copy(), str(p)


def _coarse_book_index(source: Path):
    prior = V63.CACHE_ROOT / source.name / "book_slice_index.json"
    obj = V6._read_json(prior, {}) or {}
    idx = obj.get("index") if isinstance(obj, dict) else None
    if isinstance(idx, dict) and idx.get("offsets") and idx.get("times") and idx.get("size"):
        return idx, str(prior), "REUSED_V6_3_COARSE_INDEX"
    idx = V63._build_coarse_index(source / "book_top3_events.jsonl", show=True)
    return idx, "REBUILT_COARSE_INDEX", "REBUILT_COARSE_INDEX"


def _build_prelude(source: Path, meta: dict, anchors: pd.DataFrame, *, show=True):
    cache_dir = CACHE_ROOT / source.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl = cache_dir / "pre_q5_bbo.pkl"
    manifest_path = cache_dir / "manifest.json"

    min_active = anchors.groupby("ticker")["exit_active_s"].min().astype(float).to_dict()
    tickers = sorted(set(str(x) for x in anchors["ticker"]))

    manifest = V6._read_json(manifest_path, {}) or {}
    if pkl.exists() and manifest.get("cache_version") == VERSION:
        q = pd.read_pickle(pkl)
        if set(tickers).issubset(set(q["ticker"].astype(str))):
            if show:
                print(
                    f"V7.2 PRELUDE CACHE HIT | rows={len(q):,} | "
                    f"tickers={q['ticker'].nunique():,}"
                )
            return q, str(pkl), manifest

    # We need the latest locally-known BBO for any possible larger-Q full-fill decision.
    # Since all Q>5 fills occur no earlier than the corresponding Q5 fill, M0->Q5
    # activation is a conservative/exact prelude and avoids assuming a recent quote update.
    intervals = []
    for ticker in tickers:
        m = meta.get(ticker)
        if not m or ticker not in min_active:
            continue
        start = float(m["window_start_s"]) - 0.25
        end = float(min_active[ticker]) + 0.01
        if end > start:
            intervals.append((start, end))
    intervals = V63._merge_time_intervals(intervals)

    idx, idx_source, idx_mode = _coarse_book_index(source)
    ranges = V63._intervals_to_byte_ranges(idx, intervals)
    selected_bytes = int(sum(b - a for a, b in ranges))

    if show:
        print(
            f"V7.2 PRE-Q5 BBO FIX | intervals={len(intervals)} | ranges={len(ranges)} | "
            f"read={selected_bytes/(1024**2):.1f} MiB "
            f"({100.0*selected_bytes/int(idx['size']):.2f}% of raw book)"
        )
        print("  This is a one-time prelude extraction; the full 5.67 GiB book is NOT scanned.")

    ticker_set = set(tickers)
    rows = []
    parsed = 0
    relevant = 0
    raw_path = source / "book_top3_events.jsonl"

    for raw in V63._iter_byte_ranges(raw_path, ranges):
        parsed += 1
        try:
            r = V63._loads(raw)
        except Exception:
            continue
        ticker = str(r.get("ticker") or "")
        if ticker not in ticker_set or ticker not in meta or ticker not in min_active:
            continue
        relevant += 1
        e = V63._f(r.get("elapsed_s"))
        if not (np.isfinite(e) and 0.0 <= e < M5_S):
            continue
        rt = float(meta[ticker]["window_start_s"]) + float(e)
        if rt >= float(min_active[ticker]) - EPS:
            continue
        if not bool(r.get("valid_bbo")):
            continue
        bid = V63._f(r.get("yes_bid"))
        ask = V63._f(r.get("yes_ask"))
        if not (np.isfinite(bid) and np.isfinite(ask) and 0.0 <= bid < ask <= 1.0):
            continue
        bp, bid_levels = V63._levels_fixed(r.get("bid_levels"))
        ap, ask_levels = V63._levels_fixed(r.get("ask_levels"))
        if not bid_levels or not ask_levels:
            continue
        mid = V63._f(r.get("mid"), 0.5 * (bid + ask))
        rows.append((
            ticker, float(rt), float(e), float(bid), float(ask), float(mid),
            bp[0], bp[1], bp[2], bp[3], bp[4], bp[5],
            ap[0], ap[1], ap[2], ap[3], ap[4], ap[5],
        ))

    if not rows:
        raise RuntimeError("V7.2 pre-Q5 BBO extraction returned no valid states")

    q = pd.DataFrame.from_records(rows, columns=BBO_COLS)
    q.sort_values(["ticker", "receipt_s"], kind="mergesort", inplace=True)
    q.drop_duplicates(["ticker", "receipt_s"], keep="last", inplace=True)
    q.reset_index(drop=True, inplace=True)

    missing = sorted(ticker_set - set(q["ticker"].astype(str)))
    if missing:
        raise RuntimeError(f"V7.2 prelude missing relevant tickers: {missing}")

    q.to_pickle(pkl)
    manifest = {
        "cache_version": VERSION,
        "source": str(source),
        "source_session": source.name,
        "anchor_rows": int(len(anchors)),
        "tickers": tickers,
        "coarse_index_mode": idx_mode,
        "coarse_index_source": idx_source,
        "selected_bytes": selected_bytes,
        "raw_book_size_bytes": int(idx["size"]),
        "parsed_lines": int(parsed),
        "relevant_lines": int(relevant),
        "prelude_rows": int(len(q)),
        "prelude_policy": "M0 through earliest Q5 exit_active_s per ticker; exact latest-known BBO support for Q>5 decisions",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    OOS._atomic_json(manifest_path, manifest)

    if show:
        print(
            f"V7.2 PRELUDE DONE | parsed={parsed:,} | relevant={relevant:,} | "
            f"cached={len(q):,}"
        )
    return q, str(pkl), manifest


def _augment_bbo(base: pd.DataFrame, prelude: pd.DataFrame):
    q = pd.concat([prelude[BBO_COLS], base[BBO_COLS]], ignore_index=True)
    q.sort_values(["ticker", "receipt_s"], kind="mergesort", inplace=True)
    q.drop_duplicates(["ticker", "receipt_s"], keep="last", inplace=True)
    q.reset_index(drop=True, inplace=True)
    return q


def run_join_ask_capacity_dev(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != V7.HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected development source {V7.HARD_BOUND_SESSION}, got {source.name}")

    meta = V7.V3.V1._metadata(source)
    anchors, anchor_source = _load_q5_anchors(source)

    old_loader = V7._load_book_cache
    base_bbo, m5, base_bbo_path, m5_path = old_loader(source)
    prelude, prelude_path, prelude_manifest = _build_prelude(source, meta, anchors, show=show)
    augmented = _augment_bbo(base_bbo, prelude)

    if show:
        print(
            f"V7.2 AUGMENTED BBO | prelude={len(prelude):,} + post-Q5={len(base_bbo):,} "
            f"-> combined={len(augmented):,}"
        )
        print("Running identical V7 economics + V7.1 terminal gate on corrected BBO history...")

    def patched_loader(_source):
        return augmented, m5, f"{base_bbo_path} + {prelude_path}", m5_path

    try:
        V7._load_book_cache = patched_loader
        res = V71.run_join_ask_capacity_dev(source, hard_bind=hard_bind, show=False)
    finally:
        V7._load_book_cache = old_loader

    surface = res["capacity_curve"].copy()
    detail = res["detail"].copy()
    by_asset = res["by_asset"].copy()
    out = Path(res["output_dir"])

    feasible = surface[
        surface["capacity_gate_positive_99pct_terminal_no_missing_decision"]
    ].copy()
    largest = (
        feasible.sort_values("requested_qty", ascending=False).iloc[0].to_dict()
        if len(feasible) else {}
    )
    max_pnl = (
        surface.sort_values(["join_ask_net_pnl_rounding_bound", "requested_qty"], ascending=False)
        .iloc[0].to_dict()
        if len(surface) else {}
    )

    surface.to_csv(out / "join_ask_capacity_curve_v7_2_corrected.csv", index=False)
    detail.to_csv(out / "join_ask_capacity_order_detail_v7_2_corrected.csv", index=False)
    by_asset.to_csv(out / "join_ask_capacity_by_asset_v7_2_corrected.csv", index=False)

    summary = dict(res["summary"])
    summary.update({
        "version": VERSION,
        "correction": (
            "Augment V6.3 BBO cache with M0->Q5-activation prelude so Q>5 full-fill decisions "
            "cannot be falsely missing merely because the trailing-study cache had not started yet."
        ),
        "q5_anchor_source": anchor_source,
        "prelude_cache": prelude_path,
        "prelude_manifest": prelude_manifest,
        "augmented_bbo_rows": int(len(augmented)),
        "largest_positive_qty_with_99pct_terminal_coverage_and_no_missing_decision": largest,
        "highest_development_pnl_row": max_pnl,
        "guardrail": (
            "Use V7.2, not V7/V7.1, for Q>5 capacity conclusions. Development only; "
            "the already-opened 15h sample is secondary robustness, not independent validation."
        ),
    })
    OOS._atomic_json(out / "summary_v7_2_corrected.json", summary)

    if show:
        print("=" * 174)
        print("DEEP-TAIL IMMEDIATE JOIN_ASK CAPACITY DEV V7.2 — CORRECTED PRE-Q5 BBO HISTORY")
        print("=" * 174)
        print("Strategy/execution: unchanged")
        print("Capacity gate: positive PnL + >=99% overall terminal coverage + zero missing full-entry decisions")
        print("DEVELOPMENT ONLY — 15h sample NOT read")
        print()
        cols = [
            "requested_qty", "entry_fill_events", "full_entry_fill_orders", "partial_entry_fill_orders",
            "entry_filled_qty", "decision_snapshot_missing_full_entries", "median_join_ask_quote_c",
            "passive_exit_qty", "passive_exit_fraction_of_entry_qty", "full_passive_exit_positions",
            "m5_required_qty", "m5_exit_qty", "terminal_exit_fraction_of_entry_qty",
            "m5_residual_zero_valued", "m5_taker_fees", "rounding_drag",
            "m5_only_baseline_net", "join_ask_net_pnl_rounding_bound", "incremental_vs_m5_only",
            "net_pnl_per_filled_contract", "max_drawdown_rounding_bound",
            "top1_share_of_net", "top5_share_of_net",
            "capacity_gate_positive_99pct_terminal_no_missing_decision",
        ]
        print(surface[cols].to_string(index=False))
        print()
        print("Largest quantity passing corrected predeclared capacity gate:")
        print(largest)
        print()
        print("Highest development PnL row (diagnostic only):")
        print(max_pnl)
        print()
        print("IMPORTANT: V7/V7.1 Q>5 missing-decision counts were cache-horizon artifacts; use this V7.2 table.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "capacity_curve": surface,
        "detail": detail,
        "by_asset": by_asset,
        "output_dir": str(out),
    }
