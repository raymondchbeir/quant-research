from __future__ import annotations

"""24-hour microstructure-classified census of V13 trade pillars.

V15 counts every raw V13 pillar across the formal OOS day.  After seeing V14,
that raw count is not enough: many visually dramatic pillars fail exchange-clock
book/trade reconciliation and must not be silently treated as confirmed sweeps.

V16 therefore:
1) reuses V15's exact raw pillar definition over the full 24h capture;
2) reuses V15's receipt-time BBO-at-start (>2c) diagnostic;
3) reconstructs the trade rows belonging to every detected pillar;
4) attaches the nearest prior exchange-clock BBO using V14's 250ms tolerance;
5) applies V14's exact microstructure classifier to every pillar; and
6) reports counts per 15-minute close-time window, including zero-event windows.

This is exploratory same-realization research only.  No trading policy is created
or validated.  NO API calls, NO orders, source capture read-only.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_trade_pillar_24h_census_v15 as V15
from . import mm_cycle_q10_trade_pillar_sweep_microstructure_v14 as V14

VERSION = "MM_CYCLE_Q10_TRADE_PILLAR_24H_MICROSTRUCTURE_CENSUS_V16"
HARD_BOUND_SESSION = V15.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q10_trade_pillar_24h_microstructure_census_v16"
EPS = 1e-9


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _ts(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _collect_pillar_trades(source: Path, pillars: pd.DataFrame, *, show=True):
    """Second pass over public trades, assigning rows to non-overlapping V15 bursts."""
    targets = defaultdict(list)
    for idx, r in pillars.reset_index(drop=True).iterrows():
        targets[str(r["ticker"])].append((r["receipt_start"], r["receipt_end"], int(idx)))
    for t in targets:
        targets[t].sort(key=lambda x: x[0])

    ptr = defaultdict(int)
    rows = defaultdict(list)
    scanned = 0
    assigned = 0

    for r in _iter_jsonl(source / "trades_event_time.jsonl"):
        scanned += 1
        ticker = str(r.get("ticker") or "")
        arr = targets.get(ticker)
        if not arr:
            continue
        rt = _ts(r.get("receipt_time"))
        if pd.isna(rt):
            continue
        i = ptr[ticker]
        while i < len(arr) and rt > arr[i][1] + pd.Timedelta(microseconds=1):
            i += 1
        ptr[ticker] = i
        if i >= len(arr):
            continue
        start, end, idx = arr[i]
        if start - pd.Timedelta(microseconds=1) <= rt <= end + pd.Timedelta(microseconds=1):
            px, qty = _f(r.get("yes_price")), _f(r.get("qty"))
            if np.isfinite(px) and np.isfinite(qty) and qty > 0:
                rows[idx].append({
                    "receipt_time": rt,
                    "exchange_time": _ts(r.get("exchange_time")),
                    "trade_id": str(r.get("trade_id") or ""),
                    "taker_book_side": str(r.get("taker_book_side") or "").lower(),
                    "price": float(px),
                    "qty": float(qty),
                })
                assigned += 1
        if show and scanned % 250_000 == 0:
            print(f"trade detail scan: {scanned:,} rows | assigned pillar trades={assigned:,}")
    return rows


def _target_exchange_ranges(pillars: pd.DataFrame, trade_rows):
    ranges = defaultdict(list)
    pad = pd.Timedelta(milliseconds=V14.BOOK_ASOF_TOLERANCE_MS)
    for idx, r in pillars.reset_index(drop=True).iterrows():
        g = pd.DataFrame(trade_rows.get(int(idx), []))
        if g.empty or g["exchange_time"].dropna().empty:
            continue
        x0 = g["exchange_time"].dropna().min() - pad
        x1 = g["exchange_time"].dropna().max() + pd.Timedelta(milliseconds=5)
        ranges[str(r["ticker"])].append((x0, x1, int(idx)))
    for t in ranges:
        ranges[t].sort(key=lambda x: x[0])
    return ranges


def _collect_pillar_books(source: Path, ranges, *, show=True):
    """Collect only exchange-clock book rows needed around detected pillars."""
    rows = defaultdict(list)
    scanned = 0
    kept = 0
    for r in _iter_jsonl(source / "book_top3_events.jsonl"):
        scanned += 1
        ticker = str(r.get("ticker") or "")
        arr = ranges.get(ticker)
        if not arr:
            continue
        xt = _ts(r.get("exchange_time"))
        if pd.isna(xt):
            continue
        bid, ask = _f(r.get("yes_bid")), _f(r.get("yes_ask"))
        if not (np.isfinite(bid) and np.isfinite(ask) and 0 <= bid < ask <= 1):
            continue
        # There are very few ranges per market ticker; a direct scan is robust.
        for x0, x1, idx in arr:
            if x0 <= xt <= x1:
                rows[idx].append({
                    "exchange_time": xt,
                    "bid": float(bid),
                    "ask": float(ask),
                    "spread_c": 100.0 * (float(ask) - float(bid)),
                    "event_type": str(r.get("event_type") or ""),
                })
                kept += 1
                break
        if show and scanned % 1_000_000 == 0:
            print(f"book microstructure scan: {scanned:,} rows | kept={kept:,}")
    return rows


def _classify_all(pillars: pd.DataFrame, trade_rows, book_rows):
    out_rows = []
    for idx, base in pillars.reset_index(drop=True).iterrows():
        g = pd.DataFrame(trade_rows.get(int(idx), []))
        b = pd.DataFrame(book_rows.get(int(idx), []))
        row = dict(base)
        if g.empty:
            row.update({
                "micro_trade_rows": 0,
                "prior_exchange_bbo_coverage_pct": np.nan,
                "directional_vs_prior_exchange_bbo_ok_pct": np.nan,
                "max_trades_same_exchange_ts": 0,
                "max_price_range_same_exchange_ts_c": np.nan,
                "classification": "TRADE_DETAIL_RECONSTRUCTION_MISSING",
            })
            out_rows.append(row)
            continue

        g = g.sort_values(["exchange_time", "receipt_time", "trade_id"])
        if not b.empty:
            b = b.sort_values("exchange_time")
            gx = V14._attach_exchange_bbo(b, g)
        else:
            gx = g.copy()
            for c in (
                "prior_exchange_bid", "prior_exchange_ask", "prior_exchange_spread_c",
                "prior_exchange_book_event_type", "directional_vs_prior_exchange_bbo_ok",
                "distance_beyond_touch_c",
            ):
                gx[c] = np.nan

        ts_stats = V14._same_timestamp_stats(gx)
        bid_stats = V14._side_stats(gx, "bid")
        ask_stats = V14._side_stats(gx, "ask")
        valid_bbo = gx[["prior_exchange_bid", "prior_exchange_ask"]].notna().all(axis=1)
        ok = pd.to_numeric(gx.get("directional_vs_prior_exchange_bbo_ok"), errors="coerce")
        dist = pd.to_numeric(gx.get("distance_beyond_touch_c"), errors="coerce")

        row.update({
            "micro_trade_rows": int(len(gx)),
            "prior_exchange_bbo_coverage_pct": 100.0 * float(valid_bbo.mean()) if len(gx) else np.nan,
            "directional_vs_prior_exchange_bbo_ok_pct": 100.0 * float(ok.dropna().mean()) if len(ok.dropna()) else np.nan,
            "median_distance_beyond_touch_c": float(dist.dropna().median()) if len(dist.dropna()) else np.nan,
            **bid_stats,
            **ask_stats,
            **ts_stats,
        })
        row["classification"] = V14._classify(row)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _all_windows(metadata_rows):
    by_close = defaultdict(set)
    for r in metadata_rows:
        close = str(r.get("close_time") or "")
        series = str(r.get("series") or "")
        if close and series:
            by_close[close].add(series)
    return pd.DataFrame([
        {"close_time": close, "markets_in_metadata": len(series)}
        for close, series in sorted(by_close.items())
    ])


def _aggregate_windows(metadata_rows, p: pd.DataFrame):
    out = _all_windows(metadata_rows)
    if p.empty:
        for c in (
            "raw_pillars", "raw_pillars_spread_gt_2c", "strict_one_sided_fast_sweeps",
            "strict_sweeps_spread_gt_2c", "clock_or_state_mismatches", "mixed_fast_flow",
            "other_classifications", "pillar_markets",
        ):
            out[c] = 0
        return out

    q = p.copy()
    q["is_strict"] = q["classification"].eq("LIKELY_ONE_SIDED_FAST_SWEEP")
    q["is_mismatch"] = q["classification"].eq("BOOK_TRADE_CLOCK_OR_STATE_MISMATCH")
    q["is_mixed"] = q["classification"].astype(str).str.startswith("MIXED_DIRECTION")
    q["is_other"] = ~(q["is_strict"] | q["is_mismatch"] | q["is_mixed"])
    q["wide"] = q["wide_spread_at_start"].fillna(False).astype(bool)

    rows = []
    for close, g in q.groupby("close_time", dropna=False):
        rows.append({
            "close_time": str(close),
            "raw_pillars": int(len(g)),
            "raw_pillars_spread_gt_2c": int(g["wide"].sum()),
            "strict_one_sided_fast_sweeps": int(g["is_strict"].sum()),
            "strict_sweeps_spread_gt_2c": int((g["is_strict"] & g["wide"]).sum()),
            "clock_or_state_mismatches": int(g["is_mismatch"].sum()),
            "mixed_fast_flow": int(g["is_mixed"].sum()),
            "other_classifications": int(g["is_other"].sum()),
            "pillar_markets": int(g["ticker"].nunique()),
            "raw_pillar_qty": float(pd.to_numeric(g["qty"], errors="coerce").fillna(0).sum()),
            "strict_sweep_qty": float(pd.to_numeric(g.loc[g["is_strict"], "qty"], errors="coerce").fillna(0).sum()),
            "max_price_range_c": float(pd.to_numeric(g["price_range_c"], errors="coerce").max()),
        })
    agg = pd.DataFrame(rows)
    out = out.merge(agg, on="close_time", how="left")
    count_cols = [
        "raw_pillars", "raw_pillars_spread_gt_2c", "strict_one_sided_fast_sweeps",
        "strict_sweeps_spread_gt_2c", "clock_or_state_mismatches", "mixed_fast_flow",
        "other_classifications", "pillar_markets",
    ]
    for c in count_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    for c in ("raw_pillar_qty", "strict_sweep_qty"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out.sort_values("close_time").reset_index(drop=True)


def _aggregate_assets(p: pd.DataFrame):
    if p.empty:
        return pd.DataFrame()
    q = p.copy()
    q["is_strict"] = q["classification"].eq("LIKELY_ONE_SIDED_FAST_SWEEP")
    q["is_mismatch"] = q["classification"].eq("BOOK_TRADE_CLOCK_OR_STATE_MISMATCH")
    q["wide"] = q["wide_spread_at_start"].fillna(False).astype(bool)
    rows = []
    for series, g in q.groupby("series", dropna=False):
        rows.append({
            "series": series,
            "raw_pillars": int(len(g)),
            "strict_one_sided_fast_sweeps": int(g["is_strict"].sum()),
            "strict_sweeps_spread_gt_2c": int((g["is_strict"] & g["wide"]).sum()),
            "clock_or_state_mismatches": int(g["is_mismatch"].sum()),
            "windows_with_raw_pillar": int(g["close_time"].nunique()),
            "windows_with_strict_sweep": int(g.loc[g["is_strict"], "close_time"].nunique()),
            "strict_sweep_qty": float(pd.to_numeric(g.loc[g["is_strict"], "qty"], errors="coerce").fillna(0).sum()),
            "max_price_range_c": float(pd.to_numeric(g["price_range_c"], errors="coerce").max()),
        })
    return pd.DataFrame(rows).sort_values(
        ["strict_one_sided_fast_sweeps", "raw_pillars"], ascending=False
    ).reset_index(drop=True)


def run_trade_pillar_24h_microstructure_census(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected formal OOS {HARD_BOUND_SESSION}, got {source.name}")
    for p in (source/"book_top3_events.jsonl", source/"trades_event_time.jsonl", source/"market_metadata.jsonl"):
        if not p.exists():
            raise FileNotFoundError(p)

    metadata_rows, metadata_by_ticker = V15._metadata(source)
    if show:
        print("PASS 1/4 — scanning all M1-M5 trades for raw V13 pillars...")
    pillars = V15._scan_pillars(source, metadata_by_ticker, show=show)
    if pillars.empty:
        raise RuntimeError("No raw V13 pillars found")
    if show:
        print(f"raw pillars found: {len(pillars):,}")
        print("PASS 2/4 — attaching receipt-time BBO at pillar start...")
    pillars = V15._attach_start_bbo(source, pillars, show=show).reset_index(drop=True)

    if show:
        print("PASS 3/4 — reconstructing exact public trades inside each pillar...")
    trade_rows = _collect_pillar_trades(source, pillars, show=show)
    ranges = _target_exchange_ranges(pillars, trade_rows)

    if show:
        print("PASS 4/4 — collecting exchange-clock BBO state and applying V14 classifier...")
    book_rows = _collect_pillar_books(source, ranges, show=show)
    detail = _classify_all(pillars, trade_rows, book_rows)

    by_window = _aggregate_windows(metadata_rows, detail)
    by_asset = _aggregate_assets(detail)
    out = _new_output(source.name)

    strict = detail[detail["classification"].eq("LIKELY_ONE_SIDED_FAST_SWEEP")]
    mismatch = detail[detail["classification"].eq("BOOK_TRADE_CLOCK_OR_STATE_MISMATCH")]
    mixed = detail[detail["classification"].astype(str).str.startswith("MIXED_DIRECTION")]
    wide = detail["wide_spread_at_start"].fillna(False).astype(bool)
    n_windows = len(by_window)

    summary = {
        "version": VERSION,
        "source_session": str(source),
        "windows_in_metadata": int(n_windows),
        "raw_v13_pillars": int(len(detail)),
        "raw_pillars_spread_gt_2c": int(wide.sum()),
        "strict_one_sided_fast_sweeps": int(len(strict)),
        "strict_sweeps_spread_gt_2c": int(strict["wide_spread_at_start"].fillna(False).astype(bool).sum()) if len(strict) else 0,
        "clock_or_state_mismatches": int(len(mismatch)),
        "mixed_direction_fast_flow": int(len(mixed)),
        "windows_with_raw_pillar": int((by_window["raw_pillars"] > 0).sum()),
        "windows_with_strict_sweep": int((by_window["strict_one_sided_fast_sweeps"] > 0).sum()),
        "windows_with_strict_sweep_spread_gt_2c": int((by_window["strict_sweeps_spread_gt_2c"] > 0).sum()),
        "mean_raw_pillars_per_window": float(by_window["raw_pillars"].mean()),
        "mean_strict_sweeps_per_window": float(by_window["strict_one_sided_fast_sweeps"].mean()),
        "median_strict_sweeps_per_window": float(by_window["strict_one_sided_fast_sweeps"].median()),
        "max_strict_sweeps_in_window": int(by_window["strict_one_sided_fast_sweeps"].max()),
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "guardrail": "Same historical realization; exploratory event census only. V14 classification is heuristic, not ground truth.",
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    detail.to_csv(out / "pillar_microstructure_detail.csv", index=False)
    by_window.to_csv(out / "pillar_microstructure_count_by_window.csv", index=False)
    by_asset.to_csv(out / "pillar_microstructure_count_by_asset.csv", index=False)

    if show:
        print("=" * 150)
        print("24H PILLAR MICROSTRUCTURE CENSUS V16 — READ ONLY")
        print("=" * 150)
        print("Source:", source)
        print(f"15-minute windows in metadata: {n_windows}")
        print(f"raw V13 pillars: {len(detail)}")
        print(f"raw pillars with spread >2c at start: {int(wide.sum())}")
        print(f"STRICT likely one-sided fast sweeps: {len(strict)}")
        print(f"strict sweeps with spread >2c at start: {summary['strict_sweeps_spread_gt_2c']}")
        print(f"book/trade clock-or-state mismatches: {len(mismatch)}")
        print(f"mixed-direction fast-flow classifications: {len(mixed)}")
        print()
        print(f"windows with >=1 raw pillar: {summary['windows_with_raw_pillar']} / {n_windows}")
        print(f"windows with >=1 strict sweep: {summary['windows_with_strict_sweep']} / {n_windows}")
        print(f"windows with >=1 strict sweep AND >2c spread: {summary['windows_with_strict_sweep_spread_gt_2c']} / {n_windows}")
        print(f"mean strict sweeps/window: {summary['mean_strict_sweeps_per_window']:.4f}")
        print(f"median strict sweeps/window: {summary['median_strict_sweeps_per_window']:.4f}")
        print(f"max strict sweeps in one window: {summary['max_strict_sweeps_in_window']}")
        print("\nCLASSIFICATION COUNTS")
        print(detail["classification"].value_counts().to_string())
        print("\nTOP 25 WINDOWS BY STRICT SWEEPS")
        cols = [
            "close_time", "markets_in_metadata", "strict_one_sided_fast_sweeps",
            "strict_sweeps_spread_gt_2c", "raw_pillars", "clock_or_state_mismatches",
            "pillar_markets", "strict_sweep_qty", "max_price_range_c",
        ]
        print(by_window.sort_values(
            ["strict_one_sided_fast_sweeps", "raw_pillars", "strict_sweep_qty"],
            ascending=False,
        ).head(25)[cols].to_string(index=False))
        print("\nBY ASSET")
        print(by_asset.to_string(index=False))
        print("\nInterpretation guardrail:")
        print("- Use strict sweep counts for the cleaner frequency estimate.")
        print("- Keep clock/state mismatches separate; do not count them as confirmed sweeps.")
        print("- This is exploratory same-realization research, not a strategy test.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 150)

    return {
        "summary": summary,
        "detail": detail,
        "by_window": by_window,
        "by_asset": by_asset,
        "output_dir": out,
    }


__all__ = ["VERSION", "run_trade_pillar_24h_microstructure_census"]
