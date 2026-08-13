from __future__ import annotations

"""Offline M1-M5 order-book reconstruction from recorded anchors + deltas.

This is a DATA RECOVERY / VALIDATION study, not a trading backtest.

Recorder facts used here:
- recorder.py subscribed to orderbook_delta with use_yes_price=True;
- full_books.jsonl is a periodic dump of the recorder's in-memory book;
- book_deltas.jsonl contains recorded orderbook deltas, but raw WS snapshots were
  not persisted separately.

Therefore reconstruction is conservative:
1. Only structurally valid two-sided full_books rows may seed/reset a book.
2. Recorded deltas are applied forward on the unified YES-price scale.
3. Connection-epoch changes and recorded sequence gaps invalidate the book until
   a new valid anchor appears.
4. Reconstructed BBO is compared with ticker_updates.jsonl where possible.
5. The existing >=80% M1-M5 coverage and <=5s start/end gap gate is unchanged.

No fill model, PnL, or regime threshold is evaluated here.
"""

import argparse
import bisect
import csv
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_m1_m5_feasibility as _base

STUDY_VERSION = "M1_M5_DELTA_BOOK_RECON_V1"
CRYPTO_SERIES = set(_base.CRYPTO_SERIES)
EPS = 1e-9


def _f(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _series(ticker):
    return str(ticker or "").split("-")[0]


def _window(close_ts, start_minute, end_minute, pre_seconds=0.0):
    start = close_ts - 900.0 + 60.0 * start_minute
    end = close_ts - 900.0 + 60.0 * end_minute
    return start, end, start - pre_seconds


def _parse_anchor(row, series, start_minute, end_minute, pre_seconds):
    ticker = _base._get_ticker(row)
    if not ticker or _series(ticker) not in series:
        return None
    t = _base._event_ts(row)
    close = _base._market_close_ts(row, ticker)
    if not np.isfinite(t) or not np.isfinite(close):
        return None
    wstart, wend, low = _window(close, start_minute, end_minute, pre_seconds)
    if not (low <= t <= wend + 1.0):
        return None

    bids = sorted(_base._levels(row.get("yes_bids") or []), key=lambda z: z[0], reverse=True)
    # recorder subscribed with use_yes_price=True, so yes_asks is already YES-scale.
    asks = sorted(_base._levels(row.get("yes_asks") or []), key=lambda z: z[0])
    valid = bool(bids and asks and 0.0 <= bids[0][0] < asks[0][0] <= 1.0)
    return {
        "t": float(t), "ticker": ticker, "series": _series(ticker),
        "close_ts": float(close), "epoch": row.get("connection_epoch"),
        "seq": row.get("book_seq"), "bids": bids, "asks": asks,
        "valid_anchor": valid,
        "source_age_s": _f(row.get("book_source_age_seconds")),
    }


def _parse_delta(row, series, start_minute, end_minute, pre_seconds):
    ticker = _base._get_ticker(row)
    if not ticker or _series(ticker) not in series:
        return None
    t = _base._event_ts(row)
    close = _base._market_close_ts(row, ticker)
    if not np.isfinite(t) or not np.isfinite(close):
        return None
    wstart, wend, low = _window(close, start_minute, end_minute, pre_seconds)
    if not (low <= t <= wend + 1.0):
        return None

    raw = row.get("raw_msg") if isinstance(row.get("raw_msg"), dict) else {}
    side = str(row.get("side") or raw.get("side") or "").lower()
    price = _f(row.get("price_dollars"))
    if not np.isfinite(price):
        price = _f(raw.get("price_dollars"))
    delta = _f(row.get("delta_fp"))
    if not np.isfinite(delta):
        delta = _f(raw.get("delta_fp"))
    if side not in {"yes", "no"} or not np.isfinite(price) or not np.isfinite(delta):
        return None
    if not (0.0 <= price <= 1.0):
        return None
    return {
        "t": float(t), "ticker": ticker, "series": _series(ticker),
        "close_ts": float(close), "epoch": row.get("connection_epoch"),
        "seq": row.get("seq"), "side": side,
        "price": float(price), "delta": float(delta),
    }


def _parse_ticker(row, series, start_minute, end_minute):
    ticker = _base._get_ticker(row)
    if not ticker or _series(ticker) not in series:
        return None
    t = _base._event_ts(row)
    close = _base._market_close_ts(row, ticker)
    if not np.isfinite(t) or not np.isfinite(close):
        return None
    wstart, wend, _ = _window(close, start_minute, end_minute)
    if not (wstart <= t < wend):
        return None
    bid = _base._price(row.get("yes_bid_dollars"))
    ask = _base._price(row.get("yes_ask_dollars"))
    if not np.isfinite(bid) and not np.isfinite(ask):
        return None
    return {
        "t": float(t), "ticker": ticker, "series": _series(ticker),
        "close_ts": float(close), "epoch": row.get("connection_epoch"),
        "bid": float(bid) if np.isfinite(bid) else np.nan,
        "ask": float(ask) if np.isfinite(ask) else np.nan,
    }


def _scan_jsonl(path, parser, args, label, progress_every):
    rows = defaultdict(list)
    meta = {}
    scanned = decoded = kept = 0
    t0 = time.time()
    with Path(path).open("rb") as f:
        for raw in f:
            scanned += 1
            if progress_every and scanned % progress_every == 0:
                print(f"  {label}: {scanned:,} lines | kept={kept:,} | {time.time()-t0:.1f}s")
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            decoded += 1
            item = parser(obj, *args)
            if item is None:
                continue
            ticker = item["ticker"]
            rows[ticker].append(item)
            meta[ticker] = {"ticker": ticker, "series": item["series"], "close_ts": item["close_ts"]}
            kept += 1
    for v in rows.values():
        v.sort(key=lambda z: z["t"])
    return rows, meta, {
        f"{label}_lines_scanned": scanned,
        f"{label}_lines_decoded": decoded,
        f"{label}_rows_kept": kept,
        f"{label}_seconds": time.time() - t0,
    }


def _scan_sequence_gaps(path):
    out = []
    path = Path(path)
    if not path.exists():
        return out
    with path.open("rb") as f:
        for raw in f:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if str(row.get("type") or "") != "sequence_gap":
                continue
            t = _base._event_ts(row)
            if np.isfinite(t):
                out.append((float(t), row.get("connection_epoch"), row.get("last_seq"), row.get("new_seq")))
    out.sort(key=lambda z: z[0])
    return out


def _book_status(bids, asks):
    if not bids and not asks:
        return "UNSEEDED"
    if not bids:
        return "MISSING_BID"
    if not asks:
        return "MISSING_ASK"
    bid, ask = max(bids), min(asks)
    if bid > ask + EPS:
        return "CROSSED"
    if abs(bid - ask) <= EPS:
        return "LOCKED"
    if not (0.0 <= bid < ask <= 1.0):
        return "BAD_PRICE"
    return "VALID"


def _top2(book, reverse=False):
    if not book:
        return np.nan, 0.0, np.nan, 0.0
    prices = sorted(book, reverse=reverse)
    p1 = prices[0]
    q1 = book[p1]
    if len(prices) == 1:
        return float(p1), float(q1), np.nan, 0.0
    p2 = prices[1]
    return float(p1), float(q1), float(p2), float(book[p2])


def _anchor_books(anchor):
    bids = {float(p): float(q) for p, q in anchor["bids"] if q > 0}
    asks = {float(p): float(q) for p, q in anchor["asks"] if q > 0}
    return bids, asks


def _apply_delta(bids, asks, ev):
    book = bids if ev["side"] == "yes" else asks
    p, d = ev["price"], ev["delta"]
    new = float(book.get(p, 0.0)) + d
    if new <= EPS:
        book.pop(p, None)
    else:
        book[p] = new


def _longest_false_run(mask):
    best = cur = 0
    for ok in mask:
        if ok:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def _reconstruct_contract(
    ticker, meta, anchors, deltas, ticker_updates, gap_events,
    start_minute, end_minute, min_coverage_pct, max_edge_gap_s,
    bbo_tolerance_c, writer,
):
    close = meta["close_ts"]
    wstart, wend, _ = _window(close, start_minute, end_minute)
    grid = np.arange(math.ceil(wstart), math.floor(wend), 1.0)
    if not len(grid):
        return None

    events = []
    for a in anchors:
        if a["valid_anchor"]:
            events.append((a["t"], 0, "anchor", a))
    for d in deltas:
        events.append((d["t"], 1, "delta", d))
    for gt, gep, old_seq, new_seq in gap_events:
        if wstart - 30.0 <= gt <= wend:
            events.append((gt, -1, "gap", {"epoch": gep, "old_seq": old_seq, "new_seq": new_seq}))
    events.sort(key=lambda z: (z[0], z[1]))

    bids, asks = {}, {}
    seeded = False
    epoch = None
    ei = 0
    anchor_count = delta_count = reset_count = anchor_mismatches = 0
    states = []

    for t in grid:
        while ei < len(events) and events[ei][0] <= t + EPS:
            _, _, kind, ev = events[ei]
            if kind == "gap":
                bids, asks, seeded = {}, {}, False
                epoch = ev.get("epoch")
                reset_count += 1
            elif kind == "anchor":
                if seeded and _book_status(bids, asks) == "VALID":
                    rb, ra = max(bids), min(asks)
                    ab = max(p for p, _ in ev["bids"])
                    aa = min(p for p, _ in ev["asks"])
                    if abs(rb - ab) > 1e-5 or abs(ra - aa) > 1e-5:
                        anchor_mismatches += 1
                bids, asks = _anchor_books(ev)
                seeded = True
                epoch = ev.get("epoch")
                anchor_count += 1
            else:
                ev_epoch = ev.get("epoch")
                if epoch is not None and ev_epoch is not None and ev_epoch != epoch:
                    bids, asks, seeded = {}, {}, False
                    epoch = ev_epoch
                    reset_count += 1
                if seeded:
                    _apply_delta(bids, asks, ev)
                    delta_count += 1
            ei += 1

        status = _book_status(bids, asks) if seeded else "UNSEEDED"
        if status == "VALID":
            b1, b1q, b2, b2q = _top2(bids, reverse=True)
            a1, a1q, a2, a2q = _top2(asks, reverse=False)
            mid = 0.5 * (b1 + a1)
            spread_c = 100.0 * (a1 - b1)
        else:
            b1 = b2 = a1 = a2 = mid = spread_c = np.nan
            b1q = b2q = a1q = a2q = 0.0

        row = {
            "ticker": ticker, "series": meta["series"],
            "time": datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat(),
            "ts": float(t), "minute": (t - (close - 900.0)) / 60.0,
            "status": status,
            "yes_bid1": b1, "yes_bid1_qty": b1q, "yes_bid2": b2, "yes_bid2_qty": b2q,
            "yes_ask1": a1, "yes_ask1_qty": a1q, "yes_ask2": a2, "yes_ask2_qty": a2q,
            "mid": mid, "spread_c": spread_c, "connection_epoch": epoch,
        }
        states.append(row)
        writer.writerow(row)

    valid = [r["status"] == "VALID" for r in states]
    valid_ts = [r["ts"] for r in states if r["status"] == "VALID"]
    coverage = 100.0 * sum(valid) / len(states)
    start_gap = valid_ts[0] - wstart if valid_ts else np.nan
    end_gap = wend - valid_ts[-1] if valid_ts else np.nan
    reasons = []
    if coverage < min_coverage_pct:
        reasons.append(f"reconstructed coverage {coverage:.1f}% < {min_coverage_pct:.1f}%")
    if not np.isfinite(start_gap) or start_gap > max_edge_gap_s:
        reasons.append("start gap > 5s")
    if not np.isfinite(end_gap) or end_gap > max_edge_gap_s:
        reasons.append("end gap > 5s")

    # Compare ticker BBO with the latest reconstructed one-second state at-or-before the ticker event.
    state_times = [r["ts"] for r in states]
    tol = bbo_tolerance_c / 100.0
    pair_n = pair_exact = pair_tol = 0
    bid_errors, ask_errors = [], []
    for tu in ticker_updates:
        if not (np.isfinite(tu["bid"]) and np.isfinite(tu["ask"])):
            continue
        i = bisect.bisect_right(state_times, tu["t"]) - 1
        if i < 0 or states[i]["status"] != "VALID":
            continue
        s = states[i]
        be = abs(float(s["yes_bid1"]) - tu["bid"])
        ae = abs(float(s["yes_ask1"]) - tu["ask"])
        bid_errors.append(100.0 * be)
        ask_errors.append(100.0 * ae)
        pair_n += 1
        pair_exact += int(be <= 1e-5 and ae <= 1e-5)
        pair_tol += int(be <= tol + EPS and ae <= tol + EPS)

    statuses = Counter(r["status"] for r in states)
    return {
        "ticker": ticker, "series": meta["series"],
        "close_time": datetime.fromtimestamp(close, tz=timezone.utc).isoformat(),
        "expected_seconds": len(states), "valid_seconds": int(sum(valid)),
        "reconstructed_coverage_pct": coverage,
        "start_gap_s": start_gap, "end_gap_s": end_gap,
        "longest_invalid_gap_s": float(_longest_false_run(valid)),
        "quality_ok": not reasons, "quality_reason": "; ".join(reasons) if reasons else "OK",
        "anchors_available": len(anchors), "valid_anchors_used": anchor_count,
        "deltas_available": len(deltas), "deltas_applied": delta_count,
        "state_resets": reset_count,
        "anchor_bbo_mismatches_before_resync": anchor_mismatches,
        "ticker_updates": len(ticker_updates), "ticker_pair_comparisons": pair_n,
        "bbo_both_exact_pct": 100.0 * pair_exact / pair_n if pair_n else np.nan,
        "bbo_both_within_tol_pct": 100.0 * pair_tol / pair_n if pair_n else np.nan,
        "median_abs_bid_error_c": float(np.median(bid_errors)) if bid_errors else np.nan,
        "median_abs_ask_error_c": float(np.median(ask_errors)) if ask_errors else np.nan,
        "unseeded_seconds": statuses["UNSEEDED"], "crossed_seconds": statuses["CROSSED"],
        "locked_seconds": statuses["LOCKED"], "missing_bid_seconds": statuses["MISSING_BID"],
        "missing_ask_seconds": statuses["MISSING_ASK"], "bad_price_seconds": statuses["BAD_PRICE"],
    }


def _asset_summary(df):
    if df.empty:
        return pd.DataFrame()
    rows = []
    for series, g in df.groupby("series"):
        rows.append({
            "series": series, "contracts": len(g),
            "quality_contracts": int(g["quality_ok"].sum()),
            "quality_pct": 100.0 * g["quality_ok"].mean(),
            "median_reconstructed_coverage_pct": g["reconstructed_coverage_pct"].median(),
            "median_bbo_exact_pct": g["bbo_both_exact_pct"].median(),
            "median_bbo_within_tol_pct": g["bbo_both_within_tol_pct"].median(),
            "median_ticker_pair_comparisons": g["ticker_pair_comparisons"].median(),
            "median_unseeded_seconds": g["unseeded_seconds"].median(),
            "median_crossed_seconds": g["crossed_seconds"].median(),
            "median_anchor_mismatches": g["anchor_bbo_mismatches_before_resync"].median(),
        })
    return pd.DataFrame(rows).sort_values(["quality_pct", "series"], ascending=[False, True])


def _minute_summary(samples_path):
    agg = defaultdict(Counter)
    with Path(samples_path).open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            minute = _f(r.get("minute"))
            if not np.isfinite(minute):
                continue
            lo = int(math.floor(minute))
            bucket = f"M{lo}-M{lo+1}"
            st = r.get("status") or "UNKNOWN"
            agg[bucket]["total"] += 1
            agg[bucket][st] += 1
    rows = []
    for bucket in sorted(agg):
        a = agg[bucket]
        n = a["total"]
        rows.append({
            "minute_bucket": bucket, "expected_seconds": n,
            "valid_pct": 100.0 * a["VALID"] / n if n else np.nan,
            "unseeded_pct": 100.0 * a["UNSEEDED"] / n if n else np.nan,
            "crossed_pct": 100.0 * a["CROSSED"] / n if n else np.nan,
            "locked_pct": 100.0 * a["LOCKED"] / n if n else np.nan,
            "missing_side_pct": 100.0 * (a["MISSING_BID"] + a["MISSING_ASK"]) / n if n else np.nan,
        })
    return pd.DataFrame(rows)


def _print_report(df, asset_df, minute_df, stats, out, min_cov, max_gap, tol_c):
    print("\n" + "=" * 108)
    print("M1-M5 DELTA BOOK RECONSTRUCTION + TICKER-BBO VALIDATION — NO TRADING SIMULATION")
    print("=" * 108)
    passes = int(df["quality_ok"].sum()) if len(df) else 0
    print(f"Contracts reconstructed: {len(df):,} | frozen-gate passes: {passes:,}/{len(df):,} ({100.0*passes/len(df) if len(df) else 0:.2f}%)")
    if len(df):
        print(f"Median reconstructed coverage={df['reconstructed_coverage_pct'].median():.2f}%")
        print(f"Median exact ticker-BBO agreement={df['bbo_both_exact_pct'].median():.2f}%")
        print(f"Median ticker-BBO agreement within {tol_c:.2f}c={df['bbo_both_within_tol_pct'].median():.2f}%")
        print(f"Median unseeded seconds={df['unseeded_seconds'].median():.1f}/240 | crossed={df['crossed_seconds'].median():.1f}/240")
        print(f"Median valid-anchor mismatch count={df['anchor_bbo_mismatches_before_resync'].median():.1f}")
    print(f"Frozen gate unchanged: coverage >= {min_cov:.1f}% and M1/M5 edge gaps <= {max_gap:.1f}s")

    print("\nBY ASSET")
    if len(asset_df):
        print(asset_df.round(2).to_string(index=False))
    print("\nM1-M5 RECONSTRUCTED COVERAGE")
    if len(minute_df):
        print(minute_df.round(2).to_string(index=False))

    print("\nINTERPRETATION")
    print("  High coverage + high ticker-BBO agreement => historical books are usable for the next MM replay.")
    print("  High coverage + poor BBO agreement => reconstructed state is not trustworthy.")
    print("  Low coverage dominated by UNSEEDED => not enough persisted valid anchors to recover state.")
    print("  Low coverage dominated by CROSSED => recorded state/deltas remain internally inconsistent.")
    print("  No market-making edge or PnL is inferred here.")
    print("\nOutputs:", out)
    print("=" * 108)


def run_m1_m5_delta_book_reconstruction(
    session_dir,
    output_dir=None,
    *,
    start_minute=1.0,
    end_minute=5.0,
    pre_seconds=30.0,
    crypto_series=None,
    min_book_coverage_pct=80.0,
    max_edge_gap_s=5.0,
    bbo_tolerance_c=0.25,
    show=True,
):
    session = Path(session_dir)
    if not session.exists():
        raise FileNotFoundError(session)
    full_books = session / "full_books.jsonl"
    deltas_file = session / "book_deltas.jsonl"
    ticker_file = session / "ticker_updates.jsonl"
    conn_file = session / "connection_events.jsonl"
    for p in (full_books, deltas_file, ticker_file):
        if not p.exists():
            raise FileNotFoundError(p)

    if not (0 <= start_minute < end_minute <= 15):
        raise ValueError("Require 0 <= start_minute < end_minute <= 15")
    series = set(crypto_series or CRYPTO_SERIES)

    if output_dir is None:
        root = session.resolve().parents[2] if len(session.resolve().parents) >= 3 else Path.cwd()
        output_dir = root / "results" / "kalshi_mm_m1_m5_delta_reconstruction" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{session.name}] scanning full-book anchors...")
    anchors, ma, sa = _scan_jsonl(
        full_books, _parse_anchor, (series, start_minute, end_minute, pre_seconds),
        "full_book", 250_000,
    )
    print(f"[{session.name}] scanning orderbook deltas...")
    deltas, md, sd = _scan_jsonl(
        deltas_file, _parse_delta, (series, start_minute, end_minute, pre_seconds),
        "delta", 1_000_000,
    )
    print(f"[{session.name}] scanning ticker BBO updates...")
    tickers, mt, st = _scan_jsonl(
        ticker_file, _parse_ticker, (series, start_minute, end_minute),
        "ticker", 1_000_000,
    )
    print(f"[{session.name}] reading connection-gap resets...")
    gaps = _scan_sequence_gaps(conn_file)

    meta = {}
    for src in (ma, md, mt):
        meta.update(src)
    targets = sorted(meta, key=lambda x: (meta[x]["close_ts"], x))

    sample_fields = [
        "ticker", "series", "time", "ts", "minute", "status",
        "yes_bid1", "yes_bid1_qty", "yes_bid2", "yes_bid2_qty",
        "yes_ask1", "yes_ask1_qty", "yes_ask2", "yes_ask2_qty",
        "mid", "spread_c", "connection_epoch",
    ]
    samples_path = out / "reconstructed_book_samples.csv"
    summaries = []
    print(f"[{session.name}] reconstructing {len(targets):,} contracts at 1 Hz...")
    t0 = time.time()
    with samples_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        for i, ticker in enumerate(targets, 1):
            s = _reconstruct_contract(
                ticker, meta[ticker], anchors.get(ticker, []), deltas.get(ticker, []),
                tickers.get(ticker, []), gaps, start_minute, end_minute,
                min_book_coverage_pct, max_edge_gap_s, bbo_tolerance_c, writer,
            )
            if s is not None:
                summaries.append(s)
            if i % 250 == 0 or i == len(targets):
                print(f"  reconstructed {i:,}/{len(targets):,} | {time.time()-t0:.1f}s")

    df = pd.DataFrame(summaries)
    asset_df = _asset_summary(df)
    minute_df = _minute_summary(samples_path)
    fail = Counter()
    if len(df):
        for reason in df.loc[~df["quality_ok"], "quality_reason"]:
            for part in str(reason).split("; "):
                if part:
                    fail[part] += 1
    fail_df = pd.DataFrame([{"failure_reason": k, "contracts": v} for k, v in fail.most_common()])

    stats = {**sa, **sd, **st, "sequence_gap_events": len(gaps), "contracts_reconstructed": len(df)}
    df.to_csv(out / "contract_reconstruction_summary.csv", index=False)
    asset_df.to_csv(out / "asset_reconstruction_summary.csv", index=False)
    minute_df.to_csv(out / "minute_reconstruction_summary.csv", index=False)
    fail_df.to_csv(out / "quality_failure_summary.csv", index=False)
    pd.DataFrame([stats]).to_csv(out / "scan_stats.csv", index=False)

    config = {
        "study_version": STUDY_VERSION,
        "session": str(session.resolve()),
        "start_minute": start_minute, "end_minute": end_minute,
        "pre_seconds": pre_seconds, "crypto_series": sorted(series),
        "min_book_coverage_pct": min_book_coverage_pct,
        "max_edge_gap_s": max_edge_gap_s,
        "bbo_tolerance_c": bbo_tolerance_c,
        "pricing": "UNIFIED_YES_PRICE; recorder subscribed with use_yes_price=True",
        "anchor_model": "valid full_books rows are complete state anchors; recorded deltas applied forward",
        "reset_model": "epoch changes and recorded sequence gaps invalidate until next valid anchor",
        "purpose": "reconstruction/validation only; no MM PnL simulation",
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if show:
        _print_report(df, asset_df, minute_df, stats, out, min_book_coverage_pct, max_edge_gap_s, bbo_tolerance_c)

    return {
        "output_dir": out,
        "contracts": df,
        "asset_summary": asset_df,
        "minute_summary": minute_df,
        "quality_failures": fail_df,
        "scan_stats": pd.DataFrame([stats]),
        "reconstructed_book_samples": samples_path,
    }


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--session", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--start-minute", type=float, default=1.0)
    p.add_argument("--end-minute", type=float, default=5.0)
    args = p.parse_args()
    run_m1_m5_delta_book_reconstruction(
        args.session, output_dir=args.output_dir,
        start_minute=args.start_minute, end_minute=args.end_minute, show=True,
    )


if __name__ == "__main__":
    _main()
