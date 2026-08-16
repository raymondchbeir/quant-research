from __future__ import annotations

"""Forensic audit for V4 crossed/invalid event-time books.

DATA QUALITY ONLY. No strategy and no PnL.

The first V4 audit showed excellent sub-second resolution and strong agreement
between valid reconstructed books and the independent ticker feed, but a large
number of persisted rows had bid >= ask. This module diagnoses whether those
rows are explained by:

- locked books (bid == ask) versus truly crossed books (bid > ask),
- one corrupted side versus both sides disagreeing with ticker,
- snapshot/gap-recovery transitions,
- sequence-number resets/reordering versus positive sequence jumps,
- legacy-vs-unified NO-side price interpretation (heuristic cross-check),
- series-specific concentration,
- or M5 boundary-marker races (especially ZEC).

The session is read-only and is never modified.
"""

import bisect
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_VERSION = "MM_EVENT_TIME_M0_M5_CROSSED_FORENSICS_V1"
EPS_C = 1e-6
MAX_CROSSED_SAMPLE = 150_000
MAX_RUN_SAMPLE = 100_000
NEAREST_TICKER_MAX_S = 0.75
SNAPSHOT_NEAR_S = (0.10, 0.25, 0.50, 1.00)
GAP_NEAR_S = (0.10, 0.25, 0.50, 1.00)
RNG = random.Random(20260815)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _ts(x):
    if x is None:
        return None
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except Exception:
        return None


def _f(x):
    try:
        z = float(x)
        return z if np.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _reservoir(arr, row, seen, cap):
    if len(arr) < cap:
        arr.append(row)
        return
    j = RNG.randint(0, seen - 1)
    if j < cap:
        arr[j] = row


def _q(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0, "p50": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan}
    v = np.quantile(a, [0.50, 0.90, 0.95, 0.99])
    return {"n": len(a), "p50": v[0], "p90": v[1], "p95": v[2], "p99": v[3]}


def _pct(n, d):
    return 100.0 * float(n) / float(d) if d else np.nan


def _nearest_event(times, rows, t, max_s=NEAREST_TICKER_MAX_S):
    if not times:
        return None, np.nan
    i = bisect.bisect_left(times, t)
    candidates = []
    if i < len(times):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)
    if not candidates:
        return None, np.nan
    j = min(candidates, key=lambda k: abs(times[k] - t))
    age = abs(times[j] - t)
    if age > max_s:
        return None, age
    return rows[j], age


def _load_metadata(session: Path):
    path = session / "market_metadata.jsonl"
    rows = {}
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            try:
                x = json.loads(line)
            except Exception:
                continue
            ticker = x.get("ticker")
            if not ticker:
                continue
            close = _ts(x.get("close_time"))
            rows.setdefault(str(ticker), {
                "series": x.get("series_ticker"),
                "close_ts": close,
            })
    return rows


def _load_ticker(session: Path):
    by = defaultdict(list)
    path = session / "ticker_event_time.jsonl"
    if path.exists():
        with path.open() as fh:
            for line in fh:
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                ticker = x.get("ticker")
                t = _ts(x.get("receipt_time"))
                b, a = _f(x.get("yes_bid")), _f(x.get("yes_ask"))
                if not ticker or t is None or not (np.isfinite(b) and np.isfinite(a) and 0 <= b < a <= 1):
                    continue
                by[str(ticker)].append((t, b, a))
    times, rows = {}, {}
    for ticker, vals in by.items():
        vals.sort(key=lambda z: z[0])
        times[ticker] = [v[0] for v in vals]
        rows[ticker] = vals
    return times, rows


def _connection_gaps(session: Path):
    gaps = []
    counts = Counter()
    path = session / "connection_events.jsonl"
    if not path.exists():
        return gaps, counts
    with path.open() as fh:
        for line in fh:
            try:
                x = json.loads(line)
            except Exception:
                continue
            typ = str(x.get("type") or "")
            counts[typ] += 1
            if typ != "orderbook_sequence_gap":
                continue
            t = _ts(x.get("time"))
            exp = x.get("expected_seq")
            got = x.get("received_seq")
            try:
                exp_i, got_i = int(exp), int(got)
                jump = got_i - exp_i
            except Exception:
                jump = np.nan
            gaps.append({"time": t, "expected": exp, "received": got, "jump": jump})
    gaps.sort(key=lambda z: z["time"] if z["time"] is not None else -np.inf)
    return gaps, counts


def _rotation_removals(session: Path, meta):
    removals = {}
    path = session / "market_rotations.jsonl"
    if not path.exists():
        return removals
    with path.open() as fh:
        for line in fh:
            try:
                x = json.loads(line)
            except Exception:
                continue
            t = _ts(x.get("time"))
            if t is None:
                continue
            for ticker in x.get("removed") or []:
                m = meta.get(str(ticker), {})
                close = m.get("close_ts")
                if close is None:
                    continue
                start = float(close) - 900.0
                removals[str(ticker)] = t - start
    return removals


def run_crossed_book_forensics(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if not session.exists():
        raise FileNotFoundError(session)
    book_path = session / "book_top3_events.jsonl"
    if not book_path.exists():
        raise FileNotFoundError(book_path)

    meta = _load_metadata(session)
    ticker_times, ticker_rows = _load_ticker(session)
    gaps, conn_counts = _connection_gaps(session)
    gap_times = [g["time"] for g in gaps if g.get("time") is not None]
    removals = _rotation_removals(session, meta)
    manifest = _read_json(session / "session_manifest.json", {}) or {}

    totals = Counter()
    by_series = defaultdict(Counter)
    by_type = defaultdict(Counter)
    crossed_sample = []
    crossed_seen = 0
    crossed_spreads_c = []
    snapshot_ages = []
    gap_ages = []
    last_snapshot = {}
    last_state_class = {}
    run_start = {}
    run_last = {}
    run_rows = Counter()
    run_sample = []
    run_seen = 0
    contract = defaultdict(lambda: {"first_elapsed": np.inf, "last_elapsed": -np.inf, "capture_start": np.nan, "capture_end": np.nan, "rows": 0, "crossed": 0, "locked": 0})

    def close_run(ticker, end_t):
        nonlocal run_seen
        cls = last_state_class.get(ticker)
        if cls not in {"LOCKED", "CROSSED"}:
            return
        st = run_start.get(ticker)
        lt = run_last.get(ticker)
        if st is None or lt is None:
            return
        run_seen += 1
        _reservoir(run_sample, {
            "ticker": ticker,
            "series": meta.get(ticker, {}).get("series"),
            "class": cls,
            "start_ts": st,
            "end_ts": lt,
            "duration_s": max(0.0, lt - st),
            "rows": int(run_rows[ticker]),
        }, run_seen, MAX_RUN_SAMPLE)

    with book_path.open() as fh:
        for n, line in enumerate(fh, 1):
            try:
                x = json.loads(line)
            except Exception:
                totals["json_error"] += 1
                continue
            ticker = str(x.get("ticker") or "")
            if not ticker:
                continue
            series = str(x.get("series_ticker") or meta.get(ticker, {}).get("series") or "UNKNOWN")
            typ = str(x.get("event_type") or "UNKNOWN")
            t = _ts(x.get("receipt_time"))
            elapsed = _f(x.get("elapsed_s"))
            b, a = _f(x.get("yes_bid")), _f(x.get("yes_ask"))
            valid_nums = np.isfinite(b) and np.isfinite(a)

            totals["rows"] += 1
            by_series[series]["rows"] += 1
            by_type[typ]["rows"] += 1
            c = contract[ticker]
            c["rows"] += 1
            if np.isfinite(elapsed):
                c["first_elapsed"] = min(c["first_elapsed"], elapsed)
                c["last_elapsed"] = max(c["last_elapsed"], elapsed)
            if typ == "capture_start" and np.isfinite(elapsed):
                c["capture_start"] = elapsed
            if typ == "capture_end" and np.isfinite(elapsed):
                c["capture_end"] = elapsed
            if typ == "book_snapshot" and t is not None:
                last_snapshot[ticker] = t

            if not valid_nums:
                cls = "MISSING"
                totals["missing_bbo"] += 1
                by_series[series]["missing_bbo"] += 1
                by_type[typ]["missing_bbo"] += 1
            else:
                spread_c = 100.0 * (a - b)
                if spread_c > EPS_C:
                    cls = "VALID"
                    totals["valid"] += 1
                elif abs(spread_c) <= EPS_C:
                    cls = "LOCKED"
                    totals["locked"] += 1
                    by_series[series]["locked"] += 1
                    by_type[typ]["locked"] += 1
                    c["locked"] += 1
                else:
                    cls = "CROSSED"
                    totals["crossed"] += 1
                    by_series[series]["crossed"] += 1
                    by_type[typ]["crossed"] += 1
                    c["crossed"] += 1

                if cls in {"LOCKED", "CROSSED"}:
                    if cls == "CROSSED":
                        crossed_spreads_c.append(spread_c)
                    crossed_seen += 1
                    snap_age = (t - last_snapshot[ticker]) if t is not None and ticker in last_snapshot else np.nan
                    gi = bisect.bisect_right(gap_times, t) - 1 if t is not None and gap_times else -1
                    gap_age = (t - gap_times[gi]) if gi >= 0 and t is not None else np.nan
                    snapshot_ages.append(snap_age)
                    gap_ages.append(gap_age)
                    ask_levels = x.get("ask_levels") or []
                    alt_legacy_ask = np.nan
                    try:
                        vals = [float(z[0]) for z in ask_levels if isinstance(z, (list, tuple)) and len(z) >= 1]
                        if vals:
                            alt_legacy_ask = 1.0 - max(vals)
                    except Exception:
                        pass
                    _reservoir(crossed_sample, {
                        "ticker": ticker, "series": series, "event_type": typ,
                        "receipt_ts": t, "elapsed_s": elapsed,
                        "bid": b, "ask": a, "spread_c": spread_c,
                        "snapshot_age_s": snap_age, "gap_age_s": gap_age,
                        "alt_legacy_ask_heuristic": alt_legacy_ask,
                    }, crossed_seen, MAX_CROSSED_SAMPLE)

            prev_cls = last_state_class.get(ticker)
            if cls != prev_cls:
                if prev_cls in {"LOCKED", "CROSSED"} and t is not None:
                    close_run(ticker, t)
                if cls in {"LOCKED", "CROSSED"} and t is not None:
                    run_start[ticker] = t
                    run_rows[ticker] = 0
            if cls in {"LOCKED", "CROSSED"} and t is not None:
                run_last[ticker] = t
                run_rows[ticker] += 1
            last_state_class[ticker] = cls

            if n % 1_000_000 == 0 and show:
                print(f"  streamed {n:,} book rows...")

    for ticker in list(last_state_class):
        close_run(ticker, run_last.get(ticker, 0.0))

    # Crossed/locked sampled rows versus nearest independent ticker update.
    side_counts = Counter()
    ticker_matches = []
    legacy_better = 0
    unified_better = 0
    matched_sample = 0
    for r in crossed_sample:
        ticker = r["ticker"]
        tr, age = _nearest_event(ticker_times.get(ticker, []), ticker_rows.get(ticker, []), r["receipt_ts"] if r["receipt_ts"] is not None else -np.inf)
        if tr is None:
            continue
        _, tb, ta = tr
        be = abs(float(r["bid"]) - tb) * 100.0
        ae = abs(float(r["ask"]) - ta) * 100.0
        matched_sample += 1
        bclose, aclose = be <= 1.0 + 1e-9, ae <= 1.0 + 1e-9
        if bclose and aclose:
            label = "BOTH_WITHIN_1C"
        elif bclose and not aclose:
            label = "ASK_SIDE_MISMATCH"
        elif not bclose and aclose:
            label = "BID_SIDE_MISMATCH"
        else:
            label = "BOTH_SIDES_MISMATCH"
        side_counts[label] += 1

        alt = _f(r.get("alt_legacy_ask_heuristic"))
        if np.isfinite(alt):
            direct_err = abs(float(r["ask"]) - ta)
            legacy_err = abs(alt - ta)
            if direct_err + 1e-12 < legacy_err:
                unified_better += 1
            elif legacy_err + 1e-12 < direct_err:
                legacy_better += 1

        ticker_matches.append({
            **r,
            "ticker_age_ms": age * 1000.0,
            "ticker_bid": tb, "ticker_ask": ta,
            "bid_error_c": be, "ask_error_c": ae,
            "mismatch_class": label,
        })

    # Sequence-gap structure.
    jumps = np.asarray([_f(g.get("jump")) for g in gaps], dtype=float)
    positive = int(np.sum(jumps > 0))
    negative = int(np.sum(jumps < 0))
    zero = int(np.sum(jumps == 0))
    gap_cluster_100 = gap_cluster_250 = gap_cluster_1000 = 0
    if len(gap_times) >= 2:
        dg = np.diff(np.asarray(gap_times, dtype=float))
        gap_cluster_100 = int(np.sum(dg <= 0.10))
        gap_cluster_250 = int(np.sum(dg <= 0.25))
        gap_cluster_1000 = int(np.sum(dg <= 1.0))

    # Boundary marker vs actual removal timing.
    contract_rows = []
    for ticker, c in contract.items():
        first = c["first_elapsed"] if np.isfinite(c["first_elapsed"]) else np.nan
        last = c["last_elapsed"] if np.isfinite(c["last_elapsed"]) else np.nan
        rem = _f(removals.get(ticker))
        marker_ok = np.isfinite(c["capture_start"]) and np.isfinite(c["capture_end"])
        reached_m5_by_rotation = np.isfinite(rem) and rem >= 299.0
        contract_rows.append({
            "ticker": ticker,
            "series": meta.get(ticker, {}).get("series"),
            "rows": c["rows"], "crossed_rows": c["crossed"], "locked_rows": c["locked"],
            "crossed_or_locked_pct": _pct(c["crossed"] + c["locked"], c["rows"]),
            "capture_start_elapsed_s": c["capture_start"],
            "capture_end_elapsed_s": c["capture_end"],
            "first_book_elapsed_s": first, "last_book_elapsed_s": last,
            "rotation_remove_elapsed_s": rem,
            "marker_full_m0_m5": marker_ok,
            "rotation_indicates_reached_m5": reached_m5_by_rotation,
        })
    cdf = pd.DataFrame(contract_rows)

    srows = []
    for series, z in sorted(by_series.items()):
        rows = z["rows"]
        srows.append({
            "series": series, "rows": rows,
            "valid_rows": rows - z["locked"] - z["crossed"] - z["missing_bbo"],
            "locked_rows": z["locked"], "crossed_rows": z["crossed"], "missing_bbo": z["missing_bbo"],
            "locked_pct": _pct(z["locked"], rows), "crossed_pct": _pct(z["crossed"], rows),
            "locked_or_crossed_pct": _pct(z["locked"] + z["crossed"], rows),
        })
    sdf = pd.DataFrame(srows)

    trows = []
    for typ, z in sorted(by_type.items()):
        rows = z["rows"]
        trows.append({
            "event_type": typ, "rows": rows,
            "locked": z["locked"], "crossed": z["crossed"], "missing": z["missing_bbo"],
            "locked_or_crossed_pct": _pct(z["locked"] + z["crossed"], rows),
        })
    tdf = pd.DataFrame(trows)

    mdf = pd.DataFrame(ticker_matches)
    rdf = pd.DataFrame(run_sample)
    gdf = pd.DataFrame(gaps)

    if output_dir is None:
        output_dir = session.parents[3] / "results" / "kalshi_mm_event_m0_m5_v4_crossed_forensics" / session.name
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(out / "crossed_by_series.csv", index=False)
    tdf.to_csv(out / "crossed_by_event_type.csv", index=False)
    cdf.to_csv(out / "contract_boundary_and_crossed.csv", index=False)
    mdf.to_csv(out / "crossed_sample_vs_ticker.csv", index=False)
    rdf.to_csv(out / "crossed_run_sample.csv", index=False)
    gdf.to_csv(out / "sequence_gaps.csv", index=False)

    spread_q = _q(crossed_spreads_c)
    run_q = _q(rdf.duration_s.to_numpy(float) if len(rdf) else [])
    snap_q = _q([x for x in snapshot_ages if np.isfinite(x) and x >= 0])
    gap_age_q = _q([x for x in gap_ages if np.isfinite(x) and x >= 0])

    near_snapshot = {s: sum(np.isfinite(x) and 0 <= x <= s for x in snapshot_ages) for s in SNAPSHOT_NEAR_S}
    near_gap = {s: sum(np.isfinite(x) and 0 <= x <= s for x in gap_ages) for s in GAP_NEAR_S}

    missing_end = cdf[~np.isfinite(pd.to_numeric(cdf.capture_end_elapsed_s, errors="coerce"))].copy() if len(cdf) else pd.DataFrame()
    missing_end_reached = int(missing_end.rotation_indicates_reached_m5.sum()) if len(missing_end) else 0

    summary = {
        "study_version": STUDY_VERSION,
        "session": session.name,
        "book_rows": int(totals["rows"]),
        "valid_rows": int(totals["valid"]),
        "locked_rows": int(totals["locked"]),
        "crossed_rows": int(totals["crossed"]),
        "missing_bbo_rows": int(totals["missing_bbo"]),
        "locked_pct": _pct(totals["locked"], totals["rows"]),
        "crossed_pct": _pct(totals["crossed"], totals["rows"]),
        "locked_or_crossed_pct": _pct(totals["locked"] + totals["crossed"], totals["rows"]),
        "crossed_spread_p50_c": spread_q["p50"],
        "crossed_spread_p95_c": spread_q["p95"],
        "crossed_run_duration_p50_s": run_q["p50"],
        "crossed_run_duration_p95_s": run_q["p95"],
        "sample_ticker_matches": matched_sample,
        "sample_both_within_1c_pct": _pct(side_counts["BOTH_WITHIN_1C"], matched_sample),
        "sample_ask_mismatch_pct": _pct(side_counts["ASK_SIDE_MISMATCH"], matched_sample),
        "sample_bid_mismatch_pct": _pct(side_counts["BID_SIDE_MISMATCH"], matched_sample),
        "sample_both_mismatch_pct": _pct(side_counts["BOTH_SIDES_MISMATCH"], matched_sample),
        "unified_ask_interpretation_better": unified_better,
        "legacy_ask_heuristic_better": legacy_better,
        "sequence_gaps": len(gaps),
        "positive_seq_jumps": positive,
        "negative_seq_jumps": negative,
        "zero_seq_jumps": zero,
        "gap_pairs_within_100ms": gap_cluster_100,
        "gap_pairs_within_250ms": gap_cluster_250,
        "gap_pairs_within_1s": gap_cluster_1000,
        "contracts_missing_capture_end_marker": len(missing_end),
        "missing_end_but_rotation_reached_m5": missing_end_reached,
    }
    (out / "forensics_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("\n" + "=" * 126)
        print("V4 CROSSED-BOOK / GAP FORENSICS — NO STRATEGY / NO PNL")
        print("=" * 126)
        print(f"Session: {session}")
        print(f"Duration: {manifest.get('duration_hours')} h | connection epochs={manifest.get('connection_epochs')}")
        print("\nCROSSED STATE DECOMPOSITION")
        print(f"Rows={totals['rows']:,} | valid={totals['valid']:,} | locked={totals['locked']:,} ({_pct(totals['locked'], totals['rows']):.2f}%) | crossed={totals['crossed']:,} ({_pct(totals['crossed'], totals['rows']):.2f}%) | missing={totals['missing_bbo']:,}")
        print(f"True-cross spread p50/p95/p99: {spread_q['p50']:.4f}c / {spread_q['p95']:.4f}c / {spread_q['p99']:.4f}c")
        print(f"Locked/crossed run duration p50/p95/p99: {run_q['p50']:.4f}s / {run_q['p95']:.4f}s / {run_q['p99']:.4f}s")
        print("\nBY EVENT TYPE")
        print(tdf.round(4).to_string(index=False))
        print("\nBY SERIES")
        print(sdf.round(4).to_string(index=False))
        print("\nRELATION TO SNAPSHOT / GAP RECOVERY")
        print(f"Sampled locked/crossed rows={crossed_seen:,} total; reservoir={len(crossed_sample):,}")
        for s in SNAPSHOT_NEAR_S:
            print(f"within {int(s*1000):4d}ms of latest snapshot: {_pct(near_snapshot[s], crossed_seen):7.3f}%")
        for s in GAP_NEAR_S:
            print(f"within {int(s*1000):4d}ms of latest sequence gap: {_pct(near_gap[s], crossed_seen):7.3f}%")
        print(f"snapshot-age p50/p95: {snap_q['p50']*1000:.2f} / {snap_q['p95']*1000:.2f} ms")
        print(f"gap-age p50/p95:      {gap_age_q['p50']*1000:.2f} / {gap_age_q['p95']*1000:.2f} ms")
        print("\nCROSSED/LOCKED SAMPLE vs INDEPENDENT TICKER (nearest <=750ms)")
        print(f"matches={matched_sample:,}")
        for k in ("BOTH_WITHIN_1C", "ASK_SIDE_MISMATCH", "BID_SIDE_MISMATCH", "BOTH_SIDES_MISMATCH"):
            print(f"{k:24s} {side_counts[k]:8,d}  {_pct(side_counts[k], matched_sample):7.2f}%")
        print(f"Unified YES-scale ask heuristic better: {unified_better:,} | legacy/complement heuristic better: {legacy_better:,}")
        print("\nSEQUENCE-GAP STRUCTURE")
        print(f"gaps={len(gaps):,} | positive jumps={positive:,} | negative/reset-like={negative:,} | zero={zero:,}")
        print(f"consecutive gaps clustered <=100ms / <=250ms / <=1s: {gap_cluster_100:,} / {gap_cluster_250:,} / {gap_cluster_1000:,}")
        if len(gdf):
            jq = _q(np.abs(pd.to_numeric(gdf.jump, errors='coerce').to_numpy(float)))
            print(f"|seq jump| p50/p95/p99: {jq['p50']:.1f} / {jq['p95']:.1f} / {jq['p99']:.1f}")
        print("\nM5 BOUNDARY-MARKER FORENSICS")
        print(f"contracts={len(cdf)} | missing capture_end marker={len(missing_end)} | of those rotation timestamp says recorder reached >=M4:59: {missing_end_reached}")
        if len(missing_end):
            print(missing_end[["ticker","series","last_book_elapsed_s","rotation_remove_elapsed_s","crossed_or_locked_pct"]].head(50).round(4).to_string(index=False))
        print("\nOUTPUTS:", out)
        print("NO STRATEGY OR PNL WAS EVALUATED.")
        print("=" * 126)

    return {
        "output_dir": out,
        "summary": summary,
        "by_series": sdf,
        "by_event_type": tdf,
        "contracts": cdf,
        "crossed_sample_vs_ticker": mdf,
        "runs": rdf,
        "sequence_gaps": gdf,
    }
