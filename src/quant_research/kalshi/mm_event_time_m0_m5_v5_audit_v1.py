from __future__ import annotations

"""Integrity audit for MM_EVENT_TIME_M0_M5_V5_DEV recordings.

DATA QUALITY ONLY -- NO STRATEGY, NO PNL, NO PARAMETER SELECTION.

The audit is designed to run before the pre-registered A/B/C/D development
replay. It streams the large book file and checks:
- capture_start / M5 trade_window_end / M5+30 label_tail_end completeness;
- persisted book validity and independent crossed/locked detection;
- event-time book/trade resolution and trade receipt latency;
- ticker BBO agreement with the reconstructed book, including per-series/BTC;
- repair reasons and time from repair trigger to first fresh book snapshot;
- 30-second future-mid label availability for public trades in M0-M5;
- per-contract and per-series coverage/row counts;
- recorder connection/sequence health and file sizes.

A PASS_FOR_DEVELOPMENT verdict means only that the recording is suitable for
strategy DEVELOPMENT analysis. It is not evidence that any strategy works.
"""

import bisect
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C

STUDY_VERSION = "MM_EVENT_TIME_M0_M5_V5_AUDIT_V1"
EXPECTED_CAPTURE_VERSION = "MM_EVENT_TIME_M0_M5_V5_DEV"
MAX_RESERVOIR = 200_000
MAX_BOOK_AGE_FOR_TICKER_CHECK_S = 2.0
MAX_FUTURE_LABEL_AGE_S = 2.0
RNG_SEED = 20260816
BTC_SERIES = "KXBTC15M"


def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _parse_ts(x):
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


def _pct(num, den):
    return 100.0 * float(num) / float(den) if den else np.nan


def _reservoir_add(arr, x, seen, rng):
    if not np.isfinite(x):
        return
    if len(arr) < MAX_RESERVOIR:
        arr.append(float(x))
    else:
        j = rng.randrange(seen)
        if j < MAX_RESERVOIR:
            arr[j] = float(x)


def _quantiles(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    keys = ("p10", "p25", "p50", "p75", "p90", "p95", "p99")
    if not len(a):
        return {k: np.nan for k in keys}
    q = np.quantile(a, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return dict(zip(keys, map(float, q)))


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except Exception:
                continue


def _read_metadata(path: Path):
    rows = []
    for x in _read_jsonl(path) or []:
        ticker = str(x.get("ticker") or "")
        if not ticker:
            continue
        rows.append({
            "ticker": ticker,
            "series": str(x.get("series_ticker") or ""),
            "close_time": x.get("close_time"),
            "window_start": x.get("window_start"),
            "discovered_status": x.get("discovered_status"),
        })
    if not rows:
        return pd.DataFrame(columns=["ticker", "series", "close_time", "window_start", "discovered_status"])
    return pd.DataFrame(rows).drop_duplicates("ticker", keep="first").reset_index(drop=True)


def _read_connection_events(path: Path):
    counts = Counter()
    rows = []
    snapshot_requests = []
    for x in _read_jsonl(path) or []:
        typ = str(x.get("type") or "")
        counts[typ] += 1
        rows.append(x)
        if typ == "snapshot_request":
            snapshot_requests.append(x)
    return counts, rows, snapshot_requests


def _read_repairs(path: Path):
    repairs = []
    by_ticker = defaultdict(list)
    for x in _read_jsonl(path) or []:
        t = _parse_ts(x.get("time"))
        ticker = str(x.get("ticker") or "")
        if t is None or not ticker:
            continue
        r = {
            "time_ts": float(t),
            "time": x.get("time"),
            "ticker": ticker,
            "reason": str(x.get("reason") or "unknown"),
            "seq": x.get("seq"),
            "connection_epoch": x.get("connection_epoch"),
        }
        repairs.append(r)
        by_ticker[ticker].append(r)
    repairs.sort(key=lambda z: z["time_ts"])
    for z in by_ticker.values():
        z.sort(key=lambda r: r["time_ts"])
    return repairs, by_ticker


def _read_ticker_events(path: Path):
    rows = []
    counts_by_contract = Counter()
    total = valid = 0
    for x in _read_jsonl(path) or []:
        total += 1
        t = _parse_ts(x.get("receipt_time"))
        ticker = str(x.get("ticker") or "")
        if t is None or not ticker:
            continue
        counts_by_contract[ticker] += 1
        bid, ask = _f(x.get("yes_bid")), _f(x.get("yes_ask"))
        if np.isfinite(bid) and np.isfinite(ask) and 0.0 <= bid < ask <= 1.0:
            valid += 1
            rows.append((float(t), ticker, float(bid), float(ask), _f(x.get("elapsed_s"))))
    rows.sort(key=lambda z: z[0])
    return rows, counts_by_contract, total, valid


def _read_trades_and_targets(path: Path):
    total = bad = missing_side = missing_exchange = 0
    counts_by_contract = Counter()
    research_counts_by_contract = Counter()
    targets_by_ticker = defaultdict(list)
    trade_gap_samples = []
    latency_samples = []
    last_trade_t = {}
    gap_seen = latency_seen = gap_count = 0
    sub100 = sub250 = sub500 = sub1s = 0
    rng = random.Random(RNG_SEED + 1)

    for x in _read_jsonl(path) or []:
        total += 1
        ticker = str(x.get("ticker") or "")
        t = _parse_ts(x.get("receipt_time"))
        px, qty = _f(x.get("yes_price")), _f(x.get("qty"))
        elapsed = _f(x.get("elapsed_s"))
        side = str(x.get("taker_book_side") or "").lower()
        if not ticker or t is None or not np.isfinite(px) or not (0.0 <= px <= 1.0) or not np.isfinite(qty) or qty <= 0:
            bad += 1
            continue
        t = float(t)
        counts_by_contract[ticker] += 1
        if side not in {"bid", "ask"}:
            missing_side += 1
        exch = _parse_ts(x.get("exchange_time"))
        if exch is None:
            missing_exchange += 1
        else:
            lat = (t - float(exch)) * 1000.0
            if -5000 <= lat <= 60000:
                latency_seen += 1
                _reservoir_add(latency_samples, lat, latency_seen, rng)

        prev = last_trade_t.get(ticker)
        if prev is not None:
            d = t - prev
            if d >= 0:
                gap_count += 1
                gap_seen += 1
                _reservoir_add(trade_gap_samples, d, gap_seen, rng)
                sub100 += d <= 0.100
                sub250 += d <= 0.250
                sub500 += d <= 0.500
                sub1s += d <= 1.000
        last_trade_t[ticker] = t

        if np.isfinite(elapsed) and 0.0 <= elapsed < 300.0:
            research_counts_by_contract[ticker] += 1
            targets_by_ticker[ticker].append(t + 30.0)

    for arr in targets_by_ticker.values():
        arr.sort()

    resolution = {
        "trade_gap_count": gap_count,
        "trade_gap_le_100ms_pct": _pct(sub100, gap_count),
        "trade_gap_le_250ms_pct": _pct(sub250, gap_count),
        "trade_gap_le_500ms_pct": _pct(sub500, gap_count),
        "trade_gap_le_1s_pct": _pct(sub1s, gap_count),
        **{f"trade_gap_{k}_s": v for k, v in _quantiles(trade_gap_samples).items()},
        **{f"receipt_latency_{k}_ms": v for k, v in _quantiles(latency_samples).items()},
    }
    return {
        "total": total,
        "bad": bad,
        "missing_side": missing_side,
        "missing_exchange": missing_exchange,
        "counts_by_contract": counts_by_contract,
        "research_counts_by_contract": research_counts_by_contract,
        "targets_by_ticker": targets_by_ticker,
        "resolution": resolution,
    }


def run_event_time_m0_m5_v5_audit(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if not session.exists():
        raise FileNotFoundError(session)

    paths = {
        "book": session / "book_top3_events.jsonl",
        "trade": session / "trades_event_time.jsonl",
        "ticker": session / "ticker_event_time.jsonl",
        "meta": session / "market_metadata.jsonl",
        "rotations": session / "market_rotations.jsonl",
        "connection": session / "connection_events.jsonl",
        "repairs": session / "book_repair_events.jsonl",
        "manifest": session / "session_manifest.json",
        "health": session / "health.json",
        "capture_spec": session / "capture_spec.json",
        "development_plan": session / "development_plan.json",
    }
    for name in ("book", "trade", "ticker", "meta", "connection", "repairs", "manifest"):
        if not paths[name].exists():
            raise FileNotFoundError(paths[name])

    manifest = _load_json(paths["manifest"], {}) or {}
    health = _load_json(paths["health"], {}) or {}
    capture_spec = manifest.get("capture_spec") or _load_json(paths["capture_spec"], {}) or {}
    development_plan = manifest.get("development_plan") or _load_json(paths["development_plan"], {}) or {}
    capture_version = manifest.get("study_version") or capture_spec.get("study_version")
    if capture_version != EXPECTED_CAPTURE_VERSION:
        raise RuntimeError(f"Expected {EXPECTED_CAPTURE_VERSION}, found {capture_version}")

    rng = random.Random(RNG_SEED)
    meta = _read_metadata(paths["meta"])
    ticker_to_series = dict(zip(meta.ticker.astype(str), meta.series.astype(str))) if len(meta) else {}
    conn_counts, conn_rows, snapshot_requests = _read_connection_events(paths["connection"])
    repairs, repairs_by_ticker = _read_repairs(paths["repairs"])
    ticker_events, ticker_rows_by_contract, ticker_total, ticker_valid = _read_ticker_events(paths["ticker"])
    trade_info = _read_trades_and_targets(paths["trade"])

    book_event_counts = Counter()
    book_by_contract = defaultdict(Counter)
    first_elapsed = {}
    last_elapsed = {}
    boundary = defaultdict(dict)
    snapshot_times = defaultdict(list)
    last_book_t = {}
    book_gap_samples = []
    book_gap_seen = book_gap_count = 0
    book_sub100 = book_sub250 = book_sub500 = book_sub1s = 0
    invalid_rows = crossed_rows = locked_rows = missing_bbo_rows = 0
    invalid_dynamic_rows = crossed_dynamic_rows = 0
    series_book_counts = defaultdict(Counter)

    last_book_state = {}
    book_valid_for_compare = defaultdict(bool)
    ticker_i = 0
    repair_i = 0
    ticker_cmp = defaultdict(Counter)
    ticker_age_samples = []
    ticker_age_seen = 0

    targets = trade_info["targets_by_ticker"]
    target_ptr = defaultdict(int)
    target_counts = defaultdict(Counter)
    target_age_samples = []
    target_age_seen = 0

    def apply_repairs_until(t):
        nonlocal repair_i
        while repair_i < len(repairs) and repairs[repair_i]["time_ts"] <= t:
            r = repairs[repair_i]
            book_valid_for_compare[r["ticker"]] = False
            repair_i += 1

    def compare_tickers_until(t):
        nonlocal ticker_i, ticker_age_seen
        while ticker_i < len(ticker_events) and ticker_events[ticker_i][0] <= t:
            tt, ticker, tbid, task, _elapsed = ticker_events[ticker_i]
            apply_repairs_until(tt)
            series = ticker_to_series.get(ticker, "UNKNOWN")
            c = ticker_cmp[series]
            c["ticker_valid_rows"] += 1
            state = last_book_state.get(ticker)
            if not book_valid_for_compare[ticker] or state is None:
                c["skipped_invalid_or_no_book"] += 1
            else:
                bt, bbid, bask = state
                age = tt - bt
                if age < -1e-9 or age > MAX_BOOK_AGE_FOR_TICKER_CHECK_S:
                    c["skipped_stale_book"] += 1
                else:
                    c["checked"] += 1
                    ticker_age_seen += 1
                    _reservoir_add(ticker_age_samples, age * 1000.0, ticker_age_seen, rng)
                    be = abs(tbid - bbid) * 100.0
                    ae = abs(task - bask) * 100.0
                    if be <= 1e-7 and ae <= 1e-7:
                        c["both_exact"] += 1
                    if be <= 1.0 + 1e-9 and ae <= 1.0 + 1e-9:
                        c["both_within_1c"] += 1
                    else:
                        c["either_gt_1c"] += 1
                    if be > 1.0 + 1e-9:
                        c["bid_gt_1c"] += 1
                    if ae > 1.0 + 1e-9:
                        c["ask_gt_1c"] += 1
            ticker_i += 1

    def advance_trade_targets(ticker, book_t, mid_valid):
        nonlocal target_age_seen
        if not mid_valid:
            return
        arr = targets.get(ticker)
        if not arr:
            return
        j = target_ptr[ticker]
        while j < len(arr) and arr[j] <= book_t:
            age = book_t - arr[j]
            target_counts[ticker]["total_targets_seen"] += 1
            if 0.0 <= age <= MAX_FUTURE_LABEL_AGE_S:
                target_counts[ticker]["covered"] += 1
                target_age_seen += 1
                _reservoir_add(target_age_samples, age * 1000.0, target_age_seen, rng)
            else:
                target_counts[ticker]["missed_gt_2s"] += 1
            j += 1
        target_ptr[ticker] = j

    with paths["book"].open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            try:
                x = json.loads(line)
            except Exception:
                book_event_counts["json_errors"] += 1
                continue
            t = _parse_ts(x.get("receipt_time"))
            ticker = str(x.get("ticker") or "")
            if t is None or not ticker:
                book_event_counts["missing_time_or_ticker"] += 1
                continue
            t = float(t)
            compare_tickers_until(t)
            apply_repairs_until(t)

            typ = str(x.get("event_type") or "")
            series = str(x.get("series_ticker") or ticker_to_series.get(ticker) or "UNKNOWN")
            elapsed = _f(x.get("elapsed_s"))
            valid = bool(x.get("valid_bbo"))
            bid, ask = _f(x.get("yes_bid")), _f(x.get("yes_ask"))
            independent_cross = bool(np.isfinite(bid) and np.isfinite(ask) and bid > ask + 1e-12)
            independent_lock = bool(np.isfinite(bid) and np.isfinite(ask) and abs(bid - ask) <= 1e-12)
            crossed_flag = bool(x.get("crossed_or_locked"))

            book_event_counts[typ] += 1
            book_by_contract[ticker]["rows"] += 1
            book_by_contract[ticker][typ] += 1
            series_book_counts[series]["rows"] += 1
            series_book_counts[series][typ] += 1

            if np.isfinite(elapsed):
                first_elapsed[ticker] = min(first_elapsed.get(ticker, np.inf), elapsed)
                last_elapsed[ticker] = max(last_elapsed.get(ticker, -np.inf), elapsed)
                if typ in {"capture_start", "trade_window_end", "label_tail_end"}:
                    boundary[ticker][typ] = elapsed

            if not valid:
                invalid_rows += 1
            if not np.isfinite(bid) or not np.isfinite(ask):
                missing_bbo_rows += 1
            if crossed_flag or independent_cross or independent_lock:
                crossed_rows += 1
            if independent_lock:
                locked_rows += 1
            if typ in {"book_snapshot", "book_delta"}:
                if not valid:
                    invalid_dynamic_rows += 1
                if crossed_flag or independent_cross or independent_lock:
                    crossed_dynamic_rows += 1

            if typ == "book_snapshot":
                snapshot_times[ticker].append(t)
                if valid and np.isfinite(bid) and np.isfinite(ask) and bid < ask:
                    book_valid_for_compare[ticker] = True
            elif typ == "capture_start" and valid and np.isfinite(bid) and np.isfinite(ask) and bid < ask:
                book_valid_for_compare[ticker] = True

            if valid and np.isfinite(bid) and np.isfinite(ask) and 0.0 <= bid < ask <= 1.0:
                last_book_state[ticker] = (t, float(bid), float(ask))
                advance_trade_targets(ticker, t, True)

            prev = last_book_t.get(ticker)
            if prev is not None:
                d = t - prev
                if d >= 0:
                    book_gap_count += 1
                    book_gap_seen += 1
                    _reservoir_add(book_gap_samples, d, book_gap_seen, rng)
                    book_sub100 += d <= 0.100
                    book_sub250 += d <= 0.250
                    book_sub500 += d <= 0.500
                    book_sub1s += d <= 1.000
            last_book_t[ticker] = t

            if show and n % 1_000_000 == 0:
                print(f"  streamed {n:,} / book rows...")

    compare_tickers_until(float("inf"))
    apply_repairs_until(float("inf"))

    for ticker, arr in targets.items():
        j = target_ptr[ticker]
        if j < len(arr):
            target_counts[ticker]["not_reached_before_file_end"] += len(arr) - j
            target_counts[ticker]["total_targets_seen"] += len(arr) - j
            target_ptr[ticker] = len(arr)

    repair_rows = []
    repair_reason_counts = Counter()
    repair_series_counts = defaultdict(Counter)
    recovery_samples = []
    recovery_seen = 0
    for r in repairs:
        ticker = r["ticker"]
        series = ticker_to_series.get(ticker, "UNKNOWN")
        times = snapshot_times.get(ticker, [])
        j = bisect.bisect_left(times, r["time_ts"])
        rec_t = times[j] if j < len(times) else np.nan
        rec_ms = (rec_t - r["time_ts"]) * 1000.0 if np.isfinite(rec_t) else np.nan
        if np.isfinite(rec_ms) and rec_ms >= 0:
            recovery_seen += 1
            _reservoir_add(recovery_samples, rec_ms, recovery_seen, rng)
        repair_reason_counts[r["reason"]] += 1
        repair_series_counts[series][r["reason"]] += 1
        repair_series_counts[series]["repairs"] += 1
        repair_rows.append({
            **r,
            "series": series,
            "snapshot_recovery_time": datetime.fromtimestamp(rec_t, tz=timezone.utc).isoformat() if np.isfinite(rec_t) else None,
            "snapshot_recovery_ms": rec_ms,
            "recovered": bool(np.isfinite(rec_ms) and rec_ms >= 0),
        })
    repairs_df = pd.DataFrame(repair_rows)

    all_tickers = sorted(
        set(ticker_to_series)
        | set(book_by_contract)
        | set(ticker_rows_by_contract)
        | set(trade_info["counts_by_contract"])
    )
    contract_rows = []
    for ticker in all_tickers:
        b = boundary.get(ticker, {})
        s = _f(b.get("capture_start"))
        m5 = _f(b.get("trade_window_end"))
        tail = _f(b.get("label_tail_end"))
        full_start = np.isfinite(s) and -0.5 <= s <= 2.0
        full_m5 = np.isfinite(m5) and 299.5 <= m5 <= 302.0
        full_tail = np.isfinite(tail) and 329.5 <= tail <= 332.0
        tc = target_counts[ticker]
        target_total = int(tc["total_targets_seen"])
        target_covered = int(tc["covered"])
        contract_rows.append({
            "ticker": ticker,
            "series": ticker_to_series.get(ticker, "UNKNOWN"),
            "capture_start_elapsed_s": s,
            "trade_window_end_elapsed_s": m5,
            "label_tail_end_elapsed_s": tail,
            "full_m0_m5_boundary": bool(full_start and full_m5),
            "full_m0_m5_plus_30_boundary": bool(full_start and full_m5 and full_tail),
            "first_persisted_elapsed_s": first_elapsed.get(ticker, np.nan),
            "last_persisted_elapsed_s": last_elapsed.get(ticker, np.nan),
            "book_rows": book_by_contract[ticker]["rows"],
            "book_snapshots": book_by_contract[ticker]["book_snapshot"],
            "book_top3_deltas": book_by_contract[ticker]["book_delta"],
            "ticker_rows": ticker_rows_by_contract[ticker],
            "trade_rows": trade_info["counts_by_contract"][ticker],
            "research_trade_rows": trade_info["research_counts_by_contract"][ticker],
            "trade_30s_targets": target_total,
            "trade_30s_labels_covered": target_covered,
            "trade_30s_label_coverage_pct": _pct(target_covered, target_total),
            "repair_events": len(repairs_by_ticker.get(ticker, [])),
        })
    contracts = pd.DataFrame(contract_rows)

    agreement_rows = []
    for series in sorted(set(ticker_cmp) | set(meta.series.astype(str) if len(meta) else [])):
        c = ticker_cmp[series]
        agreement_rows.append({
            "series": series,
            "ticker_valid_rows": c["ticker_valid_rows"],
            "checked": c["checked"],
            "both_exact": c["both_exact"],
            "both_exact_pct": _pct(c["both_exact"], c["checked"]),
            "both_within_1c": c["both_within_1c"],
            "both_within_1c_pct": _pct(c["both_within_1c"], c["checked"]),
            "either_gt_1c": c["either_gt_1c"],
            "either_gt_1c_pct": _pct(c["either_gt_1c"], c["checked"]),
            "bid_gt_1c": c["bid_gt_1c"],
            "ask_gt_1c": c["ask_gt_1c"],
            "skipped_invalid_or_no_book": c["skipped_invalid_or_no_book"],
            "skipped_stale_book": c["skipped_stale_book"],
        })
    agreement = pd.DataFrame(agreement_rows)

    series_rows = []
    for series, z in contracts.groupby("series", dropna=False, sort=True):
        arow = agreement[agreement.series == series]
        ac = arow.iloc[0].to_dict() if len(arow) else {}
        rc = repair_series_counts[series]
        series_rows.append({
            "series": series,
            "contracts": len(z),
            "full_m0_m5_contracts": int(z.full_m0_m5_boundary.sum()),
            "full_m0_m5_pct": _pct(z.full_m0_m5_boundary.sum(), len(z)),
            "full_tail_contracts": int(z.full_m0_m5_plus_30_boundary.sum()),
            "full_tail_pct": _pct(z.full_m0_m5_plus_30_boundary.sum(), len(z)),
            "book_rows": int(z.book_rows.sum()),
            "ticker_rows": int(z.ticker_rows.sum()),
            "trade_rows": int(z.trade_rows.sum()),
            "research_trade_rows": int(z.research_trade_rows.sum()),
            "trade_30s_targets": int(z.trade_30s_targets.sum()),
            "trade_30s_labels_covered": int(z.trade_30s_labels_covered.sum()),
            "trade_30s_label_coverage_pct": _pct(z.trade_30s_labels_covered.sum(), z.trade_30s_targets.sum()),
            "ticker_checked": int(ac.get("checked", 0) or 0),
            "ticker_both_exact_pct": ac.get("both_exact_pct", np.nan),
            "ticker_both_within_1c_pct": ac.get("both_within_1c_pct", np.nan),
            "repair_events": int(rc["repairs"]),
            "ticker_mismatch_repairs": int(rc["ticker_persistent_bbo_mismatch"]),
            "crossed_after_delta_repairs": int(rc["crossed_after_delta"]),
            "crossed_snapshot_repairs": int(rc["crossed_snapshot"]),
            "negative_level_repairs": int(rc["negative_level"]),
        })
    series_quality = pd.DataFrame(series_rows)

    complete = contracts[contracts.full_m0_m5_plus_30_boundary].copy()
    complete_targets = int(complete.trade_30s_targets.sum()) if len(complete) else 0
    complete_covered = int(complete.trade_30s_labels_covered.sum()) if len(complete) else 0

    overall_checked = int(agreement.checked.sum()) if len(agreement) else 0
    overall_exact = int(agreement.both_exact.sum()) if len(agreement) else 0
    overall_within1 = int(agreement.both_within_1c.sum()) if len(agreement) else 0

    btc = series_quality[series_quality.series == BTC_SERIES]
    btc_agree = agreement[agreement.series == BTC_SERIES]
    btc_within1 = float(btc_agree.both_within_1c_pct.iloc[0]) if len(btc_agree) else np.nan
    btc_repairs = int(btc.repair_events.iloc[0]) if len(btc) else 0
    btc_cross_repairs = int(
        (btc.crossed_after_delta_repairs.iloc[0] + btc.crossed_snapshot_repairs.iloc[0])
    ) if len(btc) else 0

    book_resolution = {
        "book_gap_count": book_gap_count,
        "book_gap_le_100ms_pct": _pct(book_sub100, book_gap_count),
        "book_gap_le_250ms_pct": _pct(book_sub250, book_gap_count),
        "book_gap_le_500ms_pct": _pct(book_sub500, book_gap_count),
        "book_gap_le_1s_pct": _pct(book_sub1s, book_gap_count),
        **{f"book_gap_{k}_s": v for k, v in _quantiles(book_gap_samples).items()},
    }
    repair_recovery_q = _quantiles(recovery_samples)
    ticker_age_q = _quantiles(ticker_age_samples)
    label_age_q = _quantiles(target_age_samples)

    final_counts = manifest.get("final_counts") or {}
    seq_gaps = int(final_counts.get("sequence_gaps", conn_counts["orderbook_sequence_gap"]) or 0)
    seq_missing = int(final_counts.get("sequence_numbers_missing", 0) or 0)
    connection_epochs = int(manifest.get("connection_epochs", conn_counts["connected"]) or 0)

    boundary_pct = _pct(complete.shape[0], len(contracts)) if len(contracts) else np.nan
    tail_label_pct = _pct(complete_covered, complete_targets)
    ticker_within1_pct = _pct(overall_within1, overall_checked)
    recovered_repairs = int(repairs_df.recovered.sum()) if len(repairs_df) else 0
    repair_recovery_pct = _pct(recovered_repairs, len(repairs_df)) if len(repairs_df) else 100.0

    gates = {
        "capture_version_correct": capture_version == EXPECTED_CAPTURE_VERSION,
        "single_connection_epoch": connection_epochs == 1,
        "zero_sequence_gaps": seq_gaps == 0 and seq_missing == 0,
        "zero_persisted_crossed_locked_rows": crossed_rows == 0,
        "zero_dynamic_crossed_locked_rows": crossed_dynamic_rows == 0,
        "full_boundary_rate_ge_90pct": bool(np.isfinite(boundary_pct) and boundary_pct >= 90.0),
        "complete_contract_trade_30s_label_coverage_ge_95pct": bool(np.isfinite(tail_label_pct) and tail_label_pct >= 95.0),
        "ticker_book_within_1c_ge_97pct": bool(np.isfinite(ticker_within1_pct) and ticker_within1_pct >= 97.0),
        "repair_snapshot_recovery_ge_95pct": bool(repair_recovery_pct >= 95.0),
        "btc_no_cross_repair": btc_cross_repairs == 0,
        "btc_ticker_within_1c_ge_97pct": bool(np.isfinite(btc_within1) and btc_within1 >= 97.0),
    }
    verdict = "PASS_FOR_DEVELOPMENT" if all(gates.values()) else "REVIEW_REQUIRED"

    file_sizes = {}
    total_bytes = 0
    for name, p in paths.items():
        if p.exists() and p.is_file():
            size = p.stat().st_size
            file_sizes[name] = size
            total_bytes += size

    summary = {
        "audit_version": STUDY_VERSION,
        "capture_version": capture_version,
        "research_stage": manifest.get("research_stage"),
        "session": session.name,
        "session_dir": str(session),
        "duration_hours": manifest.get("duration_hours"),
        "connection_epochs": connection_epochs,
        "contracts": int(len(contracts)),
        "complete_m0_m5_plus_30_contracts": int(len(complete)),
        "complete_boundary_pct": boundary_pct,
        "book_rows": int(sum(c["rows"] for c in book_by_contract.values())),
        "invalid_book_rows": int(invalid_rows),
        "invalid_dynamic_book_rows": int(invalid_dynamic_rows),
        "persisted_crossed_or_locked_rows": int(crossed_rows),
        "persisted_locked_rows": int(locked_rows),
        "dynamic_crossed_or_locked_rows": int(crossed_dynamic_rows),
        "ticker_rows": int(ticker_total),
        "ticker_valid_rows": int(ticker_valid),
        "ticker_book_checked": overall_checked,
        "ticker_book_both_exact_pct": _pct(overall_exact, overall_checked),
        "ticker_book_both_within_1c_pct": ticker_within1_pct,
        "trades": int(trade_info["total"]),
        "bad_trades": int(trade_info["bad"]),
        "missing_trade_side": int(trade_info["missing_side"]),
        "missing_trade_exchange_time": int(trade_info["missing_exchange"]),
        "complete_contract_trade_30s_targets": complete_targets,
        "complete_contract_trade_30s_labels_covered": complete_covered,
        "complete_contract_trade_30s_label_coverage_pct": tail_label_pct,
        "sequence_gaps": seq_gaps,
        "sequence_numbers_missing": seq_missing,
        "ok_seq_messages": int(final_counts.get("ok_seq_messages", 0) or 0),
        "snapshot_requests": int(final_counts.get("snapshot_requests", len(snapshot_requests)) or 0),
        "repair_events": int(len(repairs_df)),
        "repair_snapshot_recovery_pct": repair_recovery_pct,
        "repair_recovery_ms": repair_recovery_q,
        "ticker_book_age_ms": ticker_age_q,
        "future_30s_label_age_ms": label_age_q,
        "btc_ticker_book_within_1c_pct": btc_within1,
        "btc_repair_events": btc_repairs,
        "btc_cross_repair_events": btc_cross_repairs,
        "book_resolution": book_resolution,
        "trade_resolution": trade_info["resolution"],
        "repair_reason_counts": dict(repair_reason_counts),
        "connection_event_counts": dict(conn_counts),
        "file_size_gb": {k: v / (1024.0 ** 3) for k, v in file_sizes.items()},
        "total_file_size_gb": total_bytes / (1024.0 ** 3),
        "development_plan": development_plan,
        "quality_gates": gates,
        "verdict": verdict,
        "guardrail": "NO STRATEGY OR PNL WAS EVALUATED BY THIS AUDIT",
    }

    if output_dir is None:
        output_dir = C.PROJECT_ROOT / "results" / "kalshi_mm_event_m0_m5_v5_audit" / session.name
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    contracts.to_csv(out / "contract_event_time_quality.csv", index=False)
    series_quality.to_csv(out / "series_event_time_quality.csv", index=False)
    agreement.to_csv(out / "ticker_book_agreement_by_series.csv", index=False)
    repairs_df.to_csv(out / "repair_recovery.csv", index=False)
    pd.DataFrame([{"gate": k, "pass": v} for k, v in gates.items()]).to_csv(out / "quality_gates.csv", index=False)
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("\n" + "=" * 156)
        print("V5 EVENT-TIME DATA-QUALITY AUDIT — NO STRATEGY / NO PNL")
        print("=" * 156)
        print(f"session={session.name} | duration={_f(manifest.get('duration_hours')):.3f}h | connection_epochs={connection_epochs}")
        print(f"contracts={len(contracts)} | complete M0-M5+30={len(complete)} ({boundary_pct:.2f}%)")
        print(f"book rows={summary['book_rows']:,} | invalid={invalid_rows:,} | crossed/locked={crossed_rows:,} | dynamic crossed/locked={crossed_dynamic_rows:,}")
        print(f"sequence gaps={seq_gaps} | missing seq={seq_missing} | ok-seq messages={summary['ok_seq_messages']}")
        print(f"ticker/book checked={overall_checked:,} | exact={summary['ticker_book_both_exact_pct']:.2f}% | within1c={ticker_within1_pct:.2f}%")
        print(f"complete-contract 30s trade-label coverage={tail_label_pct:.2f}% ({complete_covered:,}/{complete_targets:,})")
        print(f"repairs={len(repairs_df)} | recovered by persisted snapshot={repair_recovery_pct:.2f}% | recovery p50/p95={repair_recovery_q['p50']:.2f}/{repair_recovery_q['p95']:.2f} ms")

        print("\nSERIES QUALITY")
        cols = [
            "series", "contracts", "full_tail_pct", "book_rows", "trade_rows",
            "trade_30s_label_coverage_pct", "ticker_both_within_1c_pct", "repair_events",
            "crossed_after_delta_repairs", "negative_level_repairs",
        ]
        print(series_quality[cols].round(3).to_string(index=False))

        print("\nBTC CHECK")
        if len(btc):
            print(btc.round(3).to_string(index=False))
            print(f"BTC ticker/book within1c: {btc_within1:.3f}% | BTC cross repairs: {btc_cross_repairs}")
        else:
            print("BTC series not present.")

        print("\nREPAIR REASONS")
        if repair_reason_counts:
            for k, v in repair_reason_counts.most_common():
                print(f"  {k:36s} {v}")
        else:
            print("  none")

        print("\nEVENT-TIME RESOLUTION")
        print(
            f"book gaps <=100/250/500/1000ms: "
            f"{book_resolution['book_gap_le_100ms_pct']:.2f}% / {book_resolution['book_gap_le_250ms_pct']:.2f}% / "
            f"{book_resolution['book_gap_le_500ms_pct']:.2f}% / {book_resolution['book_gap_le_1s_pct']:.2f}%"
        )
        tr = trade_info["resolution"]
        print(
            f"trade gaps <=100/250/500/1000ms: "
            f"{tr['trade_gap_le_100ms_pct']:.2f}% / {tr['trade_gap_le_250ms_pct']:.2f}% / "
            f"{tr['trade_gap_le_500ms_pct']:.2f}% / {tr['trade_gap_le_1s_pct']:.2f}%"
        )
        print(
            f"trade receipt latency p50/p95/p99: "
            f"{tr['receipt_latency_p50_ms']:.2f} / {tr['receipt_latency_p95_ms']:.2f} / {tr['receipt_latency_p99_ms']:.2f} ms"
        )

        print("\nQUALITY GATES")
        for k, v in gates.items():
            print(f"  {'PASS' if v else 'FAIL':4s}  {k}")
        print(f"\nVERDICT: {verdict}")
        print("NO A/B/C/D STRATEGY RESULTS OR PNL WERE READ.")
        print(f"OUTPUTS: {out}")
        print("=" * 156)

    return {
        "output_dir": out,
        "summary": summary,
        "contracts": contracts,
        "series_quality": series_quality,
        "ticker_agreement": agreement,
        "repairs": repairs_df,
        "quality_gates": pd.DataFrame([{"gate": k, "pass": v} for k, v in gates.items()]),
    }
