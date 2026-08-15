from __future__ import annotations

"""Data-quality audit for MM_EVENT_TIME_M0_M5_V4 sessions.

NO STRATEGY AND NO PNL.

This audit answers whether a frozen V4 event-time session is scientifically
usable for first-five-minute microstructure research. It checks:
- M0/M5 boundary capture by contract and series;
- orderbook sequence gaps and snapshot-refresh recovery;
- validity/crossing of persisted top-3 BBO states;
- event-time resolution of top-3 changes and public trades;
- trade-side and exchange-timestamp completeness;
- exchange-to-receipt latency;
- ticker BBO agreement with the reconstructed event-time top-of-book, while
  explicitly invalidating comparisons across detected orderbook sequence gaps;
- file sizes and session/connection health.

The module streams the large JSONL files and uses fixed-size reservoirs for
quantiles so multi-million-row recordings do not need to fit in RAM.
"""

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C

STUDY_VERSION = "MM_EVENT_TIME_M0_M5_AUDIT_V1"
EXPECTED_CAPTURE_VERSION = "MM_EVENT_TIME_M0_M5_V4"
MAX_RESERVOIR = 200_000
MAX_BOOK_AGE_FOR_TICKER_CHECK_S = 2.0
RNG_SEED = 20260815


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
    return 100.0 * num / den if den else np.nan


def _file_mb(path: Path):
    return path.stat().st_size / (1024.0 ** 2) if path.exists() else 0.0


def _reservoir_add(arr, x, seen, rng):
    if not np.isfinite(x):
        return
    if len(arr) < MAX_RESERVOIR:
        arr.append(float(x))
    else:
        j = rng.randint(0, seen - 1)
        if j < MAX_RESERVOIR:
            arr[j] = float(x)


def _quantiles(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {k: np.nan for k in ("p10", "p25", "p50", "p75", "p90", "p95", "p99")}
    q = np.quantile(a, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return dict(zip(("p10", "p25", "p50", "p75", "p90", "p95", "p99"), map(float, q)))


def _read_connection_events(path: Path):
    counts = Counter()
    gaps = []
    if not path.exists():
        return counts, gaps
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                x = json.loads(line)
            except Exception:
                continue
            typ = str(x.get("type") or "")
            counts[typ] += 1
            if typ == "orderbook_sequence_gap":
                t = _parse_ts(x.get("time"))
                if t is not None:
                    gaps.append(float(t))
    gaps.sort()
    return counts, gaps


def _read_metadata(path: Path):
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                rows.append({
                    "ticker": x.get("ticker"),
                    "series": x.get("series_ticker"),
                    "close_time": x.get("close_time"),
                    "discovered_status": x.get("discovered_status"),
                })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="first").reset_index(drop=True)
    return df


def _read_ticker_events(path: Path):
    events = []
    rows_by_contract = Counter()
    total_rows = valid_rows = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                total_rows += 1
                t = _parse_ts(x.get("receipt_time"))
                ticker = x.get("ticker")
                if t is None or not ticker:
                    continue
                rows_by_contract[ticker] += 1
                bid, ask = _f(x.get("yes_bid")), _f(x.get("yes_ask"))
                if np.isfinite(bid) and np.isfinite(ask) and 0 <= bid < ask <= 1:
                    valid_rows += 1
                    events.append((float(t), str(ticker), float(bid), float(ask)))
    events.sort(key=lambda z: z[0])
    return events, rows_by_contract, total_rows, valid_rows


def run_event_time_m0_m5_audit(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if not session.exists():
        raise FileNotFoundError(session)

    book_path = session / "book_top3_events.jsonl"
    trade_path = session / "trades_event_time.jsonl"
    ticker_path = session / "ticker_event_time.jsonl"
    meta_path = session / "market_metadata.jsonl"
    conn_path = session / "connection_events.jsonl"
    manifest_path = session / "session_manifest.json"
    health_path = session / "health.json"

    for p in (book_path, trade_path, ticker_path, meta_path, conn_path):
        if not p.exists():
            raise FileNotFoundError(p)

    manifest = _load_json(manifest_path, {}) or {}
    health = _load_json(health_path, {}) or {}
    capture_spec = manifest.get("capture_spec") or _load_json(session / "capture_spec.json", {}) or {}
    capture_version = manifest.get("study_version") or capture_spec.get("study_version")
    if capture_version and capture_version != EXPECTED_CAPTURE_VERSION:
        raise RuntimeError(f"Expected {EXPECTED_CAPTURE_VERSION}, found {capture_version}")

    rng = random.Random(RNG_SEED)
    connection_counts, gap_times = _read_connection_events(conn_path)
    meta = _read_metadata(meta_path)
    ticker_to_series = dict(zip(meta.ticker, meta.series)) if len(meta) else {}
    ticker_events, ticker_rows_by_contract, ticker_total_rows, ticker_valid_rows = _read_ticker_events(ticker_path)

    # Book stream state.
    book_counts = Counter()
    book_by_contract = defaultdict(Counter)
    capture_start_elapsed = {}
    capture_end_elapsed = {}
    first_book_elapsed = {}
    last_book_elapsed = {}
    last_book_receipt = {}
    last_book_state = {}
    last_snapshot_time = defaultdict(lambda: -np.inf)

    book_gap_samples = []
    book_gap_seen = book_gap_count = 0
    book_sub100 = book_sub250 = book_sub500 = book_sub1s = 0
    invalid_bbo_rows = crossed_bbo_rows = 0

    # Gap recovery: V4 invalidates every book on each global orderbook sequence gap.
    latest_gap_time = -np.inf
    gap_i = 0
    recovery_samples_ms = []
    recovery_seen = 0
    snapshot_recoveries = 0

    # Ticker/book validation.
    ticker_i = 0
    ticker_checked = ticker_exact = ticker_within_1c = 0
    ticker_skipped_gap_invalid = 0
    ticker_age_samples_ms = []
    ticker_age_seen = 0

    def advance_gaps(to_t):
        nonlocal gap_i, latest_gap_time
        while gap_i < len(gap_times) and gap_times[gap_i] <= to_t:
            latest_gap_time = gap_times[gap_i]
            gap_i += 1

    def compare_ticker_until(book_event_time):
        nonlocal ticker_i, ticker_checked, ticker_exact, ticker_within_1c
        nonlocal ticker_skipped_gap_invalid, ticker_age_seen
        while ticker_i < len(ticker_events):
            tt, ticker, tbid, task = ticker_events[ticker_i]
            if tt > book_event_time:
                break
            advance_gaps(tt)
            state = last_book_state.get(ticker)
            if state is not None:
                bt, bbid, bask = state
                # If a global gap happened after this ticker's last snapshot, the
                # reconstructed state is intentionally invalid until refresh.
                if latest_gap_time > last_snapshot_time[ticker]:
                    ticker_skipped_gap_invalid += 1
                else:
                    age = tt - bt
                    if 0 <= age <= MAX_BOOK_AGE_FOR_TICKER_CHECK_S:
                        ticker_checked += 1
                        ticker_age_seen += 1
                        _reservoir_add(ticker_age_samples_ms, age * 1000.0, ticker_age_seen, rng)
                        be, ae = abs(tbid - bbid) * 100.0, abs(task - bask) * 100.0
                        if be < 1e-7 and ae < 1e-7:
                            ticker_exact += 1
                        if be <= 1.0 + 1e-9 and ae <= 1.0 + 1e-9:
                            ticker_within_1c += 1
            ticker_i += 1

    with book_path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            try:
                x = json.loads(line)
            except Exception:
                book_counts["json_errors"] += 1
                continue
            receipt = _parse_ts(x.get("receipt_time"))
            ticker = x.get("ticker")
            typ = str(x.get("event_type") or "")
            if receipt is None or not ticker:
                continue
            receipt, ticker = float(receipt), str(ticker)
            compare_ticker_until(receipt)
            advance_gaps(receipt)

            elapsed = _f(x.get("elapsed_s"))
            book_counts[typ] += 1
            book_by_contract[ticker][typ] += 1
            book_by_contract[ticker]["rows"] += 1
            if np.isfinite(elapsed):
                first_book_elapsed[ticker] = min(first_book_elapsed.get(ticker, np.inf), elapsed)
                last_book_elapsed[ticker] = max(last_book_elapsed.get(ticker, -np.inf), elapsed)
                if typ == "capture_start":
                    capture_start_elapsed.setdefault(ticker, elapsed)
                elif typ == "capture_end":
                    capture_end_elapsed[ticker] = elapsed

            bid, ask = _f(x.get("yes_bid")), _f(x.get("yes_ask"))
            valid = bool(x.get("valid_bbo"))
            if not valid:
                invalid_bbo_rows += 1
            if np.isfinite(bid) and np.isfinite(ask) and bid >= ask:
                crossed_bbo_rows += 1

            if typ == "book_snapshot":
                # First snapshot after a gap is this ticker's recovery point.
                if latest_gap_time > last_snapshot_time[ticker] and np.isfinite(latest_gap_time):
                    rec_ms = (receipt - latest_gap_time) * 1000.0
                    if rec_ms >= 0:
                        recovery_seen += 1
                        _reservoir_add(recovery_samples_ms, rec_ms, recovery_seen, rng)
                        snapshot_recoveries += 1
                last_snapshot_time[ticker] = receipt

            if valid and np.isfinite(bid) and np.isfinite(ask) and 0 <= bid < ask <= 1:
                last_book_state[ticker] = (receipt, float(bid), float(ask))

            prev = last_book_receipt.get(ticker)
            if prev is not None:
                d = receipt - prev
                if d >= 0:
                    book_gap_count += 1
                    book_gap_seen += 1
                    _reservoir_add(book_gap_samples, d, book_gap_seen, rng)
                    book_sub100 += d <= 0.100
                    book_sub250 += d <= 0.250
                    book_sub500 += d <= 0.500
                    book_sub1s += d <= 1.000
            last_book_receipt[ticker] = receipt
            if show and n % 1_000_000 == 0:
                print(f"  streamed {n:,} book rows...")

    compare_ticker_until(float("inf"))

    # Trade stream.
    trade_rows = bad_trade_rows = missing_exchange_ts = missing_side = 0
    trade_rows_by_contract = Counter()
    last_trade_receipt = {}
    trade_gap_samples = []
    trade_latency_samples = []
    trade_gap_seen = trade_latency_seen = trade_gap_count = 0
    trade_sub100 = trade_sub250 = trade_sub500 = trade_sub1s = 0

    with trade_path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            try:
                x = json.loads(line)
            except Exception:
                bad_trade_rows += 1
                continue
            trade_rows += 1
            ticker = x.get("ticker")
            receipt = _parse_ts(x.get("receipt_time"))
            exchange = _parse_ts(x.get("exchange_time"))
            price, qty = _f(x.get("yes_price")), _f(x.get("qty"))
            side = str(x.get("taker_book_side") or "").lower()
            if not ticker or receipt is None or not np.isfinite(price) or not (0 <= price <= 1) or not np.isfinite(qty) or qty <= 0:
                bad_trade_rows += 1
                continue
            ticker, receipt = str(ticker), float(receipt)
            trade_rows_by_contract[ticker] += 1
            if side not in {"bid", "ask"}:
                missing_side += 1
            if exchange is None:
                missing_exchange_ts += 1
            else:
                lat = (receipt - float(exchange)) * 1000.0
                if -5000 <= lat <= 60000:
                    trade_latency_seen += 1
                    _reservoir_add(trade_latency_samples, lat, trade_latency_seen, rng)
            prev = last_trade_receipt.get(ticker)
            if prev is not None:
                d = receipt - prev
                if d >= 0:
                    trade_gap_count += 1
                    trade_gap_seen += 1
                    _reservoir_add(trade_gap_samples, d, trade_gap_seen, rng)
                    trade_sub100 += d <= 0.100
                    trade_sub250 += d <= 0.250
                    trade_sub500 += d <= 0.500
                    trade_sub1s += d <= 1.000
            last_trade_receipt[ticker] = receipt
            if show and n % 1_000_000 == 0:
                print(f"  streamed {n:,} trade rows...")

    # Contract and series tables.
    all_tickers = sorted(set(ticker_to_series) | set(book_by_contract) | set(trade_rows_by_contract) | set(ticker_rows_by_contract))
    rows = []
    for ticker in all_tickers:
        s, e = capture_start_elapsed.get(ticker, np.nan), capture_end_elapsed.get(ticker, np.nan)
        full_start = np.isfinite(s) and -0.5 <= s <= 2.0
        full_end = np.isfinite(e) and 299.5 <= e <= 302.0
        c = book_by_contract[ticker]
        rows.append({
            "ticker": ticker,
            "series": ticker_to_series.get(ticker),
            "capture_start_elapsed_s": s,
            "capture_end_elapsed_s": e,
            "full_boundary_capture": bool(full_start and full_end),
            "book_rows": c["rows"],
            "book_snapshots": c["book_snapshot"],
            "book_deltas_top3": c["book_delta"],
            "ticker_rows": ticker_rows_by_contract[ticker],
            "trade_rows": trade_rows_by_contract[ticker],
            "first_book_elapsed_s": first_book_elapsed.get(ticker, np.nan),
            "last_book_elapsed_s": last_book_elapsed.get(ticker, np.nan),
        })
    contracts = pd.DataFrame(rows)
    if len(contracts):
        contracts = contracts.sort_values(["series", "ticker"]).reset_index(drop=True)
        series = contracts.groupby("series", dropna=False).agg(
            contracts=("ticker", "count"),
            full_boundary_pct=("full_boundary_capture", lambda x: 100.0 * x.mean()),
            median_book_rows=("book_rows", "median"),
            median_top3_deltas=("book_deltas_top3", "median"),
            median_trade_rows=("trade_rows", "median"),
            median_ticker_rows=("ticker_rows", "median"),
        ).reset_index()
    else:
        series = pd.DataFrame()

    # Session duration fallback if shutdown happened before final manifest rewrite.
    duration_h = manifest.get("duration_hours")
    if duration_h is None:
        start = _parse_ts(manifest.get("started_at"))
        last_health = _parse_ts(health.get("time"))
        if start is not None and last_health is not None and last_health >= start:
            duration_h = (last_health - start) / 3600.0

    total_contracts = len(contracts)
    completed = int(contracts.full_boundary_capture.sum()) if len(contracts) else 0
    seq_gaps = int(connection_counts.get("orderbook_sequence_gap", 0))
    conn_ex = int(connection_counts.get("connection_exception", 0))
    final_counts = manifest.get("final_counts") or health.get("final_counts") or {}
    deltas_received = int(final_counts.get("deltas_received", 0) or 0)
    snapshot_requests = int(final_counts.get("snapshot_requests", 0) or 0)
    gap_rate_per_million = 1e6 * seq_gaps / deltas_received if deltas_received else np.nan

    bq, tq = _quantiles(book_gap_samples), _quantiles(trade_gap_samples)
    lq, aq = _quantiles(trade_latency_samples), _quantiles(ticker_age_samples_ms)
    rq = _quantiles(recovery_samples_ms)

    # Recovery assessment: requests should track gaps, and observed snapshot
    # recovery should generally be quick. We do not require zero gaps.
    request_coverage_pct = _pct(snapshot_requests, seq_gaps) if seq_gaps else 100.0

    summary = pd.DataFrame([{
        "study_version": STUDY_VERSION,
        "capture_version": capture_version,
        "session": session.name,
        "duration_hours": duration_h,
        "connection_epochs": manifest.get("connection_epochs"),
        "connection_exceptions": conn_ex,
        "deltas_received": deltas_received,
        "orderbook_sequence_gaps": seq_gaps,
        "gap_rate_per_million_deltas": gap_rate_per_million,
        "snapshot_requests": snapshot_requests,
        "snapshot_request_per_gap_pct": request_coverage_pct,
        "observed_snapshot_recoveries": snapshot_recoveries,
        "recovery_p50_ms": rq["p50"],
        "recovery_p95_ms": rq["p95"],
        "contracts": total_contracts,
        "full_m0_m5_boundary_contracts": completed,
        "full_m0_m5_boundary_pct": _pct(completed, total_contracts),
        "book_rows": int(sum(v["rows"] for v in book_by_contract.values())),
        "book_snapshot_rows_in_capture": int(book_counts["book_snapshot"]),
        "book_top3_delta_rows": int(book_counts["book_delta"]),
        "invalid_book_rows": invalid_bbo_rows,
        "crossed_book_rows": crossed_bbo_rows,
        "ticker_rows": ticker_total_rows,
        "ticker_valid_rows": ticker_valid_rows,
        "ticker_book_comparisons": ticker_checked,
        "ticker_skipped_gap_invalid": ticker_skipped_gap_invalid,
        "ticker_book_exact_pct": _pct(ticker_exact, ticker_checked),
        "ticker_book_within_1c_pct": _pct(ticker_within_1c, ticker_checked),
        "trade_rows": trade_rows,
        "bad_trade_rows": bad_trade_rows,
        "trade_side_missing_rows": missing_side,
        "trade_exchange_ts_missing_rows": missing_exchange_ts,
        "book_interarrival_le100ms_pct": _pct(book_sub100, book_gap_count),
        "book_interarrival_le250ms_pct": _pct(book_sub250, book_gap_count),
        "book_interarrival_le500ms_pct": _pct(book_sub500, book_gap_count),
        "book_interarrival_le1s_pct": _pct(book_sub1s, book_gap_count),
        "trade_interarrival_le100ms_pct": _pct(trade_sub100, trade_gap_count),
        "trade_interarrival_le250ms_pct": _pct(trade_sub250, trade_gap_count),
        "trade_interarrival_le500ms_pct": _pct(trade_sub500, trade_gap_count),
        "trade_interarrival_le1s_pct": _pct(trade_sub1s, trade_gap_count),
        "book_gap_p50_s": bq["p50"], "book_gap_p95_s": bq["p95"], "book_gap_p99_s": bq["p99"],
        "trade_gap_p50_s": tq["p50"], "trade_gap_p95_s": tq["p95"],
        "trade_latency_p50_ms": lq["p50"], "trade_latency_p95_ms": lq["p95"], "trade_latency_p99_ms": lq["p99"],
        "ticker_book_age_p50_ms": aq["p50"], "ticker_book_age_p95_ms": aq["p95"],
    }])

    if output_dir is None:
        output_dir = C.PROJECT_ROOT / "results" / "kalshi_mm_event_m0_m5_v4_audit" / session.name
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "audit_summary.csv", index=False)
    contracts.to_csv(out / "contract_event_time_quality.csv", index=False)
    series.to_csv(out / "series_event_time_quality.csv", index=False)
    (out / "audit_config.json").write_text(json.dumps({
        "study_version": STUDY_VERSION,
        "session": str(session),
        "strategy_or_pnl_evaluated": False,
        "max_book_age_for_ticker_check_s": MAX_BOOK_AGE_FOR_TICKER_CHECK_S,
        "quantile_reservoir_max": MAX_RESERVOIR,
        "gap_semantics": "global orderbook book-state invalidation until per-ticker snapshot refresh",
    }, indent=2), encoding="utf-8")

    # Practical verdict. Zero gaps are not required: what matters is low gap
    # incidence plus refresh requests/recovery and healthy independent BBO checks.
    issues, warnings = [], []
    complete_pct = _pct(completed, total_contracts)
    if conn_ex > 0:
        warnings.append(f"{conn_ex} connection exceptions")
    if total_contracts and complete_pct < 85.0:
        issues.append(f"only {complete_pct:.1f}% observed contracts have full M0-M5 boundaries")
    if crossed_bbo_rows > 0:
        issues.append(f"{crossed_bbo_rows} crossed persisted BBO rows")
    if seq_gaps:
        if request_coverage_pct < 99.0:
            issues.append(f"snapshot requests cover only {request_coverage_pct:.1f}% of sequence gaps")
        else:
            warnings.append(f"{seq_gaps} sequence gaps ({gap_rate_per_million:.1f}/million deltas), refresh requested for each")
        if np.isfinite(rq["p95"]) and rq["p95"] > 5000.0:
            warnings.append(f"snapshot recovery p95 is {rq['p95']:.0f} ms")
    if ticker_checked >= 100 and _pct(ticker_within_1c, ticker_checked) < 95.0:
        issues.append(f"ticker/book <=1c agreement only {_pct(ticker_within_1c, ticker_checked):.1f}%")
    if missing_side > 0:
        issues.append(f"{missing_side} trades missing taker side")
    if trade_rows and missing_exchange_ts / trade_rows > 0.01:
        warnings.append(f"{_pct(missing_exchange_ts, trade_rows):.2f}% trades missing exchange timestamp")

    verdict = "PASS" if not issues else "REVIEW_REQUIRED"

    if show:
        print("\n" + "=" * 118)
        print("V4 EVENT-TIME M0-M5 DATA-QUALITY AUDIT — NO STRATEGY / NO PNL")
        print("=" * 118)
        print(f"Session: {session}")
        print(f"Outputs: {out}")
        print("\nFILE SIZES")
        for p in (book_path, trade_path, ticker_path, meta_path, conn_path):
            print(f"{p.name:30s} {_file_mb(p):10.2f} MB")
        print("\nSESSION / CONNECTION")
        print(f"Duration:                    {duration_h} h")
        print(f"Connection epochs:           {manifest.get('connection_epochs')}")
        print(f"Connection exceptions:       {conn_ex}")
        print(f"Deltas received:             {deltas_received:,}")
        print(f"Sequence gaps:               {seq_gaps:,} ({gap_rate_per_million:.2f}/million deltas)")
        print(f"Snapshot requests:           {snapshot_requests:,} ({request_coverage_pct:.2f}% of gaps)")
        print(f"Snapshot recovery p50/p95:   {rq['p50']:.2f} / {rq['p95']:.2f} ms")
        print("\nM0-M5 CONTRACT BOUNDARIES")
        print(f"Contracts observed:          {total_contracts}")
        print(f"Full M0-M5 boundaries:       {completed}/{total_contracts} ({complete_pct:.2f}%)")
        print("\nBOOK EVENT INTEGRITY")
        print(f"Persisted book rows:         {int(summary.iloc[0].book_rows):,}")
        print(f"Snapshots in capture:        {book_counts['book_snapshot']:,}")
        print(f"Top-3 changing deltas:       {book_counts['book_delta']:,}")
        print(f"Invalid BBO rows:            {invalid_bbo_rows:,}")
        print(f"Crossed BBO rows:            {crossed_bbo_rows:,}")
        print("\nBOOK EVENT-TIME RESOLUTION")
        print(f"<=100ms / <=250ms / <=500ms / <=1s: {_pct(book_sub100, book_gap_count):.2f}% / {_pct(book_sub250, book_gap_count):.2f}% / {_pct(book_sub500, book_gap_count):.2f}% / {_pct(book_sub1s, book_gap_count):.2f}%")
        print(f"gap p50/p95/p99:             {bq['p50']:.4f}s / {bq['p95']:.4f}s / {bq['p99']:.4f}s")
        print("\nTRADE EVENT-TIME RESOLUTION")
        print(f"Trade rows:                  {trade_rows:,}")
        print(f"Bad rows / missing side:     {bad_trade_rows:,} / {missing_side:,}")
        print(f"Missing exchange timestamp:  {missing_exchange_ts:,}")
        print(f"<=100ms / <=250ms / <=500ms / <=1s: {_pct(trade_sub100, trade_gap_count):.2f}% / {_pct(trade_sub250, trade_gap_count):.2f}% / {_pct(trade_sub500, trade_gap_count):.2f}% / {_pct(trade_sub1s, trade_gap_count):.2f}%")
        print(f"gap p50/p95:                 {tq['p50']:.4f}s / {tq['p95']:.4f}s")
        print(f"exchange->receipt p50/p95/p99: {lq['p50']:.2f} / {lq['p95']:.2f} / {lq['p99']:.2f} ms")
        print("\nTICKER vs EVENT-TIME BOOK")
        print(f"Valid comparisons:           {ticker_checked:,}")
        print(f"Skipped during gap-invalid:  {ticker_skipped_gap_invalid:,}")
        print(f"Both exact:                  {_pct(ticker_exact, ticker_checked):.2f}%")
        print(f"Both within 1c:              {_pct(ticker_within_1c, ticker_checked):.2f}%")
        print(f"Book age p50/p95:            {aq['p50']:.2f} / {aq['p95']:.2f} ms")
        print("\nBY SERIES")
        if len(series):
            print(series.round(2).to_string(index=False))
        print("\nINCOMPLETE BOUNDARY CONTRACTS")
        if len(contracts):
            bad = contracts[~contracts.full_boundary_capture]
            if len(bad):
                print(bad[["ticker", "series", "capture_start_elapsed_s", "capture_end_elapsed_s", "book_rows", "trade_rows"]].head(50).round(3).to_string(index=False))
            else:
                print("None")
        print("\nAUTOMATIC DATA-QUALITY VERDICT")
        print(verdict)
        for x in issues:
            print("  ISSUE:", x)
        for x in warnings:
            print("  WARNING:", x)
        print("\nNO STRATEGY OR PNL WAS EVALUATED.")
        print("=" * 118)

    return {
        "output_dir": out,
        "summary": summary,
        "contracts": contracts,
        "series": series,
        "verdict": verdict,
        "issues": issues,
        "warnings": warnings,
    }
