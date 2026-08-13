from __future__ import annotations

"""Offline book-quality audit for Kalshi 15-minute crypto M1-M5 research.

Purpose
-------
Explain why contracts fail the frozen M1-M5 data-quality gate before any
market-making regime research is attempted.

This module:
  * reads full_books.jsonl only (never trades.jsonl);
  * uses the same session-level book price-convention probe as
    mm_m1_m5_feasibility_v3;
  * audits raw snapshot availability separately from valid two-sided books;
  * preserves the existing M1-M5 quality gate: >=80% valid-book coverage and
    <=5s start/end edge gaps;
  * reports missing-side, locked, crossed, bad-price, and sampling-gap causes;
  * does NOT simulate fills and does NOT select or optimize any MM threshold.
"""

import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_m1_m5_feasibility as _v1
from . import mm_m1_m5_feasibility_v3 as _v3

STUDY_VERSION = "M1_M5_BOOK_QUALITY_DIAGNOSTIC_V1"
ROW_STATUSES = (
    "VALID",
    "MISSING_BOTH",
    "MISSING_BID",
    "MISSING_OTHER_SIDE",
    "LOCKED",
    "CROSSED",
    "BAD_PRICE",
)
EPS = 1e-9


def _pct(num, den):
    return 100.0 * num / den if den else np.nan


def _minute_bucket(minute):
    if not np.isfinite(minute):
        return "NA"
    lo = int(math.floor(minute))
    return f"M{lo}-M{lo+1}"


def _classify_row(row, orientation):
    """Return (status, bid, ask) under one already-selected session schema."""
    bids = sorted(_v1._levels(row.get("yes_bids") or []), key=lambda z: z[0], reverse=True)
    other = _v1._levels(row.get("yes_asks") or [])

    if not bids and not other:
        return "MISSING_BOTH", np.nan, np.nan
    if not bids:
        return "MISSING_BID", np.nan, np.nan
    if not other:
        return "MISSING_OTHER_SIDE", np.nan, np.nan

    if orientation == "UNIFIED_YES_PRICE":
        asks = sorted(other, key=lambda z: z[0])
    elif orientation == "LEGACY_NO_PRICE":
        asks = sorted([(1.0 - p, q) for p, q in other], key=lambda z: z[0])
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    if not asks:
        return "MISSING_OTHER_SIDE", np.nan, np.nan

    bid = float(bids[0][0])
    ask = float(asks[0][0])
    if not (np.isfinite(bid) and np.isfinite(ask) and 0.0 <= bid <= 1.0 and 0.0 <= ask <= 1.0):
        return "BAD_PRICE", bid, ask
    if abs(bid - ask) <= EPS:
        return "LOCKED", bid, ask
    if bid > ask:
        return "CROSSED", bid, ask
    return "VALID", bid, ask


def _new_contract(session_name, ticker, series, close_ts):
    out = {
        "session": session_name,
        "ticker": ticker,
        "series": series,
        "close_ts": float(close_ts),
        "raw_rows": 0,
        "valid_rows": 0,
        "valid_times": [],
    }
    for status in ROW_STATUSES:
        out[f"rows_{status.lower()}"] = 0
    return out


def _new_minute_state(session_name, ticker, series, bucket):
    out = {
        "session": session_name,
        "ticker": ticker,
        "series": series,
        "minute_bucket": bucket,
        "raw_rows": 0,
        "valid_rows": 0,
    }
    for status in ROW_STATUSES:
        out[f"rows_{status.lower()}"] = 0
    return out


def _sampling_stats(times):
    if not times:
        return np.nan, np.nan, np.nan
    arr = np.asarray(sorted(times), dtype=float)
    if len(arr) < 2:
        return np.nan, np.nan, np.nan
    gaps = np.diff(arr)
    return float(np.max(gaps)), float(np.median(gaps)), float(np.quantile(gaps, 0.95))


def _likely_cause(raw_cov, valid_cov, start_gap, end_gap, validity_pct, gate_pass):
    if gate_pass:
        return "PASS"
    if not np.isfinite(raw_cov) or raw_cov < 80.0:
        return "RECORDER_COVERAGE"
    if not np.isfinite(validity_pct) or validity_pct < 80.0:
        return "BOOK_VALIDITY"
    if ((np.isfinite(start_gap) and start_gap > 5.0) or
            (np.isfinite(end_gap) and end_gap > 5.0)):
        return "EDGE_GAP"
    if valid_cov < 80.0:
        return "MIXED_COVERAGE_VALIDITY"
    return "OTHER"


def _finalize_contract(state, start_minute, end_minute, min_book_coverage_pct, max_edge_gap_s):
    duration_s = (end_minute - start_minute) * 60.0
    expected_rows = max(1.0, duration_s)
    raw_rows = int(state["raw_rows"])
    valid_rows = int(state["valid_rows"])
    valid_times = sorted(state["valid_times"])
    close_ts = float(state["close_ts"])
    contract_start = close_ts - 900.0
    window_start = contract_start + start_minute * 60.0
    window_end = contract_start + end_minute * 60.0

    raw_cov = min(100.0, 100.0 * raw_rows / expected_rows)
    valid_cov = min(100.0, 100.0 * valid_rows / expected_rows)
    validity_pct = _pct(valid_rows, raw_rows)

    if valid_times:
        start_gap = max(0.0, valid_times[0] - window_start)
        end_gap = max(0.0, window_end - valid_times[-1])
    else:
        start_gap = np.nan
        end_gap = np.nan

    max_gap, median_gap, p95_gap = _sampling_stats(valid_times)

    reasons = []
    if valid_rows < 2:
        reasons.append("<2 valid book samples")
    if valid_cov < min_book_coverage_pct:
        reasons.append(f"valid coverage < {min_book_coverage_pct:.0f}%")
    if not np.isfinite(start_gap) or start_gap > max_edge_gap_s:
        reasons.append(f"start gap > {max_edge_gap_s:.0f}s")
    if not np.isfinite(end_gap) or end_gap > max_edge_gap_s:
        reasons.append(f"end gap > {max_edge_gap_s:.0f}s")
    gate_pass = not reasons

    invalid_counts = {
        status: int(state[f"rows_{status.lower()}"])
        for status in ROW_STATUSES if status != "VALID"
    }
    dominant_invalid = max(invalid_counts, key=invalid_counts.get) if invalid_counts else None
    if dominant_invalid and invalid_counts[dominant_invalid] == 0:
        dominant_invalid = "NONE"

    row = {
        "session": state["session"],
        "ticker": state["ticker"],
        "series": state["series"],
        "close_time": datetime.fromtimestamp(close_ts, tz=timezone.utc).isoformat(),
        "expected_rows_1hz": expected_rows,
        "raw_rows": raw_rows,
        "valid_rows": valid_rows,
        "raw_coverage_pct": raw_cov,
        "valid_coverage_pct": valid_cov,
        "validity_given_recorded_pct": validity_pct,
        "start_gap_s": start_gap,
        "end_gap_s": end_gap,
        "max_valid_sample_gap_s": max_gap,
        "median_valid_sample_gap_s": median_gap,
        "p95_valid_sample_gap_s": p95_gap,
        "quality_gate_pass": gate_pass,
        "quality_failure_reason": "OK" if gate_pass else "; ".join(reasons),
        "likely_cause": _likely_cause(raw_cov, valid_cov, start_gap, end_gap, validity_pct, gate_pass),
        "dominant_invalid_status": dominant_invalid,
    }
    for status in ROW_STATUSES:
        row[f"rows_{status.lower()}"] = int(state[f"rows_{status.lower()}"])
        row[f"pct_{status.lower()}_of_raw"] = _pct(state[f"rows_{status.lower()}"] , raw_rows)
    return row


def _finalize_contract_minute(state):
    expected = 60.0
    raw = int(state["raw_rows"])
    valid = int(state["valid_rows"])
    row = {
        "session": state["session"],
        "ticker": state["ticker"],
        "series": state["series"],
        "minute_bucket": state["minute_bucket"],
        "expected_rows_1hz": expected,
        "raw_rows": raw,
        "valid_rows": valid,
        "raw_coverage_pct": min(100.0, 100.0 * raw / expected),
        "valid_coverage_pct": min(100.0, 100.0 * valid / expected),
        "validity_given_recorded_pct": _pct(valid, raw),
    }
    for status in ROW_STATUSES:
        row[f"rows_{status.lower()}"] = int(state[f"rows_{status.lower()}"])
    return row


def _asset_summary(contract_df):
    if contract_df.empty:
        return pd.DataFrame()
    rows = []
    for series, g in contract_df.groupby("series", sort=True):
        row = {
            "series": series,
            "contracts": len(g),
            "quality_contracts": int(g["quality_gate_pass"].sum()),
            "quality_pct": 100.0 * g["quality_gate_pass"].mean(),
            "median_raw_coverage_pct": g["raw_coverage_pct"].median(),
            "median_valid_coverage_pct": g["valid_coverage_pct"].median(),
            "median_validity_given_recorded_pct": g["validity_given_recorded_pct"].median(),
            "median_max_valid_gap_s": g["max_valid_sample_gap_s"].median(),
            "p90_max_valid_gap_s": g["max_valid_sample_gap_s"].quantile(0.90),
        }
        raw_total = g["raw_rows"].sum()
        valid_total = g["valid_rows"].sum()
        row["row_valid_pct"] = _pct(valid_total, raw_total)
        for status in ROW_STATUSES:
            n = g[f"rows_{status.lower()}"].sum()
            row[f"rows_{status.lower()}"] = n
            row[f"pct_{status.lower()}_rows"] = _pct(n, raw_total)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["quality_contracts", "median_valid_coverage_pct"], ascending=False)


def _minute_summary(minute_df):
    if minute_df.empty:
        return pd.DataFrame()
    rows = []
    groupers = [("ALL", minute_df)]
    groupers.extend((series, g) for series, g in minute_df.groupby("series", sort=True))
    for series, d in groupers:
        for bucket, g in d.groupby("minute_bucket", sort=True):
            contracts = g["ticker"].nunique()
            raw = g["raw_rows"].sum()
            valid = g["valid_rows"].sum()
            expected = 60.0 * contracts
            row = {
                "series": series,
                "minute_bucket": bucket,
                "contracts": contracts,
                "raw_rows": raw,
                "valid_rows": valid,
                "raw_coverage_pct": min(100.0, 100.0 * raw / expected) if expected else np.nan,
                "valid_coverage_pct": min(100.0, 100.0 * valid / expected) if expected else np.nan,
                "validity_given_recorded_pct": _pct(valid, raw),
            }
            for status in ROW_STATUSES:
                n = g[f"rows_{status.lower()}"].sum()
                row[f"rows_{status.lower()}"] = n
                row[f"pct_{status.lower()}_rows"] = _pct(n, raw)
            rows.append(row)
    return pd.DataFrame(rows)


def _failure_summary(contract_df):
    if contract_df.empty:
        return pd.DataFrame()
    fail = contract_df[~contract_df["quality_gate_pass"]].copy()
    counts = Counter()
    for text in fail["quality_failure_reason"].fillna(""):
        for part in str(text).split("; "):
            if part:
                counts[part] += 1
    rows = [{"failure_reason": k, "contracts": v, "pct_failed_contracts": _pct(v, len(fail))}
            for k, v in counts.most_common()]
    return pd.DataFrame(rows)


def _cause_summary(contract_df):
    if contract_df.empty:
        return pd.DataFrame()
    x = contract_df.groupby("likely_cause", dropna=False).size().rename("contracts").reset_index()
    x["pct_contracts"] = 100.0 * x["contracts"] / len(contract_df)
    return x.sort_values("contracts", ascending=False)


def _print_report(result, show_rows=20):
    schema = result["schema_probe"]
    stats = result["scan_stats"].iloc[0]
    contract_df = result["contract_quality"]
    asset_df = result["asset_summary"]
    minute_df = result["minute_summary"]
    cause_df = result["cause_summary"]
    failure_df = result["failure_summary"]

    print("=" * 108)
    print("M1-M5 BOOK QUALITY DIAGNOSTIC — NO TRADING SIMULATION")
    print("=" * 108)
    print(
        f"Schema: {schema['orientation']} ({schema['reason']}) | "
        f"direct valid={schema['direct_valid_pct']:.2f}% | complement valid={schema['complement_valid_pct']:.2f}%"
    )
    print(
        f"Scanned lines={int(stats['full_book_lines_scanned']):,} | "
        f"crypto decoded={int(stats['crypto_rows_decoded']):,} | "
        f"M1-M5 rows={int(stats['window_rows']):,} | contracts={len(contract_df):,}"
    )
    if len(contract_df):
        print(
            f"Frozen 80%/5s gate passes: {int(contract_df['quality_gate_pass'].sum()):,}/{len(contract_df):,} "
            f"({100.0 * contract_df['quality_gate_pass'].mean():.2f}%)"
        )
        print(
            f"Median raw coverage={contract_df['raw_coverage_pct'].median():.2f}% | "
            f"median valid coverage={contract_df['valid_coverage_pct'].median():.2f}% | "
            f"median validity given recorded={contract_df['validity_given_recorded_pct'].median():.2f}%"
        )

    print("\nLIKELY ROOT CAUSE BY CONTRACT")
    print(cause_df.head(show_rows).to_string(index=False) if len(cause_df) else "<none>")

    print("\nFROZEN GATE FAILURE COMPONENTS")
    print(failure_df.head(show_rows).to_string(index=False) if len(failure_df) else "<none>")

    print("\nBY ASSET")
    cols = [
        "series", "contracts", "quality_contracts", "quality_pct",
        "median_raw_coverage_pct", "median_valid_coverage_pct",
        "median_validity_given_recorded_pct", "row_valid_pct",
        "pct_missing_bid_rows", "pct_missing_other_side_rows",
        "pct_locked_rows", "pct_crossed_rows",
    ]
    cols = [c for c in cols if c in asset_df.columns]
    print(asset_df[cols].round(2).to_string(index=False) if len(asset_df) else "<none>")

    print("\nM1-M5 MINUTE COVERAGE — ALL CRYPTO")
    z = minute_df[minute_df["series"] == "ALL"] if len(minute_df) else pd.DataFrame()
    cols = [
        "minute_bucket", "contracts", "raw_coverage_pct", "valid_coverage_pct",
        "validity_given_recorded_pct", "pct_missing_bid_rows",
        "pct_missing_other_side_rows", "pct_locked_rows", "pct_crossed_rows",
    ]
    cols = [c for c in cols if c in z.columns]
    print(z[cols].round(2).to_string(index=False) if len(z) else "<none>")

    print("\nINTERPRETATION")
    print("  RECORDER_COVERAGE       = raw 1 Hz snapshots themselves are sparse (<80%).")
    print("  BOOK_VALIDITY           = raw coverage exists, but too many recorded rows are not valid two-sided books.")
    print("  EDGE_GAP                = aggregate coverage is adequate, but the M1 or M5 boundary is missing by >5s.")
    print("  MIXED_COVERAGE_VALIDITY = neither issue alone dominates; both contribute.")
    print("No MM economics or regime should be inferred from this diagnostic.")
    print("\nOutputs:", result["output_dir"])
    print("=" * 108)


def run_m1_m5_book_quality_diagnostic(
    session_dir,
    output_dir=None,
    *,
    start_minute=1.0,
    end_minute=5.0,
    crypto_series=None,
    min_book_coverage_pct=80.0,
    max_edge_gap_s=5.0,
    schema_probe_rows=50000,
    show=True,
    show_rows=20,
):
    """Audit recorder book quality for the M1-M5 window without simulating trades."""
    session = Path(session_dir)
    path = session / "full_books.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    if not (0 <= start_minute < end_minute <= 15):
        raise ValueError("Require 0 <= start_minute < end_minute <= 15")

    series = set(crypto_series or _v1.CRYPTO_SERIES)
    schema = _v3._probe_session_orientation(session, series, max_relevant_rows=schema_probe_rows)
    orientation = schema["orientation"]
    prefix_re = _v1._prefix_regex(series)

    if output_dir is None:
        root = session.resolve().parents[2] if len(session.resolve().parents) >= 3 else Path.cwd()
        output_dir = root / "results" / "kalshi_mm_m1_m5_book_quality" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    contract_states = {}
    minute_states = {}
    scanned = decoded = window_rows = 0
    row_status_total = Counter()
    t0 = time.time()

    with path.open("rb") as f:
        for raw in f:
            scanned += 1
            if prefix_re is not None and prefix_re.search(raw) is None:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            decoded += 1

            ticker = _v1._get_ticker(row)
            if not ticker:
                continue
            s = ticker.split("-")[0]
            if s not in series:
                continue
            t = _v1._event_ts(row)
            close_ts = _v1._market_close_ts(row, ticker)
            if not (np.isfinite(t) and np.isfinite(close_ts)):
                continue

            minute = (t - (close_ts - 900.0)) / 60.0
            if not (start_minute <= minute < end_minute):
                continue
            window_rows += 1

            status, _, _ = _classify_row(row, orientation)
            row_status_total[status] += 1

            state = contract_states.get(ticker)
            if state is None:
                state = _new_contract(session.name, ticker, s, close_ts)
                contract_states[ticker] = state
            state["raw_rows"] += 1
            state[f"rows_{status.lower()}"] += 1
            if status == "VALID":
                state["valid_rows"] += 1
                state["valid_times"].append(float(t))

            bucket = _minute_bucket(minute)
            key = (ticker, bucket)
            ms = minute_states.get(key)
            if ms is None:
                ms = _new_minute_state(session.name, ticker, s, bucket)
                minute_states[key] = ms
            ms["raw_rows"] += 1
            ms[f"rows_{status.lower()}"] += 1
            if status == "VALID":
                ms["valid_rows"] += 1

    contract_rows = [
        _finalize_contract(x, start_minute, end_minute, min_book_coverage_pct, max_edge_gap_s)
        for x in contract_states.values()
    ]
    minute_rows = [_finalize_contract_minute(x) for x in minute_states.values()]

    contract_df = pd.DataFrame(contract_rows)
    if len(contract_df):
        contract_df = contract_df.sort_values(["close_time", "series", "ticker"]).reset_index(drop=True)
    contract_minute_df = pd.DataFrame(minute_rows)
    if len(contract_minute_df):
        contract_minute_df = contract_minute_df.sort_values(["series", "ticker", "minute_bucket"]).reset_index(drop=True)

    asset_df = _asset_summary(contract_df)
    minute_summary_df = _minute_summary(contract_minute_df)
    failure_df = _failure_summary(contract_df)
    cause_df = _cause_summary(contract_df)
    row_status_df = pd.DataFrame([
        {
            "status": status,
            "rows": int(row_status_total.get(status, 0)),
            "pct_window_rows": _pct(row_status_total.get(status, 0), window_rows),
        }
        for status in ROW_STATUSES
    ])
    schema_df = pd.DataFrame([schema])
    scan_stats_df = pd.DataFrame([{
        "session": session.name,
        "full_book_lines_scanned": scanned,
        "crypto_rows_decoded": decoded,
        "window_rows": window_rows,
        "contracts": len(contract_df),
        "quality_contracts": int(contract_df["quality_gate_pass"].sum()) if len(contract_df) else 0,
        "scan_seconds": time.time() - t0,
    }])

    config = {
        "study_version": STUDY_VERSION,
        "session": str(session.resolve()),
        "crypto_series": sorted(series),
        "start_minute": float(start_minute),
        "end_minute": float(end_minute),
        "min_book_coverage_pct": float(min_book_coverage_pct),
        "max_edge_gap_s": float(max_edge_gap_s),
        "schema_probe_rows": int(schema_probe_rows),
        "selected_orientation": orientation,
        "purpose": "book-quality diagnosis only; no trading simulation or threshold optimization",
    }
    (out / "diagnostic_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    contract_df.to_csv(out / "contract_quality.csv", index=False)
    contract_minute_df.to_csv(out / "contract_minute_quality.csv", index=False)
    asset_df.to_csv(out / "asset_quality_summary.csv", index=False)
    minute_summary_df.to_csv(out / "minute_quality_summary.csv", index=False)
    failure_df.to_csv(out / "failure_reason_summary.csv", index=False)
    cause_df.to_csv(out / "likely_cause_summary.csv", index=False)
    row_status_df.to_csv(out / "row_status_summary.csv", index=False)
    schema_df.to_csv(out / "schema_probe.csv", index=False)
    scan_stats_df.to_csv(out / "scan_stats.csv", index=False)

    result = {
        "output_dir": out,
        "schema_probe": schema,
        "scan_stats": scan_stats_df,
        "contract_quality": contract_df,
        "contract_minute_quality": contract_minute_df,
        "asset_summary": asset_df,
        "minute_summary": minute_summary_df,
        "failure_summary": failure_df,
        "cause_summary": cause_df,
        "row_status_summary": row_status_df,
    }
    if show:
        _print_report(result, show_rows=show_rows)
    return result


__all__ = ["STUDY_VERSION", "run_m1_m5_book_quality_diagnostic"]
