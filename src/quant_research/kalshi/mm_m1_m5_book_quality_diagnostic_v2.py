from __future__ import annotations

"""Corrected driver for M1-M5 book-quality diagnostics.

Adds explicit zero-coverage contract-minute rows before aggregation so a fully
missing minute counts as 0/60 expected 1 Hz samples rather than disappearing
from the minute denominator. The underlying row classification, schema probe,
and frozen 80%/5s contract quality gate are unchanged.
"""

import math

import pandas as pd

from . import mm_m1_m5_book_quality_diagnostic as _v1

STUDY_VERSION = "M1_M5_BOOK_QUALITY_DIAGNOSTIC_V2_ZERO_MINUTES"


def _expected_buckets(start_minute, end_minute):
    out = []
    lo0 = int(math.floor(start_minute))
    lo1 = int(math.ceil(end_minute))
    for lo in range(lo0, lo1):
        a, b = float(lo), float(lo + 1)
        if max(a, start_minute) < min(b, end_minute):
            out.append(f"M{lo}-M{lo+1}")
    return out


def _zero_minute_row(session, ticker, series, bucket):
    row = {
        "session": session,
        "ticker": ticker,
        "series": series,
        "minute_bucket": bucket,
        "expected_rows_1hz": 60.0,
        "raw_rows": 0,
        "valid_rows": 0,
        "raw_coverage_pct": 0.0,
        "valid_coverage_pct": 0.0,
        "validity_given_recorded_pct": float("nan"),
    }
    for status in _v1.ROW_STATUSES:
        row[f"rows_{status.lower()}"] = 0
    return row


def run_m1_m5_book_quality_diagnostic(*args, **kwargs):
    """Run V1 audit, then make the minute denominator include missing minutes."""
    requested_show = kwargs.get("show", True)
    show_rows = kwargs.get("show_rows", 20)
    start_minute = float(kwargs.get("start_minute", 1.0))
    end_minute = float(kwargs.get("end_minute", 5.0))

    call_kwargs = dict(kwargs)
    call_kwargs["show"] = False
    result = _v1.run_m1_m5_book_quality_diagnostic(*args, **call_kwargs)

    contracts = result["contract_quality"]
    cm = result["contract_minute_quality"].copy()
    buckets = _expected_buckets(start_minute, end_minute)

    if len(contracts) and buckets:
        existing = set()
        if len(cm):
            existing = set(zip(cm["session"], cm["ticker"], cm["minute_bucket"]))

        missing = []
        for _, c in contracts.iterrows():
            for bucket in buckets:
                key = (c["session"], c["ticker"], bucket)
                if key not in existing:
                    missing.append(_zero_minute_row(c["session"], c["ticker"], c["series"], bucket))

        if missing:
            cm = pd.concat([cm, pd.DataFrame(missing)], ignore_index=True, sort=False)
        cm = cm.sort_values(["series", "ticker", "minute_bucket"]).reset_index(drop=True)

    minute_summary = _v1._minute_summary(cm)
    result["contract_minute_quality"] = cm
    result["minute_summary"] = minute_summary

    out = result["output_dir"]
    cm.to_csv(out / "contract_minute_quality.csv", index=False)
    minute_summary.to_csv(out / "minute_quality_summary.csv", index=False)

    if requested_show:
        print(
            f"Minute denominator correction: every observed M1-M5 contract contributes "
            f"one expected 60-s row block to each of {len(buckets)} minute buckets."
        )
        _v1._print_report(result, show_rows=show_rows)

    return result


__all__ = ["STUDY_VERSION", "run_m1_m5_book_quality_diagnostic"]
