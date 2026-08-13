from __future__ import annotations

"""Robust driver for the M1-M5 market-making feasibility replay.

This module intentionally DOES NOT change the V1 replay mechanics, queue model,
minute window, or data-quality gate. It only fixes empty-study handling and
adds diagnostics so a zero-quality/zero-episode session is reported rather
than crashing in the summary layer.
"""

from collections import Counter

import numpy as np
import pandas as pd

from . import mm_m1_m5_feasibility as _v1

STUDY_VERSION = "M1_M5_MM_FEASIBILITY_V2_DRIVER"

_ORIG_HEADLINE = _v1._headline
_ORIG_PRINT_REPORT = _v1._print_report


def _empty_side_frame(markouts):
    cols = [
        "side",
        "quote_episodes",
        "filled_episodes",
        "episode_fill_rate_pct",
        "fill_events",
        "fill_qty",
        "avg_queue_ahead",
        "avg_fill_latency_s",
        "avg_gross_edge_at_fill_c",
    ]
    for h in markouts:
        cols.extend(
            [
                f"avg_markout_{h}s_c",
                f"adverse_markout_{h}s_pct",
                f"avg_post_mid_move_{h}s_c",
                f"adverse_mid_move_{h}s_pct",
            ]
        )
    return pd.DataFrame(columns=cols)


def _safe_headline(contract_df, episodes_df, fills_df, side_df, markouts, sessions, config):
    if side_df is None or "side" not in side_df.columns:
        side_df = _empty_side_frame(markouts)
    return _ORIG_HEADLINE(
        contract_df,
        episodes_df,
        fills_df,
        side_df,
        markouts,
        sessions,
        config,
    )


def _safe_print_report(headline, side_summary, output_dir):
    if side_summary is not None and "side" in side_summary.columns:
        return _ORIG_PRINT_REPORT(headline, side_summary, output_dir)

    r = headline.iloc[0]
    print("=" * 96)
    print("M1-M5 TWO-SIDED MARKET-MAKING FEASIBILITY — EXPLORATORY / BEFORE FEES")
    print("=" * 96)
    print(f"Quality contracts: {int(r.get('quality_contracts', 0))}")
    print(f"Quote episodes:    {int(r.get('quote_episodes', 0))}")
    print(f"Fill events / qty: {int(r.get('fill_events', 0))} / {float(r.get('fill_qty', 0.0)):.2f}")
    print()
    print("No quote episodes survived into the replay. The V2 driver will print the")
    print("book-scan / quality-gate diagnostics below. No threshold has been changed.")
    print()
    print("Outputs:", output_dir)
    print("=" * 96)


def _reason_category(reason):
    text = str(reason or "")
    cats = []
    if "<2 quote-window book samples" in text:
        cats.append("<2 book samples")
    if "book coverage" in text:
        cats.append("coverage below gate")
    if "start gap" in text:
        cats.append("start-edge gap")
    if "end gap" in text:
        cats.append("end-edge gap")
    if not cats:
        cats.append(text if text else "unknown")
    return cats


def _print_diagnostics(result):
    episodes = result.get("quote_episodes")
    if isinstance(episodes, pd.DataFrame) and len(episodes):
        return

    print("\n" + "=" * 96)
    print("ZERO-EPISODE DIAGNOSTICS")
    print("=" * 96)

    stats = result.get("scan_stats")
    if isinstance(stats, pd.DataFrame) and len(stats):
        cols = [
            c
            for c in (
                "session",
                "full_book_lines_scanned",
                "crypto_book_lines_decoded",
                "book_samples_kept",
                "invalid_book_rows",
                "contracts_seen",
                "contracts_quality",
            )
            if c in stats.columns
        ]
        print("BOOK SCAN")
        print(stats[cols].to_string(index=False))

        seen = pd.to_numeric(stats.get("contracts_seen"), errors="coerce").fillna(0).sum()
        quality = pd.to_numeric(stats.get("contracts_quality"), errors="coerce").fillna(0).sum()
        invalid = pd.to_numeric(stats.get("invalid_book_rows"), errors="coerce").fillna(0).sum()

        print()
        if seen <= 0:
            print("DIAGNOSIS: zero contracts reached the M1-M5 book window.")
            print("This points to a timing/schema/universe parsing issue, not a market-making result.")
        elif quality <= 0:
            print("DIAGNOSIS: contracts were found, but none passed the frozen data-quality gate.")
            print("Do NOT relax the 80% / 5s gate yet; inspect the exclusion reasons below first.")
        else:
            print("DIAGNOSIS: quality contracts existed but no quote episodes were produced.")
            print("That indicates a replay/simulation-path bug and should be fixed before analysis.")

        if invalid > 0:
            print(f"Book rows rejected for invalid two-sided orientation: {int(invalid):,}")

    excluded = result.get("excluded_contracts")
    if isinstance(excluded, pd.DataFrame) and len(excluded):
        counter = Counter()
        if "quality_reason" in excluded.columns:
            for reason in excluded["quality_reason"]:
                for cat in _reason_category(reason):
                    counter[cat] += 1

        print("\nQUALITY-GATE EXCLUSIONS")
        print(f"Excluded contracts: {len(excluded)}")
        for name, n in counter.most_common():
            print(f"  {name:<28} {n:>6}")

        cols = [
            c
            for c in (
                "session",
                "ticker",
                "series",
                "book_coverage_pct",
                "start_gap_s",
                "end_gap_s",
                "quality_reason",
            )
            if c in excluded.columns
        ]
        if cols:
            print("\nFIRST 12 EXCLUDED CONTRACTS")
            print(excluded[cols].head(12).to_string(index=False))

    print("\nNo MM economics should be interpreted until quality contracts and quote episodes are nonzero.")
    print("=" * 96)


def run_m1_m5_mm_feasibility(*args, **kwargs):
    """Run the exact V1 replay with robust zero-study handling and diagnostics."""
    old_headline = _v1._headline
    old_print = _v1._print_report
    _v1._headline = _safe_headline
    _v1._print_report = _safe_print_report
    try:
        result = _v1.run_m1_m5_mm_feasibility(*args, **kwargs)
    finally:
        _v1._headline = old_headline
        _v1._print_report = old_print

    # Normalize an empty side summary so downstream notebook code can safely index it.
    side = result.get("side_summary")
    if not isinstance(side, pd.DataFrame) or "side" not in side.columns:
        markouts = kwargs.get("markout_seconds", _v1.DEFAULT_MARKOUT_SECONDS)
        markouts = tuple(sorted({int(x) for x in markouts if int(x) > 0}))
        result["side_summary"] = _empty_side_frame(markouts)

    if kwargs.get("show", True):
        _print_diagnostics(result)

    return result


__all__ = ["STUDY_VERSION", "run_m1_m5_mm_feasibility"]
