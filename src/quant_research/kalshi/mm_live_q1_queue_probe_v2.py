from __future__ import annotations

"""Safer wrapper for the one-contract live Kalshi queue/latency probe.

Fixes discovered production API behavior:
- passive GTC exit must NOT set reduce_only (Kalshi rejects reduce_only GTC)
- emergency/cleanup flatten remains reduce_only=True + immediate_or_cancel

Additional safety:
- remembers the selected probe ticker
- if any exception occurs after selection, checks that ticker's position and
  attempts a reduce-only IOC flatten before re-raising
- defaults queue target to 50 contracts to increase the chance that the order
  remains resting long enough for a queue-position observation

This module does not alter the frozen OOS recorder/shadow strategy.
"""

import numpy as np

from . import mm_live_q1_queue_probe_v1 as B

STUDY_VERSION = "MM_LIVE_Q1_QUEUE_PROBE_V2"
DEFAULT_TARGET_QUEUE = 50.0


def rescue_flatten_ticker(ticker: str, *, confirm_live: bool = False, show: bool = True):
    """Check one ticker and flatten at most the probe's Q1 exposure using reduce-only IOC.

    This sends a real production order when a nonzero position is present.
    """
    if confirm_live is not True:
        raise RuntimeError("Real cleanup order disabled. Re-run with confirm_live=True.")

    client = B.LiveClient()
    pos, row = B._get_position(client, ticker)
    if show:
        print(f"Position before cleanup: {pos:+.2f} contracts | {ticker}")
        if row is not None:
            print("Position row:", row)

    if abs(pos) <= 1e-9:
        if show:
            print("Already flat. No order sent.")
        return {"ticker": ticker, "position_before": pos, "position_after": pos, "orders": []}

    if abs(pos) > B.QTY + 1e-9:
        raise RuntimeError(
            f"Safety guard: {ticker} position is {pos}, larger than this Q1 probe is allowed to manage."
        )

    # Positive position_fp = long YES -> original entry direction was BID.
    # Negative position_fp = short YES / long NO -> original entry direction was ASK.
    entry_side = "bid" if pos > 0 else "ask"
    record = {}
    B._force_flatten(client, ticker, entry_side, abs(pos), record)

    final_pos = np.nan
    final_row = None
    for _ in range(10):
        final_pos, final_row = B._get_position(client, ticker)
        if abs(final_pos) <= 1e-9:
            break
        import time
        time.sleep(0.25)

    if show:
        print(f"Position after cleanup:  {final_pos:+.2f} contracts")
        print("Cleanup orders:", record.get("forced_flatten_orders") or [])

    if not np.isfinite(final_pos) or abs(final_pos) > 1e-9:
        raise RuntimeError(f"CRITICAL: position still not flat in {ticker}: {final_pos}")

    return {
        "ticker": ticker,
        "position_before": pos,
        "position_after": final_pos,
        "position_row_after": final_row,
        "orders": record.get("forced_flatten_orders") or [],
    }


def run_live_q1_queue_probe(
    *,
    confirm_live: bool = False,
    max_wait_s=B.MAX_DISCOVERY_WAIT_S,
    entry_wait_s=B.ENTRY_WAIT_S,
    exit_wait_s=B.EXIT_WAIT_S,
    target_queue: float = DEFAULT_TARGET_QUEUE,
    show: bool = True,
):
    """Run one production Q1 probe with corrected passive-exit semantics.

    Passive entry: GTC + post_only, reduce_only=False.
    Passive exit:  GTC + post_only, reduce_only=False, quantity exactly equals
                   the filled entry quantity.
    Cleanup:       IOC + reduce_only=True at current touch if needed.

    Any exception after market selection triggers a best-effort Q1 flatten of
    that selected ticker before the exception is re-raised.
    """
    if confirm_live is not True:
        raise RuntimeError("Real order submission disabled. Re-run with confirm_live=True only if you intend to send a real Q1 order.")

    target_queue = float(target_queue)
    if not (1.0 <= target_queue <= B.MAX_QUEUE_FOR_SELECTION):
        raise ValueError(f"target_queue must be within [1, {B.MAX_QUEUE_FOR_SELECTION}]")

    original_submit = B._submit_and_measure
    original_choose = B._choose_probe_market
    original_target = B.TARGET_QUEUE
    selected = {}

    def choose_capture(*args, **kwargs):
        out = original_choose(*args, **kwargs)
        selected.update(out)
        return out

    def submit_fixed(client, *, ticker, side, qty, price, predicted_queue, reduce_only, label):
        # Production discovery: Kalshi rejects reduce_only on resting GTC orders.
        # The passive EXIT count already equals the exact filled entry quantity,
        # so using reduce_only=False cannot overshoot that Q1 round trip.
        if str(label).upper() == "EXIT":
            reduce_only = False
        return original_submit(
            client,
            ticker=ticker,
            side=side,
            qty=qty,
            price=price,
            predicted_queue=predicted_queue,
            reduce_only=reduce_only,
            label=label,
        )

    B._choose_probe_market = choose_capture
    B._submit_and_measure = submit_fixed
    B.TARGET_QUEUE = target_queue

    try:
        if show:
            print(f"V2 safety wrapper | target resting queue ≈ {target_queue:.0f} contracts")
        result = B.run_live_q1_queue_probe(
            confirm_live=True,
            max_wait_s=max_wait_s,
            entry_wait_s=entry_wait_s,
            exit_wait_s=exit_wait_s,
            show=show,
        )
        if isinstance(result, dict):
            result["probe_wrapper_version"] = STUDY_VERSION
            result["target_queue"] = target_queue
        return result
    except Exception as exc:
        ticker = str(selected.get("ticker") or "")
        if ticker:
            print(f"\nPROBE ERROR: {exc!r}")
            print(f"Fail-safe: checking {ticker} for stranded Q1 inventory...")
            try:
                rescue_flatten_ticker(ticker, confirm_live=True, show=True)
            except Exception as cleanup_exc:
                print(f"CRITICAL CLEANUP ERROR: {cleanup_exc!r}")
                print(f"CHECK KALSHI POSITION MANUALLY NOW: {ticker}")
        raise
    finally:
        B._choose_probe_market = original_choose
        B._submit_and_measure = original_submit
        B.TARGET_QUEUE = original_target


__all__ = ["run_live_q1_queue_probe", "rescue_flatten_ticker"]
