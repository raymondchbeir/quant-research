from __future__ import annotations

"""V3 live Q1 queue/latency probe wrapper.

Adds a robust queue-position read on top of V2:
- first try GET /portfolio/orders/{order_id}/queue_position
- on 404/other failure, immediately fall back to
  GET /portfolio/orders/queue_positions, which Kalshi documents as returning
  queue positions for all resting orders
- retry the batch surface briefly because the order and queue-position read
  can race across API replicas

The underlying V2 safety behavior remains unchanged:
- real quantity hard cap Q1
- only probe after frozen OOS M5+30s label tail
- passive GTC entry/exit are post-only
- emergency cleanup is reduce-only IOC
- exceptions after selection trigger best-effort Q1 flatten

This module does not alter the frozen OOS recorder/shadow strategy.
"""

import time
import numpy as np

from . import mm_live_q1_queue_probe_v1 as B
from . import mm_live_q1_queue_probe_v2 as V2

STUDY_VERSION = "MM_LIVE_Q1_QUEUE_PROBE_V3"
BATCH_RETRIES = 3
BATCH_RETRY_SLEEP_S = 0.05


def _robust_queue_position(client, order_id):
    """Return earliest queue position we can observe for a newly resting order.

    Keeps the V1 return contract: (queue_position, timing_dict, error_or_none).
    Extra metadata in timing_dict identifies which API surface succeeded.
    """
    individual_error = None

    # Fast path: individual order queue endpoint.
    try:
        body, timing = client.get(f"/portfolio/orders/{order_id}/queue_position")
        q = B._f(body.get("queue_position_fp"))
        timing = dict(timing)
        timing["queue_source"] = "individual"
        timing["queue_attempt"] = 1
        if np.isfinite(q):
            return q, timing, None
    except Exception as exc:
        individual_error = repr(exc)

    batch_errors = []
    last_timing = {}

    # Fallback: Kalshi documents this endpoint as the list of queue positions
    # for all currently resting orders. Do not filter by ticker so we do not
    # need another GET merely to recover ticker metadata.
    for attempt in range(1, BATCH_RETRIES + 1):
        try:
            body, timing = client.get(
                "/portfolio/orders/queue_positions",
                params={"subaccount": 0},
            )
            last_timing = dict(timing)
            rows = body.get("queue_positions") or []
            for row in rows:
                if str(row.get("order_id") or "") == str(order_id):
                    q = B._f(row.get("queue_position_fp"))
                    last_timing["queue_source"] = "batch_fallback"
                    last_timing["queue_attempt"] = attempt
                    last_timing["individual_error"] = individual_error
                    if np.isfinite(q):
                        return q, last_timing, None
            batch_errors.append(
                f"batch attempt {attempt}: order_id absent from resting queue list"
            )
        except Exception as exc:
            batch_errors.append(f"batch attempt {attempt}: {exc!r}")

        if attempt < BATCH_RETRIES:
            time.sleep(BATCH_RETRY_SLEEP_S)

    # Diagnostic status check only after queue surfaces failed. This cannot
    # recover the historical queue position; it tells us whether the missing
    # read occurred while the order still appeared resting.
    status_note = None
    try:
        order, _ = B._get_order(client, order_id)
        status_note = (
            f"order status={order.get('status')!r}, "
            f"remaining={order.get('remaining_count_fp')!r}, "
            f"filled={order.get('fill_count_fp')!r}"
        )
    except Exception as exc:
        status_note = f"order status lookup failed: {exc!r}"

    err_parts = []
    if individual_error:
        err_parts.append("individual=" + individual_error)
    err_parts.extend(batch_errors)
    if status_note:
        err_parts.append(status_note)

    last_timing = dict(last_timing)
    last_timing["queue_source"] = "unavailable"
    last_timing["individual_error"] = individual_error
    last_timing["batch_errors"] = batch_errors
    last_timing["status_note"] = status_note
    return np.nan, last_timing, "; ".join(err_parts)


def run_live_q1_queue_probe(
    *,
    confirm_live: bool = False,
    max_wait_s=B.MAX_DISCOVERY_WAIT_S,
    entry_wait_s=B.ENTRY_WAIT_S,
    exit_wait_s=B.EXIT_WAIT_S,
    target_queue: float = V2.DEFAULT_TARGET_QUEUE,
    show: bool = True,
):
    """Run one real Q1 probe with V2 safety + robust queue-position fallback."""
    if confirm_live is not True:
        raise RuntimeError(
            "Real order submission disabled. Re-run with confirm_live=True only if you intend to send a real Q1 order."
        )

    original_queue = B._queue_position
    B._queue_position = _robust_queue_position
    try:
        if show:
            print(
                "V3 queue reader | individual endpoint + batch resting-order fallback"
            )
        result = V2.run_live_q1_queue_probe(
            confirm_live=True,
            max_wait_s=max_wait_s,
            entry_wait_s=entry_wait_s,
            exit_wait_s=exit_wait_s,
            target_queue=target_queue,
            show=show,
        )
        if isinstance(result, dict):
            result["queue_probe_version"] = STUDY_VERSION
        return result
    finally:
        B._queue_position = original_queue


rescue_flatten_ticker = V2.rescue_flatten_ticker

__all__ = ["run_live_q1_queue_probe", "rescue_flatten_ticker"]
