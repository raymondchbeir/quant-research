from __future__ import annotations

"""V4 live Q1 queue/latency probe.

Fixes the V3 queue-reader bug discovered in production:
GET /portfolio/orders/queue_positions requires market_tickers or event_ticker.

V4 therefore performs the queue read directly from the submit path while the
selected ticker is already known:
  1) POST the real Q1 passive order.
  2) Immediately GET /portfolio/orders/queue_positions with
     market_tickers=<selected ticker>, subaccount=0.
  3) Retry briefly if the new resting order has not appeared yet.
  4) Only then fall back to the individual-order queue endpoint.

Safety behavior is retained:
- hard cap Q1
- probe only after frozen OOS M5+30s label tail
- entry + passive exit are post_only GTC
- passive GTC exit does NOT use reduce_only (production API rejects that)
- emergency cleanup is reduce_only IOC
- exception after market selection triggers best-effort Q1 flatten

This module does not alter the frozen OOS recorder/shadow strategy.
"""

import time
import numpy as np

from . import mm_live_q1_queue_probe_v1 as B
from . import mm_live_q1_queue_probe_v2 as V2

STUDY_VERSION = "MM_LIVE_Q1_QUEUE_PROBE_V4"
DEFAULT_TARGET_QUEUE = 50.0
BATCH_RETRIES = 4
BATCH_RETRY_SLEEP_S = 0.025


def _batch_queue_position(client, *, ticker: str, order_id: str):
    """Read FIFO position from Kalshi's market-filtered resting-order endpoint."""
    errors = []
    last_timing = {}
    observations = []

    for attempt in range(1, BATCH_RETRIES + 1):
        try:
            body, timing = client.get(
                "/portfolio/orders/queue_positions",
                params={"market_tickers": str(ticker), "subaccount": 0},
            )
            last_timing = dict(timing)
            rows = body.get("queue_positions") or []
            observations.append({
                "attempt": attempt,
                "row_count": len(rows),
                "order_ids": [str(r.get("order_id") or "") for r in rows],
            })
            for row in rows:
                if str(row.get("order_id") or "") != str(order_id):
                    continue
                q = B._f(row.get("queue_position_fp"))
                if np.isfinite(q):
                    last_timing.update({
                        "queue_source": "batch_market_filtered",
                        "queue_attempt": attempt,
                        "market_tickers_param": str(ticker),
                        "batch_observations": observations,
                    })
                    return q, last_timing, None
            errors.append(f"batch attempt {attempt}: order_id absent")
        except Exception as exc:
            errors.append(f"batch attempt {attempt}: {exc!r}")

        if attempt < BATCH_RETRIES:
            time.sleep(BATCH_RETRY_SLEEP_S)

    # Fallback only. The individual surface has returned intermittent 404s in
    # production, so the market-filtered batch surface is deliberately primary.
    individual_error = None
    try:
        body, timing = client.get(f"/portfolio/orders/{order_id}/queue_position")
        q = B._f(body.get("queue_position_fp"))
        timing = dict(timing)
        timing.update({
            "queue_source": "individual_fallback",
            "queue_attempt": 1,
            "market_tickers_param": str(ticker),
            "batch_errors": errors,
            "batch_observations": observations,
        })
        if np.isfinite(q):
            return q, timing, None
    except Exception as exc:
        individual_error = repr(exc)

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

    last_timing = dict(last_timing)
    last_timing.update({
        "queue_source": "unavailable",
        "market_tickers_param": str(ticker),
        "batch_errors": errors,
        "batch_observations": observations,
        "individual_error": individual_error,
        "status_note": status_note,
    })
    msg = "; ".join(errors + (["individual=" + individual_error] if individual_error else []) + ([status_note] if status_note else []))
    return np.nan, last_timing, msg


def _submit_and_measure_v4(
    client,
    *,
    ticker,
    side,
    qty,
    price,
    predicted_queue,
    reduce_only,
    label,
):
    # Production discovery: Kalshi only accepts reduce_only for IOC. The
    # passive EXIT quantity is exactly the filled entry quantity, so the resting
    # exit remains bounded to this probe's Q1 round trip without reduce_only.
    if str(label).upper() == "EXIT":
        reduce_only = False

    decision_ms = time.time_ns() / 1e6
    payload = B._make_order_payload(
        ticker,
        side,
        qty,
        price,
        reduce_only=reduce_only,
        post_only=True,
        tif="good_till_canceled",
    )
    body, timing = client.post("/portfolio/events/orders", payload)
    order_id = str(body.get("order_id") or "")
    if not order_id:
        raise RuntimeError(f"{label}: create response missing order_id: {body}")

    actual_q, qtiming, qerr = _batch_queue_position(
        client,
        ticker=str(ticker),
        order_id=order_id,
    )

    engine_ms = B._f(timing.get("engine_ts_ms"))
    result = {
        "label": label,
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "price": price,
        "predicted_queue_ahead": predicted_queue,
        "actual_queue_position_first": actual_q,
        "extra_queue_vs_snapshot": (
            actual_q - predicted_queue
            if np.isfinite(actual_q) and np.isfinite(predicted_queue)
            else np.nan
        ),
        "order_id": order_id,
        "client_order_id": body.get("client_order_id") or payload["client_order_id"],
        "create_response": body,
        "create_timing": timing,
        "queue_query_timing": qtiming,
        "queue_query_error": qerr,
        "queue_source": qtiming.get("queue_source"),
        "decision_to_engine_ms_local_clock": (
            engine_ms - decision_ms if np.isfinite(engine_ms) else np.nan
        ),
        "engine_to_first_queue_observation_ms_local_clock": (
            B._f(qtiming.get("response_recv_wall_ms")) - engine_ms
            if np.isfinite(engine_ms)
            and np.isfinite(B._f(qtiming.get("response_recv_wall_ms")))
            else np.nan
        ),
    }

    print(f"\n{label} POSTED")
    print(f"  {ticker} | {str(side).upper()} | {qty:.2f} @ {price:.4f}")
    print(f"  predicted queue ahead: {predicted_queue:.2f}")
    if np.isfinite(actual_q):
        print(f"  actual first queue pos: {actual_q:.2f} [{result['queue_source']}]")
        print(f"  EXTRA ahead vs snapshot: {result['extra_queue_vs_snapshot']:+.2f}")
    else:
        print("  actual first queue pos: unavailable")
        print(f"  queue source: {result['queue_source']}")
        if qerr:
            print(f"  queue read error: {qerr}")
    print(f"  create RTT: {timing['rtt_ms']:.1f} ms")
    if np.isfinite(timing.get("send_to_engine_ms_local_clock", np.nan)):
        print(
            f"  send -> matching engine: {timing['send_to_engine_ms_local_clock']:.1f} ms "
            "(local-clock dependent)"
        )
    if np.isfinite(result["decision_to_engine_ms_local_clock"]):
        print(
            f"  decision -> matching engine: {result['decision_to_engine_ms_local_clock']:.1f} ms "
            "(local-clock dependent)"
        )
    return result


def run_live_q1_queue_probe(
    *,
    confirm_live: bool = False,
    max_wait_s=B.MAX_DISCOVERY_WAIT_S,
    entry_wait_s=B.ENTRY_WAIT_S,
    exit_wait_s=B.EXIT_WAIT_S,
    target_queue: float = DEFAULT_TARGET_QUEUE,
    show: bool = True,
):
    """Run one real Q1 probe with market-filtered batch queue observation."""
    if confirm_live is not True:
        raise RuntimeError(
            "Real order submission disabled. Re-run with confirm_live=True only if you intend to send a real Q1 order."
        )

    target_queue = float(target_queue)
    if not (1.0 <= target_queue <= B.MAX_QUEUE_FOR_SELECTION):
        raise ValueError(
            f"target_queue must be within [1, {B.MAX_QUEUE_FOR_SELECTION}]"
        )

    original_submit = B._submit_and_measure
    original_choose = B._choose_probe_market
    original_target = B.TARGET_QUEUE
    selected = {}

    def choose_capture(*args, **kwargs):
        out = original_choose(*args, **kwargs)
        selected.update(out)
        return out

    B._submit_and_measure = _submit_and_measure_v4
    B._choose_probe_market = choose_capture
    B.TARGET_QUEUE = target_queue

    try:
        if show:
            print(
                "V4 queue reader | market-filtered batch endpoint first "
                "(market_tickers=<selected ticker>)"
            )
        result = B.run_live_q1_queue_probe(
            confirm_live=True,
            max_wait_s=max_wait_s,
            entry_wait_s=entry_wait_s,
            exit_wait_s=exit_wait_s,
            show=show,
        )
        if isinstance(result, dict):
            result["queue_probe_version"] = STUDY_VERSION
            result["target_queue"] = target_queue
        return result
    except Exception as exc:
        ticker = str(selected.get("ticker") or "")
        if ticker:
            print(f"\nPROBE ERROR: {exc!r}")
            print(f"Fail-safe: checking {ticker} for stranded Q1 inventory...")
            try:
                V2.rescue_flatten_ticker(
                    ticker,
                    confirm_live=True,
                    show=True,
                )
            except Exception as cleanup_exc:
                print(f"CRITICAL CLEANUP ERROR: {cleanup_exc!r}")
                print(f"CHECK KALSHI POSITION MANUALLY NOW: {ticker}")
        raise
    finally:
        B._submit_and_measure = original_submit
        B._choose_probe_market = original_choose
        B.TARGET_QUEUE = original_target


rescue_flatten_ticker = V2.rescue_flatten_ticker

__all__ = ["run_live_q1_queue_probe", "rescue_flatten_ticker"]
