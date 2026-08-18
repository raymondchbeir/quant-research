from __future__ import annotations

"""V11 live runner: V10 recording bundle + V7 strategy mechanics + hardened V2 order transport.

Why V11 exists
--------------
The V10 Q1 recording smoke exposed two API-transport edge cases without changing
our frozen strategy hypothesis:

1) an order-create request could be accepted by the exchange while the client did
   not receive/retain the 201 response. Retrying with the same client_order_id then
   correctly produced HTTP 409 order_already_exists, but the old recovery lookup
   was too short to observe the newly created order through GET /portfolio/orders.
2) the legacy DELETE /portfolio/orders/{order_id} compatibility fallback now
   returns 410 deprecated_v1_order_endpoint. It must not be used.

V11 preserves V7 trading decisions and V10 recording. It changes only order API
transport/reconciliation:
- one client_order_id per intended order;
- recover ambiguous create results by polling current orders for that same client id;
- after a 409 duplicate, NEVER create a fresh replacement until the original order
  is recovered or the engine fails closed;
- use V2 cancel only, retrying transient/404 visibility races;
- if singular V2 cancel cannot remove a still-resting order, try documented V2
  batch cancel once;
- if the order still appears resting, trigger the strategy order group and fail
  closed;
- no deprecated V1 order mutation endpoints.

REAL ORDERS are sent only by explicitly armed launchers.
"""

import argparse
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v4 as V4
from . import mm_cycle_q10_live_strategy_v6 as V6
from . import mm_cycle_q10_live_strategy_v7 as V7
from . import mm_cycle_q10_live_strategy_v10 as V10

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V11"
EXECUTION_PARENT = V7.LIVE_VERSION
RECORDING_PARENT = V10.LIVE_VERSION
ORDER_API_SAFETY_VERSION = "MM_CYCLE_Q10_ORDER_API_SAFETY_V11"
MAX_ACTION_BOOK_AGE_S = V6.MAX_ACTION_BOOK_AGE_S

_CREATE_DIAG = Counter()


def _http_code(exc):
    m = re.search(r"->\s*(\d{3})\s*:", repr(exc))
    return int(m.group(1)) if m else None


def _find_client_order_once(client, *, cid, ticker):
    body, timing = client.get(
        "/portfolio/orders",
        params={
            "ticker": str(ticker),
            "limit": 1000,
            "subaccount": 0,
            "exchange_index": 0,
        },
    )
    for r in body.get("orders") or []:
        if str(r.get("client_order_id") or "") == str(cid):
            return r, timing
    return None, timing


def _recover_client_order(client, payload, *, timeout_s):
    """Poll the current-order view for an idempotent create result."""
    cid = str(payload["client_order_id"])
    ticker = str(payload["ticker"])
    deadline = time.monotonic() + float(timeout_s)
    polls = 0
    last_error = None
    last_timing = None
    while time.monotonic() < deadline:
        polls += 1
        try:
            row, last_timing = _find_client_order_once(client, cid=cid, ticker=ticker)
            if row is not None:
                return row, {
                    "polls": polls,
                    "last_timing": last_timing,
                    "last_error": last_error,
                }
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(min(0.06 + 0.025 * polls, 0.20))
    return None, {"polls": polls, "last_timing": last_timing, "last_error": last_error}


def _recovered_create_body(row, payload):
    return {
        "order_id": row.get("order_id"),
        "client_order_id": payload["client_order_id"],
        "fill_count": row.get("fill_count_fp", row.get("fill_count")) or "0.00",
        "remaining_count": row.get("remaining_count_fp", row.get("remaining_count")) or "0.00",
        "ts_ms": row.get("last_updated_ts_ms", row.get("created_ts_ms")),
        "recovered": True,
        "recovered_status": row.get("status"),
    }


def _post_v11(client, payload):
    """Idempotent V2 create with explicit 409 recovery and at most one resubmit.

    The same client_order_id is used throughout. A 409 means the exchange already
    knows that client id; after that response this function never POSTs again.
    """
    cid = str(payload["client_order_id"])
    ticker = str(payload["ticker"])
    first_error = None
    second_error = None

    try:
        body, timing = client.post("/portfolio/events/orders", payload)
        _CREATE_DIAG["create_201_success"] += 1
        return body, timing
    except Exception as exc:
        first_error = exc
        code = _http_code(exc)
        _CREATE_DIAG["create_initial_errors"] += 1
        if code == 409:
            _CREATE_DIAG["create_409_conflicts"] += 1

    # Before any resubmit, allow the read view time to reveal an order whose 201
    # response may have been lost/ambiguous.
    row, rec = _recover_client_order(client, payload, timeout_s=0.90)
    if row is not None:
        _CREATE_DIAG["create_recovered_without_resubmit"] += 1
        return _recovered_create_body(row, payload), {
            "v11_recovered": True,
            "phase": "after_initial_error",
            "client_order_id": cid,
            "ticker": ticker,
            "initial_error": repr(first_error),
            **rec,
        }

    first_code = _http_code(first_error)
    if first_code == 409:
        # Duplicate is definitive evidence that this client id was already accepted
        # or otherwise reserved. Never submit another order here.
        row, rec2 = _recover_client_order(client, payload, timeout_s=3.50)
        if row is not None:
            _CREATE_DIAG["create_recovered_after_409"] += 1
            return _recovered_create_body(row, payload), {
                "v11_recovered": True,
                "phase": "409_poll_only",
                "client_order_id": cid,
                "ticker": ticker,
                "initial_error": repr(first_error),
                **rec2,
            }
        _CREATE_DIAG["create_unrecoverable_409"] += 1
        raise RuntimeError(
            "V11 create fail-closed: exchange reports duplicate client_order_id but "
            f"order cannot be recovered from current orders; cid={cid} ticker={ticker} "
            f"error={first_error!r} recovery={rec2}"
        )

    # Do not retry definite non-transient 4xx request/auth errors. 429 is retryable.
    if first_code is not None and 400 <= first_code < 500 and first_code != 429:
        _CREATE_DIAG["create_permanent_errors"] += 1
        raise RuntimeError(f"V11 order create rejected without retry: {first_error!r}")

    # One idempotent resubmit is allowed for network/5xx/429 ambiguity, using the
    # exact same client_order_id documented for deduplication.
    _CREATE_DIAG["create_resubmits"] += 1
    time.sleep(0.12)
    try:
        body, timing = client.post("/portfolio/events/orders", payload)
        _CREATE_DIAG["create_resubmit_201_success"] += 1
        timing = dict(timing or {})
        timing.update({
            "v11_resubmitted_same_client_order_id": True,
            "initial_error": repr(first_error),
        })
        return body, timing
    except Exception as exc:
        second_error = exc
        code = _http_code(exc)
        _CREATE_DIAG["create_resubmit_errors"] += 1
        if code == 409:
            _CREATE_DIAG["create_409_conflicts"] += 1

    # After the resubmit, never POST again. Poll long enough for REST read-view
    # propagation, then fail closed if the accepted order cannot be identified.
    row, rec3 = _recover_client_order(client, payload, timeout_s=3.50)
    if row is not None:
        _CREATE_DIAG["create_recovered_after_resubmit"] += 1
        return _recovered_create_body(row, payload), {
            "v11_recovered": True,
            "phase": "after_single_resubmit",
            "client_order_id": cid,
            "ticker": ticker,
            "initial_error": repr(first_error),
            "second_error": repr(second_error),
            **rec3,
        }

    _CREATE_DIAG["create_unrecoverable"] += 1
    raise RuntimeError(
        "V11 create fail-closed after one idempotent resubmit; "
        f"cid={cid} ticker={ticker} initial={first_error!r} second={second_error!r} recovery={rec3}"
    )


class V2OnlyProductionEngine(V10.ProductionRecordingEngine):
    """V10 recorder + V7 decisions, with V11 V2-only cancellation/recovery."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v11_cancel_v2_retries = 0
        self.v11_cancel_v2_retry_successes = 0
        self.v11_cancel_batch_fallbacks = 0
        self.v11_cancel_batch_successes = 0
        self.v11_cancel_still_resting_failures = 0

    @staticmethod
    def _fill_sum(rows):
        total = 0.0
        for r in rows or []:
            total += B._f(r.get("count_fp", r.get("count")), 0.0)
        return float(total)

    def _resting_row(self, oid):
        rows, timing = B._resting(self.client)
        row = next((r for r in rows if str(r.get("order_id") or "") == str(oid)), None)
        return row, timing

    @staticmethod
    def _parse_reduced_by(body, submitted_qty):
        reduced_by = B._f((body or {}).get("reduced_by"), np.nan)
        if not np.isfinite(reduced_by):
            raise RuntimeError(f"V2 cancel receipt missing finite reduced_by: {body}")
        if reduced_by < -B.EPS or reduced_by > submitted_qty + B.EPS:
            raise RuntimeError(
                f"Invalid V2 cancel reduced_by={reduced_by} for submitted_qty={submitted_qty}: {body}"
            )
        return float(max(0.0, reduced_by))

    def _batch_cancel_v2(self, oid):
        payload = {
            "orders": [{"order_id": str(oid), "subaccount": 0, "exchange_index": 0}]
        }
        body, timing = self.client.request(
            "DELETE", "/portfolio/events/orders/batched", payload=payload
        )
        rows = body.get("orders") or []
        row = next((r for r in rows if str(r.get("order_id") or "") == str(oid)), None)
        if row is None:
            raise RuntimeError(f"V2 batch cancel response missing order_id={oid}: {body}")
        err = row.get("error")
        if err:
            raise RuntimeError(f"V2 batch cancel item error: {err}")
        return row, timing

    def cancel_track(self, ticker, reason):
        tr = self.active.get(ticker)
        if not tr:
            return False

        oid = str(tr["order_id"])
        old_fill = float(tr.get("last_fill", 0.0))
        submitted_qty = float(tr.get("qty", 0.0))
        before_pos = float(self.positions.get(ticker, 0.0))

        source = None
        cancel_body = None
        cancel_timing = None
        cancel_errors = []
        resting_checks = []
        batch_body = None
        batch_timing = None
        batch_error = None
        receipt_fill = old_fill

        # Primary documented V2 cancel. If a 404/transport race occurs while the
        # list endpoint still shows RESTING, retry the same cancellation briefly.
        for attempt, delay in enumerate((0.0, 0.08, 0.16, 0.30), start=1):
            if delay:
                time.sleep(delay)
                self.v11_cancel_v2_retries += 1
            try:
                cancel_body, cancel_timing = self.client.delete(
                    f"/portfolio/events/orders/{oid}",
                    params={"subaccount": 0, "exchange_index": 0},
                )
                reduced_by = self._parse_reduced_by(cancel_body, submitted_qty)
                receipt_fill = max(old_fill, submitted_qty - reduced_by)
                source = "V2_CANCEL" if attempt == 1 else "V2_CANCEL_RETRY"
                if attempt > 1:
                    self.v11_cancel_v2_retry_successes += 1
                self.v7_cancel_v2_successes += 1
                break
            except Exception as exc:
                cancel_errors.append(repr(exc))
                if attempt == 1:
                    self.v7_cancel_v2_errors += 1

                try:
                    resting, rt = self._resting_row(oid)
                except Exception as r_exc:
                    resting, rt = None, {"error": repr(r_exc)}
                resting_checks.append({
                    "attempt": attempt,
                    "resting": resting,
                    "timing": rt,
                })
                if resting is None:
                    source = "V2_ERROR_ALREADY_ABSENT"
                    self.v7_cancel_already_absent += 1
                    break

        # If singular V2 cancellation still cannot remove an order that the
        # authoritative resting set shows as live, use the documented V2 batch
        # cancel surface once. Never call deprecated /portfolio/orders/{id} DELETE.
        if source is None:
            self.v11_cancel_batch_fallbacks += 1
            try:
                batch_body, batch_timing = self._batch_cancel_v2(oid)
                reduced_by = self._parse_reduced_by(batch_body, submitted_qty)
                receipt_fill = max(old_fill, submitted_qty - reduced_by)
                source = "V2_BATCH_CANCEL_FALLBACK"
                self.v11_cancel_batch_successes += 1
            except Exception as exc:
                batch_error = repr(exc)

            time.sleep(0.12)
            try:
                resting_final, rt_final = self._resting_row(oid)
            except Exception as r_exc:
                resting_final, rt_final = None, {"error": repr(r_exc)}
            resting_checks.append({
                "attempt": "after_batch",
                "resting": resting_final,
                "timing": rt_final,
            })

            if resting_final is not None:
                self.v11_cancel_still_resting_failures += 1
                trig = self._emergency_group_trigger(
                    ticker=ticker,
                    oid=oid,
                    reason=reason,
                    detail={
                        "v2_errors": cancel_errors,
                        "batch_error": batch_error,
                        "resting_order": resting_final,
                    },
                )
                raise RuntimeError(
                    "V11 fail-closed cancel: order remains RESTING after V2 single retries "
                    f"and V2 batch fallback; ticker={ticker} order_id={oid} "
                    f"v2_errors={cancel_errors} batch_error={batch_error} group_trigger={trig}"
                )

            if source is None:
                source = "V2_CANCEL_ERRORS_BUT_NOW_ABSENT"
                self.v7_cancel_already_absent += 1

        fill_rows = []
        fill_timing = {}
        fill_error = None
        try:
            fill_rows, fill_timing = B._fills(self.client, oid)
            receipt_fill = max(receipt_fill, self._fill_sum(fill_rows))
        except Exception as exc:
            fill_error = repr(exc)

        final_fill = min(submitted_qty, max(old_fill, float(receipt_fill)))
        tr["last_fill"] = final_fill
        self.record_fills(tr)

        # Actual account position remains the final exposure authority.
        after_pos = self.refresh_position(ticker)

        B._append(self.orders, {
            "time": B._iso(),
            "action": "CANCEL_V11_V2_ONLY_VERIFIED",
            "ticker": ticker,
            "reason": reason,
            "track": tr,
            "cancel_source": source,
            "cancel_body": cancel_body,
            "cancel_timing": cancel_timing,
            "cancel_errors": cancel_errors,
            "batch_body": batch_body,
            "batch_timing": batch_timing,
            "batch_error": batch_error,
            "resting_checks": resting_checks,
            "fill_read_timing": fill_timing,
            "fill_read_error": fill_error,
            "old_fill": old_fill,
            "final_fill": final_fill,
            "position_before": before_pos,
            "position_after": after_pos,
        })

        self.active.pop(ticker, None)
        self.counts["cancels"] += 1

        raced_fill = (
            final_fill > old_fill + B.EPS
            or abs(after_pos - before_pos) > B.EPS
        )
        if raced_fill:
            self.barrier[ticker] = self.book_version[ticker]
            self.counts["fill_events"] += 1
            self.emit(
                "FILL",
                ticker,
                role=tr["role"],
                side=tr["side"],
                qty=max(0.0, final_fill - old_fill),
                position=after_pos,
                source="v11_v2_cancel_or_position",
            )
        return raced_fill

    def _v11_metrics(self):
        return {
            "live_version": LIVE_VERSION,
            "order_api_safety_version": ORDER_API_SAFETY_VERSION,
            "create": dict(_CREATE_DIAG),
            "cancel_v2_retries": self.v11_cancel_v2_retries,
            "cancel_v2_retry_successes": self.v11_cancel_v2_retry_successes,
            "cancel_batch_fallbacks": self.v11_cancel_batch_fallbacks,
            "cancel_batch_successes": self.v11_cancel_batch_successes,
            "cancel_still_resting_failures": self.v11_cancel_still_resting_failures,
            "deprecated_v1_cancel_calls": 0,
        }

    def _v10_metrics(self):
        base = super()._v10_metrics()
        base.update({
            "live_version": LIVE_VERSION,
            "recording_parent": RECORDING_PARENT,
            "order_api_safety": self._v11_metrics(),
            "strategy_mechanics_changed_from_v7": False,
        })
        return base

    @staticmethod
    def _positions_from_final(summary):
        out = {}
        for r in summary.get("final_positions") or []:
            if not isinstance(r, dict):
                continue
            t = str(r.get("ticker") or "")
            p = B._f(r.get("position_fp"), 0.0)
            if t and abs(p) > B.EPS:
                out[t] = p
        return out

    def health(self, force=False):
        super().health(force=force)
        h = B._read(self.health_path, {}) or {}
        h["live_version"] = LIVE_VERSION
        h["execution_parent"] = EXECUTION_PARENT
        h["recording_parent"] = RECORDING_PARENT
        h["v11_metrics"] = self._v11_metrics()
        if self.shutdown_started:
            summary = B._read(self.final_path, {}) or {}
            if summary:
                h["positions"] = self._positions_from_final(summary)
        B._atomic(self.health_path, h)

    def shutdown(self, reason):
        if self.shutdown_started:
            return
        super().shutdown(reason)
        summary = B._read(self.final_path, {}) or {}
        summary["live_wrapper_version"] = LIVE_VERSION
        summary["execution_parent"] = EXECUTION_PARENT
        summary["recording_parent"] = RECORDING_PARENT
        summary["order_api_safety_version"] = ORDER_API_SAFETY_VERSION
        summary["v11_metrics"] = self._v11_metrics()
        summary.setdefault("v10_metrics", self._v10_metrics())
        B._atomic(self.final_path, summary)

        h = B._read(self.health_path, {}) or {}
        h["live_version"] = LIVE_VERSION
        h["execution_parent"] = EXECUTION_PARENT
        h["recording_parent"] = RECORDING_PARENT
        h["v11_metrics"] = self._v11_metrics()
        h["positions"] = self._positions_from_final(summary)
        h["summary"] = summary
        B._atomic(self.health_path, h)
        V10.verify_recording_bundle(self.session, show=False, write_result=True)


def _write_v11_bundle(session: Path, cfg):
    session = Path(session).resolve()
    V10._write_static_bundle(session, cfg)

    prov = B._read(session / "source_provenance.json", {}) or {}
    prov.update({
        "live_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_parent": RECORDING_PARENT,
        "order_api_safety_version": ORDER_API_SAFETY_VERSION,
    })
    sources = dict(prov.get("sources") or {})
    sources["v11"] = V10._module_source(sys.modules[__name__])
    prov["sources"] = sources
    B._atomic(session / "source_provenance.json", prov)

    manifest = B._read(session / "recording_manifest.json", {}) or {}
    manifest.update({
        "live_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_parent": RECORDING_PARENT,
        "order_api_safety_version": ORDER_API_SAFETY_VERSION,
    })
    B._atomic(session / "recording_manifest.json", manifest)

    execution = B._read(session / "live_execution_spec_v10.json", {}) or {}
    execution.update({
        "live_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_parent": RECORDING_PARENT,
        "strategy_mechanics_changed_from_v7": False,
        "order_api_transport_changed_from_v10": True,
        "order_api_safety_version": ORDER_API_SAFETY_VERSION,
        "create_idempotency": "same client_order_id; 409 poll-only recovery; at most one transient resubmit",
        "cancel_safety": "V2 singular retries -> V2 batch fallback -> order-group fail closed; no deprecated V1 mutation",
    })
    B._atomic(session / "live_execution_spec_v10.json", execution)
    B._atomic(session / "live_execution_spec_v11.json", execution)

    B._atomic(session / "order_api_safety_v11.json", {
        "time": B._iso(),
        "version": ORDER_API_SAFETY_VERSION,
        "strategy_changed": False,
        "recording_changed": False,
        "create": {
            "endpoint": "POST /portfolio/events/orders",
            "client_order_id": "unique UUID per intended order",
            "duplicate_409": "recover same order by client_order_id; never place a fresh replacement",
            "max_resubmits": 1,
        },
        "cancel": {
            "primary": "DELETE /portfolio/events/orders/{order_id}",
            "retry_if_still_resting": True,
            "fallback": "DELETE /portfolio/events/orders/batched",
            "deprecated_v1_mutation_endpoint_used": False,
            "fail_closed": "trigger order group if order remains resting",
        },
    })


def account_safety_check(*, show=True):
    """Read-only current account check. Sends no orders."""
    client = B.Q1.LiveClient()
    eq, raw, bt = B._balance(client)
    positions, pt = B._positions(client)
    resting, ot = B._resting(client)
    nonzero = [r for r in positions if abs(B._f(r.get("position_fp"), 0.0)) > B.EPS]
    out = {
        "time": B._iso(),
        "equity": eq,
        "nonzero_positions": nonzero,
        "resting_orders": resting,
        "flat": not nonzero,
        "zero_resting": not resting,
        "timing": {"balance": bt, "positions": pt, "orders": ot},
        "orders_sent": False,
    }
    if show:
        print("=" * 88)
        print("V11 READ-ONLY ACCOUNT SAFETY CHECK")
        print("=" * 88)
        print("Equity:       ", eq.get("equity_usd"))
        print("Flat:         ", out["flat"])
        print("Zero resting: ", out["zero_resting"])
        print("Positions:    ", nonzero)
        print("Resting:      ", resting)
        print("ORDERS SENT:  NO")
    return out


def backlog_regression_check(session_dir, *, bucket_ms=250, show=True):
    return V10.backlog_regression_check(session_dir, bucket_ms=bucket_ms, show=show)


def verify_recording_bundle(session_dir, *, show=True, write_result=True):
    return V10.verify_recording_bundle(
        session_dir, show=show, write_result=write_result
    )


def _run_process_v11(session, cfg):
    session = Path(session).resolve()
    _write_v11_bundle(session, cfg)

    client = B.Q1.LiveClient()
    diag = V3._install(client)
    B._atomic(session / "balance_semantics.json", diag)

    B._post = _post_v11
    B._start_recorder = V4._start_recorder_fixed
    B.LiveEngine = V2OnlyProductionEngine
    B._run_process(session, cfg)


def live_preflight(*, quote_size=B.FULL_Q, runtime_hours=B.FULL_HOURS,
                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                   min_start_equity_usd=None, show=True):
    return V3.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
    )


def _launch(*, mode, q, hours, max_loss, min_equity, arm, expected):
    if str(arm) != expected:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected!r} exactly.")

    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    V3._calibrated_preflight(
        quote_size=q,
        runtime_hours=hours,
        max_loss_usd=max_loss,
        min_equity_usd=min_equity,
        mode=mode,
        save_dir=None,
        show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_{mode.lower()}_v11").resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": mode,
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_parent": RECORDING_PARENT,
        "engine_architecture": "V7_STRATEGY_V10_RECORDING_V11_V2_ORDER_TRANSPORT",
        "max_action_book_age_s": MAX_ACTION_BOOK_AGE_S,
        "recording_version": V10.RECORDING_VERSION,
        "comparison_schema_version": V10.COMPARISON_SCHEMA_VERSION,
        "order_api_safety_version": ORDER_API_SAFETY_VERSION,
    }
    B._atomic(session / "process_config.json", cfg)
    _write_v11_bundle(session, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v11",
                "--run-live-session", str(session),
                "--config", str(session / "process_config.json"),
            ],
            cwd=str(C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        fh.close()

    B._atomic(B.CONTROL_PATH, {
        "live_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_parent": RECORDING_PARENT,
        "recording_version": V10.RECORDING_VERSION,
        "order_api_safety_version": ORDER_API_SAFETY_VERSION,
        "running": True,
        "pid": p.pid,
        "session_dir": str(session),
        "mode": mode,
        "started_at": B._iso(),
        "config": cfg,
        "log_path": str(log),
    })

    deadline = time.time() + 90.0
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-16000:] if log.exists() else ""
            raise RuntimeError(f"Live V11 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-16000:] if log.exists() else ""
        raise RuntimeError(f"Live V11 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V11 PROCESS ARMED")
    print("  mode:       ", mode)
    print("  session:    ", session)
    print("  pid:        ", p.pid)
    print(f"  Q:           {q:g} per eligible market")
    print(f"  kill:        -${max_loss:.2f} from calibrated starting TOTAL account equity")
    print(f"  stale cap:   {MAX_ACTION_BOOK_AGE_S:.2f}s max latest-book age at CREATE")
    print("  strategy:    V7 mechanics unchanged")
    print("  recording:   V10 audit bundle unchanged")
    print("  order API:   V11 idempotent create + V2-only cancel safety")
    print("Use live_status(); emergency stop is kill_and_flatten_live(arm_phrase='KILL_AND_FLATTEN').")
    return live_status(show=False)


def start_live_smoke_q1_one_window(*, arm_phrase=None,
                                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                                   min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    return _launch(
        mode="SMOKE_Q1_ONE_WINDOW",
        q=B.SMOKE_Q,
        hours=1.0,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected=B.SMOKE_ARM,
    )


def start_live_cycle_q10(*, arm_phrase=None, runtime_hours=B.FULL_HOURS,
                         max_start_loss_usd=B.LOSS_LIMIT_USD,
                         min_start_equity_usd=B.FULL_MIN_EQUITY):
    if abs(float(runtime_hours) - B.FULL_HOURS) > B.EPS:
        raise RuntimeError("V11 full validation is frozen to exactly 24 hours.")
    return _launch(
        mode="LIVE_Q10_24H",
        q=B.FULL_Q,
        hours=B.FULL_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected=B.FULL_ARM,
    )


def live_status(*, show=True, tail_lines=20):
    return B.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    return B.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    ap.add_argument("--verify-recording-session")
    a = ap.parse_args()

    if a.verify_recording_session:
        verify_recording_bundle(a.verify_recording_session, show=True, write_result=True)
        return
    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}
        _run_process_v11(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "EXECUTION_PARENT",
    "RECORDING_PARENT",
    "ORDER_API_SAFETY_VERSION",
    "V2OnlyProductionEngine",
    "account_safety_check",
    "backlog_regression_check",
    "verify_recording_bundle",
    "live_preflight",
    "start_live_smoke_q1_one_window",
    "start_live_cycle_q10",
    "live_status",
    "kill_and_flatten_live",
]
