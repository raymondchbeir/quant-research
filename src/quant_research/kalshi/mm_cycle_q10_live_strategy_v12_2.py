from __future__ import annotations

"""V12.2: close the remaining V12.1 create race and audit lifecycle correctly.

Frozen Candidate-C strategy mechanics are unchanged.

Changes from V12.1
------------------
1. The raw watchdog exposes an exact byte-offset catch-up barrier. Immediately
   before a CREATE HTTP request, the engine waits for the watchdog to consume the
   raw JSONL file through a freshly sampled EOF and revalidates the certified
   quote against the newest processed raw state. If the barrier cannot certify
   freshness within a tight budget, the CREATE is skipped without touching the
   exchange.
2. Priority-cancel worker sessions are pre-warmed with read-only balance GETs
   before trading, removing first-use DNS/TLS/session latency from the cancel path.
3. Every fast invalidation receives an explicit terminal state, including a
   fast-delete error later verified/retired by the V11 fallback and a result that
   arrives after another authoritative path already retired the order.
4. Promotion gates separate already-RESTING invalidations from CREATE-FLIGHT
   invalidations. The <25/<50/<100 ms SLO applies to resting-order obsolescence;
   create-flight exposure is measured from CREATE response to cancel send.

REAL ORDERS are sent only by explicitly armed launchers.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v4 as V4
from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_cycle_q10_live_strategy_v11 as V11
from . import mm_cycle_q10_live_strategy_v12 as V12
from . import mm_cycle_q10_live_strategy_v12_1 as V121

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V12_2"
EXECUTION_PARENT = V121.LIVE_VERSION
RECORDING_PARENT = V121.RECORDING_PARENT
FRESHNESS_ARCH_VERSION = "MM_CYCLE_Q10_PRIORITY_FRESHNESS_V12_2"
STAGED_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V12_2_STAGED"

Q5_QTY = 5.0
Q5_HOURS = 1.0
Q5_MIN_EQUITY_USD = 75.0
Q5_ARM = "LIVE_Q5_1H"
Q10_ARM = B.FULL_ARM

PRESEND_EOF_TIMEOUT_MS = 25.0
PRESEND_PROGRESS_WAIT_S = 0.0005
CREATE_FLIGHT_P95_RESPONSE_TO_CANCEL_MS = 50.0
CREATE_FLIGHT_MAX_RESPONSE_TO_CANCEL_MS = 100.0


class PresendFreshnessAbort(RuntimeError):
    """No-order-sent control-flow exception for a failed pre-send freshness guard."""


class BarrierFastCancelWatchdog(V121.FixedFastCancelWatchdog):
    """V12.1 watchdog + exact byte-offset catch-up + warmed worker sessions."""

    def __init__(self, engine):
        # Reproduce V12 initialization so the thread target resolves to this
        # class's binary-offset _run implementation.
        self.engine = engine
        self.raw_path = engine.session / "raw_capture" / "book_top3_events.jsonl"
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.caught_up = threading.Event()
        self.lock = threading.RLock()
        self.progress_event = threading.Event()

        self.active = {}
        self.latest = {}
        self.pending = {}
        from collections import defaultdict, deque
        self.history = defaultdict(lambda: deque(maxlen=256))
        self.seq = 0

        self.results = Queue()
        self.executor = ThreadPoolExecutor(
            max_workers=V12.WATCHDOG_WORKERS,
            thread_name_prefix="v12-2-priority-cancel",
        )
        self.thread_clients = threading.local()
        self.thread = threading.Thread(
            target=self._run,
            name="v12-2-raw-watchdog",
            daemon=True,
        )

        self.rows_seen = 0
        self.invalidations = 0
        self.cancel_submissions = 0
        self.cancel_errors = 0
        self.max_raw_to_detect_ms = 0.0
        self.max_obsolete_to_send_ms = 0.0

        self.processed_offset = 0
        self.last_eof_offset = 0
        self.prewarm_complete = False
        self.prewarm_rows = []

    def _prewarm_workers(self):
        """Force all worker threads to own a live, TLS-warmed read-only session."""
        n = int(V12.WATCHDOG_WORKERS)
        barrier = threading.Barrier(n)

        def warm(i):
            started = V12._wall_ms()
            error = None
            timing = None
            try:
                client = self._client()
                _, timing = client.get("/portfolio/balance")
            except Exception as exc:
                error = repr(exc)
            try:
                barrier.wait(timeout=15.0)
            except Exception as exc:
                if error is None:
                    error = f"prewarm_barrier:{exc!r}"
            return {
                "worker_slot": i,
                "started_wall_ms": started,
                "done_wall_ms": V12._wall_ms(),
                "rtt_ms": B._f((timing or {}).get("rtt_ms")),
                "error": error,
            }

        futures = [self.executor.submit(warm, i) for i in range(n)]
        rows = [f.result(timeout=20.0) for f in futures]
        self.prewarm_rows = rows
        errors = [r for r in rows if r.get("error")]
        self.engine._lat(
            "WATCHDOG_WORKERS_PREWARMED",
            workers=n,
            successes=n - len(errors),
            errors=errors,
            rtt_ms=[r.get("rtt_ms") for r in rows],
        )
        if errors:
            raise RuntimeError(f"V12.2 priority-worker prewarm failed: {errors}")
        self.prewarm_complete = True

    def start(self):
        # Startup occurs before strategy arming; these are read-only GETs.
        self._prewarm_workers()
        self.thread.start()

    def _set_progress(self, offset, *, eof=False):
        with self.lock:
            self.processed_offset = max(self.processed_offset, int(offset))
            if eof:
                self.last_eof_offset = max(self.last_eof_offset, int(offset))
        self.progress_event.set()

    def _run(self):
        while not self.stop_event.is_set() and not self.raw_path.exists():
            self.stop_event.wait(0.005)
        if self.stop_event.is_set():
            return

        start_ms = V12._wall_ms()
        # Binary mode makes tell() a real byte offset, allowing a precise
        # comparison against stat().st_size.
        with self.raw_path.open("rb") as fh:
            while not self.stop_event.is_set():
                pos = fh.tell()
                line = fh.readline()

                if line and not line.endswith(b"\n"):
                    fh.seek(pos)
                    self.wake_event.wait(V12.WATCHDOG_SLEEP_S)
                    self.wake_event.clear()
                    continue

                if line:
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except Exception as exc:
                        self.engine._lat("WATCHDOG_JSON_ERROR", error=repr(exc))
                        self._set_progress(fh.tell())
                        continue
                    if isinstance(row, dict):
                        self._process_row(row)
                    self._set_progress(fh.tell())
                    continue

                eof = fh.tell()
                self._set_progress(eof, eof=True)
                if not self.caught_up.is_set():
                    self.caught_up.set()
                    self.engine._lat(
                        "WATCHDOG_CAUGHT_UP",
                        rows_seen=self.rows_seen,
                        startup_catchup_ms=V12._wall_ms() - start_ms,
                        processed_offset=eof,
                    )

                self._recheck_active()
                self.wake_event.wait(V12.WATCHDOG_SLEEP_S)
                self.wake_event.clear()

    def catch_up_to_stable_eof(self, timeout_ms=PRESEND_EOF_TIMEOUT_MS):
        """Consume every complete raw byte committed through a freshly sampled EOF.

        The target is re-sampled after each catch-up. If the writer keeps extending
        the file beyond the latency budget, the caller gets ok=False and must skip
        CREATE rather than trading on an uncertified state.
        """
        start_perf = V12._perf_ms()
        deadline = start_perf + float(timeout_ms)
        iterations = 0
        target = None

        if not self.ready() or not self.raw_path.exists():
            return {
                "ok": False,
                "reason": "WATCHDOG_NOT_READY",
                "wait_ms": V12._perf_ms() - start_perf,
                "target_offset": None,
                "processed_offset": self.processed_offset,
                "iterations": 0,
            }

        while V12._perf_ms() <= deadline:
            iterations += 1
            try:
                target = int(self.raw_path.stat().st_size)
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": f"RAW_STAT_ERROR:{exc!r}",
                    "wait_ms": V12._perf_ms() - start_perf,
                    "target_offset": target,
                    "processed_offset": self.processed_offset,
                    "iterations": iterations,
                }

            with self.lock:
                processed = int(self.processed_offset)

            if processed >= target:
                # Re-sample once more. If bytes appeared while we checked, loop
                # and consume them too; otherwise this is a stable catch-up point.
                try:
                    target2 = int(self.raw_path.stat().st_size)
                except Exception:
                    target2 = target
                if target2 <= processed:
                    return {
                        "ok": True,
                        "reason": None,
                        "wait_ms": V12._perf_ms() - start_perf,
                        "target_offset": target2,
                        "processed_offset": processed,
                        "iterations": iterations,
                    }
                target = target2

            self.progress_event.clear()
            self.wake_event.set()
            remaining_ms = max(0.0, deadline - V12._perf_ms())
            if remaining_ms <= 0:
                break
            self.progress_event.wait(
                min(PRESEND_PROGRESS_WAIT_S, remaining_ms / 1000.0)
            )

        with self.lock:
            processed = int(self.processed_offset)
        return {
            "ok": False,
            "reason": "EOF_CATCHUP_TIMEOUT",
            "wait_ms": V12._perf_ms() - start_perf,
            "target_offset": target,
            "processed_offset": processed,
            "iterations": iterations,
        }

    def metrics(self):
        out = super().metrics()
        with self.lock:
            out.update(
                {
                    "processed_offset": int(self.processed_offset),
                    "last_eof_offset": int(self.last_eof_offset),
                    "prewarm_complete": bool(self.prewarm_complete),
                    "prewarm_workers": len(self.prewarm_rows),
                }
            )
        return out


class PriorityFreshnessEngine122(V121.PriorityFreshnessEngine121):
    """V12.1 engine with pre-send raw catch-up and complete terminal accounting."""

    def __init__(self, *args, **kwargs):
        V11.V2OnlyProductionEngine.__init__(self, *args, **kwargs)
        self.latency_log_v12 = self.session / "latency_events_v12.jsonl"
        self._lat_lock = threading.Lock()
        self._api_context = {}
        self._request_original = self.client.request
        self.v12 = Counter()
        self.max_actionable = 0
        self._v121_final_guard_ms = {}
        self._v121_final_guard_track = {}
        self._instrument_client()
        self.fast = BarrierFastCancelWatchdog(self)
        self.fast.start()
        self._publish()
        self._lat("V12_2_ENGINE_READY", execution_parent=EXECUTION_PARENT)

    def _presend_create_guard(self, payload):
        ticker = str((payload or {}).get("ticker") or "")
        track = dict(self._v121_final_guard_track.get(ticker) or {})
        sync = self.fast.catch_up_to_stable_eof(PRESEND_EOF_TIMEOUT_MS)
        snap = self.fast.latest_snapshot(ticker)

        ok = bool(sync.get("ok"))
        reason = sync.get("reason")
        invalid = False
        invalid_reason = None

        if ok and not track:
            ok = False
            reason = "NO_FINAL_CERTIFIED_TRACK"
        if ok and not snap:
            ok = False
            reason = "NO_WATCHDOG_TICKER_STATE"
        if ok:
            invalid, invalid_reason = V12._track_invalidated(
                track, (snap or {}).get("row") or {}
            )
            if invalid:
                ok = False
                reason = f"LATEST_INVALID:{invalid_reason}"

        row = {
            "ticker": ticker,
            "ok": ok,
            "reason": reason,
            "sync": sync,
            "final_guard_receipt_wall_ms": B._f(
                self._v121_final_guard_ms.get(ticker)
            ),
            "latest_receipt_wall_ms": B._f((snap or {}).get("receipt_wall_ms")),
            "latest_seq": (snap or {}).get("seq"),
            "role": track.get("role"),
            "side": track.get("side"),
            "price": track.get("price"),
        }
        self._lat("PRESEND_CREATE_RAW_BARRIER", **row)
        return row

    def _instrument_client(self):
        """Main-client timing plus a raw EOF barrier immediately before CREATE."""
        def request(method, path, *, params=None, payload=None, timeout=8.0):
            wall_start = V12._wall_ms()
            perf_start = V12._perf_ms()
            timing = None
            error = None
            ctx = dict(self._api_context or {})
            attempted_exchange_call = False
            try:
                if (
                    str(method).upper() == "POST"
                    and str(path) == "/portfolio/events/orders"
                    and str(ctx.get("v12") or "").startswith("CREATE")
                    and hasattr(self, "fast")
                ):
                    guard = self._presend_create_guard(payload or {})
                    if not guard.get("ok"):
                        raise PresendFreshnessAbort(
                            f"V12.2 pre-send freshness block: {guard.get('reason')}"
                        )

                attempted_exchange_call = True
                body, timing = self._request_original(
                    method,
                    path,
                    params=params,
                    payload=payload,
                    timeout=timeout,
                )
                return body, timing
            except Exception as exc:
                error = repr(exc)
                raise
            finally:
                self._lat(
                    "MAIN_HTTP",
                    method=str(method).upper(),
                    path=str(path),
                    http_class=V12._http_class(method, path),
                    call_start_wall_ms=wall_start,
                    local_call_ms=V12._perf_ms() - perf_start,
                    request_send_wall_ms=B._f((timing or {}).get("request_send_wall_ms")),
                    response_recv_wall_ms=B._f((timing or {}).get("response_recv_wall_ms")),
                    rtt_ms=B._f((timing or {}).get("rtt_ms")),
                    attempted_exchange_call=attempted_exchange_call,
                    error=error,
                    context=ctx,
                )
        self.client.request = request

    def place(self, ticker, d, cur, elapsed):
        try:
            return super().place(ticker, d, cur, elapsed)
        except PresendFreshnessAbort as exc:
            self.v12["presend_raw_barrier_blocks"] += 1
            self._lat(
                "CREATE_ABORTED_PRESEND_RAW_BARRIER",
                ticker=ticker,
                role=d.get("role"),
                side=d.get("side"),
                price=d.get("price"),
                error=repr(exc),
            )
            # No POST was attempted; retry only from a future loop state.
            return None

    def _terminal(self, result, state, *, safe=True, detail=None):
        self._lat(
            "FAST_INVALIDATION_TERMINAL",
            invalidation_id=result.get("invalidation_id"),
            ticker=result.get("ticker"),
            order_id=result.get("order_id"),
            role=result.get("role"),
            terminal_state=str(state),
            safe=bool(safe),
            detail=detail,
        )

    def _consume_fast(self, limit=100):
        """V12.1 retirement ordering + one explicit terminal event/invalidation."""
        for result in self.fast.drain_results(limit=limit):
            ticker = str(result.get("ticker") or "")
            oid = str(result.get("order_id") or "")
            self.v12["fast_results"] += 1

            tr = self.active.get(ticker)
            if not tr or str(tr.get("order_id") or "") != oid:
                self.v12["stale_fast_results"] += 1
                self.fast.clear_pending(ticker, oid)
                self._terminal(
                    result,
                    "ALREADY_RETIRED_BY_AUTHORITATIVE_PATH",
                    safe=True,
                )
                continue

            if not result.get("success"):
                self.v12["fast_fallbacks"] += 1
                self._api_context = {
                    "v12": "FAST_CANCEL_ERROR_FALLBACK_V12_2",
                    "ticker": ticker,
                    "order_id": oid,
                }
                try:
                    V11.V2OnlyProductionEngine.cancel_track(
                        self, ticker, "V12_2_FAST_CANCEL_ERROR_FALLBACK"
                    )
                except Exception as exc:
                    self._terminal(
                        result,
                        "FALLBACK_FAILED_FAIL_CLOSED",
                        safe=False,
                        detail=repr(exc),
                    )
                    raise
                finally:
                    self._api_context = {}

                self._publish()
                self.fast.clear_pending(ticker, oid)
                self._terminal(
                    result,
                    "FAST_ERROR_FALLBACK_VERIFIED_RETIRED",
                    safe=True,
                    detail=result.get("cancel_error"),
                )
                continue

            reconcile_start_ms = V12._wall_ms()
            old_fill = float(tr.get("last_fill", 0.0))
            submitted_qty = float(tr.get("qty", 0.0))
            before_pos = float(self.positions.get(ticker, 0.0))
            reduced_by = self._parse_reduced_by(
                result.get("cancel_body") or {}, submitted_qty
            )
            final_fill = min(
                submitted_qty,
                max(old_fill, submitted_qty - reduced_by),
            )
            tr["last_fill"] = final_fill

            self._api_context = {
                "v12": "FAST_CANCEL_RECONCILE_V12_2",
                "ticker": ticker,
                "order_id": oid,
            }
            try:
                self.record_fills(tr)
                after_pos = self.refresh_position(ticker)
            finally:
                self._api_context = {}

            reconcile_done_ms = V12._wall_ms()
            response_ms = B._f(result.get("response_recv_wall_ms"))
            response_to_reconcile_ms = (
                reconcile_start_ms - response_ms
                if np.isfinite(response_ms)
                else np.nan
            )

            B._append(
                self.orders,
                {
                    "time": B._iso(),
                    "action": "CANCEL_V12_2_PRIORITY_FAST_PATH",
                    "ticker": ticker,
                    "reason": result.get("reason"),
                    "track": tr,
                    "invalidation_id": result.get("invalidation_id"),
                    "obsolete_receipt_wall_ms": result.get("obsolete_receipt_wall_ms"),
                    "cancel_body": result.get("cancel_body"),
                    "cancel_timing": result.get("cancel_timing"),
                    "obsolete_to_cancel_send_ms": result.get("obsolete_to_cancel_send_ms"),
                    "detect_to_cancel_send_ms": result.get("detect_to_cancel_send_ms"),
                    "response_to_reconcile_start_ms": response_to_reconcile_ms,
                    "reconcile_duration_ms": reconcile_done_ms - reconcile_start_ms,
                    "old_fill": old_fill,
                    "final_fill": final_fill,
                    "position_before": before_pos,
                    "position_after": after_pos,
                },
            )

            self.active.pop(ticker, None)
            self.counts["cancels"] += 1
            self.v12["fast_success"] += 1

            raced_fill = (
                final_fill > old_fill + B.EPS
                or abs(after_pos - before_pos) > B.EPS
            )
            if raced_fill:
                self.v12["fast_raced_fill"] += 1
                self.barrier[ticker] = self.book_version[ticker]
                self.counts["fill_events"] += 1
                self.emit(
                    "FILL",
                    ticker,
                    role=tr["role"],
                    side=tr["side"],
                    qty=max(0.0, final_fill - old_fill),
                    position=after_pos,
                    source="v12_2_priority_fast_cancel",
                )

            self._publish()
            self.fast.clear_pending(ticker, oid)
            self._lat(
                "FAST_CANCEL_RECONCILED",
                ticker=ticker,
                order_id=oid,
                invalidation_id=result.get("invalidation_id"),
                cancel_response_recv_wall_ms=response_ms,
                reconcile_start_wall_ms=reconcile_start_ms,
                reconcile_done_wall_ms=reconcile_done_ms,
                response_to_reconcile_start_ms=response_to_reconcile_ms,
                reconcile_duration_ms=reconcile_done_ms - reconcile_start_ms,
                raced_fill=raced_fill,
                position_after=after_pos,
                v12_2_terminal_accounting=True,
            )
            self._terminal(
                result,
                "FAST_DELETE_VERIFIED_RECONCILED",
                safe=True,
            )

    def _v12_metrics(self):
        out = super()._v12_metrics()
        out.update(
            {
                "live_version": LIVE_VERSION,
                "freshness_arch_version": FRESHNESS_ARCH_VERSION,
                "execution_parent": EXECUTION_PARENT,
                "presend_eof_timeout_ms": PRESEND_EOF_TIMEOUT_MS,
                "presend_raw_barrier_blocks": int(
                    self.v12["presend_raw_barrier_blocks"]
                ),
                "v12_2_binary_raw_offset_barrier": True,
                "v12_2_worker_sessions_prewarmed": True,
                "v12_2_terminal_accounting": True,
                "v12_2_lifecycle_aware_gate": True,
            }
        )
        return out


def _post_v12_2(client, payload):
    """V11 idempotent create, but a pre-send freshness block never becomes a retry."""
    cid = str(payload["client_order_id"])
    ticker = str(payload["ticker"])
    first_error = None

    try:
        body, timing = client.post("/portfolio/events/orders", payload)
        V11._CREATE_DIAG["create_201_success"] += 1
        return body, timing
    except PresendFreshnessAbort:
        # The request wrapper certifies attempted_exchange_call=False before
        # raising this control-flow exception.
        raise
    except Exception as exc:
        first_error = exc
        code = V11._http_code(exc)
        V11._CREATE_DIAG["create_initial_errors"] += 1
        if code == 409:
            V11._CREATE_DIAG["create_409_conflicts"] += 1

    row, rec = V11._recover_client_order(client, payload, timeout_s=0.90)
    if row is not None:
        V11._CREATE_DIAG["create_recovered_without_resubmit"] += 1
        return V11._recovered_create_body(row, payload), {
            "v11_recovered": True,
            "phase": "after_initial_error",
            "client_order_id": cid,
            "ticker": ticker,
            "initial_error": repr(first_error),
            **rec,
        }

    first_code = V11._http_code(first_error)
    if first_code == 409:
        row, rec2 = V11._recover_client_order(client, payload, timeout_s=3.50)
        if row is not None:
            V11._CREATE_DIAG["create_recovered_after_409"] += 1
            return V11._recovered_create_body(row, payload), {
                "v11_recovered": True,
                "phase": "409_poll_only",
                "client_order_id": cid,
                "ticker": ticker,
                "initial_error": repr(first_error),
                **rec2,
            }
        V11._CREATE_DIAG["create_unrecoverable_409"] += 1
        raise RuntimeError(
            "V12.2 create fail-closed: duplicate client_order_id cannot be recovered; "
            f"cid={cid} ticker={ticker} error={first_error!r} recovery={rec2}"
        )

    if first_code is not None and 400 <= first_code < 500 and first_code != 429:
        V11._CREATE_DIAG["create_permanent_errors"] += 1
        raise RuntimeError(f"V12.2 order create rejected without retry: {first_error!r}")

    V11._CREATE_DIAG["create_resubmits"] += 1
    time.sleep(0.12)
    try:
        body, timing = client.post("/portfolio/events/orders", payload)
        V11._CREATE_DIAG["create_resubmit_201_success"] += 1
        timing = dict(timing or {})
        timing.update(
            {
                "v11_resubmitted_same_client_order_id": True,
                "initial_error": repr(first_error),
            }
        )
        return body, timing
    except PresendFreshnessAbort as exc:
        # An earlier POST was ambiguous. We cannot safely convert this to a benign
        # skipped create because the first request may surface later. Fail closed;
        # the order-group shutdown remains the final orphan-order authority.
        raise RuntimeError(
            "V12.2 fail-closed: ambiguous create became stale before idempotent "
            f"resubmit; cid={cid} ticker={ticker} guard={exc!r}"
        )
    except Exception as exc:
        second_error = exc
        code = V11._http_code(exc)
        V11._CREATE_DIAG["create_resubmit_errors"] += 1
        if code == 409:
            V11._CREATE_DIAG["create_409_conflicts"] += 1

    row, rec3 = V11._recover_client_order(client, payload, timeout_s=3.50)
    if row is not None:
        V11._CREATE_DIAG["create_recovered_after_resubmit"] += 1
        return V11._recovered_create_body(row, payload), {
            "v11_recovered": True,
            "phase": "after_single_resubmit",
            "client_order_id": cid,
            "ticker": ticker,
            "initial_error": repr(first_error),
            "second_error": repr(second_error),
            **rec3,
        }

    V11._CREATE_DIAG["create_unrecoverable"] += 1
    raise RuntimeError(
        "V12.2 create fail-closed after one idempotent resubmit; "
        f"cid={cid} ticker={ticker} initial={first_error!r} "
        f"second={second_error!r} recovery={rec3}"
    )


def _read_jsonl(path):
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                if isinstance(x, dict):
                    rows.append(x)
    except FileNotFoundError:
        pass
    return rows


def _stats(vals):
    return V12._stats(vals)


def audit_stage(session_dir, *, show=True, write_result=True):
    """Lifecycle-aware V12.2 Q1/Q5/Q10 execution audit. No API calls."""
    session = Path(session_dir).resolve()
    events = _read_jsonl(session / "latency_events_v12.jsonl")
    final = B._read(session / "final_summary.json", {}) or {}
    base = V12.audit_v12_smoke(session, show=False, write_result=False)

    invalidations = [r for r in events if r.get("event") == "FAST_INVALIDATION_DETECTED"]
    results = [r for r in events if r.get("event") == "FAST_CANCEL_RESULT"]
    success = [r for r in results if r.get("success") is True]
    terminals = [r for r in events if r.get("event") == "FAST_INVALIDATION_TERMINAL"]
    creates = [r for r in events if r.get("event") == "CREATE_SENT"]
    presend_blocks = [r for r in events if r.get("event") == "CREATE_ABORTED_PRESEND_RAW_BARRIER"]
    prewarm = [r for r in events if r.get("event") == "WATCHDOG_WORKERS_PREWARMED"]
    caught_up = [r for r in events if r.get("event") == "WATCHDOG_CAUGHT_UP"]
    json_errors = [r for r in events if r.get("event") == "WATCHDOG_JSON_ERROR"]

    create_by_oid = {
        str(r.get("order_id") or ""): r
        for r in creates
        if r.get("order_id")
    }

    lifecycle_rows = []
    for r in success:
        oid = str(r.get("order_id") or "")
        c = create_by_oid.get(oid)
        if not c:
            continue
        obs = B._f(r.get("obsolete_receipt_wall_ms"))
        csend = B._f(c.get("request_send_wall_ms"))
        crecv = B._f(c.get("response_recv_wall_ms"))
        xsend = B._f(r.get("request_send_wall_ms"))
        if np.isfinite(obs) and np.isfinite(csend) and obs <= csend + 0.001:
            lifecycle = "PRE_SEND_STALE"
        elif np.isfinite(obs) and np.isfinite(crecv) and obs <= crecv + 0.001:
            lifecycle = "CREATE_FLIGHT_AFTER_SEND"
        else:
            lifecycle = "RESTING"
        lifecycle_rows.append(
            {
                "lifecycle": lifecycle,
                "obsolete_to_cancel_send_ms": xsend - obs,
                "create_response_to_cancel_send_ms": xsend - crecv,
                "detect_to_cancel_send_ms": B._f(r.get("detect_to_cancel_send_ms")),
            }
        )

    life = pd.DataFrame(lifecycle_rows)
    resting = life.loc[life["lifecycle"] == "RESTING"] if len(life) else pd.DataFrame()
    flight = life.loc[life["lifecycle"] == "CREATE_FLIGHT_AFTER_SEND"] if len(life) else pd.DataFrame()
    presend = life.loc[life["lifecycle"] == "PRE_SEND_STALE"] if len(life) else pd.DataFrame()

    resting_stats = _stats(
        resting["obsolete_to_cancel_send_ms"] if len(resting) else []
    )
    flight_response_stats = _stats(
        flight["create_response_to_cancel_send_ms"] if len(flight) else []
    )

    inv_ids = {str(r.get("invalidation_id") or "") for r in invalidations}
    terminal_by_id = {
        str(r.get("invalidation_id") or ""): r
        for r in terminals
        if r.get("invalidation_id")
    }
    missing_terminal = sorted(x for x in inv_ids if x not in terminal_by_id)
    unsafe_terminal = [
        r for r in terminals
        if str(r.get("invalidation_id") or "") in inv_ids
        and r.get("safe") is not True
    ]
    terminal_completion_pass = bool(
        len(terminal_by_id) == len(inv_ids)
        and not missing_terminal
        and not unsafe_terminal
    )

    superseded = sum(bool(r.get("superseded_at_send_detected")) for r in creates)
    create_freshness_pass = superseded == 0

    safety_pass = bool(
        final
        and final.get("flat_verified") is True
        and final.get("strategy_resting_orders_zero") is True
        and final.get("last_error") in (None, "")
    )
    prewarm_pass = bool(
        prewarm
        and int(prewarm[-1].get("successes") or 0) == V12.WATCHDOG_WORKERS
        and not prewarm[-1].get("errors")
    )
    watchdog_pass = bool(caught_up and not json_errors and prewarm_pass)

    metrics = final.get("v12_metrics") or {}
    max_actionable = B._f(metrics.get("max_actionable_tickers"))
    actionable_set_pass = bool(
        not np.isfinite(max_actionable)
        or max_actionable <= V12.MAX_ACTIONABLE_EXPECTED
    )

    resting_latency_observed = resting_stats["n"] > 0
    resting_cancel_latency_pass = bool(
        resting_latency_observed
        and resting_stats["median"] <= V12.TARGET_CANCEL_SEND_MS
        and resting_stats["p95"] <= V12.P95_CANCEL_SEND_MS
        and resting_stats["max"] <= V12.HARD_CANCEL_SEND_MS
    )

    create_flight_containment_pass = bool(
        flight_response_stats["n"] == 0
        or (
            flight_response_stats["p95"] <= CREATE_FLIGHT_P95_RESPONSE_TO_CANCEL_MS
            and flight_response_stats["max"] <= CREATE_FLIGHT_MAX_RESPONSE_TO_CANCEL_MS
        )
    )

    promotion = bool(
        safety_pass
        and watchdog_pass
        and terminal_completion_pass
        and create_freshness_pass
        and actionable_set_pass
        and resting_cancel_latency_pass
        and create_flight_containment_pass
    )

    interpretation = (
        "PASS"
        if promotion
        else (
            "INCOMPLETE_LATENCY_PROOF_NO_RESTING_INVALIDATION"
            if safety_pass and watchdog_pass and not resting_latency_observed
            else "FAIL_INVESTIGATE"
        )
    )

    terminal_counts = Counter(str(r.get("terminal_state") or "") for r in terminals)
    out = dict(base)
    out.update(
        {
            "live_version": LIVE_VERSION,
            "counts": {
                **(base.get("counts") or {}),
                "fast_invalidations": len(invalidations),
                "fast_cancel_results": len(results),
                "fast_cancel_success_results": len(success),
                "fast_cancel_initial_error_results": len(results) - len(success),
                "fast_terminal_events": len(terminals),
                "fast_terminal_states": dict(terminal_counts),
                "missing_terminal_invalidations": len(missing_terminal),
                "unsafe_terminal_invalidations": len(unsafe_terminal),
                "presend_raw_barrier_blocks": len(presend_blocks),
                "superseded_at_send_detected": superseded,
                "resting_successful_invalidations": int(len(resting)),
                "create_flight_successful_invalidations": int(len(flight)),
                "pre_send_stale_successful_invalidations": int(len(presend)),
            },
            "lifecycle_latency": {
                "resting_obsolete_to_cancel_send_ms": resting_stats,
                "create_flight_response_to_cancel_send_ms": flight_response_stats,
                "pre_send_stale_count": int(len(presend)),
            },
            "gates": {
                "safety_pass": safety_pass,
                "watchdog_and_worker_prewarm_pass": watchdog_pass,
                "terminal_completion_pass": terminal_completion_pass,
                "create_freshness_pass": create_freshness_pass,
                "actionable_set_pass": actionable_set_pass,
                "resting_cancel_latency_observed": resting_latency_observed,
                "resting_cancel_latency_pass": resting_cancel_latency_pass,
                "create_flight_containment_pass": create_flight_containment_pass,
                "promotion_ready_for_larger_smoke": promotion,
            },
            "interpretation": interpretation,
        }
    )

    if write_result:
        B._atomic(session / "v12_2_smoke_latency_audit.json", out)

    if show:
        print("=" * 108)
        print("V12.2 LIFECYCLE-AWARE LIVE EXECUTION AUDIT — READ ONLY")
        print("=" * 108)
        print("Session:                          ", session)
        print("PnL:                              ", (out.get("final") or {}).get("account_pnl_usd"))
        print("Flat / zero resting:              ", safety_pass)
        print("Watchdog + 9 worker prewarm:      ", watchdog_pass)
        print("Invalidations / terminal events:  ", len(invalidations), "/", len(terminals))
        print("Terminal states:                  ", dict(terminal_counts))
        print("Superseded CREATEs at send:       ", superseded)
        print("Pre-send barrier blocks:          ", len(presend_blocks))
        print("Resting obsolete -> cancel send:  ", resting_stats)
        print("Create-flight response -> cancel: ", flight_response_stats)
        print("Gates:                            ", out["gates"])
        print("Interpretation:                   ", interpretation)
        print("ORDERS SENT BY AUDIT: NO | EXCHANGE API CALLED BY AUDIT: NO")

    return out


def static_self_check(*, show=True):
    base = V121.static_self_check(show=False)
    checks = dict(base.get("checks") or {})
    checks.update(
        {
            "execution_parent_is_v12_1": EXECUTION_PARENT == V121.LIVE_VERSION,
            "binary_offset_watchdog_subclass": issubclass(
                BarrierFastCancelWatchdog, V121.FixedFastCancelWatchdog
            ),
            "v12_2_engine_subclass": issubclass(
                PriorityFreshnessEngine122, V121.PriorityFreshnessEngine121
            ),
            "presend_timeout_positive": PRESEND_EOF_TIMEOUT_MS > 0,
            "create_flight_slo_ordered": (
                0 < CREATE_FLIGHT_P95_RESPONSE_TO_CANCEL_MS
                <= CREATE_FLIGHT_MAX_RESPONSE_TO_CANCEL_MS
            ),
            "strategy_changed": False,
        }
    )
    out = {
        "time": B._iso(),
        "live_version": LIVE_VERSION,
        "checks": checks,
        "pass": all(checks.values()),
        "orders_sent": False,
        "exchange_api_called": False,
    }
    if show:
        print("V12.2 STATIC SELF CHECK:", "PASS" if out["pass"] else "FAIL")
        for k, v in checks.items():
            print(f"  {k:<48} {v}")
        print("  ORDERS SENT: NO | EXCHANGE API CALLED: NO")
    return out


def _write_bundle(session: Path, cfg):
    session = Path(session).resolve()
    V121._write_bundle(session, cfg)
    spec = {
        "time": B._iso(),
        "live_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_parent": RECORDING_PARENT,
        "freshness_arch_version": FRESHNESS_ARCH_VERSION,
        "presend_eof_timeout_ms": PRESEND_EOF_TIMEOUT_MS,
        "worker_sessions_prewarmed": True,
        "terminal_accounting": True,
        "resting_cancel_slo_ms": {
            "median": V12.TARGET_CANCEL_SEND_MS,
            "p95": V12.P95_CANCEL_SEND_MS,
            "max": V12.HARD_CANCEL_SEND_MS,
        },
        "create_flight_response_to_cancel_slo_ms": {
            "p95": CREATE_FLIGHT_P95_RESPONSE_TO_CANCEL_MS,
            "max": CREATE_FLIGHT_MAX_RESPONSE_TO_CANCEL_MS,
        },
        "strategy_mechanics_changed": False,
    }
    B._atomic(session / "live_execution_spec_v12_2.json", spec)


def _run_process(session, cfg):
    session = Path(session).resolve()
    _write_bundle(session, cfg)
    client = B.Q1.LiveClient()
    diag = V3._install(client)
    B._atomic(session / "balance_semantics.json", diag)
    B._post = _post_v12_2
    B._start_recorder = V4._start_recorder_fixed
    B.LiveEngine = PriorityFreshnessEngine122
    B._run_process(session, cfg)


def _stage_gate(session_dir, *, expected_q, expected_mode, show=True):
    session = Path(session_dir).resolve()
    cfg = B._read(session / "process_config.json", {}) or {}
    final = B._read(session / "final_summary.json", {}) or {}
    audit = audit_stage(session, show=show, write_result=True)
    q = B._f(cfg.get("quote_size"), np.nan)
    mode = str(cfg.get("mode") or "")
    passed = bool(
        final
        and np.isfinite(q)
        and abs(q - float(expected_q)) <= B.EPS
        and mode == str(expected_mode)
        and (audit.get("gates") or {}).get("promotion_ready_for_larger_smoke")
    )
    out = {
        "session_dir": str(session),
        "actual_q": q,
        "expected_q": float(expected_q),
        "actual_mode": mode,
        "expected_mode": expected_mode,
        "completed": bool(final),
        "audit": audit,
        "pass": passed,
    }
    if show:
        print("=" * 96)
        print("V12.2 STAGE PROMOTION GATE")
        print("=" * 96)
        print("session:    ", session)
        print("quote size: ", q, "expected", expected_q)
        print("mode:       ", mode)
        print("completed:  ", bool(final))
        print("GATE:       ", "PASS" if passed else "FAIL")
    return out


def gate_q1_for_q5(session_dir, *, show=True):
    return _stage_gate(
        session_dir,
        expected_q=B.SMOKE_Q,
        expected_mode="SMOKE_Q1_ONE_WINDOW",
        show=show,
    )


def gate_q5_for_q10(session_dir, *, show=True):
    return _stage_gate(
        session_dir,
        expected_q=Q5_QTY,
        expected_mode="LIVE_Q5_1H",
        show=show,
    )


def _launch(*, mode, q, hours, max_loss, min_equity, arm, expected_arm,
            prior_session=None, prior_expected_q=None, prior_expected_mode=None):
    if str(arm) != str(expected_arm):
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly."
        )

    if prior_session is not None:
        gate = _stage_gate(
            prior_session,
            expected_q=prior_expected_q,
            expected_mode=prior_expected_mode,
            show=True,
        )
        if not gate["pass"]:
            raise RuntimeError("Prior V12.2 stage did not pass. Refusing to arm.")

    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    static = static_self_check(show=True)
    if not static["pass"]:
        raise RuntimeError(f"V12.2 static self-check failed: {static}")

    V3._calibrated_preflight(
        quote_size=float(q),
        runtime_hours=float(hours),
        max_loss_usd=float(max_loss),
        min_equity_usd=float(min_equity),
        mode=str(mode),
        save_dir=None,
        show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_{str(mode).lower()}_v12_2").resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": str(mode),
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": LIVE_VERSION,
        "staged_launcher_version": STAGED_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_parent": RECORDING_PARENT,
        "freshness_arch_version": FRESHNESS_ARCH_VERSION,
        "recording_version": V10.RECORDING_VERSION,
        "comparison_schema_version": V10.COMPARISON_SCHEMA_VERSION,
        "order_api_safety_version": V11.ORDER_API_SAFETY_VERSION,
        "prior_stage_session": str(Path(prior_session).resolve()) if prior_session else None,
    }
    B._atomic(session / "process_config.json", cfg)
    _write_bundle(session, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v12_2",
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

    B._atomic(
        B.CONTROL_PATH,
        {
            "live_version": LIVE_VERSION,
            "running": True,
            "pid": p.pid,
            "session_dir": str(session),
            "mode": str(mode),
            "started_at": B._iso(),
            "config": cfg,
            "log_path": str(log),
        },
    )

    deadline = time.time() + 90.0
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-20000:] if log.exists() else ""
            raise RuntimeError(
                f"Live V12.2 process exited during startup rc={p.returncode}\n{tail}"
            )
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-20000:] if log.exists() else ""
        raise RuntimeError(f"Live V12.2 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V12.2 STAGE ARMED")
    print("  stage:   ", mode)
    print("  Q:       ", q)
    print("  hours:   ", hours)
    print("  session: ", session)
    print("  pid:     ", p.pid)
    print("  CREATE:  raw EOF catch-up + immediate revalidation before HTTP")
    print("  CANCEL:  9 read-only-prewarmed independent worker sessions")
    return B.live_status(show=False)


def start_live_q1(*, arm_phrase=None,
                  max_start_loss_usd=B.LOSS_LIMIT_USD,
                  min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    return _launch(
        mode="SMOKE_Q1_ONE_WINDOW",
        q=B.SMOKE_Q,
        hours=1.0,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected_arm=B.SMOKE_ARM,
    )


def start_live_q5_after_q1(*, prior_q1_session, arm_phrase=None,
                           max_start_loss_usd=B.LOSS_LIMIT_USD,
                           min_start_equity_usd=Q5_MIN_EQUITY_USD):
    return _launch(
        mode="LIVE_Q5_1H",
        q=Q5_QTY,
        hours=Q5_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected_arm=Q5_ARM,
        prior_session=prior_q1_session,
        prior_expected_q=B.SMOKE_Q,
        prior_expected_mode="SMOKE_Q1_ONE_WINDOW",
    )


def start_live_q10_after_q5(*, prior_q5_session, arm_phrase=None,
                            max_start_loss_usd=B.LOSS_LIMIT_USD,
                            min_start_equity_usd=B.FULL_MIN_EQUITY):
    return _launch(
        mode="LIVE_Q10_24H",
        q=B.FULL_Q,
        hours=B.FULL_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected_arm=Q10_ARM,
        prior_session=prior_q5_session,
        prior_expected_q=Q5_QTY,
        prior_expected_mode="LIVE_Q5_1H",
    )


def live_preflight(*, quote_size=B.SMOKE_Q, runtime_hours=1.0,
                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                   min_start_equity_usd=B.SMOKE_MIN_EQUITY, show=True):
    return V3.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
    )


def account_safety_check(*, show=True):
    return V11.account_safety_check(show=show)


def live_status(*, show=True, tail_lines=20):
    return B.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    return B.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    ap.add_argument("--audit-session")
    a = ap.parse_args()
    if a.audit_session:
        audit_stage(a.audit_session, show=True, write_result=True)
        return
    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}
        _run_process(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "STAGED_VERSION",
    "PriorityFreshnessEngine122",
    "BarrierFastCancelWatchdog",
    "PresendFreshnessAbort",
    "static_self_check",
    "audit_stage",
    "gate_q1_for_q5",
    "gate_q5_for_q10",
    "start_live_q1",
    "start_live_q5_after_q1",
    "start_live_q10_after_q5",
    "live_preflight",
    "account_safety_check",
    "live_status",
    "kill_and_flatten_live",
]
