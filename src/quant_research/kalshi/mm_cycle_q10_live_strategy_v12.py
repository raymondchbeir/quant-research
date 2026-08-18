from __future__ import annotations

"""V12: V11 order safety + independent priority freshness/cancel watchdog.

Frozen Candidate-C strategy mechanics are unchanged. V12 changes live execution
architecture and instrumentation only. The first real V12 run is Q1 one-window;
Q10 is deliberately disabled until a fresh Q1 latency audit passes.

Core V12 invariants
-------------------
1. A dedicated watchdog tails the same persisted V5 raw book independently from
   the synchronous strategy/risk loop.
2. The watchdog may CANCEL an already-authorized resting strategy order, but it
   can never CREATE an order or decide inventory.
3. No CREATE is allowed until the watchdog has caught up to the raw file and can
   certify the decision source row is not already superseded.
4. A final watchdog certification is repeated immediately before POST.
5. Priority cancel workers own independent LiveClient sessions, so a main-thread
   balance/position/queue/resting-order HTTP request cannot occupy the cancel
   request's Python call path.
6. Main-thread reconciliation remains authoritative for fills and positions.
7. Historical/finalized ticker rows are pruned from the actionable set.
8. Every quote-critical latency component is persisted for Q1 audit.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v4 as V4
from . import mm_cycle_q10_live_strategy_v6 as V6
from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_cycle_q10_live_strategy_v11 as V11

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V12"
EXECUTION_PARENT = V11.LIVE_VERSION
RECORDING_PARENT = V10.LIVE_VERSION
FRESHNESS_ARCH_VERSION = "MM_CYCLE_Q10_PRIORITY_FRESHNESS_V12"

# Engineering SLOs inferred from the interrupted V11 execution forensics.
# These are operational latency targets, NOT alpha/strategy parameters.
TARGET_CANCEL_SEND_MS = 25.0
P95_CANCEL_SEND_MS = 50.0
HARD_CANCEL_SEND_MS = 100.0

# One cancel worker per frozen series avoids queueing a ninth simultaneous
# invalidation behind four ~80-90 ms HTTP requests.
WATCHDOG_WORKERS = 9
WATCHDOG_SLEEP_S = 0.001
LOOP_SLEEP_S = 0.002
MAX_ACTIONABLE_EXPECTED = 18  # at most two adjacent 9-series windows at a boundary


def _wall_ms():
    return time.time_ns() / 1e6


def _perf_ms():
    return time.perf_counter_ns() / 1e6


def _receipt_ms(row):
    try:
        t = pd.to_datetime((row or {}).get("receipt_time"), utc=True, errors="coerce")
        return np.nan if pd.isna(t) else float(t.timestamp() * 1000.0)
    except Exception:
        return np.nan


def _top(row):
    try:
        return B.OOS._top_state(row or {})
    except Exception:
        return None


def _track_invalidated(track, row):
    """Pure public-book validity test for an already-authorized quote.

    This function intentionally does not inspect or mutate account inventory.
    ENTRY validity is exact frozen Candidate-C role/side/price validity. EXIT
    validity checks that the already-authorized exit is still at the public BBO;
    the main thread remains authoritative for exit quantity via actual position.
    """
    e = B._f((row or {}).get("elapsed_s"), np.nan)
    if not np.isfinite(e) or not (0.0 <= e < 300.0):
        return True, "OUTSIDE_M0_M5"

    cur = _top(row)
    if cur is None:
        return True, "INVALID_BOOK"

    role = str((track or {}).get("role") or "").upper()
    side = str((track or {}).get("side") or "").lower()
    px = B._f((track or {}).get("price"), np.nan)

    if role == "ENTRY":
        s = B.OOS._entry_side(cur)
        if s is None:
            return True, "ENTRY_FILTER_NONE"
        desired_side = str(s).lower()
        desired_px = float(cur["bid"] if desired_side == "bid" else cur["ask"])
        if desired_side != side:
            return True, "SIDE_CHANGE"
        if not np.isfinite(px) or abs(desired_px - px) > 1e-9:
            return True, "PRICE_CHANGE"
        return False, None

    if role == "EXIT":
        if side not in {"bid", "ask"}:
            return True, "INVALID_TRACK_SIDE"
        if not np.isfinite(px):
            return True, "INVALID_TRACK_PRICE"
        desired_px = float(cur["bid"] if side == "bid" else cur["ask"])
        changed = abs(desired_px - px) > 1e-9
        return changed, ("PRICE_CHANGE" if changed else None)

    return True, "INVALID_TRACK_ROLE"


def _http_class(method, path):
    m = str(method).upper()
    p = str(path)
    if m == "POST" and p == "/portfolio/events/orders":
        return "CREATE_POST"
    if m == "DELETE" and p == "/portfolio/events/orders/batched":
        return "CANCEL_BATCH"
    if m == "DELETE" and p.startswith("/portfolio/events/orders/"):
        return "CANCEL_REQUEST"
    if m == "GET" and p == "/portfolio/balance":
        return "BALANCE_POLL"
    if m == "GET" and p == "/portfolio/positions":
        return "POSITION_POLL"
    if m == "GET" and p == "/portfolio/orders":
        return "RESTING_ORDER_POLL"
    if m == "GET" and p == "/portfolio/orders/queue_positions":
        return "QUEUE_POLL"
    if m == "GET" and p == "/portfolio/fills":
        return "FILL_READ"
    return f"{m}_OTHER"


class FastCancelWatchdog:
    """Independent raw-book tail + priority cancel dispatcher.

    Startup is deliberately two-phase. Existing file content is first consumed to
    the current EOF with cancellation disabled. Only after WATCHDOG_CAUGHT_UP is
    emitted can CREATEs be certified or active orders be invalidated. This avoids
    treating an old pre-order raw row as a new quote invalidation.
    """

    def __init__(self, engine):
        self.engine = engine
        self.raw_path = engine.session / "raw_capture" / "book_top3_events.jsonl"
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.caught_up = threading.Event()
        self.lock = threading.RLock()

        # Snapshot copies only. The worker never touches engine.active directly.
        self.active = {}
        self.latest = {}
        self.pending = {}
        self.history = defaultdict(lambda: deque(maxlen=256))
        self.seq = 0

        self.results = Queue()
        self.executor = ThreadPoolExecutor(
            max_workers=WATCHDOG_WORKERS,
            thread_name_prefix="v12-priority-cancel",
        )
        self.thread_clients = threading.local()
        self.thread = threading.Thread(
            target=self._run,
            name="v12-raw-watchdog",
            daemon=True,
        )

        self.rows_seen = 0
        self.invalidations = 0
        self.cancel_submissions = 0
        self.cancel_errors = 0
        self.max_raw_to_detect_ms = 0.0
        self.max_obsolete_to_send_ms = 0.0

    def start(self):
        self.thread.start()

    def ready(self):
        return self.caught_up.is_set()

    def stop(self, wait_s=2.0):
        self.stop_event.set()
        self.wake_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=float(wait_s))
        # Do not block fail-closed shutdown on an 8-second HTTP timeout. Any
        # in-flight worker may finish, while base shutdown's order-group trigger
        # remains the final resting-order safety authority.
        self.executor.shutdown(wait=False, cancel_futures=False)

    def publish_active(self, active):
        snap = {}
        for ticker, tr in (active or {}).items():
            t = str(ticker)
            snap[t] = {
                k: tr.get(k)
                for k in ("order_id", "role", "side", "price", "qty", "cid")
            }
            snap[t]["ticker"] = t
        with self.lock:
            self.active = snap
        self.wake_event.set()

    def latest_snapshot(self, ticker):
        with self.lock:
            x = self.latest.get(str(ticker))
            return dict(x) if x else None

    def first_newer_before(self, ticker, source_ms, cutoff_ms):
        if not np.isfinite(B._f(source_ms)) or not np.isfinite(B._f(cutoff_ms)):
            return None
        with self.lock:
            hist = list(self.history.get(str(ticker)) or [])
        xs = [
            x for x in hist
            if np.isfinite(B._f(x.get("receipt_wall_ms")))
            and B._f(x.get("receipt_wall_ms")) > float(source_ms) + 0.001
            and B._f(x.get("receipt_wall_ms")) <= float(cutoff_ms) + 0.001
        ]
        return min(xs, key=lambda x: B._f(x.get("receipt_wall_ms"))) if xs else None

    def is_pending(self, ticker, oid=None):
        t = str(ticker)
        with self.lock:
            if oid is not None:
                return (t, str(oid)) in self.pending
            return any(k[0] == t for k in self.pending)

    def clear_pending(self, ticker, oid):
        with self.lock:
            self.pending.pop((str(ticker), str(oid)), None)

    def drain_results(self, limit=100):
        out = []
        for _ in range(int(limit)):
            try:
                out.append(self.results.get_nowait())
            except Empty:
                break
        return out

    def _client(self):
        client = getattr(self.thread_clients, "client", None)
        if client is None:
            client = B.Q1.LiveClient()
            self.thread_clients.client = client
        return client

    def _run(self):
        while not self.stop_event.is_set() and not self.raw_path.exists():
            self.stop_event.wait(0.005)
        if self.stop_event.is_set():
            return

        start_ms = _wall_ms()
        with self.raw_path.open("r", encoding="utf-8") as fh:
            while not self.stop_event.is_set():
                pos = fh.tell()
                line = fh.readline()

                # Never parse a writer's partial final JSONL record.
                if line and not line.endswith("\n"):
                    fh.seek(pos)
                    self.wake_event.wait(WATCHDOG_SLEEP_S)
                    self.wake_event.clear()
                    continue

                if line:
                    try:
                        row = json.loads(line)
                    except Exception as exc:
                        self.engine._lat("WATCHDOG_JSON_ERROR", error=repr(exc))
                        continue
                    if isinstance(row, dict):
                        self._process_row(row)
                    continue

                # First observed EOF means the initial backlog is consumed. Only
                # now may the watchdog cancel or certify a CREATE.
                if not self.caught_up.is_set():
                    self.caught_up.set()
                    self.engine._lat(
                        "WATCHDOG_CAUGHT_UP",
                        rows_seen=self.rows_seen,
                        startup_catchup_ms=_wall_ms() - start_ms,
                    )

                self._recheck_active()
                self.wake_event.wait(WATCHDOG_SLEEP_S)
                self.wake_event.clear()

    def _process_row(self, row):
        ticker = str(row.get("ticker") or "")
        if not ticker:
            return

        detect_ms = _wall_ms()
        receipt_ms = _receipt_ms(row)
        raw_to_detect_ms = (
            detect_ms - receipt_ms
            if np.isfinite(receipt_ms)
            else np.nan
        )

        self.rows_seen += 1
        if np.isfinite(raw_to_detect_ms):
            self.max_raw_to_detect_ms = max(
                self.max_raw_to_detect_ms,
                float(raw_to_detect_ms),
            )

        with self.lock:
            self.seq += 1
            item = {
                "ticker": ticker,
                "seq": self.seq,
                "row": row,
                "receipt_wall_ms": receipt_ms,
                "watchdog_detect_wall_ms": detect_ms,
                "raw_to_watchdog_ms": raw_to_detect_ms,
            }
            self.latest[ticker] = item
            self.history[ticker].append(dict(item))
            tr = dict(self.active.get(ticker) or {})

        # During initial catch-up we only populate state/history. We never cancel
        # from a historical row encountered before current EOF was established.
        if tr and self.caught_up.is_set():
            invalid, reason = _track_invalidated(tr, row)
            if invalid:
                self._submit(tr, item, reason)

    def _recheck_active(self):
        if not self.caught_up.is_set():
            return
        with self.lock:
            pairs = [
                (dict(tr), dict(self.latest[ticker] or {}))
                for ticker, tr in self.active.items()
                if self.latest.get(ticker)
            ]
        for tr, item in pairs:
            invalid, reason = _track_invalidated(tr, item.get("row") or {})
            if invalid:
                self._submit(tr, item, reason)

    def _submit(self, tr, item, reason):
        ticker = str(tr.get("ticker") or "")
        oid = str(tr.get("order_id") or "")
        if not ticker or not oid:
            return
        key = (ticker, oid)

        with self.lock:
            current = self.active.get(ticker) or {}
            if str(current.get("order_id") or "") != oid:
                return
            if key in self.pending:
                return

            detect_ms = _wall_ms()
            ctx = {
                "invalidation_id": uuid.uuid4().hex,
                "ticker": ticker,
                "order_id": oid,
                "role": tr.get("role"),
                "side": tr.get("side"),
                "price": B._f(tr.get("price")),
                "qty": B._f(tr.get("qty")),
                "reason": str(reason),
                "raw_seq": item.get("seq"),
                "obsolete_receipt_wall_ms": item.get("receipt_wall_ms"),
                "watchdog_detect_wall_ms": detect_ms,
                "watchdog_raw_to_detect_ms": item.get("raw_to_watchdog_ms"),
            }
            self.pending[key] = ctx
            self.invalidations += 1

        self.engine._lat("FAST_INVALIDATION_DETECTED", **ctx)
        self.executor.submit(self._cancel_task, ctx)

    def _cancel_task(self, ctx):
        oid = str(ctx["order_id"])
        body = None
        timing = None
        error = None
        task_start_ms = _wall_ms()

        try:
            body, timing = self._client().delete(
                f"/portfolio/events/orders/{oid}",
                params={"subaccount": 0, "exchange_index": 0},
            )
            self.cancel_submissions += 1
        except Exception as exc:
            error = repr(exc)
            self.cancel_errors += 1

        send_ms = B._f((timing or {}).get("request_send_wall_ms"))
        recv_ms = B._f((timing or {}).get("response_recv_wall_ms"))
        obsolete_ms = B._f(ctx.get("obsolete_receipt_wall_ms"))
        detect_ms = B._f(ctx.get("watchdog_detect_wall_ms"))

        obsolete_to_send_ms = (
            send_ms - obsolete_ms
            if np.isfinite(send_ms) and np.isfinite(obsolete_ms)
            else np.nan
        )
        detect_to_send_ms = (
            send_ms - detect_ms
            if np.isfinite(send_ms) and np.isfinite(detect_ms)
            else np.nan
        )
        if np.isfinite(obsolete_to_send_ms):
            self.max_obsolete_to_send_ms = max(
                self.max_obsolete_to_send_ms,
                float(obsolete_to_send_ms),
            )

        result = {
            **ctx,
            "task_start_wall_ms": task_start_ms,
            "task_done_wall_ms": _wall_ms(),
            "cancel_body": body,
            "cancel_timing": timing,
            "cancel_error": error,
            "request_send_wall_ms": send_ms,
            "response_recv_wall_ms": recv_ms,
            "obsolete_to_cancel_send_ms": obsolete_to_send_ms,
            "detect_to_cancel_send_ms": detect_to_send_ms,
            "cancel_rtt_ms": B._f((timing or {}).get("rtt_ms")),
            "success": error is None,
        }
        self.engine._lat("FAST_CANCEL_RESULT", **result)
        self.results.put(result)

    def metrics(self):
        with self.lock:
            pending = len(self.pending)
            active = len(self.active)
            latest = len(self.latest)
        return {
            "rows_seen": self.rows_seen,
            "invalidations": self.invalidations,
            "cancel_submissions": self.cancel_submissions,
            "cancel_errors": self.cancel_errors,
            "pending": pending,
            "active_published": active,
            "latest_tickers": latest,
            "caught_up": self.caught_up.is_set(),
            "max_raw_to_detect_ms": self.max_raw_to_detect_ms,
            "max_obsolete_to_cancel_send_ms": self.max_obsolete_to_send_ms,
            "workers": WATCHDOG_WORKERS,
        }


class PriorityFreshnessEngine(V11.V2OnlyProductionEngine):
    """V11 safety/recording with V12 quote-critical priority architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latency_log_v12 = self.session / "latency_events_v12.jsonl"
        self._lat_lock = threading.Lock()
        self._api_context = {}
        self._request_original = self.client.request
        self._instrument_client()
        self.v12 = Counter()
        self.max_actionable = 0
        self.fast = FastCancelWatchdog(self)
        self.fast.start()
        self._publish()

    def _lat(self, event, **kw):
        row = {
            "time": B._iso(),
            "wall_ms": _wall_ms(),
            "perf_ms": _perf_ms(),
            "event": str(event),
            "engine": LIVE_VERSION,
            **kw,
        }
        with self._lat_lock:
            B._append(self.latency_log_v12, row)

    def _instrument_client(self):
        """Measure every synchronous REST call made by the main engine client."""
        def request(method, path, *, params=None, payload=None, timeout=8.0):
            wall_start = _wall_ms()
            perf_start = _perf_ms()
            timing = None
            error = None
            ctx = dict(self._api_context or {})
            try:
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
                    http_class=_http_class(method, path),
                    call_start_wall_ms=wall_start,
                    local_call_ms=_perf_ms() - perf_start,
                    request_send_wall_ms=B._f((timing or {}).get("request_send_wall_ms")),
                    response_recv_wall_ms=B._f((timing or {}).get("response_recv_wall_ms")),
                    rtt_ms=B._f((timing or {}).get("rtt_ms")),
                    error=error,
                    context=ctx,
                )
        self.client.request = request

    def _publish(self):
        if hasattr(self, "fast"):
            self.fast.publish_active(self.active)

    def _consume_fast(self, limit=100):
        for result in self.fast.drain_results(limit=limit):
            ticker = str(result.get("ticker") or "")
            oid = str(result.get("order_id") or "")
            self.fast.clear_pending(ticker, oid)
            self.v12["fast_results"] += 1

            tr = self.active.get(ticker)
            if not tr or str(tr.get("order_id") or "") != oid:
                self.v12["stale_fast_results"] += 1
                continue

            if not result.get("success"):
                self.v12["fast_fallbacks"] += 1
                self._api_context = {
                    "v12": "FAST_CANCEL_ERROR_FALLBACK",
                    "ticker": ticker,
                    "order_id": oid,
                }
                try:
                    super().cancel_track(ticker, "V12_FAST_CANCEL_ERROR_FALLBACK")
                finally:
                    self._api_context = {}
                self._publish()
                continue

            reconcile_start_ms = _wall_ms()
            old_fill = float(tr.get("last_fill", 0.0))
            submitted_qty = float(tr.get("qty", 0.0))
            before_pos = float(self.positions.get(ticker, 0.0))
            reduced_by = self._parse_reduced_by(
                result.get("cancel_body") or {},
                submitted_qty,
            )
            final_fill = min(
                submitted_qty,
                max(old_fill, submitted_qty - reduced_by),
            )
            tr["last_fill"] = final_fill

            self._api_context = {
                "v12": "FAST_CANCEL_RECONCILE",
                "ticker": ticker,
                "order_id": oid,
            }
            try:
                self.record_fills(tr)
                after_pos = self.refresh_position(ticker)
            finally:
                self._api_context = {}

            reconcile_done_ms = _wall_ms()
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
                    "action": "CANCEL_V12_PRIORITY_FAST_PATH",
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
                    source="v12_priority_fast_cancel",
                )

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
            )
            self._publish()

    def cancel_track(self, ticker, reason):
        tr = self.active.get(ticker)
        if not tr:
            return False
        oid = str(tr.get("order_id") or "")

        if self.fast.is_pending(ticker, oid):
            self._consume_fast()
            tr = self.active.get(ticker)
            if not tr:
                return False
            oid = str(tr.get("order_id") or "")

            if self.fast.is_pending(ticker, oid):
                urgent = str(reason).upper()
                if (
                    urgent.startswith("M5")
                    or "SHUTDOWN" in urgent
                    or urgent.startswith("WALL_CLOCK")
                ):
                    deadline = time.time() + 0.15
                    while time.time() < deadline and self.fast.is_pending(ticker, oid):
                        time.sleep(0.002)
                        self._consume_fast()
                    if ticker not in self.active:
                        return False
                else:
                    # A priority singular V2 cancel is already in flight. Never
                    # launch a second normal cancel/replacement for the same order.
                    return False

        self._api_context = {
            "v12": "NORMAL_CANCEL",
            "ticker": ticker,
            "order_id": oid,
            "reason": reason,
        }
        try:
            raced = super().cancel_track(ticker, reason)
        finally:
            self._api_context = {}
        self._publish()
        return raced

    def reconcile(self, ticker, cur, elapsed):
        tr = self.active.get(ticker)
        if tr and self.fast.is_pending(ticker, tr.get("order_id")):
            return
        return super().reconcile(ticker, cur, elapsed)

    def poll_orders(self):
        """V6 all-resting poll, skipping orders already on priority cancel path."""
        now = time.time()
        if now - self.t_order < B.ORDER_POLL_S:
            return
        self.t_order = now

        resting, _ = B._resting(self.client)
        by_id = {str(r.get("order_id") or ""): r for r in resting}
        transitions = 0

        for ticker, tr in list(self.active.items()):
            oid = str(tr.get("order_id") or "")
            if self.fast.is_pending(ticker, oid):
                continue

            row = by_id.get(oid)
            old_fill = float(tr.get("last_fill", 0.0))
            if row is not None:
                fc = B._f(row.get("fill_count_fp", row.get("fill_count")), old_fill)
                if fc > old_fill + B.EPS:
                    self.cancel_track(ticker, "RESTING_SET_FILL_DETECTED")
                    transitions += 1
            else:
                created_wall = float(tr.get("created_wall", 0.0) or 0.0)
                if created_wall > 0 and now - created_wall < V6.ORDER_ABSENCE_GRACE_S:
                    continue
                self.cancel_track(ticker, "ABSENT_FROM_RESTING_SET")
                transitions += 1

            if transitions >= V6.MAX_ORDER_STATE_TRANSITIONS_PER_POLL:
                break

    def _watchdog_certification(self, ticker, track_like, source_ms, phase):
        """Return (ok, snapshot, reason) for a CREATE certification."""
        if not self.fast.ready():
            return False, None, "WATCHDOG_NOT_READY"

        snap = self.fast.latest_snapshot(ticker)
        if not snap:
            return False, None, "NO_WATCHDOG_TICKER_STATE"

        fast_ms = B._f(snap.get("receipt_wall_ms"))
        if np.isfinite(source_ms):
            if not np.isfinite(fast_ms):
                return False, snap, "WATCHDOG_RECEIPT_TIME_INVALID"
            if fast_ms + 0.001 < float(source_ms):
                # Main disk tail is ahead of watchdog. Retry next loop after the
                # watchdog reaches at least the decision source row.
                return False, snap, "WATCHDOG_BEHIND_DECISION_SOURCE"

        invalid, invalid_reason = _track_invalidated(
            track_like,
            snap.get("row") or {},
        )
        if invalid:
            return False, snap, f"LATEST_INVALID:{invalid_reason}"

        return True, snap, None

    def place(self, ticker, d, cur, elapsed):
        source_ms = _receipt_ms(self.latest_rows.get(ticker) or {})

        # Certification #1: before any CREATE bookkeeping.
        ok, guard, guard_reason = self._watchdog_certification(
            ticker,
            d,
            source_ms,
            "PREPARE",
        )
        if not ok:
            if guard_reason == "WATCHDOG_NOT_READY":
                self.v12["create_guard_watchdog_not_ready"] += 1
            elif guard_reason == "WATCHDOG_BEHIND_DECISION_SOURCE":
                self.v12["create_guard_watchdog_behind"] += 1
            else:
                self.v12["create_guard_blocks"] += 1
            self._lat(
                "CREATE_BLOCKED_FRESHNESS_GUARD",
                phase="PREPARE",
                ticker=ticker,
                role=d.get("role"),
                side=d.get("side"),
                price=d.get("price"),
                source_receipt_wall_ms=source_ms,
                watchdog_receipt_wall_ms=B._f((guard or {}).get("receipt_wall_ms")),
                reason=guard_reason,
            )
            return

        now = time.time()
        wall_e = self.wall_elapsed(ticker, now_s=now)
        age = self.row_age_s(ticker, now_s=now)
        if (
            not np.isfinite(wall_e)
            or not (0.0 <= wall_e < 300.0)
            or not np.isfinite(age)
            or age > V6.MAX_ACTION_BOOK_AGE_S
            or self.shutdown_started
        ):
            self.v7_stale_create_blocks += 1
            self._lat(
                "CREATE_BLOCKED_WALL_OR_AGE",
                ticker=ticker,
                wall_elapsed_s=wall_e,
                book_age_s=age,
            )
            return

        # Certification #2: repeat immediately before logging the decision and
        # invoking B._post. This minimizes the guard-to-request-send race.
        ok2, guard2, guard_reason2 = self._watchdog_certification(
            ticker,
            d,
            source_ms,
            "FINAL_PRE_POST",
        )
        if not ok2:
            self.v12["create_guard_final_blocks"] += 1
            self._lat(
                "CREATE_BLOCKED_FRESHNESS_GUARD",
                phase="FINAL_PRE_POST",
                ticker=ticker,
                role=d.get("role"),
                side=d.get("side"),
                price=d.get("price"),
                source_receipt_wall_ms=source_ms,
                watchdog_receipt_wall_ms=B._f((guard2 or {}).get("receipt_wall_ms")),
                reason=guard_reason2,
            )
            return

        cid = B.CLIENT_PREFIX + uuid.uuid4().hex
        payload = B._payload(
            ticker=ticker,
            side=d["side"],
            qty=d["qty"],
            price=d["price"],
            cid=cid,
            post_only=True,
            reduce_only=False,
            tif="good_till_canceled",
            group_id=self.gid,
        )

        decision_ms = _wall_ms()
        decision_version = int(self.book_version[ticker])
        B._append(
            self.decisions,
            {
                "time": B._iso(),
                "ticker": ticker,
                "series": self.series(ticker),
                "close_time": self.window_key(ticker),
                "row_elapsed_s": B._f((self.latest_rows.get(ticker) or {}).get("elapsed_s")),
                "wall_elapsed_s": wall_e,
                "book_age_s": age,
                "position": self.positions.get(ticker, 0.0),
                **d,
                "book": cur,
                "engine": LIVE_VERSION,
                "decision_book_version": decision_version,
                "source_receipt_wall_ms": source_ms,
                "watchdog_guard_receipt_wall_ms": B._f((guard2 or {}).get("receipt_wall_ms")),
                "watchdog_guard_seq": (guard2 or {}).get("seq"),
            },
        )

        self._api_context = {
            "v12": "CREATE",
            "ticker": ticker,
            "cid": cid,
            "decision_wall_ms": decision_ms,
            "source_receipt_wall_ms": source_ms,
        }
        try:
            body, timing = B._post(self.client, payload)
        finally:
            self._api_context = {}

        oid = str(body.get("order_id") or "")
        if not oid:
            raise RuntimeError(f"Create response missing order_id: {body}")

        send_ms = B._f((timing or {}).get("request_send_wall_ms"))
        recv_ms = B._f((timing or {}).get("response_recv_wall_ms"))
        decision_to_send_ms = (
            send_ms - decision_ms
            if np.isfinite(send_ms)
            else np.nan
        )
        source_age_at_send_ms = (
            send_ms - source_ms
            if np.isfinite(send_ms) and np.isfinite(source_ms)
            else np.nan
        )

        newer_before_send = (
            self.fast.first_newer_before(ticker, source_ms, send_ms)
            if np.isfinite(send_ms)
            else None
        )
        superseded_at_send = newer_before_send is not None
        if superseded_at_send:
            self.v12["superseded_at_send"] += 1

        tr = {
            "ticker": ticker,
            "order_id": oid,
            "cid": cid,
            "role": d["role"],
            "side": d["side"],
            "price": d["price"],
            "qty": d["qty"],
            "last_fill": 0.0,
            "book": cur,
            "elapsed_s": wall_e,
            "row_elapsed_s": B._f((self.latest_rows.get(ticker) or {}).get("elapsed_s")),
            "book_age_s": age,
            "created_wall": time.time(),
            "decision_book_version": decision_version,
            "source_receipt_wall_ms": source_ms,
        }
        self.active[ticker] = tr
        self._publish()  # wakes watchdog; catches changes during create flight

        B._append(
            self.orders,
            {
                "time": B._iso(),
                "action": "CREATE_V12_FRESHNESS_GUARDED",
                "ticker": ticker,
                "payload": payload,
                "response": body,
                "timing": timing,
                "wall_elapsed_s": wall_e,
                "book_age_s": age,
                "decision_book_version": decision_version,
                "source_receipt_wall_ms": source_ms,
                "decision_to_send_ms": decision_to_send_ms,
                "source_age_at_send_ms": source_age_at_send_ms,
                "superseded_at_send_detected": superseded_at_send,
            },
        )
        self.counts["orders"] += 1
        self.last_action_wall = time.time()
        self.v7_max_sent_order_book_age_s = max(
            self.v7_max_sent_order_book_age_s,
            float(age),
        )

        self._lat(
            "CREATE_SENT",
            ticker=ticker,
            order_id=oid,
            client_order_id=cid,
            role=d.get("role"),
            decision_wall_ms=decision_ms,
            request_send_wall_ms=send_ms,
            response_recv_wall_ms=recv_ms,
            create_rtt_ms=B._f((timing or {}).get("rtt_ms")),
            decision_to_send_ms=decision_to_send_ms,
            source_receipt_wall_ms=source_ms,
            source_age_at_send_ms=source_age_at_send_ms,
            decision_book_version=decision_version,
            first_newer_before_send_receipt_wall_ms=B._f(
                (newer_before_send or {}).get("receipt_wall_ms")
            ),
            superseded_at_send_detected=superseded_at_send,
        )

        self.emit(
            "ENTRY_ORDER" if d["role"] == "ENTRY" else "EXIT_ORDER",
            ticker,
            role=d["role"],
            side=d["side"],
            qty=d["qty"],
            price=d["price"],
            wall_elapsed_s=wall_e,
            book_age_s=age,
            decision_to_send_ms=decision_to_send_ms,
        )

        immediate = B._f(body.get("fill_count"), 0.0)
        if immediate > B.EPS:
            self.cancel_track(ticker, "CREATE_RESPONSE_FILL")

    def _prune(self):
        """Prevent cumulative historical ticker traversal from reappearing."""
        for ticker in list(self.latest_rows):
            if ticker in self.active:
                continue
            e = self.wall_elapsed(ticker)
            remove = ticker in self.finalized
            if ticker in self.first_seen and self.eligible.get(ticker) is False:
                remove = remove or (np.isfinite(e) and e >= 300.0)
            if remove:
                self.latest_rows.pop(ticker, None)
                self.current.pop(ticker, None)
                self.pending_first_rows.pop(ticker, None)

    def _actionable(self):
        out = []
        for ticker in self.latest_rows:
            if ticker in self.active:
                out.append(ticker)
                continue
            if ticker in self.finalized or not self.eligible.get(ticker, False):
                continue
            e = self.wall_elapsed(ticker)
            if np.isfinite(e) and 0.0 <= e < 300.0:
                out.append(ticker)
        self.max_actionable = max(self.max_actionable, len(out))
        return sorted(out)

    def process_latest_states(self):
        """Current-window action pass with no per-historical-key risk_tick."""
        self._consume_fast()
        self._prune()
        keys = self._actionable()
        if not keys:
            return

        start = self.rr_cursor % len(keys)
        ordered = keys[start:] + keys[:start]
        visited = 0

        for ticker in ordered:
            visited += 1
            if self.shutdown_started:
                break

            self._consume_fast()
            tr = self.active.get(ticker)
            if tr and self.fast.is_pending(ticker, tr.get("order_id")):
                continue

            wall_e = self.wall_elapsed(ticker)
            age = self.row_age_s(ticker)
            if np.isfinite(age):
                self.max_latest_book_age_s = max(self.max_latest_book_age_s, age)

            if not np.isfinite(wall_e) or not (0.0 <= wall_e < 300.0):
                if ticker in self.active:
                    self.cancel_track(ticker, "WALL_CLOCK_OUTSIDE_WINDOW")
                continue

            if not np.isfinite(age) or age > V6.MAX_ACTION_BOOK_AGE_S:
                if ticker in self.active:
                    self.cancel_track(ticker, "STALE_LATEST_BOOK")
                continue

            cur = _top(self.latest_rows.get(ticker) or {})
            self.current[ticker] = cur
            if cur is None:
                if ticker in self.active:
                    self.cancel_track(ticker, "INVALID_LATEST_BOOK")
                continue

            # V2/V11 reconcile remains the authoritative inventory/replacement path.
            self.reconcile(ticker, cur, wall_e)

        self.rr_cursor = (start + max(1, visited)) % len(keys)

    def ingest_book_batch(self, rows):
        if not rows:
            return
        t0 = _perf_ms()
        super().ingest_book_batch(rows)
        self._lat(
            "MAIN_BOOK_INGEST",
            rows=len(rows),
            ingest_ms=_perf_ms() - t0,
        )

    def run(self):
        self.emit("ENGINE_START", mode=self.mode, engine=LIVE_VERSION)
        self.health(force=True)
        try:
            while not self.shutdown_started:
                if self.recorder_proc.poll() is not None:
                    raise RuntimeError(
                        f"Raw V5 recorder exited rc={self.recorder_proc.returncode}"
                    )

                loop_start = _perf_ms()
                self.update_meta()

                read_start = _perf_ms()
                rows = self.book_tail.read_new()
                read_ms = _perf_ms() - read_start
                if rows:
                    self._lat("MAIN_BOOK_READ", rows=len(rows), read_ms=read_ms)
                    self.ingest_book_batch(rows)

                self._consume_fast()
                self.check_end()
                if self.shutdown_started:
                    break

                # M5 wall-clock safety before discretionary quote work.
                self.enforce_wall_clock_m5()
                if self.shutdown_started:
                    break

                self.process_latest_states()
                self._consume_fast()

                # Retain the proven balance/position/order/queue risk machinery,
                # but only once per outer pass. Priority cancel can run concurrently.
                self.risk_tick()

                self._lat(
                    "MAIN_LOOP",
                    loop_ms=_perf_ms() - loop_start,
                    actionable_tickers=len(self._actionable()),
                    active_orders=len(self.active),
                )
                time.sleep(LOOP_SLEEP_S)

        except BaseException as exc:
            self.last_error = repr(exc)
            self.emit(
                "ERROR",
                error=repr(exc),
                traceback=__import__("traceback").format_exc(),
            )
            try:
                self.shutdown("ENGINE_EXCEPTION")
            except Exception as cleanup_exc:
                self.emit(
                    "CRITICAL",
                    error=repr(cleanup_exc),
                    reason="cleanup_exception",
                )
                try:
                    self.fast.stop()
                except Exception:
                    pass
                self.stop_recorder()
            raise
        finally:
            self.health(force=True)

    def _v12_metrics(self):
        return {
            "live_version": LIVE_VERSION,
            "freshness_arch_version": FRESHNESS_ARCH_VERSION,
            "target_cancel_send_ms": TARGET_CANCEL_SEND_MS,
            "p95_cancel_send_ms": P95_CANCEL_SEND_MS,
            "hard_cancel_send_ms": HARD_CANCEL_SEND_MS,
            "watchdog": self.fast.metrics(),
            "create_guard_blocks": int(self.v12["create_guard_blocks"]),
            "create_guard_final_blocks": int(self.v12["create_guard_final_blocks"]),
            "create_guard_watchdog_not_ready": int(
                self.v12["create_guard_watchdog_not_ready"]
            ),
            "create_guard_watchdog_behind": int(
                self.v12["create_guard_watchdog_behind"]
            ),
            "create_superseded_at_send_detected": int(
                self.v12["superseded_at_send"]
            ),
            "fast_cancel_successes": int(self.v12["fast_success"]),
            "fast_cancel_fallbacks": int(self.v12["fast_fallbacks"]),
            "fast_cancel_raced_fills": int(self.v12["fast_raced_fill"]),
            "max_actionable_tickers": self.max_actionable,
            "max_actionable_tickers_expected": MAX_ACTIONABLE_EXPECTED,
            "strategy_changed": False,
            "q10_launcher_enabled": False,
        }

    def _v10_metrics(self):
        base = super()._v10_metrics()
        base.update(
            {
                "live_version": LIVE_VERSION,
                "execution_parent": EXECUTION_PARENT,
                "v12_priority_freshness": self._v12_metrics(),
            }
        )
        return base

    def health(self, force=False):
        super().health(force=force)
        h = B._read(self.health_path, {}) or {}
        h["live_version"] = LIVE_VERSION
        h["execution_parent"] = EXECUTION_PARENT
        h["v12_metrics"] = self._v12_metrics()
        B._atomic(self.health_path, h)

    def shutdown(self, reason):
        if self.shutdown_started:
            return

        # Stop creating NEW watchdog work. Base shutdown triggers the order group
        # before ordinary cancellation/flatten, so fail-closed safety remains intact.
        self.fast.stop()
        self._consume_fast(limit=1000)
        super().shutdown(reason)

        summary = B._read(self.final_path, {}) or {}
        summary["live_wrapper_version"] = LIVE_VERSION
        summary["execution_parent"] = EXECUTION_PARENT
        summary["freshness_arch_version"] = FRESHNESS_ARCH_VERSION
        summary["v12_metrics"] = self._v12_metrics()
        B._atomic(self.final_path, summary)

        h = B._read(self.health_path, {}) or {}
        h["live_version"] = LIVE_VERSION
        h["execution_parent"] = EXECUTION_PARENT
        h["v12_metrics"] = self._v12_metrics()
        h["summary"] = summary
        B._atomic(self.health_path, h)

        try:
            audit_v12_smoke(self.session, show=False, write_result=True)
        except Exception as exc:
            self._lat("SMOKE_AUDIT_ERROR", error=repr(exc))


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
    x = pd.to_numeric(pd.Series(list(vals), dtype="float64"), errors="coerce")
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "n": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "n": int(len(x)),
        "median": float(x.median()),
        "p90": float(x.quantile(0.90)),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "max": float(x.max()),
    }


def audit_v12_smoke(session_dir, *, show=True, write_result=True):
    """Read-only Q1 architecture/latency audit. Sends no orders and calls no API."""
    session = Path(session_dir).resolve()
    events = _read_jsonl(session / "latency_events_v12.jsonl")
    final = B._read(session / "final_summary.json", {}) or {}

    fast_all = [r for r in events if r.get("event") == "FAST_CANCEL_RESULT"]
    fast_success = [r for r in fast_all if r.get("success")]
    invalidations = [r for r in events if r.get("event") == "FAST_INVALIDATION_DETECTED"]
    reconciled = [r for r in events if r.get("event") == "FAST_CANCEL_RECONCILED"]
    creates = [r for r in events if r.get("event") == "CREATE_SENT"]
    guards = [r for r in events if r.get("event") == "CREATE_BLOCKED_FRESHNESS_GUARD"]
    caught_up = [r for r in events if r.get("event") == "WATCHDOG_CAUGHT_UP"]
    watchdog_json_errors = [r for r in events if r.get("event") == "WATCHDOG_JSON_ERROR"]
    http = [r for r in events if r.get("event") == "MAIN_HTTP"]
    loops = [r for r in events if r.get("event") == "MAIN_LOOP"]
    reads = [r for r in events if r.get("event") == "MAIN_BOOK_READ"]
    ingests = [r for r in events if r.get("event") == "MAIN_BOOK_INGEST"]

    obsolete_to_send = _stats(r.get("obsolete_to_cancel_send_ms") for r in fast_success)
    raw_to_detect = _stats(r.get("watchdog_raw_to_detect_ms") for r in fast_success)
    detect_to_send = _stats(r.get("detect_to_cancel_send_ms") for r in fast_success)
    cancel_rtt = _stats(r.get("cancel_rtt_ms") for r in fast_success)
    response_to_reconcile = _stats(
        r.get("response_to_reconcile_start_ms") for r in reconciled
    )
    reconcile_duration = _stats(r.get("reconcile_duration_ms") for r in reconciled)
    decision_to_send = _stats(r.get("decision_to_send_ms") for r in creates)
    source_age_at_send = _stats(r.get("source_age_at_send_ms") for r in creates)
    create_rtt = _stats(r.get("create_rtt_ms") for r in creates)
    loop_ms = _stats(r.get("loop_ms") for r in loops)
    book_read_ms = _stats(r.get("read_ms") for r in reads)
    book_ingest_ms = _stats(r.get("ingest_ms") for r in ingests)

    http_by_class = {}
    for cls in sorted({str(r.get("http_class")) for r in http}):
        rows = [r for r in http if str(r.get("http_class")) == cls]
        http_by_class[cls] = {
            "calls": len(rows),
            "rtt_ms": _stats(r.get("rtt_ms") for r in rows),
            "local_call_ms": _stats(r.get("local_call_ms") for r in rows),
        }

    superseded = sum(bool(r.get("superseded_at_send_detected")) for r in creates)
    safety_pass = bool(
        final
        and final.get("flat_verified") is True
        and final.get("strategy_resting_orders_zero") is True
        and final.get("last_error") in (None, "")
    )
    watchdog_pass = bool(caught_up and not watchdog_json_errors)
    cancel_latency_observed = obsolete_to_send["n"] > 0
    cancel_completion_pass = bool(
        cancel_latency_observed
        and len(fast_all) == len(invalidations)
        and len(fast_success) == len(invalidations)
    )
    cancel_latency_pass = bool(
        cancel_completion_pass
        and obsolete_to_send["median"] <= TARGET_CANCEL_SEND_MS
        and obsolete_to_send["p95"] <= P95_CANCEL_SEND_MS
        and obsolete_to_send["max"] <= HARD_CANCEL_SEND_MS
    )
    create_freshness_pass = superseded == 0

    metrics = final.get("v12_metrics") or {}
    max_actionable = B._f(metrics.get("max_actionable_tickers"))
    actionable_set_pass = bool(
        not np.isfinite(max_actionable)
        or max_actionable <= MAX_ACTIONABLE_EXPECTED
    )

    promotion = bool(
        safety_pass
        and watchdog_pass
        and cancel_latency_observed
        and cancel_latency_pass
        and create_freshness_pass
        and actionable_set_pass
    )

    if promotion:
        interpretation = "PASS"
    elif safety_pass and watchdog_pass and create_freshness_pass and not cancel_latency_observed:
        interpretation = "INCOMPLETE_LATENCY_PROOF_NO_INVALIDATION"
    else:
        interpretation = "FAIL_INVESTIGATE"

    out = {
        "time": B._iso(),
        "session_dir": str(session),
        "live_version": LIVE_VERSION,
        "orders_sent": False,
        "exchange_api_called": False,
        "final": {
            "shutdown_reason": final.get("shutdown_reason"),
            "account_pnl_usd": final.get("account_pnl_usd"),
            "flat_verified": final.get("flat_verified"),
            "strategy_resting_orders_zero": final.get("strategy_resting_orders_zero"),
            "last_error": final.get("last_error"),
        },
        "counts": {
            "watchdog_caught_up_events": len(caught_up),
            "watchdog_json_errors": len(watchdog_json_errors),
            "fast_invalidations": len(invalidations),
            "fast_cancel_results": len(fast_all),
            "fast_cancel_success_results": len(fast_success),
            "fast_cancel_failed_results": len(fast_all) - len(fast_success),
            "fast_cancel_reconciliations": len(reconciled),
            "creates": len(creates),
            "create_guard_blocks": len(guards),
            "superseded_at_send_detected": superseded,
            "main_http_calls": len(http),
            "main_loops": len(loops),
        },
        "latency": {
            "raw_receipt_to_watchdog_detect_ms": raw_to_detect,
            "watchdog_detect_to_cancel_send_ms": detect_to_send,
            "obsolete_receipt_to_cancel_send_ms": obsolete_to_send,
            "priority_cancel_rtt_ms": cancel_rtt,
            "cancel_response_to_main_reconcile_start_ms": response_to_reconcile,
            "priority_cancel_main_reconcile_duration_ms": reconcile_duration,
            "decision_to_create_send_ms": decision_to_send,
            "source_receipt_age_at_create_send_ms": source_age_at_send,
            "create_rtt_ms": create_rtt,
            "main_loop_ms": loop_ms,
            "main_book_read_ms": book_read_ms,
            "main_book_ingest_ms": book_ingest_ms,
            "main_http_by_class": http_by_class,
        },
        "slo": {
            "target_median_obsolete_to_cancel_send_ms": TARGET_CANCEL_SEND_MS,
            "p95_obsolete_to_cancel_send_ms": P95_CANCEL_SEND_MS,
            "hard_max_obsolete_to_cancel_send_ms": HARD_CANCEL_SEND_MS,
        },
        "gates": {
            "safety_pass": safety_pass,
            "watchdog_pass": watchdog_pass,
            "create_freshness_pass": create_freshness_pass,
            "actionable_set_pass": actionable_set_pass,
            "cancel_latency_observed": cancel_latency_observed,
            "cancel_completion_pass": cancel_completion_pass,
            "cancel_latency_pass": cancel_latency_pass,
            "promotion_ready_for_larger_smoke": promotion,
        },
        "interpretation": interpretation,
    }

    if write_result:
        B._atomic(session / "v12_smoke_latency_audit.json", out)

    if show:
        print("=" * 106)
        print("V12 Q1 SMOKE LATENCY AUDIT — READ ONLY")
        print("=" * 106)
        print("Session:                       ", session)
        print("PnL:                           ", out["final"]["account_pnl_usd"])
        print("Flat / zero resting:           ", safety_pass)
        print("Watchdog caught up / JSON OK:  ", watchdog_pass)
        print("Invalidations / fast results:  ", len(invalidations), "/", len(fast_all))
        print("Superseded CREATEs at send:    ", superseded)
        print("Obsolete -> cancel send:       ", obsolete_to_send)
        print("Raw -> watchdog detect:        ", raw_to_detect)
        print("Watchdog detect -> cancel send:", detect_to_send)
        print("Priority cancel RTT:           ", cancel_rtt)
        print("Cancel response -> reconcile:  ", response_to_reconcile)
        print("Decision -> CREATE send:       ", decision_to_send)
        print("Source age at CREATE send:     ", source_age_at_send)
        print("CREATE RTT:                    ", create_rtt)
        print("Main loop:                     ", loop_ms)
        print("Gates:                         ", out["gates"])
        print("Interpretation:                ", interpretation)
        print("ORDERS SENT BY AUDIT: NO | EXCHANGE API CALLED BY AUDIT: NO")

    return out


def static_self_check(*, show=True):
    """No API / no orders. Validate frozen wiring and V12 engineering constants."""
    checks = {
        "frozen_universe_same": tuple(B.SERIES) == tuple(B.OOS.SERIES),
        "q1_size_unchanged": abs(float(B.SMOKE_Q) - 1.0) <= B.EPS,
        "watchdog_workers_cover_universe": WATCHDOG_WORKERS >= len(B.SERIES),
        "cancel_slo_ordered": 0 < TARGET_CANCEL_SEND_MS <= P95_CANCEL_SEND_MS <= HARD_CANCEL_SEND_MS,
        "q10_disabled": True,
        "execution_parent_is_v11": EXECUTION_PARENT == V11.LIVE_VERSION,
        "recording_parent_is_v10": RECORDING_PARENT == V10.LIVE_VERSION,
    }
    out = {
        "time": B._iso(),
        "live_version": LIVE_VERSION,
        "checks": checks,
        "pass": all(checks.values()),
        "orders_sent": False,
        "exchange_api_called": False,
    }
    if show:
        print("V12 STATIC SELF CHECK:", "PASS" if out["pass"] else "FAIL")
        for k, v in checks.items():
            print(f"  {k:<36} {v}")
        print("  ORDERS SENT: NO | EXCHANGE API CALLED: NO")
    return out


def _write_v12_bundle(session: Path, cfg):
    session = Path(session).resolve()
    V11._write_v11_bundle(session, cfg)

    prov = B._read(session / "source_provenance.json", {}) or {}
    prov.update(
        {
            "live_version": LIVE_VERSION,
            "execution_parent": EXECUTION_PARENT,
            "recording_parent": RECORDING_PARENT,
            "freshness_arch_version": FRESHNESS_ARCH_VERSION,
        }
    )
    sources = dict(prov.get("sources") or {})
    sources["v12"] = V10._module_source(sys.modules[__name__])
    prov["sources"] = sources
    B._atomic(session / "source_provenance.json", prov)

    spec = B._read(session / "live_execution_spec_v11.json", {}) or {}
    spec.update(
        {
            "live_version": LIVE_VERSION,
            "execution_parent": EXECUTION_PARENT,
            "freshness_arch_version": FRESHNESS_ARCH_VERSION,
            "strategy_mechanics_changed": False,
            "priority_raw_watchdog": True,
            "watchdog_requires_initial_catchup": True,
            "priority_cancel_workers": WATCHDOG_WORKERS,
            "create_guard_against_watchdog_latest": True,
            "double_create_certification": True,
            "historical_actionable_rows_pruned": True,
            "risk_tick_per_historical_key": False,
            "latency_log": "latency_events_v12.jsonl",
            "q1_first": True,
            "q10_launcher_enabled": False,
        }
    )
    B._atomic(session / "live_execution_spec_v12.json", spec)

    B._atomic(
        session / "latency_slo_v12.json",
        {
            "time": B._iso(),
            "version": FRESHNESS_ARCH_VERSION,
            "target_median_obsolete_to_cancel_send_ms": TARGET_CANCEL_SEND_MS,
            "p95_obsolete_to_cancel_send_ms": P95_CANCEL_SEND_MS,
            "hard_max_obsolete_to_cancel_send_ms": HARD_CANCEL_SEND_MS,
            "strategy_thresholds_changed": False,
        },
    )


def _run_process_v12(session, cfg):
    session = Path(session).resolve()
    _write_v12_bundle(session, cfg)

    client = B.Q1.LiveClient()
    diag = V3._install(client)
    B._atomic(session / "balance_semantics.json", diag)

    B._post = V11._post_v11
    B._start_recorder = V4._start_recorder_fixed
    B.LiveEngine = PriorityFreshnessEngine
    B._run_process(session, cfg)


def live_preflight(
    *,
    quote_size=B.SMOKE_Q,
    runtime_hours=1.0,
    max_start_loss_usd=B.LOSS_LIMIT_USD,
    min_start_equity_usd=B.SMOKE_MIN_EQUITY,
    show=True,
):
    return V3.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
    )


def start_live_smoke_q1_one_window(
    *,
    arm_phrase=None,
    max_start_loss_usd=B.LOSS_LIMIT_USD,
    min_start_equity_usd=B.SMOKE_MIN_EQUITY,
):
    if str(arm_phrase) != B.SMOKE_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={B.SMOKE_ARM!r} exactly."
        )

    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    static = static_self_check(show=True)
    if not static["pass"]:
        raise RuntimeError(f"V12 static self-check failed: {static}")

    V3._calibrated_preflight(
        quote_size=B.SMOKE_Q,
        runtime_hours=1.0,
        max_loss_usd=float(max_start_loss_usd),
        min_equity_usd=float(min_start_equity_usd),
        mode="SMOKE_Q1_ONE_WINDOW_V12",
        save_dir=None,
        show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (
        B.ROOT / f"{stamp}_smoke_q1_one_window_v12"
    ).resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": "SMOKE_Q1_ONE_WINDOW",
        "quote_size": float(B.SMOKE_Q),
        "runtime_hours": 1.0,
        "max_start_loss_usd": float(max_start_loss_usd),
        "min_start_equity_usd": float(min_start_equity_usd),
        "live_wrapper_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_parent": RECORDING_PARENT,
        "freshness_arch_version": FRESHNESS_ARCH_VERSION,
        "engine_architecture": "V11_ORDER_SAFETY_PLUS_V12_PRIORITY_FRESHNESS_WATCHDOG",
        "recording_version": V10.RECORDING_VERSION,
        "comparison_schema_version": V10.COMPARISON_SCHEMA_VERSION,
        "order_api_safety_version": V11.ORDER_API_SAFETY_VERSION,
        "q10_launcher_enabled": False,
    }
    B._atomic(session / "process_config.json", cfg)
    _write_v12_bundle(session, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "quant_research.kalshi.mm_cycle_q10_live_strategy_v12",
                "--run-live-session",
                str(session),
                "--config",
                str(session / "process_config.json"),
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
            "execution_parent": EXECUTION_PARENT,
            "recording_parent": RECORDING_PARENT,
            "freshness_arch_version": FRESHNESS_ARCH_VERSION,
            "recording_version": V10.RECORDING_VERSION,
            "order_api_safety_version": V11.ORDER_API_SAFETY_VERSION,
            "running": True,
            "pid": p.pid,
            "session_dir": str(session),
            "mode": "SMOKE_Q1_ONE_WINDOW",
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
                f"Live V12 process exited during startup rc={p.returncode}\n{tail}"
            )
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-20000:] if log.exists() else ""
        raise RuntimeError(
            f"Live V12 startup timeout. Last health={last}\n{tail}"
        )

    print("\nLIVE V12 Q1 PROCESS ARMED")
    print("  session:    ", session)
    print("  pid:        ", p.pid)
    print("  Q:           1 per eligible market")
    print(
        f"  cancel SLO: median<={TARGET_CANCEL_SEND_MS:.0f}ms | "
        f"p95<={P95_CANCEL_SEND_MS:.0f}ms | max<={HARD_CANCEL_SEND_MS:.0f}ms"
    )
    print("  Q10:         DISABLED until this Q1 audit passes")
    print(
        "  latency:     raw->watchdog->cancel-send->response->reconcile, "
        "CREATE decision/send/RTT, all main HTTP, disk read/ingest, loop"
    )
    return live_status(show=False)


def start_live_cycle_q10(*args, **kwargs):
    raise RuntimeError(
        "V12 Q10 is intentionally disabled until a fresh V12 Q1 smoke passes "
        "audit_v12_smoke()."
    )


def live_status(*, show=True, tail_lines=20):
    return B.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    return B.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def account_safety_check(*, show=True):
    return V11.account_safety_check(show=show)


def verify_recording_bundle(session_dir, *, show=True, write_result=True):
    return V10.verify_recording_bundle(
        session_dir,
        show=show,
        write_result=write_result,
    )


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    ap.add_argument("--audit-smoke-session")
    ap.add_argument("--verify-recording-session")
    a = ap.parse_args()

    if a.audit_smoke_session:
        audit_v12_smoke(a.audit_smoke_session, show=True, write_result=True)
        return
    if a.verify_recording_session:
        verify_recording_bundle(a.verify_recording_session, show=True, write_result=True)
        return
    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}
        _run_process_v12(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "EXECUTION_PARENT",
    "RECORDING_PARENT",
    "FRESHNESS_ARCH_VERSION",
    "PriorityFreshnessEngine",
    "audit_v12_smoke",
    "static_self_check",
    "account_safety_check",
    "verify_recording_bundle",
    "live_preflight",
    "start_live_smoke_q1_one_window",
    "start_live_cycle_q10",
    "live_status",
    "kill_and_flatten_live",
]
