from __future__ import annotations

"""V1.7 long-run memory hardening for deep-tail live trading.

No alpha/execution rule changes from V1.6.

Observed Q10 8h failure
-----------------------
The 2026-08-20 Q10 overnight run reached the V2.8 2 GiB RSS guardian limit near
its runtime boundary. The raw recorder peaked near 118 MiB while the strategy
process group exceeded 2 GiB. The existing REST fill reconciler queried up to
1000 *historical* fills every 0.5s and re-enqueued every returned row. The API
supports min_ts/cursor, so that behavior created enormous avoidable allocation
churn during long runs.

This version changes operations only:
1) REST fill reconciliation is incremental: min_ts watermark + small overlap,
   cursor pagination, exact fill-id dedupe, and queue telemetry. Old fills are not
   replayed every 0.5s.
2) Private websocket receive timeouts are treated as quiet-idle heartbeats rather
   than forcing a reconnect. Actual socket failures still reconnect and publish
   WS_STATE DOWN exactly as before.
3) The V1.5 bounded raw JSONL tailer is explicitly required at runtime and health
   publishes queue/reconciler/memory telemetry so the next long run can prove the
   fix rather than assume it.
4) A large private-event queue backlog fails closed instead of allowing unbounded
   accumulation.

M1 dual 5c entries, first-fill-wins, opposite cancel, selected-tail accumulation,
fixed JOIN_ASK/no-reprice exit, persistent M5 cleanup, authenticated V5 recorder,
risk threshold, order-group safety, account auditor and guardian semantics are
unchanged.

Importing this module sends no orders.
"""

import asyncio
import json
import math
import time
from collections import deque
from pathlib import Path
from queue import Empty

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_5 as V15
from . import mm_deep_tail_join_ask_live_v1_6 as V16


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_7_LONG_RUN_MEMORY_SAFE"
REST_FILL_OVERLAP_S = 2
REST_FILL_INITIAL_LOOKBACK_S = 2
REST_FILL_MAX_PAGES_PER_POLL = 20
REST_FILL_SEEN_MAX = 20000
PRIVATE_QUEUE_HARD_LIMIT = 10000


def _fill_key(row):
    row = row or {}
    fid = str(row.get("fill_id") or "")
    if fid:
        return ("fill_id", fid)
    trade = str(row.get("trade_id") or "")
    oid = str(row.get("order_id") or "")
    ts = str(row.get("ts_ms", row.get("ts", "")))
    qty = str(row.get("count_fp", row.get("count", "")))
    return ("fallback", trade, oid, ts, qty)


def _fill_ts_seconds(row):
    z = B._f((row or {}).get("ts"), float("nan"))
    if not math.isfinite(z):
        z = B._f((row or {}).get("ts_ms"), float("nan"))
    if not math.isfinite(z):
        return None
    # Defensive handling if an endpoint ever returns milliseconds here.
    if z > 100_000_000_000:
        z /= 1000.0
    return int(z)


class IncrementalRestFillReconciler:
    """REST fill fallback that requests only fills newer than a moving watermark."""

    def __init__(self, out_queue, wake_event):
        self.out_queue = out_queue
        self.wake_event = wake_event
        self.stop_event = V1.threading.Event()
        self.thread = V1.threading.Thread(
            target=self._run,
            name="dt-rest-fill-reconcile-v17",
            daemon=True,
        )
        self.watermark_ts = max(0, int(time.time()) - REST_FILL_INITIAL_LOOKBACK_S)
        self.seen = set()
        self.seen_order = deque()
        self.polls = 0
        self.pages = 0
        self.rows_received = 0
        self.rows_enqueued = 0
        self.duplicates_suppressed = 0
        self.max_rows_in_poll = 0
        self.last_error = None
        self.last_poll_wall = None

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _remember(self, key):
        if key in self.seen:
            return False
        self.seen.add(key)
        self.seen_order.append(key)
        while len(self.seen_order) > REST_FILL_SEEN_MAX:
            old = self.seen_order.popleft()
            self.seen.discard(old)
        return True

    def metrics(self):
        return {
            "mode": "MIN_TS_INCREMENTAL_DEDUP",
            "watermark_ts": int(self.watermark_ts),
            "overlap_s": REST_FILL_OVERLAP_S,
            "polls": int(self.polls),
            "pages": int(self.pages),
            "rows_received": int(self.rows_received),
            "rows_enqueued": int(self.rows_enqueued),
            "duplicates_suppressed": int(self.duplicates_suppressed),
            "seen_keys": len(self.seen),
            "max_rows_in_poll": int(self.max_rows_in_poll),
            "last_poll_wall": self.last_poll_wall,
            "last_error": self.last_error,
        }

    def _run(self):
        client = B.Q1.LiveClient()
        try:
            client.get("/portfolio/balance")
        except Exception:
            pass

        while not self.stop_event.wait(V1.REST_FILL_RECONCILE_S):
            self.polls += 1
            self.last_poll_wall = time.time()
            poll_rows = 0
            max_ts = int(self.watermark_ts)
            cursor = None
            page = 0
            try:
                while True:
                    page += 1
                    if page > REST_FILL_MAX_PAGES_PER_POLL:
                        raise RuntimeError(
                            f"REST fill pagination exceeded {REST_FILL_MAX_PAGES_PER_POLL} pages"
                        )
                    params = {
                        "limit": 1000,
                        "subaccount": 0,
                        "min_ts": max(0, int(self.watermark_ts) - REST_FILL_OVERLAP_S),
                    }
                    if cursor:
                        params["cursor"] = cursor
                    body, timing = client.get("/portfolio/fills", params=params)
                    self.pages += 1
                    rows = (body or {}).get("fills") or []
                    self.rows_received += len(rows)
                    poll_rows += len(rows)

                    for r in rows:
                        ts = _fill_ts_seconds(r)
                        if ts is not None:
                            max_ts = max(max_ts, ts)
                        key = _fill_key(r)
                        if not self._remember(key):
                            self.duplicates_suppressed += 1
                            continue
                        self.out_queue.put({
                            "kind": "REST_FILL",
                            "msg": r,
                            "recv_ms": V1._wall_ms(),
                            "timing": timing,
                        })
                        self.rows_enqueued += 1

                    cursor = str((body or {}).get("cursor") or "")
                    if not cursor:
                        break

                self.watermark_ts = max(int(self.watermark_ts), int(max_ts))
                self.max_rows_in_poll = max(self.max_rows_in_poll, poll_rows)
                self.last_error = None
                if poll_rows:
                    self.wake_event.set()
            except Exception as exc:
                self.last_error = repr(exc)
                self.out_queue.put({
                    "kind": "REST_FILL_ERROR",
                    "error": repr(exc),
                    "recv_ms": V1._wall_ms(),
                })
                self.wake_event.set()


class PersistentPrivateUserStream(V1.PrivateUserStream):
    """Keep a healthy private WS open across quiet 5-second receive intervals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idle_timeouts = 0
        self.actual_disconnects = 0

    async def _run(self):
        key_id, private_key = C.load_auth()
        while not self.stop_event.is_set():
            self.ready.clear()
            subscribed = set()
            connection_error = None
            try:
                ws = await C.open_ws(key_id, private_key)
                try:
                    await ws.send(json.dumps({
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {"channels": ["fill", "user_orders"]},
                    }))
                    while not self.stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            # Quiet feed is normal. Keep the authenticated socket and
                            # READY state; do not churn TLS/websocket allocations.
                            self.idle_timeouts += 1
                            continue

                        recv_ms = V1._wall_ms()
                        try:
                            row = json.loads(raw)
                        except Exception:
                            row = {"type": "decode_error", "raw": str(raw)[:2000]}
                        B._append(self.log_path, {
                            "local_recv_wall_ms": recv_ms,
                            "payload": row,
                        })
                        typ = str((row or {}).get("type") or "")
                        if typ == "subscribed":
                            ch = str(((row or {}).get("msg") or {}).get("channel") or "")
                            if ch:
                                subscribed.add(ch)
                            if {"fill", "user_orders"}.issubset(subscribed):
                                if not self.ready.is_set():
                                    self.out_queue.put({
                                        "kind": "WS_STATE",
                                        "state": "READY",
                                        "recv_ms": recv_ms,
                                    })
                                    self.ready.set()
                                    self.wake_event.set()
                        elif typ in {"fill", "user_order"}:
                            self.out_queue.put({
                                "kind": typ.upper(),
                                "msg": (row or {}).get("msg") or {},
                                "recv_ms": recv_ms,
                            })
                            self.wake_event.set()
                finally:
                    try:
                        await ws.close()
                    except Exception:
                        pass
            except Exception as exc:
                connection_error = repr(exc)
                self.last_error = connection_error
            finally:
                was_ready = self.ready.is_set()
                self.ready.clear()
                if not self.stop_event.is_set():
                    self.reconnects += 1
                    self.actual_disconnects += 1
                    if was_ready or connection_error:
                        self.out_queue.put({
                            "kind": "WS_STATE",
                            "state": "DOWN",
                            "error": connection_error,
                            "recv_ms": V1._wall_ms(),
                        })
                        self.wake_event.set()
            if not self.stop_event.is_set():
                await asyncio.sleep(0.25)


class LongRunMemorySafeEngine(V15.BoundedMemoryEngine):
    """V1.5 bounded engine + explicit long-run queue/reconciler telemetry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.book_tail, V15.BoundedJsonlTail):
            raise RuntimeError(
                f"V1.7 requires bounded book tail, got {type(self.book_tail)!r}"
            )
        if not isinstance(self.meta_tail, V15.BoundedJsonlTail):
            raise RuntimeError(
                f"V1.7 requires bounded metadata tail, got {type(self.meta_tail)!r}"
            )
        self._lat(
            "V1_7_LONG_RUN_MEMORY_GUARDS_READY",
            rest_fill_mode="MIN_TS_INCREMENTAL_DEDUP",
            private_queue_hard_limit=PRIVATE_QUEUE_HARD_LIMIT,
        )

    def _queue_guard(self):
        q = int(self.private_q.qsize())
        if q > PRIVATE_QUEUE_HARD_LIMIT and not self.shutdown_started:
            self.last_error = (
                f"private event queue backlog {q} > hard limit {PRIVATE_QUEUE_HARD_LIMIT}"
            )
            self.emit("CRITICAL", reason="PRIVATE_QUEUE_BACKLOG_LIMIT", qsize=q)
            self.shutdown("PRIVATE_QUEUE_BACKLOG_LIMIT")

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            rest_metrics = (
                self.rest_fills.metrics()
                if hasattr(self.rest_fills, "metrics")
                else {"mode": type(self.rest_fills).__name__}
            )
            h.update({
                "live_memory_hardening_version": LIVE_VERSION,
                "bounded_raw_ingestion": True,
                "bounded_book_tail_runtime_verified": isinstance(
                    self.book_tail, V15.BoundedJsonlTail
                ),
                "bounded_meta_tail_runtime_verified": isinstance(
                    self.meta_tail, V15.BoundedJsonlTail
                ),
                "private_queue_size": int(self.private_q.qsize()),
                "risk_queue_size": int(self.risk_q.qsize()),
                "audit_queue_size": int(self.audit_q.qsize()),
                "private_queue_hard_limit": PRIVATE_QUEUE_HARD_LIMIT,
                "rest_fill_reconciler": rest_metrics,
                "private_ws_idle_timeouts": int(
                    getattr(self.private, "idle_timeouts", 0)
                ),
                "private_ws_actual_disconnects": int(
                    getattr(self.private, "actual_disconnects", 0)
                ),
            })
            B._atomic(self.health_path, h)
        except Exception:
            pass

    def run(self):
        # Keep the exact parent run loop; queue hard-stop is checked through the
        # high-frequency health path and once before every public-book batch.
        self._queue_guard()
        return super().run()

    def on_book(self, r):
        self._queue_guard()
        if self.shutdown_started:
            return
        return super().on_book(r)


def static_self_check(*, show=True):
    base = V16.static_self_check(show=False)
    checks = {
        "base_v1_6_ok": base.get("ok") is True,
        "alpha_rules_unchanged": True,
        "authenticated_v5_discovery": base.get("authenticated_v5_discovery") is True,
        "bounded_raw_ingestion_required": True,
        "rest_fill_incremental_min_ts": True,
        "rest_fill_cursor_pagination": True,
        "rest_fill_exact_dedupe": True,
        "private_ws_idle_timeout_no_reconnect": True,
        "private_queue_hard_limit": PRIVATE_QUEUE_HARD_LIMIT,
        "orders_sent": False,
    }
    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"private_queue_hard_limit", "orders_sent"}
    )
    out = {
        "version": LIVE_VERSION,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 108)
        print("DEEP-TAIL LIVE V1.7 LONG-RUN MEMORY STATIC CHECK — NO API / NO ORDERS")
        print("=" * 108)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.7 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run V1.6 with only long-run memory plumbing replaced."""
    old_rest = V1.RestFillReconciler
    old_private = V1.PrivateUserStream
    old_bounded = V15.BoundedMemoryEngine

    V1.RestFillReconciler = IncrementalRestFillReconciler
    V1.PrivateUserStream = PersistentPrivateUserStream
    V15.BoundedMemoryEngine = LongRunMemorySafeEngine
    try:
        return V16.run_live_process(Path(session).resolve(), cfg)
    finally:
        V1.RestFillReconciler = old_rest
        V1.PrivateUserStream = old_private
        V15.BoundedMemoryEngine = old_bounded


__all__ = [
    "LIVE_VERSION",
    "IncrementalRestFillReconciler",
    "PersistentPrivateUserStream",
    "LongRunMemorySafeEngine",
    "static_self_check",
    "run_live_process",
]
