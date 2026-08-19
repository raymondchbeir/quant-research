from __future__ import annotations

"""Real-money deep-tail 5c + fixed JOIN_ASK live engine.

Strategy hypothesis
-------------------
- Complete 15-minute crypto windows only.
- At M1, submit two passive entries simultaneously:
    BUY YES at 5c  -> V2 book side ``bid`` at yes-price 0.05
    BUY NO  at 5c  -> V2 book side ``ask`` at yes-price 0.95
- Quantity is configured by the launcher and restricted to the research ladder.
- On the FIRST fill on either tail, cancel the opposite tail immediately and never
  add the opposite exposure again in that market.
- Continue resting the selected-tail entry until the FULL requested quantity fills
  or M5 arrives.
- Only after a FULL requested entry is observed, take the freshest causally-known
  public BBO and post one FIXED reduce-only passive exit at the current outcome ask:
    YES position -> V2 ``ask`` at current yes ask
    NO position  -> V2 ``bid`` at current yes bid
  No chasing and no repricing.
- Partial entry at M5 -> no new passive-exit rule; cancel residual and reduce-only
  IOC flatten the actual position.
- M5 is enforced from wall clock independently of book-event backlog.

Execution architecture / lessons retained from the failed Candidate-C live work
-------------------------------------------------------------------------------
- V5 raw public recorder is preserved unchanged for full post-run replay.
- V3 calibrated balance/equity semantics are installed before any risk decision.
- V11 idempotent V2 CREATE path is used: one client_order_id, 409 means recover or
  fail closed; no duplicate replacement order.
- CANCEL uses only documented V2 event-order endpoints, treats ``reduced_by`` as
  matching-engine authoritative, retries safely, then uses the V2 batch-cancel
  fallback; no deprecated V1 mutation endpoint.
- V12.2 exact raw-file EOF freshness barrier is reused as a read-only BBO watchdog.
  Unlike Candidate-C, fixed 5c entries/fixed JOIN_ASK exits are NOT canceled merely
  because the BBO changes after placement.
- Create/cancel REST sessions are prewarmed and separated from risk/audit traffic.
- Authenticated WebSocket ``fill`` + ``user_orders`` is the primary low-latency
  private execution feed. REST fills are an independent reconciliation fallback.
- A dedicated balance thread can trigger the order group immediately on the loss
  threshold without waiting for the strategy loop.
- A dedicated account auditor fails closed on orphan resting orders, unexpected
  position size, or account-state divergence.
- The main loop is event-woken with a 2ms maximum idle wait and drains private fills
  before processing additional raw-book work.

Risk note
---------
The configured start-to-current loss threshold is a SOFTWARE STOP, not a guaranteed
maximum final loss. Network delay, exchange state, fills in flight, market movement,
and liquidation slippage can cause overshoot.
"""

import argparse
import asyncio
import json
import os
import threading
import time
import traceback
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v4 as V4
from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_cycle_q10_live_strategy_v11 as V11
from . import mm_cycle_q10_live_strategy_v12_2 as V122

LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1"
LADDER_Q = (1, 5, 10, 20, 30, 50, 100)
ENTRY_YES_PRICE = 0.05
ENTRY_NO_BOOK_PRICE = 0.95
M1_S = 60.0
M5_S = 300.0
ENTRY_ARM_LATE_TOLERANCE_S = 2.0
PRESEND_EOF_TIMEOUT_MS = 25.0
MAIN_IDLE_WAIT_S = 0.002
REST_FILL_RECONCILE_S = 0.50
RISK_POLL_S = 0.25
ACCOUNT_AUDIT_S = 1.00
EPS = 1e-9

ROOT = C.DATA_ROOT / "deep_tail_join_ask_live_v1"
CONTROL_PATH = ROOT / "active_live.json"
ROOT.mkdir(parents=True, exist_ok=True)


def _wall_ms():
    return time.time_ns() / 1e6


def _track_key(ticker, role, tail):
    return f"{ticker}|{role}|{tail}"


def _is_terminal_status(x):
    return str(x or "").lower() in {"canceled", "cancelled", "executed", "filled"}


def _order_fill_count(row, default=0.0):
    return B._f((row or {}).get("fill_count_fp", (row or {}).get("fill_count")), default)


def _order_remaining(row, default=np.nan):
    return B._f((row or {}).get("remaining_count_fp", (row or {}).get("remaining_count")), default)


def _fill_event_key(msg):
    oid = str((msg or {}).get("order_id") or "")
    trade = str((msg or {}).get("trade_id") or (msg or {}).get("fill_id") or "")
    ts = (msg or {}).get("ts_ms", (msg or {}).get("ts"))
    qty = (msg or {}).get("count_fp", (msg or {}).get("count"))
    return (oid, trade, str(ts), str(qty))


def _safe_cancel_v2(client, *, order_id, submitted_qty):
    """V11-style fail-closed cancel using only documented V2 mutation endpoints."""
    oid = str(order_id)
    submitted_qty = float(submitted_qty)
    errors = []

    for attempt, delay in enumerate((0.0, 0.05, 0.12, 0.25), start=1):
        if delay:
            time.sleep(delay)
        try:
            body, timing = client.delete(
                f"/portfolio/events/orders/{oid}",
                params={"subaccount": 0, "exchange_index": 0},
            )
            reduced = B._f((body or {}).get("reduced_by"), np.nan)
            if not np.isfinite(reduced) or reduced < -EPS or reduced > submitted_qty + EPS:
                raise RuntimeError(f"invalid reduced_by={reduced} body={body}")
            return {
                "ok": True,
                "source": "V2_CANCEL" if attempt == 1 else "V2_CANCEL_RETRY",
                "fill_floor": max(0.0, min(submitted_qty, submitted_qty - max(0.0, reduced))),
                "body": body,
                "timing": timing,
                "errors": errors,
            }
        except Exception as exc:
            errors.append(repr(exc))
            try:
                body, timing = client.get(f"/portfolio/orders/{oid}")
                row = (body or {}).get("order") or {}
                rem = _order_remaining(row, 0.0)
                status = str(row.get("status") or "").lower()
                if rem <= EPS or status != "resting":
                    return {
                        "ok": True,
                        "source": "V2_CANCEL_ERROR_BUT_ORDER_TERMINAL",
                        "fill_floor": min(submitted_qty, max(0.0, _order_fill_count(row, 0.0))),
                        "body": row,
                        "timing": timing,
                        "errors": errors,
                    }
            except Exception:
                pass

    batch_body = None
    batch_timing = None
    batch_error = None
    try:
        batch_body, batch_timing = client.request(
            "DELETE",
            "/portfolio/events/orders/batched",
            payload={
                "orders": [{"order_id": oid, "subaccount": 0, "exchange_index": 0}]
            },
        )
        rows = (batch_body or {}).get("orders") or []
        row = next((r for r in rows if str(r.get("order_id") or "") == oid), None)
        if row is None:
            raise RuntimeError(f"batch response missing order {oid}: {batch_body}")
        if row.get("error"):
            raise RuntimeError(f"batch item error: {row['error']}")
        reduced = B._f(row.get("reduced_by"), np.nan)
        if not np.isfinite(reduced) or reduced < -EPS or reduced > submitted_qty + EPS:
            raise RuntimeError(f"invalid batch reduced_by={reduced}: {row}")
        fill_floor = max(0.0, min(submitted_qty, submitted_qty - max(0.0, reduced)))
    except Exception as exc:
        batch_error = repr(exc)
        fill_floor = np.nan

    time.sleep(0.08)
    try:
        body, verify_timing = client.get(f"/portfolio/orders/{oid}")
        row = (body or {}).get("order") or {}
        rem = _order_remaining(row, np.nan)
        status = str(row.get("status") or "").lower()
        if status == "resting" and (not np.isfinite(rem) or rem > EPS):
            return {
                "ok": False,
                "still_resting": True,
                "fill_floor": max(0.0, _order_fill_count(row, 0.0)),
                "errors": errors,
                "batch_error": batch_error,
                "batch_body": batch_body,
                "verify": row,
                "verify_timing": verify_timing,
            }
        if not np.isfinite(fill_floor):
            fill_floor = max(0.0, _order_fill_count(row, 0.0))
        return {
            "ok": True,
            "source": "V2_BATCH_CANCEL_OR_TERMINAL_VERIFY",
            "fill_floor": min(submitted_qty, max(0.0, float(fill_floor))),
            "errors": errors,
            "batch_error": batch_error,
            "batch_body": batch_body,
            "batch_timing": batch_timing,
            "verify": row,
            "verify_timing": verify_timing,
        }
    except Exception as exc:
        return {
            "ok": False,
            "still_resting": None,
            "fill_floor": 0.0 if not np.isfinite(fill_floor) else float(fill_floor),
            "errors": errors,
            "batch_error": batch_error,
            "verify_error": repr(exc),
        }


class PrewarmedTransportPool:
    def __init__(self, wake_event, workers=9):
        self.wake_event = wake_event
        self.workers = int(workers)
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="dt-live-http")
        self.prewarm_rows = []
        self._prewarm()

    def _client(self):
        c = getattr(self.local, "client", None)
        if c is None:
            c = B.Q1.LiveClient()
            self.local.client = c
        return c

    def _prewarm(self):
        barrier = threading.Barrier(self.workers)

        def warm(slot):
            err = None
            timing = None
            try:
                _, timing = self._client().get("/portfolio/balance")
            except Exception as exc:
                err = repr(exc)
            try:
                barrier.wait(timeout=15.0)
            except Exception as exc:
                err = err or f"barrier:{exc!r}"
            return {"slot": slot, "error": err, "rtt_ms": B._f((timing or {}).get("rtt_ms"))}

        fs = [self.executor.submit(warm, i) for i in range(self.workers)]
        self.prewarm_rows = [f.result(timeout=20.0) for f in fs]
        bad = [r for r in self.prewarm_rows if r.get("error")]
        if bad:
            raise RuntimeError(f"transport worker prewarm failed: {bad}")

    def _notify(self, fut):
        self.wake_event.set()
        return fut

    def create(self, payload):
        f = self.executor.submit(V11._post_v11, self._client_proxy(), payload)
        f.add_done_callback(self._notify)
        return f

    def cancel(self, order_id, qty):
        f = self.executor.submit(_safe_cancel_v2, self._client_proxy(), order_id=order_id, submitted_qty=qty)
        f.add_done_callback(self._notify)
        return f

    def _client_proxy(self):
        # The actual client must be created inside the worker thread. Return a tiny
        # proxy whose methods resolve thread-local state at call time.
        pool = self

        class Proxy:
            def get(self, *a, **k): return pool._client().get(*a, **k)
            def post(self, *a, **k): return pool._client().post(*a, **k)
            def delete(self, *a, **k): return pool._client().delete(*a, **k)
            def request(self, *a, **k): return pool._client().request(*a, **k)
        return Proxy()

    def stop(self):
        self.executor.shutdown(wait=False, cancel_futures=False)


class PrivateUserStream:
    def __init__(self, session: Path, out_queue: Queue, wake_event: threading.Event):
        self.session = Path(session)
        self.out_queue = out_queue
        self.wake_event = wake_event
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._thread_main, name="dt-private-ws", daemon=True)
        self.log_path = self.session / "private_ws_events.jsonl"
        self.reconnects = 0
        self.last_error = None

    def start(self): self.thread.start()

    def stop(self, wait_s=2.0):
        self.stop_event.set()
        self.wake_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=float(wait_s))

    def _thread_main(self):
        asyncio.run(self._run())

    async def _run(self):
        key_id, private_key = C.load_auth()
        while not self.stop_event.is_set():
            self.ready.clear()
            subscribed = set()
            try:
                ws = await C.open_ws(key_id, private_key)
                try:
                    await ws.send(json.dumps({
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {"channels": ["fill", "user_orders"]},
                    }))
                    while not self.stop_event.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        recv_ms = _wall_ms()
                        try:
                            row = json.loads(raw)
                        except Exception:
                            row = {"type": "decode_error", "raw": str(raw)[:2000]}
                        B._append(self.log_path, {"local_recv_wall_ms": recv_ms, "payload": row})
                        typ = str((row or {}).get("type") or "")
                        if typ == "subscribed":
                            ch = str(((row or {}).get("msg") or {}).get("channel") or "")
                            if ch:
                                subscribed.add(ch)
                            if {"fill", "user_orders"}.issubset(subscribed):
                                if not self.ready.is_set():
                                    self.out_queue.put({"kind": "WS_STATE", "state": "READY", "recv_ms": recv_ms})
                                    self.ready.set()
                                    self.wake_event.set()
                        elif typ in {"fill", "user_order"}:
                            self.out_queue.put({"kind": typ.upper(), "msg": (row or {}).get("msg") or {}, "recv_ms": recv_ms})
                            self.wake_event.set()
                finally:
                    try:
                        await ws.close()
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                # A quiet private feed is normal; reconnect only if the socket itself
                # fails. asyncio timeout here just loops by reopening to verify liveness.
                self.last_error = "private websocket receive timeout"
            except Exception as exc:
                self.last_error = repr(exc)
            finally:
                was_ready = self.ready.is_set()
                self.ready.clear()
                self.reconnects += 1
                if was_ready or self.last_error:
                    self.out_queue.put({"kind": "WS_STATE", "state": "DOWN", "error": self.last_error, "recv_ms": _wall_ms()})
                    self.wake_event.set()
            if not self.stop_event.is_set():
                await asyncio.sleep(0.25)


class RestFillReconciler:
    def __init__(self, out_queue, wake_event):
        self.out_queue = out_queue
        self.wake_event = wake_event
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="dt-rest-fill-reconcile", daemon=True)

    def start(self): self.thread.start()
    def stop(self): self.stop_event.set()

    def _run(self):
        client = B.Q1.LiveClient()
        try:
            client.get("/portfolio/balance")
        except Exception:
            pass
        while not self.stop_event.wait(REST_FILL_RECONCILE_S):
            try:
                body, timing = client.get("/portfolio/fills", params={"limit": 1000, "subaccount": 0})
                for r in (body or {}).get("fills") or []:
                    self.out_queue.put({"kind": "REST_FILL", "msg": r, "recv_ms": _wall_ms(), "timing": timing})
                self.wake_event.set()
            except Exception as exc:
                self.out_queue.put({"kind": "REST_FILL_ERROR", "error": repr(exc), "recv_ms": _wall_ms()})
                self.wake_event.set()


class RiskMonitor:
    def __init__(self, start_equity, kill_equity, gid, out_queue, wake_event):
        self.start_equity = float(start_equity)
        self.kill_equity = float(kill_equity)
        self.gid = gid
        self.out_queue = out_queue
        self.wake_event = wake_event
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="dt-risk", daemon=True)
        self.triggered = False

    def start(self): self.thread.start()
    def stop(self): self.stop_event.set()

    def _run(self):
        client = B.Q1.LiveClient()
        try:
            client.get("/portfolio/balance")
        except Exception:
            pass
        while not self.stop_event.wait(RISK_POLL_S):
            try:
                body, timing = client.get("/portfolio/balance", params={"subaccount": 0})
                eq = B._equity(body)
                row = {"kind": "RISK_SNAPSHOT", "equity": eq, "raw": body, "timing": timing, "recv_ms": _wall_ms()}
                self.out_queue.put(row)
                if not self.triggered and float(eq["equity_usd"]) <= self.kill_equity + EPS:
                    self.triggered = True
                    trig = B._trigger_group(client, self.gid)
                    self.out_queue.put({
                        "kind": "RISK_BREACH",
                        "equity": eq,
                        "group_trigger": trig,
                        "recv_ms": _wall_ms(),
                    })
                self.wake_event.set()
            except Exception as exc:
                self.out_queue.put({"kind": "RISK_ERROR", "error": repr(exc), "recv_ms": _wall_ms()})
                self.wake_event.set()


class AccountAuditor:
    def __init__(self, gid, out_queue, wake_event):
        self.gid = str(gid)
        self.out_queue = out_queue
        self.wake_event = wake_event
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="dt-account-audit", daemon=True)

    def start(self): self.thread.start()
    def stop(self): self.stop_event.set()

    def _run(self):
        client = B.Q1.LiveClient()
        try:
            client.get("/portfolio/balance")
        except Exception:
            pass
        while not self.stop_event.wait(ACCOUNT_AUDIT_S):
            try:
                resting, rt = B._resting(client)
                positions, pt = B._positions(client)
                self.out_queue.put({
                    "kind": "ACCOUNT_AUDIT",
                    "resting": resting,
                    "positions": positions,
                    "timing": {"orders": rt, "positions": pt},
                    "recv_ms": _wall_ms(),
                })
                self.wake_event.set()
            except Exception as exc:
                self.out_queue.put({"kind": "ACCOUNT_AUDIT_ERROR", "error": repr(exc), "recv_ms": _wall_ms()})
                self.wake_event.set()


class DeepTailLiveEngine(B.LiveEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if int(round(self.q)) not in LADDER_Q or abs(self.q - round(self.q)) > EPS:
            raise RuntimeError(f"unsupported live ladder quantity Q{self.q}; allowed={LADDER_Q}")

        self.wake_event = threading.Event()
        self.private_q = Queue()
        self.risk_q = Queue()
        self.audit_q = Queue()
        self.dt = defaultdict(lambda: {
            "phase": "WAIT_M1",
            "entry_attempted": False,
            "chosen_tail": None,
            "full_entry_ready": False,
            "exit_posted": False,
            "disabled_reason": None,
        })
        self.order_id_to_key = {}
        self.cid_to_key = {}
        self.pending_creates = {}
        self.pending_cancels = {}
        self.unmatched_private = []
        self.seen_private_fills = set()
        self.counters_dt = Counter()

        self.latency_log = self.session / "deep_tail_latency.jsonl"
        self.transition_log = self.session / "deep_tail_transitions.jsonl"
        self.fill_detail_log = self.session / "deep_tail_actual_fills.jsonl"
        self.audit_log = self.session / "deep_tail_account_audit.jsonl"
        self._lat_lock = threading.Lock()

        # V12.2 watchdog is used only as an independent, exact-EOF public-book
        # freshness source. We never publish active fixed orders into its old
        # Candidate-C invalidation logic.
        self.fast = V122.BarrierFastCancelWatchdog(self)
        self.fast.start()

        self.transport = PrewarmedTransportPool(self.wake_event, workers=9)
        self._lat("TRANSPORT_PREWARMED", rows=self.transport.prewarm_rows)

        self.private = PrivateUserStream(self.session, self.private_q, self.wake_event)
        self.private.start()
        self.rest_fills = RestFillReconciler(self.private_q, self.wake_event)
        self.rest_fills.start()
        self.risk = RiskMonitor(self.start_equity, self.kill_equity, self.gid, self.risk_q, self.wake_event)
        self.risk.start()
        self.auditor = AccountAuditor(self.gid, self.audit_q, self.wake_event)
        self.auditor.start()
        self._lat("DEEP_TAIL_ENGINE_READY", q=self.q, max_loss=self.max_loss)

    def _lat(self, event, **kw):
        row = {"time": B._iso(), "wall_ms": _wall_ms(), "event": event, **kw}
        with self._lat_lock:
            B._append(self.latency_log, row)

    def _transition(self, ticker, event, **kw):
        row = {"time": B._iso(), "event": event, "ticker": ticker, **kw}
        B._append(self.transition_log, row)
        if event in {"ENTRY_PAIR_POSTED", "TAIL_SELECTED", "FULL_ENTRY", "EXIT_POSTED", "M5_FINALIZED", "CRITICAL_DUAL_TAIL_FILL", "WINDOW_DISABLED"}:
            print(f"[{pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S UTC')}] {event} | {ticker} | {kw}", flush=True)

    def wall_elapsed(self, ticker):
        close = pd.to_datetime((self.meta.get(ticker) or {}).get("close_time"), utc=True, errors="coerce")
        if pd.isna(close):
            return np.nan
        return time.time() - float((close - pd.Timedelta(minutes=15)).timestamp())

    def _latest_fresh_bbo(self, ticker):
        sync = self.fast.catch_up_to_stable_eof(PRESEND_EOF_TIMEOUT_MS)
        snap = self.fast.latest_snapshot(ticker)
        if not sync.get("ok") or not snap:
            return None, {"sync": sync, "reason": "NO_CERTIFIED_RAW_STATE"}
        row = (snap or {}).get("row") or {}
        cur = OOS._top_state(row)
        if cur is None:
            return None, {"sync": sync, "reason": "INVALID_BBO", "snapshot": snap}
        receipt_ms = B._f((snap or {}).get("receipt_wall_ms"), np.nan)
        age_ms = _wall_ms() - receipt_ms if np.isfinite(receipt_ms) else np.nan
        return cur, {"sync": sync, "snapshot": snap, "age_ms": age_ms}

    def _new_track(self, ticker, role, tail, side, price, qty, reduce_only):
        key = _track_key(ticker, role, tail)
        cid = f"dtjav1-{self.session.name[-12:]}-{uuid.uuid4().hex}"
        payload = B._payload(
            ticker=ticker,
            side=side,
            qty=qty,
            price=price,
            cid=cid,
            post_only=True,
            reduce_only=bool(reduce_only),
            tif="good_till_canceled",
            group_id=self.gid,
        )
        tr = {
            "key": key,
            "ticker": ticker,
            "role": role,
            "tail": tail,
            "side": side,
            "price": float(price),
            "qty": float(qty),
            "cid": cid,
            "order_id": None,
            "status": "creating",
            "fill_floor": 0.0,
            "fill_event_sum": 0.0,
            "processed_fill": 0.0,
            "cancel_requested": False,
            "cancel_reason": None,
            "created_submit_wall_ms": _wall_ms(),
            "reduce_only": bool(reduce_only),
        }
        self.active[key] = tr
        self.cid_to_key[cid] = key
        B._append(self.orders, {"time": B._iso(), "action": "CREATE_SUBMITTED_ASYNC", "track": tr, "payload": payload})
        fut = self.transport.create(payload)
        self.pending_creates[key] = fut
        return tr

    def _submit_entry_pair(self, ticker, cur, elapsed):
        st = self.dt[ticker]
        st["entry_attempted"] = True
        st["phase"] = "ENTRY_PAIR_CREATING"
        if cur["ask"] <= ENTRY_YES_PRICE + EPS or cur["bid"] >= ENTRY_NO_BOOK_PRICE - EPS:
            st["phase"] = "DISABLED"
            st["disabled_reason"] = "POST_ONLY_CROSS_GUARD"
            self._transition(ticker, "WINDOW_DISABLED", reason=st["disabled_reason"], bid=cur["bid"], ask=cur["ask"])
            return

        yes = self._new_track(ticker, "ENTRY", "YES", "bid", ENTRY_YES_PRICE, self.q, False)
        no = self._new_track(ticker, "ENTRY", "NO", "ask", ENTRY_NO_BOOK_PRICE, self.q, False)
        self.counters_dt["entry_pairs_submitted"] += 1
        self._transition(
            ticker,
            "ENTRY_PAIR_POSTED",
            elapsed_s=elapsed,
            q=self.q,
            yes_cid=yes["cid"],
            no_cid=no["cid"],
            yes_price=ENTRY_YES_PRICE,
            no_book_price=ENTRY_NO_BOOK_PRICE,
        )

    def _request_cancel_key(self, key, reason):
        tr = self.active.get(key)
        if not tr or tr.get("cancel_requested"):
            return
        tr["cancel_requested"] = True
        tr["cancel_reason"] = reason
        if not tr.get("order_id"):
            self._lat("CANCEL_DEFERRED_UNTIL_ORDER_ID", key=key, reason=reason)
            return
        t0 = _wall_ms()
        fut = self.transport.cancel(tr["order_id"], tr["qty"])
        self.pending_cancels[key] = {"future": fut, "requested_ms": t0, "reason": reason}
        self._lat("CANCEL_DISPATCHED", key=key, ticker=tr["ticker"], order_id=tr["order_id"], reason=reason, requested_ms=t0)

    def _cancel_all_for_ticker(self, ticker, reason):
        for key, tr in list(self.active.items()):
            if str(tr.get("ticker")) == str(ticker):
                self._request_cancel_key(key, reason)

    def _apply_floor(self, key, floor, source, raw=None):
        tr = self.active.get(key)
        if tr is None:
            return
        floor = min(tr["qty"], max(0.0, float(floor)))
        tr["fill_floor"] = max(float(tr.get("fill_floor", 0.0)), floor)
        self._process_effective_fill(key, source=source, raw=raw)

    def _apply_fill_event(self, msg, source, recv_ms):
        oid = str((msg or {}).get("order_id") or "")
        key = self.order_id_to_key.get(oid)
        if key is None:
            self.unmatched_private.append({"msg": msg, "source": source, "recv_ms": recv_ms})
            return False
        tr = self.active.get(key)
        if tr is None:
            return True
        ek = _fill_event_key(msg)
        if ek in self.seen_private_fills:
            return True
        self.seen_private_fills.add(ek)
        qty = B._f((msg or {}).get("count_fp", (msg or {}).get("count")), 0.0)
        if qty <= EPS:
            return True
        tr["fill_event_sum"] = min(tr["qty"], float(tr.get("fill_event_sum", 0.0)) + qty)
        post_pos = B._f((msg or {}).get("post_position_fp"), np.nan)
        if np.isfinite(post_pos):
            self.positions[tr["ticker"]] = post_pos
        exch_ms = B._f((msg or {}).get("ts_ms"), np.nan)
        self._lat(
            "PRIVATE_FILL_RECEIVED",
            ticker=tr["ticker"], key=key, order_id=oid, role=tr["role"], tail=tr["tail"],
            qty=qty, source=source, exchange_ts_ms=exch_ms, local_recv_ms=recv_ms,
            exchange_to_local_ms=(recv_ms - exch_ms if np.isfinite(exch_ms) else np.nan),
        )
        self._process_effective_fill(key, source=source, raw=msg)
        return True

    def _process_effective_fill(self, key, source, raw=None):
        tr = self.active.get(key)
        if tr is None:
            return
        effective = min(tr["qty"], max(float(tr.get("fill_floor", 0.0)), float(tr.get("fill_event_sum", 0.0))))
        old = float(tr.get("processed_fill", 0.0))
        if effective <= old + EPS:
            return
        delta = effective - old
        tr["processed_fill"] = effective
        ticker = tr["ticker"]
        B._append(self.fill_detail_log, {
            "time": B._iso(), "ticker": ticker, "key": key, "order_id": tr.get("order_id"),
            "role": tr["role"], "tail": tr["tail"], "delta": delta,
            "effective_fill": effective, "source": source, "raw": raw,
        })
        self.counts["fill_events"] += 1
        self.emit("FILL", ticker, role=tr["role"], tail=tr["tail"], qty=delta, effective_fill=effective, source=source)

        if tr["role"] == "ENTRY":
            st = self.dt[ticker]
            chosen = st.get("chosen_tail")
            if chosen is None:
                st["chosen_tail"] = tr["tail"]
                st["phase"] = "ACCUMULATING_SELECTED_TAIL"
                other = _track_key(ticker, "ENTRY", "NO" if tr["tail"] == "YES" else "YES")
                self._request_cancel_key(other, "OPPOSITE_TAIL_AFTER_FIRST_FILL")
                self._transition(ticker, "TAIL_SELECTED", tail=tr["tail"], first_fill_qty=effective)
            elif chosen != tr["tail"]:
                self._critical_dual_tail(ticker, chosen, tr["tail"], effective)
                return

            if effective >= self.q - EPS and st.get("chosen_tail") == tr["tail"]:
                tr["status"] = "filled"
                self.active.pop(key, None)
                st["full_entry_ready"] = True
                st["phase"] = "FULL_ENTRY_WAITING_OPPOSITE_CANCEL"
                self._transition(ticker, "FULL_ENTRY", tail=tr["tail"], q=self.q)
                self._maybe_post_exit(ticker)

        elif tr["role"] == "EXIT":
            if effective >= tr["qty"] - EPS:
                tr["status"] = "filled"
                self.active.pop(key, None)
                st = self.dt[ticker]
                st["phase"] = "EXIT_FILLED"
                self._transition(ticker, "EXIT_FILLED", tail=tr["tail"], qty=effective, price=tr["price"])

    def _critical_dual_tail(self, ticker, chosen, other, qty):
        self.counters_dt["dual_tail_fill"] += 1
        self._transition(ticker, "CRITICAL_DUAL_TAIL_FILL", chosen=chosen, other=other, qty=qty)
        self.last_error = f"dual-tail fill race {ticker}: chosen={chosen} other={other}"
        B._append(self.risk_log, {"time": B._iso(), "event": "DUAL_TAIL_FILL", "ticker": ticker, "chosen": chosen, "other": other, "qty": qty})
        self.shutdown("DUAL_TAIL_FILL_FAIL_CLOSED")

    def _maybe_post_exit(self, ticker):
        st = self.dt[ticker]
        if not st.get("full_entry_ready") or st.get("exit_posted") or self.shutdown_started:
            return
        chosen = st.get("chosen_tail")
        opposite_key = _track_key(ticker, "ENTRY", "NO" if chosen == "YES" else "YES")
        if opposite_key in self.active:
            return

        cur, cert = self._latest_fresh_bbo(ticker)
        if cur is None:
            st["phase"] = "HOLD_TO_M5_EXIT_NOT_POSTED"
            st["exit_posted"] = True
            self._transition(ticker, "EXIT_NOT_POSTED", reason=cert.get("reason"), cert=cert)
            return

        side = "ask" if chosen == "YES" else "bid"
        price = float(cur["ask"] if chosen == "YES" else cur["bid"])
        # If the fixed quote would already be marketable at this exact certified
        # snapshot, post-only would reject. Do not chase/reprice; M5 fallback only.
        if not (0.0 < price < 1.0):
            st["phase"] = "HOLD_TO_M5_EXIT_NOT_POSTED"
            st["exit_posted"] = True
            self._transition(ticker, "EXIT_NOT_POSTED", reason="INVALID_EXIT_PRICE", price=price)
            return

        tr = self._new_track(ticker, "EXIT", chosen, side, price, self.q, True)
        tr["full_entry_ready_wall_ms"] = _wall_ms()
        st["exit_posted"] = True
        st["phase"] = "EXIT_CREATING"
        self._lat("EXIT_CREATE_DISPATCHED", ticker=ticker, tail=chosen, price=price, cert=cert)
        self._transition(ticker, "EXIT_POSTED", tail=chosen, side=side, price=price, cid=tr["cid"])

    def _drain_create_futures(self):
        for key, fut in list(self.pending_creates.items()):
            if not fut.done():
                continue
            self.pending_creates.pop(key, None)
            tr = self.active.get(key)
            try:
                body, timing = fut.result()
            except Exception as exc:
                if tr and tr["role"] == "EXIT" and any(x in repr(exc).lower() for x in ("post_only", "post-only", "would cross")):
                    self.active.pop(key, None)
                    st = self.dt[tr["ticker"]]
                    st["phase"] = "HOLD_TO_M5_EXIT_REJECTED"
                    self._transition(tr["ticker"], "EXIT_NOT_POSTED", reason="POST_ONLY_REJECT", error=repr(exc))
                    continue
                self.last_error = f"create failure {key}: {exc!r}"
                self.emit("CRITICAL", tr["ticker"] if tr else None, reason="CREATE_FAIL_CLOSED", key=key, error=repr(exc))
                self.shutdown("CREATE_TRANSPORT_FAIL_CLOSED")
                return
            if tr is None:
                # A shutdown may have retired it while CREATE was in flight. The
                # recovered order must be canceled immediately if an id exists.
                oid = str((body or {}).get("order_id") or "")
                if oid:
                    ghost = {"key": key, "ticker": str((body or {}).get("ticker") or ""), "qty": self.q, "order_id": oid}
                    self._lat("CREATE_COMPLETED_AFTER_TRACK_RETIRED", key=key, order_id=oid)
                continue
            oid = str((body or {}).get("order_id") or "")
            if not oid:
                self.last_error = f"create response missing order id {key}: {body}"
                self.shutdown("CREATE_RESPONSE_MISSING_ID")
                return
            tr["order_id"] = oid
            tr["status"] = "resting"
            tr["create_response_wall_ms"] = _wall_ms()
            self.order_id_to_key[oid] = key
            B._append(self.orders, {"time": B._iso(), "action": "CREATE_ACK", "track": tr, "response": body, "timing": timing})
            self._lat("CREATE_ACK", key=key, ticker=tr["ticker"], role=tr["role"], tail=tr["tail"], order_id=oid, timing=timing)
            self._apply_floor(key, B._f((body or {}).get("fill_count"), 0.0), "create_response", body)
            if tr.get("cancel_requested") and key in self.active and key not in self.pending_cancels:
                self._request_cancel_key(key, tr.get("cancel_reason") or "DEFERRED_CANCEL")

        self._retry_unmatched_private()

    def _drain_cancel_futures(self):
        for key, rec in list(self.pending_cancels.items()):
            fut = rec["future"]
            if not fut.done():
                continue
            self.pending_cancels.pop(key, None)
            tr = self.active.get(key)
            try:
                result = fut.result()
            except Exception as exc:
                result = {"ok": False, "exception": repr(exc)}
            self._lat(
                "CANCEL_RESULT",
                key=key,
                ticker=(tr or {}).get("ticker"),
                reason=rec.get("reason"),
                request_to_result_ms=_wall_ms() - float(rec.get("requested_ms", _wall_ms())),
                result=result,
            )
            if not result.get("ok"):
                self.last_error = f"cancel fail-closed {key}: {result}"
                trig = B._trigger_group(self.client, self.gid)
                B._append(self.risk_log, {"time": B._iso(), "event": "CANCEL_FAIL_CLOSED_GROUP_TRIGGER", "key": key, "result": result, "group_trigger": trig})
                self.shutdown("CANCEL_FAIL_CLOSED")
                return
            if tr is not None:
                self._apply_floor(key, B._f(result.get("fill_floor"), 0.0), "cancel_receipt", result)
                tr = self.active.get(key)
                if tr is not None:
                    tr["status"] = "canceled"
                    self.active.pop(key, None)
                    self.counters_dt["cancels"] += 1
                    ticker = tr["ticker"]
                    if tr["role"] == "ENTRY" and self.dt[ticker].get("chosen_tail") == tr["tail"] and tr["processed_fill"] < self.q - EPS:
                        self.dt[ticker]["phase"] = "HOLD_PARTIAL_TO_M5"
                    self._maybe_post_exit(ticker)

    def _handle_user_order(self, msg, recv_ms):
        cid = str((msg or {}).get("client_order_id") or "")
        oid = str((msg or {}).get("order_id") or "")
        key = self.order_id_to_key.get(oid) or self.cid_to_key.get(cid)
        if key is None:
            return
        tr = self.active.get(key)
        if tr is None:
            return
        if oid and not tr.get("order_id"):
            tr["order_id"] = oid
            self.order_id_to_key[oid] = key
            if tr.get("cancel_requested") and key not in self.pending_cancels:
                self._request_cancel_key(key, tr.get("cancel_reason") or "DEFERRED_CANCEL")
        self._apply_floor(key, _order_fill_count(msg, 0.0), "user_orders_ws", msg)
        status = str((msg or {}).get("status") or "").lower()
        if _is_terminal_status(status) and key in self.active and self.active[key].get("processed_fill", 0.0) < self.active[key]["qty"] - EPS:
            # An unexpected terminal selected-entry or exit never gets replaced.
            tr = self.active.pop(key)
            tr["status"] = status
            st = self.dt[tr["ticker"]]
            if tr["role"] == "ENTRY" and st.get("chosen_tail") == tr["tail"]:
                st["phase"] = "HOLD_PARTIAL_TO_M5"
            elif tr["role"] == "EXIT":
                st["phase"] = "HOLD_TO_M5_EXIT_TERMINAL"
            self._transition(tr["ticker"], "ORDER_TERMINAL_NO_REPRICE", role=tr["role"], tail=tr["tail"], status=status)
            self._maybe_post_exit(tr["ticker"])

    def _retry_unmatched_private(self):
        if not self.unmatched_private:
            return
        keep = []
        for x in self.unmatched_private:
            if not self._apply_fill_event(x["msg"], x["source"], x["recv_ms"]):
                if _wall_ms() - float(x["recv_ms"]) < 5000.0:
                    keep.append(x)
        self.unmatched_private = keep[-1000:]

    def _drain_private(self, limit=1000):
        for _ in range(limit):
            try:
                x = self.private_q.get_nowait()
            except Empty:
                break
            kind = x.get("kind")
            if kind == "WS_STATE":
                self._lat("PRIVATE_WS_STATE", **x)
                if x.get("state") == "DOWN":
                    # Safety architecture change: if private execution visibility is
                    # lost, cancel entry orders and disable those windows. Existing
                    # reduce-only fixed exits may remain until M5.
                    for key, tr in list(self.active.items()):
                        if tr.get("role") == "ENTRY":
                            self.dt[tr["ticker"]]["disabled_reason"] = "PRIVATE_WS_DOWN"
                            self._request_cancel_key(key, "PRIVATE_WS_DOWN")
            elif kind in {"FILL", "REST_FILL"}:
                self._apply_fill_event(x.get("msg") or {}, kind, B._f(x.get("recv_ms"), _wall_ms()))
            elif kind == "USER_ORDER":
                self._handle_user_order(x.get("msg") or {}, B._f(x.get("recv_ms"), _wall_ms()))
            elif kind == "REST_FILL_ERROR":
                self._lat("REST_FILL_RECONCILE_ERROR", **x)
        self._retry_unmatched_private()

    def _drain_risk(self):
        for _ in range(100):
            try:
                x = self.risk_q.get_nowait()
            except Empty:
                break
            if x.get("kind") == "RISK_SNAPSHOT":
                eq = x["equity"]
                self.equity = float(eq["equity_usd"])
                self.peak_equity = max(self.peak_equity, self.equity)
                self.max_dd = min(self.max_dd, self.equity - self.peak_equity)
                B._append(self.pnl_log, {
                    "time": B._iso(), **eq,
                    "start_equity_usd": self.start_equity,
                    "start_pnl_usd": self.equity - self.start_equity,
                    "peak_equity_usd": self.peak_equity,
                    "peak_drawdown_usd": self.equity - self.peak_equity,
                    "kill_equity_usd": self.kill_equity,
                    "raw": x.get("raw"), "timing": x.get("timing"),
                    "source": "DEDICATED_RISK_THREAD",
                })
            elif x.get("kind") == "RISK_BREACH":
                B._append(self.risk_log, {"time": B._iso(), "event": "LOSS_LIMIT_PRIORITY_TRIGGER", **x})
                self.emit("LOSS_LIMIT", equity=(x.get("equity") or {}).get("equity_usd"), kill_equity=self.kill_equity)
                self.shutdown("LOSS_LIMIT")
                return
            elif x.get("kind") == "RISK_ERROR":
                self._lat("RISK_MONITOR_ERROR", **x)

    def _drain_audit(self):
        for _ in range(20):
            try:
                x = self.audit_q.get_nowait()
            except Empty:
                break
            B._append(self.audit_log, {"time": B._iso(), **x})
            if x.get("kind") != "ACCOUNT_AUDIT":
                continue
            resting = x.get("resting") or []
            group_resting = [r for r in resting if str(r.get("order_group_id") or "") == str(self.gid)]
            known = {str(tr.get("order_id") or "") for tr in self.active.values() if tr.get("order_id")}
            orphan = [r for r in group_resting if str(r.get("order_id") or "") not in known]
            if orphan:
                self.last_error = f"orphan strategy resting orders: {orphan}"
                self.emit("CRITICAL", reason="ORPHAN_RESTING_ORDER", orders=orphan)
                self.shutdown("ORPHAN_RESTING_ORDER")
                return

            posmap = {}
            for r in x.get("positions") or []:
                t = str(r.get("ticker") or "")
                p = B._f(r.get("position_fp"), 0.0)
                if t and abs(p) > EPS:
                    posmap[t] = p
                    self.positions[t] = p
                    if abs(p) > self.q + 0.02:
                        self.last_error = f"position exceeds Q on {t}: {p}"
                        self.emit("CRITICAL", t, reason="POSITION_LIMIT", position=p, q=self.q)
                        self.shutdown("POSITION_LIMIT")
                        return
            gross = sum(abs(v) for v in posmap.values())
            gross_cap = len(B.SERIES) * self.q + 0.10
            if gross > gross_cap:
                self.last_error = f"gross position {gross} > cap {gross_cap}"
                self.shutdown("GROSS_POSITION_LIMIT")
                return

    def poll_orders(self):
        self._drain_private()
        self._drain_create_futures()
        self._drain_cancel_futures()

    def poll_positions(self):
        self._drain_audit()

    def poll_balance(self):
        self._drain_risk()

    def poll_queue(self):
        # Queue-position REST calls are intentionally removed from the critical
        # path. Initial displayed L1 queue is recoverable from the raw V5 capture;
        # user_orders + fills record actual execution without blocking fill->cancel.
        return

    def cancel_track(self, key_or_ticker, reason):
        if key_or_ticker in self.active:
            self._request_cancel_key(key_or_ticker, reason)
            # During global shutdown the base expects cancellation to have completed
            # synchronously. Wait briefly while priority worker performs V2 cancel.
            deadline = time.time() + 4.0
            while key_or_ticker in self.active and time.time() < deadline:
                self._drain_cancel_futures()
                time.sleep(0.005)
            if key_or_ticker in self.active:
                raise RuntimeError(f"cancel did not retire {key_or_ticker} during {reason}")
            return False
        keys = [k for k, tr in self.active.items() if str(tr.get("ticker")) == str(key_or_ticker)]
        for k in keys:
            self.cancel_track(k, reason)
        return False

    def flatten(self, ticker, reason):
        for key, tr in list(self.active.items()):
            if str(tr.get("ticker")) == str(ticker):
                self.cancel_track(key, reason + "_CANCEL")
        return B.LiveEngine.flatten(self, ticker, reason)

    def finalize_m5(self, ticker):
        if ticker in self.finalized:
            return
        self.finalized.add(ticker)
        self._cancel_all_for_ticker(ticker, "M5")
        deadline = time.time() + 4.0
        while any(str(tr.get("ticker")) == str(ticker) for tr in self.active.values()) and time.time() < deadline:
            self.poll_orders()
            time.sleep(0.005)
        p = self.refresh_position(ticker)
        if abs(p) > EPS:
            self.flatten(ticker, "M5")
        self.dt[ticker]["phase"] = "M5_FINALIZED"
        self._transition(ticker, "M5_FINALIZED", position=self.positions.get(ticker, 0.0))
        self.emit("M5_FINALIZED", ticker, position=self.positions.get(ticker, 0.0))

    def enforce_wall_clock_m5(self):
        tickers = set(self.eligible) | {str(tr.get("ticker")) for tr in self.active.values()} | set(self.positions)
        for ticker in sorted(t for t in tickers if t):
            if ticker in self.finalized:
                continue
            e = self.wall_elapsed(ticker)
            if np.isfinite(e) and e >= M5_S:
                self.finalize_m5(ticker)
                if self.shutdown_started:
                    return

    def on_book(self, r):
        ticker = str(r.get("ticker") or "")
        if not ticker:
            return
        elapsed_row = B._f(r.get("elapsed_s"), np.nan)
        cur = OOS._top_state(r)
        self.book_version[ticker] += 1
        if cur is not None:
            self.current[ticker] = cur
        self.first_book(ticker, elapsed_row)

        e = self.wall_elapsed(ticker)
        if np.isfinite(e) and e >= M5_S:
            self.finalize_m5(ticker)
            return
        if ticker in self.finalized or not self.eligible.get(ticker, False):
            return
        if self.trade_deadline is not None and time.time() >= self.trade_deadline:
            return

        st = self.dt[ticker]
        if not st["entry_attempted"]:
            if np.isfinite(e) and e > M1_S + ENTRY_ARM_LATE_TOLERANCE_S:
                st["entry_attempted"] = True
                st["phase"] = "DISABLED"
                st["disabled_reason"] = "MISSED_M1_ARM_WINDOW"
                self._transition(ticker, "WINDOW_DISABLED", reason=st["disabled_reason"], elapsed_s=e)
                return
            if np.isfinite(e) and M1_S <= e <= M1_S + ENTRY_ARM_LATE_TOLERANCE_S:
                if not self.private.ready.is_set() or not self.fast.ready():
                    st["entry_attempted"] = True
                    st["phase"] = "DISABLED"
                    st["disabled_reason"] = "PRIVATE_OR_RAW_WATCHDOG_NOT_READY_AT_M1"
                    self._transition(ticker, "WINDOW_DISABLED", reason=st["disabled_reason"], elapsed_s=e)
                    return
                fresh_cur, cert = self._latest_fresh_bbo(ticker)
                if fresh_cur is None:
                    st["entry_attempted"] = True
                    st["phase"] = "DISABLED"
                    st["disabled_reason"] = "M1_RAW_FRESHNESS_CERT_FAILED"
                    self._transition(ticker, "WINDOW_DISABLED", reason=st["disabled_reason"], cert=cert)
                    return
                self._submit_entry_pair(ticker, fresh_cur, e)

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update({
                "deep_tail_live_version": LIVE_VERSION,
                "private_ws_ready": self.private.ready.is_set(),
                "private_ws_reconnects": self.private.reconnects,
                "raw_watchdog_ready": self.fast.ready(),
                "active_tracks": {
                    k: {x: tr.get(x) for x in ("ticker", "role", "tail", "side", "price", "qty", "order_id", "status", "processed_fill", "cancel_requested")}
                    for k, tr in self.active.items()
                },
                "deep_tail_states": dict(self.dt),
                "deep_tail_counts": dict(self.counters_dt),
            })
            B._atomic(self.health_path, h)
        except Exception:
            pass

    def shutdown(self, reason):
        if self.shutdown_started:
            return
        # The base shutdown first triggers the order group, then calls our
        # synthetic-key cancel_track and our flatten override. Keep private/risk
        # visibility alive through that cleanup, stop background helpers afterward.
        try:
            B.LiveEngine.shutdown(self, reason)
        finally:
            for obj in (self.private, self.rest_fills, self.risk, self.auditor):
                try:
                    obj.stop()
                except Exception:
                    pass
            try:
                self.fast.stop()
            except Exception:
                pass
            try:
                self.transport.stop()
            except Exception:
                pass

    def run(self):
        self.emit("ENGINE_START", mode=self.mode, deep_tail_live_version=LIVE_VERSION)
        self.health(force=True)
        try:
            while not self.shutdown_started:
                if self.recorder_proc.poll() is not None:
                    raise RuntimeError(f"Raw V5 recorder exited rc={self.recorder_proc.returncode}")

                # Private fills and risk always get priority over public-book work.
                self._drain_private()
                self._drain_risk()
                self._drain_create_futures()
                self._drain_cancel_futures()
                self._drain_audit()
                if self.shutdown_started:
                    break

                self.update_meta()
                rows = self.book_tail.read_new()
                for r in rows:
                    self._drain_private()
                    self._drain_risk()
                    self._drain_create_futures()
                    self._drain_cancel_futures()
                    if self.shutdown_started:
                        break
                    self.on_book(r)
                if self.shutdown_started:
                    break

                self.enforce_wall_clock_m5()
                self.check_end()
                self.health()
                self.wake_event.wait(MAIN_IDLE_WAIT_S)
                self.wake_event.clear()
        except BaseException as exc:
            self.last_error = repr(exc)
            self.emit("ERROR", error=repr(exc), traceback=traceback.format_exc())
            try:
                self.shutdown("ENGINE_EXCEPTION")
            except Exception as cleanup_exc:
                self.emit("CRITICAL", reason="cleanup_exception", error=repr(cleanup_exc))
                try:
                    self.stop_recorder()
                except Exception:
                    pass
            raise
        finally:
            self.health(force=True)


def _write_deep_tail_bundle(session: Path, cfg):
    spec = {
        "time": B._iso(),
        "version": LIVE_VERSION,
        "real_money": True,
        "universe": list(B.SERIES),
        "quantity": float(cfg["quote_size"]),
        "runtime_hours": float(cfg["runtime_hours"]),
        "entry_window": "first certified state from M1=60s through M1+2s; otherwise skip window",
        "entry_yes": {"book_side": "bid", "yes_price": ENTRY_YES_PRICE, "post_only": True},
        "entry_no": {"book_side": "ask", "yes_price": ENTRY_NO_BOOK_PRICE, "equivalent_no_price": 0.05, "post_only": True},
        "first_fill": "cancel opposite-tail entry immediately; selected-tail residual may continue to full Q",
        "full_fill_exit": "one fixed reduce-only post-only GTC at latest certified outcome ask; no reprice",
        "partial_fill": "hold partial until M5 after selected entry terminates/cancels; no new partial-exit rule",
        "m5": "cancel strategy resting orders then reduce-only IOC flatten actual position",
        "loss_limit": {
            "usd_from_starting_calibrated_equity": float(cfg["max_start_loss_usd"]),
            "software_stop_not_guaranteed_final_loss_cap": True,
            "dedicated_balance_poll_seconds": RISK_POLL_S,
        },
        "transport": {
            "create": "V11 idempotent V2 create /portfolio/events/orders",
            "cancel": "V2 single cancel retries -> V2 batch cancel -> fail closed",
            "private_primary": ["fill", "user_orders"],
            "rest_fill_fallback_seconds": REST_FILL_RECONCILE_S,
            "raw_public": "V5 recorder unchanged",
            "raw_freshness": "V12.2 exact EOF catch-up barrier",
            "main_max_idle_seconds": MAIN_IDLE_WAIT_S,
        },
        "recorded": [
            "raw_capture/*", "events.jsonl", "decisions.jsonl", "orders.jsonl",
            "fills.jsonl", "positions.jsonl", "pnl_snapshots.jsonl", "risk_events.jsonl",
            "m5_liquidations.jsonl", "private_ws_events.jsonl", "deep_tail_latency.jsonl",
            "deep_tail_transitions.jsonl", "deep_tail_actual_fills.jsonl", "deep_tail_account_audit.jsonl",
        ],
    }
    B._atomic(session / "deep_tail_strategy_spec.json", spec)
    B._atomic(session / "deep_tail_source_provenance.json", {
        "time": B._iso(),
        "version": LIVE_VERSION,
        "git": V10._git_state(),
        "sources": {
            "base_v1": V10._module_source(B),
            "balance_v3": V10._module_source(V3),
            "recorder_start_v4": V10._module_source(V4),
            "recording_v10": V10._module_source(V10),
            "transport_v11": V10._module_source(V11),
            "freshness_v12_2": V10._module_source(V122),
            "deep_tail": V10._module_source(__import__(__name__, fromlist=["x"])),
        },
    })


def _install_runtime(session: Path, cfg):
    # Calibrate current balance semantics using a fresh authenticated read.
    calibration_client = B.Q1.LiveClient()
    diag = V3._install(calibration_client)
    B._atomic(session / "balance_semantics.json", diag)

    # Preserve the proven recorder-start plumbing and proven idempotent V2 CREATE.
    B._start_recorder = V4._start_recorder_fixed
    B._post = V11._post_v11
    B.LiveEngine = DeepTailLiveEngine

    # Isolate this strategy from the old Candidate-C live control file.
    B.ROOT = ROOT
    B.CONTROL_PATH = CONTROL_PATH
    B.CLIENT_PREFIX = "dtjav1-"
    B.GROUP_LIMIT_FP = str(cfg.get("order_group_limit_fp") or max(100.0, 20.0 * float(cfg["quote_size"])))
    B.LOOP_SLEEP_S = MAIN_IDLE_WAIT_S


def run_live_process(session, cfg):
    session = Path(session).resolve()
    _install_runtime(session, cfg)
    _write_deep_tail_bundle(session, cfg)
    B._run_process(session, cfg)


def static_self_check(*, show=True):
    out = {
        "version": LIVE_VERSION,
        "allowed_ladder": LADDER_Q,
        "entry_yes": ENTRY_YES_PRICE,
        "entry_no_book": ENTRY_NO_BOOK_PRICE,
        "m1": M1_S,
        "m5": M5_S,
        "main_idle_ms": MAIN_IDLE_WAIT_S * 1000.0,
        "risk_poll_ms": RISK_POLL_S * 1000.0,
        "rest_fill_reconcile_ms": REST_FILL_RECONCILE_S * 1000.0,
        "v2_create_function": V11._post_v11.__name__,
        "v12_2_barrier": hasattr(V122.BarrierFastCancelWatchdog, "catch_up_to_stable_eof"),
        "ws_url": C.WS_URL,
        "ok": True,
        "orders_sent": False,
    }
    if show:
        print("=" * 92)
        print("DEEP-TAIL LIVE V1 STATIC SELF-CHECK — NO ORDERS")
        print("=" * 92)
        for k, v in out.items():
            print(f"{k:28s}: {v}")
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    a = ap.parse_args()
    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}
        run_live_process(Path(a.run_live_session), cfg)
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "LADDER_Q",
    "ROOT",
    "CONTROL_PATH",
    "DeepTailLiveEngine",
    "run_live_process",
    "static_self_check",
]
