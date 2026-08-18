from __future__ import annotations

"""V12: V11 order safety + independent priority freshness/cancel watchdog.

Frozen Candidate-C strategy mechanics are unchanged.  V12 changes only live
execution architecture and instrumentation.  The first real run is Q1 one-window;
Q10 is disabled until a fresh Q1 latency audit passes.
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
TARGET_CANCEL_SEND_MS = 25.0
P95_CANCEL_SEND_MS = 50.0
HARD_CANCEL_SEND_MS = 100.0
WATCHDOG_SLEEP_S = 0.001
WATCHDOG_WORKERS = 4
LOOP_SLEEP_S = 0.002
MAX_ACTIONABLE_EXPECTED = 18


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
    """Pure public-book comparison; never reads/mutates account state."""
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
        ds = str(s).lower()
        dp = float(cur["bid"] if ds == "bid" else cur["ask"])
        if ds != side:
            return True, "SIDE_CHANGE"
        if not np.isfinite(px) or abs(dp - px) > 1e-9:
            return True, "PRICE_CHANGE"
        return False, None
    if role == "EXIT":
        if side not in {"bid", "ask"}:
            return True, "INVALID_TRACK_SIDE"
        if not np.isfinite(px):
            return True, "INVALID_TRACK_PRICE"
        dp = float(cur["bid"] if side == "bid" else cur["ask"])
        changed = abs(dp - px) > 1e-9
        return changed, ("PRICE_CHANGE" if changed else None)
    return True, "INVALID_TRACK_ROLE"


def _http_class(method, path):
    m, p = str(method).upper(), str(path)
    if m == "POST" and p == "/portfolio/events/orders": return "CREATE_POST"
    if m == "DELETE" and p == "/portfolio/events/orders/batched": return "CANCEL_BATCH"
    if m == "DELETE" and p.startswith("/portfolio/events/orders/"): return "CANCEL_REQUEST"
    if m == "GET" and p == "/portfolio/balance": return "BALANCE_POLL"
    if m == "GET" and p == "/portfolio/positions": return "POSITION_POLL"
    if m == "GET" and p == "/portfolio/orders": return "RESTING_ORDER_POLL"
    if m == "GET" and p == "/portfolio/orders/queue_positions": return "QUEUE_POLL"
    if m == "GET" and p == "/portfolio/fills": return "FILL_READ"
    return f"{m}_OTHER"


class FastCancelWatchdog:
    """Tails raw book independently and may only cancel already-tracked orders."""

    def __init__(self, engine):
        self.engine = engine
        self.raw_path = engine.session / "raw_capture" / "book_top3_events.jsonl"
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.lock = threading.RLock()
        self.active, self.latest, self.pending = {}, {}, {}
        self.history = defaultdict(lambda: deque(maxlen=256))
        self.seq = 0
        self.results = Queue()
        self.executor = ThreadPoolExecutor(max_workers=WATCHDOG_WORKERS, thread_name_prefix="v12-cancel")
        self.thread_clients = threading.local()
        self.thread = threading.Thread(target=self._run, name="v12-watchdog", daemon=True)
        self.rows_seen = self.invalidations = self.cancel_submissions = self.cancel_errors = 0
        self.max_raw_to_detect_ms = self.max_obsolete_to_send_ms = 0.0

    def start(self):
        self.thread.start()

    def stop(self, wait_s=2.0):
        self.stop_event.set(); self.wake_event.set()
        if self.thread.is_alive(): self.thread.join(timeout=float(wait_s))
        self.executor.shutdown(wait=False, cancel_futures=False)

    def publish_active(self, active):
        snap = {}
        for t, tr in (active or {}).items():
            snap[str(t)] = {k: tr.get(k) for k in ("order_id", "role", "side", "price", "qty", "cid")}
            snap[str(t)]["ticker"] = str(t)
        with self.lock: self.active = snap
        self.wake_event.set()

    def latest_snapshot(self, ticker):
        with self.lock:
            x = self.latest.get(str(ticker))
            return dict(x) if x else None

    def first_newer_before(self, ticker, source_ms, cutoff_ms):
        if not np.isfinite(B._f(source_ms)) or not np.isfinite(B._f(cutoff_ms)): return None
        with self.lock: hist = list(self.history.get(str(ticker)) or [])
        xs = [x for x in hist if np.isfinite(B._f(x.get("receipt_wall_ms")))
              and B._f(x.get("receipt_wall_ms")) > float(source_ms) + 0.001
              and B._f(x.get("receipt_wall_ms")) <= float(cutoff_ms) + 0.001]
        return min(xs, key=lambda x: B._f(x.get("receipt_wall_ms"))) if xs else None

    def is_pending(self, ticker, oid=None):
        t = str(ticker)
        with self.lock:
            if oid is not None: return (t, str(oid)) in self.pending
            return any(k[0] == t for k in self.pending)

    def clear_pending(self, ticker, oid):
        with self.lock: self.pending.pop((str(ticker), str(oid)), None)

    def drain_results(self, limit=100):
        out = []
        for _ in range(int(limit)):
            try: out.append(self.results.get_nowait())
            except Empty: break
        return out

    def _client(self):
        c = getattr(self.thread_clients, "client", None)
        if c is None:
            c = B.Q1.LiveClient(); self.thread_clients.client = c
        return c

    def _run(self):
        while not self.stop_event.is_set() and not self.raw_path.exists(): self.stop_event.wait(0.005)
        if self.stop_event.is_set(): return
        with self.raw_path.open("r", encoding="utf-8") as fh:
            while not self.stop_event.is_set():
                pos = fh.tell(); line = fh.readline()
                if line and not line.endswith("\n"):
                    fh.seek(pos); self.wake_event.wait(WATCHDOG_SLEEP_S); self.wake_event.clear(); continue
                if line:
                    try: row = json.loads(line)
                    except Exception as exc:
                        self.engine._lat("WATCHDOG_JSON_ERROR", error=repr(exc)); continue
                    if isinstance(row, dict): self._process_row(row)
                else:
                    self._recheck_active()
                    self.wake_event.wait(WATCHDOG_SLEEP_S); self.wake_event.clear()

    def _process_row(self, row):
        ticker = str(row.get("ticker") or "")
        if not ticker: return
        detect_ms, receipt_ms = _wall_ms(), _receipt_ms(row)
        raw_to_detect = detect_ms - receipt_ms if np.isfinite(receipt_ms) else np.nan
        self.rows_seen += 1
        if np.isfinite(raw_to_detect): self.max_raw_to_detect_ms = max(self.max_raw_to_detect_ms, raw_to_detect)
        with self.lock:
            self.seq += 1
            item = {"ticker": ticker, "seq": self.seq, "row": row, "receipt_wall_ms": receipt_ms,
                    "watchdog_detect_wall_ms": detect_ms, "raw_to_watchdog_ms": raw_to_detect}
            self.latest[ticker] = item; self.history[ticker].append(dict(item))
            tr = dict(self.active.get(ticker) or {})
        if tr:
            bad, reason = _track_invalidated(tr, row)
            if bad: self._submit(tr, item, reason)

    def _recheck_active(self):
        with self.lock:
            pairs = [(dict(tr), dict(self.latest[t] or {})) for t, tr in self.active.items() if self.latest.get(t)]
        for tr, item in pairs:
            bad, reason = _track_invalidated(tr, item.get("row") or {})
            if bad: self._submit(tr, item, reason)

    def _submit(self, tr, item, reason):
        ticker, oid = str(tr.get("ticker") or ""), str(tr.get("order_id") or "")
        if not ticker or not oid: return
        key = (ticker, oid)
        with self.lock:
            cur = self.active.get(ticker) or {}
            if str(cur.get("order_id") or "") != oid or key in self.pending: return
            ctx = {"invalidation_id": uuid.uuid4().hex, "ticker": ticker, "order_id": oid,
                   "role": tr.get("role"), "side": tr.get("side"), "price": B._f(tr.get("price")),
                   "qty": B._f(tr.get("qty")), "reason": str(reason), "raw_seq": item.get("seq"),
                   "obsolete_receipt_wall_ms": item.get("receipt_wall_ms"),
                   "watchdog_detect_wall_ms": _wall_ms(), "watchdog_raw_to_detect_ms": item.get("raw_to_watchdog_ms")}
            self.pending[key] = ctx; self.invalidations += 1
        self.engine._lat("FAST_INVALIDATION_DETECTED", **ctx)
        self.executor.submit(self._cancel_task, ctx)

    def _cancel_task(self, ctx):
        oid = str(ctx["order_id"]); body = timing = None; error = None
        task_start = _wall_ms()
        try:
            body, timing = self._client().delete(f"/portfolio/events/orders/{oid}",
                                                 params={"subaccount": 0, "exchange_index": 0})
            self.cancel_submissions += 1
        except Exception as exc:
            error = repr(exc); self.cancel_errors += 1
        send = B._f((timing or {}).get("request_send_wall_ms")); recv = B._f((timing or {}).get("response_recv_wall_ms"))
        obs = B._f(ctx.get("obsolete_receipt_wall_ms")); det = B._f(ctx.get("watchdog_detect_wall_ms"))
        o2s = send - obs if np.isfinite(send) and np.isfinite(obs) else np.nan
        d2s = send - det if np.isfinite(send) and np.isfinite(det) else np.nan
        if np.isfinite(o2s): self.max_obsolete_to_send_ms = max(self.max_obsolete_to_send_ms, o2s)
        result = {**ctx, "task_start_wall_ms": task_start, "task_done_wall_ms": _wall_ms(),
                  "cancel_body": body, "cancel_timing": timing, "cancel_error": error,
                  "request_send_wall_ms": send, "response_recv_wall_ms": recv,
                  "obsolete_to_cancel_send_ms": o2s, "detect_to_cancel_send_ms": d2s,
                  "cancel_rtt_ms": B._f((timing or {}).get("rtt_ms")), "success": error is None}
        self.engine._lat("FAST_CANCEL_RESULT", **result); self.results.put(result)

    def metrics(self):
        with self.lock: pending, active, latest = len(self.pending), len(self.active), len(self.latest)
        return {"rows_seen": self.rows_seen, "invalidations": self.invalidations,
                "cancel_submissions": self.cancel_submissions, "cancel_errors": self.cancel_errors,
                "pending": pending, "active_published": active, "latest_tickers": latest,
                "max_raw_to_detect_ms": self.max_raw_to_detect_ms,
                "max_obsolete_to_cancel_send_ms": self.max_obsolete_to_send_ms,
                "workers": WATCHDOG_WORKERS}


class PriorityFreshnessEngine(V11.V2OnlyProductionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latency_log_v12 = self.session / "latency_events_v12.jsonl"
        self._lat_lock = threading.Lock(); self._api_context = {}
        self._request_original = self.client.request; self._instrument_client()
        self.v12 = Counter(); self.max_actionable = 0
        self.fast = FastCancelWatchdog(self); self.fast.start(); self._publish()

    def _lat(self, event, **kw):
        row = {"time": B._iso(), "wall_ms": _wall_ms(), "perf_ms": _perf_ms(),
               "event": str(event), "engine": LIVE_VERSION, **kw}
        with self._lat_lock: B._append(self.latency_log_v12, row)

    def _instrument_client(self):
        def req(method, path, *, params=None, payload=None, timeout=8.0):
            ws, ps, timing, err = _wall_ms(), _perf_ms(), None, None
            ctx = dict(self._api_context or {})
            try:
                body, timing = self._request_original(method, path, params=params, payload=payload, timeout=timeout)
                return body, timing
            except Exception as exc:
                err = repr(exc); raise
            finally:
                self._lat("MAIN_HTTP", method=str(method).upper(), path=str(path),
                          http_class=_http_class(method, path), call_start_wall_ms=ws,
                          local_call_ms=_perf_ms()-ps,
                          request_send_wall_ms=B._f((timing or {}).get("request_send_wall_ms")),
                          response_recv_wall_ms=B._f((timing or {}).get("response_recv_wall_ms")),
                          rtt_ms=B._f((timing or {}).get("rtt_ms")), error=err, context=ctx)
        self.client.request = req

    def _publish(self):
        if hasattr(self, "fast"): self.fast.publish_active(self.active)

    def _consume_fast(self, limit=100):
        for r in self.fast.drain_results(limit=limit):
            ticker, oid = str(r.get("ticker") or ""), str(r.get("order_id") or "")
            self.fast.clear_pending(ticker, oid); self.v12["fast_results"] += 1
            tr = self.active.get(ticker)
            if not tr or str(tr.get("order_id") or "") != oid:
                self.v12["stale_fast_results"] += 1; continue
            if not r.get("success"):
                self.v12["fast_fallbacks"] += 1
                self._api_context = {"v12": "FAST_CANCEL_ERROR_FALLBACK", "ticker": ticker, "order_id": oid}
                try: super().cancel_track(ticker, "V12_FAST_CANCEL_ERROR_FALLBACK")
                finally: self._api_context = {}
                self._publish(); continue

            old_fill, submitted = float(tr.get("last_fill", 0.0)), float(tr.get("qty", 0.0))
            before_pos = float(self.positions.get(ticker, 0.0))
            reduced = self._parse_reduced_by(r.get("cancel_body") or {}, submitted)
            final_fill = min(submitted, max(old_fill, submitted - reduced)); tr["last_fill"] = final_fill
            self._api_context = {"v12": "FAST_CANCEL_RECONCILE", "ticker": ticker, "order_id": oid}
            try:
                self.record_fills(tr); after_pos = self.refresh_position(ticker)
            finally: self._api_context = {}
            B._append(self.orders, {"time": B._iso(), "action": "CANCEL_V12_PRIORITY_FAST_PATH",
                                    "ticker": ticker, "reason": r.get("reason"), "track": tr,
                                    "invalidation_id": r.get("invalidation_id"),
                                    "obsolete_receipt_wall_ms": r.get("obsolete_receipt_wall_ms"),
                                    "cancel_body": r.get("cancel_body"), "cancel_timing": r.get("cancel_timing"),
                                    "obsolete_to_cancel_send_ms": r.get("obsolete_to_cancel_send_ms"),
                                    "detect_to_cancel_send_ms": r.get("detect_to_cancel_send_ms"),
                                    "old_fill": old_fill, "final_fill": final_fill,
                                    "position_before": before_pos, "position_after": after_pos})
            self.active.pop(ticker, None); self.counts["cancels"] += 1; self.v12["fast_success"] += 1
            raced = final_fill > old_fill + B.EPS or abs(after_pos-before_pos) > B.EPS
            if raced:
                self.v12["fast_raced_fill"] += 1; self.barrier[ticker] = self.book_version[ticker]
                self.counts["fill_events"] += 1
                self.emit("FILL", ticker, role=tr["role"], side=tr["side"],
                          qty=max(0.0, final_fill-old_fill), position=after_pos,
                          source="v12_priority_fast_cancel")
            self._publish()

    def cancel_track(self, ticker, reason):
        tr = self.active.get(ticker)
        if not tr: return False
        oid = str(tr.get("order_id") or "")
        if self.fast.is_pending(ticker, oid):
            self._consume_fast()
            tr = self.active.get(ticker)
            if not tr: return False
            oid = str(tr.get("order_id") or "")
            if self.fast.is_pending(ticker, oid):
                if str(reason).upper().startswith(("M5", "SHUTDOWN", "WALL_CLOCK")):
                    end = time.time() + 0.15
                    while time.time() < end and self.fast.is_pending(ticker, oid):
                        time.sleep(0.002); self._consume_fast()
                    if ticker not in self.active: return False
                else: return False
        self._api_context = {"v12": "NORMAL_CANCEL", "ticker": ticker, "order_id": oid, "reason": reason}
        try: raced = super().cancel_track(ticker, reason)
        finally: self._api_context = {}
        self._publish(); return raced

    def reconcile(self, ticker, cur, elapsed):
        tr = self.active.get(ticker)
        if tr and self.fast.is_pending(ticker, tr.get("order_id")): return
        return super().reconcile(ticker, cur, elapsed)

    def poll_orders(self):
        now = time.time()
        if now - self.t_order < B.ORDER_POLL_S: return
        self.t_order = now
        resting, _ = B._resting(self.client); by_id = {str(r.get("order_id") or ""): r for r in resting}
        transitions = 0
        for ticker, tr in list(self.active.items()):
            oid = str(tr.get("order_id") or "")
            if self.fast.is_pending(ticker, oid): continue
            row, old = by_id.get(oid), float(tr.get("last_fill", 0.0))
            if row is not None:
                fc = B._f(row.get("fill_count_fp", row.get("fill_count")), old)
                if fc > old + B.EPS: self.cancel_track(ticker, "RESTING_SET_FILL_DETECTED"); transitions += 1
            else:
                created = float(tr.get("created_wall", 0.0) or 0.0)
                if created > 0 and now-created < V6.ORDER_ABSENCE_GRACE_S: continue
                self.cancel_track(ticker, "ABSENT_FROM_RESTING_SET"); transitions += 1
            if transitions >= V6.MAX_ORDER_STATE_TRANSITIONS_PER_POLL: break

    def place(self, ticker, d, cur, elapsed):
        source_ms = _receipt_ms(self.latest_rows.get(ticker) or {})
        fast = self.fast.latest_snapshot(ticker); fast_ms = B._f((fast or {}).get("receipt_wall_ms"))
        newer = np.isfinite(source_ms) and np.isfinite(fast_ms) and fast_ms > source_ms + 0.001
        comparable = bool(fast and (not np.isfinite(source_ms) or (np.isfinite(fast_ms) and fast_ms+0.001 >= source_ms)))
        bad, bad_reason = _track_invalidated(d, (fast or {}).get("row") or {}) if comparable else (False, None)
        if newer or bad:
            self.v12["create_guard_blocks"] += 1
            self._lat("CREATE_BLOCKED_FRESHNESS_GUARD", ticker=ticker, role=d.get("role"), side=d.get("side"),
                      price=d.get("price"), source_receipt_wall_ms=source_ms,
                      fast_receipt_wall_ms=fast_ms, newer_fast_row=newer,
                      invalid_fast=bad, invalid_reason=bad_reason)
            return

        now = time.time(); wall_e = self.wall_elapsed(ticker, now_s=now); age = self.row_age_s(ticker, now_s=now)
        if not np.isfinite(wall_e) or not (0.0 <= wall_e < 300.0) or not np.isfinite(age) or age > V6.MAX_ACTION_BOOK_AGE_S or self.shutdown_started:
            self.v7_stale_create_blocks += 1; return

        cid = B.CLIENT_PREFIX + uuid.uuid4().hex
        payload = B._payload(ticker=ticker, side=d["side"], qty=d["qty"], price=d["price"], cid=cid,
                             post_only=True, reduce_only=False, tif="good_till_canceled", group_id=self.gid)
        decision_ms, version = _wall_ms(), int(self.book_version[ticker])
        B._append(self.decisions, {"time": B._iso(), "ticker": ticker, "series": self.series(ticker),
                                   "close_time": self.window_key(ticker),
                                   "row_elapsed_s": B._f((self.latest_rows.get(ticker) or {}).get("elapsed_s")),
                                   "wall_elapsed_s": wall_e, "book_age_s": age,
                                   "position": self.positions.get(ticker, 0.0), **d, "book": cur,
                                   "engine": LIVE_VERSION, "decision_book_version": version,
                                   "source_receipt_wall_ms": source_ms, "fast_guard_receipt_wall_ms": fast_ms})
        self._api_context = {"v12": "CREATE", "ticker": ticker, "cid": cid,
                             "decision_wall_ms": decision_ms, "source_receipt_wall_ms": source_ms}
        try: body, timing = B._post(self.client, payload)
        finally: self._api_context = {}
        oid = str(body.get("order_id") or "")
        if not oid: raise RuntimeError(f"Create response missing order_id: {body}")
        send = B._f((timing or {}).get("request_send_wall_ms")); recv = B._f((timing or {}).get("response_recv_wall_ms"))
        d2s = send-decision_ms if np.isfinite(send) else np.nan
        age_send = send-source_ms if np.isfinite(send) and np.isfinite(source_ms) else np.nan
        newer_before_send = self.fast.first_newer_before(ticker, source_ms, send) if np.isfinite(send) else None
        superseded = newer_before_send is not None
        if superseded: self.v12["superseded_at_send"] += 1

        tr = {"ticker": ticker, "order_id": oid, "cid": cid, "role": d["role"], "side": d["side"],
              "price": d["price"], "qty": d["qty"], "last_fill": 0.0, "book": cur,
              "elapsed_s": wall_e, "row_elapsed_s": B._f((self.latest_rows.get(ticker) or {}).get("elapsed_s")),
              "book_age_s": age, "created_wall": time.time(), "decision_book_version": version,
              "source_receipt_wall_ms": source_ms}
        self.active[ticker] = tr; self._publish()
        B._append(self.orders, {"time": B._iso(), "action": "CREATE_V12_FRESHNESS_GUARDED", "ticker": ticker,
                                "payload": payload, "response": body, "timing": timing,
                                "wall_elapsed_s": wall_e, "book_age_s": age,
                                "decision_book_version": version, "source_receipt_wall_ms": source_ms,
                                "decision_to_send_ms": d2s, "source_age_at_send_ms": age_send,
                                "superseded_at_send_detected": superseded})
        self.counts["orders"] += 1; self.last_action_wall = time.time()
        self.v7_max_sent_order_book_age_s = max(self.v7_max_sent_order_book_age_s, float(age))
        self._lat("CREATE_SENT", ticker=ticker, order_id=oid, client_order_id=cid, role=d.get("role"),
                  decision_wall_ms=decision_ms, request_send_wall_ms=send, response_recv_wall_ms=recv,
                  decision_to_send_ms=d2s, source_receipt_wall_ms=source_ms,
                  source_age_at_send_ms=age_send, decision_book_version=version,
                  first_newer_before_send_receipt_wall_ms=B._f((newer_before_send or {}).get("receipt_wall_ms")),
                  superseded_at_send_detected=superseded)
        self.emit("ENTRY_ORDER" if d["role"] == "ENTRY" else "EXIT_ORDER", ticker,
                  role=d["role"], side=d["side"], qty=d["qty"], price=d["price"],
                  wall_elapsed_s=wall_e, book_age_s=age, decision_to_send_ms=d2s)
        if B._f(body.get("fill_count"), 0.0) > B.EPS: self.cancel_track(ticker, "CREATE_RESPONSE_FILL")

    def _prune(self):
        for t in list(self.latest_rows):
            if t in self.active: continue
            e = self.wall_elapsed(t); remove = t in self.finalized
            if t in self.first_seen and self.eligible.get(t) is False:
                remove = remove or (np.isfinite(e) and e >= 300.0)
            if remove:
                self.latest_rows.pop(t, None); self.current.pop(t, None); self.pending_first_rows.pop(t, None)

    def _actionable(self):
        out = []
        for t in self.latest_rows:
            if t in self.active: out.append(t); continue
            if t in self.finalized or not self.eligible.get(t, False): continue
            e = self.wall_elapsed(t)
            if np.isfinite(e) and 0.0 <= e < 300.0: out.append(t)
        self.max_actionable = max(self.max_actionable, len(out)); return sorted(out)

    def process_latest_states(self):
        self._consume_fast(); self._prune(); keys = self._actionable()
        if not keys: return
        start = self.rr_cursor % len(keys); ordered = keys[start:] + keys[:start]; visited = 0
        for ticker in ordered:
            visited += 1
            if self.shutdown_started: break
            self._consume_fast()
            tr = self.active.get(ticker)
            if tr and self.fast.is_pending(ticker, tr.get("order_id")): continue
            e, age = self.wall_elapsed(ticker), self.row_age_s(ticker)
            if np.isfinite(age): self.max_latest_book_age_s = max(self.max_latest_book_age_s, age)
            if not np.isfinite(e) or not (0.0 <= e < 300.0):
                if ticker in self.active: self.cancel_track(ticker, "WALL_CLOCK_OUTSIDE_WINDOW")
                continue
            if not np.isfinite(age) or age > V6.MAX_ACTION_BOOK_AGE_S:
                if ticker in self.active: self.cancel_track(ticker, "STALE_LATEST_BOOK")
                continue
            cur = _top(self.latest_rows.get(ticker) or {}); self.current[ticker] = cur
            if cur is None:
                if ticker in self.active: self.cancel_track(ticker, "INVALID_LATEST_BOOK")
                continue
            self.reconcile(ticker, cur, e)
        self.rr_cursor = (start + max(1, visited)) % len(keys)

    def ingest_book_batch(self, rows):
        if not rows: return
        t0 = _perf_ms(); super().ingest_book_batch(rows)
        self._lat("MAIN_BOOK_INGEST", rows=len(rows), ingest_ms=_perf_ms()-t0)

    def run(self):
        self.emit("ENGINE_START", mode=self.mode, engine=LIVE_VERSION); self.health(force=True)
        try:
            while not self.shutdown_started:
                if self.recorder_proc.poll() is not None:
                    raise RuntimeError(f"Raw V5 recorder exited rc={self.recorder_proc.returncode}")
                t0 = _perf_ms(); self.update_meta()
                r0 = _perf_ms(); rows = self.book_tail.read_new(); read_ms = _perf_ms()-r0
                if rows:
                    self._lat("MAIN_BOOK_READ", rows=len(rows), read_ms=read_ms); self.ingest_book_batch(rows)
                self._consume_fast(); self.check_end()
                if self.shutdown_started: break
                self.enforce_wall_clock_m5()
                if self.shutdown_started: break
                self.process_latest_states(); self._consume_fast()
                self.risk_tick()
                self._lat("MAIN_LOOP", loop_ms=_perf_ms()-t0, actionable_tickers=len(self._actionable()),
                          active_orders=len(self.active))
                time.sleep(LOOP_SLEEP_S)
        except BaseException as exc:
            self.last_error = repr(exc); self.emit("ERROR", error=repr(exc), traceback=__import__("traceback").format_exc())
            try: self.shutdown("ENGINE_EXCEPTION")
            except Exception as cleanup_exc:
                self.emit("CRITICAL", error=repr(cleanup_exc), reason="cleanup_exception")
                try: self.fast.stop()
                except Exception: pass
                self.stop_recorder()
            raise
        finally: self.health(force=True)

    def _v12_metrics(self):
        return {"live_version": LIVE_VERSION, "freshness_arch_version": FRESHNESS_ARCH_VERSION,
                "target_cancel_send_ms": TARGET_CANCEL_SEND_MS, "p95_cancel_send_ms": P95_CANCEL_SEND_MS,
                "hard_cancel_send_ms": HARD_CANCEL_SEND_MS, "watchdog": self.fast.metrics(),
                "create_guard_blocks": int(self.v12["create_guard_blocks"]),
                "create_superseded_at_send_detected": int(self.v12["superseded_at_send"]),
                "fast_cancel_successes": int(self.v12["fast_success"]),
                "fast_cancel_fallbacks": int(self.v12["fast_fallbacks"]),
                "fast_cancel_raced_fills": int(self.v12["fast_raced_fill"]),
                "max_actionable_tickers": self.max_actionable,
                "max_actionable_tickers_expected": MAX_ACTIONABLE_EXPECTED,
                "strategy_changed": False, "q10_launcher_enabled": False}

    def _v10_metrics(self):
        x = super()._v10_metrics(); x.update({"live_version": LIVE_VERSION,
                                             "execution_parent": EXECUTION_PARENT,
                                             "v12_priority_freshness": self._v12_metrics()}); return x

    def health(self, force=False):
        super().health(force=force); h = B._read(self.health_path, {}) or {}
        h["live_version"] = LIVE_VERSION; h["execution_parent"] = EXECUTION_PARENT
        h["v12_metrics"] = self._v12_metrics(); B._atomic(self.health_path, h)

    def shutdown(self, reason):
        if self.shutdown_started: return
        self.fast.stop(); self._consume_fast(limit=1000); super().shutdown(reason)
        s = B._read(self.final_path, {}) or {}; s["live_wrapper_version"] = LIVE_VERSION
        s["execution_parent"] = EXECUTION_PARENT; s["freshness_arch_version"] = FRESHNESS_ARCH_VERSION
        s["v12_metrics"] = self._v12_metrics(); B._atomic(self.final_path, s)
        h = B._read(self.health_path, {}) or {}; h["live_version"] = LIVE_VERSION
        h["execution_parent"] = EXECUTION_PARENT; h["v12_metrics"] = self._v12_metrics(); h["summary"] = s
        B._atomic(self.health_path, h)
        try: audit_v12_smoke(self.session, show=False, write_result=True)
        except Exception as exc: self._lat("SMOKE_AUDIT_ERROR", error=repr(exc))


def _read_jsonl(path):
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                try: x = json.loads(line)
                except Exception: continue
                if isinstance(x, dict): rows.append(x)
    except FileNotFoundError: pass
    return rows


def _stats(vals):
    x = pd.to_numeric(pd.Series(list(vals), dtype="float64"), errors="coerce"); x = x[np.isfinite(x)]
    if not len(x): return {"n": 0, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {"n": int(len(x)), "median": float(x.median()), "p90": float(x.quantile(.90)),
            "p95": float(x.quantile(.95)), "p99": float(x.quantile(.99)), "max": float(x.max())}


def audit_v12_smoke(session_dir, *, show=True, write_result=True):
    """Read-only Q1 architecture/latency audit.  Sends no orders and calls no API."""
    session = Path(session_dir).resolve(); ev = _read_jsonl(session / "latency_events_v12.jsonl")
    final = B._read(session / "final_summary.json", {}) or {}
    fast = [r for r in ev if r.get("event") == "FAST_CANCEL_RESULT" and r.get("success")]
    inv = [r for r in ev if r.get("event") == "FAST_INVALIDATION_DETECTED"]
    creates = [r for r in ev if r.get("event") == "CREATE_SENT"]
    guards = [r for r in ev if r.get("event") == "CREATE_BLOCKED_FRESHNESS_GUARD"]
    http = [r for r in ev if r.get("event") == "MAIN_HTTP"]
    o2s = _stats(r.get("obsolete_to_cancel_send_ms") for r in fast)
    raw2det = _stats(r.get("watchdog_raw_to_detect_ms") for r in fast)
    det2send = _stats(r.get("detect_to_cancel_send_ms") for r in fast)
    crtt = _stats(r.get("cancel_rtt_ms") for r in fast)
    d2send = _stats(r.get("decision_to_send_ms") for r in creates)
    source_age = _stats(r.get("source_age_at_send_ms") for r in creates)
    http_by_class = {}
    for cls in sorted({str(r.get("http_class")) for r in http}):
        rows = [r for r in http if str(r.get("http_class")) == cls]
        http_by_class[cls] = {"calls": len(rows), "rtt_ms": _stats(r.get("rtt_ms") for r in rows),
                              "local_call_ms": _stats(r.get("local_call_ms") for r in rows)}
    superseded = sum(bool(r.get("superseded_at_send_detected")) for r in creates)
    safety = bool(final and final.get("flat_verified") is True
                  and final.get("strategy_resting_orders_zero") is True
                  and final.get("last_error") in (None, ""))
    observed = o2s["n"] > 0
    cancel_pass = bool(observed and o2s["p95"] <= P95_CANCEL_SEND_MS and o2s["max"] <= HARD_CANCEL_SEND_MS)
    create_pass = superseded == 0
    m = final.get("v12_metrics") or {}; ma = B._f(m.get("max_actionable_tickers"))
    actionable_pass = bool(not np.isfinite(ma) or ma <= MAX_ACTIONABLE_EXPECTED)
    promotion = bool(safety and observed and cancel_pass and create_pass and actionable_pass)
    out = {"time": B._iso(), "session_dir": str(session), "live_version": LIVE_VERSION,
           "orders_sent": False, "exchange_api_called": False,
           "final": {"shutdown_reason": final.get("shutdown_reason"),
                     "account_pnl_usd": final.get("account_pnl_usd"),
                     "flat_verified": final.get("flat_verified"),
                     "strategy_resting_orders_zero": final.get("strategy_resting_orders_zero"),
                     "last_error": final.get("last_error")},
           "counts": {"fast_invalidations": len(inv), "fast_cancel_success_results": len(fast),
                      "creates": len(creates), "create_guard_blocks": len(guards),
                      "superseded_at_send_detected": superseded, "main_http_calls": len(http)},
           "latency": {"raw_receipt_to_watchdog_detect_ms": raw2det,
                       "watchdog_detect_to_cancel_send_ms": det2send,
                       "obsolete_receipt_to_cancel_send_ms": o2s,
                       "priority_cancel_rtt_ms": crtt, "decision_to_create_send_ms": d2send,
                       "source_receipt_age_at_create_send_ms": source_age,
                       "main_http_by_class": http_by_class},
           "slo": {"target_cancel_send_ms": TARGET_CANCEL_SEND_MS,
                   "p95_cancel_send_ms": P95_CANCEL_SEND_MS, "hard_cancel_send_ms": HARD_CANCEL_SEND_MS},
           "gates": {"safety_pass": safety, "create_freshness_pass": create_pass,
                     "actionable_set_pass": actionable_pass, "cancel_latency_observed": observed,
                     "cancel_latency_pass": cancel_pass,
                     "promotion_ready_for_larger_smoke": promotion},
           "interpretation": ("PASS" if promotion else
                              "INCOMPLETE_LATENCY_PROOF_NO_INVALIDATION" if safety and create_pass and not observed
                              else "FAIL_INVESTIGATE")}
    if write_result: B._atomic(session / "v12_smoke_latency_audit.json", out)
    if show:
        print("="*100); print("V12 Q1 SMOKE LATENCY AUDIT — READ ONLY"); print("="*100)
        print("Session:", session); print("PnL:", out["final"]["account_pnl_usd"])
        print("Fast invalidations / cancels:", len(inv), len(fast)); print("Create guard blocks:", len(guards))
        print("Superseded creates at send:", superseded); print("Obsolete->cancel send:", o2s)
        print("Raw->watchdog:", raw2det); print("Decision->create send:", d2send); print("Cancel RTT:", crtt)
        print("Gates:", out["gates"]); print("ORDERS SENT BY AUDIT: NO | EXCHANGE API CALLED BY AUDIT: NO")
    return out


def _write_v12_bundle(session: Path, cfg):
    session = Path(session).resolve(); V11._write_v11_bundle(session, cfg)
    prov = B._read(session / "source_provenance.json", {}) or {}
    prov.update({"live_version": LIVE_VERSION, "execution_parent": EXECUTION_PARENT,
                 "recording_parent": RECORDING_PARENT, "freshness_arch_version": FRESHNESS_ARCH_VERSION})
    src = dict(prov.get("sources") or {}); src["v12"] = V10._module_source(sys.modules[__name__]); prov["sources"] = src
    B._atomic(session / "source_provenance.json", prov)
    spec = B._read(session / "live_execution_spec_v11.json", {}) or {}
    spec.update({"live_version": LIVE_VERSION, "execution_parent": EXECUTION_PARENT,
                 "freshness_arch_version": FRESHNESS_ARCH_VERSION, "strategy_mechanics_changed": False,
                 "priority_raw_watchdog": True, "priority_cancel_workers": WATCHDOG_WORKERS,
                 "create_guard_against_watchdog_latest": True, "historical_actionable_rows_pruned": True,
                 "risk_tick_per_historical_key": False, "latency_log": "latency_events_v12.jsonl",
                 "q1_first": True, "q10_launcher_enabled": False})
    B._atomic(session / "live_execution_spec_v12.json", spec)
    B._atomic(session / "latency_slo_v12.json", {"time": B._iso(), "version": FRESHNESS_ARCH_VERSION,
               "target_obsolete_to_cancel_send_ms": TARGET_CANCEL_SEND_MS,
               "p95_obsolete_to_cancel_send_ms": P95_CANCEL_SEND_MS,
               "hard_obsolete_to_cancel_send_ms": HARD_CANCEL_SEND_MS,
               "strategy_thresholds_changed": False})


def _run_process_v12(session, cfg):
    session = Path(session).resolve(); _write_v12_bundle(session, cfg)
    client = B.Q1.LiveClient(); diag = V3._install(client); B._atomic(session / "balance_semantics.json", diag)
    B._post = V11._post_v11; B._start_recorder = V4._start_recorder_fixed; B.LiveEngine = PriorityFreshnessEngine
    B._run_process(session, cfg)


def live_preflight(*, quote_size=B.SMOKE_Q, runtime_hours=1.0,
                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                   min_start_equity_usd=B.SMOKE_MIN_EQUITY, show=True):
    return V3.live_preflight(quote_size=quote_size, runtime_hours=runtime_hours,
                             max_start_loss_usd=max_start_loss_usd,
                             min_start_equity_usd=min_start_equity_usd, show=show)


def start_live_smoke_q1_one_window(*, arm_phrase=None,
                                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                                   min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    if str(arm_phrase) != B.SMOKE_ARM:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={B.SMOKE_ARM!r} exactly.")
    old = B._ctl()
    if old and B._pid_alive(old.get("pid")): raise RuntimeError(f"A live process is already running: {old}")
    V3._calibrated_preflight(quote_size=B.SMOKE_Q, runtime_hours=1.0,
                             max_loss_usd=float(max_start_loss_usd),
                             min_equity_usd=float(min_start_equity_usd),
                             mode="SMOKE_Q1_ONE_WINDOW_V12", save_dir=None, show=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_smoke_q1_one_window_v12").resolve(); session.mkdir(parents=True, exist_ok=False)
    cfg = {"mode": "SMOKE_Q1_ONE_WINDOW", "quote_size": float(B.SMOKE_Q), "runtime_hours": 1.0,
           "max_start_loss_usd": float(max_start_loss_usd), "min_start_equity_usd": float(min_start_equity_usd),
           "live_wrapper_version": LIVE_VERSION, "execution_parent": EXECUTION_PARENT,
           "recording_parent": RECORDING_PARENT, "freshness_arch_version": FRESHNESS_ARCH_VERSION,
           "engine_architecture": "V11_ORDER_SAFETY_PLUS_V12_PRIORITY_FRESHNESS_WATCHDOG",
           "recording_version": V10.RECORDING_VERSION,
           "comparison_schema_version": V10.COMPARISON_SCHEMA_VERSION,
           "order_api_safety_version": V11.ORDER_API_SAFETY_VERSION, "q10_launcher_enabled": False}
    B._atomic(session / "process_config.json", cfg); _write_v12_bundle(session, cfg)
    log = session / "live_process.log"; fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen([sys.executable, "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v12",
                              "--run-live-session", str(session), "--config", str(session / "process_config.json")],
                             cwd=str(C.PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    finally: fh.close()
    B._atomic(B.CONTROL_PATH, {"live_version": LIVE_VERSION, "execution_parent": EXECUTION_PARENT,
              "recording_parent": RECORDING_PARENT, "freshness_arch_version": FRESHNESS_ARCH_VERSION,
              "recording_version": V10.RECORDING_VERSION, "order_api_safety_version": V11.ORDER_API_SAFETY_VERSION,
              "running": True, "pid": p.pid, "session_dir": str(session), "mode": "SMOKE_Q1_ONE_WINDOW",
              "started_at": B._iso(), "config": cfg, "log_path": str(log)})
    deadline, last = time.time()+90.0, {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-20000:] if log.exists() else ""
            raise RuntimeError(f"Live V12 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}: break
        time.sleep(.5)
    else:
        raise RuntimeError(f"Live V12 startup timeout. Last health={last}")
    print("\nLIVE V12 Q1 PROCESS ARMED"); print("  session:", session); print("  pid:", p.pid)
    print(f"  cancel SLO: target {TARGET_CANCEL_SEND_MS:.0f}ms | p95 {P95_CANCEL_SEND_MS:.0f}ms | hard {HARD_CANCEL_SEND_MS:.0f}ms")
    print("  Q10: DISABLED until this Q1 audit passes")
    return live_status(show=False)


def start_live_cycle_q10(*args, **kwargs):
    raise RuntimeError("V12 Q10 is intentionally disabled until a fresh V12 Q1 smoke passes audit_v12_smoke().")


def live_status(*, show=True, tail_lines=20): return B.live_status(show=show, tail_lines=tail_lines)
def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0): return B.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)
def account_safety_check(*, show=True): return V11.account_safety_check(show=show)
def verify_recording_bundle(session_dir, *, show=True, write_result=True):
    return V10.verify_recording_bundle(session_dir, show=show, write_result=write_result)


def _main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-live-session"); ap.add_argument("--config")
    ap.add_argument("--audit-smoke-session"); ap.add_argument("--verify-recording-session"); a = ap.parse_args()
    if a.audit_smoke_session: audit_v12_smoke(a.audit_smoke_session, show=True, write_result=True); return
    if a.verify_recording_session: verify_recording_bundle(a.verify_recording_session, show=True, write_result=True); return
    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}; _run_process_v12(Path(a.run_live_session), cfg)
    else: live_status(show=True)


if __name__ == "__main__": _main()

__all__ = ["LIVE_VERSION", "EXECUTION_PARENT", "RECORDING_PARENT", "FRESHNESS_ARCH_VERSION",
           "PriorityFreshnessEngine", "audit_v12_smoke", "account_safety_check",
           "verify_recording_bundle", "live_preflight", "start_live_smoke_q1_one_window",
           "start_live_cycle_q10", "live_status", "kill_and_flatten_live"]
