from __future__ import annotations

"""V1.1 safety/latency hotfixes for the deep-tail live engine.

Three execution-plumbing fixes only; strategy mechanics are unchanged.

1. Stable private WebSocket lifecycle
   A quiet account no longer forces a reconnect every few seconds. The authenticated
   ``fill`` + ``user_orders`` subscriptions stay alive; actual socket failure still
   clears READY and invokes the entry-order safety path.

2. Session-bounded REST fill reconciliation
   The REST fallback uses ``min_ts`` from engine startup and deduplicates returned
   fill identities, so old account history cannot pollute the live event queue.

3. Wall-clock M1 scheduler
   Entry dispatch no longer depends on receiving a new public-book row in the exact
   two-second M1 arming band. The already-running 2ms engine loop checks CURRENT wall
   time and the V12.2 exact-EOF raw watcher; once M1 arrives it uses the latest
   certified BBO and submits the fixed 5c pair. If the infrastructure cannot certify
   a state before M1+2s, that market is skipped rather than entered late.

No alpha parameter, quantity, M1/M5 boundary, order price, JOIN_ASK rule, loss
threshold, recorder, or fill-accounting rule changes.
"""

import asyncio
import json
import threading
import time

import numpy as np

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1

LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_1_WALL_M1_STABLE_WS_BOUNDED_REST"


class StablePrivateUserStream(V1.PrivateUserStream):
    async def _run(self):
        key_id, private_key = C.load_auth()
        while not self.stop_event.is_set():
            self.ready.clear()
            subscribed = set()
            had_ready = False
            self.last_error = None
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
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            # Quiet account != broken socket. websockets' own
                            # ping/pong settings remain transport-liveness authority.
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
                            if (
                                {"fill", "user_orders"}.issubset(subscribed)
                                and not self.ready.is_set()
                            ):
                                self.ready.set()
                                had_ready = True
                                self.out_queue.put({
                                    "kind": "WS_STATE",
                                    "state": "READY",
                                    "recv_ms": recv_ms,
                                })
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
                self.last_error = repr(exc)
            finally:
                self.ready.clear()
                self.reconnects += 1
                if had_ready or self.last_error:
                    self.out_queue.put({
                        "kind": "WS_STATE",
                        "state": "DOWN",
                        "error": self.last_error,
                        "recv_ms": V1._wall_ms(),
                    })
                    self.wake_event.set()
            if not self.stop_event.is_set():
                await asyncio.sleep(0.25)


class BoundedRestFillReconciler(V1.RestFillReconciler):
    """REST fill fallback scoped to this engine lifetime and deduplicated."""

    def __init__(self, out_queue, wake_event):
        self.out_queue = out_queue
        self.wake_event = wake_event
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="dt-rest-fill-reconcile-bounded",
            daemon=True,
        )
        # Five seconds of pre-start overlap protects fills that race thread startup.
        self.min_ts = max(0, int(time.time()) - 5)
        self.seen = set()

    @staticmethod
    def _key(r):
        return (
            str((r or {}).get("trade_id") or ""),
            str((r or {}).get("order_id") or ""),
            str((r or {}).get("ts_ms", (r or {}).get("ts")) or ""),
            str((r or {}).get("count_fp", (r or {}).get("count")) or ""),
        )

    def _run(self):
        client = B.Q1.LiveClient()
        try:
            client.get("/portfolio/balance")
        except Exception:
            pass

        while not self.stop_event.wait(V1.REST_FILL_RECONCILE_S):
            try:
                body, timing = client.get(
                    "/portfolio/fills",
                    params={
                        "min_ts": int(self.min_ts),
                        "limit": 1000,
                        "subaccount": 0,
                    },
                )
                pushed = 0
                for r in (body or {}).get("fills") or []:
                    k = self._key(r)
                    if k in self.seen:
                        continue
                    self.seen.add(k)
                    self.out_queue.put({
                        "kind": "REST_FILL",
                        "msg": r,
                        "recv_ms": V1._wall_ms(),
                        "timing": timing,
                    })
                    pushed += 1
                if pushed:
                    self.wake_event.set()
            except Exception as exc:
                self.out_queue.put({
                    "kind": "REST_FILL_ERROR",
                    "error": repr(exc),
                    "recv_ms": V1._wall_ms(),
                })
                self.wake_event.set()


class WallClockM1DeepTailEngine(V1.DeepTailLiveEngine):
    """V1 engine with M1 scheduling independent of book-update arrival timing."""

    def enforce_wall_clock_m1(self):
        if self.shutdown_started:
            return
        tickers = set(self.eligible) | set(self.meta)
        for ticker in sorted(tickers):
            if ticker in self.finalized or not self.eligible.get(ticker, False):
                continue
            if self.trade_deadline is not None and time.time() >= self.trade_deadline:
                return

            st = self.dt[ticker]
            if st.get("entry_attempted"):
                continue
            e = self.wall_elapsed(ticker)
            if not np.isfinite(e):
                continue
            if e < V1.M1_S:
                continue
            if e > V1.M1_S + V1.ENTRY_ARM_LATE_TOLERANCE_S:
                st["entry_attempted"] = True
                st["phase"] = "DISABLED"
                st["disabled_reason"] = "MISSED_M1_ARM_WINDOW"
                self._transition(
                    ticker,
                    "WINDOW_DISABLED",
                    reason=st["disabled_reason"],
                    elapsed_s=e,
                    scheduler="WALL_CLOCK_V1_1",
                )
                continue

            # During the 2s arming band, wait rather than immediately exclude the
            # market if a safety feed is reconnecting. The hard M1+2s bound above
            # ensures we never compensate by entering late.
            if not self.private.ready.is_set() or not self.fast.ready():
                continue

            fresh_cur, cert = self._latest_fresh_bbo(ticker)
            if fresh_cur is None:
                continue

            self._lat(
                "M1_WALL_CLOCK_ENTRY_DISPATCH",
                ticker=ticker,
                wall_elapsed_s=e,
                cert=cert,
            )
            self._submit_entry_pair(ticker, fresh_cur, e)

    def enforce_wall_clock_m5(self):
        # V1's event loop already calls this every pass. Put the M1 scheduler on
        # that same starvation-proof wall-clock path before M5 enforcement.
        self.enforce_wall_clock_m1()
        return super().enforce_wall_clock_m5()


def _install_patch():
    V1.PrivateUserStream = StablePrivateUserStream
    V1.RestFillReconciler = BoundedRestFillReconciler
    V1.DeepTailLiveEngine = WallClockM1DeepTailEngine


def run_live_process(session, cfg):
    _install_patch()
    # Make provenance/health state unambiguous while keeping V1 mechanics.
    old = V1.LIVE_VERSION
    try:
        V1.LIVE_VERSION = LIVE_VERSION
        return V1.run_live_process(session, cfg)
    finally:
        V1.LIVE_VERSION = old


def static_self_check(*, show=True):
    _install_patch()
    out = V1.static_self_check(show=False)
    out = dict(out)
    out.update({
        "version": LIVE_VERSION,
        "stable_private_ws_quiet_timeout_reconnect": False,
        "private_ws_idle_wakeup_s": 1.0,
        "rest_fill_reconciler_bounded_by_min_ts": True,
        "rest_fill_reconciler_deduplicated": True,
        "m1_scheduler": "CURRENT_WALL_CLOCK_2MS_LOOP",
        "m1_late_tolerance_s": V1.ENTRY_ARM_LATE_TOLERANCE_S,
    })
    if show:
        print("=" * 92)
        print("DEEP-TAIL LIVE V1.1 STATIC SELF-CHECK — NO ORDERS")
        print("=" * 92)
        for k, v in out.items():
            print(f"{k:48s}: {v}")
    return out


__all__ = [
    "LIVE_VERSION",
    "StablePrivateUserStream",
    "BoundedRestFillReconciler",
    "WallClockM1DeepTailEngine",
    "run_live_process",
    "static_self_check",
]
