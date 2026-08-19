from __future__ import annotations

"""V1.1 safety/latency hotfixes for the deep-tail live engine.

Two execution-plumbing fixes only; strategy mechanics are unchanged.

1. Stable private WebSocket lifecycle
   V1's first implementation used a 5-second ``wait_for(ws.recv())`` at the outer
   connection scope. A quiet account could therefore reconnect every five seconds
   and unnecessarily invoke the safety path that cancels entry orders when private
   execution visibility goes down. V1.1 treats a quiet interval as normal and keeps
   the authenticated ``fill`` + ``user_orders`` subscriptions alive.

2. Session-bounded REST fill reconciliation
   V1's fallback GET /portfolio/fills read the latest 1000 account fills every
   500ms. V1.1 sets ``min_ts`` just before engine startup and deduplicates returned
   fill identities, so old account history can never pollute this run's unmatched
   private-event queue and repeated REST rows do not create avoidable CPU work.

No alpha parameter, quantity, M1/M5 timing, order price, exit rule, loss threshold,
recorder, or fill-accounting rule changes.
"""

import asyncio
import json
import threading
import time

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1

LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_1_STABLE_PRIVATE_WS_BOUNDED_REST_FILLS"


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


def _install_patch():
    V1.PrivateUserStream = StablePrivateUserStream
    V1.RestFillReconciler = BoundedRestFillReconciler


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
    "run_live_process",
    "static_self_check",
]
