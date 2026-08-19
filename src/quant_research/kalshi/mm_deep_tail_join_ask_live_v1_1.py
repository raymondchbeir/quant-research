from __future__ import annotations

"""V1.1 hotfix for the deep-tail live engine private WebSocket lifecycle.

V1's first implementation used a 5-second ``wait_for(ws.recv())`` at the outer
connection scope. A quiet account could therefore reconnect every five seconds and
unnecessarily invoke the safety path that cancels entry orders when private execution
visibility goes down. V1.1 treats a quiet interval as normal: it wakes once per second
only to observe the local stop flag, while keeping the authenticated connection and
subscriptions alive. Actual socket failures still clear READY, are recorded, and cause
entry-order cancellation through the V1 engine.

No strategy, order, risk, sizing, or recorder mechanic changes.
"""

import asyncio
import json

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1

LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_1_STABLE_PRIVATE_WS"


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
                            # Quiet account != broken socket. websockets' ping/pong
                            # settings remain the transport-liveness authority.
                            continue

                        recv_ms = V1._wall_ms()
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
                            if {"fill", "user_orders"}.issubset(subscribed) and not self.ready.is_set():
                                self.ready.set()
                                had_ready = True
                                self.out_queue.put({"kind": "WS_STATE", "state": "READY", "recv_ms": recv_ms})
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


def _install_patch():
    V1.PrivateUserStream = StablePrivateUserStream


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
    })
    if show:
        print("=" * 92)
        print("DEEP-TAIL LIVE V1.1 STATIC SELF-CHECK — NO ORDERS")
        print("=" * 92)
        for k, v in out.items():
            print(f"{k:42s}: {v}")
    return out


__all__ = [
    "LIVE_VERSION",
    "StablePrivateUserStream",
    "run_live_process",
    "static_self_check",
]
