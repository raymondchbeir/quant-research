from __future__ import annotations

"""V2.2 deployment wrapper: notebook-safe private WebSocket preflight.

Why this exists
---------------
Jupyter/IPython already owns a running asyncio event loop. V2's synchronous
``private_ws_probe`` called ``asyncio.run(...)`` directly, which is correct from a
normal synchronous Python process but raises ``RuntimeError: asyncio.run() cannot
be called from a running event loop`` inside a notebook.

This wrapper changes NO trading strategy, order transport, M1/M5 timing, sizing,
rate-limit gate, risk rule, recorder, or live-process behavior. It changes only the
READ-ONLY preflight adapter: the authenticated fill + user_orders subscription
probe is executed in a short-lived worker thread, where it owns its own event loop.

Real-money launchers remain the V2/V2.1 implementations and retain all arming and
Q1->Q5 promotion gates.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from . import mm_deep_tail_join_ask_deploy_v2 as D
from . import mm_deep_tail_join_ask_deploy_v2_1 as D21

DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_2_NOTEBOOK_SAFE_WS_PREFLIGHT"


def _run_private_probe_in_thread(timeout_s=12.0):
    """Run D._private_probe_async in a dedicated thread/event loop.

    This is intentionally used even when the caller has no active event loop so
    notebook and terminal preflights exercise one identical adapter.
    """
    timeout_s = float(timeout_s)

    def runner():
        return asyncio.run(D._private_probe_async(timeout_s=timeout_s))

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dt-ws-preflight")
    fut = pool.submit(runner)
    try:
        return fut.result(timeout=timeout_s + 8.0)
    except FutureTimeout as exc:
        fut.cancel()
        raise RuntimeError(
            f"Private WebSocket preflight worker timed out after {timeout_s + 8.0:.1f}s"
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def private_ws_probe(*, show=True, timeout_s=12.0):
    """Authenticated READ-ONLY subscription test; notebook-safe; sends no orders."""
    out = _run_private_probe_in_thread(timeout_s=timeout_s)
    out = dict(out)
    out["adapter"] = "DEDICATED_THREAD_EVENT_LOOP"
    out["orders_sent"] = False

    if show:
        print("PRIVATE fill + user_orders WS:", "PASS" if out.get("ok") else "FAIL")
        print("  subscribed:", out.get("subscribed"))
        print("  elapsed:   ", f"{out.get('elapsed_s', 0.0):.3f}s")
        print("  adapter:   ", out["adapter"])
        print("  ORDERS SENT: NO")

    if not out.get("ok"):
        raise RuntimeError(f"Private WebSocket preflight failed: {out}")
    return out


def _install_patch():
    # D._launch() resolves D.live_preflight(), which in turn resolves the global
    # D.private_ws_probe. Patching that one read-only helper makes both notebook
    # preflight and the launcher's mandatory preflight safe without changing any
    # live engine/execution object.
    D.private_ws_probe = private_ws_probe


def api_capacity_preflight(**kwargs):
    return D21.api_capacity_preflight(**kwargs)


def static_self_check(*, show=True):
    _install_patch()
    out = D21.static_self_check(show=show)
    out = dict(out)
    out.update({
        "deploy_wrapper_version": DEPLOY_VERSION,
        "private_ws_preflight_adapter": "DEDICATED_THREAD_EVENT_LOOP",
        "notebook_running_loop_safe": True,
        "orders_sent": False,
    })
    return out


def live_preflight(*, quote_size, runtime_hours, max_start_loss_usd,
                   min_start_equity_usd, show=True, probe_private_ws=True):
    _install_patch()
    cap = D21.api_capacity_preflight(show=show)
    out = D.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
        probe_private_ws=probe_private_ws,
    )
    out = dict(out)
    out["api_capacity_preflight"] = cap
    out["deploy_wrapper_version"] = DEPLOY_VERSION
    return out


def start_q1_smoke(**kwargs):
    _install_patch()
    D21.api_capacity_preflight(show=True)
    return D.start_q1_smoke(**kwargs)


def q1_promotion_check(*args, **kwargs):
    _install_patch()
    return D.q1_promotion_check(*args, **kwargs)


def start_q5_overnight(**kwargs):
    _install_patch()
    D21.api_capacity_preflight(show=True)
    return D.start_q5_overnight(**kwargs)


def start_ladder_stage(**kwargs):
    _install_patch()
    D21.api_capacity_preflight(show=True)
    return D.start_ladder_stage(**kwargs)


def live_status(**kwargs):
    _install_patch()
    return D.live_status(**kwargs)


def kill_and_flatten_live(**kwargs):
    _install_patch()
    return D.kill_and_flatten_live(**kwargs)


Q1_ARM = D.Q1_ARM
Q5_ARM = D.Q5_ARM
KILL_ARM = D.KILL_ARM
PROMOTION_PATH = D.PROMOTION_PATH


__all__ = [
    "DEPLOY_VERSION",
    "Q1_ARM",
    "Q5_ARM",
    "KILL_ARM",
    "PROMOTION_PATH",
    "api_capacity_preflight",
    "static_self_check",
    "private_ws_probe",
    "live_preflight",
    "start_q1_smoke",
    "q1_promotion_check",
    "start_q5_overnight",
    "start_ladder_stage",
    "live_status",
    "kill_and_flatten_live",
]
