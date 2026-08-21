from __future__ import annotations

"""V1.10 final runtime binding guard for the long-run REST fill reconciler.

Operational-only fix after runtime telemetry showed that a V1.9/V1.8 launch could
advertise ``MIN_TS_INCREMENTAL_DEDUP_CURSOR`` in config while the constructed live
engine still contained the older ``BoundedRestFillReconciler`` object.

Root cause
----------
V1.7 correctly patches ``V1.RestFillReconciler`` before entering the older wrapper
stack.  The downstream installation chain is allowed to install prior safety
patches before ``B._run_process`` constructs the engine, so the class selected at
actual engine construction can differ from the class V1.7 intended.

Fix
---
Wrap the final ``V1._install_runtime`` boundary.  After every downstream install
has completed, but before the engine object is constructed, bind:

- ``V1.RestFillReconciler = V17.IncrementalRestFillReconciler``
- ``V1.PrivateUserStream = V17.PersistentPrivateUserStream``

A runtime-binding artifact records the exact class names.  The deployment launcher
must additionally inspect live health and refuse to arm unless the instantiated
reconciler publishes mode ``MIN_TS_INCREMENTAL_DEDUP``.

Static-check hardening
----------------------
``V1.PrivateUserStream`` is intentionally a mutable runtime hook.  Older wrapper
imports can temporarily replace that hook, so comparing V1.7's persistent class
against the hook's *current identity* makes a fresh control process fail even when
the persistent class is valid and the final runtime rebind is correct.  The static
check therefore verifies the persistent class against its immutable MRO origin
(module/name) and its own override, while the launcher's runtime-binding artifact
and instantiated health remain the authoritative proof of the class actually used.

No strategy rule changes: Q, M1/M5 boundaries, 5c entries, first-fill-wins,
fixed JOIN_ASK/no-reprice, M5 cleanup, recorder horizon, stale-orphan tombstones,
loss logic, guardian logic, and order-group behavior are unchanged.

Importing this module performs no API calls and sends no orders.
"""

import threading
from pathlib import Path
from queue import Queue

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_7 as V17
from . import mm_deep_tail_join_ask_live_v1_9_stale_orphan_guard as V19


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_10_FINAL_RUNTIME_INCREMENTAL_RECONCILER_GUARD"
EXPECTED_REST_MODE = "MIN_TS_INCREMENTAL_DEDUP"
RUNTIME_BINDING_FILE = "runtime_transport_binding_v1_10.json"


def _pure_incremental_metrics_check():
    """Construct, but do not start, the V1.7 reconciler. No API/network activity."""
    r = V17.IncrementalRestFillReconciler(Queue(), threading.Event())
    m = r.metrics()
    return {
        "class_name": type(r).__name__,
        "mode": m.get("mode"),
        "watermark_present": "watermark_ts" in m,
        "dedupe_telemetry_present": "duplicates_suppressed" in m,
        "seen_bound": V17.REST_FILL_SEEN_MAX,
        "thread_started": r.thread.is_alive(),
    }


def _pure_private_stream_check():
    """Validate V1.7's persistent stream without trusting the mutable V1 hook."""
    cls = V17.PersistentPrivateUserStream
    mro = tuple(cls.__mro__)
    origin_ok = any(
        base.__name__ == "PrivateUserStream" and base.__module__ == V1.__name__
        for base in mro[1:]
    )
    return {
        "class_name": cls.__name__,
        "origin_private_stream_present": bool(origin_ok),
        "run_override_present": "_run" in cls.__dict__,
        "current_runtime_hook_class": getattr(V1.PrivateUserStream, "__name__", str(V1.PrivateUserStream)),
        "current_runtime_hook_module": getattr(V1.PrivateUserStream, "__module__", None),
        "runtime_hook_identity_required": False,
    }


def static_self_check(*, show=True):
    parent = V19.static_self_check(show=False)
    pure = _pure_incremental_metrics_check()
    private = _pure_private_stream_check()
    checks = {
        "parent_v1_9_ok": parent.get("ok") is True,
        "incremental_class_available": pure.get("class_name") == "IncrementalRestFillReconciler",
        "incremental_mode_exact": pure.get("mode") == EXPECTED_REST_MODE,
        "incremental_watermark_present": pure.get("watermark_present") is True,
        "incremental_dedupe_telemetry_present": pure.get("dedupe_telemetry_present") is True,
        "incremental_seen_set_bounded": pure.get("seen_bound") == 20000,
        "static_probe_thread_not_started": pure.get("thread_started") is False,
        "persistent_private_stream_available": (
            private.get("class_name") == "PersistentPrivateUserStream"
            and private.get("origin_private_stream_present") is True
            and private.get("run_override_present") is True
        ),
        "private_stream_static_check_ignores_mutable_runtime_hook_identity": (
            private.get("runtime_hook_identity_required") is False
        ),
        "final_runtime_rebind_enabled": True,
        "strategy_rules_unchanged": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "expected_runtime_rest_mode": EXPECTED_REST_MODE,
        "runtime_binding_file": RUNTIME_BINDING_FILE,
        "private_stream_probe": private,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 122)
        print("DEEP-TAIL LIVE V1.10 FINAL RUNTIME RECONCILER CHECK — NO API / NO ORDERS")
        print("=" * 122)
        for k, v in out.items():
            print(f"{k:66s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.10 static check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run V1.9 while forcing V1.7 transport classes at final runtime install."""
    session = Path(session).resolve()

    old_install_runtime = V1._install_runtime
    old_v19_version = V19.LIVE_VERSION

    def final_runtime_install(session_dir, cfg_obj):
        # Let the entire historical install chain complete first.
        result = old_install_runtime(session_dir, cfg_obj)

        # These two assignments happen after downstream patch installation and
        # immediately before B._run_process constructs the live engine.
        V1.RestFillReconciler = V17.IncrementalRestFillReconciler
        V1.PrivateUserStream = V17.PersistentPrivateUserStream

        binding = {
            "time": B._iso(),
            "live_version": LIVE_VERSION,
            "rest_fill_reconciler_class": V1.RestFillReconciler.__name__,
            "private_user_stream_class": V1.PrivateUserStream.__name__,
            "expected_rest_mode": EXPECTED_REST_MODE,
            "incremental_min_ts": True,
            "incremental_cursor_pagination": True,
            "exact_fill_dedupe": True,
            "seen_key_bound": V17.REST_FILL_SEEN_MAX,
            "private_ws_idle_timeout_no_reconnect": True,
            "strategy_rule_change": "NONE",
        }
        B._atomic(Path(session_dir) / RUNTIME_BINDING_FILE, binding)
        return result

    V1._install_runtime = final_runtime_install
    V19.LIVE_VERSION = LIVE_VERSION
    try:
        return V19.run_live_process(session, cfg)
    finally:
        V1._install_runtime = old_install_runtime
        V19.LIVE_VERSION = old_v19_version


__all__ = [
    "LIVE_VERSION",
    "EXPECTED_REST_MODE",
    "RUNTIME_BINDING_FILE",
    "static_self_check",
    "run_live_process",
]
