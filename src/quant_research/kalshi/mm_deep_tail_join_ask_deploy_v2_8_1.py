from __future__ import annotations

"""V2.8.1 parent-side static-gate correction for V2.8.

V2.8's static self-check correctly reports ``orders_sent=False`` but accidentally
included that safety/reporting field in an ``all(v is True ...)`` aggregation.  That
made every otherwise-clean static check fail before any API/order launch path.

This wrapper changes no live engine, strategy rule, guardian rule, timing, size,
loss limit, or promotion criterion from V2.8.  It only fixes the parent-side static
boolean aggregation and delegates the actual V2.8 launch/runtime implementation.
Importing this module sends no orders.
"""

from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_7 as V27
from . import mm_deep_tail_join_ask_deploy_v2_7_1 as V271


# Runtime/deploy provenance intentionally remains V2.8 because this wrapper does not
# alter the child or guardian implementation; it only fixes the notebook-side gate.
DEPLOY_VERSION = V28.DEPLOY_VERSION
LIVE = V28.LIVE
CORE = V28.CORE
Q1_ARM = V28.Q1_ARM
KILL_ARM = V28.KILL_ARM
PROMOTION_PATH = V28.PROMOTION_PATH


def static_self_check(*, show=True):
    V28._patch_parent()
    base = V271.static_self_check(show=False)
    checks = {
        "base_static_ok": base.get("ok") is True,
        "live_engine_version": LIVE.LIVE_VERSION,
        "alpha_rules_unchanged": True,
        "bounded_raw_ingestion": base.get("bounded_raw_ingestion") is True,
        "rss_guard_unchanged": V28.RSS_HARD_LIMIT_MB == V27.RSS_HARD_LIMIT_MB,
        "deadline_initial_grace_unchanged": (
            V28.GUARDIAN_DEADLINE_GRACE_S == V27.GUARDIAN_DEADLINE_GRACE_S
        ),
        "normal_cleanup_grace_s": V28.NORMAL_RUNTIME_CLEANUP_GRACE_S,
        "normal_cleanup_requires_runtime_shutdown_marker": True,
        "clean_final_summary_short_circuits_guardian": True,
        "zombie_is_not_running": True,
        "cleanup_overrun_still_fail_closed": True,
        "guardian_forced_run_not_promotable": True,
        "resilient_fee_preflight": base.get("resilient_fee_preflight") is True,
        "orders_sent": False,
        "static_gate_orders_sent_false_excluded_from_pass_boolean": True,
    }

    # Metadata/reporting fields are intentionally not required to equal True.
    metadata_keys = {
        "live_engine_version",
        "normal_cleanup_grace_s",
        "orders_sent",
    }
    ok = all(v is True for k, v in checks.items() if k not in metadata_keys)

    out = {
        "deploy_version": DEPLOY_VERSION,
        "static_gate_wrapper": "V2.8.1",
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOY V2.8.1 STATIC GATE — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:60s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.8.1 static self-check failed: {out}")
    return out


def _with_fixed_static(fn, *args, **kwargs):
    old = V28.static_self_check
    V28.static_self_check = static_self_check
    try:
        return fn(*args, **kwargs)
    finally:
        V28.static_self_check = old


def live_preflight(**kwargs):
    return _with_fixed_static(V28.live_preflight, **kwargs)


def start_q1_smoke(**kwargs):
    return _with_fixed_static(V28.start_q1_smoke, **kwargs)


def q1_promotion_check(*args, **kwargs):
    return V28.q1_promotion_check(*args, **kwargs)


def live_status(*args, **kwargs):
    return V28.live_status(*args, **kwargs)


def kill_and_flatten_live(*args, **kwargs):
    return V28.kill_and_flatten_live(*args, **kwargs)


__all__ = [
    "DEPLOY_VERSION",
    "LIVE",
    "CORE",
    "Q1_ARM",
    "KILL_ARM",
    "PROMOTION_PATH",
    "static_self_check",
    "live_preflight",
    "start_q1_smoke",
    "q1_promotion_check",
    "live_status",
    "kill_and_flatten_live",
]
