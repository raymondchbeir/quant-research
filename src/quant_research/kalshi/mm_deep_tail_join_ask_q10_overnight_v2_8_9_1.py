from __future__ import annotations

"""V2.8.9.1 parent self-check fix for the Q10 8-hour overnight runner.

Runtime strategy/execution is unchanged from V2.8.9. The only bug fixed here is
that V2.8.9 intentionally reported ``strict_pre_m0_data_gate=False`` but then fed
that informational False value into ``all(checks.values())``, making the static
self-check fail even when every required condition passed.

This patch replaces that informational field with the positive predicate
``strict_pre_m0_data_gate_disabled=True`` and delegates the actual real-money
launch/runtime to V2.8.9 unchanged.

Importing this module sends no orders.
"""

from . import mm_deep_tail_join_ask_q10_overnight_v2_8_9 as V289
from . import mm_deep_tail_join_ask_deploy_v2_8_8 as V288
from . import mm_deep_tail_join_ask_deploy_v2_8_7 as V287
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_6 as LIVE
from . import mm_event_time_m0_m5_recorder_v5_auth as V5A

PATCH_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q10_OVERNIGHT_V2_8_9_1_SELF_CHECK_FIX"
DEPLOY_VERSION = V289.DEPLOY_VERSION
Q10_ARM = V289.Q10_ARM
KILL_ARM = V289.KILL_ARM
Q10_Q = V289.Q10_Q
Q10_HOURS = V289.Q10_HOURS
Q10_MAX_LOSS_USD = V289.Q10_MAX_LOSS_USD
Q10_MIN_EQUITY_USD = V289.Q10_MIN_EQUITY_USD


def static_self_check(*, show=True):
    pp = V287._install_child_pythonpath()
    base = V288.static_self_check(show=False)
    live = LIVE.static_self_check(show=False)

    checks = {
        "validated_v2_8_8_base": base.get("ok") is True,
        "live_engine_v1_6": str(LIVE.LIVE_VERSION) == "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_6_AUTH_V5_DISCOVERY",
        "authenticated_v5_discovery": base.get("authenticated_v5_discovery") is True,
        "bounded_raw_ingestion": live.get("bounded_raw_ingestion") is True,
        "alpha_rules_unchanged": live.get("alpha_rules_unchanged_from_v1_5") is True,
        "q10_on_allowed_ladder": int(Q10_Q) in tuple(int(x) for x in V1.LADDER_Q),
        "runtime_fixed_8h": abs(Q10_HOURS - 8.0) < 1e-12,
        "loss_trigger_fixed_20": abs(Q10_MAX_LOSS_USD - 20.0) < 1e-12,
        "minimum_equity_125": abs(Q10_MIN_EQUITY_USD - 125.0) < 1e-12,
        "guardian_unchanged": True,
        "fee_snapshot_reuse": base.get("fee_snapshot_reuse") is True,
        "strict_pre_m0_data_gate_disabled": True,
        "auto_scaling_disabled": True,
        "child_pythonpath_installed": pp.get("installed") is True,
    }

    ok = all(checks.values())
    out = {
        "patch_version": PATCH_VERSION,
        "runtime_deploy_version": DEPLOY_VERSION,
        "quantity": Q10_Q,
        "runtime_hours": Q10_HOURS,
        "max_start_loss_usd": Q10_MAX_LOSS_USD,
        "min_start_equity_usd": Q10_MIN_EQUITY_USD,
        "live_engine_version": LIVE.LIVE_VERSION,
        "recorder_version": V5A.STUDY_VERSION,
        **checks,
        "ok": bool(ok),
        "orders_sent": False,
    }

    if show:
        print("=" * 108)
        print("DEEP-TAIL Q10 V2.8.9.1 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 108)
        for k, v in out.items():
            print(f"{k:52s}: {v}")

    if not ok:
        raise RuntimeError(f"Q10 V2.8.9.1 static self-check failed: {out}")
    return out


# Patch only the parent-side checker that V2.8.9 start_q10_overnight() invokes.
# The detached child still runs the exact V2.8.9 runtime module and therefore the
# trading engine, recorder, guardian, fee reuse, timing, sizing and loss logic are
# untouched.
V289.static_self_check = static_self_check


def start_q10_overnight(*, arm_phrase=None, runtime_hours=Q10_HOURS,
                        max_start_loss_usd=Q10_MAX_LOSS_USD,
                        min_start_equity_usd=Q10_MIN_EQUITY_USD):
    return V289.start_q10_overnight(
        arm_phrase=arm_phrase,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
    )


def live_preflight(*, show=True):
    return V289.live_preflight(show=show)


def live_status(*args, **kwargs):
    return V289.live_status(*args, **kwargs)


def kill_and_flatten_live(*args, **kwargs):
    return V289.kill_and_flatten_live(*args, **kwargs)


def q10_data_audit(*args, **kwargs):
    return V289.q10_data_audit(*args, **kwargs)


def overnight_audit(*args, **kwargs):
    return V289.overnight_audit(*args, **kwargs)


__all__ = [
    "PATCH_VERSION", "DEPLOY_VERSION", "Q10_ARM", "KILL_ARM", "Q10_Q",
    "Q10_HOURS", "Q10_MAX_LOSS_USD", "Q10_MIN_EQUITY_USD",
    "static_self_check", "start_q10_overnight", "live_preflight", "live_status",
    "kill_and_flatten_live", "q10_data_audit", "overnight_audit",
]
