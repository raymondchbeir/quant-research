from __future__ import annotations

"""Launcher-only fix for V12.2 static self-check aggregation.

The V12.2 engine correctly records strategy_changed=False, but its static_self_check
mistakenly included that expected False value inside all(checks.values()), making
an otherwise healthy static check fail. This module changes no execution logic.
It delegates all live/audit behavior to V12.2 and only fixes the launcher gate.
"""

from . import mm_cycle_q10_live_strategy_v12_2 as V122
from . import mm_cycle_q10_live_strategy_v1 as B

LIVE_VERSION = V122.LIVE_VERSION
STAGED_VERSION = V122.STAGED_VERSION + "_STATIC_CHECK_FIX"

# Capture the engine's original function ONCE, before any temporary monkeypatch.
# The wrapper must call this saved object rather than V122.static_self_check,
# otherwise the temporary patch recurses back into this wrapper.
_ENGINE_STATIC_SELF_CHECK = V122.static_self_check


def static_self_check(*, show=True):
    base = _ENGINE_STATIC_SELF_CHECK(show=False)
    checks = dict(base.get("checks") or {})

    # `strategy_changed=False` is a descriptive invariant, not a failed check.
    strategy_changed = checks.pop("strategy_changed", None)
    checks["strategy_unchanged"] = strategy_changed is False

    out = {
        "time": B._iso(),
        "live_version": LIVE_VERSION,
        "checks": checks,
        "pass": all(checks.values()),
        "orders_sent": False,
        "exchange_api_called": False,
        "launcher_fix": "V12_2_STATIC_CHECK_FALSE_INVARIANT_NO_RECURSION",
    }

    if show:
        print("V12.2 STATIC SELF CHECK (launcher-fixed):", "PASS" if out["pass"] else "FAIL")
        for k, v in checks.items():
            print(f"  {k:<48} {v}")
        print("  ORDERS SENT: NO | EXCHANGE API CALLED: NO")

    return out


def _delegate_with_fixed_static(fn, *args, **kwargs):
    original = V122.static_self_check
    V122.static_self_check = static_self_check
    try:
        return fn(*args, **kwargs)
    finally:
        V122.static_self_check = original


def start_live_q1(*, arm_phrase=None,
                  max_start_loss_usd=B.LOSS_LIMIT_USD,
                  min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    return _delegate_with_fixed_static(
        V122.start_live_q1,
        arm_phrase=arm_phrase,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
    )


def start_live_q5_after_q1(*, prior_q1_session, arm_phrase=None,
                           max_start_loss_usd=B.LOSS_LIMIT_USD,
                           min_start_equity_usd=V122.Q5_MIN_EQUITY_USD):
    return _delegate_with_fixed_static(
        V122.start_live_q5_after_q1,
        prior_q1_session=prior_q1_session,
        arm_phrase=arm_phrase,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
    )


def start_live_q10_after_q5(*, prior_q5_session, arm_phrase=None,
                            max_start_loss_usd=B.LOSS_LIMIT_USD,
                            min_start_equity_usd=B.FULL_MIN_EQUITY):
    return _delegate_with_fixed_static(
        V122.start_live_q10_after_q5,
        prior_q5_session=prior_q5_session,
        arm_phrase=arm_phrase,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
    )


def gate_q1_for_q5(session_dir, *, show=True):
    return V122.gate_q1_for_q5(session_dir, show=show)


def gate_q5_for_q10(session_dir, *, show=True):
    return V122.gate_q5_for_q10(session_dir, show=show)


def audit_stage(session_dir, *, show=True, write_result=True):
    return V122.audit_stage(session_dir, show=show, write_result=write_result)


def live_preflight(**kwargs):
    return V122.live_preflight(**kwargs)


def account_safety_check(*, show=True):
    return V122.account_safety_check(show=show)


def live_status(*, show=True, tail_lines=20):
    return V122.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    return V122.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


__all__ = [
    "LIVE_VERSION",
    "STAGED_VERSION",
    "static_self_check",
    "start_live_q1",
    "start_live_q5_after_q1",
    "start_live_q10_after_q5",
    "gate_q1_for_q5",
    "gate_q5_for_q10",
    "audit_stage",
    "live_preflight",
    "account_safety_check",
    "live_status",
    "kill_and_flatten_live",
]
