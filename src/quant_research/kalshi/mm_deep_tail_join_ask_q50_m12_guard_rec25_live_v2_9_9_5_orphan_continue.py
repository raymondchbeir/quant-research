from __future__ import annotations

"""V2.9.9.5 verified-orphan generation continuation.

Operational-only layer on top of V2.9.9.4 / V1.12.5.

Observed failure addressed
--------------------------
A generation can fail closed with CONFIRMED_ORPHAN_RESTING_ORDER when local order
state says terminal while an authoritative account read still finds the strategy
order resting.  The child correctly triggers its order group and shuts down.  The
parent then performs authoritative fail-closed recovery and can prove:

- zero strategy-group resting orders,
- zero nonzero positions,
- zero resting orders anywhere in the account, and
- the old trader process is dead.

V2.9.9.4 still terminated the entire 12-hour parent after that fully verified
recovery.  This layer keeps the child fail-closed behavior unchanged, but allows the
parent to launch a fresh generation after this one exact abnormal reason has been
independently recovered and verified safe.

Every other abnormal generation exit retains the original fail-closed parent stop.
Manual stop, session deadline, recorder failure, loss-floor breach, unverified
recovery, or a live old trader still terminate the parent.

Strategy mechanics are unchanged: Q50, M1, M12, danger guard, REC25, atomic trigger
snapshot, fixed passive exit semantics, exact equity, no repricing, parent M12+45s
hard recycle, guardian 90s backstop, and reduce-only IOC terminal cleanup are
inherited exactly from V2.9.9.4 / V1.12.5.

Importing this module performs no API calls and sends no orders.
"""

import inspect

from . import mm_deep_tail_join_ask_q50_m12_guard_rec25_live_v2_9_9_4_guardian_handoff as BASE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_REC25_V2_9_9_5_ORPHAN_CONTINUE"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_rec25_live_v2_9_9_5_orphan_continue"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M12_GUARD_REC25_12H_V2995"
KILL_ARM = BASE.KILL_ARM

RUNTIME = BASE.RUNTIME
P = BASE.P
H = BASE.H
V2963 = BASE.V2963
V28 = BASE.V28
V288 = BASE.V288
V111 = BASE.V111
LIVE = BASE.LIVE

Q50_Q = BASE.Q50_Q
Q50_HOURS = BASE.Q50_HOURS
Q50_MAX_LOSS_USD = BASE.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = BASE.Q50_MIN_EQUITY_USD
M1_S = BASE.M1_S
M12_S = BASE.M12_S
LABEL_TAIL_END_S = BASE.LABEL_TAIL_END_S

M12_HARD_RECYCLE_GRACE_S = BASE.M12_HARD_RECYCLE_GRACE_S
HARD_RECYCLE_RECEIPT_FILE = BASE.HARD_RECYCLE_RECEIPT_FILE
GUARDIAN_POST_M12_EXIT_TIMEOUT_S = BASE.GUARDIAN_POST_M12_EXIT_TIMEOUT_S

GENERATION_RSS_WARNING_MB = BASE.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = BASE.GENERATION_RSS_HARD_LIMIT_MB
RSS_HARD_STOP_DISABLED = BASE.RSS_HARD_STOP_DISABLED

RECOVERY_FRACTION = BASE.RECOVERY_FRACTION
PRE_LOOKBACK_S = BASE.PRE_LOOKBACK_S
PRE_EXCLUDE_S = BASE.PRE_EXCLUDE_S
PRE_FALLBACK_S = BASE.PRE_FALLBACK_S

# V2.9.9.4's BASE is the V2.9.9.3 module that owns the parent supervisor.
_PARENT_BASE = BASE.BASE

_ORIGINAL_SUPERVISOR_SOURCE = inspect.getsource(_PARENT_BASE._run_supervisor_hard_recycle)

_OLD_ABNORMAL_BLOCK = '''            if not (\n                rc == 0\n                and checkpoint.get("safe_to_rotate") is True\n                and P._is_clean_final(final, allowed_reasons={"GENERATION_ROTATION_M5_VERIFIED"})\n            ):\n                P._recover_generation_fail_closed(\n                    parent_session, current_dir, current_proc.pid,\n                    reason=f"ABNORMAL_GENERATION_EXIT:rc={rc}:final={final.get('shutdown_reason')}:checkpoint={checkpoint.get('reason')}"\n                )\n                final_reason = "ABNORMAL_GENERATION_FAIL_CLOSED_RECOVERED"\n                break\n'''

_NEW_ABNORMAL_BLOCK = '''            if not (\n                rc == 0\n                and checkpoint.get("safe_to_rotate") is True\n                and P._is_clean_final(final, allowed_reasons={"GENERATION_ROTATION_M5_VERIFIED"})\n            ):\n                abnormal_reason = (\n                    f"ABNORMAL_GENERATION_EXIT:"\n                    f"rc={rc}:"\n                    f"final={final.get('shutdown_reason')}:"\n                    f"checkpoint={checkpoint.get('reason')}"\n                )\n                recovery = P._recover_generation_fail_closed(\n                    parent_session, current_dir, current_proc.pid,\n                    reason=abnormal_reason,\n                )\n\n                # V2.9.9.5: continue ONLY after the one known orphan-order\n                # contradiction has been independently recovered and the account\n                # is authoritatively clean.  The child still fails closed first.\n                trader_stop = recovery.get("trader_stop") or {}\n                recovered_orphan_can_continue = (\n                    rc == 0\n                    and str(final.get("shutdown_reason") or "")\n                    == "CONFIRMED_ORPHAN_RESTING_ORDER"\n                    and recovery.get("recovery_verified") is True\n                    and trader_stop.get("dead") is True\n                    and not (recovery.get("group_resting") or [])\n                    and not (recovery.get("nonzero_positions") or [])\n                    and int(recovery.get("all_account_resting_count") or 0) == 0\n                    and not parent_kill\n                    and not deadline_reached\n                )\n\n                if recovered_orphan_can_continue:\n                    B._append(\n                        parent_session / P.SUPERVISOR_EVENTS_FILE,\n                        {\n                            "time": B._iso(),\n                            "event": "RECOVERED_ORPHAN_GENERATION_CONTINUE",\n                            "generation_id": int(generation_id),\n                            "generation_dir": str(current_dir),\n                            "trader_pid": int(current_proc.pid),\n                            "child_returncode": int(rc),\n                            "child_shutdown_reason": str(final.get("shutdown_reason") or ""),\n                            "recovery_verified": True,\n                            "old_trader_dead": True,\n                            "group_resting_count": len(recovery.get("group_resting") or []),\n                            "nonzero_position_count": len(recovery.get("nonzero_positions") or []),\n                            "all_account_resting_count": int(\n                                recovery.get("all_account_resting_count") or 0\n                            ),\n                            "action": "CONTINUE_PARENT_WITH_FRESH_GENERATION",\n                        },\n                    )\n                    # The next outer-loop iteration performs the normal fresh\n                    # generation preflight against the same fixed session floor.\n                    current_proc = None\n                    current_dir = None\n                    continue\n\n                final_reason = "ABNORMAL_GENERATION_FAIL_CLOSED_RECOVERED"\n                break\n'''

if _ORIGINAL_SUPERVISOR_SOURCE.count(_OLD_ABNORMAL_BLOCK) != 1:
    raise RuntimeError(
        "V2.9.9.5 refused to load: audited V2.9.9.3 abnormal-exit block changed; "
        "expected exactly one patch target."
    )

_PATCHED_SUPERVISOR_SOURCE = _ORIGINAL_SUPERVISOR_SOURCE.replace(
    _OLD_ABNORMAL_BLOCK,
    _NEW_ABNORMAL_BLOCK,
    1,
)

# Compile the audited supervisor source with only the narrow block replacement.
# Its globals are the V2.9.9.3 operational namespace, with this deployment's
# version identity overlaid so supervisor output identifies V2.9.9.5.
_SUPERVISOR_NS = dict(_PARENT_BASE.__dict__)
_SUPERVISOR_NS.update(
    {
        "DEPLOY_VERSION": DEPLOY_VERSION,
        "MODULE_NAME": MODULE_NAME,
        "Q50_ARM": Q50_ARM,
    }
)
exec(
    compile(
        _PATCHED_SUPERVISOR_SOURCE,
        str(getattr(_PARENT_BASE, "__file__", MODULE_NAME)),
        "exec",
    ),
    _SUPERVISOR_NS,
    _SUPERVISOR_NS,
)
_run_supervisor_orphan_continue = _SUPERVISOR_NS["_run_supervisor_hard_recycle"]


def _install_patch():
    """Install V2.9.9.4, then swap only the parent supervisor abnormal-exit policy."""
    BASE._install_patch()

    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q50_ARM
    RUNTIME.LIVE = LIVE

    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P._run_supervisor = _run_supervisor_orphan_continue

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.POST_M5_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state

    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline structural/regression audit. No API calls and no orders."""
    base = BASE.static_self_check(show=False)
    _install_patch()

    checks = {
        "base_v2994_ok": base.get("ok") is True,
        "live_engine_unchanged_v1_12_5": LIVE.LIVE_VERSION == BASE.LIVE.LIVE_VERSION,
        "parent_supervisor_is_orphan_continue": P._run_supervisor is _run_supervisor_orphan_continue,
        "audited_patch_target_exactly_once": _ORIGINAL_SUPERVISOR_SOURCE.count(_OLD_ABNORMAL_BLOCK) == 1,
        "orphan_reason_exact_only": '== "CONFIRMED_ORPHAN_RESTING_ORDER"' in _PATCHED_SUPERVISOR_SOURCE,
        "requires_recovery_verified": 'recovery.get("recovery_verified") is True' in _PATCHED_SUPERVISOR_SOURCE,
        "requires_old_trader_dead": 'trader_stop.get("dead") is True' in _PATCHED_SUPERVISOR_SOURCE,
        "requires_zero_group_resting": 'not (recovery.get("group_resting") or [])' in _PATCHED_SUPERVISOR_SOURCE,
        "requires_zero_positions": 'not (recovery.get("nonzero_positions") or [])' in _PATCHED_SUPERVISOR_SOURCE,
        "requires_zero_account_resting": 'all_account_resting_count' in _PATCHED_SUPERVISOR_SOURCE,
        "manual_stop_blocks_continue": 'and not parent_kill' in _PATCHED_SUPERVISOR_SOURCE,
        "deadline_blocks_continue": 'and not deadline_reached' in _PATCHED_SUPERVISOR_SOURCE,
        "fresh_generation_preflight_preserved": 'P._fresh_generation_preflight' in _PATCHED_SUPERVISOR_SOURCE,
        "fixed_session_risk_baseline_preserved": 'baseline_reset_between_generations\": False' in _PATCHED_SUPERVISOR_SOURCE,
        "m12_hard_recycle_preserved": M12_HARD_RECYCLE_GRACE_S == 45.0,
        "guardian_90s_preserved": GUARDIAN_POST_M12_EXIT_TIMEOUT_S == 90.0,
        "q50_exact_50": Q50_Q == 50.0,
        "runtime_exact_12h": Q50_HOURS == 12.0,
        "entry_m1_60": M1_S == 60.0,
        "terminal_m12_720": M12_S == 720.0,
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": LIVE.PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": LIVE.PASSIVE_EXIT_TIF == "good_till_canceled",
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "orphan_recovery_policy": {
            "child_behavior": "FAIL_CLOSED_UNCHANGED",
            "eligible_reason": "CONFIRMED_ORPHAN_RESTING_ORDER",
            "continue_gate": "RECOVERY_VERIFIED__OLD_TRADER_DEAD__ZERO_GROUP_RESTING__ZERO_POSITIONS__ZERO_ACCOUNT_RESTING",
            "next_action": "FRESH_GENERATION_PREFLIGHT_AND_CONTINUE_PARENT",
            "all_other_abnormal_exits": "STOP_PARENT_AFTER_FAIL_CLOSED_RECOVERY",
        },
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 168)
        print("V2.9.9.5 VERIFIED-ORPHAN CONTINUE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 168)
        for k, v in out.items():
            print(f"{k:104s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.9.5 static self-check failed: {out}")
    return out


def q50_preflight(*, show=True):
    """Same exact-dollar read-only preflight as V2.9.9.4."""
    _install_patch()
    static_self_check(show=show)
    V28._patch_parent()
    V28.D._guard_other_live_processes()

    old_equity = LIVE.V1.B._equity
    LIVE.V1.B._equity = LIVE.exact_equity_from_balance
    try:
        return V288.live_preflight(
            quote_size=Q50_Q,
            runtime_hours=Q50_HOURS,
            max_start_loss_usd=Q50_MAX_LOSS_USD,
            min_start_equity_usd=Q50_MIN_EQUITY_USD,
            show=show,
            probe_private_ws=True,
        )
    finally:
        LIVE.V1.B._equity = old_equity


def start_q50_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q50 / 12h with verified orphan-generation continuation."""
    _install_patch()
    return RUNTIME.start_q50_12h_smoke(arm_phrase=arm_phrase)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    return RUNTIME.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    return RUNTIME.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    _install_patch()
    return RUNTIME._main()


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "MODULE_NAME",
    "Q50_ARM",
    "KILL_ARM",
    "Q50_Q",
    "Q50_HOURS",
    "Q50_MAX_LOSS_USD",
    "Q50_MIN_EQUITY_USD",
    "M1_S",
    "M12_S",
    "LABEL_TAIL_END_S",
    "M12_HARD_RECYCLE_GRACE_S",
    "HARD_RECYCLE_RECEIPT_FILE",
    "GUARDIAN_POST_M12_EXIT_TIMEOUT_S",
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "RSS_HARD_STOP_DISABLED",
    "RECOVERY_FRACTION",
    "PRE_LOOKBACK_S",
    "PRE_EXCLUDE_S",
    "PRE_FALLBACK_S",
    "static_self_check",
    "q50_preflight",
    "start_q50_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
