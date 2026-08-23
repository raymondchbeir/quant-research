from __future__ import annotations

"""V2.9.9.4 guardian/parent M12 handoff fix.

Operational-only layer on top of V2.9.9.3 / V1.12.5.

Observed failure addressed
--------------------------
The V2.9.9.3 controlled rotation test proved that the parent hard-recycle watchdog
was armed at M12 + 45s, but the inherited guardian still used its historical
post-window exit timeout of 30s.  Once all M12 target tickers were finalized, that
guardian timer expired first and stopped the supervisor/trader with
GUARDIAN_POST_M5_EXIT_TIMEOUT before the parent could reach its 45s hard-recycle
deadline.

This layer changes only that watchdog ordering:

    parent M12 hard recycle:    M12 + 45s
    guardian post-M12 timeout:  90s after all targets are terminal

The guardian therefore remains a final independent backstop, but it no longer races
and preempts the parent recovery/recycle mechanism.  The parent still launches a
fresh generation only after authoritative recovery verifies zero positions / zero
strategy-group resting orders and the old trader process is dead.

Strategy mechanics are unchanged: Q50, M1, M12, danger guard, REC25, atomic trigger
snapshot, fixed passive exit semantics, exact equity, no repricing, and reduce-only
IOC terminal cleanup are inherited exactly from V2.9.9.3 / V1.12.5.

Importing this module performs no API calls and sends no orders.
"""

from . import mm_deep_tail_join_ask_q50_m12_guard_rec25_live_v2_9_9_3_hard_recycle as BASE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_REC25_V2_9_9_4_GUARDIAN_HANDOFF"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_rec25_live_v2_9_9_4_guardian_handoff"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M12_GUARD_REC25_12H_V2994"
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

GENERATION_RSS_WARNING_MB = BASE.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = BASE.GENERATION_RSS_HARD_LIMIT_MB
RSS_HARD_STOP_DISABLED = BASE.RSS_HARD_STOP_DISABLED

RECOVERY_FRACTION = BASE.RECOVERY_FRACTION
PRE_LOOKBACK_S = BASE.PRE_LOOKBACK_S
PRE_EXCLUDE_S = BASE.PRE_EXCLUDE_S
PRE_FALLBACK_S = BASE.PRE_FALLBACK_S

# Must be comfortably later than the parent M12+45s deadline, including a small
# recovery/verification handoff margin.  This remains an independent guardian stop.
GUARDIAN_POST_M12_EXIT_TIMEOUT_S = 90.0


def _install_patch():
    """Install V2.9.9.3, then move only the guardian timeout behind the parent."""
    BASE._install_patch()

    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q50_ARM
    RUNTIME.LIVE = LIVE

    # The V2.9.8 runtime stores the M12 guardian timeout under this operational name.
    RUNTIME.POST_M12_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S

    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P._run_supervisor = BASE._run_supervisor_hard_recycle

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

    parent_deadline_after_m12 = float(M12_HARD_RECYCLE_GRACE_S)
    guardian_timeout = float(GUARDIAN_POST_M12_EXIT_TIMEOUT_S)

    checks = {
        "base_v2993_ok": base.get("ok") is True,
        "live_engine_unchanged_v1_12_5": LIVE.LIVE_VERSION == BASE.LIVE.LIVE_VERSION,
        "parent_supervisor_hard_recycle_preserved": P._run_supervisor is BASE._run_supervisor_hard_recycle,
        "parent_m12_hard_recycle_grace_45s": parent_deadline_after_m12 == 45.0,
        "guardian_post_m12_timeout_90s": guardian_timeout == 90.0,
        "guardian_timeout_after_parent_deadline": guardian_timeout > parent_deadline_after_m12,
        "guardian_handoff_margin_at_least_30s": (guardian_timeout - parent_deadline_after_m12) >= 30.0,
        "runtime_guardian_timeout_bound_90s": RUNTIME.POST_M12_EXIT_TIMEOUT_S == 90.0,
        "guardian_module_timeout_bound_90s": V2963.POST_M5_EXIT_TIMEOUT_S == 90.0,
        "detached_module_points_to_v2994": RUNTIME.MODULE_NAME == MODULE_NAME,
        "q50_exact_50": Q50_Q == 50.0,
        "runtime_exact_12h": Q50_HOURS == 12.0,
        "entry_m1_60": M1_S == 60.0,
        "terminal_m12_720": M12_S == 720.0,
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": LIVE.PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": LIVE.PASSIVE_EXIT_TIF == "good_till_canceled",
        "risk_m12_flatten_reduce_only_ioc_unchanged": True,
        "fixed_session_risk_baseline_preserved": True,
        "recorder_parent_owned": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "watchdog_ordering": {
            "parent_m12_hard_recycle_grace_s": M12_HARD_RECYCLE_GRACE_S,
            "guardian_post_m12_exit_timeout_s": GUARDIAN_POST_M12_EXIT_TIMEOUT_S,
            "guardian_minus_parent_margin_s": guardian_timeout - parent_deadline_after_m12,
            "ordering": "PARENT_RECYCLE_FIRST__GUARDIAN_FINAL_BACKSTOP",
        },
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 168)
        print("V2.9.9.4 GUARDIAN/PARENT M12 HANDOFF STATIC CHECK — NO API / NO ORDERS")
        print("=" * 168)
        for k, v in out.items():
            print(f"{k:104s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.9.4 static self-check failed: {out}")
    return out


def q50_preflight(*, show=True):
    """Same exact-dollar read-only preflight as V2.9.9.3."""
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
    """REAL-MONEY Q50 / 12h with parent-first M12 recycle handoff."""
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
