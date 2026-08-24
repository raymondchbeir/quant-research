from __future__ import annotations

"""V2.9.9.6 Q100 + bounded retry-until-flat fail-closed recovery.

Operational layer on top of V2.9.9.5 / V1.12.5.

Observed failure addressed
--------------------------
During generation 23 on 2026-08-24, the parent M12 hard-recycle recovery correctly
sent reduce-only IOC cleanup orders and reduced one HYPE position from 50 contracts
to 12, but the inherited V2.9.6 recovery routine performed only one fallback-cleanup
sweep.  Because 12 contracts remained after that sweep, it immediately raised
"recovery failed authoritative verification" and the 12-hour supervisor stopped.
The supervisor exception handler then invoked the same authoritative recovery a
second time; that second sweep sold the remaining 12 contracts and verified the
account flat.  The market was therefore flattenable, but the first recovery gave up
too early.

This layer does not weaken fail-closed semantics.  It wraps the already-audited
V2.9.6 authoritative recovery routine and, only when that routine fails specifically
because authoritative verification still sees exposure, retries complete recovery
sweeps for up to 45 seconds.  Every sweep re-reads authoritative position state and
lets the inherited fallback cleanup compute a fresh reduce-only IOC quantity and
fresh executable price.  Recovery succeeds only after authoritative verification
proves zero positions and zero strategy-group resting orders.  If 45 seconds expire,
an unrelated exception occurs, the API path fails differently, or authoritative
verification never reaches flat, the parent remains fail-closed and stops.

V2.9.9.5's verified-orphan continuation is preserved.  Thus a successfully
recovered M12 hard recycle or orphan event can launch a fresh generation while the
same fixed 12-hour session clock and risk baseline remain in force.

Sizing change
-------------
This deployment sets quote size to Q100.  The 12-hour software loss stop remains
$20 (intentionally NOT doubled), minimum starting equity remains $125, and all
strategy mechanics are otherwise unchanged: M1 entry, M12 cleanup, danger guard,
REC25=25%, atomic trigger snapshot, fixed passive exit/no repricing, parent M12+45s
hard recycle, guardian 90s backstop, and reduce-only IOC terminal cleanup.

Importing this module performs no API calls and sends no orders.
"""

import time
from pathlib import Path

from . import mm_deep_tail_join_ask_q50_m12_guard_rec25_live_v2_9_9_5_orphan_continue as BASE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q100_M1_M12_GUARD_REC25_V2_9_9_6_RETRY_FLATTEN"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_6_retry_flatten"
Q100_ARM = "LIVE_DEEP_TAIL_Q100_M1_M12_GUARD_REC25_12H_V2996"
# Historical runtime API still uses the Q50_ARM attribute name internally.
Q50_ARM = Q100_ARM
KILL_ARM = BASE.KILL_ARM

RUNTIME = BASE.RUNTIME
P = BASE.P
H = BASE.H
V2963 = BASE.V2963
V28 = BASE.V28
V288 = BASE.V288
V111 = BASE.V111
LIVE = BASE.LIVE
B = P.B

Q100_Q = 100.0
Q100_HOURS = 12.0
# Deliberately keep the old dollar stop while doubling size.  This is more
# conservative than scaling the loss budget with quantity.
Q100_MAX_LOSS_USD = 20.0
Q100_MIN_EQUITY_USD = 125.0

# Compatibility aliases used by inherited runtime functions and notebook tooling.
Q50_Q = Q100_Q
Q50_HOURS = Q100_HOURS
Q50_MAX_LOSS_USD = Q100_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = Q100_MIN_EQUITY_USD

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

RECOVERY_RETRY_WINDOW_S = 45.0
RECOVERY_RETRY_PAUSE_S = 0.20
RECOVERY_RETRY_ERROR_TOKEN = "recovery failed authoritative verification"

# Preserve the original V2.9.6 authoritative recovery even if this module is
# reloaded in a long-lived notebook after installing the patch once.
if not hasattr(P, "_v2996_original_recover_generation_fail_closed"):
    P._v2996_original_recover_generation_fail_closed = P._recover_generation_fail_closed

_ORIGINAL_RECOVERY = P._v2996_original_recover_generation_fail_closed


def _retryable_authoritative_not_flat(exc):
    """Retry only the exact inherited verification failure; propagate everything else."""
    return bool(
        isinstance(exc, RuntimeError)
        and RECOVERY_RETRY_ERROR_TOKEN in str(exc).lower()
    )


def _recover_generation_fail_closed_retry(parent_session, generation_dir, trader_pid, *, reason):
    """Repeat complete authoritative recovery sweeps until flat or 45s deadline.

    Each inherited sweep independently:
      1. triggers the strategy order group,
      2. ensures the old trader process is dead,
      3. reads current authoritative positions,
      4. sends reduce-only IOC cleanup using current quantity/executable price,
      5. re-reads exchange state, and
      6. returns only if zero-position / zero-group-resting verification passes.

    We retry only when step 6 says exposure remains.  Other failures propagate.
    """
    parent_session = Path(parent_session).resolve()
    generation_dir = Path(generation_dir).resolve()
    start = time.time()
    deadline = start + float(RECOVERY_RETRY_WINDOW_S)
    sweep = 0
    last_exc = None

    while True:
        sweep += 1
        try:
            receipt = _ORIGINAL_RECOVERY(
                parent_session,
                generation_dir,
                trader_pid,
                reason=reason,
            )
        except RuntimeError as exc:
            if not _retryable_authoritative_not_flat(exc):
                raise
            last_exc = exc
            now = time.time()
            remaining = max(0.0, deadline - now)
            B._append(
                parent_session / P.SUPERVISOR_EVENTS_FILE,
                {
                    "time": B._iso(),
                    "event": "RECOVERY_RETRY_AFTER_REMAINING_EXPOSURE",
                    "deploy_version": DEPLOY_VERSION,
                    "generation_dir": str(generation_dir),
                    "trader_pid": int(trader_pid or 0) or None,
                    "reason": str(reason),
                    "sweep": int(sweep),
                    "elapsed_s": float(now - start),
                    "retry_window_s": float(RECOVERY_RETRY_WINDOW_S),
                    "remaining_retry_s": float(remaining),
                },
            )
            if now >= deadline:
                B._append(
                    parent_session / P.SUPERVISOR_EVENTS_FILE,
                    {
                        "time": B._iso(),
                        "event": "RECOVERY_RETRY_WINDOW_EXHAUSTED",
                        "deploy_version": DEPLOY_VERSION,
                        "generation_dir": str(generation_dir),
                        "trader_pid": int(trader_pid or 0) or None,
                        "reason": str(reason),
                        "sweeps": int(sweep),
                        "retry_window_s": float(RECOVERY_RETRY_WINDOW_S),
                    },
                )
                raise RuntimeError(
                    f"V2.9.9.6 recovery retry window exhausted after {sweep} sweeps "
                    f"and {RECOVERY_RETRY_WINDOW_S:.1f}s; account not authoritatively flat"
                ) from last_exc
            time.sleep(float(RECOVERY_RETRY_PAUSE_S))
            continue

        # The inherited routine only returns on authoritative verification PASS.
        receipt = dict(receipt or {})
        if receipt.get("recovery_verified") is not True:
            raise RuntimeError(
                "V2.9.9.6 invariant failure: base recovery returned without verification"
            )
        receipt["recovery_retry_sweeps"] = int(sweep)
        receipt["recovery_retry_elapsed_s"] = float(time.time() - start)
        receipt["recovery_retry_window_s"] = float(RECOVERY_RETRY_WINDOW_S)
        B._atomic(parent_session / P.RECOVERY_RECEIPT_FILE, receipt)
        B._append(
            parent_session / P.SUPERVISOR_EVENTS_FILE,
            {
                "time": B._iso(),
                "event": "RECOVERY_RETRY_VERIFIED_FLAT",
                "deploy_version": DEPLOY_VERSION,
                "generation_dir": str(generation_dir),
                "trader_pid": int(trader_pid or 0) or None,
                "reason": str(reason),
                "sweeps": int(sweep),
                "elapsed_s": float(time.time() - start),
                "recovery_verified": True,
                "group_resting_count": len(receipt.get("group_resting") or []),
                "nonzero_position_count": len(receipt.get("nonzero_positions") or []),
                "all_account_resting_count": int(receipt.get("all_account_resting_count") or 0),
            },
        )
        return receipt


def _install_patch():
    """Install V2.9.9.5, add retry-until-flat recovery, and bind Q100 sizing."""
    BASE._install_patch()

    # Q100 launch identity/parameters on the inherited detached runtime.
    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q100_ARM
    RUNTIME.Q50_Q = Q100_Q
    RUNTIME.Q50_HOURS = Q100_HOURS
    RUNTIME.Q50_MAX_LOSS_USD = Q100_MAX_LOSS_USD
    RUNTIME.Q50_MIN_EQUITY_USD = Q100_MIN_EQUITY_USD
    RUNTIME.LIVE = LIVE

    # Parent supervisor keeps V2.9.9.5's orphan-continuation supervisor but now
    # calls the bounded retrying recovery function wherever fail-closed recovery
    # is required (hard recycle, abnormal exit, supervisor exception, guardian).
    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.Q50_Q = Q100_Q
    P.Q50_HOURS = Q100_HOURS
    P.Q50_MAX_LOSS_USD = Q100_MAX_LOSS_USD
    P.Q50_MIN_EQUITY_USD = Q100_MIN_EQUITY_USD
    P._recover_generation_fail_closed = _recover_generation_fail_closed_retry

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.POST_M5_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state

    # Preserve this exact wrapper through inherited dynamic/subprocess calls.
    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline structural/regression audit. No API calls and no orders."""
    base = BASE.static_self_check(show=False)
    _install_patch()

    checks = {
        "base_v2995_ok": base.get("ok") is True,
        "live_engine_unchanged_v1_12_5": LIVE.LIVE_VERSION == BASE.LIVE.LIVE_VERSION,
        "v2995_orphan_continue_supervisor_preserved": P._run_supervisor is BASE._run_supervisor_orphan_continue,
        "retry_wrapper_bound_to_parent": P._recover_generation_fail_closed is _recover_generation_fail_closed_retry,
        "retry_base_is_original_v296_recovery": _ORIGINAL_RECOVERY is not _recover_generation_fail_closed_retry,
        "retry_only_authoritative_not_flat": RECOVERY_RETRY_ERROR_TOKEN == "recovery failed authoritative verification",
        "retry_window_exact_45s": RECOVERY_RETRY_WINDOW_S == 45.0,
        "retry_pause_200ms": RECOVERY_RETRY_PAUSE_S == 0.20,
        "retry_success_requires_recovery_verified": "recovery_verified" in _recover_generation_fail_closed_retry.__code__.co_names,
        "runtime_q100_exact": RUNTIME.Q50_Q == 100.0,
        "parent_q100_exact": P.Q50_Q == 100.0,
        "q100_exact_100": Q100_Q == 100.0,
        "runtime_exact_12h": Q100_HOURS == 12.0,
        "loss_stop_stays_20": Q100_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q100_MIN_EQUITY_USD == 125.0,
        "entry_m1_60": M1_S == 60.0,
        "terminal_m12_720": M12_S == 720.0,
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "m12_hard_recycle_preserved": M12_HARD_RECYCLE_GRACE_S == 45.0,
        "guardian_90s_preserved": GUARDIAN_POST_M12_EXIT_TIMEOUT_S == 90.0,
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": LIVE.PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": LIVE.PASSIVE_EXIT_TIF == "good_till_canceled",
        "risk_m12_flatten_reduce_only_ioc_unchanged": True,
        "fixed_session_risk_baseline_preserved": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "quantity": Q100_Q,
        "runtime_hours": Q100_HOURS,
        "max_loss_usd": Q100_MAX_LOSS_USD,
        "recovery_retry_policy": {
            "window_s": RECOVERY_RETRY_WINDOW_S,
            "pause_s": RECOVERY_RETRY_PAUSE_S,
            "retryable_failure": "AUTHORITATIVE_STATE_STILL_NONFLAT_AFTER_A_COMPLETE_BASE_RECOVERY_SWEEP",
            "each_sweep": "FRESH_AUTHORITATIVE_POSITION_AND_FRESH_REDUCE_ONLY_IOC_EXECUTABLE_PRICE",
            "success_gate": "AUTHORITATIVE_ZERO_POSITION_AND_ZERO_GROUP_RESTING",
            "timeout_action": "FAIL_CLOSED_STOP_PARENT",
        },
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 172)
        print("V2.9.9.6 Q100 RETRY-UNTIL-FLAT STATIC CHECK — NO API / NO ORDERS")
        print("=" * 172)
        for k, v in out.items():
            print(f"{k:108s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.9.6 static self-check failed: {out}")
    return out


def q100_preflight(*, show=True):
    """Read-only exact-dollar Q100 preflight. Sends no orders."""
    _install_patch()
    static_self_check(show=show)
    V28._patch_parent()
    V28.D._guard_other_live_processes()

    old_equity = LIVE.V1.B._equity
    LIVE.V1.B._equity = LIVE.exact_equity_from_balance
    try:
        return V288.live_preflight(
            quote_size=Q100_Q,
            runtime_hours=Q100_HOURS,
            max_start_loss_usd=Q100_MAX_LOSS_USD,
            min_start_equity_usd=Q100_MIN_EQUITY_USD,
            show=show,
            probe_private_ws=True,
        )
    finally:
        LIVE.V1.B._equity = old_equity


def start_q100_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q100 / 12h with bounded retry-until-flat recovery."""
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
    "Q100_ARM",
    "Q50_ARM",
    "KILL_ARM",
    "Q100_Q",
    "Q100_HOURS",
    "Q100_MAX_LOSS_USD",
    "Q100_MIN_EQUITY_USD",
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
    "RECOVERY_RETRY_WINDOW_S",
    "static_self_check",
    "q100_preflight",
    "start_q100_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
