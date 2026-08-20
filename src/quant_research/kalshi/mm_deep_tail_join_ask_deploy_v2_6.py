from __future__ import annotations

"""V2.6 parent-side deployment helpers for the V1.4 deep-tail engine.

This is intentionally a thin wrapper around V2.5.  The live child/strategy remains
V1.4 persistent-M5-cleanup.  V2.6 fixes one parent-control failure observed during
a V1.4 Q1 smoke: the legacy kill helper could time out, run account cleanup, and
return while the detached live OS process was still alive.

V2.6 kill semantics:
1. request the engine's normal KILL_AND_FLATTEN path;
2. if it does not exit within the grace period, trigger only this run's order group;
3. terminate the detached live process group (SIGTERM, then SIGKILL if necessary);
4. run the existing authoritative emergency account cleanup;
5. independently verify the process is dead, this strategy group has zero resting
   orders, and the account has zero nonzero positions.

Forced parent termination is recovery, not a promotion pass.  A Q1 smoke only
qualifies when the engine completes its own normal shutdown and the existing V2.5
operational audit passes on the exact current git HEAD.

Importing this module sends no orders.
"""

import os
import signal
import time
from pathlib import Path

from . import mm_deep_tail_join_ask_deploy_v2_5 as V25


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_6_PARENT_KILL_GUARD_Q10_1H"
KILL_ARM = V25.KILL_ARM
Q10_ARM = "LIVE_DEEP_TAIL_Q10_1H_V14"

# Re-export the exact V1.4 live engine/module for notebook checks.
LIVE = V25.LIVE
CORE = V25.CORE
B = V25.B


def _patch_parent():
    return V25._patch_parent()


def _current_head():
    return V25._current_head()


def static_self_check(*, show=True):
    parent = V25.static_self_check(show=False)
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_engine_version": LIVE.LIVE_VERSION,
        "v25_core_ok": parent.get("ok") is True,
        "persistent_m5_cleanup": parent.get("persistent_m5_cleanup") is True,
        "parent_kill_escalates_process_group": True,
        "parent_kill_authoritative_reconcile": True,
        "forced_kill_not_promotion": True,
        "q10_one_hour_requires_q1_receipt": True,
        "orders_sent": False,
    }
    out["ok"] = all(
        bool(v) for k, v in out.items()
        if k not in {"deploy_version", "live_engine_version", "orders_sent"}
    )
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOY V2.6 STATIC CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:48s}: {v}")
    if not out["ok"]:
        raise RuntimeError(f"V2.6 static self-check failed: {out}")
    return out


def api_capacity_preflight(**kwargs):
    return V25.api_capacity_preflight(**kwargs)


def live_preflight(**kwargs):
    _patch_parent()
    static_self_check(show=kwargs.get("show", True))
    return V25.live_preflight(**kwargs)


def start_q1_smoke(**kwargs):
    """REAL ORDERS. Delegates to the unchanged V2.5/V1.4 live child."""
    _patch_parent()
    return V25.start_q1_smoke(**kwargs)


def q1_promotion_check(*args, **kwargs):
    """Read-only existing V1.4 operational audit on the exact current git HEAD."""
    _patch_parent()
    return V25.q1_promotion_check(*args, **kwargs)


def _require_q1_promotion():
    _patch_parent()
    return V25._require_q1_promotion()


def start_q10_one_hour(*, arm_phrase=None, runtime_hours=1.0,
                       max_start_loss_usd=20.0, min_start_equity_usd=75.0):
    """REAL ORDERS. Q10 one-hour forward size test after exact-head V1.4 Q1 gate."""
    if abs(float(runtime_hours) - 1.0) > 1e-12:
        raise RuntimeError("V2.6 Q10 forward stage is fixed to exactly 1.0 hour.")
    _require_q1_promotion()
    return V25._launch(
        q=10.0,
        hours=1.0,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q10_1H_V14",
        arm_phrase=arm_phrase,
        expected_arm=Q10_ARM,
    )


def live_status(**kwargs):
    _patch_parent()
    return V25.live_status(**kwargs)


def _authoritative_state(client, gid):
    positions, pos_timing = B._positions(client)
    resting, rest_timing = B._resting(client)
    nonzero = [
        r for r in positions
        if abs(B._f(r.get("position_fp"), 0.0)) > B.EPS
    ]
    group_resting = [
        r for r in resting
        if str(r.get("order_group_id") or "") == str(gid or "")
    ]
    return {
        "positions": positions,
        "resting": resting,
        "nonzero": nonzero,
        "group_resting": group_resting,
        "position_timing": pos_timing,
        "resting_timing": rest_timing,
    }


def kill_and_flatten_live(*, arm_phrase=None, wait_s=12.0,
                          term_wait_s=6.0, kill_wait_s=3.0):
    """REAL CANCEL/CLEANUP MAY OCCUR. Guaranteed parent-side process-death check.

    Returns a structured receipt.  If ``forced_termination`` is True, that run is an
    operational failure/recovery and must not be promoted even if the account ends
    flat.
    """
    if str(arm_phrase) != str(KILL_ARM):
        raise RuntimeError(f"Emergency stop refused. Pass arm_phrase={KILL_ARM!r} exactly.")

    _patch_parent()
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    if not ctl:
        raise RuntimeError("No deep-tail live session control file.")

    session = Path(str(ctl.get("session_dir") or "")).resolve()
    pid = int(ctl.get("pid") or 0)
    if not session.exists() or pid <= 0:
        raise RuntimeError(f"Invalid live control state: {ctl}")

    group = B._read(session / "order_group.json", {}) or {}
    gid = str(group.get("order_group_id") or "")

    # 1) Ask the engine to execute its normal shutdown path first.
    B._atomic(session / "KILL_REQUEST.json", {
        "time": B._iso(),
        "reason": "MANUAL_KILL_AND_FLATTEN",
        "requested_by": DEPLOY_VERSION,
    })

    deadline = time.time() + float(wait_s)
    while time.time() < deadline:
        if not B._pid_alive(pid):
            final = B._read(session / "final_summary.json", {}) or {}
            receipt = {
                "time": B._iso(),
                "session": str(session),
                "pid": pid,
                "order_group_id": gid,
                "normal_stop": True,
                "forced_termination": False,
                "final_summary_present": bool(final),
                "final_summary": final,
                "orders_sent_possible": True,
            }
            B._atomic(session / "parent_kill_receipt_v2_6.json", receipt)
            print("Live process stopped through normal engine shutdown.")
            return receipt
        time.sleep(0.20)

    # 2) The process is stuck.  Disable only this strategy group before touching
    # the OS process so no new strategy resting order can survive the escalation.
    client = B.Q1.LiveClient()
    group_trigger = B._trigger_group(client, gid) if gid else {
        "ok": False,
        "reason": "missing_order_group_id",
    }

    # 3) Terminate the detached process group created by the launcher.
    term_sent = False
    kill_sent = False
    pgid = None
    if B._pid_alive(pid):
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = None

        if pgid is not None:
            if int(pgid) == int(os.getpgrp()):
                raise RuntimeError(
                    "REFUSING forced stop: live process shares notebook process group."
                )
            try:
                os.killpg(pgid, signal.SIGTERM)
                term_sent = True
            except ProcessLookupError:
                pass

    deadline = time.time() + float(term_wait_s)
    while B._pid_alive(pid) and time.time() < deadline:
        time.sleep(0.15)

    if B._pid_alive(pid):
        if pgid is None:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                pgid = None
        if pgid is not None:
            if int(pgid) == int(os.getpgrp()):
                raise RuntimeError(
                    "REFUSING SIGKILL: live process shares notebook process group."
                )
            try:
                os.killpg(pgid, signal.SIGKILL)
                kill_sent = True
            except ProcessLookupError:
                pass

    deadline = time.time() + float(kill_wait_s)
    while B._pid_alive(pid) and time.time() < deadline:
        time.sleep(0.10)

    if B._pid_alive(pid):
        raise RuntimeError(
            f"FORCED STOP FAILED: live PID {pid} is still alive. Do not start another strategy."
        )

    # 4) Reconcile from the exchange, not cached health.  Existing fallback only
    # cancels this run's group and reduce-only flattens recognized strategy tickers.
    fallback_cleanup = B._fallback_cleanup(ctl)
    time.sleep(0.30)
    state = _authoritative_state(client, gid)

    receipt = {
        "time": B._iso(),
        "session": str(session),
        "pid": pid,
        "process_dead": not B._pid_alive(pid),
        "process_group": pgid,
        "sigterm_sent": term_sent,
        "sigkill_sent": kill_sent,
        "normal_stop": False,
        "forced_termination": True,
        "order_group_id": gid,
        "group_trigger": group_trigger,
        "fallback_cleanup": fallback_cleanup,
        "strategy_group_resting_count": len(state["group_resting"]),
        "all_account_resting_count": len(state["resting"]),
        "nonzero_position_count": len(state["nonzero"]),
        "group_resting": state["group_resting"],
        "nonzero_positions": state["nonzero"],
        "promotion_allowed": False,
        "note": (
            "Parent forced termination was required. Account recovery may be clean, "
            "but this run is an operational failure and must not be promoted."
        ),
        "orders_sent_possible": True,
    }
    B._atomic(session / "parent_kill_receipt_v2_6.json", receipt)

    # Keep the control file truthful even when the child could not write its normal
    # STOPPED/final summary state.
    latest_ctl = B._read(CORE.CONTROL_PATH, {}) or ctl
    if str(latest_ctl.get("session_dir") or "") == str(session):
        latest_ctl.update({
            "running": False,
            "stopped_at": B._iso(),
            "shutdown_reason": "PARENT_FORCED_TERMINATION_AFTER_KILL_TIMEOUT",
            "parent_deploy_version": DEPLOY_VERSION,
        })
        B._atomic(CORE.CONTROL_PATH, latest_ctl)

    if state["group_resting"]:
        raise RuntimeError(
            f"Recovery incomplete: strategy group still resting: {state['group_resting']}"
        )
    if state["nonzero"]:
        raise RuntimeError(
            f"Recovery incomplete: account still has nonzero positions: {state['nonzero']}"
        )

    print("Forced parent recovery completed: process dead, strategy group clean, account flat.")
    print("This run is NOT eligible for promotion.")
    return receipt


__all__ = [
    "DEPLOY_VERSION",
    "KILL_ARM",
    "Q10_ARM",
    "LIVE",
    "CORE",
    "static_self_check",
    "api_capacity_preflight",
    "live_preflight",
    "start_q1_smoke",
    "q1_promotion_check",
    "start_q10_one_hour",
    "live_status",
    "kill_and_flatten_live",
]
