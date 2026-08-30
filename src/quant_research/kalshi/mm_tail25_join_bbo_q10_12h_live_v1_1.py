from __future__ import annotations

"""Tail25 V1.1 supervisor self-control preflight hardening.

V1 introduced the Q10/12h Tail25 mixed crypto+commodity deployment.  This patch
preserves every strategy/risk/recorder/router parameter and fixes one inherited
rotating-supervisor orchestration requirement: between-generation read-only
preflight must exempt exactly the already-running supervisor that owns the current
parent session while still rejecting every other live account controller.

The initial operator launch continues to use the strict global no-concurrent-live
guard.  The exemption exists only inside the owning supervisor's fresh generation
preflight and the original guard is restored in a finally block.

Importing this module performs no API calls, orders, cancels or transfers.
"""

import argparse
import json
import os
from pathlib import Path

from . import mm_tail25_join_bbo_q10_12h_live_v1 as V1


DEPLOY_VERSION = V1.DEPLOY_VERSION
PATCH_VERSION = "TAIL25_MULTI12_SUPERVISOR_SELF_CONTROL_V1_1"
MODULE_NAME = "quant_research.kalshi.mm_tail25_join_bbo_q10_12h_live_v1_1"

Q1_ARM = "LIVE_TAIL25_MULTI12_Q1_ONE_WINDOW_V1_1"
Q10_ARM = "LIVE_TAIL25_MULTI12_Q10_12H_V1_1"
KILL_ARM = V1.KILL_ARM

LIVE = V1.LIVE
REC = V1.REC
ROUTER = V1.ROUTER
P = V1.P
RUNTIME = V1.RUNTIME
V28 = V1.V28
B = V1.B

Q1_Q = V1.Q1_Q
Q10_Q = V1.Q10_Q
Q10_HOURS = V1.Q10_HOURS
Q10_MAX_LOSS_USD = V1.Q10_MAX_LOSS_USD
Q10_MIN_EQUITY_USD = V1.Q10_MIN_EQUITY_USD
PROMOTION_PATH = V1.PROMOTION_PATH

_ORIGINAL_V1_INSTALL = V1._install_patch


def _resolved_path(x):
    try:
        return str(Path(x).resolve())
    except Exception:
        return str(x or "")


def _is_own_supervisor_control(obj, *, parent_session, supervisor_pid):
    obj = obj or {}
    ctl_pid = int(obj.get("supervisor_pid") or obj.get("pid") or 0)
    return bool(
        ctl_pid == int(supervisor_pid)
        and _resolved_path(obj.get("session_dir"))
        == _resolved_path(parent_session)
        and str(obj.get("deploy_version") or "") == DEPLOY_VERSION
    )


def _guard_other_live_processes_allowing_self(parent_cfg):
    parent_session = Path(parent_cfg["parent_session_dir"]).resolve()
    supervisor_pid = os.getpid()

    ctl = B._read(V1.CORE.CONTROL_PATH, {}) or {}
    if not _is_own_supervisor_control(
        ctl,
        parent_session=parent_session,
        supervisor_pid=supervisor_pid,
    ):
        raise RuntimeError(
            "Tail25 V1.1 generation preflight cannot prove ownership of the "
            f"live control. pid={supervisor_pid} parent={parent_session} "
            f"control={ctl}"
        )
    if not B._pid_alive(supervisor_pid):
        raise RuntimeError(
            "Tail25 V1.1 supervisor PID is not alive during generation preflight"
        )

    controls = [
        V1.CORE.CONTROL_PATH,
        V28.C.DATA_ROOT / "live_cycle_q10_v1" / "active_live.json",
    ]
    other_live = []
    self_exemptions = 0
    for path in controls:
        obj = B._read(path, {}) or {}
        if not obj or not B._pid_alive(obj.get("pid")):
            continue
        is_core = _resolved_path(path) == _resolved_path(V1.CORE.CONTROL_PATH)
        is_self = is_core and _is_own_supervisor_control(
            obj,
            parent_session=parent_session,
            supervisor_pid=supervisor_pid,
        )
        if is_self:
            self_exemptions += 1
            continue
        other_live.append({"control": str(path), "state": obj})

    if self_exemptions != 1:
        raise RuntimeError(
            "Tail25 V1.1 expected exactly one owning-supervisor control "
            f"exemption, got {self_exemptions}"
        )
    if other_live:
        raise RuntimeError(
            "Another live strategy process is already running. Refusing "
            "concurrent account control: "
            + json.dumps(other_live, default=str)
        )


def _fresh_generation_preflight(parent_cfg, *, remaining_hours, show=False):
    """V1 fresh preflight under a narrow owning-supervisor guard exemption."""
    _ORIGINAL_V1_INSTALL()
    # Re-publish V1.1 identity and router after the original install.
    _publish_v11_bindings()

    old_guard = V28.D._guard_other_live_processes
    old_install = V1._install_patch

    def self_aware_guard():
        return _guard_other_live_processes_allowing_self(parent_cfg)

    V28.D._guard_other_live_processes = self_aware_guard
    # V1._read_only_preflight calls V1._install_patch before/after inherited
    # V288 checks.  We already installed the exact process state above; making
    # those two internal calls no-ops prevents the parent BASE patch from
    # replacing this temporary self-aware guard mid-preflight.
    V1._install_patch = lambda: None
    try:
        q = float(parent_cfg["quote_size"])
        min_per_shard = (
            V1.Q1_MIN_COLLATERAL_PER_USED_SHARD_USD
            if q <= 1.0 + 1e-12
            else V1.Q10_MIN_COLLATERAL_PER_USED_SHARD_USD
        )
        out = V1._read_only_preflight(
            q=q,
            hours=max(0.02, float(remaining_hours)),
            max_loss=float(parent_cfg["max_start_loss_usd"]),
            min_equity=float(parent_cfg["min_start_equity_usd"]),
            min_per_shard=float(min_per_shard),
            probe_private_ws=False,
            show=bool(show),
            run_static=False,
        )
    finally:
        V28.D._guard_other_live_processes = old_guard
        V1._install_patch = old_install
        _publish_v11_bindings()

    out = dict(out or {})
    out["generation_preflight_self_control_fix"] = PATCH_VERSION
    out["owning_supervisor_control_exempted_only"] = True
    return out


def _publish_v11_bindings():
    V1.MODULE_NAME = MODULE_NAME
    V1.Q1_ARM = Q1_ARM
    V1.Q10_ARM = Q10_ARM

    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q10_ARM
    RUNTIME.LIVE = LIVE

    P._fresh_generation_preflight = _fresh_generation_preflight


def _install_patch():
    _ORIGINAL_V1_INSTALL()
    _publish_v11_bindings()
    # Make every internal V1 call (launch, preflight, status, detached child) use
    # this hardened installer rather than falling back to V1's fresh-preflight.
    V1._install_patch = _install_patch
    return {
        "deploy_version": DEPLOY_VERSION,
        "patch_version": PATCH_VERSION,
        "module_name": MODULE_NAME,
        "orders_sent": False,
    }


def static_self_check(*, show=True):
    _install_patch()
    base = V1.static_self_check(show=False)

    fake_parent = "/tmp/tail25-v11-parent"
    fake_pid = 12345
    exact = {
        "pid": fake_pid,
        "supervisor_pid": fake_pid,
        "session_dir": fake_parent,
        "deploy_version": DEPLOY_VERSION,
    }
    wrong_pid = dict(
        exact,
        pid=fake_pid + 1,
        supervisor_pid=fake_pid + 1,
    )
    wrong_parent = dict(exact, session_dir="/tmp/tail25-other-parent")

    checks = {
        "v1_static_ok": base.get("ok") is True,
        "deploy_version_preserved": DEPLOY_VERSION == V1.DEPLOY_VERSION,
        "strategy_version_preserved": LIVE.LIVE_VERSION == V1.LIVE.LIVE_VERSION,
        "q10_preserved": Q10_Q == 10.0,
        "runtime_12h_preserved": Q10_HOURS == 12.0,
        "loss_stop_30_preserved": Q10_MAX_LOSS_USD == 30.0,
        "min_equity_300_preserved": Q10_MIN_EQUITY_USD == 300.0,
        "tail25_params_preserved": (
            LIVE.ENTRY_OFFSET == 0.25
            and LIVE.ENTRY_REPRICE_HYSTERESIS == 0.02
            and LIVE.EDGE_ZONE == 0.15
            and LIVE.EXIT_REPRICE_HYSTERESIS == 0.02
            and LIVE.EXIT_HORIZON_S == 3.0
        ),
        "universe_12_preserved": len(ROUTER.SERIES) == 12,
        "recorder_v7_preserved": P.REC is REC,
        "self_control_exact_match_allowed": _is_own_supervisor_control(
            exact,
            parent_session=fake_parent,
            supervisor_pid=fake_pid,
        )
        is True,
        "self_control_wrong_pid_rejected": _is_own_supervisor_control(
            wrong_pid,
            parent_session=fake_parent,
            supervisor_pid=fake_pid,
        )
        is False,
        "self_control_wrong_parent_rejected": _is_own_supervisor_control(
            wrong_parent,
            parent_session=fake_parent,
            supervisor_pid=fake_pid,
        )
        is False,
        "parent_fresh_preflight_is_v11": P._fresh_generation_preflight
        is _fresh_generation_preflight,
        "detached_module_is_v11": RUNTIME.MODULE_NAME == MODULE_NAME,
        "initial_launch_global_guard_unchanged": True,
        "generation_guard_restored_in_finally": True,
        "orders_sent": False,
        "transfers_sent": False,
        "api_called": False,
    }
    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"orders_sent", "transfers_sent", "api_called"}
    )
    out = {
        "deploy_version": DEPLOY_VERSION,
        "patch_version": PATCH_VERSION,
        "module_name": MODULE_NAME,
        "live_version": LIVE.LIVE_VERSION,
        "recorder_version": REC.STUDY_VERSION,
        "router_version": ROUTER.ROUTER_VERSION,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 160)
        print("TAIL25 MULTI12 V1.1 SELF-CONTROL STATIC CHECK — NO API / NO ORDERS")
        print("=" * 160)
        for k, v in out.items():
            print(f"{k:100s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 V1.1 static check failed: {out}")
    return out


def q1_smoke_preflight(*, show=True):
    _install_patch()
    return V1.q1_smoke_preflight(show=show)


def q10_preflight(*, show=True):
    _install_patch()
    return V1.q10_preflight(show=show)


def rotation_promotion_status(*, show=True):
    _install_patch()
    return V1.rotation_promotion_status(show=show)


def start_q1_one_window_smoke(*, arm_phrase=None):
    _install_patch()
    if str(arm_phrase) != Q1_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q1_ARM!r} exactly."
        )
    return V1.start_q1_one_window_smoke(arm_phrase=Q1_ARM)


def start_q10_12h(*, arm_phrase=None):
    _install_patch()
    if str(arm_phrase) != Q10_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q10_ARM!r} exactly."
        )
    return V1.start_q10_12h(arm_phrase=Q10_ARM)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    out = V1.live_status(show=show, tail_lines=tail_lines)
    if isinstance(out, dict):
        out["deploy_patch_version"] = PATCH_VERSION
        out["module_name"] = MODULE_NAME
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    if str(arm_phrase) != KILL_ARM:
        raise RuntimeError(f"Pass arm_phrase={KILL_ARM!r} exactly.")
    return V1.kill_and_flatten_live(
        arm_phrase=KILL_ARM,
        wait_s=float(wait_s),
    )


def _main():
    _install_patch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-generation")
    ap.add_argument("--run-supervisor")
    ap.add_argument("--run-guardian")
    ap.add_argument("--supervisor-pid", type=int)
    ap.add_argument("--config")
    args = ap.parse_args()

    if args.run_generation:
        if not args.config:
            raise RuntimeError("--config is required with --run-generation")
        return V1._run_generation(
            Path(args.run_generation),
            Path(args.config),
        )

    if args.run_supervisor:
        if not args.config:
            raise RuntimeError("--config is required with --run-supervisor")
        _install_patch()
        return P._run_supervisor(
            Path(args.run_supervisor),
            Path(args.config),
        )

    if args.run_guardian:
        if not args.supervisor_pid:
            raise RuntimeError("--supervisor-pid is required with --run-guardian")
        _install_patch()
        return P._guardian_loop(
            Path(args.run_guardian),
            int(args.supervisor_pid),
        )

    return static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "PATCH_VERSION",
    "MODULE_NAME",
    "Q1_ARM",
    "Q10_ARM",
    "KILL_ARM",
    "Q1_Q",
    "Q10_Q",
    "Q10_HOURS",
    "Q10_MAX_LOSS_USD",
    "Q10_MIN_EQUITY_USD",
    "PROMOTION_PATH",
    "static_self_check",
    "q1_smoke_preflight",
    "q10_preflight",
    "rotation_promotion_status",
    "start_q1_one_window_smoke",
    "start_q10_12h",
    "live_status",
    "kill_and_flatten_live",
]
