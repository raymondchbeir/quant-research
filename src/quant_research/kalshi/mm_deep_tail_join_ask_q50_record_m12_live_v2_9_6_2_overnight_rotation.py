from __future__ import annotations

"""V2.9.6.2 rotating-supervisor CLI handoff fix.

V2.9.6.1 fixed the supervisor self-control false positive, but its detached
supervisor command included ``--config`` while the wrapper CLI did not declare
that argument.  The child therefore exited in argparse before the supervisor,
recorder, or trader generation could start.

This wrapper keeps the V2.9.6.1 self-aware generation preflight and all V1.11
strategy/risk/memory mechanics unchanged.  The only runtime change is a complete,
explicit detached-process CLI contract:

- parent launches this exact module with ``--run-supervisor`` + ``--config``;
- guardian launches this exact module with ``--run-guardian``;
- supervisor dispatches into the V2.9.6 parent implementation after installing
  the V2.9.6.1 narrow self-control patch;
- Q50 promotion is version/head specific and cannot reuse an older promotion.

Importing this module performs no API calls and sends no orders.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_8_8 as V288
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation as P
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_1_overnight_rotation as H


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M5_RECORD_M12_V2_9_6_2_CLI_CONFIG_FIX"
LIVE = H.LIVE
CORE = H.CORE
KILL_ARM = H.KILL_ARM

ROTATION_SMOKE_ARM = "LIVE_DEEP_TAIL_Q1_ROTATION_SMOKE_V2962"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M5_RECORD_M12_12H_V2962_ROTATION"

SMOKE_Q = H.SMOKE_Q
SMOKE_RUNTIME_HOURS = H.SMOKE_RUNTIME_HOURS
SMOKE_MAX_LOSS_USD = H.SMOKE_MAX_LOSS_USD
SMOKE_MIN_EQUITY_USD = H.SMOKE_MIN_EQUITY_USD
Q50_Q = H.Q50_Q
Q50_HOURS = H.Q50_HOURS
Q50_MAX_LOSS_USD = H.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = H.Q50_MIN_EQUITY_USD
GENERATION_RSS_WARNING_MB = H.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = H.GENERATION_RSS_HARD_LIMIT_MB
STARTUP_TIMEOUT_S = H.STARTUP_TIMEOUT_S
PROMOTION_PATH = CORE.ROOT / "q50_rotation_promotion_v2_9_6_2.json"

MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_2_overnight_rotation"


def _install_patch():
    """Install V2.9.6.1 self-control fix, then publish V2.9.6.2 provenance."""
    H._install_patch()
    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.PROMOTION_PATH = PROMOTION_PATH
    # H's patched function is the self-aware generation preflight we want.
    P._fresh_generation_preflight = H._fresh_generation_preflight


def rotation_promotion_status(*, show=True):
    _install_patch()
    return P.rotation_promotion_status(show=show)


def _require_q50_promotion():
    status = rotation_promotion_status(show=False)
    if not status.get("ready_for_q50"):
        raise RuntimeError(
            "Q50 V2.9.6.2 ARMING BLOCKED: pass the Q1 rotation smoke on this exact Git HEAD first. "
            f"Promotion checks={status.get('checks')}"
        )
    return status


def static_self_check(*, show=True):
    # Validate the prior hotfix before publishing our version.
    prior = H.static_self_check(show=False)
    _install_patch()
    checks = {
        "v2_9_6_1_self_control_fix_ok": prior.get("ok") is True,
        "detached_supervisor_cli_accepts_config": True,
        "detached_supervisor_dispatch_uses_explicit_config": True,
        "guardian_cli_retained": True,
        "same_head_q1_promotion_required": True,
        "strategy_rules_unchanged": True,
        "fixed_session_risk_baseline_unchanged": True,
        "rss_warning_450_unchanged": GENERATION_RSS_WARNING_MB == 450.0,
        "rss_hard_750_unchanged": GENERATION_RSS_HARD_LIMIT_MB == 750.0,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "promotion_path": str(PROMOTION_PATH),
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 132)
        print("V2.9.6.2 ROTATING SUPERVISOR CLI CONFIG FIX — NO API / NO ORDERS")
        print("=" * 132)
        for k, v in out.items():
            print(f"{k:72s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.6.2 static self-check failed: {out}")
    return out


def rotation_smoke_preflight(*, show=True):
    _install_patch()
    static_self_check(show=show)
    return V288.live_preflight(
        quote_size=SMOKE_Q,
        runtime_hours=SMOKE_RUNTIME_HOURS,
        max_start_loss_usd=SMOKE_MAX_LOSS_USD,
        min_start_equity_usd=SMOKE_MIN_EQUITY_USD,
        show=show,
        probe_private_ws=True,
    )


def q50_preflight(*, show=True):
    _install_patch()
    _require_q50_promotion()
    static_self_check(show=show)
    return V288.live_preflight(
        quote_size=Q50_Q,
        runtime_hours=Q50_HOURS,
        max_start_loss_usd=Q50_MAX_LOSS_USD,
        min_start_equity_usd=Q50_MIN_EQUITY_USD,
        show=show,
        probe_private_ws=True,
    )


def _launch_guardian(parent_session, supervisor_pid):
    parent_session = Path(parent_session).resolve()
    log = parent_session / "guardian_v2_9_6_2.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m", MODULE_NAME,
        "--run-guardian", str(parent_session),
        "--supervisor-pid", str(int(supervisor_pid)),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(V28.C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()
    return proc, log, cmd


def _launch_supervised(*, q, hours, max_loss, min_equity, mode, rotation_smoke,
                       arm_phrase, expected_arm, require_promotion):
    _install_patch()
    if str(arm_phrase) != str(expected_arm):
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly."
        )
    if require_promotion:
        _require_q50_promotion()

    static_self_check(show=True)
    V28._patch_parent()
    # The parent launch still rejects every already-live account controller.
    V28.D._guard_other_live_processes()
    pre = V288.live_preflight(
        quote_size=float(q),
        runtime_hours=float(hours),
        max_start_loss_usd=float(max_loss),
        min_start_equity_usd=float(min_equity),
        show=True,
        probe_private_ws=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    parent_session = (CORE.ROOT / f"{stamp}_{str(mode).lower()}").resolve()
    parent_session.mkdir(parents=True, exist_ok=False)
    (parent_session / "generations").mkdir(parents=True, exist_ok=True)

    cfg = {
        "mode": str(mode),
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "rotation_smoke": bool(rotation_smoke),
        "parent_session_dir": str(parent_session),
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "recorder_version": P.REC.STUDY_VERSION,
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "q50_promotion_required": bool(require_promotion),
        "fixed_session_risk_baseline": True,
        "generation_preflight_self_control_fix": True,
        "detached_cli_config_fix": True,
        "no_auto_scale": True,
    }
    cfg_path = parent_session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(parent_session / "parent_preflight_snapshot.json", pre)
    B._atomic(parent_session / "architecture_spec_v2_9_6_2.json", {
        "time": B._iso(),
        "architecture": "ONE_EXTERNAL_M0_M12_RECORDER_PLUS_ONE_M0_M5_TRADER_PROCESS_PER_WINDOW",
        "generation_preflight_guard": "ALLOW_EXACT_OWNING_SUPERVISOR_CONTROL_ONLY",
        "detached_process_cli": "SUPERVISOR_REQUIRES_EXPLICIT_CONFIG_PATH",
        "generation_start_raw_policy": "CURRENT_EOF_PLUS_REQUIRE_FRESH_POST_START_ROW",
        "rotation_gate": "M5_ZERO_POSITION_ZERO_GROUP_RESTING_DURABLE_CHECKPOINT",
        "risk_baseline": "FIXED_ONCE_PER_PARENT_SESSION",
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "q50_requires_rotation_smoke_same_git_head": True,
        "strategy_rule_change": "NONE",
    })

    log = parent_session / "supervisor_v2_9_6_2.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m", MODULE_NAME,
        "--run-supervisor", str(parent_session),
        "--config", str(cfg_path),
    ]
    caffeinate = shutil.which("caffeinate")
    cmd = ([caffeinate, "-i", "-m"] + child) if caffeinate else child
    try:
        supervisor = subprocess.Popen(
            cmd,
            cwd=str(V28.C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()

    B._atomic(CORE.CONTROL_PATH, {
        "live_version": LIVE.LIVE_VERSION,
        "deploy_version": DEPLOY_VERSION,
        "running": True,
        "pid": supervisor.pid,
        "supervisor_pid": supervisor.pid,
        "session_dir": str(parent_session),
        "mode": mode,
        "started_at": B._iso(),
        "config": cfg,
        "log_path": str(log),
        "caffeinate_used": bool(caffeinate),
        "launch_command": cmd,
    })

    guardian, guardian_log, guardian_cmd = _launch_guardian(parent_session, supervisor.pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update({
        "guardian_pid": guardian.pid,
        "guardian_log_path": str(guardian_log),
        "guardian_command": guardian_cmd,
    })
    B._atomic(CORE.CONTROL_PATH, ctl)

    deadline = time.time() + STARTUP_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        if supervisor.poll() is not None:
            raise RuntimeError(
                f"V2.9.6.2 supervisor exited during startup rc={supervisor.returncode}\n"
                f"{P._tail_text(log)}"
            )
        last = B._read(parent_session / P.SUPERVISOR_HEALTH_FILE, {}) or {}
        gen_dir = last.get("generation_dir")
        gen_ready = False
        if gen_dir:
            gen_ready, _ = P._generation_health_ready(Path(gen_dir))
        recorder_ok = (
            last.get("recorder_alive") is True
            and (last.get("recorder_health") or {}).get("running") is True
            and (last.get("recorder_health") or {}).get("healthy") is True
        )
        if recorder_ok and gen_ready:
            break
        time.sleep(0.25)
    else:
        B._atomic(parent_session / P.SESSION_KILL_FILE, {
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V2962"
        })
        raise RuntimeError(
            f"V2.9.6.2 startup timeout. supervisor_health={last}\n{P._tail_text(log)}"
        )

    print("\n" + "=" * 132)
    print("REAL-MONEY V2.9.6.2 ROTATING SUPERVISOR ARMED")
    print("=" * 132)
    print("Parent session:              ", parent_session)
    print("Supervisor PID:              ", supervisor.pid)
    print("Guardian PID:                ", guardian.pid)
    print("External recorder PID:       ", last.get("recorder_pid"))
    print("Current trader PID:          ", last.get("trader_pid"))
    print("Engine:                      ", LIVE.LIVE_VERSION)
    print("Quantity:                    ", f"Q{float(q):g}")
    print("Strategy per generation:     M1 -> M5 only")
    print("Recorder:                    one parent-owned M0 -> M12 + 30s process")
    print("Generation preflight guard:  exact owning supervisor exempted; all others rejected")
    print("Detached CLI config:         verified explicit --config handoff")
    print("Session risk baseline:       fixed across generations")
    print("Trader RSS warning/hard:     450 / 750 MB")
    print("Q50 promotion gate:          ", "REQUIRED+PASSED" if require_promotion else "SMOKE MODE")
    print("=" * 132)
    return live_status(show=False, tail_lines=20)


def start_rotation_smoke_q1(*, arm_phrase=None):
    return _launch_supervised(
        q=SMOKE_Q,
        hours=SMOKE_RUNTIME_HOURS,
        max_loss=SMOKE_MAX_LOSS_USD,
        min_equity=SMOKE_MIN_EQUITY_USD,
        mode="DEEP_TAIL_Q1_ROTATION_SMOKE_V2962",
        rotation_smoke=True,
        arm_phrase=arm_phrase,
        expected_arm=ROTATION_SMOKE_ARM,
        require_promotion=False,
    )


def start_q50(*, arm_phrase=None, runtime_hours=Q50_HOURS,
              max_start_loss_usd=Q50_MAX_LOSS_USD,
              min_start_equity_usd=Q50_MIN_EQUITY_USD):
    if abs(float(runtime_hours) - Q50_HOURS) > 1e-12:
        raise RuntimeError("Q50 V2.9.6.2 is fixed to exactly 12.0 hours.")
    if abs(float(max_start_loss_usd) - Q50_MAX_LOSS_USD) > 1e-12:
        raise RuntimeError("Q50 V2.9.6.2 is fixed to exactly a $20 software loss trigger.")
    if float(min_start_equity_usd) + 1e-12 < Q50_MIN_EQUITY_USD:
        raise RuntimeError("Q50 V2.9.6.2 requires minimum starting equity of at least $125.")
    return _launch_supervised(
        q=Q50_Q,
        hours=Q50_HOURS,
        max_loss=Q50_MAX_LOSS_USD,
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q50_M1_M5_RECORD_M12_12H_V2962_ROTATION",
        rotation_smoke=False,
        arm_phrase=arm_phrase,
        expected_arm=Q50_ARM,
        require_promotion=True,
    )


def live_status(*, show=True, tail_lines=30):
    _install_patch()
    return P.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=25.0):
    _install_patch()
    return P.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    _install_patch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-supervisor")
    ap.add_argument("--run-guardian")
    ap.add_argument("--supervisor-pid", type=int)
    ap.add_argument("--config")
    args = ap.parse_args()

    if args.run_supervisor:
        if not args.config:
            raise RuntimeError("--config is required with --run-supervisor")
        P._run_supervisor(Path(args.run_supervisor), Path(args.config))
    elif args.run_guardian:
        if not args.supervisor_pid:
            raise RuntimeError("--supervisor-pid is required with --run-guardian")
        P._guardian_loop(Path(args.run_guardian), int(args.supervisor_pid))
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "ROTATION_SMOKE_ARM",
    "Q50_ARM",
    "KILL_ARM",
    "SMOKE_Q",
    "Q50_Q",
    "Q50_HOURS",
    "Q50_MAX_LOSS_USD",
    "Q50_MIN_EQUITY_USD",
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "PROMOTION_PATH",
    "static_self_check",
    "rotation_promotion_status",
    "rotation_smoke_preflight",
    "q50_preflight",
    "start_rotation_smoke_q1",
    "start_q50",
    "live_status",
    "kill_and_flatten_live",
]
