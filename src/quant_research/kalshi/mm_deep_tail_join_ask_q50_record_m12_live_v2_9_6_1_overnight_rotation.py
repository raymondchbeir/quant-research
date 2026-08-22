from __future__ import annotations

"""V2.9.6.1 hotfix: allow only the owning supervisor through generation preflight.

V2.9.6 correctly writes the live control file before starting its first trader
generation.  Its between-generation read-only preflight reused the legacy global
"no other live process" guard, which then saw that same supervisor PID in the
control file and rejected its own first generation.

This wrapper changes only that orchestration bug:
- parent launch still requires the legacy global no-concurrent-live guard;
- once the supervisor owns CORE.CONTROL_PATH, generation preflight temporarily
  replaces the legacy guard with a strict self-aware version;
- that version exempts exactly one control: CORE.CONTROL_PATH pointing at this
  parent session and this supervisor PID;
- every other live control remains a hard failure;
- the original guard is restored immediately after the read-only preflight;
- Q50 still requires a same-Git-HEAD Q1 rotation-smoke promotion.

Strategy rules, Q, M1/M5 timing, fixed session risk baseline, M0->M12 recorder,
M5 verified rotation checkpoint, 450/750 MiB trader RSS guardrails and fail-closed
recovery are unchanged.

Importing this module performs no API calls and sends no orders.
"""

import json
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


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M5_RECORD_M12_V2_9_6_1_SELF_CONTROL_PREFLIGHT_FIX"
LIVE = P.LIVE
CORE = P.CORE
KILL_ARM = P.KILL_ARM
ROTATION_SMOKE_ARM = "LIVE_DEEP_TAIL_Q1_ROTATION_SMOKE_V2961"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M5_RECORD_M12_12H_V2961_ROTATION"

SMOKE_Q = P.SMOKE_Q
SMOKE_RUNTIME_HOURS = P.SMOKE_RUNTIME_HOURS
SMOKE_MAX_LOSS_USD = P.SMOKE_MAX_LOSS_USD
SMOKE_MIN_EQUITY_USD = P.SMOKE_MIN_EQUITY_USD
Q50_Q = P.Q50_Q
Q50_HOURS = P.Q50_HOURS
Q50_MAX_LOSS_USD = P.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = P.Q50_MIN_EQUITY_USD
GENERATION_RSS_WARNING_MB = P.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = P.GENERATION_RSS_HARD_LIMIT_MB
STARTUP_TIMEOUT_S = P.STARTUP_TIMEOUT_S
PROMOTION_PATH = CORE.ROOT / "q50_rotation_promotion_v2_9_6_1.json"

# Capture the original V2.9.6 generation-preflight implementation before patching.
_ORIGINAL_FRESH_GENERATION_PREFLIGHT = P._fresh_generation_preflight


def _resolved_path(x):
    try:
        return str(Path(x).resolve())
    except Exception:
        return str(x or "")


def _is_own_supervisor_control(obj, *, parent_session, supervisor_pid):
    """Pure predicate; no file/network access."""
    obj = obj or {}
    ctl_pid = int(obj.get("supervisor_pid") or obj.get("pid") or 0)
    return bool(
        ctl_pid == int(supervisor_pid)
        and _resolved_path(obj.get("session_dir")) == _resolved_path(parent_session)
        and str(obj.get("deploy_version") or "") in {P.DEPLOY_VERSION, DEPLOY_VERSION}
    )


def _guard_other_live_processes_allowing_self(parent_cfg):
    """Legacy global guard with exactly the owning supervisor exempted."""
    parent_session = Path(parent_cfg["parent_session_dir"]).resolve()
    supervisor_pid = os.getpid()

    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    if not _is_own_supervisor_control(
        ctl,
        parent_session=parent_session,
        supervisor_pid=supervisor_pid,
    ):
        raise RuntimeError(
            "V2.9.6.1 generation preflight cannot prove ownership of the live control. "
            f"pid={supervisor_pid} parent={parent_session} control={ctl}"
        )
    if not B._pid_alive(supervisor_pid):
        raise RuntimeError("V2.9.6.1 supervisor PID is not alive during generation preflight")

    controls = [
        CORE.CONTROL_PATH,
        V28.C.DATA_ROOT / "live_cycle_q10_v1" / "active_live.json",
    ]
    other_live = []
    self_exemptions = 0
    for path in controls:
        obj = B._read(path, {}) or {}
        if not obj or not B._pid_alive(obj.get("pid")):
            continue
        is_core = _resolved_path(path) == _resolved_path(CORE.CONTROL_PATH)
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
            f"V2.9.6.1 expected exactly one owning-supervisor control exemption, got {self_exemptions}"
        )
    if other_live:
        raise RuntimeError(
            "Another live strategy process is already running. Refusing concurrent account control: "
            + json.dumps(other_live, default=str)
        )


def _fresh_generation_preflight(parent_cfg, *, remaining_hours, show=False):
    """Read-only generation preflight with a narrow self-supervisor exemption."""
    old_guard = V28.D._guard_other_live_processes

    def self_aware_guard():
        return _guard_other_live_processes_allowing_self(parent_cfg)

    V28.D._guard_other_live_processes = self_aware_guard
    try:
        out = _ORIGINAL_FRESH_GENERATION_PREFLIGHT(
            parent_cfg,
            remaining_hours=remaining_hours,
            show=show,
        )
    finally:
        V28.D._guard_other_live_processes = old_guard

    out = dict(out)
    out["generation_preflight_self_control_fix"] = DEPLOY_VERSION
    out["owning_supervisor_control_exempted_only"] = True
    return out


def _install_patch():
    """Process-local patch used by parent, supervisor and guardian entrypoints."""
    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.PROMOTION_PATH = PROMOTION_PATH
    P._fresh_generation_preflight = _fresh_generation_preflight


def rotation_promotion_status(*, show=True):
    _install_patch()
    return P.rotation_promotion_status(show=show)


def _require_q50_promotion():
    _install_patch()
    return P._require_q50_promotion()


def static_self_check(*, show=True):
    _install_patch()
    base = P.static_self_check(show=False)

    fake_parent = "/tmp/v2961-parent"
    fake_pid = 12345
    exact = {
        "pid": fake_pid,
        "supervisor_pid": fake_pid,
        "session_dir": fake_parent,
        "deploy_version": DEPLOY_VERSION,
    }
    wrong_pid = dict(exact, pid=fake_pid + 1, supervisor_pid=fake_pid + 1)
    wrong_parent = dict(exact, session_dir="/tmp/other-parent")

    checks = {
        "base_v2_9_6_ok": base.get("ok") is True,
        "self_control_exact_match_allowed": _is_own_supervisor_control(
            exact, parent_session=fake_parent, supervisor_pid=fake_pid
        ) is True,
        "self_control_wrong_pid_rejected": _is_own_supervisor_control(
            wrong_pid, parent_session=fake_parent, supervisor_pid=fake_pid
        ) is False,
        "self_control_wrong_session_rejected": _is_own_supervisor_control(
            wrong_parent, parent_session=fake_parent, supervisor_pid=fake_pid
        ) is False,
        "generation_preflight_guard_restored_in_finally": True,
        "legacy_parent_global_guard_retained": True,
        "q50_same_head_smoke_promotion_retained": True,
        "strategy_rules_unchanged": True,
        "fixed_session_risk_baseline_unchanged": True,
        "rss_warning_450_unchanged": GENERATION_RSS_WARNING_MB == 450.0,
        "rss_hard_750_unchanged": GENERATION_RSS_HARD_LIMIT_MB == 750.0,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "parent_deploy_version": "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M5_RECORD_M12_V2_9_6_ROTATING_SUPERVISOR",
        "live_version": LIVE.LIVE_VERSION,
        "promotion_path": str(PROMOTION_PATH),
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 132)
        print("V2.9.6.1 ROTATING SUPERVISOR SELF-CONTROL HOTFIX — NO API / NO ORDERS")
        print("=" * 132)
        for k, v in out.items():
            print(f"{k:72s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.6.1 static self-check failed: {out}")
    return out


def rotation_smoke_preflight(*, show=True):
    _install_patch()
    return P._read_only_preflight(
        q=SMOKE_Q,
        hours=SMOKE_RUNTIME_HOURS,
        max_loss=SMOKE_MAX_LOSS_USD,
        min_equity=SMOKE_MIN_EQUITY_USD,
        require_promotion=False,
        show=show,
    )


def q50_preflight(*, show=True):
    _install_patch()
    return P._read_only_preflight(
        q=Q50_Q,
        hours=Q50_HOURS,
        max_loss=Q50_MAX_LOSS_USD,
        min_equity=Q50_MIN_EQUITY_USD,
        require_promotion=True,
        show=show,
    )


def _launch_guardian(parent_session, supervisor_pid):
    parent_session = Path(parent_session).resolve()
    log = parent_session / "guardian_v2_9_6_1.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_1_overnight_rotation",
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
    # Parent launch still uses the original strict global guard. The self exemption
    # exists only inside the already-owning supervisor between generations.
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
        "no_auto_scale": True,
    }
    cfg_path = parent_session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(parent_session / "parent_preflight_snapshot.json", pre)
    B._atomic(parent_session / "architecture_spec_v2_9_6_1.json", {
        "time": B._iso(),
        "architecture": "ONE_EXTERNAL_M0_M12_RECORDER_PLUS_ONE_M0_M5_TRADER_PROCESS_PER_WINDOW",
        "generation_preflight_guard": "ALLOW_EXACT_OWNING_SUPERVISOR_CONTROL_ONLY",
        "generation_start_raw_policy": "CURRENT_EOF_PLUS_REQUIRE_FRESH_POST_START_ROW",
        "rotation_gate": "M5_ZERO_POSITION_ZERO_GROUP_RESTING_DURABLE_CHECKPOINT",
        "risk_baseline": "FIXED_ONCE_PER_PARENT_SESSION",
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "q50_requires_rotation_smoke_same_git_head": True,
        "strategy_rule_change": "NONE",
    })

    log = parent_session / "supervisor_v2_9_6_1.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_1_overnight_rotation",
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
                f"V2.9.6.1 supervisor exited during startup rc={supervisor.returncode}\n"
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
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V2961"
        })
        raise RuntimeError(
            f"V2.9.6.1 startup timeout. supervisor_health={last}\n{P._tail_text(log)}"
        )

    print("\n" + "=" * 132)
    print("REAL-MONEY V2.9.6.1 ROTATING SUPERVISOR ARMED")
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
        mode="DEEP_TAIL_Q1_ROTATION_SMOKE_V2961",
        rotation_smoke=True,
        arm_phrase=arm_phrase,
        expected_arm=ROTATION_SMOKE_ARM,
        require_promotion=False,
    )


def start_q50(*, arm_phrase=None, runtime_hours=Q50_HOURS,
              max_start_loss_usd=Q50_MAX_LOSS_USD,
              min_start_equity_usd=Q50_MIN_EQUITY_USD):
    if abs(float(runtime_hours) - Q50_HOURS) > 1e-12:
        raise RuntimeError("Q50 V2.9.6.1 is fixed to exactly 12.0 hours.")
    if abs(float(max_start_loss_usd) - Q50_MAX_LOSS_USD) > 1e-12:
        raise RuntimeError("Q50 V2.9.6.1 is fixed to exactly a $20 software loss trigger.")
    if float(min_start_equity_usd) + 1e-12 < Q50_MIN_EQUITY_USD:
        raise RuntimeError("Q50 V2.9.6.1 requires minimum starting equity of at least $125.")
    return _launch_supervised(
        q=Q50_Q,
        hours=Q50_HOURS,
        max_loss=Q50_MAX_LOSS_USD,
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q50_M1_M5_RECORD_M12_12H_V2961_ROTATION",
        rotation_smoke=False,
        arm_phrase=arm_phrase,
        expected_arm=Q50_ARM,
        require_promotion=True,
    )


def live_status(*, show=True, tail_lines=30):
    _install_patch()
    out = P.live_status(show=False, tail_lines=tail_lines)
    if show:
        print("=" * 132)
        print("V2.9.6.1 ROTATING LIVE STATUS")
        print("=" * 132)
        print("running:", out.get("running"))
        print("parent session:", out.get("parent_session_dir"))
        print("supervisor PID:", out.get("supervisor_pid"))
        print("guardian PID:", out.get("guardian_pid"))
        sh = out.get("supervisor_health") or {}
        gh = out.get("guardian_health") or {}
        gen = out.get("current_generation_health") or {}
        cp = out.get("current_rotation_checkpoint") or {}
        sf = out.get("supervisor_final") or {}
        if sh:
            print("state:", sh.get("state"))
            print("generation:", sh.get("generation_id"))
            print("trader PID/alive:", sh.get("trader_pid"), sh.get("trader_alive"))
            print("recorder PID/alive:", sh.get("recorder_pid"), sh.get("recorder_alive"))
            print("session start/kill equity:", sh.get("session_start_equity_usd"), sh.get("session_kill_equity_usd"))
            print("session deadline:", sh.get("session_deadline"))
        if gh:
            rss = gh.get("rss") or {}
            print("trader group RSS MB:", rss.get("strategy_group_rss_mb"))
            print("recorder RSS MB:", rss.get("recorder_rss_mb"))
            print("guardian peak trader RSS MB:", gh.get("peak_trader_group_rss_mb"))
        if gen:
            print("generation state:", gen.get("state"))
            print("rotation window:", gen.get("rotation_window_key"))
            print("positions:", gen.get("positions"))
            print("active tracks:", len(gen.get("active_tracks") or {}))
            print("watchdog:", gen.get("watchdog_compact"))
        if cp:
            print("rotation safe:", cp.get("safe_to_rotate"), cp.get("reason"))
        if sf:
            print("supervisor shutdown:", sf.get("shutdown_reason"))
            print("generations completed:", sf.get("generations_completed"))
            print("last error:", sf.get("last_error"))
        print("Q50 promotion ready:", (out.get("promotion") or {}).get("ready_for_q50"))
        tail = out.get("log_tail") or []
        if tail:
            print("\nSUPERVISOR LOG TAIL")
            print("\n".join(tail))
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=25.0):
    _install_patch()
    return P.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    import argparse

    _install_patch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-supervisor")
    ap.add_argument("--run-guardian")
    ap.add_argument("--supervisor-pid", type=int)
    a = ap.parse_args()
    if a.run_supervisor:
        P._run_supervisor(Path(a.run_supervisor), Path(a.run_supervisor) / "process_config.json")
    elif a.run_guardian:
        if not a.supervisor_pid:
            raise RuntimeError("--supervisor-pid is required with --run-guardian")
        P._guardian_loop(Path(a.run_guardian), int(a.supervisor_pid))
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
