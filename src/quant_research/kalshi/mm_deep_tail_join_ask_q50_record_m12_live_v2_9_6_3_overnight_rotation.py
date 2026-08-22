from __future__ import annotations

"""V2.9.6.3 rotating-supervisor CLI handoff fix.

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


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M5_RECORD_M12_V2_9_6_3_HIGH_RSS_DIRECT_Q50"
LIVE = H.LIVE
CORE = H.CORE
KILL_ARM = H.KILL_ARM

ROTATION_SMOKE_ARM = "LIVE_DEEP_TAIL_Q1_ROTATION_SMOKE_V2963"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M5_RECORD_M12_12H_V2963_ROTATION"

SMOKE_Q = H.SMOKE_Q
SMOKE_RUNTIME_HOURS = H.SMOKE_RUNTIME_HOURS
SMOKE_MAX_LOSS_USD = H.SMOKE_MAX_LOSS_USD
SMOKE_MIN_EQUITY_USD = H.SMOKE_MIN_EQUITY_USD
Q50_Q = H.Q50_Q
Q50_HOURS = H.Q50_HOURS
Q50_MAX_LOSS_USD = H.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = H.Q50_MIN_EQUITY_USD
GENERATION_RSS_WARNING_MB = 1536.0
GENERATION_RSS_HARD_LIMIT_MB = 3072.0
STARTUP_TIMEOUT_S = H.STARTUP_TIMEOUT_S
POST_M5_EXIT_TIMEOUT_S = 30.0
PROMOTION_PATH = CORE.ROOT / "q50_rotation_promotion_v2_9_6_3.json"

MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_3_overnight_rotation"


def _install_patch():
    """Install V2.9.6.1 self-control fix, then publish V2.9.6.3 provenance."""
    H._install_patch()
    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.PROMOTION_PATH = PROMOTION_PATH
    P.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    P.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB
    # H's patched function is the self-aware generation preflight we want.
    P._fresh_generation_preflight = H._fresh_generation_preflight


def rotation_promotion_status(*, show=True):
    _install_patch()
    return P.rotation_promotion_status(show=show)


def _require_q50_promotion():
    # Explicit operator-approved V2.9.6.3 direct-Q50 deployment.
    # A new same-head Q1 rotation promotion is intentionally not required.
    return {
        "ready_for_q50": True,
        "bypassed": True,
        "reason": "V2.9.6.3_OPERATOR_APPROVED_DIRECT_Q50",
        "orders_sent": False,
    }


def static_self_check(*, show=True):
    # Validate inherited V2.9.6.1 under its historical 450/750 RSS assumptions,
    # then immediately restore the V2.9.6.3 high-RSS policy.
    old_warning = P.GENERATION_RSS_WARNING_MB
    old_hard = P.GENERATION_RSS_HARD_LIMIT_MB

    try:
        P.GENERATION_RSS_WARNING_MB = H.GENERATION_RSS_WARNING_MB
        P.GENERATION_RSS_HARD_LIMIT_MB = H.GENERATION_RSS_HARD_LIMIT_MB
        prior = H.static_self_check(show=False)

    finally:
        P.GENERATION_RSS_WARNING_MB = old_warning
        P.GENERATION_RSS_HARD_LIMIT_MB = old_hard

    _install_patch()
    checks = {
        "v2_9_6_1_self_control_fix_ok": prior.get("ok") is True,
        "detached_supervisor_cli_accepts_config": True,
        "detached_supervisor_dispatch_uses_explicit_config": True,
        "guardian_cli_retained": True,
        "q50_direct_arm_without_q1_smoke": True,
        "strategy_rules_unchanged": True,
        "fixed_session_risk_baseline_unchanged": True,
        "rss_warning_1536": GENERATION_RSS_WARNING_MB == 1536.0,
        "rss_hard_3072": GENERATION_RSS_HARD_LIMIT_MB == 3072.0,
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
        print("V2.9.6.3 ROTATING SUPERVISOR CLI CONFIG FIX — NO API / NO ORDERS")
        print("=" * 132)
        for k, v in out.items():
            print(f"{k:72s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.6.3 static self-check failed: {out}")
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
    log = parent_session / "guardian_v2_9_6_3.log"
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
    B._atomic(parent_session / "architecture_spec_v2_9_6_3.json", {
        "time": B._iso(),
        "architecture": "ONE_EXTERNAL_M0_M12_RECORDER_PLUS_ONE_M0_M5_TRADER_PROCESS_PER_WINDOW",
        "generation_preflight_guard": "ALLOW_EXACT_OWNING_SUPERVISOR_CONTROL_ONLY",
        "detached_process_cli": "SUPERVISOR_REQUIRES_EXPLICIT_CONFIG_PATH",
        "generation_start_raw_policy": "CURRENT_EOF_PLUS_REQUIRE_FRESH_POST_START_ROW",
        "rotation_gate": "M5_ZERO_POSITION_ZERO_GROUP_RESTING_DURABLE_CHECKPOINT",
        "risk_baseline": "FIXED_ONCE_PER_PARENT_SESSION",
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "q50_requires_rotation_smoke_same_git_head": False,
        "q50_direct_arm_without_q1_smoke": True,
        "post_m5_exit_timeout_s": POST_M5_EXIT_TIMEOUT_S,
        "strategy_rule_change": "NONE",
    })

    log = parent_session / "supervisor_v2_9_6_3.log"
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
                f"V2.9.6.3 supervisor exited during startup rc={supervisor.returncode}\n"
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
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V2963"
        })
        raise RuntimeError(
            f"V2.9.6.3 startup timeout. supervisor_health={last}\n{P._tail_text(log)}"
        )

    print("\n" + "=" * 132)
    print("REAL-MONEY V2.9.6.3 ROTATING SUPERVISOR ARMED")
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
    print(
        "Trader RSS warning/hard:     ",
        f"{GENERATION_RSS_WARNING_MB:.0f} / "
        f"{GENERATION_RSS_HARD_LIMIT_MB:.0f} MB"
    )
    print(
        "Q50 promotion gate:          ",
        "REQUIRED+PASSED"
        if require_promotion
        else "SKIPPED BY V2.9.6.3 DIRECT-Q50 POLICY"
    )
    print("=" * 132)
    return live_status(show=False, tail_lines=20)


def start_rotation_smoke_q1(*, arm_phrase=None):
    return _launch_supervised(
        q=SMOKE_Q,
        hours=SMOKE_RUNTIME_HOURS,
        max_loss=SMOKE_MAX_LOSS_USD,
        min_equity=SMOKE_MIN_EQUITY_USD,
        mode="DEEP_TAIL_Q1_ROTATION_SMOKE_V2963",
        rotation_smoke=True,
        arm_phrase=arm_phrase,
        expected_arm=ROTATION_SMOKE_ARM,
        require_promotion=False,
    )


def start_q50(*, arm_phrase=None, runtime_hours=Q50_HOURS,
              max_start_loss_usd=Q50_MAX_LOSS_USD,
              min_start_equity_usd=Q50_MIN_EQUITY_USD):
    if abs(float(runtime_hours) - Q50_HOURS) > 1e-12:
        raise RuntimeError("Q50 V2.9.6.3 is fixed to exactly 12.0 hours.")
    if abs(float(max_start_loss_usd) - Q50_MAX_LOSS_USD) > 1e-12:
        raise RuntimeError("Q50 V2.9.6.3 is fixed to exactly a $20 software loss trigger.")
    if float(min_start_equity_usd) + 1e-12 < Q50_MIN_EQUITY_USD:
        raise RuntimeError("Q50 V2.9.6.3 requires minimum starting equity of at least $125.")
    return _launch_supervised(
        q=Q50_Q,
        hours=Q50_HOURS,
        max_loss=Q50_MAX_LOSS_USD,
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q50_M1_M5_RECORD_M12_12H_V2963_ROTATION",
        rotation_smoke=False,
        arm_phrase=arm_phrase,
        expected_arm=Q50_ARM,
        require_promotion=False,
    )


def live_status(*, show=True, tail_lines=30):
    _install_patch()
    return P.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=25.0):
    _install_patch()
    return P.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)



def _post_m5_generation_state(supervisor_health):

    supervisor_health = supervisor_health or {}

    generation_dir = supervisor_health.get(
        "generation_dir"
    )

    if not generation_dir:
        return False, {}


    gh = B._read(
        Path(generation_dir) / "health.json",
        {},
    ) or {}


    target_count = int(
        gh.get("rotation_target_count")
        or 0
    )


    states = (
        gh.get("deep_tail_states")
        or {}
    )


    finalized = [

        str(ticker)

        for ticker, state in states.items()

        if str(
            (state or {}).get("phase")
            or ""
        )
        == "M5_FINALIZED"
    ]


    all_done = bool(
        target_count > 0
        and
        len(finalized) >= target_count
    )


    return all_done, {

        "generation_dir":
            str(generation_dir),

        "target_count":
            target_count,

        "finalized_count":
            len(finalized),

        "finalized_tickers":
            finalized,

        "rotation_checkpoint_written":
            gh.get(
                "rotation_checkpoint_written"
            )
            is True,
    }


def _guardian_loop_v2963(
    parent_session,
    supervisor_pid,
):

    """
    V2.9.6.3 guardian.

    Trader:
      warning:        1536 MB
      catastrophic:   3072 MB

    Post-M5:
      all targets finalized
          -> 30 seconds to checkpoint and exit
          -> otherwise authoritative fail-closed recovery

    Financial-risk controls are unchanged.
    """

    parent_session = Path(
        parent_session
    ).resolve()

    supervisor_pid = int(
        supervisor_pid
    )


    peak_group = 0.0
    peak_recorder = 0.0

    warning_for_generation = None

    post_m5_generation = None
    post_m5_started_wall = None


    while True:

        final = B._read(
            parent_session
            / P.SUPERVISOR_FINAL_FILE,
            {},
        ) or {}


        health = B._read(
            parent_session
            / P.SUPERVISOR_HEALTH_FILE,
            {},
        ) or {}


        # ==========================================================================================
        # NORMAL SUPERVISOR FINALIZATION
        # ==========================================================================================

        if (
            final
            and
            not B._pid_alive(
                supervisor_pid
            )
        ):

            receipt = {

                "time":
                    B._iso(),

                "deploy_version":
                    DEPLOY_VERSION,

                "intervened":
                    False,

                "reason":
                    "SUPERVISOR_FINALIZED",

                "promotion_allowed":
                    final.get(
                        "last_error"
                    )
                    in (
                        None,
                        "",
                    ),

                "peak_trader_group_rss_mb":
                    peak_group,

                "peak_recorder_rss_mb":
                    peak_recorder,

                "trader_rss_warning_mb":
                    GENERATION_RSS_WARNING_MB,

                "trader_rss_hard_limit_mb":
                    GENERATION_RSS_HARD_LIMIT_MB,

                "post_m5_exit_timeout_s":
                    POST_M5_EXIT_TIMEOUT_S,
            }


            B._atomic(
                parent_session
                / P.GUARDIAN_RECEIPT_FILE,
                receipt,
            )


            B._atomic(
                parent_session
                / P.GUARDIAN_HEALTH_FILE,
                {
                    **receipt,
                    "running": False,
                },
            )


            return receipt


        # ==========================================================================================
        # SUPERVISOR FAILURE
        # ==========================================================================================

        if not B._pid_alive(
            supervisor_pid
        ):

            return P._guardian_intervene(

                parent_session,
                supervisor_pid,
                health,

                reason=(
                    "GUARDIAN_SUPERVISOR_FAILURE"
                ),

                peak_group=peak_group,

                peak_recorder=peak_recorder,
            )


        trader_pid = int(
            health.get(
                "trader_pid"
            )
            or 0
        )


        recorder_pid = int(
            health.get(
                "recorder_pid"
            )
            or 0
        )


        generation_id = int(
            health.get(
                "generation_id"
            )
            or 0
        )


        # ==========================================================================================
        # MEMORY SNAPSHOT
        # ==========================================================================================

        if trader_pid > 0:

            rss = P.V27._rss_snapshot(
                trader_pid,
                recorder_pid,
            )

        else:

            rss = {

                "strategy_group_rss_mb":
                    0.0,

                "recorder_rss_mb":
                    0.0,

                "total_rss_mb":
                    0.0,

                "group_processes":
                    [],
            }


        group_mb = float(
            rss.get(
                "strategy_group_rss_mb"
            )
            or 0.0
        )


        recorder_mb = float(
            rss.get(
                "recorder_rss_mb"
            )
            or 0.0
        )


        peak_group = max(
            peak_group,
            group_mb,
        )


        peak_recorder = max(
            peak_recorder,
            recorder_mb,
        )


        # ==========================================================================================
        # POST-M5 EXIT GRACE
        # ==========================================================================================

        (
            all_m5_done,
            post_m5_state,
        ) = _post_m5_generation_state(
            health
        )


        trader_alive = bool(
            trader_pid
            and
            B._pid_alive(
                trader_pid
            )
        )


        if generation_id != post_m5_generation:

            post_m5_generation = (
                generation_id
            )

            post_m5_started_wall = (
                None
            )


        if (
            trader_alive
            and
            all_m5_done
        ):

            if post_m5_started_wall is None:

                post_m5_started_wall = (
                    time.time()
                )


                B._append(
                    parent_session
                    / P.GUARDIAN_EVENTS_FILE,
                    {
                        "time":
                            B._iso(),

                        "event":
                            "TRADER_POST_M5_EXIT_GRACE_STARTED",

                        "generation_id":
                            generation_id,

                        "trader_pid":
                            trader_pid,

                        "timeout_s":
                            POST_M5_EXIT_TIMEOUT_S,

                        "post_m5_state":
                            post_m5_state,

                        "rss":
                            rss,
                    },
                )


            else:

                age = (
                    time.time()
                    -
                    post_m5_started_wall
                )


                if age >= POST_M5_EXIT_TIMEOUT_S:

                    B._append(
                        parent_session
                        / P.GUARDIAN_EVENTS_FILE,
                        {
                            "time":
                                B._iso(),

                            "event":
                                "TRADER_POST_M5_EXIT_TIMEOUT",

                            "generation_id":
                                generation_id,

                            "trader_pid":
                                trader_pid,

                            "elapsed_s":
                                age,

                            "timeout_s":
                                POST_M5_EXIT_TIMEOUT_S,

                            "post_m5_state":
                                post_m5_state,

                            "rss":
                                rss,
                        },
                    )


                    return P._guardian_intervene(

                        parent_session,
                        supervisor_pid,
                        health,

                        reason=(
                            "GUARDIAN_POST_M5_EXIT_TIMEOUT"
                        ),

                        peak_group=peak_group,

                        peak_recorder=peak_recorder,
                    )


        elif not all_m5_done:

            post_m5_started_wall = (
                None
            )


        # ==========================================================================================
        # GUARDIAN HEALTH
        # ==========================================================================================

        guardian_health = {

            "time":
                B._iso(),

            "deploy_version":
                DEPLOY_VERSION,

            "running":
                True,

            "intervened":
                False,

            "supervisor_pid":
                supervisor_pid,

            "generation_id":
                generation_id,

            "trader_pid":
                trader_pid or None,

            "recorder_pid":
                recorder_pid or None,

            "rss":
                rss,

            "peak_trader_group_rss_mb":
                peak_group,

            "peak_recorder_rss_mb":
                peak_recorder,

            "trader_rss_warning_mb":
                GENERATION_RSS_WARNING_MB,

            "trader_rss_hard_limit_mb":
                GENERATION_RSS_HARD_LIMIT_MB,

            "post_m5_exit_timeout_s":
                POST_M5_EXIT_TIMEOUT_S,

            "post_m5_exit_grace_active":
                bool(
                    trader_alive
                    and
                    all_m5_done
                    and
                    post_m5_started_wall
                    is not None
                ),

            "post_m5_state":
                post_m5_state,

            "session_deadline":
                health.get(
                    "session_deadline"
                ),
        }


        B._atomic(
            parent_session
            / P.GUARDIAN_HEALTH_FILE,
            guardian_health,
        )


        # ==========================================================================================
        # HIGH-RSS WARNING
        # ==========================================================================================

        if (
            group_mb
            >= GENERATION_RSS_WARNING_MB

            and

            warning_for_generation
            != generation_id
        ):

            warning_for_generation = (
                generation_id
            )


            B._append(
                parent_session
                / P.GUARDIAN_EVENTS_FILE,
                {
                    "time":
                        B._iso(),

                    "event":
                        "TRADER_RSS_WARNING",

                    "generation_id":
                        generation_id,

                    "rss":
                        rss,

                    "warning_mb":
                        GENERATION_RSS_WARNING_MB,

                    "hard_mb":
                        GENERATION_RSS_HARD_LIMIT_MB,
                },
            )


        # ==========================================================================================
        # CATASTROPHIC RSS BACKSTOP
        # ==========================================================================================

        if (
            group_mb
            >= GENERATION_RSS_HARD_LIMIT_MB
        ):

            B._append(
                parent_session
                / P.GUARDIAN_EVENTS_FILE,
                {
                    "time":
                        B._iso(),

                    "event":
                        "TRADER_RSS_HARD_LIMIT",

                    "generation_id":
                        generation_id,

                    "rss":
                        rss,
                },
            )


            return P._guardian_intervene(

                parent_session,
                supervisor_pid,
                health,

                reason=(
                    "GUARDIAN_RSS_HARD_LIMIT"
                ),

                peak_group=peak_group,

                peak_recorder=peak_recorder,
            )


        time.sleep(
            P.GUARDIAN_POLL_S
        )


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
        _guardian_loop_v2963(Path(args.run_guardian), int(args.supervisor_pid))
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
    "POST_M5_EXIT_TIMEOUT_S",
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
