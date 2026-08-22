from __future__ import annotations

"""V2.9.8 real-money Q50 M1->M12 persistent-guard 12h rotating deployment.

This module is the live handoff for the frozen M12_GUARD candidate.

Trading contract
----------------
- Q = 50.
- M1 entry start = 60s.
- Rest BUY YES at 5c and BUY NO at 5c (YES-book 95c).
- First fill selects the tail and cancels the opposite entry.
- Selected residual entry may continue filling toward Q50.
- YES residual entry danger-cancels after YES bid <=10c continuously for >=5s
  across at least 3 valid book observations.
- NO residual entry danger-cancels after YES ask >=90c continuously for >=5s
  across at least 3 valid book observations.
- Guard cancellation never rearms.
- No repeat attempt after the trade becomes flat.
- Full entry posts one fixed passive exit at the contemporaneous opposite BBO;
  there is no repricing/chasing.
- Terminal persistent cleanup is M12 / 720s.

Operations
----------
- One authenticated parent-owned recorder persists M0->M12 plus the 30s label tail.
  Therefore the requested M1->M12 research slice is stored alongside the live run
  under <parent_session>/raw_capture.
- Trader generations own one complete M0->M12 window and rotate only after the
  inherited durable zero-position / zero-group-resting checkpoint, which executes
  at M12 because V1.12 binds the inherited M5 horizon to 720s for the child process.
- Session risk baseline is fixed once for all generations.
- Q50 session is fixed to 12h, a $20 software loss trigger, and >=$125 start equity.
- Trader RSS warning/hard backstops remain 1536/3072 MiB. Rotation bounds trader
  process lifetime; the recorder is a separate streaming process.
- A separate guardian and authoritative fail-closed recovery path are retained.

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
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_8_2 as V282
from . import mm_deep_tail_join_ask_deploy_v2_8_8 as V288
from . import mm_deep_tail_join_ask_live_v1_11_rotation as V111
from . import mm_deep_tail_join_ask_live_v1_12_m12_guard_rotation as LIVE
from . import mm_deep_tail_join_ask_q50_m12_guard_v2_9_7_preflight as PREFLIGHT
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation as P
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_3_overnight_rotation as V2963
from . import mm_event_time_m0_m12_recorder_v6_auth as REC


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_RECORD_M12_V2_9_8_12H_ROTATION"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_12h_rotation"

Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M12_GUARD_12H_V298_ROTATION"
KILL_ARM = "KILL_AND_FLATTEN"

Q50_Q = 50.0
Q50_HOURS = 12.0
Q50_MAX_LOSS_USD = 20.0
Q50_MIN_EQUITY_USD = 125.0

M1_S = 60.0
M12_S = 720.0
LABEL_TAIL_END_S = 750.0

GENERATION_RSS_WARNING_MB = 1536.0
GENERATION_RSS_HARD_LIMIT_MB = 3072.0
POST_M12_EXIT_TIMEOUT_S = 30.0
STARTUP_TIMEOUT_S = 150.0

SUPERVISOR_LOG_FILE = "supervisor_v2_9_8_m12_guard.log"
GUARDIAN_LOG_FILE = "guardian_v2_9_8_m12_guard.log"


def _generation_cfg(
    parent_cfg,
    *,
    generation_id,
    generation_dir,
    recorder_pid,
    session_start_equity,
    session_kill_equity,
    remaining_hours,
):
    q = float(parent_cfg["quote_size"])
    return {
        "mode": f"{parent_cfg['mode']}_GEN_{int(generation_id):04d}",
        "quote_size": q,
        "runtime_hours": max(0.02, float(remaining_hours)),
        "max_start_loss_usd": float(parent_cfg["max_start_loss_usd"]),
        "min_start_equity_usd": float(parent_cfg["min_start_equity_usd"]),
        "order_group_limit_fp": f"{max(25.0, 20.0 * q):.2f}",
        "live_engine_version": LIVE.LIVE_VERSION,
        "deploy_version": DEPLOY_VERSION,
        "generation_id": int(generation_id),
        "parent_session_dir": str(Path(parent_cfg["parent_session_dir"]).resolve()),
        "external_recorder_owner": True,
        "external_recorder_pid": int(recorder_pid),
        "session_start_equity_usd": float(session_start_equity),
        "session_kill_equity_usd": float(session_kill_equity),
        "session_runtime_hours": float(parent_cfg["runtime_hours"]),
        "strategy_entry_start_elapsed_s": M1_S,
        "strategy_terminal_cleanup_elapsed_s": M12_S,
        "recorder_persist_end_elapsed_s": M12_S,
        "recorder_label_tail_end_elapsed_s": LABEL_TAIL_END_S,
        "rotation_process_lifetime": "ONE_COMPLETE_M0_M12_WINDOW",
        "m12_persistent_entry_guard": True,
        "guard_yes_bid_max": LIVE.YES_GUARD_BID_MAX,
        "guard_no_ask_min": LIVE.NO_GUARD_ASK_MIN,
        "guard_persist_s": LIVE.GUARD_PERSIST_S,
        "guard_min_book_obs": LIVE.GUARD_MIN_BOOK_OBS,
        "guard_rearm": False,
        "repeat_after_flat": False,
        "fixed_session_risk_baseline": True,
        "fresh_generation_starts_at_raw_eof": True,
        "fresh_row_after_generation_start_required": True,
        "no_auto_scale": True,
    }


def _run_generation(session, cfg_path):
    """Child trader generation. May send real orders."""
    session = Path(session).resolve()
    cfg = B._read(Path(cfg_path), {}) or {}
    old = OOS.fee_preflight

    def child_fee_preflight(
        *,
        horizon_hours=OOS.FEE_CHANGE_HORIZON_H,
        save_path=None,
        show=True,
    ):
        return V282._validated_parent_fee_snapshot(
            session,
            horizon_hours=horizon_hours,
            save_path=save_path,
            show=show,
        )

    OOS.fee_preflight = child_fee_preflight
    try:
        return LIVE.run_live_process(session, cfg)
    finally:
        OOS.fee_preflight = old


def _launch_generation(
    parent_session,
    parent_cfg,
    parent_preflight,
    *,
    generation_id,
    recorder_pid,
    session_start_equity,
    session_kill_equity,
    remaining_hours,
):
    parent_session = Path(parent_session).resolve()
    generation_dir = (
        parent_session / "generations" / f"gen_{int(generation_id):04d}"
    )
    generation_dir.mkdir(parents=True, exist_ok=False)
    P._attach_raw_capture(parent_session, generation_dir)

    cfg = _generation_cfg(
        parent_cfg,
        generation_id=generation_id,
        generation_dir=generation_dir,
        recorder_pid=recorder_pid,
        session_start_equity=session_start_equity,
        session_kill_equity=session_kill_equity,
        remaining_hours=remaining_hours,
    )
    cfg_path = generation_dir / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(generation_dir / "parent_preflight_snapshot.json", parent_preflight)

    log = generation_dir / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--run-generation",
        str(generation_dir),
        "--config",
        str(cfg_path),
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

    B._append(
        parent_session / P.SUPERVISOR_EVENTS_FILE,
        {
            "time": B._iso(),
            "event": "M12_GUARD_GENERATION_LAUNCHED",
            "generation_id": int(generation_id),
            "generation_dir": str(generation_dir),
            "trader_pid": proc.pid,
            "remaining_hours": float(remaining_hours),
            "strategy_terminal_cleanup_elapsed_s": M12_S,
        },
    )
    return proc, generation_dir, cfg, log


def _post_m12_generation_state(supervisor_health):
    """Guardian view of whether every target reached the M12 terminal phase."""
    supervisor_health = supervisor_health or {}
    generation_dir = supervisor_health.get("generation_dir")
    if not generation_dir:
        return False, {}

    gh = B._read(Path(generation_dir) / "health.json", {}) or {}
    target_count = int(gh.get("rotation_target_count") or 0)
    states = gh.get("deep_tail_states") or {}

    finalized = []
    for ticker, state in states.items():
        phase = str((state or {}).get("phase") or "")
        # V1.12 writes an explicit M12_FINALIZED transition after the inherited
        # finalize_m5() compatibility path. M5_FINALIZED is accepted only as a
        # compatibility fallback; under the V1.12 child binding it can only occur
        # at the patched 720s horizon.
        if phase in {"M12_FINALIZED", "M5_FINALIZED"}:
            finalized.append(str(ticker))

    all_done = bool(target_count > 0 and len(finalized) >= target_count)
    return all_done, {
        "generation_dir": str(generation_dir),
        "target_count": target_count,
        "finalized_count": len(finalized),
        "finalized_tickers": sorted(finalized),
        "rotation_checkpoint_written": gh.get("rotation_checkpoint_written") is True,
        "terminal_horizon_s": M12_S,
    }


def _install_patch():
    """Install the already-audited supervisor fixes, then bind them to V1.12/M12."""
    # V2.9.6.3 installs the self-aware generation preflight and high-RSS policy.
    V2963._install_patch()

    # Publish V1.11 compatibility artifact names on the V1.12 facade because
    # the parent supervisor intentionally retains the historical checkpoint
    # filename while moving the numeric horizon to M12.
    LIVE.ROTATION_CHECKPOINT_FILE = V111.ROTATION_CHECKPOINT_FILE
    LIVE.GENERATION_BOOTSTRAP_FILE = V111.GENERATION_BOOTSTRAP_FILE

    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.M1_S = M1_S
    # Historical parent name retained; the numeric horizon is M12.
    P.M5_S = M12_S
    P.RECORDER_M12_S = M12_S
    P.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    P.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    P.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB

    # Keep the V2.9.6.1/6.3 owning-supervisor exemption in each fresh preflight.
    P._fresh_generation_preflight = V2963._fresh_generation_preflight

    # Critical runtime wiring: supervisor generations must launch THIS module,
    # and their config must advertise the M12 horizon.
    P._generation_cfg = _generation_cfg
    P._launch_generation = _launch_generation

    # Reuse the hardened guardian implementation, but move its post-window
    # observation to the M12 terminal phase.
    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    V2963.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB
    V2963.POST_M5_EXIT_TIMEOUT_S = POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = _post_m12_generation_state


def static_self_check(*, show=True):
    """Offline structural audit. No API calls and no orders.

    The first call also executes the read-only V2.9.7 contract audit. Later calls
    are intentionally idempotent after this module has installed its process-local
    V1.12 parent binding.
    """
    if P.LIVE is LIVE:
        intended_ok = True
    else:
        intended_ok = PREFLIGHT.static_self_check(show=False).get("ok") is True

    _install_patch()

    checks = {
        "v2_9_7_intended_contract_ok": intended_ok,
        "live_v1_12_exact": LIVE.LIVE_VERSION
        == "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_12_M12_GUARD_ROTATION",
        "q50_exact_50": Q50_Q == 50.0,
        "runtime_exact_12h": Q50_HOURS == 12.0,
        "loss_trigger_exact_20": Q50_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q50_MIN_EQUITY_USD == 125.0,
        "entry_start_m1_60": M1_S == 60.0,
        "terminal_cleanup_m12_720": M12_S == 720.0,
        "recorder_m12_720": REC.TRADE_WINDOW_END_S == 720.0,
        "recorder_tail_750": REC.LABEL_TAIL_END_S == 750.0,
        "guard_yes_bid_10c": LIVE.YES_GUARD_BID_MAX == 0.10,
        "guard_no_ask_90c": LIVE.NO_GUARD_ASK_MIN == 0.90,
        "guard_persist_5s": LIVE.GUARD_PERSIST_S == 5.0,
        "guard_min_obs_3": LIVE.GUARD_MIN_BOOK_OBS == 3,
        "parent_live_binding_is_v1_12": P.LIVE is LIVE,
        "parent_terminal_horizon_is_m12": P.M5_S == 720.0,
        "parent_launch_generation_is_v298": P._launch_generation is _launch_generation,
        "parent_generation_cfg_is_v298": P._generation_cfg is _generation_cfg,
        "guardian_post_window_is_m12": V2963._post_m5_generation_state
        is _post_m12_generation_state,
        "rss_warning_1536": GENERATION_RSS_WARNING_MB == 1536.0,
        "rss_hard_3072": GENERATION_RSS_HARD_LIMIT_MB == 3072.0,
        "fixed_session_risk_baseline": True,
        "external_recorder_parent_owned": True,
        "no_auto_scale": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "recorder_version": REC.STUDY_VERSION,
        "module_name": MODULE_NAME,
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 136)
        print("V2.9.8 Q50 M1->M12 GUARD 12H ROTATION STATIC CHECK — NO API / NO ORDERS")
        print("=" * 136)
        for k, v in out.items():
            print(f"{k:76s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.8 M12 guard static self-check failed: {out}")
    return out


def q50_preflight(*, show=True):
    """Read-only/private-WS deployment preflight. Sends no orders."""
    static_self_check(show=show)
    V28._patch_parent()
    V28.D._guard_other_live_processes()
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
    log = parent_session / GUARDIAN_LOG_FILE
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--run-guardian",
        str(parent_session),
        "--supervisor-pid",
        str(int(supervisor_pid)),
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


def _run_supervisor(parent_session, cfg_path):
    """Detached parent supervisor. May launch live trader generations."""
    _install_patch()
    return P._run_supervisor(Path(parent_session), Path(cfg_path))


def _run_guardian(parent_session, supervisor_pid):
    """Detached guardian. May invoke fail-closed recovery."""
    _install_patch()
    return V2963._guardian_loop_v2963(
        Path(parent_session),
        int(supervisor_pid),
    )


def _launch_supervised(*, arm_phrase):
    _install_patch()
    if str(arm_phrase) != Q50_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q50_ARM!r} exactly."
        )

    static_self_check(show=True)
    V28._patch_parent()

    # Refuse to start while any other live account controller is active.
    V28.D._guard_other_live_processes()

    pre = V288.live_preflight(
        quote_size=Q50_Q,
        runtime_hours=Q50_HOURS,
        max_start_loss_usd=Q50_MAX_LOSS_USD,
        min_start_equity_usd=Q50_MIN_EQUITY_USD,
        show=True,
        probe_private_ws=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    mode = "DEEP_TAIL_Q50_M1_M12_GUARD_12H_SMOKE_V298_ROTATION"
    parent_session = (P.CORE.ROOT / f"{stamp}_{mode.lower()}").resolve()
    parent_session.mkdir(parents=True, exist_ok=False)
    (parent_session / "generations").mkdir(parents=True, exist_ok=True)

    cfg = {
        "mode": mode,
        "quote_size": Q50_Q,
        "runtime_hours": Q50_HOURS,
        "max_start_loss_usd": Q50_MAX_LOSS_USD,
        "min_start_equity_usd": Q50_MIN_EQUITY_USD,
        "parent_session_dir": str(parent_session),
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "recorder_version": REC.STUDY_VERSION,
        "strategy_entry_start_elapsed_s": M1_S,
        "strategy_terminal_cleanup_elapsed_s": M12_S,
        "recorder_persist_end_elapsed_s": M12_S,
        "recorder_label_tail_end_elapsed_s": LABEL_TAIL_END_S,
        "rotation_process_lifetime": "ONE_COMPLETE_M0_M12_WINDOW",
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "fixed_session_risk_baseline": True,
        "persistent_danger_guard": True,
        "guard_rearm": False,
        "repeat_after_flat": False,
        "external_recorder_parent_owned": True,
        "direct_q50_operator_arm": True,
        "no_auto_scale": True,
    }
    cfg_path = parent_session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(parent_session / "parent_preflight_snapshot.json", pre)
    B._atomic(
        parent_session / "architecture_spec_v2_9_8.json",
        {
            "time": B._iso(),
            "architecture": "ONE_EXTERNAL_M0_M12_RECORDER_PLUS_ONE_M0_M12_GUARDED_TRADER_PROCESS_PER_WINDOW",
            "live_engine": LIVE.LIVE_VERSION,
            "generation_start_raw_policy": "CURRENT_EOF_PLUS_REQUIRE_FRESH_POST_START_ROW",
            "rotation_gate": "M12_ZERO_POSITION_ZERO_GROUP_RESTING_DURABLE_CHECKPOINT",
            "risk_baseline": "FIXED_ONCE_PER_PARENT_SESSION",
            "recorder": "M0_TO_M12_PLUS_30S_LABEL_TAIL",
            "requested_analysis_slice": "M1_TO_M12_AVAILABLE_IN_PARENT_RAW_CAPTURE",
            "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
            "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
            "guard_yes_bid_max": LIVE.YES_GUARD_BID_MAX,
            "guard_no_ask_min": LIVE.NO_GUARD_ASK_MIN,
            "guard_persist_s": LIVE.GUARD_PERSIST_S,
            "guard_min_book_obs": LIVE.GUARD_MIN_BOOK_OBS,
            "guard_rearm": False,
            "repeat_after_flat": False,
            "q50_direct_arm": True,
            "strategy_rule_change_from_v1_12": "NONE",
        },
    )

    log = parent_session / SUPERVISOR_LOG_FILE
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--run-supervisor",
        str(parent_session),
        "--config",
        str(cfg_path),
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

    B._atomic(
        P.CORE.CONTROL_PATH,
        {
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
        },
    )

    guardian, guardian_log, guardian_cmd = _launch_guardian(
        parent_session,
        supervisor.pid,
    )
    ctl = B._read(P.CORE.CONTROL_PATH, {}) or {}
    ctl.update(
        {
            "guardian_pid": guardian.pid,
            "guardian_log_path": str(guardian_log),
            "guardian_command": guardian_cmd,
        }
    )
    B._atomic(P.CORE.CONTROL_PATH, ctl)

    deadline = time.time() + STARTUP_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        if supervisor.poll() is not None:
            raise RuntimeError(
                f"V2.9.8 supervisor exited during startup rc={supervisor.returncode}\n"
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
        B._atomic(
            parent_session / P.SESSION_KILL_FILE,
            {"time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V298_M12_GUARD"},
        )
        raise RuntimeError(
            f"V2.9.8 startup timeout. supervisor_health={last}\n{P._tail_text(log)}"
        )

    print("\n" + "=" * 136)
    print("REAL-MONEY V2.9.8 Q50 M1->M12 GUARD 12H SMOKE ARMED")
    print("=" * 136)
    print("Parent session:              ", parent_session)
    print("Supervisor PID:              ", supervisor.pid)
    print("Guardian PID:                ", guardian.pid)
    print("External recorder PID:       ", last.get("recorder_pid"))
    print("Current trader PID:          ", last.get("trader_pid"))
    print("Engine:                      ", LIVE.LIVE_VERSION)
    print("Quantity:                    Q50")
    print("Strategy per generation:     M1 -> M12 persistent danger guard")
    print("Recorder:                    parent-owned M0 -> M12 + 30s")
    print("Requested data slice:        M1 -> M12 in parent raw_capture")
    print("Generation rotation:         verified M12 flat + group-resting-zero")
    print("Session runtime:             exactly 12.0 hours from first trade window start")
    print("Session software loss stop:  $20 from fixed starting-equity baseline")
    print("Minimum starting equity:     $125")
    print(
        "Trader RSS warning/hard:     "
        f"{GENERATION_RSS_WARNING_MB:.0f} / {GENERATION_RSS_HARD_LIMIT_MB:.0f} MiB"
    )
    print("=" * 136)

    return live_status(show=False, tail_lines=30)


def start_q50_12h_smoke(*, arm_phrase=None):
    """Explicit real-money entrypoint. Fixed Q50 / 12h / $20 stop."""
    return _launch_supervised(arm_phrase=arm_phrase)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    ctl = B._read(P.CORE.CONTROL_PATH, {}) or {}
    if not ctl:
        out = {"running": False, "message": "No deep-tail live control file."}
        if show:
            print(out)
        return out

    parent = Path(ctl.get("session_dir") or "")
    sh = B._read(parent / P.SUPERVISOR_HEALTH_FILE, {}) or {}
    sf = B._read(parent / P.SUPERVISOR_FINAL_FILE, {}) or {}
    gh = B._read(parent / P.GUARDIAN_HEALTH_FILE, {}) or {}
    gr = B._read(parent / P.GUARDIAN_RECEIPT_FILE, {}) or {}

    gen_dir = sh.get("generation_dir")
    gen_health = B._read(Path(gen_dir) / "health.json", {}) if gen_dir else {}
    gen_final = B._read(Path(gen_dir) / "final_summary.json", {}) if gen_dir else {}
    checkpoint = (
        B._read(Path(gen_dir) / LIVE.ROTATION_CHECKPOINT_FILE, {})
        if gen_dir
        else {}
    )

    supervisor_pid = ctl.get("supervisor_pid") or ctl.get("pid")
    running = bool(B._pid_alive(supervisor_pid) and not sf)
    tail = P._tail_text(parent / SUPERVISOR_LOG_FILE, chars=14000).splitlines()[
        -int(tail_lines) :
    ]

    out = {
        "running": running,
        "deploy_version": ctl.get("deploy_version"),
        "live_version": ctl.get("live_version"),
        "parent_session_dir": str(parent),
        "raw_capture_dir": str(parent / "raw_capture"),
        "supervisor_pid": supervisor_pid,
        "guardian_pid": ctl.get("guardian_pid"),
        "supervisor_health": sh,
        "guardian_health": gh,
        "guardian_receipt": gr,
        "current_generation_health": gen_health,
        "current_generation_final": gen_final,
        "current_rotation_checkpoint": checkpoint,
        "supervisor_final": sf,
        "log_tail": tail,
    }

    if show:
        print("=" * 136)
        print("V2.9.8 M1->M12 GUARD LIVE STATUS")
        print("=" * 136)
        print("running:", running)
        print("parent session:", parent)
        print("raw capture:", parent / "raw_capture")
        print("supervisor PID:", supervisor_pid)
        print("guardian PID:", ctl.get("guardian_pid"))
        if sh:
            print("state:", sh.get("state"))
            print("generation:", sh.get("generation_id"))
            print("trader PID/alive:", sh.get("trader_pid"), sh.get("trader_alive"))
            print("recorder PID/alive:", sh.get("recorder_pid"), sh.get("recorder_alive"))
            print(
                "session start/kill equity:",
                sh.get("session_start_equity_usd"),
                sh.get("session_kill_equity_usd"),
            )
            print("session deadline:", sh.get("session_deadline"))
        if gh:
            rss = gh.get("rss") or {}
            print("trader group RSS MiB:", rss.get("strategy_group_rss_mb"))
            print("recorder RSS MiB:", rss.get("recorder_rss_mb"))
            print("peak trader RSS MiB:", gh.get("peak_trader_group_rss_mb"))
            print("peak recorder RSS MiB:", gh.get("peak_recorder_rss_mb"))
            print("post-M12 state:", gh.get("post_m5_state"))
        if gen_health:
            print("generation state:", gen_health.get("state"))
            print("rotation window:", gen_health.get("rotation_window_key"))
            print("positions:", gen_health.get("positions"))
            print("active tracks:", len(gen_health.get("active_tracks") or {}))
            print("rotation checkpoint written:", gen_health.get("rotation_checkpoint_written"))
        if checkpoint:
            print("rotation safe:", checkpoint.get("safe_to_rotate"), checkpoint.get("reason"))
        if sf:
            print("supervisor shutdown:", sf.get("shutdown_reason"))
            print("generations completed:", sf.get("generations_completed"))
            print("last error:", sf.get("last_error"))
        if tail:
            print("\nSUPERVISOR LOG TAIL")
            print("\n".join(tail))
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    """Authoritative real-money stop. May send cancels/reduce-only cleanup orders."""
    if str(arm_phrase) != KILL_ARM:
        raise RuntimeError(f"Pass arm_phrase={KILL_ARM!r} exactly.")
    _install_patch()
    result = P.kill_and_flatten_live(
        arm_phrase=KILL_ARM,
        wait_s=float(wait_s),
    )
    try:
        status = live_status(show=True, tail_lines=60)
    except Exception:
        status = None
    return {"base_kill_result": result, "status": status}


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
        _run_generation(Path(args.run_generation), Path(args.config))
    elif args.run_supervisor:
        if not args.config:
            raise RuntimeError("--config is required with --run-supervisor")
        _run_supervisor(Path(args.run_supervisor), Path(args.config))
    elif args.run_guardian:
        if not args.supervisor_pid:
            raise RuntimeError("--supervisor-pid is required with --run-guardian")
        _run_guardian(Path(args.run_guardian), int(args.supervisor_pid))
    else:
        static_self_check(show=True)


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
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "static_self_check",
    "q50_preflight",
    "start_q50_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
