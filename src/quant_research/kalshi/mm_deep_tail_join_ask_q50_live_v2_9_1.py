from __future__ import annotations

"""Q50 one-hour real-money validation on the V1.7 memory-safe deep-tail stack.

This is a fixed-size operational/economic validation, NOT an automatic promotion.
Strategy mechanics are unchanged: M1 dual 5c passive entries, first-fill-wins,
opposite cancellation, selected-tail accumulation, fixed JOIN_ASK/no-reprice exit,
persistent M5 cleanup, authenticated V5 recorder, and the external V2.8 guardian.

Risk configuration is fixed to Q50, one hour, and a $20 start-to-current calibrated
equity software stop. The software stop is not a guaranteed final realized-loss cap.

Importing this module sends no orders.
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
from . import mm_deep_tail_join_ask_deploy_v2_8_7 as V287
from . import mm_deep_tail_join_ask_deploy_v2_8_8 as V288
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_7 as LIVE
from . import mm_event_time_m0_m5_recorder_v5_auth as V5A


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_LIVE_V2_9_1_MEMORY_SAFE"
CORE = V282.CORE
Q50_ARM = "LIVE_DEEP_TAIL_Q50_1H_V291"
KILL_ARM = V282.KILL_ARM

Q50_Q = 50.0
Q50_HOURS = 1.0
Q50_MAX_LOSS_USD = 20.0
Q50_MIN_EQUITY_USD = 125.0
STARTUP_TIMEOUT_S = V282.STARTUP_TIMEOUT_S


def static_self_check(*, show=True):
    pp = V287._install_child_pythonpath()
    live = LIVE.static_self_check(show=False)
    checks = {
        "live_v1_7_ok": live.get("ok") is True,
        "authenticated_v5_discovery": live.get("authenticated_v5_discovery") is True,
        "bounded_raw_ingestion_required": live.get("bounded_raw_ingestion_required") is True,
        "rest_fill_incremental_min_ts": live.get("rest_fill_incremental_min_ts") is True,
        "rest_fill_exact_dedupe": live.get("rest_fill_exact_dedupe") is True,
        "private_ws_idle_timeout_no_reconnect": live.get("private_ws_idle_timeout_no_reconnect") is True,
        "alpha_rules_unchanged": live.get("alpha_rules_unchanged") is True,
        "q50_on_allowed_ladder": int(Q50_Q) in tuple(int(x) for x in V1.LADDER_Q),
        "runtime_fixed_1h": abs(Q50_HOURS - 1.0) < 1e-12,
        "loss_trigger_fixed_20": abs(Q50_MAX_LOSS_USD - 20.0) < 1e-12,
        "minimum_equity_125": abs(Q50_MIN_EQUITY_USD - 125.0) < 1e-12,
        "guardian_2gb_hard_limit_retained": abs(V28.RSS_HARD_LIMIT_MB - 2048.0) < 1e-12,
        "strict_pre_m0_data_gate_disabled": True,
        "auto_scaling_disabled": True,
        "child_pythonpath_installed": pp.get("installed") is True,
    }
    ok = all(checks.values())
    out = {
        "deploy_version": DEPLOY_VERSION,
        "quantity": Q50_Q,
        "runtime_hours": Q50_HOURS,
        "max_start_loss_usd": Q50_MAX_LOSS_USD,
        "min_start_equity_usd": Q50_MIN_EQUITY_USD,
        "live_engine_version": LIVE.LIVE_VERSION,
        "recorder_version": V5A.STUDY_VERSION,
        **checks,
        "ok": bool(ok),
        "orders_sent": False,
    }
    if show:
        print("=" * 112)
        print("DEEP-TAIL Q50 V2.9.1 1H MEMORY-SAFE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 112)
        for k, v in out.items():
            print(f"{k:56s}: {v}")
    if not ok:
        raise RuntimeError(f"Q50 V2.9.1 static self-check failed: {out}")
    return out


def live_preflight(*, show=True):
    static_self_check(show=show)
    return V288.live_preflight(
        quote_size=Q50_Q,
        runtime_hours=Q50_HOURS,
        max_start_loss_usd=Q50_MAX_LOSS_USD,
        min_start_equity_usd=Q50_MIN_EQUITY_USD,
        show=show,
        probe_private_ws=True,
    )


def _run_child(session, cfg_path):
    session = Path(session).resolve()
    cfg = B._read(Path(cfg_path), {}) or {}

    old = OOS.fee_preflight

    def child_fee_preflight(*, horizon_hours=OOS.FEE_CHANGE_HORIZON_H,
                            save_path=None, show=True):
        return V282._validated_parent_fee_snapshot(
            session,
            horizon_hours=horizon_hours,
            save_path=save_path,
            show=show,
        )

    OOS.fee_preflight = child_fee_preflight
    try:
        LIVE.run_live_process(session, cfg)
    finally:
        OOS.fee_preflight = old


def start_q50(*, arm_phrase=None, runtime_hours=Q50_HOURS,
              max_start_loss_usd=Q50_MAX_LOSS_USD,
              min_start_equity_usd=Q50_MIN_EQUITY_USD):
    """REAL ORDERS: fixed Q50, fixed one hour, fixed $20 software-loss trigger."""
    if str(arm_phrase) != Q50_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q50_ARM!r} exactly."
        )
    if abs(float(runtime_hours) - Q50_HOURS) > 1e-12:
        raise RuntimeError("Q50 V2.9.1 is fixed to exactly 1.0 hour.")
    if abs(float(max_start_loss_usd) - Q50_MAX_LOSS_USD) > 1e-12:
        raise RuntimeError("Q50 V2.9.1 is fixed to exactly a $20 software loss trigger.")
    if float(min_start_equity_usd) + 1e-12 < Q50_MIN_EQUITY_USD:
        raise RuntimeError("Q50 V2.9.1 requires minimum starting equity of at least $125.")

    static_self_check(show=True)
    V28._patch_parent()
    V28.D._guard_other_live_processes()

    pre = V288.live_preflight(
        quote_size=Q50_Q,
        runtime_hours=Q50_HOURS,
        max_start_loss_usd=Q50_MAX_LOSS_USD,
        min_start_equity_usd=float(min_start_equity_usd),
        show=True,
        probe_private_ws=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    mode = "DEEP_TAIL_Q50_1H_V17_V291"
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}").resolve()
    session.mkdir(parents=True, exist_ok=False)

    group_limit = max(25.0, 20.0 * Q50_Q)
    cfg = {
        "mode": mode,
        "quote_size": Q50_Q,
        "runtime_hours": Q50_HOURS,
        "max_start_loss_usd": Q50_MAX_LOSS_USD,
        "min_start_equity_usd": float(min_start_equity_usd),
        "order_group_limit_fp": f"{group_limit:.2f}",
        "live_engine_version": LIVE.LIVE_VERSION,
        "deploy_version": V282.RUNTIME_BASE_DEPLOY_VERSION,
        "launch_wrapper_version": DEPLOY_VERSION,
        "child_fee_preflight_mode": "REUSE_FRESH_PARENT_PASS_SNAPSHOT_FAIL_CLOSED",
        "recorder_study_version": V5A.STUDY_VERSION,
        "recorder_discovery_transport": V5A.DISCOVERY_TRANSPORT_VERSION,
        "rest_fill_reconcile_mode": "MIN_TS_INCREMENTAL_DEDUP_CURSOR",
        "scientific_status": "Q50_1H_EXPLICIT_USER_REQUEST_NO_AUTOSCALE",
        "no_auto_scale": True,
        "guardian_rss_warning_mb": V28.RSS_WARNING_MB,
        "guardian_rss_hard_limit_mb": V28.RSS_HARD_LIMIT_MB,
        "guardian_deadline_grace_s": V28.GUARDIAN_DEADLINE_GRACE_S,
        "normal_runtime_cleanup_grace_s": V28.NORMAL_RUNTIME_CLEANUP_GRACE_S,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(session / "parent_preflight_snapshot.json", pre)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_q50_live_v2_9_1",
        "--run-live-session", str(session),
        "--config", str(cfg_path),
    ]
    caffeinate = shutil.which("caffeinate")
    cmd = ([caffeinate, "-i", "-m"] + child) if caffeinate else child
    try:
        p = subprocess.Popen(
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
        "runtime_base_deploy_version": V282.RUNTIME_BASE_DEPLOY_VERSION,
        "running": True,
        "pid": p.pid,
        "session_dir": str(session),
        "mode": mode,
        "started_at": B._iso(),
        "config": cfg,
        "log_path": str(log),
        "caffeinate_used": bool(caffeinate),
        "launch_command": cmd,
    })

    deadline = time.time() + STARTUP_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
            raise RuntimeError(
                f"Q50 V2.9.1 process exited during startup rc={p.returncode}\n{tail}"
            )
        last = B._read(session / "health.json", {}) or {}
        state_ok = last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}
        private_ok = last.get("private_ws_ready") is True
        raw_ok = last.get("raw_watchdog_ready") is True
        bounded_ok = last.get("bounded_raw_ingestion") is True
        runtime_bound_ok = last.get("bounded_book_tail_runtime_verified") is True
        mem_ok = last.get("live_memory_hardening_version") == LIVE.LIVE_VERSION
        fee_ok = (
            B._read(session / "child_fee_preflight_reuse_v2_8_2.json", {}) or {}
        ).get("ok") is True
        if state_ok and private_ok and raw_ok and bounded_ok and runtime_bound_ok and mem_ok and fee_ok:
            break
        time.sleep(0.25)
    else:
        B._atomic(session / "KILL_REQUEST.json", {
            "time": B._iso(),
            "reason": "STARTUP_HEALTH_TIMEOUT_Q50_V291",
        })
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(
            f"Q50 V2.9.1 startup health timeout. Last health={last}\n{tail}"
        )

    guardian, guardian_log, guardian_cmd = V288._launch_guardian(session, p.pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update({
        "guardian_pid": guardian.pid,
        "guardian_log_path": str(guardian_log),
        "guardian_command": guardian_cmd,
    })
    B._atomic(CORE.CONTROL_PATH, ctl)

    print("\n" + "=" * 112)
    print("REAL-MONEY DEEP-TAIL Q50 V2.9.1 ARMED")
    print("=" * 112)
    print("Session:                    ", session)
    print("Main PID:                   ", p.pid)
    print("Guardian PID:               ", guardian.pid)
    print("Engine:                     ", LIVE.LIVE_VERSION)
    print("Recorder:                   ", V5A.STUDY_VERSION)
    print("Quantity:                   Q50 per eligible market")
    print("Runtime:                    1.00 hour from first complete live window")
    print("Software loss trigger:      -$20.00 from calibrated starting equity")
    print("REST fill fallback:         incremental min_ts + dedupe + cursor")
    print("Bounded raw tail verified:  YES")
    print("Guardian hard RSS limit:    ", f"{V28.RSS_HARD_LIMIT_MB:.0f} MB (UNCHANGED)")
    print("Auto-scaling:               DISABLED")
    print("=" * 112)
    return live_status(show=False, tail_lines=20)


def live_status(*args, **kwargs):
    return V288.live_status(*args, **kwargs)


def kill_and_flatten_live(*args, **kwargs):
    return V288.kill_and_flatten_live(*args, **kwargs)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    a = ap.parse_args()
    if a.run_live_session:
        if not a.config:
            raise RuntimeError("--config is required with --run-live-session")
        _run_child(Path(a.run_live_session), Path(a.config))
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "Q50_ARM",
    "KILL_ARM",
    "Q50_Q",
    "Q50_HOURS",
    "Q50_MAX_LOSS_USD",
    "Q50_MIN_EQUITY_USD",
    "static_self_check",
    "live_preflight",
    "start_q50",
    "live_status",
    "kill_and_flatten_live",
]
