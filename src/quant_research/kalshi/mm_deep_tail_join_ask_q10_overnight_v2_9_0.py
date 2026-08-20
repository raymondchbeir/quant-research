from __future__ import annotations

"""V2.9.0 Q10 8-hour overnight runner with V1.7 long-run memory hardening.

This supersedes V2.8.9 for another overnight validation after the 2026-08-20 run
hit the external 2 GiB RSS guardian near its 8-hour boundary. Trading mechanics,
Q10 sizing, 8-hour runtime and $20 software-loss trigger are unchanged.

Operational change only: use deep-tail live V1.7, which keeps authenticated V5
recording but makes REST fill reconciliation incremental/deduplicated, keeps quiet
private websocket sessions open, requires the bounded raw tail at runtime, and
publishes memory/queue telemetry.

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


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q10_OVERNIGHT_V2_9_0_MEMORY_SAFE"
CORE = V282.CORE
Q10_ARM = "LIVE_DEEP_TAIL_Q10_8H_V290"
KILL_ARM = V282.KILL_ARM

Q10_Q = 10.0
Q10_HOURS = 8.0
Q10_MAX_LOSS_USD = 20.0
Q10_MIN_EQUITY_USD = 125.0
STARTUP_TIMEOUT_S = V282.STARTUP_TIMEOUT_S


def static_self_check(*, show=True):
    pp = V287._install_child_pythonpath()
    base = V282.static_self_check(show=False)
    live = LIVE.static_self_check(show=False)
    checks = {
        "base_v2_8_2_ok": base.get("ok") is True,
        "live_v1_7_ok": live.get("ok") is True,
        "authenticated_v5_discovery": live.get("authenticated_v5_discovery") is True,
        "bounded_raw_ingestion_required": live.get("bounded_raw_ingestion_required") is True,
        "rest_fill_incremental_min_ts": live.get("rest_fill_incremental_min_ts") is True,
        "rest_fill_exact_dedupe": live.get("rest_fill_exact_dedupe") is True,
        "private_ws_idle_timeout_no_reconnect": live.get("private_ws_idle_timeout_no_reconnect") is True,
        "alpha_rules_unchanged": live.get("alpha_rules_unchanged") is True,
        "q10_on_allowed_ladder": int(Q10_Q) in tuple(int(x) for x in V1.LADDER_Q),
        "runtime_fixed_8h": abs(Q10_HOURS - 8.0) < 1e-12,
        "loss_trigger_fixed_20": abs(Q10_MAX_LOSS_USD - 20.0) < 1e-12,
        "minimum_equity_125": abs(Q10_MIN_EQUITY_USD - 125.0) < 1e-12,
        "guardian_2gb_hard_limit_retained": abs(V28.RSS_HARD_LIMIT_MB - 2048.0) < 1e-12,
        "strict_pre_m0_data_gate_disabled": True,
        "auto_scaling_disabled": True,
        "child_pythonpath_installed": pp.get("installed") is True,
    }
    ok = all(checks.values())
    out = {
        "deploy_version": DEPLOY_VERSION,
        "quantity": Q10_Q,
        "runtime_hours": Q10_HOURS,
        "max_start_loss_usd": Q10_MAX_LOSS_USD,
        "min_start_equity_usd": Q10_MIN_EQUITY_USD,
        "live_engine_version": LIVE.LIVE_VERSION,
        "recorder_version": V5A.STUDY_VERSION,
        **checks,
        "ok": bool(ok),
        "orders_sent": False,
    }
    if show:
        print("=" * 112)
        print("DEEP-TAIL Q10 V2.9.0 OVERNIGHT MEMORY-SAFE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 112)
        for k, v in out.items():
            print(f"{k:56s}: {v}")
    if not ok:
        raise RuntimeError(f"Q10 V2.9.0 static self-check failed: {out}")
    return out


def live_preflight(*, show=True):
    return V288.live_preflight(
        quote_size=Q10_Q,
        runtime_hours=Q10_HOURS,
        max_start_loss_usd=Q10_MAX_LOSS_USD,
        min_start_equity_usd=Q10_MIN_EQUITY_USD,
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


def _launch_guardian(session, main_pid):
    # Same external 2 GiB guardian. We are fixing the allocator churn, not hiding it
    # by raising the memory ceiling.
    return V288._launch_guardian(Path(session).resolve(), int(main_pid))


def start_q10_overnight(*, arm_phrase=None, runtime_hours=Q10_HOURS,
                        max_start_loss_usd=Q10_MAX_LOSS_USD,
                        min_start_equity_usd=Q10_MIN_EQUITY_USD):
    """REAL ORDERS: fixed Q10, fixed 8h, fixed $20 software-loss trigger."""
    if str(arm_phrase) != Q10_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q10_ARM!r} exactly."
        )
    if abs(float(runtime_hours) - Q10_HOURS) > 1e-12:
        raise RuntimeError("Q10 V2.9.0 overnight is fixed to exactly 8.0 hours.")
    if abs(float(max_start_loss_usd) - Q10_MAX_LOSS_USD) > 1e-12:
        raise RuntimeError("Q10 V2.9.0 overnight is fixed to exactly a $20 software loss trigger.")
    if float(min_start_equity_usd) + 1e-12 < Q10_MIN_EQUITY_USD:
        raise RuntimeError("Q10 V2.9.0 requires minimum starting equity of at least $125.")

    static_self_check(show=True)
    V28._patch_parent()
    V28.D._guard_other_live_processes()

    pre = V288.live_preflight(
        quote_size=Q10_Q,
        runtime_hours=Q10_HOURS,
        max_start_loss_usd=Q10_MAX_LOSS_USD,
        min_start_equity_usd=float(min_start_equity_usd),
        show=True,
        probe_private_ws=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    mode = "DEEP_TAIL_Q10_OVERNIGHT_8H_V17_V290"
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}").resolve()
    session.mkdir(parents=True, exist_ok=False)

    group_limit = max(25.0, 20.0 * Q10_Q)
    cfg = {
        "mode": mode,
        "quote_size": Q10_Q,
        "runtime_hours": Q10_HOURS,
        "max_start_loss_usd": Q10_MAX_LOSS_USD,
        "min_start_equity_usd": float(min_start_equity_usd),
        "order_group_limit_fp": f"{group_limit:.2f}",
        "live_engine_version": LIVE.LIVE_VERSION,
        "deploy_version": V282.RUNTIME_BASE_DEPLOY_VERSION,
        "launch_wrapper_version": DEPLOY_VERSION,
        "child_fee_preflight_mode": "REUSE_FRESH_PARENT_PASS_SNAPSHOT_FAIL_CLOSED",
        "recorder_study_version": V5A.STUDY_VERSION,
        "recorder_discovery_transport": V5A.DISCOVERY_TRANSPORT_VERSION,
        "rest_fill_reconcile_mode": "MIN_TS_INCREMENTAL_DEDUP_CURSOR",
        "scientific_status": "Q10_8H_FORWARD_AFTER_STRATEGY_RSS_FAILURE_FIX",
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
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_q10_overnight_v2_9_0",
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
                f"Q10 V2.9.0 process exited during startup rc={p.returncode}\n{tail}"
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
        if (
            state_ok and private_ok and raw_ok and bounded_ok
            and runtime_bound_ok and mem_ok and fee_ok
        ):
            break
        time.sleep(0.25)
    else:
        B._atomic(session / "KILL_REQUEST.json", {
            "time": B._iso(),
            "reason": "STARTUP_HEALTH_TIMEOUT_Q10_V290",
        })
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(
            f"Q10 V2.9.0 startup health timeout. Last health={last}\n{tail}"
        )

    guardian, guardian_log, guardian_cmd = _launch_guardian(session, p.pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update({
        "guardian_pid": guardian.pid,
        "guardian_log_path": str(guardian_log),
        "guardian_command": guardian_cmd,
    })
    B._atomic(CORE.CONTROL_PATH, ctl)

    print("\n" + "=" * 112)
    print("REAL-MONEY DEEP-TAIL Q10 OVERNIGHT V2.9.0 ARMED")
    print("=" * 112)
    print("Session:                    ", session)
    print("Main PID:                   ", p.pid)
    print("Guardian PID:               ", guardian.pid)
    print("Engine:                     ", LIVE.LIVE_VERSION)
    print("Recorder:                   ", V5A.STUDY_VERSION)
    print("Quantity:                   Q10 per eligible market")
    print("Runtime:                    8.00 hours from first complete live window")
    print("Software loss trigger:      -$20.00 from calibrated starting equity")
    print("REST fill fallback:         incremental min_ts + dedupe + cursor")
    print("Bounded raw tail verified:  YES")
    print("Guardian hard RSS limit:    ", f"{V28.RSS_HARD_LIMIT_MB:.0f} MB (UNCHANGED)")
    print("Mac sleep prevention:       ", "caffeinate enabled" if caffeinate else "caffeinate unavailable")
    print("Pre-M0 market-data gate:    NONE")
    print("Auto-scaling:               DISABLED")
    print("=" * 112)
    return live_status(show=False, tail_lines=20)


def live_status(*args, **kwargs):
    return V288.live_status(*args, **kwargs)


def kill_and_flatten_live(*args, **kwargs):
    return V288.kill_and_flatten_live(*args, **kwargs)


def q10_data_audit(session_dir, *, show=True):
    return V288.q1_data_audit(Path(session_dir).resolve(), show=show)


def overnight_audit(session_dir, *, show=True):
    session = Path(session_dir).resolve()
    cfg = B._read(session / "process_config.json", {}) or {}
    final = B._read(session / "final_summary.json", {}) or {}
    guardian = B._read(session / "guardian_receipt_v2_8.json", {}) or {}
    data = q10_data_audit(session, show=False)
    live_audit = V28.AUDIT.audit_live_session(session, show=False, write=True)

    checks = {
        "q10_size": abs(B._f(cfg.get("quote_size"), float("nan")) - Q10_Q) < 1e-9,
        "runtime_config_8h": abs(B._f(cfg.get("runtime_hours"), float("nan")) - Q10_HOURS) < 1e-9,
        "loss_trigger_config_20": abs(B._f(cfg.get("max_start_loss_usd"), float("nan")) - Q10_MAX_LOSS_USD) < 1e-9,
        "wrapper_is_v2_9_0": cfg.get("launch_wrapper_version") == DEPLOY_VERSION,
        "engine_is_v1_7": cfg.get("live_engine_version") == LIVE.LIVE_VERSION,
        "incremental_rest_fill_configured": cfg.get("rest_fill_reconcile_mode") == "MIN_TS_INCREMENTAL_DEDUP_CURSOR",
        "auth_v5_recorder_configured": cfg.get("recorder_study_version") == V5A.STUDY_VERSION,
        "completed": live_audit.get("completed") is True,
        "natural_runtime_complete": final.get("shutdown_reason") == "RUNTIME_COMPLETE",
        "clean_final": live_audit.get("clean_final") is True,
        "flat_verified": live_audit.get("flat_verified") is True,
        "zero_strategy_resting": live_audit.get("strategy_resting_orders_zero") is True,
        "no_operational_fail": live_audit.get("operational_fail") is False,
        "raw_data_audit": data.get("passed") is True,
        "guardian_receipt_present": bool(guardian),
        "guardian_did_not_intervene": guardian.get("intervened") is False,
    }
    passed = all(checks.values())
    out = {
        "session": str(session),
        "passed": bool(passed),
        "checks": checks,
        "final_summary": final,
        "guardian": guardian,
        "live_audit": live_audit,
        "data_audit": data,
        "note": "Operational/data audit only; not evidence expected PnL is positive.",
    }
    if show:
        print("=" * 112)
        print("Q10 V2.9.0 8H OVERNIGHT AUDIT — READ ONLY")
        print("=" * 112)
        for k, v in checks.items():
            print(f"{k:56s}: {v}")
        print("Final PnL:                  ", final.get("account_pnl_usd"))
        print("Shutdown reason:            ", final.get("shutdown_reason"))
        print("Guardian reason:            ", guardian.get("reason"))
        print("AUDIT:", "PASS" if passed else "FAIL / REVIEW")
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    a = ap.parse_args()
    if a.run_live_session:
        _run_child(Path(a.run_live_session), Path(a.config))
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION", "LIVE", "CORE", "Q10_ARM", "KILL_ARM",
    "Q10_Q", "Q10_HOURS", "Q10_MAX_LOSS_USD", "Q10_MIN_EQUITY_USD",
    "static_self_check", "live_preflight", "start_q10_overnight",
    "live_status", "kill_and_flatten_live", "q10_data_audit", "overnight_audit",
]
