from __future__ import annotations

"""Q10 8-hour overnight live runner on the validated V2.8.8 / V1.6 stack.

This is the economically meaningful next stage after the Q1 smoke.
It changes sizing/runtime/risk budget only:
- quantity: Q10 per eligible market;
- runtime: exactly 8.0 hours from the first complete live window;
- software start-to-current loss trigger: exactly $20;
- minimum starting equity: $125;
- no auto-scaling.

Strategy/execution rules are unchanged from the validated deep-tail stack:
- at M1, rest BUY YES 5c + BUY NO 5c;
- first tail fill selects the tail and cancels the opposite entry;
- selected entry may continue to full Q;
- after full Q entry, post one fixed passive JOIN_ASK reduce-only exit;
- no exit repricing/chasing;
- M5 cancels residual strategy orders and reduce-only IOC flattens inventory;
- V1.5 bounded raw ingestion, V2.8 guardian, V2.8.2 parent fee snapshot reuse,
  private execution websocket and account audit remain unchanged;
- raw public data uses the authenticated-discovery V5 recorder introduced in V2.8.8.

The $20 loss trigger is a software stop, not a guaranteed final-loss cap. In-flight
fills, latency, market movement and liquidation slippage can cause overshoot.

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
from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_8_2 as V282
from . import mm_deep_tail_join_ask_deploy_v2_8_7 as V287
from . import mm_deep_tail_join_ask_deploy_v2_8_8 as V288
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_6 as LIVE
from . import mm_event_time_m0_m5_recorder_v5_auth as V5A

DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q10_OVERNIGHT_V2_8_9_AUTH_V5"
CORE = V282.CORE
Q10_ARM = "LIVE_DEEP_TAIL_Q10_8H_V289"
KILL_ARM = V282.KILL_ARM

Q10_Q = 10.0
Q10_HOURS = 8.0
Q10_MAX_LOSS_USD = 20.0
Q10_MIN_EQUITY_USD = 125.0
STARTUP_TIMEOUT_S = V282.STARTUP_TIMEOUT_S


def static_self_check(*, show=True):
    pp = V287._install_child_pythonpath()
    base = V288.static_self_check(show=False)
    live = LIVE.static_self_check(show=False)
    checks = {
        "validated_v2_8_8_base": base.get("ok") is True,
        "live_engine_v1_6": str(LIVE.LIVE_VERSION) == "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_6_AUTH_V5_DISCOVERY",
        "authenticated_v5_discovery": base.get("authenticated_v5_discovery") is True,
        "bounded_raw_ingestion": live.get("bounded_raw_ingestion") is True,
        "alpha_rules_unchanged": live.get("alpha_rules_unchanged_from_v1_5") is True,
        "q10_on_allowed_ladder": int(Q10_Q) in tuple(int(x) for x in V1.LADDER_Q),
        "runtime_fixed_8h": abs(Q10_HOURS - 8.0) < 1e-12,
        "loss_trigger_fixed_20": abs(Q10_MAX_LOSS_USD - 20.0) < 1e-12,
        "minimum_equity_125": abs(Q10_MIN_EQUITY_USD - 125.0) < 1e-12,
        "guardian_unchanged": True,
        "fee_snapshot_reuse": base.get("fee_snapshot_reuse") is True,
        "strict_pre_m0_data_gate": False,
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
        print("=" * 108)
        print("DEEP-TAIL Q10 V2.8.9 OVERNIGHT STATIC CHECK — NO API / NO ORDERS")
        print("=" * 108)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    if not ok:
        raise RuntimeError(f"Q10 V2.8.9 static self-check failed: {out}")
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
    # Reuse the exact validated V2.8.8 child logic: parent fee snapshot reuse plus
    # V1.6 authenticated-V5 live runtime.
    return V288._run_child(Path(session).resolve(), Path(cfg_path).resolve())


def _launch_guardian(session, main_pid):
    return V288._launch_guardian(Path(session).resolve(), int(main_pid))


def start_q10_overnight(*, arm_phrase=None, runtime_hours=Q10_HOURS,
                        max_start_loss_usd=Q10_MAX_LOSS_USD,
                        min_start_equity_usd=Q10_MIN_EQUITY_USD):
    """REAL ORDERS: fixed Q10, fixed 8h runtime, fixed $20 software loss trigger."""
    if str(arm_phrase) != Q10_ARM:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q10_ARM!r} exactly.")
    if abs(float(runtime_hours) - Q10_HOURS) > 1e-12:
        raise RuntimeError("Q10 overnight is fixed to exactly 8.0 hours.")
    if abs(float(max_start_loss_usd) - Q10_MAX_LOSS_USD) > 1e-12:
        raise RuntimeError("Q10 overnight is fixed to exactly a $20 software loss trigger.")
    if float(min_start_equity_usd) + 1e-12 < Q10_MIN_EQUITY_USD:
        raise RuntimeError("Q10 overnight requires minimum starting equity of at least $125.")

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
    mode = "DEEP_TAIL_Q10_OVERNIGHT_8H_V16_V289"
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
        "validated_q1_base_wrapper": V288.DEPLOY_VERSION,
        "child_fee_preflight_mode": "REUSE_FRESH_PARENT_PASS_SNAPSHOT_FAIL_CLOSED",
        "recorder_study_version": V5A.STUDY_VERSION,
        "recorder_discovery_transport": V5A.DISCOVERY_TRANSPORT_VERSION,
        "scientific_status": "Q10_8H_FORWARD_VALIDATION_AFTER_Q1_AUTH_V5_SMOKE",
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
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_q10_overnight_v2_8_9",
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
            raise RuntimeError(f"Q10 overnight process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        state_ok = last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}
        private_ok = last.get("private_ws_ready") is True
        raw_ok = last.get("raw_watchdog_ready") is True
        bounded_ok = last.get("bounded_raw_ingestion") is True
        fee_ok = (B._read(session / "child_fee_preflight_reuse_v2_8_2.json", {}) or {}).get("ok") is True
        if state_ok and private_ok and raw_ok and bounded_ok and fee_ok:
            break
        time.sleep(0.25)
    else:
        B._atomic(session / "KILL_REQUEST.json", {
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_Q10_V289"
        })
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(f"Q10 overnight startup health timeout. Last health={last}\n{tail}")

    guardian, guardian_log, guardian_cmd = _launch_guardian(session, p.pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update({
        "guardian_pid": guardian.pid,
        "guardian_log_path": str(guardian_log),
        "guardian_command": guardian_cmd,
    })
    B._atomic(CORE.CONTROL_PATH, ctl)

    print("\n" + "=" * 108)
    print("REAL-MONEY DEEP-TAIL Q10 OVERNIGHT V2.8.9 ARMED")
    print("=" * 108)
    print("Session:                   ", session)
    print("Main PID:                  ", p.pid)
    print("Guardian PID:              ", guardian.pid)
    print("Engine:                    ", LIVE.LIVE_VERSION)
    print("Recorder:                  ", V5A.STUDY_VERSION)
    print("Discovery:                 authenticated signed /markets GET")
    print("Quantity:                  Q10 per eligible market")
    print("Runtime:                   8.00 hours from first complete live window")
    print("Software loss trigger:     -$20.00 from calibrated starting equity")
    print("Minimum starting equity:   ", f"${float(min_start_equity_usd):.2f}")
    print("Mac sleep prevention:      ", "caffeinate enabled" if caffeinate else "caffeinate unavailable")
    print("Pre-M0 market-data gate:   NONE")
    print("Auto-scaling:              DISABLED")
    print("IMPORTANT: $20 is a software trigger, not a guaranteed final-loss cap.")
    print("=" * 108)
    return live_status(show=False, tail_lines=20)


def live_status(*args, **kwargs):
    return V288.live_status(*args, **kwargs)


def kill_and_flatten_live(*args, **kwargs):
    return V288.kill_and_flatten_live(*args, **kwargs)


def q10_data_audit(session_dir, *, show=True):
    return V288.q1_data_audit(Path(session_dir).resolve(), show=show)


def overnight_audit(session_dir, *, show=True):
    """Read-only finished-run operational/data audit."""
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
        "wrapper_is_v2_8_9": cfg.get("launch_wrapper_version") == DEPLOY_VERSION,
        "engine_is_v1_6": cfg.get("live_engine_version") == LIVE.LIVE_VERSION,
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
    }
    if show:
        print("=" * 108)
        print("Q10 8H OVERNIGHT AUDIT — READ ONLY")
        print("=" * 108)
        for k, v in checks.items():
            print(f"{k:52s}: {v}")
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
        _run_child(Path(a.run_live_session).resolve(), Path(a.config).resolve())
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION", "Q10_ARM", "KILL_ARM", "Q10_Q", "Q10_HOURS",
    "Q10_MAX_LOSS_USD", "Q10_MIN_EQUITY_USD", "static_self_check",
    "live_preflight", "start_q10_overnight", "live_status", "kill_and_flatten_live",
    "q10_data_audit", "overnight_audit",
]
