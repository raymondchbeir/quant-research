from __future__ import annotations

"""V2.8.3 deployment: strict fail-closed live market discovery.

Operational-only on top of V2.8.2 / live V1.5. Strategy/alpha rules are unchanged.

Root bug fixed:
- legacy V5 discovery swallowed /markets exceptions and could return {};
- supervisor then set last_scan_error=None and health could say healthy=True;
- live startup therefore accepted a recorder with zero subscriptions;
- trade_start/deadline are created only after the first complete-window book row,
  so the process could remain ARMED_WAITING_FULL_WINDOW indefinitely.

V2.8.3 launches V5.1, which retries/paces discovery and raises on any unresolved
series/status discovery failure or zero raw markets for a frozen series. The live
launcher will not pass recorder startup until that strict recorder reports healthy.
The V2.8.2 fresh parent fee-snapshot reuse and V2.8 cleanup-aware guardian remain.

Importing this module sends no orders.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_live_v1 as DTLIVE
from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_8_2 as V282
from . import mm_event_time_m0_m5_recorder_v5_1 as V51


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_8_3_STRICT_DISCOVERY"
RUNTIME_BASE_DEPLOY_VERSION = V28.DEPLOY_VERSION
LIVE = V28.LIVE
CORE = V28.CORE
Q1_ARM = "LIVE_DEEP_TAIL_Q1_V15_V283"
KILL_ARM = V28.KILL_ARM
PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2_8_3.json"

Q1_DEFAULT_HOURS = 0.5
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0
STARTUP_TIMEOUT_S = 180.0


def static_self_check(*, show=True):
    base = V282.static_self_check(show=False)
    rec = V51.static_self_check(show=False)
    checks = {
        "base_static_ok": base.get("ok") is True,
        "alpha_rules_unchanged": True,
        "live_engine_version": LIVE.LIVE_VERSION,
        "bounded_raw_ingestion": base.get("bounded_raw_ingestion") is True,
        "rss_guard_unchanged": base.get("rss_guard_unchanged") is True,
        "deadline_guard_unchanged": base.get("deadline_initial_grace_unchanged") is True,
        "cleanup_overrun_still_fail_closed": base.get("cleanup_overrun_still_fail_closed") is True,
        "child_fee_snapshot_reuse": base.get("child_fee_snapshot_reuse") is True,
        "strict_recorder_static_ok": rec.get("ok") is True,
        "strict_discovery_fail_closed": rec.get("discovery_fail_closed") is True,
        "capture_window_unchanged": rec.get("base_capture_window_unchanged") is True,
        "universe_unchanged": rec.get("universe_unchanged") is True,
        "orders_sent": False,
    }
    metadata = {"live_engine_version", "orders_sent"}
    ok = all(v is True for k, v in checks.items() if k not in metadata)
    out = {"deploy_version": DEPLOY_VERSION, **checks, "ok": bool(ok)}
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOY V2.8.3 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.8.3 static check failed: {out}")
    return out


def live_preflight(*, quote_size, runtime_hours, max_start_loss_usd,
                   min_start_equity_usd, show=True, probe_private_ws=True):
    static_self_check(show=show)
    return V282.live_preflight(
        quote_size=float(quote_size),
        runtime_hours=float(runtime_hours),
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
        show=show,
        probe_private_ws=probe_private_ws,
    )


def _start_strict_recorder(session):
    session = Path(session).resolve()
    raw = session / "raw_capture"
    if raw.exists():
        raise RuntimeError(f"strict recorder requires fresh raw_capture path: {raw}")

    log = session / "raw_recorder.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable, "-m", "quant_research.kalshi.mm_event_time_m0_m5_recorder_v5_1",
        "--run-session", str(raw),
    ]
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

    deadline = time.time() + STARTUP_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace")[-12000:] if log.exists() else ""
            raise RuntimeError(f"V5.1 strict recorder startup failure rc={p.returncode}\n{tail}")
        last = B._read(raw / "health.json", {}) or {}
        if last.get("running") and last.get("healthy") and last.get("last_scan_error") is None:
            return p, last
        time.sleep(0.35)

    try:
        os.kill(p.pid, signal.SIGTERM)
    except Exception:
        pass
    tail = log.read_text(encoding="utf-8", errors="replace")[-12000:] if log.exists() else ""
    raise RuntimeError(f"V5.1 strict recorder health timeout. Last={last}\n{tail}")


def _launch_guardian(session, main_pid):
    log = Path(session) / "guardian_v2_8.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_8_3",
        "--run-guardian", str(Path(session).resolve()),
        "--main-pid", str(int(main_pid)),
    ]
    try:
        g = subprocess.Popen(
            cmd,
            cwd=str(V28.C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()
    return g, log, cmd


def _launch(*, q, hours, max_loss, min_equity, mode, arm_phrase, expected_arm):
    if str(arm_phrase) != str(expected_arm):
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly.")

    V28._patch_parent()
    V28.D._guard_other_live_processes()
    pre = live_preflight(
        quote_size=q,
        runtime_hours=hours,
        max_start_loss_usd=max_loss,
        min_start_equity_usd=min_equity,
        show=True,
        probe_private_ws=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}_deploy_v2_8_3").resolve()
    session.mkdir(parents=True, exist_ok=False)

    group_limit = max(25.0, 20.0 * float(q))
    cfg = {
        "mode": str(mode),
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "order_group_limit_fp": f"{group_limit:.2f}",
        "live_engine_version": LIVE.LIVE_VERSION,
        "deploy_version": RUNTIME_BASE_DEPLOY_VERSION,
        "launch_wrapper_version": DEPLOY_VERSION,
        "child_fee_preflight_mode": "REUSE_FRESH_PARENT_PASS_SNAPSHOT_FAIL_CLOSED",
        "raw_recorder_version": V51.STUDY_VERSION,
        "scientific_status": "FRESH_FORWARD_AFTER_V2_8_3_STRICT_DISCOVERY_FIX",
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
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_8_3",
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
        "runtime_base_deploy_version": RUNTIME_BASE_DEPLOY_VERSION,
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
            raise RuntimeError(f"Live V2.8.3 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        rh = last.get("recorder_health") or {}
        state_ok = last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}
        private_ok = last.get("private_ws_ready") is True
        raw_ok = last.get("raw_watchdog_ready") is True
        bounded_ok = last.get("bounded_raw_ingestion") is True
        fee_ok = (B._read(session / "child_fee_preflight_reuse_v2_8_2.json", {}) or {}).get("ok") is True
        strict_rec_ok = (
            rh.get("healthy") is True
            and rh.get("last_scan_error") is None
            and rh.get("study_version") == V51.STUDY_VERSION
        )
        if state_ok and private_ok and raw_ok and bounded_ok and fee_ok and strict_rec_ok:
            break
        time.sleep(0.25)
    else:
        B._atomic(session / "KILL_REQUEST.json", {"time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V2_8_3"})
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(f"V2.8.3 startup health timeout. Last={last}\n{tail}")

    guardian, guardian_log, guardian_cmd = _launch_guardian(session, p.pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update({
        "guardian_pid": guardian.pid,
        "guardian_log_path": str(guardian_log),
        "guardian_command": guardian_cmd,
    })
    B._atomic(CORE.CONTROL_PATH, ctl)

    print("\n" + "=" * 100)
    print("REAL-MONEY DEEP-TAIL V2.8.3 PROCESS ARMED")
    print("=" * 100)
    print("Session:                  ", session)
    print("Main PID:                 ", p.pid)
    print("Guardian PID:             ", guardian.pid)
    print("Quantity:                 ", f"Q{float(q):g}")
    print("Runtime:                  ", f"{float(hours):.2f}h from first complete window")
    print("Strict discovery recorder:", V51.STUDY_VERSION)
    print("Child duplicate fee burst: DISABLED")
    print("Software loss trigger:    ", f"-${float(max_loss):.2f}")
    print("Auto-scaling:              DISABLED")
    print("=" * 100)
    return live_status(show=False)


def start_q1_smoke(*, arm_phrase=None, runtime_hours=Q1_DEFAULT_HOURS,
                   max_start_loss_usd=Q1_DEFAULT_MAX_LOSS,
                   min_start_equity_usd=Q1_DEFAULT_MIN_EQUITY):
    if abs(float(runtime_hours) - Q1_DEFAULT_HOURS) > 1e-12:
        raise RuntimeError("V2.8.3 Q1 smoke is fixed to exactly 0.5 hours.")
    return _launch(
        q=1.0,
        hours=Q1_DEFAULT_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q1_SMOKE_V15_V283",
        arm_phrase=arm_phrase,
        expected_arm=Q1_ARM,
    )


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    session = Path(session_dir).resolve()
    base = V28.q1_promotion_check(session, show=False, write_receipt=False)
    cfg = B._read(session / "process_config.json", {}) or {}
    child_fee = B._read(session / "child_fee_preflight_reuse_v2_8_2.json", {}) or {}
    final_rec = B._read(session / "raw_capture" / "session_manifest.json", {}) or {}

    checks = dict(base.get("checks") or {})
    checks.update({
        "launch_wrapper_is_v2_8_3": cfg.get("launch_wrapper_version") == DEPLOY_VERSION,
        "strict_recorder_configured": cfg.get("raw_recorder_version") == V51.STUDY_VERSION,
        "strict_recorder_manifest": final_rec.get("study_version") == V51.STUDY_VERSION,
        "child_fee_snapshot_passed": child_fee.get("ok") is True,
        "child_fee_api_not_called": child_fee.get("child_fee_api_called") is False,
    })
    passed = all(checks.values())
    receipt = dict(base)
    receipt.update({
        "time": B._iso(),
        "deploy_version": DEPLOY_VERSION,
        "passed": bool(passed),
        "checks": checks,
        "child_fee_preflight": child_fee,
        "strict_recorder_manifest": final_rec,
    })
    if passed and write_receipt:
        B._atomic(PROMOTION_PATH, receipt)

    if show:
        print("=" * 100)
        print("Q1 V1.5 / V2.8.3 OPERATIONAL PROMOTION — READ ONLY")
        print("=" * 100)
        for k, v in checks.items():
            print(f"{k:50s}: {v}")
        print("PROMOTION:", "PASS" if passed else "NOT READY")
        if passed:
            print("Receipt:", PROMOTION_PATH)
    return receipt


def live_status(*args, **kwargs):
    return V28.live_status(*args, **kwargs)


def kill_and_flatten_live(*args, **kwargs):
    return V28.kill_and_flatten_live(*args, **kwargs)


def _run_child(session, cfg_path):
    session = Path(session).resolve()
    cfg = B._read(Path(cfg_path), {}) or {}

    old_fee = OOS.fee_preflight
    old_start = DTLIVE.V4._start_recorder_fixed

    def child_fee_preflight(*, horizon_hours=OOS.FEE_CHANGE_HORIZON_H,
                            save_path=None, show=True):
        return V282._validated_parent_fee_snapshot(
            session,
            horizon_hours=horizon_hours,
            save_path=save_path,
            show=show,
        )

    OOS.fee_preflight = child_fee_preflight
    DTLIVE.V4._start_recorder_fixed = _start_strict_recorder
    try:
        LIVE.run_live_process(session, cfg)
    finally:
        OOS.fee_preflight = old_fee
        DTLIVE.V4._start_recorder_fixed = old_start


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    ap.add_argument("--run-guardian")
    ap.add_argument("--main-pid", type=int)
    a = ap.parse_args()
    if a.run_guardian:
        V28._guardian_loop(Path(a.run_guardian).resolve(), int(a.main_pid))
    elif a.run_live_session:
        _run_child(Path(a.run_live_session).resolve(), Path(a.config).resolve())
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION", "LIVE", "CORE", "Q1_ARM", "KILL_ARM", "PROMOTION_PATH",
    "static_self_check", "live_preflight", "start_q1_smoke", "q1_promotion_check",
    "live_status", "kill_and_flatten_live",
]
