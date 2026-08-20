from __future__ import annotations

"""V2.8.2 deployment wrapper: reuse the already-passed parent fee snapshot in the child.

Operational-only change on top of V2.8/V2.8.1 and live V1.5. Alpha/execution rules,
Q1 size, M1/M5 timing, loss limit, guardian RSS/deadline behavior, and promotion
requirements are unchanged.

Why this exists
---------------
The parent launch already runs the resilient V2.7.1 fee preflight.  The detached live
child then entered the legacy B._run_process preflight, which independently called
OOS.fee_preflight again.  That duplicate public-series burst could hit HTTP 429 and
abort before the order group was created.

V2.8.2 does not bypass fee validation.  The child reuses only the fresh parent fee
snapshot that has already passed all frozen-series fee/fee-change checks.  It fails
closed unless the snapshot is PASS, complete for every frozen series, has the exact
fee horizon, and is no more than 300 seconds old.  No child fee API burst is made.

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
from . import mm_deep_tail_join_ask_deploy_v2_8_1 as V281


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_8_2_CHILD_FEE_SNAPSHOT_REUSE"
RUNTIME_BASE_DEPLOY_VERSION = V28.DEPLOY_VERSION
LIVE = V28.LIVE
CORE = V28.CORE
Q1_ARM = "LIVE_DEEP_TAIL_Q1_V15_V282"
KILL_ARM = V28.KILL_ARM
PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2_8_2.json"

Q1_DEFAULT_HOURS = 0.5
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0
PARENT_FEE_MAX_AGE_S = 300.0
STARTUP_TIMEOUT_S = V28.STARTUP_TIMEOUT_S


def _snapshot_age_s(fee):
    t = pd.to_datetime((fee or {}).get("time"), utc=True, errors="coerce")
    if pd.isna(t):
        return float("nan")
    return max(0.0, time.time() - float(t.timestamp()))


def _validated_parent_fee_snapshot(session, *, horizon_hours=OOS.FEE_CHANGE_HORIZON_H,
                                   save_path=None, show=True):
    """Return the fresh parent PASS fee snapshot; never calls the exchange."""
    session = Path(session).resolve()
    parent = B._read(session / "parent_preflight_snapshot.json", {}) or {}
    fee = dict(parent.get("fee_preflight") or {})

    age_s = _snapshot_age_s(fee)
    expected_series = set(str(x) for x in OOS.SERIES)
    multipliers = fee.get("multipliers") or {}
    actual_series = set(str(x) for x in multipliers)

    problems = []
    if fee.get("ok") is not True:
        problems.append("parent fee snapshot is not PASS")

    try:
        snap_horizon = float(fee.get("horizon_hours"))
    except Exception:
        snap_horizon = float("nan")
    if not (snap_horizon == snap_horizon and abs(snap_horizon - float(horizon_hours)) < 1e-9):
        problems.append(
            f"fee horizon mismatch: snapshot={fee.get('horizon_hours')!r}, "
            f"required={float(horizon_hours)}"
        )

    if actual_series != expected_series:
        problems.append(
            f"fee multiplier series mismatch: missing={sorted(expected_series-actual_series)}, "
            f"extra={sorted(actual_series-expected_series)}"
        )

    for series in sorted(expected_series):
        try:
            mult = float(multipliers.get(series))
        except Exception:
            mult = float("nan")
        if not (mult == mult and mult > 0):
            problems.append(f"invalid fee multiplier for {series}: {multipliers.get(series)!r}")

    if not (age_s == age_s and age_s <= PARENT_FEE_MAX_AGE_S):
        problems.append(
            f"parent fee snapshot stale/unparseable: age_s={age_s!r}, "
            f"max={PARENT_FEE_MAX_AGE_S}"
        )

    if problems:
        raise RuntimeError(
            "V2.8.2 child fee preflight refused parent snapshot: " + " | ".join(problems)
        )

    out = dict(fee)
    out.update({
        "child_reused_parent_snapshot": True,
        "child_fee_api_called": False,
        "parent_snapshot_age_s": float(age_s),
        "parent_snapshot_max_age_s": PARENT_FEE_MAX_AGE_S,
        "deploy_wrapper": DEPLOY_VERSION,
    })

    if save_path is not None:
        B._atomic(Path(save_path), out)

    B._atomic(session / "child_fee_preflight_reuse_v2_8_2.json", {
        "time": B._iso(),
        "ok": True,
        "deploy_wrapper": DEPLOY_VERSION,
        "parent_snapshot_age_s": float(age_s),
        "max_age_s": PARENT_FEE_MAX_AGE_S,
        "horizon_hours": float(horizon_hours),
        "series": sorted(expected_series),
        "multipliers": multipliers,
        "child_fee_api_called": False,
        "orders_sent_by_fee_preflight": False,
    })

    if show:
        print(
            "CHILD FEE PREFLIGHT: PASS — reused fresh parent PASS snapshot "
            f"({age_s:.1f}s old); child fee API calls: 0"
        )
    return out


def static_self_check(*, show=True):
    base = V281.static_self_check(show=False)
    checks = {
        "base_static_ok": base.get("ok") is True,
        "alpha_rules_unchanged": True,
        "live_engine_version": LIVE.LIVE_VERSION,
        "bounded_raw_ingestion": base.get("bounded_raw_ingestion") is True,
        "rss_guard_unchanged": base.get("rss_guard_unchanged") is True,
        "deadline_initial_grace_unchanged": base.get("deadline_initial_grace_unchanged") is True,
        "normal_cleanup_grace_s": V28.NORMAL_RUNTIME_CLEANUP_GRACE_S,
        "cleanup_overrun_still_fail_closed": base.get("cleanup_overrun_still_fail_closed") is True,
        "parent_fee_preflight_resilient": base.get("resilient_fee_preflight") is True,
        "child_fee_snapshot_reuse": True,
        "child_fee_snapshot_fail_closed": True,
        "child_fee_api_called": False,
        "parent_fee_max_age_s": PARENT_FEE_MAX_AGE_S,
        "orders_sent": False,
    }
    metadata = {
        "live_engine_version", "normal_cleanup_grace_s", "parent_fee_max_age_s",
        "child_fee_api_called", "orders_sent",
    }
    ok = all(v is True for k, v in checks.items() if k not in metadata)
    out = {"deploy_version": DEPLOY_VERSION, **checks, "ok": bool(ok)}
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOY V2.8.2 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:58s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.8.2 static self-check failed: {out}")
    return out


def live_preflight(*, quote_size, runtime_hours, max_start_loss_usd,
                   min_start_equity_usd, show=True, probe_private_ws=True):
    static_self_check(show=show)
    out = V281.live_preflight(
        quote_size=float(quote_size),
        runtime_hours=float(runtime_hours),
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
        show=show,
        probe_private_ws=probe_private_ws,
    )
    out = dict(out)
    out["launch_wrapper_version"] = DEPLOY_VERSION
    out["child_fee_preflight_mode"] = "REUSE_FRESH_PARENT_PASS_SNAPSHOT_FAIL_CLOSED"
    return out


def _launch_guardian(session, main_pid):
    log = Path(session) / "guardian_v2_8.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_8_2",
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
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly."
        )

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
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}_deploy_v2_8_2").resolve()
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
        # Keep the base field compatible with the V2.8 audit; record V2.8.2 separately.
        "deploy_version": RUNTIME_BASE_DEPLOY_VERSION,
        "launch_wrapper_version": DEPLOY_VERSION,
        "child_fee_preflight_mode": "REUSE_FRESH_PARENT_PASS_SNAPSHOT_FAIL_CLOSED",
        "scientific_status": "FRESH_FORWARD_AFTER_V2_8_2_CHILD_FEE_PREFLIGHT_FIX",
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
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_8_2",
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
            raise RuntimeError(f"Live V2.8.2 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        state_ok = last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}
        private_ok = last.get("private_ws_ready") is True
        raw_ok = last.get("raw_watchdog_ready") is True
        bounded_ok = last.get("bounded_raw_ingestion") is True
        fee_reuse_ok = (B._read(session / "child_fee_preflight_reuse_v2_8_2.json", {}) or {}).get("ok") is True
        if state_ok and private_ok and raw_ok and bounded_ok and fee_reuse_ok:
            break
        time.sleep(0.25)
    else:
        B._atomic(session / "KILL_REQUEST.json", {
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V2_8_2"
        })
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(f"V2.8.2 startup health timeout. Last health={last}\n{tail}")

    guardian, guardian_log, guardian_cmd = _launch_guardian(session, p.pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update({
        "guardian_pid": guardian.pid,
        "guardian_log_path": str(guardian_log),
        "guardian_command": guardian_cmd,
    })
    B._atomic(CORE.CONTROL_PATH, ctl)

    print("\n" + "=" * 100)
    print("REAL-MONEY DEEP-TAIL V2.8.2 PROCESS ARMED")
    print("=" * 100)
    print("Session:                  ", session)
    print("Main PID:                 ", p.pid)
    print("Guardian PID:             ", guardian.pid)
    print("Engine:                   ", LIVE.LIVE_VERSION)
    print("Quantity:                 ", f"Q{float(q):g}")
    print("Runtime:                  ", f"{float(hours):.2f}h from first complete window")
    print("Software loss trigger:    ", f"-${float(max_loss):.2f}")
    print("Child fee API burst:       DISABLED — fresh parent PASS snapshot reused")
    print("RSS hard guardian:         ", f"{V28.RSS_HARD_LIMIT_MB:.0f} MB")
    print("Initial deadline grace:    ", f"{V28.GUARDIAN_DEADLINE_GRACE_S:.1f}s")
    print("Normal cleanup grace:      ", f"{V28.NORMAL_RUNTIME_CLEANUP_GRACE_S:.1f}s")
    print("Auto-scaling:              DISABLED")
    print("=" * 100)
    return live_status(show=False)


def start_q1_smoke(*, arm_phrase=None, runtime_hours=Q1_DEFAULT_HOURS,
                   max_start_loss_usd=Q1_DEFAULT_MAX_LOSS,
                   min_start_equity_usd=Q1_DEFAULT_MIN_EQUITY):
    if abs(float(runtime_hours) - Q1_DEFAULT_HOURS) > 1e-12:
        raise RuntimeError("V2.8.2 Q1 smoke is fixed to exactly 0.5 hours.")
    return _launch(
        q=1.0,
        hours=Q1_DEFAULT_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q1_SMOKE_V15_V282",
        arm_phrase=arm_phrase,
        expected_arm=Q1_ARM,
    )


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    session = Path(session_dir).resolve()

    base = V28.q1_promotion_check(session, show=False, write_receipt=False)
    cfg = B._read(session / "process_config.json", {}) or {}
    child_fee = B._read(session / "child_fee_preflight_reuse_v2_8_2.json", {}) or {}

    checks = dict(base.get("checks") or {})
    checks.update({
        "launch_wrapper_is_v2_8_2": cfg.get("launch_wrapper_version") == DEPLOY_VERSION,
        "child_fee_snapshot_receipt_present": bool(child_fee),
        "child_fee_snapshot_passed": child_fee.get("ok") is True,
        "child_fee_api_not_called": child_fee.get("child_fee_api_called") is False,
        "child_fee_series_complete": set(child_fee.get("series") or []) == set(OOS.SERIES),
    })
    passed = all(checks.values())

    receipt = dict(base)
    receipt.update({
        "time": B._iso(),
        "deploy_version": DEPLOY_VERSION,
        "passed": bool(passed),
        "checks": checks,
        "child_fee_preflight": child_fee,
        "note": "Operational promotion only; not evidence that expected PnL is positive.",
    })
    if passed and write_receipt:
        B._atomic(PROMOTION_PATH, receipt)

    if show:
        print("=" * 100)
        print("Q1 V1.5 / V2.8.2 OPERATIONAL PROMOTION — READ ONLY")
        print("=" * 100)
        for k, v in checks.items():
            print(f"{k:48s}: {v}")
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

    old = OOS.fee_preflight

    def child_fee_preflight(*, horizon_hours=OOS.FEE_CHANGE_HORIZON_H,
                            save_path=None, show=True):
        return _validated_parent_fee_snapshot(
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
    "PARENT_FEE_MAX_AGE_S", "_validated_parent_fee_snapshot", "static_self_check",
    "live_preflight", "start_q1_smoke", "q1_promotion_check", "live_status",
    "kill_and_flatten_live",
]
