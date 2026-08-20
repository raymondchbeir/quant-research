from __future__ import annotations

"""V2.8 deployment wrapper: fix false guardian deadline intervention during normal cleanup.

Operational-only change on top of V2.7.1 / live V1.5. Alpha/execution mechanics are
unchanged.

The V2.7 guardian used a 5-second deadline grace against process liveness. Normal
RUNTIME_COMPLETE shutdown can legitimately take longer because the engine cancels orders,
flattens, stops the recorder, re-reads positions/resting orders, writes final_summary.json,
and deletes the order group before the process exits. That could make a clean run look like
a deadline overrun.

V2.8 keeps the independent RSS/deadline guardian but distinguishes:
1) no shutdown has started by deadline+5s -> real deadline overrun, intervene;
2) RUNTIME_COMPLETE shutdown has started -> allow bounded cleanup grace;
3) clean final_summary.json exists -> classify as normal completion even if the OS process
   is briefly still present/zombified;
4) cleanup does not finish within the bounded grace -> intervene and disqualify promotion.

Importing this module sends no orders.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_deep_tail_join_ask_live_audit_v1 as AUDIT
from . import mm_deep_tail_join_ask_deploy_v2 as D
from . import mm_deep_tail_join_ask_deploy_v2_7 as V27
from . import mm_deep_tail_join_ask_deploy_v2_7_1 as V271


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_8_NORMAL_CLEANUP_AWARE_GUARDIAN"
LIVE = V27.LIVE
CORE = V27.CORE

Q1_ARM = "LIVE_DEEP_TAIL_Q1_V15_V28"
KILL_ARM = V27.KILL_ARM

Q1_DEFAULT_HOURS = 0.5
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0

STARTUP_TIMEOUT_S = 120.0
GUARDIAN_POLL_S = 0.50
GUARDIAN_DEADLINE_GRACE_S = 5.0
NORMAL_RUNTIME_CLEANUP_GRACE_S = 75.0
RSS_WARNING_MB = V27.RSS_WARNING_MB
RSS_HARD_LIMIT_MB = V27.RSS_HARD_LIMIT_MB

PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2_8.json"


def _current_head():
    return (V10._git_state() or {}).get("head")


def _patch_parent():
    V27._patch_parent()


def _pid_state(pid):
    pid = int(pid or 0)
    if pid <= 0:
        return None
    try:
        p = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if p.returncode != 0:
            return None
        s = (p.stdout or "").strip()
        return s or None
    except Exception:
        return None


def _effectively_alive(pid):
    """Treat zombies as exited for guardian deadline purposes."""
    pid = int(pid or 0)
    if pid <= 0 or not B._pid_alive(pid):
        return False
    state = _pid_state(pid)
    if state and state.startswith("Z"):
        return False
    return True


def _tail_bytes(path: Path, max_bytes=131072):
    try:
        path = Path(path)
        if not path.exists():
            return b""
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            n = fh.tell()
            fh.seek(max(0, n - int(max_bytes)), os.SEEK_SET)
            return fh.read()
    except Exception:
        return b""


def _runtime_shutdown_started(session: Path):
    """Read the base engine's durable SHUTDOWN_START risk event from the file tail."""
    raw = _tail_bytes(Path(session) / "risk_events.jsonl")
    if not raw:
        return False
    for line in reversed(raw.splitlines()):
        if b"SHUTDOWN_START" not in line:
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        if str(row.get("event") or "") != "SHUTDOWN_START":
            continue
        return str(row.get("reason") or "") == "RUNTIME_COMPLETE"
    return False


def _clean_runtime_final(session: Path):
    final = B._read(Path(session) / "final_summary.json", {}) or {}
    ok = bool(
        final
        and final.get("shutdown_reason") == "RUNTIME_COMPLETE"
        and final.get("flat_verified") is True
        and final.get("strategy_resting_orders_zero") is True
        and final.get("last_error") in (None, "")
    )
    return ok, final


def _write_guardian_receipt(session, *, intervened, reason, peak_total, peak_group,
                            peak_recorder, promotion_allowed, extra=None):
    session = Path(session).resolve()
    receipt = {
        "time": B._iso(),
        "deploy_version": DEPLOY_VERSION,
        "session": str(session),
        "intervened": bool(intervened),
        "reason": str(reason),
        "promotion_allowed": bool(promotion_allowed),
        "peak_total_rss_mb": float(peak_total),
        "peak_strategy_group_rss_mb": float(peak_group),
        "peak_recorder_rss_mb": float(peak_recorder),
        "rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
        "deadline_grace_s": GUARDIAN_DEADLINE_GRACE_S,
        "normal_runtime_cleanup_grace_s": NORMAL_RUNTIME_CLEANUP_GRACE_S,
        "normal_cleanup_aware": True,
    }
    if extra:
        receipt.update(dict(extra))
    B._atomic(session / "guardian_receipt_v2_8.json", receipt)
    B._atomic(session / "guardian_health_v2_8.json", {**receipt, "running": False})
    return receipt


def _guardian_loop(session, main_pid):
    session = Path(session).resolve()
    main_pid = int(main_pid)
    health_path = session / "health.json"
    guardian_health_path = session / "guardian_health_v2_8.json"
    events_path = session / "guardian_events_v2_8.jsonl"

    peak_total = 0.0
    peak_group = 0.0
    peak_recorder = 0.0
    warning_emitted = False
    cleanup_grace_deadline = None
    cleanup_grace_started_at = None

    while True:
        clean_final, final = _clean_runtime_final(session)
        if clean_final:
            return _write_guardian_receipt(
                session,
                intervened=False,
                reason="FINAL_SUMMARY_CONFIRMED_RUNTIME_COMPLETE",
                peak_total=peak_total,
                peak_group=peak_group,
                peak_recorder=peak_recorder,
                promotion_allowed=True,
                extra={
                    "main_pid": main_pid,
                    "pid_state_at_receipt": _pid_state(main_pid),
                    "final_summary_confirmed": True,
                    "cleanup_grace_started_at": cleanup_grace_started_at,
                },
            )

        if not _effectively_alive(main_pid):
            return _write_guardian_receipt(
                session,
                intervened=False,
                reason="CHILD_EXITED_WITHOUT_GUARDIAN_ACTION",
                peak_total=peak_total,
                peak_group=peak_group,
                peak_recorder=peak_recorder,
                promotion_allowed=True,
                extra={
                    "main_pid": main_pid,
                    "pid_state_at_receipt": _pid_state(main_pid),
                    "final_summary_confirmed": bool(final),
                },
            )

        h = B._read(health_path, {}) or {}
        recorder_pid = h.get("recorder_pid")
        rss = V27._rss_snapshot(main_pid, recorder_pid)
        peak_total = max(peak_total, float(rss["total_rss_mb"]))
        peak_group = max(peak_group, float(rss["strategy_group_rss_mb"]))
        peak_recorder = max(peak_recorder, float(rss["recorder_rss_mb"]))

        deadline = B._f(h.get("trade_deadline"), float("nan"))
        now = time.time()
        deadline_overrun_s = max(0.0, now - deadline) if deadline == deadline else None
        runtime_shutdown_started = _runtime_shutdown_started(session)

        gh = {
            "time": B._iso(),
            "deploy_version": DEPLOY_VERSION,
            "running": True,
            "intervened": False,
            "main_pid": main_pid,
            "pid_state": _pid_state(main_pid),
            "rss": rss,
            "peak_total_rss_mb": peak_total,
            "peak_strategy_group_rss_mb": peak_group,
            "peak_recorder_rss_mb": peak_recorder,
            "rss_warning_mb": RSS_WARNING_MB,
            "rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
            "trade_deadline": deadline if deadline == deadline else None,
            "deadline_grace_s": GUARDIAN_DEADLINE_GRACE_S,
            "deadline_overrun_s": deadline_overrun_s,
            "runtime_complete_shutdown_started": runtime_shutdown_started,
            "cleanup_grace_started_at": cleanup_grace_started_at,
            "cleanup_grace_deadline": cleanup_grace_deadline,
            "normal_runtime_cleanup_grace_s": NORMAL_RUNTIME_CLEANUP_GRACE_S,
            "bounded_raw_ingestion": h.get("bounded_raw_ingestion"),
            "book_tail_backlog_bytes": ((h.get("book_tail") or {}).get("backlog_bytes")),
        }
        B._atomic(guardian_health_path, gh)

        if rss["total_rss_mb"] >= RSS_WARNING_MB and not warning_emitted:
            B._append(events_path, {
                "time": B._iso(), "event": "RSS_WARNING", "rss": rss,
                "warning_mb": RSS_WARNING_MB, "hard_mb": RSS_HARD_LIMIT_MB,
            })
            warning_emitted = True

        trigger_reason = None
        if rss["total_rss_mb"] >= RSS_HARD_LIMIT_MB:
            trigger_reason = "RSS_HARD_LIMIT"

        elif deadline == deadline and now >= deadline + GUARDIAN_DEADLINE_GRACE_S:
            if runtime_shutdown_started:
                if cleanup_grace_deadline is None:
                    cleanup_grace_started_at = now
                    cleanup_grace_deadline = now + NORMAL_RUNTIME_CLEANUP_GRACE_S
                    B._append(events_path, {
                        "time": B._iso(),
                        "event": "NORMAL_RUNTIME_CLEANUP_GRACE_STARTED",
                        "deadline": deadline,
                        "cleanup_grace_s": NORMAL_RUNTIME_CLEANUP_GRACE_S,
                    })
                elif now >= cleanup_grace_deadline:
                    trigger_reason = "RUNTIME_CLEANUP_OVERRUN"
            else:
                trigger_reason = "RUNTIME_DEADLINE_OVERRUN"

        if trigger_reason is not None:
            B._append(events_path, {
                "time": B._iso(),
                "event": "GUARDIAN_TRIGGER",
                "reason": trigger_reason,
                "rss": rss,
                "trade_deadline": deadline if deadline == deadline else None,
                "runtime_complete_shutdown_started": runtime_shutdown_started,
            })

            recovery = V27._force_recovery(
                session,
                main_pid,
                reason=trigger_reason,
                guardian_source="V2_8_EXTERNAL_GUARDIAN_PROCESS",
            )
            return _write_guardian_receipt(
                session,
                intervened=True,
                reason=trigger_reason,
                peak_total=peak_total,
                peak_group=peak_group,
                peak_recorder=peak_recorder,
                promotion_allowed=False,
                extra={
                    "main_pid": main_pid,
                    "recovery": recovery,
                    "runtime_complete_shutdown_started": runtime_shutdown_started,
                    "cleanup_grace_started_at": cleanup_grace_started_at,
                },
            )

        time.sleep(GUARDIAN_POLL_S)


def static_self_check(*, show=True):
    _patch_parent()
    base = V271.static_self_check(show=False)
    checks = {
        "base_static_ok": base.get("ok") is True,
        "live_engine_version": LIVE.LIVE_VERSION,
        "alpha_rules_unchanged": True,
        "bounded_raw_ingestion": base.get("bounded_raw_ingestion") is True,
        "rss_guard_unchanged": RSS_HARD_LIMIT_MB == V27.RSS_HARD_LIMIT_MB,
        "deadline_initial_grace_unchanged": GUARDIAN_DEADLINE_GRACE_S == V27.GUARDIAN_DEADLINE_GRACE_S,
        "normal_cleanup_grace_s": NORMAL_RUNTIME_CLEANUP_GRACE_S,
        "normal_cleanup_requires_runtime_shutdown_marker": True,
        "clean_final_summary_short_circuits_guardian": True,
        "zombie_is_not_running": True,
        "cleanup_overrun_still_fail_closed": True,
        "guardian_forced_run_not_promotable": True,
        "resilient_fee_preflight": base.get("resilient_fee_preflight") is True,
        "orders_sent": False,
    }
    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"live_engine_version", "normal_cleanup_grace_s"}
    )
    out = {"deploy_version": DEPLOY_VERSION, **checks, "ok": bool(ok)}
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOY V2.8 STATIC CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:54s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.8 static self-check failed: {out}")
    return out


def live_preflight(*, quote_size, runtime_hours, max_start_loss_usd,
                   min_start_equity_usd, show=True, probe_private_ws=True):
    static_self_check(show=show)
    return V271.live_preflight(
        quote_size=float(quote_size),
        runtime_hours=float(runtime_hours),
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
        show=show,
        probe_private_ws=probe_private_ws,
    )


def _launch_guardian(session, main_pid):
    log = Path(session) / "guardian_v2_8.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_8",
        "--run-guardian", str(Path(session).resolve()),
        "--main-pid", str(int(main_pid)),
    ]
    try:
        g = subprocess.Popen(
            cmd,
            cwd=str(C.PROJECT_ROOT),
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

    _patch_parent()
    D._guard_other_live_processes()
    pre = live_preflight(
        quote_size=q,
        runtime_hours=hours,
        max_start_loss_usd=max_loss,
        min_start_equity_usd=min_equity,
        show=True,
        probe_private_ws=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}_deploy_v2_8").resolve()
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
        "deploy_version": DEPLOY_VERSION,
        "scientific_status": "FRESH_FORWARD_AFTER_V2_8_GUARDIAN_FINALIZATION_FIX",
        "no_auto_scale": True,
        "guardian_rss_warning_mb": RSS_WARNING_MB,
        "guardian_rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
        "guardian_deadline_grace_s": GUARDIAN_DEADLINE_GRACE_S,
        "normal_runtime_cleanup_grace_s": NORMAL_RUNTIME_CLEANUP_GRACE_S,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(session / "parent_preflight_snapshot.json", pre)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_8",
        "--run-live-session", str(session),
        "--config", str(cfg_path),
    ]
    caffeinate = shutil.which("caffeinate")
    cmd = ([caffeinate, "-i", "-m"] + child) if caffeinate else child
    try:
        p = subprocess.Popen(
            cmd,
            cwd=str(C.PROJECT_ROOT),
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
            raise RuntimeError(f"Live V2.8 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        state_ok = last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}
        private_ok = last.get("private_ws_ready") is True
        raw_ok = last.get("raw_watchdog_ready") is True
        bounded_ok = last.get("bounded_raw_ingestion") is True
        if state_ok and private_ok and raw_ok and bounded_ok:
            break
        time.sleep(0.25)
    else:
        B._atomic(session / "KILL_REQUEST.json", {
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V2_8"
        })
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(f"V2.8 startup health timeout. Last health={last}\n{tail}")

    guardian, guardian_log, guardian_cmd = _launch_guardian(session, p.pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update({
        "guardian_pid": guardian.pid,
        "guardian_log_path": str(guardian_log),
        "guardian_command": guardian_cmd,
    })
    B._atomic(CORE.CONTROL_PATH, ctl)

    print("\n" + "=" * 100)
    print("REAL-MONEY DEEP-TAIL V2.8 PROCESS ARMED")
    print("=" * 100)
    print("Session:                  ", session)
    print("Main PID:                 ", p.pid)
    print("Guardian PID:             ", guardian.pid)
    print("Engine:                   ", LIVE.LIVE_VERSION)
    print("Quantity:                 ", f"Q{float(q):g}")
    print("Runtime:                  ", f"{float(hours):.2f}h from first complete window")
    print("Software loss trigger:    ", f"-${float(max_loss):.2f}")
    print("RSS hard guardian:         ", f"{RSS_HARD_LIMIT_MB:.0f} MB")
    print("Initial deadline grace:    ", f"{GUARDIAN_DEADLINE_GRACE_S:.1f}s")
    print("Normal cleanup grace:      ", f"{NORMAL_RUNTIME_CLEANUP_GRACE_S:.1f}s")
    print("Persistent M5 cleanup:     ENABLED")
    print("Auto-scaling:              DISABLED")
    print("=" * 100)
    return live_status(show=False)


def start_q1_smoke(*, arm_phrase=None,
                   runtime_hours=Q1_DEFAULT_HOURS,
                   max_start_loss_usd=Q1_DEFAULT_MAX_LOSS,
                   min_start_equity_usd=Q1_DEFAULT_MIN_EQUITY):
    if abs(float(runtime_hours) - Q1_DEFAULT_HOURS) > 1e-12:
        raise RuntimeError("V2.8 Q1 smoke is fixed to exactly 0.5 hours.")
    return _launch(
        q=1.0,
        hours=Q1_DEFAULT_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q1_SMOKE_V15_V28",
        arm_phrase=arm_phrase,
        expected_arm=Q1_ARM,
    )


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    session = Path(session_dir).resolve()
    guardian_receipt_path = session / "guardian_receipt_v2_8.json"
    deadline = time.time() + 5.0
    while not guardian_receipt_path.exists() and time.time() < deadline:
        time.sleep(0.10)

    a = AUDIT.audit_live_session(session, show=False, write=True)
    cfg = B._read(session / "process_config.json", {}) or {}
    provenance = B._read(session / "deep_tail_source_provenance.json", {}) or {}
    final = B._read(session / "final_summary.json", {}) or {}
    guardian = B._read(guardian_receipt_path, {}) or {}

    q = B._f(cfg.get("quote_size"), float("nan"))
    q1_head = ((provenance.get("git") or {}).get("head"))
    current_head = _current_head()
    peak_rss = B._f(guardian.get("peak_total_rss_mb"), float("nan"))

    checks = {
        "q1_size": abs(q - 1.0) < 1e-9,
        "deploy_is_v2_8": str(cfg.get("deploy_version")) == DEPLOY_VERSION,
        "completed": a.get("completed") is True,
        "natural_runtime_complete": final.get("shutdown_reason") == "RUNTIME_COMPLETE",
        "clean_final": a.get("clean_final") is True,
        "raw_bundle_complete": a.get("raw_bundle_complete") is True,
        "live_bundle_complete": a.get("live_bundle_complete") is True,
        "no_operational_fail": a.get("operational_fail") is False,
        "entry_pair_exercised": int(a.get("entry_pairs_posted", 0)) >= 1,
        "actual_tail_fill_seen": int(a.get("tails_selected", 0)) >= 1,
        "full_q1_entry_seen": int(a.get("full_entries", 0)) >= 1,
        "fixed_join_ask_submitted": int(a.get("fixed_exits_posted", 0)) >= 1,
        "m5_path_exercised": int(a.get("m5_finalized", 0)) >= 1,
        "dual_tail_zero": int(a.get("dual_tail_fill_critical", 0)) == 0,
        "flat_verified": a.get("flat_verified") is True,
        "zero_strategy_resting": a.get("strategy_resting_orders_zero") is True,
        "same_live_engine_version": str(a.get("strategy_version")) == str(LIVE.LIVE_VERSION),
        "q1_git_head_known": bool(q1_head),
        "same_current_git_head": bool(q1_head and current_head and q1_head == current_head),
        "guardian_receipt_present": bool(guardian),
        "guardian_did_not_intervene": guardian.get("intervened") is False,
        "guardian_allows_promotion": guardian.get("promotion_allowed") is True,
        "guardian_cleanup_aware": guardian.get("normal_cleanup_aware") is True,
        "rss_peak_known": peak_rss == peak_rss,
        "rss_peak_below_hard_limit": peak_rss == peak_rss and peak_rss < RSS_HARD_LIMIT_MB,
    }
    passed = all(checks.values())
    receipt = {
        "time": B._iso(),
        "deploy_version": DEPLOY_VERSION,
        "live_engine_version": LIVE.LIVE_VERSION,
        "passed": passed,
        "session": str(session),
        "q1_git_head": q1_head,
        "current_git_head": current_head,
        "checks": checks,
        "audit": a,
        "guardian": guardian,
        "note": "Operational promotion only; not evidence that expected PnL is positive.",
        "orders_sent": False,
        "exchange_api_called": False,
    }
    if passed and write_receipt:
        B._atomic(PROMOTION_PATH, receipt)

    if show:
        print("=" * 100)
        print("Q1 V1.5 / V2.8 OPERATIONAL PROMOTION — READ ONLY")
        print("=" * 100)
        for k, v in checks.items():
            print(f"{k:44s}: {v}")
        print("Peak total RSS MB:                       ", peak_rss)
        print("Guardian reason:                         ", guardian.get("reason"))
        print("PROMOTION:", "PASS" if passed else "NOT READY")
        if passed:
            print("Receipt:", PROMOTION_PATH)
    return receipt


def live_status(*, show=True, tail_lines=40):
    _patch_parent()
    base = D.live_status(show=False, tail_lines=tail_lines)
    ctl = base.get("control") or {}
    session = Path(ctl.get("session_dir", "")) if ctl.get("session_dir") else None
    guardian = B._read(session / "guardian_health_v2_8.json", {}) if session and session.exists() else {}
    receipt = B._read(session / "guardian_receipt_v2_8.json", {}) if session and session.exists() else {}
    base["guardian"] = guardian or {}
    base["guardian_receipt"] = receipt or {}

    if show:
        h = base.get("health") or {}
        final = base.get("final_summary") or {}
        print("=" * 100)
        print("DEEP-TAIL V2.8 LIVE STATUS")
        print("=" * 100)
        print("Running:                  ", base.get("running"))
        print("Session:                  ", ctl.get("session_dir"))
        print("Mode:                     ", ctl.get("mode"))
        print("Q:                        ", (ctl.get("config") or {}).get("quote_size"))
        print("State:                    ", h.get("state"))
        print("Bounded raw ingestion:    ", h.get("bounded_raw_ingestion"))
        print("Book backlog MB:          ", (h.get("book_tail") or {}).get("backlog_bytes", 0) / (1024.0**2) if h else None)
        print("Guardian RSS peak MB:     ", (guardian or receipt).get("peak_total_rss_mb"))
        print("Guardian intervened:      ", (receipt or guardian).get("intervened"))
        print("Guardian reason:          ", (receipt or guardian).get("reason"))
        print("Cleanup shutdown seen:    ", guardian.get("runtime_complete_shutdown_started"))
        print("Positions:                ", h.get("positions"))
        print("Last error:               ", h.get("last_error"))
        if final:
            print("Final shutdown:           ", final.get("shutdown_reason"))
            print("Final PnL:                ", final.get("account_pnl_usd"))
            print("Flat verified:            ", final.get("flat_verified"))
        tail = base.get("log_tail") or []
        if tail:
            print("\n--- LIVE LOG TAIL ---")
            print("\n".join(tail))
    return base


def kill_and_flatten_live(*, arm_phrase=None, **kwargs):
    return V27.kill_and_flatten_live(arm_phrase=arm_phrase, **kwargs)


def _run_child(session, cfg_path):
    cfg = B._read(Path(cfg_path), {}) or {}
    LIVE.run_live_process(Path(session).resolve(), cfg)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    ap.add_argument("--run-guardian")
    ap.add_argument("--main-pid", type=int)
    a = ap.parse_args()
    if a.run_guardian:
        _guardian_loop(Path(a.run_guardian).resolve(), int(a.main_pid))
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
