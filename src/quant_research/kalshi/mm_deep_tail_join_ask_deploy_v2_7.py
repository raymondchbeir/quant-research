from __future__ import annotations

"""V2.7 deployment for V1.5 bounded raw ingestion + independent guardian.

Operational changes only; strategy/alpha mechanics are unchanged from V1.4.

The guardian is a separate OS process.  It does not depend on the strategy event
loop, so a strategy-loop stall cannot prevent it from enforcing:
- an RSS ceiling for the live process group + raw recorder;
- the configured runtime deadline (with a small normal-shutdown grace period).

If either guard fires, the guardian first requests normal engine shutdown.  If the
engine remains alive it triggers only this run's order group, terminates the live
process group and recorder, runs the existing authoritative emergency flatten, and
verifies zero strategy-group resting orders and zero nonzero account positions.
Any guardian intervention makes the run ineligible for Q1 promotion.

Importing this module never sends orders.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_deep_tail_join_ask_live_v1 as CORE
from . import mm_deep_tail_join_ask_live_v1_5 as LIVE
from . import mm_deep_tail_join_ask_live_audit_v1 as AUDIT
from . import mm_deep_tail_join_ask_deploy_v2 as D
from . import mm_deep_tail_join_ask_deploy_v2_1 as D21
from . import mm_deep_tail_join_ask_deploy_v2_2 as D22


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_7_V1_5_BOUNDED_MEMORY_GUARDIAN"
Q1_ARM = "LIVE_DEEP_TAIL_Q1_V15"
Q10_ARM = "LIVE_DEEP_TAIL_Q10_1H_V15"
KILL_ARM = B.KILL_ARM

Q1_DEFAULT_HOURS = 0.5
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0
Q10_HOURS = 1.0
Q10_DEFAULT_MAX_LOSS = 20.0
Q10_DEFAULT_MIN_EQUITY = 75.0

STARTUP_TIMEOUT_S = 120.0
GUARDIAN_POLL_S = 0.50
GUARDIAN_DEADLINE_GRACE_S = 5.0
GUARDIAN_NORMAL_KILL_GRACE_S = 3.0
RSS_WARNING_MB = 1024.0
RSS_HARD_LIMIT_MB = 2048.0

PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2_7.json"


def _patch_parent():
    D22._install_patch()  # notebook-safe authenticated WS preflight
    D.LIVE = LIVE
    D._patch_paths()


def _current_head():
    return (V10._git_state() or {}).get("head")


def _ps_rows():
    try:
        p = subprocess.run(
            ["ps", "-axo", "pid=,pgid=,rss=,comm="],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if p.returncode != 0:
            return []
        out = []
        for line in p.stdout.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                pgid = int(parts[1])
                rss_kib = float(parts[2])
            except Exception:
                continue
            out.append({
                "pid": pid,
                "pgid": pgid,
                "rss_bytes": int(max(0.0, rss_kib) * 1024.0),
                "command": parts[3] if len(parts) > 3 else "",
            })
        return out
    except Exception:
        return []


def _pid_group(pid):
    try:
        return int(os.getpgid(int(pid)))
    except Exception:
        return None


def _rss_snapshot(main_pid, recorder_pid=None):
    rows = _ps_rows()
    pgid = _pid_group(main_pid)
    group_rows = [r for r in rows if pgid is not None and r["pgid"] == pgid]
    strategy_group = sum(r["rss_bytes"] for r in group_rows)

    recorder_pid = int(recorder_pid or 0)
    recorder = sum(
        r["rss_bytes"] for r in rows
        if recorder_pid > 0 and r["pid"] == recorder_pid
    )
    return {
        "main_pid": int(main_pid),
        "main_pgid": pgid,
        "strategy_group_rss_bytes": int(strategy_group),
        "strategy_group_rss_mb": strategy_group / (1024.0 ** 2),
        "recorder_pid": recorder_pid or None,
        "recorder_rss_bytes": int(recorder),
        "recorder_rss_mb": recorder / (1024.0 ** 2),
        "total_rss_bytes": int(strategy_group + recorder),
        "total_rss_mb": (strategy_group + recorder) / (1024.0 ** 2),
        "group_processes": group_rows,
    }


def _terminate_pid_group(pid, *, term_wait_s=4.0, kill_wait_s=2.0):
    pid = int(pid or 0)
    out = {
        "pid": pid,
        "pgid": None,
        "sigterm_sent": False,
        "sigkill_sent": False,
        "dead": not B._pid_alive(pid) if pid > 0 else True,
    }
    if pid <= 0 or out["dead"]:
        return out

    pgid = _pid_group(pid)
    out["pgid"] = pgid
    if pgid is None:
        return out
    if int(pgid) == int(os.getpgrp()):
        raise RuntimeError("REFUSING to signal guardian/notebook process group")

    try:
        os.killpg(pgid, signal.SIGTERM)
        out["sigterm_sent"] = True
    except ProcessLookupError:
        pass

    deadline = time.time() + float(term_wait_s)
    while B._pid_alive(pid) and time.time() < deadline:
        time.sleep(0.10)

    if B._pid_alive(pid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            out["sigkill_sent"] = True
        except ProcessLookupError:
            pass
        deadline = time.time() + float(kill_wait_s)
        while B._pid_alive(pid) and time.time() < deadline:
            time.sleep(0.10)

    out["dead"] = not B._pid_alive(pid)
    return out


def _authoritative_state(client, gid):
    positions, pt = B._positions(client)
    resting, rt = B._resting(client)
    nonzero = [
        r for r in positions
        if abs(B._f(r.get("position_fp"), 0.0)) > B.EPS
    ]
    group_resting = [
        r for r in resting
        if str(r.get("order_group_id") or "") == str(gid or "")
    ]
    return {
        "positions": positions,
        "resting": resting,
        "nonzero": nonzero,
        "group_resting": group_resting,
        "timing": {"positions": pt, "resting": rt},
    }


def _force_recovery(session, main_pid, *, reason, guardian_source):
    """Fail-closed account recovery used only after the live process failed to exit."""
    session = Path(session).resolve()
    main_pid = int(main_pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    group = B._read(session / "order_group.json", {}) or {}
    gid = str(group.get("order_group_id") or "")
    health = B._read(session / "health.json", {}) or {}
    recorder_pid = int(health.get("recorder_pid") or 0)

    client = B.Q1.LiveClient()
    group_trigger = B._trigger_group(client, gid) if gid else {
        "ok": False, "reason": "missing_order_group_id"
    }

    main_stop = _terminate_pid_group(main_pid)
    recorder_stop = _terminate_pid_group(recorder_pid) if recorder_pid > 0 else {
        "pid": None, "dead": True, "note": "no recorder pid"
    }

    if not main_stop.get("dead"):
        raise RuntimeError(f"guardian could not terminate live process: {main_stop}")

    # Existing fallback uses authoritative exchange reads and reduce-only IOC
    # flattening for recognized strategy tickers.
    fallback = B._fallback_cleanup(ctl)
    time.sleep(0.30)
    state = _authoritative_state(client, gid)

    receipt = {
        "time": B._iso(),
        "session": str(session),
        "main_pid": main_pid,
        "recorder_pid": recorder_pid or None,
        "reason": str(reason),
        "guardian_source": str(guardian_source),
        "intervened": True,
        "promotion_allowed": False,
        "group_trigger": group_trigger,
        "main_stop": main_stop,
        "recorder_stop": recorder_stop,
        "fallback_cleanup": fallback,
        "strategy_group_resting_count": len(state["group_resting"]),
        "all_account_resting_count": len(state["resting"]),
        "nonzero_position_count": len(state["nonzero"]),
        "group_resting": state["group_resting"],
        "nonzero_positions": state["nonzero"],
    }
    B._atomic(session / "guardian_recovery_v2_7.json", receipt)

    latest_ctl = B._read(CORE.CONTROL_PATH, {}) or ctl
    if str(latest_ctl.get("session_dir") or "") == str(session):
        latest_ctl.update({
            "running": False,
            "stopped_at": B._iso(),
            "shutdown_reason": f"V2_7_GUARDIAN_{reason}",
            "guardian_intervened": True,
        })
        B._atomic(CORE.CONTROL_PATH, latest_ctl)

    if state["group_resting"]:
        raise RuntimeError(f"guardian recovery left resting strategy orders: {state['group_resting']}")
    if state["nonzero"]:
        raise RuntimeError(f"guardian recovery left nonzero positions: {state['nonzero']}")
    return receipt


def _guardian_loop(session, main_pid):
    session = Path(session).resolve()
    main_pid = int(main_pid)
    health_path = session / "health.json"
    guardian_health_path = session / "guardian_health_v2_7.json"
    receipt_path = session / "guardian_receipt_v2_7.json"
    events_path = session / "guardian_events_v2_7.jsonl"

    peak_total = 0.0
    peak_group = 0.0
    peak_recorder = 0.0
    warning_emitted = False

    while True:
        if not B._pid_alive(main_pid):
            receipt = {
                "time": B._iso(),
                "session": str(session),
                "main_pid": main_pid,
                "intervened": False,
                "reason": "CHILD_EXITED_WITHOUT_GUARDIAN_ACTION",
                "promotion_allowed": True,
                "peak_total_rss_mb": peak_total,
                "peak_strategy_group_rss_mb": peak_group,
                "peak_recorder_rss_mb": peak_recorder,
                "rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
            }
            B._atomic(receipt_path, receipt)
            B._atomic(guardian_health_path, {**receipt, "running": False})
            return receipt

        h = B._read(health_path, {}) or {}
        recorder_pid = h.get("recorder_pid")
        rss = _rss_snapshot(main_pid, recorder_pid)
        peak_total = max(peak_total, float(rss["total_rss_mb"]))
        peak_group = max(peak_group, float(rss["strategy_group_rss_mb"]))
        peak_recorder = max(peak_recorder, float(rss["recorder_rss_mb"]))

        deadline = B._f(h.get("trade_deadline"), float("nan"))
        now = time.time()
        deadline_overrun_s = (
            max(0.0, now - deadline) if deadline == deadline else None
        )

        gh = {
            "time": B._iso(),
            "running": True,
            "intervened": False,
            "main_pid": main_pid,
            "rss": rss,
            "peak_total_rss_mb": peak_total,
            "peak_strategy_group_rss_mb": peak_group,
            "peak_recorder_rss_mb": peak_recorder,
            "rss_warning_mb": RSS_WARNING_MB,
            "rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
            "trade_deadline": deadline if deadline == deadline else None,
            "deadline_grace_s": GUARDIAN_DEADLINE_GRACE_S,
            "deadline_overrun_s": deadline_overrun_s,
            "bounded_raw_ingestion": h.get("bounded_raw_ingestion"),
            "book_tail_backlog_bytes": ((h.get("book_tail") or {}).get("backlog_bytes")),
        }
        B._atomic(guardian_health_path, gh)

        if rss["total_rss_mb"] >= RSS_WARNING_MB and not warning_emitted:
            B._append(events_path, {
                "time": B._iso(),
                "event": "RSS_WARNING",
                "rss": rss,
                "warning_mb": RSS_WARNING_MB,
                "hard_mb": RSS_HARD_LIMIT_MB,
            })
            warning_emitted = True

        trigger_reason = None
        if rss["total_rss_mb"] >= RSS_HARD_LIMIT_MB:
            trigger_reason = "RSS_HARD_LIMIT"
        elif deadline == deadline and now >= deadline + GUARDIAN_DEADLINE_GRACE_S:
            trigger_reason = "RUNTIME_DEADLINE_OVERRUN"

        if trigger_reason is not None:
            B._append(events_path, {
                "time": B._iso(),
                "event": "GUARDIAN_TRIGGER",
                "reason": trigger_reason,
                "rss": rss,
                "trade_deadline": deadline if deadline == deadline else None,
            })

            # Give the engine one short chance to execute its normal cleanup.
            B._atomic(session / "KILL_REQUEST.json", {
                "time": B._iso(),
                "reason": f"V2_7_GUARDIAN_{trigger_reason}",
            })
            grace = time.time() + GUARDIAN_NORMAL_KILL_GRACE_S
            while B._pid_alive(main_pid) and time.time() < grace:
                time.sleep(0.10)

            if B._pid_alive(main_pid):
                recovery = _force_recovery(
                    session,
                    main_pid,
                    reason=trigger_reason,
                    guardian_source="EXTERNAL_GUARDIAN_PROCESS",
                )
                receipt = {
                    "time": B._iso(),
                    "session": str(session),
                    "main_pid": main_pid,
                    "intervened": True,
                    "reason": trigger_reason,
                    "promotion_allowed": False,
                    "peak_total_rss_mb": peak_total,
                    "peak_strategy_group_rss_mb": peak_group,
                    "peak_recorder_rss_mb": peak_recorder,
                    "rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
                    "recovery": recovery,
                }
            else:
                receipt = {
                    "time": B._iso(),
                    "session": str(session),
                    "main_pid": main_pid,
                    "intervened": True,
                    "reason": trigger_reason,
                    "promotion_allowed": False,
                    "peak_total_rss_mb": peak_total,
                    "peak_strategy_group_rss_mb": peak_group,
                    "peak_recorder_rss_mb": peak_recorder,
                    "rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
                    "normal_engine_exit_after_guardian_request": True,
                }
            B._atomic(receipt_path, receipt)
            B._atomic(guardian_health_path, {**receipt, "running": False})
            return receipt

        time.sleep(GUARDIAN_POLL_S)


def static_self_check(*, show=True):
    _patch_parent()
    core = LIVE.static_self_check(show=False)
    checks = {
        "core_ok": bool(core.get("ok")),
        "engine_version": LIVE.LIVE_VERSION,
        "persistent_m5_cleanup": core.get("m5_persistent_cleanup") is True,
        "bounded_raw_ingestion": core.get("bounded_raw_ingestion") is True,
        "legacy_read_to_eof_removed": core.get("legacy_read_to_eof_removed_from_strategy_tail") is True,
        "rss_health_telemetry": core.get("rss_health_telemetry") is True,
        "alpha_rules_unchanged": core.get("alpha_rules_unchanged_from_v1_4") is True,
        "book_tail_row_budget": core.get("book_tail_max_rows_per_read") == LIVE.BOOK_TAIL_MAX_ROWS_PER_READ,
        "book_tail_byte_budget": core.get("book_tail_max_bytes_per_read") == LIVE.BOOK_TAIL_MAX_BYTES_PER_READ,
        "guardian_separate_process": True,
        "guardian_rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
        "guardian_deadline_grace_s": GUARDIAN_DEADLINE_GRACE_S,
        "guardian_forced_run_not_promotable": True,
        "ladder": tuple(CORE.LADDER_Q) == (1, 5, 10, 20, 30, 50, 100),
    }
    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"engine_version", "guardian_rss_hard_limit_mb", "guardian_deadline_grace_s"}
    )
    out = {"deploy_version": DEPLOY_VERSION, **checks, "ok": ok, "orders_sent": False}
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOY V2.7 STATIC CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:50s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.7 static self-check failed: {out}")
    return out


def api_capacity_preflight(**kwargs):
    return D21.api_capacity_preflight(**kwargs)


def live_preflight(*, quote_size, runtime_hours, max_start_loss_usd,
                   min_start_equity_usd, show=True, probe_private_ws=True):
    _patch_parent()
    static_self_check(show=show)
    cap = D21.api_capacity_preflight(show=show)
    out = D.live_preflight(
        quote_size=float(quote_size),
        runtime_hours=float(runtime_hours),
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
        show=show,
        probe_private_ws=probe_private_ws,
    )
    out = dict(out)
    out.update({
        "api_capacity_preflight": cap,
        "deploy_wrapper_version": DEPLOY_VERSION,
        "live_engine_version": LIVE.LIVE_VERSION,
        "guardian_rss_warning_mb": RSS_WARNING_MB,
        "guardian_rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
        "guardian_deadline_grace_s": GUARDIAN_DEADLINE_GRACE_S,
        "orders_sent": False,
    })
    return out


def _launch_guardian(session, main_pid):
    log = Path(session) / "guardian_v2_7.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_7",
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
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}_deploy_v2_7").resolve()
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
        "scientific_status": "FRESH_FORWARD_AFTER_BOUNDED_MEMORY_AND_INDEPENDENT_GUARDIAN_FIX",
        "no_auto_scale": True,
        "guardian_rss_warning_mb": RSS_WARNING_MB,
        "guardian_rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
        "guardian_deadline_grace_s": GUARDIAN_DEADLINE_GRACE_S,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(session / "parent_preflight_snapshot.json", pre)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_7",
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
            raise RuntimeError(f"Live V2.7 process exited during startup rc={p.returncode}\n{tail}")
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
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V2_7"
        })
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(f"V2.7 startup health timeout. Last health={last}\n{tail}")

    guardian, guardian_log, guardian_cmd = _launch_guardian(session, p.pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update({
        "guardian_pid": guardian.pid,
        "guardian_log_path": str(guardian_log),
        "guardian_command": guardian_cmd,
    })
    B._atomic(CORE.CONTROL_PATH, ctl)

    print("\n" + "=" * 100)
    print("REAL-MONEY DEEP-TAIL V2.7 PROCESS ARMED")
    print("=" * 100)
    print("Session:                  ", session)
    print("Main PID:                 ", p.pid)
    print("Guardian PID:             ", guardian.pid)
    print("Engine:                   ", LIVE.LIVE_VERSION)
    print("Quantity:                 ", f"Q{float(q):g}")
    print("Runtime:                  ", f"{float(hours):.2f}h from first complete window")
    print("Software loss trigger:    ", f"-${float(max_loss):.2f}")
    print("Bounded raw ingestion:     ENABLED")
    print("RSS warning:               ", f"{RSS_WARNING_MB:.0f} MB")
    print("RSS hard guardian:         ", f"{RSS_HARD_LIMIT_MB:.0f} MB")
    print("Deadline guardian grace:   ", f"{GUARDIAN_DEADLINE_GRACE_S:.1f}s")
    print("Persistent M5 cleanup:     ENABLED")
    print("Auto-scaling:              DISABLED")
    print("Emergency kill phrase:     KILL_AND_FLATTEN")
    print("=" * 100)
    return live_status(show=False)


def start_q1_smoke(*, arm_phrase=None,
                   runtime_hours=Q1_DEFAULT_HOURS,
                   max_start_loss_usd=Q1_DEFAULT_MAX_LOSS,
                   min_start_equity_usd=Q1_DEFAULT_MIN_EQUITY):
    return _launch(
        q=1.0,
        hours=float(runtime_hours),
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q1_SMOKE_V15",
        arm_phrase=arm_phrase,
        expected_arm=Q1_ARM,
    )


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    session = Path(session_dir).resolve()

    # Give the external guardian a moment to observe the child exit and write its
    # non-intervention receipt.
    guardian_receipt_path = session / "guardian_receipt_v2_7.json"
    deadline = time.time() + 3.0
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
        print("Q1 V1.5 -> Q10 OPERATIONAL PROMOTION — READ ONLY")
        print("=" * 100)
        for k, v in checks.items():
            print(f"{k:42s}: {v}")
        print("Peak total RSS MB:                    ", peak_rss)
        print("PROMOTION:", "PASS" if passed else "NOT READY")
        if passed:
            print("Receipt:", PROMOTION_PATH)
    return receipt


def _require_q1_promotion():
    r = B._read(PROMOTION_PATH, {}) or {}
    head = _current_head()
    if not r or r.get("passed") is not True:
        raise RuntimeError("Q10 HARD GATE: no passing V1.5 Q1 promotion receipt.")
    if str(r.get("live_engine_version")) != str(LIVE.LIVE_VERSION):
        raise RuntimeError("Q10 HARD GATE: Q1 used a different live-engine version.")
    if not head or str(r.get("q1_git_head")) != str(head):
        raise RuntimeError("Q10 HARD GATE: git HEAD changed after Q1; run a new Q1 smoke.")
    return r


def start_q10_one_hour(*, arm_phrase=None,
                       runtime_hours=Q10_HOURS,
                       max_start_loss_usd=Q10_DEFAULT_MAX_LOSS,
                       min_start_equity_usd=Q10_DEFAULT_MIN_EQUITY):
    if abs(float(runtime_hours) - Q10_HOURS) > 1e-12:
        raise RuntimeError("V2.7 Q10 stage is fixed to exactly 1.0 hour.")
    _require_q1_promotion()
    return _launch(
        q=10.0,
        hours=Q10_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q10_1H_V15",
        arm_phrase=arm_phrase,
        expected_arm=Q10_ARM,
    )


def live_status(*, show=True, tail_lines=25):
    _patch_parent()
    base = D.live_status(show=False, tail_lines=tail_lines)
    ctl = base.get("control") or {}
    session = Path(ctl.get("session_dir", "")) if ctl.get("session_dir") else None
    guardian = B._read(session / "guardian_health_v2_7.json", {}) if session and session.exists() else {}
    base["guardian"] = guardian or {}

    if show:
        h = base.get("health") or {}
        final = base.get("final_summary") or {}
        print("=" * 100)
        print("DEEP-TAIL V2.7 LIVE STATUS")
        print("=" * 100)
        print("Running:                 ", base.get("running"))
        print("Session:                 ", ctl.get("session_dir"))
        print("Mode:                    ", ctl.get("mode"))
        print("Q:                       ", (ctl.get("config") or {}).get("quote_size"))
        print("State:                   ", h.get("state"))
        print("Private WS ready:        ", h.get("private_ws_ready"))
        print("Raw watcher ready:       ", h.get("raw_watchdog_ready"))
        print("Bounded raw ingestion:   ", h.get("bounded_raw_ingestion"))
        print("Strategy RSS MB:         ", h.get("strategy_rss_mb"))
        print("Strategy RSS peak MB:    ", h.get("strategy_rss_peak_observed_mb"))
        print("Book backlog MB:         ", B._f(((h.get("book_tail") or {}).get("backlog_bytes")), 0.0) / (1024.0 ** 2))
        print("Guardian total RSS MB:   ", ((guardian or {}).get("rss") or {}).get("total_rss_mb"))
        print("Guardian peak RSS MB:    ", (guardian or {}).get("peak_total_rss_mb"))
        print("Guardian intervened:     ", (guardian or {}).get("intervened"))
        print("Positions:               ", h.get("positions"))
        print("Active tracks:           ", len(h.get("active_tracks") or {}))
        print("Trade deadline:          ", h.get("trade_deadline"))
        print("Last error:              ", h.get("last_error"))
        if final:
            print("Shutdown reason:         ", final.get("shutdown_reason"))
            print("Flat verified:           ", final.get("flat_verified"))
            print("Zero resting:            ", final.get("strategy_resting_orders_zero"))
        log_path = Path(ctl.get("log_path", "")) if ctl.get("log_path") else None
        if log_path and log_path.exists() and int(tail_lines) > 0:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n--- LIVE LOG TAIL ---")
            print("\n".join(lines[-int(tail_lines):]))
    return base


def kill_and_flatten_live(*, arm_phrase=None, wait_s=12.0):
    if str(arm_phrase) != str(KILL_ARM):
        raise RuntimeError(f"Emergency stop refused. Pass arm_phrase={KILL_ARM!r} exactly.")
    _patch_parent()
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    if not ctl:
        raise RuntimeError("No deep-tail live control file.")
    session = Path(ctl.get("session_dir", "")).resolve()
    pid = int(ctl.get("pid") or 0)

    B._atomic(session / "KILL_REQUEST.json", {
        "time": B._iso(), "reason": "MANUAL_KILL_AND_FLATTEN_V2_7"
    })
    deadline = time.time() + float(wait_s)
    while B._pid_alive(pid) and time.time() < deadline:
        time.sleep(0.20)

    if not B._pid_alive(pid):
        print("Live process stopped through normal engine shutdown.")
        return live_status(show=False)

    receipt = _force_recovery(
        session,
        pid,
        reason="MANUAL_KILL_TIMEOUT",
        guardian_source="NOTEBOOK_V2_7_KILL_HELPER",
    )
    print("Forced recovery complete; this run is NOT promotable.")
    return receipt


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
        _run_child(a.run_live_session, a.config)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "Q1_ARM",
    "Q10_ARM",
    "KILL_ARM",
    "PROMOTION_PATH",
    "RSS_WARNING_MB",
    "RSS_HARD_LIMIT_MB",
    "GUARDIAN_DEADLINE_GRACE_S",
    "static_self_check",
    "api_capacity_preflight",
    "live_preflight",
    "start_q1_smoke",
    "q1_promotion_check",
    "start_q10_one_hour",
    "live_status",
    "kill_and_flatten_live",
]
