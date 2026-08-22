from __future__ import annotations

"""V2.9.6 supervisor for compact rotating Q50 M1->M5 trader generations.

Architecture change only; frozen strategy mechanics are unchanged.

One long-lived authenticated M0->M12(+30s) recorder is owned by this supervisor.
Short-lived V1.11 trader generations attach to that recorder, trade exactly one
complete M0->M5 window, prove zero strategy-group resting orders and zero account
positions after M5, then exit. A fresh Python trader process is launched immediately
for the next window while the same recorder continues running.

Risk invariants:
- the calibrated account-equity baseline is sampled once per parent session;
- the software loss trigger is fixed to that baseline across every generation;
- Q50 remains fixed at a $20 start-to-current software stop and $125 minimum equity;
- the independent guardian warns at 450 MiB and fails closed at 750 MiB of trader
  process-group RSS (recorder RSS is measured separately and is not reset/hidden);
- any abnormal trader exit is recovered from authoritative exchange state before the
  recorder is stopped;
- Q50 arming is refused until a Q1 rotation smoke on the exact current Git HEAD has
  completed one verified M5 trader rotation and kept the same recorder alive through
  that market's M12+30 label boundary.

Importing this module performs no API calls and sends no orders.
"""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_deploy_v2_7 as V27
from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_8_2 as V282
from . import mm_deep_tail_join_ask_deploy_v2_8_7 as V287
from . import mm_deep_tail_join_ask_deploy_v2_8_8 as V288
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_11_rotation as LIVE
from . import mm_event_time_m0_m12_recorder_v6_auth as REC


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M5_RECORD_M12_V2_9_6_ROTATING_SUPERVISOR"
CORE = V282.CORE
KILL_ARM = V282.KILL_ARM

ROTATION_SMOKE_ARM = "LIVE_DEEP_TAIL_Q1_ROTATION_SMOKE_V296"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M5_RECORD_M12_12H_V296_ROTATION"

SMOKE_Q = 1.0
SMOKE_RUNTIME_HOURS = 0.5
SMOKE_MAX_LOSS_USD = 5.0
SMOKE_MIN_EQUITY_USD = 25.0

Q50_Q = 50.0
Q50_HOURS = 12.0
Q50_MAX_LOSS_USD = 20.0
Q50_MIN_EQUITY_USD = 125.0

M1_S = 60.0
M5_S = 300.0
RECORDER_M12_S = 720.0
LABEL_TAIL_END_S = 750.0

GENERATION_RSS_WARNING_MB = 450.0
GENERATION_RSS_HARD_LIMIT_MB = 750.0
SUPERVISOR_POLL_S = 0.20
GUARDIAN_POLL_S = 0.50
GUARDIAN_KILL_GRACE_S = 6.0
STARTUP_TIMEOUT_S = 150.0
SMOKE_TAIL_EXTRA_GRACE_S = 2.0

SUPERVISOR_HEALTH_FILE = "supervisor_health_v2_9_6.json"
SUPERVISOR_FINAL_FILE = "supervisor_final_v2_9_6.json"
SUPERVISOR_EVENTS_FILE = "supervisor_events_v2_9_6.jsonl"
SESSION_KILL_FILE = "SESSION_KILL_REQUEST.json"
GUARDIAN_HEALTH_FILE = "guardian_health_v2_9_6.json"
GUARDIAN_RECEIPT_FILE = "guardian_receipt_v2_9_6.json"
GUARDIAN_EVENTS_FILE = "guardian_events_v2_9_6.jsonl"
RECOVERY_RECEIPT_FILE = "supervisor_recovery_v2_9_6.json"
PROMOTION_PATH = CORE.ROOT / "q50_rotation_promotion_v2_9_6.json"


def _current_head():
    return str((V10._git_state() or {}).get("head") or "")


def _tail_text(path, chars=18000):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-int(chars):]
    except Exception:
        return ""


def _is_clean_final(final, *, allowed_reasons=None):
    final = final or {}
    reason = str(final.get("shutdown_reason") or "")
    reason_ok = True if allowed_reasons is None else reason in set(allowed_reasons)
    return bool(
        final
        and reason_ok
        and final.get("flat_verified") is True
        and final.get("strategy_resting_orders_zero") is True
        and final.get("last_error") in (None, "")
    )


def _stop_pid(pid, *, sig=signal.SIGINT, timeout_s=30.0):
    pid = int(pid or 0)
    out = {"pid": pid or None, "signal": int(sig), "sent": False, "dead": True}
    if pid <= 0 or not B._pid_alive(pid):
        return out
    out["dead"] = False
    try:
        os.kill(pid, sig)
        out["sent"] = True
    except ProcessLookupError:
        out["dead"] = True
        return out
    except Exception as exc:
        out["error"] = repr(exc)
    deadline = time.time() + float(timeout_s)
    while B._pid_alive(pid) and time.time() < deadline:
        time.sleep(0.10)
    if B._pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            out["sigterm_sent"] = True
        except Exception as exc:
            out["sigterm_error"] = repr(exc)
        deadline = time.time() + 5.0
        while B._pid_alive(pid) and time.time() < deadline:
            time.sleep(0.10)
    out["dead"] = not B._pid_alive(pid)
    return out


def _promotion_record():
    return B._read(PROMOTION_PATH, {}) or {}


def rotation_promotion_status(*, show=True):
    rec = _promotion_record()
    head = _current_head()
    checks = {
        "promotion_file_present": bool(rec),
        "passed": rec.get("passed") is True,
        "deploy_version_exact": rec.get("deploy_version") == DEPLOY_VERSION,
        "live_version_exact": rec.get("live_version") == LIVE.LIVE_VERSION,
        "git_head_exact": bool(head) and rec.get("git_head") == head,
        "same_recorder_survived_trader_rotation": rec.get("same_recorder_survived_trader_rotation") is True,
        "m12_plus_30_observed_after_rotation": rec.get("m12_plus_30_observed_after_rotation") is True,
        "rotation_checkpoint_verified": rec.get("rotation_checkpoint_verified") is True,
        "q1_only_smoke": rec.get("smoke_quantity") == SMOKE_Q,
    }
    ready = all(checks.values())
    out = {
        "ready_for_q50": bool(ready),
        "current_git_head": head,
        "promotion_path": str(PROMOTION_PATH),
        "checks": checks,
        "record": rec,
        "orders_sent": False,
    }
    if show:
        print("=" * 120)
        print("V2.9.6 Q50 ROTATION PROMOTION STATUS — READ ONLY")
        print("=" * 120)
        print("ready_for_q50:", out["ready_for_q50"])
        print("current_git_head:", head)
        print("promotion_path:", PROMOTION_PATH)
        for k, v in checks.items():
            print(f"  {k:48s}: {v}")
    return out


def _require_q50_promotion():
    status = rotation_promotion_status(show=False)
    if not status.get("ready_for_q50"):
        raise RuntimeError(
            "Q50 V2.9.6 ARMING BLOCKED: run and pass the Q1 rotation smoke on this exact Git HEAD first. "
            f"Promotion checks={status.get('checks')}"
        )
    return status


def static_self_check(*, show=True):
    pp = V287._install_child_pythonpath()
    live = LIVE.static_self_check(show=False)
    rec = REC.static_self_check(show=False)
    checks = {
        "live_v1_11_ok": live.get("ok") is True,
        "compact_generation_watchdog": live.get("watchdog_history_removed") is True,
        "fresh_row_after_generation_start_required": live.get("fresh_row_gate_regression") is True,
        "verified_m5_rotation_checkpoint": live.get("verified_m5_rotation_checkpoint_required") is True,
        "external_recorder_parent_owned": live.get("external_recorder_shutdown_is_parent_owned") is True,
        "fixed_session_risk_baseline": live.get("fixed_session_risk_baseline_required") is True,
        "recorder_m0_m12": rec.get("ok") is True and abs(REC.TRADE_WINDOW_END_S - RECORDER_M12_S) < 1e-12,
        "recorder_label_tail_750": abs(REC.LABEL_TAIL_END_S - LABEL_TAIL_END_S) < 1e-12,
        "strategy_m1_unchanged_60": abs(V1.M1_S - M1_S) < 1e-12,
        "strategy_m5_unchanged_300": abs(V1.M5_S - M5_S) < 1e-12,
        "q50_fixed_50": abs(Q50_Q - 50.0) < 1e-12,
        "q50_runtime_fixed_12h": abs(Q50_HOURS - 12.0) < 1e-12,
        "q50_loss_trigger_fixed_20": abs(Q50_MAX_LOSS_USD - 20.0) < 1e-12,
        "q50_min_equity_125": abs(Q50_MIN_EQUITY_USD - 125.0) < 1e-12,
        "trader_rss_warning_450": abs(GENERATION_RSS_WARNING_MB - 450.0) < 1e-12,
        "trader_rss_hard_750": abs(GENERATION_RSS_HARD_LIMIT_MB - 750.0) < 1e-12,
        "trader_rss_hard_below_old_2gb": GENERATION_RSS_HARD_LIMIT_MB < V28.RSS_HARD_LIMIT_MB,
        "q50_requires_same_head_smoke_promotion": True,
        "abnormal_generation_fail_closed": True,
        "child_pythonpath_installed": pp.get("installed") is True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    promo = rotation_promotion_status(show=False)
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "recorder_version": REC.STUDY_VERSION,
        "promotion_ready_for_q50": promo.get("ready_for_q50"),
        "promotion_path": str(PROMOTION_PATH),
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 128)
        print("V2.9.6 ROTATING SUPERVISOR STATIC CHECK — NO API / NO ORDERS")
        print("=" * 128)
        for k, v in out.items():
            print(f"{k:68s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.6 static self-check failed: {out}")
    return out


def _read_only_preflight(*, q, hours, max_loss, min_equity, require_promotion, show):
    if require_promotion:
        _require_q50_promotion()
    static_self_check(show=show)
    return V288.live_preflight(
        quote_size=float(q),
        runtime_hours=float(hours),
        max_start_loss_usd=float(max_loss),
        min_start_equity_usd=float(min_equity),
        show=show,
        probe_private_ws=True,
    )


def rotation_smoke_preflight(*, show=True):
    return _read_only_preflight(
        q=SMOKE_Q,
        hours=SMOKE_RUNTIME_HOURS,
        max_loss=SMOKE_MAX_LOSS_USD,
        min_equity=SMOKE_MIN_EQUITY_USD,
        require_promotion=False,
        show=show,
    )


def q50_preflight(*, show=True):
    return _read_only_preflight(
        q=Q50_Q,
        hours=Q50_HOURS,
        max_loss=Q50_MAX_LOSS_USD,
        min_equity=Q50_MIN_EQUITY_USD,
        require_promotion=True,
        show=show,
    )


def _start_external_recorder(parent_session):
    parent_session = Path(parent_session).resolve()
    return __import__(
        "quant_research.kalshi.mm_deep_tail_join_ask_live_v1_8_record_m12",
        fromlist=["_start_recorder_m0_m12_auth"],
    )._start_recorder_m0_m12_auth(parent_session)


def _stop_external_recorder(pid):
    return _stop_pid(pid, sig=signal.SIGINT, timeout_s=45.0)


def _attach_raw_capture(parent_session, generation_dir):
    parent_session = Path(parent_session).resolve()
    generation_dir = Path(generation_dir).resolve()
    source = parent_session / "raw_capture"
    target = generation_dir / "raw_capture"
    if not source.exists():
        raise RuntimeError(f"Parent raw_capture is missing: {source}")
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Generation raw_capture already exists: {target}")
    os.symlink(str(source), str(target), target_is_directory=True)
    return target


def _generation_cfg(parent_cfg, *, generation_id, generation_dir, recorder_pid,
                    session_start_equity, session_kill_equity, remaining_hours):
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
        "strategy_terminal_cleanup_elapsed_s": M5_S,
        "recorder_persist_end_elapsed_s": RECORDER_M12_S,
        "recorder_label_tail_end_elapsed_s": LABEL_TAIL_END_S,
        "rotation_process_lifetime": "ONE_COMPLETE_M0_M5_WINDOW",
        "fixed_session_risk_baseline": True,
        "fresh_generation_starts_at_raw_eof": True,
        "fresh_row_after_generation_start_required": True,
        "no_auto_scale": True,
    }


def _run_generation(session, cfg_path):
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


def _fresh_generation_preflight(parent_cfg, *, remaining_hours, show=False):
    """Fresh read-only safety/fee snapshot for one trader generation.

    V2.8.2 intentionally limits fee-snapshot reuse to 300 seconds. A 12-hour
    rotating session therefore refreshes the parent PASS snapshot between
    generations instead of pretending the launch-time fee snapshot is still fresh.
    """
    return V288.live_preflight(
        quote_size=float(parent_cfg["quote_size"]),
        runtime_hours=max(0.02, float(remaining_hours)),
        max_start_loss_usd=float(parent_cfg["max_start_loss_usd"]),
        min_start_equity_usd=float(parent_cfg["min_start_equity_usd"]),
        show=bool(show),
        probe_private_ws=False,
    )


def _launch_generation(parent_session, parent_cfg, parent_preflight, *, generation_id,
                       recorder_pid, session_start_equity, session_kill_equity,
                       remaining_hours):
    parent_session = Path(parent_session).resolve()
    generation_dir = parent_session / "generations" / f"gen_{int(generation_id):04d}"
    generation_dir.mkdir(parents=True, exist_ok=False)
    _attach_raw_capture(parent_session, generation_dir)

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
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation",
        "--run-generation", str(generation_dir),
        "--config", str(cfg_path),
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

    B._append(parent_session / SUPERVISOR_EVENTS_FILE, {
        "time": B._iso(),
        "event": "GENERATION_LAUNCHED",
        "generation_id": int(generation_id),
        "generation_dir": str(generation_dir),
        "trader_pid": proc.pid,
        "remaining_hours": float(remaining_hours),
    })
    return proc, generation_dir, cfg, log


def _recover_generation_fail_closed(parent_session, generation_dir, trader_pid, *, reason):
    """Authoritative recovery. May send cancels/reduce-only cleanup orders."""
    parent_session = Path(parent_session).resolve()
    generation_dir = Path(generation_dir).resolve()
    trader_pid = int(trader_pid or 0)
    group = B._read(generation_dir / "order_group.json", {}) or {}
    gid = str(group.get("order_group_id") or "")

    client = B.Q1.LiveClient()
    group_trigger = B._trigger_group(client, gid) if gid else {
        "ok": False, "reason": "missing_order_group_id"
    }
    trader_stop = V27._terminate_pid_group(trader_pid) if trader_pid > 0 else {
        "pid": None, "dead": True, "note": "no trader pid"
    }
    fallback = B._fallback_cleanup({"session_dir": str(generation_dir)})
    time.sleep(0.30)
    state = V27._authoritative_state(client, gid)
    receipt = {
        "time": B._iso(),
        "deploy_version": DEPLOY_VERSION,
        "parent_session": str(parent_session),
        "generation_dir": str(generation_dir),
        "trader_pid": trader_pid or None,
        "reason": str(reason),
        "group_id": gid or None,
        "group_trigger": group_trigger,
        "trader_stop": trader_stop,
        "fallback_cleanup": fallback,
        "group_resting": state.get("group_resting") or [],
        "nonzero_positions": state.get("nonzero") or [],
        "all_account_resting_count": len(state.get("resting") or []),
        "recovery_verified": not (state.get("group_resting") or []) and not (state.get("nonzero") or []),
    }
    B._atomic(parent_session / RECOVERY_RECEIPT_FILE, receipt)
    B._append(parent_session / SUPERVISOR_EVENTS_FILE, {
        "time": B._iso(), "event": "FAIL_CLOSED_RECOVERY", **receipt
    })
    if not receipt["recovery_verified"]:
        raise RuntimeError(f"V2.9.6 recovery failed authoritative verification: {receipt}")
    return receipt


def _generation_health_ready(generation_dir):
    h = B._read(Path(generation_dir) / "health.json", {}) or {}
    rest = h.get("rest_fill_reconciler") or {}
    compact = h.get("watchdog_compact") or {}
    return bool(
        h.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}
        and h.get("private_ws_ready") is True
        and h.get("raw_watchdog_ready") is True
        and h.get("bounded_raw_ingestion") is True
        and h.get("bounded_book_tail_runtime_verified") is True
        and h.get("stale_orphan_guard_version") == LIVE.LIVE_VERSION
        and h.get("live_memory_hardening_version") == LIVE.LIVE_VERSION
        and rest.get("mode") == "MIN_TS_INCREMENTAL_DEDUP"
        and compact.get("mode") == "GENERATION_START_EOF_COMPACT_CERTIFIER"
        and compact.get("history_retained") is False
        and compact.get("cancel_executor_present") is False
        and h.get("external_recorder_owned_by_supervisor") is True
        and h.get("fixed_session_start_equity_usd") is not None
        and h.get("fixed_session_kill_equity_usd") is not None
        and (B._read(Path(generation_dir) / "child_fee_preflight_reuse_v2_8_2.json", {}) or {}).get("ok") is True
    ), h


def _write_supervisor_health(parent_session, *, cfg, recorder_pid, generation_id,
                             trader_pid, generation_dir, session_start_equity,
                             session_kill_equity, session_trade_start, session_deadline,
                             state, last_error=None):
    parent_session = Path(parent_session).resolve()
    raw_health = B._read(parent_session / "raw_capture" / "health.json", {}) or {}
    gen_health = B._read(Path(generation_dir) / "health.json", {}) if generation_dir else {}
    row = {
        "time": B._iso(),
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "running": state not in {"STOPPED", "FAILED"},
        "state": str(state),
        "parent_session_dir": str(parent_session),
        "mode": cfg.get("mode"),
        "quote_size": float(cfg.get("quote_size")),
        "recorder_pid": int(recorder_pid or 0) or None,
        "recorder_alive": bool(recorder_pid and B._pid_alive(recorder_pid)),
        "recorder_health": raw_health,
        "generation_id": int(generation_id or 0),
        "trader_pid": int(trader_pid or 0) or None,
        "trader_alive": bool(trader_pid and B._pid_alive(trader_pid)),
        "generation_dir": str(generation_dir) if generation_dir else None,
        "generation_health": gen_health or {},
        "session_start_equity_usd": float(session_start_equity),
        "session_kill_equity_usd": float(session_kill_equity),
        "session_trade_start": session_trade_start,
        "session_deadline": session_deadline,
        "fixed_session_risk_baseline": True,
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "last_error": last_error,
    }
    B._atomic(parent_session / SUPERVISOR_HEALTH_FILE, row)
    return row


def _wait_smoke_tail(parent_session, recorder_pid, checkpoint):
    close = pd.to_datetime((checkpoint or {}).get("window_close_time"), utc=True, errors="coerce")
    if pd.isna(close):
        raise RuntimeError(f"Smoke checkpoint missing valid window_close_time: {checkpoint}")
    target_epoch = float(close.timestamp()) - 150.0 + SMOKE_TAIL_EXTRA_GRACE_S
    survived = True
    while time.time() < target_epoch:
        if not B._pid_alive(recorder_pid):
            survived = False
            break
        time.sleep(0.25)
    raw_health = B._read(Path(parent_session) / "raw_capture" / "health.json", {}) or {}
    return {
        "target_epoch": target_epoch,
        "target_time_utc": pd.Timestamp(target_epoch, unit="s", tz="UTC").isoformat(),
        "recorder_alive_at_tail_boundary": bool(survived and B._pid_alive(recorder_pid)),
        "raw_health_at_tail_boundary": raw_health,
    }


def _smoke_tail_event_audit(parent_session, checkpoint):
    """Verify durable label_tail_end rows for every M5-verified target ticker."""
    targets = set(str(x) for x in ((checkpoint or {}).get("target_tickers") or []) if x)
    path = Path(parent_session) / "raw_capture" / "book_top3_events.jsonl"
    seen = set()
    max_elapsed = {}
    bad_rows = 0
    if path.exists() and targets:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    bad_rows += 1
                    continue
                ticker = str((row or {}).get("ticker") or "")
                if ticker not in targets:
                    continue
                e = B._f((row or {}).get("elapsed_s"), float("nan"))
                if math.isfinite(e):
                    max_elapsed[ticker] = max(float(e), float(max_elapsed.get(ticker, -1e30)))
                if str((row or {}).get("event_type") or "") == "label_tail_end":
                    seen.add(ticker)
    return {
        "target_tickers": sorted(targets),
        "label_tail_end_tickers": sorted(seen),
        "missing_label_tail_end_tickers": sorted(targets - seen),
        "all_targets_have_label_tail_end": bool(targets and seen == targets),
        "max_elapsed_by_ticker": max_elapsed,
        "json_decode_errors": int(bad_rows),
    }


def _recorder_final_audit(parent_session):
    raw = Path(parent_session) / "raw_capture"
    capture = B._read(raw / "capture_spec.json", {}) or {}
    manifest = B._read(raw / "session_manifest.json", {}) or {}
    counts = manifest.get("final_counts") or {}
    return {
        "capture_study_version": capture.get("study_version"),
        "capture_m12": capture.get("research_elapsed_seconds") == [0.0, 720.0],
        "capture_tail_750": capture.get("persisted_elapsed_seconds") == [0.0, 750.0],
        "manifest_ended": bool(manifest.get("ended_at")),
        "book_rows_positive": int(counts.get("book_rows") or 0) > 0,
        "ticker_rows_positive": int(counts.get("ticker_rows") or 0) > 0,
        "snapshots_positive": int(counts.get("snapshots_received") or 0) > 0,
        "final_counts": counts,
    }


def _write_smoke_promotion(parent_session, cfg, checkpoint, recorder_pid, tail_result, recorder_stop):
    audit = _recorder_final_audit(parent_session)
    tail_data_audit = _smoke_tail_event_audit(parent_session, checkpoint)
    bootstrap = B._read(
        Path(checkpoint.get("generation_dir", "")) / LIVE.GENERATION_BOOTSTRAP_FILE,
        {},
    ) if checkpoint.get("generation_dir") else {}
    passed = bool(
        checkpoint.get("safe_to_rotate") is True
        and tail_result.get("recorder_alive_at_tail_boundary") is True
        and recorder_stop.get("dead") is True
        and audit.get("capture_m12") is True
        and audit.get("capture_tail_750") is True
        and audit.get("manifest_ended") is True
        and audit.get("book_rows_positive") is True
        and audit.get("snapshots_positive") is True
        and tail_data_audit.get("all_targets_have_label_tail_end") is True
    )
    record = {
        "time": B._iso(),
        "passed": passed,
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "git_head": _current_head(),
        "parent_session_dir": str(Path(parent_session).resolve()),
        "smoke_quantity": float(cfg["quote_size"]),
        "rotation_checkpoint_verified": checkpoint.get("safe_to_rotate") is True,
        "same_recorder_survived_trader_rotation": tail_result.get("recorder_alive_at_tail_boundary") is True,
        "m12_plus_30_observed_after_rotation": tail_data_audit.get("all_targets_have_label_tail_end") is True,
        "external_recorder_pid": int(recorder_pid),
        "checkpoint": checkpoint,
        "tail_result": tail_result,
        "recorder_stop": recorder_stop,
        "recorder_final_audit": audit,
        "smoke_tail_data_audit": tail_data_audit,
        "generation_bootstrap": bootstrap,
    }
    B._atomic(PROMOTION_PATH, record)
    return record


def _run_supervisor(parent_session, cfg_path):
    parent_session = Path(parent_session).resolve()
    cfg = B._read(Path(cfg_path), {}) or {}
    parent_preflight = B._read(parent_session / "parent_preflight_snapshot.json", {}) or {}
    if not parent_preflight.get("ok"):
        raise RuntimeError("Supervisor parent preflight snapshot missing/not PASS")

    session_start_equity = float((parent_preflight.get("account") or {}).get("equity_usd"))
    session_kill_equity = session_start_equity - float(cfg["max_start_loss_usd"])
    B._atomic(parent_session / "session_risk_baseline_v2_9_6.json", {
        "time": B._iso(),
        "session_start_equity_usd": session_start_equity,
        "session_kill_equity_usd": session_kill_equity,
        "max_start_loss_usd": float(cfg["max_start_loss_usd"]),
        "baseline_reset_between_generations": False,
    })

    recorder_proc = None
    recorder_pid = 0
    current_proc = None
    current_dir = None
    generation_id = 0
    session_trade_start = None
    session_deadline = None
    last_error = None
    final_reason = None
    generations = []
    smoke_checkpoint = None
    recorder_stop = None

    try:
        recorder_proc, recorder_health = _start_external_recorder(parent_session)
        recorder_pid = int(recorder_proc.pid)
        B._atomic(parent_session / "external_recorder_start_v2_9_6.json", {
            "time": B._iso(), "pid": recorder_pid, "health": recorder_health,
            "study_version": REC.STUDY_VERSION,
        })
        B._append(parent_session / SUPERVISOR_EVENTS_FILE, {
            "time": B._iso(), "event": "EXTERNAL_RECORDER_STARTED", "pid": recorder_pid
        })

        while True:
            if (parent_session / SESSION_KILL_FILE).exists():
                req = B._read(parent_session / SESSION_KILL_FILE, {}) or {}
                final_reason = str(req.get("reason") or "MANUAL_KILL_AND_FLATTEN")
                break
            if not B._pid_alive(recorder_pid):
                raise RuntimeError("External recorder exited unexpectedly")
            if session_deadline is not None and time.time() >= session_deadline:
                final_reason = "RUNTIME_COMPLETE"
                break

            generation_id += 1
            if session_deadline is None:
                remaining_h = float(cfg["runtime_hours"])
            else:
                remaining_h = max(0.02, (float(session_deadline) - time.time()) / 3600.0)
            generation_preflight = _fresh_generation_preflight(
                cfg, remaining_hours=remaining_h, show=False
            )
            generation_equity = float(
                (generation_preflight.get("account") or {}).get("equity_usd")
            )
            if generation_equity <= session_kill_equity + B.EPS:
                raise RuntimeError(
                    f"Fixed session loss trigger breached before generation {generation_id}: "
                    f"current={generation_equity:.4f} kill={session_kill_equity:.4f}"
                )
            current_proc, current_dir, gen_cfg, gen_log = _launch_generation(
                parent_session,
                cfg,
                generation_preflight,
                generation_id=generation_id,
                recorder_pid=recorder_pid,
                session_start_equity=session_start_equity,
                session_kill_equity=session_kill_equity,
                remaining_hours=remaining_h,
            )

            startup_deadline = time.time() + STARTUP_TIMEOUT_S
            ready = False
            last_h = {}
            while time.time() < startup_deadline:
                if current_proc.poll() is not None:
                    break
                if not B._pid_alive(recorder_pid):
                    break
                ready, last_h = _generation_health_ready(current_dir)
                if ready:
                    break
                _write_supervisor_health(
                    parent_session, cfg=cfg, recorder_pid=recorder_pid,
                    generation_id=generation_id, trader_pid=current_proc.pid,
                    generation_dir=current_dir, session_start_equity=session_start_equity,
                    session_kill_equity=session_kill_equity,
                    session_trade_start=session_trade_start, session_deadline=session_deadline,
                    state="STARTING_GENERATION",
                )
                time.sleep(0.20)
            if not ready:
                raise RuntimeError(
                    f"Generation {generation_id} startup failed/timeout rc={current_proc.poll()} "
                    f"health={last_h} log={_tail_text(gen_log)}"
                )

            while current_proc.poll() is None:
                if not B._pid_alive(recorder_pid):
                    raise RuntimeError("External recorder died while trader was active")
                gh = B._read(current_dir / "health.json", {}) or {}
                if session_trade_start is None:
                    ts = B._f(gh.get("trade_start"), float("nan"))
                    if math.isfinite(ts):
                        session_trade_start = float(ts)
                        session_deadline = session_trade_start + float(cfg["runtime_hours"]) * 3600.0
                        B._append(parent_session / SUPERVISOR_EVENTS_FILE, {
                            "time": B._iso(), "event": "SESSION_CLOCK_STARTED",
                            "trade_start": session_trade_start, "deadline": session_deadline,
                        })

                if (parent_session / SESSION_KILL_FILE).exists():
                    req = B._read(parent_session / SESSION_KILL_FILE, {}) or {}
                    B._atomic(current_dir / "KILL_REQUEST.json", {
                        "time": B._iso(), "reason": str(req.get("reason") or "MANUAL_KILL_AND_FLATTEN")
                    })
                elif session_deadline is not None and time.time() >= session_deadline:
                    B._atomic(current_dir / "KILL_REQUEST.json", {
                        "time": B._iso(), "reason": "RUNTIME_COMPLETE"
                    })

                _write_supervisor_health(
                    parent_session, cfg=cfg, recorder_pid=recorder_pid,
                    generation_id=generation_id, trader_pid=current_proc.pid,
                    generation_dir=current_dir, session_start_equity=session_start_equity,
                    session_kill_equity=session_kill_equity,
                    session_trade_start=session_trade_start, session_deadline=session_deadline,
                    state="RUNNING_GENERATION",
                )
                time.sleep(SUPERVISOR_POLL_S)

            rc = current_proc.returncode
            final = B._read(current_dir / "final_summary.json", {}) or {}
            checkpoint = B._read(current_dir / LIVE.ROTATION_CHECKPOINT_FILE, {}) or {}
            generation_record = {
                "generation_id": generation_id,
                "generation_dir": str(current_dir),
                "trader_pid": current_proc.pid,
                "returncode": rc,
                "final": final,
                "checkpoint": checkpoint,
            }
            generations.append(generation_record)
            B._append(parent_session / SUPERVISOR_EVENTS_FILE, {
                "time": B._iso(), "event": "GENERATION_EXITED",
                "generation_id": generation_id, "returncode": rc,
                "shutdown_reason": final.get("shutdown_reason"),
                "safe_to_rotate": checkpoint.get("safe_to_rotate"),
            })

            parent_kill = (parent_session / SESSION_KILL_FILE).exists()
            deadline_reached = session_deadline is not None and time.time() >= session_deadline
            if parent_kill or deadline_reached:
                expected = {
                    "RUNTIME_COMPLETE", "MANUAL_KILL_AND_FLATTEN", "GUARDIAN_RSS_HARD_LIMIT",
                    "GUARDIAN_SUPERVISOR_FAILURE", "SUPERVISOR_STOP_REQUEST",
                }
                if rc == 0 and _is_clean_final(final) and str(final.get("shutdown_reason") or "") in expected:
                    final_reason = str(final.get("shutdown_reason") or "RUNTIME_COMPLETE")
                    break
                _recover_generation_fail_closed(
                    parent_session, current_dir, current_proc.pid,
                    reason=f"TERMINATION_PATH_NOT_CLEAN:rc={rc}:final={final.get('shutdown_reason')}"
                )
                final_reason = "FAIL_CLOSED_RECOVERED_AT_SESSION_END"
                break

            if not (
                rc == 0
                and checkpoint.get("safe_to_rotate") is True
                and _is_clean_final(final, allowed_reasons={"GENERATION_ROTATION_M5_VERIFIED"})
            ):
                _recover_generation_fail_closed(
                    parent_session, current_dir, current_proc.pid,
                    reason=f"ABNORMAL_GENERATION_EXIT:rc={rc}:final={final.get('shutdown_reason')}:checkpoint={checkpoint.get('reason')}"
                )
                final_reason = "ABNORMAL_GENERATION_FAIL_CLOSED_RECOVERED"
                break

            checkpoint = dict(checkpoint)
            checkpoint["generation_dir"] = str(current_dir)
            if bool(cfg.get("rotation_smoke")):
                smoke_checkpoint = checkpoint
                tail_result = _wait_smoke_tail(parent_session, recorder_pid, checkpoint)
                if not tail_result.get("recorder_alive_at_tail_boundary"):
                    raise RuntimeError("Rotation smoke recorder did not survive through M12+30")
                recorder_stop = _stop_external_recorder(recorder_pid)
                promotion = _write_smoke_promotion(
                    parent_session, cfg, checkpoint, recorder_pid, tail_result, recorder_stop
                )
                if not promotion.get("passed"):
                    raise RuntimeError(f"Rotation smoke promotion audit failed: {promotion}")
                final_reason = "ROTATION_SMOKE_PASSED"
                break

            current_proc = None
            current_dir = None

        if current_proc is not None and current_proc.poll() is None:
            B._atomic(current_dir / "KILL_REQUEST.json", {
                "time": B._iso(), "reason": final_reason or "SUPERVISOR_STOP_REQUEST"
            })
            deadline = time.time() + 20.0
            while current_proc.poll() is None and time.time() < deadline:
                time.sleep(0.20)
            if current_proc.poll() is None:
                _recover_generation_fail_closed(
                    parent_session, current_dir, current_proc.pid,
                    reason=final_reason or "SUPERVISOR_STOP_REQUEST_TIMEOUT",
                )

    except BaseException as exc:
        last_error = repr(exc)
        B._append(parent_session / SUPERVISOR_EVENTS_FILE, {
            "time": B._iso(), "event": "SUPERVISOR_ERROR", "error": last_error
        })
        if current_proc is not None and current_dir is not None:
            try:
                _recover_generation_fail_closed(
                    parent_session, current_dir, current_proc.pid,
                    reason=f"SUPERVISOR_EXCEPTION:{last_error}",
                )
            except Exception as recovery_exc:
                B._append(parent_session / SUPERVISOR_EVENTS_FILE, {
                    "time": B._iso(), "event": "SUPERVISOR_RECOVERY_ERROR",
                    "error": repr(recovery_exc),
                })
        final_reason = final_reason or "SUPERVISOR_EXCEPTION"
        raise
    finally:
        if recorder_pid and B._pid_alive(recorder_pid):
            recorder_stop = _stop_external_recorder(recorder_pid)
        recorder_audit = _recorder_final_audit(parent_session) if (parent_session / "raw_capture").exists() else {}
        final = {
            "time": B._iso(),
            "deploy_version": DEPLOY_VERSION,
            "live_version": LIVE.LIVE_VERSION,
            "parent_session_dir": str(parent_session),
            "mode": cfg.get("mode"),
            "quote_size": float(cfg.get("quote_size")),
            "shutdown_reason": final_reason,
            "session_start_equity_usd": session_start_equity,
            "session_kill_equity_usd": session_kill_equity,
            "session_trade_start": session_trade_start,
            "session_deadline": session_deadline,
            "generations_completed": len(generations),
            "generations": generations,
            "recorder_pid": recorder_pid or None,
            "recorder_stop": recorder_stop,
            "recorder_final_audit": recorder_audit,
            "last_error": last_error,
            "rotation_smoke_checkpoint": smoke_checkpoint,
        }
        B._atomic(parent_session / SUPERVISOR_FINAL_FILE, final)
        _write_supervisor_health(
            parent_session, cfg=cfg, recorder_pid=recorder_pid,
            generation_id=generation_id, trader_pid=(current_proc.pid if current_proc else 0),
            generation_dir=current_dir, session_start_equity=session_start_equity,
            session_kill_equity=session_kill_equity,
            session_trade_start=session_trade_start, session_deadline=session_deadline,
            state="FAILED" if last_error else "STOPPED", last_error=last_error,
        )
        ctl = B._read(CORE.CONTROL_PATH, {}) or {}
        if str(ctl.get("session_dir") or "") == str(parent_session):
            ctl.update({
                "running": False,
                "stopped_at": B._iso(),
                "shutdown_reason": final_reason,
                "last_error": last_error,
            })
            B._atomic(CORE.CONTROL_PATH, ctl)


def _guardian_intervene(parent_session, supervisor_pid, health, *, reason, peak_group, peak_recorder):
    parent_session = Path(parent_session).resolve()
    B._atomic(parent_session / SESSION_KILL_FILE, {"time": B._iso(), "reason": str(reason)})
    trader_pid = int((health or {}).get("trader_pid") or 0)
    generation_dir = (health or {}).get("generation_dir")
    recorder_pid = int((health or {}).get("recorder_pid") or 0)

    deadline = time.time() + GUARDIAN_KILL_GRACE_S
    while trader_pid > 0 and B._pid_alive(trader_pid) and time.time() < deadline:
        time.sleep(0.20)

    recovery = None
    recovery_error = None
    if trader_pid > 0 and B._pid_alive(trader_pid) and generation_dir:
        try:
            recovery = _recover_generation_fail_closed(
                parent_session, Path(generation_dir), trader_pid,
                reason=str(reason),
            )
        except Exception as exc:
            recovery_error = repr(exc)

    supervisor_stop = None
    if B._pid_alive(supervisor_pid):
        try:
            supervisor_stop = V27._terminate_pid_group(supervisor_pid)
        except Exception as exc:
            supervisor_stop = {"error": repr(exc), "dead": False}
    recorder_stop = _stop_external_recorder(recorder_pid) if recorder_pid else {"dead": True, "pid": None}
    receipt = {
        "time": B._iso(),
        "deploy_version": DEPLOY_VERSION,
        "intervened": True,
        "reason": str(reason),
        "promotion_allowed": False,
        "peak_trader_group_rss_mb": float(peak_group),
        "peak_recorder_rss_mb": float(peak_recorder),
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "recovery": recovery,
        "recovery_error": recovery_error,
        "supervisor_stop": supervisor_stop,
        "recorder_stop": recorder_stop,
    }
    B._atomic(parent_session / GUARDIAN_RECEIPT_FILE, receipt)
    B._atomic(parent_session / GUARDIAN_HEALTH_FILE, {**receipt, "running": False})
    return receipt


def _guardian_loop(parent_session, supervisor_pid):
    parent_session = Path(parent_session).resolve()
    supervisor_pid = int(supervisor_pid)
    peak_group = 0.0
    peak_recorder = 0.0
    warning_for_generation = None

    while True:
        final = B._read(parent_session / SUPERVISOR_FINAL_FILE, {}) or {}
        health = B._read(parent_session / SUPERVISOR_HEALTH_FILE, {}) or {}
        if final and not B._pid_alive(supervisor_pid):
            receipt = {
                "time": B._iso(),
                "deploy_version": DEPLOY_VERSION,
                "intervened": False,
                "reason": "SUPERVISOR_FINALIZED",
                "promotion_allowed": final.get("last_error") in (None, ""),
                "peak_trader_group_rss_mb": peak_group,
                "peak_recorder_rss_mb": peak_recorder,
                "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
            }
            B._atomic(parent_session / GUARDIAN_RECEIPT_FILE, receipt)
            B._atomic(parent_session / GUARDIAN_HEALTH_FILE, {**receipt, "running": False})
            return receipt

        if not B._pid_alive(supervisor_pid):
            return _guardian_intervene(
                parent_session, supervisor_pid, health,
                reason="GUARDIAN_SUPERVISOR_FAILURE",
                peak_group=peak_group, peak_recorder=peak_recorder,
            )

        trader_pid = int(health.get("trader_pid") or 0)
        recorder_pid = int(health.get("recorder_pid") or 0)
        generation_id = int(health.get("generation_id") or 0)
        rss = V27._rss_snapshot(trader_pid, recorder_pid) if trader_pid > 0 else {
            "strategy_group_rss_mb": 0.0,
            "recorder_rss_mb": 0.0,
            "total_rss_mb": 0.0,
            "group_processes": [],
        }
        group_mb = float(rss.get("strategy_group_rss_mb") or 0.0)
        recorder_mb = float(rss.get("recorder_rss_mb") or 0.0)
        peak_group = max(peak_group, group_mb)
        peak_recorder = max(peak_recorder, recorder_mb)

        gh = {
            "time": B._iso(),
            "deploy_version": DEPLOY_VERSION,
            "running": True,
            "intervened": False,
            "supervisor_pid": supervisor_pid,
            "generation_id": generation_id,
            "trader_pid": trader_pid or None,
            "recorder_pid": recorder_pid or None,
            "rss": rss,
            "peak_trader_group_rss_mb": peak_group,
            "peak_recorder_rss_mb": peak_recorder,
            "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
            "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
            "session_deadline": health.get("session_deadline"),
        }
        B._atomic(parent_session / GUARDIAN_HEALTH_FILE, gh)

        if group_mb >= GENERATION_RSS_WARNING_MB and warning_for_generation != generation_id:
            warning_for_generation = generation_id
            B._append(parent_session / GUARDIAN_EVENTS_FILE, {
                "time": B._iso(), "event": "TRADER_RSS_WARNING",
                "generation_id": generation_id, "rss": rss,
                "warning_mb": GENERATION_RSS_WARNING_MB,
                "hard_mb": GENERATION_RSS_HARD_LIMIT_MB,
            })
        if group_mb >= GENERATION_RSS_HARD_LIMIT_MB:
            B._append(parent_session / GUARDIAN_EVENTS_FILE, {
                "time": B._iso(), "event": "TRADER_RSS_HARD_LIMIT",
                "generation_id": generation_id, "rss": rss,
            })
            return _guardian_intervene(
                parent_session, supervisor_pid, health,
                reason="GUARDIAN_RSS_HARD_LIMIT",
                peak_group=peak_group, peak_recorder=peak_recorder,
            )
        time.sleep(GUARDIAN_POLL_S)


def _launch_guardian(parent_session, supervisor_pid):
    parent_session = Path(parent_session).resolve()
    log = parent_session / "guardian_v2_9_6.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation",
        "--run-guardian", str(parent_session),
        "--supervisor-pid", str(int(supervisor_pid)),
    ]
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(V28.C.PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()
    return proc, log, cmd


def _launch_supervised(*, q, hours, max_loss, min_equity, mode, rotation_smoke,
                       arm_phrase, expected_arm, require_promotion):
    if str(arm_phrase) != str(expected_arm):
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly."
        )
    if require_promotion:
        _require_q50_promotion()

    static_self_check(show=True)
    V28._patch_parent()
    V28.D._guard_other_live_processes()
    pre = V288.live_preflight(
        quote_size=float(q), runtime_hours=float(hours),
        max_start_loss_usd=float(max_loss), min_start_equity_usd=float(min_equity),
        show=True, probe_private_ws=True,
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
        "recorder_version": REC.STUDY_VERSION,
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "q50_promotion_required": bool(require_promotion),
        "fixed_session_risk_baseline": True,
        "no_auto_scale": True,
    }
    cfg_path = parent_session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(parent_session / "parent_preflight_snapshot.json", pre)
    B._atomic(parent_session / "architecture_spec_v2_9_6.json", {
        "time": B._iso(),
        "architecture": "ONE_EXTERNAL_M0_M12_RECORDER_PLUS_ONE_M0_M5_TRADER_PROCESS_PER_WINDOW",
        "generation_start_raw_policy": "CURRENT_EOF_PLUS_REQUIRE_FRESH_POST_START_ROW",
        "rotation_gate": "M5_ZERO_POSITION_ZERO_GROUP_RESTING_DURABLE_CHECKPOINT",
        "risk_baseline": "FIXED_ONCE_PER_PARENT_SESSION",
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "q50_requires_rotation_smoke_same_git_head": True,
        "strategy_rule_change": "NONE",
    })

    log = parent_session / "supervisor_v2_9_6.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation",
        "--run-supervisor", str(parent_session),
        "--config", str(cfg_path),
    ]
    caffeinate = shutil.which("caffeinate")
    cmd = ([caffeinate, "-i", "-m"] + child) if caffeinate else child
    try:
        supervisor = subprocess.Popen(
            cmd, cwd=str(V28.C.PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
                f"V2.9.6 supervisor exited during startup rc={supervisor.returncode}\n{_tail_text(log)}"
            )
        last = B._read(parent_session / SUPERVISOR_HEALTH_FILE, {}) or {}
        gen_dir = last.get("generation_dir")
        gen_ready = False
        if gen_dir:
            gen_ready, _ = _generation_health_ready(Path(gen_dir))
        recorder_ok = (
            last.get("recorder_alive") is True
            and (last.get("recorder_health") or {}).get("running") is True
            and (last.get("recorder_health") or {}).get("healthy") is True
        )
        if recorder_ok and gen_ready:
            break
        time.sleep(0.25)
    else:
        B._atomic(parent_session / SESSION_KILL_FILE, {
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V296"
        })
        raise RuntimeError(
            f"V2.9.6 startup timeout. supervisor_health={last}\n{_tail_text(log)}"
        )

    print("\n" + "=" * 128)
    print("REAL-MONEY V2.9.6 ROTATING SUPERVISOR ARMED")
    print("=" * 128)
    print("Parent session:              ", parent_session)
    print("Supervisor PID:              ", supervisor.pid)
    print("Guardian PID:                ", guardian.pid)
    print("External recorder PID:       ", last.get("recorder_pid"))
    print("Current trader PID:          ", last.get("trader_pid"))
    print("Engine:                      ", LIVE.LIVE_VERSION)
    print("Quantity:                    ", f"Q{float(q):g}")
    print("Strategy per generation:     M1 -> M5 only")
    print("Recorder:                    one parent-owned M0 -> M12 + 30s process")
    print("Trader generation policy:    rotate after verified M5 flat/resting-zero checkpoint")
    print("Session risk baseline:       fixed across generations")
    print("Trader RSS warning/hard:     450 / 750 MB")
    print("Q50 promotion gate:          ", "REQUIRED+PASSED" if require_promotion else "SMOKE MODE")
    print("=" * 128)
    return live_status(show=False, tail_lines=20)


def start_rotation_smoke_q1(*, arm_phrase=None):
    return _launch_supervised(
        q=SMOKE_Q, hours=SMOKE_RUNTIME_HOURS,
        max_loss=SMOKE_MAX_LOSS_USD, min_equity=SMOKE_MIN_EQUITY_USD,
        mode="DEEP_TAIL_Q1_ROTATION_SMOKE_V296",
        rotation_smoke=True,
        arm_phrase=arm_phrase, expected_arm=ROTATION_SMOKE_ARM,
        require_promotion=False,
    )


def start_q50(*, arm_phrase=None, runtime_hours=Q50_HOURS,
              max_start_loss_usd=Q50_MAX_LOSS_USD,
              min_start_equity_usd=Q50_MIN_EQUITY_USD):
    if abs(float(runtime_hours) - Q50_HOURS) > 1e-12:
        raise RuntimeError("Q50 V2.9.6 is fixed to exactly 12.0 hours.")
    if abs(float(max_start_loss_usd) - Q50_MAX_LOSS_USD) > 1e-12:
        raise RuntimeError("Q50 V2.9.6 is fixed to exactly a $20 software loss trigger.")
    if float(min_start_equity_usd) + 1e-12 < Q50_MIN_EQUITY_USD:
        raise RuntimeError("Q50 V2.9.6 requires minimum starting equity of at least $125.")
    return _launch_supervised(
        q=Q50_Q, hours=Q50_HOURS,
        max_loss=Q50_MAX_LOSS_USD, min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q50_M1_M5_RECORD_M12_12H_V296_ROTATION",
        rotation_smoke=False,
        arm_phrase=arm_phrase, expected_arm=Q50_ARM,
        require_promotion=True,
    )


def live_status(*, show=True, tail_lines=30):
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    if not ctl:
        out = {"running": False, "message": "No deep-tail live control file."}
        if show:
            print(out)
        return out
    parent = Path(ctl.get("session_dir") or "")
    sh = B._read(parent / SUPERVISOR_HEALTH_FILE, {}) or {}
    sf = B._read(parent / SUPERVISOR_FINAL_FILE, {}) or {}
    gh = B._read(parent / GUARDIAN_HEALTH_FILE, {}) or {}
    gr = B._read(parent / GUARDIAN_RECEIPT_FILE, {}) or {}
    gen_dir = sh.get("generation_dir")
    gen_health = B._read(Path(gen_dir) / "health.json", {}) if gen_dir else {}
    gen_final = B._read(Path(gen_dir) / "final_summary.json", {}) if gen_dir else {}
    checkpoint = B._read(Path(gen_dir) / LIVE.ROTATION_CHECKPOINT_FILE, {}) if gen_dir else {}
    running = bool(B._pid_alive(ctl.get("supervisor_pid") or ctl.get("pid")) and not sf)
    tail = _tail_text(parent / "supervisor_v2_9_6.log", chars=12000).splitlines()[-int(tail_lines):]
    out = {
        "running": running,
        "parent_session_dir": str(parent),
        "supervisor_pid": ctl.get("supervisor_pid") or ctl.get("pid"),
        "guardian_pid": ctl.get("guardian_pid"),
        "supervisor_health": sh,
        "guardian_health": gh,
        "guardian_receipt": gr,
        "current_generation_health": gen_health,
        "current_generation_final": gen_final,
        "current_rotation_checkpoint": checkpoint,
        "supervisor_final": sf,
        "promotion": rotation_promotion_status(show=False),
        "log_tail": tail,
    }
    if show:
        print("=" * 128)
        print("V2.9.6 ROTATING LIVE STATUS")
        print("=" * 128)
        print("running:", running)
        print("parent session:", parent)
        print("supervisor PID:", out["supervisor_pid"])
        print("guardian PID:", out["guardian_pid"])
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
        if gen_health:
            print("generation state:", gen_health.get("state"))
            print("rotation window:", gen_health.get("rotation_window_key"))
            print("positions:", gen_health.get("positions"))
            print("active tracks:", len(gen_health.get("active_tracks") or {}))
            print("watchdog:", gen_health.get("watchdog_compact"))
        if checkpoint:
            print("rotation safe:", checkpoint.get("safe_to_rotate"), checkpoint.get("reason"))
        if sf:
            print("supervisor shutdown:", sf.get("shutdown_reason"))
            print("generations completed:", sf.get("generations_completed"))
            print("last error:", sf.get("last_error"))
        print("Q50 promotion ready:", out["promotion"].get("ready_for_q50"))
        if tail:
            print("\nSUPERVISOR LOG TAIL")
            print("\n".join(tail))
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=25.0):
    """REAL cancels/reduce-only cleanup may be sent."""
    if str(arm_phrase) != KILL_ARM:
        raise RuntimeError(f"Pass arm_phrase={KILL_ARM!r} exactly.")
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    if not ctl:
        raise RuntimeError("No V2.9.6 live session control file.")
    parent = Path(ctl.get("session_dir") or "")
    B._atomic(parent / SESSION_KILL_FILE, {
        "time": B._iso(), "reason": "MANUAL_KILL_AND_FLATTEN"
    })
    deadline = time.time() + float(wait_s)
    while time.time() < deadline:
        st = live_status(show=False)
        if not st.get("running"):
            print("V2.9.6 supervisor stopped after kill request.")
            return st
        time.sleep(0.5)

    sh = B._read(parent / SUPERVISOR_HEALTH_FILE, {}) or {}
    trader_pid = int(sh.get("trader_pid") or 0)
    gen_dir = sh.get("generation_dir")
    if trader_pid and gen_dir:
        result = _recover_generation_fail_closed(
            parent, Path(gen_dir), trader_pid,
            reason="MANUAL_KILL_TIMEOUT_DIRECT_RECOVERY",
        )
    else:
        result = {"note": "no active trader generation to recover"}
    recorder_pid = int(sh.get("recorder_pid") or 0)
    recorder_stop = _stop_external_recorder(recorder_pid) if recorder_pid else {"dead": True}
    supervisor_pid = int(ctl.get("supervisor_pid") or ctl.get("pid") or 0)
    supervisor_stop = V27._terminate_pid_group(supervisor_pid) if supervisor_pid and B._pid_alive(supervisor_pid) else {"dead": True}
    out = {"recovery": result, "recorder_stop": recorder_stop, "supervisor_stop": supervisor_stop}
    print(json.dumps(out, indent=2, default=str))
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-generation")
    ap.add_argument("--run-supervisor")
    ap.add_argument("--run-guardian")
    ap.add_argument("--supervisor-pid", type=int)
    ap.add_argument("--config")
    a = ap.parse_args()
    if a.run_generation:
        if not a.config:
            raise RuntimeError("--config is required with --run-generation")
        _run_generation(Path(a.run_generation), Path(a.config))
    elif a.run_supervisor:
        if not a.config:
            raise RuntimeError("--config is required with --run-supervisor")
        _run_supervisor(Path(a.run_supervisor), Path(a.config))
    elif a.run_guardian:
        if not a.supervisor_pid:
            raise RuntimeError("--supervisor-pid is required with --run-guardian")
        _guardian_loop(Path(a.run_guardian), int(a.supervisor_pid))
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
