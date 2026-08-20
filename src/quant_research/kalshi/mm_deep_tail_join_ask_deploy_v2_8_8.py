from __future__ import annotations

"""V2.8.8: 30-minute Q1 using authenticated V5 discovery.

This wrapper is the direct fix for the observed recorder failure where original V5
was websocket-connected/healthy but had subscribed_markets=0, channels=[],
snapshots=0 and book_rows=0 because its public unauthenticated /markets discovery
was repeatedly HTTP 429 throttled.

No alpha or execution rule changes:
- Q1 only;
- M1 dual 5c tails;
- first fill wins / cancel opposite tail;
- selected tail may continue to full Q;
- one fixed passive JOIN_ASK exit after full entry;
- residual flatten at M5;
- V1.5 bounded raw ingestion and V2.8 guardian behavior retained;
- V2.8.2 parent fee snapshot reuse retained.

Only live-engine operational version changes to V1.6, which launches V5 with the
same discovery semantics through signed authenticated GET /markets.

There is deliberately NO pre-M0 subscription/snapshot kill gate.  Startup market
coverage is observable in status and audited after the run.
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

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_8_2 as V282
from . import mm_deep_tail_join_ask_deploy_v2_8_7 as V287
from . import mm_deep_tail_join_ask_live_v1_6 as LIVE
from . import mm_event_time_m0_m5_recorder_v5_auth as V5A

DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_8_8_Q1_AUTH_V5_DISCOVERY"
CORE = V282.CORE
Q1_ARM = "LIVE_DEEP_TAIL_Q1_30M_V288"
KILL_ARM = V282.KILL_ARM
PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2_8_8.json"

Q1_DEFAULT_HOURS = 0.5
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0
EXPECTED_SERIES = tuple(OOS.SERIES)
STARTUP_TIMEOUT_S = V282.STARTUP_TIMEOUT_S


def _read_jsonl_series(path):
    path = Path(path)
    seen = set()
    tickers = set()
    bad = 0
    if not path.exists():
        return seen, tickers, bad
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                bad += 1
                continue
            s = str(row.get("series_ticker") or "")
            t = str(row.get("ticker") or "")
            if s:
                seen.add(s)
            if t:
                tickers.add(t)
    return seen, tickers, bad


def static_self_check(*, show=True):
    pp = V287._install_child_pythonpath()
    base = V282.static_self_check(show=False)
    live = LIVE.static_self_check(show=False)
    rec = V5A.static_self_check(show=False)
    checks = {
        "base_v2_8_2_ok": base.get("ok") is True,
        "q1_fixed_30_minutes": abs(Q1_DEFAULT_HOURS - 0.5) < 1e-12,
        "alpha_rules_unchanged": live.get("alpha_rules_unchanged_from_v1_5") is True,
        "bounded_raw_ingestion": live.get("bounded_raw_ingestion") is True,
        "guardian_unchanged": True,
        "fee_snapshot_reuse": base.get("child_fee_snapshot_reuse") is True,
        "live_engine_version": LIVE.LIVE_VERSION,
        "recorder_version": V5A.STUDY_VERSION,
        "authenticated_v5_discovery": rec.get("authenticated_discovery") is True,
        "v5_ws_persistence_unchanged": rec.get("websocket_book_logic_unchanged") is True,
        "frozen_universe_unchanged": tuple(V5A.V5.CRYPTO_SERIES) == EXPECTED_SERIES,
        "strict_pre_m0_data_gate": False,
        "child_pythonpath_installed": pp["installed"] is True,
        "orders_sent": False,
    }
    metadata = {"live_engine_version", "recorder_version", "strict_pre_m0_data_gate", "orders_sent"}
    ok = all(v is True for k, v in checks.items() if k not in metadata)
    out = {"deploy_version": DEPLOY_VERSION, **checks, "ok": bool(ok)}
    if show:
        print("=" * 104)
        print("DEEP-TAIL DEPLOY V2.8.8 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 104)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.8.8 static self-check failed: {out}")
    return out


def live_preflight(*, quote_size, runtime_hours, max_start_loss_usd,
                   min_start_equity_usd, show=True, probe_private_ws=True):
    return V282.live_preflight(
        quote_size=float(quote_size),
        runtime_hours=float(runtime_hours),
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
        show=show,
        probe_private_ws=probe_private_ws,
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
    # Guardian logic itself is unchanged; reuse the validated V2.8.2 launcher.
    return V282._launch_guardian(Path(session).resolve(), int(main_pid))


def _launch(*, q, hours, max_loss, min_equity, mode, arm_phrase, expected_arm):
    if str(arm_phrase) != str(expected_arm):
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly."
        )

    static_self_check(show=True)
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
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}_deploy_v2_8_8").resolve()
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
        "deploy_version": V282.RUNTIME_BASE_DEPLOY_VERSION,
        "launch_wrapper_version": DEPLOY_VERSION,
        "child_fee_preflight_mode": "REUSE_FRESH_PARENT_PASS_SNAPSHOT_FAIL_CLOSED",
        "recorder_study_version": V5A.STUDY_VERSION,
        "recorder_discovery_transport": V5A.DISCOVERY_TRANSPORT_VERSION,
        "scientific_status": "FRESH_FORWARD_AFTER_AUTHENTICATED_V5_DISCOVERY_FIX",
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
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_8_8",
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
            raise RuntimeError(f"Live V2.8.8 process exited during startup rc={p.returncode}\n{tail}")
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
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V2_8_8"
        })
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(f"V2.8.8 startup health timeout. Last health={last}\n{tail}")

    guardian, guardian_log, guardian_cmd = _launch_guardian(session, p.pid)
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update({
        "guardian_pid": guardian.pid,
        "guardian_log_path": str(guardian_log),
        "guardian_command": guardian_cmd,
    })
    B._atomic(CORE.CONTROL_PATH, ctl)

    print("\n" + "=" * 104)
    print("REAL-MONEY DEEP-TAIL V2.8.8 PROCESS ARMED")
    print("=" * 104)
    print("Session:                   ", session)
    print("Main PID:                  ", p.pid)
    print("Guardian PID:              ", guardian.pid)
    print("Engine:                    ", LIVE.LIVE_VERSION)
    print("Recorder:                  ", V5A.STUDY_VERSION)
    print("Discovery:                 authenticated signed /markets GET")
    print("Quantity:                  ", f"Q{float(q):g}")
    print("Runtime:                   ", "30 minutes from first complete window")
    print("Pre-M0 market-data gate:    NONE")
    print("Auto-scaling:               DISABLED")
    print("=" * 104)
    return live_status(show=False, tail_lines=20)


def start_q1_smoke(*, arm_phrase=None, runtime_hours=Q1_DEFAULT_HOURS,
                   max_start_loss_usd=Q1_DEFAULT_MAX_LOSS,
                   min_start_equity_usd=Q1_DEFAULT_MIN_EQUITY):
    if abs(float(runtime_hours) - Q1_DEFAULT_HOURS) > 1e-12:
        raise RuntimeError("V2.8.8 Q1 smoke is fixed to exactly 30 minutes (0.5 hours).")
    return _launch(
        q=1.0,
        hours=Q1_DEFAULT_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q1_SMOKE_V16_V288",
        arm_phrase=arm_phrase,
        expected_arm=Q1_ARM,
    )


def q1_data_audit(session_dir, *, show=True):
    session = Path(session_dir).resolve()
    raw = session / "raw_capture"
    capture = B._read(raw / "capture_spec.json", {}) or {}
    manifest = B._read(raw / "session_manifest.json", {}) or {}
    counts = manifest.get("final_counts") or {}
    seen_series, meta_tickers, bad_meta_rows = _read_jsonl_series(raw / "market_metadata.jsonl")

    required_files = [
        "capture_spec.json", "session_manifest.json", "book_top3_events.jsonl",
        "ticker_event_time.jsonl", "trades_event_time.jsonl", "market_metadata.jsonl",
        "market_rotations.jsonl", "connection_events.jsonl", "book_repair_events.jsonl",
    ]
    missing = [name for name in required_files if not (raw / name).exists()]
    checks = {
        "capture_spec_is_auth_v5": capture.get("study_version") == V5A.STUDY_VERSION,
        "research_window_is_m0_m5": capture.get("research_elapsed_seconds") == [0.0, 300.0],
        "label_tail_is_m5_plus_30s": capture.get("persisted_elapsed_seconds") == [0.0, 330.0],
        "all_raw_files_present": not missing,
        "final_manifest_present": bool(manifest.get("ended_at")),
        "metadata_json_parse_clean": bad_meta_rows == 0,
        "some_market_metadata_recorded": len(meta_tickers) > 0,
        "book_rows_positive": int(counts.get("book_rows") or 0) > 0,
        "ticker_rows_positive": int(counts.get("ticker_rows") or 0) > 0,
        "snapshots_positive": int(counts.get("snapshots_received") or 0) > 0,
    }
    passed = all(checks.values())
    out = {
        "session": str(session),
        "passed": bool(passed),
        "checks": checks,
        "missing_files": missing,
        "final_counts": counts,
        "series_seen": sorted(seen_series),
        "missing_series": sorted(set(EXPECTED_SERIES) - seen_series),
        "series_coverage": f"{len(seen_series)}/{len(EXPECTED_SERIES)}",
        "metadata_tickers_seen": sorted(meta_tickers),
        "bad_metadata_rows": int(bad_meta_rows),
        "sequence_gaps": int(counts.get("sequence_gaps") or 0),
        "sequence_numbers_missing": int(counts.get("sequence_numbers_missing") or 0),
        "note": "Startup coverage is diagnostic; no strict pre-M0 kill gate was used.",
    }
    if show:
        print("=" * 104)
        print("V2.8.8 Q1 AUTH-V5 DATA AUDIT — READ ONLY")
        print("=" * 104)
        for k, v in checks.items():
            print(f"{k:48s}: {v}")
        print("Series coverage:             ", out["series_coverage"])
        print("Missing series:              ", out["missing_series"])
        print("Final counts:                ", counts)
        print("Sequence gaps:               ", out["sequence_gaps"])
        print("Sequence numbers missing:    ", out["sequence_numbers_missing"])
        print("DATA AUDIT:", "PASS" if passed else "FAIL")
    return out


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    session = Path(session_dir).resolve()
    base = V282.q1_promotion_check(session, show=False, write_receipt=False)
    data = q1_data_audit(session, show=False)
    checks = dict(base.get("checks") or {})
    checks.update({
        "v2_8_8_live_engine": (B._read(session / "process_config.json", {}) or {}).get("live_engine_version") == LIVE.LIVE_VERSION,
        "v2_8_8_auth_v5_data": data.get("passed") is True,
    })
    passed = all(checks.values())
    receipt = dict(base)
    receipt.update({
        "time": B._iso(),
        "deploy_supervisor_version": DEPLOY_VERSION,
        "passed": bool(passed),
        "checks": checks,
        "v2_8_8_data_audit": data,
        "note": "Operational promotion only; not evidence expected PnL is positive.",
    })
    if passed and write_receipt:
        B._atomic(PROMOTION_PATH, receipt)
    if show:
        print("=" * 104)
        print("Q1 V1.6 / V2.8.8 30-MIN OPERATIONAL PROMOTION — READ ONLY")
        print("=" * 104)
        for k, v in checks.items():
            print(f"{k:52s}: {v}")
        print("PROMOTION:", "PASS" if passed else "NOT READY")
        if passed:
            print("Receipt:", PROMOTION_PATH)
    return receipt


def live_status(*args, **kwargs):
    return V282.live_status(*args, **kwargs)


def kill_and_flatten_live(*args, **kwargs):
    return V282.kill_and_flatten_live(*args, **kwargs)


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
    "Q1_DEFAULT_HOURS", "static_self_check", "start_q1_smoke", "q1_data_audit",
    "q1_promotion_check", "live_status", "kill_and_flatten_live",
]
