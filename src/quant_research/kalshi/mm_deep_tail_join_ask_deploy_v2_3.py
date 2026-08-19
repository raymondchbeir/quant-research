from __future__ import annotations

"""V2.3 staged deployment for the V1.2 verified-M5 deep-tail engine.

This launcher supersedes V2/V2.2 after the first real Q1 smoke exposed an M5 audit
race and recursive shutdown bug. Q5 remains hard-gated behind a NEW completed Q1
smoke on this exact git HEAD and V1.2 engine version.

Importing this module sends no orders. Real-money starts require exact arm phrases.
"""

import argparse
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
from . import mm_deep_tail_join_ask_live_v1 as CORE
from . import mm_deep_tail_join_ask_live_v1_2 as LIVE
from . import mm_deep_tail_join_ask_live_audit_v1 as AUDIT
from . import mm_deep_tail_join_ask_deploy_v2 as D
from . import mm_deep_tail_join_ask_deploy_v2_1 as D21
from . import mm_deep_tail_join_ask_deploy_v2_2 as D22

DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_3_V1_2_M5_VERIFIED"
Q1_ARM = "LIVE_DEEP_TAIL_Q1_V12"
Q5_ARM = "LIVE_DEEP_TAIL_Q5_OVERNIGHT_V12"
KILL_ARM = B.KILL_ARM
Q1_DEFAULT_HOURS = 1.0
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0
Q5_DEFAULT_HOURS = 8.0
Q5_DEFAULT_MAX_LOSS = 20.0
Q5_DEFAULT_MIN_EQUITY = 75.0
STARTUP_TIMEOUT_S = 120.0
PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2_3.json"


def _patch_parent():
    D22._install_patch()  # notebook-safe private WS preflight
    D.LIVE = LIVE
    D._patch_paths()


def _current_head():
    return (V10._git_state() or {}).get("head")


def static_self_check(*, show=True):
    _patch_parent()
    core = LIVE.static_self_check(show=False)
    checks = {
        "core_ok": bool(core.get("ok")),
        "engine_version": LIVE.LIVE_VERSION,
        "nonrecursive_shutdown": core.get("nonrecursive_shutdown") is True,
        "nonrecursive_flatten": core.get("nonrecursive_flatten") is True,
        "audit_orphan_fresh_confirmation": core.get("audit_orphan_requires_fresh_confirmation") is True,
        "m5_exchange_resting_verification": core.get("m5_requires_exchange_zero_resting") is True,
        "m5_flat_position_verification": core.get("m5_requires_flat_position") is True,
        "exact_raw_eof_barrier": core.get("v12_2_barrier") is True,
        "stable_private_ws": core.get("stable_private_ws_quiet_timeout_reconnect") is False,
        "v11_cancel_verification": core.get("cancel_resting_set_verification") is True,
        "cross_feed_fill_dedup": core.get("cross_feed_fill_identity") == "ORDER_ID_PLUS_TRADE_ID",
        "ladder": tuple(CORE.LADDER_Q) == (1, 5, 10, 20, 30, 50, 100),
    }
    ok = all(v is True for k, v in checks.items() if k != "engine_version")
    out = {
        "deploy_version": DEPLOY_VERSION,
        **checks,
        "ok": ok,
        "orders_sent": False,
    }
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOY V2.3 STATIC CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:44s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.3 static self-check failed: {out}")
    return out


def api_capacity_preflight(**kwargs):
    return D21.api_capacity_preflight(**kwargs)


def live_preflight(*, quote_size, runtime_hours, max_start_loss_usd,
                   min_start_equity_usd, show=True, probe_private_ws=True):
    _patch_parent()
    static_self_check(show=show)
    cap = D21.api_capacity_preflight(show=show)
    # D.live_preflight provides calibrated account/fee/flatness checks. D22 has
    # patched its private WS adapter to be notebook-safe, and D.LIVE now points to V1.2.
    out = D.live_preflight(
        quote_size=float(quote_size),
        runtime_hours=float(runtime_hours),
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
        show=show,
        probe_private_ws=probe_private_ws,
    )
    out = dict(out)
    out["api_capacity_preflight"] = cap
    out["deploy_wrapper_version"] = DEPLOY_VERSION
    out["live_engine_version"] = LIVE.LIVE_VERSION
    out["orders_sent"] = False
    return out


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
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}_deploy_v2_3").resolve()
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
        "scientific_status": "FRESH_FORWARD_LIVE_EXECUTION_VALIDATION_AFTER_OPERATIONAL_FIX",
        "no_auto_scale": True,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(session / "parent_preflight_snapshot.json", pre)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2_3",
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
            raise RuntimeError(f"Live V2.3 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        state_ok = last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}
        private_ok = last.get("private_ws_ready") is True
        raw_ok = last.get("raw_watchdog_ready") is True
        if state_ok and private_ok and raw_ok:
            break
        time.sleep(0.25)
    else:
        B._atomic(session / "KILL_REQUEST.json", {
            "time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V2_3"
        })
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(f"V2.3 startup health timeout. Last health={last}\n{tail}")

    print("\n" + "=" * 100)
    print("REAL-MONEY DEEP-TAIL V2.3 PROCESS ARMED")
    print("=" * 100)
    print("Session:                  ", session)
    print("PID:                      ", p.pid)
    print("Engine:                   ", LIVE.LIVE_VERSION)
    print("Quantity:                 ", f"Q{float(q):g}")
    print("Runtime:                  ", f"{float(hours):.2f}h from first complete window")
    print("Software loss trigger:    ", f"-${float(max_loss):.2f}")
    print("Order-group 15s cap:      ", f"{group_limit:.0f} matched contracts")
    print("Private execution WS:      READY")
    print("Raw EOF freshness watcher: READY")
    print("Verified M5 cleanup:       ENABLED")
    print("Nonrecursive shutdown:     ENABLED")
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
        mode="DEEP_TAIL_Q1_SMOKE_V12",
        arm_phrase=arm_phrase,
        expected_arm=Q1_ARM,
    )


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    session = Path(session_dir).resolve()
    a = AUDIT.audit_live_session(session, show=False, write=True)
    cfg = B._read(session / "process_config.json", {}) or {}
    provenance = B._read(session / "deep_tail_source_provenance.json", {}) or {}
    q = B._f(cfg.get("quote_size"), float("nan"))
    q1_head = ((provenance.get("git") or {}).get("head"))
    current_head = _current_head()

    checks = {
        "q1_size": abs(q - 1.0) < 1e-9,
        "completed": a.get("completed") is True,
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
        "note": "Operational promotion only; not evidence that expected PnL is positive.",
        "orders_sent": False,
        "exchange_api_called": False,
    }
    if passed and write_receipt:
        B._atomic(PROMOTION_PATH, receipt)

    if show:
        print("=" * 100)
        print("Q1 V1.2 -> Q5 OPERATIONAL PROMOTION — READ ONLY")
        print("=" * 100)
        for k, v in checks.items():
            print(f"{k:40s}: {v}")
        print("PROMOTION:", "PASS" if passed else "NOT READY")
        if passed:
            print("Receipt:", PROMOTION_PATH)
        else:
            print("Q5 remains hard-gated.")
    return receipt


def _require_q1_promotion():
    r = B._read(PROMOTION_PATH, {}) or {}
    head = _current_head()
    if not r or r.get("passed") is not True:
        raise RuntimeError("Q5 HARD GATE: no passing V1.2 Q1 promotion receipt.")
    if str(r.get("live_engine_version")) != str(LIVE.LIVE_VERSION):
        raise RuntimeError("Q5 HARD GATE: Q1 used a different live-engine version.")
    if not head or str(r.get("q1_git_head")) != str(head):
        raise RuntimeError("Q5 HARD GATE: git HEAD changed after Q1; run a new Q1 smoke.")
    return r


def start_q5_overnight(*, arm_phrase=None,
                       runtime_hours=Q5_DEFAULT_HOURS,
                       max_start_loss_usd=Q5_DEFAULT_MAX_LOSS,
                       min_start_equity_usd=Q5_DEFAULT_MIN_EQUITY):
    _require_q1_promotion()
    return _launch(
        q=5.0,
        hours=float(runtime_hours),
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q5_OVERNIGHT_V12",
        arm_phrase=arm_phrase,
        expected_arm=Q5_ARM,
    )


def live_status(**kwargs):
    _patch_parent()
    return D.live_status(**kwargs)


def kill_and_flatten_live(**kwargs):
    _patch_parent()
    return D.kill_and_flatten_live(**kwargs)


def _run_child(session, cfg_path):
    cfg = B._read(Path(cfg_path), {}) or {}
    LIVE.run_live_process(Path(session).resolve(), cfg)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    a = ap.parse_args()
    if a.run_live_session:
        _run_child(a.run_live_session, a.config)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "Q1_ARM",
    "Q5_ARM",
    "KILL_ARM",
    "PROMOTION_PATH",
    "static_self_check",
    "api_capacity_preflight",
    "live_preflight",
    "start_q1_smoke",
    "q1_promotion_check",
    "start_q5_overnight",
    "live_status",
    "kill_and_flatten_live",
]
