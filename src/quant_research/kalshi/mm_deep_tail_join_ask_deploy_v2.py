from __future__ import annotations

"""Final staged deployment interface for deep-tail 5c + fixed JOIN_ASK live V1.1.

Importing this module never places orders. Real-money starts require exact arming
phrases. Q5+ is hard-gated behind a completed, clean Q1 smoke produced by the exact
same git HEAD and live-engine version.
"""

import argparse
import asyncio
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
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_deep_tail_join_ask_live_v1 as CORE
from . import mm_deep_tail_join_ask_live_v1_1 as LIVE
from . import mm_deep_tail_join_ask_live_audit_v1 as AUDIT

DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_Q1_GATED"
Q1_ARM = "LIVE_DEEP_TAIL_Q1"
Q5_ARM = "LIVE_DEEP_TAIL_Q5_OVERNIGHT"
KILL_ARM = B.KILL_ARM
Q1_DEFAULT_HOURS = 1.0
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0
Q5_DEFAULT_HOURS = 8.0
Q5_DEFAULT_MAX_LOSS = 20.0
Q5_DEFAULT_MIN_EQUITY = 75.0
STARTUP_TIMEOUT_S = 120.0
PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2.json"


def _patch_paths():
    B.ROOT = CORE.ROOT
    B.CONTROL_PATH = CORE.CONTROL_PATH
    CORE.ROOT.mkdir(parents=True, exist_ok=True)


def _current_head():
    return (V10._git_state() or {}).get("head")


def _guard_other_live_processes():
    controls = [
        CORE.CONTROL_PATH,
        C.DATA_ROOT / "live_cycle_q10_v1" / "active_live.json",
    ]
    live = []
    for p in controls:
        obj = B._read(p, {}) or {}
        if obj and B._pid_alive(obj.get("pid")):
            live.append({"control": str(p), "state": obj})
    if live:
        raise RuntimeError(
            "Another live strategy process is already running. Refusing concurrent account control: "
            + json.dumps(live, default=str)
        )


async def _private_probe_async(timeout_s=12.0):
    key_id, private_key = C.load_auth()
    ws = await C.open_ws(key_id, private_key)
    subscribed = set()
    start = time.perf_counter()
    messages = 0
    try:
        await ws.send(json.dumps({
            "id": 991,
            "cmd": "subscribe",
            "params": {"channels": ["fill", "user_orders"]},
        }))
        while time.perf_counter() - start < float(timeout_s):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            messages += 1
            row = json.loads(raw)
            if str(row.get("type") or "") == "subscribed":
                ch = str((row.get("msg") or {}).get("channel") or "")
                if ch:
                    subscribed.add(ch)
            if {"fill", "user_orders"}.issubset(subscribed):
                return {
                    "ok": True,
                    "subscribed": sorted(subscribed),
                    "elapsed_s": time.perf_counter() - start,
                    "messages_seen": messages,
                }
        return {
            "ok": False,
            "subscribed": sorted(subscribed),
            "elapsed_s": time.perf_counter() - start,
            "messages_seen": messages,
            "reason": "subscription_timeout",
        }
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def private_ws_probe(*, show=True, timeout_s=12.0):
    """Authenticated read-only subscription test. Sends no orders."""
    out = asyncio.run(_private_probe_async(timeout_s=timeout_s))
    if show:
        print("PRIVATE fill + user_orders WS:", "PASS" if out.get("ok") else "FAIL")
        print("  subscribed:", out.get("subscribed"))
        print("  elapsed:   ", f"{out.get('elapsed_s', 0.0):.3f}s")
        print("  ORDERS SENT: NO")
    if not out.get("ok"):
        raise RuntimeError(f"Private WebSocket preflight failed: {out}")
    return out


def static_self_check(*, show=True):
    core = LIVE.static_self_check(show=False)
    checks = {
        "core_self_check": bool(core.get("ok")),
        "ladder": tuple(CORE.LADDER_Q) == (1, 5, 10, 20, 30, 50, 100),
        "yes_entry_5c": abs(float(CORE.ENTRY_YES_PRICE) - 0.05) < 1e-12,
        "no_entry_5c_equivalent": abs(float(CORE.ENTRY_NO_BOOK_PRICE) - 0.95) < 1e-12,
        "m1_60s": float(CORE.M1_S) == 60.0,
        "m5_300s": float(CORE.M5_S) == 300.0,
        "wall_clock_m1_scheduler": core.get("m1_scheduler") == "CURRENT_WALL_CLOCK_2MS_LOOP",
        "stable_private_ws": core.get("stable_private_ws_quiet_timeout_reconnect") is False,
        "bounded_rest_fill_fallback": core.get("rest_fill_reconciler_bounded_by_min_ts") is True,
        "v11_resting_set_cancel_verification": core.get("cancel_resting_set_verification") is True,
        "v2_batch_cancel_fallback": core.get("cancel_v2_batch_fallback") is True,
        "exact_raw_eof_barrier": hasattr(CORE.V122.BarrierFastCancelWatchdog, "catch_up_to_stable_eof"),
        "idempotent_v11_create": CORE.V11._post_v11.__name__ == "_post_v11",
    }
    checks["ok"] = all(checks.values())
    out = {"deploy_version": DEPLOY_VERSION, "live_version": LIVE.LIVE_VERSION, **checks, "orders_sent": False}
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOYMENT STATIC CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:40s}: {v}")
    if not checks["ok"]:
        raise RuntimeError(f"Static self-check failed: {out}")
    return out


def live_preflight(*, quote_size, runtime_hours, max_start_loss_usd,
                   min_start_equity_usd, show=True, probe_private_ws=True):
    """Read-only exchange/account preflight. Sends no orders or cancels."""
    _patch_paths()
    _guard_other_live_processes()
    static_self_check(show=show)

    q = float(quote_size)
    if q not in tuple(float(x) for x in CORE.LADDER_Q):
        raise RuntimeError(f"Q{q:g} is outside the frozen ladder {CORE.LADDER_Q}")

    report = V3._calibrated_preflight(
        quote_size=q,
        runtime_hours=float(runtime_hours),
        max_loss_usd=float(max_start_loss_usd),
        min_equity_usd=float(min_start_equity_usd),
        mode="DEEP_TAIL_PREFLIGHT_ONLY",
        save_dir=None,
        show=show,
    )
    report = dict(report)
    if probe_private_ws:
        report["private_ws_probe"] = private_ws_probe(show=show)
    report["deploy_version"] = DEPLOY_VERSION
    report["live_version"] = LIVE.LIVE_VERSION
    report["orders_sent"] = False
    if show:
        print("\nDEEP-TAIL PREFLIGHT: PASS — ORDERS SENT: NO")
        print(f"Q:                       {q:g}")
        print(f"Runtime:                 {float(runtime_hours):.2f}h")
        print(f"Software loss trigger:  -${float(max_start_loss_usd):.2f} from calibrated starting equity")
        print("This is a software trigger, NOT a guaranteed final-loss cap; fills/slippage/latency can overshoot it.")
    return report


def _launch(*, q, hours, max_loss, min_equity, mode, arm_phrase, expected_arm):
    if str(arm_phrase) != str(expected_arm):
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly.")

    _patch_paths()
    _guard_other_live_processes()
    pre = live_preflight(
        quote_size=q,
        runtime_hours=hours,
        max_start_loss_usd=max_loss,
        min_start_equity_usd=min_equity,
        show=True,
        probe_private_ws=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}_deploy_v2").resolve()
    session.mkdir(parents=True, exist_ok=False)

    # 15-second order-group match cap is an independent runaway guard. Q5 gets
    # 100 contracts, enough for normal 9-market entry+exit flow but finite.
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
        "scientific_status": "FRESH_FORWARD_LIVE_EXECUTION_VALIDATION",
        "no_auto_scale": True,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(session / "parent_preflight_snapshot.json", pre)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_deploy_v2",
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
            raise RuntimeError(f"Live process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        state_ok = last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}
        private_ok = last.get("private_ws_ready") is True
        raw_ok = last.get("raw_watchdog_ready") is True
        if state_ok and private_ok and raw_ok:
            break
        time.sleep(0.25)
    else:
        B._atomic(session / "KILL_REQUEST.json", {"time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT"})
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.exists() else ""
        raise RuntimeError(f"Startup health timeout. Last health={last}\n{tail}")

    print("\n" + "=" * 100)
    print("REAL-MONEY DEEP-TAIL PROCESS ARMED")
    print("=" * 100)
    print("Session:                  ", session)
    print("PID:                      ", p.pid)
    print("Quantity:                 ", f"Q{float(q):g}")
    print("Runtime:                  ", f"{float(hours):.2f}h from first complete window")
    print("Software loss trigger:    ", f"-${float(max_loss):.2f}")
    print("Order-group 15s cap:      ", f"{group_limit:.0f} matched contracts")
    print("Private execution WS:      READY")
    print("Raw EOF freshness watcher: READY")
    print("macOS caffeinate used:     ", bool(caffeinate))
    print("Auto-scaling:              DISABLED")
    print("Emergency kill phrase:     KILL_AND_FLATTEN")
    print("=" * 100)
    return live_status(show=False)


def start_q1_smoke(*, arm_phrase=None,
                   runtime_hours=Q1_DEFAULT_HOURS,
                   max_start_loss_usd=Q1_DEFAULT_MAX_LOSS,
                   min_start_equity_usd=Q1_DEFAULT_MIN_EQUITY):
    """REAL ORDERS. Mandatory first run for this exact code."""
    return _launch(
        q=1.0,
        hours=float(runtime_hours),
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q1_SMOKE",
        arm_phrase=arm_phrase,
        expected_arm=Q1_ARM,
    )


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    """Read-only Q1 audit. Promotion is operational, not an alpha conclusion."""
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
        print("Q1 -> Q5 OPERATIONAL PROMOTION CHECK — READ ONLY")
        print("=" * 100)
        for k, v in checks.items():
            print(f"{k:38s}: {v}")
        print("PROMOTION:", "PASS" if passed else "NOT READY")
        if passed:
            print("Receipt:", PROMOTION_PATH)
        else:
            print("Q5 remains hard-gated. Do not bypass the deployment interface.")
    return receipt


def _require_q1_promotion():
    r = B._read(PROMOTION_PATH, {}) or {}
    head = _current_head()
    if not r or r.get("passed") is not True:
        raise RuntimeError(
            "Q5 HARD GATE: no passing Q1 promotion receipt. Run Q1, wait for clean completion, "
            "then run q1_promotion_check(Q1_SESSION)."
        )
    if str(r.get("live_engine_version")) != str(LIVE.LIVE_VERSION):
        raise RuntimeError("Q5 HARD GATE: Q1 receipt used a different live-engine version.")
    if not head or str(r.get("q1_git_head")) != str(head):
        raise RuntimeError("Q5 HARD GATE: git HEAD changed after Q1. Run a new Q1 smoke on this exact code.")
    return r


def start_q5_overnight(*, arm_phrase=None,
                       runtime_hours=Q5_DEFAULT_HOURS,
                       max_start_loss_usd=Q5_DEFAULT_MAX_LOSS,
                       min_start_equity_usd=Q5_DEFAULT_MIN_EQUITY):
    """REAL ORDERS. Q5 overnight after the exact-code Q1 operational gate passes."""
    _require_q1_promotion()
    return _launch(
        q=5.0,
        hours=float(runtime_hours),
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q5_OVERNIGHT",
        arm_phrase=arm_phrase,
        expected_arm=Q5_ARM,
    )


def start_ladder_stage(*, quote_size, runtime_hours, max_start_loss_usd,
                       min_start_equity_usd, arm_phrase=None):
    q = float(quote_size)
    if q not in tuple(float(x) for x in CORE.LADDER_Q):
        raise RuntimeError(f"Q{q:g} outside ladder {CORE.LADDER_Q}")
    if q > 1.0:
        _require_q1_promotion()
    expected = f"LIVE_DEEP_TAIL_Q{int(q)}"
    return _launch(
        q=q,
        hours=float(runtime_hours),
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode=f"DEEP_TAIL_Q{int(q)}_LADDER",
        arm_phrase=arm_phrase,
        expected_arm=expected,
    )


def live_status(*, show=True, tail_lines=25):
    _patch_paths()
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    session = Path(ctl.get("session_dir", "")) if ctl.get("session_dir") else None
    health = B._read(session / "health.json", {}) if session and session.exists() else {}
    final = B._read(session / "final_summary.json", {}) if session and session.exists() else {}
    running = bool(ctl and B._pid_alive(ctl.get("pid")) and not final)
    out = {"running": running, "control": ctl, "health": health or {}, "final_summary": final or {}}
    if show:
        print("=" * 100)
        print("DEEP-TAIL LIVE STATUS")
        print("=" * 100)
        print("Running:           ", running)
        print("Session:           ", ctl.get("session_dir"))
        print("Mode:              ", ctl.get("mode"))
        print("State:             ", (health or {}).get("state"))
        print("Q:                 ", (ctl.get("config") or {}).get("quote_size"))
        print("Private WS ready:  ", (health or {}).get("private_ws_ready"))
        print("Raw watcher ready: ", (health or {}).get("raw_watchdog_ready"))
        print("Equity:            ", (health or {}).get("equity_usd"))
        print("Start PnL:         ", (health or {}).get("start_pnl_usd"))
        print("Kill equity:       ", (health or {}).get("kill_equity_usd"))
        print("Active tracks:     ", len((health or {}).get("active_tracks") or {}))
        print("Positions:         ", (health or {}).get("positions"))
        if final:
            print("Shutdown reason:   ", final.get("shutdown_reason"))
            print("Final PnL:         ", final.get("account_pnl_usd"))
            print("Flat verified:     ", final.get("flat_verified"))
            print("Zero resting:      ", final.get("strategy_resting_orders_zero"))
        log_path = Path(ctl.get("log_path", "")) if ctl.get("log_path") else None
        if log_path and log_path.exists() and int(tail_lines) > 0:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n--- LIVE LOG TAIL ---")
            print("\n".join(lines[-int(tail_lines):]))
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    if str(arm_phrase) != str(KILL_ARM):
        raise RuntimeError(f"Emergency stop refused. Pass arm_phrase={KILL_ARM!r} exactly.")
    _patch_paths()
    return B.kill_and_flatten_live(arm_phrase=KILL_ARM, wait_s=float(wait_s))


def _run_child(session, cfg_path):
    session = Path(session).resolve()
    cfg = B._read(Path(cfg_path), {}) or {}
    LIVE.run_live_process(session, cfg)


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
    "PROMOTION_PATH",
    "static_self_check",
    "private_ws_probe",
    "live_preflight",
    "start_q1_smoke",
    "q1_promotion_check",
    "start_q5_overnight",
    "start_ladder_stage",
    "live_status",
    "kill_and_flatten_live",
]
