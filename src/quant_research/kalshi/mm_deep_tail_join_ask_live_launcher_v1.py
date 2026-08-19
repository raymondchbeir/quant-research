from __future__ import annotations

"""Explicit arming launcher for the deep-tail JOIN_ASK live ladder.

The launcher is intentionally separate from the engine so importing the strategy can
never send an order. Every real-money start requires an exact phrase. Q5 overnight has
its own especially explicit phrase.
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
from . import mm_deep_tail_join_ask_live_v1 as CORE
from . import mm_deep_tail_join_ask_live_v1_1 as LIVE

LAUNCHER_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_LAUNCHER_V1"
Q5_OVERNIGHT_ARM = "LIVE_DEEP_TAIL_Q5_OVERNIGHT"
KILL_ARM = B.KILL_ARM
DEFAULT_OVERNIGHT_HOURS = 8.0
DEFAULT_Q5_MAX_LOSS = 20.0
DEFAULT_Q5_MIN_EQUITY = 75.0
STARTUP_TIMEOUT_S = 120.0


def _patch_control_paths():
    B.ROOT = CORE.ROOT
    B.CONTROL_PATH = CORE.CONTROL_PATH
    B.ROOT.mkdir(parents=True, exist_ok=True)


def _old_live_process_guard():
    # Refuse both the new deep-tail control file and the prior Candidate-C live
    # control file. Two live engines on one account would invalidate risk logic.
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
        raise RuntimeError("Another live strategy process is already running: " + json.dumps(live, default=str))


async def _private_ws_probe_async(timeout_s=12.0):
    key_id, private_key = C.load_auth()
    ws = await C.open_ws(key_id, private_key)
    subscribed = set()
    started = time.perf_counter()
    rows = []
    try:
        await ws.send(json.dumps({
            "id": 991,
            "cmd": "subscribe",
            "params": {"channels": ["fill", "user_orders"]},
        }))
        while time.perf_counter() - started < float(timeout_s):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            row = json.loads(raw)
            rows.append(row)
            if str(row.get("type") or "") == "subscribed":
                ch = str((row.get("msg") or {}).get("channel") or "")
                if ch:
                    subscribed.add(ch)
            if {"fill", "user_orders"}.issubset(subscribed):
                return {
                    "ok": True,
                    "subscribed": sorted(subscribed),
                    "elapsed_s": time.perf_counter() - started,
                    "messages_seen": len(rows),
                }
        return {
            "ok": False,
            "subscribed": sorted(subscribed),
            "elapsed_s": time.perf_counter() - started,
            "messages_seen": len(rows),
            "reason": "subscription_timeout",
        }
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def private_ws_probe(*, show=True, timeout_s=12.0):
    """Read-only authenticated WebSocket subscription test. Sends no orders."""
    out = asyncio.run(_private_ws_probe_async(timeout_s=timeout_s))
    if show:
        print("PRIVATE WS PROBE:", "PASS" if out.get("ok") else "FAIL")
        print("  subscribed:", out.get("subscribed"))
        print("  elapsed:   ", f"{out.get('elapsed_s', 0.0):.3f}s")
        print("  ORDERS SENT: NO")
    if not out.get("ok"):
        raise RuntimeError(f"Private fill/user_orders WebSocket preflight failed: {out}")
    return out


def static_self_check(*, show=True):
    core = LIVE.static_self_check(show=False)
    checks = {
        "core_ok": bool(core.get("ok")),
        "ladder_exact": tuple(CORE.LADDER_Q) == (1, 5, 10, 20, 30, 50, 100),
        "entry_yes_5c": abs(CORE.ENTRY_YES_PRICE - 0.05) < 1e-12,
        "entry_no_5c_equivalent": abs(CORE.ENTRY_NO_BOOK_PRICE - 0.95) < 1e-12,
        "m1_60s": CORE.M1_S == 60.0,
        "m5_300s": CORE.M5_S == 300.0,
        "stable_private_ws": LIVE.LIVE_VERSION.endswith("STABLE_PRIVATE_WS"),
        "v2_create_path_hardened": CORE.V11._post_v11.__name__ == "_post_v11",
        "exact_raw_eof_barrier": hasattr(CORE.V122.BarrierFastCancelWatchdog, "catch_up_to_stable_eof"),
        "orders_sent": False,
    }
    checks["ok"] = all(v for k, v in checks.items() if k not in {"orders_sent"})
    if show:
        print("=" * 96)
        print("DEEP-TAIL DEPLOYMENT STATIC CHECK — NO ORDERS")
        print("=" * 96)
        for k, v in checks.items():
            print(f"{k:34s}: {v}")
    if not checks["ok"]:
        raise RuntimeError(f"static self-check failed: {checks}")
    return checks


def live_preflight(*, quote_size=5.0, runtime_hours=DEFAULT_OVERNIGHT_HOURS,
                   max_start_loss_usd=DEFAULT_Q5_MAX_LOSS,
                   min_start_equity_usd=None, show=True,
                   probe_private_ws=True):
    """Read-only deployment preflight. No orders, cancels, or group mutations."""
    _patch_control_paths()
    _old_live_process_guard()
    static_self_check(show=show)

    q = float(quote_size)
    if q not in tuple(float(x) for x in CORE.LADDER_Q):
        raise RuntimeError(f"Q{q:g} is outside the frozen ladder {CORE.LADDER_Q}")
    if min_start_equity_usd is None:
        if q <= 5:
            min_start_equity_usd = 75.0
        elif q <= 10:
            min_start_equity_usd = 125.0
        else:
            min_start_equity_usd = max(150.0, 1.5 * q)

    report = V3._calibrated_preflight(
        quote_size=q,
        runtime_hours=float(runtime_hours),
        max_loss_usd=float(max_start_loss_usd),
        min_equity_usd=float(min_start_equity_usd),
        mode="DEEP_TAIL_PREFLIGHT_ONLY",
        save_dir=None,
        show=show,
    )
    if probe_private_ws:
        report = dict(report)
        report["private_ws_probe"] = private_ws_probe(show=show)

    if show:
        print()
        print("DEEP-TAIL PREFLIGHT COMPLETE — ORDERS SENT: NO")
        print(f"Requested Q:             {q:g}")
        print(f"Requested runtime:       {float(runtime_hours):.2f}h")
        print(f"Software loss trigger:  -${float(max_start_loss_usd):.2f} from calibrated starting equity")
        print("IMPORTANT: final loss can overshoot the software trigger due to fills/latency/slippage.")
    return report


def _launch(*, q, hours, max_loss, min_equity, mode, arm_phrase, expected_arm):
    if str(arm_phrase) != str(expected_arm):
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly.")

    _patch_control_paths()
    _old_live_process_guard()
    pre = live_preflight(
        quote_size=q,
        runtime_hours=hours,
        max_start_loss_usd=max_loss,
        min_start_equity_usd=min_equity,
        show=True,
        probe_private_ws=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (CORE.ROOT / f"{stamp}_{mode.lower()}_v1_1").resolve()
    session.mkdir(parents=True, exist_ok=False)

    group_limit = max(100.0, 20.0 * float(q))
    cfg = {
        "mode": str(mode),
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "order_group_limit_fp": f"{group_limit:.2f}",
        "live_engine_version": LIVE.LIVE_VERSION,
        "launcher_version": LAUNCHER_VERSION,
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
        "-m", "quant_research.kalshi.mm_deep_tail_join_ask_live_launcher_v1",
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
        "launcher_version": LAUNCHER_VERSION,
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
            tail = log.read_text(encoding="utf-8")[-12000:] if log.exists() else ""
            raise RuntimeError(f"Deep-tail live process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        state_ok = last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}
        private_ok = last.get("private_ws_ready") is True
        raw_ok = last.get("raw_watchdog_ready") is True
        if state_ok and private_ok and raw_ok:
            break
        time.sleep(0.25)
    else:
        tail = log.read_text(encoding="utf-8")[-12000:] if log.exists() else ""
        try:
            B._atomic(session / "KILL_REQUEST.json", {"time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT"})
        except Exception:
            pass
        raise RuntimeError(f"Deep-tail startup health timeout. Last health={last}\n{tail}")

    print("\n" + "=" * 92)
    print("DEEP-TAIL LIVE PROCESS ARMED — REAL ORDERS WILL BE SENT AT M1")
    print("=" * 92)
    print("Session:               ", session)
    print("PID:                   ", p.pid)
    print("Quantity:              ", f"Q{q:g} each tail at M1; opposite tail cancels after first fill")
    print("Runtime:               ", f"{hours:.2f} hours from first complete M0 window")
    print("Loss software trigger: ", f"-${max_loss:.2f} from calibrated starting equity")
    print("Order-group 15s limit: ", f"{group_limit:.0f} matched contracts")
    print("Private fill WS:        READY")
    print("Raw freshness watcher:  READY")
    print("macOS caffeinate:       ", bool(caffeinate))
    print("DO NOT start another live engine on this account.")
    print("Emergency: kill_and_flatten_live(arm_phrase='KILL_AND_FLATTEN')")
    print("=" * 92)
    return live_status(show=False)


def start_q5_overnight(*, arm_phrase=None,
                       runtime_hours=DEFAULT_OVERNIGHT_HOURS,
                       max_start_loss_usd=DEFAULT_Q5_MAX_LOSS,
                       min_start_equity_usd=DEFAULT_Q5_MIN_EQUITY):
    """REAL ORDERS: Q5 deep-tail for the requested overnight duration."""
    return _launch(
        q=5.0,
        hours=float(runtime_hours),
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        mode="DEEP_TAIL_Q5_OVERNIGHT",
        arm_phrase=arm_phrase,
        expected_arm=Q5_OVERNIGHT_ARM,
    )


def start_ladder_stage(*, quote_size, runtime_hours, max_start_loss_usd,
                       min_start_equity_usd=None, arm_phrase=None):
    """REAL ORDERS: generic deliberate ladder stage. Never auto-advances size."""
    q = float(quote_size)
    if q not in tuple(float(x) for x in CORE.LADDER_Q):
        raise RuntimeError(f"Q{q:g} is outside ladder {CORE.LADDER_Q}")
    expected = f"LIVE_DEEP_TAIL_Q{int(q)}"
    if min_start_equity_usd is None:
        min_start_equity_usd = 75.0 if q <= 5 else (125.0 if q <= 10 else max(150.0, 1.5*q))
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
    _patch_control_paths()
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    session = Path(ctl.get("session_dir", "")) if ctl.get("session_dir") else None
    health = B._read(session / "health.json", {}) if session and session.exists() else {}
    final = B._read(session / "final_summary.json", {}) if session and session.exists() else {}
    running = bool(ctl and B._pid_alive(ctl.get("pid")) and not final)
    out = {
        "running": running,
        "control": ctl,
        "health": health or {},
        "final_summary": final or {},
    }
    if show:
        print("=" * 96)
        print("DEEP-TAIL LIVE STATUS")
        print("=" * 96)
        print("Running:          ", running)
        print("Session:          ", ctl.get("session_dir"))
        print("Mode:             ", ctl.get("mode"))
        print("State:            ", (health or {}).get("state"))
        print("Q:                ", (health or {}).get("quote_size", (ctl.get("config") or {}).get("quote_size")))
        print("Private WS ready: ", (health or {}).get("private_ws_ready"))
        print("Raw watcher ready:", (health or {}).get("raw_watchdog_ready"))
        print("Equity:           ", (health or {}).get("equity_usd"))
        print("Start PnL:        ", (health or {}).get("start_pnl_usd"))
        print("Kill equity:      ", (health or {}).get("kill_equity_usd"))
        print("Active tracks:    ", len((health or {}).get("active_tracks") or {}))
        print("Positions:        ", (health or {}).get("positions"))
        print("Windows started:  ", (health or {}).get("windows_started"))
        if final:
            print("Shutdown reason:  ", final.get("shutdown_reason"))
            print("Final flat:       ", final.get("flat_verified"))
            print("Final PnL:        ", final.get("start_pnl_usd", final.get("net_pnl_usd")))
        log_path = Path(ctl.get("log_path", "")) if ctl.get("log_path") else None
        if log_path and log_path.exists() and tail_lines:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n--- LIVE LOG TAIL ---")
            print("\n".join(lines[-int(tail_lines):]))
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    if str(arm_phrase) != KILL_ARM:
        raise RuntimeError(f"Emergency stop refused. Pass arm_phrase={KILL_ARM!r} exactly.")
    _patch_control_paths()
    return B.kill_and_flatten_live(arm_phrase=KILL_ARM, wait_s=float(wait_s))


def _run_child(session, cfg_path):
    session = Path(session).resolve()
    cfg = B._read(Path(cfg_path), {}) or {}
    LIVE.run_live_process(session, cfg)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    args = ap.parse_args()
    if args.run_live_session:
        _run_child(args.run_live_session, args.config)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LAUNCHER_VERSION",
    "Q5_OVERNIGHT_ARM",
    "DEFAULT_OVERNIGHT_HOURS",
    "DEFAULT_Q5_MAX_LOSS",
    "static_self_check",
    "private_ws_probe",
    "live_preflight",
    "start_q5_overnight",
    "start_ladder_stage",
    "live_status",
    "kill_and_flatten_live",
]
