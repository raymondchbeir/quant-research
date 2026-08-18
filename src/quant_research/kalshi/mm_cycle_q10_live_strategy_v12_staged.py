from __future__ import annotations

"""Gated promotion launcher for V12 live execution.

Stage sequence
--------------
1. Q1 one-window architecture smoke via V12 core.
2. Q5 one-hour execution-stability validation, only after a passing Q1 V12 audit.
3. Q10 24-hour validation, only after a passing Q5 V12 audit.

This module changes no strategy mechanics. It only enforces promotion gates and
constructs the same V12 process with different quote size / runtime configuration.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_cycle_q10_live_strategy_v11 as V11
from . import mm_cycle_q10_live_strategy_v12 as V12

STAGED_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V12_STAGED"
Q5_QTY = 5.0
Q5_HOURS = 1.0
Q5_MIN_EQUITY_USD = 75.0
Q5_ARM = "LIVE_Q5_1H"
Q10_ARM = B.FULL_ARM


def _stage_gate(session_dir, *, expected_q, expected_mode=None, show=True):
    """Read-only promotion gate based on the completed V12 execution audit."""
    session = Path(session_dir).resolve()
    cfg = B._read(session / "process_config.json", {}) or {}
    final = B._read(session / "final_summary.json", {}) or {}
    audit = V12.audit_v12_smoke(session, show=show, write_result=True)

    q = B._f(cfg.get("quote_size"), np.nan)
    mode = str(cfg.get("mode") or "")
    q_ok = bool(np.isfinite(q) and abs(q - float(expected_q)) <= B.EPS)
    mode_ok = True if expected_mode is None else mode == str(expected_mode)
    completed = bool(final)
    promotion = bool((audit.get("gates") or {}).get("promotion_ready_for_larger_smoke"))

    out = {
        "session_dir": str(session),
        "expected_q": float(expected_q),
        "actual_q": q,
        "q_match": q_ok,
        "expected_mode": expected_mode,
        "actual_mode": mode,
        "mode_match": mode_ok,
        "completed": completed,
        "audit_promotion_pass": promotion,
        "pass": bool(q_ok and mode_ok and completed and promotion),
        "audit": audit,
    }
    if show:
        print("=" * 96)
        print("V12 STAGE PROMOTION GATE")
        print("=" * 96)
        print("session:           ", session)
        print("quote size:        ", q, "expected", expected_q)
        print("mode:              ", mode)
        print("completed:         ", completed)
        print("audit promotion:   ", promotion)
        print("GATE:              ", "PASS" if out["pass"] else "FAIL")
    return out


def gate_q1_for_q5(session_dir, *, show=True):
    return _stage_gate(
        session_dir,
        expected_q=B.SMOKE_Q,
        expected_mode="SMOKE_Q1_ONE_WINDOW",
        show=show,
    )


def gate_q5_for_q10(session_dir, *, show=True):
    return _stage_gate(
        session_dir,
        expected_q=Q5_QTY,
        expected_mode="LIVE_Q5_1H",
        show=show,
    )


def _launch_v12_stage(*, mode, q, hours, max_loss, min_equity, arm, expected_arm,
                      prior_session=None, prior_expected_q=None,
                      prior_expected_mode=None):
    if str(arm) != str(expected_arm):
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly."
        )

    if prior_session is not None:
        gate = _stage_gate(
            prior_session,
            expected_q=float(prior_expected_q),
            expected_mode=prior_expected_mode,
            show=True,
        )
        if not gate["pass"]:
            raise RuntimeError("Prior V12 stage did not pass promotion gate. Refusing to arm.")

    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    static = V12.static_self_check(show=True)
    if not static["pass"]:
        raise RuntimeError(f"V12 static self-check failed: {static}")

    V3._calibrated_preflight(
        quote_size=float(q),
        runtime_hours=float(hours),
        max_loss_usd=float(max_loss),
        min_equity_usd=float(min_equity),
        mode=str(mode),
        save_dir=None,
        show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_{str(mode).lower()}_v12").resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": str(mode),
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": V12.LIVE_VERSION,
        "staged_launcher_version": STAGED_VERSION,
        "execution_parent": V12.EXECUTION_PARENT,
        "recording_parent": V12.RECORDING_PARENT,
        "freshness_arch_version": V12.FRESHNESS_ARCH_VERSION,
        "engine_architecture": "V12_PRIORITY_FRESHNESS_GATED_STAGE",
        "recording_version": V10.RECORDING_VERSION,
        "comparison_schema_version": V10.COMPARISON_SCHEMA_VERSION,
        "order_api_safety_version": V11.ORDER_API_SAFETY_VERSION,
        "prior_stage_session": str(Path(prior_session).resolve()) if prior_session else None,
    }
    B._atomic(session / "process_config.json", cfg)
    V12._write_v12_bundle(session, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v12",
                "--run-live-session", str(session),
                "--config", str(session / "process_config.json"),
            ],
            cwd=str(C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        fh.close()

    B._atomic(
        B.CONTROL_PATH,
        {
            "live_version": V12.LIVE_VERSION,
            "staged_launcher_version": STAGED_VERSION,
            "execution_parent": V12.EXECUTION_PARENT,
            "recording_parent": V12.RECORDING_PARENT,
            "freshness_arch_version": V12.FRESHNESS_ARCH_VERSION,
            "recording_version": V10.RECORDING_VERSION,
            "order_api_safety_version": V11.ORDER_API_SAFETY_VERSION,
            "running": True,
            "pid": p.pid,
            "session_dir": str(session),
            "mode": str(mode),
            "started_at": B._iso(),
            "config": cfg,
            "log_path": str(log),
        },
    )

    deadline = time.time() + 90.0
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-20000:] if log.exists() else ""
            raise RuntimeError(
                f"Live V12 staged process exited during startup rc={p.returncode}\n{tail}"
            )
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-20000:] if log.exists() else ""
        raise RuntimeError(f"Live V12 staged startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V12 STAGE ARMED")
    print("  stage:      ", mode)
    print("  session:    ", session)
    print("  pid:        ", p.pid)
    print(f"  Q:           {float(q):g} per eligible market")
    print(f"  runtime:     {float(hours):g} hours from first complete M0")
    print(
        f"  cancel SLO: median<={V12.TARGET_CANCEL_SEND_MS:.0f}ms | "
        f"p95<={V12.P95_CANCEL_SEND_MS:.0f}ms | "
        f"max<={V12.HARD_CANCEL_SEND_MS:.0f}ms"
    )
    return B.live_status(show=False)


def start_live_q1(*, arm_phrase=None,
                  max_start_loss_usd=B.LOSS_LIMIT_USD,
                  min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    """Q1, exactly one synchronized M0-M5 window."""
    return V12.start_live_smoke_q1_one_window(
        arm_phrase=arm_phrase,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
    )


def start_live_q5_after_q1(*, prior_q1_session, arm_phrase=None,
                           max_start_loss_usd=B.LOSS_LIMIT_USD,
                           min_start_equity_usd=Q5_MIN_EQUITY_USD):
    """Q5 for one hour, only after a passing completed Q1 V12 audit."""
    return _launch_v12_stage(
        mode="LIVE_Q5_1H",
        q=Q5_QTY,
        hours=Q5_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected_arm=Q5_ARM,
        prior_session=prior_q1_session,
        prior_expected_q=B.SMOKE_Q,
        prior_expected_mode="SMOKE_Q1_ONE_WINDOW",
    )


def start_live_q10_after_q5(*, prior_q5_session, arm_phrase=None,
                            max_start_loss_usd=B.LOSS_LIMIT_USD,
                            min_start_equity_usd=B.FULL_MIN_EQUITY):
    """Frozen Q10/24h, only after a passing completed Q5 V12 audit."""
    return _launch_v12_stage(
        mode="LIVE_Q10_24H",
        q=B.FULL_Q,
        hours=B.FULL_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected_arm=Q10_ARM,
        prior_session=prior_q5_session,
        prior_expected_q=Q5_QTY,
        prior_expected_mode="LIVE_Q5_1H",
    )


def audit_stage(session_dir, *, show=True):
    return V12.audit_v12_smoke(session_dir, show=show, write_result=True)


def live_status(*, show=True, tail_lines=20):
    return B.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    return B.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


__all__ = [
    "STAGED_VERSION",
    "Q5_QTY",
    "Q5_HOURS",
    "Q5_MIN_EQUITY_USD",
    "Q5_ARM",
    "Q10_ARM",
    "gate_q1_for_q5",
    "gate_q5_for_q10",
    "start_live_q1",
    "start_live_q5_after_q1",
    "start_live_q10_after_q5",
    "audit_stage",
    "live_status",
    "kill_and_flatten_live",
]
