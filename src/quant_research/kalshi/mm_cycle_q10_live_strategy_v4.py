from __future__ import annotations

"""V4 live wrapper for frozen Candidate-C / CYCLE_ALWAYS_EXIT validation.

V4 keeps all V3 behavior and fixes a startup bug in V1's recorder launcher:
V1 pre-created ``raw_capture/`` before launching the V5 recorder, while the V5
recorder intentionally requires its session directory not to exist and creates
it itself.  That caused every real live launch to fail before any strategy order
could be sent.

V4 patches only that plumbing. Strategy mechanics, risk limits, balance
calibration, V2 fill-aware cancel safety, fee checks, and arming phrases are
unchanged.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V4"


def _start_recorder_fixed(session):
    """Launch V5 without pre-creating its run-session directory."""
    session = Path(session).resolve()
    raw = session / "raw_capture"
    if raw.exists():
        raise RuntimeError(
            f"Refusing recorder startup because raw_capture already exists: {raw}. "
            "A fresh live session must start with no raw_capture directory."
        )

    log = session / "raw_recorder.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_event_time_m0_m5_recorder_v5",
                "--run-session", str(raw),
            ],
            cwd=str(C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        fh.close()

    deadline = time.time() + B.RECORDER_START_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-6000:] if log.exists() else ""
            raise RuntimeError(f"V5 recorder startup failure rc={p.returncode}\n{tail}")
        last = B._read(raw / "health.json", {}) or {}
        if last.get("running") and last.get("healthy"):
            return p, last
        time.sleep(0.35)

    try:
        os.kill(p.pid, signal.SIGTERM)
    except Exception:
        pass
    raise RuntimeError(f"V5 recorder health timeout: {last}")


def _cleanup_failed_startup_group(session):
    """Best-effort cleanup for an order group created before a startup failure."""
    session = Path(session).resolve()
    group = B._read(session / "order_group.json", {}) or {}
    gid = group.get("order_group_id")
    if not gid:
        return None
    try:
        client = B.Q1.LiveClient()
        trig = B._trigger_group(client, gid)
        dele = B._delete_group(client, gid)
        B._atomic(session / "failed_startup_group_cleanup.json", {
            "time": B._iso(),
            "order_group_id": gid,
            "trigger": trig,
            "delete": dele,
        })
        return {"order_group_id": gid, "trigger": trig, "delete": dele}
    except Exception as exc:
        try:
            B._atomic(session / "failed_startup_group_cleanup.json", {
                "time": B._iso(),
                "order_group_id": gid,
                "error": repr(exc),
            })
        except Exception:
            pass
        return {"order_group_id": gid, "error": repr(exc)}


def _run_process_v4(session, cfg):
    session = Path(session).resolve()

    # Install V2 race handling + V3 calibrated-equity semantics.
    client = B.Q1.LiveClient()
    diag = V3._install(client)
    B._atomic(session / "balance_semantics.json", diag)

    # Only plumbing patch: let V5 create raw_capture itself.
    B._start_recorder = _start_recorder_fixed

    try:
        B._run_process(session, cfg)
    except BaseException:
        _cleanup_failed_startup_group(session)
        raise


def live_preflight(*, quote_size=B.FULL_Q, runtime_hours=B.FULL_HOURS,
                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                   min_start_equity_usd=None, show=True):
    return V3.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
    )


def _launch(*, mode, q, hours, max_loss, min_equity, arm, expected):
    if str(arm) != expected:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected!r} exactly.")

    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    # Parent read-only checks before the detached process exists.
    V3._calibrated_preflight(
        quote_size=q,
        runtime_hours=hours,
        max_loss_usd=max_loss,
        min_equity_usd=min_equity,
        mode=mode,
        save_dir=None,
        show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_{mode.lower()}_v4").resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": mode,
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": LIVE_VERSION,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v4",
                "--run-live-session", str(session),
                "--config", str(cfg_path),
            ],
            cwd=str(C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        fh.close()

    B._atomic(B.CONTROL_PATH, {
        "live_version": LIVE_VERSION,
        "running": True,
        "pid": p.pid,
        "session_dir": str(session),
        "mode": mode,
        "started_at": B._iso(),
        "config": cfg,
        "log_path": str(log),
    })

    deadline = time.time() + 90.0
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-8000:] if log.exists() else ""
            raise RuntimeError(f"Live V4 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-8000:] if log.exists() else ""
        raise RuntimeError(f"Live V4 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V4 PROCESS ARMED")
    print("  mode:   ", mode)
    print("  session:", session)
    print("  pid:    ", p.pid)
    print(f"  Q:       {q:g} per eligible market")
    print(f"  kill:    -${max_loss:.2f} from calibrated starting TOTAL account equity")
    print("Use live_status(); emergency stop is kill_and_flatten_live(arm_phrase='KILL_AND_FLATTEN').")
    return live_status(show=False)


def start_live_smoke_q1_one_window(*, arm_phrase=None,
                                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                                   min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    return _launch(
        mode="SMOKE_Q1_ONE_WINDOW",
        q=B.SMOKE_Q,
        hours=1.0,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected=B.SMOKE_ARM,
    )


def start_live_cycle_q10(*, arm_phrase=None, runtime_hours=B.FULL_HOURS,
                         max_start_loss_usd=B.LOSS_LIMIT_USD,
                         min_start_equity_usd=B.FULL_MIN_EQUITY):
    if abs(float(runtime_hours) - B.FULL_HOURS) > B.EPS:
        raise RuntimeError("V4 full validation is frozen to exactly 24 hours.")
    return _launch(
        mode="LIVE_Q10_24H",
        q=B.FULL_Q,
        hours=B.FULL_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected=B.FULL_ARM,
    )


def live_status(*, show=True, tail_lines=20):
    return B.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    return B.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    a = ap.parse_args()
    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}
        _run_process_v4(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "live_preflight",
    "start_live_smoke_q1_one_window",
    "start_live_cycle_q10",
    "live_status",
    "kill_and_flatten_live",
]
