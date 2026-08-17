from __future__ import annotations

"""Exploratory full-window M0-M15 event-time sidecar recorder.

This recorder is intentionally separate from the frozen M0-M5 OOS stack.
It may run in parallel with the frozen OOS recorder/shadow and must NOT be used
to change the already-running OOS strategy.

Purpose
-------
Capture the same nine crypto 15-minute series with the same corrected V5
orderbook reconstruction, sequence accounting, repair logic, ticker validation,
and exact Decimal book state, but persist the entire contract life:

    M0 <= elapsed < M15 (900s)

This dataset is EXPLORATORY/DEVELOPMENT for later questions such as:
- Does Candidate-C/L3 support retain post-fill edge in M5-M10?
- Does it retain edge in M10-M15?
- Does spread/fill capacity improve or deteriorate late in the contract?
- Could a separately frozen future strategy safely market make beyond M5?

It does not execute or shadow any strategy.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import recorder_core as C
from . import mm_event_time_m0_m5_recorder_v5 as V5

STUDY_VERSION = "MM_EVENT_TIME_M0_M15_EXPLORATORY_V1"
ROOT = C.DATA_ROOT / "mm_event_m0_m15_exploratory_v1"
CONTROL_PATH = ROOT / "active_recorder.json"
TRADE_WINDOW_START_S = 0.0
TRADE_WINDOW_END_S = 900.0
LABEL_TAIL_END_S = 900.0
PRESUBSCRIBE_LEAD_S = 300.0
STARTUP_TIMEOUT_S = 60.0
STOP_TIMEOUT_S = 45.0
ROOT.mkdir(parents=True, exist_ok=True)


def _iso():
    return pd.Timestamp.now(tz="UTC").isoformat()


def _atomic_json(path: Path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _pid_state(pid):
    try:
        p = subprocess.run(["ps", "-o", "stat=", "-p", str(int(pid))], capture_output=True, text=True, timeout=2)
        if p.returncode != 0:
            return None
        s = p.stdout.strip()
        return s or None
    except Exception:
        return None


def _pid_alive(pid):
    s = _pid_state(pid)
    return bool(s and "Z" not in s.upper())


def _session_dir_now():
    return ROOT / pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")


def _exploratory_spec():
    return {
        "study_version": STUDY_VERSION,
        "research_stage": "EXPLORATORY_FULL_WINDOW_NOT_OOS",
        "universe": list(V5.CRYPTO_SERIES),
        "capture_window": "M0 <= elapsed < M15",
        "capture_elapsed_seconds": [0.0, 900.0],
        "pre_subscribe_lead_seconds": PRESUBSCRIBE_LEAD_S,
        "orderbook": "V5 exact Decimal reconstruction; top3 changes persisted",
        "sequence_accounting": "V5 corrected sid/seq including type=ok",
        "repair_logic": "V5 sequence/cross/negative/ticker mismatch snapshot repair",
        "ticker": "every ticker event during M0-M15",
        "trades": "every public trade during M0-M15",
        "strategy_executed": False,
        "relationship_to_frozen_oos": "independent sidecar only; frozen CYCLE_ALWAYS_EXIT Q10 remains M0-M5",
        "forbidden_use": "do not use this exploratory data to alter the currently running OOS strategy",
    }


def _persist_phase(meta, t=None):
    e = V5._elapsed(meta, t)
    if e is None:
        return False, e, None
    if TRADE_WINDOW_START_S <= e < TRADE_WINDOW_END_S:
        return True, e, "M0_M15_EXPLORATORY"
    return False, e, None


def _rewrite_metadata(path: Path, obj, original_atomic):
    name = Path(path).name
    spec = _exploratory_spec()
    if name == "development_plan.json":
        original_atomic(path, {
            "research_stage": "EXPLORATORY_FULL_WINDOW_NOT_OOS",
            "note": "Compatibility filename only; no strategy family is pre-registered here.",
            "exploratory_spec": spec,
        })
        original_atomic(Path(path).with_name("exploratory_spec.json"), spec)
        return
    if name == "capture_spec.json":
        o = dict(obj)
        o.update({
            "study_version": STUDY_VERSION,
            "purpose": "full M0-M15 exploratory event-time capture while frozen M0-M5 OOS runs separately",
            "research_stage": "EXPLORATORY_FULL_WINDOW_NOT_OOS",
            "research_window": "M0 <= elapsed < M15",
            "research_elapsed_seconds": [0.0, 900.0],
            "label_tail": None,
            "persisted_elapsed_seconds": [0.0, 900.0],
            "strategy_pnl_recorded": False,
            "exploratory_spec_file": "exploratory_spec.json",
        })
        original_atomic(path, o)
        return
    if name == "session_manifest.json":
        o = dict(obj)
        o["study_version"] = STUDY_VERSION
        o["research_stage"] = "EXPLORATORY_FULL_WINDOW_NOT_OOS"
        o["exploratory_spec"] = spec
        o.pop("development_plan", None)
        cap = dict(o.get("capture_spec") or {})
        cap.update({
            "study_version": STUDY_VERSION,
            "research_stage": "EXPLORATORY_FULL_WINDOW_NOT_OOS",
            "research_window": "M0 <= elapsed < M15",
            "research_elapsed_seconds": [0.0, 900.0],
            "persisted_elapsed_seconds": [0.0, 900.0],
        })
        o["capture_spec"] = cap
        original_atomic(path, o)
        return
    if name == "health.json":
        o = dict(obj)
        o["study_version"] = STUDY_VERSION
        o["research_stage"] = "EXPLORATORY_FULL_WINDOW_NOT_OOS"
        original_atomic(path, o)
        return
    original_atomic(path, obj)


async def run_full15_recorder(session_dir: Path):
    original = {
        "TRADE_WINDOW_START_S": V5.TRADE_WINDOW_START_S,
        "TRADE_WINDOW_END_S": V5.TRADE_WINDOW_END_S,
        "LABEL_TAIL_END_S": V5.LABEL_TAIL_END_S,
        "PRESUBSCRIBE_LEAD_S": V5.PRESUBSCRIBE_LEAD_S,
        "STUDY_VERSION": V5.STUDY_VERSION,
        "persist_phase": V5._persist_phase,
        "atomic": V5._atomic_json,
    }
    original_atomic = V5._atomic_json
    V5.TRADE_WINDOW_START_S = TRADE_WINDOW_START_S
    V5.TRADE_WINDOW_END_S = TRADE_WINDOW_END_S
    V5.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    V5.PRESUBSCRIBE_LEAD_S = PRESUBSCRIBE_LEAD_S
    V5.STUDY_VERSION = STUDY_VERSION
    V5._persist_phase = _persist_phase

    def intercept(path, obj):
        return _rewrite_metadata(Path(path), obj, original_atomic)

    V5._atomic_json = intercept
    try:
        await V5.run_event_time_m0_m5_v5_recorder(Path(session_dir))
    finally:
        V5.TRADE_WINDOW_START_S = original["TRADE_WINDOW_START_S"]
        V5.TRADE_WINDOW_END_S = original["TRADE_WINDOW_END_S"]
        V5.LABEL_TAIL_END_S = original["LABEL_TAIL_END_S"]
        V5.PRESUBSCRIBE_LEAD_S = original["PRESUBSCRIBE_LEAD_S"]
        V5.STUDY_VERSION = original["STUDY_VERSION"]
        V5._persist_phase = original["persist_phase"]
        V5._atomic_json = original["atomic"]


def full15_status(*, show=True):
    ctl = _read_json(CONTROL_PATH, {}) or {}
    if not ctl:
        out = {"running": False}
    else:
        session = Path(ctl.get("session_dir", ""))
        health = _read_json(session / "health.json", {}) or {}
        out = {**ctl, **health, "pid_alive": _pid_alive(ctl.get("pid")), "pid_state": _pid_state(ctl.get("pid"))}
        out["running"] = bool(out.get("pid_alive") and health.get("running", True))
    if show:
        print(json.dumps(out, indent=2, default=str))
    return out


def start_full15_recording(*, startup_timeout_s=STARTUP_TIMEOUT_S):
    ctl = _read_json(CONTROL_PATH, {}) or {}
    if ctl and _pid_alive(ctl.get("pid")):
        raise RuntimeError(f"Full15 sidecar already running: {ctl}")
    if CONTROL_PATH.exists():
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass

    session = _session_dir_now().resolve()
    log = session.parent / f"{session.name}.startup.log"
    cmd = [sys.executable, "-m", "quant_research.kalshi.mm_event_time_m0_m15_exploratory_recorder_v1", "--run-session", str(session)]
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=str(C.PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        fh.close()

    _atomic_json(CONTROL_PATH, {
        "pid": proc.pid,
        "session_dir": str(session),
        "started_at": _iso(),
        "log_path": str(log),
        "study_version": STUDY_VERSION,
        "research_stage": "EXPLORATORY_FULL_WINDOW_NOT_OOS",
    })

    deadline = time.time() + float(startup_timeout_s)
    last = {}
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = log.read_text(encoding="utf-8")[-5000:]
            except Exception:
                pass
            raise RuntimeError(f"Full15 recorder exited during startup rc={proc.returncode}\n{tail}")
        if session.exists():
            last = _read_json(session / "health.json", {}) or {}
            if last.get("running") and last.get("healthy"):
                break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"Full15 recorder health timeout: {last}")

    print("FULL15 EXPLORATORY SIDECAR READY")
    print("Session:", session)
    print("PID:", proc.pid)
    print("Capture: M0-M15 entire contract")
    print("Frozen M0-M5 OOS stack is independent and must remain unchanged.")
    return full15_status(show=False)


def stop_full15_recording(*, expected_session=None, timeout_s=STOP_TIMEOUT_S):
    ctl = _read_json(CONTROL_PATH, {}) or {}
    if not ctl:
        print("No active Full15 sidecar control file.")
        return None
    session = Path(ctl.get("session_dir", "")).resolve()
    if expected_session is not None and session != Path(expected_session).resolve():
        raise RuntimeError(f"Session mismatch: control={session}, expected={expected_session}")
    pid = int(ctl.get("pid"))
    if _pid_alive(pid):
        os.kill(pid, signal.SIGINT)
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.25)
        if _pid_alive(pid):
            os.kill(pid, signal.SIGTERM)
            deadline = time.time() + 10.0
            while time.time() < deadline and _pid_alive(pid):
                time.sleep(0.25)
        if _pid_alive(pid):
            p = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
            cmd = p.stdout.strip()
            if "mm_event_time_m0_m15_exploratory_recorder_v1" not in cmd:
                raise RuntimeError(f"Refusing SIGKILL; command mismatch: {cmd}")
            os.kill(pid, signal.SIGKILL)
    try:
        CONTROL_PATH.unlink()
    except Exception:
        pass
    print("Full15 sidecar stopped. Session preserved:", session)
    return session


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-session", type=str)
    args = ap.parse_args()
    if args.run_session:
        asyncio.run(run_full15_recorder(Path(args.run_session)))


if __name__ == "__main__":
    _main()
