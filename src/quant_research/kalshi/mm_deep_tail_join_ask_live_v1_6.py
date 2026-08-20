from __future__ import annotations

"""V1.6 operational recorder-transport patch for deep-tail live trading.

No strategy/execution rule changes from V1.5.  The bounded-memory V1.5 engine,
M1/M5 timing, 5c dual-tail entry, first-fill-wins cancellation, fixed JOIN_ASK
exit, persistent M5 cleanup, private execution websocket, account audit and risk
logic are preserved.

The only runtime change is the raw recorder launcher: instead of launching the
original V5 module whose public unauthenticated /markets discovery can be 429
throttled, launch mm_event_time_m0_m5_recorder_v5_auth.  That module preserves
V5 websocket/data semantics and uses signed account GETs only for market
discovery.

Importing this module sends no orders.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_5 as V15
from . import mm_event_time_m0_m5_recorder_v5_auth as V5A

LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_6_AUTH_V5_DISCOVERY"


def _start_recorder_auth_v5(session):
    """V4-style fresh raw_capture startup, but with authenticated V5 discovery."""
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
                "-m", "quant_research.kalshi.mm_event_time_m0_m5_recorder_v5_auth",
                "--run-session", str(raw),
            ],
            cwd=str(C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()

    deadline = time.time() + B.RECORDER_START_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace")[-10000:] if log.exists() else ""
            raise RuntimeError(f"Authenticated V5 recorder startup failure rc={p.returncode}\n{tail}")
        last = B._read(raw / "health.json", {}) or {}
        if last.get("running") and last.get("healthy"):
            return p, last
        time.sleep(0.35)

    try:
        os.kill(p.pid, signal.SIGTERM)
    except Exception:
        pass
    tail = log.read_text(encoding="utf-8", errors="replace")[-10000:] if log.exists() else ""
    raise RuntimeError(
        f"Authenticated V5 recorder health timeout: last={last}\n{tail}"
    )


def static_self_check(*, show=True):
    base = V15.static_self_check(show=False)
    rec = V5A.static_self_check(show=False)
    out = dict(base)
    out.update({
        "version": LIVE_VERSION,
        "bounded_raw_ingestion": base.get("bounded_raw_ingestion") is True,
        "alpha_rules_unchanged_from_v1_5": True,
        "recorder_module": "mm_event_time_m0_m5_recorder_v5_auth",
        "recorder_study_version": V5A.STUDY_VERSION,
        "authenticated_v5_discovery": rec.get("authenticated_discovery") is True,
        "v5_ws_and_persistence_semantics_unchanged": rec.get("websocket_book_logic_unchanged") is True,
        "orders_sent": False,
    })
    out["ok"] = bool(base.get("ok") and rec.get("ok"))
    if show:
        print("=" * 104)
        print("DEEP-TAIL LIVE V1.6 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 104)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    return out


def run_live_process(session, cfg):
    """Run V1.5 exactly, replacing only the recorder launcher after V1 installs it."""
    original_install_runtime = V1._install_runtime
    original_v15_version = V15.LIVE_VERSION

    def install_runtime_with_auth_recorder(session_dir, cfg_obj):
        original_install_runtime(session_dir, cfg_obj)
        B._start_recorder = _start_recorder_auth_v5

    V1._install_runtime = install_runtime_with_auth_recorder
    V15.LIVE_VERSION = LIVE_VERSION
    try:
        return V15.run_live_process(Path(session).resolve(), cfg)
    finally:
        V1._install_runtime = original_install_runtime
        V15.LIVE_VERSION = original_v15_version


__all__ = [
    "LIVE_VERSION", "run_live_process", "static_self_check", "_start_recorder_auth_v5",
]
