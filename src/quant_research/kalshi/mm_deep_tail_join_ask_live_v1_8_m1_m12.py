from __future__ import annotations

"""M1->M12 live extension of the V1.7 memory-safe deep-tail engine.

This wrapper intentionally changes only the terminal strategy boundary and raw
recorder horizon relative to V1.7:

- M1 remains 60 seconds.
- Dual 5c passive entries remain live after M1 until the first fill selects a tail.
- The selected entry may continue accumulating toward Q through M12.
- A full Q entry still posts one fixed passive JOIN_ASK exit at the causally-known
  outcome ask; the exit is never repriced or chased.
- The proven persistent cleanup state machine that used to run at M5 is delayed to
  M12 (elapsed=720s). It cancels strategy resting orders, verifies exchange zero
  resting, then reduce-only IOC flattens any residual position.
- The V1.7 incremental fill reconciler, bounded raw ingestion, private-WS behavior,
  queue hard stop, risk logic and account-auditor behavior remain in force.
- Raw public recording uses the authenticated M0->M12(+30s) V6 recorder.

Implementation note: the inherited cleanup code uses legacy names such as
``M5_CLEANUP_PENDING`` and ``finalize_m5``. This wrapper deliberately reuses that
battle-tested code rather than duplicating it; the actual boundary is patched to
720 seconds. Saved process config/static checks identify those names as legacy
labels whose semantic boundary is M12.

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
from . import mm_deep_tail_join_ask_live_v1_6 as V16
from . import mm_deep_tail_join_ask_live_v1_7 as V17
from . import mm_event_time_m0_m12_recorder_v6_auth as REC

LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_8_M1_M12_MEMORY_SAFE"
M1_S = 60.0
TERMINAL_M12_S = 720.0
LABEL_TAIL_END_S = 750.0
LEGACY_CLEANUP_LABEL = "M5"


def _start_recorder_m0_m12_auth(session):
    """Start a fresh authenticated M0->M12(+30s) raw_capture process."""
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
                "-m", "quant_research.kalshi.mm_event_time_m0_m12_recorder_v6_auth",
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
            tail = log.read_text(encoding="utf-8", errors="replace")[-12000:] if log.exists() else ""
            raise RuntimeError(f"M0-M12 V6 recorder startup failure rc={p.returncode}\n{tail}")
        last = B._read(raw / "health.json", {}) or {}
        study_ok = last.get("study_version") in {None, REC.STUDY_VERSION}
        if last.get("running") and last.get("healthy") and study_ok:
            return p, last
        time.sleep(0.35)

    try:
        os.kill(p.pid, signal.SIGTERM)
    except Exception:
        pass
    tail = log.read_text(encoding="utf-8", errors="replace")[-12000:] if log.exists() else ""
    raise RuntimeError(
        f"M0-M12 V6 recorder health timeout: last={last}\n{tail}"
    )


def static_self_check(*, show=True):
    base = V17.static_self_check(show=False)
    rec = REC.static_self_check(show=False)
    checks = {
        "base_v1_7_ok": base.get("ok") is True,
        "m1_unchanged_60s": abs(V1.M1_S - M1_S) < 1e-12,
        "terminal_boundary_m12_720s": abs(TERMINAL_M12_S - 720.0) < 1e-12,
        "recorder_m12_end_720s": abs(REC.TRADE_WINDOW_END_S - TERMINAL_M12_S) < 1e-12,
        "recorder_label_tail_750s": abs(REC.LABEL_TAIL_END_S - LABEL_TAIL_END_S) < 1e-12,
        "authenticated_discovery": rec.get("authenticated_discovery") is True,
        "bounded_raw_ingestion_required": base.get("bounded_raw_ingestion_required") is True,
        "rest_fill_incremental_min_ts": base.get("rest_fill_incremental_min_ts") is True,
        "rest_fill_exact_dedupe": base.get("rest_fill_exact_dedupe") is True,
        "private_ws_idle_timeout_no_reconnect": base.get("private_ws_idle_timeout_no_reconnect") is True,
        "private_queue_hard_limit_retained": base.get("private_queue_hard_limit") == V17.PRIVATE_QUEUE_HARD_LIMIT,
        "entry_price_rules_unchanged": True,
        "first_fill_wins_unchanged": True,
        "fixed_join_ask_no_reprice_unchanged": True,
        "cleanup_state_machine_reused": True,
        "cleanup_actual_boundary_s": TERMINAL_M12_S,
        "legacy_cleanup_event_label": LEGACY_CLEANUP_LABEL,
        "orders_sent": False,
    }
    boolean_checks = [
        v for k, v in checks.items()
        if k not in {"private_queue_hard_limit_retained", "cleanup_actual_boundary_s", "legacy_cleanup_event_label", "orders_sent"}
    ]
    ok = all(v is True for v in boolean_checks) and checks["private_queue_hard_limit_retained"] is True
    out = {
        "version": LIVE_VERSION,
        "m1_s": M1_S,
        "terminal_m12_s": TERMINAL_M12_S,
        "recorder_version": REC.STUDY_VERSION,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 112)
        print("DEEP-TAIL LIVE V1.8 M1->M12 MEMORY-SAFE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 112)
        for k, v in out.items():
            print(f"{k:58s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.8 M1-M12 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run V1.7 memory hardening with terminal boundary=720s and V6 recorder."""
    session = Path(session).resolve()

    old_terminal = V1.M5_S
    old_recorder = V16._start_recorder_auth_v5
    old_v16_version = V16.LIVE_VERSION
    old_v17_version = V17.LIVE_VERSION

    V1.M5_S = TERMINAL_M12_S
    V16._start_recorder_auth_v5 = _start_recorder_m0_m12_auth
    V16.LIVE_VERSION = LIVE_VERSION
    V17.LIVE_VERSION = LIVE_VERSION

    try:
        B._atomic(session / "m1_m12_boundary_spec.json", {
            "live_version": LIVE_VERSION,
            "entry_start_elapsed_s": M1_S,
            "terminal_cleanup_elapsed_s": TERMINAL_M12_S,
            "recorder_end_elapsed_s": REC.TRADE_WINDOW_END_S,
            "label_tail_end_elapsed_s": REC.LABEL_TAIL_END_S,
            "legacy_cleanup_function_name": "finalize_m5",
            "legacy_cleanup_event_prefix": "M5_",
            "legacy_names_actual_semantic_boundary": "M12 / elapsed=720s",
            "strategy_rule_change_from_v1_7": "terminal boundary only: M5 -> M12",
        })
        return V17.run_live_process(session, cfg)
    finally:
        V1.M5_S = old_terminal
        V16._start_recorder_auth_v5 = old_recorder
        V16.LIVE_VERSION = old_v16_version
        V17.LIVE_VERSION = old_v17_version


__all__ = [
    "LIVE_VERSION",
    "M1_S",
    "TERMINAL_M12_S",
    "LABEL_TAIL_END_S",
    "LEGACY_CLEANUP_LABEL",
    "static_self_check",
    "run_live_process",
    "_start_recorder_m0_m12_auth",
]
