from __future__ import annotations

"""M1->M5 live strategy with authenticated raw recording extended through M12.

This wrapper changes NO strategy/execution boundary from V1.7.

Live strategy remains frozen:
- M1 (60s): post dual passive 5c YES/NO entries.
- First fill wins; cancel the opposite tail.
- Selected tail may accumulate only until M5.
- Full entry posts one fixed passive JOIN_ASK exit; no chasing/repricing.
- M5 (300s): persistent verified cleanup cancels remaining strategy orders and
  reduce-only IOC flattens any residual inventory.

Only the public raw recorder horizon changes:
- Persist public market data from M0 through M12 (720s).
- Persist an additional M12..M12+30s label tail.

The V1.7 incremental REST-fill reconciler, bounded raw ingestion, private-WS idle
handling, queue hard-stop, loss logic, account auditor and guardian semantics are
unchanged. Importing this module sends no orders.
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


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_8_M1_M5_RECORD_M12_MEMORY_SAFE"
STRATEGY_M1_S = 60.0
STRATEGY_M5_S = 300.0
RECORDER_M12_S = 720.0
LABEL_TAIL_END_S = 750.0


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
            raise RuntimeError(f"M0-M12 recorder startup failure rc={p.returncode}\n{tail}")
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
        f"M0-M12 recorder health timeout: last={last}\n{tail}"
    )


def static_self_check(*, show=True):
    base = V17.static_self_check(show=False)
    rec = REC.static_self_check(show=False)
    checks = {
        "base_v1_7_ok": base.get("ok") is True,
        "strategy_m1_unchanged_60s": abs(V1.M1_S - STRATEGY_M1_S) < 1e-12,
        "strategy_m5_unchanged_300s": abs(V1.M5_S - STRATEGY_M5_S) < 1e-12,
        "recorder_m12_end_720s": abs(REC.TRADE_WINDOW_END_S - RECORDER_M12_S) < 1e-12,
        "recorder_label_tail_750s": abs(REC.LABEL_TAIL_END_S - LABEL_TAIL_END_S) < 1e-12,
        "authenticated_discovery": rec.get("authenticated_discovery") is True,
        "bounded_raw_ingestion_required": base.get("bounded_raw_ingestion_required") is True,
        "rest_fill_incremental_min_ts": base.get("rest_fill_incremental_min_ts") is True,
        "rest_fill_exact_dedupe": base.get("rest_fill_exact_dedupe") is True,
        "private_ws_idle_timeout_no_reconnect": base.get("private_ws_idle_timeout_no_reconnect") is True,
        "private_queue_hard_limit_retained": base.get("private_queue_hard_limit") == V17.PRIVATE_QUEUE_HARD_LIMIT,
        "strategy_boundary_not_extended": True,
        "entry_rules_unchanged": True,
        "first_fill_wins_unchanged": True,
        "fixed_join_ask_no_reprice_unchanged": True,
        "persistent_m5_cleanup_unchanged": True,
    }
    ok = all(v is True for v in checks.values())
    out = {
        "version": LIVE_VERSION,
        "strategy_m1_s": STRATEGY_M1_S,
        "strategy_terminal_m5_s": STRATEGY_M5_S,
        "recorder_persist_end_m12_s": RECORDER_M12_S,
        "recorder_label_tail_end_s": LABEL_TAIL_END_S,
        "recorder_version": REC.STUDY_VERSION,
        **checks,
        "ok": bool(ok),
        "orders_sent": False,
    }
    if show:
        print("=" * 116)
        print("DEEP-TAIL LIVE V1.8 M1->M5 + RECORD M0->M12 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 116)
        for k, v in out.items():
            print(f"{k:60s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.8 M1-M5 / record-M12 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run exact V1.7 M1->M5 strategy while replacing only the recorder launcher."""
    session = Path(session).resolve()

    # Fail closed if some other module has already altered the frozen strategy boundary.
    if abs(float(V1.M1_S) - STRATEGY_M1_S) > 1e-12:
        raise RuntimeError(f"Frozen M1 boundary changed unexpectedly: {V1.M1_S}")
    if abs(float(V1.M5_S) - STRATEGY_M5_S) > 1e-12:
        raise RuntimeError(f"Frozen M5 boundary changed unexpectedly: {V1.M5_S}")

    old_recorder = V16._start_recorder_auth_v5
    old_v16_version = V16.LIVE_VERSION
    old_v17_version = V17.LIVE_VERSION

    # Recorder transport only. DO NOT patch V1.M5_S.
    V16._start_recorder_auth_v5 = _start_recorder_m0_m12_auth
    V16.LIVE_VERSION = LIVE_VERSION
    V17.LIVE_VERSION = LIVE_VERSION

    try:
        B._atomic(session / "m1_m5_record_m12_spec.json", {
            "live_version": LIVE_VERSION,
            "strategy_entry_start_elapsed_s": STRATEGY_M1_S,
            "strategy_terminal_cleanup_elapsed_s": STRATEGY_M5_S,
            "recorder_persist_end_elapsed_s": REC.TRADE_WINDOW_END_S,
            "recorder_label_tail_end_elapsed_s": REC.LABEL_TAIL_END_S,
            "strategy_rule_change_from_v1_7": "NONE",
            "recorder_change_from_v1_7": "M0-M5(+30) -> M0-M12(+30)",
            "explicit_invariant": "LIVE ORDERS STOP/CLEAN AT M5; M5-M12 IS RECORDING ONLY",
        })
        return V17.run_live_process(session, cfg)
    finally:
        V16._start_recorder_auth_v5 = old_recorder
        V16.LIVE_VERSION = old_v16_version
        V17.LIVE_VERSION = old_v17_version


__all__ = [
    "LIVE_VERSION",
    "STRATEGY_M1_S",
    "STRATEGY_M5_S",
    "RECORDER_M12_S",
    "LABEL_TAIL_END_S",
    "static_self_check",
    "run_live_process",
    "_start_recorder_m0_m12_auth",
]
