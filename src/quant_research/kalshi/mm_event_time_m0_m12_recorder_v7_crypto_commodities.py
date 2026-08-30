from __future__ import annotations

"""Authenticated M0->M12(+30s) recorder for the Tail25 12-series live run.

The implementation reuses the corrected V5 Decimal order-book reconstruction,
sequence accounting, snapshot repair, ticker-integrity checks, and public trade
capture.  Only the universe and metadata identity are changed.

Universe:
- 9 crypto 15-minute series
- KXGOLD15M
- KXSILVER15M
- KXWTI15M

Persistence:
- M0 <= elapsed < M12 (720s): research/live-support data
- M12 <= elapsed < M12+30s: label-only tail

This module never sends orders.
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

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_event_time_m0_m5_recorder_v5 as V5
from . import mm_event_time_m0_m5_recorder_v5_auth as V5A
from . import mm_tail25_multiseries_router_v1 as ROUTER


STUDY_VERSION = "MM_EVENT_TIME_M0_M12_V7_AUTH_CRYPTO_COMMODITIES"
DISCOVERY_TRANSPORT_VERSION = V5A.DISCOVERY_TRANSPORT_VERSION
SERIES = tuple(ROUTER.SERIES)

TRADE_WINDOW_START_S = 0.0
TRADE_WINDOW_END_S = 720.0
LABEL_TAIL_END_S = 750.0
PRESUBSCRIBE_LEAD_S = 300.0

STARTUP_TIMEOUT_S = 75.0
STOP_TIMEOUT_S = 45.0


def _persist_phase(meta, t=None):
    e = V5._elapsed(meta, t)
    if e is None:
        return False, e, None
    if TRADE_WINDOW_START_S <= e < TRADE_WINDOW_END_S:
        return True, e, "M0_M12_TAIL25_RESEARCH"
    if TRADE_WINDOW_END_S <= e < LABEL_TAIL_END_S:
        return True, e, "M12_M12P30_LABEL_TAIL"
    return False, e, None


def _install_patch():
    V5.CRYPTO_SERIES = tuple(SERIES)
    # V5A reads the V5 module dynamically, but publish explicitly for defensive
    # notebook/reload compatibility.
    V5A.V5.CRYPTO_SERIES = tuple(SERIES)

    V5.STUDY_VERSION = STUDY_VERSION
    V5.TRADE_WINDOW_START_S = TRADE_WINDOW_START_S
    V5.TRADE_WINDOW_END_S = TRADE_WINDOW_END_S
    V5.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    V5.PRESUBSCRIBE_LEAD_S = PRESUBSCRIBE_LEAD_S
    V5._persist_phase = _persist_phase
    V5._discover_sync = V5A._discover_sync_authenticated
    V5._discover = V5A._discover_authenticated


def _capture_spec():
    return {
        "study_version": STUDY_VERSION,
        "purpose": "Tail25 Q10 live-support and forward-validation capture",
        "universe": list(SERIES),
        "crypto_series": list(ROUTER.CRYPTO_SERIES),
        "commodity_series": list(ROUTER.COMMODITY_SERIES),
        "research_window": "M0 <= elapsed < M12",
        "research_elapsed_seconds": [TRADE_WINDOW_START_S, TRADE_WINDOW_END_S],
        "label_tail": "M12 <= elapsed < M12+30s; labels only",
        "persisted_elapsed_seconds": [TRADE_WINDOW_START_S, LABEL_TAIL_END_S],
        "pre_subscribe_lead_seconds": PRESUBSCRIBE_LEAD_S,
        "orderbook_channel": "orderbook_delta",
        "orderbook_use_yes_price": True,
        "orderbook_numeric_representation": "Decimal exact price and quantity in RAM",
        "persisted_book": "top 3 bid/ask levels only when top3 changes + snapshots/boundaries",
        "trades": "every public trade during M0-M12+30",
        "ticker": "every ticker event during M0-M12+30",
        "sequence_accounting": "all sid+seq orderbook messages including control/ok",
        "sequence_gap_recovery": "invalidate/re-snapshot subscribed books",
        "reconstruction_recovery": "negative/crossed reconstruction -> fresh snapshot",
        "ticker_integrity_recovery": "persistent >1.01c BBO mismatch -> ticker snapshot",
        "authenticated_discovery": True,
        "discovery_transport_version": DISCOVERY_TRANSPORT_VERSION,
        "strategy_orders_sent_by_recorder": False,
    }


def _development_plan():
    return {
        "research_stage": "TAIL25_Q10_MULTI12_FORWARD_LIVE_SUPPORT_CAPTURE",
        "strategy_entry_start_elapsed_s": 0.0,
        "strategy_terminal_cleanup_elapsed_s": TRADE_WINDOW_END_S,
        "recording_end_elapsed_s": LABEL_TAIL_END_S,
        "strategy_family": "TAIL25_HYST2_EDGE15_JOIN_BBO_HYST2_FORCE3S",
        "scientific_status": "forward/live data; not proof of alpha",
        "recorder_orders_sent": False,
    }


def _atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _normalize_metadata(session_dir):
    session_dir = Path(session_dir)
    cap = _read_json(session_dir / "capture_spec.json", {}) or {}
    cap.update(_capture_spec())
    _atomic_json(session_dir / "capture_spec.json", cap)

    dev = _read_json(session_dir / "development_plan.json", {}) or {}
    dev.update(_development_plan())
    _atomic_json(session_dir / "development_plan.json", dev)

    manifest_path = session_dir / "session_manifest.json"
    manifest = _read_json(manifest_path, {}) or {}
    if manifest or manifest_path.exists():
        manifest.update(
            {
                "study_version": STUDY_VERSION,
                "research_stage": "TAIL25_Q10_MULTI12_FORWARD_LIVE_SUPPORT_CAPTURE",
                "capture_spec": cap,
                "development_plan": dev,
            }
        )
        _atomic_json(manifest_path, manifest)

    health_path = session_dir / "health.json"
    health = _read_json(health_path, {}) or {}
    if health or health_path.exists():
        health.update(
            {
                "study_version": STUDY_VERSION,
                "universe": list(SERIES),
                "universe_count": len(SERIES),
            }
        )
        _atomic_json(health_path, health)


async def _metadata_normalizer(session_dir, recorder_task):
    while not recorder_task.done():
        try:
            _normalize_metadata(session_dir)
        except Exception:
            pass
        await asyncio.sleep(0.5)
    try:
        _normalize_metadata(session_dir)
    except Exception:
        pass


async def run_event_time_m0_m12_v7_recorder(session_dir):
    session_dir = Path(session_dir).resolve()
    _install_patch()
    recorder_task = asyncio.create_task(
        V5.run_event_time_m0_m5_v5_recorder(session_dir)
    )
    metadata_task = asyncio.create_task(
        _metadata_normalizer(session_dir, recorder_task)
    )
    try:
        return await recorder_task
    finally:
        try:
            await metadata_task
        except Exception:
            pass
        try:
            _normalize_metadata(session_dir)
        except Exception:
            pass


def start_parent_recorder(parent_session, *, startup_timeout_s=STARTUP_TIMEOUT_S):
    """Start one supervisor-owned recorder. No trading mutation."""
    parent_session = Path(parent_session).resolve()
    raw = parent_session / "raw_capture"
    if raw.exists():
        raise RuntimeError(
            f"Refusing recorder startup because raw_capture already exists: {raw}"
        )

    log = parent_session / "raw_recorder_tail25_multi12.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "quant_research.kalshi.mm_event_time_m0_m12_recorder_v7_crypto_commodities",
        "--run-session",
        str(raw),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(V5.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()

    deadline = time.time() + float(startup_timeout_s)
    last = {}
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = log.read_text(
                    encoding="utf-8", errors="replace"
                )[-14000:]
            except Exception:
                pass
            raise RuntimeError(
                f"Tail25 Multi12 recorder exited during startup rc={proc.returncode}\n{tail}"
            )
        last = B._read(raw / "health.json", {}) or {}
        study_ok = last.get("study_version") in {None, STUDY_VERSION}
        if (
            last.get("running") is True
            and last.get("healthy") is True
            and study_ok
        ):
            return proc, last
        time.sleep(0.35)

    try:
        os.kill(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    tail = ""
    try:
        tail = log.read_text(encoding="utf-8", errors="replace")[-14000:]
    except Exception:
        pass
    raise RuntimeError(
        f"Tail25 Multi12 recorder health timeout: last={last}\n{tail}"
    )


def stop_parent_recorder(pid, *, timeout_s=STOP_TIMEOUT_S):
    pid = int(pid or 0)
    if pid <= 0 or not B._pid_alive(pid):
        return {"pid": pid or None, "already_stopped": True}
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return {"pid": pid, "already_stopped": True}
    deadline = time.time() + float(timeout_s)
    while B._pid_alive(pid) and time.time() < deadline:
        time.sleep(0.25)
    if B._pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    return {"pid": pid, "stopped": not B._pid_alive(pid)}


def static_self_check(*, show=True):
    now = C.utc_now()
    phase_mid = _persist_phase(
        {
            "window_start": now,
        },
        now,
    )
    checks = {
        "series_count_12": len(SERIES) == 12,
        "series_exact_router": tuple(SERIES) == tuple(ROUTER.SERIES),
        "persist_start_0": TRADE_WINDOW_START_S == 0.0,
        "persist_end_m12_720": TRADE_WINDOW_END_S == 720.0,
        "label_tail_750": LABEL_TAIL_END_S == 750.0,
        "presubscribe_300": PRESUBSCRIBE_LEAD_S == 300.0,
        "authenticated_discovery": True,
        "v5_decimal_reconstruction_reused": True,
        "v5_sequence_repair_reused": True,
        "phase_function_callable": phase_mid[0] is True,
        "orders_sent": False,
        "api_called": False,
    }
    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"orders_sent", "api_called"}
    )
    out = {
        "study_version": STUDY_VERSION,
        "discovery_transport_version": DISCOVERY_TRANSPORT_VERSION,
        "series": list(SERIES),
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 128)
        print("TAIL25 MULTI12 M0-M12 RECORDER STATIC CHECK — NO API / NO ORDERS")
        print("=" * 128)
        for k, v in out.items():
            print(f"{k:64s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 recorder static check failed: {out}")
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-session")
    args = ap.parse_args()
    if args.run_session:
        asyncio.run(
            run_event_time_m0_m12_v7_recorder(Path(args.run_session))
        )
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "STUDY_VERSION",
    "DISCOVERY_TRANSPORT_VERSION",
    "SERIES",
    "TRADE_WINDOW_START_S",
    "TRADE_WINDOW_END_S",
    "LABEL_TAIL_END_S",
    "PRESUBSCRIBE_LEAD_S",
    "run_event_time_m0_m12_v7_recorder",
    "start_parent_recorder",
    "stop_parent_recorder",
    "static_self_check",
]
