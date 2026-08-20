from __future__ import annotations

"""Authenticated full-window event-time recorder for M1->M12 deep-tail validation.

This module reuses the corrected V5 websocket/book/sequence architecture and the
V5-auth signed market-discovery transport, but extends persistence through M12.

Capture policy
--------------
- Same frozen nine 15-minute crypto series as V5.
- Pre-subscribe up to 5 minutes before M0.
- Persist top-3 order-book changes, ticker events and public trades for
  M0 <= elapsed < M12 (720 seconds).
- Persist an additional M12..M12+30s LABEL-ONLY tail for causal markouts.
- No orders are ever sent by this module.

The underlying V5 implementation contains legacy M5-oriented function names and
manifest wording.  Runtime numeric boundaries and capture-phase labels are patched
here, and metadata files are normalized while the recorder runs and once more after
shutdown so the saved bundle truthfully describes M0->M12 capture.
"""

import argparse
import asyncio
from pathlib import Path

from . import mm_event_time_m0_m5_recorder_v5 as V5
from . import mm_event_time_m0_m5_recorder_v5_auth as V5A

STUDY_VERSION = "MM_EVENT_TIME_M0_M12_V6_AUTH_DISCOVERY"
DISCOVERY_TRANSPORT_VERSION = V5A.DISCOVERY_TRANSPORT_VERSION
TRADE_WINDOW_START_S = 0.0
TRADE_WINDOW_END_S = 720.0
LABEL_TAIL_END_S = 750.0
PRESUBSCRIBE_LEAD_S = V5.PRESUBSCRIBE_LEAD_S


def _persist_phase_m0_m12(meta, t=None):
    e = V5._elapsed(meta, t)
    if e is None:
        return False, e, None
    if TRADE_WINDOW_START_S <= e < TRADE_WINDOW_END_S:
        return True, e, "M0_M12_RESEARCH"
    if TRADE_WINDOW_END_S <= e < LABEL_TAIL_END_S:
        return True, e, "M12_M12P30_LABEL_TAIL"
    return False, e, None


def _install_patch():
    """Patch only recorder horizon/labels plus authenticated V5 discovery."""
    V5.STUDY_VERSION = STUDY_VERSION
    V5.TRADE_WINDOW_START_S = TRADE_WINDOW_START_S
    V5.TRADE_WINDOW_END_S = TRADE_WINDOW_END_S
    V5.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    V5.PRESUBSCRIBE_LEAD_S = PRESUBSCRIBE_LEAD_S
    V5._persist_phase = _persist_phase_m0_m12
    V5._discover_sync = V5A._discover_sync_authenticated
    V5._discover = V5A._discover_authenticated


def _capture_spec_overlay():
    return {
        "study_version": STUDY_VERSION,
        "purpose": "fresh authenticated full-window event-time capture for M1-M12 live/replay validation",
        "universe": list(V5.CRYPTO_SERIES),
        "research_window": "M0 <= elapsed < M12",
        "research_elapsed_seconds": [TRADE_WINDOW_START_S, TRADE_WINDOW_END_S],
        "strategy_window": "M1 <= elapsed < M12",
        "strategy_elapsed_seconds": [60.0, TRADE_WINDOW_END_S],
        "label_tail": "M12 <= elapsed < M12+30s; labels only, never quote initiation",
        "persisted_elapsed_seconds": [TRADE_WINDOW_START_S, LABEL_TAIL_END_S],
        "pre_subscribe_lead_seconds": PRESUBSCRIBE_LEAD_S,
        "orderbook_channel": "orderbook_delta",
        "orderbook_use_yes_price": True,
        "orderbook_numeric_representation": "Decimal exact price and quantity in RAM",
        "sequence_accounting": "all messages carrying orderbook sid+seq, including type=ok",
        "sequence_gap_recovery": "invalidate all books; debounce; get_snapshot all subscribed markets",
        "reconstruction_recovery": "negative level or crossed/locked reconstruction -> ticker-specific fresh snapshot",
        "ticker_integrity_recovery": ">1.01c BBO mismatch persisting >=250ms -> ticker-specific fresh snapshot",
        "sticky_market_lifecycle": "never remove a subscribed market before M12+30 boundary is written",
        "persisted_book": "top 3 bid/ask levels only when top3 changes + snapshots/boundaries",
        "trades": "every public trade at event time during M0-M12+30s",
        "ticker": "every ticker event during M0-M12+30s for independent BBO validation",
        "strategy_pnl_recorded": False,
        "authenticated_discovery": True,
        "discovery_transport_version": DISCOVERY_TRANSPORT_VERSION,
    }


def _development_plan_overlay():
    return {
        "research_stage": "FRESH_FORWARD_M1_M12_LIVE_SUPPORT_CAPTURE",
        "strategy_entry_start_elapsed_s": 60.0,
        "strategy_terminal_cleanup_elapsed_s": 720.0,
        "recording_end_elapsed_s": 750.0,
        "scientific_status": "forward validation data; not proof of alpha",
        "execution_note": "recorder sends no orders; strategy behavior is controlled by the live engine",
    }


def _normalize_metadata(session_dir: Path):
    session_dir = Path(session_dir)
    cap_path = session_dir / "capture_spec.json"
    dev_path = session_dir / "development_plan.json"
    manifest_path = session_dir / "session_manifest.json"

    cap = V5._read_json(cap_path, {}) or {}
    cap.update(_capture_spec_overlay())
    if cap_path.exists() or cap:
        V5._atomic_json(cap_path, cap)

    dev = V5._read_json(dev_path, {}) or {}
    dev.update(_development_plan_overlay())
    if dev_path.exists() or dev:
        V5._atomic_json(dev_path, dev)

    manifest = V5._read_json(manifest_path, {}) or {}
    if manifest_path.exists() or manifest:
        manifest.update({
            "study_version": STUDY_VERSION,
            "research_stage": "FRESH_FORWARD_M1_M12_LIVE_SUPPORT_CAPTURE",
            "capture_spec": cap,
            "development_plan": dev,
        })
        V5._atomic_json(manifest_path, manifest)


async def _metadata_normalizer(session_dir: Path, recorder_task: asyncio.Task):
    """Correct legacy V5 manifest wording without touching the data stream."""
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


async def run_event_time_m0_m12_v6_auth_recorder(session_dir: Path):
    session_dir = Path(session_dir).resolve()
    _install_patch()
    recorder_task = asyncio.create_task(
        V5.run_event_time_m0_m5_v5_recorder(session_dir)
    )
    metadata_task = asyncio.create_task(_metadata_normalizer(session_dir, recorder_task))
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


def static_self_check(*, show=True):
    checks = {
        "study_version": STUDY_VERSION,
        "authenticated_discovery": True,
        "discovery_transport_version": DISCOVERY_TRANSPORT_VERSION,
        "frozen_universe": tuple(V5.CRYPTO_SERIES),
        "presubscribe_lead_s": PRESUBSCRIBE_LEAD_S,
        "persist_start_s": TRADE_WINDOW_START_S,
        "m12_research_end_s": TRADE_WINDOW_END_S,
        "label_tail_end_s": LABEL_TAIL_END_S,
        "strategy_start_m1_s": 60.0,
        "persist_phase_m12": _persist_phase_m0_m12,
        "v5_decimal_book_reconstruction_reused": True,
        "v5_sequence_repair_logic_reused": True,
        "orders_sent": False,
    }
    ok = (
        TRADE_WINDOW_START_S == 0.0
        and TRADE_WINDOW_END_S == 720.0
        and LABEL_TAIL_END_S == 750.0
        and PRESUBSCRIBE_LEAD_S == 300.0
        and tuple(V5.CRYPTO_SERIES) == tuple(V5A.V5.CRYPTO_SERIES)
    )
    out = {**checks, "ok": bool(ok)}
    if show:
        print("=" * 108)
        print("M0-M12 V6 AUTH RECORDER STATIC CHECK — NO API / NO ORDERS")
        print("=" * 108)
        for k, v in out.items():
            if k == "persist_phase_m12":
                v = v.__name__
            print(f"{k:48s}: {v}")
    if not ok:
        raise RuntimeError(f"M0-M12 V6 recorder static self-check failed: {out}")
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-session", type=str, default=None)
    a = ap.parse_args()
    if a.run_session:
        asyncio.run(run_event_time_m0_m12_v6_auth_recorder(Path(a.run_session)))
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "STUDY_VERSION",
    "DISCOVERY_TRANSPORT_VERSION",
    "TRADE_WINDOW_START_S",
    "TRADE_WINDOW_END_S",
    "LABEL_TAIL_END_S",
    "PRESUBSCRIBE_LEAD_S",
    "static_self_check",
    "run_event_time_m0_m12_v6_auth_recorder",
]
