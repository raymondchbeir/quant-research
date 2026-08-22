from __future__ import annotations

"""V1.12.1 M12 guard engine with an explicit V1.8 compatibility bridge.

Why this layer exists
---------------------
V1.12 intentionally rebinds the inherited V1 ``M5_S`` cleanup clock from 300s to
720s so the frozen M12_GUARD can use the already-audited persistent cleanup and
rotation machinery at M12.  The runtime call chain still passes through V1.8,
whose historical M1->M5 wrapper contains a fail-closed assertion that
``V1.M5_S == 300``.  That assertion correctly protected the old strategy but
prevents the intentional V1.12 M12 binding before any order engine can start.

This module keeps the V1.12 trading logic unchanged and replaces only that one
historical V1.8 wrapper during the child process.  The bridge reproduces V1.8's
recorder/version handoff, but validates the intentional 720s terminal horizon and
writes truthful M1->M12 metadata.  All V1.7/V1.6/V1.5 execution, reconciliation,
memory, private-WS, account-audit, cleanup and fail-closed layers remain in the
same call chain.

Importing this module performs no API calls and sends no orders.
"""

import tempfile
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_6 as V16
from . import mm_deep_tail_join_ask_live_v1_7 as V17
from . import mm_deep_tail_join_ask_live_v1_8_record_m12 as V18
from . import mm_deep_tail_join_ask_live_v1_12_m12_guard_rotation as BASE


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_12_1_M12_GUARD_V18_COMPAT"
M12_S = BASE.M12_S
GUARD_PERSIST_S = BASE.GUARD_PERSIST_S
GUARD_MIN_BOOK_OBS = BASE.GUARD_MIN_BOOK_OBS
YES_GUARD_BID_MAX = BASE.YES_GUARD_BID_MAX
NO_GUARD_ASK_MIN = BASE.NO_GUARD_ASK_MIN
M12GuardRotatingGenerationEngine = BASE.M12GuardRotatingGenerationEngine


def _run_v18_m12_compat(session, cfg):
    """V1.8 recorder handoff with the intentional V1.12 720s terminal boundary."""
    session = Path(session).resolve()

    if abs(float(V1.M1_S) - 60.0) > 1e-12:
        raise RuntimeError(f"M12 compat: M1 boundary changed unexpectedly: {V1.M1_S}")
    if abs(float(V1.M5_S) - M12_S) > 1e-12:
        raise RuntimeError(
            f"M12 compat: expected inherited cleanup clock at 720s, got {V1.M5_S}"
        )

    old_recorder = V16._start_recorder_auth_v5
    old_v16_version = V16.LIVE_VERSION
    old_v17_version = V17.LIVE_VERSION

    # Same recorder transport substitution as V1.8.  Under rotating V1.11 the
    # actual recorder is supervisor-owned and B._run_process is replaced by the
    # external-recorder runner, but retaining this handoff preserves the exact
    # fallback call-chain contract.
    V16._start_recorder_auth_v5 = V18._start_recorder_m0_m12_auth
    V16.LIVE_VERSION = LIVE_VERSION
    V17.LIVE_VERSION = LIVE_VERSION

    try:
        B._atomic(
            session / "m1_m12_record_m12_spec.json",
            {
                "live_version": LIVE_VERSION,
                "strategy_entry_start_elapsed_s": 60.0,
                "strategy_terminal_cleanup_elapsed_s": M12_S,
                "recorder_persist_end_elapsed_s": V18.REC.TRADE_WINDOW_END_S,
                "recorder_label_tail_end_elapsed_s": V18.REC.LABEL_TAIL_END_S,
                "compatibility_bridge": "V1_8_HISTORICAL_M5_ASSERTION_TO_INTENTIONAL_M12",
                "strategy_rule_change_from_v1_12": "NONE",
                "explicit_invariant": (
                    "LIVE ORDERS MAY REST/EXIT FROM M1 UNTIL TERMINAL VERIFIED CLEANUP AT M12"
                ),
            },
        )
        return V17.run_live_process(session, cfg)
    finally:
        V16._start_recorder_auth_v5 = old_recorder
        V16.LIVE_VERSION = old_v16_version
        V17.LIVE_VERSION = old_v17_version


def _offline_v18_bridge_regression():
    """Pure regression proving the compatibility bridge accepts 720s without API use."""
    old_m5 = V1.M5_S
    old_run = V17.run_live_process

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sentinel = {"bridge_reached_v17": True}
        try:
            V1.M5_S = M12_S
            V17.run_live_process = lambda session, cfg: sentinel
            out = _run_v18_m12_compat(root, {})
            spec = B._read(root / "m1_m12_record_m12_spec.json", {}) or {}
        finally:
            V17.run_live_process = old_run
            V1.M5_S = old_m5

    return {
        "bridge_reached_v17": out is sentinel,
        "spec_terminal_720": spec.get("strategy_terminal_cleanup_elapsed_s") == 720.0,
        "spec_recorder_720": spec.get("recorder_persist_end_elapsed_s") == 720.0,
        "spec_tail_750": spec.get("recorder_label_tail_end_elapsed_s") == 750.0,
        "historical_v18_guard_not_weakened_globally": V18.STRATEGY_M5_S == 300.0,
        "orders_sent": False,
    }


def static_self_check(*, show=True):
    """Offline structural + regression check; no API calls and no orders."""
    base = BASE.static_self_check(show=False)
    reg = _offline_v18_bridge_regression()

    checks = {
        "base_v1_12_ok": base.get("ok") is True,
        "m12_cleanup_horizon_720": M12_S == 720.0,
        "guard_yes_bid_10c": YES_GUARD_BID_MAX == 0.10,
        "guard_no_ask_90c": NO_GUARD_ASK_MIN == 0.90,
        "guard_persist_5s": GUARD_PERSIST_S == 5.0,
        "guard_min_obs_3": GUARD_MIN_BOOK_OBS == 3,
        "v18_bridge_reaches_v17_at_720": reg.get("bridge_reached_v17") is True,
        "v18_bridge_truthful_terminal_metadata": reg.get("spec_terminal_720") is True,
        "v18_bridge_recorder_m12": reg.get("spec_recorder_720") is True,
        "v18_bridge_label_tail_750": reg.get("spec_tail_750") is True,
        "historical_v18_300_guard_unchanged_outside_child_bridge": (
            reg.get("historical_v18_guard_not_weakened_globally") is True
        ),
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "base_version": BASE.LIVE_VERSION,
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 136)
        print("V1.12.1 M12 V1.8 COMPATIBILITY STATIC CHECK — NO API / NO ORDERS")
        print("=" * 136)
        for k, v in out.items():
            print(f"{k:76s}: {v}")

    if not ok:
        raise RuntimeError(f"V1.12.1 M12 compatibility self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run exact V1.12 M12_GUARD with only the historical V1.8 guard bridged."""
    old_v18_run = V18.run_live_process
    old_base_version = BASE.LIVE_VERSION

    V18.run_live_process = _run_v18_m12_compat
    BASE.LIVE_VERSION = LIVE_VERSION
    try:
        return BASE.run_live_process(Path(session).resolve(), cfg)
    finally:
        BASE.LIVE_VERSION = old_base_version
        V18.run_live_process = old_v18_run


__all__ = [
    "LIVE_VERSION",
    "M12_S",
    "GUARD_PERSIST_S",
    "GUARD_MIN_BOOK_OBS",
    "YES_GUARD_BID_MAX",
    "NO_GUARD_ASK_MIN",
    "M12GuardRotatingGenerationEngine",
    "static_self_check",
    "run_live_process",
]
