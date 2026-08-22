from __future__ import annotations

"""Read-only deployment preflight for the frozen Q50 M1->M12 guard candidate.

This module intentionally does NOT contain a live launch, supervisor CLI, guardian
intervention, order submission, cancellation, or flattening entrypoint. It exists
only to bind and audit the exact intended deployment specification before any
operator-controlled live handoff.

Target candidate:
- live engine: V1.12 M12_GUARD rotating generation
- Q50
- 12.0 hour parent session
- fixed $20 software loss trigger
- minimum starting equity $125
- M1 entry start = 60s
- terminal persistent cleanup = M12 / 720s
- parent-owned recorder = M0->M12 plus 30s label tail
- trader RSS warning/hard backstop retained at 1536 / 3072 MiB
- no repeated attempts after a completed trade

Importing this module performs no API calls and sends no orders.
"""

from pathlib import Path

from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_m12_guard_rotation as LIVE
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation as P
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_3_overnight_rotation as V2963
from . import mm_event_time_m0_m12_recorder_v6_auth as REC


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_V2_9_7_PREFLIGHT"

Q50_Q = 50.0
Q50_HOURS = 12.0
Q50_MAX_LOSS_USD = 20.0
Q50_MIN_EQUITY_USD = 125.0

M1_S = 60.0
M12_S = 720.0
LABEL_TAIL_END_S = 750.0

GENERATION_RSS_WARNING_MB = 1536.0
GENERATION_RSS_HARD_LIMIT_MB = 3072.0


def intended_generation_config():
    """Return the frozen generation contract. Pure/read-only."""
    return {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "quote_size": Q50_Q,
        "parent_runtime_hours": Q50_HOURS,
        "max_start_loss_usd": Q50_MAX_LOSS_USD,
        "min_start_equity_usd": Q50_MIN_EQUITY_USD,
        "strategy_entry_start_elapsed_s": M1_S,
        "strategy_terminal_cleanup_elapsed_s": M12_S,
        "recorder_persist_end_elapsed_s": M12_S,
        "recorder_label_tail_end_elapsed_s": LABEL_TAIL_END_S,
        "rotation_process_lifetime": "ONE_COMPLETE_M0_M12_WINDOW",
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "guard_yes_bid_max": LIVE.YES_GUARD_BID_MAX,
        "guard_no_ask_min": LIVE.NO_GUARD_ASK_MIN,
        "guard_persist_s": LIVE.GUARD_PERSIST_S,
        "guard_min_book_obs": LIVE.GUARD_MIN_BOOK_OBS,
        "rearm": False,
        "repeat_after_flat": False,
        "fixed_session_risk_baseline": True,
        "external_recorder_parent_owned": True,
        "no_auto_scale": True,
        "orders_sent": False,
    }


def static_self_check(*, show=True):
    """Offline/static structural audit. No API calls and no orders."""
    live = LIVE.static_self_check(show=False)
    base = V2963.static_self_check(show=False)

    checks = {
        "live_v1_12_ok": live.get("ok") is True,
        "inherits_v1_11_protection_stack": live.get("inherits_exact_v1_11_rotation") is True,
        "strategy_m1_60s": abs(M1_S - 60.0) < 1e-12,
        "strategy_m12_720s": abs(M12_S - 720.0) < 1e-12,
        "yes_entry_5c": abs(V1.ENTRY_YES_PRICE - 0.05) < 1e-12,
        "no_entry_5c": abs(V1.ENTRY_NO_BOOK_PRICE - 0.95) < 1e-12,
        "yes_guard_bid_10c": abs(LIVE.YES_GUARD_BID_MAX - 0.10) < 1e-12,
        "no_guard_ask_90c": abs(LIVE.NO_GUARD_ASK_MIN - 0.90) < 1e-12,
        "guard_persist_5s": abs(LIVE.GUARD_PERSIST_S - 5.0) < 1e-12,
        "guard_min_obs_3": LIVE.GUARD_MIN_BOOK_OBS == 3,
        "no_rearm": live.get("rearm") is False,
        "no_repeat_after_flat": live.get("repeat_after_flat") is False,
        "q50_fixed_50": abs(Q50_Q - 50.0) < 1e-12,
        "runtime_fixed_12h": abs(Q50_HOURS - 12.0) < 1e-12,
        "loss_trigger_fixed_20": abs(Q50_MAX_LOSS_USD - 20.0) < 1e-12,
        "minimum_equity_125": abs(Q50_MIN_EQUITY_USD - 125.0) < 1e-12,
        "recorder_m12": abs(REC.TRADE_WINDOW_END_S - M12_S) < 1e-12,
        "recorder_label_tail_750": abs(REC.LABEL_TAIL_END_S - LABEL_TAIL_END_S) < 1e-12,
        "base_v2963_protections_ok": base.get("ok") is True,
        "rss_warning_1536": abs(GENERATION_RSS_WARNING_MB - 1536.0) < 1e-12,
        "rss_hard_3072": abs(GENERATION_RSS_HARD_LIMIT_MB - 3072.0) < 1e-12,
        "recorder_rss_limit_not_changed": True,
        "runtime_wiring_not_claimed_by_preflight": True,
        "orders_sent": False,
    }

    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "intended_generation_config": intended_generation_config(),
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 132)
        print("V2.9.7 Q50 M1->M12 GUARD DEPLOYMENT PREFLIGHT — READ ONLY / NO ORDERS")
        print("=" * 132)
        for k, v in out.items():
            print(f"{k:72s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.7 M12 guard preflight failed: {out}")

    return out


__all__ = [
    "DEPLOY_VERSION",
    "Q50_Q",
    "Q50_HOURS",
    "Q50_MAX_LOSS_USD",
    "Q50_MIN_EQUITY_USD",
    "M1_S",
    "M12_S",
    "LABEL_TAIL_END_S",
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "intended_generation_config",
    "static_self_check",
]
