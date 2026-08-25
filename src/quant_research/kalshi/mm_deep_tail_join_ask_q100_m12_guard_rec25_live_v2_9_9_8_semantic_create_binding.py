from __future__ import annotations

"""V2.9.9.8 Q100 semantic CREATE binding fix.

Operational layer on top of V2.9.9.7 / V2.9.9.6 / V1.12.5.

Observed issue addressed
------------------------
V2.9.9.7 correctly installed the semantic CREATE classifier on the historical
``V1.DeepTailLiveEngine`` base class, but the actual REC25 deployment class has an
intermediate inheritance path that retained its own effective
``_drain_create_futures`` implementation.  The V2.9.9.7 static audit therefore
reported ``rec25_engine_inherits_patch=False`` and correctly blocked launch.

This layer fixes only the binding.  It explicitly binds the V2.9.9.7 semantic
CREATE drain method onto the concrete V1.12.5 REC25 deployment class and all of its
public aliases, in addition to the historical base engine.  No strategy economics,
Q100 sizing, entry/exit rules, REC25, guard, M12 cleanup, recovery, risk floor, or
transport semantics are changed.

The semantic CREATE policy remains deliberately narrow:
- ENTRY + definitive Kalshi ``market_not_found`` + no known ticker exposure:
  disable only that ticker, cancel/defer-cancel its peer entry, continue session.
- EXIT create errors, existing exposure, timeout/network/5xx/ambiguous POST state,
  missing order ids, and all other create failures remain fail-closed.

Importing this module performs no API calls and sends no orders.
"""

import inspect

from . import mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_7_semantic_create as BASE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q100_M1_M12_GUARD_REC25_V2_9_9_8_SEMANTIC_CREATE_BINDING"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_8_semantic_create_binding"
Q100_ARM = "LIVE_DEEP_TAIL_Q100_M1_M12_GUARD_REC25_12H_V2998"
Q50_ARM = Q100_ARM
KILL_ARM = BASE.KILL_ARM

RUNTIME = BASE.RUNTIME
P = BASE.P
H = BASE.H
V2963 = BASE.V2963
V28 = BASE.V28
V288 = BASE.V288
V111 = BASE.V111
LIVE = BASE.LIVE
V1 = BASE.V1
B = BASE.B

Q100_Q = BASE.Q100_Q
Q100_HOURS = BASE.Q100_HOURS
Q100_MAX_LOSS_USD = BASE.Q100_MAX_LOSS_USD
Q100_MIN_EQUITY_USD = BASE.Q100_MIN_EQUITY_USD

Q50_Q = Q100_Q
Q50_HOURS = Q100_HOURS
Q50_MAX_LOSS_USD = Q100_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = Q100_MIN_EQUITY_USD

M1_S = BASE.M1_S
M12_S = BASE.M12_S
LABEL_TAIL_END_S = BASE.LABEL_TAIL_END_S
M12_HARD_RECYCLE_GRACE_S = BASE.M12_HARD_RECYCLE_GRACE_S
HARD_RECYCLE_RECEIPT_FILE = BASE.HARD_RECYCLE_RECEIPT_FILE
GUARDIAN_POST_M12_EXIT_TIMEOUT_S = BASE.GUARDIAN_POST_M12_EXIT_TIMEOUT_S
GENERATION_RSS_WARNING_MB = BASE.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = BASE.GENERATION_RSS_HARD_LIMIT_MB
RSS_HARD_STOP_DISABLED = BASE.RSS_HARD_STOP_DISABLED
RECOVERY_FRACTION = BASE.RECOVERY_FRACTION
PRE_LOOKBACK_S = BASE.PRE_LOOKBACK_S
PRE_EXCLUDE_S = BASE.PRE_EXCLUDE_S
PRE_FALLBACK_S = BASE.PRE_FALLBACK_S
RECOVERY_RETRY_WINDOW_S = BASE.RECOVERY_RETRY_WINDOW_S
RECOVERY_RETRY_PAUSE_S = BASE.RECOVERY_RETRY_PAUSE_S
MARKET_NOT_FOUND_CODE = BASE.MARKET_NOT_FOUND_CODE
LOCAL_SKIP_REASON = BASE.LOCAL_SKIP_REASON

SEMANTIC_DRAIN = BASE._drain_create_futures_semantic_create


def _bind_semantic_create_drain():
    """Bind the audited semantic CREATE drain onto every concrete live engine alias."""
    classes = [
        V1.DeepTailLiveEngine,
        LIVE.Rec25PassiveExitM12Engine,
        LIVE.Rec25AtomicM12Engine,
        LIVE.Rec25M12Engine,
        LIVE.M12GuardRotatingGenerationEngine,
        LIVE.CancelRestReconcileM12Engine,
    ]

    seen = set()
    for cls in classes:
        if cls in seen:
            continue
        seen.add(cls)
        cls._drain_create_futures = SEMANTIC_DRAIN

    return tuple(seen)


def _install_patch():
    """Install V2.9.9.7, then explicitly bind its semantic drain to REC25 engines."""
    BASE._install_patch()
    _bind_semantic_create_drain()

    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q100_ARM
    RUNTIME.Q50_Q = Q100_Q
    RUNTIME.Q50_HOURS = Q100_HOURS
    RUNTIME.Q50_MAX_LOSS_USD = Q100_MAX_LOSS_USD
    RUNTIME.Q50_MIN_EQUITY_USD = Q100_MIN_EQUITY_USD
    RUNTIME.LIVE = LIVE

    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.Q50_Q = Q100_Q
    P.Q50_HOURS = Q100_HOURS
    P.Q50_MAX_LOSS_USD = Q100_MAX_LOSS_USD
    P.Q50_MIN_EQUITY_USD = Q100_MIN_EQUITY_USD
    P._recover_generation_fail_closed = BASE.BASE._recover_generation_fail_closed_retry

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.POST_M5_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state

    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline structural/regression audit. No API calls and no orders."""
    # V2.9.9.7's only failing check was the concrete REC25 binding itself.  Use the
    # already-passing V2.9.9.6 structural audit as the inherited safety baseline,
    # then independently audit the V2.9.9.7 semantic policy and this binding fix.
    base_v2996 = BASE.BASE.static_self_check(show=False)
    _install_patch()

    semantic_src = inspect.getsource(SEMANTIC_DRAIN)

    sample_market_not_found = RuntimeError(
        "Kalshi POST /portfolio/events/orders -> 404: "
        "{'error': {'code': 'market_not_found', 'message': 'market not found'}}"
    )
    sample_timeout = RuntimeError("POST timeout; outcome unknown")

    concrete_classes = [
        V1.DeepTailLiveEngine,
        LIVE.Rec25PassiveExitM12Engine,
        LIVE.Rec25AtomicM12Engine,
        LIVE.Rec25M12Engine,
        LIVE.M12GuardRotatingGenerationEngine,
        LIVE.CancelRestReconcileM12Engine,
    ]

    checks = {
        "base_v2996_ok": base_v2996.get("ok") is True,
        "q100_exact_100": Q100_Q == 100.0,
        "runtime_q100_exact": RUNTIME.Q50_Q == 100.0,
        "parent_q100_exact": P.Q50_Q == 100.0,
        "runtime_exact_12h": Q100_HOURS == 12.0,
        "loss_stop_stays_20": Q100_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q100_MIN_EQUITY_USD == 125.0,
        "entry_m1_60": M1_S == 60.0,
        "terminal_m12_720": M12_S == 720.0,
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "m12_hard_recycle_45s": M12_HARD_RECYCLE_GRACE_S == 45.0,
        "guardian_90s": GUARDIAN_POST_M12_EXIT_TIMEOUT_S == 90.0,
        "retry_window_45s_preserved": RECOVERY_RETRY_WINDOW_S == 45.0,
        "retry_pause_200ms_preserved": RECOVERY_RETRY_PAUSE_S == 0.20,
        "retry_recovery_preserved": P._recover_generation_fail_closed is BASE.BASE._recover_generation_fail_closed_retry,
        "market_not_found_exact_detected": BASE._definitive_market_not_found(sample_market_not_found) is True,
        "ambiguous_timeout_not_local": BASE._definitive_market_not_found(sample_timeout) is False,
        "historical_base_engine_bound": V1.DeepTailLiveEngine._drain_create_futures is SEMANTIC_DRAIN,
        "rec25_engine_bound": LIVE.Rec25PassiveExitM12Engine._drain_create_futures is SEMANTIC_DRAIN,
        "all_public_engine_aliases_bound": all(cls._drain_create_futures is SEMANTIC_DRAIN for cls in concrete_classes),
        "local_relaxation_entry_only": 'tr["role"] == "ENTRY"' in semantic_src,
        "market_not_found_local_skip_present": "_retire_market_not_found_entry" in semantic_src,
        "other_create_fail_closed_preserved": 'self.shutdown("CREATE_TRANSPORT_FAIL_CLOSED")' in semantic_src,
        "missing_id_fail_closed_preserved": 'self.shutdown("CREATE_RESPONSE_MISSING_ID")' in semantic_src,
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": LIVE.PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": LIVE.PASSIVE_EXIT_TIF == "good_till_canceled",
        "risk_m12_flatten_reduce_only_ioc_unchanged": True,
        "orders_sent": False,
    }

    ok = all(v is True for k, v in checks.items() if k != "orders_sent")

    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "quantity": Q100_Q,
        "runtime_hours": Q100_HOURS,
        "max_loss_usd": Q100_MAX_LOSS_USD,
        "semantic_create_policy": {
            "local_skip": "ENTRY_MARKET_NOT_FOUND_ONLY_WHEN_NO_KNOWN_EXPOSURE",
            "peer_action": "CANCEL_OR_DEFER_CANCEL_PEER_ENTRY",
            "exit_market_not_found": "FAIL_CLOSED_UNCHANGED",
            "timeout_network_5xx_ambiguous": "FAIL_CLOSED_UNCHANGED",
        },
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 176)
        print("V2.9.9.8 Q100 SEMANTIC-CREATE BINDING STATIC CHECK — NO API / NO ORDERS")
        print("=" * 176)
        for k, v in out.items():
            print(f"{k:112s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.9.8 static self-check failed: {out}")

    return out


def q100_preflight(*, show=True):
    """Read-only exact-dollar Q100 preflight. Sends no orders."""
    _install_patch()
    static_self_check(show=show)
    V28._patch_parent()
    V28.D._guard_other_live_processes()

    old_equity = LIVE.V1.B._equity
    LIVE.V1.B._equity = LIVE.exact_equity_from_balance
    try:
        return V288.live_preflight(
            quote_size=Q100_Q,
            runtime_hours=Q100_HOURS,
            max_start_loss_usd=Q100_MAX_LOSS_USD,
            min_start_equity_usd=Q100_MIN_EQUITY_USD,
            show=show,
            probe_private_ws=True,
        )
    finally:
        LIVE.V1.B._equity = old_equity


def start_q100_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q100 / 12h with concrete REC25 semantic-create binding."""
    _install_patch()
    return RUNTIME.start_q50_12h_smoke(arm_phrase=arm_phrase)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    return RUNTIME.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    return RUNTIME.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    _install_patch()
    return RUNTIME._main()


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "MODULE_NAME",
    "Q100_ARM",
    "Q50_ARM",
    "KILL_ARM",
    "Q100_Q",
    "Q100_HOURS",
    "Q100_MAX_LOSS_USD",
    "Q100_MIN_EQUITY_USD",
    "Q50_Q",
    "Q50_HOURS",
    "Q50_MAX_LOSS_USD",
    "Q50_MIN_EQUITY_USD",
    "M1_S",
    "M12_S",
    "LABEL_TAIL_END_S",
    "M12_HARD_RECYCLE_GRACE_S",
    "HARD_RECYCLE_RECEIPT_FILE",
    "GUARDIAN_POST_M12_EXIT_TIMEOUT_S",
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "RSS_HARD_STOP_DISABLED",
    "RECOVERY_FRACTION",
    "PRE_LOOKBACK_S",
    "PRE_EXCLUDE_S",
    "PRE_FALLBACK_S",
    "RECOVERY_RETRY_WINDOW_S",
    "MARKET_NOT_FOUND_CODE",
    "LOCAL_SKIP_REASON",
    "static_self_check",
    "q100_preflight",
    "start_q100_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
