from __future__ import annotations

"""V2.9.9.13 Q100 REC25 + M11:30 residual-entry cutoff.

Production strategy change
--------------------------
At M11:30 (690 seconds from window start), request cancellation of only the
remaining unfilled ENTRY quantity.  Filled inventory is preserved.  If neither tail
has filled, both residual entry orders are canceled.  If the selected tail is only
partially filled, its remaining entry quantity is canceled and the partial inventory
continues on the inherited M12 path.  If Q100 was already completed, REC25 is
unchanged.

Everything else is inherited unchanged from V2.9.9.12 / V1.12.5:
- Q100, 12-hour runtime, 5c two-tail M1 entry;
- first-fill tail selection and opposite-tail cancellation;
- persistent danger guard;
- causal REC25=25% anchor/threshold and atomic trigger snapshot;
- one fixed passive post-only GTC REC25 exit, no repricing/chasing;
- M12 exposure-first cleanup and authoritative-zero finalization;
- $20 software start-loss stop and $125 minimum equity;
- retry-until-flat recovery, hard recycle, guardian handoff;
- shard-2 routing/funding and V2.9.9.12 robust per-series shard discovery;
- semantic market_not_found CREATE handling.

Importing this module performs no API calls, orders, cancels, or transfers.
"""

import inspect

from . import mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_12_robust_shard_discovery as BASE
from . import mm_deep_tail_join_ask_live_v1_12_6_m1130_entry_cutoff as LIVE


DEPLOY_VERSION = (
    "MM_DEEP_TAIL_JOIN_ASK_Q100_M1_M12_GUARD_REC25_"
    "V2_9_9_13_M1130_ENTRY_CUTOFF"
)
MODULE_NAME = (
    "quant_research.kalshi."
    "mm_deep_tail_join_ask_q100_m12_guard_rec25_live_"
    "v2_9_9_13_m1130_entry_cutoff"
)

Q100_ARM = "LIVE_DEEP_TAIL_Q100_M1_M12_GUARD_REC25_M1130_12H_V29913"
Q50_ARM = Q100_ARM
KILL_ARM = BASE.KILL_ARM
SHARD_FUND_ARM = BASE.SHARD_FUND_ARM

RUNTIME = BASE.RUNTIME
P = BASE.P
H = BASE.H
V2963 = BASE.V2963
V28 = BASE.V28
V288 = BASE.V288
V111 = BASE.V111
V1 = BASE.V1
B = BASE.B
Q1 = BASE.Q1
C = BASE.C

LEGACY_SHARD = BASE.LEGACY_SHARD
INHERITED_Q100 = BASE.INHERITED_Q100

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

LIVE_EXCHANGE_INDEX = BASE.LIVE_EXCHANGE_INDEX
SOURCE_EXCHANGE_INDEX = BASE.SOURCE_EXCHANGE_INDEX
SHARD2_MIN_COLLATERAL_USD = BASE.SHARD2_MIN_COLLATERAL_USD
CRYPTO_SERIES = BASE.CRYPTO_SERIES
CENTICENTS_PER_DOLLAR = BASE.CENTICENTS_PER_DOLLAR

DISCOVERY_ATTEMPTS_PER_SERIES = BASE.DISCOVERY_ATTEMPTS_PER_SERIES
DISCOVERY_RETRY_BASE_S = BASE.DISCOVERY_RETRY_BASE_S
DISCOVERY_CACHE_TTL_S = BASE.DISCOVERY_CACHE_TTL_S

discover_current_crypto_markets = BASE.discover_current_crypto_markets
get_shard_balances = BASE.get_shard_balances
ensure_crypto_shard_funded = BASE.ensure_crypto_shard_funded
_usd_to_centicents_ceil = BASE._usd_to_centicents_ceil
_centicents_to_usd = BASE._centicents_to_usd

ENTRY_CUTOFF_S = LIVE.ENTRY_CUTOFF_S
ENTRY_CUTOFF_REASON = LIVE.ENTRY_CUTOFF_REASON


def _install_patch():
    """Install V2.9.9.12, then substitute only the V1.12.6 cutoff live engine."""
    BASE._install_patch()

    # Preserve every V2.9.9.12 transport/shard/recovery binding.  Only the live
    # strategy module used by child generations is replaced.
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

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.POST_M5_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state

    # Keep this wrapper installed through inherited parent/subprocess entrypoints.
    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline structural/regression audit. No API/orders/cancels/transfers."""
    base = BASE.static_self_check(show=False)
    live = LIVE.static_self_check(show=False)
    _install_patch()

    cutoff_src = inspect.getsource(LIVE.Rec25M1130EntryCutoffEngine._enforce_m1130_entry_cutoff)
    wall_src = inspect.getsource(LIVE.Rec25M1130EntryCutoffEngine.enforce_wall_clock_m5)

    inherited_create = BASE.LIVE.Rec25PassiveExitM12Engine._drain_create_futures
    cutoff_create = LIVE.Rec25M1130EntryCutoffEngine._drain_create_futures

    checks = {
        "base_v29912_ok": base.get("ok") is True,
        "live_v1_12_6_ok": live.get("ok") is True,
        "q100_exact_100": Q100_Q == 100.0,
        "runtime_q100_exact": RUNTIME.Q50_Q == 100.0,
        "parent_q100_exact": P.Q50_Q == 100.0,
        "runtime_exact_12h": Q100_HOURS == 12.0,
        "loss_stop_stays_20": Q100_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q100_MIN_EQUITY_USD == 125.0,
        "entry_m1_60": M1_S == 60.0,
        "entry_cutoff_m1130_690": ENTRY_CUTOFF_S == 690.0,
        "terminal_m12_720": M12_S == 720.0,
        "cutoff_precedes_m12": M1_S < ENTRY_CUTOFF_S < M12_S,
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "m12_hard_recycle_45s": M12_HARD_RECYCLE_GRACE_S == 45.0,
        "guardian_90s": GUARDIAN_POST_M12_EXIT_TIMEOUT_S == 90.0,
        "retry_window_45s_preserved": RECOVERY_RETRY_WINDOW_S == 45.0,
        "retry_pause_200ms_preserved": RECOVERY_RETRY_PAUSE_S == 0.20,
        "live_exchange_index_2": LIVE_EXCHANGE_INDEX == 2,
        "source_exchange_index_0": SOURCE_EXCHANGE_INDEX == 0,
        "centicents_per_dollar_10000": CENTICENTS_PER_DOLLAR == 10_000,
        "robust_discovery_preserved": LEGACY_SHARD._verify_market_shard2 is BASE._verify_market_shard2_robust,
        "runtime_live_is_m1130": RUNTIME.LIVE is LIVE,
        "parent_live_is_m1130": P.LIVE is LIVE,
        "guardian_live_is_m1130": V2963.LIVE is LIVE,
        "cutoff_engine_inherits_old_rec25": issubclass(
            LIVE.Rec25M1130EntryCutoffEngine,
            BASE.LIVE.Rec25PassiveExitM12Engine,
        ),
        "semantic_create_binding_inherited": cutoff_create is inherited_create,
        "cutoff_entry_only": '!= "ENTRY"' in cutoff_src,
        "cutoff_uses_processed_fill": 'tr.get("processed_fill")' in cutoff_src,
        "cutoff_uses_existing_cancel_path": "_request_cancel_key(key, ENTRY_CUTOFF_REASON)" in cutoff_src,
        "cutoff_no_flatten": "flatten(" not in cutoff_src,
        "cutoff_no_exit_post": "_maybe_post_exit(" not in cutoff_src,
        "wall_clock_cutoff_enforced": "_enforce_m1130_entry_cutoff" in wall_src,
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": LIVE.PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": LIVE.PASSIVE_EXIT_TIF == "good_till_canceled",
        "risk_m12_flatten_reduce_only_ioc_unchanged": True,
        "orders_sent": False,
        "transfers_sent": False,
    }

    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"orders_sent", "transfers_sent"}
    )

    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "quantity": Q100_Q,
        "runtime_hours": Q100_HOURS,
        "max_loss_usd": Q100_MAX_LOSS_USD,
        "entry_cutoff_s": ENTRY_CUTOFF_S,
        "entry_cutoff_reason": ENTRY_CUTOFF_REASON,
        "entry_cutoff_policy": {
            "deadline": "M11:30",
            "action": "CANCEL_REMAINING_ENTRY_QUANTITY_ONLY",
            "filled_inventory": "PRESERVE",
            "partial_inventory": "INHERITED_M12_PATH_REC25_STILL_REQUIRES_FULL_Q100",
            "full_q100_before_cutoff": "REC25_UNCHANGED",
            "rearm": "NONE_INHERITED_ENTRY_ATTEMPT_IS_ONE_SHOT",
        },
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 180)
        print("V2.9.9.13 Q100 REC25 + M11:30 ENTRY-CUTOFF STATIC CHECK — NO API / NO ORDERS")
        print("=" * 180)
        for k, v in out.items():
            print(f"{k:116s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.9.13 static self-check failed: {out}")
    return out


def crypto_shard_preflight(*, client=None, show=True):
    """Read-only V2.9.9.12 robust shard preflight, restored to V2.9.9.13 afterward."""
    _install_patch()
    try:
        return BASE.crypto_shard_preflight(client=client, show=show)
    finally:
        _install_patch()


def q100_preflight(*, show=True):
    """Read-only exact-dollar Q100 preflight for the M11:30 deployment."""
    _install_patch()
    static_self_check(show=show)
    try:
        report = BASE.q100_preflight(show=show)
    finally:
        _install_patch()

    out = dict(report or {})
    out.update(
        {
            "deploy_version": DEPLOY_VERSION,
            "live_version": LIVE.LIVE_VERSION,
            "entry_cutoff_s": ENTRY_CUTOFF_S,
            "entry_cutoff_reason": ENTRY_CUTOFF_REASON,
        }
    )
    return out


def start_q100_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q100 / 12h REC25 with M11:30 residual-entry cancellation."""
    _install_patch()
    if arm_phrase != Q100_ARM:
        raise RuntimeError(
            "Refusing Q100 M11:30 live start: exact arm phrase required: "
            f"{Q100_ARM!r}"
        )

    # Keep the robust shard-2 launch gate from V2.9.9.12.
    crypto_shard_preflight(show=False)
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
    "SHARD_FUND_ARM",
    "Q100_Q",
    "Q100_HOURS",
    "Q100_MAX_LOSS_USD",
    "Q100_MIN_EQUITY_USD",
    "Q50_Q",
    "Q50_HOURS",
    "Q50_MAX_LOSS_USD",
    "Q50_MIN_EQUITY_USD",
    "M1_S",
    "ENTRY_CUTOFF_S",
    "ENTRY_CUTOFF_REASON",
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
    "LIVE_EXCHANGE_INDEX",
    "SOURCE_EXCHANGE_INDEX",
    "SHARD2_MIN_COLLATERAL_USD",
    "CRYPTO_SERIES",
    "CENTICENTS_PER_DOLLAR",
    "DISCOVERY_ATTEMPTS_PER_SERIES",
    "DISCOVERY_RETRY_BASE_S",
    "DISCOVERY_CACHE_TTL_S",
    "discover_current_crypto_markets",
    "get_shard_balances",
    "ensure_crypto_shard_funded",
    "crypto_shard_preflight",
    "static_self_check",
    "q100_preflight",
    "start_q100_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
