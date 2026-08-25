from __future__ import annotations

"""V2.9.9.14 Q100 REC25 M11:30 runtime-facade compatibility fix.

V2.9.9.13 correctly introduced the frozen strategy change: at 690 seconds cancel
only remaining unfilled ENTRY quantity, while preserving filled inventory and
leaving REC25/M12/guard/risk semantics unchanged.

Observed notebook failure
-------------------------
The inherited V2.9.8 ``live_status`` routine intentionally reads the historical
V1.11 rotation checkpoint through ``RUNTIME.LIVE.ROTATION_CHECKPOINT_FILE``.
V2.9.8 normally publishes the V1.11 artifact aliases onto its V1.12 LIVE facade.
V2.9.9.13 then substituted the new V1.12.6 LIVE module *after* that publication,
so the replacement facade did not carry ``ROTATION_CHECKPOINT_FILE`` (or the
companion generation-bootstrap alias).  As a result, the read-only Cell-2
``live_status`` call raised AttributeError before preflight.

This wrapper changes NO trading rule.  It preserves V2.9.9.13 exactly and restores
only the historical runtime-facade artifact names expected by the inherited
supervisor/status stack:
- ROTATION_CHECKPOINT_FILE
- GENERATION_BOOTSTRAP_FILE
- SESSION_RISK_BASELINE_FILE (defensive compatibility export)

Importing this module performs no API calls, orders, cancels, or transfers.
"""

from . import mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_13_m1130_entry_cutoff as BASE


DEPLOY_VERSION = (
    "MM_DEEP_TAIL_JOIN_ASK_Q100_M1_M12_GUARD_REC25_"
    "V2_9_9_14_M1130_RUNTIME_COMPAT"
)
MODULE_NAME = (
    "quant_research.kalshi."
    "mm_deep_tail_join_ask_q100_m12_guard_rec25_live_"
    "v2_9_9_14_m1130_runtime_compat"
)

Q100_ARM = "LIVE_DEEP_TAIL_Q100_M1_M12_GUARD_REC25_M1130_12H_V29914"
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
LIVE = BASE.LIVE

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
ENTRY_CUTOFF_S = BASE.ENTRY_CUTOFF_S
ENTRY_CUTOFF_REASON = BASE.ENTRY_CUTOFF_REASON
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

# Historical V1.11 artifact names are part of the inherited supervisor/runtime
# interface even though the strategy horizon is now M12 and the live engine is
# V1.12.6.  Keep these values exact; they are filenames, not strategy rules.
ROTATION_CHECKPOINT_FILE = V111.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V111.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V111.SESSION_RISK_BASELINE_FILE


def _publish_runtime_facade_compat():
    """Publish the historical artifact aliases onto the replacement LIVE facade."""
    LIVE.ROTATION_CHECKPOINT_FILE = ROTATION_CHECKPOINT_FILE
    LIVE.GENERATION_BOOTSTRAP_FILE = GENERATION_BOOTSTRAP_FILE
    LIVE.SESSION_RISK_BASELINE_FILE = SESSION_RISK_BASELINE_FILE


def _install_patch():
    """Install V2.9.9.13, then restore only inherited facade compatibility."""
    BASE._install_patch()
    _publish_runtime_facade_compat()

    # V2.9.9.13 strategy/transport/risk bindings remain exact; publish only this
    # wrapper's operational identity/arm phrase so detached subprocesses reload
    # the compatibility layer too.
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

    # Inherited entrypoints call these module globals dynamically.
    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline regression/compatibility audit. No API calls and no orders."""
    base = BASE.static_self_check(show=False)
    _install_patch()

    checks = {
        "base_v29913_ok": base.get("ok") is True,
        "strategy_live_engine_still_v1_12_6": (
            LIVE.LIVE_VERSION
            == "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_12_6_REC25_M1130_ENTRY_CUTOFF"
        ),
        "q100_exact_100": Q100_Q == 100.0,
        "runtime_q100_exact": RUNTIME.Q50_Q == 100.0,
        "parent_q100_exact": P.Q50_Q == 100.0,
        "runtime_exact_12h": Q100_HOURS == 12.0,
        "loss_stop_stays_20": Q100_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q100_MIN_EQUITY_USD == 125.0,
        "entry_m1_60": M1_S == 60.0,
        "entry_cutoff_m1130_690": ENTRY_CUTOFF_S == 690.0,
        "terminal_m12_720": M12_S == 720.0,
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "live_exchange_index_2": LIVE_EXCHANGE_INDEX == 2,
        "source_exchange_index_0": SOURCE_EXCHANGE_INDEX == 0,
        "runtime_live_is_m1130": RUNTIME.LIVE is LIVE,
        "parent_live_is_m1130": P.LIVE is LIVE,
        "guardian_live_is_m1130": V2963.LIVE is LIVE,
        "rotation_checkpoint_alias_exact": (
            getattr(LIVE, "ROTATION_CHECKPOINT_FILE", None)
            == V111.ROTATION_CHECKPOINT_FILE
        ),
        "generation_bootstrap_alias_exact": (
            getattr(LIVE, "GENERATION_BOOTSTRAP_FILE", None)
            == V111.GENERATION_BOOTSTRAP_FILE
        ),
        "session_risk_baseline_alias_exact": (
            getattr(LIVE, "SESSION_RISK_BASELINE_FILE", None)
            == V111.SESSION_RISK_BASELINE_FILE
        ),
        "robust_discovery_preserved": (
            LEGACY_SHARD._verify_market_shard2 is BASE.BASE._verify_market_shard2_robust
        ),
        "semantic_create_binding_preserved": base.get("semantic_create_binding_inherited") is True,
        "cutoff_entry_only_preserved": base.get("cutoff_entry_only") is True,
        "cutoff_uses_processed_fill_preserved": base.get("cutoff_uses_processed_fill") is True,
        "cutoff_uses_existing_cancel_path_preserved": base.get("cutoff_uses_existing_cancel_path") is True,
        "cutoff_no_flatten_preserved": base.get("cutoff_no_flatten") is True,
        "cutoff_no_exit_post_preserved": base.get("cutoff_no_exit_post") is True,
        "wall_clock_cutoff_enforced": base.get("wall_clock_cutoff_enforced") is True,
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": LIVE.PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": LIVE.PASSIVE_EXIT_TIF == "good_till_canceled",
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
        "base_deploy_version": BASE.DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "quantity": Q100_Q,
        "runtime_hours": Q100_HOURS,
        "entry_cutoff_s": ENTRY_CUTOFF_S,
        "entry_cutoff_reason": ENTRY_CUTOFF_REASON,
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 184)
        print("V2.9.9.14 Q100 M11:30 RUNTIME-COMPAT STATIC CHECK — NO API / NO ORDERS")
        print("=" * 184)
        for k, v in out.items():
            print(f"{k:120s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.9.14 static self-check failed: {out}")
    return out


def crypto_shard_preflight(*, client=None, show=True):
    """Read-only robust shard preflight; restore this wrapper afterward."""
    _install_patch()
    try:
        return BASE.crypto_shard_preflight(client=client, show=show)
    finally:
        _install_patch()


def q100_preflight(*, show=True):
    """Read-only exact-dollar Q100 preflight for the M11:30 strategy."""
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
            "runtime_facade_compat": True,
        }
    )
    return out


def start_q100_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q100 / 12h M11:30 deployment; trading rules unchanged from V2.9.9.13."""
    _install_patch()
    if arm_phrase != Q100_ARM:
        raise RuntimeError(
            "Refusing Q100 M11:30 live start: exact arm phrase required: "
            f"{Q100_ARM!r}"
        )

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
    "ROTATION_CHECKPOINT_FILE",
    "GENERATION_BOOTSTRAP_FILE",
    "SESSION_RISK_BASELINE_FILE",
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
