from __future__ import annotations

"""V2.9.9.11 Q100 crypto-shard funding unit fix.

V2.9.9.10 completed the crypto shard-2 routing migration and added an explicit
collateral-transfer helper.  Kalshi's current Intra Account Transfer schema defines
`amount` in *centicents* (1 USD = 10,000 centicents), not cents.  This wrapper keeps
all V2.9.9.10 routing/discovery/risk behavior and replaces only that funding helper
with the correct unit conversion.  It also sends the explicit event-contract source
and destination fields documented by the current endpoint.

Importing this module performs no API calls and sends no orders or transfers.
"""

import inspect
import time
from decimal import Decimal, ROUND_CEILING

from . import mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_10_crypto_shard2_funded as BASE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q100_M1_M12_GUARD_REC25_V2_9_9_11_CRYPTO_SHARD2_CENTICENT_FUNDING"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_11_crypto_shard2_centicent_funding"

Q100_ARM = "LIVE_DEEP_TAIL_Q100_M1_M12_GUARD_REC25_12H_V29911"
Q50_ARM = Q100_ARM
KILL_ARM = BASE.KILL_ARM
SHARD_FUND_ARM = "FUND_KALSHI_CRYPTO_SHARD2_TO_125_V29911"

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
Q1 = BASE.Q1
C = BASE.C

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

CENTICENTS_PER_DOLLAR = Decimal("10000")
TRANSFER_SOURCE = "event_contract"
TRANSFER_DESTINATION = "event_contract"

# Re-export the audited V2.9.9.10 helpers.
discover_current_crypto_markets = BASE.discover_current_crypto_markets
get_shard_balances = BASE.get_shard_balances
crypto_shard_preflight = BASE.crypto_shard_preflight


def _usd_to_centicents_ceil(amount_usd):
    x = Decimal(str(amount_usd))
    if not x.is_finite() or x < 0:
        raise RuntimeError(f"Invalid USD transfer amount: {amount_usd!r}")
    return int((x * CENTICENTS_PER_DOLLAR).to_integral_value(rounding=ROUND_CEILING))


def _centicents_to_usd(amount_centicents):
    return Decimal(int(amount_centicents)) / CENTICENTS_PER_DOLLAR


def ensure_crypto_shard_funded(
    *,
    arm_phrase=None,
    target_usd=SHARD2_MIN_COLLATERAL_USD,
    client=None,
    wait_s=30.0,
):
    """REAL INTERNAL BALANCE TRANSFER; sends no market order.

    Moves only the centicent-rounded shortfall needed to bring primary shard 2 to
    the frozen $125 target, sourcing primary shard 0.  Total Kalshi account equity
    is unchanged.  Requires the exact SHARD_FUND_ARM phrase.
    """
    _install_patch()

    if arm_phrase != SHARD_FUND_ARM:
        raise RuntimeError(
            "Shard funding not armed. "
            f"Pass arm_phrase={SHARD_FUND_ARM!r}."
        )

    target = Decimal(str(target_usd)).quantize(Decimal("0.0001"))
    frozen_target = Decimal(str(SHARD2_MIN_COLLATERAL_USD)).quantize(Decimal("0.0001"))
    if target != frozen_target:
        raise RuntimeError(
            f"This deployment only authorizes the frozen shard target "
            f"${SHARD2_MIN_COLLATERAL_USD:.2f}; got ${target}."
        )

    client = client or Q1.LiveClient()
    before = get_shard_balances(client)
    b = before["breakdown_usd"]
    shard2 = Decimal(str(b.get(LIVE_EXCHANGE_INDEX, 0.0))).quantize(Decimal("0.0001"))
    shard0 = Decimal(str(b.get(SOURCE_EXCHANGE_INDEX, 0.0))).quantize(Decimal("0.0001"))

    if shard2 + Decimal("0.0001") >= target:
        return {
            "ok": True,
            "transfer_sent": False,
            "target_usd": float(target),
            "before": before,
            "after": before,
        }

    shortfall_usd_exact = target - shard2
    shortfall_centicents = _usd_to_centicents_ceil(shortfall_usd_exact)
    transfer_usd = _centicents_to_usd(shortfall_centicents)

    if shard0 + Decimal("0.0001") < transfer_usd:
        raise RuntimeError(
            f"Shard 0 has ${shard0:.4f}, but ${transfer_usd:.4f} is needed "
            f"to bring shard 2 to ${target:.4f}."
        )

    payload = {
        "source": TRANSFER_SOURCE,
        "destination": TRANSFER_DESTINATION,
        "amount": int(shortfall_centicents),
        "source_exchange_shard": int(SOURCE_EXCHANGE_INDEX),
        "destination_exchange_shard": int(LIVE_EXCHANGE_INDEX),
        "source_subaccount": 0,
        "destination_subaccount": 0,
    }

    body, timing = client.post(
        "/portfolio/intra_exchange_instance_transfer",
        payload,
    )

    transfer_id = str((body or {}).get("transfer_id") or "")
    if not transfer_id:
        raise RuntimeError(
            f"Shard transfer response missing transfer_id: {body}"
        )

    deadline = time.time() + float(wait_s)
    after = None
    while time.time() < deadline:
        time.sleep(0.40)
        after = get_shard_balances(client)
        z = Decimal(
            str(after["breakdown_usd"].get(LIVE_EXCHANGE_INDEX, 0.0))
        ).quantize(Decimal("0.0001"))
        if z + Decimal("0.0001") >= target:
            return {
                "ok": True,
                "transfer_sent": True,
                "transfer_id": transfer_id,
                "transfer_amount_centicents": int(shortfall_centicents),
                "transfer_amount_usd": float(transfer_usd),
                "transfer_payload": payload,
                "transfer_body": body,
                "transfer_timing": timing,
                "target_usd": float(target),
                "before": before,
                "after": after,
            }

    raise RuntimeError(
        "Shard transfer request was accepted but shard 2 did not reach the target "
        f"${target:.4f} within {wait_s:.1f}s. "
        f"transfer_id={transfer_id!r} last_balance={after!r}"
    )


def _install_patch():
    """Install complete V2.9.9.10 shard routing, then bind V2.9.9.11 identity."""
    BASE._install_patch()

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

    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline regression audit. No API calls, orders, or transfers."""
    base = BASE.static_self_check(show=False)
    _install_patch()

    funding_src = inspect.getsource(ensure_crypto_shard_funded)

    checks = {
        "base_v29910_ok": base.get("ok") is True,
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
        "exchange_index_exact_2": LIVE_EXCHANGE_INDEX == 2,
        "centicents_per_dollar_exact_10000": CENTICENTS_PER_DOLLAR == Decimal("10000"),
        "one_dollar_is_10000_cc": _usd_to_centicents_ceil(Decimal("1.00")) == 10000,
        "125_dollars_is_1250000_cc": _usd_to_centicents_ceil(Decimal("125.00")) == 1250000,
        "roundtrip_centicent": _centicents_to_usd(1) == Decimal("0.0001"),
        "transfer_source_event_contract": TRANSFER_SOURCE == "event_contract",
        "transfer_destination_event_contract": TRANSFER_DESTINATION == "event_contract",
        "transfer_endpoint_exact": '"/portfolio/intra_exchange_instance_transfer"' in funding_src,
        "transfer_payload_has_source": '"source": TRANSFER_SOURCE' in funding_src,
        "transfer_payload_has_destination": '"destination": TRANSFER_DESTINATION' in funding_src,
        "transfer_payload_uses_centicents": '"amount": int(shortfall_centicents)' in funding_src,
        "funding_requires_exact_arm": "arm_phrase != SHARD_FUND_ARM" in funding_src,
        "funding_only_shortfall": "shortfall_usd_exact = target - shard2" in funding_src,
        "funding_target_matches_min_equity": SHARD2_MIN_COLLATERAL_USD == Q100_MIN_EQUITY_USD == 125.0,
        "dynamic_discovery_reused": discover_current_crypto_markets is BASE.discover_current_crypto_markets,
        "transport_router_reused": BASE.LIVE_EXCHANGE_INDEX == 2,
        "semantic_market_not_found_binding_preserved": LIVE.Rec25PassiveExitM12Engine._drain_create_futures is BASE.BASE.SEMANTIC_DRAIN,
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
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "quantity": Q100_Q,
        "runtime_hours": Q100_HOURS,
        "live_exchange_index": LIVE_EXCHANGE_INDEX,
        "shard2_min_collateral_usd": SHARD2_MIN_COLLATERAL_USD,
        "funding_unit": "CENTICENTS_1_USD_EQUALS_10000",
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 188)
        print("V2.9.9.11 Q100 CRYPTO-SHARD2 CENTICENT FUNDING STATIC CHECK — NO API / NO ORDERS / NO TRANSFERS")
        print("=" * 188)
        for k, v in out.items():
            print(f"{k:116s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.9.11 static self-check failed: {out}")

    return out


def q100_preflight(*, show=True):
    """Read-only official Q100 preflight after explicit shard funding."""
    _install_patch()
    static_self_check(show=show)

    # V2.9.9.10 shard preflight is read-only and is safe once Cell 2 has funded.
    shard = BASE.crypto_shard_preflight(show=show)

    # Run inherited private-WS/API/exact-equity checks, then restore this identity.
    report = BASE.BASE.q100_preflight(show=show)
    _install_patch()

    out = dict(report or {})
    out["crypto_shard"] = shard
    out["deploy_version"] = DEPLOY_VERSION
    out["ok"] = bool((report or {}).get("ok", True) and shard.get("ok") is True)
    return out


def start_q100_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q100 / 12h. Refuses launch unless shard 2 is funded/clean."""
    _install_patch()

    if arm_phrase != Q100_ARM:
        raise RuntimeError(f"Wrong Q100 arm phrase; expected {Q100_ARM!r}")

    BASE.crypto_shard_preflight(show=False)
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
    "LIVE_EXCHANGE_INDEX",
    "SOURCE_EXCHANGE_INDEX",
    "SHARD2_MIN_COLLATERAL_USD",
    "CENTICENTS_PER_DOLLAR",
    "CRYPTO_SERIES",
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
