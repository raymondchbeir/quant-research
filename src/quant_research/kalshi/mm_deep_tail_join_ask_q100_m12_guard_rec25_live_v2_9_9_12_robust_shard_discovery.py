from __future__ import annotations

"""V2.9.9.12 Q100 robust crypto-shard discovery/preflight.

Observed issue addressed
------------------------
V2.9.9.11 correctly routed live crypto mutations to exchange shard 2 and correctly
funded shard 2.  A live notebook preflight then proved all nine current 15-minute
crypto markets existed on shard 2, but the redundant V2.9.9.10 shard verifier later
reported only 5/9.  The cause was preflight discovery fragility: every verifier
attempt rescanned all nine series with both `unopened` and `open` queries (18 public
requests per pass), silently discarded individual request failures, and required all
nine series to succeed in the same pass.  Repeating that full fan-out could amplify
transient public-REST failures/rate pressure and produce a false missing-market gate.

This wrapper changes only the discovery/preflight mechanics:
- current-market verification is performed per series;
- `open` is queried first because the current M0-M15 contract is normally open;
- only a missing series is retried, with bounded backoff;
- `unopened` is a fallback at a status-transition boundary and is used separately
  for optional next-market display;
- successful series are retained instead of being discarded because another series
  had a transient read failure;
- the underlying V2.9.9.10 shard verifier is rebound to this robust implementation;
- V2.9.9.11 centicent funding, shard-2 routing, Q100 strategy/risk, REC25, M12,
  guardian, and retry-until-flat recovery remain unchanged.

Importing this module performs no API calls, orders, cancels, or transfers.
"""

import inspect
import math
import time
from datetime import timedelta

from . import mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_11_crypto_shard2_centicent_funding as BASE


DEPLOY_VERSION = (
    "MM_DEEP_TAIL_JOIN_ASK_Q100_M1_M12_GUARD_REC25_"
    "V2_9_9_12_ROBUST_SHARD_DISCOVERY"
)
MODULE_NAME = (
    "quant_research.kalshi."
    "mm_deep_tail_join_ask_q100_m12_guard_rec25_live_"
    "v2_9_9_12_robust_shard_discovery"
)

Q100_ARM = "LIVE_DEEP_TAIL_Q100_M1_M12_GUARD_REC25_12H_V29912"
Q50_ARM = Q100_ARM
KILL_ARM = BASE.KILL_ARM
# Funding semantics are unchanged, so preserve the already-audited V2.9.9.11 arm.
SHARD_FUND_ARM = BASE.SHARD_FUND_ARM

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

# V2.9.9.10 owns the original shard routing/discovery implementation.
LEGACY_SHARD = BASE.BASE
# V2.9.9.8 owns the inherited private-WS/API/exact-equity Q100 preflight.
INHERITED_Q100 = BASE.BASE.BASE

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

get_shard_balances = BASE.get_shard_balances
ensure_crypto_shard_funded = BASE.ensure_crypto_shard_funded
_usd_to_centicents_ceil = BASE._usd_to_centicents_ceil
_centicents_to_usd = BASE._centicents_to_usd

DISCOVERY_ATTEMPTS_PER_SERIES = 6
DISCOVERY_RETRY_BASE_S = 0.20
DISCOVERY_CACHE_TTL_S = 8.0

_DISCOVERY_CACHE = {
    "monotonic": 0.0,
    "data": None,
}


def _market_row(series, market, *, query_status, now):
    ticker = str((market or {}).get("ticker") or "")
    close_time = C.parse_time((market or {}).get("close_time"))
    if not ticker or close_time is None:
        return None

    window_start = close_time - timedelta(seconds=900)
    elapsed_s = (now - window_start).total_seconds()

    return {
        "series": str(series),
        "ticker": ticker,
        "status": (market or {}).get("status"),
        "query_status": str(query_status),
        "exchange_index": LEGACY_SHARD._int_or_none((market or {}).get("exchange_index")),
        "window_start": window_start,
        "close_time": close_time,
        "elapsed_s": float(elapsed_s),
        "yes_bid": (market or {}).get("yes_bid_dollars", (market or {}).get("yes_bid")),
        "yes_ask": (market or {}).get("yes_ask_dollars", (market or {}).get("yes_ask")),
        "event_ticker": (market or {}).get("event_ticker"),
    }


def _query_series_status(series, query_status, *, now=None):
    now = now or C.utc_now()
    payload = C.rest_get(
        "/markets",
        {
            "series_ticker": str(series),
            "status": str(query_status),
            "limit": 1000,
        },
    )
    rows = []
    for market in (payload or {}).get("markets") or []:
        row = _market_row(series, market, query_status=query_status, now=now)
        if row is not None:
            rows.append(row)
    return rows


def _pick_current(rows):
    candidates = [
        row for row in rows
        if 0.0 <= float(row.get("elapsed_s", -1.0)) < 900.0
    ]
    if not candidates:
        return None
    # If an endpoint ever returns overlapping candidates, the most recently
    # started M0 window is the current one.
    candidates.sort(key=lambda row: row["window_start"], reverse=True)
    return candidates[0]


def _pick_next(rows):
    candidates = [
        row for row in rows
        if -900.0 <= float(row.get("elapsed_s", 1.0)) < 0.0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: row["window_start"])
    return candidates[0]


def _discover_one_current_series(
    series,
    *,
    attempts=DISCOVERY_ATTEMPTS_PER_SERIES,
    pause_s=DISCOVERY_RETRY_BASE_S,
):
    """Find one current contract without forcing every other series to succeed."""
    errors = []
    attempts = max(1, int(attempts))

    for attempt in range(1, attempts + 1):
        now = C.utc_now()

        # Current contract is normally OPEN.  Query it first so the common path
        # is one public REST request per series, not two.
        try:
            open_rows = _query_series_status(series, "open", now=now)
            current = _pick_current(open_rows)
            if current is not None:
                return current, errors
        except Exception as exc:
            errors.append({
                "attempt": int(attempt),
                "status": "open",
                "error": repr(exc),
            })

        # At an exchange status-transition boundary the new current market can
        # briefly still appear as unopened.  This is a fallback, not the normal
        # fan-out path.
        try:
            unopened_rows = _query_series_status(series, "unopened", now=now)
            current = _pick_current(unopened_rows)
            if current is not None:
                return current, errors
        except Exception as exc:
            errors.append({
                "attempt": int(attempt),
                "status": "unopened",
                "error": repr(exc),
            })

        if attempt < attempts:
            time.sleep(float(pause_s) * min(4.0, float(attempt)))

    return None, errors


def _discover_next_once(series):
    """Best-effort NEXT market for notebook/dashboard display; never a launch gate."""
    try:
        rows = _query_series_status(series, "unopened", now=C.utc_now())
        return _pick_next(rows), None
    except Exception as exc:
        return None, repr(exc)


def discover_current_crypto_markets(*, require_all=False):
    """Robust dynamic 9-series discovery with a short complete-snapshot cache."""
    now_mono = time.monotonic()
    cached = _DISCOVERY_CACHE.get("data")
    cache_age = now_mono - float(_DISCOVERY_CACHE.get("monotonic") or 0.0)

    if (
        cached
        and cache_age <= float(DISCOVERY_CACHE_TTL_S)
        and len((cached or {}).get("current") or {}) == len(CRYPTO_SERIES)
    ):
        return cached

    current = {}
    upcoming = {}
    rows = []
    errors = {}

    # Crucial difference from V2.9.9.10: each successful series is retained.
    # A transient failure for ETH cannot erase a successful BTC/BNB/DOGE read.
    for series in CRYPTO_SERIES:
        row, series_errors = _discover_one_current_series(series)
        if row is not None:
            current[series] = row
            rows.append(dict(row, slot="CURRENT"))
        if series_errors:
            errors[series] = list(series_errors)

    # NEXT is diagnostic/display-only and cannot make a valid launch fail.
    for series in CRYPTO_SERIES:
        nxt, next_error = _discover_next_once(series)
        if nxt is not None:
            upcoming[series] = nxt
            rows.append(dict(nxt, slot="NEXT"))
        if next_error:
            errors.setdefault(series, []).append({
                "status": "unopened_next",
                "error": next_error,
            })

    out = {
        "time": C.iso_utc(C.utc_now()),
        "current": current,
        "next": upcoming,
        "rows": rows,
        "errors": errors,
        "discovery_policy": "PER_SERIES_OPEN_FIRST_BOUNDED_RETRY_RETAIN_SUCCESSES",
    }

    if len(current) == len(CRYPTO_SERIES):
        _DISCOVERY_CACHE["monotonic"] = time.monotonic()
        _DISCOVERY_CACHE["data"] = out

    if require_all and len(current) != len(CRYPTO_SERIES):
        missing = sorted(set(CRYPTO_SERIES) - set(current))
        raise RuntimeError(
            "Robust dynamic discovery failed after per-series retries: "
            f"current={len(current)}/{len(CRYPTO_SERIES)} missing={missing} "
            f"errors={{{', '.join(f'{s!r}: {errors.get(s)!r}' for s in missing)}}}"
        )

    return out


def _verify_market_shard2_robust(*, retries=None, pause_s=None):
    """Drop-in replacement for V2.9.9.10's fragile all-series same-pass gate."""
    # Keep the historical signature arguments accepted for compatibility.  The
    # robust policy retries the missing series internally instead of repeating a
    # full 18-request fan-out.
    discovery = discover_current_crypto_markets(require_all=True)
    current = discovery["current"]

    bad = {
        series: row.get("exchange_index")
        for series, row in current.items()
        if row.get("exchange_index") != LIVE_EXCHANGE_INDEX
    }
    if bad:
        raise RuntimeError(
            "Crypto shard verification failed: "
            f"current={len(current)}/{len(CRYPTO_SERIES)} wrong_shard={bad}"
        )
    return discovery


def _install_patch():
    """Install V2.9.9.11, then replace only the fragile shard verifier."""
    BASE._install_patch()

    # V2.9.9.10 crypto_shard_preflight resolves this name from its own module
    # globals at call time, so rebinding this one helper fixes both notebook and
    # launch-time shard verification without changing its account/risk checks.
    LEGACY_SHARD._verify_market_shard2 = _verify_market_shard2_robust

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
    """Offline structural/regression audit. No API/orders/cancels/transfers."""
    # Run V2.9.9.11's own audit before overlaying this verifier so its historical
    # source checks remain an independent regression baseline.
    base = BASE.static_self_check(show=False)
    _install_patch()

    discover_src = inspect.getsource(_discover_one_current_series)
    verify_src = inspect.getsource(_verify_market_shard2_robust)

    checks = {
        "base_v29911_ok": base.get("ok") is True,
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
        "centicent_funding_preserved": _usd_to_centicents_ceil(125.0) == 1_250_000,
        "robust_verifier_bound": LEGACY_SHARD._verify_market_shard2 is _verify_market_shard2_robust,
        "per_series_retry_present": "for attempt in range" in discover_src,
        "open_first_present": '_query_series_status(series, "open"' in discover_src,
        "unopened_transition_fallback_present": '_query_series_status(series, "unopened"' in discover_src,
        "successful_series_retained": "current[series] = row" in inspect.getsource(discover_current_crypto_markets),
        "full_fanout_retry_removed": "discover_current_crypto_markets(require_all=True)" in verify_src,
        "transport_shard2_preserved": Q1.LiveClient.request is LEGACY_SHARD._liveclient_request_shard2,
        "semantic_market_not_found_binding_preserved": (
            LIVE.Rec25PassiveExitM12Engine._drain_create_futures is BASE.BASE.BASE.SEMANTIC_DRAIN
        ),
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": LIVE.PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": LIVE.PASSIVE_EXIT_TIF == "good_till_canceled",
        "orders_sent": False,
        "transfers_sent": False,
    }

    ok = all(
        value is True
        for key, value in checks.items()
        if key not in {"orders_sent", "transfers_sent"}
    )

    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "quantity": Q100_Q,
        "runtime_hours": Q100_HOURS,
        "live_exchange_index": LIVE_EXCHANGE_INDEX,
        "discovery_policy": {
            "current_query": "OPEN_FIRST",
            "fallback": "UNOPENED_AT_STATUS_TRANSITION",
            "retry_scope": "ONLY_THE_MISSING_SERIES",
            "success_retention": True,
            "complete_snapshot_cache_s": DISCOVERY_CACHE_TTL_S,
        },
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 188)
        print("V2.9.9.12 Q100 ROBUST SHARD-DISCOVERY STATIC CHECK — NO API / NO ORDERS / NO TRANSFERS")
        print("=" * 188)
        for key, value in out.items():
            print(f"{key:116s}: {value}")

    if not ok:
        raise RuntimeError(f"V2.9.9.12 static self-check failed: {out}")
    return out


def crypto_shard_preflight(*, client=None, show=True):
    """Read-only V2.9.9.10 shard/account gate using the robust verifier."""
    _install_patch()
    return LEGACY_SHARD.crypto_shard_preflight(client=client, show=show)


def q100_preflight(*, show=True):
    """Read-only Q100 preflight with robust current-market shard verification."""
    _install_patch()
    static_self_check(show=show)

    shard = crypto_shard_preflight(show=show)

    # Run the inherited private-WS/API/exact-equity preflight directly, bypassing
    # the redundant V2.9.9.10/V2.9.9.11 shard-discovery wrappers.
    report = INHERITED_Q100.q100_preflight(show=show)
    _install_patch()

    out = dict(report or {})
    out["crypto_shard"] = shard
    out["deploy_version"] = DEPLOY_VERSION
    out["ok"] = bool((report or {}).get("ok", True) and shard.get("ok") is True)
    return out


def start_q100_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q100 / 12h; launch still requires robust shard/account preflight."""
    _install_patch()
    if arm_phrase != Q100_ARM:
        raise RuntimeError(f"Wrong Q100 arm phrase; expected {Q100_ARM!r}")

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
