from __future__ import annotations

"""V2.9.9.10 Q100 crypto-shard migration fix.

Kalshi moved newly-created crypto events to exchange shard 2 on 2026-08-24.
The recorder/discovery path was already correct and dynamically rediscovers the
nine 15-minute crypto markets every window.  The execution transport, however,
was inherited from an older pre-sharding stack and still routed mutations to
exchange_index=0.  That produced `market_not_found` for valid shard-2 markets.

V2.9.9.9 changed writes to shard 2 but exposed the second required migration step:
the account had no collateral provisioned on shard 2, so order-group creation
returned `user_not_found`.

This wrapper fixes the complete operational migration while preserving strategy
economics and risk:
- dynamic current-market discovery remains unchanged;
- live crypto mutations route to shard 2;
- shard-sensitive legacy reads with an old explicit shard-0 filter are corrected;
- all-account reads remain aggregate across shards;
- order-group create/trigger/delete stay on the same shard as the orders;
- an explicit, separately armed helper can move only the collateral shortfall
  from primary shard 0 to primary shard 2;
- launch refuses to arm unless all nine current crypto markets are on shard 2
  and shard 2 has at least the existing $125 deployment minimum.

No API calls occur merely by importing this module.
"""

import inspect
import math
import time
from datetime import timedelta
from decimal import Decimal, ROUND_CEILING

from . import recorder_core as C
from . import mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_8_semantic_create_binding as BASE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q100_M1_M12_GUARD_REC25_V2_9_9_10_CRYPTO_SHARD2_FUNDED"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_10_crypto_shard2_funded"

Q100_ARM = "LIVE_DEEP_TAIL_Q100_M1_M12_GUARD_REC25_12H_V29910"
Q50_ARM = Q100_ARM
KILL_ARM = BASE.KILL_ARM
SHARD_FUND_ARM = "FUND_KALSHI_CRYPTO_SHARD2_TO_125"

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
Q1 = B.Q1

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

LIVE_EXCHANGE_INDEX = 2
SOURCE_EXCHANGE_INDEX = 0
SHARD2_MIN_COLLATERAL_USD = float(Q100_MIN_EQUITY_USD)

CRYPTO_SERIES = (
    "KXBTC15M",
    "KXBNB15M",
    "KXDOGE15M",
    "KXETH15M",
    "KXHYPE15M",
    "KXNEAR15M",
    "KXSOL15M",
    "KXXRP15M",
    "KXZEC15M",
)


def _is_crypto_ticker(ticker):
    t = str(ticker or "")
    return any(t.startswith(series + "-") for series in CRYPTO_SERIES)


def _int_or_none(x):
    try:
        return int(x)
    except Exception:
        return None


def _route_request_args(method, path, *, params=None, payload=None):
    """Pure transport router used by the LiveClient shim and static tests."""
    method = str(method).upper()
    path = str(path)

    p = None if params is None else dict(params)
    body = None if payload is None else dict(payload)

    # ---------- ORDER MUTATIONS ----------
    if path == "/portfolio/events/orders" and method == "POST":
        body = dict(body or {})
        if _is_crypto_ticker(body.get("ticker")):
            body["exchange_index"] = int(LIVE_EXCHANGE_INDEX)

    elif path == "/portfolio/events/orders/batched" and method in {"POST", "DELETE"}:
        body = dict(body or {})
        routed = []
        for row in body.get("orders") or []:
            z = dict(row)
            # The deployment is crypto-only. Batch cancel rows may have only an
            # order_id, so route every row in this strategy batch to shard 2.
            z["exchange_index"] = int(LIVE_EXCHANGE_INDEX)
            routed.append(z)
        body["orders"] = routed

    elif path.startswith("/portfolio/events/orders/") and method in {"DELETE", "POST", "PUT"}:
        p = dict(p or {})
        p["subaccount"] = int(p.get("subaccount", 0) or 0)
        p["exchange_index"] = int(LIVE_EXCHANGE_INDEX)

    # ---------- ORDER GROUP MUTATIONS ----------
    elif path == "/portfolio/order_groups/create" and method == "POST":
        body = dict(body or {})
        body["subaccount"] = int(body.get("subaccount", 0) or 0)
        body["exchange_index"] = int(LIVE_EXCHANGE_INDEX)

    elif path.startswith("/portfolio/order_groups/") and method in {"PUT", "DELETE", "POST"}:
        p = dict(p or {})
        p["subaccount"] = int(p.get("subaccount", 0) or 0)
        p["exchange_index"] = int(LIVE_EXCHANGE_INDEX)

    # ---------- LEGACY SHARD-0 READ FILTERS ----------
    # Current list endpoints return all shards when exchange_index is omitted.
    # For ticker/order scoped historical helpers, route to shard 2 instead.
    if method == "GET" and path in {
        "/portfolio/orders",
        "/portfolio/positions",
        "/portfolio/fills",
    }:
        p = dict(p or {})
        old_idx = _int_or_none(p.get("exchange_index"))

        if old_idx == SOURCE_EXCHANGE_INDEX:
            if p.get("ticker") or p.get("order_id"):
                p["exchange_index"] = int(LIVE_EXCHANGE_INDEX)
            else:
                p.pop("exchange_index", None)

    return p, body


# Preserve the pristine request implementation across notebook reloads.
if not hasattr(Q1.LiveClient, "_v29910_original_request"):
    Q1.LiveClient._v29910_original_request = Q1.LiveClient.request

_ORIGINAL_LIVECLIENT_REQUEST = Q1.LiveClient._v29910_original_request


def _liveclient_request_shard2(self, method, path, *, params=None, payload=None, timeout=8.0):
    params, payload = _route_request_args(
        method,
        path,
        params=params,
        payload=payload,
    )
    return _ORIGINAL_LIVECLIENT_REQUEST(
        self,
        method,
        path,
        params=params,
        payload=payload,
        timeout=timeout,
    )


# Preserve the true pre-sharding payload builder. If V2.9.9.9 was imported in a
# long-lived notebook, it already stored that pristine function for us.
if hasattr(B, "_v2999_original_payload"):
    _ORIGINAL_PAYLOAD = B._v2999_original_payload
elif not hasattr(B, "_v29910_original_payload"):
    B._v29910_original_payload = B._payload
    _ORIGINAL_PAYLOAD = B._v29910_original_payload
else:
    _ORIGINAL_PAYLOAD = B._v29910_original_payload


def _payload_shard2(*, ticker, side, qty, price, cid, post_only, reduce_only, tif, group_id=None):
    p = _ORIGINAL_PAYLOAD(
        ticker=ticker,
        side=side,
        qty=qty,
        price=price,
        cid=cid,
        post_only=post_only,
        reduce_only=reduce_only,
        tif=tif,
        group_id=group_id,
    )
    if _is_crypto_ticker(ticker) or str(ticker) == "TEST":
        p["exchange_index"] = int(LIVE_EXCHANGE_INDEX)
    return p


def _create_group_shard2(client):
    body, timing = client.post(
        "/portfolio/order_groups/create",
        {
            "subaccount": 0,
            "contracts_limit_fp": B.GROUP_LIMIT_FP,
            "exchange_index": int(LIVE_EXCHANGE_INDEX),
        },
    )
    gid = str((body or {}).get("order_group_id") or "")
    if not gid:
        raise RuntimeError(f"Order group response missing id: {body}")
    returned_idx = _int_or_none((body or {}).get("exchange_index"))
    if returned_idx is not None and returned_idx != LIVE_EXCHANGE_INDEX:
        raise RuntimeError(
            f"Order group created on wrong shard: expected={LIVE_EXCHANGE_INDEX} body={body}"
        )
    return gid, body, timing


def _trigger_group_shard2(client, gid):
    if not gid:
        return {"ok": True, "note": "no group"}
    try:
        body, timing = client.request(
            "PUT",
            f"/portfolio/order_groups/{gid}/trigger",
            params={"subaccount": 0, "exchange_index": int(LIVE_EXCHANGE_INDEX)},
            payload={},
        )
        return {"ok": True, "body": body, "timing": timing}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _delete_group_shard2(client, gid):
    try:
        body, timing = client.delete(
            f"/portfolio/order_groups/{gid}",
            params={"subaccount": 0, "exchange_index": int(LIVE_EXCHANGE_INDEX)},
        )
        return {"ok": True, "body": body, "timing": timing}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _cancel_shard2(client, oid):
    try:
        body, timing = client.delete(
            f"/portfolio/events/orders/{oid}",
            params={"subaccount": 0, "exchange_index": int(LIVE_EXCHANGE_INDEX)},
        )
        return {"ok": True, "body": body, "timing": timing}
    except Exception as exc:
        try:
            row, timing = B._get_order(client, oid)
            rem = B._f((row or {}).get("remaining_count_fp"), 0.0)
            if rem <= B.EPS or str((row or {}).get("status") or "").lower() != "resting":
                return {"ok": True, "already_done": True, "order": row, "timing": timing}
        except Exception:
            pass
        return {"ok": False, "error": repr(exc)}


def _balance_breakdown(body):
    out = {}
    for row in (body or {}).get("balance_breakdown") or []:
        idx = _int_or_none(row.get("exchange_index"))
        if idx is None:
            continue
        try:
            usd = float(row.get("balance"))
        except Exception:
            usd = 0.0
        if math.isfinite(usd):
            out[int(idx)] = float(usd)
    return out


def get_shard_balances(client=None):
    """Read-only primary-account shard balances."""
    _install_patch()
    client = client or Q1.LiveClient()
    body, timing = client.get("/portfolio/balance", params={"subaccount": 0})
    return {
        "raw": body,
        "timing": timing,
        "breakdown_usd": _balance_breakdown(body),
        "total_balance_dollars": body.get("balance_dollars"),
    }


def discover_current_crypto_markets(*, require_all=False):
    """Mirror the proven recorder: scan open+unopened and derive M0 from close-900s."""
    now = C.utc_now()
    current = {}
    upcoming = {}
    rows = []

    for series in CRYPTO_SERIES:
        seen = {}
        for query_status in ("unopened", "open"):
            try:
                markets = C.rest_get(
                    "/markets",
                    {
                        "series_ticker": series,
                        "status": query_status,
                        "limit": 1000,
                    },
                ).get("markets") or []
            except Exception:
                continue

            for m in markets:
                ticker = str(m.get("ticker") or "")
                close_time = C.parse_time(m.get("close_time"))
                if not ticker or close_time is None:
                    continue

                window_start = close_time - timedelta(seconds=900)
                elapsed_s = (now - window_start).total_seconds()

                row = {
                    "series": series,
                    "ticker": ticker,
                    "status": m.get("status"),
                    "query_status": query_status,
                    "exchange_index": _int_or_none(m.get("exchange_index")),
                    "window_start": window_start,
                    "close_time": close_time,
                    "elapsed_s": float(elapsed_s),
                    "yes_bid": m.get("yes_bid_dollars", m.get("yes_bid")),
                    "yes_ask": m.get("yes_ask_dollars", m.get("yes_ask")),
                    "event_ticker": m.get("event_ticker"),
                }
                seen[ticker] = row

        cur = [r for r in seen.values() if 0.0 <= r["elapsed_s"] < 900.0]
        cur.sort(key=lambda r: (abs(r["elapsed_s"] - 450.0), r["close_time"]))
        if cur:
            current[series] = cur[0]
            rows.append(dict(cur[0], slot="CURRENT"))

        nxt = [r for r in seen.values() if -900.0 <= r["elapsed_s"] < 0.0]
        nxt.sort(key=lambda r: r["window_start"])
        if nxt:
            upcoming[series] = nxt[0]
            rows.append(dict(nxt[0], slot="NEXT"))

    if require_all and len(current) != len(CRYPTO_SERIES):
        missing = sorted(set(CRYPTO_SERIES) - set(current))
        raise RuntimeError(
            f"Dynamic discovery found {len(current)}/{len(CRYPTO_SERIES)} current markets; "
            f"missing={missing}"
        )

    return {
        "time": C.iso_utc(now),
        "current": current,
        "next": upcoming,
        "rows": rows,
    }


def _verify_market_shard2(*, retries=8, pause_s=1.0):
    last = None
    for _ in range(max(1, int(retries))):
        last = discover_current_crypto_markets(require_all=False)
        current = last["current"]
        if len(current) == len(CRYPTO_SERIES):
            bad = {
                s: r.get("exchange_index")
                for s, r in current.items()
                if r.get("exchange_index") != LIVE_EXCHANGE_INDEX
            }
            if not bad:
                return last
        time.sleep(float(pause_s))

    current = (last or {}).get("current") or {}
    bad = {
        s: r.get("exchange_index")
        for s, r in current.items()
        if r.get("exchange_index") != LIVE_EXCHANGE_INDEX
    }
    missing = sorted(set(CRYPTO_SERIES) - set(current))
    raise RuntimeError(
        "Crypto shard verification failed: "
        f"current={len(current)}/{len(CRYPTO_SERIES)} missing={missing} wrong_shard={bad}"
    )


def ensure_crypto_shard_funded(
    *,
    arm_phrase=None,
    target_usd=SHARD2_MIN_COLLATERAL_USD,
    client=None,
    wait_s=30.0,
):
    """REAL INTERNAL BALANCE TRANSFER; sends no market order.

    Moves only the shortfall needed to bring primary shard 2 to target_usd,
    sourcing funds from primary shard 0. The total Kalshi account balance does
    not change. Requires the exact SHARD_FUND_ARM phrase.
    """
    _install_patch()

    if arm_phrase != SHARD_FUND_ARM:
        raise RuntimeError(
            "Shard funding not armed. "
            f"Pass arm_phrase={SHARD_FUND_ARM!r}."
        )

    target = Decimal(str(target_usd)).quantize(Decimal("0.01"))
    if target != Decimal(str(SHARD2_MIN_COLLATERAL_USD)).quantize(Decimal("0.01")):
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

    shortfall_cents = int(
        ((target - shard2) * Decimal("100")).to_integral_value(rounding=ROUND_CEILING)
    )
    shortfall_usd = Decimal(shortfall_cents) / Decimal("100")

    if shard0 + Decimal("0.0001") < shortfall_usd:
        raise RuntimeError(
            f"Shard 0 has ${shard0:.4f}, but ${shortfall_usd:.2f} is needed "
            f"to bring shard 2 to ${target:.2f}."
        )

    body, timing = client.post(
        "/portfolio/intra_exchange_instance_transfer",
        {
            "amount": int(shortfall_cents),
            "source_exchange_shard": int(SOURCE_EXCHANGE_INDEX),
            "destination_exchange_shard": int(LIVE_EXCHANGE_INDEX),
            "source_subaccount": 0,
            "destination_subaccount": 0,
        },
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
                "transfer_id": (body or {}).get("transfer_id"),
                "transfer_amount_usd": float(shortfall_usd),
                "transfer_body": body,
                "transfer_timing": timing,
                "target_usd": float(target),
                "before": before,
                "after": after,
            }

    raise RuntimeError(
        "Shard transfer request was accepted but shard 2 did not reach the target "
        f"${target:.2f} within {wait_s:.1f}s. "
        f"transfer_response={body!r} last_balance={after!r}"
    )


def crypto_shard_preflight(*, client=None, show=True):
    """Read-only shard/discovery/account gate. Sends no orders or transfers."""
    _install_patch()
    client = client or Q1.LiveClient()

    discovery = _verify_market_shard2()
    balances = get_shard_balances(client)
    b = balances["breakdown_usd"]
    shard2_usd = float(b.get(LIVE_EXCHANGE_INDEX, 0.0))

    if shard2_usd + 1e-9 < SHARD2_MIN_COLLATERAL_USD:
        raise RuntimeError(
            f"Shard 2 collateral ${shard2_usd:.4f} < required "
            f"${SHARD2_MIN_COLLATERAL_USD:.2f}. "
            "Run ensure_crypto_shard_funded() with the explicit funding arm."
        )

    positions_body, _ = client.get(
        "/portfolio/positions",
        params={"count_filter": "position", "limit": 1000, "subaccount": 0},
    )
    positions = positions_body.get("market_positions") or positions_body.get("positions") or []
    nonzero = [
        r for r in positions
        if abs(B._f(r.get("position_fp", r.get("position")), 0.0)) > B.EPS
    ]

    orders_body, _ = client.get(
        "/portfolio/orders",
        params={"status": "resting", "limit": 1000, "subaccount": 0},
    )
    resting = orders_body.get("orders") or []

    if nonzero:
        raise RuntimeError(f"Account is not flat across shards: {nonzero}")
    if resting:
        raise RuntimeError(f"Account has resting orders across shards: {resting}")

    out = {
        "ok": True,
        "exchange_index": LIVE_EXCHANGE_INDEX,
        "current_market_count": len(discovery["current"]),
        "current_tickers": {
            s: r["ticker"] for s, r in discovery["current"].items()
        },
        "next_tickers": {
            s: r["ticker"] for s, r in discovery["next"].items()
        },
        "balances": balances,
        "shard2_balance_usd": shard2_usd,
        "required_shard2_usd": SHARD2_MIN_COLLATERAL_USD,
        "nonzero_positions": nonzero,
        "resting_orders": resting,
    }

    if show:
        print("=" * 176)
        print("V2.9.9.10 CRYPTO SHARD PREFLIGHT — READ ONLY")
        print("=" * 176)
        print(f"current markets:      {len(discovery['current'])}/{len(CRYPTO_SERIES)}")
        print(f"market shard:         {LIVE_EXCHANGE_INDEX}")
        print(f"shard 2 collateral:  ${shard2_usd:.4f}")
        print(f"required collateral: ${SHARD2_MIN_COLLATERAL_USD:.2f}")
        print(f"positions:            {len(nonzero)}")
        print(f"resting orders:       {len(resting)}")
        print("ORDERS SENT:          NO")
        print("TRANSFERS SENT:       NO")

    return out


def _install_patch():
    """Install V2.9.9.8, then bind complete shard-2 transport/risk routing."""
    BASE._install_patch()

    # Central safety net: catches historical shard-0 request fields in child,
    # parent fail-closed recovery, and Q1/V11 reconciliation helpers.
    Q1.LiveClient.request = _liveclient_request_shard2

    # Make logged/audited payloads and group helpers correct before transport too.
    B._payload = _payload_shard2
    B._create_group = _create_group_shard2
    B._trigger_group = _trigger_group_shard2
    B._delete_group = _delete_group_shard2
    B._cancel = _cancel_shard2

    # Deployment identity / Q100 invariants.
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
    """Offline structural/regression audit. No API calls, orders, or transfers."""
    base = BASE.static_self_check(show=False)
    _install_patch()

    sample = _payload_shard2(
        ticker="TEST",
        side="bid",
        qty=100.0,
        price=0.05,
        cid="v29910-static",
        post_only=True,
        reduce_only=False,
        tif="good_till_canceled",
        group_id="gid-test",
    )

    _, create_body = _route_request_args(
        "POST",
        "/portfolio/events/orders",
        payload={
            "ticker": "KXBTC15M-TEST",
            "exchange_index": 0,
        },
    )
    cancel_params, _ = _route_request_args(
        "DELETE",
        "/portfolio/events/orders/order-test",
        params={"subaccount": 0, "exchange_index": 0},
    )
    trigger_params, _ = _route_request_args(
        "PUT",
        "/portfolio/order_groups/group-test/trigger",
        params=None,
        payload={},
    )
    ticker_read, _ = _route_request_args(
        "GET",
        "/portfolio/orders",
        params={"ticker": "KXBTC15M-TEST", "exchange_index": 0},
    )
    aggregate_read, _ = _route_request_args(
        "GET",
        "/portfolio/orders",
        params={"status": "resting", "exchange_index": 0},
    )

    request_src = inspect.getsource(_route_request_args)
    fund_src = inspect.getsource(ensure_crypto_shard_funded)

    checks = {
        "base_v2998_ok": base.get("ok") is True,
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
        "retry_recovery_preserved": getattr(P._recover_generation_fail_closed, "__name__", "") == "_recover_generation_fail_closed_retry",
        "exchange_index_exact_2": LIVE_EXCHANGE_INDEX == 2,
        "payload_builder_shard2": int(sample.get("exchange_index", -1)) == 2,
        "transport_shim_installed": Q1.LiveClient.request is _liveclient_request_shard2,
        "create_transport_shard2": int(create_body.get("exchange_index", -1)) == 2,
        "cancel_transport_shard2": int(cancel_params.get("exchange_index", -1)) == 2,
        "group_trigger_transport_shard2": int(trigger_params.get("exchange_index", -1)) == 2,
        "ticker_read_old_filter_corrected": int(ticker_read.get("exchange_index", -1)) == 2,
        "aggregate_read_old_filter_removed": "exchange_index" not in aggregate_read,
        "group_create_explicit_shard2": '"exchange_index": int(LIVE_EXCHANGE_INDEX)' in inspect.getsource(_create_group_shard2),
        "group_trigger_explicit_shard2": '"exchange_index": int(LIVE_EXCHANGE_INDEX)' in inspect.getsource(_trigger_group_shard2),
        "group_delete_explicit_shard2": '"exchange_index": int(LIVE_EXCHANGE_INDEX)' in inspect.getsource(_delete_group_shard2),
        "dynamic_discovery_open_and_unopened": '("unopened", "open")' in inspect.getsource(discover_current_crypto_markets),
        "dynamic_discovery_close_minus_900": "timedelta(seconds=900)" in inspect.getsource(discover_current_crypto_markets),
        "funding_requires_exact_arm": "arm_phrase != SHARD_FUND_ARM" in fund_src,
        "funding_only_shortfall": "shortfall_cents" in fund_src,
        "funding_target_matches_min_equity": SHARD2_MIN_COLLATERAL_USD == Q100_MIN_EQUITY_USD == 125.0,
        "transfer_endpoint_exact": '"/portfolio/intra_exchange_instance_transfer"' in fund_src,
        "semantic_market_not_found_binding_preserved": LIVE.Rec25PassiveExitM12Engine._drain_create_futures is BASE.SEMANTIC_DRAIN,
        "request_router_has_group_and_order_paths": "/portfolio/order_groups/" in request_src and "/portfolio/events/orders" in request_src,
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
        "crypto_series": CRYPTO_SERIES,
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 184)
        print("V2.9.9.10 Q100 CRYPTO-SHARD2 STATIC CHECK — NO API / NO ORDERS / NO TRANSFERS")
        print("=" * 184)
        for k, v in out.items():
            print(f"{k:112s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.9.10 static self-check failed: {out}")

    return out


def q100_preflight(*, show=True):
    """Read-only Q100 preflight plus shard/discovery gate."""
    _install_patch()
    static_self_check(show=show)
    shard = crypto_shard_preflight(show=show)

    # Preserve inherited private-WS/API/equity preflight, then re-install this
    # child wrapper because inherited preflight calls lower-layer patch functions.
    report = BASE.q100_preflight(show=show)
    _install_patch()

    out = dict(report or {})
    out["crypto_shard"] = shard
    out["deploy_version"] = DEPLOY_VERSION
    out["ok"] = bool((report or {}).get("ok", True) and shard.get("ok") is True)
    return out


def start_q100_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q100 / 12h. Refuses launch unless shard migration is ready."""
    _install_patch()

    if arm_phrase != Q100_ARM:
        raise RuntimeError(f"Wrong Q100 arm phrase; expected {Q100_ARM!r}")

    # Independent launch-time guard even if notebook Cell 2 was skipped.
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
    "SHARD2_MIN_COLLATERAL_USD",
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
