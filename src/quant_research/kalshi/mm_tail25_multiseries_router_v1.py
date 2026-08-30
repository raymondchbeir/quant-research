from __future__ import annotations

"""Dynamic execution routing for the 12-series Tail25/JOIN_BBO deployment.

This module is additive and sends no orders merely by importing it.

It generalizes the audited single-shard crypto transport to a mixed universe:
nine 15-minute crypto series plus 15-minute gold, silver and WTI.  Current
markets are discovered per series and their exchange_index values are treated as
runtime data, never hard-coded assumptions.

The module also makes the inherited single order-group interface multi-shard
compatible by representing the generation group id as {exchange_index: group_id}.
Existing risk/shutdown call sites can keep calling B._trigger_group/_delete_group;
the patched helpers fan out to every shard-specific group.

No account mutation occurs until the normal explicitly armed live runner calls
the inherited group/order helpers.
"""

import math
import threading
import time
from datetime import timedelta

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_event_time_m0_m5_recorder_v5_auth as V5A
from . import mm_live_q1_queue_probe_v1 as Q1


ROUTER_VERSION = "TAIL25_MULTI12_DYNAMIC_SHARD_ROUTER_V1"

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
COMMODITY_SERIES = (
    "KXGOLD15M",
    "KXSILVER15M",
    "KXWTI15M",
)
SERIES = CRYPTO_SERIES + COMMODITY_SERIES

DISCOVERY_ATTEMPTS_PER_SERIES = 6
DISCOVERY_RETRY_BASE_S = 0.20
DISCOVERY_CACHE_TTL_S = 8.0

_LOCK = threading.RLock()
_DISCOVERY_CACHE = {"monotonic": 0.0, "data": None}
_TICKER_TO_SHARD: dict[str, int] = {}
_ORDER_TO_SHARD: dict[str, int] = {}

# V2.9.9.10 preserves the pristine request implementation under this attribute.
# Fall back to the currently installed method for import-time compatibility.
_ORIGINAL_REQUEST = getattr(
    Q1.LiveClient,
    "_v29910_original_request",
    Q1.LiveClient.request,
)


def _finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _int_or_none(x):
    try:
        return int(x)
    except Exception:
        return None


def _series_for_ticker(ticker):
    t = str(ticker or "")
    for series in SERIES:
        if t == series or t.startswith(series + "-"):
            return series
    return None


def _market_row(series, market, *, query_status, now):
    market = market or {}
    ticker = str(market.get("ticker") or "")
    close_time = C.parse_time(market.get("close_time"))
    if not ticker or close_time is None:
        return None
    window_start = close_time - timedelta(seconds=900)
    elapsed_s = (now - window_start).total_seconds()
    return {
        "series": str(series),
        "ticker": ticker,
        "status": market.get("status"),
        "query_status": str(query_status),
        "exchange_index": _int_or_none(market.get("exchange_index")),
        "window_start": window_start,
        "close_time": close_time,
        "elapsed_s": float(elapsed_s),
        "yes_bid": market.get("yes_bid_dollars", market.get("yes_bid")),
        "yes_ask": market.get("yes_ask_dollars", market.get("yes_ask")),
        "event_ticker": market.get("event_ticker"),
    }


def _query_series_status(series, query_status, *, now=None):
    now = now or C.utc_now()
    body = V5A._signed_get_with_retry(
        "/markets",
        {
            "series_ticker": str(series),
            "status": str(query_status),
            "limit": 1000,
        },
    )
    rows = []
    for market in (body or {}).get("markets") or []:
        row = _market_row(series, market, query_status=query_status, now=now)
        if row is not None:
            rows.append(row)
    return rows


def _pick_current(rows):
    candidates = [
        row
        for row in rows
        if 0.0 <= float(row.get("elapsed_s", -1.0)) < 900.0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: row["window_start"], reverse=True)
    return candidates[0]


def _pick_next(rows):
    candidates = [
        row
        for row in rows
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
    errors = []
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        now = C.utc_now()
        for status in ("open", "unopened"):
            try:
                current = _pick_current(
                    _query_series_status(series, status, now=now)
                )
                if current is not None:
                    return current, errors
            except Exception as exc:
                errors.append(
                    {
                        "attempt": int(attempt),
                        "status": status,
                        "error": repr(exc),
                    }
                )
        if attempt < attempts:
            time.sleep(float(pause_s) * min(4.0, float(attempt)))
    return None, errors


def discover_current_markets(*, require_all=False, refresh=False):
    """Authenticated per-series discovery. Read-only."""
    now_mono = time.monotonic()
    with _LOCK:
        cached = _DISCOVERY_CACHE.get("data")
        cache_age = now_mono - float(_DISCOVERY_CACHE.get("monotonic") or 0.0)
        if (
            not refresh
            and cached
            and cache_age <= DISCOVERY_CACHE_TTL_S
            and len((cached or {}).get("current") or {}) == len(SERIES)
        ):
            return cached

    current = {}
    upcoming = {}
    rows = []
    errors = {}

    for series in SERIES:
        row, series_errors = _discover_one_current_series(series)
        if row is not None:
            current[series] = row
            rows.append(dict(row, slot="CURRENT"))
        if series_errors:
            errors[series] = list(series_errors)

    # NEXT is diagnostic only.
    for series in SERIES:
        try:
            nxt = _pick_next(
                _query_series_status(series, "unopened", now=C.utc_now())
            )
        except Exception as exc:
            errors.setdefault(series, []).append(
                {"status": "unopened_next", "error": repr(exc)}
            )
            nxt = None
        if nxt is not None:
            upcoming[series] = nxt
            rows.append(dict(nxt, slot="NEXT"))

    missing_shard = {
        series: row.get("exchange_index")
        for series, row in current.items()
        if _int_or_none(row.get("exchange_index")) is None
    }

    out = {
        "time": C.iso_utc(C.utc_now()),
        "current": current,
        "next": upcoming,
        "rows": rows,
        "errors": errors,
        "series": list(SERIES),
        "missing_exchange_index": missing_shard,
        "discovery_policy": (
            "AUTH_PER_SERIES_OPEN_FIRST_BOUNDED_RETRY_RETAIN_SUCCESSES"
        ),
    }

    if len(current) == len(SERIES) and not missing_shard:
        mapping = {
            str(row["ticker"]): int(row["exchange_index"])
            for row in current.values()
        }
        with _LOCK:
            _TICKER_TO_SHARD.update(mapping)
            _DISCOVERY_CACHE["monotonic"] = time.monotonic()
            _DISCOVERY_CACHE["data"] = out

    if require_all:
        missing = sorted(set(SERIES) - set(current))
        if missing or missing_shard:
            raise RuntimeError(
                "12-series discovery/routing failed: "
                f"current={len(current)}/{len(SERIES)} "
                f"missing={missing} missing_exchange_index={missing_shard} "
                f"errors={errors}"
            )
    return out


def routing_snapshot(*, require_all=True, refresh=False):
    d = discover_current_markets(
        require_all=require_all,
        refresh=refresh,
    )
    current = d.get("current") or {}
    by_series = {
        series: {
            "ticker": row.get("ticker"),
            "exchange_index": _int_or_none(row.get("exchange_index")),
            "elapsed_s": row.get("elapsed_s"),
            "status": row.get("status"),
        }
        for series, row in current.items()
    }
    shards = sorted(
        {
            int(row["exchange_index"])
            for row in current.values()
            if _int_or_none(row.get("exchange_index")) is not None
        }
    )
    return {
        "time": d.get("time"),
        "series": list(SERIES),
        "by_series": by_series,
        "shards": shards,
        "current_count": len(current),
        "ok": len(current) == len(SERIES) and all(
            row.get("exchange_index") is not None
            for row in by_series.values()
        ),
        "discovery": d,
    }


def register_order_shard(order_id, exchange_index):
    oid = str(order_id or "")
    idx = _int_or_none(exchange_index)
    if not oid or idx is None:
        return False
    with _LOCK:
        _ORDER_TO_SHARD[oid] = int(idx)
    return True


def shard_for_ticker(ticker, *, refresh_if_missing=True):
    ticker = str(ticker or "")
    with _LOCK:
        idx = _TICKER_TO_SHARD.get(ticker)
    if idx is not None:
        return int(idx)

    series = _series_for_ticker(ticker)
    if series and refresh_if_missing:
        snap = routing_snapshot(require_all=True, refresh=True)
        row = (snap.get("by_series") or {}).get(series) or {}
        if str(row.get("ticker") or "") == ticker:
            idx = _int_or_none(row.get("exchange_index"))
            if idx is not None:
                with _LOCK:
                    _TICKER_TO_SHARD[ticker] = int(idx)
                return int(idx)
    raise RuntimeError(f"No authoritative exchange_index for ticker {ticker!r}")


def shard_for_order(order_id):
    with _LOCK:
        idx = _ORDER_TO_SHARD.get(str(order_id or ""))
    return None if idx is None else int(idx)


def group_ids(gid):
    if isinstance(gid, dict):
        return {
            str(v)
            for v in gid.values()
            if v not in (None, "")
        }
    return {str(gid)} if gid not in (None, "") else set()


def group_for_ticker(gid, ticker):
    if not isinstance(gid, dict):
        return gid
    idx = shard_for_ticker(ticker)
    for key in (idx, str(idx)):
        if key in gid:
            return gid[key]
    raise RuntimeError(
        f"No order group for ticker={ticker} exchange_index={idx}; groups={gid}"
    )


def _payload_multishard(
    *,
    ticker,
    side,
    qty,
    price,
    cid,
    post_only,
    reduce_only,
    tif,
    group_id=None,
):
    qty = float(qty)
    price = float(price)
    if qty <= B.EPS or not (0.0 < price < 1.0):
        raise RuntimeError(f"Invalid order qty/price: {qty}, {price}")
    idx = shard_for_ticker(ticker)
    p = {
        "ticker": str(ticker),
        "client_order_id": str(cid),
        "side": str(side).lower(),
        "count": f"{qty:.2f}",
        "price": f"{price:.4f}",
        "time_in_force": str(tif),
        "self_trade_prevention_type": "taker_at_cross",
        "post_only": bool(post_only),
        "cancel_order_on_pause": True,
        "reduce_only": bool(reduce_only),
        "subaccount": 0,
        "exchange_index": int(idx),
    }
    chosen_group = group_for_ticker(group_id, ticker) if group_id else None
    if chosen_group:
        p["order_group_id"] = str(chosen_group)
    return p


def _balance_breakdown(body):
    out = {}
    for row in (body or {}).get("balance_breakdown") or []:
        idx = _int_or_none(row.get("exchange_index"))
        amount = _finite(row.get("balance"))
        if idx is not None and amount is not None:
            out[int(idx)] = float(amount)
    return out


def get_shard_balances(client=None):
    client = client or Q1.LiveClient()
    body, timing = client.get("/portfolio/balance", params={"subaccount": 0})
    return {
        "raw": body,
        "timing": timing,
        "breakdown_usd": _balance_breakdown(body),
        "total_balance_dollars": body.get("balance_dollars"),
    }


def _create_groups(client):
    snap = routing_snapshot(require_all=True, refresh=True)
    shards = list(snap["shards"])
    if not shards:
        raise RuntimeError("No execution shards discovered for 12-series universe")
    limit = str(
        getattr(B, "GROUP_LIMIT_FP", None)
        or max(25.0, 20.0 * 10.0)
    )
    groups = {}
    responses = {}
    timings = {}
    created = []
    try:
        for idx in shards:
            body, timing = client.post(
                "/portfolio/order_groups/create",
                {
                    "subaccount": 0,
                    "contracts_limit_fp": limit,
                    "exchange_index": int(idx),
                },
            )
            gid = str((body or {}).get("order_group_id") or "")
            if not gid:
                raise RuntimeError(
                    f"Order-group response missing id for shard {idx}: {body}"
                )
            returned_idx = _int_or_none((body or {}).get("exchange_index"))
            if returned_idx is not None and returned_idx != int(idx):
                raise RuntimeError(
                    f"Order group created on wrong shard: expected={idx} body={body}"
                )
            groups[str(int(idx))] = gid
            responses[str(int(idx))] = body
            timings[str(int(idx))] = timing
            created.append((int(idx), gid))
    except Exception:
        for idx, gid in created:
            try:
                client.delete(
                    f"/portfolio/order_groups/{gid}",
                    params={"subaccount": 0, "exchange_index": int(idx)},
                )
            except Exception:
                pass
        raise
    return groups, responses, timings


def _trigger_groups(client, gid):
    if not gid:
        return {"ok": True, "groups": {}, "note": "no groups"}
    items = gid.items() if isinstance(gid, dict) else [(None, gid)]
    results = {}
    all_ok = True
    for key, group_id in items:
        if not group_id:
            continue
        idx = _int_or_none(key)
        params = {"subaccount": 0}
        if idx is not None:
            params["exchange_index"] = int(idx)
        try:
            body, timing = client.request(
                "PUT",
                f"/portfolio/order_groups/{group_id}/trigger",
                params=params,
                payload={},
            )
            rec = {"ok": True, "body": body, "timing": timing}
        except Exception as exc:
            rec = {"ok": False, "error": repr(exc)}
            all_ok = False
        results[str(key)] = rec
    return {"ok": bool(all_ok), "groups": results}


def _delete_groups(client, gid):
    if not gid:
        return {"ok": True, "groups": {}, "note": "no groups"}
    items = gid.items() if isinstance(gid, dict) else [(None, gid)]
    results = {}
    all_ok = True
    for key, group_id in items:
        if not group_id:
            continue
        idx = _int_or_none(key)
        params = {"subaccount": 0}
        if idx is not None:
            params["exchange_index"] = int(idx)
        try:
            body, timing = client.delete(
                f"/portfolio/order_groups/{group_id}",
                params=params,
            )
            rec = {"ok": True, "body": body, "timing": timing}
        except Exception as exc:
            rec = {"ok": False, "error": repr(exc)}
            all_ok = False
        results[str(key)] = rec
    return {"ok": bool(all_ok), "groups": results}


def _candidate_shards(order_id, explicit=None):
    values = []
    known = shard_for_order(order_id)
    if known is not None:
        values.append(int(known))
    ex = _int_or_none(explicit)
    if ex is not None and ex not in values:
        values.append(int(ex))
    try:
        snap = routing_snapshot(require_all=True, refresh=False)
        for idx in snap.get("shards") or []:
            if int(idx) not in values:
                values.append(int(idx))
    except Exception:
        pass
    if not values:
        values = [0]
    return values


def _safe_cancel_one_shard(client, *, order_id, submitted_qty, exchange_index):
    oid = str(order_id)
    submitted_qty = float(submitted_qty)
    idx = int(exchange_index)
    errors = []

    for attempt, delay in enumerate((0.0, 0.05, 0.12, 0.25), start=1):
        if delay:
            time.sleep(delay)
        try:
            body, timing = client.delete(
                f"/portfolio/events/orders/{oid}",
                params={"subaccount": 0, "exchange_index": idx},
            )
            reduced = B._f((body or {}).get("reduced_by"), float("nan"))
            if not math.isfinite(reduced) or reduced < -B.EPS or reduced > submitted_qty + B.EPS:
                raise RuntimeError(f"invalid reduced_by={reduced} body={body}")
            register_order_shard(oid, idx)
            return {
                "ok": True,
                "source": "V2_CANCEL_DYNAMIC_SHARD"
                if attempt == 1
                else "V2_CANCEL_DYNAMIC_SHARD_RETRY",
                "exchange_index": idx,
                "fill_floor": max(
                    0.0,
                    min(
                        submitted_qty,
                        submitted_qty - max(0.0, reduced),
                    ),
                ),
                "body": body,
                "timing": timing,
                "errors": errors,
            }
        except Exception as exc:
            errors.append(repr(exc))
            try:
                body, timing = client.get(
                    f"/portfolio/orders/{oid}",
                    params={"subaccount": 0, "exchange_index": idx},
                )
                row = (body or {}).get("order") or {}
                rem = V1._order_remaining(row, 0.0)
                status = str(row.get("status") or "").lower()
                if rem <= B.EPS or status != "resting":
                    register_order_shard(oid, idx)
                    return {
                        "ok": True,
                        "source": "V2_CANCEL_DYNAMIC_ERROR_BUT_TERMINAL",
                        "exchange_index": idx,
                        "fill_floor": min(
                            submitted_qty,
                            max(0.0, V1._order_fill_count(row, 0.0)),
                        ),
                        "body": row,
                        "timing": timing,
                        "errors": errors,
                    }
            except Exception:
                pass

    batch_body = None
    batch_timing = None
    batch_error = None
    fill_floor = float("nan")
    try:
        batch_body, batch_timing = client.request(
            "DELETE",
            "/portfolio/events/orders/batched",
            payload={
                "orders": [
                    {
                        "order_id": oid,
                        "subaccount": 0,
                        "exchange_index": idx,
                    }
                ]
            },
        )
        rows = (batch_body or {}).get("orders") or []
        row = next(
            (r for r in rows if str(r.get("order_id") or "") == oid),
            None,
        )
        if row is None:
            raise RuntimeError(f"batch response missing order {oid}: {batch_body}")
        if row.get("error"):
            raise RuntimeError(f"batch item error: {row['error']}")
        reduced = B._f(row.get("reduced_by"), float("nan"))
        if not math.isfinite(reduced) or reduced < -B.EPS or reduced > submitted_qty + B.EPS:
            raise RuntimeError(f"invalid batch reduced_by={reduced}: {row}")
        fill_floor = max(
            0.0,
            min(
                submitted_qty,
                submitted_qty - max(0.0, reduced),
            ),
        )
    except Exception as exc:
        batch_error = repr(exc)

    time.sleep(0.08)
    try:
        body, verify_timing = client.get(
            f"/portfolio/orders/{oid}",
            params={"subaccount": 0, "exchange_index": idx},
        )
        row = (body or {}).get("order") or {}
        rem = V1._order_remaining(row, float("nan"))
        status = str(row.get("status") or "").lower()
        if status == "resting" and (
            not math.isfinite(rem) or rem > B.EPS
        ):
            return {
                "ok": False,
                "still_resting": True,
                "exchange_index": idx,
                "fill_floor": max(0.0, V1._order_fill_count(row, 0.0)),
                "errors": errors,
                "batch_error": batch_error,
                "batch_body": batch_body,
                "verify": row,
                "verify_timing": verify_timing,
            }
        if not math.isfinite(fill_floor):
            fill_floor = max(0.0, V1._order_fill_count(row, 0.0))
        register_order_shard(oid, idx)
        return {
            "ok": True,
            "source": "V2_DYNAMIC_BATCH_CANCEL_OR_TERMINAL_VERIFY",
            "exchange_index": idx,
            "fill_floor": min(
                submitted_qty,
                max(0.0, float(fill_floor)),
            ),
            "errors": errors,
            "batch_error": batch_error,
            "batch_body": batch_body,
            "batch_timing": batch_timing,
            "verify": row,
            "verify_timing": verify_timing,
        }
    except Exception as exc:
        return {
            "ok": False,
            "still_resting": None,
            "exchange_index": idx,
            "fill_floor": 0.0
            if not math.isfinite(fill_floor)
            else float(fill_floor),
            "errors": errors,
            "batch_error": batch_error,
            "verify_error": repr(exc),
        }


def _safe_cancel_v2_dynamic(
    client,
    *,
    order_id,
    submitted_qty,
    exchange_index=None,
):
    attempts = []
    for idx in _candidate_shards(order_id, explicit=exchange_index):
        result = _safe_cancel_one_shard(
            client,
            order_id=order_id,
            submitted_qty=submitted_qty,
            exchange_index=idx,
        )
        attempts.append(result)
        if result.get("ok") is True:
            result["shard_attempts"] = attempts
            return result
        # Explicitly still-resting means we found the authoritative shard; do not
        # pretend another shard can make that order terminal.
        if result.get("still_resting") is True:
            result["shard_attempts"] = attempts
            return result
    return {
        "ok": False,
        "still_resting": None,
        "fill_floor": 0.0,
        "shard_attempts": attempts,
        "error": "order shard unresolved or cancel failed on every discovered shard",
    }


def _dynamic_request(
    self,
    method,
    path,
    *,
    params=None,
    payload=None,
    timeout=8.0,
):
    method = str(method).upper()
    path = str(path)
    p = None if params is None else dict(params)
    body = None if payload is None else dict(payload)

    if path == "/portfolio/events/orders" and method == "POST":
        body = dict(body or {})
        ticker = str(body.get("ticker") or "")
        body["exchange_index"] = int(shard_for_ticker(ticker))

    elif path == "/portfolio/events/orders/batched" and method in {"POST", "DELETE"}:
        body = dict(body or {})
        routed = []
        for row in body.get("orders") or []:
            z = dict(row)
            ticker = str(z.get("ticker") or "")
            oid = str(z.get("order_id") or "")
            if ticker:
                z["exchange_index"] = int(shard_for_ticker(ticker))
            elif oid:
                idx = shard_for_order(oid)
                if idx is not None:
                    z["exchange_index"] = int(idx)
            routed.append(z)
        body["orders"] = routed

    elif path.startswith("/portfolio/events/orders/") and method in {"DELETE", "POST", "PUT"}:
        oid = path.rsplit("/", 1)[-1]
        idx = shard_for_order(oid)
        p = dict(p or {})
        if idx is not None:
            p["exchange_index"] = int(idx)
        elif _int_or_none(p.get("exchange_index")) is None:
            raise RuntimeError(
                f"Mutation shard unknown for order_id={oid}; refusing ambiguous request"
            )
        p["subaccount"] = int(p.get("subaccount", 0) or 0)

    if method == "GET" and path in {
        "/portfolio/orders",
        "/portfolio/positions",
        "/portfolio/fills",
    }:
        p = dict(p or {})
        ticker = str(p.get("ticker") or "")
        oid = str(p.get("order_id") or "")
        if ticker:
            p["exchange_index"] = int(shard_for_ticker(ticker))
        elif oid:
            idx = shard_for_order(oid)
            if idx is not None:
                p["exchange_index"] = int(idx)
            else:
                p.pop("exchange_index", None)
        else:
            p.pop("exchange_index", None)

    elif method == "GET" and path.startswith("/portfolio/orders/"):
        oid = path.rsplit("/", 1)[-1]
        idx = shard_for_order(oid)
        if idx is not None:
            p = dict(p or {})
            p["subaccount"] = int(p.get("subaccount", 0) or 0)
            p["exchange_index"] = int(idx)

    result = _ORIGINAL_REQUEST(
        self,
        method,
        path,
        params=p,
        payload=body,
        timeout=timeout,
    )

    if path == "/portfolio/events/orders" and method == "POST":
        try:
            response_body = result[0] if isinstance(result, tuple) else result
            oid = str((response_body or {}).get("order_id") or "")
            ticker = str((body or {}).get("ticker") or "")
            if oid and ticker:
                register_order_shard(oid, shard_for_ticker(ticker))
        except Exception:
            pass

    return result


def install_runtime_patch():
    """Install dynamic routing into the inherited live stack. No API calls."""
    Q1.LiveClient.request = _dynamic_request
    B.Q1.LiveClient.request = _dynamic_request
    B.SERIES = tuple(SERIES)
    B._payload = _payload_multishard
    B._create_group = _create_groups
    B._trigger_group = _trigger_groups
    B._delete_group = _delete_groups
    V1._safe_cancel_v2 = _safe_cancel_v2_dynamic
    return {
        "router_version": ROUTER_VERSION,
        "series": list(SERIES),
        "dynamic_liveclient_request": Q1.LiveClient.request is _dynamic_request,
        "dynamic_payload": B._payload is _payload_multishard,
        "multi_group_create": B._create_group is _create_groups,
        "multi_group_trigger": B._trigger_group is _trigger_groups,
        "multi_group_delete": B._delete_group is _delete_groups,
        "dynamic_safe_cancel": V1._safe_cancel_v2 is _safe_cancel_v2_dynamic,
        "orders_sent": False,
    }


def static_self_check(*, show=True):
    checks = {
        "series_count_12": len(SERIES) == 12,
        "crypto_count_9": len(CRYPTO_SERIES) == 9,
        "commodity_count_3": len(COMMODITY_SERIES) == 3,
        "gold_present": "KXGOLD15M" in SERIES,
        "silver_present": "KXSILVER15M" in SERIES,
        "wti_present": "KXWTI15M" in SERIES,
        "ticker_prefix_crypto": _series_for_ticker(
            "KXBTC15M-26AUG291200-15"
        )
        == "KXBTC15M",
        "ticker_prefix_gold": _series_for_ticker(
            "KXGOLD15M-26AUG291200-15"
        )
        == "KXGOLD15M",
        "unknown_ticker_rejected": _series_for_ticker("NOTOURS-X") is None,
        "orders_sent": False,
        "api_called": False,
    }
    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"orders_sent", "api_called"}
    )
    out = {
        "router_version": ROUTER_VERSION,
        "series": list(SERIES),
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 128)
        print("TAIL25 MULTI12 ROUTER STATIC CHECK — NO API / NO ORDERS")
        print("=" * 128)
        for k, v in out.items():
            print(f"{k:64s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 router static check failed: {out}")
    return out


__all__ = [
    "ROUTER_VERSION",
    "CRYPTO_SERIES",
    "COMMODITY_SERIES",
    "SERIES",
    "discover_current_markets",
    "routing_snapshot",
    "shard_for_ticker",
    "shard_for_order",
    "register_order_shard",
    "group_ids",
    "group_for_ticker",
    "get_shard_balances",
    "install_runtime_patch",
    "static_self_check",
]
