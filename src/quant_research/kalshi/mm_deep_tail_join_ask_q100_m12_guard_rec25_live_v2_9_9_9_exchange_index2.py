from __future__ import annotations

"""V2.9.9.9 Q100 Kalshi exchange-index routing fix.

Observed failure addressed
--------------------------
On 2026-08-25 the live Q100 deployment discovered valid, active 15-minute crypto
markets, but every ENTRY create returned Kalshi ``404 market_not_found``.  A
read-only authoritative ``GET /markets/{ticker}`` check proved all nine markets
existed and were active.  The same market objects reported ``exchange_index = 2``
while every submitted order payload contained ``exchange_index = 0``.

The historical execution stack hard-coded exchange index 0 in several mutation
helpers inherited from the older Candidate-C live code:
- order CREATE payloads,
- order-group CREATE,
- single-order CANCEL,
- batched CANCEL fallback,
- order-group DELETE.

This wrapper changes ONLY those routing fields to exchange index 2 for the current
15-minute crypto deployment.  Strategy economics and risk behavior are unchanged:
Q100, 12h, M1 entry, M12 cleanup, danger guard, REC25=25%, atomic trigger snapshot,
fixed passive exit/no repricing, $20 software loss trigger, $125 minimum equity,
parent M12+45s hard recycle, 90s guardian, and V2.9.9.6 retry-until-flat recovery.

V2.9.9.8 semantic CREATE handling is retained as a secondary safety layer.  A
future definitive ENTRY market_not_found with no known exposure still disables only
that ticker; ambiguous/network/5xx/EXIT create failures remain fail-closed.

Importing this module performs no API calls and sends no orders.
"""

import inspect
import time
import numpy as np

from . import mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_8_semantic_create_binding as BASE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q100_M1_M12_GUARD_REC25_V2_9_9_9_EXCHANGE_INDEX2"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_9_exchange_index2"
Q100_ARM = "LIVE_DEEP_TAIL_Q100_M1_M12_GUARD_REC25_12H_V2999"
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

LIVE_EXCHANGE_INDEX = 2

# Preserve inherited implementations across notebook reloads.
if not hasattr(B, "_v2999_original_payload"):
    B._v2999_original_payload = B._payload
if not hasattr(B, "_v2999_original_create_group"):
    B._v2999_original_create_group = B._create_group
if not hasattr(B, "_v2999_original_delete_group"):
    B._v2999_original_delete_group = B._delete_group
if not hasattr(B, "_v2999_original_cancel"):
    B._v2999_original_cancel = B._cancel
if not hasattr(V1, "_v2999_original_safe_cancel_v2"):
    V1._v2999_original_safe_cancel_v2 = V1._safe_cancel_v2

_ORIGINAL_PAYLOAD = B._v2999_original_payload


def _payload_exchange_index2(*, ticker, side, qty, price, cid, post_only, reduce_only,
                             tif, group_id=None):
    """Build the inherited order payload, changing only exchange_index to 2."""
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
    p["exchange_index"] = int(LIVE_EXCHANGE_INDEX)
    return p


def _create_group_exchange_index2(client):
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
    return gid, body, timing


def _delete_group_exchange_index2(client, gid):
    try:
        body, timing = client.delete(
            f"/portfolio/order_groups/{gid}",
            params={
                "subaccount": 0,
                "exchange_index": int(LIVE_EXCHANGE_INDEX),
            },
        )
        return {"ok": True, "body": body, "timing": timing}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _cancel_exchange_index2(client, oid):
    try:
        body, timing = client.delete(
            f"/portfolio/events/orders/{oid}",
            params={
                "subaccount": 0,
                "exchange_index": int(LIVE_EXCHANGE_INDEX),
            },
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


def _safe_cancel_v2_exchange_index2(client, *, order_id, submitted_qty):
    """Inherited V11-style fail-closed cancel, routed to exchange index 2."""
    oid = str(order_id)
    submitted_qty = float(submitted_qty)
    errors = []

    for attempt, delay in enumerate((0.0, 0.05, 0.12, 0.25), start=1):
        if delay:
            time.sleep(delay)
        try:
            body, timing = client.delete(
                f"/portfolio/events/orders/{oid}",
                params={
                    "subaccount": 0,
                    "exchange_index": int(LIVE_EXCHANGE_INDEX),
                },
            )
            reduced = B._f((body or {}).get("reduced_by"), np.nan)
            if not np.isfinite(reduced) or reduced < -V1.EPS or reduced > submitted_qty + V1.EPS:
                raise RuntimeError(f"invalid reduced_by={reduced} body={body}")
            return {
                "ok": True,
                "source": "V2_CANCEL" if attempt == 1 else "V2_CANCEL_RETRY",
                "fill_floor": max(0.0, min(submitted_qty, submitted_qty - max(0.0, reduced))),
                "body": body,
                "timing": timing,
                "errors": errors,
            }
        except Exception as exc:
            errors.append(repr(exc))
            try:
                body, timing = client.get(f"/portfolio/orders/{oid}")
                row = (body or {}).get("order") or {}
                rem = V1._order_remaining(row, 0.0)
                status = str(row.get("status") or "").lower()
                if rem <= V1.EPS or status != "resting":
                    return {
                        "ok": True,
                        "source": "V2_CANCEL_ERROR_BUT_ORDER_TERMINAL",
                        "fill_floor": min(submitted_qty, max(0.0, V1._order_fill_count(row, 0.0))),
                        "body": row,
                        "timing": timing,
                        "errors": errors,
                    }
            except Exception:
                pass

    batch_body = None
    batch_timing = None
    batch_error = None
    try:
        batch_body, batch_timing = client.request(
            "DELETE",
            "/portfolio/events/orders/batched",
            payload={
                "orders": [
                    {
                        "order_id": oid,
                        "subaccount": 0,
                        "exchange_index": int(LIVE_EXCHANGE_INDEX),
                    }
                ]
            },
        )
        rows = (batch_body or {}).get("orders") or []
        row = next((r for r in rows if str(r.get("order_id") or "") == oid), None)
        if row is None:
            raise RuntimeError(f"batch response missing order {oid}: {batch_body}")
        if row.get("error"):
            raise RuntimeError(f"batch item error: {row['error']}")
        reduced = B._f(row.get("reduced_by"), np.nan)
        if not np.isfinite(reduced) or reduced < -V1.EPS or reduced > submitted_qty + V1.EPS:
            raise RuntimeError(f"invalid batch reduced_by={reduced}: {row}")
        fill_floor = max(0.0, min(submitted_qty, submitted_qty - max(0.0, reduced)))
    except Exception as exc:
        batch_error = repr(exc)
        fill_floor = np.nan

    time.sleep(0.08)
    try:
        body, verify_timing = client.get(f"/portfolio/orders/{oid}")
        row = (body or {}).get("order") or {}
        rem = V1._order_remaining(row, np.nan)
        status = str(row.get("status") or "").lower()
        if status == "resting" and (not np.isfinite(rem) or rem > V1.EPS):
            return {
                "ok": False,
                "still_resting": True,
                "fill_floor": max(0.0, V1._order_fill_count(row, 0.0)),
                "errors": errors,
                "batch_error": batch_error,
                "batch_body": batch_body,
                "verify": row,
                "verify_timing": verify_timing,
            }
        if not np.isfinite(fill_floor):
            fill_floor = max(0.0, V1._order_fill_count(row, 0.0))
        return {
            "ok": True,
            "source": "V2_BATCH_CANCEL_OR_TERMINAL_VERIFY",
            "fill_floor": min(submitted_qty, max(0.0, float(fill_floor))),
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
            "fill_floor": 0.0 if not np.isfinite(fill_floor) else float(fill_floor),
            "errors": errors,
            "batch_error": batch_error,
            "verify_error": repr(exc),
        }


def _install_patch():
    """Install V2.9.9.8 plus exchange-index-2 mutation routing."""
    BASE._install_patch()

    B._payload = _payload_exchange_index2
    B._create_group = _create_group_exchange_index2
    B._delete_group = _delete_group_exchange_index2
    B._cancel = _cancel_exchange_index2
    V1._safe_cancel_v2 = _safe_cancel_v2_exchange_index2

    # Q100 launch identity/parameters.
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
    P._recover_generation_fail_closed = BASE.BASE.BASE._recover_generation_fail_closed_retry

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.POST_M5_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state

    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline structural/regression audit. No API calls and no orders."""
    base = BASE.static_self_check(show=False)
    _install_patch()

    sample = _payload_exchange_index2(
        ticker="TEST",
        side="bid",
        qty=100.0,
        price=0.05,
        cid="test-v2999",
        post_only=True,
        reduce_only=False,
        tif="good_till_canceled",
        group_id="gid-test",
    )

    create_src = inspect.getsource(_create_group_exchange_index2)
    cancel_src = inspect.getsource(_safe_cancel_v2_exchange_index2)

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
        "exchange_index_exact_2": LIVE_EXCHANGE_INDEX == 2,
        "create_payload_exchange_index2": int(sample.get("exchange_index", -1)) == 2,
        "group_create_exchange_index2": '"exchange_index": int(LIVE_EXCHANGE_INDEX)' in create_src,
        "single_cancel_exchange_index2": '"exchange_index": int(LIVE_EXCHANGE_INDEX)' in cancel_src,
        "batch_cancel_exchange_index2": '"exchange_index": int(LIVE_EXCHANGE_INDEX)' in cancel_src and '"/portfolio/events/orders/batched"' in cancel_src,
        "payload_patch_installed": B._payload is _payload_exchange_index2,
        "group_patch_installed": B._create_group is _create_group_exchange_index2,
        "cancel_patch_installed": V1._safe_cancel_v2 is _safe_cancel_v2_exchange_index2,
        "semantic_market_not_found_layer_preserved": base.get("market_not_found_exact_detected") is True,
        "ambiguous_timeout_fail_closed_preserved": base.get("ambiguous_timeout_not_local") is True,
        "rec25_engine_binding_preserved": base.get("rec25_engine_bound") is True,
        "all_public_engine_aliases_binding_preserved": base.get("all_public_engine_aliases_bound") is True,
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": LIVE.PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": LIVE.PASSIVE_EXIT_TIF == "good_till_canceled",
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
        "live_exchange_index": LIVE_EXCHANGE_INDEX,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 180)
        print("V2.9.9.9 Q100 EXCHANGE-INDEX ROUTING STATIC CHECK — NO API / NO ORDERS")
        print("=" * 180)
        for k, v in out.items():
            print(f"{k:116s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.9.9 static self-check failed: {out}")
    return out


def q100_preflight(*, show=True):
    _install_patch()
    static_self_check(show=show)
    return BASE.q100_preflight(show=show)


def start_q100_12h_smoke(*, arm_phrase=None):
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
    "M12_HARD_RECYCLE_GRACE_S",
    "GUARDIAN_POST_M12_EXIT_TIMEOUT_S",
    "RECOVERY_FRACTION",
    "RECOVERY_RETRY_WINDOW_S",
    "LIVE_EXCHANGE_INDEX",
    "static_self_check",
    "q100_preflight",
    "start_q100_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
