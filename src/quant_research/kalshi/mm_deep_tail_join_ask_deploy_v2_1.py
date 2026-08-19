from __future__ import annotations

"""V2.1 deployment gate: require enough Kalshi write-bucket burst capacity.

The deep-tail strategy posts two entry orders for each of the nine markets at M1.
Current V2 creates are token-billed per order. To preserve the tested simultaneous-M1
mechanic rather than silently stagger markets, this wrapper reads the account's current
API limits and endpoint costs and refuses to arm unless the write bucket can admit the
entire 18-order pulse.

No strategy or execution rule changes. Importing this module sends no orders.
"""

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_deploy_v2 as D

DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_1_RATE_CAPACITY_GATE"


def _create_cost(client):
    body, timing = client.get("/account/endpoint_costs")
    default = int(B._f((body or {}).get("default_cost"), 10))
    cost = default
    for r in (body or {}).get("endpoint_costs") or []:
        method = str(r.get("method") or "").upper()
        path = str(r.get("path") or "").rstrip("/")
        if method == "POST" and path.endswith("/portfolio/events/orders"):
            cost = int(B._f(r.get("cost"), default))
            break
    return int(cost), body, timing


def api_capacity_preflight(*, show=True):
    """Read-only authenticated API tier/bucket check. No orders/cancels."""
    client = B.Q1.LiveClient()
    limits, limits_timing = client.get("/account/limits")
    create_cost, costs, costs_timing = _create_cost(client)

    write = (limits or {}).get("write") or {}
    refill = B._f(write.get("refill_rate"), float("nan"))
    capacity = B._f(write.get("bucket_capacity"), float("nan"))
    n_orders = 2 * len(B.SERIES)
    required = float(n_orders * create_cost)
    ok = bool(capacity == capacity and capacity >= required - 1e-9)

    out = {
        "ok": ok,
        "usage_tier": (limits or {}).get("usage_tier"),
        "write_refill_rate_tokens_per_s": refill,
        "write_bucket_capacity_tokens": capacity,
        "create_cost_tokens": create_cost,
        "simultaneous_m1_orders": n_orders,
        "required_m1_burst_tokens": required,
        "limits_timing": limits_timing,
        "endpoint_costs_timing": costs_timing,
        "orders_sent": False,
    }
    if show:
        print("=" * 100)
        print("DEEP-TAIL API CAPACITY PREFLIGHT — READ ONLY")
        print("=" * 100)
        print("Usage tier:                    ", out["usage_tier"])
        print("Write refill tokens/s:         ", refill)
        print("Write bucket capacity:         ", capacity)
        print("V2 create token cost:          ", create_cost)
        print("M1 simultaneous create orders: ", n_orders)
        print("Required M1 burst tokens:      ", required)
        print("Exact simultaneous-M1 capable: ", ok)
        print("ORDERS SENT:                   NO")
    if not ok:
        raise RuntimeError(
            "API WRITE-CAPACITY GATE FAILED. This account cannot currently admit all "
            f"{n_orders} M1 V2 creates in one burst (need {required:g} tokens, "
            f"bucket capacity={capacity}). Refusing to silently stagger entry timing, "
            "because that would change the tested strategy."
        )
    return out


def static_self_check(*, show=True):
    return D.static_self_check(show=show)


def private_ws_probe(**kwargs):
    return D.private_ws_probe(**kwargs)


def live_preflight(*, quote_size, runtime_hours, max_start_loss_usd,
                   min_start_equity_usd, show=True, probe_private_ws=True):
    cap = api_capacity_preflight(show=show)
    out = D.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
        probe_private_ws=probe_private_ws,
    )
    out = dict(out)
    out["api_capacity_preflight"] = cap
    out["deploy_wrapper_version"] = DEPLOY_VERSION
    return out


def start_q1_smoke(**kwargs):
    api_capacity_preflight(show=True)
    return D.start_q1_smoke(**kwargs)


def q1_promotion_check(*args, **kwargs):
    return D.q1_promotion_check(*args, **kwargs)


def start_q5_overnight(**kwargs):
    api_capacity_preflight(show=True)
    return D.start_q5_overnight(**kwargs)


def start_ladder_stage(**kwargs):
    api_capacity_preflight(show=True)
    return D.start_ladder_stage(**kwargs)


def live_status(**kwargs):
    return D.live_status(**kwargs)


def kill_and_flatten_live(**kwargs):
    return D.kill_and_flatten_live(**kwargs)


Q1_ARM = D.Q1_ARM
Q5_ARM = D.Q5_ARM
KILL_ARM = D.KILL_ARM
PROMOTION_PATH = D.PROMOTION_PATH


__all__ = [
    "DEPLOY_VERSION",
    "Q1_ARM",
    "Q5_ARM",
    "KILL_ARM",
    "PROMOTION_PATH",
    "api_capacity_preflight",
    "static_self_check",
    "private_ws_probe",
    "live_preflight",
    "start_q1_smoke",
    "q1_promotion_check",
    "start_q5_overnight",
    "start_ladder_stage",
    "live_status",
    "kill_and_flatten_live",
]
