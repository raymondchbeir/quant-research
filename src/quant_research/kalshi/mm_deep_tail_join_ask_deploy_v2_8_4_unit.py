from __future__ import annotations

"""No-API / no-order checks for V2.8.4 bounded strict discovery."""

from datetime import timedelta

from . import mm_deep_tail_join_ask_deploy_v2_8_4 as V284
from . import mm_event_time_m0_m5_recorder_v5_2 as V52


def _fake_market(series, status):
    now = V52.C.utc_now()
    close = now + timedelta(minutes=10)
    return {
        "ticker": f"{series}-{status}-UNIT",
        "event_ticker": f"{series}-UNIT",
        "title": "unit",
        "open_time": (now - timedelta(minutes=5)).isoformat(),
        "close_time": close.isoformat(),
        "status": status,
    }


def run(*, show=True):
    checks = {}

    static = V284.static_self_check(show=False)
    checks["static_ok"] = static.get("ok") is True
    checks["alpha_rules_unchanged"] = static.get("alpha_rules_unchanged") is True
    checks["bounded_discovery"] = static.get("discovery_budget_below_supervisor") is True
    checks["workers_cover_queries"] = static.get("discovery_workers_cover_queries") is True
    checks["capture_window_unchanged"] = static.get("capture_window_unchanged") is True
    checks["universe_unchanged"] = static.get("universe_unchanged") is True
    checks["fee_snapshot_reuse"] = static.get("child_fee_snapshot_reuse") is True
    checks["parent_timeout_exceeds_recorder"] = static.get("parent_timeout_exceeds_recorder_timeout") is True
    checks["failed_startup_cleanup_enabled"] = static.get("failed_startup_cleanup_enabled") is True

    old = V52._query_one
    try:
        def ok_query(series, status):
            # Successful query for every pair. Only one status needs a raw market
            # for each series, but returning one for both keeps the fixture simple.
            return [_fake_market(series, status)]
        V52._query_one = ok_query
        got = V52._discover_sync_bounded()
        checks["synthetic_complete_discovery_passes"] = isinstance(got, dict)

        def fail_one(series, status):
            if str(series) == str(V52.V5.CRYPTO_SERIES[0]) and str(status) == "open":
                raise RuntimeError("synthetic outage")
            return [_fake_market(series, status)]
        V52._query_one = fail_one
        try:
            V52._discover_sync_bounded()
            refused = False
        except RuntimeError as exc:
            refused = "STRICT_DISCOVERY_FAIL_CLOSED" in str(exc)
        checks["synthetic_query_failure_refused"] = refused
    finally:
        V52._query_one = old

    checks["api_called"] = False
    checks["orders_sent"] = False

    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"api_called", "orders_sent"}
    )
    out = {**checks, "ok": bool(ok)}

    if show:
        print("=" * 100)
        print("V2.8.4 BOUNDED DISCOVERY UNIT — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:48s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.8.4 unit failed: {out}")
    return out


if __name__ == "__main__":
    run(show=True)
