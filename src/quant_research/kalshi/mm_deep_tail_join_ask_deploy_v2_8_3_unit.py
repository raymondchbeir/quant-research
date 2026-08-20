from __future__ import annotations

"""No-API / no-order unit checks for V2.8.3 strict discovery."""

from . import mm_deep_tail_join_ask_deploy_v2_8_3 as V283
from . import mm_event_time_m0_m5_recorder_v5_1 as V51


def run(*, show=True):
    checks = {}

    static = V283.static_self_check(show=False)
    checks["static_ok"] = static.get("ok") is True
    checks["alpha_rules_unchanged"] = static.get("alpha_rules_unchanged") is True
    checks["strict_discovery_fail_closed"] = static.get("strict_discovery_fail_closed") is True
    checks["capture_window_unchanged"] = static.get("capture_window_unchanged") is True
    checks["universe_unchanged"] = static.get("universe_unchanged") is True
    checks["fee_snapshot_reuse"] = static.get("child_fee_snapshot_reuse") is True

    old = V51.C.rest_get
    try:
        def fail(*args, **kwargs):
            raise RuntimeError("synthetic discovery outage")
        V51.C.rest_get = fail
        try:
            V51._discover_sync_strict()
            refused = False
        except RuntimeError as exc:
            refused = "STRICT_DISCOVERY_FAIL_CLOSED" in str(exc)
        checks["all_discovery_errors_refused"] = refused
    finally:
        V51.C.rest_get = old

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
        print("V2.8.3 STRICT DISCOVERY UNIT — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:44s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.8.3 unit failed: {out}")
    return out


if __name__ == "__main__":
    run(show=True)
