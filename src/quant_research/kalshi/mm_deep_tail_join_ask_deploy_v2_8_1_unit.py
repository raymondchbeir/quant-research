from __future__ import annotations

"""No-API unit checks for the V2.8.1 parent-side static-gate fix."""

from . import mm_deep_tail_join_ask_deploy_v2_8_1 as V281


def run(*, show=True):
    static = V281.static_self_check(show=False)

    checks = {
        "static_ok": static.get("ok") is True,
        "orders_sent_is_false": static.get("orders_sent") is False,
        "orders_sent_false_excluded": (
            static.get("static_gate_orders_sent_false_excluded_from_pass_boolean") is True
        ),
        "alpha_rules_unchanged": static.get("alpha_rules_unchanged") is True,
        "bounded_raw_ingestion": static.get("bounded_raw_ingestion") is True,
        "rss_guard_unchanged": static.get("rss_guard_unchanged") is True,
        "deadline_guard_unchanged": static.get("deadline_initial_grace_unchanged") is True,
        "cleanup_overrun_fail_closed": static.get("cleanup_overrun_still_fail_closed") is True,
        "api_called": False,
        "orders_sent": False,
    }

    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"api_called", "orders_sent"}
    )

    out = {**checks, "ok": bool(ok)}

    if show:
        print("=" * 100)
        print("V2.8.1 STATIC-GATE UNIT — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:44s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.8.1 unit failed: {out}")

    return out


if __name__ == "__main__":
    run(show=True)
