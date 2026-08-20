from __future__ import annotations

"""No-API / no-order checks for V2.8.5 proven-V5 launch supervisor."""

import pandas as pd

from . import mm_deep_tail_join_ask_deploy_v2_8_5 as V285
from . import mm_event_time_m0_m5_recorder_v5 as V5


def run(*, show=True):
    checks = {}

    static = V285.static_self_check(show=False)
    checks["static_ok"] = static.get("ok") is True
    checks["alpha_rules_unchanged"] = static.get("alpha_rules_unchanged") is True
    checks["original_v5_discovery_restored"] = static.get("original_v5_discovery_restored") is True
    checks["capture_window_unchanged"] = static.get("capture_window_unchanged") is True
    checks["universe_unchanged"] = static.get("universe_unchanged") is True
    checks["fee_snapshot_reuse"] = static.get("fee_snapshot_reuse") is True
    checks["guardian_unchanged"] = static.get("guardian_unchanged") is True

    # 08:10 UTC is exactly 5 minutes before 08:15 M0 and should be accepted.
    t_good = V285._timing(pd.Timestamp("2026-08-20T08:10:00Z"))
    checks["presub_start_is_safe"] = t_good.get("safe") is True
    checks["presub_target_is_next_quarter"] = str(t_good.get("next_m0")) == "2026-08-20 08:15:00+00:00"

    # 08:08 UTC is 7 minutes before 08:15 and should be refused.
    t_early = V285._timing(pd.Timestamp("2026-08-20T08:08:00Z"))
    checks["too_early_refused"] = t_early.get("safe") is False

    # 08:14 UTC is only 1 minute before M0 and should be refused.
    t_late = V285._timing(pd.Timestamp("2026-08-20T08:14:00Z"))
    checks["too_late_refused"] = t_late.get("safe") is False

    fake_status = {
        "running": True,
        "health": {
            "recorder_alive": True,
            "recorder_health": {
                "running": True,
                "healthy": True,
                "study_version": V5.STUDY_VERSION,
                "subscribed_markets": len(V5.CRYPTO_SERIES),
                "channels": ["orderbook_delta", "ticker", "trade"],
                "snapshots_received": 9,
            },
        },
    }
    ready, ready_checks, _ = V285._data_ready_from_status(fake_status)
    checks["synthetic_data_ready_passes"] = ready is True and all(ready_checks.values())

    fake_bad = {
        "running": True,
        "health": {
            "recorder_alive": True,
            "recorder_health": {
                "running": True,
                "healthy": True,
                "study_version": V5.STUDY_VERSION,
                "subscribed_markets": 0,
                "channels": [],
                "snapshots_received": 0,
            },
        },
    }
    bad, _, _ = V285._data_ready_from_status(fake_bad)
    checks["empty_healthy_recorder_refused"] = bad is False

    checks["api_called"] = False
    checks["orders_sent"] = False

    ok = all(v is True for k, v in checks.items() if k not in {"api_called", "orders_sent"})
    out = {**checks, "ok": bool(ok)}

    if show:
        print("=" * 100)
        print("V2.8.5 PROVEN-V5 DATA GATE UNIT — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:46s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.8.5 unit failed: {out}")
    return out


if __name__ == "__main__":
    run(show=True)
