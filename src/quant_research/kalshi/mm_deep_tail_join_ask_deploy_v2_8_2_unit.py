from __future__ import annotations

"""No-API unit checks for V2.8.2 child fee snapshot reuse."""

import tempfile
import time
from pathlib import Path

import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_deploy_v2_8_2 as V282


def _fee_snapshot(*, age_s=0.0, horizon_hours=OOS.FEE_CHANGE_HORIZON_H,
                  multipliers=None, ok=True):
    if multipliers is None:
        multipliers = {str(s): 1.0 for s in OOS.SERIES}
    return {
        "time": (
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=float(age_s))
        ).isoformat(),
        "ok": bool(ok),
        "horizon_hours": float(horizon_hours),
        "multipliers": dict(multipliers),
        "series": [],
        "problems": [],
    }


def run(*, show=True):
    checks = {}

    static = V282.static_self_check(show=False)
    checks["static_ok"] = static.get("ok") is True
    checks["alpha_rules_unchanged"] = static.get("alpha_rules_unchanged") is True
    checks["child_fee_snapshot_reuse"] = static.get("child_fee_snapshot_reuse") is True
    checks["child_fee_snapshot_fail_closed"] = static.get("child_fee_snapshot_fail_closed") is True
    checks["child_fee_api_called_false"] = static.get("child_fee_api_called") is False

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        B._atomic(root / "parent_preflight_snapshot.json", {
            "fee_preflight": _fee_snapshot(age_s=1.0),
        })

        got = V282._validated_parent_fee_snapshot(root, show=False)
        receipt = B._read(root / "child_fee_preflight_reuse_v2_8_2.json", {}) or {}
        checks["fresh_parent_snapshot_accepted"] = got.get("child_reused_parent_snapshot") is True
        checks["receipt_written"] = receipt.get("ok") is True
        checks["receipt_child_api_false"] = receipt.get("child_fee_api_called") is False
        checks["series_complete"] = set(receipt.get("series") or []) == set(OOS.SERIES)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        B._atomic(root / "parent_preflight_snapshot.json", {
            "fee_preflight": _fee_snapshot(age_s=V282.PARENT_FEE_MAX_AGE_S + 60.0),
        })
        try:
            V282._validated_parent_fee_snapshot(root, show=False)
            stale_refused = False
        except RuntimeError:
            stale_refused = True
        checks["stale_parent_snapshot_refused"] = stale_refused

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bad = {str(s): 1.0 for s in OOS.SERIES}
        bad.pop(next(iter(bad)))
        B._atomic(root / "parent_preflight_snapshot.json", {
            "fee_preflight": _fee_snapshot(age_s=1.0, multipliers=bad),
        })
        try:
            V282._validated_parent_fee_snapshot(root, show=False)
            missing_refused = False
        except RuntimeError:
            missing_refused = True
        checks["missing_series_refused"] = missing_refused

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
        print("V2.8.2 CHILD FEE SNAPSHOT UNIT — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:48s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.8.2 unit failed: {out}")
    return out


if __name__ == "__main__":
    run(show=True)
