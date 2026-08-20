from __future__ import annotations

"""No-API unit checks for the V2.8 cleanup-aware guardian."""

import json
import tempfile
from pathlib import Path

from . import mm_deep_tail_join_ask_deploy_v2_8 as V28


def _append(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, separators=(",", ":")) + "\n")


def run(*, show=True):
    checks = {}

    with tempfile.TemporaryDirectory(prefix="v28_guardian_unit_") as td:
        root = Path(td)
        risk = root / "risk_events.jsonl"
        final = root / "final_summary.json"

        checks["no_shutdown_marker_false"] = V28._runtime_shutdown_started(root) is False

        _append(risk, {"event": "SHUTDOWN_START", "reason": "MANUAL_KILL"})
        checks["manual_shutdown_not_runtime"] = V28._runtime_shutdown_started(root) is False

        _append(risk, {"event": "SHUTDOWN_START", "reason": "RUNTIME_COMPLETE"})
        checks["runtime_shutdown_marker_true"] = V28._runtime_shutdown_started(root) is True

        final.write_text(json.dumps({
            "shutdown_reason": "RUNTIME_COMPLETE",
            "flat_verified": True,
            "strategy_resting_orders_zero": True,
            "last_error": None,
        }), encoding="utf-8")
        clean, body = V28._clean_runtime_final(root)
        checks["clean_runtime_final_true"] = clean is True and body.get("flat_verified") is True

        final.write_text(json.dumps({
            "shutdown_reason": "RUNTIME_COMPLETE",
            "flat_verified": False,
            "strategy_resting_orders_zero": True,
            "last_error": None,
        }), encoding="utf-8")
        clean, _ = V28._clean_runtime_final(root)
        checks["unsafe_final_not_accepted"] = clean is False

        final.write_text(json.dumps({
            "shutdown_reason": "ENGINE_EXCEPTION",
            "flat_verified": True,
            "strategy_resting_orders_zero": True,
            "last_error": None,
        }), encoding="utf-8")
        clean, _ = V28._clean_runtime_final(root)
        checks["non_runtime_final_not_accepted"] = clean is False

    static = V28.static_self_check(show=False)
    checks["static_ok"] = static.get("ok") is True
    checks["alpha_rules_unchanged"] = static.get("alpha_rules_unchanged") is True
    checks["cleanup_overrun_fail_closed"] = static.get("cleanup_overrun_still_fail_closed") is True
    checks["orders_sent"] = False
    checks["api_called"] = False

    ok = all(v is True for k, v in checks.items() if k not in {"orders_sent", "api_called"})
    out = {"ok": bool(ok), **checks}

    if show:
        print("=" * 100)
        print("V2.8 CLEANUP-AWARE GUARDIAN UNIT — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:42s}: {v}")

    if not ok:
        raise AssertionError(out)
    return out


if __name__ == "__main__":
    run(show=True)
