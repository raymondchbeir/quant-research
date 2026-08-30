from __future__ import annotations

"""Tail25 Multi12 V1.2 final deployment-audit wrapper.

This wrapper is the intended operator entrypoint.  It preserves the V1/V1.1
strategy, risk, recorder, rotating-supervisor and guardian architecture while
binding the V1.2 lifecycle-audited child engine and adding three read-only gates
found during the final source audit:

1. Fee preflight must actually cover all 12 deployed series, not the historical
   nine-series crypto constant captured by the old OOS module at import time.
2. Flat per-exchange balance breakdown must reconcile to the independently-read
   account equity before that breakdown is allowed to become the session risk
   baseline.  This prevents a unit/schema mismatch from manufacturing a bogus
   $30 kill threshold.
3. The parent-recorder stop receipt publishes the historical ``dead`` key expected
   by the mature Q1 promotion writer, and waits briefly after SIGTERM if needed.

The Q1 smoke remains mandatory on the exact same Git HEAD before Q10 can arm.
Importing this module performs no API calls, orders, cancels or transfers.
"""

import argparse
import math
import time
from pathlib import Path

from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_tail25_join_bbo_live_v1_2_audit as LIVE
from . import mm_tail25_join_bbo_q10_12h_live_v1_1 as V11


V1 = V11.V1
REC = V11.REC
ROUTER = V11.ROUTER
P = V11.P
RUNTIME = V11.RUNTIME
V28 = V11.V28
B = V11.B
CORE = V1.CORE

DEPLOY_VERSION = V11.DEPLOY_VERSION
PATCH_VERSION = "TAIL25_MULTI12_FINAL_AUDIT_V1_2"
MODULE_NAME = "quant_research.kalshi.mm_tail25_join_bbo_q10_12h_live_v1_2"

Q1_ARM = "LIVE_TAIL25_MULTI12_Q1_ONE_WINDOW_V1_2"
Q10_ARM = "LIVE_TAIL25_MULTI12_Q10_12H_V1_2"
KILL_ARM = V11.KILL_ARM

Q1_Q = V11.Q1_Q
Q10_Q = V11.Q10_Q
Q10_HOURS = V11.Q10_HOURS
Q10_MAX_LOSS_USD = V11.Q10_MAX_LOSS_USD
Q10_MIN_EQUITY_USD = V11.Q10_MIN_EQUITY_USD
PROMOTION_PATH = V11.PROMOTION_PATH

BALANCE_RECONCILE_ABS_TOL_USD = 0.05
RECORDER_POST_TERM_WAIT_S = 5.0

_ORIGINAL_V11_INSTALL = V11._install_patch
_ORIGINAL_READ_ONLY_PREFLIGHT = V1._read_only_preflight
_ORIGINAL_RECORDER_STOP = REC.stop_parent_recorder


def _fee_series_set(preflight):
    fee = (preflight or {}).get("fee_preflight") or {}
    return {
        str((row or {}).get("series") or "")
        for row in fee.get("series") or []
        if str((row or {}).get("series") or "")
    }


def _balance_reconciliation(preflight, *, tolerance_usd=BALANCE_RECONCILE_ABS_TOL_USD):
    preflight = preflight or {}
    shard_equity = V1._finite(preflight.get("risk_baseline_equity_usd"))
    inherited_equity = V1._finite(preflight.get("inherited_preflight_equity_usd"))
    if shard_equity is None or inherited_equity is None:
        return {
            "ok": False,
            "reason": "missing independent equity values",
            "shard_equity_usd": shard_equity,
            "independent_account_equity_usd": inherited_equity,
            "delta_usd": None,
            "tolerance_usd": float(tolerance_usd),
        }
    delta = float(shard_equity - inherited_equity)
    return {
        "ok": abs(delta) <= float(tolerance_usd) + 1e-12,
        "reason": None if abs(delta) <= float(tolerance_usd) + 1e-12 else "flat shard breakdown does not reconcile to account equity",
        "shard_equity_usd": float(shard_equity),
        "independent_account_equity_usd": float(inherited_equity),
        "delta_usd": delta,
        "tolerance_usd": float(tolerance_usd),
    }


def _hardened_read_only_preflight(
    *,
    q,
    hours,
    max_loss,
    min_equity,
    min_per_shard,
    probe_private_ws,
    show,
    run_static=True,
):
    """Read-only V1 preflight plus exact 12-series fee and balance gates."""
    if run_static:
        static_self_check(show=bool(show))
        _install_patch()

    # The historical OOS module captured its nine-series tuple at import time.
    # Change it only for this read-only fee snapshot, then restore it in finally so
    # inherited historical static checks remain isolated and reproducible.
    old_fee_series = OOS.SERIES
    OOS.SERIES = tuple(ROUTER.SERIES)
    try:
        out = _ORIGINAL_READ_ONLY_PREFLIGHT(
            q=float(q),
            hours=float(hours),
            max_loss=float(max_loss),
            min_equity=float(min_equity),
            min_per_shard=float(min_per_shard),
            probe_private_ws=bool(probe_private_ws),
            show=bool(show),
            run_static=False,
        )
    finally:
        OOS.SERIES = old_fee_series
        _install_patch()

    out = dict(out or {})
    expected = set(ROUTER.SERIES)
    fee_seen = _fee_series_set(out)
    fee_ok = bool(
        ((out.get("fee_preflight") or {}).get("ok") is True)
        and fee_seen == expected
    )
    if not fee_ok:
        raise RuntimeError(
            "Tail25 fee preflight did not prove the current fee contract for all "
            f"12 deployed series. expected={sorted(expected)} seen={sorted(fee_seen)} "
            f"fee_preflight={out.get('fee_preflight')}"
        )

    balance = _balance_reconciliation(out)
    if not balance["ok"]:
        raise RuntimeError(
            "Tail25 flat-balance risk baseline reconciliation failed; refusing "
            f"launch. details={balance}"
        )

    out["fee_universe_audit"] = {
        "ok": True,
        "expected_series": sorted(expected),
        "verified_series": sorted(fee_seen),
        "series_count": len(fee_seen),
        "historical_crypto_only_fee_constant_overridden_temporarily": True,
    }
    out["flat_balance_reconciliation"] = balance
    out["deploy_patch_version"] = PATCH_VERSION
    out["live_patch_version"] = LIVE.PATCH_VERSION

    if show:
        print("=" * 156)
        print("TAIL25 V1.2 FINAL READ-ONLY AUDIT GATES")
        print("=" * 156)
        print("Fee series verified:          12/12")
        print(
            "Shard/account equity delta:  "
            f"${float(balance['delta_usd']):+.4f} "
            f"(limit ${BALANCE_RECONCILE_ABS_TOL_USD:.2f})"
        )
        print("Risk baseline unit check:     PASS")
        print("ORDERS SENT:                 NO")
        print("TRANSFERS SENT:              NO")

    return out


def _stop_external_recorder_compat(pid):
    """Return P's historical ``dead`` field and make termination verification strict."""
    result = dict(_ORIGINAL_RECORDER_STOP(pid) or {})
    pid = int(pid or 0)
    if pid > 0 and B._pid_alive(pid):
        deadline = time.time() + RECORDER_POST_TERM_WAIT_S
        while B._pid_alive(pid) and time.time() < deadline:
            time.sleep(0.10)
    dead = not (pid > 0 and B._pid_alive(pid))
    result["pid"] = pid or result.get("pid")
    result["dead"] = bool(dead)
    result["stopped"] = bool(dead)
    result["tail25_stop_receipt_compat_patch"] = PATCH_VERSION
    if not dead:
        raise RuntimeError(
            f"Tail25 parent recorder did not terminate authoritatively: {result}"
        )
    return result


def _publish_final_bindings():
    # Deployment V1 globals are resolved dynamically by detached supervisor code.
    V1.LIVE = LIVE
    V1.MODULE_NAME = MODULE_NAME
    V1.Q1_ARM = Q1_ARM
    V1.Q10_ARM = Q10_ARM
    V1._read_only_preflight = _hardened_read_only_preflight
    V1._stop_external_recorder = _stop_external_recorder_compat

    # V1.1's generation self-control functions also resolve these globals at call
    # time. Publish the final child/identity before they are used.
    V11.LIVE = LIVE
    V11.MODULE_NAME = MODULE_NAME
    V11.Q1_ARM = Q1_ARM
    V11.Q10_ARM = Q10_ARM

    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q10_ARM
    RUNTIME.LIVE = LIVE

    P.LIVE = LIVE
    P.REC = REC
    P._generation_cfg = V1._generation_cfg
    P._launch_generation = V1._launch_generation
    P._recover_generation_fail_closed = V1._recover_generation_fail_closed
    P._start_external_recorder = V1._start_external_recorder
    P._stop_external_recorder = _stop_external_recorder_compat
    P._fresh_generation_preflight = V11._fresh_generation_preflight


def _install_patch():
    # Set the desired child before the inherited installers publish their dynamic
    # bindings, then restore the final identity/gates afterward.
    V1.LIVE = LIVE
    V11.LIVE = LIVE
    V11.MODULE_NAME = MODULE_NAME
    V11.Q1_ARM = Q1_ARM
    V11.Q10_ARM = Q10_ARM
    _ORIGINAL_V11_INSTALL()
    _publish_final_bindings()

    # Internal V1/V1.1 calls must remain on this final installer.
    V1._install_patch = _install_patch
    V11._install_patch = _install_patch
    return {
        "deploy_version": DEPLOY_VERSION,
        "patch_version": PATCH_VERSION,
        "live_patch_version": LIVE.PATCH_VERSION,
        "module_name": MODULE_NAME,
        "orders_sent": False,
        "transfers_sent": False,
    }


def static_self_check(*, show=True):
    # Run inherited historical checks before any temporary 12-series OOS fee
    # override; the final fee-universe verification occurs in live preflight.
    _install_patch()
    inherited = V11.static_self_check(show=False)
    engine = LIVE.static_self_check(show=False)

    synthetic_ok = {
        "risk_baseline_equity_usd": 350.0,
        "inherited_preflight_equity_usd": 350.01,
    }
    synthetic_bad = {
        "risk_baseline_equity_usd": 35000.0,
        "inherited_preflight_equity_usd": 350.0,
    }
    b_ok = _balance_reconciliation(synthetic_ok)
    b_bad = _balance_reconciliation(synthetic_bad)

    checks = {
        "v11_supervisor_self_control_static_ok": inherited.get("ok") is True,
        "final_engine_static_ok": engine.get("ok") is True,
        "deploy_version_preserved": DEPLOY_VERSION == V1.DEPLOY_VERSION,
        "patch_version_exact": PATCH_VERSION == "TAIL25_MULTI12_FINAL_AUDIT_V1_2",
        "live_patch_version_exact": LIVE.PATCH_VERSION == "TAIL25_LIFECYCLE_AUDIT_V1_2",
        "q10_exact": Q10_Q == 10.0,
        "runtime_12h_exact": Q10_HOURS == 12.0,
        "loss_stop_30_exact": Q10_MAX_LOSS_USD == 30.0,
        "min_equity_300_exact": Q10_MIN_EQUITY_USD == 300.0,
        "strategy_25_2_15_join2_force3": (
            LIVE.ENTRY_OFFSET == 0.25
            and LIVE.ENTRY_REPRICE_HYSTERESIS == 0.02
            and LIVE.EDGE_ZONE == 0.15
            and LIVE.EXIT_REPRICE_HYSTERESIS == 0.02
            and LIVE.EXIT_HORIZON_S == 3.0
        ),
        "universe_exact_12": len(ROUTER.SERIES) == 12,
        "commodities_present": set(ROUTER.COMMODITY_SERIES)
        == {"KXGOLD15M", "KXSILVER15M", "KXWTI15M"},
        "runtime_live_is_final": RUNTIME.LIVE is LIVE,
        "parent_live_is_final": P.LIVE is LIVE,
        "deployment_live_is_final": V1.LIVE is LIVE,
        "parent_recorder_v7_preserved": P.REC is REC,
        "hardened_preflight_installed": V1._read_only_preflight
        is _hardened_read_only_preflight,
        "supervisor_self_control_preserved": P._fresh_generation_preflight
        is V11._fresh_generation_preflight,
        "recorder_dead_receipt_compat_installed": P._stop_external_recorder
        is _stop_external_recorder_compat,
        "small_balance_delta_passes": b_ok.get("ok") is True,
        "cent_vs_dollar_unit_error_fails": b_bad.get("ok") is False,
        "fee_preflight_12_series_required": True,
        "q1_same_head_promotion_required": True,
        "no_auto_scale": True,
        "no_auto_transfer": True,
        "orders_sent": False,
        "transfers_sent": False,
        "api_called": False,
    }
    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"orders_sent", "transfers_sent", "api_called"}
    )
    out = {
        "deploy_version": DEPLOY_VERSION,
        "deploy_patch_version": PATCH_VERSION,
        "module_name": MODULE_NAME,
        "live_version": LIVE.LIVE_VERSION,
        "live_patch_version": LIVE.PATCH_VERSION,
        "recorder_version": REC.STUDY_VERSION,
        "router_version": ROUTER.ROUTER_VERSION,
        "promotion_path": str(PROMOTION_PATH),
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 168)
        print("TAIL25 MULTI12 V1.2 FINAL SOURCE AUDIT — NO API / NO ORDERS")
        print("=" * 168)
        for k, v in out.items():
            print(f"{k:108s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 V1.2 final static audit failed: {out}")
    return out


def q1_smoke_preflight(*, show=True):
    _install_patch()
    return _hardened_read_only_preflight(
        q=V1.Q1_Q,
        hours=V1.Q1_HOURS,
        max_loss=V1.Q1_MAX_LOSS_USD,
        min_equity=V1.Q1_MIN_EQUITY_USD,
        min_per_shard=V1.Q1_MIN_COLLATERAL_PER_USED_SHARD_USD,
        probe_private_ws=True,
        show=show,
        run_static=True,
    )


def q10_preflight(*, show=True):
    _install_patch()
    return _hardened_read_only_preflight(
        q=V1.Q10_Q,
        hours=V1.Q10_HOURS,
        max_loss=V1.Q10_MAX_LOSS_USD,
        min_equity=V1.Q10_MIN_EQUITY_USD,
        min_per_shard=V1.Q10_MIN_COLLATERAL_PER_USED_SHARD_USD,
        probe_private_ws=True,
        show=show,
        run_static=True,
    )


def rotation_promotion_status(*, show=True):
    _install_patch()
    return V1.rotation_promotion_status(show=show)


def start_q1_one_window_smoke(*, arm_phrase=None):
    _install_patch()
    if str(arm_phrase) != Q1_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q1_ARM!r} exactly."
        )
    return V1.start_q1_one_window_smoke(arm_phrase=Q1_ARM)


def start_q10_12h(*, arm_phrase=None):
    _install_patch()
    if str(arm_phrase) != Q10_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q10_ARM!r} exactly."
        )
    return V1.start_q10_12h(arm_phrase=Q10_ARM)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    out = V1.live_status(show=show, tail_lines=tail_lines)
    if isinstance(out, dict):
        out["deploy_patch_version"] = PATCH_VERSION
        out["live_patch_version"] = LIVE.PATCH_VERSION
        out["module_name"] = MODULE_NAME
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    if str(arm_phrase) != KILL_ARM:
        raise RuntimeError(f"Pass arm_phrase={KILL_ARM!r} exactly.")
    return V1.kill_and_flatten_live(
        arm_phrase=KILL_ARM,
        wait_s=float(wait_s),
    )


def _main():
    _install_patch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-generation")
    ap.add_argument("--run-supervisor")
    ap.add_argument("--run-guardian")
    ap.add_argument("--supervisor-pid", type=int)
    ap.add_argument("--config")
    args = ap.parse_args()

    if args.run_generation:
        if not args.config:
            raise RuntimeError("--config is required with --run-generation")
        return V1._run_generation(
            Path(args.run_generation),
            Path(args.config),
        )

    if args.run_supervisor:
        if not args.config:
            raise RuntimeError("--config is required with --run-supervisor")
        _install_patch()
        return P._run_supervisor(
            Path(args.run_supervisor),
            Path(args.config),
        )

    if args.run_guardian:
        if not args.supervisor_pid:
            raise RuntimeError("--supervisor-pid is required with --run-guardian")
        _install_patch()
        return P._guardian_loop(
            Path(args.run_guardian),
            int(args.supervisor_pid),
        )

    return static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "PATCH_VERSION",
    "MODULE_NAME",
    "Q1_ARM",
    "Q10_ARM",
    "KILL_ARM",
    "Q1_Q",
    "Q10_Q",
    "Q10_HOURS",
    "Q10_MAX_LOSS_USD",
    "Q10_MIN_EQUITY_USD",
    "PROMOTION_PATH",
    "static_self_check",
    "q1_smoke_preflight",
    "q10_preflight",
    "rotation_promotion_status",
    "start_q1_one_window_smoke",
    "start_q10_12h",
    "live_status",
    "kill_and_flatten_live",
]
