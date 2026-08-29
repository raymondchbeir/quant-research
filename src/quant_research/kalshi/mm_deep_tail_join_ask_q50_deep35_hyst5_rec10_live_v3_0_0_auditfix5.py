from __future__ import annotations

"""Dashboard-compatible V3.0.0 deployment shim for Deep35 AUDITFIX5.

AUDITFIX5 preserves the public V3.0.0/V1.13 strings and frozen strategy economics.
It adds two operational corrections observed in live validation:

1) V1.13.5 applies authoritative cancel reconciliation to the normal 2s force-flat
   cleanup path as well as GLOBAL_SHUTDOWN/M12.
2) Every NEW parent session derives its fixed $20 risk baseline from the current
   flat-account shard balance breakdown returned during that parent's preflight.
   This prevents an earlier parent's equity baseline from being displayed/used as
   the start of a later independently-armed parent session.

The shard-derived baseline is used only after preflight has independently proved
zero positions and zero resting orders.  Therefore cash across exchange shards is
exactly the flat account equity for this risk-baseline purpose.

Importing this module performs no API calls, orders, cancels, or transfers.
"""

import argparse
import math
from pathlib import Path

from . import mm_deep_tail_join_ask_q50_deep35_hyst5_rec10_live_v3_0_0 as V300
from . import mm_deep_tail_join_ask_live_v1_13_5_cleanup_cancel as LIVE


# Capture the original V3.0.0 preflight before this shim installs its correction.
_V300_ORIGINAL_Q50_PREFLIGHT = V300.q50_preflight

DEPLOY_VERSION = V300.DEPLOY_VERSION
PATCH_VERSION = "V3_0_0_AUDITFIX5"
MODULE_NAME = (
    "quant_research.kalshi."
    "mm_deep_tail_join_ask_q50_deep35_hyst5_rec10_live_v3_0_0_auditfix5"
)

Q50_ARM = "LIVE_DEEP35_HYST5_Q50_REC10_12H_V300_AUDITFIX5"
KILL_ARM = V300.KILL_ARM
SHARD_FUND_ARM = V300.SHARD_FUND_ARM

RUNTIME = V300.RUNTIME
P = V300.P
H = V300.H
V2963 = V300.V2963
V28 = V300.V28
V288 = V300.V288
V111 = V300.V111
V1 = V300.V1
B = V300.B
Q1 = V300.Q1
C = V300.C

Q50_Q = V300.Q50_Q
Q50_HOURS = V300.Q50_HOURS
Q50_MAX_LOSS_USD = V300.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = V300.Q50_MIN_EQUITY_USD

Q100_Q = Q50_Q
Q100_HOURS = Q50_HOURS
Q100_MAX_LOSS_USD = Q50_MAX_LOSS_USD
Q100_MIN_EQUITY_USD = Q50_MIN_EQUITY_USD

ENTRY_START_S = LIVE.ENTRY_START_S
M12_S = LIVE.M12_S
LABEL_TAIL_END_S = V300.LABEL_TAIL_END_S
DEPTH = LIVE.DEPTH
HYSTERESIS = LIVE.HYSTERESIS
RECOVERY_EDGE = LIVE.RECOVERY_EDGE
RECOVERY_HORIZON_S = LIVE.RECOVERY_HORIZON_S
SPREAD_WINDOW_S = LIVE.SPREAD_WINDOW_S
NORMAL_TOL = LIVE.NORMAL_TOL
MIN_NORMAL_OBS = LIVE.MIN_NORMAL_OBS
EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = LIVE.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

M12_HARD_RECYCLE_GRACE_S = V300.M12_HARD_RECYCLE_GRACE_S
HARD_RECYCLE_RECEIPT_FILE = V300.HARD_RECYCLE_RECEIPT_FILE
GUARDIAN_POST_M12_EXIT_TIMEOUT_S = V300.GUARDIAN_POST_M12_EXIT_TIMEOUT_S
GENERATION_RSS_WARNING_MB = V300.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = V300.GENERATION_RSS_HARD_LIMIT_MB
RSS_HARD_STOP_DISABLED = V300.RSS_HARD_STOP_DISABLED

LIVE_EXCHANGE_INDEX = V300.LIVE_EXCHANGE_INDEX
SOURCE_EXCHANGE_INDEX = V300.SOURCE_EXCHANGE_INDEX
SHARD2_MIN_COLLATERAL_USD = V300.SHARD2_MIN_COLLATERAL_USD
CRYPTO_SERIES = V300.CRYPTO_SERIES
CENTICENTS_PER_DOLLAR = V300.CENTICENTS_PER_DOLLAR

DISCOVERY_ATTEMPTS_PER_SERIES = V300.DISCOVERY_ATTEMPTS_PER_SERIES
DISCOVERY_RETRY_BASE_S = V300.DISCOVERY_RETRY_BASE_S
DISCOVERY_CACHE_TTL_S = V300.DISCOVERY_CACHE_TTL_S

ROTATION_CHECKPOINT_FILE = V300.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V300.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V300.SESSION_RISK_BASELINE_FILE

SUPERVISOR_LOG_FILE = "supervisor_v3_0_0_deep35_auditfix5.log"
GUARDIAN_LOG_FILE = "guardian_v3_0_0_deep35_auditfix5.log"
STARTUP_TIMEOUT_S = V300.STARTUP_TIMEOUT_S


def _finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _flat_shard_equity(report):
    """Return exact flat-account cash from all primary exchange shard balances."""
    report = report or {}
    shard = report.get("crypto_shard_preflight") or {}
    if shard.get("nonzero_positions"):
        raise RuntimeError("Risk-baseline correction requires authoritative flat positions")
    if shard.get("resting_orders"):
        raise RuntimeError("Risk-baseline correction requires zero resting orders")
    balances = shard.get("balances") or {}
    breakdown = balances.get("breakdown_usd") or {}
    vals = []
    for value in breakdown.values():
        z = _finite(value)
        if z is not None:
            vals.append(z)
    if not vals:
        raise RuntimeError(f"No usable shard balance breakdown in preflight: {balances!r}")
    total = float(sum(vals))
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError(f"Invalid flat shard equity: {total!r}")
    return total


def _corrected_parent_preflight(*, show=True):
    """Run inherited full preflight, then pin this NEW parent's baseline to flat cash."""
    # The original function already performs shard/account/private-WS/fee checks.
    out = dict(_V300_ORIGINAL_Q50_PREFLIGHT(show=show) or {})
    exact_flat_equity = _flat_shard_equity(out)

    account = dict(out.get("account") or {})
    inherited_equity = _finite(account.get("equity_usd"))

    account["equity_usd"] = float(exact_flat_equity)
    account["cash_balance_usd"] = float(exact_flat_equity)
    account["portfolio_value_usd"] = 0.0
    out["account"] = account
    out["kill_equity_usd"] = float(exact_flat_equity - Q50_MAX_LOSS_USD)

    out["risk_baseline_source"] = "CURRENT_FLAT_SHARD_BALANCE_BREAKDOWN"
    out["risk_baseline_equity_usd"] = float(exact_flat_equity)
    out["risk_baseline_kill_equity_usd"] = float(exact_flat_equity - Q50_MAX_LOSS_USD)
    out["inherited_preflight_equity_usd"] = inherited_equity
    out["inherited_vs_shard_equity_delta_usd"] = (
        None if inherited_equity is None else float(inherited_equity - exact_flat_equity)
    )
    out["risk_baseline_corrected_for_new_parent"] = True

    if show:
        print("=" * 176)
        print("AUDITFIX5 NEW-PARENT RISK BASELINE")
        print("=" * 176)
        print(f"Flat shard equity:          ${exact_flat_equity:.4f}")
        print(f"New-parent $20 floor:       ${exact_flat_equity - Q50_MAX_LOSS_USD:.4f}")
        if inherited_equity is not None:
            print(f"Inherited preflight equity: ${inherited_equity:.4f}")
            print(f"Difference corrected:       ${inherited_equity - exact_flat_equity:+.4f}")
        print("Baseline source:            CURRENT flat shard breakdown")
        print("ORDERS SENT:                NO")

    return out


def _install_patch():
    """Bind V1.13.5 and the new-parent flat-shard baseline correction."""
    V300.LIVE = LIVE
    V300.MODULE_NAME = MODULE_NAME
    V300.Q50_ARM = Q50_ARM
    V300.SUPERVISOR_LOG_FILE = SUPERVISOR_LOG_FILE
    V300.GUARDIAN_LOG_FILE = GUARDIAN_LOG_FILE

    V300.ENTRY_START_S = ENTRY_START_S
    V300.M12_S = M12_S
    V300.DEPTH = DEPTH
    V300.HYSTERESIS = HYSTERESIS
    V300.RECOVERY_EDGE = RECOVERY_EDGE
    V300.RECOVERY_HORIZON_S = RECOVERY_HORIZON_S
    V300.SPREAD_WINDOW_S = SPREAD_WINDOW_S
    V300.NORMAL_TOL = NORMAL_TOL
    V300.MIN_NORMAL_OBS = MIN_NORMAL_OBS
    V300.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

    V300._install_patch()

    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.LIVE = LIVE
    P.LIVE = LIVE
    V2963.LIVE = LIVE

    # V300._launch_supervised resolves its own module-global q50_preflight at call
    # time. Rebind it so the supervisor snapshot always receives this parent’s
    # current flat shard equity rather than any stale inherited equity field.
    V300.q50_preflight = _corrected_parent_preflight


def static_self_check(*, show=True):
    _install_patch()
    base = V300.static_self_check(show=False)
    live = LIVE.static_self_check(show=False)

    synthetic = {
        "crypto_shard_preflight": {
            "nonzero_positions": [],
            "resting_orders": [],
            "balances": {"breakdown_usd": {0: 197.0, 2: 200.0}},
        }
    }

    checks = {
        "v300_operations_static_ok": base.get("ok") is True,
        "cleanup_cancel_static_ok": live.get("ok") is True,
        "dashboard_deploy_version_unchanged": DEPLOY_VERSION
        == "MM_DEEP_TAIL_JOIN_ASK_Q50_DEEP35_HYST5_REC10_V3_0_0",
        "dashboard_live_version_unchanged": LIVE.LIVE_VERSION
        == "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_13_DEEP35_HYST5_REC10",
        "deploy_patch_version_exact": PATCH_VERSION == "V3_0_0_AUDITFIX5",
        "engine_patch_version_exact": getattr(LIVE, "PATCH_VERSION", None)
        == "DEEP35_CLEANUP_CANCEL_V1_13_5",
        "detached_module_is_auditfix5": RUNTIME.MODULE_NAME == MODULE_NAME,
        "runtime_live_is_v135": RUNTIME.LIVE is LIVE,
        "parent_live_is_v135": P.LIVE is LIVE,
        "guardian_live_is_v135": V2963.LIVE is LIVE,
        "parent_preflight_rebound": V300.q50_preflight is _corrected_parent_preflight,
        "flat_shard_sum_static": abs(_flat_shard_equity(synthetic) - 397.0) < 1e-12,
        "q50_exact": Q50_Q == 50.0,
        "runtime_12h_exact": Q50_HOURS == 12.0,
        "loss_stop_20_exact": Q50_MAX_LOSS_USD == 20.0,
        "min_equity_125_exact": Q50_MIN_EQUITY_USD == 125.0,
        "depth_35c_unchanged": DEPTH == 0.35,
        "hyst_5c_unchanged": HYSTERESIS == 0.05,
        "rec10_unchanged": RECOVERY_EDGE == 0.10,
        "force_flat_2s_unchanged": RECOVERY_HORIZON_S == 2.0,
        "entry_m0_unchanged": ENTRY_START_S == 0.0,
        "m12_720_unchanged": M12_S == 720.0,
        "orders_sent": False,
        "transfers_sent": False,
    }

    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"orders_sent", "transfers_sent"}
    )

    out = {
        "deploy_version": DEPLOY_VERSION,
        "deploy_patch_version": PATCH_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "live_patch_version": LIVE.PATCH_VERSION,
        "module_name": MODULE_NAME,
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 180)
        print("V3.0.0 DEEP35 AUDITFIX5 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 180)
        for k, v in out.items():
            print(f"{k:116s}: {v}")

    if not ok:
        raise RuntimeError(f"V3.0.0 AUDITFIX5 static self-check failed: {out}")

    return out


def discover_current_crypto_markets(*args, **kwargs):
    _install_patch()
    return V300.discover_current_crypto_markets(*args, **kwargs)


def get_shard_balances(*args, **kwargs):
    _install_patch()
    return V300.get_shard_balances(*args, **kwargs)


def ensure_crypto_shard_funded(*args, **kwargs):
    _install_patch()
    return V300.ensure_crypto_shard_funded(*args, **kwargs)


def crypto_shard_preflight(*args, **kwargs):
    _install_patch()
    return V300.crypto_shard_preflight(*args, **kwargs)


def q50_preflight(*, show=True):
    _install_patch()
    static_self_check(show=show)
    out = _corrected_parent_preflight(show=show)
    _install_patch()
    if isinstance(out, dict):
        out["deploy_patch_version"] = PATCH_VERSION
        out["live_patch_version"] = LIVE.PATCH_VERSION
    return out


def start_q50_12h_smoke(*, arm_phrase=None):
    _install_patch()
    return V300.start_q50_12h_smoke(arm_phrase=arm_phrase)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    return V300.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    return V300.kill_and_flatten_live(
        arm_phrase=arm_phrase,
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
        return RUNTIME._run_generation(Path(args.run_generation), Path(args.config))

    if args.run_supervisor:
        if not args.config:
            raise RuntimeError("--config is required with --run-supervisor")
        return RUNTIME._run_supervisor(Path(args.run_supervisor), Path(args.config))

    if args.run_guardian:
        if not args.supervisor_pid:
            raise RuntimeError("--supervisor-pid is required with --run-guardian")
        return RUNTIME._run_guardian(Path(args.run_guardian), int(args.supervisor_pid))

    return static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "PATCH_VERSION",
    "MODULE_NAME",
    "Q50_ARM",
    "KILL_ARM",
    "SHARD_FUND_ARM",
    "Q50_Q",
    "Q50_HOURS",
    "Q50_MAX_LOSS_USD",
    "Q50_MIN_EQUITY_USD",
    "ENTRY_START_S",
    "M12_S",
    "LABEL_TAIL_END_S",
    "DEPTH",
    "HYSTERESIS",
    "RECOVERY_EDGE",
    "RECOVERY_HORIZON_S",
    "SPREAD_WINDOW_S",
    "NORMAL_TOL",
    "MIN_NORMAL_OBS",
    "EXPECTED_EXIT_EFFECTIVE_LATENCY_MS",
    "LIVE_EXCHANGE_INDEX",
    "SOURCE_EXCHANGE_INDEX",
    "SHARD2_MIN_COLLATERAL_USD",
    "CRYPTO_SERIES",
    "ROTATION_CHECKPOINT_FILE",
    "GENERATION_BOOTSTRAP_FILE",
    "SESSION_RISK_BASELINE_FILE",
    "discover_current_crypto_markets",
    "get_shard_balances",
    "ensure_crypto_shard_funded",
    "crypto_shard_preflight",
    "static_self_check",
    "q50_preflight",
    "start_q50_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
