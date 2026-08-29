from __future__ import annotations

"""Dashboard-compatible V3.0.0 deployment shim for Deep35 AUDITFIX4.

Public V3.0.0 / V1.13 version strings are intentionally preserved so the existing
notebook dashboard continues to work unchanged.  Exact source commit plus patch
version fields distinguish this deployment from prior runs.

AUDITFIX4 keeps AUDITFIX1/2/3 identity and lifecycle protections, adds actual
force-flat fill attribution into the Deep35 strategy PnL ledger, and reconciles
GLOBAL_SHUTDOWN cancel retirement against authoritative REST before deciding a
still-present local stable key is fatal.

Importing this module performs no API calls, orders, cancels, or transfers.
"""

import argparse
from pathlib import Path

from . import mm_deep_tail_join_ask_q50_deep35_hyst5_rec10_live_v3_0_0 as V300
from . import mm_deep_tail_join_ask_live_v1_13_4_force_pnl_shutdown_cancel as LIVE


DEPLOY_VERSION = V300.DEPLOY_VERSION
PATCH_VERSION = "V3_0_0_AUDITFIX4"
MODULE_NAME = (
    "quant_research.kalshi."
    "mm_deep_tail_join_ask_q50_deep35_hyst5_rec10_live_v3_0_0_auditfix4"
)

Q50_ARM = "LIVE_DEEP35_HYST5_Q50_REC10_12H_V300_AUDITFIX4"
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

SUPERVISOR_LOG_FILE = "supervisor_v3_0_0_deep35_auditfix4.log"
GUARDIAN_LOG_FILE = "guardian_v3_0_0_deep35_auditfix4.log"
STARTUP_TIMEOUT_S = V300.STARTUP_TIMEOUT_S


def _install_patch():
    """Rebind V3.0.0 operations to V1.13.4 force-PnL/shutdown-cancel safety."""
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


def static_self_check(*, show=True):
    _install_patch()
    base = V300.static_self_check(show=False)
    live = LIVE.static_self_check(show=False)

    checks = {
        "v300_operations_static_ok": base.get("ok") is True,
        "force_pnl_shutdown_cancel_static_ok": live.get("ok") is True,
        "dashboard_deploy_version_unchanged": DEPLOY_VERSION
        == "MM_DEEP_TAIL_JOIN_ASK_Q50_DEEP35_HYST5_REC10_V3_0_0",
        "dashboard_live_version_unchanged": LIVE.LIVE_VERSION
        == "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_13_DEEP35_HYST5_REC10",
        "deploy_patch_version_exact": PATCH_VERSION == "V3_0_0_AUDITFIX4",
        "engine_patch_version_exact": getattr(LIVE, "PATCH_VERSION", None)
        == "DEEP35_FORCE_PNL_SHUTDOWN_CANCEL_V1_13_4",
        "detached_module_is_auditfix4": RUNTIME.MODULE_NAME == MODULE_NAME,
        "runtime_live_is_v134": RUNTIME.LIVE is LIVE,
        "parent_live_is_v134": P.LIVE is LIVE,
        "guardian_live_is_v134": V2963.LIVE is LIVE,
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
        print("=" * 176)
        print("V3.0.0 DEEP35 AUDITFIX4 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 176)
        for k, v in out.items():
            print(f"{k:112s}: {v}")

    if not ok:
        raise RuntimeError(f"V3.0.0 AUDITFIX4 static self-check failed: {out}")

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
    out = V300.q50_preflight(show=show)
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
