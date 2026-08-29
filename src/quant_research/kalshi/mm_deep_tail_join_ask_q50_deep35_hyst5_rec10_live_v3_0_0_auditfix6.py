from __future__ import annotations

"""Dashboard-compatible V3.0.0 deployment shim for Deep35 AUDITFIX6.

AUDITFIX6 preserves AUDITFIX5's parent preflight/risk-baseline correction and swaps
only the live child engine to V1.13.6, where a documented successful cleanup cancel
receipt retires the exact local track without requiring an immediately-consistent
follow-up REST mirror.

Public V3.0.0/V1.13 strings and frozen strategy economics remain unchanged.
Importing this module performs no API calls, orders, cancels, or transfers.
"""

import argparse
from pathlib import Path

from . import mm_deep_tail_join_ask_q50_deep35_hyst5_rec10_live_v3_0_0_auditfix5 as BASE
from . import mm_deep_tail_join_ask_live_v1_13_6_cleanup_receipt_retire as LIVE


DEPLOY_VERSION = BASE.DEPLOY_VERSION
PATCH_VERSION = "V3_0_0_AUDITFIX6"
MODULE_NAME = (
    "quant_research.kalshi."
    "mm_deep_tail_join_ask_q50_deep35_hyst5_rec10_live_v3_0_0_auditfix6"
)

Q50_ARM = "LIVE_DEEP35_HYST5_Q50_REC10_12H_V300_AUDITFIX6"
KILL_ARM = BASE.KILL_ARM
SHARD_FUND_ARM = BASE.SHARD_FUND_ARM

V300 = BASE.V300
RUNTIME = BASE.RUNTIME
P = BASE.P
H = BASE.H
V2963 = BASE.V2963
V28 = BASE.V28
V288 = BASE.V288
V111 = BASE.V111
V1 = BASE.V1
B = BASE.B
Q1 = BASE.Q1
C = BASE.C

Q50_Q = BASE.Q50_Q
Q50_HOURS = BASE.Q50_HOURS
Q50_MAX_LOSS_USD = BASE.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = BASE.Q50_MIN_EQUITY_USD

ENTRY_START_S = LIVE.ENTRY_START_S
M12_S = LIVE.M12_S
LABEL_TAIL_END_S = BASE.LABEL_TAIL_END_S
DEPTH = LIVE.DEPTH
HYSTERESIS = LIVE.HYSTERESIS
RECOVERY_EDGE = LIVE.RECOVERY_EDGE
RECOVERY_HORIZON_S = LIVE.RECOVERY_HORIZON_S
SPREAD_WINDOW_S = LIVE.SPREAD_WINDOW_S
NORMAL_TOL = LIVE.NORMAL_TOL
MIN_NORMAL_OBS = LIVE.MIN_NORMAL_OBS
EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = LIVE.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

LIVE_EXCHANGE_INDEX = BASE.LIVE_EXCHANGE_INDEX
SOURCE_EXCHANGE_INDEX = BASE.SOURCE_EXCHANGE_INDEX
SHARD2_MIN_COLLATERAL_USD = BASE.SHARD2_MIN_COLLATERAL_USD
CRYPTO_SERIES = BASE.CRYPTO_SERIES

ROTATION_CHECKPOINT_FILE = BASE.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = BASE.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = BASE.SESSION_RISK_BASELINE_FILE

SUPERVISOR_LOG_FILE = "supervisor_v3_0_0_deep35_auditfix6.log"
GUARDIAN_LOG_FILE = "guardian_v3_0_0_deep35_auditfix6.log"


def _install_patch():
    # Make AUDITFIX5's existing operations/risk-baseline shim install V1.13.6
    # instead of V1.13.5, then publish this detached module identity.
    BASE.LIVE = LIVE
    BASE.MODULE_NAME = MODULE_NAME
    BASE.Q50_ARM = Q50_ARM
    BASE.SUPERVISOR_LOG_FILE = SUPERVISOR_LOG_FILE
    BASE.GUARDIAN_LOG_FILE = GUARDIAN_LOG_FILE

    BASE.ENTRY_START_S = ENTRY_START_S
    BASE.M12_S = M12_S
    BASE.DEPTH = DEPTH
    BASE.HYSTERESIS = HYSTERESIS
    BASE.RECOVERY_EDGE = RECOVERY_EDGE
    BASE.RECOVERY_HORIZON_S = RECOVERY_HORIZON_S
    BASE.SPREAD_WINDOW_S = SPREAD_WINDOW_S
    BASE.NORMAL_TOL = NORMAL_TOL
    BASE.MIN_NORMAL_OBS = MIN_NORMAL_OBS
    BASE.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

    # The BASE installer dynamically reads the BASE globals above.
    BASE._install_patch()

    V300.LIVE = LIVE
    V300.MODULE_NAME = MODULE_NAME
    V300.Q50_ARM = Q50_ARM
    V300.SUPERVISOR_LOG_FILE = SUPERVISOR_LOG_FILE
    V300.GUARDIAN_LOG_FILE = GUARDIAN_LOG_FILE
    V300.q50_preflight = BASE._corrected_parent_preflight

    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.LIVE = LIVE
    P.LIVE = LIVE
    V2963.LIVE = LIVE


def static_self_check(*, show=True):
    _install_patch()
    live = LIVE.static_self_check(show=False)
    checks = {
        "live_v136_static_ok": live.get("ok") is True,
        "dashboard_deploy_version_unchanged": DEPLOY_VERSION
        == "MM_DEEP_TAIL_JOIN_ASK_Q50_DEEP35_HYST5_REC10_V3_0_0",
        "dashboard_live_version_unchanged": LIVE.LIVE_VERSION
        == "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_13_DEEP35_HYST5_REC10",
        "deploy_patch_version_exact": PATCH_VERSION == "V3_0_0_AUDITFIX6",
        "engine_patch_version_exact": LIVE.PATCH_VERSION
        == "DEEP35_CLEANUP_RECEIPT_RETIRE_V1_13_6",
        "runtime_live_is_v136": RUNTIME.LIVE is LIVE,
        "parent_live_is_v136": P.LIVE is LIVE,
        "guardian_live_is_v136": V2963.LIVE is LIVE,
        "detached_module_is_auditfix6": RUNTIME.MODULE_NAME == MODULE_NAME,
        "corrected_parent_preflight_preserved": V300.q50_preflight is BASE._corrected_parent_preflight,
        "q50_exact": Q50_Q == 50.0,
        "runtime_12h_exact": Q50_HOURS == 12.0,
        "loss_stop_20_exact": Q50_MAX_LOSS_USD == 20.0,
        "min_equity_125_exact": Q50_MIN_EQUITY_USD == 125.0,
        "depth_35c_unchanged": DEPTH == 0.35,
        "hyst_5c_unchanged": HYSTERESIS == 0.05,
        "rec10_unchanged": RECOVERY_EDGE == 0.10,
        "force_2s_unchanged": RECOVERY_HORIZON_S == 2.0,
        "entry_m0_unchanged": ENTRY_START_S == 0.0,
        "m12_720_unchanged": M12_S == 720.0,
        "orders_sent": False,
        "transfers_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k not in {"orders_sent", "transfers_sent"})
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
        print("=" * 184)
        print("V3.0.0 DEEP35 AUDITFIX6 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 184)
        for k, v in out.items():
            print(f"{k:120s}: {v}")
    if not ok:
        raise RuntimeError(f"V3.0.0 AUDITFIX6 static self-check failed: {out}")
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
    out = BASE._corrected_parent_preflight(show=show)
    _install_patch()
    if isinstance(out, dict):
        out["deploy_patch_version"] = PATCH_VERSION
        out["live_patch_version"] = LIVE.PATCH_VERSION
    return out


def start_q50_12h_smoke(*, arm_phrase=None):
    _install_patch()
    if arm_phrase != Q50_ARM:
        raise RuntimeError(f"Exact AUDITFIX6 arm phrase required: {Q50_ARM!r}")
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
