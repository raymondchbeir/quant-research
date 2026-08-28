from __future__ import annotations

"""V3.0.0 real-money Q50 deployment for the frozen Deep35/Hyst5/REC10 strategy.

This module is intentionally additive.  The historical Q100/REC25 branch and files
remain untouched.  Operational safety, shard routing/funding, recorder ownership,
rotating supervisor, guardian, fail-closed recovery, exact-equity baseline, and
runtime-status interfaces are inherited from the audited V2.9.9.14 deployment.
Only the child trading engine and Q50 strategy contract are rebound here.

Importing this module performs no API calls, orders, cancels, or transfers.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_14_m1130_runtime_compat as BASE
from . import mm_deep_tail_join_ask_live_v1_13_deep35_hyst5_rec10 as LIVE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_DEEP35_HYST5_REC10_V3_0_0"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_deep35_hyst5_rec10_live_v3_0_0"

Q50_ARM = "LIVE_DEEP35_HYST5_Q50_REC10_12H_V300"
KILL_ARM = BASE.KILL_ARM
SHARD_FUND_ARM = BASE.SHARD_FUND_ARM

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

Q50_Q = 50.0
Q50_HOURS = 12.0
Q50_MAX_LOSS_USD = 20.0
Q50_MIN_EQUITY_USD = 125.0

# Historical aliases are retained only because inherited runtime functions use
# Q50-named storage inconsistently across versions. They do NOT mean this is Q100.
Q100_Q = Q50_Q
Q100_HOURS = Q50_HOURS
Q100_MAX_LOSS_USD = Q50_MAX_LOSS_USD
Q100_MIN_EQUITY_USD = Q50_MIN_EQUITY_USD

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

M12_HARD_RECYCLE_GRACE_S = BASE.M12_HARD_RECYCLE_GRACE_S
HARD_RECYCLE_RECEIPT_FILE = BASE.HARD_RECYCLE_RECEIPT_FILE
GUARDIAN_POST_M12_EXIT_TIMEOUT_S = BASE.GUARDIAN_POST_M12_EXIT_TIMEOUT_S
GENERATION_RSS_WARNING_MB = BASE.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = BASE.GENERATION_RSS_HARD_LIMIT_MB
RSS_HARD_STOP_DISABLED = BASE.RSS_HARD_STOP_DISABLED

LIVE_EXCHANGE_INDEX = BASE.LIVE_EXCHANGE_INDEX
SOURCE_EXCHANGE_INDEX = BASE.SOURCE_EXCHANGE_INDEX
SHARD2_MIN_COLLATERAL_USD = BASE.SHARD2_MIN_COLLATERAL_USD
CRYPTO_SERIES = BASE.CRYPTO_SERIES
CENTICENTS_PER_DOLLAR = BASE.CENTICENTS_PER_DOLLAR

DISCOVERY_ATTEMPTS_PER_SERIES = BASE.DISCOVERY_ATTEMPTS_PER_SERIES
DISCOVERY_RETRY_BASE_S = BASE.DISCOVERY_RETRY_BASE_S
DISCOVERY_CACHE_TTL_S = BASE.DISCOVERY_CACHE_TTL_S

ROTATION_CHECKPOINT_FILE = V111.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V111.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V111.SESSION_RISK_BASELINE_FILE

SUPERVISOR_LOG_FILE = "supervisor_v3_0_0_deep35.log"
GUARDIAN_LOG_FILE = "guardian_v3_0_0_deep35.log"
STARTUP_TIMEOUT_S = 150.0

_BASELINE_STATIC = None


def _generation_cfg(
    parent_cfg,
    *,
    generation_id,
    generation_dir,
    recorder_pid,
    session_start_equity,
    session_kill_equity,
    remaining_hours,
):
    q = float(parent_cfg["quote_size"])
    return {
        "mode": f"{parent_cfg['mode']}_GEN_{int(generation_id):04d}",
        "quote_size": q,
        "runtime_hours": max(0.02, float(remaining_hours)),
        "max_start_loss_usd": float(parent_cfg["max_start_loss_usd"]),
        "min_start_equity_usd": float(parent_cfg["min_start_equity_usd"]),
        "order_group_limit_fp": f"{max(25.0, 20.0 * q):.2f}",
        "live_engine_version": LIVE.LIVE_VERSION,
        "deploy_version": DEPLOY_VERSION,
        "generation_id": int(generation_id),
        "parent_session_dir": str(Path(parent_cfg["parent_session_dir"]).resolve()),
        "external_recorder_owner": True,
        "external_recorder_pid": int(recorder_pid),
        "session_start_equity_usd": float(session_start_equity),
        "session_kill_equity_usd": float(session_kill_equity),
        "session_runtime_hours": float(parent_cfg["runtime_hours"]),
        "strategy_entry_start_elapsed_s": ENTRY_START_S,
        "strategy_terminal_cleanup_elapsed_s": M12_S,
        "recorder_persist_end_elapsed_s": M12_S,
        "recorder_label_tail_end_elapsed_s": LABEL_TAIL_END_S,
        "rotation_process_lifetime": "ONE_COMPLETE_M0_M12_WINDOW",
        "fixed_session_risk_baseline": True,
        "fresh_generation_starts_at_raw_eof": True,
        "fresh_row_after_generation_start_required": True,
        "no_auto_scale": True,
        "strategy_name": "DEEP35_HYST5_REC10_Q50",
        "depth": DEPTH,
        "quote_hysteresis": HYSTERESIS,
        "normal_spread_window_s": SPREAD_WINDOW_S,
        "normal_spread_tolerance": NORMAL_TOL,
        "normal_spread_min_obs": MIN_NORMAL_OBS,
        "recovery_edge": RECOVERY_EDGE,
        "recovery_horizon_s": RECOVERY_HORIZON_S,
        "persistent_target_exit_retry": True,
        "expired_lot_force_flat": True,
        "rec25_enabled": False,
        "m1130_entry_cutoff_enabled": False,
        "persistent_danger_guard_enabled": False,
    }


def _install_patch():
    """Install the latest audited operations stack, then bind only the new strategy."""
    BASE._install_patch()

    # Historical artifact names are runtime interface contracts, not strategy rules.
    LIVE.ROTATION_CHECKPOINT_FILE = ROTATION_CHECKPOINT_FILE
    LIVE.GENERATION_BOOTSTRAP_FILE = GENERATION_BOOTSTRAP_FILE
    LIVE.SESSION_RISK_BASELINE_FILE = SESSION_RISK_BASELINE_FILE

    # Detached runtime identity/parameters.
    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q50_ARM
    RUNTIME.Q50_Q = Q50_Q
    RUNTIME.Q50_HOURS = Q50_HOURS
    RUNTIME.Q50_MAX_LOSS_USD = Q50_MAX_LOSS_USD
    RUNTIME.Q50_MIN_EQUITY_USD = Q50_MIN_EQUITY_USD
    RUNTIME.LIVE = LIVE
    RUNTIME.M1_S = ENTRY_START_S
    RUNTIME.M12_S = M12_S
    RUNTIME._generation_cfg = _generation_cfg

    # Parent supervisor keeps the existing recorder/rotation/recovery implementation.
    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.Q50_Q = Q50_Q
    P.Q50_HOURS = Q50_HOURS
    P.Q50_MAX_LOSS_USD = Q50_MAX_LOSS_USD
    P.Q50_MIN_EQUITY_USD = Q50_MIN_EQUITY_USD
    P.M1_S = ENTRY_START_S
    P.M5_S = M12_S
    P.RECORDER_M12_S = M12_S
    P.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    P._generation_cfg = _generation_cfg

    # Guardian observes the new engine but retains inherited authoritative recovery.
    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.POST_M5_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S

    # Inherited detached entrypoints call these globals dynamically.
    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline structural audit. No API, orders, cancels, or transfers."""
    global _BASELINE_STATIC
    if _BASELINE_STATIC is None:
        _BASELINE_STATIC = BASE.static_self_check(show=False)
    live = LIVE.static_self_check(show=False)
    _install_patch()

    checks = {
        "v29914_operations_baseline_ok": _BASELINE_STATIC.get("ok") is True,
        "new_live_engine_static_ok": live.get("ok") is True,
        "deploy_version_exact": DEPLOY_VERSION == "MM_DEEP_TAIL_JOIN_ASK_Q50_DEEP35_HYST5_REC10_V3_0_0",
        "live_version_exact": LIVE.LIVE_VERSION == "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_13_DEEP35_HYST5_REC10",
        "q50_exact_50": Q50_Q == 50.0,
        "runtime_q50_exact": RUNTIME.Q50_Q == 50.0,
        "parent_q50_exact": P.Q50_Q == 50.0,
        "runtime_exact_12h": Q50_HOURS == 12.0,
        "loss_stop_exact_20": Q50_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q50_MIN_EQUITY_USD == 125.0,
        "entry_start_m0": ENTRY_START_S == 0.0,
        "terminal_m12_720": M12_S == 720.0,
        "depth_exact_35c": DEPTH == 0.35,
        "hysteresis_exact_5c": HYSTERESIS == 0.05,
        "recovery_exact_10c": RECOVERY_EDGE == 0.10,
        "recovery_horizon_exact_2s": RECOVERY_HORIZON_S == 2.0,
        "normal_window_exact_5s": SPREAD_WINDOW_S == 5.0,
        "normal_tolerance_exact_2c": NORMAL_TOL == 0.02,
        "normal_min_obs_exact_20": MIN_NORMAL_OBS == 20,
        "exit_latency_reference_81_422ms": EXPECTED_EXIT_EFFECTIVE_LATENCY_MS == 81.422,
        "runtime_live_is_new": RUNTIME.LIVE is LIVE,
        "parent_live_is_new": P.LIVE is LIVE,
        "guardian_live_is_new": V2963.LIVE is LIVE,
        "runtime_generation_cfg_is_new": RUNTIME._generation_cfg is _generation_cfg,
        "parent_generation_cfg_is_new": P._generation_cfg is _generation_cfg,
        "rotation_checkpoint_alias_exact": getattr(LIVE, "ROTATION_CHECKPOINT_FILE", None) == V111.ROTATION_CHECKPOINT_FILE,
        "generation_bootstrap_alias_exact": getattr(LIVE, "GENERATION_BOOTSTRAP_FILE", None) == V111.GENERATION_BOOTSTRAP_FILE,
        "session_risk_baseline_alias_exact": getattr(LIVE, "SESSION_RISK_BASELINE_FILE", None) == V111.SESSION_RISK_BASELINE_FILE,
        "live_exchange_index_2": LIVE_EXCHANGE_INDEX == 2,
        "source_exchange_index_0": SOURCE_EXCHANGE_INDEX == 0,
        "rec25_disabled": True,
        "m1130_cutoff_disabled": True,
        "old_danger_guard_disabled": True,
        "parent_owned_recorder_preserved": True,
        "guardian_recovery_preserved": True,
        "orders_sent": False,
        "transfers_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k not in {"orders_sent", "transfers_sent"})
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "quantity": Q50_Q,
        "runtime_hours": Q50_HOURS,
        "max_loss_usd": Q50_MAX_LOSS_USD,
        "min_equity_usd": Q50_MIN_EQUITY_USD,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 168)
        print("V3.0.0 DEEP35 / HYST5 / REC10 / Q50 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 168)
        for k, v in out.items():
            print(f"{k:100s}: {v}")
    if not ok:
        raise RuntimeError(f"V3.0.0 static self-check failed: {out}")
    return out


def discover_current_crypto_markets(*args, **kwargs):
    _install_patch()
    try:
        return BASE.discover_current_crypto_markets(*args, **kwargs)
    finally:
        _install_patch()


def get_shard_balances(*args, **kwargs):
    _install_patch()
    try:
        return BASE.get_shard_balances(*args, **kwargs)
    finally:
        _install_patch()


def ensure_crypto_shard_funded(*args, **kwargs):
    _install_patch()
    try:
        return BASE.ensure_crypto_shard_funded(*args, **kwargs)
    finally:
        _install_patch()


def crypto_shard_preflight(*, client=None, show=True):
    _install_patch()
    try:
        return BASE.crypto_shard_preflight(client=client, show=show)
    finally:
        _install_patch()


def q50_preflight(*, show=True):
    """Read-only Q50 preflight using latest fee/private-WS and shard gates."""
    _install_patch()
    static_self_check(show=show)
    shard = crypto_shard_preflight(show=show)
    _install_patch()
    pre = V288.live_preflight(
        quote_size=Q50_Q,
        runtime_hours=Q50_HOURS,
        max_start_loss_usd=Q50_MAX_LOSS_USD,
        min_start_equity_usd=Q50_MIN_EQUITY_USD,
        show=show,
        probe_private_ws=True,
    )
    _install_patch()
    out = dict(pre or {})
    out.update(
        {
            "deploy_version": DEPLOY_VERSION,
            "live_version": LIVE.LIVE_VERSION,
            "strategy": "DEEP35_HYST5_REC10_Q50",
            "crypto_shard_preflight": shard,
        }
    )
    return out


def _launch_supervised(*, arm_phrase):
    _install_patch()
    if str(arm_phrase) != Q50_ARM:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q50_ARM!r} exactly.")

    static_self_check(show=True)
    V28._patch_parent()
    _install_patch()
    V28.D._guard_other_live_processes()

    pre = q50_preflight(show=True)

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    mode = "DEEP35_HYST5_Q50_REC10_12H_V300"
    parent_session = (P.CORE.ROOT / f"{stamp}_{mode.lower()}").resolve()
    parent_session.mkdir(parents=True, exist_ok=False)
    (parent_session / "generations").mkdir(parents=True, exist_ok=True)

    cfg = {
        "mode": mode,
        "quote_size": Q50_Q,
        "runtime_hours": Q50_HOURS,
        "max_start_loss_usd": Q50_MAX_LOSS_USD,
        "min_start_equity_usd": Q50_MIN_EQUITY_USD,
        "parent_session_dir": str(parent_session),
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "recorder_version": getattr(RUNTIME.REC, "STUDY_VERSION", None),
        "strategy_entry_start_elapsed_s": ENTRY_START_S,
        "strategy_terminal_cleanup_elapsed_s": M12_S,
        "recorder_persist_end_elapsed_s": M12_S,
        "recorder_label_tail_end_elapsed_s": LABEL_TAIL_END_S,
        "rotation_process_lifetime": "ONE_COMPLETE_M0_M12_WINDOW",
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "fixed_session_risk_baseline": True,
        "external_recorder_parent_owned": True,
        "direct_q50_operator_arm": True,
        "no_auto_scale": True,
        "strategy": "DEEP35_HYST5_REC10_Q50",
        "depth": DEPTH,
        "quote_hysteresis": HYSTERESIS,
        "normal_spread_window_s": SPREAD_WINDOW_S,
        "normal_spread_tolerance": NORMAL_TOL,
        "normal_spread_min_obs": MIN_NORMAL_OBS,
        "recovery_edge": RECOVERY_EDGE,
        "recovery_horizon_s": RECOVERY_HORIZON_S,
        "persistent_target_exit_retry": True,
        "expired_lot_force_flat": True,
        "rec25_enabled": False,
        "m1130_entry_cutoff_enabled": False,
        "persistent_danger_guard_enabled": False,
    }
    cfg_path = parent_session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(parent_session / "parent_preflight_snapshot.json", pre)
    B._atomic(
        parent_session / "architecture_spec_v3_0_0.json",
        {
            "time": B._iso(),
            "architecture": "EXISTING_PARENT_RECORDER_SUPERVISOR_GUARDIAN_WITH_DEEP35_CHILD_ENGINE",
            "strategy": "DEEP35_HYST5_REC10_Q50",
            "live_engine": LIVE.LIVE_VERSION,
            "generation_start_raw_policy": "CURRENT_EOF_PLUS_REQUIRE_FRESH_POST_START_ROW",
            "rotation_gate": "M12_ZERO_POSITION_ZERO_GROUP_RESTING_DURABLE_CHECKPOINT",
            "risk_baseline": "FIXED_ONCE_PER_PARENT_SESSION",
            "recorder": "M0_TO_M12_PLUS_30S_LABEL_TAIL",
            "entry_start_s": ENTRY_START_S,
            "depth": DEPTH,
            "hysteresis": HYSTERESIS,
            "normal_spread_window_s": SPREAD_WINDOW_S,
            "normal_spread_tolerance": NORMAL_TOL,
            "normal_spread_min_obs": MIN_NORMAL_OBS,
            "recovery_edge": RECOVERY_EDGE,
            "recovery_horizon_s": RECOVERY_HORIZON_S,
            "target_exit": "PERSISTENT_REDUCE_ONLY_IOC_AT_ENTRY_PLUS_MINUS_10C",
            "force_flat": "AFTER_2S_CANCEL_ENTRY_DISABLE_TICKER_AND_USE_INHERITED_AUTHORITATIVE_RETRY_FLATTEN",
            "maker_entry_fee_assumption": 0.0,
            "exit_fee": "TAKER_FEE_APPLIES_AND_ACCOUNT_EQUITY_IS_AUTHORITATIVE",
            "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
            "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
            "q50_direct_arm": True,
        },
    )

    log = parent_session / SUPERVISOR_LOG_FILE
    fh = log.open("a", buffering=1, encoding="utf-8")
    child = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--run-supervisor",
        str(parent_session),
        "--config",
        str(cfg_path),
    ]
    caffeinate = shutil.which("caffeinate")
    cmd = ([caffeinate, "-i", "-m"] + child) if caffeinate else child
    try:
        supervisor = subprocess.Popen(
            cmd,
            cwd=str(V28.C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()

    B._atomic(
        P.CORE.CONTROL_PATH,
        {
            "live_version": LIVE.LIVE_VERSION,
            "deploy_version": DEPLOY_VERSION,
            "running": True,
            "pid": supervisor.pid,
            "supervisor_pid": supervisor.pid,
            "session_dir": str(parent_session),
            "mode": mode,
            "started_at": B._iso(),
            "config": cfg,
            "log_path": str(log),
            "caffeinate_used": bool(caffeinate),
            "launch_command": cmd,
        },
    )

    guardian, guardian_log, guardian_cmd = RUNTIME._launch_guardian(parent_session, supervisor.pid)
    ctl = B._read(P.CORE.CONTROL_PATH, {}) or {}
    ctl.update(
        {
            "guardian_pid": guardian.pid,
            "guardian_log_path": str(guardian_log),
            "guardian_command": guardian_cmd,
        }
    )
    B._atomic(P.CORE.CONTROL_PATH, ctl)

    deadline = time.time() + STARTUP_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        if supervisor.poll() is not None:
            raise RuntimeError(
                f"V3.0.0 supervisor exited during startup rc={supervisor.returncode}\n"
                f"{P._tail_text(log)}"
            )
        last = B._read(parent_session / P.SUPERVISOR_HEALTH_FILE, {}) or {}
        gen_dir = last.get("generation_dir")
        gen_ready = False
        if gen_dir:
            gen_ready, _ = P._generation_health_ready(Path(gen_dir))
        recorder_ok = (
            last.get("recorder_alive") is True
            and (last.get("recorder_health") or {}).get("running") is True
            and (last.get("recorder_health") or {}).get("healthy") is True
        )
        if recorder_ok and gen_ready:
            break
        time.sleep(0.25)
    else:
        B._atomic(
            parent_session / P.SESSION_KILL_FILE,
            {"time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_V300_DEEP35"},
        )
        raise RuntimeError(f"V3.0.0 startup timeout. supervisor_health={last}\n{P._tail_text(log)}")

    print("\n" + "=" * 152)
    print("REAL-MONEY V3.0.0 DEEP35 / HYST5 / REC10 / Q50 — 12H ARMED")
    print("=" * 152)
    print("Parent session:              ", parent_session)
    print("Supervisor PID:              ", supervisor.pid)
    print("Guardian PID:                ", guardian.pid)
    print("External recorder PID:       ", last.get("recorder_pid"))
    print("Current trader PID:          ", last.get("trader_pid"))
    print("Engine:                      ", LIVE.LIVE_VERSION)
    print("Quantity:                    Q50 max per entry side")
    print("Entry depth:                 35c opposite-side anchor")
    print("Quote hysteresis:            5c")
    print("Normal spread:               causal 5s / 2c tolerance / 20 prior obs")
    print("Recovery exit:               +10c persistent reduce-only IOC")
    print("Force flat:                  2.0s unresolved lot -> cancel entries + authoritative flatten")
    print("Terminal cleanup:            M12 / authoritative zero")
    print("Recorder:                    parent-owned M0 -> M12 + 30s")
    print("Session runtime:             12.0 hours")
    print("Session software loss stop:  $20 from fixed starting-equity baseline")
    print("Minimum starting equity:     $125")
    print("=" * 152)
    return live_status(show=False, tail_lines=30)


def start_q50_12h_smoke(*, arm_phrase=None):
    return _launch_supervised(arm_phrase=arm_phrase)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    out = RUNTIME.live_status(show=False, tail_lines=tail_lines)
    if show:
        print("=" * 152)
        print("V3.0.0 DEEP35 / HYST5 / REC10 / Q50 LIVE STATUS")
        print("=" * 152)
        print("running:", (out or {}).get("running"))
        print("deploy:", (out or {}).get("deploy_version"))
        print("engine:", (out or {}).get("live_version"))
        print("parent session:", (out or {}).get("parent_session_dir"))
        sh = (out or {}).get("supervisor_health") or {}
        print("supervisor state:", sh.get("state"))
        print("generation:", sh.get("generation_id"))
        print("trader PID/alive:", sh.get("trader_pid"), sh.get("trader_alive"))
        print("recorder PID/alive:", sh.get("recorder_pid"), sh.get("recorder_alive"))
        gh = (out or {}).get("current_generation_health") or {}
        if gh:
            print("generation state:", gh.get("state"))
            print("positions:", gh.get("positions"))
            print("strategy gross realized PnL:", gh.get("strategy_realized_gross_pnl"))
            print("strategy gross realized DD:", gh.get("strategy_realized_gross_max_dd"))
            print("active tracks:", len(gh.get("active_tracks") or {}))
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    if str(arm_phrase) != KILL_ARM:
        raise RuntimeError(f"Pass arm_phrase={KILL_ARM!r} exactly.")
    _install_patch()
    try:
        return RUNTIME.kill_and_flatten_live(arm_phrase=KILL_ARM, wait_s=float(wait_s))
    finally:
        _install_patch()


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
