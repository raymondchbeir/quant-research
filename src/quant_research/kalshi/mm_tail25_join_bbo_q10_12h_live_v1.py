from __future__ import annotations

"""Q10 / 12h Tail25 live deployment for crypto + Gold/Silver/WTI.

This deployment is intentionally additive.  It uses the latest AUDITFIX6
operations/safety stack as its parent, but replaces the Deep35 economics with the
frozen Tail25/Hyst2/Edge15/JOIN_BBO/Force3s engine and replaces crypto-only
recording/routing with a dynamic 12-series implementation.

Real-money contract
-------------------
- Q10 per entry side.
- 12 hours from the first complete live generation/window.
- fixed $30 start-to-current account-equity software stop.
- minimum starting flat equity $300.
- no auto-scaling.
- exact 12-series universe:
  nine crypto 15m series + KXGOLD15M + KXSILVER15M + KXWTI15M.
- one parent-owned authenticated M0->M12(+30s) Top3/ticker/public-trade recorder.
- one short-lived trader generation per complete M0->M12 window.
- generation rotates only after authoritative zero positions and zero resting
  strategy-group orders across every discovered exchange shard.

Promotion contract
------------------
Q10 is blocked until a Q1 one-window smoke on the exact same Git HEAD proves:
- dynamic routing/group creation works for the 12-series universe;
- the recorder survives the generation and records through M12+30;
- the generation reaches the strict zero-position/zero-group-resting checkpoint.

A software stop is not a guaranteed final-loss cap.  In-flight fills, exchange
state, network latency and liquidation slippage can overshoot it.

Importing this module performs no API calls, orders, cancels or transfers.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_q50_deep35_hyst5_rec10_live_v3_0_0_auditfix6 as BASE
from . import mm_event_time_m0_m12_recorder_v7_crypto_commodities as REC
from . import mm_tail25_join_bbo_live_v1 as LIVE
from . import mm_tail25_multiseries_router_v1 as ROUTER


DEPLOY_VERSION = "MM_TAIL25_JOIN_BBO_Q10_MULTI12_12H_V1"
MODULE_NAME = "quant_research.kalshi.mm_tail25_join_bbo_q10_12h_live_v1"

Q1_ARM = "LIVE_TAIL25_MULTI12_Q1_ONE_WINDOW_V1"
Q10_ARM = "LIVE_TAIL25_MULTI12_Q10_12H_V1"
KILL_ARM = BASE.KILL_ARM

Q1_Q = 1.0
Q1_HOURS = 0.5
Q1_MAX_LOSS_USD = 5.0
Q1_MIN_EQUITY_USD = 50.0
Q1_MIN_COLLATERAL_PER_USED_SHARD_USD = 5.0

Q10_Q = 10.0
Q10_HOURS = 12.0
Q10_MAX_LOSS_USD = 30.0
Q10_MIN_EQUITY_USD = 300.0
Q10_MIN_COLLATERAL_PER_USED_SHARD_USD = 25.0

ENTRY_START_S = LIVE.ENTRY_START_S
M12_S = LIVE.M12_S
LABEL_TAIL_END_S = REC.LABEL_TAIL_END_S

RUNTIME = BASE.RUNTIME
P = BASE.P
H = BASE.H
V2963 = BASE.V2963
V28 = BASE.V28
V288 = BASE.V288
V111 = BASE.V111
V1 = BASE.V1
Q1 = BASE.Q1
C = BASE.C

CORE = P.CORE
ROTATION_CHECKPOINT_FILE = LIVE.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = LIVE.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = LIVE.SESSION_RISK_BASELINE_FILE

GENERATION_RSS_WARNING_MB = BASE.Q50_Q and P.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = P.GENERATION_RSS_HARD_LIMIT_MB
GUARDIAN_POST_M12_EXIT_TIMEOUT_S = getattr(
    BASE,
    "GUARDIAN_POST_M12_EXIT_TIMEOUT_S",
    30.0,
)
STARTUP_TIMEOUT_S = 150.0

SUPERVISOR_LOG_FILE = "supervisor_tail25_q10_multi12_v1.log"
GUARDIAN_LOG_FILE = "guardian_tail25_q10_multi12_v1.log"
PROMOTION_PATH = CORE.ROOT / "tail25_q10_multi12_q1_promotion_v1.json"

# Preserve parent supervisor artifact names so the mature dashboard/recovery stack
# can still inspect generations without schema forks.
SUPERVISOR_HEALTH_FILE = P.SUPERVISOR_HEALTH_FILE
SUPERVISOR_FINAL_FILE = P.SUPERVISOR_FINAL_FILE
SUPERVISOR_EVENTS_FILE = P.SUPERVISOR_EVENTS_FILE
SESSION_KILL_FILE = P.SESSION_KILL_FILE
GUARDIAN_HEALTH_FILE = P.GUARDIAN_HEALTH_FILE
GUARDIAN_RECEIPT_FILE = P.GUARDIAN_RECEIPT_FILE
RECOVERY_RECEIPT_FILE = P.RECOVERY_RECEIPT_FILE

_BASE_STATIC = None


def _finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _tail_text(path, chars=18000):
    try:
        return Path(path).read_text(
            encoding="utf-8",
            errors="replace",
        )[-int(chars):]
    except Exception:
        return ""


def _flat_shard_equity(shard_balances, preflight):
    preflight = preflight or {}
    if preflight.get("positions"):
        nonzero = [
            row
            for row in preflight.get("positions") or []
            if abs(B._f((row or {}).get("position_fp"), 0.0)) > B.EPS
        ]
        if nonzero:
            raise RuntimeError(
                f"Tail25 parent baseline requires flat account; positions={nonzero}"
            )
    if preflight.get("resting_orders"):
        raise RuntimeError(
            "Tail25 parent baseline requires zero resting orders"
        )

    breakdown = (shard_balances or {}).get("breakdown_usd") or {}
    vals = []
    for value in breakdown.values():
        z = _finite(value)
        if z is not None:
            vals.append(z)
    if not vals:
        raise RuntimeError(
            f"No usable all-shard balance breakdown: {shard_balances!r}"
        )
    total = float(sum(vals))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"Invalid flat all-shard equity: {total!r}")
    return total


def _used_shard_collateral_check(
    routing,
    shard_balances,
    *,
    minimum_per_used_shard,
):
    used = [int(x) for x in (routing or {}).get("shards") or []]
    breakdown = (shard_balances or {}).get("breakdown_usd") or {}
    normalized = {}
    for key, value in breakdown.items():
        try:
            normalized[int(key)] = float(value)
        except Exception:
            continue

    missing = {
        idx: normalized.get(idx, 0.0)
        for idx in used
        if normalized.get(idx, 0.0) + 1e-12 < float(minimum_per_used_shard)
    }
    return {
        "used_shards": used,
        "balance_by_shard_usd": normalized,
        "minimum_per_used_shard_usd": float(minimum_per_used_shard),
        "underfunded_used_shards": missing,
        "ok": not missing,
    }


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
        "strategy_name": LIVE.STRATEGY_NAME,
        "entry_offset": LIVE.ENTRY_OFFSET,
        "entry_reprice_hysteresis": LIVE.ENTRY_REPRICE_HYSTERESIS,
        "edge_zone": LIVE.EDGE_ZONE,
        "exit_mode": "CONTINUOUS_JOIN_BBO",
        "exit_reprice_hysteresis": LIVE.EXIT_REPRICE_HYSTERESIS,
        "exit_horizon_s": LIVE.EXIT_HORIZON_S,
        "no_reentry_after_first_fill": True,
        "no_reentry_after_edge": True,
        "universe": list(ROUTER.SERIES),
        "dynamic_exchange_index_routing": True,
        "multi_shard_order_groups": True,
        "recorder_study_version": REC.STUDY_VERSION,
    }


def _run_generation(session, cfg_path):
    """Detached trader generation. May send real orders only after parent arm."""
    _install_patch()
    session = Path(session).resolve()
    cfg = B._read(Path(cfg_path), {}) or {}
    old_fee = OOS.fee_preflight

    def child_fee_preflight(
        *,
        horizon_hours=OOS.FEE_CHANGE_HORIZON_H,
        save_path=None,
        show=True,
    ):
        return P.V282._validated_parent_fee_snapshot(
            session,
            horizon_hours=horizon_hours,
            save_path=save_path,
            show=show,
        )

    OOS.fee_preflight = child_fee_preflight
    try:
        return LIVE.run_live_process(session, cfg)
    finally:
        OOS.fee_preflight = old_fee


def _launch_generation(
    parent_session,
    parent_cfg,
    parent_preflight,
    *,
    generation_id,
    recorder_pid,
    session_start_equity,
    session_kill_equity,
    remaining_hours,
):
    parent_session = Path(parent_session).resolve()
    generation_dir = (
        parent_session
        / "generations"
        / f"gen_{int(generation_id):04d}"
    )
    generation_dir.mkdir(parents=True, exist_ok=False)
    P._attach_raw_capture(parent_session, generation_dir)

    cfg = _generation_cfg(
        parent_cfg,
        generation_id=generation_id,
        generation_dir=generation_dir,
        recorder_pid=recorder_pid,
        session_start_equity=session_start_equity,
        session_kill_equity=session_kill_equity,
        remaining_hours=remaining_hours,
    )
    cfg_path = generation_dir / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(
        generation_dir / "parent_preflight_snapshot.json",
        parent_preflight,
    )

    log = generation_dir / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--run-generation",
        str(generation_dir),
        "--config",
        str(cfg_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(V28.C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()

    B._append(
        parent_session / SUPERVISOR_EVENTS_FILE,
        {
            "time": B._iso(),
            "event": "TAIL25_GENERATION_LAUNCHED",
            "generation_id": int(generation_id),
            "generation_dir": str(generation_dir),
            "trader_pid": proc.pid,
            "remaining_hours": float(remaining_hours),
            "strategy": LIVE.STRATEGY_NAME,
            "q": float(parent_cfg["quote_size"]),
            "terminal_cleanup_s": M12_S,
        },
    )
    return proc, generation_dir, cfg, log


def _group_resting_rows(rows, gid):
    gids = ROUTER.group_ids(gid)
    return [
        row
        for row in rows or []
        if str((row or {}).get("order_group_id") or "") in gids
    ]


def _recover_generation_fail_closed(
    parent_session,
    generation_dir,
    trader_pid,
    *,
    reason,
):
    """Authoritative mixed-shard recovery. May cancel/flatten real exposure."""
    _install_patch()
    parent_session = Path(parent_session).resolve()
    generation_dir = Path(generation_dir).resolve()
    trader_pid = int(trader_pid or 0)
    group = B._read(generation_dir / "order_group.json", {}) or {}
    gid = group.get("order_group_id")

    client = B.Q1.LiveClient()
    group_trigger = (
        B._trigger_group(client, gid)
        if gid
        else {"ok": False, "reason": "missing_order_group_id"}
    )
    trader_stop = (
        P.V27._terminate_pid_group(trader_pid)
        if trader_pid > 0
        else {"pid": None, "dead": True, "note": "no trader pid"}
    )

    # B._fallback_cleanup now uses the dynamic 12-series payload and multi-group
    # trigger installed by ROUTER.  It refuses positions outside recorded metadata.
    fallback = B._fallback_cleanup({"session_dir": str(generation_dir)})
    time.sleep(0.30)

    resting, rest_timing = B._resting(client)
    positions, pos_timing = B._positions(client)
    group_resting = _group_resting_rows(resting, gid)
    nonzero = [
        row
        for row in positions or []
        if abs(B._f((row or {}).get("position_fp"), 0.0)) > B.EPS
    ]
    verified = not group_resting and not nonzero

    receipt = {
        "time": B._iso(),
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "parent_session": str(parent_session),
        "generation_dir": str(generation_dir),
        "trader_pid": trader_pid or None,
        "reason": str(reason),
        "group_ids_by_shard": gid,
        "group_trigger": group_trigger,
        "trader_stop": trader_stop,
        "fallback_cleanup": fallback,
        "group_resting": group_resting,
        "nonzero_positions": nonzero,
        "all_account_resting_count": len(resting or []),
        "verification_timing": {
            "resting": rest_timing,
            "positions": pos_timing,
        },
        "recovery_verified": bool(verified),
    }
    B._atomic(parent_session / RECOVERY_RECEIPT_FILE, receipt)
    B._append(
        parent_session / SUPERVISOR_EVENTS_FILE,
        {"time": B._iso(), "event": "TAIL25_FAIL_CLOSED_RECOVERY", **receipt},
    )
    if not verified:
        raise RuntimeError(
            f"Tail25 mixed-shard recovery failed verification: {receipt}"
        )
    return receipt


def _stop_external_recorder(pid):
    return REC.stop_parent_recorder(pid)


def _fresh_generation_preflight(
    parent_cfg,
    *,
    remaining_hours,
    show=False,
):
    q = float(parent_cfg["quote_size"])
    min_per_shard = (
        Q1_MIN_COLLATERAL_PER_USED_SHARD_USD
        if q <= 1.0 + 1e-12
        else Q10_MIN_COLLATERAL_PER_USED_SHARD_USD
    )
    return _read_only_preflight(
        q=q,
        hours=max(0.02, float(remaining_hours)),
        max_loss=float(parent_cfg["max_start_loss_usd"]),
        min_equity=float(parent_cfg["min_start_equity_usd"]),
        min_per_shard=float(min_per_shard),
        probe_private_ws=False,
        show=bool(show),
        run_static=False,
    )


def _install_patch():
    """Install AUDITFIX6 operations first, then Tail25/12-series bindings last."""
    BASE._install_patch()

    # Recorder globals must match B.SERIES before any inherited preflight runs.
    REC._install_patch()
    ROUTER.install_runtime_patch()

    LIVE.ROTATION_CHECKPOINT_FILE = ROTATION_CHECKPOINT_FILE
    LIVE.GENERATION_BOOTSTRAP_FILE = GENERATION_BOOTSTRAP_FILE
    LIVE.SESSION_RISK_BASELINE_FILE = SESSION_RISK_BASELINE_FILE

    # Detached runtime facade.
    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q10_ARM
    RUNTIME.Q50_Q = Q10_Q
    RUNTIME.Q50_HOURS = Q10_HOURS
    RUNTIME.Q50_MAX_LOSS_USD = Q10_MAX_LOSS_USD
    RUNTIME.Q50_MIN_EQUITY_USD = Q10_MIN_EQUITY_USD
    RUNTIME.LIVE = LIVE
    RUNTIME.M1_S = ENTRY_START_S
    RUNTIME.M12_S = M12_S
    RUNTIME._generation_cfg = _generation_cfg
    RUNTIME._run_generation = _run_generation
    RUNTIME._run_supervisor = P._run_supervisor
    RUNTIME._run_guardian = P._guardian_loop
    try:
        RUNTIME.REC = REC
    except Exception:
        pass

    # Parent supervisor.  These functions are resolved from P globals at runtime.
    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.REC = REC
    P.Q50_Q = Q10_Q
    P.Q50_HOURS = Q10_HOURS
    P.Q50_MAX_LOSS_USD = Q10_MAX_LOSS_USD
    P.Q50_MIN_EQUITY_USD = Q10_MIN_EQUITY_USD
    P.M1_S = ENTRY_START_S
    P.M5_S = M12_S
    P.RECORDER_M12_S = M12_S
    P.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    P.PROMOTION_PATH = PROMOTION_PATH
    P._generation_cfg = _generation_cfg
    P._launch_generation = _launch_generation
    P._fresh_generation_preflight = _fresh_generation_preflight
    P._recover_generation_fail_closed = _recover_generation_fail_closed
    P._start_external_recorder = REC.start_parent_recorder
    P._stop_external_recorder = _stop_external_recorder

    # Guardian follows the same engine and patched mixed-shard recovery function.
    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.POST_M5_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S
    try:
        V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state
    except Exception:
        pass

    # Re-apply router LAST because AUDITFIX6 installs the historical crypto-shard2
    # transport earlier in this function.
    ROUTER.install_runtime_patch()
    return {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "recorder_version": REC.STUDY_VERSION,
        "router_version": ROUTER.ROUTER_VERSION,
        "module_name": MODULE_NAME,
        "orders_sent": False,
    }


def static_self_check(*, show=True):
    global _BASE_STATIC
    if _BASE_STATIC is None:
        _BASE_STATIC = BASE.static_self_check(show=False)
    live = LIVE.static_self_check(show=False)
    rec = REC.static_self_check(show=False)
    router = ROUTER.static_self_check(show=False)
    _install_patch()

    synthetic_routing = {"shards": [0, 2]}
    synthetic_balances = {"breakdown_usd": {0: 50.0, 2: 300.0}}
    synthetic_collateral = _used_shard_collateral_check(
        synthetic_routing,
        synthetic_balances,
        minimum_per_used_shard=25.0,
    )

    checks = {
        "auditfix6_operations_static_ok": _BASE_STATIC.get("ok") is True,
        "tail25_engine_static_ok": live.get("ok") is True,
        "multi12_recorder_static_ok": rec.get("ok") is True,
        "multi12_router_static_ok": router.get("ok") is True,
        "deploy_version_exact": DEPLOY_VERSION
        == "MM_TAIL25_JOIN_BBO_Q10_MULTI12_12H_V1",
        "live_version_exact": LIVE.LIVE_VERSION == "MM_TAIL25_JOIN_BBO_LIVE_V1",
        "q10_exact": Q10_Q == 10.0,
        "q1_smoke_exact": Q1_Q == 1.0,
        "runtime_exact_12h": Q10_HOURS == 12.0,
        "loss_stop_exact_30": Q10_MAX_LOSS_USD == 30.0,
        "min_equity_exact_300": Q10_MIN_EQUITY_USD == 300.0,
        "entry_offset_25c": LIVE.ENTRY_OFFSET == 0.25,
        "entry_hyst_2c": LIVE.ENTRY_REPRICE_HYSTERESIS == 0.02,
        "edge_zone_15c": LIVE.EDGE_ZONE == 0.15,
        "join_bbo_exit": True,
        "exit_hyst_2c": LIVE.EXIT_REPRICE_HYSTERESIS == 0.02,
        "force_flat_3s": LIVE.EXIT_HORIZON_S == 3.0,
        "entry_m0": ENTRY_START_S == 0.0,
        "terminal_m12": M12_S == 720.0,
        "recorder_m12": REC.TRADE_WINDOW_END_S == 720.0,
        "recorder_tail_750": REC.LABEL_TAIL_END_S == 750.0,
        "universe_exact_12": len(ROUTER.SERIES) == 12,
        "commodities_exact": tuple(ROUTER.COMMODITY_SERIES)
        == ("KXGOLD15M", "KXSILVER15M", "KXWTI15M"),
        "runtime_live_is_tail25": RUNTIME.LIVE is LIVE,
        "parent_live_is_tail25": P.LIVE is LIVE,
        "parent_recorder_is_v7": P.REC is REC,
        "parent_generation_cfg_is_tail25": P._generation_cfg is _generation_cfg,
        "parent_launch_generation_is_tail25": P._launch_generation is _launch_generation,
        "parent_recovery_is_multishard": P._recover_generation_fail_closed
        is _recover_generation_fail_closed,
        "router_installed_last": B._payload.__module__ == ROUTER.__name__,
        "multi_shard_collateral_regression": synthetic_collateral.get("ok") is True,
        "q10_requires_same_head_q1_promotion": True,
        "no_auto_scale": True,
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
        "module_name": MODULE_NAME,
        "live_version": LIVE.LIVE_VERSION,
        "recorder_version": REC.STUDY_VERSION,
        "router_version": ROUTER.ROUTER_VERSION,
        "quantity": Q10_Q,
        "runtime_hours": Q10_HOURS,
        "max_loss_usd": Q10_MAX_LOSS_USD,
        "min_equity_usd": Q10_MIN_EQUITY_USD,
        "promotion_path": str(PROMOTION_PATH),
        "series": list(ROUTER.SERIES),
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 156)
        print("TAIL25 Q10 MULTI12 12H STATIC CHECK — NO API / NO ORDERS")
        print("=" * 156)
        for k, v in out.items():
            print(f"{k:96s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 deployment static check failed: {out}")
    return out


def _read_only_preflight(
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
    _install_patch()
    if run_static:
        static_self_check(show=show)
        _install_patch()

    routing = ROUTER.routing_snapshot(require_all=True, refresh=True)
    shard_balances = ROUTER.get_shard_balances()
    _install_patch()

    pre = V288.live_preflight(
        quote_size=float(q),
        runtime_hours=float(hours),
        max_start_loss_usd=float(max_loss),
        min_start_equity_usd=float(min_equity),
        show=bool(show),
        probe_private_ws=bool(probe_private_ws),
    )
    _install_patch()
    pre = dict(pre or {})

    exact_flat_equity = _flat_shard_equity(shard_balances, pre)
    if exact_flat_equity + 1e-12 < float(min_equity):
        raise RuntimeError(
            f"Starting flat all-shard equity ${exact_flat_equity:.4f} "
            f"< required ${float(min_equity):.4f}"
        )
    if not (0.0 < float(max_loss) < exact_flat_equity):
        raise RuntimeError(
            f"Invalid loss stop ${max_loss:.4f} for equity ${exact_flat_equity:.4f}"
        )

    collateral = _used_shard_collateral_check(
        routing,
        shard_balances,
        minimum_per_used_shard=float(min_per_shard),
    )
    if not collateral["ok"]:
        raise RuntimeError(
            "One or more execution shards do not have the required local cash. "
            "No automatic transfer was attempted. "
            f"details={collateral}"
        )

    account = dict(pre.get("account") or {})
    inherited_equity = _finite(account.get("equity_usd"))
    account.update(
        {
            "equity_usd": float(exact_flat_equity),
            "cash_balance_usd": float(exact_flat_equity),
            "portfolio_value_usd": 0.0,
        }
    )
    pre.update(
        {
            "ok": True,
            "account": account,
            "kill_equity_usd": float(exact_flat_equity - float(max_loss)),
            "deploy_version": DEPLOY_VERSION,
            "live_version": LIVE.LIVE_VERSION,
            "recorder_version": REC.STUDY_VERSION,
            "router_version": ROUTER.ROUTER_VERSION,
            "strategy": LIVE.STRATEGY_NAME,
            "quantity": float(q),
            "runtime_hours": float(hours),
            "universe": list(ROUTER.SERIES),
            "routing": routing,
            "shard_balances": shard_balances,
            "used_shard_collateral": collateral,
            "risk_baseline_source": "CURRENT_FLAT_ALL_SHARD_BALANCE_BREAKDOWN",
            "risk_baseline_equity_usd": float(exact_flat_equity),
            "risk_baseline_kill_equity_usd": float(
                exact_flat_equity - float(max_loss)
            ),
            "inherited_preflight_equity_usd": inherited_equity,
            "risk_baseline_corrected_for_new_parent": True,
            "automatic_collateral_transfer": False,
        }
    )

    if show:
        print("=" * 156)
        print("TAIL25 MULTI12 ROUTING / RISK PREFLIGHT")
        print("=" * 156)
        print(f"Flat all-shard equity:       ${exact_flat_equity:.4f}")
        print(f"Software kill equity:        ${exact_flat_equity - float(max_loss):.4f}")
        print("Execution shards:            ", routing.get("shards"))
        print("Shard balances:              ", collateral.get("balance_by_shard_usd"))
        print("12-series routing:           PASS")
        print("Auto collateral transfer:    DISABLED")
        print("ORDERS SENT:                 NO")
        print("TRANSFERS SENT:              NO")

    return pre


def q1_smoke_preflight(*, show=True):
    return _read_only_preflight(
        q=Q1_Q,
        hours=Q1_HOURS,
        max_loss=Q1_MAX_LOSS_USD,
        min_equity=Q1_MIN_EQUITY_USD,
        min_per_shard=Q1_MIN_COLLATERAL_PER_USED_SHARD_USD,
        probe_private_ws=True,
        show=show,
    )


def q10_preflight(*, show=True):
    return _read_only_preflight(
        q=Q10_Q,
        hours=Q10_HOURS,
        max_loss=Q10_MAX_LOSS_USD,
        min_equity=Q10_MIN_EQUITY_USD,
        min_per_shard=Q10_MIN_COLLATERAL_PER_USED_SHARD_USD,
        probe_private_ws=True,
        show=show,
    )


def rotation_promotion_status(*, show=True):
    rec = B._read(PROMOTION_PATH, {}) or {}
    head = P._current_head()
    checks = {
        "promotion_file_present": bool(rec),
        "passed": rec.get("passed") is True,
        "deploy_version_exact": rec.get("deploy_version") == DEPLOY_VERSION,
        "live_version_exact": rec.get("live_version") == LIVE.LIVE_VERSION,
        "git_head_exact": bool(head) and rec.get("git_head") == head,
        "smoke_q1_exact": abs(float(rec.get("smoke_quantity") or 0.0) - 1.0) < 1e-12,
        "rotation_checkpoint_verified": rec.get("rotation_checkpoint_verified") is True,
        "same_recorder_survived_rotation": rec.get(
            "same_recorder_survived_trader_rotation"
        )
        is True,
        "m12_plus_30_recorded": rec.get("m12_plus_30_observed_after_rotation")
        is True,
    }
    ready = all(checks.values())
    out = {
        "ready_for_q10": bool(ready),
        "current_git_head": head,
        "promotion_path": str(PROMOTION_PATH),
        "checks": checks,
        "record": rec,
        "orders_sent": False,
    }
    if show:
        print("=" * 132)
        print("TAIL25 Q10 PROMOTION STATUS — READ ONLY")
        print("=" * 132)
        print("ready_for_q10:", out["ready_for_q10"])
        print("current_git_head:", head)
        print("promotion_path:", PROMOTION_PATH)
        for k, v in checks.items():
            print(f"  {k:48s}: {v}")
    return out


def _require_q10_promotion():
    status = rotation_promotion_status(show=False)
    if not status.get("ready_for_q10"):
        raise RuntimeError(
            "Q10 Tail25 arming blocked: run and pass the Q1 one-window smoke on "
            f"this exact Git HEAD first. checks={status.get('checks')}"
        )
    return status


def _launch_guardian(parent_session, supervisor_pid):
    parent_session = Path(parent_session).resolve()
    log = parent_session / GUARDIAN_LOG_FILE
    fh = log.open("a", buffering=1, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--run-guardian",
        str(parent_session),
        "--supervisor-pid",
        str(int(supervisor_pid)),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(V28.C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()
    return proc, log, cmd


def _launch_supervised(
    *,
    q,
    hours,
    max_loss,
    min_equity,
    min_per_shard,
    mode,
    rotation_smoke,
    arm_phrase,
    expected_arm,
    require_promotion,
):
    _install_patch()
    if str(arm_phrase) != str(expected_arm):
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly."
        )
    if require_promotion:
        _require_q10_promotion()

    static_self_check(show=True)
    V28._patch_parent()
    _install_patch()
    V28.D._guard_other_live_processes()

    pre = _read_only_preflight(
        q=float(q),
        hours=float(hours),
        max_loss=float(max_loss),
        min_equity=float(min_equity),
        min_per_shard=float(min_per_shard),
        probe_private_ws=True,
        show=True,
        run_static=False,
    )
    _install_patch()

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    parent_session = (
        CORE.ROOT / f"{stamp}_{str(mode).lower()}"
    ).resolve()
    parent_session.mkdir(parents=True, exist_ok=False)
    (parent_session / "generations").mkdir(parents=True, exist_ok=True)

    cfg = {
        "mode": str(mode),
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "minimum_collateral_per_used_shard_usd": float(min_per_shard),
        "rotation_smoke": bool(rotation_smoke),
        "parent_session_dir": str(parent_session),
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "recorder_version": REC.STUDY_VERSION,
        "router_version": ROUTER.ROUTER_VERSION,
        "strategy": LIVE.STRATEGY_NAME,
        "universe": list(ROUTER.SERIES),
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "q10_promotion_required": bool(require_promotion),
        "fixed_session_risk_baseline": True,
        "no_auto_scale": True,
        "automatic_collateral_transfer": False,
    }
    cfg_path = parent_session / "process_config.json"
    B._atomic(cfg_path, cfg)
    B._atomic(parent_session / "parent_preflight_snapshot.json", pre)
    B._atomic(
        parent_session / "tail25_architecture_spec_v1.json",
        {
            "time": B._iso(),
            "architecture": (
                "AUDITFIX6_OPERATIONS_PLUS_ONE_PARENT_MULTI12_RECORDER_PLUS_"
                "ONE_M0_M12_TRADER_GENERATION_PER_WINDOW"
            ),
            "strategy": LIVE.STRATEGY_NAME,
            "entry": "same-side BBO +/-25c; 2c reprice; edge15 kill",
            "exit": "continuous reduce-only JOIN_BBO; 2c reprice",
            "force_flat": "3s from first entry fill via inherited authoritative retry flatten",
            "generation_start_raw_policy": "CURRENT_EOF_PLUS_REQUIRE_FRESH_POST_START_ROW",
            "rotation_gate": "M12_ZERO_POSITION_ZERO_MULTI_GROUP_RESTING_DURABLE_CHECKPOINT",
            "risk_baseline": "FIXED_ONCE_PER_PARENT_SESSION_FROM_FLAT_ALL_SHARD_CASH",
            "software_loss_stop_usd": float(max_loss),
            "universe": list(ROUTER.SERIES),
            "dynamic_exchange_index_routing": True,
            "multi_shard_order_groups": True,
            "recorder": "M0_TO_M12_PLUS_30S_LABEL_TAIL",
            "automatic_collateral_transfer": False,
            "q10_requires_same_head_q1_smoke": True,
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
        CORE.CONTROL_PATH,
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

    guardian, guardian_log, guardian_cmd = _launch_guardian(
        parent_session,
        supervisor.pid,
    )
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    ctl.update(
        {
            "guardian_pid": guardian.pid,
            "guardian_log_path": str(guardian_log),
            "guardian_command": guardian_cmd,
        }
    )
    B._atomic(CORE.CONTROL_PATH, ctl)

    deadline = time.time() + STARTUP_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        if supervisor.poll() is not None:
            raise RuntimeError(
                f"Tail25 supervisor exited during startup rc={supervisor.returncode}\n"
                f"{_tail_text(log)}"
            )
        last = B._read(parent_session / SUPERVISOR_HEALTH_FILE, {}) or {}
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
            parent_session / SESSION_KILL_FILE,
            {"time": B._iso(), "reason": "STARTUP_HEALTH_TIMEOUT_TAIL25_V1"},
        )
        raise RuntimeError(
            f"Tail25 startup timeout. supervisor_health={last}\n{_tail_text(log)}"
        )

    print("\n" + "=" * 156)
    print("REAL-MONEY TAIL25 / HYST2 / EDGE15 / JOIN_BBO — SUPERVISOR ARMED")
    print("=" * 156)
    print("Parent session:              ", parent_session)
    print("Supervisor PID:              ", supervisor.pid)
    print("Guardian PID:                ", guardian.pid)
    print("External recorder PID:       ", last.get("recorder_pid"))
    print("Current trader PID:          ", last.get("trader_pid"))
    print("Engine:                      ", LIVE.LIVE_VERSION)
    print("Recorder:                    ", REC.STUDY_VERSION)
    print("Router:                      ", ROUTER.ROUTER_VERSION)
    print("Universe:                    9 crypto + GOLD + SILVER + WTI")
    print("Quantity:                    ", f"Q{float(q):g} max per entry side")
    print("Entry:                       same-side BBO +/-25c; 2c reprice")
    print("Edge kill:                   15c / 85c; no re-entry")
    print("Exit:                        continuous JOIN_BBO; 2c reprice")
    print("Force flat:                  3.0s from first entry fill")
    print("Generation terminal:         M12 / authoritative zero")
    print("Recorder horizon:            M0 -> M12 + 30s label tail")
    print("Session runtime:             ", f"{float(hours):.2f} hours")
    print("Session software loss stop:  ", f"${float(max_loss):.2f} from fixed flat-equity baseline")
    print("Minimum starting equity:     ", f"${float(min_equity):.2f}")
    print("Auto-scaling:                DISABLED")
    print("Auto shard transfer:         DISABLED")
    print("Mac sleep prevention:        ", "caffeinate enabled" if caffeinate else "caffeinate unavailable")
    print("Q10 promotion:               ", "REQUIRED+PASSED" if require_promotion else "Q1 SMOKE MODE")
    print("=" * 156)
    return live_status(show=False, tail_lines=30)


def start_q1_one_window_smoke(*, arm_phrase=None):
    return _launch_supervised(
        q=Q1_Q,
        hours=Q1_HOURS,
        max_loss=Q1_MAX_LOSS_USD,
        min_equity=Q1_MIN_EQUITY_USD,
        min_per_shard=Q1_MIN_COLLATERAL_PER_USED_SHARD_USD,
        mode="TAIL25_MULTI12_Q1_ONE_WINDOW_SMOKE_V1",
        rotation_smoke=True,
        arm_phrase=arm_phrase,
        expected_arm=Q1_ARM,
        require_promotion=False,
    )


def start_q10_12h(*, arm_phrase=None):
    return _launch_supervised(
        q=Q10_Q,
        hours=Q10_HOURS,
        max_loss=Q10_MAX_LOSS_USD,
        min_equity=Q10_MIN_EQUITY_USD,
        min_per_shard=Q10_MIN_COLLATERAL_PER_USED_SHARD_USD,
        mode="TAIL25_MULTI12_Q10_12H_V1",
        rotation_smoke=False,
        arm_phrase=arm_phrase,
        expected_arm=Q10_ARM,
        require_promotion=True,
    )


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    if not ctl:
        out = {"running": False, "message": "No Tail25 live control file."}
        if show:
            print(out)
        return out

    parent = Path(ctl.get("session_dir") or "")
    sh = B._read(parent / SUPERVISOR_HEALTH_FILE, {}) or {}
    sf = B._read(parent / SUPERVISOR_FINAL_FILE, {}) or {}
    gh = B._read(parent / GUARDIAN_HEALTH_FILE, {}) or {}
    gr = B._read(parent / GUARDIAN_RECEIPT_FILE, {}) or {}
    gen_dir = sh.get("generation_dir")
    gen_health = (
        B._read(Path(gen_dir) / "health.json", {}) if gen_dir else {}
    )
    gen_final = (
        B._read(Path(gen_dir) / "final_summary.json", {}) if gen_dir else {}
    )
    checkpoint = (
        B._read(Path(gen_dir) / LIVE.ROTATION_CHECKPOINT_FILE, {})
        if gen_dir
        else {}
    )
    running = bool(
        B._pid_alive(ctl.get("supervisor_pid") or ctl.get("pid"))
        and not sf
    )
    log_path = Path(ctl.get("log_path") or parent / SUPERVISOR_LOG_FILE)
    tail = _tail_text(log_path, chars=18000).splitlines()[-int(tail_lines):]
    out = {
        "running": running,
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "parent_session_dir": str(parent),
        "supervisor_pid": ctl.get("supervisor_pid") or ctl.get("pid"),
        "guardian_pid": ctl.get("guardian_pid"),
        "supervisor_health": sh,
        "guardian_health": gh,
        "guardian_receipt": gr,
        "current_generation_health": gen_health,
        "current_generation_final": gen_final,
        "current_rotation_checkpoint": checkpoint,
        "supervisor_final": sf,
        "promotion": rotation_promotion_status(show=False),
        "log_tail": tail,
    }
    if show:
        print("=" * 148)
        print("TAIL25 MULTI12 LIVE STATUS")
        print("=" * 148)
        print("running:", running)
        print("parent session:", parent)
        print("supervisor PID:", out["supervisor_pid"])
        print("guardian PID:", out["guardian_pid"])
        if sh:
            print("supervisor state:", sh.get("state"))
            print("generation:", sh.get("generation_id"))
            print("trader PID/alive:", sh.get("trader_pid"), sh.get("trader_alive"))
            print("recorder PID/alive:", sh.get("recorder_pid"), sh.get("recorder_alive"))
            print("session start/kill equity:", sh.get("session_start_equity_usd"), sh.get("session_kill_equity_usd"))
        if gen_health:
            print("generation state:", gen_health.get("state"))
            print("strategy:", gen_health.get("strategy"))
            print("positions:", gen_health.get("positions"))
            print("active tracks:", len(gen_health.get("active_tracks") or {}))
            print("strategy gross realized PnL:", gen_health.get("strategy_realized_gross_pnl"))
            print("strategy gross realized DD:", gen_health.get("strategy_realized_gross_max_dd"))
            print("Tail25 force flats:", gen_health.get("tail25_force_flats"))
            print("order groups by shard:", gen_health.get("order_groups_by_shard"))
        if checkpoint:
            print("rotation safe:", checkpoint.get("safe_to_rotate"), checkpoint.get("reason"))
        if sf:
            print("supervisor shutdown:", sf.get("shutdown_reason"))
            print("generations completed:", sf.get("generations_completed"))
            print("last error:", sf.get("last_error"))
        print("Q10 promotion ready:", out["promotion"].get("ready_for_q10"))
        if tail:
            print("\nSUPERVISOR LOG TAIL")
            print("\n".join(tail))
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    if str(arm_phrase) != str(KILL_ARM):
        raise RuntimeError(f"Pass arm_phrase={KILL_ARM!r} exactly.")
    # The inherited runtime writes the parent kill request and lets the supervisor
    # / guardian own authoritative cleanup. P._recover_generation_fail_closed has
    # been rebound above to the mixed-shard verifier.
    return RUNTIME.kill_and_flatten_live(
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
        return _run_generation(
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
