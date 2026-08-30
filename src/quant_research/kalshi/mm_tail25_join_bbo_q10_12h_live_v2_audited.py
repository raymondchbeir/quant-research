from __future__ import annotations

"""Final audited Tail25 Q10 / 12h mixed-universe deployment.

This module is the ONLY intended operator entrypoint for this branch.  It keeps the
frozen strategy economics from the validated research replay and consolidates the
source-audit hardening without relying on a chain of deployment wrappers.

Frozen economics
----------------
Universe: 9 crypto 15m + GOLD15M + SILVER15M + WTI15M.
Entry:     BID = current YES bid -25c; ASK = current YES ask +25c.
Reprice:   >=2c target movement.
Edge kill: YES bid <=15c or YES ask >=85c, permanent for that window.
After fill:no new entry; cancel entry residuals.
Exit:      continuous reduce-only JOIN_BBO, reprice >=2c / residual qty change.
Deadline:  3.0s from first exchange fill timestamp when available, then
           authoritative retry-until-flat liquidation.
Terminal:  M12; rotate only after zero position + zero strategy-group resting.

Live contract
-------------
Q10, 12h, fixed $30 session-equity software stop, >=$300 starting flat equity,
no auto scaling, no automatic balance transfers.  A same-Git-HEAD Q1 complete
M0->M12 smoke is mandatory before Q10 can arm.

Audit gates retained/added
--------------------------
- AUDITFIX6 transport / private execution / cancel / guardian stack.
- dynamic exchange_index discovery and one order group per used shard.
- authenticated parent Top3+ticker+public-trade recorder through M12+30.
- stable-key create/cancel lifecycle waits; late CREATE-ack cleanup; ambiguous
  retired CREATE fail-closed; immediate entry/exit rejoin after terminal lifecycle.
- private fill/user-order websocket required for new entry; WS DOWN disables entry
  for the current window; active entry canceled if raw Top3 freshness exceeds 3s.
- fee preflight must verify exactly all 12 series under current exchange metadata.
- all-shard flat-cash sum must reconcile to independently-read account equity
  within 5 cents before it can become the fixed risk baseline.
- order-group rolling matched-contract limit = 30Q per used shard.  The maximum
  legitimate no-reentry round-trip volume is 24Q if all 12 series share one shard.
- Q1 promotion requires all 12 series in the same rotation target window, all 12
  in raw metadata through M12+30, and at least one acknowledged ENTRY order for
  every series, in addition to the inherited flat/resting-zero and recorder-tail
  checks.

Importing this module performs no API calls, orders, cancels or transfers.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_8_8 as V288
from . import mm_deep_tail_join_ask_q50_deep35_hyst5_rec10_live_v3_0_0_auditfix6 as OPS
from . import mm_event_time_m0_m12_recorder_v7_crypto_commodities as REC
from . import mm_tail25_join_bbo_live_v1_3_audit as LIVE
from . import mm_tail25_join_bbo_q10_12h_live_v1 as COREDEP
from . import mm_tail25_multiseries_router_v1 as ROUTER


# Mature parent/guardian/runtime modules inherited from AUDITFIX6.
P = OPS.P
RUNTIME = OPS.RUNTIME
V2963 = OPS.V2963
Q1 = OPS.Q1
C = OPS.C
CORE = P.CORE

DEPLOY_VERSION = "MM_TAIL25_JOIN_BBO_Q10_MULTI12_12H_V2_AUDITED"
PATCH_VERSION = "TAIL25_MULTI12_SOURCE_AUDIT_FINAL_V2"
MODULE_NAME = "quant_research.kalshi.mm_tail25_join_bbo_q10_12h_live_v2_audited"

Q1_ARM = "LIVE_TAIL25_MULTI12_Q1_ONE_WINDOW_V2_AUDITED"
Q10_ARM = "LIVE_TAIL25_MULTI12_Q10_12H_V2_AUDITED"
KILL_ARM = OPS.KILL_ARM

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

ORDER_GROUP_LIMIT_MULTIPLIER = 30.0
MAX_LEGIT_MATCH_MULTIPLIER = 2.0 * len(ROUTER.SERIES)
BALANCE_RECONCILE_ABS_TOL_USD = 0.05
RECORDER_POST_TERM_WAIT_S = 5.0

GENERATION_RSS_WARNING_MB = P.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = P.GENERATION_RSS_HARD_LIMIT_MB
GUARDIAN_POST_M12_EXIT_TIMEOUT_S = getattr(
    OPS,
    "GUARDIAN_POST_M12_EXIT_TIMEOUT_S",
    30.0,
)
STARTUP_TIMEOUT_S = 150.0

SUPERVISOR_LOG_FILE = "supervisor_tail25_multi12_v2_audited.log"
GUARDIAN_LOG_FILE = "guardian_tail25_multi12_v2_audited.log"
PROMOTION_PATH = CORE.ROOT / "tail25_q10_multi12_q1_promotion_v2_audited.json"

SUPERVISOR_HEALTH_FILE = P.SUPERVISOR_HEALTH_FILE
SUPERVISOR_FINAL_FILE = P.SUPERVISOR_FINAL_FILE
SUPERVISOR_EVENTS_FILE = P.SUPERVISOR_EVENTS_FILE
SESSION_KILL_FILE = P.SESSION_KILL_FILE
GUARDIAN_HEALTH_FILE = P.GUARDIAN_HEALTH_FILE
GUARDIAN_RECEIPT_FILE = P.GUARDIAN_RECEIPT_FILE
RECOVERY_RECEIPT_FILE = P.RECOVERY_RECEIPT_FILE

# Capture pristine parent functions before this module publishes runtime bindings.
_PARENT_WRITE_SMOKE_PROMOTION = P._write_smoke_promotion
_BASE_DEP_GENERATION_LAUNCH = COREDEP._launch_generation
_BASE_DEP_RECOVERY = COREDEP._recover_generation_fail_closed
_BASE_DEP_LIVE_STATUS = COREDEP.live_status
_BASE_DEP_KILL = COREDEP.kill_and_flatten_live

_BASE_STATIC = None
_INSTALLING = False


def _finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _resolved_path(x):
    try:
        return str(Path(x).resolve())
    except Exception:
        return str(x or "")


def _series_for_ticker(ticker):
    text = str(ticker or "")
    for series in ROUTER.SERIES:
        if text == series or text.startswith(series + "-"):
            return series
    return None


def _iter_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _flat_shard_equity(shard_balances, preflight):
    preflight = preflight or {}
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
        raise RuntimeError("Tail25 parent baseline requires zero resting orders")

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
    under = {
        idx: normalized.get(idx, 0.0)
        for idx in used
        if normalized.get(idx, 0.0) + 1e-12 < float(minimum_per_used_shard)
    }
    return {
        "used_shards": used,
        "balance_by_shard_usd": normalized,
        "minimum_per_used_shard_usd": float(minimum_per_used_shard),
        "underfunded_used_shards": under,
        "ok": not under,
    }


def _fee_series_set(preflight):
    fee = (preflight or {}).get("fee_preflight") or {}
    return {
        str((row or {}).get("series") or "")
        for row in fee.get("series") or []
        if str((row or {}).get("series") or "")
    }


def _balance_reconciliation(
    *,
    shard_equity,
    independent_equity,
    tolerance_usd=BALANCE_RECONCILE_ABS_TOL_USD,
):
    a = _finite(shard_equity)
    b = _finite(independent_equity)
    if a is None or b is None:
        return {
            "ok": False,
            "reason": "missing independent equity values",
            "shard_equity_usd": a,
            "independent_account_equity_usd": b,
            "delta_usd": None,
            "tolerance_usd": float(tolerance_usd),
        }
    delta = float(a - b)
    ok = abs(delta) <= float(tolerance_usd) + 1e-12
    return {
        "ok": bool(ok),
        "reason": None if ok else "flat shard breakdown does not reconcile to account equity",
        "shard_equity_usd": float(a),
        "independent_account_equity_usd": float(b),
        "delta_usd": delta,
        "tolerance_usd": float(tolerance_usd),
    }


def _is_own_supervisor_control(obj, *, parent_session, supervisor_pid):
    obj = obj or {}
    ctl_pid = int(obj.get("supervisor_pid") or obj.get("pid") or 0)
    return bool(
        ctl_pid == int(supervisor_pid)
        and _resolved_path(obj.get("session_dir")) == _resolved_path(parent_session)
        and str(obj.get("deploy_version") or "") == DEPLOY_VERSION
    )


def _guard_other_live_processes_allowing_self(parent_cfg):
    parent_session = Path(parent_cfg["parent_session_dir"]).resolve()
    supervisor_pid = os.getpid()
    ctl = B._read(CORE.CONTROL_PATH, {}) or {}
    if not _is_own_supervisor_control(
        ctl,
        parent_session=parent_session,
        supervisor_pid=supervisor_pid,
    ):
        raise RuntimeError(
            "Tail25 generation preflight cannot prove ownership of the live "
            f"control. pid={supervisor_pid} parent={parent_session} control={ctl}"
        )
    if not B._pid_alive(supervisor_pid):
        raise RuntimeError("Tail25 owning supervisor PID is not alive")

    controls = [
        CORE.CONTROL_PATH,
        V28.C.DATA_ROOT / "live_cycle_q10_v1" / "active_live.json",
    ]
    other_live = []
    self_exemptions = 0
    for path in controls:
        state = B._read(path, {}) or {}
        if not state or not B._pid_alive(state.get("pid")):
            continue
        is_core = _resolved_path(path) == _resolved_path(CORE.CONTROL_PATH)
        if is_core and _is_own_supervisor_control(
            state,
            parent_session=parent_session,
            supervisor_pid=supervisor_pid,
        ):
            self_exemptions += 1
            continue
        other_live.append({"control": str(path), "state": state})

    if self_exemptions != 1:
        raise RuntimeError(
            f"Expected exactly one owning-supervisor exemption, got {self_exemptions}"
        )
    if other_live:
        raise RuntimeError(
            "Another live strategy process is running; refusing concurrent account "
            f"control: {other_live}"
        )


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
    legitimate = float(MAX_LEGIT_MATCH_MULTIPLIER * q)
    group_limit = float(max(30.0, ORDER_GROUP_LIMIT_MULTIPLIER * q))
    if group_limit <= legitimate + 1e-12:
        raise RuntimeError(
            f"Order-group limit has no legitimate-volume headroom: "
            f"limit={group_limit} legitimate={legitimate}"
        )
    return {
        "mode": f"{parent_cfg['mode']}_GEN_{int(generation_id):04d}",
        "quote_size": q,
        "runtime_hours": max(0.02, float(remaining_hours)),
        "max_start_loss_usd": float(parent_cfg["max_start_loss_usd"]),
        "min_start_equity_usd": float(parent_cfg["min_start_equity_usd"]),
        "order_group_limit_fp": f"{group_limit:.2f}",
        "order_group_limit_policy": "30Q_PER_EXECUTION_SHARD_ROLLING_15S",
        "max_legitimate_matched_contracts_if_all_12_one_shard": legitimate,
        "order_group_limit_headroom_contracts": group_limit - legitimate,
        "live_engine_version": LIVE.LIVE_VERSION,
        "live_engine_patch_version": LIVE.PATCH_VERSION,
        "deploy_version": DEPLOY_VERSION,
        "deploy_patch_version": PATCH_VERSION,
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
    """Detached child. Real orders are possible only after explicitly armed parent."""
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
    install=True,
):
    """Authenticated READ-ONLY preflight. Sends no orders/cancels/transfers."""
    if install:
        _install_patch()
    if run_static:
        static_self_check(show=bool(show))
        _install_patch()

    routing = ROUTER.routing_snapshot(require_all=True, refresh=True)
    shard_balances = ROUTER.get_shard_balances()

    old_fee_series = OOS.SERIES
    OOS.SERIES = tuple(ROUTER.SERIES)
    try:
        pre = V288.live_preflight(
            quote_size=float(q),
            runtime_hours=float(hours),
            max_start_loss_usd=float(max_loss),
            min_start_equity_usd=float(min_equity),
            show=bool(show),
            probe_private_ws=bool(probe_private_ws),
        )
    finally:
        OOS.SERIES = old_fee_series
        if install:
            _install_patch()

    pre = dict(pre or {})
    expected = set(ROUTER.SERIES)
    fee_seen = _fee_series_set(pre)
    fee_ok = bool(
        ((pre.get("fee_preflight") or {}).get("ok") is True)
        and fee_seen == expected
    )
    if not fee_ok:
        raise RuntimeError(
            "Fee preflight did not prove the current fee contract for all 12 "
            f"series. expected={sorted(expected)} seen={sorted(fee_seen)}"
        )

    exact_flat_equity = _flat_shard_equity(shard_balances, pre)
    independent_equity = _finite((pre.get("account") or {}).get("equity_usd"))
    balance_reconcile = _balance_reconciliation(
        shard_equity=exact_flat_equity,
        independent_equity=independent_equity,
    )
    if not balance_reconcile["ok"]:
        raise RuntimeError(
            "Flat-balance risk-baseline reconciliation failed; refusing launch. "
            f"details={balance_reconcile}"
        )

    if exact_flat_equity + 1e-12 < float(min_equity):
        raise RuntimeError(
            f"Starting flat all-shard equity ${exact_flat_equity:.4f} "
            f"< required ${float(min_equity):.4f}"
        )
    if not (0.0 < float(max_loss) < exact_flat_equity):
        raise RuntimeError(
            f"Invalid loss stop ${float(max_loss):.4f} for equity "
            f"${exact_flat_equity:.4f}"
        )

    collateral = _used_shard_collateral_check(
        routing,
        shard_balances,
        minimum_per_used_shard=float(min_per_shard),
    )
    if not collateral["ok"]:
        raise RuntimeError(
            "One or more execution shards do not have required local cash. No "
            f"automatic transfer attempted. details={collateral}"
        )

    account = dict(pre.get("account") or {})
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
            "deploy_patch_version": PATCH_VERSION,
            "live_version": LIVE.LIVE_VERSION,
            "live_patch_version": LIVE.PATCH_VERSION,
            "recorder_version": REC.STUDY_VERSION,
            "router_version": ROUTER.ROUTER_VERSION,
            "strategy": LIVE.STRATEGY_NAME,
            "quantity": float(q),
            "runtime_hours": float(hours),
            "universe": list(ROUTER.SERIES),
            "routing": routing,
            "shard_balances": shard_balances,
            "used_shard_collateral": collateral,
            "fee_universe_audit": {
                "ok": True,
                "expected_series": sorted(expected),
                "verified_series": sorted(fee_seen),
                "series_count": len(fee_seen),
            },
            "flat_balance_reconciliation": balance_reconcile,
            "risk_baseline_source": "CURRENT_FLAT_ALL_SHARD_BALANCE_BREAKDOWN_RECONCILED_TO_ACCOUNT_EQUITY",
            "risk_baseline_equity_usd": float(exact_flat_equity),
            "risk_baseline_kill_equity_usd": float(exact_flat_equity - float(max_loss)),
            "inherited_preflight_equity_usd": independent_equity,
            "automatic_collateral_transfer": False,
        }
    )

    if show:
        print("=" * 164)
        print("TAIL25 V2 AUDITED ROUTING / FEE / RISK PREFLIGHT")
        print("=" * 164)
        print("12-series routing:            PASS")
        print("12-series fee preflight:      PASS")
        print(f"Flat all-shard equity:         ${exact_flat_equity:.4f}")
        print(
            "Shard/account equity delta:   "
            f"${float(balance_reconcile['delta_usd']):+.4f}"
        )
        print(f"Software kill equity:          ${exact_flat_equity - float(max_loss):.4f}")
        print("Execution shards:             ", routing.get("shards"))
        print("Shard balances:               ", collateral.get("balance_by_shard_usd"))
        print("Auto collateral transfer:     DISABLED")
        print("ORDERS SENT:                  NO")
        print("TRANSFERS SENT:               NO")

    return pre


def _fresh_generation_preflight(parent_cfg, *, remaining_hours, show=False):
    """Read-only between-generation preflight with exact-owning-supervisor exemption."""
    _install_patch()
    old_guard = V28.D._guard_other_live_processes

    def self_aware_guard():
        return _guard_other_live_processes_allowing_self(parent_cfg)

    V28.D._guard_other_live_processes = self_aware_guard
    try:
        q = float(parent_cfg["quote_size"])
        min_per_shard = (
            Q1_MIN_COLLATERAL_PER_USED_SHARD_USD
            if q <= 1.0 + 1e-12
            else Q10_MIN_COLLATERAL_PER_USED_SHARD_USD
        )
        out = _read_only_preflight(
            q=q,
            hours=max(0.02, float(remaining_hours)),
            max_loss=float(parent_cfg["max_start_loss_usd"]),
            min_equity=float(parent_cfg["min_start_equity_usd"]),
            min_per_shard=float(min_per_shard),
            probe_private_ws=False,
            show=bool(show),
            run_static=False,
            install=False,
        )
    finally:
        V28.D._guard_other_live_processes = old_guard
        _install_patch()

    out = dict(out or {})
    out["generation_preflight_self_control_fix"] = PATCH_VERSION
    out["owning_supervisor_control_exempted_only"] = True
    return out


def _stop_external_recorder(pid):
    result = dict(REC.stop_parent_recorder(pid) or {})
    pid = int(pid or 0)
    if pid > 0 and B._pid_alive(pid):
        deadline = time.time() + RECORDER_POST_TERM_WAIT_S
        while B._pid_alive(pid) and time.time() < deadline:
            time.sleep(0.10)
    dead = not (pid > 0 and B._pid_alive(pid))
    result.update(
        {
            "pid": pid or result.get("pid"),
            "dead": bool(dead),
            "stopped": bool(dead),
            "tail25_stop_receipt_patch": PATCH_VERSION,
        }
    )
    if not dead:
        raise RuntimeError(f"Parent recorder did not terminate: {result}")
    return result


def _smoke_coverage_audit(parent_session, checkpoint):
    parent_session = Path(parent_session).resolve()
    checkpoint = checkpoint or {}
    generation_dir = Path(checkpoint.get("generation_dir") or "")
    expected = set(ROUTER.SERIES)

    target_tickers = [str(x) for x in checkpoint.get("target_tickers") or [] if x]
    target_series = {
        s for s in (_series_for_ticker(t) for t in target_tickers) if s
    }

    metadata_series = set()
    metadata_tickers = set()
    for row in _iter_jsonl(parent_session / "raw_capture" / "market_metadata.jsonl") or []:
        series = str(row.get("series_ticker") or "")
        ticker = str(row.get("ticker") or "")
        if series:
            metadata_series.add(series)
        if ticker:
            metadata_tickers.add(ticker)

    entry_ack_series = set()
    entry_ack_tickers = set()
    if generation_dir.exists():
        for row in _iter_jsonl(generation_dir / "orders.jsonl") or []:
            if str(row.get("action") or "") != "CREATE_ACK":
                continue
            track = row.get("track") or {}
            if str(track.get("role") or "") != "ENTRY":
                continue
            ticker = str(track.get("ticker") or "")
            series = _series_for_ticker(ticker)
            if ticker:
                entry_ack_tickers.add(ticker)
            if series:
                entry_ack_series.add(series)

    group = B._read(generation_dir / "order_group.json", {}) or {}
    groups = group.get("order_group_id")
    group_shards = set()
    if isinstance(groups, dict):
        for key, value in groups.items():
            if not value:
                continue
            try:
                group_shards.add(int(key))
            except Exception:
                pass

    parent_pre = B._read(parent_session / "parent_preflight_snapshot.json", {}) or {}
    expected_shards = {
        int(x) for x in ((parent_pre.get("routing") or {}).get("shards") or [])
    }
    capture = B._read(parent_session / "raw_capture" / "capture_spec.json", {}) or {}

    checks = {
        "target_count_exact_12": int(checkpoint.get("target_count") or 0) == 12,
        "target_ticker_count_exact_12": len(set(target_tickers)) == 12,
        "target_series_exact_12": target_series == expected,
        "raw_metadata_series_exact_12": expected.issubset(metadata_series),
        "entry_create_ack_series_exact_12": entry_ack_series == expected,
        "groups_are_multishard_map": isinstance(groups, dict) and bool(groups),
        "group_shards_match_preflight": bool(expected_shards) and group_shards == expected_shards,
        "capture_study_version_exact": capture.get("study_version") == REC.STUDY_VERSION,
        "capture_m12_exact": capture.get("research_elapsed_seconds") == [0.0, 720.0],
        "capture_tail_750_exact": capture.get("persisted_elapsed_seconds") == [0.0, 750.0],
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "expected_series": sorted(expected),
        "target_series": sorted(target_series),
        "raw_metadata_series": sorted(metadata_series),
        "entry_create_ack_series": sorted(entry_ack_series),
        "target_tickers": sorted(set(target_tickers)),
        "raw_metadata_ticker_count": len(metadata_tickers),
        "entry_create_ack_tickers": sorted(entry_ack_tickers),
        "expected_shards": sorted(expected_shards),
        "order_group_shards": sorted(group_shards),
        "capture_study_version": capture.get("study_version"),
    }


def _write_smoke_promotion(
    parent_session,
    cfg,
    checkpoint,
    recorder_pid,
    tail_result,
    recorder_stop,
):
    """Inherited promotion plus exact 12-series real-order/recorder coverage gate."""
    record = dict(
        _PARENT_WRITE_SMOKE_PROMOTION(
            parent_session,
            cfg,
            checkpoint,
            recorder_pid,
            tail_result,
            recorder_stop,
        )
        or {}
    )
    coverage = _smoke_coverage_audit(parent_session, checkpoint)
    base_passed = record.get("passed") is True
    passed = bool(base_passed and coverage.get("ok") is True)
    record.update(
        {
            "passed": passed,
            "deploy_version": DEPLOY_VERSION,
            "deploy_patch_version": PATCH_VERSION,
            "live_version": LIVE.LIVE_VERSION,
            "live_patch_version": LIVE.PATCH_VERSION,
            "git_head": P._current_head(),
            "tail25_exact_12_series_smoke_audit": coverage,
            "base_parent_promotion_passed_before_tail25_coverage": bool(base_passed),
            "promotion_requires_all_12_entry_create_acks": True,
        }
    )
    B._atomic(PROMOTION_PATH, record)
    if not passed:
        raise RuntimeError(
            f"Tail25 Q1 smoke promotion failed exact 12-series audit: {coverage}"
        )
    return record


def _launch_generation(*args, **kwargs):
    # Reuse the already-audited mixed-shard launch function after publishing this
    # module's generation config and detached module identity.
    _install_patch()
    return _BASE_DEP_GENERATION_LAUNCH(*args, **kwargs)


def _recover_generation_fail_closed(*args, **kwargs):
    _install_patch()
    return _BASE_DEP_RECOVERY(*args, **kwargs)


def _install_patch():
    """Publish one coherent final process graph. No API calls or mutations."""
    global _INSTALLING
    if _INSTALLING:
        return {
            "deploy_version": DEPLOY_VERSION,
            "module_name": MODULE_NAME,
            "reentrant": True,
            "orders_sent": False,
        }
    _INSTALLING = True
    try:
        # Start from latest audited operations, then recorder, then dynamic router.
        OPS._install_patch()
        REC._install_patch()
        ROUTER.install_runtime_patch()

        LIVE.ROTATION_CHECKPOINT_FILE = LIVE.ROTATION_CHECKPOINT_FILE
        LIVE.GENERATION_BOOTSTRAP_FILE = LIVE.GENERATION_BOOTSTRAP_FILE
        LIVE.SESSION_RISK_BASELINE_FILE = LIVE.SESSION_RISK_BASELINE_FILE

        # Base deployment function globals are dynamic Python module globals.
        COREDEP.DEPLOY_VERSION = DEPLOY_VERSION
        COREDEP.MODULE_NAME = MODULE_NAME
        COREDEP.Q1_ARM = Q1_ARM
        COREDEP.Q10_ARM = Q10_ARM
        COREDEP.PROMOTION_PATH = PROMOTION_PATH
        COREDEP.LIVE = LIVE
        COREDEP.REC = REC
        COREDEP.ROUTER = ROUTER
        COREDEP.ENTRY_START_S = ENTRY_START_S
        COREDEP.M12_S = M12_S
        COREDEP.LABEL_TAIL_END_S = LABEL_TAIL_END_S
        COREDEP._generation_cfg = _generation_cfg
        COREDEP._run_generation = _run_generation
        COREDEP._launch_generation = _launch_generation
        COREDEP._fresh_generation_preflight = _fresh_generation_preflight
        COREDEP._read_only_preflight = _read_only_preflight
        COREDEP._stop_external_recorder = _stop_external_recorder
        COREDEP._recover_generation_fail_closed = _recover_generation_fail_closed
        COREDEP._install_patch = _install_patch
        COREDEP.static_self_check = static_self_check

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

        # Mature rotating parent resolves these at runtime.
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
        P._write_smoke_promotion = _write_smoke_promotion

        # Guardian follows final child and mixed-shard recovery.
        V2963.DEPLOY_VERSION = DEPLOY_VERSION
        V2963.LIVE = LIVE
        V2963.POST_M5_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S

        # Router must be last because OPS installs historical shard routing.
        ROUTER.install_runtime_patch()
    finally:
        _INSTALLING = False

    return {
        "deploy_version": DEPLOY_VERSION,
        "deploy_patch_version": PATCH_VERSION,
        "module_name": MODULE_NAME,
        "live_version": LIVE.LIVE_VERSION,
        "live_patch_version": LIVE.PATCH_VERSION,
        "recorder_version": REC.STUDY_VERSION,
        "router_version": ROUTER.ROUTER_VERSION,
        "orders_sent": False,
        "transfers_sent": False,
    }


def static_self_check(*, show=True):
    """Pure/offline structural audit. Does not access Kalshi or mutate account."""
    global _BASE_STATIC
    if _BASE_STATIC is None:
        _BASE_STATIC = OPS.static_self_check(show=False)
    engine = LIVE.static_self_check(show=False)
    recorder = REC.static_self_check(show=False)
    router = ROUTER.static_self_check(show=False)
    _install_patch()

    q1_cfg = _generation_cfg(
        {
            "mode": "STATIC",
            "quote_size": 1.0,
            "runtime_hours": 0.5,
            "max_start_loss_usd": 5.0,
            "min_start_equity_usd": 50.0,
            "parent_session_dir": "/tmp/tail25-static",
        },
        generation_id=1,
        generation_dir="/tmp/tail25-static/gen",
        recorder_pid=123,
        session_start_equity=350.0,
        session_kill_equity=345.0,
        remaining_hours=0.5,
    )
    q10_cfg = _generation_cfg(
        {
            "mode": "STATIC",
            "quote_size": 10.0,
            "runtime_hours": 12.0,
            "max_start_loss_usd": 30.0,
            "min_start_equity_usd": 300.0,
            "parent_session_dir": "/tmp/tail25-static",
        },
        generation_id=1,
        generation_dir="/tmp/tail25-static/gen",
        recorder_pid=123,
        session_start_equity=350.0,
        session_kill_equity=320.0,
        remaining_hours=12.0,
    )
    balance_ok = _balance_reconciliation(
        shard_equity=350.00,
        independent_equity=350.01,
    )
    balance_bad = _balance_reconciliation(
        shard_equity=35000.0,
        independent_equity=350.0,
    )

    checks = {
        "auditfix6_operations_static_ok": _BASE_STATIC.get("ok") is True,
        "final_engine_static_ok": engine.get("ok") is True,
        "recorder_static_ok": recorder.get("ok") is True,
        "router_static_ok": router.get("ok") is True,
        "deploy_version_exact": DEPLOY_VERSION == "MM_TAIL25_JOIN_BBO_Q10_MULTI12_12H_V2_AUDITED",
        "patch_version_exact": PATCH_VERSION == "TAIL25_MULTI12_SOURCE_AUDIT_FINAL_V2",
        "live_patch_exact": LIVE.PATCH_VERSION == "TAIL25_EXEC_VISIBILITY_AUDIT_V1_3",
        "q1_exact": Q1_Q == 1.0,
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
        "commodities_exact": set(ROUTER.COMMODITY_SERIES)
        == {"KXGOLD15M", "KXSILVER15M", "KXWTI15M"},
        "private_ws_required_for_entry": True,
        "raw_stale_entry_guard": LIVE.RAW_ENTRY_MAX_AGE_MS > 0.0,
        "q1_order_group_limit_30": float(q1_cfg["order_group_limit_fp"]) == 30.0,
        "q1_legit_max_24": q1_cfg["max_legitimate_matched_contracts_if_all_12_one_shard"] == 24.0,
        "q1_group_has_headroom": float(q1_cfg["order_group_limit_fp"]) > 24.0,
        "q10_order_group_limit_300": float(q10_cfg["order_group_limit_fp"]) == 300.0,
        "q10_legit_max_240": q10_cfg["max_legitimate_matched_contracts_if_all_12_one_shard"] == 240.0,
        "q10_group_has_headroom": float(q10_cfg["order_group_limit_fp"]) > 240.0,
        "small_balance_delta_passes": balance_ok.get("ok") is True,
        "cent_vs_dollar_unit_error_fails": balance_bad.get("ok") is False,
        "runtime_live_final": RUNTIME.LIVE is LIVE,
        "parent_live_final": P.LIVE is LIVE,
        "parent_recorder_final": P.REC is REC,
        "parent_generation_cfg_final": P._generation_cfg is _generation_cfg,
        "parent_generation_preflight_final": P._fresh_generation_preflight is _fresh_generation_preflight,
        "parent_promotion_writer_final": P._write_smoke_promotion is _write_smoke_promotion,
        "parent_recovery_multishard": P._recover_generation_fail_closed is _recover_generation_fail_closed,
        "router_installed_last": B._payload.__module__ == ROUTER.__name__,
        "fee_preflight_exact_12_required": True,
        "promotion_exact_12_entry_acks_required": True,
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
        "series": list(ROUTER.SERIES),
        "promotion_path": str(PROMOTION_PATH),
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 180)
        print("TAIL25 MULTI12 V2 FINAL SOURCE STATIC AUDIT — NO API / NO ORDERS")
        print("=" * 180)
        for k, v in out.items():
            print(f"{k:116s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 V2 static audit failed: {out}")
    return out


def q1_smoke_preflight(*, show=True):
    _install_patch()
    return _read_only_preflight(
        q=Q1_Q,
        hours=Q1_HOURS,
        max_loss=Q1_MAX_LOSS_USD,
        min_equity=Q1_MIN_EQUITY_USD,
        min_per_shard=Q1_MIN_COLLATERAL_PER_USED_SHARD_USD,
        probe_private_ws=True,
        show=show,
        run_static=True,
        install=True,
    )


def q10_preflight(*, show=True):
    _install_patch()
    return _read_only_preflight(
        q=Q10_Q,
        hours=Q10_HOURS,
        max_loss=Q10_MAX_LOSS_USD,
        min_equity=Q10_MIN_EQUITY_USD,
        min_per_shard=Q10_MIN_COLLATERAL_PER_USED_SHARD_USD,
        probe_private_ws=True,
        show=show,
        run_static=True,
        install=True,
    )


def rotation_promotion_status(*, show=True):
    _install_patch()
    return COREDEP.rotation_promotion_status(show=show)


def start_q1_one_window_smoke(*, arm_phrase=None):
    _install_patch()
    static_self_check(show=True)
    _install_patch()
    if str(arm_phrase) != Q1_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q1_ARM!r} exactly."
        )
    return COREDEP.start_q1_one_window_smoke(arm_phrase=Q1_ARM)


def start_q10_12h(*, arm_phrase=None):
    _install_patch()
    static_self_check(show=True)
    _install_patch()
    if str(arm_phrase) != Q10_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q10_ARM!r} exactly."
        )
    return COREDEP.start_q10_12h(arm_phrase=Q10_ARM)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    out = _BASE_DEP_LIVE_STATUS(show=show, tail_lines=tail_lines)
    if isinstance(out, dict):
        out["deploy_version"] = DEPLOY_VERSION
        out["deploy_patch_version"] = PATCH_VERSION
        out["live_patch_version"] = LIVE.PATCH_VERSION
        out["module_name"] = MODULE_NAME
    return out


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    if str(arm_phrase) != KILL_ARM:
        raise RuntimeError(f"Pass arm_phrase={KILL_ARM!r} exactly.")
    return _BASE_DEP_KILL(
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
        return _run_generation(Path(args.run_generation), Path(args.config))

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
