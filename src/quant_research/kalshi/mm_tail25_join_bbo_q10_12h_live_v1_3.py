from __future__ import annotations

"""Tail25 Multi12 V1.3 intended operator entrypoint.

Final audit binding:
- V1.3 execution-visibility child engine;
- V1.2 12-series fee and flat-balance reconciliation gates;
- V1.1 owning-supervisor generation-preflight exemption;
- V7 combined crypto/commodity recorder;
- dynamic per-market exchange routing and multi-shard order groups;
- AUDITFIX6 supervisor/guardian/authoritative cleanup.

The order-group rolling 15-second matched-contract limit is raised from the old
``20*Q`` single-universe default to ``30*Q``.  With 12 series and no re-entry,
the maximum legitimate entry+exit matched volume is 24*Q contracts even if all
series share one shard.  30*Q therefore leaves operational headroom while still
providing an exchange-side runaway-volume backstop.

No alpha parameter changes. Importing performs no API calls or account mutation.
"""

import argparse
from pathlib import Path

from . import mm_tail25_join_bbo_live_v1_3_audit as LIVE
from . import mm_tail25_join_bbo_q10_12h_live_v1_2 as V12


V1 = V12.V1
V11 = V12.V11
REC = V12.REC
ROUTER = V12.ROUTER
P = V12.P
RUNTIME = V12.RUNTIME
B = V12.B

DEPLOY_VERSION = V12.DEPLOY_VERSION
PATCH_VERSION = "TAIL25_MULTI12_FINAL_AUDIT_V1_3"
MODULE_NAME = "quant_research.kalshi.mm_tail25_join_bbo_q10_12h_live_v1_3"

Q1_ARM = "LIVE_TAIL25_MULTI12_Q1_ONE_WINDOW_V1_3"
Q10_ARM = "LIVE_TAIL25_MULTI12_Q10_12H_V1_3"
KILL_ARM = V12.KILL_ARM

Q1_Q = V12.Q1_Q
Q10_Q = V12.Q10_Q
Q10_HOURS = V12.Q10_HOURS
Q10_MAX_LOSS_USD = V12.Q10_MAX_LOSS_USD
Q10_MIN_EQUITY_USD = V12.Q10_MIN_EQUITY_USD
PROMOTION_PATH = V12.PROMOTION_PATH

ORDER_GROUP_LIMIT_MULTIPLIER = 30.0
MAX_LEGIT_MATCH_MULTIPLIER = 2.0 * len(ROUTER.SERIES)  # entry + liquidation per series

_ORIGINAL_V12_INSTALL = V12._install_patch
_ORIGINAL_GENERATION_CFG = V1._generation_cfg


def _generation_cfg_final(
    parent_cfg,
    *,
    generation_id,
    generation_dir,
    recorder_pid,
    session_start_equity,
    session_kill_equity,
    remaining_hours,
):
    cfg = _ORIGINAL_GENERATION_CFG(
        parent_cfg,
        generation_id=generation_id,
        generation_dir=generation_dir,
        recorder_pid=recorder_pid,
        session_start_equity=session_start_equity,
        session_kill_equity=session_kill_equity,
        remaining_hours=remaining_hours,
    )
    q = float(parent_cfg["quote_size"])
    legitimate = float(MAX_LEGIT_MATCH_MULTIPLIER * q)
    group_limit = float(max(30.0, ORDER_GROUP_LIMIT_MULTIPLIER * q))
    if group_limit <= legitimate + 1e-12:
        raise RuntimeError(
            f"Tail25 order-group limit has no legitimate-volume headroom: "
            f"limit={group_limit} legitimate={legitimate}"
        )
    cfg.update(
        {
            "order_group_limit_fp": f"{group_limit:.2f}",
            "order_group_limit_policy": "30Q_PER_EXECUTION_SHARD_ROLLING_15S",
            "max_legitimate_matched_contracts_if_all_12_one_shard": legitimate,
            "order_group_limit_headroom_contracts": group_limit - legitimate,
            "live_engine_patch_version": LIVE.PATCH_VERSION,
            "deploy_patch_version": PATCH_VERSION,
        }
    )
    return cfg


def _publish_final_bindings():
    V1.LIVE = LIVE
    V1.MODULE_NAME = MODULE_NAME
    V1.Q1_ARM = Q1_ARM
    V1.Q10_ARM = Q10_ARM
    V1._generation_cfg = _generation_cfg_final

    V11.MODULE_NAME = MODULE_NAME
    V11.Q1_ARM = Q1_ARM
    V11.Q10_ARM = Q10_ARM

    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q10_ARM
    RUNTIME.LIVE = LIVE
    RUNTIME._generation_cfg = _generation_cfg_final

    P.LIVE = LIVE
    P.REC = REC
    P._generation_cfg = _generation_cfg_final
    P._launch_generation = V1._launch_generation
    P._recover_generation_fail_closed = V1._recover_generation_fail_closed
    P._start_external_recorder = V1._start_external_recorder
    P._stop_external_recorder = V12._stop_external_recorder_compat
    P._fresh_generation_preflight = V11._fresh_generation_preflight


def _install_patch():
    _ORIGINAL_V12_INSTALL()
    _publish_final_bindings()
    V1._install_patch = _install_patch
    V11._install_patch = _install_patch
    V12.V1._install_patch = _install_patch
    return {
        "deploy_version": DEPLOY_VERSION,
        "deploy_patch_version": PATCH_VERSION,
        "live_patch_version": LIVE.PATCH_VERSION,
        "module_name": MODULE_NAME,
        "order_group_limit_multiplier": ORDER_GROUP_LIMIT_MULTIPLIER,
        "orders_sent": False,
        "transfers_sent": False,
    }


def static_self_check(*, show=True):
    _install_patch()
    deployment = V12.static_self_check(show=False)
    engine = LIVE.static_self_check(show=False)

    q1_cfg = _generation_cfg_final(
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
    q10_cfg = _generation_cfg_final(
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

    q1_limit = float(q1_cfg["order_group_limit_fp"])
    q10_limit = float(q10_cfg["order_group_limit_fp"])
    checks = {
        "v12_deployment_audit_ok": deployment.get("ok") is True,
        "v13_visibility_engine_ok": engine.get("ok") is True,
        "deploy_version_preserved": DEPLOY_VERSION == V1.DEPLOY_VERSION,
        "patch_version_exact": PATCH_VERSION == "TAIL25_MULTI12_FINAL_AUDIT_V1_3",
        "live_patch_exact": LIVE.PATCH_VERSION == "TAIL25_EXEC_VISIBILITY_AUDIT_V1_3",
        "q10_exact": Q10_Q == 10.0,
        "runtime_12h_exact": Q10_HOURS == 12.0,
        "loss_stop_30_exact": Q10_MAX_LOSS_USD == 30.0,
        "min_equity_300_exact": Q10_MIN_EQUITY_USD == 300.0,
        "universe_12_exact": len(ROUTER.SERIES) == 12,
        "strategy_25_2_15_join2_force3": (
            LIVE.ENTRY_OFFSET == 0.25
            and LIVE.ENTRY_REPRICE_HYSTERESIS == 0.02
            and LIVE.EDGE_ZONE == 0.15
            and LIVE.EXIT_REPRICE_HYSTERESIS == 0.02
            and LIVE.EXIT_HORIZON_S == 3.0
        ),
        "private_ws_entry_guard": True,
        "raw_stale_entry_guard": LIVE.RAW_ENTRY_MAX_AGE_MS > 0.0,
        "q1_group_limit_30": q1_limit == 30.0,
        "q1_legitimate_max_24": MAX_LEGIT_MATCH_MULTIPLIER * 1.0 == 24.0,
        "q1_group_has_headroom": q1_limit > 24.0,
        "q10_group_limit_300": q10_limit == 300.0,
        "q10_legitimate_max_240": MAX_LEGIT_MATCH_MULTIPLIER * 10.0 == 240.0,
        "q10_group_has_headroom": q10_limit > 240.0,
        "parent_generation_cfg_final": P._generation_cfg is _generation_cfg_final,
        "runtime_generation_cfg_final": RUNTIME._generation_cfg is _generation_cfg_final,
        "runtime_live_final": RUNTIME.LIVE is LIVE,
        "parent_live_final": P.LIVE is LIVE,
        "q1_same_head_promotion_still_required": True,
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
        "order_group_limit_multiplier": ORDER_GROUP_LIMIT_MULTIPLIER,
        "max_legit_match_multiplier": MAX_LEGIT_MATCH_MULTIPLIER,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 172)
        print("TAIL25 MULTI12 V1.3 FINAL AUDIT — NO API / NO ORDERS")
        print("=" * 172)
        for k, v in out.items():
            print(f"{k:112s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 V1.3 final audit failed: {out}")
    return out


def q1_smoke_preflight(*, show=True):
    _install_patch()
    out = V12._hardened_read_only_preflight(
        q=V1.Q1_Q,
        hours=V1.Q1_HOURS,
        max_loss=V1.Q1_MAX_LOSS_USD,
        min_equity=V1.Q1_MIN_EQUITY_USD,
        min_per_shard=V1.Q1_MIN_COLLATERAL_PER_USED_SHARD_USD,
        probe_private_ws=True,
        show=show,
        run_static=False,
    )
    out = dict(out or {})
    out["deploy_patch_version"] = PATCH_VERSION
    out["live_patch_version"] = LIVE.PATCH_VERSION
    return out


def q10_preflight(*, show=True):
    _install_patch()
    out = V12._hardened_read_only_preflight(
        q=V1.Q10_Q,
        hours=V1.Q10_HOURS,
        max_loss=V1.Q10_MAX_LOSS_USD,
        min_equity=V1.Q10_MIN_EQUITY_USD,
        min_per_shard=V1.Q10_MIN_COLLATERAL_PER_USED_SHARD_USD,
        probe_private_ws=True,
        show=show,
        run_static=False,
    )
    out = dict(out or {})
    out["deploy_patch_version"] = PATCH_VERSION
    out["live_patch_version"] = LIVE.PATCH_VERSION
    return out


def rotation_promotion_status(*, show=True):
    _install_patch()
    return V1.rotation_promotion_status(show=show)


def start_q1_one_window_smoke(*, arm_phrase=None):
    _install_patch()
    if str(arm_phrase) != Q1_ARM:
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q1_ARM!r} exactly."
        )
    # V1 globals were republished above, so its mature launcher spawns this exact
    # module for supervisor/generation/guardian detached processes.
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
        return V1._run_generation(Path(args.run_generation), Path(args.config))

    if args.run_supervisor:
        if not args.config:
            raise RuntimeError("--config is required with --run-supervisor")
        _install_patch()
        return P._run_supervisor(Path(args.run_supervisor), Path(args.config))

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
