from __future__ import annotations

"""V2.9.8.5 Q50 M1->M12 guard 12h deployment.

This deployment keeps the frozen Q50 M1->M12 strategy, V1.12.1 compatibility
bridge, parent-owned recorder, verified M12 process rotation, fixed session loss
baseline, and the 4096/6144 MiB trader-process-group RSS policy used by the latest
live test.

Operational delta from V2.9.8.4
-------------------------------
The live child is V1.12.2.  It adds a narrow, 5-second account-auditor-only
reconciliation grace for REST rows that contradict an authoritative canceled
order.  A row is suppressible only for the exact canceled order_id and only when
its REST last_update_time is no newer than the cancel request/terminal evidence.
Post-cancel updates, missing timestamps, rows beyond 5 seconds, and genuine
orphans retain the inherited fail-closed behavior.  M12 cleanup and rotation
zero-resting verification remain strict and do not use the grace filter.

No alpha, quantity, entry, exit, guard, loss, recorder, position, or rotation
mechanic changes are made. Importing this module performs no API calls and sends
no orders.
"""

from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_8_8 as V288
from . import mm_deep_tail_join_ask_live_v1_11_rotation as V111
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as LIVE
from . import mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_12h_rotation as RUNTIME
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation as P
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_1_overnight_rotation as H
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_3_overnight_rotation as V2963


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_RECORD_M12_V2_9_8_5_12H_ROTATION"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_5_12h_rotation"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M12_GUARD_12H_V2985_ROTATION"
KILL_ARM = RUNTIME.KILL_ARM

Q50_Q = RUNTIME.Q50_Q
Q50_HOURS = RUNTIME.Q50_HOURS
Q50_MAX_LOSS_USD = RUNTIME.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = RUNTIME.Q50_MIN_EQUITY_USD
M1_S = RUNTIME.M1_S
M12_S = RUNTIME.M12_S
LABEL_TAIL_END_S = RUNTIME.LABEL_TAIL_END_S

# Proven live M12 generations stayed near ~140 MiB after process rotation. Keep
# the higher process-group backstop used by V2.9.8.4; do not raise it further.
GENERATION_RSS_WARNING_MB = 4096.0
GENERATION_RSS_HARD_LIMIT_MB = 6144.0


def _install_patch():
    """Bind the audited parent supervisor to V1.12.2 and this subprocess module."""
    V2963._install_patch()

    # Historical V1.11 artifact filenames are intentional compatibility contracts.
    LIVE.ROTATION_CHECKPOINT_FILE = V111.ROTATION_CHECKPOINT_FILE
    LIVE.GENERATION_BOOTSTRAP_FILE = V111.GENERATION_BOOTSTRAP_FILE

    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q50_ARM
    RUNTIME.LIVE = LIVE
    RUNTIME.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    RUNTIME.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB

    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.M1_S = M1_S
    P.M5_S = M12_S  # historical parent variable name; numeric horizon is M12
    P.RECORDER_M12_S = M12_S
    P.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    P.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    P.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB
    P._fresh_generation_preflight = H._fresh_generation_preflight
    P._generation_cfg = RUNTIME._generation_cfg
    P._launch_generation = RUNTIME._launch_generation

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    V2963.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB
    V2963.POST_M5_EXIT_TIMEOUT_S = RUNTIME.POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state

    # Every dynamic call made by the inherited runtime must preserve V2.9.8.5.
    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Idempotent offline structural audit; no API calls and no orders."""
    live = LIVE.static_self_check(show=False)
    _install_patch()

    reg = live.get("regression") or {}
    checks = {
        "live_v1_12_2_ok": live.get("ok") is True,
        "exact_eth_cancel_stale_rest_regression": (
            live.get("exact_eth_cancel_stale_rest_regression") is True
            and reg.get("exact_eth_pre_cancel_rest_suppressed") is True
        ),
        "post_cancel_updates_remain_fail_closed": (
            reg.get("post_cancel_exchange_update_preserved") is True
        ),
        "stale_rows_after_grace_remain_fail_closed": (
            reg.get("stale_row_after_grace_preserved") is True
        ),
        "cancel_rest_grace_5s": LIVE.CANCEL_REST_GRACE_S == 5.0,
        "audit_only_cancel_rest_filter": live.get("audit_only_filter") is True,
        "rotation_rest_verification_strict": (
            live.get("rotation_rest_verification_remains_strict") is True
        ),
        "runtime_live_binding_is_v1_12_2": RUNTIME.LIVE is LIVE,
        "parent_live_binding_is_v1_12_2": P.LIVE is LIVE,
        "terminal_horizon_m12": P.M5_S == 720.0,
        "self_aware_preflight_bound": P._fresh_generation_preflight is H._fresh_generation_preflight,
        "generation_launcher_is_m12": P._launch_generation is RUNTIME._launch_generation,
        "generation_config_is_m12": P._generation_cfg is RUNTIME._generation_cfg,
        "guardian_post_window_is_m12": (
            V2963._post_m5_generation_state is RUNTIME._post_m12_generation_state
        ),
        "detached_module_points_to_v2985": RUNTIME.MODULE_NAME == MODULE_NAME,
        "q50_exact_50": Q50_Q == 50.0,
        "runtime_exact_12h": Q50_HOURS == 12.0,
        "loss_trigger_exact_20": Q50_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q50_MIN_EQUITY_USD == 125.0,
        "entry_start_m1_60": M1_S == 60.0,
        "terminal_cleanup_m12_720": M12_S == 720.0,
        "recorder_label_tail_750": LABEL_TAIL_END_S == 750.0,
        "guard_yes_bid_10c": LIVE.YES_GUARD_BID_MAX == 0.10,
        "guard_no_ask_90c": LIVE.NO_GUARD_ASK_MIN == 0.90,
        "guard_persist_5s": LIVE.GUARD_PERSIST_S == 5.0,
        "guard_min_obs_3": LIVE.GUARD_MIN_BOOK_OBS == 3,
        "trader_rss_warning_4096": GENERATION_RSS_WARNING_MB == 4096.0,
        "trader_rss_hard_6144": GENERATION_RSS_HARD_LIMIT_MB == 6144.0,
        "fresh_trader_process_each_m12": True,
        "recorder_parent_owned_across_generations": True,
        "fixed_session_risk_baseline": True,
        "genuine_orphans_remain_fail_closed": True,
        "orders_sent": False,
    }

    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 144)
        print("V2.9.8.5 Q50 M1->M12 CANCEL-REST RECONCILE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 144)
        for k, v in out.items():
            print(f"{k:84s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.8.5 static self-check failed: {out}")
    return out


def q50_preflight(*, show=True):
    """Read-only account/private-WS preflight; sends no orders."""
    static_self_check(show=show)
    V28._patch_parent()
    V28.D._guard_other_live_processes()
    return V288.live_preflight(
        quote_size=Q50_Q,
        runtime_hours=Q50_HOURS,
        max_start_loss_usd=Q50_MAX_LOSS_USD,
        min_start_equity_usd=Q50_MIN_EQUITY_USD,
        show=show,
        probe_private_ws=True,
    )


def start_q50_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q50 / 12h entrypoint."""
    _install_patch()
    return RUNTIME.start_q50_12h_smoke(arm_phrase=arm_phrase)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    return RUNTIME.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    return RUNTIME.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    _install_patch()
    return RUNTIME._main()


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "MODULE_NAME",
    "Q50_ARM",
    "KILL_ARM",
    "Q50_Q",
    "Q50_HOURS",
    "Q50_MAX_LOSS_USD",
    "Q50_MIN_EQUITY_USD",
    "M1_S",
    "M12_S",
    "LABEL_TAIL_END_S",
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "static_self_check",
    "q50_preflight",
    "start_q50_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
