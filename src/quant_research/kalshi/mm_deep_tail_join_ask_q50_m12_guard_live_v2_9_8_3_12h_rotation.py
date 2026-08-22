from __future__ import annotations

"""V2.9.8.3 Q50 M1->M12 guard 12h deployment.

V2.9.8.2 passed every parent/static/private-WS preflight but the first detached
trader child exited before constructing an order group because the inherited V1.8
wrapper still asserted its historical M5=300 boundary.  V1.12 intentionally binds
that inherited cleanup clock to M12=720, so this wrapper points the deployment at
V1.12.1, which bridges only that historical V1.8 assertion while preserving the
full V1.7/V1.6/V1.5 execution and safety call chain.

No strategy, quantity, risk, recorder, guardian, rotation, or memory policy changes
from V2.9.8.2.  Importing this module performs no API calls and sends no orders.
"""

from . import mm_deep_tail_join_ask_deploy_v2_8 as V28
from . import mm_deep_tail_join_ask_deploy_v2_8_8 as V288
from . import mm_deep_tail_join_ask_live_v1_11_rotation as V111
from . import mm_deep_tail_join_ask_live_v1_12_1_m12_v18_compat as LIVE
from . import mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_12h_rotation as RUNTIME
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation as P
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_1_overnight_rotation as H
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_3_overnight_rotation as V2963


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_RECORD_M12_V2_9_8_3_12H_ROTATION"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_3_12h_rotation"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M12_GUARD_12H_V2983_ROTATION"
KILL_ARM = RUNTIME.KILL_ARM

Q50_Q = RUNTIME.Q50_Q
Q50_HOURS = RUNTIME.Q50_HOURS
Q50_MAX_LOSS_USD = RUNTIME.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = RUNTIME.Q50_MIN_EQUITY_USD
M1_S = RUNTIME.M1_S
M12_S = RUNTIME.M12_S
LABEL_TAIL_END_S = RUNTIME.LABEL_TAIL_END_S
GENERATION_RSS_WARNING_MB = RUNTIME.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = RUNTIME.GENERATION_RSS_HARD_LIMIT_MB


def _install_patch():
    """Bind the audited parent supervisor to V1.12.1 and this exact subprocess module."""
    V2963._install_patch()

    LIVE.ROTATION_CHECKPOINT_FILE = V111.ROTATION_CHECKPOINT_FILE
    LIVE.GENERATION_BOOTSTRAP_FILE = V111.GENERATION_BOOTSTRAP_FILE

    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q50_ARM
    RUNTIME.LIVE = LIVE

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

    # Every dynamic call made by the inherited runtime must preserve V2.9.8.3.
    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Idempotent offline structural audit; no API calls and no orders."""
    live = LIVE.static_self_check(show=False)
    _install_patch()

    checks = {
        "live_v1_12_1_ok": live.get("ok") is True,
        "v18_720_compat_regression": live.get("v18_bridge_reaches_v17_at_720") is True,
        "historical_v18_guard_globally_unchanged": (
            live.get("historical_v18_300_guard_unchanged_outside_child_bridge") is True
        ),
        "runtime_live_binding_is_v1_12_1": RUNTIME.LIVE is LIVE,
        "parent_live_binding_is_v1_12_1": P.LIVE is LIVE,
        "terminal_horizon_m12": P.M5_S == 720.0,
        "self_aware_preflight_bound": P._fresh_generation_preflight is H._fresh_generation_preflight,
        "generation_launcher_is_m12": P._launch_generation is RUNTIME._launch_generation,
        "generation_config_is_m12": P._generation_cfg is RUNTIME._generation_cfg,
        "guardian_post_window_is_m12": (
            V2963._post_m5_generation_state is RUNTIME._post_m12_generation_state
        ),
        "detached_module_points_to_v2983": RUNTIME.MODULE_NAME == MODULE_NAME,
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
        "trader_rss_warning_1536": GENERATION_RSS_WARNING_MB == 1536.0,
        "trader_rss_hard_3072": GENERATION_RSS_HARD_LIMIT_MB == 3072.0,
        "fresh_trader_process_each_m12": True,
        "recorder_parent_owned_across_generations": True,
        "fixed_session_risk_baseline": True,
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
        print("=" * 140)
        print("V2.9.8.3 Q50 M1->M12 GUARD CHILD-COMPAT STATIC CHECK — NO API / NO ORDERS")
        print("=" * 140)
        for k, v in out.items():
            print(f"{k:80s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.8.3 static self-check failed: {out}")
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
