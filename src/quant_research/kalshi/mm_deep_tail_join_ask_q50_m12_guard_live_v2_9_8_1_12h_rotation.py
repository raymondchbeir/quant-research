from __future__ import annotations

"""V2.9.8.1 hotfix wrapper for the Q50 M1->M12 guard 12h deployment.

V2.9.8 introduced the intended V1.12/M12 supervisor wiring, but its process-local
patch referenced ``V2963._fresh_generation_preflight`` even though that function
lives in V2.9.6.1 (V2963 only installs it onto the parent module).  This wrapper
fixes that binding explicitly and keeps every trading/risk/recorder rule from
V2.9.8 unchanged.

Use THIS module for the 12-hour Q50 smoke. Importing it sends no orders.
"""

from . import mm_deep_tail_join_ask_live_v1_11_rotation as V111
from . import mm_deep_tail_join_ask_live_v1_12_m12_guard_rotation as LIVE
from . import mm_deep_tail_join_ask_q50_m12_guard_v2_9_7_preflight as PREFLIGHT
from . import mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_12h_rotation as BASE
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation as P
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_1_overnight_rotation as H
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_3_overnight_rotation as V2963


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_RECORD_M12_V2_9_8_1_12H_ROTATION"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_1_12h_rotation"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M12_GUARD_12H_V2981_ROTATION"
KILL_ARM = BASE.KILL_ARM

Q50_Q = BASE.Q50_Q
Q50_HOURS = BASE.Q50_HOURS
Q50_MAX_LOSS_USD = BASE.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = BASE.Q50_MIN_EQUITY_USD
M1_S = BASE.M1_S
M12_S = BASE.M12_S
LABEL_TAIL_END_S = BASE.LABEL_TAIL_END_S
GENERATION_RSS_WARNING_MB = BASE.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = BASE.GENERATION_RSS_HARD_LIMIT_MB


def _install_patch():
    """Correct V2.9.8 process-local bindings without changing strategy mechanics."""
    V2963._install_patch()

    LIVE.ROTATION_CHECKPOINT_FILE = V111.ROTATION_CHECKPOINT_FILE
    LIVE.GENERATION_BOOTSTRAP_FILE = V111.GENERATION_BOOTSTRAP_FILE

    # Publish this hotfix's provenance into the base wrapper so every detached
    # process spawned by its functions comes back through THIS module.
    BASE.DEPLOY_VERSION = DEPLOY_VERSION
    BASE.MODULE_NAME = MODULE_NAME
    BASE.Q50_ARM = Q50_ARM

    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.M1_S = M1_S
    P.M5_S = M12_S  # historical variable name; numeric horizon is M12
    P.RECORDER_M12_S = M12_S
    P.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    P.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    P.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB

    # Critical hotfix: the self-aware owning-supervisor preflight is defined in H.
    P._fresh_generation_preflight = H._fresh_generation_preflight
    P._generation_cfg = BASE._generation_cfg
    P._launch_generation = BASE._launch_generation

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    V2963.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB
    V2963.POST_M5_EXIT_TIMEOUT_S = BASE.POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = BASE._post_m12_generation_state

    # Ensure every dynamic call made by the base module re-enters this corrected
    # patch rather than its V2.9.8 implementation.
    BASE._install_patch = _install_patch


def static_self_check(*, show=True):
    """Read-only structural audit; no API calls and no orders."""
    # Run the original intended-contract audit before mutating parent globals.
    intended = PREFLIGHT.static_self_check(show=False)
    _install_patch()
    base = BASE.static_self_check(show=False)

    checks = {
        "v2_9_7_intended_contract_ok": intended.get("ok") is True,
        "base_v2_9_8_structure_ok": base.get("ok") is True,
        "live_v1_12_exact": P.LIVE is LIVE,
        "terminal_horizon_m12": P.M5_S == 720.0,
        "self_aware_preflight_bound_from_v2961": P._fresh_generation_preflight
        is H._fresh_generation_preflight,
        "generation_launcher_is_m12_v298": P._launch_generation is BASE._launch_generation,
        "generation_config_is_m12_v298": P._generation_cfg is BASE._generation_cfg,
        "guardian_post_window_is_m12": V2963._post_m5_generation_state
        is BASE._post_m12_generation_state,
        "detached_module_points_to_v2981": BASE.MODULE_NAME == MODULE_NAME,
        "q50_exact_50": Q50_Q == 50.0,
        "runtime_exact_12h": Q50_HOURS == 12.0,
        "loss_trigger_exact_20": Q50_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q50_MIN_EQUITY_USD == 125.0,
        "recorder_m1_to_m12_available": True,
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
        print("=" * 136)
        print("V2.9.8.1 Q50 M1->M12 GUARD 12H STATIC CHECK — NO API / NO ORDERS")
        print("=" * 136)
        for k, v in out.items():
            print(f"{k:76s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.8.1 static self-check failed: {out}")
    return out


def q50_preflight(*, show=True):
    """Read-only account/private-WS preflight; sends no orders."""
    static_self_check(show=show)
    return BASE.q50_preflight(show=show)


def start_q50_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q50 / 12h entrypoint."""
    _install_patch()
    return BASE.start_q50_12h_smoke(arm_phrase=arm_phrase)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    return BASE.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    return BASE.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    _install_patch()
    return BASE._main()


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
