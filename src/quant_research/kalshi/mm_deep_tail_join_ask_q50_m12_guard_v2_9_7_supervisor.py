from __future__ import annotations

"""Internal supervisor/guardian integration for the frozen Q50 M1->M12 guard.

This is the final thin source-preparation layer above the validated V2.9.7
runtime adapter.  It deliberately reuses the V2.9.6.3 guardian rather than
copying or weakening its fail-closed recovery behavior.

Important compatibility detail:
V1.12 retains the inherited ``M5_FINALIZED`` phase and
``GENERATION_ROTATION_M5_VERIFIED`` shutdown label, but dynamically moves the
underlying strategy cleanup horizon from 300s to 720s.  Therefore the existing
V2.9.6.3 post-finalization guardian detector starts its 30-second process-exit
grace at M12 even though the historical internal label still says M5.

This module intentionally exposes no user-facing live launch, Q50 arm phrase,
manual kill/flatten helper, or promotion bypass.  Its runtime functions are
internal integration points only.

Importing this module performs no API calls and sends no orders.
"""

from contextlib import contextmanager
from pathlib import Path

from . import mm_deep_tail_join_ask_live_v1_12_m12_guard_rotation as V112
from . import mm_deep_tail_join_ask_q50_m12_guard_v2_9_7_runtime as R
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_3_overnight_rotation as V2963


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_V2_9_7_SUPERVISOR"
M12_S = 720.0
POST_M12_EXIT_TIMEOUT_S = 30.0
GENERATION_RSS_WARNING_MB = 1536.0
GENERATION_RSS_HARD_LIMIT_MB = 3072.0


@contextmanager
def _patched_guardian_identity():
    """Publish only M12 provenance around the inherited V2.9.6.3 guardian."""
    old_deploy = V2963.DEPLOY_VERSION
    old_timeout = V2963.POST_M5_EXIT_TIMEOUT_S
    old_warning = V2963.GENERATION_RSS_WARNING_MB
    old_hard = V2963.GENERATION_RSS_HARD_LIMIT_MB

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.POST_M5_EXIT_TIMEOUT_S = POST_M12_EXIT_TIMEOUT_S
    V2963.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    V2963.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB

    try:
        yield
    finally:
        V2963.GENERATION_RSS_HARD_LIMIT_MB = old_hard
        V2963.GENERATION_RSS_WARNING_MB = old_warning
        V2963.POST_M5_EXIT_TIMEOUT_S = old_timeout
        V2963.DEPLOY_VERSION = old_deploy


def _post_m12_generation_state(supervisor_health):
    """Compatibility adapter over the inherited finalized-state detector.

    The inherited detector deliberately still looks for ``M5_FINALIZED``.
    Under V1.12 that phase is emitted only when the dynamically published
    strategy cleanup horizon reaches 720 seconds.
    """
    done, state = V2963._post_m5_generation_state(supervisor_health)
    state = dict(state or {})
    state.update(
        {
            "strategy_horizon_s": M12_S,
            "semantic_boundary": "M12",
            "compatibility_finalized_phase": "M5_FINALIZED",
        }
    )
    return bool(done), state


def _run_supervisor_m12(parent_session, cfg_path):
    """Internal supervisor dispatch through the validated V2.9.7 runtime adapter."""
    return R._run_supervisor_m12(
        Path(parent_session).resolve(),
        Path(cfg_path).resolve(),
    )


def _run_guardian_m12(parent_session, supervisor_pid):
    """Internal guardian dispatch preserving the exact V2.9.6.3 fail-closed loop."""
    with _patched_guardian_identity():
        return V2963._guardian_loop_v2963(
            Path(parent_session).resolve(),
            int(supervisor_pid),
        )


def intended_supervisor_contract():
    """Pure/read-only description of the outer M12 integration contract."""
    return {
        "deploy_version": DEPLOY_VERSION,
        "runtime_version": R.DEPLOY_VERSION,
        "live_version": V112.LIVE_VERSION,
        "strategy_horizon_s": M12_S,
        "generation_process_lifetime": "ONE_COMPLETE_M0_M12_WINDOW",
        "recorder_research_end_s": 720.0,
        "recorder_label_tail_end_s": 750.0,
        "post_m12_exit_timeout_s": POST_M12_EXIT_TIMEOUT_S,
        "guardian_implementation": "V2.9.6.3_FAIL_CLOSED_COMPATIBILITY_REUSE",
        "compatibility_finalized_phase": "M5_FINALIZED",
        "compatibility_rotation_shutdown_reason": "GENERATION_ROTATION_M5_VERIFIED",
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "fixed_session_risk_baseline": True,
        "user_facing_live_launch_exposed": False,
        "manual_kill_flatten_exposed": False,
        "orders_sent": False,
    }


def static_self_check(*, show=True):
    """Pure/static integration audit.  Does not run supervisor or guardian loops."""
    runtime = R.static_self_check(show=False)
    guardian_source_names = set(V2963._guardian_loop_v2963.__code__.co_names)
    post_source_consts = set(V2963._post_m5_generation_state.__code__.co_consts)

    checks = {
        "runtime_adapter_ok": runtime.get("ok") is True,
        "live_v1_12_exact": V112.LIVE_VERSION == R.LIVE_RUNTIME.LIVE_VERSION,
        "m12_horizon_720": abs(M12_S - 720.0) < 1e-12,
        "post_m12_exit_timeout_30": abs(POST_M12_EXIT_TIMEOUT_S - 30.0) < 1e-12,
        "rss_warning_1536": abs(GENERATION_RSS_WARNING_MB - 1536.0) < 1e-12,
        "rss_hard_3072": abs(GENERATION_RSS_HARD_LIMIT_MB - 3072.0) < 1e-12,
        "guardian_reuses_v2963_intervention": "_guardian_intervene" in guardian_source_names,
        "guardian_uses_post_finalized_detector": "_post_m5_generation_state" in guardian_source_names,
        "compatibility_phase_retained": "M5_FINALIZED" in post_source_consts,
        "runtime_lifetime_m12": (
            runtime.get("runtime_contract", {}).get("rotation_process_lifetime")
            == "ONE_COMPLETE_M0_M12_WINDOW"
        ),
        "runtime_child_is_m12_specific": (
            runtime.get("runtime_contract", {}).get("generation_subprocess_module")
            == R.MODULE_NAME
        ),
        "no_user_facing_start_q50": "start_q50" not in globals(),
        "no_user_facing_kill_flatten": "kill_and_flatten_live" not in globals(),
        "no_q50_arm_phrase": "Q50_ARM" not in globals(),
        "orders_sent": False,
    }

    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": DEPLOY_VERSION,
        "supervisor_contract": intended_supervisor_contract(),
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 132)
        print("V2.9.7 M12_GUARD SUPERVISOR INTEGRATION STATIC CHECK — NO API / NO ORDERS")
        print("=" * 132)
        for k, v in out.items():
            print(f"{k:72s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.7 M12 supervisor static self-check failed: {out}")

    return out


__all__ = [
    "DEPLOY_VERSION",
    "M12_S",
    "POST_M12_EXIT_TIMEOUT_S",
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "intended_supervisor_contract",
    "static_self_check",
]
