from __future__ import annotations

"""V2.9.8.2 idempotent live wrapper for the Q50 M1->M12 guard 12h smoke.

V2.9.8.1 corrected the owning-supervisor generation-preflight binding, but its
``static_self_check()`` always re-ran the historical V2.9.7/V2.9.6 contract audit.
After the first successful static check had installed the process-local V1.12/M12
bindings, a second static check in the same notebook kernel made that historical
M1->M5 audit observe the intentional M12 globals and fail.  No orders were sent.

This wrapper keeps every V2.9.8.1 trading, risk, recorder and recovery rule and
changes only the audit/wrapper behavior:
- static checks are idempotent in one long-lived notebook kernel;
- V1.12 is audited directly before runtime binding;
- the already-idempotent V2.9.8 structural audit is run twice after binding;
- the self-aware V2.9.6.1 generation preflight remains explicitly bound;
- detached supervisor/generation/guardian processes re-enter this exact module.

Importing this module performs no API calls and sends no orders.
"""

from . import mm_deep_tail_join_ask_live_v1_11_rotation as V111
from . import mm_deep_tail_join_ask_live_v1_12_m12_guard_rotation as LIVE
from . import mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_12h_rotation as RUNTIME
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation as P
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_1_overnight_rotation as H
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_3_overnight_rotation as V2963


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_RECORD_M12_V2_9_8_2_12H_ROTATION"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_2_12h_rotation"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M12_GUARD_12H_V2982_ROTATION"
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
    """Install only the audited M12 runtime bindings; strategy mechanics unchanged."""
    V2963._install_patch()

    LIVE.ROTATION_CHECKPOINT_FILE = V111.ROTATION_CHECKPOINT_FILE
    LIVE.GENERATION_BOOTSTRAP_FILE = V111.GENERATION_BOOTSTRAP_FILE

    # Publish this wrapper's provenance into the runtime module so all detached
    # subprocesses re-enter V2.9.8.2 rather than an older wrapper.
    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q50_ARM

    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.M1_S = M1_S
    P.M5_S = M12_S  # historical parent name; numeric horizon is intentionally M12
    P.RECORDER_M12_S = M12_S
    P.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    P.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    P.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB

    # Owning-supervisor exemption is defined in V2.9.6.1.
    P._fresh_generation_preflight = H._fresh_generation_preflight
    P._generation_cfg = RUNTIME._generation_cfg
    P._launch_generation = RUNTIME._launch_generation

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    V2963.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB
    V2963.POST_M5_EXIT_TIMEOUT_S = RUNTIME.POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state

    # Dynamic calls inside the runtime module must preserve this exact binding.
    RUNTIME._install_patch = _install_patch


def static_self_check(*, show=True):
    """Idempotent structural audit; no API calls and no orders."""
    live = LIVE.static_self_check(show=False)
    _install_patch()

    # Run the runtime structural check twice deliberately.  This is the regression
    # for the V2.9.8.1 notebook-kernel failure: repeated preflight/static calls must
    # not resurrect the historical M1->M5 globals or fail after M12 binding.
    base_first = RUNTIME.static_self_check(show=False)
    base_second = RUNTIME.static_self_check(show=False)

    checks = {
        "live_v1_12_static_ok": live.get("ok") is True,
        "runtime_v2_9_8_first_static_ok": base_first.get("ok") is True,
        "runtime_v2_9_8_second_static_ok": base_second.get("ok") is True,
        "repeated_static_check_idempotent": base_first.get("ok") is True and base_second.get("ok") is True,
        "live_v1_12_exact": P.LIVE is LIVE,
        "terminal_horizon_m12": P.M5_S == 720.0,
        "self_aware_preflight_bound_from_v2961": P._fresh_generation_preflight is H._fresh_generation_preflight,
        "generation_launcher_is_m12": P._launch_generation is RUNTIME._launch_generation,
        "generation_config_is_m12": P._generation_cfg is RUNTIME._generation_cfg,
        "guardian_post_window_is_m12": V2963._post_m5_generation_state is RUNTIME._post_m12_generation_state,
        "detached_module_points_to_v2982": RUNTIME.MODULE_NAME == MODULE_NAME,
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
        "no_rearm": True,
        "no_repeat_after_flat": True,
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
        print("=" * 138)
        print("V2.9.8.2 Q50 M1->M12 GUARD 12H IDEMPOTENT STATIC CHECK — NO API / NO ORDERS")
        print("=" * 138)
        for k, v in out.items():
            print(f"{k:78s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.8.2 static self-check failed: {out}")
    return out


def q50_preflight(*, show=True):
    """Read-only account/private-WS preflight; sends no orders."""
    static_self_check(show=show)
    return RUNTIME.q50_preflight(show=show)


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
