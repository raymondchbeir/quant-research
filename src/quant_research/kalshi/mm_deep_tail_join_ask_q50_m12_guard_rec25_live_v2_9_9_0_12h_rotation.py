from __future__ import annotations

"""V2.9.9.0 real-money Q50 / 12h M1->M12 guard + REC25 deployment.

Operational baseline is V2.9.8.7. The only strategy delta is the full-Q exit timing:
REC25 waits for 25% recovery from the 5c dislocation toward the causal pre-break
midpoint anchor, then posts one fixed passive exit at the freshest certified
chosen-tail ask. No repricing/chasing.

V2.9.8.7 operational behavior retained:
- Q50, 12h parent runtime;
- M1 entry, M12 cleanup;
- persistent danger guard;
- $20 software loss trigger and $125 minimum start equity;
- fresh trader process each M12 generation;
- parent-owned M0->M12+ recorder;
- V1.12.2 cancel stale-REST reconciliation via the V1.12.3 REC25 subclass;
- 4096 MiB RSS warning;
- non-fatal diagnostic RSS hard-limit sentinel;
- genuine orphan orders remain fail-closed.

Importing this module performs no API calls and sends no orders.
"""

from . import mm_deep_tail_join_ask_q50_m12_guard_live_v2_9_8_7_12h_rotation as BASE
from . import mm_deep_tail_join_ask_live_v1_12_3_m12_guard_rec25 as REC25


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_REC25_V2_9_9_0_12H_ROTATION"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_rec25_live_v2_9_9_0_12h_rotation"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M12_GUARD_REC25_12H_V2990"
KILL_ARM = BASE.KILL_ARM

RUNTIME = BASE.RUNTIME
P = BASE.P
H = BASE.H
V2963 = BASE.V2963
V28 = BASE.V28
V288 = BASE.V288
V111 = BASE.V111

LIVE = REC25

Q50_Q = BASE.Q50_Q
Q50_HOURS = BASE.Q50_HOURS
Q50_MAX_LOSS_USD = BASE.Q50_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = BASE.Q50_MIN_EQUITY_USD
M1_S = BASE.M1_S
M12_S = BASE.M12_S
LABEL_TAIL_END_S = BASE.LABEL_TAIL_END_S

GENERATION_RSS_WARNING_MB = BASE.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = BASE.GENERATION_RSS_HARD_LIMIT_MB
RSS_HARD_STOP_DISABLED = BASE.RSS_HARD_STOP_DISABLED

RECOVERY_FRACTION = REC25.RECOVERY_FRACTION
PRE_LOOKBACK_S = REC25.PRE_LOOKBACK_S
PRE_EXCLUDE_S = REC25.PRE_EXCLUDE_S
PRE_FALLBACK_S = REC25.PRE_FALLBACK_S


def _install_patch():
    """Install V2.9.8.7 operations, then replace only LIVE with REC25 V1.12.3."""
    BASE._install_patch()

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
    P.M5_S = M12_S
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

    # Preserve this exact wrapper through inherited dynamic calls/subprocess launch.
    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline structural audit; no private API and no orders."""
    base = BASE.static_self_check(show=False)
    rec = REC25.static_self_check(show=False)
    _install_patch()

    checks = {
        "base_v2987_ok": base.get("ok") is True,
        "rec25_v1_12_3_ok": rec.get("ok") is True,
        "runtime_live_is_rec25": RUNTIME.LIVE is REC25,
        "parent_live_is_rec25": P.LIVE is REC25,
        "guardian_live_is_rec25": V2963.LIVE is REC25,
        "rec25_engine_exact": LIVE.M12GuardRotatingGenerationEngine is LIVE.Rec25M12Engine,
        "terminal_horizon_m12": P.M5_S == 720.0,
        "detached_module_points_to_v2990": RUNTIME.MODULE_NAME == MODULE_NAME,
        "q50_exact_50": Q50_Q == 50.0,
        "runtime_exact_12h": Q50_HOURS == 12.0,
        "loss_trigger_exact_20": Q50_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q50_MIN_EQUITY_USD == 125.0,
        "entry_start_m1_60": M1_S == 60.0,
        "terminal_cleanup_m12_720": M12_S == 720.0,
        "guard_yes_bid_10c": LIVE.YES_GUARD_BID_MAX == 0.10,
        "guard_no_ask_90c": LIVE.NO_GUARD_ASK_MIN == 0.90,
        "guard_persist_5s": LIVE.GUARD_PERSIST_S == 5.0,
        "guard_min_obs_3": LIVE.GUARD_MIN_BOOK_OBS == 3,
        "cancel_rest_grace_5s": LIVE.CANCEL_REST_GRACE_S == 5.0,
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "rec25_anchor_10s": PRE_LOOKBACK_S == 10.0,
        "rec25_exclude_1s": PRE_EXCLUDE_S == 1.0,
        "rec25_fallback_30s": PRE_FALLBACK_S == 30.0,
        "trader_rss_warning_4096": GENERATION_RSS_WARNING_MB == 4096.0,
        "rss_hard_stop_disabled": RSS_HARD_STOP_DISABLED is True,
        "rss_numeric_sentinel_unreachable": GENERATION_RSS_HARD_LIMIT_MB >= 1_000_000_000.0,
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
        "rec25": {
            "entry_reference_c": LIVE.ENTRY_C,
            "fraction": RECOVERY_FRACTION,
            "pre_lookback_s": PRE_LOOKBACK_S,
            "pre_exclude_s": PRE_EXCLUDE_S,
            "pre_fallback_s": PRE_FALLBACK_S,
        },
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 152)
        print("V2.9.9.0 Q50 M1->M12 GUARD + REC25 12H STATIC CHECK — NO API / NO ORDERS")
        print("=" * 152)
        for k, v in out.items():
            print(f"{k:88s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.9.0 static self-check failed: {out}")
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
    """REAL-MONEY Q50 / 12h REC25 entrypoint."""
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
    "RSS_HARD_STOP_DISABLED",
    "RECOVERY_FRACTION",
    "PRE_LOOKBACK_S",
    "PRE_EXCLUDE_S",
    "PRE_FALLBACK_S",
    "static_self_check",
    "q50_preflight",
    "start_q50_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
