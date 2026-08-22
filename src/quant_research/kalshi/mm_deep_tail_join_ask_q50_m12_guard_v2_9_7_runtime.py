from __future__ import annotations

"""Internal runtime wiring for the frozen Q50 M1->M12 guard candidate.

This module closes the process-boundary gap between the existing V2.9.6
rotating supervisor and the V1.12 M12_GUARD engine while preserving the
existing supervisor/recovery implementation.

It intentionally does NOT expose a user-facing live launch, Q50 arming helper,
manual kill/flatten helper, or promotion bypass.  The only CLI runtime accepted
here is the supervisor-owned ``--run-generation`` child handoff.

Runtime deltas relative to V2.9.6:
- generation LIVE engine -> V1.12 M12_GUARD;
- generation child module -> this M12-specific module;
- terminal cleanup metadata -> 720s;
- generation lifetime metadata -> ONE_COMPLETE_M0_M12_WINDOW;
- trader RSS warning/hard limits -> retained at 1536 / 3072 MiB.

Compatibility labels such as M5_FINALIZED and GENERATION_ROTATION_M5_VERIFIED
remain inherited on purpose.  Under V1.12 they occur at the dynamically
published 720-second horizon, so the existing supervisor and guardian contracts
remain valid without rewriting the fail-closed machinery.

Importing this module performs no API calls and sends no orders.
"""

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from . import mm_deep_tail_join_ask_live_v1_11_rotation as V111
from . import mm_deep_tail_join_ask_live_v1_12_m12_guard_rotation as V112
from . import mm_deep_tail_join_ask_q50_m12_guard_v2_9_7_preflight as PREFLIGHT
from . import mm_deep_tail_join_ask_q50_record_m12_live_v2_9_6_overnight_rotation as P


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_V2_9_7_RUNTIME"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_v2_9_7_runtime"

M12_S = 720.0
GENERATION_RSS_WARNING_MB = 1536.0
GENERATION_RSS_HARD_LIMIT_MB = 3072.0


# P expects a small V1.11-style LIVE module interface.  V1.12 deliberately
# layers on V1.11 but does not need to duplicate those filenames in its own
# public surface, so publish the exact compatibility interface here.
LIVE_RUNTIME = SimpleNamespace(
    LIVE_VERSION=V112.LIVE_VERSION,
    run_live_process=V112.run_live_process,
    static_self_check=V112.static_self_check,
    ROTATION_CHECKPOINT_FILE=V111.ROTATION_CHECKPOINT_FILE,
    GENERATION_BOOTSTRAP_FILE=V111.GENERATION_BOOTSTRAP_FILE,
    SESSION_RISK_BASELINE_FILE=V111.SESSION_RISK_BASELINE_FILE,
)


def _m12_generation_cfg(
    parent_cfg,
    *,
    generation_id,
    generation_dir,
    recorder_pid,
    session_start_equity,
    session_kill_equity,
    remaining_hours,
):
    """Build the existing generation config, then apply only the frozen M12 delta."""
    cfg = P._generation_cfg(
        parent_cfg,
        generation_id=generation_id,
        generation_dir=generation_dir,
        recorder_pid=recorder_pid,
        session_start_equity=session_start_equity,
        session_kill_equity=session_kill_equity,
        remaining_hours=remaining_hours,
    )

    cfg.update(
        {
            "live_engine_version": V112.LIVE_VERSION,
            "deploy_version": DEPLOY_VERSION,
            "strategy_terminal_cleanup_elapsed_s": M12_S,
            "recorder_persist_end_elapsed_s": M12_S,
            "rotation_process_lifetime": "ONE_COMPLETE_M0_M12_WINDOW",
            "m12_guard_yes_bid_max": V112.YES_GUARD_BID_MAX,
            "m12_guard_no_ask_min": V112.NO_GUARD_ASK_MIN,
            "m12_guard_persist_s": V112.GUARD_PERSIST_S,
            "m12_guard_min_book_obs": V112.GUARD_MIN_BOOK_OBS,
            "m12_guard_rearm": False,
            "repeat_after_flat": False,
        }
    )

    return cfg


def _run_generation_m12(session, cfg_path):
    """Supervisor-owned child entrypoint.  Delegates through V2.9.6 after patching LIVE."""
    session = Path(session).resolve()
    cfg_path = Path(cfg_path).resolve()
    cfg = P.B._read(cfg_path, {}) or {}

    if abs(float(cfg.get("strategy_terminal_cleanup_elapsed_s", -1.0)) - M12_S) > 1e-12:
        raise RuntimeError("V2.9.7 generation refuses non-M12 terminal cleanup config")
    if str(cfg.get("live_engine_version") or "") != V112.LIVE_VERSION:
        raise RuntimeError("V2.9.7 generation refuses non-V1.12 live engine config")
    if str(cfg.get("rotation_process_lifetime") or "") != "ONE_COMPLETE_M0_M12_WINDOW":
        raise RuntimeError("V2.9.7 generation refuses non-M12 process lifetime")

    old_live = P.LIVE
    old_deploy = P.DEPLOY_VERSION
    old_m5 = P.M5_S

    P.LIVE = LIVE_RUNTIME
    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.M5_S = M12_S

    try:
        return P._run_generation(session, cfg_path)
    finally:
        P.M5_S = old_m5
        P.DEPLOY_VERSION = old_deploy
        P.LIVE = old_live


def _launch_generation_m12(
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
    """Internal supervisor child launcher using this module for the child process."""
    parent_session = Path(parent_session).resolve()
    generation_dir = (
        parent_session / "generations" / f"gen_{int(generation_id):04d}"
    )
    generation_dir.mkdir(parents=True, exist_ok=False)
    P._attach_raw_capture(parent_session, generation_dir)

    cfg = _m12_generation_cfg(
        parent_cfg,
        generation_id=generation_id,
        generation_dir=generation_dir,
        recorder_pid=recorder_pid,
        session_start_equity=session_start_equity,
        session_kill_equity=session_kill_equity,
        remaining_hours=remaining_hours,
    )
    cfg_path = generation_dir / "process_config.json"
    P.B._atomic(cfg_path, cfg)
    P.B._atomic(generation_dir / "parent_preflight_snapshot.json", parent_preflight)

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
            cwd=str(P.V28.C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        fh.close()

    P.B._append(
        parent_session / P.SUPERVISOR_EVENTS_FILE,
        {
            "time": P.B._iso(),
            "event": "GENERATION_LAUNCHED",
            "generation_id": int(generation_id),
            "generation_dir": str(generation_dir),
            "trader_pid": proc.pid,
            "remaining_hours": float(remaining_hours),
            "live_engine_version": V112.LIVE_VERSION,
            "strategy_terminal_cleanup_elapsed_s": M12_S,
            "rotation_process_lifetime": "ONE_COMPLETE_M0_M12_WINDOW",
        },
    )

    return proc, generation_dir, cfg, log


@contextmanager
def _patched_v296_supervisor_runtime():
    """Temporarily bind the existing supervisor to the exact M12 runtime contract."""
    old = {
        "LIVE": P.LIVE,
        "DEPLOY_VERSION": P.DEPLOY_VERSION,
        "M5_S": P.M5_S,
        "GENERATION_RSS_WARNING_MB": P.GENERATION_RSS_WARNING_MB,
        "GENERATION_RSS_HARD_LIMIT_MB": P.GENERATION_RSS_HARD_LIMIT_MB,
        "_generation_cfg": P._generation_cfg,
        "_launch_generation": P._launch_generation,
    }

    P.LIVE = LIVE_RUNTIME
    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.M5_S = M12_S
    P.GENERATION_RSS_WARNING_MB = GENERATION_RSS_WARNING_MB
    P.GENERATION_RSS_HARD_LIMIT_MB = GENERATION_RSS_HARD_LIMIT_MB
    P._generation_cfg = _m12_generation_cfg
    P._launch_generation = _launch_generation_m12

    try:
        yield
    finally:
        P._launch_generation = old["_launch_generation"]
        P._generation_cfg = old["_generation_cfg"]
        P.GENERATION_RSS_HARD_LIMIT_MB = old["GENERATION_RSS_HARD_LIMIT_MB"]
        P.GENERATION_RSS_WARNING_MB = old["GENERATION_RSS_WARNING_MB"]
        P.M5_S = old["M5_S"]
        P.DEPLOY_VERSION = old["DEPLOY_VERSION"]
        P.LIVE = old["LIVE"]


def _run_supervisor_m12(parent_session, cfg_path):
    """Internal supervisor dispatch.  No user-facing launch helper is provided here."""
    with _patched_v296_supervisor_runtime():
        return P._run_supervisor(
            Path(parent_session).resolve(),
            Path(cfg_path).resolve(),
        )


def intended_runtime_contract():
    """Pure/read-only description of the process-boundary wiring."""
    return {
        "deploy_version": DEPLOY_VERSION,
        "live_version": V112.LIVE_VERSION,
        "generation_subprocess_module": MODULE_NAME,
        "strategy_terminal_cleanup_elapsed_s": M12_S,
        "rotation_process_lifetime": "ONE_COMPLETE_M0_M12_WINDOW",
        "rotation_checkpoint_file": LIVE_RUNTIME.ROTATION_CHECKPOINT_FILE,
        "generation_bootstrap_file": LIVE_RUNTIME.GENERATION_BOOTSTRAP_FILE,
        "compatibility_rotation_shutdown_reason": "GENERATION_ROTATION_M5_VERIFIED",
        "compatibility_finalized_phase": "M5_FINALIZED",
        "trader_rss_warning_mb": GENERATION_RSS_WARNING_MB,
        "trader_rss_hard_limit_mb": GENERATION_RSS_HARD_LIMIT_MB,
        "user_facing_live_launch_exposed": False,
        "orders_sent": False,
    }


def static_self_check(*, show=True):
    """Pure structural check.  Does not launch the supervisor or a generation."""
    pre = PREFLIGHT.static_self_check(show=False)

    dummy_parent = {
        "mode": "OFFLINE_M12_GUARD_TEST",
        "quote_size": 50.0,
        "runtime_hours": 12.0,
        "max_start_loss_usd": 20.0,
        "min_start_equity_usd": 125.0,
        "parent_session_dir": "/tmp/OFFLINE_M12_GUARD_PARENT",
    }
    cfg = _m12_generation_cfg(
        dummy_parent,
        generation_id=1,
        generation_dir=Path("/tmp/OFFLINE_M12_GUARD_PARENT/generations/gen_0001"),
        recorder_pid=12345,
        session_start_equity=150.0,
        session_kill_equity=130.0,
        remaining_hours=12.0,
    )

    checks = {
        "deployment_preflight_ok": pre.get("ok") is True,
        "v1_12_live_version_exact": LIVE_RUNTIME.LIVE_VERSION == V112.LIVE_VERSION,
        "rotation_checkpoint_compat_exported": (
            LIVE_RUNTIME.ROTATION_CHECKPOINT_FILE == V111.ROTATION_CHECKPOINT_FILE
        ),
        "generation_bootstrap_compat_exported": (
            LIVE_RUNTIME.GENERATION_BOOTSTRAP_FILE == V111.GENERATION_BOOTSTRAP_FILE
        ),
        "generation_cfg_live_v1_12": cfg.get("live_engine_version") == V112.LIVE_VERSION,
        "generation_cfg_m12_720": (
            abs(float(cfg.get("strategy_terminal_cleanup_elapsed_s")) - 720.0) < 1e-12
        ),
        "generation_cfg_lifetime_m12": (
            cfg.get("rotation_process_lifetime") == "ONE_COMPLETE_M0_M12_WINDOW"
        ),
        "generation_child_module_is_m12_specific": MODULE_NAME.endswith("v2_9_7_runtime"),
        "rss_warning_1536": abs(GENERATION_RSS_WARNING_MB - 1536.0) < 1e-12,
        "rss_hard_3072": abs(GENERATION_RSS_HARD_LIMIT_MB - 3072.0) < 1e-12,
        "no_user_facing_start_q50": "start_q50" not in globals(),
        "no_user_facing_kill_flatten": "kill_and_flatten_live" not in globals(),
        "orders_sent": False,
    }

    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": DEPLOY_VERSION,
        "runtime_contract": intended_runtime_contract(),
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 132)
        print("V2.9.7 M12_GUARD INTERNAL RUNTIME STATIC CHECK — NO API / NO ORDERS")
        print("=" * 132)
        for k, v in out.items():
            print(f"{k:72s}: {v}")

    if not ok:
        raise RuntimeError(f"V2.9.7 M12 runtime static self-check failed: {out}")

    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-generation")
    ap.add_argument("--config")
    args = ap.parse_args()

    if args.run_generation:
        if not args.config:
            raise RuntimeError("--config is required with --run-generation")
        _run_generation_m12(Path(args.run_generation), Path(args.config))
    else:
        static_self_check(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "MODULE_NAME",
    "M12_S",
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "LIVE_RUNTIME",
    "intended_runtime_contract",
    "static_self_check",
]
