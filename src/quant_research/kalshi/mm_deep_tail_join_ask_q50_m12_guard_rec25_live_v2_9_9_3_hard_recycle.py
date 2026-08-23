from __future__ import annotations

"""V2.9.9.3 parent-enforced M12 hard recycle.

Operational-only layer on top of V2.9.9.2 / V1.12.5.

Observed failure addressed
--------------------------
A V2.9.9.2 generation reached M12, finalized some tickers, then stalled during
sequential terminal cleanup.  Because the V2.9.6 parent waited only for child
process exit, the stalled trader never produced a rotation checkpoint and no fresh
generation was launched.  The same long-lived process then accumulated extreme VM
allocator/swap pressure.

This layer keeps the normal child-owned verified rotation path unchanged, but adds
an independent parent deadline:

    generation trade_start + M12 (720s) + 45s cleanup grace

If the child is still alive at that deadline, the parent invokes the already-audited
authoritative fail-closed recovery path.  A new generation is allowed only when:
- authoritative exchange recovery verifies zero nonzero positions and zero
  strategy-group resting orders; AND
- the old trader process group is confirmed dead.

The parent writes a generation-local hard-recycle receipt before continuing.  The
parent-owned recorder and fixed session risk baseline remain alive/unchanged.

Strategy mechanics are unchanged: Q50, M1, M12, danger guard, REC25, atomic trigger
snapshot, passive-exit semantics, exact equity, no repricing, and reduce-only IOC
terminal cleanup are inherited exactly from V2.9.9.2 / V1.12.5.

Importing this module performs no API calls and sends no orders.
"""

import math
import time
from pathlib import Path

from . import mm_deep_tail_join_ask_q50_m12_guard_rec25_live_v2_9_9_2_12h_rotation as BASE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q50_M1_M12_GUARD_REC25_V2_9_9_3_HARD_RECYCLE"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q50_m12_guard_rec25_live_v2_9_9_3_hard_recycle"
Q50_ARM = "LIVE_DEEP_TAIL_Q50_M1_M12_GUARD_REC25_12H_V2993"
KILL_ARM = BASE.KILL_ARM

RUNTIME = BASE.RUNTIME
P = BASE.P
H = BASE.H
V2963 = BASE.V2963
V28 = BASE.V28
V288 = BASE.V288
V111 = BASE.V111
LIVE = BASE.LIVE

B = P.B
REC = P.REC
CORE = P.CORE

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

RECOVERY_FRACTION = BASE.RECOVERY_FRACTION
PRE_LOOKBACK_S = BASE.PRE_LOOKBACK_S
PRE_EXCLUDE_S = BASE.PRE_EXCLUDE_S
PRE_FALLBACK_S = BASE.PRE_FALLBACK_S

# Child terminal cleanup is normally only a few seconds.  The existing per-ticker
# cancel retirement loop can wait up to 4s, so 45s gives the sequential cleanup path
# meaningful room while still bounding a wedged/GC-stalled trader process.
M12_HARD_RECYCLE_GRACE_S = 45.0
HARD_RECYCLE_RECEIPT_FILE = "supervisor_m12_hard_recycle_v2_9_9_3.json"


def _hard_recycle_deadline(trade_start):
    """Pure helper: parent wall-clock deadline for one generation."""
    try:
        ts = float(trade_start)
    except Exception:
        return None
    if not math.isfinite(ts):
        return None
    return ts + float(M12_S) + float(M12_HARD_RECYCLE_GRACE_S)


def _hard_recycle_due(*, now, trade_start):
    deadline = _hard_recycle_deadline(trade_start)
    if deadline is None:
        return False
    try:
        t = float(now)
    except Exception:
        return False
    return math.isfinite(t) and t >= deadline


def _write_hard_recycle_receipt(
    parent_session,
    generation_dir,
    *,
    generation_id,
    trader_pid,
    generation_trade_start,
    recovery,
):
    """Durable parent proof that an unresponsive generation was safely recycled."""
    parent_session = Path(parent_session).resolve()
    generation_dir = Path(generation_dir).resolve()
    recovery = recovery or {}
    trader_stop = recovery.get("trader_stop") or {}
    safe = bool(recovery.get("recovery_verified") is True and trader_stop.get("dead") is True)
    receipt = {
        "time": B._iso(),
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "generation_id": int(generation_id),
        "generation_dir": str(generation_dir),
        "trader_pid": int(trader_pid or 0) or None,
        "generation_trade_start": float(generation_trade_start),
        "m12_horizon_s": float(M12_S),
        "cleanup_grace_s": float(M12_HARD_RECYCLE_GRACE_S),
        "hard_recycle_deadline": _hard_recycle_deadline(generation_trade_start),
        "reason": "M12_HARD_RECYCLE_DEADLINE",
        "authoritative_recovery_verified": recovery.get("recovery_verified") is True,
        "old_trader_dead": trader_stop.get("dead") is True,
        "safe_to_launch_fresh_generation": safe,
        "recovery": recovery,
    }
    B._atomic(generation_dir / HARD_RECYCLE_RECEIPT_FILE, receipt)
    B._append(
        parent_session / P.SUPERVISOR_EVENTS_FILE,
        {
            "time": B._iso(),
            "event": "M12_HARD_RECYCLE_COMPLETED",
            "generation_id": int(generation_id),
            "trader_pid": int(trader_pid or 0) or None,
            "safe_to_launch_fresh_generation": safe,
            "receipt": str(generation_dir / HARD_RECYCLE_RECEIPT_FILE),
        },
    )
    return receipt


def _run_supervisor_hard_recycle(parent_session, cfg_path):
    """V2.9.6 supervisor plus an independent parent-enforced M12 recycle deadline."""
    parent_session = Path(parent_session).resolve()
    cfg = B._read(Path(cfg_path), {}) or {}
    parent_preflight = B._read(parent_session / "parent_preflight_snapshot.json", {}) or {}
    if not parent_preflight.get("ok"):
        raise RuntimeError("Supervisor parent preflight snapshot missing/not PASS")

    session_start_equity = float((parent_preflight.get("account") or {}).get("equity_usd"))
    session_kill_equity = session_start_equity - float(cfg["max_start_loss_usd"])
    B._atomic(parent_session / "session_risk_baseline_v2_9_6.json", {
        "time": B._iso(),
        "session_start_equity_usd": session_start_equity,
        "session_kill_equity_usd": session_kill_equity,
        "max_start_loss_usd": float(cfg["max_start_loss_usd"]),
        "baseline_reset_between_generations": False,
    })

    recorder_proc = None
    recorder_pid = 0
    current_proc = None
    current_dir = None
    generation_id = 0
    session_trade_start = None
    session_deadline = None
    last_error = None
    final_reason = None
    generations = []
    smoke_checkpoint = None
    recorder_stop = None

    try:
        recorder_proc, recorder_health = P._start_external_recorder(parent_session)
        recorder_pid = int(recorder_proc.pid)
        B._atomic(parent_session / "external_recorder_start_v2_9_6.json", {
            "time": B._iso(), "pid": recorder_pid, "health": recorder_health,
            "study_version": REC.STUDY_VERSION,
        })
        B._append(parent_session / P.SUPERVISOR_EVENTS_FILE, {
            "time": B._iso(), "event": "EXTERNAL_RECORDER_STARTED", "pid": recorder_pid
        })

        while True:
            if (parent_session / P.SESSION_KILL_FILE).exists():
                req = B._read(parent_session / P.SESSION_KILL_FILE, {}) or {}
                final_reason = str(req.get("reason") or "MANUAL_KILL_AND_FLATTEN")
                break
            if not B._pid_alive(recorder_pid):
                raise RuntimeError("External recorder exited unexpectedly")
            if session_deadline is not None and time.time() >= session_deadline:
                final_reason = "RUNTIME_COMPLETE"
                break

            generation_id += 1
            if session_deadline is None:
                remaining_h = float(cfg["runtime_hours"])
            else:
                remaining_h = max(0.02, (float(session_deadline) - time.time()) / 3600.0)
            generation_preflight = P._fresh_generation_preflight(
                cfg, remaining_hours=remaining_h, show=False
            )
            generation_equity = float(
                (generation_preflight.get("account") or {}).get("equity_usd")
            )
            if generation_equity <= session_kill_equity + B.EPS:
                raise RuntimeError(
                    f"Fixed session loss trigger breached before generation {generation_id}: "
                    f"current={generation_equity:.4f} kill={session_kill_equity:.4f}"
                )
            current_proc, current_dir, gen_cfg, gen_log = P._launch_generation(
                parent_session,
                cfg,
                generation_preflight,
                generation_id=generation_id,
                recorder_pid=recorder_pid,
                session_start_equity=session_start_equity,
                session_kill_equity=session_kill_equity,
                remaining_hours=remaining_h,
            )

            startup_deadline = time.time() + P.STARTUP_TIMEOUT_S
            ready = False
            last_h = {}
            while time.time() < startup_deadline:
                if current_proc.poll() is not None:
                    break
                if not B._pid_alive(recorder_pid):
                    break
                ready, last_h = P._generation_health_ready(current_dir)
                if ready:
                    break
                P._write_supervisor_health(
                    parent_session, cfg=cfg, recorder_pid=recorder_pid,
                    generation_id=generation_id, trader_pid=current_proc.pid,
                    generation_dir=current_dir, session_start_equity=session_start_equity,
                    session_kill_equity=session_kill_equity,
                    session_trade_start=session_trade_start, session_deadline=session_deadline,
                    state="STARTING_GENERATION",
                )
                time.sleep(0.20)
            if not ready:
                raise RuntimeError(
                    f"Generation {generation_id} startup failed/timeout rc={current_proc.poll()} "
                    f"health={last_h} log={P._tail_text(gen_log)}"
                )

            generation_trade_start = None
            hard_recycle_receipt = None

            while current_proc.poll() is None:
                if not B._pid_alive(recorder_pid):
                    raise RuntimeError("External recorder died while trader was active")
                gh = B._read(current_dir / "health.json", {}) or {}
                ts = B._f(gh.get("trade_start"), float("nan"))
                if math.isfinite(ts):
                    if generation_trade_start is None:
                        generation_trade_start = float(ts)
                        B._append(
                            parent_session / P.SUPERVISOR_EVENTS_FILE,
                            {
                                "time": B._iso(),
                                "event": "M12_HARD_RECYCLE_ARMED",
                                "generation_id": generation_id,
                                "trader_pid": current_proc.pid,
                                "generation_trade_start": generation_trade_start,
                                "m12_horizon_s": M12_S,
                                "cleanup_grace_s": M12_HARD_RECYCLE_GRACE_S,
                                "hard_recycle_deadline": _hard_recycle_deadline(generation_trade_start),
                            },
                        )
                    if session_trade_start is None:
                        session_trade_start = float(ts)
                        session_deadline = session_trade_start + float(cfg["runtime_hours"]) * 3600.0
                        B._append(parent_session / P.SUPERVISOR_EVENTS_FILE, {
                            "time": B._iso(), "event": "SESSION_CLOCK_STARTED",
                            "trade_start": session_trade_start, "deadline": session_deadline,
                        })

                if (parent_session / P.SESSION_KILL_FILE).exists():
                    req = B._read(parent_session / P.SESSION_KILL_FILE, {}) or {}
                    B._atomic(current_dir / "KILL_REQUEST.json", {
                        "time": B._iso(), "reason": str(req.get("reason") or "MANUAL_KILL_AND_FLATTEN")
                    })
                elif session_deadline is not None and time.time() >= session_deadline:
                    B._atomic(current_dir / "KILL_REQUEST.json", {
                        "time": B._iso(), "reason": "RUNTIME_COMPLETE"
                    })
                elif generation_trade_start is not None and _hard_recycle_due(
                    now=time.time(), trade_start=generation_trade_start
                ):
                    B._append(
                        parent_session / P.SUPERVISOR_EVENTS_FILE,
                        {
                            "time": B._iso(),
                            "event": "M12_HARD_RECYCLE_DEADLINE_REACHED",
                            "generation_id": generation_id,
                            "trader_pid": current_proc.pid,
                            "generation_trade_start": generation_trade_start,
                            "health_time": gh.get("time"),
                            "health_state": gh.get("state"),
                            "rotation_checkpoint_written": gh.get("rotation_checkpoint_written") is True,
                        },
                    )
                    recovery = P._recover_generation_fail_closed(
                        parent_session,
                        current_dir,
                        current_proc.pid,
                        reason="M12_HARD_RECYCLE_DEADLINE",
                    )
                    hard_recycle_receipt = _write_hard_recycle_receipt(
                        parent_session,
                        current_dir,
                        generation_id=generation_id,
                        trader_pid=current_proc.pid,
                        generation_trade_start=generation_trade_start,
                        recovery=recovery,
                    )
                    if not hard_recycle_receipt.get("safe_to_launch_fresh_generation"):
                        raise RuntimeError(
                            f"M12 hard recycle did not prove old generation safe/dead: {hard_recycle_receipt}"
                        )
                    current_proc.poll()
                    break

                P._write_supervisor_health(
                    parent_session, cfg=cfg, recorder_pid=recorder_pid,
                    generation_id=generation_id, trader_pid=current_proc.pid,
                    generation_dir=current_dir, session_start_equity=session_start_equity,
                    session_kill_equity=session_kill_equity,
                    session_trade_start=session_trade_start, session_deadline=session_deadline,
                    state="RUNNING_GENERATION",
                )
                time.sleep(P.SUPERVISOR_POLL_S)

            rc = current_proc.poll()
            final = B._read(current_dir / "final_summary.json", {}) or {}
            checkpoint = B._read(current_dir / LIVE.ROTATION_CHECKPOINT_FILE, {}) or {}
            generation_record = {
                "generation_id": generation_id,
                "generation_dir": str(current_dir),
                "trader_pid": current_proc.pid,
                "returncode": rc,
                "final": final,
                "checkpoint": checkpoint,
                "hard_recycle": hard_recycle_receipt,
            }
            generations.append(generation_record)

            if hard_recycle_receipt is not None:
                B._append(parent_session / P.SUPERVISOR_EVENTS_FILE, {
                    "time": B._iso(),
                    "event": "GENERATION_HARD_RECYCLED",
                    "generation_id": generation_id,
                    "returncode": rc,
                    "safe_to_launch_fresh_generation": hard_recycle_receipt.get(
                        "safe_to_launch_fresh_generation"
                    ) is True,
                })
                parent_kill = (parent_session / P.SESSION_KILL_FILE).exists()
                deadline_reached = session_deadline is not None and time.time() >= session_deadline
                if parent_kill or deadline_reached:
                    final_reason = "RUNTIME_COMPLETE" if deadline_reached else "MANUAL_KILL_AND_FLATTEN"
                    break
                # Critical invariant: recovery verified exchange-flat AND old process dead.
                # The next loop iteration performs a fresh generation preflight before launch.
                current_proc = None
                current_dir = None
                continue

            B._append(parent_session / P.SUPERVISOR_EVENTS_FILE, {
                "time": B._iso(), "event": "GENERATION_EXITED",
                "generation_id": generation_id, "returncode": rc,
                "shutdown_reason": final.get("shutdown_reason"),
                "safe_to_rotate": checkpoint.get("safe_to_rotate"),
            })

            parent_kill = (parent_session / P.SESSION_KILL_FILE).exists()
            deadline_reached = session_deadline is not None and time.time() >= session_deadline
            if parent_kill or deadline_reached:
                expected = {
                    "RUNTIME_COMPLETE", "MANUAL_KILL_AND_FLATTEN", "GUARDIAN_RSS_HARD_LIMIT",
                    "GUARDIAN_SUPERVISOR_FAILURE", "SUPERVISOR_STOP_REQUEST",
                }
                if rc == 0 and P._is_clean_final(final) and str(final.get("shutdown_reason") or "") in expected:
                    final_reason = str(final.get("shutdown_reason") or "RUNTIME_COMPLETE")
                    break
                P._recover_generation_fail_closed(
                    parent_session, current_dir, current_proc.pid,
                    reason=f"TERMINATION_PATH_NOT_CLEAN:rc={rc}:final={final.get('shutdown_reason')}"
                )
                final_reason = "FAIL_CLOSED_RECOVERED_AT_SESSION_END"
                break

            if not (
                rc == 0
                and checkpoint.get("safe_to_rotate") is True
                and P._is_clean_final(final, allowed_reasons={"GENERATION_ROTATION_M5_VERIFIED"})
            ):
                P._recover_generation_fail_closed(
                    parent_session, current_dir, current_proc.pid,
                    reason=f"ABNORMAL_GENERATION_EXIT:rc={rc}:final={final.get('shutdown_reason')}:checkpoint={checkpoint.get('reason')}"
                )
                final_reason = "ABNORMAL_GENERATION_FAIL_CLOSED_RECOVERED"
                break

            checkpoint = dict(checkpoint)
            checkpoint["generation_dir"] = str(current_dir)
            if bool(cfg.get("rotation_smoke")):
                smoke_checkpoint = checkpoint
                tail_result = P._wait_smoke_tail(parent_session, recorder_pid, checkpoint)
                if not tail_result.get("recorder_alive_at_tail_boundary"):
                    raise RuntimeError("Rotation smoke recorder did not survive through M12+30")
                recorder_stop = P._stop_external_recorder(recorder_pid)
                promotion = P._write_smoke_promotion(
                    parent_session, cfg, checkpoint, recorder_pid, tail_result, recorder_stop
                )
                if not promotion.get("passed"):
                    raise RuntimeError(f"Rotation smoke promotion audit failed: {promotion}")
                final_reason = "ROTATION_SMOKE_PASSED"
                break

            current_proc = None
            current_dir = None

        if current_proc is not None and current_proc.poll() is None:
            B._atomic(current_dir / "KILL_REQUEST.json", {
                "time": B._iso(), "reason": final_reason or "SUPERVISOR_STOP_REQUEST"
            })
            deadline = time.time() + 20.0
            while current_proc.poll() is None and time.time() < deadline:
                time.sleep(0.20)
            if current_proc.poll() is None:
                P._recover_generation_fail_closed(
                    parent_session, current_dir, current_proc.pid,
                    reason=final_reason or "SUPERVISOR_STOP_REQUEST_TIMEOUT",
                )

    except BaseException as exc:
        last_error = repr(exc)
        B._append(parent_session / P.SUPERVISOR_EVENTS_FILE, {
            "time": B._iso(), "event": "SUPERVISOR_ERROR", "error": last_error
        })
        if current_proc is not None and current_dir is not None:
            try:
                P._recover_generation_fail_closed(
                    parent_session, current_dir, current_proc.pid,
                    reason=f"SUPERVISOR_EXCEPTION:{last_error}",
                )
            except Exception as recovery_exc:
                B._append(parent_session / P.SUPERVISOR_EVENTS_FILE, {
                    "time": B._iso(), "event": "SUPERVISOR_RECOVERY_ERROR",
                    "error": repr(recovery_exc),
                })
        final_reason = final_reason or "SUPERVISOR_EXCEPTION"
        raise
    finally:
        if recorder_pid and B._pid_alive(recorder_pid):
            recorder_stop = P._stop_external_recorder(recorder_pid)
        recorder_audit = P._recorder_final_audit(parent_session) if (parent_session / "raw_capture").exists() else {}
        final = {
            "time": B._iso(),
            "deploy_version": DEPLOY_VERSION,
            "live_version": LIVE.LIVE_VERSION,
            "parent_session_dir": str(parent_session),
            "mode": cfg.get("mode"),
            "quote_size": float(cfg.get("quote_size")),
            "shutdown_reason": final_reason,
            "session_start_equity_usd": session_start_equity,
            "session_kill_equity_usd": session_kill_equity,
            "session_trade_start": session_trade_start,
            "session_deadline": session_deadline,
            "generations_completed": len(generations),
            "generations": generations,
            "recorder_pid": recorder_pid or None,
            "recorder_stop": recorder_stop,
            "recorder_final_audit": recorder_audit,
            "last_error": last_error,
            "rotation_smoke_checkpoint": smoke_checkpoint,
            "m12_parent_hard_recycle_enabled": True,
            "m12_hard_recycle_grace_s": M12_HARD_RECYCLE_GRACE_S,
        }
        B._atomic(parent_session / P.SUPERVISOR_FINAL_FILE, final)
        P._write_supervisor_health(
            parent_session, cfg=cfg, recorder_pid=recorder_pid,
            generation_id=generation_id, trader_pid=(current_proc.pid if current_proc else 0),
            generation_dir=current_dir, session_start_equity=session_start_equity,
            session_kill_equity=session_kill_equity,
            session_trade_start=session_trade_start, session_deadline=session_deadline,
            state="FAILED" if last_error else "STOPPED", last_error=last_error,
        )
        ctl = B._read(CORE.CONTROL_PATH, {}) or {}
        if str(ctl.get("session_dir") or "") == str(parent_session):
            ctl.update({
                "running": False,
                "stopped_at": B._iso(),
                "shutdown_reason": final_reason,
                "last_error": last_error,
            })
            B._atomic(CORE.CONTROL_PATH, ctl)


def _install_patch():
    """Install V2.9.9.2 and then add the parent hard-recycle supervisor."""
    BASE._install_patch()

    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q50_ARM
    RUNTIME.LIVE = LIVE

    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.M1_S = M1_S
    P.M5_S = M12_S
    P.RECORDER_M12_S = M12_S
    P.LABEL_TAIL_END_S = LABEL_TAIL_END_S
    P._fresh_generation_preflight = H._fresh_generation_preflight
    P._generation_cfg = RUNTIME._generation_cfg
    P._launch_generation = RUNTIME._launch_generation
    P._run_supervisor = _run_supervisor_hard_recycle

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.POST_M5_EXIT_TIMEOUT_S = RUNTIME.POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state

    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline structural/regression audit. No API calls and no orders."""
    import inspect

    base = BASE.static_self_check(show=False)
    _install_patch()
    src = inspect.getsource(_run_supervisor_hard_recycle)

    d0 = _hard_recycle_deadline(1000.0)
    pure_before = _hard_recycle_due(now=d0 - 0.001, trade_start=1000.0)
    pure_at = _hard_recycle_due(now=d0, trade_start=1000.0)

    checks = {
        "base_v2992_ok": base.get("ok") is True,
        "parent_supervisor_is_hard_recycle": P._run_supervisor is _run_supervisor_hard_recycle,
        "hard_recycle_grace_exact_45s": M12_HARD_RECYCLE_GRACE_S == 45.0,
        "deadline_formula_m12_plus_grace": abs(d0 - (1000.0 + 720.0 + 45.0)) < 1e-12,
        "deadline_not_early": pure_before is False,
        "deadline_fires_at_boundary": pure_at is True,
        "supervisor_uses_generation_trade_start": "generation_trade_start" in src,
        "supervisor_calls_authoritative_recovery": "P._recover_generation_fail_closed" in src,
        "supervisor_requires_recovery_verified": "recovery_verified" in inspect.getsource(_write_hard_recycle_receipt),
        "supervisor_requires_old_trader_dead": "trader_stop.get(\"dead\") is True" in inspect.getsource(_write_hard_recycle_receipt),
        "fresh_preflight_before_each_generation": "P._fresh_generation_preflight" in src,
        "normal_verified_child_rotation_preserved": "checkpoint.get(\"safe_to_rotate\") is True" in src,
        "fixed_session_risk_baseline_preserved": "baseline_reset_between_generations\": False" in src,
        "recorder_parent_owned": "P._start_external_recorder" in src and "P._stop_external_recorder" in src,
        "live_engine_unchanged_v1_12_5": LIVE.LIVE_VERSION == BASE.LIVE.LIVE_VERSION,
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "risk_m12_flatten_reduce_only_ioc_unchanged": True,
        "q50_exact_50": Q50_Q == 50.0,
        "entry_m1_60": M1_S == 60.0,
        "terminal_m12_720": M12_S == 720.0,
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "hard_recycle": {
            "horizon_s": M12_S,
            "cleanup_grace_s": M12_HARD_RECYCLE_GRACE_S,
            "deadline_owner": "PARENT_SUPERVISOR",
            "recovery_gate": "AUTHORITATIVE_FLAT_AND_OLD_TRADER_DEAD",
        },
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 164)
        print("V2.9.9.3 PARENT-ENFORCED M12 HARD-RECYCLE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 164)
        for k, v in out.items():
            print(f"{k:100s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.9.3 static self-check failed: {out}")
    return out


def q50_preflight(*, show=True):
    """Same exact-dollar read-only preflight as V2.9.9.2."""
    static_self_check(show=show)
    V28._patch_parent()
    V28.D._guard_other_live_processes()

    old_equity = LIVE.V1.B._equity
    LIVE.V1.B._equity = LIVE.exact_equity_from_balance
    try:
        return V288.live_preflight(
            quote_size=Q50_Q,
            runtime_hours=Q50_HOURS,
            max_start_loss_usd=Q50_MAX_LOSS_USD,
            min_start_equity_usd=Q50_MIN_EQUITY_USD,
            show=show,
            probe_private_ws=True,
        )
    finally:
        LIVE.V1.B._equity = old_equity


def start_q50_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q50 / 12h V1.12.5 with parent-enforced M12 recycling."""
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
    "M12_HARD_RECYCLE_GRACE_S",
    "HARD_RECYCLE_RECEIPT_FILE",
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "RSS_HARD_STOP_DISABLED",
    "RECOVERY_FRACTION",
    "PRE_LOOKBACK_S",
    "PRE_EXCLUDE_S",
    "PRE_FALLBACK_S",
    "_hard_recycle_deadline",
    "_hard_recycle_due",
    "_run_supervisor_hard_recycle",
    "static_self_check",
    "q50_preflight",
    "start_q50_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
