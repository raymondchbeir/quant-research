from __future__ import annotations

"""V12.1 execution hotfix after the first V12 Q1 latency smoke.

V12.1 changes execution bookkeeping/instrumentation only; frozen Candidate-C
strategy mechanics remain unchanged.

Fixes
-----
1. Keep a priority-cancel (ticker, order_id) pending until the main thread has
   retired/unpublished the order. V12 cleared pending before fill/position
   reconciliation, allowing the watchdog to submit a duplicate DELETE while the
   already-cancelled order was still published active.
2. Measure CREATE freshness from the FINAL watchdog certification watermark,
   not the older main-loop source row. A newer raw row is a CREATE defect only
   when the latest raw state at request-send invalidates the certified quote.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v4 as V4
from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_cycle_q10_live_strategy_v11 as V11
from . import mm_cycle_q10_live_strategy_v12 as V12

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V12_1"
EXECUTION_PARENT = V12.LIVE_VERSION
RECORDING_PARENT = V12.RECORDING_PARENT
FRESHNESS_ARCH_VERSION = "MM_CYCLE_Q10_PRIORITY_FRESHNESS_V12_1"
STAGED_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V12_1_STAGED"

Q5_QTY = 5.0
Q5_HOURS = 1.0
Q5_MIN_EQUITY_USD = 75.0
Q5_ARM = "LIVE_Q5_1H"
Q10_ARM = B.FULL_ARM


class FixedFastCancelWatchdog(V12.FastCancelWatchdog):
    """V12 watchdog with final-certification-aware CREATE send audit."""

    def first_newer_before(self, ticker, source_ms, cutoff_ms):
        """Return an invalidating state that is newer than FINAL certification.

        V12 used the older main-loop source receipt as the baseline. Because the
        final watchdog certification can legitimately certify an equivalent newer
        raw state, that over-counted `superseded_at_send`. Here the baseline is the
        final certification receipt. We examine the latest raw row at/before the
        actual request-send time and flag only if that state invalidates the quote.
        """
        ticker = str(ticker)
        cutoff = B._f(cutoff_ms, np.nan)
        if not np.isfinite(cutoff):
            return None

        baseline = B._f(source_ms, np.nan)
        guard_ms = B._f(self.engine._v121_final_guard_ms.get(ticker), np.nan)
        if np.isfinite(guard_ms):
            baseline = max(baseline, guard_ms) if np.isfinite(baseline) else guard_ms

        with self.lock:
            hist = list(self.history.get(ticker) or [])

        candidates = []
        for item in hist:
            receipt = B._f(item.get("receipt_wall_ms"), np.nan)
            if not np.isfinite(receipt) or receipt > cutoff + 0.001:
                continue
            if np.isfinite(baseline) and receipt <= baseline + 0.001:
                continue
            candidates.append(item)

        if not candidates:
            return None

        latest = max(candidates, key=lambda x: B._f(x.get("receipt_wall_ms"), -np.inf))
        track = self.engine._v121_final_guard_track.get(ticker)
        if not track:
            return latest

        invalid, reason = V12._track_invalidated(track, latest.get("row") or {})
        if not invalid:
            return None

        out = dict(latest)
        out["invalid_reason_at_send"] = reason
        return out


class PriorityFreshnessEngine121(V12.PriorityFreshnessEngine):
    """V12 with fixed fast-cancel retirement and CREATE freshness watermark."""

    def __init__(self, *args, **kwargs):
        # Reproduce V12 init but instantiate the fixed watchdog.
        V11.V2OnlyProductionEngine.__init__(self, *args, **kwargs)
        self.latency_log_v12 = self.session / "latency_events_v12.jsonl"
        self._lat_lock = threading.Lock()
        self._api_context = {}
        self._request_original = self.client.request
        self._instrument_client()
        self.v12 = Counter()
        self.max_actionable = 0
        self._v121_final_guard_ms = {}
        self._v121_final_guard_track = {}
        self.fast = FixedFastCancelWatchdog(self)
        self.fast.start()
        self._publish()
        self._lat("V12_1_ENGINE_READY", execution_parent=EXECUTION_PARENT)

    def _watchdog_certification(self, ticker, track_like, source_ms, phase):
        ok, snap, reason = super()._watchdog_certification(
            ticker, track_like, source_ms, phase
        )
        if ok and str(phase) == "FINAL_PRE_POST" and snap:
            t = str(ticker)
            self._v121_final_guard_ms[t] = B._f(snap.get("receipt_wall_ms"), np.nan)
            self._v121_final_guard_track[t] = dict(track_like or {})
            self._lat(
                "FINAL_CREATE_CERTIFICATION_V12_1",
                ticker=t,
                guard_receipt_wall_ms=self._v121_final_guard_ms[t],
                source_receipt_wall_ms=B._f(source_ms, np.nan),
                role=(track_like or {}).get("role"),
                side=(track_like or {}).get("side"),
                price=(track_like or {}).get("price"),
            )
        return ok, snap, reason

    def _consume_fast(self, limit=100):
        """Retire/unpublish order BEFORE clearing watchdog pending state."""
        for result in self.fast.drain_results(limit=limit):
            ticker = str(result.get("ticker") or "")
            oid = str(result.get("order_id") or "")
            self.v12["fast_results"] += 1

            tr = self.active.get(ticker)
            if not tr or str(tr.get("order_id") or "") != oid:
                self.v12["stale_fast_results"] += 1
                self.fast.clear_pending(ticker, oid)
                continue

            if not result.get("success"):
                self.v12["fast_fallbacks"] += 1
                self._api_context = {
                    "v12": "FAST_CANCEL_ERROR_FALLBACK_V12_1",
                    "ticker": ticker,
                    "order_id": oid,
                }
                try:
                    # Call V11 directly: V12.cancel_track intentionally refuses a
                    # duplicate normal cancel while the priority path is pending.
                    V11.V2OnlyProductionEngine.cancel_track(
                        self, ticker, "V12_1_FAST_CANCEL_ERROR_FALLBACK"
                    )
                finally:
                    self._api_context = {}

                # V11 has now retired the active order (or failed closed).
                self._publish()
                self.fast.clear_pending(ticker, oid)
                self._lat(
                    "FAST_RESULT_RETIRED_V12_1",
                    ticker=ticker,
                    order_id=oid,
                    initial_fast_success=False,
                )
                continue

            reconcile_start_ms = V12._wall_ms()
            old_fill = float(tr.get("last_fill", 0.0))
            submitted_qty = float(tr.get("qty", 0.0))
            before_pos = float(self.positions.get(ticker, 0.0))
            reduced_by = self._parse_reduced_by(
                result.get("cancel_body") or {}, submitted_qty
            )
            final_fill = min(
                submitted_qty,
                max(old_fill, submitted_qty - reduced_by),
            )
            tr["last_fill"] = final_fill

            self._api_context = {
                "v12": "FAST_CANCEL_RECONCILE_V12_1",
                "ticker": ticker,
                "order_id": oid,
            }
            try:
                self.record_fills(tr)
                after_pos = self.refresh_position(ticker)
            finally:
                self._api_context = {}

            reconcile_done_ms = V12._wall_ms()
            response_ms = B._f(result.get("response_recv_wall_ms"))
            response_to_reconcile_ms = (
                reconcile_start_ms - response_ms
                if np.isfinite(response_ms)
                else np.nan
            )

            B._append(
                self.orders,
                {
                    "time": B._iso(),
                    "action": "CANCEL_V12_1_PRIORITY_FAST_PATH",
                    "ticker": ticker,
                    "reason": result.get("reason"),
                    "track": tr,
                    "invalidation_id": result.get("invalidation_id"),
                    "obsolete_receipt_wall_ms": result.get("obsolete_receipt_wall_ms"),
                    "cancel_body": result.get("cancel_body"),
                    "cancel_timing": result.get("cancel_timing"),
                    "obsolete_to_cancel_send_ms": result.get("obsolete_to_cancel_send_ms"),
                    "detect_to_cancel_send_ms": result.get("detect_to_cancel_send_ms"),
                    "response_to_reconcile_start_ms": response_to_reconcile_ms,
                    "reconcile_duration_ms": reconcile_done_ms - reconcile_start_ms,
                    "old_fill": old_fill,
                    "final_fill": final_fill,
                    "position_before": before_pos,
                    "position_after": after_pos,
                },
            )

            # Critical ordering change from V12:
            # 1) retire from authoritative active map,
            # 2) publish no-longer-active snapshot to watchdog,
            # 3) only then clear pending.
            self.active.pop(ticker, None)
            self.counts["cancels"] += 1
            self.v12["fast_success"] += 1

            raced_fill = (
                final_fill > old_fill + B.EPS
                or abs(after_pos - before_pos) > B.EPS
            )
            if raced_fill:
                self.v12["fast_raced_fill"] += 1
                self.barrier[ticker] = self.book_version[ticker]
                self.counts["fill_events"] += 1
                self.emit(
                    "FILL",
                    ticker,
                    role=tr["role"],
                    side=tr["side"],
                    qty=max(0.0, final_fill - old_fill),
                    position=after_pos,
                    source="v12_1_priority_fast_cancel",
                )

            self._publish()
            self.fast.clear_pending(ticker, oid)

            self._lat(
                "FAST_CANCEL_RECONCILED",
                ticker=ticker,
                order_id=oid,
                invalidation_id=result.get("invalidation_id"),
                cancel_response_recv_wall_ms=response_ms,
                reconcile_start_wall_ms=reconcile_start_ms,
                reconcile_done_wall_ms=reconcile_done_ms,
                response_to_reconcile_start_ms=response_to_reconcile_ms,
                reconcile_duration_ms=reconcile_done_ms - reconcile_start_ms,
                raced_fill=raced_fill,
                position_after=after_pos,
                v12_1_retirement_ordering=True,
            )
            self._lat(
                "FAST_RESULT_RETIRED_V12_1",
                ticker=ticker,
                order_id=oid,
                initial_fast_success=True,
            )

    def _v12_metrics(self):
        out = super()._v12_metrics()
        out.update(
            {
                "live_version": LIVE_VERSION,
                "freshness_arch_version": FRESHNESS_ARCH_VERSION,
                "execution_parent": EXECUTION_PARENT,
                "v12_1_priority_pending_retirement_fix": True,
                "v12_1_final_create_certification_watermark": True,
            }
        )
        return out


def static_self_check(*, show=True):
    base = V12.static_self_check(show=False)
    checks = dict(base.get("checks") or {})
    checks.update(
        {
            "execution_parent_is_v12": EXECUTION_PARENT == V12.LIVE_VERSION,
            "fixed_watchdog_subclass": issubclass(FixedFastCancelWatchdog, V12.FastCancelWatchdog),
            "fixed_engine_subclass": issubclass(PriorityFreshnessEngine121, V12.PriorityFreshnessEngine),
        }
    )
    out = {
        "time": B._iso(),
        "live_version": LIVE_VERSION,
        "checks": checks,
        "pass": all(checks.values()),
        "orders_sent": False,
        "exchange_api_called": False,
    }
    if show:
        print("V12.1 STATIC SELF CHECK:", "PASS" if out["pass"] else "FAIL")
        for k, v in checks.items():
            print(f"  {k:<44} {v}")
        print("  ORDERS SENT: NO | EXCHANGE API CALLED: NO")
    return out


def audit_stage(session_dir, *, show=True):
    # Existing V12 audit is valid for V12.1 because CREATE_SENT now records the
    # corrected send-time invalidation result and duplicate fast cancels are
    # prevented by retirement ordering.
    return V12.audit_v12_smoke(session_dir, show=show, write_result=True)


def _write_bundle(session: Path, cfg):
    session = Path(session).resolve()
    V12._write_v12_bundle(session, cfg)
    spec = B._read(session / "live_execution_spec_v12.json", {}) or {}
    spec.update(
        {
            "live_version": LIVE_VERSION,
            "execution_parent": EXECUTION_PARENT,
            "freshness_arch_version": FRESHNESS_ARCH_VERSION,
            "priority_pending_cleared_after_unpublish": True,
            "create_send_audit_baseline": "FINAL_WATCHDOG_CERTIFICATION",
            "strategy_mechanics_changed": False,
        }
    )
    B._atomic(session / "live_execution_spec_v12_1.json", spec)


def _run_process(session, cfg):
    session = Path(session).resolve()
    _write_bundle(session, cfg)
    client = B.Q1.LiveClient()
    diag = V3._install(client)
    B._atomic(session / "balance_semantics.json", diag)
    B._post = V11._post_v11
    B._start_recorder = V4._start_recorder_fixed
    B.LiveEngine = PriorityFreshnessEngine121
    B._run_process(session, cfg)


def _stage_gate(session_dir, *, expected_q, expected_mode, show=True):
    session = Path(session_dir).resolve()
    cfg = B._read(session / "process_config.json", {}) or {}
    final = B._read(session / "final_summary.json", {}) or {}
    audit = audit_stage(session, show=show)
    q = B._f(cfg.get("quote_size"), np.nan)
    mode = str(cfg.get("mode") or "")
    passed = bool(
        final
        and np.isfinite(q)
        and abs(q - float(expected_q)) <= B.EPS
        and mode == str(expected_mode)
        and (audit.get("gates") or {}).get("promotion_ready_for_larger_smoke")
    )
    out = {
        "session_dir": str(session),
        "actual_q": q,
        "expected_q": float(expected_q),
        "actual_mode": mode,
        "expected_mode": expected_mode,
        "completed": bool(final),
        "audit": audit,
        "pass": passed,
    }
    if show:
        print("=" * 96)
        print("V12.1 STAGE PROMOTION GATE")
        print("=" * 96)
        print("session:    ", session)
        print("quote size: ", q, "expected", expected_q)
        print("mode:       ", mode)
        print("completed:  ", bool(final))
        print("GATE:       ", "PASS" if passed else "FAIL")
    return out


def gate_q1_for_q5(session_dir, *, show=True):
    return _stage_gate(
        session_dir,
        expected_q=B.SMOKE_Q,
        expected_mode="SMOKE_Q1_ONE_WINDOW",
        show=show,
    )


def gate_q5_for_q10(session_dir, *, show=True):
    return _stage_gate(
        session_dir,
        expected_q=Q5_QTY,
        expected_mode="LIVE_Q5_1H",
        show=show,
    )


def _launch(*, mode, q, hours, max_loss, min_equity, arm, expected_arm,
            prior_session=None, prior_expected_q=None, prior_expected_mode=None):
    if str(arm) != str(expected_arm):
        raise RuntimeError(
            f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected_arm!r} exactly."
        )

    if prior_session is not None:
        gate = _stage_gate(
            prior_session,
            expected_q=prior_expected_q,
            expected_mode=prior_expected_mode,
            show=True,
        )
        if not gate["pass"]:
            raise RuntimeError("Prior V12.1 stage did not pass. Refusing to arm.")

    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    static = static_self_check(show=True)
    if not static["pass"]:
        raise RuntimeError(f"V12.1 static self-check failed: {static}")

    V3._calibrated_preflight(
        quote_size=float(q),
        runtime_hours=float(hours),
        max_loss_usd=float(max_loss),
        min_equity_usd=float(min_equity),
        mode=str(mode),
        save_dir=None,
        show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_{str(mode).lower()}_v12_1").resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": str(mode),
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": LIVE_VERSION,
        "staged_launcher_version": STAGED_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_parent": RECORDING_PARENT,
        "freshness_arch_version": FRESHNESS_ARCH_VERSION,
        "recording_version": V10.RECORDING_VERSION,
        "comparison_schema_version": V10.COMPARISON_SCHEMA_VERSION,
        "order_api_safety_version": V11.ORDER_API_SAFETY_VERSION,
        "prior_stage_session": str(Path(prior_session).resolve()) if prior_session else None,
    }
    B._atomic(session / "process_config.json", cfg)
    _write_bundle(session, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v12_1",
                "--run-live-session", str(session),
                "--config", str(session / "process_config.json"),
            ],
            cwd=str(C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        fh.close()

    B._atomic(
        B.CONTROL_PATH,
        {
            "live_version": LIVE_VERSION,
            "running": True,
            "pid": p.pid,
            "session_dir": str(session),
            "mode": str(mode),
            "started_at": B._iso(),
            "config": cfg,
            "log_path": str(log),
        },
    )

    deadline = time.time() + 90.0
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-20000:] if log.exists() else ""
            raise RuntimeError(
                f"Live V12.1 process exited during startup rc={p.returncode}\n{tail}"
            )
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-20000:] if log.exists() else ""
        raise RuntimeError(f"Live V12.1 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V12.1 STAGE ARMED")
    print("  stage:  ", mode)
    print("  Q:      ", q)
    print("  hours:  ", hours)
    print("  session:", session)
    print("  pid:    ", p.pid)
    return B.live_status(show=False)


def start_live_q1(*, arm_phrase=None,
                  max_start_loss_usd=B.LOSS_LIMIT_USD,
                  min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    return _launch(
        mode="SMOKE_Q1_ONE_WINDOW",
        q=B.SMOKE_Q,
        hours=1.0,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected_arm=B.SMOKE_ARM,
    )


def start_live_q5_after_q1(*, prior_q1_session, arm_phrase=None,
                           max_start_loss_usd=B.LOSS_LIMIT_USD,
                           min_start_equity_usd=Q5_MIN_EQUITY_USD):
    return _launch(
        mode="LIVE_Q5_1H",
        q=Q5_QTY,
        hours=Q5_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected_arm=Q5_ARM,
        prior_session=prior_q1_session,
        prior_expected_q=B.SMOKE_Q,
        prior_expected_mode="SMOKE_Q1_ONE_WINDOW",
    )


def start_live_q10_after_q5(*, prior_q5_session, arm_phrase=None,
                            max_start_loss_usd=B.LOSS_LIMIT_USD,
                            min_start_equity_usd=B.FULL_MIN_EQUITY):
    return _launch(
        mode="LIVE_Q10_24H",
        q=B.FULL_Q,
        hours=B.FULL_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected_arm=Q10_ARM,
        prior_session=prior_q5_session,
        prior_expected_q=Q5_QTY,
        prior_expected_mode="LIVE_Q5_1H",
    )


def live_preflight(*, quote_size=B.SMOKE_Q, runtime_hours=1.0,
                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                   min_start_equity_usd=B.SMOKE_MIN_EQUITY, show=True):
    return V3.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
    )


def account_safety_check(*, show=True):
    return V11.account_safety_check(show=show)


def live_status(*, show=True, tail_lines=20):
    return B.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    return B.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    a = ap.parse_args()
    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}
        _run_process(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "STAGED_VERSION",
    "PriorityFreshnessEngine121",
    "FixedFastCancelWatchdog",
    "static_self_check",
    "audit_stage",
    "gate_q1_for_q5",
    "gate_q5_for_q10",
    "start_live_q1",
    "start_live_q5_after_q1",
    "start_live_q10_after_q5",
    "live_preflight",
    "account_safety_check",
    "live_status",
    "kill_and_flatten_live",
]
