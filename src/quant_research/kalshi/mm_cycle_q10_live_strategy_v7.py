from __future__ import annotations

"""V7 live runner: V6 latest-state architecture + dual-path cancel safety.

Why V7 exists
-------------
The V6 Q1 smoke fixed the stale-event/backlog problem: real orders were created
from fresh latest-state books with wall-clock elapsed gating.  The run then found
an API compatibility failure: DELETE /portfolio/events/orders/{order_id} returned
404 for an order that GET /portfolio/orders still reported as RESTING.

V7 leaves the frozen strategy and V6 execution architecture unchanged and makes
cancellation fail-closed:

1) use the current V2 cancel endpoint first;
2) if V2 cancel errors, inspect the authoritative RESTING order set;
3) if the order is still resting, fall back to the legacy compatibility cancel
   endpoint DELETE /portfolio/orders/{order_id};
4) verify the order is no longer resting before allowing any replacement;
5) if both cancellation surfaces fail while the order is still resting, trigger
   the whole strategy order group immediately and raise so shutdown/flatten runs;
6) reconcile fills plus the actual account position before any replacement.

V7 also separates stale CREATE candidates from actually SENT orders in diagnostics
and records maximum risk-tick spacing while the engine is alive.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v4 as V4
from . import mm_cycle_q10_live_strategy_v6 as V6

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V7"


class DualCancelLatestStateEngine(V6.LatestStateLiveEngine):
    """V6 engine with verified V2->legacy cancel fallback and clearer lag metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v7_cancel_v2_successes = 0
        self.v7_cancel_v2_errors = 0
        self.v7_cancel_legacy_fallbacks = 0
        self.v7_cancel_already_absent = 0
        self.v7_cancel_group_emergencies = 0
        self.v7_stale_create_blocks = 0
        self.v7_max_create_candidate_book_age_s = 0.0
        self.v7_max_sent_order_book_age_s = 0.0
        self.v7_max_risk_tick_gap_s = 0.0
        self.v7_last_risk_tick_start = time.time()

    @staticmethod
    def _fill_sum(rows):
        total = 0.0
        for r in rows or []:
            total += B._f(r.get("count_fp", r.get("count")), 0.0)
        return float(total)

    def _resting_row(self, oid):
        rows, timing = B._resting(self.client)
        row = next((r for r in rows if str(r.get("order_id") or "") == str(oid)), None)
        return row, timing

    def _emergency_group_trigger(self, *, ticker, oid, reason, detail):
        self.v7_cancel_group_emergencies += 1
        trig = B._trigger_group(self.client, self.gid)
        B._append(self.risk_log, {
            "time": B._iso(),
            "event": "V7_CANCEL_FAIL_CLOSED_GROUP_TRIGGER",
            "ticker": ticker,
            "order_id": oid,
            "reason": reason,
            "detail": detail,
            "group_trigger": trig,
        })
        return trig

    def cancel_track(self, ticker, reason):
        """Cancel one tracked order and prove it cannot remain live before replacement."""
        tr = self.active.get(ticker)
        if not tr:
            return False

        oid = str(tr["order_id"])
        old_fill = float(tr.get("last_fill", 0.0))
        submitted_qty = float(tr.get("qty", 0.0))
        before_pos = float(self.positions.get(ticker, 0.0))

        source = None
        v2_error = None
        v2_body = None
        v2_timing = None
        legacy_error = None
        legacy_body = None
        legacy_timing = None
        resting_before_fallback = None
        resting_after_fallback = None
        resting_timing_before = None
        resting_timing_after = None
        receipt_fill = old_fill

        # Primary: current V2 cancel surface.
        try:
            v2_body, v2_timing = self.client.delete(
                f"/portfolio/events/orders/{oid}",
                params={"subaccount": 0, "exchange_index": 0},
            )
            reduced_by = B._f(v2_body.get("reduced_by"), np.nan)
            if not np.isfinite(reduced_by):
                raise RuntimeError(f"V2 cancel receipt missing finite reduced_by: {v2_body}")
            if reduced_by < -B.EPS or reduced_by > submitted_qty + B.EPS:
                raise RuntimeError(
                    f"Invalid V2 cancel reduced_by={reduced_by} for submitted_qty={submitted_qty}: {v2_body}"
                )
            receipt_fill = max(old_fill, submitted_qty - max(0.0, reduced_by))
            source = "V2_CANCEL"
            self.v7_cancel_v2_successes += 1

        except Exception as exc:
            v2_error = repr(exc)
            self.v7_cancel_v2_errors += 1

            # Resting orders are the key safety question. If it is already absent,
            # do not issue a second cancel blindly; reconcile fills/position instead.
            resting_before_fallback, resting_timing_before = self._resting_row(oid)
            if resting_before_fallback is None:
                source = "V2_ERROR_ALREADY_ABSENT"
                self.v7_cancel_already_absent += 1
            else:
                # Compatibility fallback. The legacy endpoint is kept only as a
                # cancellation safety path; new order creation remains V2.
                try:
                    legacy_body, legacy_timing = self.client.delete(
                        f"/portfolio/orders/{oid}",
                        params={"subaccount": 0},
                    )
                    source = "LEGACY_CANCEL_FALLBACK"
                    self.v7_cancel_legacy_fallbacks += 1
                except Exception as legacy_exc:
                    legacy_error = repr(legacy_exc)

                # Never allow a replacement until the authoritative resting set says
                # the old order is gone. This also handles a cancel response race.
                resting_after_fallback, resting_timing_after = self._resting_row(oid)
                if resting_after_fallback is not None:
                    trig = self._emergency_group_trigger(
                        ticker=ticker,
                        oid=oid,
                        reason=reason,
                        detail={
                            "v2_error": v2_error,
                            "legacy_error": legacy_error,
                            "resting_order": resting_after_fallback,
                        },
                    )
                    raise RuntimeError(
                        "V7 fail-closed cancel: order remains RESTING after both cancel paths; "
                        f"ticker={ticker} order_id={oid} v2_error={v2_error} "
                        f"legacy_error={legacy_error} group_trigger={trig}"
                    )
                if source is None:
                    # Legacy request itself errored, but a fresh resting-set read proves
                    # the order disappeared. Treat as completed and reconcile state.
                    source = "BOTH_CANCEL_ERRORS_BUT_NOW_ABSENT"
                    self.v7_cancel_already_absent += 1

        # Best-effort fill reconciliation. Actual position below is the final exposure authority.
        fill_rows = []
        fill_timing = {}
        fill_error = None
        try:
            fill_rows, fill_timing = B._fills(self.client, oid)
            receipt_fill = max(receipt_fill, self._fill_sum(fill_rows))
        except Exception as exc:
            fill_error = repr(exc)

        final_fill = min(submitted_qty, max(old_fill, float(receipt_fill)))
        tr["last_fill"] = final_fill

        # Persist fill rows through the existing de-duplicating logger when available.
        self.record_fills(tr)

        # Actual account inventory is authoritative before any replacement can happen.
        after_pos = self.refresh_position(ticker)

        B._append(self.orders, {
            "time": B._iso(),
            "action": "CANCEL_V7_DUAL_PATH_VERIFIED",
            "ticker": ticker,
            "reason": reason,
            "track": tr,
            "cancel_source": source,
            "v2_body": v2_body,
            "v2_timing": v2_timing,
            "v2_error": v2_error,
            "legacy_body": legacy_body,
            "legacy_timing": legacy_timing,
            "legacy_error": legacy_error,
            "resting_before_fallback": resting_before_fallback,
            "resting_after_fallback": resting_after_fallback,
            "resting_timing_before": resting_timing_before,
            "resting_timing_after": resting_timing_after,
            "fill_read_timing": fill_timing,
            "fill_read_error": fill_error,
            "old_fill": old_fill,
            "final_fill": final_fill,
            "position_before": before_pos,
            "position_after": after_pos,
        })

        self.active.pop(ticker, None)
        self.counts["cancels"] += 1

        raced_fill = (
            final_fill > old_fill + B.EPS
            or abs(after_pos - before_pos) > B.EPS
        )
        if raced_fill:
            self.barrier[ticker] = self.book_version[ticker]
            self.counts["fill_events"] += 1
            self.emit(
                "FILL",
                ticker,
                role=tr["role"],
                side=tr["side"],
                qty=max(0.0, final_fill-old_fill),
                position=after_pos,
                source="v7_dual_cancel_or_position",
            )
        return raced_fill

    def risk_tick(self):
        now = time.time()
        gap = max(0.0, now - self.v7_last_risk_tick_start)
        self.v7_max_risk_tick_gap_s = max(self.v7_max_risk_tick_gap_s, gap)
        self.v7_last_risk_tick_start = now
        return super().risk_tick()

    def place(self, ticker, d, cur, elapsed):
        # Duplicate V6's final action gate only to make the diagnostics unambiguous:
        # candidate age can exceed the cap; SENT-order age cannot.
        now = time.time()
        wall_e = self.wall_elapsed(ticker, now_s=now)
        age = self.row_age_s(ticker, now_s=now)
        if np.isfinite(age):
            self.v7_max_create_candidate_book_age_s = max(
                self.v7_max_create_candidate_book_age_s, float(age)
            )

        if not np.isfinite(wall_e) or not (0.0 <= wall_e < 300.0):
            self.v7_stale_create_blocks += 1
            self.emit(
                "STALE_OR_OUTSIDE_WINDOW_CREATE_BLOCKED",
                ticker,
                wall_elapsed_s=wall_e,
                book_age_s=age,
                engine=LIVE_VERSION,
            )
            return
        if not np.isfinite(age) or age > V6.MAX_ACTION_BOOK_AGE_S:
            self.v7_stale_create_blocks += 1
            self.emit(
                "STALE_OR_OUTSIDE_WINDOW_CREATE_BLOCKED",
                ticker,
                wall_elapsed_s=wall_e,
                book_age_s=age,
                engine=LIVE_VERSION,
            )
            return
        if self.shutdown_started:
            return

        self.v7_max_sent_order_book_age_s = max(self.v7_max_sent_order_book_age_s, float(age))
        # V6 repeats the same gate immediately before POST and records wall/book age.
        return super().place(ticker, d, cur, elapsed)

    def _v7_metrics(self):
        base = dict(self._v6_metrics())
        base.update({
            "engine": LIVE_VERSION,
            "cancel_v2_successes": self.v7_cancel_v2_successes,
            "cancel_v2_errors": self.v7_cancel_v2_errors,
            "cancel_legacy_fallbacks": self.v7_cancel_legacy_fallbacks,
            "cancel_already_absent": self.v7_cancel_already_absent,
            "cancel_group_emergencies": self.v7_cancel_group_emergencies,
            "stale_create_blocks": self.v7_stale_create_blocks,
            "max_create_candidate_book_age_s": self.v7_max_create_candidate_book_age_s,
            "max_sent_order_book_age_s": self.v7_max_sent_order_book_age_s,
            "max_allowed_sent_order_book_age_s": V6.MAX_ACTION_BOOK_AGE_S,
            "max_risk_tick_gap_s": self.v7_max_risk_tick_gap_s,
        })
        return base

    @staticmethod
    def _stopped_health_from_summary(summary, metrics):
        return {
            "time": B._iso(),
            "live_version": LIVE_VERSION,
            "running": False,
            "state": "STOPPED",
            "mode": summary.get("mode"),
            "session_dir": summary.get("session_dir"),
            "quote_size": None,
            "windows_started": len(summary.get("windows_started") or []),
            "start_equity_usd": summary.get("start_equity_usd"),
            "equity_usd": summary.get("end_equity_usd"),
            "start_pnl_usd": summary.get("account_pnl_usd"),
            "kill_equity_usd": summary.get("kill_equity_usd"),
            "peak_equity_usd": summary.get("peak_equity_usd"),
            "max_peak_drawdown_usd": summary.get("max_peak_drawdown_usd"),
            "positions": {},
            "active_orders": {},
            "counts": summary.get("counts") or {},
            "shutdown_reason": summary.get("shutdown_reason"),
            "last_error": summary.get("last_error"),
            "recorder_alive": False,
            "summary": summary,
            "v7_metrics": metrics,
        }

    def health(self, force=False):
        # While running, reuse V6's rich live health and append corrected V7 metrics.
        super().health(force=force)
        if self.shutdown_started and self.final_path.exists():
            summary = B._read(self.final_path, {}) or {}
            B._atomic(self.health_path, self._stopped_health_from_summary(summary, self._v7_metrics()))
            return
        h = B._read(self.health_path, {}) or {}
        h["live_version"] = LIVE_VERSION
        h["v7_metrics"] = self._v7_metrics()
        B._atomic(self.health_path, h)

    def shutdown(self, reason):
        if self.shutdown_started:
            return
        super().shutdown(reason)
        summary = B._read(self.final_path, {}) or {}
        summary["live_wrapper_version"] = LIVE_VERSION
        summary["v7_metrics"] = self._v7_metrics()
        B._atomic(self.final_path, summary)
        B._atomic(self.health_path, self._stopped_health_from_summary(summary, self._v7_metrics()))


# ======================================================================================
# Process wiring / launchers
# ======================================================================================


def backlog_regression_check(session_dir, *, bucket_ms=250, show=True):
    return V6.backlog_regression_check(session_dir, bucket_ms=bucket_ms, show=show)


def _run_process_v7(session, cfg):
    session = Path(session).resolve()
    client = B.Q1.LiveClient()
    diag = V3._install(client)
    B._atomic(session / "balance_semantics.json", diag)

    B._start_recorder = V4._start_recorder_fixed
    B.LiveEngine = DualCancelLatestStateEngine
    B._run_process(session, cfg)


def live_preflight(*, quote_size=B.FULL_Q, runtime_hours=B.FULL_HOURS,
                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                   min_start_equity_usd=None, show=True):
    return V3.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
    )


def _launch(*, mode, q, hours, max_loss, min_equity, arm, expected):
    if str(arm) != expected:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected!r} exactly.")

    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    V3._calibrated_preflight(
        quote_size=q,
        runtime_hours=hours,
        max_loss_usd=max_loss,
        min_equity_usd=min_equity,
        mode=mode,
        save_dir=None,
        show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_{mode.lower()}_v7").resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": mode,
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": LIVE_VERSION,
        "engine_architecture": "LATEST_STATE_WALL_CLOCK_DUAL_CANCEL_V7",
        "max_action_book_age_s": V6.MAX_ACTION_BOOK_AGE_S,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v7",
                "--run-live-session", str(session),
                "--config", str(cfg_path),
            ],
            cwd=str(C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        fh.close()

    B._atomic(B.CONTROL_PATH, {
        "live_version": LIVE_VERSION,
        "running": True,
        "pid": p.pid,
        "session_dir": str(session),
        "mode": mode,
        "started_at": B._iso(),
        "config": cfg,
        "log_path": str(log),
    })

    deadline = time.time() + 90.0
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-12000:] if log.exists() else ""
            raise RuntimeError(f"Live V7 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-12000:] if log.exists() else ""
        raise RuntimeError(f"Live V7 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V7 PROCESS ARMED")
    print("  mode:    ", mode)
    print("  session: ", session)
    print("  pid:     ", p.pid)
    print(f"  Q:        {q:g} per eligible market")
    print(f"  kill:     -${max_loss:.2f} from calibrated starting TOTAL account equity")
    print(f"  stale cap:{V6.MAX_ACTION_BOOK_AGE_S:.2f}s max latest-book age at CREATE")
    print("  engine:   V6 latest-state/wall-clock risk + V7 dual cancel verification")
    print("Use live_status(); emergency stop is kill_and_flatten_live(arm_phrase='KILL_AND_FLATTEN').")
    return live_status(show=False)


def start_live_smoke_q1_one_window(*, arm_phrase=None,
                                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                                   min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    return _launch(
        mode="SMOKE_Q1_ONE_WINDOW",
        q=B.SMOKE_Q,
        hours=1.0,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected=B.SMOKE_ARM,
    )


def start_live_cycle_q10(*, arm_phrase=None, runtime_hours=B.FULL_HOURS,
                         max_start_loss_usd=B.LOSS_LIMIT_USD,
                         min_start_equity_usd=B.FULL_MIN_EQUITY):
    if abs(float(runtime_hours) - B.FULL_HOURS) > B.EPS:
        raise RuntimeError("V7 full validation is frozen to exactly 24 hours.")
    return _launch(
        mode="LIVE_Q10_24H",
        q=B.FULL_Q,
        hours=B.FULL_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected=B.FULL_ARM,
    )


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
        _run_process_v7(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "DualCancelLatestStateEngine",
    "backlog_regression_check",
    "live_preflight",
    "start_live_smoke_q1_one_window",
    "start_live_cycle_q10",
    "live_status",
    "kill_and_flatten_live",
]
