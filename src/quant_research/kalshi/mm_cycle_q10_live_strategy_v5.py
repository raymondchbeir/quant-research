from __future__ import annotations

"""V5 live wrapper for frozen Candidate-C / CYCLE_ALWAYS_EXIT validation.

V5 keeps V4's recorder-startup fix, V3's calibrated equity semantics, and V2's
fill-aware cancel/reprice intent, while fixing the live cancel verification path.

Observed live failure
---------------------
Kalshi's V2 cancel endpoint returns a compact receipt with ``reduced_by`` rather
than a full order object. V2 canceled the order successfully, then immediately
called legacy GET /portfolio/orders/{order_id} to reconstruct the final fill
count. In the Q1 smoke test that GET returned 404 for the just-canceled V2 order,
which V2 treated as fatal even though the account ended flat and with no resting
strategy orders.

V5 treats the V2 cancel receipt as authoritative:

    final_fill = submitted_qty - reduced_by

This detects a fill that raced cancellation without requiring a post-cancel GET.
If DELETE itself reports the order as already gone, V5 verifies that the order is
not currently resting before treating it as already complete, then reconciles
fills and the actual position. Strategy mechanics, thresholds, sizing, M0-M5
window, fees, and risk limits are otherwise unchanged.
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
from . import mm_cycle_q10_live_strategy_v2 as V2
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v4 as V4

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V5"


class CancelReceiptSafeLiveEngine(V2.SafeLiveEngine):
    """Use V2 cancel receipts + actual account state; never require post-cancel GET."""

    @staticmethod
    def _fill_sum(rows):
        total = 0.0
        for r in rows or []:
            total += B._f(r.get("count_fp", r.get("count")), 0.0)
        return float(total)

    def cancel_track(self, ticker, reason):
        tr = self.active.get(ticker)
        if not tr:
            return False

        oid = str(tr["order_id"])
        old_fill = float(tr.get("last_fill", 0.0))
        submitted_qty = float(tr.get("qty", 0.0))
        before_pos = float(self.positions.get(ticker, 0.0))

        cancel_result = None
        cancel_error = None
        already_done = False
        resting_check_timing = None
        fill_read_timing = None
        receipt_fill = old_fill

        try:
            body, timing = self.client.delete(
                f"/portfolio/events/orders/{oid}",
                params={"subaccount": 0, "exchange_index": 0},
            )
            reduced_by = B._f(body.get("reduced_by"), np.nan)
            if not np.isfinite(reduced_by):
                raise RuntimeError(f"V2 cancel receipt missing finite reduced_by: {body}")
            if reduced_by < -B.EPS or reduced_by > submitted_qty + B.EPS:
                raise RuntimeError(
                    f"Invalid V2 cancel reduced_by={reduced_by} for submitted_qty={submitted_qty}: {body}"
                )

            # Matching-engine receipt is authoritative for how much remained to cancel.
            receipt_fill = max(old_fill, submitted_qty - max(0.0, reduced_by))
            cancel_result = {
                "ok": True,
                "body": body,
                "timing": timing,
                "receipt_fill": receipt_fill,
            }

        except Exception as exc:
            cancel_error = repr(exc)

            # Fail closed unless the order is demonstrably no longer resting.
            resting, resting_check_timing = B._resting(self.client)
            live_row = next(
                (r for r in resting if str(r.get("order_id") or "") == oid),
                None,
            )
            if live_row is not None:
                raise RuntimeError(
                    f"Cancel failed and order is still resting {ticker} {oid}: {cancel_error}; row={live_row}"
                )

            # If DELETE says already gone and it is absent from the authoritative resting set,
            # reconcile any fills that completed before it disappeared.
            fill_rows, fill_read_timing = B._fills(self.client, oid)
            receipt_fill = max(old_fill, self._fill_sum(fill_rows))
            already_done = True
            cancel_result = {
                "ok": True,
                "already_done": True,
                "delete_error": cancel_error,
                "resting_check_timing": resting_check_timing,
                "fill_read_timing": fill_read_timing,
                "receipt_fill": receipt_fill,
            }

        final_fill = min(submitted_qty, max(old_fill, float(receipt_fill)))
        tr["last_fill"] = final_fill

        # Persist fill rows if portfolio/fills has them; receipt accounting still protects
        # the race even if that read path is momentarily behind the matching engine.
        self.record_fills(tr)

        # Actual account exposure is the final authority before any replacement order.
        after_pos = self.refresh_position(ticker)

        B._append(self.orders, {
            "time": B._iso(),
            "action": "CANCEL_V5_RECEIPT_VERIFIED",
            "ticker": ticker,
            "reason": reason,
            "track": tr,
            "cancel_result": cancel_result,
            "old_fill": old_fill,
            "final_fill": final_fill,
            "position_before": before_pos,
            "position_after": after_pos,
            "already_done": already_done,
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
                qty=max(0.0, final_fill - old_fill),
                position=after_pos,
                source="v2_cancel_receipt_or_position",
            )

        return raced_fill


def _run_process_v5(session, cfg):
    session = Path(session).resolve()

    # V3: calibrated equity semantics + baseline V2 installation.
    client = B.Q1.LiveClient()
    diag = V3._install(client)
    B._atomic(session / "balance_semantics.json", diag)

    # V4: fixed raw recorder startup. V5: corrected cancel verification.
    B._start_recorder = V4._start_recorder_fixed
    B.LiveEngine = CancelReceiptSafeLiveEngine

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
    session = (B.ROOT / f"{stamp}_{mode.lower()}_v5").resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": mode,
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": LIVE_VERSION,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v5",
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
            tail = log.read_text(encoding="utf-8")[-8000:] if log.exists() else ""
            raise RuntimeError(f"Live V5 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-8000:] if log.exists() else ""
        raise RuntimeError(f"Live V5 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V5 PROCESS ARMED")
    print("  mode:   ", mode)
    print("  session:", session)
    print("  pid:    ", p.pid)
    print(f"  Q:       {q:g} per eligible market")
    print(f"  kill:    -${max_loss:.2f} from calibrated starting TOTAL account equity")
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
        raise RuntimeError("V5 full validation is frozen to exactly 24 hours.")
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
        _run_process_v5(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "live_preflight",
    "start_live_smoke_q1_one_window",
    "start_live_cycle_q10",
    "live_status",
    "kill_and_flatten_live",
]
