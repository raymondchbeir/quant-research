from __future__ import annotations

"""Production wrapper for mm_cycle_q10_live_strategy_v1.

V2 fixes two safety-critical issues before any real strategy run:

1) Kalshi portfolio_value is treated as TOTAL portfolio value, not added to cash.
   The Oct-2025 Kalshi changelog describes portfolio_value as available balance plus
   current position value.  Therefore the $50 start-to-current kill is based on
   portfolio_value alone.  Cash remains a separately logged diagnostic.

2) Cancel/reprice is fill-aware.  Before a resting order is replaced, V2 verifies
   the final exchange order state.  If a fill raced the cancel, it refreshes the
   actual position, installs the frozen next-book-event barrier, and DOES NOT place
   another entry from stale flat inventory.

Use THIS module from the notebook, not V1 directly.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V2"


def _equity_total(body):
    """Kalshi portfolio_value is total portfolio value in cents; do not double-count cash."""
    cash = B._f((body or {}).get("balance"), 0.0) / 100.0
    total = B._f((body or {}).get("portfolio_value"), np.nan) / 100.0
    if not np.isfinite(total):
        raise RuntimeError("GET /portfolio/balance did not return a finite portfolio_value.")
    return {
        "cash_balance_usd": cash,
        "portfolio_value_usd": total,
        "equity_usd": total,
        "updated_ts": (body or {}).get("updated_ts"),
    }


class SafeLiveEngine(B.LiveEngine):
    def cancel_track(self, ticker, reason):
        """Cancel, then verify whether a fill raced the cancel before any replacement."""
        tr = self.active.get(ticker)
        if not tr:
            return False

        res = B._cancel(self.client, tr["order_id"])
        if not res.get("ok"):
            raise RuntimeError(f"Cancel failed {ticker}: {res}")

        # Resolve the authoritative post-cancel order state.  The cancel response can
        # race a fill, so never assume the cached fill_count is final.
        row = res.get("order") if isinstance(res, dict) else None
        if not row:
            row, _ = B._get_order(self.client, tr["order_id"])

        final_fill = B._f(row.get("fill_count_fp", row.get("fill_count")), tr.get("last_fill", 0.0))
        old_fill = float(tr.get("last_fill", 0.0))
        tr["last_fill"] = max(old_fill, final_fill)
        self.record_fills(tr)
        B._append(self.orders, {
            "time": B._iso(), "action": "CANCEL_VERIFIED", "ticker": ticker,
            "reason": reason, "track": tr, "result": res, "post_cancel_order": row,
            "old_fill": old_fill, "final_fill": final_fill,
        })
        self.active.pop(ticker, None)
        self.counts["cancels"] += 1

        raced_fill = final_fill > old_fill + B.EPS
        if raced_fill:
            p = self.refresh_position(ticker)
            self.barrier[ticker] = self.book_version[ticker]
            self.counts["fill_events"] += 1
            self.emit(
                "FILL", ticker, role=tr["role"], side=tr["side"],
                qty=final_fill-old_fill, position=p, source="cancel_verification",
            )
        return raced_fill

    def reconcile(self, ticker, cur, elapsed):
        d = self.desired(ticker, cur, elapsed)
        tr = self.active.get(ticker)
        if d is None:
            if tr:
                self.cancel_track(ticker, "DESIRED_NONE")
            return
        if self.same(tr, d):
            return

        if tr:
            raced_fill = self.cancel_track(ticker, "REPRICE_ROLE_OR_QTY")
            if raced_fill:
                # Frozen rule: after any fill, wait for the NEXT book event.
                return
            # Even when no fill was reported, refresh actual position before replacing.
            self.refresh_position(ticker)
            d = self.desired(ticker, cur, elapsed)
            if d is None:
                return

        self.place(ticker, d, cur, elapsed)


def _install():
    B._equity = _equity_total
    B.LiveEngine = SafeLiveEngine


def live_preflight(*, quote_size=B.FULL_Q, runtime_hours=B.FULL_HOURS,
                   max_start_loss_usd=B.LOSS_LIMIT_USD, min_start_equity_usd=None,
                   show=True):
    _install()
    return B.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
    )


def _launch(*, mode, q, hours, max_loss, min_equity, arm, expected):
    _install()
    if str(arm) != expected:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected!r} exactly.")
    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    B._preflight(
        quote_size=q, runtime_hours=hours, max_loss_usd=max_loss,
        min_equity_usd=min_equity, mode=mode, save_dir=None, show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_{mode.lower()}_v2").resolve()
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
                sys.executable, "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v2",
                "--run-live-session", str(session), "--config", str(cfg_path),
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
            raise RuntimeError(f"Live V2 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-8000:] if log.exists() else ""
        raise RuntimeError(f"Live V2 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V2 PROCESS ARMED")
    print("  mode:   ", mode)
    print("  session:", session)
    print("  pid:    ", p.pid)
    print(f"  Q:       {q:g} per market")
    print(f"  kill:    -${max_loss:.2f} from STARTING TOTAL PORTFOLIO VALUE")
    print("Use live_status(); emergency stop is kill_and_flatten_live(arm_phrase='KILL_AND_FLATTEN').")
    return live_status(show=False)


def start_live_smoke_q1_one_window(*, arm_phrase=None,
                                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                                   min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    return _launch(
        mode="SMOKE_Q1_ONE_WINDOW", q=B.SMOKE_Q, hours=1.0,
        max_loss=float(max_start_loss_usd), min_equity=float(min_start_equity_usd),
        arm=arm_phrase, expected=B.SMOKE_ARM,
    )


def start_live_cycle_q10(*, arm_phrase=None, runtime_hours=B.FULL_HOURS,
                         max_start_loss_usd=B.LOSS_LIMIT_USD,
                         min_start_equity_usd=B.FULL_MIN_EQUITY):
    if abs(float(runtime_hours) - B.FULL_HOURS) > B.EPS:
        raise RuntimeError("V2 full validation is frozen to exactly 24 hours.")
    return _launch(
        mode="LIVE_Q10_24H", q=B.FULL_Q, hours=B.FULL_HOURS,
        max_loss=float(max_start_loss_usd), min_equity=float(min_start_equity_usd),
        arm=arm_phrase, expected=B.FULL_ARM,
    )


def live_status(*, show=True, tail_lines=20):
    _install()
    return B.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    _install()
    return B.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    a = ap.parse_args()
    _install()
    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}
        B._run_process(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "live_preflight",
    "start_live_smoke_q1_one_window",
    "start_live_cycle_q10",
    "live_status",
    "kill_and_flatten_live",
]
