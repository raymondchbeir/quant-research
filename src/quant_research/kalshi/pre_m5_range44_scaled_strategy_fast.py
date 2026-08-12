from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pandas as pd

from .pre_m5_range44_scaled_strategy import _parse_trade
from .pre_m5_range44_scaled_strategy_v2 import (
    INTERVAL_SEC,
    Range44Q15Q5ProspectiveMonitorV2,
)

# This module is an implementation optimization only. It deliberately keeps the
# exact RANGE44_Q15_Q5_PROSPECTIVE_V2 strategy identity/output directory/freeze
# semantics from pre_m5_range44_scaled_strategy_v2.

_MONITOR = None
_MONITOR_THREAD = None
_MONITOR_LOCK = threading.RLock()

_PROGRESS_BYTES = 64 * 1024 * 1024
_YIELD_BYTES = 32 * 1024 * 1024
_YIELD_SECONDS = 0.01


class Range44Q15Q5ProspectiveMonitorFast(Range44Q15Q5ProspectiveMonitorV2):
    """V2 strategy with a filtered, bounded, cooperative historical catch-up.

    Live prospective behavior is inherited unchanged from V2:
      * new Q15/Q5 freeze is captured before live processing;
      * live finite-flow replay starts at that exact trades-file byte boundary;
      * only appended trade bytes are processed after startup.

    Historical catch-up is optimized here:
      * only tickers with authoritative FULL Q3 fills need flow reconstruction;
      * raw lines are filtered with one compiled bytes regex before json.loads;
      * pandas timestamp parsing is therefore avoided for irrelevant rows;
      * historical trades are applied directly to capacity state instead of being
        retained in the live recent-trade buffer;
      * the worker yields periodically so recorder/shadow threads are not starved.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catchup_target_tickers = 0
        self.catchup_candidate_lines = 0
        self.catchup_valid_trades = 0

    @staticmethod
    def _target_regex(tickers):
        escaped = [re.escape(str(t).encode("utf-8")) for t in sorted(set(tickers))]
        if not escaped:
            return None
        # Recorder rows may expose either ticker or market_ticker at top/raw level.
        return re.compile(
            rb'"(?:ticker|market_ticker)"\s*:\s*"(?:'
            + rb"|".join(escaped)
            + rb')"'
        )

    def _catchup_worker(self):
        try:
            with self.lock:
                self.catchup_state = "SCANNING"
                self.catchup_error = None

            shadow = self._read_shadow()
            old_shadow = shadow[shadow["decision_time"].le(self.freeze_at)].copy()

            # FiniteFlowCapacityTracker only creates replay states for FULL Q3 fills.
            # No-fill and partial rows are calibrated directly from authoritative Q3.
            self.catchup_capacity.sync_shadow(old_shadow)
            targets = set(self.catchup_capacity.states.keys())
            target_re = self._target_regex(targets)

            with self.lock:
                self.catchup_target_tickers = len(targets)

            end = int(self.catchup_trade_end_offset)
            last_progress = 0
            last_yield = 0
            candidate_lines = 0
            valid_trades = 0

            if end <= 0 or target_re is None:
                with self.lock:
                    self.catchup_bytes_scanned = max(0, end)
                    self.catchup_candidate_lines = 0
                    self.catchup_valid_trades = 0
                    self.catchup_state = "READY"
                self.refresh()
                return

            with self.trades_file.open("rb") as f:
                while not self.stop_event.is_set() and f.tell() < end:
                    pos = f.tell()
                    raw = f.readline()

                    if not raw:
                        break

                    if f.tell() > end or not raw.endswith(b"\n"):
                        f.seek(pos)
                        break

                    # Fast path: the overwhelming majority of the multi-GB trade file
                    # is irrelevant to the handful of full-Q3 opportunities.
                    if target_re.search(raw) is not None:
                        candidate_lines += 1
                        try:
                            obj = json.loads(raw)
                        except Exception:
                            obj = None

                        if obj is not None:
                            trade = _parse_trade(obj)
                            if trade is not None:
                                state = self.catchup_capacity.states.get(trade["ticker"])
                                if state is not None:
                                    # Historical replay does not need the tracker's
                                    # recent deque; apply directly and stay memory-flat.
                                    self.catchup_capacity._apply_trade(state, trade)
                                    valid_trades += 1

                    current = f.tell()

                    if current - last_progress >= _PROGRESS_BYTES:
                        with self.lock:
                            self.catchup_bytes_scanned = current
                            self.catchup_candidate_lines = candidate_lines
                            self.catchup_valid_trades = valid_trades
                        last_progress = current

                    # Be cooperative with the live recorder/primary threads. The
                    # historical pass is research convenience, never latency-critical.
                    if current - last_yield >= _YIELD_BYTES:
                        time.sleep(_YIELD_SECONDS)
                        last_yield = current

                final_pos = min(f.tell(), end)

            with self.lock:
                self.catchup_bytes_scanned = final_pos
                self.catchup_candidate_lines = candidate_lines
                self.catchup_valid_trades = valid_trades

            if self.stop_event.is_set():
                return

            with self.lock:
                self.catchup_state = "READY"

            # One rebuild now makes the completed historical catch-up visible.
            self.refresh()

        except Exception as exc:
            with self.lock:
                self.catchup_state = "ERROR"
                self.catchup_error = repr(exc)

    def status(self):
        out = super().status()
        with self.lock:
            out.update({
                "catchup_target_tickers": self.catchup_target_tickers,
                "catchup_candidate_lines": self.catchup_candidate_lines,
                "catchup_valid_trades": self.catchup_valid_trades,
            })
        return out


def start_range44_q15q5_monitor(session_dir, interval_sec=INTERVAL_SEC, show=True):
    global _MONITOR, _MONITOR_THREAD

    with _MONITOR_LOCK:
        if _MONITOR is not None and _MONITOR_THREAD is not None and _MONITOR_THREAD.is_alive():
            same = Path(session_dir).resolve() == _MONITOR.session_dir.resolve()
            if same:
                if show:
                    print("RANGE44 Q15/Q5 optimized monitor is already running for this session.")
                return _MONITOR.status()
            raise RuntimeError("Q15/Q5 optimized monitor is already running on a different session.")

        monitor = Range44Q15Q5ProspectiveMonitorFast(
            session_dir=session_dir,
            interval_sec=interval_sec,
        )

        # This refresh is intentionally fast: V2 initialized the live tracker at the
        # frozen file tail, so it only sees bytes appended after the new freeze.
        first = monitor.refresh()

        thread = threading.Thread(
            target=monitor.loop,
            name="kalshi-range44-q15q5-live-fast",
            daemon=True,
        )
        _MONITOR = monitor
        _MONITOR_THREAD = thread
        thread.start()

        # Historical reconstruction is explicitly separate and never blocks startup.
        monitor.start_catchup()

    if show:
        print("RANGE44 Q15/Q5 optimized monitor STARTED (READ-ONLY)")
        print("Session:", monitor.session_dir)
        print("Prospective frozen at:", monitor.freeze_at)
        print("Historical catch-up anchor:", monitor.catchup_anchor)
        print("Live path: incremental from frozen trades-file tail")
        print("Catch-up: filtered/cooperative background scan (NOT OOS)")

    return first


def range44_q15q5_status(show=True):
    with _MONITOR_LOCK:
        if _MONITOR is None:
            out = {
                "running": False,
                "session_dir": None,
                "summary": pd.DataFrame(),
                "latest_windows": pd.DataFrame(),
                "catchup_state": "STOPPED",
                "catchup_progress_pct": 0.0,
            }
        else:
            out = _MONITOR.status()

    if show:
        print("RANGE44 Q15/Q5 optimized:", "RUNNING" if out.get("running") else "STOPPED")
        print("Session:", out.get("session_dir"))
        print("Prospective freeze:", out.get("frozen_at_utc"))
        print(
            "Catch-up:",
            out.get("catchup_state"),
            f"{out.get('catchup_progress_pct', 0.0):.1f}%",
            f"| targets={out.get('catchup_target_tickers', 0)}",
            f"| candidate lines={out.get('catchup_candidate_lines', 0)}",
        )
        if out.get("last_error"):
            print("Last error:", out.get("last_error"))
        if out.get("catchup_error"):
            print("Catch-up error:", out.get("catchup_error"))
        summary = out.get("summary")
        if isinstance(summary, pd.DataFrame) and len(summary):
            try:
                from IPython.display import display
                display(summary.round(4))
            except Exception:
                print(summary.round(4).to_string(index=False))

    return out


def stop_range44_q15q5_monitor(show=True):
    global _MONITOR, _MONITOR_THREAD

    with _MONITOR_LOCK:
        if _MONITOR is None:
            return {"running": False}
        monitor = _MONITOR
        thread = _MONITOR_THREAD
        monitor.stop()

    if thread is not None and thread.is_alive():
        thread.join(timeout=max(2.0, monitor.interval_sec + 1.0))

    with _MONITOR_LOCK:
        _MONITOR = None
        _MONITOR_THREAD = None

    if show:
        print("RANGE44 Q15/Q5 optimized monitor STOPPED")

    return {"running": False}


__all__ = [
    "Range44Q15Q5ProspectiveMonitorFast",
    "start_range44_q15q5_monitor",
    "range44_q15q5_status",
    "stop_range44_q15q5_monitor",
]
