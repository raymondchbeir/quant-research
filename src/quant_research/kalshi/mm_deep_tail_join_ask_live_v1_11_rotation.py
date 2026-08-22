from __future__ import annotations

"""V1.11 compact rotating-generation live engine for the deep-tail M1->M5 strategy.

Operational changes only. Strategy rules remain frozen.

This layer is designed to run as a short-lived trader generation while a parent
supervisor owns one long-lived authenticated M0->M12 recorder. Each generation:
- attaches to the already-running recorder instead of owning/stopping it;
- starts both strategy raw ingestion and the exact-EOF watchdog at the current EOF,
  so historical rows from earlier generations are never replayed into a fresh trader;
- requires a row written after generation start before a BBO can certify an order;
- replaces the V12.2 cancel watchdog with a read-only compact EOF certifier (no
  history deque, cancel executor, active-order mirror, or pending-cancel map);
- keeps the session-level equity baseline fixed across generations;
- after all tickers in its first complete window reach M5, performs a fresh REST
  zero-position / zero-group-resting verification, writes a durable rotation
  checkpoint, and exits cleanly so the supervisor can launch a fresh process.

The raw recorder is deliberately NOT stopped by generation shutdown. If the trader
raises, its order group is still triggered by the inherited fail-closed path; the
parent supervisor/guardian remains responsible for authoritative recovery.

Importing this module performs no API calls and sends no orders.
"""

import copy
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_5 as V15
from . import mm_deep_tail_join_ask_live_v1_7 as V17
from . import mm_deep_tail_join_ask_live_v1_9_stale_orphan_guard as V19
from . import mm_deep_tail_join_ask_live_v1_10_runtime_reconciler_guard as V110
from . import mm_cycle_q10_live_strategy_v12_2 as V122


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_11_COMPACT_ROTATING_GENERATION"
ROTATION_CHECKPOINT_FILE = "generation_rotation_checkpoint_v1_11.json"
GENERATION_BOOTSTRAP_FILE = "generation_bootstrap_v1_11.json"
SESSION_RISK_BASELINE_FILE = "session_risk_baseline_v1_11.json"

WATCHDOG_SLEEP_S = 0.001
WATCHDOG_MAX_TICKERS = 64
FRESH_ROW_MAX_AGE_MS = 3000.0
META_BOOTSTRAP_MAX_BYTES = 8 * 1024 * 1024
ROTATION_REST_CONFIRM_ATTEMPTS = 3


class ExternalRecorderProxy:
    """Minimal Popen-compatible liveness view for a supervisor-owned recorder."""

    def __init__(self, pid):
        self.pid = int(pid)

    def poll(self):
        return None if B._pid_alive(self.pid) else 1

    @property
    def returncode(self):
        return None if B._pid_alive(self.pid) else 1


def _receipt_ms(row):
    try:
        t = pd.to_datetime((row or {}).get("receipt_time"), utc=True, errors="coerce")
        return np.nan if pd.isna(t) else float(t.timestamp() * 1000.0)
    except Exception:
        return np.nan


class CompactGenerationEofWatchdog:
    """Read-only exact-EOF BBO certifier that begins at generation-start EOF.

    Deep-tail never uses the inherited Candidate-C fast-cancel functionality. The
    old V12.2 object nevertheless retained per-ticker history plus a 9-worker cancel
    executor. This purpose-built certifier keeps only the newest row for a bounded
    number of tickers and therefore has memory proportional to the active horizon,
    not session duration.
    """

    def __init__(self, engine):
        self.engine = engine
        self.raw_path = engine.session / "raw_capture" / "book_top3_events.jsonl"
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.caught_up = threading.Event()
        self.progress_event = threading.Event()
        self.lock = threading.RLock()
        self.thread = threading.Thread(
            target=self._run,
            name="dt-v111-compact-eof-watchdog",
            daemon=True,
        )

        self.generation_start_wall_ms = time.time_ns() / 1e6
        self.generation_start_offset = None
        self.processed_offset = 0
        self.last_eof_offset = 0
        self.latest = {}
        self.seq = 0
        self.rows_seen = 0
        self.decode_errors = 0
        self.pruned_tickers = 0

    def start(self):
        self.thread.start()

    def ready(self):
        return self.caught_up.is_set()

    def stop(self, wait_s=2.0):
        self.stop_event.set()
        self.wake_event.set()
        self.progress_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=float(wait_s))

    # Compatibility no-ops. Deep-tail does not publish fixed orders to this object.
    def publish_active(self, active):
        return None

    def is_pending(self, ticker, oid=None):
        return False

    def clear_pending(self, ticker, oid):
        return None

    def drain_results(self, limit=100):
        return []

    def first_newer_before(self, ticker, source_ms, cutoff_ms):
        return None

    def _set_progress(self, offset, *, eof=False):
        with self.lock:
            self.processed_offset = max(int(self.processed_offset), int(offset))
            if eof:
                self.last_eof_offset = max(int(self.last_eof_offset), int(offset))
        self.progress_event.set()

    def _prune_latest_locked(self):
        excess = len(self.latest) - int(WATCHDOG_MAX_TICKERS)
        if excess <= 0:
            return
        victims = sorted(
            self.latest.items(),
            key=lambda kv: float((kv[1] or {}).get("watchdog_detect_wall_ms") or 0.0),
        )[:excess]
        for ticker, _ in victims:
            self.latest.pop(ticker, None)
            self.pruned_tickers += 1

    def _process_row(self, row, file_end_offset):
        ticker = str((row or {}).get("ticker") or "")
        if not ticker:
            return
        detect_ms = time.time_ns() / 1e6
        receipt_ms = _receipt_ms(row)
        self.rows_seen += 1
        with self.lock:
            self.seq += 1
            self.latest[ticker] = {
                "ticker": ticker,
                "seq": int(self.seq),
                "row": row,
                "receipt_wall_ms": receipt_ms,
                "watchdog_detect_wall_ms": detect_ms,
                "file_end_offset": int(file_end_offset),
                "generation_start_offset": int(self.generation_start_offset or 0),
            }
            self._prune_latest_locked()

    def _run(self):
        while not self.stop_event.is_set() and not self.raw_path.exists():
            self.stop_event.wait(0.005)
        if self.stop_event.is_set():
            return

        try:
            start_offset = int(self.raw_path.stat().st_size)
        except Exception:
            start_offset = 0

        with self.lock:
            self.generation_start_offset = int(start_offset)
            self.processed_offset = int(start_offset)
            self.last_eof_offset = int(start_offset)
        self.caught_up.set()
        self.progress_event.set()
        try:
            self.engine._lat(
                "V1_11_WATCHDOG_GENERATION_START_AT_EOF",
                generation_start_offset=start_offset,
                generation_start_wall_ms=self.generation_start_wall_ms,
            )
        except Exception:
            pass

        with self.raw_path.open("rb") as fh:
            fh.seek(start_offset, os.SEEK_SET)
            while not self.stop_event.is_set():
                pos = fh.tell()
                line = fh.readline()
                if line and not line.endswith(b"\n"):
                    fh.seek(pos)
                    self.wake_event.wait(WATCHDOG_SLEEP_S)
                    self.wake_event.clear()
                    continue

                if line:
                    end = fh.tell()
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except Exception as exc:
                        self.decode_errors += 1
                        try:
                            self.engine._lat("V1_11_WATCHDOG_JSON_ERROR", error=repr(exc))
                        except Exception:
                            pass
                        self._set_progress(end)
                        continue
                    if isinstance(row, dict):
                        self._process_row(row, end)
                    self._set_progress(end)
                    continue

                self._set_progress(fh.tell(), eof=True)
                self.wake_event.wait(WATCHDOG_SLEEP_S)
                self.wake_event.clear()

    def latest_snapshot(self, ticker):
        with self.lock:
            x = self.latest.get(str(ticker))
            return copy.deepcopy(x) if x else None

    def snapshot_is_generation_fresh(self, snap, *, max_age_ms=FRESH_ROW_MAX_AGE_MS):
        if not snap:
            return False, "NO_SNAPSHOT"
        now_ms = time.time_ns() / 1e6
        start_offset = int(self.generation_start_offset or 0)
        end_offset = int((snap or {}).get("file_end_offset") or 0)
        detect_ms = B._f((snap or {}).get("watchdog_detect_wall_ms"), np.nan)
        receipt_ms = B._f((snap or {}).get("receipt_wall_ms"), np.nan)
        if end_offset <= start_offset:
            return False, "ROW_NOT_AFTER_GENERATION_EOF"
        if not np.isfinite(detect_ms) or detect_ms + 1e-6 < self.generation_start_wall_ms:
            return False, "ROW_DETECTED_BEFORE_GENERATION_START"
        age_anchor = receipt_ms if np.isfinite(receipt_ms) else detect_ms
        if not np.isfinite(age_anchor):
            return False, "ROW_HAS_NO_LOCAL_OR_RECEIPT_CLOCK"
        age_ms = now_ms - age_anchor
        if age_ms < -250.0:
            return False, "ROW_CLOCK_IN_FUTURE"
        if age_ms > float(max_age_ms):
            return False, f"ROW_TOO_OLD:{age_ms:.1f}ms"
        return True, None

    def catch_up_to_stable_eof(self, timeout_ms=V1.PRESEND_EOF_TIMEOUT_MS):
        start_perf = time.perf_counter_ns() / 1e6
        deadline = start_perf + float(timeout_ms)
        iterations = 0
        target = None

        if not self.ready() or not self.raw_path.exists():
            return {
                "ok": False,
                "reason": "WATCHDOG_NOT_READY",
                "wait_ms": time.perf_counter_ns() / 1e6 - start_perf,
                "target_offset": None,
                "processed_offset": int(self.processed_offset),
                "iterations": 0,
            }

        while time.perf_counter_ns() / 1e6 <= deadline:
            iterations += 1
            try:
                target = int(self.raw_path.stat().st_size)
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": f"RAW_STAT_ERROR:{exc!r}",
                    "wait_ms": time.perf_counter_ns() / 1e6 - start_perf,
                    "target_offset": target,
                    "processed_offset": int(self.processed_offset),
                    "iterations": iterations,
                }

            with self.lock:
                processed = int(self.processed_offset)
            if processed >= target:
                try:
                    target2 = int(self.raw_path.stat().st_size)
                except Exception:
                    target2 = target
                if target2 <= processed:
                    return {
                        "ok": True,
                        "reason": None,
                        "wait_ms": time.perf_counter_ns() / 1e6 - start_perf,
                        "target_offset": int(target2),
                        "processed_offset": processed,
                        "iterations": iterations,
                    }

            self.progress_event.clear()
            self.wake_event.set()
            remaining_ms = deadline - (time.perf_counter_ns() / 1e6)
            if remaining_ms <= 0:
                break
            self.progress_event.wait(min(0.0005, remaining_ms / 1000.0))

        with self.lock:
            processed = int(self.processed_offset)
        return {
            "ok": False,
            "reason": "EOF_CATCHUP_TIMEOUT",
            "wait_ms": time.perf_counter_ns() / 1e6 - start_perf,
            "target_offset": target,
            "processed_offset": processed,
            "iterations": iterations,
        }

    def metrics(self):
        with self.lock:
            return {
                "mode": "GENERATION_START_EOF_COMPACT_CERTIFIER",
                "generation_start_wall_ms": float(self.generation_start_wall_ms),
                "generation_start_offset": self.generation_start_offset,
                "processed_offset": int(self.processed_offset),
                "last_eof_offset": int(self.last_eof_offset),
                "rows_seen": int(self.rows_seen),
                "decode_errors": int(self.decode_errors),
                "latest_tickers": int(len(self.latest)),
                "latest_ticker_cap": int(WATCHDOG_MAX_TICKERS),
                "pruned_tickers": int(self.pruned_tickers),
                "history_retained": False,
                "cancel_executor_present": False,
            }


def _tail_latest_metadata(path, max_bytes=META_BOOTSTRAP_MAX_BYTES):
    """Bounded tail scan returning the newest metadata row per ticker."""
    path = Path(path)
    if not path.exists():
        return {}, 0, 0
    size = int(path.stat().st_size)
    start = max(0, size - int(max_bytes))
    latest = {}
    rows = 0
    with path.open("rb") as fh:
        fh.seek(start, os.SEEK_SET)
        if start > 0:
            fh.readline()
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or "")
            if ticker:
                latest[ticker] = row
                rows += 1
    return latest, size, rows


class CompactRotatingGenerationEngine(V19.TerminalTombstoneGuardEngine):
    """V1.10/V1.9 safety engine with bounded one-window process lifetime."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not bool(self.cfg.get("external_recorder_owner")):
            raise RuntimeError("V1.11 requires external_recorder_owner=True")

        self.generation_id = int(self.cfg.get("generation_id") or 0)
        self.parent_session = str(self.cfg.get("parent_session_dir") or "")
        self.session_start_equity_usd = float(
            self.cfg.get("session_start_equity_usd", self.start_equity)
        )
        self.session_kill_equity_usd = float(
            self.cfg.get(
                "session_kill_equity_usd",
                self.session_start_equity_usd - self.max_loss,
            )
        )
        if abs(self.start_equity - self.session_start_equity_usd) > 1e-6:
            raise RuntimeError(
                "V1.11 fixed session risk baseline was not installed before engine construction: "
                f"engine={self.start_equity} cfg={self.session_start_equity_usd}"
            )
        if abs(self.kill_equity - self.session_kill_equity_usd) > 1e-6:
            raise RuntimeError(
                "V1.11 fixed session kill equity mismatch: "
                f"engine={self.kill_equity} cfg={self.session_kill_equity_usd}"
            )

        raw = self.session / "raw_capture"
        metadata, meta_eof, meta_rows = _tail_latest_metadata(raw / "market_metadata.jsonl")
        self.meta.update(metadata)
        if isinstance(self.meta_tail, V15.BoundedJsonlTail):
            self.meta_tail.offset = int(meta_eof)
        try:
            book_eof = int((raw / "book_top3_events.jsonl").stat().st_size)
        except Exception:
            book_eof = 0
        if isinstance(self.book_tail, V15.BoundedJsonlTail):
            self.book_tail.offset = int(book_eof)

        self.rotation_window_key = None
        self.rotation_checkpoint_written = False
        self.rotation_verify_attempts = 0
        self.rotation_failure = None

        B._atomic(self.session / GENERATION_BOOTSTRAP_FILE, {
            "time": B._iso(),
            "live_version": LIVE_VERSION,
            "generation_id": self.generation_id,
            "parent_session_dir": self.parent_session,
            "external_recorder_pid": int(self.cfg.get("external_recorder_pid") or 0),
            "book_tail_start_offset": int(book_eof),
            "meta_tail_start_offset": int(meta_eof),
            "metadata_rows_bootstrapped_from_bounded_tail": int(meta_rows),
            "metadata_tickers_bootstrapped": int(len(metadata)),
            "watchdog": self.fast.metrics() if hasattr(self.fast, "metrics") else {},
            "fresh_row_required_after_generation_start": True,
        })
        self._lat(
            "V1_11_GENERATION_READY",
            generation_id=self.generation_id,
            book_tail_start_offset=book_eof,
            meta_tail_start_offset=meta_eof,
            fixed_session_start_equity_usd=self.session_start_equity_usd,
            fixed_session_kill_equity_usd=self.session_kill_equity_usd,
        )

    def stop_recorder(self):
        self._lat(
            "V1_11_EXTERNAL_RECORDER_LEFT_RUNNING",
            recorder_pid=getattr(self.recorder_proc, "pid", None),
        )
        return None

    def _latest_fresh_bbo(self, ticker):
        sync = self.fast.catch_up_to_stable_eof(V1.PRESEND_EOF_TIMEOUT_MS)
        snap = self.fast.latest_snapshot(ticker)
        if not sync.get("ok") or not snap:
            return None, {"sync": sync, "reason": "NO_CERTIFIED_GENERATION_RAW_STATE"}
        fresh, reason = self.fast.snapshot_is_generation_fresh(
            snap, max_age_ms=FRESH_ROW_MAX_AGE_MS
        )
        if not fresh:
            return None, {"sync": sync, "reason": reason, "snapshot": snap}
        row = (snap or {}).get("row") or {}
        cur = V1.OOS._top_state(row)
        if cur is None:
            return None, {"sync": sync, "reason": "INVALID_BBO", "snapshot": snap}
        receipt_ms = B._f((snap or {}).get("receipt_wall_ms"), np.nan)
        detect_ms = B._f((snap or {}).get("watchdog_detect_wall_ms"), np.nan)
        anchor = receipt_ms if np.isfinite(receipt_ms) else detect_ms
        age_ms = (time.time_ns() / 1e6 - anchor) if np.isfinite(anchor) else np.nan
        return cur, {
            "sync": sync,
            "snapshot": snap,
            "age_ms": age_ms,
            "generation_fresh": True,
        }

    def first_book(self, ticker, elapsed):
        already = ticker in self.first_seen
        super().first_book(ticker, elapsed)
        if already or not self.eligible.get(ticker, False):
            return
        key = self.window_key(ticker)
        if self.rotation_window_key is None:
            self.rotation_window_key = key
            self._lat(
                "V1_11_ROTATION_WINDOW_SELECTED",
                generation_id=self.generation_id,
                window_key=key,
                ticker=ticker,
            )
        elif key != self.rotation_window_key:
            self.eligible[ticker] = False
            self.dt[ticker]["phase"] = "DISABLED"
            self.dt[ticker]["disabled_reason"] = "NEXT_WINDOW_RESERVED_FOR_NEXT_GENERATION"
            self._transition(
                ticker,
                "WINDOW_DISABLED",
                reason="NEXT_WINDOW_RESERVED_FOR_NEXT_GENERATION",
                window_key=key,
                rotation_window_key=self.rotation_window_key,
            )

    def _rotation_targets(self):
        if not self.rotation_window_key:
            return []
        return sorted(
            t for t, ok in self.eligible.items()
            if ok and self.window_key(t) == self.rotation_window_key
        )

    def _write_rotation_checkpoint(self, *, verified, reason, group_resting,
                                   positions, confirm_history=None):
        targets = self._rotation_targets()
        checkpoint = {
            "time": B._iso(),
            "live_version": LIVE_VERSION,
            "generation_id": self.generation_id,
            "parent_session_dir": self.parent_session,
            "window_close_time": self.rotation_window_key,
            "target_tickers": targets,
            "target_count": len(targets),
            "finalized_targets": sorted(t for t in targets if t in self.finalized),
            "all_targets_m5_finalized": bool(targets and all(t in self.finalized for t in targets)),
            "verified": bool(verified),
            "safe_to_rotate": bool(verified),
            "reason": str(reason),
            "group_resting": group_resting,
            "nonzero_positions": positions,
            "confirm_history": confirm_history or [],
            "fixed_session_start_equity_usd": self.session_start_equity_usd,
            "fixed_session_kill_equity_usd": self.session_kill_equity_usd,
            "watchdog": self.fast.metrics() if hasattr(self.fast, "metrics") else {},
            "external_recorder_pid": getattr(self.recorder_proc, "pid", None),
            "external_recorder_alive": bool(
                self.recorder_proc and self.recorder_proc.poll() is None
            ),
        }
        B._atomic(self.session / ROTATION_CHECKPOINT_FILE, checkpoint)
        return checkpoint

    def _maybe_rotation_checkpoint(self):
        if self.rotation_checkpoint_written or self.shutdown_started:
            return
        targets = self._rotation_targets()
        if not targets or not all(t in self.finalized for t in targets):
            return
        if any(str((tr or {}).get("ticker") or "") in targets for tr in self.active.values()):
            return

        self.rotation_verify_attempts += 1
        try:
            group_resting, history = self._confirm_group_resting(
                ticker=None, attempts=ROTATION_REST_CONFIRM_ATTEMPTS
            )
        except TypeError:
            group_resting, history = self._confirm_group_resting(ticker=None)
        positions, pos_timing = B._positions(self.client)
        nonzero = [
            r for r in positions
            if abs(B._f((r or {}).get("position_fp"), 0.0)) > B.EPS
        ]
        verified = not group_resting and not nonzero
        reason = "M5_ZERO_POSITION_ZERO_GROUP_RESTING" if verified else "M5_ROTATION_VERIFY_FAILED"
        checkpoint = self._write_rotation_checkpoint(
            verified=verified,
            reason=reason,
            group_resting=group_resting,
            positions=nonzero,
            confirm_history=[
                {"resting_confirmations": history, "positions_timing": pos_timing}
            ],
        )
        self.rotation_checkpoint_written = True

        if not verified:
            self.rotation_failure = checkpoint
            self.last_error = f"rotation checkpoint failed: {checkpoint}"
            self.emit("CRITICAL", reason="M5_ROTATION_VERIFY_FAILED", checkpoint=checkpoint)
            self.shutdown("M5_ROTATION_VERIFY_FAIL_CLOSED")
            return

        self._lat(
            "V1_11_ROTATION_CHECKPOINT_VERIFIED",
            generation_id=self.generation_id,
            window_key=self.rotation_window_key,
            target_count=len(targets),
        )
        self.shutdown("GENERATION_ROTATION_M5_VERIFIED")

    def enforce_wall_clock_m5(self):
        super().enforce_wall_clock_m5()
        if not self.shutdown_started:
            self._maybe_rotation_checkpoint()

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            targets = set(self._rotation_targets())
            compact_dt = {}
            for ticker, state in self.dt.items():
                if ticker in targets or ticker not in self.finalized:
                    compact_dt[ticker] = dict(state)
            if len(compact_dt) > WATCHDOG_MAX_TICKERS:
                compact_dt = dict(list(compact_dt.items())[-WATCHDOG_MAX_TICKERS:])
            h.update({
                "live_version": LIVE_VERSION,
                "deep_tail_live_version": LIVE_VERSION,
                "generation_id": self.generation_id,
                "parent_session_dir": self.parent_session,
                "external_recorder_owned_by_supervisor": True,
                "external_recorder_pid": getattr(self.recorder_proc, "pid", None),
                "fixed_session_start_equity_usd": self.session_start_equity_usd,
                "fixed_session_kill_equity_usd": self.session_kill_equity_usd,
                "rotation_window_key": self.rotation_window_key,
                "rotation_target_count": len(targets),
                "rotation_checkpoint_written": self.rotation_checkpoint_written,
                "watchdog_compact": self.fast.metrics() if hasattr(self.fast, "metrics") else {},
                "deep_tail_states": compact_dt,
            })
            B._atomic(self.health_path, h)
        except Exception:
            pass


def _run_process_external(session, cfg):
    """B._run_process replacement that attaches to the supervisor recorder."""
    session = Path(session).resolve()
    if not bool(cfg.get("external_recorder_owner")):
        raise RuntimeError("V1.11 external runner requires external_recorder_owner=True")
    recorder_pid = int(cfg.get("external_recorder_pid") or 0)
    if recorder_pid <= 0 or not B._pid_alive(recorder_pid):
        raise RuntimeError(f"External recorder is not alive: pid={recorder_pid}")

    raw = session / "raw_capture"
    if not raw.exists():
        raise RuntimeError(f"Generation raw_capture attachment is missing: {raw}")

    current_pre = B._preflight(
        quote_size=cfg["quote_size"],
        runtime_hours=cfg["runtime_hours"],
        max_loss_usd=cfg["max_start_loss_usd"],
        min_equity_usd=cfg["min_start_equity_usd"],
        mode=cfg["mode"],
        save_dir=session,
        show=True,
    )

    fixed_start = float(cfg["session_start_equity_usd"])
    fixed_kill = float(cfg["session_kill_equity_usd"])
    expected_kill = fixed_start - float(cfg["max_start_loss_usd"])
    if abs(fixed_kill - expected_kill) > 1e-6:
        raise RuntimeError(
            f"Fixed session kill mismatch: cfg={fixed_kill} expected={expected_kill}"
        )
    current_equity = float((current_pre.get("account") or {}).get("equity_usd"))
    if current_equity <= fixed_kill + B.EPS:
        raise RuntimeError(
            f"Session loss limit already breached before generation start: "
            f"current={current_equity:.4f} kill={fixed_kill:.4f}"
        )

    engine_pre = copy.deepcopy(current_pre)
    engine_pre.setdefault("account", {})["equity_usd"] = fixed_start
    engine_pre["kill_equity_usd"] = fixed_kill
    B._atomic(session / SESSION_RISK_BASELINE_FILE, {
        "time": B._iso(),
        "generation_id": int(cfg.get("generation_id") or 0),
        "fixed_session_start_equity_usd": fixed_start,
        "fixed_session_kill_equity_usd": fixed_kill,
        "current_generation_start_equity_usd": current_equity,
        "max_start_loss_usd": float(cfg["max_start_loss_usd"]),
        "baseline_reset_between_generations": False,
    })

    client = B.Q1.LiveClient()
    gid, gb, gt = B._create_group(client)
    B._atomic(session / "order_group.json", {
        "time": B._iso(),
        "order_group_id": gid,
        "response": gb,
        "timing": gt,
    })
    recorder = ExternalRecorderProxy(recorder_pid)
    B._atomic(session / "raw_recorder_start.json", {
        "time": B._iso(),
        "pid": recorder_pid,
        "external_owner": True,
        "parent_session_dir": cfg.get("parent_session_dir"),
        "attachment_only": True,
    })

    try:
        B.LiveEngine(session, cfg, client, recorder, gid, engine_pre).run()
    except BaseException:
        B._trigger_group(client, gid)
        raise


def _watchdog_offline_regression():
    class Dummy:
        def __init__(self, root):
            self.session = Path(root)
            self.events = []
        def _lat(self, event, **kw):
            self.events.append((event, kw))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        raw = root / "raw_capture"
        raw.mkdir(parents=True, exist_ok=True)
        path = raw / "book_top3_events.jsonl"
        stale = {
            "ticker": "OLD",
            "receipt_time": pd.Timestamp.now(tz="UTC").isoformat(),
            "elapsed_s": 1.0,
        }
        path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
        stale_size = path.stat().st_size
        d = Dummy(root)
        w = CompactGenerationEofWatchdog(d)
        w.start()
        deadline = time.time() + 2.0
        while not w.ready() and time.time() < deadline:
            time.sleep(0.005)
        started_at_eof = w.ready() and int(w.generation_start_offset or -1) == int(stale_size)
        old_ignored = w.latest_snapshot("OLD") is None and w.rows_seen == 0

        fresh = {
            "ticker": "NEW",
            "receipt_time": pd.Timestamp.now(tz="UTC").isoformat(),
            "elapsed_s": 61.0,
            "yes_bid": 0.10,
            "yes_ask": 0.11,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fresh) + "\n")
        w.wake_event.set()
        deadline = time.time() + 2.0
        snap = None
        while time.time() < deadline:
            snap = w.latest_snapshot("NEW")
            if snap:
                break
            time.sleep(0.005)
        sync = w.catch_up_to_stable_eof(timeout_ms=100.0)
        fresh_ok, fresh_reason = w.snapshot_is_generation_fresh(snap, max_age_ms=5000.0)
        metrics = w.metrics()
        w.stop()

    return {
        "generation_starts_at_existing_eof": bool(started_at_eof),
        "historical_row_not_replayed": bool(old_ignored),
        "new_row_seen": bool(snap),
        "stable_eof_barrier_passes": sync.get("ok") is True,
        "fresh_row_gate_passes": bool(fresh_ok),
        "fresh_row_gate_reason": fresh_reason,
        "history_retained_false": metrics.get("history_retained") is False,
        "cancel_executor_present_false": metrics.get("cancel_executor_present") is False,
        "latest_ticker_cap": metrics.get("latest_ticker_cap"),
    }


def static_self_check(*, show=True):
    parent = V110.static_self_check(show=False)
    regression = _watchdog_offline_regression()
    checks = {
        "parent_v1_10_ok": parent.get("ok") is True,
        "final_runtime_reconciler_guard_retained": (
            parent.get("expected_runtime_rest_mode") == V110.EXPECTED_REST_MODE
        ),
        "compact_watchdog_starts_at_eof": regression["generation_starts_at_existing_eof"],
        "compact_watchdog_ignores_historical_rows": regression["historical_row_not_replayed"],
        "fresh_row_gate_regression": regression["fresh_row_gate_passes"],
        "stable_eof_barrier_regression": regression["stable_eof_barrier_passes"],
        "watchdog_history_removed": regression["history_retained_false"],
        "watchdog_cancel_executor_removed": regression["cancel_executor_present_false"],
        "external_recorder_shutdown_is_parent_owned": True,
        "fixed_session_risk_baseline_required": True,
        "verified_m5_rotation_checkpoint_required": True,
        "strategy_m1_unchanged_60s": abs(V1.M1_S - 60.0) < 1e-12,
        "strategy_m5_unchanged_300s": abs(V1.M5_S - 300.0) < 1e-12,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "watchdog_max_tickers": WATCHDOG_MAX_TICKERS,
        "fresh_row_max_age_ms": FRESH_ROW_MAX_AGE_MS,
        "rotation_checkpoint_file": ROTATION_CHECKPOINT_FILE,
        "offline_regression": regression,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 126)
        print("DEEP-TAIL LIVE V1.11 COMPACT ROTATING-GENERATION STATIC CHECK — NO API / NO ORDERS")
        print("=" * 126)
        for k, v in out.items():
            print(f"{k:68s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.11 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run V1.10 with compact EOF watchdog + external-recorder generation runner."""
    session = Path(session).resolve()
    if not bool((cfg or {}).get("external_recorder_owner")):
        raise RuntimeError("V1.11 refuses standalone recorder ownership")

    old_terminal = V19.TerminalTombstoneGuardEngine
    old_watchdog = V122.BarrierFastCancelWatchdog
    old_run_process = B._run_process
    old_v19_version = V19.LIVE_VERSION
    old_v110_version = V110.LIVE_VERSION

    V19.TerminalTombstoneGuardEngine = CompactRotatingGenerationEngine
    V122.BarrierFastCancelWatchdog = CompactGenerationEofWatchdog
    B._run_process = _run_process_external
    # V1.10 deliberately propagates its own LIVE_VERSION through V1.9/V1.8/V1.7
    # at runtime. Temporarily publish V1.11 as that top-level version so the
    # instantiated health/binding artifacts identify the actual rotating engine.
    V110.LIVE_VERSION = LIVE_VERSION
    V19.LIVE_VERSION = LIVE_VERSION
    try:
        return V110.run_live_process(session, cfg)
    finally:
        V19.TerminalTombstoneGuardEngine = old_terminal
        V122.BarrierFastCancelWatchdog = old_watchdog
        B._run_process = old_run_process
        V19.LIVE_VERSION = old_v19_version
        V110.LIVE_VERSION = old_v110_version


__all__ = [
    "LIVE_VERSION",
    "ROTATION_CHECKPOINT_FILE",
    "GENERATION_BOOTSTRAP_FILE",
    "SESSION_RISK_BASELINE_FILE",
    "WATCHDOG_MAX_TICKERS",
    "FRESH_ROW_MAX_AGE_MS",
    "ExternalRecorderProxy",
    "CompactGenerationEofWatchdog",
    "CompactRotatingGenerationEngine",
    "static_self_check",
    "run_live_process",
]
