from __future__ import annotations

"""V6 latest-state live runner for frozen Candidate-C / CYCLE_ALWAYS_EXIT.

Why V6 exists
-------------
The V5 Q1 smoke test proved that reading every persisted V5 book event and doing
synchronous REST cancel/create work on each event can make the live engine fall
minutes behind the recorder.  That is unacceptable because stale book rows can
then cause stale real orders and can starve balance/position/M5 safety checks.

V6 changes only the LIVE EXECUTION ARCHITECTURE:

* the raw V5 recorder is still preserved unchanged for audit;
* every newly persisted raw book row is ingested, but only the LATEST row per
  ticker is actionable;
* elapsed time for every real action is recomputed from the market close time and
  CURRENT WALL CLOCK, never trusted from a historical row;
* stale latest book state is never used to create a new order;
* M5 finalization is enforced from wall clock independently of book-event flow;
* balance, position, kill-request, order-state and health checks run between
  bounded batches of trading actions so a book backlog cannot starve risk logic;
* order-state polling uses the all-resting-orders endpoint, never legacy
  GET /portfolio/orders/{order_id};
* queue positions are polled in one batched request across active markets;
* V5's cancel-receipt verification, V4 recorder-start fix and V3 calibrated
  equity semantics remain in force.

Frozen strategy mechanics are unchanged: same 9-series universe, Candidate-C
entry, M0<=elapsed<M5, Q1 smoke/Q10 full sizing, opposite-BBO exit after fill,
and reduce-only IOC flatten at M5.
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v4 as V4
from . import mm_cycle_q10_live_strategy_v5 as V5

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V6"

# Operational safety guards, not research alpha parameters.
MAX_ACTION_BOOK_AGE_S = 1.50
ORDER_ABSENCE_GRACE_S = 0.75
MAX_ORDER_STATE_TRANSITIONS_PER_POLL = 2
MAX_TRADING_STATE_CHANGES_PER_PASS = 2
QUEUE_POLL_S_V6 = 1.00
LOOP_SLEEP_S_V6 = 0.025


def _receipt_epoch(row):
    try:
        t = pd.to_datetime((row or {}).get("receipt_time"), utc=True, errors="coerce")
        return np.nan if pd.isna(t) else float(t.timestamp())
    except Exception:
        return np.nan


def _wall_elapsed_from_close(close_time, now_s=None):
    """Current elapsed seconds from M0, using wall clock rather than row elapsed."""
    close = pd.to_datetime(close_time, utc=True, errors="coerce")
    if pd.isna(close):
        return np.nan
    now_s = time.time() if now_s is None else float(now_s)
    m0_s = float((close - pd.Timedelta(minutes=15)).timestamp())
    return now_s - m0_s


def backlog_regression_check(session_dir, *, bucket_ms=250, show=True):
    """NO ORDERS.  Show how V6 coalesces the failed V5 smoke's raw event rate.

    Each virtual loop bucket retains at most one actionable book state per ticker,
    instead of replaying every raw top-of-book change.
    """
    session = Path(session_dir).resolve()
    p = session / "raw_capture" / "book_top3_events.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)

    bucket_ms = max(25, int(bucket_ms))
    raw_rows = 0
    buckets = {}
    max_rows_in_bucket = 0
    first_ts = np.nan
    last_ts = np.nan

    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = _receipt_epoch(r)
            ticker = str(r.get("ticker") or "")
            if not np.isfinite(t) or not ticker:
                continue
            raw_rows += 1
            first_ts = t if not np.isfinite(first_ts) else min(first_ts, t)
            last_ts = t if not np.isfinite(last_ts) else max(last_ts, t)
            k = int((t * 1000.0) // bucket_ms)
            d = buckets.setdefault(k, {"rows": 0, "tickers": set()})
            d["rows"] += 1
            d["tickers"].add(ticker)
            max_rows_in_bucket = max(max_rows_in_bucket, d["rows"])

    actionable_latest_states = sum(len(v["tickers"]) for v in buckets.values())
    max_actionable_in_bucket = max((len(v["tickers"]) for v in buckets.values()), default=0)
    compression = (
        1.0 - actionable_latest_states / raw_rows
        if raw_rows > 0 else np.nan
    )
    duration_s = (last_ts - first_ts) if np.isfinite(first_ts) and np.isfinite(last_ts) else np.nan

    out = {
        "session": str(session),
        "bucket_ms": bucket_ms,
        "raw_book_rows": raw_rows,
        "virtual_loop_buckets": len(buckets),
        "latest_states_after_coalescing": actionable_latest_states,
        "coalesced_rows": max(0, raw_rows - actionable_latest_states),
        "compression_fraction": compression,
        "max_raw_rows_in_one_bucket": max_rows_in_bucket,
        "max_actionable_tickers_in_one_bucket": max_actionable_in_bucket,
        "max_possible_tickers": len(B.SERIES),
        "duration_s": duration_s,
        "orders_sent": False,
    }
    if show:
        print("=" * 92)
        print("V6 BACKLOG REGRESSION CHECK — NO ORDERS")
        print("=" * 92)
        print("Session:                    ", session)
        print(f"Raw book rows:               {raw_rows:,}")
        print(f"Virtual loop bucket:         {bucket_ms} ms")
        print(f"Latest actionable states:    {actionable_latest_states:,}")
        print(f"Rows discarded as backlog:   {max(0, raw_rows-actionable_latest_states):,}")
        print(f"Compression:                 {100.0*compression:.2f}%" if np.isfinite(compression) else "Compression:                 n/a")
        print(f"Max raw rows / bucket:       {max_rows_in_bucket:,}")
        print(f"Max actionable / bucket:     {max_actionable_in_bucket} (hard universe max {len(B.SERIES)})")
        print("ORDERS SENT:                 NO")
    return out


class LatestStateLiveEngine(V5.CancelReceiptSafeLiveEngine):
    """Live engine that never replays historical book backlog into real orders."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_rows = {}
        self.pending_first_rows = {}
        self.queue_initial_seen = set()
        self.rr_cursor = 0

        self.rows_ingested = 0
        self.rows_coalesced = 0
        self.max_batch_rows = 0
        self.max_action_book_age_s = 0.0
        self.max_latest_book_age_s = 0.0
        self.last_ingest_wall = time.time()
        self.last_risk_tick_wall = time.time()
        self.last_action_wall = None

        # Base initializes t_queue to now.  Use a faster batched V6 cadence.
        self.t_queue = time.time()

    # ----------------------------------------------------------------------------------
    # Market-data ingestion and wall-clock safety
    # ----------------------------------------------------------------------------------

    def wall_elapsed(self, ticker, now_s=None):
        return _wall_elapsed_from_close((self.meta.get(ticker) or {}).get("close_time"), now_s=now_s)

    def row_age_s(self, ticker, now_s=None):
        row = self.latest_rows.get(ticker) or {}
        t = _receipt_epoch(row)
        if not np.isfinite(t):
            return np.inf
        now_s = time.time() if now_s is None else float(now_s)
        return max(0.0, now_s - t)

    def update_meta(self):
        super().update_meta()
        for ticker, r in list(self.pending_first_rows.items()):
            if ticker not in self.meta or ticker in self.first_seen:
                continue
            self.first_book(ticker, B._f(r.get("elapsed_s")))
            self.pending_first_rows.pop(ticker, None)

    def ingest_book_batch(self, rows):
        if not rows:
            return
        self.rows_ingested += len(rows)
        self.max_batch_rows = max(self.max_batch_rows, len(rows))

        latest = {}
        first = {}
        for r in rows:
            ticker = str(r.get("ticker") or "")
            if not ticker:
                continue
            first.setdefault(ticker, r)
            latest[ticker] = r
            # Preserve "next book event" barrier semantics even though actions are coalesced.
            self.book_version[ticker] += 1

        self.rows_coalesced += max(0, len(rows) - len(latest))
        self.latest_rows.update(latest)
        self.last_ingest_wall = time.time()

        for ticker, r in first.items():
            if ticker in self.first_seen:
                continue
            if ticker in self.meta:
                self.first_book(ticker, B._f(r.get("elapsed_s")))
            else:
                self.pending_first_rows.setdefault(ticker, r)

    def enforce_wall_clock_m5(self):
        # Any ticker that could possibly carry exposure is checked independently of
        # incoming book events.  finalize_m5() refreshes the actual position itself.
        tickers = set(self.eligible) | set(self.active) | set(self.positions)
        for ticker in sorted(tickers):
            if ticker in self.finalized:
                continue
            e = self.wall_elapsed(ticker)
            if np.isfinite(e) and e >= 300.0:
                self.finalize_m5(ticker)
                if self.shutdown_started:
                    return

    def risk_tick(self):
        """Risk work that cannot be starved by a raw-book backlog."""
        self.last_risk_tick_wall = time.time()
        self.check_end()
        if self.shutdown_started:
            return
        self.poll_balance()
        if self.shutdown_started:
            return
        self.poll_positions()
        self.enforce_wall_clock_m5()
        if self.shutdown_started:
            return
        self.poll_orders()
        self.poll_queue()
        self.health()

    # ----------------------------------------------------------------------------------
    # Current order state without legacy GET /portfolio/orders/{id}
    # ----------------------------------------------------------------------------------

    def poll_orders(self):
        now = time.time()
        if now - self.t_order < B.ORDER_POLL_S:
            return
        self.t_order = now

        resting, _ = B._resting(self.client)
        by_id = {str(r.get("order_id") or ""): r for r in resting}
        transitions = 0

        for ticker, tr in list(self.active.items()):
            oid = str(tr.get("order_id") or "")
            row = by_id.get(oid)
            old_fill = float(tr.get("last_fill", 0.0))

            if row is not None:
                fc = B._f(row.get("fill_count_fp", row.get("fill_count")), old_fill)
                if fc > old_fill + B.EPS:
                    self.cancel_track(ticker, "RESTING_SET_FILL_DETECTED")
                    transitions += 1
            else:
                # Give the list endpoint a short propagation grace after create.
                created_wall = float(tr.get("created_wall", 0.0) or 0.0)
                if created_wall > 0 and now - created_wall < ORDER_ABSENCE_GRACE_S:
                    continue
                self.cancel_track(ticker, "ABSENT_FROM_RESTING_SET")
                transitions += 1

            if transitions >= MAX_ORDER_STATE_TRANSITIONS_PER_POLL:
                break

    def poll_queue(self):
        now = time.time()
        if now - self.t_queue < QUEUE_POLL_S_V6 or not self.active:
            return
        self.t_queue = now

        tickers = sorted({str(tr.get("ticker") or t) for t, tr in self.active.items()})
        try:
            body, timing = self.client.get(
                "/portfolio/orders/queue_positions",
                params={"market_tickers": ",".join(tickers), "subaccount": 0},
            )
            qmap = {
                str(r.get("order_id") or ""): B._f(r.get("queue_position_fp"))
                for r in body.get("queue_positions") or []
            }
            err = None
        except Exception as exc:
            qmap, timing, err = {}, {}, repr(exc)

        for ticker, tr in list(self.active.items()):
            oid = str(tr.get("order_id") or "")
            predicted = tr["book"]["bid_q1"] if tr["side"] == "bid" else tr["book"]["ask_q1"]
            initial = oid not in self.queue_initial_seen
            self.queue_initial_seen.add(oid)
            B._append(self.queue_log, {
                "time": B._iso(),
                "ticker": ticker,
                "order_id": oid,
                "role": tr["role"],
                "queue_position": qmap.get(oid, np.nan),
                "displayed_l1_ahead_at_join": predicted,
                "timing": timing,
                "error": err if oid not in qmap else None,
                "initial": initial,
                "batched_v6": True,
            })

    # ----------------------------------------------------------------------------------
    # Real order placement: latest-state + wall-clock gate immediately before POST
    # ----------------------------------------------------------------------------------

    def place(self, ticker, d, cur, elapsed):
        now = time.time()
        wall_e = self.wall_elapsed(ticker, now_s=now)
        age = self.row_age_s(ticker, now_s=now)
        self.max_action_book_age_s = max(self.max_action_book_age_s, age if np.isfinite(age) else 0.0)

        if not np.isfinite(wall_e) or not (0.0 <= wall_e < 300.0):
            self.emit("STALE_OR_OUTSIDE_WINDOW_CREATE_BLOCKED", ticker, wall_elapsed_s=wall_e, book_age_s=age)
            return
        if not np.isfinite(age) or age > MAX_ACTION_BOOK_AGE_S:
            self.emit("STALE_OR_OUTSIDE_WINDOW_CREATE_BLOCKED", ticker, wall_elapsed_s=wall_e, book_age_s=age)
            return
        if self.shutdown_started:
            return

        cid = B.CLIENT_PREFIX + uuid.uuid4().hex
        payload = B._payload(
            ticker=ticker,
            side=d["side"],
            qty=d["qty"],
            price=d["price"],
            cid=cid,
            post_only=True,
            reduce_only=False,
            tif="good_till_canceled",
            group_id=self.gid,
        )
        B._append(self.decisions, {
            "time": B._iso(),
            "ticker": ticker,
            "series": self.series(ticker),
            "close_time": self.window_key(ticker),
            "row_elapsed_s": B._f((self.latest_rows.get(ticker) or {}).get("elapsed_s")),
            "wall_elapsed_s": wall_e,
            "book_age_s": age,
            "position": self.positions.get(ticker, 0.0),
            **d,
            "book": cur,
            "engine": LIVE_VERSION,
        })

        body, timing = B._post(self.client, payload)
        oid = str(body.get("order_id") or "")
        if not oid:
            raise RuntimeError(f"Create response missing order_id: {body}")

        tr = {
            "ticker": ticker,
            "order_id": oid,
            "cid": cid,
            "role": d["role"],
            "side": d["side"],
            "price": d["price"],
            "qty": d["qty"],
            "last_fill": 0.0,
            "book": cur,
            "elapsed_s": wall_e,
            "row_elapsed_s": B._f((self.latest_rows.get(ticker) or {}).get("elapsed_s")),
            "book_age_s": age,
            "created_wall": time.time(),
        }
        self.active[ticker] = tr
        B._append(self.orders, {
            "time": B._iso(),
            "action": "CREATE_V6_LATEST_STATE",
            "ticker": ticker,
            "payload": payload,
            "response": body,
            "timing": timing,
            "wall_elapsed_s": wall_e,
            "book_age_s": age,
        })
        self.counts["orders"] += 1
        self.last_action_wall = time.time()
        self.emit("ENTRY_ORDER" if d["role"] == "ENTRY" else "EXIT_ORDER", ticker,
                  role=d["role"], side=d["side"], qty=d["qty"], price=d["price"],
                  wall_elapsed_s=wall_e, book_age_s=age)

        immediate = B._f(body.get("fill_count"), 0.0)
        if immediate > B.EPS:
            # Use V5's cancel-receipt + actual-position reconciliation path.
            self.cancel_track(ticker, "CREATE_RESPONSE_FILL")

    # ----------------------------------------------------------------------------------
    # Bounded latest-state action pass
    # ----------------------------------------------------------------------------------

    def process_latest_states(self):
        keys = sorted(self.latest_rows)
        if not keys:
            return
        start = self.rr_cursor % len(keys)
        ordered = keys[start:] + keys[:start]
        visited = 0
        changes = 0

        for ticker in ordered:
            visited += 1
            if self.shutdown_started:
                break

            self.risk_tick()
            if self.shutdown_started:
                break

            if ticker in self.finalized or not self.eligible.get(ticker, False):
                continue

            wall_e = self.wall_elapsed(ticker)
            age = self.row_age_s(ticker)
            if np.isfinite(age):
                self.max_latest_book_age_s = max(self.max_latest_book_age_s, age)

            if not np.isfinite(wall_e) or wall_e < 0.0 or wall_e >= 300.0:
                if ticker in self.active:
                    before = self.counts["cancels"]
                    self.cancel_track(ticker, "WALL_CLOCK_OUTSIDE_WINDOW")
                    changes += int(self.counts["cancels"] > before)
                continue

            if not np.isfinite(age) or age > MAX_ACTION_BOOK_AGE_S:
                if ticker in self.active:
                    before = self.counts["cancels"]
                    self.cancel_track(ticker, "STALE_LATEST_BOOK")
                    changes += int(self.counts["cancels"] > before)
                continue

            row = self.latest_rows.get(ticker) or {}
            cur = B.OOS._top_state(row)
            self.current[ticker] = cur
            if cur is None:
                if ticker in self.active:
                    before = self.counts["cancels"]
                    self.cancel_track(ticker, "INVALID_LATEST_BOOK")
                    changes += int(self.counts["cancels"] > before)
                continue

            before = self.counts["orders"] + self.counts["cancels"]
            # IMPORTANT: desired() receives CURRENT wall-clock elapsed, not row elapsed.
            self.reconcile(ticker, cur, wall_e)
            after = self.counts["orders"] + self.counts["cancels"]
            if after > before:
                changes += 1
                self.last_action_wall = time.time()

            self.risk_tick()
            if changes >= MAX_TRADING_STATE_CHANGES_PER_PASS:
                break

        self.rr_cursor = (start + max(1, visited)) % len(keys)

    # ----------------------------------------------------------------------------------
    # Health/final summary with V6 lag diagnostics
    # ----------------------------------------------------------------------------------

    def _v6_metrics(self):
        now = time.time()
        ages = [self.row_age_s(t, now_s=now) for t in self.latest_rows]
        finite = [x for x in ages if np.isfinite(x)]
        return {
            "engine": LIVE_VERSION,
            "raw_book_rows_ingested": self.rows_ingested,
            "raw_book_rows_coalesced": self.rows_coalesced,
            "max_raw_batch_rows": self.max_batch_rows,
            "latest_state_tickers": len(self.latest_rows),
            "current_max_latest_book_age_s": max(finite) if finite else np.nan,
            "max_latest_book_age_s_seen": self.max_latest_book_age_s,
            "max_action_book_age_s_seen": self.max_action_book_age_s,
            "max_allowed_action_book_age_s": MAX_ACTION_BOOK_AGE_S,
            "seconds_since_last_risk_tick": max(0.0, now - self.last_risk_tick_wall),
            "seconds_since_last_ingest": max(0.0, now - self.last_ingest_wall),
            "last_action_wall": self.last_action_wall,
        }

    def health(self, force=False):
        now = time.time()
        if not force and now - self.t_health < B.HEALTH_S:
            return
        self.t_health = now

        # Once shutdown() has written the authoritative final summary, never overwrite
        # STOPPED with the old base-class SHUTTING_DOWN state in run()'s finally block.
        if self.shutdown_started and self.final_path.exists():
            summary = B._read(self.final_path, {}) or {}
            B._atomic(self.health_path, {
                "time": B._iso(),
                "live_version": LIVE_VERSION,
                "running": False,
                "state": "STOPPED",
                "summary": summary,
                "v6_metrics": self._v6_metrics(),
            })
            return

        rec_health = B._read(self.session / "raw_capture" / "health.json", {}) or {}
        B._atomic(self.health_path, {
            "time": B._iso(),
            "live_version": LIVE_VERSION,
            "running": not self.shutdown_started,
            "state": "SHUTTING_DOWN" if self.shutdown_started else ("RUNNING" if self.trade_start else "ARMED_WAITING_FULL_WINDOW"),
            "mode": self.mode,
            "session_dir": str(self.session),
            "quote_size": self.q,
            "windows_started": len(self.windows),
            "smoke_window": self.smoke_window,
            "trade_start": self.trade_start,
            "trade_deadline": self.trade_deadline,
            "start_equity_usd": self.start_equity,
            "equity_usd": self.equity,
            "start_pnl_usd": self.equity-self.start_equity,
            "kill_equity_usd": self.kill_equity,
            "peak_equity_usd": self.peak_equity,
            "max_peak_drawdown_usd": self.max_dd,
            "positions": {t: p for t, p in self.positions.items() if abs(p) > B.EPS},
            "active_orders": self.active,
            "counts": dict(self.counts),
            "shutdown_reason": self.shutdown_reason,
            "last_error": self.last_error,
            "recorder_pid": self.recorder_proc.pid if self.recorder_proc else None,
            "recorder_alive": bool(self.recorder_proc and self.recorder_proc.poll() is None),
            "recorder_health": rec_health,
            "v6_metrics": self._v6_metrics(),
        })

    def shutdown(self, reason):
        if self.shutdown_started:
            return
        super().shutdown(reason)
        summary = B._read(self.final_path, {}) or {}
        summary["live_wrapper_version"] = LIVE_VERSION
        summary["v6_metrics"] = self._v6_metrics()
        B._atomic(self.final_path, summary)
        B._atomic(self.health_path, {
            "time": B._iso(),
            "live_version": LIVE_VERSION,
            "running": False,
            "state": "STOPPED",
            "summary": summary,
            "v6_metrics": self._v6_metrics(),
        })

    # ----------------------------------------------------------------------------------
    # Main loop: drain -> coalesce -> risk -> bounded latest-state actions
    # ----------------------------------------------------------------------------------

    def run(self):
        self.emit("ENGINE_START", mode=self.mode, engine=LIVE_VERSION)
        self.health(force=True)
        try:
            while not self.shutdown_started:
                if self.recorder_proc.poll() is not None:
                    raise RuntimeError(f"Raw V5 recorder exited rc={self.recorder_proc.returncode}")

                # Fast disk-only work first.  No REST call occurs while draining backlog.
                self.update_meta()
                rows = self.book_tail.read_new()
                self.ingest_book_batch(rows)

                # Safety always gets CPU before trading actions.
                self.risk_tick()
                if self.shutdown_started:
                    break

                # At most a small bounded number of cancel/create state changes per pass.
                self.process_latest_states()

                # Safety gets CPU again immediately after the bounded action pass.
                self.risk_tick()
                time.sleep(LOOP_SLEEP_S_V6)

        except BaseException as exc:
            self.last_error = repr(exc)
            self.emit("ERROR", error=repr(exc), traceback=__import__("traceback").format_exc())
            try:
                self.shutdown("ENGINE_EXCEPTION")
            except Exception as cleanup_exc:
                self.emit("CRITICAL", error=repr(cleanup_exc), reason="cleanup_exception")
                self.stop_recorder()
            raise
        finally:
            self.health(force=True)


# ======================================================================================
# V6 process wiring / launchers
# ======================================================================================


def _run_process_v6(session, cfg):
    session = Path(session).resolve()

    client = B.Q1.LiveClient()
    diag = V3._install(client)
    B._atomic(session / "balance_semantics.json", diag)

    B._start_recorder = V4._start_recorder_fixed
    B.LiveEngine = LatestStateLiveEngine
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
    session = (B.ROOT / f"{stamp}_{mode.lower()}_v6").resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": mode,
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": LIVE_VERSION,
        "engine_architecture": "LATEST_STATE_WALL_CLOCK_V6",
        "max_action_book_age_s": MAX_ACTION_BOOK_AGE_S,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v6",
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
            tail = log.read_text(encoding="utf-8")[-10000:] if log.exists() else ""
            raise RuntimeError(f"Live V6 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-10000:] if log.exists() else ""
        raise RuntimeError(f"Live V6 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V6 PROCESS ARMED")
    print("  mode:    ", mode)
    print("  session: ", session)
    print("  pid:     ", p.pid)
    print(f"  Q:        {q:g} per eligible market")
    print(f"  kill:     -${max_loss:.2f} from calibrated starting TOTAL account equity")
    print(f"  stale cap:{MAX_ACTION_BOOK_AGE_S:.2f}s max latest-book age at CREATE")
    print("  engine:   latest-state coalescing + wall-clock M5/risk gates")
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
        raise RuntimeError("V6 full validation is frozen to exactly 24 hours.")
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
        _run_process_v6(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "LatestStateLiveEngine",
    "backlog_regression_check",
    "live_preflight",
    "start_live_smoke_q1_one_window",
    "start_live_cycle_q10",
    "live_status",
    "kill_and_flatten_live",
]
