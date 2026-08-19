from __future__ import annotations

"""V1.4 persistent M5 cleanup after the first Q5 live stop.

Observed Q5 issue
-----------------
The V1.3 Q5 run completed a real Q5 entry/passive-exit round trip, but on the
following window a fresh resting-order read at M5 still showed an entry order after
its normal cancel path. V1.2 treated any such M5 uncertainty as an immediate reason
to trigger the *entire* order group and stop the run.

V1.4 changes only M5 cleanup plumbing:
- crossing M5 puts the ticker into M5_CLEANUP_PENDING instead of immediately
  finalizing or killing the whole run;
- every engine cycle (and therefore every subsequent public-book snapshot as well)
  retries cleanup until BOTH authoritative strategy resting orders for that ticker
  are zero and the authoritative position is flat;
- local M5 cancel failures are retryable for that ticker rather than global-fatal;
- exchange-visible M5 resting rows are re-cancelled individually with the hardened
  V11 V2 cancel path; the order group is NOT triggered merely because one M5 read
  still shows a resting row;
- if a position remains after resting orders are gone, the normal reduce-only IOC
  flatten path is retried on later cycles until flat;
- the account auditor treats exchange-visible orders belonging to a ticker already
  in M5_CLEANUP_PENDING as cleanup work, not as an orphan requiring a global kill;
- a passive JOIN_ASK is never newly posted once wall time has reached M5, even if an
  entry fill arrives during cancellation.

Global hard stops remain unchanged for loss limit, dual-tail fills, position-limit
violations, non-M5 transport/cancel failures, manual kill, and engine exceptions.

No alpha rule, 5c entry, quantity, M1/M5 boundary, fixed JOIN_ASK price, no-reprice
rule, loss threshold, raw recorder, or fill accounting is changed.
"""

import time
from queue import Empty

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_1 as V11
from . import mm_deep_tail_join_ask_live_v1_3 as V13


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_4_PERSISTENT_M5_CLEANUP"

# Cleanup is deliberately throttled: enforce_wall_clock_m5() already runs from the
# starvation-proof main loop, and on_book() also calls finalize_m5() after M5.
M5_RETRY_INTERVAL_S = 0.25
M5_RETRY_LOG_EVERY = 8


class PersistentM5CleanupEngine(V13.PassiveGTCCompatibleEngine):
    """V1.3 strategy with persistent, per-ticker M5 cleanup instead of run abort."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m5_cleanup = {}

    # ------------------------------------------------------------------
    # M5 state helpers
    # ------------------------------------------------------------------

    def _ensure_m5_cleanup(self, ticker):
        ticker = str(ticker)
        st = self.m5_cleanup.get(ticker)
        if st is None:
            st = {
                "started_wall": time.time(),
                "attempts": 0,
                "next_retry_wall": 0.0,
                "last_reason": None,
                "last_resting_count": None,
                "last_position": None,
            }
            self.m5_cleanup[ticker] = st
            self.dt[ticker]["phase"] = "M5_CLEANUP_PENDING"
            self._transition(ticker, "M5_CLEANUP_PENDING", reason="M5_BOUNDARY")
            self.emit("M5_CLEANUP_PENDING", ticker, reason="M5_BOUNDARY")
        return st

    def _schedule_m5_retry(self, ticker, reason, *, resting_count=None, position=None):
        st = self._ensure_m5_cleanup(ticker)
        st["last_reason"] = str(reason)
        st["last_resting_count"] = resting_count
        st["last_position"] = position
        st["next_retry_wall"] = time.time() + M5_RETRY_INTERVAL_S
        if st["attempts"] == 1 or st["attempts"] % M5_RETRY_LOG_EVERY == 0:
            self._lat(
                "M5_CLEANUP_RETRY_SCHEDULED",
                ticker=ticker,
                attempts=st["attempts"],
                reason=reason,
                resting_count=resting_count,
                position=position,
            )
        return st

    # ------------------------------------------------------------------
    # Never create a new passive exit once M5 has arrived.
    # A fill racing with cancellation is cleaned through the M5 IOC path instead.
    # ------------------------------------------------------------------

    def _maybe_post_exit(self, ticker):
        e = self.wall_elapsed(ticker)
        if e == e and e >= V1.M5_S:
            self._ensure_m5_cleanup(ticker)
            self.dt[ticker]["phase"] = "M5_CLEANUP_PENDING"
            self._lat(
                "PASSIVE_EXIT_SUPPRESSED_AFTER_M5",
                ticker=ticker,
                wall_elapsed_s=e,
            )
            return
        return super()._maybe_post_exit(ticker)

    # ------------------------------------------------------------------
    # M5 cancel futures are retryable. Non-M5 cancellation retains the old
    # fail-closed behavior by falling through to the parent implementation.
    # ------------------------------------------------------------------

    def _drain_cancel_futures(self):
        for key, rec in list(self.pending_cancels.items()):
            fut = rec.get("future")
            reason = str(rec.get("reason") or "")
            if not reason.startswith("M5") or fut is None or not fut.done():
                continue

            self.pending_cancels.pop(key, None)
            tr = self.active.get(key)
            ticker = str((tr or {}).get("ticker") or "")

            try:
                result = fut.result()
            except Exception as exc:
                result = {"ok": False, "exception": repr(exc)}

            self._lat(
                "CANCEL_RESULT",
                key=key,
                ticker=ticker or None,
                reason=reason,
                request_to_result_ms=(
                    V1._wall_ms() - float(rec.get("requested_ms", V1._wall_ms()))
                ),
                result=result,
            )

            if not result.get("ok"):
                # Do not kill the whole run for an M5-only cancel uncertainty.
                # Make the local track eligible for another targeted cancel attempt.
                if tr is not None:
                    tr["cancel_requested"] = False
                    tr["cancel_reason"] = None
                    tr["status"] = "resting"
                if ticker:
                    st = self._ensure_m5_cleanup(ticker)
                    st["next_retry_wall"] = 0.0
                B._append(self.risk_log, {
                    "time": B._iso(),
                    "event": "M5_CANCEL_RETRY_PENDING",
                    "ticker": ticker,
                    "key": key,
                    "result": result,
                })
                continue

            if tr is not None:
                self._apply_floor(
                    key,
                    B._f(result.get("fill_floor"), 0.0),
                    "cancel_receipt",
                    result,
                )
                tr = self.active.get(key)
                if tr is not None:
                    retired = self._retire_local_track(key)
                    if retired is not None:
                        retired["status"] = "canceled"
                        self.counters_dt["cancels"] += 1
                        ticker = str(retired["ticker"])
                        if (
                            retired["role"] == "ENTRY"
                            and self.dt[ticker].get("chosen_tail") == retired["tail"]
                            and retired.get("processed_fill", 0.0) < self.q - V1.EPS
                        ):
                            self.dt[ticker]["phase"] = "M5_CLEANUP_PENDING"

            if ticker:
                st = self._ensure_m5_cleanup(ticker)
                st["next_retry_wall"] = 0.0

        # Any completed non-M5 cancels keep the original V1.3/V1 fail-closed path.
        return super()._drain_cancel_futures()

    # ------------------------------------------------------------------
    # Account auditor: an exchange-visible order for a ticker already being cleaned
    # at M5 is not a mysterious orphan. Wake that ticker's cleanup state machine and
    # leave all other orphan handling unchanged/fail-closed.
    # ------------------------------------------------------------------

    def _drain_audit(self):
        for _ in range(20):
            try:
                x = self.audit_q.get_nowait()
            except Empty:
                break

            B._append(self.audit_log, {"time": B._iso(), **x})
            if x.get("kind") != "ACCOUNT_AUDIT":
                continue

            snapshot_resting = x.get("resting") or []
            group_snapshot = [
                r for r in snapshot_resting
                if str(r.get("order_group_id") or "") == str(self.gid)
            ]
            known = {
                str(tr.get("order_id") or "")
                for tr in self.active.values()
                if tr.get("order_id")
            }
            candidate_orphans = [
                r for r in group_snapshot
                if str(r.get("order_id") or "") not in known
            ]

            if candidate_orphans:
                m5_candidates = [
                    r for r in candidate_orphans
                    if str(r.get("ticker") or "") in self.m5_cleanup
                    and str(r.get("ticker") or "") not in self.finalized
                ]
                for r in m5_candidates:
                    ticker = str(r.get("ticker") or "")
                    st = self._ensure_m5_cleanup(ticker)
                    st["next_retry_wall"] = 0.0

                if m5_candidates:
                    self._lat(
                        "AUDIT_M5_RESTING_HANDOFF_TO_RETRY",
                        snapshot_recv_ms=x.get("recv_ms"),
                        count=len(m5_candidates),
                        order_ids=[str(r.get("order_id") or "") for r in m5_candidates],
                    )

                m5_ids = {str(r.get("order_id") or "") for r in m5_candidates}
                candidate_orphans = [
                    r for r in candidate_orphans
                    if str(r.get("order_id") or "") not in m5_ids
                ]

            if candidate_orphans:
                confirmed, confirms = self._confirm_group_resting(ticker=None)
                candidate_ids = {
                    str(r.get("order_id") or "") for r in candidate_orphans
                }
                confirmed_orphans = [
                    r for r in confirmed
                    if str(r.get("order_id") or "") in candidate_ids
                    and str(r.get("order_id") or "") not in known
                ]
                self._lat(
                    "AUDIT_ORPHAN_CONFIRMATION",
                    snapshot_recv_ms=x.get("recv_ms"),
                    candidate_count=len(candidate_orphans),
                    confirmed_count=len(confirmed_orphans),
                    confirmations=confirms,
                )

                if confirmed_orphans:
                    trig = B._trigger_group(self.client, self.gid)
                    self.last_error = f"confirmed orphan strategy resting orders: {confirmed_orphans}"
                    B._append(self.risk_log, {
                        "time": B._iso(),
                        "event": "CONFIRMED_ORPHAN_GROUP_TRIGGER",
                        "orders": confirmed_orphans,
                        "group_trigger": trig,
                    })
                    self.emit(
                        "CRITICAL",
                        reason="CONFIRMED_ORPHAN_RESTING_ORDER",
                        orders=confirmed_orphans,
                        group_trigger=trig,
                    )
                    self.shutdown("CONFIRMED_ORPHAN_RESTING_ORDER")
                    return

                self._lat(
                    "AUDIT_STALE_ORPHAN_SNAPSHOT_DISMISSED",
                    snapshot_recv_ms=x.get("recv_ms"),
                    candidate_order_ids=sorted(candidate_ids),
                )

            # Preserve V1.2 position-limit and gross-position hard stops.
            posmap = {}
            for r in x.get("positions") or []:
                t = str(r.get("ticker") or "")
                p = B._f(r.get("position_fp"), 0.0)
                if t and abs(p) > V1.EPS:
                    posmap[t] = p
                    self.positions[t] = p
                    if abs(p) > self.q + 0.02:
                        self.last_error = f"position exceeds Q on {t}: {p}"
                        self.emit("CRITICAL", t, reason="POSITION_LIMIT", position=p, q=self.q)
                        self.shutdown("POSITION_LIMIT")
                        return

            gross = sum(abs(v) for v in posmap.values())
            gross_cap = len(B.SERIES) * self.q + 0.10
            if gross > gross_cap:
                self.last_error = f"gross position {gross} > cap {gross_cap}"
                self.shutdown("GROSS_POSITION_LIMIT")
                return

    # ------------------------------------------------------------------
    # Persistent M5 state machine.
    # ------------------------------------------------------------------

    def finalize_m5(self, ticker):
        ticker = str(ticker)
        if not ticker or ticker in self.finalized or self.shutdown_started:
            return

        st = self._ensure_m5_cleanup(ticker)
        now = time.time()
        if now < float(st.get("next_retry_wall", 0.0)):
            return

        st["attempts"] += 1
        st["next_retry_wall"] = now + M5_RETRY_INTERVAL_S
        attempt = int(st["attempts"])

        # 1) Retire any locally tracked entry/exit orders. Creates still in flight
        # receive cancel_requested and are cancelled as soon as their order_id arrives.
        self._cancel_all_for_ticker(ticker, f"M5_RETRY_{attempt}")
        self._drain_create_futures()
        self._drain_cancel_futures()
        self._drain_private()

        local_left = [
            {"key": k, "track": tr}
            for k, tr in self.active.items()
            if str(tr.get("ticker") or "") == ticker
        ]
        if local_left:
            self._schedule_m5_retry(
                ticker,
                "LOCAL_TRACKS_PENDING",
                resting_count=len(local_left),
            )
            return

        # 2) Local bookkeeping is empty. Trust but verify against the current
        # authoritative resting-order set. If rows remain, individually re-cancel
        # them; do NOT trigger the whole group for this transient M5 condition.
        try:
            resting, timing = self._fresh_group_resting_once(ticker=ticker)
        except Exception as exc:
            self._lat(
                "M5_RESTING_READ_RETRY",
                ticker=ticker,
                attempt=attempt,
                error=repr(exc),
            )
            self._schedule_m5_retry(ticker, "RESTING_READ_ERROR")
            return

        if resting:
            results = []
            for row in resting:
                oid = str(row.get("order_id") or "")
                if not oid:
                    continue
                submitted_qty = B._f(row.get("initial_count_fp"), self.q)
                if not (submitted_qty > 0):
                    submitted_qty = self.q
                try:
                    result = V11.safe_cancel_v2_resting_set(
                        self.client,
                        order_id=oid,
                        submitted_qty=float(submitted_qty),
                    )
                except Exception as exc:
                    result = {"ok": False, "exception": repr(exc)}
                results.append({"order_id": oid, "result": result})

            B._append(self.risk_log, {
                "time": B._iso(),
                "event": "M5_AUTHORITATIVE_RESTING_RECANCEL",
                "ticker": ticker,
                "attempt": attempt,
                "orders": [str(r.get("order_id") or "") for r in resting],
                "resting_read_timing": timing,
                "results": results,
            })
            self._schedule_m5_retry(
                ticker,
                "AUTHORITATIVE_RESTING_RECANCELLED",
                resting_count=len(resting),
            )
            return

        # 3) No strategy resting order remains for this ticker. Flatten any actual
        # residual inventory with the existing reduce-only IOC path. If touch/depth
        # is temporarily unavailable, retry on a later cycle instead of killing run.
        try:
            p = self.refresh_position(ticker)
        except Exception as exc:
            self._lat("M5_POSITION_READ_RETRY", ticker=ticker, attempt=attempt, error=repr(exc))
            self._schedule_m5_retry(ticker, "POSITION_READ_ERROR")
            return

        if abs(p) > V1.EPS:
            try:
                self.flatten(ticker, "M5_RETRY")
            except Exception as exc:
                B._append(self.risk_log, {
                    "time": B._iso(),
                    "event": "M5_FLATTEN_RETRY_PENDING",
                    "ticker": ticker,
                    "attempt": attempt,
                    "position": p,
                    "error": repr(exc),
                })
                self._schedule_m5_retry(ticker, "FLATTEN_RETRY", position=p)
                return

        # 4) Final proof: both conditions must be simultaneously true before the
        # ticker is allowed to become M5_FINALIZED.
        try:
            p2 = self.refresh_position(ticker)
            resting2, confirms2 = self._confirm_group_resting(ticker=ticker)
        except Exception as exc:
            self._lat("M5_FINAL_VERIFY_RETRY", ticker=ticker, attempt=attempt, error=repr(exc))
            self._schedule_m5_retry(ticker, "FINAL_VERIFY_READ_ERROR")
            return

        if abs(p2) > V1.EPS or resting2:
            self._lat(
                "M5_FINAL_VERIFY_NOT_CLEAN_YET",
                ticker=ticker,
                attempt=attempt,
                position=p2,
                resting_order_ids=[str(r.get("order_id") or "") for r in resting2],
                confirmations=confirms2,
            )
            self._schedule_m5_retry(
                ticker,
                "FINAL_VERIFY_NOT_CLEAN_YET",
                resting_count=len(resting2),
                position=p2,
            )
            return

        self.finalized.add(ticker)
        self.m5_cleanup.pop(ticker, None)
        self.dt[ticker]["phase"] = "M5_FINALIZED"
        self._transition(ticker, "M5_FINALIZED", position=p2, cleanup_attempts=attempt)
        self.emit("M5_FINALIZED", ticker, position=p2, cleanup_attempts=attempt)

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h["m5_cleanup_pending"] = {
                t: dict(st) for t, st in self.m5_cleanup.items()
                if t not in self.finalized
            }
            h["m5_cleanup_retry_interval_s"] = M5_RETRY_INTERVAL_S
            B._atomic(self.health_path, h)
        except Exception:
            pass


def _install_patch():
    # Install every prior safety/execution fix, then replace only the engine class.
    V13._install_patch()
    V1.DeepTailLiveEngine = PersistentM5CleanupEngine


def run_live_process(session, cfg):
    _install_patch()
    old = V1.LIVE_VERSION
    try:
        V1.LIVE_VERSION = LIVE_VERSION
        return V1.run_live_process(session, cfg)
    finally:
        V1.LIVE_VERSION = old


def static_self_check(*, show=True):
    _install_patch()
    parent = V13.static_self_check(show=False)
    out = dict(parent)
    out.update({
        "version": LIVE_VERSION,
        "m5_persistent_cleanup": True,
        "m5_immediate_group_trigger_on_resting": False,
        "m5_targeted_authoritative_recancel": True,
        "m5_retry_on_wall_clock_loop": True,
        "m5_retry_on_post_m5_book_snapshot": True,
        "m5_retry_interval_s": M5_RETRY_INTERVAL_S,
        "m5_requires_exchange_zero_resting": True,
        "m5_requires_flat_position": True,
        "m5_passive_exit_suppressed_after_boundary": True,
        "m5_auditor_handoff_instead_of_orphan_kill": True,
        "non_m5_hard_stops_preserved": True,
        "ok": bool(parent.get("ok")),
        "orders_sent": False,
    })
    if show:
        print("=" * 100)
        print("DEEP-TAIL LIVE V1.4 STATIC SELF-CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:62s}: {v}")
    return out


__all__ = [
    "LIVE_VERSION",
    "M5_RETRY_INTERVAL_S",
    "PersistentM5CleanupEngine",
    "run_live_process",
    "static_self_check",
]
