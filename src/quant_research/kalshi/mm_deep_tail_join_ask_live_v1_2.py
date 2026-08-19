from __future__ import annotations

"""V1.2 operational hardening after the first real Q1 deep-tail smoke.

Observed Q1 failure
-------------------
The first real Q1 smoke reached M5 flat but then the asynchronous account auditor
reported resting orders that were no longer present in the engine's local active map.
That may be a stale pre-cancel audit snapshot, but an orphan candidate must never be
ignored. The failure path then exposed a definite shutdown bug: V1 called
``B.LiveEngine.shutdown`` after runtime installation had replaced ``B.LiveEngine``
with the deep-tail class itself, causing infinite recursion instead of cleanup.

V1.2 changes execution safety plumbing only:
- pin the original base LiveEngine class through the deep-tail class MRO and use it
  explicitly for flatten/shutdown, eliminating monkey-patch recursion;
- treat asynchronous auditor orphan rows as candidates, then synchronously confirm
  them against fresh authoritative resting-order reads before failing closed;
- at M5, do not declare a ticker finalized until all local tracks are retired AND a
  fresh exchange resting-order check confirms zero strategy-group orders for it;
- if M5 cancellation cannot be confirmed, trigger the entire order group immediately,
  re-check, and shut the strategy down rather than continuing;
- verify flat position again before logging M5_FINALIZED.

No alpha rule, entry price, quantity, M1 timing, JOIN_ASK price, no-reprice rule,
loss threshold, raw recorder, or fill accounting is changed.
"""

import time
from pathlib import Path
from queue import Empty

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_1 as V11

LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_2_M5_VERIFIED_NONRECURSIVE_SHUTDOWN"

# Capture the immutable base class object from the class MRO. This remains valid even
# after V1._install_runtime() later assigns B.LiveEngine = VerifiedM5DeepTailEngine.
BASE_LIVE_ENGINE = V1.DeepTailLiveEngine.__mro__[1]

REST_CONFIRM_ATTEMPTS = 3
REST_CONFIRM_SLEEP_S = 0.08
M5_LOCAL_CANCEL_TIMEOUT_S = 5.0
GROUP_TRIGGER_SETTLE_S = 0.20


class VerifiedM5DeepTailEngine(V11.WallClockM1DeepTailEngine):
    """V1.1 strategy + verified M5 cleanup + nonrecursive fail-closed shutdown."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m5_finalizing = set()

    # ------------------------------------------------------------------
    # Fresh exchange verification helpers
    # ------------------------------------------------------------------

    def _fresh_group_resting_once(self, ticker=None):
        rows, timing = B._resting(self.client)
        group = [
            r for r in (rows or [])
            if str(r.get("order_group_id") or "") == str(self.gid)
        ]
        if ticker is not None:
            group = [r for r in group if str(r.get("ticker") or "") == str(ticker)]
        return group, timing

    def _confirm_group_resting(self, ticker=None, attempts=REST_CONFIRM_ATTEMPTS):
        """Return resting rows only if they persist across fresh REST confirmations."""
        attempts = max(1, int(attempts))
        history = []
        last = []
        for i in range(attempts):
            rows, timing = self._fresh_group_resting_once(ticker)
            last = rows
            history.append({
                "attempt": i + 1,
                "count": len(rows),
                "order_ids": [str(r.get("order_id") or "") for r in rows],
                "timing": timing,
            })
            if not rows:
                return [], history
            if i + 1 < attempts:
                time.sleep(REST_CONFIRM_SLEEP_S)
        return last, history

    # ------------------------------------------------------------------
    # Base operations pinned to the original class object
    # ------------------------------------------------------------------

    def flatten(self, ticker, reason):
        # First retire any strategy tracks for this ticker using deep-tail V2 cancel.
        for key, tr in list(self.active.items()):
            if str(tr.get("ticker")) == str(ticker):
                self.cancel_track(key, reason + "_CANCEL")
        # Never dereference B.LiveEngine here: runtime monkey-patching changes it.
        return BASE_LIVE_ENGINE.flatten(self, ticker, reason)

    def shutdown(self, reason):
        if self.shutdown_started:
            return
        try:
            # Base shutdown triggers the order group before cancel/flatten cleanup.
            BASE_LIVE_ENGINE.shutdown(self, reason)
        finally:
            for obj in (self.private, self.rest_fills, self.risk, self.auditor):
                try:
                    obj.stop()
                except Exception:
                    pass
            try:
                self.fast.stop()
            except Exception:
                pass
            try:
                self.transport.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Account auditor: stale snapshots are candidates, not conclusions
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
                # The auditor is asynchronous. Its snapshot may have been taken before
                # an M5 cancel and consumed after the local track was retired. Confirm
                # against fresh reads before declaring a real orphan.
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

            # Position checks remain fail-closed and use the auditor's actual account
            # position snapshot. These do not depend on the local order map.
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
    # M5: verified local + exchange cleanup before finalization
    # ------------------------------------------------------------------

    def finalize_m5(self, ticker):
        if ticker in self.finalized or ticker in self.m5_finalizing:
            return
        self.m5_finalizing.add(ticker)
        try:
            self._cancel_all_for_ticker(ticker, "M5")

            deadline = time.time() + M5_LOCAL_CANCEL_TIMEOUT_S
            while (
                any(str(tr.get("ticker")) == str(ticker) for tr in self.active.values())
                and time.time() < deadline
                and not self.shutdown_started
            ):
                self.poll_orders()
                self._drain_private()
                time.sleep(0.005)

            if self.shutdown_started:
                return

            local_left = [
                {"key": k, "track": tr}
                for k, tr in self.active.items()
                if str(tr.get("ticker")) == str(ticker)
            ]
            if local_left:
                trig = B._trigger_group(self.client, self.gid)
                self.last_error = f"M5 local cancel timeout {ticker}: {local_left}"
                B._append(self.risk_log, {
                    "time": B._iso(), "event": "M5_LOCAL_CANCEL_TIMEOUT_GROUP_TRIGGER",
                    "ticker": ticker, "tracks": local_left, "group_trigger": trig,
                })
                self.shutdown("M5_LOCAL_CANCEL_TIMEOUT")
                return

            resting, confirms = self._confirm_group_resting(ticker=ticker)
            if resting:
                # Matching-engine group trigger is the broadest emergency cancellation
                # authority and also prevents any in-flight group create from becoming
                # a new resting order after this point.
                trig = B._trigger_group(self.client, self.gid)
                B._append(self.risk_log, {
                    "time": B._iso(), "event": "M5_RESTING_GUARD_GROUP_TRIGGER",
                    "ticker": ticker, "orders": resting,
                    "confirmations": confirms, "group_trigger": trig,
                })
                time.sleep(GROUP_TRIGGER_SETTLE_S)
                after, after_confirms = self._confirm_group_resting(ticker=None)
                self.last_error = (
                    f"M5 exchange resting-order guard fired on {ticker}; "
                    f"after_group_trigger_resting={after}"
                )
                self._lat(
                    "M5_GROUP_TRIGGER_VERIFICATION",
                    ticker=ticker,
                    before=resting,
                    after=after,
                    confirmations=after_confirms,
                    group_trigger=trig,
                )
                # Even if trigger cleared the orders, the group is now triggered and
                # no longer usable for subsequent windows. End the run cleanly.
                self.shutdown("M5_RESTING_GUARD_GROUP_TRIGGER")
                return

            p = self.refresh_position(ticker)
            if abs(p) > V1.EPS:
                self.flatten(ticker, "M5")
                if self.shutdown_started:
                    return

            p2 = self.refresh_position(ticker)
            resting2, confirms2 = self._confirm_group_resting(ticker=ticker)
            if abs(p2) > V1.EPS or resting2:
                trig = B._trigger_group(self.client, self.gid)
                self.last_error = (
                    f"M5 final verification failed {ticker}: position={p2}, resting={resting2}"
                )
                B._append(self.risk_log, {
                    "time": B._iso(), "event": "M5_FINAL_VERIFY_FAIL_GROUP_TRIGGER",
                    "ticker": ticker, "position": p2, "resting": resting2,
                    "confirmations": confirms2, "group_trigger": trig,
                })
                self.shutdown("M5_FINAL_VERIFY_FAIL")
                return

            self.finalized.add(ticker)
            self.dt[ticker]["phase"] = "M5_FINALIZED"
            self._transition(ticker, "M5_FINALIZED", position=p2)
            self.emit("M5_FINALIZED", ticker, position=p2)
        finally:
            self.m5_finalizing.discard(ticker)


def _install_patch():
    # Retain all V1.1 fixes first, then replace only the engine class.
    V11._install_patch()
    V1.DeepTailLiveEngine = VerifiedM5DeepTailEngine


def run_live_process(session, cfg):
    _install_patch()
    old = V1.LIVE_VERSION
    try:
        V1.LIVE_VERSION = LIVE_VERSION
        return V1.run_live_process(Path(session), cfg)
    finally:
        V1.LIVE_VERSION = old


def static_self_check(*, show=True):
    _install_patch()
    parent = V11.static_self_check(show=False)
    out = dict(parent)
    out.update({
        "version": LIVE_VERSION,
        "base_live_engine_pinned_from_mro": BASE_LIVE_ENGINE is V1.DeepTailLiveEngine.__mro__[-2],
        "nonrecursive_shutdown": True,
        "nonrecursive_flatten": True,
        "audit_orphan_requires_fresh_confirmation": True,
        "m5_requires_exchange_zero_resting": True,
        "m5_requires_flat_position": True,
        "m5_group_trigger_on_unconfirmed_cancel": True,
        "ok": bool(parent.get("ok")),
        "orders_sent": False,
    })
    if show:
        print("=" * 100)
        print("DEEP-TAIL LIVE V1.2 STATIC SELF-CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    return out


__all__ = [
    "LIVE_VERSION",
    "BASE_LIVE_ENGINE",
    "VerifiedM5DeepTailEngine",
    "run_live_process",
    "static_self_check",
]
