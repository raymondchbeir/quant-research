from __future__ import annotations

"""V1.3 live hardening after the second real Q1 smoke.

Observed Q1 result
------------------
The V1.2 Q1 smoke successfully exercised an actual 5c entry, tail selection and full
Q1 fill.  It then tried to place the fixed passive JOIN_ASK exit as a reduce-only
GTC order.  The live V2 API rejected that request with:

    reduce_only can only be used with IoC orders

That exposed two separate execution-plumbing issues:

1. A passive GTC JOIN_ASK cannot be marked reduce_only on the current V2 API.  The
   historical strategy itself is a normal passive sell/buy at the fixed outcome ask,
   so V1.3 submits that passive GTC exit with reduce_only=False.  M5/risk liquidation
   remains reduce-only IOC exactly as before.
2. The rejected CREATE remained in the local active map.  Shutdown then tried to
   cancel a synthetic track that had no exchange order_id, delaying cleanup and
   preventing the position flatten path from completing.  V1.3 retires a definitively
   failed CREATE track before fail-closed shutdown and actively cancels any order that
   completes after its local track was retired.

Safety invariant for the non-reduce-only passive exit
-----------------------------------------------------
The exit quantity is exactly the fully observed position size Q.  Before any M5
flatten, V1.2 already requires local cancellation plus fresh exchange confirmation of
zero strategy-group resting orders.  Therefore the passive exit is never intentionally
left live while a cleanup IOC is sent.

No alpha rule, entry price, quantity, M1/M5 boundary, fixed JOIN_ASK price, no-reprice
rule, loss threshold, raw recorder, or fill accounting is changed.
"""

import time

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_1 as V11
from . import mm_deep_tail_join_ask_live_v1_2 as V12

LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_3_PASSIVE_GTC_COMPAT_FAILED_CREATE_CLEANUP"


class PassiveGTCCompatibleEngine(V12.VerifiedM5DeepTailEngine):
    """V1.2 verified-M5 engine with API-compatible passive exits."""

    def _retire_local_track(self, key):
        tr = self.active.pop(key, None)
        if tr is None:
            return None
        cid = str(tr.get("cid") or "")
        oid = str(tr.get("order_id") or "")
        if cid and self.cid_to_key.get(cid) == key:
            self.cid_to_key.pop(cid, None)
        if oid and self.order_id_to_key.get(oid) == key:
            self.order_id_to_key.pop(oid, None)
        self.pending_cancels.pop(key, None)
        return tr

    def _maybe_post_exit(self, ticker):
        st = self.dt[ticker]
        if not st.get("full_entry_ready") or st.get("exit_posted") or self.shutdown_started:
            return
        chosen = st.get("chosen_tail")
        opposite_key = V1._track_key(ticker, "ENTRY", "NO" if chosen == "YES" else "YES")
        if opposite_key in self.active:
            return

        cur, cert = self._latest_fresh_bbo(ticker)
        if cur is None:
            st["phase"] = "HOLD_TO_M5_EXIT_NOT_POSTED"
            st["exit_posted"] = True
            self._transition(ticker, "EXIT_NOT_POSTED", reason=cert.get("reason"), cert=cert)
            return

        side = "ask" if chosen == "YES" else "bid"
        price = float(cur["ask"] if chosen == "YES" else cur["bid"])
        if not (0.0 < price < 1.0):
            st["phase"] = "HOLD_TO_M5_EXIT_NOT_POSTED"
            st["exit_posted"] = True
            self._transition(ticker, "EXIT_NOT_POSTED", reason="INVALID_EXIT_PRICE", price=price)
            return

        # Current Kalshi V2 rejects reduce_only on resting/GTC orders.  This is the
        # fixed passive strategy exit, not the liquidation path.  Quantity equals the
        # fully observed position Q and V1.2 verifies this order is gone before M5 IOC.
        tr = self._new_track(ticker, "EXIT", chosen, side, price, self.q, False)
        tr["full_entry_ready_wall_ms"] = V1._wall_ms()
        tr["passive_exit_reduce_only"] = False
        tr["passive_exit_safety"] = "CANCEL_AND_EXCHANGE_VERIFY_BEFORE_ANY_M5_IOC"
        st["exit_posted"] = True
        st["phase"] = "EXIT_CREATING"
        self._lat(
            "EXIT_CREATE_DISPATCHED",
            ticker=ticker,
            tail=chosen,
            price=price,
            reduce_only=False,
            time_in_force="good_till_canceled",
            cert=cert,
        )
        self._transition(
            ticker,
            "EXIT_POSTED",
            tail=chosen,
            side=side,
            price=price,
            reduce_only=False,
            cid=tr["cid"],
        )

    def _drain_create_futures(self):
        for key, fut in list(self.pending_creates.items()):
            if not fut.done():
                continue

            self.pending_creates.pop(key, None)
            tr = self.active.get(key)

            try:
                body, timing = fut.result()
            except Exception as exc:
                # A post-only exit rejection means the fixed quote is simply not
                # restable at that instant; preserve the frozen M5 fallback rule.
                if tr and tr["role"] == "EXIT" and any(
                    x in repr(exc).lower()
                    for x in ("post_only", "post-only", "would cross")
                ):
                    failed = self._retire_local_track(key)
                    st = self.dt[failed["ticker"]]
                    st["phase"] = "HOLD_TO_M5_EXIT_REJECTED"
                    self._transition(
                        failed["ticker"],
                        "EXIT_NOT_POSTED",
                        reason="POST_ONLY_REJECT",
                        error=repr(exc),
                    )
                    continue

                # V11 does not retry a definite permanent 4xx and exhausts recovery
                # for ambiguous failures before raising.  Once it raises, this local
                # synthetic track must not block fail-closed account cleanup.
                failed = self._retire_local_track(key) if tr is not None else None
                self.last_error = f"create failure {key}: {exc!r}"
                self.emit(
                    "CRITICAL",
                    (failed or tr or {}).get("ticker"),
                    reason="CREATE_FAIL_CLOSED",
                    key=key,
                    local_track_retired=True,
                    error=repr(exc),
                )
                self.shutdown("CREATE_TRANSPORT_FAIL_CLOSED")
                return

            if tr is None:
                # A shutdown/cancel may retire a local track while CREATE is in flight.
                # If the exchange accepted it, cancel it synchronously now; do not
                # merely log a ghost order.
                oid = str((body or {}).get("order_id") or "")
                if oid:
                    qty = B._f((body or {}).get("remaining_count"), self.q)
                    qty = self.q if not (qty > 0) else min(self.q, qty)
                    self._lat(
                        "CREATE_COMPLETED_AFTER_TRACK_RETIRED",
                        key=key,
                        order_id=oid,
                        action="IMMEDIATE_V2_CANCEL",
                    )
                    result = V11.safe_cancel_v2_resting_set(
                        self.client,
                        order_id=oid,
                        submitted_qty=qty,
                    )
                    self._lat(
                        "RETIRED_TRACK_GHOST_CANCEL_RESULT",
                        key=key,
                        order_id=oid,
                        result=result,
                    )
                    if not result.get("ok"):
                        trig = B._trigger_group(self.client, self.gid)
                        self.last_error = (
                            f"accepted create completed after local retirement and "
                            f"could not be canceled: key={key} oid={oid} result={result}"
                        )
                        B._append(self.risk_log, {
                            "time": B._iso(),
                            "event": "RETIRED_TRACK_GHOST_CANCEL_FAIL_GROUP_TRIGGER",
                            "key": key,
                            "order_id": oid,
                            "result": result,
                            "group_trigger": trig,
                        })
                        self.shutdown("RETIRED_TRACK_GHOST_CANCEL_FAIL")
                        return
                continue

            oid = str((body or {}).get("order_id") or "")
            if not oid:
                failed = self._retire_local_track(key)
                self.last_error = f"create response missing order id {key}: {body}"
                self.emit(
                    "CRITICAL",
                    (failed or {}).get("ticker"),
                    reason="CREATE_RESPONSE_MISSING_ID",
                    key=key,
                )
                self.shutdown("CREATE_RESPONSE_MISSING_ID")
                return

            tr["order_id"] = oid
            tr["status"] = "resting"
            tr["create_response_wall_ms"] = V1._wall_ms()
            self.order_id_to_key[oid] = key
            B._append(self.orders, {
                "time": B._iso(),
                "action": "CREATE_ACK",
                "track": tr,
                "response": body,
                "timing": timing,
            })
            self._lat(
                "CREATE_ACK",
                key=key,
                ticker=tr["ticker"],
                role=tr["role"],
                tail=tr["tail"],
                order_id=oid,
                timing=timing,
            )
            self._apply_floor(
                key,
                B._f((body or {}).get("fill_count"), 0.0),
                "create_response",
                body,
            )
            if tr.get("cancel_requested") and key in self.active and key not in self.pending_cancels:
                self._request_cancel_key(key, tr.get("cancel_reason") or "DEFERRED_CANCEL")

        self._retry_unmatched_private()


def _install_patch():
    V12._install_patch()
    V1.DeepTailLiveEngine = PassiveGTCCompatibleEngine


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
    parent = V12.static_self_check(show=False)
    out = dict(parent)
    out.update({
        "version": LIVE_VERSION,
        "passive_join_ask_reduce_only": False,
        "passive_join_ask_tif": "good_till_canceled",
        "cleanup_reduce_only": True,
        "cleanup_tif": "immediate_or_cancel",
        "failed_create_local_track_retired_before_shutdown": True,
        "retired_track_accepted_create_cancelled": True,
        "m5_cancel_confirm_before_flatten": True,
        "ok": bool(parent.get("ok")),
        "orders_sent": False,
    })
    if show:
        print("=" * 100)
        print("DEEP-TAIL LIVE V1.3 STATIC SELF-CHECK — NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:58s}: {v}")
    return out


__all__ = [
    "LIVE_VERSION",
    "PassiveGTCCompatibleEngine",
    "run_live_process",
    "static_self_check",
]
