from __future__ import annotations

"""V2.9.9.7 Q100 semantic CREATE rejection classification.

Operational layer on top of V2.9.9.6 / V1.12.5.

Observed failure addressed
--------------------------
On 2026-08-25 an ENTRY create for one HYPE market received a definitive Kalshi
HTTP 404 response with error code ``market_not_found``.  V1 treated every
non-post-only CREATE exception as an ambiguous transport failure and immediately
shut down the full 12-hour session with ``CREATE_TRANSPORT_FAIL_CLOSED``.

A definitive ``market_not_found`` response is different from an ambiguous timeout,
network failure, 5xx response, or unknown POST outcome: Kalshi explicitly rejected
the requested market and no order was created by that failed request.

This layer makes one deliberately narrow change:

- ENTRY create + definitive ``market_not_found`` + no already-known exposure in
  that ticker -> retire the rejected entry track, disable that ticker, request
  cancellation of the peer ENTRY leg if it exists, and continue the session.
- The peer leg may still be in flight.  Existing deferred-cancel behavior is
  preserved, so if it later ACKs with an order id the cancel is dispatched.
- If exposure is already known in that ticker, the rejection is NOT treated as a
  local skip; inherited global fail-closed behavior remains.
- EXIT ``market_not_found`` is NOT relaxed because inventory may already exist.
- Every other CREATE exception, including timeout/network/5xx/ambiguous POST state,
  still takes the inherited ``CREATE_TRANSPORT_FAIL_CLOSED`` path.

V2.9.9.6 bounded retry-until-flat recovery is preserved exactly.  Q100 sizing is
preserved exactly: quantity 100, 12-hour fixed session clock, $20 software loss
stop, $125 minimum starting equity, M1 entry, M12 cleanup, danger guard, REC25=25%,
atomic trigger snapshot, fixed passive exit/no repricing, parent M12+45s hard
recycle, guardian 90s backstop, and reduce-only IOC terminal cleanup.

Importing this module performs no API calls and sends no orders.
"""

import inspect

from . import mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_6_retry_flatten as BASE


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q100_M1_M12_GUARD_REC25_V2_9_9_7_SEMANTIC_CREATE"
MODULE_NAME = "quant_research.kalshi.mm_deep_tail_join_ask_q100_m12_guard_rec25_live_v2_9_9_7_semantic_create"
Q100_ARM = "LIVE_DEEP_TAIL_Q100_M1_M12_GUARD_REC25_12H_V2997"
# Historical runtime API still uses the Q50_ARM attribute name internally.
Q50_ARM = Q100_ARM
KILL_ARM = BASE.KILL_ARM

RUNTIME = BASE.RUNTIME
P = BASE.P
H = BASE.H
V2963 = BASE.V2963
V28 = BASE.V28
V288 = BASE.V288
V111 = BASE.V111
LIVE = BASE.LIVE
V1 = LIVE.V1
B = V1.B

Q100_Q = BASE.Q100_Q
Q100_HOURS = BASE.Q100_HOURS
Q100_MAX_LOSS_USD = BASE.Q100_MAX_LOSS_USD
Q100_MIN_EQUITY_USD = BASE.Q100_MIN_EQUITY_USD

# Compatibility aliases used by inherited runtime functions and notebook tooling.
Q50_Q = Q100_Q
Q50_HOURS = Q100_HOURS
Q50_MAX_LOSS_USD = Q100_MAX_LOSS_USD
Q50_MIN_EQUITY_USD = Q100_MIN_EQUITY_USD

M1_S = BASE.M1_S
M12_S = BASE.M12_S
LABEL_TAIL_END_S = BASE.LABEL_TAIL_END_S

M12_HARD_RECYCLE_GRACE_S = BASE.M12_HARD_RECYCLE_GRACE_S
HARD_RECYCLE_RECEIPT_FILE = BASE.HARD_RECYCLE_RECEIPT_FILE
GUARDIAN_POST_M12_EXIT_TIMEOUT_S = BASE.GUARDIAN_POST_M12_EXIT_TIMEOUT_S

GENERATION_RSS_WARNING_MB = BASE.GENERATION_RSS_WARNING_MB
GENERATION_RSS_HARD_LIMIT_MB = BASE.GENERATION_RSS_HARD_LIMIT_MB
RSS_HARD_STOP_DISABLED = BASE.RSS_HARD_STOP_DISABLED

RECOVERY_FRACTION = BASE.RECOVERY_FRACTION
PRE_LOOKBACK_S = BASE.PRE_LOOKBACK_S
PRE_EXCLUDE_S = BASE.PRE_EXCLUDE_S
PRE_FALLBACK_S = BASE.PRE_FALLBACK_S

RECOVERY_RETRY_WINDOW_S = BASE.RECOVERY_RETRY_WINDOW_S
RECOVERY_RETRY_PAUSE_S = BASE.RECOVERY_RETRY_PAUSE_S

MARKET_NOT_FOUND_CODE = "market_not_found"
MARKET_NOT_FOUND_MESSAGE = "market not found"
LOCAL_SKIP_REASON = "ENTRY_CREATE_MARKET_NOT_FOUND"

# Preserve the original live-engine drain method across notebook reloads.
if not hasattr(V1.DeepTailLiveEngine, "_v2997_original_drain_create_futures"):
    V1.DeepTailLiveEngine._v2997_original_drain_create_futures = (
        V1.DeepTailLiveEngine._drain_create_futures
    )

_ORIGINAL_DRAIN_CREATE_FUTURES = (
    V1.DeepTailLiveEngine._v2997_original_drain_create_futures
)


def _definitive_market_not_found(exc):
    """True only for the exact observed definitive Kalshi market-not-found rejection."""
    text = repr(exc).lower()
    return MARKET_NOT_FOUND_CODE in text and MARKET_NOT_FOUND_MESSAGE in text


def _known_ticker_exposure(engine, ticker):
    """Fail conservative: local-skip only while the ticker has no known exposure."""
    ticker = str(ticker)
    try:
        if abs(float(engine.positions.get(ticker, 0.0))) > V1.EPS:
            return True
    except Exception:
        return True

    st = engine.dt.get(ticker) or {}
    if st.get("chosen_tail") is not None:
        return True

    for row in engine.active.values():
        if str(row.get("ticker") or "") != ticker:
            continue
        try:
            if float(row.get("processed_fill", 0.0) or 0.0) > V1.EPS:
                return True
        except Exception:
            return True
    return False


def _retire_market_not_found_entry(engine, key, tr, exc):
    """Retire one definitively rejected ENTRY and cancel/disable its peer ticker."""
    ticker = str(tr.get("ticker") or "")
    cid = str(tr.get("cid") or "")

    engine.active.pop(key, None)
    if cid:
        engine.cid_to_key.pop(cid, None)
    tr["status"] = "rejected_market_not_found"

    st = engine.dt.get(ticker)
    if st is not None:
        st["phase"] = "DISABLED"
        st["disabled_reason"] = LOCAL_SKIP_REASON

    engine.counters_dt["entry_create_market_not_found_local_skip"] += 1
    B._append(
        engine.orders,
        {
            "time": B._iso(),
            "action": "CREATE_REJECTED_LOCAL_SKIP",
            "reason": LOCAL_SKIP_REASON,
            "key": key,
            "track": tr,
            "error": repr(exc),
        },
    )
    engine._lat(
        "ENTRY_CREATE_MARKET_NOT_FOUND_LOCAL_SKIP",
        ticker=ticker,
        key=key,
        role=tr.get("role"),
        tail=tr.get("tail"),
        error=repr(exc),
        session_continues=True,
    )
    engine._transition(
        ticker,
        "WINDOW_DISABLED",
        reason=LOCAL_SKIP_REASON,
        failed_key=key,
        failed_tail=tr.get("tail"),
        error=repr(exc),
        session_continues=True,
    )

    # Disable the whole ticker.  The peer CREATE may still be in flight; inherited
    # deferred-cancel semantics safely wait for its order id if it later ACKs.
    for other_key, other in list(engine.active.items()):
        if (
            str(other.get("ticker") or "") == ticker
            and str(other.get("role") or "").upper() == "ENTRY"
        ):
            engine._request_cancel_key(other_key, "PEER_MARKET_NOT_FOUND")


def _drain_create_futures_semantic_create(self):
    """V1 drain with one narrow local-skip class for definitive ENTRY market_not_found."""
    for key, fut in list(self.pending_creates.items()):
        if not fut.done():
            continue
        self.pending_creates.pop(key, None)
        tr = self.active.get(key)
        try:
            body, timing = fut.result()
        except Exception as exc:
            # Existing passive EXIT post-only rejection remains local/no-reprice.
            if (
                tr
                and tr["role"] == "EXIT"
                and any(
                    x in repr(exc).lower()
                    for x in ("post_only", "post-only", "would cross")
                )
            ):
                self.active.pop(key, None)
                st = self.dt[tr["ticker"]]
                st["phase"] = "HOLD_TO_M5_EXIT_REJECTED"
                self._transition(
                    tr["ticker"],
                    "EXIT_NOT_POSTED",
                    reason="POST_ONLY_REJECT",
                    error=repr(exc),
                )
                continue

            market_not_found = _definitive_market_not_found(exc)

            # A definitively rejected create cannot have created an order.  If the
            # local track was already retired while the request was in flight, this
            # specific rejection is therefore safe to consume without shutdown.
            if tr is None and market_not_found:
                ticker = str(key).split("|", 1)[0]
                self._lat(
                    "CREATE_MARKET_NOT_FOUND_AFTER_TRACK_RETIRED",
                    ticker=ticker,
                    key=key,
                    error=repr(exc),
                    session_continues=True,
                )
                continue

            # Narrow relaxation: ENTRY only, exact observed semantic rejection only,
            # and only while no exposure is already known in that ticker.
            if (
                tr
                and tr["role"] == "ENTRY"
                and market_not_found
                and not _known_ticker_exposure(self, tr["ticker"])
            ):
                _retire_market_not_found_entry(self, key, tr, exc)
                continue

            # EVERYTHING ELSE stays inherited fail-closed, including:
            # - timeout / network / connection error
            # - 5xx / ambiguous POST outcome
            # - semantic rejection on an EXIT with possible inventory
            # - market_not_found if exposure is already known
            self.last_error = f"create failure {key}: {exc!r}"
            self.emit(
                "CRITICAL",
                tr["ticker"] if tr else None,
                reason="CREATE_FAIL_CLOSED",
                key=key,
                error=repr(exc),
            )
            self.shutdown("CREATE_TRANSPORT_FAIL_CLOSED")
            return

        if tr is None:
            # A shutdown/local retirement may have removed it while CREATE was in
            # flight.  Preserve inherited behavior: record a returned order id if
            # one exists.  Existing account auditing remains the final orphan guard.
            oid = str((body or {}).get("order_id") or "")
            if oid:
                ghost = {
                    "key": key,
                    "ticker": str((body or {}).get("ticker") or ""),
                    "qty": self.q,
                    "order_id": oid,
                }
                self._lat(
                    "CREATE_COMPLETED_AFTER_TRACK_RETIRED",
                    key=key,
                    order_id=oid,
                    ghost=ghost,
                )
            continue

        oid = str((body or {}).get("order_id") or "")
        if not oid:
            self.last_error = f"create response missing order id {key}: {body}"
            self.shutdown("CREATE_RESPONSE_MISSING_ID")
            return

        tr["order_id"] = oid
        tr["status"] = "resting"
        tr["create_response_wall_ms"] = V1._wall_ms()
        self.order_id_to_key[oid] = key
        B._append(
            self.orders,
            {
                "time": B._iso(),
                "action": "CREATE_ACK",
                "track": tr,
                "response": body,
                "timing": timing,
            },
        )
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
        if (
            tr.get("cancel_requested")
            and key in self.active
            and key not in self.pending_cancels
        ):
            self._request_cancel_key(
                key,
                tr.get("cancel_reason") or "DEFERRED_CANCEL",
            )

    self._retry_unmatched_private()


def _install_patch():
    """Install V2.9.9.6, then add only semantic ENTRY market-not-found handling."""
    BASE._install_patch()

    # Install the narrow child-engine create classifier.
    V1.DeepTailLiveEngine._drain_create_futures = _drain_create_futures_semantic_create

    # Q100 launch identity/parameters on the inherited detached runtime.
    RUNTIME.DEPLOY_VERSION = DEPLOY_VERSION
    RUNTIME.MODULE_NAME = MODULE_NAME
    RUNTIME.Q50_ARM = Q100_ARM
    RUNTIME.Q50_Q = Q100_Q
    RUNTIME.Q50_HOURS = Q100_HOURS
    RUNTIME.Q50_MAX_LOSS_USD = Q100_MAX_LOSS_USD
    RUNTIME.Q50_MIN_EQUITY_USD = Q100_MIN_EQUITY_USD
    RUNTIME.LIVE = LIVE

    # Preserve V2.9.9.6 recovery and Q100 parent settings; only identity changes.
    P.DEPLOY_VERSION = DEPLOY_VERSION
    P.LIVE = LIVE
    P.Q50_Q = Q100_Q
    P.Q50_HOURS = Q100_HOURS
    P.Q50_MAX_LOSS_USD = Q100_MAX_LOSS_USD
    P.Q50_MIN_EQUITY_USD = Q100_MIN_EQUITY_USD
    P._recover_generation_fail_closed = BASE._recover_generation_fail_closed_retry

    V2963.DEPLOY_VERSION = DEPLOY_VERSION
    V2963.LIVE = LIVE
    V2963.POST_M5_EXIT_TIMEOUT_S = GUARDIAN_POST_M12_EXIT_TIMEOUT_S
    V2963._post_m5_generation_state = RUNTIME._post_m12_generation_state

    # Preserve this exact wrapper through inherited dynamic/subprocess calls.
    RUNTIME._install_patch = _install_patch
    RUNTIME.static_self_check = static_self_check


def static_self_check(*, show=True):
    """Offline structural/regression audit. No API calls and no orders."""
    base = BASE.static_self_check(show=False)
    _install_patch()
    src = inspect.getsource(_drain_create_futures_semantic_create)

    sample_market_not_found = RuntimeError(
        "Kalshi POST /portfolio/events/orders -> 404: "
        "{'error': {'code': 'market_not_found', 'message': 'market not found'}}"
    )
    sample_timeout = RuntimeError("POST timeout; outcome unknown")

    checks = {
        "base_v2996_ok": base.get("ok") is True,
        "q100_exact_100": Q100_Q == 100.0,
        "runtime_q100_exact": RUNTIME.Q50_Q == 100.0,
        "parent_q100_exact": P.Q50_Q == 100.0,
        "runtime_exact_12h": Q100_HOURS == 12.0,
        "loss_stop_stays_20": Q100_MAX_LOSS_USD == 20.0,
        "minimum_equity_125": Q100_MIN_EQUITY_USD == 125.0,
        "entry_m1_60": M1_S == 60.0,
        "terminal_m12_720": M12_S == 720.0,
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "m12_hard_recycle_45s": M12_HARD_RECYCLE_GRACE_S == 45.0,
        "guardian_90s": GUARDIAN_POST_M12_EXIT_TIMEOUT_S == 90.0,
        "retry_window_45s_preserved": RECOVERY_RETRY_WINDOW_S == 45.0,
        "retry_pause_200ms_preserved": RECOVERY_RETRY_PAUSE_S == 0.20,
        "retry_recovery_preserved": P._recover_generation_fail_closed is BASE._recover_generation_fail_closed_retry,
        "market_not_found_exact_detected": _definitive_market_not_found(sample_market_not_found) is True,
        "ambiguous_timeout_not_local": _definitive_market_not_found(sample_timeout) is False,
        "base_engine_patch_installed": V1.DeepTailLiveEngine._drain_create_futures is _drain_create_futures_semantic_create,
        "rec25_engine_inherits_patch": LIVE.Rec25PassiveExitM12Engine._drain_create_futures is _drain_create_futures_semantic_create,
        "local_relaxation_entry_only": 'tr["role"] == "ENTRY"' in src,
        "market_not_found_local_skip_present": "_retire_market_not_found_entry" in src,
        "other_create_fail_closed_preserved": 'self.shutdown("CREATE_TRANSPORT_FAIL_CLOSED")' in src,
        "missing_id_fail_closed_preserved": 'self.shutdown("CREATE_RESPONSE_MISSING_ID")' in src,
        "passive_exit_reduce_only_false": LIVE.PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": LIVE.PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": LIVE.PASSIVE_EXIT_TIF == "good_till_canceled",
        "risk_m12_flatten_reduce_only_ioc_unchanged": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_version": LIVE.LIVE_VERSION,
        "module_name": MODULE_NAME,
        "quantity": Q100_Q,
        "runtime_hours": Q100_HOURS,
        "max_loss_usd": Q100_MAX_LOSS_USD,
        "semantic_create_policy": {
            "local_skip": "ENTRY_MARKET_NOT_FOUND_ONLY_WHEN_NO_KNOWN_EXPOSURE",
            "peer_action": "CANCEL_OR_DEFER_CANCEL_PEER_ENTRY",
            "exit_market_not_found": "FAIL_CLOSED_UNCHANGED",
            "timeout_network_5xx_ambiguous": "FAIL_CLOSED_UNCHANGED",
        },
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 176)
        print("V2.9.9.7 Q100 SEMANTIC-CREATE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 176)
        for k, v in out.items():
            print(f"{k:112s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.9.9.7 static self-check failed: {out}")
    return out


def q100_preflight(*, show=True):
    """Read-only exact-dollar Q100 preflight. Sends no orders."""
    _install_patch()
    static_self_check(show=show)
    V28._patch_parent()
    V28.D._guard_other_live_processes()

    old_equity = LIVE.V1.B._equity
    LIVE.V1.B._equity = LIVE.exact_equity_from_balance
    try:
        return V288.live_preflight(
            quote_size=Q100_Q,
            runtime_hours=Q100_HOURS,
            max_start_loss_usd=Q100_MAX_LOSS_USD,
            min_start_equity_usd=Q100_MIN_EQUITY_USD,
            show=show,
            probe_private_ws=True,
        )
    finally:
        LIVE.V1.B._equity = old_equity


def start_q100_12h_smoke(*, arm_phrase=None):
    """REAL-MONEY Q100 / 12h with local ENTRY market-not-found skip."""
    _install_patch()
    return RUNTIME.start_q50_12h_smoke(arm_phrase=arm_phrase)


def live_status(*, show=True, tail_lines=40):
    _install_patch()
    return RUNTIME.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=30.0):
    _install_patch()
    return RUNTIME.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    _install_patch()
    return RUNTIME._main()


if __name__ == "__main__":
    _main()


__all__ = [
    "DEPLOY_VERSION",
    "MODULE_NAME",
    "Q100_ARM",
    "Q50_ARM",
    "KILL_ARM",
    "Q100_Q",
    "Q100_HOURS",
    "Q100_MAX_LOSS_USD",
    "Q100_MIN_EQUITY_USD",
    "Q50_Q",
    "Q50_HOURS",
    "Q50_MAX_LOSS_USD",
    "Q50_MIN_EQUITY_USD",
    "M1_S",
    "M12_S",
    "LABEL_TAIL_END_S",
    "M12_HARD_RECYCLE_GRACE_S",
    "HARD_RECYCLE_RECEIPT_FILE",
    "GUARDIAN_POST_M12_EXIT_TIMEOUT_S",
    "GENERATION_RSS_WARNING_MB",
    "GENERATION_RSS_HARD_LIMIT_MB",
    "RSS_HARD_STOP_DISABLED",
    "RECOVERY_FRACTION",
    "PRE_LOOKBACK_S",
    "PRE_EXCLUDE_S",
    "PRE_FALLBACK_S",
    "RECOVERY_RETRY_WINDOW_S",
    "MARKET_NOT_FOUND_CODE",
    "LOCAL_SKIP_REASON",
    "static_self_check",
    "q100_preflight",
    "start_q100_12h_smoke",
    "live_status",
    "kill_and_flatten_live",
]
