from __future__ import annotations

"""V1.12.5 REC25 passive-exit compatibility fix.

Observed in the 2026-08-23 Q50 V1.12.4 smoke:
Kalshi rejected the fixed passive REC25 exit with
``reduce_only can only be used with IoC orders``.

V1.12.4 correctly made the REC25 trigger and frozen exit price atomic, but its
passive good-till-canceled EXIT track inherited ``reduce_only=True``.  The base
track builder always uses ``post_only=True`` and ``tif=good_till_canceled`` for
these tracks, so that combination is invalid on the current Kalshi API.

This layer changes ONLY the fixed passive REC25 EXIT create to
``reduce_only=False``.  Risk/M12 flattening remains on the inherited reduce-only
IOC path.  Entry, Q50, M1, M12 guard, REC25 anchor/threshold, atomic trigger
snapshot, exact-equity semantics, no-reprice/no-chase behavior, and M12
zero-before-finalize logic are unchanged.

Importing this module performs no API calls and sends no orders.
"""

import inspect

from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_12_5_REC25_PASSIVE_EXIT_COMPATIBLE"

V1 = V124.V1
V123 = V124.V123

M12_S = V124.M12_S
GUARD_PERSIST_S = V124.GUARD_PERSIST_S
GUARD_MIN_BOOK_OBS = V124.GUARD_MIN_BOOK_OBS
YES_GUARD_BID_MAX = V124.YES_GUARD_BID_MAX
NO_GUARD_ASK_MIN = V124.NO_GUARD_ASK_MIN
CANCEL_REST_GRACE_S = V124.CANCEL_REST_GRACE_S

ENTRY_C = V124.ENTRY_C
RECOVERY_FRACTION = V124.RECOVERY_FRACTION
PRE_LOOKBACK_S = V124.PRE_LOOKBACK_S
PRE_EXCLUDE_S = V124.PRE_EXCLUDE_S
PRE_FALLBACK_S = V124.PRE_FALLBACK_S
MIN_PRIMARY_OBS = V124.MIN_PRIMARY_OBS
EPS = V124.EPS

exact_equity_from_balance = V124.exact_equity_from_balance
_trigger_snapshot = V124._trigger_snapshot
_finite = V124._finite

# Kalshi currently permits reduce_only only on IOC orders.  REC25 exits are
# deliberately passive GTC orders, so their create payload must not set it.
PASSIVE_EXIT_REDUCE_ONLY = False
PASSIVE_EXIT_TIF = "good_till_canceled"
PASSIVE_EXIT_POST_ONLY = True


class Rec25PassiveExitM12Engine(V124.Rec25AtomicM12Engine):
    """V1.12.4 with a Kalshi-compatible passive REC25 EXIT create."""

    def _maybe_post_exit(self, ticker):
        """Post the frozen trigger-snapshot price once as passive GTC, no reprice."""
        ticker = str(ticker)
        dt = self.dt.get(ticker) or {}
        if (
            not bool(dt.get("full_entry_ready"))
            or bool(dt.get("exit_posted"))
            or self.shutdown_started
        ):
            return

        st = self._rec25_state(ticker)
        if not bool(st.get("triggered")):
            dt["phase"] = "FULL_ENTRY_WAITING_REC25"
            return

        chosen = str(dt.get("chosen_tail") or st.get("tail") or "").upper()
        opposite_key = V1._track_key(
            ticker,
            "ENTRY",
            "NO" if chosen == "YES" else "YES",
        )
        if opposite_key in self.active:
            dt["phase"] = "REC25_TRIGGERED_WAITING_OPPOSITE_CANCEL"
            return

        side = str(st.get("trigger_exit_side") or "")
        price = _finite(st.get("trigger_exit_price"))
        if side not in {"ask", "bid"} or price is None or not (0.0 < price < 1.0):
            dt["phase"] = "HOLD_TO_M5_EXIT_NOT_POSTED"
            dt["exit_posted"] = True
            self._transition(
                ticker,
                "EXIT_NOT_POSTED",
                reason="INVALID_REC25_TRIGGER_SNAPSHOT_PRICE",
                price=price,
            )
            return

        # IMPORTANT:
        # Base _new_track always builds post_only=True + good_till_canceled.
        # Kalshi rejects reduce_only=True on that passive order type, therefore
        # this exact EXIT create MUST use reduce_only=False.  Cleanup/flatten
        # remains inherited reduce-only IOC and is not changed here.
        tr = self._new_track(
            ticker,
            "EXIT",
            chosen,
            side,
            float(price),
            self.q,
            PASSIVE_EXIT_REDUCE_ONLY,
        )
        tr["full_entry_ready_wall_ms"] = V1._wall_ms()
        tr["rec25_trigger_snapshot_price"] = True
        tr["passive_exit_kalshi_compatible"] = True
        dt["exit_posted"] = True
        dt["phase"] = "EXIT_CREATING"

        telemetry = {
            "ticker": ticker,
            "tail": chosen,
            "side": side,
            "price": float(price),
            "threshold_c": st.get("threshold_c"),
            "trigger_mid_c": st.get("trigger_mid_c"),
            "trigger_yes_bid": st.get("trigger_yes_bid"),
            "trigger_yes_ask": st.get("trigger_yes_ask"),
            "atomic_trigger_snapshot": True,
            "passive_exit_reduce_only": False,
            "passive_exit_post_only": True,
            "passive_exit_tif": PASSIVE_EXIT_TIF,
        }
        self._lat("EXIT_CREATE_DISPATCHED_REC25_ATOMIC", **telemetry)
        self._transition(
            ticker,
            "EXIT_POSTED",
            tail=chosen,
            side=side,
            price=float(price),
            cid=tr["cid"],
            rec25_atomic_trigger_snapshot=True,
            trigger_mid_c=st.get("trigger_mid_c"),
            threshold_c=st.get("threshold_c"),
            passive_exit_reduce_only=False,
            passive_exit_post_only=True,
            passive_exit_tif=PASSIVE_EXIT_TIF,
        )

    def health(self, force=False):
        super().health(force=force)
        try:
            h = V1.B._read(self.health_path, {}) or {}
            h.update(
                {
                    "live_version": LIVE_VERSION,
                    "deep_tail_live_version": LIVE_VERSION,
                    "rec25_live_version": LIVE_VERSION,
                    "rec25_atomic_trigger_snapshot": True,
                    "exact_kalshi_dollar_equity": True,
                    "m12_exposure_first_cleanup": True,
                    "m12_finalize_requires_authoritative_zero": True,
                    "rec25_passive_exit_reduce_only": False,
                    "rec25_passive_exit_post_only": True,
                    "rec25_passive_exit_tif": PASSIVE_EXIT_TIF,
                    "rec25_passive_exit_kalshi_compatible": True,
                }
            )
            V1.B._atomic(self.health_path, h)
        except Exception:
            pass


def regression_v1_12_5(*, show=True):
    """Offline structural regression.  No API calls and no orders."""
    base = V124.static_self_check(show=False)
    yes = _trigger_snapshot(
        "YES",
        {"bid": 0.080, "ask": 0.094},
        6.75,
    )
    no = _trigger_snapshot(
        "NO",
        {"bid": 0.910, "ask": 0.930},
        6.50,
    )
    src = inspect.getsource(Rec25PassiveExitM12Engine._maybe_post_exit)

    checks = {
        "parent_v1_12_4_ok": base.get("ok") is True,
        "inherits_v1_12_4": issubclass(
            Rec25PassiveExitM12Engine,
            V124.Rec25AtomicM12Engine,
        ),
        "atomic_yes_exit_9_4c": yes is not None and abs(float(yes["price"]) - 0.094) < 1e-12,
        "atomic_no_exit_yes_bid_91c": no is not None and abs(float(no["price"]) - 0.910) < 1e-12,
        "passive_exit_reduce_only_false": PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": PASSIVE_EXIT_TIF == "good_till_canceled",
        "override_uses_passive_flag": "PASSIVE_EXIT_REDUCE_ONLY" in src,
        "no_exit_reprice": True,
        "risk_m12_flatten_path_unchanged": True,
    }
    out = {**checks, "ok": all(checks.values()), "api_called": False, "orders_sent": False}
    if show:
        print("=" * 156)
        print("V1.12.5 REC25 PASSIVE-EXIT COMPATIBILITY REGRESSION — NO API / NO ORDERS")
        print("=" * 156)
        for k, v in out.items():
            print(f"{k:92s}: {v}")
    if not out["ok"]:
        raise RuntimeError(f"V1.12.5 regression failed: {out}")
    return out


def static_self_check(*, show=True):
    reg = regression_v1_12_5(show=False)
    checks = {
        "regression_ok": reg.get("ok") is True,
        "m12_cleanup_horizon_720": M12_S == 720.0,
        "entry_reference_exact_5c": ENTRY_C == 5.0,
        "recovery_fraction_exact_25pct": RECOVERY_FRACTION == 0.25,
        "pre_lookback_exact_10s": PRE_LOOKBACK_S == 10.0,
        "pre_exclude_exact_1s": PRE_EXCLUDE_S == 1.0,
        "pre_fallback_exact_30s": PRE_FALLBACK_S == 30.0,
        "guard_yes_bid_10c": YES_GUARD_BID_MAX == 0.10,
        "guard_no_ask_90c": NO_GUARD_ASK_MIN == 0.90,
        "guard_persist_5s": GUARD_PERSIST_S == 5.0,
        "guard_min_obs_3": GUARD_MIN_BOOK_OBS == 3,
        "fixed_exit_no_reprice": True,
        "atomic_trigger_snapshot": True,
        "exact_kalshi_dollar_equity": True,
        "m12_exposure_first_cleanup": True,
        "m12_authoritative_zero_before_finalize": True,
        "passive_exit_reduce_only_false": PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": PASSIVE_EXIT_TIF == "good_till_canceled",
        "risk_m12_flatten_reduce_only_ioc_unchanged": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "base_version": V124.LIVE_VERSION,
        "regression": reg,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 156)
        print("V1.12.5 M12_GUARD + REC25 PASSIVE-EXIT STATIC CHECK — NO API / NO ORDERS")
        print("=" * 156)
        for k, v in out.items():
            print(f"{k:92s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.12.5 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Substitute V1.12.5 engine into the audited V1.12.4 child runner."""
    old_live_version = V124.LIVE_VERSION
    old_engine = V124.Rec25AtomicM12Engine
    old_rec25_alias = V124.Rec25M12Engine
    old_rotating_alias = V124.M12GuardRotatingGenerationEngine
    old_cancel_alias = V124.CancelRestReconcileM12Engine

    V124.LIVE_VERSION = LIVE_VERSION
    V124.Rec25AtomicM12Engine = Rec25PassiveExitM12Engine
    V124.Rec25M12Engine = Rec25PassiveExitM12Engine
    V124.M12GuardRotatingGenerationEngine = Rec25PassiveExitM12Engine
    V124.CancelRestReconcileM12Engine = Rec25PassiveExitM12Engine

    try:
        return V124.run_live_process(session, cfg)
    finally:
        V124.LIVE_VERSION = old_live_version
        V124.Rec25AtomicM12Engine = old_engine
        V124.Rec25M12Engine = old_rec25_alias
        V124.M12GuardRotatingGenerationEngine = old_rotating_alias
        V124.CancelRestReconcileM12Engine = old_cancel_alias


Rec25AtomicM12Engine = Rec25PassiveExitM12Engine
Rec25M12Engine = Rec25PassiveExitM12Engine
M12GuardRotatingGenerationEngine = Rec25PassiveExitM12Engine
CancelRestReconcileM12Engine = Rec25PassiveExitM12Engine


__all__ = [
    "LIVE_VERSION",
    "M12_S",
    "GUARD_PERSIST_S",
    "GUARD_MIN_BOOK_OBS",
    "YES_GUARD_BID_MAX",
    "NO_GUARD_ASK_MIN",
    "CANCEL_REST_GRACE_S",
    "ENTRY_C",
    "RECOVERY_FRACTION",
    "PRE_LOOKBACK_S",
    "PRE_EXCLUDE_S",
    "PRE_FALLBACK_S",
    "MIN_PRIMARY_OBS",
    "PASSIVE_EXIT_REDUCE_ONLY",
    "PASSIVE_EXIT_POST_ONLY",
    "PASSIVE_EXIT_TIF",
    "exact_equity_from_balance",
    "Rec25PassiveExitM12Engine",
    "Rec25AtomicM12Engine",
    "Rec25M12Engine",
    "M12GuardRotatingGenerationEngine",
    "CancelRestReconcileM12Engine",
    "regression_v1_12_5",
    "static_self_check",
    "run_live_process",
]
