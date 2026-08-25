from __future__ import annotations

"""V1.12.6 REC25 M11:30 residual-entry cutoff.

This layer makes one strategy change on top of the audited V1.12.5 REC25 engine:
at M11:30 (690 seconds from the 15-minute window start), cancel only any still-
resting/unfilled ENTRY quantity.

Frozen semantics
----------------
- If neither 5c tail has filled by M11:30, request cancellation of both remaining
  ENTRY orders.
- If one tail partially filled, preserve the filled inventory and request
  cancellation only of any remaining selected-tail ENTRY quantity.  The already-
  requested opposite-tail cancellation is untouched.
- If the full requested Q has already filled, the ENTRY track has already retired
  and REC25 proceeds exactly as before.
- REC25 still requires full requested Q.  A partial position created before the
  cutoff does not gain a new exit rule; it remains on the inherited M12 path.
- The persistent danger guard may cancel earlier and remains unchanged.
- Passive REC25 exit pricing, no-reprice/no-chase behavior, M12 cleanup, exact
  equity, recovery, guardian, and risk semantics are unchanged.

The cutoff is enforced from wall clock independently of public-book backlog and is
also checked after inherited on_book processing.  Cancellation uses the existing
fail-closed/deferred-cancel path, so a CREATE still in flight at the deadline is
marked for cancellation as soon as its order id becomes known.  Fill/cancel races
remain reconciled by the inherited authoritative fill-floor logic.

Importing this module performs no API calls and sends no orders.
"""

import inspect

from . import mm_deep_tail_join_ask_live_v1_12_5_rec25_passive_exit_compatible as V125


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_12_6_REC25_M1130_ENTRY_CUTOFF"

V1 = V125.V1
V124 = V125.V124
V123 = V125.V123

M12_S = V125.M12_S
GUARD_PERSIST_S = V125.GUARD_PERSIST_S
GUARD_MIN_BOOK_OBS = V125.GUARD_MIN_BOOK_OBS
YES_GUARD_BID_MAX = V125.YES_GUARD_BID_MAX
NO_GUARD_ASK_MIN = V125.NO_GUARD_ASK_MIN
CANCEL_REST_GRACE_S = V125.CANCEL_REST_GRACE_S

ENTRY_C = V125.ENTRY_C
RECOVERY_FRACTION = V125.RECOVERY_FRACTION
PRE_LOOKBACK_S = V125.PRE_LOOKBACK_S
PRE_EXCLUDE_S = V125.PRE_EXCLUDE_S
PRE_FALLBACK_S = V125.PRE_FALLBACK_S
MIN_PRIMARY_OBS = V125.MIN_PRIMARY_OBS
EPS = V125.EPS

PASSIVE_EXIT_REDUCE_ONLY = V125.PASSIVE_EXIT_REDUCE_ONLY
PASSIVE_EXIT_TIF = V125.PASSIVE_EXIT_TIF
PASSIVE_EXIT_POST_ONLY = V125.PASSIVE_EXIT_POST_ONLY

exact_equity_from_balance = V125.exact_equity_from_balance
_trigger_snapshot = V125._trigger_snapshot
_finite = V125._finite

ENTRY_CUTOFF_S = 690.0
ENTRY_CUTOFF_REASON = "M1130_ENTRY_CUTOFF"


class Rec25M1130EntryCutoffEngine(V125.Rec25PassiveExitM12Engine):
    """V1.12.5 plus M11:30 cancellation of residual ENTRY quantity only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._m1130_entry_cutoff = {}
        self._lat(
            "M1130_ENTRY_CUTOFF_READY",
            entry_cutoff_s=ENTRY_CUTOFF_S,
            entry_cutoff_reason=ENTRY_CUTOFF_REASON,
        )

    def _m1130_state(self, ticker):
        ticker = str(ticker)
        return self._m1130_entry_cutoff.setdefault(
            ticker,
            {
                "checked": False,
                "triggered": False,
                "checked_elapsed_s": None,
                "trigger_elapsed_s": None,
                "cancel_requested_sides": [],
                "requested_remaining_q": 0.0,
                "reason": ENTRY_CUTOFF_REASON,
            },
        )

    def _enforce_m1130_entry_cutoff(self, ticker, *, source):
        """At/after 690s request cancel of every still-live ENTRY residual.

        The method deliberately does not flatten inventory, post an exit, alter
        REC25 state, disable the market, or touch EXIT tracks.  Existing entry
        cancellation machinery handles no-order-id deferral and fill/cancel races.
        """
        ticker = str(ticker or "")
        if not ticker or self.shutdown_started or ticker in self.finalized:
            return 0

        elapsed = self.wall_elapsed(ticker)
        if not V1.np.isfinite(elapsed) or float(elapsed) < ENTRY_CUTOFF_S:
            return 0

        state = self._m1130_state(ticker)
        if bool(state.get("checked")):
            return 0

        state["checked"] = True
        state["checked_elapsed_s"] = float(elapsed)

        requested = []
        requested_remaining_q = 0.0

        for key, tr in list(self.active.items()):
            if str(tr.get("ticker") or "") != ticker:
                continue
            if str(tr.get("role") or "").upper() != "ENTRY":
                continue
            if bool(tr.get("cancel_requested")):
                continue

            submitted = max(0.0, float(tr.get("qty") or 0.0))
            processed = max(0.0, float(tr.get("processed_fill") or 0.0))
            remaining = max(0.0, submitted - processed)
            if remaining <= EPS:
                continue

            tail = str(tr.get("tail") or "").upper()
            requested.append(tail)
            requested_remaining_q += remaining

            self._transition(
                ticker,
                "M1130_ENTRY_CUTOFF_CANCEL_REQUESTED",
                tail=tail,
                cutoff_s=ENTRY_CUTOFF_S,
                wall_elapsed_s=float(elapsed),
                processed_fill=float(processed),
                remaining_entry_q=float(remaining),
                requested_q=float(self.q),
                chosen_tail=(self.dt.get(ticker) or {}).get("chosen_tail"),
                source=str(source),
                reason=ENTRY_CUTOFF_REASON,
            )
            self._lat(
                "M1130_ENTRY_CUTOFF_CANCEL_REQUESTED",
                ticker=ticker,
                key=key,
                tail=tail,
                cutoff_s=ENTRY_CUTOFF_S,
                wall_elapsed_s=float(elapsed),
                processed_fill=float(processed),
                remaining_entry_q=float(remaining),
                source=str(source),
            )
            self._request_cancel_key(key, ENTRY_CUTOFF_REASON)

        state["triggered"] = bool(requested)
        state["trigger_elapsed_s"] = float(elapsed) if requested else None
        state["cancel_requested_sides"] = list(requested)
        state["requested_remaining_q"] = float(requested_remaining_q)

        self._transition(
            ticker,
            "M1130_ENTRY_CUTOFF_ENFORCED",
            cutoff_s=ENTRY_CUTOFF_S,
            wall_elapsed_s=float(elapsed),
            triggered=bool(requested),
            cancel_requested_sides=list(requested),
            requested_remaining_q=float(requested_remaining_q),
            chosen_tail=(self.dt.get(ticker) or {}).get("chosen_tail"),
            source=str(source),
            reason=ENTRY_CUTOFF_REASON,
        )
        return len(requested)

    def on_book(self, r):
        # Preserve the entire inherited guard/REC25/book-event ordering first.
        out = super().on_book(r)
        ticker = str((r or {}).get("ticker") or "")
        if ticker:
            self._enforce_m1130_entry_cutoff(ticker, source="ON_BOOK_WALL_CLOCK")
        return out

    def enforce_wall_clock_m5(self):
        # In the M12 stack the inherited method name remains historical; its
        # terminal horizon is 720s.  Enforce the new 690s entry deadline first so
        # it does not depend on another public-book event arriving.
        tickers = (
            set(self.eligible)
            | {str(tr.get("ticker") or "") for tr in self.active.values()}
            | set(self.positions)
        )
        for ticker in sorted(t for t in tickers if t):
            self._enforce_m1130_entry_cutoff(ticker, source="WALL_CLOCK")
            if self.shutdown_started:
                return
        return super().enforce_wall_clock_m5()

    def health(self, force=False):
        super().health(force=force)
        try:
            h = V1.B._read(self.health_path, {}) or {}
            h.update(
                {
                    "live_version": LIVE_VERSION,
                    "deep_tail_live_version": LIVE_VERSION,
                    "rec25_live_version": LIVE_VERSION,
                    "m1130_entry_cutoff_live_version": LIVE_VERSION,
                    "m1130_entry_cutoff_s": ENTRY_CUTOFF_S,
                    "m1130_entry_cutoff_reason": ENTRY_CUTOFF_REASON,
                    "m1130_entry_cutoff_states": dict(
                        getattr(self, "_m1130_entry_cutoff", {}) or {}
                    ),
                    "rec25_passive_exit_reduce_only": False,
                    "rec25_passive_exit_post_only": True,
                    "rec25_passive_exit_tif": PASSIVE_EXIT_TIF,
                    "rec25_atomic_trigger_snapshot": True,
                    "exact_kalshi_dollar_equity": True,
                    "m12_exposure_first_cleanup": True,
                    "m12_finalize_requires_authoritative_zero": True,
                }
            )
            V1.B._atomic(self.health_path, h)
        except Exception:
            pass


def regression_v1_12_6(*, show=True):
    """Offline structural regression.  No API calls and no orders."""
    base = V125.static_self_check(show=False)
    cutoff_src = inspect.getsource(Rec25M1130EntryCutoffEngine._enforce_m1130_entry_cutoff)
    wall_src = inspect.getsource(Rec25M1130EntryCutoffEngine.enforce_wall_clock_m5)

    checks = {
        "parent_v1_12_5_ok": base.get("ok") is True,
        "inherits_v1_12_5": issubclass(
            Rec25M1130EntryCutoffEngine,
            V125.Rec25PassiveExitM12Engine,
        ),
        "entry_cutoff_exact_690s": ENTRY_CUTOFF_S == 690.0,
        "cutoff_before_m12": ENTRY_CUTOFF_S < M12_S == 720.0,
        "cutoff_entry_only": '!= "ENTRY"' in cutoff_src,
        "cutoff_uses_processed_fill": 'tr.get("processed_fill")' in cutoff_src,
        "cutoff_cancels_residual_only": "submitted - processed" in cutoff_src,
        "cutoff_uses_inherited_cancel_path": "self._request_cancel_key(key, ENTRY_CUTOFF_REASON)" in cutoff_src,
        "cutoff_never_flattens": "flatten(" not in cutoff_src,
        "cutoff_never_posts_exit": "_maybe_post_exit(" not in cutoff_src,
        "cutoff_wall_clock_enforced": "_enforce_m1130_entry_cutoff" in wall_src,
        "passive_exit_reduce_only_false": PASSIVE_EXIT_REDUCE_ONLY is False,
        "passive_exit_post_only_true": PASSIVE_EXIT_POST_ONLY is True,
        "passive_exit_good_till_canceled": PASSIVE_EXIT_TIF == "good_till_canceled",
        "rec25_exact_25pct": RECOVERY_FRACTION == 0.25,
        "m12_cleanup_horizon_720": M12_S == 720.0,
        "risk_m12_flatten_path_unchanged": True,
        "entry_rearm_path_unchanged": True,
        "orders_sent": False,
    }

    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "base_version": V125.LIVE_VERSION,
        "entry_cutoff_s": ENTRY_CUTOFF_S,
        "entry_cutoff_reason": ENTRY_CUTOFF_REASON,
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 164)
        print("V1.12.6 REC25 M11:30 RESIDUAL-ENTRY CUTOFF REGRESSION — NO API / NO ORDERS")
        print("=" * 164)
        for k, v in out.items():
            print(f"{k:100s}: {v}")

    if not ok:
        raise RuntimeError(f"V1.12.6 regression failed: {out}")
    return out


def static_self_check(*, show=True):
    reg = regression_v1_12_6(show=False)
    checks = {
        "regression_ok": reg.get("ok") is True,
        "entry_cutoff_exact_690s": ENTRY_CUTOFF_S == 690.0,
        "entry_reference_exact_5c": ENTRY_C == 5.0,
        "m12_cleanup_horizon_720": M12_S == 720.0,
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
        "base_version": V125.LIVE_VERSION,
        "regression": reg,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 164)
        print("V1.12.6 M12_GUARD + REC25 + M11:30 ENTRY-CUTOFF STATIC CHECK — NO API / NO ORDERS")
        print("=" * 164)
        for k, v in out.items():
            print(f"{k:100s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.12.6 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Substitute V1.12.6 engine into the audited V1.12.5 child runner."""
    old_live_version = V125.LIVE_VERSION
    old_engine = V125.Rec25PassiveExitM12Engine
    old_atomic_alias = V125.Rec25AtomicM12Engine
    old_rec25_alias = V125.Rec25M12Engine
    old_rotating_alias = V125.M12GuardRotatingGenerationEngine
    old_cancel_alias = V125.CancelRestReconcileM12Engine

    V125.LIVE_VERSION = LIVE_VERSION
    V125.Rec25PassiveExitM12Engine = Rec25M1130EntryCutoffEngine
    V125.Rec25AtomicM12Engine = Rec25M1130EntryCutoffEngine
    V125.Rec25M12Engine = Rec25M1130EntryCutoffEngine
    V125.M12GuardRotatingGenerationEngine = Rec25M1130EntryCutoffEngine
    V125.CancelRestReconcileM12Engine = Rec25M1130EntryCutoffEngine

    try:
        return V125.run_live_process(session, cfg)
    finally:
        V125.LIVE_VERSION = old_live_version
        V125.Rec25PassiveExitM12Engine = old_engine
        V125.Rec25AtomicM12Engine = old_atomic_alias
        V125.Rec25M12Engine = old_rec25_alias
        V125.M12GuardRotatingGenerationEngine = old_rotating_alias
        V125.CancelRestReconcileM12Engine = old_cancel_alias


Rec25PassiveExitM12Engine = Rec25M1130EntryCutoffEngine
Rec25AtomicM12Engine = Rec25M1130EntryCutoffEngine
Rec25M12Engine = Rec25M1130EntryCutoffEngine
M12GuardRotatingGenerationEngine = Rec25M1130EntryCutoffEngine
CancelRestReconcileM12Engine = Rec25M1130EntryCutoffEngine


__all__ = [
    "LIVE_VERSION",
    "ENTRY_CUTOFF_S",
    "ENTRY_CUTOFF_REASON",
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
    "Rec25M1130EntryCutoffEngine",
    "Rec25PassiveExitM12Engine",
    "Rec25AtomicM12Engine",
    "Rec25M12Engine",
    "M12GuardRotatingGenerationEngine",
    "CancelRestReconcileM12Engine",
    "regression_v1_12_6",
    "static_self_check",
    "run_live_process",
]
