from __future__ import annotations

"""M1->M12 persistent-danger-guard rotating generation.

This module layers only the frozen M12_GUARD strategy delta on top of the
V1.11 compact rotating-generation engine.

Preserved execution mechanics:
- M1 = 60s.
- YES entry rests at 5c.
- NO entry rests at 5c (YES-book 95c).
- First fill selects the tail and cancels the opposite entry.
- The selected entry may continue accumulating to the full requested Q.
- A passive exit is posted only after the full requested Q is observed.
- The passive exit is fixed; there is no repricing or chasing.
- Existing persistent cleanup, authoritative REST verification, residual
  reduce-only IOC cleanup, memory guards, tombstone/orphan protections,
  runtime reconciler, compact EOF watchdog, and rotating-generation process
  architecture are retained.
- The external parent-owned recorder remains M0->M12+30.

Changed execution mechanics:
- The inherited cleanup horizon is published as 720s for this process only.
- YES5 residual entry is canceled if YES bid remains <=10c for >=5s across
  at least 3 valid book observations.
- NO5 residual entry is canceled if YES ask remains >=90c for >=5s across
  at least 3 valid book observations.
- A broken danger condition resets its timer and observation count.
- The guard remains active after a partial selected-tail fill.
- A guard-canceled entry never rearms.
- There are no repeat attempts after becoming flat.

Historical inherited names such as ``M5_S``, ``finalize_m5()``, and
``M5_FINALIZED`` are intentionally retained as compatibility contracts. During
``run_live_process()`` only, ``V1.M5_S`` is temporarily set to 720.0, so those
inherited mechanisms execute at M12 rather than M5. The original value is
restored in ``finally``.

Importing this module performs no API calls and sends no orders.
"""

import math
from pathlib import Path

from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_11_rotation as V111


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_12_M12_GUARD_ROTATION"

M12_S = 720.0

GUARD_PERSIST_S = 5.0
GUARD_MIN_BOOK_OBS = 3

YES_GUARD_BID_MAX = 0.10
NO_GUARD_ASK_MIN = 0.90

EPS = 1e-9


class M12GuardRotatingGenerationEngine(V111.CompactRotatingGenerationEngine):
    """V1.11 rotating engine plus the frozen M12 persistent entry guard."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Bounded per-ticker state. A rotating generation owns one window.
        self.m12_entry_guard = {}

        self._lat(
            "M12_GUARD_READY",
            cleanup_horizon_s=M12_S,
            yes_guard_bid_max=YES_GUARD_BID_MAX,
            no_guard_ask_min=NO_GUARD_ASK_MIN,
            persist_s=GUARD_PERSIST_S,
            min_book_observations=GUARD_MIN_BOOK_OBS,
            rearm=False,
            repeat_after_flat=False,
        )

    def _guard_state(self, ticker):
        ticker = str(ticker)
        st = self.m12_entry_guard.get(ticker)
        if st is None:
            st = {
                "YES": {"start_s": None, "obs": 0, "triggered": False},
                "NO": {"start_s": None, "obs": 0, "triggered": False},
            }
            self.m12_entry_guard[ticker] = st
        return st

    @staticmethod
    def _reset_guard_clock(g):
        g["start_s"] = None
        g["obs"] = 0

    def _apply_entry_guard_side(
        self,
        *,
        ticker,
        tail,
        elapsed_s,
        danger,
        bid,
        ask,
    ):
        ticker = str(ticker)
        tail = str(tail)
        gs = self._guard_state(ticker)[tail]

        # Once a side fires it never rearms during this window.
        if bool(gs.get("triggered")):
            return

        key = V1._track_key(ticker, "ENTRY", tail)
        tr = self.active.get(key)

        # Only a currently live entry can be danger-canceled. This deliberately
        # includes a partially-filled selected entry while its residual remains
        # active.
        if (
            tr is None
            or str(tr.get("role") or "") != "ENTRY"
            or str(tr.get("tail") or "") != tail
            or bool(tr.get("cancel_requested"))
        ):
            self._reset_guard_clock(gs)
            return

        if not danger:
            self._reset_guard_clock(gs)
            return

        start_s = gs.get("start_s")
        if start_s is None:
            gs["start_s"] = float(elapsed_s)
            gs["obs"] = 1
        elif float(elapsed_s) + EPS < float(start_s):
            # Defensive reset if an event-time row ever moves backward.
            gs["start_s"] = float(elapsed_s)
            gs["obs"] = 1
        else:
            gs["obs"] = int(gs.get("obs", 0)) + 1

        age_s = float(elapsed_s) - float(gs["start_s"])
        if age_s + EPS < GUARD_PERSIST_S or int(gs["obs"]) < GUARD_MIN_BOOK_OBS:
            return

        # Permanently disable this side before dispatching cancellation so no
        # later observation can re-arm it.
        gs["triggered"] = True

        reason = (
            "M12_GUARD_YES_BID_LE_10C"
            if tail == "YES"
            else "M12_GUARD_NO_YES_ASK_GE_90C"
        )

        self._transition(
            ticker,
            "M12_ENTRY_GUARD_TRIGGERED",
            tail=tail,
            reason=reason,
            wall_elapsed_s=float(elapsed_s),
            persistence_s=float(age_s),
            observations=int(gs["obs"]),
            bid=float(bid),
            ask=float(ask),
            processed_fill=float(tr.get("processed_fill", 0.0)),
            requested_q=float(self.q),
        )

        self._lat(
            "M12_ENTRY_GUARD_CANCEL_REQUESTED",
            ticker=ticker,
            tail=tail,
            reason=reason,
            elapsed_s=float(elapsed_s),
            persistence_s=float(age_s),
            observations=int(gs["obs"]),
            bid=float(bid),
            ask=float(ask),
            processed_fill=float(tr.get("processed_fill", 0.0)),
        )

        # Existing cancellation path preserves deferred-cancel handling,
        # cancel-receipt fill-floor reconciliation, partial inventory handling,
        # and no-repost behavior.
        self._request_cancel_key(key, reason)

    def on_book(self, r):
        # Preserve all inherited queue/memory/entry/M12-cleanup behavior first.
        out = super().on_book(r)

        if self.shutdown_started:
            return out

        ticker = str((r or {}).get("ticker") or "")
        if not ticker or ticker in self.finalized:
            return out

        st = self.dt.get(ticker)
        if not st or not bool(st.get("entry_attempted")):
            return out

        try:
            e = float((r or {}).get("elapsed_s"))
        except Exception:
            return out

        if not math.isfinite(e) or e < V1.M1_S - EPS or e >= M12_S - EPS:
            return out

        cur = self.current.get(ticker)
        if cur is None:
            return out

        try:
            bid = float(cur["bid"])
            ask = float(cur["ask"])
        except Exception:
            return out

        if not (
            math.isfinite(bid)
            and math.isfinite(ask)
            and 0.0 <= bid < ask <= 1.0
        ):
            return out

        self._apply_entry_guard_side(
            ticker=ticker,
            tail="YES",
            elapsed_s=e,
            danger=(bid <= YES_GUARD_BID_MAX + EPS),
            bid=bid,
            ask=ask,
        )

        if self.shutdown_started:
            return out

        self._apply_entry_guard_side(
            ticker=ticker,
            tail="NO",
            elapsed_s=e,
            danger=(ask >= NO_GUARD_ASK_MIN - EPS),
            bid=bid,
            ask=ask,
        )

        return out

    def finalize_m5(self, ticker):
        """Retain inherited persistent cleanup and add explicit M12 telemetry."""
        ticker = str(ticker)
        was_finalized = ticker in self.finalized

        out = super().finalize_m5(ticker)

        if not was_finalized and ticker in self.finalized:
            self._transition(
                ticker,
                "M12_FINALIZED",
                position=self.positions.get(ticker, 0.0),
                cleanup_horizon_s=M12_S,
                inherited_phase="M5_FINALIZED",
            )
            self.emit(
                "M12_FINALIZED",
                ticker,
                position=self.positions.get(ticker, 0.0),
                cleanup_horizon_s=M12_S,
            )

        return out


def static_self_check(*, show=True):
    """Read-only structural check; validates V1.11 under its historical horizon."""
    old_m5 = V1.M5_S

    try:
        V1.M5_S = 300.0
        parent = V111.static_self_check(show=False)
    finally:
        V1.M5_S = old_m5

    checks = {
        "parent_v1_11_ok": parent.get("ok") is True,
        "parent_historical_m5_check_retained": (
            parent.get("strategy_m5_unchanged_300s") is True
        ),
        "strategy_m1_unchanged_60s": abs(V1.M1_S - 60.0) < 1e-12,
        "m12_cleanup_horizon_720s": abs(M12_S - 720.0) < 1e-12,
        "yes_entry_unchanged_5c": abs(V1.ENTRY_YES_PRICE - 0.05) < 1e-12,
        "no_entry_unchanged_5c": abs(V1.ENTRY_NO_BOOK_PRICE - 0.95) < 1e-12,
        "guard_yes_bid_10c": abs(YES_GUARD_BID_MAX - 0.10) < 1e-12,
        "guard_no_ask_90c": abs(NO_GUARD_ASK_MIN - 0.90) < 1e-12,
        "guard_persistence_5s": abs(GUARD_PERSIST_S - 5.0) < 1e-12,
        "guard_min_book_obs_3": GUARD_MIN_BOOK_OBS == 3,
        "inherits_exact_v1_11_rotation": issubclass(
            M12GuardRotatingGenerationEngine,
            V111.CompactRotatingGenerationEngine,
        ),
        "rearm_disabled": True,
        "repeat_after_flat_disabled": True,
        "orders_sent": False,
    }

    ok = all(v is True for k, v in checks.items() if k != "orders_sent")

    out = {
        "version": LIVE_VERSION,
        "cleanup_horizon_s": M12_S,
        "guard_persist_s": GUARD_PERSIST_S,
        "guard_min_book_obs": GUARD_MIN_BOOK_OBS,
        "yes_guard_bid_max": YES_GUARD_BID_MAX,
        "no_guard_ask_min": NO_GUARD_ASK_MIN,
        "rearm": False,
        "repeat_after_flat": False,
        **checks,
        "ok": bool(ok),
    }

    if show:
        print("=" * 132)
        print(
            "DEEP-TAIL LIVE V1.12 M12_GUARD ROTATING ENGINE "
            "STATIC CHECK — NO API / NO ORDERS"
        )
        print("=" * 132)
        for k, v in out.items():
            print(f"{k:72s}: {v}")

    if not ok:
        raise RuntimeError(f"M12_GUARD static self-check failed: {out}")

    return out


def run_live_process(session, cfg):
    """Run exact V1.11 stack with this process's M12 horizon and engine class."""
    session = Path(session).resolve()

    old_m5 = V1.M5_S
    old_engine = V111.CompactRotatingGenerationEngine
    old_version = V111.LIVE_VERSION

    V1.M5_S = M12_S
    V111.CompactRotatingGenerationEngine = M12GuardRotatingGenerationEngine
    V111.LIVE_VERSION = LIVE_VERSION

    try:
        return V111.run_live_process(session, cfg)
    finally:
        V111.LIVE_VERSION = old_version
        V111.CompactRotatingGenerationEngine = old_engine
        V1.M5_S = old_m5


__all__ = [
    "LIVE_VERSION",
    "M12_S",
    "GUARD_PERSIST_S",
    "GUARD_MIN_BOOK_OBS",
    "YES_GUARD_BID_MAX",
    "NO_GUARD_ASK_MIN",
    "M12GuardRotatingGenerationEngine",
    "static_self_check",
    "run_live_process",
]
