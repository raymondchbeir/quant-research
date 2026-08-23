from __future__ import annotations

"""V1.12.3 M12_GUARD + causal REC25 fixed passive exit.

Only the full-Q exit timing rule changes relative to V1.12.2.

REC25 rule validated in the historical study:
- causal pre-break anchor = median chosen-tail midpoint in
  [first_fill-10s, first_fill-1s];
- if that primary window has fewer than 3 observations, use the median in
  [first_fill-30s, first_fill);
- break_c = pre_mid_c - 5c;
- after FULL Q, trigger at the first chosen-tail midpoint satisfying
  5c + 0.25 * break_c;
- then post one fixed reduce-only post-only GTC at the inherited freshest
  certified chosen-tail ask; never reprice or chase;
- if no valid positive break / no REC25 trigger, inherited M12 cleanup remains
  the fallback.

Entry, Q, M1, M12 guard, risk, stale-REST reconciliation, orphan fail-closed,
recorder, rotation and M12 cleanup are otherwise unchanged.

The live anchor buffer is bounded to 31 seconds and records only L1 state changes
(price or L1 size), so deeper-book-only updates do not overweight the midpoint
median. Importing this module performs no API calls and sends no orders.
"""

from collections import deque
from pathlib import Path
import math

import numpy as np

from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_m12_guard_rotation as V112
from . import mm_deep_tail_join_ask_live_v1_12_1_m12_v18_compat as V1121
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_12_3_M12_GUARD_REC25"

M12_S = V1122.M12_S
GUARD_PERSIST_S = V1122.GUARD_PERSIST_S
GUARD_MIN_BOOK_OBS = V1122.GUARD_MIN_BOOK_OBS
YES_GUARD_BID_MAX = V1122.YES_GUARD_BID_MAX
NO_GUARD_ASK_MIN = V1122.NO_GUARD_ASK_MIN
CANCEL_REST_GRACE_S = V1122.CANCEL_REST_GRACE_S

ENTRY_C = 5.0
RECOVERY_FRACTION = 0.25
PRE_LOOKBACK_S = 10.0
PRE_EXCLUDE_S = 1.0
PRE_FALLBACK_S = 30.0
HISTORY_KEEP_S = PRE_FALLBACK_S + 1.0
MIN_PRIMARY_OBS = 3
EPS = 1e-9


def _finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _chosen_mid_c(tail, yes_bid, yes_ask):
    bid = _finite(yes_bid)
    ask = _finite(yes_ask)
    if bid is None or ask is None or not (0.0 <= bid < ask <= 1.0):
        return None
    yes_mid_c = 50.0 * (bid + ask)
    tail = str(tail or "").upper()
    if tail == "YES":
        return yes_mid_c
    if tail == "NO":
        return 100.0 - yes_mid_c
    return None


def _anchor_from_history(history, *, tail, first_fill_s):
    """Pure implementation of the validated historical causal anchor."""
    first_fill_s = _finite(first_fill_s)
    if first_fill_s is None:
        return {
            "pre_mid_c": None,
            "pre_obs": 0,
            "anchor_mode": "MISSING_FIRST_FILL_TIME",
            "break_c": None,
            "threshold_c": None,
        }

    rows = list(history or [])
    primary_lo = max(0.0, first_fill_s - PRE_LOOKBACK_S)
    primary_hi = max(0.0, first_fill_s - PRE_EXCLUDE_S)
    primary = [r for r in rows if primary_lo <= float(r["elapsed_s"]) <= primary_hi]

    if len(primary) >= MIN_PRIMARY_OBS:
        selected = primary
        mode = "PRIMARY_10S_TO_1S"
    else:
        fallback_lo = max(0.0, first_fill_s - PRE_FALLBACK_S)
        selected = [r for r in rows if fallback_lo <= float(r["elapsed_s"]) < first_fill_s]
        mode = "FALLBACK_30S"

    mids = [
        _chosen_mid_c(tail, r.get("yes_bid"), r.get("yes_ask"))
        for r in selected
    ]
    mids = [float(x) for x in mids if x is not None and math.isfinite(float(x))]
    pre_mid_c = float(np.median(np.asarray(mids, dtype=float))) if mids else None
    break_c = (pre_mid_c - ENTRY_C) if pre_mid_c is not None else None
    threshold_c = (
        ENTRY_C + RECOVERY_FRACTION * break_c
        if break_c is not None and break_c > 0.0
        else None
    )
    return {
        "pre_mid_c": pre_mid_c,
        "pre_obs": len(mids),
        "anchor_mode": mode,
        "break_c": break_c,
        "threshold_c": threshold_c,
    }


class Rec25M12Engine(V1122.CancelRestReconcileM12Engine):
    """Exact V1.12.2 engine with delayed REC25 full-Q exit timing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rec25_history = {}
        self._rec25_last_l1 = {}
        self._rec25 = {}
        self._lat(
            "REC25_EXIT_POLICY_READY",
            entry_c=ENTRY_C,
            recovery_fraction=RECOVERY_FRACTION,
            pre_lookback_s=PRE_LOOKBACK_S,
            pre_exclude_s=PRE_EXCLUDE_S,
            pre_fallback_s=PRE_FALLBACK_S,
            primary_min_obs=MIN_PRIMARY_OBS,
            history_keep_s=HISTORY_KEEP_S,
            fixed_exit=True,
            reprice=False,
            chase=False,
        )

    def _rec25_state(self, ticker):
        ticker = str(ticker)
        st = self._rec25.get(ticker)
        if st is None:
            st = {
                "first_fill_s": None,
                "full_entry_s": None,
                "tail": None,
                "pre_mid_c": None,
                "pre_obs": 0,
                "anchor_mode": None,
                "break_c": None,
                "threshold_c": None,
                "anchor_ready": False,
                "triggered": False,
                "trigger_s": None,
                "trigger_mid_c": None,
            }
            self._rec25[ticker] = st
        return st

    def _record_l1(self, r):
        ticker = str((r or {}).get("ticker") or "")
        e = _finite((r or {}).get("elapsed_s"))
        cur = self.current.get(ticker) if ticker else None
        if not ticker or e is None or cur is None:
            return

        bid = _finite(cur.get("bid"))
        ask = _finite(cur.get("ask"))
        bid_size = _finite(cur.get("bid_size"))
        ask_size = _finite(cur.get("ask_size"))
        if bid is None or ask is None or not (0.0 <= bid < ask <= 1.0):
            return

        signature = (bid, ask, bid_size, ask_size)
        if self._rec25_last_l1.get(ticker) == signature:
            return
        self._rec25_last_l1[ticker] = signature

        hist = self._rec25_history.get(ticker)
        if hist is None:
            hist = deque()
            self._rec25_history[ticker] = hist
        hist.append({"elapsed_s": float(e), "yes_bid": float(bid), "yes_ask": float(ask)})
        cutoff = float(e) - HISTORY_KEEP_S
        while hist and float(hist[0]["elapsed_s"]) < cutoff:
            hist.popleft()

    def _capture_first_fill_anchor(self, ticker, tail):
        st = self._rec25_state(ticker)
        if st.get("anchor_ready"):
            return
        st["first_fill_s"] = _finite(self.wall_elapsed(ticker))
        st["tail"] = str(tail or "").upper()
        st.update(
            _anchor_from_history(
                self._rec25_history.get(str(ticker), ()),
                tail=st["tail"],
                first_fill_s=st["first_fill_s"],
            )
        )
        st["anchor_ready"] = True
        payload = {k: st.get(k) for k in (
            "tail", "first_fill_s", "pre_mid_c", "pre_obs",
            "anchor_mode", "break_c", "threshold_c",
        )}
        self._transition(ticker, "REC25_ANCHOR_FROZEN", **payload)
        self._lat("REC25_ANCHOR_FROZEN", ticker=ticker, **payload)

    def _capture_full_entry_time(self, ticker):
        st = self._rec25_state(ticker)
        if st.get("full_entry_s") is None:
            st["full_entry_s"] = _finite(self.wall_elapsed(ticker))
            self._lat(
                "REC25_FULL_ENTRY_CLOCK",
                ticker=ticker,
                full_entry_s=st["full_entry_s"],
                threshold_c=st.get("threshold_c"),
            )

    def _process_effective_fill(self, key, source, raw=None):
        tr_before = self.active.get(key)
        ticker = str((tr_before or {}).get("ticker") or "")
        role = str((tr_before or {}).get("role") or "")
        dt_before = self.dt.get(ticker) if ticker else None
        chosen_before = (dt_before or {}).get("chosen_tail")
        full_before = bool((dt_before or {}).get("full_entry_ready"))

        out = super()._process_effective_fill(key, source, raw=raw)

        if role == "ENTRY" and ticker:
            dt_after = self.dt.get(ticker) or {}
            chosen_after = dt_after.get("chosen_tail")
            full_after = bool(dt_after.get("full_entry_ready"))
            if chosen_before is None and chosen_after is not None:
                self._capture_first_fill_anchor(ticker, chosen_after)
            if (not full_before) and full_after:
                self._capture_full_entry_time(ticker)
                self._evaluate_rec25(ticker)
        return out

    def _evaluate_rec25(self, ticker, elapsed_s=None):
        ticker = str(ticker)
        if self.shutdown_started or ticker in self.finalized:
            return False

        dt = self.dt.get(ticker) or {}
        if not bool(dt.get("full_entry_ready")) or bool(dt.get("exit_posted")):
            return False

        st = self._rec25_state(ticker)
        threshold_c = _finite(st.get("threshold_c")) if st.get("anchor_ready") else None
        if threshold_c is None:
            dt["phase"] = "FULL_ENTRY_WAITING_REC25_NO_VALID_THRESHOLD"
            return False

        full_s = _finite(st.get("full_entry_s"))
        e = _finite(elapsed_s)
        if e is None:
            e = _finite(self.wall_elapsed(ticker))
        if e is None or (full_s is not None and e + EPS < full_s):
            return False

        cur = self.current.get(ticker)
        if cur is None:
            return False
        mid_c = _chosen_mid_c(st.get("tail"), cur.get("bid"), cur.get("ask"))
        if mid_c is None or mid_c + 1e-7 < threshold_c:
            if not st.get("triggered"):
                dt["phase"] = "FULL_ENTRY_WAITING_REC25"
            return False

        if not st.get("triggered"):
            st["triggered"] = True
            st["trigger_s"] = float(e)
            st["trigger_mid_c"] = float(mid_c)
            payload = {
                "tail": st.get("tail"),
                "elapsed_s": float(e),
                "full_entry_s": full_s,
                "delay_after_full_s": (float(e) - full_s if full_s is not None else None),
                "pre_mid_c": st.get("pre_mid_c"),
                "break_c": st.get("break_c"),
                "threshold_c": threshold_c,
                "current_mid_c": float(mid_c),
            }
            self._transition(ticker, "REC25_TRIGGERED", **payload)
            self._lat("REC25_TRIGGERED", ticker=ticker, **payload)

        self._maybe_post_exit(ticker)
        return True

    def _maybe_post_exit(self, ticker):
        """Gate inherited fixed passive exit until REC25 has causally triggered."""
        ticker = str(ticker)
        dt = self.dt.get(ticker) or {}
        if not bool(dt.get("full_entry_ready")) or bool(dt.get("exit_posted")):
            return
        if not bool(self._rec25_state(ticker).get("triggered")):
            dt["phase"] = "FULL_ENTRY_WAITING_REC25"
            return
        return super()._maybe_post_exit(ticker)

    def on_book(self, r):
        out = super().on_book(r)
        self._record_l1(r)
        ticker = str((r or {}).get("ticker") or "")
        e = _finite((r or {}).get("elapsed_s"))
        if ticker and e is not None:
            self._evaluate_rec25(ticker, elapsed_s=e)
        return out

    def health(self, force=False):
        super().health(force=force)
        try:
            h = V1.B._read(self.health_path, {}) or {}
            compact = {
                str(ticker): {
                    k: st.get(k)
                    for k in (
                        "tail", "first_fill_s", "full_entry_s", "pre_mid_c",
                        "pre_obs", "anchor_mode", "break_c", "threshold_c",
                        "triggered", "trigger_s", "trigger_mid_c",
                    )
                }
                for ticker, st in self._rec25.items()
            }
            h.update(
                {
                    "rec25_live_version": LIVE_VERSION,
                    "rec25_recovery_fraction": RECOVERY_FRACTION,
                    "rec25_pre_lookback_s": PRE_LOOKBACK_S,
                    "rec25_pre_exclude_s": PRE_EXCLUDE_S,
                    "rec25_pre_fallback_s": PRE_FALLBACK_S,
                    "rec25_states": compact,
                    "rec25_history_rows": {str(k): len(v) for k, v in self._rec25_history.items()},
                    "rec25_fixed_exit_no_reprice": True,
                }
            )
            V1.B._atomic(self.health_path, h)
        except Exception:
            pass


def regression_rec25_rule(*, show=True):
    """Pure offline regression for primary/fallback windows, NO transform and threshold."""
    primary_hist = [
        {"elapsed_s": 90.0, "yes_bid": 0.19, "yes_ask": 0.21},  # 20c midpoint
        {"elapsed_s": 92.0, "yes_bid": 0.21, "yes_ask": 0.23},  # 22c midpoint
        {"elapsed_s": 94.0, "yes_bid": 0.20, "yes_ask": 0.22},  # 21c midpoint
        {"elapsed_s": 99.2, "yes_bid": 0.01, "yes_ask": 0.03},  # excluded final second
    ]
    yes = _anchor_from_history(primary_hist, tail="YES", first_fill_s=100.0)
    no = _anchor_from_history(primary_hist, tail="NO", first_fill_s=100.0)

    fallback_hist = [
        {"elapsed_s": 75.0, "yes_bid": 0.29, "yes_ask": 0.31},  # 30c
        {"elapsed_s": 95.0, "yes_bid": 0.09, "yes_ask": 0.11},  # 10c
    ]
    fb = _anchor_from_history(fallback_hist, tail="YES", first_fill_s=100.0)

    checks = {
        "primary_mode": yes["anchor_mode"] == "PRIMARY_10S_TO_1S",
        "primary_exact_3_obs": yes["pre_obs"] == 3,
        "primary_final_second_excluded": abs(float(yes["pre_mid_c"]) - 21.0) < 1e-12,
        "primary_yes_break_16c": abs(float(yes["break_c"]) - 16.0) < 1e-12,
        "primary_yes_rec25_threshold_9c": abs(float(yes["threshold_c"]) - 9.0) < 1e-12,
        "no_mid_transform_79c": abs(float(no["pre_mid_c"]) - 79.0) < 1e-12,
        "fallback_mode": fb["anchor_mode"] == "FALLBACK_30S",
        "fallback_used_when_primary_lt3": fb["pre_obs"] == 2,
        "fallback_mid_20c": abs(float(fb["pre_mid_c"]) - 20.0) < 1e-12,
        "inherits_v1_12_2": issubclass(Rec25M12Engine, V1122.CancelRestReconcileM12Engine),
    }
    out = {**checks, "ok": all(checks.values()), "api_called": False, "orders_sent": False}
    if show:
        print("=" * 140)
        print("V1.12.3 REC25 RULE REGRESSION — NO API / NO ORDERS")
        print("=" * 140)
        for k, v in out.items():
            print(f"{k:80s}: {v}")
    if not out["ok"]:
        raise RuntimeError(f"REC25 regression failed: {out}")
    return out


def static_self_check(*, show=True):
    base = V1122.static_self_check(show=False)
    reg = regression_rec25_rule(show=False)
    checks = {
        "base_v1_12_2_ok": base.get("ok") is True,
        "rec25_rule_regression": reg.get("ok") is True,
        "m12_cleanup_horizon_720": M12_S == 720.0,
        "entry_reference_exact_5c": ENTRY_C == 5.0,
        "recovery_fraction_exact_25pct": RECOVERY_FRACTION == 0.25,
        "pre_lookback_exact_10s": PRE_LOOKBACK_S == 10.0,
        "pre_exclude_exact_1s": PRE_EXCLUDE_S == 1.0,
        "pre_fallback_exact_30s": PRE_FALLBACK_S == 30.0,
        "primary_min_obs_exact_3": MIN_PRIMARY_OBS == 3,
        "guard_yes_bid_10c": YES_GUARD_BID_MAX == 0.10,
        "guard_no_ask_90c": NO_GUARD_ASK_MIN == 0.90,
        "guard_persist_5s": GUARD_PERSIST_S == 5.0,
        "guard_min_obs_3": GUARD_MIN_BOOK_OBS == 3,
        "cancel_rest_grace_5s": CANCEL_REST_GRACE_S == 5.0,
        "fixed_exit_no_reprice": True,
        "m12_fallback_preserved": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "base_version": V1122.LIVE_VERSION,
        "regression": reg,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 148)
        print("V1.12.3 M12_GUARD + REC25 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 148)
        for k, v in out.items():
            print(f"{k:84s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.12.3 REC25 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run the exact V1.12.2/M12 stack with only REC25 exit timing changed."""
    session = Path(session).resolve()

    old_v112_engine = V112.M12GuardRotatingGenerationEngine
    old_v1121_engine = V1121.M12GuardRotatingGenerationEngine
    old_v1121_version = V1121.LIVE_VERSION

    V112.M12GuardRotatingGenerationEngine = Rec25M12Engine
    V1121.M12GuardRotatingGenerationEngine = Rec25M12Engine
    V1121.LIVE_VERSION = LIVE_VERSION

    try:
        V1.B._atomic(
            session / "rec25_exit_policy.json",
            {
                "live_version": LIVE_VERSION,
                "entry_reference_c": ENTRY_C,
                "recovery_fraction": RECOVERY_FRACTION,
                "primary_anchor": "median chosen-tail midpoint [first_fill-10s, first_fill-1s]",
                "fallback_anchor": "median chosen-tail midpoint [first_fill-30s, first_fill) when primary has <3 observations",
                "trigger": "first post-full-Q chosen-tail midpoint >= 5c + 0.25*(pre_mid_c-5c)",
                "exit": "one fixed reduce-only post-only GTC at freshest certified chosen-tail ask",
                "reprice": False,
                "chase": False,
                "m12_cleanup_fallback": True,
            },
        )
    except Exception:
        pass

    try:
        return V1121.run_live_process(session, cfg)
    finally:
        V1121.LIVE_VERSION = old_v1121_version
        V1121.M12GuardRotatingGenerationEngine = old_v1121_engine
        V112.M12GuardRotatingGenerationEngine = old_v112_engine


M12GuardRotatingGenerationEngine = Rec25M12Engine
CancelRestReconcileM12Engine = Rec25M12Engine


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
    "Rec25M12Engine",
    "M12GuardRotatingGenerationEngine",
    "regression_rec25_rule",
    "static_self_check",
    "run_live_process",
]
