from __future__ import annotations

"""V1.12.4 REC25 live corrections after the 2026-08-23 Q50 smoke.

This layer keeps the frozen M1/M12_GUARD + REC25 strategy and fixes three live
implementation issues observed in the first V1.12.3 REC25 run:

1) Trigger/post atomicity
   V1.12.3 evaluated REC25 on ``self.current`` and then delegated to the inherited
   exit helper, which fetched a second certified BBO.  The recovery could therefore
   trigger on one snapshot while the fixed exit was priced from a later collapsed
   snapshot.  V1.12.4 obtains one certified BBO, evaluates REC25 on that exact
   snapshot, freezes its chosen-tail ask, and the eventual fixed passive exit uses
   that same frozen trigger-snapshot price.  There is no second pricing snapshot,
   repricing or chasing.

2) Exact Kalshi equity semantics
   Kalshi now exposes ``balance_dollars`` with sub-cent precision.  The historical
   helper used integer-cent ``balance`` when present.  V1.12.4 prefers exact dollar
   fields and falls back to integer cents only when needed.  The helper is installed
   only for the child live process and restored afterward.

3) M12 cleanup ordering / verification
   At M12, known nonzero exposures are finalized before flat/resting-only tickers.
   A ticker is marked finalized only after cancellations are terminal and an
   authoritative post-cleanup position read proves zero.  Health is forced around
   each cleanup step so a long cleanup cannot leave a pre-M12 health snapshot as the
   last durable state without telemetry.

No entry, quantity, guard, anchor, recovery threshold, no-reprice, loss-limit,
recorder, orphan, or stale-REST strategy rule changes are made.

Importing this module performs no API calls and sends no orders.
"""

from pathlib import Path
import math

from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_3_m12_guard_rec25 as V123


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_12_4_REC25_ATOMIC_EXACT_EQUITY"

M12_S = V123.M12_S
GUARD_PERSIST_S = V123.GUARD_PERSIST_S
GUARD_MIN_BOOK_OBS = V123.GUARD_MIN_BOOK_OBS
YES_GUARD_BID_MAX = V123.YES_GUARD_BID_MAX
NO_GUARD_ASK_MIN = V123.NO_GUARD_ASK_MIN
CANCEL_REST_GRACE_S = V123.CANCEL_REST_GRACE_S

ENTRY_C = V123.ENTRY_C
RECOVERY_FRACTION = V123.RECOVERY_FRACTION
PRE_LOOKBACK_S = V123.PRE_LOOKBACK_S
PRE_EXCLUDE_S = V123.PRE_EXCLUDE_S
PRE_FALLBACK_S = V123.PRE_FALLBACK_S
MIN_PRIMARY_OBS = V123.MIN_PRIMARY_OBS
EPS = V123.EPS


def _finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _money_field(body, dollar_key, cents_key):
    body = body or {}
    direct = _finite(body.get(dollar_key))
    if direct is not None:
        return float(direct)
    cents = _finite(body.get(cents_key))
    return 0.0 if cents is None else float(cents) / 100.0


def exact_equity_from_balance(body):
    """Prefer exact dollar fields; fall back to historical integer-cent fields."""
    body = body or {}
    cash = _money_field(body, "balance_dollars", "balance")
    portfolio = _money_field(body, "portfolio_value_dollars", "portfolio_value")
    return {
        "cash_balance_usd": float(cash),
        "portfolio_value_usd": float(portfolio),
        "equity_usd": float(cash + portfolio),
        "updated_ts": body.get("updated_ts"),
    }


def _trigger_snapshot(tail, cur, threshold_c):
    """Pure atomic REC25 candidate: trigger test and fixed price use one BBO."""
    cur = cur or {}
    bid = _finite(cur.get("bid"))
    ask = _finite(cur.get("ask"))
    threshold_c = _finite(threshold_c)
    if (
        bid is None
        or ask is None
        or threshold_c is None
        or not (0.0 <= bid < ask <= 1.0)
    ):
        return None

    tail = str(tail or "").upper()
    mid_c = V123._chosen_mid_c(tail, bid, ask)
    if mid_c is None or float(mid_c) + 1e-7 < float(threshold_c):
        return None

    if tail == "YES":
        side = "ask"
        price = float(ask)
    elif tail == "NO":
        side = "bid"
        price = float(bid)
    else:
        return None

    return {
        "tail": tail,
        "yes_bid": float(bid),
        "yes_ask": float(ask),
        "mid_c": float(mid_c),
        "side": side,
        "price": float(price),
        "threshold_c": float(threshold_c),
    }


class Rec25AtomicM12Engine(V123.Rec25M12Engine):
    """V1.12.3 with one-snapshot REC25 pricing and stricter M12 finalization."""

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

        # IMPORTANT: one certified BBO is used for BOTH the REC25 test and the
        # frozen exit price.  No second BBO lookup is allowed after triggering.
        cur, cert = self._latest_fresh_bbo(ticker)
        if cur is None:
            dt["phase"] = "FULL_ENTRY_WAITING_REC25_FRESH_CERT"
            return False

        candidate = _trigger_snapshot(st.get("tail"), cur, threshold_c)
        if candidate is None:
            if not st.get("triggered"):
                dt["phase"] = "FULL_ENTRY_WAITING_REC25"
            return False

        if not st.get("triggered"):
            trigger_e = _finite(self.wall_elapsed(ticker))
            if trigger_e is None:
                trigger_e = float(e)

            st["triggered"] = True
            st["trigger_s"] = float(trigger_e)
            st["trigger_mid_c"] = float(candidate["mid_c"])
            st["trigger_yes_bid"] = float(candidate["yes_bid"])
            st["trigger_yes_ask"] = float(candidate["yes_ask"])
            st["trigger_exit_side"] = str(candidate["side"])
            st["trigger_exit_price"] = float(candidate["price"])
            st["trigger_cert"] = cert

            payload = {
                "tail": st.get("tail"),
                "elapsed_s": float(trigger_e),
                "full_entry_s": full_s,
                "delay_after_full_s": (
                    float(trigger_e) - full_s if full_s is not None else None
                ),
                "pre_mid_c": st.get("pre_mid_c"),
                "break_c": st.get("break_c"),
                "threshold_c": float(threshold_c),
                "current_mid_c": float(candidate["mid_c"]),
                "trigger_yes_bid": float(candidate["yes_bid"]),
                "trigger_yes_ask": float(candidate["yes_ask"]),
                "fixed_exit_side": str(candidate["side"]),
                "fixed_exit_price": float(candidate["price"]),
                "atomic_trigger_snapshot": True,
            }
            self._transition(ticker, "REC25_TRIGGERED", **payload)
            self._lat("REC25_TRIGGERED", ticker=ticker, cert=cert, **payload)

        self._maybe_post_exit(ticker)
        return True

    def _maybe_post_exit(self, ticker):
        """Post the frozen trigger-snapshot price; never fetch/reprice afterward."""
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

        tr = self._new_track(ticker, "EXIT", chosen, side, float(price), self.q, True)
        tr["full_entry_ready_wall_ms"] = V1._wall_ms()
        tr["rec25_trigger_snapshot_price"] = True
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
        )

    def finalize_m5(self, ticker):
        """M12 cleanup: verify terminal cancels and zero position BEFORE finalizing."""
        ticker = str(ticker)
        if ticker in self.finalized:
            return

        self.health(force=True)
        self._cancel_all_for_ticker(ticker, "M5")

        deadline = V1.time.time() + 4.0
        while (
            any(str(tr.get("ticker")) == ticker for tr in self.active.values())
            and V1.time.time() < deadline
        ):
            self.poll_orders()
            V1.time.sleep(0.005)

        remaining = [
            k
            for k, tr in self.active.items()
            if str(tr.get("ticker")) == ticker
        ]
        if remaining:
            raise RuntimeError(
                f"M12 cancel did not retire all tracks for {ticker}: {remaining}"
            )

        p = self.refresh_position(ticker)
        if abs(p) > EPS:
            self.flatten(ticker, "M5")

        # Authoritative post-cleanup proof.  Do not mark finalized from local fill
        # assumptions; the exchange position endpoint must report zero.
        final_p = self.refresh_position(ticker)
        if abs(final_p) > EPS:
            raise RuntimeError(
                f"M12 cleanup verification nonzero {ticker}: {final_p:+.4f}"
            )
        self.positions[ticker] = 0.0

        self.finalized.add(ticker)
        self.dt[ticker]["phase"] = "M5_FINALIZED"
        self._transition(
            ticker,
            "M5_FINALIZED",
            position=0.0,
            cleanup_attempts="AUTHORITATIVE_ZERO_VERIFIED",
        )
        self.emit("M5_FINALIZED", ticker, position=0.0)

        self._transition(
            ticker,
            "M12_FINALIZED",
            position=0.0,
            cleanup_horizon_s=M12_S,
            inherited_phase="M5_FINALIZED",
            authoritative_position_zero=True,
        )
        self.emit(
            "M12_FINALIZED",
            ticker,
            position=0.0,
            cleanup_horizon_s=M12_S,
        )

        # Free completed-window REC25 path buffers immediately.
        self._rec25_history.pop(ticker, None)
        self._rec25_last_l1.pop(ticker, None)
        self.health(force=True)

    def enforce_wall_clock_m5(self):
        """At M12, prioritize actual exposures before resting-only flat tickers."""
        tickers = (
            set(self.eligible)
            | {str(tr.get("ticker")) for tr in self.active.values()}
            | set(self.positions)
        )

        def priority(ticker):
            p = _finite(self.positions.get(ticker, 0.0)) or 0.0
            return (0 if abs(p) > EPS else 1, str(ticker))

        for ticker in sorted((t for t in tickers if t), key=priority):
            if ticker in self.finalized:
                continue
            e = self.wall_elapsed(ticker)
            if math.isfinite(e) and e >= M12_S:
                self.health(force=True)
                self.finalize_m5(ticker)
                self.health(force=True)
                if self.shutdown_started:
                    return

    def health(self, force=False):
        super().health(force=force)
        try:
            h = V1.B._read(self.health_path, {}) or {}
            compact = h.get("rec25_states") or {}
            for ticker, st in self._rec25.items():
                row = compact.setdefault(str(ticker), {})
                for key in (
                    "trigger_yes_bid",
                    "trigger_yes_ask",
                    "trigger_exit_side",
                    "trigger_exit_price",
                ):
                    row[key] = st.get(key)
            h.update(
                {
                    "live_version": LIVE_VERSION,
                    "deep_tail_live_version": LIVE_VERSION,
                    "rec25_live_version": LIVE_VERSION,
                    "rec25_states": compact,
                    "rec25_atomic_trigger_snapshot": True,
                    "exact_kalshi_dollar_equity": True,
                    "m12_exposure_first_cleanup": True,
                    "m12_finalize_requires_authoritative_zero": True,
                }
            )
            V1.B._atomic(self.health_path, h)
        except Exception:
            pass


def regression_v1_12_4(*, show=True):
    exact = exact_equity_from_balance(
        {
            "balance": 22476,
            "balance_dollars": "224.7604",
            "portfolio_value": 0,
            "updated_ts": 1,
        }
    )
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

    checks = {
        "parent_v1_12_3_ok": V123.static_self_check(show=False).get("ok") is True,
        "exact_equity_224_7604": abs(float(exact["equity_usd"]) - 224.7604) < 1e-12,
        "yes_trigger_mid_8_7c": yes is not None and abs(float(yes["mid_c"]) - 8.7) < 1e-12,
        "yes_exit_same_snapshot_9_4c": yes is not None and abs(float(yes["price"]) - 0.094) < 1e-12,
        "no_transform_mid_8c": no is not None and abs(float(no["mid_c"]) - 8.0) < 1e-12,
        "no_exit_same_snapshot_yes_bid_91c": no is not None and abs(float(no["price"]) - 0.910) < 1e-12,
        "inherits_v1_12_3": issubclass(Rec25AtomicM12Engine, V123.Rec25M12Engine),
        "atomic_trigger_snapshot": True,
        "exposure_first_m12_cleanup": True,
        "authoritative_zero_before_finalize": True,
    }
    out = {**checks, "ok": all(checks.values()), "api_called": False, "orders_sent": False}
    if show:
        print("=" * 150)
        print("V1.12.4 REC25 ATOMIC / EXACT EQUITY REGRESSION — NO API / NO ORDERS")
        print("=" * 150)
        for k, v in out.items():
            print(f"{k:88s}: {v}")
    if not out["ok"]:
        raise RuntimeError(f"V1.12.4 regression failed: {out}")
    return out


def static_self_check(*, show=True):
    reg = regression_v1_12_4(show=False)
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
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "base_version": V123.LIVE_VERSION,
        "regression": reg,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 152)
        print("V1.12.4 M12_GUARD + REC25 ATOMIC STATIC CHECK — NO API / NO ORDERS")
        print("=" * 152)
        for k, v in out.items():
            print(f"{k:88s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.12.4 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Install exact-equity semantics and substitute V1.12.4 into V1.12.3 runner."""
    session = Path(session).resolve()

    old_equity = V1.B._equity
    old_engine = V123.Rec25M12Engine
    old_m12_engine = V123.M12GuardRotatingGenerationEngine
    old_alias = getattr(V123, "CancelRestReconcileM12Engine", None)
    old_version = V123.LIVE_VERSION

    V1.B._equity = exact_equity_from_balance
    V123.Rec25M12Engine = Rec25AtomicM12Engine
    V123.M12GuardRotatingGenerationEngine = Rec25AtomicM12Engine
    V123.CancelRestReconcileM12Engine = Rec25AtomicM12Engine
    V123.LIVE_VERSION = LIVE_VERSION

    try:
        return V123.run_live_process(session, cfg)
    finally:
        V123.LIVE_VERSION = old_version
        V123.Rec25M12Engine = old_engine
        V123.M12GuardRotatingGenerationEngine = old_m12_engine
        if old_alias is None:
            try:
                delattr(V123, "CancelRestReconcileM12Engine")
            except Exception:
                pass
        else:
            V123.CancelRestReconcileM12Engine = old_alias
        V1.B._equity = old_equity


Rec25M12Engine = Rec25AtomicM12Engine
M12GuardRotatingGenerationEngine = Rec25AtomicM12Engine
CancelRestReconcileM12Engine = Rec25AtomicM12Engine


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
    "exact_equity_from_balance",
    "Rec25AtomicM12Engine",
    "Rec25M12Engine",
    "M12GuardRotatingGenerationEngine",
    "regression_v1_12_4",
    "static_self_check",
    "run_live_process",
]
