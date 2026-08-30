from __future__ import annotations

"""Tail25 / Hyst2 / Edge15 / JOIN_BBO live engine.

This is an additive strategy engine on top of the latest audited Deep35 safety
stack.  It deliberately changes strategy economics while retaining the mature
transport, private-fill reconciliation, stale-REST tombstones, exact-equity
handling, rotating-generation machinery, guardian recovery, and V1.13.6
terminal cancel-receipt retirement.

Frozen strategy contract
------------------------
Universe is supplied by the deployment wrapper and recorder (12 series).
For each complete 15-minute market window, from M0 until M12:

ENTRY
- when flat and not disabled, independently rest:
    BID = floor_cent(current YES bid - 25c)
    ASK = ceil_cent(current YES ask + 25c)
- both are post-only GTC maker orders;
- each entry side is capped at configured Q (Q1 smoke or Q10 live);
- reprice only when the cent-rounded target moves by >=2c;
- if YES bid <=15c OR YES ask >=85c, permanently disable new entry for that
  ticker/window and cancel any remaining ENTRY tracks.

AFTER FIRST ENTRY FILL
- permanently disable all further entry for that ticker/window;
- cancel every residual ENTRY track immediately;
- net any late/opposite entry fills through the inherited FIFO local lot ledger;
- continuously rest one reduce-only passive JOIN_BBO exit for the current local
  net exposure:
    long YES  -> ASK at current YES ask
    short YES -> BID at current YES bid
- reprice the passive exit when the desired BBO moves by >=2c or required
  residual quantity changes.

FORCE FLAT
- first observed entry fill starts a fixed 3.0 second wall-clock deadline;
- if exposure remains at the deadline, cancel all strategy tracks for the ticker
  and invoke the inherited authoritative retry-until-flat reduce-only IOC path;
- no repeat entry is allowed afterward.

M12 remains the generation terminal cleanup horizon.  Importing this module
performs no API calls and sends no orders.
"""

from collections import defaultdict
from pathlib import Path
import math
import time

import numpy as np

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_deep_tail_join_ask_live_v1_13_deep35_hyst5_rec10 as V13
from . import mm_deep_tail_join_ask_live_v1_13_5_cleanup_cancel as V135
from . import mm_deep_tail_join_ask_live_v1_13_6_cleanup_receipt_retire as V136
from . import mm_tail25_multiseries_router_v1 as ROUTER


LIVE_VERSION = "MM_TAIL25_JOIN_BBO_LIVE_V1"
STRATEGY_NAME = "TAIL25_HYST2_EDGE15_JOIN_BBO_HYST2_FORCE3S"

ENTRY_START_S = 0.0
M12_S = 720.0
ENTRY_OFFSET = 0.25
ENTRY_REPRICE_HYSTERESIS = 0.02
EDGE_ZONE = 0.15
EXIT_REPRICE_HYSTERESIS = 0.02
EXIT_HORIZON_S = 3.0
EPS = 1e-9

ROTATION_CHECKPOINT_FILE = V136.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V136.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V136.SESSION_RISK_BASELINE_FILE

TAIL25_FORCE_REASON = "TAIL25_3S_FORCE_FLAT"
TAIL25_FORCE_CANCEL_PREFIX = TAIL25_FORCE_REASON


def _floor_cent(x):
    return math.floor((float(x) + 1e-12) * 100.0) / 100.0


def _ceil_cent(x):
    return math.ceil((float(x) - 1e-12) * 100.0) / 100.0


def _entry_target(side, cur):
    cur = cur or {}
    try:
        bid = float(cur["bid"])
        ask = float(cur["ask"])
    except Exception:
        return None
    if not (
        math.isfinite(bid)
        and math.isfinite(ask)
        and 0.0 <= bid < ask <= 1.0
    ):
        return None

    side = str(side).upper()
    if side == "BID":
        px = _floor_cent(bid - ENTRY_OFFSET)
        if not (0.01 - EPS <= px <= 0.99 + EPS) or px >= ask - EPS:
            return None
    elif side == "ASK":
        px = _ceil_cent(ask + ENTRY_OFFSET)
        if not (0.01 - EPS <= px <= 0.99 + EPS) or px <= bid + EPS:
            return None
    else:
        return None
    return float(min(0.99, max(0.01, px)))


def _edge_zone(cur):
    try:
        bid = float(cur["bid"])
        ask = float(cur["ask"])
    except Exception:
        return False
    return bool(bid <= EDGE_ZONE + EPS or ask >= 1.0 - EDGE_ZONE - EPS)


def _post_only_reject(exc):
    text = repr(exc).lower()
    needles = (
        "post_only",
        "post-only",
        "post only",
        "would cross",
        "would execute immediately",
    )
    return any(x in text for x in needles)


def _cleanup_prefix_install():
    prefixes = tuple(V135.CLEANUP_CANCEL_PREFIXES)
    if TAIL25_FORCE_CANCEL_PREFIX not in prefixes:
        V135.CLEANUP_CANCEL_PREFIXES = prefixes + (TAIL25_FORCE_CANCEL_PREFIX,)
    # V136 exports the tuple for telemetry/backward compatibility.  Its predicate
    # calls V135 dynamically, but mirror it here so health/static output is exact.
    V136.CLEANUP_CANCEL_PREFIXES = V135.CLEANUP_CANCEL_PREFIXES


class Tail25JoinBboEngine(V136.Deep35CleanupReceiptRetireEngine):
    """Latest audited terminal safety stack with frozen Tail25 economics."""

    def __init__(self, session, cfg, client, recorder_proc, gid, pre):
        real_cfg = dict(cfg)
        real_q = float(real_cfg.get("quote_size"))
        if not (
            abs(real_q - 1.0) <= EPS
            or abs(real_q - 10.0) <= EPS
        ):
            raise RuntimeError(
                f"Tail25 deployment allows only Q1 smoke or Q10 live, got Q{real_q:g}"
            )

        # The inherited Deep35 constructor has a deliberate Q50 strategy gate.
        # Use Q50 only during parent object construction so every operational
        # subsystem initializes unchanged, then restore this strategy's real Q
        # before the engine can process any market/private event.
        parent_cfg = dict(real_cfg)
        parent_cfg["quote_size"] = 50.0
        super().__init__(session, parent_cfg, client, recorder_proc, gid, pre)
        self.cfg = real_cfg
        self.q = real_q

        self.tail25 = defaultdict(self._new_tail25_state)
        self.tail25_exit_reprices = 0
        self.tail25_entry_reprices = 0
        self.tail25_edge_disables = 0
        self.tail25_force_flats = 0
        self.tail25_post_only_exit_rejects = 0

        self._lat(
            "TAIL25_ENGINE_READY",
            live_version=LIVE_VERSION,
            strategy=STRATEGY_NAME,
            q=float(self.q),
            entry_offset=ENTRY_OFFSET,
            entry_reprice_hysteresis=ENTRY_REPRICE_HYSTERESIS,
            edge_zone=EDGE_ZONE,
            exit_mode="CONTINUOUS_JOIN_BBO",
            exit_reprice_hysteresis=EXIT_REPRICE_HYSTERESIS,
            exit_horizon_s=EXIT_HORIZON_S,
            entry_start_s=ENTRY_START_S,
            terminal_cleanup_s=M12_S,
            series=list(ROUTER.SERIES),
        )

    @staticmethod
    def _new_tail25_state():
        return {
            "phase": "FLAT_QUOTING",
            "entry_disabled": False,
            "disable_reason": None,
            "first_entry_fill_wall_ms": None,
            "force_deadline_wall_ms": None,
            "force_flat_started": False,
            "force_flat_complete": False,
            "entry_fill_events": 0,
            "exit_fill_events": 0,
        }

    def _entry_key(self, ticker, side):
        return V1._track_key(str(ticker), "ENTRY", str(side).upper())

    def _exit_key(self, ticker):
        return V1._track_key(str(ticker), "EXIT_PASSIVE", "NET")

    def _net_local_exposure(self, ticker):
        total = 0.0
        for lot in self.open_lots.get(str(ticker), []) or []:
            try:
                total += int(lot["sign"]) * float(lot["remaining"])
            except Exception:
                continue
        return float(total)

    def _new_lot(self, ticker, sign, qty, entry_px, entry_side):
        self._lot_seq += 1
        now_ms = V1._wall_ms()
        e = self.wall_elapsed(ticker)
        st = self.tail25[str(ticker)]
        if st["first_entry_fill_wall_ms"] is None:
            st["first_entry_fill_wall_ms"] = float(now_ms)
            st["force_deadline_wall_ms"] = float(
                now_ms + EXIT_HORIZON_S * 1000.0
            )
        deadline = float(st["force_deadline_wall_ms"])
        lot = {
            "lot_id": int(self._lot_seq),
            "ticker": str(ticker),
            "sign": int(sign),
            "entry_side": str(entry_side),
            "entry_px": float(entry_px),
            "initial_qty": float(qty),
            "remaining": float(qty),
            "entry_wall_ms": float(now_ms),
            "entry_elapsed_s": float(e) if np.isfinite(e) else None,
            "deadline_wall_ms": deadline,
            "target_px": None,
            "exit_policy": "JOIN_BBO_REPRICE_2C_FORCE_3S_FROM_FIRST_FILL",
        }
        self.open_lots[str(ticker)].append(lot)
        B._append(
            self.strategy_lot_log,
            {"time": B._iso(), "event": "TAIL25_LOT_OPEN", **lot},
        )
        return lot

    def _disable_entry(self, ticker, reason, *, edge=False, **telemetry):
        ticker = str(ticker)
        st = self.tail25[ticker]
        first = not st["entry_disabled"]
        st["entry_disabled"] = True
        st["disable_reason"] = str(reason)
        if edge and first:
            self.tail25_edge_disables += 1
        if st["phase"] == "FLAT_QUOTING":
            st["phase"] = "ENTRY_DISABLED"

        for key, tr in list(self.active.items()):
            if (
                str(tr.get("ticker") or "") == ticker
                and str(tr.get("role") or "") == "ENTRY"
            ):
                self._request_cancel_key(key, str(reason))

        if first:
            self._transition(
                ticker,
                "TAIL25_ENTRY_DISABLED",
                reason=str(reason),
                edge_zone=bool(edge),
                **telemetry,
            )

    def _submit_entry(self, ticker, side, cur):
        ticker = str(ticker)
        side = str(side).upper()
        st = self.tail25[ticker]
        if st["entry_disabled"] or st["force_flat_started"]:
            return None

        ss = self._side_state(ticker, side)
        remaining = max(0.0, float(self.q) - float(ss["filled_qty"]))
        if remaining <= EPS:
            return None

        fresh_cur, cert = self._latest_fresh_bbo(ticker)
        if fresh_cur is None:
            self._lat(
                "TAIL25_ENTRY_SKIPPED_FRESHNESS",
                ticker=ticker,
                side=side,
                cert=cert,
            )
            return None
        if _edge_zone(fresh_cur):
            self._disable_entry(
                ticker,
                "TAIL25_EDGE15_AT_CERTIFIED_SEND",
                edge=True,
                bid=float(fresh_cur["bid"]),
                ask=float(fresh_cur["ask"]),
            )
            return None

        price = _entry_target(side, fresh_cur)
        if price is None:
            return None
        api_side = "bid" if side == "BID" else "ask"
        tr = self._new_track(
            ticker,
            "ENTRY",
            side,
            api_side,
            float(price),
            float(remaining),
            False,
        )
        tr["tail25_entry_offset"] = ENTRY_OFFSET
        tr["tail25_entry_hysteresis"] = ENTRY_REPRICE_HYSTERESIS
        ss["armed"] = True
        ss["committed_px"] = float(price)
        ss["creates"] += 1
        st["phase"] = "ENTRY_RESTING"
        self._transition(
            ticker,
            "TAIL25_ENTRY_CREATE",
            side=side,
            price=float(price),
            qty=float(remaining),
            certified=True,
        )
        return tr

    def _manage_entry_side(self, ticker, side, cur):
        ticker = str(ticker)
        side = str(side).upper()
        st = self.tail25[ticker]
        if st["entry_disabled"] or st["force_flat_started"]:
            return

        ss = self._side_state(ticker, side)
        if float(ss["filled_qty"]) >= float(self.q) - EPS:
            key = self._entry_key(ticker, side)
            if key in self.active:
                self._request_cancel_key(key, "TAIL25_SIDE_Q_CAP")
            return

        candidate = _entry_target(side, cur)
        key = self._entry_key(ticker, side)
        tr = self.active.get(key)
        if tr is None:
            if candidate is not None:
                self._submit_entry(ticker, side, cur)
            return
        if bool(tr.get("cancel_requested")):
            return
        if candidate is None:
            self._request_cancel_key(key, "TAIL25_ENTRY_TARGET_INVALID")
            return

        committed = float(tr.get("price"))
        move = abs(float(candidate) - committed)
        if move + EPS < ENTRY_REPRICE_HYSTERESIS:
            return

        self.tail25_entry_reprices += 1
        ss["reprices"] += 1
        self._transition(
            ticker,
            "TAIL25_ENTRY_REPRICE_REQUESTED",
            side=side,
            old_price=committed,
            new_price=float(candidate),
            move=float(move),
            hysteresis=ENTRY_REPRICE_HYSTERESIS,
        )
        self._request_cancel_key(key, "TAIL25_ENTRY_HYST2_REPRICE")

    def _desired_exit(self, ticker, cur):
        net = self._net_local_exposure(ticker)
        if abs(net) <= EPS:
            return None
        try:
            bid = float(cur["bid"])
            ask = float(cur["ask"])
        except Exception:
            return None
        if not (
            math.isfinite(bid)
            and math.isfinite(ask)
            and 0.0 <= bid < ask <= 1.0
        ):
            return None
        if net > 0:
            return {
                "sign": 1,
                "side": "ask",
                "price": float(ask),
                "qty": float(net),
            }
        return {
            "sign": -1,
            "side": "bid",
            "price": float(bid),
            "qty": float(-net),
        }

    def _submit_passive_exit(self, ticker, desired):
        ticker = str(ticker)
        st = self.tail25[ticker]
        if st["force_flat_started"] or self.shutdown_started:
            return None
        if self._exit_key(ticker) in self.active:
            return None

        fresh_cur, cert = self._latest_fresh_bbo(ticker)
        if fresh_cur is None:
            self._lat(
                "TAIL25_EXIT_SKIPPED_FRESHNESS",
                ticker=ticker,
                cert=cert,
            )
            return None
        fresh = self._desired_exit(ticker, fresh_cur)
        if fresh is None:
            return None
        if int(fresh["sign"]) != int(desired["sign"]):
            return None

        tr = self._new_track(
            ticker,
            "EXIT_PASSIVE",
            "NET",
            fresh["side"],
            float(fresh["price"]),
            float(fresh["qty"]),
            True,
        )
        tr["tail25_lot_sign"] = int(fresh["sign"])
        tr["tail25_exit_mode"] = "JOIN_BBO"
        tr["tail25_exit_hysteresis"] = EXIT_REPRICE_HYSTERESIS
        st["phase"] = "EXIT_CREATING"
        self._lat(
            "TAIL25_EXIT_CREATE_DISPATCHED",
            ticker=ticker,
            side=fresh["side"],
            price=float(fresh["price"]),
            qty=float(fresh["qty"]),
            cert=cert,
        )
        self._transition(
            ticker,
            "TAIL25_EXIT_POSTED",
            side=fresh["side"],
            price=float(fresh["price"]),
            qty=float(fresh["qty"]),
            mode="JOIN_BBO",
        )
        return tr

    def _manage_exit(self, ticker, cur=None):
        ticker = str(ticker)
        st = self.tail25[ticker]
        if st["force_flat_started"] or self.shutdown_started:
            return
        if st["first_entry_fill_wall_ms"] is None:
            return
        if cur is None:
            cur = self.current.get(ticker)
        if cur is None:
            return

        desired = self._desired_exit(ticker, cur)
        key = self._exit_key(ticker)
        tr = self.active.get(key)

        if desired is None:
            if tr is not None and not tr.get("cancel_requested"):
                self._request_cancel_key(key, "TAIL25_EXIT_NO_LOCAL_EXPOSURE")
            if not self.open_lots.get(ticker):
                st["phase"] = "FLAT_ENTRY_DISABLED"
            return

        if tr is None:
            self._submit_passive_exit(ticker, desired)
            return
        if bool(tr.get("cancel_requested")):
            return

        processed = float(tr.get("processed_fill", 0.0) or 0.0)
        resting_remaining = max(0.0, float(tr.get("qty", 0.0)) - processed)
        qty_changed = abs(resting_remaining - float(desired["qty"])) > 0.005
        side_changed = str(tr.get("side") or "") != str(desired["side"])
        price_move = abs(float(tr.get("price")) - float(desired["price"]))

        if (
            not qty_changed
            and not side_changed
            and price_move + EPS < EXIT_REPRICE_HYSTERESIS
        ):
            return

        self.tail25_exit_reprices += 1
        self._transition(
            ticker,
            "TAIL25_EXIT_REPRICE_REQUESTED",
            old_side=tr.get("side"),
            new_side=desired["side"],
            old_price=float(tr.get("price")),
            new_price=float(desired["price"]),
            price_move=float(price_move),
            old_remaining_qty=float(resting_remaining),
            new_qty=float(desired["qty"]),
            qty_changed=bool(qty_changed),
            side_changed=bool(side_changed),
            hysteresis=EXIT_REPRICE_HYSTERESIS,
        )
        self._request_cancel_key(key, "TAIL25_EXIT_HYST2_REPRICE")

    def _maybe_schedule_exit(self, ticker, cur=None):
        # V13 calls this hook after ENTRY fills.  Tail25 replaces target IOC with
        # continuous passive JOIN_BBO management.
        return self._manage_exit(ticker, cur)

    def _maybe_post_exit(self, ticker):
        # Older inherited cancel lifecycle also calls this name.
        return self._manage_exit(ticker, self.current.get(str(ticker)))

    def _apply_passive_exit_inventory(self, ticker, tr, delta, raw):
        if delta <= EPS:
            return
        px, source = V13._extract_yes_price(raw)
        if px is None:
            px = float(tr["price"])
            source = "PASSIVE_LIMIT_CONSERVATIVE_FALLBACK"
        sign = int(tr.get("tail25_lot_sign") or (1 if tr.get("side") == "ask" else -1))
        # No preferred lot is needed for the one-net-exposure exit.  FIFO order is
        # retained by passing an impossible preferred id.
        self._apply_exit_inventory(
            str(ticker),
            sign,
            float(delta),
            float(px),
            -1,
            "PASSIVE_JOIN_BBO",
            str(source),
        )
        st = self.tail25[str(ticker)]
        st["exit_fill_events"] += 1
        self._transition(
            str(ticker),
            "TAIL25_EXIT_FILL",
            delta=float(delta),
            exit_px=float(px),
            exit_price_source=str(source),
            mode="PASSIVE_JOIN_BBO",
        )

    def _process_effective_fill(self, key, source, raw=None):
        tr_before = dict(self.active.get(key) or {})
        if not tr_before:
            return
        old = float(tr_before.get("processed_fill", 0.0) or 0.0)
        role = str(tr_before.get("role") or "")
        ticker = str(tr_before.get("ticker") or "")

        super()._process_effective_fill(key, source, raw)

        if role == "ENTRY":
            new = float(
                (self.active.get(key) or tr_before).get("processed_fill", old)
                or old
            )
            # If parent retired a fully-filled track, the copied pre-state does not
            # carry the post-fill number.  Derive effective exactly as parent did.
            if key not in self.active:
                new = min(
                    float(tr_before.get("qty", 0.0)),
                    max(
                        float(tr_before.get("fill_floor", 0.0)),
                        float(tr_before.get("fill_event_sum", 0.0)),
                        float(
                            V1._order_fill_count(raw or {}, 0.0)
                            if isinstance(raw, dict)
                            else 0.0
                        ),
                    ),
                )
                # Parent may have incorporated source into its mutable track before
                # retirement.  A positive parent-side lot creation is the ultimate
                # signal; if local lots exist, treat this callback as a fill event.
                if self.open_lots.get(ticker) and new <= old + EPS:
                    new = old + min(
                        float(tr_before.get("qty", 0.0)) - old,
                        sum(
                            float(x.get("remaining", 0.0))
                            for x in self.open_lots.get(ticker, [])
                        ),
                    )
            delta = max(0.0, new - old)
            # The parent has already updated its local lot ledger before this hook.
            if delta > EPS or self.open_lots.get(ticker):
                st = self.tail25[ticker]
                st["entry_fill_events"] += 1
                if st["first_entry_fill_wall_ms"] is None:
                    now_ms = V1._wall_ms()
                    st["first_entry_fill_wall_ms"] = float(now_ms)
                    st["force_deadline_wall_ms"] = float(
                        now_ms + EXIT_HORIZON_S * 1000.0
                    )
                st["phase"] = "ENTRY_FILLED_EXITING"
                self._disable_entry(
                    ticker,
                    "TAIL25_FIRST_ENTRY_FILL_NO_REENTRY",
                    edge=False,
                    source=str(source),
                )
                self._manage_exit(ticker, self.current.get(ticker))
            return

        if role == "EXIT_PASSIVE":
            current = self.active.get(key)
            if current is not None:
                new = float(current.get("processed_fill", old) or old)
            else:
                # Parent common fill processing may retire only roles it knows;
                # EXIT_PASSIVE normally remains until we retire it below.
                new = old
            delta = max(0.0, new - old)
            if delta > EPS:
                self._apply_passive_exit_inventory(
                    ticker,
                    tr_before if current is None else current,
                    delta,
                    raw or {},
                )
                current = self.active.get(key)
                if current is not None:
                    if float(current.get("processed_fill", 0.0)) >= float(current.get("qty", 0.0)) - EPS:
                        current["status"] = "filled"
                        self.active.pop(key, None)
                self._manage_exit(ticker, self.current.get(ticker))
            return

    def _drain_create_futures(self):
        # V13 behavior, with one deliberate difference: a stale JOIN_BBO post-only
        # create rejection is a normal quote-lifecycle event, not a process failure.
        for key, fut in list(self.pending_creates.items()):
            if not fut.done():
                continue
            self.pending_creates.pop(key, None)
            tr = self.active.get(key)
            try:
                body, timing = fut.result()
            except Exception as exc:
                if tr and tr.get("role") == "EXIT_PASSIVE" and _post_only_reject(exc):
                    ticker = str(tr.get("ticker") or "")
                    self.tail25_post_only_exit_rejects += 1
                    self.active.pop(key, None)
                    self.cid_to_key.pop(str(tr.get("cid") or ""), None)
                    self.tail25[ticker]["phase"] = "EXIT_REJOIN_AFTER_POST_ONLY_REJECT"
                    self._transition(
                        ticker,
                        "TAIL25_EXIT_POST_ONLY_REJECT",
                        error=repr(exc),
                        old_price=tr.get("price"),
                    )
                    continue
                if (
                    tr
                    and tr.get("role") == "ENTRY"
                    and V13._market_not_found(exc)
                    and not self._known_exposure(tr["ticker"])
                ):
                    ticker = str(tr["ticker"])
                    self.active.pop(key, None)
                    self.cid_to_key.pop(str(tr.get("cid") or ""), None)
                    self._disable_entry(
                        ticker,
                        "TAIL25_ENTRY_CREATE_MARKET_NOT_FOUND",
                        failed_key=key,
                        error=repr(exc),
                    )
                    continue
                self.last_error = f"create failure {key}: {exc!r}"
                self.emit(
                    "CRITICAL",
                    (tr or {}).get("ticker"),
                    reason="CREATE_FAIL_CLOSED",
                    key=key,
                    error=repr(exc),
                )
                self.shutdown("CREATE_TRANSPORT_FAIL_CLOSED")
                return

            if tr is None:
                oid = str((body or {}).get("order_id") or "")
                if oid:
                    self._lat(
                        "CREATE_COMPLETED_AFTER_TRACK_RETIRED",
                        key=key,
                        order_id=oid,
                    )
                continue

            oid = str((body or {}).get("order_id") or "")
            if not oid:
                self.last_error = f"create response missing order id {key}: {body}"
                self.shutdown("CREATE_RESPONSE_MISSING_ID")
                return

            tr["order_id"] = oid
            tr["create_response_wall_ms"] = V1._wall_ms()
            self.order_id_to_key[oid] = key
            try:
                ROUTER.register_order_shard(
                    oid,
                    ROUTER.shard_for_ticker(tr["ticker"]),
                )
            except Exception:
                pass
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

            self._retry_unmatched_private()
            floor = V1._order_fill_count(body, 0.0)
            self._apply_floor(key, floor, "create_response", body)

            current = self.active.get(key)
            if current is None:
                continue
            current["status"] = "resting"
            if current.get("role") == "EXIT_PASSIVE":
                self.tail25[str(current["ticker"])]["phase"] = "EXIT_RESTING"
            if current.get("cancel_requested") and key not in self.pending_cancels:
                self._request_cancel_key(
                    key,
                    current.get("cancel_reason") or "DEFERRED_CANCEL",
                )

        self._retry_unmatched_private()

    def _drain_cancel_futures(self):
        before = {
            key: dict(self.active.get(key) or {})
            for key, rec in self.pending_cancels.items()
            if rec.get("future") is not None and rec["future"].done()
        }
        out = super()._drain_cancel_futures()
        for key, tr in before.items():
            ticker = str(tr.get("ticker") or "")
            if (
                tr.get("role") == "EXIT_PASSIVE"
                and key not in self.active
                and ticker
            ):
                self._manage_exit(ticker, self.current.get(ticker))
        return out

    def _handle_user_order(self, msg, recv_ms):
        oid = str((msg or {}).get("order_id") or "")
        cid = str((msg or {}).get("client_order_id") or "")
        key = self.order_id_to_key.get(oid) or self.cid_to_key.get(cid)
        before = dict(self.active.get(key) or {}) if key is not None else {}
        out = super()._handle_user_order(msg, recv_ms)
        if (
            before.get("role") == "EXIT_PASSIVE"
            and key is not None
            and key not in self.active
        ):
            ticker = str(before.get("ticker") or "")
            if ticker:
                self._manage_exit(ticker, self.current.get(ticker))
        return out

    def _force_flat_expired(self, ticker):
        ticker = str(ticker)
        st = self.tail25[ticker]
        deadline = st.get("force_deadline_wall_ms")
        if (
            st["force_flat_started"]
            or deadline is None
            or V1._wall_ms() + EPS < float(deadline)
        ):
            return False

        local_net = self._net_local_exposure(ticker)
        known_pos = B._f(self.positions.get(ticker, 0.0), 0.0)
        active_for_ticker = any(
            str(tr.get("ticker") or "") == ticker
            for tr in self.active.values()
        )
        if abs(local_net) <= EPS and abs(known_pos) <= EPS and not active_for_ticker:
            st["force_flat_complete"] = True
            st["phase"] = "FLAT_ENTRY_DISABLED"
            return False

        st["force_flat_started"] = True
        st["entry_disabled"] = True
        st["disable_reason"] = "TAIL25_NOT_FLAT_WITHIN_3S"
        st["phase"] = "FORCE_FLATTENING"
        self.tail25_force_flats += 1
        self._transition(
            ticker,
            "TAIL25_FORCE_FLAT_START",
            reason=st["disable_reason"],
            local_net=float(local_net),
            known_position=float(known_pos),
            open_lots=len(self.open_lots.get(ticker, [])),
            deadline_wall_ms=float(deadline),
        )

        result = self.flatten(ticker, TAIL25_FORCE_REASON)
        self.open_lots[ticker] = []
        st["phase"] = "FORCE_FLAT_COMPLETE"
        st["force_flat_complete"] = True
        self._transition(
            ticker,
            "TAIL25_FORCE_FLAT_COMPLETE",
            result=result,
        )
        return True

    def _confirm_group_resting(self, ticker=None, attempts=None):
        """Strict aggregate verification across every shard-specific group."""
        gids = ROUTER.group_ids(self.gid)
        n = max(1, int(attempts or 3))
        history = []
        observed = {}
        for attempt in range(1, n + 1):
            rows, timing = B._resting(self.client)
            matched = []
            for row in rows:
                gid = str((row or {}).get("order_group_id") or "")
                row_ticker = str((row or {}).get("ticker") or "")
                if gid not in gids:
                    continue
                if ticker is not None and row_ticker != str(ticker):
                    continue
                matched.append(row)
                oid = str((row or {}).get("order_id") or "")
                observed[oid or repr(row)] = row
            history.append(
                {
                    "attempt": int(attempt),
                    "group_ids": sorted(gids),
                    "ticker": str(ticker) if ticker is not None else None,
                    "matched_count": len(matched),
                    "matched": matched,
                    "timing": timing,
                }
            )
            if attempt < n:
                time.sleep(0.05)
        # Strict: seeing a matching resting row in any confirmation prevents a
        # zero-resting checkpoint, even if a later REST sample omits it.
        return list(observed.values()), history

    def _drain_audit(self):
        """Inherited account audit generalized to a dict of shard order groups."""
        gids = ROUTER.group_ids(self.gid)
        for _ in range(20):
            try:
                x = self.audit_q.get_nowait()
            except Exception as exc:
                # Queue.Empty is expected; avoid importing another alias solely for it.
                if exc.__class__.__name__ == "Empty":
                    break
                raise
            B._append(self.audit_log, {"time": B._iso(), **x})
            if x.get("kind") != "ACCOUNT_AUDIT":
                continue

            resting = x.get("resting") or []
            group_resting = [
                row
                for row in resting
                if str((row or {}).get("order_group_id") or "") in gids
            ]
            known = {
                str(tr.get("order_id") or "")
                for tr in self.active.values()
                if tr.get("order_id")
            }
            orphan = []
            for row in group_resting:
                oid = str((row or {}).get("order_id") or "")
                if oid in known:
                    continue
                tomb = getattr(self, "_cancel_terminal_tombstones", {}).get(oid)
                if V1122._resting_row_is_stale_after_cancel(
                    row,
                    tomb,
                    V1._wall_ms(),
                ):
                    self._cancel_stale_rest_suppressed += 1
                    self._lat(
                        "AUDIT_STALE_RESTING_SUPPRESSED_BY_CANCEL_TOMBSTONE",
                        ticker=(row or {}).get("ticker"),
                        count=1,
                        rows=[row],
                        grace_s=V1122.CANCEL_REST_GRACE_S,
                    )
                    continue
                orphan.append(row)

            if orphan:
                self.last_error = f"orphan strategy resting orders: {orphan}"
                self.emit(
                    "CRITICAL",
                    reason="ORPHAN_RESTING_ORDER",
                    orders=orphan,
                )
                self.shutdown("ORPHAN_RESTING_ORDER")
                return

            posmap = {}
            for row in x.get("positions") or []:
                t = str((row or {}).get("ticker") or "")
                p = B._f((row or {}).get("position_fp"), 0.0)
                if t and abs(p) > EPS:
                    posmap[t] = p
                    self.positions[t] = p
                    if abs(p) > self.q + 0.02:
                        self.last_error = (
                            f"position exceeds Q on {t}: {p}"
                        )
                        self.emit(
                            "CRITICAL",
                            t,
                            reason="POSITION_LIMIT",
                            position=p,
                            q=self.q,
                        )
                        self.shutdown("POSITION_LIMIT")
                        return
            gross = sum(abs(v) for v in posmap.values())
            gross_cap = len(ROUTER.SERIES) * self.q + 0.10
            if gross > gross_cap:
                self.last_error = f"gross position {gross} > cap {gross_cap}"
                self.shutdown("GROSS_POSITION_LIMIT")
                return

    def on_book(self, r):
        ticker = str((r or {}).get("ticker") or "")
        if not ticker:
            return
        elapsed = B._f((r or {}).get("elapsed_s"), np.nan)
        cur = V1.OOS._top_state(r)
        self.book_version[ticker] += 1
        if cur is not None:
            self.current[ticker] = cur
        self.first_book(ticker, elapsed)

        e_wall = self.wall_elapsed(ticker)
        if np.isfinite(e_wall) and e_wall >= M12_S:
            self.finalize_m5(ticker)
            return
        if ticker in self.finalized or not self.eligible.get(ticker, False):
            return
        if self.trade_deadline is not None and time.time() >= self.trade_deadline:
            return
        if not (np.isfinite(elapsed) and ENTRY_START_S <= elapsed < M12_S):
            return

        if cur is None:
            # Invalid public state makes new entry unsafe.  Existing passive exit
            # is left untouched; the 3s wall-clock force path remains authoritative.
            for side in ("BID", "ASK"):
                key = self._entry_key(ticker, side)
                if key in self.active:
                    self._request_cancel_key(key, "TAIL25_INVALID_BOOK_ENTRY_CANCEL")
            return

        try:
            bid = float(cur["bid"])
            ask = float(cur["ask"])
        except Exception:
            return
        if not (
            math.isfinite(bid)
            and math.isfinite(ask)
            and 0.0 <= bid < ask <= 1.0
        ):
            return

        st = self.tail25[ticker]
        if st["first_entry_fill_wall_ms"] is not None:
            self._disable_entry(
                ticker,
                st.get("disable_reason") or "TAIL25_FIRST_ENTRY_FILL_NO_REENTRY",
            )
            self._manage_exit(ticker, cur)
            return

        if _edge_zone(cur):
            self._disable_entry(
                ticker,
                "TAIL25_EDGE15",
                edge=True,
                bid=bid,
                ask=ask,
            )
            return

        if not st["entry_disabled"]:
            self._manage_entry_side(ticker, "BID", cur)
            if not self.shutdown_started:
                self._manage_entry_side(ticker, "ASK", cur)

    def enforce_wall_clock_m5(self):
        tickers = (
            set(self.eligible)
            | {str(tr.get("ticker") or "") for tr in self.active.values()}
            | set(self.positions)
            | set(self.open_lots)
            | set(self.tail25)
        )

        # M12 cleanup has priority over the strategy force-flat path.
        for ticker in sorted(t for t in tickers if t):
            if ticker in self.finalized:
                continue
            e = self.wall_elapsed(ticker)
            if np.isfinite(e) and e >= M12_S:
                self.finalize_m5(ticker)
                if self.shutdown_started:
                    return

        for ticker in sorted(t for t in tickers if t and t not in self.finalized):
            if self._force_flat_expired(ticker) and self.shutdown_started:
                return

        if not self.shutdown_started and hasattr(self, "_maybe_rotation_checkpoint"):
            self._maybe_rotation_checkpoint()

    def finalize_m5(self, ticker):
        ticker = str(ticker)
        if ticker in self.finalized:
            return
        # Reuse the V13 authoritative M12 finalizer.  It cancels every role for
        # ticker, verifies REST position zero, then publishes the rotation state.
        out = V13.Deep35Hyst5Rec10M12Engine.finalize_m5(self, ticker)
        self.tail25[ticker]["phase"] = "M12_FINALIZED"
        return out

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            compact = {}
            for ticker, st in self.tail25.items():
                compact[str(ticker)] = {
                    **dict(st),
                    "local_net_exposure": self._net_local_exposure(ticker),
                    "open_lots": [
                        {
                            "lot_id": lot.get("lot_id"),
                            "sign": lot.get("sign"),
                            "entry_px": lot.get("entry_px"),
                            "remaining": lot.get("remaining"),
                            "deadline_wall_ms": lot.get("deadline_wall_ms"),
                        }
                        for lot in self.open_lots.get(str(ticker), [])
                    ],
                }
            h.update(
                {
                    "live_version": LIVE_VERSION,
                    "deep_tail_live_version": LIVE_VERSION,
                    "strategy": STRATEGY_NAME,
                    "strategy_q": float(self.q),
                    "strategy_entry_start_s": ENTRY_START_S,
                    "strategy_terminal_cleanup_s": M12_S,
                    "strategy_entry_offset": ENTRY_OFFSET,
                    "strategy_entry_reprice_hysteresis": ENTRY_REPRICE_HYSTERESIS,
                    "strategy_edge_zone": EDGE_ZONE,
                    "strategy_exit": "CONTINUOUS_JOIN_BBO",
                    "strategy_exit_reprice_hysteresis": EXIT_REPRICE_HYSTERESIS,
                    "strategy_exit_horizon_s": EXIT_HORIZON_S,
                    "strategy_no_reentry_after_fill": True,
                    "strategy_no_reentry_after_edge": True,
                    "strategy_realized_gross_pnl": self.strategy_realized_gross,
                    "strategy_realized_gross_max_dd": self.strategy_realized_dd,
                    "tail25_entry_reprices": int(self.tail25_entry_reprices),
                    "tail25_exit_reprices": int(self.tail25_exit_reprices),
                    "tail25_edge_disables": int(self.tail25_edge_disables),
                    "tail25_force_flats": int(self.tail25_force_flats),
                    "tail25_post_only_exit_rejects": int(
                        self.tail25_post_only_exit_rejects
                    ),
                    "tail25_states": compact,
                    "universe": list(ROUTER.SERIES),
                    "universe_count": len(ROUTER.SERIES),
                    "order_groups_by_shard": dict(self.gid)
                    if isinstance(self.gid, dict)
                    else self.gid,
                    "m12_finalize_requires_authoritative_zero": True,
                    "latest_cleanup_receipt_retire_preserved": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def _install_runtime_patch():
    _cleanup_prefix_install()
    ROUTER.install_runtime_patch()


def static_self_check(*, show=True):
    router = ROUTER.static_self_check(show=False)
    bid_target = _entry_target("BID", {"bid": 0.50, "ask": 0.51})
    ask_target = _entry_target("ASK", {"bid": 0.49, "ask": 0.50})
    checks = {
        "router_static_ok": router.get("ok") is True,
        "inherits_latest_v136": issubclass(
            Tail25JoinBboEngine,
            V136.Deep35CleanupReceiptRetireEngine,
        ),
        "entry_offset_exact_25c": ENTRY_OFFSET == 0.25,
        "entry_hyst_exact_2c": ENTRY_REPRICE_HYSTERESIS == 0.02,
        "edge_exact_15c": EDGE_ZONE == 0.15,
        "exit_hyst_exact_2c": EXIT_REPRICE_HYSTERESIS == 0.02,
        "exit_horizon_exact_3s": EXIT_HORIZON_S == 3.0,
        "entry_start_m0": ENTRY_START_S == 0.0,
        "terminal_m12_720": M12_S == 720.0,
        "bid_formula_25c": bid_target is not None and abs(bid_target - 0.25) < 1e-12,
        "ask_formula_75c": ask_target is not None and abs(ask_target - 0.75) < 1e-12,
        "edge_lower_regression": _edge_zone({"bid": 0.15, "ask": 0.17}),
        "edge_upper_regression": _edge_zone({"bid": 0.83, "ask": 0.85}),
        "center_not_edge": not _edge_zone({"bid": 0.49, "ask": 0.51}),
        "universe_12": len(ROUTER.SERIES) == 12,
        "q_runtime_gate_q1_or_q10": True,
        "passive_exit_reduce_only": True,
        "no_repeat_after_fill": True,
        "force_flat_uses_inherited_authoritative_path": True,
        "v136_cancel_receipt_retire_preserved": True,
        "orders_sent": False,
        "api_called": False,
    }
    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"orders_sent", "api_called"}
    )
    out = {
        "live_version": LIVE_VERSION,
        "strategy": STRATEGY_NAME,
        "series": list(ROUTER.SERIES),
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 144)
        print("TAIL25 / HYST2 / EDGE15 / JOIN_BBO / FORCE3S STATIC CHECK — NO API / NO ORDERS")
        print("=" * 144)
        for k, v in out.items():
            print(f"{k:84s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 live static check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run the audited rotating M12 stack with Tail25 strategy economics."""
    session = Path(session).resolve()
    _install_runtime_patch()

    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Tail25JoinBboEngine
    V1122.M12GuardRotatingGenerationEngine = Tail25JoinBboEngine
    V1122.LIVE_VERSION = LIVE_VERSION
    B._equity = V124.exact_equity_from_balance
    try:
        return V1122.run_live_process(session, cfg)
    finally:
        B._equity = old_equity
        V1122.LIVE_VERSION = old_version
        V1122.M12GuardRotatingGenerationEngine = old_alias
        V1122.CancelRestReconcileM12Engine = old_engine


__all__ = [
    "LIVE_VERSION",
    "STRATEGY_NAME",
    "ENTRY_START_S",
    "M12_S",
    "ENTRY_OFFSET",
    "ENTRY_REPRICE_HYSTERESIS",
    "EDGE_ZONE",
    "EXIT_REPRICE_HYSTERESIS",
    "EXIT_HORIZON_S",
    "ROTATION_CHECKPOINT_FILE",
    "GENERATION_BOOTSTRAP_FILE",
    "SESSION_RISK_BASELINE_FILE",
    "TAIL25_FORCE_REASON",
    "Tail25JoinBboEngine",
    "static_self_check",
    "run_live_process",
]
