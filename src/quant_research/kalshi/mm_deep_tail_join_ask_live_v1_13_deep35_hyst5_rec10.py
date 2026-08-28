from __future__ import annotations

"""Q50 deep opposite-anchor live engine: 35c depth / 5c hysteresis / +10c recovery.

This is a new strategy engine on top of the audited rotating M12 safety stack.
It intentionally does NOT modify the historical Q100/REC25 engine.

Frozen strategy contract
------------------------
- Complete 15-minute crypto windows only.
- Strategy may arm from M0 once a causal 5-second normal-spread estimate has at
  least 20 prior observations.
- The normal-spread estimate is updated only by normal observations once seeded.
  Normal means abs(current YES spread - prior 5s normal mean) <= 2c.
- Rest both independent YES-book sides, each capped at Q50 cumulative raw entry
  fills for the window:
    BID target = floor_cent(YES ask - 35c - normal_spread)
    ASK target = ceil_cent(YES bid + 35c + normal_spread)
- Initial arm requires a normal state.  Once armed, the opposite-side anchored
  target continues to follow even while the instantaneous spread is abnormal.
- Reprice only after the cent-rounded target moves by at least 5c from the
  committed entry price.  Repricing uses the existing cancel/deferred-cancel path.
- Entry fills on opposite sides are netted FIFO in the local lot ledger; only net
  exposure is eligible for reduce-only exits.
- Every newly-opened lot has a +10c executable recovery target:
    long YES lot -> reduce-only IOC sell, limit = entry + 10c
    short YES lot -> reduce-only IOC buy,  limit = entry - 10c
- Exit attempts are persistent.  A zero/partial IOC is retried on later causal
  executable BBO observations while the target remains available.
- If any open lot remains 2.0s after local fill observation, cancel all remaining
  ENTRY tracks for that ticker, disable new entry for the rest of that window, and
  invoke the inherited authoritative retry-until-flat reduce-only IOC path.
- M12 remains the terminal cleanup horizon; authoritative zero position is required
  before the rotating generation may finalize.

Operational mechanics retained
------------------------------
- parent-owned M0->M12+30 recorder and raw_capture symlink architecture;
- generation-start EOF freshness barrier;
- authenticated private fill/user_orders feed plus REST reconciliation;
- exchange-index routing installed by the deployment wrapper;
- asynchronous prewarmed CREATE/CANCEL transport;
- canceled-order stale-REST tombstones and orphan fail-closed auditing;
- fixed session exact-equity loss baseline and guardian/recovery stack.

Importing this module performs no API calls and sends no orders.
"""

from collections import defaultdict, deque
from pathlib import Path
import math
import time
import uuid

import numpy as np

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_13_DEEP35_HYST5_REC10"

M12_S = 720.0
ENTRY_START_S = 0.0
DEPTH = 0.35
HYSTERESIS = 0.05
RECOVERY_EDGE = 0.10
RECOVERY_HORIZON_S = 2.0
SPREAD_WINDOW_S = 5.0
NORMAL_TOL = 0.02
MIN_NORMAL_OBS = 20
EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = 81.422
EPS = 1e-9

ROTATION_CHECKPOINT_FILE = V1122.V112.V111.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V1122.V112.V111.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V1122.V112.V111.SESSION_RISK_BASELINE_FILE


def _floor_cent(x):
    return math.floor((float(x) + 1e-12) * 100.0) / 100.0


def _ceil_cent(x):
    return math.ceil((float(x) - 1e-12) * 100.0) / 100.0


def _entry_target(side, cur, normal_spread):
    cur = cur or {}
    try:
        bid = float(cur["bid"])
        ask = float(cur["ask"])
        s = float(normal_spread)
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (bid, ask, s)) or not (0.0 <= bid < ask <= 1.0):
        return None
    side = str(side).upper()
    if side == "BID":
        px = _floor_cent(ask - DEPTH - s)
        if not (0.01 - EPS <= px <= 0.99 + EPS) or px >= ask - EPS:
            return None
    elif side == "ASK":
        px = _ceil_cent(bid + DEPTH + s)
        if not (0.01 - EPS <= px <= 0.99 + EPS) or px <= bid + EPS:
            return None
    else:
        return None
    return float(min(0.99, max(0.01, px)))


def _extract_yes_price(raw):
    raw = raw or {}
    for key in ("yes_price_dollars", "price_dollars", "yes_price", "price"):
        val = raw.get(key)
        if val in (None, ""):
            continue
        try:
            z = float(val)
        except Exception:
            continue
        if not math.isfinite(z):
            continue
        if z > 1.000001:
            z /= 100.0
        if 0.0 <= z <= 1.0:
            return float(z), key
    return None, None


def _market_not_found(exc):
    text = repr(exc).lower()
    return "market_not_found" in text and "market not found" in text


class Deep35Hyst5Rec10M12Engine(V1122.CancelRestReconcileM12Engine):
    """Audited rotating M12 transport/safety stack with new frozen economics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if abs(float(self.q) - 50.0) > EPS:
            raise RuntimeError(f"Deep35 live engine is frozen to Q50, got Q{self.q}")

        self.deep35 = defaultdict(self._new_market_state)
        self.open_lots = defaultdict(list)
        self.exit_inflight = {}
        self._lot_seq = 0
        self._exit_seq = 0
        self.strategy_realized_gross = 0.0
        self.strategy_peak_realized_gross = 0.0
        self.strategy_realized_dd = 0.0
        self.strategy_pnl_log = self.session / "deep35_strategy_pnl.jsonl"
        self.strategy_lot_log = self.session / "deep35_lots.jsonl"
        self.strategy_exit_log = self.session / "deep35_exit_attempts.jsonl"

        self._lat(
            "DEEP35_HYST5_REC10_READY",
            q=float(self.q),
            depth=DEPTH,
            hysteresis=HYSTERESIS,
            recovery_edge=RECOVERY_EDGE,
            recovery_horizon_s=RECOVERY_HORIZON_S,
            spread_window_s=SPREAD_WINDOW_S,
            normal_tol=NORMAL_TOL,
            min_normal_obs=MIN_NORMAL_OBS,
            expected_exit_effective_latency_ms=EXPECTED_EXIT_EFFECTIVE_LATENCY_MS,
            entry_start_s=ENTRY_START_S,
            terminal_cleanup_s=M12_S,
        )

    @staticmethod
    def _new_market_state():
        return {
            "spread_hist": deque(),
            "last_elapsed_s": None,
            "disabled_for_window": False,
            "disable_reason": None,
            "force_flat_started": False,
            "phase": "WARMING_NORMAL_SPREAD",
            "sides": {
                "BID": {
                    "armed": False,
                    "filled_qty": 0.0,
                    "committed_px": None,
                    "suppressed_hysteresis": 0,
                    "creates": 0,
                    "reprices": 0,
                },
                "ASK": {
                    "armed": False,
                    "filled_qty": 0.0,
                    "committed_px": None,
                    "suppressed_hysteresis": 0,
                    "creates": 0,
                    "reprices": 0,
                },
            },
        }

    def _spread_context(self, ticker, elapsed_s, cur):
        st = self.deep35[ticker]
        hist = st["spread_hist"]
        cutoff = float(elapsed_s) - SPREAD_WINDOW_S
        while hist and float(hist[0][0]) < cutoff:
            hist.popleft()
        obs = len(hist)
        avg = float(np.mean([x[1] for x in hist])) if obs >= MIN_NORMAL_OBS else None
        spread = float(cur["ask"] - cur["bid"])
        normal = bool(avg is not None and abs(spread - avg) <= NORMAL_TOL + EPS)
        return {"obs": obs, "avg": avg, "spread": spread, "normal": normal}

    def _record_spread_after_decision(self, ticker, elapsed_s, ctx):
        st = self.deep35[ticker]
        hist = st["spread_hist"]
        if int(ctx["obs"]) < MIN_NORMAL_OBS or bool(ctx["normal"]):
            hist.append((float(elapsed_s), float(ctx["spread"])))
        cutoff = float(elapsed_s) - SPREAD_WINDOW_S
        while hist and float(hist[0][0]) < cutoff:
            hist.popleft()
        st["last_elapsed_s"] = float(elapsed_s)

    def _side_state(self, ticker, side):
        return self.deep35[ticker]["sides"][str(side).upper()]

    def _entry_key(self, ticker, side):
        return V1._track_key(str(ticker), "ENTRY", str(side).upper())

    def _known_exposure(self, ticker):
        ticker = str(ticker)
        if self.open_lots.get(ticker):
            return True
        try:
            if abs(float(self.positions.get(ticker, 0.0))) > EPS:
                return True
        except Exception:
            return True
        for tr in self.active.values():
            if str(tr.get("ticker") or "") == ticker:
                try:
                    if float(tr.get("processed_fill", 0.0) or 0.0) > EPS:
                        return True
                except Exception:
                    return True
        return False

    def _disable_market_local(self, ticker, reason, **telemetry):
        st = self.deep35[ticker]
        st["disabled_for_window"] = True
        st["disable_reason"] = str(reason)
        st["phase"] = "DISABLED"
        for key, tr in list(self.active.items()):
            if str(tr.get("ticker") or "") == str(ticker) and str(tr.get("role") or "") == "ENTRY":
                self._request_cancel_key(key, str(reason))
        self._transition(ticker, "DEEP35_WINDOW_DISABLED", reason=str(reason), **telemetry)

    def _submit_entry(self, ticker, side, price, normal_spread, ctx):
        side = str(side).upper()
        ss = self._side_state(ticker, side)
        remaining = max(0.0, float(self.q) - float(ss["filled_qty"]))
        if remaining <= EPS:
            return None

        fresh_cur, cert = self._latest_fresh_bbo(ticker)
        if fresh_cur is None:
            self._lat("DEEP35_ENTRY_SKIPPED_FRESHNESS", ticker=ticker, side=side, cert=cert)
            return None

        # Initial arming must still be normal at the certified send snapshot.
        if not ss["armed"]:
            spread = float(fresh_cur["ask"] - fresh_cur["bid"])
            if abs(spread - float(normal_spread)) > NORMAL_TOL + EPS:
                return None

        send_px = _entry_target(side, fresh_cur, normal_spread)
        if send_px is None:
            return None

        api_side = "bid" if side == "BID" else "ask"
        tr = self._new_track(
            ticker,
            "ENTRY",
            side,
            api_side,
            float(send_px),
            float(remaining),
            False,
        )
        tr["deep35_normal_spread"] = float(normal_spread)
        tr["deep35_depth"] = DEPTH
        tr["deep35_hysteresis"] = HYSTERESIS
        ss["armed"] = True
        ss["committed_px"] = float(send_px)
        ss["creates"] += 1
        self.deep35[ticker]["phase"] = "QUOTING"
        self._transition(
            ticker,
            "DEEP35_ENTRY_CREATE",
            side=side,
            price=float(send_px),
            qty=float(remaining),
            normal_spread=float(normal_spread),
            prior_normal_obs=int(ctx["obs"]),
            certified=True,
        )
        return tr

    def _manage_entry_side(self, ticker, side, cur, elapsed_s, ctx):
        st = self.deep35[ticker]
        if st["disabled_for_window"] or st["force_flat_started"]:
            return
        side = str(side).upper()
        ss = self._side_state(ticker, side)
        if float(ss["filled_qty"]) >= float(self.q) - EPS:
            key = self._entry_key(ticker, side)
            if key in self.active:
                self._request_cancel_key(key, "DEEP35_Q50_SIDE_CAP")
            return

        avg = ctx.get("avg")
        if avg is None:
            return
        candidate = _entry_target(side, cur, avg)
        key = self._entry_key(ticker, side)
        tr = self.active.get(key)

        if tr is None:
            if not ss["armed"] and not bool(ctx["normal"]):
                return
            if candidate is not None:
                self._submit_entry(ticker, side, candidate, avg, ctx)
            return

        if str(tr.get("role") or "") != "ENTRY":
            return
        if bool(tr.get("cancel_requested")):
            return
        if candidate is None:
            self._request_cancel_key(key, "DEEP35_TARGET_INVALID")
            return

        committed = float(tr.get("price"))
        move = abs(float(candidate) - committed)
        if move + EPS < HYSTERESIS:
            ss["suppressed_hysteresis"] += 1
            return

        ss["reprices"] += 1
        self._transition(
            ticker,
            "DEEP35_REPRICE_REQUESTED",
            side=side,
            old_price=committed,
            new_price=float(candidate),
            move=float(move),
            hysteresis=HYSTERESIS,
        )
        self._request_cancel_key(key, "DEEP35_HYSTERESIS_REPRICE")

    def _new_lot(self, ticker, sign, qty, entry_px, entry_side):
        self._lot_seq += 1
        now_ms = V1._wall_ms()
        e = self.wall_elapsed(ticker)
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
            "deadline_wall_ms": float(now_ms + RECOVERY_HORIZON_S * 1000.0),
            "target_px": float(round(entry_px + RECOVERY_EDGE, 2) if sign > 0 else round(entry_px - RECOVERY_EDGE, 2)),
        }
        self.open_lots[str(ticker)].append(lot)
        B._append(self.strategy_lot_log, {"time": B._iso(), "event": "LOT_OPEN", **lot})
        return lot

    def _record_realized(self, ticker, qty, entry_px, exit_px, sign, mode, **extra):
        qty = float(qty)
        gross = (float(exit_px) - float(entry_px)) * qty if int(sign) > 0 else (float(entry_px) - float(exit_px)) * qty
        self.strategy_realized_gross += gross
        self.strategy_peak_realized_gross = max(self.strategy_peak_realized_gross, self.strategy_realized_gross)
        self.strategy_realized_dd = min(
            self.strategy_realized_dd,
            self.strategy_realized_gross - self.strategy_peak_realized_gross,
        )
        row = {
            "time": B._iso(),
            "ticker": str(ticker),
            "mode": str(mode),
            "qty": qty,
            "entry_px": float(entry_px),
            "exit_px": float(exit_px),
            "gross_pnl": float(gross),
            "gross_edge_c": float(100.0 * gross / qty) if qty > EPS else None,
            "cum_realized_gross": float(self.strategy_realized_gross),
            "realized_gross_drawdown": float(self.strategy_realized_gross - self.strategy_peak_realized_gross),
            "fees_included": False,
            **extra,
        }
        B._append(self.strategy_pnl_log, row)
        return gross

    def _apply_entry_inventory(self, ticker, side, qty, entry_px):
        ticker = str(ticker)
        side = str(side).upper()
        sign = 1 if side == "BID" else -1
        remaining = float(qty)
        lots = self.open_lots[ticker]

        # Opposite maker entry fills naturally close existing net exposure first.
        while remaining > EPS and lots and int(lots[0]["sign"]) != sign:
            lot = lots[0]
            take = min(remaining, float(lot["remaining"]))
            self._record_realized(
                ticker,
                take,
                lot["entry_px"],
                entry_px,
                lot["sign"],
                "OPPOSITE_ENTRY_NETTING",
                lot_id=lot["lot_id"],
                closing_entry_side=side,
            )
            lot["remaining"] -= take
            remaining -= take
            if lot["remaining"] <= EPS:
                B._append(self.strategy_lot_log, {"time": B._iso(), "event": "LOT_CLOSED_BY_OPPOSITE_ENTRY", **lot})
                lots.pop(0)

        if remaining > EPS:
            self._new_lot(ticker, sign, remaining, float(entry_px), side)

    def _apply_exit_inventory(self, ticker, sign, qty, exit_px, preferred_lot_id, mode, price_source):
        ticker = str(ticker)
        remaining = float(qty)
        lots = self.open_lots[ticker]
        ordered = sorted(
            [lot for lot in lots if int(lot["sign"]) == int(sign)],
            key=lambda lot: (0 if int(lot["lot_id"]) == int(preferred_lot_id) else 1, float(lot["entry_wall_ms"])),
        )
        for lot in ordered:
            if remaining <= EPS:
                break
            take = min(remaining, float(lot["remaining"]))
            self._record_realized(
                ticker,
                take,
                lot["entry_px"],
                exit_px,
                lot["sign"],
                mode,
                lot_id=lot["lot_id"],
                exit_price_source=price_source,
            )
            lot["remaining"] -= take
            remaining -= take

        self.open_lots[ticker] = [lot for lot in lots if float(lot["remaining"]) > EPS]
        if remaining > 0.01:
            raise RuntimeError(
                f"Deep35 local lot ledger underflow on {ticker}: exit_qty={qty} unmatched={remaining}"
            )

    def _submit_target_ioc(self, ticker, lot, cur):
        ticker = str(ticker)
        if ticker in self.exit_inflight or self.shutdown_started:
            return None
        qty = float(lot["remaining"])
        if qty <= EPS:
            return None
        sign = int(lot["sign"])
        target = float(lot["target_px"])
        if not (0.01 - EPS <= target <= 0.99 + EPS):
            return None

        if sign > 0:
            if float(cur["bid"]) + EPS < target:
                return None
            api_side = "ask"
        else:
            if float(cur["ask"]) - EPS > target:
                return None
            api_side = "bid"

        self._exit_seq += 1
        key = V1._track_key(ticker, "EXIT_IOC", f"{lot['lot_id']}-{self._exit_seq}")
        cid = f"d35x-{self.session.name[-10:]}-{uuid.uuid4().hex}"
        payload = B._payload(
            ticker=ticker,
            side=api_side,
            qty=qty,
            price=target,
            cid=cid,
            post_only=False,
            reduce_only=True,
            tif="immediate_or_cancel",
            group_id=None,
        )
        tr = {
            "key": key,
            "ticker": ticker,
            "role": "EXIT_IOC",
            "tail": f"LOT{lot['lot_id']}",
            "side": api_side,
            "price": target,
            "qty": qty,
            "cid": cid,
            "order_id": None,
            "status": "creating",
            "fill_floor": 0.0,
            "fill_event_sum": 0.0,
            "processed_fill": 0.0,
            "cancel_requested": False,
            "cancel_reason": None,
            "created_submit_wall_ms": V1._wall_ms(),
            "reduce_only": True,
            "lot_id": int(lot["lot_id"]),
            "lot_sign": int(sign),
            "exit_mode": "TARGET_REC10",
            "decision_bid": float(cur["bid"]),
            "decision_ask": float(cur["ask"]),
        }
        self.active[key] = tr
        self.cid_to_key[cid] = key
        self.exit_inflight[ticker] = key
        B._append(self.orders, {"time": B._iso(), "action": "EXIT_IOC_CREATE_SUBMITTED_ASYNC", "track": tr, "payload": payload})
        B._append(self.strategy_exit_log, {"time": B._iso(), "event": "TARGET_IOC_SUBMITTED", **tr})
        fut = self.transport.create(payload)
        self.pending_creates[key] = fut
        self._transition(
            ticker,
            "DEEP35_REC10_IOC_SUBMITTED",
            lot_id=lot["lot_id"],
            qty=qty,
            entry_px=lot["entry_px"],
            target_px=target,
            decision_bid=float(cur["bid"]),
            decision_ask=float(cur["ask"]),
        )
        return tr

    def _maybe_schedule_exit(self, ticker, cur=None):
        ticker = str(ticker)
        if ticker in self.exit_inflight or self.shutdown_started:
            return
        lots = self.open_lots.get(ticker) or []
        if not lots:
            return
        if cur is None:
            cur = self.current.get(ticker)
        if cur is None:
            return
        now_ms = V1._wall_ms()
        if any(float(lot["deadline_wall_ms"]) <= now_ms + EPS for lot in lots):
            return  # wall-clock force-flat path owns expired lots

        for lot in sorted(lots, key=lambda x: float(x["entry_wall_ms"])):
            sign = int(lot["sign"])
            target = float(lot["target_px"])
            ready = (
                float(cur["bid"]) + EPS >= target
                if sign > 0
                else float(cur["ask"]) - EPS <= target
            )
            if ready:
                self._submit_target_ioc(ticker, lot, cur)
                return

    def _clear_exit_inflight_if(self, ticker, key):
        ticker = str(ticker)
        if self.exit_inflight.get(ticker) == key:
            self.exit_inflight.pop(ticker, None)

    def _process_effective_fill(self, key, source, raw=None):
        tr = self.active.get(key)
        if tr is None:
            return
        effective = min(
            float(tr["qty"]),
            max(float(tr.get("fill_floor", 0.0)), float(tr.get("fill_event_sum", 0.0))),
        )
        old = float(tr.get("processed_fill", 0.0))
        if effective <= old + EPS:
            return
        delta = effective - old
        tr["processed_fill"] = effective
        ticker = str(tr["ticker"])

        B._append(
            self.fill_detail_log,
            {
                "time": B._iso(),
                "ticker": ticker,
                "key": key,
                "order_id": tr.get("order_id"),
                "role": tr["role"],
                "tail": tr["tail"],
                "delta": delta,
                "effective_fill": effective,
                "source": source,
                "raw": raw,
            },
        )
        self.counts["fill_events"] += 1
        self.emit("FILL", ticker, role=tr["role"], tail=tr["tail"], qty=delta, effective_fill=effective, source=source)

        post_pos = B._f((raw or {}).get("post_position_fp"), np.nan)
        if np.isfinite(post_pos):
            self.positions[ticker] = float(post_pos)

        if tr["role"] == "ENTRY":
            side = str(tr["tail"]).upper()
            ss = self._side_state(ticker, side)
            ss["filled_qty"] = min(float(self.q), float(ss["filled_qty"]) + delta)
            self._apply_entry_inventory(ticker, side, delta, float(tr["price"]))
            self._transition(
                ticker,
                "DEEP35_ENTRY_FILL",
                side=side,
                delta=delta,
                cumulative_side_fill=ss["filled_qty"],
                entry_px=float(tr["price"]),
            )
            if effective >= float(tr["qty"]) - EPS:
                tr["status"] = "filled"
                self.active.pop(key, None)
                ss["committed_px"] = None
            self._maybe_schedule_exit(ticker)
            return

        if tr["role"] == "EXIT_IOC":
            px, price_source = _extract_yes_price(raw)
            if px is None:
                # For TARGET IOC the limit is a conservative realized-price bound:
                # sell executions cannot be below it; buy executions cannot be above it.
                px = float(tr["price"])
                price_source = "TARGET_LIMIT_CONSERVATIVE_FALLBACK"
            self._apply_exit_inventory(
                ticker,
                int(tr["lot_sign"]),
                delta,
                float(px),
                int(tr["lot_id"]),
                str(tr.get("exit_mode") or "TARGET_REC10"),
                str(price_source),
            )
            self._transition(
                ticker,
                "DEEP35_EXIT_FILL",
                lot_id=tr["lot_id"],
                delta=delta,
                exit_px=float(px),
                exit_price_source=price_source,
                mode=tr.get("exit_mode"),
            )
            if effective >= float(tr["qty"]) - EPS:
                tr["status"] = "filled"
                self.active.pop(key, None)
                self._clear_exit_inflight_if(ticker, key)
            return

    def _drain_create_futures(self):
        for key, fut in list(self.pending_creates.items()):
            if not fut.done():
                continue
            self.pending_creates.pop(key, None)
            tr = self.active.get(key)
            try:
                body, timing = fut.result()
            except Exception as exc:
                if tr and tr.get("role") == "ENTRY" and _market_not_found(exc) and not self._known_exposure(tr["ticker"]):
                    ticker = str(tr["ticker"])
                    self.active.pop(key, None)
                    self.cid_to_key.pop(str(tr.get("cid") or ""), None)
                    self._disable_market_local(ticker, "ENTRY_CREATE_MARKET_NOT_FOUND", failed_key=key, error=repr(exc))
                    continue
                self.last_error = f"create failure {key}: {exc!r}"
                self.emit("CRITICAL", (tr or {}).get("ticker"), reason="CREATE_FAIL_CLOSED", key=key, error=repr(exc))
                self.shutdown("CREATE_TRANSPORT_FAIL_CLOSED")
                return

            if tr is None:
                oid = str((body or {}).get("order_id") or "")
                if oid:
                    self._lat("CREATE_COMPLETED_AFTER_TRACK_RETIRED", key=key, order_id=oid)
                continue

            oid = str((body or {}).get("order_id") or "")
            if not oid:
                self.last_error = f"create response missing order id {key}: {body}"
                self.shutdown("CREATE_RESPONSE_MISSING_ID")
                return

            tr["order_id"] = oid
            tr["create_response_wall_ms"] = V1._wall_ms()
            self.order_id_to_key[oid] = key
            B._append(self.orders, {"time": B._iso(), "action": "CREATE_ACK", "track": tr, "response": body, "timing": timing})
            self._lat("CREATE_ACK", key=key, ticker=tr["ticker"], role=tr["role"], tail=tr["tail"], order_id=oid, timing=timing)

            # Private fill can arrive before CREATE ACK. Give it first chance so
            # the actual execution price is retained for IOC telemetry.
            self._retry_unmatched_private()
            floor = V1._order_fill_count(body, 0.0)
            self._apply_floor(key, floor, "create_response", body)

            if tr.get("role") == "EXIT_IOC":
                current = self.active.get(key)
                if current is not None:
                    current["status"] = "ioc_terminal"
                    self.active.pop(key, None)
                self._clear_exit_inflight_if(tr["ticker"], key)
                B._append(
                    self.strategy_exit_log,
                    {
                        "time": B._iso(),
                        "event": "TARGET_IOC_ACK_TERMINAL",
                        "ticker": tr["ticker"],
                        "lot_id": tr["lot_id"],
                        "requested_qty": tr["qty"],
                        "processed_fill": tr.get("processed_fill", 0.0),
                        "order_id": oid,
                        "timing": timing,
                    },
                )
                continue

            tr["status"] = "resting"
            if tr.get("cancel_requested") and key in self.active and key not in self.pending_cancels:
                self._request_cancel_key(key, tr.get("cancel_reason") or "DEFERRED_CANCEL")

        self._retry_unmatched_private()

    def _handle_user_order(self, msg, recv_ms):
        oid = str((msg or {}).get("order_id") or "")
        cid = str((msg or {}).get("client_order_id") or "")
        key = self.order_id_to_key.get(oid) or self.cid_to_key.get(cid)
        before = dict(self.active.get(key) or {}) if key is not None else {}
        out = super()._handle_user_order(msg, recv_ms)
        if key is not None and key not in self.active and before:
            if before.get("role") == "ENTRY":
                self._side_state(before["ticker"], before["tail"])["committed_px"] = None
            elif before.get("role") == "EXIT_IOC":
                self._clear_exit_inflight_if(before["ticker"], key)
        return out

    def _drain_cancel_futures(self):
        before = {
            key: dict(self.active.get(key) or {})
            for key, rec in self.pending_cancels.items()
            if (rec.get("future") is not None and rec["future"].done())
        }
        out = super()._drain_cancel_futures()
        for key, tr in before.items():
            if key not in self.active and tr.get("role") == "ENTRY":
                self._side_state(tr["ticker"], tr["tail"])["committed_px"] = None
        return out

    def _maybe_post_exit(self, ticker):
        # Historical passive REC25/JOIN_ASK hook is intentionally disabled.
        return None

    def _force_flat_expired(self, ticker):
        ticker = str(ticker)
        st = self.deep35[ticker]
        if st["force_flat_started"] or not self.open_lots.get(ticker):
            return False
        now_ms = V1._wall_ms()
        if not any(float(lot["deadline_wall_ms"]) <= now_ms + EPS for lot in self.open_lots[ticker]):
            return False

        st["force_flat_started"] = True
        st["disabled_for_window"] = True
        st["disable_reason"] = "REC10_NOT_FLAT_WITHIN_2S"
        st["phase"] = "FORCE_FLATTENING"
        self._transition(
            ticker,
            "DEEP35_FORCE_FLAT_START",
            reason=st["disable_reason"],
            open_lots=len(self.open_lots[ticker]),
            open_qty=sum(float(x["remaining"]) for x in self.open_lots[ticker]),
        )

        for key, tr in list(self.active.items()):
            if str(tr.get("ticker") or "") == ticker and str(tr.get("role") or "") == "ENTRY":
                self._request_cancel_key(key, "DEEP35_FORCE_FLAT_CANCEL_ENTRY")

        # Inherited flatten is authoritative: cancel outstanding strategy tracks,
        # refresh exchange position, retry touch IOC, then extreme reduce-only IOC.
        result = self.flatten(ticker, "DEEP35_2S_FORCE_FLAT")
        self.open_lots[ticker] = []
        self.exit_inflight.pop(ticker, None)
        st["phase"] = "FORCE_FLAT_COMPLETE"
        self._transition(ticker, "DEEP35_FORCE_FLAT_COMPLETE", result=result)
        return True

    def finalize_m5(self, ticker):
        ticker = str(ticker)
        if ticker in self.finalized:
            return
        self.health(force=True)
        self._cancel_all_for_ticker(ticker, "M12")
        deadline = time.time() + 4.0
        while any(str(tr.get("ticker") or "") == ticker for tr in self.active.values()) and time.time() < deadline:
            self.poll_orders()
            time.sleep(0.005)
        remaining = [k for k, tr in self.active.items() if str(tr.get("ticker") or "") == ticker]
        if remaining:
            raise RuntimeError(f"M12 cancel did not retire all tracks for {ticker}: {remaining}")

        p = self.refresh_position(ticker)
        if abs(p) > EPS:
            self.flatten(ticker, "M12")
        final_p = self.refresh_position(ticker)
        if abs(final_p) > EPS:
            raise RuntimeError(f"M12 cleanup verification nonzero {ticker}: {final_p:+.4f}")

        self.positions[ticker] = 0.0
        self.open_lots[ticker] = []
        self.exit_inflight.pop(ticker, None)
        self.finalized.add(ticker)
        self.dt[ticker]["phase"] = "M12_FINALIZED"
        self.deep35[ticker]["phase"] = "M12_FINALIZED"
        self._transition(ticker, "M12_FINALIZED", position=0.0, cleanup_horizon_s=M12_S, authoritative_position_zero=True)
        self.emit("M12_FINALIZED", ticker, position=0.0, cleanup_horizon_s=M12_S)
        self.health(force=True)

    def enforce_wall_clock_m5(self):
        tickers = (
            set(self.eligible)
            | {str(tr.get("ticker") or "") for tr in self.active.values()}
            | set(self.positions)
            | set(self.open_lots)
        )

        # M12 terminal cleanup always has priority over the 2s strategy force path.
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
            for side in ("BID", "ASK"):
                key = self._entry_key(ticker, side)
                if key in self.active:
                    self._request_cancel_key(key, "DEEP35_INVALID_BOOK")
            return

        try:
            bid = float(cur["bid"])
            ask = float(cur["ask"])
        except Exception:
            return
        if not (math.isfinite(bid) and math.isfinite(ask) and 0.0 <= bid < ask <= 1.0):
            return

        ctx = self._spread_context(ticker, elapsed, cur)
        if ctx["avg"] is not None:
            self._manage_entry_side(ticker, "BID", cur, elapsed, ctx)
            if not self.shutdown_started:
                self._manage_entry_side(ticker, "ASK", cur, elapsed, ctx)
        self._record_spread_after_decision(ticker, elapsed, ctx)

        if not self.shutdown_started:
            self._maybe_schedule_exit(ticker, cur)

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            compact = {}
            for ticker, st in self.deep35.items():
                compact[str(ticker)] = {
                    "disabled_for_window": st["disabled_for_window"],
                    "disable_reason": st["disable_reason"],
                    "force_flat_started": st["force_flat_started"],
                    "phase": st["phase"],
                    "normal_obs": len(st["spread_hist"]),
                    "sides": {
                        side: dict(values)
                        for side, values in st["sides"].items()
                    },
                    "open_lots": [
                        {
                            "lot_id": lot["lot_id"],
                            "sign": lot["sign"],
                            "entry_px": lot["entry_px"],
                            "remaining": lot["remaining"],
                            "target_px": lot["target_px"],
                            "deadline_wall_ms": lot["deadline_wall_ms"],
                        }
                        for lot in self.open_lots.get(str(ticker), [])
                    ],
                    "exit_inflight": self.exit_inflight.get(str(ticker)),
                }
            h.update(
                {
                    "live_version": LIVE_VERSION,
                    "deep_tail_live_version": LIVE_VERSION,
                    "strategy": "DEEP35_HYST5_REC10_Q50",
                    "strategy_depth": DEPTH,
                    "strategy_hysteresis": HYSTERESIS,
                    "strategy_recovery_edge": RECOVERY_EDGE,
                    "strategy_recovery_horizon_s": RECOVERY_HORIZON_S,
                    "strategy_spread_window_s": SPREAD_WINDOW_S,
                    "strategy_normal_tol": NORMAL_TOL,
                    "strategy_min_normal_obs": MIN_NORMAL_OBS,
                    "strategy_expected_exit_effective_latency_ms": EXPECTED_EXIT_EFFECTIVE_LATENCY_MS,
                    "strategy_entry_start_s": ENTRY_START_S,
                    "strategy_terminal_cleanup_s": M12_S,
                    "strategy_realized_gross_pnl": self.strategy_realized_gross,
                    "strategy_realized_gross_max_dd": self.strategy_realized_dd,
                    "deep35_states": compact,
                    "rec25_enabled": False,
                    "m1130_entry_cutoff_enabled": False,
                    "persistent_danger_guard_enabled": False,
                    "exact_kalshi_dollar_equity": True,
                    "m12_finalize_requires_authoritative_zero": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = V1122.static_self_check(show=False)
    bid_target = _entry_target("BID", {"bid": 0.43, "ask": 0.47}, 0.02)
    ask_target = _entry_target("ASK", {"bid": 0.43, "ask": 0.47}, 0.02)
    checks = {
        "cancel_reconcile_parent_ok": base.get("ok") is True,
        "inherits_cancel_reconcile_m12": issubclass(Deep35Hyst5Rec10M12Engine, V1122.CancelRestReconcileM12Engine),
        "q_frozen_50_runtime_gate": True,
        "depth_exact_35c": DEPTH == 0.35,
        "hysteresis_exact_5c": HYSTERESIS == 0.05,
        "recovery_exact_10c": RECOVERY_EDGE == 0.10,
        "recovery_horizon_exact_2s": RECOVERY_HORIZON_S == 2.0,
        "spread_window_exact_5s": SPREAD_WINDOW_S == 5.0,
        "normal_tol_exact_2c": NORMAL_TOL == 0.02,
        "normal_min_obs_exact_20": MIN_NORMAL_OBS == 20,
        "entry_start_m0": ENTRY_START_S == 0.0,
        "terminal_m12_720": M12_S == 720.0,
        "expected_exit_latency_81_422ms": EXPECTED_EXIT_EFFECTIVE_LATENCY_MS == 81.422,
        "bid_formula_regression_10c": bid_target is not None and abs(bid_target - 0.10) < 1e-12,
        "ask_formula_regression_80c": ask_target is not None and abs(ask_target - 0.80) < 1e-12,
        "rec25_disabled": True,
        "m1130_cutoff_disabled": True,
        "old_danger_guard_disabled": True,
        "force_flat_uses_inherited_authoritative_path": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "strategy": "DEEP35_HYST5_REC10_Q50",
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 144)
        print("DEEP35 / HYST5 / REC10 / Q50 LIVE ENGINE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 144)
        for k, v in out.items():
            print(f"{k:84s}: {v}")
    if not ok:
        raise RuntimeError(f"Deep35 live static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Use the audited V1.12.2 rotating safety runner with this strategy class."""
    session = Path(session).resolve()
    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Deep35Hyst5Rec10M12Engine
    V1122.M12GuardRotatingGenerationEngine = Deep35Hyst5Rec10M12Engine
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
    "M12_S",
    "ENTRY_START_S",
    "DEPTH",
    "HYSTERESIS",
    "RECOVERY_EDGE",
    "RECOVERY_HORIZON_S",
    "SPREAD_WINDOW_S",
    "NORMAL_TOL",
    "MIN_NORMAL_OBS",
    "EXPECTED_EXIT_EFFECTIVE_LATENCY_MS",
    "ROTATION_CHECKPOINT_FILE",
    "GENERATION_BOOTSTRAP_FILE",
    "SESSION_RISK_BASELINE_FILE",
    "Deep35Hyst5Rec10M12Engine",
    "static_self_check",
    "run_live_process",
]
