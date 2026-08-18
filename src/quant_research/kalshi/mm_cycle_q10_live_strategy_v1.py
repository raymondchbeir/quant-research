from __future__ import annotations

"""Real-money validation runner for the frozen Candidate-C / CYCLE_ALWAYS_EXIT strategy.

REAL ORDERS ARE SENT ONLY BY THE EXPLICITLY ARMED start helpers.

Frozen behavior
---------------
Universe: the same nine 15-minute crypto series as the frozen OOS study.
Window: M0 <= elapsed < M5.
Flat: Candidate C entry = L3-supported side + natural YES spread >= 2c;
      join the public BBO with a post-only GTC order.
After first fill: cancel the residual entry, stop adding inventory, and quote only
                  the opposite BBO until flat.  Exit qty = actual abs(position).
M5: cancel passive order and flatten residual with reduce-only IOC.

Risk / audit
------------
- Full run: Q10 per market, 24 hours from the first complete M0 window.
- Smoke: Q1 per market for exactly one complete M0-M5 strategy window.
- Clean account required at startup: no open positions and no resting orders.
- Start-to-current account-equity loss kill defaults to $50.  Equity is Kalshi
  available balance + current portfolio value.
- All resting strategy orders are attached to a Kalshi order group.  Shutdown
  triggers the group before flattening.
- All resting orders use cancel_order_on_pause=True.
- Cleanup orders are reduce-only IOC and do not use the triggered group.
- Each run records raw V5 public market data plus decisions, orders, fills,
  queue positions, account equity, positions, risk actions and a final summary.

A software loss trigger cannot guarantee the final loss is exactly the configured
amount: network delay, exchange pauses, market moves and liquidation slippage can
cause overshoot.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_event_time_m0_m5_recorder_v5 as V5
from . import mm_event_time_m0_m15_exploratory_recorder_v1 as FULL15
from . import mm_live_q1_queue_probe_v1 as Q1

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V1"
SERIES = tuple(OOS.SERIES)
SPREAD_FLOOR_C = float(OOS.SPREAD_FLOOR_C)
EPS = 1e-9

FULL_Q = 10.0
SMOKE_Q = 1.0
FULL_HOURS = 24.0
LOSS_LIMIT_USD = 50.0
FULL_MIN_EQUITY = 125.0
SMOKE_MIN_EQUITY = 10.0
FULL_ARM = "LIVE_Q10_24H"
SMOKE_ARM = "Q1_ONE_WINDOW"
KILL_ARM = "KILL_AND_FLATTEN"
CLIENT_PREFIX = "mmq10v1-"
GROUP_LIMIT_FP = "1000.00"

FULL_WINDOW_GRACE_S = 5.0
ORDER_POLL_S = 0.20
POSITION_POLL_S = 1.00
BALANCE_POLL_S = 0.50
QUEUE_POLL_S = 2.00
HEALTH_S = 1.00
LOOP_SLEEP_S = 0.025
RECORDER_START_TIMEOUT_S = 60.0
RECORDER_STOP_TIMEOUT_S = 45.0

ROOT = C.DATA_ROOT / "live_cycle_q10_v1"
CONTROL_PATH = ROOT / "active_live.json"
ROOT.mkdir(parents=True, exist_ok=True)


def _iso():
    return pd.Timestamp.now(tz="UTC").isoformat()


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _append(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1, encoding="utf-8") as fh:
        fh.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")


def _ctl():
    return _read(CONTROL_PATH, {}) or {}


def _pid_alive(pid):
    return OOS._pid_alive(pid)


def _equity(body):
    # Current Kalshi balance endpoint reports both fields in cents.
    cash = _f((body or {}).get("balance"), 0.0) / 100.0
    portfolio = _f((body or {}).get("portfolio_value"), 0.0) / 100.0
    return {
        "cash_balance_usd": cash,
        "portfolio_value_usd": portfolio,
        "equity_usd": cash + portfolio,
        "updated_ts": (body or {}).get("updated_ts"),
    }


def _positions(client, ticker=None):
    params = {"count_filter": "position", "limit": 1000, "subaccount": 0}
    if ticker:
        params["ticker"] = str(ticker)
    body, timing = client.get("/portfolio/positions", params=params)
    return body.get("market_positions") or [], timing


def _position(client, ticker):
    rows, timing = _positions(client, ticker=ticker)
    for r in rows:
        if str(r.get("ticker") or "") == str(ticker):
            return _f(r.get("position_fp"), 0.0), r, timing
    return 0.0, None, timing


def _resting(client):
    body, timing = client.get(
        "/portfolio/orders",
        params={"status": "resting", "limit": 1000, "subaccount": 0},
    )
    return body.get("orders") or [], timing


def _balance(client):
    body, timing = client.get("/portfolio/balance", params={"subaccount": 0})
    return _equity(body), body, timing


def _other_recorders_running():
    out = []
    try:
        c = OOS._control()
        if c and _pid_alive(c.get("recorder_pid")):
            out.append("frozen OOS recorder")
    except Exception:
        pass
    try:
        c = FULL15._read_json(FULL15.CONTROL_PATH, {}) or {}
        if c and FULL15._pid_alive(c.get("pid")):
            out.append("M0-M15 exploratory recorder")
    except Exception:
        pass
    try:
        c = V5._read_json(V5.CONTROL_PATH, {}) or {}
        if c and V5._pid_alive(c.get("pid")):
            out.append("standalone V5 recorder")
    except Exception:
        pass
    return out


def _preflight(*, quote_size, runtime_hours, max_loss_usd, min_equity_usd,
               mode, save_dir=None, show=True):
    if tuple(V5.CRYPTO_SERIES) != SERIES:
        raise RuntimeError("Frozen universe mismatch between live strategy and V5 recorder.")
    running = _other_recorders_running()
    if running:
        raise RuntimeError(
            "Stop/finalize research recording before live trading. Running: " + ", ".join(running)
        )

    client = Q1.LiveClient()
    limits, lt = client.get("/account/limits")
    fees = OOS.fee_preflight(horizon_hours=OOS.FEE_CHANGE_HORIZON_H, show=False)
    eq, raw_balance, bt = _balance(client)
    pos, pt = _positions(client)
    resting, ot = _resting(client)

    nonzero = [r for r in pos if abs(_f(r.get("position_fp"), 0.0)) > EPS]
    if nonzero:
        raise RuntimeError("Live validation requires a flat account: " + json.dumps(nonzero, default=str))
    if resting:
        raise RuntimeError("Live validation requires zero resting orders: " + json.dumps(resting, default=str))
    if eq["equity_usd"] + EPS < float(min_equity_usd):
        raise RuntimeError(
            f"Starting equity ${eq['equity_usd']:.2f} < required ${float(min_equity_usd):.2f}."
        )
    if not (0 < float(max_loss_usd) < eq["equity_usd"]):
        raise RuntimeError("Loss limit must be positive and smaller than starting equity.")

    rr = _f(((limits or {}).get("read") or {}).get("refill_rate"))
    wr = _f(((limits or {}).get("write") or {}).get("refill_rate"))
    if np.isfinite(rr) and rr < 180:
        raise RuntimeError(f"Read API refill {rr} tokens/s is below the 180 token/s live-run guard.")
    if np.isfinite(wr) and wr < 100:
        raise RuntimeError(f"Write API refill {wr} tokens/s is below the 100 token/s live-run guard.")

    report = {
        "time": _iso(), "ok": True, "mode": mode, "quote_size": float(quote_size),
        "runtime_hours": float(runtime_hours), "max_loss_usd": float(max_loss_usd),
        "min_equity_usd": float(min_equity_usd), "account": eq,
        "kill_equity_usd": eq["equity_usd"] - float(max_loss_usd),
        "api_limits": limits, "fee_preflight": fees,
        "positions": pos, "resting_orders": resting,
        "timing": {"limits": lt, "balance": bt, "positions": pt, "orders": ot},
    }
    if save_dir:
        d = Path(save_dir)
        _atomic(d / "preflight.json", report)
        _atomic(d / "starting_account.json", {"raw": raw_balance, **eq})
        _atomic(d / "fee_preflight.json", fees)
    if show:
        print("LIVE PREFLIGHT: PASS")
        print(f"  mode:             {mode}")
        print(f"  quote size:       Q{float(quote_size):g} per market")
        print(f"  account equity:   ${eq['equity_usd']:.2f}")
        print(f"  loss kill equity: ${report['kill_equity_usd']:.2f}")
        print(f"  API tier:         {(limits or {}).get('usage_tier')}")
        print(f"  read/write:       {rr}/{wr} tokens/s")
        print("  positions:        flat")
        print("  resting orders:   none")
        print("  fee preflight:    PASS")
        print("  ORDERS SENT:      NO")
    return report


def live_preflight(*, quote_size=FULL_Q, runtime_hours=FULL_HOURS,
                   max_start_loss_usd=LOSS_LIMIT_USD, min_start_equity_usd=None,
                   show=True):
    """Read-only.  Never sends/cancels an order."""
    q = float(quote_size)
    minimum = min_start_equity_usd
    if minimum is None:
        minimum = FULL_MIN_EQUITY if q > 1 else SMOKE_MIN_EQUITY
    return _preflight(
        quote_size=q, runtime_hours=float(runtime_hours),
        max_loss_usd=float(max_start_loss_usd), min_equity_usd=float(minimum),
        mode="PREFLIGHT_ONLY", save_dir=None, show=show,
    )


def _payload(*, ticker, side, qty, price, cid, post_only, reduce_only,
             tif, group_id=None):
    if qty <= EPS or not (0 < float(price) < 1):
        raise RuntimeError(f"Invalid order qty/price: {qty}, {price}")
    p = {
        "ticker": str(ticker), "client_order_id": str(cid),
        "side": str(side).lower(), "count": f"{float(qty):.2f}",
        "price": f"{float(price):.4f}", "time_in_force": str(tif),
        "self_trade_prevention_type": "taker_at_cross",
        "post_only": bool(post_only), "cancel_order_on_pause": True,
        "reduce_only": bool(reduce_only), "subaccount": 0, "exchange_index": 0,
    }
    if group_id:
        p["order_group_id"] = str(group_id)
    return p


def _find_client_order(client, cid):
    try:
        body, _ = client.get("/portfolio/orders", params={"limit": 1000, "subaccount": 0})
        for r in body.get("orders") or []:
            if str(r.get("client_order_id") or "") == str(cid):
                return r
    except Exception:
        pass
    return None


def _post(client, payload):
    """Idempotent-ish retry: reuse client_order_id and recover from order list."""
    last = None
    for n in range(3):
        try:
            return client.post("/portfolio/events/orders", payload)
        except Exception as exc:
            last = exc
            found = _find_client_order(client, payload["client_order_id"])
            if found:
                body = {
                    "order_id": found.get("order_id"),
                    "client_order_id": payload["client_order_id"],
                    "fill_count": found.get("fill_count_fp") or "0.00",
                    "remaining_count": found.get("remaining_count_fp") or "0.00",
                    "recovered": True,
                }
                return body, {"recovered_after": repr(exc), "attempt": n + 1}
            time.sleep(0.12 * (n + 1))
    raise RuntimeError(f"Order create failed: {last!r}")


def _create_group(client):
    body, timing = client.post(
        "/portfolio/order_groups/create",
        {"subaccount": 0, "contracts_limit_fp": GROUP_LIMIT_FP, "exchange_index": 0},
    )
    gid = str(body.get("order_group_id") or "")
    if not gid:
        raise RuntimeError(f"Order group response missing id: {body}")
    return gid, body, timing


def _trigger_group(client, gid):
    if not gid:
        return {"ok": True, "note": "no group"}
    try:
        body, timing = client.request("PUT", f"/portfolio/order_groups/{gid}/trigger", payload={})
        return {"ok": True, "body": body, "timing": timing}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _delete_group(client, gid):
    try:
        body, timing = client.delete(
            f"/portfolio/order_groups/{gid}",
            params={"subaccount": 0, "exchange_index": 0},
        )
        return {"ok": True, "body": body, "timing": timing}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _get_order(client, oid):
    body, timing = client.get(f"/portfolio/orders/{oid}")
    return body.get("order") or {}, timing


def _cancel(client, oid):
    try:
        body, timing = client.delete(
            f"/portfolio/events/orders/{oid}",
            params={"subaccount": 0, "exchange_index": 0},
        )
        return {"ok": True, "body": body, "timing": timing}
    except Exception as exc:
        try:
            row, timing = _get_order(client, oid)
            rem = _f(row.get("remaining_count_fp"), 0.0)
            if rem <= EPS or str(row.get("status") or "").lower() != "resting":
                return {"ok": True, "already_done": True, "order": row, "timing": timing}
        except Exception:
            pass
        return {"ok": False, "error": repr(exc)}


def _fills(client, oid):
    body, timing = client.get(
        "/portfolio/fills",
        params={"order_id": str(oid), "limit": 1000, "subaccount": 0},
    )
    return body.get("fills") or [], timing


def _queue(client, ticker, oid):
    try:
        body, timing = client.get(
            "/portfolio/orders/queue_positions",
            params={"market_tickers": str(ticker), "subaccount": 0},
        )
        for r in body.get("queue_positions") or []:
            if str(r.get("order_id") or "") == str(oid):
                return _f(r.get("queue_position_fp")), timing, None
        return np.nan, timing, "order absent"
    except Exception as exc:
        return np.nan, {}, repr(exc)


class LiveEngine:
    def __init__(self, session, cfg, client, recorder_proc, gid, pre):
        self.session = Path(session).resolve()
        self.cfg = cfg
        self.client = client
        self.recorder_proc = recorder_proc
        self.gid = gid
        self.mode = str(cfg["mode"])
        self.q = float(cfg["quote_size"])
        self.hours = float(cfg["runtime_hours"])
        self.max_loss = float(cfg["max_start_loss_usd"])
        self.start_equity = float(pre["account"]["equity_usd"])
        self.kill_equity = self.start_equity - self.max_loss
        self.equity = self.start_equity
        self.peak_equity = self.start_equity
        self.max_dd = 0.0

        raw = self.session / "raw_capture"
        self.book_tail = OOS.JsonlTail(raw / "book_top3_events.jsonl")
        self.meta_tail = OOS.JsonlTail(raw / "market_metadata.jsonl")

        self.meta = {}
        self.current = {}
        self.positions = defaultdict(float)
        self.active = {}            # real order tracks only
        self.barrier = {}           # ticker -> book version at last fill
        self.book_version = defaultdict(int)
        self.first_seen = set()
        self.eligible = {}
        self.finalized = set()
        self.windows = set()
        self.smoke_window = None
        self.smoke_end = None
        self.trade_start = None
        self.trade_deadline = None
        self.seen_fills = set()
        self.counts = Counter()
        self.shutdown_started = False
        self.shutdown_reason = None
        self.last_error = None

        self.events = self.session / "events.jsonl"
        self.decisions = self.session / "decisions.jsonl"
        self.orders = self.session / "orders.jsonl"
        self.queue_log = self.session / "queue_positions.jsonl"
        self.fills_log = self.session / "fills.jsonl"
        self.positions_log = self.session / "positions.jsonl"
        self.pnl_log = self.session / "pnl_snapshots.jsonl"
        self.risk_log = self.session / "risk_events.jsonl"
        self.m5_log = self.session / "m5_liquidations.jsonl"
        self.health_path = self.session / "health.json"
        self.final_path = self.session / "final_summary.json"
        self.kill_request = self.session / "KILL_REQUEST.json"

        now = time.time()
        self.t_order = now
        self.t_pos = now
        self.t_bal = now
        self.t_queue = now
        self.t_health = 0.0

    def emit(self, event, ticker=None, **kw):
        _append(self.events, {"time": _iso(), "event": event, "ticker": ticker, **kw})
        if event in {"ENTRY_ORDER", "EXIT_ORDER", "FILL", "M5_FLATTEN", "LOSS_LIMIT", "SHUTDOWN", "ERROR", "CRITICAL"}:
            print(f"[{pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S UTC')}] {event} | {ticker or ''} | {kw}", flush=True)

    def update_meta(self):
        for r in self.meta_tail.read_new():
            t = str(r.get("ticker") or "")
            if t:
                self.meta[t] = r

    def window_key(self, ticker):
        return str((self.meta.get(ticker) or {}).get("close_time") or "")

    def series(self, ticker):
        return str((self.meta.get(ticker) or {}).get("series_ticker") or "")

    def m0_ts(self, ticker):
        close = pd.to_datetime((self.meta.get(ticker) or {}).get("close_time"), utc=True, errors="coerce")
        return np.nan if pd.isna(close) else float((close - pd.Timedelta(minutes=15)).timestamp())

    def first_book(self, ticker, elapsed):
        if ticker in self.first_seen:
            return
        self.first_seen.add(ticker)
        ok = bool(np.isfinite(elapsed) and -0.25 <= elapsed <= FULL_WINDOW_GRACE_S)
        key = self.window_key(ticker)
        if not key:
            ok = False
        if self.mode == "SMOKE_Q1_ONE_WINDOW" and ok:
            if self.smoke_window is None:
                self.smoke_window = key
                close = pd.to_datetime(key, utc=True, errors="coerce")
                if not pd.isna(close):
                    self.smoke_end = float((close - pd.Timedelta(minutes=10)).timestamp()) + 8.0
            elif key != self.smoke_window:
                ok = False
        self.eligible[ticker] = ok
        if not ok:
            self.emit("SKIP_PARTIAL_OR_OTHER_WINDOW", ticker, elapsed_s=elapsed, close_time=key)
            return
        if key not in self.windows:
            self.windows.add(key)
            self.emit("WINDOW_START", ticker, close_time=key, window_number=len(self.windows))
        if self.trade_start is None:
            m0 = self.m0_ts(ticker)
            self.trade_start = m0 if np.isfinite(m0) else time.time()
            self.trade_deadline = self.trade_start + self.hours * 3600.0
            self.emit("LIVE_CLOCK_START", ticker, start=self.trade_start, deadline=self.trade_deadline)

    def desired(self, ticker, cur, elapsed):
        if ticker in self.finalized or not self.eligible.get(ticker, False):
            return None
        if not (np.isfinite(elapsed) and 0 <= elapsed < 300):
            return None
        if self.trade_deadline is not None and time.time() >= self.trade_deadline:
            return None
        if ticker in self.barrier:
            if self.book_version[ticker] <= self.barrier[ticker]:
                return None
            self.barrier.pop(ticker, None)

        pos = float(self.positions.get(ticker, 0.0))
        if abs(pos) <= EPS:
            s = OOS._entry_side(cur)
            if s is None:
                return None
            side = s.lower()
            return {"role": "ENTRY", "side": side,
                    "price": float(cur["bid"] if side == "bid" else cur["ask"]),
                    "qty": self.q}
        if abs(pos) > self.q + 0.02:
            raise RuntimeError(f"Position safety violation {ticker}: {pos:+.4f} > Q{self.q:g}")
        side = "ask" if pos > 0 else "bid"
        return {"role": "EXIT", "side": side,
                "price": float(cur["ask"] if side == "ask" else cur["bid"]),
                "qty": abs(pos)}

    @staticmethod
    def same(track, d):
        return bool(track and d and track["role"] == d["role"] and track["side"] == d["side"]
                    and abs(track["price"] - d["price"]) <= 1e-9
                    and abs(track["qty"] - d["qty"]) <= 0.005)

    def record_fills(self, track):
        try:
            rows, timing = _fills(self.client, track["order_id"])
        except Exception as exc:
            self.emit("FILL_READ_ERROR", track["ticker"], error=repr(exc))
            return
        for r in rows:
            fid = str(r.get("fill_id") or r.get("trade_id") or "") or json.dumps(r, sort_keys=True, default=str)
            if fid in self.seen_fills:
                continue
            self.seen_fills.add(fid)
            _append(self.fills_log, {
                "observed_time": _iso(), "role": track.get("role"),
                "strategy_side": track.get("side"), "decision_book": track.get("book"),
                "order_id": track["order_id"], "client_order_id": track["cid"],
                "timing": timing, **r,
            })
            self.counts["fills"] += 1

    def refresh_position(self, ticker):
        p, row, timing = _position(self.client, ticker)
        self.positions[ticker] = p
        _append(self.positions_log, {"time": _iso(), "ticker": ticker, "position": p, "row": row, "timing": timing})
        return p

    def cancel_track(self, ticker, reason):
        tr = self.active.get(ticker)
        if not tr:
            return
        res = _cancel(self.client, tr["order_id"])
        self.record_fills(tr)
        _append(self.orders, {"time": _iso(), "action": "CANCEL", "ticker": ticker, "reason": reason, "track": tr, "result": res})
        if not res.get("ok"):
            raise RuntimeError(f"Cancel failed {ticker}: {res}")
        self.active.pop(ticker, None)
        self.counts["cancels"] += 1

    def place(self, ticker, d, cur, elapsed):
        cid = CLIENT_PREFIX + uuid.uuid4().hex
        payload = _payload(
            ticker=ticker, side=d["side"], qty=d["qty"], price=d["price"], cid=cid,
            post_only=True, reduce_only=False, tif="good_till_canceled", group_id=self.gid,
        )
        _append(self.decisions, {
            "time": _iso(), "ticker": ticker, "series": self.series(ticker),
            "close_time": self.window_key(ticker), "elapsed_s": elapsed,
            "position": self.positions.get(ticker, 0.0), **d, "book": cur,
        })
        body, timing = _post(self.client, payload)
        oid = str(body.get("order_id") or "")
        if not oid:
            raise RuntimeError(f"Create response missing order_id: {body}")
        tr = {"ticker": ticker, "order_id": oid, "cid": cid, "role": d["role"],
              "side": d["side"], "price": d["price"], "qty": d["qty"],
              "last_fill": 0.0, "book": cur, "elapsed_s": elapsed}
        self.active[ticker] = tr
        q, qt, qe = _queue(self.client, ticker, oid)
        predicted = cur["bid_q1"] if d["side"] == "bid" else cur["ask_q1"]
        _append(self.orders, {"time": _iso(), "action": "CREATE", "ticker": ticker,
                              "payload": payload, "response": body, "timing": timing})
        _append(self.queue_log, {"time": _iso(), "ticker": ticker, "order_id": oid,
                                 "role": d["role"], "queue_position": q,
                                 "displayed_l1_ahead": predicted, "timing": qt, "error": qe, "initial": True})
        self.counts["orders"] += 1
        self.emit("ENTRY_ORDER" if d["role"] == "ENTRY" else "EXIT_ORDER", ticker,
                  role=d["role"], side=d["side"], qty=d["qty"], price=d["price"])
        immediate = _f(body.get("fill_count"), 0.0)
        if immediate > EPS:
            self.fill_progress(ticker, tr, immediate, "create_response")

    def reconcile(self, ticker, cur, elapsed):
        d = self.desired(ticker, cur, elapsed)
        tr = self.active.get(ticker)
        if d is None:
            if tr:
                self.cancel_track(ticker, "DESIRED_NONE")
            return
        if self.same(tr, d):
            return
        if tr:
            self.cancel_track(ticker, "REPRICE_ROLE_OR_QTY")
        self.place(ticker, d, cur, elapsed)

    def fill_progress(self, ticker, tr, fill_count, source):
        old = float(tr.get("last_fill", 0.0))
        fill_count = float(fill_count)
        if fill_count <= old + EPS:
            return
        tr["last_fill"] = fill_count
        self.record_fills(tr)
        res = _cancel(self.client, tr["order_id"])
        _append(self.orders, {"time": _iso(), "action": "CANCEL_AFTER_FILL", "ticker": ticker,
                              "source": source, "old_fill": old, "new_fill": fill_count, "result": res})
        if not res.get("ok"):
            raise RuntimeError(f"Cancel-after-fill failed {ticker}: {res}")
        self.active.pop(ticker, None)
        p = self.refresh_position(ticker)
        self.barrier[ticker] = self.book_version[ticker]
        self.counts["fill_events"] += 1
        self.emit("FILL", ticker, role=tr["role"], side=tr["side"], qty=fill_count-old, position=p)

    def poll_orders(self):
        now = time.time()
        if now - self.t_order < ORDER_POLL_S:
            return
        self.t_order = now
        for ticker, tr in list(self.active.items()):
            row, _ = _get_order(self.client, tr["order_id"])
            fc = _f(row.get("fill_count_fp", row.get("fill_count")), 0.0)
            if fc > tr["last_fill"] + EPS:
                self.fill_progress(ticker, tr, fc, "order_poll")
                continue
            rem = _f(row.get("remaining_count_fp", row.get("remaining_count")), 0.0)
            if rem <= EPS or str(row.get("status") or "").lower() != "resting":
                self.record_fills(tr)
                self.active.pop(ticker, None)

    def poll_positions(self):
        now = time.time()
        if now - self.t_pos < POSITION_POLL_S:
            return
        self.t_pos = now
        rows, timing = _positions(self.client)
        new = defaultdict(float)
        for r in rows:
            t = str(r.get("ticker") or "")
            if t:
                new[t] = _f(r.get("position_fp"), 0.0)
        for t in set(self.positions) | set(new):
            self.positions[t] = new.get(t, 0.0)
            if abs(self.positions[t]) > self.q + 0.02:
                raise RuntimeError(f"Position safety violation {t}: {self.positions[t]:+.4f}")
        _append(self.positions_log, {"time": _iso(), "scope": "all", "positions": dict(new), "rows": rows, "timing": timing})

    def poll_balance(self):
        now = time.time()
        if now - self.t_bal < BALANCE_POLL_S:
            return
        self.t_bal = now
        eq, raw, timing = _balance(self.client)
        self.equity = float(eq["equity_usd"])
        self.peak_equity = max(self.peak_equity, self.equity)
        self.max_dd = min(self.max_dd, self.equity - self.peak_equity)
        row = {"time": _iso(), **eq, "start_equity_usd": self.start_equity,
               "start_pnl_usd": self.equity-self.start_equity,
               "peak_equity_usd": self.peak_equity,
               "peak_drawdown_usd": self.equity-self.peak_equity,
               "kill_equity_usd": self.kill_equity, "raw": raw, "timing": timing}
        _append(self.pnl_log, row)
        if self.equity <= self.kill_equity + EPS:
            _append(self.risk_log, {"time": _iso(), "event": "LOSS_LIMIT", **row})
            self.emit("LOSS_LIMIT", equity=self.equity, kill_equity=self.kill_equity)
            self.shutdown("LOSS_LIMIT")

    def poll_queue(self):
        now = time.time()
        if now - self.t_queue < QUEUE_POLL_S:
            return
        self.t_queue = now
        for ticker, tr in list(self.active.items()):
            q, timing, err = _queue(self.client, ticker, tr["order_id"])
            predicted = tr["book"]["bid_q1"] if tr["side"] == "bid" else tr["book"]["ask_q1"]
            _append(self.queue_log, {"time": _iso(), "ticker": ticker, "order_id": tr["order_id"],
                                     "role": tr["role"], "queue_position": q,
                                     "displayed_l1_ahead_at_join": predicted,
                                     "timing": timing, "error": err, "initial": False})

    def flatten(self, ticker, reason):
        if ticker in self.active:
            self.cancel_track(ticker, reason + "_CANCEL")
        attempts = []
        for n in range(14):
            p = self.refresh_position(ticker)
            if abs(p) <= EPS:
                return {"flat": True, "attempts": attempts, "final_position": p}
            cur, book_timing = Q1._current_book(self.client, ticker)
            side = "ask" if p > 0 else "bid"
            px = float(cur["bid"] if p > 0 else cur["ask"])
            cid = CLIENT_PREFIX + "flat-" + uuid.uuid4().hex
            payload = _payload(ticker=ticker, side=side, qty=abs(p), price=px, cid=cid,
                               post_only=False, reduce_only=True, tif="immediate_or_cancel", group_id=None)
            body, timing = _post(self.client, payload)
            rec = {"time": _iso(), "reason": reason, "ticker": ticker, "attempt": n+1,
                   "position_before": p, "qty": abs(p), "side": side, "price": px,
                   "book": cur, "book_timing": book_timing, "response": body, "timing": timing}
            attempts.append(rec)
            _append(self.m5_log if reason == "M5" else self.risk_log, rec)
            oid = str(body.get("order_id") or "")
            if oid:
                self.record_fills({"ticker": ticker, "order_id": oid, "cid": cid,
                                   "role": "M5_FLATTEN" if reason == "M5" else "RISK_FLATTEN",
                                   "side": side, "book": cur})
            self.emit("M5_FLATTEN" if reason == "M5" else "RISK_FLATTEN",
                      ticker, qty=abs(p), price=px, position=p, reason=reason)
            time.sleep(0.12)

        # Last-resort reduce-only IOC to sweep available depth if touch-only retries fail.
        for n in range(3):
            p = self.refresh_position(ticker)
            if abs(p) <= EPS:
                return {"flat": True, "attempts": attempts, "final_position": p}
            side = "ask" if p > 0 else "bid"
            px = 0.01 if p > 0 else 0.99
            cid = CLIENT_PREFIX + "xf-" + uuid.uuid4().hex
            payload = _payload(ticker=ticker, side=side, qty=abs(p), price=px, cid=cid,
                               post_only=False, reduce_only=True, tif="immediate_or_cancel", group_id=None)
            body, timing = _post(self.client, payload)
            rec = {"time": _iso(), "reason": reason + "_EXTREME", "ticker": ticker,
                   "attempt": n+1, "position_before": p, "qty": abs(p), "side": side,
                   "price": px, "response": body, "timing": timing}
            attempts.append(rec)
            _append(self.risk_log, rec)
            time.sleep(0.20)
        p = self.refresh_position(ticker)
        if abs(p) > EPS:
            raise RuntimeError(f"CRITICAL unable to flatten {ticker}: {p:+.4f}")
        return {"flat": True, "attempts": attempts, "final_position": p}

    def finalize_m5(self, ticker):
        if ticker in self.finalized:
            return
        self.finalized.add(ticker)
        if ticker in self.active:
            self.cancel_track(ticker, "M5")
        p = self.refresh_position(ticker)
        if abs(p) > EPS:
            self.flatten(ticker, "M5")
        self.emit("M5_FINALIZED", ticker, position=self.positions.get(ticker, 0.0))

    def on_book(self, r):
        ticker = str(r.get("ticker") or "")
        if not ticker:
            return
        elapsed = _f(r.get("elapsed_s"))
        cur = OOS._top_state(r)
        self.book_version[ticker] += 1
        if cur is not None:
            self.current[ticker] = cur
        self.first_book(ticker, elapsed)

        typ = str(r.get("event_type") or "")
        if typ == "trade_window_end" or (np.isfinite(elapsed) and elapsed >= 300):
            self.finalize_m5(ticker)
            return
        if not (np.isfinite(elapsed) and 0 <= elapsed < 300):
            return
        if cur is None:
            if ticker in self.active:
                self.cancel_track(ticker, "INVALID_BOOK")
            return
        self.reconcile(ticker, cur, elapsed)

    def stop_recorder(self):
        p = self.recorder_proc
        if not p or p.poll() is not None:
            return
        try:
            os.kill(p.pid, signal.SIGINT)
        except Exception:
            pass
        deadline = time.time() + RECORDER_STOP_TIMEOUT_S
        while time.time() < deadline and p.poll() is None:
            time.sleep(0.20)
        if p.poll() is None:
            try:
                os.kill(p.pid, signal.SIGTERM)
            except Exception:
                pass

    def health(self, force=False):
        now = time.time()
        if not force and now - self.t_health < HEALTH_S:
            return
        self.t_health = now
        rec_health = _read(self.session / "raw_capture" / "health.json", {}) or {}
        _atomic(self.health_path, {
            "time": _iso(), "live_version": LIVE_VERSION,
            "running": not self.shutdown_started,
            "state": "SHUTTING_DOWN" if self.shutdown_started else ("RUNNING" if self.trade_start else "ARMED_WAITING_FULL_WINDOW"),
            "mode": self.mode, "session_dir": str(self.session), "quote_size": self.q,
            "windows_started": len(self.windows), "smoke_window": self.smoke_window,
            "trade_start": self.trade_start, "trade_deadline": self.trade_deadline,
            "start_equity_usd": self.start_equity, "equity_usd": self.equity,
            "start_pnl_usd": self.equity-self.start_equity,
            "kill_equity_usd": self.kill_equity, "peak_equity_usd": self.peak_equity,
            "max_peak_drawdown_usd": self.max_dd,
            "positions": {t: p for t, p in self.positions.items() if abs(p) > EPS},
            "active_orders": self.active, "counts": dict(self.counts),
            "shutdown_reason": self.shutdown_reason, "last_error": self.last_error,
            "recorder_pid": self.recorder_proc.pid if self.recorder_proc else None,
            "recorder_alive": bool(self.recorder_proc and self.recorder_proc.poll() is None),
            "recorder_health": rec_health,
        })

    def shutdown(self, reason):
        if self.shutdown_started:
            return
        self.shutdown_started = True
        self.shutdown_reason = str(reason)
        self.emit("SHUTDOWN", reason=self.shutdown_reason, equity=self.equity)
        _append(self.risk_log, {"time": _iso(), "event": "SHUTDOWN_START", "reason": reason})
        _append(self.risk_log, {"time": _iso(), "event": "ORDER_GROUP_TRIGGER", "result": _trigger_group(self.client, self.gid)})

        for ticker in list(self.active):
            try:
                self.cancel_track(ticker, "GLOBAL_SHUTDOWN")
            except Exception as exc:
                self.emit("ERROR", ticker, error=repr(exc), reason="cancel_during_shutdown")

        rows, _ = _positions(self.client)
        for r in rows:
            ticker = str(r.get("ticker") or "")
            p = _f(r.get("position_fp"), 0.0)
            if abs(p) <= EPS:
                continue
            # Account was required flat at startup.  Refuse to touch a ticker the live recorder never saw.
            if ticker not in self.meta and ticker not in self.positions:
                self.emit("CRITICAL", ticker, reason="UNRECOGNIZED_POSITION", position=p)
                continue
            self.positions[ticker] = p
            try:
                self.flatten(ticker, "GLOBAL_SHUTDOWN")
            except Exception as exc:
                self.last_error = repr(exc)
                self.emit("CRITICAL", ticker, reason="FLATTEN_FAILED", position=p, error=repr(exc))

        self.stop_recorder()
        try:
            end_eq, raw, _ = _balance(self.client)
        except Exception as exc:
            end_eq, raw = {"equity_usd": np.nan}, {"error": repr(exc)}
        try:
            final_pos, _ = _positions(self.client)
        except Exception as exc:
            final_pos = [{"error": repr(exc)}]
        try:
            final_orders, _ = _resting(self.client)
        except Exception as exc:
            final_orders = [{"error": repr(exc)}]
        strategy_resting = [r for r in final_orders if str(r.get("client_order_id") or "").startswith(CLIENT_PREFIX)] if isinstance(final_orders, list) else final_orders
        summary = {
            "time": _iso(), "live_version": LIVE_VERSION, "mode": self.mode,
            "session_dir": str(self.session), "shutdown_reason": self.shutdown_reason,
            "start_equity_usd": self.start_equity,
            "end_equity_usd": _f(end_eq.get("equity_usd")),
            "account_pnl_usd": _f(end_eq.get("equity_usd")) - self.start_equity,
            "kill_equity_usd": self.kill_equity, "peak_equity_usd": self.peak_equity,
            "max_peak_drawdown_usd": self.max_dd, "windows_started": sorted(self.windows),
            "counts": dict(self.counts), "final_positions": final_pos,
            "final_strategy_resting_orders": strategy_resting,
            "flat_verified": bool(isinstance(final_pos, list) and not any(abs(_f(r.get("position_fp"), 0.0)) > EPS for r in final_pos if isinstance(r, dict))),
            "strategy_resting_orders_zero": bool(isinstance(strategy_resting, list) and not strategy_resting),
            "end_balance_raw": raw, "order_group_delete": _delete_group(self.client, self.gid),
            "last_error": self.last_error,
        }
        _atomic(self.final_path, summary)
        _atomic(self.health_path, {"time": _iso(), "running": False, "state": "STOPPED", "summary": summary})
        c = _ctl()
        if str(c.get("session_dir")) == str(self.session):
            c.update({"running": False, "stopped_at": _iso(), "shutdown_reason": self.shutdown_reason})
            _atomic(CONTROL_PATH, c)

    def check_end(self):
        if self.shutdown_started:
            return
        now = time.time()
        if self.kill_request.exists():
            req = _read(self.kill_request, {}) or {}
            self.shutdown(str(req.get("reason") or "MANUAL_KILL"))
            return
        if self.mode == "SMOKE_Q1_ONE_WINDOW" and self.smoke_end is not None and now >= self.smoke_end:
            self.shutdown("SMOKE_ONE_WINDOW_COMPLETE")
            return
        if self.trade_deadline is not None and now >= self.trade_deadline:
            self.shutdown("RUNTIME_COMPLETE")

    def run(self):
        self.emit("ENGINE_START", mode=self.mode)
        self.health(force=True)
        try:
            while not self.shutdown_started:
                if self.recorder_proc.poll() is not None:
                    raise RuntimeError(f"Raw V5 recorder exited rc={self.recorder_proc.returncode}")
                self.update_meta()
                for r in self.book_tail.read_new():
                    self.on_book(r)
                    if self.shutdown_started:
                        break
                if self.shutdown_started:
                    break
                self.poll_orders()
                self.poll_positions()
                self.poll_balance()
                self.poll_queue()
                self.check_end()
                self.health()
                time.sleep(LOOP_SLEEP_S)
        except BaseException as exc:
            self.last_error = repr(exc)
            self.emit("ERROR", error=repr(exc), traceback=traceback.format_exc())
            try:
                self.shutdown("ENGINE_EXCEPTION")
            except Exception as cleanup_exc:
                self.emit("CRITICAL", error=repr(cleanup_exc), reason="cleanup_exception")
                self.stop_recorder()
            raise
        finally:
            self.health(force=True)


def _start_recorder(session):
    raw = Path(session) / "raw_capture"
    raw.mkdir(parents=True, exist_ok=True)
    log = Path(session) / "raw_recorder.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [sys.executable, "-m", "quant_research.kalshi.mm_event_time_m0_m5_recorder_v5", "--run-session", str(raw)],
            cwd=str(C.PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
        )
    finally:
        fh.close()
    deadline = time.time() + RECORDER_START_TIMEOUT_S
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-6000:] if log.exists() else ""
            raise RuntimeError(f"V5 recorder startup failure rc={p.returncode}\n{tail}")
        last = _read(raw / "health.json", {}) or {}
        if last.get("running") and last.get("healthy"):
            return p, last
        time.sleep(0.35)
    try:
        os.kill(p.pid, signal.SIGTERM)
    except Exception:
        pass
    raise RuntimeError(f"V5 recorder health timeout: {last}")


def _run_process(session, cfg):
    session = Path(session).resolve()
    spec = {
        "live_version": LIVE_VERSION, "mode": cfg["mode"], "real_money": True,
        "universe": list(SERIES), "quote_size": cfg["quote_size"],
        "window": "M0 <= elapsed < M5", "entry": "Candidate C: L3 support + spread >=2c",
        "after_fill": "stop adding; opposite BBO exit only", "m5": "reduce-only IOC flatten",
        "runtime_hours": cfg["runtime_hours"], "max_start_loss_usd": cfg["max_start_loss_usd"],
        "loss_metric": "available balance + portfolio value", "cancel_order_on_pause": True,
    }
    _atomic(session / "run_spec.json", spec)
    _atomic(session / "raw_capture_note.json", {
        "time": _iso(), "note": "raw_capture reuses the proven V5 public recorder unchanged; its internal development labels describe the recorder component only."
    })
    pre = _preflight(
        quote_size=cfg["quote_size"], runtime_hours=cfg["runtime_hours"],
        max_loss_usd=cfg["max_start_loss_usd"], min_equity_usd=cfg["min_start_equity_usd"],
        mode=cfg["mode"], save_dir=session, show=True,
    )
    client = Q1.LiveClient()
    gid, gb, gt = _create_group(client)
    _atomic(session / "order_group.json", {"time": _iso(), "order_group_id": gid, "response": gb, "timing": gt})
    recorder = None
    try:
        recorder, rh = _start_recorder(session)
        _atomic(session / "raw_recorder_start.json", {"time": _iso(), "pid": recorder.pid, "health": rh})
        LiveEngine(session, cfg, client, recorder, gid, pre).run()
    except BaseException:
        _trigger_group(client, gid)
        if recorder is not None and recorder.poll() is None:
            try:
                os.kill(recorder.pid, signal.SIGINT)
            except Exception:
                pass
        raise


def _launch(*, mode, q, hours, max_loss, min_equity, arm, expected):
    if str(arm) != expected:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected!r} exactly.")
    old = _ctl()
    if old and _pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    _preflight(quote_size=q, runtime_hours=hours, max_loss_usd=max_loss,
               min_equity_usd=min_equity, mode=mode, save_dir=None, show=True)

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (ROOT / f"{stamp}_{mode.lower()}").resolve()
    session.mkdir(parents=True, exist_ok=False)
    cfg = {"mode": mode, "quote_size": float(q), "runtime_hours": float(hours),
           "max_start_loss_usd": float(max_loss), "min_start_equity_usd": float(min_equity)}
    cfg_path = session / "process_config.json"
    _atomic(cfg_path, cfg)
    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [sys.executable, "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v1",
             "--run-live-session", str(session), "--config", str(cfg_path)],
            cwd=str(C.PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
        )
    finally:
        fh.close()
    _atomic(CONTROL_PATH, {"live_version": LIVE_VERSION, "running": True, "pid": p.pid,
                           "session_dir": str(session), "mode": mode, "started_at": _iso(),
                           "config": cfg, "log_path": str(log)})

    deadline = time.time() + 90
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-8000:] if log.exists() else ""
            raise RuntimeError(f"Live process exited during startup rc={p.returncode}\n{tail}")
        last = _read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-8000:] if log.exists() else ""
        raise RuntimeError(f"Live startup timeout. Last health={last}\n{tail}")

    print("\nLIVE PROCESS ARMED")
    print("  mode:   ", mode)
    print("  session:", session)
    print("  pid:    ", p.pid)
    print(f"  Q:       {q:g} per market")
    print(f"  kill:    -${max_loss:.2f} from starting account equity")
    print("Use live_status(); emergency stop is kill_and_flatten_live(arm_phrase='KILL_AND_FLATTEN').")
    return live_status(show=False)


def start_live_smoke_q1_one_window(*, arm_phrase=None,
                                   max_start_loss_usd=LOSS_LIMIT_USD,
                                   min_start_equity_usd=SMOKE_MIN_EQUITY):
    """Q1 per market, exact frozen system, one complete synchronized strategy window."""
    return _launch(mode="SMOKE_Q1_ONE_WINDOW", q=SMOKE_Q, hours=1.0,
                   max_loss=float(max_start_loss_usd), min_equity=float(min_start_equity_usd),
                   arm=arm_phrase, expected=SMOKE_ARM)


def start_live_cycle_q10(*, arm_phrase=None, runtime_hours=FULL_HOURS,
                         max_start_loss_usd=LOSS_LIMIT_USD,
                         min_start_equity_usd=FULL_MIN_EQUITY):
    """Frozen Q10, 24h from first complete M0 window."""
    if abs(float(runtime_hours) - FULL_HOURS) > EPS:
        raise RuntimeError("V1 full validation is frozen to exactly 24 hours.")
    return _launch(mode="LIVE_Q10_24H", q=FULL_Q, hours=FULL_HOURS,
                   max_loss=float(max_start_loss_usd), min_equity=float(min_start_equity_usd),
                   arm=arm_phrase, expected=FULL_ARM)


def live_status(*, show=True, tail_lines=20):
    c = _ctl()
    if not c:
        out = {"running": False, "message": "No live control file."}
        if show:
            print(out)
        return out
    session = Path(c.get("session_dir", ""))
    health = _read(session / "health.json", {}) or {}
    final = _read(session / "final_summary.json", {}) or {}
    running = bool(_pid_alive(c.get("pid")) and health.get("state") != "STOPPED")
    tail = []
    try:
        tail = (session / "live_process.log").read_text(encoding="utf-8").splitlines()[-int(tail_lines):]
    except Exception:
        pass
    out = {"running": running, "pid": c.get("pid"), "session_dir": str(session),
           "mode": c.get("mode"), "health": health, "final_summary": final, "log_tail": tail}
    if show:
        print("=" * 100)
        print("LIVE STRATEGY STATUS")
        print("=" * 100)
        print("running:", running)
        print("session:", session)
        if health:
            print("state:", health.get("state"))
            print("equity:", health.get("equity_usd"))
            print("start PnL:", health.get("start_pnl_usd"))
            print("kill equity:", health.get("kill_equity_usd"))
            print("windows:", health.get("windows_started"))
            print("positions:", health.get("positions"))
            print("active orders:", len(health.get("active_orders") or {}))
            print("recorder alive:", health.get("recorder_alive"))
            print("last error:", health.get("last_error"))
        if final:
            print("final PnL:", final.get("account_pnl_usd"))
            print("flat verified:", final.get("flat_verified"))
            print("shutdown:", final.get("shutdown_reason"))
        if tail:
            print("\nLOG TAIL")
            print("\n".join(tail))
    return out


def _fallback_cleanup(c):
    session = Path(c.get("session_dir", ""))
    group = _read(session / "order_group.json", {}) or {}
    client = Q1.LiveClient()
    result = {"time": _iso(), "group_trigger": _trigger_group(client, group.get("order_group_id")), "actions": []}
    known = set()
    meta_path = session / "raw_capture" / "market_metadata.jsonl"
    try:
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("ticker"):
                known.add(str(r["ticker"]))
    except Exception:
        pass
    rows, _ = _positions(client)
    for r in rows:
        ticker = str(r.get("ticker") or "")
        p = _f(r.get("position_fp"), 0.0)
        if abs(p) <= EPS:
            continue
        if ticker not in known:
            result["actions"].append({"ticker": ticker, "position": p, "action": "REFUSED_UNRECOGNIZED"})
            continue
        attempts = []
        for _ in range(14):
            p, _, _ = _position(client, ticker)
            if abs(p) <= EPS:
                break
            cur, _ = Q1._current_book(client, ticker)
            side = "ask" if p > 0 else "bid"
            px = cur["bid"] if p > 0 else cur["ask"]
            payload = _payload(ticker=ticker, side=side, qty=abs(p), price=px,
                               cid=CLIENT_PREFIX+"manualflat-"+uuid.uuid4().hex,
                               post_only=False, reduce_only=True, tif="immediate_or_cancel")
            try:
                body, timing = _post(client, payload)
                attempts.append({"position": p, "payload": payload, "response": body, "timing": timing})
            except Exception as exc:
                attempts.append({"position": p, "error": repr(exc)})
            time.sleep(0.15)
        p, _, _ = _position(client, ticker)
        result["actions"].append({"ticker": ticker, "attempts": attempts, "final_position": p})
    _atomic(session / "manual_emergency_cleanup.json", result)
    return result


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    """May send real cancels and reduce-only IOC cleanup orders."""
    if str(arm_phrase) != KILL_ARM:
        raise RuntimeError("Pass arm_phrase='KILL_AND_FLATTEN' exactly.")
    c = _ctl()
    if not c:
        raise RuntimeError("No live session control file.")
    session = Path(c["session_dir"])
    _atomic(session / "KILL_REQUEST.json", {"time": _iso(), "reason": "MANUAL_KILL_AND_FLATTEN"})
    deadline = time.time() + float(wait_s)
    while time.time() < deadline:
        st = live_status(show=False)
        if not st.get("running"):
            print("Live process stopped after kill request.")
            return st
        time.sleep(0.5)
    print("No stop confirmation; running direct cleanup fallback.")
    result = _fallback_cleanup(c)
    print(json.dumps(result, indent=2, default=str))
    return result


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    a = ap.parse_args()
    if a.run_live_session:
        cfg = _read(Path(a.config), {}) or {}
        _run_process(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "live_preflight", "start_live_smoke_q1_one_window", "start_live_cycle_q10",
    "live_status", "kill_and_flatten_live",
]
