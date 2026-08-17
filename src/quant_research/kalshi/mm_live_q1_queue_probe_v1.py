from __future__ import annotations

"""One-contract LIVE Kalshi queue-position / latency probe.

This is NOT a strategy backtest and NOT part of the frozen OOS shadow.
It submits at most one real Q1 passive entry, measures where the order actually
lands in FIFO, and if any quantity fills, attempts one passive reduce-only exit.
Any remaining filled quantity is flattened with reduce-only IOC orders.

Scientific guardrail
--------------------
The probe may run while the frozen M0-M5 OOS recorder is active ONLY on a
contract that is already beyond M5+30s (elapsed >= 330s). That keeps the real
order outside the frozen strategy window and outside its 30s label tail.
If the exploratory M0-M15 sidecar is running, this script writes a contamination
marker so that contract can be excluded from later exploratory analysis.

Real-money guardrails
---------------------
- quantity hard-coded to <= 1.00 contract
- available balance must be >= $2
- entry and passive exit are post_only=True
- exit is reduce_only=True
- remaining inventory is flattened with reduce_only immediate_or_cancel
- refuses a selected ticker with pre-existing position or resting order
- no API key/private key material is printed or persisted
"""

import base64
import json
import math
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_event_time_m0_m15_exploratory_recorder_v1 as FULL15

STUDY_VERSION = "MM_LIVE_Q1_QUEUE_PROBE_V1"
QTY = 1.0
SAFE_AFTER_OOS_S = 330.0      # M5 + 30s label tail
LATEST_ENTRY_S = 720.0         # do not start after M12
MIN_BALANCE_DOLLARS = 2.0
MIN_PRICE = 0.10
MAX_PRICE = 0.90
TARGET_QUEUE = 20.0
MAX_QUEUE_FOR_SELECTION = 150.0
ORDER_POLL_S = 0.50
ENTRY_WAIT_S = 35.0
EXIT_WAIT_S = 35.0
MAX_DISCOVERY_WAIT_S = 20 * 60.0
RESULT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_live_q1_queue_probe"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _iso_now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def _sign(private_key, ts_ms: str, method: str, full_path: str) -> str:
    msg = (str(ts_ms) + method.upper() + full_path.split("?", 1)[0]).encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


class LiveClient:
    def __init__(self):
        self.key_id, self.private_key = C.load_auth()
        self.http = requests.Session()

    def request(self, method, path, *, params=None, payload=None, timeout=8.0):
        method = method.upper()
        url = C.REST_BASE + path
        sign_path = urlparse(url).path
        ts = str(int(time.time() * 1000))

        sign_t0 = time.perf_counter_ns()
        sig = _sign(self.private_key, ts, method, sign_path)
        sign_t1 = time.perf_counter_ns()

        headers = {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"

        send_wall_ms = time.time_ns() / 1e6
        send_perf_ns = time.perf_counter_ns()
        r = self.http.request(
            method,
            url,
            params=params,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        recv_perf_ns = time.perf_counter_ns()
        recv_wall_ms = time.time_ns() / 1e6

        try:
            body = r.json()
        except Exception:
            body = {"raw_text": r.text}
        if not r.ok:
            raise RuntimeError(f"Kalshi {method} {path} -> {r.status_code}: {body}")

        engine_ms = _f(body.get("ts_ms")) if isinstance(body, dict) else np.nan
        timing = {
            "sign_ms": (sign_t1 - sign_t0) / 1e6,
            "request_send_wall_ms": send_wall_ms,
            "response_recv_wall_ms": recv_wall_ms,
            "rtt_ms": (recv_perf_ns - send_perf_ns) / 1e6,
            "engine_ts_ms": engine_ms,
            "send_to_engine_ms_local_clock": (
                engine_ms - send_wall_ms if np.isfinite(engine_ms) else np.nan
            ),
            "engine_to_response_ms_local_clock": (
                recv_wall_ms - engine_ms if np.isfinite(engine_ms) else np.nan
            ),
        }
        return body, timing

    def get(self, path, params=None):
        return self.request("GET", path, params=params)

    def post(self, path, payload):
        return self.request("POST", path, payload=payload)

    def delete(self, path, params=None):
        return self.request("DELETE", path, params=params)


def _book_state(payload):
    ob = (payload or {}).get("orderbook_fp") or {}
    yes = ob.get("yes_dollars") or []
    no = ob.get("no_dollars") or []
    if not yes or not no:
        return None
    try:
        yes_levels = [(float(p), float(q)) for p, q in yes]
        no_levels = [(float(p), float(q)) for p, q in no]
    except Exception:
        return None
    yes_levels.sort(key=lambda x: x[0])
    no_levels.sort(key=lambda x: x[0])
    yes_bid, bid_q1 = yes_levels[-1]
    best_no_bid, ask_q1 = no_levels[-1]
    yes_ask = 1.0 - best_no_bid
    if not (0.0 < yes_bid < yes_ask < 1.0):
        return None
    bid_depth3 = sum(q for _, q in yes_levels[-3:])
    ask_depth3 = sum(q for _, q in no_levels[-3:])
    return {
        "bid": yes_bid,
        "ask": yes_ask,
        "bid_q1": bid_q1,
        "ask_q1": ask_q1,
        "bid_depth3": bid_depth3,
        "ask_depth3": ask_depth3,
        "spread_c": 100.0 * (yes_ask - yes_bid),
    }


def _elapsed_from_market(m, now=None):
    now = now or pd.Timestamp.now(tz="UTC")
    close = pd.to_datetime(m.get("close_time"), utc=True, errors="coerce")
    if pd.isna(close):
        return np.nan
    m0 = close - pd.Timedelta(minutes=15)
    return float((now - m0).total_seconds())


def _fee_ok(series):
    s = (C.rest_get(f"/series/{series}", {}).get("series") or {})
    return str(s.get("fee_type") or "").strip().lower() == "quadratic"


def _discover_safe_markets():
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for series in OOS.SERIES:
        try:
            markets = C.rest_get(
                "/markets",
                {"series_ticker": series, "status": "open", "limit": 100},
            ).get("markets") or []
        except Exception:
            continue
        for m in markets:
            elapsed = _elapsed_from_market(m, now)
            if not np.isfinite(elapsed):
                continue
            if SAFE_AFTER_OOS_S <= elapsed <= LATEST_ENTRY_S:
                z = dict(m)
                z["series_ticker"] = series
                z["elapsed_s"] = elapsed
                out.append(z)
    return out


def _get_position(client, ticker):
    body, _ = client.get(
        "/portfolio/positions",
        params={
            "ticker": ticker,
            "count_filter": "position",
            "subaccount": 0,
            "exchange_index": 0,
        },
    )
    rows = body.get("market_positions") or []
    for r in rows:
        if str(r.get("ticker")) == ticker:
            return _f(r.get("position_fp"), 0.0), r
    return 0.0, None


def _resting_orders(client, ticker):
    body, _ = client.get(
        "/portfolio/orders",
        params={"ticker": ticker, "status": "resting", "limit": 100},
    )
    return [r for r in (body.get("orders") or []) if str(r.get("status")) == "resting"]


def _get_order(client, order_id):
    body, timing = client.get(f"/portfolio/orders/{order_id}")
    return body.get("order") or {}, timing


def _queue_position(client, order_id):
    try:
        body, timing = client.get(f"/portfolio/orders/{order_id}/queue_position")
        return _f(body.get("queue_position_fp")), timing, None
    except Exception as exc:
        return np.nan, {}, repr(exc)


def _fills(client, order_id):
    body, timing = client.get(
        "/portfolio/fills",
        params={"order_id": order_id, "limit": 1000, "subaccount": 0, "exchange_index": 0},
    )
    return body.get("fills") or [], timing


def _weighted_fill_price(fills):
    q = 0.0
    x = 0.0
    fees = 0.0
    for f in fills:
        n = _f(f.get("count_fp"), 0.0)
        p = _f(f.get("yes_price_dollars"))
        if n > 0 and np.isfinite(p):
            q += n
            x += n * p
        fees += _f(f.get("fee_cost"), 0.0)
    return (x / q if q > 1e-12 else np.nan), q, fees


def _make_order_payload(ticker, side, qty, price, *, reduce_only, post_only, tif="good_till_canceled"):
    if qty <= 0 or qty > QTY + 1e-9:
        raise RuntimeError(f"Probe quantity guard violated: {qty}")
    return {
        "ticker": ticker,
        "client_order_id": str(uuid.uuid4()),
        "side": str(side).lower(),
        "count": f"{qty:.2f}",
        "price": f"{float(price):.4f}",
        "time_in_force": tif,
        "self_trade_prevention_type": "taker_at_cross",
        "post_only": bool(post_only),
        "cancel_order_on_pause": True,
        "reduce_only": bool(reduce_only),
        "subaccount": 0,
        "exchange_index": 0,
    }


def _cancel_if_remaining(client, order_id):
    try:
        order, _ = _get_order(client, order_id)
        rem = _f(order.get("remaining_count_fp"), 0.0)
    except Exception:
        rem = 0.0
    if rem <= 1e-12:
        return None
    body, timing = client.delete(
        f"/portfolio/events/orders/{order_id}",
        params={"subaccount": 0, "exchange_index": 0},
    )
    return {"response": body, "timing": timing}


def _wait_for_any_fill(client, order_id, timeout_s):
    deadline = time.time() + float(timeout_s)
    last = {}
    while time.time() < deadline:
        order, _ = _get_order(client, order_id)
        last = order
        filled = _f(order.get("fill_count_fp"), 0.0)
        remaining = _f(order.get("remaining_count_fp"), 0.0)
        if filled > 1e-12 or remaining <= 1e-12:
            return order
        time.sleep(ORDER_POLL_S)
    return last


def _current_book(client, ticker):
    body, timing = client.get(f"/markets/{ticker}/orderbook")
    cur = _book_state(body)
    if cur is None:
        raise RuntimeError(f"No valid two-sided book for {ticker}")
    timing = dict(timing)
    timing["decision_book_recv_wall_ms"] = timing.get("response_recv_wall_ms")
    return cur, timing


def _mark_full15_contamination(record):
    ctl = FULL15._read_json(FULL15.CONTROL_PATH, {}) or {}
    if not ctl or not FULL15._pid_alive(ctl.get("pid")):
        return None
    session = Path(ctl.get("session_dir", ""))
    if not session.exists():
        return None
    path = session / "live_probe_contamination.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
    return str(path)


def _choose_probe_market(client, *, max_wait_s=MAX_DISCOVERY_WAIT_S, show=True):
    deadline = time.time() + float(max_wait_s)
    last_print = 0.0
    while time.time() < deadline:
        markets = _discover_safe_markets()
        scored = []
        for m in markets:
            ticker = str(m.get("ticker") or "")
            series = str(m.get("series_ticker") or "")
            if not ticker:
                continue
            try:
                cur, bt = _current_book(client, ticker)
            except Exception:
                continue
            # Avoid extreme prices and intentionally target a moderate queue so
            # the order has time to rest long enough for a queue-position read.
            sides = [
                ("bid", cur["bid"], cur["bid_q1"]),
                ("ask", cur["ask"], cur["ask_q1"]),
            ]
            for side, price, q in sides:
                if not (MIN_PRICE <= price <= MAX_PRICE):
                    continue
                if not (0.0 < q <= MAX_QUEUE_FOR_SELECTION):
                    continue
                scored.append((abs(q - TARGET_QUEUE), q, ticker, series, side, price, cur, bt, m))
        if scored:
            scored.sort(key=lambda x: (x[0], x[1], x[2], x[4]))
            _, q, ticker, series, side, price, cur, bt, m = scored[0]
            # Fee structure check and account-state check are done immediately
            # before we actually place anything.
            return {
                "ticker": ticker,
                "series": series,
                "side": side,
                "price": price,
                "predicted_queue": q,
                "book": cur,
                "book_timing": bt,
                "market": m,
            }
        if show and time.time() - last_print >= 10.0:
            print("Waiting for a safe post-M5+30s frozen-universe market with a measurable queue...")
            last_print = time.time()
        time.sleep(2.0)
    raise TimeoutError("No suitable post-M5 probe market appeared before max_wait_s.")


def _submit_and_measure(client, *, ticker, side, qty, price, predicted_queue, reduce_only, label):
    decision_ms = time.time_ns() / 1e6
    payload = _make_order_payload(
        ticker, side, qty, price,
        reduce_only=reduce_only,
        post_only=True,
        tif="good_till_canceled",
    )
    body, timing = client.post("/portfolio/events/orders", payload)
    order_id = str(body.get("order_id") or "")
    if not order_id:
        raise RuntimeError(f"{label}: create response missing order_id: {body}")

    actual_q, qtiming, qerr = _queue_position(client, order_id)
    engine_ms = _f(timing.get("engine_ts_ms"))
    result = {
        "label": label,
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "price": price,
        "predicted_queue_ahead": predicted_queue,
        "actual_queue_position_first": actual_q,
        "extra_queue_vs_snapshot": (
            actual_q - predicted_queue
            if np.isfinite(actual_q) and np.isfinite(predicted_queue) else np.nan
        ),
        "order_id": order_id,
        "client_order_id": body.get("client_order_id") or payload["client_order_id"],
        "create_response": body,
        "create_timing": timing,
        "queue_query_timing": qtiming,
        "queue_query_error": qerr,
        "decision_to_engine_ms_local_clock": (
            engine_ms - decision_ms if np.isfinite(engine_ms) else np.nan
        ),
        "engine_to_first_queue_observation_ms_local_clock": (
            _f(qtiming.get("response_recv_wall_ms")) - engine_ms
            if np.isfinite(engine_ms) and np.isfinite(_f(qtiming.get("response_recv_wall_ms"))) else np.nan
        ),
    }
    print(f"\n{label} POSTED")
    print(f"  {ticker} | {side.upper()} | {qty:.2f} @ {price:.4f}")
    print(f"  predicted queue ahead: {predicted_queue:.2f}")
    print(f"  actual first queue pos: {actual_q:.2f}" if np.isfinite(actual_q) else "  actual first queue pos: unavailable")
    if np.isfinite(result["extra_queue_vs_snapshot"]):
        print(f"  EXTRA ahead vs snapshot: {result['extra_queue_vs_snapshot']:+.2f}")
    print(f"  create RTT: {timing['rtt_ms']:.1f} ms")
    if np.isfinite(timing.get("send_to_engine_ms_local_clock", np.nan)):
        print(f"  send -> matching engine: {timing['send_to_engine_ms_local_clock']:.1f} ms (local-clock dependent)")
    if np.isfinite(result["decision_to_engine_ms_local_clock"]):
        print(f"  decision -> matching engine: {result['decision_to_engine_ms_local_clock']:.1f} ms (local-clock dependent)")
    return result


def _force_flatten(client, ticker, entry_side, remaining_qty, record):
    remaining_qty = float(remaining_qty)
    forced_orders = []
    for attempt in range(1, 6):
        if remaining_qty <= 1e-9:
            break
        cur, _ = _current_book(client, ticker)
        if entry_side == "bid":
            side = "ask"
            price = cur["bid"]  # marketable against current YES bid
        else:
            side = "bid"
            price = cur["ask"]  # marketable against current YES ask
        payload = _make_order_payload(
            ticker, side, remaining_qty, price,
            reduce_only=True,
            post_only=False,
            tif="immediate_or_cancel",
        )
        body, timing = client.post("/portfolio/events/orders", payload)
        oid = str(body.get("order_id") or "")
        fills, _ = _fills(client, oid) if oid else ([], {})
        _, fq, _ = _weighted_fill_price(fills)
        remaining_qty = max(0.0, remaining_qty - fq)
        forced_orders.append({
            "attempt": attempt,
            "order_id": oid,
            "side": side,
            "price": price,
            "requested_qty": float(payload["count"]),
            "filled_qty": fq,
            "remaining_qty_after": remaining_qty,
            "response": body,
            "timing": timing,
            "fills": fills,
        })
        if remaining_qty > 1e-9:
            time.sleep(0.25)
    record["forced_flatten_orders"] = forced_orders
    if remaining_qty > 1e-9:
        raise RuntimeError(
            f"CRITICAL: probe still tracks {remaining_qty:.4f} unflattened contracts in {ticker}. "
            "Check Kalshi immediately."
        )
    return remaining_qty


def run_live_q1_queue_probe(
    *,
    confirm_live=False,
    max_wait_s=MAX_DISCOVERY_WAIT_S,
    entry_wait_s=ENTRY_WAIT_S,
    exit_wait_s=EXIT_WAIT_S,
    show=True,
):
    """Run one production Q1 queue probe.

    The probe intentionally waits for M5+30s on a frozen-universe contract, so
    an active M0-M5 OOS run remains observationally clean.
    """
    if confirm_live is not True:
        raise RuntimeError("Real order submission disabled. Re-run with confirm_live=True only if you intend to send a real Q1 order.")

    client = LiveClient()
    balance_body, balance_timing = client.get(
        "/portfolio/balance",
        params={"subaccount": 0, "exchange_index": 0},
    )
    balance = _f(balance_body.get("balance_dollars"))
    if not np.isfinite(balance) or balance < MIN_BALANCE_DOLLARS:
        raise RuntimeError(f"Available balance ${balance:.2f} is below probe guardrail ${MIN_BALANCE_DOLLARS:.2f}.")

    if show:
        print("LIVE Q1 QUEUE PROBE")
        print(f"Available balance: ${balance:.2f}")
        print("Size hard cap: 1.00 contract")
        print("Waiting only for M5+30s through M12 so frozen M0-M5 OOS is not touched.")

    chosen = _choose_probe_market(client, max_wait_s=max_wait_s, show=show)
    ticker = chosen["ticker"]
    series = chosen["series"]
    elapsed = _f(chosen["market"].get("elapsed_s"))

    if elapsed < SAFE_AFTER_OOS_S - 1e-9:
        raise RuntimeError("Safety guard: selected contract is inside OOS strategy/label window.")
    if not _fee_ok(series):
        raise RuntimeError(f"Selected series {series} no longer has the expected quadratic fee type.")

    pos, pos_row = _get_position(client, ticker)
    if abs(pos) > 1e-9:
        raise RuntimeError(f"Refusing probe: pre-existing position {pos} in {ticker}: {pos_row}")
    resting = _resting_orders(client, ticker)
    if resting:
        raise RuntimeError(f"Refusing probe: pre-existing resting order(s) in {ticker}.")

    record = {
        "study_version": STUDY_VERSION,
        "started_at": _iso_now(),
        "real_money": True,
        "quantity_cap": QTY,
        "balance_before": balance_body,
        "balance_query_timing": balance_timing,
        "ticker": ticker,
        "series": series,
        "elapsed_s_at_selection": elapsed,
        "oos_safety": "selected only after M5+30s; frozen OOS strategy and 30s tail already complete for this contract",
        "selection": chosen,
    }
    contam = {
        "time": _iso_now(),
        "ticker": ticker,
        "series": series,
        "elapsed_s": elapsed,
        "event": "REAL_Q1_QUEUE_PROBE_CONTAMINATION",
        "instruction": "exclude/flag this contract from full15 exploratory inference after probe time",
    }
    record["full15_contamination_marker"] = _mark_full15_contamination(contam)

    # ENTRY
    entry = _submit_and_measure(
        client,
        ticker=ticker,
        side=chosen["side"],
        qty=QTY,
        price=chosen["price"],
        predicted_queue=chosen["predicted_queue"],
        reduce_only=False,
        label="ENTRY",
    )
    record["entry"] = entry

    entry_order = _wait_for_any_fill(client, entry["order_id"], entry_wait_s)
    entry["order_after_wait"] = entry_order
    entry_filled = _f(entry_order.get("fill_count_fp"), 0.0)
    entry_remaining = _f(entry_order.get("remaining_count_fp"), 0.0)

    if entry_filled <= 1e-12:
        entry["cancel"] = _cancel_if_remaining(client, entry["order_id"])
        record["outcome"] = "ENTRY_NOT_FILLED"
        record["finished_at"] = _iso_now()
        out = _save_record(record)
        print("\nEntry did not fill within timeout. Resting order canceled; no position opened.")
        print("Queue/latency measurement is still valid.")
        print("Saved:", out)
        return record

    # Frozen strategy convention: after any entry fill, cancel entry residual.
    if entry_remaining > 1e-12:
        entry["cancel_after_partial_fill"] = _cancel_if_remaining(client, entry["order_id"])

    entry_fills, _ = _fills(client, entry["order_id"])
    entry_px, entry_fill_qty, entry_fees = _weighted_fill_price(entry_fills)
    if entry_fill_qty <= 1e-12:
        entry_fill_qty = entry_filled
    entry["fills"] = entry_fills
    entry["weighted_fill_price"] = entry_px
    entry["filled_qty"] = entry_fill_qty
    entry["fees"] = entry_fees
    print(f"\nENTRY FILLED: {entry_fill_qty:.2f} @ {entry_px:.4f}" if np.isfinite(entry_px) else f"\nENTRY FILLED: {entry_fill_qty:.2f}")

    # PASSIVE EXIT
    cur_exit, exit_book_timing = _current_book(client, ticker)
    exit_side = "ask" if chosen["side"] == "bid" else "bid"
    exit_price = cur_exit["ask"] if exit_side == "ask" else cur_exit["bid"]
    exit_q = cur_exit["ask_q1"] if exit_side == "ask" else cur_exit["bid_q1"]
    exit_rec = _submit_and_measure(
        client,
        ticker=ticker,
        side=exit_side,
        qty=entry_fill_qty,
        price=exit_price,
        predicted_queue=exit_q,
        reduce_only=True,
        label="EXIT",
    )
    exit_rec["book_timing"] = exit_book_timing
    record["exit"] = exit_rec

    exit_order = _wait_for_any_fill(client, exit_rec["order_id"], exit_wait_s)
    exit_rec["order_after_wait"] = exit_order
    exit_filled = _f(exit_order.get("fill_count_fp"), 0.0)
    exit_remaining = max(0.0, entry_fill_qty - exit_filled)
    if _f(exit_order.get("remaining_count_fp"), 0.0) > 1e-12:
        exit_rec["cancel"] = _cancel_if_remaining(client, exit_rec["order_id"])

    exit_fills, _ = _fills(client, exit_rec["order_id"])
    exit_px_w, exit_fill_qty_rest, exit_fees = _weighted_fill_price(exit_fills)
    exit_rec["fills"] = exit_fills
    exit_rec["weighted_fill_price"] = exit_px_w
    exit_rec["filled_qty"] = exit_fill_qty_rest
    exit_rec["fees"] = exit_fees
    exit_remaining = max(0.0, entry_fill_qty - exit_fill_qty_rest)

    if exit_remaining > 1e-9:
        print(f"\nPassive exit left {exit_remaining:.2f}; flattening reduce-only with IOC at current touch...")
        _force_flatten(client, ticker, chosen["side"], exit_remaining, record)

    # Gather all exit fills, including forced IOC.
    all_exit_fills = list(exit_fills)
    for fo in record.get("forced_flatten_orders") or []:
        all_exit_fills.extend(fo.get("fills") or [])
    all_exit_px, all_exit_qty, all_exit_fees = _weighted_fill_price(all_exit_fills)

    gross = np.nan
    if np.isfinite(entry_px) and np.isfinite(all_exit_px) and all_exit_qty > 1e-12:
        matched_qty = min(entry_fill_qty, all_exit_qty)
        if chosen["side"] == "bid":
            gross = (all_exit_px - entry_px) * matched_qty
        else:
            gross = (entry_px - all_exit_px) * matched_qty
    total_fees = entry_fees + all_exit_fees
    net = gross - total_fees if np.isfinite(gross) else np.nan

    # Position endpoint can lag; poll briefly for a flat confirmation.
    final_pos = np.nan
    final_pos_row = None
    for _ in range(10):
        try:
            final_pos, final_pos_row = _get_position(client, ticker)
            if abs(final_pos) <= 1e-9:
                break
        except Exception:
            pass
        time.sleep(0.5)

    record.update({
        "all_exit_weighted_price": all_exit_px,
        "all_exit_qty": all_exit_qty,
        "all_exit_fees": all_exit_fees,
        "gross_roundtrip_pnl": gross,
        "total_fees": total_fees,
        "net_roundtrip_pnl": net,
        "final_position_fp": final_pos,
        "final_position_row": final_pos_row,
        "outcome": "ROUNDTRIP_ATTEMPT_COMPLETE",
        "finished_at": _iso_now(),
    })

    balance_after, _ = client.get(
        "/portfolio/balance",
        params={"subaccount": 0, "exchange_index": 0},
    )
    record["balance_after"] = balance_after
    out = _save_record(record)

    print("\n" + "=" * 72)
    print("LIVE Q1 PROBE RESULT")
    print("=" * 72)
    print(f"Ticker: {ticker}")
    print(f"Entry queue snapshot -> actual: {entry['predicted_queue_ahead']:.2f} -> " +
          (f"{entry['actual_queue_position_first']:.2f}" if np.isfinite(entry['actual_queue_position_first']) else "n/a"))
    if np.isfinite(entry["extra_queue_vs_snapshot"]):
        print(f"Entry extra queue: {entry['extra_queue_vs_snapshot']:+.2f}")
    print(f"Exit queue snapshot -> actual: {exit_rec['predicted_queue_ahead']:.2f} -> " +
          (f"{exit_rec['actual_queue_position_first']:.2f}" if np.isfinite(exit_rec['actual_queue_position_first']) else "n/a"))
    if np.isfinite(exit_rec["extra_queue_vs_snapshot"]):
        print(f"Exit extra queue: {exit_rec['extra_queue_vs_snapshot']:+.2f}")
    print(f"Entry create RTT: {entry['create_timing']['rtt_ms']:.1f} ms")
    print(f"Exit create RTT:  {exit_rec['create_timing']['rtt_ms']:.1f} ms")
    print(f"Gross roundtrip PnL: {gross:+.4f}" if np.isfinite(gross) else "Gross roundtrip PnL: n/a")
    print(f"Fees: ${total_fees:.4f}")
    print(f"Net roundtrip PnL: ${net:+.4f}" if np.isfinite(net) else "Net roundtrip PnL: n/a")
    print(f"Final position_fp: {final_pos}")
    print("Saved:", out)
    print("=" * 72)
    return record


def _save_record(record):
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    ticker = str(record.get("ticker") or "UNKNOWN").replace("/", "_")
    path = RESULT_ROOT / f"{stamp}_{ticker}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
