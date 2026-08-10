from __future__ import annotations

import json
import re
import threading
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import shadow_trader as S

# This module intentionally patches ShadowTrader to match the final notebook shadow
# implementation + the user's trade-parser and full-book-parser patches.

_TICKER_CLOSE_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{6})-\d{2}$")


def _parse_ts(x):
    if x is None:
        return pd.NaT
    try:
        if isinstance(x, (int, float)):
            z = float(x)
            if z > 1e17:
                return pd.to_datetime(int(z), unit="ns", utc=True)
            if z > 1e14:
                return pd.to_datetime(int(z), unit="us", utc=True)
            if z > 1e11:
                return pd.to_datetime(int(z), unit="ms", utc=True)
            return pd.to_datetime(z, unit="s", utc=True)
        return pd.to_datetime(x, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def _dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _dicts(value)


def _first_value(obj, keys):
    for d in _dicts(obj):
        for key in keys:
            if key in d and d[key] is not None:
                return d[key]
    return None


def _exact_nested(obj, key):
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for value in obj.values():
            out = _exact_nested(value, key)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for value in obj:
            out = _exact_nested(value, key)
            if out is not None:
                return out
    return None


def _event_ts(obj):
    # Exact notebook precedence: receipt/event timestamp at the top level first.
    if isinstance(obj, dict):
        for key in [
            "received_ts", "recv_ts", "received_at", "timestamp", "ts",
            "time", "created_time",
        ]:
            if key in obj:
                out = _parse_ts(obj[key])
                if not pd.isna(out):
                    return out
    out = _parse_ts(_first_value(obj, ["timestamp", "ts", "time", "created_time"]))
    return pd.Timestamp.now(tz="UTC") if pd.isna(out) else out


def _ticker(obj):
    value = _first_value(obj, ["ticker", "market_ticker", "marketTicker"])
    return None if value is None else str(value)


def _close_from_ticker(ticker):
    if not ticker:
        return pd.NaT
    match = _TICKER_CLOSE_RE.search(str(ticker))
    if not match:
        return pd.NaT
    try:
        local = pd.to_datetime(match.group(1), format="%y%b%d%H%M")
        local = local.tz_localize(ZoneInfo("America/New_York"))
        return local.tz_convert("UTC")
    except Exception:
        return pd.NaT


def _quote(obj):
    bid_keys = ["yes_bid", "yes_bid_dollars", "best_yes_bid", "best_bid"]
    ask_keys = ["yes_ask", "yes_ask_dollars", "best_yes_ask", "best_ask"]
    for d in _dicts(obj):
        bid = next((d[k] for k in bid_keys if k in d), None)
        ask = next((d[k] for k in ask_keys if k in d), None)
        if bid is None or ask is None:
            continue
        b, a = S._price(bid), S._price(ask)
        if np.isfinite(b) and np.isfinite(a):
            return b, a
    return None


def _result(obj):
    value = _first_value(obj, ["result", "settlement_result", "outcome"])
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"yes", "y", "1", "true"}:
        return "YES"
    if s in {"no", "n", "0", "false"}:
        return "NO"
    return None


def _levels(x):
    # IMPORTANT: this intentionally rounds book levels to TWO decimals, exactly
    # like sh_levels() in the patched notebook. This affects queue-ahead on
    # tapered/deci-cent books and therefore can change fills.
    out = {}
    if x is None:
        return out
    if isinstance(x, dict):
        scalar_map = all(not isinstance(v, (dict, list, tuple)) for v in x.values())
        if scalar_map:
            for k, v in x.items():
                p, q = S._price(k), S._num(v)
                if np.isfinite(p) and np.isfinite(q):
                    out[round(p, 2)] = float(q)
            return out
    if isinstance(x, list):
        for row in x:
            if isinstance(row, (list, tuple)):
                if len(row) < 2:
                    continue
                p, q = S._price(row[0]), S._num(row[1])
            elif isinstance(row, dict):
                p = S._price(row.get("price", row.get("price_dollars", row.get("yes_price"))))
                q = S._num(row.get("quantity", row.get("count", row.get("size"))))
            else:
                continue
            if np.isfinite(p) and np.isfinite(q):
                out[round(p, 2)] = float(q)
    return out


def _on_ticker(self, obj):
    ticker = _ticker(obj)
    if not ticker:
        return
    ts = _event_ts(obj)
    quote = _quote(obj)
    result = _result(obj)
    close_time = _close_from_ticker(ticker)
    if not pd.isna(close_time):
        self.close_times[ticker] = close_time
    with self.lock:
        if quote is not None:
            self.quotes[ticker].append((ts, quote[0], quote[1]))
            pos = self.positions.get(ticker)
            if (
                pos is not None and not pos["settled"] and pos["open_qty"] > 0
                and ticker not in self.exit_orders and pd.isna(pos["trigger_time"])
            ):
                _, held_ask = self._held_quote(pos["direction"], quote[0], quote[1])
                if held_ask <= S.EXIT_PRICE + 1e-12:
                    pos["trigger_time"] = ts
                    pos["confirm_time"] = ts.floor("min") + pd.Timedelta(minutes=1)
                    self._record(ticker)["exit_trigger"] = ts
                    self.emit(
                        "EXIT_TRIGGER", ticker,
                        f"held ask={100*held_ask:.1f}c | confirm={pos['confirm_time']}"
                    )
        if result is not None:
            self.results[ticker] = result
            self._try_settle(ticker)


def _on_book(self, obj):
    ticker = _ticker(obj)
    if not ticker:
        return
    parsed = None
    # Exact final notebook patch: require recorder yes_bids + yes_asks and
    # synthesize NO bids from YES asks.
    for d in _dicts(obj):
        if "yes_bids" not in d or "yes_asks" not in d:
            continue
        yes_bids = _levels(d["yes_bids"])
        yes_asks = _levels(d["yes_asks"])
        if not yes_bids and not yes_asks:
            continue
        no_bids = {}
        for yes_ask_price, qty in yes_asks.items():
            no_price = round(1.0 - float(yes_ask_price), 3)
            if 0.0 <= no_price <= 1.0 and np.isfinite(qty):
                no_bids[no_price] = float(qty)
        parsed = yes_bids, no_bids
        break
    if parsed is None:
        return
    ts = _event_ts(obj)
    with self.lock:
        self.books[ticker].append((ts, parsed[0], parsed[1]))


def _extract_trade(self, obj):
    ticker = _ticker(obj)
    if not ticker:
        return None

    price = _exact_nested(obj, "yes_price")
    if price is None:
        price = _exact_nested(obj, "yes_price_dollars")

    # Exact user's patch: prefer raw count_fp, then fallbacks.
    qty = _exact_nested(obj, "count_fp")
    if qty is None:
        for key in ["count", "quantity_fp", "quantity", "size_fp", "size", "contracts"]:
            qty = _exact_nested(obj, key)
            if qty is not None:
                break

    # Exact user's patch: taker_book_side only; never fall back to outcome side.
    side = _exact_nested(obj, "taker_book_side")
    if price is None or qty is None or side is None:
        return None

    p, q = S._price(price), S._num(qty)
    side = str(side).lower()
    if not np.isfinite(p) or not np.isfinite(q) or q <= 0 or side not in {"bid", "ask"}:
        return None

    ts = _event_ts(obj)
    trade_id = _exact_nested(obj, "trade_id")
    if trade_id is None:
        trade_id = _exact_nested(obj, "id")
    key = str(trade_id) if trade_id is not None else f"{ticker}|{ts}|{p:.4f}|{q:.8f}|{side}"
    return {
        "ticker": ticker,
        "ts": ts,
        "yes_price": float(p),
        "qty": float(q),
        "taker_book_side": side,
        "key": key,
    }


def _scheduler(self):
    while not self.stop_event.is_set():
        try:
            now = pd.Timestamp.now(tz="UTC")
            with self.lock:
                for ticker in list(self.quotes.keys()):
                    if ticker in self.decisions_done or ticker.split("-")[0] not in S.FROZEN_SERIES:
                        continue
                    close_time = _close_from_ticker(ticker)
                    if pd.isna(close_time):
                        continue
                    decision_time = close_time - pd.Timedelta(minutes=10)

                    # Exact notebook prospectivity behavior.
                    if decision_time < self.started_at:
                        continue

                    if now >= decision_time:
                        lateness = (now - decision_time).total_seconds()
                        if lateness > 5:
                            self.decisions_done.add(ticker)
                            self.emit("SKIP", ticker, f"shadow scheduler late by {lateness:.2f}s")
                            continue
                        self._process_decision(ticker, decision_time)

                for ticker, order in list(self.entry_orders.items()):
                    if order["status"] == "OPEN" and now >= order["cancel_time"]:
                        order["status"] = "CANCELLED"
                        rec = self._record(ticker)
                        rec["status"] = "POSITION_OPEN" if order["filled_qty"] > 0 else "NO_FILL"
                        self.emit(
                            "ENTRY_CANCEL", ticker,
                            f"filled={order['filled_qty']:.2f}/{S.QTY:g} | cancelled={order['remaining']:.2f}"
                        )
                        self._save_state()

                for ticker, pos in list(self.positions.items()):
                    if pos["settled"] or pos["open_qty"] <= 0 or ticker in self.exit_orders:
                        continue
                    self._check_exit_confirmation(ticker, pos, now)

                for ticker in list(self.positions):
                    self._try_settle(ticker)
        except Exception as exc:
            self.emit("ERROR", detail=f"scheduler: {exc!r}")
            traceback.print_exc()
        self.stop_event.wait(0.10)


def _start(self):
    if any(t.is_alive() for t in self.threads):
        print("Shadow trader is already running.")
        return self

    print(f"Shadow session: {self.session_dir}\nShadow output: {self.out_dir}")

    # Match notebook startup state: seed quote + full-book state, but do not
    # preload recent trades into the live fill buffer.
    for obj in S._read_recent_jsonl(self.ticker_file, 16 * 1024 * 1024):
        self._on_ticker(obj)
    for obj in S._read_recent_jsonl(self.book_file, 64 * 1024 * 1024):
        self._on_book(obj)

    self._seed_coinbase()

    specs = [
        (self._follow_file, (self.ticker_file, self._on_ticker, "ticker"), "shadow-ticker"),
        (self._follow_file, (self.book_file, self._on_book, "book"), "shadow-book"),
        (self._follow_file, (self.trades_file, self._on_trade, "trades"), "shadow-trades"),
        (self._scheduler, (), "shadow-scheduler"),
        (self._coinbase_loop, (), "shadow-coinbase"),
        # Keep the later settlement helper used by the final notebook workflow.
        (self._settlement_loop, (), "shadow-settlement"),
    ]
    self.threads.clear()
    for target, args, name in specs:
        th = threading.Thread(target=target, args=args, name=name, daemon=True)
        th.start()
        self.threads.append(th)

    print("Shadow trader running: -3c / 15s / 3 contracts | 10c candle-confirmed passive salvage | READ-ONLY")
    return self


# Apply exact-parity methods.
S.ShadowTrader._on_ticker = _on_ticker
S.ShadowTrader._on_book = _on_book
S.ShadowTrader._extract_trade = _extract_trade
S.ShadowTrader._scheduler = _scheduler
S.ShadowTrader.start = _start

PARITY_VERSION = "NOTEBOOK_FINAL_PLUS_TRADE_BOOK_PATCHES_V1"
