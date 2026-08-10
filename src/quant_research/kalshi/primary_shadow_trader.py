from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

FROZEN_SERIES = {
    "KXBNB15M", "KXDOGE15M", "KXETH15M", "KXHYPE15M",
    "KXNEAR15M", "KXSOL15M", "KXXRP15M", "KXZEC15M",
}
QTY = 3.0
DEPTH_C = 3.0
LIFETIME_SEC = 15.0
MAX_SPREAD_C = 2.0
MAX_SCHEDULER_LATENESS_SEC = 1.0
MAX_QUOTE_AGE_SEC = 5.0
MAX_BOOK_SOURCE_AGE_SEC = 5.0

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60"
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
KALSHI_REST_BASE = "https://external-api.kalshi.com/trade-api/v2"
_TICKER_CLOSE_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{6})-\d{2}$")


def _ts(x):
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


def _num(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _price(x):
    p = _num(x)
    if np.isfinite(p) and p > 1.5:
        p /= 100.0
    return p


def _read_recent_jsonl(path: Path, max_bytes: int):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("rb") as f:
        size = f.seek(0, os.SEEK_END)
        start = max(0, size - max_bytes)
        f.seek(start)
        if start:
            f.readline()
        raw = f.read().decode("utf-8", errors="ignore")
    out = []
    for line in raw.splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _close_from_ticker(ticker):
    match = _TICKER_CLOSE_RE.search(str(ticker or ""))
    if not match:
        return pd.NaT
    try:
        local = pd.to_datetime(match.group(1), format="%y%b%d%H%M")
        local = local.tz_localize(ZoneInfo("America/New_York"))
        return local.tz_convert("UTC")
    except Exception:
        return pd.NaT


def _event_ts(obj):
    for key in ("received_ts", "recv_ts", "received_at", "time", "timestamp", "ts", "created_time"):
        if isinstance(obj, dict) and obj.get(key) is not None:
            out = _ts(obj.get(key))
            if not pd.isna(out):
                return out
    return pd.NaT


def _exchange_ts(obj):
    if not isinstance(obj, dict):
        return pd.NaT
    raw = obj.get("raw_msg") if isinstance(obj.get("raw_msg"), dict) else {}
    for value in (obj.get("ts_ms"), raw.get("ts_ms"), raw.get("timestamp"), raw.get("time")):
        out = _ts(value)
        if not pd.isna(out):
            return out
    return pd.NaT


def _get_ticker(obj):
    if not isinstance(obj, dict):
        return None
    raw = obj.get("raw_msg") if isinstance(obj.get("raw_msg"), dict) else {}
    value = obj.get("ticker") or obj.get("market_ticker") or raw.get("market_ticker")
    return None if value is None else str(value)


def _levels(rows):
    out = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        p, q = _price(row[0]), _num(row[1])
        if np.isfinite(p) and np.isfinite(q) and q > 0:
            out[round(float(p), 2)] = float(q)
    return out


def _historical_btc_return_from_candles(candles, when):
    when = pd.Timestamp(when)
    when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    rows = []
    for row in candles:
        if len(row) >= 5:
            rows.append((pd.to_datetime(row[0], unit="s", utc=True) + pd.Timedelta(minutes=1), float(row[4])))
    rows.sort(key=lambda z: z[0])

    def at(t):
        ans = None
        for available, close in rows:
            if available <= t:
                ans = (available, close)
            else:
                break
        return ans

    a = at(when)
    b = at(when - pd.Timedelta(minutes=15))
    return None if a is None or b is None else a[1] / b[1] - 1.0


class PrimaryShadowTrader:
    """Frozen confirmatory shadow. All Kalshi actions are simulated/read-only."""

    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        self.ticker_file = self.session_dir / "ticker_updates.jsonl"
        self.trades_file = self.session_dir / "trades.jsonl"
        self.book_file = self.session_dir / "full_books.jsonl"
        for path in (self.ticker_file, self.trades_file, self.book_file):
            if not path.exists():
                raise FileNotFoundError(f"Missing recorder file: {path}")

        self.out_dir = self.session_dir / "PRIMARY_SHADOW_M5_MINUS3C_15S_3CT_HOLD_V1"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.event_file = self.out_dir / "shadow_events.jsonl"
        self.state_file = self.out_dir / "shadow_records.csv"

        self.lock = threading.RLock()
        self.log_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.started_at = pd.Timestamp.now(tz="UTC")
        self.quotes = defaultdict(lambda: deque(maxlen=10000))
        self.books = defaultdict(lambda: deque(maxlen=1800))
        self.trades = defaultdict(lambda: deque(maxlen=50000))
        self.btc = deque(maxlen=10000)
        self._btc_bucket_start = None
        self._btc_bucket_close = None
        self.close_times = {}
        self.decisions_done = set()
        self.entry_orders = {}
        self.positions = {}
        self.results = {}
        self.records = {}
        self.threads = []
        self._rest = requests.Session()

    def emit(self, event, ticker=None, detail=None, **extra):
        row = {"time": pd.Timestamp.now(tz="UTC").isoformat(), "event": event, "ticker": ticker, "detail": detail, **extra}
        with self.log_lock:
            with self.event_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
        if event in {"SIGNAL", "ENTRY_POST", "ENTRY_FILL", "ENTRY_CANCEL", "DATA_INVALID", "SETTLED", "ERROR"}:
            prefix = pd.Timestamp.now(tz="UTC").strftime("[%H:%M:%S UTC]")
            print(f"{prefix} {event}" + (f" | {ticker}" if ticker else "") + (f" | {detail}" if detail else ""))

    def _record(self, ticker):
        if ticker not in self.records:
            self.records[ticker] = {
                "ticker": ticker, "series": ticker.split("-")[0], "decision_time": pd.NaT,
                "actual_post_time": pd.NaT, "cancel_time": pd.NaT, "direction": None,
                "midpoint": np.nan, "spread_c": np.nan, "btc_return": np.nan,
                "btc_now_time": pd.NaT, "btc_past_time": pd.NaT, "entry_price": np.nan,
                "entry_queue": np.nan, "qty": QTY, "quote_time": pd.NaT,
                "quote_exchange_time": pd.NaT, "quote_age_s": np.nan, "book_sample_time": pd.NaT,
                "book_source_time": pd.NaT, "book_exchange_time": pd.NaT, "book_age_s": np.nan,
                "book_seq": None, "connection_epoch": None, "entry_fill_qty": 0.0,
                "entry_first_fill": pd.NaT, "entry_last_fill": pd.NaT, "result": None,
                "realized_pnl": 0.0, "status": None, "data_invalid_reason": None,
            }
        return self.records[ticker]

    def _save_state(self):
        with self.lock:
            df = pd.DataFrame(list(self.records.values()))
        if len(df):
            tmp = self.state_file.with_suffix(".tmp")
            df.sort_values("decision_time", na_position="last").to_csv(tmp, index=False)
            tmp.replace(self.state_file)

    @staticmethod
    def _held_quote(direction, yes_bid, yes_ask):
        return (yes_bid, yes_ask) if direction == "YES" else (1.0 - yes_ask, 1.0 - yes_bid)

    def _quote_at(self, ticker, when):
        for row in reversed(self.quotes.get(ticker, ())):
            if row["receipt_time"] <= when:
                return row
        return None

    def _book_at(self, ticker, when):
        for row in reversed(self.books.get(ticker, ())):
            if row["sample_time"] <= when:
                return row
        return None

    def _recorder_health(self):
        try:
            from . import recorder as R
            return R.recorder_health_snapshot()
        except Exception as exc:
            return {"running": False, "healthy": False, "connection_epoch": None, "error": repr(exc)}

    def _on_ticker(self, obj):
        ticker = _get_ticker(obj)
        if not ticker:
            return
        receipt_time = _event_ts(obj)
        if pd.isna(receipt_time):
            return
        bid, ask = _price(obj.get("yes_bid_dollars")), _price(obj.get("yes_ask_dollars"))
        close_time = _close_from_ticker(ticker)
        if not pd.isna(close_time):
            self.close_times[ticker] = close_time
        result = str(obj.get("result") or "").upper()
        with self.lock:
            if np.isfinite(bid) and np.isfinite(ask) and 0 <= bid <= ask <= 1:
                self.quotes[ticker].append({
                    "receipt_time": receipt_time, "exchange_time": _exchange_ts(obj),
                    "bid": float(bid), "ask": float(ask), "connection_epoch": obj.get("connection_epoch"),
                })
            if result in {"YES", "NO"}:
                self.results[ticker] = result
                self._try_settle(ticker)

    def _on_book(self, obj):
        ticker = _get_ticker(obj)
        sample_time = _event_ts(obj)
        if not ticker or pd.isna(sample_time):
            return
        yes_bids = _levels(obj.get("yes_bids") or [])
        yes_asks = _levels(obj.get("yes_asks") or [])
        if not yes_bids and not yes_asks:
            return
        no_bids = {round(1.0 - p, 2): q for p, q in yes_asks.items() if 0 <= 1.0 - p <= 1.0}
        with self.lock:
            self.books[ticker].append({
                "sample_time": sample_time, "source_time": _ts(obj.get("book_source_time")),
                "exchange_time": _ts(obj.get("book_exchange_time")), "seq": obj.get("book_seq"),
                "connection_epoch": obj.get("connection_epoch"), "yes_bids": yes_bids, "no_bids": no_bids,
            })

    def _extract_trade(self, obj):
        ticker = _get_ticker(obj)
        if not ticker:
            return None
        raw = obj.get("raw_msg") if isinstance(obj.get("raw_msg"), dict) else {}
        p = _price(obj.get("yes_price") if obj.get("yes_price") is not None else raw.get("yes_price_dollars"))
        qty = _num(raw.get("count_fp") if raw.get("count_fp") is not None else obj.get("qty"))
        side = str(raw.get("taker_book_side") or obj.get("taker_book_side") or "").lower()
        ts = _event_ts(obj)
        trade_id = raw.get("trade_id") or obj.get("trade_id")
        if not np.isfinite(p) or not np.isfinite(qty) or qty <= 0 or side not in {"bid", "ask"} or pd.isna(ts):
            return None
        key = str(trade_id) if trade_id is not None else f"{ticker}|{ts}|{p:.4f}|{qty:.8f}|{side}"
        return {"ticker": ticker, "ts": ts, "yes_price": float(p), "qty": float(qty), "taker_book_side": side, "key": key}

    def _on_trade(self, obj):
        trade = self._extract_trade(obj)
        if trade is None:
            return
        ticker = trade["ticker"]
        with self.lock:
            self.trades[ticker].append(trade)
            if ticker in self.entry_orders:
                self._apply_entry_trade(self.entry_orders[ticker], trade)

    def _entry_queue(self, direction, entry_price, book):
        levels = book["yes_bids"] if direction == "YES" else book["no_bids"]
        return float(levels.get(round(entry_price, 2), 0.0))

    def _apply_entry_trade(self, order, trade):
        if order["status"] != "OPEN" or trade["key"] in order["seen"]:
            return
        order["seen"].add(trade["key"])
        ts = trade["ts"]
        if ts < order["order_time"] or ts > order["cancel_time"]:
            return
        q, p, qty, side = order["price"], trade["yes_price"], trade["qty"], trade["taker_book_side"]
        exact = through = False
        if order["direction"] == "YES":
            if side != "ask":
                return
            through, exact = p < q - 1e-9, abs(p - q) < 1e-9
        else:
            yes_equiv = 1.0 - q
            if side != "bid":
                return
            through, exact = p > yes_equiv + 1e-9, abs(p - yes_equiv) < 1e-9
        fill_qty = order["remaining"] if through else 0.0
        if exact:
            queue = order["queue_ahead"]
            if not np.isfinite(queue):
                return
            if qty <= queue:
                order["queue_ahead"] = queue - qty
                return
            order["queue_ahead"] = 0.0
            fill_qty = min(order["remaining"], qty - queue)
        if fill_qty <= 0:
            return
        order["remaining"] -= fill_qty
        order["filled_qty"] += fill_qty
        rec = self._record(order["ticker"])
        rec["entry_fill_qty"] += fill_qty
        if pd.isna(rec["entry_first_fill"]):
            rec["entry_first_fill"] = ts
        rec["entry_last_fill"] = ts
        ticker = order["ticker"]
        if ticker not in self.positions:
            self.positions[ticker] = {
                "ticker": ticker, "direction": order["direction"], "entry_price": order["price"],
                "qty": 0.0, "open_qty": 0.0, "realized_pnl": 0.0, "settled": False,
            }
        pos = self.positions[ticker]
        pos["qty"] += fill_qty
        pos["open_qty"] += fill_qty
        rec["status"] = "POSITION_OPEN"
        reason = "TRADED_THROUGH" if through else "QUEUE_CONSUMED"
        self.emit(
            "ENTRY_FILL", ticker, f"{fill_qty:.2f}ctrs @ {100*q:.0f}c | {reason}",
            fill_qty=float(fill_qty), entry_price=float(q), fill_time=ts, reason=reason,
            total_filled=float(order["filled_qty"]), qty=float(QTY),
        )
        if order["remaining"] <= 1e-12:
            order["status"] = "FILLED"
        self._save_state()

    def _try_settle(self, ticker):
        pos = self.positions.get(ticker)
        result = self.results.get(ticker)
        if pos is None or pos["settled"] or result not in {"YES", "NO"}:
            return
        remaining = float(pos["open_qty"])
        settlement_value = 1.0 if result == pos["direction"] else 0.0
        pos["realized_pnl"] += remaining * (settlement_value - pos["entry_price"])
        pos["open_qty"] = 0.0
        pos["settled"] = True
        rec = self._record(ticker)
        rec.update({"result": result, "realized_pnl": pos["realized_pnl"], "status": "SETTLED"})
        self.emit(
            "SETTLED", ticker,
            f"result={result} | held={pos['direction']} | qty={pos['qty']:.2f} | final pnl=${pos['realized_pnl']:+.4f}",
            result=result, held_direction=pos["direction"], qty=float(pos["qty"]),
            entry_price=float(pos["entry_price"]), final_pnl=float(pos["realized_pnl"]),
        )
        self._save_state()

    def _btc_on_tick(self, ts, price):
        ts, price = _ts(ts), _num(price)
        if pd.isna(ts) or not np.isfinite(price) or price <= 0:
            return
        bucket = ts.floor("min")
        with self.lock:
            if self._btc_bucket_start is None:
                self._btc_bucket_start, self._btc_bucket_close = bucket, float(price)
                return
            if bucket == self._btc_bucket_start:
                self._btc_bucket_close = float(price)
                return
            if bucket > self._btc_bucket_start:
                available = self._btc_bucket_start + pd.Timedelta(minutes=1)
                if self.btc and available == self.btc[-1][0]:
                    self.btc[-1] = (available, float(self._btc_bucket_close))
                elif not self.btc or available > self.btc[-1][0]:
                    self.btc.append((available, float(self._btc_bucket_close)))
                self._btc_bucket_start, self._btc_bucket_close = bucket, float(price)

    def _btc_price_at(self, when):
        for available_time, close in reversed(self.btc):
            if available_time <= when:
                return available_time, close
        return None

    def _btc_return_15m(self, when):
        a = self._btc_price_at(when)
        b = self._btc_price_at(when - pd.Timedelta(minutes=15))
        return None if a is None or b is None else (a[1] / b[1] - 1.0, a, b)

    def _seed_coinbase(self):
        req = urllib.request.Request(COINBASE_CANDLES_URL, headers={"User-Agent": "kalshi-shadow-research/2.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            candles = json.loads(response.read().decode("utf-8"))
        now = pd.Timestamp.now(tz="UTC")
        seed = []
        for row in candles:
            if len(row) >= 5:
                available = pd.to_datetime(row[0], unit="s", utc=True) + pd.Timedelta(minutes=1)
                if available <= now:
                    seed.append((available, float(row[4])))
        seed.sort(key=lambda x: x[0])
        with self.lock:
            self.btc.clear()
            self.btc.extend(seed)
        checks = mismatches = 0
        for available, _ in seed[-120:]:
            live = self._btc_return_15m(available)
            ref = _historical_btc_return_from_candles(candles, available)
            if live is None or ref is None:
                continue
            checks += 1
            if abs(float(live[0]) - float(ref)) > 1e-12:
                mismatches += 1
        self.emit(
            "BTC_PARITY", detail=f"completed-minute causal parity: {checks - mismatches}/{checks}",
            checks=checks, mismatches=mismatches, method="completed_1m_close_available_at_bucket_end",
        )
        if checks == 0 or mismatches:
            raise RuntimeError(f"BTC causal parity failed: checks={checks}, mismatches={mismatches}")
        print(f"Coinbase BTC history seeded: {len(seed)} completed 1m candles")
        print(f"BTC causal parity check: PASS ({checks}/{checks})")

    def _coinbase_loop(self):
        try:
            import websocket
        except Exception as exc:
            self.emit("ERROR", detail=f"Coinbase websocket dependency missing: {exc!r}")
            return
        while not self.stop_event.is_set():
            try:
                def on_open(ws):
                    ws.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]}))

                def on_message(ws, message):
                    try:
                        data = json.loads(message)
                        if data.get("type") == "ticker" and data.get("product_id") == "BTC-USD":
                            self._btc_on_tick(data.get("time"), data.get("price"))
                    except Exception:
                        pass

                websocket.WebSocketApp(COINBASE_WS_URL, on_open=on_open, on_message=on_message).run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self.emit("ERROR", detail=f"Coinbase websocket: {exc!r}")
            if not self.stop_event.is_set():
                self.stop_event.wait(2)

    def _mark_data_invalid(self, ticker, decision_time, reason, **fields):
        self.decisions_done.add(ticker)
        rec = self._record(ticker)
        rec["decision_time"] = decision_time
        rec["status"] = "DATA_INVALID"
        rec["data_invalid_reason"] = reason
        for key, value in fields.items():
            if key in rec:
                rec[key] = value
        self.emit("DATA_INVALID", ticker, reason, decision_time=decision_time, reason=reason, **fields)
        self._save_state()

    def _process_decision(self, ticker, decision_time, actual_post_time):
        health = self._recorder_health()
        if not health.get("healthy"):
            self._mark_data_invalid(
                ticker, decision_time, "recorder unhealthy at decision",
                connection_epoch=health.get("connection_epoch"), recorder_health=health,
            )
            return "DONE"
        current_epoch = health.get("connection_epoch")
        quote = self._quote_at(ticker, decision_time)
        if quote is None:
            self._mark_data_invalid(ticker, decision_time, "no causal quote at M5")
            return "DONE"
        quote_age = (decision_time - quote["receipt_time"]).total_seconds()
        if quote_age < -1e-9 or quote_age > MAX_QUOTE_AGE_SEC:
            self._mark_data_invalid(
                ticker, decision_time, f"quote age invalid: {quote_age:.3f}s",
                quote_time=quote["receipt_time"], quote_exchange_time=quote["exchange_time"],
                quote_age_s=quote_age, connection_epoch=quote.get("connection_epoch"),
            )
            return "DONE"
        if quote.get("connection_epoch") != current_epoch:
            self._mark_data_invalid(
                ticker, decision_time, "quote belongs to stale recorder connection epoch",
                quote_time=quote["receipt_time"], quote_age_s=quote_age,
                connection_epoch=quote.get("connection_epoch"),
            )
            return "DONE"

        yes_bid, yes_ask = quote["bid"], quote["ask"]
        spread = yes_ask - yes_bid
        midpoint = (yes_bid + yes_ask) / 2.0
        if spread > MAX_SPREAD_C / 100.0 + 1e-12:
            self.decisions_done.add(ticker)
            self.emit(
                "SKIP", ticker, f"spread={100*spread:.1f}c >2c", decision_time=decision_time,
                midpoint=float(midpoint), spread_c=float(100 * spread), quote_time=quote["receipt_time"],
                quote_age_s=float(quote_age), connection_epoch=current_epoch, reason="SPREAD_GT_2C",
            )
            return "DONE"
        if abs(midpoint - 0.50) < 1e-12:
            self.decisions_done.add(ticker)
            self.emit(
                "SKIP", ticker, "midpoint exactly 50c", decision_time=decision_time,
                midpoint=float(midpoint), spread_c=float(100 * spread), reason="MIDPOINT_50",
            )
            return "DONE"

        direction = "YES" if midpoint > 0.50 else "NO"
        sign = 1 if direction == "YES" else -1
        btc = self._btc_return_15m(decision_time)
        if btc is None:
            return "WAIT_BTC"
        btc_ret, btc_now, btc_past = btc
        if btc_ret * sign >= 0:
            self.decisions_done.add(ticker)
            self.emit(
                "SKIP", ticker, f"BTC not opposition | signal={direction} | btc15={10000*btc_ret:+.2f}bp",
                decision_time=decision_time, direction=direction, midpoint=float(midpoint),
                spread_c=float(100 * spread), btc_return=float(btc_ret), btc_now_time=btc_now[0],
                btc_past_time=btc_past[0], quote_time=quote["receipt_time"], quote_age_s=float(quote_age),
                connection_epoch=current_epoch, reason="BTC_NOT_OPPOSITION",
            )
            return "DONE"

        held_bid, held_ask = self._held_quote(direction, yes_bid, yes_ask)
        entry_price = round(held_bid - DEPTH_C / 100.0, 2)
        if entry_price < 0.01 or entry_price > 0.99 or entry_price >= held_ask:
            self.decisions_done.add(ticker)
            self.emit(
                "SKIP", ticker,
                f"illegal/non-passive -3c quote | bid={held_bid:.2f} ask={held_ask:.2f} entry={entry_price:.2f}",
                decision_time=decision_time, direction=direction, entry_price=float(entry_price),
                held_bid=float(held_bid), held_ask=float(held_ask), reason="ILLEGAL_NONPASSIVE_ENTRY",
            )
            return "DONE"

        book = self._book_at(ticker, actual_post_time)
        if book is None:
            self._mark_data_invalid(
                ticker, decision_time, "no causal full book at actual post time",
                actual_post_time=actual_post_time, connection_epoch=current_epoch,
            )
            return "DONE"
        source_time = book.get("source_time")
        if pd.isna(source_time):
            self._mark_data_invalid(
                ticker, decision_time, "book missing source-update timestamp",
                actual_post_time=actual_post_time, book_sample_time=book.get("sample_time"),
                connection_epoch=book.get("connection_epoch"),
            )
            return "DONE"
        book_age = (actual_post_time - source_time).total_seconds()
        if book_age < -1e-9 or book_age > MAX_BOOK_SOURCE_AGE_SEC:
            self._mark_data_invalid(
                ticker, decision_time, f"book source age invalid: {book_age:.3f}s",
                actual_post_time=actual_post_time, book_sample_time=book.get("sample_time"),
                book_source_time=source_time, book_exchange_time=book.get("exchange_time"),
                book_age_s=book_age, book_seq=book.get("seq"), connection_epoch=book.get("connection_epoch"),
            )
            return "DONE"
        if book.get("connection_epoch") != current_epoch:
            self._mark_data_invalid(
                ticker, decision_time, "book belongs to stale recorder connection epoch",
                actual_post_time=actual_post_time, book_sample_time=book.get("sample_time"),
                book_source_time=source_time, book_age_s=book_age, book_seq=book.get("seq"),
                connection_epoch=book.get("connection_epoch"),
            )
            return "DONE"

        queue = self._entry_queue(direction, entry_price, book)
        cancel_time = actual_post_time + pd.Timedelta(seconds=LIFETIME_SEC)
        rec = self._record(ticker)
        rec.update({
            "decision_time": decision_time, "actual_post_time": actual_post_time, "cancel_time": cancel_time,
            "direction": direction, "midpoint": midpoint, "spread_c": 100 * spread, "btc_return": btc_ret,
            "btc_now_time": btc_now[0], "btc_past_time": btc_past[0], "entry_price": entry_price,
            "entry_queue": queue, "qty": QTY, "quote_time": quote["receipt_time"],
            "quote_exchange_time": quote["exchange_time"], "quote_age_s": quote_age,
            "book_sample_time": book["sample_time"], "book_source_time": source_time,
            "book_exchange_time": book.get("exchange_time"), "book_age_s": book_age,
            "book_seq": book.get("seq"), "connection_epoch": current_epoch,
            "status": "ENTRY_OPEN", "data_invalid_reason": None,
        })
        self.entry_orders[ticker] = {
            "ticker": ticker, "direction": direction, "order_time": actual_post_time,
            "cancel_time": cancel_time, "price": entry_price, "qty": QTY, "remaining": QTY,
            "filled_qty": 0.0, "queue_ahead": queue, "status": "OPEN", "seen": set(),
        }
        self.decisions_done.add(ticker)

        self.emit(
            "SIGNAL", ticker,
            f"{direction} | mid={100*midpoint:.1f}c | spread={100*spread:.1f}c | BTC15={10000*btc_ret:+.2f}bp",
            decision_time=decision_time, actual_post_time=actual_post_time, direction=direction,
            midpoint=float(midpoint), spread_c=float(100 * spread), btc_return=float(btc_ret),
            btc_now_time=btc_now[0], btc_past_time=btc_past[0], quote_time=quote["receipt_time"],
            quote_exchange_time=quote["exchange_time"], quote_age_s=float(quote_age), connection_epoch=current_epoch,
        )
        self.emit(
            "ENTRY_POST", ticker,
            f"{QTY:g}ctrs @ {100*entry_price:.0f}c | queue={queue:.2f} | cancel in {LIFETIME_SEC:.0f}s",
            decision_time=decision_time, actual_post_time=actual_post_time, cancel_time=cancel_time,
            direction=direction, entry_price=float(entry_price), queue_ahead=float(queue), qty=float(QTY),
            book_sample_time=book["sample_time"], book_source_time=source_time,
            book_exchange_time=book.get("exchange_time"), book_age_s=float(book_age),
            book_seq=book.get("seq"), connection_epoch=current_epoch,
        )
        # Never replay anything before the actual simulated post timestamp.
        for trade in list(self.trades.get(ticker, ())):
            if trade["ts"] >= actual_post_time:
                self._apply_entry_trade(self.entry_orders[ticker], trade)
        self._save_state()
        return "DONE"

    def _scheduler(self):
        while not self.stop_event.is_set():
            try:
                now = pd.Timestamp.now(tz="UTC")
                health = self._recorder_health()
                if not health.get("running"):
                    self.emit("ERROR", detail="recorder stopped while primary shadow was running; shadow stopping")
                    self.stop_event.set()
                    break
                with self.lock:
                    for ticker in list(self.quotes.keys()):
                        if ticker in self.decisions_done or ticker.split("-")[0] not in FROZEN_SERIES:
                            continue
                        close_time = self.close_times.get(ticker)
                        if close_time is None or pd.isna(close_time):
                            close_time = _close_from_ticker(ticker)
                        if pd.isna(close_time):
                            continue
                        decision_time = close_time - pd.Timedelta(minutes=10)
                        if decision_time < self.started_at:
                            if now >= decision_time:
                                self.decisions_done.add(ticker)
                            continue
                        if now < decision_time:
                            continue
                        lateness = (now - decision_time).total_seconds()
                        if lateness > MAX_SCHEDULER_LATENESS_SEC:
                            self._mark_data_invalid(
                                ticker, decision_time, f"scheduler late by {lateness:.3f}s", actual_post_time=now,
                            )
                            continue
                        self._process_decision(ticker, decision_time, now)

                    for ticker, order in list(self.entry_orders.items()):
                        if order["status"] == "OPEN" and now >= order["cancel_time"]:
                            order["status"] = "CANCELLED"
                            rec = self._record(ticker)
                            rec["status"] = "POSITION_OPEN" if order["filled_qty"] > 0 else "NO_FILL"
                            self.emit(
                                "ENTRY_CANCEL", ticker,
                                f"filled={order['filled_qty']:.2f}/{QTY:g} | cancelled={order['remaining']:.2f}",
                                cancel_time=order["cancel_time"], actual_cancel_time=now,
                                filled_qty=float(order["filled_qty"]), cancelled_qty=float(order["remaining"]), qty=float(QTY),
                            )
                            self._save_state()
                    for ticker in list(self.positions):
                        self._try_settle(ticker)
            except Exception as exc:
                self.emit("ERROR", detail=f"scheduler: {exc!r}")
            self.stop_event.wait(0.05)

    def _follow_file(self, path, handler, label):
        while not path.exists() and not self.stop_event.is_set():
            self.stop_event.wait(0.25)
        if self.stop_event.is_set():
            return
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(0, os.SEEK_END)
            while not self.stop_event.is_set():
                line = f.readline()
                if not line:
                    self.stop_event.wait(0.02)
                    continue
                try:
                    handler(json.loads(line))
                except Exception as exc:
                    self.emit("ERROR", detail=f"{label}: {exc!r}")

    def _settlement_loop(self):
        while not self.stop_event.is_set():
            try:
                now = pd.Timestamp.now(tz="UTC")
                with self.lock:
                    pending = [
                        ticker for ticker, pos in self.positions.items()
                        if not pos["settled"] and self.close_times.get(ticker) is not None and now >= self.close_times[ticker]
                    ]
                for ticker in pending:
                    try:
                        response = self._rest.get(f"{KALSHI_REST_BASE}/markets/{ticker}", timeout=8)
                        response.raise_for_status()
                        payload = response.json()
                        result = str((payload.get("market") or payload).get("result") or "").upper()
                        if result in {"YES", "NO"}:
                            with self.lock:
                                self.results[ticker] = result
                                self._try_settle(ticker)
                    except Exception as exc:
                        self.emit("ERROR", ticker, f"settlement lookup: {exc!r}")
            except Exception as exc:
                self.emit("ERROR", detail=f"settlement loop: {exc!r}")
            self.stop_event.wait(10)

    def start(self):
        if any(thread.is_alive() for thread in self.threads):
            print("Primary shadow trader is already running.")
            return self
        health = self._recorder_health()
        if not health.get("running"):
            raise RuntimeError("Recorder is not running. Start a fresh recorder first.")
        if not health.get("healthy"):
            raise RuntimeError(f"Recorder is not healthy enough for confirmatory OOS: {health}")
        print(f"Primary shadow session: {self.session_dir}")
        print(f"Primary shadow output: {self.out_dir}")
        for obj in _read_recent_jsonl(self.ticker_file, 16 * 1024 * 1024):
            self._on_ticker(obj)
        for obj in _read_recent_jsonl(self.book_file, 64 * 1024 * 1024):
            self._on_book(obj)
        self._seed_coinbase()
        specs = [
            (self._follow_file, (self.ticker_file, self._on_ticker, "ticker"), "shadow-ticker"),
            (self._follow_file, (self.book_file, self._on_book, "book"), "shadow-book"),
            (self._follow_file, (self.trades_file, self._on_trade, "trades"), "shadow-trades"),
            (self._scheduler, (), "shadow-scheduler"),
            (self._coinbase_loop, (), "shadow-coinbase"),
            (self._settlement_loop, (), "shadow-settlement"),
        ]
        self.threads.clear()
        for target, args, name in specs:
            thread = threading.Thread(target=target, args=args, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)
        print(
            "PRIMARY SHADOW: M5 | BTC completed-1m 15m opposition | spread<=2c | "
            "-3c | actual-post 15s | 3ct | HOLD TO SETTLEMENT | READ-ONLY"
        )
        return self

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=3)
        self._save_state()
        print(f"Primary shadow trader stopped. Saved to: {self.out_dir}")
        return self.out_dir

    def is_running(self):
        return any(thread.is_alive() for thread in self.threads)


_SHADOW = None


def start_primary_shadow_trader(session_dir):
    global _SHADOW
    target = Path(session_dir)
    if _SHADOW is not None and _SHADOW.is_running():
        if _SHADOW.session_dir.resolve() == target.resolve():
            print("Primary shadow trader is already running for this recorder session.")
            return _SHADOW
        raise RuntimeError(f"Primary shadow already running on {_SHADOW.session_dir}; refusing to switch sessions.")
    _SHADOW = PrimaryShadowTrader(target).start()
    return _SHADOW


def stop_primary_shadow_trader():
    global _SHADOW
    if _SHADOW is None:
        print("Primary shadow trader is not running.")
        return None
    out = _SHADOW.stop()
    _SHADOW = None
    return out


def primary_shadow_running():
    return _SHADOW is not None and _SHADOW.is_running()


start_shadow_trader = start_primary_shadow_trader
stop_shadow_trader = stop_primary_shadow_trader
