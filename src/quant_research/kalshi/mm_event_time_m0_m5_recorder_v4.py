from __future__ import annotations

"""Compact event-time microstructure recorder for the FIRST five minutes only.

RESEARCH DATA ACQUISITION ONLY -- no trading policy is encoded here.

Capture window
--------------
For every fixed-universe 15-minute crypto contract:
    M0 <= elapsed < M5   i.e. 0 <= elapsed < 300 seconds.

The recorder discovers/subscribes up to five minutes BEFORE M0 so the book can
be initialized before capture starts, but it does not persist pre-M0 market
microstructure. Markets are removed after M5.

Channels
--------
- orderbook_delta with use_yes_price=True
  * initial orderbook_snapshot initializes the full in-memory book;
  * every delta updates the full in-memory book;
  * only changes to the top 3 YES-price-scale bid/ask levels are persisted;
  * sequence gaps invalidate books and request fresh snapshots.
- trade
  * every public trade during M0-M5 is persisted at event time; no 1s bucketing.
- ticker
  * every ticker update during M0-M5 is persisted as an independent BBO cross-check.

Storage is intentionally compact: the full order book exists only in memory.
Deep deltas that do not change the top 3 are not written to disk.

Important limitations
---------------------
- Exactly M0-M5 is persisted. There is no post-M5 tail, so a 30s markout is not
  available for observations after M4:30, and a 60s markout is not available
  after M4:00 unless a later recorder version explicitly adds a tail.
- This records public market state only. It does not solve the counterfactual
  market-impact problem of hypothetical inside-spread Q100 quotes.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import recorder_core as C

STUDY_VERSION = "MM_EVENT_TIME_M0_M5_V4"
TOP_LEVELS = 3
CAPTURE_START_ELAPSED_S = 0.0
CAPTURE_END_ELAPSED_S = 300.0
PRESUBSCRIBE_LEAD_S = 300.0
MARKET_RESCAN_S = 5.0
BOUNDARY_POLL_S = 0.10
HEALTH_INTERVAL_S = 5.0
STARTUP_TIMEOUT_S = 40.0
STOP_TIMEOUT_S = 45.0

CRYPTO_SERIES = (
    "KXBTC15M",
    "KXBNB15M",
    "KXDOGE15M",
    "KXETH15M",
    "KXHYPE15M",
    "KXNEAR15M",
    "KXSOL15M",
    "KXXRP15M",
    "KXZEC15M",
)

PROJECT_ROOT = C.PROJECT_ROOT
ROOT = C.DATA_ROOT / "mm_event_m0_m5_v4"
CONTROL_PATH = ROOT / "active_recorder.json"
ROOT.mkdir(parents=True, exist_ok=True)


def _iso(dt=None):
    return C.iso_utc(dt or C.utc_now())


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _pid_alive(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _f(x):
    try:
        z = float(x)
        return z if np.isfinite(z) else None
    except Exception:
        return None


def _q(x):
    z = _f(x)
    return z if z is not None and z >= 0.0 else None


def _session_dir_now():
    return ROOT / C.utc_now().strftime("%Y%m%d_%H%M%S")


class Jsonl:
    def __init__(self, path):
        self.path = Path(path)
        self.fh = self.path.open("a", buffering=1, encoding="utf-8")

    def write(self, obj):
        self.fh.write(json.dumps(obj, default=str, separators=(",", ":")) + "\n")

    def close(self):
        try:
            self.fh.flush()
            self.fh.close()
        except Exception:
            pass


def _market_row(series, m):
    ticker = str(m.get("ticker") or "")
    close = C.parse_time(m.get("close_time"))
    if not ticker or close is None:
        return None
    return {
        "ticker": ticker,
        "event_ticker": m.get("event_ticker"),
        "series_ticker": series,
        "market_title": m.get("title") or m.get("yes_sub_title") or "",
        "open_time": C.parse_time(m.get("open_time")),
        "close_time": close,
        "window_start": close - timedelta(seconds=900),
        "rest_status": m.get("status"),
    }


def _elapsed(meta, t=None):
    if not meta or meta.get("window_start") is None:
        return None
    t = t or C.utc_now()
    if not isinstance(t, datetime):
        t = datetime.fromtimestamp(float(t), tz=timezone.utc)
    return (t - meta["window_start"]).total_seconds()


def _capture(meta, t=None):
    e = _elapsed(meta, t)
    return (
        e is not None
        and CAPTURE_START_ELAPSED_S <= e < CAPTURE_END_ELAPSED_S,
        e,
    )


def _discover_sync():
    now = C.utc_now()
    out = {}
    for series in CRYPTO_SERIES:
        for query_status in ("unopened", "open"):
            try:
                markets = C.rest_get(
                    "/markets",
                    {"series_ticker": series, "status": query_status, "limit": 1000},
                ).get("markets") or []
            except Exception:
                continue
            for m in markets:
                row = _market_row(series, m)
                if row is None:
                    continue
                e = _elapsed(row, now)
                if e is None:
                    continue
                # Subscribe at most five minutes before M0 and remove after M5.
                if -PRESUBSCRIBE_LEAD_S <= e < CAPTURE_END_ELAPSED_S:
                    out[row["ticker"]] = row
    return out


async def _discover():
    return await asyncio.to_thread(_discover_sync)


def _exchange_time(msg):
    if not isinstance(msg, dict):
        return None
    ms = msg.get("ts_ms")
    if ms is not None:
        try:
            return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
        except Exception:
            pass
    ts = msg.get("ts")
    if ts is not None:
        try:
            z = float(ts)
            if z > 1e11:
                z /= 1000.0
            return datetime.fromtimestamp(z, tz=timezone.utc)
        except Exception:
            try:
                return C.parse_time(ts)
            except Exception:
                pass
    return None


def _empty_book():
    return {"yes": {}, "no": {}}


def _apply_snapshot(book, msg):
    book["yes"].clear()
    book["no"].clear()
    for side, field in (("yes", "yes_dollars_fp"), ("no", "no_dollars_fp")):
        for level in msg.get(field) or []:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue
            p, q = _f(level[0]), _q(level[1])
            if p is not None and q is not None and q > 0:
                book[side][float(p)] = float(q)


def _apply_delta(book, msg):
    side = str(msg.get("side") or "").lower()
    p, d = _f(msg.get("price_dollars")), _f(msg.get("delta_fp"))
    if side not in {"yes", "no"} or p is None or d is None:
        return False
    new = float(book[side].get(float(p), 0.0)) + float(d)
    if new <= 1e-12:
        book[side].pop(float(p), None)
    else:
        book[side][float(p)] = new
    return True


def _top_state(book):
    bids = sorted(
        ((float(p), float(q)) for p, q in book["yes"].items() if q > 0),
        reverse=True,
    )[:TOP_LEVELS]
    asks = sorted(
        ((float(p), float(q)) for p, q in book["no"].items() if q > 0),
        key=lambda x: x[0],
    )[:TOP_LEVELS]

    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None
    bq = bids[0][1] if bids else None
    aq = asks[0][1] if asks else None
    valid = bid is not None and ask is not None and 0.0 <= bid < ask <= 1.0
    spread_c = 100.0 * (ask - bid) if valid else None
    mid = 0.5 * (bid + ask) if valid else None
    return {
        "valid_bbo": bool(valid),
        "yes_bid": bid,
        "yes_ask": ask,
        "yes_bid_size": bq,
        "yes_ask_size": aq,
        "spread_c": spread_c,
        "mid": mid,
        "bid_levels": [[p, q] for p, q in bids],
        "ask_levels": [[p, q] for p, q in asks],
    }


def _signature(top):
    def norm(levels):
        return tuple((round(float(p), 6), round(float(q), 6)) for p, q in levels)
    return (norm(top.get("bid_levels") or []), norm(top.get("ask_levels") or []))


def _book_event_row(ticker, meta, top, receipt, event_type, *, seq=None, msg=None):
    e = _elapsed(meta, receipt)
    row = {
        "receipt_time": _iso(receipt),
        "exchange_time": _iso(_exchange_time(msg)) if msg and _exchange_time(msg) else None,
        "ticker": ticker,
        "series_ticker": meta.get("series_ticker"),
        "close_time": _iso(meta.get("close_time")),
        "elapsed_s": e,
        "event_type": event_type,
        "seq": seq,
        **top,
    }
    if msg is not None and event_type == "book_delta":
        row.update({
            "delta_side": msg.get("side"),
            "delta_price": _f(msg.get("price_dollars")),
            "delta_qty": _f(msg.get("delta_fp")),
        })
    return row


def _trade_book_side(msg):
    side = str(msg.get("taker_book_side") or "").lower()
    if side in {"bid", "ask"}:
        return side
    outcome = str(msg.get("taker_outcome_side") or msg.get("taker_side") or "").lower()
    if outcome == "yes":
        return "bid"
    if outcome == "no":
        return "ask"
    return None


async def _send(ws, lock, obj):
    async with lock:
        await ws.send(json.dumps(obj))


async def _subscribe(ws, lock, state, channel, tickers):
    if not tickers or channel in state["pending"]:
        return
    rid = state["next_id"]
    state["next_id"] += 1
    state["pending"].add(channel)
    params = {"channels": [channel], "market_tickers": sorted(tickers)}
    if channel == "orderbook_delta":
        params["use_yes_price"] = True
    await _send(ws, lock, {"id": rid, "cmd": "subscribe", "params": params})


async def _update(ws, lock, state, channel, action, tickers):
    sid = state["sids"].get(channel)
    if sid is None or not tickers:
        return
    rid = state["next_id"]
    state["next_id"] += 1
    await _send(
        ws,
        lock,
        {
            "id": rid,
            "cmd": "update_subscription",
            "params": {
                "sid": sid,
                "market_tickers": sorted(tickers),
                "action": action,
            },
        },
    )


async def _request_snapshots(ws, lock, state, tickers, reason):
    sid = state["sids"].get("orderbook_delta")
    if sid is None or not tickers:
        return
    rid = state["next_id"]
    state["next_id"] += 1
    await _send(
        ws,
        lock,
        {
            "id": rid,
            "cmd": "update_subscription",
            "params": {
                "sid": sid,
                "market_tickers": sorted(tickers),
                "action": "get_snapshot",
            },
        },
    )
    state["counters"]["snapshot_requests"] += 1
    state["last_snapshot_request_reason"] = reason


async def _supervisor(ws, lock, state, files, stop_event):
    last_scan = 0.0
    latest = {}
    while not stop_event.is_set():
        now_mono = time.monotonic()
        if not latest or now_mono - last_scan >= MARKET_RESCAN_S:
            try:
                latest = await asyncio.wait_for(_discover(), timeout=20.0)
                last_scan = time.monotonic()
                state["last_scan_error"] = None
                state["last_scan_mono"] = last_scan
            except Exception as exc:
                state["last_scan_error"] = repr(exc)
                files["connection_events"].write(
                    {"time": _iso(), "type": "discovery_error", "error": repr(exc)}
                )
                await asyncio.sleep(1.0)
                continue

        desired = set(latest)
        current = set(state["markets"])
        add = desired - current
        remove = current - desired
        state["meta"].update(latest)

        for channel in ("orderbook_delta", "ticker", "trade"):
            if desired and channel not in state["sids"] and channel not in state["pending"]:
                await _subscribe(ws, lock, state, channel, desired)

        if add:
            for channel in ("orderbook_delta", "ticker", "trade"):
                await _update(ws, lock, state, channel, "add_markets", add)
            for ticker in sorted(add):
                m = latest[ticker]
                state["books"].setdefault(ticker, _empty_book())
                state["book_valid"].setdefault(ticker, False)
                files["market_metadata"].write({
                    "time": _iso(),
                    "connection_epoch": state["connection_epoch"],
                    "ticker": ticker,
                    "event_ticker": m.get("event_ticker"),
                    "series_ticker": m.get("series_ticker"),
                    "market_title": m.get("market_title"),
                    "open_time": _iso(m.get("open_time")) if m.get("open_time") else None,
                    "window_start": _iso(m.get("window_start")),
                    "close_time": _iso(m.get("close_time")),
                    "discovered_status": m.get("rest_status"),
                    "capture_start_elapsed_s": CAPTURE_START_ELAPSED_S,
                    "capture_end_elapsed_s": CAPTURE_END_ELAPSED_S,
                })

        if remove:
            for channel in ("orderbook_delta", "ticker", "trade"):
                await _update(ws, lock, state, channel, "delete_markets", remove)
            for ticker in remove:
                state["books"].pop(ticker, None)
                state["book_valid"].pop(ticker, None)
                state["meta"].pop(ticker, None)

        if add or remove:
            files["market_rotations"].write({
                "time": _iso(),
                "connection_epoch": state["connection_epoch"],
                "subscribed_count": len(desired),
                "added": sorted(add),
                "removed": sorted(remove),
            })

        state["markets"] = desired
        state["supervisor_heartbeat_mono"] = time.monotonic()
        await asyncio.sleep(0.5)


async def _boundary_writer(state, files, stop_event):
    while not stop_event.is_set():
        now = C.utc_now()
        for ticker, meta in list(state["meta"].items()):
            e = _elapsed(meta, now)
            if e is None:
                continue
            if e >= CAPTURE_START_ELAPSED_S and ticker not in state["capture_started"]:
                top = _top_state(state["books"].get(ticker, _empty_book()))
                files["book_top3_events"].write(
                    _book_event_row(ticker, meta, top, now, "capture_start")
                )
                state["capture_started"].add(ticker)
                state["counters"]["book_rows"] += 1
            if e >= CAPTURE_END_ELAPSED_S and ticker not in state["capture_ended"]:
                top = _top_state(state["books"].get(ticker, _empty_book()))
                files["book_top3_events"].write(
                    _book_event_row(ticker, meta, top, now, "capture_end")
                )
                state["capture_ended"].add(ticker)
                state["counters"]["book_rows"] += 1
        await asyncio.sleep(BOUNDARY_POLL_S)


async def _consumer(ws, lock, state, files, stop_event):
    async for raw in ws:
        if stop_event.is_set():
            break
        receipt = C.utc_now()
        state["last_ws_mono"] = time.monotonic()
        try:
            data = json.loads(raw)
        except Exception:
            state["counters"]["json_errors"] += 1
            continue

        typ = data.get("type")
        msg = data.get("msg") or {}
        sid = data.get("sid")
        seq = data.get("seq")

        if typ == "subscribed":
            channel = msg.get("channel")
            resolved_sid = msg.get("sid", sid)
            if channel and resolved_sid is not None:
                state["sids"][channel] = resolved_sid
                state["pending"].discard(channel)
                files["connection_events"].write({
                    "time": _iso(receipt),
                    "type": "subscribed",
                    "channel": channel,
                    "sid": resolved_sid,
                    "connection_epoch": state["connection_epoch"],
                })
            continue

        if typ == "error":
            state["counters"]["ws_errors"] += 1
            files["connection_events"].write({
                "time": _iso(receipt),
                "type": "ws_error",
                "payload": data,
                "connection_epoch": state["connection_epoch"],
            })
            continue

        if typ not in {"orderbook_snapshot", "orderbook_delta", "ticker", "trade"}:
            continue

        ticker = msg.get("market_ticker")
        if not ticker or ticker not in state["markets"]:
            continue
        meta = state["meta"].get(ticker)
        state["last_market_data_mono"] = time.monotonic()

        if typ in {"orderbook_snapshot", "orderbook_delta"} and sid is not None and seq is not None:
            try:
                seqi = int(seq)
                prev = state["last_orderbook_seq"].get(sid)
                if prev is not None and seqi != prev + 1:
                    state["counters"]["sequence_gaps"] += 1
                    for t in list(state["markets"]):
                        state["book_valid"][t] = False
                        state["books"][t] = _empty_book()
                    files["connection_events"].write({
                        "time": _iso(receipt),
                        "type": "orderbook_sequence_gap",
                        "sid": sid,
                        "expected_seq": prev + 1,
                        "received_seq": seqi,
                        "connection_epoch": state["connection_epoch"],
                    })
                    await _request_snapshots(
                        ws, lock, state, set(state["markets"]), "sequence_gap"
                    )
                    state["last_orderbook_seq"][sid] = seqi
                    if typ == "orderbook_delta":
                        continue
                state["last_orderbook_seq"][sid] = seqi
            except Exception:
                pass

        if typ == "orderbook_snapshot":
            book = state["books"].setdefault(ticker, _empty_book())
            _apply_snapshot(book, msg)
            state["book_valid"][ticker] = True
            state["counters"]["snapshots_received"] += 1
            capture, _ = _capture(meta, receipt)
            if capture:
                top = _top_state(book)
                files["book_top3_events"].write(
                    _book_event_row(
                        ticker, meta, top, receipt, "book_snapshot", seq=seq, msg=msg
                    )
                )
                state["counters"]["book_rows"] += 1
            continue

        if typ == "orderbook_delta":
            state["counters"]["deltas_received"] += 1
            if not state["book_valid"].get(ticker, False):
                state["counters"]["deltas_skipped_uninitialized"] += 1
                continue
            book = state["books"].setdefault(ticker, _empty_book())
            before = _top_state(book)
            before_sig = _signature(before)
            if not _apply_delta(book, msg):
                state["counters"]["bad_deltas"] += 1
                continue
            after = _top_state(book)
            capture, _ = _capture(meta, receipt)
            if capture and _signature(after) != before_sig:
                files["book_top3_events"].write(
                    _book_event_row(
                        ticker, meta, after, receipt, "book_delta", seq=seq, msg=msg
                    )
                )
                state["counters"]["book_rows"] += 1
                state["counters"]["top3_changing_deltas"] += 1
            continue

        capture, e = _capture(meta, receipt)
        if not capture:
            continue

        if typ == "ticker":
            files["ticker_event_time"].write({
                "receipt_time": _iso(receipt),
                "exchange_time": _iso(_exchange_time(msg)) if _exchange_time(msg) else None,
                "ticker": ticker,
                "series_ticker": meta.get("series_ticker"),
                "close_time": _iso(meta.get("close_time")),
                "elapsed_s": e,
                "yes_bid": _f(msg.get("yes_bid_dollars")),
                "yes_ask": _f(msg.get("yes_ask_dollars")),
                "yes_bid_size": _q(msg.get("yes_bid_size_fp")),
                "yes_ask_size": _q(msg.get("yes_ask_size_fp")),
                "last_price": _f(msg.get("price_dollars")),
                "last_trade_size": _q(msg.get("last_trade_size_fp")),
                "ts_ms": msg.get("ts_ms"),
            })
            state["counters"]["ticker_rows"] += 1
            continue

        if typ == "trade":
            price = _f(msg.get("yes_price_dollars") or msg.get("price_dollars"))
            qty = _q(msg.get("count_fp") or msg.get("count"))
            if price is None or qty is None or qty <= 0:
                state["counters"]["bad_trades"] += 1
                continue
            files["trades_event_time"].write({
                "receipt_time": _iso(receipt),
                "exchange_time": _iso(_exchange_time(msg)) if _exchange_time(msg) else None,
                "ticker": ticker,
                "series_ticker": meta.get("series_ticker"),
                "close_time": _iso(meta.get("close_time")),
                "elapsed_s": e,
                "trade_id": msg.get("trade_id"),
                "yes_price": price,
                "no_price": _f(msg.get("no_price_dollars")),
                "qty": qty,
                "taker_book_side": _trade_book_side(msg),
                "taker_outcome_side": msg.get("taker_outcome_side") or msg.get("taker_side"),
                "is_block_trade": msg.get("is_block_trade"),
                "ts_ms": msg.get("ts_ms"),
            })
            state["counters"]["trade_rows"] += 1
            continue


async def _health_writer(runtime, session_dir, stop_event):
    path = session_dir / "health.json"
    while not stop_event.is_set():
        state = runtime.get("state")
        now_mono = time.monotonic()
        if state is None:
            obj = {
                "time": _iso(),
                "pid": os.getpid(),
                "running": True,
                "healthy": False,
                "connection_epoch": runtime.get("connection_epoch", 0),
            }
        else:
            md_age = (
                now_mono - state["last_market_data_mono"]
                if state.get("last_market_data_mono") is not None
                else None
            )
            active_capture = 0
            now = C.utc_now()
            for m in state["meta"].values():
                if _capture(m, now)[0]:
                    active_capture += 1
            healthy = (
                state.get("connected", False)
                and state.get("supervisor_heartbeat_mono") is not None
                and now_mono - state["supervisor_heartbeat_mono"] < 10.0
                and state.get("last_scan_error") is None
            )
            obj = {
                "time": _iso(),
                "pid": os.getpid(),
                "running": True,
                "healthy": bool(healthy),
                "study_version": STUDY_VERSION,
                "connection_epoch": state["connection_epoch"],
                "subscribed_markets": len(state["markets"]),
                "active_m0_m5_markets": active_capture,
                "channels": sorted(state["sids"]),
                "market_data_age_s": md_age,
                "last_scan_error": state.get("last_scan_error"),
                "book_rows": state["counters"]["book_rows"],
                "ticker_rows": state["counters"]["ticker_rows"],
                "trade_rows": state["counters"]["trade_rows"],
                "snapshots_received": state["counters"]["snapshots_received"],
                "deltas_received": state["counters"]["deltas_received"],
                "top3_changing_deltas": state["counters"]["top3_changing_deltas"],
                "sequence_gaps": state["counters"]["sequence_gaps"],
                "snapshot_requests": state["counters"]["snapshot_requests"],
            }
        _atomic_json(path, obj)
        await asyncio.sleep(HEALTH_INTERVAL_S)


async def _connection_once(key_id, private_key, runtime, files, stop_event):
    ws = await C.open_ws(key_id, private_key)
    lock = asyncio.Lock()
    runtime["connection_epoch"] += 1
    from collections import Counter
    state = {
        "connection_epoch": runtime["connection_epoch"],
        "connected": True,
        "next_id": 1,
        "sids": {},
        "pending": set(),
        "markets": set(),
        "meta": {},
        "books": {},
        "book_valid": {},
        "last_orderbook_seq": {},
        "capture_started": set(),
        "capture_ended": set(),
        "last_ws_mono": time.monotonic(),
        "last_market_data_mono": None,
        "supervisor_heartbeat_mono": None,
        "last_scan_mono": None,
        "last_scan_error": None,
        "last_snapshot_request_reason": None,
        "counters": Counter(),
    }
    runtime["state"] = state
    files["connection_events"].write({
        "time": _iso(),
        "type": "connected",
        "connection_epoch": state["connection_epoch"],
    })

    tasks = [
        asyncio.create_task(_supervisor(ws, lock, state, files, stop_event)),
        asyncio.create_task(_consumer(ws, lock, state, files, stop_event)),
        asyncio.create_task(_boundary_writer(state, files, stop_event)),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            exc = t.exception()
            if exc is not None:
                raise exc
    finally:
        state["connected"] = False
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await ws.close()
        except Exception:
            pass
        files["connection_events"].write({
            "time": _iso(),
            "type": "disconnected",
            "connection_epoch": state["connection_epoch"],
        })


async def run_event_time_m0_m5_recorder(session_dir: Path):
    session_dir = Path(session_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=False)

    files = {
        "book_top3_events": Jsonl(session_dir / "book_top3_events.jsonl"),
        "trades_event_time": Jsonl(session_dir / "trades_event_time.jsonl"),
        "ticker_event_time": Jsonl(session_dir / "ticker_event_time.jsonl"),
        "market_metadata": Jsonl(session_dir / "market_metadata.jsonl"),
        "market_rotations": Jsonl(session_dir / "market_rotations.jsonl"),
        "connection_events": Jsonl(session_dir / "connection_events.jsonl"),
    }

    started = C.utc_now()
    capture_spec = {
        "study_version": STUDY_VERSION,
        "purpose": "event-time microstructure development data for first-five-minute MM research",
        "universe": list(CRYPTO_SERIES),
        "persisted_window": "M0 <= elapsed < M5",
        "persisted_elapsed_seconds": [0.0, 300.0],
        "pre_subscribe_lead_seconds": PRESUBSCRIBE_LEAD_S,
        "orderbook_channel": "orderbook_delta",
        "orderbook_use_yes_price": True,
        "in_memory_book": "full aggregated price-level book",
        "persisted_book": "top 3 bid/ask levels only when top3 changes + boundary/snapshot records",
        "trades": "every public trade at event time during M0-M5; no aggregation",
        "ticker": "every ticker event during M0-M5 for independent BBO validation",
        "post_m5_tail_seconds": 0.0,
        "strategy_pnl_recorded": False,
        "raw_deep_delta_persistence": False,
    }
    _atomic_json(session_dir / "capture_spec.json", capture_spec)
    _atomic_json(session_dir / "session_manifest.json", {
        "study_version": STUDY_VERSION,
        "session_dir": str(session_dir),
        "started_at": _iso(started),
        "pid": os.getpid(),
        "capture_spec": capture_spec,
    })

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except Exception:
            pass

    key_id, private_key = C.load_auth()
    runtime = {"connection_epoch": 0, "state": None}
    health_task = asyncio.create_task(_health_writer(runtime, session_dir, stop_event))

    try:
        while not stop_event.is_set():
            try:
                await _connection_once(key_id, private_key, runtime, files, stop_event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                files["connection_events"].write({
                    "time": _iso(),
                    "type": "connection_exception",
                    "connection_epoch": runtime.get("connection_epoch", 0),
                    "error": repr(exc),
                })
                if not stop_event.is_set():
                    await asyncio.sleep(3.0)
    finally:
        stop_event.set()
        health_task.cancel()
        await asyncio.gather(health_task, return_exceptions=True)
        ended = C.utc_now()
        final_state = runtime.get("state")
        final_counts = dict(final_state["counters"]) if final_state else {}
        _atomic_json(session_dir / "session_manifest.json", {
            "study_version": STUDY_VERSION,
            "session_dir": str(session_dir),
            "started_at": _iso(started),
            "ended_at": _iso(ended),
            "duration_hours": (ended - started).total_seconds() / 3600.0,
            "pid": os.getpid(),
            "connection_epochs": runtime.get("connection_epoch", 0),
            "final_counts": final_counts,
            "capture_spec": capture_spec,
        })
        _atomic_json(session_dir / "health.json", {
            "time": _iso(ended),
            "pid": os.getpid(),
            "running": False,
            "healthy": False,
            "study_version": STUDY_VERSION,
            "final_counts": final_counts,
        })
        for f in files.values():
            f.close()


def event_time_m0_m5_status():
    ctl = _read_json(CONTROL_PATH, {}) or {}
    if not ctl:
        print("No V4 recorder control file.")
        return {"running": False}
    session = Path(ctl.get("session_dir", ""))
    health = _read_json(session / "health.json", {}) or {}
    out = {**ctl, **health, "pid_alive": _pid_alive(ctl.get("pid"))}
    out["running"] = bool(out.get("pid_alive") and health.get("running", True))
    print(json.dumps(out, indent=2, default=str))
    return out


def start_event_time_m0_m5_recording(*, startup_timeout_s=STARTUP_TIMEOUT_S):
    ctl = _read_json(CONTROL_PATH, {}) or {}
    if ctl and _pid_alive(ctl.get("pid")):
        raise RuntimeError(
            f"V4 recorder already running: pid={ctl.get('pid')} session={ctl.get('session_dir')}"
        )
    if CONTROL_PATH.exists():
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass

    session_dir = _session_dir_now().resolve()
    if session_dir.exists():
        raise FileExistsError(session_dir)
    log_path = session_dir.parent / f"{session_dir.name}.startup.log"
    cmd = [
        sys.executable,
        "-m",
        "quant_research.kalshi.mm_event_time_m0_m5_recorder_v4",
        "--run-session",
        str(session_dir),
    ]
    log_fh = log_path.open("a", buffering=1, encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_fh.close()

    _atomic_json(CONTROL_PATH, {
        "pid": proc.pid,
        "session_dir": str(session_dir),
        "started_at": _iso(),
        "launcher_pid": os.getpid(),
        "log_path": str(log_path),
        "study_version": STUDY_VERSION,
    })

    deadline = time.time() + float(startup_timeout_s)
    last_health = None
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
            except Exception:
                pass
            raise RuntimeError(
                f"V4 recorder exited during startup with code {proc.returncode}.\n{tail}"
            )
        hp = session_dir / "health.json"
        if hp.exists():
            last_health = _read_json(hp, {}) or {}
            if last_health.get("healthy"):
                print("Event-time M0-M5 recorder V4 is healthy.")
                print(f"PID: {proc.pid}")
                print(f"SESSION: {session_dir}")
                print("CAPTURE: M0-M5 ONLY [0s,300s) | pre-subscribe lead=5m")
                print("BOOK: full in memory -> persist top3 changes only")
                print("TRADES: every event | TICKER: every event | no 1s aggregation")
                print("ORDERBOOK: use_yes_price=true | gap -> snapshot refresh")
                print(f"LOG: {log_path}")
                return session_dir
        time.sleep(0.5)

    if proc.poll() is None:
        print("V4 process is running but healthy=True was not reached before timeout.")
        print(f"SESSION: {session_dir}")
        print(f"LAST HEALTH: {last_health}")
        print(f"LOG: {log_path}")
        return session_dir
    raise RuntimeError("V4 recorder failed to start")


def stop_event_time_m0_m5_recording(*, expected_session=None, timeout_s=STOP_TIMEOUT_S):
    ctl = _read_json(CONTROL_PATH, {}) or {}
    if not ctl:
        print("No active V4 recorder control file.")
        return None
    session = Path(ctl.get("session_dir", "")).resolve()
    if expected_session is not None and session != Path(expected_session).resolve():
        raise RuntimeError(f"Refusing stop: active={session}, expected={Path(expected_session).resolve()}")
    pid = ctl.get("pid")
    if not _pid_alive(pid):
        print(f"V4 pid is not alive: {pid}. Session preserved: {session}")
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass
        return session

    print(f"Stopping V4 recorder pid={pid} ...")
    try:
        os.kill(int(pid), signal.SIGINT)
    except Exception:
        pass
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.5)
    if _pid_alive(pid):
        print("Graceful stop timed out; sending SIGTERM.")
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
        deadline = time.time() + 10.0
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.25)
    if _pid_alive(pid):
        raise RuntimeError(f"V4 recorder pid={pid} did not stop")
    try:
        CONTROL_PATH.unlink()
    except Exception:
        pass
    print(f"SAVED SESSION: {session}")
    manifest = _read_json(session / "session_manifest.json", {}) or {}
    if manifest.get("duration_hours") is not None:
        print(f"DURATION: {float(manifest['duration_hours']):.3f} hours")
    return session


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--run-session", default=None)
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    args = p.parse_args()
    if args.run_session:
        asyncio.run(run_event_time_m0_m5_recorder(Path(args.run_session)))
        return
    if args.status:
        event_time_m0_m5_status()
        return
    if args.stop:
        stop_event_time_m0_m5_recording()
        return
    p.error("Use --run-session PATH, --status, or --stop")


if __name__ == "__main__":
    _main()
