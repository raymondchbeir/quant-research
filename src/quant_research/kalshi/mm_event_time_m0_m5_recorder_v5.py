from __future__ import annotations

"""Corrected event-time M0-M5(+30s label tail) recorder for V5 development.

RESEARCH DATA ACQUISITION ONLY -- no trading policy is executed.

Why V5 exists
-------------
V4 established that event-time public data are useful, but exposed two recorder
problems that must be corrected before the next development round:

1) Sequence accounting must include *all* orderbook-subscription messages that
   carry the same sid/seq, including ``type='ok'`` responses from
   ``update_subscription``. Ignoring those control responses can manufacture
   apparent positive sequence gaps.
2) V4 reconstructed locked/crossed states only for the very high-event-rate BTC
   series. V5 keeps orderbook prices/quantities as exact Decimal values in
   memory, rejects impossible negative-level updates, and immediately repairs a
   crossed reconstructed book from a fresh server snapshot.

Capture design
--------------
- Pre-subscribe up to 5 minutes before M0 to initialize books.
- Persist event-time top-3 book changes, ticker updates, and public trades for
  M0 <= elapsed < M5+30s.
- The trading/research decision window remains M0 <= elapsed < M5.
- M5..M5+30s is a LABEL-ONLY tail so every trade/fill before M5 can receive a
  causal 30-second markout. No later strategy is allowed to initiate quotes in
  this tail merely because it is recorded.
- Full aggregated books live only in RAM; only top-3 changes are persisted.

Pre-registered V5-development candidate family
----------------------------------------------
A) L3 support only.
B) L1 AND L3 support.
C) L3 support AND natural spread >= 2c.
D) L1 AND L3 support AND natural spread >= 2c.

No additional numeric threshold sweep is pre-authorized on this session.
Capacity reporting will include Q1 as the execution reference and Q2/Q5/Q10 as
explicitly counterfactual capacity scenarios. The economic target is at least
$100/day; merely positive tiny PnL is not sufficient.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import recorder_core as C

STUDY_VERSION = "MM_EVENT_TIME_M0_M5_V5_DEV"
TOP_LEVELS = 3
TRADE_WINDOW_START_S = 0.0
TRADE_WINDOW_END_S = 300.0
LABEL_TAIL_END_S = 330.0
PRESUBSCRIBE_LEAD_S = 300.0
MARKET_RESCAN_S = 5.0
BOUNDARY_POLL_S = 0.05
HEALTH_INTERVAL_S = 5.0
STARTUP_TIMEOUT_S = 45.0
STOP_TIMEOUT_S = 45.0
SNAPSHOT_DEBOUNCE_S = 0.075
TICKER_MISMATCH_TOL_C = 1.01
TICKER_MISMATCH_PERSIST_S = 0.250
TICKER_RESYNC_COOLDOWN_S = 1.0
ZERO = Decimal("0")

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
ROOT = C.DATA_ROOT / "mm_event_m0_m5_v5_dev"
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


def _pid_state(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return None
        p = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if p.returncode != 0:
            return None
        s = p.stdout.strip()
        return s or None
    except Exception:
        try:
            os.kill(int(pid), 0)
            return "?"
        except Exception:
            return None


def _pid_alive(pid):
    s = _pid_state(pid)
    return bool(s and "Z" not in s.upper())


def _f(x):
    try:
        z = float(x)
        return z if z == z and abs(z) != float("inf") else None
    except Exception:
        return None


def _q(x):
    z = _f(x)
    return z if z is not None and z >= 0.0 else None


def _dec(x):
    try:
        z = Decimal(str(x))
        return z if z.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


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


def _persist_phase(meta, t=None):
    e = _elapsed(meta, t)
    if e is None:
        return False, e, None
    if TRADE_WINDOW_START_S <= e < TRADE_WINDOW_END_S:
        return True, e, "M0_M5_RESEARCH"
    if TRADE_WINDOW_END_S <= e < LABEL_TAIL_END_S:
        return True, e, "M5_M5P30_LABEL_TAIL"
    return False, e, None


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
                if -PRESUBSCRIBE_LEAD_S <= e < LABEL_TAIL_END_S:
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
            p, q = _dec(level[0]), _dec(level[1])
            if p is not None and q is not None and q > ZERO:
                book[side][p] = q


def _apply_delta(book, msg):
    side = str(msg.get("side") or "").lower()
    p, d = _dec(msg.get("price_dollars")), _dec(msg.get("delta_fp"))
    if side not in {"yes", "no"} or p is None or d is None:
        return False, "malformed_delta"
    old = book[side].get(p, ZERO)
    new = old + d
    if new < ZERO:
        return False, "negative_level"
    if new == ZERO:
        book[side].pop(p, None)
    else:
        book[side][p] = new
    return True, None


def _top_state(book):
    bids_d = sorted(
        ((p, q) for p, q in book["yes"].items() if q > ZERO),
        key=lambda x: x[0],
        reverse=True,
    )[:TOP_LEVELS]
    asks_d = sorted(
        ((p, q) for p, q in book["no"].items() if q > ZERO),
        key=lambda x: x[0],
    )[:TOP_LEVELS]

    bids = [(float(p), float(q)) for p, q in bids_d]
    asks = [(float(p), float(q)) for p, q in asks_d]
    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None
    bq = bids[0][1] if bids else None
    aq = asks[0][1] if asks else None
    crossed_or_locked = bid is not None and ask is not None and bid >= ask
    valid = bid is not None and ask is not None and 0.0 <= bid < ask <= 1.0
    return {
        "valid_bbo": bool(valid),
        "crossed_or_locked": bool(crossed_or_locked),
        "yes_bid": bid,
        "yes_ask": ask,
        "yes_bid_size": bq,
        "yes_ask_size": aq,
        "spread_c": 100.0 * (ask - bid) if valid else None,
        "mid": 0.5 * (bid + ask) if valid else None,
        "bid_levels": [[p, q] for p, q in bids],
        "ask_levels": [[p, q] for p, q in asks],
    }


def _signature(top):
    def norm(levels):
        return tuple((round(float(p), 8), round(float(q), 8)) for p, q in levels)
    return (norm(top.get("bid_levels") or []), norm(top.get("ask_levels") or []))


def _book_event_row(ticker, meta, top, receipt, event_type, *, seq=None, msg=None):
    e = _elapsed(meta, receipt)
    _, _, phase = _persist_phase(meta, receipt)
    exch = _exchange_time(msg) if msg else None
    row = {
        "receipt_time": _iso(receipt),
        "exchange_time": _iso(exch) if exch else None,
        "ticker": ticker,
        "series_ticker": meta.get("series_ticker"),
        "close_time": _iso(meta.get("close_time")),
        "elapsed_s": e,
        "capture_phase": phase,
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
            "params": {"sid": sid, "market_tickers": sorted(tickers), "action": action},
        },
    )


async def _request_snapshots(ws, lock, state, tickers, reason):
    sid = state["sids"].get("orderbook_delta")
    tickers = set(tickers) & set(state["markets"])
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
            "params": {"sid": sid, "market_tickers": sorted(tickers), "action": "get_snapshot"},
        },
    )
    state["counters"]["snapshot_requests"] += 1
    state["counters"]["snapshot_tickers_requested"] += len(tickers)
    state["last_snapshot_request_reason"] = reason
    state["last_snapshot_request_mono"] = time.monotonic()
    state["files"]["connection_events"].write({
        "time": _iso(),
        "type": "snapshot_request",
        "reason": reason,
        "tickers": sorted(tickers),
        "sid": sid,
        "connection_epoch": state["connection_epoch"],
    })


async def _snapshot_batcher(ws, lock, state):
    try:
        while True:
            await asyncio.sleep(SNAPSHOT_DEBOUNCE_S)
            tickers = set(state["snapshot_refresh_tickers"])
            reasons = sorted(state["snapshot_refresh_reasons"])
            state["snapshot_refresh_tickers"].clear()
            state["snapshot_refresh_reasons"].clear()
            if not tickers:
                break
            await _request_snapshots(ws, lock, state, tickers, "+".join(reasons) if reasons else "repair")
            if not state["snapshot_refresh_tickers"]:
                break
    finally:
        state["snapshot_batch_task"] = None
        if state["snapshot_refresh_tickers"] and state.get("connected"):
            state["snapshot_batch_task"] = asyncio.create_task(_snapshot_batcher(ws, lock, state))


def _schedule_snapshot_refresh(ws, lock, state, tickers, reason):
    tickers = set(tickers) & set(state["markets"])
    if not tickers:
        return
    state["snapshot_refresh_tickers"].update(tickers)
    state["snapshot_refresh_reasons"].add(str(reason))
    task = state.get("snapshot_batch_task")
    if task is None or task.done():
        state["snapshot_batch_task"] = asyncio.create_task(_snapshot_batcher(ws, lock, state))


def _invalidate_tickers(state, tickers):
    for ticker in set(tickers):
        if ticker in state["markets"]:
            state["book_initialized"][ticker] = False
            state["books"][ticker] = _empty_book()
            state["ticker_mismatch_start_mono"].pop(ticker, None)


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
                files["connection_events"].write({"time": _iso(), "type": "discovery_error", "error": repr(exc)})
                await asyncio.sleep(1.0)
                continue

        current = set(state["markets"])
        desired = set(latest)
        now = C.utc_now()
        for ticker, known_meta in list(state["meta"].items()):
            e_known = _elapsed(known_meta, now)
            if ticker not in state["tail_ended"] and e_known is not None and -PRESUBSCRIBE_LEAD_S <= e_known < LABEL_TAIL_END_S:
                desired.add(ticker)
        desired -= set(state["tail_ended"])

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
                m = state["meta"].get(ticker) or latest.get(ticker)
                if m is None:
                    continue
                state["books"].setdefault(ticker, _empty_book())
                state["book_initialized"].setdefault(ticker, False)
                files["market_metadata"].write({
                    "time": _iso(), "connection_epoch": state["connection_epoch"],
                    "ticker": ticker, "event_ticker": m.get("event_ticker"),
                    "series_ticker": m.get("series_ticker"), "market_title": m.get("market_title"),
                    "open_time": _iso(m.get("open_time")) if m.get("open_time") else None,
                    "window_start": _iso(m.get("window_start")), "close_time": _iso(m.get("close_time")),
                    "discovered_status": m.get("rest_status"),
                    "trade_window_start_elapsed_s": TRADE_WINDOW_START_S,
                    "trade_window_end_elapsed_s": TRADE_WINDOW_END_S,
                    "label_tail_end_elapsed_s": LABEL_TAIL_END_S,
                })

        if remove:
            for channel in ("orderbook_delta", "ticker", "trade"):
                await _update(ws, lock, state, channel, "delete_markets", remove)
            for ticker in remove:
                state["books"].pop(ticker, None)
                state["book_initialized"].pop(ticker, None)
                state["meta"].pop(ticker, None)
                state["last_book_event_mono"].pop(ticker, None)
                state["ticker_mismatch_start_mono"].pop(ticker, None)

        if add or remove:
            files["market_rotations"].write({
                "time": _iso(), "connection_epoch": state["connection_epoch"],
                "subscribed_count": len(desired), "added": sorted(add), "removed": sorted(remove),
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
            if e >= TRADE_WINDOW_START_S and ticker not in state["capture_started"]:
                top = _top_state(state["books"].get(ticker, _empty_book()))
                files["book_top3_events"].write(_book_event_row(ticker, meta, top, now, "capture_start"))
                state["capture_started"].add(ticker)
                state["counters"]["book_rows"] += 1
            if e >= TRADE_WINDOW_END_S and ticker not in state["trade_window_ended"]:
                top = _top_state(state["books"].get(ticker, _empty_book()))
                files["book_top3_events"].write(_book_event_row(ticker, meta, top, now, "trade_window_end"))
                state["trade_window_ended"].add(ticker)
                state["counters"]["book_rows"] += 1
            if e >= LABEL_TAIL_END_S and ticker not in state["tail_ended"]:
                top = _top_state(state["books"].get(ticker, _empty_book()))
                files["book_top3_events"].write(_book_event_row(ticker, meta, top, now, "label_tail_end"))
                state["tail_ended"].add(ticker)
                state["counters"]["book_rows"] += 1
        await asyncio.sleep(BOUNDARY_POLL_S)


def _record_repair(files, state, receipt, ticker, reason, *, seq=None, msg=None, before=None, after=None):
    files["book_repair_events"].write({
        "time": _iso(receipt), "type": "book_repair", "reason": reason,
        "ticker": ticker, "seq": seq, "connection_epoch": state["connection_epoch"],
        "before": before, "after": after,
        "trigger": {
            "side": msg.get("side") if isinstance(msg, dict) else None,
            "price_dollars": msg.get("price_dollars") if isinstance(msg, dict) else None,
            "delta_fp": msg.get("delta_fp") if isinstance(msg, dict) else None,
        },
    })


def _advance_orderbook_sequence(ws, lock, state, files, data, receipt):
    sid = data.get("sid")
    seq = data.get("seq")
    ob_sid = state["sids"].get("orderbook_delta")
    if sid is None or seq is None or ob_sid is None or sid != ob_sid:
        return False
    try:
        seqi = int(seq)
    except Exception:
        return False

    typ = str(data.get("type") or "")
    prev = state["last_seq_by_sid"].get(sid)
    gap = False
    if prev is not None:
        if seqi > prev + 1:
            gap = True
            state["counters"]["sequence_gaps"] += 1
            missing = seqi - (prev + 1)
            state["counters"]["sequence_numbers_missing"] += missing
            _invalidate_tickers(state, set(state["markets"]))
            files["connection_events"].write({
                "time": _iso(receipt), "type": "orderbook_sequence_gap", "sid": sid,
                "message_type": typ, "expected_seq": prev + 1, "received_seq": seqi,
                "missing_count": missing, "connection_epoch": state["connection_epoch"],
            })
            _schedule_snapshot_refresh(ws, lock, state, set(state["markets"]), "sequence_gap")
        elif seqi <= prev:
            state["counters"]["nonmonotone_or_duplicate_seq"] += 1
            files["connection_events"].write({
                "time": _iso(receipt), "type": "orderbook_nonmonotone_seq", "sid": sid,
                "message_type": typ, "previous_seq": prev, "received_seq": seqi,
                "connection_epoch": state["connection_epoch"],
            })

    if prev is None or seqi > prev:
        state["last_seq_by_sid"][sid] = seqi
    if typ == "ok":
        state["counters"]["ok_seq_messages"] += 1
    return gap


async def _consumer(ws, lock, state, files, stop_event):
    async for raw in ws:
        if stop_event.is_set():
            break
        receipt = C.utc_now()
        now_mono = time.monotonic()
        state["last_ws_mono"] = now_mono
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
                state["sid_channels"][resolved_sid] = channel
                state["pending"].discard(channel)
                files["connection_events"].write({
                    "time": _iso(receipt), "type": "subscribed", "channel": channel,
                    "sid": resolved_sid, "connection_epoch": state["connection_epoch"],
                })
            continue

        if typ == "error":
            state["counters"]["ws_errors"] += 1
            files["connection_events"].write({
                "time": _iso(receipt), "type": "ws_error", "payload": data,
                "connection_epoch": state["connection_epoch"],
            })
            continue

        gap_this_message = _advance_orderbook_sequence(ws, lock, state, files, data, receipt)
        if typ == "ok":
            continue
        if typ not in {"orderbook_snapshot", "orderbook_delta", "ticker", "trade"}:
            continue

        ticker = msg.get("market_ticker")
        if not ticker or ticker not in state["markets"]:
            continue
        meta = state["meta"].get(ticker)
        state["last_market_data_mono"] = now_mono

        if typ == "orderbook_snapshot":
            book = state["books"].setdefault(ticker, _empty_book())
            _apply_snapshot(book, msg)
            top = _top_state(book)
            state["counters"]["snapshots_received"] += 1
            if top["crossed_or_locked"]:
                state["book_initialized"][ticker] = False
                state["counters"]["crossed_snapshot_resyncs"] += 1
                _record_repair(files, state, receipt, ticker, "crossed_snapshot", seq=seq, msg=msg, after=top)
                _invalidate_tickers(state, {ticker})
                _schedule_snapshot_refresh(ws, lock, state, {ticker}, "crossed_snapshot")
                continue
            state["book_initialized"][ticker] = True
            state["last_book_event_mono"][ticker] = now_mono
            state["ticker_mismatch_start_mono"].pop(ticker, None)
            persist, _, _ = _persist_phase(meta, receipt)
            if persist:
                files["book_top3_events"].write(_book_event_row(ticker, meta, top, receipt, "book_snapshot", seq=seq, msg=msg))
                state["counters"]["book_rows"] += 1
            continue

        if typ == "orderbook_delta":
            state["counters"]["deltas_received"] += 1
            if gap_this_message or not state["book_initialized"].get(ticker, False):
                state["counters"]["deltas_skipped_uninitialized"] += 1
                continue
            book = state["books"].setdefault(ticker, _empty_book())
            before = _top_state(book)
            before_sig = _signature(before)
            ok, err = _apply_delta(book, msg)
            if not ok:
                state["counters"]["bad_or_inconsistent_deltas"] += 1
                if err == "negative_level":
                    state["counters"]["negative_level_resyncs"] += 1
                _record_repair(files, state, receipt, ticker, err or "delta_error", seq=seq, msg=msg, before=before)
                _invalidate_tickers(state, {ticker})
                _schedule_snapshot_refresh(ws, lock, state, {ticker}, err or "delta_error")
                continue
            after = _top_state(book)
            if after["crossed_or_locked"]:
                state["counters"]["crossed_after_delta_resyncs"] += 1
                _record_repair(files, state, receipt, ticker, "crossed_after_delta", seq=seq, msg=msg, before=before, after=after)
                _invalidate_tickers(state, {ticker})
                _schedule_snapshot_refresh(ws, lock, state, {ticker}, "crossed_after_delta")
                continue
            state["last_book_event_mono"][ticker] = now_mono
            persist, _, _ = _persist_phase(meta, receipt)
            if persist and _signature(after) != before_sig:
                files["book_top3_events"].write(_book_event_row(ticker, meta, after, receipt, "book_delta", seq=seq, msg=msg))
                state["counters"]["book_rows"] += 1
                state["counters"]["top3_changing_deltas"] += 1
            continue

        persist, e, phase = _persist_phase(meta, receipt)
        if not persist:
            continue

        if typ == "ticker":
            tbid = _f(msg.get("yes_bid_dollars"))
            task = _f(msg.get("yes_ask_dollars"))
            files["ticker_event_time"].write({
                "receipt_time": _iso(receipt), "exchange_time": _iso(_exchange_time(msg)) if _exchange_time(msg) else None,
                "ticker": ticker, "series_ticker": meta.get("series_ticker"), "close_time": _iso(meta.get("close_time")),
                "elapsed_s": e, "capture_phase": phase, "yes_bid": tbid, "yes_ask": task,
                "yes_bid_size": _q(msg.get("yes_bid_size_fp")), "yes_ask_size": _q(msg.get("yes_ask_size_fp")),
                "last_price": _f(msg.get("price_dollars")), "last_trade_size": _q(msg.get("last_trade_size_fp")),
                "ts_ms": msg.get("ts_ms"),
            })
            state["counters"]["ticker_rows"] += 1

            if tbid is not None and task is not None and tbid < task and state["book_initialized"].get(ticker, False):
                top = _top_state(state["books"].get(ticker, _empty_book()))
                if top["valid_bbo"]:
                    mismatch = (
                        abs(float(top["yes_bid"]) - tbid) * 100.0 > TICKER_MISMATCH_TOL_C
                        or abs(float(top["yes_ask"]) - task) * 100.0 > TICKER_MISMATCH_TOL_C
                    )
                    if mismatch:
                        start = state["ticker_mismatch_start_mono"].setdefault(ticker, now_mono)
                        last = state["ticker_last_resync_mono"].get(ticker, -1e30)
                        if now_mono - start >= TICKER_MISMATCH_PERSIST_S and now_mono - last >= TICKER_RESYNC_COOLDOWN_S:
                            state["counters"]["ticker_persistent_mismatch_resyncs"] += 1
                            state["ticker_last_resync_mono"][ticker] = now_mono
                            _record_repair(files, state, receipt, ticker, "ticker_persistent_bbo_mismatch", after=top)
                            _invalidate_tickers(state, {ticker})
                            _schedule_snapshot_refresh(ws, lock, state, {ticker}, "ticker_persistent_bbo_mismatch")
                    else:
                        state["ticker_mismatch_start_mono"].pop(ticker, None)
            continue

        if typ == "trade":
            price = _f(msg.get("yes_price_dollars") or msg.get("price_dollars"))
            qty = _q(msg.get("count_fp") or msg.get("count"))
            if price is None or qty is None or qty <= 0:
                state["counters"]["bad_trades"] += 1
                continue
            files["trades_event_time"].write({
                "receipt_time": _iso(receipt), "exchange_time": _iso(_exchange_time(msg)) if _exchange_time(msg) else None,
                "ticker": ticker, "series_ticker": meta.get("series_ticker"), "close_time": _iso(meta.get("close_time")),
                "elapsed_s": e, "capture_phase": phase, "trade_id": msg.get("trade_id"),
                "yes_price": price, "no_price": _f(msg.get("no_price_dollars")), "qty": qty,
                "taker_book_side": _trade_book_side(msg),
                "taker_outcome_side": msg.get("taker_outcome_side") or msg.get("taker_side"),
                "is_block_trade": msg.get("is_block_trade"), "ts_ms": msg.get("ts_ms"),
            })
            state["counters"]["trade_rows"] += 1


async def _health_writer(runtime, session_dir, stop_event):
    path = session_dir / "health.json"
    while not stop_event.is_set():
        state = runtime.get("state")
        now_mono = time.monotonic()
        if state is None:
            obj = {"time": _iso(), "pid": os.getpid(), "running": True, "healthy": False, "connection_epoch": runtime.get("connection_epoch", 0)}
        else:
            md_age = now_mono - state["last_market_data_mono"] if state.get("last_market_data_mono") is not None else None
            active_research = 0
            active_tail = 0
            now = C.utc_now()
            for m in state["meta"].values():
                e = _elapsed(m, now)
                if e is None:
                    continue
                if TRADE_WINDOW_START_S <= e < TRADE_WINDOW_END_S:
                    active_research += 1
                elif TRADE_WINDOW_END_S <= e < LABEL_TAIL_END_S:
                    active_tail += 1
            healthy = (
                state.get("connected", False)
                and state.get("supervisor_heartbeat_mono") is not None
                and now_mono - state["supervisor_heartbeat_mono"] < 10.0
                and state.get("last_scan_error") is None
            )
            c = state["counters"]
            obj = {
                "time": _iso(), "pid": os.getpid(), "running": True, "healthy": bool(healthy),
                "study_version": STUDY_VERSION, "connection_epoch": state["connection_epoch"],
                "subscribed_markets": len(state["markets"]), "active_m0_m5_markets": active_research,
                "active_label_tail_markets": active_tail, "channels": sorted(state["sids"]),
                "market_data_age_s": md_age, "last_scan_error": state.get("last_scan_error"),
                "book_rows": c["book_rows"], "ticker_rows": c["ticker_rows"], "trade_rows": c["trade_rows"],
                "snapshots_received": c["snapshots_received"], "deltas_received": c["deltas_received"],
                "top3_changing_deltas": c["top3_changing_deltas"], "sequence_gaps": c["sequence_gaps"],
                "sequence_numbers_missing": c["sequence_numbers_missing"], "ok_seq_messages": c["ok_seq_messages"],
                "snapshot_requests": c["snapshot_requests"], "crossed_after_delta_resyncs": c["crossed_after_delta_resyncs"],
                "negative_level_resyncs": c["negative_level_resyncs"],
                "ticker_persistent_mismatch_resyncs": c["ticker_persistent_mismatch_resyncs"],
            }
        _atomic_json(path, obj)
        await asyncio.sleep(HEALTH_INTERVAL_S)


async def _connection_once(key_id, private_key, runtime, files, stop_event):
    ws = await C.open_ws(key_id, private_key)
    lock = asyncio.Lock()
    runtime["connection_epoch"] += 1
    state = {
        "connection_epoch": runtime["connection_epoch"], "connected": True, "next_id": 1,
        "sids": {}, "sid_channels": {}, "pending": set(), "markets": set(),
        "meta": runtime["known_meta"], "books": {}, "book_initialized": {}, "last_seq_by_sid": {},
        "capture_started": runtime["capture_started"], "trade_window_ended": runtime["trade_window_ended"],
        "tail_ended": runtime["tail_ended"], "last_ws_mono": time.monotonic(), "last_market_data_mono": None,
        "supervisor_heartbeat_mono": None, "last_scan_mono": None, "last_scan_error": None,
        "last_snapshot_request_reason": None, "last_snapshot_request_mono": None,
        "snapshot_refresh_tickers": set(), "snapshot_refresh_reasons": set(), "snapshot_batch_task": None,
        "last_book_event_mono": {}, "ticker_mismatch_start_mono": {}, "ticker_last_resync_mono": {},
        "counters": runtime["counters"], "files": files,
    }
    runtime["state"] = state
    files["connection_events"].write({"time": _iso(), "type": "connected", "connection_epoch": state["connection_epoch"]})

    tasks = [
        asyncio.create_task(_supervisor(ws, lock, state, files, stop_event)),
        asyncio.create_task(_consumer(ws, lock, state, files, stop_event)),
        asyncio.create_task(_boundary_writer(state, files, stop_event)),
    ]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            exc = t.exception()
            if exc is not None:
                raise exc
    finally:
        state["connected"] = False
        batch = state.get("snapshot_batch_task")
        if batch is not None:
            batch.cancel()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if batch is not None:
            await asyncio.gather(batch, return_exceptions=True)
        try:
            await ws.close()
        except Exception:
            pass
        files["connection_events"].write({"time": _iso(), "type": "disconnected", "connection_epoch": state["connection_epoch"]})


async def run_event_time_m0_m5_v5_recorder(session_dir: Path):
    session_dir = Path(session_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "book_top3_events": Jsonl(session_dir / "book_top3_events.jsonl"),
        "trades_event_time": Jsonl(session_dir / "trades_event_time.jsonl"),
        "ticker_event_time": Jsonl(session_dir / "ticker_event_time.jsonl"),
        "market_metadata": Jsonl(session_dir / "market_metadata.jsonl"),
        "market_rotations": Jsonl(session_dir / "market_rotations.jsonl"),
        "connection_events": Jsonl(session_dir / "connection_events.jsonl"),
        "book_repair_events": Jsonl(session_dir / "book_repair_events.jsonl"),
    }

    started = C.utc_now()
    development_plan = {
        "research_stage": "V5_DEVELOPMENT_NOT_OOS",
        "economic_target_usd_per_day": 100.0,
        "candidate_family": {
            "A": "L3 side support only",
            "B": "L1 and L3 both support the quoted side",
            "C": "L3 side support and natural spread >= 2c",
            "D": "L1 and L3 both support the quoted side and natural spread >= 2c",
        },
        "threshold_policy": "No additional numeric threshold sweep on this session",
        "execution_reference": "Q1 at public BBO with conservative FIFO/no cancellation-ahead credit",
        "capacity_scenarios": [1, 2, 5, 10],
        "capacity_warning": "Q2/Q5/Q10 are counterfactual; never assume linear scaling from Q1",
        "minimum_economic_requirements": [
            "plausible path to >= $100/day", "matched round-trip PnL positive",
            "residual inventory not carrying the result", "material fee/slippage cushion",
            "healthy 5s/15s/30s markouts", "chronological stability", "no single asset carrying the result",
        ],
    }
    capture_spec = {
        "study_version": STUDY_VERSION,
        "purpose": "fresh corrected event-time DEVELOPMENT data for $100/day MM capacity research",
        "universe": list(CRYPTO_SERIES), "research_window": "M0 <= elapsed < M5",
        "research_elapsed_seconds": [0.0, 300.0],
        "label_tail": "M5 <= elapsed < M5+30s; labels only, never quote initiation",
        "persisted_elapsed_seconds": [0.0, 330.0], "pre_subscribe_lead_seconds": PRESUBSCRIBE_LEAD_S,
        "orderbook_channel": "orderbook_delta", "orderbook_use_yes_price": True,
        "orderbook_numeric_representation": "Decimal exact price and quantity in RAM",
        "sequence_accounting": "all messages carrying orderbook sid+seq, including type=ok",
        "sequence_gap_recovery": "invalidate all books; debounce; get_snapshot all subscribed markets",
        "reconstruction_recovery": "negative level or crossed/locked reconstruction -> ticker-specific fresh snapshot",
        "ticker_integrity_recovery": ">1.01c BBO mismatch persisting >=250ms -> ticker-specific fresh snapshot",
        "sticky_market_lifecycle": "never remove a subscribed market before label_tail_end boundary is written",
        "persisted_book": "top 3 bid/ask levels only when top3 changes + snapshots/boundaries",
        "trades": "every public trade at event time during M0-M5+30s",
        "ticker": "every ticker event during M0-M5+30s for independent BBO validation",
        "strategy_pnl_recorded": False, "development_plan_file": "development_plan.json",
    }
    _atomic_json(session_dir / "capture_spec.json", capture_spec)
    _atomic_json(session_dir / "development_plan.json", development_plan)
    _atomic_json(session_dir / "session_manifest.json", {
        "study_version": STUDY_VERSION, "research_stage": "V5_DEVELOPMENT_NOT_OOS",
        "session_dir": str(session_dir), "started_at": _iso(started), "pid": os.getpid(),
        "capture_spec": capture_spec, "development_plan": development_plan,
    })

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except Exception:
            pass

    key_id, private_key = C.load_auth()
    runtime = {
        "connection_epoch": 0, "state": None, "counters": Counter(),
        "capture_started": set(), "trade_window_ended": set(), "tail_ended": set(), "known_meta": {},
    }
    health_task = asyncio.create_task(_health_writer(runtime, session_dir, stop_event))
    try:
        while not stop_event.is_set():
            try:
                await _connection_once(key_id, private_key, runtime, files, stop_event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                files["connection_events"].write({
                    "time": _iso(), "type": "connection_exception",
                    "connection_epoch": runtime.get("connection_epoch", 0), "error": repr(exc),
                })
                if not stop_event.is_set():
                    await asyncio.sleep(3.0)
    finally:
        stop_event.set()
        health_task.cancel()
        await asyncio.gather(health_task, return_exceptions=True)
        ended = C.utc_now()
        final_counts = dict(runtime["counters"])
        _atomic_json(session_dir / "session_manifest.json", {
            "study_version": STUDY_VERSION, "research_stage": "V5_DEVELOPMENT_NOT_OOS",
            "session_dir": str(session_dir), "started_at": _iso(started), "ended_at": _iso(ended),
            "duration_hours": (ended - started).total_seconds() / 3600.0, "pid": os.getpid(),
            "connection_epochs": runtime.get("connection_epoch", 0), "final_counts": final_counts,
            "capture_spec": capture_spec, "development_plan": development_plan,
        })
        _atomic_json(session_dir / "health.json", {
            "time": _iso(ended), "pid": os.getpid(), "running": False, "healthy": False,
            "study_version": STUDY_VERSION, "final_counts": final_counts,
        })
        for f in files.values():
            f.close()


def event_time_m0_m5_v5_status():
    ctl = _read_json(CONTROL_PATH, {}) or {}
    if not ctl:
        print("No V5 development recorder control file.")
        return {"running": False}
    session = Path(ctl.get("session_dir", ""))
    health = _read_json(session / "health.json", {}) or {}
    pid_state = _pid_state(ctl.get("pid"))
    out = {**ctl, **health, "pid_state": pid_state, "pid_alive": _pid_alive(ctl.get("pid"))}
    out["running"] = bool(out.get("pid_alive") and health.get("running", True))
    print(json.dumps(out, indent=2, default=str))
    return out


def start_event_time_m0_m5_v5_recording(*, startup_timeout_s=STARTUP_TIMEOUT_S):
    ctl = _read_json(CONTROL_PATH, {}) or {}
    if ctl and _pid_alive(ctl.get("pid")):
        raise RuntimeError(f"V5 recorder already running: pid={ctl.get('pid')} session={ctl.get('session_dir')}")
    if CONTROL_PATH.exists():
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass

    session_dir = _session_dir_now().resolve()
    if session_dir.exists():
        raise FileExistsError(session_dir)
    log_path = session_dir.parent / f"{session_dir.name}.startup.log"
    cmd = [sys.executable, "-m", "quant_research.kalshi.mm_event_time_m0_m5_recorder_v5", "--run-session", str(session_dir)]
    log_fh = log_path.open("a", buffering=1, encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log_fh.close()

    _atomic_json(CONTROL_PATH, {
        "pid": proc.pid, "session_dir": str(session_dir), "started_at": _iso(), "launcher_pid": os.getpid(),
        "log_path": str(log_path), "study_version": STUDY_VERSION, "research_stage": "V5_DEVELOPMENT_NOT_OOS",
    })

    deadline = time.time() + float(startup_timeout_s)
    last_health = None
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-60:])
            except Exception:
                pass
            raise RuntimeError(f"V5 recorder exited during startup with code {proc.returncode}.\n{tail}")
        hp = session_dir / "health.json"
        if hp.exists():
            last_health = _read_json(hp, {}) or {}
            if last_health.get("healthy"):
                print("Event-time M0-M5 recorder V5 DEVELOPMENT is healthy.")
                print(f"PID: {proc.pid}")
                print(f"SESSION: {session_dir}")
                print("RESEARCH WINDOW: M0-M5 [0s,300s)")
                print("LABEL TAIL: M5-M5+30s [300s,330s) for 30s markouts only")
                print("BOOK: exact Decimal full book in RAM -> persist top3 changes only")
                print("SEQUENCE: orderbook sid seq includes OK/control responses")
                print("REPAIR: gaps/crosses/negative levels/persistent ticker mismatch -> snapshot")
                print("THIS SESSION IS DEVELOPMENT, NOT OOS VALIDATION")
                print("ECONOMIC TARGET: >= $100/day plausible capacity")
                print(f"LOG: {log_path}")
                return session_dir
        time.sleep(0.5)

    if proc.poll() is None:
        print("V5 process is running but healthy=True was not reached before timeout.")
        print(f"SESSION: {session_dir}")
        print(f"LAST HEALTH: {last_health}")
        print(f"LOG: {log_path}")
        return session_dir
    raise RuntimeError("V5 recorder failed to start")


def stop_event_time_m0_m5_v5_recording(*, expected_session=None, timeout_s=STOP_TIMEOUT_S):
    ctl = _read_json(CONTROL_PATH, {}) or {}
    if not ctl:
        print("No active V5 development recorder control file.")
        return None
    session = Path(ctl.get("session_dir", "")).resolve()
    if expected_session is not None and session != Path(expected_session).resolve():
        raise RuntimeError(f"Refusing stop: active={session}, expected={Path(expected_session).resolve()}")
    pid = ctl.get("pid")
    if not _pid_alive(pid):
        print(f"V5 pid is already dead/zombie: {pid}. Session preserved: {session}")
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass
        return session

    print(f"Stopping V5 recorder pid={pid} ...")
    for sig, wait_s, label in ((signal.SIGINT, float(timeout_s), "SIGINT"), (signal.SIGTERM, 10.0, "SIGTERM")):
        try:
            os.kill(int(pid), sig)
        except ProcessLookupError:
            break
        except Exception:
            pass
        deadline = time.time() + wait_s
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.25)
        if not _pid_alive(pid):
            break
        print(f"{label} did not finish before timeout.")

    if _pid_alive(pid):
        p = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
        cmd = p.stdout.strip()
        if "mm_event_time_m0_m5_recorder_v5" not in cmd:
            raise RuntimeError(f"Refusing SIGKILL: pid={pid} no longer looks like V5 recorder: {cmd}")
        print("Sending verified SIGKILL as final fallback.")
        try:
            os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass
        deadline = time.time() + 5.0
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.1)

    if _pid_alive(pid):
        raise RuntimeError(f"V5 recorder pid={pid} did not stop")
    try:
        CONTROL_PATH.unlink()
    except Exception:
        pass
    print(f"SAVED SESSION: {session}")
    manifest = _read_json(session / "session_manifest.json", {}) or {}
    if manifest.get("duration_hours") is not None:
        print(f"DURATION: {float(manifest['duration_hours']):.3f} h")
    return session


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-session", type=str, default=None)
    args = ap.parse_args()
    if args.run_session:
        asyncio.run(run_event_time_m0_m5_v5_recorder(Path(args.run_session)))
    else:
        event_time_m0_m5_v5_status()


if __name__ == "__main__":
    _main()
