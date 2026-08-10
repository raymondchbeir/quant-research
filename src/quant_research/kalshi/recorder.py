from __future__ import annotations

import asyncio
import json
import time
import traceback

from . import recorder_core as C

_TASK = None
_STOP = None
_SESSION = None
_STATE = None


async def _send(ws, lock, obj):
    async with lock:
        await ws.send(json.dumps(obj))


async def _subscribe(ws, lock, state, ch, tickers):
    if not tickers or ch in state["pending"]:
        return
    i = state["next_id"]
    state["next_id"] += 1
    params = {"channels": [ch], "market_tickers": sorted(tickers)}
    if ch == "orderbook_delta":
        params["use_yes_price"] = True
    state["pending"].add(ch)
    state["pending_ids"][i] = ch
    await _send(ws, lock, {"id": i, "cmd": "subscribe", "params": params})


async def _update(ws, lock, state, ch, action, tickers):
    if not tickers or state["sids"].get(ch) is None:
        return
    i = state["next_id"]
    state["next_id"] += 1
    await _send(
        ws,
        lock,
        {
            "id": i,
            "cmd": "update_subscription",
            "params": {
                "sid": state["sids"][ch],
                "market_tickers": sorted(tickers),
                "action": action,
            },
        },
    )


async def _snapshots(ws, lock, state, tickers):
    sid = state["sids"].get("orderbook_delta")
    if not tickers or sid is None:
        return
    i = state["next_id"]
    state["next_id"] += 1
    await _send(
        ws,
        lock,
        {
            "id": i,
            "cmd": "update_subscription",
            "params": {
                "sid": sid,
                "market_tickers": sorted(tickers),
                "action": "get_snapshot",
            },
        },
    )


async def _supervisor(ws, lock, state, files):
    latest = []
    last_scan = 0.0

    while not _STOP.is_set():
        state["supervisor_heartbeat_mono"] = time.monotonic()
        now_mono = time.monotonic()

        if not latest or now_mono - last_scan >= C.MARKET_RESCAN_SECONDS:
            scan_started = time.monotonic()
            try:
                latest = await asyncio.wait_for(C.scan_open_15m_markets(), timeout=12.0)
                last_scan = time.monotonic()
                state["last_scan_ok_mono"] = last_scan
                state["last_scan_seconds"] = last_scan - scan_started
                state["last_scan_error"] = None
            except asyncio.TimeoutError:
                state["last_scan_error"] = "market scan timed out after 12s"
                C.write_jsonl(
                    files["connection_events"],
                    {"time": C.iso_utc(), "type": "discovery_timeout", "timeout_seconds": 12},
                )
            except Exception as exc:
                state["last_scan_error"] = repr(exc)
                C.write_jsonl(
                    files["connection_events"],
                    {"time": C.iso_utc(), "type": "discovery_error", "error": repr(exc)},
                )

        # Even if a fresh scan fails, expire old contracts from the previous
        # successful scan immediately. This prevents stale closed markets from
        # remaining in recorder state.
        now = C.utc_now()
        meta = {
            m["ticker"]: m
            for m in latest
            if m.get("close_time") is None or m["close_time"] > now
        }
        desired = set(meta)
        current = set(state["markets"])
        add = desired - current
        delete = current - desired
        state["meta"].update(meta)

        if desired:
            for ch in ("orderbook_delta", "trade", "ticker"):
                if ch not in state["sids"] and ch not in state["pending"]:
                    await _subscribe(ws, lock, state, ch, desired)

        if add:
            for ch in ("orderbook_delta", "trade", "ticker"):
                await _update(ws, lock, state, ch, "add_markets", add)
            await _snapshots(ws, lock, state, add)

        if delete:
            for ch in ("orderbook_delta", "trade", "ticker"):
                await _update(ws, lock, state, ch, "delete_markets", delete)
            for ticker in delete:
                state["books"].pop(ticker, None)

        # Always commit desired state, even when there was no websocket update.
        state["markets"] = desired

        if add or delete:
            state["last_rotation_mono"] = time.monotonic()
            C.write_jsonl(
                files["market_rotations"],
                {
                    "time": C.iso_utc(now),
                    "active_count": len(desired),
                    "added": sorted(add),
                    "removed": sorted(delete),
                    "active": [
                        {
                            "ticker": x["ticker"],
                            "series_ticker": x.get("series_ticker"),
                            "series_title": x.get("series_title"),
                            "close_time": C.iso_utc(x.get("close_time")) if x.get("close_time") else None,
                        }
                        for x in meta.values()
                    ],
                },
            )
            print(
                f"[{now:%H:%M:%S} UTC] 15m active={len(desired)} "
                f"added={len(add)} removed={len(delete)}"
            )
            for ticker in sorted(add):
                print(f" + {ticker} | {meta[ticker].get('series_title', '')}")
            for ticker in sorted(delete):
                print(f" - {ticker}")

        state["supervisor_heartbeat_mono"] = time.monotonic()
        await asyncio.sleep(C.SUPERVISOR_INTERVAL_SECONDS)


async def _book_sampler(state, files):
    while not _STOP.is_set():
        now = C.utc_now()
        for ticker in list(state["markets"]):
            book = state["books"].get(ticker)
            meta = state["meta"].get(ticker)
            if book is None or meta is None:
                continue
            if meta.get("close_time") is not None and now >= meta["close_time"]:
                continue
            C.write_jsonl(files["full_books"], C.full_book_row(ticker, book, meta))
        await asyncio.sleep(C.FULL_BOOK_INTERVAL_SECONDS)


async def _regime_sampler(state, files):
    while not _STOP.is_set():
        now = C.utc_now()
        for ticker in sorted(state["markets"]):
            meta = state["meta"].get(ticker)
            if not meta:
                continue
            close_time = meta.get("close_time")
            remaining = C.seconds_until(close_time, now) if close_time else None
            C.write_jsonl(
                files["regime_snapshots"],
                {
                    "time": C.iso_utc(now),
                    "ticker": ticker,
                    "event_ticker": meta.get("event_ticker"),
                    "series_ticker": meta.get("series_ticker"),
                    "series_title": meta.get("series_title"),
                    "series_category": meta.get("series_category"),
                    "series_frequency": meta.get("series_frequency"),
                    "market_title": meta.get("market_title"),
                    "market_open": C.iso_utc(meta.get("open_time")) if meta.get("open_time") else None,
                    "market_close": C.iso_utc(close_time) if close_time else None,
                    "seconds_to_close": remaining,
                    "minutes_to_close": remaining / 60 if remaining is not None else None,
                    "regime": C.regime_from_close(close_time, now),
                },
            )
        await asyncio.sleep(C.REGIME_INTERVAL_SECONDS)


async def _consumer(ws, lock, state, files):
    async for raw in ws:
        if _STOP.is_set():
            break

        state["last_ws_message_mono"] = time.monotonic()

        try:
            data = json.loads(raw)
        except Exception:
            continue

        typ = data.get("type")
        msg = data.get("msg", {})
        sid = data.get("sid")
        seq = data.get("seq")

        if typ == "subscribed":
            ch = msg.get("channel")
            resolved_sid = msg.get("sid", sid)
            if ch and resolved_sid is not None:
                state["sids"][ch] = resolved_sid
                state["pending"].discard(ch)
                for i, pending_ch in list(state["pending_ids"].items()):
                    if pending_ch == ch:
                        state["pending_ids"].pop(i, None)
                C.write_jsonl(
                    files["connection_events"],
                    {"time": C.iso_utc(), "type": "subscribed", "channel": ch, "sid": resolved_sid},
                )
            continue

        if typ == "error":
            C.write_jsonl(
                files["connection_events"],
                {"time": C.iso_utc(), "type": "ws_error", "payload": data},
            )
            continue

        if typ in {"orderbook_snapshot", "orderbook_delta", "trade", "ticker"}:
            state["last_market_data_mono"] = time.monotonic()

        if typ in {"orderbook_snapshot", "orderbook_delta"} and seq is not None:
            last = state.get("seq")
            if last is not None and typ == "orderbook_delta" and seq != last + 1:
                C.write_jsonl(
                    files["connection_events"],
                    {"time": C.iso_utc(), "type": "sequence_gap", "last_seq": last, "new_seq": seq},
                )
                state["books"].clear()
                await _snapshots(ws, lock, state, state["markets"])
                state["seq"] = seq
                continue
            state["seq"] = seq

        if typ == "orderbook_snapshot":
            C.apply_snapshot(state["books"], msg)
            continue

        ticker = msg.get("market_ticker")
        meta = state["meta"].get(ticker, {}) if ticker else {}
        close_time = meta.get("close_time")

        if typ == "orderbook_delta":
            C.write_jsonl(
                files["book_deltas"],
                {
                    "time": C.iso_utc(),
                    "ticker": ticker,
                    "event_ticker": meta.get("event_ticker"),
                    "series_ticker": meta.get("series_ticker"),
                    "market_close": C.iso_utc(close_time) if close_time else None,
                    "sid": sid,
                    "seq": seq,
                    "side": msg.get("side"),
                    "price_dollars": msg.get("price_dollars"),
                    "delta_fp": msg.get("delta_fp"),
                    "ts_ms": msg.get("ts_ms"),
                    "raw_msg": msg,
                },
            )
            C.apply_delta(state["books"], msg)
            continue

        if typ == "trade":
            outcome_side = str(msg.get("taker_outcome_side") or msg.get("taker_side") or "").lower()
            qty = msg.get("count_fp") or msg.get("count")
            price = msg.get("yes_price_dollars") or msg.get("price_dollars")
            C.write_jsonl(
                files["trades"],
                {
                    "time": C.iso_utc(),
                    "ticker": ticker,
                    "event_ticker": meta.get("event_ticker"),
                    "series_ticker": meta.get("series_ticker"),
                    "series_title": meta.get("series_title"),
                    "series_category": meta.get("series_category"),
                    "market_close": C.iso_utc(close_time) if close_time else None,
                    "seconds_to_close": C.seconds_until(close_time) if close_time else None,
                    "regime": C.regime_from_close(close_time),
                    "yes_price": C.safe_float(price, None),
                    "qty": C.safe_float(qty, None),
                    "volume": C.safe_float(qty, None),
                    "action": "BUY" if outcome_side == "yes" else "SELL" if outcome_side == "no" else "UNKNOWN",
                    "taker_side": outcome_side,
                    "taker_book_side": msg.get("taker_book_side"),
                    "trade_id": msg.get("trade_id"),
                    "ts_ms": msg.get("ts_ms"),
                    "raw_msg": msg,
                },
            )
            continue

        if typ == "ticker":
            C.write_jsonl(
                files["ticker_updates"],
                {
                    "time": C.iso_utc(),
                    "ticker": ticker,
                    "event_ticker": meta.get("event_ticker"),
                    "series_ticker": meta.get("series_ticker"),
                    "market_close": C.iso_utc(close_time) if close_time else None,
                    "seconds_to_close": C.seconds_until(close_time) if close_time else None,
                    "regime": C.regime_from_close(close_time),
                    "yes_bid_dollars": msg.get("yes_bid_dollars"),
                    "yes_ask_dollars": msg.get("yes_ask_dollars"),
                    "yes_bid_size_fp": msg.get("yes_bid_size_fp"),
                    "yes_ask_size_fp": msg.get("yes_ask_size_fp"),
                    "price_dollars": msg.get("price_dollars"),
                    "volume_fp": msg.get("volume_fp"),
                    "result": msg.get("result"),
                    "ts_ms": msg.get("ts_ms"),
                    "raw_msg": msg,
                },
            )


async def _watchdog(state, files):
    """Force a reconnect instead of silently leaving expired contracts in state."""
    while not _STOP.is_set():
        await asyncio.sleep(5)
        now_mono = time.monotonic()
        supervisor_age = now_mono - state.get("supervisor_heartbeat_mono", now_mono)

        if supervisor_age > 25:
            msg = f"supervisor heartbeat stale for {supervisor_age:.1f}s"
            C.write_jsonl(files["connection_events"], {"time": C.iso_utc(), "type": "watchdog_reconnect", "reason": msg})
            raise RuntimeError(msg)

        now = C.utc_now()
        expired = []
        for ticker in state.get("markets", set()):
            close_time = (state.get("meta", {}).get(ticker) or {}).get("close_time")
            if close_time is not None and (now - close_time).total_seconds() > 30:
                expired.append(ticker)

        if expired:
            msg = f"expired contracts remained active: {sorted(expired)[:5]}"
            C.write_jsonl(files["connection_events"], {"time": C.iso_utc(), "type": "watchdog_reconnect", "reason": msg})
            raise RuntimeError(msg)


async def _connection(files, key_id, key):
    global _STATE

    ws = await C.open_ws(key_id, key)
    lock = asyncio.Lock()
    now_mono = time.monotonic()
    state = {
        "books": {},
        "meta": {},
        "markets": set(),
        "sids": {},
        "pending": set(),
        "pending_ids": {},
        "next_id": 1,
        "seq": None,
        "connected_mono": now_mono,
        "supervisor_heartbeat_mono": now_mono,
        "last_scan_ok_mono": None,
        "last_scan_seconds": None,
        "last_scan_error": None,
        "last_rotation_mono": None,
        "last_ws_message_mono": None,
        "last_market_data_mono": None,
    }
    _STATE = state

    C.write_jsonl(files["connection_events"], {"time": C.iso_utc(), "type": "connected"})
    print("Kalshi WS connected.")

    tasks = [
        asyncio.create_task(_supervisor(ws, lock, state, files)),
        asyncio.create_task(_book_sampler(state, files)),
        asyncio.create_task(_regime_sampler(state, files)),
        asyncio.create_task(_consumer(ws, lock, state, files)),
        asyncio.create_task(_watchdog(state, files)),
    ]

    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.exception() is not None:
                raise task.exception()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await ws.close()
        except Exception:
            pass


async def run_recorder(duration_minutes=None, key_id=None, private_key_path=None):
    global _SESSION

    kid, key = C.load_auth(key_id, private_key_path)
    session_dir, files = C.make_session()
    _SESSION = session_dir
    started = C.utc_now()

    try:
        series = await C.discover_15m_series(True)
    except Exception:
        series = []

    manifest = {
        "version": "V3.6_DYNAMIC_15M_WATCHDOG",
        "started_at": C.iso_utc(started),
        "ended_at": None,
        "duration_minutes_requested": duration_minutes,
        "market_rescan_seconds": C.MARKET_RESCAN_SECONDS,
        "series_rescan_seconds": C.SERIES_RESCAN_SECONDS,
        "full_book_interval_seconds": C.FULL_BOOK_INTERVAL_SECONDS,
        "dynamic_rotation": True,
        "use_yes_price": True,
        "initial_discovered_series": [
            {
                "ticker": s.get("ticker"),
                "title": s.get("title"),
                "frequency": s.get("frequency"),
                "category": s.get("category"),
            }
            for s in series
        ],
    }
    manifest_path = session_dir / "session_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"Recorder session: {session_dir}")
    print(f"15m series discovered: {len(series)}")
    if duration_minutes is None:
        print("Recorder duration: manual stop")
    else:
        print(f"Recorder duration: {float(duration_minutes):g} minutes")

    async def timer():
        if duration_minutes is None:
            return
        await asyncio.sleep(float(duration_minutes) * 60)
        _STOP.set()

    timer_task = asyncio.create_task(timer())

    try:
        while not _STOP.is_set():
            try:
                await _connection(files, kid, key)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                C.write_jsonl(
                    files["connection_events"],
                    {
                        "time": C.iso_utc(),
                        "type": "connection_exception",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                if not _STOP.is_set():
                    print(f"Recorder reconnecting after: {exc}")
            if not _STOP.is_set():
                await asyncio.sleep(C.RECONNECT_DELAY_SECONDS)
    finally:
        timer_task.cancel()
        await asyncio.gather(timer_task, return_exceptions=True)
        ended = C.utc_now()
        manifest["ended_at"] = C.iso_utc(ended)
        manifest["actual_duration_minutes"] = (ended - started).total_seconds() / 60
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        C.close_files(files)
        print(f"15-minute recorder stopped. Saved to: {session_dir}")

    return session_dir


async def start_recorder(duration_minutes=None, key_id=None, private_key_path=None):
    global _TASK, _STOP, _SESSION

    if _TASK is not None and not _TASK.done():
        print(f"Recorder is already running: {_SESSION}")
        return _TASK

    _STOP = asyncio.Event()
    _SESSION = None
    _TASK = asyncio.create_task(run_recorder(duration_minutes, key_id, private_key_path))

    for _ in range(100):
        await asyncio.sleep(0.05)
        if _SESSION is not None:
            break
        if _TASK.done():
            await _TASK

    return _TASK


async def stop_recorder(expected_session=None):
    if expected_session is not None and _SESSION is not None:
        from pathlib import Path
        if Path(expected_session).resolve() != Path(_SESSION).resolve():
            raise RuntimeError(
                f"Recorder session mismatch: expected {Path(expected_session).resolve()} but active recorder is {Path(_SESSION).resolve()}. "
                "Refusing ambiguous stop. Restart the kernel if stale module state exists."
            )

    if _TASK is None or _TASK.done():
        print(f"Recorder is not running. Last session: {_SESSION}")
        return _SESSION

    _STOP.set()
    await _TASK
    print(f"Saved to: {_SESSION}")
    return _SESSION


def current_session_dir():
    return _SESSION


def recorder_status():
    state = _STATE or {}
    now_mono = time.monotonic()
    now = C.utc_now()

    supervisor_age = None
    if state.get("supervisor_heartbeat_mono") is not None:
        supervisor_age = now_mono - state["supervisor_heartbeat_mono"]

    market_data_age = None
    if state.get("last_market_data_mono") is not None:
        market_data_age = now_mono - state["last_market_data_mono"]

    expired_active = 0
    for ticker in state.get("markets", set()):
        close_time = (state.get("meta", {}).get(ticker) or {}).get("close_time")
        if close_time is not None and close_time <= now:
            expired_active += 1

    running = _TASK is not None and not _TASK.done()
    channels = sorted(state.get("sids", {}))
    healthy = (
        running
        and expired_active == 0
        and (supervisor_age is None or supervisor_age < 25)
        and {"orderbook_delta", "ticker", "trade"}.issubset(set(channels))
    )

    out = {
        "running": running,
        "healthy": healthy,
        "session_dir": str(_SESSION) if _SESSION else None,
        "active_markets": len(state.get("markets", [])),
        "expired_active_markets": expired_active,
        "books_in_memory": len(state.get("books", {})),
        "channels": channels,
        "supervisor_age_s": None if supervisor_age is None else round(supervisor_age, 2),
        "market_data_age_s": None if market_data_age is None else round(market_data_age, 2),
        "last_scan_s": state.get("last_scan_seconds"),
        "last_scan_error": state.get("last_scan_error"),
    }
    print(out)
    return out


preview_15m_markets = C.preview_15m_markets
