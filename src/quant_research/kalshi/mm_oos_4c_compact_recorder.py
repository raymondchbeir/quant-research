from __future__ import annotations

"""Compact prospective OOS recorder for the frozen M1-M5 4c MM candidate.

Purpose
-------
Record a fresh, storage-conscious dataset for ONE frozen prospective strategy:
Defensive MM V1 with a 4.00c minimum spread. This recorder is deliberately not
an event-time reconstruction archive.

Storage policy
--------------
- fixed crypto 15-minute universe only;
- subscribe to ticker + trade channels only (no orderbook_delta persistence);
- keep current/next contracts subscribed for clean rotations;
- persist market data only from M0 through M6 for each 15-minute contract;
  M1-M5 is the trading window and M5-M6 supports the 60s markout;
- write ONE compact BBO record per wall-clock second containing all captured
  markets, including YES bid/ask and displayed L1 sizes from the latest ticker;
- aggregate aggressive trades into ONE-second buckets by ticker, taker-book
  side, and YES price before writing;
- write small metadata / connection / health files.

The resulting dataset is intended for the frozen 1 Hz OOS replay only. It is
NOT sufficient for later event-time delta reconstruction or arbitrary strategy
retuning. The frozen strategy definition is written into every session.

The module exposes notebook-friendly synchronous helpers:
    start_oos_mm4c_recording()
    stop_oos_mm4c_recording()
    oos_mm4c_recording_status()

The start helper launches a detached child Python process, so the recording is
not tied to the lifetime of the notebook cell or browser tab. The stop helper
uses a dedicated PID/control file under data/kalshi_15m/oos_mm4c_compact.
"""

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import recorder_core as C

STUDY_VERSION = "OOS_MM4C_COMPACT_1HZ_V1"
SAMPLE_INTERVAL_SECONDS = 1.0
MARKET_RESCAN_SECONDS = 15.0
SUBSCRIBE_HORIZON_SECONDS = 20.0 * 60.0
CAPTURE_PRESTART_SECONDS = 30.0
CAPTURE_END_ELAPSED_SECONDS = 6.0 * 60.0
STARTUP_TIMEOUT_SECONDS = 35.0
STOP_TIMEOUT_SECONDS = 45.0

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

FROZEN_STRATEGY = {
    "strategy_id": "M1_M5_DEFENSIVE_V1_MIN_SPREAD_4C_OOS_V1",
    "status": "FROZEN_PROSPECTIVE_OOS",
    "universe": list(CRYPTO_SERIES),
    "quote_window": "M1-M5 elapsed [60s, 300s)",
    "quote_qty": 1.0,
    "quote_price": "current 1Hz YES BBO",
    "min_spread_c": 4.0,
    "momentum_lookback_s": 3.0,
    "max_adverse_momentum_c": 1.0,
    "flow_lookback_s": 5.0,
    "max_adverse_flow_imbalance": 0.60,
    "inventory_soft_limit": 2.0,
    "inventory_hard_limit": 3.0,
    "same_side_cooldown_s": 3.0,
    "queue_model": (
        "join back of displayed L1; exact-price aggressive flow depletes queue; "
        "no cancellation-ahead credit; trade-through fills remainder"
    ),
    "inventory_at_m5": "mark to latest valid 1Hz midpoint",
    "fees": "excluded from research replay; must be assessed before deployment",
    "forbidden_same_session_retuning": True,
}

PROJECT_ROOT = C.PROJECT_ROOT
OOS_ROOT = C.DATA_ROOT / "oos_mm4c_compact"
CONTROL_PATH = OOS_ROOT / "active_recorder.json"
OOS_ROOT.mkdir(parents=True, exist_ok=True)


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


def _session_dir_now():
    return OOS_ROOT / C.utc_now().strftime("%Y%m%d_%H%M%S")


def _market_row(series, m):
    ticker = str(m.get("ticker") or "")
    close = C.parse_time(m.get("close_time"))
    if not ticker or close is None:
        return None
    open_time = C.parse_time(m.get("open_time"))
    return {
        "ticker": ticker,
        "event_ticker": m.get("event_ticker"),
        "series_ticker": series,
        "market_title": m.get("title") or m.get("yes_sub_title") or "",
        "open_time": open_time,
        "close_time": close,
        "window_start": close - timedelta(seconds=900),
    }


def _discover_candidate_markets_sync():
    """Discover fixed-universe contracts that can matter within ~20 minutes."""
    now = C.utc_now()
    horizon = now + timedelta(seconds=SUBSCRIBE_HORIZON_SECONDS)
    out = {}
    for series in CRYPTO_SERIES:
        try:
            markets = C.rest_get(
                "/markets",
                {"series_ticker": series, "status": "open", "limit": 1000},
            ).get("markets") or []
        except Exception:
            continue
        for m in markets:
            row = _market_row(series, m)
            if row is None:
                continue
            close = row["close_time"]
            if close <= now or close > horizon:
                continue
            out[row["ticker"]] = row
    return out


async def _discover_candidate_markets():
    return await asyncio.to_thread(_discover_candidate_markets_sync)


def _capture_phase(meta, t):
    if not meta:
        return False, None
    close = meta.get("close_time")
    if close is None:
        return False, None
    if not isinstance(t, datetime):
        t = datetime.fromtimestamp(float(t), tz=timezone.utc)
    elapsed = (t - (close - timedelta(seconds=900))).total_seconds()
    capture = -CAPTURE_PRESTART_SECONDS <= elapsed <= CAPTURE_END_ELAPSED_SECONDS
    return capture, elapsed


def _exchange_time(msg):
    if not isinstance(msg, dict):
        return None
    for value in (msg.get("ts_ms"), msg.get("timestamp"), msg.get("time")):
        if value is None:
            continue
        try:
            z = float(value)
            if z > 1e14:
                z /= 1_000_000.0
            elif z > 1e11:
                z /= 1_000.0
            return datetime.fromtimestamp(z, tz=timezone.utc)
        except Exception:
            try:
                return C.parse_time(value)
            except Exception:
                pass
    return None


def _price(x):
    try:
        z = float(x)
        return z if np.isfinite(z) else None
    except Exception:
        return None


def _qty(x):
    try:
        z = float(x)
        return z if np.isfinite(z) and z >= 0 else None
    except Exception:
        return None


def _valid_bbo(row):
    if not row:
        return False
    b, a = row.get("yes_bid"), row.get("yes_ask")
    return b is not None and a is not None and 0.0 <= b < a <= 1.0


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


async def _send(ws, lock, obj):
    async with lock:
        await ws.send(json.dumps(obj))


async def _subscribe(ws, lock, state, channel, tickers):
    if not tickers or channel in state["pending"]:
        return
    rid = state["next_id"]
    state["next_id"] += 1
    state["pending"].add(channel)
    state["pending_ids"][rid] = channel
    await _send(
        ws,
        lock,
        {
            "id": rid,
            "cmd": "subscribe",
            "params": {"channels": [channel], "market_tickers": sorted(tickers)},
        },
    )


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


async def _supervisor(ws, lock, state, files, stop_event):
    last_scan = 0.0
    latest = {}
    while not stop_event.is_set():
        now_mono = time.monotonic()
        if not latest or now_mono - last_scan >= MARKET_RESCAN_SECONDS:
            try:
                latest = await asyncio.wait_for(_discover_candidate_markets(), timeout=20.0)
                last_scan = time.monotonic()
                state["last_scan_error"] = None
                state["last_scan_mono"] = last_scan
            except Exception as exc:
                state["last_scan_error"] = repr(exc)
                files["events"].write({"time": _iso(), "type": "discovery_error", "error": repr(exc)})
                await asyncio.sleep(2.0)
                continue

        desired = set(latest)
        current = set(state["markets"])
        add = desired - current
        remove = current - desired
        state["meta"].update(latest)

        for channel in ("ticker", "trade"):
            if desired and channel not in state["sids"] and channel not in state["pending"]:
                await _subscribe(ws, lock, state, channel, desired)

        if add:
            for channel in ("ticker", "trade"):
                await _update(ws, lock, state, channel, "add_markets", add)
            for ticker in sorted(add):
                m = latest[ticker]
                files["market_metadata"].write(
                    {
                        "time": _iso(),
                        "connection_epoch": state["connection_epoch"],
                        "ticker": ticker,
                        "event_ticker": m.get("event_ticker"),
                        "series_ticker": m.get("series_ticker"),
                        "market_title": m.get("market_title"),
                        "open_time": _iso(m.get("open_time")) if m.get("open_time") else None,
                        "window_start": _iso(m.get("window_start")),
                        "close_time": _iso(m.get("close_time")),
                    }
                )

        if remove:
            for channel in ("ticker", "trade"):
                await _update(ws, lock, state, channel, "delete_markets", remove)
            for ticker in remove:
                state["latest_bbo"].pop(ticker, None)
                state["meta"].pop(ticker, None)

        if add or remove:
            files["rotations"].write(
                {
                    "time": _iso(),
                    "connection_epoch": state["connection_epoch"],
                    "subscribed_count": len(desired),
                    "added": sorted(add),
                    "removed": sorted(remove),
                }
            )
        state["markets"] = desired
        state["supervisor_heartbeat_mono"] = time.monotonic()
        await asyncio.sleep(1.0)


async def _consumer(ws, state, files, stop_event):
    async for raw in ws:
        if stop_event.is_set():
            break
        receipt = C.utc_now()
        state["last_ws_mono"] = time.monotonic()
        try:
            data = json.loads(raw)
        except Exception:
            continue

        typ = data.get("type")
        msg = data.get("msg") or {}
        sid = data.get("sid")

        if typ == "subscribed":
            channel = msg.get("channel")
            resolved_sid = msg.get("sid", sid)
            if channel and resolved_sid is not None:
                state["sids"][channel] = resolved_sid
                state["pending"].discard(channel)
                files["events"].write(
                    {
                        "time": _iso(receipt),
                        "type": "subscribed",
                        "channel": channel,
                        "sid": resolved_sid,
                        "connection_epoch": state["connection_epoch"],
                    }
                )
            continue

        if typ == "error":
            files["events"].write(
                {
                    "time": _iso(receipt),
                    "type": "ws_error",
                    "payload": data,
                    "connection_epoch": state["connection_epoch"],
                }
            )
            continue

        if typ not in {"ticker", "trade"}:
            continue
        state["last_market_data_mono"] = time.monotonic()

        ticker = msg.get("market_ticker")
        if not ticker or ticker not in state["markets"]:
            continue
        meta = state["meta"].get(ticker)

        if typ == "ticker":
            bid = _price(msg.get("yes_bid_dollars"))
            ask = _price(msg.get("yes_ask_dollars"))
            state["latest_bbo"][ticker] = {
                "receipt_time": receipt,
                "exchange_time": _exchange_time(msg),
                "yes_bid": bid,
                "yes_ask": ask,
                "yes_bid_size": _qty(msg.get("yes_bid_size_fp")),
                "yes_ask_size": _qty(msg.get("yes_ask_size_fp")),
            }
            continue

        capture, elapsed = _capture_phase(meta, receipt)
        if not capture:
            continue
        price = _price(msg.get("yes_price_dollars") or msg.get("price_dollars"))
        qty = _qty(msg.get("count_fp") or msg.get("count"))
        book_side = str(msg.get("taker_book_side") or "").lower()
        if price is None or qty is None or qty <= 0 or book_side not in {"bid", "ask"}:
            continue
        sec = int(receipt.timestamp())
        key = (ticker, book_side, round(float(price), 6))
        bucket = state["trade_buckets"][sec]
        rec = bucket.get(key)
        if rec is None:
            rec = {
                "ticker": ticker,
                "series_ticker": meta.get("series_ticker"),
                "taker_book_side": book_side,
                "yes_price": float(price),
                "qty": 0.0,
                "trade_count": 0,
            }
            bucket[key] = rec
        rec["qty"] += float(qty)
        rec["trade_count"] += 1


async def _one_hz_sampler(state, files, stop_event, session_dir):
    while not stop_event.is_set():
        now = time.time()
        next_sec = math.floor(now) + 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=max(0.0, next_sec - now))
            break
        except asyncio.TimeoutError:
            pass

        sample_dt = datetime.fromtimestamp(next_sec, tz=timezone.utc)
        markets = []
        valid_count = 0
        for ticker in sorted(state["markets"]):
            meta = state["meta"].get(ticker)
            capture, elapsed = _capture_phase(meta, sample_dt)
            if not capture:
                continue
            bbo = state["latest_bbo"].get(ticker)
            row = {
                "ticker": ticker,
                "series_ticker": meta.get("series_ticker") if meta else None,
                "elapsed_seconds": elapsed,
                "close_time": _iso(meta.get("close_time")) if meta and meta.get("close_time") else None,
                "valid": bool(_valid_bbo(bbo)),
                "yes_bid": bbo.get("yes_bid") if bbo else None,
                "yes_ask": bbo.get("yes_ask") if bbo else None,
                "yes_bid_size": bbo.get("yes_bid_size") if bbo else None,
                "yes_ask_size": bbo.get("yes_ask_size") if bbo else None,
                "source_time": _iso(bbo.get("receipt_time")) if bbo and bbo.get("receipt_time") else None,
                "source_age_ms": (
                    1000.0 * (sample_dt - bbo["receipt_time"]).total_seconds()
                    if bbo and bbo.get("receipt_time") else None
                ),
            }
            if row["valid"]:
                valid_count += 1
            markets.append(row)

        files["bbo_1hz"].write(
            {
                "time": _iso(sample_dt),
                "epoch_second": next_sec,
                "connection_epoch": state["connection_epoch"],
                "markets": markets,
            }
        )

        flush_secs = sorted(k for k in state["trade_buckets"] if k < next_sec)
        for sec in flush_secs:
            records = list(state["trade_buckets"].pop(sec).values())
            files["trades_1hz"].write(
                {
                    "bucket_time": _iso(datetime.fromtimestamp(sec, tz=timezone.utc)),
                    "epoch_second": sec,
                    "connection_epoch": state["connection_epoch"],
                    "trades": records,
                }
            )

        now_mono = time.monotonic()
        market_age = (
            None if state.get("last_market_data_mono") is None
            else now_mono - state["last_market_data_mono"]
        )
        health = {
            "time": _iso(sample_dt),
            "pid": os.getpid(),
            "running": True,
            "healthy": bool(
                {"ticker", "trade"}.issubset(set(state["sids"]))
                and state.get("last_scan_error") is None
                and (market_age is None or market_age < 30.0)
            ),
            "connection_epoch": state["connection_epoch"],
            "subscribed_markets": len(state["markets"]),
            "captured_markets_this_second": len(markets),
            "valid_bbo_markets_this_second": valid_count,
            "channels": sorted(state["sids"]),
            "market_data_age_s": market_age,
            "last_scan_error": state.get("last_scan_error"),
            "session_dir": str(session_dir),
        }
        _atomic_json(session_dir / "health.json", health)


async def _watchdog(state, files, stop_event):
    while not stop_event.is_set():
        await asyncio.sleep(5.0)
        if stop_event.is_set():
            break
        now_mono = time.monotonic()
        sup = state.get("supervisor_heartbeat_mono")
        if sup is not None and now_mono - sup > 30.0:
            raise RuntimeError(f"supervisor stale for {now_mono - sup:.1f}s")
        last = state.get("last_market_data_mono")
        if state["markets"] and last is not None and now_mono - last > 45.0:
            raise RuntimeError(f"market data stale for {now_mono - last:.1f}s")


async def _connection(key_id, key, files, stop_event, session_dir, epoch):
    ws = await C.open_ws(key_id, key)
    lock = asyncio.Lock()
    state = {
        "connection_epoch": epoch,
        "markets": set(),
        "meta": {},
        "latest_bbo": {},
        "trade_buckets": defaultdict(dict),
        "sids": {},
        "pending": set(),
        "pending_ids": {},
        "next_id": 1,
        "last_scan_error": None,
        "last_scan_mono": None,
        "last_ws_mono": None,
        "last_market_data_mono": None,
        "supervisor_heartbeat_mono": time.monotonic(),
    }
    files["events"].write({"time": _iso(), "type": "connected", "connection_epoch": epoch})

    tasks = [
        asyncio.create_task(_supervisor(ws, lock, state, files, stop_event)),
        asyncio.create_task(_consumer(ws, state, files, stop_event)),
        asyncio.create_task(_one_hz_sampler(state, files, stop_event, session_dir)),
        asyncio.create_task(_watchdog(state, files, stop_event)),
    ]
    try:
        while not stop_event.is_set():
            done, _ = await asyncio.wait(tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                continue
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    raise exc
                if not stop_event.is_set():
                    raise RuntimeError("recorder connection task ended unexpectedly")
            break
    finally:
        # Flush any remaining aggregated trades before closing this epoch.
        for sec in sorted(state["trade_buckets"]):
            files["trades_1hz"].write(
                {
                    "bucket_time": _iso(datetime.fromtimestamp(sec, tz=timezone.utc)),
                    "epoch_second": sec,
                    "connection_epoch": epoch,
                    "trades": list(state["trade_buckets"][sec].values()),
                }
            )
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await ws.close()
        except Exception:
            pass


def _open_files(session_dir):
    return {
        "bbo_1hz": Jsonl(session_dir / "bbo_1hz.jsonl"),
        "trades_1hz": Jsonl(session_dir / "aggressive_trades_1hz.jsonl"),
        "market_metadata": Jsonl(session_dir / "market_metadata.jsonl"),
        "rotations": Jsonl(session_dir / "market_rotations.jsonl"),
        "events": Jsonl(session_dir / "connection_events.jsonl"),
    }


def _close_files(files):
    for f in files.values():
        f.close()


async def run_oos_mm4c_recorder(session_dir):
    session_dir = Path(session_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=False)
    files = _open_files(session_dir)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    started = C.utc_now()
    manifest = {
        "study_version": STUDY_VERSION,
        "purpose": "fresh prospective OOS data for frozen 4c M1-M5 MM candidate",
        "started_at": _iso(started),
        "ended_at": None,
        "pid": os.getpid(),
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "storage_mode": "compact 1Hz BBO + 1Hz aggressive-trade aggregation",
        "capture_phase": "M0 through M6 only",
        "capture_prestart_seconds": CAPTURE_PRESTART_SECONDS,
        "capture_end_elapsed_seconds": CAPTURE_END_ELAPSED_SECONDS,
        "raw_orderbook_deltas_persisted": False,
        "raw_ticker_updates_persisted": False,
        "raw_trade_messages_persisted": False,
        "ticker_subscription": True,
        "trade_subscription": True,
        "orderbook_delta_subscription": False,
        "fixed_crypto_series": list(CRYPTO_SERIES),
        "warning": (
            "This compact dataset is for the frozen 1Hz strategy only. "
            "Do not use it for event-time reconstruction or same-session retuning."
        ),
    }
    _atomic_json(session_dir / "session_manifest.json", manifest)
    _atomic_json(session_dir / "frozen_strategy.json", FROZEN_STRATEGY)

    key_id, key = C.load_auth()
    epoch = 0
    try:
        while not stop_event.is_set():
            epoch += 1
            try:
                await _connection(key_id, key, files, stop_event, session_dir, epoch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                files["events"].write(
                    {
                        "time": _iso(),
                        "type": "connection_exception",
                        "connection_epoch": epoch,
                        "error": repr(exc),
                    }
                )
                if not stop_event.is_set():
                    await asyncio.sleep(3.0)
    finally:
        ended = C.utc_now()
        manifest["ended_at"] = _iso(ended)
        manifest["actual_duration_hours"] = (ended - started).total_seconds() / 3600.0
        _atomic_json(session_dir / "session_manifest.json", manifest)
        _atomic_json(
            session_dir / "health.json",
            {
                "time": _iso(ended),
                "pid": os.getpid(),
                "running": False,
                "healthy": False,
                "session_dir": str(session_dir),
            },
        )
        _close_files(files)
        control = _read_json(CONTROL_PATH, {}) or {}
        if int(control.get("pid") or -1) == os.getpid():
            try:
                CONTROL_PATH.unlink()
            except FileNotFoundError:
                pass


def start_oos_mm4c_recording(*, startup_timeout_s=STARTUP_TIMEOUT_SECONDS):
    """Launch the compact recorder as a detached child process and return session Path."""
    active = _read_json(CONTROL_PATH, {}) or {}
    if active and _pid_alive(active.get("pid")):
        raise RuntimeError(
            "Compact OOS recorder is already running: "
            f"pid={active.get('pid')} session={active.get('session_dir')}"
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
        "quant_research.kalshi.mm_oos_4c_compact_recorder",
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

    control = {
        "pid": proc.pid,
        "session_dir": str(session_dir),
        "started_at": _iso(),
        "launcher_pid": os.getpid(),
        "log_path": str(log_path),
        "study_version": STUDY_VERSION,
    }
    _atomic_json(CONTROL_PATH, control)

    deadline = time.time() + float(startup_timeout_s)
    last_health = None
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
            except Exception:
                pass
            raise RuntimeError(
                f"Recorder exited during startup with code {proc.returncode}.\n{tail}"
            )
        health_path = session_dir / "health.json"
        if health_path.exists():
            last_health = _read_json(health_path, {}) or {}
            if last_health.get("healthy"):
                print("OOS MM4C compact recorder is healthy.")
                print(f"PID: {proc.pid}")
                print(f"SESSION: {session_dir}")
                print("MODE: 1Hz compact BBO + 1Hz aggregated trades, M0-M6 only")
                print(f"LOG: {log_path}")
                return session_dir
        time.sleep(0.5)

    # Keep the process running if connected but currently waiting for market traffic;
    # however surface the state clearly instead of silently claiming healthy.
    if proc.poll() is None and session_dir.exists():
        print("Recorder process is running but did not reach healthy=True before timeout.")
        print(f"SESSION: {session_dir}")
        print(f"LAST HEALTH: {last_health}")
        print(f"LOG: {log_path}")
        return session_dir
    raise RuntimeError("Recorder failed to start")


def oos_mm4c_recording_status():
    control = _read_json(CONTROL_PATH, {}) or {}
    if not control:
        out = {"running": False, "message": "No active compact OOS recorder control file."}
        print(out)
        return out
    pid = control.get("pid")
    session = Path(control.get("session_dir"))
    health = _read_json(session / "health.json", {}) or {}
    out = {
        **control,
        **health,
        "pid_alive": _pid_alive(pid),
        "running": bool(_pid_alive(pid)),
    }
    print(json.dumps(out, indent=2, default=str))
    return out


def _session_size_mb(session_dir):
    total = 0
    rows = []
    for p in sorted(Path(session_dir).glob("*")):
        if p.is_file():
            n = p.stat().st_size
            total += n
            rows.append((p.name, n / (1024 * 1024)))
    return total / (1024 * 1024), rows


def stop_oos_mm4c_recording(*, expected_session=None, timeout_s=STOP_TIMEOUT_SECONDS):
    """Gracefully stop the dedicated compact recorder and return the saved session Path."""
    control = _read_json(CONTROL_PATH, {}) or {}
    if not control:
        if expected_session is not None:
            session = Path(expected_session).resolve()
            if session.exists():
                print(f"No active recorder. Session already exists: {session}")
                return session
        print("No active compact OOS recorder found.")
        return None

    pid = int(control.get("pid"))
    session = Path(control.get("session_dir")).resolve()
    if expected_session is not None and Path(expected_session).resolve() != session:
        raise RuntimeError(
            f"Session mismatch: expected {Path(expected_session).resolve()} but active is {session}. "
            "Refusing ambiguous stop."
        )

    if _pid_alive(pid):
        print(f"Stopping compact OOS recorder pid={pid} ...")
        os.kill(pid, signal.SIGINT)
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.25)
        if _pid_alive(pid):
            print("Graceful stop timed out; sending SIGTERM.")
            os.kill(pid, signal.SIGTERM)
            deadline = time.time() + 10.0
            while time.time() < deadline and _pid_alive(pid):
                time.sleep(0.25)

    if CONTROL_PATH.exists():
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass

    print(f"SAVED SESSION: {session}")
    if session.exists():
        total_mb, rows = _session_size_mb(session)
        print(f"TOTAL SIZE: {total_mb:.1f} MB")
        for name, mb in rows:
            print(f"  {name:<32} {mb:8.2f} MB")
        manifest = _read_json(session / "session_manifest.json", {}) or {}
        if manifest.get("actual_duration_hours") is not None:
            print(f"DURATION: {float(manifest['actual_duration_hours']):.2f} hours")
    return session


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--run-session", default=None)
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    args = p.parse_args()

    if args.run_session:
        asyncio.run(run_oos_mm4c_recorder(Path(args.run_session)))
        return
    if args.status:
        oos_mm4c_recording_status()
        return
    if args.stop:
        stop_oos_mm4c_recording()
        return
    p.error("Use --run-session PATH, --status, or --stop")


if __name__ == "__main__":
    _main()
