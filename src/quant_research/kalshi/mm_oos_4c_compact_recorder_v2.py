from __future__ import annotations

"""Final storage-conscious prospective OOS recorder for the frozen 4c MM candidate.

This V2 keeps the market-state stream at 1 Hz while preserving the minimal
per-trade event timing needed for the frozen strategy's FIFO queue depletion,
trade-through fills, cooldowns, and 5-second aggressive-flow guard.

Storage is kept small by:
- fixed crypto 15-minute universe only;
- no orderbook_delta subscription or persistence;
- no raw ticker persistence;
- 1 Hz compact BBO snapshots only;
- compact trade rows with only essential fields (no raw_msg / repeated payload);
- persistence only from M0 through M6 for each contract. M1-M5 is the quote
  window and M5-M6 is retained for the longest 60-second markout.

This dataset is intentionally for the frozen 1 Hz OOS MM replay only. It is not
an event-time order-book reconstruction archive and must not be reused for
same-session threshold optimization.
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
from datetime import datetime, timezone
from pathlib import Path

from . import recorder_core as C
from . import mm_oos_4c_compact_recorder as R

STUDY_VERSION = "OOS_MM4C_COMPACT_1HZ_V2"
SAMPLE_INTERVAL_SECONDS = 1.0
STARTUP_TIMEOUT_SECONDS = 35.0
STOP_TIMEOUT_SECONDS = 45.0

CRYPTO_SERIES = R.CRYPTO_SERIES
FROZEN_STRATEGY = dict(R.FROZEN_STRATEGY)
FROZEN_STRATEGY["recorder_version"] = STUDY_VERSION
FROZEN_STRATEGY["market_state_frequency"] = "1 Hz"
FROZEN_STRATEGY["trade_storage"] = "compact event-time rows during M0-M6"

PROJECT_ROOT = R.PROJECT_ROOT
OOS_ROOT = R.OOS_ROOT
CONTROL_PATH = R.CONTROL_PATH


def _iso(dt=None):
    return C.iso_utc(dt or C.utc_now())


def _compact_trade_row(msg, receipt, meta, epoch):
    capture, elapsed = R._capture_phase(meta, receipt)
    if not capture:
        return None
    price = R._price(msg.get("yes_price_dollars") or msg.get("price_dollars"))
    qty = R._qty(msg.get("count_fp") or msg.get("count"))
    book_side = str(msg.get("taker_book_side") or "").lower()
    if price is None or qty is None or qty <= 0 or book_side not in {"bid", "ask"}:
        return None
    return {
        "time": _iso(receipt),
        "connection_epoch": epoch,
        "ticker": msg.get("market_ticker"),
        "series_ticker": meta.get("series_ticker") if meta else None,
        "elapsed_seconds": elapsed,
        "taker_book_side": book_side,
        "yes_price": float(price),
        "qty": float(qty),
        "trade_id": msg.get("trade_id"),
        "exchange_time": _iso(R._exchange_time(msg)) if R._exchange_time(msg) else None,
    }


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
            state["latest_bbo"][ticker] = {
                "receipt_time": receipt,
                "exchange_time": R._exchange_time(msg),
                "yes_bid": R._price(msg.get("yes_bid_dollars")),
                "yes_ask": R._price(msg.get("yes_ask_dollars")),
                "yes_bid_size": R._qty(msg.get("yes_bid_size_fp")),
                "yes_ask_size": R._qty(msg.get("yes_ask_size_fp")),
            }
            continue

        row = _compact_trade_row(msg, receipt, meta, state["connection_epoch"])
        if row is not None:
            files["trades_compact"].write(row)
            state["compact_trade_rows"] += 1


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
            capture, elapsed = R._capture_phase(meta, sample_dt)
            if not capture:
                continue
            bbo = state["latest_bbo"].get(ticker)
            valid = bool(R._valid_bbo(bbo))
            if valid:
                valid_count += 1
            source_time = bbo.get("receipt_time") if bbo else None
            markets.append(
                {
                    "ticker": ticker,
                    "series_ticker": meta.get("series_ticker") if meta else None,
                    "elapsed_seconds": elapsed,
                    "close_time": _iso(meta.get("close_time")) if meta and meta.get("close_time") else None,
                    "valid": valid,
                    "yes_bid": bbo.get("yes_bid") if bbo else None,
                    "yes_ask": bbo.get("yes_ask") if bbo else None,
                    "yes_bid_size": bbo.get("yes_bid_size") if bbo else None,
                    "yes_ask_size": bbo.get("yes_ask_size") if bbo else None,
                    "source_time": _iso(source_time) if source_time else None,
                    "source_age_ms": (
                        1000.0 * (sample_dt - source_time).total_seconds()
                        if source_time else None
                    ),
                }
            )

        files["bbo_1hz"].write(
            {
                "time": _iso(sample_dt),
                "epoch_second": next_sec,
                "connection_epoch": state["connection_epoch"],
                "markets": markets,
            }
        )
        state["bbo_sample_rows"] += 1

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
            "bbo_sample_rows": state["bbo_sample_rows"],
            "compact_trade_rows": state["compact_trade_rows"],
            "channels": sorted(state["sids"]),
            "market_data_age_s": market_age,
            "last_scan_error": state.get("last_scan_error"),
            "session_dir": str(session_dir),
        }
        R._atomic_json(session_dir / "health.json", health)


async def _connection(key_id, key, files, stop_event, session_dir, epoch):
    ws = await C.open_ws(key_id, key)
    lock = asyncio.Lock()
    state = {
        "connection_epoch": epoch,
        "markets": set(),
        "meta": {},
        "latest_bbo": {},
        "sids": {},
        "pending": set(),
        "pending_ids": {},
        "next_id": 1,
        "last_scan_error": None,
        "last_scan_mono": None,
        "last_ws_mono": None,
        "last_market_data_mono": None,
        "supervisor_heartbeat_mono": time.monotonic(),
        "bbo_sample_rows": 0,
        "compact_trade_rows": 0,
    }
    files["events"].write({"time": _iso(), "type": "connected", "connection_epoch": epoch})

    tasks = [
        asyncio.create_task(R._supervisor(ws, lock, state, files, stop_event)),
        asyncio.create_task(_consumer(ws, state, files, stop_event)),
        asyncio.create_task(_one_hz_sampler(state, files, stop_event, session_dir)),
        asyncio.create_task(R._watchdog(state, files, stop_event)),
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
        "bbo_1hz": R.Jsonl(session_dir / "bbo_1hz.jsonl"),
        "trades_compact": R.Jsonl(session_dir / "aggressive_trades_compact.jsonl"),
        "market_metadata": R.Jsonl(session_dir / "market_metadata.jsonl"),
        "rotations": R.Jsonl(session_dir / "market_rotations.jsonl"),
        "events": R.Jsonl(session_dir / "connection_events.jsonl"),
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
        "market_state_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "storage_mode": "1Hz compact BBO + compact event-time aggressive trades",
        "capture_phase": "M0 through M6 only",
        "capture_prestart_seconds": R.CAPTURE_PRESTART_SECONDS,
        "capture_end_elapsed_seconds": R.CAPTURE_END_ELAPSED_SECONDS,
        "raw_orderbook_deltas_persisted": False,
        "orderbook_delta_subscription": False,
        "raw_ticker_updates_persisted": False,
        "raw_trade_payloads_persisted": False,
        "compact_event_time_trades_persisted": True,
        "ticker_subscription": True,
        "trade_subscription": True,
        "fixed_crypto_series": list(CRYPTO_SERIES),
        "warning": (
            "Use only for the pre-frozen 1Hz 4c OOS strategy. "
            "Not an event-time book reconstruction dataset and not for same-session retuning."
        ),
    }
    R._atomic_json(session_dir / "session_manifest.json", manifest)
    R._atomic_json(session_dir / "frozen_strategy.json", FROZEN_STRATEGY)

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
        R._atomic_json(session_dir / "session_manifest.json", manifest)
        R._atomic_json(
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
        control = R._read_json(CONTROL_PATH, {}) or {}
        if int(control.get("pid") or -1) == os.getpid():
            try:
                CONTROL_PATH.unlink()
            except FileNotFoundError:
                pass


def start_oos_mm4c_recording(*, startup_timeout_s=STARTUP_TIMEOUT_SECONDS):
    """Launch V2 as a detached child process and return the new session Path."""
    active = R._read_json(CONTROL_PATH, {}) or {}
    if active and R._pid_alive(active.get("pid")):
        raise RuntimeError(
            "Compact OOS recorder is already running: "
            f"pid={active.get('pid')} session={active.get('session_dir')}"
        )
    if CONTROL_PATH.exists():
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass

    session_dir = R._session_dir_now().resolve()
    if session_dir.exists():
        raise FileExistsError(session_dir)
    log_path = session_dir.parent / f"{session_dir.name}.startup.log"
    cmd = [
        sys.executable,
        "-m",
        "quant_research.kalshi.mm_oos_4c_compact_recorder_v2",
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

    R._atomic_json(
        CONTROL_PATH,
        {
            "pid": proc.pid,
            "session_dir": str(session_dir),
            "started_at": _iso(),
            "launcher_pid": os.getpid(),
            "log_path": str(log_path),
            "study_version": STUDY_VERSION,
        },
    )

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
            last_health = R._read_json(health_path, {}) or {}
            if last_health.get("healthy"):
                print("OOS MM4C V2 recorder is healthy.")
                print(f"PID: {proc.pid}")
                print(f"SESSION: {session_dir}")
                print("BBO: 1 Hz | TRADES: compact event-time | CAPTURE: M0-M6")
                print("RAW DELTAS: OFF")
                print(f"LOG: {log_path}")
                return session_dir
        time.sleep(0.5)

    if proc.poll() is None and session_dir.exists():
        print("Recorder process is running but did not reach healthy=True before timeout.")
        print(f"SESSION: {session_dir}")
        print(f"LAST HEALTH: {last_health}")
        print(f"LOG: {log_path}")
        return session_dir
    raise RuntimeError("Recorder failed to start")


# Reuse the dedicated control-file status/stop logic from V1. It is agnostic to
# the child module and therefore remains safe across notebook/kernel restarts.
def oos_mm4c_recording_status():
    return R.oos_mm4c_recording_status()


def stop_oos_mm4c_recording(*, expected_session=None, timeout_s=STOP_TIMEOUT_SECONDS):
    return R.stop_oos_mm4c_recording(
        expected_session=expected_session,
        timeout_s=timeout_s,
    )


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
