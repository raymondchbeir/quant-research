from __future__ import annotations

"""Pre-open compact recorder for the frozen NAT4->2 + pre-M1 range study.

This is a DATA-CAPTURE compatibility fix only. It preserves the V2 compact
1 Hz BBO + event-time trade storage, but fixes M0-M1 startup coverage by:

1. discovering both `unopened` and `open` 15-minute contracts;
2. rescanning every 5 seconds;
3. seeding/refeshing `latest_bbo` from the REST market object whenever the
   websocket ticker state is not yet valid.

No trading-strategy threshold, sizing, inventory, momentum, flow, or cooldown
parameter is changed here.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import recorder_core as C
from . import mm_oos_4c_compact_recorder as BASE
from . import mm_oos_4c_compact_recorder_v2 as V2

STUDY_VERSION = "OOS_MM4C_COMPACT_1HZ_PREOPEN_V3"
PREOPEN_RESCAN_SECONDS = 5.0
STARTUP_TIMEOUT_SECONDS = V2.STARTUP_TIMEOUT_SECONDS
STOP_TIMEOUT_SECONDS = V2.STOP_TIMEOUT_SECONDS

CRYPTO_SERIES = BASE.CRYPTO_SERIES
PROJECT_ROOT = V2.PROJECT_ROOT
OOS_ROOT = V2.OOS_ROOT
CONTROL_PATH = V2.CONTROL_PATH


def _iso(dt=None):
    return C.iso_utc(dt or C.utc_now())


def _f(x):
    try:
        z = float(x)
        return z
    except Exception:
        return None


def _market_row_with_seed(series, m):
    row = BASE._market_row(series, m)
    if row is None:
        return None
    row["rest_status"] = m.get("status")
    row["seed_yes_bid"] = _f(m.get("yes_bid_dollars"))
    row["seed_yes_ask"] = _f(m.get("yes_ask_dollars"))
    row["seed_yes_bid_size"] = _f(m.get("yes_bid_size_fp"))
    row["seed_yes_ask_size"] = _f(m.get("yes_ask_size_fp"))
    return row


def _discover_preopen_markets_sync():
    """Return relevant unopened + open contracts within the existing horizon."""
    now = C.utc_now()
    horizon = now.timestamp() + BASE.SUBSCRIBE_HORIZON_SECONDS
    out = {}

    for series in CRYPTO_SERIES:
        for status in ("unopened", "open"):
            try:
                markets = C.rest_get(
                    "/markets",
                    {"series_ticker": series, "status": status, "limit": 1000},
                ).get("markets") or []
            except Exception:
                continue

            for m in markets:
                row = _market_row_with_seed(series, m)
                if row is None:
                    continue
                close = row["close_time"]
                close_ts = close.timestamp()
                if close_ts <= now.timestamp() or close_ts > horizon:
                    continue
                out[row["ticker"]] = row

    return out


async def _discover_preopen_markets():
    return await asyncio.to_thread(_discover_preopen_markets_sync)


def _seed_bbo_from_rest(state, ticker, meta, receipt):
    bid = meta.get("seed_yes_bid")
    ask = meta.get("seed_yes_ask")
    bq = meta.get("seed_yes_bid_size")
    aq = meta.get("seed_yes_ask_size")

    if bid is None or ask is None or not (0.0 <= bid < ask <= 1.0):
        return False

    state["latest_bbo"][ticker] = {
        "receipt_time": receipt,
        "exchange_time": None,
        "yes_bid": float(bid),
        "yes_ask": float(ask),
        "yes_bid_size": 0.0 if bq is None else max(0.0, float(bq)),
        "yes_ask_size": 0.0 if aq is None else max(0.0, float(aq)),
    }
    return True


async def _supervisor_preopen(ws, lock, state, files, stop_event):
    last_scan = 0.0
    latest = {}

    while not stop_event.is_set():
        now_mono = time.monotonic()
        if not latest or now_mono - last_scan >= PREOPEN_RESCAN_SECONDS:
            try:
                latest = await asyncio.wait_for(
                    _discover_preopen_markets(), timeout=20.0
                )
                last_scan = time.monotonic()
                state["last_scan_error"] = None
                state["last_scan_mono"] = last_scan
            except Exception as exc:
                state["last_scan_error"] = repr(exc)
                files["events"].write(
                    {"time": _iso(), "type": "discovery_error", "error": repr(exc)}
                )
                await asyncio.sleep(1.0)
                continue

        desired = set(latest)
        current = set(state["markets"])
        add = desired - current
        remove = current - desired
        state["meta"].update(latest)

        for channel in ("ticker", "trade"):
            if desired and channel not in state["sids"] and channel not in state["pending"]:
                await BASE._subscribe(ws, lock, state, channel, desired)

        if add:
            for channel in ("ticker", "trade"):
                await BASE._update(ws, lock, state, channel, "add_markets", add)

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
                        "discovered_status": m.get("rest_status"),
                    }
                )

        # REST seed is only a startup fallback. Once websocket ticker state is
        # valid, websocket updates own the state. If no valid state exists yet,
        # use the latest REST market snapshot so M0 does not start blank.
        receipt = C.utc_now()
        for ticker in sorted(desired):
            if not BASE._valid_bbo(state["latest_bbo"].get(ticker)):
                _seed_bbo_from_rest(state, ticker, latest[ticker], receipt)

        if remove:
            for channel in ("ticker", "trade"):
                await BASE._update(ws, lock, state, channel, "delete_markets", remove)
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
                    "preopen_discovery": True,
                }
            )

        state["markets"] = desired
        state["supervisor_heartbeat_mono"] = time.monotonic()
        await asyncio.sleep(1.0)


async def run_preopen_recorder(session_dir):
    # V2's connection uses BASE._supervisor. Patch only that data-acquisition
    # component for this child process.
    old_supervisor = BASE._supervisor
    old_version = V2.STUDY_VERSION
    old_frozen = V2.FROZEN_STRATEGY

    frozen = dict(old_frozen)
    frozen["recorder_version"] = STUDY_VERSION
    frozen["preopen_discovery"] = True
    frozen["preopen_statuses"] = ["unopened", "open"]
    frozen["preopen_rescan_seconds"] = PREOPEN_RESCAN_SECONDS
    frozen["rest_bbo_seed_when_ws_uninitialized"] = True

    BASE._supervisor = _supervisor_preopen
    V2.STUDY_VERSION = STUDY_VERSION
    V2.FROZEN_STRATEGY = frozen
    try:
        await V2.run_oos_mm4c_recorder(Path(session_dir))
    finally:
        BASE._supervisor = old_supervisor
        V2.STUDY_VERSION = old_version
        V2.FROZEN_STRATEGY = old_frozen


def start_preopen_oos_recording(*, startup_timeout_s=STARTUP_TIMEOUT_SECONDS):
    active = BASE._read_json(CONTROL_PATH, {}) or {}
    if active and BASE._pid_alive(active.get("pid")):
        raise RuntimeError(
            "Compact OOS recorder is already running: "
            f"pid={active.get('pid')} session={active.get('session_dir')}"
        )
    if CONTROL_PATH.exists():
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass

    session_dir = BASE._session_dir_now().resolve()
    if session_dir.exists():
        raise FileExistsError(session_dir)

    log_path = session_dir.parent / f"{session_dir.name}.startup.log"
    cmd = [
        sys.executable,
        "-m",
        "quant_research.kalshi.mm_oos_4c_compact_recorder_preopen_v3",
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

    BASE._atomic_json(
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
            last_health = BASE._read_json(health_path, {}) or {}
            if last_health.get("healthy"):
                print("Pre-open OOS recorder V3 is healthy.")
                print(f"PID: {proc.pid}")
                print(f"SESSION: {session_dir}")
                print("BBO: 1 Hz | TRADES: compact event-time | CAPTURE: M0-M6")
                print("DISCOVERY: unopened + open | REST BBO seed fallback | 5s rescan")
                print("RAW DELTAS: OFF")
                print(f"LOG: {log_path}")
                return session_dir
        time.sleep(0.5)

    if proc.poll() is None and session_dir.exists():
        print("Recorder is running but did not reach healthy=True before timeout.")
        print(f"SESSION: {session_dir}")
        print(f"LAST HEALTH: {last_health}")
        print(f"LOG: {log_path}")
        return session_dir

    raise RuntimeError("Pre-open recorder failed to start")


def preopen_oos_recording_status():
    return V2.oos_mm4c_recording_status()


def stop_preopen_oos_recording(*, expected_session=None, timeout_s=STOP_TIMEOUT_SECONDS):
    return V2.stop_oos_mm4c_recording(
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
        asyncio.run(run_preopen_recorder(Path(args.run_session)))
        return
    if args.status:
        preopen_oos_recording_status()
        return
    if args.stop:
        stop_preopen_oos_recording()
        return
    p.error("Use --run-session PATH, --status, or --stop")


if __name__ == "__main__":
    _main()
