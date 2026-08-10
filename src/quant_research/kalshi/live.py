from __future__ import annotations

import asyncio
from pathlib import Path

from .recorder import current_session_dir, recorder_status, start_recorder, stop_recorder
from .shadow_trader import shadow_status, start_shadow_trader, stop_shadow_trader


async def start_live(duration_minutes: float | None = None):
    await start_recorder(duration_minutes=duration_minutes)
    session = current_session_dir()
    for _ in range(100):
        if session is not None and all((session / name).exists() for name in ("ticker_updates.jsonl", "trades.jsonl", "full_books.jsonl")):
            break
        await asyncio.sleep(0.05)
        session = current_session_dir()
    if session is None:
        raise RuntimeError("Recorder session did not initialize.")
    start_shadow_trader(session)
    print(f"LIVE research session started: {session}")
    return session


def live_status():
    print("RECORDER")
    recorder_status()
    print("\nSHADOW TRADER")
    return shadow_status()


async def stop_live() -> Path | None:
    stop_shadow_trader()
    session = await stop_recorder()
    print(f"All live research stopped. Saved to: {session}")
    return session
