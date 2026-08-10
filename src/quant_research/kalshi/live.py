from __future__ import annotations

from pathlib import Path

from . import primary_shadow_trader as P
from . import recorder as R
from . import shadow_dashboard as D


async def start_live(duration_minutes: float | None = None):
    await R.start_recorder(duration_minutes=duration_minutes)
    session = R.current_session_dir()
    if session is None:
        raise RuntimeError("Recorder did not start a live session.")
    P.start_primary_shadow_trader(session)
    print(f"LIVE research session started: {session}")
    return session


def live_status():
    print("RECORDER")
    R.recorder_status()
    print("\nPRIMARY SHADOW TRADER")
    return D.primary_shadow_status()


async def stop_live() -> Path | None:
    if P.primary_shadow_running():
        P.stop_primary_shadow_trader()
    session = R.current_session_dir() or R.last_session_dir()
    await R.stop_recorder(expected_session=session)
    print(f"All live research stopped. Saved to: {session}")
    return session
