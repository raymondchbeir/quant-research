from __future__ import annotations
import asyncio
from pathlib import Path
from . import runtime_fix as _runtime_fix
from . import recorder as R
from . import primary_shadow_trader as S

async def start_live(duration_minutes:float|None=None):
    await R.start_recorder(duration_minutes=duration_minutes);session=R.current_session_dir()
    for _ in range(100):
        state=R._STATE or {};channels=set(state.get("sids",{}))
        if session is not None and len(state.get("markets",()))>0 and {"orderbook_delta","trade","ticker"}.issubset(channels):break
        await asyncio.sleep(.2);session=R.current_session_dir()
    else:
        await R.stop_recorder();raise RuntimeError("Recorder connected but detected/subscribed to 0 active 15-minute markets after 20 seconds.")
    S.start_shadow_trader(session);print(f"LIVE research session started: {session}");return session

def live_status():
    print("RECORDER");R.recorder_status();print("\nPRIMARY SHADOW TRADER");return S.shadow_status()

async def stop_live()->Path|None:
    S.stop_shadow_trader();session=await R.stop_recorder();print(f"All live research stopped. Saved to: {session}");return session
