from __future__ import annotations
from pathlib import Path
from . import shadow_trader as S


def _resolve_session(session_dir=None):
    if session_dir is not None:return Path(session_dir)
    from .recorder import current_session_dir
    session=current_session_dir()
    if session is None:raise RuntimeError("No recorder session is running. Start the recorder first.")
    return Path(session)


def start_primary_shadow_trader(session_dir=None):
    session=_resolve_session(session_dir)
    if S._SHADOW is not None and any(t.is_alive() for t in S._SHADOW.threads):
        if Path(S._SHADOW.session_dir).resolve()==session.resolve():
            print("Primary shadow trader is already running for this recorder session.")
            return S._SHADOW
        S._SHADOW.stop();S._SHADOW=None
    S._SHADOW=S.ShadowTrader(session).start()
    print("PRIMARY SHADOW: M5 | BTC opposition | spread<=2c | -3c | 15s | 3ct | 10c candle-confirmed passive salvage | READ-ONLY")
    return S._SHADOW


def stop_primary_shadow_trader():
    if S._SHADOW is None:print("Primary shadow trader is not running.");return None
    out=S._SHADOW.stop();S._SHADOW=None;return out


def primary_shadow_status():
    if S._SHADOW is None:print("Primary shadow trader is not running.");return None
    return S._SHADOW.status()

start_shadow_trader=start_primary_shadow_trader
stop_shadow_trader=stop_primary_shadow_trader
shadow_status=primary_shadow_status
