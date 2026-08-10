from __future__ import annotations

from pathlib import Path

from . import recorder as R
from . import primary_shadow_trader as P
from . import shadow_dashboard as D
from . import shadow_trader as S

_ACTIVE_RECORDER_SESSION = None
_LAST_SHADOW_SESSION = None


def _resolved(path):
    return None if path is None else Path(path).resolve()


def _assert_pair_consistent():
    recorder_session = _resolved(R.current_session_dir())
    shadow_session = _resolved(S._SHADOW.session_dir) if S._SHADOW is not None else None
    if recorder_session is not None and shadow_session is not None and recorder_session != shadow_session:
        raise RuntimeError(
            "Recorder/shadow session mismatch detected.\n"
            f"Recorder: {recorder_session}\n"
            f"Shadow:   {shadow_session}\n"
            "Refusing to continue with ambiguous state. Restart the Jupyter kernel before starting a fresh run."
        )
    return recorder_session, shadow_session


async def start_recorder(duration_minutes=None, key_id=None, private_key_path=None):
    global _ACTIVE_RECORDER_SESSION
    task = await R.start_recorder(
        duration_minutes=duration_minutes,
        key_id=key_id,
        private_key_path=private_key_path,
    )
    _ACTIVE_RECORDER_SESSION = _resolved(R.current_session_dir())
    return task


async def stop_recorder():
    current = _resolved(R.current_session_dir())
    expected = _ACTIVE_RECORDER_SESSION
    if expected is not None and current is not None and expected != current:
        raise RuntimeError(
            "Recorder session changed inside this kernel.\n"
            f"Started through runtime API: {expected}\n"
            f"Recorder module now reports: {current}\n"
            "Refusing ambiguous stop. Restart the kernel."
        )
    return await R.stop_recorder(expected_session=expected)


def current_session_dir():
    return R.current_session_dir()


def recorder_status():
    out = R.recorder_status()
    _, shadow_session = _assert_pair_consistent()
    out["shadow_session_dir"] = str(shadow_session) if shadow_session else None
    out["session_pair_consistent"] = True
    return out


async def preview_15m_markets():
    return await R.preview_15m_markets()


def start_primary_shadow_trader(session_dir=None):
    global _LAST_SHADOW_SESSION
    target = _resolved(session_dir) if session_dir is not None else _resolved(R.current_session_dir())
    if target is None:
        raise RuntimeError("No recorder session is running. Start the recorder first.")

    recorder_session = _resolved(R.current_session_dir())
    if recorder_session is not None and target != recorder_session:
        raise RuntimeError(
            f"Requested shadow session {target} does not match active recorder {recorder_session}."
        )

    out = P.start_primary_shadow_trader(target)
    _LAST_SHADOW_SESSION = target
    _assert_pair_consistent()
    return out


def stop_primary_shadow_trader():
    global _LAST_SHADOW_SESSION
    recorder_session, shadow_session = _assert_pair_consistent()
    if shadow_session is None:
        print("Primary shadow trader is not running.")
        return None
    if recorder_session is not None and recorder_session != shadow_session:
        raise RuntimeError("Refusing to stop mismatched shadow/recorder sessions.")
    _LAST_SHADOW_SESSION = shadow_session
    return P.stop_primary_shadow_trader()


def primary_shadow_status(show_rows=10):
    _assert_pair_consistent()
    return D.primary_shadow_status(show_rows=show_rows)


async def watch_primary_shadow_status(refresh_seconds=2.0, show_rows=8):
    _assert_pair_consistent()
    from .shadow_dashboard_live_v3 import watch_primary_shadow_status as _watch
    return await _watch(refresh_seconds=refresh_seconds, show_rows=show_rows)


start_shadow_trader = start_primary_shadow_trader
stop_shadow_trader = stop_primary_shadow_trader
shadow_status = primary_shadow_status
watch_shadow_status = watch_primary_shadow_status
