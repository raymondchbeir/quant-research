from __future__ import annotations

from pathlib import Path

# Thin compatibility layer around dashboard v4. It preserves the existing
# recorder/Q3/Q3-Q1 lifecycle and swaps only the Q15/Q5 implementation for the
# optimized, nonblocking catch-up monitor.

from . import live_research_stack_v4 as _v4
from .live_research_stack import stop_live_research_stack as _stop_base_live_stack
from .pre_m5_range44_scaled_strategy_fast import (
    range44_q15q5_status,
    start_range44_q15q5_monitor,
    stop_range44_q15q5_monitor,
)

DASHBOARD_VERSION = "KALSHI_LIVE_RESEARCH_DASHBOARD_V5_FAST_Q15_Q5"


def _wire_fast_monitor():
    _v4.range44_q15q5_status = range44_q15q5_status
    _v4.start_range44_q15q5_monitor = start_range44_q15q5_monitor
    _v4.stop_range44_q15q5_monitor = stop_range44_q15q5_monitor
    if hasattr(_v4, "DASHBOARD_VERSION"):
        _v4.DASHBOARD_VERSION = DASHBOARD_VERSION


def render_live_research_stack_html(show_rows=10):
    _wire_fast_monitor()
    return _v4.render_live_research_stack_html(show_rows=show_rows)


async def watch_live_research_stack(refresh_seconds=2.0, show_rows=10):
    _wire_fast_monitor()
    return await _v4.watch_live_research_stack(
        refresh_seconds=refresh_seconds,
        show_rows=show_rows,
    )


async def start_live_research_stack(
    refresh_seconds=2.0,
    show_rows=10,
    duration_minutes=None,
    key_id=None,
    private_key_path=None,
):
    """Adopt the existing live stack, attach optimized Q15/Q5, then watch."""
    _wire_fast_monitor()

    check = await _v4.prepare_live_research_stack(
        duration_minutes=duration_minutes,
        key_id=key_id,
        private_key_path=private_key_path,
    )
    session_dir = Path(check["recorder_session"])

    st = range44_q15q5_status(show=False)
    if st.get("running") and not _v4._same_session(st.get("session_dir"), session_dir):
        stop_range44_q15q5_monitor(show=False)
        st = {"running": False}

    if not st.get("running"):
        start_range44_q15q5_monitor(
            session_dir=session_dir,
            interval_sec=10.0,
            show=True,
        )

    st = range44_q15q5_status(show=False)
    if not st.get("running") or not _v4._same_session(st.get("session_dir"), session_dir):
        raise RuntimeError("Optimized Q15/Q5 monitor failed session-alignment gate.")
    if st.get("last_error"):
        raise RuntimeError(f"Optimized Q15/Q5 monitor error: {st.get('last_error')}")

    print()
    print("LIVE STACK V5 READY")
    print("Session:", session_dir)
    print("Recorder: RUNNING / UNCHANGED")
    print("Frozen Q3 primary: RUNNING / UNCHANGED")
    print("Range44 Q3/Q1: RUNNING / BENCHMARK")
    print("Range44 Q15/Q5: RUNNING / NEW PROSPECTIVE FREEZE")
    print("Q15/Q5 freeze:", st.get("frozen_at_utc"))
    print("Historical catch-up:", st.get("catchup_state"), "(filtered background scan, NOT OOS)")
    print("Dashboard:", DASHBOARD_VERSION)

    await watch_live_research_stack(
        refresh_seconds=refresh_seconds,
        show_rows=show_rows,
    )


async def stop_live_research_stack():
    """Intentional full-session shutdown only: scaled monitor first, then base stack."""
    _wire_fast_monitor()
    try:
        stop_range44_q15q5_monitor(show=False)
    except Exception:
        pass
    return await _stop_base_live_stack()


live_stack_snapshot = _v4.live_stack_snapshot


__all__ = [
    "DASHBOARD_VERSION",
    "render_live_research_stack_html",
    "watch_live_research_stack",
    "start_live_research_stack",
    "stop_live_research_stack",
    "live_stack_snapshot",
]
