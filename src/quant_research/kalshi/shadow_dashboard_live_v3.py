from __future__ import annotations

import asyncio

from .shadow_dashboard_live_v2 import render_primary_shadow_html


async def watch_primary_shadow_status(refresh_seconds=2.0, show_rows=8):
    """
    Jupyter-safe live dashboard.

    IMPORTANT: this is async on purpose. The recorder itself runs on the
    notebook's asyncio event loop, so a synchronous while/time.sleep refresh
    loop would starve the recorder, market rotation, reconnect watchdog, and
    stop calls.

    Use:
        await watch_primary_shadow_status(refresh_seconds=2)

    Interrupting this cell stops only the dashboard refresh. Recorder and
    shadow threads/tasks keep running.
    """
    refresh_seconds = max(0.5, float(refresh_seconds))

    try:
        from IPython.display import HTML, display
    except Exception as exc:
        raise RuntimeError(
            "Stable live dashboard requires Jupyter/IPython display support."
        ) from exc

    handle = display(
        HTML(render_primary_shadow_html(show_rows=show_rows)),
        display_id=True,
    )

    try:
        while True:
            # Yield to the notebook event loop BEFORE refreshing. This lets the
            # recorder supervisor, websocket consumer, watchdog and timers run.
            await asyncio.sleep(refresh_seconds)
            handle.update(
                HTML(render_primary_shadow_html(show_rows=show_rows))
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        try:
            handle.update(
                HTML(render_primary_shadow_html(show_rows=show_rows))
            )
        except Exception:
            pass
        print(
            "Dashboard refresh stopped. Recorder and shadow trader are still running."
        )
        return None


watch_shadow_status = watch_primary_shadow_status
