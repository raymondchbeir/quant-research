from __future__ import annotations

import asyncio
import html

import numpy as np
import pandas as pd

from . import shadow_dashboard as D


def _money(x):
    return f"${float(x):+.4f}"


def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{float(x):.2f}%"


def _card(label, value, sub=""):
    sub_html = f'<div style="font-size:10px;opacity:.62;">{html.escape(str(sub))}</div>' if sub else ""
    return (
        '<div style="padding:9px 11px;border:1px solid #ddd;border-radius:8px;min-width:125px;">'
        f'<div style="font-size:11px;opacity:.65;">{html.escape(str(label))}</div>'
        f'<div style="font-size:19px;font-weight:600;">{html.escape(str(value))}</div>'
        f'{sub_html}</div>'
    )


def render_primary_shadow_html(show_rows=8):
    snap = D.status_snapshot()
    if snap is None:
        return '<div style="font-family:system-ui"><b>Primary shadow trader is not running.</b></div>'
    health, perf, contracts, df = snap["health"], snap["metrics"], snap["contracts"], snap["df"]
    overall_ok = bool(health.get("healthy")) and all(snap["thread_state"].values())
    banner = "HEALTHY" if overall_ok else "ATTENTION"
    banner_bg = "#e9f7ef" if overall_ok else "#fff3cd"
    now_text = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    parts = [
        '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;">',
        f'<div style="padding:10px 12px;border-radius:8px;background:{banner_bg};margin-bottom:10px;">',
        f'<b>PRIMARY SHADOW — {banner}</b> &nbsp; <span style="opacity:.7">{html.escape(now_text)}</span><br>',
        '<span style="font-size:12px">M5 | BTC completed-1m 15m opposition | spread ≤2c | -3c | actual-post 15s | 3ct | HOLD TO SETTLEMENT | READ-ONLY</span>',
        '</div>',
    ]
    if not health.get("healthy"):
        parts.append(
            '<div style="padding:9px;border:1px solid #c90;border-radius:6px;margin-bottom:10px;">'
            '<b>Recorder/data health invalid.</b> New M5 decisions become DATA_INVALID, not trades. '
            f'epoch={html.escape(str(health.get("connection_epoch")))}; '
            f'market-data age={html.escape(str(health.get("market_data_age_s")))}s; '
            f'expired={html.escape(str(health.get("expired_active_markets")))}.</div>'
        )
    parts.append('<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">')
    parts.extend([
        _card("Signals", perf["signals"]),
        _card("DATA_INVALID", perf["data_invalid"]),
        _card("Any fills", perf["fills"], _pct(perf["fill_rate"])),
        _card("Realized PnL", _money(perf["pnl"])),
        _card("Max DD", f'${perf["max_dd"]:.4f}'),
        _card("W / L", f'{perf["wins"]} / {perf["losses"]}', _pct(perf["filled_accuracy"])),
        _card("Open positions", perf["open_positions"], f'{perf["open_contracts"]:.2f} contracts'),
        _card("Decisions", snap["evaluated"], f'{sum(snap["skips"].values())} skipped'),
    ])
    parts.append('</div>')
    parts.append('<h4 style="margin:8px 0 4px">Active frozen contracts</h4>')
    if contracts.empty:
        parts.append('<div style="opacity:.65">No active frozen-universe contracts currently in recorder state.</div>')
    else:
        cols = ["series", "to_M5", "to_close", "YES_bid", "YES_ask", "mid_c", "spread_c", "lean", "BTC15_bp", "opp?", "shadow_state", "position", "fill_qty", "realized_pnl", "quote_age_s", "book_age_s", "epoch", "ticker"]
        parts.append('<div style="overflow-x:auto;font-size:12px;">' + contracts[cols].to_html(index=False, border=0) + '</div>')
    parts.append('<div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:12px;">')
    parts.append(
        '<div><b>Recorder health</b><br>'
        f'running={html.escape(str(health.get("running")))} &nbsp; healthy={html.escape(str(health.get("healthy")))}<br>'
        f'epoch={html.escape(str(health.get("connection_epoch")))} &nbsp; active={html.escape(str(health.get("active_markets")))}<br>'
        f'supervisor age={html.escape(str(health.get("supervisor_age_s")))}s<br>'
        f'market-data age={html.escape(str(health.get("market_data_age_s")))}s<br></div>'
    )
    parts.append('<div><b>Decision skips</b><br>')
    if snap["skips"]:
        for reason, count in snap["skips"].most_common():
            parts.append(f'{html.escape(str(reason))}: {count}<br>')
    else:
        parts.append('none yet<br>')
    parts.append('</div><div><b>DATA_INVALID</b><br>')
    if snap["invalids"]:
        for reason, count in snap["invalids"].most_common():
            parts.append(f'{html.escape(str(reason))}: {count}<br>')
    else:
        parts.append('none<br>')
    parts.append('</div><div><b>Thread health</b><br>')
    for name, alive in snap["thread_state"].items():
        parts.append(f'{html.escape(name)}: {"OK" if alive else "DEAD"}<br>')
    parts.append('</div></div>')
    if show_rows and not df.empty:
        latest = df.tail(int(show_rows))[["decision_time", "actual_post_time", "ticker", "direction", "entry_price", "entry_queue", "quote_age_s", "book_age_s", "entry_fill_qty", "result", "realized_pnl", "status"]]
        parts.append('<h4 style="margin:12px 0 4px">Latest decisions / signals</h4>')
        parts.append('<div style="overflow-x:auto;font-size:12px;">' + latest.to_html(index=False, border=0) + '</div>')
    parts.append('</div>')
    return ''.join(parts)


async def watch_primary_shadow_status(refresh_seconds=2.0, show_rows=8):
    """Update one Jupyter display in place without blocking recorder asyncio tasks."""
    refresh_seconds = max(0.5, float(refresh_seconds))
    try:
        from IPython.display import HTML, display
    except Exception as exc:
        raise RuntimeError("Live dashboard requires Jupyter/IPython display support.") from exc
    handle = display(HTML(render_primary_shadow_html(show_rows=show_rows)), display_id=True)
    try:
        while True:
            await asyncio.sleep(refresh_seconds)
            handle.update(HTML(render_primary_shadow_html(show_rows=show_rows)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        handle.update(HTML(render_primary_shadow_html(show_rows=show_rows)))
        print("Dashboard refresh stopped. Recorder and primary shadow remain unchanged.")


watch_shadow_status = watch_primary_shadow_status
