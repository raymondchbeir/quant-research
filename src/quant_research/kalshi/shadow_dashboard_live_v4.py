from __future__ import annotations

import asyncio
import html
from pathlib import Path

import pandas as pd

from . import shadow_dashboard as D
from .shadow_dashboard_live_v3 import render_primary_shadow_html as _render_v3


def _same_session(a, b):
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


def _strategy_footer_html(primary_session=None):
    try:
        from .pre_m5_risk_strategy import pre_m5_risk_strategy_status

        st = pre_m5_risk_strategy_status(show=False)
        running = bool(st.get("running"))
        session = st.get("session_dir")
        match = _same_session(primary_session, session) if primary_session else True
        state = "RUNNING" if running else "STOPPED"
        if running:
            state += " · SAME SESSION" if match else " · SESSION MISMATCH"
        border = "#c7e8d4" if running and match else "#f0c36d" if running else "#ddd"

        parts = [
            '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;'
            'margin-top:10px;padding:10px;border:1px solid %s;border-radius:8px;">' % border,
            f'<b>M1→M5 risk-sizing strategy:</b> {html.escape(state)}',
            '<div style="font-size:11px;opacity:.65;margin:4px 0 7px">'
            'DEVELOPMENT COUNTERFACTUAL · Q3 normally · in ≥3-signal windows, ≥2/4 frozen risk votes → Q1 · READ-ONLY'
            '</div>',
        ]

        if running:
            parts.append(
                f'<div style="font-size:11px;opacity:.65;margin-bottom:6px">'
                f'Frozen: {html.escape(str(st.get("frozen_at_utc")))}<br>'
                f'{html.escape(str(session))}'
                '</div>'
            )
            summary = st.get("summary")
            if isinstance(summary, pd.DataFrame) and len(summary):
                cols = [
                    "prospective_windows", "eligible_windows", "high_breadth_windows", "flagged_windows",
                    "settled_contracts", "open_contracts", "strategy_realized_pnl",
                    "q3_realized_pnl_same_sample", "pnl_change_vs_q3", "max_drawdown",
                    "q3_max_drawdown_same_sample", "worst_complete_window_pnl",
                ]
                t = summary[[c for c in cols if c in summary.columns]].copy()
                rename = {
                    "prospective_windows": "prospective windows",
                    "eligible_windows": "eligible",
                    "high_breadth_windows": "HB",
                    "flagged_windows": "flagged",
                    "settled_contracts": "settled ct",
                    "open_contracts": "open ct",
                    "strategy_realized_pnl": "M1M5 PnL",
                    "q3_realized_pnl_same_sample": "Q3 same-sample PnL",
                    "pnl_change_vs_q3": "Δ vs Q3",
                    "max_drawdown": "M1M5 max DD",
                    "q3_max_drawdown_same_sample": "Q3 max DD",
                    "worst_complete_window_pnl": "worst window",
                }
                t = t.rename(columns=rename)
                for c in ("settled ct", "open ct", "M1M5 PnL", "Q3 same-sample PnL", "Δ vs Q3", "M1M5 max DD", "Q3 max DD", "worst window"):
                    if c in t.columns:
                        t[c] = pd.to_numeric(t[c], errors="coerce").round(4)
                parts.append('<div style="overflow-x:auto;font-size:12px;">' + t.to_html(index=False, border=0) + '</div>')

            latest = st.get("latest_windows")
            if isinstance(latest, pd.DataFrame) and len(latest):
                cols = [
                    "decision_time", "signals", "risk_score", "risk_flagged", "strategy_decision",
                    "strategy_realized_pnl", "q3_realized_pnl_same_sample", "pnl_change_vs_q3",
                ]
                t = latest[[c for c in cols if c in latest.columns]].copy().tail(5)
                t = t.rename(columns={
                    "decision_time": "M5", "signals": "signals", "risk_score": "risk score",
                    "risk_flagged": "flagged", "strategy_decision": "decision",
                    "strategy_realized_pnl": "M1M5 PnL", "q3_realized_pnl_same_sample": "Q3 PnL",
                    "pnl_change_vs_q3": "Δ",
                })
                for c in ("M1M5 PnL", "Q3 PnL", "Δ"):
                    if c in t.columns:
                        t[c] = pd.to_numeric(t[c], errors="coerce").round(4)
                parts.append('<div style="font-size:11px;font-weight:600;margin-top:7px">Latest strategy-prospective windows</div>')
                parts.append('<div style="overflow-x:auto;font-size:12px;">' + t.to_html(index=False, border=0) + '</div>')
            else:
                parts.append('<div style="opacity:.65">No post-strategy-freeze M5 window yet.</div>')

            if not match:
                parts.append(
                    '<div style="margin-top:6px;color:#8a5a00;font-size:12px;"><b>Warning:</b> '
                    'strategy counterfactual is attached to a different session.</div>'
                )
        else:
            parts.append('<div style="opacity:.65">Start the M1→M5 risk strategy monitor to collect prospective PnL.</div>')

        if st.get("last_error"):
            parts.append(f'<div style="color:#a00;font-size:12px">last error: {html.escape(str(st.get("last_error")))}</div>')
        parts.append('</div>')
        return ''.join(parts)
    except Exception as exc:
        return (
            '<div style="font-family:system-ui;margin-top:10px;padding:10px;border:1px solid #f0c36d;border-radius:8px;">'
            f'<b>M1→M5 risk-sizing strategy:</b> status unavailable · {html.escape(repr(exc))}</div>'
        )


def render_primary_shadow_html(show_rows=8):
    base = _render_v3(show_rows=show_rows)
    snap = D.status_snapshot()
    primary_session = getattr(snap.get("sh"), "session_dir", None) if snap is not None else None
    return base + _strategy_footer_html(primary_session=primary_session)


async def watch_primary_shadow_status(refresh_seconds=2.0, show_rows=8):
    """Update one Jupyter display in place; all research sections are read-only."""
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
        print("Dashboard refresh stopped. Recorder, primary shadow, and research monitors remain unchanged.")


watch_shadow_status = watch_primary_shadow_status
