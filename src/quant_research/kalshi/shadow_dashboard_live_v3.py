from __future__ import annotations

import asyncio
import html
from pathlib import Path

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


def _same_session(a, b):
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


def _research_footer_html(primary_session=None):
    """Render read-only research monitor state from in-memory snapshots only.

    This helper never refreshes/recomputes either monitor and never touches recorder/shadow
    state. It is intentionally safe to call every dashboard refresh.
    """
    parts = [
        '<div style="margin-top:18px;padding-top:12px;border-top:2px solid #ddd;">',
        '<h3 style="margin:0 0 8px">Research monitors — read-only</h3>',
        '<div style="font-size:11px;opacity:.65;margin-bottom:10px;">'
        'Display-only snapshots. No orders, cancels, recorder state, or primary-shadow state are changed.'</n        '</div>',
    ]

    # Counterfactual Q3/Q2/Q1/MAX2/MAX3 monitor.
    try:
        from .risk_control_counterfactual import counterfactual_risk_status

        cf = counterfactual_risk_status(show=False)
        running = bool(cf.get("running"))
        cf_session = cf.get("session_dir")
        match = _same_session(primary_session, cf_session) if primary_session else True
        state = "RUNNING" if running else "STOPPED"
        state += " · SAME SESSION" if running and match else " · SESSION MISMATCH" if running else ""
        border = "#c7e8d4" if running and match else "#f0c36d" if running else "#ddd"
        parts.append(
            f'<div style="padding:10px;border:1px solid {border};border-radius:8px;margin-bottom:10px;">'
            f'<b>Counterfactual risk controls:</b> {html.escape(state)}'
        )
        if running:
            parts.append(
                f'<div style="font-size:11px;opacity:.65;margin:3px 0 7px">'
                f'{html.escape(str(cf_session))}'
                '</div>'
            )
            summary = cf.get("summary")
            if isinstance(summary, pd.DataFrame) and len(summary):
                cols = [
                    "scenario", "filled_contracts", "settled_filled_assets",
                    "unsettled_filled_assets", "realized_pnl", "max_drawdown",
                    "realized_pnl_change_vs_baseline",
                ]
                t = summary[[c for c in cols if c in summary.columns]].copy()
                rename = {
                    "scenario": "scenario",
                    "filled_contracts": "contracts",
                    "settled_filled_assets": "settled assets",
                    "unsettled_filled_assets": "open assets",
                    "realized_pnl": "realized PnL",
                    "max_drawdown": "max DD",
                    "realized_pnl_change_vs_baseline": "Δ vs Q3",
                }
                t = t.rename(columns=rename)
                for c in ("contracts", "realized PnL", "max DD", "Δ vs Q3"):
                    if c in t.columns:
                        t[c] = pd.to_numeric(t[c], errors="coerce").round(4)
                parts.append('<div style="overflow-x:auto;font-size:12px;">' + t.to_html(index=False, border=0) + '</div>')
            else:
                parts.append('<div style="opacity:.65">No counterfactual summary available yet.</div>')
            if not match:
                parts.append(
                    '<div style="margin-top:6px;color:#8a5a00;font-size:12px;"><b>Warning:</b> '
                    'counterfactual monitor is attached to a different session; its PnL is not comparable to this primary dashboard.</div>'
                )
        else:
            parts.append('<div style="opacity:.65;margin-top:4px">Counterfactual monitor is not running in this kernel.</div>')
        if cf.get("last_error"):
            parts.append(f'<div style="color:#a00;font-size:12px;margin-top:5px">last error: {html.escape(str(cf.get("last_error")))}</div>')
        parts.append('</div>')
    except Exception as exc:
        parts.append(
            '<div style="padding:10px;border:1px solid #f0c36d;border-radius:8px;margin-bottom:10px;">'
            f'<b>Counterfactual risk controls:</b> status unavailable · {html.escape(repr(exc))}</div>'
        )

    # Prospective M1->M5 frozen feature logger.
    try:
        from .pre_m5_prospective_monitor import pre_m5_prospective_risk_status

        pm = pre_m5_prospective_risk_status(show=False)
        running = bool(pm.get("running"))
        pm_session = pm.get("session_dir")
        match = _same_session(primary_session, pm_session) if primary_session else True
        state = "RUNNING" if running else "STOPPED"
        state += " · SAME SESSION" if running and match else " · SESSION MISMATCH" if running else ""
        border = "#c7e8d4" if running and match else "#f0c36d" if running else "#ddd"
        parts.append(
            f'<div style="padding:10px;border:1px solid {border};border-radius:8px;">'
            f'<b>Prospective M1→M5 risk collection:</b> {html.escape(state)}'
        )
        if running:
            frozen = pm.get("frozen_at_utc")
            parts.append(
                '<div style="font-size:12px;margin-top:5px">'
                f'Frozen: <b>{html.escape(str(frozen))}</b><br>'
                f'Ready windows: <b>{pm.get("ready_windows_total", 0)}</b> &nbsp; '
                f'Prospective: <b>{pm.get("prospective_windows", 0)}</b> &nbsp; '
                f'Research-eligible: <b>{pm.get("prospective_research_eligible", 0)}</b> &nbsp; '
                f'High-breadth eligible: <b>{pm.get("prospective_high_breadth_eligible", 0)}</b> &nbsp; '
                f'Execution-complete: <b>{pm.get("prospective_execution_complete", 0)}</b>'
                '</div>'
            )
            latest = pm.get("latest_windows")
            if isinstance(latest, pd.DataFrame) and len(latest):
                cols = [
                    "decision_time", "signals", "filled_assets", "actual_pnl",
                    "execution_complete", "path_complete_share_pct",
                    "max_mid_path_length_c", "max_mid_range_c", "max_mid_rv_c",
                    "m1_m5_dominant_move_share", "primary_eval_eligible",
                ]
                t = latest[[c for c in cols if c in latest.columns]].copy().tail(5)
                rename = {
                    "decision_time": "M5",
                    "signals": "signals",
                    "filled_assets": "fills",
                    "actual_pnl": "Q3 PnL",
                    "execution_complete": "settled",
                    "path_complete_share_pct": "path %",
                    "max_mid_path_length_c": "max path c",
                    "max_mid_range_c": "max range c",
                    "max_mid_rv_c": "max RV c",
                    "m1_m5_dominant_move_share": "dominant share",
                    "primary_eval_eligible": "HB eligible",
                }
                t = t.rename(columns=rename)
                for c in ("Q3 PnL", "path %", "max path c", "max range c", "max RV c", "dominant share"):
                    if c in t.columns:
                        t[c] = pd.to_numeric(t[c], errors="coerce").round(4)
                parts.append('<div style="margin-top:7px;overflow-x:auto;font-size:12px;">' + t.to_html(index=False, border=0) + '</div>')
            else:
                parts.append('<div style="opacity:.65;margin-top:5px">No post-freeze prospective M5 window yet.</div>')
            if not match:
                parts.append(
                    '<div style="margin-top:6px;color:#8a5a00;font-size:12px;"><b>Warning:</b> '
                    'M1→M5 monitor is attached to a different session.</div>'
                )
        else:
            parts.append('<div style="opacity:.65;margin-top:4px">Prospective M1→M5 monitor is not running in this kernel.</div>')
        if pm.get("last_error"):
            parts.append(f'<div style="color:#a00;font-size:12px;margin-top:5px">last error: {html.escape(str(pm.get("last_error")))}</div>')
        parts.append('</div>')
    except Exception as exc:
        parts.append(
            '<div style="padding:10px;border:1px solid #f0c36d;border-radius:8px;">'
            f'<b>Prospective M1→M5 risk collection:</b> status unavailable · {html.escape(repr(exc))}</div>'
        )

    parts.append('</div>')
    return ''.join(parts)


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

    primary_session = getattr(snap.get("sh"), "session_dir", None)
    parts.append(_research_footer_html(primary_session=primary_session))
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
