from __future__ import annotations

import asyncio
import html
from pathlib import Path

import numpy as np
import pandas as pd

from .live_research_stack import live_stack_snapshot, prepare_live_research_stack
from .live_research_stack_v2 import render_live_research_stack_html as _render_v2_html
from .pre_m5_range44_scaled_strategy_v2 import (
    range44_q15q5_status,
    start_range44_q15q5_monitor,
    stop_range44_q15q5_monitor,
)

DASHBOARD_VERSION = "KALSHI_LIVE_RESEARCH_DASHBOARD_V4_Q15_Q5_NONBLOCKING"


def _same_session(a, b):
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


def _num(x, default=0.0):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _money(x):
    try:
        y = float(x)
        return f"${y:+.4f}" if np.isfinite(y) else "n/a"
    except Exception:
        return "n/a"


def _metric(label, value, sub="", state="neutral"):
    colors = {
        "ok": ("#f0fff4", "#1a7f37"),
        "warn": ("#fffbea", "#9a6700"),
        "bad": ("#fff1f0", "#cf222e"),
        "info": ("#f0f7ff", "#0969da"),
        "neutral": ("#f8fafc", "#64748b"),
    }
    bg, border = colors.get(state, colors["neutral"])
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:10px 12px;min-width:145px;flex:1">'
        f'<div style="font-size:10px;font-weight:700;opacity:.65;text-transform:uppercase">{html.escape(str(label))}</div>'
        f'<div style="font-size:21px;font-weight:800;margin-top:3px">{html.escape(str(value))}</div>'
        + (f'<div style="font-size:11px;opacity:.68;margin-top:3px">{html.escape(str(sub))}</div>' if sub else "")
        + '</div>'
    )


def _table(df, cols, rename=None):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return '<div style="font-size:12px;opacity:.6">No prospective Q15/Q5 windows yet.</div>'
    t = df[[c for c in cols if c in df.columns]].copy()
    if rename:
        t = t.rename(columns=rename)
    for c in t.columns:
        if pd.api.types.is_numeric_dtype(t[c]):
            t[c] = pd.to_numeric(t[c], errors="coerce").round(4)
    return '<div style="overflow-x:auto;font-size:12px">' + t.to_html(index=False, border=0) + '</div>'


def _scaled_panel(base, st):
    summary = st.get("summary")
    rr = summary.iloc[0] if isinstance(summary, pd.DataFrame) and len(summary) else pd.Series(dtype=object)

    healthy = bool(st.get("running")) and _same_session(st.get("session_dir"), base.get("recorder_session")) and not st.get("last_error")
    catch_state = str(st.get("catchup_state") or "PENDING")
    catch_pct = _num(st.get("catchup_progress_pct"), 0.0)

    pro_pnl = _num(rr.get("strategy_realized_pnl"), 0.0)
    pro_q3 = _num(rr.get("q3_realized_pnl_same_sample"), 0.0)
    pro_dd = _num(rr.get("max_drawdown"), 0.0)
    pro_worst = _num(rr.get("worst_complete_window_pnl"), 0.0)
    pro_windows = int(_num(rr.get("prospective_windows"), 0))
    pro_flags = int(_num(rr.get("flagged_windows"), 0))

    catch_pnl_raw = rr.get("catchup_realized_pnl", np.nan)
    catch_q3_raw = rr.get("catchup_q3_realized_pnl", np.nan)
    catch_pnl = _money(catch_pnl_raw) if pd.notna(catch_pnl_raw) else "scanning"
    catch_q3 = _money(catch_q3_raw) if pd.notna(catch_q3_raw) else "scanning"
    catch_dd_raw = rr.get("catchup_max_drawdown", np.nan)
    catch_dd = _money(catch_dd_raw) if pd.notna(catch_dd_raw) else "scanning"
    catch_windows = int(_num(rr.get("catchup_windows"), 0))
    catch_flags = int(_num(rr.get("catchup_flagged_windows"), 0))

    border = "#1a7f37" if healthy else "#cf222e"

    parts = [
        '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1450px;color:#111827">',
        f'<div style="border:2px solid {border};border-radius:15px;padding:15px 16px;margin-bottom:12px;background:#f8fbff">',
        '<div style="font-size:11px;font-weight:800;color:#0969da;letter-spacing:.08em;text-transform:uppercase">NEW SCALED CANDIDATE</div>',
        '<div style="font-size:28px;font-weight:900;margin-top:2px">RANGE44 Q15/Q5</div>',
        '<div style="font-size:12px;margin-top:4px">Q15 normal windows · Q5 Range44-flagged windows · read-only finite-flow capacity shadow.</div>',
        f'<div style="font-size:10px;opacity:.65;margin-top:5px">Prospective freeze: {html.escape(str(st.get("frozen_at_utc")))} · dashboard {DASHBOARD_VERSION}</div>',
        '</div>',

        '<div style="border:2px solid #0969da;border-radius:13px;padding:13px 14px;margin-bottom:11px;background:#eef6ff">',
        '<div style="font-size:18px;font-weight:850">TRUE PROSPECTIVE — NEW FREEZE ONLY</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:9px">',
        _metric("Q15/Q5 PnL", _money(pro_pnl), "prospective only", "ok" if pro_pnl >= 0 else "warn"),
        _metric("Q3 same sample", _money(pro_q3), "same new windows", "neutral"),
        _metric("Δ vs Q3", _money(pro_pnl - pro_q3), "prospective", "ok" if pro_pnl - pro_q3 >= 0 else "warn"),
        _metric("Max DD", _money(pro_dd), "prospective", "neutral"),
        _metric("Worst window", _money(pro_worst), "prospective", "neutral"),
        _metric("Windows / flags", f"{pro_windows} / {pro_flags}", "Q15 normal / Q5 flagged", "info"),
        '</div></div>',

        '<div style="border:1px solid #9a6700;border-radius:13px;padding:13px 14px;margin-bottom:11px;background:#fffbea">',
        '<div style="font-size:17px;font-weight:850">HISTORICAL CATCH-UP — NOT OOS</div>',
        f'<div style="font-size:11px;opacity:.7;margin-top:2px">State: <b>{html.escape(catch_state)}</b> · scanned {catch_pct:.1f}% of the frozen historical trade snapshot. '
        'This scan is separate from live prospective tracking.</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:9px">',
        _metric("Catch-up PnL", catch_pnl, "historical counterfactual", "warn" if catch_state != "READY" else "neutral"),
        _metric("Q3 catch-up", catch_q3, "same historical period", "neutral"),
        _metric("Catch-up DD", catch_dd, "NOT OOS", "neutral"),
        _metric("Windows / flags", f"{catch_windows} / {catch_flags}", "old Range44 sample", "info"),
        '</div></div>',

        '<div style="font-size:17px;font-weight:850;margin:14px 0 6px">LATEST Q15/Q5 PROSPECTIVE WINDOWS</div>',
        _table(
            st.get("latest_windows"),
            ["decision_time", "signals", "max_mid_range_c", "range_flagged", "target_qty_per_asset", "strategy_filled_contracts", "strategy_open_contracts", "strategy_realized_pnl", "q3_realized_pnl_same_sample"],
            {"decision_time": "M5", "max_mid_range_c": "max range c", "range_flagged": "flagged", "target_qty_per_asset": "target Q", "strategy_filled_contracts": "Q15/Q5 ct", "strategy_open_contracts": "open ct", "strategy_realized_pnl": "Q15/Q5 PnL", "q3_realized_pnl_same_sample": "Q3 PnL"},
        ),
        '</div>',
    ]
    return ''.join(parts)


def render_live_research_stack_html(show_rows=10):
    base = live_stack_snapshot()
    st = range44_q15q5_status(show=False)
    old = _render_v2_html(show_rows=show_rows)
    old = old.replace("Current lead candidate", "Prior frozen benchmark candidate", 1)
    return _scaled_panel(base, st) + old


async def watch_live_research_stack(refresh_seconds=2.0, show_rows=10):
    refresh_seconds = max(0.5, float(refresh_seconds))
    from IPython.display import HTML, display
    handle = display(HTML(render_live_research_stack_html(show_rows=show_rows)), display_id=True)
    try:
        while True:
            await asyncio.sleep(refresh_seconds)
            handle.update(HTML(render_live_research_stack_html(show_rows=show_rows)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        handle.update(HTML(render_live_research_stack_html(show_rows=show_rows)))
        print("Dashboard refresh stopped. Recorder, Q3 primary, Q3/Q1 benchmark, Q15/Q5 monitor, and catch-up worker remain running.")


async def start_live_research_stack(refresh_seconds=2.0, show_rows=10, duration_minutes=None, key_id=None, private_key_path=None):
    check = await prepare_live_research_stack(
        duration_minutes=duration_minutes,
        key_id=key_id,
        private_key_path=private_key_path,
    )
    session_dir = Path(check["recorder_session"])

    st = range44_q15q5_status(show=False)
    if st.get("running") and not _same_session(st.get("session_dir"), session_dir):
        stop_range44_q15q5_monitor(show=False)
        st = {"running": False}
    if not st.get("running"):
        start_range44_q15q5_monitor(session_dir=session_dir, interval_sec=10.0, show=True)

    st = range44_q15q5_status(show=False)
    if not st.get("running") or not _same_session(st.get("session_dir"), session_dir):
        raise RuntimeError("Q15/Q5 V2 monitor failed session-alignment gate.")

    print()
    print("LIVE STACK V4 READY")
    print("Session:", session_dir)
    print("Recorder: RUNNING / UNCHANGED")
    print("Frozen Q3 primary: RUNNING / UNCHANGED")
    print("Range44 Q3/Q1: RUNNING / BENCHMARK")
    print("Range44 Q15/Q5 V2: RUNNING / NEW PROSPECTIVE FREEZE")
    print("Q15/Q5 freeze:", st.get("frozen_at_utc"))
    print("Historical catch-up:", st.get("catchup_state"), "(background, NOT OOS)")

    await watch_live_research_stack(refresh_seconds=refresh_seconds, show_rows=show_rows)


__all__ = ["start_live_research_stack", "watch_live_research_stack", "render_live_research_stack_html"]
