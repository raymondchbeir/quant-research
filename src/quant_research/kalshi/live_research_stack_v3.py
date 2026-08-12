from __future__ import annotations

import asyncio
import html
from pathlib import Path

import numpy as np
import pandas as pd

from .live_research_stack import (
    live_stack_snapshot,
    prepare_live_research_stack,
    stop_live_research_stack as _stop_base_live_stack,
)
from .live_research_stack_v2 import (
    render_live_research_stack_html as _render_v2_html,
)
from .pre_m5_range44_scaled_strategy import (
    NORMAL_QTY,
    FLAGGED_QTY,
    range44_q15q5_status,
    start_range44_q15q5_monitor,
    stop_range44_q15q5_monitor,
)

DASHBOARD_VERSION = "KALSHI_LIVE_RESEARCH_DASHBOARD_V3_Q15_Q5"


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


def _same_session(a, b):
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


def _metric(label, value, sub="", state="neutral", large=False):
    colors = {
        "ok": ("#f0fff4", "#1a7f37"),
        "warn": ("#fffbea", "#9a6700"),
        "bad": ("#fff1f0", "#cf222e"),
        "info": ("#f0f7ff", "#0969da"),
        "neutral": ("#f8fafc", "#64748b"),
    }
    bg, border = colors.get(state, colors["neutral"])
    value_size = "27px" if large else "20px"
    min_width = "165px" if large else "135px"
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:11px;'
        f'padding:11px 13px;min-width:{min_width};flex:1">'
        f'<div style="font-size:10px;font-weight:700;opacity:.65;text-transform:uppercase;'
        f'letter-spacing:.055em">{html.escape(str(label))}</div>'
        f'<div style="font-size:{value_size};font-weight:800;line-height:1.15;margin-top:3px">'
        f'{html.escape(str(value))}</div>'
        + (
            f'<div style="font-size:11px;opacity:.68;margin-top:3px">{html.escape(str(sub))}</div>'
            if sub
            else ""
        )
        + "</div>"
    )


def _table(df, cols, rename=None, digits=4, empty="No rows yet."):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return f'<div style="opacity:.58;font-size:12px;padding:7px 2px">{html.escape(empty)}</div>'

    t = df[[c for c in cols if c in df.columns]].copy()
    if rename:
        t = t.rename(columns=rename)

    for c in t.columns:
        if pd.api.types.is_numeric_dtype(t[c]):
            t[c] = pd.to_numeric(t[c], errors="coerce").round(digits)

    return (
        '<div style="overflow-x:auto;border:1px solid #e5e7eb;border-radius:10px;'
        'padding:3px 7px;background:white;font-size:12px">'
        + t.to_html(index=False, border=0)
        + "</div>"
    )


def _summary_row(status):
    summary = status.get("summary")
    if isinstance(summary, pd.DataFrame) and len(summary):
        return summary.iloc[0]
    return pd.Series(dtype=object)


def _render_scaled_candidate_panel(base_snapshot, scaled_status):
    rr = _summary_row(scaled_status)

    running = bool(scaled_status.get("running"))
    session_match = bool(
        running
        and _same_session(
            scaled_status.get("session_dir"),
            base_snapshot.get("recorder_session"),
        )
    )
    healthy = running and session_match and not scaled_status.get("last_error")

    pro_pnl = _num(rr.get("strategy_realized_pnl"), 0.0)
    pro_q3 = _num(rr.get("q3_realized_pnl_same_sample"), 0.0)
    pro_delta = pro_pnl - pro_q3
    pro_dd = _num(rr.get("max_drawdown"), 0.0)
    pro_worst = _num(rr.get("worst_complete_window_pnl"), 0.0)
    pro_windows = int(_num(rr.get("prospective_windows"), 0))
    pro_flags = int(_num(rr.get("flagged_windows"), 0))
    pro_settled = _num(rr.get("settled_contracts"), 0.0)
    pro_open = _num(rr.get("open_contracts"), 0.0)

    catch_pnl = _num(rr.get("catchup_realized_pnl"), 0.0)
    catch_q3 = _num(rr.get("catchup_q3_realized_pnl"), 0.0)
    catch_delta = catch_pnl - catch_q3
    catch_dd = _num(rr.get("catchup_max_drawdown"), 0.0)
    catch_worst = _num(rr.get("catchup_worst_complete_window_pnl"), 0.0)
    catch_windows = int(_num(rr.get("catchup_windows"), 0))
    catch_flags = int(_num(rr.get("catchup_flagged_windows"), 0))
    catch_settled = _num(rr.get("catchup_settled_contracts"), 0.0)
    catch_open = _num(rr.get("catchup_open_contracts"), 0.0)

    border = "#1a7f37" if healthy else "#cf222e"
    bg = "#f0fff4" if healthy else "#fff1f0"
    state_text = "RUNNING / SAME SESSION" if healthy else "ATTENTION"

    parts = [
        '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;'
        'max-width:1450px;color:#111827;margin-bottom:14px">',
        f'<div style="border:2px solid {border};background:{bg};border-radius:15px;'
        'padding:15px 16px;margin-bottom:12px">',
        '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap">',
        '<div>',
        '<div style="font-size:11px;font-weight:800;color:#0969da;letter-spacing:.08em;'
        'text-transform:uppercase">NEW SCALED CANDIDATE</div>',
        '<div style="font-size:28px;font-weight:900;margin-top:2px">RANGE44 Q15/Q5</div>',
        f'<div style="font-size:13px;margin-top:4px"><b>Q{int(NORMAL_QTY)} normally</b> · '
        f'Range44-flagged windows size to <b>Q{int(FLAGGED_QTY)}</b>.</div>',
        '<div style="font-size:11px;opacity:.68;margin-top:5px">'
        'Same frozen signal, entry price, 15-second lifetime, Range44 flags, and settlement labels. '
        'Only sizing/capacity changes.</div>',
        '</div>',
        '<div style="text-align:right">',
        f'<div style="font-size:12px;font-weight:800;color:{border}">{html.escape(state_text)}</div>',
        f'<div style="font-size:10px;opacity:.62;margin-top:4px">Dashboard: {DASHBOARD_VERSION}</div>',
        '</div></div>',
        f'<div style="font-size:10px;opacity:.62;margin-top:8px">'
        f'Prospective freeze: {html.escape(str(scaled_status.get("frozen_at_utc")))} · '
        f'Catch-up anchor: {html.escape(str(scaled_status.get("catchup_anchor_utc")))}</div>',
        '</div>',

        '<div style="border:2px solid #0969da;background:linear-gradient(135deg,#eef6ff,#f8fbff);'
        'border-radius:15px;padding:14px 15px;margin-bottom:12px">',
        '<div style="font-size:18px;font-weight:850">TRUE PROSPECTIVE Q15/Q5</div>',
        '<div style="font-size:11px;opacity:.66;margin-top:2px">'
        'Only decisions strictly after the new Q15/Q5 freeze count here.</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:11px">',
        _metric("Q15/Q5 PnL", _money(pro_pnl), "prospective only", "ok" if pro_pnl >= 0 else "warn", True),
        _metric("Q3 same sample", _money(pro_q3), "same new windows", "neutral", True),
        _metric("Δ vs Q3", _money(pro_delta), "prospective", "ok" if pro_delta >= 0 else "warn", True),
        _metric("Max DD", _money(pro_dd), "complete prospective windows", "neutral", True),
        _metric("Worst window", _money(pro_worst), "complete prospective", "neutral", True),
        '</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">',
        _metric("Prospective windows", pro_windows, "new freeze onward", "info"),
        _metric("Range44 flagged", pro_flags, "Q5 windows", "warn"),
        _metric("Settled / open", f"{pro_settled:.2f} / {pro_open:.2f}", "contracts", "neutral"),
        '</div>',
        '</div>',

        '<div style="border:1px solid #9a6700;background:#fffbea;border-radius:13px;'
        'padding:13px 14px;margin-bottom:12px">',
        '<div style="font-size:17px;font-weight:850">HISTORICAL CATCH-UP — NOT OOS</div>',
        '<div style="font-size:11px;opacity:.68;margin-top:2px">'
        'Reconstructs Q15/Q5 over the original Range44 prospective sample using calibrated finite-flow capacity. '
        'This was selected after seeing that period, so it is context only.</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">',
        _metric("Catch-up PnL", _money(catch_pnl), "historical counterfactual", "ok" if catch_pnl >= 0 else "warn", True),
        _metric("Q3 catch-up", _money(catch_q3), "same historical windows", "neutral", True),
        _metric("Δ vs Q3", _money(catch_delta), "NOT OOS", "ok" if catch_delta >= 0 else "warn", True),
        _metric("Catch-up DD", _money(catch_dd), "historical", "neutral", True),
        _metric("Worst window", _money(catch_worst), "historical", "neutral", True),
        '</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">',
        _metric("Catch-up windows", catch_windows, "original Range44 sample", "info"),
        _metric("Catch-up flags", catch_flags, "historical Q5 windows", "warn"),
        _metric("Settled / open", f"{catch_settled:.2f} / {catch_open:.2f}", "contracts", "neutral"),
        '</div>',
        '</div>',
    ]

    latest = scaled_status.get("latest_windows")
    parts += [
        '<div style="font-size:17px;font-weight:850;margin:15px 0 6px">LATEST Q15/Q5 PROSPECTIVE WINDOWS</div>',
        _table(
            latest,
            [
                "decision_time", "signals", "max_mid_range_c", "range_flagged",
                "target_qty_per_asset", "strategy_filled_contracts",
                "strategy_open_contracts", "strategy_realized_pnl",
                "q3_realized_pnl_same_sample",
            ],
            {
                "decision_time": "M5",
                "max_mid_range_c": "max range c",
                "range_flagged": "flagged",
                "target_qty_per_asset": "target Q",
                "strategy_filled_contracts": "Q15/Q5 ct",
                "strategy_open_contracts": "open ct",
                "strategy_realized_pnl": "Q15/Q5 PnL",
                "q3_realized_pnl_same_sample": "Q3 PnL",
            },
            empty="No post-Q15/Q5-freeze window yet.",
        ),
        '</div>',
    ]

    return "".join(parts)


def render_live_research_stack_html(show_rows=8):
    base_snapshot = live_stack_snapshot()
    scaled = range44_q15q5_status(show=False)

    scaled_panel = _render_scaled_candidate_panel(base_snapshot, scaled)

    base_html = _render_v2_html(show_rows=show_rows)
    base_html = base_html.replace(
        "Current lead candidate",
        "Prior frozen benchmark candidate",
        1,
    )
    base_html = base_html.replace(
        ">RANGE44_Q1<",
        ">RANGE44 Q3/Q1<",
        1,
    )

    return scaled_panel + base_html


async def watch_live_research_stack(refresh_seconds=2.0, show_rows=8):
    refresh_seconds = max(0.5, float(refresh_seconds))

    try:
        from IPython.display import HTML, display
    except Exception as exc:
        raise RuntimeError(
            "Live research dashboard requires Jupyter/IPython display support."
        ) from exc

    handle = display(
        HTML(render_live_research_stack_html(show_rows=show_rows)),
        display_id=True,
    )

    try:
        while True:
            await asyncio.sleep(refresh_seconds)
            handle.update(
                HTML(render_live_research_stack_html(show_rows=show_rows))
            )

    except (KeyboardInterrupt, asyncio.CancelledError):
        handle.update(
            HTML(render_live_research_stack_html(show_rows=show_rows))
        )
        print(
            "Dashboard refresh stopped. Recorder, frozen Q3 primary, Q3/Q1 Range44, "
            "Q15/Q5 scaled monitor, and read-only controls remain running."
        )


async def start_live_research_stack(
    refresh_seconds=2.0,
    show_rows=10,
    duration_minutes=None,
    key_id=None,
    private_key_path=None,
):
    """Adopt the existing live session, attach Q15/Q5, catch up, then watch."""
    check = await prepare_live_research_stack(
        duration_minutes=duration_minutes,
        key_id=key_id,
        private_key_path=private_key_path,
    )

    session_dir = Path(check["recorder_session"])

    scaled = range44_q15q5_status(show=False)

    if scaled.get("running") and not _same_session(
        scaled.get("session_dir"), session_dir
    ):
        stop_range44_q15q5_monitor(show=False)
        scaled = {"running": False}

    if not scaled.get("running"):
        scaled = start_range44_q15q5_monitor(
            session_dir=session_dir,
            interval_sec=10.0,
            show=True,
        )

    final_scaled = range44_q15q5_status(show=False)

    if not final_scaled.get("running"):
        raise RuntimeError("Q15/Q5 monitor failed to start.")

    if not _same_session(final_scaled.get("session_dir"), session_dir):
        raise RuntimeError(
            "Q15/Q5 monitor session mismatch; refusing to show a mixed-session dashboard."
        )

    if final_scaled.get("last_error"):
        raise RuntimeError(
            f"Q15/Q5 monitor started with error: {final_scaled.get('last_error')}"
        )

    print()
    print("LIVE STACK V3 READY")
    print("Session:", session_dir)
    print("Frozen Q3 primary: RUNNING / UNCHANGED")
    print("Original Range44 Q3/Q1: RUNNING / BENCHMARK")
    print("New scaled candidate: RANGE44 Q15/Q5 / RUNNING")
    print("Q15/Q5 prospective frozen at:", final_scaled.get("frozen_at_utc"))
    print("Q15/Q5 historical catch-up anchor:", final_scaled.get("catchup_anchor_utc"))
    print("Catch-up is HISTORICAL / NOT OOS.")
    print("All execution remains simulated/read-only.")

    await watch_live_research_stack(
        refresh_seconds=refresh_seconds,
        show_rows=show_rows,
    )


async def stop_live_research_stack():
    """Orderly full cleanup when intentionally ending the session."""
    try:
        stop_range44_q15q5_monitor(show=False)
    except Exception:
        pass

    return await _stop_base_live_stack()


__all__ = [
    "render_live_research_stack_html",
    "watch_live_research_stack",
    "start_live_research_stack",
    "stop_live_research_stack",
    "live_stack_snapshot",
]
