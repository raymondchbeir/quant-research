from __future__ import annotations

import asyncio
import html

import numpy as np
import pandas as pd

from .live_research_stack import (
    STACK_VERSION,
    live_stack_snapshot,
    prepare_live_research_stack,
    stop_live_research_stack,
)
from .pre_m5_range44_strategy import RANGE_THRESHOLD_C

DASHBOARD_VERSION = "KALSHI_LIVE_RESEARCH_DASHBOARD_V2"

# Frozen Aug-10 development evidence used only to explain why RANGE44_Q1 is the
# current lead candidate. These are NOT prospective performance statistics.
DEV_RANGE_ROBUST_CELLS = 9
DEV_RANGE_GRID_CELLS = 13
DEV_MEDOID_THRESHOLD_C = 43.6375
DEV_MEDOID_DELTA_VS_Q3 = 5.50
DEV_MEDOID_LOO_MIN_DELTA = 1.36
DEV_MEDOID_WORST_SESSION_DELTA = 1.16
DEV_PAIR_INCREMENTAL_DELTA = 0.00


def _num(x, default=0.0):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _money(x, signed=True):
    try:
        y = float(x)
        if not np.isfinite(y):
            return "n/a"
        return f"${y:+.4f}" if signed else f"${y:.4f}"
    except Exception:
        return "n/a"


def _pct(x):
    try:
        y = float(x)
        return f"{y:.1f}%" if np.isfinite(y) else "n/a"
    except Exception:
        return "n/a"


def _pill(text, state="neutral"):
    colors = {
        "ok": ("#dafbe1", "#1a7f37"),
        "warn": ("#fff8c5", "#9a6700"),
        "bad": ("#ffebe9", "#cf222e"),
        "info": ("#ddf4ff", "#0969da"),
        "neutral": ("#f6f8fa", "#57606a"),
    }
    bg, fg = colors.get(state, colors["neutral"])
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};font-size:11px;'
        f'font-weight:700;border-radius:999px;padding:4px 8px;margin-right:5px">'
        f'{html.escape(str(text))}</span>'
    )


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
        + (f'<div style="font-size:11px;opacity:.68;margin-top:3px">{html.escape(str(sub))}</div>' if sub else "")
        + '</div>'
    )


def _section(title, subtitle=""):
    out = (
        '<div style="margin-top:18px;margin-bottom:7px">'
        f'<div style="font-size:17px;font-weight:800">{html.escape(str(title))}</div>'
    )
    if subtitle:
        out += f'<div style="font-size:11px;opacity:.62;margin-top:2px">{html.escape(str(subtitle))}</div>'
    return out + '</div>'


def _table(df, cols, rename=None, digits=4, empty="No rows yet."):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return f'<div style="opacity:.58;font-size:12px;padding:7px 2px">{html.escape(empty)}</div>'
    t = df[[c for c in cols if c in df.columns]].copy()
    if rename:
        t = t.rename(columns=rename)
    for c in t.columns:
        if pd.api.types.is_numeric_dtype(t[c]):
            t[c] = pd.to_numeric(t[c], errors="coerce").round(digits)
    raw = t.to_html(index=False, border=0, classes="qr-table")
    return (
        '<div style="overflow-x:auto;border:1px solid #e5e7eb;border-radius:10px;padding:3px 7px;'
        'background:white;font-size:12px">' + raw + '</div>'
    )


def _candidate_row(snapshot):
    rg = snapshot["range44"]
    summary = rg.get("summary")
    if isinstance(summary, pd.DataFrame) and len(summary):
        return summary.iloc[0]
    return pd.Series(dtype=object)


def render_live_research_stack_html(show_rows=8):
    s = live_stack_snapshot()
    h = s["health"]
    m = s.get("metrics") or {}
    cf = s["counterfactual"]
    rg = s["range44"]
    rr = _candidate_row(s)

    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    all_ok = bool(s.get("critical_ok"))
    header_bg = "#ecfdf3" if all_ok else "#fff8e6"
    header_border = "#1a7f37" if all_ok else "#9a6700"

    range_pnl = _num(rr.get("strategy_realized_pnl"), 0.0)
    q3_same = _num(rr.get("q3_realized_pnl_same_sample"), 0.0)
    delta = _num(rr.get("pnl_change_vs_q3"), 0.0)
    range_dd = _num(rr.get("max_drawdown"), 0.0)
    q3_same_dd = _num(rr.get("q3_max_drawdown_same_sample"), 0.0)
    prospective = int(_num(rr.get("prospective_windows"), 0))
    eligible = int(_num(rr.get("eligible_windows"), 0))
    hb = int(_num(rr.get("high_breadth_windows"), 0))
    flagged = int(_num(rr.get("flagged_windows"), 0))
    settled_ct = _num(rr.get("settled_contracts"), 0.0)
    open_ct = _num(rr.get("open_contracts"), 0.0)

    validation_state = "COLLECTING OOS" if s.get("range44_match") else "MONITOR ATTENTION"
    validation_style = "ok" if s.get("range44_match") else "bad"

    parts = [
        '<style>'
        '.qr-table{border-collapse:collapse;width:100%;white-space:nowrap}'
        '.qr-table th{font-size:10px;text-transform:uppercase;letter-spacing:.035em;opacity:.68;text-align:left;padding:7px 8px;border-bottom:1px solid #e5e7eb}'
        '.qr-table td{padding:7px 8px;border-bottom:1px solid #f0f2f4}'
        '.qr-table tr:last-child td{border-bottom:none}'
        '</style>',
        '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1450px;color:#111827">',
        f'<div style="background:{header_bg};border:1px solid {header_border};border-radius:13px;padding:13px 16px;margin-bottom:11px">',
        f'<div style="font-size:21px;font-weight:850">KALSHI 15-MIN RESEARCH DASHBOARD · {"ALL SYSTEMS GO" if all_ok else "CHECK REQUIRED"}</div>',
        f'<div style="font-size:11px;opacity:.68;margin-top:3px">{html.escape(now)} · {DASHBOARD_VERSION} · simulated/read-only execution</div>',
        f'<div style="font-size:10px;opacity:.58;margin-top:2px">Session: {html.escape(str(s.get("recorder_session")))}</div>',
        '</div>',

        # Candidate hero first.
        '<div style="border:2px solid #0969da;background:linear-gradient(135deg,#eef6ff,#f8fbff);border-radius:15px;padding:15px 16px;margin-bottom:13px">',
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap">',
        '<div>',
        '<div style="font-size:11px;font-weight:800;color:#0969da;letter-spacing:.08em;text-transform:uppercase">Current lead candidate</div>',
        '<div style="font-size:27px;font-weight:900;margin-top:2px">RANGE44_Q1</div>',
        f'<div style="font-size:13px;margin-top:4px"><b>Q3 normally</b> · if M5 breadth ≥3 and max M1→M5 midpoint range ≥ <b>{RANGE_THRESHOLD_C:.0f}¢</b>, size down to <b>Q1</b>.</div>',
        '<div style="font-size:11px;opacity:.66;margin-top:5px">Goal: keep primary upside while reducing exposure in the high-breadth windows that showed the worst loss concentration.</div>',
        '</div>',
        '<div style="text-align:right">',
        _pill("DEVELOPMENT-SELECTED", "info"),
        _pill(validation_state, validation_style),
        '<div style="font-size:10px;opacity:.58;margin-top:5px">Not yet declared validated; only post-freeze windows count.</div>',
        '</div></div>',

        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:13px">',
        _metric("Range44 PnL", _money(range_pnl), "prospective only", "ok" if range_pnl >= 0 else "warn", large=True),
        _metric("Q3 same sample", _money(q3_same), "exact same prospective windows", "neutral", large=True),
        _metric("Δ vs Q3", _money(delta), "candidate lift", "ok" if delta >= 0 else "warn", large=True),
        _metric("Range44 max DD", _money(range_dd), "complete prospective windows", "ok" if range_dd >= q3_same_dd else "warn", large=True),
        _metric("Q3 same-sample DD", _money(q3_same_dd), "benchmark DD", "neutral", large=True),
        '</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">',
        _metric("Prospective windows", prospective, "post-freeze", "info"),
        _metric("Eligible", eligible, "usable path data", "info"),
        _metric("High breadth", hb, "signals ≥3", "info"),
        _metric("Range44 flagged", flagged, "Q1 windows", "warn"),
        _metric("Settled / open", f"{settled_ct:.2f} / {open_ct:.2f}", "contracts", "neutral"),
        '</div>',
        '<div style="margin-top:11px;padding-top:9px;border-top:1px solid #bfdbfe;font-size:11px">',
        '<b>Why this is the lead candidate:</b> '
        f'range-only had {DEV_RANGE_ROBUST_CELLS}/{DEV_RANGE_GRID_CELLS} robust grid cells; '
        f'the robust medoid at {DEV_MEDOID_THRESHOLD_C:.2f}¢ improved development PnL by {_money(DEV_MEDOID_DELTA_VS_Q3)}; '
        f'worst leave-one-window-out improvement {_money(DEV_MEDOID_LOO_MIN_DELTA)}; '
        f'worst session improvement {_money(DEV_MEDOID_WORST_SESSION_DELTA)}; '
        f'adding path or RV contributed {_money(DEV_PAIR_INCREMENTAL_DELTA)} incremental PnL. '
        f'The frozen threshold was rounded to {RANGE_THRESHOLD_C:.0f}¢ inside that stable region.',
        '</div>',
        f'<div style="font-size:10px;opacity:.55;margin-top:5px">Frozen prospective line: {html.escape(str(rg.get("frozen_at_utc")))}</div>',
        '</div>',
    ]

    # Live markets directly after the candidate.
    parts.append(_section("LIVE MARKETS NOW", "Current frozen-universe BBO, M5 countdown, BTC opposition state, shadow state, and fills."))
    parts.append(_table(
        s.get("contracts"),
        ["series", "to_M5", "to_close", "YES_bid", "YES_ask", "mid_c", "spread_c", "lean", "BTC15_bp", "opp?", "shadow_state", "position", "fill_qty", "realized_pnl"],
        {"realized_pnl": "PnL"},
        digits=3,
        empty="No active frozen-universe markets in recorder state.",
    ))

    # Compact health strip.
    parts.append(_section("SYSTEM HEALTH", "Everything required for a clean prospective session."))
    parts.append('<div style="display:flex;gap:7px;flex-wrap:wrap">')
    parts.extend([
        _metric("Recorder", "HEALTHY" if h.get("healthy") else "ATTENTION", f'epoch {h.get("connection_epoch")} · active {h.get("active_markets")}', "ok" if h.get("healthy") else "bad"),
        _metric("Frozen Q3", "RUNNING" if s.get("primary_running") else "STOPPED", "benchmark unchanged", "ok" if s.get("primary_running") else "bad"),
        _metric("Session pair", "MATCH" if s.get("pair_match") else "MISMATCH", "recorder ↔ primary", "ok" if s.get("pair_match") else "bad"),
        _metric("BTC opposition", "OK" if s.get("btc_ok") else "ATTENTION", "Coinbase + parity gate", "ok" if s.get("btc_ok") else "bad"),
        _metric("Counterfactuals", "RUNNING" if s.get("counterfactual_match") else "ATTENTION", "same session", "ok" if s.get("counterfactual_match") else "bad"),
        _metric("Range44", "RUNNING" if s.get("range44_match") else "ATTENTION", "same session", "ok" if s.get("range44_match") else "bad"),
        _metric("Legacy 2/4", "STOPPED" if not s["legacy_strategy"].get("running") else "RUNNING", "must stay stopped", "ok" if not s["legacy_strategy"].get("running") else "bad"),
    ])
    parts.append('</div>')

    if not all_ok:
        issues = []
        if not h.get("healthy"):
            issues.append("recorder unhealthy")
        if not s.get("primary_running"):
            issues.append("primary stopped")
        if not s.get("primary_threads_ok"):
            issues.append("primary thread failure")
        if not s.get("pair_match"):
            issues.append("recorder/primary mismatch")
        if not s.get("btc_ok"):
            issues.append("BTC feed/parity attention")
        if not s.get("counterfactual_match"):
            issues.append("counterfactual monitor mismatch/stopped")
        if not s.get("range44_match"):
            issues.append("Range44 monitor mismatch/stopped")
        if s["legacy_strategy"].get("running"):
            issues.append("legacy 2/4 unexpectedly running")
        parts.append('<div style="margin-top:8px;background:#fff1f0;border:1px solid #cf222e;border-radius:9px;padding:9px 11px;font-size:12px"><b>Attention:</b> ' + html.escape('; '.join(issues)) + '</div>')

    # Benchmark performance.
    parts.append(_section("Q3 BENCHMARK — FROZEN PRIMARY", "This remains the untouched execution benchmark. Range44 is a read-only sizing overlay."))
    parts.append('<div style="display:flex;gap:8px;flex-wrap:wrap">')
    parts.extend([
        _metric("Signals", m.get("signals", 0), f'{m.get("windows", 0)} M5 windows', "info"),
        _metric("Filled assets", m.get("fills", 0), _pct(m.get("fill_rate")), "info"),
        _metric("Realized PnL", _money(m.get("pnl", 0.0)), "all Q3 session fills", "ok" if _num(m.get("pnl"), 0) >= 0 else "warn"),
        _metric("Max DD", _money(-abs(_num(m.get("max_dd"), 0))), "Q3 session", "neutral"),
        _metric("W / L", f'{m.get("wins", 0)} / {m.get("losses", 0)}', _pct(m.get("filled_accuracy")), "neutral"),
        _metric("Open", f'{_num(m.get("open_contracts"), 0):.2f} ct', f'{m.get("open_positions", 0)} assets', "neutral"),
        _metric("DATA_INVALID", m.get("data_invalid", 0), "causal/data-health rejects", "ok" if int(_num(m.get("data_invalid"), 0)) == 0 else "warn"),
    ])
    parts.append('</div>')

    # Other controls explicitly demoted.
    parts.append(_section("OTHER COUNTERFACTUAL CONTROLS", "Research controls only. These are not the currently selected lead candidate."))
    parts.append(_table(
        cf.get("summary"),
        ["scenario", "filled_contracts", "settled_filled_assets", "unsettled_filled_assets", "realized_pnl", "max_drawdown", "realized_pnl_change_vs_baseline"],
        {
            "filled_contracts": "contracts",
            "settled_filled_assets": "settled assets",
            "unsettled_filled_assets": "open assets",
            "realized_pnl": "PnL",
            "max_drawdown": "max DD",
            "realized_pnl_change_vs_baseline": "Δ vs Q3",
        },
    ))

    # Candidate history before primary raw fills.
    parts.append(_section("LATEST RANGE44 PROSPECTIVE WINDOWS", "Only windows after the frozen prospective timestamp appear here."))
    parts.append(_table(
        rg.get("latest_windows"),
        ["decision_time", "signals", "max_mid_range_c", "path_complete_share_pct", "range_flagged", "strategy_decision", "strategy_filled_contracts", "strategy_open_contracts", "strategy_realized_pnl", "q3_realized_pnl_same_sample", "pnl_change_vs_q3"],
        {
            "decision_time": "M5",
            "max_mid_range_c": "max range c",
            "path_complete_share_pct": "path %",
            "range_flagged": "flagged",
            "strategy_decision": "decision",
            "strategy_filled_contracts": "Range44 ct",
            "strategy_open_contracts": "open ct",
            "strategy_realized_pnl": "Range44 PnL",
            "q3_realized_pnl_same_sample": "Q3 PnL",
            "pnl_change_vs_q3": "Δ",
        },
        empty="No post-freeze Range44 window yet.",
    ))

    parts.append(_section("LATEST PRIMARY FILLS / NO-FILLS", "Actual frozen-Q3 shadow decisions. fill qty >0 = simulated passive execution; NO_FILL = no execution in 15s."))
    pdf = s.get("primary_df")
    if isinstance(pdf, pd.DataFrame) and len(pdf):
        z = pdf.sort_values("decision_time", na_position="last").tail(int(show_rows))
        parts.append(_table(
            z,
            ["decision_time", "ticker", "direction", "entry_price", "entry_queue", "entry_fill_qty", "result", "realized_pnl", "status", "quote_age_s", "book_age_s"],
            {
                "decision_time": "M5",
                "entry_price": "entry",
                "entry_queue": "queue",
                "entry_fill_qty": "fill qty",
                "realized_pnl": "PnL",
            },
            digits=4,
        ))
    else:
        parts.append('<div style="opacity:.58;font-size:12px">No primary decisions yet.</div>')

    # Diagnostics last.
    parts.append(_section("DIAGNOSTICS", "Low-level recorder, thread, session, and intentionally stopped legacy states."))
    parts += [
        '<div style="display:flex;gap:28px;flex-wrap:wrap;background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;font-size:11px">',
        '<div><b>Recorder</b><br>'
        f'running={html.escape(str(h.get("running")))} · healthy={html.escape(str(h.get("healthy")))}<br>'
        f'epoch={html.escape(str(h.get("connection_epoch")))} · active={html.escape(str(h.get("active_markets")))} · expired={html.escape(str(h.get("expired_active_markets")))}<br>'
        f'supervisor age={html.escape(str(h.get("supervisor_age_s")))}s · data age={html.escape(str(h.get("market_data_age_s")))}s<br>'
        f'last scan error={html.escape(str(h.get("last_scan_error")))}</div>',
        '<div><b>Primary threads</b><br>' + ''.join(f'{html.escape(str(k))}: {"OK" if v else "DEAD"}<br>' for k, v in s.get("thread_state", {}).items()) + '</div>',
        '<div><b>Session alignment</b><br>'
        f'primary: {html.escape(str(s.get("primary_session")))}<br>'
        f'counterfactual: {html.escape(str(cf.get("session_dir")))}<br>'
        f'Range44: {html.escape(str(rg.get("session_dir")))}</div>',
        '<div><b>Legacy state</b><br>'
        f'4-feature collector: {"RUNNING" if s["legacy_collector"].get("running") else "STOPPED"}<br>'
        f'2/4-vote strategy: {"RUNNING" if s["legacy_strategy"].get("running") else "STOPPED"}<br>'
        'Range-only development: COMPLETE</div>',
        '</div>',
    ]

    errors = []
    if cf.get("last_error"):
        errors.append("counterfactual: " + str(cf.get("last_error")))
    if rg.get("last_error"):
        errors.append("Range44: " + str(rg.get("last_error")))
    if s["legacy_collector"].get("last_error"):
        errors.append("legacy collector: " + str(s["legacy_collector"].get("last_error")))
    if errors:
        parts.append('<div style="margin-top:9px;border:1px solid #cf222e;background:#fff1f0;border-radius:8px;padding:8px;font-size:11px"><b>Monitor errors</b><br>' + '<br>'.join(html.escape(x) for x in errors) + '</div>')

    parts.append('</div>')
    return ''.join(parts)


async def watch_live_research_stack(refresh_seconds=2.0, show_rows=8):
    refresh_seconds = max(0.5, float(refresh_seconds))
    try:
        from IPython.display import HTML, display
    except Exception as exc:
        raise RuntimeError("Live research dashboard requires Jupyter/IPython display support.") from exc

    handle = display(HTML(render_live_research_stack_html(show_rows=show_rows)), display_id=True)
    try:
        while True:
            await asyncio.sleep(refresh_seconds)
            handle.update(HTML(render_live_research_stack_html(show_rows=show_rows)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        handle.update(HTML(render_live_research_stack_html(show_rows=show_rows)))
        print("Dashboard refresh stopped. Recorder, frozen Q3 primary, and read-only monitors remain running.")


async def start_live_research_stack(
    refresh_seconds=2.0,
    show_rows=8,
    duration_minutes=None,
    key_id=None,
    private_key_path=None,
):
    check = await prepare_live_research_stack(
        duration_minutes=duration_minutes,
        key_id=key_id,
        private_key_path=private_key_path,
    )
    print("LIVE STACK READY")
    print("Session:", check["recorder_session"])
    print("LEAD CANDIDATE: RANGE44_Q1 — development-selected, prospective validation running")
    print("Frozen Q3 primary: RUNNING")
    print("Counterfactual controls: RUNNING / SAME SESSION")
    print(f"Range44 prospective: RUNNING / SAME SESSION / frozen={check['range44'].get('frozen_at_utc')}")
    print("Legacy 2/4 strategy: STOPPED")
    await watch_live_research_stack(refresh_seconds=refresh_seconds, show_rows=show_rows)


__all__ = [
    "render_live_research_stack_html",
    "watch_live_research_stack",
    "start_live_research_stack",
    "prepare_live_research_stack",
    "stop_live_research_stack",
    "live_stack_snapshot",
]
