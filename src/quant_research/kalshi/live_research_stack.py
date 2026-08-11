from __future__ import annotations

import asyncio
import html
from pathlib import Path

import numpy as np
import pandas as pd

from . import primary_shadow_trader as P
from . import recorder as R
from . import shadow_dashboard as D
from . import runtime_api as API
from .pre_m5_range44_strategy import (
    RANGE_THRESHOLD_C,
    range44_prospective_status,
    start_range44_prospective_monitor,
    stop_range44_prospective_monitor,
)
from .risk_control_counterfactual import (
    counterfactual_risk_status,
    start_counterfactual_risk_monitor,
    stop_counterfactual_risk_monitor,
)

STACK_VERSION = "KALSHI_LIVE_RESEARCH_STACK_V1"


def _same_session(a, b):
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


def _money(x):
    try:
        return f"${float(x):+.4f}"
    except Exception:
        return "n/a"


def _pct(x):
    try:
        return f"{float(x):.1f}%" if np.isfinite(float(x)) else "n/a"
    except Exception:
        return "n/a"


def _card(label, value, sub="", state="neutral"):
    palette = {
        "ok": ("#eaf7ee", "#238636"),
        "warn": ("#fff7e6", "#9a6700"),
        "bad": ("#ffebe9", "#cf222e"),
        "neutral": ("#f6f8fa", "#57606a"),
        "info": ("#eef5ff", "#0969da"),
    }
    bg, border = palette.get(state, palette["neutral"])
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:11px 13px;min-width:150px;flex:1;">'
        f'<div style="font-size:11px;opacity:.72;text-transform:uppercase;letter-spacing:.04em">{html.escape(str(label))}</div>'
        f'<div style="font-size:21px;font-weight:700;margin-top:2px">{html.escape(str(value))}</div>'
        + (f'<div style="font-size:11px;opacity:.7;margin-top:2px">{html.escape(str(sub))}</div>' if sub else "")
        + '</div>'
    )


def _table(df, cols, rename=None, digits=4):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return '<div style="opacity:.6;font-size:12px">No rows yet.</div>'
    t = df[[c for c in cols if c in df.columns]].copy()
    if rename:
        t = t.rename(columns=rename)
    for c in t.columns:
        if pd.api.types.is_numeric_dtype(t[c]):
            t[c] = pd.to_numeric(t[c], errors="coerce").round(digits)
    return '<div style="overflow-x:auto;font-size:12px">' + t.to_html(index=False, border=0) + '</div>'


def _legacy_status():
    collector = {"running": False, "session_dir": None, "last_error": None}
    strategy = {"running": False, "session_dir": None, "last_error": None}
    try:
        from .pre_m5_prospective_monitor import pre_m5_prospective_risk_status
        collector = pre_m5_prospective_risk_status(show=False)
    except Exception as exc:
        collector = {"running": False, "session_dir": None, "last_error": repr(exc)}
    try:
        from .pre_m5_risk_strategy import pre_m5_risk_strategy_status
        strategy = pre_m5_risk_strategy_status(show=False)
    except Exception as exc:
        strategy = {"running": False, "session_dir": None, "last_error": repr(exc)}
    return collector, strategy


def live_stack_snapshot():
    snap = D.status_snapshot()
    health = R.recorder_health_snapshot()
    primary_session = None
    thread_state = {}
    metrics = {}
    contracts = pd.DataFrame()
    primary_df = pd.DataFrame()
    if snap is not None:
        primary_session = str(snap["sh"].session_dir)
        thread_state = snap.get("thread_state", {})
        metrics = snap.get("metrics", {})
        contracts = snap.get("contracts", pd.DataFrame())
        primary_df = snap.get("df", pd.DataFrame())

    cf = counterfactual_risk_status(show=False)
    rg = range44_prospective_status(show=False)
    legacy_collector, legacy_strategy = _legacy_status()

    recorder_session = health.get("session_dir")
    pair_match = bool(primary_session and recorder_session and _same_session(primary_session, recorder_session))
    cf_match = bool(cf.get("running") and _same_session(cf.get("session_dir"), recorder_session))
    range_match = bool(rg.get("running") and _same_session(rg.get("session_dir"), recorder_session))
    primary_running = snap is not None and P.primary_shadow_running()
    primary_threads_ok = bool(thread_state) and all(thread_state.values())
    btc_thread_ok = bool(thread_state.get("shadow-coinbase", False))
    btc_seeded = bool(snap is not None and len(getattr(snap["sh"], "btc", [])) > 0)
    btc_ok = primary_running and btc_thread_ok and btc_seeded

    critical_ok = (
        bool(health.get("running"))
        and bool(health.get("healthy"))
        and primary_running
        and primary_threads_ok
        and pair_match
        and btc_ok
        and cf_match
        and range_match
        and not bool(legacy_strategy.get("running"))
    )

    return {
        "primary": snap,
        "health": health,
        "primary_session": primary_session,
        "recorder_session": recorder_session,
        "pair_match": pair_match,
        "primary_running": primary_running,
        "primary_threads_ok": primary_threads_ok,
        "btc_ok": btc_ok,
        "counterfactual": cf,
        "counterfactual_match": cf_match,
        "range44": rg,
        "range44_match": range_match,
        "legacy_collector": legacy_collector,
        "legacy_strategy": legacy_strategy,
        "metrics": metrics,
        "contracts": contracts,
        "primary_df": primary_df,
        "thread_state": thread_state,
        "critical_ok": critical_ok,
    }


def render_live_research_stack_html(show_rows=8):
    s = live_stack_snapshot()
    h = s["health"]
    m = s["metrics"] or {}
    cf = s["counterfactual"]
    rg = s["range44"]
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    overall = "ALL SYSTEMS GO" if s["critical_ok"] else "CHECK REQUIRED"
    header_bg = "#eaf7ee" if s["critical_ok"] else "#fff7e6"
    header_border = "#238636" if s["critical_ok"] else "#9a6700"

    parts = [
        '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1400px">',
        f'<div style="background:{header_bg};border:1px solid {header_border};border-radius:12px;padding:14px 16px;margin-bottom:12px">',
        f'<div style="font-size:22px;font-weight:800">KALSHI 15-MIN LIVE RESEARCH STACK — {overall}</div>',
        f'<div style="font-size:12px;opacity:.72;margin-top:3px">{html.escape(now)} · {STACK_VERSION} · all execution is simulated/read-only</div>',
        f'<div style="font-size:11px;opacity:.68;margin-top:3px">Session: {html.escape(str(s["recorder_session"]))}</div>',
        '</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">',
        _card("Recorder", "HEALTHY" if h.get("healthy") else "ATTENTION", f'epoch {h.get("connection_epoch")} · active {h.get("active_markets")}', "ok" if h.get("healthy") else "bad"),
        _card("Frozen Q3 primary", "RUNNING" if s["primary_running"] else "STOPPED", "M5 · BTC opposition · spread≤2c · -3c · 15s · Q3", "ok" if s["primary_running"] else "bad"),
        _card("Session pair", "MATCH" if s["pair_match"] else "MISMATCH", "recorder ↔ primary", "ok" if s["pair_match"] else "bad"),
        _card("BTC opposition feed", "OK" if s["btc_ok"] else "ATTENTION", "startup parity gate + live Coinbase thread", "ok" if s["btc_ok"] else "bad"),
        _card("Counterfactuals", "RUNNING" if s["counterfactual_match"] else "ATTENTION", "Q3 / HB-Q2 / HB-Q1 / MAX2 / MAX3", "ok" if s["counterfactual_match"] else "bad"),
        _card("Range44 prospective", "RUNNING" if s["range44_match"] else "ATTENTION", "HB≥3 & max M1→M5 range≥44c → Q1", "ok" if s["range44_match"] else "bad"),
        _card("Legacy 2/4 strategy", "STOPPED" if not s["legacy_strategy"].get("running") else "RUNNING", "must remain stopped", "ok" if not s["legacy_strategy"].get("running") else "bad"),
        '</div>',
    ]

    if not s["critical_ok"]:
        issues = []
        if not h.get("healthy"):
            issues.append("recorder unhealthy")
        if not s["primary_running"]:
            issues.append("primary shadow stopped")
        if not s["primary_threads_ok"]:
            issues.append("one or more primary threads dead")
        if not s["pair_match"]:
            issues.append("recorder/primary session mismatch")
        if not s["btc_ok"]:
            issues.append("BTC feed/parity state not healthy")
        if not s["counterfactual_match"]:
            issues.append("counterfactual monitor stopped or on wrong session")
        if not s["range44_match"]:
            issues.append("Range44 monitor stopped or on wrong session")
        if s["legacy_strategy"].get("running"):
            issues.append("legacy 2/4 strategy unexpectedly running")
        parts.append('<div style="border:1px solid #cf222e;background:#ffebe9;border-radius:10px;padding:10px 12px;margin-bottom:12px"><b>Attention:</b> ' + html.escape("; ".join(issues)) + '</div>')

    parts += [
        '<h3 style="margin:14px 0 7px">Primary Q3 performance</h3>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">',
        _card("Signals", m.get("signals", 0), f'{m.get("windows", 0)} M5 windows', "info"),
        _card("Filled assets", m.get("fills", 0), _pct(m.get("fill_rate")), "info"),
        _card("Realized PnL", _money(m.get("pnl", 0.0)), "frozen primary", "ok" if float(m.get("pnl", 0.0) or 0.0) >= 0 else "warn"),
        _card("Max drawdown", f'${float(m.get("max_dd", 0.0) or 0.0):.4f}', "primary Q3", "neutral"),
        _card("W / L", f'{m.get("wins", 0)} / {m.get("losses", 0)}', _pct(m.get("filled_accuracy")), "neutral"),
        _card("Open", f'{float(m.get("open_contracts", 0.0) or 0.0):.2f} ct', f'{m.get("open_positions", 0)} assets', "neutral"),
        _card("DATA_INVALID", m.get("data_invalid", 0), "should stay low", "ok" if int(m.get("data_invalid", 0) or 0) == 0 else "warn"),
        '</div>',
    ]

    rsum = rg.get("summary")
    rrow = rsum.iloc[0] if isinstance(rsum, pd.DataFrame) and len(rsum) else pd.Series(dtype=object)
    parts += [
        '<div style="border:1px solid #0969da;background:#eef5ff;border-radius:12px;padding:12px 14px;margin-bottom:12px">',
        '<div style="font-size:17px;font-weight:750">Frozen prospective candidate — RANGE44_Q1</div>',
        f'<div style="font-size:12px;opacity:.72;margin:3px 0 8px">Rule: signals ≥3 and max M1→M5 midpoint range ≥ {RANGE_THRESHOLD_C:.0f}¢ → Q1; otherwise Q3. Development complete; only post-freeze windows count.</div>',
        f'<div style="font-size:11px;opacity:.65;margin-bottom:8px">Frozen at: {html.escape(str(rg.get("frozen_at_utc")))}</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap">',
        _card("Prospective windows", int(rrow.get("prospective_windows", 0) or 0), "post-freeze", "info"),
        _card("Eligible", int(rrow.get("eligible_windows", 0) or 0), "usable path data", "info"),
        _card("High breadth", int(rrow.get("high_breadth_windows", 0) or 0), "signals ≥3", "info"),
        _card("Range44 flagged", int(rrow.get("flagged_windows", 0) or 0), "sized Q1", "warn"),
        _card("Range44 PnL", _money(rrow.get("strategy_realized_pnl", 0.0)), "prospective only", "ok" if float(rrow.get("strategy_realized_pnl", 0.0) or 0.0) >= 0 else "warn"),
        _card("Q3 same sample", _money(rrow.get("q3_realized_pnl_same_sample", 0.0)), "exact same windows", "neutral"),
        _card("Δ vs Q3", _money(rrow.get("pnl_change_vs_q3", 0.0)), "primary comparison", "ok" if float(rrow.get("pnl_change_vs_q3", 0.0) or 0.0) >= 0 else "warn"),
        _card("Range44 max DD", _money(rrow.get("max_drawdown", 0.0)), "complete windows", "neutral"),
        '</div></div>',
    ]

    parts.append('<h3 style="margin:14px 0 6px">Counterfactual sizing controls</h3>')
    csum = cf.get("summary")
    parts.append(_table(
        csum,
        ["scenario", "filled_contracts", "settled_filled_assets", "unsettled_filled_assets", "realized_pnl", "max_drawdown", "realized_pnl_change_vs_baseline"],
        {
            "filled_contracts": "contracts", "settled_filled_assets": "settled assets",
            "unsettled_filled_assets": "open assets", "realized_pnl": "PnL",
            "max_drawdown": "max DD", "realized_pnl_change_vs_baseline": "Δ vs Q3",
        },
    ))

    parts.append('<h3 style="margin:14px 0 6px">Active frozen-universe markets</h3>')
    parts.append(_table(
        s["contracts"],
        ["series", "to_M5", "to_close", "YES_bid", "YES_ask", "mid_c", "spread_c", "lean", "BTC15_bp", "opp?", "shadow_state", "position", "fill_qty", "realized_pnl"],
        {"realized_pnl": "PnL"},
        digits=3,
    ))

    latest_range = rg.get("latest_windows")
    parts.append('<h3 style="margin:14px 0 6px">Latest RANGE44 prospective windows</h3>')
    parts.append(_table(
        latest_range,
        ["decision_time", "signals", "max_mid_range_c", "path_complete_share_pct", "range_flagged", "strategy_decision", "strategy_filled_contracts", "strategy_open_contracts", "strategy_realized_pnl", "q3_realized_pnl_same_sample", "pnl_change_vs_q3"],
        {
            "decision_time": "M5", "max_mid_range_c": "max range c", "path_complete_share_pct": "path %",
            "range_flagged": "flagged", "strategy_decision": "decision",
            "strategy_filled_contracts": "Range44 ct", "strategy_open_contracts": "open ct",
            "strategy_realized_pnl": "Range44 PnL", "q3_realized_pnl_same_sample": "Q3 PnL", "pnl_change_vs_q3": "Δ",
        },
    ))

    pdf = s["primary_df"]
    parts.append('<h3 style="margin:14px 0 6px">Latest primary decisions / fills</h3>')
    if isinstance(pdf, pd.DataFrame) and len(pdf):
        z = pdf.sort_values("decision_time", na_position="last").tail(int(show_rows))
        parts.append(_table(
            z,
            ["decision_time", "ticker", "direction", "entry_price", "entry_queue", "entry_fill_qty", "result", "realized_pnl", "status", "quote_age_s", "book_age_s"],
            {"decision_time": "M5", "entry_price": "entry", "entry_queue": "queue", "entry_fill_qty": "fill qty", "realized_pnl": "PnL"},
            digits=4,
        ))
    else:
        parts.append('<div style="opacity:.6;font-size:12px">No primary decisions yet.</div>')

    parts += [
        '<h3 style="margin:14px 0 6px">Data / thread health</h3>',
        '<div style="display:flex;gap:24px;flex-wrap:wrap;font-size:12px">',
        '<div><b>Recorder</b><br>'
        f'running={html.escape(str(h.get("running")))} · healthy={html.escape(str(h.get("healthy")))}<br>'
        f'epoch={html.escape(str(h.get("connection_epoch")))} · active={html.escape(str(h.get("active_markets")))} · expired={html.escape(str(h.get("expired_active_markets")))}<br>'
        f'supervisor age={html.escape(str(h.get("supervisor_age_s")))}s · market-data age={html.escape(str(h.get("market_data_age_s")))}s<br>'
        f'last scan error={html.escape(str(h.get("last_scan_error")))}</div>',
        '<div><b>Primary threads</b><br>' + ''.join(f'{html.escape(str(k))}: {"OK" if v else "DEAD"}<br>' for k, v in s["thread_state"].items()) + '</div>',
        '<div><b>Monitor sessions</b><br>'
        f'primary: {html.escape(str(s["primary_session"]))}<br>'
        f'counterfactual: {html.escape(str(cf.get("session_dir")))}<br>'
        f'Range44: {html.escape(str(rg.get("session_dir")))}</div>',
        '<div><b>Intentional legacy state</b><br>'
        f'4-feature collector: {"RUNNING" if s["legacy_collector"].get("running") else "STOPPED"}<br>'
        f'2/4-vote strategy: {"RUNNING" if s["legacy_strategy"].get("running") else "STOPPED"}<br>'
        'Range-only candidate: DEVELOPMENT COMPLETE</div>',
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
        parts.append('<div style="margin-top:10px;border:1px solid #cf222e;background:#ffebe9;border-radius:8px;padding:8px;font-size:12px"><b>Monitor errors</b><br>' + '<br>'.join(html.escape(x) for x in errors) + '</div>')

    parts.append('</div>')
    return ''.join(parts)


def _stop_legacy_monitors():
    try:
        from .pre_m5_risk_strategy import pre_m5_risk_strategy_status, stop_pre_m5_risk_strategy_monitor
        if pre_m5_risk_strategy_status(show=False).get("running"):
            stop_pre_m5_risk_strategy_monitor(show=False)
    except Exception:
        pass
    try:
        from .pre_m5_prospective_monitor import pre_m5_prospective_risk_status, stop_pre_m5_prospective_risk_monitor
        if pre_m5_prospective_risk_status(show=False).get("running"):
            stop_pre_m5_prospective_risk_monitor(show=False)
    except Exception:
        pass


def _attach_counterfactual(session_dir):
    cf = counterfactual_risk_status(show=False)
    if cf.get("running") and not _same_session(cf.get("session_dir"), session_dir):
        stop_counterfactual_risk_monitor(show=False)
        cf = {"running": False}
    if not cf.get("running"):
        # The primary producer opens this file append-only. Creating the empty log here
        # lets the read-only reconstruction monitor start before the first M5 event.
        sh = P._SHADOW
        if sh is None:
            raise RuntimeError("Primary shadow is not initialized.")
        sh.event_file.parent.mkdir(parents=True, exist_ok=True)
        sh.event_file.touch(exist_ok=True)
        start_counterfactual_risk_monitor(session_dir=session_dir, interval_sec=5.0, show=False)


def _attach_range44(session_dir):
    st = range44_prospective_status(show=False)
    if st.get("running") and not _same_session(st.get("session_dir"), session_dir):
        stop_range44_prospective_monitor(show=False)
        st = {"running": False}
    if not st.get("running"):
        start_range44_prospective_monitor(session_dir=session_dir, interval_sec=10.0, show=False)


async def prepare_live_research_stack(duration_minutes=None, key_id=None, private_key_path=None):
    """Idempotently prepare one clean recorder/primary/research-monitor stack.

    If a healthy recorder is already running in this kernel, adopt it. Otherwise start a
    fresh session. Never starts a second recorder beside an existing one.
    """
    health = R.recorder_health_snapshot()
    if health.get("running"):
        if not health.get("healthy"):
            # Give a transient reconnect a short chance to recover before refusing.
            for _ in range(30):
                await asyncio.sleep(0.5)
                health = R.recorder_health_snapshot()
                if health.get("healthy"):
                    break
            if not health.get("healthy"):
                raise RuntimeError(f"Recorder is already running but unhealthy; refusing to layer a live stack on it: {health}")
        session_dir = Path(health["session_dir"])
    else:
        await API.start_recorder(duration_minutes=duration_minutes, key_id=key_id, private_key_path=private_key_path)
        health = R.recorder_health_snapshot()
        if not health.get("healthy"):
            raise RuntimeError(f"Recorder failed live-stack health gate: {health}")
        session_dir = Path(health["session_dir"])

    if P.primary_shadow_running():
        if P._SHADOW is None or P._SHADOW.session_dir.resolve() != session_dir.resolve():
            raise RuntimeError(f"Primary shadow is already running on a different session: {getattr(P._SHADOW, 'session_dir', None)}")
    else:
        API.start_primary_shadow_trader(session_dir=session_dir)

    # Superseded exploratory M1->M5 monitors stay off by design.
    _stop_legacy_monitors()
    _attach_counterfactual(session_dir)
    _attach_range44(session_dir)

    check = live_stack_snapshot()
    if not check["critical_ok"]:
        raise RuntimeError("Live stack started but failed the final consistency gate. Inspect live_stack_snapshot().")
    return check


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
    """One-cell entry point: prepare the stack, pass all gates, then watch dashboard."""
    check = await prepare_live_research_stack(
        duration_minutes=duration_minutes,
        key_id=key_id,
        private_key_path=private_key_path,
    )
    print("LIVE STACK READY")
    print("Session:", check["recorder_session"])
    print("Frozen Q3 primary: RUNNING")
    print("Counterfactual controls: RUNNING / SAME SESSION")
    print(f"Range44 prospective: RUNNING / SAME SESSION / frozen={check['range44'].get('frozen_at_utc')}")
    print("Legacy 2/4 strategy: STOPPED")
    await watch_live_research_stack(refresh_seconds=refresh_seconds, show_rows=show_rows)


async def stop_live_research_stack():
    """Orderly cleanup: research monitors -> primary -> recorder."""
    _stop_legacy_monitors()
    try:
        stop_range44_prospective_monitor(show=False)
    except Exception:
        pass
    try:
        stop_counterfactual_risk_monitor(show=False)
    except Exception:
        pass
    if P.primary_shadow_running():
        API.stop_primary_shadow_trader()
    try:
        return await API.stop_recorder()
    except asyncio.CancelledError:
        # websockets may surface cancellation during its close handshake even after the
        # recorder stop event has been honored. Treat it as clean only if it is stopped.
        health = R.recorder_health_snapshot()
        if not health.get("running"):
            print("Recorder stopped; websocket close handshake surfaced CancelledError after shutdown.")
            return R.last_session_dir()
        raise
