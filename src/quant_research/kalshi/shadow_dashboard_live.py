from __future__ import annotations

import html
import time

import numpy as np
import pandas as pd

from . import recorder as R
from . import shadow_dashboard as D
from . import shadow_notebook_parity as P
from . import shadow_trader as S


def _money(x):
    return f"${float(x):+.4f}"


def _pct(n, d):
    return "n/a" if not d else f"{100.0 * float(n) / float(d):.2f}%"


def _recorder_health():
    state = R._STATE or {}
    now_mono = time.monotonic()
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()

    supervisor_age = None
    if state.get("supervisor_heartbeat_mono") is not None:
        supervisor_age = now_mono - state["supervisor_heartbeat_mono"]

    market_data_age = None
    if state.get("last_market_data_mono") is not None:
        market_data_age = now_mono - state["last_market_data_mono"]

    expired = []
    for ticker in state.get("markets", set()):
        close_time = (state.get("meta", {}).get(ticker) or {}).get("close_time")
        if close_time is not None and close_time <= now:
            expired.append(ticker)

    running = R._TASK is not None and not R._TASK.done()
    channels = set(state.get("sids", {}))
    healthy = (
        running
        and not expired
        and (supervisor_age is None or supervisor_age < 25)
        and {"orderbook_delta", "trade", "ticker"}.issubset(channels)
    )

    return {
        "running": running,
        "healthy": healthy,
        "expired": expired,
        "supervisor_age": supervisor_age,
        "market_data_age": market_data_age,
        "active_markets": len(state.get("markets", set())),
        "books": len(state.get("books", {})),
        "last_scan_seconds": state.get("last_scan_seconds"),
        "last_scan_error": state.get("last_scan_error"),
    }


def _performance(sh):
    df = D._performance_frame(sh)
    if df.empty:
        return {
            "df": df,
            "signals": 0,
            "fills": 0,
            "fill_rate": "n/a",
            "settled": 0,
            "pnl": 0.0,
            "max_dd": 0.0,
            "wins": 0,
            "losses": 0,
            "filled_accuracy": "n/a",
            "open_positions": 0,
            "open_contracts": 0.0,
            "profit_factor": "n/a",
        }

    filled = df["entry_fill_qty"].fillna(0) > 1e-12
    settled = df["status"].eq("SETTLED")
    settled_filled = settled & filled
    known = df["result"].isin(["YES", "NO"])
    correct = known & df["direction"].eq(df["result"])

    trade_pnl = df.loc[settled_filled].sort_values("decision_time")["realized_pnl"].fillna(0).astype(float)
    if len(trade_pnl):
        equity = trade_pnl.cumsum().to_numpy(float)
        peaks = np.maximum.accumulate(np.r_[0.0, equity])[-len(equity):]
        max_dd = float((peaks - equity).max())
    else:
        max_dd = 0.0

    gross_profit = float(trade_pnl[trade_pnl > 0].sum())
    gross_loss = float(-trade_pnl[trade_pnl < 0].sum())
    if gross_loss > 0:
        pf = f"{gross_profit / gross_loss:.3f}"
    elif gross_profit > 0:
        pf = "inf"
    else:
        pf = "n/a"

    open_qty = (df["entry_fill_qty"].fillna(0) - df["exit_fill_qty"].fillna(0)).clip(lower=0)
    open_mask = filled & ~settled & (open_qty > 1e-12)

    settled_count = int(settled_filled.sum())
    wins = int((trade_pnl > 0).sum())
    losses = int((trade_pnl < 0).sum())

    return {
        "df": df,
        "signals": len(df),
        "fills": int(filled.sum()),
        "fill_rate": _pct(int(filled.sum()), len(df)),
        "settled": settled_count,
        "pnl": float(df["realized_pnl"].fillna(0).sum()),
        "max_dd": max_dd,
        "wins": wins,
        "losses": losses,
        "filled_accuracy": _pct(int((correct & settled_filled).sum()), settled_count),
        "open_positions": int(open_mask.sum()),
        "open_contracts": float(open_qty[open_mask].sum()),
        "profit_factor": pf,
    }


def _card(label, value, sub=""):
    return (
        '<div style="padding:10px 12px;border:1px solid #ddd;border-radius:8px;min-width:130px;">'
        f'<div style="font-size:11px;opacity:.65;">{html.escape(str(label))}</div>'
        f'<div style="font-size:20px;font-weight:600;">{html.escape(str(value))}</div>'
        + (f'<div style="font-size:10px;opacity:.6;">{html.escape(str(sub))}</div>' if sub else '')
        + '</div>'
    )


def render_primary_shadow_html(show_rows=8):
    sh = S._SHADOW
    if sh is None:
        return '<div><b>Primary shadow trader is not running.</b></div>'

    recorder_session = R.current_session_dir()
    shadow_session = sh.session_dir
    pair_ok = (
        recorder_session is not None
        and pd.Path(recorder_session).resolve() == pd.Path(shadow_session).resolve()
        if False else str(recorder_session) == str(shadow_session)
    )

    health = _recorder_health()
    events = D._read_events(sh.event_file)
    latest_events, skips, evaluated = D._event_maps(events)
    contracts = D._active_contract_table(sh, latest_events)
    perf = _performance(sh)

    now_text = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    healthy = health["healthy"] and pair_ok
    banner = "HEALTHY" if healthy else "ATTENTION"
    banner_bg = "#e9f7ef" if healthy else "#fff3cd"

    parts = [
        '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;">',
        f'<div style="padding:10px 12px;border-radius:8px;background:{banner_bg};margin-bottom:10px;">',
        f'<b>PRIMARY SHADOW — {banner}</b> &nbsp; <span style="opacity:.7">{html.escape(now_text)}</span><br>',
        f'<span style="font-size:12px">M5 | BTC opposition | spread ≤2c | -3c | 15s | 3ct | 10c salvage | {html.escape(P.PARITY_VERSION)}</span>',
        '</div>',
    ]

    if not pair_ok:
        parts.append(
            '<div style="padding:9px;border:1px solid #c00;border-radius:6px;margin-bottom:10px;">'
            f'<b>SESSION MISMATCH</b><br>Recorder: {html.escape(str(recorder_session))}<br>'
            f'Shadow: {html.escape(str(shadow_session))}</div>'
        )

    if health["expired"]:
        parts.append(
            '<div style="padding:9px;border:1px solid #c90;border-radius:6px;margin-bottom:10px;">'
            f'<b>Recorder rotation stale:</b> {len(health["expired"])} expired contract(s) still active. '
            'The watchdog should reconnect automatically.</div>'
        )

    parts.append('<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">')
    parts.append(_card("Signals", perf["signals"]))
    parts.append(_card("Any fills", perf["fills"], perf["fill_rate"]))
    parts.append(_card("Realized PnL", _money(perf["pnl"])))
    parts.append(_card("Max DD", f'${perf["max_dd"]:.4f}'))
    parts.append(_card("W / L", f'{perf["wins"]} / {perf["losses"]}', perf["filled_accuracy"]))
    parts.append(_card("Profit factor", perf["profit_factor"]))
    parts.append(_card("Open positions", perf["open_positions"], f'{perf["open_contracts"]:.2f} contracts'))
    parts.append(_card("Decisions", evaluated, f'{sum(skips.values())} skipped'))
    parts.append('</div>')

    parts.append('<h4 style="margin:8px 0 4px">Active frozen contracts</h4>')
    if contracts.empty:
        parts.append('<div style="opacity:.65">No active frozen-universe contracts currently in recorder state.</div>')
    else:
        cols = [
            "series", "to_M5", "to_close", "YES_bid", "YES_ask", "mid_c", "spread_c",
            "lean", "BTC15_bp", "opp?", "shadow_state", "position", "fill_qty", "realized_pnl",
            "quote_age_s", "ticker",
        ]
        table = contracts[cols].to_html(index=False, border=0, classes="shadow-contract-table")
        parts.append(
            '<div style="overflow-x:auto;font-size:12px;">'
            + table.replace('<table border="0" class="dataframe shadow-contract-table">',
                            '<table style="border-collapse:collapse;width:100%;" class="shadow-contract-table">')
            + '</div>'
        )

    parts.append('<div style="display:flex;gap:24px;flex-wrap:wrap;margin-top:12px;">')
    parts.append('<div><b>Recorder health</b><br>')
    parts.append(f'active={health["active_markets"]} &nbsp; books={health["books"]}<br>')
    parts.append(f'supervisor age={"n/a" if health["supervisor_age"] is None else f"{health["supervisor_age"]:.1f}s"}<br>')
    parts.append(f'market-data age={"n/a" if health["market_data_age"] is None else f"{health["market_data_age"]:.1f}s"}<br>')
    if health["last_scan_error"]:
        parts.append(f'last scan error={html.escape(str(health["last_scan_error"]))}<br>')
    parts.append('</div>')

    parts.append('<div><b>Decision skips</b><br>')
    if skips:
        for reason, count in skips.most_common():
            parts.append(f'{html.escape(str(reason))}: {count}<br>')
    else:
        parts.append('none yet<br>')
    parts.append('</div>')

    parts.append('<div><b>Thread health</b><br>')
    for thread in sh.threads:
        parts.append(f'{html.escape(thread.name)}: {"OK" if thread.is_alive() else "DEAD"}<br>')
    parts.append('</div></div>')

    if show_rows and not perf["df"].empty:
        latest = perf["df"].tail(int(show_rows))[
            ["decision_time", "ticker", "direction", "entry_price", "entry_queue", "entry_fill_qty", "result", "realized_pnl", "status"]
        ]
        parts.append('<h4 style="margin:12px 0 4px">Latest signals</h4>')
        parts.append('<div style="overflow-x:auto;font-size:12px;">' + latest.to_html(index=False, border=0) + '</div>')

    parts.append('</div>')
    return ''.join(parts)


def watch_primary_shadow_status(refresh_seconds=2.0, show_rows=8):
    """Update one Jupyter display in place. Never clears the whole cell output."""
    refresh_seconds = max(0.5, float(refresh_seconds))
    try:
        from IPython.display import HTML, display
    except Exception as exc:
        raise RuntimeError("Stable live dashboard requires IPython/Jupyter display support.") from exc

    handle = display(HTML(render_primary_shadow_html(show_rows=show_rows)), display_id=True)
    try:
        while True:
            handle.update(HTML(render_primary_shadow_html(show_rows=show_rows)))
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        handle.update(HTML(render_primary_shadow_html(show_rows=show_rows)))
        print("Dashboard refresh stopped. Recorder and shadow trader are still running.")


watch_shadow_status = watch_primary_shadow_status
