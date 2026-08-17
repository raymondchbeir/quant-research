from __future__ import annotations

"""Display-only enhanced dashboard for the frozen Candidate-C/Q10 OOS stack.

Safe to import while an existing V1 OOS session is running. This module does not
alter the recorder, shadow engine, strategy, queue state, fills, fees, or OOS
session. It only reads mm_cycle_q10_oos_stack_v1 live state and renders it.
"""

import asyncio
import html

import numpy as np
import pandas as pd

from . import mm_cycle_q10_oos_stack_v1 as B

DASHBOARD_VERSION = "MM_CYCLE_Q10_OOS_DASHBOARD_V3"

start_cycle_q10_oos_stack = B.start_cycle_q10_oos_stack
stop_cycle_q10_oos_stack = B.stop_cycle_q10_oos_stack
oos_stack_status = B.oos_stack_status
fee_preflight = B.fee_preflight


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _money(x, digits=2):
    z = _f(x)
    return f"${z:+,.{digits}f}" if np.isfinite(z) else "n/a"


def _pct(x):
    z = _f(x)
    return f"{z:.1f}%" if np.isfinite(z) else "n/a"


def _card(label, value, sub="", state="neutral"):
    pal = {
        "ok": ("#eaf7ee", "#238636"),
        "warn": ("#fff7e6", "#9a6700"),
        "bad": ("#ffebe9", "#cf222e"),
        "info": ("#eef5ff", "#0969da"),
        "neutral": ("#f6f8fa", "#57606a"),
    }
    bg, border = pal[state]
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
        'padding:10px 12px;min-width:145px;flex:1">'
        f'<div style="font-size:10px;opacity:.7;text-transform:uppercase">{html.escape(str(label))}</div>'
        f'<div style="font-size:20px;font-weight:750">{html.escape(str(value))}</div>'
        + (f'<div style="font-size:10px;opacity:.68">{html.escape(str(sub))}</div>' if sub else "")
        + "</div>"
    )


def live_market_table() -> pd.DataFrame:
    sh = B._SHADOW
    if sh is None:
        return pd.DataFrame()

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    frozen = set(B.SERIES)

    with sh.lock:
        tickers = sorted(set(sh.current) | set(sh.meta) | set(sh.quote) | set(sh.inventory))
        for ticker in tickers:
            series = sh.series_by_ticker.get(ticker, "")
            if series not in frozen:
                continue

            cur = sh.current.get(ticker)
            q = sh.quote.get(ticker)
            inv = float(sh.inventory.get(ticker, 0.0))
            meta = sh.meta.get(ticker, {}) or {}
            close = pd.to_datetime(
                sh.close_by_ticker.get(ticker) or meta.get("close_time"),
                utc=True,
                errors="coerce",
            )

            if pd.isna(close):
                elapsed = np.nan
                to_m5 = np.nan
                phase = "UNKNOWN"
            else:
                m0 = close - pd.Timedelta(seconds=900)
                elapsed = (now - m0).total_seconds()
                to_m5 = 300.0 - elapsed
                if elapsed < 0:
                    phase = "PRE-M0"
                elif elapsed < 300:
                    phase = "M0-M5"
                elif elapsed < 330:
                    phase = "LABEL TAIL"
                else:
                    phase = "DONE"

            if q is None and abs(inv) <= B.EPS:
                state = "FLAT"
            elif q is not None:
                state = f"{q['role']} {q['side']}"
            else:
                state = "INVENTORY"

            bid = ask = mid = spread = bid_l3 = ask_l3 = np.nan
            support = ""
            eligible = "NO"
            liq_px = np.nan
            open_exec = np.nan

            if cur is not None:
                bid = _f(cur.get("bid"))
                ask = _f(cur.get("ask"))
                mid = _f(cur.get("mid"))
                spread = _f(cur.get("spread_c"))
                bid_l3 = _f(cur.get("bid_depth3"), 0.0)
                ask_l3 = _f(cur.get("ask_depth3"), 0.0)

                if bid_l3 > ask_l3 + B.EPS:
                    support = "BID"
                elif ask_l3 > bid_l3 + B.EPS:
                    support = "ASK"
                else:
                    support = "TIE"

                eligible = "YES" if (
                    abs(inv) <= B.EPS
                    and np.isfinite(elapsed)
                    and 0.0 <= elapsed < 300.0
                    and spread + 1e-9 >= B.SPREAD_FLOOR_C
                    and support in {"BID", "ASK"}
                ) else "NO"

                liq_px = bid if inv > B.EPS else ask if inv < -B.EPS else np.nan
                if abs(inv) > B.EPS:
                    try:
                        gross, liq_qty, px = sh._liquidation_gross(ticker, cur)
                        mult = sh.fee_mult.get(series)
                        fee = B._quadratic_taker_fee(liq_qty, px, mult) if (
                            liq_qty > B.EPS and mult is not None and np.isfinite(px)
                        ) else 0.0
                        open_exec = gross - fee if np.isfinite(gross) else np.nan
                    except Exception:
                        open_exec = np.nan

            rows.append({
                "series": series,
                "ticker": ticker,
                "phase": phase,
                "elapsed_s": elapsed,
                "to_m5_s": to_m5,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_c": spread,
                "bid_l3": bid_l3,
                "ask_l3": ask_l3,
                "l3_support": support,
                "entry_eligible": eligible,
                "state": state,
                "inventory": inv,
                "quote_px": _f(q.get("price")) if q else np.nan,
                "quote_qty": _f(q.get("qty")) if q else np.nan,
                "queue_ahead": _f(q.get("queue_ahead")) if q else np.nan,
                "liq_px": liq_px,
                "open_exec_pnl_net": open_exec,
            })

    out = pd.DataFrame(rows)
    if len(out):
        rank = {"M0-M5": 0, "PRE-M0": 1, "LABEL TAIL": 2, "DONE": 3, "UNKNOWN": 4}
        out["_rank"] = out.phase.map(rank).fillna(9)
        out = out.sort_values(["_rank", "series", "ticker"]).drop(columns="_rank").reset_index(drop=True)
    return out


def _market_table_html(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return '<div style="opacity:.6;font-size:12px">No current market rows yet.</div>'

    z = df.copy()
    for c in ["bid", "ask", "mid", "quote_px", "liq_px"]:
        if c in z:
            z[c] = pd.to_numeric(z[c], errors="coerce").map(
                lambda x: f"{100*x:.1f}¢" if np.isfinite(x) else ""
            )
    if "spread_c" in z:
        z["spread_c"] = pd.to_numeric(z["spread_c"], errors="coerce").map(
            lambda x: f"{x:.1f}¢" if np.isfinite(x) else ""
        )
    for c in ["bid_l3", "ask_l3", "inventory", "quote_qty", "queue_ahead"]:
        if c in z:
            z[c] = pd.to_numeric(z[c], errors="coerce").map(
                lambda x: f"{x:.2f}" if np.isfinite(x) else ""
            )
    if "open_exec_pnl_net" in z:
        z["open_exec_pnl_net"] = pd.to_numeric(z["open_exec_pnl_net"], errors="coerce").map(
            lambda x: f"${x:+.3f}" if np.isfinite(x) else ""
        )
    z["elapsed"] = pd.to_numeric(z["elapsed_s"], errors="coerce").map(
        lambda x: f"M{x/60:.2f}" if np.isfinite(x) else ""
    )
    z["to_M5"] = pd.to_numeric(z["to_m5_s"], errors="coerce").map(
        lambda x: f"{max(0.0,x):.1f}s" if np.isfinite(x) else ""
    )

    cols = [
        "series", "ticker", "phase", "elapsed", "to_M5",
        "bid", "ask", "mid", "spread_c", "bid_l3", "ask_l3",
        "l3_support", "entry_eligible", "state", "inventory",
        "quote_px", "quote_qty", "queue_ahead", "liq_px", "open_exec_pnl_net",
    ]
    z = z[[c for c in cols if c in z.columns]].rename(columns={
        "series": "series", "ticker": "ticker", "phase": "phase",
        "elapsed": "elapsed", "to_M5": "to M5", "bid": "YES bid",
        "ask": "YES ask", "mid": "mid", "spread_c": "spread",
        "bid_l3": "bid L3", "ask_l3": "ask L3", "l3_support": "L3 support",
        "entry_eligible": "C eligible", "state": "shadow state",
        "inventory": "inv", "quote_px": "our quote", "quote_qty": "quote qty",
        "queue_ahead": "queue ahead", "liq_px": "cross px",
        "open_exec_pnl_net": "open exec PnL",
    })
    return '<div style="overflow-x:auto;font-size:11px">' + z.to_html(index=False, border=0) + '</div>'


def render_cycle_q10_oos_dashboard_html(show_rows=12):
    st = B.oos_stack_status(show=False)
    s = st.get("shadow") or {}
    h = st.get("health") or {}
    fee = st.get("fee_preflight") or {}

    recorder_ok = bool(st.get("running") and h.get("healthy"))
    shadow_ok = bool(s.get("thread_alive")) and not s.get("last_error")
    fees_ok = bool(fee.get("ok"))
    all_ok = recorder_ok and shadow_ok and fees_ok
    mature = bool(s.get("oos_mature"))

    markets = live_market_table()
    live = markets[markets.phase == "M0-M5"].copy() if len(markets) else pd.DataFrame()
    trading = markets[(markets.state != "FLAT") | (markets.inventory.abs() > B.EPS)].copy() if len(markets) else pd.DataFrame()
    eligible_n = int((live.entry_eligible == "YES").sum()) if len(live) else 0
    quoting_n = int(live.state.str.startswith(("ENTRY", "EXIT")).sum()) if len(live) else 0

    title_state = "OOS MATURE — KEEP FROZEN" if mature else "OOS RUNNING — NOT YET MATURE"
    bg = "#eaf7ee" if all_ok else "#ffebe9"
    border = "#238636" if all_ok else "#cf222e"

    parts = [
        '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1800px">',
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:14px 16px;margin-bottom:10px">',
        '<div style="font-size:22px;font-weight:800">FROZEN CYCLE_ALWAYS_EXIT Q10 — LIVE OOS SHADOW</div>',
        f'<div style="font-size:12px;opacity:.72">{title_state} · {DASHBOARD_VERSION} · {html.escape(str(s.get("time") or B._iso_ts()))}</div>',
        f'<div style="font-size:11px;opacity:.65">Session: {html.escape(str(st.get("session_dir")))}</div>',
        '</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">',
        _card("Recorder", "HEALTHY" if recorder_ok else "ATTENTION", f"epoch {h.get('connection_epoch')} · gaps {h.get('sequence_gaps')}", "ok" if recorder_ok else "bad"),
        _card("Shadow", "RUNNING" if shadow_ok else "ATTENTION", "read-only / no real orders", "ok" if shadow_ok else "bad"),
        _card("Fee preflight", "PASS" if fees_ok else "FAIL", "frozen fee assumptions", "ok" if fees_ok else "bad"),
        _card("OOS maturity", "MATURE" if mature else "COLLECTING", s.get("oos_maturity_rule", ""), "ok" if mature else "info"),
        _card("Runtime", f"{_f(s.get('runtime_hours'),0):.2f} h", f"{int(s.get('complete_9_series_windows',0) or 0)} complete 9-series windows", "info"),
        _card("Live M0-M5", len(live), f"{eligible_n} entry-eligible · {quoting_n} quoted", "info"),
        '</div>',
        '<h3 style="margin:12px 0 6px">Net economics</h3>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">',
        _card("Passive matched", _money(s.get("passive_matched_pnl")), "maker fills", "ok" if _f(s.get("passive_matched_pnl"),0) >= 0 else "warn"),
        _card("Forced liq gross", _money(s.get("forced_liq_gross_pnl")), "M5 residual only", "neutral"),
        _card("Taker fees", _money(-_f(s.get("taker_trade_fees"),0)), "forced M5 only", "warn"),
        _card("Realized net", _money(s.get("realized_net_trade_fee_only")), "trade-fee adjusted", "ok" if _f(s.get("realized_net_trade_fee_only"),0) >= 0 else "warn"),
        _card("Open executable", _money(s.get("current_residual_net")), "cross all inventory now", "neutral"),
        _card("Net executable", _money(s.get("net_executable_pnl_trade_fee_only")), "live OOS PnL", "ok" if _f(s.get("net_executable_pnl_trade_fee_only"),0) >= 0 else "warn"),
        _card("Sample-rate / day", _money(s.get("sample_rate_net_per_day")), "judge only after maturity", "info"),
        _card("Dev benchmark", _money(s.get("development_benchmark_per_day")), "fee-adjusted development", "neutral"),
        '</div>',
        '<h3 style="margin:12px 0 6px">Execution / inventory</h3>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">',
        _card("Fills", int(s.get("fill_events",0) or 0), f"{_f(s.get('fill_qty'),0):,.1f} contracts", "info"),
        _card("Cycles", f"{int(s.get('cycles_completed',0) or 0)} / {int(s.get('cycles_started',0) or 0)}", _pct(s.get("cycle_completion_pct")), "info"),
        _card("Forced M5", int(s.get("forced_liquidations",0) or 0), f"{_f(s.get('forced_liq_qty'),0):,.1f} contracts", "warn"),
        _card("Open inventory", f"{_f(s.get('open_abs_inventory'),0):,.2f} ct", f"{int(s.get('open_inventory_contracts',0) or 0)} markets", "neutral"),
        _card("5s markout", f"{_f(s.get('markout_5s_c')):+.3f}¢", "qty-weighted", "neutral"),
        _card("15s markout", f"{_f(s.get('markout_15s_c')):+.3f}¢", "qty-weighted", "neutral"),
        _card("30s markout", f"{_f(s.get('markout_30s_c')):+.3f}¢", "qty-weighted", "neutral"),
        '</div>',
    ]

    parts.append('<h3 style="margin:14px 0 4px">Current frozen-universe markets — where they are now</h3>')
    parts.append(
        '<div style="font-size:11px;opacity:.68;margin-bottom:6px">'
        'Every currently observed contract in the frozen nine-series universe. C eligible means the frozen Candidate-C entry condition is true right now while flat; this is display-only.'
        '</div>'
    )
    parts.append(_market_table_html(live if len(live) else markets.head(max(show_rows, 9))))

    parts.append('<h3 style="margin:14px 0 4px">Markets we are currently shadow-trading / carrying</h3>')
    if len(trading):
        parts.append(_market_table_html(trading))
    else:
        parts.append('<div style="opacity:.6;font-size:12px">No active entry/exit quote or inventory right now.</div>')

    parts.append(
        '<div style="border:1px solid #9a6700;background:#fff7e6;border-radius:10px;padding:9px 11px;font-size:11px;margin-top:10px">'
        '<b>OOS discipline:</b> this dashboard is display-only. It reads the frozen V1 shadow state and does not alter the recorder, strategy, fills, queue state, thresholds, size, or fee accounting.'
        '</div>'
    )

    if s.get("last_error"):
        parts.append(
            '<div style="border:1px solid #cf222e;background:#ffebe9;border-radius:8px;padding:8px;margin-top:10px">'
            f'<b>Shadow error:</b> {html.escape(str(s.get("last_error")))}</div>'
        )

    parts.append('</div>')
    return ''.join(parts)


async def watch_cycle_q10_oos_dashboard(refresh_seconds=2.0, show_rows=12):
    refresh_seconds = max(0.5, float(refresh_seconds))
    try:
        from IPython.display import HTML, display
    except Exception as exc:
        raise RuntimeError("Dashboard requires Jupyter/IPython display support.") from exc

    handle = display(HTML(render_cycle_q10_oos_dashboard_html(show_rows=show_rows)), display_id=True)
    try:
        while True:
            await asyncio.sleep(refresh_seconds)
            handle.update(HTML(render_cycle_q10_oos_dashboard_html(show_rows=show_rows)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        handle.update(HTML(render_cycle_q10_oos_dashboard_html(show_rows=show_rows)))
        print("Dashboard refresh stopped. Recorder + frozen shadow remain running unchanged.")
