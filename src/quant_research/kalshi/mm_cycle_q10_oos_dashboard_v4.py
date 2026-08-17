from __future__ import annotations

"""Disk-backed display-only dashboard for the frozen Candidate-C/Q10 OOS stack.

This fixes the V3 failure mode where the dashboard could lose the in-memory
shadow module instance after notebook reloads. Economics still come from the
frozen V1 shadow summary, but current market rows are reconstructed directly
from the authoritative raw OOS recorder files on disk.

No recorder, shadow state, strategy rule, quote, fill, threshold, size, or fee
calculation is modified by this module.
"""

import asyncio
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_cycle_q10_oos_stack_v1 as B

DASHBOARD_VERSION = "MM_CYCLE_Q10_OOS_DASHBOARD_V4_DISK_BACKED"
BOOK_TAIL_BYTES = 64 * 1024 * 1024
BOOK_TAIL_BYTES_MAX = 256 * 1024 * 1024

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


def _read_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _read_jsonl_tail(path: Path, max_bytes: int):
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open("rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        start = max(0, size - int(max_bytes))
        fh.seek(start)
        if start:
            fh.readline()
        raw = fh.read().decode("utf-8", errors="ignore")
    out = []
    for line in raw.splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _session_path():
    st = B.oos_stack_status(show=False)
    p = st.get("session_dir")
    return Path(p).resolve() if p else None


def _current_contract_meta(session: Path):
    now = pd.Timestamp.now(tz="UTC")
    rows = _read_jsonl(session / "market_metadata.jsonl")
    latest = {}
    for r in rows:
        ticker = str(r.get("ticker") or "")
        series = str(r.get("series_ticker") or "")
        if not ticker or series not in set(B.SERIES):
            continue
        close = pd.to_datetime(r.get("close_time"), utc=True, errors="coerce")
        if pd.isna(close):
            continue
        m0 = close - pd.Timedelta(seconds=900)
        if m0 - pd.Timedelta(seconds=300) <= now < close + pd.Timedelta(seconds=30):
            latest[ticker] = {**r, "_close": close, "_m0": m0}
    return latest


def _latest_book_by_ticker(session: Path, tickers):
    tickers = set(tickers)
    if not tickers:
        return {}
    path = session / "book_top3_events.jsonl"
    found = {}
    for nbytes in (BOOK_TAIL_BYTES, BOOK_TAIL_BYTES_MAX):
        rows = _read_jsonl_tail(path, nbytes)
        for r in rows:
            ticker = str(r.get("ticker") or "")
            if ticker in tickers:
                found[ticker] = r
        if tickers.issubset(found):
            break
    return found


def _shadow_inventory_from_disk(session: Path):
    out_dir = session / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1"
    fills = _read_jsonl(out_dir / "shadow_fills.jsonl")
    events = _read_jsonl(out_dir / "shadow_events.jsonl")
    inv = {}
    for f in fills:
        ticker = str(f.get("ticker") or "")
        if not ticker:
            continue
        qty = _f(f.get("qty"), 0.0)
        side = str(f.get("side") or "")
        inv[ticker] = inv.get(ticker, 0.0) + (qty if side == "BID" else -qty if side == "ASK" else 0.0)
    for e in events:
        if str(e.get("event") or "") == "M5_LIQUIDATE":
            ticker = str(e.get("ticker") or "")
            if ticker:
                inv[ticker] = 0.0
    return inv


def live_market_table() -> pd.DataFrame:
    session = _session_path()
    if session is None or not session.exists():
        return pd.DataFrame()

    now = pd.Timestamp.now(tz="UTC")
    meta = _current_contract_meta(session)
    books = _latest_book_by_ticker(session, meta)
    inv_map = _shadow_inventory_from_disk(session)
    rows = []

    for ticker, m in meta.items():
        series = str(m.get("series_ticker") or "")
        close = m["_close"]
        m0 = m["_m0"]
        elapsed = (now - m0).total_seconds()
        to_m5 = 300.0 - elapsed
        to_close = (close - now).total_seconds()
        if elapsed < 0:
            phase = "PRE-M0"
        elif elapsed < 300:
            phase = "M0-M5"
        elif elapsed < 900:
            phase = "M5-M15"
        elif elapsed < 930:
            phase = "POST-CLOSE"
        else:
            phase = "DONE"

        r = books.get(ticker) or {}
        bid = _f(r.get("yes_bid"))
        ask = _f(r.get("yes_ask"))
        mid = _f(r.get("mid"))
        spread = _f(r.get("spread_c"))
        bids = r.get("bid_levels") or []
        asks = r.get("ask_levels") or []
        bid_l3 = sum(_f(x[1], 0.0) for x in bids[:3] if isinstance(x, (list, tuple)) and len(x) >= 2)
        ask_l3 = sum(_f(x[1], 0.0) for x in asks[:3] if isinstance(x, (list, tuple)) and len(x) >= 2)
        support = "BID" if bid_l3 > ask_l3 + B.EPS else "ASK" if ask_l3 > bid_l3 + B.EPS else "TIE"
        inv = float(inv_map.get(ticker, 0.0))
        eligible = bool(
            abs(inv) <= B.EPS
            and 0.0 <= elapsed < 300.0
            and np.isfinite(spread)
            and spread + 1e-9 >= B.SPREAD_FLOOR_C
            and support in {"BID", "ASK"}
        )
        state = "EXIT / INVENTORY" if abs(inv) > B.EPS else "ENTRY ELIGIBLE" if eligible else "FLAT"

        liq_px = bid if inv > B.EPS else ask if inv < -B.EPS else np.nan
        rows.append({
            "series": series,
            "ticker": ticker,
            "phase": phase,
            "elapsed_s": elapsed,
            "to_m5_s": to_m5,
            "to_close_s": to_close,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_c": spread,
            "bid_l3": bid_l3,
            "ask_l3": ask_l3,
            "l3_support": support,
            "entry_eligible": "YES" if eligible else "NO",
            "state": state,
            "inventory": inv,
            "liq_px": liq_px,
            "book_event_type": r.get("event_type"),
            "book_receipt_time": r.get("receipt_time"),
        })

    out = pd.DataFrame(rows)
    if len(out):
        rank = {"M0-M5": 0, "PRE-M0": 1, "M5-M15": 2, "POST-CLOSE": 3, "DONE": 4}
        out["_rank"] = out.phase.map(rank).fillna(9)
        out = out.sort_values(["_rank", "series", "ticker"]).drop(columns="_rank").reset_index(drop=True)
    return out


def _market_table_html(df):
    if df is None or len(df) == 0:
        return '<div style="opacity:.6;font-size:12px">No raw-recorder market rows yet.</div>'
    z = df.copy()
    for c in ["bid", "ask", "mid", "liq_px"]:
        z[c] = pd.to_numeric(z[c], errors="coerce").map(lambda x: f"{100*x:.1f}¢" if np.isfinite(x) else "")
    z["spread_c"] = pd.to_numeric(z["spread_c"], errors="coerce").map(lambda x: f"{x:.1f}¢" if np.isfinite(x) else "")
    for c in ["bid_l3", "ask_l3", "inventory"]:
        z[c] = pd.to_numeric(z[c], errors="coerce").map(lambda x: f"{x:.2f}" if np.isfinite(x) else "")
    z["elapsed"] = pd.to_numeric(z["elapsed_s"], errors="coerce").map(lambda x: f"M{x/60:.2f}" if np.isfinite(x) else "")
    z["to_M5"] = pd.to_numeric(z["to_m5_s"], errors="coerce").map(lambda x: f"{max(0.0,x):.1f}s" if np.isfinite(x) else "")
    z["to_close"] = pd.to_numeric(z["to_close_s"], errors="coerce").map(lambda x: f"{max(0.0,x):.1f}s" if np.isfinite(x) else "")
    cols = [
        "series", "ticker", "phase", "elapsed", "to_M5", "to_close",
        "bid", "ask", "mid", "spread_c", "bid_l3", "ask_l3", "l3_support",
        "entry_eligible", "state", "inventory", "liq_px",
    ]
    return '<div style="overflow-x:auto;font-size:11px">' + z[cols].to_html(index=False, border=0, escape=True) + '</div>'


def render_cycle_q10_oos_dashboard_html(show_rows=12):
    st = B.oos_stack_status(show=False)
    s = st.get("shadow") or {}
    h = st.get("health") or {}
    fee = st.get("fee_preflight") or {}
    markets = live_market_table()
    live = markets[markets.phase == "M0-M5"].copy() if len(markets) else markets
    trading = markets[markets.inventory.abs() > B.EPS].copy() if len(markets) else markets

    recorder_ok = bool(st.get("running") and h.get("healthy"))
    shadow_ok = not bool(s.get("last_error")) and bool(s)
    fees_ok = bool(fee.get("ok"))
    all_ok = recorder_ok and shadow_ok and fees_ok
    mature = bool(s.get("oos_mature"))
    title_state = "OOS MATURE — KEEP FROZEN" if mature else "OOS RUNNING — NOT YET MATURE"
    bg = "#eaf7ee" if all_ok else "#ffebe9"
    border = "#238636" if all_ok else "#cf222e"

    parts = [
        '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1800px">',
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:14px 16px;margin-bottom:10px">',
        '<div style="font-size:22px;font-weight:800">FROZEN CYCLE_ALWAYS_EXIT Q10 — LIVE OOS SHADOW</div>',
        f'<div style="font-size:12px;opacity:.72">{title_state} · {DASHBOARD_VERSION}</div>',
        f'<div style="font-size:11px;opacity:.65">Session: {html.escape(str(st.get("session_dir")))}</div>',
        '</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">',
        _card("Recorder", "HEALTHY" if recorder_ok else "ATTENTION", f"epoch {h.get('connection_epoch')} · gaps {h.get('sequence_gaps')}", "ok" if recorder_ok else "bad"),
        _card("Shadow", "RUNNING" if shadow_ok else "ATTENTION", "frozen read-only", "ok" if shadow_ok else "bad"),
        _card("Fee preflight", "PASS" if fees_ok else "FAIL", "frozen assumptions", "ok" if fees_ok else "bad"),
        _card("OOS maturity", "MATURE" if mature else "COLLECTING", s.get("oos_maturity_rule", ""), "ok" if mature else "info"),
        _card("Runtime", f"{_f(s.get('runtime_hours'),0):.2f} h", f"{int(s.get('complete_9_series_windows',0) or 0)} complete windows", "info"),
        _card("Live M0-M5", len(live), f"{int((live.entry_eligible == 'YES').sum()) if len(live) else 0} entry-eligible · {len(trading)} carrying", "info"),
        '</div>',

        '<h3 style="margin:12px 0 6px">Net economics</h3>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">',
        _card("Passive matched", _money(s.get("passive_matched_pnl")), "maker fills", "ok" if _f(s.get("passive_matched_pnl"),0)>=0 else "warn"),
        _card("Forced liq gross", _money(s.get("forced_liq_gross_pnl")), "M5 residual", "neutral"),
        _card("Taker fees", _money(-_f(s.get("taker_trade_fees"),0)), "forced M5", "warn"),
        _card("Realized net", _money(s.get("realized_net_trade_fee_only")), "fee adjusted", "ok" if _f(s.get("realized_net_trade_fee_only"),0)>=0 else "warn"),
        _card("Open executable", _money(s.get("current_residual_net")), "if crossed now", "neutral"),
        _card("Net executable", _money(s.get("net_executable_pnl_trade_fee_only")), "live OOS", "ok" if _f(s.get("net_executable_pnl_trade_fee_only"),0)>=0 else "warn"),
        _card("Sample-rate / day", _money(s.get("sample_rate_net_per_day")), "judge after maturity", "info"),
        _card("Dev benchmark", _money(s.get("development_benchmark_per_day")), "fee-adjusted", "neutral"),
        '</div>',

        '<h3 style="margin:12px 0 6px">Execution / inventory</h3>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">',
        _card("Fills", int(s.get("fill_events",0) or 0), f"{_f(s.get('fill_qty'),0):,.1f} contracts", "info"),
        _card("Cycles", f"{int(s.get('cycles_completed',0) or 0)} / {int(s.get('cycles_started',0) or 0)}", _pct(s.get("cycle_completion_pct")), "info"),
        _card("Forced M5", int(s.get("forced_liquidations",0) or 0), f"{_f(s.get('forced_liq_qty'),0):,.1f} contracts", "warn"),
        _card("Open inventory", f"{_f(s.get('open_abs_inventory'),0):,.2f} ct", f"{int(s.get('open_inventory_contracts',0) or 0)} markets", "neutral"),
        _card("5s markout", f"{_f(s.get('markout_5s_c')):+.3f}¢", "qty weighted", "neutral"),
        _card("15s markout", f"{_f(s.get('markout_15s_c')):+.3f}¢", "qty weighted", "neutral"),
        _card("30s markout", f"{_f(s.get('markout_30s_c')):+.3f}¢", "qty weighted", "neutral"),
        '</div>',

        '<h3 style="margin:14px 0 4px">Current frozen-universe markets — raw recorder view</h3>',
        '<div style="font-size:11px;opacity:.68;margin-bottom:6px">Disk-backed: this table comes from market_metadata.jsonl + book_top3_events.jsonl, so notebook module reloads cannot make it disappear.</div>',
        _market_table_html(markets),
        '<h3 style="margin:14px 0 4px">Markets currently carrying shadow inventory</h3>',
        _market_table_html(trading) if len(trading) else '<div style="opacity:.6;font-size:12px">No open shadow inventory right now.</div>',
        '<div style="border:1px solid #9a6700;background:#fff7e6;border-radius:10px;padding:9px 11px;font-size:11px;margin-top:10px"><b>OOS discipline:</b> V4 is display-only and disk-backed. It does not alter the running recorder or shadow engine.</div>',
        '</div>',
    ]
    return ''.join(parts)


async def watch_cycle_q10_oos_dashboard(refresh_seconds=2.0, show_rows=12):
    refresh_seconds = max(0.5, float(refresh_seconds))
    from IPython.display import HTML, display
    handle = display(HTML(render_cycle_q10_oos_dashboard_html(show_rows=show_rows)), display_id=True)
    try:
        while True:
            await asyncio.sleep(refresh_seconds)
            handle.update(HTML(render_cycle_q10_oos_dashboard_html(show_rows=show_rows)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        handle.update(HTML(render_cycle_q10_oos_dashboard_html(show_rows=show_rows)))
        print("Dashboard refresh stopped. Recorder + frozen shadow remain running unchanged.")
