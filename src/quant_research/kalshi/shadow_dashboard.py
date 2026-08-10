from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder as R
from . import shadow_trader as S
from . import shadow_notebook_parity as P


def _pct(n, d):
    return np.nan if not d else 100.0 * float(n) / float(d)


def _fmt_pct(x):
    return "n/a" if not np.isfinite(x) else f"{x:.2f}%"


def _fmt_money(x):
    return f"${float(x):+.4f}"


def _fmt_seconds(x):
    if x is None or not np.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(int(round(x)))
    m, s = divmod(x, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{sign}{h:02d}:{m:02d}:{s:02d}"
    return f"{sign}{m:02d}:{s:02d}"


def _read_events(path):
    rows = []
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return rows
        with p.open("rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - 4 * 1024 * 1024))
            if size > 4 * 1024 * 1024:
                f.readline()
            for raw in f:
                try:
                    rows.append(json.loads(raw.decode("utf-8", errors="ignore")))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def _skip_reason(detail):
    d = str(detail or "")
    if "spread=" in d and ">2c" in d:
        return "spread > 2c"
    if "BTC not opposition" in d:
        return "BTC not opposition"
    if "missing causal Coinbase" in d:
        return "missing BTC history"
    if "no causal quote" in d:
        return "no causal quote"
    if "midpoint exactly 50c" in d:
        return "midpoint = 50c"
    if "illegal/non-passive" in d:
        return "illegal/non-passive -3c"
    if "shadow scheduler late" in d:
        return "scheduler late >5s"
    return None


def _event_maps(events):
    latest = {}
    skips = Counter()
    evaluated = 0
    for e in events:
        ev = e.get("event")
        ticker = e.get("ticker")
        if ticker:
            latest[ticker] = e
        if ev == "SIGNAL":
            evaluated += 1
        elif ev == "SKIP":
            reason = _skip_reason(e.get("detail"))
            if reason is not None:
                skips[reason] += 1
                evaluated += 1
    return latest, skips, evaluated


def _latest_quote(sh, ticker):
    hist = sh.quotes.get(ticker)
    if not hist:
        return None
    return hist[-1]


def _current_btc15(sh, now):
    try:
        out = sh._btc_return_15m(now)
        if out is None:
            return np.nan
        return float(out[0])
    except Exception:
        return np.nan


def _contract_state(sh, ticker, latest_event):
    rec = sh.records.get(ticker)
    order = sh.entry_orders.get(ticker)
    pos = sh.positions.get(ticker)
    exit_order = sh.exit_orders.get(ticker)

    if pos is not None:
        open_qty = float(pos.get("open_qty", 0.0) or 0.0)
        direction = str(pos.get("direction") or "")
        if pos.get("settled"):
            position = "FLAT"
        elif open_qty > 1e-12:
            position = f"{direction} {open_qty:.2f}"
        else:
            position = "FLAT"
    elif order is not None and order.get("status") == "OPEN":
        direction = str(order.get("direction") or "")
        remaining = float(order.get("remaining", 0.0) or 0.0)
        price = float(order.get("price", np.nan))
        position = f"BID {direction} {remaining:.2f}@{100*price:.0f}c"
    else:
        position = "FLAT"

    if exit_order is not None and exit_order.get("status") == "OPEN":
        state = "EXIT_OPEN"
    elif order is not None and order.get("status") == "OPEN":
        state = "ENTRY_OPEN"
    elif rec is not None and rec.get("status"):
        state = str(rec.get("status"))
    elif ticker in sh.decisions_done:
        state = "SKIPPED"
    else:
        state = "WAIT_M5"

    detail = ""
    if latest_event is not None:
        ev = latest_event.get("event") or ""
        d = str(latest_event.get("detail") or "")
        detail = f"{ev}: {d}" if d else str(ev)
        if len(detail) > 80:
            detail = detail[:77] + "..."

    fill_qty = 0.0
    realized = 0.0
    if rec is not None:
        fill_qty = float(rec.get("entry_fill_qty", 0.0) or 0.0)
        realized = float(rec.get("realized_pnl", 0.0) or 0.0)

    return state, position, fill_qty, realized, detail


def _active_contract_table(sh, latest_events):
    now = pd.Timestamp.now(tz="UTC")
    state = R._STATE or {}
    active = sorted(state.get("markets", set()))
    active = [t for t in active if str(t).split("-")[0] in S.FROZEN_SERIES]

    btc15 = _current_btc15(sh, now)
    rows = []

    for ticker in active:
        close_time = P._close_from_ticker(ticker)
        decision_time = close_time - pd.Timedelta(minutes=10) if not pd.isna(close_time) else pd.NaT
        quote = _latest_quote(sh, ticker)

        if quote is None:
            quote_ts = pd.NaT
            bid = ask = mid = spread = np.nan
            lean = "n/a"
        else:
            quote_ts, bid, ask = quote
            mid = (bid + ask) / 2.0
            spread = ask - bid
            lean = "YES" if mid > 0.50 else "NO" if mid < 0.50 else "50/50"

        if lean in {"YES", "NO"} and np.isfinite(btc15):
            sign = 1 if lean == "YES" else -1
            opposition = "YES" if btc15 * sign < 0 else "NO"
        else:
            opposition = "n/a"

        state_text, position, fill_qty, realized, detail = _contract_state(
            sh, ticker, latest_events.get(ticker)
        )

        to_close = (close_time - now).total_seconds() if not pd.isna(close_time) else np.nan
        to_m5 = (decision_time - now).total_seconds() if not pd.isna(decision_time) else np.nan

        if ticker in sh.decisions_done:
            m5_text = "DONE"
        elif np.isfinite(to_m5) and to_m5 <= 0:
            m5_text = "DUE"
        else:
            m5_text = _fmt_seconds(to_m5)

        quote_age = (now - quote_ts).total_seconds() if not pd.isna(quote_ts) else np.nan

        rows.append({
            "series": ticker.split("-")[0],
            "ticker": ticker,
            "to_M5": m5_text,
            "to_close": _fmt_seconds(to_close),
            "YES_bid": np.nan if not np.isfinite(bid) else round(100 * bid, 2),
            "YES_ask": np.nan if not np.isfinite(ask) else round(100 * ask, 2),
            "mid_c": np.nan if not np.isfinite(mid) else round(100 * mid, 2),
            "spread_c": np.nan if not np.isfinite(spread) else round(100 * spread, 2),
            "lean": lean,
            "BTC15_bp": np.nan if not np.isfinite(btc15) else round(10000 * btc15, 2),
            "opp?": opposition,
            "shadow_state": state_text,
            "position": position,
            "fill_qty": round(fill_qty, 2),
            "realized_pnl": round(realized, 4),
            "quote_age_s": np.nan if not np.isfinite(quote_age) else round(quote_age, 2),
            "last_event": detail,
        })

    return pd.DataFrame(rows)


def _performance_frame(sh):
    df = pd.DataFrame(list(sh.records.values()))
    if df.empty:
        return df
    for c in ["entry_fill_qty", "exit_fill_qty", "entry_price", "entry_queue", "realized_pnl", "midpoint", "spread_c"]:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["decision_time", "entry_first_fill", "entry_last_fill", "exit_trigger", "exit_post"]:
        if c not in df:
            df[c] = pd.NaT
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    for c in ["result", "direction", "status", "ticker", "series"]:
        if c not in df:
            df[c] = None
    return df.sort_values("decision_time", na_position="last").reset_index(drop=True)


def primary_shadow_status(show_rows=10):
    if S._SHADOW is None:
        print("Primary shadow trader is not running.")
        return pd.DataFrame()

    sh = S._SHADOW
    with sh.lock:
        thread_state = {t.name: t.is_alive() for t in sh.threads}
        df = _performance_frame(sh)

    events = _read_events(sh.event_file)
    latest_events, skips, evaluated = _event_maps(events)
    contracts = _active_contract_table(sh, latest_events)

    print("=" * 118)
    print("PRIMARY SHADOW — LIVE CONTRACT MONITOR")
    print("M5 | BTC 15m OPPOSITION | SPREAD <=2c | PASSIVE -3c | 15s | 3 CONTRACTS | 10c SALVAGE | READ-ONLY")
    print(f"Parity: {P.PARITY_VERSION} | UTC now: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 118)

    print("\nACTIVE FROZEN CONTRACTS")
    if contracts.empty:
        print("  Recorder currently has no active frozen-universe contracts.")
    else:
        cols = [
            "series", "to_M5", "to_close", "YES_bid", "YES_ask", "mid_c", "spread_c",
            "lean", "BTC15_bp", "opp?", "shadow_state", "position", "fill_qty", "realized_pnl",
            "quote_age_s", "ticker",
        ]
        try:
            from IPython.display import display
            display(contracts[cols])
        except Exception:
            print(contracts[cols].to_string(index=False))

    signals = len(df)
    print("\nDECISION ENGINE")
    print(f"  Active frozen contracts       {len(contracts):>6}     decisions evaluated          {evaluated:>6}")
    print(f"  Eligible signals              {signals:>6}     decisions_done set           {len(sh.decisions_done):>6}")
    if skips:
        print("  Skip reasons:")
        for reason, count in skips.most_common():
            print(f"    {reason:<32} {count:>6}")
    elif evaluated == 0:
        print("  No M5 decision has fired since this shadow instance started.")

    if df.empty:
        print("\nPERFORMANCE")
        print("  Signals                           0     Any fills                         0")
        print("  Realized PnL                $+0.0000     Settled max drawdown        $0.0000")
        print("  Open positions                    0     Open contracts                  0.00")
    else:
        filled = df["entry_fill_qty"].fillna(0) > 1e-12
        full = df["entry_fill_qty"].fillna(0) >= S.QTY - 1e-12
        known = df["result"].isin(["YES", "NO"])
        settled = df["status"].eq("SETTLED")
        settled_filled = settled & filled
        correct = known & df["direction"].eq(df["result"])
        wrong = known & ~df["direction"].eq(df["result"])

        pnl = df["realized_pnl"].fillna(0).astype(float)
        settled_pnl = df.loc[settled, "realized_pnl"].fillna(0).astype(float)
        booked_pnl = float(pnl.sum())
        settled_total = float(settled_pnl.sum())
        settled_trades = df.loc[settled_filled].copy().sort_values("decision_time")
        trade_pnl = settled_trades["realized_pnl"].fillna(0).astype(float)

        if len(trade_pnl):
            equity = trade_pnl.cumsum().to_numpy(float)
            peaks = np.maximum.accumulate(np.r_[0.0, equity])[-len(equity):]
            max_dd = float((peaks - equity).max())
            peak_eq = float(max(0.0, equity.max()))
        else:
            max_dd = peak_eq = 0.0

        gross_profit = float(trade_pnl[trade_pnl > 0].sum())
        gross_loss = float(-trade_pnl[trade_pnl < 0].sum())
        pf = np.inf if gross_loss == 0 and gross_profit > 0 else (np.nan if gross_loss == 0 else gross_profit / gross_loss)
        pf_text = "inf" if np.isinf(pf) else ("n/a" if not np.isfinite(pf) else f"{pf:.3f}")

        wins = int((trade_pnl > 0).sum())
        losses = int((trade_pnl < 0).sum())
        scratches = int((trade_pnl == 0).sum())
        settled_fill_count = int(settled_filled.sum())
        filled_acc = _pct(int((correct & settled_filled).sum()), settled_fill_count)
        raw_acc = _pct(int(correct.sum()), int(known.sum()))

        winner_signals = int(correct.sum())
        loser_signals = int(wrong.sum())
        winner_fills = int((correct & filled).sum())
        loser_fills = int((wrong & filled).sum())
        winner_fr = _pct(winner_fills, winner_signals)
        loser_fr = _pct(loser_fills, loser_signals)
        adverse = np.nan
        if np.isfinite(winner_fr) and winner_fr > 0 and np.isfinite(loser_fr):
            adverse = loser_fr / winner_fr

        open_qty = (df["entry_fill_qty"].fillna(0) - df["exit_fill_qty"].fillna(0)).clip(lower=0)
        open_mask = filled & ~settled & (open_qty > 1e-12)
        open_positions = int(open_mask.sum())
        open_contracts = float(open_qty[open_mask].sum())
        open_cost = float((open_qty[open_mask] * df.loc[open_mask, "entry_price"]).fillna(0).sum())

        settled_cost = float((df.loc[settled_filled, "entry_price"] * df.loc[settled_filled, "entry_fill_qty"]).fillna(0).sum())
        roi = np.nan if settled_cost <= 0 else 100.0 * settled_total / settled_cost
        windows = int(df.loc[df["decision_time"].notna(), "decision_time"].nunique())
        exit_triggers = int(df["exit_trigger"].notna().sum())
        exit_posts = int(df["exit_post"].notna().sum())
        exit_fills = int((df["exit_fill_qty"].fillna(0) > 1e-12).sum())

        print("\nPERFORMANCE / EXECUTION")
        print(f"  Signals                      {signals:>6}     Independent windows          {windows:>6}")
        print(f"  Any fills                    {int(filled.sum()):>6}     Fill rate                  {_fmt_pct(_pct(int(filled.sum()), signals)):>8}")
        print(f"  Full 3-contract fills        {int(full.sum()):>6}     Settled fills              {settled_fill_count:>6}")
        print(f"  Realized PnL            {_fmt_money(booked_pnl):>12}     Settled PnL             {_fmt_money(settled_total):>12}")
        print(f"  Settled max drawdown        ${max_dd:>8.4f}     Peak settled equity        ${peak_eq:>8.4f}")
        print(f"  ROI on settled entry cost   {_fmt_pct(roi):>10}     Profit factor              {pf_text:>8}")
        print(f"  Wins / losses / scratches   {wins:>3} / {losses:<3} / {scratches:<3}     Filled accuracy            {_fmt_pct(filled_acc):>8}")
        print(f"  Raw signal accuracy         {_fmt_pct(raw_acc):>10}     PnL / signal             {_fmt_money(booked_pnl/signals):>12}")

        print("\nADVERSE SELECTION")
        print(f"  Winner fill rate            {_fmt_pct(winner_fr):>10}     ({winner_fills}/{winner_signals})")
        print(f"  Loser fill rate             {_fmt_pct(loser_fr):>10}     ({loser_fills}/{loser_signals})")
        adverse_text = "n/a" if not np.isfinite(adverse) else f"{adverse:.3f}x"
        print(f"  Loser / winner fill ratio   {adverse_text:>10}     <1.0 is preferred")

        print("\nSALVAGE / CURRENT RISK")
        print(f"  10c triggers / posts / fills {exit_triggers:>3} / {exit_posts:<3} / {exit_fills:<3}")
        print(f"  Open filled positions        {open_positions:>6}     Open contracts              {open_contracts:>8.2f}")
        print(f"  Open entry cost              ${open_cost:>8.4f}     no MTM included")

    print("\nTHREAD HEALTH")
    for name, alive in thread_state.items():
        print(f"  {name:<22} {'OK' if alive else 'DEAD'}")

    if show_rows and not df.empty:
        print("\nLATEST SIGNALS")
        cols = [
            "decision_time", "ticker", "direction", "midpoint", "spread_c", "entry_price",
            "entry_queue", "entry_fill_qty", "result", "realized_pnl", "status",
        ]
        try:
            from IPython.display import display
            display(df[cols].tail(int(show_rows)))
        except Exception:
            print(df[cols].tail(int(show_rows)).to_string(index=False))

    print("=" * 118)
    return df


def watch_primary_shadow_status(refresh_seconds=2.0, show_rows=8):
    refresh_seconds = max(0.5, float(refresh_seconds))
    try:
        from IPython.display import clear_output
    except Exception:
        clear_output = None

    print("Live dashboard running. Interrupt the cell to stop refreshing; the shadow trader keeps running.")
    try:
        while True:
            if clear_output is not None:
                clear_output(wait=True)
            primary_shadow_status(show_rows=show_rows)
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        print("Live dashboard refresh stopped. Shadow trader was NOT stopped.")


shadow_status = primary_shadow_status
watch_shadow_status = watch_primary_shadow_status
