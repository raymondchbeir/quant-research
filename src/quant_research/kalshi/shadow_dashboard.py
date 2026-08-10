from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import primary_shadow_trader as P
from . import recorder as R


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
    minutes, seconds = divmod(x, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{sign}{minutes:02d}:{seconds:02d}"


def _read_events(path):
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return rows


def _event_maps(events):
    latest = {}
    skips = Counter()
    invalids = Counter()
    evaluated = 0
    for event in events:
        ticker = event.get("ticker")
        if ticker:
            latest[ticker] = event
        typ = event.get("event")
        if typ == "SIGNAL":
            evaluated += 1
        elif typ == "SKIP":
            evaluated += 1
            skips[event.get("reason") or event.get("detail") or "SKIP"] += 1
        elif typ == "DATA_INVALID":
            evaluated += 1
            invalids[event.get("reason") or event.get("detail") or "DATA_INVALID"] += 1
    return latest, skips, invalids, evaluated


def _performance_frame(sh):
    df = pd.DataFrame(list(sh.records.values()))
    if df.empty:
        return df
    for col in ["entry_fill_qty", "entry_price", "entry_queue", "realized_pnl", "midpoint", "spread_c", "quote_age_s", "book_age_s", "btc_return", "qty"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["decision_time", "actual_post_time", "cancel_time", "entry_first_fill", "entry_last_fill", "quote_time", "quote_exchange_time", "book_sample_time", "book_source_time", "book_exchange_time", "btc_now_time", "btc_past_time"]:
        if col not in df:
            df[col] = pd.NaT
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ["result", "direction", "status", "ticker", "series", "data_invalid_reason"]:
        if col not in df:
            df[col] = None
    return df.sort_values("decision_time", na_position="last").reset_index(drop=True)


def _active_contract_table(sh, latest_events):
    now = pd.Timestamp.now(tz="UTC")
    state = R._STATE or {}
    active = sorted(t for t in state.get("markets", set()) if str(t).split("-")[0] in P.FROZEN_SERIES)
    btc = sh._btc_return_15m(now)
    btc15 = np.nan if btc is None else float(btc[0])
    rows = []
    for ticker in active:
        close_time = P._close_from_ticker(ticker)
        decision_time = close_time - pd.Timedelta(minutes=10) if not pd.isna(close_time) else pd.NaT
        quote_hist = sh.quotes.get(ticker)
        quote = quote_hist[-1] if quote_hist else None
        book_hist = sh.books.get(ticker)
        book = book_hist[-1] if book_hist else None
        if quote is None:
            bid = ask = mid = spread = quote_age = np.nan
            lean = "n/a"
        else:
            bid, ask = quote["bid"], quote["ask"]
            mid = (bid + ask) / 2.0
            spread = ask - bid
            lean = "YES" if mid > 0.50 else "NO" if mid < 0.50 else "50/50"
            quote_age = (now - quote["receipt_time"]).total_seconds()
        book_age = np.nan
        if book is not None and not pd.isna(book.get("source_time")):
            book_age = (now - book["source_time"]).total_seconds()
        opposition = "n/a"
        if lean in {"YES", "NO"} and np.isfinite(btc15):
            opposition = "YES" if btc15 * (1 if lean == "YES" else -1) < 0 else "NO"

        rec = sh.records.get(ticker)
        order = sh.entry_orders.get(ticker)
        pos = sh.positions.get(ticker)
        if pos is not None and not pos.get("settled") and float(pos.get("open_qty", 0.0) or 0.0) > 0:
            position = f"{pos.get('direction')} {float(pos.get('open_qty', 0.0)):.2f}"
        elif order is not None and order.get("status") == "OPEN":
            position = f"BID {order.get('direction')} {float(order.get('remaining', 0.0)):.2f}@{100*float(order.get('price', 0)):.0f}c"
        else:
            position = "FLAT"

        if order is not None and order.get("status") == "OPEN":
            shadow_state = "ENTRY_OPEN"
        elif rec is not None and rec.get("status"):
            shadow_state = str(rec.get("status"))
        elif ticker in sh.decisions_done:
            shadow_state = "SKIPPED"
        else:
            shadow_state = "WAIT_M5"

        to_close = (close_time - now).total_seconds() if not pd.isna(close_time) else np.nan
        to_m5 = (decision_time - now).total_seconds() if not pd.isna(decision_time) else np.nan
        m5_text = "DONE" if ticker in sh.decisions_done else "DUE" if np.isfinite(to_m5) and to_m5 <= 0 else _fmt_seconds(to_m5)
        fill_qty = float(rec.get("entry_fill_qty", 0.0) or 0.0) if rec else 0.0
        realized = float(rec.get("realized_pnl", 0.0) or 0.0) if rec else 0.0
        latest = latest_events.get(ticker) or {}
        rows.append({
            "series": ticker.split("-")[0], "to_M5": m5_text, "to_close": _fmt_seconds(to_close),
            "YES_bid": np.nan if not np.isfinite(bid) else round(100 * bid, 2),
            "YES_ask": np.nan if not np.isfinite(ask) else round(100 * ask, 2),
            "mid_c": np.nan if not np.isfinite(mid) else round(100 * mid, 2),
            "spread_c": np.nan if not np.isfinite(spread) else round(100 * spread, 2),
            "lean": lean, "BTC15_bp": np.nan if not np.isfinite(btc15) else round(10000 * btc15, 2),
            "opp?": opposition, "shadow_state": shadow_state, "position": position,
            "fill_qty": round(fill_qty, 2), "realized_pnl": round(realized, 4),
            "quote_age_s": np.nan if not np.isfinite(quote_age) else round(quote_age, 2),
            "book_age_s": np.nan if not np.isfinite(book_age) else round(book_age, 2),
            "epoch": state.get("connection_epoch"), "last_event": latest.get("event"), "ticker": ticker,
        })
    return pd.DataFrame(rows)


def status_snapshot():
    sh = P._SHADOW
    if sh is None:
        return None
    with sh.lock:
        df = _performance_frame(sh)
        thread_state = {thread.name: thread.is_alive() for thread in sh.threads}
    latest_events, skips, invalids, evaluated = _event_maps(_read_events(sh.event_file))
    contracts = _active_contract_table(sh, latest_events)
    health = R.recorder_health_snapshot()

    if df.empty:
        metrics = {
            "signals": 0, "data_invalid": 0, "fills": 0, "fill_rate": np.nan,
            "settled_fills": 0, "pnl": 0.0, "settled_pnl": 0.0, "max_dd": 0.0,
            "peak_equity": 0.0, "wins": 0, "losses": 0, "scratches": 0,
            "filled_accuracy": np.nan, "raw_accuracy": np.nan, "profit_factor": np.nan,
            "open_positions": 0, "open_contracts": 0.0, "open_cost": 0.0,
            "winner_fill_rate": np.nan, "loser_fill_rate": np.nan, "adverse_ratio": np.nan,
            "windows": 0,
        }
    else:
        eligible = df["direction"].isin(["YES", "NO"]) & df["entry_price"].notna() & ~df["status"].eq("DATA_INVALID")
        signal_df = df.loc[eligible].copy()
        invalid = df["status"].eq("DATA_INVALID")
        filled = eligible & (df["entry_fill_qty"].fillna(0) > 1e-12)
        settled = df["status"].eq("SETTLED")
        settled_filled = settled & filled
        known = eligible & df["result"].isin(["YES", "NO"])
        correct = known & df["direction"].eq(df["result"])
        wrong = known & ~df["direction"].eq(df["result"])
        trade_pnl = df.loc[settled_filled].sort_values("decision_time")["realized_pnl"].fillna(0).astype(float)
        if len(trade_pnl):
            equity = trade_pnl.cumsum().to_numpy(float)
            peaks = np.maximum.accumulate(np.r_[0.0, equity])[-len(equity):]
            max_dd = float((peaks - equity).max())
            peak_equity = float(max(0.0, equity.max()))
        else:
            max_dd = peak_equity = 0.0
        gross_profit = float(trade_pnl[trade_pnl > 0].sum())
        gross_loss = float(-trade_pnl[trade_pnl < 0].sum())
        pf = np.inf if gross_loss == 0 and gross_profit > 0 else np.nan if gross_loss == 0 else gross_profit / gross_loss
        open_qty = df["entry_fill_qty"].fillna(0).clip(lower=0)
        open_mask = filled & ~settled & (open_qty > 1e-12)
        winner_signals, loser_signals = int(correct.sum()), int(wrong.sum())
        winner_fills, loser_fills = int((correct & filled).sum()), int((wrong & filled).sum())
        winner_fr, loser_fr = _pct(winner_fills, winner_signals), _pct(loser_fills, loser_signals)
        adverse = loser_fr / winner_fr if np.isfinite(winner_fr) and winner_fr > 0 and np.isfinite(loser_fr) else np.nan
        metrics = {
            "signals": int(eligible.sum()), "data_invalid": int(invalid.sum()), "fills": int(filled.sum()),
            "fill_rate": _pct(int(filled.sum()), int(eligible.sum())), "settled_fills": int(settled_filled.sum()),
            "pnl": float(df["realized_pnl"].fillna(0).sum()),
            "settled_pnl": float(df.loc[settled, "realized_pnl"].fillna(0).sum()),
            "max_dd": max_dd, "peak_equity": peak_equity, "wins": int((trade_pnl > 0).sum()),
            "losses": int((trade_pnl < 0).sum()), "scratches": int((trade_pnl == 0).sum()),
            "filled_accuracy": _pct(int((correct & settled_filled).sum()), int(settled_filled.sum())),
            "raw_accuracy": _pct(int(correct.sum()), int(known.sum())), "profit_factor": pf,
            "open_positions": int(open_mask.sum()), "open_contracts": float(open_qty[open_mask].sum()),
            "open_cost": float((open_qty[open_mask] * df.loc[open_mask, "entry_price"]).fillna(0).sum()),
            "winner_fill_rate": winner_fr, "loser_fill_rate": loser_fr, "adverse_ratio": adverse,
            "windows": int(signal_df.loc[signal_df["decision_time"].notna(), "decision_time"].nunique()),
        }

    return {
        "sh": sh, "df": df, "contracts": contracts, "health": health, "thread_state": thread_state,
        "skips": skips, "invalids": invalids, "evaluated": evaluated, "metrics": metrics,
    }


def primary_shadow_status(show_rows=10):
    snap = status_snapshot()
    if snap is None:
        print("Primary shadow trader is not running.")
        return pd.DataFrame()
    df, contracts, health, metrics = snap["df"], snap["contracts"], snap["health"], snap["metrics"]
    print("=" * 120)
    print("PRIMARY SHADOW — CONFIRMATORY LIVE MONITOR")
    print("M5 | BTC completed-1m 15m OPPOSITION | SPREAD <=2c | PASSIVE -3c | ACTUAL-POST 15s | 3 CONTRACTS | HOLD TO SETTLEMENT | READ-ONLY")
    print(f"UTC now: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')} | recorder epoch={health.get('connection_epoch')}")
    print("=" * 120)

    print("\nACTIVE FROZEN CONTRACTS")
    if contracts.empty:
        print("  No active frozen-universe contracts currently in recorder state.")
    else:
        cols = ["series", "to_M5", "to_close", "YES_bid", "YES_ask", "mid_c", "spread_c", "lean", "BTC15_bp", "opp?", "shadow_state", "position", "fill_qty", "realized_pnl", "quote_age_s", "book_age_s", "epoch", "ticker"]
        try:
            from IPython.display import display
            display(contracts[cols])
        except Exception:
            print(contracts[cols].to_string(index=False))

    print("\nRECORDER / DATA HEALTH")
    print(f"  Running / healthy            {str(health.get('running')):>6} / {str(health.get('healthy')):<6}     Connection epoch          {str(health.get('connection_epoch')):>6}")
    print(f"  Active / expired contracts   {health.get('active_markets', 0):>6} / {health.get('expired_active_markets', 0):<6}     Books in memory           {health.get('books_in_memory', 0):>6}")
    print(f"  Supervisor age               {str(health.get('supervisor_age_s')):>8}s     Market-data age           {str(health.get('market_data_age_s')):>8}s")

    print("\nDECISION ENGINE")
    print(f"  Decisions evaluated          {snap['evaluated']:>6}     Eligible signals           {metrics['signals']:>6}")
    print(f"  DATA_INVALID                 {metrics['data_invalid']:>6}     Strategy skips             {sum(snap['skips'].values()):>6}")
    if snap["invalids"]:
        print("  Data-invalid reasons:")
        for reason, count in snap["invalids"].most_common():
            print(f"    {str(reason):<42} {count:>6}")
    if snap["skips"]:
        print("  Strategy skip reasons:")
        for reason, count in snap["skips"].most_common():
            print(f"    {str(reason):<42} {count:>6}")

    print("\nPERFORMANCE / EXECUTION")
    print(f"  Signals                      {metrics['signals']:>6}     Independent windows        {metrics['windows']:>6}")
    print(f"  Any fills                    {metrics['fills']:>6}     Fill rate                 {_fmt_pct(metrics['fill_rate']):>8}")
    print(f"  Settled fills                {metrics['settled_fills']:>6}     Open positions             {metrics['open_positions']:>6}")
    print(f"  Realized PnL            {_fmt_money(metrics['pnl']):>12}     Settled PnL             {_fmt_money(metrics['settled_pnl']):>12}")
    print(f"  Settled max drawdown        ${metrics['max_dd']:>8.4f}     Peak settled equity      ${metrics['peak_equity']:>8.4f}")
    pf = metrics["profit_factor"]
    pf_text = "inf" if np.isinf(pf) else "n/a" if not np.isfinite(pf) else f"{pf:.3f}"
    print(f"  Profit factor                {pf_text:>8}     Filled accuracy           {_fmt_pct(metrics['filled_accuracy']):>8}")
    print(f"  Wins / losses / scratches   {metrics['wins']:>3} / {metrics['losses']:<3} / {metrics['scratches']:<3}     Raw signal accuracy        {_fmt_pct(metrics['raw_accuracy']):>8}")
    print(f"  Open contracts               {metrics['open_contracts']:>8.2f}     Open entry cost          ${metrics['open_cost']:>8.4f}")

    print("\nADVERSE SELECTION")
    print(f"  Winner fill rate             {_fmt_pct(metrics['winner_fill_rate']):>8}")
    print(f"  Loser fill rate              {_fmt_pct(metrics['loser_fill_rate']):>8}")
    adverse = metrics["adverse_ratio"]
    print(f"  Loser / winner fill ratio    {('n/a' if not np.isfinite(adverse) else f'{adverse:.3f}x'):>8}     <1.0 preferred")

    print("\nTHREAD HEALTH")
    for name, alive in snap["thread_state"].items():
        print(f"  {name:<22} {'OK' if alive else 'DEAD'}")

    if show_rows and not df.empty:
        print("\nLATEST DECISIONS / SIGNALS")
        cols = ["decision_time", "actual_post_time", "ticker", "direction", "midpoint", "spread_c", "btc_return", "entry_price", "entry_queue", "quote_age_s", "book_age_s", "entry_fill_qty", "result", "realized_pnl", "status", "data_invalid_reason"]
        try:
            from IPython.display import display
            display(df[cols].tail(int(show_rows)))
        except Exception:
            print(df[cols].tail(int(show_rows)).to_string(index=False))
    print("=" * 120)
    return df


shadow_status = primary_shadow_status
