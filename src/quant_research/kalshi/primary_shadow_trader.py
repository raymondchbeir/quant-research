from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from . import shadow_trader as S


def _resolve_session(session_dir=None):
    if session_dir is not None:
        return Path(session_dir)
    from .recorder import current_session_dir
    session = current_session_dir()
    if session is None:
        raise RuntimeError("No recorder session is running. Start the recorder first.")
    return Path(session)


def start_primary_shadow_trader(session_dir=None):
    session = _resolve_session(session_dir)
    if S._SHADOW is not None and any(t.is_alive() for t in S._SHADOW.threads):
        if Path(S._SHADOW.session_dir).resolve() == session.resolve():
            print("Primary shadow trader is already running for this recorder session.")
            return S._SHADOW
        S._SHADOW.stop()
        S._SHADOW = None
    S._SHADOW = S.ShadowTrader(session).start()
    print("PRIMARY SHADOW: M5 | BTC opposition | spread<=2c | -3c | 15s | 3ct | 10c candle-confirmed passive salvage | READ-ONLY")
    return S._SHADOW


def stop_primary_shadow_trader():
    if S._SHADOW is None:
        print("Primary shadow trader is not running.")
        return None
    out = S._SHADOW.stop()
    S._SHADOW = None
    return out


def _pct(n, d):
    return np.nan if not d else 100.0 * float(n) / float(d)


def _fmt_pct(x):
    return "n/a" if not np.isfinite(x) else f"{x:.2f}%"


def _fmt_money(x):
    return f"${float(x):+.4f}"


def _mean_finite(s):
    x = pd.to_numeric(s, errors="coerce")
    x = x[np.isfinite(x)]
    return np.nan if len(x) == 0 else float(x.mean())


def primary_shadow_status(show_rows=15):
    if S._SHADOW is None:
        print("Primary shadow trader is not running.")
        return pd.DataFrame()

    sh = S._SHADOW
    with sh.lock:
        df = pd.DataFrame(list(sh.records.values()))
        thread_state = {t.name: t.is_alive() for t in sh.threads}

    print("=" * 108)
    print("PRIMARY SHADOW — LIVE RESEARCH DASHBOARD")
    print("M5 | BTC 15m OPPOSITION | SPREAD <=2c | PASSIVE -3c | 15s | 3 CONTRACTS | 10c SALVAGE | READ-ONLY")
    print("=" * 108)

    if df.empty:
        print("No eligible shadow signals yet.")
        print("\nTHREADS")
        for name, alive in thread_state.items():
            print(f"  {name:<22} {'OK' if alive else 'DEAD'}")
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

    df = df.sort_values("decision_time", na_position="last").reset_index(drop=True)
    filled = df["entry_fill_qty"].fillna(0) > 1e-12
    full = df["entry_fill_qty"].fillna(0) >= 3.0 - 1e-12
    outcome_known = df["result"].isin(["YES", "NO"])
    settled = df["status"].eq("SETTLED")
    settled_filled = settled & filled
    correct = outcome_known & df["direction"].eq(df["result"])
    wrong = outcome_known & ~df["direction"].eq(df["result"])

    signals = len(df)
    windows = int(df.loc[df["decision_time"].notna(), "decision_time"].nunique())
    filled_windows = int(df.loc[filled & df["decision_time"].notna(), "decision_time"].nunique())
    any_fills = int(filled.sum())
    full_fills = int(full.sum())
    known = int(outcome_known.sum())
    settled_count = int(settled.sum())
    settled_fill_count = int(settled_filled.sum())

    booked_pnl = float(df["realized_pnl"].fillna(0).sum())
    settled_pnl = float(df.loc[settled, "realized_pnl"].fillna(0).sum())
    settled_cost = float((df.loc[settled_filled, "entry_price"] * df.loc[settled_filled, "entry_fill_qty"]).fillna(0).sum())
    roi = np.nan if settled_cost <= 0 else 100.0 * settled_pnl / settled_cost

    settled_trades = df.loc[settled_filled].copy().sort_values("decision_time")
    trade_pnl = settled_trades["realized_pnl"].fillna(0).astype(float)
    gross_profit = float(trade_pnl[trade_pnl > 0].sum())
    gross_loss = float(-trade_pnl[trade_pnl < 0].sum())
    profit_factor = np.inf if gross_loss == 0 and gross_profit > 0 else (np.nan if gross_loss == 0 else gross_profit / gross_loss)
    wins = int((trade_pnl > 0).sum())
    losses = int((trade_pnl < 0).sum())
    scratches = int((trade_pnl == 0).sum())

    if len(trade_pnl):
        equity = trade_pnl.cumsum().to_numpy(float)
        peak = np.maximum.accumulate(np.r_[0.0, equity])[-len(equity):]
        drawdown = peak - equity
        max_dd = float(drawdown.max())
        peak_equity = float(max(0.0, equity.max()))
        best_idx = trade_pnl.idxmax()
        worst_idx = trade_pnl.idxmin()
        best_trade = (str(df.loc[best_idx, "ticker"]), float(df.loc[best_idx, "realized_pnl"]))
        worst_trade = (str(df.loc[worst_idx, "ticker"]), float(df.loc[worst_idx, "realized_pnl"]))
    else:
        max_dd = peak_equity = 0.0
        best_trade = worst_trade = ("n/a", 0.0)

    signal_accuracy = _pct(int(correct.sum()), known)
    filled_accuracy = _pct(int(correct[settled_filled].sum()), settled_fill_count)
    winner_signals = int(correct.sum())
    loser_signals = int(wrong.sum())
    winner_fills = int((correct & filled).sum())
    loser_fills = int((wrong & filled).sum())
    winner_fill_rate = _pct(winner_fills, winner_signals)
    loser_fill_rate = _pct(loser_fills, loser_signals)
    adverse_ratio = np.nan
    if np.isfinite(winner_fill_rate) and winner_fill_rate > 0 and np.isfinite(loser_fill_rate):
        adverse_ratio = loser_fill_rate / winner_fill_rate

    exit_posts = int(df["exit_post"].notna().sum())
    exit_triggers = int(df["exit_trigger"].notna().sum())
    exit_fills = int((df["exit_fill_qty"].fillna(0) > 1e-12).sum())

    open_qty = (df["entry_fill_qty"].fillna(0) - df["exit_fill_qty"].fillna(0)).clip(lower=0)
    open_mask = filled & ~settled & (open_qty > 1e-12)
    open_positions = int(open_mask.sum())
    open_contracts = float(open_qty[open_mask].sum())
    open_cost = float((open_qty[open_mask] * df.loc[open_mask, "entry_price"]).fillna(0).sum())

    avg_queue_filled = _mean_finite(df.loc[filled, "entry_queue"])
    avg_queue_no_fill = _mean_finite(df.loc[~filled, "entry_queue"])
    zero_queue = df["entry_queue"].fillna(np.inf).abs() <= 1e-12
    zero_queue_fill_rate = _pct(int((zero_queue & filled).sum()), int(zero_queue.sum()))

    print("\nSAMPLE / EXECUTION")
    print(f"  Signals                     {signals:>6}     Independent windows       {windows:>6}")
    print(f"  Any entry fills             {any_fills:>6}     Fill rate                 {_fmt_pct(_pct(any_fills, signals)):>8}")
    print(f"  Full 3-contract fills       {full_fills:>6}     Filled windows             {filled_windows:>6}")
    print(f"  Outcomes known              {known:>6}     Settled fills               {settled_fill_count:>6}")

    print("\nPERFORMANCE — REALIZED / SETTLED")
    print(f"  Booked realized PnL         {_fmt_money(booked_pnl):>12}     Settled PnL              {_fmt_money(settled_pnl):>12}")
    print(f"  Return on deployed cost     {_fmt_pct(roi):>12}     Settled entry cost         ${settled_cost:>11.4f}")
    print(f"  PnL / signal                {_fmt_money(booked_pnl / signals):>12}     PnL / settled fill       {_fmt_money(settled_pnl / settled_fill_count if settled_fill_count else 0):>12}")
    print(f"  Wins / losses / scratches   {wins:>3} / {losses:<3} / {scratches:<3}     Filled accuracy           {_fmt_pct(filled_accuracy):>8}")
    pf_text = "inf" if np.isinf(profit_factor) else ("n/a" if not np.isfinite(profit_factor) else f"{profit_factor:.3f}")
    print(f"  Gross profit / gross loss   ${gross_profit:>8.4f} / ${gross_loss:<8.4f}     Profit factor              {pf_text:>8}")
    print(f"  Settled max drawdown        ${max_dd:>11.4f}     Peak settled equity       ${peak_equity:>11.4f}")
    print(f"  Best trade                  {_fmt_money(best_trade[1]):>12}     {best_trade[0]}")
    print(f"  Worst trade                 {_fmt_money(worst_trade[1]):>12}     {worst_trade[0]}")

    print("\nADVERSE-SELECTION DIAGNOSTICS")
    print(f"  Raw signal accuracy         {_fmt_pct(signal_accuracy):>12}")
    print(f"  Winner fill rate            {_fmt_pct(winner_fill_rate):>12}     ({winner_fills}/{winner_signals})")
    print(f"  Loser fill rate             {_fmt_pct(loser_fill_rate):>12}     ({loser_fills}/{loser_signals})")
    ratio_text = "n/a" if not np.isfinite(adverse_ratio) else f"{adverse_ratio:.3f}x"
    print(f"  Loser / winner fill ratio   {ratio_text:>12}     <1.0 is what we want")
    print(f"  Avg queue — filled          {avg_queue_filled:>12.2f}     Avg queue — no fill       {avg_queue_no_fill:>12.2f}")
    print(f"  Zero-queue fill rate        {_fmt_pct(zero_queue_fill_rate):>12}")

    print("\n10c SALVAGE / CURRENT RISK")
    print(f"  Exit triggers / posts/fills {exit_triggers:>3} / {exit_posts:<3} / {exit_fills:<3}")
    print(f"  Open filled positions       {open_positions:>6}     Open contracts             {open_contracts:>8.2f}")
    print(f"  Open entry cost             ${open_cost:>11.4f}     (no mark-to-market included)")

    print("\nTHREAD HEALTH")
    for name, alive in thread_state.items():
        print(f"  {name:<22} {'OK' if alive else 'DEAD'}")

    if show_rows and signals:
        print("\nLATEST SIGNALS")
        cols = ["decision_time", "ticker", "direction", "midpoint", "spread_c", "entry_price", "entry_queue", "entry_fill_qty", "result", "realized_pnl", "status"]
        view = df[cols].tail(int(show_rows)).copy()
        try:
            from IPython.display import display
            display(view)
        except Exception:
            print(view.to_string(index=False))

    print("=" * 108)
    return df


start_shadow_trader = start_primary_shadow_trader
stop_shadow_trader = stop_primary_shadow_trader
shadow_status = primary_shadow_status
