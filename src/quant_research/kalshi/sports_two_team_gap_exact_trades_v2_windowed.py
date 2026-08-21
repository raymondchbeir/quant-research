from __future__ import annotations

"""Targeted exact-trade verification around minute quote-mid crossings.

Public GET only. No account/order endpoints, no orders, no live-Q50 imports.
Results checkpoint after every market and resume safely.
"""

import json, math, os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .sports_two_team_gap_backfill_v1 import PublicKalshi

VERSION = "KALSHI_SPORTS_TWO_TEAM_GAP_EXACT_TRADES_V2_WINDOWED_CHECKPOINTED"


def _ts(x: Any) -> float | None:
    if x is None or x == "": return None
    if isinstance(x, (int, float)):
        try:
            z = float(x); return z if math.isfinite(z) else None
        except Exception: return None
    try: return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
    except Exception: return None


def _price(row: dict[str, Any]) -> float | None:
    for k in ("yes_price_dollars", "yes_price"):
        try:
            z = float(row.get(k))
            if k == "yes_price" and z > 1: z /= 100.0
            if math.isfinite(z) and 0 <= z <= 1: return z
        except Exception: pass
    return None


def _state(p: float, deadband: float) -> int:
    if p >= 0.5 + deadband: return 1
    if p <= 0.5 - deadband: return -1
    return 0


def _compress(states: list[int]) -> list[int]:
    out = []
    for s in states:
        if s and (not out or out[-1] != s): out.append(int(s))
    return out


def _fetch(client: PublicKalshi, path: str, ticker: str, lo: int, hi: int) -> list[dict[str, Any]]:
    try:
        return client.paged(path, "trades", {
            "ticker": ticker, "min_ts": int(lo), "max_ts": int(hi),
            "is_block_trade": "false",
        }, limit=1000)
    except Exception:
        return []


def fetch_window_trades(client: PublicKalshi, ticker: str, lo: int, hi: int) -> list[dict[str, Any]]:
    rows = [
        *_fetch(client, "/historical/trades", ticker, lo, hi),
        *_fetch(client, "/markets/trades", ticker, lo, hi),
    ]
    by_id = {}
    for i, row in enumerate(rows):
        tid = str(row.get("trade_id") or f"fallback-{i}-{row.get('created_time')}-{row.get('yes_price_dollars', row.get('yes_price'))}")
        by_id[tid] = row
    return list(by_id.values())


def _quote_crosses(paths: pd.DataFrame, deadband: float, max_spread: float) -> pd.DataFrame:
    p = paths.copy()
    for c in ("yes_mid", "quote_spread", "end_period_ts", "elapsed_from_start_s"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[p.yes_mid.between(0, 1) & p.quote_spread.between(0, max_spread)].copy()
    p = p[((p.phase == "pregame") & p.elapsed_from_start_s.between(-86400, 0, inclusive="left")) |
          ((p.phase == "in_game") & (p.elapsed_from_start_s >= 0))].copy()

    out = []
    for (ticker, phase), g in p.groupby(["ticker", "phase"], sort=False):
        g = g.sort_values("end_period_ts")
        last_state = 0; last_ts = None; last_mid = None; idx = 0
        for r in g.itertuples(index=False):
            mid = float(r.yes_mid); ts = int(r.end_period_ts); state = _state(mid, deadband)
            if not state: continue
            if last_state and state != last_state and last_ts is not None:
                idx += 1
                out.append({
                    "quote_cross_id": f"{ticker}|{phase}|{idx}",
                    "ticker": str(ticker), "event_ticker": str(r.event_ticker),
                    "series_ticker": str(r.series_ticker), "sport": str(r.sport), "phase": str(phase),
                    "quote_cross_index": idx, "quote_from_state": last_state, "quote_to_state": state,
                    "quote_prev_ts": int(last_ts), "quote_cross_ts": ts,
                    "quote_prev_mid": float(last_mid), "quote_cross_mid": mid,
                })
            last_state = state; last_ts = ts; last_mid = mid
    return pd.DataFrame(out)


def _merged_windows(ev: pd.DataFrame, pre: int, post: int) -> list[tuple[int, int]]:
    raw = sorted((int(min(r.quote_prev_ts, r.quote_cross_ts)-pre), int(max(r.quote_prev_ts, r.quote_cross_ts)+post))
                 for r in ev.itertuples(index=False))
    merged = []
    for lo, hi in raw:
        if not merged or lo > merged[-1][1] + 1: merged.append([lo, hi])
        else: merged[-1][1] = max(merged[-1][1], hi)
    return [(a, b) for a, b in merged]


def _parse(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    out = []
    for row in rows:
        t = _ts(row.get("created_time")); p = _price(row)
        if t is not None and p is not None: out.append((float(t), float(p)))
    return sorted(out)


def _eval(event: Any, trades: list[tuple[float, float]], deadband: float, pre: int, post: int) -> dict[str, Any]:
    lo = int(min(event.quote_prev_ts, event.quote_cross_ts)-pre)
    hi = int(max(event.quote_prev_ts, event.quote_cross_ts)+post)
    sub = [x for x in trades if lo <= x[0] <= hi]
    prices = [x[1] for x in sub]; states = [_state(p, deadband) for p in prices]; seq = _compress(states)
    fr = int(event.quote_from_state); to = int(event.quote_to_state)
    verified = any(a == fr and b == to for a, b in zip(seq, seq[1:]))
    nz = [s for s in states if s]
    if verified: status = "verified_directed_trade_flip"
    elif not sub: status = "no_trades_in_window"
    elif not nz: status = "trades_only_inside_50_deadband"
    elif fr in nz and to in nz: status = "both_sides_seen_order_unverified"
    elif fr in nz: status = "only_quote_from_side_seen"
    elif to in nz: status = "only_quote_to_side_seen"
    else: status = "neither_quote_side_seen"
    return {
        "quote_cross_id": event.quote_cross_id, "ticker": event.ticker,
        "event_ticker": event.event_ticker, "series_ticker": event.series_ticker,
        "sport": event.sport, "phase": event.phase,
        "quote_cross_index": int(event.quote_cross_index),
        "quote_from_state": fr, "quote_to_state": to,
        "quote_prev_ts": int(event.quote_prev_ts), "quote_cross_ts": int(event.quote_cross_ts),
        "quote_prev_mid": float(event.quote_prev_mid), "quote_cross_mid": float(event.quote_cross_mid),
        "window_start_ts": lo, "window_end_ts": hi, "trade_count": len(sub),
        "trade_min_yes": min(prices) if prices else None, "trade_max_yes": max(prices) if prices else None,
        "trade_compressed_states": ",".join(map(str, seq)),
        "trade_verified_flip": bool(verified), "verification_status": status,
    }


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".tmp"); df.to_csv(tmp, index=False); os.replace(tmp, path)


def _atomic_json(obj: dict[str, Any], path: Path) -> None:
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(obj, indent=2, default=str)); os.replace(tmp, path)


def static_self_check(show: bool = True) -> dict[str, Any]:
    out = {
        "version": VERSION, "public_trade_get_only": True, "non_block_trades_only": True,
        "windowed_around_quote_crossings": True, "merged_windows_per_market": True,
        "checkpoint_after_each_market": True, "resume_supported": True,
        "full_day_trade_download": False, "account_endpoints_used": False,
        "orders_sent": False, "live_q50_modules_imported": False, "ok": True,
    }
    if show:
        print("="*120); print("SPORTS EXACT TRADE V2 WINDOWED CHECK — PUBLIC READ ONLY"); print("="*120)
        for k, v in out.items(): print(f"{k:48s}: {v}")
    return out


def run(*, run_dir: str | Path, requests_per_second: float = 1.0,
        cross_deadband: float = 0.005, max_quote_spread: float = 0.20,
        pre_pad_s: int = 300, post_pad_s: int = 300, max_markets: int = 250,
        resume: bool = True, show: bool = True) -> dict[str, Any]:
    static_self_check(show=show)
    root = Path(run_dir).expanduser().resolve()
    paths = pd.read_csv(root/"minute_paths.csv.gz", compression="gzip")
    events = _quote_crosses(paths, cross_deadband, max_quote_spread)
    tickers = events.ticker.drop_duplicates().astype(str).tolist()
    if max_markets: tickers = tickers[:int(max_markets)]
    events = events[events.ticker.isin(tickers)].copy()
    _atomic_csv(events, root/"exact_trade_quote_cross_events.csv")

    checkpoint = root/"exact_trade_windowed_checkpoint.csv"
    existing = pd.read_csv(checkpoint) if resume and checkpoint.exists() else pd.DataFrame()
    completed = set(existing.ticker.dropna().astype(str).unique()) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    client = PublicKalshi(requests_per_second=float(requests_per_second))

    if show:
        print("\n"+"="*120); print("WINDOWED EXACT-TRADE REFINEMENT"); print("="*120)
        print("quote_cross_events:", len(events)); print("candidate_markets:", len(tickers)); print("already_completed:", len(completed))

    for ticker in tickers:
        if ticker in completed: continue
        ev = events[events.ticker == ticker].sort_values("quote_cross_ts")
        windows = _merged_windows(ev, pre_pad_s, post_pad_s)
        fetched = []
        for lo, hi in windows: fetched.extend(fetch_window_trades(client, ticker, lo, hi))
        by_id = {}
        for i, row in enumerate(fetched): by_id[str(row.get("trade_id") or f"f-{i}-{row.get('created_time')}")] = row
        trades = _parse(list(by_id.values()))
        for event in ev.itertuples(index=False): rows.append(_eval(event, trades, cross_deadband, pre_pad_s, post_pad_s))
        cur = pd.DataFrame(rows); _atomic_csv(cur, checkpoint)
        done = cur.ticker.nunique()
        progress = {
            "version": VERSION, "candidate_markets": len(tickers), "completed_markets": int(done),
            "remaining_markets": int(len(tickers)-done), "quote_cross_events": len(events),
            "checkpoint_rows": len(cur), "api_calls": client.calls, "api_retries": client.retries,
            "last_completed_ticker": ticker, "orders_sent": False,
        }
        _atomic_json(progress, root/"exact_trade_windowed_progress.json")
        if show:
            v = int(cur.loc[cur.ticker == ticker, "trade_verified_flip"].astype(bool).sum())
            print(f"  exact-window markets {done}/{len(tickers)} | {ticker} | crosses={len(ev)} | merged_windows={len(windows)} | verified={v} | api_calls={client.calls} retries={client.retries}")

    out = pd.DataFrame(rows); _atomic_csv(out, root/"exact_trade_crossing_windows.csv")
    if out.empty: summary = pd.DataFrame()
    else:
        summary = out.groupby(["ticker","event_ticker","series_ticker","sport","phase"], dropna=False).agg(
            quote_cross_events=("quote_cross_id","count"), verified_trade_flips=("trade_verified_flip","sum"),
            windows_with_trades=("trade_count", lambda s: int((s>0).sum())), total_trades_in_windows=("trade_count","sum")
        ).reset_index()
    _atomic_csv(summary, root/"exact_trade_crossing_summary_v2.csv")
    verified = int(out.trade_verified_flip.astype(bool).sum()) if not out.empty else 0
    with_trades = int((out.trade_count>0).sum()) if not out.empty else 0
    headline = {
        "version": VERSION, "quote_cross_events": len(out), "markets_refined": int(out.ticker.nunique()) if not out.empty else 0,
        "windows_with_trades": with_trades, "verified_quote_cross_events": verified,
        "verified_fraction_all_quote_crosses": verified/len(out) if len(out) else None,
        "verified_fraction_windows_with_trades": verified/with_trades if with_trades else None,
        "api_calls": client.calls, "api_retries": client.retries, "requests_per_second": requests_per_second,
        "cross_deadband": cross_deadband, "max_quote_spread": max_quote_spread,
        "pre_pad_s": pre_pad_s, "post_pad_s": post_pad_s, "orders_sent": False,
        "note": "Trade-price verification around quote-mid crossings; unverified may be sparse prints; not an execution/PnL backtest.",
    }
    _atomic_json(headline, root/"exact_trade_windowed_headline.json")
    if show:
        print("\n"+"="*120); print("WINDOWED EXACT-TRADE REFINEMENT COMPLETE"); print("="*120)
        for k, v in headline.items(): print(f"{k:42s}: {v}")
        if not out.empty: print("\n"+out.verification_status.value_counts().to_string())
    return headline


__all__ = ["VERSION", "static_self_check", "run"]
