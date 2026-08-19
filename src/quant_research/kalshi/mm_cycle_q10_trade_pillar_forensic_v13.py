from __future__ import annotations

"""Read-only forensic for vertical public-trade 'pillars' in formal Q10 OOS plots.

The V5 recorder stores trade elapsed_s from LOCAL RECEIPT time. If a burst of
historical/public trades is delivered together, many trades with different true exchange
times can appear at essentially one x-coordinate when plotted against elapsed_s.

This script tests that directly for four distinct M1-M5 windows with spread >2c:
- select the four widest distinct windows;
- compare trade receipt-time vs exchange-time placement;
- detect dense receipt-time bursts with large price range;
- measure the exchange-time span hidden inside each receipt burst;
- compare each trade to the latest receipt-time BBO;
- plot each window twice: receipt-clock and exchange-clock.

NO exchange/API calls. NO orders. Source session is read-only.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "MM_CYCLE_Q10_TRADE_PILLAR_FORENSIC_V13"
HARD_BOUND_SESSION = "20260817_064143"
M1_S = 60.0
M5_S = 300.0
SPREAD_THRESHOLD_C = 2.0
N_WINDOWS = 4
BURST_GAP_MS = 25.0
BURST_MIN_TRADES = 8
BURST_MIN_RANGE_C = 10.0
EPS = 1e-12


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict):
                yield r


def _ts(x):
    z = pd.to_datetime(x, utc=True, errors="coerce")
    return z


def _f(x):
    try:
        z = float(x)
        return z if np.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _meta(source: Path):
    out = {}
    for r in _iter_jsonl(source / "market_metadata.jsonl"):
        t = str(r.get("ticker") or "")
        if t:
            out[t] = r
    return out


def _elapsed_from_exchange(row):
    x = _ts(row.get("exchange_time"))
    close = _ts(row.get("close_time"))
    if pd.isna(x) or pd.isna(close):
        return np.nan
    start = close - pd.Timedelta(minutes=15)
    return (x - start).total_seconds()


def _select_windows(source: Path, meta):
    best = {}
    for r in _iter_jsonl(source / "book_top3_events.jsonl"):
        t = str(r.get("ticker") or "")
        if t not in meta:
            continue
        e = _f(r.get("elapsed_s"))
        bid, ask = _f(r.get("yes_bid")), _f(r.get("yes_ask"))
        if not (np.isfinite(e) and M1_S <= e < M5_S):
            continue
        if not (np.isfinite(bid) and np.isfinite(ask) and 0 <= bid < ask <= 1):
            continue
        sp = 100.0 * (ask - bid)
        if sp <= SPREAD_THRESHOLD_C:
            continue
        close = str(meta[t].get("close_time") or r.get("close_time") or "")
        old = best.get(close)
        if old is None or sp > old["max_spread_c"]:
            best[close] = {
                "close_time": close,
                "ticker": t,
                "series": str(meta[t].get("series_ticker") or r.get("series_ticker") or ""),
                "max_spread_c": float(sp),
            }
    return sorted(best.values(), key=lambda z: z["max_spread_c"], reverse=True)[:N_WINDOWS]


def _collect(source: Path, selected):
    tickers = {x["ticker"] for x in selected}
    books = defaultdict(list)
    trades = defaultdict(list)

    for r in _iter_jsonl(source / "book_top3_events.jsonl"):
        t = str(r.get("ticker") or "")
        if t not in tickers:
            continue
        er = _f(r.get("elapsed_s"))
        bid, ask = _f(r.get("yes_bid")), _f(r.get("yes_ask"))
        if not (np.isfinite(er) and M1_S <= er < M5_S):
            continue
        if not (np.isfinite(bid) and np.isfinite(ask) and 0 <= bid < ask <= 1):
            continue
        books[t].append({
            "receipt_time": _ts(r.get("receipt_time")),
            "exchange_time": _ts(r.get("exchange_time")),
            "receipt_elapsed_s": er,
            "exchange_elapsed_s": _elapsed_from_exchange(r),
            "bid": bid,
            "ask": ask,
            "spread_c": 100.0 * (ask - bid),
            "event_type": str(r.get("event_type") or ""),
        })

    for r in _iter_jsonl(source / "trades_event_time.jsonl"):
        t = str(r.get("ticker") or "")
        if t not in tickers:
            continue
        er = _f(r.get("elapsed_s"))
        px, qty = _f(r.get("yes_price")), _f(r.get("qty"))
        if not (np.isfinite(er) and M1_S <= er < M5_S):
            continue
        if not (np.isfinite(px) and 0 <= px <= 1 and np.isfinite(qty) and qty > 0):
            continue
        rt = _ts(r.get("receipt_time"))
        xt = _ts(r.get("exchange_time"))
        trades[t].append({
            "receipt_time": rt,
            "exchange_time": xt,
            "receipt_elapsed_s": er,
            "exchange_elapsed_s": _elapsed_from_exchange(r),
            "price": px,
            "qty": qty,
            "trade_id": str(r.get("trade_id") or ""),
            "taker_book_side": str(r.get("taker_book_side") or ""),
            "receipt_minus_exchange_ms": (rt - xt).total_seconds() * 1000.0 if not pd.isna(rt) and not pd.isna(xt) else np.nan,
        })

    return books, trades


def _attach_receipt_bbo(bdf, tdf):
    if bdf.empty or tdf.empty:
        return tdf
    b = bdf.dropna(subset=["receipt_time"]).sort_values("receipt_time")[["receipt_time", "bid", "ask"]]
    t = tdf.dropna(subset=["receipt_time"]).sort_values("receipt_time")
    if b.empty or t.empty:
        return tdf
    m = pd.merge_asof(t, b, on="receipt_time", direction="backward", tolerance=pd.Timedelta(seconds=2))
    m["outside_receipt_bbo"] = (m["price"] < m["bid"] - EPS) | (m["price"] > m["ask"] + EPS)
    m["distance_outside_bbo_c"] = np.where(
        m["price"] < m["bid"], 100.0 * (m["bid"] - m["price"]),
        np.where(m["price"] > m["ask"], 100.0 * (m["price"] - m["ask"]), 0.0),
    )
    return m


def _burst_table(ticker, tdf):
    if tdf.empty:
        return pd.DataFrame()
    x = tdf.dropna(subset=["receipt_time"]).sort_values("receipt_time").copy()
    if x.empty:
        return pd.DataFrame()
    gap = x["receipt_time"].diff().dt.total_seconds().mul(1000.0)
    x["burst_id"] = (gap.isna() | (gap > BURST_GAP_MS)).cumsum()
    rows = []
    for bid, g in x.groupby("burst_id"):
        n = len(g)
        pr = 100.0 * (g["price"].max() - g["price"].min())
        if n < BURST_MIN_TRADES or pr < BURST_MIN_RANGE_C:
            continue
        r0, r1 = g["receipt_time"].min(), g["receipt_time"].max()
        xe = g["exchange_time"].dropna()
        x0 = xe.min() if len(xe) else pd.NaT
        x1 = xe.max() if len(xe) else pd.NaT
        receipt_span_ms = (r1-r0).total_seconds()*1000.0
        exchange_span_ms = (x1-x0).total_seconds()*1000.0 if not pd.isna(x0) and not pd.isna(x1) else np.nan
        outside = pd.to_numeric(g.get("outside_receipt_bbo"), errors="coerce") if "outside_receipt_bbo" in g else pd.Series(dtype=float)
        label = "LIKELY_RECEIPT_BATCH_BACKFILL" if receipt_span_ms <= 250 and np.isfinite(exchange_span_ms) and exchange_span_ms >= 2000 else "UNKNOWN_OR_TRUE_FAST_MOVE"
        rows.append({
            "ticker": ticker,
            "burst_id": int(bid),
            "trades": n,
            "qty": float(g["qty"].sum()),
            "receipt_start": r0,
            "receipt_span_ms": receipt_span_ms,
            "receipt_elapsed_min": float(g["receipt_elapsed_s"].min()/60.0),
            "receipt_elapsed_max": float(g["receipt_elapsed_s"].max()/60.0),
            "exchange_start": x0,
            "exchange_end": x1,
            "exchange_span_ms": exchange_span_ms,
            "exchange_elapsed_min": float(pd.to_numeric(g["exchange_elapsed_s"], errors="coerce").min()/60.0),
            "exchange_elapsed_max": float(pd.to_numeric(g["exchange_elapsed_s"], errors="coerce").max()/60.0),
            "price_min": float(g["price"].min()),
            "price_max": float(g["price"].max()),
            "price_range_c": pr,
            "unique_trade_ids": int(g["trade_id"].nunique()),
            "median_receipt_minus_exchange_ms": float(pd.to_numeric(g["receipt_minus_exchange_ms"], errors="coerce").median()),
            "outside_receipt_bbo_pct": float(100.0 * outside.mean()) if len(outside) else np.nan,
            "classification": label,
        })
    return pd.DataFrame(rows).sort_values(["price_range_c", "trades"], ascending=False) if rows else pd.DataFrame()


def run_trade_pillar_forensic(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected formal OOS {HARD_BOUND_SESSION}, got {source.name}")
    for p in (source/"book_top3_events.jsonl", source/"trades_event_time.jsonl", source/"market_metadata.jsonl"):
        if not p.exists():
            raise FileNotFoundError(p)

    meta = _meta(source)
    selected = _select_windows(source, meta)
    if len(selected) < N_WINDOWS:
        raise RuntimeError(f"Only found {len(selected)} eligible distinct windows")
    books, trades = _collect(source, selected)

    all_bursts = []
    trade_tables = {}
    for info in selected:
        t = info["ticker"]
        bdf = pd.DataFrame(books[t]).sort_values("receipt_time")
        tdf = pd.DataFrame(trades[t]).sort_values("receipt_time")
        tdf = _attach_receipt_bbo(bdf, tdf)
        trade_tables[t] = tdf
        bt = _burst_table(t, tdf)
        if not bt.empty:
            bt["close_time"] = info["close_time"]
            bt["series"] = info["series"]
            all_bursts.append(bt)

        if show:
            print("\n" + "="*130)
            print(f"{info['series']} | {t} | close={info['close_time']} | max spread={info['max_spread_c']:.2f}c")
            print("="*130)
            print(f"book rows={len(bdf):,} | trades={len(tdf):,}")
            if not tdf.empty:
                print("trade receipt-exchange lag ms median/p95:",
                      round(float(pd.to_numeric(tdf['receipt_minus_exchange_ms'], errors='coerce').median()),3), "/",
                      round(float(pd.to_numeric(tdf['receipt_minus_exchange_ms'], errors='coerce').quantile(.95)),3))
                if "outside_receipt_bbo" in tdf:
                    print("trades outside latest receipt-time BBO:", f"{100.0*tdf['outside_receipt_bbo'].mean():.2f}%")
            print("detected pillars:")
            print(bt.to_string(index=False) if not bt.empty else "  none by configured threshold")

            # Receipt-clock plot
            fig, ax = plt.subplots(figsize=(14,5))
            ax.step(bdf["receipt_elapsed_s"]/60.0, bdf["bid"], where="post", label="YES Bid")
            ax.step(bdf["receipt_elapsed_s"]/60.0, bdf["ask"], where="post", label="YES Ask")
            ax.scatter(tdf["receipt_elapsed_s"]/60.0, tdf["price"], s=10, alpha=.55, label="Trades (receipt clock)")
            ax.set_xlim(1,5); ax.set_ylim(0,1); ax.grid(alpha=.2); ax.legend()
            ax.set_title(f"RECEIPT CLOCK — {t}")
            ax.set_xlabel("Minutes after contract open"); ax.set_ylabel("YES price")
            plt.tight_layout(); plt.show()

            # Exchange-clock plot; use only book rows carrying exchange_time.
            bx = bdf.dropna(subset=["exchange_elapsed_s"]).sort_values("exchange_elapsed_s")
            tx = tdf.dropna(subset=["exchange_elapsed_s"]).sort_values("exchange_elapsed_s")
            fig, ax = plt.subplots(figsize=(14,5))
            if not bx.empty:
                ax.step(bx["exchange_elapsed_s"]/60.0, bx["bid"], where="post", label="YES Bid (book exchange clock)")
                ax.step(bx["exchange_elapsed_s"]/60.0, bx["ask"], where="post", label="YES Ask (book exchange clock)")
            ax.scatter(tx["exchange_elapsed_s"]/60.0, tx["price"], s=10, alpha=.55, label="Trades (exchange clock)")
            ax.set_xlim(1,5); ax.set_ylim(0,1); ax.grid(alpha=.2); ax.legend()
            ax.set_title(f"EXCHANGE CLOCK — {t}")
            ax.set_xlabel("Minutes after contract open"); ax.set_ylabel("YES price")
            plt.tight_layout(); plt.show()

    bursts = pd.concat(all_bursts, ignore_index=True) if all_bursts else pd.DataFrame()
    if show:
        print("\n" + "="*130)
        print("ALL DETECTED PILLARS")
        print("="*130)
        print(bursts.to_string(index=False) if not bursts.empty else "none")
        if not bursts.empty:
            print("\nClassification counts:")
            print(bursts["classification"].value_counts().to_string())
            print("\nInterpretation:")
            print("- LIKELY_RECEIPT_BATCH_BACKFILL: trades arrived in <=250ms locally but span >=2s of exchange time.")
            print("- If a vertical receipt-clock pillar spreads out on the exchange-clock plot, it is not a true instantaneous price jump.")
            print("- Same-realization forensic only; this does not create a new strategy signal by itself.")
        print("\nSOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")

    return {"selected": pd.DataFrame(selected), "bursts": bursts, "trades": trade_tables, "books": books, "version": VERSION}


__all__ = ["VERSION", "run_trade_pillar_forensic"]
