from __future__ import annotations

"""Read-only microstructure forensic for V13 vertical public-trade pillars.

V13 ruled out the simple receipt-batch/backfill explanation for the four widest
M1-M5 examples: the large price-range bursts also occur within only milliseconds
of *exchange* time, and many prints share the exact same millisecond timestamp.

V14 asks the next question: are these bursts consistent with true aggressive
sweeps through a very wide book, mixed-direction crossed flow, or a remaining
feed/book-timestamp inconsistency?

For each V13 pillar V14 reports:
- taker book-side trade/quantity composition;
- price range by taker side;
- exact-exchange-timestamp clustering;
- largest same-timestamp price range;
- nearest prior exchange-clock BBO for every trade;
- directional consistency vs that BBO (bid taker should execute at/above the
  prior YES ask; ask taker should execute at/below the prior YES bid, allowing
  deeper sweep prices);
- BBO movement over a narrow exchange-time event window;
- a heuristic classification intended only to guide further research.

It also prints and plots the largest burst in each selected ticker with exchange
milliseconds on the x-axis, Bid/Ask lines, and trades separated by taker side.

NO API calls. NO orders. Source session is read-only.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import recorder_core as C
from . import mm_cycle_q10_trade_pillar_forensic_v13 as V13

VERSION = "MM_CYCLE_Q10_TRADE_PILLAR_SWEEP_MICROSTRUCTURE_V14"
HARD_BOUND_SESSION = V13.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q10_trade_pillar_sweep_microstructure_v14"
EPS = 1e-9
BOOK_ASOF_TOLERANCE_MS = 250.0
PLOT_PAD_MS = 150.0


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _assign_bursts(tdf: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty:
        return tdf.copy()
    x = tdf.dropna(subset=["receipt_time"]).sort_values("receipt_time").copy()
    gap = x["receipt_time"].diff().dt.total_seconds().mul(1000.0)
    x["burst_id"] = (gap.isna() | (gap > V13.BURST_GAP_MS)).cumsum().astype(int)
    return x


def _attach_exchange_bbo(bdf: pd.DataFrame, tdf: pd.DataFrame) -> pd.DataFrame:
    if bdf.empty or tdf.empty:
        return tdf.copy()
    b = (
        bdf.dropna(subset=["exchange_time"])
        .sort_values("exchange_time")[["exchange_time", "bid", "ask", "spread_c", "event_type"]]
        .rename(columns={
            "bid": "prior_exchange_bid",
            "ask": "prior_exchange_ask",
            "spread_c": "prior_exchange_spread_c",
            "event_type": "prior_exchange_book_event_type",
        })
    )
    t = tdf.dropna(subset=["exchange_time"]).sort_values(["exchange_time", "receipt_time"]).copy()
    if b.empty or t.empty:
        return tdf.copy()
    m = pd.merge_asof(
        t,
        b,
        on="exchange_time",
        direction="backward",
        tolerance=pd.Timedelta(milliseconds=BOOK_ASOF_TOLERANCE_MS),
    )
    side = m["taker_book_side"].astype(str).str.lower()
    px = pd.to_numeric(m["price"], errors="coerce")
    bid = pd.to_numeric(m["prior_exchange_bid"], errors="coerce")
    ask = pd.to_numeric(m["prior_exchange_ask"], errors="coerce")

    # A bid taker buys YES and consumes asks, so executions should be at the
    # current ask or deeper (higher). An ask taker sells YES and consumes bids,
    # so executions should be at the current bid or deeper (lower).
    consistent = np.where(
        side.eq("bid"),
        px >= ask - EPS,
        np.where(side.eq("ask"), px <= bid + EPS, np.nan),
    )
    m["directional_vs_prior_exchange_bbo_ok"] = consistent
    m["distance_beyond_touch_c"] = np.where(
        side.eq("bid"),
        100.0 * (px - ask),
        np.where(side.eq("ask"), 100.0 * (bid - px), np.nan),
    )
    return m


def _same_timestamp_stats(g: pd.DataFrame):
    z = g.dropna(subset=["exchange_time"]).copy()
    if z.empty:
        return {
            "exchange_timestamp_groups": 0,
            "max_trades_same_exchange_ts": 0,
            "max_qty_same_exchange_ts": 0.0,
            "max_price_range_same_exchange_ts_c": np.nan,
            "same_exchange_ts_group_with_max_range": None,
        }
    rows = []
    for ts, h in z.groupby("exchange_time", sort=True):
        rows.append({
            "exchange_time": ts,
            "trades": len(h),
            "qty": float(pd.to_numeric(h["qty"], errors="coerce").fillna(0).sum()),
            "price_range_c": 100.0 * (float(h["price"].max()) - float(h["price"].min())),
        })
    q = pd.DataFrame(rows)
    imax = q["price_range_c"].idxmax()
    return {
        "exchange_timestamp_groups": int(len(q)),
        "max_trades_same_exchange_ts": int(q["trades"].max()),
        "max_qty_same_exchange_ts": float(q["qty"].max()),
        "max_price_range_same_exchange_ts_c": float(q.loc[imax, "price_range_c"]),
        "same_exchange_ts_group_with_max_range": q.loc[imax, "exchange_time"],
    }


def _side_stats(g: pd.DataFrame, side: str):
    h = g[g["taker_book_side"].astype(str).str.lower() == side]
    if h.empty:
        return {
            f"{side}_trades": 0,
            f"{side}_qty": 0.0,
            f"{side}_price_min": np.nan,
            f"{side}_price_max": np.nan,
            f"{side}_price_range_c": np.nan,
        }
    return {
        f"{side}_trades": int(len(h)),
        f"{side}_qty": float(pd.to_numeric(h["qty"], errors="coerce").fillna(0).sum()),
        f"{side}_price_min": float(h["price"].min()),
        f"{side}_price_max": float(h["price"].max()),
        f"{side}_price_range_c": 100.0 * (float(h["price"].max()) - float(h["price"].min())),
    }


def _classify(row):
    known = int(row.get("bid_trades", 0)) + int(row.get("ask_trades", 0))
    n = max(1, int(row.get("trades", 0)))
    one_side_share = max(int(row.get("bid_trades", 0)), int(row.get("ask_trades", 0))) / n
    ok = row.get("directional_vs_prior_exchange_bbo_ok_pct")
    same_range = row.get("max_price_range_same_exchange_ts_c")

    if known == 0:
        return "NO_TAKER_SIDE_DATA"
    if np.isfinite(ok) and ok < 80.0:
        return "BOOK_TRADE_CLOCK_OR_STATE_MISMATCH"
    if one_side_share >= 0.90 and np.isfinite(same_range) and same_range >= 10.0:
        return "LIKELY_ONE_SIDED_FAST_SWEEP"
    if one_side_share < 0.75 and np.isfinite(same_range) and same_range >= 10.0:
        return "MIXED_DIRECTION_ULTRA_FAST_FLOW"
    if one_side_share >= 0.90:
        return "LIKELY_ONE_SIDED_SWEEP"
    return "MIXED_DIRECTION_FAST_FLOW"


def run_trade_pillar_sweep_microstructure(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected formal OOS {HARD_BOUND_SESSION}, got {source.name}")
    for p in (source/"book_top3_events.jsonl", source/"trades_event_time.jsonl", source/"market_metadata.jsonl"):
        if not p.exists():
            raise FileNotFoundError(p)

    # Reuse V13 selection and raw field handling so V14 investigates exactly the
    # same four visual examples and exactly the same pillar definition.
    base = V13.run_trade_pillar_forensic(source, hard_bind=hard_bind, show=False)
    selected = base["selected"].copy()
    v13_bursts = base["bursts"].copy()
    if v13_bursts.empty:
        raise RuntimeError("V13 found no pillars to investigate")

    out = _new_output(source.name)
    detail_rows = []
    burst_trade_frames = []
    largest_by_ticker = {}

    for _, info in selected.iterrows():
        ticker = str(info["ticker"])
        bdf = pd.DataFrame(base["books"].get(ticker, [])).sort_values("receipt_time")
        tdf = base["trades"].get(ticker, pd.DataFrame()).copy()
        if tdf.empty:
            continue
        tdf = _assign_bursts(tdf)
        tdf = _attach_exchange_bbo(bdf, tdf)

        target_ids = set(
            pd.to_numeric(
                v13_bursts.loc[v13_bursts["ticker"].eq(ticker), "burst_id"],
                errors="coerce",
            ).dropna().astype(int)
        )

        for burst_id in sorted(target_ids):
            g = tdf[tdf["burst_id"].eq(burst_id)].copy()
            if g.empty:
                continue
            g = g.sort_values(["exchange_time", "receipt_time", "trade_id"])
            g["ticker"] = ticker
            g["close_time"] = info["close_time"]
            g["series"] = info["series"]
            burst_trade_frames.append(g)

            ts_stats = _same_timestamp_stats(g)
            side_bid = _side_stats(g, "bid")
            side_ask = _side_stats(g, "ask")
            side_unknown = int((~g["taker_book_side"].astype(str).str.lower().isin(["bid", "ask"])).sum())

            ok = pd.to_numeric(g.get("directional_vs_prior_exchange_bbo_ok"), errors="coerce")
            dist = pd.to_numeric(g.get("distance_beyond_touch_c"), errors="coerce")
            valid_bbo = g[["prior_exchange_bid", "prior_exchange_ask"]].notna().all(axis=1)

            r0, r1 = g["receipt_time"].min(), g["receipt_time"].max()
            x0, x1 = g["exchange_time"].min(), g["exchange_time"].max()

            row = {
                "ticker": ticker,
                "series": info["series"],
                "close_time": info["close_time"],
                "burst_id": int(burst_id),
                "trades": int(len(g)),
                "qty": float(pd.to_numeric(g["qty"], errors="coerce").fillna(0).sum()),
                "receipt_start": r0,
                "receipt_span_ms": float((r1-r0).total_seconds()*1000.0),
                "exchange_start": x0,
                "exchange_end": x1,
                "exchange_span_ms": float((x1-x0).total_seconds()*1000.0) if not pd.isna(x0) and not pd.isna(x1) else np.nan,
                "price_min": float(g["price"].min()),
                "price_max": float(g["price"].max()),
                "price_range_c": 100.0 * (float(g["price"].max()) - float(g["price"].min())),
                "unique_trade_ids": int(g["trade_id"].nunique()),
                "unknown_side_trades": side_unknown,
                "prior_exchange_bbo_coverage_pct": 100.0 * float(valid_bbo.mean()),
                "directional_vs_prior_exchange_bbo_ok_pct": 100.0 * float(ok.dropna().mean()) if len(ok.dropna()) else np.nan,
                "median_distance_beyond_touch_c": float(dist.dropna().median()) if len(dist.dropna()) else np.nan,
                "p95_distance_beyond_touch_c": float(dist.dropna().quantile(.95)) if len(dist.dropna()) else np.nan,
                **side_bid,
                **side_ask,
                **ts_stats,
            }
            row["classification"] = _classify(row)
            detail_rows.append(row)

        tg = v13_bursts[v13_bursts["ticker"].eq(ticker)]
        if not tg.empty:
            largest_by_ticker[ticker] = int(tg.sort_values(["price_range_c", "trades"], ascending=False).iloc[0]["burst_id"])

    detail = pd.DataFrame(detail_rows).sort_values(["price_range_c", "trades"], ascending=False)
    burst_trades = pd.concat(burst_trade_frames, ignore_index=True) if burst_trade_frames else pd.DataFrame()

    detail.to_csv(out / "pillar_microstructure_summary.csv", index=False)
    burst_trades.to_csv(out / "pillar_trade_detail.csv", index=False)

    if show:
        print("="*156)
        print("Q10 TRADE-PILLAR SWEEP MICROSTRUCTURE V14 — READ ONLY")
        print("="*156)
        print("Source:", source)
        print("Pillars:", len(detail))
        print()
        cols = [
            "ticker", "burst_id", "trades", "qty", "exchange_span_ms", "price_range_c",
            "bid_trades", "bid_qty", "bid_price_min", "bid_price_max",
            "ask_trades", "ask_qty", "ask_price_min", "ask_price_max",
            "max_trades_same_exchange_ts", "max_price_range_same_exchange_ts_c",
            "prior_exchange_bbo_coverage_pct", "directional_vs_prior_exchange_bbo_ok_pct",
            "median_distance_beyond_touch_c", "classification",
        ]
        print("MICROSTRUCTURE SUMMARY")
        print(detail[cols].to_string(index=False))
        print("\nClassification counts:")
        print(detail["classification"].value_counts().to_string())

        for _, info in selected.iterrows():
            ticker = str(info["ticker"])
            burst_id = largest_by_ticker.get(ticker)
            if burst_id is None:
                continue
            g = burst_trades[(burst_trades["ticker"].eq(ticker)) & (burst_trades["burst_id"].eq(burst_id))].copy()
            if g.empty:
                continue
            bdf = pd.DataFrame(base["books"].get(ticker, [])).dropna(subset=["exchange_time"]).sort_values("exchange_time")
            t0, t1 = g["exchange_time"].min(), g["exchange_time"].max()
            if pd.isna(t0) or pd.isna(t1):
                continue
            p0 = t0 - pd.Timedelta(milliseconds=PLOT_PAD_MS)
            p1 = t1 + pd.Timedelta(milliseconds=PLOT_PAD_MS)
            bz = bdf[(bdf["exchange_time"] >= p0) & (bdf["exchange_time"] <= p1)].copy()
            origin = t0

            print("\n" + "="*156)
            print(f"LARGEST PILLAR DETAIL — {ticker} burst={burst_id}")
            print("="*156)
            show_cols = [
                "exchange_time", "receipt_time", "trade_id", "taker_book_side", "price", "qty",
                "prior_exchange_bid", "prior_exchange_ask", "prior_exchange_spread_c",
                "directional_vs_prior_exchange_bbo_ok", "distance_beyond_touch_c",
            ]
            print(g[show_cols].head(80).to_string(index=False))
            if len(g) > 80:
                print(f"... {len(g)-80} additional trades written to pillar_trade_detail.csv")

            fig, ax = plt.subplots(figsize=(15, 6))
            if not bz.empty:
                bx = (bz["exchange_time"] - origin).dt.total_seconds() * 1000.0
                ax.step(bx, bz["bid"], where="post", linewidth=1.2, label="YES Bid")
                ax.step(bx, bz["ask"], where="post", linewidth=1.2, label="YES Ask")
            for side, marker in (("bid", "^"), ("ask", "v")):
                h = g[g["taker_book_side"].astype(str).str.lower().eq(side)]
                if h.empty:
                    continue
                tx = (h["exchange_time"] - origin).dt.total_seconds() * 1000.0
                ax.scatter(tx, h["price"], s=24, alpha=.7, marker=marker, label=f"Trades taker={side}")
            u = g[~g["taker_book_side"].astype(str).str.lower().isin(["bid", "ask"])]
            if not u.empty:
                tx = (u["exchange_time"] - origin).dt.total_seconds() * 1000.0
                ax.scatter(tx, u["price"], s=20, alpha=.7, marker="o", label="Trades side=unknown")
            ax.axvline(0.0, linestyle="--", alpha=.35)
            ax.grid(alpha=.2)
            ax.legend()
            ax.set_xlabel("Exchange milliseconds from first trade in pillar")
            ax.set_ylabel("YES price")
            ax.set_title(f"EXCHANGE-TIME MICROSTRUCTURE — {ticker} burst {burst_id}")
            plt.tight_layout(); plt.show()

        print("\nInterpretation guide:")
        print("- LIKELY_ONE_SIDED_FAST_SWEEP: >=90% one taker side with large same-ms price sweep and exchange-BBO consistency.")
        print("- MIXED_DIRECTION_ULTRA_FAST_FLOW: both taker directions materially present inside the same ultra-fast price-range event.")
        print("- BOOK_TRADE_CLOCK_OR_STATE_MISMATCH: many trades cannot be reconciled to the nearest prior exchange-clock BBO; investigate feed ordering/state before economic interpretation.")
        print("- This is a same-realization forensic, not a strategy or profitability test.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("="*156)

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "pillars": int(len(detail)),
        "classifications": detail["classification"].value_counts().to_dict(),
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "guardrail": "Same-realization market-data forensic only. No strategy inference is valid until book/trade event consistency is established.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return {
        "summary": summary,
        "detail": detail,
        "burst_trades": burst_trades,
        "selected": selected,
        "output_dir": out,
    }


__all__ = ["VERSION", "run_trade_pillar_sweep_microstructure"]
