from __future__ import annotations

"""Read-only schema/timestamp/side diagnostic for Q5 live fills vs public trades.

Purpose
-------
V1/V2 public-trade reconciliation returned zero matches for every real ENTRY fill.
This module deliberately removes the previous price/side compatibility assumptions
and diagnoses which field semantics disagree: timestamps, price orientation,
book-side orientation, or trade identifiers.

Safety
------
- SAME-REALIZATION forensic only.
- NO exchange/API calls.
- NO orders.
- Source session is read-only.
- Writes only under results/kalshi_q5_public_trade_schema_diagnostic_v3/.
"""

import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q5_same_realization_shadow_v1 as BASE
from . import mm_cycle_q5_public_trade_fill_reconciliation_v1 as V1

VERSION = "MM_CYCLE_Q5_PUBLIC_TRADE_SCHEMA_DIAGNOSTIC_V3"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_public_trade_schema_diagnostic_v3"
TOLS = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0)
EPS = 1e-8


def _f(x, default=np.nan):
    return OOS._f(x, default)


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _nearest_indices(times, t, k=3):
    if not times or not np.isfinite(t):
        return []
    j = bisect.bisect_left(times, t)
    cand = []
    lo = max(0, j - k)
    hi = min(len(times), j + k + 1)
    for i in range(lo, hi):
        cand.append(i)
    cand.sort(key=lambda i: abs(times[i] - t))
    return cand[:k]


def _relation(fill_px, trade_px):
    if not (np.isfinite(fill_px) and np.isfinite(trade_px)):
        return "INVALID"
    if abs(fill_px - trade_px) <= EPS:
        return "SAME_PRICE"
    if abs(fill_px - (1.0 - trade_px)) <= EPS:
        return "COMPLEMENT_PRICE"
    return "OTHER_PRICE"


def _expected_consumed_side(passive_side):
    s = str(passive_side).upper()
    if s == "BID":
        return "bid"
    if s == "ASK":
        return "ask"
    return ""


def _opposite_side(side):
    return "ask" if side == "bid" else "bid" if side == "ask" else ""


def run_q5_public_trade_schema_diagnostic(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"

    windows = BASE._live_windows(source)
    meta_rows, meta_by_ticker = BASE._metadata(raw)
    selected_tickers = {
        t for t, r in meta_by_ticker.items()
        if str(r.get("close_time") or "") in set(windows)
    }
    if not selected_tickers:
        raise RuntimeError("No selected tickers for live Q5 windows.")

    orders = V1._load_order_catalog(source)
    fills, fill_time_sources = V1._load_live_entry_fills(source, orders, selected_tickers)
    trades, by_trade_id, by_ticker = V1._load_public_trades(raw, selected_tickers)

    if not fills or not trades:
        raise RuntimeError("Need both live ENTRY fills and public trades.")

    # Build ticker-local sorted time arrays for both exchange and receipt clocks.
    ticker_exchange = {}
    ticker_receipt = {}
    for ticker, inds in by_ticker.items():
        exch_pairs = [(trades[i]["exchange_s"], i) for i in inds if np.isfinite(trades[i]["exchange_s"])]
        exch_pairs.sort(key=lambda z: z[0])
        rec_pairs = [(trades[i]["receipt_s"], i) for i in inds if np.isfinite(trades[i]["receipt_s"])]
        rec_pairs.sort(key=lambda z: z[0])
        ticker_exchange[ticker] = ([x[0] for x in exch_pairs], [x[1] for x in exch_pairs])
        ticker_receipt[ticker] = ([x[0] for x in rec_pairs], [x[1] for x in rec_pairs])

    live_ids = {str(f.get("trade_id") or "") for f in fills if str(f.get("trade_id") or "")}
    pub_ids = {str(t.get("trade_id") or "") for t in trades if str(t.get("trade_id") or "")}
    id_overlap = live_ids & pub_ids

    rows = []
    for f in fills:
        ticker = f["ticker"]
        fill_t = f["fill_time_s"]
        fill_px = f["price"]
        expected = _expected_consumed_side(f["side"])
        opposite = _opposite_side(expected)

        etimes, einds = ticker_exchange.get(ticker, ([], []))
        rtimes, rinds = ticker_receipt.get(ticker, ([], []))

        nearest_e = []
        for pos in _nearest_indices(etimes, fill_t, k=5):
            tr = trades[einds[pos]]
            nearest_e.append((abs(tr["exchange_s"] - fill_t), tr))
        nearest_r = []
        for pos in _nearest_indices(rtimes, fill_t, k=5):
            tr = trades[rinds[pos]]
            nearest_r.append((abs(tr["receipt_s"] - fill_t), tr))

        ne = nearest_e[0][1] if nearest_e else None
        nr = nearest_r[0][1] if nearest_r else None

        # Search all nearby exchange-time trades within 5s, with NO prior side/price filtering.
        same_px_same_side_dt = []
        same_px_opp_side_dt = []
        comp_px_same_side_dt = []
        comp_px_opp_side_dt = []
        any_dt = []
        if etimes:
            lo = bisect.bisect_left(etimes, fill_t - 5.0)
            hi = bisect.bisect_right(etimes, fill_t + 5.0)
            for pos in range(lo, hi):
                tr = trades[einds[pos]]
                dt = tr["exchange_s"] - fill_t
                any_dt.append(abs(dt))
                rel = _relation(fill_px, tr["price"])
                side = str(tr.get("taker_book_side") or "").lower()
                if rel == "SAME_PRICE" and side == expected:
                    same_px_same_side_dt.append(abs(dt))
                elif rel == "SAME_PRICE" and side == opposite:
                    same_px_opp_side_dt.append(abs(dt))
                elif rel == "COMPLEMENT_PRICE" and side == expected:
                    comp_px_same_side_dt.append(abs(dt))
                elif rel == "COMPLEMENT_PRICE" and side == opposite:
                    comp_px_opp_side_dt.append(abs(dt))

        rows.append({
            "fill_id": f.get("fill_id"),
            "live_trade_id": f.get("trade_id"),
            "ticker": ticker,
            "side": f["side"],
            "fill_qty": f["qty"],
            "fill_price": fill_px,
            "fill_time": pd.Timestamp(fill_t, unit="s", tz="UTC").isoformat(),
            "expected_consumed_book_side": expected,
            "nearest_exchange_dt_ms": 1000.0 * (ne["exchange_s"] - fill_t) if ne else np.nan,
            "nearest_exchange_price": ne["price"] if ne else np.nan,
            "nearest_exchange_side": ne["taker_book_side"] if ne else None,
            "nearest_exchange_trade_id": ne["trade_id"] if ne else None,
            "nearest_exchange_price_relation": _relation(fill_px, ne["price"]) if ne else None,
            "nearest_receipt_dt_ms": 1000.0 * (nr["receipt_s"] - fill_t) if nr else np.nan,
            "nearest_receipt_price": nr["price"] if nr else np.nan,
            "nearest_receipt_side": nr["taker_book_side"] if nr else None,
            "min_same_price_same_side_ms": 1000.0 * min(same_px_same_side_dt) if same_px_same_side_dt else np.nan,
            "min_same_price_opposite_side_ms": 1000.0 * min(same_px_opp_side_dt) if same_px_opp_side_dt else np.nan,
            "min_complement_price_same_side_ms": 1000.0 * min(comp_px_same_side_dt) if comp_px_same_side_dt else np.nan,
            "min_complement_price_opposite_side_ms": 1000.0 * min(comp_px_opp_side_dt) if comp_px_opp_side_dt else np.nan,
            "min_any_exchange_trade_ms": 1000.0 * min(any_dt) if any_dt else np.nan,
        })

    df = pd.DataFrame(rows)

    # Tolerance sensitivity under four competing semantic hypotheses.
    sens = []
    for tol in TOLS:
        ms = tol * 1000.0
        sens.append({
            "tolerance_s": tol,
            "any_same_ticker_time_match_qty": float(df.loc[df["min_any_exchange_trade_ms"] <= ms, "fill_qty"].sum()),
            "same_price_same_side_qty": float(df.loc[df["min_same_price_same_side_ms"] <= ms, "fill_qty"].sum()),
            "same_price_opposite_side_qty": float(df.loc[df["min_same_price_opposite_side_ms"] <= ms, "fill_qty"].sum()),
            "complement_price_same_side_qty": float(df.loc[df["min_complement_price_same_side_ms"] <= ms, "fill_qty"].sum()),
            "complement_price_opposite_side_qty": float(df.loc[df["min_complement_price_opposite_side_ms"] <= ms, "fill_qty"].sum()),
        })
    sens_df = pd.DataFrame(sens)

    def stats(series):
        a = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if not len(a):
            return {"n": 0, "median": np.nan, "p10": np.nan, "p90": np.nan, "p95": np.nan, "max_abs": np.nan}
        return {
            "n": int(len(a)),
            "median": float(np.median(a)),
            "p10": float(np.quantile(a, 0.10)),
            "p90": float(np.quantile(a, 0.90)),
            "p95": float(np.quantile(a, 0.95)),
            "max_abs": float(np.max(np.abs(a))),
        }

    out = _new_output(source.name)
    df.to_csv(out / "fill_nearest_public_trade_diagnostic.csv", index=False)
    sens_df.to_csv(out / "semantic_hypothesis_sensitivity.csv", index=False)

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "fill_rows": len(fills),
        "fill_qty": float(sum(f["qty"] for f in fills)),
        "public_trade_rows": len(trades),
        "fill_time_sources": fill_time_sources,
        "live_trade_ids": len(live_ids),
        "public_trade_ids": len(pub_ids),
        "exact_trade_id_overlap_count": len(id_overlap),
        "nearest_exchange_dt_ms": stats(df["nearest_exchange_dt_ms"]),
        "nearest_receipt_dt_ms": stats(df["nearest_receipt_dt_ms"]),
        "semantic_sensitivity": sens_df.to_dict(orient="records"),
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
    }
    (out / "schema_diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("=" * 124)
        print("Q5 PUBLIC-TRADE SCHEMA / TIMESTAMP / SIDE DIAGNOSTIC V3 — READ ONLY")
        print("=" * 124)
        print("Source:", source)
        print("Live ENTRY fills / qty:", len(fills), "/", f"{summary['fill_qty']:.4f}")
        print("Public trades:", len(trades))
        print("Exact trade-id overlap count:", len(id_overlap), "of", len(live_ids), "live trade ids")
        print("Fill time sources:", fill_time_sources)
        print()
        print("NEAREST SAME-TICKER PUBLIC TRADE, IGNORING PRICE/SIDE")
        print(" exchange_time - fill created_time ms:", summary["nearest_exchange_dt_ms"])
        print(" receipt_time  - fill created_time ms:", summary["nearest_receipt_dt_ms"])
        print()
        print("SEMANTIC HYPOTHESIS SENSITIVITY — MATCHED LIVE FILL QTY")
        print(sens_df.to_string(index=False))
        print()
        print("NEAREST 25 FILLS")
        cols = [
            "ticker", "side", "fill_qty", "fill_price",
            "nearest_exchange_dt_ms", "nearest_exchange_price",
            "nearest_exchange_side", "nearest_exchange_price_relation",
            "min_same_price_same_side_ms", "min_same_price_opposite_side_ms",
            "min_complement_price_same_side_ms", "min_complement_price_opposite_side_ms",
        ]
        print(df[cols].head(25).to_string(index=False))
        print()
        print("Interpretation:")
        print("  - High ANY-time match but low price/side match => schema semantics mismatch, not missing public trades.")
        print("  - SAME_PRICE + OPPOSITE_SIDE dominating => taker_book_side orientation is reversed relative to the live order side.")
        print("  - COMPLEMENT_PRICE dominating => YES/NO price orientation mismatch.")
        print("  - Even ANY-time match low at 5s => fill created_time and public trade timestamps are not comparable or the recorder missed fill-causing trades.")
        print("  - Do not use the earlier queue-reduction estimates until this diagnostic resolves the schema.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 124)

    return {"summary": summary, "detail": df, "sensitivity": sens_df, "output_dir": out}


__all__ = ["run_q5_public_trade_schema_diagnostic", "VERSION"]
