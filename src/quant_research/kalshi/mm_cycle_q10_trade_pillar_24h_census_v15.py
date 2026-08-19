from __future__ import annotations

"""24-hour census of ultra-fast public-trade pillars in the formal Q10 OOS capture.

This extends V13's descriptive pillar definition across the entire untouched
formal OOS realization ``20260817_064143``.

Operational pillar definition (unchanged from V13)
--------------------------------------------------
Within one market ticker and M1 <= elapsed < M5:
- consecutive public trades separated by <= 25 ms on LOCAL RECEIPT time;
- burst contains >= 8 trades;
- YES trade-price range across the burst is >= 10 cents.

The script counts BOTH:
1) every pillar meeting the V13 definition; and
2) the subset whose latest observed BBO at burst start had spread > 2 cents.

This lets us answer the user's original visual question without silently assuming
that every fast burst occurred in a wide-spread state.

Outputs
-------
- pillar_detail.csv: one row per detected pillar;
- pillar_count_by_window.csv: all 15-minute close-time windows, including zeros;
- pillar_count_by_asset.csv;
- pillar_count_distribution.csv;
- summary.json.

Scientific guardrails
---------------------
- SAME historical realization / exploratory forensic only.
- This does not define or validate a new trading strategy.
- No thresholds are tuned here; V13's thresholds are reused verbatim.
- NO exchange/API calls and NO orders.
- Source capture is read-only.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C

VERSION = "MM_CYCLE_Q10_TRADE_PILLAR_24H_CENSUS_V15"
HARD_BOUND_SESSION = "20260817_064143"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q10_trade_pillar_24h_census_v15"

M1_S = 60.0
M5_S = 300.0
BURST_GAP_MS = 25.0
BURST_MIN_TRADES = 8
BURST_MIN_RANGE_C = 10.0
WIDE_SPREAD_C = 2.0
BBO_MAX_AGE_S = 2.0
EPS = 1e-12


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _ts(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _metadata(source: Path):
    rows = []
    by_ticker = {}
    for r in _iter_jsonl(source / "market_metadata.jsonl"):
        ticker = str(r.get("ticker") or "")
        if not ticker:
            continue
        z = {
            "ticker": ticker,
            "series": str(r.get("series_ticker") or ""),
            "close_time": str(r.get("close_time") or ""),
        }
        rows.append(z)
        by_ticker[ticker] = z
    return rows, by_ticker


def _new_burst(row, receipt, exchange, price, qty, side):
    return {
        "ticker": str(row.get("ticker") or ""),
        "series": str(row.get("series_ticker") or ""),
        "close_time": str(row.get("close_time") or ""),
        "receipt_start": receipt,
        "receipt_end": receipt,
        "exchange_start": exchange,
        "exchange_end": exchange,
        "receipt_elapsed_min": _f(row.get("elapsed_s")) / 60.0,
        "receipt_elapsed_max": _f(row.get("elapsed_s")) / 60.0,
        "trades": 1,
        "qty": float(qty),
        "price_min": float(price),
        "price_max": float(price),
        "trade_ids": {str(row.get("trade_id") or "")},
        "bid_trades": int(side == "bid"),
        "ask_trades": int(side == "ask"),
        "unknown_side_trades": int(side not in {"bid", "ask"}),
        "receipt_minus_exchange_ms": [
            (receipt - exchange).total_seconds() * 1000.0
            if not pd.isna(receipt) and not pd.isna(exchange)
            else np.nan
        ],
    }


def _extend_burst(b, row, receipt, exchange, price, qty, side):
    b["receipt_end"] = receipt
    if not pd.isna(exchange):
        if pd.isna(b["exchange_start"]) or exchange < b["exchange_start"]:
            b["exchange_start"] = exchange
        if pd.isna(b["exchange_end"]) or exchange > b["exchange_end"]:
            b["exchange_end"] = exchange
    e = _f(row.get("elapsed_s")) / 60.0
    b["receipt_elapsed_min"] = min(b["receipt_elapsed_min"], e)
    b["receipt_elapsed_max"] = max(b["receipt_elapsed_max"], e)
    b["trades"] += 1
    b["qty"] += float(qty)
    b["price_min"] = min(b["price_min"], float(price))
    b["price_max"] = max(b["price_max"], float(price))
    b["trade_ids"].add(str(row.get("trade_id") or ""))
    b["bid_trades"] += int(side == "bid")
    b["ask_trades"] += int(side == "ask")
    b["unknown_side_trades"] += int(side not in {"bid", "ask"})
    b["receipt_minus_exchange_ms"].append(
        (receipt - exchange).total_seconds() * 1000.0
        if not pd.isna(receipt) and not pd.isna(exchange)
        else np.nan
    )


def _finish_burst(b, burst_id):
    if b is None:
        return None
    price_range_c = 100.0 * (float(b["price_max"]) - float(b["price_min"]))
    if int(b["trades"]) < BURST_MIN_TRADES or price_range_c < BURST_MIN_RANGE_C:
        return None

    rs = b["receipt_start"]
    re = b["receipt_end"]
    xs = b["exchange_start"]
    xe = b["exchange_end"]
    receipt_span_ms = (re - rs).total_seconds() * 1000.0
    exchange_span_ms = (
        (xe - xs).total_seconds() * 1000.0
        if not pd.isna(xs) and not pd.isna(xe)
        else np.nan
    )
    lags = np.asarray([x for x in b["receipt_minus_exchange_ms"] if np.isfinite(x)], dtype=float)
    directional = int(b["bid_trades"] + b["ask_trades"])
    dominant = max(int(b["bid_trades"]), int(b["ask_trades"]))
    dominance = dominant / directional if directional else np.nan
    direction_class = (
        "BID_DOMINANT" if directional and b["bid_trades"] > b["ask_trades"]
        else "ASK_DOMINANT" if directional and b["ask_trades"] > b["bid_trades"]
        else "BALANCED_OR_UNKNOWN"
    )

    return {
        "ticker": b["ticker"],
        "series": b["series"],
        "close_time": b["close_time"],
        "burst_id": int(burst_id),
        "receipt_start": rs,
        "receipt_end": re,
        "receipt_span_ms": float(receipt_span_ms),
        "receipt_elapsed_min": float(b["receipt_elapsed_min"]),
        "receipt_elapsed_max": float(b["receipt_elapsed_max"]),
        "exchange_start": xs,
        "exchange_end": xe,
        "exchange_span_ms": float(exchange_span_ms) if np.isfinite(exchange_span_ms) else np.nan,
        "trades": int(b["trades"]),
        "qty": float(b["qty"]),
        "price_min": float(b["price_min"]),
        "price_max": float(b["price_max"]),
        "price_range_c": float(price_range_c),
        "unique_trade_ids": int(len(b["trade_ids"] - {""})),
        "bid_trades": int(b["bid_trades"]),
        "ask_trades": int(b["ask_trades"]),
        "unknown_side_trades": int(b["unknown_side_trades"]),
        "dominant_taker_fraction": float(dominance) if np.isfinite(dominance) else np.nan,
        "direction_class": direction_class,
        "median_receipt_minus_exchange_ms": float(np.median(lags)) if len(lags) else np.nan,
    }


def _scan_pillars(source: Path, metadata_by_ticker, *, show=True):
    active = {}
    burst_seq = defaultdict(int)
    rows = []
    scanned = 0
    selected = 0

    def finalize(ticker):
        b = active.pop(ticker, None)
        if b is None:
            return
        burst_seq[ticker] += 1
        z = _finish_burst(b, burst_seq[ticker])
        if z is not None:
            rows.append(z)

    for r in _iter_jsonl(source / "trades_event_time.jsonl"):
        scanned += 1
        ticker = str(r.get("ticker") or "")
        if ticker not in metadata_by_ticker:
            continue
        e = _f(r.get("elapsed_s"))
        if not (np.isfinite(e) and M1_S <= e < M5_S):
            continue
        price = _f(r.get("yes_price"))
        qty = _f(r.get("qty"))
        if not (np.isfinite(price) and 0.0 <= price <= 1.0 and np.isfinite(qty) and qty > 0):
            continue
        receipt = _ts(r.get("receipt_time"))
        if pd.isna(receipt):
            continue
        exchange = _ts(r.get("exchange_time"))
        side = str(r.get("taker_book_side") or "").lower()
        selected += 1

        old = active.get(ticker)
        if old is None:
            active[ticker] = _new_burst(r, receipt, exchange, price, qty, side)
        else:
            gap_ms = (receipt - old["receipt_end"]).total_seconds() * 1000.0
            if gap_ms <= BURST_GAP_MS + EPS:
                _extend_burst(old, r, receipt, exchange, price, qty, side)
            else:
                finalize(ticker)
                active[ticker] = _new_burst(r, receipt, exchange, price, qty, side)

        if show and scanned % 250_000 == 0:
            print(
                f"trade scan: {scanned:,} rows | M1-M5 valid={selected:,} | "
                f"pillars found={len(rows):,}"
            )

    for ticker in list(active):
        finalize(ticker)

    return pd.DataFrame(rows)


def _attach_start_bbo(source: Path, pillars: pd.DataFrame, *, show=True):
    if pillars.empty:
        return pillars

    out = pillars.copy().reset_index(drop=True)
    out["bbo_receipt_time"] = pd.NaT
    out["bbo_age_ms"] = np.nan
    out["bid_at_start"] = np.nan
    out["ask_at_start"] = np.nan
    out["spread_at_start_c"] = np.nan

    targets = defaultdict(list)
    for idx, row in out.iterrows():
        targets[str(row["ticker"])].append((row["receipt_start"], int(idx)))
    for ticker in targets:
        targets[ticker].sort(key=lambda z: z[0])

    ptr = defaultdict(int)
    latest = {}
    scanned = 0

    def assign_until(ticker, cutoff):
        arr = targets.get(ticker)
        if not arr:
            return
        i = ptr[ticker]
        last = latest.get(ticker)
        while i < len(arr) and arr[i][0] < cutoff:
            target_ts, idx = arr[i]
            if last is not None:
                bt, bid, ask = last
                age_ms = (target_ts - bt).total_seconds() * 1000.0
                if -EPS <= age_ms <= BBO_MAX_AGE_S * 1000.0 + EPS:
                    out.at[idx, "bbo_receipt_time"] = bt
                    out.at[idx, "bbo_age_ms"] = age_ms
                    out.at[idx, "bid_at_start"] = bid
                    out.at[idx, "ask_at_start"] = ask
                    out.at[idx, "spread_at_start_c"] = 100.0 * (ask - bid)
            i += 1
        ptr[ticker] = i

    for r in _iter_jsonl(source / "book_top3_events.jsonl"):
        scanned += 1
        ticker = str(r.get("ticker") or "")
        if ticker not in targets:
            continue
        rt = _ts(r.get("receipt_time"))
        bid = _f(r.get("yes_bid"))
        ask = _f(r.get("yes_ask"))
        if pd.isna(rt) or not (np.isfinite(bid) and np.isfinite(ask) and 0 <= bid < ask <= 1):
            continue
        # Targets strictly before this book event use the previous BBO.
        assign_until(ticker, rt)
        latest[ticker] = (rt, float(bid), float(ask))

        if show and scanned % 1_000_000 == 0:
            done = sum(ptr.values())
            print(f"book scan: {scanned:,} rows | pillar BBOs assigned/advanced={done:,}/{len(out):,}")

    # Flush remaining targets against final known BBO per ticker.
    for ticker, arr in targets.items():
        i = ptr[ticker]
        last = latest.get(ticker)
        while i < len(arr):
            target_ts, idx = arr[i]
            if last is not None:
                bt, bid, ask = last
                age_ms = (target_ts - bt).total_seconds() * 1000.0
                if -EPS <= age_ms <= BBO_MAX_AGE_S * 1000.0 + EPS:
                    out.at[idx, "bbo_receipt_time"] = bt
                    out.at[idx, "bbo_age_ms"] = age_ms
                    out.at[idx, "bid_at_start"] = bid
                    out.at[idx, "ask_at_start"] = ask
                    out.at[idx, "spread_at_start_c"] = 100.0 * (ask - bid)
            i += 1
        ptr[ticker] = i

    out["wide_spread_at_start"] = pd.to_numeric(
        out["spread_at_start_c"], errors="coerce"
    ) > WIDE_SPREAD_C
    out["bbo_start_available"] = pd.to_numeric(
        out["spread_at_start_c"], errors="coerce"
    ).notna()
    return out


def _all_windows(metadata_rows):
    rows = []
    by_close = defaultdict(set)
    for r in metadata_rows:
        close = str(r.get("close_time") or "")
        series = str(r.get("series") or "")
        if close and series:
            by_close[close].add(series)
    for close, series_set in sorted(by_close.items()):
        rows.append({
            "close_time": close,
            "markets_in_metadata": len(series_set),
        })
    return pd.DataFrame(rows)


def _aggregate_windows(all_windows: pd.DataFrame, pillars: pd.DataFrame):
    if pillars.empty:
        out = all_windows.copy()
        for c in [
            "pillar_count_all", "pillar_count_spread_gt_2c", "markets_with_pillar_all",
            "markets_with_pillar_spread_gt_2c", "pillar_trades_all", "pillar_qty_all",
            "pillar_qty_spread_gt_2c", "max_price_range_c", "max_trades_in_pillar",
        ]:
            out[c] = 0
        return out

    p = pillars.copy()
    p["wide_i"] = p["wide_spread_at_start"].astype(int)
    p["wide_qty"] = p["qty"] * p["wide_i"]

    agg = (
        p.groupby("close_time", as_index=False)
        .agg(
            pillar_count_all=("ticker", "size"),
            pillar_count_spread_gt_2c=("wide_i", "sum"),
            markets_with_pillar_all=("ticker", "nunique"),
            pillar_trades_all=("trades", "sum"),
            pillar_qty_all=("qty", "sum"),
            pillar_qty_spread_gt_2c=("wide_qty", "sum"),
            max_price_range_c=("price_range_c", "max"),
            max_trades_in_pillar=("trades", "max"),
        )
    )
    wide_markets = (
        p[p["wide_spread_at_start"]]
        .groupby("close_time")["ticker"]
        .nunique()
        .rename("markets_with_pillar_spread_gt_2c")
        .reset_index()
    )
    agg = agg.merge(wide_markets, on="close_time", how="left")

    out = all_windows.merge(agg, on="close_time", how="left")
    count_cols = [
        "pillar_count_all", "pillar_count_spread_gt_2c", "markets_with_pillar_all",
        "markets_with_pillar_spread_gt_2c", "pillar_trades_all", "max_trades_in_pillar",
    ]
    for c in count_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    for c in ["pillar_qty_all", "pillar_qty_spread_gt_2c", "max_price_range_c"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out.sort_values("close_time").reset_index(drop=True)


def _aggregate_assets(pillars: pd.DataFrame, n_windows: int):
    if pillars.empty:
        return pd.DataFrame()
    p = pillars.copy()
    p["wide_i"] = p["wide_spread_at_start"].astype(int)
    p["wide_qty"] = p["qty"] * p["wide_i"]
    out = (
        p.groupby("series", as_index=False)
        .agg(
            pillar_count_all=("ticker", "size"),
            pillar_count_spread_gt_2c=("wide_i", "sum"),
            windows_with_pillar=("close_time", "nunique"),
            total_trades=("trades", "sum"),
            total_qty=("qty", "sum"),
            wide_qty=("wide_qty", "sum"),
            median_price_range_c=("price_range_c", "median"),
            max_price_range_c=("price_range_c", "max"),
            median_exchange_span_ms=("exchange_span_ms", "median"),
            median_dominant_taker_fraction=("dominant_taker_fraction", "median"),
        )
    )
    out["pillars_per_15m_window"] = out["pillar_count_all"] / max(1, n_windows)
    out["wide_pillars_per_15m_window"] = out["pillar_count_spread_gt_2c"] / max(1, n_windows)
    return out.sort_values("pillar_count_all", ascending=False).reset_index(drop=True)


def _count_distribution(by_window: pd.DataFrame):
    rows = []
    for col, label in [
        ("pillar_count_all", "ALL_PILLARS"),
        ("pillar_count_spread_gt_2c", "SPREAD_GT_2C_AT_START"),
    ]:
        s = pd.to_numeric(by_window[col], errors="coerce").fillna(0).astype(int)
        counts = Counter(s.tolist())
        for k in sorted(counts):
            rows.append({"scope": label, "pillars_in_window": int(k), "window_count": int(counts[k])})
    return pd.DataFrame(rows)


def run_trade_pillar_24h_census(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected formal OOS {HARD_BOUND_SESSION}, got {source.name}")

    required = [
        source / "trades_event_time.jsonl",
        source / "book_top3_events.jsonl",
        source / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + " | ".join(missing))

    metadata_rows, metadata_by_ticker = _metadata(source)
    all_windows = _all_windows(metadata_rows)

    if show:
        print("Scanning ALL M1-M5 public trades for V13-defined pillars...")
    pillars = _scan_pillars(source, metadata_by_ticker, show=show)

    if show:
        print(f"\nDetected {len(pillars):,} raw V13-defined pillars. Attaching BBO/spread at burst start...")
    pillars = _attach_start_bbo(source, pillars, show=show)

    by_window = _aggregate_windows(all_windows, pillars)
    by_asset = _aggregate_assets(pillars, len(all_windows))
    distribution = _count_distribution(by_window)

    out = _new_output(source.name)
    pillars.to_csv(out / "pillar_detail.csv", index=False)
    by_window.to_csv(out / "pillar_count_by_window.csv", index=False)
    by_asset.to_csv(out / "pillar_count_by_asset.csv", index=False)
    distribution.to_csv(out / "pillar_count_distribution.csv", index=False)

    total = int(len(pillars))
    wide = int(pillars["wide_spread_at_start"].sum()) if total else 0
    nwin = int(len(by_window))
    windows_any = int((by_window["pillar_count_all"] > 0).sum()) if nwin else 0
    windows_wide = int((by_window["pillar_count_spread_gt_2c"] > 0).sum()) if nwin else 0

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "same_realization_only": True,
        "independent_validation": False,
        "definition": {
            "window": "M1 <= elapsed < M5",
            "max_receipt_gap_ms": BURST_GAP_MS,
            "minimum_trades": BURST_MIN_TRADES,
            "minimum_price_range_c": BURST_MIN_RANGE_C,
            "wide_spread_threshold_c": WIDE_SPREAD_C,
            "wide_spread_clock": "latest valid BBO by local receipt time at burst start",
            "maximum_bbo_age_s": BBO_MAX_AGE_S,
        },
        "metadata_windows": nwin,
        "metadata_markets": len(metadata_by_ticker),
        "pillar_count_all": total,
        "pillar_count_spread_gt_2c": wide,
        "wide_fraction_of_pillars": wide / total if total else np.nan,
        "windows_with_any_pillar": windows_any,
        "windows_with_wide_spread_pillar": windows_wide,
        "fraction_windows_with_any_pillar": windows_any / nwin if nwin else np.nan,
        "fraction_windows_with_wide_spread_pillar": windows_wide / nwin if nwin else np.nan,
        "mean_pillars_per_window_all": float(by_window["pillar_count_all"].mean()) if nwin else np.nan,
        "median_pillars_per_window_all": float(by_window["pillar_count_all"].median()) if nwin else np.nan,
        "max_pillars_in_one_window_all": int(by_window["pillar_count_all"].max()) if nwin else 0,
        "mean_wide_pillars_per_window": float(by_window["pillar_count_spread_gt_2c"].mean()) if nwin else np.nan,
        "median_wide_pillars_per_window": float(by_window["pillar_count_spread_gt_2c"].median()) if nwin else np.nan,
        "max_wide_pillars_in_one_window": int(by_window["pillar_count_spread_gt_2c"].max()) if nwin else 0,
        "bbo_start_coverage": float(pillars["bbo_start_available"].mean()) if total else np.nan,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "guardrail": (
            "This is a descriptive census using the already-inspected V13 pillar definition. "
            "It is not a pre-registered signal test and must not be treated as new OOS alpha evidence."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("\n" + "=" * 150)
        print("24H TRADE-PILLAR CENSUS V15 — FORMAL OOS / READ ONLY")
        print("=" * 150)
        print("Source:", source)
        print("Operational definition: >=8 trades, <=25ms consecutive receipt gaps, >=10c price range, M1-M5")
        print(f"Metadata windows: {nwin} | markets: {len(metadata_by_ticker)}")
        print()
        print("CENSUS")
        print(f"  pillars, all:                         {total}")
        print(f"  pillars with spread >2c at start:     {wide}")
        print(f"  fraction of pillars in >2c spread:    {100.0 * wide / total:.2f}%" if total else "  fraction of pillars in >2c spread:    n/a")
        print(f"  windows with >=1 pillar:              {windows_any} / {nwin} ({100.0*windows_any/nwin:.2f}%)" if nwin else "")
        print(f"  windows with >=1 >2c-spread pillar:   {windows_wide} / {nwin} ({100.0*windows_wide/nwin:.2f}%)" if nwin else "")
        print(f"  mean / median pillars per window:     {summary['mean_pillars_per_window_all']:.3f} / {summary['median_pillars_per_window_all']:.3f}")
        print(f"  max pillars in one window:            {summary['max_pillars_in_one_window_all']}")
        print(f"  mean / median >2c pillars per window: {summary['mean_wide_pillars_per_window']:.3f} / {summary['median_wide_pillars_per_window']:.3f}")
        print(f"  max >2c pillars in one window:        {summary['max_wide_pillars_in_one_window']}")
        print(f"  BBO-at-start coverage:                {100.0*summary['bbo_start_coverage']:.2f}%" if np.isfinite(summary['bbo_start_coverage']) else "  BBO-at-start coverage: n/a")
        print()
        print("TOP 20 WINDOWS BY >2c-SPREAD PILLAR COUNT")
        top = by_window.sort_values(
            ["pillar_count_spread_gt_2c", "pillar_count_all", "pillar_qty_all"],
            ascending=False,
        ).head(20)
        print(top.to_string(index=False))
        print()
        print("BY ASSET")
        print(by_asset.to_string(index=False) if not by_asset.empty else "  none")
        print()
        print("COUNT DISTRIBUTION")
        print(distribution.to_string(index=False) if not distribution.empty else "  none")
        print()
        print("Guardrail: descriptive same-realization census only; NOT a new strategy backtest.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 150)

    return {
        "summary": summary,
        "pillars": pillars,
        "by_window": by_window,
        "by_asset": by_asset,
        "distribution": distribution,
        "output_dir": out,
    }


__all__ = ["VERSION", "run_trade_pillar_24h_census"]
