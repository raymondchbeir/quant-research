from __future__ import annotations

"""Q1 M1-M5 two-sided market-making backtest on validated reconstructed books.

This is the first unconditional MM feasibility backtest after validating the
historical book reconstruction against ticker BBO and full-book anchors.

Fixed policy (not optimized):
- use only contracts that passed the existing reconstructed-book quality gate;
- quote 1 contract at the reconstructed YES best bid and YES best ask;
- 1 Hz quote refresh: keep queue priority while that side's BBO price is unchanged;
- cancel/rejoin when that side's 1 Hz BBO changes;
- cancel both sides whenever the reconstructed 1 Hz state is invalid;
- join the back of displayed L1 FIFO queue;
- only recorded aggressive trades deplete queue ahead (no cancellation credit);
- exact-price aggressive flow consumes queue ahead first;
- genuine trade-through fills the remaining tiny order;
- after a complete fill, rejoin that side at the next valid 1 Hz book sample;
- stop at M5 and mark remaining YES-equivalent inventory to the last valid M5 book;
- fees are excluded; break-even fee per filled contract is reported.

ASK-side fills are represented as short YES-equivalent inventory. Economically,
this is the same PnL as passively buying NO at the complementary price.

No asset, spread, imbalance, volatility, or minute regime is selected here.
"""

import argparse
import bisect
import csv
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_m1_m5_feasibility as _base

STUDY_VERSION = "M1_M5_RECON_Q1_MM_BACKTEST_V1"
EPS = 1e-9
DEFAULT_MARKOUT_SECONDS = (5, 15, 30, 60)


@dataclass(slots=True)
class Sample:
    t: float
    minute: float
    status: str
    bid1: float
    bid1_qty: float
    ask1: float
    ask1_qty: float
    mid: float
    spread_c: float


def _f(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def _ts(x):
    return _base._ts_seconds(x)


def _iso(x):
    return _base._iso(x)


def _load_quality_contracts(reconstruction_dir):
    path = Path(reconstruction_dir) / "contract_reconstruction_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"ticker", "series", "close_time", "quality_ok"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    good = df[df["quality_ok"].map(_bool)].copy()
    if good.empty:
        raise RuntimeError("No quality_ok contracts in reconstruction summary")
    good["close_ts"] = good["close_time"].map(_ts)
    good = good[np.isfinite(good["close_ts"])].copy()
    if good.empty:
        raise RuntimeError("No quality contracts have parseable close_time")
    meta = {
        str(r.ticker): {
            "ticker": str(r.ticker),
            "series": str(r.series),
            "close_ts": float(r.close_ts),
            "reconstructed_coverage_pct": _f(getattr(r, "reconstructed_coverage_pct", np.nan)),
        }
        for r in good.itertuples(index=False)
    }
    return good, meta


def _load_reconstructed_samples(reconstruction_dir, eligible, progress_every=100_000):
    path = Path(reconstruction_dir) / "reconstructed_book_samples.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    out = defaultdict(list)
    scanned = kept = 0
    t0 = time.time()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scanned += 1
            if progress_every and scanned % progress_every == 0:
                print(f"  reconstructed samples: {scanned:,} rows | kept={kept:,} | {time.time()-t0:.1f}s")
            ticker = str(row.get("ticker") or "")
            if ticker not in eligible:
                continue
            t = _f(row.get("ts"))
            if not np.isfinite(t):
                t = _ts(row.get("time"))
            if not np.isfinite(t):
                continue
            status = str(row.get("status") or "UNKNOWN").upper()
            s = Sample(
                t=float(t),
                minute=_f(row.get("minute")),
                status=status,
                bid1=_f(row.get("yes_bid1")),
                bid1_qty=max(0.0, _f(row.get("yes_bid1_qty"), 0.0)),
                ask1=_f(row.get("yes_ask1")),
                ask1_qty=max(0.0, _f(row.get("yes_ask1_qty"), 0.0)),
                mid=_f(row.get("mid")),
                spread_c=_f(row.get("spread_c")),
            )
            out[ticker].append(s)
            kept += 1
    for ticker in out:
        out[ticker].sort(key=lambda z: z.t)
    return out, {
        "reconstructed_sample_rows_scanned": scanned,
        "reconstructed_sample_rows_kept": kept,
        "reconstructed_sample_seconds": time.time() - t0,
    }


def _scan_trades(session_dir, meta, progress_every=1_000_000):
    path = Path(session_dir) / "trades.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    eligible = set(meta)
    out = defaultdict(list)
    scanned = decoded = kept = 0
    t0 = time.time()
    with path.open("rb") as f:
        for raw in f:
            scanned += 1
            if progress_every and scanned % progress_every == 0:
                print(f"  trades: {scanned:,} lines | kept={kept:,} | {time.time()-t0:.1f}s")
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            decoded += 1
            parsed = _base._parse_trade(obj)
            if parsed is None:
                continue
            ticker, tr = parsed
            if ticker not in eligible:
                continue
            close = meta[ticker]["close_ts"]
            wstart = close - 900.0 + 60.0
            wend = close - 900.0 + 300.0
            if wstart <= tr.t < wend:
                out[ticker].append(tr)
                kept += 1
    for ticker in out:
        out[ticker].sort(key=lambda z: z.t)
    return out, {
        "trade_lines_scanned": scanned,
        "trade_lines_decoded": decoded,
        "trades_kept": kept,
        "trade_scan_seconds": time.time() - t0,
    }


def _valid_sample(s):
    return (
        s.status == "VALID"
        and np.isfinite(s.bid1)
        and np.isfinite(s.ask1)
        and np.isfinite(s.mid)
        and 0.0 <= s.bid1 < s.ask1 <= 1.0
    )


def _future_valid_sample(samples, times, target, max_lag_s=2.0):
    i = bisect.bisect_left(times, target)
    while i < len(samples):
        s = samples[i]
        if s.t - target > max_lag_s + EPS:
            return None
        if _valid_sample(s):
            return s
        i += 1
    return None


def _match_roundtrips(fills):
    longs = deque()
    shorts = deque()
    qty_done = pnl = 0.0
    for f in sorted(fills, key=lambda x: x["fill_ts"]):
        qty = float(f["qty"])
        px = float(f["price"])
        if f["side"] == "BID":
            while qty > EPS and shorts:
                sell_px, q0 = shorts[0]
                m = min(qty, q0)
                pnl += (sell_px - px) * m
                qty_done += m
                qty -= m
                q0 -= m
                if q0 <= EPS:
                    shorts.popleft()
                else:
                    shorts[0] = (sell_px, q0)
            if qty > EPS:
                longs.append((px, qty))
        else:
            while qty > EPS and longs:
                buy_px, q0 = longs[0]
                m = min(qty, q0)
                pnl += (px - buy_px) * m
                qty_done += m
                qty -= m
                q0 -= m
                if q0 <= EPS:
                    longs.popleft()
                else:
                    longs[0] = (buy_px, q0)
            if qty > EPS:
                shorts.append((px, qty))
    return qty_done, pnl


def _close_episode(ep, t, reason):
    if ep is None:
        return
    if not np.isfinite(_f(ep.get("end_ts"))):
        ep["end_ts"] = float(t)
        ep["end_time"] = _iso(t)
        ep["end_reason"] = reason
    ep["duration_s"] = max(0.0, float(ep["end_ts"]) - float(ep["join_ts"]))
    ep["filled_any"] = bool(ep["fill_qty"] > EPS)


def _simulate_contract(ticker, meta, samples, trades, quote_qty, markouts, max_markout_lag_s):
    close = meta["close_ts"]
    wstart = close - 900.0 + 60.0
    wend = close - 900.0 + 300.0
    series = meta["series"]
    samples = [s for s in samples if wstart <= s.t < wend]
    if not samples:
        return [], [], None
    samples.sort(key=lambda z: z.t)
    times = [s.t for s in samples]

    episodes = []
    fills = []
    active = {"BID": None, "ASK": None}
    remaining = {"BID": 0.0, "ASK": 0.0}
    episode_id = 0
    inventory = 0.0
    cash = 0.0
    max_abs_inventory = 0.0
    last_valid_mid = np.nan
    last_sample_t = samples[0].t
    trade_times = [tr.t for tr in trades]
    tr_idx = bisect.bisect_left(trade_times, last_sample_t)

    def open_order(side, s):
        nonlocal episode_id
        episode_id += 1
        if side == "BID":
            px, qa = s.bid1, s.bid1_qty
        else:
            px, qa = s.ask1, s.ask1_qty
        ep = {
            "ticker": ticker,
            "series": series,
            "episode_id": f"{ticker}:{side}:{episode_id}",
            "side": side,
            "join_ts": s.t,
            "join_time": _iso(s.t),
            "join_minute": s.minute,
            "price": px,
            "queue_ahead_initial": qa,
            "queue_ahead_final": qa,
            "spread_c_at_join": s.spread_c,
            "mid_at_join": s.mid,
            "fill_qty": 0.0,
            "first_fill_ts": np.nan,
            "last_fill_ts": np.nan,
            "fill_latency_s": np.nan,
            "end_ts": np.nan,
            "end_time": None,
            "end_reason": None,
        }
        episodes.append(ep)
        active[side] = ep
        remaining[side] = float(quote_qty)

    def cancel(side, t, reason):
        ep = active[side]
        if ep is not None:
            _close_episode(ep, t, reason)
        active[side] = None
        remaining[side] = 0.0

    def fill_order(side, tr, current_mid):
        nonlocal inventory, cash, max_abs_inventory
        ep = active[side]
        if ep is None or not np.isfinite(current_mid):
            return
        qpx = float(ep["price"])
        if side == "BID":
            if tr.taker_book_side != "ask":
                return
            exact = abs(tr.yes_price - qpx) <= EPS
            through = tr.yes_price < qpx - EPS
        else:
            if tr.taker_book_side != "bid":
                return
            exact = abs(tr.yes_price - qpx) <= EPS
            through = tr.yes_price > qpx + EPS
        if not (exact or through):
            return

        qty = 0.0
        if through:
            qty = remaining[side]
        else:
            qa = max(0.0, float(ep["queue_ahead_final"]))
            if tr.qty <= qa + EPS:
                ep["queue_ahead_final"] = max(0.0, qa - tr.qty)
                return
            ep["queue_ahead_final"] = 0.0
            qty = min(remaining[side], tr.qty - qa)
        if qty <= EPS:
            return

        ep["fill_qty"] += qty
        if not np.isfinite(_f(ep["first_fill_ts"])):
            ep["first_fill_ts"] = tr.t
            ep["fill_latency_s"] = tr.t - ep["join_ts"]
        ep["last_fill_ts"] = tr.t
        remaining[side] -= qty

        sign = 1.0 if side == "BID" else -1.0
        gross_edge_c = sign * (current_mid - qpx) * 100.0
        fill = {
            "ticker": ticker,
            "series": series,
            "episode_id": ep["episode_id"],
            "side": side,
            "fill_ts": tr.t,
            "fill_time": _iso(tr.t),
            "fill_minute": (tr.t - (close - 900.0)) / 60.0,
            "qty": qty,
            "price": qpx,
            "mid_at_fill": current_mid,
            "gross_edge_at_fill_c": gross_edge_c,
            "queue_ahead_initial": ep["queue_ahead_initial"],
            "fill_latency_s": ep["fill_latency_s"],
            "spread_c_at_join": ep["spread_c_at_join"],
        }
        for h in markouts:
            fs = _future_valid_sample(samples, times, tr.t + h, max_lag_s=max_markout_lag_s)
            if fs is None:
                fill[f"future_mid_{h}s"] = np.nan
                fill[f"post_mid_move_{h}s_c"] = np.nan
                fill[f"markout_{h}s_c"] = np.nan
            else:
                fill[f"future_mid_{h}s"] = fs.mid
                fill[f"post_mid_move_{h}s_c"] = sign * (fs.mid - current_mid) * 100.0
                fill[f"markout_{h}s_c"] = sign * (fs.mid - qpx) * 100.0
        fills.append(fill)

        if side == "BID":
            inventory += qty
            cash -= qpx * qty
        else:
            inventory -= qty
            cash += qpx * qty
        max_abs_inventory = max(max_abs_inventory, abs(inventory))

        if remaining[side] <= EPS:
            _close_episode(ep, tr.t, "FILLED")
            active[side] = None
            remaining[side] = 0.0

    current_mid = np.nan
    for i, s in enumerate(samples):
        if i == 0:
            last_sample_t = s.t
        else:
            while tr_idx < len(trades) and trades[tr_idx].t <= s.t + EPS:
                tr = trades[tr_idx]
                if tr.t > last_sample_t + EPS:
                    fill_order("BID", tr, current_mid)
                    fill_order("ASK", tr, current_mid)
                tr_idx += 1
            last_sample_t = s.t

        if not _valid_sample(s):
            cancel("BID", s.t, "INVALID_BOOK")
            cancel("ASK", s.t, "INVALID_BOOK")
            current_mid = np.nan
            continue

        current_mid = s.mid
        last_valid_mid = s.mid
        for side, px in (("BID", s.bid1), ("ASK", s.ask1)):
            ep = active[side]
            if ep is None:
                open_order(side, s)
            elif abs(float(ep["price"]) - px) > EPS:
                cancel(side, s.t, "BBO_REPRICE")
                open_order(side, s)

    end_t = min(wend, samples[-1].t)
    cancel("BID", end_t, "M5_END")
    cancel("ASK", end_t, "M5_END")

    if not np.isfinite(last_valid_mid):
        return episodes, fills, None
    final_mid = last_valid_mid
    net_mtm = cash + inventory * final_mid
    gross_capture = sum((f["gross_edge_at_fill_c"] / 100.0) * f["qty"] for f in fills)
    adverse_to_m5 = net_mtm - gross_capture
    matched_qty, matched_pnl = _match_roundtrips(fills)
    bid_qty = sum(f["qty"] for f in fills if f["side"] == "BID")
    ask_qty = sum(f["qty"] for f in fills if f["side"] == "ASK")
    both = bid_qty > EPS and ask_qty > EPS
    one = (bid_qty > EPS) ^ (ask_qty > EPS)

    contract = {
        "ticker": ticker,
        "series": series,
        "close_time": _iso(close),
        "close_ts": close,
        "reconstructed_coverage_pct": meta.get("reconstructed_coverage_pct"),
        "sample_rows": len(samples),
        "valid_sample_rows": sum(_valid_sample(s) for s in samples),
        "bid_quote_episodes": sum(e["side"] == "BID" for e in episodes),
        "ask_quote_episodes": sum(e["side"] == "ASK" for e in episodes),
        "bid_filled_episodes": sum(e["side"] == "BID" and e.get("filled_any") for e in episodes),
        "ask_filled_episodes": sum(e["side"] == "ASK" and e.get("filled_any") for e in episodes),
        "bid_fill_qty": bid_qty,
        "ask_fill_qty": ask_qty,
        "fill_qty": bid_qty + ask_qty,
        "both_sides_filled": both,
        "one_sided_fill": one,
        "no_fill": bid_qty <= EPS and ask_qty <= EPS,
        "max_abs_inventory": max_abs_inventory,
        "ending_inventory_yes_equiv": inventory,
        "final_mid_m5": final_mid,
        "cash": cash,
        "gross_spread_capture_dollars": gross_capture,
        "adverse_selection_to_m5_dollars": adverse_to_m5,
        "net_mtm_pnl_before_fees": net_mtm,
        "matched_roundtrip_qty": matched_qty,
        "matched_roundtrip_pnl": matched_pnl,
    }
    for h in markouts:
        vals = [f[f"markout_{h}s_c"] for f in fills if np.isfinite(_f(f.get(f"markout_{h}s_c")))]
        moves = [f[f"post_mid_move_{h}s_c"] for f in fills if np.isfinite(_f(f.get(f"post_mid_move_{h}s_c")))]
        contract[f"mean_markout_{h}s_c"] = float(np.mean(vals)) if vals else np.nan
        contract[f"mean_post_mid_move_{h}s_c"] = float(np.mean(moves)) if moves else np.nan
    return episodes, fills, contract


def _side_summary(episodes, fills, markouts):
    edf = pd.DataFrame(episodes)
    fdf = pd.DataFrame(fills)
    rows = []
    for side in ("BID", "ASK", "ALL"):
        e = edf if side == "ALL" else edf[edf["side"] == side]
        f = fdf if side == "ALL" else fdf[fdf["side"] == side]
        row = {
            "side": side,
            "quote_episodes": len(e),
            "filled_episodes": int(e["filled_any"].sum()) if len(e) else 0,
            "episode_fill_rate_pct": 100.0 * e["filled_any"].mean() if len(e) else np.nan,
            "fill_events": len(f),
            "fill_qty": pd.to_numeric(f.get("qty"), errors="coerce").sum() if len(f) else 0.0,
            "avg_queue_ahead": pd.to_numeric(e.get("queue_ahead_initial"), errors="coerce").mean() if len(e) else np.nan,
            "median_queue_ahead": pd.to_numeric(e.get("queue_ahead_initial"), errors="coerce").median() if len(e) else np.nan,
            "avg_fill_latency_s": pd.to_numeric(f.get("fill_latency_s"), errors="coerce").mean() if len(f) else np.nan,
            "avg_gross_edge_at_fill_c": pd.to_numeric(f.get("gross_edge_at_fill_c"), errors="coerce").mean() if len(f) else np.nan,
        }
        for h in markouts:
            mo = pd.to_numeric(f.get(f"markout_{h}s_c"), errors="coerce") if len(f) else pd.Series(dtype=float)
            pm = pd.to_numeric(f.get(f"post_mid_move_{h}s_c"), errors="coerce") if len(f) else pd.Series(dtype=float)
            row[f"avg_markout_{h}s_c"] = mo.mean() if mo.notna().any() else np.nan
            row[f"median_markout_{h}s_c"] = mo.median() if mo.notna().any() else np.nan
            row[f"adverse_markout_{h}s_pct"] = 100.0 * (mo.dropna() < 0).mean() if mo.notna().any() else np.nan
            row[f"avg_post_mid_move_{h}s_c"] = pm.mean() if pm.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _asset_summary(contract_df):
    rows = []
    for series, g in contract_df.groupby("series"):
        filled = g[g["fill_qty"] > EPS]
        rows.append({
            "series": series,
            "eligible_contracts": len(g),
            "contracts_with_fill": len(filled),
            "contract_fill_pct": 100.0 * len(filled) / len(g) if len(g) else np.nan,
            "fill_qty": g["fill_qty"].sum(),
            "both_sides_pct": 100.0 * g["both_sides_filled"].mean(),
            "one_sided_pct": 100.0 * g["one_sided_fill"].mean(),
            "avg_max_abs_inventory": g["max_abs_inventory"].mean(),
            "gross_spread_capture_dollars": g["gross_spread_capture_dollars"].sum(),
            "adverse_selection_to_m5_dollars": g["adverse_selection_to_m5_dollars"].sum(),
            "net_mtm_pnl_before_fees": g["net_mtm_pnl_before_fees"].sum(),
            "pnl_per_eligible_contract": g["net_mtm_pnl_before_fees"].mean(),
            "matched_roundtrip_pnl": g["matched_roundtrip_pnl"].sum(),
        })
    return pd.DataFrame(rows).sort_values("net_mtm_pnl_before_fees", ascending=False)


def _minute_summary(fills, markouts):
    if not fills:
        return pd.DataFrame()
    df = pd.DataFrame(fills)
    df["minute_bucket"] = df["fill_minute"].map(lambda x: f"M{int(math.floor(x))}-M{int(math.floor(x))+1}")
    rows = []
    for (bucket, side), g in df.groupby(["minute_bucket", "side"]):
        row = {
            "minute_bucket": bucket,
            "side": side,
            "fill_events": len(g),
            "fill_qty": g["qty"].sum(),
            "avg_gross_edge_at_fill_c": g["gross_edge_at_fill_c"].mean(),
        }
        for h in markouts:
            row[f"avg_markout_{h}s_c"] = pd.to_numeric(g[f"markout_{h}s_c"], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["minute_bucket", "side"])


def _window_summary(contract_df):
    rows = []
    for close_time, g in contract_df.groupby("close_time", sort=True):
        rows.append({
            "close_time": close_time,
            "eligible_assets": len(g),
            "filled_assets": int((g["fill_qty"] > EPS).sum()),
            "fill_qty": g["fill_qty"].sum(),
            "gross_spread_capture_dollars": g["gross_spread_capture_dollars"].sum(),
            "adverse_selection_to_m5_dollars": g["adverse_selection_to_m5_dollars"].sum(),
            "net_mtm_pnl_before_fees": g["net_mtm_pnl_before_fees"].sum(),
            "max_contract_loss": g["net_mtm_pnl_before_fees"].min(),
            "max_contract_gain": g["net_mtm_pnl_before_fees"].max(),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["close_ts"] = out["close_time"].map(_ts)
    out = out.sort_values("close_ts").reset_index(drop=True)
    out["cumulative_pnl"] = out["net_mtm_pnl_before_fees"].cumsum()
    out["running_peak"] = out["cumulative_pnl"].cummax().clip(lower=0.0)
    out["drawdown"] = out["cumulative_pnl"] - out["running_peak"]
    return out


def _headline(contract_df, side_df, fills, window_df, markouts):
    fdf = pd.DataFrame(fills)
    total_pnl = contract_df["net_mtm_pnl_before_fees"].sum()
    fill_qty = fdf["qty"].sum() if len(fdf) else 0.0
    filled_contracts = int((contract_df["fill_qty"] > EPS).sum())
    row = {
        "study_version": STUDY_VERSION,
        "eligible_contracts": len(contract_df),
        "independent_windows": len(window_df),
        "contracts_with_fill": filled_contracts,
        "contract_fill_pct": 100.0 * filled_contracts / len(contract_df) if len(contract_df) else np.nan,
        "fill_events": len(fdf),
        "fill_qty": fill_qty,
        "both_sides_contract_pct": 100.0 * contract_df["both_sides_filled"].mean() if len(contract_df) else np.nan,
        "one_sided_contract_pct": 100.0 * contract_df["one_sided_fill"].mean() if len(contract_df) else np.nan,
        "no_fill_contract_pct": 100.0 * contract_df["no_fill"].mean() if len(contract_df) else np.nan,
        "avg_max_abs_inventory": contract_df["max_abs_inventory"].mean() if len(contract_df) else np.nan,
        "p95_max_abs_inventory": contract_df["max_abs_inventory"].quantile(0.95) if len(contract_df) else np.nan,
        "gross_spread_capture_dollars": contract_df["gross_spread_capture_dollars"].sum(),
        "adverse_selection_to_m5_dollars": contract_df["adverse_selection_to_m5_dollars"].sum(),
        "net_mtm_pnl_before_fees": total_pnl,
        "matched_roundtrip_qty": contract_df["matched_roundtrip_qty"].sum(),
        "matched_roundtrip_pnl": contract_df["matched_roundtrip_pnl"].sum(),
        "pnl_per_eligible_contract": total_pnl / len(contract_df) if len(contract_df) else np.nan,
        "pnl_per_filled_contract": total_pnl / filled_contracts if filled_contracts else np.nan,
        "pnl_per_window": total_pnl / len(window_df) if len(window_df) else np.nan,
        "break_even_fee_c_per_filled_qty": 100.0 * total_pnl / fill_qty if fill_qty > EPS else np.nan,
        "worst_contract_pnl": contract_df["net_mtm_pnl_before_fees"].min() if len(contract_df) else np.nan,
        "best_contract_pnl": contract_df["net_mtm_pnl_before_fees"].max() if len(contract_df) else np.nan,
        "contract_p10_pnl": contract_df["net_mtm_pnl_before_fees"].quantile(0.10) if len(contract_df) else np.nan,
        "contract_median_pnl": contract_df["net_mtm_pnl_before_fees"].median() if len(contract_df) else np.nan,
        "contract_p90_pnl": contract_df["net_mtm_pnl_before_fees"].quantile(0.90) if len(contract_df) else np.nan,
        "worst_window_pnl": window_df["net_mtm_pnl_before_fees"].min() if len(window_df) else np.nan,
        "best_window_pnl": window_df["net_mtm_pnl_before_fees"].max() if len(window_df) else np.nan,
        "window_p10_pnl": window_df["net_mtm_pnl_before_fees"].quantile(0.10) if len(window_df) else np.nan,
        "window_median_pnl": window_df["net_mtm_pnl_before_fees"].median() if len(window_df) else np.nan,
        "window_p90_pnl": window_df["net_mtm_pnl_before_fees"].quantile(0.90) if len(window_df) else np.nan,
        "max_drawdown": window_df["drawdown"].min() if len(window_df) else np.nan,
    }
    all_side = side_df[side_df["side"] == "ALL"]
    if len(all_side):
        a = all_side.iloc[0]
        row["episode_fill_rate_pct"] = a["episode_fill_rate_pct"]
        row["avg_gross_edge_at_fill_c"] = a["avg_gross_edge_at_fill_c"]
        for h in markouts:
            row[f"avg_markout_{h}s_c"] = a.get(f"avg_markout_{h}s_c", np.nan)
            row[f"adverse_markout_{h}s_pct"] = a.get(f"adverse_markout_{h}s_pct", np.nan)
    for side in ("BID", "ASK"):
        z = side_df[side_df["side"] == side]
        if len(z):
            row[f"{side.lower()}_episode_fill_rate_pct"] = z.iloc[0]["episode_fill_rate_pct"]
            row[f"{side.lower()}_fill_qty"] = z.iloc[0]["fill_qty"]
    return pd.DataFrame([row])


def _print_report(headline, side_df, asset_df, window_df, output_dir, markouts):
    r = headline.iloc[0]
    print("\n" + "=" * 108)
    print("M1-M5 Q1 TWO-SIDED MARKET-MAKING BACKTEST — VALIDATED RECONSTRUCTED BOOKS / BEFORE FEES")
    print("=" * 108)
    print(f"Eligible contracts:   {int(r['eligible_contracts']):,}")
    print(f"Independent windows:  {int(r['independent_windows']):,}")
    print(f"Contracts with fill:  {int(r['contracts_with_fill']):,} ({r['contract_fill_pct']:.2f}%)")
    print(f"Fill events / qty:    {int(r['fill_events']):,} / {r['fill_qty']:.2f}")

    print("\nPASSIVE FILL MECHANICS")
    for _, x in side_df[side_df["side"].isin(["BID", "ASK"])].iterrows():
        print(
            f"  {x['side']:>3}: episode fill={x['episode_fill_rate_pct']:.2f}% | "
            f"fill_qty={x['fill_qty']:.2f} | avg queue={x['avg_queue_ahead']:.1f} | "
            f"median queue={x['median_queue_ahead']:.1f} | avg latency={x['avg_fill_latency_s']:.2f}s"
        )

    print("\nPOST-FILL ECONOMICS")
    print(f"  Gross edge at fill:       {r.get('avg_gross_edge_at_fill_c', np.nan):+.3f} c/ct")
    for h in markouts:
        print(
            f"  {h:>2}s markout:              {r.get(f'avg_markout_{h}s_c', np.nan):+.3f} c/ct | "
            f"adverse={r.get(f'adverse_markout_{h}s_pct', np.nan):.2f}%"
        )

    print("\nINVENTORY")
    print(f"  Both sides filled:        {r['both_sides_contract_pct']:.2f}% of contracts")
    print(f"  One-sided / stuck:        {r['one_sided_contract_pct']:.2f}%")
    print(f"  No fill:                  {r['no_fill_contract_pct']:.2f}%")
    print(f"  Avg max |inventory|:      {r['avg_max_abs_inventory']:.3f} ct")
    print(f"  P95 max |inventory|:      {r['p95_max_abs_inventory']:.3f} ct")

    print("\nECONOMICS — BEFORE FEES")
    print(f"  Gross spread/edge capture:${r['gross_spread_capture_dollars']:+.4f}")
    print(f"  Adverse selection to M5:  ${r['adverse_selection_to_m5_dollars']:+.4f}")
    print(f"  Net M1-M5 MTM:            ${r['net_mtm_pnl_before_fees']:+.4f}")
    print(f"  Matched round-trip PnL:   ${r['matched_roundtrip_pnl']:+.4f} on {r['matched_roundtrip_qty']:.2f} ct")
    print(f"  PnL / eligible contract:  ${r['pnl_per_eligible_contract']:+.5f}")
    print(f"  PnL / filled contract:    ${r['pnl_per_filled_contract']:+.5f}")
    print(f"  PnL / 15m window:         ${r['pnl_per_window']:+.5f}")
    print(f"  Break-even fee / fill qty:{r['break_even_fee_c_per_filled_qty']:+.3f} c")

    print("\nTAIL / RISK")
    print(f"  Worst contract:           ${r['worst_contract_pnl']:+.4f}")
    print(f"  Best contract:            ${r['best_contract_pnl']:+.4f}")
    print(f"  Contract P10/median/P90:  ${r['contract_p10_pnl']:+.4f} / ${r['contract_median_pnl']:+.4f} / ${r['contract_p90_pnl']:+.4f}")
    print(f"  Worst 15m window:         ${r['worst_window_pnl']:+.4f}")
    print(f"  Best 15m window:          ${r['best_window_pnl']:+.4f}")
    print(f"  Window P10/median/P90:    ${r['window_p10_pnl']:+.4f} / ${r['window_median_pnl']:+.4f} / ${r['window_p90_pnl']:+.4f}")
    print(f"  Max drawdown:             ${r['max_drawdown']:+.4f}")

    print("\nBY ASSET")
    if len(asset_df):
        cols = [
            "series", "eligible_contracts", "contracts_with_fill", "contract_fill_pct", "fill_qty",
            "both_sides_pct", "one_sided_pct", "avg_max_abs_inventory",
            "gross_spread_capture_dollars", "adverse_selection_to_m5_dollars",
            "net_mtm_pnl_before_fees", "pnl_per_eligible_contract", "matched_roundtrip_pnl",
        ]
        print(asset_df[cols].round(4).to_string(index=False))

    print("\nInterpretation rule: this is the unconditional Q1/MM baseline. Do not select a regime from this same run.")
    print("Fees are not applied; the reported break-even fee says how much edge per filled contract remains for fees.")
    print("Outputs:", output_dir)
    print("=" * 108)


def run_reconstructed_m1_m5_mm_backtest(
    session_dir,
    reconstruction_dir,
    output_dir=None,
    *,
    quote_qty=1.0,
    markout_seconds=DEFAULT_MARKOUT_SECONDS,
    max_markout_lag_s=2.0,
    show=True,
):
    session = Path(session_dir)
    recon = Path(reconstruction_dir)
    if not session.exists():
        raise FileNotFoundError(session)
    if not recon.exists():
        raise FileNotFoundError(recon)
    if quote_qty <= 0:
        raise ValueError("quote_qty must be positive")
    markouts = tuple(sorted({int(x) for x in markout_seconds if int(x) > 0}))

    quality_df, meta = _load_quality_contracts(recon)
    eligible = set(meta)
    print(f"Validated reconstruction contracts: {len(eligible):,}")
    print("Loading 1 Hz reconstructed books for frozen-gate contracts...")
    samples, sample_stats = _load_reconstructed_samples(recon, eligible)
    missing_samples = sorted(eligible - set(samples))
    if missing_samples:
        raise RuntimeError(f"{len(missing_samples)} eligible contracts have no reconstructed samples; first={missing_samples[:3]}")

    print(f"Streaming trades for {len(eligible):,} eligible contracts...")
    trades, trade_stats = _scan_trades(session, meta)

    if output_dir is None:
        root = session.resolve().parents[2] if len(session.resolve().parents) >= 3 else Path.cwd()
        output_dir = root / "results" / "kalshi_mm_m1_m5_reconstructed_q1" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_episodes, all_fills, contracts = [], [], []
    targets = sorted(eligible, key=lambda x: (meta[x]["close_ts"], x))
    t0 = time.time()
    print(f"Replaying Q1 two-sided MM on {len(targets):,} contracts...")
    for i, ticker in enumerate(targets, 1):
        eps, fills, contract = _simulate_contract(
            ticker, meta[ticker], samples[ticker], trades.get(ticker, []),
            float(quote_qty), markouts, float(max_markout_lag_s),
        )
        all_episodes.extend(eps)
        all_fills.extend(fills)
        if contract is not None:
            contracts.append(contract)
        if i % 100 == 0 or i == len(targets):
            print(f"  replayed {i:,}/{len(targets):,} | fills={len(all_fills):,} | {time.time()-t0:.1f}s")

    contract_df = pd.DataFrame(contracts)
    if len(contract_df) != len(targets):
        raise RuntimeError(f"Only {len(contract_df)}/{len(targets)} eligible contracts produced summaries")
    episodes_df = pd.DataFrame(all_episodes)
    fills_df = pd.DataFrame(all_fills)
    side_df = _side_summary(all_episodes, all_fills, markouts)
    asset_df = _asset_summary(contract_df)
    minute_df = _minute_summary(all_fills, markouts)
    window_df = _window_summary(contract_df)
    headline = _headline(contract_df, side_df, all_fills, window_df, markouts)

    episodes_df.to_csv(out / "quote_episodes.csv", index=False)
    fills_df.to_csv(out / "fills.csv", index=False)
    contract_df.to_csv(out / "contract_summary.csv", index=False)
    side_df.to_csv(out / "side_summary.csv", index=False)
    asset_df.to_csv(out / "asset_summary.csv", index=False)
    minute_df.to_csv(out / "minute_fill_summary.csv", index=False)
    window_df.to_csv(out / "window_summary.csv", index=False)
    headline.to_csv(out / "headline_summary.csv", index=False)
    pd.DataFrame([{**sample_stats, **trade_stats}]).to_csv(out / "scan_stats.csv", index=False)

    config = {
        "study_version": STUDY_VERSION,
        "session": str(session.resolve()),
        "reconstruction_dir": str(recon.resolve()),
        "eligible_contracts": len(targets),
        "quote_qty": quote_qty,
        "quote_window": "M1-M5",
        "quote_refresh": "1 Hz reconstructed BBO; keep priority if side price unchanged; reprice on side BBO change",
        "invalid_book_policy": "cancel both sides until next VALID reconstructed sample",
        "queue_model": "join back of displayed L1; exact aggressive flow depletes queue; no cancellation credit; trade-through fills remainder",
        "ask_representation": "short YES-equivalent == passive NO bid at complement",
        "markout_seconds": list(markouts),
        "max_markout_lag_s": max_markout_lag_s,
        "m5_inventory": "marked to last valid reconstructed M1-M5 midpoint",
        "tail_policy": "no fills credited after last reconstructed 1 Hz sample",
        "fees": "excluded; break-even fee per filled qty reported",
        "purpose": "unconditional MM feasibility baseline; no regime selection",
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if show:
        _print_report(headline, side_df, asset_df, window_df, out, markouts)
        try:
            from IPython.display import display
            print("\nSIDE SUMMARY")
            display(side_df.round(4))
            print("\nWORST 15 WINDOWS")
            display(window_df.nsmallest(15, "net_mtm_pnl_before_fees").round(4))
            print("\nBEST 15 WINDOWS")
            display(window_df.nlargest(15, "net_mtm_pnl_before_fees").round(4))
        except Exception:
            pass

    return {
        "output_dir": out,
        "headline": headline,
        "contracts": contract_df,
        "fills": fills_df,
        "quote_episodes": episodes_df,
        "side_summary": side_df,
        "asset_summary": asset_df,
        "minute_summary": minute_df,
        "window_summary": window_df,
    }


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--session", required=True)
    p.add_argument("--reconstruction-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--quote-qty", type=float, default=1.0)
    args = p.parse_args()
    run_reconstructed_m1_m5_mm_backtest(
        args.session,
        args.reconstruction_dir,
        output_dir=args.output_dir,
        quote_qty=args.quote_qty,
        show=True,
    )


if __name__ == "__main__":
    _main()
