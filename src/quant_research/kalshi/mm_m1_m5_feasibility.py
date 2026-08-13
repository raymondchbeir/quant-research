from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

STUDY_VERSION = "M1_M5_MM_FEASIBILITY_V1"
CRYPTO_SERIES = {
    "KXBTC15M",
    "KXBNB15M",
    "KXDOGE15M",
    "KXETH15M",
    "KXHYPE15M",
    "KXNEAR15M",
    "KXSOL15M",
    "KXXRP15M",
    "KXZEC15M",
}
DEFAULT_MARKOUT_SECONDS = (5, 15, 30, 60)
_TICKER_CLOSE_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{6})-\d{2}$")
EPS = 1e-9


@dataclass(slots=True)
class BookSnap:
    t: float
    minute: float
    bid1: float
    bid1_qty: float
    bid2: float
    bid2_qty: float
    ask1: float
    ask1_qty: float
    ask2: float
    ask2_qty: float
    mid: float
    spread: float
    imbalance: float


@dataclass(slots=True)
class Trade:
    t: float
    yes_price: float
    qty: float
    taker_book_side: str


def _float(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _price(x):
    p = _float(x)
    if np.isfinite(p) and p > 1.5:
        p /= 100.0
    return p


def _ts_seconds(x):
    if x is None:
        return np.nan
    try:
        if isinstance(x, (int, float)):
            z = float(x)
            if z > 1e17:
                return z / 1e9
            if z > 1e14:
                return z / 1e6
            if z > 1e11:
                return z / 1e3
            return z
        s = str(x).strip()
        if not s:
            return np.nan
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).timestamp()
    except Exception:
        return np.nan


def _iso(ts):
    if not np.isfinite(ts):
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _event_ts(obj):
    if not isinstance(obj, dict):
        return np.nan
    for key in ("received_ts", "recv_ts", "received_at", "time", "timestamp", "ts", "created_time"):
        if obj.get(key) is not None:
            out = _ts_seconds(obj.get(key))
            if np.isfinite(out):
                return out
    return np.nan


def _get_ticker(obj):
    if not isinstance(obj, dict):
        return None
    raw = obj.get("raw_msg") if isinstance(obj.get("raw_msg"), dict) else {}
    value = obj.get("ticker") or obj.get("market_ticker") or raw.get("market_ticker")
    return None if value is None else str(value)


def _close_from_ticker(ticker):
    m = _TICKER_CLOSE_RE.search(str(ticker or ""))
    if not m:
        return np.nan
    try:
        dt = datetime.strptime(m.group(1), "%y%b%d%H%M")
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
        return dt.astimezone(timezone.utc).timestamp()
    except Exception:
        return np.nan


def _market_close_ts(row, ticker):
    for key in ("market_close", "close_time", "close_ts"):
        if row.get(key) is not None:
            ts = _ts_seconds(row.get(key))
            if np.isfinite(ts):
                return ts
    return _close_from_ticker(ticker)


def _levels(rows):
    out = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        p, q = _price(row[0]), _float(row[1])
        if np.isfinite(p) and np.isfinite(q) and 0 <= p <= 1 and q > 0:
            out.append((float(p), float(q)))
    return out


def _normalize_book(row, close_ts, start_minute):
    """Normalize recorder full_books rows into the YES book.

    recorder_core.full_book_row writes:
      yes_bids <- YES bid book
      yes_asks <- raw NO bid book (historical field name)

    Therefore the executable YES ask ladder is the complement of the raw NO bids:
      YES ask = 1 - NO bid.
    """
    t = _event_ts(row)
    if not np.isfinite(t):
        return None

    yes_bids = sorted(_levels(row.get("yes_bids") or []), key=lambda z: z[0], reverse=True)
    raw_no_bids = _levels(row.get("yes_asks") or [])
    yes_asks = sorted([(1.0 - p, q) for p, q in raw_no_bids], key=lambda z: z[0])

    if not yes_bids or not yes_asks:
        return None

    bid1, bid1q = yes_bids[0]
    ask1, ask1q = yes_asks[0]
    if not (0 <= bid1 < ask1 <= 1):
        return None

    bid2, bid2q = (yes_bids[1] if len(yes_bids) > 1 else (np.nan, 0.0))
    ask2, ask2q = (yes_asks[1] if len(yes_asks) > 1 else (np.nan, 0.0))
    mid = 0.5 * (bid1 + ask1)
    spread = ask1 - bid1
    den = bid1q + ask1q
    imbalance = (bid1q - ask1q) / den if den > 0 else np.nan
    contract_start = close_ts - 15.0 * 60.0
    minute = (t - contract_start) / 60.0

    return BookSnap(
        t=float(t), minute=float(minute),
        bid1=float(bid1), bid1_qty=float(bid1q),
        bid2=float(bid2) if np.isfinite(bid2) else np.nan, bid2_qty=float(bid2q),
        ask1=float(ask1), ask1_qty=float(ask1q),
        ask2=float(ask2) if np.isfinite(ask2) else np.nan, ask2_qty=float(ask2q),
        mid=float(mid), spread=float(spread), imbalance=float(imbalance),
    )


def _parse_trade(obj):
    ticker = _get_ticker(obj)
    if not ticker:
        return None
    raw = obj.get("raw_msg") if isinstance(obj.get("raw_msg"), dict) else {}
    p = _price(obj.get("yes_price") if obj.get("yes_price") is not None else raw.get("yes_price_dollars"))
    qty = _float(raw.get("count_fp") if raw.get("count_fp") is not None else obj.get("qty"))
    side = str(raw.get("taker_book_side") or obj.get("taker_book_side") or "").lower()
    ts = _event_ts(obj)
    if not np.isfinite(p) or not np.isfinite(qty) or qty <= 0 or side not in {"bid", "ask"} or not np.isfinite(ts):
        return None
    return ticker, Trade(float(ts), float(p), float(qty), side)


def _prefix_regex(series):
    parts = [re.escape((s + "-").encode("utf-8")) for s in sorted(series)]
    return re.compile(rb"(?:" + rb"|".join(parts) + rb")") if parts else None


def _spread_bucket(spread_c):
    if not np.isfinite(spread_c):
        return "NA"
    if spread_c <= 1.0 + EPS:
        return "<=1c"
    if spread_c <= 2.0 + EPS:
        return "1-2c"
    if spread_c <= 4.0 + EPS:
        return "2-4c"
    return ">4c"


def _queue_bucket(q):
    if not np.isfinite(q):
        return "NA"
    if q <= EPS:
        return "0"
    if q <= 10:
        return "0-10"
    if q <= 50:
        return "10-50"
    if q <= 200:
        return "50-200"
    return ">200"


def _mid_bucket(mid_c):
    if not np.isfinite(mid_c):
        return "NA"
    lo = int(max(0, min(80, math.floor(mid_c / 20.0) * 20)))
    return f"{lo:02d}-{lo+20:02d}c"


def _minute_bucket(minute):
    if not np.isfinite(minute):
        return "NA"
    lo = int(math.floor(minute))
    return f"M{lo}-M{lo+1}"


def _imbalance_bucket(x):
    if not np.isfinite(x):
        return "NA"
    if x <= -0.5:
        return "<=-0.5"
    if x <= -0.2:
        return "-0.5:-0.2"
    if x < 0.2:
        return "-0.2:0.2"
    if x < 0.5:
        return "0.2:0.5"
    return ">=0.5"


def _flow_bucket(q):
    if not np.isfinite(q) or q <= EPS:
        return "0"
    if q <= 10:
        return "0-10"
    if q <= 50:
        return "10-50"
    if q <= 200:
        return "50-200"
    return ">200"


def _quality_gate(snaps, window_start, window_end, expected_hz, min_coverage_pct, max_edge_gap_s):
    q = [s for s in snaps if window_start <= s.t < window_end]
    expected = max(1.0, (window_end - window_start) * expected_hz)
    coverage = min(100.0, 100.0 * len(q) / expected)
    if len(q) < 2:
        return False, coverage, np.nan, np.nan, "<2 quote-window book samples"
    start_gap = max(0.0, q[0].t - window_start)
    end_gap = max(0.0, window_end - q[-1].t)
    reasons = []
    if coverage < min_coverage_pct:
        reasons.append(f"book coverage {coverage:.1f}% < {min_coverage_pct:.1f}%")
    if start_gap > max_edge_gap_s:
        reasons.append(f"start gap {start_gap:.2f}s > {max_edge_gap_s:.2f}s")
    if end_gap > max_edge_gap_s:
        reasons.append(f"end gap {end_gap:.2f}s > {max_edge_gap_s:.2f}s")
    return not reasons, coverage, start_gap, end_gap, "; ".join(reasons) if reasons else "OK"


def _scan_books(session_dir, series, start_minute, end_minute, max_markout_s, min_book_coverage_pct, max_edge_gap_s, book_writer=None):
    path = Path(session_dir) / "full_books.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    prefix_re = _prefix_regex(series)
    books = defaultdict(list)
    meta = {}
    scanned = decoded = normalized = invalid_orientation = 0
    keep_pre_s = 30.0
    start_elapsed = start_minute * 60.0
    end_elapsed = end_minute * 60.0
    low_elapsed = max(0.0, start_elapsed - keep_pre_s)
    high_elapsed = end_elapsed + max_markout_s + 3.0

    t0 = time.time()
    with path.open("rb") as f:
        for raw in f:
            scanned += 1
            if prefix_re is not None and prefix_re.search(raw) is None:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            decoded += 1
            ticker = _get_ticker(row)
            if not ticker:
                continue
            s = ticker.split("-")[0]
            if s not in series:
                continue
            close_ts = _market_close_ts(row, ticker)
            if not np.isfinite(close_ts):
                continue
            snap = _normalize_book(row, close_ts, start_minute)
            if snap is None:
                invalid_orientation += 1
                continue
            elapsed = snap.t - (close_ts - 900.0)
            if not (low_elapsed <= elapsed <= high_elapsed):
                continue
            normalized += 1
            books[ticker].append(snap)
            meta[ticker] = {
                "ticker": ticker,
                "series": s,
                "close_ts": close_ts,
                "window_start": close_ts - 900.0 + start_elapsed,
                "window_end": close_ts - 900.0 + end_elapsed,
            }

    excluded = []
    good = {}
    for ticker, snaps in books.items():
        snaps.sort(key=lambda z: z.t)
        m = meta[ticker]
        ok, coverage, start_gap, end_gap, reason = _quality_gate(
            snaps, m["window_start"], m["window_end"], 1.0,
            min_book_coverage_pct, max_edge_gap_s,
        )
        m.update({
            "book_coverage_pct": coverage,
            "start_gap_s": start_gap,
            "end_gap_s": end_gap,
            "quality_ok": ok,
            "quality_reason": reason,
        })
        if ok:
            good[ticker] = snaps
        else:
            excluded.append(dict(m))

    if book_writer is not None:
        for ticker, snaps in books.items():
            m = meta[ticker]
            for s in snaps:
                if m["window_start"] <= s.t < m["window_end"]:
                    book_writer.writerow({
                        "session": Path(session_dir).name,
                        "ticker": ticker,
                        "series": m["series"],
                        "time": _iso(s.t),
                        "minute": s.minute,
                        "yes_bid1": s.bid1,
                        "yes_bid1_qty": s.bid1_qty,
                        "yes_bid2": s.bid2,
                        "yes_bid2_qty": s.bid2_qty,
                        "yes_ask1": s.ask1,
                        "yes_ask1_qty": s.ask1_qty,
                        "yes_ask2": s.ask2,
                        "yes_ask2_qty": s.ask2_qty,
                        "mid": s.mid,
                        "spread_c": 100.0 * s.spread,
                        "l1_imbalance": s.imbalance,
                        "quality_ok": m["quality_ok"],
                    })

    return {
        "books": good,
        "meta": meta,
        "excluded": excluded,
        "stats": {
            "full_book_lines_scanned": scanned,
            "crypto_book_lines_decoded": decoded,
            "book_samples_kept": normalized,
            "invalid_book_rows": invalid_orientation,
            "contracts_seen": len(books),
            "contracts_quality": len(good),
            "seconds": time.time() - t0,
        },
    }


def _scan_trades(session_dir, meta, good_tickers, series):
    path = Path(session_dir) / "trades.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    prefix_re = _prefix_regex(series)
    trades = defaultdict(list)
    scanned = decoded = kept = 0
    t0 = time.time()
    with path.open("rb") as f:
        for raw in f:
            scanned += 1
            if prefix_re is not None and prefix_re.search(raw) is None:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            decoded += 1
            parsed = _parse_trade(obj)
            if parsed is None:
                continue
            ticker, tr = parsed
            if ticker not in good_tickers:
                continue
            m = meta[ticker]
            if m["window_start"] - 30.0 <= tr.t < m["window_end"]:
                trades[ticker].append(tr)
                kept += 1
    for v in trades.values():
        v.sort(key=lambda z: z.t)
    return trades, {
        "trade_lines_scanned": scanned,
        "crypto_trade_lines_decoded": decoded,
        "trades_kept": kept,
        "seconds": time.time() - t0,
    }


class _TradePrefix:
    def __init__(self, trades):
        self.trades = trades
        self.times = [t.t for t in trades]
        self.buy_cum = [0.0]
        self.sell_cum = [0.0]
        for tr in trades:
            self.buy_cum.append(self.buy_cum[-1] + (tr.qty if tr.taker_book_side == "bid" else 0.0))
            self.sell_cum.append(self.sell_cum[-1] + (tr.qty if tr.taker_book_side == "ask" else 0.0))

    def window(self, a, b):
        i = bisect.bisect_left(self.times, a)
        j = bisect.bisect_left(self.times, b)
        buy = self.buy_cum[j] - self.buy_cum[i]
        sell = self.sell_cum[j] - self.sell_cum[i]
        total = buy + sell
        return buy, sell, total, ((buy - sell) / total if total > 0 else 0.0)


def _pre_mid_range(snaps, times, t, lookback=30.0):
    i = bisect.bisect_left(times, t - lookback)
    j = bisect.bisect_left(times, t)
    if j <= i:
        return 0.0
    mids = [snaps[k].mid for k in range(i, j) if np.isfinite(snaps[k].mid)]
    return 100.0 * (max(mids) - min(mids)) if mids else 0.0


def _episode_record(session, ticker, series, side, episode_id, snap, price, queue, l2_price, l2_qty, trade_prefix, snaps, snap_times):
    buy30, sell30, vol30, flow30 = trade_prefix.window(snap.t - 30.0, snap.t)
    return {
        "session": session,
        "ticker": ticker,
        "series": series,
        "episode_id": episode_id,
        "side": side,
        "join_time": _iso(snap.t),
        "join_ts": snap.t,
        "join_minute": snap.minute,
        "price": price,
        "queue_ahead_initial": queue,
        "queue_ahead_final": queue,
        "l2_price": l2_price,
        "l2_qty": l2_qty,
        "mid_at_join": snap.mid,
        "spread_c": 100.0 * snap.spread,
        "l1_imbalance": snap.imbalance,
        "pre30_aggr_buy_qty": buy30,
        "pre30_aggr_sell_qty": sell30,
        "pre30_aggr_total_qty": vol30,
        "pre30_flow_imbalance": flow30,
        "pre30_mid_range_c": _pre_mid_range(snaps, snap_times, snap.t, 30.0),
        "fill_qty": 0.0,
        "first_fill_ts": np.nan,
        "last_fill_ts": np.nan,
        "fill_latency_s": np.nan,
        "cancel_ts": np.nan,
        "cancel_reason": None,
        "complete_fill": False,
    }


def _close_episode(ep, t, reason):
    if ep is None:
        return
    if not np.isfinite(ep["cancel_ts"]):
        ep["cancel_ts"] = t
        ep["cancel_reason"] = reason
    ep["duration_s"] = max(0.0, ep["cancel_ts"] - ep["join_ts"])
    ep["filled_any"] = bool(ep["fill_qty"] > EPS)
    ep["fill_rate_qty"] = ep["fill_qty"]
    ep["spread_bucket"] = _spread_bucket(ep["spread_c"])
    ep["queue_bucket"] = _queue_bucket(ep["queue_ahead_initial"])
    ep["mid_bucket"] = _mid_bucket(100.0 * ep["mid_at_join"])
    ep["minute_bucket"] = _minute_bucket(ep["join_minute"])
    ep["imbalance_bucket"] = _imbalance_bucket(ep["l1_imbalance"])
    ep["flow30_bucket"] = _flow_bucket(ep["pre30_aggr_total_qty"])


def _future_snap(snaps, times, target, max_lag_s=3.0):
    i = bisect.bisect_left(times, target)
    if i >= len(snaps):
        return None
    s = snaps[i]
    return s if s.t - target <= max_lag_s else None


def _last_snap_at_or_before(snaps, times, t):
    i = bisect.bisect_right(times, t) - 1
    return snaps[i] if i >= 0 else None


def _match_cycle_pnl(fills):
    longs = deque()
    shorts = deque()
    pnl = qty_done = 0.0
    for f in sorted(fills, key=lambda x: x["fill_ts"]):
        qty = float(f["qty"])
        price = float(f["price"])
        if f["side"] == "BID":
            while qty > EPS and shorts:
                sell_price, q0 = shorts[0]
                m = min(qty, q0)
                pnl += (sell_price - price) * m
                qty_done += m
                qty -= m
                q0 -= m
                if q0 <= EPS:
                    shorts.popleft()
                else:
                    shorts[0] = (sell_price, q0)
            if qty > EPS:
                longs.append((price, qty))
        else:
            while qty > EPS and longs:
                buy_price, q0 = longs[0]
                m = min(qty, q0)
                pnl += (price - buy_price) * m
                qty_done += m
                qty -= m
                q0 -= m
                if q0 <= EPS:
                    longs.popleft()
                else:
                    longs[0] = (buy_price, q0)
            if qty > EPS:
                shorts.append((price, qty))
    return qty_done, pnl


def _simulate_contract(session_name, ticker, meta, snaps, trades, quote_qty, markouts, inventory_writer=None):
    series = meta["series"]
    wstart, wend = meta["window_start"], meta["window_end"]
    quote_snaps = [s for s in snaps if wstart <= s.t < wend]
    if len(quote_snaps) < 2:
        return [], [], None

    snap_times = [s.t for s in snaps]
    trade_prefix = _TradePrefix(trades)
    episodes = []
    fills = []
    active = {"BID": None, "ASK": None}
    remaining = {"BID": 0.0, "ASK": 0.0}
    episode_counter = 0
    inventory = cash = 0.0
    max_abs_inventory = 0.0
    trade_times = [t.t for t in trades]
    tr_idx = bisect.bisect_left(trade_times, quote_snaps[0].t)

    def open_order(side, snap):
        nonlocal episode_counter
        episode_counter += 1
        if side == "BID":
            price, queue, l2p, l2q = snap.bid1, snap.bid1_qty, snap.bid2, snap.bid2_qty
        else:
            price, queue, l2p, l2q = snap.ask1, snap.ask1_qty, snap.ask2, snap.ask2_qty
        ep_id = f"{session_name}:{ticker}:{side}:{episode_counter}"
        ep = _episode_record(
            session_name, ticker, series, side, ep_id, snap,
            price, queue, l2p, l2q, trade_prefix, snaps, snap_times,
        )
        episodes.append(ep)
        active[side] = ep
        remaining[side] = quote_qty

    def cancel(side, t, reason):
        ep = active[side]
        if ep is not None:
            _close_episode(ep, t, reason)
        active[side] = None
        remaining[side] = 0.0

    def apply_trade(tr):
        nonlocal inventory, cash, max_abs_inventory
        if tr.taker_book_side == "ask":
            side = "BID"
            ep = active[side]
            if ep is None:
                return
            qpx = ep["price"]
            exact = abs(tr.yes_price - qpx) <= EPS
            through = tr.yes_price < qpx - EPS
        else:
            side = "ASK"
            ep = active[side]
            if ep is None:
                return
            qpx = ep["price"]
            exact = abs(tr.yes_price - qpx) <= EPS
            through = tr.yes_price > qpx + EPS

        if not exact and not through:
            return

        fill_qty = 0.0
        if through:
            fill_qty = remaining[side]
        elif exact:
            qa = max(0.0, float(ep["queue_ahead_final"]))
            if tr.qty <= qa + EPS:
                ep["queue_ahead_final"] = max(0.0, qa - tr.qty)
                return
            ep["queue_ahead_final"] = 0.0
            fill_qty = min(remaining[side], tr.qty - qa)

        if fill_qty <= EPS:
            return

        mid_snap = _last_snap_at_or_before(snaps, snap_times, tr.t)
        if mid_snap is None:
            return

        ep["fill_qty"] += fill_qty
        if not np.isfinite(ep["first_fill_ts"]):
            ep["first_fill_ts"] = tr.t
            ep["fill_latency_s"] = tr.t - ep["join_ts"]
        ep["last_fill_ts"] = tr.t
        remaining[side] -= fill_qty

        sign = 1.0 if side == "BID" else -1.0
        gross_edge_c = sign * (mid_snap.mid - ep["price"]) * 100.0
        fill = {
            "session": session_name,
            "ticker": ticker,
            "series": series,
            "episode_id": ep["episode_id"],
            "side": side,
            "fill_time": _iso(tr.t),
            "fill_ts": tr.t,
            "qty": fill_qty,
            "price": ep["price"],
            "mid_at_fill": mid_snap.mid,
            "spread_c_at_join": ep["spread_c"],
            "queue_ahead_initial": ep["queue_ahead_initial"],
            "fill_latency_s": ep["fill_latency_s"],
            "gross_edge_at_fill_c": gross_edge_c,
            "join_minute": ep["join_minute"],
            "pre30_aggr_total_qty": ep["pre30_aggr_total_qty"],
            "pre30_flow_imbalance": ep["pre30_flow_imbalance"],
            "pre30_mid_range_c": ep["pre30_mid_range_c"],
            "l1_imbalance": ep["l1_imbalance"],
        }
        for h in markouts:
            fs = _future_snap(snaps, snap_times, tr.t + h)
            if fs is None:
                fill[f"future_mid_{h}s"] = np.nan
                fill[f"post_mid_move_{h}s_c"] = np.nan
                fill[f"markout_{h}s_c"] = np.nan
            else:
                fill[f"future_mid_{h}s"] = fs.mid
                fill[f"post_mid_move_{h}s_c"] = sign * (fs.mid - mid_snap.mid) * 100.0
                fill[f"markout_{h}s_c"] = sign * (fs.mid - ep["price"]) * 100.0
        fills.append(fill)

        if side == "BID":
            inventory += fill_qty
            cash -= ep["price"] * fill_qty
        else:
            inventory -= fill_qty
            cash += ep["price"] * fill_qty
        max_abs_inventory = max(max_abs_inventory, abs(inventory))

        if inventory_writer is not None:
            inventory_writer.writerow({
                "session": session_name,
                "ticker": ticker,
                "series": series,
                "time": _iso(tr.t),
                "event": f"{side}_FILL",
                "inventory_yes_equiv": inventory,
                "cash": cash,
                "mid": mid_snap.mid,
                "mtm_pnl": cash + inventory * mid_snap.mid,
            })

        if remaining[side] <= EPS:
            ep["complete_fill"] = True
            _close_episode(ep, tr.t, "FILLED")
            active[side] = None
            remaining[side] = 0.0

    open_order("BID", quote_snaps[0])
    open_order("ASK", quote_snaps[0])
    last_snap_t = quote_snaps[0].t

    for snap in quote_snaps[1:]:
        while tr_idx < len(trades) and trades[tr_idx].t <= snap.t + EPS:
            tr = trades[tr_idx]
            if tr.t > last_snap_t + EPS:
                apply_trade(tr)
            tr_idx += 1

        for side, px in (("BID", snap.bid1), ("ASK", snap.ask1)):
            ep = active[side]
            if ep is None:
                open_order(side, snap)
            elif abs(ep["price"] - px) > EPS:
                cancel(side, snap.t, "BBO_REPRICE")
                open_order(side, snap)
        last_snap_t = snap.t

    while tr_idx < len(trades) and trades[tr_idx].t < wend - EPS:
        tr = trades[tr_idx]
        if tr.t > last_snap_t + EPS:
            apply_trade(tr)
        tr_idx += 1

    cancel("BID", wend, "M5_END")
    cancel("ASK", wend, "M5_END")

    final_snap = _last_snap_at_or_before(snaps, snap_times, wend)
    final_mid = final_snap.mid if final_snap is not None else quote_snaps[-1].mid
    net_mtm = cash + inventory * final_mid
    gross_capture = sum((f["gross_edge_at_fill_c"] / 100.0) * f["qty"] for f in fills)
    adverse_to_end = net_mtm - gross_capture
    matched_qty, matched_pnl = _match_cycle_pnl(fills)

    bid_qty = sum(f["qty"] for f in fills if f["side"] == "BID")
    ask_qty = sum(f["qty"] for f in fills if f["side"] == "ASK")
    both = bid_qty > EPS and ask_qty > EPS
    one = (bid_qty > EPS) ^ (ask_qty > EPS)

    qtrades = [t for t in trades if wstart <= t.t < wend]
    buy_qty = sum(t.qty for t in qtrades if t.taker_book_side == "bid")
    sell_qty = sum(t.qty for t in qtrades if t.taker_book_side == "ask")
    total_flow = buy_qty + sell_qty

    qs = quote_snaps
    summary = {
        "session": session_name,
        "ticker": ticker,
        "series": series,
        "close_time": _iso(meta["close_ts"]),
        "book_coverage_pct": meta["book_coverage_pct"],
        "book_samples": len(qs),
        "avg_spread_c": float(np.mean([100.0 * s.spread for s in qs])),
        "median_spread_c": float(np.median([100.0 * s.spread for s in qs])),
        "avg_bid_l1_qty": float(np.mean([s.bid1_qty for s in qs])),
        "avg_ask_l1_qty": float(np.mean([s.ask1_qty for s in qs])),
        "avg_bid_l2_qty": float(np.mean([s.bid2_qty for s in qs])),
        "avg_ask_l2_qty": float(np.mean([s.ask2_qty for s in qs])),
        "aggr_buy_trades": sum(1 for t in qtrades if t.taker_book_side == "bid"),
        "aggr_sell_trades": sum(1 for t in qtrades if t.taker_book_side == "ask"),
        "aggr_buy_qty": buy_qty,
        "aggr_sell_qty": sell_qty,
        "flow_imbalance": ((buy_qty - sell_qty) / total_flow if total_flow > 0 else 0.0),
        "bid_quote_episodes": sum(1 for e in episodes if e["side"] == "BID"),
        "ask_quote_episodes": sum(1 for e in episodes if e["side"] == "ASK"),
        "bid_filled_episodes": sum(1 for e in episodes if e["side"] == "BID" and e.get("filled_any")),
        "ask_filled_episodes": sum(1 for e in episodes if e["side"] == "ASK" and e.get("filled_any")),
        "bid_fill_qty": bid_qty,
        "ask_fill_qty": ask_qty,
        "both_sides_filled": both,
        "one_sided_fill": one,
        "no_fill": bid_qty <= EPS and ask_qty <= EPS,
        "max_abs_inventory": max_abs_inventory,
        "ending_inventory_yes_equiv": inventory,
        "final_mid_m5": final_mid,
        "cash": cash,
        "gross_spread_capture_dollars": gross_capture,
        "adverse_selection_to_m5_dollars": adverse_to_end,
        "net_mtm_pnl_before_fees": net_mtm,
        "matched_roundtrip_qty": matched_qty,
        "matched_roundtrip_pnl": matched_pnl,
    }
    for h in markouts:
        vals = [f[f"markout_{h}s_c"] for f in fills if np.isfinite(f[f"markout_{h}s_c"])]
        moves = [f[f"post_mid_move_{h}s_c"] for f in fills if np.isfinite(f[f"post_mid_move_{h}s_c"])]
        summary[f"mean_markout_{h}s_c"] = float(np.mean(vals)) if vals else np.nan
        summary[f"mean_post_mid_move_{h}s_c"] = float(np.mean(moves)) if moves else np.nan

    if inventory_writer is not None:
        inventory_writer.writerow({
            "session": session_name,
            "ticker": ticker,
            "series": series,
            "time": _iso(wend),
            "event": "M5_END",
            "inventory_yes_equiv": inventory,
            "cash": cash,
            "mid": final_mid,
            "mtm_pnl": net_mtm,
        })

    return episodes, fills, summary


def _regime_summary(episodes_df, fills_df, markouts):
    if episodes_df.empty:
        return pd.DataFrame()
    fill_cols = ["episode_id", "qty", "gross_edge_at_fill_c"]
    fill_cols += [f"markout_{h}s_c" for h in markouts]
    ff = fills_df[fill_cols].copy() if not fills_df.empty else pd.DataFrame(columns=fill_cols)
    if not ff.empty:
        agg_map = {"qty": "sum", "gross_edge_at_fill_c": "mean"}
        agg_map.update({f"markout_{h}s_c": "mean" for h in markouts})
        ff = ff.groupby("episode_id", as_index=False).agg(agg_map)
        ff = ff.rename(columns={"qty": "episode_fill_qty"})
    else:
        ff["episode_fill_qty"] = pd.Series(dtype=float)
    e = episodes_df.merge(ff, on="episode_id", how="left")
    e["episode_fill_qty"] = pd.to_numeric(e.get("episode_fill_qty"), errors="coerce").fillna(0.0)
    e["filled_any"] = e["episode_fill_qty"] > EPS

    dims = ["series", "side", "minute_bucket", "spread_bucket", "queue_bucket", "mid_bucket", "imbalance_bucket", "flow30_bucket"]
    rows = []
    for dim in dims:
        if dim not in e.columns:
            continue
        for (value, side), g in e.groupby([dim, "side"], dropna=False):
            filled = g[g["filled_any"]]
            row = {
                "dimension": dim,
                "value": str(value),
                "side": side,
                "episodes": len(g),
                "filled_episodes": int(g["filled_any"].sum()),
                "fill_rate_pct": 100.0 * g["filled_any"].mean() if len(g) else np.nan,
                "fill_qty": g["episode_fill_qty"].sum(),
                "avg_queue_ahead": pd.to_numeric(g["queue_ahead_initial"], errors="coerce").mean(),
                "avg_spread_c": pd.to_numeric(g["spread_c"], errors="coerce").mean(),
                "avg_fill_latency_s": pd.to_numeric(filled["fill_latency_s"], errors="coerce").mean() if len(filled) else np.nan,
                "avg_gross_edge_at_fill_c": pd.to_numeric(filled.get("gross_edge_at_fill_c"), errors="coerce").mean() if len(filled) else np.nan,
            }
            for h in markouts:
                row[f"avg_markout_{h}s_c"] = pd.to_numeric(filled.get(f"markout_{h}s_c"), errors="coerce").mean() if len(filled) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _side_summary(episodes_df, fills_df, markouts):
    rows = []
    for side in ("BID", "ASK", "ALL"):
        e = episodes_df if side == "ALL" else episodes_df[episodes_df["side"] == side]
        f = fills_df if side == "ALL" else fills_df[fills_df["side"] == side]
        row = {
            "side": side,
            "quote_episodes": len(e),
            "filled_episodes": int(e["filled_any"].sum()) if len(e) and "filled_any" in e else 0,
            "episode_fill_rate_pct": 100.0 * e["filled_any"].mean() if len(e) and "filled_any" in e else np.nan,
            "fill_events": len(f),
            "fill_qty": f["qty"].sum() if len(f) else 0.0,
            "avg_queue_ahead": pd.to_numeric(e.get("queue_ahead_initial"), errors="coerce").mean() if len(e) else np.nan,
            "avg_fill_latency_s": pd.to_numeric(f.get("fill_latency_s"), errors="coerce").mean() if len(f) else np.nan,
            "avg_gross_edge_at_fill_c": pd.to_numeric(f.get("gross_edge_at_fill_c"), errors="coerce").mean() if len(f) else np.nan,
        }
        for h in markouts:
            mo = pd.to_numeric(f.get(f"markout_{h}s_c"), errors="coerce") if len(f) else pd.Series(dtype=float)
            pm = pd.to_numeric(f.get(f"post_mid_move_{h}s_c"), errors="coerce") if len(f) else pd.Series(dtype=float)
            row[f"avg_markout_{h}s_c"] = mo.mean() if len(mo) else np.nan
            row[f"adverse_markout_{h}s_pct"] = 100.0 * (mo < 0).mean() if mo.notna().any() else np.nan
            row[f"avg_post_mid_move_{h}s_c"] = pm.mean() if len(pm) else np.nan
            row[f"adverse_mid_move_{h}s_pct"] = 100.0 * (pm < 0).mean() if pm.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _headline(contract_df, episodes_df, fills_df, side_df, markouts, sessions, config):
    q = contract_df
    row = {
        "study_version": STUDY_VERSION,
        "sessions": len(sessions),
        "quality_contracts": len(q),
        "quote_qty": config["quote_qty"],
        "start_minute": config["start_minute"],
        "end_minute": config["end_minute"],
        "quote_episodes": len(episodes_df),
        "fill_events": len(fills_df),
        "fill_qty": fills_df["qty"].sum() if len(fills_df) else 0.0,
        "both_sides_contract_pct": 100.0 * q["both_sides_filled"].mean() if len(q) else np.nan,
        "one_sided_contract_pct": 100.0 * q["one_sided_fill"].mean() if len(q) else np.nan,
        "no_fill_contract_pct": 100.0 * q["no_fill"].mean() if len(q) else np.nan,
        "avg_max_abs_inventory": q["max_abs_inventory"].mean() if len(q) else np.nan,
        "gross_spread_capture_dollars": q["gross_spread_capture_dollars"].sum() if len(q) else 0.0,
        "adverse_selection_to_m5_dollars": q["adverse_selection_to_m5_dollars"].sum() if len(q) else 0.0,
        "net_mtm_pnl_before_fees": q["net_mtm_pnl_before_fees"].sum() if len(q) else 0.0,
        "net_mtm_per_contract_before_fees": q["net_mtm_pnl_before_fees"].mean() if len(q) else np.nan,
        "matched_roundtrip_qty": q["matched_roundtrip_qty"].sum() if len(q) else 0.0,
        "matched_roundtrip_pnl": q["matched_roundtrip_pnl"].sum() if len(q) else 0.0,
    }
    if len(fills_df):
        row["avg_gross_edge_at_fill_c"] = pd.to_numeric(fills_df["gross_edge_at_fill_c"], errors="coerce").mean()
        for h in markouts:
            row[f"avg_markout_{h}s_c"] = pd.to_numeric(fills_df[f"markout_{h}s_c"], errors="coerce").mean()
            row[f"avg_post_mid_move_{h}s_c"] = pd.to_numeric(fills_df[f"post_mid_move_{h}s_c"], errors="coerce").mean()
    else:
        row["avg_gross_edge_at_fill_c"] = np.nan
        for h in markouts:
            row[f"avg_markout_{h}s_c"] = np.nan
            row[f"avg_post_mid_move_{h}s_c"] = np.nan
    for side in ("BID", "ASK"):
        z = side_df[side_df["side"] == side]
        if len(z):
            row[f"{side.lower()}_episode_fill_rate_pct"] = z.iloc[0]["episode_fill_rate_pct"]
    return pd.DataFrame([row])


def _print_report(headline, side_summary, output_dir):
    r = headline.iloc[0]
    print("=" * 96)
    print("M1-M5 TWO-SIDED MARKET-MAKING FEASIBILITY — EXPLORATORY / BEFORE FEES")
    print("=" * 96)
    print(f"Quality contracts: {int(r['quality_contracts'])}")
    print(f"Quote episodes:    {int(r['quote_episodes'])}")
    print(f"Fill events / qty: {int(r['fill_events'])} / {r['fill_qty']:.2f}")
    print()
    print("PASSIVE FILL MECHANICS")
    for _, x in side_summary[side_summary["side"].isin(["BID", "ASK"])].iterrows():
        print(
            f"  {x['side']:>3}: episode fill rate={x['episode_fill_rate_pct']:.2f}% | "
            f"avg queue={x['avg_queue_ahead']:.2f} | avg latency={x['avg_fill_latency_s']:.2f}s"
        )
    print()
    print("POST-FILL ECONOMICS")
    print(f"  Gross edge at fill:       {r['avg_gross_edge_at_fill_c']:+.3f} c/contract")
    for h in (5, 15, 30, 60):
        c = f"avg_markout_{h}s_c"
        if c in r.index:
            print(f"  {h:>2}s markout:              {r[c]:+.3f} c/contract")
    print()
    print("CONTINUOUS TWO-SIDED INVENTORY")
    print(f"  Both sides filled:        {r['both_sides_contract_pct']:.2f}% of contracts")
    print(f"  One-sided / stuck:        {r['one_sided_contract_pct']:.2f}%")
    print(f"  No fill:                  {r['no_fill_contract_pct']:.2f}%")
    print(f"  Avg max |inventory|:      {r['avg_max_abs_inventory']:.3f} ct")
    print()
    print("BIG DECOMPOSITION")
    print(f"  Gross spread capture:     ${r['gross_spread_capture_dollars']:+.4f}")
    print(f"  Adverse selection to M5:  ${r['adverse_selection_to_m5_dollars']:+.4f}")
    print(f"  Net M1-M5 MTM:            ${r['net_mtm_pnl_before_fees']:+.4f}  (BEFORE FEES)")
    print(f"  Net per quality contract: ${r['net_mtm_per_contract_before_fees']:+.5f}")
    print(f"  Matched round-trip PnL:   ${r['matched_roundtrip_pnl']:+.4f} on {r['matched_roundtrip_qty']:.2f} ct")
    print()
    print("Interpretation rule: positive spread capture is not enough; 15-60s markouts and M5 adverse selection")
    print("must remain positive after fills, and one-sided inventory must be tolerable. No threshold is selected here.")
    print()
    print("Outputs:", output_dir)
    print("=" * 96)


def run_m1_m5_mm_feasibility(
    session_dirs,
    output_dir=None,
    *,
    start_minute=1.0,
    end_minute=5.0,
    quote_qty=1.0,
    markout_seconds=DEFAULT_MARKOUT_SECONDS,
    crypto_series=None,
    min_book_coverage_pct=80.0,
    max_edge_gap_s=5.0,
    show=True,
):
    """Run an offline minute-1-to-minute-5 passive MM feasibility study.

    Simulation definition (fixed for feasibility, not optimized):
      * one passive quote_qty order at the current YES best bid and YES best ask;
      * join the back of displayed L1 FIFO queue;
      * cancel/replace only when the corresponding BBO price changes on the 1 Hz full-book snapshots;
      * queue depletion is credited only from recorded aggressive trades, never from cancellations;
      * trade-through fills the remaining tiny order; exact-price aggressive flow consumes queue ahead first;
      * after a fill, that side is rejoined at the next full-book snapshot;
      * quoting stops at minute 5 and inventory is marked to the minute-5 midpoint;
      * no fees are applied; this is a feasibility screen, not a production backtest.
    """
    if isinstance(session_dirs, (str, Path)):
        sessions = [Path(session_dirs)]
    else:
        sessions = [Path(x) for x in session_dirs]
    if not sessions:
        raise ValueError("session_dirs is empty")
    for s in sessions:
        if not s.exists():
            raise FileNotFoundError(s)
        for name in ("full_books.jsonl", "trades.jsonl"):
            if not (s / name).exists():
                raise FileNotFoundError(s / name)

    if not (0 <= start_minute < end_minute <= 15):
        raise ValueError("Require 0 <= start_minute < end_minute <= 15")
    if quote_qty <= 0:
        raise ValueError("quote_qty must be positive")
    markouts = tuple(sorted({int(x) for x in markout_seconds if int(x) > 0}))
    series = set(crypto_series or CRYPTO_SERIES)

    if output_dir is None:
        root = sessions[0].resolve().parents[2] if len(sessions[0].resolve().parents) >= 3 else Path.cwd()
        output_dir = root / "results" / "kalshi_mm_m1_m5" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "study_version": STUDY_VERSION,
        "sessions": [str(s.resolve()) for s in sessions],
        "crypto_series": sorted(series),
        "start_minute": float(start_minute),
        "end_minute": float(end_minute),
        "quote_qty": float(quote_qty),
        "markout_seconds": list(markouts),
        "min_book_coverage_pct": float(min_book_coverage_pct),
        "max_edge_gap_s": float(max_edge_gap_s),
        "book_schema_note": "full_books.yes_asks stores raw NO bids; YES asks are reconstructed as 1-NO_bid",
        "queue_model": "trade-only FIFO depletion; no cancellation credit; trade-through fills remaining tiny order",
        "reprice_model": "1 Hz full-book BBO cancel/replace",
        "fees": "excluded",
        "purpose": "exploratory feasibility only; no regime/threshold is selected",
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    book_fields = [
        "session", "ticker", "series", "time", "minute",
        "yes_bid1", "yes_bid1_qty", "yes_bid2", "yes_bid2_qty",
        "yes_ask1", "yes_ask1_qty", "yes_ask2", "yes_ask2_qty",
        "mid", "spread_c", "l1_imbalance", "quality_ok",
    ]
    inv_fields = ["session", "ticker", "series", "time", "event", "inventory_yes_equiv", "cash", "mid", "mtm_pnl"]

    all_episodes, all_fills, all_contracts, all_excluded = [], [], [], []
    scan_stats = []

    with (out / "book_samples.csv").open("w", newline="", encoding="utf-8") as bfh, \
         (out / "inventory_path.csv").open("w", newline="", encoding="utf-8") as ifh:
        bw = csv.DictWriter(bfh, fieldnames=book_fields)
        iw = csv.DictWriter(ifh, fieldnames=inv_fields)
        bw.writeheader()
        iw.writeheader()

        for session in sessions:
            if show:
                print(f"\n[{session.name}] scanning 1 Hz full books...")
            b = _scan_books(
                session, series, start_minute, end_minute,
                max(markouts) if markouts else 0,
                min_book_coverage_pct, max_edge_gap_s,
                book_writer=bw,
            )
            if show:
                st = b["stats"]
                print(
                    f"  contracts seen={st['contracts_seen']} | quality={st['contracts_quality']} | "
                    f"kept book samples={st['book_samples_kept']:,} | {st['seconds']:.1f}s"
                )
                if st["invalid_book_rows"]:
                    print(f"  invalid normalized book rows={st['invalid_book_rows']:,}")

            good_tickers = set(b["books"])
            all_excluded.extend([{**x, "session": session.name} for x in b["excluded"]])
            if not good_tickers:
                scan_stats.append({"session": session.name, **b["stats"]})
                continue

            if show:
                print(f"[{session.name}] streaming trades for {len(good_tickers)} quality contracts...")
            trades, ts = _scan_trades(session, b["meta"], good_tickers, series)
            if show:
                print(
                    f"  trade lines scanned={ts['trade_lines_scanned']:,} | "
                    f"crypto decoded={ts['crypto_trade_lines_decoded']:,} | kept={ts['trades_kept']:,} | {ts['seconds']:.1f}s"
                )
                print(f"[{session.name}] replaying two-sided FIFO quotes...")

            for ticker in sorted(good_tickers, key=lambda x: b["meta"][x]["close_ts"]):
                eps, fills, cs = _simulate_contract(
                    session.name, ticker, b["meta"][ticker], b["books"][ticker], trades.get(ticker, []),
                    float(quote_qty), markouts, inventory_writer=iw,
                )
                all_episodes.extend(eps)
                all_fills.extend(fills)
                if cs is not None:
                    all_contracts.append(cs)

            scan_stats.append({"session": session.name, **b["stats"], **ts})
            del trades, b

    episodes_df = pd.DataFrame(all_episodes)
    fills_df = pd.DataFrame(all_fills)
    contract_df = pd.DataFrame(all_contracts)
    excluded_df = pd.DataFrame(all_excluded)

    if not episodes_df.empty:
        for ep in all_episodes:
            if "filled_any" not in ep:
                _close_episode(ep, ep.get("cancel_ts", ep["join_ts"]), ep.get("cancel_reason") or "UNKNOWN")
        episodes_df = pd.DataFrame(all_episodes)

    side_df = _side_summary(episodes_df, fills_df, markouts) if not episodes_df.empty else pd.DataFrame()
    regime_df = _regime_summary(episodes_df, fills_df, markouts) if not episodes_df.empty else pd.DataFrame()
    headline = _headline(contract_df, episodes_df, fills_df, side_df, markouts, sessions, config)

    episodes_df.to_csv(out / "quote_episodes.csv", index=False)
    fills_df.to_csv(out / "fills.csv", index=False)
    contract_df.to_csv(out / "contract_summary.csv", index=False)
    excluded_df.to_csv(out / "excluded_contracts.csv", index=False)
    side_df.to_csv(out / "side_summary.csv", index=False)
    regime_df.to_csv(out / "regime_summary.csv", index=False)
    headline.to_csv(out / "headline_summary.csv", index=False)
    pd.DataFrame(scan_stats).to_csv(out / "scan_stats.csv", index=False)

    if show:
        _print_report(headline, side_df, out)
        if len(side_df):
            try:
                from IPython.display import display
                print("\nSIDE SUMMARY")
                display(side_df.round(4))
                if len(regime_df):
                    print("\nDESCRIPTIVE REGIME SUMMARY (first 40 rows; no threshold selected)")
                    display(regime_df.head(40).round(4))
            except Exception:
                pass

    return {
        "output_dir": out,
        "headline": headline,
        "side_summary": side_df,
        "regime_summary": regime_df,
        "contracts": contract_df,
        "fills": fills_df,
        "quote_episodes": episodes_df,
        "excluded_contracts": excluded_df,
        "scan_stats": pd.DataFrame(scan_stats),
    }


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--session", action="append", required=True, help="Recorder session directory; repeat for multiple sessions")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--start-minute", type=float, default=1.0)
    p.add_argument("--end-minute", type=float, default=5.0)
    p.add_argument("--quote-qty", type=float, default=1.0)
    args = p.parse_args()
    run_m1_m5_mm_feasibility(
        args.session,
        output_dir=args.output_dir,
        start_minute=args.start_minute,
        end_minute=args.end_minute,
        quote_qty=args.quote_qty,
        show=True,
    )


if __name__ == "__main__":
    _main()
