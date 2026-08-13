from __future__ import annotations

"""Event-time BBO validation for M1-M5 reconstructed Kalshi books.

This is a DATA VALIDATION study, not a trading backtest.

The recorder subscribed to orderbook_delta with use_yes_price=True, so the
recorded `no` side is already represented on the YES price scale. We maintain a
per-market book by:
  * seeding/resyncing only from structurally valid full_books rows;
  * applying recorded deltas in receipt-time order;
  * invalidating all books on recorded global sequence gaps;
  * invalidating a market when its connection epoch changes until a new valid
    full-book anchor appears.

Each ticker update is compared against the reconstructed BBO at that exact
receipt-time event. Results are broken out by asset and by age of the last book
update. No fill model, PnL, or market-making threshold is evaluated here.
"""

import csv
import heapq
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_m1_m5_feasibility as _base

STUDY_VERSION = "M1_M5_EVENT_TIME_BBO_VALIDATION_V1"
CRYPTO_SERIES = set(_base.CRYPTO_SERIES)
EPS = 1e-9


def _f(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _series(ticker):
    return str(ticker or "").split("-")[0]


def _event_ts(obj):
    return _base._event_ts(obj)


def _exchange_ts(obj):
    if not isinstance(obj, dict):
        return np.nan
    raw = obj.get("raw_msg") if isinstance(obj.get("raw_msg"), dict) else {}
    for value in (
        obj.get("ts_ms"), raw.get("ts_ms"), raw.get("timestamp"), raw.get("time")
    ):
        z = _base._ts_seconds(value)
        if np.isfinite(z):
            return float(z)
    return np.nan


def _close_ts(obj, ticker):
    return _base._market_close_ts(obj, ticker)


def _inside_window(t, close, start_minute, end_minute, pre_seconds=0.0):
    if not np.isfinite(t) or not np.isfinite(close):
        return False
    start = close - 900.0 + 60.0 * start_minute
    end = close - 900.0 + 60.0 * end_minute
    return start - pre_seconds <= t < end


def _valid_anchor(obj, allowed_series, start_minute, end_minute, pre_seconds):
    ticker = _base._get_ticker(obj)
    if not ticker or _series(ticker) not in allowed_series:
        return None
    t = _event_ts(obj)
    close = _close_ts(obj, ticker)
    if not _inside_window(t, close, start_minute, end_minute, pre_seconds):
        return None

    bids = sorted(_base._levels(obj.get("yes_bids") or []), key=lambda z: z[0], reverse=True)
    # use_yes_price=True: recorder's no-side levels are already YES-price scale.
    asks = sorted(_base._levels(obj.get("yes_asks") or []), key=lambda z: z[0])
    if not bids or not asks or not (0.0 <= bids[0][0] < asks[0][0] <= 1.0):
        return None

    source_t = _base._ts_seconds(obj.get("book_source_time"))
    exch_t = _base._ts_seconds(obj.get("book_exchange_time"))
    return {
        "kind": "anchor",
        "t": float(t),
        "ticker": ticker,
        "series": _series(ticker),
        "close": float(close),
        "epoch": obj.get("connection_epoch"),
        "seq": obj.get("book_seq"),
        "bids": bids,
        "asks": asks,
        "source_t": float(source_t) if np.isfinite(source_t) else float(t),
        "exchange_t": float(exch_t) if np.isfinite(exch_t) else np.nan,
    }


def _valid_delta(obj, allowed_series, start_minute, end_minute, pre_seconds):
    ticker = _base._get_ticker(obj)
    if not ticker or _series(ticker) not in allowed_series:
        return None
    t = _event_ts(obj)
    close = _close_ts(obj, ticker)
    if not _inside_window(t, close, start_minute, end_minute, pre_seconds):
        return None
    raw = obj.get("raw_msg") if isinstance(obj.get("raw_msg"), dict) else {}
    side = str(obj.get("side") or raw.get("side") or "").lower()
    price = _f(obj.get("price_dollars"))
    if not np.isfinite(price):
        price = _f(raw.get("price_dollars"))
    delta = _f(obj.get("delta_fp"))
    if not np.isfinite(delta):
        delta = _f(raw.get("delta_fp"))
    if side not in {"yes", "no"} or not np.isfinite(price) or not np.isfinite(delta):
        return None
    if not 0.0 <= price <= 1.0:
        return None
    return {
        "kind": "delta",
        "t": float(t),
        "ticker": ticker,
        "series": _series(ticker),
        "close": float(close),
        "epoch": obj.get("connection_epoch"),
        "seq": obj.get("seq"),
        "side": side,
        "price": float(price),
        "delta": float(delta),
        "exchange_t": _exchange_ts(obj),
    }


def _valid_ticker(obj, allowed_series, start_minute, end_minute):
    ticker = _base._get_ticker(obj)
    if not ticker or _series(ticker) not in allowed_series:
        return None
    t = _event_ts(obj)
    close = _close_ts(obj, ticker)
    if not _inside_window(t, close, start_minute, end_minute, 0.0):
        return None
    bid = _base._price(obj.get("yes_bid_dollars"))
    ask = _base._price(obj.get("yes_ask_dollars"))
    if not np.isfinite(bid) or not np.isfinite(ask):
        return None
    if not (0.0 <= bid <= ask <= 1.0):
        return None
    return {
        "kind": "ticker",
        "t": float(t),
        "ticker": ticker,
        "series": _series(ticker),
        "close": float(close),
        "epoch": obj.get("connection_epoch"),
        "bid": float(bid),
        "ask": float(ask),
        "exchange_t": _exchange_ts(obj),
    }


def _valid_gap(obj):
    if str(obj.get("type") or "") != "sequence_gap":
        return None
    t = _event_ts(obj)
    if not np.isfinite(t):
        return None
    return {
        "kind": "gap",
        "t": float(t),
        "epoch": obj.get("connection_epoch"),
        "last_seq": obj.get("last_seq"),
        "new_seq": obj.get("new_seq"),
    }


def _iter_jsonl(path, kind, parser, parser_args, progress_every):
    path = Path(path)
    scanned = kept = 0
    t0 = time.time()
    with path.open("rb") as fh:
        for raw in fh:
            scanned += 1
            if progress_every and scanned % progress_every == 0:
                print(
                    f"  {kind}: {scanned:,} lines | kept={kept:,} | "
                    f"{time.time()-t0:.1f}s"
                )
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            item = parser(obj, *parser_args)
            if item is None:
                continue
            kept += 1
            yield item
    print(
        f"  {kind}: DONE {scanned:,} lines | kept={kept:,} | "
        f"{time.time()-t0:.1f}s"
    )


def _merge_streams(streams):
    heap = []
    for priority, (name, it) in enumerate(streams):
        try:
            item = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (item["t"], priority, name, item, it))
    while heap:
        _, priority, name, item, it = heapq.heappop(heap)
        yield name, item
        try:
            nxt = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (nxt["t"], priority, name, nxt, it))


def _status(state):
    if state is None or not state.get("seeded"):
        return "UNSEEDED"
    bids, asks = state["bids"], state["asks"]
    if not bids:
        return "MISSING_BID"
    if not asks:
        return "MISSING_ASK"
    bid, ask = max(bids), min(asks)
    if bid > ask + EPS:
        return "CROSSED"
    if abs(bid - ask) <= EPS:
        return "LOCKED"
    if not (0.0 <= bid < ask <= 1.0):
        return "BAD_PRICE"
    return "VALID"


def _apply_delta(state, ev):
    book = state["bids"] if ev["side"] == "yes" else state["asks"]
    p = ev["price"]
    new = float(book.get(p, 0.0)) + ev["delta"]
    if new <= EPS:
        book.pop(p, None)
    else:
        book[p] = new


def _age_bucket(age_s):
    if not np.isfinite(age_s):
        return "NA"
    if age_s <= 0.100:
        return "<=100ms"
    if age_s <= 0.250:
        return "100-250ms"
    if age_s <= 0.500:
        return "250-500ms"
    if age_s <= 1.000:
        return "500ms-1s"
    if age_s <= 2.000:
        return "1-2s"
    return ">2s"


@dataclass
class Agg:
    n: int = 0
    bid_exact: int = 0
    ask_exact: int = 0
    both_exact: int = 0
    both_tol: int = 0
    both_1c: int = 0
    bid_errors_c: list = field(default_factory=list)
    ask_errors_c: list = field(default_factory=list)
    book_ages_s: list = field(default_factory=list)
    exchange_ages_ms: list = field(default_factory=list)

    def add(self, bid_err_c, ask_err_c, age_s, exchange_age_ms, tol_c):
        self.n += 1
        be = abs(float(bid_err_c))
        ae = abs(float(ask_err_c))
        self.bid_errors_c.append(be)
        self.ask_errors_c.append(ae)
        if np.isfinite(age_s):
            self.book_ages_s.append(float(age_s))
        if np.isfinite(exchange_age_ms):
            self.exchange_ages_ms.append(float(exchange_age_ms))
        bx = be <= 1e-6
        ax = ae <= 1e-6
        self.bid_exact += int(bx)
        self.ask_exact += int(ax)
        self.both_exact += int(bx and ax)
        self.both_tol += int(be <= tol_c + 1e-9 and ae <= tol_c + 1e-9)
        self.both_1c += int(be <= 1.0 + 1e-9 and ae <= 1.0 + 1e-9)

    def row(self, label):
        def pct(x):
            return 100.0 * x / self.n if self.n else np.nan
        return {
            "group": label,
            "comparisons": self.n,
            "bid_exact_pct": pct(self.bid_exact),
            "ask_exact_pct": pct(self.ask_exact),
            "both_exact_pct": pct(self.both_exact),
            "both_within_tol_pct": pct(self.both_tol),
            "both_within_1c_pct": pct(self.both_1c),
            "median_bid_error_c": float(np.median(self.bid_errors_c)) if self.bid_errors_c else np.nan,
            "p95_bid_error_c": float(np.percentile(self.bid_errors_c, 95)) if self.bid_errors_c else np.nan,
            "median_ask_error_c": float(np.median(self.ask_errors_c)) if self.ask_errors_c else np.nan,
            "p95_ask_error_c": float(np.percentile(self.ask_errors_c, 95)) if self.ask_errors_c else np.nan,
            "median_book_age_ms": 1000.0 * float(np.median(self.book_ages_s)) if self.book_ages_s else np.nan,
            "median_exchange_age_ms": float(np.median(self.exchange_ages_ms)) if self.exchange_ages_ms else np.nan,
        }


def run_m1_m5_event_time_bbo_validation(
    session_dir,
    output_dir=None,
    *,
    start_minute=1.0,
    end_minute=5.0,
    pre_seconds=30.0,
    tolerance_c=0.25,
    crypto_series=None,
    progress_delta_every=5_000_000,
    show=True,
):
    session = Path(session_dir)
    required = [
        "full_books.jsonl", "book_deltas.jsonl", "ticker_updates.jsonl",
        "connection_events.jsonl",
    ]
    for name in required:
        if not (session / name).exists():
            raise FileNotFoundError(session / name)
    if not (0 <= start_minute < end_minute <= 15):
        raise ValueError("Require 0 <= start_minute < end_minute <= 15")

    series = set(crypto_series or CRYPTO_SERIES)
    if output_dir is None:
        root = session.resolve().parents[2] if len(session.resolve().parents) >= 3 else Path.cwd()
        output_dir = root / "results" / "kalshi_mm_m1_m5_event_time_validation" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if show:
        print("=" * 108)
        print("M1-M5 EVENT-TIME BBO VALIDATION — NO TRADING SIMULATION")
        print("=" * 108)
        print("Streaming full-book anchors + deltas + ticker events in receipt-time order.")
        print("Books reset on sequence gaps / connection-epoch changes; valid full books re-seed state.")
        print()

    anchors = _iter_jsonl(
        session / "full_books.jsonl", "full_book", _valid_anchor,
        (series, start_minute, end_minute, pre_seconds), 250_000,
    )
    deltas = _iter_jsonl(
        session / "book_deltas.jsonl", "delta", _valid_delta,
        (series, start_minute, end_minute, pre_seconds), progress_delta_every,
    )
    tickers = _iter_jsonl(
        session / "ticker_updates.jsonl", "ticker", _valid_ticker,
        (series, start_minute, end_minute), 1_000_000,
    )
    gaps = _iter_jsonl(
        session / "connection_events.jsonl", "gap", _valid_gap, (), 100_000,
    )

    states = {}
    skip = Counter()
    counts = Counter()
    global_agg = Agg()
    asset_aggs = defaultdict(Agg)
    age_aggs = defaultdict(Agg)
    anchor_errors = Agg()

    comp_fields = [
        "time", "ticker", "series", "connection_epoch", "book_status",
        "recon_bid", "ticker_bid", "bid_error_c",
        "recon_ask", "ticker_ask", "ask_error_c",
        "both_exact", "both_within_tol", "both_within_1c",
        "book_age_ms", "book_age_bucket", "exchange_age_ms",
    ]

    t0 = time.time()
    with (out / "event_time_comparisons.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=comp_fields)
        writer.writeheader()

        streams = [
            ("gap", iter(gaps)),
            ("anchor", iter(anchors)),
            ("delta", iter(deltas)),
            ("ticker", iter(tickers)),
        ]
        for name, ev in _merge_streams(streams):
            counts[name] += 1

            if name == "gap":
                states.clear()
                counts["global_resets"] += 1
                continue

            ticker = ev["ticker"]

            if name == "anchor":
                old = states.get(ticker)
                if old is not None and _status(old) == "VALID" and old.get("epoch") == ev.get("epoch"):
                    rb, ra = max(old["bids"]), min(old["asks"])
                    ab, aa = ev["bids"][0][0], ev["asks"][0][0]
                    be, ae = 100.0 * (rb - ab), 100.0 * (ra - aa)
                    age = ev["t"] - old.get("last_book_t", ev["t"])
                    anchor_errors.add(be, ae, age, np.nan, tolerance_c)
                states[ticker] = {
                    "seeded": True,
                    "epoch": ev.get("epoch"),
                    "bids": {float(p): float(q) for p, q in ev["bids"] if q > 0},
                    "asks": {float(p): float(q) for p, q in ev["asks"] if q > 0},
                    "last_book_t": min(ev["t"], ev.get("source_t", ev["t"])),
                    "last_book_exchange_t": ev.get("exchange_t", np.nan),
                    "last_seq": ev.get("seq"),
                }
                continue

            if name == "delta":
                st = states.get(ticker)
                if st is None or not st.get("seeded"):
                    skip["delta_unseeded"] += 1
                    continue
                if st.get("epoch") is not None and ev.get("epoch") is not None and st.get("epoch") != ev.get("epoch"):
                    states.pop(ticker, None)
                    skip["delta_epoch_reset"] += 1
                    continue
                _apply_delta(st, ev)
                st["last_book_t"] = ev["t"]
                st["last_book_exchange_t"] = ev.get("exchange_t", np.nan)
                st["last_seq"] = ev.get("seq")
                continue

            # ticker event: compare immediately against current reconstructed state.
            st = states.get(ticker)
            if st is None or not st.get("seeded"):
                skip["ticker_unseeded"] += 1
                continue
            if st.get("epoch") is not None and ev.get("epoch") is not None and st.get("epoch") != ev.get("epoch"):
                skip["ticker_epoch_mismatch"] += 1
                continue
            stat = _status(st)
            if stat != "VALID":
                skip[f"ticker_book_{stat.lower()}"] += 1
                continue

            rb, ra = max(st["bids"]), min(st["asks"])
            be = 100.0 * (rb - ev["bid"])
            ae = 100.0 * (ra - ev["ask"])
            age_s = max(0.0, ev["t"] - st.get("last_book_t", ev["t"]))
            last_ex = st.get("last_book_exchange_t", np.nan)
            exch_age_ms = (
                1000.0 * (ev["exchange_t"] - last_ex)
                if np.isfinite(ev.get("exchange_t", np.nan)) and np.isfinite(last_ex)
                else np.nan
            )
            bucket = _age_bucket(age_s)

            global_agg.add(be, ae, age_s, exch_age_ms, tolerance_c)
            asset_aggs[ev["series"]].add(be, ae, age_s, exch_age_ms, tolerance_c)
            age_aggs[bucket].add(be, ae, age_s, exch_age_ms, tolerance_c)

            bx = abs(be) <= 1e-6
            ax = abs(ae) <= 1e-6
            writer.writerow({
                "time": datetime.fromtimestamp(ev["t"], tz=timezone.utc).isoformat(),
                "ticker": ticker,
                "series": ev["series"],
                "connection_epoch": ev.get("epoch"),
                "book_status": stat,
                "recon_bid": rb,
                "ticker_bid": ev["bid"],
                "bid_error_c": be,
                "recon_ask": ra,
                "ticker_ask": ev["ask"],
                "ask_error_c": ae,
                "both_exact": bool(bx and ax),
                "both_within_tol": bool(abs(be) <= tolerance_c + 1e-9 and abs(ae) <= tolerance_c + 1e-9),
                "both_within_1c": bool(abs(be) <= 1.0 + 1e-9 and abs(ae) <= 1.0 + 1e-9),
                "book_age_ms": 1000.0 * age_s,
                "book_age_bucket": bucket,
                "exchange_age_ms": exch_age_ms,
            })

    overall_df = pd.DataFrame([global_agg.row("ALL")])
    asset_df = pd.DataFrame([asset_aggs[k].row(k) for k in sorted(asset_aggs)])
    age_order = ["<=100ms", "100-250ms", "250-500ms", "500ms-1s", "1-2s", ">2s", "NA"]
    age_df = pd.DataFrame([age_aggs[k].row(k) for k in age_order if k in age_aggs])
    anchor_df = pd.DataFrame([anchor_errors.row("PRE_ANCHOR_RECON_VS_ANCHOR")])
    skip_df = pd.DataFrame(
        [{"reason": k, "count": v} for k, v in sorted(skip.items(), key=lambda x: -x[1])]
    )
    stream_df = pd.DataFrame(
        [{"event": k, "count": v} for k, v in sorted(counts.items())]
    )

    overall_df.to_csv(out / "overall_bbo_validation.csv", index=False)
    asset_df.to_csv(out / "asset_bbo_validation.csv", index=False)
    age_df.to_csv(out / "book_age_bbo_validation.csv", index=False)
    anchor_df.to_csv(out / "anchor_consistency.csv", index=False)
    skip_df.to_csv(out / "skip_reasons.csv", index=False)
    stream_df.to_csv(out / "stream_counts.csv", index=False)

    config = {
        "study_version": STUDY_VERSION,
        "session": str(session.resolve()),
        "start_minute": start_minute,
        "end_minute": end_minute,
        "pre_seconds": pre_seconds,
        "tolerance_c": tolerance_c,
        "crypto_series": sorted(series),
        "price_convention": "use_yes_price=True; recorder no-side levels are YES-price scale",
        "comparison_clock": "receipt-time event order",
        "purpose": "data validation only; no MM PnL or threshold inference",
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if show:
        r = overall_df.iloc[0]
        print("\n" + "=" * 108)
        print("EVENT-TIME BBO VALIDATION RESULT")
        print("=" * 108)
        print(f"Ticker BBO comparisons: {int(r['comparisons']):,}")
        print(f"Bid exact:              {r['bid_exact_pct']:.2f}%")
        print(f"Ask exact:              {r['ask_exact_pct']:.2f}%")
        print(f"Both exact:             {r['both_exact_pct']:.2f}%")
        print(f"Both within {tolerance_c:.2f}c:      {r['both_within_tol_pct']:.2f}%")
        print(f"Both within 1.00c:      {r['both_within_1c_pct']:.2f}%")
        print(f"Median bid error:       {r['median_bid_error_c']:.3f}c | p95={r['p95_bid_error_c']:.3f}c")
        print(f"Median ask error:       {r['median_ask_error_c']:.3f}c | p95={r['p95_ask_error_c']:.3f}c")
        print(f"Median book age:        {r['median_book_age_ms']:.1f} ms")
        print(f"Elapsed:                {time.time()-t0:.1f}s")
        print("\nBY ASSET")
        if len(asset_df):
            print(asset_df.to_string(index=False))
        print("\nBY LAST-BOOK AGE")
        if len(age_df):
            print(age_df.to_string(index=False))
        print("\nSKIPPED TICKER/DELTA EVENTS")
        if len(skip_df):
            print(skip_df.to_string(index=False))
        print("\nANCHOR CONSISTENCY (reconstruction immediately before a new valid full-book anchor)")
        print(anchor_df.to_string(index=False))
        print("\nInterpretation: if fresh-book buckets (especially <=100ms / <=250ms) show very high BBO agreement,")
        print("the earlier ~50% one-second validation was mainly a stale-time comparison artifact. If fresh buckets")
        print("remain poor, the historical reconstructed state is not reliable enough for MM replay.")
        print("Outputs:", out)
        print("=" * 108)

    return {
        "output_dir": out,
        "overall": overall_df,
        "by_asset": asset_df,
        "by_book_age": age_df,
        "anchor_consistency": anchor_df,
        "skip_reasons": skip_df,
        "stream_counts": stream_df,
    }


__all__ = ["STUDY_VERSION", "run_m1_m5_event_time_bbo_validation"]
