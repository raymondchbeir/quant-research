from __future__ import annotations

"""M1-M5 market-making theory diagnostic suite.

This script does NOT optimize a new strategy. It takes the frozen Defensive V1
fills/episodes as the common reference sample and tests the mechanisms proposed
after V1:

1) event-time stale-quote cancellation at 0/50/100/250/500 ms;
2) queue-ahead cancellation-credit evidence;
3) BID/ASK asymmetry;
4) L1 microprice and L2 depth pressure;
5) sub-second aggressive-flow acceleration;
6) spread compensation versus toxicity;
7) BTC common-shock exposure;
8) quote-age / maximum-quote-lifetime effects;
9) queue-burn velocity before fills;
10) forced inventory flattening (V2 versus V1, if supplied).

Important:
- stale/lifetime "counterfactuals" are STATIC fill-removal diagnostics. They ask
  what the realized V1 fill set would have looked like if fills already stale
  by the specified latency/lifetime were absent. They are NOT path-exact
  strategy replays because removing a fill changes later inventory/cooldown.
- queue-cancellation credit is an evidence diagnostic, not a new FIFO fill
  model. It estimates displayed size disappearance not explained by recorded
  exact-price aggressive flow.
- all feature buckets are descriptive. Do not choose thresholds from this same
  sample and call them validated.

The purpose is to decide WHICH mechanism deserves a separately frozen, exact
event-time replay/OOS test.
"""

import argparse
import bisect
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_reconstructed_m1_m5_backtest as B
from . import mm_m1_m5_event_time_bbo_validation as EV

STUDY_VERSION = "M1_M5_MM_THEORY_SUITE_V1"
EPS = 1e-9
LATENCIES_MS = (0, 50, 100, 250, 500)
QUOTE_LIFETIMES_S = (1, 2, 3, 5, 10)
FLOW_WINDOWS_S = (0.25, 0.5, 1.0, 5.0)


def _f(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _sign(side):
    return 1.0 if str(side).upper() == "BID" else -1.0


def _weighted_mean(df, col, w="qty"):
    if df.empty or col not in df:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce")
    ww = pd.to_numeric(df[w], errors="coerce") if w in df else pd.Series(1.0, index=df.index)
    m = x.notna() & ww.notna() & (ww > 0)
    if not m.any():
        return np.nan
    return float(np.average(x[m], weights=ww[m]))


def _load_reference(defensive_v1_dir):
    root = Path(defensive_v1_dir)
    paths = {
        "fills": root / "fills.csv",
        "episodes": root / "quote_episodes.csv",
        "contracts": root / "contract_summary.csv",
        "headline": root / "headline_summary.csv",
        "windows": root / "window_summary.csv",
    }
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)

    fills = pd.read_csv(paths["fills"])
    episodes = pd.read_csv(paths["episodes"])
    contracts = pd.read_csv(paths["contracts"])
    headline = pd.read_csv(paths["headline"])
    windows = pd.read_csv(paths["windows"])

    if fills.empty:
        raise RuntimeError("Defensive V1 has no fills")
    if contracts.empty or headline.empty:
        raise RuntimeError("Defensive V1 summary is empty")

    for c in ("fill_ts", "qty", "price", "fill_latency_s", "spread_c_at_join"):
        if c in fills:
            fills[c] = pd.to_numeric(fills[c], errors="coerce")
    for c in ("join_ts", "end_ts", "price", "queue_ahead_initial", "queue_ahead_final"):
        if c in episodes:
            episodes[c] = pd.to_numeric(episodes[c], errors="coerce")

    epcols = [
        c for c in (
            "episode_id", "join_ts", "end_ts", "end_reason", "queue_ahead_initial",
            "queue_ahead_final", "mid_at_join", "momentum_3s_c_at_join",
            "flow_imbalance_5s_at_join", "inventory_at_join"
        ) if c in episodes.columns
    ]
    ep = episodes[epcols].drop_duplicates("episode_id")
    fills = fills.merge(ep, on="episode_id", how="left", suffixes=("", "_ep"))

    cmeta = contracts[[
        c for c in ("ticker", "series", "close_ts", "close_time", "final_mid_m5",
                    "net_mtm_pnl_before_fees") if c in contracts.columns
    ]].drop_duplicates("ticker")
    merge_keys = [c for c in ("ticker", "series") if c in cmeta.columns and c in fills.columns]
    fills = fills.merge(cmeta, on=merge_keys, how="left")

    fills["side_sign"] = fills["side"].map(_sign)
    fills["pnl_to_m5_c"] = fills["side_sign"] * (fills["final_mid_m5"] - fills["price"]) * 100.0
    fills["pnl_to_m5_dollars"] = fills["pnl_to_m5_c"] / 100.0 * fills["qty"]

    return fills, episodes, contracts, headline, windows


def _load_recon_features(reconstruction_dir, eligible):
    path = Path(reconstruction_dir) / "reconstructed_book_samples.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    use = [
        "ticker", "series", "ts", "status", "yes_bid1", "yes_bid1_qty",
        "yes_bid2", "yes_bid2_qty", "yes_ask1", "yes_ask1_qty",
        "yes_ask2", "yes_ask2_qty", "mid", "spread_c",
    ]
    print("Loading reconstructed L1/L2 feature grid...")
    df = pd.read_csv(path, usecols=lambda c: c in set(use))
    df = df[df["ticker"].astype(str).isin(eligible)].copy()
    for c in [x for x in use if x not in ("ticker", "series", "status") and x in df]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["status"].astype(str).str.upper().eq("VALID")].copy()
    df.sort_values(["ticker", "ts"], inplace=True)

    bq = df["yes_bid1_qty"].clip(lower=0)
    aq = df["yes_ask1_qty"].clip(lower=0)
    den = bq + aq
    df["l1_microprice"] = np.where(
        den > EPS,
        (df["yes_ask1"] * bq + df["yes_bid1"] * aq) / den,
        df["mid"],
    )
    b2 = pd.to_numeric(df.get("yes_bid2_qty"), errors="coerce").fillna(0).clip(lower=0)
    a2 = pd.to_numeric(df.get("yes_ask2_qty"), errors="coerce").fillna(0).clip(lower=0)
    btot = bq.fillna(0) + b2
    atot = aq.fillna(0) + a2
    depth = btot + atot
    df["l2_imbalance"] = np.where(depth > EPS, (btot - atot) / depth, 0.0)
    return df


def _attach_join_book_features(fills, recon):
    cols = [
        "ticker", "ts", "yes_bid1", "yes_bid1_qty", "yes_bid2", "yes_bid2_qty",
        "yes_ask1", "yes_ask1_qty", "yes_ask2", "yes_ask2_qty", "mid",
        "spread_c", "l1_microprice", "l2_imbalance",
    ]
    right = recon[[c for c in cols if c in recon]].copy().rename(columns={"ts": "join_book_ts"})
    left = fills.sort_values(["ticker", "join_ts"]).copy()
    out_parts = []
    for ticker, g in left.groupby("ticker", sort=False):
        rr = right[right["ticker"] == ticker].sort_values("join_book_ts")
        if rr.empty:
            out_parts.append(g)
            continue
        x = pd.merge_asof(
            g.sort_values("join_ts"),
            rr.drop(columns=["ticker"]),
            left_on="join_ts",
            right_on="join_book_ts",
            direction="backward",
            tolerance=1.1,
        )
        out_parts.append(x)
    out = pd.concat(out_parts, ignore_index=True) if out_parts else left
    out["micropressure_c"] = out["side_sign"] * (out["l1_microprice"] - out["mid"]) * 100.0
    out["l2_pressure"] = out["side_sign"] * out["l2_imbalance"]
    return out


def _scan_ticker_bbo(session_dir, meta, eligible):
    path = Path(session_dir) / "ticker_updates.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    allowed_series = set(meta["series"].dropna().astype(str))
    out = defaultdict(list)
    scanned = kept = 0
    t0 = time.time()
    with path.open("rb") as fh:
        for raw in fh:
            scanned += 1
            if scanned % 500_000 == 0:
                print(f"  ticker BBO: {scanned:,} lines | kept={kept:,} | {time.time()-t0:.1f}s")
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            item = EV._valid_ticker(obj, allowed_series, 1.0, 5.0)
            if item is None or item["ticker"] not in eligible:
                continue
            out[item["ticker"]].append((float(item["t"]), float(item["bid"]), float(item["ask"])))
            kept += 1
    for k in out:
        out[k].sort()
    print(f"  ticker BBO: DONE {scanned:,} lines | kept={kept:,} | {time.time()-t0:.1f}s")
    return out


def _attach_stale_bbo(fills, ticker_bbo):
    stale_ts = np.full(len(fills), np.nan)
    time_cache = {k: [x[0] for x in v] for k, v in ticker_bbo.items()}
    for i, r in enumerate(fills.itertuples(index=False)):
        arr = ticker_bbo.get(str(r.ticker), [])
        if not arr:
            continue
        times = time_cache[str(r.ticker)]
        j = bisect.bisect_left(times, _f(r.join_ts))
        fill_t = _f(r.fill_ts)
        px = _f(r.price)
        side = str(r.side)
        while j < len(arr) and arr[j][0] < fill_t - EPS:
            t, bid, ask = arr[j]
            nowpx = bid if side == "BID" else ask
            if np.isfinite(nowpx) and abs(nowpx - px) > EPS:
                stale_ts[i] = t
                break
            j += 1
    fills = fills.copy()
    fills["first_bbo_change_ts"] = stale_ts
    fills["stale_before_fill"] = np.isfinite(stale_ts)
    fills["stale_age_to_fill_ms"] = np.where(
        fills["stale_before_fill"],
        (fills["fill_ts"] - fills["first_bbo_change_ts"]) * 1000.0,
        np.nan,
    )
    return fills


def _trade_arrays(trades):
    out = {}
    for ticker, arr in trades.items():
        if not arr:
            continue
        arr = sorted(arr, key=lambda z: z.t)
        t = np.array([z.t for z in arr], dtype=float)
        signed = np.array([
            float(z.qty) if z.taker_book_side == "bid"
            else -float(z.qty) if z.taker_book_side == "ask"
            else 0.0
            for z in arr
        ])
        qty = np.abs(signed)
        out[ticker] = {
            "times": t,
            "signed": signed,
            "qty": qty,
            "cs_signed": np.r_[0.0, np.cumsum(signed)],
            "cs_qty": np.r_[0.0, np.cumsum(qty)],
            "prices": np.array([z.yes_price for z in arr], dtype=float),
            "book_side": np.array([z.taker_book_side for z in arr], dtype=object),
        }
    return out


def _window_sum(a, t0, t1, cumulative_key):
    times = a["times"]
    i = bisect.bisect_left(times, t0)
    j = bisect.bisect_left(times, t1)
    cs = a[cumulative_key]
    return float(cs[j] - cs[i])


def _flow_features_for_fill(r, arr):
    if arr is None:
        return {}
    t = _f(r.fill_ts)
    sign = _f(r.side_sign)
    d = {}
    for h in FLOW_WINDOWS_S:
        s = _window_sum(arr, t - h, t, "cs_signed")
        q = _window_sum(arr, t - h, t, "cs_qty")
        imb = s / q if q > EPS else 0.0
        key = str(h).replace(".", "p")
        d[f"flow_imb_{key}s"] = imb
        d[f"side_flow_{key}s"] = sign * imb
        d[f"flow_qty_{key}s"] = q
    d["flow_accel_0p5_vs_5"] = d["side_flow_0p5s"] - d["side_flow_5p0s"]

    px = _f(r.price)
    side = str(r.side)
    relevant_book_side = "ask" if side == "BID" else "bid"
    for h in (0.25, 0.5, 1.0):
        i = bisect.bisect_left(arr["times"], t - h)
        j = bisect.bisect_left(arr["times"], t)
        m = (
            (arr["book_side"][i:j] == relevant_book_side)
            & (np.abs(arr["prices"][i:j] - px) <= EPS)
        )
        vol = float(arr["qty"][i:j][m].sum()) if j > i else 0.0
        q0 = max(1.0, _f(getattr(r, "queue_ahead_initial", np.nan), 1.0))
        key = str(h).replace(".", "p")
        d[f"queue_burn_exact_{key}s_qty"] = vol
        d[f"queue_burn_exact_{key}s_ratio"] = vol / q0
    return d


def _attach_flow_and_queue_burn(fills, trades):
    arrays = _trade_arrays(trades)
    rows = []
    for r in fills.itertuples(index=False):
        rows.append(_flow_features_for_fill(r, arrays.get(str(r.ticker))))
    feat = pd.DataFrame(rows)
    return pd.concat([fills.reset_index(drop=True), feat], axis=1), arrays


def _estimate_queue_cancellation_credit(fills, recon, trade_arrays):
    credit = np.full(len(fills), np.nan)
    credit_ratio = np.full(len(fills), np.nan)
    by_ticker = {k: g.sort_values("ts") for k, g in recon.groupby("ticker")}
    for i, r in enumerate(fills.itertuples(index=False)):
        g = by_ticker.get(str(r.ticker))
        arr = trade_arrays.get(str(r.ticker))
        if g is None or arr is None:
            continue
        t0, t1 = _f(r.join_ts), _f(r.fill_ts)
        px = _f(r.price)
        side = str(r.side)
        qtycol = "yes_bid1_qty" if side == "BID" else "yes_ask1_qty"
        pxcol = "yes_bid1" if side == "BID" else "yes_ask1"
        z = g[(g["ts"] >= t0 - EPS) & (g["ts"] < t1 - EPS)]
        z = z[np.abs(pd.to_numeric(z[pxcol], errors="coerce") - px) <= EPS]
        if z.empty:
            continue
        q0 = max(0.0, _f(getattr(r, "queue_ahead_initial", np.nan), 0.0))
        minq = float(pd.to_numeric(z[qtycol], errors="coerce").min())
        display_drop = max(0.0, q0 - minq)

        tt = arr["times"]
        a = bisect.bisect_left(tt, t0)
        b = bisect.bisect_left(tt, t1)
        relevant = "ask" if side == "BID" else "bid"
        m = (
            (arr["book_side"][a:b] == relevant)
            & (np.abs(arr["prices"][a:b] - px) <= EPS)
        )
        aggressive = float(arr["qty"][a:b][m].sum()) if b > a else 0.0
        c = max(0.0, display_drop - aggressive)
        credit[i] = c
        credit_ratio[i] = c / max(q0, 1.0)

    x = fills.copy()
    x["queue_cancel_credit_lb_qty"] = credit
    x["queue_cancel_credit_lb_ratio"] = credit_ratio
    return x


def _btc_series(recon):
    btc = recon[recon["series"].astype(str).eq("KXBTC15M")].copy()
    btc = btc[np.isfinite(pd.to_numeric(btc["mid"], errors="coerce"))]
    btc = btc.sort_values("ts")
    if btc.empty:
        return np.array([]), np.array([])
    btc = btc.drop_duplicates("ts", keep="last")
    return btc["ts"].to_numpy(float), btc["mid"].to_numpy(float)


def _attach_btc_shock(fills, recon):
    times, mids = _btc_series(recon)
    x = fills.copy()
    for h in (1.0, 3.0):
        vals = np.full(len(x), np.nan)
        if len(times):
            for i, r in enumerate(x.itertuples(index=False)):
                t = _f(r.fill_ts)
                j = bisect.bisect_right(times, t) - 1
                k = bisect.bisect_right(times, t - h) - 1
                if j >= 0 and k >= 0 and t - times[j] <= 2.0 and (t - h) - times[k] <= 2.0:
                    vals[i] = (mids[j] - mids[k]) * 100.0
        x[f"btc_move_{int(h)}s_c"] = vals
        x[f"abs_btc_move_{int(h)}s_c"] = np.abs(vals)
        x[f"side_btc_move_{int(h)}s_c"] = x["side_sign"] * vals
    return x


def _bucket_metrics(df, bucket_col, order=None):
    if df.empty:
        return pd.DataFrame()
    rows = []
    groups = df.groupby(bucket_col, observed=False, dropna=False)
    for key, g in groups:
        qty = pd.to_numeric(g["qty"], errors="coerce").sum()
        pnl = pd.to_numeric(g["pnl_to_m5_dollars"], errors="coerce").sum()
        row = {
            bucket_col: str(key),
            "fills": len(g),
            "fill_qty": qty,
            "net_to_m5_dollars": pnl,
            "pnl_c_per_qty_to_m5": 100.0 * pnl / max(qty, EPS),
            "avg_gross_edge_c": _weighted_mean(g, "gross_edge_at_fill_c"),
        }
        for h in (5, 15, 30, 60):
            c = f"markout_{h}s_c"
            if c in g:
                row[f"avg_markout_{h}s_c"] = _weighted_mean(g, c)
        rows.append(row)
    out = pd.DataFrame(rows)
    if order is not None and not out.empty:
        rank = {str(v): i for i, v in enumerate(order)}
        out["_rank"] = out[bucket_col].map(lambda z: rank.get(str(z), 9999))
        out.sort_values("_rank", inplace=True)
        out.drop(columns="_rank", inplace=True)
    return out


def _static_counterfactual(df, keep_mask, scenario, family):
    keep_mask = pd.Series(keep_mask, index=df.index).fillna(False).astype(bool)
    g = df[keep_mask].copy()
    removed = df[~keep_mask].copy()
    qty = pd.to_numeric(g["qty"], errors="coerce").sum()
    base_qty = pd.to_numeric(df["qty"], errors="coerce").sum()
    net = pd.to_numeric(g["pnl_to_m5_dollars"], errors="coerce").sum()
    base_net = pd.to_numeric(df["pnl_to_m5_dollars"], errors="coerce").sum()
    row = {
        "family": family,
        "scenario": scenario,
        "fills_kept": len(g),
        "fills_removed": len(removed),
        "fill_qty_kept": qty,
        "fill_qty_removed": base_qty - qty,
        "static_net_to_m5_dollars": net,
        "static_change_vs_v1_dollars": net - base_net,
        "removed_realized_pnl_dollars": pd.to_numeric(removed["pnl_to_m5_dollars"], errors="coerce").sum(),
        "avg_gross_edge_c": _weighted_mean(g, "gross_edge_at_fill_c"),
    }
    for h in (5, 15, 30, 60):
        c = f"markout_{h}s_c"
        if c in g:
            row[f"avg_markout_{h}s_c"] = _weighted_mean(g, c)
    return row


def _scenario_tables(fills):
    rows = []
    for ms in LATENCIES_MS:
        avoid = fills["stale_before_fill"].fillna(False) & (fills["stale_age_to_fill_ms"] >= ms - EPS)
        rows.append(_static_counterfactual(fills, ~avoid, f"CANCEL_{ms}MS", "STALE_BBO_CANCEL"))
    for life in QUOTE_LIFETIMES_S:
        keep = pd.to_numeric(fills["fill_latency_s"], errors="coerce") <= life + EPS
        rows.append(_static_counterfactual(fills, keep, f"MAX_AGE_{life}S", "QUOTE_LIFETIME"))
    return pd.DataFrame(rows)


def _make_bucket_tables(fills):
    tables = {}
    x = fills.copy()

    x["quote_age_bucket"] = pd.cut(
        x["fill_latency_s"], [-np.inf, 1, 2, 3, 5, 10, np.inf],
        labels=["<=1s", "1-2s", "2-3s", "3-5s", "5-10s", ">10s"],
    )
    tables["quote_age"] = _bucket_metrics(x, "quote_age_bucket", ["<=1s", "1-2s", "2-3s", "3-5s", "5-10s", ">10s"])

    x["spread_bucket"] = pd.cut(
        x["spread_c_at_join"], [-np.inf, 2.000001, 3.000001, 4.000001, 5.000001, np.inf],
        labels=["~2c", "2-3c", "3-4c", "4-5c", ">5c"], include_lowest=True,
    )
    tables["spread"] = _bucket_metrics(x, "spread_bucket", ["~2c", "2-3c", "3-4c", "4-5c", ">5c"])

    x["micropressure_bucket"] = pd.cut(
        x["micropressure_c"], [-np.inf, -0.25, 0.25, np.inf],
        labels=["ADVERSE_<-0.25c", "NEUTRAL", "FAVORABLE_>0.25c"],
    )
    tables["microprice"] = _bucket_metrics(x, "micropressure_bucket", ["ADVERSE_<-0.25c", "NEUTRAL", "FAVORABLE_>0.25c"])

    x["l2_pressure_bucket"] = pd.cut(
        x["l2_pressure"], [-np.inf, -0.25, 0.25, np.inf],
        labels=["ADVERSE_<-0.25", "NEUTRAL", "FAVORABLE_>0.25"],
    )
    tables["l2_pressure"] = _bucket_metrics(x, "l2_pressure_bucket", ["ADVERSE_<-0.25", "NEUTRAL", "FAVORABLE_>0.25"])

    x["flow_accel_bucket"] = pd.cut(
        x["flow_accel_0p5_vs_5"], [-np.inf, -0.50, 0.50, np.inf],
        labels=["ADVERSE_ACCEL", "NEUTRAL", "FAVORABLE_ACCEL"],
    )
    tables["flow_acceleration"] = _bucket_metrics(x, "flow_accel_bucket", ["ADVERSE_ACCEL", "NEUTRAL", "FAVORABLE_ACCEL"])

    x["queue_burn_bucket"] = pd.cut(
        x["queue_burn_exact_0p5s_ratio"], [-np.inf, 0.10, 0.50, 1.0, np.inf],
        labels=["<=10%", "10-50%", "50-100%", ">100%"],
    )
    tables["queue_burn"] = _bucket_metrics(x, "queue_burn_bucket", ["<=10%", "10-50%", "50-100%", ">100%"])

    x["queue_cancel_credit_bucket"] = pd.cut(
        x["queue_cancel_credit_lb_ratio"], [-np.inf, 0.01, 0.25, 0.50, np.inf],
        labels=["~0%", "1-25%", "25-50%", ">50%"],
    )
    tables["queue_credit"] = _bucket_metrics(x, "queue_cancel_credit_bucket", ["~0%", "1-25%", "25-50%", ">50%"])

    alt = x[~x["series"].astype(str).eq("KXBTC15M")].copy()
    alt["btc_shock_bucket"] = pd.cut(
        alt["abs_btc_move_3s_c"], [-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=["<0.5c", "0.5-1c", "1-2c", ">=2c"],
    )
    tables["btc_shock"] = _bucket_metrics(alt, "btc_shock_bucket", ["<0.5c", "0.5-1c", "1-2c", ">=2c"])
    tables["side"] = _bucket_metrics(x, "side", ["BID", "ASK"])
    return tables


def _forced_flatten_comparison(defensive_v1_dir, inventory_v2_dir):
    if inventory_v2_dir is None:
        return pd.DataFrame()
    p1 = Path(defensive_v1_dir) / "headline_summary.csv"
    p2 = Path(inventory_v2_dir) / "headline_summary.csv"
    if not p2.exists():
        return pd.DataFrame()
    a = pd.read_csv(p1)
    b = pd.read_csv(p2)
    if a.empty or b.empty:
        return pd.DataFrame()
    fields = [
        "fill_qty", "avg_markout_5s_c", "avg_markout_15s_c", "avg_markout_30s_c",
        "avg_markout_60s_c", "net_mtm_pnl_before_fees", "pnl_per_window",
        "avg_max_abs_inventory", "p95_max_abs_inventory", "max_drawdown",
    ]
    rows = []
    for f in fields:
        av = _f(a.iloc[0].get(f))
        bv = _f(b.iloc[0].get(f))
        rows.append({"metric": f, "defensive_v1": av, "forced_flatten_v2": bv, "v2_minus_v1": bv-av})
    return pd.DataFrame(rows)


def _scoreboard(fills, scenario_df, tables, forced_flat):
    rows = []
    stale = scenario_df[scenario_df["family"] == "STALE_BBO_CANCEL"]
    s100 = stale[stale["scenario"] == "CANCEL_100MS"]
    if len(s100):
        r = s100.iloc[0]
        rows.append({
            "theory": "event_time_stale_cancel",
            "primary_stat": "100ms static change vs V1 ($)",
            "value": r["static_change_vs_v1_dollars"],
            "direction_if_true": "positive",
            "notes": "static fill-removal diagnostic; exact replay still required",
        })

    side = tables["side"].set_index("side") if not tables["side"].empty else pd.DataFrame()
    if not side.empty and "BID" in side.index and "ASK" in side.index:
        rows.append({
            "theory": "bid_ask_asymmetry",
            "primary_stat": "BID minus ASK 60s markout (c)",
            "value": _f(side.loc["BID", "avg_markout_60s_c"]) - _f(side.loc["ASK", "avg_markout_60s_c"]),
            "direction_if_true": "large positive magnitude",
            "notes": "prospective asymmetric rules require fresh/OOS data",
        })

    qb = tables["queue_burn"]
    low = qb[qb["queue_burn_bucket"] == "<=10%"] if len(qb) else pd.DataFrame()
    high = qb[qb["queue_burn_bucket"] == ">100%"] if len(qb) else pd.DataFrame()
    if len(low) and len(high):
        rows.append({
            "theory": "queue_burn_toxicity",
            "primary_stat": "high-burn minus low-burn 5s markout (c)",
            "value": _f(high.iloc[0].get("avg_markout_5s_c")) - _f(low.iloc[0].get("avg_markout_5s_c")),
            "direction_if_true": "negative",
            "notes": "tests whether fast queue consumption predicts toxicity",
        })

    fa = tables["flow_acceleration"]
    adv = fa[fa["flow_accel_bucket"] == "ADVERSE_ACCEL"] if len(fa) else pd.DataFrame()
    fav = fa[fa["flow_accel_bucket"] == "FAVORABLE_ACCEL"] if len(fa) else pd.DataFrame()
    if len(adv) and len(fav):
        rows.append({
            "theory": "subsecond_flow_acceleration",
            "primary_stat": "adverse minus favorable 5s markout (c)",
            "value": _f(adv.iloc[0].get("avg_markout_5s_c")) - _f(fav.iloc[0].get("avg_markout_5s_c")),
            "direction_if_true": "negative",
            "notes": "uses only flow strictly before fill",
        })

    mp = tables["microprice"]
    adv = mp[mp["micropressure_bucket"] == "ADVERSE_<-0.25c"] if len(mp) else pd.DataFrame()
    fav = mp[mp["micropressure_bucket"] == "FAVORABLE_>0.25c"] if len(mp) else pd.DataFrame()
    if len(adv) and len(fav):
        rows.append({
            "theory": "microprice_pressure",
            "primary_stat": "favorable minus adverse 15s markout (c)",
            "value": _f(fav.iloc[0].get("avg_markout_15s_c")) - _f(adv.iloc[0].get("avg_markout_15s_c")),
            "direction_if_true": "positive",
            "notes": "L1 microprice at quote join",
        })

    bs = tables["btc_shock"]
    calm = bs[bs["btc_shock_bucket"] == "<0.5c"] if len(bs) else pd.DataFrame()
    shock = bs[bs["btc_shock_bucket"] == ">=2c"] if len(bs) else pd.DataFrame()
    if len(calm) and len(shock):
        rows.append({
            "theory": "btc_common_shock",
            "primary_stat": "shock minus calm 15s markout (c)",
            "value": _f(shock.iloc[0].get("avg_markout_15s_c")) - _f(calm.iloc[0].get("avg_markout_15s_c")),
            "direction_if_true": "negative",
            "notes": "altcoin fills only",
        })

    qc = pd.to_numeric(fills["queue_cancel_credit_lb_ratio"], errors="coerce")
    rows.append({
        "theory": "cancellation_ahead_credit",
        "primary_stat": "median lower-bound cancellation credit / initial queue",
        "value": qc.median(),
        "direction_if_true": "materially > 0",
        "notes": "evidence only; not a path-exact queue replay",
    })

    age = tables["quote_age"]
    young = age[age["quote_age_bucket"] == "<=1s"] if len(age) else pd.DataFrame()
    old = age[age["quote_age_bucket"] == ">10s"] if len(age) else pd.DataFrame()
    if len(young) and len(old):
        rows.append({
            "theory": "maximum_quote_lifetime",
            "primary_stat": "old minus young 15s markout (c)",
            "value": _f(old.iloc[0].get("avg_markout_15s_c")) - _f(young.iloc[0].get("avg_markout_15s_c")),
            "direction_if_true": "negative",
            "notes": "quote age measured at fill",
        })

    if not forced_flat.empty:
        z = forced_flat[forced_flat["metric"] == "net_mtm_pnl_before_fees"]
        if len(z):
            rows.append({
                "theory": "forced_inventory_flatten",
                "primary_stat": "V2 minus V1 net PnL ($)",
                "value": _f(z.iloc[0]["v2_minus_v1"]),
                "direction_if_true": "positive",
                "notes": "already tested as exact V2 structural strategy",
            })
    return pd.DataFrame(rows)


def _print_table(title, df):
    print("\n" + title)
    if df is None or df.empty:
        print("  <no data>")
        return
    print(df.round(4).to_string(index=False))


def run_mm_theory_suite(
    session_dir,
    reconstruction_dir,
    defensive_v1_dir,
    inventory_v2_dir=None,
    output_dir=None,
    *,
    show=True,
):
    session = Path(session_dir)
    recon_dir = Path(reconstruction_dir)
    v1_dir = Path(defensive_v1_dir)
    for p in (session, recon_dir, v1_dir):
        if not p.exists():
            raise FileNotFoundError(p)

    fills, episodes, contracts, headline, windows = _load_reference(v1_dir)
    eligible = set(contracts["ticker"].astype(str))
    meta = contracts[[c for c in ("ticker", "series", "close_ts", "close_time") if c in contracts]].copy()

    md = {}
    for r in meta.itertuples(index=False):
        close_ts = _f(getattr(r, "close_ts", np.nan))
        if not np.isfinite(close_ts):
            close_ts = B._ts(getattr(r, "close_time", None))
        md[str(r.ticker)] = {"ticker": str(r.ticker), "series": str(r.series), "close_ts": close_ts}

    recon = _load_recon_features(recon_dir, eligible)
    fills = _attach_join_book_features(fills, recon)

    print("Streaming ticker BBO for stale-quote diagnostics...")
    ticker_bbo = _scan_ticker_bbo(session, meta, eligible)
    fills = _attach_stale_bbo(fills, ticker_bbo)

    print("Streaming trades for flow / queue-burn diagnostics...")
    trades, trade_stats = B._scan_trades(session, md)
    fills, trade_arrays = _attach_flow_and_queue_burn(fills, trades)

    print("Estimating displayed cancellation-ahead credit...")
    fills = _estimate_queue_cancellation_credit(fills, recon, trade_arrays)

    print("Attaching BTC common-shock features...")
    fills = _attach_btc_shock(fills, recon)

    scenario_df = _scenario_tables(fills)
    tables = _make_bucket_tables(fills)
    forced_flat = _forced_flatten_comparison(v1_dir, inventory_v2_dir)
    scoreboard = _scoreboard(fills, scenario_df, tables, forced_flat)

    if output_dir is None:
        root = session.resolve().parents[2] if len(session.resolve().parents) >= 3 else Path.cwd()
        output_dir = root / "results" / "kalshi_mm_m1_m5_theory_suite" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fills.to_csv(out / "fill_theory_features.csv", index=False)
    scenario_df.to_csv(out / "static_counterfactual_scenarios.csv", index=False)
    scoreboard.to_csv(out / "theory_scoreboard.csv", index=False)
    forced_flat.to_csv(out / "forced_flatten_v2_vs_v1.csv", index=False)
    for name, df in tables.items():
        df.to_csv(out / f"{name}_buckets.csv", index=False)

    cfg = {
        "study_version": STUDY_VERSION,
        "session": str(session.resolve()),
        "reconstruction_dir": str(recon_dir.resolve()),
        "defensive_v1_dir": str(v1_dir.resolve()),
        "inventory_v2_dir": str(Path(inventory_v2_dir).resolve()) if inventory_v2_dir else None,
        "reference_fills": len(fills),
        "latency_scenarios_ms": list(LATENCIES_MS),
        "quote_lifetime_scenarios_s": list(QUOTE_LIFETIMES_S),
        "flow_windows_s": list(FLOW_WINDOWS_S),
        "interpretation": (
            "Mechanism diagnostics only. Static cancellation/lifetime scenarios are not path-exact strategy replays. "
            "Feature buckets are descriptive and must not be threshold-mined into a validated strategy on this same sample."
        ),
    }
    (out / "study_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if show:
        print("\n" + "=" * 122)
        print("M1-M5 MARKET-MAKING THEORY SUITE — DEFENSIVE V1 REFERENCE SAMPLE")
        print("=" * 122)
        print(f"Reference fills:       {len(fills):,}")
        print(f"Reference fill qty:    {fills['qty'].sum():.2f}")
        print(f"Reference fill-level M5 PnL: ${fills['pnl_to_m5_dollars'].sum():+.4f}")
        if not headline.empty:
            print(f"V1 reported strategy PnL:    ${_f(headline.iloc[0].get('net_mtm_pnl_before_fees')):+.4f}")
        print("\nIMPORTANT: stale-cancel and quote-lifetime scenario tables are STATIC fill-set counterfactuals, not path-exact strategy replays.")

        _print_table("THEORY SCOREBOARD", scoreboard)
        _print_table("STALE-BBO CANCEL LATENCY SCENARIOS", scenario_df[scenario_df["family"] == "STALE_BBO_CANCEL"])
        _print_table("MAX QUOTE LIFETIME SCENARIOS", scenario_df[scenario_df["family"] == "QUOTE_LIFETIME"])
        _print_table("BID / ASK ASYMMETRY", tables["side"])
        _print_table("QUEUE-BURN VELOCITY", tables["queue_burn"])
        _print_table("SUB-SECOND FLOW ACCELERATION", tables["flow_acceleration"])
        _print_table("MICROPRICE PRESSURE", tables["microprice"])
        _print_table("L2 DEPTH PRESSURE", tables["l2_pressure"])
        _print_table("SPREAD COMPENSATION", tables["spread"])
        _print_table("QUOTE AGE", tables["quote_age"])
        _print_table("QUEUE CANCELLATION-CREDIT EVIDENCE", tables["queue_credit"])
        _print_table("BTC COMMON SHOCK — ALTCOINS", tables["btc_shock"])
        if not forced_flat.empty:
            _print_table("FORCED INVENTORY FLATTEN: V2 VS V1", forced_flat)

        stale = fills["stale_before_fill"].fillna(False)
        print("\nSTALE-FILL RAW DIAGNOSTIC")
        print(f"  Fills with BBO change before fill: {int(stale.sum()):,}/{len(fills):,} ({100.0*stale.mean():.2f}%)")
        if stale.any():
            print(f"  Median stale-age-to-fill: {fills.loc[stale, 'stale_age_to_fill_ms'].median():.1f} ms")
            print(f"  P90 stale-age-to-fill:    {fills.loc[stale, 'stale_age_to_fill_ms'].quantile(.90):.1f} ms")

        qc = pd.to_numeric(fills["queue_cancel_credit_lb_ratio"], errors="coerce")
        print("\nQUEUE-CREDIT RAW DIAGNOSTIC")
        print(f"  Median lower-bound credit ratio: {qc.median():.3f}")
        print(f"  P75 lower-bound credit ratio:    {qc.quantile(.75):.3f}")
        print(f"  P90 lower-bound credit ratio:    {qc.quantile(.90):.3f}")

        print("\nINTERPRETATION RULE")
        print("  Use this suite to select ONE mechanism for a separately frozen exact replay.")
        print("  Do not combine favorable buckets from this same dataset into a new 'winning' strategy.")
        print("Outputs:", out)
        print("=" * 122)

        try:
            from IPython.display import display
            print("\nTHEORY SCOREBOARD")
            display(scoreboard.round(4))
        except Exception:
            pass

    return {
        "output_dir": out,
        "scoreboard": scoreboard,
        "scenario_summary": scenario_df,
        "fill_features": fills,
        "side": tables["side"],
        "queue_burn": tables["queue_burn"],
        "flow_acceleration": tables["flow_acceleration"],
        "microprice": tables["microprice"],
        "l2_pressure": tables["l2_pressure"],
        "spread": tables["spread"],
        "quote_age": tables["quote_age"],
        "queue_credit": tables["queue_credit"],
        "btc_shock": tables["btc_shock"],
        "forced_flatten": forced_flat,
    }


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--session", required=True)
    p.add_argument("--reconstruction-dir", required=True)
    p.add_argument("--defensive-v1-dir", required=True)
    p.add_argument("--inventory-v2-dir", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    run_mm_theory_suite(
        args.session,
        args.reconstruction_dir,
        args.defensive_v1_dir,
        inventory_v2_dir=args.inventory_v2_dir,
        output_dir=args.output_dir,
        show=True,
    )


if __name__ == "__main__":
    _main()
