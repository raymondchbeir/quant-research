from __future__ import annotations

"""Corrected same-realization live-order/public-trade queue replay for Q5.

Why V4
------
Earlier reconciliation V1/V2 returned zero public matches, but schema diagnostic
V3 proved that every real ENTRY fill has a same-ticker public trade within ~1 ms
on exchange time and that the public ``taker_book_side`` is the aggressor order
side: passive BID fills correspond to public ASK aggressors; passive ASK fills
correspond to public BID aggressors.

This module therefore bypasses the earlier matcher and directly reconstructs each
real filled ENTRY order from its actual CREATE send/ack timestamps, displayed L1
queue ahead, live fill timestamps, and the same-ticker public trade tape.

It reports two activity bounds:
- SEND bound: order may be active from request send.
- ACK bound: order may be active only from create response receipt.

For each bound it measures exact-price queue burn, trade-through occurrence, and
how much queue ahead must have disappeared beyond observed public exact-price
trade volume for the real fill to be possible.

Scientific guardrails
---------------------
- SAME-REALIZATION forensic only; not independent validation.
- NO exchange/API calls; NO orders.
- Source session read-only.
- Public exchange timestamps are used to identify fill-causing events.
- Local public receipt timestamps are used for send/ack causal queue replay.
- Exact exchange acceptance time is unavailable, hence SEND/ACK bounds.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q5_same_realization_shadow_v1 as BASE
from . import mm_cycle_q5_public_trade_fill_reconciliation_v1 as V1

VERSION = "MM_CYCLE_Q5_LIVE_ORDER_PUBLIC_TRADE_QUEUE_REPLAY_V4"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_live_order_public_trade_queue_replay_v4"
EPS = 1e-9
FILL_MATCH_TOL_S = 0.050


def _f(x, default=np.nan):
    return OOS._f(x, default)


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _aggressor_compatible(passive_side: str, order_px: float, tr: dict):
    """Public side is aggressor order side, hence opposite the passive order."""
    side = str(passive_side).upper()
    agg = str(tr.get("taker_book_side") or "").lower()
    px = _f(tr.get("price"))
    if not np.isfinite(px):
        return False, None
    if side == "BID":
        if agg != "ask" or px > order_px + EPS:
            return False, None
        return True, "EXACT" if abs(px - order_px) <= EPS else "TRADE_THROUGH"
    if side == "ASK":
        if agg != "bid" or px < order_px - EPS:
            return False, None
        return True, "EXACT" if abs(px - order_px) <= EPS else "TRADE_THROUGH"
    return False, None


def _nearest_fill_trade(fill, trades, by_ticker):
    best = None
    for i in by_ticker.get(fill["ticker"], []):
        tr = trades[i]
        et = _f(tr.get("exchange_s"))
        if not np.isfinite(et):
            continue
        dt = et - fill["fill_time_s"]
        if abs(dt) > FILL_MATCH_TOL_S:
            continue
        ok, kind = _aggressor_compatible(fill["side"], fill["price"], tr)
        if not ok:
            continue
        cand = (abs(dt), 0 if kind == "EXACT" else 1, i, dt, kind)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    _, _, i, dt, kind = best
    return {"trade_index": i, "dt_exchange_s": dt, "kind": kind}


def _decision_queue_ahead(fills, side):
    key = "bid_q1" if str(side).upper() == "BID" else "ask_q1"
    for f in fills:
        z = _f((f.get("decision_book") or {}).get(key))
        if np.isfinite(z):
            return max(0.0, float(z)), "fill_decision_book"
    return np.nan, None


def _first_prefill_queue(order, first_fill_s):
    rows = [q for q in (order.get("queue_samples") or []) if _f(q.get("time_s")) <= first_fill_s + EPS]
    return rows[0] if rows else None


def _replay_bound(order, fills, trades, by_ticker, active_start_s):
    first_fill_s = min(f["fill_time_s"] for f in fills)
    last_fill_s = max(f["fill_time_s"] for f in fills)
    fill_qty = sum(float(f["qty"]) for f in fills)
    q0, qsrc = _decision_queue_ahead(fills, order["side"])

    exact_qty = 0.0
    exact_rows = 0
    through = False
    through_qty = 0.0
    compatible_qty = 0.0
    compatible_rows = 0

    # Replay on local receipt clock because CREATE send/ack are local wall-clock
    # timestamps and receipt_time is the local causal observation of public flow.
    for i in by_ticker.get(order["ticker"], []):
        tr = trades[i]
        rt = _f(tr.get("receipt_s"))
        if not np.isfinite(rt):
            continue
        if rt + EPS < active_start_s or rt > last_fill_s + 0.250:
            continue
        ok, kind = _aggressor_compatible(order["side"], order["price"], tr)
        if not ok:
            continue
        compatible_rows += 1
        compatible_qty += float(tr["qty"])
        if kind == "EXACT":
            exact_rows += 1
            exact_qty += float(tr["qty"])
        else:
            through = True
            through_qty += float(tr["qty"])

    if through:
        any_explained = True
        full_explained = True
        reduction_any = 0.0
        reduction_full = 0.0
    elif np.isfinite(q0):
        any_explained = exact_qty > q0 + EPS
        full_explained = exact_qty + EPS >= q0 + fill_qty
        reduction_any = max(0.0, q0 - exact_qty + 1e-9)
        reduction_full = max(0.0, q0 + fill_qty - exact_qty)
    else:
        any_explained = None
        full_explained = None
        reduction_any = np.nan
        reduction_full = np.nan

    return {
        "active_start_s": active_start_s,
        "displayed_queue_ahead": q0,
        "displayed_queue_source": qsrc,
        "exact_trade_qty_until_fill": exact_qty,
        "exact_trade_rows_until_fill": exact_rows,
        "compatible_trade_qty_until_fill": compatible_qty,
        "compatible_trade_rows_until_fill": compatible_rows,
        "trade_through_seen": through,
        "trade_through_qty": through_qty,
        "any_fill_explained": any_explained,
        "full_fill_explained": full_explained,
        "queue_reduction_required_any": reduction_any,
        "queue_reduction_required_full": reduction_full,
    }


def _qty_bool(rows, key):
    out = {"true_qty": 0.0, "false_qty": 0.0, "unknown_qty": 0.0}
    for r in rows:
        q = float(r["fill_qty"])
        v = r.get(key)
        if v is True:
            out["true_qty"] += q
        elif v is False:
            out["false_qty"] += q
        else:
            out["unknown_qty"] += q
    return out


def _stats(vals):
    a = np.asarray([float(x) for x in vals if np.isfinite(_f(x))], dtype=float)
    if not len(a):
        return {"n": 0, "mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
    }


def run_q5_live_order_public_trade_queue_replay(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"

    windows = BASE._live_windows(source)
    meta_rows, meta_by_ticker = BASE._metadata(raw)
    selected_tickers = {
        t for t, r in meta_by_ticker.items()
        if str(r.get("close_time") or "") in set(windows)
    }
    if not selected_tickers:
        raise RuntimeError("No selected Q5 tickers.")

    orders = V1._load_order_catalog(source)
    fills, fill_time_sources = V1._load_live_entry_fills(source, orders, selected_tickers)
    trades, by_trade_id, by_ticker = V1._load_public_trades(raw, selected_tickers)
    if not fills or not trades:
        raise RuntimeError("Need live ENTRY fills and public trades.")

    # Direct exchange-time fill reconciliation under semantics proven by V3.
    fill_matches = {}
    match_rows = []
    for f in fills:
        m = _nearest_fill_trade(f, trades, by_ticker)
        fill_matches[id(f)] = m
        match_rows.append({
            "ticker": f["ticker"], "order_id": f["order_id"], "fill_id": f["fill_id"],
            "fill_qty": f["qty"], "fill_price": f["price"], "side": f["side"],
            "matched": m is not None,
            "match_kind": m["kind"] if m else None,
            "exchange_dt_ms": 1000.0 * m["dt_exchange_s"] if m else np.nan,
            "public_trade_id": trades[m["trade_index"]]["trade_id"] if m else None,
            "live_trade_id": f.get("trade_id"),
        })
    match_df = pd.DataFrame(match_rows)

    fills_by_order = defaultdict(list)
    for f in fills:
        fills_by_order[f["order_id"]].append(f)

    rows = []
    for oid, fs in fills_by_order.items():
        o = orders.get(oid)
        if not o:
            continue
        fs.sort(key=lambda z: z["fill_time_s"])
        send_s = _f(o.get("send_s"))
        ack_s = _f(o.get("ack_s"))
        if not np.isfinite(send_s) or not np.isfinite(ack_s):
            continue
        send = _replay_bound(o, fs, trades, by_ticker, float(send_s))
        ack = _replay_bound(o, fs, trades, by_ticker, float(ack_s))
        first_fill_s = min(f["fill_time_s"] for f in fs)
        qobs = _first_prefill_queue(o, first_fill_s)
        meta = meta_by_ticker.get(o["ticker"]) or {}
        rows.append({
            "ticker": o["ticker"], "series": str(meta.get("series_ticker") or ""),
            "close_time": str(meta.get("close_time") or ""), "order_id": oid,
            "side": o["side"], "price": o["price"], "fill_qty": sum(f["qty"] for f in fs),
            "displayed_queue_ahead": send["displayed_queue_ahead"],
            "first_prefill_queue_position": _f((qobs or {}).get("queue_position")),
            "observed_queue_minus_displayed": (
                _f((qobs or {}).get("queue_position")) - send["displayed_queue_ahead"]
                if qobs and np.isfinite(send["displayed_queue_ahead"]) else np.nan
            ),
            "send_exact_qty": send["exact_trade_qty_until_fill"],
            "ack_exact_qty": ack["exact_trade_qty_until_fill"],
            "send_trade_through": send["trade_through_seen"],
            "ack_trade_through": ack["trade_through_seen"],
            "send_any_fill_explained": send["any_fill_explained"],
            "ack_any_fill_explained": ack["any_fill_explained"],
            "send_full_fill_explained": send["full_fill_explained"],
            "ack_full_fill_explained": ack["full_fill_explained"],
            "send_queue_reduction_required_any": send["queue_reduction_required_any"],
            "ack_queue_reduction_required_any": ack["queue_reduction_required_any"],
            "send_queue_reduction_required_full": send["queue_reduction_required_full"],
            "ack_queue_reduction_required_full": ack["queue_reduction_required_full"],
        })

    df = pd.DataFrame(rows)
    total_qty = float(sum(f["qty"] for f in fills))
    matched_qty = float(match_df.loc[match_df["matched"], "fill_qty"].sum())
    prefill = df["first_prefill_queue_position"].notna() if len(df) else pd.Series(dtype=bool)

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "fill_rows": len(fills), "fill_qty": total_qty,
        "filled_orders": len(df), "public_trade_rows": len(trades),
        "fill_time_sources": fill_time_sources,
        "exchange_time_reconciled_fill_qty": matched_qty,
        "exchange_time_reconciled_fill_qty_pct": 100.0 * matched_qty / total_qty if total_qty else np.nan,
        "fill_exchange_dt_ms": _stats(match_df.loc[match_df["matched"], "exchange_dt_ms"]),
        "prefill_queue_observation_coverage_pct": 100.0 * float(prefill.sum()) / len(df) if len(df) else np.nan,
        "first_prefill_queue_minus_displayed": _stats(df.loc[prefill, "observed_queue_minus_displayed"]) if len(df) else _stats([]),
        "send_any_fill_explained_qty": _qty_bool(rows, "send_any_fill_explained"),
        "ack_any_fill_explained_qty": _qty_bool(rows, "ack_any_fill_explained"),
        "send_full_fill_explained_qty": _qty_bool(rows, "send_full_fill_explained"),
        "ack_full_fill_explained_qty": _qty_bool(rows, "ack_full_fill_explained"),
        "send_queue_reduction_required_any": _stats(df["send_queue_reduction_required_any"]) if len(df) else _stats([]),
        "ack_queue_reduction_required_any": _stats(df["ack_queue_reduction_required_any"]) if len(df) else _stats([]),
        "send_queue_reduction_required_full": _stats(df["send_queue_reduction_required_full"]) if len(df) else _stats([]),
        "ack_queue_reduction_required_full": _stats(df["ack_queue_reduction_required_full"]) if len(df) else _stats([]),
        "same_realization_only": True, "independent_validation": False,
        "source_modified": False, "exchange_api_called": False, "orders_sent": False,
    }

    out = _new_output(source.name)
    match_df.to_csv(out / "live_fill_public_trade_matches.csv", index=False)
    df.to_csv(out / "actual_live_order_queue_replay.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("=" * 124)
        print("Q5 CORRECTED LIVE-ORDER / PUBLIC-TRADE QUEUE REPLAY V4 — READ ONLY")
        print("=" * 124)
        print("Source:", source)
        print("Live ENTRY fills / qty:", len(fills), "/", f"{total_qty:.4f}")
        print("Filled ENTRY orders:", len(df))
        print("Public trade rows:", len(trades))
        print("Exchange-time reconciled fill qty %:", summary["exchange_time_reconciled_fill_qty_pct"])
        print("Fill/public exchange dt ms:", summary["fill_exchange_dt_ms"])
        print()
        print("PREFILL QUEUE OBSERVATIONS")
        print(" coverage %:", summary["prefill_queue_observation_coverage_pct"])
        print(" first prefill queue - displayed:", summary["first_prefill_queue_minus_displayed"])
        print()
        print("DISPLAYED-L1 + PUBLIC-TRADE REPLAY")
        print(" SEND any fill explained qty:", summary["send_any_fill_explained_qty"])
        print(" ACK  any fill explained qty:", summary["ack_any_fill_explained_qty"])
        print(" SEND full fill explained qty:", summary["send_full_fill_explained_qty"])
        print(" ACK  full fill explained qty:", summary["ack_full_fill_explained_qty"])
        print(" SEND queue reduction required ANY:", summary["send_queue_reduction_required_any"])
        print(" ACK  queue reduction required ANY:", summary["ack_queue_reduction_required_any"])
        print(" SEND queue reduction required FULL:", summary["send_queue_reduction_required_full"])
        print(" ACK  queue reduction required FULL:", summary["ack_queue_reduction_required_full"])
        print()
        print("TOP 30 ORDERS REQUIRING MOST ACK-BOUND UNOBSERVED QUEUE REDUCTION")
        cols = ["ticker", "series", "close_time", "order_id", "side", "price", "fill_qty",
                "displayed_queue_ahead", "first_prefill_queue_position", "observed_queue_minus_displayed",
                "ack_exact_qty", "ack_trade_through", "ack_any_fill_explained",
                "ack_queue_reduction_required_any", "ack_queue_reduction_required_full"]
        if len(df):
            print(df.sort_values("ack_queue_reduction_required_any", ascending=False)[cols].head(30).to_string(index=False))
        print()
        print("Interpretation:")
        print("  - ~100% exchange-time fill reconciliation confirms the public tape contains the fill-causing flow.")
        print("  - High explained qty means displayed L1 + public trades can reproduce real fills from actual live CREATEs.")
        print("  - Low explained qty with positive required reduction means queue ahead disappeared without exact-price public trades burning it.")
        print("  - SEND/ACK bracket unknown exact exchange acceptance time; conclusions robust to both are strongest.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 124)

    return {"summary": summary, "fill_matches": match_df, "order_replay": df, "output_dir": out}


__all__ = ["run_q5_live_order_public_trade_queue_replay", "VERSION"]