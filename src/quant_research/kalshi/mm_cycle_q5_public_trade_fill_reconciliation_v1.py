from __future__ import annotations

"""Q5 live-fill vs public-trade reconciliation for a completed V12.x session.

Diagnostic purpose
------------------
Determine whether real passive ENTRY fills can be reconciled to the authoritative
public trade tape and, when they can, whether the frozen displayed-L1 queue model
would have predicted those fills from the *actual live CREATE* rather than from a
hypothetical shadow quote path.

This test answers two separate questions:
1) Does the public trade feed contain the trade that caused each real fill?
2) If yes, did enough compatible public trade quantity occur to burn the displayed
   queue ahead, or does the real fill require unobserved queue advancement such as
   cancellations ahead / queue-priority changes?

Scientific guardrails
---------------------
- SAME-REALIZATION forensic only; not independent validation.
- NO exchange/API calls and NO orders.
- Source session is read-only.
- Uses exact trade_id reconciliation first when available.
- Fallback price/side/time matching is explicitly labeled and sensitivity-tested.
- CREATE exchange-active time is unavailable.  Queue-replay results are reported
  under both optimistic SEND-bound and conservative ACK-bound activity assumptions.
- Public trade receipt_time is used for local causal ordering; exchange_time is used
  only for fill/trade timestamp diagnostics and fallback matching where available.
- Writes only under results/kalshi_q5_public_trade_fill_reconciliation/.
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

VERSION = "MM_CYCLE_Q5_PUBLIC_TRADE_FILL_RECONCILIATION_V1"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_public_trade_fill_reconciliation"
EPS = 1e-9
FALLBACK_TOL_S = 2.0
FALLBACK_SENSITIVITY_S = (0.10, 0.25, 0.50, 1.0, 2.0, 5.0)
AMBIGUITY_NEAR_S = 0.050


def _f(x, default=np.nan):
    return OOS._f(x, default)


def _iter_jsonl(path: Path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _ts_s(x):
    return OOS._ts(x)


def _ms_s(x):
    z = _f(x)
    return float(z) / 1000.0 if np.isfinite(z) else np.nan


def _price(x):
    z = _f(x)
    return round(float(z), 6) if np.isfinite(z) else np.nan


def _qty(row):
    for k in ("count_fp", "count", "qty", "quantity", "fill_count_fp", "fill_count"):
        z = _f(row.get(k))
        if np.isfinite(z):
            return abs(float(z))
    return np.nan


def _fill_price(row):
    for k in ("yes_price_dollars", "yes_price", "price_dollars", "price"):
        z = _f(row.get(k))
        if np.isfinite(z):
            return float(z)
    return np.nan


def _fill_time(row):
    # Exchange fill rows normally carry created_time. Keep fallbacks explicit.
    for k in ("created_time", "created_at", "trade_time", "time"):
        t = _ts_s(row.get(k))
        if np.isfinite(t):
            return float(t), k
    t = _ts_s(row.get("observed_time"))
    return (float(t), "observed_time") if np.isfinite(t) else (np.nan, None)


def _trade_time_exchange(row):
    t = _ts_s(row.get("exchange_time"))
    if np.isfinite(t):
        return float(t)
    z = _f(row.get("ts_ms"))
    if np.isfinite(z):
        return float(z) / 1000.0
    return np.nan


def _trade_time_receipt(row):
    t = _ts_s(row.get("receipt_time"))
    return float(t) if np.isfinite(t) else np.nan


def _trade_compatible(passive_side: str, order_px: float, tr: dict):
    taker = str(tr.get("taker_book_side") or "").lower()
    px = _f(tr.get("yes_price"))
    if not np.isfinite(px):
        return False, None
    side = str(passive_side).upper()
    if side == "BID":
        if taker != "ask" or px > order_px + EPS:
            return False, None
        kind = "EXACT" if abs(px - order_px) <= EPS else "TRADE_THROUGH"
        return True, kind
    if side == "ASK":
        if taker != "bid" or px < order_px - EPS:
            return False, None
        kind = "EXACT" if abs(px - order_px) <= EPS else "TRADE_THROUGH"
        return True, kind
    return False, None


def _new_output(source_name: str):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / source_name
    if out.exists():
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_ROOT / f"{source_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _load_order_catalog(session: Path):
    role_by_oid = {}
    create_latency = {}
    superseded = {}

    for r in _iter_jsonl(session / "latency_events_v12.jsonl") or []:
        if str(r.get("event") or "") != "CREATE_SENT":
            continue
        oid = str(r.get("order_id") or "")
        if not oid:
            continue
        role_by_oid[oid] = str(r.get("role") or "").upper()
        create_latency[oid] = r
        superseded[oid] = bool(r.get("superseded_at_send_detected"))

    queue_by_oid = defaultdict(list)
    for r in _iter_jsonl(session / "queue_positions.jsonl") or []:
        oid = str(r.get("order_id") or "")
        if not oid:
            continue
        t = _ts_s(r.get("time"))
        q = _f(r.get("queue_position"))
        if not np.isfinite(t) or not np.isfinite(q):
            continue
        queue_by_oid[oid].append({"time_s": float(t), "queue_position": max(0.0, float(q)), "raw": r})
    for rows in queue_by_oid.values():
        rows.sort(key=lambda z: z["time_s"])

    orders = {}
    for r in _iter_jsonl(session / "orders.jsonl") or []:
        if not str(r.get("action") or "").startswith("CREATE"):
            continue
        payload = r.get("payload") or {}
        response = r.get("response") or {}
        timing = r.get("timing") or {}
        oid = str(response.get("order_id") or "")
        if not oid:
            continue
        ticker = str(r.get("ticker") or payload.get("ticker") or "")
        side = str(payload.get("side") or "").upper()
        px = _f(payload.get("price"))
        qty = _f(payload.get("count"))
        send_s = _ms_s(timing.get("request_send_wall_ms"))
        ack_s = _ms_s(timing.get("response_recv_wall_ms"))
        lat = create_latency.get(oid) or {}
        if not np.isfinite(send_s):
            send_s = _ms_s(lat.get("request_send_wall_ms"))
        if not np.isfinite(ack_s):
            ack_s = _ms_s(lat.get("response_recv_wall_ms"))
        orders[oid] = {
            "order_id": oid,
            "ticker": ticker,
            "role": role_by_oid.get(oid, ""),
            "side": side,
            "price": float(px) if np.isfinite(px) else np.nan,
            "submitted_qty": float(qty) if np.isfinite(qty) else np.nan,
            "send_s": float(send_s) if np.isfinite(send_s) else np.nan,
            "ack_s": float(ack_s) if np.isfinite(ack_s) else np.nan,
            "superseded_at_send": bool(superseded.get(oid, False)),
            "queue_samples": list(queue_by_oid.get(oid) or []),
            "raw_create": r,
        }
    return orders


def _load_live_entry_fills(session: Path, orders: dict, selected_tickers: set[str]):
    rows = []
    time_sources = Counter()
    for r in _iter_jsonl(session / "fills.jsonl") or []:
        oid = str(r.get("order_id") or "")
        o = orders.get(oid)
        role = str(r.get("role") or (o or {}).get("role") or "").upper()
        if role != "ENTRY":
            continue
        ticker = str(r.get("market_ticker") or r.get("ticker") or (o or {}).get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        qty = _qty(r)
        px = _fill_price(r)
        t, src = _fill_time(r)
        side = str(r.get("strategy_side") or (o or {}).get("side") or "").upper()
        if not (np.isfinite(qty) and qty > 0 and np.isfinite(px) and np.isfinite(t) and side in {"BID", "ASK"}):
            continue
        time_sources[src] += 1
        rows.append({
            "fill_id": str(r.get("fill_id") or ""),
            "trade_id": str(r.get("trade_id") or ""),
            "order_id": oid,
            "ticker": ticker,
            "side": side,
            "qty": float(qty),
            "price": float(px),
            "fill_time_s": float(t),
            "fill_time_source": src,
            "decision_book": r.get("decision_book") or {},
            "raw_fill": r,
        })
    rows.sort(key=lambda z: (z["fill_time_s"], z["ticker"], z["order_id"]))
    return rows, dict(time_sources)


def _load_public_trades(raw: Path, selected_tickers: set[str]):
    trades = []
    by_trade_id = defaultdict(list)
    by_ticker = defaultdict(list)
    for r in _iter_jsonl(raw / "trades_event_time.jsonl") or []:
        ticker = str(r.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        px = _f(r.get("yes_price"))
        qty = _f(r.get("qty"))
        rt = _trade_time_receipt(r)
        et = _trade_time_exchange(r)
        if not (np.isfinite(px) and np.isfinite(qty) and qty > 0 and np.isfinite(rt)):
            continue
        z = {
            "ticker": ticker,
            "trade_id": str(r.get("trade_id") or ""),
            "price": float(px),
            "qty": float(qty),
            "taker_book_side": str(r.get("taker_book_side") or "").lower(),
            "receipt_s": float(rt),
            "exchange_s": float(et) if np.isfinite(et) else np.nan,
            "raw": r,
        }
        idx = len(trades)
        trades.append(z)
        by_ticker[ticker].append(idx)
        if z["trade_id"]:
            by_trade_id[z["trade_id"]].append(idx)
    for ticker, inds in by_ticker.items():
        inds.sort(key=lambda i: trades[i]["receipt_s"])
    return trades, by_trade_id, by_ticker


def _fallback_candidates(fill, trades, by_ticker, tol_s):
    out = []
    for i in by_ticker.get(fill["ticker"], []):
        tr = trades[i]
        ok, kind = _trade_compatible(fill["side"], fill["price"], tr)
        if not ok:
            continue
        # Prefer exchange timestamp when both sides have exchange timestamps.
        tt = tr["exchange_s"] if np.isfinite(tr["exchange_s"]) else tr["receipt_s"]
        dt = tt - fill["fill_time_s"]
        if abs(dt) <= float(tol_s):
            out.append((abs(dt), 0 if kind == "EXACT" else 1, dt, i, kind))
    out.sort(key=lambda z: (z[0], z[1], z[3]))
    return out


def _match_fill(fill, trades, by_trade_id, by_ticker, tol_s=FALLBACK_TOL_S):
    tid = fill["trade_id"]
    exact_id = []
    if tid:
        for i in by_trade_id.get(tid, []):
            tr = trades[i]
            ok, kind = _trade_compatible(fill["side"], fill["price"], tr)
            if ok:
                exact_id.append((i, kind))
    if exact_id:
        # trade_id should uniquely identify the public event; if duplicated, preserve ambiguity.
        i, kind = exact_id[0]
        tr = trades[i]
        dt_exch = tr["exchange_s"] - fill["fill_time_s"] if np.isfinite(tr["exchange_s"]) else np.nan
        return {
            "classification": "PUBLIC_TRADE_ID_MATCH" if len(exact_id) == 1 else "PUBLIC_TRADE_ID_AMBIGUOUS",
            "match_kind": kind,
            "trade_index": i,
            "candidate_count": len(exact_id),
            "dt_exchange_s": dt_exch,
            "dt_receipt_s": tr["receipt_s"] - fill["fill_time_s"],
        }

    cands = _fallback_candidates(fill, trades, by_ticker, tol_s)
    if not cands:
        return {
            "classification": "NO_PUBLIC_TRADE_MATCH",
            "match_kind": None,
            "trade_index": None,
            "candidate_count": 0,
            "dt_exchange_s": np.nan,
            "dt_receipt_s": np.nan,
        }
    best = cands[0]
    near = [x for x in cands if x[0] <= best[0] + AMBIGUITY_NEAR_S]
    _, _, dt, i, kind = best
    tr = trades[i]
    return {
        "classification": "PUBLIC_FALLBACK_MATCH" if len(near) == 1 else "PUBLIC_FALLBACK_AMBIGUOUS",
        "match_kind": kind,
        "trade_index": i,
        "candidate_count": len(near),
        "dt_exchange_s": dt if np.isfinite(tr["exchange_s"]) else np.nan,
        "dt_receipt_s": tr["receipt_s"] - fill["fill_time_s"],
    }


def _decision_queue_ahead(order_fills, side):
    for f in order_fills:
        b = f.get("decision_book") or {}
        key = "bid_q1" if str(side).upper() == "BID" else "ask_q1"
        z = _f(b.get(key))
        if np.isfinite(z):
            return max(0.0, float(z)), "fill_decision_book"
    return np.nan, None


def _first_prefill_queue_sample(order, first_fill_s):
    rows = [q for q in (order.get("queue_samples") or []) if q["time_s"] <= first_fill_s + EPS]
    return rows[0] if rows else None


def _public_path_stats(order, order_fills, fill_matches, trades, by_ticker, active_start_s):
    first_fill_s = min(f["fill_time_s"] for f in order_fills)
    last_fill_s = max(f["fill_time_s"] for f in order_fills)
    fill_qty = sum(float(f["qty"]) for f in order_fills)
    q0, q0_source = _decision_queue_ahead(order_fills, order["side"])

    matched_trade_receipts = []
    matched_trade_indices = []
    for f in order_fills:
        m = fill_matches[id(f)]
        i = m.get("trade_index")
        if i is not None:
            matched_trade_indices.append(i)
            matched_trade_receipts.append(trades[i]["receipt_s"])
    horizon_s = max(matched_trade_receipts) if matched_trade_receipts else last_fill_s + FALLBACK_TOL_S

    exact_qty = 0.0
    through_seen = False
    through_first_s = np.nan
    exact_rows = 0
    compatible_rows = 0
    for i in by_ticker.get(order["ticker"], []):
        tr = trades[i]
        if tr["receipt_s"] + EPS < active_start_s or tr["receipt_s"] > horizon_s + EPS:
            continue
        ok, kind = _trade_compatible(order["side"], order["price"], tr)
        if not ok:
            continue
        compatible_rows += 1
        if kind == "EXACT":
            exact_qty += tr["qty"]
            exact_rows += 1
        else:
            through_seen = True
            if not np.isfinite(through_first_s):
                through_first_s = tr["receipt_s"]

    if through_seen:
        any_fill_explained = True
        full_fill_explained = True
        needed_queue_reduction_for_any = 0.0
        needed_queue_reduction_for_full = 0.0
    elif np.isfinite(q0):
        any_fill_explained = bool(exact_qty > q0 + EPS)
        full_fill_explained = bool(exact_qty + EPS >= q0 + fill_qty)
        needed_queue_reduction_for_any = max(0.0, q0 - exact_qty + 1e-9)
        needed_queue_reduction_for_full = max(0.0, q0 + fill_qty - exact_qty)
    else:
        any_fill_explained = None
        full_fill_explained = None
        needed_queue_reduction_for_any = np.nan
        needed_queue_reduction_for_full = np.nan

    return {
        "active_start_s": active_start_s,
        "first_fill_s": first_fill_s,
        "last_fill_s": last_fill_s,
        "fill_qty": fill_qty,
        "displayed_queue_ahead": q0,
        "displayed_queue_source": q0_source,
        "public_exact_qty_until_fill": exact_qty,
        "public_exact_trade_rows_until_fill": exact_rows,
        "public_compatible_trade_rows_until_fill": compatible_rows,
        "public_trade_through_seen": through_seen,
        "first_trade_through_receipt_s": through_first_s,
        "displayed_model_any_fill_explained": any_fill_explained,
        "displayed_model_full_fill_explained": full_fill_explained,
        "queue_reduction_required_for_any_fill": needed_queue_reduction_for_any,
        "queue_reduction_required_for_full_fill": needed_queue_reduction_for_full,
    }


def _stats(vals):
    a = np.asarray([x for x in vals if np.isfinite(_f(x))], dtype=float)
    if not a.size:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
    }


def run_q5_public_trade_fill_reconciliation(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"
    required = [
        source / "fills.jsonl",
        source / "orders.jsonl",
        source / "latency_events_v12.jsonl",
        source / "queue_positions.jsonl",
        source / "events.jsonl",
        raw / "trades_event_time.jsonl",
        raw / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts: " + " | ".join(missing))

    cfg = OOS._read_json(source / "process_config.json", {}) or {}
    if str(cfg.get("mode") or "") != "LIVE_Q5_1H" or abs(_f(cfg.get("quote_size")) - BASE.QTY) > 1e-9:
        raise RuntimeError("Expected completed LIVE_Q5_1H / Q5 source session.")

    windows = BASE._live_windows(source)
    meta_rows, meta_by_ticker = BASE._metadata(raw)
    selected_tickers = {
        t for t, r in meta_by_ticker.items()
        if str(r.get("close_time") or "") in set(windows)
    }
    if not selected_tickers:
        raise RuntimeError("No raw tickers matched live windows.")

    orders = _load_order_catalog(source)
    fills, fill_time_sources = _load_live_entry_fills(source, orders, selected_tickers)
    if not fills:
        raise RuntimeError("No live ENTRY fills reconstructed.")

    trades, by_trade_id, by_ticker = _load_public_trades(raw, selected_tickers)
    if not trades:
        raise RuntimeError("No public trades reconstructed.")

    # Match every real ENTRY fill to the public tape.
    fill_matches = {}
    fill_rows = []
    for f in fills:
        m = _match_fill(f, trades, by_trade_id, by_ticker, FALLBACK_TOL_S)
        fill_matches[id(f)] = m
        tr = trades[m["trade_index"]] if m.get("trade_index") is not None else None
        meta = meta_by_ticker.get(f["ticker"]) or {}
        fill_rows.append({
            "fill_id": f["fill_id"],
            "trade_id": f["trade_id"],
            "order_id": f["order_id"],
            "ticker": f["ticker"],
            "series": str(meta.get("series_ticker") or ""),
            "close_time": str(meta.get("close_time") or ""),
            "side": f["side"],
            "fill_qty": f["qty"],
            "fill_price": f["price"],
            "fill_time": pd.Timestamp(f["fill_time_s"], unit="s", tz="UTC").isoformat(),
            "classification": m["classification"],
            "match_kind": m["match_kind"],
            "candidate_count": m["candidate_count"],
            "public_trade_id": tr["trade_id"] if tr else None,
            "public_price": tr["price"] if tr else np.nan,
            "public_qty": tr["qty"] if tr else np.nan,
            "public_taker_book_side": tr["taker_book_side"] if tr else None,
            "public_exchange_minus_fill_ms": 1000.0 * m["dt_exchange_s"] if np.isfinite(m["dt_exchange_s"]) else np.nan,
            "public_receipt_minus_fill_ms": 1000.0 * m["dt_receipt_s"] if np.isfinite(m["dt_receipt_s"]) else np.nan,
        })

    fill_df = pd.DataFrame(fill_rows)

    # Fallback tolerance sensitivity only for fills not already trade-id reconciled.
    sensitivity = []
    for tol in FALLBACK_SENSITIVITY_S:
        qty_id = qty_fb = qty_none = 0.0
        for f in fills:
            m0 = _match_fill(f, trades, by_trade_id, by_ticker, tol)
            q = f["qty"]
            if m0["classification"].startswith("PUBLIC_TRADE_ID"):
                qty_id += q
            elif m0["classification"].startswith("PUBLIC_FALLBACK"):
                qty_fb += q
            else:
                qty_none += q
        sensitivity.append({
            "tolerance_s": tol,
            "trade_id_match_qty": qty_id,
            "fallback_match_qty": qty_fb,
            "no_public_match_qty": qty_none,
        })
    sensitivity_df = pd.DataFrame(sensitivity)

    # Order-level queue replay from the *actual live CREATE*.
    fills_by_order = defaultdict(list)
    for f in fills:
        fills_by_order[f["order_id"]].append(f)

    order_rows = []
    for oid, fs in fills_by_order.items():
        o = orders.get(oid)
        if not o:
            continue
        fs.sort(key=lambda z: z["fill_time_s"])
        send_s = _f(o.get("send_s"))
        ack_s = _f(o.get("ack_s"))
        if not np.isfinite(send_s):
            continue
        send_path = _public_path_stats(o, fs, fill_matches, trades, by_ticker, float(send_s))
        ack_path = _public_path_stats(o, fs, fill_matches, trades, by_ticker, float(ack_s) if np.isfinite(ack_s) else float(send_s))
        prefill_q = _first_prefill_queue_sample(o, fs[0]["fill_time_s"])
        q0 = send_path["displayed_queue_ahead"]
        observed_q = _f((prefill_q or {}).get("queue_position"))
        observed_shift = observed_q - q0 if np.isfinite(observed_q) and np.isfinite(q0) else np.nan
        matched_qty = sum(
            f["qty"] for f in fs
            if fill_matches[id(f)]["classification"] != "NO_PUBLIC_TRADE_MATCH"
        )
        id_match_qty = sum(
            f["qty"] for f in fs
            if fill_matches[id(f)]["classification"].startswith("PUBLIC_TRADE_ID")
        )
        meta = meta_by_ticker.get(o["ticker"]) or {}
        order_rows.append({
            "order_id": oid,
            "ticker": o["ticker"],
            "series": str(meta.get("series_ticker") or ""),
            "close_time": str(meta.get("close_time") or ""),
            "side": o["side"],
            "price": o["price"],
            "fill_qty": send_path["fill_qty"],
            "public_matched_fill_qty": matched_qty,
            "trade_id_matched_fill_qty": id_match_qty,
            "superseded_at_send": o["superseded_at_send"],
            "send_to_ack_ms": 1000.0 * (ack_s - send_s) if np.isfinite(ack_s) else np.nan,
            "displayed_queue_ahead": q0,
            "prefill_queue_observed": np.isfinite(observed_q),
            "first_prefill_queue_position": observed_q,
            "observed_queue_minus_displayed": observed_shift,
            "send_exact_qty": send_path["public_exact_qty_until_fill"],
            "send_trade_through": send_path["public_trade_through_seen"],
            "send_any_fill_explained": send_path["displayed_model_any_fill_explained"],
            "send_full_fill_explained": send_path["displayed_model_full_fill_explained"],
            "send_queue_reduction_required_any": send_path["queue_reduction_required_for_any_fill"],
            "send_queue_reduction_required_full": send_path["queue_reduction_required_for_full_fill"],
            "ack_exact_qty": ack_path["public_exact_qty_until_fill"],
            "ack_trade_through": ack_path["public_trade_through_seen"],
            "ack_any_fill_explained": ack_path["displayed_model_any_fill_explained"],
            "ack_full_fill_explained": ack_path["displayed_model_full_fill_explained"],
            "ack_queue_reduction_required_any": ack_path["queue_reduction_required_for_any_fill"],
            "ack_queue_reduction_required_full": ack_path["queue_reduction_required_for_full_fill"],
        })

    order_df = pd.DataFrame(order_rows)

    total_fill_qty = float(fill_df["fill_qty"].sum())
    by_class = (
        fill_df.groupby("classification", as_index=False)
        .agg(fill_rows=("fill_id", "size"), fill_qty=("fill_qty", "sum"))
        .sort_values("fill_qty", ascending=False)
    )
    by_kind = (
        fill_df.groupby("match_kind", dropna=False, as_index=False)
        .agg(fill_rows=("fill_id", "size"), fill_qty=("fill_qty", "sum"))
        .sort_values("fill_qty", ascending=False)
    )
    by_window = (
        fill_df.groupby(["close_time", "classification"], as_index=False)
        .agg(fill_rows=("fill_id", "size"), fill_qty=("fill_qty", "sum"))
        .sort_values(["close_time", "classification"])
    )
    by_asset = (
        fill_df.groupby(["series", "classification"], as_index=False)
        .agg(fill_rows=("fill_id", "size"), fill_qty=("fill_qty", "sum"))
        .sort_values(["series", "classification"])
    )

    def _bool_qty(col):
        if order_df.empty:
            return {"true_qty": 0.0, "false_qty": 0.0, "unknown_qty": 0.0}
        tq = fq = uq = 0.0
        for _, r in order_df.iterrows():
            v, q = r.get(col), _f(r.get("fill_qty"), 0.0)
            if v is True or isinstance(v, (np.bool_, bool)) and bool(v):
                tq += q
            elif v is False or isinstance(v, (np.bool_, bool)) and not bool(v):
                fq += q
            else:
                uq += q
        return {"true_qty": tq, "false_qty": fq, "unknown_qty": uq}

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "live_windows": windows,
        "selected_tickers": len(selected_tickers),
        "live_entry_fill_rows": len(fill_df),
        "live_entry_fill_qty": total_fill_qty,
        "fill_time_sources": fill_time_sources,
        "public_trade_rows": len(trades),
        "classification_qty": {str(r.classification): float(r.fill_qty) for r in by_class.itertuples()},
        "trade_id_match_pct_qty": 100.0 * float(fill_df.loc[fill_df["classification"].str.startswith("PUBLIC_TRADE_ID"), "fill_qty"].sum()) / total_fill_qty if total_fill_qty else np.nan,
        "any_public_match_pct_qty": 100.0 * float(fill_df.loc[fill_df["classification"] != "NO_PUBLIC_TRADE_MATCH", "fill_qty"].sum()) / total_fill_qty if total_fill_qty else np.nan,
        "public_exchange_minus_fill_ms": _stats(fill_df["public_exchange_minus_fill_ms"].tolist()),
        "public_receipt_minus_fill_ms": _stats(fill_df["public_receipt_minus_fill_ms"].tolist()),
        "filled_orders": len(order_df),
        "prefill_queue_observation_orders": int(order_df["prefill_queue_observed"].sum()) if not order_df.empty else 0,
        "prefill_queue_observation_pct": 100.0 * float(order_df["prefill_queue_observed"].mean()) if not order_df.empty else np.nan,
        "observed_queue_minus_displayed": _stats(order_df["observed_queue_minus_displayed"].tolist()) if not order_df.empty else _stats([]),
        "send_bound_any_fill_explained": _bool_qty("send_any_fill_explained"),
        "send_bound_full_fill_explained": _bool_qty("send_full_fill_explained"),
        "ack_bound_any_fill_explained": _bool_qty("ack_any_fill_explained"),
        "ack_bound_full_fill_explained": _bool_qty("ack_full_fill_explained"),
        "send_queue_reduction_required_any": _stats(order_df["send_queue_reduction_required_any"].tolist()) if not order_df.empty else _stats([]),
        "send_queue_reduction_required_full": _stats(order_df["send_queue_reduction_required_full"].tolist()) if not order_df.empty else _stats([]),
        "ack_queue_reduction_required_any": _stats(order_df["ack_queue_reduction_required_any"].tolist()) if not order_df.empty else _stats([]),
        "ack_queue_reduction_required_full": _stats(order_df["ack_queue_reduction_required_full"].tolist()) if not order_df.empty else _stats([]),
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "interpretation_guardrail": (
            "Exact trade_id matches establish public-tape presence. Queue replay under SEND/ACK bounds tests whether the frozen displayed-L1/no-cancellation-credit model would have filled the actual live order. A large unmatched queue-reduction requirement supports cancellation-ahead or other queue-state dynamics missing from the frozen shadow; it is not proof of a specific exchange mechanism."
        ),
    }

    out = _new_output(source.name)
    summary["output_dir"] = str(out)
    OOS._atomic_json(out / "summary.json", summary)
    fill_df.to_csv(out / "live_entry_fill_public_trade_matches.csv", index=False)
    order_df.to_csv(out / "filled_entry_order_queue_replay.csv", index=False)
    sensitivity_df.to_csv(out / "fallback_tolerance_sensitivity.csv", index=False)
    by_class.to_csv(out / "by_classification.csv", index=False)
    by_kind.to_csv(out / "by_match_kind.csv", index=False)
    by_window.to_csv(out / "by_window.csv", index=False)
    by_asset.to_csv(out / "by_asset.csv", index=False)

    if show:
        print("=" * 124)
        print("Q5 PUBLIC-TRADE / LIVE-FILL RECONCILIATION — READ ONLY / NO API / NO ORDERS")
        print("=" * 124)
        print("Source:", source)
        print("Live ENTRY fills / qty:", len(fill_df), "/", f"{total_fill_qty:.4f}")
        print("Public trade rows:", f"{len(trades):,}")
        print("Fill time sources:", fill_time_sources)
        print()
        print("PUBLIC TRADE RECONCILIATION")
        print(by_class.to_string(index=False))
        print("trade-id matched qty %:", summary["trade_id_match_pct_qty"])
        print("any public match qty %:", summary["any_public_match_pct_qty"])
        print("public exchange_time - fill created_time ms:", summary["public_exchange_minus_fill_ms"])
        print("public receipt_time - fill created_time ms:", summary["public_receipt_minus_fill_ms"])
        print()
        print("MATCH KIND")
        print(by_kind.to_string(index=False))
        print()
        print("FALLBACK TOLERANCE SENSITIVITY")
        print(sensitivity_df.to_string(index=False))
        print()
        print("ACTUAL LIVE CREATE -> DISPLAYED-L1 QUEUE REPLAY")
        print("filled ENTRY orders:", summary["filled_orders"])
        print("prefill queue observation coverage %:", summary["prefill_queue_observation_pct"])
        print("first prefill queue - displayed queue:", summary["observed_queue_minus_displayed"])
        print("SEND-bound any fill explained qty:", summary["send_bound_any_fill_explained"])
        print("SEND-bound full fill explained qty:", summary["send_bound_full_fill_explained"])
        print("ACK-bound any fill explained qty:", summary["ack_bound_any_fill_explained"])
        print("ACK-bound full fill explained qty:", summary["ack_bound_full_fill_explained"])
        print("SEND-bound queue reduction required for any fill:", summary["send_queue_reduction_required_any"])
        print("ACK-bound queue reduction required for any fill:", summary["ack_queue_reduction_required_any"])
        print()
        print("BY WINDOW")
        print(by_window.to_string(index=False))
        print()
        print("BY ASSET")
        print(by_asset.to_string(index=False))
        print()
        if not order_df.empty:
            worst = order_df.sort_values("ack_queue_reduction_required_any", ascending=False).head(30)
            cols = [
                "ticker", "series", "close_time", "order_id", "side", "price", "fill_qty",
                "public_matched_fill_qty", "trade_id_matched_fill_qty", "displayed_queue_ahead",
                "first_prefill_queue_position", "observed_queue_minus_displayed",
                "send_exact_qty", "ack_exact_qty", "send_trade_through", "ack_trade_through",
                "send_any_fill_explained", "ack_any_fill_explained",
                "send_queue_reduction_required_any", "ack_queue_reduction_required_any",
            ]
            print("ORDERS REQUIRING THE MOST UNOBSERVED QUEUE ADVANCEMENT (ACK BOUND)")
            print(worst[cols].to_string(index=False))
        print()
        print("Interpretation guide:")
        print("  1) High trade-id match % => public trade tape contains the actual fill-causing events.")
        print("  2) High public-match but low displayed-L1 replay explanation => queue-ahead is disappearing without public trades burning it.")
        print("  3) Large required queue reduction quantifies how much cancellation-ahead / queue motion the frozen shadow is missing.")
        print("  4) Low public-match % => public trade capture/timestamp semantics must be investigated before queue conclusions.")
        print("  5) SEND vs ACK bounds bracket the unknown exact exchange acceptance time of each CREATE.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 124)

    return {
        "summary": summary,
        "fill_matches": fill_df,
        "order_replay": order_df,
        "sensitivity": sensitivity_df,
        "by_classification": by_class,
        "by_window": by_window,
        "by_asset": by_asset,
        "output_dir": out,
    }


__all__ = ["run_q5_public_trade_fill_reconciliation", "VERSION"]
