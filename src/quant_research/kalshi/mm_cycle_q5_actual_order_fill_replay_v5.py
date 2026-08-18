from __future__ import annotations

"""Q5 actual-live-order tape fill replay against the frozen public-trade queue model.

Purpose
-------
Condition on the orders that the live V12.2 engine actually submitted, rather than
on the hypothetical quote path produced by the frozen strategy shadow.  For every
passive ENTRY/EXIT CREATE we reconstruct a plausible active interval from the real
CREATE and CANCEL timestamps, seed queue-ahead from the displayed L1 quantity at
join, and replay the recorded public trade tape using the frozen queue mechanics:

- exact-price aggressive flow burns queue ahead first;
- once exact-price flow reaches our order, leftover quantity fills us;
- a trade-through fills the resting order;
- after the first predicted fill event, residual quantity is treated as cancelled,
  matching the frozen shadow's any-fill-cancels-residual convention.

This separates two layers:
1) strategy/order-path divergence: which orders were actually placed and when;
2) fill-model divergence: whether the frozen queue model reproduces fills once
   given those exact live orders.

Activity bounds
---------------
Exact exchange-active/cancel-effective instants are unavailable, so two causal
bounds are reported:
- NARROW: CREATE ACK -> CANCEL SEND (minimum plausible resting exposure)
- WIDE:   CREATE SEND -> CANCEL ACK (maximum plausible resting exposure)

If a successful cancel timestamp is unavailable, the next CREATE send for the same
ticker is used as a hard upper bound; otherwise the strategy M5 boundary is used.

Scientific guardrails
---------------------
- SAME-REALIZATION execution forensic only; NOT independent validation.
- NO exchange/API calls and NO orders.
- Source session is read-only.
- Does NOT claim counterfactual strategy PnL: the live CREATE/CANCEL tape is
  endogenous to the real fills/positions.  This module validates fill mechanics,
  not a self-consistent alternative portfolio path.
- Writes only under results/kalshi_q5_actual_order_fill_replay_v5/.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q5_same_realization_shadow_v1 as BASE
from . import mm_cycle_q5_public_trade_fill_reconciliation_v1 as V1
from . import mm_cycle_q5_live_shadow_fill_forensics_v1 as FSEL
from . import mm_cycle_q5_live_order_public_trade_queue_replay_v4 as V4

VERSION = "MM_CYCLE_Q5_ACTUAL_ORDER_FILL_REPLAY_V5"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_actual_order_fill_replay_v5"
EPS = 1e-9


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


def _ms_s(x):
    z = _f(x)
    return float(z) / 1000.0 if np.isfinite(z) else np.nan


def _ts_s(x):
    return OOS._ts(x)


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_ROOT / f"{name}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _timing_pair(obj):
    """Extract request-send / response-receive wall times from a timing-like dict."""
    if not isinstance(obj, dict):
        return np.nan, np.nan
    s = _ms_s(obj.get("request_send_wall_ms"))
    a = _ms_s(obj.get("response_recv_wall_ms"))
    return s, a


def _nested_timing_candidates(row):
    """Yield timing dictionaries from the cancellation logging shapes used by V11/V12."""
    if not isinstance(row, dict):
        return
    for key in ("cancel_timing", "batch_timing", "timing"):
        z = row.get(key)
        if isinstance(z, dict):
            yield key, z
    result = row.get("result")
    if isinstance(result, dict):
        for key in ("timing", "cancel_timing", "batch_timing"):
            z = result.get(key)
            if isinstance(z, dict):
                yield f"result.{key}", z


def _cancel_catalog(session: Path):
    """Earliest observed cancel-send/ack candidates by order id.

    Successful priority-cancel results are taken from latency_events_v12. Main-path
    verified V11/V12 cancel rows are taken from orders.jsonl.  We preserve all
    candidates and later use the earliest finite send/ack for each order.
    """
    by_oid = defaultdict(list)

    for r in _iter_jsonl(session / "latency_events_v12.jsonl") or []:
        if str(r.get("event") or "") != "FAST_CANCEL_RESULT":
            continue
        if r.get("success") is not True:
            continue
        oid = str(r.get("order_id") or "")
        if not oid:
            continue
        send_s = _ms_s(r.get("request_send_wall_ms"))
        ack_s = _ms_s(r.get("response_recv_wall_ms"))
        if np.isfinite(send_s) or np.isfinite(ack_s):
            by_oid[oid].append({
                "source": "FAST_CANCEL_RESULT",
                "send_s": send_s,
                "ack_s": ack_s,
                "raw": r,
            })

    for r in _iter_jsonl(session / "orders.jsonl") or []:
        action = str(r.get("action") or "").upper()
        if "CANCEL" not in action:
            continue
        track = r.get("track") or {}
        oid = str(r.get("order_id") or track.get("order_id") or "")
        if not oid:
            # Some older cancellation rows may nest the order id in a result/body.
            for container_key in ("result", "cancel_body", "batch_body"):
                z = r.get(container_key)
                if isinstance(z, dict) and z.get("order_id"):
                    oid = str(z.get("order_id"))
                    break
        if not oid:
            continue
        found = False
        for label, timing in _nested_timing_candidates(r):
            send_s, ack_s = _timing_pair(timing)
            if np.isfinite(send_s) or np.isfinite(ack_s):
                by_oid[oid].append({
                    "source": f"ORDERS:{action}:{label}",
                    "send_s": send_s,
                    "ack_s": ack_s,
                    "raw": r,
                })
                found = True
        # If the row itself has no nested timing but its logged wall time exists,
        # retain it as an ACK-like upper bound only. Never fabricate a send time.
        if not found:
            t = _ts_s(r.get("time"))
            if np.isfinite(t):
                by_oid[oid].append({
                    "source": f"ORDERS:{action}:row_time_upper_bound",
                    "send_s": np.nan,
                    "ack_s": float(t),
                    "raw": r,
                })

    out = {}
    for oid, rows in by_oid.items():
        sends = [(x["send_s"], x["source"]) for x in rows if np.isfinite(x["send_s"])]
        acks = [(x["ack_s"], x["source"]) for x in rows if np.isfinite(x["ack_s"])]
        send_s, send_src = min(sends, default=(np.nan, None), key=lambda z: z[0])
        ack_s, ack_src = min(acks, default=(np.nan, None), key=lambda z: z[0])
        out[oid] = {
            "cancel_send_s": float(send_s) if np.isfinite(send_s) else np.nan,
            "cancel_ack_s": float(ack_s) if np.isfinite(ack_s) else np.nan,
            "cancel_send_source": send_src,
            "cancel_ack_source": ack_src,
            "candidate_count": len(rows),
        }
    return out


def _displayed_queue_catalog(session: Path, orders: dict, live_fills: list[dict]):
    """Displayed queue at join, without requiring a successful queue-position poll."""
    out = {}

    # V12 queue logs may record displayed L1 even when queue_position itself is NaN.
    for r in _iter_jsonl(session / "queue_positions.jsonl") or []:
        oid = str(r.get("order_id") or "")
        if not oid or oid in out:
            continue
        for key in ("displayed_l1_ahead_at_join", "displayed_l1_ahead"):
            z = _f(r.get(key))
            if np.isfinite(z):
                out[oid] = {"queue_ahead": max(0.0, float(z)), "source": f"queue_positions.{key}"}
                break

    # Actual fills carry the decision book that generated their order. This closes
    # coverage for short-lived filled orders whose queue poll never returned.
    fills_by_oid = defaultdict(list)
    for f in live_fills:
        fills_by_oid[str(f.get("order_id") or "")].append(f)

    # FSEL's compact live-fill loader does not retain decision_book, so scan the
    # raw fill log directly for these fallbacks.
    raw_fill_book = {}
    for r in _iter_jsonl(session / "fills.jsonl") or []:
        oid = str(r.get("order_id") or "")
        if not oid or oid in raw_fill_book:
            continue
        b = r.get("decision_book") or {}
        if isinstance(b, dict):
            raw_fill_book[oid] = b

    for oid, o in orders.items():
        if oid in out:
            continue
        b = raw_fill_book.get(oid) or {}
        side = str(o.get("side") or "").upper()
        key = "bid_q1" if side == "BID" else "ask_q1" if side == "ASK" else None
        z = _f(b.get(key)) if key else np.nan
        if np.isfinite(z):
            out[oid] = {"queue_ahead": max(0.0, float(z)), "source": f"fills.decision_book.{key}"}

    return out


def _m5_end_s(meta):
    close = _ts_s((meta or {}).get("close_time"))
    return float(close - 600.0) if np.isfinite(close) else np.nan


def _next_create_send_by_order(orders: dict):
    by_ticker = defaultdict(list)
    for oid, o in orders.items():
        s = _f(o.get("send_s"))
        if o.get("ticker") and np.isfinite(s):
            by_ticker[str(o["ticker"])].append((float(s), oid))
    out = {}
    for ticker, rows in by_ticker.items():
        rows.sort()
        for i, (s, oid) in enumerate(rows):
            out[oid] = rows[i + 1][0] if i + 1 < len(rows) else np.nan
    return out


def _active_bounds(order, cancel, next_create_s, meta):
    create_send = _f(order.get("send_s"))
    create_ack = _f(order.get("ack_s"))
    cancel_send = _f((cancel or {}).get("cancel_send_s"))
    cancel_ack = _f((cancel or {}).get("cancel_ack_s"))
    m5 = _m5_end_s(meta)

    # End fallbacks are conservative and deterministic. A next CREATE for a ticker
    # cannot coexist with the prior active-track order under the live engine's
    # one-active-order-per-ticker invariant.
    fallback_end = next_create_s if np.isfinite(_f(next_create_s)) else m5

    narrow_start = create_ack if np.isfinite(create_ack) else create_send
    wide_start = create_send if np.isfinite(create_send) else create_ack

    narrow_end = cancel_send
    narrow_end_src = (cancel or {}).get("cancel_send_source")
    if not np.isfinite(narrow_end):
        narrow_end = cancel_ack
        narrow_end_src = (cancel or {}).get("cancel_ack_source")
    if not np.isfinite(narrow_end):
        narrow_end = fallback_end
        narrow_end_src = "NEXT_CREATE_OR_M5_FALLBACK"

    wide_end = cancel_ack
    wide_end_src = (cancel or {}).get("cancel_ack_source")
    if not np.isfinite(wide_end):
        wide_end = cancel_send
        wide_end_src = (cancel or {}).get("cancel_send_source")
    if not np.isfinite(wide_end):
        wide_end = fallback_end
        wide_end_src = "NEXT_CREATE_OR_M5_FALLBACK"

    # Never let an interval extend through the next CREATE or beyond M5.
    caps = [x for x in (next_create_s, m5) if np.isfinite(_f(x))]
    if caps:
        cap = min(float(x) for x in caps)
        if np.isfinite(narrow_end):
            narrow_end = min(float(narrow_end), cap)
        if np.isfinite(wide_end):
            wide_end = min(float(wide_end), cap)

    return {
        "create_send_s": create_send,
        "create_ack_s": create_ack,
        "cancel_send_s": cancel_send,
        "cancel_ack_s": cancel_ack,
        "narrow_start_s": narrow_start,
        "narrow_end_s": narrow_end,
        "wide_start_s": wide_start,
        "wide_end_s": wide_end,
        "narrow_end_source": narrow_end_src,
        "wide_end_source": wide_end_src,
    }


def _replay_one(order, queue_ahead, trades, by_ticker, start_s, end_s):
    """Frozen queue fill mechanics on one actual live resting order."""
    if not (np.isfinite(_f(start_s)) and np.isfinite(_f(end_s)) and end_s + EPS >= start_s):
        return {"known": False, "reason": "INVALID_ACTIVE_INTERVAL"}
    if not np.isfinite(_f(queue_ahead)):
        return {"known": False, "reason": "MISSING_DISPLAYED_QUEUE"}

    ticker = str(order.get("ticker") or "")
    side = str(order.get("side") or "").upper()
    px = _f(order.get("price"))
    qty = _f(order.get("submitted_qty"))
    if not ticker or side not in {"BID", "ASK"} or not np.isfinite(px) or not np.isfinite(qty):
        return {"known": False, "reason": "INVALID_ORDER_FIELDS"}

    ahead = max(0.0, float(queue_ahead))
    submitted = max(0.0, float(qty))
    exact_seen = 0.0
    compatible_seen = 0.0
    predicted_fill = 0.0
    predicted_fill_s = np.nan
    predicted_kind = None
    predicted_trade_id = None

    for i in by_ticker.get(ticker, []):
        tr = trades[i]
        rt = _f(tr.get("receipt_s"))
        if not np.isfinite(rt):
            continue
        if rt + EPS < start_s:
            continue
        if rt > end_s + EPS:
            break
        ok, kind = V4._aggressor_compatible(side, float(px), tr)
        if not ok:
            continue
        tq = max(0.0, float(tr["qty"]))
        compatible_seen += tq

        if kind == "TRADE_THROUGH":
            predicted_fill = submitted
            predicted_fill_s = float(rt)
            predicted_kind = kind
            predicted_trade_id = tr.get("trade_id")
            break

        # EXACT price: trade quantity first burns queue ahead. Any excess reaches
        # our resting order. The frozen shadow cancels residual after any fill.
        exact_seen += tq
        if ahead > EPS:
            burned = min(ahead, tq)
            ahead -= burned
            tq -= burned
        if tq > EPS and submitted > EPS:
            predicted_fill = min(submitted, tq)
            predicted_fill_s = float(rt)
            predicted_kind = "EXACT"
            predicted_trade_id = tr.get("trade_id")
            break

    return {
        "known": True,
        "reason": None,
        "predicted_fill_qty": float(predicted_fill),
        "predicted_filled": bool(predicted_fill > EPS),
        "predicted_fill_s": predicted_fill_s,
        "predicted_fill_kind": predicted_kind,
        "predicted_trade_id": predicted_trade_id,
        "exact_trade_qty_seen": float(exact_seen),
        "compatible_trade_qty_seen": float(compatible_seen),
        "queue_ahead_remaining": float(ahead),
    }


def _actual_fill_catalog(live_fills):
    by_oid = defaultdict(list)
    for f in live_fills:
        by_oid[str(f.get("order_id") or "")].append(f)
    out = {}
    for oid, fs in by_oid.items():
        fs.sort(key=lambda z: z["fill_s"])
        out[oid] = {
            "actual_fill_qty": float(sum(float(f["qty"]) for f in fs)),
            "actual_fill_rows": len(fs),
            "actual_first_fill_s": float(fs[0]["fill_s"]),
            "actual_last_fill_s": float(fs[-1]["fill_s"]),
            "actual_role": str(fs[0].get("role") or "").upper(),
        }
    return out


def _confusion(df, prefix, role):
    x = df[(df["role"] == role) & df[f"{prefix}_known"]].copy()
    if x.empty:
        return {
            "known_orders": 0, "actual_filled_orders": 0, "predicted_filled_orders": 0,
            "tp": 0, "fn": 0, "fp": 0, "tn": 0,
            "actual_fill_qty": 0.0, "predicted_fill_qty": 0.0,
        }
    actual = x["actual_fill_qty"] > EPS
    pred = x[f"{prefix}_predicted_fill_qty"] > EPS
    return {
        "known_orders": int(len(x)),
        "actual_filled_orders": int(actual.sum()),
        "predicted_filled_orders": int(pred.sum()),
        "tp": int((actual & pred).sum()),
        "fn": int((actual & ~pred).sum()),
        "fp": int((~actual & pred).sum()),
        "tn": int((~actual & ~pred).sum()),
        "actual_fill_qty": float(x["actual_fill_qty"].sum()),
        "predicted_fill_qty": float(x[f"{prefix}_predicted_fill_qty"].sum()),
        "actual_qty_on_tp": float(x.loc[actual & pred, "actual_fill_qty"].sum()),
        "predicted_qty_on_tp": float(x.loc[actual & pred, f"{prefix}_predicted_fill_qty"].sum()),
        "actual_qty_on_fn": float(x.loc[actual & ~pred, "actual_fill_qty"].sum()),
        "predicted_qty_on_fp": float(x.loc[~actual & pred, f"{prefix}_predicted_fill_qty"].sum()),
    }


def _timing_stats(df, prefix, role):
    x = df[
        (df["role"] == role)
        & (df["actual_fill_qty"] > EPS)
        & (df[f"{prefix}_predicted_fill_qty"] > EPS)
    ].copy()
    if x.empty:
        return {"n": 0, "median_ms": np.nan, "p10_ms": np.nan, "p90_ms": np.nan, "p95_ms": np.nan}
    a = 1000.0 * (x[f"{prefix}_predicted_fill_s"] - x["actual_first_fill_s"])
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(a):
        return {"n": 0, "median_ms": np.nan, "p10_ms": np.nan, "p90_ms": np.nan, "p95_ms": np.nan}
    return {
        "n": int(len(a)),
        "median_ms": float(np.median(a)),
        "p10_ms": float(np.quantile(a, 0.10)),
        "p90_ms": float(np.quantile(a, 0.90)),
        "p95_ms": float(np.quantile(a, 0.95)),
    }


def _baseline_shadow_entry_qty(source: Path):
    try:
        d, _ = FSEL._find_baseline_shadow(source)
        path = d / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1" / "shadow_fills.jsonl"
        q = 0.0
        rows = 0
        for r in _iter_jsonl(path) or []:
            if str(r.get("role") or "").upper() != "ENTRY":
                continue
            z = _f(r.get("qty"))
            if np.isfinite(z):
                q += float(z)
                rows += 1
        return {"entry_qty": q, "entry_fill_rows": rows, "shadow_dir": str(d)}
    except Exception as exc:
        return {"entry_qty": np.nan, "entry_fill_rows": 0, "shadow_dir": None, "error": repr(exc)}


def run_q5_actual_order_fill_replay(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"

    required = [
        source / "orders.jsonl",
        source / "fills.jsonl",
        source / "latency_events_v12.jsonl",
        source / "queue_positions.jsonl",
        raw / "trades_event_time.jsonl",
        raw / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts: " + " | ".join(missing))

    cfg = OOS._read_json(source / "process_config.json", {}) or {}
    if str(cfg.get("mode") or "") != "LIVE_Q5_1H":
        raise RuntimeError("Expected completed LIVE_Q5_1H source session.")

    windows = BASE._live_windows(source)
    meta_rows, meta_by_ticker = BASE._metadata(raw)
    selected_tickers = {
        t for t, r in meta_by_ticker.items()
        if str(r.get("close_time") or "") in set(windows)
    }
    if not selected_tickers:
        raise RuntimeError("No selected Q5 tickers.")

    orders = V1._load_order_catalog(source)
    # Keep only the exact completed Q5 market set and passive strategy roles.
    orders = {
        oid: o for oid, o in orders.items()
        if str(o.get("ticker") or "") in selected_tickers
        and str(o.get("role") or "").upper() in {"ENTRY", "EXIT"}
    }
    if not orders:
        raise RuntimeError("No passive live CREATE orders reconstructed.")

    live_fills, fill_time_sources = FSEL._load_live_fills(source)
    live_fills = [
        f for f in live_fills
        if str(f.get("ticker") or "") in selected_tickers
        and str(f.get("role") or "").upper() in {"ENTRY", "EXIT"}
    ]
    actual = _actual_fill_catalog(live_fills)

    trades, _, by_ticker = V1._load_public_trades(raw, selected_tickers)
    if not trades:
        raise RuntimeError("No public trades reconstructed.")

    cancels = _cancel_catalog(source)
    qcat = _displayed_queue_catalog(source, orders, live_fills)
    next_create = _next_create_send_by_order(orders)

    rows = []
    for oid, o in sorted(orders.items(), key=lambda kv: _f(kv[1].get("send_s"), 1e99)):
        ticker = str(o.get("ticker") or "")
        meta = meta_by_ticker.get(ticker) or {}
        role = str(o.get("role") or "").upper()
        qrec = qcat.get(oid) or {}
        q0 = _f(qrec.get("queue_ahead"))
        bounds = _active_bounds(o, cancels.get(oid), next_create.get(oid), meta)
        narrow = _replay_one(
            o, q0, trades, by_ticker,
            bounds["narrow_start_s"], bounds["narrow_end_s"],
        )
        wide = _replay_one(
            o, q0, trades, by_ticker,
            bounds["wide_start_s"], bounds["wide_end_s"],
        )
        ar = actual.get(oid) or {}
        actual_qty = float(ar.get("actual_fill_qty", 0.0))
        actual_first = _f(ar.get("actual_first_fill_s"))

        rows.append({
            "ticker": ticker,
            "series": str(meta.get("series_ticker") or ""),
            "close_time": str(meta.get("close_time") or ""),
            "order_id": oid,
            "role": role,
            "side": str(o.get("side") or "").upper(),
            "price": _f(o.get("price")),
            "submitted_qty": _f(o.get("submitted_qty")),
            "displayed_queue_ahead": q0,
            "displayed_queue_source": qrec.get("source"),
            "actual_fill_qty": actual_qty,
            "actual_fill_rows": int(ar.get("actual_fill_rows", 0)),
            "actual_first_fill_s": actual_first,
            "actual_filled": bool(actual_qty > EPS),
            **bounds,
            "narrow_known": bool(narrow.get("known")),
            "narrow_reason": narrow.get("reason"),
            "narrow_predicted_fill_qty": _f(narrow.get("predicted_fill_qty"), 0.0) if narrow.get("known") else np.nan,
            "narrow_predicted_fill_s": _f(narrow.get("predicted_fill_s")),
            "narrow_predicted_fill_kind": narrow.get("predicted_fill_kind"),
            "narrow_exact_trade_qty_seen": _f(narrow.get("exact_trade_qty_seen")),
            "narrow_queue_ahead_remaining": _f(narrow.get("queue_ahead_remaining")),
            "wide_known": bool(wide.get("known")),
            "wide_reason": wide.get("reason"),
            "wide_predicted_fill_qty": _f(wide.get("predicted_fill_qty"), 0.0) if wide.get("known") else np.nan,
            "wide_predicted_fill_s": _f(wide.get("predicted_fill_s")),
            "wide_predicted_fill_kind": wide.get("predicted_fill_kind"),
            "wide_exact_trade_qty_seen": _f(wide.get("exact_trade_qty_seen")),
            "wide_queue_ahead_remaining": _f(wide.get("queue_ahead_remaining")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No order replay rows produced.")

    baseline = _baseline_shadow_entry_qty(source)
    entry_actual_all = float(sum(f["qty"] for f in live_fills if str(f.get("role") or "").upper() == "ENTRY"))
    exit_actual_all = float(sum(f["qty"] for f in live_fills if str(f.get("role") or "").upper() == "EXIT"))

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "live_windows": windows,
        "selected_tickers": len(selected_tickers),
        "passive_create_orders": len(df),
        "entry_create_orders": int((df["role"] == "ENTRY").sum()),
        "exit_create_orders": int((df["role"] == "EXIT").sum()),
        "displayed_queue_known_orders": int(df["displayed_queue_ahead"].notna().sum()),
        "displayed_queue_coverage_pct": 100.0 * float(df["displayed_queue_ahead"].notna().sum()) / len(df),
        "cancel_send_known_orders": int(df["cancel_send_s"].notna().sum()),
        "cancel_ack_known_orders": int(df["cancel_ack_s"].notna().sum()),
        "public_trade_rows": len(trades),
        "fill_time_sources": fill_time_sources,
        "actual_live_entry_fill_qty_all": entry_actual_all,
        "actual_live_exit_fill_qty_all": exit_actual_all,
        "baseline_strategy_shadow": baseline,
        "narrow_entry": _confusion(df, "narrow", "ENTRY"),
        "wide_entry": _confusion(df, "wide", "ENTRY"),
        "narrow_exit": _confusion(df, "narrow", "EXIT"),
        "wide_exit": _confusion(df, "wide", "EXIT"),
        "narrow_entry_fill_time_error": _timing_stats(df, "narrow", "ENTRY"),
        "wide_entry_fill_time_error": _timing_stats(df, "wide", "ENTRY"),
        "narrow_exit_fill_time_error": _timing_stats(df, "narrow", "EXIT"),
        "wide_exit_fill_time_error": _timing_stats(df, "wide", "EXIT"),
        "same_realization_only": True,
        "independent_validation": False,
        "counterfactual_strategy_pnl_valid": False,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "interpretation_guardrail": (
            "This conditions on the actual live CREATE/CANCEL tape, which is endogenous to real fills. "
            "Use it to validate fill mechanics and isolate order-path divergence, not as a self-consistent strategy backtest."
        ),
    }

    # Window/asset detail for ENTRY orders, where the live-vs-shadow quantity gap is central.
    detail_rows = []
    for keys, g in df[df["role"] == "ENTRY"].groupby(["close_time", "series"], dropna=False):
        close, series = keys
        detail_rows.append({
            "close_time": close,
            "series": series,
            "orders": int(len(g)),
            "known_queue_orders": int(g["displayed_queue_ahead"].notna().sum()),
            "actual_fill_qty": float(g["actual_fill_qty"].sum()),
            "narrow_predicted_fill_qty": float(g["narrow_predicted_fill_qty"].fillna(0.0).sum()),
            "wide_predicted_fill_qty": float(g["wide_predicted_fill_qty"].fillna(0.0).sum()),
            "narrow_tp": int(((g["actual_fill_qty"] > EPS) & (g["narrow_predicted_fill_qty"].fillna(0.0) > EPS)).sum()),
            "narrow_fn": int(((g["actual_fill_qty"] > EPS) & ~(g["narrow_predicted_fill_qty"].fillna(0.0) > EPS)).sum()),
            "narrow_fp": int((~(g["actual_fill_qty"] > EPS) & (g["narrow_predicted_fill_qty"].fillna(0.0) > EPS)).sum()),
            "wide_tp": int(((g["actual_fill_qty"] > EPS) & (g["wide_predicted_fill_qty"].fillna(0.0) > EPS)).sum()),
            "wide_fn": int(((g["actual_fill_qty"] > EPS) & ~(g["wide_predicted_fill_qty"].fillna(0.0) > EPS)).sum()),
            "wide_fp": int((~(g["actual_fill_qty"] > EPS) & (g["wide_predicted_fill_qty"].fillna(0.0) > EPS)).sum()),
        })
    detail_df = pd.DataFrame(detail_rows)

    out = _new_output(source.name)
    df.to_csv(out / "actual_order_fill_replay_orders.csv", index=False)
    detail_df.to_csv(out / "actual_order_fill_replay_entry_by_window_asset.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("=" * 128)
        print("Q5 ACTUAL LIVE ORDER-TAPE FILL REPLAY V5 — SAME REALIZATION / READ ONLY")
        print("=" * 128)
        print("Source:", source)
        print("Passive CREATE orders:", len(df), f"(ENTRY={summary['entry_create_orders']}, EXIT={summary['exit_create_orders']})")
        print("Displayed-queue coverage:", f"{summary['displayed_queue_coverage_pct']:.2f}%")
        print("Cancel send/ack known orders:", summary["cancel_send_known_orders"], "/", summary["cancel_ack_known_orders"])
        print("Public trade rows:", len(trades))
        print()
        print("ENTRY QUANTITY — THE CENTRAL COMPARISON")
        print("  actual live ENTRY qty:            ", f"{entry_actual_all:.4f}")
        print("  frozen strategy shadow ENTRY qty: ", f"{_f(baseline.get('entry_qty')):.4f}")
        print("  actual-order replay NARROW qty:   ", f"{summary['narrow_entry']['predicted_fill_qty']:.4f}")
        print("  actual-order replay WIDE qty:     ", f"{summary['wide_entry']['predicted_fill_qty']:.4f}")
        print()
        for label, key in (("NARROW  ACK->CANCEL_SEND", "narrow_entry"), ("WIDE    SEND->CANCEL_ACK", "wide_entry")):
            z = summary[key]
            print(label)
            print("  known orders:", z["known_orders"])
            print("  actual/predicted filled orders:", z["actual_filled_orders"], "/", z["predicted_filled_orders"])
            print("  TP / FN / FP / TN:", z["tp"], "/", z["fn"], "/", z["fp"], "/", z["tn"])
            print("  actual fill qty on TP:", f"{z.get('actual_qty_on_tp', 0.0):.4f}")
            print("  actual fill qty on FN:", f"{z.get('actual_qty_on_fn', 0.0):.4f}")
            print("  predicted qty on FP:", f"{z.get('predicted_qty_on_fp', 0.0):.4f}")
            print("  first-fill timing error ms:", summary[("narrow" if key.startswith("narrow") else "wide") + "_entry_fill_time_error"])
            print()

        print("EXIT CONDITIONAL FILL REPRODUCTION")
        for label, key in (("NARROW", "narrow_exit"), ("WIDE", "wide_exit")):
            z = summary[key]
            print(
                f"  {label:<6} actual_qty={z['actual_fill_qty']:.4f} predicted_qty={z['predicted_fill_qty']:.4f} "
                f"TP/FN/FP/TN={z['tp']}/{z['fn']}/{z['fp']}/{z['tn']}"
            )
        print()
        print("ENTRY BY WINDOW / ASSET")
        if not detail_df.empty:
            print(detail_df.to_string(index=False))
        print()
        print("Interpretation:")
        print("  - If actual-order replay ENTRY qty moves from the frozen shadow's ~221 toward live ~629,")
        print("    the dominant mismatch is the strategy/order path, not the frozen public-trade fill rule.")
        print("  - Remaining false negatives quantify fills requiring queue advancement beyond displayed-L1 trade burn.")
        print("  - False positives expose places where the trade-only queue model predicts a fill that live did not receive.")
        print("  - NARROW/WIDE agreement means CREATE/CANCEL effective-time uncertainty is not driving the result.")
        print("  - Do NOT interpret this as counterfactual strategy PnL; the actual order tape is endogenous to real fills.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 128)

    return {
        "summary": summary,
        "orders": df,
        "entry_by_window_asset": detail_df,
        "output_dir": out,
    }


__all__ = ["run_q5_actual_order_fill_replay", "VERSION"]
