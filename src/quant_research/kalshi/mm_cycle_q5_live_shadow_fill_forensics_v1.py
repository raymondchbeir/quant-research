from __future__ import annotations

"""Read-only live-vs-shadow fill-selection forensic for a completed Q5 session.

Purpose
-------
Explain the live-minus-shadow PnL gap by comparing actual live ENTRY fills with
frozen same-realization shadow ENTRY fills.

Primary buckets
---------------
LIVE_PLUS_SHADOW
    Actual live ENTRY quantity that can be matched to a shadow ENTRY fill on the
    same ticker/side/price within a tight time window.
LIVE_ONLY
    Actual live ENTRY quantity that the frozen shadow did not predict.
SHADOW_ONLY
    Residual shadow ENTRY quantity not matched to live.

For each live ENTRY fill this test computes signed 1s/5s/15s/30s mid markouts,
passive ENTRY->EXIT FIFO PnL attribution, first observed live queue position when
available, and Candidate-C validity at the actual CREATE request-send timestamp.

Scientific guardrails
---------------------
- SAME-REALIZATION diagnostic only; not independent validation.
- NO exchange/API calls.
- NO orders.
- Source session is read-only.
- Reads the already-generated frozen Q5 same-realization shadow output.
- Writes only under results/kalshi_q5_live_shadow_fill_forensics/.
"""

import bisect
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q5_same_realization_shadow_v1 as BASE

VERSION = "MM_CYCLE_Q5_LIVE_SHADOW_FILL_FORENSICS_V1"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_live_shadow_fill_forensics"
PRIMARY_MATCH_TOL_S = 2.0
MARKOUTS_S = (1.0, 5.0, 15.0, 30.0)
MARKOUT_MAX_AGE_S = 2.0
EPS = 1e-10


def _iter_jsonl(path):
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


def _f(x, default=np.nan):
    return OOS._f(x, default)


def _to_s(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        z = _f(x)
        if not np.isfinite(z):
            return np.nan
        if z > 1e14:
            return z / 1e9
        if z > 1e11:
            return z / 1e3
        return z
    try:
        t = pd.to_datetime(x, utc=True, errors="coerce")
        return float(t.timestamp()) if not pd.isna(t) else np.nan
    except Exception:
        return np.nan


def _live_fill_time(row):
    for k in (
        "created_time", "created_at", "fill_time", "trade_time",
        "timestamp", "ts", "ts_ms",
    ):
        if row.get(k) is not None:
            t = _to_s(row.get(k))
            if np.isfinite(t):
                return t, k
    return _to_s(row.get("observed_time")), "observed_time"


def _qty(row):
    for k in ("count_fp", "count", "qty", "quantity", "contracts"):
        z = _f(row.get(k))
        if np.isfinite(z) and z > EPS:
            return float(z)
    return np.nan


def _price(row):
    for k in ("yes_price_dollars", "yes_price", "price_dollars", "price"):
        z = _f(row.get(k))
        if np.isfinite(z):
            if 1.0 < z <= 100.0:
                z /= 100.0
            if 0.0 <= z <= 1.0:
                return float(z)
    return np.nan


def _side(row):
    s = str(row.get("strategy_side") or row.get("side") or "").strip().upper()
    if s in {"BID", "BUY", "YES"}:
        return "BID"
    if s in {"ASK", "SELL", "NO"}:
        return "ASK"
    return s


def _role(row):
    return str(row.get("role") or "").strip().upper()


def _unique_output(source_name):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / source_name
    if out.exists():
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_ROOT / f"{source_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _find_baseline_shadow(source):
    root = Path(BASE.OUTPUT_ROOT)
    candidates = []
    if root.exists():
        for d in root.glob(source.name + "*"):
            summary_path = d / "q5_same_realization_shadow_summary.json"
            fills_path = d / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1" / "shadow_fills.jsonl"
            if not (summary_path.exists() and fills_path.exists()):
                continue
            obj = OOS._read_json(summary_path, {}) or {}
            try:
                same = Path(obj.get("source_session", "")).resolve() == source.resolve()
            except Exception:
                same = False
            if same:
                candidates.append((summary_path.stat().st_mtime, d.resolve(), obj))
    if not candidates:
        raise FileNotFoundError(
            "No completed same-realization Q5 shadow output found. "
            "Run mm_cycle_q5_same_realization_shadow_v1 first."
        )
    _, d, summary = max(candidates, key=lambda z: z[0])
    return d, summary


def _metadata(raw):
    out = {}
    for r in _iter_jsonl(raw / "market_metadata.jsonl") or []:
        ticker = str(r.get("ticker") or "")
        if ticker:
            out[ticker] = r
    return out


def _load_live_fills(session):
    out = []
    time_sources = defaultdict(int)
    for r in _iter_jsonl(session / "fills.jsonl") or []:
        t, tsrc = _live_fill_time(r)
        q = _qty(r)
        p = _price(r)
        ticker = str(r.get("ticker") or "")
        side = _side(r)
        role = _role(r)
        oid = str(r.get("order_id") or "")
        if (
            ticker and oid and role in {"ENTRY", "EXIT"}
            and side in {"BID", "ASK"}
            and np.isfinite(t) and np.isfinite(q) and np.isfinite(p)
        ):
            time_sources[tsrc] += 1
            out.append({
                "ticker": ticker,
                "order_id": oid,
                "role": role,
                "side": side,
                "qty": float(q),
                "price": float(p),
                "fill_s": float(t),
                "fill_time_source": tsrc,
                "fill_id": str(r.get("fill_id") or r.get("trade_id") or ""),
            })
    out.sort(key=lambda z: (z["fill_s"], z["ticker"], z["role"]))
    return out, dict(time_sources)


def _load_shadow_entries(shadow_dir):
    path = shadow_dir / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1" / "shadow_fills.jsonl"
    out = []
    for r in _iter_jsonl(path) or []:
        if str(r.get("role") or "").upper() != "ENTRY":
            continue
        t = _f(r.get("fill_ts"))
        q = _f(r.get("qty"))
        p = _f(r.get("price"))
        ticker = str(r.get("ticker") or "")
        side = str(r.get("side") or "").upper()
        if (
            ticker and side in {"BID", "ASK"}
            and np.isfinite(t) and np.isfinite(q) and q > EPS and np.isfinite(p)
        ):
            out.append({
                "ticker": ticker,
                "side": side,
                "price": float(p),
                "qty": float(q),
                "remaining": float(q),
                "fill_s": float(t),
            })
    out.sort(key=lambda z: (z["fill_s"], z["ticker"]))
    return out


def _load_order_catalog(session):
    role_by_oid = {}
    send_by_oid = {}
    superseded_by_oid = {}
    for r in _iter_jsonl(session / "latency_events_v12.jsonl") or []:
        if str(r.get("event") or "") != "CREATE_SENT":
            continue
        oid = str(r.get("order_id") or "")
        if not oid:
            continue
        role_by_oid[oid] = str(r.get("role") or "").upper()
        z = _f(r.get("request_send_wall_ms"))
        send_by_oid[oid] = z / 1000.0 if np.isfinite(z) else np.nan
        superseded_by_oid[oid] = bool(r.get("superseded_at_send_detected"))

    out = {}
    for r in _iter_jsonl(session / "orders.jsonl") or []:
        if not str(r.get("action") or "").startswith("CREATE"):
            continue
        response = r.get("response") or {}
        payload = r.get("payload") or {}
        oid = str(response.get("order_id") or "")
        if not oid:
            continue
        z = _f((r.get("timing") or {}).get("request_send_wall_ms"))
        send_s = z / 1000.0 if np.isfinite(z) else send_by_oid.get(oid, np.nan)
        out[oid] = {
            "ticker": str(r.get("ticker") or payload.get("ticker") or ""),
            "role": role_by_oid.get(oid, ""),
            "side": str(payload.get("side") or "").upper(),
            "price": _f(payload.get("price")),
            "send_s": send_s,
            "superseded_at_send_detected": superseded_by_oid.get(oid, False),
        }
    return out


def _load_first_queue(session, order_catalog):
    out = {}
    for r in _iter_jsonl(session / "queue_positions.jsonl") or []:
        oid = str(r.get("order_id") or "")
        if oid not in order_catalog or oid in out:
            continue
        q = _f(r.get("queue_position"))
        t = _to_s(r.get("time"))
        if not (np.isfinite(q) and np.isfinite(t)):
            continue
        disp = _f(r.get("displayed_l1_ahead_at_join", r.get("displayed_l1_ahead")))
        send_s = _f(order_catalog[oid].get("send_s"))
        out[oid] = {
            "first_queue_position": float(q),
            "displayed_l1_ahead": float(disp) if np.isfinite(disp) else np.nan,
            "queue_minus_displayed": float(q - disp) if np.isfinite(disp) else np.nan,
            "send_to_queue_ms": 1000.0 * (t - send_s) if np.isfinite(send_s) else np.nan,
        }
    return out


def _load_book_index(raw, selected):
    by_ticker = defaultdict(list)
    for r in _iter_jsonl(raw / "book_top3_events.jsonl") or []:
        ticker = str(r.get("ticker") or "")
        if ticker not in selected:
            continue
        t = OOS._ts(r.get("receipt_time"))
        if not np.isfinite(t):
            continue
        cur = OOS._top_state(r)
        if cur is None:
            continue
        by_ticker[ticker].append((float(t), r, cur))
    times = {}
    for ticker, rows in by_ticker.items():
        rows.sort(key=lambda z: z[0])
        times[ticker] = [z[0] for z in rows]
    return dict(by_ticker), times


def _book_before(ticker, t, books, times):
    arr = times.get(ticker) or []
    if not arr:
        return None
    i = bisect.bisect_right(arr, float(t)) - 1
    return books[ticker][i] if i >= 0 else None


def _book_after(ticker, t, books, times):
    arr = times.get(ticker) or []
    if not arr:
        return None
    i = bisect.bisect_left(arr, float(t))
    if i >= len(arr):
        return None
    rec = books[ticker][i]
    return rec if rec[0] - float(t) <= MARKOUT_MAX_AGE_S + EPS else None


def _candidate_at_send(order, books, times):
    ticker = str(order.get("ticker") or "")
    send = _f(order.get("send_s"))
    rec = _book_before(ticker, send, books, times) if np.isfinite(send) else None
    if rec is None:
        return {"valid_at_send": False, "reason": "NO_RAW_BOOK_AT_OR_BEFORE_SEND"}
    t, row, cur = rec
    elapsed = _f(row.get("elapsed_s"))
    desired = OOS._entry_side(cur)
    side = str(order.get("side") or "").upper()
    px = _f(order.get("price"))
    wanted_px = cur["bid"] if desired == "BID" else cur["ask"] if desired == "ASK" else np.nan
    if not np.isfinite(elapsed) or not (0.0 <= elapsed < 300.0):
        ok, reason = False, "OUTSIDE_M0_M5"
    elif desired is None:
        ok, reason = False, "ENTRY_FILTER_NONE"
    elif desired != side:
        ok, reason = False, "SIDE_MISMATCH"
    elif not np.isfinite(px) or abs(float(px) - float(wanted_px)) > 1e-9:
        ok, reason = False, "PRICE_MISMATCH"
    else:
        ok, reason = True, None
    return {
        "valid_at_send": bool(ok),
        "reason": reason,
        "raw_receipt_s": t,
        "raw_age_at_send_ms": 1000.0 * (send - t),
        "raw_elapsed_s": elapsed,
        "candidate_side": desired,
        "candidate_price": wanted_px,
        "spread_c": cur.get("spread_c"),
    }


def _match_live_to_shadow(live_entries, shadow_entries, tol_s):
    groups = defaultdict(list)
    for i, s in enumerate(shadow_entries):
        groups[(s["ticker"], s["side"], round(s["price"], 4))].append(i)
    rows = []
    for lf in live_entries:
        key = (lf["ticker"], lf["side"], round(lf["price"], 4))
        cands = []
        for i in groups.get(key, []):
            s = shadow_entries[i]
            if s["remaining"] <= EPS:
                continue
            dt = abs(s["fill_s"] - lf["fill_s"])
            if dt <= tol_s + EPS:
                cands.append((dt, i))
        rem = float(lf["qty"])
        matched = 0.0
        weighted_dt = 0.0
        for dt, i in sorted(cands):
            if rem <= EPS:
                break
            s = shadow_entries[i]
            q = min(rem, s["remaining"])
            s["remaining"] -= q
            rem -= q
            matched += q
            weighted_dt += q * dt
        rows.append({
            **lf,
            "shadow_matched_qty": matched,
            "live_only_qty": max(0.0, rem),
            "matched_dt_ms": 1000.0 * weighted_dt / matched if matched > EPS else np.nan,
        })
    shadow_only = sum(max(0.0, s["remaining"]) for s in shadow_entries)
    return rows, shadow_only


def _attach_markouts(rows, books, times):
    for r in rows:
        sign = 1.0 if r["side"] == "BID" else -1.0
        for h in MARKOUTS_S:
            rec = _book_after(r["ticker"], r["fill_s"] + h, books, times)
            key = f"markout_{int(h)}s_c"
            r[key] = np.nan if rec is None else sign * (float(rec[2]["mid"]) - float(r["price"])) * 100.0


def _passive_fifo_attribution(all_live_fills, entry_rows):
    components = defaultdict(list)
    for i, r in enumerate(entry_rows):
        if r["shadow_matched_qty"] > EPS:
            components[(r["order_id"], r["fill_s"], r["price"])].append(
                {"row_index": i, "bucket": "LIVE_PLUS_SHADOW", "qty": r["shadow_matched_qty"]}
            )
        if r["live_only_qty"] > EPS:
            components[(r["order_id"], r["fill_s"], r["price"])].append(
                {"row_index": i, "bucket": "LIVE_ONLY", "qty": r["live_only_qty"]}
            )

    open_lots = defaultdict(deque)
    for f in all_live_fills:
        ticker = f["ticker"]
        if f["role"] == "ENTRY":
            for c in components.get((f["order_id"], f["fill_s"], f["price"]), []):
                open_lots[ticker].append({
                    "row_index": c["row_index"],
                    "bucket": c["bucket"],
                    "qty": float(c["qty"]),
                    "entry_side": f["side"],
                    "entry_price": float(f["price"]),
                })
            continue
        if f["role"] != "EXIT":
            continue
        rem = float(f["qty"])
        while rem > EPS and open_lots[ticker]:
            lot = open_lots[ticker][0]
            q = min(rem, lot["qty"])
            pnl = (
                (float(f["price"]) - lot["entry_price"]) * q
                if lot["entry_side"] == "BID"
                else (lot["entry_price"] - float(f["price"])) * q
            )
            rr = entry_rows[lot["row_index"]]
            rr["passive_realized_pnl"] = rr.get("passive_realized_pnl", 0.0) + pnl
            rr["passive_realized_qty"] = rr.get("passive_realized_qty", 0.0) + q
            rem -= q
            lot["qty"] -= q
            if lot["qty"] <= EPS:
                open_lots[ticker].popleft()

    for r in entry_rows:
        r.setdefault("passive_realized_pnl", 0.0)
        r.setdefault("passive_realized_qty", 0.0)

    residual = defaultdict(float)
    for lots in open_lots.values():
        for lot in lots:
            residual[lot["bucket"]] += float(lot["qty"])
    return dict(residual)


def _weighted_avg(df, value, weight):
    x = pd.to_numeric(df[value], errors="coerce")
    w = pd.to_numeric(df[weight], errors="coerce")
    m = x.notna() & w.notna() & (w > EPS)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def run_q5_live_shadow_fill_forensics(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"
    required = [
        source / "fills.jsonl",
        source / "orders.jsonl",
        source / "latency_events_v12.jsonl",
        source / "queue_positions.jsonl",
        source / "final_summary.json",
        raw / "book_top3_events.jsonl",
        raw / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts: " + " | ".join(missing))

    cfg = OOS._read_json(source / "process_config.json", {}) or {}
    if str(cfg.get("mode") or "") != "LIVE_Q5_1H":
        raise RuntimeError(f"Expected LIVE_Q5_1H source, got {cfg.get('mode')!r}")

    shadow_dir, shadow_summary = _find_baseline_shadow(source)
    shadow_entries = _load_shadow_entries(shadow_dir)
    all_live, time_sources = _load_live_fills(source)
    live_entries = [x for x in all_live if x["role"] == "ENTRY"]
    if not live_entries:
        raise RuntimeError("No parseable live ENTRY fills found.")

    meta = _metadata(raw)
    selected = {x["ticker"] for x in live_entries} | {x["ticker"] for x in shadow_entries}
    books, times = _load_book_index(raw, selected)
    order_catalog = _load_order_catalog(source)
    first_queue = _load_first_queue(source, order_catalog)

    rows, shadow_only_qty = _match_live_to_shadow(live_entries, shadow_entries, PRIMARY_MATCH_TOL_S)
    _attach_markouts(rows, books, times)

    for r in rows:
        order = order_catalog.get(r["order_id"]) or {}
        cand = _candidate_at_send(order, books, times) if order else {
            "valid_at_send": False,
            "reason": "CREATE_ORDER_NOT_FOUND",
        }
        r.update({f"send_{k}": v for k, v in cand.items()})
        r["superseded_at_send_detected"] = bool(order.get("superseded_at_send_detected", False))
        r.update(first_queue.get(r["order_id"]) or {})
        m = meta.get(r["ticker"]) or {}
        r["series"] = str(m.get("series_ticker") or "")
        r["close_time"] = str(m.get("close_time") or "")

    passive_residual = _passive_fifo_attribution(all_live, rows)
    df = pd.DataFrame(rows)

    parts = []
    for _, r in df.iterrows():
        for bucket, qcol in (("LIVE_PLUS_SHADOW", "shadow_matched_qty"), ("LIVE_ONLY", "live_only_qty")):
            q = _f(r[qcol], 0.0)
            if q <= EPS:
                continue
            d = r.to_dict()
            d["bucket"] = bucket
            d["bucket_qty"] = q
            fq = _f(r["qty"], 0.0)
            d["bucket_passive_realized_pnl"] = _f(r["passive_realized_pnl"], 0.0) * q / fq if fq > EPS else 0.0
            d["bucket_passive_realized_qty"] = _f(r["passive_realized_qty"], 0.0) * q / fq if fq > EPS else 0.0
            parts.append(d)
    bdf = pd.DataFrame(parts)

    bucket_rows = []
    for bucket, g in bdf.groupby("bucket", sort=False):
        rec = {
            "bucket": bucket,
            "entry_fill_rows": len(g),
            "entry_qty": float(g["bucket_qty"].sum()),
            "passive_realized_qty": float(g["bucket_passive_realized_qty"].sum()),
            "passive_realized_pnl": float(g["bucket_passive_realized_pnl"].sum()),
            "residual_unmatched_to_passive_exit_qty": float(passive_residual.get(bucket, 0.0)),
            "candidate_valid_at_send_qty": float(g.loc[g["send_valid_at_send"] == True, "bucket_qty"].sum()),
            "candidate_invalid_at_send_qty": float(g.loc[g["send_valid_at_send"] != True, "bucket_qty"].sum()),
            "superseded_at_send_qty": float(g.loc[g["superseded_at_send_detected"] == True, "bucket_qty"].sum()),
        }
        for h in MARKOUTS_S:
            rec[f"markout_{int(h)}s_c"] = _weighted_avg(g, f"markout_{int(h)}s_c", "bucket_qty")
        if "queue_minus_displayed" in g:
            rec["first_queue_minus_displayed"] = _weighted_avg(g, "queue_minus_displayed", "bucket_qty")
            rec["send_to_first_queue_ms"] = _weighted_avg(g, "send_to_queue_ms", "bucket_qty")
            rec["queue_observed_qty"] = float(g.loc[pd.to_numeric(g["first_queue_position"], errors="coerce").notna(), "bucket_qty"].sum())
        else:
            rec["first_queue_minus_displayed"] = np.nan
            rec["send_to_first_queue_ms"] = np.nan
            rec["queue_observed_qty"] = 0.0
        bucket_rows.append(rec)
    buckets = pd.DataFrame(bucket_rows)

    by_window = bdf.groupby(["close_time", "bucket"], as_index=False).agg(
        entry_qty=("bucket_qty", "sum"),
        passive_realized_qty=("bucket_passive_realized_qty", "sum"),
        passive_realized_pnl=("bucket_passive_realized_pnl", "sum"),
    )
    by_asset = bdf.groupby(["series", "bucket"], as_index=False).agg(
        entry_qty=("bucket_qty", "sum"),
        passive_realized_qty=("bucket_passive_realized_qty", "sum"),
        passive_realized_pnl=("bucket_passive_realized_pnl", "sum"),
    )

    sensitivity = []
    for tol in (0.25, 0.5, 1.0, 2.0, 5.0):
        shadow_copy = _load_shadow_entries(shadow_dir)
        rr, so = _match_live_to_shadow(live_entries, shadow_copy, tol)
        sensitivity.append({
            "tolerance_s": tol,
            "matched_live_qty": float(sum(x["shadow_matched_qty"] for x in rr)),
            "live_only_qty": float(sum(x["live_only_qty"] for x in rr)),
            "shadow_only_qty": float(so),
        })
    sensitivity_df = pd.DataFrame(sensitivity)

    live_final = OOS._read_json(source / "final_summary.json", {}) or {}
    live_pnl = _f(live_final.get("account_pnl_usd"))
    shadow_pnl = _f(shadow_summary.get("shadow_net_pnl"))
    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "baseline_shadow_dir": str(shadow_dir),
        "live_account_pnl": live_pnl,
        "baseline_shadow_pnl": shadow_pnl,
        "live_minus_shadow_pnl": live_pnl - shadow_pnl if np.isfinite(live_pnl) and np.isfinite(shadow_pnl) else np.nan,
        "live_entry_fill_rows": len(live_entries),
        "live_entry_qty": float(sum(x["qty"] for x in live_entries)),
        "shadow_entry_fill_rows": len(shadow_entries),
        "shadow_entry_qty": float(sum(x["qty"] for x in shadow_entries)),
        "primary_match_tolerance_s": PRIMARY_MATCH_TOL_S,
        "matched_live_entry_qty": float(df["shadow_matched_qty"].sum()),
        "live_only_entry_qty": float(df["live_only_qty"].sum()),
        "shadow_only_entry_qty": float(shadow_only_qty),
        "live_fill_time_sources": time_sources,
        "passive_fifo_residual_qty_by_bucket": passive_residual,
        "same_realization_only": True,
        "independent_validation": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "source_modified": False,
        "important_limitation": (
            "Passive realized PnL attribution uses logged ENTRY/EXIT fills only and excludes forced M5 IOC economics. "
            "Markouts are the cleaner toxic-fill diagnostic; residual quantities identify inventory that must be "
            "investigated through M5 liquidation logs next."
        ),
    }

    out = _unique_output(source.name)
    OOS._atomic_json(out / "fill_forensics_summary.json", summary)
    df.to_csv(out / "live_entry_fill_detail.csv", index=False)
    bdf.to_csv(out / "live_entry_bucket_detail.csv", index=False)
    buckets.to_csv(out / "bucket_summary.csv", index=False)
    by_window.to_csv(out / "bucket_by_window.csv", index=False)
    by_asset.to_csv(out / "bucket_by_asset.csv", index=False)
    sensitivity_df.to_csv(out / "match_tolerance_sensitivity.csv", index=False)

    live_only = df[df["live_only_qty"] > EPS].copy()
    if not live_only.empty:
        live_only["sort_score"] = pd.to_numeric(live_only["markout_15s_c"], errors="coerce")
        live_only = live_only.sort_values("sort_score", na_position="last")
        live_only.head(50).to_csv(out / "worst_live_only_entries.csv", index=False)

    if show:
        print("=" * 124)
        print("Q5 LIVE-vs-SHADOW FILL-SELECTION FORENSIC — READ ONLY / NO API / NO ORDERS")
        print("=" * 124)
        print("Source:", source)
        print("Baseline shadow:", shadow_dir)
        print(f"Live PnL:         ${live_pnl:+.4f}")
        print(f"Shadow PnL:       ${shadow_pnl:+.4f}")
        print(f"Live-shadow gap:  ${live_pnl-shadow_pnl:+.4f}")
        print()
        print("ENTRY FILL QUANTITY CLASSIFICATION")
        print(f"  live ENTRY qty:       {summary['live_entry_qty']:.4f}")
        print(f"  shadow ENTRY qty:     {summary['shadow_entry_qty']:.4f}")
        print(f"  LIVE+SHADOW qty:      {summary['matched_live_entry_qty']:.4f}")
        print(f"  LIVE-ONLY qty:        {summary['live_only_entry_qty']:.4f}")
        print(f"  SHADOW-ONLY qty:      {summary['shadow_only_entry_qty']:.4f}")
        print("  live fill time source:", time_sources)
        print()
        print("MATCH TOLERANCE SENSITIVITY")
        print(sensitivity_df.to_string(index=False))
        print()
        print("BUCKET ECONOMICS / MARKOUTS")
        print(buckets.to_string(index=False))
        print()
        print("BY WINDOW")
        print(by_window.to_string(index=False))
        print()
        print("BY ASSET")
        print(by_asset.to_string(index=False))
        print()
        print("PASSIVE FIFO RESIDUAL QTY (not explained by passive EXIT fills)")
        print(passive_residual)
        print()
        if not live_only.empty:
            cols = [
                "ticker", "series", "close_time", "order_id", "side", "qty", "live_only_qty",
                "price", "markout_1s_c", "markout_5s_c", "markout_15s_c", "markout_30s_c",
                "passive_realized_pnl", "passive_realized_qty", "send_valid_at_send", "send_reason",
                "superseded_at_send_detected", "first_queue_position", "queue_minus_displayed", "send_to_queue_ms",
            ]
            cols = [c for c in cols if c in live_only.columns]
            print("WORST LIVE-ONLY ENTRY ROWS (sorted by 15s markout)")
            print(live_only[cols].head(30).to_string(index=False))
            print()
        print("Interpretation guide:")
        print("  - Strongly negative LIVE_ONLY markouts/PnL => actual fills absent from shadow are toxic.")
        print("  - Similar LIVE_ONLY and LIVE+SHADOW markouts => entry selection is not the main gap.")
        print("  - Large passive residual qty => inspect M5/exit mechanics next; this test does not invent IOC fills.")
        print("  - Filled Candidate-C invalid-at-send qty should be ~0 if the five superseded CREATEs never filled.")
        print()
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 124)

    return {
        "summary": summary,
        "bucket_summary": buckets,
        "by_window": by_window,
        "by_asset": by_asset,
        "detail": df,
        "output_dir": str(out),
    }


__all__ = ["run_q5_live_shadow_fill_forensics"]
