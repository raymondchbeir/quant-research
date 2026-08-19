from __future__ import annotations

"""Q5 cross-feed exchange-time vs receipt-time causality forensic.

Purpose
-------
V8 showed that most losing same-quote TP live fills occurred while the frozen
shadow was still resting at the same side/price at the live fill exchange
timestamp.  This module follows only those V8 STAYED_SAME_QUOTE_UNTIL_LIVE_FILL
orders from the live fill timestamp through receipt of the exact public trade
that caused the fill.

The central question is whether asynchronous book/trade receipt ordering lets the
receipt-time shadow cancel/reprice *before it receives a trade that had already
happened at the exchange*.

For each target we reconcile the first real live ENTRY fill to the public trade
tape by exact trade_id when possible and record:
- live fill exchange timestamp;
- causal public trade exchange timestamp;
- causal public trade local receipt timestamp;
- first shadow departure after the live fill but before causal trade receipt;
- departure event receipt timestamp and exchange timestamp when available;
- shadow state immediately before/after the causal trade receipt;
- queue ahead immediately before processing the causal trade.

Primary causality classes
-------------------------
B_RECEIPT_INVERSION_TRADE_EXCHANGE_FIRST_BOOK_RECEIPT_FIRST
    A book event was received locally before the causal trade receipt, causing
    the shadow to leave, while exchange timestamps show the causal trade itself
    happened before that book event. This is the strongest evidence of
    cross-feed receipt-order hindsight in the shadow.
C_BOOK_EXCHANGE_FIRST
    The invalidating book event also happened first at the exchange; this is not
    a cross-feed inversion.
BOOK_RECEIPT_FIRST_BOOK_EXCHANGE_UNKNOWN
    Receipt ordering is inverted but the book exchange timestamp is unavailable.
OTHER_TRADE_CHANGED_SHADOW_BEFORE_CAUSAL_RECEIPT
    Another public trade changed shadow inventory before the exact causal trade
    receipt; this is fill-path divergence, not a book cancellation issue.
D_CAUSAL_TRADE_RECEIPT_FILLED_SHADOW
    The shadow remained on the quote until the exact causal trade was processed
    and then filled.
A_CAUSAL_TRADE_RECEIPT_NO_SHADOW_FILL_QUEUE
    The shadow remained on the quote through the exact causal trade receipt but
    its queue model did not fill. This points back to queue/fill modeling rather
    than cross-feed event ordering.

Scientific guardrails
---------------------
- SAME-REALIZATION execution forensic only; NOT independent validation.
- NO exchange/API calls and NO orders.
- Source session is read-only.
- Candidate-C, queue rules, assets, and thresholds are unchanged.
- Exchange timestamps are used only for causality diagnostics; the frozen shadow
  itself is replayed exactly in local receipt order, as before.
- Writes only under results/kalshi_q5_cross_feed_causality_forensic_v9/.
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
from . import mm_cycle_q5_tp_fn_economics_v6 as V6
from . import mm_cycle_q5_live_shadow_order_path_forensic_v7 as V7
from . import mm_cycle_q5_shadow_departure_live_cancel_forensic_v8 as V8

VERSION = "MM_CYCLE_Q5_CROSS_FEED_CAUSALITY_FORENSIC_V9"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_cross_feed_causality_forensic_v9"
EPS = 1e-9
TIME_MATCH_TOL_S = 0.003


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


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _find_v8_output(source: Path):
    root = Path(V8.OUTPUT_ROOT)
    candidates = []
    if root.exists():
        for d in root.glob(source.name + "*"):
            sp = d / "summary.json"
            dp = d / "same_quote_tp_departure_cancel_detail.csv"
            if not (sp.exists() and dp.exists()):
                continue
            obj = OOS._read_json(sp, {}) or {}
            try:
                same = Path(obj.get("source_session", "")).resolve() == source.resolve()
            except Exception:
                same = False
            if same:
                candidates.append((sp.stat().st_mtime, d.resolve(), obj, dp))
    if not candidates:
        res = V8.run_q5_shadow_departure_live_cancel_forensic(source, show=False)
        return Path(res["output_dir"]).resolve(), res["summary"], res["detail"].copy()
    _, d, summary, dp = max(candidates, key=lambda z: z[0])
    return d, summary, pd.read_csv(dp)


def _load_first_entry_fills(source: Path, selected_tickers: set[str]):
    rows, _time_sources, _fee_keys = V6._load_all_strategy_fills(source)
    by_oid = defaultdict(list)
    for r in rows:
        if (
            str(r.get("role") or "").upper() == "ENTRY"
            and str(r.get("ticker") or "") in selected_tickers
        ):
            by_oid[str(r.get("order_id") or "")].append(r)
    out = {}
    for oid, xs in by_oid.items():
        xs.sort(key=lambda z: float(z["fill_s"]))
        out[oid] = xs[0]
    return out


def _exact_causal_trade_map(source: Path, selected_tickers: set[str], first_fills: dict):
    trades, by_trade_id, by_ticker = V1._load_public_trades(
        source / "raw_capture", selected_tickers
    )
    out = {}
    method_counts = Counter()

    for oid, f in first_fills.items():
        ticker = str(f.get("ticker") or "")
        tid = str(f.get("trade_id") or "")
        fill_s = float(f["fill_s"])
        side = str(f.get("side") or "").upper()
        price = _f(f.get("price"))

        exact = []
        if tid:
            for i in by_trade_id.get(tid, []):
                tr = trades[i]
                if str(tr.get("ticker") or "") != ticker:
                    continue
                ok, kind = V1._trade_compatible(side, price, tr)
                if not ok:
                    continue
                tx = _f(tr.get("exchange_s"))
                distance = abs(tx - fill_s) if np.isfinite(tx) else abs(float(tr["receipt_s"]) - fill_s)
                exact.append((distance, i, kind))

        if exact:
            exact.sort(key=lambda z: (z[0], z[1]))
            _, i, kind = exact[0]
            tr = trades[i]
            method = "EXACT_TRADE_ID"
            method_counts[method] += 1
            out[oid] = {
                "causal_match_method": method,
                "causal_trade_candidate_count": len(exact),
                "causal_trade_id": str(tr.get("trade_id") or ""),
                "causal_trade_index": int(i),
                "causal_trade_kind": kind,
                "causal_trade_exchange_s": _f(tr.get("exchange_s")),
                "causal_trade_receipt_s": float(tr["receipt_s"]),
                "causal_trade_price": _f(tr.get("price")),
                "causal_trade_qty": _f(tr.get("qty")),
                "causal_trade_taker_book_side": str(tr.get("taker_book_side") or ""),
            }
            continue

        # Defensive fallback. Exact-ID reconciliation is expected for this Q5
        # realization, but do not silently drop a row if a schema changes.
        ff = {
            "trade_id": tid,
            "ticker": ticker,
            "side": side,
            "price": price,
            "fill_time_s": fill_s,
        }
        m = V1._match_fill(ff, trades, by_trade_id, by_ticker)
        i = m.get("trade_index")
        if i is not None:
            tr = trades[i]
            method = "FALLBACK_V1_MATCH"
            method_counts[method] += 1
            out[oid] = {
                "causal_match_method": method,
                "causal_trade_candidate_count": int(m.get("candidate_count") or 0),
                "causal_trade_id": str(tr.get("trade_id") or ""),
                "causal_trade_index": int(i),
                "causal_trade_kind": m.get("match_kind"),
                "causal_trade_exchange_s": _f(tr.get("exchange_s")),
                "causal_trade_receipt_s": float(tr["receipt_s"]),
                "causal_trade_price": _f(tr.get("price")),
                "causal_trade_qty": _f(tr.get("qty")),
                "causal_trade_taker_book_side": str(tr.get("taker_book_side") or ""),
            }
        else:
            method_counts["NO_MATCH"] += 1

    return out, dict(method_counts)


class QuietPathShadow(V7.InstrumentedPathShadow):
    """V7 instrumented shadow without console/event-file chatter."""

    def emit(self, event, ticker=None, **detail):
        # Emission is observational only in FrozenCycleShadow; fill/inventory
        # mechanics are updated before emit() is called.
        return None


def _same_target_state(shadow, target):
    return V8._same_target_state(shadow, target)


def _exchange_s(row):
    return OOS._ts((row or {}).get("exchange_time"))


def _weighted(g: pd.DataFrame, col: str, weight="actual_fill_qty"):
    if col not in g or weight not in g:
        return np.nan
    x = pd.to_numeric(g[col], errors="coerce")
    w = pd.to_numeric(g[weight], errors="coerce")
    m = x.notna() & w.notna() & (w > EPS)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def _classify(row):
    if not bool(row.get("same_quote_at_live_fill_boundary")):
        return "START_STATE_NOT_REPRODUCED"

    dep_s = _f(row.get("postfill_departure_receipt_s"))
    trade_receipt = _f(row.get("causal_trade_receipt_s"))
    trade_exchange = _f(row.get("causal_trade_exchange_s"))
    dep_exchange = _f(row.get("postfill_departure_exchange_s"))
    dep_typ = str(row.get("postfill_departure_event_type") or "")

    if np.isfinite(dep_s) and np.isfinite(trade_receipt) and dep_s < trade_receipt - EPS:
        if dep_typ == "BOOK":
            if np.isfinite(dep_exchange) and np.isfinite(trade_exchange):
                if trade_exchange < dep_exchange - EPS:
                    return "B_RECEIPT_INVERSION_TRADE_EXCHANGE_FIRST_BOOK_RECEIPT_FIRST"
                if dep_exchange < trade_exchange - EPS:
                    return "C_BOOK_EXCHANGE_FIRST"
                return "BOOK_TRADE_EXCHANGE_TIE_OR_AMBIGUOUS"
            return "BOOK_RECEIPT_FIRST_BOOK_EXCHANGE_UNKNOWN"
        if dep_typ == "TRADE":
            return "OTHER_TRADE_CHANGED_SHADOW_BEFORE_CAUSAL_RECEIPT"
        return "OTHER_SHADOW_DEPARTURE_BEFORE_CAUSAL_RECEIPT"

    if bool(row.get("causal_trade_seen")):
        if bool(row.get("shadow_filled_on_causal_trade")):
            return "D_CAUSAL_TRADE_RECEIPT_FILLED_SHADOW"
        if bool(row.get("same_quote_before_causal_trade")) and bool(row.get("same_quote_after_causal_trade")):
            return "A_CAUSAL_TRADE_RECEIPT_NO_SHADOW_FILL_QUEUE"
        if bool(row.get("same_quote_before_causal_trade")):
            return "CAUSAL_TRADE_CHANGED_SHADOW_OTHER"
        return "SHADOW_NOT_ON_QUOTE_AT_CAUSAL_RECEIPT_UNCLASSIFIED"

    return "CAUSAL_TRADE_NOT_SEEN_IN_REPLAY"


def _group_summary(df: pd.DataFrame):
    rows = []
    for cls, g in df.groupby("causality_class", sort=False, dropna=False):
        qty = float(pd.to_numeric(g["actual_fill_qty"], errors="coerce").fillna(0.0).sum())
        pnl = float(pd.to_numeric(g.get("econ_total_realized_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum())
        rows.append({
            "causality_class": cls,
            "orders": int(len(g)),
            "fill_qty": qty,
            "realized_gross_pnl": pnl,
            "gross_pnl_per_contract_c": 100.0 * pnl / qty if qty > EPS else np.nan,
            "trade_receipt_delay_ms_weighted": _weighted(g, "causal_trade_receipt_delay_ms"),
            "departure_receipt_before_trade_ms_weighted": _weighted(g, "departure_receipt_before_causal_trade_receipt_ms"),
            "trade_exchange_before_departure_exchange_ms_weighted": _weighted(g, "trade_exchange_before_departure_exchange_ms"),
            "shadow_queue_before_causal_weighted": _weighted(g, "shadow_queue_ahead_before_causal_trade"),
            "markout_250ms_c": _weighted(g, "markout_250ms_c"),
            "markout_1s_c": _weighted(g, "markout_1s_c"),
            "markout_5s_c": _weighted(g, "markout_5s_c"),
        })
    return pd.DataFrame(rows).sort_values("realized_gross_pnl") if rows else pd.DataFrame()


def run_q5_cross_feed_causality_forensic(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"
    required = [
        source / "process_config.json",
        source / "fee_preflight.json",
        source / "fills.jsonl",
        source / "events.jsonl",
        raw / "book_top3_events.jsonl",
        raw / "trades_event_time.jsonl",
        raw / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts: " + " | ".join(missing))

    cfg = OOS._read_json(source / "process_config.json", {}) or {}
    if str(cfg.get("mode") or "") != "LIVE_Q5_1H":
        raise RuntimeError("Expected completed LIVE_Q5_1H source session.")
    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored fee preflight was not PASS.")

    windows = BASE._live_windows(source)
    meta_rows, meta_by_ticker = BASE._metadata(raw)
    selected_tickers = {
        t for t, r in meta_by_ticker.items()
        if str(r.get("close_time") or "") in set(windows)
    }
    if not selected_tickers:
        raise RuntimeError("No selected Q5 tickers.")

    v8_dir, v8_summary, v8_detail = _find_v8_output(source)
    stayed = v8_detail[
        (v8_detail["shadow_departure_class"].astype(str) == "STAYED_SAME_QUOTE_UNTIL_LIVE_FILL")
        & (pd.to_numeric(v8_detail["actual_fill_qty"], errors="coerce").fillna(0.0) > EPS)
    ].copy()
    if stayed.empty:
        raise RuntimeError("No V8 STAYED_SAME_QUOTE_UNTIL_LIVE_FILL targets found.")

    first_fills = _load_first_entry_fills(source, selected_tickers)
    causal_map, match_counts = _exact_causal_trade_map(source, selected_tickers, first_fills)

    targets = []
    for _, r in stayed.iterrows():
        oid = str(r["order_id"])
        ff = first_fills.get(oid)
        cm = causal_map.get(oid)
        if ff is None or cm is None:
            continue
        rec = r.to_dict()
        rec["order_id"] = oid
        rec["actual_first_fill_s"] = float(ff["fill_s"])
        rec["actual_first_fill_trade_id"] = str(ff.get("trade_id") or "")
        rec.update(cm)
        tx = _f(rec.get("causal_trade_exchange_s"))
        trc = _f(rec.get("causal_trade_receipt_s"))
        rec["live_fill_minus_causal_trade_exchange_ms"] = (
            1000.0 * (float(rec["actual_first_fill_s"]) - tx)
            if np.isfinite(tx) else np.nan
        )
        rec["causal_trade_receipt_delay_ms"] = (
            1000.0 * (trc - tx)
            if np.isfinite(trc) and np.isfinite(tx) else np.nan
        )
        targets.append(rec)

    if not targets:
        raise RuntimeError("No stayed-same-quote targets had a causal public trade match.")

    targets.sort(key=lambda z: float(z["actual_first_fill_s"]))
    out = _new_output(source.name)
    workspace = out / "shadow_trace_workspace"
    workspace.mkdir(parents=True, exist_ok=False)

    shadow = QuietPathShadow(workspace, fee)
    for m in meta_rows:
        ticker = str(m.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        shadow.meta[ticker] = m
        shadow.series_by_ticker[ticker] = str(m.get("series_ticker") or "")
        shadow.close_by_ticker[ticker] = str(m.get("close_time") or "")

    book_it = iter(_iter_jsonl(raw / "book_top3_events.jsonl") or [])
    trade_it = iter(_iter_jsonl(raw / "trades_event_time.jsonl") or [])
    b = BASE._next_selected(book_it, selected_tickers)
    tr = BASE._next_selected(trade_it, selected_tickers)
    if b is None and tr is None:
        raise RuntimeError("No selected raw events found.")

    first_ts = min(x[0] for x in (b, tr) if x is not None)
    shadow.started_at = pd.Timestamp(first_ts, unit="s", tz="UTC")
    shadow.thread_alive = True

    pending = list(targets)
    pidx = 0
    active = {}
    finished = {}
    raw_events = 0

    def next_raw_time():
        xs = [x[0] for x in (b, tr) if x is not None]
        return min(xs) if xs else np.inf

    def next_boundary_time():
        xs = []
        if pidx < len(pending):
            xs.append(float(pending[pidx]["actual_first_fill_s"]))
        for tgt in active.values():
            xs.append(float(tgt["causal_trade_receipt_s"]))
        return min(xs) if xs else np.inf

    def activate(tgt):
        z = dict(tgt)
        z["same_quote_at_live_fill_boundary"] = _same_target_state(shadow, z)
        q = shadow.quote.get(str(z["ticker"])) or {}
        z["shadow_queue_at_live_fill_boundary"] = _f(q.get("queue_ahead"))
        z["postfill_departure_receipt_s"] = np.nan
        z["postfill_departure_exchange_s"] = np.nan
        z["postfill_departure_event_type"] = None
        z["postfill_departure_transition_reason"] = None
        z["causal_trade_seen"] = False
        z["same_quote_before_causal_trade"] = False
        z["same_quote_after_causal_trade"] = False
        z["shadow_filled_on_causal_trade"] = False
        z["shadow_queue_ahead_before_causal_trade"] = np.nan
        z["shadow_remaining_qty_before_causal_trade"] = np.nan
        active[str(z["order_id"])] = z

    def finalize(oid):
        z = active.pop(oid)
        z["causality_class"] = _classify(z)
        dep_r = _f(z.get("postfill_departure_receipt_s"))
        dep_x = _f(z.get("postfill_departure_exchange_s"))
        trc = _f(z.get("causal_trade_receipt_s"))
        tx = _f(z.get("causal_trade_exchange_s"))
        z["departure_receipt_before_causal_trade_receipt_ms"] = (
            1000.0 * (trc - dep_r)
            if np.isfinite(trc) and np.isfinite(dep_r) else np.nan
        )
        z["trade_exchange_before_departure_exchange_ms"] = (
            1000.0 * (dep_x - tx)
            if np.isfinite(dep_x) and np.isfinite(tx) else np.nan
        )
        z["departure_book_receipt_delay_ms"] = (
            1000.0 * (dep_r - dep_x)
            if np.isfinite(dep_r) and np.isfinite(dep_x) else np.nan
        )
        finished[oid] = z

    while pidx < len(pending) or active:
        raw_t = next_raw_time()
        boundary_t = next_boundary_time()

        # Receipt-time raw event wins exact ties, matching the frozen shadow's
        # existing event-order convention.
        if raw_t <= boundary_t + EPS:
            if tr is None:
                choose_book = True
            elif b is None:
                choose_book = False
            elif b[0] < tr[0] - BASE.EPS:
                choose_book = True
            elif tr[0] < b[0] - BASE.EPS:
                choose_book = False
            else:
                choose_book = True

            if choose_book:
                t, row = b
                typ = "BOOK"
            else:
                t, row = tr
                typ = "TRADE"

            ticker = str(row.get("ticker") or "")
            affected = [
                oid for oid, tgt in active.items()
                if str(tgt["ticker"]) == ticker
                and float(tgt["actual_first_fill_s"]) <= float(t) + EPS
                and float(t) <= float(tgt["causal_trade_receipt_s"]) + TIME_MATCH_TOL_S
            ]

            before_same = {oid: _same_target_state(shadow, active[oid]) for oid in affected}
            before_q = {}
            for oid in affected:
                q = shadow.quote.get(ticker) or {}
                before_q[oid] = {
                    "queue_ahead": _f(q.get("queue_ahead")),
                    "remaining_qty": _f(q.get("remaining_qty")),
                }

            trans_before = len(shadow.path_transitions.get(ticker) or [])

            if choose_book:
                shadow._on_book(t, row)
                b = BASE._next_selected(book_it, selected_tickers)
            else:
                shadow._on_trade(t, row)
                tr = BASE._next_selected(trade_it, selected_tickers)
            shadow._update_drawdown()
            raw_events += 1

            new_trans = (shadow.path_transitions.get(ticker) or [])[trans_before:]
            cancel_trans = next((x for x in new_trans if x.get("action") == "CANCEL"), None)
            reason = (cancel_trans or {}).get("reason")

            for oid in affected:
                tgt = active[oid]
                is_causal = bool(
                    typ == "TRADE"
                    and str(row.get("trade_id") or "") == str(tgt.get("causal_trade_id") or "")
                    and abs(float(t) - float(tgt["causal_trade_receipt_s"])) <= TIME_MATCH_TOL_S
                )

                after_same = _same_target_state(shadow, tgt)

                if (
                    not np.isfinite(_f(tgt.get("postfill_departure_receipt_s")))
                    and before_same.get(oid)
                    and not after_same
                ):
                    tgt["postfill_departure_receipt_s"] = float(t)
                    tgt["postfill_departure_exchange_s"] = _exchange_s(row)
                    tgt["postfill_departure_event_type"] = typ
                    tgt["postfill_departure_transition_reason"] = reason
                    tgt["postfill_departure_raw_event_type"] = str(row.get("event_type") or "")
                    tgt["postfill_departure_raw_elapsed_s"] = _f(row.get("elapsed_s"))

                if is_causal:
                    tgt["causal_trade_seen"] = True
                    tgt["same_quote_before_causal_trade"] = bool(before_same.get(oid))
                    tgt["same_quote_after_causal_trade"] = bool(after_same)
                    tgt["shadow_queue_ahead_before_causal_trade"] = before_q[oid]["queue_ahead"]
                    tgt["shadow_remaining_qty_before_causal_trade"] = before_q[oid]["remaining_qty"]
                    tgt["shadow_filled_on_causal_trade"] = bool(
                        before_same.get(oid)
                        and not after_same
                        and (str(reason or "") == "FILL_POP" or abs(float(shadow.inventory.get(ticker, 0.0))) > OOS.EPS)
                    )

                    # The exact causal trade has now been processed in local
                    # receipt order; all evidence needed for this target exists.
                    finalize(oid)
            continue

        # Boundary before next raw receipt: activate a target at its real fill
        # exchange timestamp, or fail-close a target whose causal receipt boundary
        # was somehow reached without seeing the expected trade record.
        if pidx < len(pending) and float(pending[pidx]["actual_first_fill_s"]) <= boundary_t + EPS:
            activate(pending[pidx])
            pidx += 1
            continue

        due = [
            oid for oid, tgt in active.items()
            if float(tgt["causal_trade_receipt_s"]) <= boundary_t + EPS
        ]
        for oid in due:
            finalize(oid)

    shadow.thread_alive = False

    df = pd.DataFrame(list(finished.values()))
    if df.empty:
        raise RuntimeError("No completed cross-feed causality traces.")

    by_class = _group_summary(df)

    qty_total = float(pd.to_numeric(df["actual_fill_qty"], errors="coerce").fillna(0.0).sum())
    pnl_total = float(pd.to_numeric(df.get("econ_total_realized_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum())
    smoking = df[df["causality_class"] == "B_RECEIPT_INVERSION_TRADE_EXCHANGE_FIRST_BOOK_RECEIPT_FIRST"].copy()
    queue_miss = df[df["causality_class"] == "A_CAUSAL_TRADE_RECEIPT_NO_SHADOW_FILL_QUEUE"].copy()
    causal_fill = df[df["causality_class"] == "D_CAUSAL_TRADE_RECEIPT_FILLED_SHADOW"].copy()

    def qp(x):
        if x.empty:
            return 0.0, 0.0
        q = float(pd.to_numeric(x["actual_fill_qty"], errors="coerce").fillna(0.0).sum())
        p = float(pd.to_numeric(x.get("econ_total_realized_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum())
        return q, p

    smoke_q, smoke_p = qp(smoking)
    queue_q, queue_p = qp(queue_miss)
    fill_q, fill_p = qp(causal_fill)

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "v8_source": str(v8_dir),
        "v8_stayed_orders": int(len(stayed)),
        "targets_with_causal_trade_match": int(len(df)),
        "causal_match_methods": match_counts,
        "same_quote_state_reproduced_at_live_fill": int(df["same_quote_at_live_fill_boundary"].fillna(False).sum()),
        "target_fill_qty": qty_total,
        "target_realized_gross_pnl": pnl_total,
        "causality_counts": dict(Counter(df["causality_class"].astype(str))),
        "smoking_gun_receipt_inversion": {"orders": int(len(smoking)), "qty": smoke_q, "gross_pnl": smoke_p},
        "causal_trade_queue_miss": {"orders": int(len(queue_miss)), "qty": queue_q, "gross_pnl": queue_p},
        "causal_trade_filled_shadow": {"orders": int(len(causal_fill)), "qty": fill_q, "gross_pnl": fill_p},
        "live_fill_minus_causal_trade_exchange_ms_abs_max": float(
            pd.to_numeric(df["live_fill_minus_causal_trade_exchange_ms"], errors="coerce").abs().max()
        ),
        "causal_trade_receipt_delay_ms_median": float(
            pd.to_numeric(df["causal_trade_receipt_delay_ms"], errors="coerce").median()
        ),
        "same_realization_only": True,
        "independent_validation": False,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "interpretation_guardrail": (
            "Class B is evidence that receipt-time replay can observe a later-exchange book state before receiving an earlier-exchange causal trade. "
            "It diagnoses replay causality, not future profitability. Do not alter or promote the live strategy from this same-realization forensic alone."
        ),
    }

    detail = df.sort_values(
        "econ_total_realized_gross_pnl" if "econ_total_realized_gross_pnl" in df else "actual_fill_qty"
    )
    detail.to_csv(out / "cross_feed_causality_detail.csv", index=False)
    by_class.to_csv(out / "economics_by_causality_class.csv", index=False)
    smoking.sort_values(
        "econ_total_realized_gross_pnl" if "econ_total_realized_gross_pnl" in smoking else "actual_fill_qty"
    ).to_csv(out / "receipt_inversion_smoking_gun_rows.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("=" * 142)
        print("Q5 CROSS-FEED EXCHANGE-vs-RECEIPT CAUSALITY FORENSIC V9 — READ ONLY")
        print("=" * 142)
        print("Source:", source)
        print("V8 source:", v8_dir)
        print("V8 stayed-same targets:", len(stayed))
        print("Targets with causal trade match:", len(df))
        print("Causal match methods:", match_counts)
        print("Same quote reproduced at live-fill boundary:", summary["same_quote_state_reproduced_at_live_fill"], "/", len(df))
        print("Fill qty / gross PnL:", f"{qty_total:.4f}", "/", f"{pnl_total:+.4f}")
        print("Median causal trade exchange->receipt delay ms:", f"{summary['causal_trade_receipt_delay_ms_median']:.3f}")
        print("Max |live fill - causal trade exchange| ms:", f"{summary['live_fill_minus_causal_trade_exchange_ms_abs_max']:.3f}")
        print("Raw events replayed:", f"{raw_events:,}")
        print()
        print("ECONOMICS BY CAUSALITY CLASS")
        if not by_class.empty:
            print(by_class.to_string(index=False))
        print()
        print("CORE CAUSALITY DECOMPOSITION")
        print(f"  B receipt inversion smoking gun: orders={len(smoking)} qty={smoke_q:.4f} pnl={smoke_p:+.4f}")
        print(f"  A causal trade received, queue model did NOT fill: orders={len(queue_miss)} qty={queue_q:.4f} pnl={queue_p:+.4f}")
        print(f"  D causal trade received and shadow filled: orders={len(causal_fill)} qty={fill_q:.4f} pnl={fill_p:+.4f}")
        print()
        if len(smoking):
            print("RECEIPT-INVERSION SMOKING-GUN ROWS")
            cols = [
                "close_time", "series", "ticker", "order_id", "actual_fill_qty",
                "econ_total_realized_gross_pnl", "causal_trade_id",
                "live_fill_minus_causal_trade_exchange_ms",
                "causal_trade_receipt_delay_ms",
                "postfill_departure_transition_reason",
                "departure_receipt_before_causal_trade_receipt_ms",
                "trade_exchange_before_departure_exchange_ms",
                "departure_book_receipt_delay_ms",
                "markout_250ms_c", "markout_1s_c", "markout_5s_c",
            ]
            print(smoking[[c for c in cols if c in smoking.columns]].sort_values("econ_total_realized_gross_pnl").to_string(index=False))
            print()
        print("Interpretation:")
        print("  - B_RECEIPT_INVERSION... is the strongest evidence that the receipt-time shadow cancelled using a book event that was later at the exchange than the already-executed causal trade.")
        print("  - C_BOOK_EXCHANGE_FIRST means the book really changed first at the exchange; that is not a feed-order hindsight artifact.")
        print("  - A_CAUSAL_TRADE_RECEIPT_NO_SHADOW_FILL_QUEUE means the quote survived through causal trade receipt but the queue model still did not fill it.")
        print("  - D_CAUSAL_TRADE_RECEIPT_FILLED_SHADOW means this exact causal trade does fill the shadow once processed in receipt order.")
        print("  - Same realization only; this diagnoses simulator causality, not guaranteed alpha or future PnL.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 142)

    return {
        "summary": summary,
        "detail": detail,
        "by_class": by_class,
        "smoking_gun": smoking,
        "output_dir": out,
    }


__all__ = ["run_q5_cross_feed_causality_forensic", "VERSION"]
