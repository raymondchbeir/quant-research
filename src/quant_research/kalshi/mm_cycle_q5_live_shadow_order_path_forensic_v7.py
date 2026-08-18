from __future__ import annotations

"""Q5 live-vs-frozen-shadow order-path forensic.

Purpose
-------
The prior same-realization diagnostics established that:
- the actual live V12.2 order tape produces far more ENTRY exposure than the
  frozen event-by-event shadow;
- once the frozen public-trade queue model is conditioned on the actual live
  CREATE/CANCEL tape, it reproduces most real fills;
- the queue-model false negatives were profitable in the observed Q5 sample,
  while the true-positive fills caused the realized damage.

This module therefore asks the next structural question:

    At the exact causal time of each actual live ENTRY CREATE send, what was the
    independently replayed frozen shadow doing on that same ticker?

The frozen shadow is replayed from the authoritative raw book/trade files, using
exactly the same event ordering as mm_cycle_q5_same_realization_shadow_v1.  Actual
CREATE timestamps are inserted as read-only observation points.  No live state is
fed into the shadow.

Primary path classes
--------------------
SHADOW_ALREADY_RESTING_SAME_QUOTE
    Frozen shadow already had an ENTRY quote at the same side/price.
SHADOW_DIFFERENT_PRICE
    Frozen shadow had an ENTRY quote on the same side but a different price.
SHADOW_DIFFERENT_SIDE
    Frozen shadow had an ENTRY quote on the opposite side.
SHADOW_IN_INVENTORY_CYCLE
    Frozen shadow was carrying inventory / quoting EXIT instead of being flat.
SHADOW_CANCELLED_RECENTLY
    Frozen shadow was flat with no quote and its most recent quote removal was
    within RECENT_CANCEL_S.  This is a descriptive diagnostic threshold, not a
    strategy parameter; exact cancel age is always retained.
SHADOW_NO_CANDIDATE
    Frozen shadow was flat with no active quote and no recent removal.

For same-quote cases we also compare the frozen shadow's current retained queue
position against the displayed L1 queue seen by the later live CREATE.  This is
important because a later live rejoin can inherit a materially different queue
state even when side and price are identical.

Scientific guardrails
---------------------
- SAME-REALIZATION execution forensic only; NOT independent validation.
- NO exchange/API calls and NO orders.
- Source session is read-only.
- Does not change Candidate-C, queue rules, assets, or thresholds.
- Does not claim that fixing path divergence guarantees profitability.  Four Q5
  windows are diagnostic execution evidence, not a new profitability validation.
- Writes only under results/kalshi_q5_live_shadow_order_path_forensic_v7/.
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
from . import mm_cycle_q5_public_trade_fill_reconciliation_v1 as V1
from . import mm_cycle_q5_actual_order_fill_replay_v5 as V5
from . import mm_cycle_q5_tp_fn_economics_v6 as V6

VERSION = "MM_CYCLE_Q5_LIVE_SHADOW_ORDER_PATH_FORENSIC_V7"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_live_shadow_order_path_forensic_v7"
EPS = 1e-9
RECENT_CANCEL_S = 1.0  # diagnostic label only; exact ages are retained


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


def _find_v5_output(source: Path):
    root = Path(V5.OUTPUT_ROOT)
    candidates = []
    if root.exists():
        for d in root.glob(source.name + "*"):
            sp = d / "summary.json"
            op = d / "actual_order_fill_replay_orders.csv"
            if not (sp.exists() and op.exists()):
                continue
            obj = OOS._read_json(sp, {}) or {}
            try:
                same = Path(obj.get("source_session", "")).resolve() == source.resolve()
            except Exception:
                same = False
            if same:
                candidates.append((sp.stat().st_mtime, d.resolve(), obj, op))
    if not candidates:
        res = V5.run_q5_actual_order_fill_replay(source, show=False)
        return Path(res["output_dir"]).resolve(), res["summary"], res["orders"].copy()
    _, d, summary, op = max(candidates, key=lambda z: z[0])
    return d, summary, pd.read_csv(op)


def _find_v6_output(source: Path):
    root = Path(V6.OUTPUT_ROOT)
    candidates = []
    if root.exists():
        for d in root.glob(source.name + "*"):
            sp = d / "summary.json"
            dp = d / "tp_fn_entry_fill_detail.csv"
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
        res = V6.run_q5_tp_fn_economics(source, show=False)
        return Path(res["output_dir"]).resolve(), res["summary"], res["detail"].copy()
    _, d, summary, dp = max(candidates, key=lambda z: z[0])
    return d, summary, pd.read_csv(dp)


def _quote_sig(q):
    if not isinstance(q, dict):
        return None
    return (
        str(q.get("role") or "").upper(),
        str(q.get("side") or "").upper(),
        round(_f(q.get("price")), 10) if np.isfinite(_f(q.get("price"))) else None,
        round(_f(q.get("qty")), 6) if np.isfinite(_f(q.get("qty"))) else None,
    )


def _transition_reason(old, desired):
    if old is None:
        return None
    if desired is None:
        return "DESIRED_NONE"
    if str(old.get("role") or "").upper() != str(desired.get("role") or "").upper():
        return "ROLE_CHANGE"
    if str(old.get("side") or "").upper() != str(desired.get("side") or "").upper():
        return "SIDE_CHANGE"
    if abs(_f(old.get("price")) - _f(desired.get("price"))) > EPS:
        return "PRICE_CHANGE"
    if abs(_f(old.get("qty")) - _f(desired.get("qty"))) > 0.005:
        return "QTY_CHANGE"
    return "OTHER_CHANGE"


class InstrumentedPathShadow(BASE.Q5FrozenCycleShadow):
    """Frozen Q5 shadow plus read-only state-transition instrumentation."""

    def __init__(self, session_dir, fee_preflight_result):
        super().__init__(session_dir, fee_preflight_result)
        self.path_transitions = defaultdict(list)
        self.latest_elapsed = {}
        self.latest_book_ts = {}
        self.raw_candidate_sig = {}
        self.raw_candidate_since = {}
        self.last_raw_candidate_change = {}

    def _record_transition(self, ticker, t, action, quote=None, reason=None):
        self.path_transitions[str(ticker)].append({
            "t": float(t),
            "action": str(action),
            "reason": reason,
            "quote": dict(quote or {}),
        })

    def _reconcile_quote(self, ticker, cur, elapsed, t):
        old = dict(self.quote.get(ticker) or {}) if self.quote.get(ticker) is not None else None
        desired = self._desired_quote(ticker, cur, elapsed)
        old_sig = _quote_sig(old)
        desired_sig = _quote_sig(desired)
        reason = _transition_reason(old, desired)
        super()._reconcile_quote(ticker, cur, elapsed, t)
        new = dict(self.quote.get(ticker) or {}) if self.quote.get(ticker) is not None else None
        new_sig = _quote_sig(new)

        if old is not None and old_sig != new_sig:
            self._record_transition(ticker, t, "CANCEL", old, reason)
        if new is not None and old_sig != new_sig:
            self._record_transition(ticker, t, "OPEN", new, reason)

    def _on_book(self, t, r):
        ticker = str(r.get("ticker") or "")
        cur = OOS._top_state(r)
        elapsed = _f(r.get("elapsed_s"))
        self.latest_elapsed[ticker] = elapsed
        self.latest_book_ts[ticker] = float(t)

        sig = None
        if cur is not None and np.isfinite(elapsed) and 0.0 <= elapsed < 300.0:
            side = OOS._entry_side(cur)
            if side is not None:
                px = float(cur["bid"] if side == "BID" else cur["ask"])
                sig = (str(side), round(px, 10))
        if self.raw_candidate_sig.get(ticker) != sig:
            self.last_raw_candidate_change[ticker] = float(t)
            self.raw_candidate_since[ticker] = float(t)
            self.raw_candidate_sig[ticker] = sig

        before = dict(self.quote.get(ticker) or {}) if self.quote.get(ticker) is not None else None
        before_sig = _quote_sig(before)
        super()._on_book(t, r)
        after = dict(self.quote.get(ticker) or {}) if self.quote.get(ticker) is not None else None
        after_sig = _quote_sig(after)

        # M5/invalid-book paths can remove a quote outside _reconcile_quote.
        if before is not None and after is None and before_sig == _quote_sig(before):
            xs = self.path_transitions.get(ticker) or []
            already = bool(xs and abs(xs[-1]["t"] - float(t)) <= EPS and xs[-1]["action"] == "CANCEL")
            if not already:
                reason = "M5_OR_INVALID_BOOK" if (np.isfinite(elapsed) and elapsed >= 300.0) else "BOOK_PATH_REMOVAL"
                self._record_transition(ticker, t, "CANCEL", before, reason)

    def _on_trade(self, t, r):
        ticker = str(r.get("ticker") or "")
        before = dict(self.quote.get(ticker) or {}) if self.quote.get(ticker) is not None else None
        super()._on_trade(t, r)
        after = self.quote.get(ticker)
        if before is not None and after is None:
            self._record_transition(ticker, t, "CANCEL", before, "FILL_POP")

    def last_transition(self, ticker, action=None, before_t=None):
        xs = self.path_transitions.get(str(ticker)) or []
        for x in reversed(xs):
            if before_t is not None and x["t"] > float(before_t) + EPS:
                continue
            if action is None or x["action"] == action:
                return x
        return None

    def last_matching_transition(self, ticker, side, price, action, before_t):
        xs = self.path_transitions.get(str(ticker)) or []
        side = str(side).upper()
        price = float(price)
        for x in reversed(xs):
            if x["t"] > float(before_t) + EPS or x["action"] != action:
                continue
            q = x.get("quote") or {}
            if str(q.get("role") or "").upper() != "ENTRY":
                continue
            if str(q.get("side") or "").upper() != side:
                continue
            if abs(_f(q.get("price")) - price) <= EPS:
                return x
        return None


def _classify_path(shadow: InstrumentedPathShadow, ticker: str, side: str, price: float, t: float):
    q = dict(shadow.quote.get(ticker) or {}) if shadow.quote.get(ticker) is not None else None
    inv = float(shadow.inventory.get(ticker, 0.0))
    side = str(side).upper()
    price = float(price)

    last_any = shadow.last_transition(ticker, before_t=t)
    last_cancel = shadow.last_transition(ticker, action="CANCEL", before_t=t)
    last_same_open = shadow.last_matching_transition(ticker, side, price, "OPEN", t)
    last_same_cancel = shadow.last_matching_transition(ticker, side, price, "CANCEL", t)

    if abs(inv) > OOS.EPS or (q is not None and str(q.get("role") or "").upper() == "EXIT"):
        cls = "SHADOW_IN_INVENTORY_CYCLE"
    elif q is not None and str(q.get("role") or "").upper() == "ENTRY":
        qside = str(q.get("side") or "").upper()
        qpx = _f(q.get("price"))
        if qside == side and np.isfinite(qpx) and abs(qpx - price) <= EPS:
            cls = "SHADOW_ALREADY_RESTING_SAME_QUOTE"
        elif qside != side:
            cls = "SHADOW_DIFFERENT_SIDE"
        else:
            cls = "SHADOW_DIFFERENT_PRICE"
    else:
        cancel_age = float(t - last_cancel["t"]) if last_cancel is not None else np.nan
        cls = (
            "SHADOW_CANCELLED_RECENTLY"
            if np.isfinite(cancel_age) and -EPS <= cancel_age <= RECENT_CANCEL_S + EPS
            else "SHADOW_NO_CANDIDATE"
        )

    cur = shadow.current.get(ticker)
    elapsed = _f(shadow.latest_elapsed.get(ticker))
    desired = shadow._desired_quote(ticker, cur, elapsed) if cur is not None and np.isfinite(elapsed) else None
    raw_sig = shadow.raw_candidate_sig.get(ticker)

    q_join = _f((q or {}).get("join_ts"))
    q_ahead = _f((q or {}).get("queue_ahead"))
    q_initial = _f((q or {}).get("queue_ahead_initial"))

    return {
        "path_class": cls,
        "shadow_inventory": inv,
        "shadow_quote_role": str((q or {}).get("role") or ""),
        "shadow_quote_side": str((q or {}).get("side") or ""),
        "shadow_quote_price": _f((q or {}).get("price")),
        "shadow_quote_qty": _f((q or {}).get("qty")),
        "shadow_quote_remaining_qty": _f((q or {}).get("remaining_qty")),
        "shadow_quote_queue_ahead": q_ahead,
        "shadow_quote_queue_ahead_initial": q_initial,
        "shadow_quote_age_at_live_send_ms": 1000.0 * (t - q_join) if np.isfinite(q_join) else np.nan,
        "shadow_desired_role": str((desired or {}).get("role") or ""),
        "shadow_desired_side": str((desired or {}).get("side") or ""),
        "shadow_desired_price": _f((desired or {}).get("price")),
        "shadow_elapsed_s": elapsed,
        "shadow_latest_book_age_ms": 1000.0 * (t - _f(shadow.latest_book_ts.get(ticker))) if np.isfinite(_f(shadow.latest_book_ts.get(ticker))) else np.nan,
        "raw_candidate_side": raw_sig[0] if raw_sig else None,
        "raw_candidate_price": raw_sig[1] if raw_sig else np.nan,
        "raw_candidate_age_ms": 1000.0 * (t - _f(shadow.raw_candidate_since.get(ticker))) if np.isfinite(_f(shadow.raw_candidate_since.get(ticker))) else np.nan,
        "last_shadow_transition_action": (last_any or {}).get("action"),
        "last_shadow_transition_reason": (last_any or {}).get("reason"),
        "last_shadow_transition_age_ms": 1000.0 * (t - last_any["t"]) if last_any is not None else np.nan,
        "last_shadow_cancel_reason": (last_cancel or {}).get("reason"),
        "last_shadow_cancel_age_ms": 1000.0 * (t - last_cancel["t"]) if last_cancel is not None else np.nan,
        "last_same_quote_open_age_ms": 1000.0 * (t - last_same_open["t"]) if last_same_open is not None else np.nan,
        "last_same_quote_cancel_age_ms": 1000.0 * (t - last_same_cancel["t"]) if last_same_cancel is not None else np.nan,
    }


def _weighted(g: pd.DataFrame, col: str, weight: str):
    if col not in g or weight not in g:
        return np.nan
    x = pd.to_numeric(g[col], errors="coerce")
    w = pd.to_numeric(g[weight], errors="coerce")
    m = x.notna() & w.notna() & (w > EPS)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def _econ_by_order(econ: pd.DataFrame):
    if econ is None or econ.empty or "order_id" not in econ:
        return {}
    out = {}
    for oid, g in econ.groupby("order_id", sort=False):
        qty_col = "qty" if "qty" in g else "entry_qty" if "entry_qty" in g else None
        if qty_col is None:
            continue
        q = float(pd.to_numeric(g[qty_col], errors="coerce").fillna(0.0).sum())
        rec = {
            "econ_entry_qty": q,
            "econ_wide_bucket": str(g["wide_bucket"].dropna().iloc[0]) if "wide_bucket" in g and g["wide_bucket"].notna().any() else None,
            "econ_total_realized_gross_pnl": float(pd.to_numeric(g.get("total_realized_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum()) if "total_realized_gross_pnl" in g else np.nan,
            "econ_passive_gross_pnl": float(pd.to_numeric(g.get("passive_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum()) if "passive_gross_pnl" in g else np.nan,
            "econ_m5_gross_pnl": float(pd.to_numeric(g.get("m5_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum()) if "m5_gross_pnl" in g else np.nan,
            "econ_residual_open_qty": float(pd.to_numeric(g.get("residual_open_qty", 0.0), errors="coerce").fillna(0.0).sum()) if "residual_open_qty" in g else np.nan,
        }
        for col in ("markout_50ms_c", "markout_100ms_c", "markout_250ms_c", "markout_1s_c", "markout_5s_c"):
            rec[col] = _weighted(g, col, qty_col) if col in g else np.nan
        out[str(oid)] = rec
    return out


def _live_fill_catalog(source: Path):
    rows, _ = V6._load_all_strategy_fills(source)
    by_oid = defaultdict(list)
    by_ticker_entry_times = defaultdict(list)
    for r in rows:
        if r["role"] == "ENTRY":
            by_oid[str(r["order_id"])].append(r)
            by_ticker_entry_times[str(r["ticker"])].append(float(r["fill_s"]))
    out = {}
    for oid, xs in by_oid.items():
        xs.sort(key=lambda z: z["fill_s"])
        out[oid] = {
            "actual_first_fill_s": float(xs[0]["fill_s"]),
            "actual_last_fill_s": float(xs[-1]["fill_s"]),
            "actual_entry_fill_qty": float(sum(x["qty"] for x in xs)),
            "actual_entry_fill_rows": len(xs),
        }
    for ticker in by_ticker_entry_times:
        by_ticker_entry_times[ticker].sort()
    return out, dict(by_ticker_entry_times)


def _recent_live_path_features(entry_creates: list[dict], entry_fill_times: dict[str, list[float]]):
    by_ticker = defaultdict(list)
    out = {}
    for r in sorted(entry_creates, key=lambda z: z["send_s"]):
        ticker = r["ticker"]
        t = float(r["send_s"])
        side = str(r["side"]).upper()
        px = float(r["price"])
        prev = by_ticker[ticker]
        same_1s = sum(1 for x in prev if t - x["send_s"] <= 1.0 + EPS and x["side"] == side and abs(x["price"] - px) <= EPS)
        same_5s = sum(1 for x in prev if t - x["send_s"] <= 5.0 + EPS and x["side"] == side and abs(x["price"] - px) <= EPS)
        same_30s = sum(1 for x in prev if t - x["send_s"] <= 30.0 + EPS and x["side"] == side and abs(x["price"] - px) <= EPS)

        fills = entry_fill_times.get(ticker) or []
        j = bisect.bisect_left(fills, t) - 1
        last_fill = fills[j] if j >= 0 else -np.inf
        requotes_since_fill = sum(1 for x in prev if x["send_s"] > last_fill + EPS) + 1

        out[r["order_id"]] = {
            "prior_same_quote_creates_1s": same_1s,
            "prior_same_quote_creates_5s": same_5s,
            "prior_same_quote_creates_30s": same_30s,
            "entry_create_number_since_prior_entry_fill": requotes_since_fill,
            "prior_entry_fill_age_ms": 1000.0 * (t - last_fill) if np.isfinite(last_fill) else np.nan,
        }
        prev.append({"send_s": t, "side": side, "price": px, "order_id": r["order_id"]})
    return out


def run_q5_live_shadow_order_path_forensic(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"
    required = [
        source / "process_config.json",
        source / "fee_preflight.json",
        source / "orders.jsonl",
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

    v5_dir, v5_summary, v5_orders = _find_v5_output(source)
    v6_dir, v6_summary, v6_detail = _find_v6_output(source)
    econ_by_oid = _econ_by_order(v6_detail)

    # Reconstruct exact CREATE send timestamps from the live order catalog and
    # attach V5 fill-model classification fields.
    live_catalog = V1._load_order_catalog(source)
    v5_lookup = {str(r["order_id"]): r for _, r in v5_orders.iterrows()}
    fill_catalog, entry_fill_times = _live_fill_catalog(source)

    entry_creates = []
    for oid, o in live_catalog.items():
        ticker = str(o.get("ticker") or "")
        role = str(o.get("role") or "").upper()
        send_s = _f(o.get("send_s"))
        if ticker not in selected_tickers or role != "ENTRY" or not np.isfinite(send_s):
            continue
        side = str(o.get("side") or "").upper()
        px = _f(o.get("price"))
        if side not in {"BID", "ASK"} or not np.isfinite(px):
            continue
        vr = v5_lookup.get(str(oid))
        actual_qty = _f(vr.get("actual_fill_qty"), 0.0) if vr is not None else 0.0
        wide_known = bool(vr.get("wide_known")) if vr is not None else False
        wide_pred = _f(vr.get("wide_predicted_fill_qty"), 0.0) if vr is not None else 0.0
        if actual_qty > EPS and wide_known:
            wide_bucket = "TP" if wide_pred > EPS else "FN"
        elif actual_qty > EPS:
            wide_bucket = "UNKNOWN"
        else:
            wide_bucket = "NOT_ACTUAL_FILL"
        meta = meta_by_ticker.get(ticker) or {}
        rec = {
            "order_id": str(oid),
            "ticker": ticker,
            "series": str(meta.get("series_ticker") or ""),
            "close_time": str(meta.get("close_time") or ""),
            "send_s": float(send_s),
            "ack_s": _f(o.get("ack_s")),
            "side": side,
            "price": float(px),
            "submitted_qty": _f(o.get("submitted_qty")),
            "actual_fill_qty": float(actual_qty),
            "wide_predicted_fill_qty": float(wide_pred),
            "wide_bucket": wide_bucket,
            "live_displayed_queue_ahead": _f(vr.get("displayed_queue_ahead")) if vr is not None else np.nan,
        }
        rec.update(fill_catalog.get(str(oid)) or {})
        er = econ_by_oid.get(str(oid)) or {}
        rec.update(er)
        entry_creates.append(rec)

    if not entry_creates:
        raise RuntimeError("No live ENTRY CREATEs reconstructed.")
    entry_creates.sort(key=lambda z: z["send_s"])
    live_path_features = _recent_live_path_features(entry_creates, entry_fill_times)

    out = _new_output(source.name)
    shadow = InstrumentedPathShadow(out, fee)
    for row in meta_rows:
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        shadow.meta[ticker] = row
        shadow.series_by_ticker[ticker] = str(row.get("series_ticker") or "")
        shadow.close_by_ticker[ticker] = str(row.get("close_time") or "")

    book_it = iter(_iter_jsonl(raw / "book_top3_events.jsonl") or [])
    trade_it = iter(_iter_jsonl(raw / "trades_event_time.jsonl") or [])
    b = BASE._next_selected(book_it, selected_tickers)
    tr = BASE._next_selected(trade_it, selected_tickers)
    if b is None and tr is None:
        raise RuntimeError("No selected raw events.")

    first_ts = min(x[0] for x in (b, tr) if x is not None)
    shadow.started_at = pd.Timestamp(first_ts, unit="s", tz="UTC")
    shadow.thread_alive = True
    shadow.emit("Q5_ORDER_PATH_FORENSIC_REPLAY_START", detail=str(source))

    snapshots = []
    raw_events = 0

    def process_one_raw():
        nonlocal b, tr, raw_events
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
            shadow._on_book(t, row)
            b = BASE._next_selected(book_it, selected_tickers)
        else:
            t, row = tr
            shadow._on_trade(t, row)
            tr = BASE._next_selected(trade_it, selected_tickers)
        shadow._update_drawdown()
        raw_events += 1

    for idx, order in enumerate(entry_creates, start=1):
        send_t = float(order["send_s"])
        while b is not None or tr is not None:
            next_raw = min(x[0] for x in (b, tr) if x is not None)
            if next_raw > send_t + EPS:
                break
            process_one_raw()

        state = _classify_path(
            shadow, order["ticker"], order["side"], order["price"], send_t
        )
        row = dict(order)
        row.update(state)
        row.update(live_path_features.get(order["order_id"]) or {})

        first_fill = _f(row.get("actual_first_fill_s"))
        row["live_create_to_first_fill_ms"] = (
            1000.0 * (first_fill - send_t)
            if np.isfinite(first_fill) else np.nan
        )
        row["live_queue_advantage_vs_shadow_remaining"] = (
            _f(row.get("shadow_quote_queue_ahead")) - _f(row.get("live_displayed_queue_ahead"))
            if np.isfinite(_f(row.get("shadow_quote_queue_ahead")))
            and np.isfinite(_f(row.get("live_displayed_queue_ahead")))
            else np.nan
        )
        row["live_queue_advantage_vs_shadow_initial"] = (
            _f(row.get("shadow_quote_queue_ahead_initial")) - _f(row.get("live_displayed_queue_ahead"))
            if np.isfinite(_f(row.get("shadow_quote_queue_ahead_initial")))
            and np.isfinite(_f(row.get("live_displayed_queue_ahead")))
            else np.nan
        )
        snapshots.append(row)

        if show and idx % 500 == 0:
            print(f"classified {idx:,}/{len(entry_creates):,} live ENTRY CREATEs | raw events replayed={raw_events:,}")

    shadow.thread_alive = False
    shadow.emit("Q5_ORDER_PATH_FORENSIC_REPLAY_STOP", creates=len(snapshots), raw_events=raw_events)

    df = pd.DataFrame(snapshots)
    tp = df[df["wide_bucket"] == "TP"].copy()
    fn = df[df["wide_bucket"] == "FN"].copy()

    def path_summary(frame: pd.DataFrame, label: str):
        rows = []
        if frame.empty:
            return pd.DataFrame()
        for cls, g in frame.groupby("path_class", sort=False):
            q = float(pd.to_numeric(g["actual_fill_qty"], errors="coerce").fillna(0.0).sum())
            gross = float(pd.to_numeric(g.get("econ_total_realized_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum()) if "econ_total_realized_gross_pnl" in g else np.nan
            rows.append({
                "bucket": label,
                "path_class": cls,
                "orders": int(len(g)),
                "filled_orders": int((pd.to_numeric(g["actual_fill_qty"], errors="coerce").fillna(0.0) > EPS).sum()),
                "actual_fill_qty": q,
                "realized_gross_pnl": gross,
                "gross_pnl_per_filled_contract_c": 100.0 * gross / q if q > EPS and np.isfinite(gross) else np.nan,
                "create_to_fill_median_ms": float(pd.to_numeric(g["live_create_to_first_fill_ms"], errors="coerce").median()),
                "shadow_quote_age_weighted_ms": _weighted(g, "shadow_quote_age_at_live_send_ms", "actual_fill_qty"),
                "live_displayed_queue_weighted": _weighted(g, "live_displayed_queue_ahead", "actual_fill_qty"),
                "shadow_queue_remaining_weighted": _weighted(g, "shadow_quote_queue_ahead", "actual_fill_qty"),
                "live_queue_advantage_vs_shadow_remaining_weighted": _weighted(g, "live_queue_advantage_vs_shadow_remaining", "actual_fill_qty"),
                "raw_candidate_age_weighted_ms": _weighted(g, "raw_candidate_age_ms", "actual_fill_qty"),
                "prior_same_quote_creates_5s_weighted": _weighted(g, "prior_same_quote_creates_5s", "actual_fill_qty"),
                "requotes_since_prior_fill_weighted": _weighted(g, "entry_create_number_since_prior_entry_fill", "actual_fill_qty"),
                "markout_250ms_c": _weighted(g, "markout_250ms_c", "actual_fill_qty"),
                "markout_1s_c": _weighted(g, "markout_1s_c", "actual_fill_qty"),
                "markout_5s_c": _weighted(g, "markout_5s_c", "actual_fill_qty"),
            })
        return pd.DataFrame(rows).sort_values("realized_gross_pnl")

    tp_path = path_summary(tp, "TP")
    fn_path = path_summary(fn, "FN")

    all_path = (
        df.groupby(["wide_bucket", "path_class"], as_index=False)
        .agg(
            orders=("order_id", "count"),
            actual_fill_qty=("actual_fill_qty", "sum"),
            wide_predicted_fill_qty=("wide_predicted_fill_qty", "sum"),
        )
        .sort_values(["wide_bucket", "orders"], ascending=[True, False])
    )

    # First path mismatch in each market/window is useful for identifying where
    # live and shadow state machines first depart before later inventory cascades.
    first_div_rows = []
    for (close, ticker), g in df.groupby(["close_time", "ticker"], sort=False):
        g = g.sort_values("send_s")
        bad = g[g["path_class"] != "SHADOW_ALREADY_RESTING_SAME_QUOTE"]
        if bad.empty:
            continue
        r = bad.iloc[0]
        first_div_rows.append({
            "close_time": close,
            "series": r["series"],
            "ticker": ticker,
            "order_id": r["order_id"],
            "send_time": pd.Timestamp(float(r["send_s"]), unit="s", tz="UTC").isoformat(),
            "path_class": r["path_class"],
            "wide_bucket": r["wide_bucket"],
            "actual_fill_qty": r["actual_fill_qty"],
            "side": r["side"],
            "price": r["price"],
            "shadow_inventory": r["shadow_inventory"],
            "shadow_quote_role": r["shadow_quote_role"],
            "shadow_quote_side": r["shadow_quote_side"],
            "shadow_quote_price": r["shadow_quote_price"],
            "last_shadow_transition_reason": r["last_shadow_transition_reason"],
            "last_shadow_transition_age_ms": r["last_shadow_transition_age_ms"],
        })
    first_div = pd.DataFrame(first_div_rows)

    # Worst TP orders by actual realized economics.
    worst_tp_cols = [
        "close_time", "series", "ticker", "order_id", "path_class", "side", "price",
        "actual_fill_qty", "econ_total_realized_gross_pnl", "markout_250ms_c", "markout_1s_c", "markout_5s_c",
        "live_create_to_first_fill_ms", "shadow_quote_age_at_live_send_ms",
        "live_displayed_queue_ahead", "shadow_quote_queue_ahead",
        "live_queue_advantage_vs_shadow_remaining",
        "prior_same_quote_creates_5s", "entry_create_number_since_prior_entry_fill",
        "shadow_inventory", "last_shadow_transition_reason", "last_shadow_transition_age_ms",
    ]
    worst_tp = tp.copy()
    if "econ_total_realized_gross_pnl" in worst_tp:
        worst_tp = worst_tp.sort_values("econ_total_realized_gross_pnl").head(40)
    else:
        worst_tp = worst_tp.head(40)
    worst_tp = worst_tp[[c for c in worst_tp_cols if c in worst_tp.columns]]

    same_tp = tp[tp["path_class"] == "SHADOW_ALREADY_RESTING_SAME_QUOTE"].copy()
    same_quote_stats = {
        "orders": int(len(same_tp)),
        "fill_qty": float(pd.to_numeric(same_tp.get("actual_fill_qty", 0.0), errors="coerce").fillna(0.0).sum()) if len(same_tp) else 0.0,
        "shadow_quote_age_ms_weighted": _weighted(same_tp, "shadow_quote_age_at_live_send_ms", "actual_fill_qty") if len(same_tp) else np.nan,
        "live_displayed_queue_weighted": _weighted(same_tp, "live_displayed_queue_ahead", "actual_fill_qty") if len(same_tp) else np.nan,
        "shadow_queue_remaining_weighted": _weighted(same_tp, "shadow_quote_queue_ahead", "actual_fill_qty") if len(same_tp) else np.nan,
        "live_queue_advantage_vs_shadow_remaining_weighted": _weighted(same_tp, "live_queue_advantage_vs_shadow_remaining", "actual_fill_qty") if len(same_tp) else np.nan,
        "realized_gross_pnl": float(pd.to_numeric(same_tp.get("econ_total_realized_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum()) if len(same_tp) else 0.0,
    }

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "v5_source": str(v5_dir),
        "v6_source": str(v6_dir),
        "live_windows": windows,
        "entry_creates": int(len(df)),
        "tp_entry_orders": int(len(tp)),
        "tp_entry_qty": float(pd.to_numeric(tp.get("actual_fill_qty", 0.0), errors="coerce").fillna(0.0).sum()),
        "fn_entry_orders": int(len(fn)),
        "fn_entry_qty": float(pd.to_numeric(fn.get("actual_fill_qty", 0.0), errors="coerce").fillna(0.0).sum()),
        "raw_events_replayed_through_last_create": int(raw_events),
        "recent_cancel_diagnostic_seconds": RECENT_CANCEL_S,
        "tp_path_counts": dict(Counter(tp["path_class"])) if len(tp) else {},
        "tp_path_fill_qty": {
            str(k): float(pd.to_numeric(g["actual_fill_qty"], errors="coerce").fillna(0.0).sum())
            for k, g in tp.groupby("path_class")
        } if len(tp) else {},
        "same_quote_tp_diagnostics": same_quote_stats,
        "same_realization_only": True,
        "independent_validation": False,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "profitability_guardrail": (
            "This forensic may identify why live execution departed from the frozen shadow path, "
            "but fixing that mechanism does not establish future profitability. A changed execution "
            "engine requires fresh Q1/Q5 forward validation and profitability still requires independent OOS evidence."
        ),
    }

    df.to_csv(out / "live_create_shadow_state_detail.csv", index=False)
    tp_path.to_csv(out / "tp_economics_by_shadow_path.csv", index=False)
    fn_path.to_csv(out / "fn_economics_by_shadow_path.csv", index=False)
    all_path.to_csv(out / "all_entry_creates_by_shadow_path.csv", index=False)
    first_div.to_csv(out / "first_path_divergence_by_market_window.csv", index=False)
    worst_tp.to_csv(out / "worst_tp_orders_with_shadow_state.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("=" * 136)
        print("Q5 LIVE-vs-FROZEN-SHADOW ORDER-PATH FORENSIC V7 — READ ONLY / SAME REALIZATION")
        print("=" * 136)
        print("Source:", source)
        print("V5 source:", v5_dir)
        print("V6 source:", v6_dir)
        print("Live ENTRY CREATEs classified:", len(df))
        print("TP filled ENTRY orders / qty:", len(tp), "/", f"{summary['tp_entry_qty']:.4f}")
        print("FN filled ENTRY orders / qty:", len(fn), "/", f"{summary['fn_entry_qty']:.4f}")
        print("Raw events replayed through final CREATE:", f"{raw_events:,}")
        print()
        print("TP PATH CLASSIFICATION — PRIMARY")
        if not tp_path.empty:
            print(tp_path.to_string(index=False))
        else:
            print("  no TP rows")
        print()
        print("FN PATH CLASSIFICATION — SENSITIVITY")
        if not fn_path.empty:
            print(fn_path.to_string(index=False))
        else:
            print("  no FN rows")
        print()
        print("SAME-QUOTE TP DIAGNOSTIC")
        print(json.dumps(same_quote_stats, indent=2, default=str))
        print()
        print("FIRST LIVE/SHADOW PATH DIVERGENCE BY MARKET/WINDOW")
        if not first_div.empty:
            print(first_div.to_string(index=False))
        else:
            print("  none")
        print()
        print("WORST 40 TP ORDERS WITH SHADOW STATE")
        if not worst_tp.empty:
            print(worst_tp.to_string(index=False))
        print()
        print("Interpretation guide:")
        print("  1) TP concentrated in SHADOW_IN_INVENTORY_CYCLE => fill-path divergence cascades into different future ENTRY opportunities.")
        print("  2) TP concentrated in DIFFERENT_SIDE/PRICE/NO_CANDIDATE => live decision/requote timing itself departs from the frozen shadow state.")
        print("  3) TP concentrated in SAME_QUOTE but live queue advantage is strongly positive => both chose the same price, but the live rejoin inherited less queue ahead than the older conservative shadow quote.")
        print("  4) Large same-price reentry/requote counts => repeated live cancel/recreate behavior may be resetting the path relative to one continuously resting shadow quote.")
        print("  5) This is diagnostic only. Do not infer guaranteed profit from repairing a path discrepancy; any execution change requires fresh Q1 then Q5.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 136)

    return {
        "summary": summary,
        "detail": df,
        "tp_path_summary": tp_path,
        "fn_path_summary": fn_path,
        "all_path_summary": all_path,
        "first_divergence": first_div,
        "worst_tp": worst_tp,
        "output_dir": out,
    }


__all__ = ["run_q5_live_shadow_order_path_forensic", "VERSION"]
