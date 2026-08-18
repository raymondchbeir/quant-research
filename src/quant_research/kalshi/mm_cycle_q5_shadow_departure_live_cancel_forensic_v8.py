from __future__ import annotations

"""Q5 same-quote shadow-departure vs live-cancel forensic.

Purpose
-------
V7 showed that most losing TP ENTRY fills occurred when the frozen shadow was
already resting at the exact same side/price as the actual live CREATE.  This
module follows those SAME-QUOTE TP orders from actual CREATE send until actual
first fill and asks exactly what happens first:

1) the frozen shadow stays on the same quote until the live fill;
2) the shadow leaves because its own hypothetical queue model fills first
   (FILL_POP / inventory transition);
3) the shadow leaves because a later raw BOOK state invalidates/reprices/flips the
   quote, in which case we compare that exact departure with the live V12.2
   watchdog invalidation, cancel send, cancel acknowledgement, and real fill.

This separates three very different mechanisms:
- fill-path divergence (shadow filled earlier, so no equivalent live cancel is
  logically expected merely because the shadow changed inventory);
- live semantic/reaction failure (book invalidation but no timely live cancel);
- irreducible cancel-flight exposure (live cancel sent promptly, fill occurred
  before cancel acknowledgement/effectiveness could be established).

Scientific guardrails
---------------------
- SAME-REALIZATION execution forensic only; NOT independent validation.
- NO exchange/API calls and NO orders.
- Source session is read-only.
- Candidate-C and all strategy thresholds remain unchanged.
- Exchange cancel-effective time is not directly observed. DELETE send/response
  provide causal bounds, not proof that response time equals matching-engine
  cancellation effectiveness.
- Writes only under results/kalshi_q5_shadow_departure_live_cancel_forensic_v8/.
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
from . import mm_cycle_q5_actual_order_fill_replay_v5 as V5
from . import mm_cycle_q5_tp_fn_economics_v6 as V6
from . import mm_cycle_q5_live_shadow_order_path_forensic_v7 as V7
from . import mm_cycle_q5_live_shadow_order_path_forensic_v7_1 as V71

VERSION = "MM_CYCLE_Q5_SHADOW_DEPARTURE_LIVE_CANCEL_FORENSIC_V8"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_shadow_departure_live_cancel_forensic_v8"
EPS = 1e-9
MATCH_RECEIPT_TOL_MS = 3.0


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


def _find_v7_output(source: Path):
    root = Path(V7.OUTPUT_ROOT)
    candidates = []
    if root.exists():
        for d in root.glob(source.name + "*"):
            sp = d / "summary.json"
            dp = d / "live_create_shadow_state_detail.csv"
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
        res = V71.run_q5_live_shadow_order_path_forensic(source, show=False)
        return Path(res["output_dir"]).resolve(), res["summary"], res["detail"].copy()
    _, d, summary, dp = max(candidates, key=lambda z: z[0])
    return d, summary, pd.read_csv(dp)


def _same_target_state(shadow, target):
    ticker = str(target["ticker"])
    q = shadow.quote.get(ticker)
    inv = float(shadow.inventory.get(ticker, 0.0))
    if abs(inv) > OOS.EPS or not isinstance(q, dict):
        return False
    return bool(
        str(q.get("role") or "").upper() == "ENTRY"
        and str(q.get("side") or "").upper() == str(target["side"]).upper()
        and np.isfinite(_f(q.get("price")))
        and abs(_f(q.get("price")) - float(target["price"])) <= EPS
    )


def _departure_class(event_type, transition_reason, shadow, ticker):
    reason = str(transition_reason or "")
    q = shadow.quote.get(str(ticker))
    inv = float(shadow.inventory.get(str(ticker), 0.0))

    if reason == "FILL_POP" or (str(event_type) == "TRADE" and abs(inv) > OOS.EPS):
        return "SHADOW_FILLED_BEFORE_LIVE"
    if reason == "SIDE_CHANGE":
        return "BOOK_SIDE_CHANGE"
    if reason == "PRICE_CHANGE":
        return "BOOK_PRICE_CHANGE"
    if reason == "DESIRED_NONE":
        return "BOOK_ENTRY_FILTER_NONE"
    if reason in {"M5_OR_INVALID_BOOK", "BOOK_PATH_REMOVAL"}:
        return "BOOK_OUTSIDE_OR_INVALID"
    if abs(inv) > OOS.EPS or (isinstance(q, dict) and str(q.get("role") or "").upper() == "EXIT"):
        return "SHADOW_INVENTORY_TRANSITION_OTHER"
    if q is None:
        return "SHADOW_QUOTE_REMOVED_OTHER"
    return "SHADOW_REQUOTE_OTHER"


def _load_entry_fill_map(source: Path, selected_tickers: set[str]):
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


def _load_public_first_fill_receipts(source: Path, selected_tickers: set[str], first_fill_map):
    trades, by_trade_id, by_ticker = V1._load_public_trades(source / "raw_capture", selected_tickers)
    out = {}
    for oid, f in first_fill_map.items():
        tid = str(f.get("trade_id") or "")
        candidates = by_trade_id.get(tid, []) if tid else []
        if candidates:
            # Prefer the exact-id event nearest the live fill exchange timestamp.
            fill_s = float(f["fill_s"])
            i = min(
                candidates,
                key=lambda j: abs(
                    (_f(trades[j].get("exchange_s"), trades[j]["receipt_s"])) - fill_s
                ),
            )
            tr = trades[i]
            out[oid] = {
                "public_trade_id": tid,
                "public_fill_receipt_s": float(tr["receipt_s"]),
                "public_fill_exchange_s": _f(tr.get("exchange_s")),
            }
            continue

        # Defensive fallback through V1's same-price/side/time matcher.
        ff = {
            "trade_id": tid,
            "ticker": f["ticker"],
            "side": f["side"],
            "price": f["price"],
            "fill_time_s": f["fill_s"],
        }
        m = V1._match_fill(ff, trades, by_trade_id, by_ticker)
        idx = m.get("trade_index")
        if idx is not None:
            tr = trades[idx]
            out[oid] = {
                "public_trade_id": str(tr.get("trade_id") or ""),
                "public_fill_receipt_s": float(tr["receipt_s"]),
                "public_fill_exchange_s": _f(tr.get("exchange_s")),
            }
    return out


def _load_fast_cancel_events(source: Path):
    invalidations_by_oid = defaultdict(list)
    result_by_inv = {}
    terminal_by_inv = {}

    for r in _iter_jsonl(source / "latency_events_v12.jsonl") or []:
        event = str(r.get("event") or "")
        oid = str(r.get("order_id") or "")
        iid = str(r.get("invalidation_id") or "")
        if event == "FAST_INVALIDATION_DETECTED" and oid:
            z = dict(r)
            z["obsolete_receipt_s"] = _f(r.get("obsolete_receipt_wall_ms")) / 1000.0
            z["detect_s"] = _f(r.get("watchdog_detect_wall_ms")) / 1000.0
            invalidations_by_oid[oid].append(z)
        elif event == "FAST_CANCEL_RESULT" and iid:
            result_by_inv[iid] = dict(r)
        elif event == "FAST_INVALIDATION_TERMINAL" and iid:
            terminal_by_inv[iid] = dict(r)

    for rows in invalidations_by_oid.values():
        rows.sort(key=lambda z: _f(z.get("obsolete_receipt_s"), np.inf))

    return dict(invalidations_by_oid), result_by_inv, terminal_by_inv


def _all_cancel_requests(source: Path):
    """Preserve every observed cancel request timing by order id."""
    out = defaultdict(list)

    # Priority path.
    for r in _iter_jsonl(source / "latency_events_v12.jsonl") or []:
        if str(r.get("event") or "") != "FAST_CANCEL_RESULT":
            continue
        oid = str(r.get("order_id") or "")
        if not oid:
            continue
        timing = r.get("cancel_timing") or {}
        send_s = _f(timing.get("request_send_wall_ms")) / 1000.0
        ack_s = _f(timing.get("response_recv_wall_ms")) / 1000.0
        out[oid].append({
            "source": "FAST_CANCEL_RESULT",
            "send_s": send_s,
            "ack_s": ack_s,
            "success": r.get("success"),
            "reason": r.get("reason"),
            "invalidation_id": r.get("invalidation_id"),
        })

    # Main/fallback path from orders.jsonl.
    for r in _iter_jsonl(source / "orders.jsonl") or []:
        action = str(r.get("action") or "").upper()
        if "CANCEL" not in action:
            continue
        track = r.get("track") or {}
        oid = str(r.get("order_id") or track.get("order_id") or "")
        if not oid:
            for container_key in ("result", "cancel_body", "batch_body"):
                z = r.get(container_key)
                if isinstance(z, dict) and z.get("order_id"):
                    oid = str(z.get("order_id"))
                    break
        if not oid:
            continue
        found = False
        for label, timing in V5._nested_timing_candidates(r):
            send_s, ack_s = V5._timing_pair(timing)
            if np.isfinite(send_s) or np.isfinite(ack_s):
                out[oid].append({
                    "source": f"ORDERS:{action}:{label}",
                    "send_s": send_s,
                    "ack_s": ack_s,
                    "success": True,
                    "reason": r.get("reason") or r.get("source"),
                    "invalidation_id": None,
                })
                found = True
        if not found:
            t = OOS._ts(r.get("time"))
            if np.isfinite(t):
                out[oid].append({
                    "source": f"ORDERS:{action}:row_time_upper_bound",
                    "send_s": np.nan,
                    "ack_s": float(t),
                    "success": None,
                    "reason": r.get("reason") or r.get("source"),
                    "invalidation_id": None,
                })

    for oid in out:
        out[oid].sort(key=lambda z: _f(z.get("send_s"), _f(z.get("ack_s"), np.inf)))
    return dict(out)


def _match_live_reaction(row, invalidations_by_oid, result_by_inv, terminal_by_inv, cancel_requests):
    oid = str(row["order_id"])
    dep_s = _f(row.get("shadow_departure_s"))
    fill_s = _f(row.get("actual_first_fill_s"))
    dep_class = str(row.get("shadow_departure_class") or "")

    if dep_class == "STAYED_SAME_QUOTE_UNTIL_LIVE_FILL":
        return {"live_reaction_class": "SHADOW_STAYED_TO_FILL"}

    # A shadow fill is endogenous to the shadow queue model. Real live inventory
    # remained flat until the actual fill, so there is no equivalent public-book
    # invalidation that live should blindly mirror.
    if dep_class == "SHADOW_FILLED_BEFORE_LIVE":
        return {"live_reaction_class": "SHADOW_FILL_POP_NOT_A_LIVE_CANCEL_SIGNAL"}

    invs = invalidations_by_oid.get(oid) or []
    matched_inv = None
    if np.isfinite(dep_s):
        for x in invs:
            obs = _f(x.get("obsolete_receipt_s"))
            if np.isfinite(obs) and obs >= dep_s - MATCH_RECEIPT_TOL_MS / 1000.0:
                matched_inv = x
                break

    iid = str((matched_inv or {}).get("invalidation_id") or "")
    result = result_by_inv.get(iid) or {}
    terminal = terminal_by_inv.get(iid) or {}
    timing = result.get("cancel_timing") or {}
    fast_send_s = _f(timing.get("request_send_wall_ms")) / 1000.0
    fast_ack_s = _f(timing.get("response_recv_wall_ms")) / 1000.0

    # If the exact fast result did not expose timing, find the earliest observed
    # cancel request at/after the shadow book departure across all cancel paths.
    chosen = None
    if np.isfinite(fast_send_s) or np.isfinite(fast_ack_s):
        chosen = {
            "source": "MATCHED_FAST_INVALIDATION",
            "send_s": fast_send_s,
            "ack_s": fast_ack_s,
            "success": result.get("success"),
            "reason": result.get("reason"),
        }
    else:
        for c in cancel_requests.get(oid) or []:
            send = _f(c.get("send_s"))
            ack = _f(c.get("ack_s"))
            basis = send if np.isfinite(send) else ack
            if np.isfinite(basis) and (not np.isfinite(dep_s) or basis >= dep_s - 0.002):
                chosen = c
                break

    send_s = _f((chosen or {}).get("send_s"))
    ack_s = _f((chosen or {}).get("ack_s"))

    if not np.isfinite(send_s):
        reaction = "BOOK_DEPARTURE_NO_LIVE_CANCEL_SEND_OBSERVED"
    elif np.isfinite(fill_s) and send_s >= fill_s - EPS:
        reaction = "BOOK_DEPARTURE_CANCEL_SENT_AFTER_OR_AT_FILL"
    elif np.isfinite(fill_s) and send_s < fill_s - EPS:
        if np.isfinite(ack_s) and ack_s >= fill_s - EPS:
            reaction = "BOOK_DEPARTURE_FILL_DURING_CANCEL_FLIGHT"
        elif np.isfinite(ack_s) and ack_s < fill_s - EPS:
            reaction = "BOOK_DEPARTURE_CANCEL_ACK_BEFORE_FILL_TIMESTAMP"
        else:
            reaction = "BOOK_DEPARTURE_CANCEL_SENT_BEFORE_FILL_ACK_UNKNOWN"
    else:
        reaction = "BOOK_DEPARTURE_CANCEL_TIMING_INCOMPLETE"

    inv_obs_s = _f((matched_inv or {}).get("obsolete_receipt_s"))
    detect_s = _f((matched_inv or {}).get("detect_s"))

    return {
        "live_reaction_class": reaction,
        "matched_invalidation_id": iid or None,
        "matched_invalidation_reason": (matched_inv or {}).get("reason"),
        "matched_invalidation_obsolete_receipt_s": inv_obs_s,
        "matched_invalidation_detect_s": detect_s,
        "invalidation_receipt_minus_shadow_departure_ms": 1000.0 * (inv_obs_s - dep_s) if np.isfinite(inv_obs_s) and np.isfinite(dep_s) else np.nan,
        "shadow_departure_to_watchdog_detect_ms": 1000.0 * (detect_s - dep_s) if np.isfinite(detect_s) and np.isfinite(dep_s) else np.nan,
        "live_cancel_source": (chosen or {}).get("source"),
        "live_cancel_success": (chosen or {}).get("success"),
        "live_cancel_send_s": send_s,
        "live_cancel_ack_s": ack_s,
        "shadow_departure_to_cancel_send_ms": 1000.0 * (send_s - dep_s) if np.isfinite(send_s) and np.isfinite(dep_s) else np.nan,
        "cancel_send_to_fill_ms": 1000.0 * (fill_s - send_s) if np.isfinite(fill_s) and np.isfinite(send_s) else np.nan,
        "cancel_ack_minus_fill_ms": 1000.0 * (ack_s - fill_s) if np.isfinite(fill_s) and np.isfinite(ack_s) else np.nan,
        "terminal_state": terminal.get("state"),
        "terminal_safe": terminal.get("safe"),
    }


def _weighted(g: pd.DataFrame, col: str, weight="actual_fill_qty"):
    if col not in g or weight not in g:
        return np.nan
    x = pd.to_numeric(g[col], errors="coerce")
    w = pd.to_numeric(g[weight], errors="coerce")
    m = x.notna() & w.notna() & (w > EPS)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def _group_summary(df: pd.DataFrame, group_col: str):
    rows = []
    for key, g in df.groupby(group_col, dropna=False, sort=False):
        qty = float(pd.to_numeric(g["actual_fill_qty"], errors="coerce").fillna(0.0).sum())
        pnl = float(pd.to_numeric(g.get("econ_total_realized_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum())
        rows.append({
            group_col: key,
            "orders": int(len(g)),
            "fill_qty": qty,
            "realized_gross_pnl": pnl,
            "gross_pnl_per_contract_c": 100.0 * pnl / qty if qty > EPS else np.nan,
            "create_to_shadow_departure_ms_weighted": _weighted(g, "create_to_shadow_departure_ms"),
            "shadow_departure_to_live_fill_ms_weighted": _weighted(g, "shadow_departure_to_live_fill_ms"),
            "shadow_departure_to_cancel_send_ms_weighted": _weighted(g, "shadow_departure_to_cancel_send_ms"),
            "cancel_send_to_fill_ms_weighted": _weighted(g, "cancel_send_to_fill_ms"),
            "markout_250ms_c": _weighted(g, "markout_250ms_c"),
            "markout_1s_c": _weighted(g, "markout_1s_c"),
            "markout_5s_c": _weighted(g, "markout_5s_c"),
        })
    return pd.DataFrame(rows).sort_values("realized_gross_pnl") if rows else pd.DataFrame()


def run_q5_shadow_departure_live_cancel_forensic(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"
    required = [
        source / "process_config.json",
        source / "fee_preflight.json",
        source / "latency_events_v12.jsonl",
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

    v7_dir, v7_summary, v7_detail = _find_v7_output(source)
    targets = v7_detail[
        (v7_detail["wide_bucket"].astype(str) == "TP")
        & (v7_detail["path_class"].astype(str) == "SHADOW_ALREADY_RESTING_SAME_QUOTE")
        & (pd.to_numeric(v7_detail["actual_fill_qty"], errors="coerce").fillna(0.0) > EPS)
    ].copy()
    if targets.empty:
        raise RuntimeError("No SAME-QUOTE TP ENTRY orders found in V7 output.")

    first_fill_map = _load_entry_fill_map(source, selected_tickers)
    public_fill = _load_public_first_fill_receipts(source, selected_tickers, first_fill_map)

    target_rows = []
    for _, r in targets.iterrows():
        oid = str(r["order_id"])
        ff = first_fill_map.get(oid)
        if not ff:
            continue
        rec = r.to_dict()
        rec["send_s"] = float(_f(rec.get("send_s")))
        rec["actual_first_fill_s"] = float(ff["fill_s"])
        rec["actual_first_fill_trade_id"] = str(ff.get("trade_id") or "")
        rec.update(public_fill.get(oid) or {})
        target_rows.append(rec)
    if not target_rows:
        raise RuntimeError("No SAME-QUOTE TP targets had parseable actual first fills.")
    target_rows.sort(key=lambda z: z["send_s"])

    out = _new_output(source.name)
    workspace = out / "shadow_trace_workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    shadow = V7.InstrumentedPathShadow(workspace, fee)
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
        raise RuntimeError("No selected raw events.")

    first_ts = min(x[0] for x in (b, tr) if x is not None)
    shadow.started_at = pd.Timestamp(first_ts, unit="s", tz="UTC")
    shadow.thread_alive = True

    target_by_oid = {str(r["order_id"]): r for r in target_rows}
    pending = list(target_rows)
    pidx = 0
    active = {}
    finished = {}
    raw_events = 0

    def next_raw_time():
        xs = [x[0] for x in (b, tr) if x is not None]
        return min(xs) if xs else np.inf

    def next_fill_time():
        if not active:
            return np.inf
        return min(float(x["actual_first_fill_s"]) for x in active.values())

    def activate_target(tgt):
        oid = str(tgt["order_id"])
        tgt = dict(tgt)
        tgt["same_quote_at_trace_start"] = _same_target_state(shadow, tgt)
        tgt["shadow_departure_s"] = np.nan
        tgt["shadow_departure_event_type"] = None
        tgt["shadow_departure_transition_reason"] = None
        tgt["shadow_departure_class"] = None
        active[oid] = tgt

    def finalize_target(oid):
        tgt = active.pop(oid)
        if not np.isfinite(_f(tgt.get("shadow_departure_s"))):
            tgt["shadow_departure_class"] = "STAYED_SAME_QUOTE_UNTIL_LIVE_FILL"
            tgt["shadow_departure_s"] = float(tgt["actual_first_fill_s"])
            tgt["shadow_departure_event_type"] = "LIVE_FILL_BOUNDARY"
        tgt["create_to_shadow_departure_ms"] = 1000.0 * (
            float(tgt["shadow_departure_s"]) - float(tgt["send_s"])
        )
        tgt["shadow_departure_to_live_fill_ms"] = 1000.0 * (
            float(tgt["actual_first_fill_s"]) - float(tgt["shadow_departure_s"])
        )
        finished[oid] = tgt

    while pidx < len(pending) or active:
        raw_t = next_raw_time()
        send_t = float(pending[pidx]["send_s"]) if pidx < len(pending) else np.inf
        fill_t = next_fill_time()

        # Raw public event wins exact timestamp ties, matching the V7 replay's
        # process-all-raw-events-through-send observation convention.
        if raw_t <= send_t + EPS and raw_t <= fill_t + EPS:
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
                and float(tgt["send_s"]) <= float(t) + EPS
                and float(t) <= float(tgt["actual_first_fill_s"]) + EPS
                and not np.isfinite(_f(tgt.get("shadow_departure_s")))
            ]
            before_same = {oid: _same_target_state(shadow, active[oid]) for oid in affected}
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
                if before_same.get(oid) and not _same_target_state(shadow, tgt):
                    tgt["shadow_departure_s"] = float(t)
                    tgt["shadow_departure_event_type"] = typ
                    tgt["shadow_departure_transition_reason"] = reason
                    tgt["shadow_departure_class"] = _departure_class(
                        typ, reason, shadow, ticker
                    )
                    tgt["shadow_departure_raw_elapsed_s"] = _f(row.get("elapsed_s"))
                    tgt["shadow_inventory_after_departure"] = float(shadow.inventory.get(ticker, 0.0))
                    q = shadow.quote.get(ticker) or {}
                    tgt["shadow_quote_role_after_departure"] = str(q.get("role") or "")
                    tgt["shadow_quote_side_after_departure"] = str(q.get("side") or "")
                    tgt["shadow_quote_price_after_departure"] = _f(q.get("price"))
            continue

        if send_t <= fill_t + EPS:
            activate_target(pending[pidx])
            pidx += 1
            continue

        # Fill boundary reached before next raw receipt.
        due = [
            oid for oid, tgt in active.items()
            if float(tgt["actual_first_fill_s"]) <= fill_t + EPS
        ]
        for oid in due:
            finalize_target(oid)

    shadow.thread_alive = False

    df = pd.DataFrame(list(finished.values()))
    if df.empty:
        raise RuntimeError("No completed target traces.")

    invalidations_by_oid, result_by_inv, terminal_by_inv = _load_fast_cancel_events(source)
    cancel_requests = _all_cancel_requests(source)

    reaction_rows = []
    for _, r in df.iterrows():
        d = r.to_dict()
        d.update(
            _match_live_reaction(
                d,
                invalidations_by_oid,
                result_by_inv,
                terminal_by_inv,
                cancel_requests,
            )
        )
        pub_receipt = _f(d.get("public_fill_receipt_s"))
        cancel_send = _f(d.get("live_cancel_send_s"))
        d["cancel_send_to_public_fill_receipt_ms"] = (
            1000.0 * (pub_receipt - cancel_send)
            if np.isfinite(pub_receipt) and np.isfinite(cancel_send)
            else np.nan
        )
        reaction_rows.append(d)
    df = pd.DataFrame(reaction_rows)

    by_departure = _group_summary(df, "shadow_departure_class")
    by_reaction = _group_summary(df, "live_reaction_class")

    # Book-driven rows are the only rows for which 'why did live not cancel when
    # the shadow cancelled?' is a semantically valid question.
    book_mask = df["shadow_departure_class"].astype(str).str.startswith("BOOK_")
    book_df = df[book_mask].copy()
    fillpop_df = df[df["shadow_departure_class"] == "SHADOW_FILLED_BEFORE_LIVE"].copy()
    stayed_df = df[df["shadow_departure_class"] == "STAYED_SAME_QUOTE_UNTIL_LIVE_FILL"].copy()

    def qty_pnl(x):
        if x.empty:
            return 0.0, 0.0
        q = float(pd.to_numeric(x["actual_fill_qty"], errors="coerce").fillna(0.0).sum())
        p = float(pd.to_numeric(x.get("econ_total_realized_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum())
        return q, p

    book_qty, book_pnl = qty_pnl(book_df)
    fillpop_qty, fillpop_pnl = qty_pnl(fillpop_df)
    stayed_qty, stayed_pnl = qty_pnl(stayed_df)

    reaction_counts = dict(Counter(df["live_reaction_class"].astype(str)))
    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "v7_source": str(v7_dir),
        "same_quote_tp_orders": int(len(df)),
        "same_quote_tp_fill_qty": float(pd.to_numeric(df["actual_fill_qty"], errors="coerce").fillna(0.0).sum()),
        "same_quote_tp_realized_gross_pnl": float(pd.to_numeric(df.get("econ_total_realized_gross_pnl", 0.0), errors="coerce").fillna(0.0).sum()),
        "starting_state_reproduced_same_quote_orders": int(df["same_quote_at_trace_start"].fillna(False).sum()),
        "shadow_fill_pop_before_live": {"orders": int(len(fillpop_df)), "qty": fillpop_qty, "gross_pnl": fillpop_pnl},
        "shadow_book_departure_before_live": {"orders": int(len(book_df)), "qty": book_qty, "gross_pnl": book_pnl},
        "shadow_stayed_same_until_live_fill": {"orders": int(len(stayed_df)), "qty": stayed_qty, "gross_pnl": stayed_pnl},
        "live_reaction_counts": reaction_counts,
        "book_departure_to_cancel_send_ms_weighted": _weighted(book_df, "shadow_departure_to_cancel_send_ms") if len(book_df) else np.nan,
        "book_cancel_send_to_fill_ms_weighted": _weighted(book_df, "cancel_send_to_fill_ms") if len(book_df) else np.nan,
        "same_realization_only": True,
        "independent_validation": False,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "interpretation_guardrail": (
            "Shadow FILL_POP is not itself a live cancellation signal because the real order had not filled. "
            "Only BOOK-driven shadow departure can be compared one-for-one with the V12.2 public-book watchdog. "
            "DELETE response time bounds but does not directly reveal exchange cancel-effective time."
        ),
    }

    # Worst rows first for manual inspection.
    detail_sort = "econ_total_realized_gross_pnl" if "econ_total_realized_gross_pnl" in df else "actual_fill_qty"
    detail = df.sort_values(detail_sort, ascending=True)

    detail.to_csv(out / "same_quote_tp_departure_cancel_detail.csv", index=False)
    by_departure.to_csv(out / "economics_by_shadow_departure.csv", index=False)
    by_reaction.to_csv(out / "economics_by_live_reaction.csv", index=False)
    book_df.sort_values(detail_sort).to_csv(out / "book_departures_only.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("=" * 138)
        print("Q5 SHADOW-DEPARTURE vs LIVE-CANCEL FORENSIC V8 — SAME QUOTE TP ORDERS / READ ONLY")
        print("=" * 138)
        print("Source:", source)
        print("V7 source:", v7_dir)
        print("Same-quote TP orders / qty:", len(df), "/", f"{summary['same_quote_tp_fill_qty']:.4f}")
        print("Same-quote TP gross PnL:", f"{summary['same_quote_tp_realized_gross_pnl']:+.4f}")
        print("Starting same-quote state reproduced:", summary["starting_state_reproduced_same_quote_orders"], "/", len(df))
        print("Raw events replayed:", f"{raw_events:,}")
        print()
        print("SHADOW DEPARTURE ECONOMICS")
        if not by_departure.empty:
            print(by_departure.to_string(index=False))
        print()
        print("LIVE REACTION ECONOMICS")
        if not by_reaction.empty:
            print(by_reaction.to_string(index=False))
        print()
        print("CORE DECOMPOSITION")
        print(f"  shadow filled first (FILL_POP): orders={len(fillpop_df)} qty={fillpop_qty:.4f} pnl={fillpop_pnl:+.4f}")
        print(f"  book-driven shadow departure:  orders={len(book_df)} qty={book_qty:.4f} pnl={book_pnl:+.4f}")
        print(f"  shadow stayed to live fill:     orders={len(stayed_df)} qty={stayed_qty:.4f} pnl={stayed_pnl:+.4f}")
        print()
        if len(book_df):
            print("BOOK-DRIVEN DEPARTURES — CANCEL TIMING")
            cols = [
                "close_time", "series", "ticker", "order_id", "shadow_departure_class",
                "shadow_departure_transition_reason", "actual_fill_qty",
                "econ_total_realized_gross_pnl", "shadow_departure_to_live_fill_ms",
                "matched_invalidation_reason", "shadow_departure_to_watchdog_detect_ms",
                "shadow_departure_to_cancel_send_ms", "cancel_send_to_fill_ms",
                "cancel_ack_minus_fill_ms", "live_reaction_class",
            ]
            print(book_df[[c for c in cols if c in book_df.columns]].sort_values("econ_total_realized_gross_pnl").to_string(index=False))
            print()
        print("Interpretation:")
        print("  - SHADOW_FILLED_BEFORE_LIVE means the shadow changed state because its hypothetical order filled first; live should not cancel merely because that shadow inventory changed.")
        print("  - BOOK_DEPARTURE_FILL_DURING_CANCEL_FLIGHT means live saw the same invalidating book state and sent DELETE before the fill, but the fill timestamp lies before cancel acknowledgement.")
        print("  - BOOK_DEPARTURE_NO_LIVE_CANCEL_SEND_OBSERVED or CANCEL_SENT_AFTER_OR_AT_FILL means a genuine semantic/reaction-path problem remains.")
        print("  - SHADOW_STAYED_TO_FILL means cancellation cannot explain that divergence; the fill model/queue path must explain it.")
        print("  - Same realization only. Do not change strategy or promote Q10 from this forensic alone.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 138)

    return {
        "summary": summary,
        "detail": detail,
        "by_departure": by_departure,
        "by_reaction": by_reaction,
        "book_departures": book_df,
        "output_dir": out,
    }


__all__ = ["run_q5_shadow_departure_live_cancel_forensic", "VERSION"]
