from __future__ import annotations

"""Q5 TP-vs-FN economics forensic conditioned on the actual live order tape.

Purpose
-------
V5 showed that, once given the real V12.2 CREATE/CANCEL tape, the frozen
public-trade queue model predicts most real passive fills.  This module asks the
next economic question: are the real fills that the model *misses* (false
negatives, FN) disproportionately toxic, or do most losses come from true-positive
(TP) fills that the model already predicts?

Primary classification
----------------------
WIDE classification from V5 is primary because it grants the largest plausible
resting interval (CREATE send -> CANCEL ack):
- TP: actual ENTRY order filled and V5 predicted a fill.
- FN: actual ENTRY order filled but V5 predicted no fill.
NARROW (CREATE ack -> CANCEL send) is reported as a sensitivity check.

For each real ENTRY fill this diagnostic computes:
- 50ms / 100ms / 250ms / 1s / 5s signed mid markouts;
- FIFO realized gross PnL through passive EXIT fills;
- FIFO realized gross PnL through M5_FLATTEN fills;
- risk-flatten gross PnL separately if present;
- residual quantity not yet matched to a closing fill;
- observed fee fields when present;
- ACK-bound unobserved queue reduction required by the corrected V4 replay;
- first observed live queue movement relative to displayed L1 when available.

Scientific guardrails
---------------------
- SAME-REALIZATION execution forensic only; NOT independent validation.
- NO exchange/API calls and NO orders.
- Source session is read-only.
- TP/FN economics are descriptive attribution, not causal treatment effects.
- Gross PnL is primary. Net PnL is only emitted when the relevant fill fee fields
  are actually present; otherwise it remains NaN rather than assuming zero fees.
- Writes only under results/kalshi_q5_tp_fn_economics_v6/.
"""

import bisect
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q5_same_realization_shadow_v1 as BASE
from . import mm_cycle_q5_public_trade_fill_reconciliation_v1 as V1
from . import mm_cycle_q5_live_shadow_fill_forensics_v1 as FSEL
from . import mm_cycle_q5_live_order_public_trade_queue_replay_v4 as V4
from . import mm_cycle_q5_actual_order_fill_replay_v5 as V5

VERSION = "MM_CYCLE_Q5_TP_FN_ECONOMICS_V6"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_tp_fn_economics_v6"
EPS = 1e-9
MARKOUTS_S = (0.050, 0.100, 0.250, 1.0, 5.0)
MARKOUT_MAX_AGE_S = 2.0


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


def _fee(row):
    """Return an observed fee value in dollars when the fill schema exposes one.

    We intentionally do not invent unit conversions. Current live fill rows use
    dollar-like fixed-point fields when present. If none of the known fee fields
    exists, return NaN so downstream net PnL is withheld.
    """
    for key in (
        "fee_cost", "fee_cost_dollars", "fee_dollars", "fee_usd",
        "taker_fee", "maker_fee", "fee",
    ):
        if key in row and row.get(key) is not None:
            z = _f(row.get(key))
            if np.isfinite(z):
                return float(z), key
    return np.nan, None


def _load_all_strategy_fills(session: Path):
    rows = []
    fee_keys = Counter()
    time_sources = Counter()
    allowed = {"ENTRY", "EXIT", "M5_FLATTEN", "RISK_FLATTEN"}
    for r in _iter_jsonl(session / "fills.jsonl") or []:
        role = FSEL._role(r)
        if role not in allowed:
            continue
        ticker = str(r.get("ticker") or r.get("market_ticker") or "")
        oid = str(r.get("order_id") or "")
        side = FSEL._side(r)
        qty = FSEL._qty(r)
        px = FSEL._price(r)
        t, tsrc = FSEL._live_fill_time(r)
        if not (
            ticker and oid and side in {"BID", "ASK"}
            and np.isfinite(qty) and qty > EPS
            and np.isfinite(px) and np.isfinite(t)
        ):
            continue
        fee, fee_key = _fee(r)
        if fee_key:
            fee_keys[fee_key] += 1
        time_sources[tsrc] += 1
        rows.append({
            "ticker": ticker,
            "order_id": oid,
            "fill_id": str(r.get("fill_id") or r.get("trade_id") or ""),
            "trade_id": str(r.get("trade_id") or ""),
            "role": role,
            "side": side,
            "qty": float(qty),
            "price": float(px),
            "fill_s": float(t),
            "fee_usd": float(fee) if np.isfinite(fee) else np.nan,
            "fee_key": fee_key,
            "fill_time_source": tsrc,
        })
    role_rank = {"ENTRY": 0, "EXIT": 1, "M5_FLATTEN": 2, "RISK_FLATTEN": 3}
    rows.sort(key=lambda z: (z["fill_s"], role_rank.get(z["role"], 9), z["ticker"], z["order_id"]))
    return rows, dict(time_sources), dict(fee_keys)


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
        # Safe fallback: V5 itself is read-only with respect to the source session.
        res = V5.run_q5_actual_order_fill_replay(source, show=False)
        return Path(res["output_dir"]).resolve(), res["summary"], res["orders"].copy()
    _, d, summary, op = max(candidates, key=lambda z: z[0])
    return d, summary, pd.read_csv(op)


def _load_book_index(raw: Path, selected: set[str]):
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
        by_ticker[ticker].append((float(t), cur))
    times = {}
    for ticker, xs in by_ticker.items():
        xs.sort(key=lambda z: z[0])
        times[ticker] = [z[0] for z in xs]
    return dict(by_ticker), times


def _book_after(ticker, target_s, books, times):
    arr = times.get(ticker) or []
    if not arr:
        return None
    i = bisect.bisect_left(arr, float(target_s))
    if i >= len(arr):
        return None
    rec = books[ticker][i]
    if rec[0] - float(target_s) > MARKOUT_MAX_AGE_S + EPS:
        return None
    return rec


def _markout_label(h):
    if h < 1.0:
        return f"markout_{int(round(h * 1000.0))}ms_c"
    return f"markout_{int(round(h))}s_c"


def _attach_markouts(entry_rows, books, times):
    for r in entry_rows:
        sign = 1.0 if r["side"] == "BID" else -1.0
        for h in MARKOUTS_S:
            rec = _book_after(r["ticker"], r["fill_s"] + h, books, times)
            key = _markout_label(h)
            r[key] = np.nan if rec is None else sign * (float(rec[1]["mid"]) - float(r["price"])) * 100.0


def _classify_order(row, prefix):
    actual = bool(_f(row.get("actual_fill_qty"), 0.0) > EPS)
    known = bool(row.get(f"{prefix}_known"))
    pred = bool(_f(row.get(f"{prefix}_predicted_fill_qty"), 0.0) > EPS) if known else False
    if not actual:
        return "NOT_ACTUAL_FILL"
    if not known:
        return "UNKNOWN"
    return "TP" if pred else "FN"


def _queue_diagnostics(source: Path, selected_tickers: set[str]):
    orders = V1._load_order_catalog(source)
    entry_fills, _ = V1._load_live_entry_fills(source, orders, selected_tickers)
    trades, _, by_ticker = V1._load_public_trades(source / "raw_capture", selected_tickers)
    by_oid = defaultdict(list)
    for f in entry_fills:
        by_oid[f["order_id"]].append(f)
    out = {}
    for oid, fs in by_oid.items():
        order = orders.get(oid)
        if not order:
            continue
        fs.sort(key=lambda z: z["fill_time_s"])
        ack_s = _f(order.get("ack_s"))
        if not np.isfinite(ack_s):
            continue
        q = V4._replay_bound(order, fs, trades, by_ticker, float(ack_s))
        first_fill = min(f["fill_time_s"] for f in fs)
        qobs = V4._first_prefill_queue(order, first_fill)
        q0 = _f(q.get("displayed_queue_ahead"))
        qpos = _f((qobs or {}).get("queue_position"))
        out[oid] = {
            "ack_queue_reduction_required_any": _f(q.get("queue_reduction_required_any")),
            "ack_queue_reduction_required_full": _f(q.get("queue_reduction_required_full")),
            "displayed_queue_ahead": q0,
            "first_prefill_queue_position": qpos,
            "first_prefill_queue_minus_displayed": qpos - q0 if np.isfinite(qpos) and np.isfinite(q0) else np.nan,
        }
    return out


def _fifo_realized_attribution(all_fills, entry_rows):
    """Attach actual realized gross PnL to ENTRY fill lots under ticker FIFO."""
    entry_lookup = defaultdict(deque)
    for i, r in enumerate(entry_rows):
        entry_lookup[(r["order_id"], r["fill_id"])].append(i)

    open_lots = defaultdict(deque)
    fee_complete = True
    unmatched_closing_qty = defaultdict(float)

    for f in all_fills:
        ticker = f["ticker"]
        role = f["role"]
        if role == "ENTRY":
            key = (f["order_id"], f["fill_id"])
            if entry_lookup[key]:
                idx = entry_lookup[key].popleft()
            else:
                # Defensive fallback for rare empty/non-unique fill ids.
                idx = next((j for j, r in enumerate(entry_rows)
                            if r["order_id"] == f["order_id"]
                            and abs(r["fill_s"] - f["fill_s"]) <= 1e-6
                            and abs(r["qty"] - f["qty"]) <= 1e-6), None)
            if idx is None:
                continue
            fee_per_qty = f["fee_usd"] / f["qty"] if np.isfinite(f["fee_usd"]) else np.nan
            if not np.isfinite(fee_per_qty):
                fee_complete = False
            open_lots[ticker].append({
                "row_index": idx,
                "qty": float(f["qty"]),
                "entry_side": f["side"],
                "entry_price": float(f["price"]),
                "entry_fee_per_qty": fee_per_qty,
            })
            continue

        if role not in {"EXIT", "M5_FLATTEN", "RISK_FLATTEN"}:
            continue

        rem = float(f["qty"])
        exit_fee_per_qty = f["fee_usd"] / f["qty"] if np.isfinite(f["fee_usd"]) else np.nan
        if not np.isfinite(exit_fee_per_qty):
            fee_complete = False

        while rem > EPS and open_lots[ticker]:
            lot = open_lots[ticker][0]
            q = min(rem, lot["qty"])
            gross = (
                (float(f["price"]) - lot["entry_price"]) * q
                if lot["entry_side"] == "BID"
                else (lot["entry_price"] - float(f["price"])) * q
            )
            rr = entry_rows[lot["row_index"]]
            if role == "EXIT":
                prefix = "passive"
            elif role == "M5_FLATTEN":
                prefix = "m5"
            else:
                prefix = "risk"
            rr[f"{prefix}_realized_qty"] += q
            rr[f"{prefix}_gross_pnl"] += gross
            if np.isfinite(lot["entry_fee_per_qty"]):
                rr["realized_entry_fee_usd"] += lot["entry_fee_per_qty"] * q
            else:
                rr["fee_complete"] = False
            if np.isfinite(exit_fee_per_qty):
                rr[f"{prefix}_exit_fee_usd"] += exit_fee_per_qty * q
            else:
                rr["fee_complete"] = False

            rem -= q
            lot["qty"] -= q
            if lot["qty"] <= EPS:
                open_lots[ticker].popleft()

        if rem > EPS:
            unmatched_closing_qty[role] += rem

    residual = defaultdict(float)
    for ticker, lots in open_lots.items():
        for lot in lots:
            residual[ticker] += lot["qty"]
            entry_rows[lot["row_index"]]["residual_open_qty"] += lot["qty"]

    for r in entry_rows:
        r["total_realized_qty"] = r["passive_realized_qty"] + r["m5_realized_qty"] + r["risk_realized_qty"]
        r["total_realized_gross_pnl"] = r["passive_gross_pnl"] + r["m5_gross_pnl"] + r["risk_gross_pnl"]
        fees = r["realized_entry_fee_usd"] + r["passive_exit_fee_usd"] + r["m5_exit_fee_usd"] + r["risk_exit_fee_usd"]
        r["total_observed_fee_usd"] = fees if r["fee_complete"] else np.nan
        r["total_realized_net_pnl"] = r["total_realized_gross_pnl"] - fees if r["fee_complete"] else np.nan

    return {
        "residual_open_qty_by_ticker": dict(residual),
        "unmatched_closing_qty_by_role": dict(unmatched_closing_qty),
        "fee_fields_complete_globally": bool(fee_complete),
    }


def _weighted_avg(g, value, weight="qty"):
    x = pd.to_numeric(g[value], errors="coerce")
    w = pd.to_numeric(g[weight], errors="coerce")
    m = x.notna() & w.notna() & (w > EPS)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def _bucket_summary(df: pd.DataFrame, class_col: str):
    rows = []
    for bucket in ("TP", "FN", "UNKNOWN"):
        g = df[df[class_col] == bucket].copy()
        if g.empty:
            continue
        rec = {
            "bucket": bucket,
            "entry_fill_rows": int(len(g)),
            "entry_orders": int(g["order_id"].nunique()),
            "entry_qty": float(g["qty"].sum()),
            "passive_realized_qty": float(g["passive_realized_qty"].sum()),
            "passive_gross_pnl": float(g["passive_gross_pnl"].sum()),
            "m5_realized_qty": float(g["m5_realized_qty"].sum()),
            "m5_gross_pnl": float(g["m5_gross_pnl"].sum()),
            "risk_realized_qty": float(g["risk_realized_qty"].sum()),
            "risk_gross_pnl": float(g["risk_gross_pnl"].sum()),
            "total_realized_qty": float(g["total_realized_qty"].sum()),
            "total_realized_gross_pnl": float(g["total_realized_gross_pnl"].sum()),
            "residual_open_qty": float(g["residual_open_qty"].sum()),
            "fee_complete_qty": float(g.loc[g["fee_complete"] == True, "qty"].sum()),
            "observed_fee_usd": float(pd.to_numeric(g["total_observed_fee_usd"], errors="coerce").sum(min_count=1)) if pd.to_numeric(g["total_observed_fee_usd"], errors="coerce").notna().any() else np.nan,
            "realized_net_pnl_when_fee_complete": float(pd.to_numeric(g["total_realized_net_pnl"], errors="coerce").sum(min_count=1)) if pd.to_numeric(g["total_realized_net_pnl"], errors="coerce").notna().any() else np.nan,
            "gross_pnl_per_entry_contract_c": 100.0 * float(g["total_realized_gross_pnl"].sum()) / float(g["qty"].sum()) if float(g["qty"].sum()) > EPS else np.nan,
            "queue_diag_qty": float(g.loc[pd.to_numeric(g["ack_queue_reduction_required_any"], errors="coerce").notna(), "qty"].sum()),
            "queue_reduction_required_any_weighted": _weighted_avg(g, "ack_queue_reduction_required_any"),
            "queue_reduction_required_full_weighted": _weighted_avg(g, "ack_queue_reduction_required_full"),
            "first_prefill_queue_minus_displayed_weighted": _weighted_avg(g, "first_prefill_queue_minus_displayed"),
        }
        for h in MARKOUTS_S:
            key = _markout_label(h)
            rec[key] = _weighted_avg(g, key)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_q5_tp_fn_economics(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"
    required = [
        source / "fills.jsonl",
        source / "orders.jsonl",
        source / "latency_events_v12.jsonl",
        source / "queue_positions.jsonl",
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

    v5_dir, v5_summary, v5_orders = _find_v5_output(source)
    if v5_orders.empty:
        raise RuntimeError("V5 actual-order replay contains no orders.")

    windows = BASE._live_windows(source)
    _, meta_by_ticker = BASE._metadata(raw)
    selected_tickers = {
        t for t, r in meta_by_ticker.items()
        if str(r.get("close_time") or "") in set(windows)
    }

    all_fills, fill_time_sources, fee_key_counts = _load_all_strategy_fills(source)
    all_fills = [f for f in all_fills if f["ticker"] in selected_tickers]
    entry_fills = [dict(f) for f in all_fills if f["role"] == "ENTRY"]
    if not entry_fills:
        raise RuntimeError("No actual ENTRY fills reconstructed.")

    v5_map = {str(r["order_id"]): r for _, r in v5_orders.iterrows()}
    qdiag = _queue_diagnostics(source, selected_tickers)
    books, times = _load_book_index(raw, {f["ticker"] for f in entry_fills})

    rows = []
    for f in entry_fills:
        vr = v5_map.get(f["order_id"])
        if vr is None:
            narrow_bucket = wide_bucket = "UNKNOWN"
        else:
            narrow_bucket = _classify_order(vr, "narrow")
            wide_bucket = _classify_order(vr, "wide")
        meta = meta_by_ticker.get(f["ticker"]) or {}
        z = {
            **f,
            "series": str(meta.get("series_ticker") or ""),
            "close_time": str(meta.get("close_time") or ""),
            "narrow_bucket": narrow_bucket,
            "wide_bucket": wide_bucket,
            "passive_realized_qty": 0.0,
            "passive_gross_pnl": 0.0,
            "m5_realized_qty": 0.0,
            "m5_gross_pnl": 0.0,
            "risk_realized_qty": 0.0,
            "risk_gross_pnl": 0.0,
            "realized_entry_fee_usd": 0.0,
            "passive_exit_fee_usd": 0.0,
            "m5_exit_fee_usd": 0.0,
            "risk_exit_fee_usd": 0.0,
            "residual_open_qty": 0.0,
            "fee_complete": bool(np.isfinite(f["fee_usd"])),
        }
        z.update(qdiag.get(f["order_id"]) or {})
        rows.append(z)

    _attach_markouts(rows, books, times)
    attribution = _fifo_realized_attribution(all_fills, rows)
    df = pd.DataFrame(rows)

    # Sanity check against V5 actual ENTRY quantity.
    entry_qty = float(df["qty"].sum())
    expected_qty = _f(v5_summary.get("actual_live_entry_fill_qty_all"))
    if np.isfinite(expected_qty) and abs(entry_qty - expected_qty) > 1e-6:
        raise RuntimeError(
            f"ENTRY quantity mismatch vs V5: this={entry_qty:.6f} V5={expected_qty:.6f}"
        )

    wide = _bucket_summary(df, "wide_bucket")
    narrow = _bucket_summary(df, "narrow_bucket")

    by_window = df.groupby(["close_time", "wide_bucket"], as_index=False).agg(
        entry_qty=("qty", "sum"),
        passive_gross_pnl=("passive_gross_pnl", "sum"),
        m5_gross_pnl=("m5_gross_pnl", "sum"),
        risk_gross_pnl=("risk_gross_pnl", "sum"),
        total_realized_gross_pnl=("total_realized_gross_pnl", "sum"),
        residual_open_qty=("residual_open_qty", "sum"),
    )
    by_asset = df.groupby(["series", "wide_bucket"], as_index=False).agg(
        entry_qty=("qty", "sum"),
        passive_gross_pnl=("passive_gross_pnl", "sum"),
        m5_gross_pnl=("m5_gross_pnl", "sum"),
        risk_gross_pnl=("risk_gross_pnl", "sum"),
        total_realized_gross_pnl=("total_realized_gross_pnl", "sum"),
        residual_open_qty=("residual_open_qty", "sum"),
    )

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "v5_output": str(v5_dir),
        "actual_entry_fill_rows": int(len(df)),
        "actual_entry_qty": entry_qty,
        "fill_time_sources": fill_time_sources,
        "fee_key_counts": fee_key_counts,
        "attribution": attribution,
        "wide_primary": wide.to_dict(orient="records"),
        "narrow_sensitivity": narrow.to_dict(orient="records"),
        "same_realization_only": True,
        "independent_validation": False,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "interpretation_guardrail": (
            "TP/FN buckets are descriptive classifications conditional on the actual live order tape. "
            "They do not identify a causal treatment effect of queue advancement."
        ),
    }

    out = _new_output(source.name)
    df.to_csv(out / "tp_fn_entry_fill_detail.csv", index=False)
    wide.to_csv(out / "tp_fn_wide_bucket_summary.csv", index=False)
    narrow.to_csv(out / "tp_fn_narrow_bucket_summary.csv", index=False)
    by_window.to_csv(out / "tp_fn_wide_by_window.csv", index=False)
    by_asset.to_csv(out / "tp_fn_wide_by_asset.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("=" * 132)
        print("Q5 TP-vs-FN ECONOMICS V6 — ACTUAL LIVE FILLS / READ ONLY")
        print("=" * 132)
        print("Source:", source)
        print("V5 source:", v5_dir)
        print("Actual ENTRY fills / qty:", len(df), "/", f"{entry_qty:.4f}")
        print("Fee fields observed:", fee_key_counts if fee_key_counts else "NONE — gross PnL only")
        print("Residual open qty after FIFO attribution:", attribution["residual_open_qty_by_ticker"])
        print("Unmatched closing qty by role:", attribution["unmatched_closing_qty_by_role"])
        print()
        print("WIDE PRIMARY — SEND -> CANCEL ACK")
        if not wide.empty:
            print(wide.to_string(index=False))
        print()
        print("NARROW SENSITIVITY — ACK -> CANCEL SEND")
        if not narrow.empty:
            print(narrow.to_string(index=False))
        print()
        print("WIDE BY WINDOW")
        if not by_window.empty:
            print(by_window.to_string(index=False))
        print()
        print("WIDE BY ASSET")
        if not by_asset.empty:
            print(by_asset.to_string(index=False))
        print()
        print("Interpretation:")
        print("  - Strongly worse FN markouts / realized PnL => fills missed by the trade-only queue model are economically toxic.")
        print("  - Similar FN and TP economics => unmodeled queue advancement is mainly a fill-frequency issue, not the loss mechanism.")
        print("  - Worse TP economics => most realized damage came from fills the frozen model already predicts once given the actual live orders.")
        print("  - M5 gross PnL is shown separately from passive EXIT PnL.")
        print("  - Gross PnL is primary; net PnL is withheld wherever fee fields are missing.")
        print("  - Same-realization descriptive forensic only; not independent validation and not a causal queue experiment.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 132)

    return {
        "summary": summary,
        "detail": df,
        "wide_bucket_summary": wide,
        "narrow_bucket_summary": narrow,
        "by_window": by_window,
        "by_asset": by_asset,
        "output_dir": out,
    }


__all__ = ["run_q5_tp_fn_economics", "VERSION"]
