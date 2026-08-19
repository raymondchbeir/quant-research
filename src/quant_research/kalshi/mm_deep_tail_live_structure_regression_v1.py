from __future__ import annotations

"""Development-only structural regression for the live one-net-position mechanic.

The historical V4 research evaluated YES-tail and NO-tail 5c positions independently.
A real Kalshi market has one normalized YES book: a 5c YES bid and a 5c NO bid are a
YES bid at .05 and a YES ask at .95. Once one side fills, continuing to treat the two
tails as independent inventories is not a faithful account-state model.

The live implementation therefore cancels the opposite 5c tail after the first observed
entry fill. This script quantifies that executable one-tail-at-a-time interpretation on
the original 24h DEVELOPMENT realization using the already-built compact trade/BBO/M5
caches. It is NOT validation and it never reads the 15h or held-out samples.

The selected tail is the first strict-through entry event observed causally. That tail
may continue filling to Q5; the opposite tail is ignored for PnL after selection. We
also report whether opposite-tail strict-through flow appears inside 25/50/100/200ms
after selection as a cancellation-race diagnostic. Full selected Q5 entries use the
same fixed JOIN_ASK + 100ms historical replay rule; partial entries remain M5-only.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_join_ask_capacity_dev_v7 as V7
from . import mm_deep_tail_join_ask_capacity_dev_v7_2 as V72
from . import mm_deep_tail_passive_exit_dev_v4 as V4

VERSION = "MM_DEEP_TAIL_LIVE_STRUCTURE_REGRESSION_V1"
HARD_BOUND_SESSION = V7.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_live_structure_regression_v1"
QTY = 5.0
ENTRY = 0.05
EPS = 1e-10
CANCEL_RACE_MS = (25, 50, 100, 200)


def _new_output(name):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_ROOT / str(name)
    if p.exists():
        p = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    p.mkdir(parents=True, exist_ok=False)
    return p.resolve()


def _outcome_px(tail, yes):
    return float(yes) if tail == "YES" else 1.0 - float(yes)


def _seller_side(tail):
    return "ask" if tail == "YES" else "bid"


def _strict_events(rows, tail, active_s):
    seller = _seller_side(tail)
    out = []
    for tr in rows:
        if tr["exec_s"] + EPS < active_s or tr["receipt_s"] + EPS < active_s:
            continue
        if str(tr["taker_book_side"]) != seller:
            continue
        opx = _outcome_px(tail, tr["yes_price"])
        if opx >= ENTRY - EPS:
            continue
        out.append({
            "tail": tail,
            "qty": max(0.0, float(tr["qty"])),
            "exec_s": float(tr["exec_s"]),
            "receipt_s": float(tr["receipt_s"]),
            "obs_s": float(max(tr["exec_s"], tr["receipt_s"])),
            "trade_id": str(tr.get("trade_id") or ""),
        })
    return out


def _selected_fill(events, tail):
    rem = QTY
    filled = 0.0
    first_exec = np.nan
    first_obs = np.nan
    full_exec = np.nan
    full_obs = np.nan
    for e in events:
        if e["tail"] != tail:
            continue
        take = min(rem, float(e["qty"]))
        if take <= EPS:
            continue
        if not np.isfinite(first_exec):
            first_exec = e["exec_s"]
            first_obs = e["obs_s"]
        filled += take
        rem -= take
        if rem <= EPS:
            full_exec = e["exec_s"]
            full_obs = e["obs_s"]
            break
    return {
        "entry_filled_qty": filled,
        "full": bool(filled >= QTY - EPS),
        "first_exec_s": first_exec,
        "first_obs_s": first_obs,
        "full_exec_s": full_exec,
        "full_obs_s": full_obs,
    }


def _latest_bbo(g, t):
    if g is None or g.empty or not np.isfinite(t):
        return None
    x = g["receipt_s"].to_numpy(float)
    i = int(np.searchsorted(x, float(t) + EPS, side="right") - 1)
    return None if i < 0 else g.iloc[i]


def _outcome_bbo(row, tail):
    if tail == "YES":
        return float(row["yes_ask"]), max(0.0, float(row["ask_q1"]))
    return 1.0 - float(row["yes_bid"]), max(0.0, float(row["bid_q1"]))


def _zero_m5(residual=0.0):
    return {
        "exit_qty": 0.0,
        "residual_qty_zero_valued": float(residual),
        "m5_exit_proceeds": 0.0,
        "m5_taker_fee": 0.0,
    }


def run_structure_regression(source_session, *, hard_bind=True, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected development session {HARD_BOUND_SESSION}, got {source.name}")

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored development fee preflight is not PASS")
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    meta = V7.V3.V1._metadata(source)
    v3, v4, _, _ = V7._load_required_prior_results(source)
    base_bbo, m5, _, _ = V7._load_book_cache(source)
    anchors, _ = V72._load_q5_anchors(source)
    prelude, _, _ = V72._build_prelude(source, meta, anchors, show=False)
    bbo = V72._augment_bbo(base_bbo, prelude)

    q5 = v3[np.isclose(pd.to_numeric(v3["requested_qty"], errors="coerce"), 5.0)].copy()
    q5 = q5[
        q5["coverage_eligible"].astype(bool)
        & (pd.to_numeric(q5["entry_filled_qty"], errors="coerce").fillna(0.0) > EPS)
    ]
    tickers = set(q5["ticker"].astype(str))
    trades_df, trade_source = V7._load_m1_trade_cache(source, meta, tickers, show=show)
    trades = {
        str(t): g.to_dict("records")
        for t, g in trades_df.groupby("ticker", sort=False)
    }
    bbo_by_ticker = {
        str(t): g.sort_values("receipt_s", kind="mergesort").reset_index(drop=True)
        for t, g in bbo.groupby("ticker", sort=False)
    }

    series_by_ticker = {
        str(r["ticker"]): str(r.get("series") or "")
        for _, r in q5.drop_duplicates("ticker").iterrows()
    }

    rows = []
    for ticker in sorted(tickers):
        active_s = float(meta[ticker]["window_start_s"]) + V7.M1_S + V7.ACTIVATION_LATENCY_MS / 1000.0
        ev = _strict_events(trades.get(ticker, []), "YES", active_s)
        ev += _strict_events(trades.get(ticker, []), "NO", active_s)
        ev.sort(key=lambda z: (z["obs_s"], z["exec_s"], z["trade_id"], z["tail"]))
        if not ev:
            continue

        chosen = str(ev[0]["tail"])
        opposite = "NO" if chosen == "YES" else "YES"
        fill = _selected_fill(ev, chosen)
        first_obs = float(fill["first_obs_s"])

        race = {}
        for ms in CANCEL_RACE_MS:
            deadline = first_obs + ms / 1000.0
            q = sum(
                float(e["qty"]) for e in ev
                if e["tail"] == opposite
                and e["exec_s"] <= deadline + EPS
                and e["receipt_s"] <= deadline + EPS
            )
            race[ms] = q

        entry_qty = float(fill["entry_filled_qty"])
        series = series_by_ticker.get(ticker, str(meta[ticker].get("series") or ""))
        mult = float(fee_mult.get(series, np.nan))
        snap_m5 = m5.get(ticker)
        if not np.isfinite(mult) or mult <= 0 or not snap_m5:
            raise RuntimeError(f"Missing fee/M5 cache for {ticker}")

        passive_qty = 0.0
        quote = np.nan
        queue = np.nan
        if fill["full"]:
            snap = _latest_bbo(bbo_by_ticker.get(ticker), fill["full_obs_s"])
            if snap is None:
                raise RuntimeError(f"Missing BBO at selected full fill for {ticker} {chosen}")
            quote, queue = _outcome_bbo(snap, chosen)
            pex = V4._simulate_passive_exit(
                trades.get(ticker, []),
                chosen,
                {"quote_price": quote, "queue_ahead_initial": queue},
                float(fill["full_obs_s"]) + V7.ACTIVATION_LATENCY_MS / 1000.0,
                entry_qty,
            )
            passive_qty = float(pex["passive_exit_qty"])

        residual = max(0.0, entry_qty - passive_qty)
        m5ex = (
            V7.V3._consume_m5_depth(chosen, residual, snap_m5, mult)
            if residual > EPS else _zero_m5(0.0)
        )
        rounding = V7.ROUNDING_DRAG if float(m5ex["exit_qty"]) > EPS else 0.0
        net = (
            passive_qty * (quote if np.isfinite(quote) else 0.0)
            + float(m5ex["m5_exit_proceeds"])
            - ENTRY * entry_qty
            - float(m5ex["m5_taker_fee"])
            - rounding
        )
        rows.append({
            "ticker": ticker,
            "series": series,
            "chosen_tail": chosen,
            "opposite_tail": opposite,
            "entry_filled_qty": entry_qty,
            "full_q5": bool(fill["full"]),
            "first_observed_fill_s": first_obs,
            "full_observed_fill_s": fill["full_obs_s"],
            "join_ask_c": 100.0 * quote if np.isfinite(quote) else np.nan,
            "passive_exit_qty": passive_qty,
            "m5_exit_qty": float(m5ex["exit_qty"]),
            "m5_residual_zero_valued": float(m5ex["residual_qty_zero_valued"]),
            "net_pnl_rounding_bound": float(net),
            **{f"opposite_strict_through_qty_within_{ms}ms": float(race[ms]) for ms in CANCEL_RACE_MS},
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No one-net-position Q5 events reconstructed")

    old_join = v4[v4["variant"].astype(str).eq("JOIN_ASK")].copy()
    old_net = float(pd.to_numeric(old_join["net_pnl_rounding_bound"], errors="coerce").fillna(0.0).sum())
    net = float(detail["net_pnl_rounding_bound"].sum())
    entry_qty = float(detail["entry_filled_qty"].sum())
    pqty = float(detail["passive_exit_qty"].sum())
    m5q = float(detail["m5_exit_qty"].sum())
    resid = float(detail["m5_residual_zero_valued"].sum())

    races = {}
    for ms in CANCEL_RACE_MS:
        col = f"opposite_strict_through_qty_within_{ms}ms"
        races[str(ms)] = {
            "tickers": int((detail[col] > EPS).sum()),
            "qty": float(detail[col].sum()),
        }

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source": str(source),
        "research_stage": "DEVELOPMENT_STRUCTURAL_REGRESSION_NOT_VALIDATION",
        "old_independent_tail_join_ask_net": old_net,
        "one_net_position_first_fill_cancel_net": net,
        "difference_vs_old": net - old_net,
        "tickers_selected": int(len(detail)),
        "full_q5": int(detail["full_q5"].sum()),
        "partial_q5": int((~detail["full_q5"]).sum()),
        "entry_filled_qty": entry_qty,
        "passive_exit_qty": pqty,
        "passive_exit_fraction": pqty / entry_qty if entry_qty > EPS else np.nan,
        "terminal_coverage": (pqty + m5q) / entry_qty if entry_qty > EPS else np.nan,
        "residual_zero_valued": resid,
        "pnl_per_filled_contract": net / entry_qty if entry_qty > EPS else np.nan,
        "opposite_tail_cancel_race_diagnostic": races,
        "trade_cache": trade_source,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
    }

    out = _new_output(source.name)
    detail.to_csv(out / "one_net_position_q5_detail.csv", index=False)
    OOS._atomic_json(out / "summary.json", summary)

    if show:
        print("=" * 108)
        print("DEEP-TAIL LIVE-STRUCTURE REGRESSION — DEVELOPMENT ONLY")
        print("=" * 108)
        print(f"Old independent-tail V4 JOIN_ASK:     ${old_net:+.4f}")
        print(f"One-net-position first-fill-cancel:   ${net:+.4f}")
        print(f"Difference:                           ${net-old_net:+.4f}")
        print(f"Selected tickers:                     {len(detail)}")
        print(f"Full / partial Q5:                    {int(detail['full_q5'].sum())} / {int((~detail['full_q5']).sum())}")
        print(f"Entered contracts:                    {entry_qty:.1f}")
        print(f"PnL / filled contract:                {100.0*summary['pnl_per_filled_contract']:+.2f}c")
        print(f"Passive exit fraction:                {100.0*summary['passive_exit_fraction']:.1f}%")
        print(f"Terminal coverage:                    {100.0*summary['terminal_coverage']:.1f}%")
        print(f"Unproven residual:                    {resid:.1f}")
        print("Opposite-tail strict-through during hypothetical cancel window:")
        for ms in CANCEL_RACE_MS:
            z = races[str(ms)]
            print(f"  {ms:3d}ms: {z['tickers']} tickers | {z['qty']:.2f} contracts")
        print("DEVELOPMENT STRUCTURAL CHECK ONLY — NOT VALIDATION")
        print("NO API CALLED | NO ORDERS SENT")
        print("Output:", out)

    return {"summary": summary, "detail": detail, "output_dir": str(out)}


__all__ = ["VERSION", "run_structure_regression"]
