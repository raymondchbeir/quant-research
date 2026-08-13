from __future__ import annotations

"""Inside-spread Q100 development replay on compact 1 Hz Kalshi recordings.

This is a NEW exploratory mechanism test. It does not validate the previously
frozen 4c strategy. The compact recording supplied here becomes development
material for this new hypothesis.

Pre-specified scenarios:
- natural YES spread == 3c -> quote one tick inside each side -> our spread 1c
- natural YES spread == 4c -> quote one tick inside each side -> our spread 2c
- natural YES spread == 5c -> quote one tick inside each side -> our spread 3c

Fixed mechanics:
- M1-M5 only;
- 100 contracts displayed per side;
- BID = natural best bid + 1c; ASK = natural best ask - 1c;
- queue ahead is zero because the hypothetical order improves BBO;
- ONLY recorded aggressive trades can fill the hypothetical order;
- BID fills when a recorded aggressive YES sell traded at/below our bid;
- ASK fills when a recorded aggressive YES buy traded at/above our ask;
- BBO moves without a recorded aggressive trade never manufacture a fill;
- after ANY partial/full fill, cancel residual same-side size and apply 3s cooldown;
- 3s momentum and prior-5s aggressive-flow guards remain unchanged;
- inventory limits are scaled 100x from V1: soft 200, hard projected cap 300;
- public 1 Hz BBO is treated as exogenous (our hypothetical size has no feedback).

Because Q100 at an improved BBO can affect subsequent market behavior, this is
necessarily a counterfactual replay. Reported results should be treated as a
mechanism diagnostic, not executable PnL or OOS validation.
"""

import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_m1_m5_feasibility as F
from . import mm_reconstructed_m1_m5_backtest as B
from . import mm_defensive_m1_m5_v1 as D
from . import mm_exact_quote_lifetime_m1_m5_v1 as L
from . import mm_exact_min_spread_m1_m5_v1 as S
from . import mm_oos_4c_audit_replay as O
from . import mm_oos_4c_compact_recorder_v2 as R

STUDY_VERSION = "M1_M5_INSIDE_SPREAD_Q100_DEV_V1"
EPS = 1e-9
QUOTE_QTY = 100.0
IMPROVE_C = 1.0
SOFT_INVENTORY = 200.0
HARD_INVENTORY = 300.0
SPREAD_TOL_C = 0.05
MARKOUTS = (5, 15, 30, 60)
QUALITY_GATE_PCT = 80.0

SCENARIOS = (
    {"scenario": "NAT3_TO_1", "natural_spread_c": 3.0, "our_spread_c": 1.0},
    {"scenario": "NAT4_TO_2", "natural_spread_c": 4.0, "our_spread_c": 2.0},
    {"scenario": "NAT5_TO_3", "natural_spread_c": 5.0, "our_spread_c": 3.0},
)


def _wavg(df, col):
    if df.empty or col not in df:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce")
    w = pd.to_numeric(df["qty"], errors="coerce")
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def _match_spread(s, target):
    return B._valid_sample(s) and abs(float(s.spread_c) - float(target)) <= SPREAD_TOL_C


def _quote_price(side, s):
    improve = IMPROVE_C / 100.0
    return float(s.bid1 + improve) if side == "BID" else float(s.ask1 - improve)


def _policy_reasons(side, s, inventory, last_fill_ts, now_t, mom, flow, target_spread_c):
    reasons = []
    if not _match_spread(s, target_spread_c):
        reasons.append("NATURAL_SPREAD")
    if side == "BID":
        if mom <= -D.DEFAULT_POLICY["max_adverse_momentum_c"] + EPS:
            reasons.append("MOMENTUM")
        if flow <= -D.DEFAULT_POLICY["max_adverse_flow_imbalance"] + EPS:
            reasons.append("FLOW")
        if inventory >= SOFT_INVENTORY - EPS:
            reasons.append("INVENTORY_SOFT")
        if inventory + QUOTE_QTY > HARD_INVENTORY + EPS:
            reasons.append("INVENTORY_HARD")
    else:
        if mom >= D.DEFAULT_POLICY["max_adverse_momentum_c"] - EPS:
            reasons.append("MOMENTUM")
        if flow >= D.DEFAULT_POLICY["max_adverse_flow_imbalance"] - EPS:
            reasons.append("FLOW")
        if inventory <= -SOFT_INVENTORY + EPS:
            reasons.append("INVENTORY_SOFT")
        if inventory - QUOTE_QTY < -HARD_INVENTORY - EPS:
            reasons.append("INVENTORY_HARD")
    if now_t < last_fill_ts.get(side, -np.inf) + D.DEFAULT_POLICY["cooldown_s"] - EPS:
        reasons.append("COOLDOWN")
    return reasons


def _simulate_contract(ticker, meta, samples, trades, target_spread_c, our_spread_c):
    close = float(meta["close_ts"])
    wstart = close - 840.0
    wend = close - 600.0
    series = meta["series"]
    samples = sorted([s for s in samples if wstart <= s.t < wend], key=lambda z: z.t)
    if not samples:
        return [], [], None, Counter()
    times = [s.t for s in samples]
    trade_times = [tr.t for tr in trades]
    tr_idx = B.bisect.bisect_left(trade_times, samples[0].t)

    active = {"BID": None, "ASK": None}
    remaining = {"BID": 0.0, "ASK": 0.0}
    last_fill_ts = {"BID": -np.inf, "ASK": -np.inf}
    episodes, fills = [], []
    counts = Counter()
    mid_hist, flow_hist = deque(), deque()
    inventory = cash = max_abs_inventory = 0.0
    current_sample = None
    current_mid = np.nan
    last_valid_mid = np.nan
    episode_id = 0
    last_sample_t = samples[0].t

    def features(now_t):
        D._purge_flow_history(flow_hist, now_t, D.DEFAULT_POLICY["flow_lookback_s"])
        return D._momentum_c(mid_hist), D._flow_imbalance(flow_hist)

    def cancel(side, t, reason):
        ep = active[side]
        if ep is not None:
            B._close_episode(ep, t, reason)
            counts[f"{side}_CANCEL_{reason}"] += 1
        active[side] = None
        remaining[side] = 0.0

    def open_order(side, s, mom, flow):
        nonlocal episode_id
        px = _quote_price(side, s)
        other = _quote_price("ASK" if side == "BID" else "BID", s)
        if (side == "BID" and px >= other - EPS) or (side == "ASK" and other >= px - EPS):
            counts[f"{side}_BLOCK_CROSSED_OURS"] += 1
            return
        episode_id += 1
        ep = {
            "ticker": ticker, "series": series,
            "episode_id": f"{ticker}:{target_spread_c:g}c:{side}:{episode_id}",
            "scenario": f"NAT{target_spread_c:g}_TO_{our_spread_c:g}",
            "side": side, "join_ts": s.t, "join_time": B._iso(s.t), "join_minute": s.minute,
            "price": px, "queue_ahead_initial": 0.0, "queue_ahead_final": 0.0,
            "natural_bid_at_join": s.bid1, "natural_ask_at_join": s.ask1,
            "natural_spread_c_at_join": s.spread_c, "our_spread_c": our_spread_c,
            "mid_at_join": s.mid, "momentum_3s_c_at_join": mom,
            "flow_imbalance_5s_at_join": flow, "inventory_at_join": inventory,
            "fill_qty": 0.0, "first_fill_ts": np.nan, "last_fill_ts": np.nan,
            "fill_latency_s": np.nan, "end_ts": np.nan, "end_time": None, "end_reason": None,
        }
        episodes.append(ep)
        active[side] = ep
        remaining[side] = QUOTE_QTY
        counts[f"{side}_OPEN"] += 1

    def enforce(now_t, s, allow_open):
        if s is None or not B._valid_sample(s):
            cancel("BID", now_t, "INVALID_BOOK"); cancel("ASK", now_t, "INVALID_BOOK")
            return
        mom, flow = features(now_t)
        for side in ("BID", "ASK"):
            reasons = _policy_reasons(side, s, inventory, last_fill_ts, now_t, mom, flow, target_spread_c)
            if reasons:
                for r in reasons:
                    counts[f"{side}_BLOCK_{r}"] += 1
                if active[side] is not None:
                    cancel(side, now_t, "POLICY_" + reasons[0])
            elif allow_open and active[side] is None:
                open_order(side, s, mom, flow)

    def fill_order(side, tr, fill_mid):
        nonlocal inventory, cash, max_abs_inventory
        ep = active[side]
        if ep is None or not np.isfinite(fill_mid):
            return False
        qpx = float(ep["price"])
        if side == "BID":
            crossed = tr.taker_book_side == "ask" and tr.yes_price <= qpx + EPS
        else:
            crossed = tr.taker_book_side == "bid" and tr.yes_price >= qpx - EPS
        if not crossed:
            return False
        qty = min(float(remaining[side]), float(tr.qty))
        if qty <= EPS:
            return False

        ep["fill_qty"] += qty
        if not np.isfinite(B._f(ep.get("first_fill_ts"))):
            ep["first_fill_ts"] = tr.t
            ep["fill_latency_s"] = tr.t - ep["join_ts"]
        ep["last_fill_ts"] = tr.t
        remaining[side] -= qty

        sign = 1.0 if side == "BID" else -1.0
        fill = {
            "ticker": ticker, "series": series, "scenario": ep["scenario"],
            "episode_id": ep["episode_id"], "side": side,
            "fill_ts": tr.t, "fill_time": B._iso(tr.t),
            "fill_minute": (tr.t - (close - 900.0)) / 60.0,
            "qty": qty, "price": qpx, "mid_at_fill": fill_mid,
            "gross_edge_at_fill_c": sign * (fill_mid - qpx) * 100.0,
            "queue_ahead_initial": 0.0, "fill_latency_s": ep["fill_latency_s"],
            "natural_spread_c_at_join": ep["natural_spread_c_at_join"],
            "our_spread_c": our_spread_c,
            "momentum_3s_c_at_join": ep["momentum_3s_c_at_join"],
            "flow_imbalance_5s_at_join": ep["flow_imbalance_5s_at_join"],
            "inventory_at_join": ep["inventory_at_join"],
            "historical_trade_price": tr.yes_price, "historical_trade_qty": tr.qty,
            "historical_trade_participation_pct": 100.0 * qty / tr.qty if tr.qty > EPS else np.nan,
        }
        for h in MARKOUTS:
            fs = B._future_valid_sample(samples, times, tr.t + h, max_lag_s=2.0)
            if fs is None:
                fill[f"future_mid_{h}s"] = np.nan
                fill[f"markout_{h}s_c"] = np.nan
                fill[f"post_mid_move_{h}s_c"] = np.nan
            else:
                fill[f"future_mid_{h}s"] = fs.mid
                fill[f"markout_{h}s_c"] = sign * (fs.mid - qpx) * 100.0
                fill[f"post_mid_move_{h}s_c"] = sign * (fs.mid - fill_mid) * 100.0
        fills.append(fill)

        if side == "BID": inventory += qty; cash -= qpx * qty
        else: inventory -= qty; cash += qpx * qty
        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        last_fill_ts[side] = tr.t
        counts[f"{side}_FILL_EVENT"] += 1
        B._close_episode(ep, tr.t, "FILLED_COOLDOWN")
        active[side] = None; remaining[side] = 0.0
        return True

    for i, s in enumerate(samples):
        if i > 0:
            while tr_idx < len(trades) and trades[tr_idx].t <= s.t + EPS:
                tr = trades[tr_idx]
                if tr.t > last_sample_t + EPS:
                    fill_order("BID", tr, current_mid)
                    fill_order("ASK", tr, current_mid)
                    signed = tr.qty if tr.taker_book_side == "bid" else -tr.qty if tr.taker_book_side == "ask" else 0.0
                    if abs(signed) > EPS:
                        flow_hist.append((tr.t, float(signed)))
                        D._purge_flow_history(flow_hist, tr.t, D.DEFAULT_POLICY["flow_lookback_s"])
                    enforce(tr.t, current_sample, False)
                tr_idx += 1
            last_sample_t = s.t

        if not B._valid_sample(s):
            current_sample = None; current_mid = np.nan; mid_hist.clear()
            cancel("BID", s.t, "INVALID_BOOK"); cancel("ASK", s.t, "INVALID_BOOK")
            continue

        current_sample = s; current_mid = s.mid; last_valid_mid = s.mid
        mid_hist.append((s.t, s.mid))
        D._purge_mid_history(mid_hist, s.t, D.DEFAULT_POLICY["momentum_lookback_s"])

        # Natural BBO is exogenous. A sample change only cancels/reprices; it never creates a fill.
        for side in ("BID", "ASK"):
            ep = active[side]
            if ep is not None:
                desired = _quote_price(side, s) if _match_spread(s, target_spread_c) else np.nan
                if not np.isfinite(desired) or abs(float(ep["price"]) - desired) > EPS:
                    cancel(side, s.t, "NATURAL_BBO_REPRICE")
        enforce(s.t, s, True)

    end_t = min(wend, samples[-1].t)
    cancel("BID", end_t, "M5_END"); cancel("ASK", end_t, "M5_END")
    if not np.isfinite(last_valid_mid):
        return episodes, fills, None, counts

    final_mid = last_valid_mid
    net = cash + inventory * final_mid
    gross = sum((f["gross_edge_at_fill_c"] / 100.0) * f["qty"] for f in fills)
    matched_qty, matched_pnl = B._match_roundtrips(fills)
    bid_qty = sum(f["qty"] for f in fills if f["side"] == "BID")
    ask_qty = sum(f["qty"] for f in fills if f["side"] == "ASK")
    contract = {
        "ticker": ticker, "series": series, "close_time": B._iso(close), "close_ts": close,
        "scenario": f"NAT{target_spread_c:g}_TO_{our_spread_c:g}",
        "natural_spread_c": target_spread_c, "our_spread_c": our_spread_c,
        "sample_rows": len(samples), "valid_sample_rows": sum(B._valid_sample(x) for x in samples),
        "bid_fill_qty": bid_qty, "ask_fill_qty": ask_qty, "fill_qty": bid_qty + ask_qty,
        "both_sides_filled": bid_qty > EPS and ask_qty > EPS,
        "one_sided_fill": (bid_qty > EPS) ^ (ask_qty > EPS),
        "max_abs_inventory": max_abs_inventory, "ending_inventory_yes_equiv": inventory,
        "final_mid_m5": final_mid, "cash": cash,
        "gross_spread_capture_dollars": gross,
        "adverse_selection_to_m5_dollars": net - gross,
        "net_mtm_pnl_before_fees": net,
        "matched_roundtrip_qty": matched_qty, "matched_roundtrip_pnl": matched_pnl,
    }
    return episodes, fills, contract, counts


def _headline(scenario, contract_df, fills_df, windows):
    qty = fills_df.qty.sum() if len(fills_df) else 0.0
    pnl = contract_df.net_mtm_pnl_before_fees.sum() if len(contract_df) else 0.0
    row = {
        "scenario": scenario, "eligible_contracts": len(contract_df), "independent_windows": len(windows),
        "contracts_with_fill": int((contract_df.fill_qty > EPS).sum()), "fill_events": len(fills_df),
        "fill_qty": qty, "net_mtm_pnl_before_fees": pnl,
        "pnl_per_window": pnl / len(windows) if len(windows) else np.nan,
        "gross_capture_dollars": contract_df.gross_spread_capture_dollars.sum(),
        "adverse_selection_to_m5_dollars": contract_df.adverse_selection_to_m5_dollars.sum(),
        "matched_roundtrip_pnl": contract_df.matched_roundtrip_pnl.sum(),
        "qty_weighted_gross_edge_c": _wavg(fills_df, "gross_edge_at_fill_c"),
        "avg_fill_qty": fills_df.qty.mean() if len(fills_df) else np.nan,
        "p95_fill_qty": fills_df.qty.quantile(.95) if len(fills_df) else np.nan,
        "p95_max_abs_inventory": contract_df.max_abs_inventory.quantile(.95),
        "worst_window_pnl": windows.net_mtm_pnl_before_fees.min(),
        "max_drawdown": windows.drawdown.min(),
        "break_even_fee_c_per_fill_qty": 100.0 * pnl / qty if qty > EPS else np.nan,
        "avg_historical_trade_participation_pct": fills_df.historical_trade_participation_pct.mean() if len(fills_df) else np.nan,
    }
    for h in MARKOUTS:
        row[f"qty_weighted_markout_{h}s_c"] = _wavg(fills_df, f"markout_{h}s_c")
    return row


def _side(scenario, fills_df):
    rows = []
    for side in ("BID", "ASK", "ALL"):
        z = fills_df if side == "ALL" else fills_df[fills_df.side == side]
        if z.empty: continue
        r = {"scenario": scenario, "side": side, "fill_events": len(z), "fill_qty": z.qty.sum(),
             "avg_fill_qty": z.qty.mean(), "qty_weighted_gross_edge_c": _wavg(z, "gross_edge_at_fill_c")}
        for h in MARKOUTS: r[f"qty_weighted_markout_{h}s_c"] = _wavg(z, f"markout_{h}s_c")
        rows.append(r)
    return pd.DataFrame(rows)


def run_inside_spread_q100_development(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    manifest, frozen, duration_h = O._verify(session)
    meta = O._metadata(session)
    samples, info, duplicates, _ = O._bbo(session, meta)
    trades, _ = O._trades(session, meta)
    audit = O._audit(meta, samples, info, duplicates, trades)
    good = audit[audit.quality_ok].copy()
    if good.empty: raise RuntimeError("No contracts pass the existing 80% quality gate")
    sim_meta = {str(r.ticker): {"ticker": str(r.ticker), "series": str(r.series), "close_ts": float(r.close_ts)} for r in good.itertuples(index=False)}
    targets = sorted(sim_meta, key=lambda t: (sim_meta[t]["close_ts"], t))

    all_head, all_side, all_chrono, all_contracts, all_fills, all_eps, all_counts = [], [], [], [], [], [], []
    print(f"Inside-spread development replay | {len(targets)} quality contracts | {good.close_ts.nunique()} windows")
    for sc in SCENARIOS:
        name, natural, ours = sc["scenario"], sc["natural_spread_c"], sc["our_spread_c"]
        print(f"\n{name}: natural={natural:.0f}c -> ours={ours:.0f}c | Q100/side")
        contracts, fills, eps, counts = [], [], [], Counter()
        for i, ticker in enumerate(targets, 1):
            e, f, c, k = _simulate_contract(ticker, sim_meta[ticker], samples.get(ticker, []), trades.get(ticker, []), natural, ours)
            eps.extend(e); fills.extend(f); counts.update(k)
            if c is not None: contracts.append(c)
            if i % 100 == 0 or i == len(targets): print(f"  replay {i}/{len(targets)} | fills={len(fills)} | qty={sum(x['qty'] for x in fills):.1f}")
        cdf, fdf, edf = pd.DataFrame(contracts), pd.DataFrame(fills), pd.DataFrame(eps)
        wdf = L._window_summary(cdf)
        chrono = S._chronological_detail(wdf, name)
        head = _headline(name, cdf, fdf, wdf)
        all_head.append(head); all_side.append(_side(name, fdf)); all_chrono.append(chrono)
        if len(cdf): all_contracts.append(cdf)
        if len(fdf): all_fills.append(fdf)
        if len(edf): all_eps.append(edf)
        all_counts.append(pd.DataFrame([{"scenario": name, "reason": k, "count": v} for k, v in counts.most_common()]))

    headline = pd.DataFrame(all_head)
    side = pd.concat(all_side, ignore_index=True) if all_side else pd.DataFrame()
    chrono = pd.concat(all_chrono, ignore_index=True) if all_chrono else pd.DataFrame()
    contracts = pd.concat(all_contracts, ignore_index=True) if all_contracts else pd.DataFrame()
    fills = pd.concat(all_fills, ignore_index=True) if all_fills else pd.DataFrame()
    episodes = pd.concat(all_eps, ignore_index=True) if all_eps else pd.DataFrame()
    counts = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()

    out = Path(output_dir) if output_dir else R.PROJECT_ROOT / "results" / "kalshi_inside_spread_q100_dev" / f"{session.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    for n, df in {"headline_summary": headline, "chronological_robustness": chrono, "side_summary": side,
                  "contract_summary": contracts, "fills": fills, "quote_episodes": episodes,
                  "policy_counts": counts, "data_quality": audit}.items():
        df.to_csv(out / f"{n}.csv", index=False)
    (out / "study_config.json").write_text(json.dumps({
        "study_version": STUDY_VERSION, "session": str(session), "session_duration_hours": duration_h,
        "quote_qty": QUOTE_QTY, "improve_each_side_c": IMPROVE_C,
        "inventory_soft": SOFT_INVENTORY, "inventory_hard": HARD_INVENTORY,
        "scenarios": SCENARIOS, "quality_gate_pct": QUALITY_GATE_PCT,
        "fill_rule": "recorded aggressive trades only; no fills from BBO moves",
        "market_feedback": "ignored; public 1Hz BBO treated as exogenous",
        "fees": "excluded", "status": "development mechanism diagnostic, not OOS",
    }, indent=2), encoding="utf-8")

    if show:
        cols = ["scenario", "fill_events", "fill_qty", "net_mtm_pnl_before_fees", "pnl_per_window",
                "qty_weighted_gross_edge_c", "qty_weighted_markout_5s_c", "qty_weighted_markout_15s_c",
                "qty_weighted_markout_30s_c", "qty_weighted_markout_60s_c", "p95_max_abs_inventory",
                "worst_window_pnl", "max_drawdown", "break_even_fee_c_per_fill_qty",
                "avg_historical_trade_participation_pct"]
        print("\n" + "="*132)
        print("INSIDE-SPREAD Q100 DEVELOPMENT REPLAY — RECORDED AGGRESSIVE FLOW ONLY")
        print("="*132)
        print(headline[cols].round(4).to_string(index=False))
        print("\nCHRONOLOGY\n", chrono.round(4).to_string(index=False))
        print("\nSIDE\n", side.round(4).to_string(index=False))
        print("\nCAVEAT: Q100 improves BBO and can change future behavior; historical public BBO is treated as exogenous.")
        print("This recording is now DEVELOPMENT data for this new hypothesis. A second already-seen recording can test cross-sample robustness, not true OOS.")
        print("Outputs:", out)
        print("="*132)
    return {"output_dir": out, "headline": headline, "chronology": chrono, "side": side,
            "contracts": contracts, "fills": fills, "episodes": episodes, "audit": audit}
