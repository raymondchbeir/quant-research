from __future__ import annotations

"""Audit + exact frozen 4c OOS replay for compact 1 Hz recordings.

No tuning is performed. The module verifies the precommitted recorder/strategy,
applies the existing 80% M1-M5 validity gate, converts compact BBO/trade rows
to the same objects used by Defensive V1, then calls that exact simulator with
min_spread_c fixed at 4.00c.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_m1_m5_feasibility as F
from . import mm_reconstructed_m1_m5_backtest as B
from . import mm_defensive_m1_m5_v1 as D
from . import mm_exact_quote_lifetime_m1_m5_v1 as L
from . import mm_exact_min_spread_m1_m5_v1 as S
from . import mm_oos_4c_compact_recorder_v2 as R

STUDY_VERSION = "MM4C_FROZEN_OOS_AUDIT_REPLAY_V1"
QUALITY_GATE_PCT = 80.0
EXPECTED_SECONDS = 240
MARKOUTS = (5, 15, 30, 60)
EPS = 1e-9


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify(session):
    manifest = _load_json(session / "session_manifest.json")
    frozen = _load_json(session / "frozen_strategy.json")
    expected = {
        "min_spread_c": 4.0,
        "quote_qty": 1.0,
        "momentum_lookback_s": 3.0,
        "max_adverse_momentum_c": 1.0,
        "flow_lookback_s": 5.0,
        "max_adverse_flow_imbalance": 0.60,
        "inventory_soft_limit": 2.0,
        "inventory_hard_limit": 3.0,
        "same_side_cooldown_s": 3.0,
    }
    errors = []
    if manifest.get("study_version") != R.STUDY_VERSION:
        errors.append(f"wrong recorder version: {manifest.get('study_version')}")
    if abs(_f(manifest.get("market_state_interval_seconds")) - 1.0) > EPS:
        errors.append("market state was not 1 Hz")
    if frozen.get("status") != "FROZEN_PROSPECTIVE_OOS":
        errors.append("strategy was not marked frozen prospective OOS")
    if set(frozen.get("universe") or []) != set(R.CRYPTO_SERIES):
        errors.append("universe differs from frozen universe")
    for k, v in expected.items():
        if abs(_f(frozen.get(k)) - v) > EPS:
            errors.append(f"{k}={frozen.get(k)!r}, expected {v}")
    if errors:
        raise RuntimeError("Freeze verification failed:\n - " + "\n - ".join(errors))
    s = F._ts_seconds(manifest.get("started_at"))
    e = F._ts_seconds(manifest.get("ended_at"))
    duration_h = (e - s) / 3600.0 if np.isfinite(s) and np.isfinite(e) else _f(manifest.get("actual_duration_hours"))
    return manifest, frozen, duration_h


def _metadata(session):
    meta = {}
    p = session / "market_metadata.jsonl"
    with p.open("rb") as f:
        for raw in f:
            try:
                x = json.loads(raw)
            except Exception:
                continue
            t = str(x.get("ticker") or "")
            c = F._ts_seconds(x.get("close_time"))
            if t and np.isfinite(c):
                meta[t] = {"ticker": t, "series": str(x.get("series_ticker") or t.split("-")[0]), "close_ts": float(c)}
    return meta


def _bbo(session, meta):
    samples = defaultdict(dict)
    info = defaultdict(dict)
    duplicates = Counter()
    lines = rows = 0
    with (session / "bbo_1hz.jsonl").open("rb") as f:
        for raw in f:
            lines += 1
            try:
                x = json.loads(raw)
            except Exception:
                continue
            ts = _f(x.get("epoch_second"))
            if not np.isfinite(ts):
                ts = F._ts_seconds(x.get("time"))
            if not np.isfinite(ts):
                continue
            for m in x.get("markets") or []:
                ticker = str(m.get("ticker") or "")
                close = F._ts_seconds(m.get("close_time"))
                if ticker and np.isfinite(close):
                    meta[ticker] = {"ticker": ticker, "series": str(m.get("series_ticker") or ticker.split("-")[0]), "close_ts": float(close)}
                if ticker not in meta:
                    continue
                elapsed = _f(m.get("elapsed_seconds"), ts - (meta[ticker]["close_ts"] - 900.0))
                bid, ask = _f(m.get("yes_bid")), _f(m.get("yes_ask"))
                bq, aq = _f(m.get("yes_bid_size")), _f(m.get("yes_ask_size"))
                valid = bool(m.get("valid")) and all(np.isfinite(z) for z in (bid, ask, bq, aq)) and 0 <= bid < ask <= 1 and bq >= 0 and aq >= 0
                s = B.Sample(
                    t=float(ts), minute=float(elapsed / 60.0), status="VALID" if valid else "INVALID",
                    bid1=bid, bid1_qty=max(0.0, bq) if np.isfinite(bq) else 0.0,
                    ask1=ask, ask1_qty=max(0.0, aq) if np.isfinite(aq) else 0.0,
                    mid=(bid + ask) / 2.0 if valid else np.nan,
                    spread_c=100.0 * (ask - bid) if valid else np.nan,
                )
                if ts in samples[ticker]:
                    duplicates[ticker] += 1
                samples[ticker][float(ts)] = s
                info[ticker][float(ts)] = {"recorder_valid": bool(m.get("valid")), "source_age_ms": _f(m.get("source_age_ms"))}
                rows += 1
    out = {k: sorted(v.values(), key=lambda z: z.t) for k, v in samples.items()}
    return out, info, duplicates, {"bbo_lines": lines, "bbo_market_rows": rows}


def _trades(session, meta):
    out = defaultdict(list)
    lines = valid = 0
    with (session / "aggressive_trades_compact.jsonl").open("rb") as f:
        for raw in f:
            lines += 1
            try:
                x = json.loads(raw)
            except Exception:
                continue
            ticker = str(x.get("ticker") or "")
            t, p, q = F._ts_seconds(x.get("time")), _f(x.get("yes_price")), _f(x.get("qty"))
            side = str(x.get("taker_book_side") or "").lower()
            if ticker in meta and np.isfinite(t) and np.isfinite(p) and np.isfinite(q) and q > 0 and 0 <= p <= 1 and side in {"bid", "ask"}:
                out[ticker].append(F.Trade(float(t), float(p), float(q), side))
                valid += 1
    for k in out:
        out[k].sort(key=lambda z: z.t)
    return out, {"trade_lines": lines, "valid_trades": valid}


def _event_counts(session):
    c = Counter()
    p = session / "connection_events.jsonl"
    if p.exists():
        with p.open("rb") as f:
            for raw in f:
                try:
                    c[str(json.loads(raw).get("type") or "unknown")] += 1
                except Exception:
                    c["decode_error"] += 1
    return c


def _audit(meta, samples, info, duplicates, trades):
    rows = []
    for ticker, m in sorted(meta.items(), key=lambda kv: (kv[1]["close_ts"], kv[0])):
        start, end = m["close_ts"] - 840.0, m["close_ts"] - 600.0
        ss = [s for s in samples.get(ticker, []) if start <= s.t < end]
        vv = [s for s in ss if B._valid_sample(s)]
        ages = [_f(info[ticker].get(s.t, {}).get("source_age_ms")) for s in ss]
        ages = [x for x in ages if np.isfinite(x)]
        times = [s.t for s in vv]
        gaps = [times[0] - start] + list(np.diff(times)) + [end - times[-1]] if times else []
        valid_cov = 100.0 * len(vv) / EXPECTED_SECONDS
        tr = [x for x in trades.get(ticker, []) if start <= x.t < end]
        rows.append({
            "ticker": ticker, "series": m["series"], "close_ts": m["close_ts"], "close_time": B._iso(m["close_ts"]),
            "observed_seconds": len(ss), "strict_valid_seconds": len(vv),
            "row_coverage_pct": 100.0 * len(ss) / EXPECTED_SECONDS,
            "valid_coverage_pct": valid_cov, "duplicate_seconds": int(duplicates.get(ticker, 0)),
            "max_valid_gap_s": max(gaps) if gaps else np.nan,
            "median_source_age_ms": float(np.median(ages)) if ages else np.nan,
            "p95_source_age_ms": float(np.quantile(ages, .95)) if ages else np.nan,
            "m1_m5_trade_events": len(tr), "m1_m5_trade_qty": sum(x.qty for x in tr),
            "quality_ok": valid_cov >= QUALITY_GATE_PCT - EPS,
        })
    return pd.DataFrame(rows)


def _asset(contract_df):
    return contract_df.groupby("series", as_index=False).agg(
        eligible_contracts=("ticker", "count"), fill_qty=("fill_qty", "sum"),
        net_mtm_pnl_before_fees=("net_mtm_pnl_before_fees", "sum"),
        avg_max_abs_inventory=("max_abs_inventory", "mean"),
    ).sort_values("net_mtm_pnl_before_fees", ascending=False)


def _side(fills_df):
    rows = []
    for side in ("BID", "ASK", "ALL"):
        z = fills_df if side == "ALL" else fills_df[fills_df.side == side]
        if z.empty:
            continue
        r = {"side": side, "fill_events": len(z), "fill_qty": z.qty.sum(), "avg_gross_edge_c": z.gross_edge_at_fill_c.mean()}
        for h in MARKOUTS:
            r[f"avg_markout_{h}s_c"] = pd.to_numeric(z[f"markout_{h}s_c"], errors="coerce").mean()
        rows.append(r)
    return pd.DataFrame(rows)


def _headline(contract_df, fills_df, windows, chrono):
    pnl = contract_df.net_mtm_pnl_before_fees.sum()
    qty = fills_df.qty.sum() if len(fills_df) else 0.0
    first = chrono[chrono.split == "FIRST_HALF"]
    second = chrono[chrono.split == "SECOND_HALF"]
    r = {
        "eligible_contracts": len(contract_df), "independent_windows": len(windows),
        "contracts_with_fill": int((contract_df.fill_qty > EPS).sum()), "fill_events": len(fills_df), "fill_qty": qty,
        "gross_capture_dollars": contract_df.gross_spread_capture_dollars.sum(),
        "adverse_selection_to_m5_dollars": contract_df.adverse_selection_to_m5_dollars.sum(),
        "net_mtm_pnl_before_fees": pnl, "pnl_per_window": pnl / len(windows) if len(windows) else np.nan,
        "pnl_c_per_filled_qty": 100.0 * pnl / qty if qty > EPS else np.nan,
        "matched_roundtrip_pnl": contract_df.matched_roundtrip_pnl.sum(),
        "avg_gross_edge_at_fill_c": fills_df.gross_edge_at_fill_c.mean() if len(fills_df) else np.nan,
        "avg_spread_at_join_c": fills_df.spread_c_at_join.mean() if len(fills_df) else np.nan,
        "p95_max_abs_inventory": contract_df.max_abs_inventory.quantile(.95),
        "worst_window_pnl": windows.net_mtm_pnl_before_fees.min(), "max_drawdown": windows.drawdown.min(),
        "break_even_fee_c_per_fill_qty": 100.0 * pnl / qty if qty > EPS else np.nan,
        "first_half_pnl": first.net_pnl.iloc[0] if len(first) else np.nan,
        "second_half_pnl": second.net_pnl.iloc[0] if len(second) else np.nan,
    }
    for h in MARKOUTS:
        r[f"avg_markout_{h}s_c"] = pd.to_numeric(fills_df[f"markout_{h}s_c"], errors="coerce").mean() if len(fills_df) else np.nan
    return pd.DataFrame([r])


def run_frozen_mm4c_oos_audit_replay(session_dir, development_4c_dir=None, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    manifest, frozen, duration_h = _verify(session)
    print(f"Freeze verification: PASS | recorder={manifest.get('study_version')} | duration={duration_h:.3f}h")

    meta = _metadata(session)
    samples, info, duplicates, bbo_stats = _bbo(session, meta)
    trades, trade_stats = _trades(session, meta)
    events = _event_counts(session)
    audit = _audit(meta, samples, info, duplicates, trades)
    good = audit[audit.quality_ok].copy()
    if good.empty:
        raise RuntimeError("No contracts passed the fixed 80% M1-M5 1 Hz quality gate")

    audit_summary = pd.DataFrame([{
        "session_duration_hours": duration_h, "clean_stop": bool(manifest.get("ended_at")),
        "discovered_contracts": len(audit), "quality_pass_contracts": len(good),
        "quality_pass_pct": 100.0 * len(good) / len(audit), "quality_pass_windows": good.close_ts.nunique(),
        "median_valid_coverage_pct": audit.valid_coverage_pct.median(), "median_max_valid_gap_s": audit.max_valid_gap_s.median(),
        "bbo_lines": bbo_stats["bbo_lines"], "bbo_market_rows": bbo_stats["bbo_market_rows"],
        "trade_lines": trade_stats["trade_lines"], "valid_trades": trade_stats["valid_trades"],
        "connections": events.get("connected", 0), "connection_exceptions": events.get("connection_exception", 0),
        "ws_errors": events.get("ws_error", 0), "discovery_errors": events.get("discovery_error", 0),
    }])

    sim_meta = {str(r.ticker): {"ticker": str(r.ticker), "series": str(r.series), "close_ts": float(r.close_ts), "reconstructed_coverage_pct": float(r.valid_coverage_pct)} for r in good.itertuples(index=False)}
    policy = dict(D.DEFAULT_POLICY); policy["min_spread_c"] = 4.0
    eps_all, fills_all, contracts, counts = [], [], [], Counter()
    targets = sorted(sim_meta, key=lambda t: (sim_meta[t]["close_ts"], t))
    print(f"Audit pass: {len(targets)}/{len(audit)} contracts | {good.close_ts.nunique()} windows")
    for i, ticker in enumerate(targets, 1):
        eps, fills, contract, c = D._simulate_contract(ticker, sim_meta[ticker], samples.get(ticker, []), trades.get(ticker, []), 1.0, MARKOUTS, 2.0, policy)
        eps_all.extend(eps); fills_all.extend(fills); counts.update(c)
        if contract is not None: contracts.append(contract)
        if i % 100 == 0 or i == len(targets): print(f"  replay {i}/{len(targets)} | fills={len(fills_all)}")

    contract_df, fills_df, episodes_df = pd.DataFrame(contracts), pd.DataFrame(fills_all), pd.DataFrame(eps_all)
    windows = L._window_summary(contract_df)
    chrono = S._chronological_detail(windows, "OOS_4C").drop(columns=["scenario"])
    side, asset = _side(fills_df), _asset(contract_df)
    headline = _headline(contract_df, fills_df, windows, chrono)

    comparison = pd.DataFrame()
    if development_4c_dir is not None:
        dev = pd.read_csv(Path(development_4c_dir) / "scenario_summary.csv")
        d = dev[np.isclose(pd.to_numeric(dev.min_spread_c, errors="coerce"), 4.0)].iloc[0]
        o = headline.iloc[0]
        fields = ["net_mtm_pnl_before_fees", "pnl_per_window", "matched_roundtrip_pnl", "avg_gross_edge_at_fill_c", "avg_markout_5s_c", "avg_markout_15s_c", "avg_markout_30s_c", "avg_markout_60s_c", "p95_max_abs_inventory", "worst_window_pnl", "max_drawdown", "break_even_fee_c_per_fill_qty"]
        comparison = pd.DataFrame([{"metric": k, "development_4c": _f(d.get(k)), "oos_4c": _f(o.get(k)), "difference": _f(o.get(k)) - _f(d.get(k))} for k in fields])

    out = Path(output_dir) if output_dir else R.PROJECT_ROOT / "results" / "kalshi_mm4c_frozen_oos" / f"{session.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in {"data_quality_by_contract": audit, "data_quality_summary": audit_summary, "oos_contract_summary": contract_df, "oos_fills": fills_df, "oos_quote_episodes": episodes_df, "oos_window_summary": windows, "chronological_robustness": chrono, "side_summary": side, "asset_summary": asset, "oos_headline_summary": headline, "development_vs_oos": comparison}.items():
        df.to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame([{"reason": k, "count": v} for k, v in counts.most_common()]).to_csv(out / "policy_counts.csv", index=False)
    (out / "study_config.json").write_text(json.dumps({"study_version": STUDY_VERSION, "session": str(session), "frozen_strategy": frozen, "quality_gate_pct": QUALITY_GATE_PCT, "policy": policy, "fees": "excluded", "retuning": "none"}, indent=2, default=str), encoding="utf-8")

    if show:
        a, h = audit_summary.iloc[0], headline.iloc[0]
        print("\n" + "="*118); print("FROZEN 4c MM — FRESH OOS AUDIT + EXACT 1 Hz REPLAY"); print("="*118)
        print(f"AUDIT: duration={a.session_duration_hours:.3f}h | pass={int(a.quality_pass_contracts)}/{int(a.discovered_contracts)} contracts | windows={int(a.quality_pass_windows)} | median coverage={a.median_valid_coverage_pct:.2f}% | connections/exceptions={int(a.connections)}/{int(a.connection_exceptions)}")
        print(f"OOS: fills={int(h.fill_events)} qty={h.fill_qty:.2f} | net=${h.net_mtm_pnl_before_fees:+.4f} | pnl/window=${h.pnl_per_window:+.5f} | cushion={h.break_even_fee_c_per_fill_qty:+.3f}c | DD=${h.max_drawdown:+.4f}")
        print(f"HALVES: ${h.first_half_pnl:+.4f} / ${h.second_half_pnl:+.4f}")
        print("MARKOUTS:", " | ".join(f"{x}s={h[f'avg_markout_{x}s_c']:+.3f}c" for x in MARKOUTS))
        print("\nCHRONOLOGY\n", chrono.round(4).to_string(index=False))
        print("\nSIDE\n", side.round(4).to_string(index=False) if len(side) else "no fills")
        print("\nASSET — diagnostic only, no posthoc filtering\n", asset.round(4).to_string(index=False))
        if len(comparison): print("\nDEVELOPMENT 4c VS OOS\n", comparison.round(4).to_string(index=False))
        print("\nOutputs:", out); print("="*118)

    return {"output_dir": out, "audit_summary": audit_summary, "audit": audit, "headline": headline, "contracts": contract_df, "fills": fills_df, "windows": windows, "chronology": chrono, "side": side, "asset": asset, "development_comparison": comparison}
