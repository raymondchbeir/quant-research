from __future__ import annotations

"""Frozen OOS #1 audit + replay for NAT4->2 with pre-M1 max-range <23c.

Hard-bound to OOS session 20260813_190334 so a later recording cannot be read
accidentally. No threshold, asset, side, sizing, or execution tuning is run.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_m1_m5_feasibility as F
from . import mm_reconstructed_m1_m5_backtest as B
from . import mm_oos_4c_audit_replay as O
from . import mm_oos_4c_compact_recorder_v2 as R
from . import mm_nat4_to_2_inventory_target_dev_v1 as INV
from . import mm_exact_quote_lifetime_m1_m5_v1 as L
from . import mm_exact_min_spread_m1_m5_v1 as S
from . import mm_inside_spread_q100_dev_v1 as Q

STUDY_VERSION = "NAT4_TO_2_PREM1_LT23_FROZEN_OOS1_V1"
EXPECTED_SESSION_NAME = "20260813_190334"
EPS = 1e-9
PRE_RANGE_CUTOFF_C = 23.0
M1_M5_QUALITY_GATE_PCT = 80.0
PRE_CONTRACT_QUALITY_GATE_PCT = 80.0
PRE_WINDOW_CONTRACT_GATE_PCT = 80.0
PRE_EXPECTED_SECONDS = 60
POST_EXPECTED_SECONDS = 60
MARKOUTS = (5, 15, 30, 60)


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_frozen_session(session):
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"This script is frozen to {EXPECTED_SESSION_NAME}; got {session.name}. "
            "The later recording must remain untouched."
        )
    required = [
        "session_manifest.json", "health.json", "bbo_1hz.jsonl",
        "aggressive_trades_compact.jsonl", "market_metadata.jsonl",
        "connection_events.jsonl",
    ]
    missing = [x for x in required if not (session / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required OOS files: {missing}")

    manifest = _load_json(session / "session_manifest.json")
    health = _load_json(session / "health.json")
    errors = []
    if manifest.get("study_version") != R.STUDY_VERSION:
        errors.append(
            f"recorder={manifest.get('study_version')!r}, expected={R.STUDY_VERSION!r}"
        )
    if abs(_f(manifest.get("market_state_interval_seconds")) - 1.0) > EPS:
        errors.append("market_state_interval_seconds was not 1.0")
    if not manifest.get("ended_at"):
        errors.append("manifest has no ended_at; session is not frozen")
    if bool(health.get("running")):
        errors.append("health.json still says running=true")

    s = F._ts_seconds(manifest.get("started_at"))
    e = F._ts_seconds(manifest.get("ended_at"))
    duration_h = (
        (e - s) / 3600.0
        if np.isfinite(s) and np.isfinite(e)
        else _f(manifest.get("actual_duration_hours"))
    )
    if not np.isfinite(duration_h) or duration_h <= 0:
        errors.append("could not verify positive session duration")
    if errors:
        raise RuntimeError("OOS verification failed:\n - " + "\n - ".join(errors))
    return manifest, health, float(duration_h)


def _segment_stats(ticker, meta, samples, info, start_elapsed, end_elapsed, expected):
    close = float(meta["close_ts"])
    contract_start = close - 900.0
    start = contract_start + float(start_elapsed)
    end = contract_start + float(end_elapsed)
    ss = [s for s in samples.get(ticker, []) if start <= s.t < end]
    vv = [s for s in ss if B._valid_sample(s)]
    times = [float(s.t) for s in vv]
    if times:
        gaps = [times[0] - start]
        if len(times) > 1:
            gaps.extend(np.diff(times).tolist())
        gaps.append(end - times[-1])
        max_gap = float(max(gaps))
    else:
        max_gap = np.nan
    ages = [
        _f(info.get(ticker, {}).get(s.t, {}).get("source_age_ms"))
        for s in ss
    ]
    ages = [x for x in ages if np.isfinite(x)]
    mids = np.asarray([float(s.mid) for s in vv], dtype=float)
    mid_range_c = (
        100.0 * (float(mids.max()) - float(mids.min()))
        if len(mids) else np.nan
    )
    return {
        "observed_seconds": len(ss),
        "valid_seconds": len(vv),
        "row_coverage_pct": 100.0 * len(ss) / expected,
        "valid_coverage_pct": 100.0 * len(vv) / expected,
        "max_valid_gap_s": max_gap,
        "median_source_age_ms": float(np.median(ages)) if ages else np.nan,
        "p95_source_age_ms": float(np.quantile(ages, 0.95)) if ages else np.nan,
        "mid_range_c": mid_range_c,
    }


def _full_audit(meta, samples, info, duplicates, trades):
    base = O._audit(meta, samples, info, duplicates, trades)
    if base.empty:
        return base
    rows = []
    for r in base.itertuples(index=False):
        ticker = str(r.ticker)
        m = meta[ticker]
        pre = _segment_stats(ticker, m, samples, info, 0.0, 60.0, 60)
        post = _segment_stats(ticker, m, samples, info, 300.0, 360.0, 60)
        rows.append({
            "ticker": ticker,
            "pre_valid_seconds": pre["valid_seconds"],
            "pre_valid_coverage_pct": pre["valid_coverage_pct"],
            "pre_max_valid_gap_s": pre["max_valid_gap_s"],
            "pre_median_source_age_ms": pre["median_source_age_ms"],
            "pre_p95_source_age_ms": pre["p95_source_age_ms"],
            "pre_mid_range_c": pre["mid_range_c"],
            "pre_feature_ok": bool(
                pre["valid_coverage_pct"] >= PRE_CONTRACT_QUALITY_GATE_PCT - EPS
                and np.isfinite(pre["mid_range_c"])
            ),
            "post_valid_seconds": post["valid_seconds"],
            "post_valid_coverage_pct": post["valid_coverage_pct"],
            "post_max_valid_gap_s": post["max_valid_gap_s"],
        })
    out = base.merge(pd.DataFrame(rows), on="ticker", how="left", validate="one_to_one")
    out["m1_m5_quality_ok"] = (
        pd.to_numeric(out["valid_coverage_pct"], errors="coerce")
        >= M1_M5_QUALITY_GATE_PCT - EPS
    )
    return out


def _window_gate(audit):
    m1 = audit[audit["m1_m5_quality_ok"].astype(bool)].copy()
    rows = []
    for close_ts, g in m1.groupby("close_ts", sort=True):
        pre = g[g["pre_feature_ok"].astype(bool)]
        n_m1, n_pre = len(g), len(pre)
        cov = 100.0 * n_pre / n_m1 if n_m1 else 0.0
        ranges = pd.to_numeric(pre["pre_mid_range_c"], errors="coerce")
        max_range = float(ranges.max()) if ranges.notna().any() else np.nan
        fq = bool(
            np.isfinite(max_range)
            and cov >= PRE_WINDOW_CONTRACT_GATE_PCT - EPS
        )
        passed = bool(fq and max_range < PRE_RANGE_CUTOFF_C - EPS)
        rows.append({
            "close_ts": float(close_ts),
            "close_time": B._iso(float(close_ts)),
            "m1_m5_quality_contracts": n_m1,
            "pre_feature_contracts": n_pre,
            "pre_contract_coverage_pct": cov,
            "pre_m0_m1_max_mid_range_c": max_range,
            "feature_quality_ok": fq,
            "range_pass_lt23c": passed,
            "range_fail_ge23c": bool(fq and not passed),
        })
    return pd.DataFrame(rows).sort_values("close_ts").reset_index(drop=True)


def _audit_summary(duration_h, audit, gate, bbo_stats, trade_stats, events):
    m1 = audit[audit["m1_m5_quality_ok"].astype(bool)]
    return pd.DataFrame([{
        "session": EXPECTED_SESSION_NAME,
        "duration_hours": duration_h,
        "discovered_contracts": len(audit),
        "m1_m5_quality_contracts": len(m1),
        "m1_m5_quality_pct": 100.0 * len(m1) / len(audit) if len(audit) else np.nan,
        "m1_m5_quality_windows": int(m1["close_ts"].nunique()) if len(m1) else 0,
        "feature_quality_windows": int(gate["feature_quality_ok"].sum()) if len(gate) else 0,
        "range_pass_windows_lt23c": int(gate["range_pass_lt23c"].sum()) if len(gate) else 0,
        "range_fail_windows_ge23c": int(gate["range_fail_ge23c"].sum()) if len(gate) else 0,
        "missing_or_low_pre_feature_windows": int((~gate["feature_quality_ok"].astype(bool)).sum()) if len(gate) else 0,
        "median_m1_m5_valid_coverage_pct": float(pd.to_numeric(audit["valid_coverage_pct"], errors="coerce").median()),
        "median_pre_valid_coverage_pct": float(pd.to_numeric(audit["pre_valid_coverage_pct"], errors="coerce").median()),
        "median_post_valid_coverage_pct": float(pd.to_numeric(audit["post_valid_coverage_pct"], errors="coerce").median()),
        "bbo_lines": bbo_stats.get("bbo_lines"),
        "bbo_market_rows": bbo_stats.get("bbo_market_rows"),
        "trade_lines": trade_stats.get("trade_lines"),
        "valid_trades": trade_stats.get("valid_trades"),
        "connections": events.get("connected", 0),
        "connection_exceptions": events.get("connection_exception", 0),
        "ws_errors": events.get("ws_error", 0),
        "discovery_errors": events.get("discovery_error", 0),
    }])


def _replay(samples, trades, audit, gate):
    pass_closes = set(
        pd.to_numeric(
            gate.loc[gate["range_pass_lt23c"].astype(bool), "close_ts"],
            errors="coerce",
        ).dropna().astype(float)
    )
    good = audit[
        audit["m1_m5_quality_ok"].astype(bool)
        & audit["close_ts"].astype(float).isin(pass_closes)
    ].copy()
    sim_meta = {
        str(r.ticker): {
            "ticker": str(r.ticker), "series": str(r.series),
            "close_ts": float(r.close_ts),
            "reconstructed_coverage_pct": float(r.valid_coverage_pct),
        }
        for r in good.itertuples(index=False)
    }
    targets = sorted(sim_meta, key=lambda t: (sim_meta[t]["close_ts"], t))
    contracts, fills, episodes, counts = [], [], [], Counter()
    print(
        f"Frozen filter pass: {len(pass_closes)} windows | "
        f"replaying {len(targets)} M1-M5 quality contracts"
    )
    for i, ticker in enumerate(targets, 1):
        e, f, c, k = INV._simulate_contract(
            ticker, sim_meta[ticker], samples.get(ticker, []), trades.get(ticker, [])
        )
        episodes.extend(e); fills.extend(f); counts.update(k)
        if c is not None:
            contracts.append(c)
        if i % 100 == 0 or i == len(targets):
            print(
                f"  replay {i}/{len(targets)} | fills={len(fills)} | "
                f"qty={sum(float(x['qty']) for x in fills):.2f}"
            )
    return (
        pd.DataFrame(contracts), pd.DataFrame(fills), pd.DataFrame(episodes),
        pd.DataFrame([{"reason": k, "count": v} for k, v in counts.most_common()]),
    )


def _headline(cdf, fdf, wdf, edf, gate):
    if cdf.empty:
        return pd.DataFrame([{
            "scenario": "NAT4_TO_2_PREM1_LT23_FROZEN_OOS1",
            "eligible_contracts": 0, "independent_windows": 0,
            "fill_events": 0, "fill_qty": 0.0,
            "net_mtm_pnl_before_fees": 0.0, "pnl_per_window": np.nan,
            "matched_roundtrip_pnl": 0.0, "residual_inventory_mtm_pnl": 0.0,
            "feature_quality_windows": int(gate["feature_quality_ok"].sum()),
            "range_pass_windows": int(gate["range_pass_lt23c"].sum()),
        }])
    h = INV._headline(cdf, fdf, wdf, edf).copy()
    h["scenario"] = "NAT4_TO_2_PREM1_LT23_FROZEN_OOS1"
    h["feature_quality_windows"] = int(gate["feature_quality_ok"].sum())
    h["range_pass_windows"] = int(gate["range_pass_lt23c"].sum())
    h["range_fail_windows"] = int(gate["range_fail_ge23c"].sum())
    h["residual_inventory_mtm_pnl"] = (
        h["net_mtm_pnl_before_fees"] - h["matched_roundtrip_pnl"]
    )
    return h


def _print_report(audit_summary, gate, headline, chrono, side, out):
    a, h = audit_summary.iloc[0], headline.iloc[0]
    print("\n" + "=" * 132)
    print("FROZEN OOS #1 — NAT4->2 + PRE-M1 MAX RANGE <23c")
    print("=" * 132)
    print(
        f"Session={EXPECTED_SESSION_NAME} | duration={a['duration_hours']:.2f}h | "
        f"connections={int(a['connections'])} | exceptions={int(a['connection_exceptions'])}"
    )
    print(
        f"M1-M5 quality: {int(a['m1_m5_quality_contracts'])}/{int(a['discovered_contracts'])} contracts | "
        f"{int(a['m1_m5_quality_windows'])} windows"
    )
    print(
        f"Pre-feature quality={int(a['feature_quality_windows'])} windows | "
        f"<23c pass={int(a['range_pass_windows_lt23c'])} | "
        f">=23c fail={int(a['range_fail_windows_ge23c'])} | "
        f"missing/low={int(a['missing_or_low_pre_feature_windows'])}"
    )
    print(
        f"Median coverage: pre={a['median_pre_valid_coverage_pct']:.1f}% | "
        f"M1-M5={a['median_m1_m5_valid_coverage_pct']:.1f}% | "
        f"M5-M6={a['median_post_valid_coverage_pct']:.1f}%"
    )
    print("\nPRIMARY FROZEN RESULT")
    print(
        f"sim windows={int(_f(h.get('independent_windows'), 0))} | "
        f"contracts={int(_f(h.get('eligible_contracts'), 0))} | "
        f"fills={int(_f(h.get('fill_events'), 0))} | qty={_f(h.get('fill_qty'), 0):.2f}"
    )
    print(
        f"net=${_f(h.get('net_mtm_pnl_before_fees'), 0):+.4f} | "
        f"pnl/window=${_f(h.get('pnl_per_window')):+.4f} | "
        f"matched=${_f(h.get('matched_roundtrip_pnl'), 0):+.4f} | "
        f"residual=${_f(h.get('residual_inventory_mtm_pnl'), 0):+.4f}"
    )
    print(
        f"gross=${_f(h.get('gross_capture_dollars'), 0):+.4f} | "
        f"adverse=${_f(h.get('adverse_selection_to_m5_dollars'), 0):+.4f} | "
        f"worst=${_f(h.get('worst_window_pnl')):+.4f} | DD=${_f(h.get('max_drawdown')):+.4f}"
    )
    if all(f"qty_weighted_markout_{m}s_c" in headline.columns for m in MARKOUTS):
        print("markouts: " + " | ".join(
            f"{m}s={_f(h.get(f'qty_weighted_markout_{m}s_c')):+.3f}c" for m in MARKOUTS
        ))
    if len(chrono):
        print("\nCHRONOLOGY")
        print(chrono.round(4).to_string(index=False))
    if len(side):
        print("\nSIDE")
        print(side.round(4).to_string(index=False))
    print("\nPRE-M1 WINDOW GATE")
    cols = [
        "close_time", "m1_m5_quality_contracts", "pre_feature_contracts",
        "pre_contract_coverage_pct", "pre_m0_m1_max_mid_range_c",
        "feature_quality_ok", "range_pass_lt23c",
    ]
    if len(gate):
        print(gate[cols].round(3).to_string(index=False))
    print("\nGUARDRAILS")
    print("  Hard-bound to OOS #1; the later recording is never read.")
    print("  <23c and all execution/inventory rules are frozen; no sweep is run.")
    print("  Independent inference unit is the 15-minute window.")
    print("  Q100 inside-spread execution remains counterfactual to historical public flow.")
    print("Outputs:", out)
    print("=" * 132)


def run_nat4_to_2_prem1_lt23_frozen_oos1(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if not session.exists():
        raise FileNotFoundError(session)
    manifest, health, duration_h = _verify_frozen_session(session)
    print(
        f"OOS #1 verification: PASS | session={session.name} | duration={duration_h:.3f}h"
    )

    meta = O._metadata(session)
    samples, info, duplicates, bbo_stats = O._bbo(session, meta)
    trades, trade_stats = O._trades(session, meta)
    events = O._event_counts(session)
    audit = _full_audit(meta, samples, info, duplicates, trades)
    if audit.empty:
        raise RuntimeError("No contracts found in compact OOS recording")
    gate = _window_gate(audit)
    audit_summary = _audit_summary(
        duration_h, audit, gate, bbo_stats, trade_stats, events
    )

    if output_dir is None:
        output_dir = (
            R.PROJECT_ROOT / "results" / "kalshi_nat4_to_2_prem1_lt23_frozen_oos1"
            / f"{session.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    audit.to_csv(out / "contract_data_quality.csv", index=False)
    gate.to_csv(out / "pre_m1_window_gate.csv", index=False)
    audit_summary.to_csv(out / "audit_summary.csv", index=False)

    if int(gate["feature_quality_ok"].sum()) == 0:
        raise RuntimeError(
            f"No feature-quality windows. Audit saved to {out}; strategy was not replayed."
        )

    cdf, fdf, edf, counts = _replay(samples, trades, audit, gate)
    if cdf.empty:
        wdf = chrono = side = inventory = pd.DataFrame()
    else:
        wdf = L._window_summary(cdf)
        chrono = S._chronological_detail(wdf, "NAT4_TO_2_PREM1_LT23_FROZEN_OOS1")
        side = Q._side("NAT4_TO_2_PREM1_LT23_FROZEN_OOS1", fdf) if len(fdf) else pd.DataFrame()
        inventory = INV._inventory_bucket(fdf) if len(fdf) else pd.DataFrame()
    headline = _headline(cdf, fdf, wdf, edf, gate)

    for name, df in {
        "headline_summary.csv": headline,
        "window_summary.csv": wdf,
        "chronological_robustness.csv": chrono,
        "side_summary.csv": side,
        "inventory_bucket_summary.csv": inventory,
        "contract_summary.csv": cdf,
        "fills.csv": fdf,
        "quote_episodes.csv": edf,
        "policy_counts.csv": counts,
    }.items():
        df.to_csv(out / name, index=False)

    config = {
        "study_version": STUDY_VERSION,
        "session": str(session),
        "expected_session_name": EXPECTED_SESSION_NAME,
        "session_duration_hours": duration_h,
        "later_recording_accessed": False,
        "pre_m1_feature": "max cross-contract M0-M1 midpoint range",
        "pre_m1_range_cutoff_c": PRE_RANGE_CUTOFF_C,
        "pre_m1_rule": "strictly less than 23.00c",
        "m1_m5_quality_gate_pct": M1_M5_QUALITY_GATE_PCT,
        "pre_contract_quality_gate_pct": PRE_CONTRACT_QUALITY_GATE_PCT,
        "pre_window_contract_gate_pct": PRE_WINDOW_CONTRACT_GATE_PCT,
        "natural_spread_c": INV.NATURAL_SPREAD_C,
        "our_spread_c": INV.OUR_SPREAD_C,
        "improve_each_side_c": INV.IMPROVE_C,
        "max_quote_qty": INV.MAX_QUOTE_QTY,
        "inventory_target": INV.TARGET_INVENTORY,
        "soft_inventory": INV.SOFT_INVENTORY,
        "hard_inventory": INV.HARD_INVENTORY,
        "momentum_guard": "unchanged 3s Defensive V1 rule",
        "flow_guard": "unchanged prior-5s Defensive V1 rule",
        "same_side_cooldown_s": 3.0,
        "fill_rule": "recorded aggressive trades only; BBO movement never creates fill",
        "fees": 0.0,
        "markout_seconds": list(MARKOUTS),
        "counterfactual_caveat": (
            "Q100 one tick inside historical BBO could have changed future public flow; "
            "historical flow is treated as exogenous."
        ),
        "status": "FROZEN_OOS1_NO_TUNING",
    }
    (out / "study_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )

    if show:
        _print_report(audit_summary, gate, headline, chrono, side, out)
    return {
        "output_dir": out, "audit_summary": audit_summary, "audit": audit,
        "gate": gate, "headline": headline, "windows": wdf,
        "chronology": chrono, "side": side, "inventory": inventory,
        "contracts": cdf, "fills": fdf, "episodes": edf,
    }
