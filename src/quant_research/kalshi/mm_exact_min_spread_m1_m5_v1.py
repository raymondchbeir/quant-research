from __future__ import annotations

"""Exact path-dependent minimum-spread replay for Defensive M1-M5 MM V1.

This study changes ONE mechanism only: the minimum quoted spread required to
place/keep a Defensive V1 order.

For each spread scenario (2c, 3c, 4c, 5c):
- use the same validated reconstructed 1 Hz books and recorded trades;
- replay every eligible contract from scratch;
- keep Defensive V1 momentum, flow, inventory, cooldown, FIFO, asset universe,
  and M1-M5 timing unchanged;
- cancel an active quote whenever the current valid book no longer satisfies
  that scenario's minimum spread;
- only re-enter at a later valid book sample if the complete V1 policy passes;
- recompute all fills, inventory paths, matched round trips, and PnL.

The 2c scenario is required to reproduce the saved Defensive V1 result before
higher-spread scenarios are interpreted. Fees are excluded.
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_reconstructed_m1_m5_backtest as B
from . import mm_defensive_m1_m5_v1 as D
from . import mm_exact_quote_lifetime_m1_m5_v1 as L

STUDY_VERSION = "M1_M5_DEFENSIVE_V1_EXACT_MIN_SPREAD_V1"
EPS = 1e-9
DEFAULT_SPREADS_C = (2.0, 3.0, 4.0, 5.0)
DEFAULT_MARKOUT_SECONDS = (5, 15, 30, 60)


def _scenario_name(spread_c):
    return f"MIN_SPREAD_{float(spread_c):g}C"


def _chronological_detail(window_df, scenario):
    if window_df.empty:
        return pd.DataFrame()
    w = window_df.sort_values("close_ts").reset_index(drop=True)
    n = len(w)
    rows = []

    def add(name, z):
        if z.empty:
            return
        pnl = z["net_mtm_pnl_before_fees"].astype(float)
        cum = pnl.cumsum()
        peak = cum.cummax()
        dd = cum - peak
        rows.append({
            "scenario": scenario,
            "split": name,
            "windows": len(z),
            "fill_qty": z["fill_qty"].sum(),
            "net_pnl": pnl.sum(),
            "pnl_per_window": pnl.mean(),
            "median_window_pnl": pnl.median(),
            "positive_window_pct": 100.0 * (pnl > 0).mean(),
            "worst_window": pnl.min(),
            "best_window": pnl.max(),
            "max_drawdown_within_split": dd.min(),
        })

    cut = n // 2
    add("FIRST_HALF", w.iloc[:cut])
    add("SECOND_HALF", w.iloc[cut:])
    for q in range(4):
        lo = round(q * n / 4)
        hi = round((q + 1) * n / 4)
        add(f"QUARTILE_{q+1}", w.iloc[lo:hi])
    return pd.DataFrame(rows)


def _scenario_summary(
    scenario,
    min_spread_c,
    contract_df,
    fills_df,
    window_df,
    side_df,
    counts,
    markouts,
):
    total_pnl = contract_df["net_mtm_pnl_before_fees"].sum()
    fill_qty = fills_df["qty"].sum() if len(fills_df) else 0.0
    row = {
        "scenario": scenario,
        "min_spread_c": float(min_spread_c),
        "eligible_contracts": len(contract_df),
        "independent_windows": len(window_df),
        "contracts_with_fill": int((contract_df["fill_qty"] > EPS).sum()),
        "fill_events": len(fills_df),
        "fill_qty": fill_qty,
        "gross_capture_dollars": contract_df["gross_spread_capture_dollars"].sum(),
        "adverse_selection_to_m5_dollars": contract_df["adverse_selection_to_m5_dollars"].sum(),
        "net_mtm_pnl_before_fees": total_pnl,
        "pnl_per_window": total_pnl / len(window_df) if len(window_df) else np.nan,
        "matched_roundtrip_qty": contract_df["matched_roundtrip_qty"].sum(),
        "matched_roundtrip_pnl": contract_df["matched_roundtrip_pnl"].sum(),
        "avg_max_abs_inventory": contract_df["max_abs_inventory"].mean(),
        "p95_max_abs_inventory": contract_df["max_abs_inventory"].quantile(0.95),
        "window_median_pnl": window_df["net_mtm_pnl_before_fees"].median() if len(window_df) else np.nan,
        "positive_window_pct": 100.0 * (window_df["net_mtm_pnl_before_fees"] > 0).mean() if len(window_df) else np.nan,
        "worst_window_pnl": window_df["net_mtm_pnl_before_fees"].min() if len(window_df) else np.nan,
        "best_window_pnl": window_df["net_mtm_pnl_before_fees"].max() if len(window_df) else np.nan,
        "max_drawdown": window_df["drawdown"].min() if len(window_df) else np.nan,
        "avg_gross_edge_at_fill_c": fills_df["gross_edge_at_fill_c"].mean() if len(fills_df) else np.nan,
        "avg_spread_at_join_c": fills_df["spread_c_at_join"].mean() if len(fills_df) else np.nan,
        "break_even_fee_c_per_fill_qty": 100.0 * total_pnl / fill_qty if fill_qty > EPS else np.nan,
        "spread_block_count": counts.get("BID_BLOCK_SPREAD", 0) + counts.get("ASK_BLOCK_SPREAD", 0),
    }
    for h in markouts:
        row[f"avg_markout_{h}s_c"] = (
            pd.to_numeric(fills_df.get(f"markout_{h}s_c"), errors="coerce").mean()
            if len(fills_df) else np.nan
        )

    for side in ("BID", "ASK"):
        z = side_df[side_df["side"] == side]
        if len(z):
            x = z.iloc[0]
            row[f"{side.lower()}_fill_qty"] = x.get("fill_qty", np.nan)
            row[f"{side.lower()}_avg_gross_edge_c"] = x.get("avg_gross_edge_at_fill_c", np.nan)
            for h in markouts:
                row[f"{side.lower()}_avg_markout_{h}s_c"] = x.get(f"avg_markout_{h}s_c", np.nan)

    chrono = L._chronology(window_df)
    row.update(chrono)
    return row


def _load_v1_reference(defensive_v1_dir):
    p = Path(defensive_v1_dir) / "headline_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError(f"Empty V1 headline: {p}")
    return df.iloc[0]


def _print_report(summary_df, chrono_df, v1_ref, out):
    print("\n" + "=" * 132)
    print("EXACT PATH-DEPENDENT MINIMUM-SPREAD REPLAY — DEFENSIVE V1 OTHERWISE UNCHANGED")
    print("=" * 132)
    cols = [
        "scenario", "fill_qty", "net_mtm_pnl_before_fees", "pnl_per_window",
        "matched_roundtrip_pnl", "avg_spread_at_join_c", "avg_gross_edge_at_fill_c",
        "avg_markout_5s_c", "avg_markout_15s_c", "avg_markout_30s_c", "avg_markout_60s_c",
        "p95_max_abs_inventory", "worst_window_pnl", "max_drawdown",
        "first_half_pnl", "second_half_pnl", "positive_window_pct",
        "break_even_fee_c_per_fill_qty",
    ]
    print(summary_df[cols].round(4).to_string(index=False))

    base = summary_df.loc[summary_df["min_spread_c"] == 2.0].iloc[0]
    saved = B._f(v1_ref.get("net_mtm_pnl_before_fees"))
    diff = float(base["net_mtm_pnl_before_fees"]) - saved
    print("\nREPRODUCTION CHECK")
    print(f"  Saved Defensive V1 PnL: ${saved:+.6f}")
    print(f"  Exact 2c replay PnL:    ${base['net_mtm_pnl_before_fees']:+.6f}")
    print(f"  Difference:             ${diff:+.8f}")

    print("\nCHRONOLOGICAL ROBUSTNESS")
    if not chrono_df.empty:
        print(chrono_df.round(4).to_string(index=False))

    print("\nINTERPRETATION RULE")
    print("  The 2c replay must reproduce V1 before 3c/4c/5c are trusted.")
    print("  This is same-session exploratory mechanism testing, not OOS validation.")
    print("  Look for broad improvement across multiple adjacent spread floors and across time, not one lucky threshold.")
    print("  Fees are excluded; a small positive result is not deployment-ready.")
    print("Outputs:", out)
    print("=" * 132)


def run_exact_min_spread_replay(
    session_dir,
    reconstruction_dir,
    defensive_v1_dir,
    output_dir=None,
    *,
    min_spreads_c=DEFAULT_SPREADS_C,
    quote_qty=1.0,
    markout_seconds=DEFAULT_MARKOUT_SECONDS,
    max_markout_lag_s=2.0,
    reproduction_tolerance_dollars=0.02,
    show=True,
):
    session = Path(session_dir)
    recon = Path(reconstruction_dir)
    v1_dir = Path(defensive_v1_dir)
    for p in (session, recon, v1_dir):
        if not p.exists():
            raise FileNotFoundError(p)

    spreads = sorted({float(x) for x in min_spreads_c})
    if any(x <= 0 for x in spreads):
        raise ValueError("All min_spreads_c values must be > 0")
    if 2.0 not in spreads:
        spreads = [2.0] + spreads
    spreads = sorted(set(spreads))
    markouts = tuple(sorted({int(x) for x in markout_seconds if int(x) > 0}))

    quality_df, meta = B._load_quality_contracts(recon)
    eligible = set(meta)
    print(f"Validated reconstruction contracts: {len(eligible):,}")
    print("Loading validated 1 Hz reconstructed books once...")
    samples, sample_stats = B._load_reconstructed_samples(recon, eligible)
    missing = sorted(eligible - set(samples))
    if missing:
        raise RuntimeError(f"{len(missing)} eligible contracts missing samples; first={missing[:3]}")

    print("Streaming trades once...")
    trades, trade_stats = B._scan_trades(session, meta)

    if output_dir is None:
        root = session.resolve().parents[2] if len(session.resolve().parents) >= 3 else Path.cwd()
        output_dir = (
            root / "results" / "kalshi_mm_m1_m5_exact_min_spread"
            / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targets = sorted(eligible, key=lambda x: (meta[x]["close_ts"], x))
    summaries = []
    asset_tables = []
    side_tables = []
    chrono_tables = []
    scenario_returns = {}

    for min_spread_c in spreads:
        name = _scenario_name(min_spread_c)
        policy = dict(D.DEFAULT_POLICY)
        policy["min_spread_c"] = float(min_spread_c)

        print(f"\nReplaying {name} on {len(targets):,} contracts...")
        t0 = time.time()
        episodes_all = []
        fills_all = []
        contracts = []
        counts = Counter()

        for i, ticker in enumerate(targets, 1):
            eps, fills, contract, c = L._simulate_contract(
                ticker,
                meta[ticker],
                samples[ticker],
                trades.get(ticker, []),
                float(quote_qty),
                markouts,
                float(max_markout_lag_s),
                policy,
                None,
            )
            episodes_all.extend(eps)
            fills_all.extend(fills)
            counts.update(c)
            if contract is not None:
                contracts.append(contract)
            if i % 100 == 0 or i == len(targets):
                print(
                    f"  {name}: {i:,}/{len(targets):,} | fills={len(fills_all):,} | "
                    f"spread_blocks={counts.get('BID_BLOCK_SPREAD',0)+counts.get('ASK_BLOCK_SPREAD',0):,} | "
                    f"{time.time()-t0:.1f}s"
                )

        contract_df = pd.DataFrame(contracts)
        fills_df = pd.DataFrame(fills_all)
        episodes_df = pd.DataFrame(episodes_all)
        window_df = L._window_summary(contract_df)
        side_df = B._side_summary(episodes_all, fills_all, markouts)
        side_df.insert(0, "scenario", name)
        asset_df = L._asset_summary(name, contract_df)
        chrono_df = _chronological_detail(window_df, name)

        summary = _scenario_summary(
            name, min_spread_c, contract_df, fills_df, window_df,
            side_df, counts, markouts,
        )
        summaries.append(summary)
        asset_tables.append(asset_df)
        side_tables.append(side_df)
        chrono_tables.append(chrono_df)
        scenario_returns[name] = {
            "contracts": contract_df,
            "fills": fills_df,
            "episodes": episodes_df,
            "windows": window_df,
            "side_summary": side_df,
        }

        stem = name.lower()
        contract_df.to_csv(out / f"{stem}_contract_summary.csv", index=False)
        fills_df.to_csv(out / f"{stem}_fills.csv", index=False)
        episodes_df.to_csv(out / f"{stem}_quote_episodes.csv", index=False)
        window_df.to_csv(out / f"{stem}_window_summary.csv", index=False)
        side_df.to_csv(out / f"{stem}_side_summary.csv", index=False)
        pd.DataFrame([{"reason": k, "count": v} for k, v in counts.most_common()]).to_csv(
            out / f"{stem}_policy_counts.csv", index=False
        )

    summary_df = pd.DataFrame(summaries).sort_values("min_spread_c").reset_index(drop=True)
    base_pnl = float(summary_df.loc[summary_df["min_spread_c"] == 2.0, "net_mtm_pnl_before_fees"].iloc[0])
    base_ppw = float(summary_df.loc[summary_df["min_spread_c"] == 2.0, "pnl_per_window"].iloc[0])
    summary_df["pnl_change_vs_2c"] = summary_df["net_mtm_pnl_before_fees"] - base_pnl
    summary_df["pnl_per_window_change_vs_2c"] = summary_df["pnl_per_window"] - base_ppw

    asset_all = pd.concat(asset_tables, ignore_index=True) if asset_tables else pd.DataFrame()
    side_all = pd.concat(side_tables, ignore_index=True) if side_tables else pd.DataFrame()
    chrono_all = pd.concat(chrono_tables, ignore_index=True) if chrono_tables else pd.DataFrame()

    v1_ref = _load_v1_reference(v1_dir)
    saved_v1_pnl = B._f(v1_ref.get("net_mtm_pnl_before_fees"))
    reproduction_diff = base_pnl - saved_v1_pnl
    reproduction_ok = abs(reproduction_diff) <= float(reproduction_tolerance_dollars)

    summary_df.to_csv(out / "scenario_summary.csv", index=False)
    asset_all.to_csv(out / "asset_by_scenario.csv", index=False)
    side_all.to_csv(out / "side_by_scenario.csv", index=False)
    chrono_all.to_csv(out / "chronology_by_scenario.csv", index=False)
    pd.DataFrame([{**sample_stats, **trade_stats}]).to_csv(out / "scan_stats.csv", index=False)

    config = {
        "study_version": STUDY_VERSION,
        "session": str(session.resolve()),
        "reconstruction_dir": str(recon.resolve()),
        "defensive_v1_dir": str(v1_dir.resolve()),
        "eligible_contracts": len(targets),
        "min_spreads_c": spreads,
        "quote_qty": quote_qty,
        "base_policy": dict(D.DEFAULT_POLICY),
        "mechanism_changed": "minimum spread requirement only",
        "unchanged": [
            "momentum threshold/lookback", "flow threshold/lookback",
            "inventory soft/hard limits", "same-side cooldown", "FIFO model",
            "asset universe", "M1-M5 quote window", "no quote lifetime",
        ],
        "fees": "excluded",
        "saved_v1_pnl": saved_v1_pnl,
        "exact_2c_replay_pnl": base_pnl,
        "reproduction_difference_dollars": reproduction_diff,
        "reproduction_tolerance_dollars": reproduction_tolerance_dollars,
        "reproduction_ok": reproduction_ok,
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if show:
        _print_report(summary_df, chrono_all, v1_ref, out)
        try:
            from IPython.display import display
            print("\nSCENARIO SUMMARY")
            display(summary_df.round(4))
            print("\nBY ASSET / SCENARIO")
            display(asset_all.round(4))
            print("\nSIDE / SCENARIO")
            display(side_all.round(4))
        except Exception:
            pass

    if not reproduction_ok:
        raise RuntimeError(
            f"2c replay failed V1 reproduction: exact={base_pnl:+.6f}, "
            f"saved={saved_v1_pnl:+.6f}, diff={reproduction_diff:+.6f}, "
            f"tolerance={reproduction_tolerance_dollars:.6f}. Do not interpret higher spreads."
        )

    return {
        "output_dir": out,
        "scenario_summary": summary_df,
        "asset_summary": asset_all,
        "side_summary": side_all,
        "chronology": chrono_all,
        "scenarios": scenario_returns,
    }


def _main():
    p = argparse.ArgumentParser(description=STUDY_VERSION)
    p.add_argument("--session", required=True)
    p.add_argument("--reconstruction-dir", required=True)
    p.add_argument("--defensive-v1-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--spreads", default="2,3,4,5")
    args = p.parse_args()
    spreads = tuple(float(x.strip()) for x in args.spreads.split(",") if x.strip())
    run_exact_min_spread_replay(
        args.session,
        args.reconstruction_dir,
        args.defensive_v1_dir,
        output_dir=args.output_dir,
        min_spreads_c=spreads,
        show=True,
    )


if __name__ == "__main__":
    _main()
