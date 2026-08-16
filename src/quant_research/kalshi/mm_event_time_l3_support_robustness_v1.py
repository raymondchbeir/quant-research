from __future__ import annotations

"""Robustness suite for the already-developed L3-support Q1 event-time candidate.

DEVELOPMENT ROBUSTNESS ONLY -- no new strategy rules, thresholds, or asset
selection are tested here.

This module reads the CSV outputs produced by
`mm_event_time_l3_state_machine_replay_v1.py` for the hard-bound V4 development
session 20260815_043130. It does NOT re-read the 3GB event-time book and does
NOT evaluate a new policy.

Primary comparison
------------------
    BASELINE_BBO_Q1
vs
    L3_SUPPORT_ONLY_Q1

Questions answered
------------------
1. Is the improvement paired window-by-window or carried by a few windows?
2. What is the window-bootstrap uncertainty of incremental PnL?
3. Does the conclusion survive leave-one-window-out and leave-one-series-out?
4. Is candidate PnL broad across series and M0-M5 entry minutes?
5. How much PnL comes from matched round trips vs residual M5 inventory MTM?
6. How concentrated are profits/losses in the best/worst windows?
7. What pre-fee cushion exists per filled contract?

No post-hoc strategy choice should be made from this module alone.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C

STUDY_VERSION = "MM_EVENT_TIME_L3_SUPPORT_ROBUSTNESS_V1"
EXPECTED_SESSION_NAME = "20260815_043130"
BASELINE = "BASELINE_BBO_Q1"
CANDIDATE = "L3_SUPPORT_ONLY_Q1"
BOOTSTRAP_REPS = 20_000
RNG_SEED = 20260815
EPS = 1e-12


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _replay_output_dir(session_name: str) -> Path:
    return (
        C.PROJECT_ROOT
        / "results"
        / "kalshi_mm_event_l3_state_machine_replay"
        / str(session_name)
    )


def _require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _weighted_mean(df: pd.DataFrame, col: str, weight: str = "qty"):
    if df.empty or col not in df.columns or weight not in df.columns:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    w = pd.to_numeric(df[weight], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def _paired_windows(windows: pd.DataFrame) -> pd.DataFrame:
    need = {"policy", "close_ts", "net_mtm_pnl_before_fees"}
    missing = need - set(windows.columns)
    if missing:
        raise RuntimeError(f"window_results.csv missing columns: {sorted(missing)}")

    z = windows[windows.policy.isin([BASELINE, CANDIDATE])].copy()
    piv = z.pivot_table(
        index=["close_ts", "close_time"],
        columns="policy",
        values=[
            "net_mtm_pnl_before_fees",
            "fill_qty",
            "gross_capture",
            "adverse_selection_to_m5",
            "ending_abs_inventory_sum",
        ],
        aggfunc="sum",
    )
    piv.columns = [f"{a}__{b}" for a, b in piv.columns]
    piv = piv.reset_index().sort_values("close_ts").reset_index(drop=True)

    for metric in (
        "net_mtm_pnl_before_fees",
        "fill_qty",
        "gross_capture",
        "adverse_selection_to_m5",
        "ending_abs_inventory_sum",
    ):
        b = f"{metric}__{BASELINE}"
        c = f"{metric}__{CANDIDATE}"
        if b in piv.columns and c in piv.columns:
            piv[f"delta_{metric}"] = piv[c] - piv[b]

    piv["candidate_beats_baseline"] = piv["delta_net_mtm_pnl_before_fees"] > 0
    return piv


def _bootstrap_paired_delta(paired: pd.DataFrame):
    x = pd.to_numeric(
        paired["delta_net_mtm_pnl_before_fees"], errors="coerce"
    ).dropna().to_numpy(float)
    n = len(x)
    if n == 0:
        return pd.DataFrame(), {}

    rng = np.random.default_rng(RNG_SEED)
    # Independent unit is the 15-minute close window. Resample windows, not fills.
    idx = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
    means = x[idx].mean(axis=1)
    totals = means * n

    stats = {
        "windows": n,
        "observed_incremental_total": float(x.sum()),
        "observed_incremental_per_window": float(x.mean()),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_mean_delta_per_window_p2_5": float(np.quantile(means, 0.025)),
        "bootstrap_mean_delta_per_window_p50": float(np.quantile(means, 0.50)),
        "bootstrap_mean_delta_per_window_p97_5": float(np.quantile(means, 0.975)),
        "bootstrap_total_delta_p2_5": float(np.quantile(totals, 0.025)),
        "bootstrap_total_delta_p50": float(np.quantile(totals, 0.50)),
        "bootstrap_total_delta_p97_5": float(np.quantile(totals, 0.975)),
        "bootstrap_probability_incremental_total_gt_0": float(np.mean(totals > 0)),
    }
    dist = pd.DataFrame({
        "bootstrap_rep": np.arange(BOOTSTRAP_REPS),
        "mean_delta_per_window": means,
        "implied_total_delta": totals,
    })
    return dist, stats


def _leave_one_window_out(paired: pd.DataFrame) -> pd.DataFrame:
    x = pd.to_numeric(
        paired["delta_net_mtm_pnl_before_fees"], errors="coerce"
    ).to_numpy(float)
    cand = pd.to_numeric(
        paired[f"net_mtm_pnl_before_fees__{CANDIDATE}"], errors="coerce"
    ).to_numpy(float)
    base = pd.to_numeric(
        paired[f"net_mtm_pnl_before_fees__{BASELINE}"], errors="coerce"
    ).to_numpy(float)
    total_x = np.nansum(x)
    total_c = np.nansum(cand)
    total_b = np.nansum(base)
    rows = []
    for i, r in paired.reset_index(drop=True).iterrows():
        rows.append({
            "omitted_close_ts": r.close_ts,
            "omitted_close_time": r.close_time,
            "omitted_incremental_pnl": x[i],
            "candidate_pnl_without_window": total_c - cand[i],
            "baseline_pnl_without_window": total_b - base[i],
            "incremental_pnl_without_window": total_x - x[i],
        })
    return pd.DataFrame(rows)


def _series_table(contracts: pd.DataFrame) -> pd.DataFrame:
    z = contracts[contracts.policy.isin([BASELINE, CANDIDATE])].copy()
    g = (
        z.groupby(["series", "policy"], as_index=False)
        .agg(
            contracts=("ticker", "count"),
            fill_qty=("fill_qty", "sum"),
            net_pnl=("net_mtm_pnl_before_fees", "sum"),
            gross_capture=("gross_spread_capture_dollars", "sum"),
            adverse_selection=("adverse_selection_to_m5_dollars", "sum"),
            median_abs_ending_inventory=(
                "ending_inventory_yes_equiv",
                lambda x: pd.to_numeric(x, errors="coerce").abs().median(),
            ),
            p95_max_abs_inventory=("max_abs_inventory", lambda x: pd.to_numeric(x, errors="coerce").quantile(.95)),
        )
    )
    rows = []
    for series, q in g.groupby("series", sort=True):
        d = {str(r.policy): r for r in q.itertuples(index=False)}
        if BASELINE not in d or CANDIDATE not in d:
            continue
        b, c = d[BASELINE], d[CANDIDATE]
        rows.append({
            "series": series,
            "candidate_contracts": c.contracts,
            "candidate_fill_qty": c.fill_qty,
            "candidate_net_pnl": c.net_pnl,
            "candidate_gross_capture": c.gross_capture,
            "candidate_adverse_selection": c.adverse_selection,
            "baseline_net_pnl": b.net_pnl,
            "incremental_net_vs_baseline": c.net_pnl - b.net_pnl,
            "candidate_median_abs_ending_inventory": c.median_abs_ending_inventory,
            "candidate_p95_max_abs_inventory": c.p95_max_abs_inventory,
        })
    return pd.DataFrame(rows).sort_values("series").reset_index(drop=True)


def _leave_one_series_out(series_table: pd.DataFrame) -> pd.DataFrame:
    if series_table.empty:
        return pd.DataFrame()
    total_c = float(series_table.candidate_net_pnl.sum())
    total_b = float(series_table.baseline_net_pnl.sum())
    total_d = float(series_table.incremental_net_vs_baseline.sum())
    rows = []
    for r in series_table.itertuples(index=False):
        rows.append({
            "omitted_series": r.series,
            "candidate_pnl_without_series": total_c - float(r.candidate_net_pnl),
            "baseline_pnl_without_series": total_b - float(r.baseline_net_pnl),
            "incremental_pnl_without_series": total_d - float(r.incremental_net_vs_baseline),
        })
    return pd.DataFrame(rows).sort_values("omitted_series").reset_index(drop=True)


def _minute_bucket(fill_ts: float, close_ts: float):
    elapsed = float(fill_ts) - (float(close_ts) - 900.0)
    if not np.isfinite(elapsed) or elapsed < 0 or elapsed >= 300:
        return None
    minute = int(elapsed // 60)
    return f"M{minute}-M{minute+1}"


def _fifo_decomposition_one_contract(f: pd.DataFrame, final_mid: float):
    """Exact cash-consistent FIFO decomposition.

    Lots are signed YES inventory. Realized round-trip PnL is allocated to the
    ENTRY minute of the lot that gets closed. Residual M5 MTM is allocated to
    the entry minute of the still-open lot. Therefore minute contributions sum
    exactly to contract PnL (up to floating point).
    """
    lots = deque()  # dict(sign, qty, price, entry_minute, side)
    matched_by_minute = {f"M{i}-M{i+1}": 0.0 for i in range(5)}
    residual_by_minute = {f"M{i}-M{i+1}": 0.0 for i in range(5)}
    matched_total = 0.0
    matched_qty = 0.0

    z = f.sort_values(["fill_ts", "side"]).copy()
    for r in z.itertuples(index=False):
        qty = float(r.qty)
        px = float(r.price)
        side = str(r.side)
        sign = 1 if side == "BID" else -1
        entry_minute = _minute_bucket(float(r.fill_ts), float(r.close_ts))
        remaining = qty

        while remaining > EPS and lots and int(lots[0]["sign"]) == -sign:
            lot = lots[0]
            q = min(remaining, float(lot["qty"]))
            if lot["sign"] == 1 and sign == -1:
                pnl = q * (px - float(lot["price"]))
            elif lot["sign"] == -1 and sign == 1:
                pnl = q * (float(lot["price"]) - px)
            else:
                raise AssertionError("opposite-sign FIFO invariant")
            matched_total += pnl
            matched_qty += q
            if lot["entry_minute"] is not None:
                matched_by_minute[lot["entry_minute"]] += pnl
            lot["qty"] -= q
            remaining -= q
            if lot["qty"] <= EPS:
                lots.popleft()

        if remaining > EPS:
            lots.append({
                "sign": sign,
                "qty": remaining,
                "price": px,
                "entry_minute": entry_minute,
            })

    residual_total = 0.0
    residual_abs_qty = 0.0
    for lot in lots:
        q = float(lot["qty"])
        residual_abs_qty += q
        if lot["sign"] == 1:
            pnl = q * (float(final_mid) - float(lot["price"]))
        else:
            pnl = q * (float(lot["price"]) - float(final_mid))
        residual_total += pnl
        if lot["entry_minute"] is not None:
            residual_by_minute[lot["entry_minute"]] += pnl

    return {
        "matched_roundtrip_pnl": matched_total,
        "matched_qty": matched_qty,
        "residual_inventory_mtm": residual_total,
        "residual_abs_qty": residual_abs_qty,
        "matched_by_minute": matched_by_minute,
        "residual_by_minute": residual_by_minute,
    }


def _decomposition(contracts: pd.DataFrame, fills: pd.DataFrame):
    c = contracts[contracts.policy == CANDIDATE].copy()
    f = fills[fills.policy == CANDIDATE].copy()
    fgroups = {str(t): z.copy() for t, z in f.groupby("ticker", sort=False)}
    rows = []
    minute = {f"M{i}-M{i+1}": {"matched": 0.0, "residual": 0.0} for i in range(5)}

    for r in c.itertuples(index=False):
        z = fgroups.get(str(r.ticker), pd.DataFrame())
        final_mid = _f(r.final_mid_m5)
        if z.empty or not np.isfinite(final_mid):
            matched = 0.0
            residual = _f(r.net_mtm_pnl_before_fees, 0.0)
            dec = {
                "matched_roundtrip_pnl": matched,
                "matched_qty": 0.0,
                "residual_inventory_mtm": residual,
                "residual_abs_qty": abs(_f(r.ending_inventory_yes_equiv, 0.0)),
                "matched_by_minute": {k: 0.0 for k in minute},
                "residual_by_minute": {k: 0.0 for k in minute},
            }
        else:
            dec = _fifo_decomposition_one_contract(z, final_mid)

        reconstructed = dec["matched_roundtrip_pnl"] + dec["residual_inventory_mtm"]
        reported = _f(r.net_mtm_pnl_before_fees, 0.0)
        rows.append({
            "ticker": r.ticker,
            "series": r.series,
            "close_ts": r.close_ts,
            "reported_net_pnl": reported,
            "matched_roundtrip_pnl": dec["matched_roundtrip_pnl"],
            "matched_qty": dec["matched_qty"],
            "residual_inventory_mtm": dec["residual_inventory_mtm"],
            "residual_abs_qty": dec["residual_abs_qty"],
            "reconstructed_net_pnl": reconstructed,
            "reconstruction_error": reconstructed - reported,
        })
        for k in minute:
            minute[k]["matched"] += dec["matched_by_minute"][k]
            minute[k]["residual"] += dec["residual_by_minute"][k]

    minute_rows = []
    for k, v in minute.items():
        minute_rows.append({
            "entry_minute": k,
            "matched_roundtrip_pnl": v["matched"],
            "residual_inventory_mtm": v["residual"],
            "net_attributed_pnl": v["matched"] + v["residual"],
        })
    return pd.DataFrame(rows), pd.DataFrame(minute_rows)


def _fill_economics_by_minute(fills: pd.DataFrame) -> pd.DataFrame:
    z = fills[fills.policy == CANDIDATE].copy()
    if z.empty:
        return pd.DataFrame()
    z["entry_minute"] = [
        _minute_bucket(t, c)
        for t, c in zip(
            pd.to_numeric(z.fill_ts, errors="coerce"),
            pd.to_numeric(z.close_ts, errors="coerce"),
        )
    ]
    z = z[z.entry_minute.notna()].copy()
    rows = []
    for minute, q in z.groupby("entry_minute", sort=True):
        rows.append({
            "entry_minute": minute,
            "fill_events": len(q),
            "fill_qty": pd.to_numeric(q.qty, errors="coerce").sum(),
            "bid_qty": pd.to_numeric(q.loc[q.side == "BID", "qty"], errors="coerce").sum(),
            "ask_qty": pd.to_numeric(q.loc[q.side == "ASK", "qty"], errors="coerce").sum(),
            "gross_edge_c": _weighted_mean(q, "gross_edge_at_fill_c"),
            "markout_5s_c": _weighted_mean(q, "markout_5s_c"),
            "markout_15s_c": _weighted_mean(q, "markout_15s_c"),
            "markout_30s_c": _weighted_mean(q, "markout_30s_c"),
            "post_mid_move_30s_c": _weighted_mean(q, "post_mid_move_30s_c"),
        })
    return pd.DataFrame(rows)


def _side_summary(fills: pd.DataFrame) -> pd.DataFrame:
    z = fills[fills.policy == CANDIDATE].copy()
    rows = []
    for side, q in z.groupby("side", sort=True):
        rows.append({
            "side": side,
            "fill_events": len(q),
            "fill_qty": pd.to_numeric(q.qty, errors="coerce").sum(),
            "gross_edge_c": _weighted_mean(q, "gross_edge_at_fill_c"),
            "markout_5s_c": _weighted_mean(q, "markout_5s_c"),
            "markout_15s_c": _weighted_mean(q, "markout_15s_c"),
            "markout_30s_c": _weighted_mean(q, "markout_30s_c"),
            "post_mid_move_30s_c": _weighted_mean(q, "post_mid_move_30s_c"),
        })
    return pd.DataFrame(rows)


def _window_concentration(paired: pd.DataFrame):
    ccol = f"net_mtm_pnl_before_fees__{CANDIDATE}"
    z = paired[["close_ts", "close_time", ccol, "delta_net_mtm_pnl_before_fees"]].copy()
    z = z.rename(columns={ccol: "candidate_window_pnl"})
    z["abs_candidate_pnl"] = z.candidate_window_pnl.abs()
    total = float(z.candidate_window_pnl.sum())
    pos = z[z.candidate_window_pnl > 0].sort_values("candidate_window_pnl", ascending=False)
    neg = z[z.candidate_window_pnl < 0].sort_values("candidate_window_pnl")
    rows = [{
        "metric": "candidate_total_pnl",
        "value": total,
    }, {
        "metric": "positive_window_pct",
        "value": 100.0 * float((z.candidate_window_pnl > 0).mean()),
    }, {
        "metric": "candidate_beats_baseline_window_pct",
        "value": 100.0 * float((z.delta_net_mtm_pnl_before_fees > 0).mean()),
    }, {
        "metric": "top_1_window_pnl",
        "value": float(pos.head(1).candidate_window_pnl.sum()),
    }, {
        "metric": "top_3_windows_pnl",
        "value": float(pos.head(3).candidate_window_pnl.sum()),
    }, {
        "metric": "top_5_windows_pnl",
        "value": float(pos.head(5).candidate_window_pnl.sum()),
    }, {
        "metric": "worst_1_window_pnl",
        "value": float(neg.head(1).candidate_window_pnl.sum()),
    }, {
        "metric": "worst_3_windows_pnl",
        "value": float(neg.head(3).candidate_window_pnl.sum()),
    }, {
        "metric": "worst_5_windows_pnl",
        "value": float(neg.head(5).candidate_window_pnl.sum()),
    }]
    return z.sort_values("candidate_window_pnl", ascending=False), pd.DataFrame(rows)


def run_l3_support_robustness(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"Robustness suite is hard-bound to {EXPECTED_SESSION_NAME}; got {session.name}."
        )

    replay = _replay_output_dir(session.name)
    contracts = _require_csv(replay / "contract_results.csv")
    fills = _require_csv(replay / "fills.csv")
    windows = _require_csv(replay / "window_results.csv")
    policy_summary = _require_csv(replay / "policy_summary.csv")
    chronology = _require_csv(replay / "chronology.csv")

    if output_dir is None:
        output_dir = (
            C.PROJECT_ROOT
            / "results"
            / "kalshi_mm_event_l3_support_robustness"
            / session.name
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paired = _paired_windows(windows)
    bootstrap_dist, bootstrap_stats = _bootstrap_paired_delta(paired)
    loo_window = _leave_one_window_out(paired)
    series = _series_table(contracts)
    loo_series = _leave_one_series_out(series)
    dec_contract, minute_pnl = _decomposition(contracts, fills)
    minute_fills = _fill_economics_by_minute(fills)
    minute = minute_pnl.merge(minute_fills, on="entry_minute", how="outer")
    side = _side_summary(fills)
    ranked_windows, concentration = _window_concentration(paired)

    cand = policy_summary[policy_summary.policy == CANDIDATE]
    base = policy_summary[policy_summary.policy == BASELINE]
    if cand.empty or base.empty:
        raise RuntimeError("Prior replay policy_summary.csv lacks baseline/candidate rows")
    cand = cand.iloc[0]
    base = base.iloc[0]

    candidate_net = float(cand.net_pnl)
    candidate_qty = float(cand.fill_qty)
    fee_cushion_c_per_contract = (
        100.0 * candidate_net / candidate_qty if candidate_qty > EPS else np.nan
    )

    dec_error = pd.to_numeric(dec_contract.reconstruction_error, errors="coerce")
    matched_total = float(pd.to_numeric(dec_contract.matched_roundtrip_pnl, errors="coerce").sum())
    residual_total = float(pd.to_numeric(dec_contract.residual_inventory_mtm, errors="coerce").sum())

    summary = {
        "study_version": STUDY_VERSION,
        "session": session.name,
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "windows": len(paired),
        "candidate_net_pnl": candidate_net,
        "baseline_net_pnl": float(base.net_pnl),
        "incremental_net_vs_baseline": candidate_net - float(base.net_pnl),
        "candidate_fill_qty": candidate_qty,
        "candidate_fee_cushion_c_per_filled_contract": fee_cushion_c_per_contract,
        "candidate_beats_baseline_window_pct": 100.0 * float(paired.candidate_beats_baseline.mean()),
        "candidate_positive_window_pct": 100.0 * float((paired[f"net_mtm_pnl_before_fees__{CANDIDATE}"] > 0).mean()),
        "paired_incremental_median_window": float(pd.to_numeric(paired.delta_net_mtm_pnl_before_fees, errors="coerce").median()),
        "paired_incremental_worst_window": float(pd.to_numeric(paired.delta_net_mtm_pnl_before_fees, errors="coerce").min()),
        "paired_incremental_best_window": float(pd.to_numeric(paired.delta_net_mtm_pnl_before_fees, errors="coerce").max()),
        **bootstrap_stats,
        "loo_window_min_candidate_pnl": float(pd.to_numeric(loo_window.candidate_pnl_without_window, errors="coerce").min()),
        "loo_window_min_incremental_pnl": float(pd.to_numeric(loo_window.incremental_pnl_without_window, errors="coerce").min()),
        "loo_window_candidate_positive_all": bool((loo_window.candidate_pnl_without_window > 0).all()),
        "loo_window_incremental_positive_all": bool((loo_window.incremental_pnl_without_window > 0).all()),
        "loo_series_min_candidate_pnl": float(pd.to_numeric(loo_series.candidate_pnl_without_series, errors="coerce").min()) if len(loo_series) else np.nan,
        "loo_series_min_incremental_pnl": float(pd.to_numeric(loo_series.incremental_pnl_without_series, errors="coerce").min()) if len(loo_series) else np.nan,
        "loo_series_candidate_positive_all": bool((loo_series.candidate_pnl_without_series > 0).all()) if len(loo_series) else False,
        "loo_series_incremental_positive_all": bool((loo_series.incremental_pnl_without_series > 0).all()) if len(loo_series) else False,
        "matched_roundtrip_pnl": matched_total,
        "residual_inventory_mtm": residual_total,
        "decomposition_reconstructed_net": matched_total + residual_total,
        "decomposition_max_abs_error_contract": float(dec_error.abs().max()) if len(dec_error) else np.nan,
    }

    paired.to_csv(out / "paired_windows.csv", index=False)
    bootstrap_dist.to_csv(out / "bootstrap_distribution.csv", index=False)
    pd.DataFrame([bootstrap_stats]).to_csv(out / "bootstrap_summary.csv", index=False)
    loo_window.to_csv(out / "leave_one_window_out.csv", index=False)
    series.to_csv(out / "series_contribution.csv", index=False)
    loo_series.to_csv(out / "leave_one_series_out.csv", index=False)
    dec_contract.to_csv(out / "contract_fifo_decomposition.csv", index=False)
    minute.to_csv(out / "minute_entry_attribution.csv", index=False)
    side.to_csv(out / "side_fill_economics.csv", index=False)
    ranked_windows.to_csv(out / "ranked_windows.csv", index=False)
    concentration.to_csv(out / "window_concentration.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "robustness_summary.csv", index=False)
    (out / "study_config.json").write_text(json.dumps({
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "input_replay_output": str(replay),
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_unit": "15-minute close window",
        "bootstrap_seed": RNG_SEED,
        "new_strategy_rules_tested": False,
        "decomposition": "FIFO matched round trips plus residual M5 MTM; realized PnL attributed to lot entry minute",
        "purpose": "robustness assessment only before fresh V5 development data",
    }, indent=2), encoding="utf-8")

    if show:
        print("\n" + "=" * 148)
        print("L3 SUPPORT Q1 ROBUSTNESS SUITE — DEVELOPMENT ONLY / NO NEW STRATEGY RULE")
        print("=" * 148)
        print(f"session={session.name} | paired windows={len(paired)}")

        print("\nPRIMARY PAIRED RESULT")
        print(f"Candidate net PnL:                 ${candidate_net:+.4f}")
        print(f"Baseline net PnL:                  ${float(base.net_pnl):+.4f}")
        print(f"Incremental net vs baseline:       ${summary['incremental_net_vs_baseline']:+.4f}")
        print(f"Candidate positive windows:        {summary['candidate_positive_window_pct']:.2f}%")
        print(f"Candidate beats baseline windows:  {summary['candidate_beats_baseline_window_pct']:.2f}%")
        print(f"Median paired incremental/window:  ${summary['paired_incremental_median_window']:+.4f}")

        print("\nWINDOW BOOTSTRAP — 20,000 RESAMPLES")
        print(
            "Incremental mean/window 95% CI: "
            f"${bootstrap_stats['bootstrap_mean_delta_per_window_p2_5']:+.4f} .. "
            f"${bootstrap_stats['bootstrap_mean_delta_per_window_p97_5']:+.4f}"
        )
        print(
            "Implied 76-window total 95% CI: "
            f"${bootstrap_stats['bootstrap_total_delta_p2_5']:+.4f} .. "
            f"${bootstrap_stats['bootstrap_total_delta_p97_5']:+.4f}"
        )
        print(
            "Bootstrap P(incremental > 0):    "
            f"{100.0 * bootstrap_stats['bootstrap_probability_incremental_total_gt_0']:.2f}%"
        )

        print("\nLEAVE-ONE-WINDOW-OUT")
        print(f"Candidate stays positive every omission:   {summary['loo_window_candidate_positive_all']}")
        print(f"Incremental stays positive every omission: {summary['loo_window_incremental_positive_all']}")
        print(f"Worst candidate total after omission:      ${summary['loo_window_min_candidate_pnl']:+.4f}")
        print(f"Worst incremental after omission:          ${summary['loo_window_min_incremental_pnl']:+.4f}")

        print("\nBY SERIES")
        print(series.round(4).to_string(index=False))

        print("\nLEAVE-ONE-SERIES-OUT")
        print(loo_series.round(4).to_string(index=False))
        print(f"Candidate stays positive every series omission:   {summary['loo_series_candidate_positive_all']}")
        print(f"Incremental stays positive every series omission: {summary['loo_series_incremental_positive_all']}")

        print("\nFIFO MATCHED vs RESIDUAL INVENTORY")
        print(f"Matched round-trip PnL:    ${matched_total:+.4f}")
        print(f"Residual inventory M5 MTM: ${residual_total:+.4f}")
        print(f"Reconstructed total:       ${matched_total + residual_total:+.4f}")
        print(f"Max contract recon error:  ${summary['decomposition_max_abs_error_contract']:.10f}")

        print("\nENTRY-MINUTE ATTRIBUTION + FILL ECONOMICS")
        print(minute.round(4).to_string(index=False))

        print("\nSIDE FILL ECONOMICS")
        print(side.round(4).to_string(index=False))

        print("\nWINDOW CONCENTRATION")
        print(concentration.round(4).to_string(index=False))
        print("\nTOP 10 WINDOWS")
        print(ranked_windows.head(10).round(4).to_string(index=False))
        print("\nBOTTOM 10 WINDOWS")
        print(ranked_windows.tail(10).sort_values("candidate_window_pnl").round(4).to_string(index=False))

        print("\nPRE-FEE CUSHION")
        print(f"Break-even fee cushion: {fee_cushion_c_per_contract:+.4f} cents / filled contract")

        print("\nINTERPRETATION GUARDRAILS")
        print("  - No strategy parameter or threshold was changed or swept in this suite.")
        print("  - Bootstrap unit is the independent 15-minute close window, not individual fills.")
        print("  - Leave-one-out stability is descriptive; this remains already-explored development data.")
        print("  - Minute PnL is FIFO-attributed to the ENTRY minute of each inventory lot.")
        print("  - The next legitimate place to compare new simple candidate rules is fresh V5-DEVELOPMENT data.")
        print(f"\nOUTPUTS: {out}")
        print("=" * 148)

    return {
        "output_dir": out,
        "summary": summary,
        "paired_windows": paired,
        "bootstrap": bootstrap_dist,
        "leave_one_window_out": loo_window,
        "series": series,
        "leave_one_series_out": loo_series,
        "decomposition": dec_contract,
        "minute": minute,
        "side": side,
        "ranked_windows": ranked_windows,
        "concentration": concentration,
        "chronology": chronology,
    }
