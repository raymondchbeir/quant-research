from __future__ import annotations

"""Fill-conditioned nonlinear toxicity-aware market-making development V2.

DEVELOPMENT ONLY. This module is hard-bound to the already-burned compact
session 20260813_190334 and refuses any other session name. In particular it
must never read 20260814_185216, which remains reserved for later validation of
one frozen candidate.

Why V2 exists
-------------
V1 predicted raw 30-second midpoint movement with a linear ridge model. Its
chronological internal-test correlation collapsed, although side suppression
reduced some adverse selection. V2 changes the target and model architecture:

1. Build the same causal 1 Hz exact-natural-4c state features as V1.
2. Replay the unchanged NAT4->2 baseline once on the development session.
3. Join each realized baseline fill to the state at which its quote episode
   opened.
4. Train TWO nonlinear boosted-tree regressors:
      BID model target = realized signed 30s BID fill edge (markout_30s_c)
      ASK model target = realized signed 30s ASK fill edge (markout_30s_c)
   Thus the target is the economic outcome we care about conditional on an
   actual historical fill, not raw future midpoint movement.
5. Chronological three-way development protocol:
      first 50% windows       -> TRAIN
      next 25% windows        -> DEV_DIAGNOSTIC
      last 25% windows        -> FINAL_INTERNAL_HOLDOUT
   A stage-1 model is trained only on TRAIN and evaluated on DEV_DIAGNOSTIC.
   A final model is then fit on TRAIN+DEV_DIAGNOSTIC using the same fixed
   hyperparameters and evaluated once on FINAL_INTERNAL_HOLDOUT.
6. Fixed economic quote gate, no threshold sweep:
      quote BID iff predicted BID fill edge > 0
      quote ASK iff predicted ASK fill edge > 0
7. Post-fill toxicity state variant:
   after a fill on a side, that same side is locked (in addition to the
   existing 3s cooldown) until its predicted edge is positive AND exceeds the
   opposite-side predicted edge. No time-duration threshold is tuned.
8. Compare on the final internal holdout only:
      BASELINE_NAT4_TO_2
      TREE_FILL_EDGE_GT0
      TREE_FILL_EDGE_POSTFILL_STATE
      ORACLE_30S_LOOKAHEAD_CEILING

The oracle uses future information and is only an upper-bound diagnostic.

Historical limitations remain:
- BBO is persisted at 1 Hz, so sub-second BBO awareness is not testable.
- Q100 inside-spread quoting is counterfactual; public future flow is treated
  as exogenous even though our hypothetical quote could have changed it.
- Real exchange queue/market-impact/live-maker behavior is not testable here.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "mm_fill_conditioned_toxicity_v2 requires scikit-learn. "
        "Install it in the quant-proj environment before running."
    ) from exc

from . import mm_state_aware_toxicity_research_v1 as V1
from . import mm_nat4_to_2_inventory_target_dev_v1 as INV
from . import mm_exact_quote_lifetime_m1_m5_v1 as L
from . import mm_oos_4c_audit_replay as O
from . import mm_oos_4c_compact_recorder_v2 as R

STUDY_VERSION = "FILL_CONDITIONED_TOXICITY_MM_DEV_V2"
EXPECTED_SESSION_NAME = "20260813_190334"
RESERVED_VALIDATION_SESSION_NAME = "20260814_185216"
EPS = 1e-9
MARKOUTS = (5, 15, 30, 60)
TRAIN_END_FRAC = 0.50
DEV_END_FRAC = 0.75
RANDOM_SEED = 20260814

# Fixed before final internal holdout. No hyperparameter sweep is run here.
TREE_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 140,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 10.0,
    "early_stopping": False,
    "random_state": RANDOM_SEED,
}


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _numeric_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].apply(pd.to_numeric, errors="coerce").copy()
    return x.replace([np.inf, -np.inf], np.nan)


def _split_windows(opp: pd.DataFrame):
    windows = np.asarray(sorted(pd.to_numeric(opp["close_ts"], errors="coerce").dropna().unique()), dtype=float)
    if len(windows) < 12:
        raise RuntimeError(f"Need at least 12 windows for three-way development split; got {len(windows)}")
    c1 = max(1, int(np.floor(len(windows) * TRAIN_END_FRAC)))
    c2 = max(c1 + 1, int(np.floor(len(windows) * DEV_END_FRAC)))
    c2 = min(c2, len(windows) - 1)
    return {
        "TRAIN": set(windows[:c1]),
        "DEV_DIAGNOSTIC": set(windows[c1:c2]),
        "FINAL_INTERNAL_HOLDOUT": set(windows[c2:]),
    }


def _assign_split(close_ts: pd.Series, splits) -> np.ndarray:
    out = np.full(len(close_ts), "UNKNOWN", dtype=object)
    x = pd.to_numeric(close_ts, errors="coerce").to_numpy(float)
    for label, vals in splits.items():
        mask = np.asarray([v in vals for v in x], dtype=bool)
        out[mask] = label
    return out


def _baseline_all(audit, samples, trades):
    good = audit[audit["quality_ok"].astype(bool)].copy()
    sim_meta = {
        str(r.ticker): {
            "ticker": str(r.ticker),
            "series": str(r.series),
            "close_ts": float(r.close_ts),
        }
        for r in good.itertuples(index=False)
    }
    targets = sorted(sim_meta, key=lambda t: (sim_meta[t]["close_ts"], t))
    print(f"Baseline full-development replay for fill labels: {len(targets)} contracts")
    cdf, fdf, edf, counts = V1._simulate_policy(
        "BASELINE_LABEL_SOURCE", targets, sim_meta, samples, trades, allowed=None
    )
    return cdf, fdf, edf, counts, sim_meta


def _join_fill_training(fills: pd.DataFrame, episodes: pd.DataFrame, opp: pd.DataFrame, splits):
    if fills.empty or episodes.empty:
        raise RuntimeError("Baseline replay produced no fills/episodes")

    ep_cols = [c for c in ["episode_id", "ticker", "join_ts"] if c in episodes.columns]
    if len(ep_cols) < 3:
        raise KeyError("Baseline quote episodes need episode_id, ticker, join_ts")
    ep = episodes[ep_cols].drop_duplicates("episode_id").copy()

    state = opp.copy()
    state["join_ts"] = pd.to_numeric(state["ts"], errors="coerce")
    features = V1._feature_columns(state)
    state_cols = ["ticker", "join_ts", "close_ts", "series"] + features
    state = state[state_cols].drop_duplicates(["ticker", "join_ts"]).copy()

    x = fills.merge(ep, on=["episode_id", "ticker"], how="left", validate="many_to_one")
    x = x.merge(state, on=["ticker", "join_ts"], how="left", validate="many_to_one")
    x["realized_fill_edge_30s_c"] = pd.to_numeric(x.get("markout_30s_c"), errors="coerce")
    x["realized_fill_edge_5s_c"] = pd.to_numeric(x.get("markout_5s_c"), errors="coerce")
    x["realized_fill_edge_15s_c"] = pd.to_numeric(x.get("markout_15s_c"), errors="coerce")
    x["realized_fill_edge_60s_c"] = pd.to_numeric(x.get("markout_60s_c"), errors="coerce")
    x["split"] = _assign_split(x["close_ts"], splits)
    x["qty"] = pd.to_numeric(x["qty"], errors="coerce")
    x = x[np.isfinite(x["realized_fill_edge_30s_c"]) & np.isfinite(x["qty"]) & (x["qty"] > 0)].copy()
    if x.empty:
        raise RuntimeError("No baseline fills have finite joined 30s economic edge")
    return x, features


def _fit_tree(df: pd.DataFrame, features: list[str], side: str):
    z = df[df["side"].astype(str) == side].copy()
    y = pd.to_numeric(z["realized_fill_edge_30s_c"], errors="coerce").to_numpy(float)
    w = pd.to_numeric(z["qty"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
    z = z.loc[ok].copy()
    y, w = y[ok], w[ok]
    if len(z) < 80:
        raise RuntimeError(f"Only {len(z)} usable {side} fill labels; need at least 80")
    model = HistGradientBoostingRegressor(**TREE_PARAMS)
    model.fit(_numeric_X(z, features), y, sample_weight=w)
    return model, len(z), float(w.sum())


def _predict(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=float)
    return np.asarray(model.predict(_numeric_X(df, features)), dtype=float)


def _weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def _model_perf(df: pd.DataFrame, pred_col: str, split: str, side: str, stage: str):
    z = df[(df["split"] == split) & (df["side"].astype(str) == side)].copy()
    y = pd.to_numeric(z["realized_fill_edge_30s_c"], errors="coerce")
    p = pd.to_numeric(z[pred_col], errors="coerce")
    w = pd.to_numeric(z["qty"], errors="coerce")
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y, p, w = y[ok], p[ok], w[ok]
    if len(y) == 0:
        return {"stage": stage, "split": split, "side": side, "fills": 0}
    err = p - y
    zz = pd.DataFrame({"y": y, "p": p})
    return {
        "stage": stage,
        "split": split,
        "side": side,
        "fills": len(y),
        "fill_qty": float(w.sum()),
        "actual_edge_qw_c": _weighted_mean(y, w),
        "pred_edge_qw_c": _weighted_mean(p, w),
        "mae_qw_c": _weighted_mean(np.abs(err), w),
        "rmse_qw_c": float(np.sqrt(_weighted_mean(np.asarray(err) ** 2, w))),
        "pearson": zz.corr(method="pearson").iloc[0, 1] if len(zz) >= 3 else np.nan,
        "spearman": zz.corr(method="spearman").iloc[0, 1] if len(zz) >= 3 else np.nan,
        "positive_edge_accuracy_pct": 100.0 * ((y > 0) == (p > 0)).mean(),
        "actual_positive_fill_pct": 100.0 * (y > 0).mean(),
        "pred_positive_fill_pct": 100.0 * (p > 0).mean(),
        "actual_toxic_fill_pct": 100.0 * (y < 0).mean(),
    }


def _prediction_buckets(df: pd.DataFrame, pred_col: str, split: str, side: str, label: str):
    z = df[(df["split"] == split) & (df["side"].astype(str) == side)].copy()
    z[pred_col] = pd.to_numeric(z[pred_col], errors="coerce")
    z["realized_fill_edge_30s_c"] = pd.to_numeric(z["realized_fill_edge_30s_c"], errors="coerce")
    z["qty"] = pd.to_numeric(z["qty"], errors="coerce")
    z = z[np.isfinite(z[pred_col]) & np.isfinite(z.realized_fill_edge_30s_c) & np.isfinite(z.qty) & (z.qty > 0)].copy()
    if len(z) < 10:
        return pd.DataFrame()
    try:
        z["bucket"] = pd.qcut(z[pred_col], 5, duplicates="drop")
    except Exception:
        z["bucket"] = "ALL"
    rows = []
    for i, (b, g) in enumerate(z.groupby("bucket", observed=True, sort=True), 1):
        rows.append({
            "model_stage": label,
            "split": split,
            "side": side,
            "bucket_rank": i,
            "bucket": str(b),
            "fills": len(g),
            "fill_qty": float(g.qty.sum()),
            "pred_edge_qw_c": _weighted_mean(g[pred_col], g.qty),
            "actual_edge_30s_qw_c": _weighted_mean(g.realized_fill_edge_30s_c, g.qty),
            "actual_edge_5s_qw_c": _weighted_mean(g.realized_fill_edge_5s_c, g.qty),
            "actual_edge_15s_qw_c": _weighted_mean(g.realized_fill_edge_15s_c, g.qty),
            "actual_edge_60s_qw_c": _weighted_mean(g.realized_fill_edge_60s_c, g.qty),
            "positive_fill_pct": 100.0 * (g.realized_fill_edge_30s_c > 0).mean(),
        })
    return pd.DataFrame(rows)


def _permutation_table(model, df, features, side, split="DEV_DIAGNOSTIC"):
    z = df[(df["split"] == split) & (df["side"].astype(str) == side)].copy()
    y = pd.to_numeric(z["realized_fill_edge_30s_c"], errors="coerce")
    ok = np.isfinite(y)
    z, y = z.loc[ok].copy(), y.loc[ok]
    if len(z) < 30:
        return pd.DataFrame()
    # Keep this diagnostic bounded in runtime; deterministic sample if needed.
    if len(z) > 1000:
        z = z.sample(1000, random_state=RANDOM_SEED)
        y = pd.to_numeric(z["realized_fill_edge_30s_c"], errors="coerce")
    pi = permutation_importance(
        model,
        _numeric_X(z, features),
        y.to_numpy(float),
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    out = pd.DataFrame({
        "side": side,
        "feature": features,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    })
    out["abs_importance"] = out.importance_mean.abs()
    return out.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def _score_opportunities(opp, features, bid_model, ask_model, prefix):
    x = opp.copy()
    x[f"{prefix}_bid_edge_c"] = _predict(bid_model, x, features)
    x[f"{prefix}_ask_edge_c"] = _predict(ask_model, x, features)
    b = x[f"{prefix}_bid_edge_c"]
    a = x[f"{prefix}_ask_edge_c"]
    x[f"{prefix}_quote_state"] = np.select(
        [(b > 0) & (a > 0), (b > 0) & ~(a > 0), ~(b > 0) & (a > 0)],
        ["TWO_SIDED", "BID_ONLY", "ASK_ONLY"],
        default="NO_QUOTE",
    )
    return x


def _edge_map(opp: pd.DataFrame, bid_col: str, ask_col: str, oracle=False):
    out = {}
    for r in opp.itertuples(index=False):
        ticker, ts = str(r.ticker), float(r.ts)
        if oracle:
            move = _f(getattr(r, "future_mid_move_30s_c", np.nan))
            if not np.isfinite(move):
                continue
            out[(ticker, ts, "BID")] = 1.0 + move
            out[(ticker, ts, "ASK")] = 1.0 - move
        else:
            b = _f(getattr(r, bid_col, np.nan))
            a = _f(getattr(r, ask_col, np.nan))
            if np.isfinite(b):
                out[(ticker, ts, "BID")] = b
            if np.isfinite(a):
                out[(ticker, ts, "ASK")] = a
    return out


def _simulate_tree_policy(name, targets, sim_meta, samples, trades, edge_map=None, postfill_state=False):
    original = INV._base_reasons
    holder = {
        "ticker": None,
        "seen_fill": {"BID": -np.inf, "ASK": -np.inf},
        "locked": {"BID": False, "ASK": False},
    }

    if edge_map is not None:
        def gated(side, s, last_fill_ts, now_t, mom, flow):
            reasons = list(original(side, s, last_fill_ts, now_t, mom, flow))
            ticker = holder["ticker"]
            key = (ticker, float(s.t), side)
            edge = _f(edge_map.get(key))
            if not np.isfinite(edge) or edge <= 0.0 + EPS:
                reasons.append("TREE_FILL_EDGE")

            if postfill_state:
                lf = _f(last_fill_ts.get(side, -np.inf), -np.inf)
                if np.isfinite(lf) and lf > holder["seen_fill"][side] + EPS:
                    holder["seen_fill"][side] = lf
                    holder["locked"][side] = True

                if holder["locked"][side]:
                    other = "ASK" if side == "BID" else "BID"
                    other_edge = _f(edge_map.get((ticker, float(s.t), other)))
                    # Exit the post-fill toxic state only when this side is not
                    # merely positive, but stronger than the opposite side.
                    if (
                        np.isfinite(edge)
                        and edge > 0.0 + EPS
                        and (not np.isfinite(other_edge) or edge > other_edge + EPS)
                    ):
                        holder["locked"][side] = False
                    else:
                        reasons.append("POST_FILL_TOXIC_STATE")
            return reasons

        INV._base_reasons = gated

    contracts, fills, episodes, counts = [], [], [], Counter()
    try:
        for i, ticker in enumerate(targets, 1):
            holder["ticker"] = ticker
            holder["seen_fill"] = {"BID": -np.inf, "ASK": -np.inf}
            holder["locked"] = {"BID": False, "ASK": False}
            e, f, c, k = INV._simulate_contract(
                ticker, sim_meta[ticker], samples.get(ticker, []), trades.get(ticker, [])
            )
            for x in e:
                x["policy_name"] = name
            for x in f:
                x["policy_name"] = name
            if c is not None:
                c["policy_name"] = name
                contracts.append(c)
            episodes.extend(e)
            fills.extend(f)
            counts.update(k)
            if i % 100 == 0 or i == len(targets):
                print(
                    f"  {name}: {i}/{len(targets)} | fills={len(fills)} | "
                    f"qty={sum(float(x['qty']) for x in fills):.2f}"
                )
    finally:
        INV._base_reasons = original

    return (
        pd.DataFrame(contracts),
        pd.DataFrame(fills),
        pd.DataFrame(episodes),
        pd.DataFrame([{"policy_name": name, "reason": k, "count": v} for k, v in counts.most_common()]),
    )


def _policy_outputs(name, cdf, fdf, close_map):
    wdf = L._window_summary(cdf) if len(cdf) else pd.DataFrame()
    if len(wdf):
        wdf["policy"] = name
    summary = V1._policy_summary(name, cdf, fdf, wdf)
    portfolio = V1._portfolio_inventory(name, fdf, close_map, wdf)
    return summary, wdf, portfolio


def _baseline_fill_prediction_buckets(final_fills, final_episodes, final_opp, bid_col, ask_col):
    if final_fills.empty or final_episodes.empty:
        return pd.DataFrame()
    ep = final_episodes[["episode_id", "ticker", "join_ts"]].drop_duplicates("episode_id")
    p = final_opp[["ticker", "ts", bid_col, ask_col]].copy().rename(columns={"ts": "join_ts"})
    x = final_fills.merge(ep, on=["episode_id", "ticker"], how="left")
    x = x.merge(p, on=["ticker", "join_ts"], how="left")
    x["pred_edge_c"] = np.where(
        x.side.astype(str) == "BID",
        pd.to_numeric(x[bid_col], errors="coerce"),
        pd.to_numeric(x[ask_col], errors="coerce"),
    )
    x["actual_edge_30s_c"] = pd.to_numeric(x.get("markout_30s_c"), errors="coerce")
    x["qty"] = pd.to_numeric(x["qty"], errors="coerce")
    x = x[np.isfinite(x.pred_edge_c) & np.isfinite(x.actual_edge_30s_c) & np.isfinite(x.qty) & (x.qty > 0)].copy()
    rows = []
    for side, z0 in x.groupby("side", sort=True):
        z = z0.copy()
        try:
            z["bucket"] = pd.qcut(z.pred_edge_c, 5, duplicates="drop")
        except Exception:
            z["bucket"] = "ALL"
        for i, (b, g) in enumerate(z.groupby("bucket", observed=True, sort=True), 1):
            rows.append({
                "side": side,
                "bucket_rank": i,
                "bucket": str(b),
                "fill_events": len(g),
                "fill_qty": float(g.qty.sum()),
                "pred_edge_qw_c": _weighted_mean(g.pred_edge_c, g.qty),
                "actual_edge_30s_qw_c": _weighted_mean(g.actual_edge_30s_c, g.qty),
                "positive_fill_pct": 100.0 * (g.actual_edge_30s_c > 0).mean(),
            })
    return pd.DataFrame(rows)


def _bucket_shape_summary(buckets: pd.DataFrame):
    rows = []
    if buckets.empty:
        return pd.DataFrame()
    for side, z in buckets.groupby("side", sort=True):
        z = z.sort_values("bucket_rank")
        if len(z) >= 3:
            rho = z[["bucket_rank", "actual_edge_30s_qw_c"]].corr(method="spearman").iloc[0, 1]
            monotone = bool(np.all(np.diff(z.actual_edge_30s_qw_c.to_numpy(float)) >= -EPS))
        else:
            rho, monotone = np.nan, False
        rows.append({
            "side": side,
            "buckets": len(z),
            "bucket_rank_vs_actual_edge_spearman": rho,
            "actual_edge_monotone_increasing": monotone,
            "lowest_bucket_actual_edge_c": z.actual_edge_30s_qw_c.iloc[0] if len(z) else np.nan,
            "highest_bucket_actual_edge_c": z.actual_edge_30s_qw_c.iloc[-1] if len(z) else np.nan,
        })
    return pd.DataFrame(rows)


def _print_report(audit, splits, fill_labels, perf, importance, buckets, bucket_shape, policy, portfolio, counts, out):
    print("\n" + "=" * 158)
    print("FILL-CONDITIONED NONLINEAR TOXICITY MM V2 — DEVELOPMENT ONLY")
    print("=" * 158)
    print(
        f"session={EXPECTED_SESSION_NAME} | reserved validation={RESERVED_VALIDATION_SESSION_NAME} NOT READ"
    )
    print(
        f"M1-M5 quality={int(audit.quality_ok.sum())}/{len(audit)} contracts | "
        f"windows={audit.loc[audit.quality_ok, 'close_ts'].nunique()}"
    )
    print(
        "window split: "
        + " | ".join(f"{k}={len(v)}" for k, v in splits.items())
    )
    print(
        f"baseline fill labels={len(fill_labels)} | qty={pd.to_numeric(fill_labels.qty, errors='coerce').sum():.2f}"
    )

    print("\nMODEL PERFORMANCE — TARGET IS REALIZED SIGNED 30s FILL EDGE")
    print(perf.round(4).to_string(index=False))

    if len(importance):
        print("\nTOP NONLINEAR PERMUTATION IMPORTANCE ON DEV_DIAGNOSTIC")
        for side in ("BID", "ASK"):
            z = importance[importance.side == side].head(15)
            print(f"\n{side}")
            print(z[["feature", "importance_mean", "importance_std"]].round(4).to_string(index=False))

    if len(buckets):
        print("\nFINAL INTERNAL HOLDOUT — BASELINE FILLS BY PREDICTED SIDE-SPECIFIC EDGE")
        print(buckets.round(4).to_string(index=False))
    if len(bucket_shape):
        print("\nFINAL BUCKET SHAPE CHECK")
        print(bucket_shape.round(4).to_string(index=False))

    print("\nFINAL INTERNAL HOLDOUT POLICY REPLAY")
    print(policy.round(4).to_string(index=False))

    if len(portfolio):
        ps = portfolio.groupby("policy", as_index=False).agg(
            windows=("close_ts", "count"),
            median_max_abs_portfolio_inventory=("max_abs_portfolio_inventory", "median"),
            p95_max_abs_portfolio_inventory=("max_abs_portfolio_inventory", lambda x: x.quantile(.95)),
            median_abs_ending_portfolio_inventory=("ending_portfolio_inventory", lambda x: x.abs().median()),
        )
        print("\nPORTFOLIO INVENTORY")
        print(ps.round(3).to_string(index=False))

    if len(counts):
        model_counts = counts[counts.reason.astype(str).str.contains("TREE_FILL_EDGE|POST_FILL_TOXIC_STATE", regex=True, na=False)]
        if len(model_counts):
            print("\nMODEL / POST-FILL BLOCK COUNTS")
            print(model_counts.to_string(index=False))

    print("\nINTERPRETATION GUARDRAILS")
    print("  - FINAL_INTERNAL_HOLDOUT is still development data, not external OOS validation.")
    print("  - Reserved V3 session is not read anywhere in this module.")
    print("  - No model threshold sweep: economic gate is fixed at predicted fill edge > 0c.")
    print("  - Post-fill state has no tuned duration; it exits only when same-side edge is positive and stronger than opposite-side edge.")
    print("  - No asset filter, no BID-only/ASK-only rule, no pre-M1 cutoff rescue.")
    print("  - 1Hz BBO and Q100 counterfactual limitations remain.")
    print("  - Oracle uses future data and is not a tradable strategy.")
    print("Outputs:", out)
    print("=" * 158)


def run_fill_conditioned_toxicity_v2(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"V2 is hard-bound to development session {EXPECTED_SESSION_NAME}; got {session.name}. "
            f"Reserved validation session {RESERVED_VALIDATION_SESSION_NAME} must remain untouched."
        )
    if not session.exists():
        raise FileNotFoundError(session)

    meta = O._metadata(session)
    samples, info, duplicates, bbo_stats = O._bbo(session, meta)
    trades, trade_stats = O._trades(session, meta)
    audit = O._audit(meta, samples, info, duplicates, trades)
    if audit.empty:
        raise RuntimeError("No compact-session contracts found")
    good = audit[audit.quality_ok.astype(bool)].copy()
    if good.empty:
        raise RuntimeError("No contracts pass the unchanged 80% M1-M5 quality gate")

    print(f"Quality pass: {len(good)}/{len(audit)} contracts | {good.close_ts.nunique()} windows")
    print("Building causal exact-4c state table...")
    opp, pre_contract = V1._build_opportunities(audit, samples, info, trades)
    if opp.empty:
        raise RuntimeError("No exact-4c opportunity states found")

    splits = _split_windows(opp)
    opp["split"] = _assign_split(opp["close_ts"], splits)

    baseline_c_all, baseline_f_all, baseline_e_all, baseline_counts_all, sim_meta_all = _baseline_all(
        audit, samples, trades
    )
    fill_labels, features = _join_fill_training(baseline_f_all, baseline_e_all, opp, splits)

    print(
        "Fill labels by split/side:\n"
        + fill_labels.groupby(["split", "side"])["qty"].agg(["count", "sum"]).round(2).to_string()
    )

    # Stage 1: fit TRAIN only, inspect DEV_DIAGNOSTIC without changing params.
    train_labels = fill_labels[fill_labels.split == "TRAIN"].copy()
    bid_stage1, _, _ = _fit_tree(train_labels, features, "BID")
    ask_stage1, _, _ = _fit_tree(train_labels, features, "ASK")
    fill_labels["stage1_pred_edge_c"] = np.nan
    for side, model in (("BID", bid_stage1), ("ASK", ask_stage1)):
        mask = fill_labels.side.astype(str) == side
        fill_labels.loc[mask, "stage1_pred_edge_c"] = _predict(model, fill_labels.loc[mask], features)

    stage1_perf = []
    for split in ("TRAIN", "DEV_DIAGNOSTIC"):
        for side in ("BID", "ASK"):
            stage1_perf.append(_model_perf(fill_labels, "stage1_pred_edge_c", split, side, "STAGE1_TRAIN_ONLY"))

    imp_parts = [
        _permutation_table(bid_stage1, fill_labels, features, "BID"),
        _permutation_table(ask_stage1, fill_labels, features, "ASK"),
    ]
    importance = pd.concat([x for x in imp_parts if len(x)], ignore_index=True) if any(len(x) for x in imp_parts) else pd.DataFrame()

    # Final model: fixed architecture refit on TRAIN+DEV, then score final holdout once.
    fit_labels = fill_labels[fill_labels.split.isin(["TRAIN", "DEV_DIAGNOSTIC"])].copy()
    bid_final, bid_n, bid_qty = _fit_tree(fit_labels, features, "BID")
    ask_final, ask_n, ask_qty = _fit_tree(fit_labels, features, "ASK")

    fill_labels["final_pred_edge_c"] = np.nan
    for side, model in (("BID", bid_final), ("ASK", ask_final)):
        mask = fill_labels.side.astype(str) == side
        fill_labels.loc[mask, "final_pred_edge_c"] = _predict(model, fill_labels.loc[mask], features)

    final_perf = []
    for split in ("TRAIN", "DEV_DIAGNOSTIC", "FINAL_INTERNAL_HOLDOUT"):
        for side in ("BID", "ASK"):
            final_perf.append(_model_perf(fill_labels, "final_pred_edge_c", split, side, "FINAL_TRAIN_PLUS_DEV"))
    perf = pd.DataFrame(stage1_perf + final_perf)

    # Score every exact4 opportunity state using final side-specific models.
    opp = _score_opportunities(opp, features, bid_final, ask_final, "final")

    # Final holdout calibration on actual baseline fills, before policy replay.
    hold_fill = fill_labels[fill_labels.split == "FINAL_INTERNAL_HOLDOUT"].copy()
    bucket_parts = []
    for side in ("BID", "ASK"):
        b = _prediction_buckets(hold_fill, "final_pred_edge_c", "FINAL_INTERNAL_HOLDOUT", side, "FINAL_MODEL")
        if len(b):
            bucket_parts.append(b)
    model_fill_buckets = pd.concat(bucket_parts, ignore_index=True) if bucket_parts else pd.DataFrame()

    # Policy replay on FINAL_INTERNAL_HOLDOUT only.
    final_windows = splits["FINAL_INTERNAL_HOLDOUT"]
    final_good = good[good.close_ts.astype(float).isin(final_windows)].copy()
    sim_meta = {
        str(r.ticker): {
            "ticker": str(r.ticker),
            "series": str(r.series),
            "close_ts": float(r.close_ts),
        }
        for r in final_good.itertuples(index=False)
    }
    targets = sorted(sim_meta, key=lambda t: (sim_meta[t]["close_ts"], t))
    close_map = {t: float(m["close_ts"]) for t, m in sim_meta.items()}
    final_opp = opp[opp.split == "FINAL_INTERNAL_HOLDOUT"].copy()
    tree_edges = _edge_map(final_opp, "final_bid_edge_c", "final_ask_edge_c", oracle=False)
    oracle_edges = _edge_map(final_opp, "final_bid_edge_c", "final_ask_edge_c", oracle=True)

    print(f"Final internal holdout replay: {len(final_windows)} windows | {len(targets)} quality contracts")
    policy_runs = []
    baseline_c, baseline_f, baseline_e, baseline_counts = _simulate_tree_policy(
        "BASELINE_NAT4_TO_2", targets, sim_meta, samples, trades, edge_map=None, postfill_state=False
    )
    policy_runs.append(("BASELINE_NAT4_TO_2", baseline_c, baseline_f, baseline_e, baseline_counts))

    tree_c, tree_f, tree_e, tree_counts = _simulate_tree_policy(
        "TREE_FILL_EDGE_GT0", targets, sim_meta, samples, trades, edge_map=tree_edges, postfill_state=False
    )
    policy_runs.append(("TREE_FILL_EDGE_GT0", tree_c, tree_f, tree_e, tree_counts))

    post_c, post_f, post_e, post_counts = _simulate_tree_policy(
        "TREE_FILL_EDGE_POSTFILL_STATE", targets, sim_meta, samples, trades, edge_map=tree_edges, postfill_state=True
    )
    policy_runs.append(("TREE_FILL_EDGE_POSTFILL_STATE", post_c, post_f, post_e, post_counts))

    oracle_c, oracle_f, oracle_e, oracle_counts = _simulate_tree_policy(
        "ORACLE_30S_LOOKAHEAD_CEILING", targets, sim_meta, samples, trades, edge_map=oracle_edges, postfill_state=False
    )
    policy_runs.append(("ORACLE_30S_LOOKAHEAD_CEILING", oracle_c, oracle_f, oracle_e, oracle_counts))

    policy_rows, window_parts, portfolio_parts, count_parts = [], [], [], []
    for name, cdf, fdf, edf, counts in policy_runs:
        summary, wdf, portfolio = _policy_outputs(name, cdf, fdf, close_map)
        policy_rows.append(summary)
        if len(wdf):
            window_parts.append(wdf)
        if len(portfolio):
            portfolio_parts.append(portfolio)
        if len(counts):
            count_parts.append(counts)

    policy = pd.DataFrame(policy_rows)
    policy_windows = pd.concat(window_parts, ignore_index=True) if window_parts else pd.DataFrame()
    portfolio = pd.concat(portfolio_parts, ignore_index=True) if portfolio_parts else pd.DataFrame()
    policy_counts = pd.concat(count_parts, ignore_index=True) if count_parts else pd.DataFrame()

    baseline_pred_buckets = _baseline_fill_prediction_buckets(
        baseline_f, baseline_e, final_opp, "final_bid_edge_c", "final_ask_edge_c"
    )
    bucket_shape = _bucket_shape_summary(baseline_pred_buckets)

    if output_dir is None:
        output_dir = (
            R.PROJECT_ROOT
            / "results"
            / "kalshi_fill_conditioned_toxicity_mm_v2_dev"
            / f"{session.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    outputs = {
        "contract_quality.csv": audit,
        "pre_m1_contract_range.csv": pre_contract,
        "opportunity_state_features_scored.csv": opp,
        "baseline_fill_training_labels.csv": fill_labels,
        "model_performance.csv": perf,
        "model_permutation_importance_dev.csv": importance,
        "fill_label_prediction_buckets_final.csv": model_fill_buckets,
        "baseline_fill_prediction_buckets_final.csv": baseline_pred_buckets,
        "baseline_fill_bucket_shape_final.csv": bucket_shape,
        "policy_summary_final_internal_holdout.csv": policy,
        "policy_windows_final_internal_holdout.csv": policy_windows,
        "portfolio_inventory_final_internal_holdout.csv": portfolio,
        "policy_counts_final_internal_holdout.csv": policy_counts,
        "baseline_fills_final_internal_holdout.csv": baseline_f,
        "tree_fills_final_internal_holdout.csv": tree_f,
        "postfill_tree_fills_final_internal_holdout.csv": post_f,
        "oracle_fills_final_internal_holdout.csv": oracle_f,
        "baseline_quote_episodes_final_internal_holdout.csv": baseline_e,
        "tree_quote_episodes_final_internal_holdout.csv": tree_e,
        "postfill_tree_quote_episodes_final_internal_holdout.csv": post_e,
    }
    for fname, df in outputs.items():
        df.to_csv(out / fname, index=False)

    config = {
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "hard_bound_session_name": EXPECTED_SESSION_NAME,
        "reserved_validation_session_name": RESERVED_VALIDATION_SESSION_NAME,
        "reserved_validation_accessed": False,
        "m1_m5_quality_gate_pct": 80.0,
        "opportunity": "natural spread exactly 4c (+/-0.05c), NAT4->2 inside-spread mechanism",
        "target": "realized signed fill markout_30s_c from baseline historical fill",
        "side_specific_models": ["BID", "ASK"],
        "model": "HistGradientBoostingRegressor",
        "tree_params": TREE_PARAMS,
        "split": {
            "TRAIN_fraction": TRAIN_END_FRAC,
            "DEV_DIAGNOSTIC_end_fraction": DEV_END_FRAC,
            "unit": "15-minute close_ts window chronological",
        },
        "stage1": "fit TRAIN only; evaluate DEV_DIAGNOSTIC",
        "final_model": "same fixed architecture refit TRAIN+DEV; evaluate FINAL_INTERNAL_HOLDOUT once",
        "final_fit_labels": {
            "BID_fills": bid_n,
            "BID_qty": bid_qty,
            "ASK_fills": ask_n,
            "ASK_qty": ask_qty,
        },
        "economic_gate": "quote side iff predicted side-specific realized fill edge > 0c",
        "postfill_state": (
            "after a fill, same side remains locked beyond ordinary cooldown until its predicted edge "
            "is positive and exceeds opposite-side predicted edge"
        ),
        "threshold_sweep": False,
        "asset_filters": False,
        "side_filters": False,
        "features": features,
        "limitations": [
            "1Hz BBO only; no sub-second event-time BBO test",
            "Q100 inside-spread historical flow is counterfactual/exogenous",
            "training is conditional on historical baseline fills and therefore has fill-selection bias",
            "live maker / market impact cannot be tested on historical archive",
            "FINAL_INTERNAL_HOLDOUT remains development data, not external validation",
        ],
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    if show:
        _print_report(
            audit, splits, fill_labels, perf, importance,
            baseline_pred_buckets, bucket_shape, policy, portfolio, policy_counts, out,
        )

    return {
        "output_dir": out,
        "opportunities": opp,
        "fill_labels": fill_labels,
        "model_performance": perf,
        "permutation_importance": importance,
        "prediction_buckets": baseline_pred_buckets,
        "bucket_shape": bucket_shape,
        "policy_summary": policy,
        "policy_windows": policy_windows,
        "portfolio_inventory": portfolio,
        "policy_counts": policy_counts,
        "baseline_fills": baseline_f,
        "tree_fills": tree_f,
        "postfill_tree_fills": post_f,
        "oracle_fills": oracle_f,
    }
