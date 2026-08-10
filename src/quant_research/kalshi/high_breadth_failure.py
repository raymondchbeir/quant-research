from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .window_regime import load_live_primary_signals

PRIMARY_DIR = "PRIMARY_SHADOW_M5_MINUS3C_15S_3CT_HOLD_V1"

# Keep this small and interpretable. All are known at M5 / order-post time.
DEFAULT_FEATURES = [
    "conf_mean_c", "conf_min_c", "conf_max_c", "conf_std_c", "mid_dispersion_c",
    "entry_mean_c", "entry_min_c", "entry_max_c", "entry_std_c",
    "spread_mean_c", "spread_max_c",
    "share_conf_ge20", "share_conf_ge30", "share_entry_ge70", "share_entry_ge80",
    "btc_ret_5m_bp", "btc_ret_15m_bp", "btc_ret_30m_bp", "btc_ret_60m_bp", "btc_ret_120m_bp",
    "btc_abs_15m_bp", "btc_rv_60m_bp", "btc_rv_6h_bp", "btc_rv_24h_bp",
    "btc_trend_alignment", "btc_5v60_reversal", "btc_dist_ma60_bp",
]

SPLITS = {
    "APR_DISCOVERY": ("2026-04-01", "2026-05-01"),
    "MAY_VALID": ("2026-05-01", "2026-06-01"),
    "JUN1_28_LOCKED": ("2026-06-01", "2026-06-29"),
    "JUN29_JUL3_STRESS": ("2026-06-29", "2026-07-04"),
    "AUG10_LIVE": ("2026-08-10", "2026-08-11"),
}


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC")
    return df[(df["decision_time"] >= start) & (df["decision_time"] < end)].copy()


def _edge_stats(g: pd.DataFrame, flag: pd.Series):
    flag = flag.reindex(g.index).fillna(False).astype(bool)
    a = g[flag]
    b = g[~flag]
    return {
        "windows": len(g),
        "flagged": len(a),
        "flag_rate_pct": 100.0 * len(a) / len(g) if len(g) else np.nan,
        "flagged_edge_c": 100.0 * a["signal_edge"].mean() if len(a) else np.nan,
        "unflagged_edge_c": 100.0 * b["signal_edge"].mean() if len(b) else np.nan,
        "flag_minus_unflagged_c": 100.0 * (a["signal_edge"].mean() - b["signal_edge"].mean()) if len(a) and len(b) else np.nan,
        "flagged_negative_pct": 100.0 * (a["signal_edge"] < 0).mean() if len(a) else np.nan,
    }


def _apply_rule(g: pd.DataFrame, feature: str, direction: str, threshold: float) -> pd.Series:
    x = pd.to_numeric(g[feature], errors="coerce")
    return (x >= threshold) if direction == "HIGH" else (x <= threshold)


def discover_high_breadth_rules(scored_windows: pd.DataFrame, features=None, min_flagged=8):
    """Discover one-sided tail rules on April only, validate sign on May.

    The threshold is fixed from April's 25th/75th percentile. June, the stress episode,
    and August never influence threshold or direction selection.
    """
    features = features or DEFAULT_FEATURES
    z = scored_windows.copy()
    z["decision_time"] = pd.to_datetime(z["decision_time"], utc=True, errors="coerce")
    z = z[(z["signals"] >= 3) & z["signal_edge"].notna()].copy()

    parts = {name: _slice(z, *bounds) for name, bounds in SPLITS.items()}
    apr = parts["APR_DISCOVERY"]
    may = parts["MAY_VALID"]

    rows = []
    for feature in features:
        if feature not in z.columns:
            continue
        xa = pd.to_numeric(apr[feature], errors="coerce")
        if xa.notna().sum() < 20 or xa.nunique(dropna=True) < 3:
            continue

        q25, q75 = xa.quantile([0.25, 0.75])
        candidates = [("LOW", float(q25)), ("HIGH", float(q75))]
        apr_candidates = []
        for direction, threshold in candidates:
            flag = _apply_rule(apr, feature, direction, threshold)
            s = _edge_stats(apr, flag)
            if s["flagged"] >= min_flagged and (s["windows"] - s["flagged"]) >= min_flagged:
                apr_candidates.append((s["flag_minus_unflagged_c"], direction, threshold, s))
        if not apr_candidates:
            continue

        # April alone chooses the direction: the tail with the worse edge.
        apr_candidates.sort(key=lambda x: x[0])
        _, direction, threshold, apr_stats = apr_candidates[0]

        row = {
            "feature": feature,
            "direction": direction,
            "threshold": threshold,
            "apr_windows": len(apr),
            "apr_flag_rate_pct": apr_stats["flag_rate_pct"],
            "apr_flagged_edge_c": apr_stats["flagged_edge_c"],
            "apr_unflagged_edge_c": apr_stats["unflagged_edge_c"],
            "apr_effect_c": apr_stats["flag_minus_unflagged_c"],
        }

        for split_name in ["MAY_VALID", "JUN1_28_LOCKED", "JUN29_JUL3_STRESS", "AUG10_LIVE"]:
            g = parts[split_name]
            if g.empty:
                stats = {k: np.nan for k in ["flag_rate_pct", "flagged_edge_c", "unflagged_edge_c", "flag_minus_unflagged_c", "flagged_negative_pct"]}
                nwin = 0
            else:
                stats = _edge_stats(g, _apply_rule(g, feature, direction, threshold))
                nwin = len(g)
            prefix = {
                "MAY_VALID": "may",
                "JUN1_28_LOCKED": "jun",
                "JUN29_JUL3_STRESS": "stress",
                "AUG10_LIVE": "aug",
            }[split_name]
            row[f"{prefix}_windows"] = nwin
            row[f"{prefix}_flag_rate_pct"] = stats["flag_rate_pct"]
            row[f"{prefix}_flagged_edge_c"] = stats["flagged_edge_c"]
            row[f"{prefix}_unflagged_edge_c"] = stats["unflagged_edge_c"]
            row[f"{prefix}_effect_c"] = stats["flag_minus_unflagged_c"]

        # A candidate only survives discovery if May independently points the same way.
        row["may_confirms"] = np.isfinite(row["may_effect_c"]) and row["apr_effect_c"] < 0 and row["may_effect_c"] < 0
        # June is a locked historical test, not used to choose threshold/direction.
        row["jun_confirms"] = np.isfinite(row["jun_effect_c"]) and row["jun_effect_c"] < 0
        row["stress_confirms"] = np.isfinite(row["stress_effect_c"]) and row["stress_effect_c"] < 0
        row["aug_confirms"] = np.isfinite(row["aug_effect_c"]) and row["aug_effect_c"] < 0
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, parts

    # Rank uses April magnitude only among May-confirmed rules. June/stress/Aug do not affect rank.
    out["discovery_rank_score"] = np.where(
        out["may_confirms"],
        -out["apr_effect_c"],
        -np.inf,
    )
    out = out.sort_values(["may_confirms", "discovery_rank_score"], ascending=[False, False]).reset_index(drop=True)
    return out, parts


def _parse_fill_events(session_dir: Path) -> pd.DataFrame:
    event_path = session_dir / PRIMARY_DIR / "shadow_events.jsonl"
    if not event_path.exists():
        raise FileNotFoundError(f"Missing shadow event log: {event_path}")

    rows = []
    with event_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("event") != "ENTRY_FILL":
                continue
            ticker = obj.get("ticker")
            qty = pd.to_numeric(obj.get("fill_qty"), errors="coerce")
            ts = pd.to_datetime(obj.get("fill_time") or obj.get("time"), utc=True, errors="coerce")
            price = pd.to_numeric(obj.get("entry_price"), errors="coerce")
            if ticker and pd.notna(qty) and float(qty) > 0 and pd.notna(ts):
                rows.append({
                    "ticker": str(ticker),
                    "fill_time": ts,
                    "fill_qty": float(qty),
                    "entry_price_event": float(price) if pd.notna(price) else np.nan,
                    "reason": obj.get("reason"),
                })
    return pd.DataFrame(rows).sort_values(["fill_time", "ticker"]).reset_index(drop=True) if rows else pd.DataFrame(columns=["ticker", "fill_time", "fill_qty", "entry_price_event", "reason"])


def _scenario_replay(live_signals: pd.DataFrame, fill_events: pd.DataFrame, scenario: str):
    sig = live_signals.copy()
    sig["decision_time"] = pd.to_datetime(sig["decision_time"], utc=True, errors="coerce").dt.floor("min")
    sig = sig.drop_duplicates(["ticker", "decision_time"], keep="last")
    meta = sig.set_index("ticker")[["decision_time", "signal_edge", "entry_price", "result_final", "series"]].to_dict("index")

    if scenario == "BASELINE_Q3":
        high_breadth_qty = 3.0
        asset_cap = None
    elif scenario == "HIGH_BREADTH_Q2":
        high_breadth_qty = 2.0
        asset_cap = None
    elif scenario == "HIGH_BREADTH_Q1":
        high_breadth_qty = 1.0
        asset_cap = None
    elif scenario == "MAX_2_FILLED_ASSETS":
        high_breadth_qty = 3.0
        asset_cap = 2
    elif scenario == "MAX_3_FILLED_ASSETS":
        high_breadth_qty = 3.0
        asset_cap = 3
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    breadth = sig.groupby("decision_time")["ticker"].size().to_dict()
    target_qty = {}
    for _, row in sig.iterrows():
        n = int(breadth.get(row["decision_time"], 1))
        target_qty[row["ticker"]] = high_breadth_qty if n >= 3 else 3.0

    accepted_qty = {t: 0.0 for t in meta}
    allowed_assets = {}
    blocked_assets = set()
    accepted_events = []

    for _, ev in fill_events.iterrows():
        ticker = ev["ticker"]
        if ticker not in meta:
            continue
        window = meta[ticker]["decision_time"]
        n = int(breadth.get(window, 1))

        if asset_cap is not None and n >= 3:
            allowed = allowed_assets.setdefault(window, [])
            if ticker in blocked_assets:
                continue
            if ticker not in allowed:
                if len(allowed) >= asset_cap:
                    blocked_assets.add(ticker)
                    continue
                allowed.append(ticker)

        remaining = target_qty[ticker] - accepted_qty[ticker]
        if remaining <= 1e-12:
            continue
        q = min(float(ev["fill_qty"]), remaining)
        if q <= 0:
            continue
        accepted_qty[ticker] += q
        accepted_events.append({
            "scenario": scenario,
            "ticker": ticker,
            "decision_time": window,
            "fill_time": ev["fill_time"],
            "accepted_fill_qty": q,
            "source_fill_qty": float(ev["fill_qty"]),
            "reason": ev.get("reason"),
        })

    rows = []
    for ticker, m in meta.items():
        q = accepted_qty.get(ticker, 0.0)
        pnl = q * float(m["signal_edge"])
        rows.append({
            "scenario": scenario,
            "ticker": ticker,
            "decision_time": m["decision_time"],
            "series": m["series"],
            "accepted_qty": q,
            "pnl": pnl,
            "signal_edge": m["signal_edge"],
            "high_breadth": int(breadth.get(m["decision_time"], 1)) >= 3,
            "window_signals": int(breadth.get(m["decision_time"], 1)),
        })
    detail = pd.DataFrame(rows)
    events = pd.DataFrame(accepted_events)

    per_window = detail.groupby("decision_time").agg(
        signals=("ticker", "size"),
        filled_assets=("accepted_qty", lambda x: int((x > 1e-12).sum())),
        filled_contracts=("accepted_qty", "sum"),
        pnl=("pnl", "sum"),
    ).reset_index()

    summary = {
        "scenario": scenario,
        "pnl": detail["pnl"].sum(),
        "filled_assets": int((detail["accepted_qty"] > 1e-12).sum()),
        "filled_contracts": detail["accepted_qty"].sum(),
        "high_breadth_pnl": detail.loc[detail["high_breadth"], "pnl"].sum(),
        "low_breadth_pnl": detail.loc[~detail["high_breadth"], "pnl"].sum(),
        "worst_window_pnl": per_window["pnl"].min() if len(per_window) else np.nan,
        "best_window_pnl": per_window["pnl"].max() if len(per_window) else np.nan,
    }
    return summary, detail, events, per_window


def replay_live_exposure_caps(session_dir, show=True):
    session_dir = Path(session_dir)
    live, coverage = load_live_primary_signals(session_dir, settle_missing=True, show=False)
    fills = _parse_fill_events(session_dir)

    scenarios = [
        "BASELINE_Q3",
        "HIGH_BREADTH_Q2",
        "HIGH_BREADTH_Q1",
        "MAX_2_FILLED_ASSETS",
        "MAX_3_FILLED_ASSETS",
    ]

    summaries = []
    details = {}
    events = {}
    windows = {}
    for scenario in scenarios:
        s, d, e, w = _scenario_replay(live, fills, scenario)
        summaries.append(s)
        details[scenario] = d
        events[scenario] = e
        windows[scenario] = w

    summary = pd.DataFrame(summaries)
    baseline = float(summary.loc[summary["scenario"] == "BASELINE_Q3", "pnl"].iloc[0])
    summary["pnl_change_vs_baseline"] = summary["pnl"] - baseline

    if show:
        print("Live settlement coverage:")
        _display(coverage.round(3))
        print("\nCAUSAL EXPOSURE-CAP REPLAY")
        _display(summary.round(4))
        print("\nReplay semantics: actual ENTRY_FILL timestamps/quantities are processed chronologically.\n"
              "Smaller-qty scenarios cap accepted quantity after the same queue-triggering fills.\n"
              "Asset-cap scenarios allow the first N distinct filled assets in a >=3-signal window and reject later assets.")

    return {
        "summary": summary,
        "details": details,
        "events": events,
        "windows": windows,
        "live_signals": live,
        "fill_events": fills,
        "coverage": coverage,
    }


def run_high_breadth_failure_study(
    regime_study,
    toxic_study,
    live_session,
    features=None,
    show=True,
):
    """Two-part study: detect high-breadth signal failure, then replay portfolio caps."""
    scored = toxic_study.get("scored_windows")
    if scored is None or len(scored) == 0:
        raise ValueError("toxic_study must contain non-empty scored_windows from run_toxic_window_detector().")

    rules, parts = discover_high_breadth_rules(scored, features=features)
    replay = replay_live_exposure_caps(live_session, show=False)

    split_rows = []
    for name, g in parts.items():
        split_rows.append({
            "split": name,
            "windows": len(g),
            "negative_window_pct": 100.0 * (g["signal_edge"] < 0).mean() if len(g) else np.nan,
            "mean_edge_c": 100.0 * g["signal_edge"].mean() if len(g) else np.nan,
            "median_edge_c": 100.0 * g["signal_edge"].median() if len(g) else np.nan,
        })
    split_summary = pd.DataFrame(split_rows)

    if show:
        print("=" * 118)
        print("HIGH-BREADTH FAILURE STUDY")
        print("=" * 118)
        print("Only windows with >=3 simultaneous frozen signals are used in the detector section.")
        print("Threshold/direction: April only. May: validation. Jun1-28: locked test. Jun29-Jul3: stress. Aug10: live validation.")

        print("\nHIGH-BREADTH SPLIT SUMMARY")
        _display(split_summary.round(3))

        print("\nWALK-FORWARD FEATURE RULES")
        if rules.empty:
            print("No usable April tail rules found.")
        else:
            cols = [
                "feature", "direction", "threshold",
                "apr_flag_rate_pct", "apr_effect_c",
                "may_flag_rate_pct", "may_effect_c", "may_confirms",
                "jun_flag_rate_pct", "jun_effect_c", "jun_confirms",
                "stress_flag_rate_pct", "stress_effect_c", "stress_confirms",
                "aug_flag_rate_pct", "aug_effect_c", "aug_confirms",
            ]
            _display(rules[cols].head(25).round(3))

        print("\nCANDIDATES THAT SURVIVE APRIL -> MAY -> LOCKED JUNE")
        if rules.empty:
            _display(pd.DataFrame())
        else:
            survived = rules[rules["may_confirms"] & rules["jun_confirms"]].copy()
            _display(survived.head(15).round(3))

        print("\nCAUSAL LIVE EXPOSURE-CAP REPLAY")
        _display(replay["summary"].round(4))

        print("\nInterpretation discipline:")
        print("  1) Do not retune a rule because Aug10 disagrees.")
        print("  2) A detector is interesting only if it survives May and locked Jun before the stress/Aug checks.")
        print("  3) Exposure caps are NEW development hypotheses; this replay does not alter the frozen live strategy.")

    return {
        "split_summary": split_summary,
        "walk_forward_rules": rules,
        "parts": parts,
        "exposure_replay": replay,
        "regime_study": regime_study,
        "toxic_study": toxic_study,
    }
