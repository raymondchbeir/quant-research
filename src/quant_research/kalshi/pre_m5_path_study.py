from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .window_regime import PRIMARY_DIR, load_live_primary_signals

STUDY_VERSION = "PRE_M5_PATH_STUDY_V1"

_TICKER_RE = re.compile(r'"(?:ticker|market_ticker)"\s*:\s*"([^"]+)"')


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _num(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _price(x):
    y = _num(x)
    if not np.isfinite(y):
        return np.nan
    if y > 1.000001:
        y /= 100.0
    return y if 0.0 <= y <= 1.0 else np.nan


def _utc_scalar(x):
    if x is None:
        return pd.NaT
    if isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x):
        v = float(x)
        try:
            if v > 1e17:
                return pd.to_datetime(v, unit="ns", utc=True, errors="coerce")
            if v > 1e14:
                return pd.to_datetime(v, unit="us", utc=True, errors="coerce")
            if v > 1e11:
                return pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
            if v > 1e9:
                return pd.to_datetime(v, unit="s", utc=True, errors="coerce")
        except Exception:
            return pd.NaT
    return pd.to_datetime(x, utc=True, errors="coerce")


def _candidate_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for key in ("msg", "data", "message", "payload"):
            z = obj.get(key)
            if isinstance(z, dict):
                yield z


def _first(d, keys):
    for key in keys:
        if key in d and d.get(key) is not None:
            return d.get(key)
    return None


def _extract_quote(obj, source_kind: str):
    outer = obj if isinstance(obj, dict) else {}
    outer_time = _first(outer, ("time", "timestamp", "ts", "recv_time", "received_at", "created_time"))

    for d in _candidate_dicts(obj):
        ticker = _first(d, ("ticker", "market_ticker")) or _first(outer, ("ticker", "market_ticker"))
        if not ticker:
            continue

        ts = _utc_scalar(
            _first(d, ("time", "timestamp", "ts", "recv_time", "received_at", "created_time"))
            or outer_time
        )
        if pd.isna(ts):
            continue

        if source_kind == "full_books" or ("yes_bids" in d and "yes_asks" in d):
            bids = d.get("yes_bids") or []
            asks = d.get("yes_asks") or []
            try:
                bid = _price(bids[0][0]) if bids else np.nan
                ask = _price(asks[0][0]) if asks else np.nan
            except Exception:
                bid = ask = np.nan
        else:
            bid = _price(_first(d, (
                "yes_bid_dollars", "yes_bid_dollars_fp", "yes_bid", "best_yes_bid",
                "best_bid_dollars", "best_bid",
            )))
            ask = _price(_first(d, (
                "yes_ask_dollars", "yes_ask_dollars_fp", "yes_ask", "best_yes_ask",
                "best_ask_dollars", "best_ask",
            )))

        if not np.isfinite(bid) or not np.isfinite(ask) or ask < bid:
            continue
        return {
            "time": ts,
            "ticker": str(ticker),
            "yes_bid": float(bid),
            "yes_ask": float(ask),
            "mid": float((bid + ask) / 2.0),
            "spread_c": float(100.0 * (ask - bid)),
            "quote_source": source_kind,
        }
    return None


def _read_shadow_signals(session_dir: Path, settle_missing=True):
    shadow = session_dir / PRIMARY_DIR / "shadow_records.csv"
    if not shadow.exists():
        raise FileNotFoundError(f"Missing primary shadow records: {shadow}")

    raw = pd.read_csv(shadow, low_memory=False)
    z = pd.DataFrame(index=raw.index)
    z["session"] = session_dir.name
    z["ticker"] = raw["ticker"].astype(str)
    z["series"] = raw["series"].astype(str) if "series" in raw.columns else z["ticker"].str.split("-").str[0]
    z["decision_time"] = pd.to_datetime(raw["decision_time"], utc=True, errors="coerce").dt.floor("min")
    z["direction"] = raw["direction"].astype(str).str.upper().str.strip()
    z["entry_price"] = pd.to_numeric(raw.get("entry_price"), errors="coerce")
    z["entry_fill_qty"] = pd.to_numeric(raw.get("entry_fill_qty"), errors="coerce").fillna(0.0)
    z["realized_pnl_raw"] = pd.to_numeric(raw.get("realized_pnl"), errors="coerce").fillna(0.0)
    z["status"] = raw["status"].astype(str).str.upper().str.strip() if "status" in raw.columns else ""
    z["result_raw"] = raw["result"].astype(str).str.upper().str.strip() if "result" in raw.columns else ""
    z["midpoint_m5_shadow"] = pd.to_numeric(raw.get("midpoint"), errors="coerce")
    z["spread_m5_shadow_c"] = pd.to_numeric(raw.get("spread_c"), errors="coerce")

    z = z[
        z["decision_time"].notna()
        & z["direction"].isin(["YES", "NO"])
        & z["entry_price"].between(0.01, 0.99, inclusive="both")
        & ~z["status"].eq("DATA_INVALID")
    ].copy()
    z = z.drop_duplicates(["ticker", "decision_time"], keep="last")

    # Fill settlement outcomes for no-fill contracts from the existing analysis cache / REST
    # when available. This is for labels only; no post-M5 data enters path features.
    try:
        settled, coverage = load_live_primary_signals(
            session_dir,
            settle_missing=settle_missing,
            show=False,
        )
        m = settled[[
            "ticker", "decision_time", "result_final", "signal_edge", "actual_pnl", "correct"
        ]].copy()
        z = z.merge(m, on=["ticker", "decision_time"], how="left")
    except Exception:
        coverage = pd.DataFrame()
        z["result_final"] = np.nan
        z["signal_edge"] = np.nan
        z["actual_pnl"] = np.nan
        z["correct"] = np.nan

    z["actual_pnl_effective"] = pd.to_numeric(z.get("actual_pnl"), errors="coerce")
    z["actual_pnl_effective"] = z["actual_pnl_effective"].fillna(z["realized_pnl_raw"])
    z["filled"] = z["entry_fill_qty"] > 1e-12
    z["open_position"] = z["filled"] & ~z["result_final"].isin(["YES", "NO"])
    return z.reset_index(drop=True), coverage


def _target_ranges(signals: pd.DataFrame, seed_lookback_sec=120):
    out = {}
    for _, r in signals.iterrows():
        t = r["decision_time"]
        out[str(r["ticker"])] = (
            t - pd.Timedelta(minutes=4, seconds=seed_lookback_sec),
            t,
        )
    return out


def _stream_quotes(path: Path, source_kind: str, ranges: dict):
    rows = []
    if not path.exists():
        return pd.DataFrame()

    targets = set(ranges)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _TICKER_RE.search(line)
            if m is None:
                continue
            ticker = m.group(1)
            if ticker not in targets:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            q = _extract_quote(obj, source_kind)
            if q is None or q["ticker"] not in ranges:
                continue
            start, end = ranges[q["ticker"]]
            if start <= q["time"] <= end:
                rows.append(q)

    if not rows:
        return pd.DataFrame(columns=["time", "ticker", "yes_bid", "yes_ask", "mid", "spread_c", "quote_source"])
    z = pd.DataFrame(rows)
    z = z.drop_duplicates(["ticker", "time"], keep="last").sort_values(["ticker", "time"]).reset_index(drop=True)
    return z


def load_pre_m5_quotes(
    session_dir,
    signals: pd.DataFrame,
    source="auto",
    full_book_fallback=True,
    seed_lookback_sec=120,
    show=True,
):
    session_dir = Path(session_dir)
    ranges = _target_ranges(signals, seed_lookback_sec=seed_lookback_sec)

    if source not in {"auto", "ticker_updates", "full_books"}:
        raise ValueError("source must be auto, ticker_updates, or full_books")

    if source in {"auto", "ticker_updates"}:
        ticker_path = session_dir / "ticker_updates.jsonl"
        q = _stream_quotes(ticker_path, "ticker_updates", ranges)
        coverage = q["ticker"].nunique() / max(1, len(ranges)) if len(q) else 0.0
        enough = len(q) >= max(5, 2 * len(ranges)) and coverage >= 0.60
        if enough or source == "ticker_updates":
            if show:
                print(f"{session_dir.name}: using ticker_updates.jsonl | quotes={len(q):,} | ticker coverage={100*coverage:.1f}%")
            return q
        if show:
            print(
                f"{session_dir.name}: ticker_updates insufficient for BBO path "
                f"(quotes={len(q):,}, ticker coverage={100*coverage:.1f}%)."
            )

    if not full_book_fallback:
        raise RuntimeError(
            f"Could not reconstruct enough BBO quotes from ticker_updates for {session_dir}. "
            "Set full_book_fallback=True to scan full_books.jsonl."
        )

    full_path = session_dir / "full_books.jsonl"
    if show:
        try:
            gb = full_path.stat().st_size / (1024 ** 3)
            print(f"{session_dir.name}: falling back to full_books.jsonl ({gb:.2f} GB); one-time scan may take a while.")
        except Exception:
            print(f"{session_dir.name}: falling back to full_books.jsonl; one-time scan may take a while.")
    q = _stream_quotes(full_path, "full_books", ranges)
    if show:
        coverage = q["ticker"].nunique() / max(1, len(ranges)) if len(q) else 0.0
        print(f"{session_dir.name}: full-book quotes={len(q):,} | ticker coverage={100*coverage:.1f}%")
    return q


def _anchor_value(g: pd.DataFrame, anchor, col: str, max_age_sec: float):
    if g.empty:
        return np.nan, np.nan
    times = g["time"].to_numpy(dtype="datetime64[ns]")
    a = np.datetime64(pd.Timestamp(anchor).tz_convert("UTC").tz_localize(None))
    i = np.searchsorted(times, a, side="right") - 1
    if i < 0:
        return np.nan, np.nan
    ts = g.iloc[i]["time"]
    age = (pd.Timestamp(anchor) - ts).total_seconds()
    if age < -1e-9 or age > max_age_sec:
        return np.nan, age
    return float(g.iloc[i][col]), float(age)


def _count_state_flips(values, center=0.0):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0
    s = np.sign(x - center)
    s = s[s != 0]
    if len(s) < 2:
        return 0
    s = s[np.r_[True, s[1:] != s[:-1]]]
    return max(0, len(s) - 1)


def _contract_path_features(signal_row, quotes: pd.DataFrame, max_anchor_age_sec=90.0):
    decision = signal_row["decision_time"]
    m1_time = decision - pd.Timedelta(minutes=4)
    anchors = [decision - pd.Timedelta(minutes=k) for k in (4, 3, 2, 1, 0)]

    g = quotes[
        (quotes["ticker"] == signal_row["ticker"])
        & (quotes["time"] <= decision)
        & (quotes["time"] >= m1_time - pd.Timedelta(seconds=max_anchor_age_sec))
    ].sort_values("time")

    mids, spreads, ages = [], [], []
    for a in anchors:
        v, age = _anchor_value(g, a, "mid", max_anchor_age_sec)
        s, _ = _anchor_value(g, a, "spread_c", max_anchor_age_sec)
        mids.append(v)
        spreads.append(s)
        ages.append(age)

    start_mid = mids[0]
    path = g[(g["time"] > m1_time) & (g["time"] <= decision)].copy()
    if np.isfinite(start_mid):
        seed = pd.DataFrame([{
            "time": m1_time,
            "ticker": signal_row["ticker"],
            "mid": start_mid,
            "spread_c": spreads[0],
        }])
        path = pd.concat([seed, path[["time", "ticker", "mid", "spread_c"]]], ignore_index=True)
    path = path.drop_duplicates("time", keep="last").sort_values("time")

    px = pd.to_numeric(path.get("mid"), errors="coerce").dropna().to_numpy(float)
    spr = pd.to_numeric(path.get("spread_c"), errors="coerce").dropna().to_numpy(float)
    d_c = np.diff(px) * 100.0 if len(px) >= 2 else np.array([], dtype=float)

    net_c = 100.0 * (mids[4] - mids[0]) if np.isfinite(mids[0]) and np.isfinite(mids[4]) else np.nan
    m4_m5_c = 100.0 * (mids[4] - mids[3]) if np.isfinite(mids[3]) and np.isfinite(mids[4]) else np.nan
    path_length = float(np.abs(d_c).sum()) if len(d_c) else 0.0
    rv = float(np.sqrt(np.square(d_c).sum())) if len(d_c) else 0.0
    rng = float(100.0 * (np.nanmax(px) - np.nanmin(px))) if len(px) else np.nan
    efficiency = (abs(net_c) / path_length) if np.isfinite(net_c) and path_length > 1e-12 else (1.0 if path_length <= 1e-12 and len(px) else np.nan)

    direction_sign = 1.0 if signal_row["direction"] == "YES" else -1.0
    aligned = direction_sign * 100.0 * (px - px[0]) if len(px) else np.array([], dtype=float)
    favorable = max(0.0, float(np.nanmax(aligned))) if len(aligned) else np.nan
    adverse = max(0.0, float(-np.nanmin(aligned))) if len(aligned) else np.nan

    return {
        "session": signal_row["session"],
        "ticker": signal_row["ticker"],
        "series": signal_row["series"],
        "decision_time": decision,
        "direction": signal_row["direction"],
        "entry_price": signal_row["entry_price"],
        "entry_fill_qty": signal_row["entry_fill_qty"],
        "filled": signal_row["filled"],
        "open_position": signal_row["open_position"],
        "status": signal_row["status"],
        "result_final": signal_row.get("result_final"),
        "signal_edge": signal_row.get("signal_edge"),
        "actual_pnl": signal_row["actual_pnl_effective"],
        "m1_mid_c": 100.0 * mids[0] if np.isfinite(mids[0]) else np.nan,
        "m2_mid_c": 100.0 * mids[1] if np.isfinite(mids[1]) else np.nan,
        "m3_mid_c": 100.0 * mids[2] if np.isfinite(mids[2]) else np.nan,
        "m4_mid_c": 100.0 * mids[3] if np.isfinite(mids[3]) else np.nan,
        "m5_mid_c": 100.0 * mids[4] if np.isfinite(mids[4]) else np.nan,
        "m1_spread_c": spreads[0],
        "m2_spread_c": spreads[1],
        "m3_spread_c": spreads[2],
        "m4_spread_c": spreads[3],
        "m5_spread_c": spreads[4],
        "max_anchor_age_s": float(np.nanmax(ages)) if np.isfinite(ages).any() else np.nan,
        "anchors_present": int(np.isfinite(mids).sum()),
        "path_complete": bool(np.isfinite(mids).all()),
        "quote_observations_m1_m5": int(len(px)),
        "m1_to_m5_c": net_c,
        "abs_m1_to_m5_c": abs(net_c) if np.isfinite(net_c) else np.nan,
        "m4_to_m5_c": m4_m5_c,
        "abs_m4_to_m5_c": abs(m4_m5_c) if np.isfinite(m4_m5_c) else np.nan,
        "toward_signal_m1_m5_c": direction_sign * net_c if np.isfinite(net_c) else np.nan,
        "toward_signal_m4_m5_c": direction_sign * m4_m5_c if np.isfinite(m4_m5_c) else np.nan,
        "mid_rv_m1_m5_c": rv,
        "mid_range_m1_m5_c": rng,
        "mid_path_length_c": path_length,
        "mid_path_efficiency": efficiency,
        "cross_50_count": _count_state_flips(px, center=0.50),
        "move_direction_flips": _count_state_flips(d_c, center=0.0),
        "spread_mean_m1_m5_c": float(np.nanmean(spr)) if len(spr) else np.nan,
        "spread_max_m1_m5_c": float(np.nanmax(spr)) if len(spr) else np.nan,
        "spread_std_m1_m5_c": float(np.nanstd(spr)) if len(spr) else np.nan,
        "max_favorable_excursion_c": favorable,
        "max_adverse_excursion_c": adverse,
    }


def build_contract_paths(signals: pd.DataFrame, quotes: pd.DataFrame, max_anchor_age_sec=90.0):
    rows = [
        _contract_path_features(r, quotes, max_anchor_age_sec=max_anchor_age_sec)
        for _, r in signals.iterrows()
    ]
    return pd.DataFrame(rows)


def _dominant_share(x):
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    x = x[np.abs(x) > 1e-12]
    if len(x) == 0:
        return np.nan
    pos = (x > 0).mean()
    return float(max(pos, 1.0 - pos))


def _mean_pair_corr_minute_moves(g: pd.DataFrame):
    cols = ["m1_mid_c", "m2_mid_c", "m3_mid_c", "m4_mid_c", "m5_mid_c"]
    a = g[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
    if len(a) < 2:
        return np.nan
    moves = np.diff(a, axis=1)
    valid = np.nanstd(moves, axis=1) > 1e-12
    moves = moves[valid]
    if len(moves) < 2:
        return np.nan
    c = np.corrcoef(moves)
    tri = c[np.triu_indices(len(c), 1)]
    tri = tri[np.isfinite(tri)]
    return float(tri.mean()) if len(tri) else np.nan


def build_window_paths(contract_paths: pd.DataFrame):
    rows = []
    for (session, decision), g in contract_paths.groupby(["session", "decision_time"], sort=True):
        breadth = len(g)
        filled = g[g["entry_fill_qty"] > 1e-12]
        settled_signal = g["signal_edge"].notna()
        actual_pnl = pd.to_numeric(g["actual_pnl"], errors="coerce").fillna(0.0).sum()
        open_positions = int(g["open_position"].fillna(False).sum())

        def mean(col):
            return pd.to_numeric(g[col], errors="coerce").mean()

        def maxv(col):
            return pd.to_numeric(g[col], errors="coerce").max()

        rows.append({
            "session": session,
            "decision_time": decision,
            "utc_day": decision.floor("D"),
            "signals": breadth,
            "filled_assets": int(len(filled)),
            "filled_contracts": pd.to_numeric(filled["entry_fill_qty"], errors="coerce").sum(),
            "actual_pnl": float(actual_pnl),
            "open_positions": open_positions,
            "execution_complete": open_positions == 0,
            "settled_signal_coverage_pct": 100.0 * settled_signal.mean() if breadth else np.nan,
            "all_signal_edge_c": 100.0 * pd.to_numeric(g.loc[settled_signal, "signal_edge"], errors="coerce").mean() if settled_signal.any() else np.nan,
            "path_complete_share_pct": 100.0 * g["path_complete"].mean() if breadth else np.nan,
            "mean_mid_rv_c": mean("mid_rv_m1_m5_c"),
            "max_mid_rv_c": maxv("mid_rv_m1_m5_c"),
            "mean_mid_range_c": mean("mid_range_m1_m5_c"),
            "max_mid_range_c": maxv("mid_range_m1_m5_c"),
            "mean_mid_path_length_c": mean("mid_path_length_c"),
            "max_mid_path_length_c": maxv("mid_path_length_c"),
            "mean_path_efficiency": mean("mid_path_efficiency"),
            "mean_cross_50_count": mean("cross_50_count"),
            "share_crossed_50_pct": 100.0 * (pd.to_numeric(g["cross_50_count"], errors="coerce") > 0).mean(),
            "mean_move_direction_flips": mean("move_direction_flips"),
            "mean_abs_m1_m5_c": mean("abs_m1_to_m5_c"),
            "max_abs_m1_m5_c": maxv("abs_m1_to_m5_c"),
            "mean_toward_signal_m1_m5_c": mean("toward_signal_m1_m5_c"),
            "share_m1_m5_toward_signal_pct": 100.0 * (pd.to_numeric(g["toward_signal_m1_m5_c"], errors="coerce") > 0).mean(),
            "mean_abs_m4_m5_c": mean("abs_m4_to_m5_c"),
            "max_abs_m4_m5_c": maxv("abs_m4_to_m5_c"),
            "mean_toward_signal_m4_m5_c": mean("toward_signal_m4_m5_c"),
            "share_m4_m5_toward_signal_pct": 100.0 * (pd.to_numeric(g["toward_signal_m4_m5_c"], errors="coerce") > 0).mean(),
            "m1_m5_dominant_move_share": _dominant_share(g["m1_to_m5_c"]),
            "m4_m5_dominant_move_share": _dominant_share(g["m4_to_m5_c"]),
            "cross_asset_move_std_c": pd.to_numeric(g["m1_to_m5_c"], errors="coerce").std(ddof=0),
            "mean_pair_corr_minute_moves": _mean_pair_corr_minute_moves(g),
            "mean_spread_m1_m5_c": mean("spread_mean_m1_m5_c"),
            "max_spread_m1_m5_c": maxv("spread_max_m1_m5_c"),
            "mean_spread_vol_c": mean("spread_std_m1_m5_c"),
            "mean_max_adverse_excursion_c": mean("max_adverse_excursion_c"),
            "max_adverse_excursion_c": maxv("max_adverse_excursion_c"),
            "mean_max_favorable_excursion_c": mean("max_favorable_excursion_c"),
            "mean_quote_observations": mean("quote_observations_m1_m5"),
        })
    return pd.DataFrame(rows).sort_values(["decision_time", "session"]).reset_index(drop=True)


DEFAULT_COMPARE_FEATURES = [
    "signals",
    "mean_mid_rv_c", "max_mid_rv_c",
    "mean_mid_range_c", "max_mid_range_c",
    "mean_mid_path_length_c", "max_mid_path_length_c",
    "mean_path_efficiency",
    "mean_cross_50_count", "share_crossed_50_pct", "mean_move_direction_flips",
    "mean_abs_m1_m5_c", "max_abs_m1_m5_c", "mean_toward_signal_m1_m5_c",
    "share_m1_m5_toward_signal_pct",
    "mean_abs_m4_m5_c", "max_abs_m4_m5_c", "mean_toward_signal_m4_m5_c",
    "share_m4_m5_toward_signal_pct",
    "m1_m5_dominant_move_share", "m4_m5_dominant_move_share",
    "cross_asset_move_std_c", "mean_pair_corr_minute_moves",
    "mean_spread_m1_m5_c", "max_spread_m1_m5_c", "mean_spread_vol_c",
    "mean_max_adverse_excursion_c", "max_adverse_excursion_c",
    "mean_max_favorable_excursion_c",
]


def _bootstrap_diff(a, b, n=5000, seed=17):
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sims = np.empty(n, dtype=float)
    for i in range(n):
        sims[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    return tuple(np.quantile(sims, [0.025, 0.975]))


def compare_window_groups(windows: pd.DataFrame, bad_mask: pd.Series, features=None, bootstrap_n=5000, seed=17):
    features = features or DEFAULT_COMPARE_FEATURES
    bad_mask = bad_mask.reindex(windows.index).fillna(False).astype(bool)
    bad = windows[bad_mask]
    normal = windows[~bad_mask]
    rows = []
    for j, feature in enumerate(features):
        if feature not in windows.columns:
            continue
        a = pd.to_numeric(bad[feature], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(normal[feature], errors="coerce").dropna().to_numpy(float)
        if len(a) == 0 or len(b) == 0:
            continue
        diff = a.mean() - b.mean()
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0) if len(a) > 1 and len(b) > 1 else np.nan
        smd = diff / pooled if np.isfinite(pooled) and pooled > 1e-12 else np.nan
        lo, hi = _bootstrap_diff(a, b, n=bootstrap_n, seed=seed + j)
        x = pd.to_numeric(windows[feature], errors="coerce")
        y = pd.to_numeric(windows["actual_pnl"], errors="coerce")
        valid = x.notna() & y.notna()
        spearman = x[valid].rank().corr(y[valid].rank()) if valid.sum() >= 3 else np.nan
        rows.append({
            "feature": feature,
            "bad_n": len(a),
            "normal_n": len(b),
            "bad_mean": a.mean(),
            "normal_mean": b.mean(),
            "bad_minus_normal": diff,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "bad_median": np.median(a),
            "normal_median": np.median(b),
            "standardized_diff": smd,
            "spearman_vs_actual_pnl": spearman,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["abs_standardized_diff"] = out["standardized_diff"].abs()
        out = out.sort_values(["abs_standardized_diff", "feature"], ascending=[False, True]).reset_index(drop=True)
    return out


def _max_drawdown(pnls):
    x = pd.to_numeric(pd.Series(pnls), errors="coerce").fillna(0.0).to_numpy(float)
    if len(x) == 0:
        return np.nan
    equity = np.cumsum(x)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    dd = np.r_[0.0, equity] - peak
    return float(dd.min())


def daily_forensics(windows: pd.DataFrame):
    rows = []
    complete = windows[windows["execution_complete"]].copy()
    for day, g in complete.groupby("utc_day", sort=True):
        g = g.sort_values("decision_time")
        filled = g[g["filled_assets"] > 0]
        hb = g[g["signals"] >= 3]
        rows.append({
            "utc_day": day,
            "sessions": ",".join(sorted(g["session"].unique())),
            "windows": len(g),
            "filled_windows": len(filled),
            "signals": int(g["signals"].sum()),
            "high_breadth_windows": len(hb),
            "actual_pnl": g["actual_pnl"].sum(),
            "max_drawdown": _max_drawdown(g["actual_pnl"]),
            "worst_window_pnl": g["actual_pnl"].min(),
            "high_breadth_pnl": hb["actual_pnl"].sum() if len(hb) else 0.0,
            "mean_mid_rv_c": g["mean_mid_rv_c"].mean(),
            "mean_mid_range_c": g["mean_mid_range_c"].mean(),
            "mean_abs_m4_m5_c": g["mean_abs_m4_m5_c"].mean(),
            "mean_cross_50_count": g["mean_cross_50_count"].mean(),
            "mean_pair_corr_minute_moves": g["mean_pair_corr_minute_moves"].mean(),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["max_drawdown", "actual_pnl"], ascending=[True, True]).reset_index(drop=True) if len(out) else out


def run_pre_m5_path_study(
    session_dirs,
    source="auto",
    full_book_fallback=True,
    max_anchor_age_sec=90.0,
    min_window_path_coverage_pct=75.0,
    bad_window_quantile=0.25,
    bootstrap_n=5000,
    worst_n=12,
    settle_missing=True,
    show=True,
):
    """Forensic M1->M5 path study using only quotes known at/before the frozen M5 decision.

    Labels (settlement / realized PnL) are used only after features are built. An active window
    with an open filled position is excluded from bad-vs-normal execution comparisons until settled.
    """
    if isinstance(session_dirs, (str, Path)):
        session_dirs = [session_dirs]
    session_dirs = [Path(x) for x in session_dirs]

    all_signals = []
    all_quotes = []
    all_contracts = []
    coverage_rows = []

    if show:
        print("=" * 122)
        print("CAUSAL M1 -> M5 KALSHI PATH FORENSIC STUDY")
        print("=" * 122)
        print("Features use only BBO observations at/before M5. Settlement/PnL is used only as the outcome label.")

    for session_dir in session_dirs:
        signals, settlement_coverage = _read_shadow_signals(session_dir, settle_missing=settle_missing)
        if signals.empty:
            if show:
                print(f"{session_dir.name}: no valid frozen primary signals; skipped")
            continue
        quotes = load_pre_m5_quotes(
            session_dir,
            signals,
            source=source,
            full_book_fallback=full_book_fallback,
            show=show,
        )
        contracts = build_contract_paths(signals, quotes, max_anchor_age_sec=max_anchor_age_sec)

        coverage_rows.append({
            "session": session_dir.name,
            "valid_signals": len(signals),
            "quote_rows": len(quotes),
            "tickers_with_quotes": quotes["ticker"].nunique() if len(quotes) else 0,
            "complete_m1_m5_paths": int(contracts["path_complete"].sum()) if len(contracts) else 0,
            "complete_path_pct": 100.0 * contracts["path_complete"].mean() if len(contracts) else np.nan,
            "filled_signals": int(signals["filled"].sum()),
            "open_positions": int(signals["open_position"].sum()),
        })
        all_signals.append(signals)
        all_quotes.append(quotes.assign(session=session_dir.name))
        all_contracts.append(contracts)

    if not all_contracts:
        raise RuntimeError("No M1-M5 contract paths could be built from the supplied sessions.")

    signals = pd.concat(all_signals, ignore_index=True)
    quotes = pd.concat(all_quotes, ignore_index=True) if all_quotes else pd.DataFrame()
    contracts = pd.concat(all_contracts, ignore_index=True)
    windows = build_window_paths(contracts)
    coverage = pd.DataFrame(coverage_rows)

    universe = windows[
        windows["execution_complete"]
        & (windows["filled_assets"] > 0)
        & (windows["path_complete_share_pct"] >= float(min_window_path_coverage_pct))
    ].copy()

    if len(universe) >= 4:
        bad_cutoff = float(universe["actual_pnl"].quantile(bad_window_quantile))
        bad_mask = universe["actual_pnl"] <= bad_cutoff
    else:
        bad_cutoff = np.nan
        bad_mask = pd.Series(False, index=universe.index)

    worst_quartile_compare = compare_window_groups(
        universe,
        bad_mask,
        bootstrap_n=bootstrap_n,
        seed=17,
    )
    loss_compare = compare_window_groups(
        universe,
        universe["actual_pnl"] < 0,
        bootstrap_n=bootstrap_n,
        seed=1701,
    ) if len(universe) else pd.DataFrame()

    worst_windows = universe.sort_values(["actual_pnl", "decision_time"]).head(worst_n).copy()
    worst_keys = set(zip(worst_windows["session"], worst_windows["decision_time"]))
    worst_contracts = contracts[
        contracts.apply(lambda r: (r["session"], r["decision_time"]) in worst_keys, axis=1)
    ].sort_values(["actual_pnl", "decision_time", "ticker"])
    days = daily_forensics(windows)

    if show:
        print("\nSESSION / PATH COVERAGE")
        _display(coverage.round(3))

        print("\nWORST SETTLED FILLED WINDOWS")
        cols = [
            "session", "decision_time", "signals", "filled_assets", "actual_pnl", "all_signal_edge_c",
            "mean_mid_rv_c", "max_mid_rv_c", "mean_mid_range_c", "max_mid_range_c",
            "mean_abs_m1_m5_c", "mean_abs_m4_m5_c", "mean_cross_50_count",
            "mean_move_direction_flips", "m1_m5_dominant_move_share",
            "mean_pair_corr_minute_moves", "mean_spread_m1_m5_c", "max_spread_m1_m5_c",
        ]
        _display(worst_windows[cols].round(3))

        print(f"\nWORST-{100*bad_window_quantile:.0f}% WINDOWS VS THE REST")
        if np.isfinite(bad_cutoff):
            print(f"Bad-window cutoff by actual PnL: <= ${bad_cutoff:.4f} | bad={int(bad_mask.sum())} | normal={int((~bad_mask).sum())}")
        _display(worst_quartile_compare.head(30).round(4))

        print("\nALL LOSING FILLED WINDOWS VS NON-LOSING FILLED WINDOWS")
        _display(loss_compare.head(30).round(4))

        print("\nDAILY DRAWDOWN FORENSICS")
        _display(days.round(4))

        print("\nCONTRACT M1->M5 PATHS INSIDE THE WORST WINDOWS")
        ccols = [
            "session", "decision_time", "ticker", "direction", "entry_fill_qty", "actual_pnl",
            "m1_mid_c", "m2_mid_c", "m3_mid_c", "m4_mid_c", "m5_mid_c",
            "mid_rv_m1_m5_c", "mid_range_m1_m5_c", "m1_to_m5_c", "m4_to_m5_c",
            "toward_signal_m1_m5_c", "toward_signal_m4_m5_c", "cross_50_count",
            "move_direction_flips", "max_adverse_excursion_c", "spread_max_m1_m5_c",
        ]
        _display(worst_contracts[ccols].round(3))

        print("\nInterpretation discipline:")
        print("  1) This is an August recorder forensic study; Apr-Jul minute1-4 paths are unavailable in the monthly M5 files.")
        print("  2) Bottom-quartile and loss comparisons are descriptive, not a new trading filter.")
        print("  3) Any candidate M1-M5 feature must be frozen and validated on future sessions before changing the primary strategy.")
        print("  4) Active windows with open positions are excluded from execution bad-vs-normal comparisons until settled.")

    return {
        "coverage": coverage,
        "signals": signals,
        "quotes": quotes,
        "contract_paths": contracts,
        "window_paths": windows,
        "analysis_universe": universe,
        "bad_window_cutoff": bad_cutoff,
        "worst_windows": worst_windows,
        "worst_contracts": worst_contracts,
        "worst_quartile_compare": worst_quartile_compare,
        "loss_compare": loss_compare,
        "daily_forensics": days,
    }
