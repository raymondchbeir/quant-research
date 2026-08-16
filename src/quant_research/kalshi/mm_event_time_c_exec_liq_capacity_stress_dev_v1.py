from __future__ import annotations

"""Executable-liquidation and fill-retention stress for V5 candidate C.

DEVELOPMENT ONLY. NO NEW SIGNAL RULES OR THRESHOLDS.

This module is hard-bound to the already-explored V5 development session
20260816_070627 and to the previously evaluated candidate:

    C_L3_SPREAD2
    same-side L3 depth > opposite-side L3 depth
    AND natural spread >= 2c

It does not replay or alter the quoting signal. It consumes the exact historical
replay outputs already produced by MM_EVENT_TIME_ABCD_CAPACITY_DEV_V1 and asks:

1. What happens if residual M5 inventory is liquidated at the observed M5
   executable side of the book instead of marked at midpoint?
2. Does recorded top-3 depth appear sufficient to support that liquidation?
3. How concentrated are Q10 fills, queue positions, residual inventory and PnL?
4. How does Q10 behave by chronology and series after executable-side marking?
5. Under fixed, seeded random fill-retention stress (75/50/25% of simulated fill
   events retained), how often does the candidate still clear $100/day?

Important:
- BBO liquidation assumes all residual quantity can execute at the best quote.
  This is an optimistic execution-price assumption. Recorded top-3 depth is
  reported separately as a capacity check.
- The top-3 sweep is only a depth diagnostic because deeper book levels were not
  persisted. If recorded top-3 depth is insufficient, actual full liquidation
  cost is unknown and is NOT invented.
- Fill-retention stress is a post-replay robustness exercise, not a new execution
  simulation. It does not model our hypothetical quote changing public flow.
"""

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C

STUDY_VERSION = "MM_EVENT_TIME_C_EXEC_LIQ_CAPACITY_STRESS_DEV_V1"
EXPECTED_SESSION_NAME = "20260816_070627"
EXPECTED_CAPTURE_VERSION = "MM_EVENT_TIME_M0_M5_V5_DEV"
EXPECTED_AUDIT_VERSION = "MM_EVENT_TIME_M0_M5_V5_AUDIT_V1"
EXPECTED_AUDIT_VERDICT = "PASS_FOR_DEVELOPMENT"
EXPECTED_REPLAY_VERSION = "MM_EVENT_TIME_ABCD_CAPACITY_DEV_V1"

POLICY = "C_L3_SPREAD2"
SIZES = (1.0, 2.0, 5.0, 10.0)
FOCUS_SIZE = 10.0
TARGET_USD_PER_DAY = 100.0
BTC_SERIES = "KXBTC15M"
EPS = 1e-12

RETENTION_PROBS = (0.75, 0.50, 0.25)
RETENTION_DRAWS = 2000
RETENTION_SEED = 20260816


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _ts(x):
    if x is None:
        return np.nan
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except Exception:
        return np.nan


def _iso(x):
    try:
        return datetime.fromtimestamp(float(x), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _bool_series(s):
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_and_validate(session: Path):
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"This development stress is hard-bound to {EXPECTED_SESSION_NAME}; got {session.name}"
        )
    manifest = _read_json(session / "session_manifest.json", {}) or {}
    if manifest.get("study_version") != EXPECTED_CAPTURE_VERSION:
        raise RuntimeError(
            f"Expected capture {EXPECTED_CAPTURE_VERSION}, found {manifest.get('study_version')}"
        )

    audit_dir = C.PROJECT_ROOT / "results" / "kalshi_mm_event_m0_m5_v5_audit" / session.name
    audit = _read_json(audit_dir / "audit_summary.json", {}) or {}
    if audit.get("audit_version") != EXPECTED_AUDIT_VERSION:
        raise RuntimeError("Required V5 audit output missing or wrong version")
    if audit.get("verdict") != EXPECTED_AUDIT_VERDICT:
        raise RuntimeError(f"Audit verdict is {audit.get('verdict')!r}, refusing analysis")

    replay_dir = C.PROJECT_ROOT / "results" / "kalshi_mm_event_abcd_capacity_dev" / session.name
    spec = _read_json(replay_dir / "study_spec.json", {}) or {}
    if spec.get("study_version") != EXPECTED_REPLAY_VERSION:
        raise RuntimeError("Required A/B/C/D replay output missing or wrong version")
    if spec.get("hard_bound_session_name") != EXPECTED_SESSION_NAME:
        raise RuntimeError("A/B/C/D replay was not hard-bound to expected V5 development session")
    if abs(_f(spec.get("economic_target_usd_per_day")) - TARGET_USD_PER_DAY) > 1e-9:
        raise RuntimeError("A/B/C/D replay target does not match the pre-registered $100/day target")

    quality = pd.read_csv(audit_dir / "contract_event_time_quality.csv")
    quality = quality[_bool_series(quality["full_m0_m5_plus_30_boundary"])].copy()
    eligible = set(quality.ticker.astype(str))
    if not eligible:
        raise RuntimeError("No complete contracts are eligible")

    contracts = pd.read_csv(replay_dir / "contract_results.csv")
    fills = pd.read_csv(replay_dir / "fills.csv")
    counts = pd.read_csv(replay_dir / "policy_counts.csv")

    contracts = contracts[
        (contracts.policy == POLICY)
        & pd.to_numeric(contracts.quote_size, errors="coerce").isin(SIZES)
        & contracts.ticker.astype(str).isin(eligible)
    ].copy()
    fills = fills[
        (fills.policy == POLICY)
        & pd.to_numeric(fills.quote_size, errors="coerce").isin(SIZES)
        & fills.ticker.astype(str).isin(eligible)
    ].copy()
    counts = counts[
        (counts.policy == POLICY)
        & pd.to_numeric(counts.quote_size, errors="coerce").isin(SIZES)
        & counts.ticker.astype(str).isin(eligible)
    ].copy()

    expected_contract_rows = len(eligible) * len(SIZES)
    if len(contracts) != expected_contract_rows:
        raise RuntimeError(
            f"Expected {expected_contract_rows} C-policy contract rows, found {len(contracts)}"
        )

    exposure_hours = _f(spec.get("analyzed_exposure_hours"))
    if not np.isfinite(exposure_hours) or exposure_hours <= 0:
        raise RuntimeError("Missing analyzed exposure hours from prior replay")

    return audit, spec, quality, eligible, contracts, fills, counts, exposure_hours


def _extract_m5_books(session: Path, eligible: set[str], *, show=True):
    """Stream persisted book once and retain the best available M5 state per ticker.

    Preference:
      1. valid trade_window_end boundary row (book state held at the M5 boundary)
      2. latest valid state with elapsed <= 300s
    """
    boundary_state = {}
    last_research_state = {}
    path = session / "book_top3_events.jsonl"

    with path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ticker = str(r.get("ticker") or "")
            if ticker not in eligible:
                continue

            valid = bool(r.get("valid_bbo"))
            bid, ask = _f(r.get("yes_bid")), _f(r.get("yes_ask"))
            bids = r.get("bid_levels") or []
            asks = r.get("ask_levels") or []
            elapsed = _f(r.get("elapsed_s"))
            typ = str(r.get("event_type") or "")
            t = _ts(r.get("receipt_time"))

            if not (
                valid
                and np.isfinite(bid)
                and np.isfinite(ask)
                and 0.0 <= bid < ask <= 1.0
                and bids
                and asks
            ):
                continue

            try:
                bid_levels = [
                    (float(x[0]), max(0.0, float(x[1])))
                    for x in bids[:3]
                    if len(x) >= 2 and float(x[1]) > 0
                ]
                ask_levels = [
                    (float(x[0]), max(0.0, float(x[1])))
                    for x in asks[:3]
                    if len(x) >= 2 and float(x[1]) > 0
                ]
            except Exception:
                continue
            if not bid_levels or not ask_levels:
                continue

            state = {
                "ticker": ticker,
                "receipt_ts": t,
                "receipt_time": _iso(t),
                "elapsed_s": elapsed,
                "event_type": typ,
                "bid": float(bid),
                "ask": float(ask),
                "mid": 0.5 * (float(bid) + float(ask)),
                "spread_c": 100.0 * (float(ask) - float(bid)),
                "bid_levels": bid_levels,
                "ask_levels": ask_levels,
                "bid_depth3": float(sum(q for _, q in bid_levels)),
                "ask_depth3": float(sum(q for _, q in ask_levels)),
            }

            if typ == "trade_window_end":
                prev = boundary_state.get(ticker)
                if prev is None or (
                    np.isfinite(elapsed)
                    and (
                        not np.isfinite(_f(prev.get("elapsed_s")))
                        or abs(elapsed - 300.0) < abs(_f(prev.get("elapsed_s")) - 300.0)
                    )
                ):
                    boundary_state[ticker] = state

            if np.isfinite(elapsed) and elapsed <= 300.0 + 1e-9:
                prev = last_research_state.get(ticker)
                if prev is None or elapsed >= _f(prev.get("elapsed_s"), -np.inf):
                    last_research_state[ticker] = state

            if show and n % 1_000_000 == 0:
                print(f"  streamed {n:,} / book rows...")

    out = {}
    for ticker in eligible:
        s = boundary_state.get(ticker) or last_research_state.get(ticker)
        if s is not None:
            s = dict(s)
            s["source"] = "trade_window_end" if ticker in boundary_state else "latest_valid_le_m5"
            out[ticker] = s
    return out


def _fifo_lots(fill_rows):
    long_lots = deque()
    short_lots = deque()
    matched = 0.0

    rows = sorted(
        fill_rows,
        key=lambda x: (float(x["fill_ts"]), 0 if str(x["side"]) == "BID" else 1),
    )
    for f in rows:
        qty = float(f["qty"])
        px = float(f["price"])
        side = str(f["side"])

        if side == "BID":
            remaining = qty
            while remaining > EPS and short_lots:
                sq, spx = short_lots[0]
                m = min(remaining, sq)
                matched += (spx - px) * m
                remaining -= m
                sq -= m
                if sq <= EPS:
                    short_lots.popleft()
                else:
                    short_lots[0] = (sq, spx)
            if remaining > EPS:
                long_lots.append((remaining, px))
        elif side == "ASK":
            remaining = qty
            while remaining > EPS and long_lots:
                lq, lpx = long_lots[0]
                m = min(remaining, lq)
                matched += (px - lpx) * m
                remaining -= m
                lq -= m
                if lq <= EPS:
                    long_lots.popleft()
                else:
                    long_lots[0] = (lq, lpx)
            if remaining > EPS:
                short_lots.append((remaining, px))

    return float(matched), list(long_lots), list(short_lots)


def _residual_inventory(long_lots, short_lots):
    return float(sum(q for q, _ in long_lots) - sum(q for q, _ in short_lots))


def _residual_mid_pnl(long_lots, short_lots, mid):
    if not np.isfinite(mid):
        return np.nan
    pnl = sum((mid - px) * q for q, px in long_lots)
    pnl += sum((px - mid) * q for q, px in short_lots)
    return float(pnl)


def _residual_bbo_pnl(long_lots, short_lots, state):
    if state is None:
        return np.nan
    bid, ask = _f(state.get("bid")), _f(state.get("ask"))
    if not (np.isfinite(bid) and np.isfinite(ask)):
        return np.nan
    pnl = sum((bid - px) * q for q, px in long_lots)
    pnl += sum((px - ask) * q for q, px in short_lots)
    return float(pnl)


def _top3_sweep(long_lots, short_lots, state):
    """Sweep residual inventory against recorded top3 depth.

    Returns realized PnL on the quantity that can be covered by top3, plus
    depth coverage. No price is invented for quantity beyond top3.
    """
    inv = _residual_inventory(long_lots, short_lots)
    target_qty = abs(inv)
    if target_qty <= EPS:
        return {
            "target_qty": 0.0,
            "available_top3_qty": np.nan,
            "liquidated_qty": 0.0,
            "coverage_pct": 100.0,
            "full_cover": True,
            "partial_realized_pnl": 0.0,
            "vwap": np.nan,
        }
    if state is None:
        return {
            "target_qty": target_qty,
            "available_top3_qty": 0.0,
            "liquidated_qty": 0.0,
            "coverage_pct": 0.0,
            "full_cover": False,
            "partial_realized_pnl": np.nan,
            "vwap": np.nan,
        }

    if inv > 0:
        levels = [(float(px), float(q)) for px, q in state.get("bid_levels", []) if q > 0]
        lots = deque((float(q), float(px)) for q, px in long_lots)
        direction = "SELL"
    else:
        levels = [(float(px), float(q)) for px, q in state.get("ask_levels", []) if q > 0]
        lots = deque((float(q), float(px)) for q, px in short_lots)
        direction = "BUY"

    available = float(sum(q for _, q in levels))
    remaining = target_qty
    liquidated = 0.0
    proceeds_or_cost = 0.0
    pnl = 0.0

    level_i = 0
    level_remaining = levels[0][1] if levels else 0.0

    while remaining > EPS and lots and level_i < len(levels):
        exit_px = levels[level_i][0]
        lot_q, entry_px = lots[0]
        m = min(remaining, lot_q, level_remaining)
        if m <= EPS:
            if level_remaining <= EPS:
                level_i += 1
                if level_i < len(levels):
                    level_remaining = levels[level_i][1]
            if lot_q <= EPS:
                lots.popleft()
            continue

        if direction == "SELL":
            pnl += (exit_px - entry_px) * m
        else:
            pnl += (entry_px - exit_px) * m

        proceeds_or_cost += exit_px * m
        liquidated += m
        remaining -= m
        lot_q -= m
        level_remaining -= m

        if lot_q <= EPS:
            lots.popleft()
        else:
            lots[0] = (lot_q, entry_px)

        if level_remaining <= EPS:
            level_i += 1
            if level_i < len(levels):
                level_remaining = levels[level_i][1]

    coverage = 100.0 * liquidated / target_qty if target_qty > EPS else 100.0
    return {
        "target_qty": target_qty,
        "available_top3_qty": available,
        "liquidated_qty": liquidated,
        "coverage_pct": coverage,
        "full_cover": bool(liquidated >= target_qty - 1e-9),
        "partial_realized_pnl": float(pnl) if liquidated > EPS else np.nan,
        "vwap": proceeds_or_cost / liquidated if liquidated > EPS else np.nan,
    }


def _contract_liquidation_rows(contracts, fills, m5_books):
    rows = []
    fill_groups = {
        (str(t), float(q)): z.to_dict("records")
        for (t, q), z in fills.groupby(["ticker", "quote_size"], sort=False)
    }

    for r in contracts.itertuples(index=False):
        ticker = str(r.ticker)
        size = float(r.quote_size)
        fr = fill_groups.get((ticker, size), [])
        matched, long_lots, short_lots = _fifo_lots(fr)
        inv = _residual_inventory(long_lots, short_lots)
        state = m5_books.get(ticker)

        mid = _f(getattr(r, "final_mid_m5", np.nan))
        residual_mid = _residual_mid_pnl(long_lots, short_lots, mid)
        residual_bbo = _residual_bbo_pnl(long_lots, short_lots, state)
        sweep = _top3_sweep(long_lots, short_lots, state)

        original_matched = _f(getattr(r, "matched_roundtrip_pnl", np.nan))
        original_residual = _f(getattr(r, "residual_inventory_m5_mtm", np.nan))
        original_net = _f(getattr(r, "net_mtm_pnl_before_fees", np.nan))

        rows.append({
            "policy": POLICY,
            "quote_size": size,
            "ticker": ticker,
            "series": str(r.series),
            "close_ts": _f(r.close_ts),
            "close_time": getattr(r, "close_time", None),
            "fill_events": int(getattr(r, "fill_events", len(fr))),
            "fill_qty": _f(getattr(r, "fill_qty", np.nan)),
            "ending_inventory_yes_equiv": inv,
            "max_abs_inventory": _f(getattr(r, "max_abs_inventory", np.nan)),
            "matched_roundtrip_pnl_recomputed": matched,
            "matched_roundtrip_pnl_original": original_matched,
            "matched_recompute_error": matched - original_matched if np.isfinite(original_matched) else np.nan,
            "residual_mid_pnl_recomputed": residual_mid,
            "residual_mid_pnl_original": original_residual,
            "residual_mid_recompute_error": residual_mid - original_residual if np.isfinite(original_residual) else np.nan,
            "net_mid_original": original_net,
            "m5_book_available": state is not None,
            "m5_state_source": state.get("source") if state else None,
            "m5_state_elapsed_s": _f(state.get("elapsed_s")) if state else np.nan,
            "m5_bid": _f(state.get("bid")) if state else np.nan,
            "m5_ask": _f(state.get("ask")) if state else np.nan,
            "m5_spread_c": _f(state.get("spread_c")) if state else np.nan,
            "m5_bid_depth3": _f(state.get("bid_depth3")) if state else np.nan,
            "m5_ask_depth3": _f(state.get("ask_depth3")) if state else np.nan,
            "residual_bbo_liquidation_pnl": residual_bbo,
            "net_bbo_liquidation_pnl": matched + residual_bbo if np.isfinite(residual_bbo) else np.nan,
            "bbo_vs_mid_liquidation_cost": residual_bbo - residual_mid if np.isfinite(residual_bbo) and np.isfinite(residual_mid) else np.nan,
            "top3_target_qty": sweep["target_qty"],
            "top3_available_qty": sweep["available_top3_qty"],
            "top3_liquidated_qty": sweep["liquidated_qty"],
            "top3_coverage_pct": sweep["coverage_pct"],
            "top3_full_cover": sweep["full_cover"],
            "top3_partial_realized_residual_pnl": sweep["partial_realized_pnl"],
            "top3_vwap": sweep["vwap"],
            "net_top3_full_liquidation_pnl": (
                matched + sweep["partial_realized_pnl"]
                if sweep["full_cover"] and np.isfinite(_f(sweep["partial_realized_pnl"]))
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _exposure_scale(exposure_hours):
    return 24.0 / float(exposure_hours)


def _summarize_sizes(liq, fills, exposure_hours):
    scale = _exposure_scale(exposure_hours)
    rows = []
    scopes = {
        "ALL_9": lambda z: z,
        "NON_BTC_8": lambda z: z[z.series != BTC_SERIES],
    }

    for size in SIZES:
        c0 = liq[np.isclose(liq.quote_size.astype(float), size)].copy()
        f0 = fills[np.isclose(pd.to_numeric(fills.quote_size, errors="coerce"), size)].copy()

        for scope, fn in scopes.items():
            c = fn(c0)
            f = fn(f0) if len(f0) else f0

            mid_net = pd.to_numeric(c.net_mid_original, errors="coerce").sum()
            matched = pd.to_numeric(c.matched_roundtrip_pnl_recomputed, errors="coerce").sum()
            bbo_resid = pd.to_numeric(c.residual_bbo_liquidation_pnl, errors="coerce").sum(min_count=1)
            bbo_net = pd.to_numeric(c.net_bbo_liquidation_pnl, errors="coerce").sum(min_count=1)
            qty = pd.to_numeric(f.qty, errors="coerce").sum() if len(f) else 0.0

            resid_mask = pd.to_numeric(c.ending_inventory_yes_equiv, errors="coerce").abs() > EPS
            resid = c[resid_mask]
            full_cover_pct = (
                100.0 * resid.top3_full_cover.astype(bool).mean()
                if len(resid) else 100.0
            )
            target_qty = pd.to_numeric(resid.top3_target_qty, errors="coerce").sum()
            liq_qty = pd.to_numeric(resid.top3_liquidated_qty, errors="coerce").sum()
            weighted_top3_cov = 100.0 * liq_qty / target_qty if target_qty > EPS else 100.0

            rows.append({
                "policy": POLICY,
                "quote_size": size,
                "scope": scope,
                "contracts": len(c),
                "fill_events": len(f),
                "fill_qty": qty,
                "mid_mark_net_pnl": mid_net,
                "mid_mark_net_pnl_per_day": mid_net * scale,
                "matched_roundtrip_pnl": matched,
                "matched_roundtrip_pnl_per_day": matched * scale,
                "bbo_residual_liquidation_pnl": bbo_resid,
                "bbo_residual_liquidation_pnl_per_day": bbo_resid * scale if np.isfinite(bbo_resid) else np.nan,
                "bbo_liquidated_net_pnl": bbo_net,
                "bbo_liquidated_net_pnl_per_day": bbo_net * scale if np.isfinite(bbo_net) else np.nan,
                "bbo_net_cents_per_filled_contract": 100.0 * bbo_net / qty if qty > EPS and np.isfinite(bbo_net) else np.nan,
                "mid_to_bbo_pnl_change_per_day": (bbo_net - mid_net) * scale if np.isfinite(bbo_net) else np.nan,
                "residual_contracts": len(resid),
                "top3_full_cover_contract_pct": full_cover_pct,
                "top3_residual_qty_coverage_pct": weighted_top3_cov,
                "target_100_day_after_bbo_liquidation": bool(np.isfinite(bbo_net) and bbo_net * scale >= TARGET_USD_PER_DAY),
                "matched_positive": bool(matched > 0),
            })
    return pd.DataFrame(rows)


def _window_table(c):
    rows = []
    for close_ts, z in c.groupby("close_ts", sort=True):
        rows.append({
            "close_ts": float(close_ts),
            "close_time": _iso(close_ts),
            "contracts": len(z),
            "mid_mark_net_pnl": pd.to_numeric(z.net_mid_original, errors="coerce").sum(),
            "matched_roundtrip_pnl": pd.to_numeric(z.matched_roundtrip_pnl_recomputed, errors="coerce").sum(),
            "bbo_residual_liquidation_pnl": pd.to_numeric(z.residual_bbo_liquidation_pnl, errors="coerce").sum(min_count=1),
            "bbo_liquidated_net_pnl": pd.to_numeric(z.net_bbo_liquidation_pnl, errors="coerce").sum(min_count=1),
        })
    w = pd.DataFrame(rows).sort_values("close_ts").reset_index(drop=True)
    if len(w):
        w["cum_bbo_pnl"] = w.bbo_liquidated_net_pnl.cumsum()
        w["running_peak"] = np.maximum(0.0, w.cum_bbo_pnl.cummax())
        w["drawdown"] = w.cum_bbo_pnl - w.running_peak
    return w


def _chronology_q10(liq, exposure_hours):
    q = liq[np.isclose(liq.quote_size.astype(float), FOCUS_SIZE)].copy()
    w = _window_table(q)
    if w.empty:
        return pd.DataFrame()
    cut = len(w) // 2
    rows = []
    for label, z in (("EARLY_HALF", w.iloc[:cut]), ("LATE_HALF", w.iloc[cut:])):
        if z.empty:
            continue
        hours = len(z) * 0.25
        scale = 24.0 / hours if hours > 0 else np.nan
        rows.append({
            "half": label,
            "windows": len(z),
            "hours": hours,
            "bbo_liquidated_net_pnl": z.bbo_liquidated_net_pnl.sum(),
            "bbo_liquidated_pnl_per_day": z.bbo_liquidated_net_pnl.sum() * scale,
            "matched_pnl": z.matched_roundtrip_pnl.sum(),
            "bbo_residual_pnl": z.bbo_residual_liquidation_pnl.sum(),
            "median_window_pnl": z.bbo_liquidated_net_pnl.median(),
            "positive_window_pct": 100.0 * (z.bbo_liquidated_net_pnl > 0).mean(),
            "worst_window": z.bbo_liquidated_net_pnl.min(),
            "max_drawdown_within_half": (
                (
                    z.bbo_liquidated_net_pnl.cumsum()
                    - np.maximum(0.0, z.bbo_liquidated_net_pnl.cumsum().cummax())
                ).min()
            ),
        })
    return pd.DataFrame(rows)


def _by_series_q10(liq, exposure_hours):
    q = liq[np.isclose(liq.quote_size.astype(float), FOCUS_SIZE)].copy()
    scale = _exposure_scale(exposure_hours)
    rows = []
    for series, z in q.groupby("series", sort=True):
        resid = z[pd.to_numeric(z.ending_inventory_yes_equiv, errors="coerce").abs() > EPS]
        target = pd.to_numeric(resid.top3_target_qty, errors="coerce").sum()
        covered = pd.to_numeric(resid.top3_liquidated_qty, errors="coerce").sum()
        rows.append({
            "series": series,
            "contracts": len(z),
            "fill_qty": pd.to_numeric(z.fill_qty, errors="coerce").sum(),
            "mid_mark_net_pnl": pd.to_numeric(z.net_mid_original, errors="coerce").sum(),
            "bbo_liquidated_net_pnl": pd.to_numeric(z.net_bbo_liquidation_pnl, errors="coerce").sum(min_count=1),
            "bbo_liquidated_pnl_per_day": pd.to_numeric(z.net_bbo_liquidation_pnl, errors="coerce").sum(min_count=1) * scale,
            "matched_roundtrip_pnl": pd.to_numeric(z.matched_roundtrip_pnl_recomputed, errors="coerce").sum(),
            "bbo_residual_pnl": pd.to_numeric(z.residual_bbo_liquidation_pnl, errors="coerce").sum(min_count=1),
            "residual_contracts": len(resid),
            "top3_full_cover_contract_pct": 100.0 * resid.top3_full_cover.astype(bool).mean() if len(resid) else 100.0,
            "top3_residual_qty_coverage_pct": 100.0 * covered / target if target > EPS else 100.0,
        })
    return pd.DataFrame(rows)


def _fill_capacity_diag_q10(fills, counts):
    f = fills[np.isclose(pd.to_numeric(fills.quote_size, errors="coerce"), FOCUS_SIZE)].copy()
    c = counts[np.isclose(pd.to_numeric(counts.quote_size, errors="coerce"), FOCUS_SIZE)].copy()

    qty = pd.to_numeric(f.qty, errors="coerce")
    qa = pd.to_numeric(f.queue_ahead_initial, errors="coerce")
    avail = pd.to_numeric(f.aggressive_qty_available_after_queue, errors="coerce")

    opens = int(
        pd.to_numeric(
            c[c.reason.astype(str).isin({"BID_OPEN", "ASK_OPEN"})]["count"],
            errors="coerce",
        ).sum()
    )
    fills_n = len(f)
    full = int((qty >= FOCUS_SIZE - 1e-9).sum())
    partial = fills_n - full

    qvals = qty.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if fills_n else pd.Series(dtype=float)
    qq = qa.dropna().quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if qa.notna().any() else pd.Series(dtype=float)

    return pd.DataFrame([{
        "policy": POLICY,
        "quote_size": FOCUS_SIZE,
        "quote_opens": opens,
        "fill_events": fills_n,
        "any_fill_event_per_open_pct": 100.0 * fills_n / opens if opens else np.nan,
        "full_q10_fill_events": full,
        "partial_fill_events": partial,
        "full_q10_pct_of_filled_episodes": 100.0 * full / fills_n if fills_n else np.nan,
        "full_q10_pct_of_all_quote_opens": 100.0 * full / opens if opens else np.nan,
        "fill_qty_mean": qty.mean() if fills_n else np.nan,
        "fill_qty_p10": qvals.get(0.10, np.nan),
        "fill_qty_p25": qvals.get(0.25, np.nan),
        "fill_qty_p50": qvals.get(0.50, np.nan),
        "fill_qty_p75": qvals.get(0.75, np.nan),
        "fill_qty_p90": qvals.get(0.90, np.nan),
        "fill_qty_p95": qvals.get(0.95, np.nan),
        "fill_qty_p99": qvals.get(0.99, np.nan),
        "queue_ahead_p10": qq.get(0.10, np.nan),
        "queue_ahead_p25": qq.get(0.25, np.nan),
        "queue_ahead_p50": qq.get(0.50, np.nan),
        "queue_ahead_p75": qq.get(0.75, np.nan),
        "queue_ahead_p90": qq.get(0.90, np.nan),
        "queue_ahead_p95": qq.get(0.95, np.nan),
        "queue_ahead_p99": qq.get(0.99, np.nan),
        "exact_price_fill_pct": 100.0 * (~f.trade_through.astype(bool)).mean() if fills_n else np.nan,
        "trade_through_fill_pct": 100.0 * f.trade_through.astype(bool).mean() if fills_n else np.nan,
        "median_aggressive_qty_after_queue_exact_price": avail[~f.trade_through.astype(bool)].median() if fills_n else np.nan,
    }])


def _retention_stress_q10(liq, fills, m5_books, exposure_hours):
    """Seeded Bernoulli fill-event retention stress.

    Each existing Q10 fill event is independently retained with probability p.
    Retained events keep their simulated quantity. FIFO matched PnL and residual
    inventory are then recomputed and residual inventory is liquidated at M5 BBO.
    """
    qfills = fills[np.isclose(pd.to_numeric(fills.quote_size, errors="coerce"), FOCUS_SIZE)].copy()
    if qfills.empty:
        return pd.DataFrame(), pd.DataFrame()

    qfills = qfills.sort_values(["ticker", "fill_ts", "side", "price", "qty"]).reset_index(drop=True)
    records = qfills.to_dict("records")
    by_ticker_idx = defaultdict(list)
    for i, r in enumerate(records):
        by_ticker_idx[str(r["ticker"])].append(i)

    rng = np.random.default_rng(RETENTION_SEED)
    scale = _exposure_scale(exposure_hours)
    draw_rows = []

    for p in RETENTION_PROBS:
        for draw in range(RETENTION_DRAWS):
            keep = rng.random(len(records)) < p
            total_matched = 0.0
            total_resid = 0.0
            total_qty = 0.0
            missing_bbo = False

            for ticker, idxs in by_ticker_idx.items():
                selected = [records[i] for i in idxs if keep[i]]
                if not selected:
                    continue
                matched, longs, shorts = _fifo_lots(selected)
                resid = _residual_bbo_pnl(longs, shorts, m5_books.get(ticker))
                if not np.isfinite(resid):
                    missing_bbo = True
                    break
                total_matched += matched
                total_resid += resid
                total_qty += sum(float(x["qty"]) for x in selected)

            net = np.nan if missing_bbo else total_matched + total_resid

            draw_rows.append({
                "retention_prob": p,
                "draw": draw,
                "retained_fill_qty": total_qty,
                "matched_pnl": total_matched,
                "bbo_residual_pnl": total_resid,
                "net_bbo_pnl": net,
                "net_bbo_pnl_per_day": net * scale if np.isfinite(net) else np.nan,
                "target_100_day_reached": bool(np.isfinite(net) and net * scale >= TARGET_USD_PER_DAY),
            })

    draws = pd.DataFrame(draw_rows)
    summary_rows = []
    for p, z in draws.groupby("retention_prob", sort=False):
        x = pd.to_numeric(z.net_bbo_pnl_per_day, errors="coerce").dropna()
        q = x.quantile([0.05, 0.25, 0.50, 0.75, 0.95]) if len(x) else pd.Series(dtype=float)
        summary_rows.append({
            "retention_prob": p,
            "draws": len(z),
            "mean_retained_fill_qty": pd.to_numeric(z.retained_fill_qty, errors="coerce").mean(),
            "mean_net_bbo_pnl_per_day": x.mean() if len(x) else np.nan,
            "p05_net_bbo_pnl_per_day": q.get(0.05, np.nan),
            "p25_net_bbo_pnl_per_day": q.get(0.25, np.nan),
            "median_net_bbo_pnl_per_day": q.get(0.50, np.nan),
            "p75_net_bbo_pnl_per_day": q.get(0.75, np.nan),
            "p95_net_bbo_pnl_per_day": q.get(0.95, np.nan),
            "prob_ge_100_day_pct": 100.0 * (x >= TARGET_USD_PER_DAY).mean() if len(x) else np.nan,
            "prob_positive_day_pct": 100.0 * (x > 0).mean() if len(x) else np.nan,
        })
    return pd.DataFrame(summary_rows), draws


def run_c_executable_liquidation_capacity_stress(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if not session.exists():
        raise FileNotFoundError(session)

    audit, prior_spec, quality, eligible, contracts, fills, counts, exposure_hours = _load_and_validate(session)

    if show:
        print(
            f"Eligible complete contracts: {len(eligible)} | "
            f"exposure={exposure_hours:.2f}h | candidate={POLICY}"
        )
        print("Streaming V5 book once to recover observed M5 executable BBO/top3 state...")

    m5_books = _extract_m5_books(session, eligible, show=show)
    if len(m5_books) != len(eligible):
        missing = sorted(eligible - set(m5_books))
        raise RuntimeError(
            f"M5 executable book unavailable for {len(missing)} complete contracts; "
            f"first few={missing[:10]}"
        )

    liquidation = _contract_liquidation_rows(contracts, fills, m5_books)

    max_match_err = pd.to_numeric(
        liquidation.matched_recompute_error, errors="coerce"
    ).abs().max()
    max_mid_err = pd.to_numeric(
        liquidation.residual_mid_recompute_error, errors="coerce"
    ).abs().max()
    if max_match_err > 1e-8 or max_mid_err > 1e-8:
        raise RuntimeError(
            f"Replay reconstruction mismatch: matched={max_match_err}, residual_mid={max_mid_err}"
        )

    size_summary = _summarize_sizes(liquidation, fills, exposure_hours)
    q10_chronology = _chronology_q10(liquidation, exposure_hours)
    q10_series = _by_series_q10(liquidation, exposure_hours)
    q10_fill_capacity = _fill_capacity_diag_q10(fills, counts)
    retention_summary, retention_draws = _retention_stress_q10(
        liquidation, fills, m5_books, exposure_hours
    )

    q10 = liquidation[np.isclose(liquidation.quote_size.astype(float), FOCUS_SIZE)].copy()
    q10_windows = _window_table(q10)

    q10_all = size_summary[
        (size_summary.scope == "ALL_9")
        & np.isclose(size_summary.quote_size.astype(float), FOCUS_SIZE)
    ].iloc[0]

    primary_gate = {
        "q10_bbo_liquidated_rate_ge_100": bool(q10_all.bbo_liquidated_net_pnl_per_day >= TARGET_USD_PER_DAY),
        "q10_matched_positive": bool(q10_all.matched_roundtrip_pnl > 0),
        "q10_top3_residual_qty_coverage_ge_95pct": bool(q10_all.top3_residual_qty_coverage_pct >= 95.0),
        "q10_top3_full_cover_contract_pct_ge_90pct": bool(q10_all.top3_full_cover_contract_pct >= 90.0),
        "q10_both_chronological_halves_positive": bool(
            len(q10_chronology) == 2
            and (pd.to_numeric(q10_chronology.bbo_liquidated_net_pnl, errors="coerce") > 0).all()
        ),
    }
    dev_status = (
        "EXEC_LIQUIDATION_PASS_FOR_FREEZE_REVIEW"
        if all(primary_gate.values())
        else "DEVELOPMENT_ECONOMICS_INSUFFICIENT_OR_FRAGILE"
    )

    if output_dir is None:
        output_dir = (
            C.PROJECT_ROOT
            / "results"
            / "kalshi_mm_event_c_exec_liq_capacity_stress_dev"
            / session.name
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    liquidation.to_csv(out / "contract_liquidation_results.csv", index=False)
    size_summary.to_csv(out / "size_liquidation_summary.csv", index=False)
    q10_windows.to_csv(out / "q10_window_results.csv", index=False)
    q10_chronology.to_csv(out / "q10_chronology.csv", index=False)
    q10_series.to_csv(out / "q10_by_series.csv", index=False)
    q10_fill_capacity.to_csv(out / "q10_fill_capacity_diagnostics.csv", index=False)
    retention_summary.to_csv(out / "q10_fill_retention_stress_summary.csv", index=False)
    retention_draws.to_csv(out / "q10_fill_retention_stress_draws.csv", index=False)
    pd.DataFrame([{"gate": k, "pass": v} for k, v in primary_gate.items()]).to_csv(
        out / "freeze_review_gates.csv", index=False
    )

    spec = {
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "hard_bound_session_name": EXPECTED_SESSION_NAME,
        "candidate": POLICY,
        "signal_changed": False,
        "signal_rule": "same-side L3 support AND natural spread >= 2c",
        "sizes_reused_from_prior_replay": list(SIZES),
        "focus_size": FOCUS_SIZE,
        "economic_target_usd_per_day": TARGET_USD_PER_DAY,
        "m5_liquidation_primary": "residual long inventory sold at M5 best bid; residual short inventory bought at M5 best ask",
        "m5_liquidation_caveat": "BBO liquidation assumes sufficient quantity at best quote; top3 depth coverage is reported separately",
        "top3_rule": "sweep only persisted top3 depth; no deeper price or quantity is invented",
        "fill_retention_stress": {
            "probabilities": list(RETENTION_PROBS),
            "draws_per_probability": RETENTION_DRAWS,
            "seed": RETENTION_SEED,
            "unit": "simulated fill event",
            "meaning": "post-replay robustness haircut only; not a new queue/fill simulation",
        },
        "fees": 0.0,
        "threshold_sweep": False,
        "asset_filter": False,
        "side_filter": False,
        "minute_filter": False,
        "max_matched_reconstruction_error": max_match_err,
        "max_mid_residual_reconstruction_error": max_mid_err,
        "freeze_review_gates": primary_gate,
        "development_status": dev_status,
    }
    (out / "study_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    if show:
        print("\n" + "=" * 170)
        print("CANDIDATE C — EXECUTABLE M5 LIQUIDATION + Q10 CAPACITY STRESS — DEVELOPMENT ONLY")
        print("=" * 170)
        print(
            f"session={session.name} | contracts={len(eligible)} | exposure={exposure_hours:.2f}h | "
            f"candidate={POLICY}"
        )
        print("NO SIGNAL CHANGE: L3 support + natural spread >=2c")
        print(f"reconstruction max error: matched=${max_match_err:.12f} | residual-mid=${max_mid_err:.12f}")

        print("\nSIZE CURVE AFTER M5 BBO LIQUIDATION — ALL 9 + NON-BTC 8")
        cols = [
            "quote_size", "scope", "fill_qty",
            "mid_mark_net_pnl_per_day", "bbo_liquidated_net_pnl_per_day",
            "matched_roundtrip_pnl_per_day", "bbo_residual_liquidation_pnl_per_day",
            "bbo_net_cents_per_filled_contract", "mid_to_bbo_pnl_change_per_day",
            "top3_full_cover_contract_pct", "top3_residual_qty_coverage_pct",
            "target_100_day_after_bbo_liquidation", "matched_positive",
        ]
        print(size_summary[cols].round(4).to_string(index=False))

        print("\nQ10 CHRONOLOGY — EXECUTABLE BBO LIQUIDATION")
        print(q10_chronology.round(4).to_string(index=False))

        print("\nQ10 BY SERIES — EXECUTABLE BBO LIQUIDATION")
        print(q10_series.round(4).to_string(index=False))

        print("\nQ10 FILL / QUEUE CAPACITY DIAGNOSTICS")
        print(q10_fill_capacity.round(4).to_string(index=False))

        print("\nQ10 FILL-RETENTION STRESS — SEEDED MONTE CARLO")
        print(retention_summary.round(4).to_string(index=False))

        print("\nFREEZE-REVIEW GATES")
        for k, v in primary_gate.items():
            print(f"  {'PASS' if v else 'FAIL':4s}  {k}")
        print(f"\nDEVELOPMENT STATUS: {dev_status}")

        print("\nINTERPRETATION GUARDRAILS")
        print("  - This is still development data; no OOS claim is allowed.")
        print("  - Candidate C's signal and 2c spread rule were NOT changed.")
        print("  - BBO liquidation assumes sufficient quantity at the best quote; top3 depth coverage is the recorded capacity check.")
        print("  - If top3 cannot cover residual quantity, deeper liquidation cost is unknown and is not fabricated.")
        print("  - 75/50/25% retention is a post-replay robustness haircut, not proof of live fill probability.")
        print("  - Do not add asset, side, minute, spread, or imbalance filters after seeing this output.")
        print(f"\nOUTPUTS: {out}")
        print("=" * 170)

    return {
        "output_dir": out,
        "size_summary": size_summary,
        "q10_windows": q10_windows,
        "q10_chronology": q10_chronology,
        "q10_by_series": q10_series,
        "q10_fill_capacity": q10_fill_capacity,
        "retention_summary": retention_summary,
        "retention_draws": retention_draws,
        "contracts": liquidation,
        "freeze_review_gates": pd.DataFrame(
            [{"gate": k, "pass": v} for k, v in primary_gate.items()]
        ),
        "spec": spec,
    }
