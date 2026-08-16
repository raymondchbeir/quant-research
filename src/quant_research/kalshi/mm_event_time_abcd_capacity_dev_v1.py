from __future__ import annotations

"""Pre-registered V5 event-time A/B/C/D market-making development replay.

DEVELOPMENT ONLY. This module is hard-bound to the already-designated V5
session 20260816_070627 and refuses to run unless the prior V5 integrity audit
returned PASS_FOR_DEVELOPMENT.

No threshold search is performed. The only four quoting rules are exactly the
candidate family written into development_plan.json before this recording:

A  L3 side support only
B  L1 and L3 both support the quoted side
C  L3 side support and natural spread >= 2c
D  L1 and L3 both support the quoted side and natural spread >= 2c

Execution reference
-------------------
- Q1 at the public YES BBO.
- Join behind displayed L1 quantity.
- Exact-price opposing aggressive trades consume queue ahead first.
- Trade-through fills the remaining hypothetical quote.
- No cancellation-ahead credit.
- Any partial/full fill cancels the residual same-side quote, matching the
  prior event-time development replay convention.
- Quotes cancel/reprice causally on event-time BBO/state changes.
- No fees; break-even fee cushion is reported.
- Inventory remaining at M5 is marked to the final valid midpoint at/before M5.

Capacity scenarios
------------------
Q2/Q5/Q10 use the SAME historical-flow replay rather than naive linear PnL
multiplication. They remain counterfactual because our larger hypothetical
quote could have changed queue dynamics and the public historical flow.

The economic target pre-registered before collection is >= $100/day plausible
capacity. A positive historical rate is not proof of live capacity.
"""

import heapq
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C

STUDY_VERSION = "MM_EVENT_TIME_ABCD_CAPACITY_DEV_V1"
EXPECTED_SESSION_NAME = "20260816_070627"
EXPECTED_CAPTURE_VERSION = "MM_EVENT_TIME_M0_M5_V5_DEV"
EXPECTED_AUDIT_VERSION = "MM_EVENT_TIME_M0_M5_V5_AUDIT_V1"
EXPECTED_AUDIT_VERDICT = "PASS_FOR_DEVELOPMENT"

POLICIES = ("A_L3", "B_L1_L3", "C_L3_SPREAD2", "D_L1_L3_SPREAD2")
SIZES = (1.0, 2.0, 5.0, 10.0)
REFERENCE_SIZE = 1.0
MARKOUTS_S = (5.0, 15.0, 30.0)
SPREAD_FLOOR_C = 2.0  # pre-registered, not selected from this session
BTC_SERIES = "KXBTC15M"
EPS = 1e-12
MAX_FUTURE_MID_AGE_S = 2.0


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


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _bool_series(s):
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def _top_state(row):
    if not bool(row.get("valid_bbo")):
        return None
    bids = row.get("bid_levels") or []
    asks = row.get("ask_levels") or []
    if not bids or not asks:
        return None
    try:
        bid = float(row["yes_bid"])
        ask = float(row["yes_ask"])
        bq = max(0.0, float(row["yes_bid_size"]))
        aq = max(0.0, float(row["yes_ask_size"]))
        bdepth3 = max(0.0, sum(float(x[1]) for x in bids[:3]))
        adepth3 = max(0.0, sum(float(x[1]) for x in asks[:3]))
    except Exception:
        return None
    if not (0.0 <= bid < ask <= 1.0):
        return None
    return {
        "bid": bid,
        "ask": ask,
        "bid_q1": bq,
        "ask_q1": aq,
        "bid_depth3": bdepth3,
        "ask_depth3": adepth3,
        "mid": 0.5 * (bid + ask),
        "spread_c": 100.0 * (ask - bid),
    }


def _policy_allows(policy: str, side: str, cur: dict) -> bool:
    if side == "BID":
        l1 = cur["bid_q1"] > cur["ask_q1"] + EPS
        l3 = cur["bid_depth3"] > cur["ask_depth3"] + EPS
    else:
        l1 = cur["ask_q1"] > cur["bid_q1"] + EPS
        l3 = cur["ask_depth3"] > cur["bid_depth3"] + EPS
    spread2 = cur["spread_c"] + 1e-9 >= SPREAD_FLOOR_C

    if policy == "A_L3":
        return bool(l3)
    if policy == "B_L1_L3":
        return bool(l1 and l3)
    if policy == "C_L3_SPREAD2":
        return bool(l3 and spread2)
    if policy == "D_L1_L3_SPREAD2":
        return bool(l1 and l3 and spread2)
    raise RuntimeError(f"Unknown policy: {policy}")


def _load_inputs(session: Path):
    manifest = _read_json(session / "session_manifest.json", {}) or {}
    plan = _read_json(session / "development_plan.json", {}) or manifest.get("development_plan") or {}
    capture_version = manifest.get("study_version")
    if capture_version != EXPECTED_CAPTURE_VERSION:
        raise RuntimeError(f"Expected capture {EXPECTED_CAPTURE_VERSION}, found {capture_version}")

    audit_dir = C.PROJECT_ROOT / "results" / "kalshi_mm_event_m0_m5_v5_audit" / session.name
    audit_summary = _read_json(audit_dir / "audit_summary.json", {}) or {}
    if audit_summary.get("audit_version") != EXPECTED_AUDIT_VERSION:
        raise RuntimeError("Required V5 audit output is missing or wrong version")
    if audit_summary.get("verdict") != EXPECTED_AUDIT_VERDICT:
        raise RuntimeError(
            f"Refusing strategy replay because audit verdict is {audit_summary.get('verdict')!r}"
        )
    gates = audit_summary.get("quality_gates") or {}
    if not gates or not all(bool(v) for v in gates.values()):
        raise RuntimeError("Refusing strategy replay because not all V5 quality gates passed")

    qpath = audit_dir / "contract_event_time_quality.csv"
    quality = pd.read_csv(qpath)
    if "full_m0_m5_plus_30_boundary" not in quality.columns:
        raise RuntimeError("Audit contract quality file lacks full tail boundary field")
    quality = quality[_bool_series(quality["full_m0_m5_plus_30_boundary"])].copy()
    eligible = set(quality.ticker.astype(str))
    if not eligible:
        raise RuntimeError("No complete V5 contracts are eligible")

    target = _f(plan.get("economic_target_usd_per_day"))
    if not np.isfinite(target) or abs(target - 100.0) > 1e-9:
        raise RuntimeError(f"Expected pre-registered $100/day target, found {target}")
    fam = plan.get("candidate_family") or {}
    if set(fam) != {"A", "B", "C", "D"}:
        raise RuntimeError(f"Unexpected pre-registered candidate family keys: {sorted(fam)}")
    if list(plan.get("capacity_scenarios") or []) != [1, 2, 5, 10]:
        raise RuntimeError("Expected pre-registered Q1/Q2/Q5/Q10 capacity scenarios")

    return manifest, plan, audit_summary, quality, eligible


def _load_meta(session: Path, eligible: set[str]):
    out = {}
    with (session / "market_metadata.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ticker = str(r.get("ticker") or "")
            if ticker not in eligible:
                continue
            close = _ts(r.get("close_time"))
            if not np.isfinite(close):
                continue
            out[ticker] = {
                "ticker": ticker,
                "series": str(r.get("series_ticker") or "UNKNOWN"),
                "close_ts": float(close),
                "m0_ts": float(close) - 900.0,
                "m5_ts": float(close) - 600.0,
                "tail_end_ts": float(close) - 570.0,
            }
    missing = eligible - set(out)
    if missing:
        raise RuntimeError(f"Missing metadata for {len(missing)} eligible contracts")
    return out


def _load_research_trades(session: Path, eligible: set[str], meta: dict):
    out = defaultdict(list)
    path = session / "trades_event_time.jsonl"
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ticker = str(r.get("ticker") or "")
            if ticker not in eligible:
                continue
            t = _ts(r.get("receipt_time"))
            px = _f(r.get("yes_price"))
            qty = _f(r.get("qty"))
            side = str(r.get("taker_book_side") or "").lower()
            elapsed = _f(r.get("elapsed_s"))
            if not (np.isfinite(t) and np.isfinite(px) and np.isfinite(qty) and qty > 0):
                continue
            if side not in {"bid", "ask"}:
                continue
            if np.isfinite(elapsed):
                if not (0.0 <= elapsed < 300.0):
                    continue
            elif not (meta[ticker]["m0_ts"] <= t < meta[ticker]["m5_ts"]):
                continue
            out[ticker].append((float(t), float(px), float(qty), side))
    for ticker in out:
        out[ticker].sort(key=lambda x: x[0])
    return dict(out)


class Sim:
    __slots__ = (
        "policy", "size", "ticker", "meta", "active", "fills", "counts",
        "inventory", "cash", "max_abs_inventory", "last_research_mid", "last_research_mid_t",
        "episode_no",
    )

    def __init__(self, policy, size, ticker, meta):
        self.policy = policy
        self.size = float(size)
        self.ticker = ticker
        self.meta = meta
        self.active = {"BID": None, "ASK": None}
        self.fills = []
        self.counts = Counter()
        self.inventory = 0.0
        self.cash = 0.0
        self.max_abs_inventory = 0.0
        self.last_research_mid = np.nan
        self.last_research_mid_t = np.nan
        self.episode_no = 0

    def cancel(self, side, reason):
        if self.active[side] is not None:
            self.active[side] = None
            self.counts[f"{side}_CANCEL_{reason}"] += 1

    def cancel_all(self, reason):
        self.cancel("BID", reason)
        self.cancel("ASK", reason)

    def open(self, side, t, cur):
        px = cur["bid"] if side == "BID" else cur["ask"]
        qahead = cur["bid_q1"] if side == "BID" else cur["ask_q1"]
        self.episode_no += 1
        self.active[side] = {
            "price": float(px),
            "queue_ahead": float(qahead),
            "queue_ahead_initial": float(qahead),
            "remaining_qty": float(self.size),
            "join_ts": float(t),
            "spread_c_at_join": float(cur["spread_c"]),
            "l1_imbalance_at_join": (
                (cur["bid_q1"] - cur["ask_q1"]) / (cur["bid_q1"] + cur["ask_q1"])
                if cur["bid_q1"] + cur["ask_q1"] > EPS else 0.0
            ),
            "l3_imbalance_at_join": (
                (cur["bid_depth3"] - cur["ask_depth3"]) / (cur["bid_depth3"] + cur["ask_depth3"])
                if cur["bid_depth3"] + cur["ask_depth3"] > EPS else 0.0
            ),
        }
        self.counts[f"{side}_OPEN"] += 1

    def on_book(self, t, cur, in_research):
        if in_research:
            self.last_research_mid = float(cur["mid"])
            self.last_research_mid_t = float(t)
        if not in_research:
            self.cancel_all("M5_END")
            return

        for side in ("BID", "ASK"):
            desired_px = cur["bid"] if side == "BID" else cur["ask"]
            ep = self.active[side]
            if ep is not None and abs(float(ep["price"]) - float(desired_px)) > EPS:
                self.cancel(side, "BBO_REPRICE")
            allow = _policy_allows(self.policy, side, cur)
            if self.active[side] is not None and not allow:
                self.cancel(side, "STATE_FALSE")
            if self.active[side] is None and allow:
                self.open(side, t, cur)

    def apply_trade(self, tr_t, trade_px, trade_qty, taker_side, mid_at_fill, pending_heap, pending_counter):
        side = "BID" if taker_side == "ask" else "ASK" if taker_side == "bid" else None
        if side is None:
            return pending_counter
        ep = self.active[side]
        if ep is None:
            return pending_counter

        qpx = float(ep["price"])
        trade_through = False
        aggressive_available = 0.0
        if side == "BID":
            if trade_px < qpx - EPS:
                trade_through = True
                aggressive_available = float(ep["remaining_qty"])
            elif abs(trade_px - qpx) <= EPS:
                ahead = float(ep["queue_ahead"])
                used = min(ahead, trade_qty)
                ep["queue_ahead"] = ahead - used
                aggressive_available = max(0.0, trade_qty - used)
            else:
                return pending_counter
        else:
            if trade_px > qpx + EPS:
                trade_through = True
                aggressive_available = float(ep["remaining_qty"])
            elif abs(trade_px - qpx) <= EPS:
                ahead = float(ep["queue_ahead"])
                used = min(ahead, trade_qty)
                ep["queue_ahead"] = ahead - used
                aggressive_available = max(0.0, trade_qty - used)
            else:
                return pending_counter

        if aggressive_available <= EPS:
            return pending_counter

        qty = min(float(ep["remaining_qty"]), aggressive_available)
        if qty <= EPS:
            return pending_counter

        sign = 1.0 if side == "BID" else -1.0
        inv_before = self.inventory
        self.inventory += sign * qty
        self.cash += (-qpx * qty) if side == "BID" else (qpx * qty)
        self.max_abs_inventory = max(self.max_abs_inventory, abs(self.inventory))

        elapsed = float(tr_t) - float(self.meta["m0_ts"])
        fill = {
            "policy": self.policy,
            "quote_size": self.size,
            "capacity_scenario": "REFERENCE_Q1" if abs(self.size - 1.0) < EPS else "COUNTERFACTUAL",
            "ticker": self.ticker,
            "series": self.meta["series"],
            "close_ts": self.meta["close_ts"],
            "close_time": _iso(self.meta["close_ts"]),
            "side": side,
            "fill_ts": float(tr_t),
            "fill_time": _iso(tr_t),
            "elapsed_s": elapsed,
            "entry_minute": f"M{min(4, max(0, int(elapsed // 60)))}-M{min(5, max(1, int(elapsed // 60) + 1))}",
            "qty": qty,
            "price": qpx,
            "mid_at_fill": float(mid_at_fill) if np.isfinite(mid_at_fill) else np.nan,
            "gross_edge_at_fill_c": sign * (float(mid_at_fill) - qpx) * 100.0 if np.isfinite(mid_at_fill) else np.nan,
            "trade_through": bool(trade_through),
            "historical_trade_price": float(trade_px),
            "historical_trade_qty": float(trade_qty),
            "aggressive_qty_available_after_queue": aggressive_available if not trade_through else np.nan,
            "queue_ahead_initial": ep["queue_ahead_initial"],
            "spread_c_at_join": ep["spread_c_at_join"],
            "l1_imbalance_at_join": ep["l1_imbalance_at_join"],
            "l3_imbalance_at_join": ep["l3_imbalance_at_join"],
            "inventory_before_fill": inv_before,
            "inventory_after_fill": self.inventory,
        }
        for h in MARKOUTS_S:
            tag = f"{int(h)}s"
            fill[f"future_mid_{tag}"] = np.nan
            fill[f"markout_{tag}_c"] = np.nan
            fill[f"post_mid_move_{tag}_c"] = np.nan
            pending_counter += 1
            heapq.heappush(pending_heap, (float(tr_t) + h, pending_counter, fill, h, sign))
        self.fills.append(fill)
        self.counts[f"{side}_FILL_EVENT"] += 1
        self.counts[f"{side}_FILL_QTY_X1000"] += int(round(qty * 1000.0))
        if trade_through:
            self.counts[f"{side}_TRADE_THROUGH_FILL"] += 1
        else:
            self.counts[f"{side}_EXACT_PRICE_FILL"] += 1

        # Preserve the prior exact-replay convention: cancel any residual after a fill.
        self.cancel(side, "FILL_CANCEL_RESIDUAL")
        return pending_counter


def _resolve_pending(heap, book_t, mid):
    if not np.isfinite(mid):
        return
    while heap and heap[0][0] <= float(book_t) + EPS:
        target, _, fill, h, sign = heapq.heappop(heap)
        age = float(book_t) - float(target)
        tag = f"{int(h)}s"
        if -EPS <= age <= MAX_FUTURE_MID_AGE_S + EPS:
            fill[f"future_mid_{tag}"] = float(mid)
            fill[f"markout_{tag}_c"] = sign * (float(mid) - float(fill["price"])) * 100.0
            if np.isfinite(_f(fill.get("mid_at_fill"))):
                fill[f"post_mid_move_{tag}_c"] = sign * (float(mid) - float(fill["mid_at_fill"])) * 100.0


def _fifo_decompose(fills: list[dict], final_mid: float):
    long_lots = deque()
    short_lots = deque()
    matched = 0.0

    for f in sorted(fills, key=lambda x: (float(x["fill_ts"]), 0 if x["side"] == "BID" else 1)):
        qty = float(f["qty"])
        px = float(f["price"])
        if f["side"] == "BID":
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
        else:
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

    residual = np.nan
    residual_inv = sum(q for q, _ in long_lots) - sum(q for q, _ in short_lots)
    if np.isfinite(final_mid):
        residual = sum((final_mid - px) * q for q, px in long_lots)
        residual += sum((px - final_mid) * q for q, px in short_lots)
    return float(matched), float(residual) if np.isfinite(residual) else np.nan, float(residual_inv)


def _contract_result(sim: Sim):
    mid = sim.last_research_mid
    net = sim.cash + sim.inventory * mid if np.isfinite(mid) else np.nan
    gross = 0.0
    for f in sim.fills:
        ge = _f(f.get("gross_edge_at_fill_c"))
        if np.isfinite(ge):
            gross += ge / 100.0 * float(f["qty"])
    matched, residual, residual_inv = _fifo_decompose(sim.fills, mid)
    recon = matched + residual if np.isfinite(residual) else np.nan
    return {
        "policy": sim.policy,
        "quote_size": sim.size,
        "capacity_scenario": "REFERENCE_Q1" if abs(sim.size - 1.0) < EPS else "COUNTERFACTUAL",
        "ticker": sim.ticker,
        "series": sim.meta["series"],
        "close_ts": sim.meta["close_ts"],
        "close_time": _iso(sim.meta["close_ts"]),
        "fill_events": len(sim.fills),
        "fill_qty": sum(float(f["qty"]) for f in sim.fills),
        "bid_fill_qty": sum(float(f["qty"]) for f in sim.fills if f["side"] == "BID"),
        "ask_fill_qty": sum(float(f["qty"]) for f in sim.fills if f["side"] == "ASK"),
        "ending_inventory_yes_equiv": sim.inventory,
        "fifo_residual_inventory": residual_inv,
        "max_abs_inventory": sim.max_abs_inventory,
        "final_mid_m5": mid,
        "final_mid_age_to_m5_s": sim.meta["m5_ts"] - sim.last_research_mid_t if np.isfinite(sim.last_research_mid_t) else np.nan,
        "cash": sim.cash,
        "gross_spread_capture_dollars": gross,
        "matched_roundtrip_pnl": matched,
        "residual_inventory_m5_mtm": residual,
        "reconstructed_total_pnl": recon,
        "net_mtm_pnl_before_fees": net,
        "reconstruction_error": net - recon if np.isfinite(net) and np.isfinite(recon) else np.nan,
    }


def _wavg(df, col, weight="qty"):
    if df.empty or col not in df.columns:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    w = pd.to_numeric(df[weight], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def _window_table(cdf):
    rows = []
    for close, z in cdf.groupby("close_ts", sort=True):
        rows.append({
            "close_ts": float(close),
            "close_time": _iso(close),
            "contracts": len(z),
            "fill_qty": pd.to_numeric(z.fill_qty, errors="coerce").sum(),
            "matched_roundtrip_pnl": pd.to_numeric(z.matched_roundtrip_pnl, errors="coerce").sum(),
            "residual_inventory_m5_mtm": pd.to_numeric(z.residual_inventory_m5_mtm, errors="coerce").sum(),
            "net_mtm_pnl_before_fees": pd.to_numeric(z.net_mtm_pnl_before_fees, errors="coerce").sum(),
        })
    w = pd.DataFrame(rows).sort_values("close_ts").reset_index(drop=True)
    if len(w):
        w["cum_pnl"] = w.net_mtm_pnl_before_fees.cumsum()
        w["running_peak"] = np.maximum(0.0, w.cum_pnl.cummax())
        w["drawdown"] = w.cum_pnl - w.running_peak
    return w


def _aggregate(label, cdf, fdf, exposure_hours):
    w = _window_table(cdf)
    net = float(pd.to_numeric(cdf.net_mtm_pnl_before_fees, errors="coerce").sum()) if len(cdf) else 0.0
    matched = float(pd.to_numeric(cdf.matched_roundtrip_pnl, errors="coerce").sum()) if len(cdf) else 0.0
    residual = float(pd.to_numeric(cdf.residual_inventory_m5_mtm, errors="coerce").sum()) if len(cdf) else 0.0
    gross = float(pd.to_numeric(cdf.gross_spread_capture_dollars, errors="coerce").sum()) if len(cdf) else 0.0
    qty = float(pd.to_numeric(fdf.qty, errors="coerce").sum()) if len(fdf) else 0.0
    fill_events = len(fdf)
    scale = 24.0 / exposure_hours if exposure_hours > 0 else np.nan
    row = {
        "scope": label,
        "windows": len(w),
        "contracts": len(cdf),
        "exposure_hours": exposure_hours,
        "fill_events": fill_events,
        "fill_qty": qty,
        "fill_qty_per_day": qty * scale if np.isfinite(scale) else np.nan,
        "net_pnl": net,
        "sample_rate_net_pnl_per_day": net * scale if np.isfinite(scale) else np.nan,
        "matched_roundtrip_pnl": matched,
        "sample_rate_matched_pnl_per_day": matched * scale if np.isfinite(scale) else np.nan,
        "residual_inventory_m5_mtm": residual,
        "sample_rate_residual_pnl_per_day": residual * scale if np.isfinite(scale) else np.nan,
        "gross_capture": gross,
        "net_cents_per_filled_contract": 100.0 * net / qty if qty > EPS else np.nan,
        "gross_cents_per_filled_contract": 100.0 * gross / qty if qty > EPS else np.nan,
        "break_even_fee_cents_per_filled_contract": 100.0 * net / qty if qty > EPS else np.nan,
        "positive_window_pct": 100.0 * (w.net_mtm_pnl_before_fees > 0).mean() if len(w) else np.nan,
        "median_window_pnl": w.net_mtm_pnl_before_fees.median() if len(w) else np.nan,
        "worst_window": w.net_mtm_pnl_before_fees.min() if len(w) else np.nan,
        "max_drawdown": w.drawdown.min() if len(w) else np.nan,
        "median_abs_ending_inventory_contract": pd.to_numeric(cdf.ending_inventory_yes_equiv, errors="coerce").abs().median() if len(cdf) else np.nan,
        "p95_max_abs_inventory_contract": pd.to_numeric(cdf.max_abs_inventory, errors="coerce").quantile(.95) if len(cdf) else np.nan,
        "trade_through_fill_pct": 100.0 * pd.Series(fdf.trade_through).astype(bool).mean() if len(fdf) else np.nan,
    }
    for h in MARKOUTS_S:
        tag = f"{int(h)}s"
        row[f"qw_markout_{tag}_c"] = _wavg(fdf, f"markout_{tag}_c")
        row[f"qw_post_mid_move_{tag}_c"] = _wavg(fdf, f"post_mid_move_{tag}_c")
        if len(fdf) and f"markout_{tag}_c" in fdf:
            row[f"markout_{tag}_coverage_pct"] = 100.0 * pd.to_numeric(fdf[f"markout_{tag}_c"], errors="coerce").notna().mean()
        else:
            row[f"markout_{tag}_coverage_pct"] = np.nan
    row["matched_positive"] = bool(matched > 0)
    row["target_100_day_rate_reached"] = bool(np.isfinite(row["sample_rate_net_pnl_per_day"]) and row["sample_rate_net_pnl_per_day"] >= 100.0)
    row["target_multiple"] = row["sample_rate_net_pnl_per_day"] / 100.0 if np.isfinite(row["sample_rate_net_pnl_per_day"]) else np.nan
    return row, w


def _chronology(cdf):
    w = _window_table(cdf)
    if w.empty:
        return pd.DataFrame()
    cut = len(w) // 2
    rows = []
    for label, z in (("EARLY_HALF", w.iloc[:cut]), ("LATE_HALF", w.iloc[cut:])):
        if z.empty:
            continue
        rows.append({
            "half": label,
            "windows": len(z),
            "net_pnl": z.net_mtm_pnl_before_fees.sum(),
            "matched_pnl": z.matched_roundtrip_pnl.sum(),
            "residual_pnl": z.residual_inventory_m5_mtm.sum(),
            "pnl_per_window": z.net_mtm_pnl_before_fees.mean(),
            "median_window_pnl": z.net_mtm_pnl_before_fees.median(),
            "positive_window_pct": 100.0 * (z.net_mtm_pnl_before_fees > 0).mean(),
            "worst_window": z.net_mtm_pnl_before_fees.min(),
        })
    return pd.DataFrame(rows)


def run_abcd_capacity_development(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"This development replay is hard-bound to {EXPECTED_SESSION_NAME}; got {session.name}."
        )
    if not session.exists():
        raise FileNotFoundError(session)

    manifest, plan, audit_summary, quality, eligible = _load_inputs(session)
    meta = _load_meta(session, eligible)
    close_windows = sorted({meta[t]["close_ts"] for t in eligible})
    exposure_hours = 0.25 * len(close_windows)

    print(
        f"Eligible complete contracts: {len(eligible)} across "
        f"{quality.series.nunique()} series | windows={len(close_windows)} | exposure={exposure_hours:.2f}h"
    )
    print("Loading M0-M5 aggressive trades...")
    trades = _load_research_trades(session, eligible, meta)

    sims = {
        (ticker, policy, size): Sim(policy, size, ticker, meta[ticker])
        for ticker in eligible for policy in POLICIES for size in SIZES
    }
    sim_groups = {
        ticker: [sims[(ticker, p, q)] for p in POLICIES for q in SIZES]
        for ticker in eligible
    }
    trade_i = defaultdict(int)
    pending = {ticker: [] for ticker in eligible}
    pending_counter = 0
    current_mid = defaultdict(lambda: np.nan)

    print("Streaming V5 book once: causal replay + online 5/15/30s markout resolution...")
    book_path = session / "book_top3_events.jsonl"
    with book_path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ticker = str(r.get("ticker") or "")
            if ticker not in eligible:
                continue
            t = _ts(r.get("receipt_time"))
            if not np.isfinite(t):
                continue
            t = float(t)

            # Trades that arrived before this book event see the previous book/quotes.
            arr = trades.get(ticker, [])
            j = trade_i[ticker]
            while j < len(arr) and arr[j][0] < t - EPS:
                tr_t, tr_px, tr_qty, tr_side = arr[j]
                if tr_t < meta[ticker]["m5_ts"] - EPS:
                    mid_for_fill = current_mid[ticker]
                    for sim in sim_groups[ticker]:
                        pending_counter = sim.apply_trade(
                            tr_t, tr_px, tr_qty, tr_side, mid_for_fill,
                            pending[ticker], pending_counter,
                        )
                j += 1
            trade_i[ticker] = j

            cur = _top_state(r)
            elapsed = _f(r.get("elapsed_s"))
            in_research = bool(np.isfinite(elapsed) and 0.0 <= elapsed < 300.0)

            if cur is None:
                if in_research:
                    for sim in sim_groups[ticker]:
                        sim.cancel_all("INVALID_BOOK")
                continue

            current_mid[ticker] = float(cur["mid"])
            _resolve_pending(pending[ticker], t, float(cur["mid"]))

            for sim in sim_groups[ticker]:
                sim.on_book(t, cur, in_research)

            if show and n % 1_000_000 == 0:
                print(f"  streamed {n:,} / book rows...")

    # Flush any remaining M0-M5 trades using the last known research book state.
    for ticker in sorted(eligible):
        arr = trades.get(ticker, [])
        j = trade_i[ticker]
        while j < len(arr):
            tr_t, tr_px, tr_qty, tr_side = arr[j]
            if tr_t < meta[ticker]["m5_ts"] - EPS:
                for sim in sim_groups[ticker]:
                    pending_counter = sim.apply_trade(
                        tr_t, tr_px, tr_qty, tr_side, current_mid[ticker],
                        pending[ticker], pending_counter,
                    )
            j += 1
        trade_i[ticker] = j
        for sim in sim_groups[ticker]:
            sim.cancel_all("FILE_END")

    contract_rows = []
    fill_rows = []
    count_rows = []
    for ticker in sorted(eligible, key=lambda x: (meta[x]["close_ts"], x)):
        for policy in POLICIES:
            for size in SIZES:
                sim = sims[(ticker, policy, size)]
                contract_rows.append(_contract_result(sim))
                fill_rows.extend(sim.fills)
                count_rows.extend(
                    {
                        "ticker": ticker,
                        "series": meta[ticker]["series"],
                        "policy": policy,
                        "quote_size": size,
                        "reason": k,
                        "count": v,
                    }
                    for k, v in sim.counts.items()
                )

    contracts = pd.DataFrame(contract_rows)
    fills = pd.DataFrame(fill_rows)
    counts = pd.DataFrame(count_rows)

    summary_rows = []
    windows_parts = []
    chronology_parts = []
    series_rows = []
    side_rows = []
    minute_rows = []

    scopes = {
        "ALL_9": lambda df: df,
        "NON_BTC_8": lambda df: df[df.series != BTC_SERIES],
    }

    for policy in POLICIES:
        for size in SIZES:
            c0 = contracts[(contracts.policy == policy) & (contracts.quote_size == size)].copy()
            f0 = fills[(fills.policy == policy) & (fills.quote_size == size)].copy() if len(fills) else pd.DataFrame()

            for scope, fn in scopes.items():
                cdf = fn(c0)
                fdf = fn(f0) if len(f0) else f0
                row, w = _aggregate(scope, cdf, fdf, exposure_hours)
                row.update({
                    "policy": policy,
                    "quote_size": size,
                    "capacity_scenario": "REFERENCE_Q1" if abs(size - 1.0) < EPS else "COUNTERFACTUAL",
                })
                summary_rows.append(row)
                if len(w):
                    w["policy"] = policy
                    w["quote_size"] = size
                    w["scope"] = scope
                    windows_parts.append(w)

            # Chronology and granular diagnostics use ALL_9, with NON_BTC summary already above.
            ch = _chronology(c0)
            if len(ch):
                ch["policy"] = policy
                ch["quote_size"] = size
                chronology_parts.append(ch)

            for series, cdf in c0.groupby("series", sort=True):
                fdf = f0[f0.series == series] if len(f0) else f0
                row, _ = _aggregate(str(series), cdf, fdf, exposure_hours)
                row.update({"policy": policy, "quote_size": size, "series": series})
                series_rows.append(row)

            if len(f0):
                for side, z in f0.groupby("side", sort=True):
                    side_rows.append({
                        "policy": policy,
                        "quote_size": size,
                        "side": side,
                        "fill_events": len(z),
                        "fill_qty": z.qty.sum(),
                        "gross_edge_c": _wavg(z, "gross_edge_at_fill_c"),
                        "markout_5s_c": _wavg(z, "markout_5s_c"),
                        "markout_15s_c": _wavg(z, "markout_15s_c"),
                        "markout_30s_c": _wavg(z, "markout_30s_c"),
                        "post_mid_move_30s_c": _wavg(z, "post_mid_move_30s_c"),
                        "trade_through_fill_pct": 100.0 * z.trade_through.astype(bool).mean(),
                    })
                for minute, z in f0.groupby("entry_minute", sort=True):
                    minute_rows.append({
                        "policy": policy,
                        "quote_size": size,
                        "entry_minute": minute,
                        "fill_events": len(z),
                        "fill_qty": z.qty.sum(),
                        "gross_edge_c": _wavg(z, "gross_edge_at_fill_c"),
                        "markout_5s_c": _wavg(z, "markout_5s_c"),
                        "markout_15s_c": _wavg(z, "markout_15s_c"),
                        "markout_30s_c": _wavg(z, "markout_30s_c"),
                    })

    summary = pd.DataFrame(summary_rows)
    windows = pd.concat(windows_parts, ignore_index=True) if windows_parts else pd.DataFrame()
    chronology = pd.concat(chronology_parts, ignore_index=True) if chronology_parts else pd.DataFrame()
    by_series = pd.DataFrame(series_rows)
    by_side = pd.DataFrame(side_rows)
    by_minute = pd.DataFrame(minute_rows)

    # Compact target screen. Counterfactual rows are never labeled validated capacity.
    target = summary[summary.scope == "ALL_9"].copy()
    target["capacity_status"] = np.where(
        target.quote_size == REFERENCE_SIZE,
        np.where(target.target_100_day_rate_reached & target.matched_positive, "Q1_RATE_AND_MATCHED_PASS", "Q1_FAIL_TARGET_OR_MATCHED"),
        np.where(target.target_100_day_rate_reached & target.matched_positive, "COUNTERFACTUAL_TARGET_PLAUSIBLE", "COUNTERFACTUAL_BELOW_TARGET_OR_MATCHED_FAIL"),
    )
    target["live_proof"] = False
    target["capacity_caveat"] = np.where(
        target.quote_size == REFERENCE_SIZE,
        "Q1 historical execution reference; still counterfactual to our presence",
        "Q>1 historical-flow scenario; larger quote can alter queue/fill dynamics; not linear scaling proof",
    )

    max_recon_error = pd.to_numeric(contracts.reconstruction_error, errors="coerce").abs().max()

    if output_dir is None:
        output_dir = C.PROJECT_ROOT / "results" / "kalshi_mm_event_abcd_capacity_dev" / session.name
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    quality.to_csv(out / "eligible_contracts.csv", index=False)
    contracts.to_csv(out / "contract_results.csv", index=False)
    fills.to_csv(out / "fills.csv", index=False)
    counts.to_csv(out / "policy_counts.csv", index=False)
    summary.to_csv(out / "summary_all_scopes.csv", index=False)
    target.to_csv(out / "target_capacity_screen.csv", index=False)
    windows.to_csv(out / "window_results.csv", index=False)
    chronology.to_csv(out / "chronology.csv", index=False)
    by_series.to_csv(out / "by_series.csv", index=False)
    by_side.to_csv(out / "by_side.csv", index=False)
    by_minute.to_csv(out / "by_entry_minute.csv", index=False)

    spec = {
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "hard_bound_session_name": EXPECTED_SESSION_NAME,
        "audit_verdict_required": EXPECTED_AUDIT_VERDICT,
        "eligible_contract_rule": "full_m0_m5_plus_30_boundary == True",
        "eligible_contracts": len(eligible),
        "windows": len(close_windows),
        "analyzed_exposure_hours": exposure_hours,
        "policies": {
            "A_L3": "same-side L3 depth > opposite-side L3 depth",
            "B_L1_L3": "same-side L1 and L3 depth both > opposite side",
            "C_L3_SPREAD2": "A plus natural spread >= 2c",
            "D_L1_L3_SPREAD2": "B plus natural spread >= 2c",
        },
        "quote_sizes": list(SIZES),
        "reference_size": REFERENCE_SIZE,
        "capacity_scenarios": [2, 5, 10],
        "economic_target_usd_per_day": 100.0,
        "execution": "public BBO; back of displayed L1 queue; exact-price trades burn queue; trade-through fills; no cancellation-ahead credit; any fill cancels residual",
        "fees": 0.0,
        "threshold_sweep": False,
        "asset_performance_filter": False,
        "pnl_day_definition": "observed PnL scaled by 24 / analyzed complete-window exposure hours",
        "capacity_warning": "Q2/Q5/Q10 replay historical public flow as exogenous and are counterfactual, not proven live scalable capacity",
        "max_contract_fifo_reconstruction_error": max_recon_error,
    }
    (out / "study_spec.json").write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")

    if show:
        print("\n" + "=" * 170)
        print("V5 A/B/C/D EVENT-TIME MM CAPACITY DEVELOPMENT — PRE-REGISTERED FAMILY ONLY")
        print("=" * 170)
        print(
            f"session={session.name} | complete contracts={len(eligible)} | series={quality.series.nunique()} | "
            f"windows={len(close_windows)} | exposure={exposure_hours:.2f}h"
        )
        print("Q1 = execution reference | Q2/Q5/Q10 = COUNTERFACTUAL historical-flow capacity scenarios")
        print("Economic target: >= $100/day plausible capacity | fees=0 | no threshold sweep")
        print(f"FIFO reconstruction max abs error: ${max_recon_error:.12f}")

        cols = [
            "policy", "quote_size", "capacity_scenario", "scope", "fill_qty",
            "sample_rate_net_pnl_per_day", "sample_rate_matched_pnl_per_day",
            "sample_rate_residual_pnl_per_day", "net_cents_per_filled_contract",
            "break_even_fee_cents_per_filled_contract", "qw_markout_5s_c",
            "qw_markout_15s_c", "qw_markout_30s_c", "worst_window", "max_drawdown",
            "target_100_day_rate_reached", "matched_positive",
        ]
        print("\nECONOMIC SUMMARY — ALL 9 + NON-BTC 8")
        print(summary[cols].round(4).to_string(index=False))

        print("\n$100/DAY CAPACITY SCREEN — ALL 9")
        tcols = [
            "policy", "quote_size", "capacity_scenario", "sample_rate_net_pnl_per_day",
            "sample_rate_matched_pnl_per_day", "sample_rate_residual_pnl_per_day",
            "fill_qty_per_day", "net_cents_per_filled_contract", "trade_through_fill_pct",
            "qw_markout_30s_c", "capacity_status",
        ]
        print(target[tcols].round(4).to_string(index=False))

        print("\nQ1 CHRONOLOGY — ALL 9")
        q1c = chronology[chronology.quote_size == REFERENCE_SIZE] if len(chronology) else chronology
        print(q1c.round(4).to_string(index=False) if len(q1c) else "none")

        print("\nQ1 BY SERIES")
        q1s = by_series[by_series.quote_size == REFERENCE_SIZE] if len(by_series) else by_series
        scols = [
            "policy", "series", "fill_qty", "net_pnl", "sample_rate_net_pnl_per_day",
            "matched_roundtrip_pnl", "residual_inventory_m5_mtm",
            "net_cents_per_filled_contract", "qw_markout_30s_c",
        ]
        print(q1s[scols].round(4).to_string(index=False) if len(q1s) else "none")

        print("\nQ1 SIDE ECONOMICS")
        q1side = by_side[by_side.quote_size == REFERENCE_SIZE] if len(by_side) else by_side
        print(q1side.round(4).to_string(index=False) if len(q1side) else "none")

        print("\nQ1 ENTRY-MINUTE FILL ECONOMICS")
        q1m = by_minute[by_minute.quote_size == REFERENCE_SIZE] if len(by_minute) else by_minute
        print(q1m.round(4).to_string(index=False) if len(q1m) else "none")

        print("\nINTERPRETATION GUARDRAILS")
        print("  - This is V5 DEVELOPMENT, not OOS validation.")
        print("  - Only the four pre-registered A/B/C/D rules were evaluated; no parameter sweep occurred.")
        print("  - $/day is a sample-rate extrapolation from complete 15-minute windows, not a guaranteed calendar-day return.")
        print("  - Q2/Q5/Q10 use historical public flow as exogenous and are explicitly counterfactual capacity scenarios.")
        print("  - Do not select asset-specific, minute-specific, side-specific, or new spread thresholds after reading this output.")
        print("  - A candidate only deserves freezing if matched economics are positive and a credible path toward $100/day exists without residual inventory carrying the result.")
        print(f"\nOUTPUTS: {out}")
        print("=" * 170)

    return {
        "output_dir": out,
        "summary": summary,
        "target_screen": target,
        "chronology": chronology,
        "by_series": by_series,
        "by_side": by_side,
        "by_entry_minute": by_minute,
        "windows": windows,
        "contracts": contracts,
        "fills": fills,
        "counts": counts,
        "spec": spec,
    }
