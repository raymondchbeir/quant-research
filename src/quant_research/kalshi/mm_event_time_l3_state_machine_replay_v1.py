from __future__ import annotations

"""Event-time L3 support + 500ms replenishment market-making replay.

DEVELOPMENT ONLY. Hard-bound to the already-explored V4 session
20260815_043130. This module does not read any future validation recording.

Scientific purpose
------------------
Translate the pre-specified event-time toxicity finding into an exact passive
quoting mechanism without tuning a numeric threshold on the development data.

Quoted universe
---------------
- Complete M0-M5 contracts from the prior V4 quality audit.
- KXBTC15M is excluded as a quoted market because the prior pre-PnL forensic
  audit found all reconstructed locked/crossed book states were isolated to
  BTC. This is a data-quality exclusion, not a performance filter.
- The remaining eight crypto series are eligible.

Execution model
---------------
- M0 <= elapsed < M5 only.
- Q1 passive order at the current public YES best bid / best ask.
- Join at the back of displayed L1 size when an order is opened or repriced.
- Exact-price opposing aggressive trades consume queue ahead first.
- Trade-through fills the remaining order.
- No cancellation-ahead credit. Book-size decreases do not move us forward.
- Any fill cancels residual same-side quantity.
- Public BBO changes cancel/reprice quotes causally at event time.
- No fees.
- Remaining inventory is marked at the final valid midpoint before M5.

Fixed policies
--------------
1. BASELINE_BBO_Q1
   Quote both sides whenever the event-time book is valid.

2. L3_SUPPORT_ONLY_Q1
   BID may quote iff L3 bid depth > L3 ask depth.
   ASK may quote iff L3 ask depth > L3 bid depth.
   Natural zero imbalance boundary; no tuned threshold.

3. L3_SUPPORT_REPLENISH_STATE_Q1  [primary candidate]
   Side state uses support plus same-side L3 depth change over the prior 500ms:

       STRONG:
           supported AND same-side depth3 change > 0
           -> may OPEN and maintain a quote.

       SUPPORTED:
           supported but not replenishing
           -> may MAINTAIN an already-live quote, but may not open/re-enter.

       WEAK:
           not supported
           -> cancel / remain out.

   Because every fill cancels the residual same-side quote, re-entry after a
   fill requires a later STRONG state. There is no tuned cooldown duration.

This is a counterfactual historical replay. Even Q1 can alter the public book,
so live maker validation would still be required before deployment.
"""

import bisect
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C

STUDY_VERSION = "MM_EVENT_TIME_L3_STATE_MACHINE_REPLAY_V1"
EXPECTED_SESSION_NAME = "20260815_043130"
BTC_SERIES = "KXBTC15M"
QUOTE_QTY = 1.0
REPLENISH_LOOKBACK_S = 0.500
EPS = 1e-12
MARKOUTS_S = (5.0, 15.0, 30.0)

POLICIES = (
    "BASELINE_BBO_Q1",
    "L3_SUPPORT_ONLY_Q1",
    "L3_SUPPORT_REPLENISH_STATE_Q1",
)


def _ts(x):
    if x is None:
        return np.nan
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except Exception:
        return np.nan


def _iso(ts):
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
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


def _top_state(row):
    bids = row.get("bid_levels") or []
    asks = row.get("ask_levels") or []
    if not bool(row.get("valid_bbo")) or not bids or not asks:
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


def _past_state(hist, target_t):
    for e in reversed(hist):
        if float(e["t"]) <= float(target_t) + EPS:
            return e
    return None


def _side_micro_state(side: str, cur: dict, hist: deque, now_t: float):
    past = _past_state(hist, float(now_t) - REPLENISH_LOOKBACK_S)
    if side == "BID":
        support = cur["bid_depth3"] > cur["ask_depth3"] + EPS
        change = (
            cur["bid_depth3"] - past["bid_depth3"]
            if past is not None else np.nan
        )
    else:
        support = cur["ask_depth3"] > cur["bid_depth3"] + EPS
        change = (
            cur["ask_depth3"] - past["ask_depth3"]
            if past is not None else np.nan
        )

    replenishing = bool(np.isfinite(change) and change > 0.0 + EPS)
    if support and replenishing:
        state = "STRONG"
    elif support:
        state = "SUPPORTED"
    else:
        state = "WEAK"
    return state, bool(support), replenishing, change


def _load_quality(session: Path):
    audit_path = (
        C.PROJECT_ROOT
        / "results"
        / "kalshi_mm_event_m0_m5_v4_audit"
        / session.name
        / "contract_event_time_quality.csv"
    )
    if not audit_path.exists():
        raise FileNotFoundError(
            f"Prior V4 audit output is required before replay: {audit_path}"
        )
    q = pd.read_csv(audit_path)
    need = {"ticker", "series", "full_boundary_capture"}
    missing = need - set(q.columns)
    if missing:
        raise RuntimeError(f"Audit CSV missing columns: {sorted(missing)}")
    ok = q["full_boundary_capture"].astype(str).str.lower().isin({"true", "1"})
    q = q[ok & (q["series"].astype(str) != BTC_SERIES)].copy()
    return q


def _load_meta(session: Path):
    out = {}
    p = session / "market_metadata.jsonl"
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ticker = str(r.get("ticker") or "")
            if not ticker:
                continue
            close = _ts(r.get("close_time"))
            if not np.isfinite(close):
                continue
            out[ticker] = {
                "ticker": ticker,
                "series": str(r.get("series_ticker") or ""),
                "close_ts": float(close),
            }
    return out


def _load_trades(session: Path, eligible: set[str]):
    rows = []
    p = session / "trades_event_time.jsonl"
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
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
            if not np.isfinite(t) or not np.isfinite(px) or not np.isfinite(qty) or qty <= 0:
                continue
            if side not in {"bid", "ask"}:
                continue
            rows.append((ticker, float(t), float(px), float(qty), side))
    out = {}
    for ticker, z in pd.DataFrame(
        rows, columns=["ticker", "t", "price", "qty", "taker_book_side"]
    ).groupby("ticker", sort=False):
        z = z.sort_values("t").reset_index(drop=True)
        out[str(ticker)] = z
    return out


def _load_future_mid_series(session: Path, eligible: set[str]):
    # The persisted top3 stream itself is the causal/replay book source. We make
    # one compact pass containing only valid midpoints for post-fill markouts.
    mids = defaultdict(list)
    p = session / "book_top3_events.jsonl"
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ticker = str(r.get("ticker") or "")
            if ticker not in eligible:
                continue
            s = _top_state(r)
            if s is None:
                continue
            t = _ts(r.get("receipt_time"))
            if np.isfinite(t):
                mids[ticker].append((float(t), float(s["mid"])))
            if i % 1_000_000 == 0:
                print(f"  markout pass: streamed {i:,} book rows...")
    out = {}
    for ticker, rows in mids.items():
        if not rows:
            continue
        rows.sort()
        out[ticker] = (
            np.asarray([x[0] for x in rows], dtype=float),
            np.asarray([x[1] for x in rows], dtype=float),
        )
    return out


def _future_mid(mid_pack, target_t, max_age_s=2.0):
    if mid_pack is None:
        return np.nan
    tt, mm = mid_pack
    j = int(np.searchsorted(tt, float(target_t), side="left"))
    if j >= len(tt):
        return np.nan
    if float(tt[j]) - float(target_t) > max_age_s + EPS:
        return np.nan
    return float(mm[j])


class PolicySim:
    def __init__(self, name, ticker, meta, trades, future_mids):
        self.name = name
        self.ticker = ticker
        self.meta = meta
        self.trades = trades
        self.future_mids = future_mids
        self.trade_i = 0
        self.active = {"BID": None, "ASK": None}
        self.fills = []
        self.episodes = []
        self.counts = Counter()
        self.inventory = 0.0
        self.cash = 0.0
        self.max_abs_inventory = 0.0
        self.last_mid = np.nan
        self.episode_no = 0

    def _cancel(self, side, t, reason):
        ep = self.active[side]
        if ep is None:
            return
        ep["end_ts"] = float(t)
        ep["end_time"] = _iso(t)
        ep["end_reason"] = str(reason)
        self.active[side] = None
        self.counts[f"{side}_CANCEL_{reason}"] += 1

    def _open(self, side, t, cur, micro_state, support, replenishing, depth_change):
        px = cur["bid"] if side == "BID" else cur["ask"]
        qahead = cur["bid_q1"] if side == "BID" else cur["ask_q1"]
        self.episode_no += 1
        ep = {
            "policy": self.name,
            "ticker": self.ticker,
            "series": self.meta["series"],
            "close_ts": self.meta["close_ts"],
            "episode_id": f"{self.ticker}:{self.name}:{side}:{self.episode_no}",
            "side": side,
            "join_ts": float(t),
            "join_time": _iso(t),
            "price": float(px),
            "quote_qty": QUOTE_QTY,
            "remaining_qty": QUOTE_QTY,
            "queue_ahead_initial": float(qahead),
            "queue_ahead": float(qahead),
            "mid_at_join": cur["mid"],
            "spread_c_at_join": cur["spread_c"],
            "l3_bid_depth_at_join": cur["bid_depth3"],
            "l3_ask_depth_at_join": cur["ask_depth3"],
            "l3_imbalance_at_join": (
                (cur["bid_depth3"] - cur["ask_depth3"])
                / (cur["bid_depth3"] + cur["ask_depth3"])
                if cur["bid_depth3"] + cur["ask_depth3"] > EPS else 0.0
            ),
            "micro_state_at_join": micro_state,
            "support_at_join": bool(support),
            "replenishing_at_join": bool(replenishing),
            "same_side_depth3_change_500ms_at_join": depth_change,
            "inventory_at_join": self.inventory,
            "fill_qty": 0.0,
            "end_ts": np.nan,
            "end_time": None,
            "end_reason": None,
        }
        self.episodes.append(ep)
        self.active[side] = ep
        self.counts[f"{side}_OPEN"] += 1
        self.counts[f"{side}_OPEN_STATE_{micro_state}"] += 1

    def _fill(self, side, tr_t, trade_px, trade_qty, mid_at_fill, trade_through):
        ep = self.active[side]
        if ep is None:
            return
        qty = min(float(ep["remaining_qty"]), float(trade_qty))
        if qty <= EPS:
            return
        sign = 1.0 if side == "BID" else -1.0
        inv_before = self.inventory
        self.inventory += sign * qty
        self.cash += (-float(ep["price"]) * qty) if side == "BID" else (float(ep["price"]) * qty)
        self.max_abs_inventory = max(self.max_abs_inventory, abs(self.inventory))
        ep["fill_qty"] += qty
        ep["remaining_qty"] -= qty

        row = {
            "policy": self.name,
            "ticker": self.ticker,
            "series": self.meta["series"],
            "close_ts": self.meta["close_ts"],
            "episode_id": ep["episode_id"],
            "side": side,
            "fill_ts": float(tr_t),
            "fill_time": _iso(tr_t),
            "qty": qty,
            "price": float(ep["price"]),
            "mid_at_fill": float(mid_at_fill) if np.isfinite(mid_at_fill) else np.nan,
            "gross_edge_at_fill_c": (
                sign * (float(mid_at_fill) - float(ep["price"])) * 100.0
                if np.isfinite(mid_at_fill) else np.nan
            ),
            "trade_through": bool(trade_through),
            "historical_trade_price": float(trade_px),
            "historical_trade_qty": float(trade_qty),
            "inventory_before_fill": inv_before,
            "inventory_after_fill": self.inventory,
            "micro_state_at_join": ep["micro_state_at_join"],
            "l3_imbalance_at_join": ep["l3_imbalance_at_join"],
            "same_side_depth3_change_500ms_at_join": ep["same_side_depth3_change_500ms_at_join"],
        }
        for h in MARKOUTS_S:
            fm = _future_mid(self.future_mids, float(tr_t) + h)
            tag = f"{int(h)}s"
            row[f"future_mid_{tag}"] = fm
            row[f"markout_{tag}_c"] = (
                sign * (fm - float(ep["price"])) * 100.0
                if np.isfinite(fm) else np.nan
            )
            row[f"post_mid_move_{tag}_c"] = (
                sign * (fm - float(mid_at_fill)) * 100.0
                if np.isfinite(fm) and np.isfinite(mid_at_fill) else np.nan
            )
        self.fills.append(row)
        self.counts[f"{side}_FILL_EVENT"] += 1

        # Fixed conservative behavior: any fill removes the residual order.
        self._cancel(side, tr_t, "FILL_CANCEL_RESIDUAL")

    def process_trades_before(self, t, current_mid):
        if self.trades is None or self.trades.empty:
            return
        while self.trade_i < len(self.trades):
            tr = self.trades.iloc[self.trade_i]
            tr_t = float(tr.t)
            if tr_t >= float(t) - EPS:
                break
            trade_px = float(tr.price)
            trade_qty = float(tr.qty)
            taker = str(tr.taker_book_side)

            # Passive BID is hit by an aggressive seller (taker book side ask).
            ep = self.active["BID"]
            if ep is not None and taker == "ask":
                qpx = float(ep["price"])
                if trade_px < qpx - EPS:
                    self._fill("BID", tr_t, trade_px, trade_qty, current_mid, True)
                elif abs(trade_px - qpx) <= EPS:
                    ep = self.active["BID"]
                    if ep is not None:
                        ahead = float(ep["queue_ahead"])
                        used = min(ahead, trade_qty)
                        ep["queue_ahead"] = ahead - used
                        residual = trade_qty - used
                        if residual > EPS:
                            self._fill("BID", tr_t, trade_px, residual, current_mid, False)

            # Passive ASK is lifted by an aggressive buyer (taker book side bid).
            ep = self.active["ASK"]
            if ep is not None and taker == "bid":
                qpx = float(ep["price"])
                if trade_px > qpx + EPS:
                    self._fill("ASK", tr_t, trade_px, trade_qty, current_mid, True)
                elif abs(trade_px - qpx) <= EPS:
                    ep = self.active["ASK"]
                    if ep is not None:
                        ahead = float(ep["queue_ahead"])
                        used = min(ahead, trade_qty)
                        ep["queue_ahead"] = ahead - used
                        residual = trade_qty - used
                        if residual > EPS:
                            self._fill("ASK", tr_t, trade_px, residual, current_mid, False)

            self.trade_i += 1

    def on_book(self, t, cur, hist):
        self.last_mid = float(cur["mid"])
        for side in ("BID", "ASK"):
            state, support, replenishing, change = _side_micro_state(side, cur, hist, t)
            desired_px = cur["bid"] if side == "BID" else cur["ask"]
            ep = self.active[side]

            # Public BBO changed: our old order is no longer the current-BBO quote.
            if ep is not None and abs(float(ep["price"]) - float(desired_px)) > EPS:
                self._cancel(side, t, "BBO_REPRICE")
                ep = None

            if self.name == "BASELINE_BBO_Q1":
                allow_maintain = True
                allow_open = True
            elif self.name == "L3_SUPPORT_ONLY_Q1":
                allow_maintain = support
                allow_open = support
            elif self.name == "L3_SUPPORT_REPLENISH_STATE_Q1":
                allow_maintain = state in {"STRONG", "SUPPORTED"}
                allow_open = state == "STRONG"
            else:
                raise RuntimeError(self.name)

            ep = self.active[side]
            if ep is not None and not allow_maintain:
                self._cancel(side, t, f"STATE_{state}")
                ep = None

            if ep is None and allow_open:
                self._open(side, t, cur, state, support, replenishing, change)
            else:
                self.counts[f"{side}_OBS_STATE_{state}"] += 1

    def finish(self, end_t):
        # Process trades up to M5 using the last causal book midpoint.
        self.process_trades_before(end_t, self.last_mid)
        self._cancel("BID", end_t, "M5_END")
        self._cancel("ASK", end_t, "M5_END")
        if not np.isfinite(self.last_mid):
            return None

        net = self.cash + self.inventory * self.last_mid
        gross = 0.0
        for f in self.fills:
            ge = _f(f.get("gross_edge_at_fill_c"))
            if np.isfinite(ge):
                gross += ge / 100.0 * float(f["qty"])

        bid_qty = sum(float(f["qty"]) for f in self.fills if f["side"] == "BID")
        ask_qty = sum(float(f["qty"]) for f in self.fills if f["side"] == "ASK")
        return {
            "policy": self.name,
            "ticker": self.ticker,
            "series": self.meta["series"],
            "close_ts": self.meta["close_ts"],
            "close_time": _iso(self.meta["close_ts"]),
            "bid_fill_qty": bid_qty,
            "ask_fill_qty": ask_qty,
            "fill_qty": bid_qty + ask_qty,
            "fill_events": len(self.fills),
            "ending_inventory_yes_equiv": self.inventory,
            "max_abs_inventory": self.max_abs_inventory,
            "final_mid_m5": self.last_mid,
            "cash": self.cash,
            "gross_spread_capture_dollars": gross,
            "adverse_selection_to_m5_dollars": net - gross,
            "net_mtm_pnl_before_fees": net,
        }


def _wavg(df, col, weight="qty"):
    if df.empty or col not in df.columns:
        return np.nan
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    w = pd.to_numeric(df[weight], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def _window_summary(cdf):
    if cdf.empty:
        return pd.DataFrame()
    rows = []
    for close, z in cdf.groupby("close_ts", sort=True):
        rows.append({
            "close_ts": float(close),
            "close_time": _iso(close),
            "contracts": len(z),
            "fill_qty": pd.to_numeric(z.fill_qty, errors="coerce").sum(),
            "net_mtm_pnl_before_fees": pd.to_numeric(z.net_mtm_pnl_before_fees, errors="coerce").sum(),
            "gross_capture": pd.to_numeric(z.gross_spread_capture_dollars, errors="coerce").sum(),
            "adverse_selection_to_m5": pd.to_numeric(z.adverse_selection_to_m5_dollars, errors="coerce").sum(),
            "ending_abs_inventory_sum": pd.to_numeric(z.ending_inventory_yes_equiv, errors="coerce").abs().sum(),
        })
    out = pd.DataFrame(rows).sort_values("close_ts").reset_index(drop=True)
    out["cum_pnl"] = out.net_mtm_pnl_before_fees.cumsum()
    out["running_peak"] = out.cum_pnl.cummax()
    out["drawdown"] = out.cum_pnl - out.running_peak
    return out


def _policy_summary(name, cdf, fdf, wdf):
    pnl = pd.to_numeric(cdf.net_mtm_pnl_before_fees, errors="coerce").sum() if len(cdf) else 0.0
    qty = pd.to_numeric(fdf.qty, errors="coerce").sum() if len(fdf) else 0.0
    gross = pd.to_numeric(cdf.gross_spread_capture_dollars, errors="coerce").sum() if len(cdf) else 0.0
    adv = pd.to_numeric(cdf.adverse_selection_to_m5_dollars, errors="coerce").sum() if len(cdf) else 0.0
    row = {
        "policy": name,
        "windows": len(wdf),
        "contracts": len(cdf),
        "fill_events": len(fdf),
        "fill_qty": qty,
        "net_pnl": pnl,
        "pnl_per_window": pnl / len(wdf) if len(wdf) else np.nan,
        "gross_capture": gross,
        "adverse_selection_to_m5": adv,
        "worst_window": pd.to_numeric(wdf.net_mtm_pnl_before_fees, errors="coerce").min() if len(wdf) else np.nan,
        "max_drawdown": pd.to_numeric(wdf.drawdown, errors="coerce").min() if len(wdf) else np.nan,
        "median_abs_ending_inventory_contract": pd.to_numeric(cdf.ending_inventory_yes_equiv, errors="coerce").abs().median() if len(cdf) else np.nan,
        "p95_max_abs_inventory_contract": pd.to_numeric(cdf.max_abs_inventory, errors="coerce").quantile(.95) if len(cdf) else np.nan,
    }
    for h in MARKOUTS_S:
        tag = f"{int(h)}s"
        row[f"qw_markout_{tag}_c"] = _wavg(fdf, f"markout_{tag}_c")
        row[f"qw_post_mid_move_{tag}_c"] = _wavg(fdf, f"post_mid_move_{tag}_c")
    return row


def _chronology(windows_all):
    rows = []
    if windows_all.empty:
        return pd.DataFrame()
    for policy, z in windows_all.groupby("policy", sort=False):
        z = z.sort_values("close_ts").reset_index(drop=True)
        cut = len(z) // 2
        for label, q in (("EARLY_HALF", z.iloc[:cut]), ("LATE_HALF", z.iloc[cut:])):
            if q.empty:
                continue
            pnl = q.net_mtm_pnl_before_fees.sum()
            rows.append({
                "policy": policy,
                "half": label,
                "windows": len(q),
                "net_pnl": pnl,
                "pnl_per_window": pnl / len(q),
                "median_window_pnl": q.net_mtm_pnl_before_fees.median(),
                "positive_window_pct": 100.0 * (q.net_mtm_pnl_before_fees > 0).mean(),
                "worst_window": q.net_mtm_pnl_before_fees.min(),
            })
    return pd.DataFrame(rows)


def run_event_time_l3_state_machine_replay(session_dir, output_dir=None, *, show=True):
    session = Path(session_dir).resolve()
    if session.name != EXPECTED_SESSION_NAME:
        raise RuntimeError(
            f"Development replay is hard-bound to {EXPECTED_SESSION_NAME}; got {session.name}."
        )
    if not session.exists():
        raise FileNotFoundError(session)

    quality = _load_quality(session)
    meta_all = _load_meta(session)
    eligible = set(quality.ticker.astype(str)) & set(meta_all)
    meta = {t: meta_all[t] for t in eligible}
    if not eligible:
        raise RuntimeError("No complete non-BTC contracts are eligible")

    print(f"Eligible contracts: {len(eligible)} across {quality[quality.ticker.isin(eligible)].series.nunique()} non-BTC series")
    print("Loading event-time aggressive trades...")
    trades = _load_trades(session, eligible)

    print("Building valid midpoint series for markouts (first 3GB streaming pass)...")
    future_mids = _load_future_mid_series(session, eligible)

    sims = {
        (p, t): PolicySim(p, t, meta[t], trades.get(t), future_mids.get(t))
        for p in POLICIES for t in eligible
    }
    hists = {t: deque() for t in eligible}
    current_mid = {t: np.nan for t in eligible}
    seen_book = set()

    print("Running exact event-time policy replay (second 3GB streaming pass)...")
    book_path = session / "book_top3_events.jsonl"
    with book_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
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

            # Trades preceding this book event see only the previously known book.
            for p in POLICIES:
                sims[(p, ticker)].process_trades_before(t, current_mid[ticker])

            cur = _top_state(r)
            if cur is None:
                # Invalid book means no passive quoting until a valid event arrives.
                for p in POLICIES:
                    sims[(p, ticker)]._cancel("BID", t, "INVALID_BOOK")
                    sims[(p, ticker)]._cancel("ASK", t, "INVALID_BOOK")
                current_mid[ticker] = np.nan
                hists[ticker].clear()
                continue

            current_mid[ticker] = cur["mid"]
            hist = hists[ticker]
            # Keep a little more than 500ms of state history.
            while hist and float(hist[0]["t"]) < float(t) - REPLENISH_LOOKBACK_S - 0.05:
                hist.popleft()
            hist.append({"t": float(t), **cur})
            seen_book.add(ticker)

            for p in POLICIES:
                sims[(p, ticker)].on_book(t, cur, hist)

            if i % 1_000_000 == 0:
                print(f"  replay pass: streamed {i:,} book rows...")

    contracts_all = []
    fills_all = []
    episodes_all = []
    counts_all = []

    for p in POLICIES:
        for ticker in sorted(eligible, key=lambda x: (meta[x]["close_ts"], x)):
            sim = sims[(p, ticker)]
            end_t = meta[ticker]["close_ts"] - 600.0  # M5
            c = sim.finish(end_t)
            if c is not None:
                contracts_all.append(c)
            fills_all.extend(sim.fills)
            episodes_all.extend(sim.episodes)
            counts_all.extend(
                {"policy": p, "ticker": ticker, "reason": k, "count": v}
                for k, v in sim.counts.items()
            )

    contracts = pd.DataFrame(contracts_all)
    fills = pd.DataFrame(fills_all)
    episodes = pd.DataFrame(episodes_all)
    counts = pd.DataFrame(counts_all)

    policy_rows = []
    windows_list = []
    for p in POLICIES:
        cdf = contracts[contracts.policy == p].copy() if len(contracts) else pd.DataFrame()
        fdf = fills[fills.policy == p].copy() if len(fills) else pd.DataFrame()
        wdf = _window_summary(cdf)
        if len(wdf):
            wdf["policy"] = p
            windows_list.append(wdf)
        policy_rows.append(_policy_summary(p, cdf, fdf, wdf))

    policy_summary = pd.DataFrame(policy_rows)
    windows_all = pd.concat(windows_list, ignore_index=True) if windows_list else pd.DataFrame()
    chronology = _chronology(windows_all)

    # Primary incremental comparison to baseline.
    if len(policy_summary):
        base = policy_summary[policy_summary.policy == "BASELINE_BBO_Q1"].iloc[0]
        policy_summary["delta_net_vs_baseline"] = policy_summary.net_pnl - float(base.net_pnl)
        policy_summary["delta_adverse_selection_vs_baseline"] = (
            policy_summary.adverse_selection_to_m5 - float(base.adverse_selection_to_m5)
        )
        for h in MARKOUTS_S:
            tag = f"{int(h)}s"
            col = f"qw_markout_{tag}_c"
            policy_summary[f"delta_{col}_vs_baseline"] = policy_summary[col] - float(base[col])

    if output_dir is None:
        output_dir = (
            C.PROJECT_ROOT
            / "results"
            / "kalshi_mm_event_l3_state_machine_replay"
            / session.name
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    quality.to_csv(out / "eligible_contract_quality.csv", index=False)
    contracts.to_csv(out / "contract_results.csv", index=False)
    fills.to_csv(out / "fills.csv", index=False)
    episodes.to_csv(out / "quote_episodes.csv", index=False)
    counts.to_csv(out / "policy_counts.csv", index=False)
    policy_summary.to_csv(out / "policy_summary.csv", index=False)
    windows_all.to_csv(out / "window_results.csv", index=False)
    chronology.to_csv(out / "chronology.csv", index=False)

    config = {
        "study_version": STUDY_VERSION,
        "development_session": str(session),
        "hard_bound_session_name": EXPECTED_SESSION_NAME,
        "quoted_universe": "complete M0-M5 non-BTC V4 contracts",
        "btc_quote_exclusion_reason": "pre-PnL reconstructed-book data-quality failure",
        "quote_qty": QUOTE_QTY,
        "quote_location": "current public YES BBO",
        "queue_model": "join back of displayed L1; exact-price aggressive trades consume queue; no cancellation-ahead credit; trade-through fills",
        "replenish_lookback_ms": int(REPLENISH_LOOKBACK_S * 1000),
        "thresholds": {
            "support": "same-side L3 depth > opposite-side L3 depth (natural zero imbalance)",
            "replenishing": "same-side L3 depth change over 500ms > 0 (natural zero change)",
        },
        "primary_policy": "L3_SUPPORT_REPLENISH_STATE_Q1",
        "fees": 0.0,
        "threshold_sweep": False,
        "asset_performance_filter": False,
        "limitations": [
            "historical public flow is treated as exogenous to our hypothetical Q1 quote",
            "aggregated L1 FIFO queue approximation; no order-level queue position",
            "no cancellation-ahead credit",
            "BTC reconstructed depth excluded for prior data-quality reasons",
            "development sample already explored; any promising result requires fresh V5 validation",
        ],
    }
    (out / "study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if show:
        print("\n" + "=" * 150)
        print("EVENT-TIME L3 SUPPORT + 500MS REPLENISHMENT MM REPLAY — DEVELOPMENT ONLY")
        print("=" * 150)
        print(f"session={session.name} | eligible contracts={len(eligible)} | windows={quality[quality.ticker.isin(eligible)].shape[0] and quality[quality.ticker.isin(eligible)].ticker.nunique()}")
        print("Quoted universe: complete non-BTC M0-M5 contracts only; BTC excluded for pre-PnL depth corruption.")
        print("Q1 at public BBO | conservative FIFO | no cancellation-ahead credit | fees=0")
        print("\nPOLICY SUMMARY")
        print(policy_summary.round(4).to_string(index=False))
        print("\nCHRONOLOGY")
        print(chronology.round(4).to_string(index=False))

        if len(fills):
            print("\nFILL SIDE SUMMARY")
            side_rows = []
            for (p, side), z in fills.groupby(["policy", "side"], sort=False):
                side_rows.append({
                    "policy": p,
                    "side": side,
                    "fill_events": len(z),
                    "qty": z.qty.sum(),
                    "gross_edge_c": _wavg(z, "gross_edge_at_fill_c"),
                    "markout_5s_c": _wavg(z, "markout_5s_c"),
                    "markout_15s_c": _wavg(z, "markout_15s_c"),
                    "markout_30s_c": _wavg(z, "markout_30s_c"),
                })
            print(pd.DataFrame(side_rows).round(4).to_string(index=False))

        print("\nINTERPRETATION GUARDRAILS")
        print("  - This is development replay on an already-explored sample, not external OOS validation.")
        print("  - No threshold sweep: support and replenishment use zero boundaries only.")
        print("  - Do not add asset/side/spread thresholds after seeing this output.")
        print("  - Primary question: does L3_SUPPORT_REPLENISH_STATE_Q1 improve markouts/adverse selection/PnL and remain stable across both halves?")
        print("  - Even if positive, freeze before reading a fresh corrected V5 validation session.")
        print(f"\nOUTPUTS: {out}")
        print("=" * 150)

    return {
        "output_dir": out,
        "policy_summary": policy_summary,
        "chronology": chronology,
        "windows": windows_all,
        "contracts": contracts,
        "fills": fills,
        "episodes": episodes,
        "counts": counts,
    }
