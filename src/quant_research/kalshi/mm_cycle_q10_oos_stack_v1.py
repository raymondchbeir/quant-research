from __future__ import annotations

"""Frozen Candidate-C/Q10 live OOS recorder + shadow market maker + dashboard.

SHADOW ONLY: this module does not submit, amend, cancel, or cross real Kalshi orders.

Frozen strategy
---------------
Universe: KXBTC15M, KXBNB15M, KXDOGE15M, KXETH15M, KXHYPE15M,
          KXNEAR15M, KXSOL15M, KXXRP15M, KXZEC15M
Window:   M0 <= elapsed < M5
Entry:    Q10 at public BBO on the L3-supported side when natural YES spread >=2c.
After any entry fill, stop adding inventory and quote ONLY the opposite BBO until flat.
M5:       cancel passive shadow quote and liquidate any residual at executable BBO.
Queue:    join behind displayed L1; exact-price aggressive flow burns queue ahead;
          trade-through fills; no cancellation-ahead credit; any fill cancels residual.
Fees:     maker fills assumed zero only after live fee preflight verifies fee_type=quadratic.
          Forced M5 liquidation pays current quadratic taker trade fee. Exchange balance-
          rounding adjustments cannot be known in a shadow account, so the dashboard also
          reports a conservative upper-bound rounding drag of $0.0099 per forced liquidation.

OOS discipline
--------------
No parameters may change after OOS starts. The raw V5 capture mechanics are reused exactly,
but metadata is rewritten as OOS/frozen rather than development. Recommended maturity gate:
>=24h elapsed AND >=90 complete 9-series M0-M5 windows. Do not stop early based on PnL.
"""

import argparse
import asyncio
import heapq
import html
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_event_time_m0_m5_recorder_v5 as V5

STACK_VERSION = "MM_CYCLE_Q10_OOS_STACK_V1"
CAPTURE_VERSION = "MM_EVENT_TIME_M0_M5_OOS_CYCLE_Q10_V1"
SHADOW_VERSION = "MM_CYCLE_Q10_OOS_SHADOW_V1"

SERIES = tuple(V5.CRYPTO_SERIES)
QUOTE_SIZE = 10.0
SPREAD_FLOOR_C = 2.0
EPS = 1e-12
REORDER_LAG_S = 1.0
POLL_S = 0.10
MAX_MARKOUT_AGE_S = 2.0
MARKOUTS_S = (5.0, 15.0, 30.0)
FEE_CHANGE_HORIZON_H = 72.0
MIN_OOS_HOURS = 24.0
MIN_COMPLETE_WINDOWS = 90
DEV_FEE_ADJUSTED_BENCHMARK_PER_DAY = 75.42

ROOT = C.DATA_ROOT / "mm_event_m0_m5_oos_cycle_q10_v1"
CONTROL_PATH = ROOT / "active_stack.json"
ROOT.mkdir(parents=True, exist_ok=True)

_SHADOW = None
_SHADOW_THREAD = None


def _iso_ts(ts=None):
    if ts is None:
        return pd.Timestamp.now(tz="UTC").isoformat()
    try:
        return pd.Timestamp(float(ts), unit="s", tz="UTC").isoformat()
    except Exception:
        return str(ts)


def _ts(x):
    if x is None:
        return np.nan
    try:
        z = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(z):
            return np.nan
        return float(z.timestamp())
    except Exception:
        return np.nan


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _pid_state(pid):
    try:
        p = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(int(pid))],
            capture_output=True, text=True, timeout=2,
        )
        if p.returncode != 0:
            return None
        s = p.stdout.strip()
        return s or None
    except Exception:
        return None


def _pid_alive(pid):
    s = _pid_state(pid)
    return bool(s and "Z" not in s.upper())


def _session_dir_now():
    return ROOT / pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")


def _frozen_spec():
    return {
        "stack_version": STACK_VERSION,
        "capture_version": CAPTURE_VERSION,
        "shadow_version": SHADOW_VERSION,
        "research_stage": "FROZEN_OOS_SHADOW",
        "frozen_before_oos": True,
        "development_source_session": "20260816_070627",
        "development_fee_adjusted_benchmark_usd_per_day": DEV_FEE_ADJUSTED_BENCHMARK_PER_DAY,
        "universe": list(SERIES),
        "window": "M0 <= elapsed < M5",
        "quote_size": QUOTE_SIZE,
        "entry": {
            "rule": "Candidate C",
            "same_side_l3_depth_gt_opposite_side_l3_depth": True,
            "natural_yes_spread_min_cents": SPREAD_FLOOR_C,
            "price": "public YES BBO",
        },
        "inventory": {
            "after_first_entry_fill": "stop adding inventory",
            "exit": "quote only opposite public BBO until flat",
            "exit_filter": "none",
            "exit_qty": "absolute current inventory",
            "m5": "cross remaining inventory at executable BBO",
        },
        "queue_model": {
            "join": "back of displayed L1",
            "exact_price_trade": "burn queue ahead first",
            "trade_through": "fills remaining hypothetical quote",
            "cancellation_ahead_credit": False,
            "after_any_fill": "cancel residual; re-evaluate/rejoin on next book event",
        },
        "fee_model": {
            "maker": "must preflight fee_type=quadratic; modeled $0",
            "forced_m5": "quadratic taker trade fee using live series fee_multiplier",
            "trade_fee_rounding": "ceil to nearest $0.0001",
            "balance_rounding": "unknown in shadow; conservative upper bound $0.0099 per forced liquidation also reported",
        },
        "oos_maturity": {
            "minimum_elapsed_hours": MIN_OOS_HOURS,
            "minimum_complete_9_series_windows": MIN_COMPLETE_WINDOWS,
            "stop_early_based_on_pnl": False,
        },
        "forbidden_after_start": [
            "asset filter", "side filter", "minute filter", "spread threshold change",
            "L1/L3 threshold change", "quote size change", "exit mechanic change",
            "fee assumption change without declaring OOS invalid",
        ],
    }


def fee_preflight(*, horizon_hours=FEE_CHANGE_HORIZON_H, save_path=None, show=True):
    """Verify the frozen zero-maker-fee assumption and capture current multipliers."""
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    problems = []
    multipliers = {}

    for series in SERIES:
        try:
            payload = C.rest_get(f"/series/{series}", {})
            s = payload.get("series") or {}
        except Exception as exc:
            problems.append(f"{series}: get-series failed: {exc!r}")
            continue

        fee_type = str(s.get("fee_type") or "").strip().lower()
        mult = _f(s.get("fee_multiplier"))
        if fee_type != "quadratic":
            problems.append(f"{series}: fee_type={fee_type!r}, expected 'quadratic' (zero-maker structure)")
        if not np.isfinite(mult) or mult <= 0:
            problems.append(f"{series}: invalid fee_multiplier={s.get('fee_multiplier')!r}")
        else:
            multipliers[series] = float(mult)

        upcoming = []
        try:
            fc = C.rest_get("/series/fee_changes", {"series_ticker": series, "show_historical": False})
            upcoming = fc.get("series_fee_change_arr") or []
        except Exception as exc:
            problems.append(f"{series}: fee-change preflight failed: {exc!r}")

        near = []
        for r in upcoming:
            t = pd.to_datetime(r.get("scheduled_ts"), utc=True, errors="coerce")
            if pd.isna(t):
                continue
            hours = (t - now).total_seconds() / 3600.0
            if -1e-9 <= hours <= float(horizon_hours):
                near.append(r)
        if near:
            problems.append(f"{series}: scheduled fee change within {horizon_hours:.0f}h: {near}")

        rows.append({
            "series": series,
            "fee_type": fee_type,
            "fee_multiplier": mult,
            "last_updated_ts": s.get("last_updated_ts"),
            "upcoming_fee_changes": upcoming,
            "near_horizon_fee_changes": near,
        })

    ok = (len(problems) == 0 and len(multipliers) == len(SERIES))
    out = {
        "time": _iso_ts(),
        "ok": ok,
        "horizon_hours": float(horizon_hours),
        "series": rows,
        "multipliers": multipliers,
        "problems": problems,
        "guardrail": "OOS stack refuses startup unless maker-fee structure is verified for every frozen series.",
    }
    if save_path is not None:
        _atomic_json(save_path, out)
    if show:
        print("FEE PREFLIGHT:", "PASS" if ok else "FAIL")
        for r in rows:
            print(f"  {r['series']:10s} type={r['fee_type']!s:28s} multiplier={r['fee_multiplier']}")
        for p in problems:
            print("  ERROR:", p)
    if not ok:
        raise RuntimeError("Fee preflight failed; refusing frozen OOS startup. " + " | ".join(problems))
    return out


def _rewrite_oos_metadata(path: Path, obj, original_atomic):
    """Intercept V5's metadata writes while leaving capture mechanics untouched."""
    name = Path(path).name
    spec = _frozen_spec()
    if name == "development_plan.json":
        o = {
            "research_stage": "FROZEN_OOS_SHADOW",
            "note": "Filename retained only for compatibility with V5 capture code; this is NOT development.",
            "frozen_strategy": spec,
        }
        original_atomic(path, o)
        original_atomic(Path(path).with_name("frozen_strategy.json"), spec)
        return

    if name == "capture_spec.json":
        o = dict(obj)
        o.update({
            "study_version": CAPTURE_VERSION,
            "purpose": "fresh frozen OOS event-time capture for CYCLE_ALWAYS_EXIT Q10 shadow validation",
            "research_stage": "FROZEN_OOS_SHADOW",
            "strategy_pnl_recorded": False,
            "frozen_strategy_file": "frozen_strategy.json",
            "development_plan_file": None,
        })
        original_atomic(path, o)
        return

    if name == "session_manifest.json":
        o = dict(obj)
        o["study_version"] = CAPTURE_VERSION
        o["research_stage"] = "FROZEN_OOS_SHADOW"
        o["frozen_strategy"] = spec
        o.pop("development_plan", None)
        cap = dict(o.get("capture_spec") or {})
        cap["study_version"] = CAPTURE_VERSION
        cap["research_stage"] = "FROZEN_OOS_SHADOW"
        cap["purpose"] = "fresh frozen OOS event-time capture for CYCLE_ALWAYS_EXIT Q10 shadow validation"
        o["capture_spec"] = cap
        original_atomic(path, o)
        return

    if name == "health.json":
        o = dict(obj)
        o["study_version"] = CAPTURE_VERSION
        o["research_stage"] = "FROZEN_OOS_SHADOW"
        original_atomic(path, o)
        return

    original_atomic(path, obj)


async def _run_oos_recorder(session_dir):
    original_atomic = V5._atomic_json
    old_study = V5.STUDY_VERSION
    V5.STUDY_VERSION = CAPTURE_VERSION

    def intercept(path, obj):
        return _rewrite_oos_metadata(Path(path), obj, original_atomic)

    V5._atomic_json = intercept
    try:
        await V5.run_event_time_m0_m5_v5_recorder(Path(session_dir))
    finally:
        V5._atomic_json = original_atomic
        V5.STUDY_VERSION = old_study


class JsonlTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.partial = b""

    def read_new(self):
        if not self.path.exists():
            return []
        out = []
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
            self.offset = fh.tell()
        if not chunk:
            return out
        raw = self.partial + chunk
        lines = raw.split(b"\n")
        self.partial = lines.pop() if lines else b""
        for line in lines:
            if not line:
                continue
            try:
                out.append(json.loads(line.decode("utf-8")))
            except Exception:
                pass
        return out


def _top_state(r):
    if not bool(r.get("valid_bbo")):
        return None
    bids = r.get("bid_levels") or []
    asks = r.get("ask_levels") or []
    if not bids or not asks:
        return None
    try:
        bid = float(r["yes_bid"])
        ask = float(r["yes_ask"])
        bq = max(0.0, float(r["yes_bid_size"]))
        aq = max(0.0, float(r["yes_ask_size"]))
        b3 = max(0.0, sum(float(x[1]) for x in bids[:3]))
        a3 = max(0.0, sum(float(x[1]) for x in asks[:3]))
    except Exception:
        return None
    if not (0.0 <= bid < ask <= 1.0):
        return None
    return {
        "bid": bid, "ask": ask, "bid_q1": bq, "ask_q1": aq,
        "bid_depth3": b3, "ask_depth3": a3,
        "mid": 0.5 * (bid + ask), "spread_c": 100.0 * (ask - bid),
        "bid_levels": [(float(p), float(q)) for p, q in bids[:3]],
        "ask_levels": [(float(p), float(q)) for p, q in asks[:3]],
    }


def _entry_side(cur):
    if cur is None or cur["spread_c"] + 1e-9 < SPREAD_FLOOR_C:
        return None
    if cur["bid_depth3"] > cur["ask_depth3"] + EPS:
        return "BID"
    if cur["ask_depth3"] > cur["bid_depth3"] + EPS:
        return "ASK"
    return None


def _quadratic_taker_fee(qty, price, multiplier):
    qty, price, multiplier = float(qty), float(price), float(multiplier)
    if qty <= EPS:
        return 0.0
    raw = 0.07 * multiplier * qty * price * (1.0 - price)
    return math.ceil(max(0.0, raw) * 10000.0 - 1e-12) / 10000.0


class FrozenCycleShadow:
    def __init__(self, session_dir, fee_preflight_result):
        self.session_dir = Path(session_dir).resolve()
        self.out_dir = self.session_dir / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.out_dir / "shadow_events.jsonl"
        self.fill_path = self.out_dir / "shadow_fills.jsonl"
        self.summary_path = self.out_dir / "shadow_summary.json"
        self.contract_path = self.out_dir / "shadow_contracts.csv"
        self.fee_path = self.out_dir / "fee_preflight.json"
        _atomic_json(self.fee_path, fee_preflight_result)
        _atomic_json(self.out_dir / "frozen_strategy.json", _frozen_spec())

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.started_at = pd.Timestamp.now(tz="UTC")
        self.last_error = None
        self.thread_alive = False

        self.book_tail = JsonlTail(self.session_dir / "book_top3_events.jsonl")
        self.trade_tail = JsonlTail(self.session_dir / "trades_event_time.jsonl")
        self.meta_tail = JsonlTail(self.session_dir / "market_metadata.jsonl")

        self.event_heap = []
        self.event_counter = 0
        self.current = {}
        self.meta = {}
        self.series_by_ticker = {}
        self.close_by_ticker = {}
        self.quote = {}
        self.inventory = defaultdict(float)
        self.long_lots = defaultdict(deque)
        self.short_lots = defaultdict(deque)
        self.pending_markouts = []
        self.markout_counter = 0
        self.fills = []
        self.contracts = {}
        self.finalized = set()
        self.window_series = defaultdict(set)

        self.c = Counter()
        self.passive_matched_pnl = 0.0
        self.forced_liq_gross_pnl = 0.0
        self.taker_trade_fees = 0.0
        self.rounding_fee_upper_bound = 0.0
        self.maker_fees = 0.0
        self.max_abs_inventory = 0.0
        self.running_peak = 0.0
        self.max_drawdown = 0.0
        self.window_pnl = defaultdict(float)

        self.fee_mult = {str(k): float(v) for k, v in (fee_preflight_result.get("multipliers") or {}).items()}
        self._last_save = 0.0

    def emit(self, event, ticker=None, **detail):
        row = {"time": _iso_ts(), "event": event, "ticker": ticker, **detail}
        with self.event_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
        if event in {"ENTRY_FILL", "EXIT_FILL", "M5_LIQUIDATE", "ERROR"}:
            msg = f"[{pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S UTC')}] {event}"
            if ticker:
                msg += f" | {ticker}"
            if detail:
                msg += " | " + " ".join(f"{k}={v}" for k, v in detail.items() if k in {"qty","price","pnl","fee","inventory"})
            print(msg)

    def _write_fill(self, f):
        with self.fill_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(f, separators=(",", ":"), default=str) + "\n")

    def _update_meta(self):
        for r in self.meta_tail.read_new():
            ticker = str(r.get("ticker") or "")
            if not ticker:
                continue
            series = str(r.get("series_ticker") or "")
            self.meta[ticker] = r
            self.series_by_ticker[ticker] = series
            self.close_by_ticker[ticker] = str(r.get("close_time") or "")

    def _push_new_events(self):
        self._update_meta()
        for r in self.book_tail.read_new():
            t = _ts(r.get("receipt_time"))
            ticker = str(r.get("ticker") or "")
            if not np.isfinite(t) or not ticker:
                continue
            self.event_counter += 1
            heapq.heappush(self.event_heap, (float(t), 0, self.event_counter, "book", r))
        for r in self.trade_tail.read_new():
            t = _ts(r.get("receipt_time"))
            ticker = str(r.get("ticker") or "")
            if not np.isfinite(t) or not ticker:
                continue
            self.event_counter += 1
            heapq.heappush(self.event_heap, (float(t), 1, self.event_counter, "trade", r))

    def _desired_quote(self, ticker, cur, elapsed):
        if cur is None or not (0.0 <= elapsed < 300.0) or ticker in self.finalized:
            return None
        inv = float(self.inventory[ticker])
        if abs(inv) <= EPS:
            side = _entry_side(cur)
            if side is None:
                return None
            return {
                "role": "ENTRY", "side": side,
                "price": cur["bid"] if side == "BID" else cur["ask"],
                "qty": QUOTE_SIZE,
                "queue_ahead": cur["bid_q1"] if side == "BID" else cur["ask_q1"],
            }
        side = "ASK" if inv > 0 else "BID"
        return {
            "role": "EXIT", "side": side,
            "price": cur["ask"] if side == "ASK" else cur["bid"],
            "qty": abs(inv),
            "queue_ahead": cur["ask_q1"] if side == "ASK" else cur["bid_q1"],
        }

    @staticmethod
    def _quote_same(a, b):
        if a is None or b is None:
            return a is None and b is None
        return (
            a["role"] == b["role"]
            and a["side"] == b["side"]
            and abs(float(a["price"]) - float(b["price"])) <= EPS
            and abs(float(a["qty"]) - float(b["qty"])) <= EPS
        )

    def _reconcile_quote(self, ticker, cur, elapsed, t):
        desired = self._desired_quote(ticker, cur, elapsed)
        old = self.quote.get(ticker)
        if self._quote_same(old, desired):
            return
        if old is not None:
            self.c["quote_cancels"] += 1
            self.quote.pop(ticker, None)
        if desired is not None:
            q = dict(desired)
            q.update({
                "join_ts": float(t),
                "queue_ahead_initial": float(desired["queue_ahead"]),
                "remaining_qty": float(desired["qty"]),
            })
            self.quote[ticker] = q
            self.c[f"{q['role']}_quote_opens"] += 1

    def _resolve_markouts(self, t, mid):
        while self.pending_markouts and self.pending_markouts[0][0] <= t + EPS:
            target, _, fill_idx, h, sign = heapq.heappop(self.pending_markouts)
            age = t - target
            if -EPS <= age <= MAX_MARKOUT_AGE_S + EPS and 0 <= fill_idx < len(self.fills):
                f = self.fills[fill_idx]
                tag = f"{int(h)}s"
                f[f"future_mid_{tag}"] = float(mid)
                f[f"markout_{tag}_c"] = float(sign) * (float(mid) - float(f["price"])) * 100.0

    def _on_book(self, t, r):
        ticker = str(r.get("ticker") or "")
        cur = _top_state(r)
        elapsed = _f(r.get("elapsed_s"))
        if cur is not None:
            self.current[ticker] = cur
            self._resolve_markouts(t, cur["mid"])
        typ = str(r.get("event_type") or "")

        if typ == "trade_window_end" or (np.isfinite(elapsed) and elapsed >= 300.0 and ticker not in self.finalized):
            if cur is not None:
                self.current[ticker] = cur
            self._finalize_m5(ticker, t)
            return

        if not np.isfinite(elapsed):
            return
        if cur is None:
            if 0.0 <= elapsed < 300.0:
                self.quote.pop(ticker, None)
            return
        self._reconcile_quote(ticker, cur, float(elapsed), t)

    def _passive_match(self, ticker, side, qty, px):
        pnl = 0.0
        if side == "BID":
            rem = qty
            shorts = self.short_lots[ticker]
            while rem > EPS and shorts:
                sq, spx = shorts[0]
                m = min(rem, sq)
                pnl += (spx - px) * m
                rem -= m
                sq -= m
                if sq <= EPS:
                    shorts.popleft()
                else:
                    shorts[0] = (sq, spx)
            if rem > EPS:
                self.long_lots[ticker].append((rem, px))
        else:
            rem = qty
            longs = self.long_lots[ticker]
            while rem > EPS and longs:
                lq, lpx = longs[0]
                m = min(rem, lq)
                pnl += (px - lpx) * m
                rem -= m
                lq -= m
                if lq <= EPS:
                    longs.popleft()
                else:
                    longs[0] = (lq, lpx)
            if rem > EPS:
                self.short_lots[ticker].append((rem, px))
        self.passive_matched_pnl += pnl
        return pnl

    def _on_trade(self, t, r):
        ticker = str(r.get("ticker") or "")
        elapsed = _f(r.get("elapsed_s"))
        if not np.isfinite(elapsed) or not (0.0 <= elapsed < 300.0) or ticker in self.finalized:
            return
        q = self.quote.get(ticker)
        if q is None:
            return

        trade_px = _f(r.get("yes_price"))
        trade_qty = _f(r.get("qty"))
        taker = str(r.get("taker_book_side") or "").lower()
        if not (np.isfinite(trade_px) and np.isfinite(trade_qty) and trade_qty > 0):
            return
        side = "BID" if taker == "ask" else "ASK" if taker == "bid" else None
        if side != q["side"]:
            return

        qpx = float(q["price"])
        trade_through = False
        available = 0.0
        if side == "BID":
            if trade_px < qpx - EPS:
                trade_through = True
                available = float(q["remaining_qty"])
            elif abs(trade_px - qpx) <= EPS:
                burn = min(float(q["queue_ahead"]), trade_qty)
                q["queue_ahead"] -= burn
                available = max(0.0, trade_qty - burn)
            else:
                return
        else:
            if trade_px > qpx + EPS:
                trade_through = True
                available = float(q["remaining_qty"])
            elif abs(trade_px - qpx) <= EPS:
                burn = min(float(q["queue_ahead"]), trade_qty)
                q["queue_ahead"] -= burn
                available = max(0.0, trade_qty - burn)
            else:
                return
        if available <= EPS:
            return

        qty = min(float(q["remaining_qty"]), available)
        if qty <= EPS:
            return
        role = q["role"]
        sign = 1.0 if side == "BID" else -1.0
        inv_before = float(self.inventory[ticker])
        self.inventory[ticker] += sign * qty
        if abs(self.inventory[ticker]) < 1e-9:
            self.inventory[ticker] = 0.0
        self.max_abs_inventory = max(self.max_abs_inventory, abs(self.inventory[ticker]))
        matched_delta = self._passive_match(ticker, side, qty, qpx)

        cur = self.current.get(ticker)
        fill = {
            "time": _iso_ts(t), "fill_ts": float(t), "ticker": ticker,
            "series": self.series_by_ticker.get(ticker, str(r.get("series_ticker") or "")),
            "role": role, "side": side, "qty": qty, "price": qpx,
            "trade_through": bool(trade_through),
            "historical_trade_price": trade_px, "historical_trade_qty": trade_qty,
            "queue_ahead_initial": q["queue_ahead_initial"],
            "inventory_before": inv_before, "inventory_after": float(self.inventory[ticker]),
            "matched_pnl_delta": matched_delta,
            "mid_at_fill": cur["mid"] if cur else np.nan,
            "markout_5s_c": np.nan, "markout_15s_c": np.nan, "markout_30s_c": np.nan,
        }
        self.fills.append(fill)
        idx = len(self.fills) - 1
        for h in MARKOUTS_S:
            self.markout_counter += 1
            heapq.heappush(self.pending_markouts, (t + h, self.markout_counter, idx, h, sign))
        self._write_fill(fill)

        self.c["fill_events"] += 1
        self.c["fill_qty_x1000"] += int(round(qty * 1000.0))
        self.c[f"{role}_fill_events"] += 1
        if trade_through:
            self.c["trade_through_fills"] += 1

        if role == "ENTRY" and abs(inv_before) <= EPS:
            self.c["cycles_started"] += 1
        if role == "EXIT" and abs(self.inventory[ticker]) <= EPS:
            self.c["cycles_completed"] += 1

        self.quote.pop(ticker, None)
        self.emit(f"{role}_FILL", ticker, qty=round(qty, 4), price=qpx,
                  inventory=round(self.inventory[ticker], 4), pnl=round(matched_delta, 6))

    def _liquidation_gross(self, ticker, cur):
        pnl = 0.0
        qty = 0.0
        if cur is None:
            return np.nan, 0.0, np.nan
        for q, entry in self.long_lots[ticker]:
            pnl += (cur["bid"] - entry) * q
            qty += q
        for q, entry in self.short_lots[ticker]:
            pnl += (entry - cur["ask"]) * q
            qty += q
        inv = float(self.inventory[ticker])
        px = cur["bid"] if inv > EPS else cur["ask"] if inv < -EPS else np.nan
        return float(pnl), float(qty), float(px) if np.isfinite(px) else np.nan

    def _current_executable(self):
        gross = 0.0
        fees = 0.0
        rounding_upper = 0.0
        open_contracts = 0
        abs_inv = 0.0
        for ticker, inv in list(self.inventory.items()):
            if abs(inv) <= EPS or ticker in self.finalized:
                continue
            cur = self.current.get(ticker)
            g, qty, px = self._liquidation_gross(ticker, cur)
            if not np.isfinite(g) or not np.isfinite(px):
                continue
            series = self.series_by_ticker.get(ticker, "")
            mult = self.fee_mult.get(series)
            if mult is None:
                continue
            gross += g
            fees += _quadratic_taker_fee(qty, px, mult)
            rounding_upper += 0.0099
            open_contracts += 1
            abs_inv += abs(inv)
        return {
            "open_inventory_contracts": open_contracts,
            "open_abs_inventory": abs_inv,
            "current_residual_gross": gross,
            "current_estimated_taker_fee": fees,
            "current_rounding_upper": rounding_upper,
            "current_residual_net": gross - fees,
            "current_residual_net_conservative": gross - fees - rounding_upper,
        }

    def _finalize_m5(self, ticker, t):
        if ticker in self.finalized:
            return
        self.quote.pop(ticker, None)
        cur = self.current.get(ticker)
        gross, qty, px = self._liquidation_gross(ticker, cur)
        series = self.series_by_ticker.get(ticker, "")
        fee = 0.0
        rounding_upper = 0.0
        if qty > EPS:
            if not np.isfinite(gross) or not np.isfinite(px):
                self.emit("ERROR", ticker, detail="M5 residual but no valid BBO")
                self.last_error = f"{ticker}: M5 residual but no valid BBO"
                return
            mult = self.fee_mult.get(series)
            if mult is None:
                self.emit("ERROR", ticker, detail=f"Missing fee multiplier for {series}")
                self.last_error = f"{ticker}: missing fee multiplier"
                return
            fee = _quadratic_taker_fee(qty, px, mult)
            rounding_upper = 0.0099
            self.forced_liq_gross_pnl += gross
            self.taker_trade_fees += fee
            self.rounding_fee_upper_bound += rounding_upper
            self.c["forced_liquidations"] += 1
            self.c["forced_liq_qty_x1000"] += int(round(qty * 1000.0))
            self.emit("M5_LIQUIDATE", ticker, qty=round(qty, 4), price=px,
                      pnl=round(gross, 6), fee=round(fee, 6))
        self.long_lots[ticker].clear()
        self.short_lots[ticker].clear()
        self.inventory[ticker] = 0.0
        self.finalized.add(ticker)

        close = self.close_by_ticker.get(ticker) or str(self.meta.get(ticker, {}).get("close_time") or "")
        if close:
            self.window_series[close].add(series)
        net_delta = (gross if np.isfinite(gross) else 0.0) - fee
        self.window_pnl[close] += net_delta

        self.contracts[ticker] = {
            "ticker": ticker, "series": series, "close_time": close,
            "finalized_time": _iso_ts(t), "forced_liquidation_qty": qty,
            "forced_liquidation_gross_pnl": gross, "taker_trade_fee": fee,
            "rounding_fee_upper_bound": rounding_upper,
        }

    def _update_drawdown(self):
        total = self.passive_matched_pnl + self.forced_liq_gross_pnl - self.taker_trade_fees
        self.running_peak = max(self.running_peak, total)
        self.max_drawdown = min(self.max_drawdown, total - self.running_peak)

    def _summary(self):
        now = pd.Timestamp.now(tz="UTC")
        runtime_h = max(0.0, (now - self.started_at).total_seconds() / 3600.0)
        current = self._current_executable()
        realized_net = self.passive_matched_pnl + self.forced_liq_gross_pnl - self.taker_trade_fees
        executable_total = realized_net + current["current_residual_net"]
        conservative_total = realized_net - self.rounding_fee_upper_bound + current["current_residual_net_conservative"]
        scale = 24.0 / runtime_h if runtime_h > 1e-6 else np.nan
        complete_windows = sum(1 for v in self.window_series.values() if set(SERIES).issubset(v))
        cycles_started = int(self.c["cycles_started"])
        cycles_completed = int(self.c["cycles_completed"])
        m = {}
        for h in MARKOUTS_S:
            key = f"markout_{int(h)}s_c"
            vals, weights = [], []
            for f in self.fills:
                z = _f(f.get(key))
                if np.isfinite(z):
                    vals.append(z)
                    weights.append(float(f["qty"]))
            m[key] = float(np.average(vals, weights=weights)) if vals else np.nan

        recorder_health = _read_json(self.session_dir / "health.json", {}) or {}
        maturity = bool(runtime_h >= MIN_OOS_HOURS and complete_windows >= MIN_COMPLETE_WINDOWS)
        return {
            "time": now.isoformat(), "stack_version": STACK_VERSION,
            "shadow_version": SHADOW_VERSION, "session_dir": str(self.session_dir),
            "runtime_hours": runtime_h, "thread_alive": self.thread_alive,
            "last_error": self.last_error,
            "recorder_healthy": bool(recorder_health.get("healthy")),
            "recorder_running": bool(recorder_health.get("running")),
            "connection_epoch": recorder_health.get("connection_epoch"),
            "sequence_gaps": recorder_health.get("sequence_gaps"),
            "sequence_numbers_missing": recorder_health.get("sequence_numbers_missing"),
            "complete_9_series_windows": complete_windows,
            "finalized_contracts": len(self.finalized),
            "passive_matched_pnl": self.passive_matched_pnl,
            "forced_liq_gross_pnl": self.forced_liq_gross_pnl,
            "maker_fees": self.maker_fees,
            "taker_trade_fees": self.taker_trade_fees,
            "rounding_fee_upper_bound_realized": self.rounding_fee_upper_bound,
            "realized_net_trade_fee_only": realized_net,
            "realized_net_conservative_rounding": realized_net - self.rounding_fee_upper_bound,
            **current,
            "net_executable_pnl_trade_fee_only": executable_total,
            "net_executable_pnl_conservative_rounding": conservative_total,
            "sample_rate_net_per_day": executable_total * scale if np.isfinite(scale) else np.nan,
            "sample_rate_net_per_day_conservative": conservative_total * scale if np.isfinite(scale) else np.nan,
            "development_benchmark_per_day": DEV_FEE_ADJUSTED_BENCHMARK_PER_DAY,
            "fill_events": int(self.c["fill_events"]),
            "fill_qty": self.c["fill_qty_x1000"] / 1000.0,
            "cycles_started": cycles_started, "cycles_completed": cycles_completed,
            "cycle_completion_pct": 100.0 * cycles_completed / cycles_started if cycles_started else np.nan,
            "forced_liquidations": int(self.c["forced_liquidations"]),
            "forced_liq_qty": self.c["forced_liq_qty_x1000"] / 1000.0,
            "max_abs_inventory": self.max_abs_inventory,
            "max_drawdown_online": self.max_drawdown,
            **m,
            "oos_mature": maturity,
            "oos_maturity_rule": f">={MIN_OOS_HOURS:.0f}h AND >={MIN_COMPLETE_WINDOWS} complete 9-series windows; never stop early based on PnL",
        }

    def _save(self):
        s = self._summary()
        _atomic_json(self.summary_path, s)
        if self.contracts:
            df = pd.DataFrame(list(self.contracts.values()))
            tmp = self.contract_path.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(self.contract_path)
        self._last_save = time.monotonic()

    def process_ready(self, *, flush=False):
        watermark = float("inf") if flush else time.time() - REORDER_LAG_S
        while self.event_heap and self.event_heap[0][0] <= watermark:
            t, _, _, typ, r = heapq.heappop(self.event_heap)
            if typ == "book":
                self._on_book(t, r)
            else:
                self._on_trade(t, r)
            self._update_drawdown()

    def run(self):
        self.thread_alive = True
        self.emit("SHADOW_START", detail="Frozen CYCLE_ALWAYS_EXIT Q10 OOS shadow")
        try:
            while not self.stop_event.is_set():
                self._push_new_events()
                self.process_ready()
                if time.monotonic() - self._last_save >= 1.0:
                    self._save()
                time.sleep(POLL_S)
            for _ in range(20):
                self._push_new_events()
                time.sleep(0.05)
            self.process_ready(flush=True)
            self._save()
        except Exception as exc:
            self.last_error = repr(exc)
            self.emit("ERROR", detail=repr(exc))
            try:
                self._save()
            except Exception:
                pass
        finally:
            self.thread_alive = False
            self.emit("SHADOW_STOP")
            try:
                self._save()
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()

    def snapshot(self):
        with self.lock:
            return self._summary()

    def active_table(self):
        rows = []
        for ticker in sorted(set(self.current) | set(self.quote) | set(self.inventory)):
            cur = self.current.get(ticker)
            q = self.quote.get(ticker)
            inv = float(self.inventory.get(ticker, 0.0))
            if q is None and abs(inv) <= EPS:
                state = "FLAT"
            elif q is not None:
                state = f"{q['role']} {q['side']}"
            else:
                state = "INVENTORY"
            rows.append({
                "ticker": ticker, "series": self.series_by_ticker.get(ticker, ""),
                "state": state, "inventory": inv,
                "bid": cur["bid"] if cur else np.nan, "ask": cur["ask"] if cur else np.nan,
                "spread_c": cur["spread_c"] if cur else np.nan,
                "quote_px": q["price"] if q else np.nan,
                "queue_ahead": q["queue_ahead"] if q else np.nan,
                "quote_qty": q["qty"] if q else np.nan,
            })
        return pd.DataFrame(rows)


def _control():
    return _read_json(CONTROL_PATH, {}) or {}


def oos_stack_status(*, show=True):
    ctl = _control()
    session = Path(ctl.get("session_dir", "")) if ctl.get("session_dir") else None
    health = _read_json(session / "health.json", {}) if session else {}
    shadow = None
    if _SHADOW is not None and session is not None and _SHADOW.session_dir == session.resolve():
        shadow = _SHADOW.snapshot()
    elif session is not None:
        shadow = _read_json(session / "FROZEN_CYCLE_ALWAYS_EXIT_Q10_SHADOW_V1" / "shadow_summary.json", {})
    out = {
        "running": bool(ctl and _pid_alive(ctl.get("recorder_pid"))),
        "recorder_pid": ctl.get("recorder_pid"),
        "pid_state": _pid_state(ctl.get("recorder_pid")) if ctl else None,
        "session_dir": str(session) if session else None,
        "health": health or {},
        "shadow": shadow or {},
        "fee_preflight": _read_json(session / "fee_preflight.json", {}) if session else {},
    }
    if show:
        print(json.dumps(out, indent=2, default=str))
    return out


def start_cycle_q10_oos_stack(*, startup_timeout_s=60.0, fee_horizon_hours=FEE_CHANGE_HORIZON_H):
    """Start fresh OOS recorder subprocess, then attach frozen read-only shadow."""
    global _SHADOW, _SHADOW_THREAD
    ctl = _control()
    if ctl and _pid_alive(ctl.get("recorder_pid")):
        raise RuntimeError(f"OOS recorder already running: {ctl}")
    if CONTROL_PATH.exists():
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass

    preflight = fee_preflight(horizon_hours=fee_horizon_hours, show=True)

    session = _session_dir_now().resolve()
    log_path = session.parent / f"{session.name}.startup.log"
    cmd = [sys.executable, "-m", "quant_research.kalshi.mm_cycle_q10_oos_stack_v1",
           "--run-recorder", str(session)]
    log_fh = log_path.open("a", buffering=1, encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(C.PROJECT_ROOT), stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_fh.close()

    _atomic_json(CONTROL_PATH, {
        "stack_version": STACK_VERSION, "capture_version": CAPTURE_VERSION,
        "recorder_pid": proc.pid, "session_dir": str(session),
        "started_at": _iso_ts(), "startup_log": str(log_path),
        "frozen_strategy": _frozen_spec(),
    })

    deadline = time.time() + float(startup_timeout_s)
    last = {}
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8")[-5000:]
            except Exception:
                pass
            raise RuntimeError(f"OOS recorder exited during startup rc={proc.returncode}\n{tail}")
        if session.exists():
            last = _read_json(session / "health.json", {}) or {}
            if last.get("running") and last.get("healthy"):
                break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"OOS recorder failed health timeout. Last health={last}")

    if last.get("study_version") != CAPTURE_VERSION:
        raise RuntimeError(f"Wrong capture version at startup: {last.get('study_version')}")

    _atomic_json(session / "fee_preflight.json", preflight)
    _atomic_json(session / "frozen_strategy.json", _frozen_spec())

    _SHADOW = FrozenCycleShadow(session, preflight)
    _SHADOW_THREAD = threading.Thread(
        target=_SHADOW.run, name="frozen-cycle-q10-oos-shadow", daemon=True
    )
    _SHADOW_THREAD.start()

    time.sleep(0.5)
    if not _SHADOW_THREAD.is_alive():
        raise RuntimeError(f"Shadow thread failed to start: {_SHADOW.last_error}")

    print("\nFROZEN OOS STACK READY")
    print("Session:", session)
    print("Recorder PID:", proc.pid)
    print("Strategy: CYCLE_ALWAYS_EXIT Q10 — FROZEN")
    print(f"OOS maturity rule: >= {MIN_OOS_HOURS:.0f}h AND >= {MIN_COMPLETE_WINDOWS} complete 9-series windows")
    print("Do not stop early because PnL looks good or bad.")
    return oos_stack_status(show=False)


def _stop_recorder(expected_session=None, timeout_s=45.0):
    ctl = _control()
    if not ctl:
        return None
    session = Path(ctl.get("session_dir", "")).resolve()
    if expected_session is not None and session != Path(expected_session).resolve():
        raise RuntimeError(f"Session mismatch: control={session}, expected={expected_session}")
    pid = int(ctl.get("recorder_pid"))
    if _pid_alive(pid):
        os.kill(pid, signal.SIGINT)
        deadline = time.time() + timeout_s
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.25)
        if _pid_alive(pid):
            os.kill(pid, signal.SIGTERM)
            deadline = time.time() + 10.0
            while time.time() < deadline and _pid_alive(pid):
                time.sleep(0.25)
        if _pid_alive(pid):
            p = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                               capture_output=True, text=True)
            cmd = p.stdout.strip()
            if "mm_cycle_q10_oos_stack_v1" not in cmd:
                raise RuntimeError(f"Refusing SIGKILL; PID command mismatch: {cmd}")
            os.kill(pid, signal.SIGKILL)
    return session


def stop_cycle_q10_oos_stack(*, timeout_s=45.0):
    """Recorder first, then drain and stop shadow so raw OOS data remain authoritative."""
    global _SHADOW, _SHADOW_THREAD
    ctl = _control()
    if not ctl:
        print("No active OOS stack control file.")
        return None
    session = Path(ctl["session_dir"]).resolve()
    print("Stopping OOS recorder...")
    _stop_recorder(expected_session=session, timeout_s=timeout_s)
    time.sleep(REORDER_LAG_S + 0.5)

    if _SHADOW is not None:
        print("Draining/stopping shadow...")
        _SHADOW.stop()
    if _SHADOW_THREAD is not None:
        _SHADOW_THREAD.join(timeout=10.0)

    final = oos_stack_status(show=False)
    try:
        CONTROL_PATH.unlink()
    except Exception:
        pass
    print("OOS stack stopped.")
    print("Session preserved:", session)
    s = final.get("shadow") or {}
    if s:
        print(f"Net executable PnL: ${_f(s.get('net_executable_pnl_trade_fee_only'), 0):+.4f}")
        print(f"Sample-rate net/day: ${_f(s.get('sample_rate_net_per_day'), np.nan):+.2f}")
        print("OOS mature:", s.get("oos_mature"))
    return session


def _money(x, digits=2):
    try:
        z = float(x)
        return f"${z:+,.{digits}f}" if np.isfinite(z) else "n/a"
    except Exception:
        return "n/a"


def _pct(x):
    try:
        z = float(x)
        return f"{z:.1f}%" if np.isfinite(z) else "n/a"
    except Exception:
        return "n/a"


def _card(label, value, sub="", state="neutral"):
    pal = {
        "ok": ("#eaf7ee", "#238636"), "warn": ("#fff7e6", "#9a6700"),
        "bad": ("#ffebe9", "#cf222e"), "info": ("#eef5ff", "#0969da"),
        "neutral": ("#f6f8fa", "#57606a"),
    }
    bg, border = pal[state]
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
        'padding:10px 12px;min-width:145px;flex:1">'
        f'<div style="font-size:10px;opacity:.7;text-transform:uppercase">{html.escape(str(label))}</div>'
        f'<div style="font-size:20px;font-weight:750">{html.escape(str(value))}</div>'
        + (f'<div style="font-size:10px;opacity:.68">{html.escape(str(sub))}</div>' if sub else "")
        + "</div>"
    )


def render_cycle_q10_oos_dashboard_html(show_rows=12):
    st = oos_stack_status(show=False)
    s = st.get("shadow") or {}
    h = st.get("health") or {}
    fee = st.get("fee_preflight") or {}
    recorder_ok = bool(st.get("running") and h.get("healthy"))
    shadow_ok = bool(s.get("thread_alive")) and not s.get("last_error")
    fees_ok = bool(fee.get("ok"))
    all_ok = recorder_ok and shadow_ok and fees_ok
    mature = bool(s.get("oos_mature"))
    title_state = "OOS MATURE — KEEP FROZEN" if mature else "OOS RUNNING — NOT YET MATURE"
    bg = "#eaf7ee" if all_ok else "#ffebe9"
    border = "#238636" if all_ok else "#cf222e"

    parts = [
        '<div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1500px">',
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:14px 16px;margin-bottom:10px">',
        f'<div style="font-size:22px;font-weight:800">FROZEN CYCLE_ALWAYS_EXIT Q10 — LIVE OOS SHADOW</div>',
        f'<div style="font-size:12px;opacity:.72">{title_state} · {html.escape(str(s.get("time") or _iso_ts()))}</div>',
        f'<div style="font-size:11px;opacity:.65">Session: {html.escape(str(st.get("session_dir")))}</div>',
        '</div>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">',
        _card("Recorder", "HEALTHY" if recorder_ok else "ATTENTION",
              f"epoch {h.get('connection_epoch')} · gaps {h.get('sequence_gaps')}", "ok" if recorder_ok else "bad"),
        _card("Shadow", "RUNNING" if shadow_ok else "ATTENTION",
              "read-only / no real orders", "ok" if shadow_ok else "bad"),
        _card("Fee preflight", "PASS" if fees_ok else "FAIL",
              "quadratic / zero-maker structure", "ok" if fees_ok else "bad"),
        _card("OOS maturity", "MATURE" if mature else "COLLECTING",
              s.get("oos_maturity_rule", ""), "ok" if mature else "info"),
        _card("Runtime", f"{_f(s.get('runtime_hours'), 0):.2f} h",
              f"{int(s.get('complete_9_series_windows', 0) or 0)} complete 9-series windows", "info"),
        '</div>',

        '<h3 style="margin:12px 0 6px">Net economics</h3>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">',
        _card("Passive matched", _money(s.get("passive_matched_pnl")), "maker fee $0 after preflight",
              "ok" if _f(s.get("passive_matched_pnl"), 0) >= 0 else "warn"),
        _card("Forced liq gross", _money(s.get("forced_liq_gross_pnl")), "M5 residual only", "neutral"),
        _card("Taker trade fees", _money(-_f(s.get("taker_trade_fees"), 0)), "quadratic formula", "warn"),
        _card("Realized net", _money(s.get("realized_net_trade_fee_only")), "completed/passively matched + M5 liquidations",
              "ok" if _f(s.get("realized_net_trade_fee_only"), 0) >= 0 else "warn"),
        _card("Open executable", _money(s.get("current_residual_net")), "if all current inventory crossed now", "neutral"),
        _card("Net executable", _money(s.get("net_executable_pnl_trade_fee_only")), "primary live shadow PnL",
              "ok" if _f(s.get("net_executable_pnl_trade_fee_only"), 0) >= 0 else "warn"),
        _card("Sample-rate / day", _money(s.get("sample_rate_net_per_day")), "compare only after OOS maturity", "info"),
        _card("Dev benchmark", _money(s.get("development_benchmark_per_day")), "fee-adjusted development", "neutral"),
        '</div>',

        '<h3 style="margin:12px 0 6px">Execution / inventory</h3>',
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">',
        _card("Fills", int(s.get("fill_events", 0) or 0), f"{_f(s.get('fill_qty'), 0):,.1f} contracts", "info"),
        _card("Cycles", f"{int(s.get('cycles_completed',0) or 0)} / {int(s.get('cycles_started',0) or 0)}",
              _pct(s.get("cycle_completion_pct")), "info"),
        _card("Forced M5", int(s.get("forced_liquidations", 0) or 0),
              f"{_f(s.get('forced_liq_qty'),0):,.1f} contracts", "warn"),
        _card("Open inventory", f"{_f(s.get('open_abs_inventory'),0):,.2f} ct",
              f"{int(s.get('open_inventory_contracts',0) or 0)} markets", "neutral"),
        _card("Max |inventory|", f"{_f(s.get('max_abs_inventory'),0):,.2f} ct", "per contract state", "neutral"),
        _card("5s markout", f"{_f(s.get('markout_5s_c')):+.3f}¢", "qty-weighted", "neutral"),
        _card("15s markout", f"{_f(s.get('markout_15s_c')):+.3f}¢", "qty-weighted", "neutral"),
        _card("30s markout", f"{_f(s.get('markout_30s_c')):+.3f}¢", "qty-weighted", "neutral"),
        '</div>',

        '<div style="border:1px solid #9a6700;background:#fff7e6;border-radius:10px;padding:9px 11px;font-size:11px;margin-bottom:10px">',
        '<b>Fee note:</b> maker fee is modeled as zero only because startup preflight verifies fee_type=quadratic. '
        'The shadow computes the quadratic M5 taker trade fee. Kalshi balance-rounding adjustments are account/order-state dependent, '
        f'so a conservative upper bound is also tracked: realized upper-bound drag {_money(s.get("rounding_fee_upper_bound_realized"),4)}. '
        f'Conservative net executable: {_money(s.get("net_executable_pnl_conservative_rounding"))}.'
        '</div>',
    ]

    if _SHADOW is not None:
        t = _SHADOW.active_table()
        if len(t):
            z = t[(t["state"] != "FLAT") | (t["inventory"].abs() > EPS)].copy()
            if len(z) == 0:
                z = t.tail(show_rows)
            else:
                z = z.tail(show_rows)
            parts.append('<h3 style="margin:12px 0 6px">Active shadow markets</h3>')
            parts.append('<div style="overflow-x:auto;font-size:11px">' +
                         z.round(4).to_html(index=False, border=0) + '</div>')

    if s.get("last_error"):
        parts.append('<div style="border:1px solid #cf222e;background:#ffebe9;border-radius:8px;padding:8px;margin-top:10px">'
                     f'<b>Shadow error:</b> {html.escape(str(s.get("last_error")))}</div>')

    parts.append('<div style="font-size:10px;opacity:.62;margin-top:10px">'
                 'OOS guardrail: dashboard observation does not authorize parameter changes. Raw recorder is authoritative; '
                 'formal offline frozen replay must reconcile against this shadow after collection.</div></div>')
    return "".join(parts)


async def watch_cycle_q10_oos_dashboard(refresh_seconds=2.0, show_rows=12):
    try:
        from IPython.display import HTML, display
    except Exception as exc:
        raise RuntimeError("Dashboard requires Jupyter/IPython.") from exc
    refresh_seconds = max(0.5, float(refresh_seconds))
    handle = display(HTML(render_cycle_q10_oos_dashboard_html(show_rows=show_rows)), display_id=True)
    try:
        while True:
            await asyncio.sleep(refresh_seconds)
            handle.update(HTML(render_cycle_q10_oos_dashboard_html(show_rows=show_rows)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        handle.update(HTML(render_cycle_q10_oos_dashboard_html(show_rows=show_rows)))
        print("Dashboard refresh stopped. Recorder + frozen shadow remain running.")


async def start_and_watch_cycle_q10_oos_stack(refresh_seconds=2.0, show_rows=12):
    start_cycle_q10_oos_stack()
    await watch_cycle_q10_oos_dashboard(refresh_seconds=refresh_seconds, show_rows=show_rows)


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-recorder")
    args = p.parse_args()
    if args.run_recorder:
        asyncio.run(_run_oos_recorder(Path(args.run_recorder)))
        return
    p.error("Use notebook helper start_cycle_q10_oos_stack(), or --run-recorder internally.")


if __name__ == "__main__":
    _main()
