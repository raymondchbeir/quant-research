from __future__ import annotations

import json
import os
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .window_regime import PRIMARY_DIR

STRATEGY_VERSION = "RANGE44_Q15_Q5_PROSPECTIVE_V1"
OUTPUT_DIR = STRATEGY_VERSION
BASE_RANGE44_DIR = "RANGE44_Q1_PROSPECTIVE_V1"

RANGE_THRESHOLD_C = 44.0
HIGH_BREADTH_MIN_SIGNALS = 3
NORMAL_QTY = 15.0
FLAGGED_QTY = 5.0
PRIMARY_QTY = 3.0

INTERVAL_SEC = 10.0
RECENT_TRADE_BUFFER_SEC = 180.0
EPS = 1e-9

_MONITOR = None
_MONITOR_THREAD = None
_MONITOR_LOCK = threading.RLock()


def _utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def _ts(x):
    if x is None:
        return pd.NaT
    try:
        if isinstance(x, (int, float)):
            z = float(x)
            if z > 1e17:
                return pd.to_datetime(int(z), unit="ns", utc=True)
            if z > 1e14:
                return pd.to_datetime(int(z), unit="us", utc=True)
            if z > 1e11:
                return pd.to_datetime(int(z), unit="ms", utc=True)
            return pd.to_datetime(z, unit="s", utc=True)
        return pd.to_datetime(x, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def _num(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _price(x):
    p = _num(x)
    if np.isfinite(p) and p > 1.5:
        p /= 100.0
    return p


def _atomic_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _bool_col(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _max_drawdown(window_pnl: pd.Series) -> float:
    x = pd.to_numeric(window_pnl, errors="coerce").fillna(0.0).to_numpy(float)
    if len(x) == 0:
        return 0.0
    eq = np.cumsum(x)
    eq0 = np.r_[0.0, eq]
    peaks = np.maximum.accumulate(eq0)
    return float((eq0 - peaks).min())


def _event_ts(obj):
    if not isinstance(obj, dict):
        return pd.NaT
    for key in ("received_ts", "recv_ts", "received_at", "time", "timestamp", "ts", "created_time"):
        if obj.get(key) is not None:
            out = _ts(obj.get(key))
            if not pd.isna(out):
                return out
    return pd.NaT


def _get_ticker(obj):
    if not isinstance(obj, dict):
        return None
    raw = obj.get("raw_msg") if isinstance(obj.get("raw_msg"), dict) else {}
    value = obj.get("ticker") or obj.get("market_ticker") or raw.get("market_ticker")
    return None if value is None else str(value)


def _parse_trade(obj):
    """Parse the recorder trade using taker_book_side directly.

    This intentionally does NOT use generic side aliases; the earlier capacity
    bug came from reading yes/no taker_side before bid/ask taker_book_side.
    """
    if not isinstance(obj, dict):
        return None
    raw = obj.get("raw_msg") if isinstance(obj.get("raw_msg"), dict) else {}
    ticker = _get_ticker(obj)
    if not ticker:
        return None

    raw_price = obj.get("yes_price")
    if raw_price is None:
        raw_price = raw.get("yes_price_dollars")
    p = _price(raw_price)

    raw_qty = raw.get("count_fp")
    if raw_qty is None:
        raw_qty = obj.get("qty")
    qty = _num(raw_qty)

    side = str(raw.get("taker_book_side") or obj.get("taker_book_side") or "").lower()
    ts = _event_ts(obj)
    trade_id = raw.get("trade_id") or obj.get("trade_id")

    if not np.isfinite(p) or not np.isfinite(qty) or qty <= 0:
        return None
    if side not in {"bid", "ask"} or pd.isna(ts):
        return None

    key = str(trade_id) if trade_id is not None else f"{ticker}|{ts}|{p:.4f}|{qty:.8f}|{side}"
    return {
        "ticker": ticker,
        "ts": ts,
        "yes_price": float(p),
        "qty": float(qty),
        "taker_book_side": side,
        "key": key,
    }


class FiniteFlowCapacityTracker:
    """Observed-flow capacity tracker calibrated to authoritative primary Q3 fills.

    Calibration:
      actual Q3 no-fill -> 0
      actual Q3 partial -> actual partial
      actual Q3 full    -> max(3, reconstructed finite aggressive flow)

    Historical startup streams trades.jsonl exactly once. After that it reads only
    appended bytes. A small recent-trade buffer lets a newly-written Q3 shadow row
    recover trades that arrived just before shadow_records.csv was atomically updated.
    """

    def __init__(self, trades_file: Path):
        self.trades_file = Path(trades_file)
        self.offset = 0
        self.initialized = False
        self.states = {}
        self.recent = deque()
        self.latest_trade_ts = pd.NaT

    @staticmethod
    def _row_state(row):
        return {
            "ticker": str(row["ticker"]),
            "direction": str(row.get("direction") or "").upper(),
            "entry_price": _num(row.get("entry_price")),
            "order_time": _utc(row.get("actual_post_time")),
            "cancel_time": _utc(row.get("cancel_time")),
            "queue_initial": _num(row.get("entry_queue")),
            "queue_remaining": _num(row.get("entry_queue")),
            "flow_capacity_ct": 0.0,
            "through_seen": False,
            "seen": set(),
        }

    def _apply_trade(self, state, trade):
        if trade["ticker"] != state["ticker"]:
            return
        if trade["key"] in state["seen"]:
            return

        ts = trade["ts"]
        if pd.isna(state["order_time"]) or pd.isna(state["cancel_time"]):
            return
        if ts < state["order_time"] or ts > state["cancel_time"]:
            return

        q = state["entry_price"]
        if not np.isfinite(q):
            return

        p = trade["yes_price"]
        qty = trade["qty"]
        side = trade["taker_book_side"]

        exact = False
        through = False

        if state["direction"] == "YES":
            if side != "ask":
                return
            exact = abs(p - q) < 1e-9
            through = p < q - 1e-9

        elif state["direction"] == "NO":
            if side != "bid":
                return
            yes_equiv = 1.0 - q
            exact = abs(p - yes_equiv) < 1e-9
            through = p > yes_equiv + 1e-9

        else:
            return

        if not exact and not through:
            return

        state["seen"].add(trade["key"])

        if through:
            state["through_seen"] = True
            state["queue_remaining"] = 0.0
            state["flow_capacity_ct"] += qty
            return

        queue = state["queue_remaining"]
        if not np.isfinite(queue):
            return

        if qty <= queue + EPS:
            state["queue_remaining"] = max(0.0, queue - qty)
            return

        state["flow_capacity_ct"] += max(0.0, qty - queue)
        state["queue_remaining"] = 0.0

    def _trim_recent(self):
        if pd.isna(self.latest_trade_ts):
            return
        cutoff = self.latest_trade_ts - pd.Timedelta(seconds=RECENT_TRADE_BUFFER_SEC)
        while self.recent and self.recent[0]["ts"] < cutoff:
            self.recent.popleft()

    def _replay_recent_for_state(self, state):
        for trade in self.recent:
            self._apply_trade(state, trade)

    def sync_shadow(self, shadow: pd.DataFrame):
        if len(shadow) == 0:
            return

        z = shadow.copy()
        z["entry_fill_qty"] = pd.to_numeric(z.get("entry_fill_qty"), errors="coerce").fillna(0.0)
        full = z[z["entry_fill_qty"] >= PRIMARY_QTY - EPS]

        for _, row in full.iterrows():
            ticker = str(row["ticker"])
            if ticker in self.states:
                continue
            state = self._row_state(row)
            self.states[ticker] = state
            if self.initialized:
                self._replay_recent_for_state(state)

    def _process_trade(self, trade):
        self.recent.append(trade)
        if pd.isna(self.latest_trade_ts) or trade["ts"] > self.latest_trade_ts:
            self.latest_trade_ts = trade["ts"]

        state = self.states.get(trade["ticker"])
        if state is not None:
            self._apply_trade(state, trade)

        self._trim_recent()

    def scan_new(self):
        if not self.trades_file.exists():
            raise FileNotFoundError(self.trades_file)

        size = self.trades_file.stat().st_size

        if size < self.offset:
            self.offset = 0
            self.initialized = False
            self.recent.clear()
            self.latest_trade_ts = pd.NaT
            for state in self.states.values():
                state["queue_remaining"] = state["queue_initial"]
                state["flow_capacity_ct"] = 0.0
                state["through_seen"] = False
                state["seen"] = set()

        with self.trades_file.open("rb") as f:
            f.seek(self.offset)

            while True:
                pos = f.tell()
                raw = f.readline()

                if not raw:
                    break

                if not raw.endswith(b"\n"):
                    f.seek(pos)
                    break

                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue

                trade = _parse_trade(obj)
                if trade is not None:
                    self._process_trade(trade)

            self.offset = f.tell()

        self.initialized = True

    def capacity_frame(self, shadow: pd.DataFrame):
        if len(shadow) == 0:
            return pd.DataFrame(columns=[
                "ticker", "actual_q3_fill", "flow_capacity_ct", "scalable_capacity_ct"
            ])

        z = shadow[["ticker", "entry_fill_qty"]].copy()
        z["ticker"] = z["ticker"].astype(str)
        z["actual_q3_fill"] = pd.to_numeric(z["entry_fill_qty"], errors="coerce").fillna(0.0)
        z["flow_capacity_ct"] = z["ticker"].map(
            {ticker: state["flow_capacity_ct"] for ticker, state in self.states.items()}
        ).fillna(0.0)

        actual = z["actual_q3_fill"].to_numpy(float)
        flow = z["flow_capacity_ct"].to_numpy(float)

        scalable = np.where(
            actual <= EPS,
            0.0,
            np.where(
                actual < PRIMARY_QTY - EPS,
                actual,
                np.maximum(actual, flow),
            ),
        )

        z["scalable_capacity_ct"] = scalable
        return z[["ticker", "actual_q3_fill", "flow_capacity_ct", "scalable_capacity_ct"]]


def _empty_summary(freeze_at, catchup_anchor):
    return pd.DataFrame([{
        "strategy": STRATEGY_VERSION,
        "frozen_at_utc": freeze_at,
        "catchup_anchor_utc": catchup_anchor,
        "prospective_windows": 0,
        "eligible_windows": 0,
        "flagged_windows": 0,
        "settled_contracts": 0.0,
        "open_contracts": 0.0,
        "strategy_realized_pnl": 0.0,
        "q3_realized_pnl_same_sample": 0.0,
        "max_drawdown": 0.0,
        "worst_complete_window_pnl": np.nan,
        "catchup_windows": 0,
        "catchup_flagged_windows": 0,
        "catchup_settled_contracts": 0.0,
        "catchup_open_contracts": 0.0,
        "catchup_realized_pnl": 0.0,
        "catchup_q3_realized_pnl": 0.0,
        "catchup_max_drawdown": 0.0,
        "catchup_worst_complete_window_pnl": np.nan,
    }])


class Range44Q15Q5ProspectiveMonitor:
    """Read-only scaled Range44 capacity monitor.

    Exact Range44 flags/eligibility come from RANGE44_Q1_PROSPECTIVE_V1.
    Catch-up is historical/counterfactual; only the new freeze is prospective.
    """

    def __init__(self, session_dir, interval_sec=INTERVAL_SEC, output_dir=None):
        self.session_dir = Path(session_dir)
        self.primary_dir = self.session_dir / PRIMARY_DIR
        self.shadow_records = self.primary_dir / "shadow_records.csv"
        self.trades_file = self.session_dir / "trades.jsonl"

        self.base_dir = self.session_dir / BASE_RANGE44_DIR
        self.base_contract_file = self.base_dir / "contract_detail.csv"
        self.base_window_file = self.base_dir / "window_detail.csv"
        self.base_hypothesis_file = self.base_dir / "hypothesis.json"

        for path in (
            self.shadow_records,
            self.trades_file,
            self.base_contract_file,
            self.base_window_file,
            self.base_hypothesis_file,
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing required scaled-Range44 input: {path}")

        self.out_dir = Path(output_dir) if output_dir else self.session_dir / OUTPUT_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.interval_sec = float(interval_sec)
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.last_refresh = None
        self.last_error = None
        self.capacity = FiniteFlowCapacityTracker(self.trades_file)

        base_meta = json.loads(self.base_hypothesis_file.read_text(encoding="utf-8"))
        self.catchup_anchor = _utc(base_meta.get("frozen_at_utc"))
        if pd.isna(self.catchup_anchor):
            raise RuntimeError("Original Range44 freeze timestamp is missing/invalid.")

        hp = self.out_dir / "hypothesis.json"
        existing = None
        if hp.exists():
            try:
                existing = json.loads(hp.read_text(encoding="utf-8"))
            except Exception:
                existing = None

        if isinstance(existing, dict) and existing.get("version") == STRATEGY_VERSION:
            self.freeze_at = _utc(existing.get("frozen_at_utc"))
            if pd.isna(self.freeze_at):
                self.freeze_at = pd.Timestamp.now(tz="UTC")
        else:
            self.freeze_at = pd.Timestamp.now(tz="UTC")

        self.summary = _empty_summary(self.freeze_at, self.catchup_anchor)
        self.windows = pd.DataFrame()
        self.contracts = pd.DataFrame()

        meta = {
            "version": STRATEGY_VERSION,
            "frozen_at_utc": self.freeze_at.isoformat(),
            "catchup_anchor_utc": self.catchup_anchor.isoformat(),
            "session_dir": str(self.session_dir),
            "read_only": True,
            "primary_strategy_unchanged": True,
            "base_range44_strategy": BASE_RANGE44_DIR,
            "prospective_definition": "decision_time strictly after this monitor's frozen_at_utc",
            "catchup_definition": (
                "decision_time strictly after original RANGE44_Q1 freeze; historical counterfactual, NOT OOS"
            ),
            "rule": {
                "normal_qty_per_asset": NORMAL_QTY,
                "flagged_qty_per_asset": FLAGGED_QTY,
                "range_threshold_c": RANGE_THRESHOLD_C,
                "high_breadth_min_signals": HIGH_BREADTH_MIN_SIGNALS,
            },
            "capacity_calibration": {
                "actual_q3_no_fill": 0,
                "actual_q3_partial": "actual primary shadow partial fill",
                "actual_q3_full": "max(actual Q3 fill, finite observed aggressive flow)",
                "trade_side_field": "taker_book_side",
                "large_size_note": (
                    "Observed-flow/no-market-impact counterfactual. Not proof of live deployable capacity."
                ),
            },
        }
        _atomic_json(meta, hp)
        _atomic_csv(self.summary, self.out_dir / "summary.csv")

    def _status_payload(self):
        with self.lock:
            summary = self.summary.copy()
            windows = self.windows.copy()
            last_refresh = self.last_refresh
            last_error = self.last_error

        latest = pd.DataFrame()
        if len(windows):
            prospective = windows[windows["strategy_prospective"].fillna(False)].copy()
            if len(prospective):
                cols = [
                    "decision_time", "signals", "max_mid_range_c", "range_flagged",
                    "target_qty_per_asset", "strategy_filled_contracts",
                    "strategy_open_contracts", "strategy_realized_pnl",
                    "q3_realized_pnl_same_sample",
                ]
                latest = prospective.sort_values("decision_time").tail(8)
                latest = latest[[c for c in cols if c in latest.columns]]

        return {
            "running": not self.stop_event.is_set(),
            "session_dir": str(self.session_dir),
            "output_dir": str(self.out_dir),
            "frozen_at_utc": self.freeze_at,
            "catchup_anchor_utc": self.catchup_anchor,
            "last_refresh": last_refresh,
            "last_error": last_error,
            "summary": summary,
            "latest_windows": latest,
            "capacity_scan_offset": self.capacity.offset,
            "capacity_scan_initialized": self.capacity.initialized,
        }

    def refresh(self):
        try:
            now = pd.Timestamp.now(tz="UTC")

            shadow = pd.read_csv(self.shadow_records)
            if "ticker" not in shadow.columns or "entry_fill_qty" not in shadow.columns:
                raise RuntimeError("Primary shadow record schema missing ticker/entry_fill_qty.")

            self.capacity.sync_shadow(shadow)
            self.capacity.scan_new()
            cap = self.capacity.capacity_frame(shadow)

            base_contracts = pd.read_csv(self.base_contract_file)
            base_windows = pd.read_csv(self.base_window_file)

            for df in (base_contracts, base_windows):
                if "decision_time" in df.columns:
                    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce")

            for c in ("strategy_eval", "range_flagged", "high_breadth"):
                if c in base_contracts.columns:
                    base_contracts[c] = _bool_col(base_contracts[c])
                if c in base_windows.columns:
                    base_windows[c] = _bool_col(base_windows[c])

            needed = {"ticker", "decision_time", "strategy_eval", "range_flagged", "signal_edge"}
            missing = needed - set(base_contracts.columns)
            if missing:
                raise RuntimeError(f"Base Range44 contract detail missing columns: {sorted(missing)}")

            contracts = base_contracts.merge(cap, on="ticker", how="left")
            for c in ("actual_q3_fill", "flow_capacity_ct", "scalable_capacity_ct", "signal_edge"):
                contracts[c] = pd.to_numeric(contracts.get(c), errors="coerce")

            contracts["actual_q3_fill"] = contracts["actual_q3_fill"].fillna(0.0)
            contracts["flow_capacity_ct"] = contracts["flow_capacity_ct"].fillna(0.0)
            contracts["scalable_capacity_ct"] = contracts["scalable_capacity_ct"].fillna(0.0)

            contracts["strategy_catchup"] = contracts["strategy_eval"].fillna(False)
            contracts["strategy_prospective"] = (
                contracts["strategy_catchup"]
                & contracts["decision_time"].gt(self.freeze_at)
            )

            contracts["target_qty_per_asset"] = np.where(
                contracts["range_flagged"].fillna(False),
                FLAGGED_QTY,
                NORMAL_QTY,
            )

            contracts["scaled_fill_qty"] = np.minimum(
                contracts["target_qty_per_asset"],
                contracts["scalable_capacity_ct"],
            )
            contracts["q3_fill_qty_same_sample"] = np.minimum(
                PRIMARY_QTY,
                contracts["scalable_capacity_ct"],
            )

            settled = contracts["signal_edge"].notna()
            contracts["strategy_realized_pnl_all"] = np.where(
                settled,
                contracts["scaled_fill_qty"] * contracts["signal_edge"],
                0.0,
            )
            contracts["q3_realized_pnl_all"] = np.where(
                settled,
                contracts["q3_fill_qty_same_sample"] * contracts["signal_edge"],
                0.0,
            )

            contracts["catchup_accepted_qty"] = np.where(
                contracts["strategy_catchup"], contracts["scaled_fill_qty"], 0.0
            )
            contracts["catchup_realized_pnl"] = np.where(
                contracts["strategy_catchup"], contracts["strategy_realized_pnl_all"], 0.0
            )
            contracts["catchup_q3_realized_pnl"] = np.where(
                contracts["strategy_catchup"], contracts["q3_realized_pnl_all"], 0.0
            )
            contracts["catchup_open_qty"] = np.where(
                contracts["strategy_catchup"] & ~settled, contracts["scaled_fill_qty"], 0.0
            )

            contracts["strategy_accepted_qty"] = np.where(
                contracts["strategy_prospective"], contracts["scaled_fill_qty"], 0.0
            )
            contracts["strategy_realized_pnl"] = np.where(
                contracts["strategy_prospective"], contracts["strategy_realized_pnl_all"], 0.0
            )
            contracts["q3_realized_pnl_same_sample"] = np.where(
                contracts["strategy_prospective"], contracts["q3_realized_pnl_all"], 0.0
            )
            contracts["strategy_open_qty"] = np.where(
                contracts["strategy_prospective"] & ~settled, contracts["scaled_fill_qty"], 0.0
            )

            agg = contracts.groupby("decision_time", as_index=False).agg(
                strategy_filled_contracts=("strategy_accepted_qty", "sum"),
                strategy_open_contracts=("strategy_open_qty", "sum"),
                strategy_realized_pnl=("strategy_realized_pnl", "sum"),
                q3_realized_pnl_same_sample=("q3_realized_pnl_same_sample", "sum"),
                catchup_filled_contracts=("catchup_accepted_qty", "sum"),
                catchup_open_contracts=("catchup_open_qty", "sum"),
                catchup_realized_pnl=("catchup_realized_pnl", "sum"),
                catchup_q3_realized_pnl=("catchup_q3_realized_pnl", "sum"),
            )

            window_cols = [
                c for c in (
                    "decision_time", "signals", "max_mid_range_c", "path_complete_share_pct",
                    "strategy_eval", "high_breadth", "range_flagged", "strategy_decision"
                )
                if c in base_windows.columns
            ]
            windows = base_windows[window_cols].copy()
            windows = windows.merge(agg, on="decision_time", how="left")

            for c in (
                "strategy_filled_contracts", "strategy_open_contracts",
                "strategy_realized_pnl", "q3_realized_pnl_same_sample",
                "catchup_filled_contracts", "catchup_open_contracts",
                "catchup_realized_pnl", "catchup_q3_realized_pnl",
            ):
                windows[c] = pd.to_numeric(windows.get(c), errors="coerce").fillna(0.0)

            windows["strategy_catchup"] = windows["strategy_eval"].fillna(False)
            windows["strategy_prospective"] = (
                windows["strategy_catchup"] & windows["decision_time"].gt(self.freeze_at)
            )
            windows["target_qty_per_asset"] = np.where(
                windows["range_flagged"].fillna(False), FLAGGED_QTY, NORMAL_QTY
            )

            prospective = windows[windows["strategy_prospective"]].sort_values("decision_time").copy()
            p_complete = prospective[prospective["strategy_open_contracts"] <= EPS].copy()

            catchup = windows[windows["strategy_catchup"]].sort_values("decision_time").copy()
            c_complete = catchup[catchup["catchup_open_contracts"] <= EPS].copy()

            c_pro = contracts[contracts["strategy_prospective"]].copy()
            c_cat = contracts[contracts["strategy_catchup"]].copy()

            summary = pd.DataFrame([{
                "strategy": STRATEGY_VERSION,
                "frozen_at_utc": self.freeze_at,
                "catchup_anchor_utc": self.catchup_anchor,
                "prospective_windows": int(len(prospective)),
                "eligible_windows": int(len(prospective)),
                "flagged_windows": int(prospective["range_flagged"].fillna(False).sum()) if len(prospective) else 0,
                "settled_contracts": float(c_pro.loc[c_pro["signal_edge"].notna(), "strategy_accepted_qty"].sum()),
                "open_contracts": float(c_pro["strategy_open_qty"].sum()),
                "strategy_realized_pnl": float(c_pro["strategy_realized_pnl"].sum()),
                "q3_realized_pnl_same_sample": float(c_pro["q3_realized_pnl_same_sample"].sum()),
                "max_drawdown": _max_drawdown(p_complete["strategy_realized_pnl"]),
                "worst_complete_window_pnl": float(p_complete["strategy_realized_pnl"].min()) if len(p_complete) else np.nan,
                "catchup_windows": int(len(catchup)),
                "catchup_flagged_windows": int(catchup["range_flagged"].fillna(False).sum()) if len(catchup) else 0,
                "catchup_settled_contracts": float(c_cat.loc[c_cat["signal_edge"].notna(), "catchup_accepted_qty"].sum()),
                "catchup_open_contracts": float(c_cat["catchup_open_qty"].sum()),
                "catchup_realized_pnl": float(c_cat["catchup_realized_pnl"].sum()),
                "catchup_q3_realized_pnl": float(c_cat["catchup_q3_realized_pnl"].sum()),
                "catchup_max_drawdown": _max_drawdown(c_complete["catchup_realized_pnl"]),
                "catchup_worst_complete_window_pnl": float(c_complete["catchup_realized_pnl"].min()) if len(c_complete) else np.nan,
            }])

            with self.lock:
                self.summary = summary
                self.windows = windows
                self.contracts = contracts
                self.last_refresh = now
                self.last_error = None

            _atomic_csv(summary, self.out_dir / "summary.csv")
            _atomic_csv(windows, self.out_dir / "window_detail.csv")
            _atomic_csv(contracts, self.out_dir / "contract_detail.csv")
            _atomic_csv(cap, self.out_dir / "capacity_detail.csv")
            _atomic_json(
                {k: v for k, v in self._status_payload().items() if k not in {"summary", "latest_windows"}},
                self.out_dir / "status.json",
            )
            return self._status_payload()

        except Exception as exc:
            with self.lock:
                self.last_error = repr(exc)
            raise

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.refresh()
            except Exception as exc:
                print(f"[range44-q15q5] refresh error: {exc!r}")
            self.stop_event.wait(self.interval_sec)

    def status(self):
        return self._status_payload()

    def stop(self):
        self.stop_event.set()


def start_range44_q15q5_monitor(session_dir, interval_sec=INTERVAL_SEC, show=True):
    global _MONITOR, _MONITOR_THREAD

    with _MONITOR_LOCK:
        if _MONITOR is not None and _MONITOR_THREAD is not None and _MONITOR_THREAD.is_alive():
            same = Path(session_dir).resolve() == _MONITOR.session_dir.resolve()
            if same:
                if show:
                    print("RANGE44 Q15/Q5 monitor is already running for this session.")
                return _MONITOR.status()
            raise RuntimeError(
                "RANGE44 Q15/Q5 monitor is already running for a different session. Stop it first."
            )

        monitor = Range44Q15Q5ProspectiveMonitor(session_dir=session_dir, interval_sec=interval_sec)
        first = monitor.refresh()

        thread = threading.Thread(
            target=monitor.loop,
            name="kalshi-range44-q15q5-monitor",
            daemon=True,
        )
        _MONITOR = monitor
        _MONITOR_THREAD = thread
        thread.start()

    if show:
        print("RANGE44 Q15/Q5 monitor STARTED (READ-ONLY)")
        print("Session:", monitor.session_dir)
        print("Prospective frozen at:", monitor.freeze_at)
        print("Historical catch-up anchor:", monitor.catchup_anchor)
        print("Rule: normal Q15; Range44-flagged Q5")
        print("Catch-up is historical/counterfactual and NOT OOS.")

    return first


def range44_q15q5_status(show=True):
    with _MONITOR_LOCK:
        if _MONITOR is None:
            out = {
                "running": False,
                "session_dir": None,
                "frozen_at_utc": None,
                "catchup_anchor_utc": None,
                "last_refresh": None,
                "last_error": None,
                "summary": pd.DataFrame(),
                "latest_windows": pd.DataFrame(),
            }
        else:
            out = _MONITOR.status()

    if show:
        print("RANGE44 Q15/Q5 monitor:", "RUNNING" if out.get("running") else "STOPPED")
        print("Session:", out.get("session_dir"))
        print("Prospective frozen at:", out.get("frozen_at_utc"))
        print("Catch-up anchor:", out.get("catchup_anchor_utc"))
        if out.get("last_error"):
            print("Last error:", out.get("last_error"))
        summary = out.get("summary")
        if isinstance(summary, pd.DataFrame) and len(summary):
            try:
                from IPython.display import display
                display(summary.round(4))
            except Exception:
                print(summary.round(4).to_string(index=False))
    return out


def stop_range44_q15q5_monitor(show=True):
    global _MONITOR, _MONITOR_THREAD

    with _MONITOR_LOCK:
        if _MONITOR is None:
            if show:
                print("RANGE44 Q15/Q5 monitor is not running.")
            return {"running": False}

        monitor = _MONITOR
        thread = _MONITOR_THREAD
        monitor.stop()

    if thread is not None and thread.is_alive():
        thread.join(timeout=max(2.0, monitor.interval_sec + 1.0))

    status = monitor.status()
    status["running"] = False

    with _MONITOR_LOCK:
        _MONITOR = None
        _MONITOR_THREAD = None

    if show:
        print("RANGE44 Q15/Q5 monitor STOPPED")

    return status


__all__ = [
    "STRATEGY_VERSION",
    "NORMAL_QTY",
    "FLAGGED_QTY",
    "Range44Q15Q5ProspectiveMonitor",
    "start_range44_q15q5_monitor",
    "range44_q15q5_status",
    "stop_range44_q15q5_monitor",
]
