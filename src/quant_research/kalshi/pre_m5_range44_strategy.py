from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .pre_m5_path_study import (
    _read_shadow_signals,
    build_contract_paths,
    build_window_paths,
    load_pre_m5_quotes,
)
from .window_regime import PRIMARY_DIR

STRATEGY_VERSION = "RANGE44_Q1_PROSPECTIVE_V1"
OUTPUT_DIR = STRATEGY_VERSION
RANGE_THRESHOLD_C = 44.0
HIGH_BREADTH_MIN_SIGNALS = 3
NORMAL_QTY = 3.0
FLAGGED_QTY = 1.0
MIN_PATH_COVERAGE_PCT = 75.0
MAX_ANCHOR_AGE_SEC = 90.0
READY_DELAY_SEC = 20.0

_MONITOR = None
_MONITOR_THREAD = None
_MONITOR_LOCK = threading.RLock()


def _utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def _num(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _atomic_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _max_drawdown(window_pnl: pd.Series) -> float:
    x = pd.to_numeric(window_pnl, errors="coerce").fillna(0.0).to_numpy(float)
    if len(x) == 0:
        return 0.0
    eq = np.cumsum(x)
    peaks = np.maximum.accumulate(np.r_[0.0, eq])
    dd = np.r_[0.0, eq] - peaks
    return float(dd.min())


def _empty_summary(freeze_at):
    return pd.DataFrame([{
        "strategy": STRATEGY_VERSION,
        "frozen_at_utc": freeze_at,
        "prospective_windows": 0,
        "eligible_windows": 0,
        "high_breadth_windows": 0,
        "flagged_windows": 0,
        "settled_contracts": 0.0,
        "open_contracts": 0.0,
        "strategy_realized_pnl": 0.0,
        "q3_realized_pnl_same_sample": 0.0,
        "pnl_change_vs_q3": 0.0,
        "max_drawdown": 0.0,
        "q3_max_drawdown_same_sample": 0.0,
        "worst_complete_window_pnl": np.nan,
    }])


class Range44ProspectiveMonitor:
    """Read-only prospective sizing counterfactual frozen after development.

    Frozen rule:
      * default Q3 per filled asset;
      * if an M5 window has >=3 valid primary signals AND max M1->M5 midpoint
        range across those signals is >=44c, cap each primary fill at Q1;
      * use the exact primary Q3 fill path as a subset counterfactual;
      * count only M5 decision times strictly after this monitor's freeze time.

    No orders or cancels are placed. Settlement/PnL is an outcome label only.
    """

    def __init__(
        self,
        session_dir,
        interval_sec=10.0,
        output_dir=None,
        max_anchor_age_sec=MAX_ANCHOR_AGE_SEC,
        min_path_coverage_pct=MIN_PATH_COVERAGE_PCT,
        ready_delay_sec=READY_DELAY_SEC,
    ):
        self.session_dir = Path(session_dir)
        self.primary_dir = self.session_dir / PRIMARY_DIR
        self.shadow_records = self.primary_dir / "shadow_records.csv"
        self.ticker_updates = self.session_dir / "ticker_updates.jsonl"
        if not self.ticker_updates.exists():
            raise FileNotFoundError(f"Missing recorder ticker stream: {self.ticker_updates}")

        self.out_dir = Path(output_dir) if output_dir else self.session_dir / OUTPUT_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.interval_sec = float(interval_sec)
        self.max_anchor_age_sec = float(max_anchor_age_sec)
        self.min_path_coverage_pct = float(min_path_coverage_pct)
        self.ready_delay_sec = float(ready_delay_sec)
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.last_refresh = None
        self.last_error = None
        self.summary = _empty_summary(pd.NaT)
        self.windows = pd.DataFrame()
        self.contracts = pd.DataFrame()

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

        self.summary = _empty_summary(self.freeze_at)
        meta = {
            "version": STRATEGY_VERSION,
            "frozen_at_utc": self.freeze_at.isoformat(),
            "session_dir": str(self.session_dir),
            "read_only": True,
            "primary_strategy_unchanged": True,
            "development_complete": True,
            "prospective_definition": "decision_time strictly after frozen_at_utc",
            "rule": {
                "default_qty_per_filled_asset": NORMAL_QTY,
                "high_breadth_min_signals": HIGH_BREADTH_MIN_SIGNALS,
                "max_mid_range_c_gte": RANGE_THRESHOLD_C,
                "flagged_qty_per_filled_asset": FLAGGED_QTY,
                "min_path_coverage_pct": self.min_path_coverage_pct,
                "max_anchor_age_sec": self.max_anchor_age_sec,
                "ready_delay_sec": self.ready_delay_sec,
            },
            "threshold_origin": (
                "44c is a rounded interior threshold from the Aug-10 development range plateau. "
                "Grid, leave-one-window-out, session split, candidate-medoid, and ablation checks "
                "showed range-only matched range+path and range+RV; 44c was frozen only after those checks."
            ),
            "execution_counterfactual": (
                "Subset of primary Q3 fills at the same price: accepted_qty=min(primary_fill_qty,target_qty)."
            ),
        }
        _atomic_json(meta, hp)
        _atomic_csv(self.summary, self.out_dir / "summary.csv")

    def _status_payload(self, now=None):
        with self.lock:
            summary = self.summary.copy()
            windows = self.windows.copy()
            last_refresh = self.last_refresh
            last_error = self.last_error
        latest = pd.DataFrame()
        if len(windows):
            z = windows[windows.get("strategy_prospective", False).fillna(False)].copy()
            if len(z):
                cols = [
                    "decision_time", "signals", "max_mid_range_c", "path_complete_share_pct",
                    "range_flagged", "strategy_decision", "strategy_filled_contracts",
                    "strategy_open_contracts", "strategy_realized_pnl",
                    "q3_realized_pnl_same_sample", "pnl_change_vs_q3",
                ]
                latest = z.sort_values("decision_time").tail(8)
                latest = latest[[c for c in cols if c in latest.columns]]
        return {
            "running": not self.stop_event.is_set(),
            "session_dir": str(self.session_dir),
            "output_dir": str(self.out_dir),
            "frozen_at_utc": self.freeze_at,
            "threshold_c": RANGE_THRESHOLD_C,
            "last_refresh": last_refresh,
            "last_error": last_error,
            "summary": summary,
            "latest_windows": latest,
        }

    def refresh(self):
        try:
            now = pd.Timestamp.now(tz="UTC")
            if not self.shadow_records.exists() or self.shadow_records.stat().st_size == 0:
                summary = _empty_summary(self.freeze_at)
                with self.lock:
                    self.summary = summary
                    self.windows = pd.DataFrame()
                    self.contracts = pd.DataFrame()
                    self.last_refresh = now
                    self.last_error = None
                _atomic_csv(summary, self.out_dir / "summary.csv")
                _atomic_json({k: v for k, v in self._status_payload(now).items() if k not in {"summary", "latest_windows"}}, self.out_dir / "status.json")
                return self._status_payload(now)

            signals, _ = _read_shadow_signals(self.session_dir, settle_missing=False)
            if len(signals) == 0:
                summary = _empty_summary(self.freeze_at)
                with self.lock:
                    self.summary = summary
                    self.windows = pd.DataFrame()
                    self.contracts = pd.DataFrame()
                    self.last_refresh = now
                    self.last_error = None
                _atomic_csv(summary, self.out_dir / "summary.csv")
                return self._status_payload(now)

            quotes = load_pre_m5_quotes(
                self.session_dir,
                signals,
                source="ticker_updates",
                full_book_fallback=False,
                seed_lookback_sec=120,
                show=False,
            )
            contracts = build_contract_paths(
                signals,
                quotes,
                max_anchor_age_sec=self.max_anchor_age_sec,
            )
            windows = build_window_paths(contracts)
            if len(windows) == 0:
                summary = _empty_summary(self.freeze_at)
                with self.lock:
                    self.summary = summary
                    self.windows = windows
                    self.contracts = contracts
                    self.last_refresh = now
                    self.last_error = None
                _atomic_csv(summary, self.out_dir / "summary.csv")
                return self._status_payload(now)

            windows["decision_time"] = pd.to_datetime(windows["decision_time"], utc=True, errors="coerce")
            contracts["decision_time"] = pd.to_datetime(contracts["decision_time"], utc=True, errors="coerce")
            windows = windows[windows["decision_time"].notna()].copy()
            contracts = contracts[contracts["decision_time"].notna()].copy()

            signals_n = pd.to_numeric(windows.get("signals"), errors="coerce").fillna(0)
            rng = pd.to_numeric(windows.get("max_mid_range_c"), errors="coerce")
            coverage = pd.to_numeric(windows.get("path_complete_share_pct"), errors="coerce")
            windows["window_ready"] = now >= (windows["decision_time"] + pd.to_timedelta(self.ready_delay_sec, unit="s"))
            windows["strategy_prospective"] = windows["decision_time"] > self.freeze_at
            windows["high_breadth"] = signals_n >= HIGH_BREADTH_MIN_SIGNALS
            windows["path_usable"] = coverage.ge(self.min_path_coverage_pct) & rng.notna()
            windows["range_flagged"] = windows["high_breadth"] & windows["path_usable"] & rng.ge(RANGE_THRESHOLD_C)
            windows["strategy_eligible"] = windows["window_ready"] & (~windows["high_breadth"] | windows["path_usable"])
            windows["strategy_eval"] = windows["strategy_prospective"] & windows["strategy_eligible"]
            windows["target_qty_per_asset"] = np.where(windows["range_flagged"], FLAGGED_QTY, NORMAL_QTY)
            windows["strategy_decision"] = np.where(
                ~windows["window_ready"], "WAIT_WINDOW",
                np.where(
                    windows["high_breadth"] & ~windows["path_usable"], "NO_DECISION_DATA",
                    np.where(windows["range_flagged"], "Q1_RANGE44", np.where(windows["high_breadth"], "Q3_RANGE_LT44", "Q3_LOW_BREADTH")),
                ),
            )

            decision_cols = [
                "decision_time", "strategy_prospective", "strategy_eval", "high_breadth",
                "range_flagged", "target_qty_per_asset", "strategy_decision",
            ]
            contracts = contracts.merge(windows[decision_cols], on="decision_time", how="left")
            contracts["entry_fill_qty"] = pd.to_numeric(contracts.get("entry_fill_qty"), errors="coerce").fillna(0.0)
            contracts["signal_edge"] = pd.to_numeric(contracts.get("signal_edge"), errors="coerce")
            contracts["target_qty_per_asset"] = pd.to_numeric(contracts.get("target_qty_per_asset"), errors="coerce")
            eval_mask = contracts["strategy_eval"].fillna(False)
            contracts["strategy_accepted_qty"] = np.where(
                eval_mask,
                np.minimum(contracts["entry_fill_qty"], contracts["target_qty_per_asset"]),
                0.0,
            )
            contracts["q3_accepted_qty_same_sample"] = np.where(eval_mask, contracts["entry_fill_qty"], 0.0)
            settled = contracts["signal_edge"].notna()
            contracts["strategy_settled_qty"] = np.where(settled, contracts["strategy_accepted_qty"], 0.0)
            contracts["strategy_open_qty"] = np.where(~settled, contracts["strategy_accepted_qty"], 0.0)
            contracts["strategy_realized_pnl"] = np.where(settled, contracts["strategy_accepted_qty"] * contracts["signal_edge"], 0.0)
            contracts["q3_realized_pnl_same_sample"] = np.where(settled, contracts["q3_accepted_qty_same_sample"] * contracts["signal_edge"], 0.0)

            agg = contracts.groupby("decision_time", as_index=False).agg(
                strategy_filled_contracts=("strategy_accepted_qty", "sum"),
                strategy_settled_contracts=("strategy_settled_qty", "sum"),
                strategy_open_contracts=("strategy_open_qty", "sum"),
                strategy_realized_pnl=("strategy_realized_pnl", "sum"),
                q3_realized_pnl_same_sample=("q3_realized_pnl_same_sample", "sum"),
            )
            windows = windows.merge(agg, on="decision_time", how="left")
            for col in (
                "strategy_filled_contracts", "strategy_settled_contracts", "strategy_open_contracts",
                "strategy_realized_pnl", "q3_realized_pnl_same_sample",
            ):
                windows[col] = pd.to_numeric(windows[col], errors="coerce").fillna(0.0)
            windows["pnl_change_vs_q3"] = windows["strategy_realized_pnl"] - windows["q3_realized_pnl_same_sample"]

            eligible = windows[windows["strategy_eval"]].sort_values("decision_time").copy()
            complete = eligible[eligible["strategy_open_contracts"] <= 1e-12].copy()
            c_eval = contracts[eval_mask].copy()
            summary = pd.DataFrame([{
                "strategy": STRATEGY_VERSION,
                "frozen_at_utc": self.freeze_at,
                "prospective_windows": int(windows["strategy_prospective"].sum()),
                "eligible_windows": int(windows["strategy_eval"].sum()),
                "high_breadth_windows": int((windows["strategy_eval"] & windows["high_breadth"]).sum()),
                "flagged_windows": int((windows["strategy_eval"] & windows["range_flagged"]).sum()),
                "settled_contracts": float(c_eval["strategy_settled_qty"].sum()),
                "open_contracts": float(c_eval["strategy_open_qty"].sum()),
                "strategy_realized_pnl": float(c_eval["strategy_realized_pnl"].sum()),
                "q3_realized_pnl_same_sample": float(c_eval["q3_realized_pnl_same_sample"].sum()),
                "pnl_change_vs_q3": float(c_eval["strategy_realized_pnl"].sum() - c_eval["q3_realized_pnl_same_sample"].sum()),
                "max_drawdown": _max_drawdown(complete["strategy_realized_pnl"]),
                "q3_max_drawdown_same_sample": _max_drawdown(complete["q3_realized_pnl_same_sample"]),
                "worst_complete_window_pnl": float(complete["strategy_realized_pnl"].min()) if len(complete) else np.nan,
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
            _atomic_json({k: v for k, v in self._status_payload(now).items() if k not in {"summary", "latest_windows"}}, self.out_dir / "status.json")
            return self._status_payload(now)
        except Exception as exc:
            with self.lock:
                self.last_error = repr(exc)
            raise

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.refresh()
            except Exception as exc:
                print(f"[range44] refresh error: {exc!r}")
            self.stop_event.wait(self.interval_sec)

    def status(self):
        return self._status_payload()

    def stop(self):
        self.stop_event.set()


def start_range44_prospective_monitor(session_dir, interval_sec=10.0, show=True):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is not None and _MONITOR_THREAD is not None and _MONITOR_THREAD.is_alive():
            same = Path(session_dir).resolve() == _MONITOR.session_dir.resolve()
            if same:
                if show:
                    print("Range44 prospective monitor is already running for this session.")
                return _MONITOR.status()
            raise RuntimeError("Range44 monitor is already running for a different session. Stop it first.")

        monitor = Range44ProspectiveMonitor(session_dir, interval_sec=interval_sec)
        first = monitor.refresh()
        thread = threading.Thread(target=monitor.loop, name="kalshi-range44-prospective-monitor", daemon=True)
        _MONITOR = monitor
        _MONITOR_THREAD = thread
        thread.start()

    if show:
        print("RANGE44_Q1 prospective monitor STARTED (READ-ONLY)")
        print("Session:", monitor.session_dir)
        print("Frozen at:", monitor.freeze_at)
        print("Rule: signals>=3 AND max M1->M5 range>=44c => Q1; otherwise Q3")
    return first


def range44_prospective_status(show=True):
    with _MONITOR_LOCK:
        if _MONITOR is None:
            out = {
                "running": False,
                "session_dir": None,
                "frozen_at_utc": None,
                "threshold_c": RANGE_THRESHOLD_C,
                "last_refresh": None,
                "last_error": None,
                "summary": pd.DataFrame(),
                "latest_windows": pd.DataFrame(),
            }
        else:
            out = _MONITOR.status()
    if show:
        print("Range44 prospective monitor:", "RUNNING" if out.get("running") else "STOPPED")
        print("Session:", out.get("session_dir"))
        print("Frozen at:", out.get("frozen_at_utc"))
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


def stop_range44_prospective_monitor(show=True):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is None:
            if show:
                print("Range44 prospective monitor is not running.")
            return {"running": False}
        monitor = _MONITOR
        thread = _MONITOR_THREAD
        monitor.stop()
    if thread is not None and thread.is_alive():
        thread.join(timeout=max(2.0, monitor.interval_sec + 1.0))
    try:
        monitor.refresh()
    except Exception:
        pass
    status = monitor.status()
    status["running"] = False
    with _MONITOR_LOCK:
        _MONITOR = None
        _MONITOR_THREAD = None
    if show:
        print("Range44 prospective monitor STOPPED")
    return status
