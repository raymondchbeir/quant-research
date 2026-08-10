from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .pre_m5_path_study import (
    PRIMARY_DIR,
    _read_shadow_signals,
    build_contract_paths,
    build_window_paths,
    load_pre_m5_quotes,
)

MONITOR_VERSION = "PRE_M5_PROSPECTIVE_RISK_V1"
OUTPUT_DIR = "PRE_M5_PROSPECTIVE_RISK_V1"

# Frozen after the 2026-08-10 August forensic study. No trading thresholds are
# attached to these metrics. Directionality is the only frozen hypothesis.
FROZEN_METRICS = (
    "max_mid_path_length_c",
    "max_mid_range_c",
    "max_mid_rv_c",
    "m1_m5_dominant_move_share",
)

FROZEN_HYPOTHESES = {
    "max_mid_path_length_c": {
        "direction": "higher_is_riskier",
        "meaning": "largest total absolute quote-to-quote midpoint travel among signals in the M1-M5 window",
    },
    "max_mid_range_c": {
        "direction": "higher_is_riskier",
        "meaning": "largest high-low midpoint range among signals in the M1-M5 window",
    },
    "max_mid_rv_c": {
        "direction": "higher_is_riskier",
        "meaning": "largest sqrt(sum squared quote-to-quote midpoint moves) among signals in the M1-M5 window",
    },
    "m1_m5_dominant_move_share": {
        "direction": "higher_is_riskier",
        "meaning": "share of signals whose net M1-M5 midpoint move has the dominant sign in the window",
    },
}

_MONITOR = None
_MONITOR_THREAD = None
_MONITOR_LOCK = threading.RLock()


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


class PreM5ProspectiveRiskMonitor:
    """Read-only prospective logger for the four frozen M1->M5 risk metrics.

    The monitor does not place/cancel orders and does not change the primary shadow.
    It reconstructs each M1->M5 path from recorder ticker BBOs at/before M5, waits
    briefly for the primary M5 batch to finish, and then writes deterministic window
    metrics. Settlement/fill fields are outcome labels only and may update later.
    """

    def __init__(
        self,
        session_dir,
        interval_sec=10.0,
        max_anchor_age_sec=90.0,
        min_path_coverage_pct=75.0,
        ready_delay_sec=20.0,
        output_dir=None,
    ):
        self.session_dir = Path(session_dir)
        self.primary_dir = self.session_dir / PRIMARY_DIR
        self.shadow_records = self.primary_dir / "shadow_records.csv"
        self.ticker_updates = self.session_dir / "ticker_updates.jsonl"
        if not self.shadow_records.exists():
            raise FileNotFoundError(f"Missing primary shadow records: {self.shadow_records}")
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
        self.windows = pd.DataFrame()
        self.contracts = pd.DataFrame()

        hypothesis_path = self.out_dir / "hypothesis.json"
        existing = None
        if hypothesis_path.exists():
            try:
                existing = json.loads(hypothesis_path.read_text(encoding="utf-8"))
            except Exception:
                existing = None

        if isinstance(existing, dict) and existing.get("version") == MONITOR_VERSION:
            self.freeze_at = _utc(existing.get("frozen_at_utc"))
            if pd.isna(self.freeze_at):
                self.freeze_at = pd.Timestamp.now(tz="UTC")
        else:
            self.freeze_at = pd.Timestamp.now(tz="UTC")

        meta = {
            "version": MONITOR_VERSION,
            "frozen_at_utc": self.freeze_at.isoformat(),
            "session_dir": str(self.session_dir),
            "source": "ticker_updates.jsonl BBO observations at/before M5",
            "read_only": True,
            "primary_strategy_unchanged": True,
            "metrics": FROZEN_HYPOTHESES,
            "thresholds": None,
            "threshold_policy": "No thresholds frozen. Test monotonic direction/distribution prospectively before any filter is proposed.",
            "primary_evaluation_population": "high-breadth windows with >=3 valid primary signals",
            "secondary_evaluation_population": "all valid primary windows",
            "prospective_definition": "decision_time strictly after frozen_at_utc",
            "window_ready_definition": f"current UTC time >= decision_time + {self.ready_delay_sec:g}s",
            "path_coverage_requirement_pct": self.min_path_coverage_pct,
            "anchor_max_age_sec": self.max_anchor_age_sec,
            "notes": [
                "Features are reconstructed only from quotes timestamped at/before M5.",
                "Fill, settlement, and actual PnL are labels only and cannot affect feature values.",
                "Rows from before frozen_at_utc are retained as BACKFILL and must not be counted as prospective validation.",
            ],
        }
        _atomic_json(meta, hypothesis_path)

    def refresh(self):
        try:
            signals, _ = _read_shadow_signals(self.session_dir, settle_missing=False)
            if len(signals):
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
            else:
                contracts = pd.DataFrame()
                windows = pd.DataFrame()

            now = pd.Timestamp.now(tz="UTC")
            if len(windows):
                dt = pd.to_datetime(windows["decision_time"], utc=True, errors="coerce")
                windows["window_ready"] = now >= (dt + pd.to_timedelta(self.ready_delay_sec, unit="s"))
                windows["prospective"] = dt > self.freeze_at
                windows["sample"] = np.where(windows["prospective"], "PROSPECTIVE", "BACKFILL")
                windows["high_breadth"] = pd.to_numeric(windows["signals"], errors="coerce") >= 3
                windows["path_usable"] = (
                    pd.to_numeric(windows["path_complete_share_pct"], errors="coerce")
                    >= self.min_path_coverage_pct
                )
                metric_complete = windows[list(FROZEN_METRICS)].apply(
                    lambda s: pd.to_numeric(s, errors="coerce")
                ).notna().all(axis=1)
                windows["frozen_metrics_complete"] = metric_complete
                windows["research_eligible"] = (
                    windows["window_ready"]
                    & windows["prospective"]
                    & windows["path_usable"]
                    & windows["frozen_metrics_complete"]
                )
                windows["primary_eval_eligible"] = windows["research_eligible"] & windows["high_breadth"]

            if len(contracts):
                cdt = pd.to_datetime(contracts["decision_time"], utc=True, errors="coerce")
                contracts["prospective"] = cdt > self.freeze_at
                contracts["sample"] = np.where(contracts["prospective"], "PROSPECTIVE", "BACKFILL")

            ready_windows = windows[windows["window_ready"]].copy() if len(windows) else windows.copy()
            prospective = ready_windows[ready_windows["prospective"]].copy() if len(ready_windows) else ready_windows.copy()

            with self.lock:
                self.windows = ready_windows
                self.contracts = contracts
                self.last_refresh = now
                self.last_error = None

            _atomic_csv(ready_windows, self.out_dir / "window_metrics.csv")
            _atomic_csv(prospective, self.out_dir / "prospective_windows.csv")
            _atomic_csv(contracts, self.out_dir / "contract_metrics.csv")

            status = self._status_payload(now=now)
            _atomic_json({k: v for k, v in status.items() if k != "latest_windows"}, self.out_dir / "status.json")
            return status
        except Exception as exc:
            with self.lock:
                self.last_error = repr(exc)
            raise

    def _status_payload(self, now=None):
        with self.lock:
            w = self.windows.copy()
            last_refresh = self.last_refresh
            last_error = self.last_error
        now = now or pd.Timestamp.now(tz="UTC")
        if len(w):
            prospective = w[w["prospective"]].copy()
            eligible = prospective[prospective["research_eligible"]].copy()
            primary = prospective[prospective["primary_eval_eligible"]].copy()
            latest_cols = [
                "decision_time", "sample", "signals", "filled_assets", "actual_pnl",
                "execution_complete", "path_complete_share_pct",
                "max_mid_path_length_c", "max_mid_range_c", "max_mid_rv_c",
                "m1_m5_dominant_move_share", "research_eligible", "primary_eval_eligible",
            ]
            latest = prospective.sort_values("decision_time").tail(10)
            latest = latest[[c for c in latest_cols if c in latest.columns]]
        else:
            prospective = eligible = primary = pd.DataFrame()
            latest = pd.DataFrame()

        return {
            "running": not self.stop_event.is_set(),
            "session_dir": str(self.session_dir),
            "output_dir": str(self.out_dir),
            "frozen_at_utc": self.freeze_at,
            "last_refresh": last_refresh,
            "last_error": last_error,
            "ready_windows_total": int(len(w)),
            "prospective_windows": int(len(prospective)),
            "prospective_research_eligible": int(len(eligible)),
            "prospective_high_breadth_eligible": int(len(primary)),
            "prospective_execution_complete": int(prospective["execution_complete"].sum()) if len(prospective) else 0,
            "latest_windows": latest,
            "utc_now": now,
        }

    def status(self):
        return self._status_payload()

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.refresh()
            except Exception as exc:
                print(f"[pre-m5-prospective-risk] refresh error: {exc!r}")
            self.stop_event.wait(self.interval_sec)

    def stop(self):
        self.stop_event.set()


def _show_status(status):
    print("PRE-M5 PROSPECTIVE RISK MONITOR:", "RUNNING" if status.get("running") else "STOPPED")
    print("Session:", status.get("session_dir"))
    print("Frozen at UTC:", status.get("frozen_at_utc"))
    print("Last refresh:", status.get("last_refresh"))
    if status.get("last_error"):
        print("Last error:", status.get("last_error"))
    print(
        "Ready windows / prospective / research-eligible / high-breadth eligible:",
        status.get("ready_windows_total", 0), "/",
        status.get("prospective_windows", 0), "/",
        status.get("prospective_research_eligible", 0), "/",
        status.get("prospective_high_breadth_eligible", 0),
    )
    latest = status.get("latest_windows")
    if isinstance(latest, pd.DataFrame) and len(latest):
        print("\nLATEST PROSPECTIVE WINDOWS")
        _display(latest.round(4))


def start_pre_m5_prospective_risk_monitor(
    session_dir,
    interval_sec=10.0,
    max_anchor_age_sec=90.0,
    min_path_coverage_pct=75.0,
    ready_delay_sec=20.0,
    output_dir=None,
    show=True,
):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is not None and _MONITOR_THREAD is not None and _MONITOR_THREAD.is_alive():
            same = Path(session_dir).resolve() == _MONITOR.session_dir.resolve()
            if same:
                status = _MONITOR.status()
                if show:
                    print("Pre-M5 prospective risk monitor is already running for this session.")
                    _show_status(status)
                return status
            raise RuntimeError(
                "A pre-M5 prospective risk monitor is already running for a different session. "
                "Stop it before starting another one."
            )

        monitor = PreM5ProspectiveRiskMonitor(
            session_dir=session_dir,
            interval_sec=interval_sec,
            max_anchor_age_sec=max_anchor_age_sec,
            min_path_coverage_pct=min_path_coverage_pct,
            ready_delay_sec=ready_delay_sec,
            output_dir=output_dir,
        )
        first = monitor.refresh()
        thread = threading.Thread(
            target=monitor.loop,
            name="kalshi-pre-m5-prospective-risk-monitor",
            daemon=True,
        )
        _MONITOR = monitor
        _MONITOR_THREAD = thread
        thread.start()

    if show:
        print("Pre-M5 prospective risk monitor STARTED (READ-ONLY)")
        print("Frozen metrics:", ", ".join(FROZEN_METRICS))
        _show_status(first)
    return first


def pre_m5_prospective_risk_status(show=True):
    with _MONITOR_LOCK:
        if _MONITOR is None:
            out = {
                "running": False,
                "last_error": None,
                "latest_windows": pd.DataFrame(),
            }
        else:
            out = _MONITOR.status()
    if show:
        _show_status(out)
    return out


def stop_pre_m5_prospective_risk_monitor(show=True):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is None:
            if show:
                print("Pre-M5 prospective risk monitor is not running.")
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
        print("Pre-M5 prospective risk monitor STOPPED")
        _show_status(status)
    return status
