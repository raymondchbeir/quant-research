from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGY_VERSION = "M1M5_RISK_Q1_V1"
OUTPUT_DIR = "M1M5_RISK_Q1_V1"
PRE_M5_DIR = "PRE_M5_PROSPECTIVE_RISK_V1"

# Frozen development rule derived from the pre-freeze Aug-10 forensic comparison.
# These thresholds were rounded from the separation between the observed bad-window
# and normal-window means; they were NOT selected by a PnL grid search.
PATH_LENGTH_THRESHOLD_C = 180.0
MID_RANGE_THRESHOLD_C = 38.0
MID_RV_THRESHOLD_C = 26.0
DOMINANT_MOVE_SHARE_THRESHOLD = 0.999
RISK_VOTES_REQUIRED = 2
HIGH_BREADTH_MIN_SIGNALS = 3
NORMAL_QTY = 3.0
FLAGGED_QTY = 1.0
MIN_PATH_COVERAGE_PCT = 75.0

_MONITOR = None
_MONITOR_THREAD = None
_MONITOR_LOCK = threading.RLock()


def _utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


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


def _safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _risk_rule(window_row):
    signals = int(pd.to_numeric(pd.Series([window_row.get("signals")]), errors="coerce").fillna(0).iloc[0])
    high_breadth = signals >= HIGH_BREADTH_MIN_SIGNALS

    path = pd.to_numeric(pd.Series([window_row.get("max_mid_path_length_c")]), errors="coerce").iloc[0]
    rng = pd.to_numeric(pd.Series([window_row.get("max_mid_range_c")]), errors="coerce").iloc[0]
    rv = pd.to_numeric(pd.Series([window_row.get("max_mid_rv_c")]), errors="coerce").iloc[0]
    dom = pd.to_numeric(pd.Series([window_row.get("m1_m5_dominant_move_share")]), errors="coerce").iloc[0]
    coverage = pd.to_numeric(pd.Series([window_row.get("path_complete_share_pct")]), errors="coerce").iloc[0]

    metrics_complete = all(np.isfinite(x) for x in (path, rng, rv, dom))
    path_usable = np.isfinite(coverage) and coverage >= MIN_PATH_COVERAGE_PCT

    flags = {
        "path_length_flag": bool(np.isfinite(path) and path >= PATH_LENGTH_THRESHOLD_C),
        "mid_range_flag": bool(np.isfinite(rng) and rng >= MID_RANGE_THRESHOLD_C),
        "mid_rv_flag": bool(np.isfinite(rv) and rv >= MID_RV_THRESHOLD_C),
        "dominant_move_flag": bool(np.isfinite(dom) and dom >= DOMINANT_MOVE_SHARE_THRESHOLD),
    }
    score = int(sum(flags.values()))

    # Low-breadth windows remain Q3 and do not require the risk metrics for a decision.
    # High-breadth windows require usable frozen metrics; otherwise this research
    # counterfactual declares the window unevaluable rather than inventing a fallback.
    if not high_breadth:
        strategy_eligible = True
        flagged = False
        target_qty = NORMAL_QTY
        decision = "Q3_LOW_BREADTH"
    elif not (metrics_complete and path_usable):
        strategy_eligible = False
        flagged = False
        target_qty = np.nan
        decision = "NO_DECISION_DATA"
    else:
        strategy_eligible = True
        flagged = score >= RISK_VOTES_REQUIRED
        target_qty = FLAGGED_QTY if flagged else NORMAL_QTY
        decision = "Q1_FLAGGED" if flagged else "Q3_NOT_FLAGGED"

    return {
        **flags,
        "risk_score": score,
        "high_breadth": high_breadth,
        "strategy_eligible": strategy_eligible,
        "risk_flagged": flagged,
        "target_qty_per_asset": target_qty,
        "strategy_decision": decision,
    }


class PreM5RiskStrategyMonitor:
    """Read-only prospective PnL monitor for M1M5_RISK_Q1_V1.

    Rule: keep Q3 normally. In >=3-signal windows, if at least two of four
    frozen pre-M5 risk conditions fire, cap each asset at Q1. The monitor uses
    the primary Q3 fill path as a subset counterfactual: accepted quantity is
    min(primary filled quantity, strategy target quantity). It never places orders.
    """

    def __init__(self, session_dir, interval_sec=10.0, output_dir=None):
        self.session_dir = Path(session_dir)
        self.pre_dir = self.session_dir / PRE_M5_DIR
        self.window_path = self.pre_dir / "window_metrics.csv"
        self.contract_path = self.pre_dir / "contract_metrics.csv"
        if not self.pre_dir.exists():
            raise FileNotFoundError(
                f"Missing {PRE_M5_DIR}: {self.pre_dir}. Start the pre-M5 prospective monitor first."
            )

        self.out_dir = Path(output_dir) if output_dir else self.session_dir / OUTPUT_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.interval_sec = float(interval_sec)
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.last_refresh = None
        self.last_error = None
        self.summary = pd.DataFrame()
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

        meta = {
            "version": STRATEGY_VERSION,
            "frozen_at_utc": self.freeze_at.isoformat(),
            "session_dir": str(self.session_dir),
            "read_only": True,
            "primary_strategy_unchanged": True,
            "development_counterfactual": True,
            "validation_rule": "Only windows with decision_time strictly after this strategy freeze count prospectively.",
            "rule": {
                "default_qty_per_asset": NORMAL_QTY,
                "high_breadth_min_signals": HIGH_BREADTH_MIN_SIGNALS,
                "risk_votes_required": RISK_VOTES_REQUIRED,
                "flagged_qty_per_asset": FLAGGED_QTY,
                "conditions": {
                    "max_mid_path_length_c_gte": PATH_LENGTH_THRESHOLD_C,
                    "max_mid_range_c_gte": MID_RANGE_THRESHOLD_C,
                    "max_mid_rv_c_gte": MID_RV_THRESHOLD_C,
                    "m1_m5_dominant_move_share_gte": DOMINANT_MOVE_SHARE_THRESHOLD,
                },
                "min_path_coverage_pct": MIN_PATH_COVERAGE_PCT,
            },
            "threshold_origin": (
                "Rounded descriptive separators from the pre-freeze Aug-10 bad-vs-normal forensic table; "
                "not optimized by a PnL grid search. This remains a post-hoc development hypothesis."
            ),
            "execution_counterfactual": (
                "Subset of primary Q3 fills at the same price: accepted_qty=min(primary_fill_qty,target_qty). "
                "This is a read-only sizing counterfactual, not an independent order-placement simulation."
            ),
        }
        _atomic_json(meta, hp)

    def refresh(self):
        try:
            w = _safe_read_csv(self.window_path)
            c = _safe_read_csv(self.contract_path)
            now = pd.Timestamp.now(tz="UTC")

            if len(w) == 0 or len(c) == 0:
                summary = pd.DataFrame([{
                    "strategy": STRATEGY_VERSION,
                    "prospective_windows": 0,
                    "eligible_windows": 0,
                    "flagged_windows": 0,
                    "settled_contracts": 0.0,
                    "open_contracts": 0.0,
                    "strategy_realized_pnl": 0.0,
                    "q3_realized_pnl_same_sample": 0.0,
                    "pnl_change_vs_q3": 0.0,
                    "max_drawdown": 0.0,
                }])
                with self.lock:
                    self.summary, self.windows, self.contracts = summary, pd.DataFrame(), pd.DataFrame()
                    self.last_refresh, self.last_error = now, None
                _atomic_csv(summary, self.out_dir / "summary.csv")
                return self._status_payload(now)

            w["decision_time"] = pd.to_datetime(w["decision_time"], utc=True, errors="coerce")
            c["decision_time"] = pd.to_datetime(c["decision_time"], utc=True, errors="coerce")
            w = w[w["decision_time"].notna()].copy()
            c = c[c["decision_time"].notna()].copy()

            # Strategy's own line in the sand. Earlier rows can be inspected but cannot
            # count toward its prospective PnL.
            w["strategy_prospective"] = w["decision_time"] > self.freeze_at
            rules = w.apply(_risk_rule, axis=1, result_type="expand")
            w = pd.concat([w.reset_index(drop=True), rules.reset_index(drop=True)], axis=1)
            w["strategy_sample"] = np.where(w["strategy_prospective"], "PROSPECTIVE", "BACKFILL")
            w["strategy_eval"] = w["strategy_prospective"] & w["strategy_eligible"]

            decision_cols = [
                "decision_time", "strategy_prospective", "strategy_eval", "strategy_eligible",
                "risk_flagged", "risk_score", "target_qty_per_asset", "strategy_decision",
                "path_length_flag", "mid_range_flag", "mid_rv_flag", "dominant_move_flag",
            ]
            c = c.merge(w[decision_cols], on="decision_time", how="left")
            c["entry_fill_qty"] = pd.to_numeric(c.get("entry_fill_qty"), errors="coerce").fillna(0.0)
            c["signal_edge"] = pd.to_numeric(c.get("signal_edge"), errors="coerce")
            c["target_qty_per_asset"] = pd.to_numeric(c.get("target_qty_per_asset"), errors="coerce")
            c["strategy_accepted_qty"] = np.where(
                c["strategy_eval"].fillna(False),
                np.minimum(c["entry_fill_qty"], c["target_qty_per_asset"]),
                0.0,
            )
            c["q3_accepted_qty_same_sample"] = np.where(
                c["strategy_eval"].fillna(False), c["entry_fill_qty"], 0.0
            )
            settled = c["signal_edge"].notna()
            c["strategy_settled_qty"] = np.where(settled, c["strategy_accepted_qty"], 0.0)
            c["strategy_open_qty"] = np.where(~settled, c["strategy_accepted_qty"], 0.0)
            c["strategy_realized_pnl"] = np.where(
                settled, c["strategy_accepted_qty"] * c["signal_edge"], 0.0
            )
            c["q3_realized_pnl_same_sample"] = np.where(
                settled, c["q3_accepted_qty_same_sample"] * c["signal_edge"], 0.0
            )

            agg = c.groupby("decision_time", as_index=False).agg(
                strategy_filled_contracts=("strategy_accepted_qty", "sum"),
                strategy_settled_contracts=("strategy_settled_qty", "sum"),
                strategy_open_contracts=("strategy_open_qty", "sum"),
                strategy_realized_pnl=("strategy_realized_pnl", "sum"),
                q3_realized_pnl_same_sample=("q3_realized_pnl_same_sample", "sum"),
            )
            w = w.merge(agg, on="decision_time", how="left")
            for col in (
                "strategy_filled_contracts", "strategy_settled_contracts", "strategy_open_contracts",
                "strategy_realized_pnl", "q3_realized_pnl_same_sample",
            ):
                w[col] = pd.to_numeric(w[col], errors="coerce").fillna(0.0)
            w["pnl_change_vs_q3"] = w["strategy_realized_pnl"] - w["q3_realized_pnl_same_sample"]

            e = w[w["strategy_eval"]].sort_values("decision_time").copy()
            complete = e[e["strategy_open_contracts"] <= 1e-12].copy()
            summary = pd.DataFrame([{
                "strategy": STRATEGY_VERSION,
                "frozen_at_utc": self.freeze_at,
                "prospective_windows": int(w["strategy_prospective"].sum()),
                "eligible_windows": int(w["strategy_eval"].sum()),
                "high_breadth_windows": int((w["strategy_eval"] & w["high_breadth"]).sum()),
                "flagged_windows": int((w["strategy_eval"] & w["risk_flagged"]).sum()),
                "settled_contracts": float(c["strategy_settled_qty"].sum()),
                "open_contracts": float(c["strategy_open_qty"].sum()),
                "strategy_realized_pnl": float(c["strategy_realized_pnl"].sum()),
                "q3_realized_pnl_same_sample": float(c["q3_realized_pnl_same_sample"].sum()),
                "pnl_change_vs_q3": float(c["strategy_realized_pnl"].sum() - c["q3_realized_pnl_same_sample"].sum()),
                "max_drawdown": _max_drawdown(complete["strategy_realized_pnl"]),
                "q3_max_drawdown_same_sample": _max_drawdown(complete["q3_realized_pnl_same_sample"]),
                "worst_complete_window_pnl": float(complete["strategy_realized_pnl"].min()) if len(complete) else np.nan,
            }])

            with self.lock:
                self.summary, self.windows, self.contracts = summary, w, c
                self.last_refresh, self.last_error = now, None

            _atomic_csv(summary, self.out_dir / "summary.csv")
            _atomic_csv(w, self.out_dir / "window_detail.csv")
            _atomic_csv(c, self.out_dir / "contract_detail.csv")
            _atomic_json({k: v for k, v in self._status_payload(now).items() if k not in {"summary", "latest_windows"}}, self.out_dir / "status.json")
            return self._status_payload(now)
        except Exception as exc:
            with self.lock:
                self.last_error = repr(exc)
            raise

    def _status_payload(self, now=None):
        with self.lock:
            summary = self.summary.copy()
            windows = self.windows.copy()
            last_refresh = self.last_refresh
            last_error = self.last_error
        now = now or pd.Timestamp.now(tz="UTC")
        latest = pd.DataFrame()
        if len(windows):
            z = windows[windows.get("strategy_prospective", False)].sort_values("decision_time").tail(8)
            cols = [
                "decision_time", "signals", "risk_score", "risk_flagged", "strategy_decision",
                "max_mid_path_length_c", "max_mid_range_c", "max_mid_rv_c",
                "m1_m5_dominant_move_share", "strategy_filled_contracts",
                "strategy_open_contracts", "strategy_realized_pnl",
                "q3_realized_pnl_same_sample", "pnl_change_vs_q3",
            ]
            latest = z[[x for x in cols if x in z.columns]].copy()
        return {
            "running": not self.stop_event.is_set(),
            "session_dir": str(self.session_dir),
            "output_dir": str(self.out_dir),
            "frozen_at_utc": self.freeze_at,
            "last_refresh": last_refresh,
            "last_error": last_error,
            "summary": summary,
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
                print(f"[m1m5-risk-strategy] refresh error: {exc!r}")
            self.stop_event.wait(self.interval_sec)

    def stop(self):
        self.stop_event.set()


def _show_status(status):
    print("M1->M5 RISK STRATEGY COUNTERFACTUAL:", "RUNNING" if status.get("running") else "STOPPED")
    print("Session:", status.get("session_dir"))
    print("Frozen at UTC:", status.get("frozen_at_utc"))
    if status.get("last_error"):
        print("Last error:", status.get("last_error"))
    summary = status.get("summary")
    if isinstance(summary, pd.DataFrame) and len(summary):
        try:
            from IPython.display import display
            display(summary.round(4))
        except Exception:
            print(summary.round(4).to_string(index=False))
    latest = status.get("latest_windows")
    if isinstance(latest, pd.DataFrame) and len(latest):
        print("\nLATEST STRATEGY-PROSPECTIVE WINDOWS")
        try:
            from IPython.display import display
            display(latest.round(4))
        except Exception:
            print(latest.round(4).to_string(index=False))


def start_pre_m5_risk_strategy_monitor(session_dir, interval_sec=10.0, output_dir=None, show=True):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is not None and _MONITOR_THREAD is not None and _MONITOR_THREAD.is_alive():
            same = Path(session_dir).resolve() == _MONITOR.session_dir.resolve()
            if same:
                out = _MONITOR.status()
                if show:
                    print("M1->M5 risk strategy monitor is already running for this session.")
                    _show_status(out)
                return out
            raise RuntimeError("A M1->M5 risk strategy monitor is already running for a different session.")

        monitor = PreM5RiskStrategyMonitor(session_dir, interval_sec=interval_sec, output_dir=output_dir)
        first = monitor.refresh()
        thread = threading.Thread(
            target=monitor.loop,
            name="kalshi-m1m5-risk-strategy",
            daemon=True,
        )
        _MONITOR = monitor
        _MONITOR_THREAD = thread
        thread.start()

    if show:
        print("M1->M5 risk strategy monitor STARTED (READ-ONLY)")
        print(
            f"Rule: Q3 normally; in >=3-signal windows, >=2/4 risk votes -> Q1. "
            f"Thresholds: path>={PATH_LENGTH_THRESHOLD_C:g}c, range>={MID_RANGE_THRESHOLD_C:g}c, "
            f"RV>={MID_RV_THRESHOLD_C:g}c, dominant>={DOMINANT_MOVE_SHARE_THRESHOLD:g}."
        )
        _show_status(first)
    return first


def pre_m5_risk_strategy_status(show=True):
    with _MONITOR_LOCK:
        if _MONITOR is None:
            out = {"running": False, "last_error": None, "summary": pd.DataFrame(), "latest_windows": pd.DataFrame()}
        else:
            out = _MONITOR.status()
    if show:
        _show_status(out)
    return out


def stop_pre_m5_risk_strategy_monitor(show=True):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is None:
            if show:
                print("M1->M5 risk strategy monitor is not running.")
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
    out = monitor.status()
    out["running"] = False

    with _MONITOR_LOCK:
        _MONITOR = None
        _MONITOR_THREAD = None

    if show:
        print("M1->M5 risk strategy monitor STOPPED")
        _show_status(out)
    return out
