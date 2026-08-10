from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .window_toxicity_history import PROJECT_ROOT

PRIMARY_DIR = "PRIMARY_SHADOW_M5_MINUS3C_15S_3CT_HOLD_V1"
COUNTERFACTUAL_DIR = "COUNTERFACTUAL_RISK_CONTROLS_V1"
MONITOR_VERSION = "COUNTERFACTUAL_RISK_CONTROLS_V1"

SCENARIOS = (
    "BASELINE_Q3",
    "HIGH_BREADTH_Q2",
    "HIGH_BREADTH_Q1",
    "MAX_2_FILLED_ASSETS",
    "MAX_3_FILLED_ASSETS",
)

RESEARCH_SLICES = {
    "APR_DISCOVERY": ("2026-04-01", "2026-05-01"),
    "MAY_VALID": ("2026-05-01", "2026-06-01"),
    "JUN1_28_LOCKED": ("2026-06-01", "2026-06-29"),
    "JUN29_JUL3_STRESS": ("2026-06-29", "2026-07-04"),
    "AUG10_LIVE": ("2026-08-10", "2026-08-11"),
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


def _as_utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def _num(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _research_slice(ts):
    if pd.isna(ts):
        return None
    for name, (start, end) in RESEARCH_SLICES.items():
        if pd.Timestamp(start, tz="UTC") <= ts < pd.Timestamp(end, tz="UTC"):
            return name
    return None


def _max_drawdown(pnls: pd.Series) -> float:
    x = pd.to_numeric(pnls, errors="coerce").fillna(0.0).to_numpy(float)
    if len(x) == 0:
        return np.nan
    equity = np.cumsum(x)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    dd = np.r_[0.0, equity] - peaks
    return float(dd.min())


def _scenario_target_qty(window_signals: int, scenario: str) -> float:
    if scenario == "BASELINE_Q3":
        return 3.0
    if scenario == "HIGH_BREADTH_Q2":
        return 2.0 if window_signals >= 3 else 3.0
    if scenario == "HIGH_BREADTH_Q1":
        return 1.0 if window_signals >= 3 else 3.0
    raise ValueError(f"Historical all-fill sizing does not support scenario {scenario}")


def _historical_scenario(signals: pd.DataFrame, scenario: str):
    z = signals.copy()
    z["decision_time"] = _as_utc(z["decision_time"]).dt.floor("min")
    z = z[z["decision_time"].notna() & z["signal_edge"].notna()].copy()
    breadth = z.groupby("decision_time")["ticker"].size().rename("window_signals")
    z = z.merge(breadth, on="decision_time", how="left")
    z["scenario"] = scenario
    z["target_qty"] = z["window_signals"].map(lambda n: _scenario_target_qty(int(n), scenario))
    z["counterfactual_pnl"] = z["target_qty"] * z["signal_edge"]
    z["high_breadth"] = z["window_signals"] >= 3
    z["research_slice"] = z["decision_time"].map(_research_slice)

    windows = z.groupby(["research_slice", "decision_time"], dropna=False).agg(
        signals=("ticker", "size"),
        contracts=("target_qty", "sum"),
        pnl=("counterfactual_pnl", "sum"),
        mean_signal_edge=("signal_edge", "mean"),
    ).reset_index().sort_values("decision_time")
    windows["high_breadth"] = windows["signals"] >= 3
    return z, windows


def run_historical_risk_control_study(regime_study, show=True):
    """Historical all-fill signal-level sizing comparison.

    This is intentionally NOT an execution replay for Apr-Jul. The historical M5 cache has
    signal/settlement information but not the 15-second FIFO trade path needed for asset caps.
    """
    signals = regime_study.get("signals") if isinstance(regime_study, dict) else None
    if signals is None or len(signals) == 0:
        raise ValueError("regime_study must contain non-empty 'signals'.")

    hist = signals.copy()
    hist["decision_time"] = _as_utc(hist["decision_time"])
    hist = hist[
        (hist["decision_time"] >= pd.Timestamp("2026-04-01", tz="UTC"))
        & (hist["decision_time"] < pd.Timestamp("2026-07-04", tz="UTC"))
    ].copy()

    scenario_details = {}
    scenario_windows = {}
    summary_rows = []
    slice_rows = []

    for scenario in ("BASELINE_Q3", "HIGH_BREADTH_Q2", "HIGH_BREADTH_Q1"):
        d, w = _historical_scenario(hist, scenario)
        scenario_details[scenario] = d
        scenario_windows[scenario] = w

        summary_rows.append({
            "scenario": scenario,
            "signals": len(d),
            "windows": d["decision_time"].nunique(),
            "contracts": d["target_qty"].sum(),
            "signal_pnl": d["counterfactual_pnl"].sum(),
            "pnl_per_contract_c": 100.0 * d["counterfactual_pnl"].sum() / d["target_qty"].sum(),
            "high_breadth_pnl": d.loc[d["high_breadth"], "counterfactual_pnl"].sum(),
            "low_breadth_pnl": d.loc[~d["high_breadth"], "counterfactual_pnl"].sum(),
            "worst_window_pnl": w["pnl"].min(),
            "p05_window_pnl": w["pnl"].quantile(0.05),
            "max_drawdown": _max_drawdown(w.sort_values("decision_time")["pnl"]),
        })

        for slice_name, g in w.groupby("research_slice", dropna=False):
            if pd.isna(slice_name):
                continue
            gd = d[d["research_slice"] == slice_name]
            slice_rows.append({
                "scenario": scenario,
                "research_slice": slice_name,
                "windows": len(g),
                "high_breadth_windows": int((g["signals"] >= 3).sum()),
                "contracts": gd["target_qty"].sum(),
                "signal_pnl": gd["counterfactual_pnl"].sum(),
                "pnl_per_contract_c": 100.0 * gd["counterfactual_pnl"].sum() / gd["target_qty"].sum() if gd["target_qty"].sum() else np.nan,
                "high_breadth_pnl": gd.loc[gd["high_breadth"], "counterfactual_pnl"].sum(),
                "low_breadth_pnl": gd.loc[~gd["high_breadth"], "counterfactual_pnl"].sum(),
                "worst_window_pnl": g["pnl"].min(),
                "p05_window_pnl": g["pnl"].quantile(0.05),
                "max_drawdown": _max_drawdown(g.sort_values("decision_time")["pnl"]),
            })

    summary = pd.DataFrame(summary_rows)
    baseline_pnl = float(summary.loc[summary["scenario"] == "BASELINE_Q3", "signal_pnl"].iloc[0])
    baseline_contracts = float(summary.loc[summary["scenario"] == "BASELINE_Q3", "contracts"].iloc[0])
    summary["pnl_change_vs_q3"] = summary["signal_pnl"] - baseline_pnl
    summary["pnl_retained_pct"] = 100.0 * summary["signal_pnl"] / baseline_pnl if baseline_pnl != 0 else np.nan
    summary["contract_reduction_pct"] = 100.0 * (1.0 - summary["contracts"] / baseline_contracts)

    by_slice = pd.DataFrame(slice_rows)
    if len(by_slice):
        base = by_slice[by_slice["scenario"] == "BASELINE_Q3"][
            ["research_slice", "signal_pnl", "max_drawdown", "worst_window_pnl"]
        ].rename(columns={
            "signal_pnl": "baseline_signal_pnl",
            "max_drawdown": "baseline_max_drawdown",
            "worst_window_pnl": "baseline_worst_window_pnl",
        })
        by_slice = by_slice.merge(base, on="research_slice", how="left")
        by_slice["pnl_change_vs_q3"] = by_slice["signal_pnl"] - by_slice["baseline_signal_pnl"]
        by_slice["drawdown_change_vs_q3"] = by_slice["max_drawdown"] - by_slice["baseline_max_drawdown"]
        by_slice["worst_window_change_vs_q3"] = by_slice["worst_window_pnl"] - by_slice["baseline_worst_window_pnl"]

    if show:
        print("=" * 118)
        print("HISTORICAL RISK-CONTROL SIZING STUDY — ALL-FILL SIGNAL COUNTERFACTUAL")
        print("=" * 118)
        print("Apr-Jul historical files do not contain the 15s FIFO path, so MAX_2/MAX_3 asset caps are NOT backfilled here.")
        print("Q3/Q2/Q1 sizing uses only signal breadth known at M5 and settlement outcomes.")
        print("\nOVERALL HISTORICAL COST / RETENTION")
        _display(summary.round(4))
        print("\nBY RESEARCH SLICE")
        _display(by_slice.round(4))

    return {
        "summary": summary,
        "by_slice": by_slice,
        "details": scenario_details,
        "windows": scenario_windows,
    }


def _read_shadow_events(event_path: Path):
    rows = []
    if not event_path.exists():
        return rows
    with event_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                # Active writer may leave a partial final line for a moment.
                continue
    return rows


def _event_time(obj, *keys):
    for key in keys:
        if obj.get(key) is not None:
            ts = _as_utc(obj.get(key))
            if not pd.isna(ts):
                return ts
    return pd.NaT


def _reconstruct_primary_event_state(events):
    posts = {}
    fills = []
    settlements = {}

    for seq, obj in enumerate(events):
        event = str(obj.get("event") or "")
        ticker = obj.get("ticker")
        if not ticker:
            continue
        ticker = str(ticker)

        if event == "ENTRY_POST":
            decision_time = _event_time(obj, "decision_time")
            if pd.isna(decision_time):
                continue
            posts[ticker] = {
                "ticker": ticker,
                "decision_time": decision_time.floor("min"),
                "actual_post_time": _event_time(obj, "actual_post_time", "time"),
                "direction": str(obj.get("direction") or "").upper(),
                "entry_price": _num(obj.get("entry_price")),
                "primary_qty": _num(obj.get("qty"), 3.0),
                "series": ticker.split("-")[0],
            }

        elif event == "ENTRY_FILL":
            ts = _event_time(obj, "fill_time", "time")
            qty = _num(obj.get("fill_qty"), 0.0)
            if not pd.isna(ts) and qty > 0:
                fills.append({
                    "seq": seq,
                    "ticker": ticker,
                    "fill_time": ts,
                    "fill_qty": qty,
                    "reason": obj.get("reason"),
                })

        elif event == "SETTLED":
            result = str(obj.get("result") or "").upper()
            held = str(obj.get("held_direction") or "").upper()
            qty = _num(obj.get("qty"), np.nan)
            final_pnl = _num(obj.get("final_pnl"), np.nan)
            entry_price = _num(obj.get("entry_price"), np.nan)
            if result in {"YES", "NO"} and held in {"YES", "NO"}:
                if np.isfinite(qty) and qty > 0 and np.isfinite(final_pnl):
                    edge = final_pnl / qty
                elif np.isfinite(entry_price):
                    edge = (1.0 - entry_price) if result == held else -entry_price
                else:
                    edge = np.nan
                settlements[ticker] = {
                    "ticker": ticker,
                    "settle_time": _event_time(obj, "time"),
                    "result": result,
                    "held_direction": held,
                    "edge_per_contract": edge,
                }

    fills = pd.DataFrame(fills)
    if len(fills):
        fills = fills.sort_values(["fill_time", "seq", "ticker"]).reset_index(drop=True)
    else:
        fills = pd.DataFrame(columns=["seq", "ticker", "fill_time", "fill_qty", "reason"])
    return posts, fills, settlements


def _prospective_scenario(posts, fills: pd.DataFrame, settlements, scenario: str):
    breadth = {}
    for meta in posts.values():
        window = meta["decision_time"]
        breadth[window] = breadth.get(window, 0) + 1

    if scenario == "BASELINE_Q3":
        hb_qty, asset_cap = 3.0, None
    elif scenario == "HIGH_BREADTH_Q2":
        hb_qty, asset_cap = 2.0, None
    elif scenario == "HIGH_BREADTH_Q1":
        hb_qty, asset_cap = 1.0, None
    elif scenario == "MAX_2_FILLED_ASSETS":
        hb_qty, asset_cap = 3.0, 2
    elif scenario == "MAX_3_FILLED_ASSETS":
        hb_qty, asset_cap = 3.0, 3
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    target = {}
    for ticker, meta in posts.items():
        n = int(breadth.get(meta["decision_time"], 1))
        target[ticker] = hb_qty if n >= 3 else 3.0

    accepted = {ticker: 0.0 for ticker in posts}
    allowed_assets = {}
    accepted_fill_rows = []

    for _, ev in fills.iterrows():
        ticker = ev["ticker"]
        if ticker not in posts:
            continue
        meta = posts[ticker]
        window = meta["decision_time"]
        n = int(breadth.get(window, 1))

        if asset_cap is not None and n >= 3:
            allowed = allowed_assets.setdefault(window, [])
            if ticker not in allowed:
                if len(allowed) >= asset_cap:
                    continue
                allowed.append(ticker)

        remaining = target[ticker] - accepted[ticker]
        if remaining <= 1e-12:
            continue
        q = min(float(ev["fill_qty"]), remaining)
        if q <= 0:
            continue
        accepted[ticker] += q
        accepted_fill_rows.append({
            "scenario": scenario,
            "ticker": ticker,
            "decision_time": window,
            "fill_time": ev["fill_time"],
            "accepted_fill_qty": q,
            "source_fill_qty": float(ev["fill_qty"]),
            "reason": ev.get("reason"),
        })

    detail_rows = []
    for ticker, meta in posts.items():
        q = float(accepted.get(ticker, 0.0))
        settlement = settlements.get(ticker)
        edge = settlement.get("edge_per_contract") if settlement else np.nan
        pnl = q * edge if q > 0 and np.isfinite(edge) else np.nan
        detail_rows.append({
            "scenario": scenario,
            "ticker": ticker,
            "series": meta["series"],
            "decision_time": meta["decision_time"],
            "window_signals": int(breadth.get(meta["decision_time"], 1)),
            "high_breadth": int(breadth.get(meta["decision_time"], 1)) >= 3,
            "target_qty": target[ticker],
            "accepted_qty": q,
            "settled": settlement is not None and np.isfinite(edge),
            "edge_per_contract": edge,
            "pnl": pnl,
        })

    detail = pd.DataFrame(detail_rows)
    accepted_fills = pd.DataFrame(accepted_fill_rows)

    if len(detail):
        settled_detail = detail[detail["settled"]].copy()
        windows = settled_detail.groupby("decision_time").agg(
            signals=("ticker", "size"),
            filled_assets=("accepted_qty", lambda x: int((x > 1e-12).sum())),
            filled_contracts=("accepted_qty", "sum"),
            pnl=("pnl", "sum"),
        ).reset_index().sort_values("decision_time")
    else:
        settled_detail = detail.copy()
        windows = pd.DataFrame(columns=["decision_time", "signals", "filled_assets", "filled_contracts", "pnl"])

    filled = detail[detail["accepted_qty"] > 1e-12] if len(detail) else detail
    settled_filled = filled[filled["settled"]] if len(filled) else filled
    unsettled_filled = filled[~filled["settled"]] if len(filled) else filled

    summary = {
        "scenario": scenario,
        "signals_seen": len(detail),
        "windows_seen": detail["decision_time"].nunique() if len(detail) else 0,
        "filled_assets": len(filled),
        "filled_contracts": filled["accepted_qty"].sum() if len(filled) else 0.0,
        "settled_filled_assets": len(settled_filled),
        "unsettled_filled_assets": len(unsettled_filled),
        "realized_pnl": settled_filled["pnl"].sum() if len(settled_filled) else 0.0,
        "high_breadth_realized_pnl": settled_filled.loc[settled_filled["high_breadth"], "pnl"].sum() if len(settled_filled) else 0.0,
        "low_breadth_realized_pnl": settled_filled.loc[~settled_filled["high_breadth"], "pnl"].sum() if len(settled_filled) else 0.0,
        "worst_settled_window_pnl": windows["pnl"].min() if len(windows) else np.nan,
        "best_settled_window_pnl": windows["pnl"].max() if len(windows) else np.nan,
        "max_drawdown": _max_drawdown(windows["pnl"]) if len(windows) else np.nan,
    }
    return summary, detail, accepted_fills, windows


class CounterfactualRiskMonitor:
    """Read-only monitor over PRIMARY_SHADOW events.

    It never talks to Kalshi for orders, never changes primary shadow state, and rebuilds the
    counterfactuals deterministically from the append-only primary shadow event log.
    """

    def __init__(self, session_dir, interval_sec=5.0):
        self.session_dir = Path(session_dir)
        self.primary_dir = self.session_dir / PRIMARY_DIR
        self.event_path = self.primary_dir / "shadow_events.jsonl"
        if not self.event_path.exists():
            raise FileNotFoundError(f"Missing primary shadow event log: {self.event_path}")

        self.out_dir = self.session_dir / COUNTERFACTUAL_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.interval_sec = float(interval_sec)
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.last_refresh = None
        self.last_error = None
        self.summary = pd.DataFrame()
        self.details = {}
        self.windows = {}
        self.accepted_fills = {}

        meta = {
            "version": MONITOR_VERSION,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "source_event_log": str(self.event_path),
            "read_only": True,
            "scenarios": list(SCENARIOS),
            "high_breadth_definition": "number of valid ENTRY_POST signals sharing the same floored M5 decision_time >= 3",
            "scenario_semantics": {
                "BASELINE_Q3": "3 contracts per posted signal",
                "HIGH_BREADTH_Q2": "3 contracts if breadth<3; otherwise cap accepted fill quantity at 2 per asset",
                "HIGH_BREADTH_Q1": "3 contracts if breadth<3; otherwise cap accepted fill quantity at 1 per asset",
                "MAX_2_FILLED_ASSETS": "Q3; in breadth>=3 windows keep first 2 distinct assets to fill, reject later assets",
                "MAX_3_FILLED_ASSETS": "Q3; in breadth>=3 windows keep first 3 distinct assets to fill, reject later assets",
            },
        }
        (self.out_dir / "hypothesis.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def refresh(self):
        try:
            events = _read_shadow_events(self.event_path)
            posts, fills, settlements = _reconstruct_primary_event_state(events)
            summaries = []
            details = {}
            windows = {}
            accepted_fills = {}

            for scenario in SCENARIOS:
                s, d, a, w = _prospective_scenario(posts, fills, settlements, scenario)
                summaries.append(s)
                details[scenario] = d
                windows[scenario] = w
                accepted_fills[scenario] = a

            summary = pd.DataFrame(summaries)
            if len(summary):
                baseline = float(summary.loc[summary["scenario"] == "BASELINE_Q3", "realized_pnl"].iloc[0])
                summary["realized_pnl_change_vs_baseline"] = summary["realized_pnl"] - baseline

            with self.lock:
                self.summary = summary
                self.details = details
                self.windows = windows
                self.accepted_fills = accepted_fills
                self.last_refresh = pd.Timestamp.now(tz="UTC")
                self.last_error = None

            summary.to_csv(self.out_dir / "summary.csv", index=False)
            all_details = pd.concat(details.values(), ignore_index=True) if details else pd.DataFrame()
            all_windows = pd.concat(
                [w.assign(scenario=k) for k, w in windows.items()], ignore_index=True
            ) if windows else pd.DataFrame()
            all_fills = pd.concat(accepted_fills.values(), ignore_index=True) if accepted_fills else pd.DataFrame()
            all_details.to_csv(self.out_dir / "ticker_detail.csv", index=False)
            all_windows.to_csv(self.out_dir / "window_detail.csv", index=False)
            all_fills.to_csv(self.out_dir / "accepted_fills.csv", index=False)
            return summary
        except Exception as exc:
            with self.lock:
                self.last_error = repr(exc)
            raise

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.refresh()
            except Exception as exc:
                print(f"[counterfactual-risk] refresh error: {exc!r}")
            self.stop_event.wait(self.interval_sec)

    def status(self):
        with self.lock:
            return {
                "running": not self.stop_event.is_set(),
                "session_dir": str(self.session_dir),
                "output_dir": str(self.out_dir),
                "last_refresh": self.last_refresh,
                "last_error": self.last_error,
                "summary": self.summary.copy(),
            }

    def stop(self):
        self.stop_event.set()


def start_counterfactual_risk_monitor(session_dir, interval_sec=5.0, show=True):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is not None and _MONITOR_THREAD is not None and _MONITOR_THREAD.is_alive():
            same = Path(session_dir).resolve() == _MONITOR.session_dir.resolve()
            if same:
                if show:
                    print("Counterfactual risk monitor is already running for this session.")
                    _display(_MONITOR.status()["summary"].round(4))
                return _MONITOR.status()
            raise RuntimeError(
                "A counterfactual risk monitor is already running for a different session. "
                "Stop it before starting another one."
            )

        monitor = CounterfactualRiskMonitor(session_dir, interval_sec=interval_sec)
        first = monitor.refresh()
        thread = threading.Thread(
            target=monitor.loop,
            name="kalshi-counterfactual-risk-monitor",
            daemon=True,
        )
        _MONITOR = monitor
        _MONITOR_THREAD = thread
        thread.start()

    if show:
        print("Counterfactual risk monitor STARTED (READ-ONLY)")
        print("Session:", monitor.session_dir)
        print("Output:", monitor.out_dir)
        _display(first.round(4))
    return monitor.status()


def counterfactual_risk_status(show=True):
    with _MONITOR_LOCK:
        if _MONITOR is None:
            out = {"running": False, "summary": pd.DataFrame(), "last_error": None}
        else:
            out = _MONITOR.status()
    if show:
        print("Counterfactual risk monitor:", "RUNNING" if out.get("running") else "STOPPED")
        if out.get("last_refresh") is not None:
            print("Last refresh:", out["last_refresh"])
        if out.get("last_error"):
            print("Last error:", out["last_error"])
        summary = out.get("summary")
        if isinstance(summary, pd.DataFrame) and len(summary):
            _display(summary.round(4))
    return out


def stop_counterfactual_risk_monitor(show=True):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is None:
            if show:
                print("Counterfactual risk monitor is not running.")
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
        print("Counterfactual risk monitor STOPPED")
        summary = status.get("summary")
        if isinstance(summary, pd.DataFrame) and len(summary):
            _display(summary.round(4))
    return status
