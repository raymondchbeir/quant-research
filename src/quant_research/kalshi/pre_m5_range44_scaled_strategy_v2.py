from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .pre_m5_range44_scaled_strategy import (
    EPS,
    FLAGGED_QTY,
    NORMAL_QTY,
    PRIMARY_QTY,
    FiniteFlowCapacityTracker,
    _atomic_csv,
    _atomic_json,
    _bool_col,
    _max_drawdown,
    _parse_trade,
    _utc,
)
from .window_regime import PRIMARY_DIR

STRATEGY_VERSION = "RANGE44_Q15_Q5_PROSPECTIVE_V2"
OUTPUT_DIR = STRATEGY_VERSION
BASE_RANGE44_DIR = "RANGE44_Q1_PROSPECTIVE_V1"
INTERVAL_SEC = 10.0

_MONITOR = None
_MONITOR_THREAD = None
_MONITOR_LOCK = threading.RLock()


def _last_complete_line_offset(path: Path) -> int:
    size = path.stat().st_size
    if size <= 0:
        return 0
    with path.open("rb") as f:
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return size
        chunk = min(size, 1024 * 1024)
        f.seek(size - chunk)
        raw = f.read(chunk)
        idx = raw.rfind(b"\n")
        return size - chunk + idx + 1 if idx >= 0 else 0


def _unit_pnl(df: pd.DataFrame) -> pd.Series:
    if "signal_edge" in df.columns:
        out = pd.to_numeric(df["signal_edge"], errors="coerce")
    else:
        out = pd.Series(np.nan, index=df.index, dtype=float)

    missing = out.isna()
    if not missing.any():
        return out

    result_col = "result_final" if "result_final" in df.columns else "result" if "result" in df.columns else None
    if result_col is None or "direction" not in df.columns or "entry_price" not in df.columns:
        return out

    result = df[result_col].astype(str).str.upper()
    direction = df["direction"].astype(str).str.upper()
    entry = pd.to_numeric(df["entry_price"], errors="coerce")
    calc = np.where(result.eq(direction), 1.0 - entry, -entry)
    out.loc[missing] = pd.Series(calc, index=df.index).loc[missing]
    return out


class Range44Q15Q5ProspectiveMonitorV2:
    """Nonblocking Q15/Q5 monitor with separate live and historical capacity clocks."""

    def __init__(self, session_dir, interval_sec=INTERVAL_SEC, output_dir=None):
        self.session_dir = Path(session_dir)
        self.primary_dir = self.session_dir / PRIMARY_DIR
        self.shadow_records = self.primary_dir / "shadow_records.csv"
        self.trades_file = self.session_dir / "trades.jsonl"
        self.base_dir = self.session_dir / BASE_RANGE44_DIR
        self.base_contract_file = self.base_dir / "contract_detail.csv"
        self.base_window_file = self.base_dir / "window_detail.csv"
        self.base_hypothesis_file = self.base_dir / "hypothesis.json"

        for p in (
            self.shadow_records,
            self.trades_file,
            self.base_contract_file,
            self.base_window_file,
            self.base_hypothesis_file,
        ):
            if not p.exists():
                raise FileNotFoundError(p)

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

        base_meta = json.loads(self.base_hypothesis_file.read_text(encoding="utf-8"))
        self.catchup_anchor = _utc(base_meta.get("frozen_at_utc"))
        if pd.isna(self.catchup_anchor):
            raise RuntimeError("Original Range44 freeze is missing.")

        hp = self.out_dir / "hypothesis.json"
        if hp.exists():
            existing = json.loads(hp.read_text(encoding="utf-8"))
            self.freeze_at = _utc(existing.get("frozen_at_utc"))
            self.catchup_trade_end_offset = int(existing.get("catchup_trade_end_offset", 0))
        else:
            self.freeze_at = pd.Timestamp.now(tz="UTC")
            self.catchup_trade_end_offset = _last_complete_line_offset(self.trades_file)
            _atomic_json({
                "version": STRATEGY_VERSION,
                "frozen_at_utc": self.freeze_at.isoformat(),
                "catchup_anchor_utc": self.catchup_anchor.isoformat(),
                "catchup_trade_end_offset": self.catchup_trade_end_offset,
                "session_dir": str(self.session_dir),
                "read_only": True,
                "primary_strategy_unchanged": True,
                "prospective_definition": "decision_time strictly after frozen_at_utc",
                "catchup_definition": "original Range44 freeze < decision_time <= new Q15/Q5 freeze; NOT OOS",
                "rule": {"normal_q": NORMAL_QTY, "flagged_q": FLAGGED_QTY},
                "capacity": "Q3 no fill->0; partial->actual; full->max(3, finite observed flow)",
            }, hp)

        # Live tracker starts at the exact historical snapshot boundary; it never rescans old history.
        self.live_capacity = FiniteFlowCapacityTracker(self.trades_file)
        self.live_capacity.offset = self.catchup_trade_end_offset
        self.live_capacity.initialized = True

        # Historical tracker owns only the pre-freeze snapshot and runs in a separate thread.
        self.catchup_capacity = FiniteFlowCapacityTracker(self.trades_file)
        self.catchup_state = "PENDING"
        self.catchup_error = None
        self.catchup_bytes_scanned = 0
        self.catchup_total_bytes = self.catchup_trade_end_offset
        self.catchup_thread = None

    def _read_shadow(self):
        sh = pd.read_csv(self.shadow_records)
        sh["decision_time"] = pd.to_datetime(sh.get("decision_time"), utc=True, errors="coerce")
        sh["entry_fill_qty"] = pd.to_numeric(sh.get("entry_fill_qty"), errors="coerce").fillna(0.0)
        return sh

    def start_catchup(self):
        if self.catchup_thread is not None and self.catchup_thread.is_alive():
            return
        self.catchup_thread = threading.Thread(target=self._catchup_worker, name="range44-q15q5-catchup", daemon=True)
        self.catchup_thread.start()

    def _catchup_worker(self):
        try:
            with self.lock:
                self.catchup_state = "SCANNING"

            shadow = self._read_shadow()
            old_shadow = shadow[shadow["decision_time"].le(self.freeze_at)].copy()
            self.catchup_capacity.sync_shadow(old_shadow)

            end = self.catchup_trade_end_offset
            with self.trades_file.open("rb") as f:
                while not self.stop_event.is_set() and f.tell() < end:
                    pos = f.tell()
                    raw = f.readline()
                    if not raw:
                        break
                    if f.tell() > end or not raw.endswith(b"\n"):
                        f.seek(pos)
                        break
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except Exception:
                        continue
                    trade = _parse_trade(obj)
                    if trade is not None:
                        self.catchup_capacity._process_trade(trade)
                    if f.tell() - self.catchup_bytes_scanned >= 16 * 1024 * 1024:
                        with self.lock:
                            self.catchup_bytes_scanned = f.tell()
                with self.lock:
                    self.catchup_bytes_scanned = min(f.tell(), end)

            if self.stop_event.is_set():
                return

            with self.lock:
                self.catchup_state = "READY"
            self.refresh()

        except Exception as exc:
            with self.lock:
                self.catchup_state = "ERROR"
                self.catchup_error = repr(exc)

    def _build(self, shadow, live_cap, catch_cap=None):
        base_contracts = pd.read_csv(self.base_contract_file)
        base_windows = pd.read_csv(self.base_window_file)

        for df in (base_contracts, base_windows):
            df["decision_time"] = pd.to_datetime(df.get("decision_time"), utc=True, errors="coerce")
            for c in ("strategy_eval", "range_flagged", "high_breadth"):
                if c in df.columns:
                    df[c] = _bool_col(df[c])

        needed = {"ticker", "decision_time", "strategy_eval", "range_flagged", "direction", "entry_price"}
        missing = needed - set(base_contracts.columns)
        if missing:
            raise RuntimeError(f"Base Range44 contract_detail missing: {sorted(missing)}")

        contracts = base_contracts.copy()
        contracts["unit_pnl"] = _unit_pnl(contracts)
        contracts["strategy_catchup"] = (
            contracts["strategy_eval"].fillna(False)
            & contracts["decision_time"].gt(self.catchup_anchor)
            & contracts["decision_time"].le(self.freeze_at)
        )
        contracts["strategy_prospective"] = (
            contracts["strategy_eval"].fillna(False)
            & contracts["decision_time"].gt(self.freeze_at)
        )
        contracts["target_qty_per_asset"] = np.where(
            contracts["range_flagged"].fillna(False), FLAGGED_QTY, NORMAL_QTY
        )

        live_map = live_cap.set_index("ticker") if len(live_cap) else pd.DataFrame()
        if catch_cap is not None and len(catch_cap):
            catch_map = catch_cap.set_index("ticker")
        else:
            catch_map = pd.DataFrame()

        contracts["live_scalable_capacity_ct"] = contracts["ticker"].map(
            live_map["scalable_capacity_ct"] if len(live_map) else {}
        ).fillna(0.0)
        contracts["catchup_scalable_capacity_ct"] = contracts["ticker"].map(
            catch_map["scalable_capacity_ct"] if len(catch_map) else {}
        ).fillna(0.0)

        contracts["strategy_accepted_qty"] = np.where(
            contracts["strategy_prospective"],
            np.minimum(contracts["target_qty_per_asset"], contracts["live_scalable_capacity_ct"]),
            0.0,
        )
        contracts["q3_accepted_qty_same_sample"] = np.where(
            contracts["strategy_prospective"],
            np.minimum(PRIMARY_QTY, contracts["live_scalable_capacity_ct"]),
            0.0,
        )

        catch_ready = catch_cap is not None
        contracts["catchup_accepted_qty"] = np.where(
            contracts["strategy_catchup"] & catch_ready,
            np.minimum(contracts["target_qty_per_asset"], contracts["catchup_scalable_capacity_ct"]),
            0.0,
        )
        contracts["catchup_q3_qty"] = np.where(
            contracts["strategy_catchup"] & catch_ready,
            np.minimum(PRIMARY_QTY, contracts["catchup_scalable_capacity_ct"]),
            0.0,
        )

        settled = contracts["unit_pnl"].notna()
        contracts["strategy_realized_pnl"] = np.where(settled, contracts["strategy_accepted_qty"] * contracts["unit_pnl"], 0.0)
        contracts["q3_realized_pnl_same_sample"] = np.where(settled, contracts["q3_accepted_qty_same_sample"] * contracts["unit_pnl"], 0.0)
        contracts["strategy_open_qty"] = np.where(~settled, contracts["strategy_accepted_qty"], 0.0)
        contracts["catchup_realized_pnl"] = np.where(settled, contracts["catchup_accepted_qty"] * contracts["unit_pnl"], 0.0)
        contracts["catchup_q3_realized_pnl"] = np.where(settled, contracts["catchup_q3_qty"] * contracts["unit_pnl"], 0.0)
        contracts["catchup_open_qty"] = np.where(~settled, contracts["catchup_accepted_qty"], 0.0)

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

        keep = [c for c in (
            "decision_time", "signals", "max_mid_range_c", "path_complete_share_pct",
            "strategy_eval", "high_breadth", "range_flagged", "strategy_decision"
        ) if c in base_windows.columns]
        windows = base_windows[keep].copy().merge(agg, on="decision_time", how="left")
        for c in agg.columns:
            if c != "decision_time":
                windows[c] = pd.to_numeric(windows[c], errors="coerce").fillna(0.0)

        windows["strategy_catchup"] = (
            windows["strategy_eval"].fillna(False)
            & windows["decision_time"].gt(self.catchup_anchor)
            & windows["decision_time"].le(self.freeze_at)
        )
        windows["strategy_prospective"] = (
            windows["strategy_eval"].fillna(False)
            & windows["decision_time"].gt(self.freeze_at)
        )
        windows["target_qty_per_asset"] = np.where(windows["range_flagged"].fillna(False), FLAGGED_QTY, NORMAL_QTY)
        return contracts, windows

    def refresh(self):
        try:
            now = pd.Timestamp.now(tz="UTC")
            shadow = self._read_shadow()
            live_shadow = shadow[shadow["decision_time"].gt(self.freeze_at)].copy()
            self.live_capacity.sync_shadow(live_shadow)
            self.live_capacity.scan_new()
            live_cap = self.live_capacity.capacity_frame(live_shadow)

            catch_cap = None
            with self.lock:
                catch_state = self.catchup_state
            if catch_state == "READY":
                old_shadow = shadow[shadow["decision_time"].le(self.freeze_at)].copy()
                catch_cap = self.catchup_capacity.capacity_frame(old_shadow)

            contracts, windows = self._build(shadow, live_cap, catch_cap)

            pro = windows[windows["strategy_prospective"]].sort_values("decision_time")
            pro_complete = pro[pro["strategy_open_contracts"] <= EPS]
            cpro = contracts[contracts["strategy_prospective"]]

            cat = windows[windows["strategy_catchup"]].sort_values("decision_time")
            cat_complete = cat[cat["catchup_open_contracts"] <= EPS]
            ccat = contracts[contracts["strategy_catchup"]]

            catch_ready = catch_state == "READY"
            summary = pd.DataFrame([{
                "strategy": STRATEGY_VERSION,
                "frozen_at_utc": self.freeze_at,
                "catchup_anchor_utc": self.catchup_anchor,
                "catchup_state": catch_state,
                "prospective_windows": len(pro),
                "flagged_windows": int(pro["range_flagged"].fillna(False).sum()) if len(pro) else 0,
                "settled_contracts": float(cpro.loc[cpro["unit_pnl"].notna(), "strategy_accepted_qty"].sum()),
                "open_contracts": float(cpro["strategy_open_qty"].sum()),
                "strategy_realized_pnl": float(cpro["strategy_realized_pnl"].sum()),
                "q3_realized_pnl_same_sample": float(cpro["q3_realized_pnl_same_sample"].sum()),
                "max_drawdown": _max_drawdown(pro_complete["strategy_realized_pnl"]),
                "worst_complete_window_pnl": float(pro_complete["strategy_realized_pnl"].min()) if len(pro_complete) else np.nan,
                "catchup_windows": len(cat),
                "catchup_flagged_windows": int(cat["range_flagged"].fillna(False).sum()) if len(cat) else 0,
                "catchup_settled_contracts": float(ccat.loc[ccat["unit_pnl"].notna(), "catchup_accepted_qty"].sum()) if catch_ready else np.nan,
                "catchup_open_contracts": float(ccat["catchup_open_qty"].sum()) if catch_ready else np.nan,
                "catchup_realized_pnl": float(ccat["catchup_realized_pnl"].sum()) if catch_ready else np.nan,
                "catchup_q3_realized_pnl": float(ccat["catchup_q3_realized_pnl"].sum()) if catch_ready else np.nan,
                "catchup_max_drawdown": _max_drawdown(cat_complete["catchup_realized_pnl"]) if catch_ready else np.nan,
                "catchup_worst_complete_window_pnl": float(cat_complete["catchup_realized_pnl"].min()) if catch_ready and len(cat_complete) else np.nan,
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
            _atomic_csv(live_cap, self.out_dir / "live_capacity_detail.csv")
            if catch_cap is not None:
                _atomic_csv(catch_cap, self.out_dir / "catchup_capacity_detail.csv")
            _atomic_json({k: v for k, v in self.status().items() if k not in {"summary", "latest_windows"}}, self.out_dir / "status.json")
            return self.status()
        except Exception as exc:
            with self.lock:
                self.last_error = repr(exc)
            raise

    def status(self):
        with self.lock:
            summary = self.summary.copy()
            windows = self.windows.copy()
            last_refresh = self.last_refresh
            last_error = self.last_error
            catch_state = self.catchup_state
            catch_err = self.catchup_error
            scanned = self.catchup_bytes_scanned
            total = self.catchup_total_bytes

        latest = pd.DataFrame()
        if len(windows):
            z = windows[windows["strategy_prospective"].fillna(False)].sort_values("decision_time").tail(8)
            cols = [c for c in (
                "decision_time", "signals", "max_mid_range_c", "range_flagged", "target_qty_per_asset",
                "strategy_filled_contracts", "strategy_open_contracts", "strategy_realized_pnl", "q3_realized_pnl_same_sample"
            ) if c in z.columns]
            latest = z[cols]

        return {
            "running": not self.stop_event.is_set(),
            "session_dir": str(self.session_dir),
            "output_dir": str(self.out_dir),
            "frozen_at_utc": self.freeze_at,
            "catchup_anchor_utc": self.catchup_anchor,
            "last_refresh": last_refresh,
            "last_error": last_error,
            "catchup_state": catch_state,
            "catchup_error": catch_err,
            "catchup_bytes_scanned": scanned,
            "catchup_total_bytes": total,
            "catchup_progress_pct": (100.0 * scanned / total) if total else 100.0,
            "summary": summary,
            "latest_windows": latest,
        }

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.refresh()
            except Exception as exc:
                print(f"[range44-q15q5-v2] refresh error: {exc!r}")
            self.stop_event.wait(self.interval_sec)

    def stop(self):
        self.stop_event.set()


def start_range44_q15q5_monitor(session_dir, interval_sec=INTERVAL_SEC, show=True):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is not None and _MONITOR_THREAD is not None and _MONITOR_THREAD.is_alive():
            same = Path(session_dir).resolve() == _MONITOR.session_dir.resolve()
            if same:
                if show:
                    print("RANGE44 Q15/Q5 V2 monitor is already running for this session.")
                return _MONITOR.status()
            raise RuntimeError("Q15/Q5 V2 monitor is already running on a different session.")

        monitor = Range44Q15Q5ProspectiveMonitorV2(session_dir, interval_sec=interval_sec)
        # Fast refresh only: live tracker starts at the frozen tail, so no historical scan occurs here.
        first = monitor.refresh()
        thread = threading.Thread(target=monitor.loop, name="kalshi-range44-q15q5-v2", daemon=True)
        _MONITOR = monitor
        _MONITOR_THREAD = thread
        thread.start()
        monitor.start_catchup()

    if show:
        print("RANGE44 Q15/Q5 V2 monitor STARTED (READ-ONLY)")
        print("Session:", monitor.session_dir)
        print("Prospective frozen at:", monitor.freeze_at)
        print("Historical catch-up anchor:", monitor.catchup_anchor)
        print("Live capacity starts at frozen trades-file tail; historical catch-up runs separately.")
    return first


def range44_q15q5_status(show=True):
    with _MONITOR_LOCK:
        if _MONITOR is None:
            out = {"running": False, "session_dir": None, "summary": pd.DataFrame(), "latest_windows": pd.DataFrame(), "catchup_state": "STOPPED"}
        else:
            out = _MONITOR.status()
    if show:
        print("RANGE44 Q15/Q5 V2:", "RUNNING" if out.get("running") else "STOPPED")
        print("Session:", out.get("session_dir"))
        print("Prospective freeze:", out.get("frozen_at_utc"))
        print("Catch-up:", out.get("catchup_state"), f"{out.get('catchup_progress_pct', 0):.1f}%")
        if out.get("last_error"):
            print("Last error:", out.get("last_error"))
        if out.get("catchup_error"):
            print("Catch-up error:", out.get("catchup_error"))
        if isinstance(out.get("summary"), pd.DataFrame) and len(out["summary"]):
            try:
                from IPython.display import display
                display(out["summary"].round(4))
            except Exception:
                print(out["summary"].round(4).to_string(index=False))
    return out


def stop_range44_q15q5_monitor(show=True):
    global _MONITOR, _MONITOR_THREAD
    with _MONITOR_LOCK:
        if _MONITOR is None:
            return {"running": False}
        monitor = _MONITOR
        thread = _MONITOR_THREAD
        monitor.stop()
    if thread is not None and thread.is_alive():
        thread.join(timeout=max(2.0, monitor.interval_sec + 1.0))
    with _MONITOR_LOCK:
        _MONITOR = None
        _MONITOR_THREAD = None
    if show:
        print("RANGE44 Q15/Q5 V2 monitor STOPPED")
    return {"running": False}


__all__ = [
    "STRATEGY_VERSION",
    "start_range44_q15q5_monitor",
    "range44_q15q5_status",
    "stop_range44_q15q5_monitor",
]
