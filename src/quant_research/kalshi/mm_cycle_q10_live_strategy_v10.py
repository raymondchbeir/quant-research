from __future__ import annotations

"""V10 production live runner: V7 execution mechanics + audit-complete recording bundle.

V10 deliberately does NOT change the frozen trading strategy or the V7 execution
mechanics that passed the end-to-end Q1 smoke. It adds recording/provenance and
post-run comparison readiness so a 24h Q10 live run can be replayed against the
same raw public market-data realization after the live process stops.
"""

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v3 as V3
from . import mm_cycle_q10_live_strategy_v4 as V4
from . import mm_cycle_q10_live_strategy_v6 as V6
from . import mm_cycle_q10_live_strategy_v7 as V7

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V10"
EXECUTION_PARENT = V7.LIVE_VERSION
RECORDING_VERSION = "MM_CYCLE_Q10_LIVE_RECORDING_BUNDLE_V1"
COMPARISON_SCHEMA_VERSION = "MM_CYCLE_Q10_LIVE_VS_SHADOW_SCHEMA_V1"
MAX_ACTION_BOOK_AGE_S = V6.MAX_ACTION_BOOK_AGE_S
RECORDING_HEALTH_S = 5.0

RAW_REQUIRED = (
    "raw_capture/book_top3_events.jsonl",
    "raw_capture/trades_event_time.jsonl",
    "raw_capture/market_metadata.jsonl",
)
RAW_OPTIONAL = (
    "raw_capture/ticker_event_time.jsonl",
    "raw_capture/connection_events.jsonl",
    "raw_capture/health.json",
    "raw_capture/session_manifest.json",
    "raw_capture/capture_spec.json",
    "raw_capture/final_summary.json",
)
LIVE_REQUIRED = (
    "events.jsonl",
    "decisions.jsonl",
    "orders.jsonl",
    "queue_positions.jsonl",
    "fills.jsonl",
    "positions.jsonl",
    "pnl_snapshots.jsonl",
    "risk_events.jsonl",
    "final_summary.json",
)
LIVE_OPTIONAL = (
    "m5_liquidations.jsonl",
    "health.json",
    "live_process.log",
    "raw_recorder.log",
)
META_REQUIRED = (
    "process_config.json",
    "run_spec.json",
    "preflight.json",
    "fee_preflight.json",
    "starting_account.json",
    "balance_semantics.json",
    "order_group.json",
    "raw_recorder_start.json",
    "recording_manifest.json",
    "source_provenance.json",
    "frozen_strategy_snapshot.json",
    "live_execution_spec_v10.json",
    "live_shadow_comparison_schema.json",
)


def _sha256_file(path: Path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_source(module):
    try:
        p = Path(module.__file__).resolve()
    except Exception:
        return {"path": None, "sha256": None}
    return {"path": str(p), "sha256": _sha256_file(p)}


def _git_state():
    out = {"head": None, "branch": None, "status_porcelain": None}
    try:
        out["head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(C.PROJECT_ROOT),
            capture_output=True, text=True, timeout=3,
        ).stdout.strip() or None
    except Exception:
        pass
    try:
        out["branch"] = subprocess.run(
            ["git", "branch", "--show-current"], cwd=str(C.PROJECT_ROOT),
            capture_output=True, text=True, timeout=3,
        ).stdout.strip() or None
    except Exception:
        pass
    try:
        out["status_porcelain"] = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(C.PROJECT_ROOT),
            capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
    except Exception:
        pass
    return out


def _file_state(session: Path, rel: str):
    p = session / rel
    try:
        st = p.stat()
        return {
            "path": rel,
            "exists": True,
            "is_file": p.is_file(),
            "size_bytes": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    except Exception:
        return {
            "path": rel,
            "exists": False,
            "is_file": False,
            "size_bytes": 0,
            "mtime_ns": None,
        }


def _artifact_inventory(session: Path):
    session = Path(session).resolve()
    rels = list(dict.fromkeys(
        list(RAW_REQUIRED) + list(RAW_OPTIONAL)
        + list(LIVE_REQUIRED) + list(LIVE_OPTIONAL)
        + list(META_REQUIRED)
    ))
    return {rel: _file_state(session, rel) for rel in rels}


def _frozen_strategy_snapshot():
    try:
        return B.OOS._frozen_spec()
    except Exception as exc:
        return {
            "error": repr(exc),
            "fallback": {
                "universe": list(B.SERIES),
                "window": "M0 <= elapsed < M5",
                "quote_size": B.FULL_Q,
                "entry": "Candidate C: L3-supported side, natural YES spread >=2c, public BBO",
                "after_fill": "stop adding; opposite BBO only until flat",
                "m5": "cross remaining inventory at executable BBO",
            },
        }


def _source_provenance():
    return {
        "time": B._iso(),
        "live_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_version": RECORDING_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "git": _git_state(),
        "sources": {
            "v1": _module_source(B),
            "v3": _module_source(V3),
            "v4": _module_source(V4),
            "v6": _module_source(V6),
            "v7": _module_source(V7),
            "v10": _module_source(sys.modules[__name__]),
        },
        "note": (
            "Source SHA256 values are authoritative for locally executed files. "
            "git HEAD may not include manually materialized branch files."
        ),
    }


def _recording_manifest(session: Path, cfg):
    return {
        "time": B._iso(),
        "session_dir": str(Path(session).resolve()),
        "live_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_version": RECORDING_VERSION,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "mode": cfg.get("mode"),
        "quote_size": cfg.get("quote_size"),
        "runtime_hours": cfg.get("runtime_hours"),
        "raw_recorder": {
            "component": "MM_EVENT_TIME_M0_M5_V5",
            "path": "raw_capture/",
            "purpose": (
                "authoritative public market-data capture for live execution audit "
                "and post-run frozen-shadow replay on the same realization"
            ),
            "required": list(RAW_REQUIRED),
            "optional": list(RAW_OPTIONAL),
        },
        "actual_execution": {
            "engine": EXECUTION_PARENT,
            "required": list(LIVE_REQUIRED),
            "optional": list(LIVE_OPTIONAL),
        },
        "metadata": {"required": list(META_REQUIRED)},
        "postrun_design": {
            "parallel_shadow_during_live": False,
            "reason": (
                "Avoid duplicate high-rate websocket/disk/CPU load during real-money execution. "
                "Replay the frozen shadow offline on this exact raw_capture after shutdown."
            ),
            "comparison_ready_file": "postrun_comparison_ready.json",
            "comparison_function": "verify_recording_bundle(session_dir)",
        },
    }


def _live_execution_spec(cfg):
    return {
        "time": B._iso(),
        "live_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "execution_mechanics_changed_from_v7": False,
        "recording_only_changes_from_v7": True,
        "mode": cfg.get("mode"),
        "real_money": True,
        "universe": list(B.SERIES),
        "quote_size": float(cfg.get("quote_size", np.nan)),
        "runtime_hours": float(cfg.get("runtime_hours", np.nan)),
        "window": "M0 <= CURRENT wall-clock elapsed < M5",
        "entry": "Frozen Candidate C",
        "after_fill": "stop adding inventory; quote only opposite public BBO until flat",
        "m5": "cancel passive; reduce-only IOC flatten",
        "latest_state_coalescing": True,
        "max_sent_order_book_age_s": float(MAX_ACTION_BOOK_AGE_S),
        "cancel_safety": "V7 dual-path fail-closed verification",
        "loss_limit_usd_from_starting_total_equity": float(cfg.get("max_start_loss_usd", np.nan)),
        "raw_recorder": "V5 unchanged",
    }


def _comparison_schema(session: Path):
    session = Path(session).resolve()
    return {
        "time": B._iso(),
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "session_dir": str(session),
        "raw_source": "raw_capture/",
        "actual_sources": {
            "orders": "orders.jsonl",
            "queue": "queue_positions.jsonl",
            "fills": "fills.jsonl",
            "positions": "positions.jsonl",
            "pnl": "pnl_snapshots.jsonl",
            "m5": "m5_liquidations.jsonl",
            "decisions": "decisions.jsonl",
            "risk": "risk_events.jsonl",
            "final": "final_summary.json",
        },
        "shadow_replay": {
            "strategy_snapshot": "frozen_strategy_snapshot.json",
            "must_use_same_raw_capture": True,
            "must_not_tune": True,
            "run_after_live_shutdown": True,
        },
        "comparison_metrics": [
            "eligible decisions/live creates vs shadow quotes",
            "actual first queue position vs displayed/model queue ahead",
            "passive entry fill count/qty/timing",
            "passive exit completion count/qty/timing",
            "actual vs shadow forced M5 residual count/qty",
            "actual maker/taker fees vs shadow fee model",
            "actual account PnL vs reconstructed execution PnL vs shadow PnL",
            "per-series live vs shadow PnL",
            "per-window live vs shadow PnL",
            "fill retention = actual passive fills / shadow passive fills",
            "PnL retention = actual net / shadow net",
            "latency and stale-create blocks",
            "max risk tick gap and operational exceptions",
        ],
    }


def _write_static_bundle(session: Path, cfg):
    session = Path(session).resolve()
    B._atomic(session / "source_provenance.json", _source_provenance())
    B._atomic(session / "frozen_strategy_snapshot.json", _frozen_strategy_snapshot())
    B._atomic(session / "live_execution_spec_v10.json", _live_execution_spec(cfg))
    B._atomic(session / "live_shadow_comparison_schema.json", _comparison_schema(session))
    B._atomic(session / "recording_manifest.json", _recording_manifest(session, cfg))


def _recording_health(session: Path, *, engine_state=None, metrics=None):
    session = Path(session).resolve()
    inv = _artifact_inventory(session)
    required = list(RAW_REQUIRED) + list(LIVE_REQUIRED) + list(META_REQUIRED)
    present = sum(1 for rel in required if inv.get(rel, {}).get("exists"))
    return {
        "time": B._iso(),
        "live_version": LIVE_VERSION,
        "recording_version": RECORDING_VERSION,
        "engine_state": engine_state,
        "required_artifacts_present": present,
        "required_artifacts_total": len(required),
        "inventory": inv,
        "v7_metrics": metrics,
    }


def verify_recording_bundle(session_dir, *, show=True, write_result=True):
    """Read-only local-file verification. Sends no orders and calls no exchange API."""
    session = Path(session_dir).resolve()
    inv = _artifact_inventory(session)

    raw_ok = all(inv[r]["exists"] and inv[r]["size_bytes"] > 0 for r in RAW_REQUIRED)
    live_ok = all(inv[r]["exists"] for r in LIVE_REQUIRED)
    meta_ok = all(inv[r]["exists"] for r in META_REQUIRED)

    final = B._read(session / "final_summary.json", {}) or {}
    stopped = bool(final)
    flat = final.get("flat_verified") is True if stopped else None
    zero_resting = final.get("strategy_resting_orders_zero") is True if stopped else None
    last_error = final.get("last_error") if stopped else None
    clean_shutdown = bool(
        stopped and flat is True and zero_resting is True and last_error in (None, "")
    )

    comparison_ready = bool(raw_ok and live_ok and meta_ok and clean_shutdown)
    out = {
        "time": B._iso(),
        "session_dir": str(session),
        "live_version": LIVE_VERSION,
        "raw_required_ok": raw_ok,
        "live_required_ok": live_ok,
        "metadata_required_ok": meta_ok,
        "stopped_with_final_summary": stopped,
        "flat_verified": flat,
        "strategy_resting_orders_zero": zero_resting,
        "last_error": last_error,
        "shutdown_reason": final.get("shutdown_reason") if stopped else None,
        "comparison_ready": comparison_ready,
        "inventory": inv,
        "next_step": (
            "Replay frozen shadow OFFLINE on this exact raw_capture and reconcile to actual logs."
            if comparison_ready
            else "Repair/understand missing or unsafe artifacts before any live-vs-shadow conclusion."
        ),
        "orders_sent": False,
        "exchange_api_called": False,
    }
    if write_result:
        B._atomic(session / "postrun_comparison_ready.json", out)

    if show:
        print("=" * 100)
        print("V10 RECORDING BUNDLE VERIFY — NO ORDERS / NO EXCHANGE API")
        print("=" * 100)
        print("Session:                  ", session)
        print("Raw public capture:       ", "PASS" if raw_ok else "FAIL")
        print("Actual execution logs:    ", "PASS" if live_ok else "FAIL")
        print("Metadata/provenance:      ", "PASS" if meta_ok else "FAIL")
        print("Final flat verified:      ", flat)
        print("Zero resting orders:      ", zero_resting)
        print("Last error:               ", last_error)
        print("Comparison ready:         ", comparison_ready)
        print("ORDERS SENT:               NO")
        print("EXCHANGE API CALLED:       NO")
    return out


class ProductionRecordingEngine(V7.DualCancelLatestStateEngine):
    """V7 execution engine unchanged; adds only lightweight recording inventories."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v10_last_recording_health = 0.0

    def _v10_metrics(self):
        return {
            "live_version": LIVE_VERSION,
            "execution_parent": EXECUTION_PARENT,
            "recording_version": RECORDING_VERSION,
            "execution_mechanics_changed_from_v7": False,
            "v7_metrics": self._v7_metrics(),
        }

    def _write_recording_health(self, force=False):
        now = time.time()
        if not force and now - self.v10_last_recording_health < RECORDING_HEALTH_S:
            return
        self.v10_last_recording_health = now
        state = (
            "STOPPED" if self.shutdown_started and self.final_path.exists()
            else ("SHUTTING_DOWN" if self.shutdown_started
                  else ("RUNNING" if self.trade_start else "ARMED_WAITING_FULL_WINDOW"))
        )
        B._atomic(
            self.session / "recording_health.json",
            _recording_health(self.session, engine_state=state, metrics=self._v7_metrics()),
        )

    def health(self, force=False):
        super().health(force=force)
        h = B._read(self.health_path, {}) or {}
        h["live_version"] = LIVE_VERSION
        h["execution_parent"] = EXECUTION_PARENT
        h["v10_metrics"] = self._v10_metrics()
        B._atomic(self.health_path, h)
        self._write_recording_health(force=force)

    def shutdown(self, reason):
        if self.shutdown_started:
            return
        super().shutdown(reason)

        summary = B._read(self.final_path, {}) or {}
        summary["live_wrapper_version"] = LIVE_VERSION
        summary["execution_parent"] = EXECUTION_PARENT
        summary["recording_version"] = RECORDING_VERSION
        summary["comparison_schema_version"] = COMPARISON_SCHEMA_VERSION
        summary["v10_metrics"] = self._v10_metrics()
        B._atomic(self.final_path, summary)

        h = B._read(self.health_path, {}) or {}
        h["live_version"] = LIVE_VERSION
        h["execution_parent"] = EXECUTION_PARENT
        h["v10_metrics"] = self._v10_metrics()
        h["summary"] = summary
        B._atomic(self.health_path, h)

        self._write_recording_health(force=True)
        verify_recording_bundle(self.session, show=False, write_result=True)


def backlog_regression_check(session_dir, *, bucket_ms=250, show=True):
    return V7.backlog_regression_check(session_dir, bucket_ms=bucket_ms, show=show)


def _run_process_v10(session, cfg):
    session = Path(session).resolve()
    _write_static_bundle(session, cfg)

    client = B.Q1.LiveClient()
    diag = V3._install(client)
    B._atomic(session / "balance_semantics.json", diag)

    B._start_recorder = V4._start_recorder_fixed
    B.LiveEngine = ProductionRecordingEngine
    B._run_process(session, cfg)


def live_preflight(*, quote_size=B.FULL_Q, runtime_hours=B.FULL_HOURS,
                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                   min_start_equity_usd=None, show=True):
    return V3.live_preflight(
        quote_size=quote_size,
        runtime_hours=runtime_hours,
        max_start_loss_usd=max_start_loss_usd,
        min_start_equity_usd=min_start_equity_usd,
        show=show,
    )


def _launch(*, mode, q, hours, max_loss, min_equity, arm, expected):
    if str(arm) != expected:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected!r} exactly.")

    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    V3._calibrated_preflight(
        quote_size=q,
        runtime_hours=hours,
        max_loss_usd=max_loss,
        min_equity_usd=min_equity,
        mode=mode,
        save_dir=None,
        show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_{mode.lower()}_v10").resolve()
    session.mkdir(parents=True, exist_ok=False)

    cfg = {
        "mode": mode,
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "engine_architecture": "V7_EXECUTION_UNCHANGED_PLUS_V10_AUDIT_BUNDLE",
        "max_action_book_age_s": MAX_ACTION_BOOK_AGE_S,
        "recording_version": RECORDING_VERSION,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
    }
    B._atomic(session / "process_config.json", cfg)
    _write_static_bundle(session, cfg)

    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v10",
                "--run-live-session", str(session),
                "--config", str(session / "process_config.json"),
            ],
            cwd=str(C.PROJECT_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        fh.close()

    B._atomic(B.CONTROL_PATH, {
        "live_version": LIVE_VERSION,
        "execution_parent": EXECUTION_PARENT,
        "recording_version": RECORDING_VERSION,
        "running": True,
        "pid": p.pid,
        "session_dir": str(session),
        "mode": mode,
        "started_at": B._iso(),
        "config": cfg,
        "log_path": str(log),
    })

    deadline = time.time() + 90.0
    last = {}
    while time.time() < deadline:
        if p.poll() is not None:
            tail = log.read_text(encoding="utf-8")[-16000:] if log.exists() else ""
            raise RuntimeError(f"Live V10 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-16000:] if log.exists() else ""
        raise RuntimeError(f"Live V10 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V10 PROCESS ARMED")
    print("  mode:       ", mode)
    print("  session:    ", session)
    print("  pid:        ", p.pid)
    print(f"  Q:           {q:g} per eligible market")
    print(f"  kill:        -${max_loss:.2f} from calibrated starting TOTAL account equity")
    print(f"  stale cap:   {MAX_ACTION_BOOK_AGE_S:.2f}s max latest-book age at CREATE")
    print("  execution:   V7 mechanics unchanged")
    print("  recording:   V5 raw + actual live execution + V10 provenance/comparison bundle")
    print("  shadow:      replay OFFLINE on this exact raw_capture after shutdown")
    print("Use live_status(); emergency stop is kill_and_flatten_live(arm_phrase='KILL_AND_FLATTEN').")
    return live_status(show=False)


def start_live_smoke_q1_one_window(*, arm_phrase=None,
                                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                                   min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    return _launch(
        mode="SMOKE_Q1_ONE_WINDOW",
        q=B.SMOKE_Q,
        hours=1.0,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected=B.SMOKE_ARM,
    )


def start_live_cycle_q10(*, arm_phrase=None, runtime_hours=B.FULL_HOURS,
                         max_start_loss_usd=B.LOSS_LIMIT_USD,
                         min_start_equity_usd=B.FULL_MIN_EQUITY):
    if abs(float(runtime_hours) - B.FULL_HOURS) > B.EPS:
        raise RuntimeError("V10 full validation is frozen to exactly 24 hours.")
    return _launch(
        mode="LIVE_Q10_24H",
        q=B.FULL_Q,
        hours=B.FULL_HOURS,
        max_loss=float(max_start_loss_usd),
        min_equity=float(min_start_equity_usd),
        arm=arm_phrase,
        expected=B.FULL_ARM,
    )


def live_status(*, show=True, tail_lines=20):
    return B.live_status(show=show, tail_lines=tail_lines)


def kill_and_flatten_live(*, arm_phrase=None, wait_s=20.0):
    return B.kill_and_flatten_live(arm_phrase=arm_phrase, wait_s=wait_s)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-live-session")
    ap.add_argument("--config")
    ap.add_argument("--verify-recording-session")
    a = ap.parse_args()

    if a.verify_recording_session:
        verify_recording_bundle(a.verify_recording_session, show=True, write_result=True)
        return

    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}
        _run_process_v10(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "LIVE_VERSION",
    "EXECUTION_PARENT",
    "RECORDING_VERSION",
    "COMPARISON_SCHEMA_VERSION",
    "ProductionRecordingEngine",
    "backlog_regression_check",
    "verify_recording_bundle",
    "live_preflight",
    "start_live_smoke_q1_one_window",
    "start_live_cycle_q10",
    "live_status",
    "kill_and_flatten_live",
]
