from __future__ import annotations

"""Production entrypoint for the frozen Candidate-C / CYCLE_ALWAYS_EXIT live validation.

Use V3 from the notebook.

V3 keeps V2's fill-aware cancel/reprice safety and adds a startup calibration for
Kalshi's currently inconsistent portfolio_value documentation.  The account must
be flat before arming anyway, so V3 observes GET /portfolio/balance while flat:

- portfolio_value ~= balance  -> portfolio_value already represents TOTAL equity.
- portfolio_value ~= 0        -> portfolio_value represents positions only, so
                                 total equity = balance + portfolio_value.
- anything ambiguous          -> REFUSE TO ARM.

The detected interpretation is frozen for that process and logged.  This avoids
silently double-counting or omitting cash in the $50 start-to-current loss kill.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_cycle_q10_live_strategy_v2 as V2

LIVE_VERSION = "MM_CYCLE_Q10_LIVE_STRATEGY_V3"
_BALANCE_MODE = None
_BALANCE_MODE_DIAGNOSTIC = None


def _raw_positions(client):
    body, timing = client.get(
        "/portfolio/positions",
        params={"count_filter": "position", "limit": 1000, "subaccount": 0},
    )
    return body.get("market_positions") or [], timing


def _detect_balance_mode(client, *, tolerance_usd=0.02):
    """Detect portfolio_value semantics only while the account is flat."""
    global _BALANCE_MODE, _BALANCE_MODE_DIAGNOSTIC

    body, timing = client.get("/portfolio/balance", params={"subaccount": 0})
    rows, pt = _raw_positions(client)
    nonzero = [r for r in rows if abs(B._f(r.get("position_fp"), 0.0)) > B.EPS]
    if nonzero:
        raise RuntimeError(
            "Cannot calibrate portfolio_value semantics with open positions. "
            "Live validation requires a flat account anyway."
        )

    cash = B._f(body.get("balance"), np.nan) / 100.0
    pv = B._f(body.get("portfolio_value"), np.nan) / 100.0
    if not np.isfinite(cash) or not np.isfinite(pv):
        raise RuntimeError(f"Invalid /portfolio/balance response: {body}")

    tol = max(float(tolerance_usd), 0.005)
    if abs(pv - cash) <= tol:
        mode = "PORTFOLIO_VALUE_IS_TOTAL"
    elif abs(pv) <= tol:
        mode = "PORTFOLIO_VALUE_IS_POSITIONS_ONLY"
    else:
        raise RuntimeError(
            "Ambiguous Kalshi portfolio_value semantics while flat: "
            f"balance=${cash:.4f}, portfolio_value=${pv:.4f}. "
            "Refusing to arm a real-money loss limit until this is resolved."
        )

    _BALANCE_MODE = mode
    _BALANCE_MODE_DIAGNOSTIC = {
        "time": B._iso(),
        "mode": mode,
        "flat_balance_usd": cash,
        "flat_portfolio_value_usd": pv,
        "tolerance_usd": tol,
        "balance_response": body,
        "positions": rows,
        "timing": {"balance": timing, "positions": pt},
    }
    return dict(_BALANCE_MODE_DIAGNOSTIC)


def _equity_calibrated(body):
    cash = B._f((body or {}).get("balance"), np.nan) / 100.0
    pv = B._f((body or {}).get("portfolio_value"), np.nan) / 100.0
    if not np.isfinite(cash) or not np.isfinite(pv):
        raise RuntimeError(f"Invalid /portfolio/balance response: {body}")
    if _BALANCE_MODE == "PORTFOLIO_VALUE_IS_TOTAL":
        total = pv
    elif _BALANCE_MODE == "PORTFOLIO_VALUE_IS_POSITIONS_ONLY":
        total = cash + pv
    else:
        raise RuntimeError("Balance semantics were not calibrated before equity evaluation.")
    return {
        "cash_balance_usd": cash,
        "portfolio_value_field_usd": pv,
        "equity_usd": total,
        "portfolio_value_semantics": _BALANCE_MODE,
        "updated_ts": (body or {}).get("updated_ts"),
    }


def _install(client=None):
    """Install V2 race handling plus V3 calibrated equity interpretation."""
    B.LiveEngine = V2.SafeLiveEngine
    B._equity = _equity_calibrated
    if client is not None:
        return _detect_balance_mode(client)
    return _BALANCE_MODE_DIAGNOSTIC


def _calibrated_preflight(*, quote_size, runtime_hours, max_loss_usd,
                          min_equity_usd, mode, save_dir=None, show=True):
    client = B.Q1.LiveClient()
    diag = _install(client)
    report = B._preflight(
        quote_size=float(quote_size),
        runtime_hours=float(runtime_hours),
        max_loss_usd=float(max_loss_usd),
        min_equity_usd=float(min_equity_usd),
        mode=str(mode),
        save_dir=save_dir,
        show=show,
    )
    report["balance_semantics"] = diag
    if save_dir is not None:
        B._atomic(Path(save_dir) / "balance_semantics.json", diag)
        # Rewrite preflight so the calibration is durable in the same artifact.
        B._atomic(Path(save_dir) / "preflight.json", report)
    if show:
        print("  balance semantics:", diag["mode"])
        print(
            "  flat calibration:  "
            f"balance=${diag['flat_balance_usd']:.2f} | "
            f"portfolio_value=${diag['flat_portfolio_value_usd']:.2f}"
        )
    return report


def live_preflight(*, quote_size=B.FULL_Q, runtime_hours=B.FULL_HOURS,
                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                   min_start_equity_usd=None, show=True):
    """READ ONLY.  Sends no orders and refuses ambiguous balance semantics."""
    q = float(quote_size)
    minimum = min_start_equity_usd
    if minimum is None:
        minimum = B.FULL_MIN_EQUITY if q > 1 else B.SMOKE_MIN_EQUITY
    return _calibrated_preflight(
        quote_size=q,
        runtime_hours=float(runtime_hours),
        max_loss_usd=float(max_start_loss_usd),
        min_equity_usd=float(minimum),
        mode="PREFLIGHT_ONLY_V3",
        save_dir=None,
        show=show,
    )


def _run_process_v3(session, cfg):
    session = Path(session).resolve()
    client = B.Q1.LiveClient()
    diag = _install(client)
    B._atomic(session / "balance_semantics.json", diag)

    # B._run_process performs its own second clean-account/fee/API preflight and
    # creates the order group only after that passes.  Its LiveEngine reference is
    # patched to V2.SafeLiveEngine, and its equity helper is patched to V3.
    B._run_process(session, cfg)


def _launch(*, mode, q, hours, max_loss, min_equity, arm, expected):
    if str(arm) != expected:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={expected!r} exactly.")
    old = B._ctl()
    if old and B._pid_alive(old.get("pid")):
        raise RuntimeError(f"A live process is already running: {old}")

    # Parent preflight before creating the detached real-money process.
    _calibrated_preflight(
        quote_size=q, runtime_hours=hours, max_loss_usd=max_loss,
        min_equity_usd=min_equity, mode=mode, save_dir=None, show=True,
    )

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    session = (B.ROOT / f"{stamp}_{mode.lower()}_v3").resolve()
    session.mkdir(parents=True, exist_ok=False)
    cfg = {
        "mode": mode,
        "quote_size": float(q),
        "runtime_hours": float(hours),
        "max_start_loss_usd": float(max_loss),
        "min_start_equity_usd": float(min_equity),
        "live_wrapper_version": LIVE_VERSION,
    }
    cfg_path = session / "process_config.json"
    B._atomic(cfg_path, cfg)
    log = session / "live_process.log"
    fh = log.open("a", buffering=1, encoding="utf-8")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m", "quant_research.kalshi.mm_cycle_q10_live_strategy_v3",
                "--run-live-session", str(session),
                "--config", str(cfg_path),
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
            tail = log.read_text(encoding="utf-8")[-8000:] if log.exists() else ""
            raise RuntimeError(f"Live V3 process exited during startup rc={p.returncode}\n{tail}")
        last = B._read(session / "health.json", {}) or {}
        if last.get("state") in {"ARMED_WAITING_FULL_WINDOW", "RUNNING"}:
            break
        time.sleep(0.5)
    else:
        tail = log.read_text(encoding="utf-8")[-8000:] if log.exists() else ""
        raise RuntimeError(f"Live V3 startup timeout. Last health={last}\n{tail}")

    print("\nLIVE V3 PROCESS ARMED")
    print("  mode:   ", mode)
    print("  session:", session)
    print("  pid:    ", p.pid)
    print(f"  Q:       {q:g} per eligible market")
    print(f"  kill:    -${max_loss:.2f} from calibrated starting TOTAL account equity")
    print("Use live_status(); emergency stop is kill_and_flatten_live(arm_phrase='KILL_AND_FLATTEN').")
    return live_status(show=False)


def start_live_smoke_q1_one_window(*, arm_phrase=None,
                                   max_start_loss_usd=B.LOSS_LIMIT_USD,
                                   min_start_equity_usd=B.SMOKE_MIN_EQUITY):
    """REAL ORDERS: Q1 per eligible frozen-universe market, one synchronized M0-M5 window."""
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
    """REAL ORDERS: exact frozen Q10, exactly 24h from first complete M0 window."""
    if abs(float(runtime_hours) - B.FULL_HOURS) > B.EPS:
        raise RuntimeError("V3 full validation is frozen to exactly 24 hours.")
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
    a = ap.parse_args()
    if a.run_live_session:
        cfg = B._read(Path(a.config), {}) or {}
        _run_process_v3(Path(a.run_live_session), cfg)
    else:
        live_status(show=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "live_preflight",
    "start_live_smoke_q1_one_window",
    "start_live_cycle_q10",
    "live_status",
    "kill_and_flatten_live",
]
