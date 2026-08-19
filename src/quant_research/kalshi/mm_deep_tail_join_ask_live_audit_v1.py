from __future__ import annotations

"""Read-only operational/execution audit for deep-tail JOIN_ASK live sessions.

No exchange API calls. No orders. The audit reads only the completed/current local
session bundle and writes one derived JSON summary below that same session.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B

VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_AUDIT_V1"

REQUIRED_RAW = (
    "raw_capture/book_top3_events.jsonl",
    "raw_capture/trades_event_time.jsonl",
    "raw_capture/market_metadata.jsonl",
)
REQUIRED_LIVE = (
    "events.jsonl",
    "orders.jsonl",
    "pnl_snapshots.jsonl",
    "risk_events.jsonl",
    "deep_tail_strategy_spec.json",
    "deep_tail_latency.jsonl",
    "deep_tail_transitions.jsonl",
    "deep_tail_actual_fills.jsonl",
    "deep_tail_account_audit.jsonl",
    "private_ws_events.jsonl",
)


def _iter(path):
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                x = json.loads(line)
            except Exception:
                continue
            if isinstance(x, dict):
                yield x


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _pct(a):
    x = np.asarray([_f(v) for v in a], dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "median": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "n": int(len(x)),
        "median": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def audit_live_session(session_dir, *, show=True, write=True):
    session = Path(session_dir).resolve()
    if not session.exists():
        raise FileNotFoundError(session)

    missing_raw = [x for x in REQUIRED_RAW if not (session / x).exists()]
    missing_live = [x for x in REQUIRED_LIVE if not (session / x).exists()]

    transitions = list(_iter(session / "deep_tail_transitions.jsonl") or [])
    fills = list(_iter(session / "deep_tail_actual_fills.jsonl") or [])
    latency = list(_iter(session / "deep_tail_latency.jsonl") or [])
    events = list(_iter(session / "events.jsonl") or [])
    risk = list(_iter(session / "risk_events.jsonl") or [])
    audits = list(_iter(session / "deep_tail_account_audit.jsonl") or [])
    pnl = list(_iter(session / "pnl_snapshots.jsonl") or [])

    tc = Counter(str(r.get("event") or "") for r in transitions)
    ec = Counter(str(r.get("event") or "") for r in events)
    rc = Counter(str(r.get("event") or "") for r in risk)

    selected = {str(r.get("ticker")): str(r.get("tail")) for r in transitions if r.get("event") == "TAIL_SELECTED"}
    full = {str(r.get("ticker")): str(r.get("tail")) for r in transitions if r.get("event") == "FULL_ENTRY"}
    exit_posted = {str(r.get("ticker")): str(r.get("tail")) for r in transitions if r.get("event") == "EXIT_POSTED"}
    exit_filled = {str(r.get("ticker")): str(r.get("tail")) for r in transitions if r.get("event") == "EXIT_FILLED"}
    m5 = {str(r.get("ticker")) for r in transitions if r.get("event") == "M5_FINALIZED"}

    private_fill_latency = [
        r.get("exchange_to_local_ms") for r in latency
        if str(r.get("event") or "") == "PRIVATE_FILL_RECEIVED"
    ]
    cancel_latency = [
        r.get("request_to_result_ms") for r in latency
        if str(r.get("event") or "") == "CANCEL_RESULT"
    ]
    create_rtt = []
    for r in latency:
        if str(r.get("event") or "") != "CREATE_ACK":
            continue
        t = r.get("timing") or {}
        create_rtt.append(t.get("rtt_ms"))

    dual_tail = tc.get("CRITICAL_DUAL_TAIL_FILL", 0)
    disabled = [r for r in transitions if r.get("event") == "WINDOW_DISABLED"]
    disabled_reasons = Counter(str(r.get("reason") or "") for r in disabled)

    orphan_events = [r for r in events if str(r.get("reason") or "") == "ORPHAN_RESTING_ORDER"]
    critical_events = [r for r in events if str(r.get("event") or "") == "CRITICAL"]
    errors = [r for r in events if str(r.get("event") or "") == "ERROR"]

    final = B._read(session / "final_summary.json", {}) or {}
    health = B._read(session / "health.json", {}) or {}
    spec = B._read(session / "deep_tail_strategy_spec.json", {}) or {}
    cfg = B._read(session / "process_config.json", {}) or {}

    final_pnl = _f(final.get("account_pnl_usd")) if final else np.nan
    if not np.isfinite(final_pnl) and pnl:
        final_pnl = _f(pnl[-1].get("start_pnl_usd"))

    clean_final = bool(
        final
        and final.get("flat_verified") is True
        and final.get("strategy_resting_orders_zero") is True
        and final.get("last_error") in (None, "")
    )

    # Audit rows themselves are retained because they are the independent REST
    # account-state snapshots produced by the live process.
    latest_audit = next((r for r in reversed(audits) if r.get("kind") == "ACCOUNT_AUDIT"), {})

    out = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "session": str(session),
        "config": cfg,
        "strategy_version": spec.get("version"),
        "completed": bool(final),
        "clean_final": clean_final if final else None,
        "shutdown_reason": final.get("shutdown_reason") if final else None,
        "account_pnl_usd": final_pnl,
        "flat_verified": final.get("flat_verified") if final else None,
        "strategy_resting_orders_zero": final.get("strategy_resting_orders_zero") if final else None,
        "last_error": final.get("last_error") if final else health.get("last_error"),
        "raw_bundle_complete": not missing_raw,
        "live_bundle_complete": not missing_live,
        "missing_raw": missing_raw,
        "missing_live": missing_live,
        "entry_pairs_posted": int(tc.get("ENTRY_PAIR_POSTED", 0)),
        "tails_selected": int(tc.get("TAIL_SELECTED", 0)),
        "full_entries": int(tc.get("FULL_ENTRY", 0)),
        "fixed_exits_posted": int(tc.get("EXIT_POSTED", 0)),
        "fixed_exits_filled": int(tc.get("EXIT_FILLED", 0)),
        "m5_finalized": int(tc.get("M5_FINALIZED", 0)),
        "windows_disabled": int(tc.get("WINDOW_DISABLED", 0)),
        "disabled_reasons": dict(disabled_reasons),
        "dual_tail_fill_critical": int(dual_tail),
        "critical_event_count": int(len(critical_events)),
        "engine_error_count": int(len(errors)),
        "risk_event_counts": dict(rc),
        "private_fill_latency_ms": _pct(private_fill_latency),
        "create_rtt_ms": _pct(create_rtt),
        "cancel_request_to_result_ms": _pct(cancel_latency),
        "actual_fill_log_rows": int(len(fills)),
        "selected_tail_by_ticker": selected,
        "full_entry_tail_by_ticker": full,
        "exit_posted_tail_by_ticker": exit_posted,
        "exit_filled_tail_by_ticker": exit_filled,
        "m5_tickers": sorted(m5),
        "orphan_resting_critical_count": int(len(orphan_events)),
        "latest_account_audit": latest_audit,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
    }

    out["operational_fail"] = bool(
        dual_tail
        or orphan_events
        or critical_events
        or errors
        or (final and not clean_final)
        or missing_raw
    )

    if write:
        B._atomic(session / "deep_tail_postrun_audit_v1.json", out)

    if show:
        print("=" * 100)
        print("DEEP-TAIL LIVE AUDIT V1 — READ ONLY / NO EXCHANGE API")
        print("=" * 100)
        print("Session:                    ", session)
        print("Completed:                  ", bool(final))
        print("Account PnL:                ", f"${final_pnl:+.4f}" if np.isfinite(final_pnl) else "NA")
        print("Flat verified:              ", out["flat_verified"])
        print("Zero strategy resting:      ", out["strategy_resting_orders_zero"])
        print("Entry pairs / selected:     ", out["entry_pairs_posted"], "/", out["tails_selected"])
        print("Full entries:               ", out["full_entries"])
        print("Exit posted / filled:       ", out["fixed_exits_posted"], "/", out["fixed_exits_filled"])
        print("M5 finalized:               ", out["m5_finalized"])
        print("Disabled windows:           ", out["windows_disabled"], out["disabled_reasons"])
        print("Dual-tail criticals:        ", out["dual_tail_fill_critical"])
        print("Engine critical/errors:     ", out["critical_event_count"], "/", out["engine_error_count"])
        print("Raw/live bundle complete:   ", out["raw_bundle_complete"], "/", out["live_bundle_complete"])
        print("Private fill latency ms:    ", out["private_fill_latency_ms"])
        print("Create RTT ms:              ", out["create_rtt_ms"])
        print("Cancel request->result ms:  ", out["cancel_request_to_result_ms"])
        print("Operational fail:           ", out["operational_fail"])
        print("EXCHANGE API CALLED:        NO")
        print("ORDERS SENT:                NO")
    return out


__all__ = ["VERSION", "audit_live_session"]
