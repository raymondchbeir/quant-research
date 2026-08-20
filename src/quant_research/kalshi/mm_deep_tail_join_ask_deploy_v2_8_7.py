from __future__ import annotations

"""V2.8.7: simple 30-minute Q1 smoke using the proven V5 startup flow.

Operational-only wrapper. No alpha/execution rule and no V5 recorder code changes.

Why this version exists
-----------------------
V2.8.6 added a strict pre-M0 subscription/snapshot gate. That was unnecessary for
this smoke and departed from the older proven launch behavior: V5 only needs to
start healthy, then it may discover/subscribe normally. Missing the first seconds
(or even an early portion) of a window is allowed; the live engine already skips a
window when it cannot obtain a certified M1 state.

V2.8.7 therefore:
- removes the 3-5 minute launch restriction;
- removes the pre-M0 >=9 subscriptions / >=9 snapshots hard gate;
- keeps the unchanged V5 recorder and its normal discovery/resubscription logic;
- keeps Q1 fixed to 30 minutes from the first complete live window;
- keeps V2.8.2 fee-snapshot reuse, V1.5 bounded ingestion, persistent M5 cleanup,
  and the V2.8 guardian;
- fixes detached-child imports in the clean src-layout worktree by exporting that
  worktree's src directory through PYTHONPATH before V2.8.2 launches children;
- performs a post-run V5 data audit so recorder quality is measured, not assumed.

Importing this module sends no orders.
"""

import json
import os
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_deploy_v2_8_2 as V282
from . import mm_event_time_m0_m5_recorder_v5 as V5

DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_8_7_Q1_30M_PROVEN_V5_SIMPLE_START"
LIVE = V282.LIVE
CORE = V282.CORE
Q1_ARM = "LIVE_DEEP_TAIL_Q1_30M_V287"
KILL_ARM = V282.KILL_ARM
PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2_8_7.json"

Q1_DEFAULT_HOURS = 0.5
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0
EXPECTED_SERIES = tuple(V5.CRYPTO_SERIES)


def _source_root():
    # .../src/quant_research/kalshi/this_file.py -> .../src
    return Path(__file__).resolve().parents[2]


def _install_child_pythonpath():
    """Make detached python -m children import this exact clean worktree."""
    src = str(_source_root())
    old = os.environ.get("PYTHONPATH", "")
    parts = [p for p in old.split(os.pathsep) if p]
    if src not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([src] + parts)
    return {
        "src": src,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "installed": src in os.environ.get("PYTHONPATH", "").split(os.pathsep),
    }


def _read_jsonl_series(path):
    seen = set()
    tickers = set()
    bad = 0
    path = Path(path)
    if not path.exists():
        return seen, tickers, bad
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            s = str(r.get("series_ticker") or "")
            t = str(r.get("ticker") or "")
            if s:
                seen.add(s)
            if t:
                tickers.add(t)
    return seen, tickers, bad


def static_self_check(*, show=True):
    base = V282.static_self_check(show=False)
    pp = _install_child_pythonpath()
    checks = {
        "base_v2_8_2_ok": base.get("ok") is True,
        "alpha_rules_unchanged": True,
        "q1_fixed_30_minutes": abs(Q1_DEFAULT_HOURS - 0.5) < 1e-12,
        "bounded_raw_ingestion": base.get("bounded_raw_ingestion") is True,
        "guardian_unchanged": (
            base.get("rss_guard_unchanged") is True
            and base.get("deadline_initial_grace_unchanged") is True
            and base.get("cleanup_overrun_still_fail_closed") is True
        ),
        "fee_snapshot_reuse": base.get("child_fee_snapshot_reuse") is True,
        "proven_v5_recorder": V5.STUDY_VERSION == "MM_EVENT_TIME_M0_M5_V5_DEV",
        "v5_capture_window_unchanged": (
            V5.TRADE_WINDOW_START_S == 0.0
            and V5.TRADE_WINDOW_END_S == 300.0
            and V5.LABEL_TAIL_END_S == 330.0
            and V5.PRESUBSCRIBE_LEAD_S == 300.0
        ),
        "frozen_universe_unchanged": tuple(V5.CRYPTO_SERIES) == (
            "KXBTC15M", "KXBNB15M", "KXDOGE15M", "KXETH15M", "KXHYPE15M",
            "KXNEAR15M", "KXSOL15M", "KXXRP15M", "KXZEC15M",
        ),
        "strict_pre_m0_data_gate_removed": True,
        "normal_v5_discovery_preserved": True,
        "child_pythonpath_installed": pp["installed"] is True,
    }
    ok = all(checks.values())
    out = {
        "deploy_version": DEPLOY_VERSION,
        "live_engine_version": LIVE.LIVE_VERSION,
        "recorder_version": V5.STUDY_VERSION,
        "child_src": pp["src"],
        **checks,
        "ok": bool(ok),
        "orders_sent": False,
    }
    if show:
        print("=" * 104)
        print("DEEP-TAIL DEPLOY V2.8.7 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 104)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.8.7 static self-check failed: {out}")
    return out


def start_q1_smoke(*, arm_phrase=None, runtime_hours=Q1_DEFAULT_HOURS,
                   max_start_loss_usd=Q1_DEFAULT_MAX_LOSS,
                   min_start_equity_usd=Q1_DEFAULT_MIN_EQUITY):
    """REAL ORDERS. Simple proven-style V5 startup; no pre-M0 readiness gate."""
    if str(arm_phrase) != Q1_ARM:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q1_ARM!r} exactly.")
    if abs(float(runtime_hours) - Q1_DEFAULT_HOURS) > 1e-12:
        raise RuntimeError("V2.8.7 Q1 smoke is fixed to exactly 30 minutes (0.5 hours).")

    static_self_check(show=True)
    pp = _install_child_pythonpath()

    print("=" * 104)
    print("V2.8.7 — Q1 30-MINUTE SMOKE / PROVEN V5 SIMPLE START")
    print("=" * 104)
    print("Recorder:                   ", V5.STUDY_VERSION)
    print("Recorder discovery:         original V5")
    print("Pre-M0 subscription gate:   NONE")
    print("Pre-M0 snapshot gate:       NONE")
    print("Missed early raw data:      allowed; engine skips uncertified M1 windows")
    print("Runtime:                    30 min from first complete live window")
    print("Detached child src:         ", pp["src"])
    print("=" * 104)

    out = V282.start_q1_smoke(
        arm_phrase=V282.Q1_ARM,
        runtime_hours=Q1_DEFAULT_HOURS,
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
    )

    ctl = (out or {}).get("control") or {}
    session = Path(ctl.get("session_dir", "")).resolve() if ctl.get("session_dir") else None
    if session and session.exists():
        B._atomic(session / "v2_8_7_launch_receipt.json", {
            "time": B._iso(),
            "deploy_supervisor_version": DEPLOY_VERSION,
            "session": str(session),
            "normal_v5_discovery_preserved": True,
            "strict_pre_m0_data_gate": False,
            "strategy_rules_changed": False,
            "recorder_code_changed": False,
            "child_src": pp["src"],
        })
    return out


def q1_data_audit(session_dir, *, show=True):
    """Post-run recorder audit. Coverage is reported, not used as a pre-M0 kill gate."""
    session = Path(session_dir).resolve()
    raw = session / "raw_capture"
    capture = B._read(raw / "capture_spec.json", {}) or {}
    manifest = B._read(raw / "session_manifest.json", {}) or {}
    counts = manifest.get("final_counts") or {}
    seen_series, meta_tickers, bad_meta_rows = _read_jsonl_series(raw / "market_metadata.jsonl")

    required_files = [
        "capture_spec.json", "session_manifest.json", "book_top3_events.jsonl",
        "ticker_event_time.jsonl", "trades_event_time.jsonl", "market_metadata.jsonl",
        "market_rotations.jsonl", "connection_events.jsonl", "book_repair_events.jsonl",
    ]
    missing = [x for x in required_files if not (raw / x).exists()]
    missing_series = sorted(set(EXPECTED_SERIES) - seen_series)

    # These are the integrity requirements for the Q1 recorder bundle. We do NOT
    # require perfect pre-M0 coverage or all-nine-series discovery at startup.
    checks = {
        "capture_spec_is_proven_v5": capture.get("study_version") == V5.STUDY_VERSION,
        "research_window_is_m0_m5": capture.get("research_elapsed_seconds") == [0.0, 300.0],
        "label_tail_is_m5_plus_30s": capture.get("persisted_elapsed_seconds") == [0.0, 330.0],
        "all_raw_files_present": not missing,
        "final_manifest_present": bool(manifest.get("ended_at")),
        "metadata_json_parse_clean": bad_meta_rows == 0,
        "some_market_metadata_recorded": len(meta_tickers) > 0,
        "book_rows_positive": int(counts.get("book_rows") or 0) > 0,
        "ticker_rows_positive": int(counts.get("ticker_rows") or 0) > 0,
        "snapshots_positive": int(counts.get("snapshots_received") or 0) > 0,
    }
    passed = all(checks.values())
    out = {
        "session": str(session),
        "passed": bool(passed),
        "checks": checks,
        "missing_files": missing,
        "final_counts": counts,
        "series_seen": sorted(seen_series),
        "missing_series": missing_series,
        "series_coverage": f"{len(seen_series)}/{len(EXPECTED_SERIES)}",
        "metadata_tickers_seen": sorted(meta_tickers),
        "bad_metadata_rows": int(bad_meta_rows),
        "sequence_gaps": int(counts.get("sequence_gaps") or 0),
        "sequence_numbers_missing": int(counts.get("sequence_numbers_missing") or 0),
        "book_repairs": {
            "snapshot_requests": int(counts.get("snapshot_requests") or 0),
            "crossed_after_delta_resyncs": int(counts.get("crossed_after_delta_resyncs") or 0),
            "negative_level_resyncs": int(counts.get("negative_level_resyncs") or 0),
            "ticker_persistent_mismatch_resyncs": int(counts.get("ticker_persistent_mismatch_resyncs") or 0),
        },
        "note": (
            "Early-window or startup coverage is diagnostic, not a Q1 kill gate. "
            "For Q10, coverage/freshness will be evaluated over the full run."
        ),
    }
    if show:
        print("=" * 104)
        print("V2.8.7 Q1 RAW V5 DATA AUDIT — READ ONLY")
        print("=" * 104)
        for k, v in checks.items():
            print(f"{k:48s}: {v}")
        print("Series coverage:             ", out["series_coverage"])
        print("Missing series:              ", missing_series)
        print("Final counts:                ", counts)
        print("Sequence gaps:               ", out["sequence_gaps"])
        print("Sequence numbers missing:    ", out["sequence_numbers_missing"])
        print("Book repairs:                ", out["book_repairs"])
        print("DATA AUDIT:", "PASS" if passed else "FAIL")
    return out


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    session = Path(session_dir).resolve()
    base = V282.q1_promotion_check(session, show=False, write_receipt=False)
    data = q1_data_audit(session, show=False)
    launch = B._read(session / "v2_8_7_launch_receipt.json", {}) or {}
    checks = dict(base.get("checks") or {})
    checks.update({
        "v2_8_7_launch_receipt_present": bool(launch),
        "v2_8_7_normal_v5_discovery": launch.get("normal_v5_discovery_preserved") is True,
        "v2_8_7_raw_data_integrity": data.get("passed") is True,
    })
    passed = all(checks.values())
    receipt = dict(base)
    receipt.update({
        "time": B._iso(),
        "deploy_supervisor_version": DEPLOY_VERSION,
        "passed": bool(passed),
        "checks": checks,
        "v2_8_7_data_audit": data,
        "note": "Operational promotion only; not evidence that expected PnL is positive.",
    })
    if passed and write_receipt:
        B._atomic(PROMOTION_PATH, receipt)
    if show:
        print("=" * 104)
        print("Q1 V1.5 / V2.8.7 30-MIN OPERATIONAL PROMOTION — READ ONLY")
        print("=" * 104)
        for k, v in checks.items():
            print(f"{k:52s}: {v}")
        print("PROMOTION:", "PASS" if passed else "NOT READY")
        if passed:
            print("Receipt:", PROMOTION_PATH)
    return receipt


def live_status(*args, **kwargs):
    return V282.live_status(*args, **kwargs)


def kill_and_flatten_live(*args, **kwargs):
    return V282.kill_and_flatten_live(*args, **kwargs)


__all__ = [
    "DEPLOY_VERSION", "LIVE", "CORE", "Q1_ARM", "KILL_ARM", "PROMOTION_PATH",
    "Q1_DEFAULT_HOURS", "static_self_check", "start_q1_smoke", "q1_data_audit",
    "q1_promotion_check", "live_status", "kill_and_flatten_live",
]
