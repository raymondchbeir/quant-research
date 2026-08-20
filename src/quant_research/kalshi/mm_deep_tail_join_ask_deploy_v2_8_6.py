from __future__ import annotations

"""V2.8.6: hardened 30-minute Q1 operational smoke on the proven V5 recorder.

This is an operational wrapper only. It changes no strategy/alpha rule and does not
modify the V5 recorder. The child remains V2.8.2 / live V1.5, preserving:
- Q1 5c YES + 5c NO M1 entry pair;
- first-fill-wins opposite-tail cancellation;
- fixed passive JOIN_ASK exit after a full observable entry fill;
- persistent M5 cleanup and reduce-only residual flattening;
- bounded raw ingestion and independent V2.8 guardian;
- V2.8.2 reuse of the already-passed parent fee snapshot.

What V2.8.6 adds:
1) Q1 is fixed to exactly 30 minutes (0.5 h) from the first complete window.
2) Launch is allowed only 3-5 minutes before the next 15-minute M0 boundary.
3) Before that M0, the unchanged V5 raw_capture must prove, using local files only:
   - healthy/running V5 process;
   - all 3 required public channels;
   - at least 9 subscribed markets;
   - at least 9 orderbook snapshots;
   - metadata coverage of all nine frozen crypto series.
4) Promotion additionally audits the finished raw V5 bundle/counters.

The data gate makes no exchange/API requests. If it cannot prove readiness before M0,
the run is killed/flattened and is not promotable.

Importing this module sends no orders.
"""

import json
import time
from pathlib import Path

import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_deploy_v2_8_2 as V282
from . import mm_event_time_m0_m5_recorder_v5 as V5


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_8_6_Q1_30M_PROVEN_V5_FULL_DATA_GATE"
LIVE = V282.LIVE
CORE = V282.CORE
Q1_ARM = "LIVE_DEEP_TAIL_Q1_30M_V286"
KILL_ARM = V282.KILL_ARM
PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2_8_6.json"

Q1_DEFAULT_HOURS = 0.5
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0
SAFE_PRESUB_MIN_LEAD_S = 180.0
SAFE_PRESUB_MAX_LEAD_S = 300.0
READY_GUARD_BEFORE_M0_S = 10.0
DATA_READY_MAX_WAIT_S = 120.0
REQUIRED_CHANNELS = {"orderbook_delta", "ticker", "trade"}
EXPECTED_SERIES = tuple(V5.CRYPTO_SERIES)
MIN_SUBSCRIBED_MARKETS = len(EXPECTED_SERIES)
MIN_INITIAL_SNAPSHOTS = len(EXPECTED_SERIES)


def _timing(now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    epoch = float(now.timestamp())
    next_m0_epoch = (int(epoch // 900.0) + 1) * 900.0
    next_m0 = pd.Timestamp(next_m0_epoch, unit="s", tz="UTC")
    lead_s = float(next_m0_epoch - epoch)
    return {
        "now": now,
        "next_m0": next_m0,
        "lead_s": lead_s,
        "safe": SAFE_PRESUB_MIN_LEAD_S <= lead_s <= SAFE_PRESUB_MAX_LEAD_S,
        "safe_start": next_m0 - pd.Timedelta(seconds=SAFE_PRESUB_MAX_LEAD_S),
        "safe_end": next_m0 - pd.Timedelta(seconds=SAFE_PRESUB_MIN_LEAD_S),
    }


def _read_jsonl_series(path):
    path = Path(path)
    seen = set()
    tickers = set()
    bad_rows = 0
    if not path.exists():
        return seen, tickers, bad_rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                bad_rows += 1
                continue
            s = str(row.get("series_ticker") or "")
            t = str(row.get("ticker") or "")
            if s:
                seen.add(s)
            if t:
                tickers.add(t)
    return seen, tickers, bad_rows


def _raw_readiness(session):
    session = Path(session).resolve()
    raw = session / "raw_capture"
    rh = B._read(raw / "health.json", {}) or {}
    seen_series, meta_tickers, bad_meta_rows = _read_jsonl_series(raw / "market_metadata.jsonl")
    channels = set(str(x) for x in (rh.get("channels") or []))
    subscribed = int(rh.get("subscribed_markets") or 0)
    snapshots = int(rh.get("snapshots_received") or 0)
    checks = {
        "raw_capture_exists": raw.exists(),
        "recorder_running": rh.get("running") is True,
        "recorder_healthy": rh.get("healthy") is True,
        "proven_v5_version": str(rh.get("study_version")) == str(V5.STUDY_VERSION),
        "required_channels_present": REQUIRED_CHANNELS.issubset(channels),
        "subscribed_at_least_frozen_universe": subscribed >= MIN_SUBSCRIBED_MARKETS,
        "snapshots_at_least_frozen_universe": snapshots >= MIN_INITIAL_SNAPSHOTS,
        "all_frozen_series_in_metadata": set(EXPECTED_SERIES).issubset(seen_series),
        "metadata_json_parse_clean": bad_meta_rows == 0,
    }
    details = {
        "raw_capture": str(raw),
        "recorder_health": rh,
        "series_seen": sorted(seen_series),
        "metadata_tickers_seen": sorted(meta_tickers),
        "bad_metadata_rows": int(bad_meta_rows),
        "required_series": list(EXPECTED_SERIES),
    }
    return all(checks.values()), checks, details


def static_self_check(*, show=True):
    base = V282.static_self_check(show=False)
    checks = {
        "base_v2_8_2_ok": base.get("ok") is True,
        "alpha_rules_unchanged": True,
        "q1_fixed_30_minutes": abs(Q1_DEFAULT_HOURS - 0.5) < 1e-12,
        "live_engine_version": LIVE.LIVE_VERSION,
        "bounded_raw_ingestion": base.get("bounded_raw_ingestion") is True,
        "guardian_unchanged": (
            base.get("rss_guard_unchanged") is True
            and base.get("deadline_initial_grace_unchanged") is True
            and base.get("cleanup_overrun_still_fail_closed") is True
        ),
        "fee_snapshot_reuse": base.get("child_fee_snapshot_reuse") is True,
        "proven_recorder_version": V5.STUDY_VERSION,
        "capture_window_unchanged": (
            V5.TRADE_WINDOW_START_S == 0.0
            and V5.TRADE_WINDOW_END_S == 300.0
            and V5.LABEL_TAIL_END_S == 330.0
            and V5.PRESUBSCRIBE_LEAD_S == 300.0
        ),
        "frozen_universe_unchanged": tuple(V5.CRYPTO_SERIES) == (
            "KXBTC15M", "KXBNB15M", "KXDOGE15M", "KXETH15M", "KXHYPE15M",
            "KXNEAR15M", "KXSOL15M", "KXXRP15M", "KXZEC15M",
        ),
        "original_v5_discovery_unchanged": True,
        "data_gate_local_files_only": True,
        "orders_sent": False,
    }
    metadata = {"live_engine_version", "proven_recorder_version", "orders_sent"}
    ok = all(v is True for k, v in checks.items() if k not in metadata)
    out = {"deploy_version": DEPLOY_VERSION, **checks, "ok": bool(ok)}
    if show:
        print("=" * 104)
        print("DEEP-TAIL DEPLOY V2.8.6 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 104)
        for k, v in out.items():
            print(f"{k:56s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.8.6 static self-check failed: {out}")
    return out


def start_q1_smoke(*, arm_phrase=None, runtime_hours=Q1_DEFAULT_HOURS,
                   max_start_loss_usd=Q1_DEFAULT_MAX_LOSS,
                   min_start_equity_usd=Q1_DEFAULT_MIN_EQUITY):
    if str(arm_phrase) != Q1_ARM:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q1_ARM!r} exactly.")
    if abs(float(runtime_hours) - Q1_DEFAULT_HOURS) > 1e-12:
        raise RuntimeError("V2.8.6 Q1 smoke is fixed to exactly 30 minutes (0.5 hours).")

    static_self_check(show=True)
    timing = _timing()
    if not timing["safe"]:
        raise RuntimeError(
            "V2.8.6 REFUSED TO ARM OUTSIDE THE 3-5 MINUTE PRE-SUB WINDOW. "
            f"Now={timing['now']} | next M0={timing['next_m0']} | "
            f"safe launch window={timing['safe_start']} .. {timing['safe_end']}."
        )

    print("=" * 104)
    print("V2.8.6 — Q1 30-MINUTE SMOKE / PROVEN V5 FULL-DATA GATE")
    print("=" * 104)
    print("Target fresh M0:            ", timing["next_m0"])
    print("Seconds until M0:           ", f"{timing['lead_s']:.1f}")
    print("Runtime after first window: 30 minutes")
    print("Recorder:                   ", V5.STUDY_VERSION)
    print("Required series:            ", len(EXPECTED_SERIES))
    print("Required snapshots:         ", MIN_INITIAL_SNAPSHOTS)
    print("Gate API calls:              0")
    print("=" * 104)

    q1 = V282.start_q1_smoke(
        arm_phrase=V282.Q1_ARM,
        runtime_hours=Q1_DEFAULT_HOURS,
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
    )
    ctl = (q1 or {}).get("control") or {}
    session = Path(ctl.get("session_dir", "")).resolve()
    if not session.exists():
        raise RuntimeError("V2.8.6 could not resolve the launched V2.8.2 session directory.")

    hard_deadline_epoch = min(
        time.time() + DATA_READY_MAX_WAIT_S,
        float(timing["next_m0"].timestamp()) - READY_GUARD_BEFORE_M0_S,
    )
    last_checks = {}
    last_details = {}
    while time.time() < hard_deadline_epoch:
        ready, last_checks, last_details = _raw_readiness(session)
        if ready:
            receipt = {
                "time": B._iso(),
                "deploy_supervisor_version": DEPLOY_VERSION,
                "session": str(session),
                "target_m0": str(timing["next_m0"]),
                "ready_before_m0": time.time() < float(timing["next_m0"].timestamp()),
                "checks": last_checks,
                "details": last_details,
                "strategy_rules_changed": False,
                "recorder_code_changed": False,
                "recorder_discovery_changed": False,
                "gate_exchange_api_calls": 0,
                "orders_sent_by_gate": False,
            }
            B._atomic(session / "v2_8_6_proven_v5_full_data_ready.json", receipt)
            print("\n" + "=" * 104)
            print("V2.8.6 PRE-M0 DATA GATE: PASS")
            print("=" * 104)
            print("Session:                    ", session)
            print("Subscribed markets:         ", last_details["recorder_health"].get("subscribed_markets"))
            print("Snapshots received:         ", last_details["recorder_health"].get("snapshots_received"))
            print("Channels:                   ", last_details["recorder_health"].get("channels"))
            print("Series in metadata:         ", last_details.get("series_seen"))
            print("Target M0:                  ", timing["next_m0"])
            print("The detached Q1 process is now running; do not interrupt it.")
            print("=" * 104)
            return V282.live_status(show=False, tail_lines=20)

        status = V282.live_status(show=False, tail_lines=0)
        if status.get("running") is not True:
            break
        if (status.get("guardian_receipt") or {}).get("intervened") is True:
            break
        time.sleep(1.0)

    try:
        kill = V282.kill_and_flatten_live(arm_phrase=V282.KILL_ARM, wait_s=20.0)
    except Exception as exc:
        kill = {"kill_error": repr(exc)}
    raise RuntimeError(
        "V2.8.6 PRE-M0 DATA GATE FAILED. The unchanged V5 recorder did not prove all nine "
        "series + channels + >=9 snapshots before the fresh M0. The run was stopped and is "
        f"not promotable. checks={last_checks} details={last_details} kill={kill}"
    )


def q1_data_audit(session_dir, *, show=True):
    session = Path(session_dir).resolve()
    raw = session / "raw_capture"
    capture = B._read(raw / "capture_spec.json", {}) or {}
    manifest = B._read(raw / "session_manifest.json", {}) or {}
    ready = B._read(session / "v2_8_6_proven_v5_full_data_ready.json", {}) or {}
    counts = manifest.get("final_counts") or {}
    seen_series, meta_tickers, bad_meta_rows = _read_jsonl_series(raw / "market_metadata.jsonl")

    required_files = [
        "capture_spec.json", "session_manifest.json", "book_top3_events.jsonl",
        "ticker_event_time.jsonl", "trades_event_time.jsonl", "market_metadata.jsonl",
        "market_rotations.jsonl", "connection_events.jsonl", "book_repair_events.jsonl",
    ]
    missing = [name for name in required_files if not (raw / name).exists()]
    checks = {
        "pre_m0_data_ready_receipt_present": bool(ready),
        "pre_m0_data_gate_passed": bool(ready) and all((ready.get("checks") or {}).values()),
        "capture_spec_is_proven_v5": capture.get("study_version") == V5.STUDY_VERSION,
        "research_window_is_m0_m5": capture.get("research_elapsed_seconds") == [0.0, 300.0],
        "label_tail_is_m5_plus_30s": capture.get("persisted_elapsed_seconds") == [0.0, 330.0],
        "all_raw_files_present": not missing,
        "final_manifest_present": bool(manifest.get("ended_at")),
        "all_frozen_series_recorded": set(EXPECTED_SERIES).issubset(seen_series),
        "metadata_json_parse_clean": bad_meta_rows == 0,
        "book_rows_positive": int(counts.get("book_rows") or 0) > 0,
        "ticker_rows_positive": int(counts.get("ticker_rows") or 0) > 0,
        "snapshots_at_least_frozen_universe": int(counts.get("snapshots_received") or 0) >= MIN_INITIAL_SNAPSHOTS,
    }
    passed = all(checks.values())
    out = {
        "session": str(session),
        "passed": bool(passed),
        "checks": checks,
        "missing_files": missing,
        "final_counts": counts,
        "series_seen": sorted(seen_series),
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
        "note": "Sequence/repair counts are reported for review, not silently post-hoc filtered.",
    }
    if show:
        print("=" * 104)
        print("V2.8.6 Q1 RAW V5 DATA AUDIT — READ ONLY")
        print("=" * 104)
        for k, v in checks.items():
            print(f"{k:52s}: {v}")
        print("Final counts:               ", counts)
        print("Sequence gaps:              ", out["sequence_gaps"])
        print("Sequence numbers missing:   ", out["sequence_numbers_missing"])
        print("Book repairs:               ", out["book_repairs"])
        print("DATA AUDIT:", "PASS" if passed else "FAIL")
    return out


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    session = Path(session_dir).resolve()
    base = V282.q1_promotion_check(session, show=False, write_receipt=False)
    data = q1_data_audit(session, show=False)
    checks = dict(base.get("checks") or {})
    checks.update({
        "v2_8_6_pre_m0_data_gate": data.get("checks", {}).get("pre_m0_data_gate_passed") is True,
        "v2_8_6_raw_data_audit": data.get("passed") is True,
    })
    passed = all(checks.values())
    receipt = dict(base)
    receipt.update({
        "time": B._iso(),
        "deploy_supervisor_version": DEPLOY_VERSION,
        "passed": bool(passed),
        "checks": checks,
        "v2_8_6_data_audit": data,
        "note": "Operational promotion only; not evidence that expected PnL is positive.",
    })
    if passed and write_receipt:
        B._atomic(PROMOTION_PATH, receipt)
    if show:
        print("=" * 104)
        print("Q1 V1.5 / V2.8.6 30-MIN OPERATIONAL PROMOTION — READ ONLY")
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
    "Q1_DEFAULT_HOURS", "SAFE_PRESUB_MIN_LEAD_S", "SAFE_PRESUB_MAX_LEAD_S",
    "static_self_check", "start_q1_smoke", "q1_data_audit", "q1_promotion_check",
    "live_status", "kill_and_flatten_live", "_timing", "_raw_readiness",
]
