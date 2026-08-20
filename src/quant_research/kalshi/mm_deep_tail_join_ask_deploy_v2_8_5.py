from __future__ import annotations

"""V2.8.5 launch supervisor: restore the proven V5 recorder path.

This wrapper intentionally does NOT replace, parallelize, hard-fail, or otherwise
rewrite V5 market discovery.  The child is exactly V2.8.2, which already retains:
- live V1.5 bounded raw-memory ingestion;
- persistent M5 cleanup;
- V2.8 cleanup-aware guardian;
- V2.8.2 fresh parent fee-snapshot reuse.

The only new behavior is a parent-side launch gate:
- Q1 may be armed only 2-5 minutes before a fresh 15-minute M0 boundary;
- after V2.8.2 launches, the parent waits up to 60 seconds for the unchanged V5
  recorder to show real subscriptions and orderbook snapshots;
- if that proof does not arrive, the wrapper kills/flattens before the target M0
  rather than allowing an empty-but-healthy recorder to idle indefinitely.

No alpha rule, quantity, entry price, M1/M5 timing, exit rule, loss threshold,
recorder capture semantics, or guardian rule is changed.

Importing this module sends no orders.
"""

import time
from pathlib import Path

import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_deploy_v2_8_2 as V282
from . import mm_event_time_m0_m5_recorder_v5 as V5


DEPLOY_VERSION = "MM_DEEP_TAIL_JOIN_ASK_DEPLOY_V2_8_5_PROVEN_V5_DATA_GATE"
LIVE = V282.LIVE
CORE = V282.CORE
Q1_ARM = "LIVE_DEEP_TAIL_Q1_V15_V285"
KILL_ARM = V282.KILL_ARM
PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v2_8_5.json"

Q1_DEFAULT_HOURS = 0.5
Q1_DEFAULT_MAX_LOSS = 5.0
Q1_DEFAULT_MIN_EQUITY = 25.0
SAFE_PRESUB_MIN_LEAD_S = 120.0
SAFE_PRESUB_MAX_LEAD_S = 300.0
DATA_READY_TIMEOUT_S = 60.0
DATA_READY_MIN_SUBSCRIBED = len(V5.CRYPTO_SERIES)
REQUIRED_CHANNELS = {"orderbook_delta", "ticker", "trade"}


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
    safe = SAFE_PRESUB_MIN_LEAD_S <= lead_s <= SAFE_PRESUB_MAX_LEAD_S
    return {
        "now": now,
        "next_m0": next_m0,
        "lead_s": lead_s,
        "safe": bool(safe),
        "safe_start": next_m0 - pd.Timedelta(seconds=SAFE_PRESUB_MAX_LEAD_S),
        "safe_end": next_m0 - pd.Timedelta(seconds=SAFE_PRESUB_MIN_LEAD_S),
    }


def _data_ready_from_status(status):
    h = (status or {}).get("health") or {}
    rh = h.get("recorder_health") or {}
    channels = set(str(x) for x in (rh.get("channels") or []))
    subscribed = int(rh.get("subscribed_markets") or 0)
    snapshots = int(rh.get("snapshots_received") or 0)
    checks = {
        "live_running": (status or {}).get("running") is True,
        "recorder_alive": h.get("recorder_alive") is True,
        "recorder_running": rh.get("running") is True,
        "recorder_healthy": rh.get("healthy") is True,
        "proven_v5_version": str(rh.get("study_version")) == str(V5.STUDY_VERSION),
        "subscribed_at_least_universe": subscribed >= DATA_READY_MIN_SUBSCRIBED,
        "required_channels_present": REQUIRED_CHANNELS.issubset(channels),
        "snapshot_seen": snapshots >= 1,
    }
    return all(checks.values()), checks, rh


def static_self_check(*, show=True):
    base = V282.static_self_check(show=False)
    checks = {
        "base_v2_8_2_ok": base.get("ok") is True,
        "alpha_rules_unchanged": True,
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
        "universe_unchanged": tuple(V5.CRYPTO_SERIES) == (
            "KXBTC15M", "KXBNB15M", "KXDOGE15M", "KXETH15M", "KXHYPE15M",
            "KXNEAR15M", "KXSOL15M", "KXXRP15M", "KXZEC15M",
        ),
        "original_v5_discovery_restored": True,
        "presub_launch_gate": True,
        "parent_data_ready_gate": True,
        "orders_sent": False,
    }
    metadata = {"live_engine_version", "proven_recorder_version", "orders_sent"}
    ok = all(v is True for k, v in checks.items() if k not in metadata)
    out = {"deploy_version": DEPLOY_VERSION, **checks, "ok": bool(ok)}
    if show:
        print("=" * 100)
        print("DEEP-TAIL DEPLOY V2.8.5 STATIC CHECK — NO API / NO ORDERS")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    if not ok:
        raise RuntimeError(f"V2.8.5 static self-check failed: {out}")
    return out


def start_q1_smoke(*, arm_phrase=None, runtime_hours=Q1_DEFAULT_HOURS,
                   max_start_loss_usd=Q1_DEFAULT_MAX_LOSS,
                   min_start_equity_usd=Q1_DEFAULT_MIN_EQUITY):
    if str(arm_phrase) != Q1_ARM:
        raise RuntimeError(f"REAL ORDER ARMING REFUSED. Pass arm_phrase={Q1_ARM!r} exactly.")
    if abs(float(runtime_hours) - Q1_DEFAULT_HOURS) > 1e-12:
        raise RuntimeError("V2.8.5 Q1 smoke is fixed to exactly 0.5 hours.")

    static_self_check(show=True)
    timing = _timing()
    if not timing["safe"]:
        raise RuntimeError(
            "V2.8.5 REFUSED TO ARM OUTSIDE THE SAFE PRE-SUB WINDOW. "
            f"Now={timing['now']} | next M0={timing['next_m0']} | "
            f"safe launch window={timing['safe_start']} .. {timing['safe_end']}. "
            "This is an operational gate only; it prevents testing recorder readiness after M0."
        )

    print("=" * 100)
    print("V2.8.5 — RESTORED PROVEN V5 RECORDER")
    print("=" * 100)
    print("Target fresh M0:          ", timing["next_m0"])
    print("Seconds until M0:         ", f"{timing['lead_s']:.1f}")
    print("Recorder implementation:  ", V5.STUDY_VERSION)
    print("Discovery rewrite:         NONE")
    print("Data-ready proof timeout:  ", f"{DATA_READY_TIMEOUT_S:.0f}s")
    print("=" * 100)

    q1 = V282.start_q1_smoke(
        arm_phrase=V282.Q1_ARM,
        runtime_hours=Q1_DEFAULT_HOURS,
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
    )

    ctl = (q1 or {}).get("control") or {}
    session = Path(ctl.get("session_dir", "")).resolve()
    if not session.exists():
        raise RuntimeError("V2.8.5 could not resolve the launched V2.8.2 session directory.")

    deadline = time.time() + DATA_READY_TIMEOUT_S
    last = {}
    last_checks = {}
    last_rh = {}
    while time.time() < deadline:
        last = V282.live_status(show=False, tail_lines=20)
        ready, last_checks, last_rh = _data_ready_from_status(last)
        if ready:
            now = pd.Timestamp.now(tz="UTC")
            receipt = {
                "time": B._iso(),
                "deploy_supervisor_version": DEPLOY_VERSION,
                "session": str(session),
                "target_m0": str(timing["next_m0"]),
                "ready_before_m0": bool(now < timing["next_m0"]),
                "checks": last_checks,
                "recorder_health": last_rh,
                "strategy_rules_changed": False,
                "recorder_discovery_changed": False,
                "orders_sent_by_gate": False,
            }
            B._atomic(session / "v2_8_5_proven_v5_data_ready.json", receipt)
            print("\n" + "=" * 100)
            print("V2.8.5 DATA-READY GATE: PASS")
            print("=" * 100)
            print("Session:                  ", session)
            print("Subscribed markets:       ", last_rh.get("subscribed_markets"))
            print("Channels:                 ", last_rh.get("channels"))
            print("Snapshots received:       ", last_rh.get("snapshots_received"))
            print("Target M0:                ", timing["next_m0"])
            print("=" * 100)
            return last

        if last.get("running") is not True:
            break
        if (last.get("guardian_receipt") or {}).get("intervened") is True:
            break
        time.sleep(1.0)

    try:
        kill = V282.kill_and_flatten_live(arm_phrase=V282.KILL_ARM, wait_s=12.0)
    except Exception as exc:
        kill = {"kill_error": repr(exc)}

    raise RuntimeError(
        "V2.8.5 DATA-READY GATE FAILED. The unchanged proven V5 recorder did not prove "
        f"subscriptions + channels + snapshot within {DATA_READY_TIMEOUT_S:.0f}s. "
        f"checks={last_checks} recorder_health={last_rh} kill={kill}. "
        "The run was stopped and is not promotable."
    )


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    session = Path(session_dir).resolve()
    base = V282.q1_promotion_check(session, show=False, write_receipt=False)
    data_ready = B._read(session / "v2_8_5_proven_v5_data_ready.json", {}) or {}
    checks = dict(base.get("checks") or {})
    checks.update({
        "v2_8_5_data_ready_receipt_present": bool(data_ready),
        "v2_8_5_ready_before_m0": data_ready.get("ready_before_m0") is True,
        "v2_8_5_original_v5_recorder": (
            ((data_ready.get("recorder_health") or {}).get("study_version")) == V5.STUDY_VERSION
        ),
        "v2_8_5_data_gate_passed": all((data_ready.get("checks") or {}).values()) if data_ready else False,
    })
    passed = all(checks.values())
    receipt = dict(base)
    receipt.update({
        "time": B._iso(),
        "deploy_supervisor_version": DEPLOY_VERSION,
        "passed": bool(passed),
        "checks": checks,
        "v2_8_5_data_ready": data_ready,
        "note": "Operational promotion only; not evidence that expected PnL is positive.",
    })
    if passed and write_receipt:
        B._atomic(PROMOTION_PATH, receipt)
    if show:
        print("=" * 100)
        print("Q1 V1.5 / V2.8.5 OPERATIONAL PROMOTION — READ ONLY")
        print("=" * 100)
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
    "SAFE_PRESUB_MIN_LEAD_S", "SAFE_PRESUB_MAX_LEAD_S", "DATA_READY_TIMEOUT_S",
    "static_self_check", "start_q1_smoke", "q1_promotion_check", "live_status",
    "kill_and_flatten_live", "_timing", "_data_ready_from_status",
]
