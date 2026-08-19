from __future__ import annotations

"""Staged live launcher: Q1 operational proof is mandatory before Q5+.

This wrapper does not change the live engine. It prevents the brand-new deep-tail
execution stack from being first exercised overnight at Q5. A completed Q1 run must
show at least one full 5c entry and one fixed JOIN_ASK submission, complete M5 cleanup,
clean final account state, and a complete raw/live recording bundle. The promotion
receipt is bound to the exact git HEAD used by that Q1 session; any code change after
Q1 invalidates promotion and requires a new Q1 smoke.
"""

from pathlib import Path

from . import mm_cycle_q10_live_strategy_v10 as V10
from . import mm_deep_tail_join_ask_live_v1 as CORE
from . import mm_deep_tail_join_ask_live_v1_1 as LIVE
from . import mm_deep_tail_join_ask_live_launcher_v1 as L
from . import mm_deep_tail_join_ask_live_audit_v1 as AUDIT
from . import mm_cycle_q10_live_strategy_v1 as B

LAUNCHER_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_LAUNCHER_V1_1_Q1_GATED"
Q1_ARM = "LIVE_DEEP_TAIL_Q1"
Q5_OVERNIGHT_ARM = L.Q5_OVERNIGHT_ARM
PROMOTION_PATH = CORE.ROOT / "q1_operational_promotion_v1.json"


def _current_head():
    return (V10._git_state() or {}).get("head")


def q1_promotion_check(session_dir, *, show=True, write_receipt=True):
    """Read-only session audit; writes only a local promotion receipt when PASS."""
    session = Path(session_dir).resolve()
    a = AUDIT.audit_live_session(session, show=False, write=True)
    cfg = B._read(session / "process_config.json", {}) or {}
    provenance = B._read(session / "deep_tail_source_provenance.json", {}) or {}
    q = B._f(cfg.get("quote_size"), float("nan"))
    q1_head = ((provenance.get("git") or {}).get("head"))
    current_head = _current_head()

    checks = {
        "q1_size": abs(q - 1.0) < 1e-9,
        "completed": a.get("completed") is True,
        "clean_final": a.get("clean_final") is True,
        "raw_bundle_complete": a.get("raw_bundle_complete") is True,
        "live_bundle_complete": a.get("live_bundle_complete") is True,
        "no_operational_fail": a.get("operational_fail") is False,
        "entry_pair_exercised": int(a.get("entry_pairs_posted", 0)) >= 1,
        "actual_tail_fill_seen": int(a.get("tails_selected", 0)) >= 1,
        "full_entry_seen": int(a.get("full_entries", 0)) >= 1,
        "fixed_join_ask_submitted": int(a.get("fixed_exits_posted", 0)) >= 1,
        "m5_path_exercised": int(a.get("m5_finalized", 0)) >= 1,
        "dual_tail_zero": int(a.get("dual_tail_fill_critical", 0)) == 0,
        "flat_verified": a.get("flat_verified") is True,
        "zero_strategy_resting": a.get("strategy_resting_orders_zero") is True,
        "q1_git_head_known": bool(q1_head),
        "same_current_git_head": bool(q1_head and current_head and q1_head == current_head),
    }
    passed = all(checks.values())
    receipt = {
        "time": B._iso(),
        "version": LAUNCHER_VERSION,
        "passed": passed,
        "session": str(session),
        "q1_git_head": q1_head,
        "current_git_head": current_head,
        "live_engine_version": LIVE.LIVE_VERSION,
        "checks": checks,
        "audit": a,
        "note": "Operational promotion only; this does not establish strategy profitability.",
        "orders_sent": False,
        "exchange_api_called": False,
    }
    if passed and write_receipt:
        B._atomic(PROMOTION_PATH, receipt)

    if show:
        print("=" * 100)
        print("DEEP-TAIL Q1 -> Q5 OPERATIONAL PROMOTION CHECK — NO EXCHANGE API")
        print("=" * 100)
        for k, v in checks.items():
            print(f"{k:34s}: {v}")
        print("PROMOTION:", "PASS" if passed else "NOT READY")
        if passed:
            print("Receipt:", PROMOTION_PATH)
        else:
            print("Q5 remains hard-gated. Do not bypass this by importing the lower-level launcher.")
    return receipt


def _require_q1_promotion():
    r = B._read(PROMOTION_PATH, {}) or {}
    current_head = _current_head()
    if not r or r.get("passed") is not True:
        raise RuntimeError(
            "Q5+ HARD GATE: no passing Q1 operational promotion receipt. "
            "Run Q1, let it finish cleanly, then call q1_promotion_check(Q1_SESSION)."
        )
    if str(r.get("live_engine_version")) != str(LIVE.LIVE_VERSION):
        raise RuntimeError("Q5+ HARD GATE: Q1 receipt was produced by a different live-engine version.")
    if not current_head or str(r.get("q1_git_head")) != str(current_head):
        raise RuntimeError(
            "Q5+ HARD GATE: git HEAD changed after the passing Q1 smoke. "
            "A new Q1 smoke is required for this exact code."
        )
    return r


def static_self_check(*, show=True):
    out = L.static_self_check(show=show)
    if show:
        print("Q1 promotion gate path:       ", PROMOTION_PATH)
        print("Q5 direct lower-level bypass:  NOT part of this deployment workflow")
    return out


def live_preflight(**kwargs):
    return L.live_preflight(**kwargs)


def start_q1_smoke(*, arm_phrase=None, runtime_hours=1.0,
                   max_start_loss_usd=5.0, min_start_equity_usd=25.0):
    """REAL ORDERS: first mandatory stage for this exact deployment code."""
    return L.start_ladder_stage(
        quote_size=1,
        runtime_hours=float(runtime_hours),
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
        arm_phrase=arm_phrase,
    )


def start_q5_overnight(*, arm_phrase=None,
                       runtime_hours=L.DEFAULT_OVERNIGHT_HOURS,
                       max_start_loss_usd=L.DEFAULT_Q5_MAX_LOSS,
                       min_start_equity_usd=L.DEFAULT_Q5_MIN_EQUITY):
    """REAL ORDERS: Q5 only after same-code Q1 operational promotion."""
    _require_q1_promotion()
    return L.start_q5_overnight(
        arm_phrase=arm_phrase,
        runtime_hours=float(runtime_hours),
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=float(min_start_equity_usd),
    )


def start_ladder_stage(*, quote_size, runtime_hours, max_start_loss_usd,
                       min_start_equity_usd=None, arm_phrase=None):
    q = float(quote_size)
    if q > 1.0:
        _require_q1_promotion()
    return L.start_ladder_stage(
        quote_size=q,
        runtime_hours=float(runtime_hours),
        max_start_loss_usd=float(max_start_loss_usd),
        min_start_equity_usd=min_start_equity_usd,
        arm_phrase=arm_phrase,
    )


def live_status(**kwargs):
    return L.live_status(**kwargs)


def kill_and_flatten_live(**kwargs):
    return L.kill_and_flatten_live(**kwargs)


__all__ = [
    "LAUNCHER_VERSION",
    "Q1_ARM",
    "Q5_OVERNIGHT_ARM",
    "PROMOTION_PATH",
    "static_self_check",
    "live_preflight",
    "start_q1_smoke",
    "q1_promotion_check",
    "start_q5_overnight",
    "start_ladder_stage",
    "live_status",
    "kill_and_flatten_live",
]
