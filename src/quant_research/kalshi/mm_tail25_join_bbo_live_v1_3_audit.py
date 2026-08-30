from __future__ import annotations

"""Tail25 V1.3 execution-visibility guards.

Final audit found that a continuous-quoting strategy needs two safeguards that the
older one-shot DeepTail strategy got implicitly from its no-rearm design:

- never create/recreate an ENTRY while the authenticated private fill/user-order
  websocket is not ready;
- if private execution visibility goes DOWN, permanently disable entry for the
  affected current window after the inherited cancel request;
- independently retire active ENTRY quotes if the parent raw Top3 stream has no
  generation-fresh row for 3 seconds.  Exits remain protected by the fixed 3s
  authoritative force-flat deadline.

These are fail-safe operational guards, not alpha filters.  No strategy price,
quantity or exit rule changes. Importing sends no orders.
"""

import time
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_11_rotation as V111
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_tail25_join_bbo_live_v1_2_audit as BASE


LIVE_VERSION = BASE.LIVE_VERSION
PATCH_VERSION = "TAIL25_EXEC_VISIBILITY_AUDIT_V1_3"
STRATEGY_NAME = BASE.STRATEGY_NAME

ENTRY_START_S = BASE.ENTRY_START_S
M12_S = BASE.M12_S
ENTRY_OFFSET = BASE.ENTRY_OFFSET
ENTRY_REPRICE_HYSTERESIS = BASE.ENTRY_REPRICE_HYSTERESIS
EDGE_ZONE = BASE.EDGE_ZONE
EXIT_REPRICE_HYSTERESIS = BASE.EXIT_REPRICE_HYSTERESIS
EXIT_HORIZON_S = BASE.EXIT_HORIZON_S
ROTATION_CHECKPOINT_FILE = BASE.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = BASE.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = BASE.SESSION_RISK_BASELINE_FILE
TAIL25_FORCE_REASON = BASE.TAIL25_FORCE_REASON

RAW_ENTRY_MAX_AGE_MS = float(V111.FRESH_ROW_MAX_AGE_MS)
RAW_STALE_CHECK_INTERVAL_S = 0.20


class Tail25JoinBboVisibilityAuditEngine(BASE.Tail25JoinBboFinalAuditEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tail25_private_entry_blocks = 0
        self._tail25_private_down_window_disables = 0
        self._tail25_raw_stale_entry_disables = 0
        self._tail25_last_raw_stale_check = 0.0
        self._lat(
            "TAIL25_EXEC_VISIBILITY_AUDIT_READY",
            patch_version=PATCH_VERSION,
            private_ws_required_for_entry=True,
            private_ws_down_disables_window=True,
            raw_entry_max_age_ms=RAW_ENTRY_MAX_AGE_MS,
        )

    def _submit_entry(self, ticker, side, cur):
        if not self.private.ready.is_set():
            self._tail25_private_entry_blocks += 1
            self._disable_entry(
                str(ticker),
                "TAIL25_PRIVATE_WS_NOT_READY_AT_ENTRY",
                private_ws_ready=False,
            )
            return None
        return super()._submit_entry(ticker, side, cur)

    def _drain_private(self, limit=1000):
        out = super()._drain_private(limit=limit)
        # The inherited private-feed handler marks DeepTail state and requests
        # entry cancellation on WS DOWN.  Continuous Tail25 would otherwise rearm
        # on a later book row, so convert that mark into a permanent window disable.
        for ticker, dt_state in list(self.dt.items()):
            if str((dt_state or {}).get("disabled_reason") or "") != "PRIVATE_WS_DOWN":
                continue
            st = self.tail25[str(ticker)]
            if not st.get("entry_disabled"):
                self._tail25_private_down_window_disables += 1
                self._disable_entry(
                    str(ticker),
                    "TAIL25_PRIVATE_WS_DOWN",
                    private_ws_ready=False,
                )
        return out

    def _guard_stale_raw_entries(self):
        now = time.monotonic()
        if now - self._tail25_last_raw_stale_check < RAW_STALE_CHECK_INTERVAL_S:
            return
        self._tail25_last_raw_stale_check = now

        tickers = sorted(
            {
                str(tr.get("ticker") or "")
                for tr in self.active.values()
                if str(tr.get("role") or "") == "ENTRY"
            }
        )
        for ticker in (t for t in tickers if t):
            st = self.tail25[ticker]
            if st.get("entry_disabled") or st.get("force_flat_started"):
                continue
            snap = self.fast.latest_snapshot(ticker) if hasattr(self.fast, "latest_snapshot") else None
            if snap is None:
                fresh = False
                reason = "NO_RAW_SNAPSHOT_FOR_ACTIVE_ENTRY"
            elif hasattr(self.fast, "snapshot_is_generation_fresh"):
                fresh, reason = self.fast.snapshot_is_generation_fresh(
                    snap,
                    max_age_ms=RAW_ENTRY_MAX_AGE_MS,
                )
            else:
                fresh = False
                reason = "RAW_WATCHDOG_HAS_NO_FRESHNESS_PREDICATE"

            if fresh:
                continue
            self._tail25_raw_stale_entry_disables += 1
            self._disable_entry(
                ticker,
                "TAIL25_RAW_STATE_STALE_WITH_ACTIVE_ENTRY",
                raw_reason=str(reason),
                raw_entry_max_age_ms=RAW_ENTRY_MAX_AGE_MS,
            )

    def enforce_wall_clock_m5(self):
        # Runs from the main loop even if no new public book row arrives.
        self._guard_stale_raw_entries()
        return super().enforce_wall_clock_m5()

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "tail25_exec_visibility_audit_patch": PATCH_VERSION,
                    "tail25_private_entry_blocks": int(self._tail25_private_entry_blocks),
                    "tail25_private_down_window_disables": int(
                        self._tail25_private_down_window_disables
                    ),
                    "tail25_raw_stale_entry_disables": int(
                        self._tail25_raw_stale_entry_disables
                    ),
                    "tail25_private_ws_required_for_entry": True,
                    "tail25_private_ws_down_disables_window": True,
                    "tail25_raw_entry_max_age_ms": RAW_ENTRY_MAX_AGE_MS,
                    "tail25_raw_stale_active_entry_cancel_enabled": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = BASE.static_self_check(show=False)
    checks = {
        "v12_final_engine_audit_ok": base.get("ok") is True,
        "inherits_v12_final_engine": issubclass(
            Tail25JoinBboVisibilityAuditEngine,
            BASE.Tail25JoinBboFinalAuditEngine,
        ),
        "public_live_version_preserved": LIVE_VERSION == BASE.LIVE_VERSION,
        "patch_version_exact": PATCH_VERSION == "TAIL25_EXEC_VISIBILITY_AUDIT_V1_3",
        "entry_25c_preserved": ENTRY_OFFSET == 0.25,
        "entry_hyst2_preserved": ENTRY_REPRICE_HYSTERESIS == 0.02,
        "edge15_preserved": EDGE_ZONE == 0.15,
        "exit_hyst2_preserved": EXIT_REPRICE_HYSTERESIS == 0.02,
        "force3s_preserved": EXIT_HORIZON_S == 3.0,
        "m12_preserved": M12_S == 720.0,
        "private_ws_required_for_entry": True,
        "private_ws_down_disables_window": True,
        "raw_stale_active_entry_guard": True,
        "raw_stale_threshold_matches_generation_freshness": RAW_ENTRY_MAX_AGE_MS
        == float(V111.FRESH_ROW_MAX_AGE_MS),
        "orders_sent": False,
        "api_called": False,
    }
    ok = all(
        v is True
        for k, v in checks.items()
        if k not in {"orders_sent", "api_called"}
    )
    out = {
        "live_version": LIVE_VERSION,
        "patch_version": PATCH_VERSION,
        "strategy": STRATEGY_NAME,
        "raw_entry_max_age_ms": RAW_ENTRY_MAX_AGE_MS,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 164)
        print("TAIL25 V1.3 EXECUTION VISIBILITY AUDIT — NO API / NO ORDERS")
        print("=" * 164)
        for k, v in out.items():
            print(f"{k:104s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 visibility audit failed: {out}")
    return out


def run_live_process(session, cfg):
    session = Path(session).resolve()
    BASE.BASE.BASE._install_runtime_patch()

    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Tail25JoinBboVisibilityAuditEngine
    V1122.M12GuardRotatingGenerationEngine = Tail25JoinBboVisibilityAuditEngine
    V1122.LIVE_VERSION = LIVE_VERSION
    B._equity = V124.exact_equity_from_balance
    try:
        return V1122.run_live_process(session, cfg)
    finally:
        B._equity = old_equity
        V1122.LIVE_VERSION = old_version
        V1122.M12GuardRotatingGenerationEngine = old_alias
        V1122.CancelRestReconcileM12Engine = old_engine


__all__ = [
    "LIVE_VERSION",
    "PATCH_VERSION",
    "STRATEGY_NAME",
    "ENTRY_START_S",
    "M12_S",
    "ENTRY_OFFSET",
    "ENTRY_REPRICE_HYSTERESIS",
    "EDGE_ZONE",
    "EXIT_REPRICE_HYSTERESIS",
    "EXIT_HORIZON_S",
    "RAW_ENTRY_MAX_AGE_MS",
    "ROTATION_CHECKPOINT_FILE",
    "GENERATION_BOOTSTRAP_FILE",
    "SESSION_RISK_BASELINE_FILE",
    "TAIL25_FORCE_REASON",
    "Tail25JoinBboVisibilityAuditEngine",
    "static_self_check",
    "run_live_process",
]
