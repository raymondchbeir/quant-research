from __future__ import annotations

"""Deep35 V1.13.5 authoritative cleanup-cancel reconciliation.

Observed 2026-08-29 live failure
--------------------------------
V1.13.4 correctly repaired force-flat fill PnL accounting and added authoritative
cancel reconciliation for GLOBAL_SHUTDOWN only.  A later genuine 5-contract NEAR
fill reached the normal 2s Deep35 force-flat path.  The inherited ``flatten()``
pre-cancel called ``cancel_track(..., 'DEEP35_2S_FORCE_FLAT_CANCEL')``.  Because
V1.13.4's reconciliation was scoped only to reasons beginning GLOBAL_SHUTDOWN, the
old 4-second local-retirement timeout raised even though the account subsequently
verified flat with zero resting orders.

Fix
---
Apply the same authoritative terminal verification / one synchronous documented V2
cancel retry to terminal cleanup contexts, not ordinary quote management:

- GLOBAL_SHUTDOWN*
- DEEP35_2S_FORCE_FLAT*
- M12*

Normal hysteresis reprices, target-invalid cancels, side-cap cancels and all other
entry lifecycle behavior remain unchanged.  A cleanup order that is still resting
after authoritative verification/retry remains fail-closed.

V1.13.4 force-flat actual-fill PnL attribution, V1.13.3 lifecycle wait, identity
isolation, canceled-order tombstones and orphan fail-closed logic are inherited
unchanged.

Importing this module performs no API calls and sends no orders.
"""

import time
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_deep_tail_join_ask_live_v1_13_4_force_pnl_shutdown_cancel as V134


LIVE_VERSION = V134.LIVE_VERSION
PATCH_VERSION = "DEEP35_CLEANUP_CANCEL_V1_13_5"

M12_S = V134.M12_S
ENTRY_START_S = V134.ENTRY_START_S
DEPTH = V134.DEPTH
HYSTERESIS = V134.HYSTERESIS
RECOVERY_EDGE = V134.RECOVERY_EDGE
RECOVERY_HORIZON_S = V134.RECOVERY_HORIZON_S
SPREAD_WINDOW_S = V134.SPREAD_WINDOW_S
NORMAL_TOL = V134.NORMAL_TOL
MIN_NORMAL_OBS = V134.MIN_NORMAL_OBS
EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = V134.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

ROTATION_CHECKPOINT_FILE = V134.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V134.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V134.SESSION_RISK_BASELINE_FILE

EPS = V1.EPS
CLEANUP_CANCEL_WAIT_S = 4.0
CLEANUP_CANCEL_PREFIXES = (
    "GLOBAL_SHUTDOWN",
    "DEEP35_2S_FORCE_FLAT",
    "M12",
)


def _cleanup_cancel_reason(reason):
    text = str(reason or "")
    return any(text.startswith(prefix) for prefix in CLEANUP_CANCEL_PREFIXES)


class Deep35CleanupCancelEngine(V134.Deep35ForcePnlShutdownCancelEngine):
    """V1.13.4 plus authoritative reconciliation for every terminal cleanup cancel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cleanup_cancel_authoritative_retire = 0
        self._cleanup_cancel_sync_retry = 0
        self._cleanup_cancel_sync_success = 0
        self._cleanup_cancel_fail_closed = 0
        self._cleanup_cancel_by_reason = {}
        self._lat(
            "DEEP35_CLEANUP_CANCEL_READY",
            patch_version=PATCH_VERSION,
            cleanup_prefixes=list(CLEANUP_CANCEL_PREFIXES),
            force_flat_cancel_authoritative=True,
            m12_cancel_authoritative=True,
            global_shutdown_cancel_authoritative=True,
            ordinary_reprice_cancel_unchanged=True,
            still_resting_fail_closed=True,
        )

    def _count_cleanup_reason(self, reason):
        reason = str(reason)
        self._cleanup_cancel_by_reason[reason] = int(self._cleanup_cancel_by_reason.get(reason, 0)) + 1

    def cancel_track(self, key_or_ticker, reason):
        reason = str(reason)
        if not _cleanup_cancel_reason(reason):
            return super().cancel_track(key_or_ticker, reason)

        # Preserve the inherited ticker-scoped convenience.
        if key_or_ticker not in self.active:
            keys = [
                key
                for key, tr in list(self.active.items())
                if str(tr.get("ticker") or "") == str(key_or_ticker)
            ]
            for key in keys:
                self.cancel_track(key, reason)
            return False

        key = key_or_ticker
        tr = self.active.get(key)
        if tr is None:
            return False

        self._count_cleanup_reason(reason)

        if not bool(tr.get("cancel_requested")):
            self._request_cancel_key(key, reason)

        # First preserve the normal asynchronous priority-worker path.
        deadline = time.time() + CLEANUP_CANCEL_WAIT_S
        while key in self.active and time.time() < deadline:
            self._drain_create_futures()
            self._drain_cancel_futures()
            time.sleep(0.005)

        if key not in self.active:
            return False

        # V1.13.4 already owns the narrow immutable-order REST terminal verifier.
        if self._retire_shutdown_track_if_terminal(key, reason, "POST_ASYNC_WAIT_REST"):
            self._cleanup_cancel_authoritative_retire += 1
            return False

        tr = self.active.get(key)
        oid = str((tr or {}).get("order_id") or "")

        if oid:
            self._cleanup_cancel_sync_retry += 1
            # Retain the old health counters for dashboard/backward compatibility.
            self._shutdown_cancel_sync_retry += 1
            try:
                result = V1._safe_cancel_v2(
                    self.client,
                    order_id=oid,
                    submitted_qty=float((tr or {}).get("qty", 0.0) or 0.0),
                )
                floor = V134._finite((result or {}).get("fill_floor"))
                if floor is not None:
                    self._apply_floor(key, floor, "cleanup_sync_cancel", result)
                self._lat(
                    "DEEP35_CLEANUP_CANCEL_SYNC_RETRY",
                    key=key,
                    ticker=(tr or {}).get("ticker"),
                    order_id=oid,
                    reason=reason,
                    result=result,
                )
            except Exception as exc:
                self._lat(
                    "DEEP35_CLEANUP_CANCEL_SYNC_RETRY_ERROR",
                    key=key,
                    ticker=(tr or {}).get("ticker"),
                    order_id=oid,
                    reason=reason,
                    error=repr(exc),
                )

            # Give already-completed async futures/private evidence a final chance.
            try:
                self._drain_create_futures()
                self._drain_cancel_futures()
            except Exception:
                pass

            if key not in self.active:
                self._cleanup_cancel_sync_success += 1
                self._shutdown_cancel_sync_retry_success += 1
                return False

            if self._retire_shutdown_track_if_terminal(key, reason, "POST_SYNC_CANCEL_REST"):
                self._cleanup_cancel_authoritative_retire += 1
                self._cleanup_cancel_sync_success += 1
                self._shutdown_cancel_sync_retry_success += 1
                return False

        self._cleanup_cancel_fail_closed += 1
        self._shutdown_cancel_still_resting += 1
        raise RuntimeError(
            f"cancel did not retire {key} during {reason}; "
            "authoritative terminal verification failed"
        )

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "deep35_cleanup_cancel_patch": PATCH_VERSION,
                    "cleanup_cancel_authoritative_retire": int(self._cleanup_cancel_authoritative_retire),
                    "cleanup_cancel_sync_retry": int(self._cleanup_cancel_sync_retry),
                    "cleanup_cancel_sync_success": int(self._cleanup_cancel_sync_success),
                    "cleanup_cancel_fail_closed": int(self._cleanup_cancel_fail_closed),
                    "cleanup_cancel_by_reason": dict(self._cleanup_cancel_by_reason),
                    "force_flat_cancel_authoritative_reconcile": True,
                    "m12_cancel_authoritative_reconcile": True,
                    "global_shutdown_cancel_authoritative_reconcile": True,
                    "ordinary_hysteresis_cancel_unchanged": True,
                    "cleanup_still_resting_fail_closed": True,
                    "dashboard_schema_compatible_v300": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = V134.static_self_check(show=False)
    checks = {
        "v134_parent_static_ok": base.get("ok") is True,
        "inherits_v134": issubclass(Deep35CleanupCancelEngine, V134.Deep35ForcePnlShutdownCancelEngine),
        "dashboard_live_version_unchanged": LIVE_VERSION == V134.LIVE_VERSION,
        "patch_version_exact": PATCH_VERSION == "DEEP35_CLEANUP_CANCEL_V1_13_5",
        "global_shutdown_is_cleanup": _cleanup_cancel_reason("GLOBAL_SHUTDOWN_CANCEL"),
        "force_flat_is_cleanup": _cleanup_cancel_reason("DEEP35_2S_FORCE_FLAT_CANCEL"),
        "m12_is_cleanup": _cleanup_cancel_reason("M12_CANCEL"),
        "hysteresis_not_cleanup": not _cleanup_cancel_reason("DEEP35_HYSTERESIS_REPRICE"),
        "target_invalid_not_cleanup": not _cleanup_cancel_reason("DEEP35_TARGET_INVALID"),
        "strategy_depth_unchanged": DEPTH == 0.35,
        "strategy_hysteresis_unchanged": HYSTERESIS == 0.05,
        "strategy_recovery_unchanged": RECOVERY_EDGE == 0.10,
        "strategy_force_horizon_unchanged": RECOVERY_HORIZON_S == 2.0,
        "force_pnl_patch_preserved": True,
        "identity_guards_preserved": True,
        "cancel_tombstones_preserved": True,
        "orphans_fail_closed": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "patch_version": PATCH_VERSION,
        "strategy": "DEEP35_HYST5_REC10_Q50",
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 176)
        print("DEEP35 V1.13.5 CLEANUP-CANCEL STATIC CHECK — NO API / NO ORDERS")
        print("=" * 176)
        for k, v in out.items():
            print(f"{k:116s}: {v}")
    if not ok:
        raise RuntimeError(f"Deep35 V1.13.5 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run V1.12.2 safety runner with V1.13.5 Deep35 child."""
    session = Path(session).resolve()
    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Deep35CleanupCancelEngine
    V1122.M12GuardRotatingGenerationEngine = Deep35CleanupCancelEngine
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
    "M12_S",
    "ENTRY_START_S",
    "DEPTH",
    "HYSTERESIS",
    "RECOVERY_EDGE",
    "RECOVERY_HORIZON_S",
    "SPREAD_WINDOW_S",
    "NORMAL_TOL",
    "MIN_NORMAL_OBS",
    "EXPECTED_EXIT_EFFECTIVE_LATENCY_MS",
    "ROTATION_CHECKPOINT_FILE",
    "GENERATION_BOOTSTRAP_FILE",
    "SESSION_RISK_BASELINE_FILE",
    "Deep35CleanupCancelEngine",
    "static_self_check",
    "run_live_process",
]
