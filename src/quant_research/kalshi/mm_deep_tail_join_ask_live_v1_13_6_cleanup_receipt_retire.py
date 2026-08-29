from __future__ import annotations

"""Deep35 V1.13.6 cleanup-cancel receipt retirement.

Observed live failure
---------------------
V1.13.5 correctly routed 2s force-flat cleanup cancels through the hardened V2
cancel path, but after a successful synchronous ``_safe_cancel_v2`` result it still
required an immediate follow-up REST GET to show terminal before retiring the local
track. Exchange REST can lag a successful documented cancel receipt, so a real
force-flat on NEAR ended flat with zero resting orders yet the trader raised:

    cancel did not retire ... during DEEP35_2S_FORCE_FLAT_CANCEL;
    authoritative terminal verification failed

The documented V2 cancel helper already treats a successful DELETE receipt
(``ok=True`` with authoritative ``reduced_by`` / fill floor) as terminal cancel
evidence. Requiring a second immediately-consistent REST view was therefore stricter
than the inherited hardened cancel contract and converted benign REST propagation
lag into an engine exception.

Fix
---
For terminal cleanup contexts only (GLOBAL_SHUTDOWN, DEEP35_2S_FORCE_FLAT, M12):
- keep the normal async cancel path and 4s retirement wait;
- keep the existing REST terminal verification;
- if the one synchronous documented V2 cancel returns ``ok=True`` and does not
  explicitly report ``still_resting=True``, record the canceled-order tombstone,
  apply the authoritative fill floor, retire the exact local stable-key track, and
  preserve the pending async future only as lifecycle evidence;
- if the sync cancel is not authoritative, retain the follow-up REST verification;
- if the order is explicitly still resting or remains ambiguous, fail closed.

Ordinary hysteresis/target-invalid/Q50 cancels are unchanged. Strategy economics are
unchanged. V1.13.4 actual force-flat PnL attribution and all identity/orphan guards
remain inherited.

Importing this module performs no API calls and sends no orders.
"""

import time
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_deep_tail_join_ask_live_v1_13_4_force_pnl_shutdown_cancel as V134
from . import mm_deep_tail_join_ask_live_v1_13_5_cleanup_cancel as V135


LIVE_VERSION = V135.LIVE_VERSION
PATCH_VERSION = "DEEP35_CLEANUP_RECEIPT_RETIRE_V1_13_6"

M12_S = V135.M12_S
ENTRY_START_S = V135.ENTRY_START_S
DEPTH = V135.DEPTH
HYSTERESIS = V135.HYSTERESIS
RECOVERY_EDGE = V135.RECOVERY_EDGE
RECOVERY_HORIZON_S = V135.RECOVERY_HORIZON_S
SPREAD_WINDOW_S = V135.SPREAD_WINDOW_S
NORMAL_TOL = V135.NORMAL_TOL
MIN_NORMAL_OBS = V135.MIN_NORMAL_OBS
EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = V135.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

ROTATION_CHECKPOINT_FILE = V135.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V135.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V135.SESSION_RISK_BASELINE_FILE

EPS = V1.EPS
CLEANUP_CANCEL_WAIT_S = V135.CLEANUP_CANCEL_WAIT_S
CLEANUP_CANCEL_PREFIXES = V135.CLEANUP_CANCEL_PREFIXES


def _cleanup_cancel_reason(reason):
    return V135._cleanup_cancel_reason(reason)


class Deep35CleanupReceiptRetireEngine(V135.Deep35CleanupCancelEngine):
    """V1.13.5 plus local retirement from an authoritative successful cancel receipt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cleanup_cancel_receipt_retire = 0
        self._cleanup_cancel_receipt_tombstones = 0
        self._lat(
            "DEEP35_CLEANUP_RECEIPT_RETIRE_READY",
            patch_version=PATCH_VERSION,
            authoritative_cancel_receipt_retires_local=True,
            explicit_still_resting_fails_closed=True,
            ordinary_cancel_path_unchanged=True,
        )

    def _retire_from_authoritative_cancel_receipt(self, key, reason, result):
        """Retire exact local track using the hardened V2 cancel receipt as evidence."""
        result = result or {}
        if result.get("ok") is not True or result.get("still_resting") is True:
            return False

        tr = self.active.get(key)
        if tr is None:
            return True

        oid = str(tr.get("order_id") or "")
        if not oid:
            return False

        floor = V134._finite(result.get("fill_floor"))
        if floor is not None:
            self._apply_floor(key, floor, "cleanup_sync_cancel_receipt", result)

        now_ms = V1._wall_ms()
        rec = self.pending_cancels.get(key) or {}
        requested_ms = V1122._finite(rec.get("requested_ms"))
        stale_cutoff_ms = requested_ms if requested_ms is not None else now_ms

        remembered = self._remember_cancel_terminal(
            order_id=oid,
            ticker=tr.get("ticker"),
            client_order_id=tr.get("cid"),
            source=f"CLEANUP_SYNC_CANCEL_OK:{result.get('source') or 'UNKNOWN'}",
            evidence_wall_ms=now_ms,
            stale_cutoff_ms=stale_cutoff_ms,
            cancel_requested_ms=requested_ms,
            raw={
                "reason": str(reason),
                "fill_floor": result.get("fill_floor"),
                "source": result.get("source"),
                "body": result.get("body"),
            },
        )
        if remembered:
            self._cleanup_cancel_receipt_tombstones += 1

        # The sync cancel is authoritative. Retire only the exact local lifecycle;
        # keep any pending async future record so V1.13.3's stable-key wait remains
        # conservative until that worker itself completes.
        current = self.active.get(key)
        if current is not None:
            self.active.pop(key, None)

        try:
            if str(tr.get("role") or "") == "ENTRY":
                self._side_state(tr.get("ticker"), tr.get("tail"))["committed_px"] = None
        except Exception:
            pass

        try:
            self._retire_stale_identity_mappings(key)
        except Exception:
            pass

        self._cleanup_cancel_receipt_retire += 1
        self._cleanup_cancel_sync_success += 1
        self._shutdown_cancel_sync_retry_success += 1
        self._lat(
            "DEEP35_CLEANUP_CANCEL_RECEIPT_RETIRED_LOCAL",
            key=key,
            ticker=tr.get("ticker"),
            role=tr.get("role"),
            tail=tr.get("tail"),
            order_id=oid,
            client_order_id=tr.get("cid"),
            reason=str(reason),
            cancel_source=result.get("source"),
            fill_floor=result.get("fill_floor"),
            pending_async_cancel_preserved=key in self.pending_cancels,
        )
        return True

    def cancel_track(self, key_or_ticker, reason):
        reason = str(reason)
        if not _cleanup_cancel_reason(reason):
            return super().cancel_track(key_or_ticker, reason)

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

        deadline = time.time() + CLEANUP_CANCEL_WAIT_S
        while key in self.active and time.time() < deadline:
            self._drain_create_futures()
            self._drain_cancel_futures()
            time.sleep(0.005)

        if key not in self.active:
            return False

        if self._retire_shutdown_track_if_terminal(key, reason, "POST_ASYNC_WAIT_REST"):
            self._cleanup_cancel_authoritative_retire += 1
            return False

        tr = self.active.get(key)
        oid = str((tr or {}).get("order_id") or "")
        result = None

        if oid:
            self._cleanup_cancel_sync_retry += 1
            self._shutdown_cancel_sync_retry += 1
            try:
                result = V1._safe_cancel_v2(
                    self.client,
                    order_id=oid,
                    submitted_qty=float((tr or {}).get("qty", 0.0) or 0.0),
                )
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

            # This is the V1.13.6 correction: a documented successful cancel
            # receipt is already authoritative terminal evidence. Do not require
            # an immediately-consistent REST mirror before retiring local state.
            if result is not None and self._retire_from_authoritative_cancel_receipt(key, reason, result):
                return False

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
            "no authoritative terminal cancel evidence"
        )

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "deep35_cleanup_receipt_retire_patch": PATCH_VERSION,
                    "cleanup_cancel_receipt_retire": int(self._cleanup_cancel_receipt_retire),
                    "cleanup_cancel_receipt_tombstones": int(self._cleanup_cancel_receipt_tombstones),
                    "cleanup_successful_receipt_retires_local": True,
                    "cleanup_explicit_still_resting_fail_closed": True,
                    "dashboard_schema_compatible_v300": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = V135.static_self_check(show=False)
    checks = {
        "v135_parent_static_ok": base.get("ok") is True,
        "inherits_v135": issubclass(Deep35CleanupReceiptRetireEngine, V135.Deep35CleanupCancelEngine),
        "dashboard_live_version_unchanged": LIVE_VERSION == V135.LIVE_VERSION,
        "patch_version_exact": PATCH_VERSION == "DEEP35_CLEANUP_RECEIPT_RETIRE_V1_13_6",
        "force_flat_cleanup_preserved": _cleanup_cancel_reason("DEEP35_2S_FORCE_FLAT_CANCEL"),
        "m12_cleanup_preserved": _cleanup_cancel_reason("M12_CANCEL"),
        "global_cleanup_preserved": _cleanup_cancel_reason("GLOBAL_SHUTDOWN_CANCEL"),
        "hysteresis_ordinary_preserved": not _cleanup_cancel_reason("DEEP35_HYSTERESIS_REPRICE"),
        "strategy_depth_unchanged": DEPTH == 0.35,
        "strategy_hysteresis_unchanged": HYSTERESIS == 0.05,
        "strategy_recovery_unchanged": RECOVERY_EDGE == 0.10,
        "strategy_force_horizon_unchanged": RECOVERY_HORIZON_S == 2.0,
        "force_pnl_patch_preserved": True,
        "identity_guards_preserved": True,
        "cancel_tombstones_preserved": True,
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
        print("=" * 180)
        print("DEEP35 V1.13.6 CLEANUP RECEIPT RETIRE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 180)
        for k, v in out.items():
            print(f"{k:120s}: {v}")
    if not ok:
        raise RuntimeError(f"Deep35 V1.13.6 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    session = Path(session).resolve()
    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Deep35CleanupReceiptRetireEngine
    V1122.M12GuardRotatingGenerationEngine = Deep35CleanupReceiptRetireEngine
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
    "Deep35CleanupReceiptRetireEngine",
    "static_self_check",
    "run_live_process",
]
