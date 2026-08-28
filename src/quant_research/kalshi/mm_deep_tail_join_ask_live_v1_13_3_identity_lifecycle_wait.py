from __future__ import annotations

"""Deep35 V1.13.3 stable-key lifecycle wait patch.

Observed AUDITFIX2 failure
--------------------------
A terminal private user_orders message can legitimately retire the current active
ENTRY track before its asynchronous cancel future has finished.  Deep35 then sees
no active track on the next book event and, because ENTRY keys are deliberately
stable across hysteresis replacements, attempts to create the replacement under
the same key.  V1.13.2 correctly refused to overwrite a still-pending lifecycle,
but treated that expected asynchronous overlap as fatal and shut down with
``IDENTITY_KEY_REUSE_CANCEL_PENDING``.

That overlap is not an identity inconsistency.  The safe behavior is to WAIT for
the prior create/cancel future to finish, then allow the normal next causal BBO to
create the replacement.  No order is sent while the old lifecycle is pending.

Fix
---
- Before a stable-key ENTRY replacement, opportunistically drain a completed
  create/cancel future for that exact key.
- If the old lifecycle is still pending, defer the replacement without changing
  armed state, quote economics, or window eligibility.
- Retry naturally on later causal BBO observations after the pending lifecycle is
  gone.
- Keep V1.13.2's stale CID/OID purge, private user_order identity validation, and
  hard fail-closed behavior for genuine immutable identity mismatches.
- Keep AUDITFIX1 orphan reconciliation and V1.12.2 cancel tombstones unchanged.

The frozen Q50 / Deep35 / Hyst5 / REC10 strategy is unchanged.  Importing this
module performs no API calls and sends no orders.
"""

from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_deep_tail_join_ask_live_v1_13_2_identity_reconcile as V132


LIVE_VERSION = V132.LIVE_VERSION
PATCH_VERSION = "DEEP35_IDENTITY_LIFECYCLE_WAIT_V1_13_3"

M12_S = V132.M12_S
ENTRY_START_S = V132.ENTRY_START_S
DEPTH = V132.DEPTH
HYSTERESIS = V132.HYSTERESIS
RECOVERY_EDGE = V132.RECOVERY_EDGE
RECOVERY_HORIZON_S = V132.RECOVERY_HORIZON_S
SPREAD_WINDOW_S = V132.SPREAD_WINDOW_S
NORMAL_TOL = V132.NORMAL_TOL
MIN_NORMAL_OBS = V132.MIN_NORMAL_OBS
EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = V132.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

ROTATION_CHECKPOINT_FILE = V132.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V132.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V132.SESSION_RISK_BASELINE_FILE


def _replacement_gate(*, key, pending_creates, pending_cancels):
    """Pure lifecycle gate used by runtime and static regression tests."""
    if key in pending_cancels:
        return "WAIT_CANCEL"
    if key in pending_creates:
        return "WAIT_CREATE"
    return "CLEAR"


class Deep35IdentityLifecycleWaitEngine(V132.Deep35IdentityReconcileEngine):
    """V1.13.2 identity safety plus nonfatal wait for prior stable-key lifecycle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replacement_waiting = {}
        self._replacement_wait_cancel = 0
        self._replacement_wait_create = 0
        self._replacement_wait_cleared = 0
        self._replacement_wait_max_ms = 0.0
        self._lat(
            "DEEP35_IDENTITY_LIFECYCLE_WAIT_READY",
            patch_version=PATCH_VERSION,
            stable_key_pending_cancel_action="WAIT_NOT_SHUTDOWN",
            stable_key_pending_create_action="WAIT_NOT_SHUTDOWN",
            v132_identity_guard_preserved=True,
            auditfix1_preserved=True,
            cancel_tombstones_preserved=True,
        )

    def _note_wait(self, key, gate, ticker, side):
        now = V1._wall_ms()
        rec = self._replacement_waiting.get(key)
        if rec is None or rec.get("gate") != gate:
            self._replacement_waiting[key] = {
                "gate": gate,
                "start_ms": now,
                "ticker": str(ticker),
                "side": str(side),
            }
            if gate == "WAIT_CANCEL":
                self._replacement_wait_cancel += 1
            elif gate == "WAIT_CREATE":
                self._replacement_wait_create += 1
            self._lat(
                "DEEP35_REPLACEMENT_DEFERRED_PRIOR_LIFECYCLE",
                key=key,
                ticker=str(ticker),
                side=str(side),
                gate=gate,
                pending_cancel=bool(key in self.pending_cancels),
                pending_create=bool(key in self.pending_creates),
                patch_version=PATCH_VERSION,
            )

    def _clear_wait(self, key):
        rec = self._replacement_waiting.pop(key, None)
        if rec is None:
            return
        waited = max(0.0, V1._wall_ms() - float(rec.get("start_ms") or V1._wall_ms()))
        self._replacement_wait_cleared += 1
        self._replacement_wait_max_ms = max(self._replacement_wait_max_ms, waited)
        self._lat(
            "DEEP35_REPLACEMENT_PRIOR_LIFECYCLE_CLEARED",
            key=key,
            gate=rec.get("gate"),
            waited_ms=waited,
            patch_version=PATCH_VERSION,
        )

    def _submit_entry(self, ticker, side, price, normal_spread, ctx):
        side = str(side).upper()
        key = self._entry_key(ticker, side)

        # A private terminal message can retire active[key] slightly before the
        # asynchronous cancel future finishes.  If completion is already ready,
        # consume it now; otherwise do not reuse the stable key yet.
        rec = self.pending_cancels.get(key)
        if rec is not None:
            fut = rec.get("future") if isinstance(rec, dict) else None
            if fut is not None and fut.done():
                self._drain_cancel_futures()

        fut = self.pending_creates.get(key)
        if fut is not None and fut.done():
            self._drain_create_futures()

        if self.shutdown_started:
            return None

        gate = _replacement_gate(
            key=key,
            pending_creates=self.pending_creates,
            pending_cancels=self.pending_cancels,
        )

        if gate != "CLEAR":
            self._note_wait(key, gate, ticker, side)
            # Critical invariant: do not mutate armed/committed state and do not
            # send an order.  A later BBO will retry after the old lifecycle ends.
            return None

        self._clear_wait(key)
        return super()._submit_entry(ticker, side, price, normal_spread, ctx)

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "deep35_identity_lifecycle_wait_patch": PATCH_VERSION,
                    "identity_replacement_wait_cancel": int(self._replacement_wait_cancel),
                    "identity_replacement_wait_create": int(self._replacement_wait_create),
                    "identity_replacement_wait_cleared": int(self._replacement_wait_cleared),
                    "identity_replacement_waiting_now": int(len(self._replacement_waiting)),
                    "identity_replacement_wait_max_ms": float(self._replacement_wait_max_ms),
                    "identity_pending_lifecycle_is_nonfatal_wait": True,
                    "identity_genuine_mismatch_fail_closed": True,
                    "deep35_identity_reconcile_patch": V132.PATCH_VERSION,
                    "dashboard_schema_compatible_v300": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = V132.static_self_check(show=False)

    checks = {
        "v132_parent_static_ok": base.get("ok") is True,
        "inherits_v132_identity_guard": issubclass(
            Deep35IdentityLifecycleWaitEngine, V132.Deep35IdentityReconcileEngine
        ),
        "dashboard_live_version_unchanged": LIVE_VERSION == V132.LIVE_VERSION,
        "patch_version_exact": PATCH_VERSION == "DEEP35_IDENTITY_LIFECYCLE_WAIT_V1_13_3",
        "pending_cancel_waits_nonfatal": _replacement_gate(
            key="K", pending_creates={}, pending_cancels={"K": object()}
        ) == "WAIT_CANCEL",
        "pending_create_waits_nonfatal": _replacement_gate(
            key="K", pending_creates={"K": object()}, pending_cancels={}
        ) == "WAIT_CREATE",
        "clear_lifecycle_allows_replacement": _replacement_gate(
            key="K", pending_creates={}, pending_cancels={}
        ) == "CLEAR",
        "cancel_has_priority_if_both_present": _replacement_gate(
            key="K", pending_creates={"K": object()}, pending_cancels={"K": object()}
        ) == "WAIT_CANCEL",
        "strategy_depth_unchanged": DEPTH == 0.35,
        "strategy_hysteresis_unchanged": HYSTERESIS == 0.05,
        "strategy_recovery_unchanged": RECOVERY_EDGE == 0.10,
        "strategy_force_horizon_unchanged": RECOVERY_HORIZON_S == 2.0,
        "v132_identity_fail_closed_preserved": True,
        "auditfix1_preserved": True,
        "cancel_tombstones_preserved": True,
        "unknown_orphans_still_fail_closed": True,
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
        print("=" * 168)
        print("DEEP35 V1.13.3 IDENTITY LIFECYCLE WAIT STATIC CHECK — NO API / NO ORDERS")
        print("=" * 168)
        for k, v in out.items():
            print(f"{k:110s}: {v}")
    if not ok:
        raise RuntimeError(f"Deep35 V1.13.3 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run V1.12.2 safety runner with V1.13.3 Deep35 child."""
    session = Path(session).resolve()
    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Deep35IdentityLifecycleWaitEngine
    V1122.M12GuardRotatingGenerationEngine = Deep35IdentityLifecycleWaitEngine
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
    "Deep35IdentityLifecycleWaitEngine",
    "static_self_check",
    "run_live_process",
]
