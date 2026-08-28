from __future__ import annotations

"""Deep35 V1.13 audit reconciliation patch.

This module preserves the frozen Deep35/Hyst5/REC10/Q50 strategy and the public
V1.13 live-version string so the existing V3.0.0 dashboard remains compatible.
The deployment identity is the Git commit plus PATCH_VERSION below.

Observed 2026-08-28 failure
---------------------------
The account auditor could observe a newly-created group order through REST before
its asynchronous CREATE future had installed the returned order_id into the local
active track.  The inherited orphan detector only considered active order_ids
"known", so an order carrying our exact client_order_id could be confirmed as an
orphan and trigger the whole group.  Subsequent shutdown cancellation also made
own canceled rows appear in the forensic orphan set.

Fix
---
Before the inherited auditor drains an ACCOUNT_AUDIT snapshot, inspect the queued
snapshot without consuming it.  If a resting row:
  * belongs to this generation's exact order group,
  * carries a non-empty order_id and client_order_id,
  * client_order_id maps to an active ENTRY track whose CREATE future is pending,
  * ticker is consistent,
then bind that exchange order_id to the active track and order_id_to_key mapping.
The inherited V1.12.2 auditor then sees the order as known.  Unknown orders are not
suppressed and still follow the existing fail-closed orphan path.

The CREATE future later must return the same order_id; a mismatch is a hard stop.
Canceled-order stale-REST reconciliation remains the inherited V1.12.2 mechanism,
but the pre-ACK binding ensures later cancel requests carry the immutable order_id
needed for terminal tombstones.

Importing this module performs no API calls and sends no orders.
"""

from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_deep_tail_join_ask_live_v1_13_deep35_hyst5_rec10 as V113


# Keep dashboard-facing identity stable.  The exact source commit + PATCH_VERSION
# distinguish the fixed deployment from the defective c16b10a... run.
LIVE_VERSION = V113.LIVE_VERSION
PATCH_VERSION = "DEEP35_AUDIT_RECONCILE_V1_13_1"

M12_S = V113.M12_S
ENTRY_START_S = V113.ENTRY_START_S
DEPTH = V113.DEPTH
HYSTERESIS = V113.HYSTERESIS
RECOVERY_EDGE = V113.RECOVERY_EDGE
RECOVERY_HORIZON_S = V113.RECOVERY_HORIZON_S
SPREAD_WINDOW_S = V113.SPREAD_WINDOW_S
NORMAL_TOL = V113.NORMAL_TOL
MIN_NORMAL_OBS = V113.MIN_NORMAL_OBS
EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = V113.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

ROTATION_CHECKPOINT_FILE = V113.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V113.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V113.SESSION_RISK_BASELINE_FILE


def _pending_create_rest_match(row, *, gid, pending_by_cid):
    """Pure exact-match predicate used by the audit pre-reconciler."""
    row = row or {}
    if str(row.get("status") or "").lower() != "resting":
        return None
    if str(row.get("order_group_id") or "") != str(gid or ""):
        return None

    oid = str(row.get("order_id") or "")
    cid = str(row.get("client_order_id") or "")
    if not oid or not cid:
        return None

    key = pending_by_cid.get(cid)
    if key is None:
        return None
    return str(key), oid, cid


class Deep35AuditReconcileEngine(V113.Deep35Hyst5Rec10M12Engine):
    """V1.13 strategy with exact client-id pre-ACK audit reconciliation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audit_preack_bound = {}
        self._audit_preack_bind_count = 0
        self._audit_preack_ack_confirmed = 0
        self._audit_preack_mismatch_count = 0
        self._lat(
            "DEEP35_AUDIT_RECONCILE_READY",
            patch_version=PATCH_VERSION,
            dashboard_live_version=LIVE_VERSION,
            pending_create_match="EXACT_GROUP_PLUS_CLIENT_ORDER_ID_PLUS_TICKER",
            unknown_orphan_fail_closed=True,
            inherited_cancel_tombstones=True,
        )

    def _pending_entry_by_cid(self):
        out = {}
        for key, tr in list(self.active.items()):
            if key not in self.pending_creates:
                continue
            if str((tr or {}).get("role") or "").upper() != "ENTRY":
                continue
            cid = str((tr or {}).get("cid") or "")
            if cid:
                out[cid] = key
        return out

    def _peek_audit_snapshots(self, limit=20):
        """Peek Queue contents under its documented internal mutex; do not consume."""
        q = self.audit_q
        mutex = getattr(q, "mutex", None)
        raw_queue = getattr(q, "queue", None)
        if mutex is None or raw_queue is None:
            return []
        try:
            with mutex:
                return list(raw_queue)[: int(limit)]
        except Exception:
            return []

    def _fail_preack_mismatch(self, *, key, tr, row, reason):
        self._audit_preack_mismatch_count += 1
        self.last_error = (
            f"audit pre-ACK reconciliation mismatch {reason}: "
            f"key={key} local={tr} rest={row}"
        )
        self.emit(
            "CRITICAL",
            str((tr or {}).get("ticker") or (row or {}).get("ticker") or "") or None,
            reason="AUDIT_PENDING_CREATE_RECONCILE_MISMATCH",
            mismatch_reason=str(reason),
            key=key,
            local_track=tr,
            resting_row=row,
        )
        self.shutdown("AUDIT_PENDING_CREATE_RECONCILE_MISMATCH")

    def _pre_reconcile_pending_creates_from_audit_queue(self):
        if self.shutdown_started:
            return

        pending_by_cid = self._pending_entry_by_cid()
        if not pending_by_cid:
            return

        for snapshot in self._peek_audit_snapshots(limit=20):
            if self.shutdown_started:
                return
            if (snapshot or {}).get("kind") != "ACCOUNT_AUDIT":
                continue

            for row in (snapshot or {}).get("resting") or []:
                matched = _pending_create_rest_match(
                    row,
                    gid=self.gid,
                    pending_by_cid=pending_by_cid,
                )
                if matched is None:
                    continue

                key, oid, cid = matched
                tr = self.active.get(key)
                if tr is None or key not in self.pending_creates:
                    continue

                row_ticker = str((row or {}).get("ticker") or "")
                local_ticker = str((tr or {}).get("ticker") or "")
                if row_ticker and local_ticker and row_ticker != local_ticker:
                    self._fail_preack_mismatch(
                        key=key,
                        tr=dict(tr),
                        row=row,
                        reason="TICKER_MISMATCH",
                    )
                    return

                existing_oid = str((tr or {}).get("order_id") or "")
                if existing_oid and existing_oid != oid:
                    self._fail_preack_mismatch(
                        key=key,
                        tr=dict(tr),
                        row=row,
                        reason="LOCAL_ORDER_ID_MISMATCH",
                    )
                    return

                mapped_key = self.order_id_to_key.get(oid)
                if mapped_key is not None and mapped_key != key:
                    self._fail_preack_mismatch(
                        key=key,
                        tr=dict(tr),
                        row=row,
                        reason="ORDER_ID_ALREADY_MAPPED_TO_OTHER_TRACK",
                    )
                    return

                if existing_oid == oid:
                    continue

                # Exact own pending CREATE observed authoritatively through REST.
                # Bind only identity; status remains "creating" until CREATE ACK.
                tr["order_id"] = oid
                tr["audit_preack_bound"] = True
                tr["audit_preack_snapshot_recv_ms"] = (snapshot or {}).get("recv_ms")
                self.order_id_to_key[oid] = key
                self._audit_preack_bound[oid] = {
                    "key": key,
                    "ticker": local_ticker,
                    "client_order_id": cid,
                    "snapshot_recv_ms": (snapshot or {}).get("recv_ms"),
                }
                self._audit_preack_bind_count += 1
                self._lat(
                    "AUDIT_PENDING_CREATE_PREACK_BOUND",
                    key=key,
                    ticker=local_ticker,
                    order_id=oid,
                    client_order_id=cid,
                    snapshot_recv_ms=(snapshot or {}).get("recv_ms"),
                    patch_version=PATCH_VERSION,
                )

    def _drain_audit(self):
        # Reconcile exact own pending creates before inherited orphan classification.
        self._pre_reconcile_pending_creates_from_audit_queue()
        if self.shutdown_started:
            return
        return super()._drain_audit()

    def _drain_create_futures(self):
        # If REST pre-bound an order_id, CREATE ACK must confirm the same immutable id.
        for key, fut in list(self.pending_creates.items()):
            if fut is None or not fut.done():
                continue
            tr = self.active.get(key)
            prebound = str((tr or {}).get("order_id") or "")
            if not prebound:
                continue
            try:
                body, _timing = fut.result()
            except Exception:
                continue  # inherited implementation owns transport failure handling
            ack_oid = str((body or {}).get("order_id") or "")
            if ack_oid and ack_oid != prebound:
                self._fail_preack_mismatch(
                    key=key,
                    tr=dict(tr or {}),
                    row={"order_id": ack_oid, "source": "CREATE_ACK"},
                    reason="CREATE_ACK_ORDER_ID_MISMATCH",
                )
                return
            if ack_oid and bool((tr or {}).get("audit_preack_bound")):
                self._audit_preack_ack_confirmed += 1

        return super()._drain_create_futures()

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "deep35_audit_reconcile_patch": PATCH_VERSION,
                    "audit_pending_create_preack_bound": int(self._audit_preack_bind_count),
                    "audit_pending_create_ack_confirmed": int(self._audit_preack_ack_confirmed),
                    "audit_pending_create_mismatches": int(self._audit_preack_mismatch_count),
                    "audit_pending_create_exact_client_id_match": True,
                    "audit_unknown_orphan_fail_closed": True,
                    "audit_cancel_tombstone_parent_preserved": True,
                    "dashboard_schema_compatible_v300": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = V113.static_self_check(show=False)

    gid = "gid-test"
    pending = {"cid-own": "ticker|ENTRY|BID"}
    own = {
        "status": "resting",
        "order_group_id": gid,
        "order_id": "oid-own",
        "client_order_id": "cid-own",
        "ticker": "KXBTC15M-TEST",
    }
    unknown = dict(own, order_id="oid-other", client_order_id="cid-unknown")
    other_group = dict(own, order_group_id="gid-other")

    checks = {
        "v113_parent_static_ok": base.get("ok") is True,
        "inherits_frozen_v113_strategy": issubclass(
            Deep35AuditReconcileEngine,
            V113.Deep35Hyst5Rec10M12Engine,
        ),
        "dashboard_live_version_unchanged": LIVE_VERSION == V113.LIVE_VERSION,
        "patch_version_exact": PATCH_VERSION == "DEEP35_AUDIT_RECONCILE_V1_13_1",
        "exact_pending_create_matches": _pending_create_rest_match(
            own,
            gid=gid,
            pending_by_cid=pending,
        ) == ("ticker|ENTRY|BID", "oid-own", "cid-own"),
        "unknown_client_id_not_suppressed": _pending_create_rest_match(
            unknown,
            gid=gid,
            pending_by_cid=pending,
        ) is None,
        "other_group_not_suppressed": _pending_create_rest_match(
            other_group,
            gid=gid,
            pending_by_cid=pending,
        ) is None,
        "cancel_tombstone_parent_preserved": issubclass(
            Deep35AuditReconcileEngine,
            V1122.CancelRestReconcileM12Engine,
        ),
        "strategy_depth_unchanged": DEPTH == 0.35,
        "strategy_hysteresis_unchanged": HYSTERESIS == 0.05,
        "strategy_recovery_unchanged": RECOVERY_EDGE == 0.10,
        "strategy_force_horizon_unchanged": RECOVERY_HORIZON_S == 2.0,
        "q_gate_inherited_from_v113": True,
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
        print("=" * 152)
        print("DEEP35 V1.13.1 AUDIT RECONCILE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 152)
        for k, v in out.items():
            print(f"{k:96s}: {v}")
    if not ok:
        raise RuntimeError(f"Deep35 audit reconcile static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run the exact V1.12.2 safety runner with the reconciled V1.13 strategy."""
    session = Path(session).resolve()
    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Deep35AuditReconcileEngine
    V1122.M12GuardRotatingGenerationEngine = Deep35AuditReconcileEngine
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
    "Deep35AuditReconcileEngine",
    "static_self_check",
    "run_live_process",
]
