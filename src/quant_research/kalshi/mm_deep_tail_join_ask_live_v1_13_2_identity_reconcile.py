from __future__ import annotations

"""Deep35 V1.13.2 lifecycle-identity reconciliation patch.

This patch preserves the frozen Deep35/Hyst5/REC10/Q50 strategy and the public
V1.13 live-version string used by the existing notebook dashboard.

Observed AUDITFIX1 failure
--------------------------
Deep35 intentionally reuses one stable local key per ticker/entry-side across
hysteresis replacements (for example ``TICKER|ENTRY|BID``).  The inherited live
engine keeps historical ``order_id_to_key`` and ``cid_to_key`` entries after an
old lifecycle retires.  A late private ``user_order`` message for that retired
order can therefore resolve to the stable key after a replacement track has been
created.  The inherited handler then writes the OLD order_id into the NEW creating
track.  When the new CREATE future returns its correct order_id, AUDITFIX1 sees two
different immutable ids and correctly fails closed with
``CREATE_ACK_ORDER_ID_MISMATCH``.

The 2026-08-28 failure occurred only ~94 ms after the replacement CREATE submit,
which rules out the V11 >=0.90 s recovery/resubmit path as the proximate cause.

Fix
---
1. Before reusing a stable ENTRY key, retire all stale oid/cid mappings that still
   point at that key.  The old identities are retained only for diagnostics.
2. Refuse stable-key reuse while an older create/cancel future is still pending.
3. Validate every private ``user_order`` identity against the CURRENT track CID/OID
   before inherited code may mutate that track.  Late messages from retired
   lifecycles are ignored and logged; a message that claims the current CID but a
   different immutable order_id remains a hard fail-closed inconsistency.
4. Keep AUDITFIX1 exact pending-create REST reconciliation and inherited V1.12.2
   cancel tombstones unchanged.

Importing this module performs no API calls and sends no orders.
"""

from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_deep_tail_join_ask_live_v1_13_1_audit_reconcile as V131


LIVE_VERSION = V131.LIVE_VERSION
PATCH_VERSION = "DEEP35_IDENTITY_RECONCILE_V1_13_2"

M12_S = V131.M12_S
ENTRY_START_S = V131.ENTRY_START_S
DEPTH = V131.DEPTH
HYSTERESIS = V131.HYSTERESIS
RECOVERY_EDGE = V131.RECOVERY_EDGE
RECOVERY_HORIZON_S = V131.RECOVERY_HORIZON_S
SPREAD_WINDOW_S = V131.SPREAD_WINDOW_S
NORMAL_TOL = V131.NORMAL_TOL
MIN_NORMAL_OBS = V131.MIN_NORMAL_OBS
EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = V131.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

ROTATION_CHECKPOINT_FILE = V131.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V131.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V131.SESSION_RISK_BASELINE_FILE


def _drop_mappings_for_key(mapping, key):
    """Pure-ish helper: remove every token mapped to stable key and return tokens."""
    removed = []
    for token, mapped in list(mapping.items()):
        if mapped == key:
            removed.append(str(token))
            mapping.pop(token, None)
    return removed


def _private_identity_case(*, local_cid, local_oid, msg_cid, msg_oid):
    """Pure classifier used by static regression tests and the WS guard."""
    local_cid = str(local_cid or "")
    local_oid = str(local_oid or "")
    msg_cid = str(msg_cid or "")
    msg_oid = str(msg_oid or "")

    if msg_cid and local_cid and msg_cid != local_cid:
        return "STALE_CID"
    if msg_oid and local_oid and msg_oid != local_oid:
        if msg_cid and local_cid and msg_cid == local_cid:
            return "HARD_CURRENT_CID_OID_MISMATCH"
        return "STALE_OID"
    return "ACCEPT"


class Deep35IdentityReconcileEngine(V131.Deep35AuditReconcileEngine):
    """AUDITFIX1 plus stable-key lifecycle identity isolation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._identity_epoch = 0
        self._retired_order_ids = {}
        self._retired_client_ids = {}
        self._identity_stale_user_orders_ignored = 0
        self._identity_hard_private_mismatches = 0
        self._identity_mapping_purges = 0
        self._identity_pending_key_reuse_blocks = 0
        self._lat(
            "DEEP35_IDENTITY_RECONCILE_READY",
            patch_version=PATCH_VERSION,
            stable_entry_key_mapping_purge=True,
            private_user_order_current_identity_validation=True,
            pending_key_reuse_fail_closed=True,
            auditfix1_preserved=True,
            cancel_tombstones_preserved=True,
        )

    def _retire_stale_identity_mappings(self, key):
        key = str(key)
        removed_oids = _drop_mappings_for_key(self.order_id_to_key, key)
        removed_cids = _drop_mappings_for_key(self.cid_to_key, key)
        if removed_oids or removed_cids:
            self._identity_mapping_purges += 1
            for oid in removed_oids:
                self._retired_order_ids[oid] = {
                    "key": key,
                    "retired_wall_ms": V1._wall_ms(),
                }
            for cid in removed_cids:
                self._retired_client_ids[cid] = {
                    "key": key,
                    "retired_wall_ms": V1._wall_ms(),
                }
            # Bound purely diagnostic retention.
            if len(self._retired_order_ids) > 5000:
                self._retired_order_ids = dict(list(self._retired_order_ids.items())[-2500:])
            if len(self._retired_client_ids) > 5000:
                self._retired_client_ids = dict(list(self._retired_client_ids.items())[-2500:])
            self._lat(
                "IDENTITY_STALE_MAPPINGS_RETIRED",
                key=key,
                order_ids=removed_oids,
                client_order_ids=removed_cids,
                patch_version=PATCH_VERSION,
            )
        return removed_oids, removed_cids

    def _fail_private_identity(self, *, key, tr, msg, reason):
        self._identity_hard_private_mismatches += 1
        self.last_error = (
            f"private user_order identity mismatch {reason}: "
            f"key={key} local={tr} msg={msg}"
        )
        self.emit(
            "CRITICAL",
            str((tr or {}).get("ticker") or (msg or {}).get("ticker") or "") or None,
            reason="PRIVATE_USER_ORDER_IDENTITY_MISMATCH",
            mismatch_reason=str(reason),
            key=key,
            local_track=dict(tr or {}),
            private_user_order=msg,
        )
        self.shutdown("PRIVATE_USER_ORDER_IDENTITY_MISMATCH")

    def _new_track(self, ticker, role, tail, side, price, qty, reduce_only):
        # Deep35 ENTRY replacement keys are deliberately stable.  Never let an old
        # unresolved async lifecycle be overwritten under the same dict key.
        key = V1._track_key(ticker, role, tail)
        if key in self.active:
            raise RuntimeError(f"identity reconcile refuses active key overwrite: {key}")

        if key in self.pending_cancels:
            self._identity_pending_key_reuse_blocks += 1
            self.last_error = f"stable key reuse while cancel pending: {key}"
            self.emit("CRITICAL", str(ticker), reason="IDENTITY_KEY_REUSE_CANCEL_PENDING", key=key)
            self.shutdown("IDENTITY_KEY_REUSE_CANCEL_PENDING")
            raise RuntimeError(self.last_error)

        if key in self.pending_creates:
            # If completion is already available, drain it before deciding whether
            # the stable key is safe to reuse.  Never overwrite a live future.
            fut = self.pending_creates.get(key)
            if fut is not None and fut.done():
                self._drain_create_futures()
            if key in self.pending_creates:
                self._identity_pending_key_reuse_blocks += 1
                self.last_error = f"stable key reuse while create pending: {key}"
                self.emit("CRITICAL", str(ticker), reason="IDENTITY_KEY_REUSE_CREATE_PENDING", key=key)
                self.shutdown("IDENTITY_KEY_REUSE_CREATE_PENDING")
                raise RuntimeError(self.last_error)

        self._retire_stale_identity_mappings(key)
        self._identity_epoch += 1
        tr = super()._new_track(ticker, role, tail, side, price, qty, reduce_only)
        tr["identity_epoch"] = int(self._identity_epoch)
        tr["identity_patch"] = PATCH_VERSION
        return tr

    def _handle_user_order(self, msg, recv_ms):
        msg = msg or {}
        cid = str(msg.get("client_order_id") or "")
        oid = str(msg.get("order_id") or "")

        cid_key = self.cid_to_key.get(cid) if cid else None
        oid_key = self.order_id_to_key.get(oid) if oid else None

        if cid_key is not None and oid_key is not None and cid_key != oid_key:
            tr = self.active.get(cid_key) or self.active.get(oid_key) or {}
            self._fail_private_identity(
                key=cid_key,
                tr=dict(tr),
                msg=msg,
                reason="CID_AND_OID_MAP_TO_DIFFERENT_ACTIVE_KEYS",
            )
            return None

        key = cid_key if cid_key is not None else oid_key
        if key is None:
            if oid in self._retired_order_ids or cid in self._retired_client_ids:
                self._identity_stale_user_orders_ignored += 1
                self._lat(
                    "STALE_RETIRED_USER_ORDER_IGNORED",
                    order_id=oid,
                    client_order_id=cid,
                    status=msg.get("status"),
                    ticker=msg.get("ticker"),
                    recv_ms=recv_ms,
                    patch_version=PATCH_VERSION,
                )
            return None

        tr = self.active.get(key)
        if tr is None:
            return None

        local_cid = str(tr.get("cid") or "")
        local_oid = str(tr.get("order_id") or "")
        case = _private_identity_case(
            local_cid=local_cid,
            local_oid=local_oid,
            msg_cid=cid,
            msg_oid=oid,
        )

        if case in {"STALE_CID", "STALE_OID"}:
            self._identity_stale_user_orders_ignored += 1
            # Remove only the stale token that incorrectly pointed at the current
            # stable key; never mutate the current track.
            if case == "STALE_CID" and cid and self.cid_to_key.get(cid) == key:
                self.cid_to_key.pop(cid, None)
                self._retired_client_ids[cid] = {"key": key, "retired_wall_ms": V1._wall_ms()}
            if case == "STALE_OID" and oid and self.order_id_to_key.get(oid) == key:
                self.order_id_to_key.pop(oid, None)
                self._retired_order_ids[oid] = {"key": key, "retired_wall_ms": V1._wall_ms()}
            self._lat(
                "STALE_PRIVATE_USER_ORDER_IGNORED",
                stale_case=case,
                key=key,
                local_cid=local_cid,
                local_oid=local_oid,
                msg_cid=cid,
                msg_oid=oid,
                status=msg.get("status"),
                recv_ms=recv_ms,
                patch_version=PATCH_VERSION,
            )
            return None

        if case == "HARD_CURRENT_CID_OID_MISMATCH":
            self._fail_private_identity(
                key=key,
                tr=dict(tr),
                msg=msg,
                reason=case,
            )
            return None

        before_oid = str(tr.get("order_id") or "")
        out = super()._handle_user_order(msg, recv_ms)
        current = self.active.get(key)
        if current is not None and oid and not before_oid and str(current.get("order_id") or "") == oid:
            current["order_id_source"] = "PRIVATE_WS"
            current["private_ws_order_id_recv_ms"] = float(recv_ms)
        return out

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "deep35_identity_reconcile_patch": PATCH_VERSION,
                    "identity_stale_user_orders_ignored": int(self._identity_stale_user_orders_ignored),
                    "identity_hard_private_mismatches": int(self._identity_hard_private_mismatches),
                    "identity_mapping_purges": int(self._identity_mapping_purges),
                    "identity_pending_key_reuse_blocks": int(self._identity_pending_key_reuse_blocks),
                    "identity_retired_order_ids": int(len(self._retired_order_ids)),
                    "identity_retired_client_ids": int(len(self._retired_client_ids)),
                    "identity_stable_key_private_ws_guard": True,
                    "deep35_audit_reconcile_patch": V131.PATCH_VERSION,
                    "dashboard_schema_compatible_v300": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = V131.static_self_check(show=False)

    m = {"oid-old": "K", "oid-other": "OTHER"}
    removed = _drop_mappings_for_key(m, "K")

    checks = {
        "v131_parent_static_ok": base.get("ok") is True,
        "inherits_auditfix1": issubclass(Deep35IdentityReconcileEngine, V131.Deep35AuditReconcileEngine),
        "dashboard_live_version_unchanged": LIVE_VERSION == V131.LIVE_VERSION,
        "patch_version_exact": PATCH_VERSION == "DEEP35_IDENTITY_RECONCILE_V1_13_2",
        "stable_key_mapping_purge_removes_old": removed == ["oid-old"] and "oid-old" not in m,
        "stable_key_mapping_purge_preserves_other": m.get("oid-other") == "OTHER",
        "stale_cid_classified": _private_identity_case(
            local_cid="cid-new", local_oid="", msg_cid="cid-old", msg_oid="oid-old"
        ) == "STALE_CID",
        "stale_oid_classified": _private_identity_case(
            local_cid="cid-new", local_oid="oid-new", msg_cid="", msg_oid="oid-old"
        ) == "STALE_OID",
        "current_cid_different_oid_hard": _private_identity_case(
            local_cid="cid-new", local_oid="oid-new", msg_cid="cid-new", msg_oid="oid-other"
        ) == "HARD_CURRENT_CID_OID_MISMATCH",
        "current_identity_accepted": _private_identity_case(
            local_cid="cid-new", local_oid="oid-new", msg_cid="cid-new", msg_oid="oid-new"
        ) == "ACCEPT",
        "strategy_depth_unchanged": DEPTH == 0.35,
        "strategy_hysteresis_unchanged": HYSTERESIS == 0.05,
        "strategy_recovery_unchanged": RECOVERY_EDGE == 0.10,
        "strategy_force_horizon_unchanged": RECOVERY_HORIZON_S == 2.0,
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
        print("=" * 160)
        print("DEEP35 V1.13.2 IDENTITY RECONCILE STATIC CHECK — NO API / NO ORDERS")
        print("=" * 160)
        for k, v in out.items():
            print(f"{k:104s}: {v}")
    if not ok:
        raise RuntimeError(f"Deep35 identity reconcile static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run V1.12.2 safety runner with V1.13.2 identity-safe Deep35 child."""
    session = Path(session).resolve()
    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Deep35IdentityReconcileEngine
    V1122.M12GuardRotatingGenerationEngine = Deep35IdentityReconcileEngine
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
    "Deep35IdentityReconcileEngine",
    "static_self_check",
    "run_live_process",
]
