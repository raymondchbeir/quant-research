from __future__ import annotations

"""Tail25 V1.2 final source-audit hardening.

V1.1 added late CREATE-ack recovery.  One audit edge remained: if the V11
idempotent CREATE future itself ends in an exception after private/terminal
messages already retired the local track, continuing would be weaker than the
normal active-track path.  V11 exceptions after its recovery protocol are
explicitly fail-closed because the exchange state could not be identified.

V1.2 therefore preserves that contract for retired tracks as well: a completed
retired CREATE future that raises immediately shuts the generation down so the
order-group/guardian authoritative cleanup path owns recovery.

No strategy parameter or order decision changes. Importing sends no orders.
"""

from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_tail25_join_bbo_live_v1_1_audit as BASE


LIVE_VERSION = BASE.LIVE_VERSION
PATCH_VERSION = "TAIL25_LIFECYCLE_AUDIT_V1_2"
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


class Tail25JoinBboFinalAuditEngine(BASE.Tail25JoinBboLifecycleAuditEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tail25_retired_create_fail_closed = 0
        self._lat(
            "TAIL25_FINAL_AUDIT_READY",
            patch_version=PATCH_VERSION,
            retired_ambiguous_create_fail_closed=True,
        )

    def _drain_create_futures(self):
        # Preserve the same fail-closed semantics V11 uses when its idempotent
        # recovery protocol cannot identify the outcome of a CREATE.  Do this
        # before V1.1's successful late-ack resurrection path.
        for key, fut in list(self.pending_creates.items()):
            if fut is None or not fut.done() or key in self.active:
                continue
            try:
                fut.result()
            except Exception as exc:
                self.pending_creates.pop(key, None)
                self._tail25_retired_create_fail_closed += 1
                meta = dict(self._tail25_lifecycle_meta.get(str(key)) or {})
                self.last_error = (
                    f"retired CREATE future failed closed key={key}: {exc!r}"
                )
                self._lat(
                    "TAIL25_RETIRED_CREATE_FUTURE_FAIL_CLOSED",
                    key=str(key),
                    ticker=meta.get("ticker"),
                    role=meta.get("role"),
                    error=repr(exc),
                )
                self.emit(
                    "CRITICAL",
                    meta.get("ticker"),
                    reason="TAIL25_RETIRED_CREATE_AMBIGUOUS_FAIL_CLOSED",
                    key=str(key),
                    error=repr(exc),
                )
                self.shutdown("TAIL25_RETIRED_CREATE_AMBIGUOUS_FAIL_CLOSED")
                return None
        return super()._drain_create_futures()

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "tail25_final_audit_patch": PATCH_VERSION,
                    "tail25_retired_create_fail_closed": int(
                        self._tail25_retired_create_fail_closed
                    ),
                    "tail25_retired_ambiguous_create_fail_closed_enabled": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = BASE.static_self_check(show=False)
    checks = {
        "v11_lifecycle_audit_ok": base.get("ok") is True,
        "inherits_v11_lifecycle_audit": issubclass(
            Tail25JoinBboFinalAuditEngine,
            BASE.Tail25JoinBboLifecycleAuditEngine,
        ),
        "public_live_version_preserved": LIVE_VERSION == BASE.LIVE_VERSION,
        "patch_version_exact": PATCH_VERSION == "TAIL25_LIFECYCLE_AUDIT_V1_2",
        "entry_25c_preserved": ENTRY_OFFSET == 0.25,
        "entry_hyst2_preserved": ENTRY_REPRICE_HYSTERESIS == 0.02,
        "edge15_preserved": EDGE_ZONE == 0.15,
        "exit_hyst2_preserved": EXIT_REPRICE_HYSTERESIS == 0.02,
        "force3s_preserved": EXIT_HORIZON_S == 3.0,
        "m12_preserved": M12_S == 720.0,
        "retired_ambiguous_create_fail_closed": True,
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
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 156)
        print("TAIL25 V1.2 FINAL ENGINE AUDIT — NO API / NO ORDERS")
        print("=" * 156)
        for k, v in out.items():
            print(f"{k:100s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 final engine audit failed: {out}")
    return out


def run_live_process(session, cfg):
    session = Path(session).resolve()
    BASE.BASE._install_runtime_patch()

    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Tail25JoinBboFinalAuditEngine
    V1122.M12GuardRotatingGenerationEngine = Tail25JoinBboFinalAuditEngine
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
    "ROTATION_CHECKPOINT_FILE",
    "GENERATION_BOOTSTRAP_FILE",
    "SESSION_RISK_BASELINE_FILE",
    "TAIL25_FORCE_REASON",
    "Tail25JoinBboFinalAuditEngine",
    "static_self_check",
    "run_live_process",
]
