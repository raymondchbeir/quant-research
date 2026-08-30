from __future__ import annotations

"""Tail25 V1.1 source-audit hardening.

This is an additive operational patch over ``mm_tail25_join_bbo_live_v1``.  It
changes no quote-price, size, edge-zone, exit-price, or M12 strategy parameter.

Audit fixes
-----------
1. Preserve V1.13.3's stable-key lifecycle WAIT semantics for both ENTRY and the
   reusable EXIT_PASSIVE key.  A replacement is never submitted while an older
   create/cancel future for that exact key is still pending.
2. If a CREATE acknowledgement arrives after private/terminal evidence already
   retired the local track, reconstruct that exact lifecycle from immutable local
   metadata, reconcile any fill evidence, and immediately cancel any residual.
   The old code only logged this late acknowledgement, which could leave an
   untracked resting order until the account auditor noticed it.
3. After an ordinary ENTRY reprice cancel completes, immediately re-evaluate the
   freshest current book rather than waiting for another public book event.
4. After a stale post-only JOIN_BBO create rejection, immediately re-evaluate and
   rejoin from a newly certified BBO rather than burning part of the 3s horizon.
5. Anchor the 3.0s exit deadline to the exchange fill timestamp when a plausible
   ``ts_ms`` is available; otherwise fall back to local receipt time.  This makes
   delayed REST reconciliation conservative instead of extending the hold period.

Importing this module performs no API calls and sends no orders.
"""

import math
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_tail25_join_bbo_live_v1 as BASE
from . import mm_tail25_multiseries_router_v1 as ROUTER


LIVE_VERSION = BASE.LIVE_VERSION
PATCH_VERSION = "TAIL25_LIFECYCLE_AUDIT_V1_1"
STRATEGY_NAME = BASE.STRATEGY_NAME

ENTRY_START_S = BASE.ENTRY_START_S
M12_S = BASE.M12_S
ENTRY_OFFSET = BASE.ENTRY_OFFSET
ENTRY_REPRICE_HYSTERESIS = BASE.ENTRY_REPRICE_HYSTERESIS
EDGE_ZONE = BASE.EDGE_ZONE
EXIT_REPRICE_HYSTERESIS = BASE.EXIT_REPRICE_HYSTERESIS
EXIT_HORIZON_S = BASE.EXIT_HORIZON_S
EPS = BASE.EPS

ROTATION_CHECKPOINT_FILE = BASE.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = BASE.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = BASE.SESSION_RISK_BASELINE_FILE
TAIL25_FORCE_REASON = BASE.TAIL25_FORCE_REASON


def _plausible_exchange_fill_ms(raw, *, now_ms=None):
    """Return a plausible exchange wall timestamp in ms, else current wall time."""
    now_ms = float(V1._wall_ms() if now_ms is None else now_ms)
    raw = raw or {}
    try:
        z = float(raw.get("ts_ms"))
    except Exception:
        z = float("nan")
    if math.isfinite(z):
        # Defensive support for a seconds-valued timestamp despite the field name.
        if z < 10_000_000_000.0:
            z *= 1000.0
        # Do not let malformed clocks manufacture a much earlier/later deadline.
        if now_ms - 10.0 * 60.0 * 1000.0 <= z <= now_ms + 1000.0:
            return float(z), "EXCHANGE_TS_MS"
    return now_ms, "LOCAL_RECEIPT_WALL_MS"


def _lifecycle_gate(key, pending_creates, pending_cancels):
    if key in pending_cancels:
        return "WAIT_CANCEL"
    if key in pending_creates:
        return "WAIT_CREATE"
    return "CLEAR"


class Tail25JoinBboLifecycleAuditEngine(BASE.Tail25JoinBboEngine):
    """Frozen Tail25 economics with audited stable-lifecycle behavior."""

    def __init__(self, *args, **kwargs):
        # Retain immutable request metadata independently from active-track state.
        # This exists before parent construction for defensive future compatibility.
        self._tail25_lifecycle_meta = {}
        self._tail25_lifecycle_waits = 0
        self._tail25_ghost_create_acks = 0
        self._tail25_ghost_create_cleanup_dispatched = 0
        self._tail25_entry_immediate_rejoins = 0
        self._tail25_exit_reject_immediate_rejoins = 0
        self._tail25_exchange_deadline_anchors = 0
        self._tail25_local_deadline_anchors = 0
        super().__init__(*args, **kwargs)
        self._lat(
            "TAIL25_LIFECYCLE_AUDIT_READY",
            patch_version=PATCH_VERSION,
            stable_key_wait=True,
            ghost_create_ack_cleanup=True,
            immediate_entry_rejoin_after_cancel=True,
            immediate_exit_rejoin_after_post_only_reject=True,
            exchange_fill_deadline_anchor=True,
        )

    def _new_track(self, ticker, role, tail, side, price, qty, reduce_only):
        tr = super()._new_track(ticker, role, tail, side, price, qty, reduce_only)
        self._tail25_lifecycle_meta[str(tr["key"])] = dict(tr)
        return tr

    def _drain_done_lifecycle_for_key(self, key):
        rec = self.pending_cancels.get(key)
        if isinstance(rec, dict):
            fut = rec.get("future")
            if fut is not None and fut.done():
                self._drain_cancel_futures()
        fut = self.pending_creates.get(key)
        if fut is not None and fut.done():
            self._drain_create_futures()

    def _replacement_clear(self, key, *, ticker, role):
        self._drain_done_lifecycle_for_key(key)
        if self.shutdown_started:
            return False
        gate = _lifecycle_gate(key, self.pending_creates, self.pending_cancels)
        if gate != "CLEAR":
            self._tail25_lifecycle_waits += 1
            self._lat(
                "TAIL25_REPLACEMENT_DEFERRED_PRIOR_LIFECYCLE",
                key=str(key),
                ticker=str(ticker),
                role=str(role),
                gate=gate,
                pending_create=key in self.pending_creates,
                pending_cancel=key in self.pending_cancels,
            )
            return False
        return True

    def _submit_entry(self, ticker, side, cur):
        key = self._entry_key(ticker, side)
        if not self._replacement_clear(key, ticker=ticker, role="ENTRY"):
            return None
        return super()._submit_entry(ticker, side, cur)

    def _submit_passive_exit(self, ticker, desired):
        key = self._exit_key(ticker)
        if not self._replacement_clear(key, ticker=ticker, role="EXIT_PASSIVE"):
            return None
        return super()._submit_passive_exit(ticker, desired)

    def _resurrect_late_create_ack(self, key, body, timing):
        """Reconcile/cancel an acknowledged order whose active track retired first."""
        oid = str((body or {}).get("order_id") or "")
        meta = dict(self._tail25_lifecycle_meta.get(str(key)) or {})
        if not oid or not meta:
            self.last_error = (
                f"late CREATE ack cannot be reconciled key={key} oid={oid!r} meta={bool(meta)}"
            )
            self.emit(
                "CRITICAL",
                meta.get("ticker"),
                reason="TAIL25_GHOST_CREATE_ACK_UNRECONCILABLE",
                key=str(key),
                order_id=oid or None,
            )
            self.shutdown("TAIL25_GHOST_CREATE_ACK_UNRECONCILABLE")
            return

        self._tail25_ghost_create_acks += 1
        tr = dict(meta)
        tr["order_id"] = oid
        tr["status"] = "LATE_CREATE_ACK_CLEANUP"
        tr["create_response_wall_ms"] = V1._wall_ms()
        tr["cancel_requested"] = False
        tr["cancel_reason"] = None
        self.active[key] = tr
        self.cid_to_key[str(tr.get("cid") or "")] = key
        self.order_id_to_key[oid] = key
        try:
            ROUTER.register_order_shard(oid, ROUTER.shard_for_ticker(tr["ticker"]))
        except Exception:
            pass

        B._append(
            self.orders,
            {
                "time": B._iso(),
                "action": "LATE_CREATE_ACK_RECONCILED_FOR_CLEANUP",
                "track": tr,
                "response": body,
                "timing": timing,
            },
        )
        self._lat(
            "TAIL25_LATE_CREATE_ACK_RECONCILED",
            key=str(key),
            ticker=tr.get("ticker"),
            role=tr.get("role"),
            order_id=oid,
        )

        # Private execution may have arrived before the REST acknowledgement.
        self._retry_unmatched_private()
        if key in self.active:
            self._apply_floor(key, V1._order_fill_count(body, 0.0), "late_create_response", body)

        # A full fill may have retired the order while creating exposure.  In that
        # case there is no residual order to cancel; normal Tail25 exit logic owns
        # the newly-created inventory.  Any residual is canceled immediately.
        if key in self.active:
            self._tail25_ghost_create_cleanup_dispatched += 1
            self._request_cancel_key(key, "TAIL25_LATE_CREATE_ACK_CANCEL")

    def _drain_create_futures(self):
        # Handle retired-track CREATE completions before BASE consumes and merely
        # logs them.  Remove only the exact futures handled here.
        for key, fut in list(self.pending_creates.items()):
            if fut is None or not fut.done() or key in self.active:
                continue
            self.pending_creates.pop(key, None)
            try:
                body, timing = fut.result()
            except Exception as exc:
                # A failed create cannot leave a live order according to the
                # idempotent V11 create contract; preserve diagnostics and continue.
                self._lat(
                    "TAIL25_RETIRED_CREATE_FUTURE_FAILED",
                    key=str(key),
                    error=repr(exc),
                )
                continue
            self._resurrect_late_create_ack(key, body, timing)
            if self.shutdown_started:
                return

        # Capture completed post-only rejections so we can immediately rejoin after
        # BASE performs its normal retirement/telemetry.
        rejected_exit_tickers = []
        for key, fut in list(self.pending_creates.items()):
            tr = self.active.get(key)
            if fut is None or not fut.done() or not tr or tr.get("role") != "EXIT_PASSIVE":
                continue
            try:
                exc = fut.exception()
            except Exception:
                exc = None
            if exc is not None and BASE._post_only_reject(exc):
                rejected_exit_tickers.append(str(tr.get("ticker") or ""))

        out = super()._drain_create_futures()

        for ticker in rejected_exit_tickers:
            if not ticker or self.shutdown_started:
                continue
            if self._exit_key(ticker) in self.active:
                continue
            self._tail25_exit_reject_immediate_rejoins += 1
            self._manage_exit(ticker, self.current.get(ticker))
        return out

    def _drain_cancel_futures(self):
        completed_entries = []
        for key, rec in list(self.pending_cancels.items()):
            fut = rec.get("future") if isinstance(rec, dict) else None
            tr = self.active.get(key)
            if fut is not None and fut.done() and tr and tr.get("role") == "ENTRY":
                completed_entries.append(
                    (str(tr.get("ticker") or ""), str(tr.get("tail") or "").upper(), str(tr.get("cancel_reason") or rec.get("reason") or ""))
                )

        out = super()._drain_cancel_futures()

        for ticker, side, reason in completed_entries:
            if not ticker or side not in {"BID", "ASK"} or self.shutdown_started:
                continue
            st = self.tail25[ticker]
            if (
                st.get("entry_disabled")
                or st.get("force_flat_started")
                or st.get("first_entry_fill_wall_ms") is not None
            ):
                continue
            key = self._entry_key(ticker, side)
            if key in self.active:
                continue
            cur = self.current.get(ticker)
            if cur is None:
                continue
            self._tail25_entry_immediate_rejoins += 1
            self._lat(
                "TAIL25_ENTRY_IMMEDIATE_REJOIN_AFTER_CANCEL",
                ticker=ticker,
                side=side,
                prior_cancel_reason=reason,
            )
            self._manage_entry_side(ticker, side, cur)
        return out

    def _process_effective_fill(self, key, source, raw=None):
        tr = self.active.get(key)
        role = str((tr or {}).get("role") or "")
        ticker = str((tr or {}).get("ticker") or "")
        had_first = bool(
            ticker
            and self.tail25[ticker].get("first_entry_fill_wall_ms") is not None
        )
        fill_ms = None
        fill_clock_source = None
        if role == "ENTRY" and not had_first:
            fill_ms, fill_clock_source = _plausible_exchange_fill_ms(raw)

        out = super()._process_effective_fill(key, source, raw)

        if role == "ENTRY" and ticker and not had_first:
            st = self.tail25[ticker]
            if st.get("first_entry_fill_wall_ms") is not None:
                # BASE may have temporarily used local-now while opening the lot.
                # Tighten to the exchange timestamp when available; never extend
                # beyond local observation when a timestamp is implausible.
                anchor = float(fill_ms if fill_ms is not None else V1._wall_ms())
                st["first_entry_fill_wall_ms"] = anchor
                st["force_deadline_wall_ms"] = anchor + EXIT_HORIZON_S * 1000.0
                for lot in self.open_lots.get(ticker, []) or []:
                    lot["deadline_wall_ms"] = st["force_deadline_wall_ms"]
                if fill_clock_source == "EXCHANGE_TS_MS":
                    self._tail25_exchange_deadline_anchors += 1
                else:
                    self._tail25_local_deadline_anchors += 1
                self._lat(
                    "TAIL25_FIRST_FILL_DEADLINE_ANCHORED",
                    ticker=ticker,
                    source=str(source),
                    fill_clock_source=fill_clock_source,
                    first_fill_wall_ms=anchor,
                    force_deadline_wall_ms=st["force_deadline_wall_ms"],
                )
                if V1._wall_ms() + EPS >= float(st["force_deadline_wall_ms"]):
                    self._force_flat_expired(ticker)
        return out

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "tail25_lifecycle_audit_patch": PATCH_VERSION,
                    "tail25_lifecycle_waits": int(self._tail25_lifecycle_waits),
                    "tail25_ghost_create_acks": int(self._tail25_ghost_create_acks),
                    "tail25_ghost_create_cleanup_dispatched": int(self._tail25_ghost_create_cleanup_dispatched),
                    "tail25_entry_immediate_rejoins": int(self._tail25_entry_immediate_rejoins),
                    "tail25_exit_reject_immediate_rejoins": int(self._tail25_exit_reject_immediate_rejoins),
                    "tail25_exchange_deadline_anchors": int(self._tail25_exchange_deadline_anchors),
                    "tail25_local_deadline_anchors": int(self._tail25_local_deadline_anchors),
                    "tail25_pending_lifecycle_nonfatal_wait": True,
                    "tail25_late_create_ack_cleanup": True,
                    "tail25_force_deadline_exchange_anchored_when_available": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = BASE.static_self_check(show=False)
    now = 2_000_000_000_000.0
    exchange, exchange_src = _plausible_exchange_fill_ms(
        {"ts_ms": now - 125.0}, now_ms=now
    )
    fallback, fallback_src = _plausible_exchange_fill_ms(
        {"ts_ms": now - 3_600_000.0}, now_ms=now
    )
    checks = {
        "base_tail25_static_ok": base.get("ok") is True,
        "inherits_tail25_v1": issubclass(
            Tail25JoinBboLifecycleAuditEngine,
            BASE.Tail25JoinBboEngine,
        ),
        "public_live_version_preserved": LIVE_VERSION == BASE.LIVE_VERSION,
        "patch_version_exact": PATCH_VERSION == "TAIL25_LIFECYCLE_AUDIT_V1_1",
        "entry_offset_25c_preserved": ENTRY_OFFSET == 0.25,
        "entry_hyst_2c_preserved": ENTRY_REPRICE_HYSTERESIS == 0.02,
        "edge15_preserved": EDGE_ZONE == 0.15,
        "exit_hyst_2c_preserved": EXIT_REPRICE_HYSTERESIS == 0.02,
        "force3s_preserved": EXIT_HORIZON_S == 3.0,
        "m12_preserved": M12_S == 720.0,
        "pending_create_waits": _lifecycle_gate("K", {"K": object()}, {}) == "WAIT_CREATE",
        "pending_cancel_waits": _lifecycle_gate("K", {}, {"K": object()}) == "WAIT_CANCEL",
        "clear_lifecycle_passes": _lifecycle_gate("K", {}, {}) == "CLEAR",
        "exchange_fill_clock_used_when_plausible": exchange == now - 125.0 and exchange_src == "EXCHANGE_TS_MS",
        "implausible_old_clock_falls_back_local": fallback == now and fallback_src == "LOCAL_RECEIPT_WALL_MS",
        "ghost_create_cleanup_enabled": True,
        "entry_immediate_rejoin_enabled": True,
        "exit_reject_immediate_rejoin_enabled": True,
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
        print("TAIL25 V1.1 LIFECYCLE AUDIT STATIC CHECK — NO API / NO ORDERS")
        print("=" * 156)
        for k, v in out.items():
            print(f"{k:100s}: {v}")
    if not ok:
        raise RuntimeError(f"Tail25 lifecycle audit static check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run the existing audited M12 stack with the V1.1 Tail25 child class."""
    session = Path(session).resolve()
    BASE._install_runtime_patch()

    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Tail25JoinBboLifecycleAuditEngine
    V1122.M12GuardRotatingGenerationEngine = Tail25JoinBboLifecycleAuditEngine
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
    "Tail25JoinBboLifecycleAuditEngine",
    "static_self_check",
    "run_live_process",
]
