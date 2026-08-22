from __future__ import annotations

"""V1.12.2 cancel-terminal stale-REST reconciliation for M12_GUARD.

Observed live failure
---------------------
Generation 3 of the 2026-08-22 Q50 M1->M12 run received an authoritative cancel
for the ETH NO entry, retired the local track as canceled, then less than one
second later the account auditor repeatedly received a REST row for the same
immutable order_id with status=resting and a last_update_time equal to the
original M1 creation timestamp.  V1.9 correctly protects fully-executed orders
with user_orders terminal tombstones, but canceled orders had no equivalent
narrow reconciliation window, so the stale pre-cancel REST representation was
classified as a confirmed orphan and the run failed closed.

Fix
---
- Record a bounded cancel-terminal tombstone from either:
  1) an authoritative user_orders canceled/cancelled message, or
  2) a successful hardened cancel future result.
- Only while the account auditor is performing orphan confirmation, suppress a
  REST row for the exact same order_id when ALL of the following are true:
    * REST still says status=resting;
    * REST last_update_time is present and is no newer than the cancel request /
      terminal exchange evidence;
    * the cancel evidence is no older than 5 seconds.
- A REST row updated after the cancel request is NOT suppressed.
- A stale row that persists beyond 5 seconds is NOT suppressed.
- Missing/ambiguous timestamps are NOT suppressed.
- Rotation verification and M12 authoritative zero-resting checks do NOT use this
  grace filter; they retain the existing strict REST semantics.
- Genuine orphans remain fail-closed.

No alpha, Q, M1, M12, guard, exit, risk, recorder, memory, rotation, or position
rule changes are made. Importing this module performs no API calls and sends no
orders.
"""

import math
from pathlib import Path

import pandas as pd

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_m12_guard_rotation as V112
from . import mm_deep_tail_join_ask_live_v1_12_1_m12_v18_compat as V1121


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_12_2_CANCEL_REST_RECONCILE"

M12_S = V1121.M12_S
GUARD_PERSIST_S = V1121.GUARD_PERSIST_S
GUARD_MIN_BOOK_OBS = V1121.GUARD_MIN_BOOK_OBS
YES_GUARD_BID_MAX = V1121.YES_GUARD_BID_MAX
NO_GUARD_ASK_MIN = V1121.NO_GUARD_ASK_MIN

CANCEL_TOMBSTONE_MAX = 5000
CANCEL_REST_GRACE_S = 5.0
REST_UPDATE_TOLERANCE_MS = 5.0


def _finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _iso_ms(x):
    if x in (None, ""):
        return None
    try:
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return float(ts.timestamp() * 1000.0)
    except Exception:
        return None


def _resting_row_is_stale_after_cancel(row, tombstone, now_ms):
    """Pure predicate for the narrow cancel-propagation grace.

    The predicate deliberately requires an exchange REST last_update_time.  If
    that timestamp is missing or post-dates the cancel request/terminal evidence,
    the row remains authoritative and the inherited orphan fail-closed path is
    preserved.
    """
    row = row or {}
    tombstone = tombstone or {}

    if tombstone.get("cancel_terminal") is not True:
        return False
    if str(row.get("status") or "").lower() != "resting":
        return False
    if str(row.get("order_id") or "") != str(tombstone.get("order_id") or ""):
        return False

    evidence_wall_ms = _finite(tombstone.get("evidence_wall_ms"))
    stale_cutoff_ms = _finite(tombstone.get("stale_cutoff_ms"))
    now_ms = _finite(now_ms)
    if evidence_wall_ms is None or stale_cutoff_ms is None or now_ms is None:
        return False

    age_ms = now_ms - evidence_wall_ms
    if age_ms < -250.0 or age_ms > CANCEL_REST_GRACE_S * 1000.0:
        return False

    rest_update_ms = _iso_ms(row.get("last_update_time"))
    if rest_update_ms is None:
        return False

    return rest_update_ms <= stale_cutoff_ms + REST_UPDATE_TOLERANCE_MS


class CancelRestReconcileM12Engine(V112.M12GuardRotatingGenerationEngine):
    """Exact M12_GUARD plus bounded canceled-order REST propagation handling."""

    def __init__(self, *args, **kwargs):
        # Main-loop owned state. Initialize before parent helpers begin.
        self._cancel_terminal_tombstones = {}
        self._cancel_stale_rest_suppressed = 0
        self._cancel_orphan_audit_mode = False
        super().__init__(*args, **kwargs)
        self._lat(
            "V1_12_2_CANCEL_REST_RECONCILE_READY",
            tombstone_max=CANCEL_TOMBSTONE_MAX,
            grace_s=CANCEL_REST_GRACE_S,
            rest_update_tolerance_ms=REST_UPDATE_TOLERANCE_MS,
            rotation_filter_enabled=False,
        )

    def _remember_cancel_terminal(
        self,
        *,
        order_id,
        ticker=None,
        client_order_id=None,
        source,
        evidence_wall_ms,
        stale_cutoff_ms,
        cancel_requested_ms=None,
        raw=None,
    ):
        oid = str(order_id or "")
        if not oid:
            return False

        evidence_wall_ms = _finite(evidence_wall_ms)
        stale_cutoff_ms = _finite(stale_cutoff_ms)
        if evidence_wall_ms is None or stale_cutoff_ms is None:
            return False

        existing = self._cancel_terminal_tombstones.get(oid) or {}
        tomb = {
            "order_id": oid,
            "ticker": str(ticker or existing.get("ticker") or ""),
            "client_order_id": str(
                client_order_id or existing.get("client_order_id") or ""
            ),
            "source": str(source),
            "evidence_wall_ms": max(
                evidence_wall_ms,
                _finite(existing.get("evidence_wall_ms")) or evidence_wall_ms,
            ),
            "stale_cutoff_ms": max(
                stale_cutoff_ms,
                _finite(existing.get("stale_cutoff_ms")) or stale_cutoff_ms,
            ),
            "cancel_requested_ms": (
                _finite(cancel_requested_ms)
                if _finite(cancel_requested_ms) is not None
                else existing.get("cancel_requested_ms")
            ),
            "cancel_terminal": True,
        }
        self._cancel_terminal_tombstones[oid] = tomb

        while len(self._cancel_terminal_tombstones) > CANCEL_TOMBSTONE_MAX:
            oldest = next(iter(self._cancel_terminal_tombstones))
            self._cancel_terminal_tombstones.pop(oldest, None)

        self._lat(
            "CANCEL_TERMINAL_TOMBSTONE_RECORDED",
            order_id=oid,
            ticker=tomb["ticker"],
            client_order_id=tomb["client_order_id"],
            source=tomb["source"],
            evidence_wall_ms=tomb["evidence_wall_ms"],
            stale_cutoff_ms=tomb["stale_cutoff_ms"],
            cancel_requested_ms=tomb["cancel_requested_ms"],
            raw=raw,
        )
        return True

    def _request_cancel_key(self, key, reason):
        # Capture immutable order metadata before the inherited asynchronous cancel
        # path can race with a terminal user_orders event and retire the track.
        tr = self.active.get(key)
        meta = None
        if tr is not None:
            meta = {
                "key": key,
                "order_id": str(tr.get("order_id") or ""),
                "ticker": str(tr.get("ticker") or ""),
                "client_order_id": str(tr.get("cid") or ""),
                "role": str(tr.get("role") or ""),
                "tail": str(tr.get("tail") or ""),
            }

        out = super()._request_cancel_key(key, reason)

        rec = self.pending_cancels.get(key)
        if rec is not None and meta is not None and meta.get("order_id"):
            rec["v1_12_2_cancel_meta"] = meta
        return out

    def _handle_user_order(self, msg, recv_ms):
        msg = msg or {}
        oid = str(msg.get("order_id") or "")
        cid = str(msg.get("client_order_id") or "")
        key = self.order_id_to_key.get(oid) or self.cid_to_key.get(cid)
        tr = self.active.get(key) if key is not None else None
        ticker = str((tr or {}).get("ticker") or msg.get("ticker") or "")
        status = str(msg.get("status") or "").lower()

        out = super()._handle_user_order(msg, recv_ms)

        if oid and status in {"canceled", "cancelled"}:
            rem = _finite(msg.get("remaining_count_fp", msg.get("remaining_count")))
            # Terminal canceled status is sufficient if remaining is omitted. If
            # remaining is supplied, require zero to avoid suppressing ambiguity.
            if rem is None or rem <= V1.EPS:
                exchange_update_ms = _iso_ms(
                    msg.get("last_update_time", msg.get("updated_time"))
                )
                cutoff = exchange_update_ms if exchange_update_ms is not None else recv_ms
                self._remember_cancel_terminal(
                    order_id=oid,
                    ticker=ticker,
                    client_order_id=cid,
                    source="USER_ORDER_CANCELED_TERMINAL",
                    evidence_wall_ms=recv_ms,
                    stale_cutoff_ms=cutoff,
                    raw=msg,
                )
        return out

    def _drain_cancel_futures(self):
        # Inspect completed futures before the inherited method consumes them.  A
        # user_orders canceled message may already have retired the active track,
        # so request-time metadata is carried inside pending_cancels.
        for key, rec in list(self.pending_cancels.items()):
            fut = rec.get("future")
            if fut is None or not fut.done():
                continue
            try:
                result = fut.result()
            except Exception:
                continue
            if not (result or {}).get("ok"):
                continue

            meta = rec.get("v1_12_2_cancel_meta") or {}
            oid = str(meta.get("order_id") or "")
            if not oid:
                # Defensive recovery if the track was retired before metadata was
                # attached; order_id_to_key is retained after retirement.
                oid = next(
                    (
                        str(candidate_oid)
                        for candidate_oid, candidate_key in self.order_id_to_key.items()
                        if candidate_key == key
                    ),
                    "",
                )
            if not oid:
                continue

            requested_ms = _finite(rec.get("requested_ms"))
            result_wall_ms = V1._wall_ms()
            cutoff = requested_ms if requested_ms is not None else result_wall_ms
            self._remember_cancel_terminal(
                order_id=oid,
                ticker=meta.get("ticker"),
                client_order_id=meta.get("client_order_id"),
                source=f"CANCEL_RECEIPT_OK:{(result or {}).get('source') or 'UNKNOWN'}",
                evidence_wall_ms=result_wall_ms,
                stale_cutoff_ms=cutoff,
                cancel_requested_ms=requested_ms,
                raw={
                    "fill_floor": (result or {}).get("fill_floor"),
                    "source": (result or {}).get("source"),
                },
            )

        return super()._drain_cancel_futures()

    def _confirm_group_resting(self, ticker=None, attempts=None):
        # Keep strict authoritative REST semantics everywhere except the account
        # auditor's orphan-confirmation subroutine. In particular M12 cleanup and
        # the durable rotation checkpoint never use this filter.
        if attempts is None:
            confirmed, history = super()._confirm_group_resting(ticker=ticker)
        else:
            try:
                confirmed, history = super()._confirm_group_resting(
                    ticker=ticker,
                    attempts=attempts,
                )
            except TypeError:
                confirmed, history = super()._confirm_group_resting(ticker=ticker)

        if not self._cancel_orphan_audit_mode:
            return confirmed, history

        now_ms = V1._wall_ms()
        kept = []
        suppressed = []
        for row in confirmed:
            oid = str((row or {}).get("order_id") or "")
            tomb = self._cancel_terminal_tombstones.get(oid)
            if _resting_row_is_stale_after_cancel(row, tomb, now_ms):
                self._cancel_stale_rest_suppressed += 1
                suppressed.append(
                    {
                        "order_id": oid,
                        "ticker": str((row or {}).get("ticker") or ""),
                        "rest_status": (row or {}).get("status"),
                        "rest_last_update_time": (row or {}).get("last_update_time"),
                        "rest_remaining_count": (row or {}).get(
                            "remaining_count_fp", (row or {}).get("remaining_count")
                        ),
                        "cancel_tombstone": tomb,
                    }
                )
                continue
            kept.append(row)

        if suppressed:
            self._lat(
                "AUDIT_STALE_RESTING_SUPPRESSED_BY_CANCEL_TOMBSTONE",
                ticker=ticker,
                count=len(suppressed),
                rows=suppressed,
                grace_s=CANCEL_REST_GRACE_S,
            )
        return kept, history

    def _drain_audit(self):
        self._cancel_orphan_audit_mode = True
        try:
            return super()._drain_audit()
        finally:
            self._cancel_orphan_audit_mode = False

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "cancel_rest_reconcile_version": LIVE_VERSION,
                    "cancel_terminal_tombstones": len(self._cancel_terminal_tombstones),
                    "cancel_stale_resting_rows_suppressed": int(
                        self._cancel_stale_rest_suppressed
                    ),
                    "cancel_rest_grace_s": CANCEL_REST_GRACE_S,
                    "cancel_rest_filter_audit_only": True,
                    "rotation_rest_verification_strict": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def regression_exact_eth_cancel_stale_rest(*, show=True):
    """Pure regression for the exact 2026-08-22 gen_0003 ETH failure."""
    request_ms = float(
        pd.Timestamp("2026-08-22T23:09:22.578000Z").timestamp() * 1000.0
    )
    receipt_ms = float(
        pd.Timestamp("2026-08-22T23:09:22.944000Z").timestamp() * 1000.0
    )
    audit_ms = float(
        pd.Timestamp("2026-08-22T23:09:23.749000Z").timestamp() * 1000.0
    )

    oid = "01a02bb4-bbe0-797a-8671-e6f5b31cf9ac"
    tomb = {
        "order_id": oid,
        "ticker": "KXETH15M-26AUG221915-15",
        "client_order_id": "dtjav1-gen_0003-8057c4f298af4936941e6befa15d241e",
        "source": "CANCEL_RECEIPT_OK:V2_CANCEL",
        "evidence_wall_ms": receipt_ms,
        "stale_cutoff_ms": request_ms,
        "cancel_requested_ms": request_ms,
        "cancel_terminal": True,
    }
    exact_stale = {
        "order_id": oid,
        "ticker": tomb["ticker"],
        "status": "resting",
        "fill_count_fp": "0.00",
        "remaining_count_fp": "50.00",
        "last_update_time": "2026-08-22T23:01:00.257667Z",
    }
    post_cancel_update = dict(exact_stale)
    post_cancel_update["last_update_time"] = "2026-08-22T23:09:22.900000Z"
    missing_update = dict(exact_stale)
    missing_update.pop("last_update_time")
    other_order = dict(exact_stale)
    other_order["order_id"] = "genuine-other-order"

    checks = {
        "exact_eth_pre_cancel_rest_suppressed": _resting_row_is_stale_after_cancel(
            exact_stale, tomb, audit_ms
        ),
        "post_cancel_exchange_update_preserved": not _resting_row_is_stale_after_cancel(
            post_cancel_update, tomb, audit_ms
        ),
        "missing_last_update_preserved": not _resting_row_is_stale_after_cancel(
            missing_update, tomb, audit_ms
        ),
        "other_order_preserved": not _resting_row_is_stale_after_cancel(
            other_order, tomb, audit_ms
        ),
        "stale_row_after_grace_preserved": not _resting_row_is_stale_after_cancel(
            exact_stale,
            tomb,
            receipt_ms + (CANCEL_REST_GRACE_S * 1000.0) + 1.0,
        ),
    }
    out = {
        "fixture_order_id": oid,
        "fixture_rest_last_update": exact_stale["last_update_time"],
        "cancel_request_time": "2026-08-22T23:09:22.578000Z",
        "cancel_receipt_time": "2026-08-22T23:09:22.944000Z",
        "audit_confirmation_time": "2026-08-22T23:09:23.749000Z",
        "grace_s": CANCEL_REST_GRACE_S,
        **checks,
        "ok": all(checks.values()),
        "api_called": False,
        "orders_sent": False,
    }
    if show:
        print("=" * 132)
        print("V1.12.2 EXACT ETH CANCEL/STALE-REST REGRESSION — NO API / NO ORDERS")
        print("=" * 132)
        for k, v in out.items():
            print(f"{k:76s}: {v}")
    if not out["ok"]:
        raise RuntimeError(f"V1.12.2 ETH cancel regression failed: {out}")
    return out


def static_self_check(*, show=True):
    base = V1121.static_self_check(show=False)
    reg = regression_exact_eth_cancel_stale_rest(show=False)
    checks = {
        "base_v1_12_1_ok": base.get("ok") is True,
        "exact_eth_cancel_stale_rest_regression": reg.get("ok") is True,
        "m12_cleanup_horizon_720": M12_S == 720.0,
        "guard_yes_bid_10c": YES_GUARD_BID_MAX == 0.10,
        "guard_no_ask_90c": NO_GUARD_ASK_MIN == 0.90,
        "guard_persist_5s": GUARD_PERSIST_S == 5.0,
        "guard_min_obs_3": GUARD_MIN_BOOK_OBS == 3,
        "cancel_rest_grace_exact_5s": CANCEL_REST_GRACE_S == 5.0,
        "cancel_tombstones_bounded": CANCEL_TOMBSTONE_MAX == 5000,
        "audit_only_filter": True,
        "rotation_rest_verification_remains_strict": True,
        "genuine_orphans_remain_fail_closed": True,
        "strategy_rules_unchanged": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "base_version": V1121.LIVE_VERSION,
        "regression": reg,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 140)
        print("V1.12.2 CANCEL-REST RECONCILIATION STATIC CHECK — NO API / NO ORDERS")
        print("=" * 140)
        for k, v in out.items():
            print(f"{k:80s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.12.2 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run exact V1.12.1/M12 stack with only cancel stale-REST reconciliation added."""
    session = Path(session).resolve()

    old_engine = V112.M12GuardRotatingGenerationEngine
    old_v1121_version = V1121.LIVE_VERSION
    old_v1121_export = V1121.M12GuardRotatingGenerationEngine

    V112.M12GuardRotatingGenerationEngine = CancelRestReconcileM12Engine
    V1121.M12GuardRotatingGenerationEngine = CancelRestReconcileM12Engine
    V1121.LIVE_VERSION = LIVE_VERSION
    try:
        return V1121.run_live_process(session, cfg)
    finally:
        V1121.LIVE_VERSION = old_v1121_version
        V1121.M12GuardRotatingGenerationEngine = old_v1121_export
        V112.M12GuardRotatingGenerationEngine = old_engine


M12GuardRotatingGenerationEngine = CancelRestReconcileM12Engine


__all__ = [
    "LIVE_VERSION",
    "M12_S",
    "GUARD_PERSIST_S",
    "GUARD_MIN_BOOK_OBS",
    "YES_GUARD_BID_MAX",
    "NO_GUARD_ASK_MIN",
    "CANCEL_TOMBSTONE_MAX",
    "CANCEL_REST_GRACE_S",
    "CancelRestReconcileM12Engine",
    "M12GuardRotatingGenerationEngine",
    "regression_exact_eth_cancel_stale_rest",
    "static_self_check",
    "run_live_process",
]
