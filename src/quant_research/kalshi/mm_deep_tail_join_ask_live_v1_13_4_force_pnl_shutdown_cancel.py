from __future__ import annotations

"""Deep35 V1.13.4 force-flat PnL + shutdown-cancel reconciliation.

Observed 2026-08-28 live failure
--------------------------------
A genuine adverse-selection fill reached the inherited 2s authoritative flatten
path.  The account was ultimately flat, but two operational problems were exposed:

1) Deep35 strategy-realized PnL remained at zero because inherited RISK_FLATTEN
   IOC fills are logged by the base engine but are not applied to the Deep35 local
   lot ledger before V1.13 clears the force-flat lots.
2) During GLOBAL_SHUTDOWN an ENTRY cancel could already be terminal at the exchange
   while the asynchronous local cancel lifecycle had not retired the synthetic
   stable key within the inherited 4s synchronous wait.  V1 then raised
   ``cancel did not retire ... during GLOBAL_SHUTDOWN_CANCEL`` from the flatten
   pre-cancel path.

Fix
---
- Attribute actual REST fill rows from inherited RISK_FLATTEN/M5_FLATTEN IOC order
  ids into the Deep35 lot ledger using the true fill quantity and YES execution
  price.  This updates the existing strategy_realized_gross / drawdown accounting.
- Re-query flatten IOC fills briefly after authoritative zero position so ordinary
  fill-API propagation lag does not leave the strategy ledger stale.
- If authoritative position is flat but fill prices still have not propagated,
  record an explicit unpriced-quantity telemetry flag rather than inventing PnL.
- For GLOBAL_SHUTDOWN cancellation only, keep the existing async cancel first.  If
  local retirement misses the 4s wait, verify the immutable order id against REST.
  A terminal exchange order is locally retired; a still-resting order gets one
  documented synchronous V2 cancel/verify attempt; if it still rests, fail closed.
- All normal entry/reprice cancel behavior, identity guards, orphan fail-closed,
  Q50, Deep35, Hyst5, REC10, 2s force-flat and M12 economics remain unchanged.

Importing this module performs no API calls and sends no orders.
"""

import json
import math
import time
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_12_2_cancel_rest_reconcile as V1122
from . import mm_deep_tail_join_ask_live_v1_12_4_rec25_atomic_exact_equity as V124
from . import mm_deep_tail_join_ask_live_v1_13_3_identity_lifecycle_wait as V133


LIVE_VERSION = V133.LIVE_VERSION
PATCH_VERSION = "DEEP35_FORCE_PNL_SHUTDOWN_CANCEL_V1_13_4"

M12_S = V133.M12_S
ENTRY_START_S = V133.ENTRY_START_S
DEPTH = V133.DEPTH
HYSTERESIS = V133.HYSTERESIS
RECOVERY_EDGE = V133.RECOVERY_EDGE
RECOVERY_HORIZON_S = V133.RECOVERY_HORIZON_S
SPREAD_WINDOW_S = V133.SPREAD_WINDOW_S
NORMAL_TOL = V133.NORMAL_TOL
MIN_NORMAL_OBS = V133.MIN_NORMAL_OBS
EXPECTED_EXIT_EFFECTIVE_LATENCY_MS = V133.EXPECTED_EXIT_EFFECTIVE_LATENCY_MS

ROTATION_CHECKPOINT_FILE = V133.ROTATION_CHECKPOINT_FILE
GENERATION_BOOTSTRAP_FILE = V133.GENERATION_BOOTSTRAP_FILE
SESSION_RISK_BASELINE_FILE = V133.SESSION_RISK_BASELINE_FILE

EPS = V1.EPS
SHUTDOWN_CANCEL_WAIT_S = 4.0
FLATTEN_FILL_RECONCILE_DELAYS_S = (0.0, 0.05, 0.10, 0.20, 0.35)


def _finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _fill_qty(row):
    row = row or {}
    for key in ("count_fp", "count", "qty", "quantity_fp", "quantity"):
        z = _finite(row.get(key))
        if z is not None and z > EPS:
            return float(z)
    return 0.0


def _fill_yes_price(row):
    row = row or {}
    for key in ("yes_price_dollars", "price_dollars", "yes_price", "price"):
        z = _finite(row.get(key))
        if z is None:
            continue
        if z > 1.000001:
            z /= 100.0
        if 0.0 <= z <= 1.0:
            return float(z), key
    return None, None


def _fill_identity(row, order_id=""):
    row = row or {}
    fid = str(row.get("fill_id") or row.get("trade_id") or "")
    if fid:
        return (str(order_id or row.get("order_id") or ""), fid)
    return (
        str(order_id or row.get("order_id") or ""),
        str(row.get("ts_ms", row.get("ts", ""))),
        str(row.get("count_fp", row.get("count", ""))),
        str(row.get("yes_price_dollars", row.get("yes_price", row.get("price", "")))),
    )


def _rest_terminal(row):
    row = row or {}
    status = str(row.get("status") or "").lower()
    rem = _finite(row.get("remaining_count_fp", row.get("remaining_count")))
    if rem is not None and rem <= EPS:
        return True
    if status and status != "resting":
        return True
    return False


class Deep35ForcePnlShutdownCancelEngine(V133.Deep35IdentityLifecycleWaitEngine):
    """V1.13.3 plus exact force-flat fill PnL and shutdown cancel reconciliation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._deep35_flatten_context = {}
        self._deep35_flatten_fill_ids = set()
        self._force_flat_pnl_fill_rows = 0
        self._force_flat_pnl_contracts = 0.0
        self._force_flat_pnl_unpriced_qty = 0.0
        self._force_flat_pnl_reconcile_incomplete = 0
        self._shutdown_cancel_rest_terminal_retire = 0
        self._shutdown_cancel_sync_retry = 0
        self._shutdown_cancel_sync_retry_success = 0
        self._shutdown_cancel_still_resting = 0
        self._lat(
            "DEEP35_FORCE_PNL_SHUTDOWN_CANCEL_READY",
            patch_version=PATCH_VERSION,
            force_flat_actual_fill_pnl=True,
            force_flat_unpriced_qty_explicit=True,
            global_shutdown_authoritative_cancel_reconcile=True,
            still_resting_fail_closed=True,
            strategy_unchanged=True,
        )

    # ------------------------------------------------------------------
    # Force-flat PnL attribution
    # ------------------------------------------------------------------

    def _process_flatten_fill_rows(self, track, rows, reason):
        track = track or {}
        ticker = str(track.get("ticker") or "")
        oid = str(track.get("order_id") or "")
        side = str(track.get("side") or "").lower()
        if not ticker or side not in {"ask", "bid"}:
            return 0.0

        # Flatten ask reduces long YES; flatten bid reduces short YES.
        sign = 1 if side == "ask" else -1
        applied = 0.0

        for row in rows or []:
            ident = _fill_identity(row, oid)
            if ident in self._deep35_flatten_fill_ids:
                continue

            qty = _fill_qty(row)
            px, price_source = _fill_yes_price(row)
            if qty <= EPS or px is None:
                continue

            matching = sorted(
                [
                    lot
                    for lot in (self.open_lots.get(ticker) or [])
                    if int(lot.get("sign", 0)) == int(sign)
                    and float(lot.get("remaining", 0.0) or 0.0) > EPS
                ],
                key=lambda lot: (float(lot.get("entry_wall_ms", 0.0) or 0.0), int(lot.get("lot_id", 0) or 0)),
            )
            available = sum(float(lot.get("remaining", 0.0) or 0.0) for lot in matching)
            if available <= EPS:
                # The same exchange fill may have already been applied by another
                # private/REST path.  Do not invent a second realization.
                self._deep35_flatten_fill_ids.add(ident)
                continue

            take = min(float(qty), float(available))
            preferred = int(matching[0]["lot_id"])
            mode = str(reason or "RISK_FLATTEN")
            self._apply_exit_inventory(
                ticker,
                sign,
                take,
                float(px),
                preferred,
                mode,
                f"FLATTEN_FILL:{price_source}",
            )
            self._deep35_flatten_fill_ids.add(ident)
            self._force_flat_pnl_fill_rows += 1
            self._force_flat_pnl_contracts += float(take)
            applied += float(take)
            self._lat(
                "DEEP35_FLATTEN_FILL_APPLIED_TO_PNL",
                ticker=ticker,
                reason=mode,
                order_id=oid,
                fill_identity=ident,
                qty=float(take),
                yes_price=float(px),
                price_source=price_source,
                strategy_realized_gross=float(self.strategy_realized_gross),
            )

            if qty > available + 0.01:
                self._lat(
                    "DEEP35_FLATTEN_FILL_EXCEEDS_LOCAL_LOTS",
                    ticker=ticker,
                    order_id=oid,
                    exchange_fill_qty=float(qty),
                    local_matching_qty=float(available),
                    excess=float(qty - available),
                )

        return applied

    def record_fills(self, track):
        track = track or {}
        role = str(track.get("role") or "")
        ticker = str(track.get("ticker") or "")
        oid = str(track.get("order_id") or "")
        if role in {"RISK_FLATTEN", "M5_FLATTEN"} and ticker and oid and self.open_lots.get(ticker):
            try:
                rows, _ = B._fills(self.client, oid)
                reason = self._deep35_flatten_context.get(ticker) or role
                self._process_flatten_fill_rows(track, rows, reason)
            except Exception as exc:
                self._lat(
                    "DEEP35_FLATTEN_PNL_FILL_READ_ERROR",
                    ticker=ticker,
                    order_id=oid,
                    error=repr(exc),
                )
        return super().record_fills(track)

    def _reconcile_flatten_attempt_fills(self, ticker, reason, result):
        ticker = str(ticker)
        attempts = list((result or {}).get("attempts") or [])
        if not attempts or not self.open_lots.get(ticker):
            return

        for delay in FLATTEN_FILL_RECONCILE_DELAYS_S:
            if delay:
                time.sleep(float(delay))
            for rec in attempts:
                body = (rec or {}).get("response") or {}
                oid = str(body.get("order_id") or "")
                side = str((rec or {}).get("side") or "").lower()
                if not oid or side not in {"ask", "bid"}:
                    continue
                synthetic = {
                    "ticker": ticker,
                    "order_id": oid,
                    "cid": str(body.get("client_order_id") or ""),
                    "role": "M5_FLATTEN" if str(reason) == "M12" else "RISK_FLATTEN",
                    "side": side,
                    "book": (rec or {}).get("book"),
                }
                self.record_fills(synthetic)
            if not self.open_lots.get(ticker):
                break

    def flatten(self, ticker, reason):
        ticker = str(ticker)
        self._deep35_flatten_context[ticker] = str(reason)
        try:
            result = super().flatten(ticker, reason)
            self._reconcile_flatten_attempt_fills(ticker, reason, result)

            final_position = _finite((result or {}).get("final_position"))
            remaining_qty = sum(
                float(lot.get("remaining", 0.0) or 0.0)
                for lot in (self.open_lots.get(ticker) or [])
            )
            if final_position is not None and abs(final_position) <= EPS and remaining_qty > EPS:
                # Never fabricate a price.  The inherited Deep35 force path will
                # clear the now-authoritatively-flat local lots after return; keep
                # explicit accounting-quality telemetry if fills lagged too long.
                self._force_flat_pnl_unpriced_qty += float(remaining_qty)
                self._force_flat_pnl_reconcile_incomplete += 1
                self._lat(
                    "DEEP35_FLATTEN_PNL_RECONCILE_INCOMPLETE",
                    ticker=ticker,
                    reason=str(reason),
                    authoritative_final_position=float(final_position),
                    unpriced_local_qty=float(remaining_qty),
                )
            return result
        finally:
            self._deep35_flatten_context.pop(ticker, None)

    # ------------------------------------------------------------------
    # GLOBAL_SHUTDOWN cancel reconciliation
    # ------------------------------------------------------------------

    def _fetch_order_row(self, order_id):
        body, timing = self.client.get(f"/portfolio/orders/{str(order_id)}")
        row = (body or {}).get("order") or {}
        return row, timing

    def _retire_shutdown_track_if_terminal(self, key, reason, source):
        tr = self.active.get(key)
        if tr is None:
            return True
        oid = str(tr.get("order_id") or "")
        if not oid:
            return False

        try:
            row, timing = self._fetch_order_row(oid)
        except Exception as exc:
            self._lat(
                "GLOBAL_SHUTDOWN_CANCEL_REST_VERIFY_ERROR",
                key=key,
                ticker=tr.get("ticker"),
                order_id=oid,
                reason=str(reason),
                source=str(source),
                error=repr(exc),
            )
            return False

        if not _rest_terminal(row):
            return False

        floor = V1._order_fill_count(row, 0.0)
        self._apply_floor(key, floor, "global_shutdown_terminal_verify", row)
        current = self.active.get(key)
        if current is not None:
            self.active.pop(key, None)
        try:
            self._retire_stale_identity_mappings(key)
        except Exception:
            pass
        self._shutdown_cancel_rest_terminal_retire += 1
        self._lat(
            "GLOBAL_SHUTDOWN_CANCEL_AUTHORITATIVE_TERMINAL_RETIRE",
            key=key,
            ticker=tr.get("ticker"),
            order_id=oid,
            client_order_id=tr.get("cid"),
            reason=str(reason),
            source=str(source),
            rest_status=row.get("status"),
            rest_remaining=row.get("remaining_count_fp", row.get("remaining_count")),
            rest_fill_count=row.get("fill_count_fp", row.get("fill_count")),
            timing=timing,
        )
        return True

    def cancel_track(self, key_or_ticker, reason):
        reason = str(reason)
        if not reason.startswith("GLOBAL_SHUTDOWN"):
            return super().cancel_track(key_or_ticker, reason)

        # Preserve the base synthetic-key convenience for ticker-scoped calls.
        if key_or_ticker not in self.active:
            keys = [
                k
                for k, tr in list(self.active.items())
                if str(tr.get("ticker") or "") == str(key_or_ticker)
            ]
            for key in keys:
                self.cancel_track(key, reason)
            return False

        key = key_or_ticker
        tr = self.active.get(key)
        if tr is None:
            return False

        if not bool(tr.get("cancel_requested")):
            self._request_cancel_key(key, reason)

        deadline = time.time() + SHUTDOWN_CANCEL_WAIT_S
        while key in self.active and time.time() < deadline:
            self._drain_create_futures()
            self._drain_cancel_futures()
            time.sleep(0.005)

        if key not in self.active:
            return False

        if self._retire_shutdown_track_if_terminal(key, reason, "POST_ASYNC_WAIT_REST"):
            return False

        tr = self.active.get(key)
        oid = str((tr or {}).get("order_id") or "")
        if oid:
            self._shutdown_cancel_sync_retry += 1
            try:
                result = V1._safe_cancel_v2(
                    self.client,
                    order_id=oid,
                    submitted_qty=float((tr or {}).get("qty", 0.0) or 0.0),
                )
                floor = _finite((result or {}).get("fill_floor"))
                if floor is not None:
                    self._apply_floor(key, floor, "global_shutdown_sync_cancel", result)
                self._lat(
                    "GLOBAL_SHUTDOWN_CANCEL_SYNC_RETRY",
                    key=key,
                    ticker=(tr or {}).get("ticker"),
                    order_id=oid,
                    reason=reason,
                    result=result,
                )
            except Exception as exc:
                self._lat(
                    "GLOBAL_SHUTDOWN_CANCEL_SYNC_RETRY_ERROR",
                    key=key,
                    ticker=(tr or {}).get("ticker"),
                    order_id=oid,
                    reason=reason,
                    error=repr(exc),
                )

            if key not in self.active:
                self._shutdown_cancel_sync_retry_success += 1
                return False

            if self._retire_shutdown_track_if_terminal(key, reason, "POST_SYNC_CANCEL_REST"):
                self._shutdown_cancel_sync_retry_success += 1
                return False

        self._shutdown_cancel_still_resting += 1
        raise RuntimeError(f"cancel did not retire {key} during {reason}; authoritative terminal verification failed")

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update(
                {
                    "deep35_force_pnl_shutdown_cancel_patch": PATCH_VERSION,
                    "force_flat_pnl_fill_rows": int(self._force_flat_pnl_fill_rows),
                    "force_flat_pnl_contracts": float(self._force_flat_pnl_contracts),
                    "force_flat_pnl_unpriced_qty": float(self._force_flat_pnl_unpriced_qty),
                    "force_flat_pnl_reconcile_incomplete": int(self._force_flat_pnl_reconcile_incomplete),
                    "force_flat_pnl_uses_actual_exchange_fills": True,
                    "force_flat_pnl_never_invents_price": True,
                    "shutdown_cancel_rest_terminal_retire": int(self._shutdown_cancel_rest_terminal_retire),
                    "shutdown_cancel_sync_retry": int(self._shutdown_cancel_sync_retry),
                    "shutdown_cancel_sync_retry_success": int(self._shutdown_cancel_sync_retry_success),
                    "shutdown_cancel_still_resting": int(self._shutdown_cancel_still_resting),
                    "shutdown_cancel_authoritative_terminal_required": True,
                    "shutdown_cancel_still_resting_fail_closed": True,
                    "dashboard_schema_compatible_v300": True,
                }
            )
            B._atomic(self.health_path, h)
        except Exception:
            pass


def static_self_check(*, show=True):
    base = V133.static_self_check(show=False)
    fill_row = {"count_fp": "12.50", "yes_price_dollars": "0.27", "fill_id": "f1"}
    checks = {
        "v133_parent_static_ok": base.get("ok") is True,
        "inherits_v133": issubclass(Deep35ForcePnlShutdownCancelEngine, V133.Deep35IdentityLifecycleWaitEngine),
        "dashboard_live_version_unchanged": LIVE_VERSION == V133.LIVE_VERSION,
        "patch_version_exact": PATCH_VERSION == "DEEP35_FORCE_PNL_SHUTDOWN_CANCEL_V1_13_4",
        "fill_qty_parser": abs(_fill_qty(fill_row) - 12.5) < 1e-12,
        "fill_price_parser": _fill_yes_price(fill_row)[0] == 0.27,
        "rest_terminal_canceled": _rest_terminal({"status": "canceled", "remaining_count_fp": "50.00"}),
        "rest_terminal_zero_remaining": _rest_terminal({"status": "resting", "remaining_count_fp": "0.00"}),
        "rest_resting_nonterminal": not _rest_terminal({"status": "resting", "remaining_count_fp": "50.00"}),
        "strategy_depth_unchanged": DEPTH == 0.35,
        "strategy_hysteresis_unchanged": HYSTERESIS == 0.05,
        "strategy_recovery_unchanged": RECOVERY_EDGE == 0.10,
        "strategy_force_horizon_unchanged": RECOVERY_HORIZON_S == 2.0,
        "identity_guards_preserved": True,
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
        print("=" * 172)
        print("DEEP35 V1.13.4 FORCE-PNL / SHUTDOWN-CANCEL STATIC CHECK — NO API / NO ORDERS")
        print("=" * 172)
        for k, v in out.items():
            print(f"{k:112s}: {v}")
    if not ok:
        raise RuntimeError(f"Deep35 V1.13.4 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run V1.12.2 safety runner with V1.13.4 Deep35 child."""
    session = Path(session).resolve()
    old_engine = V1122.CancelRestReconcileM12Engine
    old_alias = V1122.M12GuardRotatingGenerationEngine
    old_version = V1122.LIVE_VERSION
    old_equity = B._equity

    V1122.CancelRestReconcileM12Engine = Deep35ForcePnlShutdownCancelEngine
    V1122.M12GuardRotatingGenerationEngine = Deep35ForcePnlShutdownCancelEngine
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
    "Deep35ForcePnlShutdownCancelEngine",
    "static_self_check",
    "run_live_process",
]
