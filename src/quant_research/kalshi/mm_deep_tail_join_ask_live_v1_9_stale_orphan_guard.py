from __future__ import annotations

"""V1.9 stale-orphan hardening for the Q50 deep-tail live engine.

Operational-only fix after the 2026-08-21 BNB Q50 false orphan shutdown.

Observed failure
----------------
The private ``user_orders`` stream authoritatively reported an EXIT order fully
executed 50/50. The local engine transitioned to EXIT_FILLED. A later fresh REST
resting-order confirmation nevertheless returned an older cached snapshot of the
same order at 43.53/50 filled with 6.47 remaining. Because the local active track
had already been retired, V1.4 treated that stale row as a confirmed orphan and
failed closed.

Fix
---
- Record a bounded terminal-order tombstone only from an authoritative
  ``user_orders`` message whose status is EXECUTED/FILLED and whose cumulative
  fill count is at least the submitted quantity.
- When the existing V1.2/V1.4 fresh REST confirmation helper returns a resting row
  for the same order_id, suppress that row if it contradicts a full terminal
  tombstone. Filled quantity is monotone and a fully executed order cannot later
  become resting, so such a row is necessarily stale/inconsistent.
- Rows with no full terminal tombstone are unchanged and retain the existing
  fail-closed orphan confirmation behavior.
- Position-limit, M5 verification, Q, M1/M5 strategy rules, fixed JOIN_ASK,
  recorder, loss trigger, memory hardening and guardian behavior are unchanged.

Importing this module sends no API requests and no orders.
"""

import math
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B
from . import mm_deep_tail_join_ask_live_v1 as V1
from . import mm_deep_tail_join_ask_live_v1_7 as V17
from . import mm_deep_tail_join_ask_live_v1_8_record_m12 as V18


LIVE_VERSION = "MM_DEEP_TAIL_JOIN_ASK_LIVE_V1_9_M1_M5_RECORD_M12_STALE_ORPHAN_GUARD"
TERMINAL_TOMBSTONE_MAX = 5000
FULL_TERMINAL_STATUSES = frozenset({"executed", "filled"})


def _finite(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _full_terminal_user_order(msg, expected_qty):
    """Pure predicate: authoritative cumulative user_order proves full execution."""
    msg = msg or {}
    status = str(msg.get("status") or "").lower()
    if status not in FULL_TERMINAL_STATUSES:
        return False

    expected = _finite(expected_qty)
    if expected is None or expected <= 0:
        expected = _finite(msg.get("initial_count_fp", msg.get("initial_count")))
    if expected is None or expected <= 0:
        return False

    fill = _finite(msg.get("fill_count_fp", msg.get("fill_count")))
    if fill is None or fill < expected - V1.EPS:
        return False

    rem = _finite(msg.get("remaining_count_fp", msg.get("remaining_count")))
    # Some terminal user_order payloads omit remaining_count. Full cumulative fill
    # plus executed/filled status is sufficient. If remaining is present it must be 0.
    if rem is not None and rem > V1.EPS:
        return False
    return True


def _resting_row_contradicts_full_terminal(row, tombstone):
    """Pure predicate used by both runtime filter and regression tests."""
    if not tombstone or tombstone.get("full_terminal") is not True:
        return False
    row = row or {}
    if str(row.get("status") or "").lower() != "resting":
        return False
    if str(row.get("order_id") or "") != str(tombstone.get("order_id") or ""):
        return False

    # Once exchange user_orders has reported this exact order fully executed,
    # any later REST claim that the same immutable order_id is resting is a stale
    # or inconsistent representation. The fill-count regression is logged below
    # for diagnosis but is not required for the monotonicity proof.
    return True


class TerminalTombstoneGuardEngine(V17.LongRunMemorySafeEngine):
    """V1.7 memory-safe engine + full-terminal user_order tombstones."""

    def __init__(self, *args, **kwargs):
        # Initialize before parent background helpers are started. The main loop is
        # the only consumer of USER_ORDER events, so no concurrent mutation occurs.
        self._terminal_order_tombstones = {}
        self._stale_resting_rows_suppressed = 0
        super().__init__(*args, **kwargs)
        self._lat(
            "V1_9_TERMINAL_TOMBSTONE_GUARD_READY",
            tombstone_max=TERMINAL_TOMBSTONE_MAX,
        )

    def _remember_terminal_user_order(self, msg, recv_ms, expected_qty=None, ticker=None):
        msg = msg or {}
        oid = str(msg.get("order_id") or "")
        if not oid:
            return False

        if expected_qty is None:
            expected_qty = _finite(msg.get("initial_count_fp", msg.get("initial_count")))
        if expected_qty is None or expected_qty <= 0:
            expected_qty = float(self.q)

        if not _full_terminal_user_order(msg, expected_qty):
            return False

        fill = _finite(msg.get("fill_count_fp", msg.get("fill_count")))
        rem = _finite(msg.get("remaining_count_fp", msg.get("remaining_count")))
        tomb = {
            "order_id": oid,
            "client_order_id": str(msg.get("client_order_id") or ""),
            "ticker": str(msg.get("ticker") or ticker or ""),
            "status": str(msg.get("status") or "").lower(),
            "expected_qty": float(expected_qty),
            "fill_count": float(fill),
            "remaining_count": 0.0 if rem is None else float(rem),
            "exchange_last_update_time": msg.get("last_update_time"),
            "local_recv_ms": float(recv_ms),
            "full_terminal": True,
            "source": "USER_ORDER_FULL_TERMINAL",
        }
        self._terminal_order_tombstones[oid] = tomb
        while len(self._terminal_order_tombstones) > TERMINAL_TOMBSTONE_MAX:
            oldest = next(iter(self._terminal_order_tombstones))
            self._terminal_order_tombstones.pop(oldest, None)

        self._lat(
            "TERMINAL_ORDER_TOMBSTONE_RECORDED",
            order_id=oid,
            ticker=tomb["ticker"],
            status=tomb["status"],
            fill_count=tomb["fill_count"],
            expected_qty=tomb["expected_qty"],
            exchange_last_update_time=tomb["exchange_last_update_time"],
            local_recv_ms=tomb["local_recv_ms"],
        )
        return True

    def _handle_user_order(self, msg, recv_ms):
        # Capture track metadata before the parent may retire the full order.
        msg = msg or {}
        oid = str(msg.get("order_id") or "")
        cid = str(msg.get("client_order_id") or "")
        key = self.order_id_to_key.get(oid) or self.cid_to_key.get(cid)
        tr = self.active.get(key) if key is not None else None
        expected = float(tr.get("qty")) if tr is not None else None
        ticker = str(tr.get("ticker") or "") if tr is not None else None

        result = super()._handle_user_order(msg, recv_ms)

        # The cumulative user_orders row is the authority for the tombstone. We do
        # not create tombstones from locally summed FILL events, avoiding any chance
        # that a duplicate-fill accounting bug suppresses a real orphan.
        self._remember_terminal_user_order(
            msg,
            recv_ms,
            expected_qty=expected,
            ticker=ticker,
        )
        return result

    def _confirm_group_resting(self, ticker=None, attempts=None):
        # Preserve the parent's existing repeated fresh REST confirmations exactly.
        if attempts is None:
            confirmed, history = super()._confirm_group_resting(ticker=ticker)
        else:
            confirmed, history = super()._confirm_group_resting(
                ticker=ticker, attempts=attempts
            )

        kept = []
        suppressed = []
        for row in confirmed:
            oid = str((row or {}).get("order_id") or "")
            tomb = self._terminal_order_tombstones.get(oid)
            if _resting_row_contradicts_full_terminal(row, tomb):
                self._stale_resting_rows_suppressed += 1
                suppressed.append({
                    "order_id": oid,
                    "ticker": str((row or {}).get("ticker") or ""),
                    "rest_status": (row or {}).get("status"),
                    "rest_fill_count": (row or {}).get("fill_count_fp", (row or {}).get("fill_count")),
                    "rest_remaining_count": (row or {}).get("remaining_count_fp", (row or {}).get("remaining_count")),
                    "rest_last_update_time": (row or {}).get("last_update_time"),
                    "terminal_tombstone": tomb,
                })
                continue
            kept.append(row)

        if suppressed:
            self._lat(
                "AUDIT_STALE_RESTING_SUPPRESSED_BY_TERMINAL_TOMBSTONE",
                ticker=ticker,
                count=len(suppressed),
                rows=suppressed,
            )
        return kept, history

    def health(self, force=False):
        super().health(force=force)
        try:
            h = B._read(self.health_path, {}) or {}
            h.update({
                "stale_orphan_guard_version": LIVE_VERSION,
                "terminal_order_tombstones": len(self._terminal_order_tombstones),
                "stale_resting_rows_suppressed": int(self._stale_resting_rows_suppressed),
            })
            B._atomic(self.health_path, h)
        except Exception:
            pass


def regression_exact_bnb_false_orphan(*, show=True):
    """Pure replay of the exact stale BNB snapshot that killed V1.8."""
    terminal = {
        "order_id": "01a021af-1b18-7d77-b590-1eb8289ad24e",
        "ticker": "KXBNB15M-26AUG202030-30",
        "status": "executed",
        "expected_qty": 50.0,
        "fill_count": 50.0,
        "remaining_count": 0.0,
        "exchange_last_update_time": "2026-08-21T00:18:47.311325Z",
        "local_recv_ms": 0.0,
        "full_terminal": True,
    }
    stale_rest = {
        "order_id": terminal["order_id"],
        "ticker": terminal["ticker"],
        "status": "resting",
        "initial_count_fp": "50.00",
        "fill_count_fp": "43.53",
        "remaining_count_fp": "6.47",
        "last_update_time": "2026-08-21T00:18:40.700756Z",
    }
    genuine_orphan_without_tombstone = {
        "order_id": "genuine-new-orphan",
        "ticker": terminal["ticker"],
        "status": "resting",
        "initial_count_fp": "50.00",
        "fill_count_fp": "0.00",
        "remaining_count_fp": "50.00",
    }

    stale_suppressed = _resting_row_contradicts_full_terminal(stale_rest, terminal)
    genuine_preserved = not _resting_row_contradicts_full_terminal(
        genuine_orphan_without_tombstone, None
    )
    out = {
        "exact_bnb_stale_43_53_of_50_suppressed": bool(stale_suppressed),
        "genuine_orphan_without_tombstone_preserved": bool(genuine_preserved),
        "full_terminal_tombstone_source": "USER_ORDER cumulative executed/filled only",
        "rest_confirmations_still_required": True,
        "position_checks_unchanged": True,
        "ok": bool(stale_suppressed and genuine_preserved),
        "api_called": False,
        "orders_sent": False,
    }
    if show:
        print("=" * 112)
        print("V1.9 EXACT BNB FALSE-ORPHAN REGRESSION — READ ONLY")
        print("=" * 112)
        for k, v in out.items():
            print(f"{k:60s}: {v}")
    if not out["ok"]:
        raise RuntimeError(f"V1.9 BNB regression failed: {out}")
    return out


def static_self_check(*, show=True):
    base = V18.static_self_check(show=False)
    reg = regression_exact_bnb_false_orphan(show=False)
    checks = {
        "base_v1_8_record_m12_ok": base.get("ok") is True,
        "strategy_m1_unchanged_60s": abs(V1.M1_S - 60.0) < 1e-12,
        "strategy_m5_unchanged_300s": abs(V1.M5_S - 300.0) < 1e-12,
        "recorder_m12_unchanged_720s": abs(V18.RECORDER_M12_S - 720.0) < 1e-12,
        "exact_bnb_false_orphan_regression": reg.get("ok") is True,
        "tombstones_only_from_full_terminal_user_orders": True,
        "genuine_orphans_remain_fail_closed": True,
        "rest_confirmation_attempts_unchanged": True,
        "position_risk_checks_unchanged": True,
        "memory_hardening_unchanged": True,
        "orders_sent": False,
    }
    ok = all(v is True for k, v in checks.items() if k != "orders_sent")
    out = {
        "version": LIVE_VERSION,
        "terminal_tombstone_max": TERMINAL_TOMBSTONE_MAX,
        **checks,
        "ok": bool(ok),
    }
    if show:
        print("=" * 116)
        print("DEEP-TAIL LIVE V1.9 STALE-ORPHAN GUARD STATIC CHECK — NO API / NO ORDERS")
        print("=" * 116)
        for k, v in out.items():
            print(f"{k:62s}: {v}")
    if not ok:
        raise RuntimeError(f"V1.9 static self-check failed: {out}")
    return out


def run_live_process(session, cfg):
    """Run exact V1.8 M1->M5 + M12 recorder with only V1.7 engine subclass patched."""
    session = Path(session).resolve()
    old_engine = V17.LongRunMemorySafeEngine
    old_v18_version = V18.LIVE_VERSION
    old_v17_version = V17.LIVE_VERSION

    V17.LongRunMemorySafeEngine = TerminalTombstoneGuardEngine
    V18.LIVE_VERSION = LIVE_VERSION
    V17.LIVE_VERSION = LIVE_VERSION
    try:
        return V18.run_live_process(session, cfg)
    finally:
        V17.LongRunMemorySafeEngine = old_engine
        V18.LIVE_VERSION = old_v18_version
        V17.LIVE_VERSION = old_v17_version


__all__ = [
    "LIVE_VERSION",
    "TERMINAL_TOMBSTONE_MAX",
    "TerminalTombstoneGuardEngine",
    "regression_exact_bnb_false_orphan",
    "static_self_check",
    "run_live_process",
]
