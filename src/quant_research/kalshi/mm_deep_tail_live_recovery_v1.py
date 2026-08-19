from __future__ import annotations

"""Recovery/reconciliation helper for failed deep-tail live sessions.

The only write action in this module is the explicitly armed order-group trigger,
which cancels all orders in that strategy group and prevents new group orders until
reset. It never creates an order and never attempts to infer or flatten a position.
After triggering, it polls authoritative resting orders and account positions so the
operator can verify the account state before another live run.
"""

import time
from pathlib import Path

from . import mm_cycle_q10_live_strategy_v1 as B

ARM = "TRIGGER_FAILED_DEEP_TAIL_GROUP"


def reconcile_failed_session(session_dir, *, arm_phrase=None, wait_s=4.0, show=True):
    if str(arm_phrase) != ARM:
        raise RuntimeError(f"Recovery write refused. Pass arm_phrase={ARM!r} exactly.")

    session = Path(session_dir).resolve()
    group = B._read(session / "order_group.json", {}) or {}
    gid = str(group.get("order_group_id") or "")
    if not gid:
        raise RuntimeError(f"Missing order_group_id in {session / 'order_group.json'}")

    client = B.Q1.LiveClient()
    before_resting, before_timing = B._resting(client)
    before_group = [
        r for r in before_resting
        if str(r.get("order_group_id") or "") == gid
    ]
    before_positions, pos_before_timing = B._positions(client)

    trigger = B._trigger_group(client, gid)
    if trigger.get("ok") is not True:
        raise RuntimeError(f"Order-group trigger failed: {trigger}")

    deadline = time.time() + max(0.5, float(wait_s))
    polls = []
    after_group = before_group
    after_positions = before_positions
    while time.time() < deadline:
        resting, rt = B._resting(client)
        after_group = [
            r for r in resting
            if str(r.get("order_group_id") or "") == gid
        ]
        after_positions, pt = B._positions(client)
        polls.append({
            "time": B._iso(),
            "group_resting_count": len(after_group),
            "group_order_ids": [str(r.get("order_id") or "") for r in after_group],
            "orders_timing": rt,
            "positions_timing": pt,
        })
        if not after_group:
            break
        time.sleep(0.20)

    nonzero_positions = [
        r for r in after_positions
        if abs(B._f(r.get("position_fp"), 0.0)) > B.EPS
    ]

    out = {
        "time": B._iso(),
        "session": str(session),
        "order_group_id": gid,
        "before_group_resting": before_group,
        "before_positions": before_positions,
        "trigger": trigger,
        "polls": polls,
        "after_group_resting": after_group,
        "after_positions": after_positions,
        "nonzero_positions": nonzero_positions,
        "resting_zero": not after_group,
        "flat": not nonzero_positions,
        "orders_created": False,
        "orders_canceled_via_group_trigger": True,
        "positions_modified": False,
    }

    B._atomic(session / "post_failure_group_reconciliation.json", out)

    if show:
        print("=" * 100)
        print("FAILED DEEP-TAIL SESSION RECONCILIATION")
        print("=" * 100)
        print("Session:                 ", session)
        print("Order group:             ", gid)
        print("Resting before trigger:  ", len(before_group))
        print("Group trigger:           ", "PASS" if trigger.get("ok") else "FAIL")
        print("Resting after trigger:   ", len(after_group))
        print("Nonzero positions:       ", len(nonzero_positions))
        print("RESTING ZERO:            ", not after_group)
        print("ACCOUNT FLAT:            ", not nonzero_positions)
        print("NEW ORDERS CREATED:      NO")
        if nonzero_positions:
            print("\nNONZERO POSITIONS — DO NOT START ANOTHER LIVE RUN:")
            for r in nonzero_positions:
                print(r)

    return out


__all__ = ["ARM", "reconcile_failed_session"]
