from __future__ import annotations

"""No-API regression tests for deep-tail live V1.4.

No LiveClient, WebSocket, or exchange mutation is created. The tests cover the exact
Q5 failure mode that motivated V1.4:
- an exchange-visible resting row at the first M5 check is individually re-cancelled
  and leaves the ticker pending rather than shutting the whole run down;
- a later clean check finalizes the ticker;
- an M5 cancel-future failure is made retryable rather than global-fatal;
- a fill arriving at/after M5 cannot create a new passive JOIN_ASK.
"""

from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory

from . import mm_deep_tail_join_ask_live_v1_4 as V14


def _base_engine(tmp: Path, ticker="TEST-TICKER"):
    e = object.__new__(V14.PersistentM5CleanupEngine)
    e.m5_cleanup = {}
    e.finalized = set()
    e.shutdown_started = False
    e.dt = {ticker: {"phase": "WAIT", "chosen_tail": None}}
    e.active = {}
    e.pending_creates = {}
    e.pending_cancels = {}
    e.positions = {}
    e.q = 5.0
    e.gid = "test-group"
    e.client = object()
    e.risk_log = tmp / "risk.jsonl"
    e.counters_dt = {}
    e._transition = lambda *a, **k: None
    e.emit = lambda *a, **k: None
    e._lat = lambda *a, **k: None
    e._drain_create_futures = lambda: None
    e._drain_private = lambda: None
    return e


def _test_resting_at_m5_retries_then_finalizes():
    ticker = "TEST-TICKER"
    with TemporaryDirectory() as td:
        e = _base_engine(Path(td), ticker)
        e._cancel_all_for_ticker = lambda *a, **k: None
        e._drain_cancel_futures = lambda: None
        e.refresh_position = lambda t: 0.0

        row = {
            "ticker": ticker,
            "order_id": "oid-1",
            "initial_count_fp": "5.00",
            "remaining_count_fp": "5.00",
            "order_group_id": e.gid,
        }
        reads = [[row], [], []]

        def fresh(ticker=None):
            rows = reads.pop(0) if reads else []
            return rows, {"unit": True}

        e._fresh_group_resting_once = fresh

        original_cancel = V14.V11.safe_cancel_v2_resting_set
        cancels = []
        try:
            def fake_cancel(client, *, order_id, submitted_qty):
                cancels.append((order_id, submitted_qty))
                return {"ok": True, "source": "UNIT_CANCEL"}

            V14.V11.safe_cancel_v2_resting_set = fake_cancel

            e.finalize_m5(ticker)
            assert ticker not in e.finalized
            assert ticker in e.m5_cleanup
            assert cancels == [("oid-1", 5.0)]
            assert e.shutdown_started is False

            e.m5_cleanup[ticker]["next_retry_wall"] = 0.0
            e.finalize_m5(ticker)

            assert ticker in e.finalized
            assert ticker not in e.m5_cleanup
            assert e.dt[ticker]["phase"] == "M5_FINALIZED"
            assert e.shutdown_started is False
        finally:
            V14.V11.safe_cancel_v2_resting_set = original_cancel


def _test_m5_cancel_future_failure_becomes_retry():
    ticker = "TEST-TICKER"
    key = f"{ticker}|ENTRY|YES"
    with TemporaryDirectory() as td:
        e = _base_engine(Path(td), ticker)
        fut = Future()
        fut.set_result({"ok": False, "still_resting": True})
        e.active[key] = {
            "key": key,
            "ticker": ticker,
            "role": "ENTRY",
            "tail": "YES",
            "qty": 5.0,
            "processed_fill": 0.0,
            "cancel_requested": True,
            "cancel_reason": "M5_RETRY_1",
            "status": "resting",
            "cid": "cid-1",
            "order_id": "oid-1",
        }
        e.pending_cancels[key] = {
            "future": fut,
            "requested_ms": V14.V1._wall_ms(),
            "reason": "M5_RETRY_1",
        }
        e.cid_to_key = {"cid-1": key}
        e.order_id_to_key = {"oid-1": key}
        e.unmatched_private = []
        e.seen_private_fills = set()

        shutdowns = []
        e.shutdown = lambda reason: shutdowns.append(reason)

        V14.PersistentM5CleanupEngine._drain_cancel_futures(e)

        assert shutdowns == []
        assert key in e.active
        assert e.active[key]["cancel_requested"] is False
        assert ticker in e.m5_cleanup
        assert key not in e.pending_cancels


def _test_no_new_passive_exit_after_m5():
    ticker = "TEST-TICKER"
    with TemporaryDirectory() as td:
        e = _base_engine(Path(td), ticker)
        e.dt[ticker].update({
            "full_entry_ready": True,
            "exit_posted": False,
            "chosen_tail": "YES",
        })
        e.wall_elapsed = lambda t: 300.25
        created = []
        e._new_track = lambda *a, **k: created.append((a, k))

        V14.PersistentM5CleanupEngine._maybe_post_exit(e, ticker)

        assert created == []
        assert ticker in e.m5_cleanup
        assert e.dt[ticker]["phase"] == "M5_CLEANUP_PENDING"


def run(show=True):
    _test_resting_at_m5_retries_then_finalizes()
    _test_m5_cancel_future_failure_becomes_retry()
    _test_no_new_passive_exit_after_m5()
    out = {
        "m5_resting_row_retries_not_shutdown": True,
        "m5_later_clean_state_finalizes": True,
        "m5_cancel_failure_retryable": True,
        "no_passive_exit_created_after_m5": True,
        "api_called": False,
        "orders_sent": False,
        "ok": True,
    }
    if show:
        print("=" * 100)
        print("DEEP-TAIL V1.4 NO-API M5 RETRY REGRESSION")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    return out


if __name__ == "__main__":
    run(show=True)
