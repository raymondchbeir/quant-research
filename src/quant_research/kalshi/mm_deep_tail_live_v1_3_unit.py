from __future__ import annotations

"""No-API regression tests for deep-tail live V1.3.

These tests create no LiveClient, open no WebSocket and send no orders.  They exercise
the exact two failure paths exposed by the second Q1 smoke:
- fixed passive EXIT must be GTC-compatible and therefore reduce_only=False;
- a definite CREATE failure must retire its local synthetic track before shutdown.
"""

from concurrent.futures import Future

from . import mm_deep_tail_join_ask_live_v1_3 as V13


def _test_passive_exit_reduce_only_false():
    e = object.__new__(V13.PassiveGTCCompatibleEngine)
    ticker = "TEST-TICKER"
    e.dt = {
        ticker: {
            "full_entry_ready": True,
            "exit_posted": False,
            "chosen_tail": "YES",
            "phase": "FULL_ENTRY_WAITING_OPPOSITE_CANCEL",
        }
    }
    e.active = {}
    e.shutdown_started = False
    e.q = 1.0
    captured = {}

    e._latest_fresh_bbo = lambda t: ({"bid": 0.55, "ask": 0.56}, {"ok": True})

    def fake_new_track(ticker_, role, tail, side, price, qty, reduce_only):
        captured.update({
            "ticker": ticker_, "role": role, "tail": tail, "side": side,
            "price": price, "qty": qty, "reduce_only": reduce_only,
        })
        return {"cid": "unit-cid"}

    e._new_track = fake_new_track
    e._lat = lambda *a, **k: None
    e._transition = lambda *a, **k: None

    V13.PassiveGTCCompatibleEngine._maybe_post_exit(e, ticker)

    assert captured["role"] == "EXIT"
    assert captured["side"] == "ask"
    assert captured["price"] == 0.56
    assert captured["qty"] == 1.0
    assert captured["reduce_only"] is False
    assert e.dt[ticker]["exit_posted"] is True


def _test_failed_create_retires_track_before_shutdown():
    e = object.__new__(V13.PassiveGTCCompatibleEngine)
    key = "TEST|EXIT|YES"
    cid = "unit-failed-cid"
    ticker = "TEST"

    f = Future()
    f.set_exception(RuntimeError("permanent 400 invalid order"))

    e.pending_creates = {key: f}
    e.pending_cancels = {}
    e.active = {
        key: {
            "key": key,
            "ticker": ticker,
            "role": "EXIT",
            "tail": "YES",
            "cid": cid,
            "order_id": None,
            "qty": 1.0,
        }
    }
    e.cid_to_key = {cid: key}
    e.order_id_to_key = {}
    e.last_error = None
    events = []
    shutdowns = []
    e.emit = lambda *a, **k: events.append((a, k))
    e.shutdown = lambda reason: shutdowns.append((reason, key in e.active))
    e._retry_unmatched_private = lambda: None

    V13.PassiveGTCCompatibleEngine._drain_create_futures(e)

    assert key not in e.active
    assert cid not in e.cid_to_key
    assert key not in e.pending_creates
    assert shutdowns == [("CREATE_TRANSPORT_FAIL_CLOSED", False)]
    assert any((k.get("reason") == "CREATE_FAIL_CLOSED") for _, k in events)


def run(show=True):
    _test_passive_exit_reduce_only_false()
    _test_failed_create_retires_track_before_shutdown()
    out = {
        "passive_exit_reduce_only_false": True,
        "failed_create_track_retired_before_shutdown": True,
        "api_called": False,
        "orders_sent": False,
        "ok": True,
    }
    if show:
        print("=" * 100)
        print("DEEP-TAIL V1.3 NO-API UNIT REGRESSION")
        print("=" * 100)
        for k, v in out.items():
            print(f"{k:48s}: {v}")
    return out


if __name__ == "__main__":
    run(show=True)
