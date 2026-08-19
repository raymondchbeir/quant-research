from __future__ import annotations

"""V9.1 hardening wrapper for cross-feed causality forensic.

V3 established that the completed Q5 live ENTRY fill trade_ids overlap the public
trade tape exactly, while the older V1 semantic matcher had an unresolved zero-
match paradox. V9 initially reused V1._trade_compatible while selecting exact-ID
candidates. That can unnecessarily reintroduce the old matcher ambiguity.

This wrapper changes only causal-trade identification:
- exact trade_id + same ticker is primary identity;
- when duplicate same-ticker IDs exist, choose the event nearest the live fill
  exchange timestamp;
- side/price compatibility is recorded as a diagnostic, never used to discard an
  exact-ID candidate;
- V1 fallback is retained only when no exact-ID same-ticker event exists.

All receipt-order shadow replay and causality classification remain V9 unchanged.
READ ONLY. NO API. NO ORDERS. Source session is not modified.
"""

from collections import Counter

import numpy as np

from . import mm_cycle_q5_cross_feed_causality_forensic_v9 as V9
from . import mm_cycle_q5_public_trade_fill_reconciliation_v1 as V1

VERSION = "MM_CYCLE_Q5_CROSS_FEED_CAUSALITY_FORENSIC_V9_1"
OUTPUT_ROOT = V9.OUTPUT_ROOT


def _exact_causal_trade_map_hardened(source, selected_tickers, first_fills):
    trades, by_trade_id, by_ticker = V1._load_public_trades(
        source / "raw_capture", selected_tickers
    )
    out = {}
    method_counts = Counter()

    for oid, f in first_fills.items():
        ticker = str(f.get("ticker") or "")
        tid = str(f.get("trade_id") or "")
        fill_s = float(f["fill_s"])
        side = str(f.get("side") or "").upper()
        price = V9._f(f.get("price"))

        exact = []
        if tid:
            for i in by_trade_id.get(tid, []):
                tr = trades[i]
                if str(tr.get("ticker") or "") != ticker:
                    continue
                tx = V9._f(tr.get("exchange_s"))
                distance = (
                    abs(tx - fill_s)
                    if np.isfinite(tx)
                    else abs(float(tr["receipt_s"]) - fill_s)
                )
                compatible, kind = V1._trade_compatible(side, price, tr)
                exact.append((distance, i, bool(compatible), kind))

        if exact:
            exact.sort(key=lambda z: (z[0], z[1]))
            _, i, compatible, kind = exact[0]
            tr = trades[i]
            method = "EXACT_TRADE_ID_SAME_TICKER"
            method_counts[method] += 1
            out[oid] = {
                "causal_match_method": method,
                "causal_trade_candidate_count": len(exact),
                "causal_trade_id": str(tr.get("trade_id") or ""),
                "causal_trade_index": int(i),
                "causal_trade_semantic_compatible": compatible,
                "causal_trade_kind": kind,
                "causal_trade_exchange_s": V9._f(tr.get("exchange_s")),
                "causal_trade_receipt_s": float(tr["receipt_s"]),
                "causal_trade_price": V9._f(tr.get("price")),
                "causal_trade_qty": V9._f(tr.get("qty")),
                "causal_trade_taker_book_side": str(tr.get("taker_book_side") or ""),
            }
            continue

        # Defensive fallback only if the exact ID cannot be found on the same
        # ticker. This should be rare/zero for the known Q5 realization.
        ff = {
            "trade_id": tid,
            "ticker": ticker,
            "side": side,
            "price": price,
            "fill_time_s": fill_s,
        }
        m = V1._match_fill(ff, trades, by_trade_id, by_ticker)
        i = m.get("trade_index")
        if i is not None:
            tr = trades[i]
            method = "FALLBACK_V1_MATCH"
            method_counts[method] += 1
            compatible, kind = V1._trade_compatible(side, price, tr)
            out[oid] = {
                "causal_match_method": method,
                "causal_trade_candidate_count": int(m.get("candidate_count") or 0),
                "causal_trade_id": str(tr.get("trade_id") or ""),
                "causal_trade_index": int(i),
                "causal_trade_semantic_compatible": bool(compatible),
                "causal_trade_kind": kind or m.get("match_kind"),
                "causal_trade_exchange_s": V9._f(tr.get("exchange_s")),
                "causal_trade_receipt_s": float(tr["receipt_s"]),
                "causal_trade_price": V9._f(tr.get("price")),
                "causal_trade_qty": V9._f(tr.get("qty")),
                "causal_trade_taker_book_side": str(tr.get("taker_book_side") or ""),
            }
        else:
            method_counts["NO_MATCH"] += 1

    return out, dict(method_counts)


def run_q5_cross_feed_causality_forensic(source_session, *, show=True):
    original = V9._exact_causal_trade_map
    V9._exact_causal_trade_map = _exact_causal_trade_map_hardened
    try:
        result = V9.run_q5_cross_feed_causality_forensic(source_session, show=show)
        result["summary"]["wrapper_version"] = VERSION
        return result
    finally:
        V9._exact_causal_trade_map = original


__all__ = ["run_q5_cross_feed_causality_forensic", "VERSION", "OUTPUT_ROOT"]
