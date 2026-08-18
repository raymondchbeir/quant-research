from __future__ import annotations

"""V2 wrapper for Q5 public-trade/live-fill reconciliation.

Fixes a V1 schema-normalization bug: Kalshi live fill rows may expose yes_price
in integer cents (for example 42) while the public recorder stores yes_price in
dollars (0.42). V1 accepted the first numeric fill-price field without converting
cent-valued prices, which can force every fallback match to fail even when the
underlying public trade is present.

V2 preserves V1's reconciliation logic and scientific guardrails, but patches
fill-price normalization to the same cents->dollars convention already used by
mm_cycle_q5_live_shadow_fill_forensics_v1.

READ ONLY / NO API / NO ORDERS.
"""

from pathlib import Path
import json
import numpy as np

from . import mm_cycle_q5_public_trade_fill_reconciliation_v1 as V1

VERSION = "MM_CYCLE_Q5_PUBLIC_TRADE_FILL_RECONCILIATION_V2"


def _fill_price_normalized(row):
    for k in ("yes_price_dollars", "yes_price", "price_dollars", "price"):
        z = V1._f(row.get(k))
        if not np.isfinite(z):
            continue
        z = float(z)
        if 1.0 < z <= 100.0:
            z /= 100.0
        if 0.0 <= z <= 1.0:
            return z
    return np.nan


def _schema_precheck(source_session, max_rows=5000):
    source = Path(source_session).resolve()
    fill_path = source / "fills.jsonl"
    trade_path = source / "raw_capture" / "trades_event_time.jsonl"

    raw_fill_prices = []
    normalized_fill_prices = []
    public_prices = []
    fill_trade_ids = 0
    public_trade_ids = 0

    if fill_path.exists():
        with fill_path.open("r", encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if n >= max_rows:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if str(r.get("role") or "").upper() != "ENTRY":
                    continue
                for k in ("yes_price_dollars", "yes_price", "price_dollars", "price"):
                    z = V1._f(r.get(k))
                    if np.isfinite(z):
                        raw_fill_prices.append(float(z))
                        break
                p = _fill_price_normalized(r)
                if np.isfinite(p):
                    normalized_fill_prices.append(float(p))
                if str(r.get("trade_id") or ""):
                    fill_trade_ids += 1

    if trade_path.exists():
        with trade_path.open("r", encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if n >= max_rows:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                z = V1._f(r.get("yes_price"))
                if np.isfinite(z):
                    public_prices.append(float(z))
                if str(r.get("trade_id") or ""):
                    public_trade_ids += 1

    def rng(x):
        if not x:
            return {"n": 0, "min": np.nan, "median": np.nan, "max": np.nan}
        a = np.asarray(x, dtype=float)
        return {
            "n": int(a.size),
            "min": float(np.min(a)),
            "median": float(np.median(a)),
            "max": float(np.max(a)),
        }

    return {
        "raw_live_fill_price_range": rng(raw_fill_prices),
        "normalized_live_fill_price_range": rng(normalized_fill_prices),
        "public_trade_price_range": rng(public_prices),
        "sample_live_fill_rows_with_trade_id": int(fill_trade_ids),
        "sample_public_trade_rows_with_trade_id": int(public_trade_ids),
    }


def run_q5_public_trade_fill_reconciliation(source_session, *, show=True):
    pre = _schema_precheck(source_session)
    if show:
        print("=" * 120)
        print("V2 SCHEMA NORMALIZATION PRECHECK")
        print("=" * 120)
        print("raw live fill price range:       ", pre["raw_live_fill_price_range"])
        print("normalized live fill price range:", pre["normalized_live_fill_price_range"])
        print("public trade price range:        ", pre["public_trade_price_range"])
        print("sample live fill trade_ids:      ", pre["sample_live_fill_rows_with_trade_id"])
        print("sample public trade_ids:         ", pre["sample_public_trade_rows_with_trade_id"])
        print("=" * 120)

    old = V1._fill_price
    old_version = V1.VERSION
    try:
        V1._fill_price = _fill_price_normalized
        V1.VERSION = VERSION
        out = V1.run_q5_public_trade_fill_reconciliation(source_session, show=show)
    finally:
        V1._fill_price = old
        V1.VERSION = old_version

    if isinstance(out, dict):
        out["schema_normalization_precheck"] = pre
    return out


__all__ = ["VERSION", "run_q5_public_trade_fill_reconciliation"]
