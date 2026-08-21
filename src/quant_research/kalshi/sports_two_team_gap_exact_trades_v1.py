from __future__ import annotations

"""Exact public-trade crossing refinement for the sports gap study.

Runs after sports_two_team_gap_analysis_v1. It only fetches individual public
non-block trades for markets whose minute quote-mid path already showed at least
one favorite flip. This is a sequencing refinement, not an execution backtest.

Safety: public GET only, no live strategy imports, no account/order endpoints,
no orders, and conservative throttling.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .sports_two_team_gap_backfill_v1 import PublicKalshi

VERSION = "KALSHI_SPORTS_TWO_TEAM_GAP_EXACT_TRADES_V1"


def _ts(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _price(row: dict[str, Any]) -> float | None:
    for k in ("yes_price_dollars", "yes_price"):
        try:
            z = float(row.get(k))
            if math.isfinite(z):
                return z
        except Exception:
            pass
    return None


def _fetch_all(client: PublicKalshi, path: str, ticker: str,
               min_ts: int, max_ts: int) -> list[dict[str, Any]]:
    try:
        return client.paged(
            path,
            "trades",
            {
                "ticker": ticker,
                "min_ts": int(min_ts),
                "max_ts": int(max_ts),
                "is_block_trade": "false",
            },
            limit=1000,
        )
    except Exception:
        return []


def fetch_trades(client: PublicKalshi, ticker: str,
                 min_ts: int, max_ts: int) -> list[dict[str, Any]]:
    rows = [
        *_fetch_all(client, "/historical/trades", ticker, min_ts, max_ts),
        *_fetch_all(client, "/markets/trades", ticker, min_ts, max_ts),
    ]
    by_id: dict[str, dict[str, Any]] = {}
    fallback = 0
    for row in rows:
        tid = str(row.get("trade_id") or "")
        if not tid:
            fallback += 1
            tid = f"fallback-{fallback}-{row.get('created_time')}-{row.get('yes_price_dollars')}-{row.get('count_fp')}"
        by_id[tid] = row
    return list(by_id.values())


def _states(prices: np.ndarray, deadband: float) -> np.ndarray:
    out = np.zeros(len(prices), dtype=np.int8)
    out[prices >= 0.5 + deadband] = 1
    out[prices <= 0.5 - deadband] = -1
    return out


def _cross_count(prices: np.ndarray, deadband: float) -> int:
    seq: list[int] = []
    for s in _states(prices, deadband):
        v = int(s)
        if v == 0:
            continue
        if not seq or seq[-1] != v:
            seq.append(v)
    return max(0, len(seq) - 1)


def static_self_check(show: bool = True) -> dict[str, Any]:
    out = {
        "version": VERSION,
        "public_trade_get_only": True,
        "non_block_trades_only": True,
        "only_broad_crossers_refined": True,
        "account_endpoints_used": False,
        "orders_sent": False,
        "live_q50_modules_imported": False,
        "ok": True,
    }
    if show:
        print("=" * 112)
        print("SPORTS EXACT TRADE CROSSING STATIC CHECK — PUBLIC READ ONLY")
        print("=" * 112)
        for k, v in out.items():
            print(f"{k:50s}: {v}")
    return out


def run(*, run_dir: str | Path, requests_per_second: float = 1.0,
        cross_deadband: float = 0.005, max_markets: int = 250,
        show: bool = True) -> dict[str, Any]:
    static_self_check(show=show)
    root = Path(run_dir).expanduser().resolve()
    phase = pd.read_csv(root / "game_phase_crossing_summary.csv")
    markets = pd.read_csv(root / "markets.csv")

    candidates = phase.loc[phase["cross_count"] > 0, "ticker"].dropna().astype(str).unique().tolist()
    candidates = candidates[: int(max_markets)] if max_markets else candidates
    market_map = markets.set_index("ticker").to_dict("index") if not markets.empty else {}
    client = PublicKalshi(requests_per_second=requests_per_second)

    out_rows: list[dict[str, Any]] = []
    for i, ticker in enumerate(candidates, 1):
        m = market_map.get(ticker) or {}
        start = float(m.get("game_start_ts"))
        req_start = int(float(m.get("request_start_ts", start - 24 * 3600)))
        req_end = int(float(m.get("request_end_ts", start + 12 * 3600)))
        rows = fetch_trades(client, ticker, req_start, req_end)
        parsed = []
        for row in rows:
            t = _ts(row.get("created_time"))
            p = _price(row)
            if t is None or p is None or not (0 <= p <= 1):
                continue
            parsed.append((t, p, float(row.get("count_fp", row.get("count", 0)) or 0)))
        parsed.sort(key=lambda x: x[0])

        for phase_name, lo, hi in (
            ("pregame", req_start, start),
            ("in_game", start, req_end),
        ):
            sub = [x for x in parsed if lo <= x[0] <= hi]
            prices = np.asarray([x[1] for x in sub], dtype=float)
            out_rows.append({
                "ticker": ticker,
                "event_ticker": m.get("event_ticker"),
                "series_ticker": m.get("series_ticker"),
                "sport": m.get("sport"),
                "phase": phase_name,
                "trade_count": len(sub),
                "trade_cross_count": _cross_count(prices, cross_deadband) if len(prices) else 0,
                "trade_touch_50_band": bool(np.any(np.abs(prices - 0.5) <= cross_deadband)) if len(prices) else False,
                "min_trade_price": float(np.min(prices)) if len(prices) else None,
                "max_trade_price": float(np.max(prices)) if len(prices) else None,
            })
        if show and (i == len(candidates) or i % 25 == 0):
            print(f"  exact trades {i}/{len(candidates)} | api_calls={client.calls} retries={client.retries}")

    out = pd.DataFrame(out_rows)
    out.to_csv(root / "exact_trade_crossing_summary.csv", index=False)
    headline = {
        "version": VERSION,
        "markets_refined": len(candidates),
        "api_calls": client.calls,
        "api_retries": client.retries,
        "requests_per_second": requests_per_second,
        "cross_deadband": cross_deadband,
        "orders_sent": False,
        "note": "Trade-price crossings refine sequencing but are not quote/execution crossings.",
    }
    (root / "exact_trade_headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    if show:
        print("\n" + "=" * 112)
        print("EXACT TRADE CROSSING REFINEMENT COMPLETE")
        print("=" * 112)
        for k, v in headline.items():
            print(f"{k:36s}: {v}")
        if not out.empty:
            print("\nTRADE CROSS COUNT DISTRIBUTION")
            print(out.groupby(["phase", "trade_cross_count"]).size().rename("markets").reset_index().to_string(index=False))
    return headline


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--requests-per-second", type=float, default=1.0)
    ap.add_argument("--cross-deadband", type=float, default=0.005)
    ap.add_argument("--max-markets", type=int, default=250)
    a = ap.parse_args()
    run(
        run_dir=a.run_dir,
        requests_per_second=a.requests_per_second,
        cross_deadband=a.cross_deadband,
        max_markets=a.max_markets,
        show=True,
    )


if __name__ == "__main__":
    _main()

__all__ = ["VERSION", "fetch_trades", "static_self_check", "run"]
