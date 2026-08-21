from __future__ import annotations

"""V1.1 hardening for the isolated sports gap backfill.

Changes from V1 only:
- historical market discovery uses series_ticker alone, avoiding unnecessary
  filter combinations on the historical endpoint;
- a game is included only when Kalshi supplies a precise market
  occurrence_datetime or related milestone start_date. Event strike_date is not
  used as a kickoff proxy.

No live-strategy imports, account endpoints, control files, or orders.
"""

from pathlib import Path
from typing import Any

from . import sports_two_team_gap_backfill_v1 as V1

VERSION = "KALSHI_SPORTS_TWO_TEAM_GAP_BACKFILL_V1_1_PRECISE_START"


def fetch_markets_for_series(client: V1.PublicKalshi, series_ticker: str) -> list[dict[str, Any]]:
    current = client.paged(
        "/markets",
        "markets",
        {"series_ticker": series_ticker, "mve_filter": "exclude"},
        limit=1000,
    )
    historical = client.paged(
        "/historical/markets",
        "markets",
        {"series_ticker": series_ticker},
        limit=1000,
    )
    by_ticker: dict[str, dict[str, Any]] = {}
    for row in [*historical, *current]:
        ticker = str(row.get("ticker") or "")
        if ticker:
            by_ticker[ticker] = row
    return list(by_ticker.values())


def choose_game_start(event: dict[str, Any], markets: list[dict[str, Any]],
                      milestone_starts: dict[str, float]) -> tuple[float | None, str]:
    for market in markets:
        ts = V1._iso_to_ts(market.get("occurrence_datetime"))
        if ts is not None:
            return ts, "market_occurrence_datetime"
    event_ticker = str(event.get("event_ticker") or "")
    if event_ticker in milestone_starts:
        return float(milestone_starts[event_ticker]), "milestone_start_date"
    return None, "missing_precise_game_start"


def static_self_check(show: bool = True) -> dict[str, Any]:
    base = V1.static_self_check(show=False)
    out = dict(base)
    out.update({
        "version": VERSION,
        "historical_series_filter_only": True,
        "precise_game_start_required": True,
        "event_strike_date_kickoff_fallback": False,
        "orders_sent": False,
        "live_q50_modules_imported": False,
        "control_files_written": False,
        "ok": bool(base.get("ok")),
    })
    if show:
        print("=" * 116)
        print("SPORTS TWO-TEAM GAP BACKFILL V1.1 STATIC CHECK — PUBLIC READ ONLY")
        print("=" * 116)
        for k, v in out.items():
            print(f"{k:52s}: {v}")
    return out


def run(*, run_dir: str | Path, lookback_days: int = 365, max_events: int = 800,
        requests_per_second: float = 1.5, pregame_hours: float = 24.0,
        max_game_hours: float = 12.0, show: bool = True) -> dict[str, Any]:
    static_self_check(show=show)
    old_fetch = V1.fetch_markets_for_series
    old_start = V1.choose_game_start
    V1.fetch_markets_for_series = fetch_markets_for_series
    V1.choose_game_start = choose_game_start
    try:
        result = V1.run(
            run_dir=run_dir,
            lookback_days=lookback_days,
            max_events=max_events,
            requests_per_second=requests_per_second,
            pregame_hours=pregame_hours,
            max_game_hours=max_game_hours,
            show=show,
        )
        result["wrapper_version"] = VERSION
        return result
    finally:
        V1.fetch_markets_for_series = old_fetch
        V1.choose_game_start = old_start


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--lookback-days", type=int, default=365)
    ap.add_argument("--max-events", type=int, default=800)
    ap.add_argument("--requests-per-second", type=float, default=1.5)
    ap.add_argument("--pregame-hours", type=float, default=24.0)
    ap.add_argument("--max-game-hours", type=float, default=12.0)
    a = ap.parse_args()
    run(
        run_dir=a.run_dir,
        lookback_days=a.lookback_days,
        max_events=a.max_events,
        requests_per_second=a.requests_per_second,
        pregame_hours=a.pregame_hours,
        max_game_hours=a.max_game_hours,
        show=True,
    )


if __name__ == "__main__":
    _main()

__all__ = ["VERSION", "fetch_markets_for_series", "choose_game_start", "static_self_check", "run"]
