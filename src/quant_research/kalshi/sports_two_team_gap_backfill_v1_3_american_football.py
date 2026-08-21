from __future__ import annotations

"""American-football-only universe for the Kalshi sports gap study.

V1.3 narrows V1.2 to full-game team-vs-team winner markets in American football:
- NFL / pro football
- NCAA / college football

It preserves V1.2 exclusions for halves/quarters, spreads/totals, props,
futures, and other non-full-game winner derivatives.

Public-read research only: no account/order endpoints, no orders, no live-Q50
imports, and no live control-file writes.
"""

from pathlib import Path
from typing import Any

from . import sports_two_team_gap_backfill_v1 as V1
from . import sports_two_team_gap_backfill_v1_2_strict_team_games as V12

VERSION = "KALSHI_SPORTS_TWO_TEAM_GAP_BACKFILL_V1_3_AMERICAN_FOOTBALL"


def _football_sport(row: dict[str, Any]) -> str:
    ticker = str(row.get("ticker") or "").upper()
    title = str(row.get("title") or "").upper()
    tags = " ".join(str(x) for x in (row.get("tags") or [])).upper()
    text = f"{ticker} {title} {tags}"

    if "NCAAF" in ticker or "COLLEGE FOOTBALL" in text or " CFB" in text:
        return "NCAAF"
    if "NFL" in ticker or "PRO FOOTBALL" in text:
        return "NFL"
    return "OTHER"


def strict_series_is_american_football(row: dict[str, Any]) -> bool:
    if not V12.strict_series_is_team_game(row):
        return False
    return _football_sport(row) in {"NFL", "NCAAF"}


def discover_series(client: V1.PublicKalshi) -> list[dict[str, Any]]:
    body = client.get("/series", {
        "category": "Sports",
        "include_product_metadata": "true",
        "include_volume": "true",
    })
    rows = [x for x in (body.get("series") or []) if isinstance(x, dict)]
    return [x for x in rows if strict_series_is_american_football(x)]


def static_self_check(show: bool = True) -> dict[str, Any]:
    base = V12.static_self_check(show=False)
    out = dict(base)
    out.update({
        "version": VERSION,
        "american_football_only": True,
        "nfl_included": True,
        "college_football_included": True,
        "other_team_sports_excluded": True,
        "full_game_winner_only": True,
        "orders_sent": False,
        "live_q50_modules_imported": False,
        "control_files_written": False,
        "ok": bool(base.get("ok")),
    })
    if show:
        print("=" * 120)
        print("SPORTS GAP BACKFILL V1.3 AMERICAN FOOTBALL — PUBLIC READ ONLY")
        print("=" * 120)
        for k, v in out.items():
            print(f"{k:58s}: {v}")
    return out


def run(*, run_dir: str | Path, lookback_days: int = 365, max_events: int = 800,
        requests_per_second: float = 1.5, pregame_hours: float = 24.0,
        max_game_hours: float = 12.0, show: bool = True) -> dict[str, Any]:
    static_self_check(show=show)
    old_discover = V1.discover_series
    old_sport = V1._infer_sport
    V1.discover_series = discover_series
    V1._infer_sport = lambda ticker, title, tags: _football_sport({
        "ticker": ticker, "title": title, "tags": list(tags or [])
    })
    try:
        result = V12.run(
            run_dir=run_dir,
            lookback_days=lookback_days,
            max_events=max_events,
            requests_per_second=requests_per_second,
            pregame_hours=pregame_hours,
            max_game_hours=max_game_hours,
            show=show,
        )
        result["football_wrapper_version"] = VERSION
        return result
    finally:
        V1.discover_series = old_discover
        V1._infer_sport = old_sport


__all__ = [
    "VERSION",
    "strict_series_is_american_football",
    "discover_series",
    "static_self_check",
    "run",
]
