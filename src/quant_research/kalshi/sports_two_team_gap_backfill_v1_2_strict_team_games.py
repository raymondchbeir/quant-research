from __future__ import annotations

"""Strict team-vs-team universe for the Kalshi sports gap study.

V1.2 narrows the broad V1/V1.1 text filter to the phenomenon actually under
study: full-game/full-match winner markets where two teams compete directly.

Included sport families:
- football (NFL / college football),
- basketball (NBA / WNBA / college basketball),
- baseball,
- hockey,
- soccer.

Explicitly excluded:
- tennis/table tennis and esports,
- halves/quarters/periods/innings/sets/maps/rounds,
- spreads/totals/margins/exact-score markets,
- player/team-stat props (points, goals, TDs, FGs, yards, etc.),
- season/tournament/division/conference/championship/playoff winner markets,
- H2H stat props and other non-game-winner derivatives.

This is a public-read research wrapper only. It imports no live-Q50 modules,
uses no account/order endpoints, writes no live control files, and sends no orders.
"""

from pathlib import Path
from typing import Any

from . import sports_two_team_gap_backfill_v1 as V1
from . import sports_two_team_gap_backfill_v1_1 as V11

VERSION = "KALSHI_SPORTS_TWO_TEAM_GAP_BACKFILL_V1_2_STRICT_TEAM_GAMES"

TEAM_TAG_TERMS = (
    "football",
    "basketball",
    "baseball",
    "hockey",
    "soccer",
)

# Any of these terms means the series is not the full-game/full-match winner
# object we want. Matching is deliberately conservative.
STRICT_EXCLUDE_TERMS = (
    "spread",
    "total",
    "over/under",
    "over under",
    "margin",
    "exact score",
    "exact match score",
    "highest scoring",
    "lowest scoring",
    "points",
    "pts",
    "rebounds",
    "assists",
    "pra",
    "bench",
    "goals",
    "goal in",
    "field goal",
    "field goals",
    "touchdown",
    "touchdowns",
    "yards",
    "strikeout",
    "strikeouts",
    "home run",
    "home runs",
    "first 3 innings",
    "first 5 innings",
    "first 7 innings",
    "1st half",
    "2nd half",
    "first half",
    "second half",
    "quarter",
    "period winner",
    "period",
    "inning winner",
    "innings winner",
    "set winner",
    "game winner",  # tennis sub-game semantics; team full-game tickers are caught separately below
    "map winner",
    " map ",
    "round winner",
    "first to",
    "race to",
    "winning streak",
    "winning 10 games",
    "championship",
    "champion",
    "tournament winner",
    "series winner",
    "season wins",
    "division",
    "conference",
    "playoffs",
    "make the playoffs",
    "group winner",
    "group winners",
    "qualify",
    "award",
    "mvp",
    "derby",
    "head-to-head points",
    "head to head points",
    "head-to-head combined",
    "head to head combined",
    "head-to-head bench",
    "head to head bench",
)

# Non-team competitive categories that happened to pass the broad game/match filter.
STRICT_NON_TEAM_TERMS = (
    "tennis",
    "table tennis",
    "atp",
    "wta",
    "counter-strike",
    "counter strike",
    "cs2",
    "csgo",
    "valorant",
    "league of legends",
    "dota",
    "rocket league",
    "call of duty",
    "esports",
)


def _blob(row: dict[str, Any]) -> str:
    ticker = str(row.get("ticker") or "")
    title = str(row.get("title") or "")
    tags = [str(x) for x in (row.get("tags") or [])]
    return " ".join([ticker, title, *tags]).lower()


def _strict_sport(row: dict[str, Any]) -> str:
    """Specific-before-broad league/sport classifier."""
    ticker = str(row.get("ticker") or "").upper()
    title = str(row.get("title") or "").upper()
    tags = " ".join(str(x) for x in (row.get("tags") or [])).upper()
    text = f"{ticker} {title} {tags}"

    if "NCAAF" in ticker or "COLLEGE FOOTBALL" in text or " CFB" in text:
        return "NCAAF"
    if "NFL" in ticker or "PRO FOOTBALL" in text:
        return "NFL"
    if "WNBA" in ticker or "WOMEN'S PRO BASKETBALL" in text or "WOMENS PRO BASKETBALL" in text:
        return "WNBA"
    if "NCAAB" in ticker or "COLLEGE BASKETBALL" in text:
        return "NCAAB"
    if "NBA" in ticker or "PRO BASKETBALL" in text:
        return "NBA"
    if "NHL" in ticker or "HOCKEY" in tags:
        return "NHL"
    if any(x in ticker for x in ("MLB", "KBO", "LMB", "MILB")) or "BASEBALL" in tags:
        return "BASEBALL"
    if "SOCCER" in tags or any(x in text for x in (
        " MLS ", " EPL ", "UEFA", "FIFA", "LALIGA", "BUNDESLIGA",
        "SERIE A", "LIGUE 1", "FA CUP", "WORLD CUP", "CHAMPIONS LEAGUE",
    )):
        return "SOCCER"
    return "OTHER"


def strict_series_is_team_game(row: dict[str, Any]) -> bool:
    text = _blob(row)
    ticker = str(row.get("ticker") or "").upper()
    title = str(row.get("title") or "").strip().lower()
    tags = " ".join(str(x) for x in (row.get("tags") or [])).lower()

    # Must be a team-sport family.
    if not any(term in tags or term in text for term in TEAM_TAG_TERMS):
        return False
    if any(term in text for term in STRICT_NON_TEAM_TERMS):
        return False

    # Strong derivative/prop rejection.
    if any(term in text for term in STRICT_EXCLUDE_TERMS):
        # Allow ordinary team GAME tickers whose English title literally ends in
        # "Game"; the generic "game winner" exclusion above is aimed at tennis-like
        # sub-game markets, not KXNFLGAME/KXNBAGAME/etc.
        if "game winner" not in text:
            return False
        if not (ticker.endswith("GAME") and title.endswith("game")):
            return False

    # Full-event semantic anchor. We deliberately do not accept bare "Winner"
    # series because that admits division/season/tournament futures.
    full_event_anchor = (
        "game" in title
        or "match" in title
        or ticker.endswith("GAME")
        or ticker.endswith("MATCH")
    )
    if not full_event_anchor:
        return False

    # Final guard for numbered/sub-event language not covered above.
    if any(term in title for term in (
        "half", "quarter", "inning", "period", "set ", " set", "map ", " map",
        "round", "first to", "race to",
    )):
        return False

    return _strict_sport(row) != "OTHER"


def discover_series(client: V1.PublicKalshi) -> list[dict[str, Any]]:
    body = client.get("/series", {
        "category": "Sports",
        "include_product_metadata": "true",
        "include_volume": "true",
    })
    rows = [x for x in (body.get("series") or []) if isinstance(x, dict)]
    return [x for x in rows if strict_series_is_team_game(x)]


def static_self_check(show: bool = True) -> dict[str, Any]:
    base = V11.static_self_check(show=False)
    out = dict(base)
    out.update({
        "version": VERSION,
        "strict_full_game_team_only": True,
        "team_sports_only": True,
        "tennis_excluded": True,
        "esports_excluded": True,
        "half_period_inning_set_map_excluded": True,
        "props_spreads_totals_futures_excluded": True,
        "specific_sport_classifier": True,
        "orders_sent": False,
        "live_q50_modules_imported": False,
        "control_files_written": False,
        "ok": bool(base.get("ok")),
    })
    if show:
        print("=" * 120)
        print("SPORTS TWO-TEAM GAP BACKFILL V1.2 STRICT TEAM-GAME CHECK — PUBLIC READ ONLY")
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
    V1._infer_sport = lambda ticker, title, tags: _strict_sport({
        "ticker": ticker, "title": title, "tags": list(tags or [])
    })
    try:
        result = V11.run(
            run_dir=run_dir,
            lookback_days=lookback_days,
            max_events=max_events,
            requests_per_second=requests_per_second,
            pregame_hours=pregame_hours,
            max_game_hours=max_game_hours,
            show=show,
        )
        result["strict_wrapper_version"] = VERSION
        return result
    finally:
        V1.discover_series = old_discover
        V1._infer_sport = old_sport


__all__ = [
    "VERSION", "strict_series_is_team_game", "discover_series",
    "static_self_check", "run",
]
