from __future__ import annotations

"""Public-read Kalshi sports head-to-head backfill.

Build a minute-resolution sample for two-team/two-outcome sports winner markets,
covering T-24h through the game window.

Safety / isolation:
- PUBLIC market-data GETs only.
- No portfolio/account/order endpoints.
- No orders.
- No control-file writes.
- No imports from the live Q50 engine.
- Conservative request throttling so it can run beside a live strategy process.
"""

import argparse
import csv
import gzip
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

VERSION = "KALSHI_SPORTS_TWO_TEAM_GAP_BACKFILL_V1"
BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

POSITIVE_SERIES_TERMS = (
    " game", "game ", "match", "winner", "moneyline", "money line",
    "head-to-head", "head to head", " h2h", "versus", " vs ",
)
EXCLUDE_SERIES_TERMS = (
    "spread", "total", "over/under", "over under", "points scored",
    "goals scored", "exact score", "margin", "player", "team total",
    "first half", "1st half", "quarter", "period", "championship",
    "champion", "tournament winner", "season wins", "division", "conference",
    "playoffs", "make the playoffs", "award", "mvp", "qualify", "series winner",
)
SPORT_HINTS = {
    "NFL": ("NFL", "FOOTBALL"),
    "NCAAF": ("NCAAF", "COLLEGE FOOTBALL", "CFB"),
    "NBA": ("NBA", "BASKETBALL"),
    "WNBA": ("WNBA",),
    "NCAAB": ("NCAAB", "COLLEGE BASKETBALL"),
    "MLB": ("MLB", "BASEBALL"),
    "NHL": ("NHL", "HOCKEY"),
    "SOCCER": ("SOCCER", "MLS", "EPL", "UEFA", "FIFA", "LALIGA", "BUNDESLIGA"),
    "TENNIS": ("TENNIS", "ATP", "WTA"),
}

PATH_FIELDS = [
    "event_ticker", "series_ticker", "series_title", "sport", "ticker",
    "game_start_ts", "game_start_iso", "end_period_ts", "wall_iso",
    "elapsed_from_start_s", "phase", "yes_bid", "yes_ask", "yes_mid",
    "quote_spread", "trade_open", "trade_high", "trade_low", "trade_close",
    "trade_mean", "trade_previous", "volume", "open_interest",
]


def _iso_to_ts(x: Any) -> float | None:
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    try:
        return datetime.fromisoformat(str(x).strip().replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _ts_to_iso(ts: float | int | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _f(x: Any) -> float | None:
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _node_float(node: Any, key: str) -> float | None:
    if not isinstance(node, dict):
        return None
    for k in (f"{key}_dollars", key):
        if k in node:
            z = _f(node.get(k))
            if z is not None:
                return z
    return None


def _infer_sport(series_ticker: str, title: str, tags: Iterable[str]) -> str:
    text = " ".join([series_ticker or "", title or "", *(tags or [])]).upper()
    for sport, hints in SPORT_HINTS.items():
        if any(h in text for h in hints):
            return sport
    return "OTHER"


def _series_is_head_to_head(row: dict[str, Any]) -> bool:
    ticker = str(row.get("ticker") or "")
    title = str(row.get("title") or "")
    tags = [str(x) for x in (row.get("tags") or [])]
    text = " ".join([ticker, title, *tags]).lower()
    if any(term in text for term in EXCLUDE_SERIES_TERMS):
        return False
    if "GAME" in ticker.upper():
        return True
    return any(term in f" {text} " for term in POSITIVE_SERIES_TERMS)


def _market_volume(row: dict[str, Any]) -> float:
    return _f(row.get("volume_fp", row.get("volume"))) or 0.0


def _settlement_value(row: dict[str, Any]) -> float | None:
    return _f(row.get("settlement_value_dollars", row.get("settlement_value")))


def _event_group_is_two_outcome(markets: list[dict[str, Any]]) -> bool:
    if len(markets) == 1:
        return True
    if len(markets) != 2:
        return False
    vals = [_settlement_value(x) for x in markets]
    if all(v is not None for v in vals):
        return abs(sum(vals) - 1.0) <= 0.02
    return True


@dataclass
class PublicKalshi:
    requests_per_second: float = 1.5
    timeout_s: float = 30.0
    max_retries: int = 5

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self._last_request_wall = 0.0
        self.calls = 0
        self.retries = 0

    def _throttle(self) -> None:
        min_dt = 1.0 / max(0.05, float(self.requests_per_second))
        wait = min_dt - (time.time() - self._last_request_wall)
        if wait > 0:
            time.sleep(wait)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = BASE_URL + path
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self.session.get(url, params=params or {}, timeout=self.timeout_s)
                self._last_request_wall = time.time()
                self.calls += 1
                if r.status_code in {429, 500, 502, 503, 504}:
                    self.retries += 1
                    time.sleep(min(8.0, 0.5 * (2 ** attempt)))
                    continue
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Non-object JSON from {path}: {type(data)!r}")
                return data
            except Exception as exc:
                last_exc = exc
                self._last_request_wall = time.time()
                if attempt + 1 >= self.max_retries:
                    break
                self.retries += 1
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        raise RuntimeError(f"GET failed path={path} params={params}: {last_exc!r}")

    def paged(self, path: str, key: str, params: dict[str, Any] | None = None,
              limit: int = 1000) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor = ""
        page = 0
        while True:
            page += 1
            q = dict(params or {})
            q["limit"] = limit
            if cursor:
                q["cursor"] = cursor
            body = self.get(path, q)
            rows = body.get(key) or []
            if not isinstance(rows, list):
                raise RuntimeError(f"{path} returned non-list {key}")
            out.extend(x for x in rows if isinstance(x, dict))
            cursor = str(body.get("cursor") or "")
            if not cursor:
                break
            if page > 10000:
                raise RuntimeError(f"Pagination runaway on {path}")
        return out


def discover_series(client: PublicKalshi) -> list[dict[str, Any]]:
    body = client.get("/series", {
        "category": "Sports",
        "include_product_metadata": "true",
        "include_volume": "true",
    })
    rows = [x for x in (body.get("series") or []) if isinstance(x, dict)]
    return [x for x in rows if _series_is_head_to_head(x)]


def fetch_events_for_series(client: PublicKalshi, series_ticker: str,
                            min_close_ts: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    events: list[dict[str, Any]] = []
    milestone_starts: dict[str, float] = {}
    cursor = ""
    page = 0
    while True:
        page += 1
        q: dict[str, Any] = {
            "series_ticker": series_ticker,
            "with_milestones": "true",
            "min_close_ts": int(min_close_ts),
            "limit": 200,
        }
        if cursor:
            q["cursor"] = cursor
        body = client.get("/events", q)
        events.extend(x for x in (body.get("events") or []) if isinstance(x, dict))
        for m in body.get("milestones") or []:
            if not isinstance(m, dict):
                continue
            start = _iso_to_ts(m.get("start_date"))
            if start is None:
                continue
            event_ids = list(m.get("related_event_tickers") or []) + list(m.get("primary_event_tickers") or [])
            for et in event_ids:
                if et:
                    milestone_starts[str(et)] = float(start)
        cursor = str(body.get("cursor") or "")
        if not cursor:
            break
        if page > 10000:
            raise RuntimeError(f"Event pagination runaway for {series_ticker}")
    return events, milestone_starts


def fetch_markets_for_series(client: PublicKalshi, series_ticker: str) -> list[dict[str, Any]]:
    current = client.paged(
        "/markets", "markets",
        {"series_ticker": series_ticker, "mve_filter": "exclude"},
        limit=1000,
    )
    historical = client.paged(
        "/historical/markets", "markets",
        {"series_ticker": series_ticker, "mve_filter": "exclude"},
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
    for m in markets:
        t = _iso_to_ts(m.get("occurrence_datetime"))
        if t is not None:
            return t, "market_occurrence_datetime"
    et = str(event.get("event_ticker") or "")
    if et in milestone_starts:
        return milestone_starts[et], "milestone_start_date"
    t = _iso_to_ts(event.get("strike_date"))
    if t is not None:
        return t, "event_strike_date"
    return None, "missing"


def choose_representative_market(markets: list[dict[str, Any]]) -> dict[str, Any]:
    return max(markets, key=lambda x: (_market_volume(x), str(x.get("ticker") or "")))


def candle_endpoint_and_rows(client: PublicKalshi, market: dict[str, Any], series_ticker: str,
                             start_ts: int, end_ts: int, historical_cutoff_ts: float | None
                             ) -> tuple[str, list[dict[str, Any]]]:
    ticker = str(market.get("ticker") or "")
    settle = _iso_to_ts(market.get("settlement_ts"))
    use_historical = historical_cutoff_ts is not None and settle is not None and settle < historical_cutoff_ts
    q = {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": 1}
    paths = (
        [f"/historical/markets/{ticker}/candlesticks",
         f"/series/{series_ticker}/markets/{ticker}/candlesticks"]
        if use_historical else
        [f"/series/{series_ticker}/markets/{ticker}/candlesticks",
         f"/historical/markets/{ticker}/candlesticks"]
    )
    errors = []
    for path in paths:
        try:
            body = client.get(path, q)
            rows = [x for x in (body.get("candlesticks") or []) if isinstance(x, dict)]
            if rows:
                return path, rows
            errors.append(f"{path}: empty")
        except Exception as exc:
            errors.append(f"{path}: {exc!r}")
    raise RuntimeError("; ".join(errors))


def normalize_candle(row: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any] | None:
    ts = _f(row.get("end_period_ts"))
    if ts is None:
        return None
    bid = _node_float(row.get("yes_bid"), "close")
    ask = _node_float(row.get("yes_ask"), "close")
    mid = None
    spread = None
    if bid is not None and ask is not None and 0 <= bid <= ask <= 1:
        mid = (bid + ask) / 2.0
        spread = ask - bid
    price = row.get("price") or {}
    game_start = float(meta["game_start_ts"])
    elapsed = ts - game_start
    return {
        **meta,
        "end_period_ts": int(ts),
        "wall_iso": _ts_to_iso(ts),
        "elapsed_from_start_s": float(elapsed),
        "phase": "pregame" if elapsed < 0 else "in_game",
        "yes_bid": bid,
        "yes_ask": ask,
        "yes_mid": mid,
        "quote_spread": spread,
        "trade_open": _node_float(price, "open"),
        "trade_high": _node_float(price, "high"),
        "trade_low": _node_float(price, "low"),
        "trade_close": _node_float(price, "close"),
        "trade_mean": _node_float(price, "mean"),
        "trade_previous": _node_float(price, "previous"),
        "volume": _f(row.get("volume_fp", row.get("volume"))),
        "open_interest": _f(row.get("open_interest_fp", row.get("open_interest"))),
    }


def static_self_check(show: bool = True) -> dict[str, Any]:
    out = {
        "version": VERSION,
        "public_get_only": True,
        "portfolio_endpoints_used": False,
        "order_endpoints_used": False,
        "orders_sent": False,
        "live_q50_modules_imported": False,
        "control_files_written": False,
        "minute_candles": True,
        "pregame_hours": 24,
        "max_game_hours_default": 12,
        "dedupe_one_market_per_event": True,
        "ok": True,
    }
    if show:
        print("=" * 112)
        print("SPORTS TWO-TEAM GAP BACKFILL STATIC CHECK — PUBLIC READ ONLY")
        print("=" * 112)
        for k, v in out.items():
            print(f"{k:46s}: {v}")
    return out


def run(*, run_dir: str | Path, lookback_days: int = 365, max_events: int = 800,
        requests_per_second: float = 1.5, pregame_hours: float = 24.0,
        max_game_hours: float = 12.0, show: bool = True) -> dict[str, Any]:
    static_self_check(show=show)
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)

    client = PublicKalshi(requests_per_second=requests_per_second)
    now = time.time()
    min_start = now - float(lookback_days) * 86400.0
    min_close_ts = int(min_start - 86400.0)

    cutoff: dict[str, Any] = {}
    cutoff_market_ts = None
    try:
        cutoff = client.get("/historical/cutoff")
        cutoff_market_ts = _iso_to_ts(cutoff.get("market_settled_ts"))
    except Exception:
        pass

    series = discover_series(client)
    series_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if show:
        print(f"\nHead-to-head sports series discovered: {len(series)}")

    for s_idx, s in enumerate(series, 1):
        st = str(s.get("ticker") or "")
        title = str(s.get("title") or "")
        tags = [str(x) for x in (s.get("tags") or [])]
        sport = _infer_sport(st, title, tags)
        series_rows.append({
            "series_ticker": st,
            "series_title": title,
            "sport": sport,
            "tags": "|".join(tags),
            "series_volume": _market_volume(s),
        })
        try:
            events, milestone_starts = fetch_events_for_series(client, st, min_close_ts)
            event_by_id = {str(e.get("event_ticker") or ""): e for e in events}
            markets = fetch_markets_for_series(client, st)
            grouped: dict[str, list[dict[str, Any]]] = {}
            for m in markets:
                et = str(m.get("event_ticker") or "")
                if et in event_by_id:
                    grouped.setdefault(et, []).append(m)

            for et, group in grouped.items():
                if not _event_group_is_two_outcome(group):
                    continue
                event = event_by_id[et]
                start, start_source = choose_game_start(event, group, milestone_starts)
                if start is None or start < min_start or start > now + 86400:
                    continue
                chosen = choose_representative_market(group)
                candidates.append({
                    "series_ticker": st,
                    "series_title": title,
                    "sport": sport,
                    "event": event,
                    "event_ticker": et,
                    "event_market_count": len(group),
                    "market": chosen,
                    "game_start_ts": start,
                    "game_start_source": start_source,
                })
        except Exception as exc:
            errors.append({"stage": "discover_series", "series_ticker": st, "error": repr(exc)})
        if show and (s_idx == len(series) or s_idx % 10 == 0):
            print(f"  series {s_idx}/{len(series)} | candidates={len(candidates)} | api_calls={client.calls}")

    candidates.sort(key=lambda x: (x["game_start_ts"], x["event_ticker"]), reverse=True)
    if max_events and len(candidates) > int(max_events):
        candidates = candidates[:int(max_events)]

    pd.DataFrame(series_rows).to_csv(root / "series.csv", index=False)

    market_rows: list[dict[str, Any]] = []
    paths_path = root / "minute_paths.csv.gz"
    with gzip.open(paths_path, "wt", encoding="utf-8", newline="") as gz:
        writer = csv.DictWriter(gz, fieldnames=PATH_FIELDS)
        writer.writeheader()

        for i, item in enumerate(candidates, 1):
            m = item["market"]
            ticker = str(m.get("ticker") or "")
            start = float(item["game_start_ts"])
            close_ts = _iso_to_ts(m.get("close_time"))
            settle_ts = _iso_to_ts(m.get("settlement_ts"))
            natural_end = close_ts or settle_ts or (start + max_game_hours * 3600.0)
            end_ts = min(max(natural_end, start + 30 * 60.0), start + max_game_hours * 3600.0)
            req_start = int(start - pregame_hours * 3600.0)
            req_end = int(end_ts)

            row_meta = {
                "event_ticker": item["event_ticker"],
                "series_ticker": item["series_ticker"],
                "series_title": item["series_title"],
                "sport": item["sport"],
                "ticker": ticker,
                "game_start_ts": start,
                "game_start_iso": _ts_to_iso(start),
            }
            market_out = {
                **row_meta,
                "game_start_source": item["game_start_source"],
                "event_market_count": item["event_market_count"],
                "event_title": str(item["event"].get("title") or ""),
                "event_sub_title": str(item["event"].get("sub_title") or ""),
                "market_title": str(m.get("title") or ""),
                "market_subtitle": str(m.get("subtitle") or ""),
                "yes_sub_title": str(m.get("yes_sub_title") or ""),
                "no_sub_title": str(m.get("no_sub_title") or ""),
                "market_volume": _market_volume(m),
                "settlement_value": _settlement_value(m),
                "request_start_ts": req_start,
                "request_end_ts": req_end,
                "close_time": m.get("close_time"),
                "settlement_ts": m.get("settlement_ts"),
                "candle_endpoint": None,
                "candle_rows": 0,
                "valid_quote_rows": 0,
                "status": "pending",
            }
            try:
                endpoint, candles = candle_endpoint_and_rows(
                    client, m, item["series_ticker"], req_start, req_end, cutoff_market_ts
                )
                valid = 0
                kept = 0
                for c in candles:
                    n = normalize_candle(c, row_meta)
                    if n is None:
                        continue
                    if n["end_period_ts"] < req_start or n["end_period_ts"] > req_end:
                        continue
                    writer.writerow({k: n.get(k) for k in PATH_FIELDS})
                    kept += 1
                    if n["yes_mid"] is not None:
                        valid += 1
                market_out.update({
                    "candle_endpoint": endpoint,
                    "candle_rows": kept,
                    "valid_quote_rows": valid,
                    "status": "ok" if valid else "no_valid_quotes",
                })
            except Exception as exc:
                market_out["status"] = "error"
                market_out["error"] = repr(exc)
                errors.append({
                    "stage": "candles", "ticker": ticker,
                    "event_ticker": item["event_ticker"], "error": repr(exc),
                })
            market_rows.append(market_out)
            if show and (i == len(candidates) or i % 25 == 0):
                ok_n = sum(x.get("status") == "ok" for x in market_rows)
                print(f"  markets {i}/{len(candidates)} | ok={ok_n} | api_calls={client.calls} retries={client.retries}")

    markets_df = pd.DataFrame(market_rows)
    markets_df.to_csv(root / "markets.csv", index=False)
    if errors:
        with (root / "errors.jsonl").open("w", encoding="utf-8") as fh:
            for row in errors:
                fh.write(json.dumps(row, default=str) + "\n")

    manifest = {
        "version": VERSION,
        "created_at": _ts_to_iso(time.time()),
        "run_dir": str(root),
        "lookback_days": int(lookback_days),
        "pregame_hours": float(pregame_hours),
        "max_game_hours": float(max_game_hours),
        "max_events": int(max_events),
        "requests_per_second": float(requests_per_second),
        "sports_series_discovered": len(series),
        "candidate_events": len(candidates),
        "markets_ok": int((markets_df.get("status") == "ok").sum()) if not markets_df.empty else 0,
        "api_calls": client.calls,
        "api_retries": client.retries,
        "historical_cutoff": cutoff,
        "public_get_only": True,
        "orders_sent": False,
        "live_q50_control_touched": False,
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    if show:
        print("\n" + "=" * 112)
        print("SPORTS GAP BACKFILL COMPLETE")
        print("=" * 112)
        for k, v in manifest.items():
            if k != "historical_cutoff":
                print(f"{k:36s}: {v}")
        print("Output:", root)
    return manifest


def _main() -> None:
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

__all__ = ["VERSION", "PublicKalshi", "static_self_check", "run"]
