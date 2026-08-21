from __future__ import annotations

"""American-football-only random-sample backfill for the Kalshi sports gap study.

V1.4 fixes the V1.3 wrapper-composition bug by running the V1 discovery/candle
pipeline directly instead of nesting through V1.2.run(), which overwrote the
football-only discovery monkeypatch.

Universe:
- NFL / pro football
- NCAA / college football
- full-game team-vs-team winner markets only
- precise Kalshi game start required

Sampling:
- discover the complete eligible football event pool in the lookback window;
- uniformly sample up to ``max_events`` events without replacement;
- fixed seed by default for exact reproducibility;
- sort the sampled events by game time only after selection, so the sample is
  random rather than a recency truncation.

Public-read research only: no account/order endpoints, no orders, no live-Q50
imports, and no live control-file writes.
"""

import csv
import gzip
import json
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd

from . import sports_two_team_gap_backfill_v1 as V1
from . import sports_two_team_gap_backfill_v1_1 as V11
from . import sports_two_team_gap_backfill_v1_3_american_football as V13

VERSION = "KALSHI_SPORTS_TWO_TEAM_GAP_BACKFILL_V1_4_FOOTBALL_RANDOM_SAMPLE"
DEFAULT_SAMPLE_SEED = 20260821


def static_self_check(show: bool = True) -> dict[str, Any]:
    out = {
        "version": VERSION,
        "public_get_only": True,
        "portfolio_endpoints_used": False,
        "order_endpoints_used": False,
        "orders_sent": False,
        "live_q50_modules_imported": False,
        "control_files_written": False,
        "american_football_only": True,
        "nfl_included": True,
        "college_football_included": True,
        "full_game_winner_only": True,
        "precise_game_start_required": True,
        "uniform_event_random_sample": True,
        "sampling_without_replacement": True,
        "sample_seed_default": DEFAULT_SAMPLE_SEED,
        "v13_nested_wrapper_bug_bypassed": True,
        "ok": True,
    }
    if show:
        print("=" * 120)
        print("SPORTS GAP BACKFILL V1.4 FOOTBALL RANDOM SAMPLE — PUBLIC READ ONLY")
        print("=" * 120)
        for k, v in out.items():
            print(f"{k:58s}: {v}")
    return out


def _collect_candidate_pool(
    client: V1.PublicKalshi,
    *,
    lookback_days: int,
    show: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], float | None]:
    """Discover the full eligible NFL/NCAAF event pool before any sampling."""
    now = time.time()
    min_start = now - float(lookback_days) * 86400.0
    min_close_ts = int(min_start - 86400.0)

    cutoff: dict[str, Any] = {}
    cutoff_market_ts = None
    try:
        cutoff = client.get("/historical/cutoff")
        cutoff_market_ts = V1._iso_to_ts(cutoff.get("market_settled_ts"))
    except Exception:
        pass

    # IMPORTANT: call the football-specific discovery function directly.
    # Do not call V13.run()/V12.run(); their nested monkeypatch composition was
    # the reason the old run still reported the all-sports 191-series universe.
    series = V13.discover_series(client)

    series_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if show:
        print(f"\nAmerican-football series discovered: {len(series)}")

    for s_idx, s in enumerate(series, 1):
        st = str(s.get("ticker") or "")
        title = str(s.get("title") or "")
        tags = [str(x) for x in (s.get("tags") or [])]
        sport = V13._football_sport(s)

        series_rows.append({
            "series_ticker": st,
            "series_title": title,
            "sport": sport,
            "tags": "|".join(tags),
            "series_volume": V1._market_volume(s),
        })

        try:
            events, milestone_starts = V1.fetch_events_for_series(client, st, min_close_ts)
            event_by_id = {
                str(e.get("event_ticker") or ""): e
                for e in events
            }

            # Use V1.1's hardened market fetch: historical series_ticker only.
            markets = V11.fetch_markets_for_series(client, st)

            grouped: dict[str, list[dict[str, Any]]] = {}
            for m in markets:
                et = str(m.get("event_ticker") or "")
                if et in event_by_id:
                    grouped.setdefault(et, []).append(m)

            for et, group in grouped.items():
                if not V1._event_group_is_two_outcome(group):
                    continue

                event = event_by_id[et]
                start, start_source = V11.choose_game_start(
                    event,
                    group,
                    milestone_starts,
                )

                # Historical study: no future events.
                if start is None or start < min_start or start > now:
                    continue

                chosen = V1.choose_representative_market(group)

                candidates.append({
                    "series_ticker": st,
                    "series_title": title,
                    "sport": sport,
                    "event": event,
                    "event_ticker": et,
                    "event_market_count": len(group),
                    "market": chosen,
                    "game_start_ts": float(start),
                    "game_start_source": start_source,
                })

        except Exception as exc:
            errors.append({
                "stage": "discover_series",
                "series_ticker": st,
                "error": repr(exc),
            })

        if show and (s_idx == len(series) or s_idx % 5 == 0):
            print(
                f"  football series {s_idx}/{len(series)} | "
                f"eligible_pool={len(candidates)} | api_calls={client.calls}"
            )

    # Defensive dedupe by event. Keep the highest-volume representative if a
    # duplicate event somehow arrived through multiple series.
    by_event: dict[str, dict[str, Any]] = {}
    for item in candidates:
        et = str(item["event_ticker"])
        prev = by_event.get(et)
        if prev is None:
            by_event[et] = item
            continue
        if V1._market_volume(item["market"]) > V1._market_volume(prev["market"]):
            by_event[et] = item

    candidates = list(by_event.values())

    if show:
        nfl = sum(x["sport"] == "NFL" for x in candidates)
        ncaaf = sum(x["sport"] == "NCAAF" for x in candidates)
        print("\nEligible football event pool:", len(candidates))
        print("  NFL:  ", nfl)
        print("  NCAAF:", ncaaf)

    return series_rows, candidates, errors, cutoff, cutoff_market_ts


def run(
    *,
    run_dir: str | Path,
    lookback_days: int = 365,
    max_events: int = 250,
    requests_per_second: float = 1.5,
    pregame_hours: float = 24.0,
    max_game_hours: float = 12.0,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    show: bool = True,
) -> dict[str, Any]:
    static_self_check(show=show)

    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)

    client = V1.PublicKalshi(requests_per_second=requests_per_second)

    series_rows, pool, errors, cutoff, cutoff_market_ts = _collect_candidate_pool(
        client,
        lookback_days=lookback_days,
        show=show,
    )

    pool_size = len(pool)
    sample_n = min(int(max_events), pool_size) if max_events else pool_size

    if sample_n < pool_size:
        rng = random.Random(int(sample_seed))
        candidates = rng.sample(pool, sample_n)
    else:
        candidates = list(pool)

    # Sort only AFTER random selection for stable chronological output.
    candidates.sort(
        key=lambda x: (x["game_start_ts"], x["event_ticker"]),
        reverse=True,
    )

    sampled_nfl = sum(x["sport"] == "NFL" for x in candidates)
    sampled_ncaaf = sum(x["sport"] == "NCAAF" for x in candidates)

    if show:
        print("\n" + "=" * 112)
        print("FOOTBALL RANDOM SAMPLE SELECTED")
        print("=" * 112)
        print("eligible_pool_events: ", pool_size)
        print("sampled_events:       ", len(candidates))
        print("sample_seed:          ", int(sample_seed))
        print("sampled_NFL:          ", sampled_nfl)
        print("sampled_NCAAF:        ", sampled_ncaaf)

    pd.DataFrame(series_rows).to_csv(root / "series.csv", index=False)

    # Save the selected event list before candle work so the exact sample is
    # auditable even if the process is interrupted later.
    sample_rows = []
    for item in candidates:
        m = item["market"]
        sample_rows.append({
            "event_ticker": item["event_ticker"],
            "series_ticker": item["series_ticker"],
            "series_title": item["series_title"],
            "sport": item["sport"],
            "ticker": str(m.get("ticker") or ""),
            "game_start_ts": item["game_start_ts"],
            "game_start_iso": V1._ts_to_iso(item["game_start_ts"]),
            "game_start_source": item["game_start_source"],
            "market_volume": V1._market_volume(m),
        })
    pd.DataFrame(sample_rows).to_csv(root / "sampled_events.csv", index=False)

    market_rows: list[dict[str, Any]] = []
    paths_path = root / "minute_paths.csv.gz"

    with gzip.open(paths_path, "wt", encoding="utf-8", newline="") as gz:
        writer = csv.DictWriter(gz, fieldnames=V1.PATH_FIELDS)
        writer.writeheader()

        for i, item in enumerate(candidates, 1):
            m = item["market"]
            ticker = str(m.get("ticker") or "")
            start = float(item["game_start_ts"])

            close_ts = V1._iso_to_ts(m.get("close_time"))
            settle_ts = V1._iso_to_ts(m.get("settlement_ts"))
            natural_end = close_ts or settle_ts or (
                start + max_game_hours * 3600.0
            )
            end_ts = min(
                max(natural_end, start + 30 * 60.0),
                start + max_game_hours * 3600.0,
            )
            req_start = int(start - pregame_hours * 3600.0)
            req_end = int(end_ts)

            row_meta = {
                "event_ticker": item["event_ticker"],
                "series_ticker": item["series_ticker"],
                "series_title": item["series_title"],
                "sport": item["sport"],
                "ticker": ticker,
                "game_start_ts": start,
                "game_start_iso": V1._ts_to_iso(start),
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
                "market_volume": V1._market_volume(m),
                "settlement_value": V1._settlement_value(m),
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
                endpoint, candles = V1.candle_endpoint_and_rows(
                    client,
                    m,
                    item["series_ticker"],
                    req_start,
                    req_end,
                    cutoff_market_ts,
                )

                valid = 0
                kept = 0

                for c in candles:
                    n = V1.normalize_candle(c, row_meta)
                    if n is None:
                        continue
                    if n["end_period_ts"] < req_start or n["end_period_ts"] > req_end:
                        continue

                    writer.writerow({k: n.get(k) for k in V1.PATH_FIELDS})
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
                    "stage": "candles",
                    "ticker": ticker,
                    "event_ticker": item["event_ticker"],
                    "error": repr(exc),
                })

            market_rows.append(market_out)

            if show and (i == len(candidates) or i % 25 == 0):
                ok_n = sum(x.get("status") == "ok" for x in market_rows)
                print(
                    f"  sampled markets {i}/{len(candidates)} | ok={ok_n} | "
                    f"api_calls={client.calls} retries={client.retries}"
                )

    markets_df = pd.DataFrame(market_rows)
    markets_df.to_csv(root / "markets.csv", index=False)

    if errors:
        with (root / "errors.jsonl").open("w", encoding="utf-8") as fh:
            for row in errors:
                fh.write(json.dumps(row, default=str) + "\n")

    manifest = {
        "version": VERSION,
        "created_at": V1._ts_to_iso(time.time()),
        "run_dir": str(root),
        "lookback_days": int(lookback_days),
        "pregame_hours": float(pregame_hours),
        "max_game_hours": float(max_game_hours),
        "sample_size_requested": int(max_events),
        "sample_seed": int(sample_seed),
        "sampling_mode": "UNIFORM_EVENT_WITHOUT_REPLACEMENT_FIXED_SEED",
        "sports_series_discovered": len(series_rows),
        "candidate_pool_events": pool_size,
        "candidate_events": len(candidates),
        "sampled_nfl": sampled_nfl,
        "sampled_ncaaf": sampled_ncaaf,
        "markets_ok": int((markets_df.get("status") == "ok").sum()) if not markets_df.empty else 0,
        "api_calls": client.calls,
        "api_retries": client.retries,
        "historical_cutoff": cutoff,
        "public_get_only": True,
        "orders_sent": False,
        "live_q50_control_touched": False,
    }

    (root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    if show:
        print("\n" + "=" * 112)
        print("FOOTBALL RANDOM-SAMPLE BACKFILL COMPLETE")
        print("=" * 112)
        for k, v in manifest.items():
            if k != "historical_cutoff":
                print(f"{k:36s}: {v}")
        print("Output:", root)

    return manifest


__all__ = [
    "VERSION",
    "DEFAULT_SAMPLE_SEED",
    "static_self_check",
    "run",
]
