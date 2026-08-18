from __future__ import annotations

"""Read-only same-realization Q5 shadow replay for a completed live V12.x session.

Purpose
-------
Replay the exact frozen Candidate-C mechanics against the authoritative V5 raw
book/trade capture from a completed live Q5 session, but with quote size Q5 so
that realized live PnL and shadow PnL are directly comparable in dollars.

Scientific guardrails
---------------------
- SAME-REALIZATION diagnostic only; this is not independent validation.
- Reads only the completed live session and its raw_capture directory.
- Makes NO exchange/API calls and sends NO orders.
- Writes only under results/kalshi_q5_same_realization_shadow/.
- Reuses the frozen OOS shadow mechanics rather than re-implementing fill logic.
- Restricts replay to the exact windows the live engine recorded as WINDOW_START.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS

REPLAY_VERSION = "MM_CYCLE_Q5_SAME_REALIZATION_SHADOW_V1"
QTY = 5.0
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_same_realization_shadow"
EPS = 1e-10


def _iter_jsonl(path: Path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _new_output_dir(source_name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_ROOT / source_name
    if not base.exists():
        return base.resolve()
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    return (OUTPUT_ROOT / f"{source_name}_{stamp}").resolve()


def _live_windows(session: Path):
    out = []
    seen = set()
    for row in _iter_jsonl(session / "events.jsonl") or []:
        if str(row.get("event") or "") != "WINDOW_START":
            continue
        close = str(row.get("close_time") or "")
        if close and close not in seen:
            seen.add(close)
            out.append(close)
    return out


def _metadata(raw: Path):
    rows = []
    by_ticker = {}
    for row in _iter_jsonl(raw / "market_metadata.jsonl") or []:
        ticker = str(row.get("ticker") or "")
        if not ticker:
            continue
        rows.append(row)
        by_ticker[ticker] = row
    return rows, by_ticker


def _next_selected(it, selected_tickers):
    for row in it:
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        t = OOS._ts(row.get("receipt_time"))
        if np.isfinite(t):
            return float(t), row
    return None


class Q5FrozenCycleShadow(OOS.FrozenCycleShadow):
    """Exact frozen Candidate-C shadow mechanics with Q5 instead of frozen Q10."""

    def _desired_quote(self, ticker, cur, elapsed):
        if cur is None or not (0.0 <= elapsed < 300.0) or ticker in self.finalized:
            return None
        inv = float(self.inventory[ticker])
        if abs(inv) <= OOS.EPS:
            side = OOS._entry_side(cur)
            if side is None:
                return None
            return {
                "role": "ENTRY",
                "side": side,
                "price": cur["bid"] if side == "BID" else cur["ask"],
                "qty": QTY,
                "queue_ahead": cur["bid_q1"] if side == "BID" else cur["ask_q1"],
            }
        side = "ASK" if inv > 0 else "BID"
        return {
            "role": "EXIT",
            "side": side,
            "price": cur["ask"] if side == "ASK" else cur["bid"],
            "qty": abs(inv),
            "queue_ahead": cur["ask_q1"] if side == "ASK" else cur["bid_q1"],
        }


def run_q5_same_realization_shadow(source_session, *, show=True):
    """Replay a completed live Q5 raw capture. NO API. NO ORDERS. Source read-only."""
    source = Path(source_session).resolve()
    raw = source / "raw_capture"

    required = [
        source / "events.jsonl",
        source / "fee_preflight.json",
        source / "final_summary.json",
        raw / "book_top3_events.jsonl",
        raw / "trades_event_time.jsonl",
        raw / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Q5 artifacts: " + " | ".join(missing))

    cfg = OOS._read_json(source / "process_config.json", {}) or {}
    if str(cfg.get("mode") or "") != "LIVE_Q5_1H":
        raise RuntimeError(f"Expected LIVE_Q5_1H session, got mode={cfg.get('mode')!r}")
    actual_q = OOS._f(cfg.get("quote_size"), np.nan)
    if not np.isfinite(actual_q) or abs(actual_q - QTY) > 1e-9:
        raise RuntimeError(f"Expected live quote size Q5, got {actual_q}")

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored live fee preflight was not PASS; refusing shadow replay.")

    windows = _live_windows(source)
    if not windows:
        raise RuntimeError("No live WINDOW_START events found.")

    meta_rows, meta_by_ticker = _metadata(raw)
    selected_tickers = {
        t for t, row in meta_by_ticker.items()
        if str(row.get("close_time") or "") in set(windows)
    }
    if not selected_tickers:
        raise RuntimeError("No raw tickers matched the live WINDOW_START close times.")

    out = _new_output_dir(source.name)
    out.mkdir(parents=True, exist_ok=False)

    # FrozenCycleShadow writes only below the supplied workspace. We invoke its
    # event handlers directly, so source raw files remain untouched.
    health_src = raw / "health.json"
    if health_src.exists():
        try:
            (out / "health.json").write_text(health_src.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    shadow = Q5FrozenCycleShadow(out, fee)

    for row in meta_rows:
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        series = str(row.get("series_ticker") or "")
        shadow.meta[ticker] = row
        shadow.series_by_ticker[ticker] = series
        shadow.close_by_ticker[ticker] = str(row.get("close_time") or "")

    book_it = iter(_iter_jsonl(raw / "book_top3_events.jsonl") or [])
    trade_it = iter(_iter_jsonl(raw / "trades_event_time.jsonl") or [])
    b = _next_selected(book_it, selected_tickers)
    tr = _next_selected(trade_it, selected_tickers)
    if b is None and tr is None:
        raise RuntimeError("No selected timed raw events found.")

    first_ts = min(x[0] for x in (b, tr) if x is not None)
    last_ts = first_ts
    shadow.started_at = pd.Timestamp(first_ts, unit="s", tz="UTC")
    shadow.thread_alive = True
    shadow.emit("Q5_SAME_REALIZATION_REPLAY_START", detail=str(source))

    book_rows = 0
    trade_rows = 0
    events = 0

    while b is not None or tr is not None:
        if tr is None:
            choose_book = True
        elif b is None:
            choose_book = False
        elif b[0] < tr[0] - EPS:
            choose_book = True
        elif tr[0] < b[0] - EPS:
            choose_book = False
        else:
            choose_book = True

        if choose_book:
            t, row = b
            shadow._on_book(t, row)
            book_rows += 1
            b = _next_selected(book_it, selected_tickers)
        else:
            t, row = tr
            shadow._on_trade(t, row)
            trade_rows += 1
            tr = _next_selected(trade_it, selected_tickers)

        last_ts = max(last_ts, float(t))
        events += 1
        shadow._update_drawdown()

        if show and events % 500_000 == 0:
            print(
                f"replayed {events:,} events | books={book_rows:,} trades={trade_rows:,} "
                f"fills={int(shadow.c['fill_events']):,}"
            )

    shadow.thread_alive = False
    shadow.emit("Q5_SAME_REALIZATION_REPLAY_STOP", events=events)

    # Build exact per-window/per-asset economics from matched passive PnL plus
    # M5 liquidation gross minus taker fee.
    agg = defaultdict(lambda: {
        "passive_matched_pnl": 0.0,
        "forced_liq_gross_pnl": 0.0,
        "taker_trade_fees": 0.0,
        "fill_events": 0,
        "fill_qty": 0.0,
        "forced_liq_qty": 0.0,
    })

    for fill in shadow.fills:
        ticker = str(fill.get("ticker") or "")
        close = str(shadow.close_by_ticker.get(ticker) or "")
        series = str(shadow.series_by_ticker.get(ticker) or "")
        key = (close, series, ticker)
        a = agg[key]
        a["passive_matched_pnl"] += OOS._f(fill.get("matched_pnl_delta"), 0.0)
        a["fill_events"] += 1
        a["fill_qty"] += OOS._f(fill.get("qty"), 0.0)

    for ticker, c in shadow.contracts.items():
        close = str(c.get("close_time") or shadow.close_by_ticker.get(ticker) or "")
        series = str(c.get("series") or shadow.series_by_ticker.get(ticker) or "")
        key = (close, series, ticker)
        a = agg[key]
        a["forced_liq_gross_pnl"] += OOS._f(c.get("forced_liquidation_gross_pnl"), 0.0)
        a["taker_trade_fees"] += OOS._f(c.get("taker_trade_fee"), 0.0)
        a["forced_liq_qty"] += OOS._f(c.get("forced_liquidation_qty"), 0.0)

    detail_rows = []
    for (close, series, ticker), a in agg.items():
        net = (
            a["passive_matched_pnl"]
            + a["forced_liq_gross_pnl"]
            - a["taker_trade_fees"]
        )
        detail_rows.append({
            "close_time": close,
            "series": series,
            "ticker": ticker,
            **a,
            "net_pnl": net,
        })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        detail = pd.DataFrame(columns=[
            "close_time", "series", "ticker", "passive_matched_pnl",
            "forced_liq_gross_pnl", "taker_trade_fees", "fill_events",
            "fill_qty", "forced_liq_qty", "net_pnl",
        ])

    by_window = (
        detail.groupby("close_time", as_index=False)
        .agg(
            passive_matched_pnl=("passive_matched_pnl", "sum"),
            forced_liq_gross_pnl=("forced_liq_gross_pnl", "sum"),
            taker_trade_fees=("taker_trade_fees", "sum"),
            fill_events=("fill_events", "sum"),
            fill_qty=("fill_qty", "sum"),
            forced_liq_qty=("forced_liq_qty", "sum"),
            net_pnl=("net_pnl", "sum"),
        )
        .sort_values("close_time")
    )

    by_asset = (
        detail.groupby("series", as_index=False)
        .agg(
            passive_matched_pnl=("passive_matched_pnl", "sum"),
            forced_liq_gross_pnl=("forced_liq_gross_pnl", "sum"),
            taker_trade_fees=("taker_trade_fees", "sum"),
            fill_events=("fill_events", "sum"),
            fill_qty=("fill_qty", "sum"),
            forced_liq_qty=("forced_liq_qty", "sum"),
            net_pnl=("net_pnl", "sum"),
        )
        .sort_values("net_pnl")
    )

    passive = float(shadow.passive_matched_pnl)
    forced = float(shadow.forced_liq_gross_pnl)
    fees = float(shadow.taker_trade_fees)
    shadow_net = passive + forced - fees

    live_final = OOS._read_json(source / "final_summary.json", {}) or {}
    live_pnl = OOS._f(live_final.get("account_pnl_usd"), np.nan)
    gap = live_pnl - shadow_net if np.isfinite(live_pnl) else np.nan

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "replay_version": REPLAY_VERSION,
        "source_session": str(source),
        "raw_source": str(raw),
        "output_dir": str(out),
        "same_realization_only": True,
        "independent_validation": False,
        "quote_size": QTY,
        "live_windows": windows,
        "selected_tickers": len(selected_tickers),
        "book_rows_replayed": book_rows,
        "trade_rows_replayed": trade_rows,
        "total_events_replayed": events,
        "first_receipt_ts": first_ts,
        "last_receipt_ts": last_ts,
        "source_event_span_hours": (last_ts - first_ts) / 3600.0,
        "shadow_passive_matched_pnl": passive,
        "shadow_forced_liq_gross_pnl": forced,
        "shadow_taker_trade_fees": fees,
        "shadow_net_pnl": shadow_net,
        "shadow_fill_events": int(shadow.c["fill_events"]),
        "shadow_fill_qty": shadow.c["fill_qty_x1000"] / 1000.0,
        "shadow_cycles_started": int(shadow.c["cycles_started"]),
        "shadow_cycles_completed": int(shadow.c["cycles_completed"]),
        "shadow_forced_liquidations": int(shadow.c["forced_liquidations"]),
        "shadow_forced_liq_qty": shadow.c["forced_liq_qty_x1000"] / 1000.0,
        "shadow_max_drawdown_online": float(shadow.max_drawdown),
        "live_account_pnl": live_pnl,
        "live_minus_shadow_pnl": gap,
        "orders_sent": False,
        "exchange_api_called": False,
        "source_modified": False,
        "interpretation_guardrail": (
            "This is a same-realization diagnostic. A shadow loss similar to live supports a bad market realization; "
            "a materially better shadow than live points toward fill/execution selection. It is not new OOS evidence."
        ),
    }

    OOS._atomic_json(out / "q5_same_realization_shadow_summary.json", summary)
    detail.to_csv(out / "q5_shadow_by_contract.csv", index=False)
    by_window.to_csv(out / "q5_shadow_by_window.csv", index=False)
    by_asset.to_csv(out / "q5_shadow_by_asset.csv", index=False)

    if show:
        print("=" * 116)
        print("Q5 SAME-REALIZATION FROZEN SHADOW — READ ONLY / NO API / NO ORDERS")
        print("=" * 116)
        print("Source:                    ", source)
        print("Output:                    ", out)
        print("Live windows:              ", len(windows), windows)
        print("Selected contracts:        ", len(selected_tickers))
        print(f"Events replayed:            {events:,}")
        print(f"Shadow passive PnL:        ${passive:+.4f}")
        print(f"Shadow forced M5 gross:    ${forced:+.4f}")
        print(f"Shadow taker fees:         ${fees:.4f}")
        print(f"SHADOW NET Q5:             ${shadow_net:+.4f}")
        print(f"LIVE NET Q5:               ${live_pnl:+.4f}" if np.isfinite(live_pnl) else "LIVE NET Q5: unknown")
        print(f"LIVE - SHADOW GAP:         ${gap:+.4f}" if np.isfinite(gap) else "LIVE - SHADOW GAP: unknown")
        print(f"Shadow fills / qty:         {int(shadow.c['fill_events'])} / {shadow.c['fill_qty_x1000']/1000.0:.3f}")
        print(f"Shadow cycles:              {int(shadow.c['cycles_completed'])}/{int(shadow.c['cycles_started'])} complete")
        print(f"Shadow forced M5 qty:       {shadow.c['forced_liq_qty_x1000']/1000.0:.3f}")
        print(f"Shadow max DD:             ${float(shadow.max_drawdown):+.4f}")
        print("\nBY WINDOW")
        print(by_window.to_string(index=False))
        print("\nBY ASSET")
        print(by_asset.to_string(index=False))
        print("\nSAME-REALIZATION ONLY — NOT INDEPENDENT VALIDATION")
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 116)

    return {
        "summary": summary,
        "by_window": by_window,
        "by_asset": by_asset,
        "by_contract": detail,
        "output_dir": str(out),
    }


__all__ = ["run_q5_same_realization_shadow", "REPLAY_VERSION", "QTY"]
