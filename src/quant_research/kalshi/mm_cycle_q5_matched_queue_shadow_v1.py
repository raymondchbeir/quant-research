from __future__ import annotations

"""Same-realization Q5 shadow with observed live queue-position corrections.

Diagnostic only.  This is NOT independent validation and NOT a production model.
It reads a completed live Q5 session, matches shadow quote opens to nearby real
CREATEs with the same ticker/role/side/price, and tests whether the queue-position
trajectory observed while those real orders rested explains the live/shadow gap.

Variants
--------
DISPLAYED_L1
    Frozen shadow baseline: join behind displayed L1, no cancellation-ahead credit.
SNAPSHOT_CORRECTED
    Start at displayed L1.  When a recorded live queue-position observation for the
    matched order arrives at its recorded timestamp, replace shadow queue_ahead by
    that observed absolute queue position.  No future queue observation is used.
FULL_QUEUE_ORACLE
    At quote join, seed queue_ahead with the first future observed live queue
    position, then apply all later observations.  This has deliberate look-ahead
    and is only an upper-information diagnostic for whether queue placement/motion
    could explain the PnL gap.

Important limitation
--------------------
V12.x CREATE does not query queue position immediately after every POST.  The live
engine inherits periodic queue polling, so fast orders that fill/cancel before a
poll may have no queue observation.  Coverage is therefore reported explicitly.

Safety
------
- NO exchange/API calls.
- NO orders.
- Source session is read-only.
- Writes only under results/kalshi_q5_matched_queue_shadow/.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q5_same_realization_shadow_v1 as BASE

REPLAY_VERSION = "MM_CYCLE_Q5_MATCHED_QUEUE_SHADOW_V1"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_matched_queue_shadow"
MATCH_MAX_LAG_S = 2.0
MATCH_MAX_LEAD_S = 0.05
EPS = 1e-10


def _ts_ms_to_s(x):
    z = OOS._f(x, np.nan)
    return z / 1000.0 if np.isfinite(z) else np.nan


def _parse_iso_s(x):
    return OOS._ts(x)


def _create_order_id(row):
    return str((row.get("response") or {}).get("order_id") or "")


def _price_key(x):
    z = OOS._f(x, np.nan)
    return round(float(z), 4) if np.isfinite(z) else None


def _load_live_order_catalog(session: Path):
    queue_by_order = defaultdict(list)
    role_by_order = {}

    for r in BASE._iter_jsonl(session / "queue_positions.jsonl") or []:
        oid = str(r.get("order_id") or "")
        q = OOS._f(r.get("queue_position"), np.nan)
        t = _parse_iso_s(r.get("time"))
        if not oid or not np.isfinite(q) or not np.isfinite(t):
            continue
        role = str(r.get("role") or "").upper()
        if role:
            role_by_order[oid] = role
        queue_by_order[oid].append(
            {
                "time_s": float(t),
                "queue_position": max(0.0, float(q)),
                "initial_flag": r.get("initial"),
                "raw": r,
            }
        )

    for rows in queue_by_order.values():
        rows.sort(key=lambda x: x["time_s"])

    # CREATE_SENT supplies role even when an order never survives to a queue poll.
    for r in BASE._iter_jsonl(session / "latency_events_v12.jsonl") or []:
        if str(r.get("event") or "") != "CREATE_SENT":
            continue
        oid = str(r.get("order_id") or "")
        role = str(r.get("role") or "").upper()
        if oid and role:
            role_by_order.setdefault(oid, role)

    orders = []
    by_id = {}
    for r in BASE._iter_jsonl(session / "orders.jsonl") or []:
        action = str(r.get("action") or "")
        if not action.startswith("CREATE"):
            continue
        oid = _create_order_id(r)
        payload = r.get("payload") or {}
        ticker = str(r.get("ticker") or payload.get("ticker") or "")
        side = str(payload.get("side") or "").upper()
        price = _price_key(payload.get("price"))
        send_s = _ts_ms_to_s((r.get("timing") or {}).get("request_send_wall_ms"))
        if not oid or not ticker or side not in {"BID", "ASK"} or price is None or not np.isfinite(send_s):
            continue
        rec = {
            "order_id": oid,
            "ticker": ticker,
            "role": role_by_order.get(oid, ""),
            "side": side,
            "price": price,
            "send_s": float(send_s),
            "queue_samples": list(queue_by_order.get(oid) or []),
        }
        if rec["queue_samples"]:
            rec["first_queue_position"] = rec["queue_samples"][0]["queue_position"]
            rec["first_queue_time_s"] = rec["queue_samples"][0]["time_s"]
        else:
            rec["first_queue_position"] = np.nan
            rec["first_queue_time_s"] = np.nan
        orders.append(rec)
        by_id[oid] = rec

    orders.sort(key=lambda x: x["send_s"])
    return orders, by_id


class LiveOrderMatcher:
    def __init__(self, orders):
        self.orders = list(orders)
        self.used = set()
        self.attempts = 0
        self.matches = 0
        self.matches_with_queue = 0
        self.join_send_lags_ms = []

    def match(self, ticker, desired, join_s):
        self.attempts += 1
        role = str(desired.get("role") or "").upper()
        side = str(desired.get("side") or "").upper()
        price = _price_key(desired.get("price"))
        candidates = []
        for o in self.orders:
            oid = o["order_id"]
            if oid in self.used:
                continue
            if o["ticker"] != str(ticker):
                continue
            if o["role"] and o["role"] != role:
                continue
            if o["side"] != side or o["price"] != price:
                continue
            lag = float(o["send_s"] - join_s)
            if lag < -MATCH_MAX_LEAD_S or lag > MATCH_MAX_LAG_S:
                continue
            candidates.append((abs(lag), lag, o))
        if not candidates:
            return None
        _, lag, o = min(candidates, key=lambda z: (z[0], z[2]["send_s"]))
        self.used.add(o["order_id"])
        self.matches += 1
        if o["queue_samples"]:
            self.matches_with_queue += 1
        self.join_send_lags_ms.append(1000.0 * lag)
        return o

    def summary(self):
        x = np.asarray(self.join_send_lags_ms, dtype=float)
        return {
            "quote_open_match_attempts": self.attempts,
            "matched_live_orders": self.matches,
            "matched_with_queue_observation": self.matches_with_queue,
            "match_pct": 100.0 * self.matches / self.attempts if self.attempts else np.nan,
            "queue_coverage_pct_of_matches": 100.0 * self.matches_with_queue / self.matches if self.matches else np.nan,
            "join_to_live_send_ms_median": float(np.median(x)) if x.size else np.nan,
            "join_to_live_send_ms_p95": float(np.quantile(x, 0.95)) if x.size else np.nan,
            "join_to_live_send_ms_max": float(np.max(x)) if x.size else np.nan,
        }


class MatchedQueueShadow(BASE.Q5FrozenCycleShadow):
    def __init__(self, session_dir, fee, matcher, mode):
        super().__init__(session_dir, fee)
        self.matcher = matcher
        self.mode = str(mode)
        self.queue_corrections = 0
        self.queue_forward = 0
        self.queue_backward = 0
        self.queue_same = 0
        self.queue_abs_change = 0.0
        self.oracle_seeded = 0
        self.matched_quote_opens = 0
        self.matched_quote_opens_with_queue = 0

    def _reconcile_quote(self, ticker, cur, elapsed, t):
        desired = self._desired_quote(ticker, cur, elapsed)
        old = self.quote.get(ticker)
        if self._quote_same(old, desired):
            return
        if old is not None:
            self.c["quote_cancels"] += 1
            self.quote.pop(ticker, None)
        if desired is None:
            return

        q = dict(desired)
        q.update(
            {
                "join_ts": float(t),
                "queue_ahead_initial": float(desired["queue_ahead"]),
                "remaining_qty": float(desired["qty"]),
            }
        )

        match = self.matcher.match(ticker, desired, float(t))
        if match is not None:
            self.matched_quote_opens += 1
            q["matched_live_order_id"] = match["order_id"]
            q["matched_live_send_s"] = match["send_s"]
            if match["queue_samples"]:
                self.matched_quote_opens_with_queue += 1
            if self.mode == "FULL_QUEUE_ORACLE" and np.isfinite(match.get("first_queue_position", np.nan)):
                q["queue_ahead"] = float(match["first_queue_position"])
                q["queue_ahead_oracle_seed"] = float(match["first_queue_position"])
                self.oracle_seeded += 1

        self.quote[ticker] = q
        self.c[f"{q['role']}_quote_opens"] += 1

    def apply_queue_snapshot(self, row):
        if self.mode == "DISPLAYED_L1":
            return
        ticker = str(row.get("ticker") or "")
        oid = str(row.get("order_id") or "")
        qpos = OOS._f(row.get("queue_position"), np.nan)
        if not ticker or not oid or not np.isfinite(qpos):
            return
        q = self.quote.get(ticker)
        if q is None or str(q.get("matched_live_order_id") or "") != oid:
            return
        old = float(q.get("queue_ahead", 0.0))
        new = max(0.0, float(qpos))
        q["queue_ahead"] = new
        self.queue_corrections += 1
        self.queue_abs_change += abs(new - old)
        if new < old - OOS.EPS:
            self.queue_forward += 1
        elif new > old + OOS.EPS:
            self.queue_backward += 1
        else:
            self.queue_same += 1

    def queue_summary(self):
        return {
            "mode": self.mode,
            "matched_quote_opens": self.matched_quote_opens,
            "matched_quote_opens_with_queue": self.matched_quote_opens_with_queue,
            "oracle_seeded_quote_opens": self.oracle_seeded,
            "queue_snapshot_corrections_applied": self.queue_corrections,
            "queue_forward_corrections": self.queue_forward,
            "queue_backward_corrections": self.queue_backward,
            "queue_same_corrections": self.queue_same,
            "queue_abs_change_total": self.queue_abs_change,
            **self.matcher.summary(),
        }


def _next_queue(it, selected_tickers, known_orders):
    for row in it:
        ticker = str(row.get("ticker") or "")
        oid = str(row.get("order_id") or "")
        if ticker not in selected_tickers or oid not in known_orders:
            continue
        t = _parse_iso_s(row.get("time"))
        q = OOS._f(row.get("queue_position"), np.nan)
        if np.isfinite(t) and np.isfinite(q):
            return float(t), row
    return None


def _run_variant(source, raw, selected_tickers, meta_rows, fee, orders, by_id, mode, show=False):
    workspace = OUTPUT_ROOT / source.name / mode.lower()
    workspace.mkdir(parents=True, exist_ok=False)
    matcher = LiveOrderMatcher(orders)
    shadow = MatchedQueueShadow(workspace, fee, matcher, mode)

    for row in meta_rows:
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        shadow.meta[ticker] = row
        shadow.series_by_ticker[ticker] = str(row.get("series_ticker") or "")
        shadow.close_by_ticker[ticker] = str(row.get("close_time") or "")

    book_it = iter(BASE._iter_jsonl(raw / "book_top3_events.jsonl") or [])
    trade_it = iter(BASE._iter_jsonl(raw / "trades_event_time.jsonl") or [])
    queue_it = iter(BASE._iter_jsonl(source / "queue_positions.jsonl") or [])

    b = BASE._next_selected(book_it, selected_tickers)
    tr = BASE._next_selected(trade_it, selected_tickers)
    qev = _next_queue(queue_it, selected_tickers, set(by_id)) if mode != "DISPLAYED_L1" else None
    if b is None and tr is None:
        raise RuntimeError("No selected raw events found.")

    first_ts = min(x[0] for x in (b, tr) if x is not None)
    last_ts = first_ts
    shadow.started_at = pd.Timestamp(first_ts, unit="s", tz="UTC")
    shadow.thread_alive = True
    shadow.emit("MATCHED_QUEUE_REPLAY_START", detail=mode)

    events = books = trades = queues = 0
    while b is not None or tr is not None or qev is not None:
        choices = []
        if b is not None:
            choices.append((b[0], 0, "book"))
        if tr is not None:
            choices.append((tr[0], 1, "trade"))
        if qev is not None:
            choices.append((qev[0], 2, "queue"))
        _, _, typ = min(choices)

        if typ == "book":
            t, row = b
            shadow._on_book(t, row)
            books += 1
            b = BASE._next_selected(book_it, selected_tickers)
        elif typ == "trade":
            t, row = tr
            shadow._on_trade(t, row)
            trades += 1
            tr = BASE._next_selected(trade_it, selected_tickers)
        else:
            t, row = qev
            shadow.apply_queue_snapshot(row)
            queues += 1
            qev = _next_queue(queue_it, selected_tickers, set(by_id))

        last_ts = max(last_ts, float(t))
        events += 1
        shadow._update_drawdown()
        if show and events % 500_000 == 0:
            print(mode, "events", f"{events:,}", "fills", int(shadow.c["fill_events"]))

    shadow.thread_alive = False
    shadow.emit("MATCHED_QUEUE_REPLAY_STOP", events=events)
    shadow._save()

    passive = float(shadow.passive_matched_pnl)
    forced = float(shadow.forced_liq_gross_pnl)
    fees = float(shadow.taker_trade_fees)
    net = passive + forced - fees

    return {
        "mode": mode,
        "shadow_net_pnl": net,
        "passive_matched_pnl": passive,
        "forced_liq_gross_pnl": forced,
        "taker_trade_fees": fees,
        "fill_events": int(shadow.c["fill_events"]),
        "fill_qty": shadow.c["fill_qty_x1000"] / 1000.0,
        "cycles_started": int(shadow.c["cycles_started"]),
        "cycles_completed": int(shadow.c["cycles_completed"]),
        "forced_liq_qty": shadow.c["forced_liq_qty_x1000"] / 1000.0,
        "max_drawdown": float(shadow.max_drawdown),
        "events_replayed": events,
        "book_rows": books,
        "trade_rows": trades,
        "queue_rows_seen": queues,
        "first_receipt_ts": first_ts,
        "last_receipt_ts": last_ts,
        "queue": shadow.queue_summary(),
    }


def run_q5_matched_queue_shadow(source_session, *, show=True):
    """Run baseline + queue-corrected Q5 same-realization diagnostics. NO API."""
    source = Path(source_session).resolve()
    raw = source / "raw_capture"
    required = [
        source / "events.jsonl",
        source / "orders.jsonl",
        source / "queue_positions.jsonl",
        source / "latency_events_v12.jsonl",
        source / "fee_preflight.json",
        source / "final_summary.json",
        raw / "book_top3_events.jsonl",
        raw / "trades_event_time.jsonl",
        raw / "market_metadata.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required matched-queue artifacts: " + " | ".join(missing))

    cfg = OOS._read_json(source / "process_config.json", {}) or {}
    if str(cfg.get("mode") or "") != "LIVE_Q5_1H" or abs(OOS._f(cfg.get("quote_size"), np.nan) - BASE.QTY) > 1e-9:
        raise RuntimeError("Expected completed LIVE_Q5_1H / Q5 source session.")
    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored fee preflight was not PASS.")

    windows = BASE._live_windows(source)
    meta_rows, meta_by_ticker = BASE._metadata(raw)
    selected_tickers = {
        t for t, row in meta_by_ticker.items()
        if str(row.get("close_time") or "") in set(windows)
    }
    orders, by_id = _load_live_order_catalog(source)
    if not orders:
        raise RuntimeError("No live CREATE catalog could be reconstructed.")

    base_out = OUTPUT_ROOT / source.name
    if base_out.exists():
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        base_out = OUTPUT_ROOT / f"{source.name}_{stamp}"
    base_out.mkdir(parents=True, exist_ok=False)

    # _run_variant expects OUTPUT_ROOT/source.name; use a temporary rebinding by
    # passing a lightweight source proxy name through a dedicated local root.
    global OUTPUT_ROOT
    original_root = OUTPUT_ROOT
    OUTPUT_ROOT = base_out.parent
    proxy_source = source
    if base_out.name != source.name:
        # Avoid collision by making variant workspaces explicitly below base_out.
        pass

    results = []
    try:
        # Create variant directories directly below the unique base_out.  We do so
        # by temporarily setting OUTPUT_ROOT to its parent and using a proxy path
        # whose name is the unique output name.
        class _Proxy:
            pass
        proxy = _Proxy()
        proxy.name = base_out.name
        # _run_variant uses source for queue paths, so keep real source and instead
        # temporarily root at base_out.parent; unique base_out name is handled by
        # a small local wrapper below.
        def run_mode(mode):
            workspace = base_out / mode.lower()
            workspace.mkdir(parents=True, exist_ok=False)
            matcher = LiveOrderMatcher(orders)
            shadow = MatchedQueueShadow(workspace, fee, matcher, mode)
            for row in meta_rows:
                ticker = str(row.get("ticker") or "")
                if ticker in selected_tickers:
                    shadow.meta[ticker] = row
                    shadow.series_by_ticker[ticker] = str(row.get("series_ticker") or "")
                    shadow.close_by_ticker[ticker] = str(row.get("close_time") or "")
            book_it = iter(BASE._iter_jsonl(raw / "book_top3_events.jsonl") or [])
            trade_it = iter(BASE._iter_jsonl(raw / "trades_event_time.jsonl") or [])
            queue_it = iter(BASE._iter_jsonl(source / "queue_positions.jsonl") or [])
            b = BASE._next_selected(book_it, selected_tickers)
            tr = BASE._next_selected(trade_it, selected_tickers)
            qev = _next_queue(queue_it, selected_tickers, set(by_id)) if mode != "DISPLAYED_L1" else None
            first_ts = min(x[0] for x in (b, tr) if x is not None)
            last_ts = first_ts
            shadow.started_at = pd.Timestamp(first_ts, unit="s", tz="UTC")
            shadow.thread_alive = True
            events = books = trades = queues = 0
            while b is not None or tr is not None or qev is not None:
                choices = []
                if b is not None: choices.append((b[0], 0, "book"))
                if tr is not None: choices.append((tr[0], 1, "trade"))
                if qev is not None: choices.append((qev[0], 2, "queue"))
                _, _, typ = min(choices)
                if typ == "book":
                    t, row = b; shadow._on_book(t, row); books += 1
                    b = BASE._next_selected(book_it, selected_tickers)
                elif typ == "trade":
                    t, row = tr; shadow._on_trade(t, row); trades += 1
                    tr = BASE._next_selected(trade_it, selected_tickers)
                else:
                    t, row = qev; shadow.apply_queue_snapshot(row); queues += 1
                    qev = _next_queue(queue_it, selected_tickers, set(by_id))
                last_ts = max(last_ts, float(t)); events += 1; shadow._update_drawdown()
            shadow.thread_alive = False
            shadow._save()
            passive = float(shadow.passive_matched_pnl)
            forced = float(shadow.forced_liq_gross_pnl)
            fees = float(shadow.taker_trade_fees)
            return {
                "mode": mode,
                "shadow_net_pnl": passive + forced - fees,
                "passive_matched_pnl": passive,
                "forced_liq_gross_pnl": forced,
                "taker_trade_fees": fees,
                "fill_events": int(shadow.c["fill_events"]),
                "fill_qty": shadow.c["fill_qty_x1000"] / 1000.0,
                "cycles_started": int(shadow.c["cycles_started"]),
                "cycles_completed": int(shadow.c["cycles_completed"]),
                "forced_liq_qty": shadow.c["forced_liq_qty_x1000"] / 1000.0,
                "max_drawdown": float(shadow.max_drawdown),
                "queue_rows_seen": queues,
                "queue": shadow.queue_summary(),
            }

        for mode in ("DISPLAYED_L1", "SNAPSHOT_CORRECTED", "FULL_QUEUE_ORACLE"):
            results.append(run_mode(mode))
    finally:
        OUTPUT_ROOT = original_root

    live_final = OOS._read_json(source / "final_summary.json", {}) or {}
    live_pnl = OOS._f(live_final.get("account_pnl_usd"), np.nan)
    for r in results:
        r["live_account_pnl"] = live_pnl
        r["live_minus_shadow"] = live_pnl - r["shadow_net_pnl"] if np.isfinite(live_pnl) else np.nan

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "replay_version": REPLAY_VERSION,
        "source_session": str(source),
        "output_dir": str(base_out),
        "live_windows": windows,
        "selected_tickers": len(selected_tickers),
        "live_create_orders_cataloged": len(orders),
        "live_orders_with_queue_observations": sum(bool(o["queue_samples"]) for o in orders),
        "live_account_pnl": live_pnl,
        "variants": results,
        "same_realization_only": True,
        "independent_validation": False,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "limitation": (
            "V12.x queue positions were periodically polled, not sampled at every CREATE. "
            "SNAPSHOT_CORRECTED uses only observations once they existed; FULL_QUEUE_ORACLE seeds with the first future observation and therefore has deliberate look-ahead."
        ),
    }
    OOS._atomic_json(base_out / "matched_queue_shadow_summary.json", summary)
    pd.DataFrame([
        {
            "mode": r["mode"],
            "shadow_net_pnl": r["shadow_net_pnl"],
            "live_account_pnl": r["live_account_pnl"],
            "live_minus_shadow": r["live_minus_shadow"],
            "fill_events": r["fill_events"],
            "fill_qty": r["fill_qty"],
            "forced_liq_qty": r["forced_liq_qty"],
            "max_drawdown": r["max_drawdown"],
            **{f"queue_{k}": v for k, v in r["queue"].items() if k != "mode"},
        }
        for r in results
    ]).to_csv(base_out / "matched_queue_variants.csv", index=False)

    if show:
        print("=" * 118)
        print("Q5 MATCHED LIVE-QUEUE SHADOW — SAME REALIZATION / READ ONLY")
        print("=" * 118)
        print("Source:", source)
        print("Live PnL:", f"${live_pnl:+.4f}")
        print("Live CREATEs cataloged:", len(orders))
        print("CREATEs with >=1 queue observation:", sum(bool(o["queue_samples"]) for o in orders))
        print()
        for r in results:
            q = r["queue"]
            print(f"{r['mode']:<22} shadow=${r['shadow_net_pnl']:+.4f}  live-shadow=${r['live_minus_shadow']:+.4f}  fills={r['fill_events']} qty={r['fill_qty']:.2f}")
            print(
                "  match attempts/matched/with-queue:",
                q["quote_open_match_attempts"], q["matched_live_orders"], q["matched_with_queue_observation"],
                "| snapshot corrections:", q["queue_snapshot_corrections_applied"],
                "| forward/backward:", q["queue_forward_corrections"], q["queue_backward_corrections"],
            )
            print(
                "  join->send ms median/p95/max:",
                q["join_to_live_send_ms_median"], q["join_to_live_send_ms_p95"], q["join_to_live_send_ms_max"],
            )
        print()
        print("Interpretation:")
        print("  SNAPSHOT_CORRECTED moves only when a queue observation actually existed at that time.")
        print("  FULL_QUEUE_ORACLE also seeds from the first future queue observation; it is deliberately non-causal.")
        print("  If either closes much of the $6.43 live-shadow gap, queue placement/motion is a major suspect.")
        print("  If not, focus shifts to fill selection/exit timing rather than queue-ahead mechanics.")
        print("Output:", base_out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 118)

    return summary


__all__ = ["run_q5_matched_queue_shadow"]
