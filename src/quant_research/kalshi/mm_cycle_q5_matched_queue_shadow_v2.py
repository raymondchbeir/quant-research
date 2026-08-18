from __future__ import annotations

"""Same-realization Q5 shadow with observed live queue-position corrections.

Diagnostic only. NO exchange/API calls and NO orders.

Variants:
- DISPLAYED_L1: frozen baseline.
- SNAPSHOT_CORRECTED: start at displayed L1, then replace queue_ahead with
  recorded live queue position when that observation actually occurred.
- FULL_QUEUE_ORACLE: additionally seed queue_ahead at quote join with the first
  future live queue observation. This is deliberately non-causal and is used
  only to test whether queue placement/motion could plausibly explain the gap.

Important limitation: V12.x did not query queue position immediately after every
CREATE; queue_positions.jsonl is periodic. Fast orders can therefore have no
queue observation. Coverage is reported explicitly.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q5_same_realization_shadow_v1 as BASE

REPLAY_VERSION = "MM_CYCLE_Q5_MATCHED_QUEUE_SHADOW_V2"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_matched_queue_shadow_v2"
MATCH_MAX_LAG_S = 2.0
MATCH_MAX_LEAD_S = 0.05


def _parse_s(x):
    return OOS._ts(x)


def _ms_to_s(x):
    z = OOS._f(x, np.nan)
    return z / 1000.0 if np.isfinite(z) else np.nan


def _price_key(x):
    z = OOS._f(x, np.nan)
    return round(float(z), 4) if np.isfinite(z) else None


def _unique_output(source_name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / source_name
    if out.exists():
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_ROOT / f"{source_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _live_order_catalog(session: Path):
    queue_by_oid = defaultdict(list)
    role_by_oid = {}

    for r in BASE._iter_jsonl(session / "queue_positions.jsonl") or []:
        oid = str(r.get("order_id") or "")
        t = _parse_s(r.get("time"))
        q = OOS._f(r.get("queue_position"), np.nan)
        if not oid or not np.isfinite(t) or not np.isfinite(q):
            continue
        role = str(r.get("role") or "").upper()
        if role:
            role_by_oid[oid] = role
        disp = OOS._f(
            r.get("displayed_l1_ahead_at_join", r.get("displayed_l1_ahead")),
            np.nan,
        )
        queue_by_oid[oid].append(
            {
                "time_s": float(t),
                "queue_position": max(0.0, float(q)),
                "displayed_l1_ahead_at_join": float(disp) if np.isfinite(disp) else np.nan,
            }
        )

    for rows in queue_by_oid.values():
        rows.sort(key=lambda z: z["time_s"])

    for r in BASE._iter_jsonl(session / "latency_events_v12.jsonl") or []:
        if str(r.get("event") or "") != "CREATE_SENT":
            continue
        oid = str(r.get("order_id") or "")
        role = str(r.get("role") or "").upper()
        if oid and role:
            role_by_oid.setdefault(oid, role)

    orders = []
    by_id = {}
    for r in BASE._iter_jsonl(session / "orders.jsonl") or []:
        if not str(r.get("action") or "").startswith("CREATE"):
            continue
        response = r.get("response") or {}
        payload = r.get("payload") or {}
        oid = str(response.get("order_id") or "")
        ticker = str(r.get("ticker") or payload.get("ticker") or "")
        side = str(payload.get("side") or "").upper()
        price = _price_key(payload.get("price"))
        send_s = _ms_to_s((r.get("timing") or {}).get("request_send_wall_ms"))
        if (
            not oid
            or not ticker
            or side not in {"BID", "ASK"}
            or price is None
            or not np.isfinite(send_s)
        ):
            continue

        samples = list(queue_by_oid.get(oid) or [])
        rec = {
            "order_id": oid,
            "ticker": ticker,
            "role": role_by_oid.get(oid, ""),
            "side": side,
            "price": price,
            "send_s": float(send_s),
            "queue_samples": samples,
            "first_queue_position": samples[0]["queue_position"] if samples else np.nan,
            "first_queue_time_s": samples[0]["time_s"] if samples else np.nan,
            "first_displayed_l1": samples[0]["displayed_l1_ahead_at_join"] if samples else np.nan,
            "last_queue_position": samples[-1]["queue_position"] if samples else np.nan,
        }
        orders.append(rec)
        by_id[oid] = rec

    orders.sort(key=lambda z: z["send_s"])
    return orders, by_id


def _catalog_stats(orders):
    with_q = [o for o in orders if o["queue_samples"]]
    diffs, motions, first_lags = [], [], []

    for o in with_q:
        fq = OOS._f(o.get("first_queue_position"), np.nan)
        fd = OOS._f(o.get("first_displayed_l1"), np.nan)
        lq = OOS._f(o.get("last_queue_position"), np.nan)
        ft = OOS._f(o.get("first_queue_time_s"), np.nan)
        ss = OOS._f(o.get("send_s"), np.nan)
        if np.isfinite(fq) and np.isfinite(fd):
            diffs.append(fq - fd)
        if len(o["queue_samples"]) >= 2 and np.isfinite(fq) and np.isfinite(lq):
            motions.append(lq - fq)
        if np.isfinite(ft) and np.isfinite(ss):
            first_lags.append(1000.0 * (ft - ss))

    def stats(x):
        a = np.asarray(x, dtype=float)
        if not a.size:
            return {"n": 0, "mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan}
        return {
            "n": int(a.size),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p10": float(np.quantile(a, 0.10)),
            "p90": float(np.quantile(a, 0.90)),
        }

    return {
        "create_orders": len(orders),
        "orders_with_queue_observation": len(with_q),
        "coverage_pct": 100.0 * len(with_q) / len(orders) if orders else np.nan,
        "first_observed_queue_minus_displayed_l1": stats(diffs),
        "last_minus_first_observed_queue": stats(motions),
        "create_send_to_first_queue_observation_ms": stats(first_lags),
    }


class LiveOrderMatcher:
    def __init__(self, orders):
        self.orders = list(orders)
        self.used = set()
        self.attempts = 0
        self.matches = 0
        self.matches_with_queue = 0
        self.join_to_send_ms = []

    def match(self, ticker, desired, join_s):
        self.attempts += 1
        role = str(desired.get("role") or "").upper()
        side = str(desired.get("side") or "").upper()
        price = _price_key(desired.get("price"))
        candidates = []
        for o in self.orders:
            if o["order_id"] in self.used:
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
        self.join_to_send_ms.append(1000.0 * lag)
        return o

    def summary(self):
        a = np.asarray(self.join_to_send_ms, dtype=float)
        return {
            "quote_open_match_attempts": self.attempts,
            "matched_live_orders": self.matches,
            "matched_with_queue_observation": self.matches_with_queue,
            "match_pct": 100.0 * self.matches / self.attempts if self.attempts else np.nan,
            "queue_coverage_pct_of_matches": 100.0 * self.matches_with_queue / self.matches if self.matches else np.nan,
            "join_to_live_send_ms_median": float(np.median(a)) if a.size else np.nan,
            "join_to_live_send_ms_p95": float(np.quantile(a, 0.95)) if a.size else np.nan,
            "join_to_live_send_ms_max": float(np.max(a)) if a.size else np.nan,
        }


class MatchedQueueShadow(BASE.Q5FrozenCycleShadow):
    def __init__(self, workspace, fee, matcher, mode):
        super().__init__(workspace, fee)
        self.matcher = matcher
        self.mode = str(mode)
        self.queue_corrections = 0
        self.queue_forward = 0
        self.queue_backward = 0
        self.queue_same = 0
        self.oracle_seeded = 0

    def emit(self, event, ticker=None, **detail):
        row = {"time": OOS._iso_ts(), "event": event, "ticker": ticker, **detail}
        with self.event_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")

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
            q["matched_live_order_id"] = match["order_id"]
            q["matched_live_send_s"] = match["send_s"]
            if (
                self.mode == "FULL_QUEUE_ORACLE"
                and np.isfinite(OOS._f(match.get("first_queue_position"), np.nan))
            ):
                q["queue_ahead"] = max(0.0, float(match["first_queue_position"]))
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
        if new < old - OOS.EPS:
            self.queue_forward += 1
        elif new > old + OOS.EPS:
            self.queue_backward += 1
        else:
            self.queue_same += 1

    def queue_summary(self):
        return {
            "mode": self.mode,
            "oracle_seeded_quote_opens": self.oracle_seeded,
            "queue_snapshot_corrections_applied": self.queue_corrections,
            "queue_forward_corrections": self.queue_forward,
            "queue_backward_corrections": self.queue_backward,
            "queue_same_corrections": self.queue_same,
            **self.matcher.summary(),
        }


def _next_queue(it, selected_tickers, known_order_ids):
    for row in it:
        ticker = str(row.get("ticker") or "")
        oid = str(row.get("order_id") or "")
        t = _parse_s(row.get("time"))
        q = OOS._f(row.get("queue_position"), np.nan)
        if (
            ticker in selected_tickers
            and oid in known_order_ids
            and np.isfinite(t)
            and np.isfinite(q)
        ):
            return float(t), row
    return None


def _window_pnl(shadow):
    pnl = defaultdict(float)
    for f in shadow.fills:
        ticker = str(f.get("ticker") or "")
        close = str(shadow.close_by_ticker.get(ticker) or "")
        pnl[close] += OOS._f(f.get("matched_pnl_delta"), 0.0)
    for ticker, c in shadow.contracts.items():
        close = str(c.get("close_time") or shadow.close_by_ticker.get(ticker) or "")
        pnl[close] += OOS._f(c.get("forced_liquidation_gross_pnl"), 0.0) - OOS._f(c.get("taker_trade_fee"), 0.0)
    return dict(pnl)


def _run_variant(*, source, raw, workspace, selected_tickers, meta_rows, fee, orders, known_order_ids, mode):
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
    qev = _next_queue(queue_it, selected_tickers, known_order_ids) if mode != "DISPLAYED_L1" else None
    if b is None and tr is None:
        raise RuntimeError("No selected raw events found.")

    first_ts = min(x[0] for x in (b, tr) if x is not None)
    shadow.started_at = pd.Timestamp(first_ts, unit="s", tz="UTC")
    shadow.thread_alive = True

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
            b = BASE._next_selected(book_it, selected_tickers)
        elif typ == "trade":
            t, row = tr
            shadow._on_trade(t, row)
            tr = BASE._next_selected(trade_it, selected_tickers)
        else:
            _, row = qev
            shadow.apply_queue_snapshot(row)
            qev = _next_queue(queue_it, selected_tickers, known_order_ids)
        shadow._update_drawdown()

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
        "queue": shadow.queue_summary(),
        "by_window": _window_pnl(shadow),
    }


def run_q5_matched_queue_shadow(source_session, *, show=True):
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
        raise FileNotFoundError("Missing required artifacts: " + " | ".join(missing))

    cfg = OOS._read_json(source / "process_config.json", {}) or {}
    if str(cfg.get("mode") or "") != "LIVE_Q5_1H" or abs(OOS._f(cfg.get("quote_size"), np.nan) - BASE.QTY) > 1e-9:
        raise RuntimeError("Expected completed LIVE_Q5_1H / Q5 source session.")
    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored fee preflight was not PASS.")

    windows = BASE._live_windows(source)
    meta_rows, meta_by_ticker = BASE._metadata(raw)
    selected_tickers = {
        ticker for ticker, row in meta_by_ticker.items()
        if str(row.get("close_time") or "") in set(windows)
    }
    orders, by_id = _live_order_catalog(source)
    if not orders:
        raise RuntimeError("No live CREATE order catalog could be reconstructed.")

    out = _unique_output(source.name)
    variants = []
    for mode in ("DISPLAYED_L1", "SNAPSHOT_CORRECTED", "FULL_QUEUE_ORACLE"):
        variants.append(
            _run_variant(
                source=source,
                raw=raw,
                workspace=out / mode.lower(),
                selected_tickers=selected_tickers,
                meta_rows=meta_rows,
                fee=fee,
                orders=orders,
                known_order_ids=set(by_id),
                mode=mode,
            )
        )

    live_final = OOS._read_json(source / "final_summary.json", {}) or {}
    live_pnl = OOS._f(live_final.get("account_pnl_usd"), np.nan)
    for r in variants:
        r["live_account_pnl"] = live_pnl
        r["live_minus_shadow"] = live_pnl - r["shadow_net_pnl"] if np.isfinite(live_pnl) else np.nan

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "replay_version": REPLAY_VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "live_windows": windows,
        "selected_tickers": len(selected_tickers),
        "catalog": _catalog_stats(orders),
        "live_account_pnl": live_pnl,
        "variants": variants,
        "same_realization_only": True,
        "independent_validation": False,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "limitation": (
            "Queue positions were periodically polled, not sampled at every CREATE. "
            "SNAPSHOT_CORRECTED uses observations only when they existed. "
            "FULL_QUEUE_ORACLE deliberately uses the first future observation at join."
        ),
    }
    OOS._atomic_json(out / "matched_queue_shadow_summary.json", summary)

    window_rows = []
    for close in windows:
        row = {"close_time": close}
        for r in variants:
            row[r["mode"]] = OOS._f(r["by_window"].get(close), 0.0)
        row["SNAPSHOT_MINUS_BASELINE"] = row["SNAPSHOT_CORRECTED"] - row["DISPLAYED_L1"]
        row["ORACLE_MINUS_BASELINE"] = row["FULL_QUEUE_ORACLE"] - row["DISPLAYED_L1"]
        window_rows.append(row)
    pd.DataFrame(window_rows).to_csv(out / "matched_queue_by_window.csv", index=False)

    pd.DataFrame([
        {
            "mode": r["mode"],
            "shadow_net_pnl": r["shadow_net_pnl"],
            "live_account_pnl": live_pnl,
            "live_minus_shadow": r["live_minus_shadow"],
            "fill_events": r["fill_events"],
            "fill_qty": r["fill_qty"],
            "forced_liq_qty": r["forced_liq_qty"],
            "max_drawdown": r["max_drawdown"],
            **{f"queue_{k}": v for k, v in r["queue"].items() if k != "mode"},
        }
        for r in variants
    ]).to_csv(out / "matched_queue_variants.csv", index=False)

    if show:
        print("=" * 118)
        print("Q5 MATCHED LIVE-QUEUE SHADOW — SAME REALIZATION / READ ONLY")
        print("=" * 118)
        print("Source:", source)
        print("Live PnL:", f"${live_pnl:+.4f}")
        print()
        c = summary["catalog"]
        print("LIVE QUEUE OBSERVATION COVERAGE")
        print("  CREATE orders:", c["create_orders"])
        print("  with >=1 queue observation:", c["orders_with_queue_observation"])
        print("  coverage %:", c["coverage_pct"])
        print("  first observed queue - displayed L1:", c["first_observed_queue_minus_displayed_l1"])
        print("  last - first observed queue:", c["last_minus_first_observed_queue"])
        print("  CREATE send -> first queue observation ms:", c["create_send_to_first_queue_observation_ms"])
        print()
        for r in variants:
            q = r["queue"]
            print(
                f"{r['mode']:<22} shadow=${r['shadow_net_pnl']:+.4f}  "
                f"live-shadow=${r['live_minus_shadow']:+.4f}  "
                f"fills={r['fill_events']} qty={r['fill_qty']:.2f}"
            )
            print(
                "  matched/attempted/with-queue:",
                q["matched_live_orders"], "/", q["quote_open_match_attempts"], "/", q["matched_with_queue_observation"],
            )
            print(
                "  queue corrections forward/back/same:",
                q["queue_forward_corrections"], "/", q["queue_backward_corrections"], "/", q["queue_same_corrections"],
            )
            print(
                "  join->live-send ms median/p95/max:",
                q["join_to_live_send_ms_median"], q["join_to_live_send_ms_p95"], q["join_to_live_send_ms_max"],
            )
            print()
        print("BY WINDOW")
        print(pd.DataFrame(window_rows).to_string(index=False))
        print()
        print("Interpretation:")
        print("  DISPLAYED_L1 should reproduce the prior frozen Q5 shadow.")
        print("  SNAPSHOT_CORRECTED tests observed queue movement without future queue look-ahead.")
        print("  FULL_QUEUE_ORACLE also seeds the first observed queue at join; it is deliberately non-causal.")
        print("  If queue-aware variants move materially toward live -$6.69, queue mechanics are a major part of the gap.")
        print("  Low queue/match coverage means this run cannot answer the queue hypothesis completely.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 118)

    return summary


__all__ = ["run_q5_matched_queue_shadow"]
