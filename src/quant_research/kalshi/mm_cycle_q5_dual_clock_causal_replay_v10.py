from __future__ import annotations

"""Dual-clock causal same-realization replay for the completed Q5 live session.

Why this exists
---------------
The frozen receipt-time shadow uses local receipt timestamps for both strategy
book decisions and passive trade execution.  Q5 cross-feed forensics showed that
this can let a later local book receipt remove a hypothetical quote before the
public trade receipt arrives even though the corresponding trade had already
executed at the exchange.

This replay separates those clocks without changing Candidate-C:

1. BOOK / strategy decisions use local ``receipt_time``.  This is when public
   information became actionable to the strategy.
2. PASSIVE execution uses public trade ``exchange_time`` when available.  An
   execution is economically irreversible at that time.
3. FILL OBSERVATION uses that trade's local ``receipt_time``.  Until observation,
   the strategy's known inventory can differ from economic inventory.

The simulator therefore keeps separate:
- ``inventory``: economically executed inventory;
- ``known_inventory``: inventory revealed to the strategy by received public
  trade observations;
- ``exchange_quote``: the hypothetical order economically resting at the venue;
- ``quote``: the order the strategy locally believes is working.

Frozen mechanics retained
-------------------------
- exact Candidate-C entry rule;
- Q5 quote size;
- same public BBO prices;
- after an observed entry fill, stop adding and quote only the opposite BBO;
- displayed-L1 FIFO queue model;
- exact-price aggressive flow burns queue first;
- trade-through fills;
- no cancellation-ahead credit;
- any passive fill removes the economically resting residual;
- M5 crosses actual economic residual at executable BBO with the stored fee model.

Important model limitations
---------------------------
- This is SAME-REALIZATION forensic evidence, not independent validation.
- Simulated CREATE/CANCEL is assumed effective at the local book receipt timestamp;
  real exchange command latency is not modeled here.  That makes this an execution
  causality correction, not a production latency simulator.
- Public trade receipt is used as the fill-observation proxy.  A private fill feed
  could be earlier or later in production.
- The frozen conservative FIFO model remains unchanged; cancellation-ahead is not
  added in this test.
- Exact timestamp ties are resolved execution-first, preventing a same-time local
  book receipt from retroactively escaping an exchange execution.
- NO exchange/API calls and NO orders. Source session is read-only.
"""

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_cycle_q5_same_realization_shadow_v1 as BASE

VERSION = "MM_CYCLE_Q5_DUAL_CLOCK_CAUSAL_REPLAY_V10"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_q5_dual_clock_causal_replay_v10"
QTY = BASE.QTY
EPS = 1e-10


def _f(x, default=np.nan):
    return OOS._f(x, default)


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


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _exchange_s(row):
    t = OOS._ts((row or {}).get("exchange_time"))
    if np.isfinite(t):
        return float(t)
    z = _f((row or {}).get("ts_ms"))
    if np.isfinite(z):
        return float(z) / 1000.0
    return np.nan


def _receipt_s(row):
    t = OOS._ts((row or {}).get("receipt_time"))
    return float(t) if np.isfinite(t) else np.nan


def _window_start_s(meta):
    close = OOS._ts((meta or {}).get("close_time"))
    return float(close - 900.0) if np.isfinite(close) else np.nan


def _trade_elapsed_at_execution(row, meta, exec_s):
    ws = _window_start_s(meta)
    if np.isfinite(ws) and np.isfinite(exec_s):
        return float(exec_s - ws)
    return _f((row or {}).get("elapsed_s"))


def _find_baseline(source: Path):
    root = Path(BASE.OUTPUT_ROOT)
    candidates = []
    if root.exists():
        for d in root.glob(source.name + "*"):
            sp = d / "q5_same_realization_shadow_summary.json"
            if not sp.exists():
                continue
            obj = OOS._read_json(sp, {}) or {}
            try:
                same = Path(obj.get("source_session", "")).resolve() == source.resolve()
            except Exception:
                same = False
            if same:
                candidates.append((sp.stat().st_mtime, d.resolve(), obj))
    if not candidates:
        return None, None
    _, d, obj = max(candidates, key=lambda z: z[0])
    return d, obj


class QuietDualClockQ5Shadow(BASE.Q5FrozenCycleShadow):
    """Frozen Candidate-C with exchange execution and receipt-time observation."""

    def __init__(self, workspace, fee_preflight_result):
        super().__init__(workspace, fee_preflight_result)
        self.known_inventory = defaultdict(float)
        self.exchange_quote = {}
        self.order_seq = 0
        self.pending_observations_by_trade = defaultdict(list)
        self.pending_hidden_qty = defaultdict(float)
        self.pending_hidden_fills = defaultdict(int)
        self.dual = Counter()
        self.dual_qty = defaultdict(float)
        self.execution_receipt_lags_ms = []

    def emit(self, event, ticker=None, **detail):
        # Base emission is observational only and extremely noisy in a forensic.
        return None

    def _desired_quote(self, ticker, cur, elapsed):
        if cur is None or not (0.0 <= elapsed < 300.0) or ticker in self.finalized:
            return None
        inv = float(self.known_inventory[ticker])
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

    def _cancel_local_quote(self, ticker, *, reason):
        old = self.quote.get(ticker)
        if old is None:
            return
        oid = old.get("sim_order_id")
        ex = self.exchange_quote.get(ticker)
        if ex is not None and ex.get("sim_order_id") == oid:
            self.exchange_quote.pop(ticker, None)
            self.dual["exchange_cancels_effective"] += 1
        else:
            # Local cancellation after the order already executed economically.
            self.dual["cancel_after_hidden_execution"] += 1
            self.dual_qty["cancel_after_hidden_execution_pending_qty"] += float(
                self.pending_hidden_qty.get(ticker, 0.0)
            )
        self.quote.pop(ticker, None)
        self.c["quote_cancels"] += 1
        self.dual[f"cancel_reason_{reason}"] += 1

    def _reconcile_quote(self, ticker, cur, elapsed, t):
        desired = self._desired_quote(ticker, cur, elapsed)
        old = self.quote.get(ticker)
        if self._quote_same(old, desired):
            return

        if old is not None:
            self._cancel_local_quote(ticker, reason="BOOK_RECONCILE")

        if desired is None:
            return

        self.order_seq += 1
        q = dict(desired)
        q.update(
            {
                "join_ts": float(t),
                "queue_ahead_initial": float(desired["queue_ahead"]),
                "remaining_qty": float(desired["qty"]),
                "sim_order_id": int(self.order_seq),
            }
        )
        self.quote[ticker] = dict(q)
        self.exchange_quote[ticker] = dict(q)
        self.c[f"{q['role']}_quote_opens"] += 1
        self.dual["sim_creates"] += 1
        if self.pending_hidden_fills.get(ticker, 0) > 0:
            self.dual["creates_while_fill_hidden"] += 1
            self.dual_qty["creates_while_fill_hidden_submitted_qty"] += float(q["qty"])

    def on_book_receipt(self, t, row):
        ticker = str(row.get("ticker") or "")
        cur = OOS._top_state(row)
        elapsed = _f(row.get("elapsed_s"))
        if cur is not None:
            self.current[ticker] = cur
            self._resolve_markouts(t, cur["mid"])

        typ = str(row.get("event_type") or "")
        if typ == "trade_window_end" or (
            np.isfinite(elapsed) and elapsed >= 300.0 and ticker not in self.finalized
        ):
            if cur is not None:
                self.current[ticker] = cur
            self._finalize_m5_dual(ticker, t)
            return

        if not np.isfinite(elapsed):
            return

        if cur is None:
            if 0.0 <= elapsed < 300.0:
                if self.quote.get(ticker) is not None:
                    self._cancel_local_quote(ticker, reason="INVALID_BOOK")
            return

        self._reconcile_quote(ticker, cur, float(elapsed), t)

    def _economic_fill(self, ticker, q, side, qty, qpx, trade_row, exec_s, receipt_s, trade_through):
        role = str(q["role"])
        sign = 1.0 if side == "BID" else -1.0
        inv_before = float(self.inventory[ticker])
        self.inventory[ticker] += sign * qty
        if abs(self.inventory[ticker]) < 1e-9:
            self.inventory[ticker] = 0.0
        self.max_abs_inventory = max(self.max_abs_inventory, abs(self.inventory[ticker]))
        matched_delta = self._passive_match(ticker, side, qty, qpx)

        cur = self.current.get(ticker)
        fill = {
            "time": OOS._iso_ts(exec_s),
            "fill_ts": float(exec_s),
            "observation_ts": float(receipt_s),
            "ticker": ticker,
            "series": self.series_by_ticker.get(ticker, str(trade_row.get("series_ticker") or "")),
            "role": role,
            "side": side,
            "qty": float(qty),
            "price": float(qpx),
            "trade_through": bool(trade_through),
            "historical_trade_id": str(trade_row.get("trade_id") or ""),
            "historical_trade_price": _f(trade_row.get("yes_price")),
            "historical_trade_qty": _f(trade_row.get("qty")),
            "queue_ahead_initial": float(q.get("queue_ahead_initial", np.nan)),
            "queue_ahead_before_fill": float(q.get("queue_ahead", np.nan)),
            "sim_order_id": int(q.get("sim_order_id")),
            "inventory_before": inv_before,
            "inventory_after": float(self.inventory[ticker]),
            "known_inventory_at_execution": float(self.known_inventory[ticker]),
            "matched_pnl_delta": matched_delta,
            "mid_at_fill": cur["mid"] if cur else np.nan,
            "markout_5s_c": np.nan,
            "markout_15s_c": np.nan,
            "markout_30s_c": np.nan,
        }
        self.fills.append(fill)
        idx = len(self.fills) - 1
        for h in OOS.MARKOUTS_S:
            self.markout_counter += 1
            heapq.heappush(
                self.pending_markouts,
                (exec_s + h, self.markout_counter, idx, h, sign),
            )
        self._write_fill(fill)

        self.c["fill_events"] += 1
        self.c["fill_qty_x1000"] += int(round(qty * 1000.0))
        self.c[f"{role}_fill_events"] += 1
        if trade_through:
            self.c["trade_through_fills"] += 1
        if role == "ENTRY" and abs(inv_before) <= OOS.EPS:
            self.c["cycles_started"] += 1
        if role == "EXIT" and abs(self.inventory[ticker]) <= OOS.EPS:
            self.c["cycles_completed"] += 1

        obs = {
            "ticker": ticker,
            "role": role,
            "side": side,
            "qty": float(qty),
            "price": float(qpx),
            "sign": sign,
            "sim_order_id": int(q.get("sim_order_id")),
            "execution_s": float(exec_s),
            "observation_s": float(receipt_s),
            "fill_index": idx,
        }
        self.pending_observations_by_trade[id(trade_row)].append(obs)
        self.pending_hidden_fills[ticker] += 1
        self.pending_hidden_qty[ticker] += float(qty)
        self.dual["economic_fill_events"] += 1
        self.dual_qty["economic_fill_qty"] += float(qty)

    def on_trade_execution(self, exec_s, receipt_s, row):
        ticker = str(row.get("ticker") or "")
        if ticker in self.finalized:
            return
        elapsed = _trade_elapsed_at_execution(row, self.meta.get(ticker) or {}, exec_s)
        if not np.isfinite(elapsed) or not (0.0 <= elapsed < 300.0):
            return

        q = self.exchange_quote.get(ticker)
        if q is None:
            return

        trade_px = _f(row.get("yes_price"))
        trade_qty = _f(row.get("qty"))
        taker = str(row.get("taker_book_side") or "").lower()
        if not (np.isfinite(trade_px) and np.isfinite(trade_qty) and trade_qty > 0):
            return

        side = "BID" if taker == "ask" else "ASK" if taker == "bid" else None
        if side != q["side"]:
            return

        qpx = float(q["price"])
        trade_through = False
        available = 0.0
        if side == "BID":
            if trade_px < qpx - OOS.EPS:
                trade_through = True
                available = float(q["remaining_qty"])
            elif abs(trade_px - qpx) <= OOS.EPS:
                burn = min(float(q["queue_ahead"]), trade_qty)
                q["queue_ahead"] -= burn
                available = max(0.0, trade_qty - burn)
            else:
                return
        else:
            if trade_px > qpx + OOS.EPS:
                trade_through = True
                available = float(q["remaining_qty"])
            elif abs(trade_px - qpx) <= OOS.EPS:
                burn = min(float(q["queue_ahead"]), trade_qty)
                q["queue_ahead"] -= burn
                available = max(0.0, trade_qty - burn)
            else:
                return

        # Preserve the frozen queue state economically.
        self.exchange_quote[ticker] = q
        if available <= OOS.EPS:
            return

        qty = min(float(q["remaining_qty"]), available)
        if qty <= OOS.EPS:
            return

        self._economic_fill(
            ticker,
            q,
            side,
            qty,
            qpx,
            row,
            exec_s,
            receipt_s,
            trade_through,
        )

        # Frozen rule: any passive fill removes residual economically.
        self.exchange_quote.pop(ticker, None)

    def on_trade_observation(self, receipt_s, row):
        observations = self.pending_observations_by_trade.pop(id(row), [])
        if not observations:
            return

        for obs in observations:
            ticker = str(obs["ticker"])
            self.pending_hidden_fills[ticker] = max(
                0, int(self.pending_hidden_fills.get(ticker, 0)) - 1
            )
            self.pending_hidden_qty[ticker] = max(
                0.0, float(self.pending_hidden_qty.get(ticker, 0.0)) - float(obs["qty"])
            )

            if ticker in self.finalized:
                self.dual["fill_observed_after_m5_finalize"] += 1
                continue

            known_before = float(self.known_inventory[ticker])
            self.known_inventory[ticker] += float(obs["sign"]) * float(obs["qty"])
            if abs(self.known_inventory[ticker]) < 1e-9:
                self.known_inventory[ticker] = 0.0

            idx = int(obs["fill_index"])
            if 0 <= idx < len(self.fills):
                self.fills[idx]["known_inventory_before_observation"] = known_before
                self.fills[idx]["known_inventory_after_observation"] = float(
                    self.known_inventory[ticker]
                )
                self.fills[idx]["observation_delay_ms"] = 1000.0 * (
                    float(receipt_s) - float(obs["execution_s"])
                )

            self.dual["fill_observations"] += 1
            self.dual_qty["fill_observed_qty"] += float(obs["qty"])

            # Preserve frozen 'any fill cancels residual, re-evaluate on next
            # book event'.  If a newer quote was created while this fill was
            # hidden, cancel it as soon as inventory becomes known.
            if self.quote.get(ticker) is not None:
                current_oid = self.quote[ticker].get("sim_order_id")
                if current_oid != obs["sim_order_id"]:
                    self.dual["newer_quote_cancelled_on_late_fill_observation"] += 1
                self._cancel_local_quote(ticker, reason="FILL_OBSERVED")

    def _finalize_m5_dual(self, ticker, t):
        # Cancel any locally believed / economically resting quote before crossing.
        if self.quote.get(ticker) is not None:
            self._cancel_local_quote(ticker, reason="M5")
        self.exchange_quote.pop(ticker, None)
        super()._finalize_m5(ticker, t)
        self.known_inventory[ticker] = 0.0


def _build_events(raw: Path, selected_tickers: set[str]):
    events = []
    stats = Counter()
    lag_ms = []

    # Priority: exchange execution (0), local book receipt (1), local trade
    # observation (2).  This intentionally resolves exact timestamp ties against
    # retroactive fill escape.
    for i, row in enumerate(_iter_jsonl(raw / "book_top3_events.jsonl") or []):
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        rt = _receipt_s(row)
        if not np.isfinite(rt):
            continue
        events.append((float(rt), 1, i, "BOOK", row, float(rt)))
        stats["book_receipts"] += 1

    trade_base = 10_000_000
    for i, row in enumerate(_iter_jsonl(raw / "trades_event_time.jsonl") or []):
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        rt = _receipt_s(row)
        if not np.isfinite(rt):
            continue
        xt = _exchange_s(row)
        if np.isfinite(xt):
            exec_t = float(xt)
            stats["trades_with_exchange_time"] += 1
            lag = 1000.0 * (float(rt) - exec_t)
            lag_ms.append(lag)
            if exec_t <= float(rt) + EPS:
                stats["exchange_at_or_before_receipt"] += 1
            else:
                stats["exchange_after_receipt"] += 1
        else:
            exec_t = float(rt)
            stats["trades_missing_exchange_time"] += 1

        key = trade_base + i
        events.append((float(exec_t), 0, key, "TRADE_EXEC", row, float(rt)))
        events.append((float(rt), 2, key, "TRADE_OBS", row, float(rt)))
        stats["trade_rows"] += 1

    events.sort(key=lambda z: (z[0], z[1], z[2]))
    return events, stats, lag_ms


def _economics(shadow: QuietDualClockQ5Shadow):
    agg = defaultdict(
        lambda: {
            "passive_matched_pnl": 0.0,
            "forced_liq_gross_pnl": 0.0,
            "taker_trade_fees": 0.0,
            "fill_events": 0,
            "fill_qty": 0.0,
            "forced_liq_qty": 0.0,
        }
    )

    for fill in shadow.fills:
        ticker = str(fill.get("ticker") or "")
        close = str(shadow.close_by_ticker.get(ticker) or "")
        series = str(shadow.series_by_ticker.get(ticker) or "")
        a = agg[(close, series, ticker)]
        a["passive_matched_pnl"] += _f(fill.get("matched_pnl_delta"), 0.0)
        a["fill_events"] += 1
        a["fill_qty"] += _f(fill.get("qty"), 0.0)

    for ticker, c in shadow.contracts.items():
        close = str(c.get("close_time") or shadow.close_by_ticker.get(ticker) or "")
        series = str(c.get("series") or shadow.series_by_ticker.get(ticker) or "")
        a = agg[(close, series, ticker)]
        a["forced_liq_gross_pnl"] += _f(c.get("forced_liquidation_gross_pnl"), 0.0)
        a["taker_trade_fees"] += _f(c.get("taker_trade_fee"), 0.0)
        a["forced_liq_qty"] += _f(c.get("forced_liquidation_qty"), 0.0)

    rows = []
    for (close, series, ticker), a in agg.items():
        rows.append(
            {
                "close_time": close,
                "series": series,
                "ticker": ticker,
                **a,
                "net_pnl": a["passive_matched_pnl"]
                + a["forced_liq_gross_pnl"]
                - a["taker_trade_fees"],
            }
        )

    detail = pd.DataFrame(rows)
    cols = [
        "close_time",
        "series",
        "ticker",
        "passive_matched_pnl",
        "forced_liq_gross_pnl",
        "taker_trade_fees",
        "fill_events",
        "fill_qty",
        "forced_liq_qty",
        "net_pnl",
    ]
    if detail.empty:
        detail = pd.DataFrame(columns=cols)

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
    return detail, by_window, by_asset


def run_q5_dual_clock_causal_replay(source_session, *, show=True):
    source = Path(source_session).resolve()
    raw = source / "raw_capture"
    required = [
        source / "process_config.json",
        source / "fee_preflight.json",
        source / "events.jsonl",
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
        raise RuntimeError(f"Expected LIVE_Q5_1H session, got {cfg.get('mode')!r}")
    actual_q = _f(cfg.get("quote_size"))
    if not np.isfinite(actual_q) or abs(actual_q - QTY) > 1e-9:
        raise RuntimeError(f"Expected Q5 live quote size, got {actual_q}")

    fee = OOS._read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored fee preflight was not PASS.")

    windows = BASE._live_windows(source)
    meta_rows, meta_by_ticker = BASE._metadata(raw)
    selected_tickers = {
        t
        for t, r in meta_by_ticker.items()
        if str(r.get("close_time") or "") in set(windows)
    }
    if not selected_tickers:
        raise RuntimeError("No raw tickers matched the live Q5 windows.")

    out = _new_output(source.name)
    workspace = out / "dual_clock_workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    shadow = QuietDualClockQ5Shadow(workspace, fee)

    for row in meta_rows:
        ticker = str(row.get("ticker") or "")
        if ticker not in selected_tickers:
            continue
        shadow.meta[ticker] = row
        shadow.series_by_ticker[ticker] = str(row.get("series_ticker") or "")
        shadow.close_by_ticker[ticker] = str(row.get("close_time") or "")

    events, event_stats, receipt_lags_ms = _build_events(raw, selected_tickers)
    if not events:
        raise RuntimeError("No selected dual-clock events found.")

    shadow.started_at = pd.Timestamp(events[0][0], unit="s", tz="UTC")
    shadow.thread_alive = True

    for n, (t, _priority, _seq, typ, row, receipt_t) in enumerate(events, start=1):
        if typ == "BOOK":
            shadow.on_book_receipt(t, row)
        elif typ == "TRADE_EXEC":
            shadow.on_trade_execution(t, receipt_t, row)
        else:
            shadow.on_trade_observation(t, row)
        shadow._update_drawdown()
        if show and n % 500_000 == 0:
            print(
                f"processed {n:,}/{len(events):,} dual-clock events | "
                f"economic fills={int(shadow.c['fill_events']):,} "
                f"observed fills={int(shadow.dual['fill_observations']):,}"
            )

    # Some markets may not emit a trade_window_end row in the selected stream.
    # Match the baseline behavior as closely as possible by finalizing only those
    # already finalized by M5 book events; report any remainder fail-closed.
    unfinalized = sorted(
        t for t in selected_tickers if t not in shadow.finalized
    )

    shadow.thread_alive = False

    detail, by_window, by_asset = _economics(shadow)
    fills = pd.DataFrame(shadow.fills)

    passive = float(shadow.passive_matched_pnl)
    forced = float(shadow.forced_liq_gross_pnl)
    fees = float(shadow.taker_trade_fees)
    causal_net = passive + forced - fees

    baseline_dir, baseline = _find_baseline(source)
    baseline_net = _f((baseline or {}).get("shadow_net_pnl"))
    baseline_passive = _f((baseline or {}).get("shadow_passive_matched_pnl"))
    baseline_forced = _f((baseline or {}).get("shadow_forced_liq_gross_pnl"))
    baseline_fees = _f((baseline or {}).get("shadow_taker_trade_fees"))

    live_final = OOS._read_json(source / "final_summary.json", {}) or {}
    live_pnl = _f(live_final.get("account_pnl_usd"))

    lags = np.asarray([x for x in receipt_lags_ms if np.isfinite(x)], dtype=float)
    obs_delays = (
        pd.to_numeric(fills.get("observation_delay_ms"), errors="coerce").dropna()
        if not fills.empty and "observation_delay_ms" in fills
        else pd.Series(dtype=float)
    )

    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "source_session": str(source),
        "output_dir": str(out),
        "baseline_source": str(baseline_dir) if baseline_dir else None,
        "same_realization_only": True,
        "independent_validation": False,
        "quote_size": QTY,
        "live_windows": windows,
        "selected_tickers": len(selected_tickers),
        "dual_clock_event_count": len(events),
        "event_stats": dict(event_stats),
        "trade_exchange_to_receipt_ms_median": float(np.median(lags)) if len(lags) else np.nan,
        "trade_exchange_to_receipt_ms_p95": float(np.percentile(lags, 95)) if len(lags) else np.nan,
        "economic_fill_observation_delay_ms_median": float(obs_delays.median()) if len(obs_delays) else np.nan,
        "causal_passive_matched_pnl": passive,
        "causal_forced_liq_gross_pnl": forced,
        "causal_taker_trade_fees": fees,
        "causal_net_pnl": causal_net,
        "causal_fill_events": int(shadow.c["fill_events"]),
        "causal_fill_qty": shadow.c["fill_qty_x1000"] / 1000.0,
        "causal_cycles_started": int(shadow.c["cycles_started"]),
        "causal_cycles_completed": int(shadow.c["cycles_completed"]),
        "causal_forced_liquidations": int(shadow.c["forced_liquidations"]),
        "causal_forced_liq_qty": shadow.c["forced_liq_qty_x1000"] / 1000.0,
        "causal_max_drawdown_online": float(shadow.max_drawdown),
        "baseline_receipt_shadow_net_pnl": baseline_net,
        "baseline_receipt_shadow_passive_pnl": baseline_passive,
        "baseline_receipt_shadow_forced_gross": baseline_forced,
        "baseline_receipt_shadow_fees": baseline_fees,
        "causal_minus_baseline_net_pnl": (
            causal_net - baseline_net if np.isfinite(baseline_net) else np.nan
        ),
        "live_account_pnl": live_pnl,
        "live_minus_causal_pnl": (
            live_pnl - causal_net if np.isfinite(live_pnl) else np.nan
        ),
        "cancel_after_hidden_execution": int(shadow.dual["cancel_after_hidden_execution"]),
        "creates_while_fill_hidden": int(shadow.dual["creates_while_fill_hidden"]),
        "newer_quote_cancelled_on_late_fill_observation": int(
            shadow.dual["newer_quote_cancelled_on_late_fill_observation"]
        ),
        "fill_observations": int(shadow.dual["fill_observations"]),
        "fill_observed_after_m5_finalize": int(shadow.dual["fill_observed_after_m5_finalize"]),
        "unfinalized_tickers": unfinalized,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "model_limitations": {
            "create_cancel_effective_time": "local book receipt timestamp; zero command latency",
            "fill_execution_time": "public trade exchange_time when available, receipt_time fallback",
            "fill_observation_time": "public trade receipt_time proxy",
            "queue_model": "frozen displayed-L1 FIFO; no cancellation-ahead credit",
            "tie_policy": "trade execution before local book receipt before trade observation",
        },
        "interpretation_guardrail": (
            "This same-realization dual-clock replay tests whether the frozen strategy survives removal of receipt-order fill escape. "
            "It is not independent profitability validation and it does not model real CREATE/CANCEL network latency."
        ),
    }

    OOS._atomic_json(out / "summary.json", summary)
    detail.to_csv(out / "causal_by_contract.csv", index=False)
    by_window.to_csv(out / "causal_by_window.csv", index=False)
    by_asset.to_csv(out / "causal_by_asset.csv", index=False)
    fills.to_csv(out / "causal_passive_fills.csv", index=False)

    if show:
        print("=" * 142)
        print("Q5 DUAL-CLOCK CAUSAL REPLAY V10 — SAME REALIZATION / READ ONLY")
        print("=" * 142)
        print("Source:", source)
        print("Baseline receipt-time shadow:", baseline_dir)
        print("Windows:", len(windows), "| tickers:", len(selected_tickers))
        print("Events processed:", f"{len(events):,}")
        print("Trade exchange-time coverage:", event_stats.get("trades_with_exchange_time", 0), "/", event_stats.get("trade_rows", 0))
        print("Median trade exchange->receipt ms:", f"{summary['trade_exchange_to_receipt_ms_median']:.3f}")
        print("Median executed-fill observation delay ms:", f"{summary['economic_fill_observation_delay_ms_median']:.3f}")
        print()
        print("CORE ECONOMICS")
        print(f"  baseline receipt-time shadow net: {baseline_net:+.4f}" if np.isfinite(baseline_net) else "  baseline receipt-time shadow net: unavailable")
        print(f"  causal dual-clock passive PnL:    {passive:+.4f}")
        print(f"  causal dual-clock M5 gross:       {forced:+.4f}")
        print(f"  causal dual-clock taker fees:     {fees:.4f}")
        print(f"  CAUSAL DUAL-CLOCK NET:            {causal_net:+.4f}")
        if np.isfinite(baseline_net):
            print(f"  causal - baseline:                {causal_net - baseline_net:+.4f}")
        if np.isfinite(live_pnl):
            print(f"  live account PnL:                 {live_pnl:+.4f}")
            print(f"  live - causal replay:             {live_pnl - causal_net:+.4f}")
        print()
        print("CAUSALITY / HIDDEN-FILL DIAGNOSTICS")
        print("  passive fill events / qty:", int(shadow.c["fill_events"]), "/", f"{summary['causal_fill_qty']:.4f}")
        print("  cycles started / completed:", int(shadow.c["cycles_started"]), "/", int(shadow.c["cycles_completed"]))
        print("  cancel after hidden execution:", int(shadow.dual["cancel_after_hidden_execution"]))
        print("  creates while a fill was hidden:", int(shadow.dual["creates_while_fill_hidden"]))
        print("  newer quotes cancelled on late fill observation:", int(shadow.dual["newer_quote_cancelled_on_late_fill_observation"]))
        print("  M5 fill observations arriving after finalize:", int(shadow.dual["fill_observed_after_m5_finalize"]))
        print("  unfinalized tickers:", unfinalized)
        print()
        print("BY WINDOW")
        print(by_window.to_string(index=False) if not by_window.empty else "  none")
        print()
        print("BY ASSET")
        print(by_asset.to_string(index=False) if not by_asset.empty else "  none")
        print()
        print("Interpretation guardrails:")
        print("  - Candidate-C and frozen FIFO queue rules are unchanged.")
        print("  - Only execution/observation causality is changed.")
        print("  - CREATE/CANCEL network latency is still idealized to zero at local receipt time.")
        print("  - Same realization only: this does NOT establish future profitability.")
        print("Output:", out)
        print("SOURCE MODIFIED: NO | EXCHANGE API CALLED: NO | ORDERS SENT: NO")
        print("=" * 142)

    return {
        "summary": summary,
        "detail": detail,
        "by_window": by_window,
        "by_asset": by_asset,
        "fills": fills,
        "output_dir": out,
    }


__all__ = ["run_q5_dual_clock_causal_replay", "VERSION"]
