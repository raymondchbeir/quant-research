from __future__ import annotations

"""Fast DEVELOPMENT-only capacity study for the M5->M12 5c deep-tail hypothesis.

NO API CALLS. NO ORDERS. SOURCE FILES READ ONLY.

The exploratory full-window notebook established that the 5c strict-through ->
fixed JOIN_ASK phenomenon persists after M5, but at much higher frequency and
much thinner edge per fill. This module freezes the primary late-window
hypothesis and varies only requested quantity.

Frozen mechanics
----------------
- M5 + 100ms: rest BUY YES @ 5c and BUY NO @ 5c.
- First observed strict-through fill chooses the tail; opposite tail is cancelled.
- Confirmed entry capacity is cumulative same-tail aggressive-seller quantity
  trading STRICTLY THROUGH 5c; exact 5c prints are excluded.
- After FULL requested Q is locally observable: fixed JOIN_ASK +100ms, no reprice.
- Exact-price aggressive buyer flow burns displayed L1 queue ahead first.
- Trade-through beyond the fixed sell quote fills all remaining quantity.
- At M12 any residual crosses recorded top-3 outcome bids.
- Quadratic taker fee is charged per consumed M12 level.
- Residual beyond top-3 is valued at zero.
- Add the same $0.0099 conservative balance-rounding drag per nonzero marketable
  boundary exit used by earlier capacity studies.

Quantity grid: Q1, Q5, Q10, Q20, Q30, Q50, Q100.

Performance design
------------------
The prior notebook spent most of its time fully JSON-decoding ~28.5M book rows.
This module does not. The hot book loop extracts only ticker, elapsed_s and
valid_bbo with bytes.find(). Full JSON decoding is deferred until the end and is
performed only for the small set of decision/M12 rows actually needed.

No third-party package is installed. If orjson is already present it is used;
otherwise stdlib json is used. If torch+MPS are already present, MPS is used for
cumulative-PnL/drawdown aggregation only. JSON parsing and file I/O are CPU tasks
and are not GPU workloads.

Scientific status: DEVELOPMENT / EXPLORATORY CAPACITY ONLY. Any late-window
candidate selected here needs separate fresh forward validation.
"""

import argparse
import json
import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_cycle_q10_oos_stack_v1 as OOS
from . import mm_deep_tail_capacity_dev_v3 as V3

VERSION = "MM_DEEP_TAIL_M5_M12_CAPACITY_FAST_V1"
FULL15_ROOT = C.DATA_ROOT / "mm_event_m0_m15_exploratory_v1"
SESSION_NAMES = ("20260817_064152", "20260817_005114")
FEE_SESSION = C.DATA_ROOT / "mm_event_m0_m5_oos_cycle_q10_v1" / "20260817_064143"
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_m5_m12_capacity_fast_v1"

ENTRY = 0.05
ENTRY_C = 5
START_E = 300.0
END_E = 720.0
WINDOW_S = END_E - START_E
ENTRY_ACTIVATION_S = 0.100
EXIT_ACTIVATION_S = 0.100
CANCEL_RACE_S = 0.100
MIN_BOOK_COVERAGE = 0.95
Q_GRID = (1, 5, 10, 20, 30, 50, 100)
ROUNDING_DRAG = V3.BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS
EPS = 1e-10

TICKER_KEY = b'"ticker":"'
ELAPSED_KEY = b'"elapsed_s":'
VALID_KEY = b'"valid_bbo":'

try:
    import orjson as _orjson

    def _loads(raw):
        return _orjson.loads(raw)

    JSON_ENGINE = "orjson (already installed)"
except Exception:
    def _loads(raw):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    JSON_ENGINE = "stdlib json"

try:
    import torch as _torch
    MPS_AVAILABLE = bool(
        hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available()
    )
except Exception:
    _torch = None
    MPS_AVAILABLE = False

AGG_DEVICE = "mps" if MPS_AVAILABLE else "cpu"


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _iter_jsonl(path: Path):
    with Path(path).open("rb") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                row = _loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _fast_float(raw: bytes, key: bytes):
    i = raw.find(key)
    if i < 0:
        return None
    i += len(key)
    j = i
    n = len(raw)
    while j < n and raw[j] not in b",}":
        j += 1
    try:
        return float(raw[i:j])
    except Exception:
        return None


def _fast_ticker(raw: bytes):
    i = raw.find(TICKER_KEY)
    if i < 0:
        return None
    i += len(TICKER_KEY)
    j = raw.find(b'"', i)
    if j < 0:
        return None
    try:
        return raw[i:j].decode("ascii")
    except Exception:
        return raw[i:j].decode("utf-8", errors="ignore")


def _fast_valid(raw: bytes):
    i = raw.find(VALID_KEY)
    if i < 0:
        return False
    i += len(VALID_KEY)
    return raw[i:i + 4] == b"true"


def _iso_seconds(x):
    if x in (None, ""):
        return np.nan
    try:
        s = str(x)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return float(datetime.fromisoformat(s).timestamp())
    except Exception:
        try:
            z = pd.to_datetime(x, utc=True, errors="coerce")
            return np.nan if pd.isna(z) else float(z.timestamp())
        except Exception:
            return np.nan


def _fingerprint(path: Path):
    st = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _cache_ok(path: Path, expected):
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return obj == expected


def _metadata(session: Path):
    out = {}
    for r in _iter_jsonl(session / "market_metadata.jsonl"):
        ticker = str(r.get("ticker") or "")
        series = str(r.get("series_ticker") or "")
        close = pd.to_datetime(r.get("close_time"), utc=True, errors="coerce")
        if not ticker or pd.isna(close):
            continue
        start = close - pd.Timedelta(minutes=15)
        out[ticker] = {
            "ticker": ticker,
            "series": series,
            "close_time": close.isoformat(),
            "close_s": float(close.timestamp()),
            "window_start_s": float(start.timestamp()),
        }
    return out


def _fee_multipliers():
    p = FEE_SESSION / "fee_preflight.json"
    if not p.exists():
        raise FileNotFoundError(f"Stored fee preflight missing: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("ok") is not True:
        raise RuntimeError("Stored historical fee preflight is not PASS")
    mult = {str(k): float(v) for k, v in (obj.get("multipliers") or {}).items()}
    return mult, str(p)


def _trade_fields(row, meta_row):
    e = _f(row.get("elapsed_s"))
    yes = _f(row.get("yes_price"))
    qty = _f(row.get("qty"))
    side = str(row.get("taker_book_side") or "").lower()
    if not (
        np.isfinite(e)
        and START_E <= e < END_E
        and np.isfinite(yes)
        and 0.0 <= yes <= 1.0
        and np.isfinite(qty)
        and qty > 0.0
        and side in {"bid", "ask"}
    ):
        return None

    # Recorder elapsed_s is receipt-clock elapsed from this market's M0.
    receipt_s = float(meta_row["window_start_s"]) + float(e)
    exchange_s = _iso_seconds(row.get("exchange_time"))
    if np.isfinite(exchange_s) and exchange_s <= receipt_s + EPS:
        exec_s = float(exchange_s)
        clock = "EXCHANGE"
    elif np.isfinite(exchange_s):
        exec_s = float(receipt_s)
        clock = "CLAMP_EXCHANGE_AFTER_RECEIPT"
    else:
        exec_s = float(receipt_s)
        clock = "RECEIPT_FALLBACK"
    obs_s = max(exec_s, receipt_s)
    return {
        "e": float(e),
        "yes": float(yes),
        "qty": float(qty),
        "side": side,
        "receipt_s": float(receipt_s),
        "exec_s": float(exec_s),
        "obs_s": float(obs_s),
        "trade_id": str(row.get("trade_id") or ""),
        "clock": clock,
    }


def _strict_tail(side, yes):
    if side == "ask" and yes < ENTRY - EPS:
        return "YES"
    if side == "bid" and yes > 1.0 - ENTRY + EPS:
        return "NO"
    return None


def _scan_entries(session: Path, meta: dict, *, show=True):
    """Trade pass 1: strict-through events and first-fill-wins capacity."""
    path = session / "trades_event_time.jsonl"
    by_ticker = defaultdict(list)
    clock = defaultdict(int)
    t0 = time.time()
    read = selected = strict_n = 0

    if show:
        print("PASS 1/3 — strict-through entry flow")

    with path.open("rb") as fh:
        for raw in fh:
            read += 1
            e = _fast_float(raw, ELAPSED_KEY)
            if e is None or e < START_E or e >= END_E:
                continue
            ticker = _fast_ticker(raw)
            if not ticker or ticker not in meta:
                continue
            try:
                row = _loads(raw)
            except Exception:
                continue
            tr = _trade_fields(row, meta[ticker])
            if tr is None:
                continue
            active_s = float(meta[ticker]["window_start_s"]) + START_E + ENTRY_ACTIVATION_S
            if tr["exec_s"] + EPS < active_s or tr["receipt_s"] + EPS < active_s:
                continue
            selected += 1
            clock[tr["clock"]] += 1
            tail = _strict_tail(tr["side"], tr["yes"])
            if tail is None:
                continue
            strict_n += 1
            by_ticker[ticker].append(
                (tr["obs_s"], tr["exec_s"], tr["trade_id"], tail, tr["qty"])
            )
            if show and read % 1_000_000 == 0:
                dt = max(1e-9, time.time() - t0)
                print(
                    f"  read={read:,} selected={selected:,} strict={strict_n:,} "
                    f"rate={read/dt:,.0f} lines/s"
                )

    candidates = {}
    for ticker, m in meta.items():
        ev = by_ticker.get(ticker, [])
        ev.sort(key=lambda z: (z[0], z[1], z[2]))
        if not ev:
            chosen_tail = None
            first_obs = np.nan
            first_exec = np.nan
            race = False
            chosen = []
        else:
            first = ev[0]
            chosen_tail = first[3]
            first_obs = float(first[0])
            first_exec = float(first[1])
            opposite_obs = [float(z[0]) for z in ev if z[3] != chosen_tail]
            race = bool(
                opposite_obs
                and -EPS <= min(opposite_obs) - first_obs <= CANCEL_RACE_S + EPS
            )
            chosen = [z for z in ev if z[3] == chosen_tail]

        cum = 0.0
        full = {}
        for z in chosen:
            cum += float(z[4])
            for q in Q_GRID:
                if q not in full and cum >= float(q) - EPS:
                    full[q] = {
                        "full_entry_obs_s": float(z[0]),
                        "full_entry_exec_s": float(z[1]),
                    }

        qrows = {}
        for q in Q_GRID:
            fill = min(float(q), float(cum))
            qrows[q] = {
                "requested_q": float(q),
                "entry_filled_qty": float(fill),
                "full_entry": bool(fill >= float(q) - EPS),
                "partial_entry": bool(EPS < fill < float(q) - EPS),
                "full_entry_obs_s": _f((full.get(q) or {}).get("full_entry_obs_s")),
                "full_entry_exec_s": _f((full.get(q) or {}).get("full_entry_exec_s")),
            }

        candidates[ticker] = {
            "ticker": ticker,
            "series": m["series"],
            "close_time": m["close_time"],
            "close_s": m["close_s"],
            "window_start_s": m["window_start_s"],
            "tail": chosen_tail,
            "first_fill_obs_s": first_obs,
            "first_fill_exec_s": first_exec,
            "cancel_race_100ms": race,
            "strict_capacity_qty": float(cum),
            "q": qrows,
        }

    stats = {
        "raw_lines": int(read),
        "selected_m5_m12_trades": int(selected),
        "strict_through_rows": int(strict_n),
        "clock": dict(clock),
        "tickers_with_any_strict_fill": int(sum(c["tail"] is not None for c in candidates.values())),
        "cancel_race_tickers": int(sum(c["cancel_race_100ms"] for c in candidates.values())),
        "elapsed_s": float(time.time() - t0),
    }
    return candidates, stats


def _parse_book_snapshot(raw):
    if raw is None:
        return None
    try:
        row = _loads(raw)
    except Exception:
        return None
    cur = OOS._top_state(row)
    if cur is None:
        return None
    return {
        "elapsed_s": _f(row.get("elapsed_s")),
        "yes_bid": float(cur["bid"]),
        "yes_ask": float(cur["ask"]),
        "yes_bid_q1": float(cur["bid_q1"]),
        "yes_ask_q1": float(cur["ask_q1"]),
        "yes_mid": float(cur["mid"]),
        "bid_levels": [(float(p), float(q)) for p, q in cur["bid_levels"]],
        "ask_levels": [(float(p), float(q)) for p, q in cur["ask_levels"]],
    }


def _scan_books_fast(session: Path, meta: dict, candidates: dict, *, show=True):
    """Book pass with no full JSON decode in the 28M-row hot loop."""
    path = session / "book_top3_events.jsonl"
    covered = defaultdict(float)
    last_e = {}
    last_valid = {}

    needed = {ticker for ticker, c in candidates.items() if c["strict_capacity_qty"] > EPS}
    targets = {}
    target_idx = {}
    captured_raw = {}

    for ticker in needed:
        c = candidates[ticker]
        arr = []
        if not c["cancel_race_100ms"]:
            for q in Q_GRID:
                z = c["q"][q]
                if z["full_entry"] and np.isfinite(z["full_entry_obs_s"]):
                    de = float(z["full_entry_obs_s"]) - float(c["window_start_s"])
                    arr.append((de, int(q)))
        arr.sort()
        targets[ticker] = arr
        target_idx[ticker] = 0

    last_valid_raw = {}
    terminal_raw = {}
    t0 = time.time()
    read = fast_rows = 0

    def finalize_before(ticker, e, inclusive=False):
        arr = targets.get(ticker)
        if not arr:
            return
        i = target_idx[ticker]
        while i < len(arr):
            de, q = arr[i]
            ok = de <= e + EPS if inclusive else de < e - EPS
            if not ok:
                break
            captured_raw[(ticker, q)] = last_valid_raw.get(ticker)
            i += 1
        target_idx[ticker] = i

    if show:
        print("PASS 2/3 — fast book coverage + decision/M12 snapshots")
        print("  hot-loop full JSON decodes: 0")

    with path.open("rb") as fh:
        for raw in fh:
            read += 1
            ticker = _fast_ticker(raw)
            if not ticker or ticker not in meta:
                continue
            e = _fast_float(raw, ELAPSED_KEY)
            if e is None or not np.isfinite(e):
                continue
            valid = _fast_valid(raw)
            fast_rows += 1

            pe = last_e.get(ticker)
            if pe is not None:
                a = max(float(pe), START_E)
                b = min(float(e), END_E)
                if b > a and last_valid.get(ticker, False):
                    covered[ticker] += b - a

            if ticker in needed:
                finalize_before(ticker, float(e), inclusive=False)

            last_e[ticker] = float(e)
            last_valid[ticker] = bool(valid)

            if ticker in needed and valid and float(e) < END_E + EPS:
                last_valid_raw[ticker] = bytes(raw)
                if float(e) < END_E - EPS:
                    terminal_raw[ticker] = bytes(raw)
                finalize_before(ticker, float(e), inclusive=True)

            if show and read % 5_000_000 == 0:
                dt = max(1e-9, time.time() - t0)
                print(
                    f"  book lines={read:,} rate={read/dt:,.0f}/s "
                    f"needed_tickers={len(needed):,}"
                )

    for ticker in meta:
        pe = last_e.get(ticker)
        if pe is None:
            continue
        a = max(float(pe), START_E)
        if END_E > a and last_valid.get(ticker, False):
            covered[ticker] += END_E - a

    for ticker in needed:
        finalize_before(ticker, END_E, inclusive=True)

    decision = {
        key: _parse_book_snapshot(raw)
        for key, raw in captured_raw.items()
        if raw is not None
    }
    terminal = {
        ticker: _parse_book_snapshot(raw)
        for ticker, raw in terminal_raw.items()
        if raw is not None
    }
    coverage = {
        ticker: min(WINDOW_S, float(covered.get(ticker, 0.0))) / WINDOW_S
        for ticker in meta
    }

    stats = {
        "raw_lines": int(read),
        "fast_field_rows": int(fast_rows),
        "needed_filled_tickers": int(len(needed)),
        "decision_targets": int(sum(len(v) for v in targets.values())),
        "decision_snapshots": int(len(decision)),
        "terminal_snapshots": int(len(terminal)),
        "eligible_95pct_with_terminal": int(
            sum(
                coverage.get(t, 0.0) >= MIN_BOOK_COVERAGE - EPS and t in terminal
                for t in meta
            )
        ),
        "elapsed_s": float(time.time() - t0),
    }
    return coverage, decision, terminal, stats


def _outcome_price(tail, yes):
    return float(yes) if tail == "YES" else 1.0 - float(yes)


def _buyer_side(tail):
    return "bid" if tail == "YES" else "ask"


def _decision_quote(tail, snap):
    if snap is None:
        return None
    if tail == "YES":
        return {"quote": float(snap["yes_ask"]), "queue": max(0.0, float(snap["yes_ask_q1"]))}
    return {"quote": 1.0 - float(snap["yes_bid"]), "queue": max(0.0, float(snap["yes_bid_q1"]))}


def _scan_passive_exits(session, meta, candidates, decision, coverage, terminal, *, show=True):
    """Trade pass 2: simulate all Q JOIN_ASK exits simultaneously."""
    orders = defaultdict(list)
    for ticker, c in candidates.items():
        if (
            c["tail"] is None
            or c["cancel_race_100ms"]
            or coverage.get(ticker, 0.0) < MIN_BOOK_COVERAGE - EPS
            or ticker not in terminal
        ):
            continue
        for q in Q_GRID:
            z = c["q"][q]
            if not z["full_entry"]:
                continue
            dq = _decision_quote(c["tail"], decision.get((ticker, q)))
            if dq is None:
                continue
            orders[ticker].append({
                "q": int(q),
                "tail": c["tail"],
                "active_s": float(z["full_entry_obs_s"]) + EXIT_ACTIVATION_S,
                "end_s": float(c["window_start_s"]) + END_E,
                "quote": float(dq["quote"]),
                "queue": float(dq["queue"]),
                "remaining": float(q),
                "passive_qty": 0.0,
                "trade_through": False,
                "full_exec_s": np.nan,
            })

    path = session / "trades_event_time.jsonl"
    t0 = time.time()
    read = selected = 0
    if show:
        print("PASS 3/3 — queue-aware fixed JOIN_ASK exits")

    with path.open("rb") as fh:
        for raw in fh:
            read += 1
            e = _fast_float(raw, ELAPSED_KEY)
            if e is None or e < START_E or e >= END_E:
                continue
            ticker = _fast_ticker(raw)
            if not ticker or ticker not in orders:
                continue
            try:
                row = _loads(raw)
            except Exception:
                continue
            tr = _trade_fields(row, meta[ticker])
            if tr is None:
                continue
            selected += 1

            for o in orders[ticker]:
                if o["remaining"] <= EPS:
                    continue
                if tr["exec_s"] + EPS < o["active_s"] or tr["receipt_s"] + EPS < o["active_s"]:
                    continue
                if tr["obs_s"] >= o["end_s"] - EPS:
                    continue
                if tr["side"] != _buyer_side(o["tail"]):
                    continue

                opx = _outcome_price(o["tail"], tr["yes"])
                qpx = float(o["quote"])
                if opx > qpx + EPS:
                    o["passive_qty"] += o["remaining"]
                    o["remaining"] = 0.0
                    o["trade_through"] = True
                    o["full_exec_s"] = float(tr["exec_s"])
                    continue

                if abs(opx - qpx) <= EPS:
                    burn = min(o["queue"], tr["qty"])
                    o["queue"] -= burn
                    avail = max(0.0, tr["qty"] - burn)
                    take = min(o["remaining"], avail)
                    if take > EPS:
                        o["passive_qty"] += take
                        o["remaining"] -= take
                        if o["remaining"] <= EPS:
                            o["remaining"] = 0.0
                            o["full_exec_s"] = float(tr["exec_s"])

            if show and read % 1_000_000 == 0:
                done = sum(o["remaining"] <= EPS for lst in orders.values() for o in lst)
                total = sum(len(lst) for lst in orders.values())
                dt = max(1e-9, time.time() - t0)
                print(
                    f"  read={read:,} relevant={selected:,} "
                    f"full_passive={done:,}/{total:,} rate={read/dt:,.0f}/s"
                )

    passive = {}
    for ticker, lst in orders.items():
        for o in lst:
            passive[(ticker, int(o["q"]))] = {
                "passive_exit_qty": float(o["passive_qty"]),
                "passive_exit_full": bool(o["remaining"] <= EPS),
                "passive_residual": float(o["remaining"]),
                "trade_through_exit": bool(o["trade_through"]),
                "passive_full_exec_s": _f(o["full_exec_s"]),
                "exit_quote": float(o["quote"]),
                "queue_remaining": float(o["queue"]),
            }

    stats = {
        "raw_lines": int(read),
        "relevant_trade_rows": int(selected),
        "passive_orders": int(sum(len(v) for v in orders.values())),
        "full_passive_orders": int(sum(o["remaining"] <= EPS for v in orders.values() for o in v)),
        "elapsed_s": float(time.time() - t0),
    }
    return passive, stats


def _terminal_for_v3(snap):
    if snap is None:
        return None
    return {
        "yes_bid": float(snap["yes_bid"]),
        "yes_ask": float(snap["yes_ask"]),
        "yes_mid": float(snap["yes_mid"]),
        "bid_levels": list(snap["bid_levels"]),
        "ask_levels": list(snap["ask_levels"]),
    }


def _gpu_total_dd(pnl):
    x = np.asarray(pnl, dtype=np.float32)
    if len(x) == 0:
        return 0.0, 0.0
    if MPS_AVAILABLE and _torch is not None:
        t = _torch.as_tensor(x, dtype=_torch.float32, device="mps")
        eq = _torch.cumsum(t, dim=0)
        eq0 = _torch.cat([_torch.zeros(1, device="mps"), eq])
        peak = _torch.cummax(eq0, dim=0).values
        dd = (eq0 - peak).min()
        return float(t.sum().cpu().item()), float(dd.cpu().item())
    eq = np.cumsum(x)
    eq0 = np.r_[0.0, eq]
    peak = np.maximum.accumulate(eq0)
    return float(x.sum()), float((eq0 - peak).min())


def _economics(session_name, candidates, coverage, decision, terminal, passive, fee_mult):
    rows = []
    for ticker, c in candidates.items():
        series = str(c["series"])
        mult = _f(fee_mult.get(series))
        term = terminal.get(ticker)
        coverage_ok = coverage.get(ticker, 0.0) >= MIN_BOOK_COVERAGE - EPS
        eligible = bool(coverage_ok and term is not None and np.isfinite(mult) and mult > 0)

        for q in Q_GRID:
            z = c["q"][q]
            entry_qty = float(z["entry_filled_qty"]) if eligible else 0.0
            full_entry = bool(z["full_entry"]) if eligible else False
            partial_entry = bool(z["partial_entry"]) if eligible else False

            p = passive.get((ticker, int(q))) if eligible else None
            if p is not None and full_entry and not c["cancel_race_100ms"] and (ticker, q) in decision:
                passive_qty = min(entry_qty, float(p["passive_exit_qty"]))
                exit_quote = float(p["exit_quote"])
            else:
                passive_qty = 0.0
                exit_quote = np.nan

            passive_proceeds = passive_qty * exit_quote if passive_qty > EPS and np.isfinite(exit_quote) else 0.0
            forced_qty = max(0.0, entry_qty - passive_qty)

            if forced_qty > EPS and eligible:
                ex = V3._consume_m5_depth(
                    c["tail"], forced_qty, _terminal_for_v3(term), float(mult)
                )
            else:
                ex = {
                    "exit_qty": 0.0,
                    "residual_qty_zero_valued": forced_qty,
                    "m5_exit_proceeds": 0.0,
                    "m5_taker_fee": 0.0,
                    "m5_levels_consumed": 0,
                    "m5_slippage_vs_best_bid": 0.0,
                }

            market_exit_qty = float(ex.get("exit_qty") or 0.0)
            residual_zero = float(ex.get("residual_qty_zero_valued") or 0.0)
            market_proceeds = float(ex.get("m5_exit_proceeds") or 0.0)
            taker_fee = float(ex.get("m5_taker_fee") or 0.0)
            rounding = ROUNDING_DRAG if market_exit_qty > EPS else 0.0
            entry_cost = ENTRY * entry_qty
            net_before_round = passive_proceeds + market_proceeds - entry_cost - taker_fee
            net = net_before_round - rounding
            terminal_qty = passive_qty + market_exit_qty
            terminal_frac = terminal_qty / entry_qty if entry_qty > EPS else np.nan

            rows.append({
                "session": session_name,
                "ticker": ticker,
                "series": series,
                "close_s": float(c["close_s"]),
                "Q": int(q),
                "coverage_fraction": float(coverage.get(ticker, 0.0)),
                "coverage_eligible": bool(eligible),
                "tail": c["tail"],
                "cancel_race_100ms": bool(c["cancel_race_100ms"]),
                "first_fill_obs_s": _f(c["first_fill_obs_s"]),
                "strict_capacity_qty": float(c["strict_capacity_qty"]),
                "entry_filled_qty": float(entry_qty),
                "full_entry": bool(full_entry),
                "partial_entry": bool(partial_entry),
                "decision_snapshot_known": bool((ticker, q) in decision),
                "passive_exit_qty": float(passive_qty),
                "full_passive_exit": bool(passive_qty >= float(q) - EPS),
                "exit_quote": exit_quote,
                "m12_required_qty": float(forced_qty),
                "m12_exit_qty": float(market_exit_qty),
                "terminal_exit_qty": float(terminal_qty),
                "terminal_exit_fraction": terminal_frac,
                "m12_residual_zero_valued": float(residual_zero),
                "m12_taker_fee": float(taker_fee),
                "balance_rounding_drag": float(rounding),
                "passive_proceeds": float(passive_proceeds),
                "m12_proceeds": float(market_proceeds),
                "net_pnl_before_rounding": float(net_before_round),
                "net_pnl": float(net),
                "m12_levels_consumed": int(ex.get("m5_levels_consumed") or 0),
                "m12_slippage_vs_best_bid": float(ex.get("m5_slippage_vs_best_bid") or 0.0),
            })
    return pd.DataFrame(rows)


def _session_span_hours(meta):
    x = [float(v["close_s"]) for v in meta.values()]
    if len(x) < 2:
        return np.nan
    return (max(x) - min(x)) / 3600.0 + 0.25


def _curve(detail, meta):
    rows = []
    span_h = _session_span_hours(meta)
    for q, g0 in detail.groupby("Q", sort=True):
        g = g0[g0["coverage_eligible"]].copy()
        filled = g[g["entry_filled_qty"] > EPS].copy()
        filled.sort_values(["first_fill_obs_s", "ticker"], kind="mergesort", inplace=True)

        entry_qty = float(g["entry_filled_qty"].sum())
        passive_qty = float(g["passive_exit_qty"].sum())
        market_req = float(g["m12_required_qty"].sum())
        market_exit = float(g["m12_exit_qty"].sum())
        terminal_qty = float(g["terminal_exit_qty"].sum())
        residual = float(g["m12_residual_zero_valued"].sum())
        total_pnl, dd = _gpu_total_dd(filled["net_pnl"].fillna(0.0).to_numpy())
        terminal_frac = terminal_qty / entry_qty if entry_qty > EPS else np.nan
        missing_decision = int((g["full_entry"] & ~g["decision_snapshot_known"]).sum())
        races = int(((g["entry_filled_qty"] > EPS) & g["cancel_race_100ms"]).sum())
        requested_total = float(len(g) * int(q))

        rows.append({
            "Q": int(q),
            "eligible_markets": int(len(g)),
            "entry_events": int(len(filled)),
            "full_entries": int(g["full_entry"].sum()),
            "partial_entries": int(g["partial_entry"].sum()),
            "requested_contracts": requested_total,
            "entry_filled_qty": entry_qty,
            "entry_fill_fraction_requested": entry_qty / requested_total if requested_total > EPS else np.nan,
            "passive_exit_qty": passive_qty,
            "passive_exit_fraction_entry": passive_qty / entry_qty if entry_qty > EPS else np.nan,
            "full_passive_positions": int(g["full_passive_exit"].sum()),
            "m12_required_qty": market_req,
            "m12_exit_qty": market_exit,
            "terminal_exit_fraction": terminal_frac,
            "m12_residual_zero_valued": residual,
            "m12_taker_fees": float(g["m12_taker_fee"].sum()),
            "rounding_drag": float(g["balance_rounding_drag"].sum()),
            "net_pnl": float(total_pnl),
            "net_per_filled_contract": total_pnl / entry_qty if entry_qty > EPS else np.nan,
            "normalized_pnl_24h": total_pnl * 24.0 / span_h if np.isfinite(span_h) and span_h > 0 else np.nan,
            "max_drawdown": float(dd),
            "missing_full_entry_decisions": int(missing_decision),
            "cancel_race_fill_tickers": int(races),
            "capacity_gate_positive_99pct_terminal_no_missing_no_race": bool(
                total_pnl > 0
                and np.isfinite(terminal_frac)
                and terminal_frac >= 0.99 - EPS
                and missing_decision == 0
                and races == 0
            ),
        })
    return pd.DataFrame(rows).sort_values("Q").reset_index(drop=True)


def _print_curve(session_name, curve):
    print()
    print("=" * 112)
    print(f"M5->M12 FEE-ADJUSTED CAPACITY | SESSION {session_name}")
    print("=" * 112)
    for _, r in curve.iterrows():
        print()
        print(f"Q{int(r['Q'])}")
        print(
            f"  entry events:           {int(r['entry_events'])}"
            f" | full={int(r['full_entries'])} partial={int(r['partial_entries'])}"
        )
        print(
            f"  entry filled qty:       {r['entry_filled_qty']:.2f}"
            f" | fill/requested={100*r['entry_fill_fraction_requested']:.2f}%"
        )
        print(
            f"  passive exit qty:       {r['passive_exit_qty']:.2f}"
            f" | {100*r['passive_exit_fraction_entry']:.2f}% of entry"
        )
        if np.isfinite(r["terminal_exit_fraction"]):
            print(f"  terminal coverage:      {100*r['terminal_exit_fraction']:.3f}%")
        else:
            print("  terminal coverage:      n/a")
        print(f"  M12 residual zero:      {r['m12_residual_zero_valued']:.2f}")
        print(
            f"  fees + rounding:        ${r['m12_taker_fees']:.4f} + ${r['rounding_drag']:.4f}"
        )
        print(
            f"  NET PnL:                ${r['net_pnl']:+.4f}"
            f" | {100*r['net_per_filled_contract']:+.3f}c/fill"
        )
        print(f"  normalized / 24h:       ${r['normalized_pnl_24h']:+.2f}")
        print(f"  realized-sequence DD:   ${r['max_drawdown']:.4f}")
        print(
            "  >=99% capacity gate:    "
            + ("PASS" if r["capacity_gate_positive_99pct_terminal_no_missing_no_race"] else "FAIL")
        )


def run(*, force_rebuild=False, show=True):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fee_mult, fee_source = _fee_multipliers()

    if show:
        print("=" * 112)
        print("FAST M5->M12 5c / FIXED JOIN_ASK CAPACITY STUDY")
        print("=" * 112)
        print("Version:                    ", VERSION)
        print("JSON engine:                ", JSON_ENGINE)
        print("Aggregation device:         ", AGG_DEVICE.upper())
        print("New packages installed:     NO")
        print("Entry:                       5c, M5+100ms")
        print("Boundary:                    M12")
        print("First fill wins:             YES")
        print("Entry capacity:              strict-through qty only; exact 5c excluded")
        print("Exit:                        fixed JOIN_ASK +100ms; queue-aware")
        print("M12 fallback:                recorded top3 + fee; deeper residual zero")
        print("Q grid:                      ", Q_GRID)
        print("Fee source:                  ", fee_source)
        print("Scientific status:           DEVELOPMENT / EXPLORATORY")
        print("API CALLED:                  NO")
        print("ORDERS SENT:                 NO")
        if MPS_AVAILABLE:
            print(
                "MPS note: GPU is used for cumulative-PnL/drawdown only; the raw-file "
                "speedup comes from avoiding full JSON decode of the 28.5M-row book."
            )
        else:
            print(
                "MPS note: unavailable in this kernel; no install is attempted. "
                "The main speedup is the bytes-level book scanner."
            )

    all_curves = {}
    all_detail = {}
    summary = {
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "version": VERSION,
        "research_stage": "DEVELOPMENT_EXPLORATORY_CAPACITY",
        "json_engine": JSON_ENGINE,
        "aggregation_device": AGG_DEVICE,
        "fee_source": fee_source,
        "sessions": {},
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
    }

    for session_name in SESSION_NAMES:
        session = (FULL15_ROOT / session_name).resolve()
        if not session.exists():
            raise FileNotFoundError(f"Missing full-window session: {session}")
        required = [
            session / "market_metadata.jsonl",
            session / "trades_event_time.jsonl",
            session / "book_top3_events.jsonl",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing source files: " + " | ".join(missing))

        out = OUTPUT_ROOT / session_name
        cache = out / "cache"
        out.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)

        source_fp = {
            "version": VERSION,
            "market_metadata": _fingerprint(session / "market_metadata.jsonl"),
            "trades": _fingerprint(session / "trades_event_time.jsonl"),
            "book": _fingerprint(session / "book_top3_events.jsonl"),
            "entry_c": ENTRY_C,
            "start_e": START_E,
            "end_e": END_E,
            "q_grid": list(Q_GRID),
        }
        fp_path = cache / "source_fingerprint.json"

        if show:
            print()
            print("#" * 112)
            print("SESSION:", session_name)
            print("#" * 112)

        meta = _metadata(session)
        if not meta:
            raise RuntimeError(f"No metadata in {session}")

        entry_cache = cache / "entries.pkl"
        book_cache = cache / "book_support.pkl"
        passive_cache = cache / "passive.pkl"
        cache_valid = _cache_ok(fp_path, source_fp)

        if not force_rebuild and entry_cache.exists() and cache_valid:
            with entry_cache.open("rb") as fh:
                candidates, entry_stats = pickle.load(fh)
            if show:
                print(f"PASS 1 CACHE HIT | filled tickers={entry_stats['tickers_with_any_strict_fill']:,}")
        else:
            candidates, entry_stats = _scan_entries(session, meta, show=show)
            with entry_cache.open("wb") as fh:
                pickle.dump((candidates, entry_stats), fh, protocol=pickle.HIGHEST_PROTOCOL)

        if not force_rebuild and book_cache.exists() and cache_valid:
            with book_cache.open("rb") as fh:
                coverage, decision, terminal, book_stats = pickle.load(fh)
            if show:
                print(f"PASS 2 CACHE HIT | decisions={len(decision):,} terminals={len(terminal):,}")
        else:
            coverage, decision, terminal, book_stats = _scan_books_fast(
                session, meta, candidates, show=show
            )
            with book_cache.open("wb") as fh:
                pickle.dump((coverage, decision, terminal, book_stats), fh, protocol=pickle.HIGHEST_PROTOCOL)

        if not force_rebuild and passive_cache.exists() and cache_valid:
            with passive_cache.open("rb") as fh:
                passive, passive_stats = pickle.load(fh)
            if show:
                print(f"PASS 3 CACHE HIT | passive orders={passive_stats['passive_orders']:,}")
        else:
            passive, passive_stats = _scan_passive_exits(
                session, meta, candidates, decision, coverage, terminal, show=show
            )
            with passive_cache.open("wb") as fh:
                pickle.dump((passive, passive_stats), fh, protocol=pickle.HIGHEST_PROTOCOL)

        _atomic_json(fp_path, source_fp)

        detail = _economics(
            session_name, candidates, coverage, decision, terminal, passive, fee_mult
        )
        curve = _curve(detail, meta)
        detail.to_csv(out / "m5_m12_capacity_detail.csv", index=False)
        curve.to_csv(out / "m5_m12_capacity_curve.csv", index=False)

        if show:
            _print_curve(session_name, curve)

        q1 = curve[curve["Q"] == 1]
        q1_events = int(q1.iloc[0]["entry_events"]) if len(q1) else 0
        all_detail[session_name] = detail
        all_curves[session_name] = curve
        summary["sessions"][session_name] = {
            "source": str(session),
            "metadata_markets": int(len(meta)),
            "q1_entry_events": q1_events,
            "entry_stats": entry_stats,
            "book_stats": book_stats,
            "passive_stats": passive_stats,
            "curve_csv": str(out / "m5_m12_capacity_curve.csv"),
            "detail_csv": str(out / "m5_m12_capacity_detail.csv"),
        }

    if show:
        print()
        print("=" * 112)
        print("CROSS-SESSION CAPACITY COMPARISON")
        print("=" * 112)
        for q in Q_GRID:
            parts = []
            for s in SESSION_NAMES:
                r = all_curves[s]
                r = r[r["Q"] == q]
                if r.empty:
                    continue
                x = r.iloc[0]
                parts.append(
                    f"{s}: net=${x['net_pnl']:+.3f}, "
                    f"edge={100*x['net_per_filled_contract']:+.3f}c, "
                    f"terminal={100*x['terminal_exit_fraction']:.2f}%, "
                    f"gate={'PASS' if x['capacity_gate_positive_99pct_terminal_no_missing_no_race'] else 'FAIL'}"
                )
            print(f"Q{q:<3d} | " + " || ".join(parts))
        print()
        print("SANITY TARGET FROM PRIOR PHENOMENON NOTEBOOK:")
        print("  20260817_064152 Q1 strict-through events should be about 478")
        print("  20260817_005114 Q1 strict-through events should be about 42")
        print("If these are materially different, do not interpret the Q curve.")
        print()
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")
        print("Output root:", OUTPUT_ROOT)

    _atomic_json(OUTPUT_ROOT / "run_summary.json", summary)
    return {
        "summary": summary,
        "curves": all_curves,
        "detail": all_detail,
        "output_root": str(OUTPUT_ROOT),
    }


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-rebuild", action="store_true")
    a = ap.parse_args()
    run(force_rebuild=bool(a.force_rebuild), show=True)


if __name__ == "__main__":
    _main()
