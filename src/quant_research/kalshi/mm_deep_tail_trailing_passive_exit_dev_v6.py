from __future__ import annotations

"""Fast DEVELOPMENT-only causal trailing passive-exit replay for the 5c/Q5 deep-tail entry.

Scientific question
-------------------
The 24h development sample showed that after a conservative Q5 @ 5c fill, the outcome
book often rebounds and then migrates back toward the 5-6c region before M5.  This module
turns that descriptive result into a causal strategy test:

    5c/Q5 passive entry -> track RUNNING executable-bid high -> detect a drawdown from
    that running high -> place one fixed passive SELL at the contemporaneous outcome ask
    -> leave it resting -> M5 top-3 fallback for any residual.

The future/global peak is NEVER used by the strategy.  Only the running high known at the
current receipt-clock book state is used.

Development variants
--------------------
All variants require the running bid high to reach at least 7c before a trigger can fire.
This is development/discovery, not validation.

Absolute trailing drawdown:
    ABS_1C, ABS_2C, ABS_3C, ABS_5C

Fractional rebound retention (entry anchor = 5c):
    RETAIN_75: trigger after 25% of the 5c->running-high rebound is given back
    RETAIN_50: trigger after 50% is given back
    RETAIN_25: trigger after 75% is given back

Execution
---------
- Trigger is formed from receipt-clock BBO states only.
- Exit decision quote = outcome best ask at the trigger state.
- Assumed order activation latency = 100ms, same engineering assumption used elsewhere.
- At activation, if the stale sell quote would cross the current bid, a conservative
  post-only guard rejects the quote and the position remains M5-only.
- Queue ahead is measured at activation when the quote is visible in recorded top-3.
  If the quote is inside the spread, queue ahead is zero.  If it is outside recorded
  top-3, exact-price queue is unknown, so exact-price prints do NOT grant a fill; only a
  strict trade-through can fill the order.
- Passive fills use the same public buyer-flow logic as V4.  A trade strictly above our
  active sell quote proves all lower-price resting sell liquidity, including our Q5,
  was consumed before that higher trade.
- Residual inventory at M5 crosses recorded outcome top-3 using V3, including historical
  taker fees and the same conservative $0.0099 balance-rounding drag per M5 cross.

Speed
-----
Prior studies repeatedly parsed ~9M book rows.  This module builds a reusable LOCAL cache
for only the 32-ish tickers that actually produced 5c/Q5 entry fills.  It uses system
`grep -F` to prefilter raw JSONL before JSON decoding, then stores compact pandas pickle
files.  First run builds the cache; subsequent development reruns load it directly and
should be dramatically faster.

This module is hard-bound to the already-inspected 24h DEVELOPMENT source.  It does not
read the 15h validation sample.  No API calls.  No orders.  Raw source files are read-only.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C
from . import mm_deep_tail_passive_feasibility_dev_v1 as V1
from . import mm_deep_tail_capacity_dev_v3 as V3
from . import mm_deep_tail_passive_exit_dev_v4 as V4
from . import mm_deep_tail_reversion_exit_dev_v5 as V5

VERSION = "MM_DEEP_TAIL_TRAILING_PASSIVE_EXIT_DEV_V6"
HARD_BOUND_SESSION = V1.HARD_BOUND_SESSION
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_trailing_passive_exit_dev_v6"
CACHE_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_fast_cache_v1"

ENTRY_C = 5.0
ENTRY = 0.05
QTY = 5.0
MIN_RUNNING_HIGH_C = 7.0
ACTIVATION_LATENCY_MS = V1.ACTIVATION_LATENCY_MS
ROUNDING_DRAG = V1.BALANCE_ROUNDING_UPPER_BOUND_PER_CROSS
M1_S = V1.M1_S
M5_S = V1.M5_S
EPS = 1e-10

VARIANTS = (
    ("ABS_1C", "ABS", 1.0),
    ("ABS_2C", "ABS", 2.0),
    ("ABS_3C", "ABS", 3.0),
    ("ABS_5C", "ABS", 5.0),
    ("RETAIN_75", "RETAIN", 0.75),
    ("RETAIN_50", "RETAIN", 0.50),
    ("RETAIN_25", "RETAIN", 0.25),
)


def _new_output(name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / name
    if out.exists():
        out = OUTPUT_ROOT / f"{name}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out.resolve()


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_json(path: Path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _f(x, default=np.nan):
    try:
        z = float(x)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _latest_result_file(root: Path, session_name: str, filename: str):
    root = Path(root)
    if not root.exists():
        return None
    candidates = []
    for d in root.glob(f"{session_name}*"):
        p = d / filename
        if p.exists():
            try:
                mt = p.stat().st_mtime
            except Exception:
                mt = 0.0
            candidates.append((mt, p))
    if not candidates:
        return None
    candidates.sort(key=lambda z: z[0], reverse=True)
    return candidates[0][1]


def _load_or_build_anchors(source: Path, meta: dict, *, show=True):
    """Prefer the already-produced V5 anchor CSV; fallback only if it is unavailable."""
    p = _latest_result_file(
        C.PROJECT_ROOT / "results" / "kalshi_deep_tail_reversion_exit_dev_v5",
        source.name,
        "entry_fill_anchors.csv",
    )
    if p is not None:
        if show:
            print("FAST PATH: loading frozen 5c/Q5 entry anchors from", p)
        a = pd.read_csv(p)
        needed = {"ticker", "tail", "series", "entry_filled_qty", "exit_active_s"}
        if needed.issubset(a.columns):
            return a.copy(), str(p), "REUSED_V5_ANCHOR_CACHE"

    if show:
        print("Anchor cache missing/incompatible; one-time fallback trade scan...")
    trades, _ = V1._load_trades(source, meta, show=show)
    a = V5._entry_anchors(meta, trades)
    if a.empty:
        raise RuntimeError("No strict-through 5c/Q5 fills found while rebuilding anchors")
    return a, "RAW_REBUILT", "RAW_REBUILT"


def _grep_json_lines(path: Path, tickers: set[str]):
    """Yield only lines containing relevant tickers, using native grep when available."""
    tickers = {str(x) for x in tickers if str(x)}
    if not tickers:
        return

    grep = shutil.which("grep")
    if grep:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
            pattern_path = Path(fh.name)
            for t in sorted(tickers):
                fh.write(t + "\n")
        env = dict(os.environ)
        env["LC_ALL"] = "C"
        try:
            proc = subprocess.Popen(
                [grep, "-F", "-f", str(pattern_path), str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.strip():
                    yield line
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            rc = proc.wait()
            if rc not in (0, 1):
                raise RuntimeError(f"grep failed rc={rc}: {stderr[:500]}")
        finally:
            try:
                pattern_path.unlink()
            except Exception:
                pass
        return

    # Portable fallback.  Slower, but still avoids JSON decoding for irrelevant lines.
    needles = [t.encode("utf-8") for t in sorted(tickers)]
    with Path(path).open("rb") as fh:
        for raw in fh:
            if any(n in raw for n in needles):
                yield raw.decode("utf-8", errors="ignore")


def _fixed_levels(cur: dict, side: str):
    levels = list(cur.get(f"{side}_levels") or [])[:3]
    out = []
    for i in range(3):
        if i < len(levels):
            try:
                p, q = float(levels[i][0]), max(0.0, float(levels[i][1]))
            except Exception:
                p, q = np.nan, np.nan
        else:
            p, q = np.nan, np.nan
        out.extend([p, q])
    return out


def _state_to_m5(cur: dict, receipt_s: float, elapsed_s: float):
    return {
        "yes_bid": float(cur["bid"]),
        "yes_ask": float(cur["ask"]),
        "yes_mid": float(cur["mid"]),
        "bid_levels": [(float(p), float(q)) for p, q in cur["bid_levels"]],
        "ask_levels": [(float(p), float(q)) for p, q in cur["ask_levels"]],
        "receipt_s": float(receipt_s),
        "snapshot_elapsed_s": float(elapsed_s),
        "true_m5_finalized": True,
    }


def _build_book_cache(source: Path, anchors: pd.DataFrame, cache_dir: Path, *, show=True):
    tickers = set(anchors["ticker"].astype(str))
    min_active = (
        anchors.groupby("ticker")["exit_active_s"].min().astype(float).to_dict()
    )
    rows = []
    last = {}
    m5 = {}
    finalized = set()
    relevant_lines = 0

    if show:
        print(f"FAST CACHE: prefiltering ~9M book rows to {len(tickers)} relevant tickers via grep -F...")

    for line in _grep_json_lines(source / "book_top3_events.jsonl", tickers):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ticker = str(r.get("ticker") or "")
        if ticker not in tickers or ticker in finalized:
            continue
        relevant_lines += 1
        e = _f(r.get("elapsed_s"))
        rt = V1._ts(r.get("receipt_time"))
        cur = V1.OOS._top_state(r)
        if not (np.isfinite(e) and np.isfinite(rt) and cur is not None):
            continue

        if e >= M5_S:
            prev = last.get(ticker)
            if prev is not None:
                m5[ticker] = prev["m5"]
            finalized.add(ticker)
            last.pop(ticker, None)
            continue

        if e < M1_S:
            continue

        bp = _fixed_levels(cur, "bid")
        ap = _fixed_levels(cur, "ask")
        m5state = _state_to_m5(cur, rt, e)
        last[ticker] = {"m5": m5state}

        if rt + EPS < float(min_active.get(ticker, -np.inf)):
            continue

        rows.append({
            "ticker": ticker,
            "receipt_s": float(rt),
            "elapsed_s": float(e),
            "yes_bid": float(cur["bid"]),
            "yes_ask": float(cur["ask"]),
            "yes_mid": float(cur["mid"]),
            "bid_p1": bp[0], "bid_q1": bp[1],
            "bid_p2": bp[2], "bid_q2": bp[3],
            "bid_p3": bp[4], "bid_q3": bp[5],
            "ask_p1": ap[0], "ask_q1": ap[1],
            "ask_p2": ap[2], "ask_q2": ap[3],
            "ask_p3": ap[4], "ask_q3": ap[5],
        })

        if show and relevant_lines % 100_000 == 0:
            print(f"  relevant book lines parsed: {relevant_lines:,} | cached states={len(rows):,}")

    if not rows:
        raise RuntimeError("Fast book cache found no relevant BBO states")

    df = pd.DataFrame(rows).sort_values(["ticker", "receipt_s"], kind="mergesort").reset_index(drop=True)
    df.to_pickle(cache_dir / "relevant_post_fill_bbo.pkl")
    _atomic_json(cache_dir / "m5_top3.json", m5)

    if show:
        print(f"FAST CACHE: book states saved={len(df):,}; M5 snapshots={len(m5):,}")
    return df, m5


def _build_trade_cache(source: Path, anchors: pd.DataFrame, cache_dir: Path, *, show=True):
    tickers = set(anchors["ticker"].astype(str))
    min_active = anchors.groupby("ticker")["exit_active_s"].min().astype(float).to_dict()
    rows = []
    stats = defaultdict(int)

    if show:
        print(f"FAST CACHE: prefiltering trade tape to {len(tickers)} relevant tickers via grep -F...")

    for line in _grep_json_lines(source / "trades_event_time.jsonl", tickers):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ticker = str(r.get("ticker") or "")
        if ticker not in tickers:
            continue
        e = _f(r.get("elapsed_s"))
        if not (np.isfinite(e) and M1_S <= e < M5_S):
            continue
        yes = _f(r.get("yes_price"))
        qty = _f(r.get("qty"))
        side = str(r.get("taker_book_side") or "").lower()
        rt = V1.V11._receipt_s(r)
        if not (
            np.isfinite(yes) and 0.0 <= yes <= 1.0
            and np.isfinite(qty) and qty > 0
            and side in {"bid", "ask"}
            and np.isfinite(rt)
        ):
            continue
        exec_s, source_name = V1.V11._causal_exec_s(r, rt)
        obs_s = float(max(exec_s, rt))
        if obs_s + EPS < float(min_active.get(ticker, -np.inf)):
            continue
        rows.append({
            "ticker": ticker,
            "exec_s": float(exec_s),
            "receipt_s": float(rt),
            "obs_s": obs_s,
            "yes_price": float(yes),
            "qty": float(qty),
            "taker_book_side": side,
            "trade_id": str(r.get("trade_id") or ""),
        })
        stats[str(source_name)] += 1

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["ticker", "exec_s", "receipt_s", "trade_id"], kind="mergesort").reset_index(drop=True)
    df.to_pickle(cache_dir / "relevant_post_fill_trades.pkl")
    _atomic_json(cache_dir / "trade_clock_stats.json", dict(stats))
    if show:
        print(f"FAST CACHE: relevant trades saved={len(df):,}")
    return df, dict(stats)


def _load_or_build_fast_cache(source: Path, anchors: pd.DataFrame, *, rebuild=False, show=True):
    cache_dir = CACHE_ROOT / source.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    bbo_path = cache_dir / "relevant_post_fill_bbo.pkl"
    trades_path = cache_dir / "relevant_post_fill_trades.pkl"
    m5_path = cache_dir / "m5_top3.json"
    stats_path = cache_dir / "trade_clock_stats.json"

    if not rebuild and bbo_path.exists() and trades_path.exists() and m5_path.exists():
        if show:
            print("FAST CACHE HIT:", cache_dir)
        bbo = pd.read_pickle(bbo_path)
        trades = pd.read_pickle(trades_path)
        m5 = _read_json(m5_path, {}) or {}
        stats = _read_json(stats_path, {}) or {}
        return bbo, trades, m5, stats, "CACHE_HIT"

    if show:
        print("FAST CACHE MISS — building once. Future runs reuse this cache.")
    bbo, m5 = _build_book_cache(source, anchors, cache_dir, show=show)
    trades, stats = _build_trade_cache(source, anchors, cache_dir, show=show)
    _atomic_json(cache_dir / "manifest.json", {
        "version": VERSION,
        "source": str(source),
        "source_session": source.name,
        "tickers": sorted(set(anchors["ticker"].astype(str))),
        "anchors": int(len(anchors)),
        "bbo_rows": int(len(bbo)),
        "trade_rows": int(len(trades)),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    return bbo, trades, m5, stats, "CACHE_BUILT"


def _outcome_bbo_arrays(g: pd.DataFrame, tail: str):
    if tail == "YES":
        bid = g["yes_bid"].to_numpy(float)
        ask = g["yes_ask"].to_numpy(float)
    else:
        bid = 1.0 - g["yes_ask"].to_numpy(float)
        ask = 1.0 - g["yes_bid"].to_numpy(float)
    return bid, ask


def _outcome_ask_levels(row: pd.Series, tail: str):
    levels = []
    if tail == "YES":
        for i in (1, 2, 3):
            p = _f(row.get(f"ask_p{i}"))
            q = _f(row.get(f"ask_q{i}"))
            if np.isfinite(p) and np.isfinite(q) and q >= 0:
                levels.append((p, q))
    else:
        for i in (1, 2, 3):
            p = _f(row.get(f"bid_p{i}"))
            q = _f(row.get(f"bid_q{i}"))
            if np.isfinite(p) and np.isfinite(q) and q >= 0:
                levels.append((1.0 - p, q))
        levels.sort(key=lambda z: z[0])
    return levels


def _queue_at(levels, price):
    for p, q in levels:
        if abs(float(p) - float(price)) <= EPS:
            return max(0.0, float(q)), True
    return np.inf, False


def _trigger_index(bids: np.ndarray, kind: str, value: float):
    if len(bids) == 0:
        return None, None
    running = np.maximum.accumulate(bids)
    armed = running >= MIN_RUNNING_HIGH_C / 100.0 - EPS
    if kind == "ABS":
        cond = armed & ((running - bids) >= float(value) / 100.0 - EPS)
    elif kind == "RETAIN":
        threshold = ENTRY + float(value) * (running - ENTRY)
        cond = armed & (bids <= threshold + EPS)
    else:
        raise ValueError(kind)
    idx = np.flatnonzero(cond)
    if len(idx) == 0:
        return None, running
    return int(idx[0]), running


def _m5_for_v3(raw: dict | None):
    if not raw:
        return None
    return {
        "yes_bid": float(raw["yes_bid"]),
        "yes_ask": float(raw["yes_ask"]),
        "yes_mid": float(raw["yes_mid"]),
        "bid_levels": [(float(p), float(q)) for p, q in raw.get("bid_levels", [])],
        "ask_levels": [(float(p), float(q)) for p, q in raw.get("ask_levels", [])],
        "snapshot_elapsed_s": float(raw.get("snapshot_elapsed_s", np.nan)),
        "true_m5_finalized": bool(raw.get("true_m5_finalized", False)),
    }


def _baseline_m5(tail: str, qty: float, m5snap: dict, mult: float):
    ex = V3._consume_m5_depth(tail, qty, m5snap, mult)
    rounding = ROUNDING_DRAG if ex["exit_qty"] > EPS else 0.0
    net = (
        float(ex["m5_exit_proceeds"])
        - ENTRY * qty
        - float(ex["m5_taker_fee"])
        - rounding
    )
    return ex, rounding, float(net)


def _load_v4_join_benchmark(source_name: str):
    p = _latest_result_file(
        C.PROJECT_ROOT / "results" / "kalshi_deep_tail_passive_exit_dev_v4",
        source_name,
        "passive_exit_variant_summary.csv",
    )
    if p is None:
        return np.nan, None
    try:
        x = pd.read_csv(p)
        q = x[x["variant"].astype(str).eq("JOIN_ASK")]
        if len(q):
            return float(q.iloc[0]["total_net_pnl_rounding_bound"]), str(p)
    except Exception:
        pass
    return np.nan, str(p)


def _evaluate(source: Path, anchors: pd.DataFrame, bbo: pd.DataFrame, trades_df: pd.DataFrame, m5: dict, *, show=True):
    fee = _read_json(source / "fee_preflight.json", {}) or {}
    if not fee.get("ok"):
        raise RuntimeError("Stored development fee preflight is not PASS")
    fee_mult = {str(k): float(v) for k, v in (fee.get("multipliers") or {}).items()}

    bbo_by_ticker = {str(t): g.sort_values("receipt_s", kind="mergesort").reset_index(drop=True) for t, g in bbo.groupby("ticker", sort=False)}
    trades_by_ticker = {}
    if len(trades_df):
        for t, g in trades_df.groupby("ticker", sort=False):
            trades_by_ticker[str(t)] = g.to_dict("records")

    rows = []
    for _, a in anchors.iterrows():
        ticker = str(a["ticker"])
        tail = str(a["tail"])
        series = str(a.get("series") or "")
        qty = float(a["entry_filled_qty"])
        if qty <= EPS:
            continue
        mult = _f(fee_mult.get(series))
        m5snap = _m5_for_v3(m5.get(ticker))
        if not (np.isfinite(mult) and mult > 0 and m5snap is not None):
            continue

        base_ex, base_round, base_net = _baseline_m5(tail, qty, m5snap, mult)
        g0 = bbo_by_ticker.get(ticker)
        if g0 is None or g0.empty:
            continue
        active0 = float(a["exit_active_s"])
        g = g0[g0["receipt_s"] + EPS >= active0].copy().reset_index(drop=True)
        if g.empty:
            continue
        bids, asks = _outcome_bbo_arrays(g, tail)
        times = g["receipt_s"].to_numpy(float)

        for name, kind, value in VARIANTS:
            idx, running = _trigger_index(bids, kind, value)
            triggered = idx is not None
            passive_qty = 0.0
            residual = qty
            quote_px = np.nan
            queue0 = np.nan
            queue_known = False
            post_only_reject = False
            trigger_s = np.nan
            trigger_bid = np.nan
            trigger_ask = np.nan
            trigger_high = np.nan
            quote_active_s = np.nan
            activation_bid = np.nan
            activation_ask = np.nan
            passive_full = False
            first_fill_s = np.nan
            full_fill_s = np.nan

            if triggered:
                trigger_s = float(times[idx])
                trigger_bid = float(bids[idx])
                trigger_ask = float(asks[idx])
                trigger_high = float(running[idx])
                quote_px = trigger_ask
                quote_active_s = trigger_s + ACTIVATION_LATENCY_MS / 1000.0

                ai = int(np.searchsorted(times, quote_active_s - EPS, side="left"))
                if ai >= len(g):
                    post_only_reject = True
                else:
                    activation_row = g.iloc[ai]
                    abid, aask = _outcome_bbo_arrays(g.iloc[[ai]], tail)
                    activation_bid = float(abid[0])
                    activation_ask = float(aask[0])

                    if quote_px <= activation_bid + EPS:
                        # Conservative post-only assumption: marketable stale quote is rejected.
                        post_only_reject = True
                    else:
                        levels = _outcome_ask_levels(activation_row, tail)
                        if quote_px < activation_ask - EPS:
                            queue0 = 0.0
                            queue_known = True
                        else:
                            queue0, queue_known = _queue_at(levels, quote_px)

                        quote = {
                            "quote_price": float(quote_px),
                            "queue_ahead_initial": float(queue0),
                        }
                        ex = V4._simulate_passive_exit(
                            trades_by_ticker.get(ticker, []),
                            tail,
                            quote,
                            quote_active_s,
                            qty,
                        )
                        passive_qty = float(ex["passive_exit_qty"])
                        residual = float(ex["passive_exit_residual_qty"])
                        passive_full = bool(ex["passive_exit_full"])
                        first_fill_s = _f(ex.get("first_passive_exit_exec_s"))
                        full_fill_s = _f(ex.get("full_passive_exit_exec_s"))

            if residual > EPS:
                m5ex = V3._consume_m5_depth(tail, residual, m5snap, mult)
            else:
                m5ex = {
                    "exit_qty": 0.0,
                    "residual_qty_zero_valued": 0.0,
                    "m5_exit_proceeds": 0.0,
                    "m5_taker_fee": 0.0,
                    "m5_slippage_vs_best_bid": 0.0,
                    "m5_cross_cost_vs_mid": 0.0,
                }
            rounding = ROUNDING_DRAG if m5ex["exit_qty"] > EPS else 0.0
            passive_proceeds = passive_qty * (quote_px if np.isfinite(quote_px) else 0.0)
            net = (
                passive_proceeds
                + float(m5ex["m5_exit_proceeds"])
                - ENTRY * qty
                - float(m5ex["m5_taker_fee"])
                - rounding
            )

            rows.append({
                "variant": name,
                "trigger_kind": kind,
                "trigger_value": value,
                "ticker": ticker,
                "series": series,
                "close_time": str(a.get("close_time") or ""),
                "tail": tail,
                "entry_filled_qty": qty,
                "triggered": bool(triggered),
                "trigger_s": trigger_s,
                "seconds_fill_active_to_trigger": trigger_s - active0 if np.isfinite(trigger_s) else np.nan,
                "running_high_c_at_trigger": 100.0 * trigger_high if np.isfinite(trigger_high) else np.nan,
                "trigger_bid_c": 100.0 * trigger_bid if np.isfinite(trigger_bid) else np.nan,
                "trigger_ask_c": 100.0 * trigger_ask if np.isfinite(trigger_ask) else np.nan,
                "quote_c": 100.0 * quote_px if np.isfinite(quote_px) else np.nan,
                "quote_active_s": quote_active_s,
                "activation_bid_c": 100.0 * activation_bid if np.isfinite(activation_bid) else np.nan,
                "activation_ask_c": 100.0 * activation_ask if np.isfinite(activation_ask) else np.nan,
                "post_only_reject": bool(post_only_reject),
                "queue_ahead_initial": queue0,
                "queue_known_top3": bool(queue_known),
                "passive_exit_qty": passive_qty,
                "passive_exit_full": bool(passive_full),
                "seconds_trigger_to_first_passive_fill": first_fill_s - trigger_s if np.isfinite(first_fill_s) and np.isfinite(trigger_s) else np.nan,
                "seconds_trigger_to_full_passive_exit": full_fill_s - trigger_s if np.isfinite(full_fill_s) and np.isfinite(trigger_s) else np.nan,
                "m5_exit_qty": float(m5ex["exit_qty"]),
                "m5_residual_zero_valued": float(m5ex["residual_qty_zero_valued"]),
                "m5_taker_fee": float(m5ex["m5_taker_fee"]),
                "rounding_drag": float(rounding),
                "net_pnl_rounding_bound": float(net),
                "baseline_m5_only_net": float(base_net),
                "incremental_vs_m5_only": float(net - base_net),
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("Trailing replay produced no detail rows")
    return detail


def _max_drawdown(values):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return np.nan
    curve = np.cumsum(x)
    peak = np.maximum.accumulate(np.r_[0.0, curve])[:-1]
    return float(np.min(curve - peak))


def _aggregate(detail: pd.DataFrame, immediate_join_net: float):
    rows = []
    for variant, g0 in detail.groupby("variant", sort=False):
        g = g0.sort_values(["close_time", "ticker", "tail"], kind="mergesort").copy()
        qty = float(g["entry_filled_qty"].sum())
        pqty = float(g["passive_exit_qty"].sum())
        trig = g[g["triggered"]]
        full = g[g["passive_exit_full"]]
        net = float(g["net_pnl_rounding_bound"].sum())
        base = float(g["baseline_m5_only_net"].sum())
        rows.append({
            "variant": variant,
            "positions": int(len(g)),
            "triggered_positions": int(g["triggered"].sum()),
            "trigger_rate": float(g["triggered"].mean()),
            "post_only_rejects": int(g["post_only_reject"].sum()),
            "median_running_high_c_at_trigger": float(pd.to_numeric(trig["running_high_c_at_trigger"], errors="coerce").median()) if len(trig) else np.nan,
            "median_trigger_bid_c": float(pd.to_numeric(trig["trigger_bid_c"], errors="coerce").median()) if len(trig) else np.nan,
            "median_trigger_ask_c": float(pd.to_numeric(trig["trigger_ask_c"], errors="coerce").median()) if len(trig) else np.nan,
            "median_quote_c": float(pd.to_numeric(trig["quote_c"], errors="coerce").median()) if len(trig) else np.nan,
            "entry_filled_qty": qty,
            "passive_exit_qty": pqty,
            "passive_exit_fraction_of_entry_qty": pqty / qty if qty > EPS else np.nan,
            "full_passive_exit_positions": int(len(full)),
            "full_passive_exit_rate": float(len(full) / len(g)) if len(g) else np.nan,
            "median_seconds_trigger_to_full_exit": float(pd.to_numeric(full["seconds_trigger_to_full_passive_exit"], errors="coerce").dropna().median()) if len(full) else np.nan,
            "m5_exit_qty": float(g["m5_exit_qty"].sum()),
            "m5_residual_zero_valued": float(g["m5_residual_zero_valued"].sum()),
            "m5_taker_fees": float(g["m5_taker_fee"].sum()),
            "rounding_drag": float(g["rounding_drag"].sum()),
            "total_net_pnl_rounding_bound": net,
            "m5_only_baseline_net": base,
            "incremental_vs_m5_only": net - base,
            "immediate_join_ask_benchmark_net": immediate_join_net,
            "incremental_vs_immediate_join_ask": net - immediate_join_net if np.isfinite(immediate_join_net) else np.nan,
            "positive_incremental_positions_vs_m5": int((g["incremental_vs_m5_only"] > EPS).sum()),
            "negative_incremental_positions_vs_m5": int((g["incremental_vs_m5_only"] < -EPS).sum()),
            "strategy_max_drawdown": _max_drawdown(g["net_pnl_rounding_bound"].to_numpy(float)),
        })
    return pd.DataFrame(rows).sort_values("total_net_pnl_rounding_bound", ascending=False).reset_index(drop=True)


def _by_asset(detail: pd.DataFrame):
    rows = []
    for (variant, series), g in detail.groupby(["variant", "series"], sort=True):
        rows.append({
            "variant": variant,
            "series": series,
            "positions": int(len(g)),
            "triggered": int(g["triggered"].sum()),
            "passive_exit_qty": float(g["passive_exit_qty"].sum()),
            "net_pnl_rounding_bound": float(g["net_pnl_rounding_bound"].sum()),
            "incremental_vs_m5_only": float(g["incremental_vs_m5_only"].sum()),
        })
    return pd.DataFrame(rows)


def run_trailing_passive_exit_dev(source_session, *, hard_bind=True, rebuild_cache=False, show=True):
    source = Path(source_session).resolve()
    if hard_bind and source.name != HARD_BOUND_SESSION:
        raise RuntimeError(f"Expected development source {HARD_BOUND_SESSION}, got {source.name}")
    if hard_bind and "mm_event_m0_m5_oos_cycle_q10_v1" not in str(source.parent):
        raise RuntimeError("V6 is hard-bound to the 24h development capture root")

    required = [
        source / "book_top3_events.jsonl",
        source / "trades_event_time.jsonl",
        source / "market_metadata.jsonl",
        source / "fee_preflight.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + " | ".join(missing))

    if show:
        print("=" * 150)
        print("DEEP-TAIL CAUSAL TRAILING PASSIVE EXIT DEV V6 — FAST CACHE")
        print("=" * 150)
        print("Source:", source)
        print("Entry: Q5 @ 5c, both tails, unchanged")
        print("Minimum running high before trigger: 7c")
        print("Variants:", [x[0] for x in VARIANTS])
        print("Exit: JOIN contemporaneous ask at causal rollover trigger; fixed quote; M5 fallback")
        print("DEVELOPMENT ONLY — 15h validation is NOT read")
        print()

    meta = V1._metadata(source)
    anchors, anchor_source, anchor_mode = _load_or_build_anchors(source, meta, show=show)
    anchors = anchors[pd.to_numeric(anchors["entry_filled_qty"], errors="coerce") > EPS].copy()
    if "coverage_eligible" in anchors.columns:
        anchors = anchors[anchors["coverage_eligible"].astype(bool)].copy()
    anchors = anchors.reset_index(drop=True)
    if anchors.empty:
        raise RuntimeError("No eligible cached entry anchors")

    bbo, trades_df, m5, trade_stats, cache_mode = _load_or_build_fast_cache(
        source, anchors, rebuild=rebuild_cache, show=show
    )
    immediate_join_net, join_source = _load_v4_join_benchmark(source.name)

    if show:
        print("Evaluating causal trailing triggers on compact cache...")
    detail = _evaluate(source, anchors, bbo, trades_df, m5, show=show)
    surface = _aggregate(detail, immediate_join_net)
    by_asset = _by_asset(detail)

    out = _new_output(source.name)
    detail.to_csv(out / "trailing_exit_detail.csv", index=False)
    surface.to_csv(out / "trailing_exit_surface.csv", index=False)
    by_asset.to_csv(out / "trailing_exit_by_asset.csv", index=False)

    best = surface.iloc[0].to_dict() if len(surface) else {}
    summary = {
        "version": VERSION,
        "source": str(source),
        "research_stage": "DEVELOPMENT_ONLY",
        "entry_c": ENTRY_C,
        "qty": QTY,
        "minimum_running_high_c": MIN_RUNNING_HIGH_C,
        "variants": [x[0] for x in VARIANTS],
        "anchors": int(len(anchors)),
        "anchor_source": anchor_source,
        "anchor_mode": anchor_mode,
        "cache_mode": cache_mode,
        "cached_bbo_rows": int(len(bbo)),
        "cached_trade_rows": int(len(trades_df)),
        "trade_clock_stats": trade_stats,
        "immediate_join_ask_benchmark_net": immediate_join_net,
        "immediate_join_ask_benchmark_source": join_source,
        "best_development_row": best,
        "source_modified": False,
        "exchange_api_called": False,
        "orders_sent": False,
        "guardrail": (
            "Trailing-exit mechanics were developed after the 15h Q5 validation was opened. "
            "That 15h sample cannot be used as independent validation of any V6 variant."
        ),
    }
    _atomic_json(out / "summary.json", summary)

    if show:
        print("=" * 150)
        print("TRAILING PASSIVE EXIT DEVELOPMENT SURFACE")
        print("=" * 150)
        cols = [
            "variant", "positions", "triggered_positions", "trigger_rate", "post_only_rejects",
            "median_running_high_c_at_trigger", "median_trigger_bid_c", "median_trigger_ask_c",
            "median_quote_c", "passive_exit_qty", "passive_exit_fraction_of_entry_qty",
            "full_passive_exit_positions", "full_passive_exit_rate",
            "median_seconds_trigger_to_full_exit", "m5_exit_qty", "m5_residual_zero_valued",
            "m5_taker_fees", "rounding_drag", "total_net_pnl_rounding_bound",
            "incremental_vs_m5_only", "incremental_vs_immediate_join_ask", "strategy_max_drawdown",
        ]
        print(surface[cols].to_string(index=False))
        print()
        print("M5-only benchmark is recomputed position-by-position inside each variant.")
        print("Immediate JOIN_ASK V4 benchmark:", f"{immediate_join_net:+.5f}" if np.isfinite(immediate_join_net) else "UNAVAILABLE")
        print("Best DEVELOPMENT row:")
        print(best)
        print()
        print("CACHE MODE:", cache_mode)
        print("Output:", out)
        print("SOURCE MODIFIED: NO | API CALLED: NO | ORDERS SENT: NO")

    return {
        "summary": summary,
        "surface": surface,
        "detail": detail,
        "by_asset": by_asset,
        "output_dir": str(out),
    }
