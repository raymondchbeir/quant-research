from __future__ import annotations

"""Read-only postmortem for Q50 orphan-order failures + fresh M1->M12 replay.

NO API CALLS. NO ORDERS.

Part A reconstructs the local/exchange lifecycle of the confirmed orphan resting
order from a stopped deep-tail live session.

Part B replays the historical frozen deep-tail capacity model on that session's
M0->M12 raw_capture, with Q fixed to 50 and the pre-registered extension boundary
M1 (60s) -> M12 (720s). The historical mechanics are loaded verbatim from commit
498aee5b8a55f8ddbc27597c5bb27ffd302d23fc and only START_E and Q_GRID are patched.

Important: M0 is the recorder boundary. The hypothetical strategy still begins at
M1. This script does not test a new M0-entry strategy.
"""

import ast
import json
import re
import subprocess
import types
from pathlib import Path

import numpy as np
import pandas as pd

from . import recorder_core as C


VERSION = "MM_Q50_ORPHAN_DIAG_M1_M12_REPLAY_V1"
HISTORICAL_COMMIT = "498aee5b8a55f8ddbc27597c5bb27ffd302d23fc"
HISTORICAL_PATH = "src/quant_research/kalshi/mm_deep_tail_m5_m12_capacity_fast_v1.py"
Q = 50
START_E = 60.0
END_E = 720.0
OUTPUT_DIRNAME = "postmortem_q50_orphan_m1_m12_v1"


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _iter_jsonl(path: Path):
    try:
        fh = Path(path).open("r", encoding="utf-8", errors="replace")
    except Exception:
        return
    with fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield i, row


def _extract_orphan(final_summary: dict, health: dict):
    msg = str(final_summary.get("last_error") or health.get("last_error") or "")
    orphan_rows = []
    marker = "confirmed orphan strategy resting orders:"
    if marker in msg:
        tail = msg.split(marker, 1)[1].strip()
        try:
            obj = ast.literal_eval(tail)
            if isinstance(obj, list):
                orphan_rows = [x for x in obj if isinstance(x, dict)]
        except Exception:
            pass

    if not orphan_rows:
        oid = re.search(r"['\"]order_id['\"]\s*:\s*['\"]([^'\"]+)", msg)
        ticker = re.search(r"['\"]ticker['\"]\s*:\s*['\"]([^'\"]+)", msg)
        if oid or ticker:
            orphan_rows = [{
                "order_id": oid.group(1) if oid else None,
                "ticker": ticker.group(1) if ticker else None,
            }]
    return orphan_rows, msg


def _row_order_ids(row):
    ids = set()

    def rec(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in {"order_id", "client_order_id", "cid"} and v not in (None, ""):
                    ids.add(str(v))
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(row)
    return ids


def _pick(row, *keys):
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return row.get(k)
    tr = row.get("track")
    if isinstance(tr, dict):
        for k in keys:
            if k in tr and tr.get(k) not in (None, ""):
                return tr.get(k)
    raw = row.get("raw")
    if isinstance(raw, dict):
        for k in keys:
            if k in raw and raw.get(k) not in (None, ""):
                return raw.get(k)
    return None


def diagnose_orphan(session, *, show=True):
    session = Path(session).resolve()
    outdir = session / OUTPUT_DIRNAME
    outdir.mkdir(parents=True, exist_ok=True)

    final_summary = _read_json(session / "final_summary.json", {})
    health = _read_json(session / "health.json", {})
    orphans, error_msg = _extract_orphan(final_summary, health)

    if not orphans:
        raise RuntimeError("No confirmed orphan order could be extracted from final summary/health.")

    orphan = orphans[0]
    target_ticker = str(orphan.get("ticker") or "")
    target_oid = str(orphan.get("order_id") or "")
    target_cid = str(orphan.get("client_order_id") or "")
    needles = {x for x in (target_ticker, target_oid, target_cid) if x}

    rows = []
    # Strategy/private/audit files only. Deliberately exclude raw_capture's huge public files.
    for path in sorted(session.glob("*.jsonl")):
        for line_no, row in _iter_jsonl(path):
            blob = json.dumps(row, default=str, separators=(",", ":"))
            if needles and not any(n in blob for n in needles):
                continue
            ids = _row_order_ids(row)
            rows.append({
                "file": path.name,
                "line": line_no,
                "time": _pick(row, "time", "recv_time", "ts", "timestamp"),
                "event": _pick(row, "event", "action", "kind", "reason"),
                "action": _pick(row, "action"),
                "kind": _pick(row, "kind"),
                "role": _pick(row, "role"),
                "tail": _pick(row, "tail"),
                "ticker": _pick(row, "ticker"),
                "order_id": _pick(row, "order_id"),
                "client_order_id": _pick(row, "client_order_id", "cid"),
                "status": _pick(row, "status"),
                "price": _pick(row, "price", "yes_price", "yes_price_dollars", "no_price_dollars"),
                "qty": _pick(row, "qty", "quantity", "initial_count_fp"),
                "fill_count": _pick(row, "fill_count", "fill_count_fp", "effective_fill", "processed_fill"),
                "remaining": _pick(row, "remaining", "remaining_count", "remaining_count_fp"),
                "source": _pick(row, "source"),
                "matched_order_ids": ",".join(sorted(ids)),
                "row_json": blob,
            })

    timeline = pd.DataFrame(rows)
    if len(timeline):
        timeline["time_parsed"] = pd.to_datetime(timeline["time"], utc=True, errors="coerce")
        timeline.sort_values(["time_parsed", "file", "line"], inplace=True, kind="mergesort")
        timeline.reset_index(drop=True, inplace=True)

    timeline.to_csv(outdir / "orphan_timeline.csv", index=False)

    summary = {
        "version": VERSION,
        "session": str(session),
        "shutdown_reason": final_summary.get("shutdown_reason") or health.get("shutdown_reason"),
        "error_message": error_msg,
        "orphan": orphan,
        "target_ticker": target_ticker,
        "target_order_id": target_oid,
        "timeline_rows": int(len(timeline)),
        "final_positions": final_summary.get("final_positions"),
        "final_strategy_resting_orders": final_summary.get("final_strategy_resting_orders"),
        "flat_verified": final_summary.get("flat_verified"),
        "strategy_resting_orders_zero": final_summary.get("strategy_resting_orders_zero"),
        "account_pnl_usd": final_summary.get("account_pnl_usd"),
        "exchange_orphan_initial_qty": orphan.get("initial_count_fp"),
        "exchange_orphan_fill_qty": orphan.get("fill_count_fp"),
        "exchange_orphan_remaining_qty": orphan.get("remaining_count_fp"),
        "api_called": False,
        "orders_sent": False,
    }
    (outdir / "orphan_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if show:
        print("=" * 118)
        print("PART A — CONFIRMED ORPHAN ORDER POSTMORTEM — READ ONLY")
        print("=" * 118)
        for k in [
            "shutdown_reason", "target_ticker", "target_order_id",
            "exchange_orphan_initial_qty", "exchange_orphan_fill_qty",
            "exchange_orphan_remaining_qty", "timeline_rows", "flat_verified",
            "strategy_resting_orders_zero", "account_pnl_usd",
        ]:
            print(f"{k:38s}: {summary.get(k)}")
        print("\nTIMELINE")
        if len(timeline):
            cols = [
                "time", "file", "event", "action", "kind", "role", "tail",
                "status", "price", "qty", "fill_count", "remaining", "source",
                "order_id", "client_order_id",
            ]
            cols = [c for c in cols if c in timeline.columns]
            with pd.option_context("display.max_rows", 200, "display.max_colwidth", 120, "display.width", 220):
                print(timeline[cols].to_string(index=False))
        else:
            print("No matching local JSONL rows found.")
        print("\nSaved:", outdir / "orphan_timeline.csv")
    return {"summary": summary, "timeline": timeline, "outdir": outdir}


def _load_historical_parent():
    repo = Path(C.PROJECT_ROOT).resolve()
    spec = f"{HISTORICAL_COMMIT}:{HISTORICAL_PATH}"
    proc = subprocess.run(
        ["git", "show", spec], cwd=repo, text=True, capture_output=True, check=False
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"Could not load frozen historical replay source: {spec}\n{proc.stderr[-4000:]}")

    mod = types.ModuleType("quant_research.kalshi._q50_frozen_capacity_parent")
    mod.__file__ = f"git:{spec}"
    mod.__package__ = "quant_research.kalshi"
    exec(compile(proc.stdout, mod.__file__, "exec"), mod.__dict__)

    mod.VERSION = VERSION
    mod.START_E = START_E
    mod.END_E = END_E
    mod.WINDOW_S = END_E - START_E
    mod.Q_GRID = (Q,)
    return mod


def _fee_multipliers_from_live_session(session: Path):
    child = _read_json(session / "child_fee_preflight_reuse_v2_8_2.json", {})
    mult = child.get("multipliers") or {}
    if mult:
        return {str(k): float(v) for k, v in mult.items()}, str(session / "child_fee_preflight_reuse_v2_8_2.json")

    parent = _read_json(session / "parent_preflight_snapshot.json", {})
    fee = parent.get("fee_preflight") or {}
    mult = fee.get("multipliers") or {}
    if mult:
        return {str(k): float(v) for k, v in mult.items()}, str(session / "parent_preflight_snapshot.json")

    raise RuntimeError("No persisted fee multipliers found in stopped live session.")


def replay_q50_m1_m12(session, *, show=True):
    session = Path(session).resolve()
    raw = session / "raw_capture"
    outdir = session / OUTPUT_DIRNAME
    outdir.mkdir(parents=True, exist_ok=True)

    required = [
        raw / "market_metadata.jsonl",
        raw / "trades_event_time.jsonl",
        raw / "book_top3_events.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"M0->M12 raw capture incomplete/missing files: {missing}")

    cap = _read_json(raw / "capture_spec.json", {})
    persisted = cap.get("persisted_elapsed_seconds") or []
    if persisted and len(persisted) >= 2 and float(persisted[1]) + 1e-9 < END_E:
        raise RuntimeError(f"Recorder metadata does not reach M12: persisted={persisted}")

    p = _load_historical_parent()
    fee_mult, fee_source = _fee_multipliers_from_live_session(session)

    if show:
        print("\n" + "=" * 118)
        print("PART B — HYPOTHETICAL Q50 M1->M12 REPLAY ON FRESH M0->M12 CAPTURE")
        print("=" * 118)
        print("NO API CALLS / NO ORDERS")
        print("Historical mechanics commit:", HISTORICAL_COMMIT)
        print("Entry activation window:    M1=60s -> M12=720s")
        print("Requested quantity:         Q50")
        print("Fee source:                 ", fee_source)

    meta = p._metadata(raw)
    candidates, entry_stats = p._scan_entries(raw, meta, show=show)
    coverage, decision, terminal, book_stats = p._scan_books_fast(
        raw, meta, candidates, show=show
    )
    passive, passive_stats = p._scan_passive_exits(
        raw, meta, candidates, decision, coverage, terminal, show=show
    )
    detail = p._economics(
        session.name, candidates, coverage, decision, terminal, passive, fee_mult
    )
    curve = p._curve(detail, meta)

    detail.to_csv(outdir / "q50_m1_m12_detail.csv", index=False)
    curve.to_csv(outdir / "q50_m1_m12_curve.csv", index=False)

    q50 = curve[curve["Q"] == Q].copy()
    if len(q50) != 1:
        raise RuntimeError(f"Expected exactly one Q50 curve row, got {len(q50)}")
    row = q50.iloc[0].to_dict()

    filled = detail[(detail["Q"] == Q) & (detail["entry_filled_qty"] > p.EPS)].copy()
    filled.sort_values(["first_fill_obs_s", "ticker"], inplace=True, kind="mergesort")
    filled.to_csv(outdir / "q50_m1_m12_filled_markets.csv", index=False)

    summary = {
        "version": VERSION,
        "session": str(session),
        "raw_capture": str(raw),
        "historical_parent_commit": HISTORICAL_COMMIT,
        "start_e_s": START_E,
        "end_e_s": END_E,
        "Q": Q,
        "fee_source": fee_source,
        "entry_stats": entry_stats,
        "book_stats": book_stats,
        "passive_stats": passive_stats,
        "curve_q50": row,
        "api_called": False,
        "orders_sent": False,
    }
    (outdir / "q50_m1_m12_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    if show:
        print("\n" + "-" * 118)
        print("Q50 M1->M12 RESULT")
        print("-" * 118)
        for k in [
            "eligible_markets", "entry_events", "full_entries", "partial_entries",
            "requested_contracts", "entry_filled_qty", "entry_fill_fraction_requested",
            "passive_exit_qty", "passive_exit_fraction_entry", "full_passive_positions",
            "m12_required_qty", "m12_exit_qty", "terminal_exit_fraction",
            "m12_residual_zero_valued", "total_net_pnl", "net_cents_per_filled_contract",
            "max_drawdown", "cancel_race_markets", "missing_decision_snapshots",
        ]:
            if k in row:
                print(f"{k:42s}: {row.get(k)}")

        print("\nFILLED MARKETS")
        if len(filled):
            cols = [
                "ticker", "series", "tail", "entry_filled_qty", "full_entry",
                "passive_exit_qty", "exit_quote", "m12_required_qty", "m12_exit_qty",
                "m12_residual_zero_valued", "m12_taker_fee", "net_pnl",
                "coverage_fraction", "cancel_race_100ms",
            ]
            cols = [c for c in cols if c in filled.columns]
            with pd.option_context("display.max_rows", 200, "display.width", 220):
                print(filled[cols].to_string(index=False))
        else:
            print("No Q50 fills in eligible complete M1->M12 windows.")

        print("\nSaved:")
        print(" ", outdir / "q50_m1_m12_curve.csv")
        print(" ", outdir / "q50_m1_m12_detail.csv")
        print(" ", outdir / "q50_m1_m12_filled_markets.csv")

    return {
        "summary": summary,
        "curve": curve,
        "detail": detail,
        "filled": filled,
        "outdir": outdir,
    }


def run(session, *, show=True):
    session = Path(session).resolve()
    diag = diagnose_orphan(session, show=show)
    replay = replay_q50_m1_m12(session, show=show)
    return {"diagnostic": diag, "replay": replay}


__all__ = [
    "VERSION",
    "HISTORICAL_COMMIT",
    "Q",
    "START_E",
    "END_E",
    "diagnose_orphan",
    "replay_q50_m1_m12",
    "run",
]
