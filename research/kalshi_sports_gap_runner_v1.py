from __future__ import annotations

"""Standalone runner for isolated Kalshi sports gap research.

This file is intentionally outside quant_research.kalshi so executing it does NOT
run quant_research/kalshi/__init__.py. It injects lightweight package shims only so
relative imports inside the sports-only modules resolve without importing live Q50
modules.

Safety:
- public Kalshi GETs only through sports modules;
- no account/portfolio/order endpoints;
- no orders;
- no live Q50 module imports;
- no control-file writes;
- designed for a separate worktree/process while live Q50 continues elsewhere.
"""

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

VERSION = "KALSHI_SPORTS_GAP_STANDALONE_RUNNER_V1_4_FOOTBALL_RANDOM_SAMPLE"

LIVE_MARKERS = (
    "mm_deep_tail_join_ask_live",
    "mm_deep_tail_join_ask_q50",
    "mm_cycle_q10_live_strategy",
    "primary_shadow_trader",
    "quant_research.kalshi.live",
)


def _install_package_shims(repo_root: Path) -> None:
    src = repo_root / "src"
    qroot = src / "quant_research"
    kroot = qroot / "kalshi"
    if not kroot.exists():
        raise RuntimeError(f"Kalshi source tree missing: {kroot}")

    q = types.ModuleType("quant_research")
    q.__path__ = [str(qroot)]
    q.__package__ = "quant_research"
    sys.modules["quant_research"] = q

    k = types.ModuleType("quant_research.kalshi")
    k.__path__ = [str(kroot)]
    k.__package__ = "quant_research.kalshi"
    sys.modules["quant_research.kalshi"] = k


def _assert_no_live_imports() -> None:
    bad = [
        name for name in sys.modules
        if name.startswith("quant_research.kalshi")
        and any(marker in name for marker in LIVE_MARKERS)
    ]
    if bad:
        raise RuntimeError(f"Isolation failure: live modules imported: {bad}")


def load_modules(repo_root: Path):
    _install_package_shims(repo_root)
    backfill = importlib.import_module(
        "quant_research.kalshi.sports_two_team_gap_backfill_v1_4_football_random_sample"
    )
    analysis = importlib.import_module(
        "quant_research.kalshi.sports_two_team_gap_analysis_v1"
    )
    exact = importlib.import_module(
        "quant_research.kalshi.sports_two_team_gap_exact_trades_v1"
    )
    _assert_no_live_imports()
    return backfill, analysis, exact


def run(*, repo_root: Path, run_dir: Path, lookback_days: int = 365,
        max_events: int = 250, public_rps: float = 1.0,
        exact_rps: float = 0.75, max_exact_markets: int = 250) -> dict:
    repo_root = Path(repo_root).resolve()
    run_dir = Path(run_dir).resolve()

    # The backfill owns creation of the run directory and deliberately uses
    # exist_ok=False so a prior/partial run can never be silently overwritten.
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise RuntimeError(
            f"Fresh sports run directory already exists: {run_dir}. "
            "Choose a new run_dir; refusing to overwrite a prior/partial study."
        )

    backfill, analysis, exact = load_modules(repo_root)

    print("=" * 128)
    print("SPORTS GAP STUDY — STANDALONE ISOLATED RUNNER")
    print("=" * 128)
    print("runner_version:", VERSION)
    print("repo_root:     ", repo_root)
    print("run_dir:       ", run_dir)
    print("python:        ", sys.executable)
    print()

    b = backfill.static_self_check(show=True)
    a = analysis.static_self_check(show=True)
    e = exact.static_self_check(show=True)

    if not (b.get("ok") and a.get("ok") and e.get("ok")):
        raise RuntimeError("Sports module static check failed")
    _assert_no_live_imports()

    result_backfill = backfill.run(
        run_dir=run_dir,
        lookback_days=lookback_days,
        max_events=max_events,
        requests_per_second=public_rps,
        pregame_hours=24.0,
        max_game_hours=12.0,
        show=True,
    )
    _assert_no_live_imports()

    result_analysis = analysis.run(
        run_dir=run_dir,
        max_quote_spread=0.20,
        cross_deadband=0.005,
        show=True,
    )
    _assert_no_live_imports()

    result_exact = exact.run(
        run_dir=run_dir,
        requests_per_second=exact_rps,
        cross_deadband=0.005,
        max_markets=max_exact_markets,
        show=True,
    )
    _assert_no_live_imports()

    out = {
        "runner_version": VERSION,
        "python": sys.executable,
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "lookback_days": int(lookback_days),
        "max_events": int(max_events),
        "public_rps": float(public_rps),
        "exact_rps": float(exact_rps),
        "max_exact_markets": int(max_exact_markets),
        "orders_sent": False,
        "live_modules_imported": False,
        "backfill": result_backfill,
        "analysis": result_analysis,
        "exact": result_exact,
    }
    (run_dir / "standalone_runner_summary.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("\n" + "=" * 128)
    print("SPORTS GAP STUDY COMPLETE")
    print("=" * 128)
    print("output:", run_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--lookback-days", type=int, default=365)
    ap.add_argument("--max-events", type=int, default=250)
    ap.add_argument("--public-rps", type=float, default=1.0)
    ap.add_argument("--exact-rps", type=float, default=0.75)
    ap.add_argument("--max-exact-markets", type=int, default=250)
    a = ap.parse_args()
    run(
        repo_root=Path(a.repo_root),
        run_dir=Path(a.run_dir),
        lookback_days=a.lookback_days,
        max_events=a.max_events,
        public_rps=a.public_rps,
        exact_rps=a.exact_rps,
        max_exact_markets=a.max_exact_markets,
    )


if __name__ == "__main__":
    main()
