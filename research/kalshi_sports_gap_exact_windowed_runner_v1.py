from __future__ import annotations

"""Standalone isolated runner for the windowed sports exact-trade verifier."""

import argparse, importlib, sys, types
from pathlib import Path

VERSION = "KALSHI_SPORTS_EXACT_WINDOWED_STANDALONE_RUNNER_V1"
LIVE_MARKERS = (
    "mm_deep_tail_join_ask_live", "mm_deep_tail_join_ask_q50",
    "mm_cycle_q10_live_strategy", "primary_shadow_trader", "quant_research.kalshi.live",
)


def _install_shims(repo_root: Path) -> None:
    src = repo_root/"src"; qroot = src/"quant_research"; kroot = qroot/"kalshi"
    if not kroot.exists(): raise RuntimeError(f"Kalshi source tree missing: {kroot}")
    q = types.ModuleType("quant_research"); q.__path__ = [str(qroot)]; q.__package__ = "quant_research"; sys.modules["quant_research"] = q
    k = types.ModuleType("quant_research.kalshi"); k.__path__ = [str(kroot)]; k.__package__ = "quant_research.kalshi"; sys.modules["quant_research.kalshi"] = k


def _assert_no_live_imports() -> None:
    bad = [name for name in sys.modules if name.startswith("quant_research.kalshi") and any(m in name for m in LIVE_MARKERS)]
    if bad: raise RuntimeError(f"Isolation failure: live modules imported: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--deadband", type=float, default=0.005)
    ap.add_argument("--max-quote-spread", type=float, default=0.20)
    ap.add_argument("--pre-pad-s", type=int, default=300)
    ap.add_argument("--post-pad-s", type=int, default=300)
    ap.add_argument("--max-markets", type=int, default=250)
    a = ap.parse_args()

    repo_root = Path(a.repo_root).resolve(); run_dir = Path(a.run_dir).resolve()
    _install_shims(repo_root)
    mod = importlib.import_module("quant_research.kalshi.sports_two_team_gap_exact_trades_v2_windowed")
    _assert_no_live_imports()

    print("="*128)
    print("SPORTS EXACT WINDOWED — STANDALONE ISOLATED RUNNER")
    print("="*128)
    print("runner_version:", VERSION)
    print("module_version:", mod.VERSION)
    print("repo_root:     ", repo_root)
    print("run_dir:       ", run_dir)
    print("python:        ", sys.executable)
    print()

    check = mod.static_self_check(show=True)
    if not check.get("ok"): raise RuntimeError("Exact-windowed static check failed")
    _assert_no_live_imports()

    mod.run(
        run_dir=run_dir,
        requests_per_second=a.rps,
        cross_deadband=a.deadband,
        max_quote_spread=a.max_quote_spread,
        pre_pad_s=a.pre_pad_s,
        post_pad_s=a.post_pad_s,
        max_markets=a.max_markets,
        resume=True,
        show=True,
    )
    _assert_no_live_imports()
    print("\nSPORTS EXACT WINDOWED RUN COMPLETE")


if __name__ == "__main__":
    main()
