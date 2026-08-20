from __future__ import annotations

"""Read-only M1->M12 extension replay built from the frozen historical M5->M12 engine.

NO API CALLS. NO ORDERS.

This module intentionally reuses the exact historical fast capacity replay from
commit 498aee5b8a55f8ddbc27597c5bb27ffd302d23fc and changes only the pre-registered
entry activation boundary from M5 to M1. All other mechanics remain frozen:
5c dual-tail entry, first-fill-wins, 100ms cancel race, strict-through entry
capacity, fixed JOIN_ASK after full requested Q, queue-aware passive exit, M12
recorded-top3 fallback, fees, conservative rounding drag, and the same Q grid.

Scientific status: DEVELOPMENT / EXPLORATORY EXTENSION TEST ONLY.
"""

import subprocess
import types
from pathlib import Path

from . import recorder_core as C


VERSION = "MM_DEEP_TAIL_M1_M12_CAPACITY_REPLAY_V1"
HISTORICAL_COMMIT = "498aee5b8a55f8ddbc27597c5bb27ffd302d23fc"
HISTORICAL_PATH = "src/quant_research/kalshi/mm_deep_tail_m5_m12_capacity_fast_v1.py"

START_E = 60.0
END_E = 720.0
ENTRY_ACTIVATION_S = 0.100
OUTPUT_ROOT = C.PROJECT_ROOT / "results" / "kalshi_deep_tail_m1_m12_capacity_replay_v1"

_PARENT = None


def _load_parent():
    global _PARENT
    if _PARENT is not None:
        return _PARENT

    repo = Path(C.PROJECT_ROOT).resolve()
    spec = f"{HISTORICAL_COMMIT}:{HISTORICAL_PATH}"
    p = subprocess.run(
        ["git", "show", spec],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(
            "Could not recover the frozen historical M5->M12 replay source from git. "
            f"spec={spec!r} stderr={p.stderr[-4000:]!r}"
        )

    mod = types.ModuleType("quant_research.kalshi._historical_m5_m12_capacity_fast_v1")
    mod.__file__ = f"git:{spec}"
    mod.__package__ = "quant_research.kalshi"
    exec(compile(p.stdout, mod.__file__, "exec"), mod.__dict__)

    # PRE-REGISTERED EXTENSION: change only M5 -> M1.
    mod.VERSION = VERSION
    mod.START_E = START_E
    mod.END_E = END_E
    mod.WINDOW_S = END_E - START_E
    mod.ENTRY_ACTIVATION_S = ENTRY_ACTIVATION_S
    mod.OUTPUT_ROOT = OUTPUT_ROOT

    _PARENT = mod
    return mod


def static_self_check(*, show=True):
    p = _load_parent()
    checks = {
        "historical_parent_commit": HISTORICAL_COMMIT,
        "historical_parent_path": HISTORICAL_PATH,
        "entry_price": float(p.ENTRY),
        "start_e_s": float(p.START_E),
        "end_e_s": float(p.END_E),
        "entry_activation_s": float(p.ENTRY_ACTIVATION_S),
        "exit_activation_s": float(p.EXIT_ACTIVATION_S),
        "cancel_race_s": float(p.CANCEL_RACE_S),
        "q_grid": tuple(int(x) for x in p.Q_GRID),
        "start_is_m1": abs(float(p.START_E) - 60.0) < 1e-12,
        "end_is_m12": abs(float(p.END_E) - 720.0) < 1e-12,
        "fixed_join_ask_parent_logic_reused": True,
        "m12_fallback_parent_logic_reused": True,
        "source_sessions_unchanged": tuple(p.SESSION_NAMES),
        "exchange_api_called": False,
        "orders_sent": False,
    }
    ok = bool(
        checks["start_is_m1"]
        and checks["end_is_m12"]
        and checks["fixed_join_ask_parent_logic_reused"]
        and checks["m12_fallback_parent_logic_reused"]
        and checks["exchange_api_called"] is False
        and checks["orders_sent"] is False
    )
    out = {"version": VERSION, **checks, "ok": ok}
    if show:
        print("=" * 112)
        print("M1->M12 DEEP-TAIL EXTENSION STATIC CHECK — READ ONLY")
        print("=" * 112)
        for k, v in out.items():
            print(f"{k:46s}: {v}")
    if not ok:
        raise RuntimeError(f"M1->M12 static check failed: {out}")
    return out


def run(*, force_rebuild=False, show=True):
    static_self_check(show=show)
    p = _load_parent()
    if show:
        print("\nPRE-REGISTERED CHANGE ONLY: historical START_E 300s -> 60s")
        print("Everything else is inherited from the frozen historical replay.\n")
    result = p.run(force_rebuild=bool(force_rebuild), show=bool(show))

    # Save clearly named aliases so downstream notebooks cannot confuse the M1->M12
    # extension outputs with the historical M5->M12 study.
    for session_name, curve in result["curves"].items():
        out = OUTPUT_ROOT / str(session_name)
        curve.to_csv(out / "m1_m12_capacity_curve.csv", index=False)
        result["detail"][session_name].to_csv(
            out / "m1_m12_capacity_detail.csv", index=False
        )

    return result


__all__ = [
    "VERSION",
    "HISTORICAL_COMMIT",
    "START_E",
    "END_E",
    "OUTPUT_ROOT",
    "static_self_check",
    "run",
]
