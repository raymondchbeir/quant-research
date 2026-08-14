from __future__ import annotations

"""Frozen OOS #1 replay using the EXACT development pre-M1 feature semantics.

Why this wrapper exists
-----------------------
The first OOS replay implementation added an 80% per-contract pre-M1 coverage
requirement and an 80% per-window contract-coverage requirement. Those gates
were NOT present in the development code that produced the <23c candidate.

The development feature was:
1. for each M1-M5 quality contract, use every valid observed 1 Hz midpoint in
   scheduled M0 <= t < M1;
2. if at least one valid midpoint exists, contract range = max(mid)-min(mid);
3. window feature = max finite contract range across the contracts in that
   15-minute window;
4. if no contract has a finite range, the window feature is missing;
5. frozen candidate passes iff window feature < 23c.

This module restores those exact semantics. It does NOT change the 80% M1-M5
execution-quality gate, the <23c cutoff, NAT4->2 execution, sizing, inventory,
momentum, flow, cooldown, fill assumptions, or fees.

It remains hard-bound to OOS #1 (20260813_190334) through the V1 verifier and
never reads a later recording.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import mm_nat4_to_2_prem1_lt23_frozen_oos1_v1 as V1
from . import mm_oos_4c_compact_recorder_v2 as R
from . import mm_reconstructed_m1_m5_backtest as B

STUDY_VERSION = "NAT4_TO_2_PREM1_LT23_FROZEN_OOS1_EXACT_DEV_SEM_V2"
EPS = 1e-9


def _window_gate_exact_development_semantics(audit: pd.DataFrame) -> pd.DataFrame:
    """Match the development window-diagnostic feature construction exactly."""
    m1 = audit[audit["m1_m5_quality_ok"].astype(bool)].copy()
    rows = []

    for close_ts, g in m1.groupby("close_ts", sort=True):
        ranges = pd.to_numeric(g["pre_mid_range_c"], errors="coerce")
        finite = np.isfinite(ranges.to_numpy(float))

        n_m1 = int(len(g))
        n_pre = int(finite.sum())
        diagnostic_cov = 100.0 * n_pre / n_m1 if n_m1 else 0.0

        if n_pre:
            max_range = float(ranges[finite].max())
            feature_ok = True
        else:
            max_range = np.nan
            feature_ok = False

        passed = bool(
            feature_ok
            and max_range < V1.PRE_RANGE_CUTOFF_C - EPS
        )

        rows.append({
            "close_ts": float(close_ts),
            "close_time": B._iso(float(close_ts)),
            "m1_m5_quality_contracts": n_m1,
            "pre_feature_contracts": n_pre,
            # Diagnostic only. This is NOT a gate in exact development semantics.
            "pre_contract_coverage_pct": diagnostic_cov,
            "pre_m0_m1_max_mid_range_c": max_range,
            "feature_quality_ok": feature_ok,
            "range_pass_lt23c": passed,
            "range_fail_ge23c": bool(feature_ok and not passed),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "close_ts", "close_time", "m1_m5_quality_contracts",
            "pre_feature_contracts", "pre_contract_coverage_pct",
            "pre_m0_m1_max_mid_range_c", "feature_quality_ok",
            "range_pass_lt23c", "range_fail_ge23c",
        ])

    return pd.DataFrame(rows).sort_values("close_ts").reset_index(drop=True)


def run_nat4_to_2_prem1_lt23_frozen_oos1_exact_dev_semantics(
    session_dir,
    output_dir=None,
    *,
    show=True,
):
    """Run OOS #1 once using the frozen feature exactly as developed."""
    session = Path(session_dir).resolve()

    if output_dir is None:
        output_dir = (
            R.PROJECT_ROOT
            / "results"
            / "kalshi_nat4_to_2_prem1_lt23_frozen_oos1_exact_dev_sem"
            / f"{session.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )

    old_gate = V1._window_gate
    V1._window_gate = _window_gate_exact_development_semantics

    try:
        print("=" * 132)
        print("OOS #1 — EXACT DEVELOPMENT PRE-M1 FEATURE SEMANTICS")
        print("No pre-M1 80% coverage gate; M1-M5 80% quality gate remains unchanged.")
        print("Frozen rule remains strictly pre-M1 max midpoint range < 23c.")
        print("=" * 132)

        study = V1.run_nat4_to_2_prem1_lt23_frozen_oos1(
            session_dir=session,
            output_dir=output_dir,
            show=show,
        )
    finally:
        V1._window_gate = old_gate

    out = Path(study["output_dir"])
    cfg_path = out / "study_config.json"
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    cfg.update({
        "study_version": STUDY_VERSION,
        "pre_m1_feature_semantics": (
            "EXACT DEVELOPMENT: per-contract range over all valid observed M0-M1 "
            "1Hz midpoint samples; window feature is max finite contract range"
        ),
        "pre_contract_quality_gate_pct": None,
        "pre_window_contract_gate_pct": None,
        "pre_m1_coverage_is_diagnostic_only": True,
        "feature_missing_rule": "window missing only if every M1-M5 quality contract has no finite M0-M1 range",
        "m1_m5_quality_gate_pct": V1.M1_M5_QUALITY_GATE_PCT,
        "pre_m1_range_cutoff_c": V1.PRE_RANGE_CUTOFF_C,
        "correction_reason": (
            "Removed post-development 80% pre-M1 coverage gates after source-code audit "
            "showed they were absent from the development feature definition. Correction "
            "was made before observing OOS #1 strategy PnL."
        ),
        "later_recording_accessed": False,
        "status": "FROZEN_OOS1_EXACT_DEVELOPMENT_SEMANTICS_NO_TUNING",
    })
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    # Save an explicit provenance note next to the result.
    (out / "feature_semantics_provenance.txt").write_text(
        "Pre-M1 feature semantics intentionally match the development code exactly.\n"
        "No 80% pre-M1 contract or window coverage gate is applied.\n"
        "M1-M5 execution quality still requires >=80% valid 1Hz coverage.\n"
        "Frozen cutoff remains strictly <23c. No threshold sweep was run.\n",
        encoding="utf-8",
    )

    return study
