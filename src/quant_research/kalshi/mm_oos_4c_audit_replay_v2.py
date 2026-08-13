from __future__ import annotations

"""Compatibility-fixed wrapper for the frozen 4c OOS audit/replay.

V1 called ``mm_defensive_m1_m5_v1._simulate_contract`` even though the
existing Defensive V1 implementation exposes the exact simulator as
``_simulate_contract_defensive``.  This module fixes only that API-name
mismatch and delegates the complete audit/replay logic to V1.

No strategy, quality gate, queue model, thresholds, data handling, or OOS
semantics are changed.
"""

from . import mm_defensive_m1_m5_v1 as D
from . import mm_oos_4c_audit_replay as V1

STUDY_VERSION = "MM4C_FROZEN_OOS_AUDIT_REPLAY_V2_API_FIX"


def run_frozen_mm4c_oos_audit_replay(*args, **kwargs):
    """Run V1 after installing the correct Defensive V1 simulator alias."""
    previous = getattr(D, "_simulate_contract", None)
    D._simulate_contract = D._simulate_contract_defensive
    try:
        return V1.run_frozen_mm4c_oos_audit_replay(*args, **kwargs)
    finally:
        if previous is None:
            try:
                delattr(D, "_simulate_contract")
            except AttributeError:
                pass
        else:
            D._simulate_contract = previous
