from __future__ import annotations

import numpy as np
import pandas as pd

from . import mm_nat4_to_2_pre_m1_range_monotonicity_v1 as V1


def _perm_p(x, observed):
    rng = np.random.default_rng(V1.SEED)
    xr = pd.Series(x[V1.FEATURE]).rank().to_numpy(float).copy()
    yr = pd.Series(x[V1.PNL]).rank().to_numpy(float).copy()
    xr -= xr.mean()
    yr -= yr.mean()
    denom_x = np.sqrt(np.sum(xr * xr))
    exceed = 0
    for _ in range(V1.N_PERM):
        yp = rng.permutation(yr)
        den = denom_x * np.sqrt(np.sum(yp * yp))
        rho = np.sum(xr * yp) / den if den > V1.EPS else np.nan
        if np.isfinite(rho) and abs(rho) >= abs(observed) - V1.EPS:
            exceed += 1
    return (exceed + 1.0) / (V1.N_PERM + 1.0)


V1._perm_p = _perm_p
run_pre_m1_range_monotonicity = V1.run_pre_m1_range_monotonicity
