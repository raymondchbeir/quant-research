from __future__ import annotations

import pandas as pd

from . import mm_nat4_to_2_window_diagnostics_v1 as V1

STUDY_VERSION = "NAT4_TO_2_WINDOW_DIAGNOSTICS_V2"
_ORIG = V1._fill_window_features


def _fixed_fill_features(fdf, ticker_close, enriched_episodes):
    x = fdf.copy()
    if not x.empty and "inventory_at_join" not in x.columns:
        if {"episode_id", "inventory_at_join"}.issubset(enriched_episodes.columns):
            inv = enriched_episodes[["episode_id", "inventory_at_join"]].drop_duplicates("episode_id")
            x = x.merge(inv, on="episode_id", how="left", validate="many_to_one")
        else:
            x["inventory_at_join"] = pd.NA
    return _ORIG(x, ticker_close, enriched_episodes)


def run_nat4_to_2_window_diagnostics(session_dir, strategy_result_dir, output_dir=None, *, show=True):
    old = V1._fill_window_features
    V1._fill_window_features = _fixed_fill_features
    try:
        return V1.run_nat4_to_2_window_diagnostics(
            session_dir=session_dir,
            strategy_result_dir=strategy_result_dir,
            output_dir=output_dir,
            show=show,
        )
    finally:
        V1._fill_window_features = old
