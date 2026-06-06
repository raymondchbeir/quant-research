from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.utils.paths import PROCESSED_DATA_DIR


CLASS_NAMES = ["neutral", "up", "down"]


def load_validation_predictions(results_dir: Path) -> pd.DataFrame:
    """
    Load validation prediction parquet files from every walk-forward fold.

    Expected folder structure:
        results_dir/
            fold_01/
                validation_predictions.parquet
            fold_02/
                validation_predictions.parquet
            ...
    """
    prediction_paths = sorted(
        results_dir.glob("fold_*/validation_predictions.parquet")
    )

    if not prediction_paths:
        raise FileNotFoundError(
            f"No validation prediction files found in {results_dir}"
        )

    prediction_dfs = []

    for path in prediction_paths:
        fold_name = path.parent.name

        fold_df = pd.read_parquet(path)
        fold_df["fold"] = fold_name

        prediction_dfs.append(fold_df)

    predictions = pd.concat(
        prediction_dfs,
        axis=0,
        ignore_index=True,
    )

    required_columns = [
        "timestamp",
        "timestamp_pt",
        "date",
        "target_name",
        "target_class",
        "p_neutral",
        "p_up",
        "p_down",
        "fold",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Prediction files are missing required columns:\n"
            f"{missing_columns}"
        )

    return predictions


def apply_threshold_rule(
    predictions: pd.DataFrame,
    edge_threshold: float,
    min_direction_probability: float | None = None,
) -> pd.DataFrame:
    """
    Apply a simple probability edge rule.

    Long signal:
        P(up) - P(down) >= edge_threshold

    Short signal:
        P(down) - P(up) >= edge_threshold

    Flat:
        otherwise

    Optional:
        min_direction_probability requires P(up) or P(down)
        to be above a minimum absolute probability.
    """
    df = predictions.copy()

    up_edge = df["p_up"] - df["p_down"]
    down_edge = df["p_down"] - df["p_up"]

    long_mask = up_edge >= edge_threshold
    short_mask = down_edge >= edge_threshold

    if min_direction_probability is not None:
        long_mask = long_mask & (
            df["p_up"] >= min_direction_probability
        )

        short_mask = short_mask & (
            df["p_down"] >= min_direction_probability
        )

    df["signal"] = "flat"
    df.loc[long_mask, "signal"] = "long"
    df.loc[short_mask, "signal"] = "short"

    return df


def summarize_threshold_result(
    threshold_df: pd.DataFrame,
    edge_threshold: float,
    min_direction_probability: float | None = None,
) -> dict:
    """
    Summarize signal quality for one threshold setting.
    """
    total_rows = len(threshold_df)

    long_df = threshold_df.loc[
        threshold_df["signal"] == "long"
    ]

    short_df = threshold_df.loc[
        threshold_df["signal"] == "short"
    ]

    flat_df = threshold_df.loc[
        threshold_df["signal"] == "flat"
    ]

    signal_df = threshold_df.loc[
        threshold_df["signal"].isin(["long", "short"])
    ]

    long_count = len(long_df)
    short_count = len(short_df)
    flat_count = len(flat_df)
    signal_count = len(signal_df)

    long_precision = (
        (long_df["target_name"] == "up").mean()
        if long_count > 0
        else np.nan
    )

    short_precision = (
        (short_df["target_name"] == "down").mean()
        if short_count > 0
        else np.nan
    )

    directional_precision = (
        (
            ((signal_df["signal"] == "long")
             & (signal_df["target_name"] == "up"))
            |
            ((signal_df["signal"] == "short")
             & (signal_df["target_name"] == "down"))
        ).mean()
        if signal_count > 0
        else np.nan
    )

    long_base_rate = (
        (threshold_df["target_name"] == "up").mean()
    )

    short_base_rate = (
        (threshold_df["target_name"] == "down").mean()
    )

    directional_base_rate = (
        threshold_df["target_name"].isin(["up", "down"]).mean()
    )

    return {
        "edge_threshold": edge_threshold,
        "min_direction_probability": min_direction_probability,
        "total_rows": total_rows,

        "long_count": long_count,
        "short_count": short_count,
        "flat_count": flat_count,
        "signal_count": signal_count,

        "long_rate": long_count / total_rows,
        "short_rate": short_count / total_rows,
        "flat_rate": flat_count / total_rows,
        "signal_rate": signal_count / total_rows,

        "long_precision": long_precision,
        "short_precision": short_precision,
        "directional_precision": directional_precision,

        "long_base_rate": long_base_rate,
        "short_base_rate": short_base_rate,
        "directional_base_rate": directional_base_rate,

        "long_edge_over_base": (
            long_precision - long_base_rate
            if not np.isnan(long_precision)
            else np.nan
        ),
        "short_edge_over_base": (
            short_precision - short_base_rate
            if not np.isnan(short_precision)
            else np.nan
        ),
        "directional_edge_over_base": (
            directional_precision - directional_base_rate
            if not np.isnan(directional_precision)
            else np.nan
        ),
    }


def summarize_threshold_by_fold(
    threshold_df: pd.DataFrame,
    edge_threshold: float,
    min_direction_probability: float | None = None,
) -> pd.DataFrame:
    """
    Summarize threshold result separately for each walk-forward fold.
    """
    rows = []

    for fold_name, fold_df in threshold_df.groupby("fold", sort=True):
        summary = summarize_threshold_result(
            threshold_df=fold_df,
            edge_threshold=edge_threshold,
            min_direction_probability=min_direction_probability,
        )

        summary["fold"] = fold_name
        rows.append(summary)

    return pd.DataFrame(rows)


def run_threshold_sweep(
    experiment_name: str,
    results_dir: Path,
    output_dir: Path,
    edge_thresholds: list[float] | None = None,
    min_direction_probabilities: list[float | None] | None = None,
) -> pd.DataFrame:
    """
    Run a threshold sweep over saved XGBoost validation probabilities.

    Saves:
        threshold_sweep_summary.csv
        threshold_sweep_by_fold.csv
    """
    if edge_thresholds is None:
        edge_thresholds = [
            0.00,
            0.025,
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.175,
            0.20,
            0.25,
            0.30,
        ]

    if min_direction_probabilities is None:
        min_direction_probabilities = [
            None,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
        ]

    predictions = load_validation_predictions(results_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    fold_summary_dfs = []

    for min_prob in min_direction_probabilities:
        for edge_threshold in edge_thresholds:
            threshold_df = apply_threshold_rule(
                predictions=predictions,
                edge_threshold=edge_threshold,
                min_direction_probability=min_prob,
            )

            summary = summarize_threshold_result(
                threshold_df=threshold_df,
                edge_threshold=edge_threshold,
                min_direction_probability=min_prob,
            )

            summary["experiment_name"] = experiment_name
            summary_rows.append(summary)

            fold_summary = summarize_threshold_by_fold(
                threshold_df=threshold_df,
                edge_threshold=edge_threshold,
                min_direction_probability=min_prob,
            )

            fold_summary["experiment_name"] = experiment_name
            fold_summary_dfs.append(fold_summary)

    summary_df = pd.DataFrame(summary_rows)

    by_fold_df = pd.concat(
        fold_summary_dfs,
        axis=0,
        ignore_index=True,
    )

    summary_path = output_dir / "threshold_sweep_summary.csv"
    by_fold_path = output_dir / "threshold_sweep_by_fold.csv"

    summary_df.to_csv(summary_path, index=False)
    by_fold_df.to_csv(by_fold_path, index=False)

    print(f"Experiment: {experiment_name}")
    print(f"Loaded predictions: {len(predictions):,} rows")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved fold results to: {by_fold_path}")

    print("\nTop settings by directional precision:")
    display_columns = [
        "edge_threshold",
        "min_direction_probability",
        "signal_rate",
        "long_precision",
        "short_precision",
        "directional_precision",
        "directional_edge_over_base",
    ]

    print(
        summary_df
        .sort_values(
            ["directional_precision", "signal_rate"],
            ascending=[False, False],
        )[display_columns]
        .head(20)
        .to_string(index=False)
    )

    return summary_df


def run_default_xgboost_threshold_sweeps(
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Run threshold sweeps for the default XGBoost experiment folders.

    Expected folders:
        xgboost_walk_forward_5min
        xgboost_walk_forward_5min_balanced
    """
    processed_dir = Path(processed_dir)

    experiments = {
        "lagged_unweighted": (
            processed_dir
            / f"xgboost_walk_forward_{timeframe}"
        ),
        "lagged_balanced": (
            processed_dir
            / f"xgboost_walk_forward_{timeframe}_balanced"
        ),
    }

    outputs = {}

    for experiment_name, results_dir in experiments.items():
        if not results_dir.exists():
            print(
                f"Skipping {experiment_name}: "
                f"missing directory {results_dir}"
            )
            continue

        output_dir = (
            processed_dir
            / f"threshold_sweep_{timeframe}"
            / experiment_name
        )

        outputs[experiment_name] = run_threshold_sweep(
            experiment_name=experiment_name,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    return outputs