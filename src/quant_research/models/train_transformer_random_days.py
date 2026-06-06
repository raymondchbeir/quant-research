from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.models.train_sequence_walk_forward import (
    train_transformer_one_fold,
)
from quant_research.utils.paths import PROCESSED_DATA_DIR


def create_random_day_split(
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_name: str = "random_day_split_5min",
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    random_seed: int = 42,
) -> Path:
    """
    Create one random-day train/validation split from the development period.

    The most recent test_fraction of dates is still excluded as locked test.
    From the remaining development dates, validation days are sampled randomly.

    All rows from a date stay together.
    """
    processed_dir = Path(processed_dir)

    input_path = processed_dir / f"model_features_{timeframe}.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing model feature file: {input_path}")

    df = pd.read_parquet(input_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
    df["date"] = df["timestamp_pt"].dt.date

    df = df.sort_values("timestamp").reset_index(drop=True)

    unique_dates = np.array(sorted(df["date"].unique()))
    total_days = len(unique_dates)

    test_start_index = int(total_days * (1 - test_fraction))

    development_dates = unique_dates[:test_start_index]
    locked_test_dates = unique_dates[test_start_index:]

    rng = np.random.default_rng(random_seed)

    shuffled_dev_dates = development_dates.copy()
    rng.shuffle(shuffled_dev_dates)

    validation_day_count = int(
        round(len(development_dates) * validation_fraction)
    )

    validation_dates = np.sort(shuffled_dev_dates[:validation_day_count])
    train_dates = np.sort(shuffled_dev_dates[validation_day_count:])

    split_directory = processed_dir / output_name
    fold_directory = split_directory / "fold_01"

    fold_directory.mkdir(parents=True, exist_ok=True)

    train_df = df.loc[df["date"].isin(train_dates)].copy()
    validation_df = df.loc[df["date"].isin(validation_dates)].copy()
    locked_test_df = df.loc[df["date"].isin(locked_test_dates)].copy()

    train_path = fold_directory / "train.parquet"
    validation_path = fold_directory / "validation.parquet"
    locked_test_path = split_directory / "locked_test.parquet"

    train_df.to_parquet(train_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    locked_test_df.to_parquet(locked_test_path, index=False)

    manifest = {
        "timeframe": timeframe,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "random_seed": random_seed,
        "total_days": int(total_days),
        "development_days": int(len(development_dates)),
        "train_days": int(len(train_dates)),
        "validation_days": int(len(validation_dates)),
        "locked_test_days": int(len(locked_test_dates)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "locked_test_rows": int(len(locked_test_df)),
        "train_valid_targets": int((train_df["label_status"] == "valid").sum()),
        "validation_valid_targets": int(
            (validation_df["label_status"] == "valid").sum()
        ),
        "locked_test_valid_targets": int(
            (locked_test_df["label_status"] == "valid").sum()
        ),
        "train_start": str(train_dates[0]),
        "train_end": str(train_dates[-1]),
        "validation_start": str(validation_dates[0]),
        "validation_end": str(validation_dates[-1]),
        "locked_test_start": str(locked_test_dates[0]),
        "locked_test_end": str(locked_test_dates[-1]),
    }

    with (split_directory / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print("Created random-day split")
    print(f"Split directory: {split_directory}")
    print(f"Train days: {len(train_dates):,}")
    print(f"Validation days: {len(validation_dates):,}")
    print(f"Locked test days: {len(locked_test_dates):,}")
    print(f"Train valid targets: {manifest['train_valid_targets']:,}")
    print(f"Validation valid targets: {manifest['validation_valid_targets']:,}")
    print(f"Locked test valid targets: {manifest['locked_test_valid_targets']:,}")

    return split_directory


def train_transformer_random_day_split(
    experiment_name: str,
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    split_name: str = "random_day_split_5min",
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    random_seed: int = 42,
    recreate_split: bool = True,
    sequence_length: int = 21,
    batch_size: int = 256,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.10,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-6,
    max_epochs: int = 30,
    patience: int = 5,
    use_class_weights: bool = False,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Train one transformer model using a random-day train/validation split.

    This is a diagnostic experiment, not a replacement for walk-forward.
    """
    processed_dir = Path(processed_dir)

    split_directory = processed_dir / split_name

    if recreate_split or not split_directory.exists():
        split_directory = create_random_day_split(
            timeframe=timeframe,
            processed_dir=processed_dir,
            output_name=split_name,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            random_seed=random_seed,
        )

    fold_directory = split_directory / "fold_01"

    output_directory = (
        processed_dir
        / f"transformer_random_days_{timeframe}"
        / experiment_name
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    metrics = train_transformer_one_fold(
        fold_directory=fold_directory,
        output_directory=output_directory,
        sequence_length=sequence_length,
        batch_size=batch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        use_class_weights=use_class_weights,
        device=device,
        random_seed=random_seed,
    )

    summary_df = pd.DataFrame([metrics])

    summary_path = output_directory / "random_day_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    with (output_directory / "aggregate_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    print("\nRandom-day transformer result")
    print(
        summary_df[
            [
                "fold",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "log_loss",
                "multiclass_brier_score",
                "best_epoch",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved random-day result to: {output_directory}")

    return summary_df