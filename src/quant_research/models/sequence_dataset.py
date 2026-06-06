from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from quant_research.features.feature_builder import MODEL_FEATURES


class SequenceClassificationDataset(Dataset):
    """
    PyTorch dataset for sequence classification.

    X shape per sample:
        [sequence_length, num_features]

    y:
        target_class at the final row of the sequence.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        metadata: pd.DataFrame,
    ):
        self.sequences = torch.tensor(
            sequences,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

        self.metadata = metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


def load_split_dataframe(path: Path) -> pd.DataFrame:
    """
    Load one walk-forward split parquet file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_parquet(path)

    required_columns = [
        "timestamp",
        "timestamp_pt",
        "date",
        "label_status",
        "target_name",
        "target_class",
        *MODEL_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns:\n"
            f"{missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
    df["date"] = df["timestamp_pt"].dt.date

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def build_sequences_from_dataframe(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: list[str] = MODEL_FEATURES,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Fast sequence builder.

    Builds sequences within each trading day using numpy sliding windows.

    Context rows may include warmup/non-valid rows, but the final
    target row must have label_status == valid.

    The sequence ends at the target row:
        [t - sequence_length + 1, ..., t]
    """
    all_sequences = []
    all_labels = []
    all_metadata = []

    metadata_columns = [
        "timestamp",
        "timestamp_pt",
        "date",
        "target_name",
        "target_class",
    ]

    for _, day_df in df.groupby("date", sort=False):
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)

        if len(day_df) < sequence_length:
            continue

        features = day_df[feature_columns].to_numpy(dtype=np.float32)

        # Shape:
        # [num_possible_windows, sequence_length, num_features]
        windows = np.lib.stride_tricks.sliding_window_view(
            features,
            window_shape=sequence_length,
            axis=0,
        )

        # sliding_window_view gives:
        # [num_windows, num_features, sequence_length]
        # so we transpose it to:
        # [num_windows, sequence_length, num_features]
        windows = np.transpose(windows, (0, 2, 1))

        # The end row for each window is:
        # sequence_length - 1, sequence_length, ...
        end_positions = np.arange(
            sequence_length - 1,
            len(day_df),
        )

        valid_target_mask = (
            day_df["label_status"].eq("valid")
            & day_df["target_class"].notna()
        ).to_numpy()

        usable_mask = valid_target_mask[end_positions]

        if not usable_mask.any():
            continue

        usable_windows = windows[usable_mask]

        # Drop sequences with NaN/inf values.
        finite_mask = np.isfinite(usable_windows).all(axis=(1, 2))

        if not finite_mask.any():
            continue

        usable_windows = usable_windows[finite_mask]

        usable_end_positions = end_positions[usable_mask][finite_mask]

        labels = (
            day_df
            .loc[usable_end_positions, "target_class"]
            .astype(int)
            .to_numpy(dtype=np.int64)
        )

        metadata = (
            day_df
            .loc[usable_end_positions, metadata_columns]
            .copy()
            .reset_index(drop=True)
        )

        all_sequences.append(usable_windows.astype(np.float32))
        all_labels.append(labels)
        all_metadata.append(metadata)

    if not all_sequences:
        raise ValueError(
            "No valid sequences were built. "
            "Check sequence_length and label_status."
        )

    sequences_array = np.concatenate(all_sequences, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    metadata = pd.concat(
        all_metadata,
        axis=0,
        ignore_index=True,
    )

    return sequences_array, labels_array, metadata
def fit_sequence_scaler(train_sequences: np.ndarray) -> StandardScaler:
    """
    Fit StandardScaler using only training sequence tokens.

    Input:
        [num_sequences, sequence_length, num_features]

    Scaler fit shape:
        [num_sequences * sequence_length, num_features]
    """
    num_sequences, sequence_length, num_features = train_sequences.shape

    flattened = train_sequences.reshape(
        num_sequences * sequence_length,
        num_features,
    )

    scaler = StandardScaler()
    scaler.fit(flattened)

    return scaler


def transform_sequences(
    sequences: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply a fitted scaler to sequence tokens.
    """
    num_sequences, sequence_length, num_features = sequences.shape

    flattened = sequences.reshape(
        num_sequences * sequence_length,
        num_features,
    )

    scaled = scaler.transform(flattened)

    return scaled.reshape(
        num_sequences,
        sequence_length,
        num_features,
    ).astype(np.float32)


def build_train_validation_datasets(
    train_path: Path,
    validation_path: Path,
    sequence_length: int,
    feature_columns: list[str] = MODEL_FEATURES,
):
    """
    Build train/validation sequence datasets and metadata.

    The scaler is fitted only on the training sequences.
    """
    train_df = load_split_dataframe(train_path)
    validation_df = load_split_dataframe(validation_path)

    train_sequences, train_labels, train_metadata = (
        build_sequences_from_dataframe(
            df=train_df,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
        )
    )

    validation_sequences, validation_labels, validation_metadata = (
        build_sequences_from_dataframe(
            df=validation_df,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
        )
    )

    scaler = fit_sequence_scaler(train_sequences)

    train_sequences = transform_sequences(
        train_sequences,
        scaler,
    )

    validation_sequences = transform_sequences(
        validation_sequences,
        scaler,
    )

    train_dataset = SequenceClassificationDataset(
        sequences=train_sequences,
        labels=train_labels,
        metadata=train_metadata,
    )

    validation_dataset = SequenceClassificationDataset(
        sequences=validation_sequences,
        labels=validation_labels,
        metadata=validation_metadata,
    )

    return train_dataset, validation_dataset, scaler