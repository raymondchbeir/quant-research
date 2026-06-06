from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.utils.paths import PROCESSED_DATA_DIR


VOLUME_MICROSTRUCTURE_FEATURES = [
    "relative_volume_by_time",
    "volume_zscore_24",
    "signed_volume_ratio_12",
    "lower_wick_absorption",
    "below_vwap_x_volume_z12_x_close_location",
]


def _prepare_market_dataframe(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize timestamps and required base columns.
    """
    df = market_df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
    df["date"] = df["timestamp_pt"].dt.date

    df = df.sort_values("timestamp").reset_index(drop=True)

    required_columns = [
        "timestamp",
        "timestamp_pt",
        "date",
        "NVDA_open",
        "NVDA_high",
        "NVDA_low",
        "NVDA_close",
        "NVDA_volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Market dataframe is missing required columns: "
            f"{missing_columns}"
        )

    return df


def _prepare_split_dataframe(split_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize timestamps in train/validation/test split files.
    """
    df = split_df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if "timestamp_pt" in df.columns:
        df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
        df["date"] = df["timestamp_pt"].dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        raise ValueError(
            "Split dataframe must contain timestamp_pt or date."
        )

    return df.sort_values("timestamp").reset_index(drop=True)


def calculate_slot_volume_means(reference_market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average volume by intraday bar number using reference data only.

    For walk-forward folds, reference data should be the training split only.
    For locked test, reference data should be final train only.

    This avoids leaking validation/test volume patterns into the feature.
    """
    reference = _prepare_market_dataframe(reference_market_df)

    reference["bar_number"] = reference.groupby("date").cumcount() + 1

    slot_volume_means = (
        reference
        .groupby("bar_number")["NVDA_volume"]
        .mean()
        .rename("slot_volume_mean")
        .reset_index()
    )

    return slot_volume_means


def compute_volume_microstructure_features(
    market_df: pd.DataFrame,
    slot_volume_means: pd.DataFrame,
    below_vwap_cutoff: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Compute causal 5-minute volume microstructure features.

    Features:
        1. relative_volume_by_time
        2. volume_zscore_24
        3. signed_volume_ratio_12
        4. lower_wick_absorption
        5. below_vwap_x_volume_z12_x_close_location

    Returns:
        feature dataframe with timestamp + feature columns,
        below_vwap_cutoff used for below_vwap_extreme.
    """
    df = _prepare_market_dataframe(market_df)

    df["price"] = df["NVDA_close"]
    df["bar_number"] = df.groupby("date").cumcount() + 1

    # ------------------------------------------------------------
    # Candle shape
    # ------------------------------------------------------------
    df["candle_direction"] = np.sign(
        df["NVDA_close"] - df["NVDA_open"]
    )

    df["candle_range"] = df["NVDA_high"] - df["NVDA_low"]

    df["close_location"] = np.where(
        df["candle_range"] > 0,
        (df["NVDA_close"] - df["NVDA_low"]) / df["candle_range"],
        0.5,
    )

    df["lower_wick"] = (
        np.minimum(df["NVDA_open"], df["NVDA_close"])
        - df["NVDA_low"]
    )

    df["lower_wick_ratio"] = np.where(
        df["candle_range"] > 0,
        df["lower_wick"] / df["candle_range"],
        0.0,
    )

    # ------------------------------------------------------------
    # Causal VWAP so far today
    # ------------------------------------------------------------
    df["typical_price"] = (
        df["NVDA_high"] + df["NVDA_low"] + df["NVDA_close"]
    ) / 3

    df["dollar_volume_proxy"] = (
        df["typical_price"] * df["NVDA_volume"]
    )

    df["cum_dollar_volume"] = (
        df.groupby("date")["dollar_volume_proxy"]
        .cumsum()
    )

    df["cum_volume"] = (
        df.groupby("date")["NVDA_volume"]
        .cumsum()
    )

    df["vwap_so_far"] = (
        df["cum_dollar_volume"] / df["cum_volume"]
    )

    df["deviation_from_vwap"] = (
        df["price"] / df["vwap_so_far"] - 1
    )

    # ------------------------------------------------------------
    # 1. Relative volume by time of day
    # ------------------------------------------------------------
    slot_volume_means = slot_volume_means.copy()
    df = df.merge(
        slot_volume_means,
        on="bar_number",
        how="left",
    )

    df["relative_volume_by_time"] = (
        df["NVDA_volume"] / df["slot_volume_mean"]
    )

    df["relative_volume_by_time"] = (
        df["relative_volume_by_time"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    # ------------------------------------------------------------
    # 2. Volume z-score, previous bars only
    # ------------------------------------------------------------
    df["rolling_volume_mean_24"] = (
        df.groupby("date")["NVDA_volume"]
        .transform(lambda x: x.rolling(24, min_periods=24).mean().shift(1))
    )

    df["rolling_volume_std_24"] = (
        df.groupby("date")["NVDA_volume"]
        .transform(lambda x: x.rolling(24, min_periods=24).std().shift(1))
    )

    df["volume_zscore_24"] = (
        (df["NVDA_volume"] - df["rolling_volume_mean_24"])
        / df["rolling_volume_std_24"]
    )

    df["volume_zscore_24"] = (
        df["volume_zscore_24"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Need zscore_12 only as an intermediate for feature #5.
    df["rolling_volume_mean_12"] = (
        df.groupby("date")["NVDA_volume"]
        .transform(lambda x: x.rolling(12, min_periods=12).mean().shift(1))
    )

    df["rolling_volume_std_12"] = (
        df.groupby("date")["NVDA_volume"]
        .transform(lambda x: x.rolling(12, min_periods=12).std().shift(1))
    )

    df["volume_zscore_12"] = (
        (df["NVDA_volume"] - df["rolling_volume_mean_12"])
        / df["rolling_volume_std_12"]
    )

    df["volume_zscore_12"] = (
        df["volume_zscore_12"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # ------------------------------------------------------------
    # 3. Signed volume rolling 12
    # ------------------------------------------------------------
    df["signed_volume"] = (
        df["candle_direction"] * df["NVDA_volume"]
    )

    df["rolling_signed_volume_12"] = (
        df.groupby("date")["signed_volume"]
        .transform(lambda x: x.rolling(12, min_periods=12).sum().shift(1))
    )

    df["rolling_total_volume_12"] = (
        df.groupby("date")["NVDA_volume"]
        .transform(lambda x: x.rolling(12, min_periods=12).sum().shift(1))
    )

    df["signed_volume_ratio_12"] = (
        df["rolling_signed_volume_12"]
        / df["rolling_total_volume_12"]
    )

    df["signed_volume_ratio_12"] = (
        df["signed_volume_ratio_12"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # ------------------------------------------------------------
    # 4. Lower wick absorption
    # ------------------------------------------------------------
    df["lower_wick_absorption"] = (
        df["lower_wick_ratio"] * df["relative_volume_by_time"]
    )

    df["lower_wick_absorption"] = (
        df["lower_wick_absorption"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # ------------------------------------------------------------
    # 5. Below VWAP x volume z-score x close location
    # ------------------------------------------------------------
    if below_vwap_cutoff is None:
        below_vwap_cutoff = float(
            df["deviation_from_vwap"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .quantile(0.10)
        )

    df["below_vwap_extreme"] = (
        df["deviation_from_vwap"] <= below_vwap_cutoff
    )

    df["below_vwap_x_volume_z12_x_close_location"] = (
        df["below_vwap_extreme"].astype(float)
        * df["volume_zscore_12"].clip(lower=0.0)
        * df["close_location"]
    )

    df["below_vwap_x_volume_z12_x_close_location"] = (
        df["below_vwap_x_volume_z12_x_close_location"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    feature_df = df[
        ["timestamp"] + VOLUME_MICROSTRUCTURE_FEATURES
    ].copy()

    for column in VOLUME_MICROSTRUCTURE_FEATURES:
        feature_df[column] = (
            feature_df[column]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )

    return feature_df, float(below_vwap_cutoff)


def _merge_features_into_split(
    split_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge engineered features into an existing model split.
    """
    split = _prepare_split_dataframe(split_df)

    existing_feature_columns = [
        column
        for column in VOLUME_MICROSTRUCTURE_FEATURES
        if column in split.columns
    ]

    if existing_feature_columns:
        split = split.drop(columns=existing_feature_columns)

    augmented = split.merge(
        feature_df,
        on="timestamp",
        how="left",
    )

    missing_counts = augmented[VOLUME_MICROSTRUCTURE_FEATURES].isna().sum()

    if missing_counts.sum() > 0:
        print("Missing volume feature counts after merge:")
        print(missing_counts)

    for column in VOLUME_MICROSTRUCTURE_FEATURES:
        augmented[column] = (
            augmented[column]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )

    return augmented


def augment_walk_forward_splits_with_volume_features(
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    input_name: str = "walk_forward_5min",
    output_name: str = "walk_forward_volume_5min",
) -> Path:
    """
    Add volume microstructure features to walk-forward folds.

    Fast version:
        - load market once
        - index market by timestamp once
        - select fold rows with .loc[timestamps] instead of repeated isin()

    For each fold:
        - fit slot-volume means on that fold's train split only
        - fit below-VWAP cutoff on that fold's train split only
        - transform train and validation using those train-only stats

    This avoids validation leakage.
    """
    processed_dir = Path(processed_dir)

    input_dir = processed_dir / input_name
    output_dir = processed_dir / output_name
    market_path = processed_dir / f"featured_{timeframe}.parquet"

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input fold directory: {input_dir}")

    if not market_path.exists():
        raise FileNotFoundError(f"Missing market file: {market_path}")

    print(f"Loading market data from: {market_path}")
    market = _prepare_market_dataframe(pd.read_parquet(market_path))

    # Fast timestamp lookup.
    market_indexed = (
        market
        .drop_duplicates(subset=["timestamp"], keep="last")
        .set_index("timestamp")
        .sort_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(
        path
        for path in input_dir.glob("fold_*")
        if path.is_dir()
    )

    if not fold_dirs:
        raise ValueError(f"No fold directories found in {input_dir}")

    manifest_rows = []

    for fold_number, fold_dir in enumerate(fold_dirs, start=1):
        fold_name = fold_dir.name

        print(f"\n[{fold_number}/{len(fold_dirs)}] Processing {fold_name}")

        train_path = fold_dir / "train.parquet"
        validation_path = fold_dir / "validation.parquet"

        if not train_path.exists() or not validation_path.exists():
            raise FileNotFoundError(
                f"Fold {fold_name} is missing train/validation parquet files."
            )

        train_split = _prepare_split_dataframe(pd.read_parquet(train_path))
        validation_split = _prepare_split_dataframe(pd.read_parquet(validation_path))

        train_timestamps = pd.Index(train_split["timestamp"])
        validation_timestamps = pd.Index(validation_split["timestamp"])

        missing_train = train_timestamps.difference(market_indexed.index)
        missing_validation = validation_timestamps.difference(market_indexed.index)

        if len(missing_train) > 0:
            raise ValueError(
                f"{fold_name}: {len(missing_train)} train timestamps "
                f"are missing from market data. First few: "
                f"{list(missing_train[:5])}"
            )

        if len(missing_validation) > 0:
            raise ValueError(
                f"{fold_name}: {len(missing_validation)} validation timestamps "
                f"are missing from market data. First few: "
                f"{list(missing_validation[:5])}"
            )

        # Fast lookup by timestamp, then restore timestamp as a column.
        train_market = (
            market_indexed
            .loc[train_timestamps]
            .reset_index()
        )

        validation_market = (
            market_indexed
            .loc[validation_timestamps]
            .reset_index()
        )

        slot_volume_means = calculate_slot_volume_means(train_market)

        train_features, below_vwap_cutoff = compute_volume_microstructure_features(
            market_df=train_market,
            slot_volume_means=slot_volume_means,
            below_vwap_cutoff=None,
        )

        validation_features, _ = compute_volume_microstructure_features(
            market_df=validation_market,
            slot_volume_means=slot_volume_means,
            below_vwap_cutoff=below_vwap_cutoff,
        )

        augmented_train = _merge_features_into_split(
            split_df=train_split,
            feature_df=train_features,
        )

        augmented_validation = _merge_features_into_split(
            split_df=validation_split,
            feature_df=validation_features,
        )

        fold_output_dir = output_dir / fold_name
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        augmented_train.to_parquet(
            fold_output_dir / "train.parquet",
            index=False,
        )

        augmented_validation.to_parquet(
            fold_output_dir / "validation.parquet",
            index=False,
        )

        fold_manifest = {
            "fold": fold_name,
            "train_rows": int(len(augmented_train)),
            "validation_rows": int(len(augmented_validation)),
            "below_vwap_cutoff": float(below_vwap_cutoff),
            "feature_count_added": int(len(VOLUME_MICROSTRUCTURE_FEATURES)),
            "features_added": VOLUME_MICROSTRUCTURE_FEATURES,
        }

        with (fold_output_dir / "volume_feature_manifest.json").open("w") as file:
            json.dump(fold_manifest, file, indent=2)

        manifest_rows.append(fold_manifest)

        print(
            f"{fold_name}: saved "
            f"train={len(augmented_train):,}, "
            f"validation={len(augmented_validation):,}, "
            f"cutoff={below_vwap_cutoff:.6f}"
        )

    manifest = {
        "timeframe": timeframe,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "features_added": VOLUME_MICROSTRUCTURE_FEATURES,
        "folds": manifest_rows,
    }

    with (output_dir / "volume_feature_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print(f"\nSaved volume-augmented walk-forward splits to: {output_dir}")

    return output_dir


def augment_locked_test_split_with_volume_features(
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    input_name: str = "locked_test_transformer_split_5min",
    output_name: str = "locked_test_transformer_split_volume_5min",
) -> Path:
    """
    Add volume microstructure features to final locked-test split.

    Fast version:
        - load market once
        - index market by timestamp once
        - select split rows with .loc[timestamps]

    Uses final train only to fit:
        - slot volume by time of day
        - below VWAP cutoff

    Then applies those stats to early_stop and locked_test.
    """
    processed_dir = Path(processed_dir)

    input_dir = processed_dir / input_name
    output_dir = processed_dir / output_name
    market_path = processed_dir / f"featured_{timeframe}.parquet"

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing locked split directory: {input_dir}")

    if not market_path.exists():
        raise FileNotFoundError(f"Missing market file: {market_path}")

    print(f"Loading market data from: {market_path}")
    market = _prepare_market_dataframe(pd.read_parquet(market_path))

    market_indexed = (
        market
        .drop_duplicates(subset=["timestamp"], keep="last")
        .set_index("timestamp")
        .sort_index()
    )

    train_split = _prepare_split_dataframe(
        pd.read_parquet(input_dir / "train.parquet")
    )
    early_split = _prepare_split_dataframe(
        pd.read_parquet(input_dir / "early_stop.parquet")
    )
    locked_split = _prepare_split_dataframe(
        pd.read_parquet(input_dir / "locked_test.parquet")
    )

    def select_market_rows(split: pd.DataFrame, split_name: str) -> pd.DataFrame:
        timestamps = pd.Index(split["timestamp"])
        missing = timestamps.difference(market_indexed.index)

        if len(missing) > 0:
            raise ValueError(
                f"{split_name}: {len(missing)} timestamps are missing "
                f"from market data. First few: {list(missing[:5])}"
            )

        return (
            market_indexed
            .loc[timestamps]
            .reset_index()
        )

    train_market = select_market_rows(train_split, "train")
    early_market = select_market_rows(early_split, "early_stop")
    locked_market = select_market_rows(locked_split, "locked_test")

    slot_volume_means = calculate_slot_volume_means(train_market)

    train_features, below_vwap_cutoff = compute_volume_microstructure_features(
        market_df=train_market,
        slot_volume_means=slot_volume_means,
        below_vwap_cutoff=None,
    )

    early_features, _ = compute_volume_microstructure_features(
        market_df=early_market,
        slot_volume_means=slot_volume_means,
        below_vwap_cutoff=below_vwap_cutoff,
    )

    locked_features, _ = compute_volume_microstructure_features(
        market_df=locked_market,
        slot_volume_means=slot_volume_means,
        below_vwap_cutoff=below_vwap_cutoff,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    augmented_train = _merge_features_into_split(
        split_df=train_split,
        feature_df=train_features,
    )

    augmented_early = _merge_features_into_split(
        split_df=early_split,
        feature_df=early_features,
    )

    augmented_locked = _merge_features_into_split(
        split_df=locked_split,
        feature_df=locked_features,
    )

    augmented_train.to_parquet(output_dir / "train.parquet", index=False)
    augmented_early.to_parquet(output_dir / "early_stop.parquet", index=False)
    augmented_locked.to_parquet(output_dir / "locked_test.parquet", index=False)

    manifest = {
        "timeframe": timeframe,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "below_vwap_cutoff": float(below_vwap_cutoff),
        "features_added": VOLUME_MICROSTRUCTURE_FEATURES,
        "train_rows": int(len(augmented_train)),
        "early_stop_rows": int(len(augmented_early)),
        "locked_test_rows": int(len(augmented_locked)),
    }

    with (output_dir / "volume_feature_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print("Saved volume-augmented locked-test split")
    print(json.dumps(manifest, indent=2))

    return output_dir