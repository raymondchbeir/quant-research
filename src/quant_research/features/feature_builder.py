from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.data.data_downloader import confirm_file_action
from quant_research.utils.paths import PROCESSED_DATA_DIR

#load the labled data 
def load_labeled_data( timeframe = "5min", raw_dir = PROCESSED_DATA_DIR):
    """
    Load raw Alpaca parquet data for labeled data.
    """
    # Makes the path per symbol
    path = Path(raw_dir) / f"labeled_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data for labeled : {path}")
    
    # Reading the data
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError("missing timestamp column.")
    
    #changing the timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    #returning the dataframe
    return df

def build_features(timeframe = "5min", processed_dir=PROCESSED_DATA_DIR):
    #loading data and making sure everything looks right 
    
    df = load_labeled_data(timeframe = timeframe, raw_dir = processed_dir)
    df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
    df["date"] = df["timestamp_pt"].dt.date
    df = df.sort_values("timestamp").reset_index(drop=True)
    original_columns = set(df.columns)
    #body of the wick relative to proce
    df["NVDA_body_pct"] = (
    (df["NVDA_close"] - df["NVDA_open"])
    / df["NVDA_open"]
    )

    # range relative to price
    df["NVDA_range_pct"] = (
        (df["NVDA_high"] - df["NVDA_low"])
        / df["NVDA_open"]
    )

    # upper wick reative to price
    df["NVDA_upper_wick_pct"] = (
        df["NVDA_high"]
        - df[["NVDA_open", "NVDA_close"]].max(axis=1)
    ) / df["NVDA_open"]

    #lower wick relative to price
    df["NVDA_lower_wick_pct"] = (
        df[["NVDA_open", "NVDA_close"]].min(axis=1)
        - df["NVDA_low"]
    ) / df["NVDA_open"]

    #distance from vwap 
    df["NVDA_vwap_distance"] = (
        (df["NVDA_close"] - df["NVDA_vwap"])
        / df["NVDA_close"]
    )

    #log of. the volume
    df["NVDA_log_volume"] = np.log1p(df["NVDA_volume"])


    #log of the trade count
    df["NVDA_log_trade_count"] = np.log1p(
        df["NVDA_trade_count"]
    )

    #average trade size
    df["NVDA_log_average_trade_size"] = np.log1p(
        df["NVDA_volume"]
        / df["NVDA_trade_count"].clip(lower=1)
    )


    #log dollar volume
    df["NVDA_log_dollar_volume"] = np.log1p(
        df["NVDA_volume"] * df["NVDA_vwap"]
    )

    #now the features that require grouping
    grouped = df.groupby("date", sort=False)

    #getting the previous_close etc...
    previous_close = grouped["NVDA_close"].shift(1)
    previous_volume = grouped["NVDA_volume"].shift(1)
    previous_trade_count = grouped["NVDA_trade_count"].shift(1)

    #getting the log returns 
    df["NVDA_log_return_1"] = np.log(
    df["NVDA_close"] / previous_close
    )

    #volume change
    df["NVDA_volume_change_1"] = np.log(
    (df["NVDA_volume"] + 1)
    / (previous_volume + 1)
    )

    #trade count change
    df["NVDA_trade_count_change_1"] = np.log(
    (df["NVDA_trade_count"] + 1)
    / (previous_trade_count + 1)
    )
    minutes_from_midnight = (

        df["timestamp_pt"].dt.hour * 60

        + df["timestamp_pt"].dt.minute

    )

    # Circular time-of-day encoding.

    minutes_per_day = 24 * 60

    df["time_of_day_sin"] = np.sin(

        2 * np.pi * minutes_from_midnight / minutes_per_day

    )

    df["time_of_day_cos"] = np.cos(

        2 * np.pi * minutes_from_midnight / minutes_per_day

    )

    # Position relative to the regular market session.

    market_open_minutes = 6 * 60 + 30

    regular_session_minutes = 390

    df["minutes_since_open"] = (

        minutes_from_midnight - market_open_minutes

    )

    df["session_position"] = (

        df["minutes_since_open"] / regular_session_minutes

    )

    # Circular weekday encoding.

    day_of_week = df["timestamp_pt"].dt.dayofweek

    trading_days_per_week = 5

    df["day_of_week_sin"] = np.sin(

        2 * np.pi * day_of_week / trading_days_per_week

    )

    df["day_of_week_cos"] = np.cos(

        2 * np.pi * day_of_week / trading_days_per_week

    )
    candle_range = df["NVDA_high"] - df["NVDA_low"]

    df["NVDA_close_position"] = np.where(
        candle_range > 0,
        (df["NVDA_close"] - df["NVDA_low"]) / candle_range,
        0.5,
    )
    # ---------------------------------------------------------
    # QQQ and SPY features
    # ---------------------------------------------------------

    for symbol in ["QQQ", "SPY"]:

        # Candle body relative to price
        df[f"{symbol}_body_pct"] = (
            (df[f"{symbol}_close"] - df[f"{symbol}_open"])
            / df[f"{symbol}_open"]
        )

        # Full candle range relative to price
        df[f"{symbol}_range_pct"] = (
            (df[f"{symbol}_high"] - df[f"{symbol}_low"])
            / df[f"{symbol}_open"]
        )

        # Upper wick relative to price
        df[f"{symbol}_upper_wick_pct"] = (
            df[f"{symbol}_high"]
            - df[[f"{symbol}_open", f"{symbol}_close"]].max(axis=1)
        ) / df[f"{symbol}_open"]

        # Lower wick relative to price
        df[f"{symbol}_lower_wick_pct"] = (
            df[[f"{symbol}_open", f"{symbol}_close"]].min(axis=1)
            - df[f"{symbol}_low"]
        ) / df[f"{symbol}_open"]

        # Close location inside the candle's high-low range
        candle_range = (
            df[f"{symbol}_high"] - df[f"{symbol}_low"]
        )

        df[f"{symbol}_close_position"] = np.where(
            candle_range > 0,
            (
                df[f"{symbol}_close"] - df[f"{symbol}_low"]
            ) / candle_range,
            0.5,
        )

        # Distance between close and VWAP
        df[f"{symbol}_vwap_distance"] = (
            (df[f"{symbol}_close"] - df[f"{symbol}_vwap"])
            / df[f"{symbol}_close"]
        )

        # Volume and activity features
        df[f"{symbol}_log_volume"] = np.log1p(
            df[f"{symbol}_volume"]
        )

        df[f"{symbol}_log_trade_count"] = np.log1p(
            df[f"{symbol}_trade_count"]
        )

        df[f"{symbol}_log_average_trade_size"] = np.log1p(
            df[f"{symbol}_volume"]
            / df[f"{symbol}_trade_count"].clip(lower=1)
        )

        df[f"{symbol}_log_dollar_volume"] = np.log1p(
            df[f"{symbol}_volume"] * df[f"{symbol}_vwap"]
        )

        # Previous-candle values within the same day
        previous_close = grouped[f"{symbol}_close"].shift(1)
        previous_volume = grouped[f"{symbol}_volume"].shift(1)
        previous_trade_count = grouped[f"{symbol}_trade_count"].shift(1)

        # One-candle changes
        df[f"{symbol}_log_return_1"] = np.log(
            df[f"{symbol}_close"] / previous_close
        )

        df[f"{symbol}_volume_change_1"] = np.log(
            (df[f"{symbol}_volume"] + 1)
            / (previous_volume + 1)
        )

        df[f"{symbol}_trade_count_change_1"] = np.log(
            (df[f"{symbol}_trade_count"] + 1)
            / (previous_trade_count + 1)
        )

    # ---------------------------------------------------------
    # Cross-asset relative-return features
    # ---------------------------------------------------------

    df["NVDA_minus_QQQ_log_return_1"] = (
        df["NVDA_log_return_1"]
        - df["QQQ_log_return_1"]
    )

    df["NVDA_minus_SPY_log_return_1"] = (
        df["NVDA_log_return_1"]
        - df["SPY_log_return_1"]
    )

    df["QQQ_minus_SPY_log_return_1"] = (
        df["QQQ_log_return_1"]
        - df["SPY_log_return_1"]
    )
    # ---------------------------------------------------------
    # Lagged and rolling history features for XGBoost
    # ---------------------------------------------------------

    rolling_windows = [3, 6, 12]

    for symbol in ["NVDA", "QQQ", "SPY"]:
        grouped = df.groupby("date", sort=False)

        # Multi-bar cumulative log returns.
        for window in rolling_windows:
            df[f"{symbol}_log_return_sum_{window}"] = (
                grouped[f"{symbol}_log_return_1"]
                .rolling(window=window, min_periods=window)
                .sum()
                .reset_index(level=0, drop=True)
            )

            df[f"{symbol}_range_mean_{window}"] = (
                grouped[f"{symbol}_range_pct"]
                .rolling(window=window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"{symbol}_range_max_{window}"] = (
                grouped[f"{symbol}_range_pct"]
                .rolling(window=window, min_periods=window)
                .max()
                .reset_index(level=0, drop=True)
            )

            df[f"{symbol}_log_volume_mean_{window}"] = (
                grouped[f"{symbol}_log_volume"]
                .rolling(window=window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"{symbol}_log_volume_std_{window}"] = (
                grouped[f"{symbol}_log_volume"]
                .rolling(window=window, min_periods=window)
                .std()
                .reset_index(level=0, drop=True)
            )

            df[f"{symbol}_realized_vol_{window}"] = (
                grouped[f"{symbol}_log_return_1"]
                .rolling(window=window, min_periods=window)
                .std()
                .reset_index(level=0, drop=True)
            )

    # Cross-asset rolling relative returns.
    for window in rolling_windows:
        df[f"NVDA_minus_QQQ_log_return_sum_{window}"] = (
            df[f"NVDA_log_return_sum_{window}"]
            - df[f"QQQ_log_return_sum_{window}"]
        )

        df[f"NVDA_minus_SPY_log_return_sum_{window}"] = (
            df[f"NVDA_log_return_sum_{window}"]
            - df[f"SPY_log_return_sum_{window}"]
        )

        df[f"QQQ_minus_SPY_log_return_sum_{window}"] = (
            df[f"QQQ_log_return_sum_{window}"]
            - df[f"SPY_log_return_sum_{window}"]
        )
    # ---------------------------------------------------------
    # Validate newly engineered features
    # ---------------------------------------------------------

    new_feature_columns = [
        column
        for column in df.columns
        if column not in original_columns
    ]

    numeric_features = df[new_feature_columns].select_dtypes(
        include=[np.number]
    )

    infinite_counts = np.isinf(numeric_features).sum()
    infinite_counts = infinite_counts[infinite_counts > 0]

    if not infinite_counts.empty:
        raise ValueError(
            f"Engineered features contain infinite values:\n"
            f"{infinite_counts}"
        )

    print("\nMissing engineered feature values:")
    print(
        numeric_features.isna()
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )

    print(f"\nCreated {len(new_feature_columns)} features:")
    print(new_feature_columns)

    output_path = Path(processed_dir) / f"featured_{timeframe}.parquet"
    df.to_parquet(output_path, index=False)

    print(
        f"Saved featured dataset: "
        f"{len(df):,} rows -> {output_path}"
    )
MODEL_FEATURES = [
    # NVDA price-action and activity features
    "NVDA_body_pct",
    "NVDA_range_pct",
    "NVDA_upper_wick_pct",
    "NVDA_lower_wick_pct",
    "NVDA_close_position",
    "NVDA_vwap_distance",
    "NVDA_log_volume",
    "NVDA_log_trade_count",
    "NVDA_log_average_trade_size",
    "NVDA_log_dollar_volume",
    "NVDA_log_return_1",
    "NVDA_volume_change_1",
    "NVDA_trade_count_change_1",

    # QQQ price-action and activity features
    "QQQ_body_pct",
    "QQQ_range_pct",
    "QQQ_upper_wick_pct",
    "QQQ_lower_wick_pct",
    "QQQ_close_position",
    "QQQ_vwap_distance",
    "QQQ_log_volume",
    "QQQ_log_trade_count",
    "QQQ_log_average_trade_size",
    "QQQ_log_dollar_volume",
    "QQQ_log_return_1",
    "QQQ_volume_change_1",
    "QQQ_trade_count_change_1",

    # SPY price-action and activity features
    "SPY_body_pct",
    "SPY_range_pct",
    "SPY_upper_wick_pct",
    "SPY_lower_wick_pct",
    "SPY_close_position",
    "SPY_vwap_distance",
    "SPY_log_volume",
    "SPY_log_trade_count",
    "SPY_log_average_trade_size",
    "SPY_log_dollar_volume",
    "SPY_log_return_1",
    "SPY_volume_change_1",
    "SPY_trade_count_change_1",

    # Cross-asset features
    "NVDA_minus_QQQ_log_return_1",
    "NVDA_minus_SPY_log_return_1",
    "QQQ_minus_SPY_log_return_1",

    # Time features
    "time_of_day_sin",
    "time_of_day_cos",
    "session_position",
    "day_of_week_sin",
    "day_of_week_cos",

    # Lagged and rolling NVDA features
    "NVDA_log_return_sum_3",
    "NVDA_log_return_sum_6",
    "NVDA_log_return_sum_12",
    "NVDA_range_mean_3",
    "NVDA_range_mean_6",
    "NVDA_range_mean_12",
    "NVDA_range_max_3",
    "NVDA_range_max_6",
    "NVDA_range_max_12",
    "NVDA_log_volume_mean_3",
    "NVDA_log_volume_mean_6",
    "NVDA_log_volume_mean_12",
    "NVDA_log_volume_std_3",
    "NVDA_log_volume_std_6",
    "NVDA_log_volume_std_12",
    "NVDA_realized_vol_3",
    "NVDA_realized_vol_6",
    "NVDA_realized_vol_12",

    # Lagged and rolling QQQ features
    "QQQ_log_return_sum_3",
    "QQQ_log_return_sum_6",
    "QQQ_log_return_sum_12",
    "QQQ_range_mean_3",
    "QQQ_range_mean_6",
    "QQQ_range_mean_12",
    "QQQ_range_max_3",
    "QQQ_range_max_6",
    "QQQ_range_max_12",
    "QQQ_log_volume_mean_3",
    "QQQ_log_volume_mean_6",
    "QQQ_log_volume_mean_12",
    "QQQ_log_volume_std_3",
    "QQQ_log_volume_std_6",
    "QQQ_log_volume_std_12",
    "QQQ_realized_vol_3",
    "QQQ_realized_vol_6",
    "QQQ_realized_vol_12",

    # Lagged and rolling SPY features
    "SPY_log_return_sum_3",
    "SPY_log_return_sum_6",
    "SPY_log_return_sum_12",
    "SPY_range_mean_3",
    "SPY_range_mean_6",
    "SPY_range_mean_12",
    "SPY_range_max_3",
    "SPY_range_max_6",
    "SPY_range_max_12",
    "SPY_log_volume_mean_3",
    "SPY_log_volume_mean_6",
    "SPY_log_volume_mean_12",
    "SPY_log_volume_std_3",
    "SPY_log_volume_std_6",
    "SPY_log_volume_std_12",
    "SPY_realized_vol_3",
    "SPY_realized_vol_6",
    "SPY_realized_vol_12",

    # Cross-asset rolling relative-return features
    "NVDA_minus_QQQ_log_return_sum_3",
    "NVDA_minus_QQQ_log_return_sum_6",
    "NVDA_minus_QQQ_log_return_sum_12",
    "NVDA_minus_SPY_log_return_sum_3",
    "NVDA_minus_SPY_log_return_sum_6",
    "NVDA_minus_SPY_log_return_sum_12",
    "QQQ_minus_SPY_log_return_sum_3",
    "QQQ_minus_SPY_log_return_sum_6",
    "QQQ_minus_SPY_log_return_sum_12",
]


def build_model_feature_dataset(
    timeframe="5min",
    processed_dir=PROCESSED_DATA_DIR,
):
    """
    Create a model-safe feature dataset.

    Keeps all rows so pre-open candles can be used as transformer
    sequence context. Only rows with label_status == "valid" should
    later be used as supervised prediction targets.
    """
    input_path = Path(processed_dir) / f"featured_{timeframe}.parquet"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing featured dataset: {input_path}"
        )

    df = pd.read_parquet(input_path)

    metadata_columns = [
        "timestamp",
        "timestamp_pt",
        "date",
        "time",
        "label_status",
        "target_name",
        "target_class",
    ]

    required_columns = metadata_columns + MODEL_FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required model dataset columns:\n"
            f"{missing_columns}"
        )

    model_df = df[required_columns].copy()

    # Make sure feature columns contain no infinite values.
    infinite_counts = np.isinf(model_df[MODEL_FEATURES]).sum()
    infinite_counts = infinite_counts[infinite_counts > 0]

    if not infinite_counts.empty:
        raise ValueError(
            f"Model features contain infinite values:\n"
            f"{infinite_counts}"
        )

    # Valid supervised targets should have complete features.
    valid_target_rows = model_df.loc[
        model_df["label_status"] == "valid"
    ]

    missing_valid_features = (
        valid_target_rows[MODEL_FEATURES]
        .isna()
        .sum()
    )

    missing_valid_features = missing_valid_features[
        missing_valid_features > 0
    ]

    if not missing_valid_features.empty:
        print(
            "\nWarning: valid target rows contain missing features. "
            "Dropping those rows from supervised target eligibility."
        )
        print(missing_valid_features)

        complete_feature_mask = (
            model_df[MODEL_FEATURES]
            .notna()
            .all(axis=1)
        )

        model_df.loc[
            (model_df["label_status"] == "valid")
            & (~complete_feature_mask),
            "label_status",
        ] = "valid_but_missing_features"

    output_path = (
        Path(processed_dir)
        / f"model_features_{timeframe}.parquet"
    )

    model_df.to_parquet(output_path, index=False)

    print(f"Saved model feature dataset: {len(model_df):,} rows")
    print(f"Number of approved features: {len(MODEL_FEATURES)}")
    print(f"Valid supervised targets: {len(valid_target_rows):,}")
    print(f"Output path: {output_path}")
def build_walk_forward_splits(
    timeframe="5min",
    processed_dir=PROCESSED_DATA_DIR,
    test_fraction=0.20,
    initial_train_days=504,
    validation_days=63,
    step_days=63,
):
    """
    Create expanding walk-forward train/validation folds and one
    locked final test set.

    All rows from the same trading date remain in the same split.
    """
    input_path = (
        Path(processed_dir)
        / f"model_features_{timeframe}.parquet"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing model feature dataset: {input_path}"
        )

    df = pd.read_parquet(input_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
    df["date"] = df["timestamp_pt"].dt.date

    df = df.sort_values("timestamp").reset_index(drop=True)

    unique_dates = np.array(sorted(df["date"].unique()))
    total_days = len(unique_dates)

    # Reserve the most recent portion as the locked test set.
    test_start_index = int(total_days * (1 - test_fraction))

    development_dates = unique_dates[:test_start_index]
    test_dates = unique_dates[test_start_index:]

    if len(development_dates) < initial_train_days + validation_days:
        raise ValueError(
            "Not enough development dates for the requested "
            "initial train and validation windows."
        )

    split_directory = (
        Path(processed_dir)
        / f"walk_forward_{timeframe}"
    )

    split_directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Save locked final test set
    # ---------------------------------------------------------

    test_df = df.loc[df["date"].isin(test_dates)].copy()

    test_path = split_directory / "locked_test.parquet"
    test_df.to_parquet(test_path, index=False)

    print(
        f"Locked test: {len(test_dates):,} days, "
        f"{len(test_df):,} total rows"
    )
    print(
        f"Test dates: {test_dates[0]} through {test_dates[-1]}"
    )

    # ---------------------------------------------------------
    # Create expanding walk-forward folds
    # ---------------------------------------------------------

    fold_records = []

    fold_number = 1
    train_end_index = initial_train_days

    while train_end_index + validation_days <= len(development_dates):

        train_dates = development_dates[:train_end_index]

        validation_dates = development_dates[
            train_end_index:
            train_end_index + validation_days
        ]

        train_df = df.loc[
            df["date"].isin(train_dates)
        ].copy()

        validation_df = df.loc[
            df["date"].isin(validation_dates)
        ].copy()

        fold_directory = (
            split_directory
            / f"fold_{fold_number:02d}"
        )

        fold_directory.mkdir(parents=True, exist_ok=True)

        train_path = fold_directory / "train.parquet"
        validation_path = fold_directory / "validation.parquet"

        train_df.to_parquet(train_path, index=False)
        validation_df.to_parquet(validation_path, index=False)

        train_valid_targets = (
            train_df["label_status"] == "valid"
        ).sum()

        validation_valid_targets = (
            validation_df["label_status"] == "valid"
        ).sum()

        fold_records.append(
            {
                "fold": fold_number,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "validation_start": validation_dates[0],
                "validation_end": validation_dates[-1],
                "train_days": len(train_dates),
                "validation_days": len(validation_dates),
                "train_rows": len(train_df),
                "validation_rows": len(validation_df),
                "train_valid_targets": train_valid_targets,
                "validation_valid_targets": validation_valid_targets,
            }
        )

        print(f"\nFold {fold_number}")
        print(
            f"Train: {train_dates[0]} through {train_dates[-1]} "
            f"({len(train_dates):,} days)"
        )
        print(
            f"Validation: {validation_dates[0]} through "
            f"{validation_dates[-1]} "
            f"({len(validation_dates):,} days)"
        )
        print(
            f"Valid targets: "
            f"{train_valid_targets:,} train, "
            f"{validation_valid_targets:,} validation"
        )

        fold_number += 1
        train_end_index += step_days

    manifest_df = pd.DataFrame(fold_records)

    manifest_path = split_directory / "walk_forward_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"\nCreated {len(manifest_df)} walk-forward folds.")
    print(f"Split directory: {split_directory}")
    print(f"Manifest saved to: {manifest_path}")

    return manifest_df