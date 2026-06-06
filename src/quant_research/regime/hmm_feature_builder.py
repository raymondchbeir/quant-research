from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from quant_research.utils.paths import PROCESSED_DATA_DIR


HMM_INPUT_FEATURES = [
    "NVDA_log_return_1",
    "QQQ_log_return_1",
    "SPY_log_return_1",

    "NVDA_log_return_sum_3",
    "NVDA_log_return_sum_6",
    "NVDA_log_return_sum_12",

    "QQQ_log_return_sum_3",
    "QQQ_log_return_sum_6",
    "QQQ_log_return_sum_12",

    "SPY_log_return_sum_3",
    "SPY_log_return_sum_6",
    "SPY_log_return_sum_12",

    "NVDA_realized_vol_3",
    "NVDA_realized_vol_6",
    "NVDA_realized_vol_12",

    "QQQ_realized_vol_3",
    "QQQ_realized_vol_6",
    "QQQ_realized_vol_12",

    "SPY_realized_vol_3",
    "SPY_realized_vol_6",
    "SPY_realized_vol_12",

    "NVDA_minus_QQQ_log_return_1",
    "NVDA_minus_SPY_log_return_1",
    "QQQ_minus_SPY_log_return_1",

    "NVDA_minus_QQQ_log_return_sum_3",
    "NVDA_minus_QQQ_log_return_sum_6",
    "NVDA_minus_QQQ_log_return_sum_12",

    "NVDA_minus_SPY_log_return_sum_3",
    "NVDA_minus_SPY_log_return_sum_6",
    "NVDA_minus_SPY_log_return_sum_12",

    "session_position",
]


HMM_OUTPUT_FEATURES_3_STATE = [
    "hmm_state",
    "hmm_state_prob_0",
    "hmm_state_prob_1",
    "hmm_state_prob_2",
    "hmm_state_return_rank",
    "hmm_state_vol_rank",
]


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_parquet(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
    df["date"] = df["timestamp_pt"].dt.date

    return df.sort_values("timestamp").reset_index(drop=True)


def check_hmm_columns(
    df: pd.DataFrame,
    hmm_input_features: list[str],
) -> None:
    missing = [
        column
        for column in hmm_input_features
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing HMM input features:\n{missing}"
        )


def make_hmm_training_matrix(
    df: pd.DataFrame,
    hmm_input_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    HMM can use all finite rows, not only valid target rows.

    We keep timestamp columns so we can merge predictions back.
    """
    check_hmm_columns(df, hmm_input_features)

    X_df = df[hmm_input_features].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    finite_mask = X_df.notna().all(axis=1)

    clean_df = df.loc[finite_mask].copy().reset_index(drop=True)

    X = clean_df[hmm_input_features].to_numpy(dtype=np.float64)

    return clean_df, X


def summarize_hmm_states(
    df_with_states: pd.DataFrame,
    number_of_states: int,
) -> pd.DataFrame:
    """
    Create state-level return/volatility rankings.

    State IDs themselves are arbitrary, so ranks make the feature
    more interpretable.
    """
    rows = []

    for state in range(number_of_states):
        state_df = df_with_states.loc[
            df_with_states["hmm_state"] == state
        ]

        if state_df.empty:
            rows.append(
                {
                    "hmm_state": state,
                    "state_row_count": 0,
                    "state_mean_NVDA_log_return_1": 0.0,
                    "state_std_NVDA_log_return_1": 0.0,
                }
            )
            continue

        returns = state_df["NVDA_log_return_1"]

        rows.append(
            {
                "hmm_state": state,
                "state_row_count": int(len(state_df)),
                "state_mean_NVDA_log_return_1": float(returns.mean()),
                "state_std_NVDA_log_return_1": float(returns.std()),
            }
        )

    state_summary = pd.DataFrame(rows)

    state_summary["hmm_state_return_rank"] = (
        state_summary["state_mean_NVDA_log_return_1"]
        .rank(method="dense")
        .astype(int)
        - 1
    )

    state_summary["hmm_state_vol_rank"] = (
        state_summary["state_std_NVDA_log_return_1"]
        .rank(method="dense")
        .astype(int)
        - 1
    )

    return state_summary


def fit_hmm_on_train(
    train_df: pd.DataFrame,
    hmm_input_features: list[str] = HMM_INPUT_FEATURES,
    number_of_states: int = 3,
    random_seed: int = 42,
    covariance_type: str = "diag",
    n_iter: int = 300,
) -> tuple[GaussianHMM, StandardScaler, pd.DataFrame]:
    """
    Fit scaler + HMM using training data only.
    """
    clean_train_df, X_train = make_hmm_training_matrix(
        df=train_df,
        hmm_input_features=hmm_input_features,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = GaussianHMM(
        n_components=number_of_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_seed,
        verbose=False,
    )

    model.fit(X_train_scaled)

    train_states = model.predict(X_train_scaled)

    train_with_states = clean_train_df.copy()
    train_with_states["hmm_state"] = train_states

    state_summary = summarize_hmm_states(
        df_with_states=train_with_states,
        number_of_states=number_of_states,
    )

    return model, scaler, state_summary


def add_hmm_features_to_dataframe(
    df: pd.DataFrame,
    model: GaussianHMM,
    scaler: StandardScaler,
    state_summary: pd.DataFrame,
    hmm_input_features: list[str] = HMM_INPUT_FEATURES,
    number_of_states: int = 3,
) -> pd.DataFrame:
    """
    Add HMM state and posterior probabilities to every finite row.

    Rows with missing HMM inputs get neutral fallback values.
    """
    df = df.copy()

    check_hmm_columns(df, hmm_input_features)

    X_df = df[hmm_input_features].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    finite_mask = X_df.notna().all(axis=1)

    # Defaults for rows where HMM inputs are not available.
    df["hmm_state"] = -1

    for state in range(number_of_states):
        df[f"hmm_state_prob_{state}"] = 0.0

    df["hmm_state_return_rank"] = -1
    df["hmm_state_vol_rank"] = -1

    if finite_mask.any():
        X = X_df.loc[finite_mask].to_numpy(dtype=np.float64)
        X_scaled = scaler.transform(X)

        states = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)

        finite_indices = df.index[finite_mask]

        df.loc[finite_indices, "hmm_state"] = states

        for state in range(number_of_states):
            df.loc[
                finite_indices,
                f"hmm_state_prob_{state}",
            ] = probs[:, state]

        state_rank_lookup = state_summary.set_index("hmm_state")[
            [
                "hmm_state_return_rank",
                "hmm_state_vol_rank",
            ]
        ]

        rank_df = (
            pd.DataFrame({"hmm_state": states}, index=finite_indices)
            .join(state_rank_lookup, on="hmm_state")
        )

        df.loc[
            finite_indices,
            "hmm_state_return_rank",
        ] = rank_df["hmm_state_return_rank"].to_numpy()

        df.loc[
            finite_indices,
            "hmm_state_vol_rank",
        ] = rank_df["hmm_state_vol_rank"].to_numpy()

    # Make state/rank numeric and safe for model input.
    df["hmm_state"] = df["hmm_state"].astype(float)
    df["hmm_state_return_rank"] = df[
        "hmm_state_return_rank"
    ].astype(float)
    df["hmm_state_vol_rank"] = df[
        "hmm_state_vol_rank"
    ].astype(float)

    return df


def build_hmm_augmented_walk_forward_splits(
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_name: str = "walk_forward_hmm_5min",
    hmm_input_features: list[str] = HMM_INPUT_FEATURES,
    number_of_states: int = 3,
    random_seed: int = 42,
    covariance_type: str = "diag",
) -> Path:
    """
    Create HMM-augmented train/validation splits.

    For each fold:
        HMM is fit on fold train only.
        HMM features are added to train and validation.
    """
    processed_dir = Path(processed_dir)

    input_dir = processed_dir / f"walk_forward_{timeframe}"
    output_dir = processed_dir / output_name

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing walk-forward dir: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(
        path for path in input_dir.glob("fold_*") if path.is_dir()
    )

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        print(f"Building HMM features for {fold_name}")

        train_df = load_split(fold_dir / "train.parquet")
        validation_df = load_split(fold_dir / "validation.parquet")

        model, scaler, state_summary = fit_hmm_on_train(
            train_df=train_df,
            hmm_input_features=hmm_input_features,
            number_of_states=number_of_states,
            random_seed=random_seed,
            covariance_type=covariance_type,
        )

        train_augmented = add_hmm_features_to_dataframe(
            df=train_df,
            model=model,
            scaler=scaler,
            state_summary=state_summary,
            hmm_input_features=hmm_input_features,
            number_of_states=number_of_states,
        )

        validation_augmented = add_hmm_features_to_dataframe(
            df=validation_df,
            model=model,
            scaler=scaler,
            state_summary=state_summary,
            hmm_input_features=hmm_input_features,
            number_of_states=number_of_states,
        )

        output_fold_dir = output_dir / fold_name
        output_fold_dir.mkdir(parents=True, exist_ok=True)

        train_augmented.to_parquet(
            output_fold_dir / "train.parquet",
            index=False,
        )

        validation_augmented.to_parquet(
            output_fold_dir / "validation.parquet",
            index=False,
        )

        state_summary.to_csv(
            output_fold_dir / "hmm_state_summary.csv",
            index=False,
        )

        with (output_fold_dir / "hmm_model.pkl").open("wb") as file:
            pickle.dump(model, file)

        with (output_fold_dir / "hmm_scaler.pkl").open("wb") as file:
            pickle.dump(scaler, file)

    config = {
        "timeframe": timeframe,
        "output_name": output_name,
        "number_of_states": number_of_states,
        "hmm_input_features": hmm_input_features,
        "hmm_output_features": HMM_OUTPUT_FEATURES_3_STATE,
        "random_seed": random_seed,
        "covariance_type": covariance_type,
    }

    with (output_dir / "hmm_config.json").open("w") as file:
        json.dump(config, file, indent=2)

    print(f"\nSaved HMM-augmented splits to: {output_dir}")

    return output_dir


def build_hmm_augmented_locked_test_split(
    split_name: str = "locked_test_transformer_split_5min",
    output_name: str = "locked_test_transformer_split_hmm_5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    hmm_input_features: list[str] = HMM_INPUT_FEATURES,
    number_of_states: int = 3,
    random_seed: int = 42,
    covariance_type: str = "diag",
) -> Path:
    """
    Add HMM features to final train / early_stop / locked_test split.

    HMM is fit only on final train split.
    """
    processed_dir = Path(processed_dir)

    input_dir = processed_dir / split_name
    output_dir = processed_dir / output_name

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing locked-test split: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(input_dir / "train.parquet")
    early_stop_df = load_split(input_dir / "early_stop.parquet")
    locked_test_df = load_split(input_dir / "locked_test.parquet")

    model, scaler, state_summary = fit_hmm_on_train(
        train_df=train_df,
        hmm_input_features=hmm_input_features,
        number_of_states=number_of_states,
        random_seed=random_seed,
        covariance_type=covariance_type,
    )

    train_augmented = add_hmm_features_to_dataframe(
        df=train_df,
        model=model,
        scaler=scaler,
        state_summary=state_summary,
        hmm_input_features=hmm_input_features,
        number_of_states=number_of_states,
    )

    early_stop_augmented = add_hmm_features_to_dataframe(
        df=early_stop_df,
        model=model,
        scaler=scaler,
        state_summary=state_summary,
        hmm_input_features=hmm_input_features,
        number_of_states=number_of_states,
    )

    locked_test_augmented = add_hmm_features_to_dataframe(
        df=locked_test_df,
        model=model,
        scaler=scaler,
        state_summary=state_summary,
        hmm_input_features=hmm_input_features,
        number_of_states=number_of_states,
    )

    train_augmented.to_parquet(output_dir / "train.parquet", index=False)
    early_stop_augmented.to_parquet(output_dir / "early_stop.parquet", index=False)
    locked_test_augmented.to_parquet(output_dir / "locked_test.parquet", index=False)

    state_summary.to_csv(output_dir / "hmm_state_summary.csv", index=False)

    with (output_dir / "hmm_model.pkl").open("wb") as file:
        pickle.dump(model, file)

    with (output_dir / "hmm_scaler.pkl").open("wb") as file:
        pickle.dump(scaler, file)

    config = {
        "input_split": split_name,
        "output_name": output_name,
        "number_of_states": number_of_states,
        "hmm_input_features": hmm_input_features,
        "hmm_output_features": HMM_OUTPUT_FEATURES_3_STATE,
        "random_seed": random_seed,
        "covariance_type": covariance_type,
    }

    with (output_dir / "hmm_config.json").open("w") as file:
        json.dump(config, file, indent=2)

    print(f"Saved HMM-augmented locked-test split to: {output_dir}")

    return output_dir