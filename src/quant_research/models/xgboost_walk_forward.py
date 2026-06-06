from __future__ import annotations

import json
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from quant_research.features.feature_builder import MODEL_FEATURES
from quant_research.utils.paths import PROCESSED_DATA_DIR


CLASS_IDS = [0, 1, 2]

CLASS_NAMES = {
    0: "neutral",
    1: "up",
    2: "down",
}


def load_valid_targets(path: Path) -> pd.DataFrame:
    """
    Load one walk-forward parquet file and retain only valid
    supervised target rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_parquet(path)

    required_columns = [
        "timestamp",
        "timestamp_pt",
        "date",
        "label_status",
        "target_class",
        "target_name",
        *MODEL_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns:\n"
            f"{missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_pt"] = pd.to_datetime(df["timestamp_pt"])
    df["date"] = df["timestamp_pt"].dt.date

    df = (
        df.loc[df["label_status"].eq("valid")]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(f"No valid target rows found in {path}")

    missing_features = df[MODEL_FEATURES].isna().sum()
    missing_features = missing_features[missing_features > 0]

    if not missing_features.empty:
        raise ValueError(
            f"{path.name} contains missing model features:\n"
            f"{missing_features}"
        )

    numeric_features = df[MODEL_FEATURES].to_numpy(dtype=np.float32)

    if not np.isfinite(numeric_features).all():
        raise ValueError(
            f"{path.name} contains non-finite feature values."
        )

    df["target_class"] = df["target_class"].astype(int)

    unknown_classes = sorted(
        set(df["target_class"].unique()) - set(CLASS_IDS)
    )

    if unknown_classes:
        raise ValueError(
            f"{path.name} contains unknown target classes: "
            f"{unknown_classes}"
        )

    return df


def split_training_for_early_stopping(
    train_df: pd.DataFrame,
    early_stopping_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use the final portion of the training dates as an internal
    early-stopping set.

    The actual walk-forward validation set remains fully unseen.
    """
    unique_dates = np.array(sorted(train_df["date"].unique()))

    number_of_early_stopping_days = max(
        1,
        int(len(unique_dates) * early_stopping_fraction),
    )

    if number_of_early_stopping_days >= len(unique_dates):
        raise ValueError(
            "Not enough training dates to create an internal "
            "early-stopping period."
        )

    fit_dates = unique_dates[:-number_of_early_stopping_days]
    early_stopping_dates = unique_dates[-number_of_early_stopping_days:]

    fit_df = train_df.loc[
        train_df["date"].isin(fit_dates)
    ].copy()

    early_stopping_df = train_df.loc[
        train_df["date"].isin(early_stopping_dates)
    ].copy()

    return fit_df, early_stopping_df


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int = 3,
) -> float:
    """
    Calculate the multiclass Brier score.

    Lower is better.
    """
    one_hot_targets = np.eye(number_of_classes)[y_true]

    return float(
        np.mean(
            np.sum(
                (probabilities - one_hot_targets) ** 2,
                axis=1,
            )
        )
    )


def calculate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict, np.ndarray, dict]:
    """
    Calculate overall and per-class classification metrics.
    """
    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=CLASS_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=CLASS_IDS,
                average="weighted",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=CLASS_IDS,
            )
        ),
        "multiclass_brier_score": multiclass_brier_score(
            y_true=y_true,
            probabilities=probabilities,
            number_of_classes=len(CLASS_IDS),
        ),
    }

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=CLASS_IDS,
    )

    report = classification_report(
        y_true,
        predictions,
        labels=CLASS_IDS,
        target_names=[
            CLASS_NAMES[class_id]
            for class_id in CLASS_IDS
        ],
        output_dict=True,
        zero_division=0,
    )

    return metrics, matrix, report


def train_one_fold(
    fold_directory: Path,
    output_directory: Path,
    early_stopping_fraction: float = 0.10,
    use_balanced_sample_weights: bool = False,
    random_seed: int = 42,
) -> dict:
    """
    Train and evaluate one expanding walk-forward XGBoost fold.
    """
    fold_name = fold_directory.name

    train_path = fold_directory / "train.parquet"
    validation_path = fold_directory / "validation.parquet"

    full_train_df = load_valid_targets(train_path)
    validation_df = load_valid_targets(validation_path)

    fit_df, early_stopping_df = split_training_for_early_stopping(
        train_df=full_train_df,
        early_stopping_fraction=early_stopping_fraction,
    )

    X_fit = fit_df[MODEL_FEATURES].astype(np.float32)
    y_fit = fit_df["target_class"].to_numpy(dtype=np.int64)

    X_early_stopping = early_stopping_df[
        MODEL_FEATURES
    ].astype(np.float32)

    y_early_stopping = early_stopping_df[
        "target_class"
    ].to_numpy(dtype=np.int64)

    X_validation = validation_df[
        MODEL_FEATURES
    ].astype(np.float32)

    y_validation = validation_df[
        "target_class"
    ].to_numpy(dtype=np.int64)

    sample_weight = None

    if use_balanced_sample_weights:
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_fit,
        )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_IDS),
        n_estimators=2_000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=1.00,
        gamma=0.00,
        eval_metric="mlogloss",
        early_stopping_rounds=75,
        tree_method="hist",
        random_state=random_seed,
        n_jobs=-1,
    )

    model.fit(
        X_fit,
        y_fit,
        sample_weight=sample_weight,
        eval_set=[
            (X_early_stopping, y_early_stopping),
        ],
        verbose=False,
    )

    validation_probabilities = model.predict_proba(X_validation)

    validation_predictions = np.argmax(
        validation_probabilities,
        axis=1,
    )

    metrics, matrix, report = calculate_metrics(
        y_true=y_validation,
        predictions=validation_predictions,
        probabilities=validation_probabilities,
    )

    metrics.update(
        {
            "fold": fold_name,
            "fit_start": str(fit_df["date"].min()),
            "fit_end": str(fit_df["date"].max()),
            "early_stopping_start": str(
                early_stopping_df["date"].min()
            ),
            "early_stopping_end": str(
                early_stopping_df["date"].max()
            ),
            "validation_start": str(
                validation_df["date"].min()
            ),
            "validation_end": str(
                validation_df["date"].max()
            ),
            "fit_rows": int(len(fit_df)),
            "early_stopping_rows": int(
                len(early_stopping_df)
            ),
            "validation_rows": int(len(validation_df)),
            "best_iteration": int(model.best_iteration),
            "best_score": float(model.best_score),
            "balanced_sample_weights": bool(
                use_balanced_sample_weights
            ),
        }
    )

    fold_output_directory = output_directory / fold_name
    fold_output_directory.mkdir(parents=True, exist_ok=True)

    # Save trained model.
    model_path = fold_output_directory / "xgboost_model.json"
    model.save_model(model_path)

    # Save validation predictions and probabilities.
    prediction_df = validation_df[
        [
            "timestamp",
            "timestamp_pt",
            "date",
            "target_name",
            "target_class",
        ]
    ].copy()

    prediction_df["predicted_class"] = validation_predictions
    prediction_df["predicted_name"] = prediction_df[
        "predicted_class"
    ].map(CLASS_NAMES)

    prediction_df["p_neutral"] = validation_probabilities[:, 0]
    prediction_df["p_up"] = validation_probabilities[:, 1]
    prediction_df["p_down"] = validation_probabilities[:, 2]

    predictions_path = (
        fold_output_directory
        / "validation_predictions.parquet"
    )

    prediction_df.to_parquet(predictions_path, index=False)

    # Save confusion matrix.
    confusion_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{CLASS_NAMES[class_id]}"
            for class_id in CLASS_IDS
        ],
        columns=[
            f"predicted_{CLASS_NAMES[class_id]}"
            for class_id in CLASS_IDS
        ],
    )

    confusion_path = (
        fold_output_directory
        / "confusion_matrix.csv"
    )

    confusion_df.to_csv(confusion_path)

    # Save classification report.
    report_path = (
        fold_output_directory
        / "classification_report.json"
    )

    with report_path.open("w") as file:
        json.dump(report, file, indent=2)

    # Save fold metrics.
    metrics_path = fold_output_directory / "metrics.json"

    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=2)

    print(f"\n{fold_name}")
    print(
        f"Fit dates: {metrics['fit_start']} "
        f"through {metrics['fit_end']}"
    )
    print(
        f"Validation dates: {metrics['validation_start']} "
        f"through {metrics['validation_end']}"
    )
    print(f"Best iteration: {metrics['best_iteration']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")

    return metrics


def train_xgboost_walk_forward(
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    early_stopping_fraction: float = 0.10,
    use_balanced_sample_weights: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Train XGBoost across every walk-forward fold.

    Does not access the locked final test set.
    """
    walk_forward_directory = (
        Path(processed_dir)
        / f"walk_forward_{timeframe}"
    )

    if not walk_forward_directory.exists():
        raise FileNotFoundError(
            f"Missing walk-forward directory: "
            f"{walk_forward_directory}"
        )

    fold_directories = sorted(
        path
        for path in walk_forward_directory.glob("fold_*")
        if path.is_dir()
    )

    if not fold_directories:
        raise ValueError(
            f"No walk-forward fold directories found in "
            f"{walk_forward_directory}"
        )

    output_directory = (
        Path(processed_dir)
        / f"xgboost_walk_forward_{timeframe}_balanced"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    fold_progress = tqdm(
        fold_directories,
        desc="Training XGBoost walk-forward folds",
        unit="fold",
    )

        # ---------------------------------------------------------
    # Train every walk-forward fold
    # ---------------------------------------------------------

    all_metrics = []

    fold_progress = tqdm(
        fold_directories,
        desc="Training XGBoost walk-forward folds",
        unit="fold",
    )

    for fold_directory in fold_progress:
        fold_progress.set_description(
            f"Training {fold_directory.name}"
        )

        fold_metrics = train_one_fold(
            fold_directory=fold_directory,
            output_directory=output_directory,
            early_stopping_fraction=early_stopping_fraction,
            use_balanced_sample_weights=use_balanced_sample_weights,
            random_seed=random_seed,
        )

        all_metrics.append(fold_metrics)

        fold_progress.set_postfix(
            {
                "macro_f1": f"{fold_metrics['macro_f1']:.4f}",
                "accuracy": f"{fold_metrics['accuracy']:.4f}",
                "best_iter": fold_metrics["best_iteration"],
            }
        )

    # ---------------------------------------------------------
    # Save fold-level summary metrics
    # ---------------------------------------------------------

    summary_df = pd.DataFrame(all_metrics)

    if summary_df.empty:
        raise ValueError("No fold metrics were generated.")

    summary_path = output_directory / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ---------------------------------------------------------
    # Save aggregate metrics across folds
    # ---------------------------------------------------------

    aggregate_metrics = {
        "number_of_folds": int(len(summary_df)),
        "mean_accuracy": float(summary_df["accuracy"].mean()),
        "mean_balanced_accuracy": float(
            summary_df["balanced_accuracy"].mean()
        ),
        "mean_macro_f1": float(summary_df["macro_f1"].mean()),
        "mean_weighted_f1": float(summary_df["weighted_f1"].mean()),
        "mean_log_loss": float(summary_df["log_loss"].mean()),
        "mean_multiclass_brier_score": float(
            summary_df["multiclass_brier_score"].mean()
        ),
        "std_accuracy": float(summary_df["accuracy"].std()),
        "std_balanced_accuracy": float(
            summary_df["balanced_accuracy"].std()
        ),
        "std_macro_f1": float(summary_df["macro_f1"].std()),
        "std_log_loss": float(summary_df["log_loss"].std()),
    }

    aggregate_path = output_directory / "aggregate_metrics.json"

    with aggregate_path.open("w") as file:
        json.dump(aggregate_metrics, file, indent=2)

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    display_columns = [
        "fold",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "multiclass_brier_score",
        "best_iteration",
    ]

    print("\nWalk-forward fold results")
    print(summary_df[display_columns].to_string(index=False))

    print("\nMean metrics across folds")
    for name, value in aggregate_metrics.items():
        if name == "number_of_folds":
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.6f}")

    print(f"\nSaved fold summary to: {summary_path}")
    print(f"Saved aggregate metrics to: {aggregate_path}")
    print(f"Saved fold outputs to: {output_directory}")

    return summary_df