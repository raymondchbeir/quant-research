from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from quant_research.features.feature_builder import MODEL_FEATURES
from quant_research.models.neural_models import TransformerSequenceClassifier
from quant_research.models.sequence_dataset import (
    SequenceClassificationDataset,
    build_sequences_from_dataframe,
    fit_sequence_scaler,
    load_split_dataframe,
    transform_sequences,
)
from quant_research.models.train_sequence_walk_forward import (
    CLASS_IDS,
    CLASS_NAMES,
    calculate_metrics,
    get_device,
    predict_probabilities,
    train_one_epoch,
)
from quant_research.utils.paths import PROCESSED_DATA_DIR


def create_locked_test_training_split(
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_name: str = "locked_test_transformer_split_5min",
    test_fraction: float = 0.20,
    early_stopping_fraction: float = 0.10,
) -> Path:
    """
    Create final train / early-stop / locked-test split.

    The locked test is the most recent test_fraction of dates.
    From the development period, the most recent early_stopping_fraction
    is used for early stopping.

    This function should be used for the base feature split.
    For HMM features, first create this base split, then run the HMM
    locked-test feature builder to create an HMM-augmented split.
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

    early_stop_start_index = int(
        len(development_dates) * (1 - early_stopping_fraction)
    )

    train_dates = development_dates[:early_stop_start_index]
    early_stop_dates = development_dates[early_stop_start_index:]

    split_dir = processed_dir / output_name
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df = df.loc[df["date"].isin(train_dates)].copy()
    early_stop_df = df.loc[df["date"].isin(early_stop_dates)].copy()
    locked_test_df = df.loc[df["date"].isin(locked_test_dates)].copy()

    train_df.to_parquet(split_dir / "train.parquet", index=False)
    early_stop_df.to_parquet(split_dir / "early_stop.parquet", index=False)
    locked_test_df.to_parquet(split_dir / "locked_test.parquet", index=False)

    manifest = {
        "timeframe": timeframe,
        "test_fraction": test_fraction,
        "early_stopping_fraction": early_stopping_fraction,
        "total_days": int(total_days),
        "train_days": int(len(train_dates)),
        "early_stop_days": int(len(early_stop_dates)),
        "locked_test_days": int(len(locked_test_dates)),
        "train_start": str(train_dates[0]),
        "train_end": str(train_dates[-1]),
        "early_stop_start": str(early_stop_dates[0]),
        "early_stop_end": str(early_stop_dates[-1]),
        "locked_test_start": str(locked_test_dates[0]),
        "locked_test_end": str(locked_test_dates[-1]),
        "train_rows": int(len(train_df)),
        "early_stop_rows": int(len(early_stop_df)),
        "locked_test_rows": int(len(locked_test_df)),
        "train_valid_targets": int((train_df["label_status"] == "valid").sum()),
        "early_stop_valid_targets": int(
            (early_stop_df["label_status"] == "valid").sum()
        ),
        "locked_test_valid_targets": int(
            (locked_test_df["label_status"] == "valid").sum()
        ),
    }

    with (split_dir / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print("Created locked-test transformer split")
    print(json.dumps(manifest, indent=2))

    return split_dir


def validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> None:
    """
    Fail early if the selected feature list is not available.
    """
    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing selected feature columns:\n"
            f"{missing_columns}"
        )


def build_datasets_for_locked_test(
    split_dir: Path,
    sequence_length: int = 21,
    feature_columns: list[str] | None = None,
) -> tuple[
    SequenceClassificationDataset,
    SequenceClassificationDataset,
    SequenceClassificationDataset,
    object,
]:
    """
    Build train, early-stop, and locked-test sequence datasets.

    Scaler is fitted only on training sequences.

    feature_columns controls whether we train the base transformer
    or the HMM-feature transformer.
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    split_dir = Path(split_dir)

    train_df = load_split_dataframe(split_dir / "train.parquet")
    early_stop_df = load_split_dataframe(split_dir / "early_stop.parquet")
    locked_test_df = load_split_dataframe(split_dir / "locked_test.parquet")

    validate_feature_columns(
        df=train_df,
        feature_columns=feature_columns,
        split_name="train split",
    )
    validate_feature_columns(
        df=early_stop_df,
        feature_columns=feature_columns,
        split_name="early-stop split",
    )
    validate_feature_columns(
        df=locked_test_df,
        feature_columns=feature_columns,
        split_name="locked-test split",
    )

    train_sequences, train_labels, train_metadata = build_sequences_from_dataframe(
        df=train_df,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
    )

    early_sequences, early_labels, early_metadata = build_sequences_from_dataframe(
        df=early_stop_df,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
    )

    test_sequences, test_labels, test_metadata = build_sequences_from_dataframe(
        df=locked_test_df,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
    )

    scaler = fit_sequence_scaler(train_sequences)

    train_sequences = transform_sequences(train_sequences, scaler)
    early_sequences = transform_sequences(early_sequences, scaler)
    test_sequences = transform_sequences(test_sequences, scaler)

    train_dataset = SequenceClassificationDataset(
        train_sequences,
        train_labels,
        train_metadata,
    )

    early_stop_dataset = SequenceClassificationDataset(
        early_sequences,
        early_labels,
        early_metadata,
    )

    locked_test_dataset = SequenceClassificationDataset(
        test_sequences,
        test_labels,
        test_metadata,
    )

    return train_dataset, early_stop_dataset, locked_test_dataset, scaler


def train_final_transformer_locked_test(
    experiment_name: str,
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    split_name: str = "locked_test_transformer_split_5min",
    recreate_split: bool = True,
    test_fraction: float = 0.20,
    early_stopping_fraction: float = 0.10,
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
    continue_training: bool = True,
    continue_learning_rate: float = 2e-5,
    continue_weight_decay: float = 1e-5,
    continue_epochs: int = 10,
    continue_patience: int = 3,
    device: str | None = None,
    random_seed: int = 42,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Train final transformer using train/early-stop development data,
    then evaluate once on the locked test set.

    If feature_columns is None, the model uses MODEL_FEATURES.

    For HMM transformer:
        pass MODEL_FEATURES + HMM_OUTPUT_FEATURES_3_STATE
        and use split_name="locked_test_transformer_split_hmm_5min"
        with recreate_split=False.
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    processed_dir = Path(processed_dir)
    split_dir = processed_dir / split_name

    if recreate_split or not split_dir.exists():
        split_dir = create_locked_test_training_split(
            timeframe=timeframe,
            processed_dir=processed_dir,
            output_name=split_name,
            test_fraction=test_fraction,
            early_stopping_fraction=early_stopping_fraction,
        )

    output_dir = (
        processed_dir
        / f"transformer_locked_test_{timeframe}"
        / experiment_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    device_obj = get_device(device)

    print(f"Training locked-test model with {len(feature_columns)} features")
    print(f"Split directory: {split_dir}")
    print(f"Output directory: {output_dir}")

    train_dataset, early_stop_dataset, locked_test_dataset, scaler = (
        build_datasets_for_locked_test(
            split_dir=split_dir,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    early_loader = DataLoader(
        early_stop_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        locked_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = TransformerSequenceClassifier(
        num_features=len(feature_columns),
        sequence_length=sequence_length,
        num_classes=len(CLASS_IDS),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device_obj)

    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_path = output_dir / "transformer_model.pt"
    scaler_path = output_dir / "scaler.pkl"

    config = {
        "sequence_length": int(sequence_length),
        "num_features": int(len(feature_columns)),
        "d_model": int(d_model),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "dim_feedforward": int(dim_feedforward),
        "dropout": float(dropout),
    }

    def run_training_phase(
        phase_name: str,
        phase_learning_rate: float,
        phase_weight_decay: float,
        phase_epochs: int,
        phase_patience: int,
        starting_best_loss: float = np.inf,
    ) -> tuple[float, int, pd.DataFrame]:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=phase_learning_rate,
            weight_decay=phase_weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        best_loss = starting_best_loss
        best_epoch = 0
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, phase_epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device_obj,
            )

            early_probs, early_labels = predict_probabilities(
                model=model,
                dataloader=early_loader,
                device=device_obj,
            )

            early_metrics, _, _ = calculate_metrics(
                y_true=early_labels,
                probabilities=early_probs,
            )

            early_loss = early_metrics["log_loss"]

            scheduler.step(early_loss)

            row = {
                "phase": phase_name,
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "early_stop_loss": float(early_loss),
                **early_metrics,
            }

            history.append(row)

            print(
                f"{phase_name} epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"early_loss={early_loss:.4f}, "
                f"macro_f1={early_metrics['macro_f1']:.4f}"
            )

            if early_loss < best_loss:
                best_loss = early_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                    },
                    checkpoint_path,
                )

                with scaler_path.open("wb") as file:
                    pickle.dump(scaler, file)

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= phase_patience:
                break

        return float(best_loss), int(best_epoch), pd.DataFrame(history)

    best_loss, best_epoch, history_stage_1 = run_training_phase(
        phase_name="stage_1",
        phase_learning_rate=learning_rate,
        phase_weight_decay=weight_decay,
        phase_epochs=max_epochs,
        phase_patience=patience,
    )

    histories = [history_stage_1]

    if continue_training:
        checkpoint = torch.load(checkpoint_path, map_location=device_obj)
        model.load_state_dict(checkpoint["model_state_dict"])

        best_loss, continued_best_epoch, history_stage_2 = run_training_phase(
            phase_name="stage_2_continue",
            phase_learning_rate=continue_learning_rate,
            phase_weight_decay=continue_weight_decay,
            phase_epochs=continue_epochs,
            phase_patience=continue_patience,
            starting_best_loss=best_loss,
        )

        histories.append(history_stage_2)
    else:
        continued_best_epoch = None

    training_history = pd.concat(histories, axis=0, ignore_index=True)
    training_history.to_csv(output_dir / "training_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_probs, test_labels = predict_probabilities(
        model=model,
        dataloader=test_loader,
        device=device_obj,
    )

    test_metrics, test_confusion, test_report = calculate_metrics(
        y_true=test_labels,
        probabilities=test_probs,
    )

    test_metadata = locked_test_dataset.metadata.copy()
    test_predictions = test_metadata.copy()

    test_predictions["predicted_class"] = test_probs.argmax(axis=1)
    test_predictions["predicted_name"] = test_predictions[
        "predicted_class"
    ].map(CLASS_NAMES)

    test_predictions["p_neutral"] = test_probs[:, 0]
    test_predictions["p_up"] = test_probs[:, 1]
    test_predictions["p_down"] = test_probs[:, 2]

    test_predictions.to_parquet(
        output_dir / "locked_test_predictions.parquet",
        index=False,
    )

    # Compatibility with existing probability_backtester:
    # it expects fold_*/validation_predictions.parquet.
    fold_dir = output_dir / "fold_01"
    fold_dir.mkdir(parents=True, exist_ok=True)

    test_predictions.to_parquet(
        fold_dir / "validation_predictions.parquet",
        index=False,
    )

    confusion_df = pd.DataFrame(
        test_confusion,
        index=[f"actual_{CLASS_NAMES[i]}" for i in CLASS_IDS],
        columns=[f"predicted_{CLASS_NAMES[i]}" for i in CLASS_IDS],
    )

    confusion_df.to_csv(output_dir / "locked_test_confusion_matrix.csv")

    with (output_dir / "locked_test_classification_report.json").open("w") as file:
        json.dump(test_report, file, indent=2)

    summary = {
        "experiment_name": experiment_name,
        "best_early_stop_loss": float(best_loss),
        "stage_1_best_epoch": int(best_epoch),
        "stage_2_continued_best_epoch": (
            None if continued_best_epoch is None else int(continued_best_epoch)
        ),
        "train_sequences": int(len(train_dataset)),
        "early_stop_sequences": int(len(early_stop_dataset)),
        "locked_test_sequences": int(len(locked_test_dataset)),
        "sequence_length": int(sequence_length),
        "num_features": int(len(feature_columns)),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "continue_training": bool(continue_training),
        "continue_learning_rate": float(continue_learning_rate),
        "continue_weight_decay": float(continue_weight_decay),
        "device": str(device_obj),
        **test_metrics,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "locked_test_summary.csv", index=False)

    with (output_dir / "locked_test_metrics.json").open("w") as file:
        json.dump(summary, file, indent=2)

    print("\nLocked test classification result")
    print(
        summary_df[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "log_loss",
                "multiclass_brier_score",
                "num_features",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved locked-test outputs to: {output_dir}")

    return summary_df