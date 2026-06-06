from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from quant_research.features.feature_builder import MODEL_FEATURES
from quant_research.models.neural_models import (
    TransformerSequenceClassifier,
)
from quant_research.models.sequence_dataset import (
    build_train_validation_datasets,
)
from quant_research.utils.paths import PROCESSED_DATA_DIR


CLASS_IDS = [0, 1, 2]

CLASS_NAMES = {
    0: "neutral",
    1: "up",
    2: "down",
}


def get_device(device: str | None = None) -> torch.device:
    """
    Pick device.

    On Mac, MPS may work, but CPU is often more stable for debugging.
    """
    if device is not None:
        return torch.device(device)

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int = 3,
) -> float:
    one_hot = np.eye(number_of_classes)[y_true]

    return float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1,
            )
        )
    )


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict, np.ndarray, dict]:
    predictions = probabilities.argmax(axis=1)

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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = len(y_batch)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    probabilities = []
    labels = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)

        logits = model(X_batch)
        probs = torch.softmax(logits, dim=1)

        probabilities.append(probs.cpu().numpy())
        labels.append(y_batch.numpy())

    probabilities = np.concatenate(probabilities, axis=0)
    labels = np.concatenate(labels, axis=0)

    return probabilities, labels


def get_class_weights_from_dataset(dataset) -> torch.Tensor:
    labels = dataset.labels.numpy()

    counts = np.bincount(labels, minlength=3)
    total = counts.sum()

    weights = total / (len(counts) * np.maximum(counts, 1))

    return torch.tensor(weights, dtype=torch.float32)


def train_transformer_one_fold(
    fold_directory: Path,
    output_directory: Path,
    sequence_length: int = 21,
    batch_size: int = 256,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 5,
    use_class_weights: bool = False,
    device: str | None = None,
    random_seed: int = 42,
    feature_columns: list[str] | None = None,
) -> dict:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device_obj = get_device(device)

    fold_name = fold_directory.name
    if feature_columns is None:
        feature_columns = MODEL_FEATURES
    train_path = fold_directory / "train.parquet"
    validation_path = fold_directory / "validation.parquet"

    train_dataset, validation_dataset, scaler = (
        build_train_validation_datasets(
            train_path=train_path,
            validation_path=validation_path,
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

    validation_loader = DataLoader(
        validation_dataset,
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

    if use_class_weights:
        class_weights = get_class_weights_from_dataset(
            train_dataset
        ).to(device_obj)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    fold_output_directory = output_directory / fold_name
    fold_output_directory.mkdir(parents=True, exist_ok=True)

    best_validation_loss = np.inf
    best_epoch = -1
    epochs_without_improvement = 0

    history = []

    best_model_path = fold_output_directory / "transformer_model.pt"
    scaler_path = fold_output_directory / "scaler.pkl"

    progress = tqdm(
        range(1, max_epochs + 1),
        desc=f"Training {fold_name}",
        unit="epoch",
    )

    for epoch in progress:
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device_obj,
        )

        validation_probabilities, validation_labels = (
            predict_probabilities(
                model=model,
                dataloader=validation_loader,
                device=device_obj,
            )
        )

        validation_loss = log_loss(
            validation_labels,
            validation_probabilities,
            labels=CLASS_IDS,
        )

        scheduler.step(validation_loss)

        epoch_metrics, _, _ = calculate_metrics(
            y_true=validation_labels,
            probabilities=validation_probabilities,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **epoch_metrics,
        }

        history.append(history_row)

        progress.set_postfix(
            val_loss=f"{validation_loss:.4f}",
            macro_f1=f"{epoch_metrics['macro_f1']:.4f}",
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "sequence_length": sequence_length,
                        "num_features": len(feature_columns),
                        "d_model": d_model,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                        "dim_feedforward": dim_feedforward,
                        "dropout": dropout,
                    },
                },
                best_model_path,
            )

            with scaler_path.open("wb") as file:
                pickle.dump(scaler, file)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Reload best model before final validation predictions.
    checkpoint = torch.load(
        best_model_path,
        map_location=device_obj,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    validation_probabilities, validation_labels = (
        predict_probabilities(
            model=model,
            dataloader=validation_loader,
            device=device_obj,
        )
    )

    metrics, matrix, report = calculate_metrics(
        y_true=validation_labels,
        probabilities=validation_probabilities,
    )
    metrics.update(
        {
            "fold": fold_name,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss),
            "train_sequences": int(len(train_dataset)),
            "validation_sequences": int(len(validation_dataset)),
            "sequence_length": int(sequence_length),
            "num_features": int(len(feature_columns)),
            "d_model": int(d_model),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "use_class_weights": bool(use_class_weights),
            "device": str(device_obj),
        }
    )
    if feature_columns is None:
        feature_columns = MODEL_FEATURES
    validation_metadata = validation_dataset.metadata.copy()
    validation_predictions = validation_metadata.copy()
    print(f"{fold_name}: training with {len(feature_columns)} features")
    validation_predictions["predicted_class"] = (
        validation_probabilities.argmax(axis=1)
    )

    validation_predictions["predicted_name"] = (
        validation_predictions["predicted_class"].map(CLASS_NAMES)
    )

    validation_predictions["p_neutral"] = validation_probabilities[:, 0]
    validation_predictions["p_up"] = validation_probabilities[:, 1]
    validation_predictions["p_down"] = validation_probabilities[:, 2]

    validation_predictions_path = (
        fold_output_directory
        / "validation_predictions.parquet"
    )

    validation_predictions.to_parquet(
        validation_predictions_path,
        index=False,
    )

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

    confusion_df.to_csv(
        fold_output_directory / "confusion_matrix.csv"
    )

    with (
        fold_output_directory / "classification_report.json"
    ).open("w") as file:
        json.dump(report, file, indent=2)

    with (fold_output_directory / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        fold_output_directory / "training_history.csv",
        index=False,
    )

    tqdm.write(
        f"{fold_name}: "
        f"best_epoch={best_epoch}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"log_loss={metrics['log_loss']:.4f}"
    )

    return metrics


def train_transformer_walk_forward(
    experiment_name: str,
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    max_folds: int | None = None,
    sequence_length: int = 21,
    batch_size: int = 256,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 5,
    use_class_weights: bool = False,
    device: str | None = None,
    random_seed: int = 42,
    walk_forward_directory: Path | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Train transformer sequence classifier across walk-forward folds.

    Hyperparameters are passed from the notebook.
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURES
    if walk_forward_directory is None:
        walk_forward_directory = (
            Path(processed_dir)
            / f"walk_forward_{timeframe}"
        )
    else:
        walk_forward_directory = Path(walk_forward_directory)

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

    if max_folds is not None:
        fold_directories = fold_directories[:max_folds]

    if not fold_directories:
        raise ValueError("No fold directories found.")

    output_directory = (
        Path(processed_dir)
        / f"transformer_walk_forward_{timeframe}"
        / experiment_name
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for fold_directory in fold_directories:
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
            feature_columns=feature_columns,
        )

        all_metrics.append(metrics)

    summary_df = pd.DataFrame(all_metrics)

    summary_path = output_directory / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)

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
    }

    with (output_directory / "aggregate_metrics.json").open("w") as file:
        json.dump(aggregate_metrics, file, indent=2)

    print("\nTransformer walk-forward results")
    display_columns = [
        "fold",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "multiclass_brier_score",
        "best_epoch",
        "num_features",
    ]

    print(summary_df[display_columns].to_string(index=False))

    print("\nMean metrics")
    for name, value in aggregate_metrics.items():
        print(f"{name}: {value}")

    print(f"\nSaved results to: {output_directory}")

    return summary_df
def continue_transformer_one_fold(
    source_fold_directory: Path,
    split_fold_directory: Path,
    output_directory: Path,
    sequence_length: int = 21,
    batch_size: int = 256,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-6,
    extra_epochs: int = 10,
    patience: int = 5,
    use_class_weights: bool = False,
    device: str | None = None,
    random_seed: int = 42,
    feature_columns: list[str] | None = None,
    walk_forward_directory: Path | None = None,
) -> dict:
    """
    Continue training an already-saved transformer fold checkpoint.

    source_fold_directory:
        Folder containing the existing transformer_model.pt.

    split_fold_directory:
        Original walk-forward fold folder containing train.parquet
        and validation.parquet.

    output_directory:
        New experiment directory where the continued model and outputs
        will be saved.
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURES
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device_obj = get_device(device)

    fold_name = split_fold_directory.name

    checkpoint_path = source_fold_directory / "transformer_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing existing transformer checkpoint: {checkpoint_path}"
        )

    train_path = split_fold_directory / "train.parquet"
    validation_path = split_fold_directory / "validation.parquet"

    train_dataset, validation_dataset, scaler = (
        build_train_validation_datasets(
            train_path=train_path,
            validation_path=validation_path,
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

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device_obj,
    )

    config = checkpoint["config"]

    model = TransformerSequenceClassifier(
        num_features=config["num_features"],
        sequence_length=config["sequence_length"],
        num_classes=len(CLASS_IDS),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
    ).to(device_obj)

    model.load_state_dict(checkpoint["model_state_dict"])

    if use_class_weights:
        class_weights = get_class_weights_from_dataset(
            train_dataset
        ).to(device_obj)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    fold_output_directory = output_directory / fold_name
    fold_output_directory.mkdir(parents=True, exist_ok=True)

    best_model_path = fold_output_directory / "transformer_model.pt"
    scaler_path = fold_output_directory / "scaler.pkl"

    # Start from the source model's validation loss.
    validation_probabilities, validation_labels = (
        predict_probabilities(
            model=model,
            dataloader=validation_loader,
            device=device_obj,
        )
    )

    best_validation_loss = log_loss(
        validation_labels,
        validation_probabilities,
        labels=CLASS_IDS,
    )

    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        best_model_path,
    )

    with scaler_path.open("wb") as file:
        pickle.dump(scaler, file)

    progress = tqdm(
        range(1, extra_epochs + 1),
        desc=f"Continuing {fold_name}",
        unit="epoch",
    )

    for epoch in progress:
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device_obj,
        )

        validation_probabilities, validation_labels = (
            predict_probabilities(
                model=model,
                dataloader=validation_loader,
                device=device_obj,
            )
        )

        validation_loss = log_loss(
            validation_labels,
            validation_probabilities,
            labels=CLASS_IDS,
        )

        scheduler.step(validation_loss)

        epoch_metrics, _, _ = calculate_metrics(
            y_true=validation_labels,
            probabilities=validation_probabilities,
        )

        history_row = {
            "continued_epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **epoch_metrics,
        }

        history.append(history_row)

        progress.set_postfix(
            val_loss=f"{validation_loss:.4f}",
            macro_f1=f"{epoch_metrics['macro_f1']:.4f}",
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                best_model_path,
            )

            with scaler_path.open("wb") as file:
                pickle.dump(scaler, file)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(
        best_model_path,
        map_location=device_obj,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    validation_probabilities, validation_labels = (
        predict_probabilities(
            model=model,
            dataloader=validation_loader,
            device=device_obj,
        )
    )

    metrics, matrix, report = calculate_metrics(
        y_true=validation_labels,
        probabilities=validation_probabilities,
    )

    metrics.update(

        {

            "fold": fold_name,

            "best_epoch": int(best_epoch),

            "best_validation_loss": float(best_validation_loss),

            "train_sequences": int(len(train_dataset)),

            "validation_sequences": int(len(validation_dataset)),

            "sequence_length": int(sequence_length),

            "num_features": int(len(feature_columns)),   # ADD THIS

            "d_model": int(d_model),

            "num_heads": int(num_heads),

            "num_layers": int(num_layers),

            "dim_feedforward": int(dim_feedforward),

            "dropout": float(dropout),

            "learning_rate": float(learning_rate),

            "weight_decay": float(weight_decay),

            "batch_size": int(batch_size),

            "use_class_weights": bool(use_class_weights),

            "device": str(device_obj),

        }

    )

    validation_metadata = validation_dataset.metadata.copy()
    validation_predictions = validation_metadata.copy()

    validation_predictions["predicted_class"] = (
        validation_probabilities.argmax(axis=1)
    )

    validation_predictions["predicted_name"] = (
        validation_predictions["predicted_class"].map(CLASS_NAMES)
    )

    validation_predictions["p_neutral"] = validation_probabilities[:, 0]
    validation_predictions["p_up"] = validation_probabilities[:, 1]
    validation_predictions["p_down"] = validation_probabilities[:, 2]

    validation_predictions.to_parquet(
        fold_output_directory / "validation_predictions.parquet",
        index=False,
    )

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

    confusion_df.to_csv(
        fold_output_directory / "confusion_matrix.csv"
    )

    with (
        fold_output_directory / "classification_report.json"
    ).open("w") as file:
        json.dump(report, file, indent=2)

    with (fold_output_directory / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        fold_output_directory / "continued_training_history.csv",
        index=False,
    )

    tqdm.write(
        f"{fold_name}: "
        f"continued_best_epoch={best_epoch}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"log_loss={metrics['log_loss']:.4f}"
    )

    return metrics


def continue_transformer_walk_forward(
    source_experiment_name: str,
    continued_experiment_name: str,
    timeframe: str = "5min",
    processed_dir: Path = PROCESSED_DATA_DIR,
    max_folds: int | None = None,
    sequence_length: int = 21,
    batch_size: int = 256,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-6,
    extra_epochs: int = 10,
    patience: int = 5,
    use_class_weights: bool = False,
    device: str | None = None,
    random_seed: int = 42,
    walk_forward_directory: Path | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Continue training an existing transformer walk-forward experiment.

    This resumes from each fold's saved transformer_model.pt and saves
    the continued version as a new experiment.
    """
    processed_dir = Path(processed_dir)

    source_experiment_directory = (
        processed_dir
        / f"transformer_walk_forward_{timeframe}"
        / source_experiment_name
    )

    if not source_experiment_directory.exists():
        raise FileNotFoundError(
            f"Missing source experiment directory: "
            f"{source_experiment_directory}"
        )

    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    if walk_forward_directory is None:
        walk_forward_directory = (
            processed_dir
            / f"walk_forward_{timeframe}"
        )
    else:
        walk_forward_directory = Path(walk_forward_directory)

    if not walk_forward_directory.exists():
        raise FileNotFoundError(
            f"Missing walk-forward directory: {walk_forward_directory}"
        )

    split_fold_directories = sorted(
        path
        for path in walk_forward_directory.glob("fold_*")
        if path.is_dir()
    )

    if max_folds is not None:
        split_fold_directories = split_fold_directories[:max_folds]

    output_directory = (
        processed_dir
        / f"transformer_walk_forward_{timeframe}"
        / continued_experiment_name
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for split_fold_directory in split_fold_directories:
        fold_name = split_fold_directory.name

        source_fold_directory = (
            source_experiment_directory
            / fold_name
        )
        metrics = continue_transformer_one_fold(
            source_fold_directory=source_fold_directory,
            split_fold_directory=split_fold_directory,
            output_directory=output_directory,
            sequence_length=sequence_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            extra_epochs=extra_epochs,
            patience=patience,
            use_class_weights=use_class_weights,
            device=device,
            random_seed=random_seed,
            feature_columns=feature_columns,
        )

        all_metrics.append(metrics)

    summary_df = pd.DataFrame(all_metrics)

    summary_path = output_directory / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)

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
    }

    with (output_directory / "aggregate_metrics.json").open("w") as file:
        json.dump(aggregate_metrics, file, indent=2)

    print("\nContinued transformer walk-forward results")
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
                "continued_best_epoch",
            ]
        ].to_string(index=False)
    )

    print("\nMean metrics")
    for name, value in aggregate_metrics.items():
        print(f"{name}: {value}")

    print(f"\nSaved continued results to: {output_directory}")

    return summary_df