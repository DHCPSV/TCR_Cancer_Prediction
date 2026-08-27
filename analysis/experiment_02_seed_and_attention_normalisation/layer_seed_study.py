"""Layer-seed comparison and the optional 300-epoch diagnostic."""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analysis import protocol, reporting, training
from analysis.attention_mil import AttentionMIL
from pipeline import provenance


EXPERIMENT_ID = "experiment_02_seed_and_attention_normalisation"
EPOCHS = 50
METRIC_INTERVAL = 5
LEARNING_RATE = 1e-3
ACCUMULATION = 4
SHUFFLE_SEED = 913271
FOLDS = range(5)
CHECKPOINT_EPOCHS = (0, 5, 10, 25, 50, 100, 200, 300)
FACTORIZATION_SEEDS = (
    ("S03", "lucky"),
    ("S04", "moderate"),
    ("S06", "moderate"),
    ("S08", "ordinary"),
    ("S09", "ordinary"),
)
BEHAVIOUR_ORDER = ("lucky", "moderate", "ordinary")
TRAJECTORY_SEEDS = ("S01", "S02", "S03")
FACTORIZATION_HISTORY_EPOCHS = (1, *range(METRIC_INTERVAL, EPOCHS + 1, METRIC_INTERVAL))
RUN_ARTIFACTS = protocol.repo_path(f"artifacts/runs/{EXPERIMENT_ID}")
CHECKPOINTS = protocol.repo_path(f"artifacts/checkpoints/{EXPERIMENT_ID}/layer_seed_study")


def seed_value(seed_id: str) -> int:
    return int(protocol.SEED_REGISTRY[seed_id]["value"])


def internal_rows(chain: str = "alpha") -> pd.DataFrame:
    frame = protocol.read_manifest("internal_representations")
    frame = frame.loc[
        (frame["role"] == "internal")
        & (frame["chain"] == chain)
        & (frame["representation_method"] == "sceptr")
    ].copy()
    if frame["subject_id"].nunique() != 169:
        raise ValueError(f"{chain}: expected 169 internal patients")
    if set(frame["embedding_dim"].astype(int)) != {64}:
        raise ValueError("Seed diagnostics require 64D SCEPTR embeddings")
    folds = protocol.read_fixed_folds(chain, frame["subject_id"])
    frame["fold"] = frame["subject_id"].map(folds)
    # Preserve the recorded representation-manifest order before applying
    # the fixed per-epoch patient shuffle.
    return frame.reset_index(drop=True)


def reference_model(seed: int) -> AttentionMIL:
    training.set_seed(seed)
    return AttentionMIL("sparsemax")


def layer_seeded_model(attention_seed: int, classifier_seed: int, device: torch.device) -> AttentionMIL:
    model = reference_model(0)
    attention_reference = reference_model(attention_seed)
    classifier_reference = reference_model(classifier_seed)
    model.attention_score.load_state_dict(attention_reference.attention_score.state_dict())
    model.classifier.load_state_dict(classifier_reference.classifier.state_dict())
    return model.to(device)


def factor_checkpoint(attention_id: str, classifier_id: str, fold: int) -> Path:
    return CHECKPOINTS / "factorization" / f"att_{attention_id}_cls_{classifier_id}" / f"fold_{fold}.pt"


def factor_history_path(attention_id: str, classifier_id: str, fold: int) -> Path:
    return (
        RUN_ARTIFACTS
        / "factorization_training_metrics"
        / f"att_{attention_id}_cls_{classifier_id}"
        / f"fold_{fold}.csv"
    )


def trajectory_checkpoint(seed_id: str, fold: int, epoch: int) -> Path:
    return CHECKPOINTS / "trajectory" / seed_id / f"fold_{fold}" / f"epoch_{epoch}.pt"


def factor_checkpoint_contract(attention_id: str, classifier_id: str, fold: int) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "kind": "attention_classifier_factorization",
        "attention_seed_id": attention_id,
        "attention_seed_value": str(seed_value(attention_id)),
        "classifier_seed_id": classifier_id,
        "classifier_seed_value": str(seed_value(classifier_id)),
        "fold": fold,
        "epochs": EPOCHS,
        "metric_interval": METRIC_INTERVAL,
        "learning_rate": LEARNING_RATE,
        "accumulation": ACCUMULATION,
        "shuffle_seed": SHUFFLE_SEED,
    }


def save_factor_checkpoint(path: Path, model: AttentionMIL, attention_id: str, classifier_id: str, fold: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            **factor_checkpoint_contract(attention_id, classifier_id, fold),
        },
        temporary,
    )
    temporary.replace(path)


def load_factor_checkpoint(path: Path, attention_id: str, classifier_id: str, fold: int, device: torch.device) -> AttentionMIL:
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = factor_checkpoint_contract(attention_id, classifier_id, fold)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: {key}={payload.get(key)!r}, expected {value!r}")
    model = layer_seeded_model(seed_value(attention_id), seed_value(classifier_id), device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def factor_history_frame(
    rows: list[dict],
    attention_id: str,
    classifier_id: str,
    fold: int,
) -> pd.DataFrame:
    identity = {
        "attention_seed_id": attention_id,
        "attention_seed_value": str(seed_value(attention_id)),
        "attention_seed_group": dict(FACTORIZATION_SEEDS)[attention_id],
        "classifier_seed_id": classifier_id,
        "classifier_seed_value": str(seed_value(classifier_id)),
        "classifier_seed_group": dict(FACTORIZATION_SEEDS)[classifier_id],
        "fold": fold,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "accumulation": ACCUMULATION,
        "metric_interval": METRIC_INTERVAL,
        "shuffle_seed": SHUFFLE_SEED,
    }
    return pd.DataFrame([identity | row for row in rows])


def validate_factor_history(
    frame: pd.DataFrame,
    attention_id: str,
    classifier_id: str,
    fold: int,
) -> pd.DataFrame:
    required = {
        "attention_seed_id", "attention_seed_value", "attention_seed_group",
        "classifier_seed_id", "classifier_seed_value", "classifier_seed_group",
        "fold", "epochs", "learning_rate", "accumulation", "metric_interval", "shuffle_seed", "epoch",
        "training_auc", "held_out_auc", "training_bce", "held_out_bce",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Factorization history is missing columns: {sorted(missing)}")
    expected_identity = {
        "attention_seed_id": attention_id,
        "attention_seed_value": str(seed_value(attention_id)),
        "attention_seed_group": dict(FACTORIZATION_SEEDS)[attention_id],
        "classifier_seed_id": classifier_id,
        "classifier_seed_value": str(seed_value(classifier_id)),
        "classifier_seed_group": dict(FACTORIZATION_SEEDS)[classifier_id],
        "fold": fold,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "accumulation": ACCUMULATION,
        "metric_interval": METRIC_INTERVAL,
        "shuffle_seed": SHUFFLE_SEED,
    }
    for column, expected in expected_identity.items():
        if isinstance(expected, str):
            values = set(frame[column].astype(str))
            matches = values == {expected}
        elif isinstance(expected, float):
            numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            values = set(numeric)
            matches = len(values) == 1 and bool(np.isclose(numeric[0], expected, rtol=0, atol=1e-15))
        else:
            numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            integral = np.isfinite(numeric).all() and np.equal(numeric, np.floor(numeric)).all()
            values = set(numeric.astype(int)) if integral else set(numeric)
            matches = integral and values == {expected}
        if not matches:
            raise ValueError(f"Factorization history {column}={values!r}, expected {expected!r}")
    epoch_values = pd.to_numeric(frame["epoch"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(epoch_values).all() or not np.equal(epoch_values, np.floor(epoch_values)).all():
        raise ValueError("Factorization history epochs must be finite integers")
    epochs = tuple(epoch_values.astype(int))
    if epochs != FACTORIZATION_HISTORY_EPOCHS:
        raise ValueError(f"Factorization history epochs={epochs!r}, expected {FACTORIZATION_HISTORY_EPOCHS!r}")
    metric_columns = ["training_auc", "held_out_auc", "training_bce", "held_out_bce"]
    values = frame[metric_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Factorization history contains non-finite metrics")
    auc_values = frame[["training_auc", "held_out_auc"]].to_numpy(dtype=float)
    bce_values = frame[["training_bce", "held_out_bce"]].to_numpy(dtype=float)
    if ((auc_values < 0) | (auc_values > 1)).any():
        raise ValueError("Factorization history AUC values must be between 0 and 1")
    if (bce_values < 0).any():
        raise ValueError("Factorization history BCE values must be non-negative")
    return frame.sort_values("epoch").reset_index(drop=True)


def read_factor_history(
    attention_id: str,
    classifier_id: str,
    fold: int,
    *,
    strict_pair: bool = False,
) -> pd.DataFrame:
    path = factor_history_path(attention_id, classifier_id, fold)
    checkpoint = factor_checkpoint(attention_id, classifier_id, fold)
    contract = factor_checkpoint_contract(attention_id, classifier_id, fold)
    if strict_pair:
        training.validate_checkpoint_history_pair(
            checkpoint, path, contract, repo=protocol.REPO
        )
    elif not training.checkpoint_history_pair_is_valid(
        checkpoint, path, contract, repo=protocol.REPO
    ):
        raise ValueError(f"Stale or incomplete checkpoint/history pair: {path}")
    frame = pd.read_csv(
        path,
        dtype={"attention_seed_value": str, "classifier_seed_value": str},
    )
    return validate_factor_history(frame, attention_id, classifier_id, fold)


def validate_factor_history_collection(frame: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    expected_blocks = len(FACTORIZATION_SEEDS) ** 2 * len(FOLDS)
    expected_rows = expected_blocks * len(FACTORIZATION_HISTORY_EPOCHS)
    if len(frame) != expected_rows:
        raise ValueError(f"Factorization history has {len(frame)} rows, expected {expected_rows}")
    for attention_id, _ in FACTORIZATION_SEEDS:
        for classifier_id, _ in FACTORIZATION_SEEDS:
            for fold in FOLDS:
                block = frame.loc[
                    (frame["attention_seed_id"] == attention_id)
                    & (frame["classifier_seed_id"] == classifier_id)
                    & (pd.to_numeric(frame["fold"], errors="coerce") == fold)
                ].copy()
                blocks.append(validate_factor_history(block, attention_id, classifier_id, fold))
    if len(blocks) != expected_blocks or sum(len(block) for block in blocks) != expected_rows:
        raise ValueError("Incomplete attention/classifier factorization training history")
    return pd.concat(blocks, ignore_index=True)


def factorization_internal(device: torch.device) -> pd.DataFrame:
    frame = internal_rows("alpha")
    records = frame.to_dict("records")
    tensors = protocol.load_tensors(frame, device)
    output = []
    histories = []
    total = len(FACTORIZATION_SEEDS) ** 2
    progress = provenance.Progress(
        "Experiment 02 layer-initialisation factorisation",
        total,
        updates=5,
    )
    reused_folds = 0
    for attention_id, attention_label in FACTORIZATION_SEEDS:
        for classifier_id, classifier_label in FACTORIZATION_SEEDS:
            for fold in FOLDS:
                train_rows = [row for row in records if int(row["fold"]) != fold]
                test_rows = [row for row in records if int(row["fold"]) == fold]
                path = factor_checkpoint(attention_id, classifier_id, fold)
                try:
                    model = load_factor_checkpoint(path, attention_id, classifier_id, fold, device)
                    history = read_factor_history(attention_id, classifier_id, fold)
                except (
                    FileNotFoundError,
                    EOFError,
                    OSError,
                    KeyError,
                    RuntimeError,
                    ValueError,
                    pickle.UnpicklingError,
                ):
                    model = layer_seeded_model(seed_value(attention_id), seed_value(classifier_id), device)
                    history_rows: list[dict] = []
                    training.train_binary_model(
                        model,
                        train_rows,
                        tensors,
                        epochs=EPOCHS,
                        learning_rate=LEARNING_RATE,
                        accumulation=ACCUMULATION,
                        shuffle_seed=SHUFFLE_SEED,
                        held_out_rows=test_rows,
                        metric_interval=METRIC_INTERVAL,
                        history=history_rows,
                    )
                    save_factor_checkpoint(path, model, attention_id, classifier_id, fold)
                    history = factor_history_frame(history_rows, attention_id, classifier_id, fold)
                    history = validate_factor_history(history, attention_id, classifier_id, fold)
                    history_path = factor_history_path(attention_id, classifier_id, fold)
                    protocol.atomic_csv(history, history_path)
                    training.bind_checkpoint_history(
                        path,
                        history_path,
                        factor_checkpoint_contract(attention_id, classifier_id, fold),
                        repo=protocol.REPO,
                    )
                else:
                    reused_folds += 1
                histories.append(history)
                scores = training.predict_scores(model, test_rows, tensors)
                for row in test_rows:
                    output.append(
                        {
                            "attention_seed_id": attention_id,
                            "attention_seed_value": str(seed_value(attention_id)),
                            "attention_seed_group": attention_label,
                            "classifier_seed_id": classifier_id,
                            "classifier_seed_value": str(seed_value(classifier_id)),
                            "classifier_seed_group": classifier_label,
                            "fold": fold,
                            "subject_id": row["subject_id"],
                            "cohort": row["cohort"],
                            "label": int(row["label"]),
                            "score": scores[row["subject_id"]],
                            "tcr_count": int(row["tcr_count"]),
                        }
                    )
            progress.advance()
    result = pd.DataFrame(output)
    if len(result) != 25 * 169:
        raise ValueError(f"Factorization produced {len(result)} rows, expected {25 * 169}")
    training_metrics = validate_factor_history_collection(pd.concat(histories, ignore_index=True))
    protocol.atomic_csv(result, RUN_ARTIFACTS / "factorization_predictions.csv")
    protocol.atomic_csv(training_metrics, RUN_ARTIFACTS / "factorization_training_metrics.csv")
    progress.summary(
        reused_folds=reused_folds,
        trained_folds=total * len(FOLDS) - reused_folds,
    )
    return result


def evaluate_rows(
    model: AttentionMIL,
    rows: list[dict],
    tensors: dict[str, torch.Tensor],
    weights: dict[int, float],
) -> tuple[dict[str, float], dict[str, float]]:
    labels = []
    scores = {}
    losses = []
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    model.eval()
    with torch.no_grad():
        for row in rows:
            label = int(row["label"])
            target = torch.tensor([[float(label)]], device=tensors[row["subject_id"]].device)
            logit = model(tensors[row["subject_id"]])
            loss = criterion(logit, target).mean() * weights[label]
            score = float(torch.sigmoid(logit).detach().cpu().reshape(()).item())
            labels.append(label)
            scores[row["subject_id"]] = score
            losses.append(float(loss.detach().cpu().item()))
    metrics = reporting.binary_metrics(labels, [scores[row["subject_id"]] for row in rows])
    metrics["balanced_bce"] = float(np.mean(losses))
    return metrics, scores


def save_trajectory_checkpoint(path: Path, model: AttentionMIL, seed_id: str, fold: int, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "experiment_id": EXPERIMENT_ID,
            "kind": "layer_checkpoint_trajectory",
            "seed_id": seed_id,
            "seed_value": str(seed_value(seed_id)),
            "fold": fold,
            "checkpoint_epoch": epoch,
            "shuffle_seed": SHUFFLE_SEED,
        },
        temporary,
    )
    temporary.replace(path)


def load_trajectory_checkpoint(path: Path, seed_id: str, fold: int, epoch: int, device: torch.device) -> AttentionMIL:
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "kind": "layer_checkpoint_trajectory",
        "seed_id": seed_id,
        "seed_value": str(seed_value(seed_id)),
        "fold": fold,
        "checkpoint_epoch": epoch,
        "shuffle_seed": SHUFFLE_SEED,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: {key}={payload.get(key)!r}, expected {value!r}")
    model = layer_seeded_model(seed_value(seed_id), seed_value(seed_id), device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def record_trajectory_state(
    model: AttentionMIL,
    seed_id: str,
    fold: int,
    epoch: int,
    train_rows: list[dict],
    test_rows: list[dict],
    tensors: dict[str, torch.Tensor],
    weights: dict[int, float],
) -> tuple[dict, list[dict]]:
    train_metrics, _ = evaluate_rows(model, train_rows, tensors, weights)
    test_metrics, scores = evaluate_rows(model, test_rows, tensors, weights)
    metric = {
        "seed_id": seed_id,
        "seed_value": str(seed_value(seed_id)),
        "fold": fold,
        "checkpoint_epoch": epoch,
        "train_auc": train_metrics["auc"],
        "test_auc": test_metrics["auc"],
        "train_balanced_bce": train_metrics["balanced_bce"],
        "test_balanced_bce": test_metrics["balanced_bce"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    }
    predictions = [
        {
            "seed_id": seed_id,
            "seed_value": str(seed_value(seed_id)),
            "fold": fold,
            "checkpoint_epoch": epoch,
            "split": "internal_oof",
            "subject_id": row["subject_id"],
            "cohort": row["cohort"],
            "label": int(row["label"]),
            "score": scores[row["subject_id"]],
            "tcr_count": int(row["tcr_count"]),
        }
        for row in test_rows
    ]
    return metric, predictions


def train_trajectory_fold(
    seed_id: str,
    fold: int,
    train_rows: list[dict],
    test_rows: list[dict],
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    all_paths = [trajectory_checkpoint(seed_id, fold, epoch) for epoch in CHECKPOINT_EPOCHS]
    weights = training.class_weights([int(row["label"]) for row in train_rows])
    metric_rows = []
    prediction_rows = []
    if all(path.exists() for path in all_paths):
        for epoch, path in zip(CHECKPOINT_EPOCHS, all_paths):
            model = load_trajectory_checkpoint(path, seed_id, fold, epoch, device)
            metric, predictions = record_trajectory_state(
                model, seed_id, fold, epoch, train_rows, test_rows, tensors, weights
            )
            metric_rows.append(metric)
            prediction_rows.extend(predictions)
        return metric_rows, prediction_rows

    model = layer_seeded_model(seed_value(seed_id), seed_value(seed_id), device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    shuffler = random.Random(SHUFFLE_SEED)

    def record(epoch: int) -> None:
        save_trajectory_checkpoint(
            trajectory_checkpoint(seed_id, fold, epoch),
            model,
            seed_id,
            fold,
            epoch,
        )
        metric, predictions = record_trajectory_state(
            model, seed_id, fold, epoch, train_rows, test_rows, tensors, weights
        )
        metric_rows.append(metric)
        prediction_rows.extend(predictions)

    record(0)
    for epoch in range(1, CHECKPOINT_EPOCHS[-1] + 1):
        order = train_rows.copy()
        shuffler.shuffle(order)
        model.train()
        optimiser.zero_grad(set_to_none=True)
        for index, row in enumerate(order, 1):
            target = torch.tensor([[float(row["label"])]], device=device)
            loss = criterion(model(tensors[row["subject_id"]]), target).mean() * weights[int(row["label"])]
            loss.backward()  # summed, matching the released 50-epoch models
            if index % ACCUMULATION == 0 or index == len(order):
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
        if epoch in CHECKPOINT_EPOCHS:
            record(epoch)
    return metric_rows, prediction_rows


def diagnostics(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the optional selected-seed 300-epoch learning diagnostic."""
    frame = internal_rows("alpha")
    records = frame.to_dict("records")
    tensors = protocol.load_tensors(frame, device)
    all_metrics = []
    all_predictions = []
    total = len(TRAJECTORY_SEEDS) * len(FOLDS)
    progress = provenance.Progress(
        "Experiment 02 selected-seed 300-epoch diagnostic",
        total,
        updates=5,
    )
    reused_folds = 0
    for seed_id in TRAJECTORY_SEEDS:
        for fold in FOLDS:
            train_rows = [row for row in records if int(row["fold"]) != fold]
            test_rows = [row for row in records if int(row["fold"]) == fold]
            reused = all(
                trajectory_checkpoint(seed_id, fold, epoch).exists()
                for epoch in CHECKPOINT_EPOCHS
            )
            metrics, predictions = train_trajectory_fold(
                seed_id,
                fold,
                train_rows,
                test_rows,
                tensors,
                device,
            )
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)
            reused_folds += int(reused)
            progress.advance()
    fold_metrics = pd.DataFrame(all_metrics)
    predictions = pd.DataFrame(all_predictions)
    expected_metric_rows = len(TRAJECTORY_SEEDS) * len(FOLDS) * len(CHECKPOINT_EPOCHS)
    expected_prediction_rows = len(TRAJECTORY_SEEDS) * len(CHECKPOINT_EPOCHS) * 169
    if len(fold_metrics) != expected_metric_rows:
        raise ValueError(
            f"300-epoch diagnostic produced {len(fold_metrics)} metric rows; "
            f"expected {expected_metric_rows}"
        )
    if len(predictions) != expected_prediction_rows:
        raise ValueError(
            f"300-epoch diagnostic produced {len(predictions)} predictions; "
            f"expected {expected_prediction_rows}"
        )
    metric_key = ["seed_id", "fold", "checkpoint_epoch"]
    prediction_key = ["seed_id", "checkpoint_epoch", "subject_id"]
    if fold_metrics.duplicated(metric_key).any():
        raise ValueError("Duplicate fold metric in 300-epoch diagnostic")
    if predictions.duplicated(prediction_key).any():
        raise ValueError("Duplicate patient prediction in 300-epoch diagnostic")
    protocol.atomic_csv(fold_metrics, RUN_ARTIFACTS / "trajectory_fold_metrics.csv")
    protocol.atomic_csv(predictions, RUN_ARTIFACTS / "trajectory_oof.csv")
    progress.summary(
        reused_folds=reused_folds,
        trained_folds=total - reused_folds,
    )
    return fold_metrics, predictions


def self_test() -> None:
    first = layer_seeded_model(11, 17, torch.device("cpu"))
    second = layer_seeded_model(11, 23, torch.device("cpu"))
    assert all(torch.equal(first.attention_score.state_dict()[key], second.attention_score.state_dict()[key]) for key in first.attention_score.state_dict())
    assert any(not torch.equal(first.classifier.state_dict()[key], second.classifier.state_dict()[key]) for key in first.classifier.state_dict())
    values = torch.randn(13, 64)
    assert torch.allclose(first(values), first(values[torch.randperm(13)]), atol=1e-6)
    synthetic_history = []
    for attention_id, attention_tier in FACTORIZATION_SEEDS:
        for classifier_id, classifier_tier in FACTORIZATION_SEEDS:
            for fold in FOLDS:
                for epoch in FACTORIZATION_HISTORY_EPOCHS:
                    synthetic_history.append(
                        {
                            "attention_seed_id": attention_id,
                            "attention_seed_value": str(seed_value(attention_id)),
                            "attention_seed_group": attention_tier,
                            "classifier_seed_id": classifier_id,
                            "classifier_seed_value": str(seed_value(classifier_id)),
                            "classifier_seed_group": classifier_tier,
                            "fold": fold,
                            "epochs": EPOCHS,
                            "learning_rate": LEARNING_RATE,
                            "accumulation": ACCUMULATION,
                            "metric_interval": METRIC_INTERVAL,
                            "shuffle_seed": SHUFFLE_SEED,
                            "epoch": epoch,
                            "training_auc": 0.60,
                            "held_out_auc": 0.55,
                            "training_bce": 0.65,
                            "held_out_bce": 0.70,
                        }
                    )
    validated = validate_factor_history_collection(pd.DataFrame(synthetic_history))
    expected_rows = (
        len(FACTORIZATION_SEEDS) ** 2
        * len(FOLDS)
        * len(FACTORIZATION_HISTORY_EPOCHS)
    )
    assert len(validated) == expected_rows
    print("experiment 02 layer-seed self-test passed")


def internal(device: torch.device) -> pd.DataFrame:
    """Run the 50-epoch attention/classifier factorisation."""
    return factorization_internal(device)
