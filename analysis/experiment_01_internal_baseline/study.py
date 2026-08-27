"""Training and prediction study for the internal representation comparison.

This module owns the experiment contract and artifacts.  It intentionally has
no command-line or plotting code; ``run.py`` orchestrates it and ``report.py``
turns its outputs into manuscript-facing tables and figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis import protocol, training
from analysis.attention_mil import AttentionMIL
from analysis.experiment_01_internal_baseline.methods import METHODS


EXPERIMENT_ID = "experiment_01_internal_baseline"
CHAINS = ("alpha", "beta")
SEEDS = {
    "seed_1": {"value": 777, "seed_group": "lucky", "label": "Seed 1: 777"},
    "seed_2": {"value": 42, "seed_group": "ordinary", "label": "Seed 2: 42"},
    "seed_3": {"value": 123, "seed_group": "moderate", "label": "Seed 3: 123"},
}
SEED_IDS = tuple(SEEDS)
FOLDS = range(5)
EPOCHS = 50
METRIC_INTERVAL = 5
LEARNING_RATE = 1e-3
ACCUMULATION = 4
SHUFFLE_SEED = 913271

RUN_ARTIFACTS = protocol.repo_path(f"artifacts/runs/{EXPERIMENT_ID}")
CHECKPOINTS = protocol.repo_path(f"artifacts/checkpoints/{EXPERIMENT_ID}")
RESULTS = protocol.repo_path(f"results/{EXPERIMENT_ID}")
EXPECTED_SUBJECTS = {"alpha": 169, "beta": 169}
HISTORY_COLUMNS = (
    "epoch",
    "training_auc",
    "held_out_auc",
    "training_bce",
    "held_out_bce",
    "method_id",
    "method_label",
    "chain",
    "seed_id",
    "seed_value",
    "fold",
)
HISTORY_METRICS = ("training_auc", "held_out_auc", "training_bce", "held_out_bce")
HISTORY_KEY = ("method_id", "chain", "seed_id", "fold", "epoch")
HISTORY_EPOCHS = (1, *range(METRIC_INTERVAL, EPOCHS + 1, METRIC_INTERVAL))


def seed_metadata(seed_id: str) -> tuple[int, str]:
    row = SEEDS[seed_id]
    return int(row["value"]), str(row["seed_group"])


def internal_manifest() -> pd.DataFrame:
    frame = protocol.read_manifest("representation_comparison_internal")
    required = {
        "subject_id",
        "chain",
        "cohort",
        "label",
        "role",
        "embedding_file",
        "tcr_count",
        "embedding_dim",
        "representation_method",
        "sha256",
    }
    if missing := required - set(frame):
        raise ValueError(f"Internal representation manifest lacks {sorted(missing)}")
    if not frame["role"].eq("internal").all():
        raise ValueError("Experiment 01 accepts internal subjects only")
    if frame.duplicated(["subject_id", "chain", "representation_method"]).any():
        raise ValueError("Duplicate internal representation rows")
    return frame


def method_rows(manifest: pd.DataFrame, method: tuple, chain: str) -> pd.DataFrame:
    method_id, _, representation, embedding_dim = method
    rows = manifest.loc[
        (manifest["chain"] == chain)
        & (manifest["representation_method"] == representation)
    ].copy()
    if rows["subject_id"].nunique() != EXPECTED_SUBJECTS[chain]:
        raise ValueError(f"{method_id}/{chain}: wrong subject count")
    if set(rows["embedding_dim"].astype(int)) != {embedding_dim}:
        raise ValueError(f"{method_id}/{chain}: wrong embedding dimension")
    folds = protocol.read_fixed_folds(chain, rows["subject_id"])
    rows["fold"] = rows["subject_id"].map(folds)
    canonical = protocol.read_manifest("internal_representations")
    canonical = canonical.loc[canonical["chain"] == chain, "subject_id"].tolist()
    order = {subject_id: index for index, subject_id in enumerate(canonical)}
    rows["_canonical_order"] = rows["subject_id"].map(order)
    if rows["_canonical_order"].isna().any():
        raise ValueError(f"{method_id}/{chain}: unknown canonical subject")
    return rows.sort_values("_canonical_order").drop(columns="_canonical_order").reset_index(drop=True)


def checkpoint_path(method_id: str, chain: str, seed_id: str, fold: int) -> Path:
    return CHECKPOINTS / method_id / chain / seed_id / f"fold_{fold}.pt"


def history_path(method_id: str, chain: str, seed_id: str, fold: int) -> Path:
    return RUN_ARTIFACTS / "training_metrics" / method_id / chain / seed_id / f"fold_{fold}.csv"


def normalise_history(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    if list(frame.columns) != list(HISTORY_COLUMNS):
        raise ValueError(f"{context}: unexpected training-history columns")
    result = frame.copy()
    for column in ("fold", "epoch"):
        try:
            values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{context}: {column} must contain integers") from error
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{context}: {column} must contain finite integers")
        result[column] = values.astype(int)
    for column in HISTORY_METRICS:
        try:
            values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{context}: {column} must contain numeric values") from error
        if not np.isfinite(values).all():
            raise ValueError(f"{context}: {column} contains non-finite values")
        if column.endswith("_auc") and ((values < 0) | (values > 1)).any():
            raise ValueError(f"{context}: {column} must be between 0 and 1")
        if column.endswith("_bce") and (values < 0).any():
            raise ValueError(f"{context}: {column} must be non-negative")
        result[column] = values
    if result.duplicated(list(HISTORY_KEY)).any():
        raise ValueError(f"{context}: duplicate configuration-fold-epoch rows")
    return result


def expected_history_keys() -> set[tuple[str, str, str, int, int]]:
    return {
        (method_id, chain, seed_id, fold, epoch)
        for method_id, _, _, _ in METHODS
        for chain in CHAINS
        for seed_id in SEED_IDS
        for fold in FOLDS
        for epoch in HISTORY_EPOCHS
    }


def validate_history_metadata(frame: pd.DataFrame, context: str) -> None:
    method_labels = {method_id: method_label for method_id, method_label, _, _ in METHODS}
    seed_values = {seed_id: str(seed_metadata(seed_id)[0]) for seed_id in SEED_IDS}
    expected_labels = frame["method_id"].map(method_labels)
    expected_seeds = frame["seed_id"].map(seed_values)
    if expected_labels.isna().any() or not frame["method_label"].astype(str).eq(expected_labels).all():
        raise ValueError(f"{context}: method labels do not match the registered methods")
    if expected_seeds.isna().any() or not frame["seed_value"].astype(str).eq(expected_seeds).all():
        raise ValueError(f"{context}: seed values do not match the registered seeds")


def validate_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = normalise_history(frame, "Experiment 01 aggregate training history")
    actual = set(result[list(HISTORY_KEY)].itertuples(index=False, name=None))
    expected = expected_history_keys()
    if actual != expected:
        raise ValueError(
            "Experiment 01 aggregate training history has an invalid configuration-fold-epoch "
            f"set (missing={len(expected - actual)}, extra={len(actual - expected)})"
        )
    validate_history_metadata(result, "Experiment 01 aggregate training history")
    return result


def validate_fold_history(
    frame: pd.DataFrame,
    method: tuple,
    chain: str,
    seed_id: str,
    fold: int,
    context: str,
) -> pd.DataFrame:
    result = normalise_history(frame, context)
    method_id = method[0]
    expected = {(method_id, chain, seed_id, fold, epoch) for epoch in HISTORY_EPOCHS}
    actual = set(result[list(HISTORY_KEY)].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"{context}: wrong configuration, fold, or exact epoch set")
    validate_history_metadata(result, context)
    return result


def checkpoint_contract(method: tuple, chain: str, seed_id: str, fold: int) -> dict:
    method_id, _, _, embedding_dim = method
    seed = seed_metadata(seed_id)[0]
    return {
        "experiment_id": EXPERIMENT_ID,
        "method_id": method_id,
        "chain": chain,
        "seed_id": seed_id,
        "seed_value": str(seed),
        "fold": fold,
        "epochs": EPOCHS,
        "metric_interval": METRIC_INTERVAL,
        "learning_rate": LEARNING_RATE,
        "accumulation": ACCUMULATION,
        "shuffle_seed": SHUFFLE_SEED,
        "embedding_dim": embedding_dim,
        "normalizer": "sparsemax",
    }


def read_history(
    path: Path,
    method: tuple,
    chain: str,
    seed_id: str,
    fold: int,
    *,
    strict_pair: bool = False,
) -> pd.DataFrame | None:
    checkpoint = checkpoint_path(method[0], chain, seed_id, fold)
    contract = checkpoint_contract(method, chain, seed_id, fold)
    if strict_pair:
        training.validate_checkpoint_history_pair(
            checkpoint, path, contract, repo=protocol.REPO
        )
        frame = pd.read_csv(path, dtype={"seed_value": str})
        return validate_fold_history(frame, method, chain, seed_id, fold, str(path))
    if not training.checkpoint_history_pair_is_valid(
        checkpoint, path, contract, repo=protocol.REPO
    ):
        return None
    try:
        frame = pd.read_csv(path, dtype={"seed_value": str})
        return validate_fold_history(frame, method, chain, seed_id, fold, str(path))
    except (OSError, TypeError, ValueError):
        return None


def build_model(embedding_dim: int, seed: int, device: torch.device) -> AttentionMIL:
    training.set_seed(seed)
    return AttentionMIL("sparsemax", embedding_dim).to(device)


def save_checkpoint(
    path: Path,
    model: SparsemaxAttentionMIL,
    method: tuple,
    chain: str,
    seed_id: str,
    fold: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {"model_state": model.state_dict(), **checkpoint_contract(method, chain, seed_id, fold)},
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    method: tuple,
    chain: str,
    seed_id: str,
    fold: int,
    device: torch.device,
) -> SparsemaxAttentionMIL:
    embedding_dim = method[3]
    seed = seed_metadata(seed_id)[0]
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = checkpoint_contract(method, chain, seed_id, fold)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: {key}={payload.get(key)!r}, expected {value!r}")
    model = build_model(embedding_dim, seed, device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def prediction_rows(
    method: tuple,
    chain: str,
    seed_id: str,
    fold: int,
    rows: list[dict],
    scores: dict[str, float],
) -> list[dict]:
    method_id, method_label, _, _ = method
    seed, tier = seed_metadata(seed_id)
    return [
        {
            "experiment_id": EXPERIMENT_ID,
            "method_id": method_id,
            "method_label": method_label,
            "chain": chain,
            "seed_id": seed_id,
            "seed_value": str(seed),
            "seed_group": tier,
            "fold": fold,
            "split": "internal_oof",
            "subject_id": row["subject_id"],
            "cohort": row["cohort"],
            "label": int(row["label"]),
            "score": scores[row["subject_id"]],
            "tcr_count": int(row["tcr_count"]),
        }
        for row in rows
    ]


def validate_predictions(frame: pd.DataFrame) -> None:
    if set(frame["split"]) != {"internal_oof"}:
        raise ValueError("Experiment 01 results must be internal OOF only")
    for method_id, _, _, _ in METHODS:
        for chain in CHAINS:
            for seed_id in SEED_IDS:
                rows = frame.loc[
                    (frame["method_id"] == method_id)
                    & (frame["chain"] == chain)
                    & (frame["seed_id"] == seed_id)
                ]
                expected = EXPECTED_SUBJECTS[chain]
                if len(rows) != expected or rows["subject_id"].nunique() != expected:
                    raise ValueError(f"Incomplete OOF: {method_id}/{chain}/{seed_id}")


def internal(device: torch.device) -> pd.DataFrame:
    manifest = internal_manifest()
    output: list[dict] = []
    histories: list[pd.DataFrame] = []
    completed = 0
    total = len(METHODS) * len(CHAINS) * len(SEED_IDS)
    for method in METHODS:
        method_id, _, _, embedding_dim = method
        for chain in CHAINS:
            frame = method_rows(manifest, method, chain)
            records = frame.to_dict("records")
            tensors = protocol.load_tensors(frame, device)
            for seed_id in SEED_IDS:
                seed = seed_metadata(seed_id)[0]
                for fold in FOLDS:
                    train_rows = [row for row in records if int(row["fold"]) != fold]
                    test_rows = [row for row in records if int(row["fold"]) == fold]
                    checkpoint = checkpoint_path(method_id, chain, seed_id, fold)
                    metrics_path = history_path(method_id, chain, seed_id, fold)
                    fold_history = read_history(metrics_path, method, chain, seed_id, fold)
                    model = None
                    if checkpoint.exists() and fold_history is not None:
                        try:
                            model = load_checkpoint(checkpoint, method, chain, seed_id, fold, device)
                        except (KeyError, OSError, RuntimeError, ValueError):
                            model = None
                    if model is None:
                        model = build_model(embedding_dim, seed, device)
                        values: list[dict] = []
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
                            history=values,
                        )
                        save_checkpoint(checkpoint, model, method, chain, seed_id, fold)
                        fold_history = pd.DataFrame(values).assign(
                            method_id=method_id,
                            method_label=method[1],
                            chain=chain,
                            seed_id=seed_id,
                            seed_value=str(seed),
                            fold=fold,
                        )
                        fold_history = validate_fold_history(
                            fold_history, method, chain, seed_id, fold, str(metrics_path)
                        )
                        protocol.atomic_csv(fold_history[list(HISTORY_COLUMNS)], metrics_path)
                        training.bind_checkpoint_history(
                            checkpoint,
                            metrics_path,
                            checkpoint_contract(method, chain, seed_id, fold),
                            repo=protocol.REPO,
                        )
                    histories.append(fold_history)
                    scores = training.predict_scores(model, test_rows, tensors)
                    output.extend(prediction_rows(method, chain, seed_id, fold, test_rows, scores))
                completed += 1
                print(
                    f"[training] Experiment 01: {completed}/{total} {method_id}, {chain}, {seed_id}",
                    flush=True,
                )
            del tensors
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result = pd.DataFrame(output, columns=protocol.PREDICTION_COLUMNS)
    validate_predictions(result)
    protocol.atomic_csv(result, RUN_ARTIFACTS / "internal_predictions.csv")
    history = validate_history(pd.concat(histories, ignore_index=True))
    protocol.atomic_csv(history[list(HISTORY_COLUMNS)], RUN_ARTIFACTS / "training_metrics.csv")
    return result


def self_test() -> None:
    for _, _, _, embedding_dim in METHODS:
        training.set_seed(17)
        model = AttentionMIL("sparsemax", embedding_dim)
        values = torch.randn(13, embedding_dim)
        weights = model.attention_weights(values)
        assert tuple(weights.shape) == (13, 1)
        assert torch.all(weights >= 0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(model(values), model(values[torch.randperm(13)]), atol=1e-6)
    print("experiment 01 internal-only self-test passed")
