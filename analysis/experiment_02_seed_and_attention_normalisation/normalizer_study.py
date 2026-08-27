from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis import protocol, training
from analysis.experiment_02_seed_and_attention_normalisation.methods import AttentionMIL
from pipeline import provenance


EXPERIMENT_ID = "experiment_02_seed_and_attention_normalisation"
CHAINS = ("alpha", "beta")
SEED_IDS = tuple(f"S{index:02d}" for index in range(1, 10))
FOLDS = range(5)
EPOCHS = 50
METRIC_INTERVAL = 5
LEARNING_RATE = 1e-3
ACCUMULATION = 4
SHUFFLE_SEED = 913271

METHODS = (
    ("sceptr_softmax", "Softmax", "softmax"),
    ("sceptr_entmax15", "Entmax-1.5", "entmax15"),
    ("sceptr_sparsemax", "Sparsemax", "sparsemax"),
)

RUN_ARTIFACTS = protocol.repo_path(f"artifacts/runs/{EXPERIMENT_ID}")
CHECKPOINTS = protocol.repo_path(f"artifacts/checkpoints/{EXPERIMENT_ID}/normalizers")
EXPECTED_SUBJECTS = {"alpha": 169, "beta": 169}
HISTORY_COLUMNS = (
    "experiment_id",
    "method_id",
    "method_label",
    "chain",
    "seed_id",
    "seed_value",
    "seed_group",
    "fold",
    "epoch",
    "training_auc",
    "held_out_auc",
    "training_bce",
    "held_out_bce",
)
HISTORY_METRICS = ("training_auc", "held_out_auc", "training_bce", "held_out_bce")
HISTORY_KEY = ("method_id", "chain", "seed_id", "fold", "epoch")
HISTORY_EPOCHS = (1, *range(METRIC_INTERVAL, EPOCHS + 1, METRIC_INTERVAL))


def seed_metadata(seed_id: str) -> tuple[int, str]:
    row = protocol.SEED_REGISTRY[seed_id]
    return int(row["value"]), str(row["seed_group"])


def internal_rows(chain: str) -> pd.DataFrame:
    frame = protocol.read_manifest("internal_representations")
    frame = frame.loc[
        (frame["role"] == "internal")
        & (frame["chain"] == chain)
        & (frame["representation_method"] == "sceptr")
    ].copy()
    if frame["subject_id"].nunique() != EXPECTED_SUBJECTS[chain]:
        raise ValueError(f"{chain}: wrong internal subject count")
    if set(frame["embedding_dim"].astype(int)) != {64}:
        raise ValueError(f"{chain}: expected 64D SCEPTR representations")
    folds = protocol.read_fixed_folds(chain, frame["subject_id"])
    frame["fold"] = frame["subject_id"].map(folds)
    # Keep the frozen representation-manifest order.  The patient shuffle is
    # deterministic only relative to this starting order, so re-sorting here
    # changes the complete optimisation path even when every seed is fixed.
    return frame.reset_index(drop=True)


def checkpoint_path(method_id: str, chain: str, seed_id: str, fold: int) -> Path:
    return CHECKPOINTS / method_id / chain / seed_id / f"fold_{fold}.pt"


def history_path(method_id: str, chain: str, seed_id: str, fold: int) -> Path:
    return (
        RUN_ARTIFACTS
        / "normalizer_training_metrics"
        / method_id
        / chain
        / seed_id
        / f"fold_{fold}.csv"
    )


def build_model(normalizer: str, seed: int, device: torch.device) -> AttentionMIL:
    training.set_seed(seed)
    return AttentionMIL(normalizer).to(device)


def checkpoint_contract(method: tuple, chain: str, seed_id: str, fold: int) -> dict:
    method_id, _, normalizer = method
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
        "embedding_dim": 64,
        "normalizer": normalizer,
    }


def save_checkpoint(path: Path, model: AttentionMIL, method: tuple, chain: str, seed_id: str, fold: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            **checkpoint_contract(method, chain, seed_id, fold),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: Path, method: tuple, chain: str, seed_id: str, fold: int, device: torch.device) -> AttentionMIL:
    normalizer = method[2]
    seed = seed_metadata(seed_id)[0]
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = checkpoint_contract(method, chain, seed_id, fold)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: {key}={payload.get(key)!r}, expected {value!r}")
    model = build_model(normalizer, seed, device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def history_metadata(method: tuple, chain: str, seed_id: str, fold: int) -> dict:
    method_id, method_label, _ = method
    seed, group = seed_metadata(seed_id)
    return {
        "experiment_id": EXPERIMENT_ID,
        "method_id": method_id,
        "method_label": f"SCEPTR + {method_label} Attention MIL",
        "chain": chain,
        "seed_id": seed_id,
        "seed_value": str(seed),
        "seed_group": group,
        "fold": fold,
    }


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


def validate_history_metadata(frame: pd.DataFrame, context: str) -> None:
    method_labels = {
        method_id: f"SCEPTR + {method_label} Attention MIL"
        for method_id, method_label, _ in METHODS
    }
    seed_values = {seed_id: str(seed_metadata(seed_id)[0]) for seed_id in SEED_IDS}
    seed_groups = {seed_id: seed_metadata(seed_id)[1] for seed_id in SEED_IDS}
    expected_labels = frame["method_id"].map(method_labels)
    expected_seeds = frame["seed_id"].map(seed_values)
    expected_groups = frame["seed_id"].map(seed_groups)
    if not frame["experiment_id"].astype(str).eq(EXPERIMENT_ID).all():
        raise ValueError(f"{context}: experiment_id does not match Experiment 02")
    if expected_labels.isna().any() or not frame["method_label"].astype(str).eq(expected_labels).all():
        raise ValueError(f"{context}: method labels do not match the registered methods")
    if expected_seeds.isna().any() or not frame["seed_value"].astype(str).eq(expected_seeds).all():
        raise ValueError(f"{context}: seed values do not match the registered seeds")
    if expected_groups.isna().any() or not frame["seed_group"].astype(str).eq(expected_groups).all():
        raise ValueError(f"{context}: seed groups do not match the seed registry")


def expected_history_keys() -> set[tuple[str, str, str, int, int]]:
    return {
        (method_id, chain, seed_id, fold, epoch)
        for method_id, _, _ in METHODS
        for chain in CHAINS
        for seed_id in SEED_IDS
        for fold in FOLDS
        for epoch in HISTORY_EPOCHS
    }


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


def read_history(
    path: Path,
    method: tuple,
    chain: str,
    seed_id: str,
    fold: int,
    *,
    strict_pair: bool = False,
) -> pd.DataFrame:
    checkpoint = checkpoint_path(method[0], chain, seed_id, fold)
    contract = checkpoint_contract(method, chain, seed_id, fold)
    if strict_pair:
        training.validate_checkpoint_history_pair(
            checkpoint, path, contract, repo=protocol.REPO
        )
    elif not training.checkpoint_history_pair_is_valid(
        checkpoint, path, contract, repo=protocol.REPO
    ):
        raise ValueError(f"Stale or incomplete checkpoint/history pair: {path}")
    frame = pd.read_csv(path, dtype={"seed_value": str})
    return validate_fold_history(frame, method, chain, seed_id, fold, str(path))


def make_history(
    rows: list[dict],
    method: tuple,
    chain: str,
    seed_id: str,
    fold: int,
) -> pd.DataFrame:
    metadata = history_metadata(method, chain, seed_id, fold)
    return pd.DataFrame([metadata | row for row in rows], columns=HISTORY_COLUMNS)


def validate_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = normalise_history(frame, "Experiment 02 normalizer aggregate training history")
    actual = set(result[list(HISTORY_KEY)].itertuples(index=False, name=None))
    expected = expected_history_keys()
    if actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise ValueError(
            "Experiment 02 normalizer aggregate training history has an invalid "
            f"configuration-fold-epoch set (missing={missing}, extra={extra})"
        )
    validate_history_metadata(result, "Experiment 02 normalizer aggregate training history")
    return result


def prediction_rows(method: tuple, chain: str, seed_id: str, fold: int, rows: list[dict], scores: dict[str, float]) -> list[dict]:
    method_id, method_label, _ = method
    seed, group = seed_metadata(seed_id)
    return [
        {
            "experiment_id": EXPERIMENT_ID,
            "method_id": method_id,
            "method_label": f"SCEPTR + {method_label} Attention MIL",
            "chain": chain,
            "seed_id": seed_id,
            "seed_value": str(seed),
            "seed_group": group,
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
        raise ValueError("Experiment 02 accepts internal OOF predictions only")
    for method_id, _, _ in METHODS:
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
    output: list[dict] = []
    histories: list[pd.DataFrame] = []
    total = len(METHODS) * len(CHAINS)
    progress = provenance.Progress(
        "Experiment 02 attention normalisers",
        total,
        updates=total,
    )
    reused_folds = 0
    for method in METHODS:
        method_id, _, normalizer = method
        for chain in CHAINS:
            frame = internal_rows(chain)
            records = frame.to_dict("records")
            tensors = protocol.load_tensors(frame, device)
            for seed_id in SEED_IDS:
                seed = seed_metadata(seed_id)[0]
                reused = 0
                for fold in FOLDS:
                    train_rows = [row for row in records if int(row["fold"]) != fold]
                    test_rows = [row for row in records if int(row["fold"]) == fold]
                    path = checkpoint_path(method_id, chain, seed_id, fold)
                    metrics_path = history_path(method_id, chain, seed_id, fold)
                    cached = False
                    if training.checkpoint_history_pair_is_valid(
                        path,
                        metrics_path,
                        checkpoint_contract(method, chain, seed_id, fold),
                        repo=protocol.REPO,
                    ):
                        try:
                            model = load_checkpoint(path, method, chain, seed_id, fold, device)
                            history = read_history(metrics_path, method, chain, seed_id, fold)
                            cached = True
                            reused += 1
                        except (KeyError, OSError, RuntimeError, ValueError):
                            cached = False
                    if not cached:
                        model = build_model(normalizer, seed, device)
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
                        save_checkpoint(path, model, method, chain, seed_id, fold)
                        history = make_history(history_rows, method, chain, seed_id, fold)
                        history = validate_fold_history(
                            history,
                            method,
                            chain,
                            seed_id,
                            fold,
                            str(metrics_path),
                        )
                        protocol.atomic_csv(history, metrics_path)
                        training.bind_checkpoint_history(
                            path,
                            metrics_path,
                            checkpoint_contract(method, chain, seed_id, fold),
                            repo=protocol.REPO,
                        )
                    histories.append(history)
                    scores = training.predict_scores(model, test_rows, tensors)
                    output.extend(prediction_rows(method, chain, seed_id, fold, test_rows, scores))
                reused_folds += reused
            progress.advance()
            del tensors
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result = pd.DataFrame(output, columns=protocol.PREDICTION_COLUMNS)
    validate_predictions(result)
    protocol.atomic_csv(result, RUN_ARTIFACTS / "normalizer_predictions.csv")
    history = validate_history(pd.concat(histories, ignore_index=True))
    protocol.atomic_csv(history[list(HISTORY_COLUMNS)], RUN_ARTIFACTS / "normalizer_training_metrics.csv")
    total_folds = total * len(SEED_IDS) * len(FOLDS)
    progress.summary(
        reused_folds=reused_folds,
        trained_folds=total_folds - reused_folds,
    )
    return result


def self_test() -> None:
    for _, _, normalizer in METHODS:
        training.set_seed(29)
        model = AttentionMIL(normalizer)
        values = torch.randn(17, 64)
        weights = model.attention_weights(values)
        assert tuple(weights.shape) == (17, 1)
        assert torch.all(weights >= 0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(model(values), model(values[torch.randperm(17)]), atol=1e-6)
    if torch.cuda.is_available():
        long_values = torch.randn(50_441, 64, device="cuda")
        for normalizer in ("entmax15", "sparsemax"):
            model = AttentionMIL(normalizer).cuda()
            weights = model.attention_weights(long_values)
            assert torch.isfinite(weights).all()
            assert torch.allclose(weights.sum(), torch.tensor(1.0, device="cuda"), atol=2e-5)
    print("experiment 02 attention-normalisation self-test passed")
