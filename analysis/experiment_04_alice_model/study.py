from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch

from analysis import protocol, training
from analysis.experiment_04_alice_model import learning_curves, methods
from analysis.experiment_04_alice_model.methods import alice_pool
from pipeline import provenance, workflow


class PreparationOptions(Protocol):
    rscript: str
    olga: str
    workers: int


@dataclass(frozen=True)
class _Study:
    experiment_id: str
    cache_consumer: str
    methods: tuple[methods.Method, ...]
    checkpoints: Path
    run_artifacts: Path
    output: Path
    progress_label: str
    record_history: bool = True
    freeze_experiment_id: str = methods.EXPERIMENT_ID
    verify_main_checkpoints: bool = True


_MAIN = _Study(
    experiment_id=methods.EXPERIMENT_ID,
    cache_consumer=methods.EXPERIMENT_ID,
    methods=methods.MAIN_METHODS,
    checkpoints=methods.CHECKPOINTS,
    run_artifacts=methods.RUN_ARTIFACTS,
    output=methods.RESULTS,
    progress_label="Experiment 04 hard-gate models",
)

_NO_HARD_GATE = _Study(
    experiment_id=methods.NO_HARD_GATE_EXPERIMENT_ID,
    cache_consumer=methods.NO_HARD_GATE_EXPERIMENT_ID,
    methods=methods.NO_HARD_GATE_METHODS,
    checkpoints=methods.NO_HARD_GATE_CHECKPOINTS,
    run_artifacts=methods.NO_HARD_GATE_RUN_ARTIFACTS,
    output=methods.NO_HARD_GATE_RESULTS,
    progress_label="Experiment 04 models without the hard gate",
)

_REPRESENTATION = _Study(
    experiment_id=methods.EXPERIMENT_ID,
    cache_consumer=f"{methods.EXPERIMENT_ID}.future_work",
    methods=methods.REPRESENTATION_METHODS,
    checkpoints=methods.CHECKPOINTS,
    run_artifacts=methods.RUN_ARTIFACTS / "future_work",
    output=methods.RESULTS / "future_work",
    progress_label="Experiment 04 preliminary representation dependence",
    record_history=False,
)


def alice_input_states(split: str) -> tuple[Path, ...]:
    return (
        workflow.STATE_ROOT / "data" / f"{split}.json",
        methods.ALICE / f"{split}_manifest.csv",
        methods.ALICE / f"{split}_file_manifest.csv",
    )


def prepare_alice_inputs(split: str, args: PreparationOptions) -> tuple[Path, ...]:
    from analysis.experiment_04_alice_model import prepare as alice_preparation

    if split == "internal":
        workflow.ensure_internal()
    elif split == "external":
        workflow.ensure_external()
    else:
        raise ValueError(f"Unknown data split: {split}")
    alice_preparation.prepare(split, args)
    frozen_splits = ("internal",) if split == "internal" else ("internal", "external")
    alice_preparation.write_runtime_freeze(
        frozen_splits,
        require_core_checkpoints=(split == "external"),
        require_future_checkpoints=False,
    )
    return alice_input_states(split)


def prepare_future_work_inputs(split: str, args: PreparationOptions) -> tuple[Path, ...]:
    from analysis.experiment_01_internal_baseline import prepare as representation_preparation
    from analysis.experiment_04_alice_model import prepare as alice_preparation

    if split == "internal":
        upstream = workflow.ensure_internal()
    elif split == "external":
        upstream = workflow.ensure_external()
    else:
        raise ValueError(f"Unknown data split: {split}")
    representation_preparation.ensure(split, upstream)
    alice_preparation.prepare(split, args)
    frozen_splits = ("internal",) if split == "internal" else ("internal", "external")
    alice_preparation.write_runtime_freeze(
        frozen_splits,
        include_descriptors=True,
        require_core_checkpoints=True,
        require_future_checkpoints=(split == "external"),
    )
    return alice_input_states(split) + (
        protocol.MANIFESTS / f"representation_comparison_{split}.csv",
    )


def _guard_internal_cache(spec: _Study, states: tuple[Path, ...]) -> None:
    targets = tuple(
        spec.checkpoints / method.method_id for method in spec.methods
    ) + (spec.run_artifacts,)
    workflow.guard_consumer(spec.cache_consumer, targets, states)


def dataset(split: str, chain: str, representation: str = "sceptr") -> pd.DataFrame:
    subjects = protocol.read_manifest(f"{split}_subjects")
    manifest = (
        f"{split}_representations"
        if representation == "sceptr"
        else f"representation_comparison_{split}"
    )
    representations = protocol.read_manifest(manifest)
    representations = representations.loc[
        (representations["chain"] == chain)
        & (representations["representation_method"] == representation)
    ].copy()
    alice = pd.read_csv(
        methods.ALICE / f"{split}_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )
    alice = alice.loc[(alice["split"] == split) & (alice["chain"] == chain)].copy()
    frame = (
        subjects[["subject_id", "cohort", "label"]]
        .merge(
            representations[
                [
                    "subject_id",
                    "source_patient_id",
                    "embedding_file",
                    "tcr_count",
                    "embedding_dim",
                    "sha256",
                ]
            ],
            on="subject_id",
            validate="one_to_one",
        )
        .merge(
            alice[["subject_id", "evidence_file", "alice_burden"]],
            on="subject_id",
            validate="one_to_one",
        )
    )
    frame["label"] = frame["label"].astype(int)
    frame["tcr_count"] = frame["tcr_count"].astype(int)
    frame["embedding_dim"] = frame["embedding_dim"].astype(int)
    if split == "internal":
        folds = protocol.read_fixed_folds(chain, frame["subject_id"])
        frame["fold"] = frame["subject_id"].map(folds).astype(int)
    return frame.sort_values("subject_id").reset_index(drop=True)


def load_features(frame: pd.DataFrame) -> dict[str, dict]:
    features: dict[str, dict] = {}
    for row in frame.itertuples(index=False):
        embedding_path = protocol.repo_path(row.embedding_file)
        if row.sha256 and protocol.sha256(embedding_path) != row.sha256:
            raise ValueError(f"{row.subject_id}: representation checksum mismatch")
        embedding = torch.load(
            embedding_path,
            map_location="cpu",
            weights_only=True,
        ).float()
        evidence = np.load(protocol.repo_path(row.evidence_file), allow_pickle=False)
        expected = (int(row.tcr_count), int(row.embedding_dim))
        if tuple(embedding.shape) != expected or len(evidence) != len(embedding):
            raise ValueError(f"{row.subject_id}: embedding/evidence mismatch")
        features[row.subject_id] = {
            "embedding": embedding,
            "evidence": evidence.astype(np.float32, copy=False),
            "burden": float(row.alice_burden),
        }
    return features


def patient_vectors(
    frame: pd.DataFrame,
    features: dict[str, dict],
    method: methods.Method,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        subject_id: alice_pool(
            feature["embedding"].to(device),
            feature["evidence"],
            norm=method.norm,
            threshold=method.threshold,
        )
        for subject_id, feature in features.items()
    }


def checkpoint(spec: _Study, method: methods.Method, chain: str, seed_id: str, fold: int) -> Path:
    return methods.checkpoint_path(spec.checkpoints, method, chain, seed_id, fold)


def checkpoint_metadata(
    spec: _Study,
    method: methods.Method,
    chain: str,
    seed_id: str,
    fold: int,
) -> dict:
    metadata = {
        "experiment_id": spec.experiment_id,
        "method_id": method.method_id,
        "chain": chain,
        "seed_id": seed_id,
        "seed_value": str(protocol.SEED_REGISTRY[seed_id]["value"]),
        "fold": fold,
        "epochs": methods.EPOCHS,
    }
    if spec is not _NO_HARD_GATE:
        metadata.update(
            representation=method.representation,
            dimensions=method.dimensions,
        )
    if spec is _MAIN:
        metadata.update(
            metric_interval=learning_curves.METRIC_INTERVAL,
            learning_rate=methods.LEARNING_RATE,
            accumulation=methods.ACCUMULATION,
            shuffle_seed=protocol.FOLD_SEED,
            hard_threshold=float(method.threshold),
            pooling_normalisation=method.norm,
            classifier_head=method.head,
            classifier_hidden_dimensions=32 if method.head == "mlp" else 0,
            classifier_dropout=0.2 if method.head == "mlp" else 0.0,
        )
    elif spec is _NO_HARD_GATE:
        metadata.update(
            metric_interval=learning_curves.METRIC_INTERVAL,
            learning_rate=methods.LEARNING_RATE,
            accumulation=methods.ACCUMULATION,
            shuffle_seed=protocol.FOLD_SEED,
            pooling_normalisation=method.norm,
            classifier_head=method.head,
            evidence_mapping="unchanged_raw_alice_evidence",
            hard_threshold="none",
            training_loop=f"per_patient_gradient_accumulation_{methods.ACCUMULATION}",
        )
    return metadata


def bound_checkpoint_contract(
    spec: _Study,
    method: methods.Method,
    chain: str,
    seed_id: str,
    fold: int,
) -> dict:
    path = checkpoint(spec, method, chain, seed_id, fold)
    if not path.exists():
        return checkpoint_metadata(spec, method, chain, seed_id, fold)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {key: value for key, value in payload.items() if key != "model_state"}


def load_checkpoint(
    spec: _Study,
    method: methods.Method,
    chain: str,
    seed_id: str,
    fold: int,
    device: torch.device,
) -> torch.nn.Module:
    path = checkpoint(spec, method, chain, seed_id, fold)
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = checkpoint_metadata(spec, method, chain, seed_id, fold)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Checkpoint metadata mismatch: {path}")
    training.set_seed(int(protocol.SEED_REGISTRY[seed_id]["value"]))
    model = method.build().to(device)
    model.load_state_dict(payload["model_state"])
    return model


def save_checkpoint(
    spec: _Study,
    model: torch.nn.Module,
    method: methods.Method,
    chain: str,
    seed_id: str,
    fold: int,
) -> None:
    path = checkpoint(spec, method, chain, seed_id, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        checkpoint_metadata(spec, method, chain, seed_id, fold)
        | {"model_state": model.state_dict()},
        temporary,
    )
    temporary.replace(path)


def training_history_identity(
    spec: _Study,
    method: methods.Method,
    chain: str,
    seed_id: str,
    fold: int,
) -> dict:
    return {
        "experiment_id": spec.experiment_id,
        "method_id": method.method_id,
        "method_label": method.label,
        "chain": chain,
        "seed_id": seed_id,
        "seed_value": str(protocol.SEED_REGISTRY[seed_id]["value"]),
        "fold": fold,
    }


def load_or_fit(
    spec: _Study,
    method: methods.Method,
    chain: str,
    seed_id: str,
    fold: int,
    train: pd.DataFrame,
    held_out: pd.DataFrame,
    vectors: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    path = checkpoint(spec, method, chain, seed_id, fold)
    identity = training_history_identity(spec, method, chain, seed_id, fold)
    history_path = learning_curves.fold_history_path(
        spec.run_artifacts,
        method.method_id,
        chain,
        seed_id,
        fold,
    )
    contract = bound_checkpoint_contract(spec, method, chain, seed_id, fold)
    reusable = path.exists() and (
        not spec.record_history
        or learning_curves.history_is_complete(
            history_path,
            methods.EPOCHS,
            identity,
            path,
            contract,
        )
    )
    if reusable:
        return load_checkpoint(spec, method, chain, seed_id, fold, device)

    training.set_seed(int(protocol.SEED_REGISTRY[seed_id]["value"]))
    model = method.build().to(device)
    history: list[dict] = []
    training.train_binary_model(
        model.classifier,
        train.to_dict("records"),
        vectors,
        epochs=methods.EPOCHS,
        learning_rate=methods.LEARNING_RATE,
        accumulation=methods.ACCUMULATION,
        shuffle_seed=protocol.FOLD_SEED,
        held_out_rows=held_out.to_dict("records") if spec.record_history else None,
        metric_interval=learning_curves.METRIC_INTERVAL,
        history=history if spec.record_history else None,
    )
    save_checkpoint(spec, model, method, chain, seed_id, fold)
    if spec.record_history:
        learning_curves.write_fold_history(
            history_path,
            history,
            identity,
            path,
            checkpoint_metadata(spec, method, chain, seed_id, fold),
        )
    return model


def scores(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    vectors: dict[str, torch.Tensor],
) -> dict[str, float]:
    return training.predict_scores(
        model.classifier,
        frame.to_dict("records"),
        vectors,
    )


def prediction_row(
    spec: _Study,
    method: methods.Method,
    chain: str,
    seed_id: str,
    fold: int,
    split: str,
    row,
    score: float,
) -> dict:
    seed = protocol.SEED_REGISTRY[seed_id]
    return {
        "experiment_id": spec.experiment_id,
        "method_id": method.method_id,
        "method_label": method.label,
        "chain": chain,
        "seed_id": seed_id,
        "seed_value": str(seed["value"]),
        "seed_group": seed["seed_group"],
        "fold": fold,
        "split": split,
        "subject_id": row.subject_id,
        "cohort": row.cohort,
        "label": int(row.label),
        "score": score,
        "tcr_count": int(row.tcr_count),
    }


def learning_history_jobs(spec: _Study) -> list[dict]:
    return [
        training_history_identity(spec, method, chain, seed_id, fold)
        | {
            "checkpoint_path": str(checkpoint(spec, method, chain, seed_id, fold)),
            "checkpoint_contract": bound_checkpoint_contract(
                spec,
                method,
                chain,
                seed_id,
                fold,
            ),
        }
        for method in spec.methods
        for chain in method.chains
        for seed_id in method.seed_ids
        for fold in range(5)
    ]


def aggregate_learning_history(spec: _Study) -> pd.DataFrame:
    history = learning_curves.collect_fold_histories(
        spec.run_artifacts,
        learning_history_jobs(spec),
        methods.EPOCHS,
    )
    aggregate = spec.run_artifacts / "training_metrics.csv"
    if not aggregate.exists():
        protocol.atomic_csv(history, aggregate)
        return history
    recorded = pd.read_csv(aggregate, dtype={"seed_value": str})
    order = ["method_id", "chain", "seed_id", "fold", "epoch"]
    pd.testing.assert_frame_equal(
        recorded.sort_values(order).reset_index(drop=True),
        history.sort_values(order).reset_index(drop=True),
        check_dtype=False,
        obj=f"{spec.experiment_id} aggregate and per-fold histories",
    )
    return recorded


def verify_future_work_checkpoint_freeze() -> None:
    freeze = json.loads(protocol.ALICE_FREEZE.read_text(encoding="utf-8"))
    entry = freeze.get("future_work", {}).get("representation_dependence")
    if not isinstance(entry, dict):
        raise ValueError("Missing representation-dependence freeze contract")
    expected_ids = [method.method_id for method in methods.REPRESENTATION_METHODS]
    if entry.get("checkpoint_method_ids") != expected_ids:
        raise ValueError("Future-work checkpoint method registry differs from the freeze")
    checkpoints = [
        path
        for method_id in expected_ids
        for path in (methods.CHECKPOINTS / method_id).rglob("*.pt")
    ]
    expected_count = sum(
        len(method.chains) * len(method.seed_ids) * 5
        for method in methods.REPRESENTATION_METHODS
    )
    if len(checkpoints) != expected_count or int(entry.get("checkpoint_count", -1)) != expected_count:
        raise ValueError("Future-work checkpoint coverage differs from the 120-model contract")
    if protocol.file_set_digest(checkpoints) != entry.get("checkpoint_digest"):
        raise ValueError("Future-work checkpoint content differs from the freeze")


def _verify_inputs(spec: _Study, split: str, *, internal_stage: bool) -> None:
    extra = (
        (f"representation_comparison_{split}",)
        if any(method.representation != "sceptr" for method in spec.methods)
        else ()
    )
    protocol.verify_alice_freeze(
        split,
        spec.freeze_experiment_id,
        extra,
        verify_checkpoints=(
            spec.verify_main_checkpoints
            and not (spec is _MAIN and internal_stage)
        ),
    )
    if spec is _REPRESENTATION and not internal_stage:
        verify_future_work_checkpoint_freeze()


def _run_internal(spec: _Study, device: torch.device) -> pd.DataFrame:
    _verify_inputs(spec, "internal", internal_stage=True)
    result: list[dict] = []
    cache: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, dict]]] = {}
    vector_cache: dict[tuple[str, str, str, float | None], dict[str, torch.Tensor]] = {}
    total_groups = sum(len(method.chains) for method in spec.methods)
    total_folds = sum(
        len(method.chains) * len(method.seed_ids) * 5 for method in spec.methods
    )
    progress = provenance.Progress(spec.progress_label, total_groups, updates=6)
    reused_folds = 0
    for method in spec.methods:
        for chain in method.chains:
            data_key = (chain, method.representation)
            if data_key not in cache:
                frame = dataset("internal", chain, method.representation)
                cache[data_key] = frame, load_features(frame)
            frame, features = cache[data_key]
            vector_key = (
                chain,
                method.representation,
                method.norm,
                method.threshold,
            )
            if vector_key not in vector_cache:
                vector_cache[vector_key] = patient_vectors(
                    frame,
                    features,
                    method,
                    device,
                )
            vectors = vector_cache[vector_key]
            for seed_id in method.seed_ids:
                for fold in range(5):
                    train = frame.loc[frame["fold"] != fold]
                    held_out = frame.loc[frame["fold"] == fold]
                    identity = training_history_identity(
                        spec,
                        method,
                        chain,
                        seed_id,
                        fold,
                    )
                    history_path = learning_curves.fold_history_path(
                        spec.run_artifacts,
                        method.method_id,
                        chain,
                        seed_id,
                        fold,
                    )
                    path = checkpoint(spec, method, chain, seed_id, fold)
                    contract = bound_checkpoint_contract(
                        spec,
                        method,
                        chain,
                        seed_id,
                        fold,
                    )
                    reused_folds += int(
                        path.exists()
                        and (
                            not spec.record_history
                            or learning_curves.history_is_complete(
                                history_path,
                                methods.EPOCHS,
                                identity,
                                path,
                                contract,
                            )
                        )
                    )
                    model = load_or_fit(
                        spec,
                        method,
                        chain,
                        seed_id,
                        fold,
                        train,
                        held_out,
                        vectors,
                        device,
                    )
                    fold_scores = scores(model, held_out, vectors)
                    result.extend(
                        prediction_row(
                            spec,
                            method,
                            chain,
                            seed_id,
                            fold,
                            "internal_oof",
                            row,
                            fold_scores[row.subject_id],
                        )
                        for row in held_out.itertuples(index=False)
                    )
            progress.advance()
    predictions = pd.DataFrame(result, columns=protocol.PREDICTION_COLUMNS)
    protocol.atomic_csv(predictions, spec.run_artifacts / "internal_predictions.csv")
    if spec.record_history:
        aggregate_learning_history(spec)
    if spec is _MAIN:
        from analysis.experiment_04_alice_model.prepare import write_runtime_freeze

        write_runtime_freeze(
            ("internal",),
            require_core_checkpoints=True,
            require_future_checkpoints=False,
        )
    elif spec is _NO_HARD_GATE:
        from analysis.experiment_04_alice_model.prepare import write_runtime_freeze

        write_runtime_freeze(
            ("internal",),
            require_core_checkpoints=True,
            require_future_checkpoints=False,
            require_no_hard_gate_checkpoints=True,
        )
    elif spec is _REPRESENTATION:
        from analysis.experiment_04_alice_model.prepare import write_runtime_freeze

        write_runtime_freeze(
            ("internal",),
            include_descriptors=True,
            require_core_checkpoints=True,
            require_future_checkpoints=True,
        )
    progress.summary(
        reused_folds=reused_folds,
        trained_folds=total_folds - reused_folds,
    )
    return predictions


def _run_validation(spec: _Study, device: torch.device) -> pd.DataFrame:
    _verify_inputs(spec, "internal", internal_stage=False)
    _verify_inputs(spec, "external", internal_stage=False)
    missing = [
        checkpoint(spec, method, chain, seed_id, fold)
        for method in spec.methods
        for chain in method.chains
        for seed_id in method.seed_ids
        for fold in range(5)
        if not checkpoint(spec, method, chain, seed_id, fold).exists()
    ]
    if missing:
        raise RuntimeError(
            f"Locked external evaluation requires all checkpoints; missing {len(missing)}"
        )
    result: list[dict] = []
    cache: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, dict]]] = {}
    vector_cache: dict[tuple[str, str, str, float | None], dict[str, torch.Tensor]] = {}
    total_groups = sum(len(method.chains) for method in spec.methods)
    progress = provenance.Progress(
        f"{spec.progress_label} locked external evaluation",
        total_groups,
        updates=6,
    )
    for method in spec.methods:
        for chain in method.chains:
            data_key = (chain, method.representation)
            if data_key not in cache:
                frame = dataset("external", chain, method.representation)
                cache[data_key] = frame, load_features(frame)
            frame, features = cache[data_key]
            vector_key = (
                chain,
                method.representation,
                method.norm,
                method.threshold,
            )
            if vector_key not in vector_cache:
                vector_cache[vector_key] = patient_vectors(
                    frame,
                    features,
                    method,
                    device,
                )
            vectors = vector_cache[vector_key]
            for seed_id in method.seed_ids:
                per_fold = [
                    scores(
                        load_checkpoint(
                            spec,
                            method,
                            chain,
                            seed_id,
                            fold,
                            device,
                        ),
                        frame,
                        vectors,
                    )
                    for fold in range(5)
                ]
                for row in frame.itertuples(index=False):
                    mean_score = float(
                        np.mean([part[row.subject_id] for part in per_fold])
                    )
                    result.append(
                        prediction_row(
                            spec,
                            method,
                            chain,
                            seed_id,
                            -1,
                            "locked_external",
                            row,
                            mean_score,
                        )
                    )
            progress.advance()
    predictions = pd.DataFrame(result, columns=protocol.PREDICTION_COLUMNS)
    protocol.atomic_csv(predictions, spec.run_artifacts / "external_predictions.csv")
    progress.summary(
        evaluated_seed_configurations=sum(
            len(method.chains) * len(method.seed_ids) for method in spec.methods
        )
    )
    return predictions


def guard_main_cache(states: tuple[Path, ...]) -> None:
    _guard_internal_cache(_MAIN, states)


def guard_no_hard_gate_cache(states: tuple[Path, ...]) -> None:
    _guard_internal_cache(_NO_HARD_GATE, states)


def guard_representation_cache(states: tuple[Path, ...]) -> None:
    _guard_internal_cache(_REPRESENTATION, states)


def run_hard_gate(device: torch.device) -> pd.DataFrame:
    return _run_internal(_MAIN, device)


def validate_hard_gate(device: torch.device) -> pd.DataFrame:
    return _run_validation(_MAIN, device)


def run_no_hard_gate(device: torch.device) -> pd.DataFrame:
    return _run_internal(_NO_HARD_GATE, device)


def validate_no_hard_gate(device: torch.device) -> pd.DataFrame:
    return _run_validation(_NO_HARD_GATE, device)


def run_representation_comparison(device: torch.device) -> pd.DataFrame:
    return _run_internal(_REPRESENTATION, device)


def validate_representation_comparison(device: torch.device) -> pd.DataFrame:
    return _run_validation(_REPRESENTATION, device)


def main_checkpoint_metadata(
    method: methods.Method,
    chain: str,
    seed_id: str,
    fold: int,
) -> dict:
    return checkpoint_metadata(_MAIN, method, chain, seed_id, fold)


def main_learning_history() -> pd.DataFrame:
    return aggregate_learning_history(_MAIN)


def no_hard_gate_learning_history() -> pd.DataFrame:
    return aggregate_learning_history(_NO_HARD_GATE)


def self_test() -> None:
    if len(methods.REPRESENTATION_METHODS) != 8:
        raise AssertionError("Expected eight preliminary representation methods")
    if any(method.chains != ("alpha",) for method in methods.REPRESENTATION_METHODS):
        raise AssertionError("Representation dependence is an Alpha-only partial study")
