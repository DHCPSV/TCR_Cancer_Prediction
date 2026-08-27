from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis import protocol, training
from analysis.experiment_02_seed_and_attention_normalisation.methods import AttentionMIL
from pipeline import provenance


EXPERIMENT_ID = "experiment_03_external_generalisation"
CHAINS = ("alpha", "beta")
SEED_IDS = tuple(f"S{index:02d}" for index in range(1, 10))
N_FOLDS = 5
METHODS = (
    ("sceptr_softmax", "Softmax", "softmax"),
    ("sceptr_entmax15", "Entmax-1.5", "entmax15"),
    ("sceptr_sparsemax", "Sparsemax", "sparsemax"),
)
TRANSFER_METHODS = tuple(method[0] for method in METHODS)
METHOD_LABELS = {
    method_id: f"SCEPTR + {label} Attention MIL"
    for method_id, label, _ in METHODS
}
EPOCHS = 50

RUNS = protocol.REPO / "artifacts" / "runs" / EXPERIMENT_ID
FROZEN_PREDICTIONS = RUNS / "frozen_inputs" / "predictions"
GEOMETRY = RUNS / "geometry"
CHECKPOINTS = protocol.REPO / "artifacts" / "checkpoints" / EXPERIMENT_ID
PREDECESSORS = (
    CHECKPOINTS
    / "frozen_predecessors"
    / "experiment_02_seed_and_attention_normalisation"
)
RESULTS = protocol.REPO / "results" / EXPERIMENT_ID
MANIFESTS = protocol.REPO / "artifacts" / "manifests"
INPUT_MANIFEST = MANIFESTS / f"{EXPERIMENT_ID}_inputs.csv"
MODEL_MANIFEST = MANIFESTS / f"{EXPERIMENT_ID}_models.csv"
FREEZE = MANIFESTS / f"{EXPERIMENT_ID}_freeze.json"


def relative(path: Path) -> str:
    return path.relative_to(protocol.REPO).as_posix()


def seed_metadata(seed_id: str) -> tuple[int, str]:
    row = protocol.SEED_REGISTRY[seed_id]
    return int(row["value"]), str(row["seed_group"])


def representation_rows(role: str, chain: str) -> pd.DataFrame:
    frame = protocol.read_manifest(f"{role}_representations")
    frame = frame.loc[
        (frame["role"] == role)
        & (frame["chain"] == chain)
        & (frame["representation_method"] == "sceptr")
    ].copy()
    expected = 169 if role == "internal" else (109 if chain == "alpha" else 110)
    if len(frame) != expected or frame["subject_id"].nunique() != expected:
        raise ValueError(f"Incomplete {role} {chain} representations")
    return frame.reset_index(drop=True)


def predecessor_checkpoint(
    method_id: str, chain: str, seed_id: str, fold: int
) -> Path:
    return (
        protocol.REPO
        / "artifacts/checkpoints/experiment_02_seed_and_attention_normalisation/normalizers"
        / method_id
        / chain
        / seed_id
        / f"fold_{fold}.pt"
    )


def frozen_checkpoint(method_id: str, chain: str, seed_id: str, fold: int) -> Path:
    return PREDECESSORS / method_id / chain / seed_id / f"fold_{fold}.pt"


def freeze_predecessor_models() -> list[dict]:
    records: list[dict] = []
    for method_id, _, _ in METHODS:
        for chain in CHAINS:
            for seed_id in SEED_IDS:
                for fold in range(N_FOLDS):
                    source = predecessor_checkpoint(method_id, chain, seed_id, fold)
                    if not source.exists():
                        raise FileNotFoundError(f"Run Experiment 02 first: {source}")
                    target = frozen_checkpoint(method_id, chain, seed_id, fold)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if (
                        not target.exists()
                        or protocol.sha256(target) != protocol.sha256(source)
                    ):
                        shutil.copy2(source, target)
                    records.append(
                        {
                            "method_id": method_id,
                            "chain": chain,
                            "seed_id": seed_id,
                            "fold": fold,
                            "frozen_path": relative(target),
                            "bytes": target.stat().st_size,
                            "sha256": protocol.sha256(target),
                            "source_experiment": (
                                "experiment_02_seed_and_attention_normalisation"
                            ),
                            "source_path": relative(source),
                        }
                    )
    return records


def load_attention_model(
    method: tuple[str, str, str],
    chain: str,
    seed_id: str,
    fold: int,
    device: torch.device,
) -> AttentionMIL:
    method_id, _, normalizer = method
    path = frozen_checkpoint(method_id, chain, seed_id, fold)
    if not path.exists():
        raise FileNotFoundError(f"Prepare Experiment 03 frozen models first: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = {
        "experiment_id": "experiment_02_seed_and_attention_normalisation",
        "method_id": method_id,
        "chain": chain,
        "seed_id": seed_id,
        "fold": fold,
        "epochs": EPOCHS,
        "normalizer": normalizer,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Checkpoint metadata mismatch: {path}")
    model = AttentionMIL(normalizer).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval()


def internal_predictions() -> pd.DataFrame:
    source = (
        protocol.REPO
        / "artifacts/runs/experiment_02_seed_and_attention_normalisation/normalizer_predictions.csv"
    )
    frame = pd.read_csv(source, dtype={"seed_value": str})
    frame = frame.loc[frame["method_id"].isin(TRANSFER_METHODS)].copy()
    expected_rows = len(METHODS) * len(CHAINS) * len(SEED_IDS) * 169
    if len(frame) != expected_rows:
        raise ValueError("Experiment 02 internal OOF predictions are incomplete")
    frame["experiment_id"] = EXPERIMENT_ID
    result = frame.loc[:, protocol.PREDICTION_COLUMNS]
    protocol.atomic_csv(
        result,
        FROZEN_PREDICTIONS / "baseline_internal_predictions.csv",
    )
    return result


def external_predictions(device: torch.device) -> pd.DataFrame:
    output: list[dict] = []
    progress = provenance.Progress(
        "Experiment 03 locked external transfer",
        len(METHODS) * len(CHAINS),
        updates=len(METHODS) * len(CHAINS),
    )
    for chain in CHAINS:
        frame = representation_rows("external", chain)
        rows = frame.to_dict("records")
        tensors = protocol.load_tensors(frame, device)
        for method in METHODS:
            method_id, _, _ = method
            for seed_id in SEED_IDS:
                seed_value, tier = seed_metadata(seed_id)
                fold_scores = {row["subject_id"]: [] for row in rows}
                for fold in range(N_FOLDS):
                    model = load_attention_model(
                        method, chain, seed_id, fold, device
                    )
                    scores = training.predict_scores(model, rows, tensors)
                    for subject_id, score in scores.items():
                        fold_scores[subject_id].append(score)
                for row in rows:
                    scores = fold_scores[row["subject_id"]]
                    if len(scores) != N_FOLDS:
                        raise ValueError(
                            "Locked external score is not a five-fold mean: "
                            f"{method_id}/{chain}/{seed_id}/{row['subject_id']}"
                        )
                    output.append(
                        {
                            "experiment_id": EXPERIMENT_ID,
                            "method_id": method_id,
                            "method_label": METHOD_LABELS[method_id],
                            "chain": chain,
                            "seed_id": seed_id,
                            "seed_value": str(seed_value),
                            "seed_group": tier,
                            "fold": -1,
                            "split": "locked_external",
                            "subject_id": row["subject_id"],
                            "cohort": row["cohort"],
                            "label": int(row["label"]),
                            "score": float(np.mean(scores)),
                            "tcr_count": int(row["tcr_count"]),
                        }
                    )
            progress.advance()
        del tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = pd.DataFrame(output, columns=protocol.PREDICTION_COLUMNS)
    protocol.atomic_csv(
        result,
        FROZEN_PREDICTIONS / "baseline_external_predictions.csv",
    )
    progress.summary(
        evaluated_seed_configurations=(
            len(METHODS) * len(CHAINS) * len(SEED_IDS)
        )
    )
    return result


def prepare_transfer(
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    model_records = freeze_predecessor_models()
    internal = internal_predictions()
    external = external_predictions(device)
    return internal, external, model_records


def input_record(
    artifact_id: str,
    path: Path,
    source_experiment: str,
    source_path: Path,
) -> dict:
    return {
        "artifact_id": artifact_id,
        "frozen_path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": protocol.sha256(path),
        "source_experiment": source_experiment,
        "source_path": relative(source_path),
    }


def write_manifests(model_records: list[dict]) -> None:
    source_internal = (
        protocol.REPO
        / "artifacts/runs/experiment_02_seed_and_attention_normalisation/normalizer_predictions.csv"
    )
    records = [
        input_record(
            "baseline_internal_predictions",
            FROZEN_PREDICTIONS / "baseline_internal_predictions.csv",
            "experiment_02_seed_and_attention_normalisation",
            source_internal,
        ),
        input_record(
            "baseline_external_predictions",
            FROZEN_PREDICTIONS / "baseline_external_predictions.csv",
            EXPERIMENT_ID,
            FROZEN_PREDICTIONS / "baseline_external_predictions.csv",
        ),
    ]
    protocol.atomic_csv(pd.DataFrame(records), INPUT_MANIFEST)
    protocol.atomic_csv(pd.DataFrame(model_records), MODEL_MANIFEST)


def verify_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = pd.read_csv(INPUT_MANIFEST, dtype=str)
    models = pd.read_csv(MODEL_MANIFEST, dtype=str)
    required_inputs = {
        "artifact_id",
        "frozen_path",
        "bytes",
        "sha256",
        "source_experiment",
        "source_path",
    }
    required_models = {
        "method_id",
        "chain",
        "seed_id",
        "fold",
        "frozen_path",
        "bytes",
        "sha256",
        "source_experiment",
        "source_path",
    }
    if missing := required_inputs - set(inputs):
        raise ValueError(f"Frozen-input manifest lacks {sorted(missing)}")
    if missing := required_models - set(models):
        raise ValueError(f"Frozen-model manifest lacks {sorted(missing)}")
    expected_artifacts = {
        "baseline_internal_predictions",
        "baseline_external_predictions",
    }
    if set(inputs["artifact_id"]) != expected_artifacts:
        raise ValueError(
            "Frozen-input manifest differs from the transfer/PCA contract: "
            f"{sorted(set(inputs['artifact_id']))}"
        )
    if inputs["artifact_id"].duplicated().any() or models["frozen_path"].duplicated().any():
        raise ValueError("Frozen predecessor manifests contain duplicates")
    for row in pd.concat(
        [
            inputs[["frozen_path", "bytes", "sha256"]],
            models[["frozen_path", "bytes", "sha256"]],
        ],
        ignore_index=True,
    ).itertuples(index=False):
        path = protocol.repo_path(row.frozen_path)
        if path.stat().st_size != int(row.bytes) or protocol.sha256(path) != row.sha256:
            raise ValueError(f"Frozen predecessor changed: {row.frozen_path}")

    expected_count = len(CHAINS) * len(SEED_IDS) * N_FOLDS
    actual_counts = models.groupby("method_id").size().to_dict()
    required_counts = {method_id: expected_count for method_id in TRANSFER_METHODS}
    if actual_counts != required_counts:
        raise ValueError(f"Frozen model coverage differs from contract: {actual_counts}")
    for (method_id, chain, seed_id), part in models.groupby(
        ["method_id", "chain", "seed_id"]
    ):
        if set(part["fold"].astype(int)) != set(range(N_FOLDS)):
            raise ValueError(
                f"Incomplete frozen folds: {method_id}/{chain}/{seed_id}"
            )
    for row in models.itertuples(index=False):
        payload = torch.load(
            protocol.repo_path(row.frozen_path),
            map_location="cpu",
            weights_only=True,
        )
        if "model_state" not in payload:
            raise ValueError(f"Missing model state: {row.frozen_path}")
        if str(payload.get("method_id")) != row.method_id:
            raise ValueError(f"Method metadata mismatch: {row.frozen_path}")
    return inputs, models


def frozen_artifact(inputs: pd.DataFrame, artifact_id: str) -> Path:
    row = inputs.loc[inputs["artifact_id"] == artifact_id]
    if len(row) != 1:
        raise ValueError(f"Expected one frozen artifact named {artifact_id}")
    return protocol.repo_path(row.iloc[0]["frozen_path"])


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"seed_value": str})
    frame = frame.loc[frame["method_id"].isin(TRANSFER_METHODS)].copy()
    frame["experiment_id"] = EXPERIMENT_ID
    frame["method_label"] = frame["method_id"].map(METHOD_LABELS)
    return frame.loc[:, protocol.PREDICTION_COLUMNS]


def assemble() -> pd.DataFrame:
    inputs, _ = verify_frozen_inputs()
    internal = _read_predictions(
        frozen_artifact(inputs, "baseline_internal_predictions")
    )
    external = _read_predictions(
        frozen_artifact(inputs, "baseline_external_predictions")
    )
    transfer = pd.concat([internal, external], ignore_index=True)
    if set(transfer["split"]) != {"internal_oof", "locked_external"}:
        raise ValueError(f"Unexpected transfer splits: {set(transfer['split'])}")
    for key, part in transfer.groupby(
        ["method_id", "chain", "seed_id", "split"]
    ):
        expected = 169 if key[-1] == "internal_oof" else (
            109 if key[1] == "alpha" else 110
        )
        if len(part) != expected or part["subject_id"].nunique() != expected:
            raise ValueError(f"Incomplete frozen predictions: {key}")
    protocol.atomic_csv(transfer, RUNS / "transfer_predictions.csv")
    return transfer


def freeze() -> dict:
    verify_frozen_inputs()
    experiment = protocol.REPO / "analysis" / EXPERIMENT_ID
    files = [
        protocol.REPO / "analysis" / "protocol.py",
        protocol.REPO / "analysis" / "training.py",
        protocol.REPO / "analysis" / "reporting.py",
        experiment / "run.py",
        experiment / "study.py",
        experiment / "geometry.py",
        experiment / "report.py",
        protocol.REPO / "pipeline" / "prepare_data.py",
        protocol.REPO / "pipeline" / "embed_sceptr.py",
        protocol.REPO / "pipeline" / "sceptr_fast.py",
        protocol.REPO / "pipeline" / "build_manifests.py",
        INPUT_MANIFEST,
        MODEL_MANIFEST,
        RUNS / "transfer_predictions.csv",
        GEOMETRY / "per_seed_patient_coordinates.csv",
        RESULTS / "metrics_by_seed.csv",
        RESULTS / "metrics_summary.csv",
        RESULTS / "per_seed_pca_summary.csv",
        RESULTS / "pca_transfer_example_manifest.csv",
        RESULTS / "figures" / "frozen_internal_external_auc.png",
        RESULTS / "figures" / "seed_resolved_pca_transfer_examples.png",
    ]
    files.extend(
        RESULTS / "supplementary" / "per_seed_pc1_atlas" / f"{chain}_page_{page}.png"
        for chain in CHAINS
        for page in range(1, 4)
    )
    payload = {
        "schema_version": 2,
        "experiment_id": EXPERIMENT_ID,
        "evaluation": "locked_external_transfer",
        "external_selection": False,
        "checkpoint_count": len(TRANSFER_METHODS) * len(CHAINS) * len(SEED_IDS) * N_FOLDS,
        "fold_count": N_FOLDS,
        "fold_aggregation": "mean",
        "pca_fit_split": "internal_oof",
        "external_projection": "transform_only",
        "files": {
            path.relative_to(protocol.REPO).as_posix(): protocol.sha256(path)
            for path in files
        },
    }
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def self_test() -> None:
    assert len(TRANSFER_METHODS) == 3
    assert len(CHAINS) * len(SEED_IDS) * N_FOLDS == 90
    assert len(TRANSFER_METHODS) * len(CHAINS) * len(SEED_IDS) * N_FOLDS == 270
    assert set(METHOD_LABELS) == set(TRANSFER_METHODS)
    print("experiment_03_external_generalisation study self-test passed")
