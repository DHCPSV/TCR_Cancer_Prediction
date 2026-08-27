"""File selection and integrity checks for the optional report cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analysis import protocol


CONTENTS_MANIFEST = protocol.repo_path("artifacts/report_cache_contents.json")
RELEASE_MANIFEST = protocol.repo_path("artifacts/report_cache_release.json")

CHECKPOINT_DIRS = (
    "artifacts/checkpoints/experiment_01_internal_baseline",
    "artifacts/checkpoints/experiment_02_seed_and_attention_normalisation",
    "artifacts/checkpoints/experiment_04_alice_model",
    "artifacts/checkpoints/experiment_04_alice_model_no_hard_gate",
)

RUN_FILES = (
    "artifacts/runs/experiment_01_internal_baseline/internal_predictions.csv",
    "artifacts/runs/experiment_01_internal_baseline/training_metrics.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/factorization_predictions.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/factorization_training_metrics.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/normalizer_predictions.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/normalizer_training_metrics.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/trajectory_fold_metrics.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/trajectory_oof.csv",
    "artifacts/runs/experiment_03_external_generalisation/transfer_predictions.csv",
    "artifacts/runs/experiment_03_external_generalisation/pca/per_seed_patient_coordinates.csv",
    "artifacts/runs/experiment_04_alice_model/internal_predictions.csv",
    "artifacts/runs/experiment_04_alice_model/external_predictions.csv",
    "artifacts/runs/experiment_04_alice_model/training_metrics.csv",
    "artifacts/runs/experiment_04_alice_model/retention_by_subject.csv",
    "artifacts/runs/experiment_04_alice_model/families/patient_local_families.csv",
    "artifacts/runs/experiment_04_alice_model/future_work/internal_predictions.csv",
    "artifacts/runs/experiment_04_alice_model/future_work/external_predictions.csv",
    "artifacts/runs/experiment_04_alice_model/future_work/gate_free_pca/patient_coordinates.csv",
    "artifacts/runs/experiment_04_alice_model/future_work/gate_free_pca/summary.csv",
    "artifacts/runs/experiment_04_alice_model_no_hard_gate/internal_predictions.csv",
    "artifacts/runs/experiment_04_alice_model_no_hard_gate/external_predictions.csv",
    "artifacts/runs/experiment_04_alice_model_no_hard_gate/training_metrics.csv",
)

HISTORY_DIRS = (
    "artifacts/runs/experiment_01_internal_baseline/training_metrics",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/factorization_training_metrics",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/normalizer_training_metrics",
    "artifacts/runs/experiment_04_alice_model/training_metrics",
    "artifacts/runs/experiment_04_alice_model_no_hard_gate/training_metrics",
)

MANIFEST_FILES = (
    "artifacts/manifests/experiment_03_external_generalisation_models.csv",
    "artifacts/manifests/experiment_03_external_generalisation_freeze.json",
    "artifacts/manifests/alice_runtime_freeze.json",
)


def selected_files(repo: Path = protocol.REPO) -> list[Path]:
    """Return the exact files included in the report cache."""
    files = [repo / name for name in RUN_FILES + MANIFEST_FILES]
    for name in CHECKPOINT_DIRS + HISTORY_DIRS:
        root = repo / name
        if not root.is_dir():
            raise FileNotFoundError(f"Report-cache directory is missing: {root}")
        files.extend(path for path in root.rglob("*") if path.is_file())
    missing = [path for path in files if not path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Report-cache inputs are missing:\n{preview}")
    unique = {path.resolve(): path for path in files}
    return sorted(unique.values(), key=lambda path: path.relative_to(repo).as_posix())


def file_record(path: Path, repo: Path = protocol.REPO) -> dict:
    return {
        "path": path.relative_to(repo).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": protocol.sha256(path),
    }


def write_contents_manifest(
    files: list[Path],
    target: Path = CONTENTS_MANIFEST,
    repo: Path = protocol.REPO,
) -> dict:
    payload = {
        "schema_version": 1,
        "files": [file_record(path, repo) for path in files],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(target)
    return payload


def verify(repo: Path = protocol.REPO, *, check_checkpoints: bool = True) -> dict:
    """Verify every extracted cache file and optionally read each checkpoint."""
    manifest = repo / CONTENTS_MANIFEST.relative_to(protocol.REPO)
    if not manifest.is_file():
        raise FileNotFoundError(
            "The report cache is not installed. Extract the published ZIP into the "
            "repository root, then run this command again."
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), list):
        raise ValueError("Unsupported report-cache manifest")
    records = payload["files"]
    if len({record.get("path") for record in records}) != len(records):
        raise ValueError("The report-cache manifest contains duplicate paths")
    expected = {
        path.relative_to(repo).as_posix()
        for path in selected_files(repo)
    }
    recorded = {record.get("path") for record in records}
    if recorded != expected:
        missing = sorted(expected - recorded)
        extra = sorted(recorded - expected)
        details = []
        if missing:
            details.append(f"missing entries: {missing[:5]}")
        if extra:
            details.append(f"unexpected entries: {extra[:5]}")
        raise ValueError("Report-cache contents do not match the allowlist; " + "; ".join(details))
    for record in records:
        path = repo / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Report-cache file is missing: {record['path']}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Report-cache size mismatch: {record['path']}")
        if protocol.sha256(path) != record["sha256"]:
            raise ValueError(f"Report-cache checksum mismatch: {record['path']}")
    if check_checkpoints:
        import torch

        for record in records:
            if not record["path"].endswith(".pt"):
                continue
            payload = torch.load(repo / record["path"], map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
                raise ValueError(f"Invalid checkpoint payload: {record['path']}")
    return {
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "contents_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
