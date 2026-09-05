"""Selection and integrity checks for the optional reproduction cache."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

from analysis import protocol


CONTENTS_MANIFEST = protocol.repo_path("artifacts/reproduction_cache_contents.json")
RELEASE_MANIFEST = protocol.repo_path("artifacts/reproduction_cache_release.json")

CHECKPOINT_DIRS = (
    "artifacts/checkpoints/experiment_01_internal_baseline",
    "artifacts/checkpoints/experiment_02_seed_and_attention_normalisation",
    "artifacts/checkpoints/experiment_04_alice_model",
    "artifacts/checkpoints/experiment_04_alice_model_no_hard_gate",
)

HISTORY_DIRS = (
    "artifacts/runs/experiment_01_internal_baseline/training_metrics",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
    "factorization_training_metrics",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
    "normalizer_training_metrics",
    "artifacts/runs/experiment_04_alice_model/training_metrics",
    "artifacts/runs/experiment_04_alice_model_no_hard_gate/training_metrics",
)

REUSABLE_INPUT_DIRS = (
    "artifacts/representations",
    "artifacts/tcr_tables",
    "artifacts/provenance",
    "artifacts/alice/evidence",
    "artifacts/alice/inputs",
    "artifacts/alice/outputs",
    "artifacts/alice/sequence_maps",
)

RUN_FILES = (
    "artifacts/runs/experiment_01_internal_baseline/internal_predictions.csv",
    "artifacts/runs/experiment_01_internal_baseline/training_metrics.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
    "factorization_predictions.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
    "factorization_training_metrics.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
    "normalizer_predictions.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
    "normalizer_training_metrics.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
    "trajectory_fold_metrics.csv",
    "artifacts/runs/experiment_02_seed_and_attention_normalisation/trajectory_oof.csv",
    "artifacts/runs/experiment_03_external_generalisation/transfer_predictions.csv",
    "artifacts/runs/experiment_03_external_generalisation/pca/"
    "per_seed_patient_coordinates.csv",
    "artifacts/runs/experiment_04_alice_model/internal_predictions.csv",
    "artifacts/runs/experiment_04_alice_model/external_predictions.csv",
    "artifacts/runs/experiment_04_alice_model/training_metrics.csv",
    "artifacts/runs/experiment_04_alice_model/retention_by_subject.csv",
    "artifacts/runs/experiment_04_alice_model/families/patient_local_families.csv",
    "artifacts/runs/experiment_04_alice_model/future_work/internal_predictions.csv",
    "artifacts/runs/experiment_04_alice_model/future_work/external_predictions.csv",
    "artifacts/runs/experiment_04_alice_model/future_work/gate_free_pca/"
    "patient_coordinates.csv",
    "artifacts/runs/experiment_04_alice_model/future_work/gate_free_pca/summary.csv",
    "artifacts/runs/experiment_04_alice_model_no_hard_gate/internal_predictions.csv",
    "artifacts/runs/experiment_04_alice_model_no_hard_gate/external_predictions.csv",
    "artifacts/runs/experiment_04_alice_model_no_hard_gate/training_metrics.csv",
)

MANIFEST_FILES = (
    "artifacts/manifests/alice_runtime_freeze.json",
    "artifacts/manifests/experiment_02_development_seed_screen.csv",
    "artifacts/manifests/experiment_03_external_generalisation_freeze.json",
    "artifacts/manifests/experiment_03_external_generalisation_models.csv",
    "artifacts/manifests/external_representations.csv",
    "artifacts/manifests/external_samples.csv",
    "artifacts/manifests/external_subjects.csv",
    "artifacts/manifests/fivefold_seed913271.csv",
    "artifacts/manifests/internal_representations.csv",
    "artifacts/manifests/internal_samples.csv",
    "artifacts/manifests/internal_subjects.csv",
    "artifacts/manifests/representation_comparison_external.csv",
    "artifacts/manifests/representation_comparison_internal.csv",
    "artifacts/manifests/representations.csv",
    "artifacts/manifests/samples.csv",
    "artifacts/manifests/subjects.csv",
    "artifacts/alice/external_file_manifest.csv",
    "artifacts/alice/external_manifest.csv",
    "artifacts/alice/external_samples.csv",
    "artifacts/alice/internal_file_manifest.csv",
    "artifacts/alice/internal_manifest.csv",
    "artifacts/alice/internal_samples.csv",
)

PACKAGE_DIRS = CHECKPOINT_DIRS + HISTORY_DIRS + REUSABLE_INPUT_DIRS


def raw_file_hashes(repo: Path = protocol.REPO) -> dict[str, str]:
    """Select the source repertoires named in the completed sample manifest."""
    raw_root = (repo / "data/raw").resolve()
    hashes: dict[str, str] = {}
    with (repo / "artifacts/manifests/samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            name, expected = row["raw_file"], row["raw_sha256"]
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or not name.startswith("data/raw/")
                or not name.endswith((".tsv", ".tsv.gz"))
                or not (repo / name).resolve().is_relative_to(raw_root)
            ):
                raise ValueError(f"Invalid raw repertoire path: {name}")
            if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise ValueError(f"Missing or invalid raw checksum: {name}")
            if name in hashes and hashes[name] != expected:
                raise ValueError(f"Conflicting raw checksums: {name}")
            hashes[name] = expected
    if not hashes:
        raise ValueError("The sample manifest contains no raw repertoires")
    return hashes


def selected_files(repo: Path = protocol.REPO) -> list[Path]:
    """Return the exact files included in the reproduction cache."""
    files = [repo / name for name in RUN_FILES + MANIFEST_FILES]
    files.extend(repo / name for name in raw_file_hashes(repo))
    for name in PACKAGE_DIRS:
        root = repo / name
        if not root.is_dir():
            raise FileNotFoundError(f"Reproduction-cache directory is missing: {root}")
        contents = [path for path in root.rglob("*") if path.is_file()]
        if not contents:
            raise FileNotFoundError(f"Reproduction-cache directory is empty: {root}")
        files.extend(contents)
    missing = [path for path in files if not path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Reproduction-cache inputs are missing:\n{preview}")
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
        "schema_version": 2,
        "package_type": "reproduction_cache",
        "files": [file_record(path, repo) for path in files],
    }
    recorded_hashes = {item["path"]: item["sha256"] for item in payload["files"]}
    for name, expected in raw_file_hashes(repo).items():
        if recorded_hashes.get(name) != expected:
            raise ValueError(f"Raw repertoire does not match the sample manifest: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(target)
    return payload


def _manifest(repo: Path) -> tuple[Path, dict, list[dict]]:
    manifest = repo / CONTENTS_MANIFEST.relative_to(protocol.REPO)
    if not manifest.is_file():
        raise FileNotFoundError(
            "The reproduction cache is not installed. Extract the published ZIP "
            "into the repository root, then run this command again."
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 2
        or payload.get("package_type") != "reproduction_cache"
        or not isinstance(payload.get("files"), list)
    ):
        raise ValueError("Unsupported reproduction-cache manifest")
    records = payload["files"]
    if len({record.get("path") for record in records}) != len(records):
        raise ValueError("The reproduction-cache manifest contains duplicate paths")
    return manifest, payload, records


def verify(
    repo: Path = protocol.REPO,
    *,
    check_hashes: bool = True,
    check_checkpoints: bool = True,
) -> dict:
    """Verify the extracted cache against its allowlist and contents manifest."""
    manifest, _, records = _manifest(repo)
    raw_hashes = raw_file_hashes(repo)
    expected = {path.relative_to(repo).as_posix() for path in selected_files(repo)}
    recorded = {record.get("path") for record in records}
    if recorded != expected:
        missing = sorted(expected - recorded)
        extra = sorted(recorded - expected)
        details = []
        if missing:
            details.append(f"missing entries: {missing[:5]}")
        if extra:
            details.append(f"unexpected entries: {extra[:5]}")
        raise ValueError(
            "Reproduction-cache contents do not match the allowlist; "
            + "; ".join(details)
        )
    for record in records:
        if record["path"] in raw_hashes and record["sha256"] != raw_hashes[record["path"]]:
            raise ValueError(f"Raw checksum differs from sample manifest: {record['path']}")
        path = repo / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Reproduction-cache file is missing: {record['path']}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Reproduction-cache size mismatch: {record['path']}")
        if check_hashes and protocol.sha256(path) != record["sha256"]:
            raise ValueError(f"Reproduction-cache checksum mismatch: {record['path']}")
    if check_checkpoints:
        import torch

        for root_name in CHECKPOINT_DIRS:
            for path in (repo / root_name).rglob("*.pt"):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict) or not isinstance(
                    checkpoint.get("model_state"), dict
                ):
                    relative = path.relative_to(repo).as_posix()
                    raise ValueError(f"Invalid checkpoint payload: {relative}")
    return {
        "file_count": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "contents_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def verify_layout(repo: Path = protocol.REPO) -> dict:
    """Quickly check paths and sizes after a prior full cache verification."""
    return verify(repo, check_hashes=False, check_checkpoints=False)
