"""Stable paths, schemas, folds, manifests, and artifact contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "artifacts" / "manifests"
ALICE_FREEZE = MANIFESTS / "alice_runtime_freeze.json"
_VERIFIED_ALICE_ASSETS: set[str] = set()
_VERIFIED_CHECKPOINTS: set[str] = set()
FOLD_SEED = 913271

SEED_REGISTRY = {
    "S01": {"value": 777, "seed_group": "lucky"},
    "S02": {"value": 2232016200, "seed_group": "lucky"},
    "S03": {"value": 12906138414127427065, "seed_group": "lucky"},
    "S04": {"value": 6902826956892357559, "seed_group": "moderate"},
    "S05": {"value": 2712386035810402826, "seed_group": "moderate"},
    "S06": {"value": 14320595710489364965, "seed_group": "moderate"},
    "S07": {"value": 11327508419378867602, "seed_group": "ordinary"},
    "S08": {"value": 10762549616907469203, "seed_group": "ordinary"},
    "S09": {"value": 12293968614973973933, "seed_group": "ordinary"},
    "S10": {"value": 8365301900047924020, "seed_group": "lucky"},
    "S11": {"value": 2817756917657900866, "seed_group": "moderate"},
    "S12": {"value": 10694063091131643398, "seed_group": "ordinary"},
    "S13": {"value": 4496119368746308026, "seed_group": "ordinary"},
}

PREDICTION_COLUMNS = [
    "experiment_id",
    "method_id",
    "method_label",
    "chain",
    "seed_id",
    "seed_value",
    "seed_group",
    "fold",
    "split",
    "subject_id",
    "cohort",
    "label",
    "score",
    "tcr_count",
]


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_set_digest(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    resolved = sorted((repo_path(path) for path in paths), key=lambda path: path.as_posix())
    for path in resolved:
        relative = path.relative_to(REPO).as_posix()
        digest.update(f"{relative}\t{path.stat().st_size}\t{sha256(path)}\n".encode())
    return digest.hexdigest()


def verify_alice_freeze(
    split: str,
    experiment_id: str,
    extra_manifests: Sequence[str] = (),
    *,
    verify_checkpoints: bool = True,
) -> None:
    if split not in {"internal", "external"}:
        raise ValueError(f"Unknown ALICE split: {split}")
    if not ALICE_FREEZE.exists():
        raise FileNotFoundError(ALICE_FREEZE)
    freeze = json.loads(ALICE_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != 2:
        raise ValueError("Unsupported ALICE freeze schema")
    experiment = freeze.get("experiments", {}).get(experiment_id)
    if experiment is None:
        raise ValueError(f"{experiment_id} is not bound to the frozen ALICE inputs")
    for relative, expected_hash in experiment.get("model_code_files", {}).items():
        if sha256(REPO / relative) != expected_hash:
            raise ValueError(f"Frozen ALICE formula/code mismatch: {relative}")
    if verify_checkpoints and experiment_id not in _VERIFIED_CHECKPOINTS:
        checkpoint_root = REPO / "artifacts" / "checkpoints" / experiment_id
        method_ids = experiment.get("checkpoint_method_ids")
        if method_ids is None:
            checkpoints = list(checkpoint_root.rglob("*.pt"))
        else:
            if not isinstance(method_ids, list) or not method_ids or len(set(method_ids)) != len(method_ids):
                raise ValueError(f"Invalid checkpoint_method_ids: {experiment_id}")
            checkpoints = []
            for method_id in method_ids:
                method_root = checkpoint_root / str(method_id)
                if not method_root.is_dir():
                    raise FileNotFoundError(method_root)
                checkpoints.extend(method_root.rglob("*.pt"))
        if len(checkpoints) != int(experiment["checkpoint_count"]):
            raise ValueError(f"Frozen checkpoint count mismatch: {experiment_id}")
        if file_set_digest(checkpoints) != experiment["checkpoint_digest"]:
            raise ValueError(f"Frozen checkpoint content mismatch: {experiment_id}")
        _VERIFIED_CHECKPOINTS.add(experiment_id)
    required = [
        "artifacts/manifests/fivefold_seed913271.csv",
        f"artifacts/manifests/{split}_subjects.csv",
        f"artifacts/manifests/{split}_representations.csv",
        f"artifacts/alice/{split}_manifest.csv",
        f"artifacts/alice/{split}_file_manifest.csv",
        *[f"artifacts/manifests/{name}.csv" for name in extra_manifests],
    ]
    expected = freeze.get("files", {})
    for relative in required:
        path = REPO / relative
        if expected.get(relative) != sha256(path):
            raise ValueError(f"Frozen ALICE provenance mismatch: {relative}")
    if split not in _VERIFIED_ALICE_ASSETS:
        assets = pd.read_csv(
            REPO / "artifacts" / "alice" / f"{split}_file_manifest.csv",
            dtype={"path": str, "bytes": int, "sha256": str},
        )
        if assets["path"].duplicated().any() or assets.empty:
            raise ValueError(f"Invalid {split} ALICE file manifest")
        for row in assets.itertuples(index=False):
            path = repo_path(row.path)
            if path.stat().st_size != int(row.bytes) or sha256(path) != row.sha256:
                raise ValueError(f"Frozen ALICE asset mismatch: {row.path}")
        _VERIFIED_ALICE_ASSETS.add(split)


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output = repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    temporary.replace(output)


def read_manifest(name: str) -> pd.DataFrame:
    filename = name if name.endswith(".csv") else f"{name}.csv"
    path = MANIFESTS / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_fixed_folds(chain: str, expected_subjects: Iterable[str]) -> dict[str, int]:
    frame = read_manifest("fivefold_seed913271")
    frame = frame.loc[frame["chain"] == chain].copy()
    frame["fold"] = frame["fold"].astype(int)
    if frame["subject_id"].duplicated().any() or set(frame["fold"]) != set(range(5)):
        raise ValueError(f"Invalid fixed folds for {chain}")
    mapping = frame.set_index("subject_id")["fold"].to_dict()
    if set(mapping) != set(expected_subjects):
        raise ValueError(f"Fold membership differs from the {chain} representation manifest")
    return mapping


def load_tensors(rows: pd.DataFrame, device: torch.device) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for row in rows.itertuples(index=False):
        path = repo_path(row.embedding_file)
        if getattr(row, "sha256", "") and sha256(path) != row.sha256:
            raise ValueError(f"{row.subject_id}: representation checksum mismatch")
        tensor = torch.load(path, map_location="cpu", weights_only=True).float()
        expected = (int(row.tcr_count), int(row.embedding_dim))
        if tuple(tensor.shape) != expected or not torch.isfinite(tensor).all():
            raise ValueError(
                f"{row.subject_id}: expected finite tensor {expected}, got {tuple(tensor.shape)}"
            )
        tensors[row.subject_id] = tensor.to(device)
    return tensors
