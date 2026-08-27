"""Shared training, prediction, and checkpoint/history compatibility logic."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _resolve(path: str | Path, repo: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_pair_sidecar(history_path: str | Path, *, repo: Path) -> Path:
    history = _resolve(history_path, repo)
    return history.with_suffix(history.suffix + ".checkpoint.json")


def training_contract_fingerprint(contract: dict) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_checkpoint_history(
    checkpoint_path: str | Path,
    history_path: str | Path,
    contract: dict,
    *,
    repo: Path,
) -> Path:
    checkpoint = _resolve(checkpoint_path, repo)
    history = _resolve(history_path, repo)
    if not checkpoint.is_file() or not history.is_file():
        raise FileNotFoundError("Checkpoint and training history must both exist before binding")
    sidecar = training_pair_sidecar(history, repo=repo)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "checkpoint_path": checkpoint.relative_to(repo).as_posix(),
        "history_path": history.relative_to(repo).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "history_sha256": _sha256(history),
        "contract_sha256": training_contract_fingerprint(contract),
    }
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(sidecar)
    return sidecar


def validate_checkpoint_history_pair(
    checkpoint_path: str | Path,
    history_path: str | Path,
    contract: dict,
    *,
    repo: Path,
) -> None:
    checkpoint = _resolve(checkpoint_path, repo)
    history = _resolve(history_path, repo)
    sidecar = training_pair_sidecar(history, repo=repo)
    if not checkpoint.is_file() or not history.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint/history pair: {history}")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint/history sidecar: {sidecar}") from error
    expected = {
        "schema_version": 1,
        "checkpoint_path": checkpoint.relative_to(repo).as_posix(),
        "history_path": history.relative_to(repo).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "history_sha256": _sha256(history),
        "contract_sha256": training_contract_fingerprint(contract),
    }
    if payload != expected:
        raise ValueError(f"Stale or mismatched checkpoint/history pair: {history}")


def checkpoint_history_pair_is_valid(
    checkpoint_path: str | Path,
    history_path: str | Path,
    contract: dict,
    *,
    repo: Path,
) -> bool:
    try:
        validate_checkpoint_history_pair(
            checkpoint_path,
            history_path,
            contract,
            repo=repo,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    return True


def device_from(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: Sequence[int]) -> dict[int, float]:
    values = np.asarray(labels, dtype=int)
    counts = np.bincount(values, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required, got {counts.tolist()}")
    return {label: len(values) / (2 * int(counts[label])) for label in (0, 1)}


def training_metrics(
    model: torch.nn.Module,
    rows: Sequence[dict],
    tensors: dict[str, torch.Tensor],
    weights: dict[int, float],
) -> dict[str, float]:
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    labels = []
    scores = []
    losses = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            label = int(row["label"])
            value = tensors[row["subject_id"]]
            target = torch.tensor([[float(label)]], device=value.device)
            logit = model(value)
            losses.append((criterion(logit, target).mean() * weights[label]).detach())
            labels.append(label)
            scores.append(float(torch.sigmoid(logit).reshape(()).cpu()))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "bce": float(torch.stack(losses).mean().cpu()),
    }


def train_binary_model(
    model: torch.nn.Module,
    rows: list[dict],
    tensors: dict[str, torch.Tensor],
    *,
    epochs: int,
    learning_rate: float,
    accumulation: int,
    shuffle_seed: int,
    held_out_rows: Sequence[dict] | None = None,
    metric_interval: int = 5,
    history: list[dict] | None = None,
) -> torch.nn.Module:
    """Train with summed patient losses before each optimiser update.

    ``accumulation`` controls how many class-weighted patient losses contribute
    to one update.  The losses are intentionally summed, not averaged, to
    preserve the training procedure used by the released checkpoints.
    """
    weights = class_weights([int(row["label"]) for row in rows])
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    shuffler = random.Random(shuffle_seed)
    for epoch in range(1, epochs + 1):
        order = rows.copy()
        shuffler.shuffle(order)
        model.train()
        optimiser.zero_grad(set_to_none=True)
        for index, row in enumerate(order, 1):
            value = tensors[row["subject_id"]]
            target = torch.tensor([[float(row["label"])]], device=value.device)
            loss = criterion(model(value), target).mean() * weights[int(row["label"])]
            loss.backward()  # summed with the other patient losses in this update
            if index % accumulation == 0 or index == len(order):
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
        if history is not None and held_out_rows is not None and (
            epoch == 1 or epoch % metric_interval == 0 or epoch == epochs
        ):
            training = training_metrics(model, rows, tensors, weights)
            held_out_weights = class_weights([int(row["label"]) for row in held_out_rows])
            held_out = training_metrics(model, held_out_rows, tensors, held_out_weights)
            history.append(
                {
                    "epoch": epoch,
                    "training_auc": training["auc"],
                    "held_out_auc": held_out["auc"],
                    "training_bce": training["bce"],
                    "held_out_bce": held_out["bce"],
                }
            )
    return model


def predict_scores(
    model: torch.nn.Module,
    rows: Sequence[dict],
    tensors: dict[str, torch.Tensor],
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        return {
            row["subject_id"]: float(
                torch.sigmoid(model(tensors[row["subject_id"]])).reshape(()).cpu()
            )
            for row in rows
        }
