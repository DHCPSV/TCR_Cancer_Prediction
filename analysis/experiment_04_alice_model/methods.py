from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from analysis import protocol


EXPERIMENT_ID = "experiment_04_alice_model"
NO_HARD_GATE_EXPERIMENT_ID = "experiment_04_alice_model_no_hard_gate"
RESULTS = protocol.repo_path(f"results/{EXPERIMENT_ID}")
CHECKPOINTS = protocol.repo_path(f"artifacts/checkpoints/{EXPERIMENT_ID}")
RUN_ARTIFACTS = protocol.repo_path(f"artifacts/runs/{EXPERIMENT_ID}")
NO_HARD_GATE_CHECKPOINTS = protocol.repo_path(
    f"artifacts/checkpoints/{NO_HARD_GATE_EXPERIMENT_ID}"
)
NO_HARD_GATE_RUN_ARTIFACTS = protocol.repo_path(
    f"artifacts/runs/{NO_HARD_GATE_EXPERIMENT_ID}"
)
NO_HARD_GATE_RESULTS = RESULTS / "no_hard_gate"
ALICE = protocol.repo_path("artifacts/alice")

CHAINS = ("alpha", "beta")
EPOCHS = 50
LEARNING_RATE = 1e-3
ACCUMULATION = 4
HARD_THRESHOLD = 0.30
ALICE_SEEDS = (
    "S01",
    "S02",
    "S10",
    "S04",
    "S05",
    "S11",
    "S07",
    "S08",
    "S12",
    "S13",
)
REPRESENTATIVE_SEEDS = ("S03", "S06", "S09")


def alice_pool(
    embeddings: torch.Tensor,
    evidence: np.ndarray | torch.Tensor,
    *,
    norm: str,
    threshold: float | None = HARD_THRESHOLD,
) -> torch.Tensor:
    """Pool one repertoire with its ALICE evidence."""
    weights = torch.as_tensor(
        evidence,
        dtype=embeddings.dtype,
        device=embeddings.device,
    ).reshape(-1)
    if embeddings.ndim != 2 or len(embeddings) != len(weights):
        raise ValueError("Expected N x D embeddings and one ALICE evidence value per TCR")
    if torch.any(weights < 0):
        raise ValueError("ALICE evidence must be non-negative")
    if threshold is not None:
        weights = torch.where(weights > threshold, weights, 0.0)
    numerator = torch.sum(weights[:, None] * embeddings, dim=0, keepdim=True)
    if norm == "l1":
        denominator = weights.sum()
    elif norm == "l2":
        denominator = torch.linalg.vector_norm(weights, ord=2)
    else:
        raise ValueError(f"Unknown ALICE pooling normalisation: {norm}")
    return numerator / denominator.clamp_min(1e-12)


class AlicePoolingClassifier(torch.nn.Module):
    def __init__(
        self,
        dimensions: int,
        *,
        norm: str,
        head: str,
        threshold: float | None = HARD_THRESHOLD,
    ):
        super().__init__()
        self.norm = norm
        self.head = head
        self.threshold = threshold
        if head == "linear":
            self.classifier = torch.nn.Linear(dimensions, 1)
        elif head == "mlp":
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(dimensions, 32),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(32, 1),
            )
        else:
            raise ValueError(f"Unknown classifier head: {head}")

    def patient_vector(self, feature: dict) -> torch.Tensor:
        return alice_pool(
            feature["embedding"],
            feature["evidence"],
            norm=self.norm,
            threshold=self.threshold,
        )

    def forward(self, feature: dict) -> torch.Tensor:
        return self.classifier(self.patient_vector(feature))


class LinearHead(torch.nn.Module):
    """Checkpoint-compatible view of the linear classifier head."""

    def __init__(self, dimensions: int = 64):
        super().__init__()
        self.classifier = torch.nn.Linear(dimensions, 1)

    def forward(self, patient_vector: torch.Tensor) -> torch.Tensor:
        return self.classifier(patient_vector)


@dataclass(frozen=True)
class Method:
    method_id: str
    label: str
    chains: tuple[str, ...]
    seed_ids: tuple[str, ...]
    dimensions: int
    norm: str
    head: str
    threshold: float | None
    representation: str = "sceptr"

    def build(self) -> AlicePoolingClassifier:
        return AlicePoolingClassifier(
            self.dimensions,
            norm=self.norm,
            head=self.head,
            threshold=self.threshold,
        )


MAIN_METHODS = (
    Method("alice_l1_linear", "L1 + Linear", CHAINS, ALICE_SEEDS, 64, "l1", "linear", HARD_THRESHOLD),
    Method("alice_l1_mlp", "L1 + MLP", CHAINS, ALICE_SEEDS, 64, "l1", "mlp", HARD_THRESHOLD),
    Method("alice_l2_linear", "L2 + Linear", CHAINS, ALICE_SEEDS, 64, "l2", "linear", HARD_THRESHOLD),
    Method("alice_l2_mlp", "L2 + MLP", CHAINS, ALICE_SEEDS, 64, "l2", "mlp", HARD_THRESHOLD),
)

# These IDs are retained because they are stored inside the released checkpoints.
NO_HARD_GATE_METHODS = (
    Method("no_gate_l1_linear", "No gate + L1 + Linear", CHAINS, ALICE_SEEDS, 64, "l1", "linear", None),
    Method("no_gate_l1_mlp", "No gate + L1 + MLP", CHAINS, ALICE_SEEDS, 64, "l1", "mlp", None),
    Method("no_gate_l2_linear", "No gate + L2 + Linear", CHAINS, ALICE_SEEDS, 64, "l2", "linear", None),
    Method("no_gate_l2_mlp", "No gate + L2 + MLP", CHAINS, ALICE_SEEDS, 64, "l2", "mlp", None),
)

REPRESENTATION_LABELS = {
    "sceptr": "SCEPTR",
    "atchley": "Atchley factors",
    "kidera": "Kidera factors",
    "aa_property": "Amino-acid properties",
}
REPRESENTATION_METHODS = tuple(
    Method(
        f"{representation}_{norm}",
        f"{REPRESENTATION_LABELS[representation]} + ALICE {norm.upper()}",
        ("alpha",),
        REPRESENTATIVE_SEEDS,
        dimensions,
        norm,
        "linear",
        HARD_THRESHOLD,
        representation,
    )
    for representation, dimensions in (
        ("sceptr", 64),
        ("atchley", 5),
        ("kidera", 10),
        ("aa_property", 14),
    )
    for norm in ("l1", "l2")
)


def checkpoint_path(
    root: Path,
    method: Method,
    chain: str,
    seed_id: str,
    fold: int,
) -> Path:
    return root / method.method_id / chain / seed_id / f"fold_{fold}.pt"


def self_test() -> None:
    embedding = torch.tensor([[1.0, 0.0], [3.0, 2.0], [20.0, 20.0]])
    evidence = np.array([0.6, 0.4, 0.2], dtype=np.float32)
    hard_l1 = alice_pool(embedding, evidence, norm="l1")
    hard_l2 = alice_pool(embedding, evidence, norm="l2")
    no_hard_gate_l1 = alice_pool(embedding, evidence, norm="l1", threshold=None)
    torch.testing.assert_close(hard_l1, torch.tensor([[1.8, 0.8]]))
    torch.testing.assert_close(hard_l2, hard_l1 / np.sqrt(0.52))
    torch.testing.assert_close(no_hard_gate_l1, torch.tensor([[4.8333335, 4.0]]))
    assert len(MAIN_METHODS) == 4
    assert len(NO_HARD_GATE_METHODS) == 4
    assert len(REPRESENTATION_METHODS) == 8
    for method in MAIN_METHODS + NO_HARD_GATE_METHODS + REPRESENTATION_METHODS:
        model = method.build()
        output = model({"embedding": torch.randn(3, method.dimensions), "evidence": evidence})
        assert output.shape == (1, 1)
        assert all(key.startswith("classifier.") for key in model.state_dict())
