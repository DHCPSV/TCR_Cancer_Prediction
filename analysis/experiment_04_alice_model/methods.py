from __future__ import annotations

import numpy as np
import torch


HARD_THRESHOLD = 0.30


def alice_pool(
    embeddings: torch.Tensor,
    evidence: np.ndarray | torch.Tensor,
    *,
    norm: str,
    threshold: float | None = HARD_THRESHOLD,
) -> torch.Tensor:
    """Pool one repertoire with the ALICE evidence used in Experiment 04."""
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


class AliceOnly(torch.nn.Module):
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
    """Checkpoint-compatible view of the retained linear classifier head."""

    def __init__(self, dimensions: int = 64):
        super().__init__()
        self.classifier = torch.nn.Linear(dimensions, 1)

    def forward(self, patient_vector: torch.Tensor) -> torch.Tensor:
        return self.classifier(patient_vector)


def self_test() -> None:
    embedding = torch.tensor([[1.0, 0.0], [3.0, 2.0], [20.0, 20.0]])
    evidence = np.array([0.6, 0.4, 0.2], dtype=np.float32)
    hard_l1 = alice_pool(embedding, evidence, norm="l1")
    hard_l2 = alice_pool(embedding, evidence, norm="l2")
    no_gate_l1 = alice_pool(embedding, evidence, norm="l1", threshold=None)
    torch.testing.assert_close(hard_l1, torch.tensor([[1.8, 0.8]]))
    torch.testing.assert_close(hard_l2, hard_l1 / np.sqrt(0.52))
    torch.testing.assert_close(no_gate_l1, torch.tensor([[4.8333335, 4.0]]))
    for norm in ("l1", "l2"):
        for head in ("linear", "mlp"):
            model = AliceOnly(2, norm=norm, head=head)
            output = model({"embedding": embedding, "evidence": evidence})
            assert output.shape == (1, 1)
            output.sum().backward()
