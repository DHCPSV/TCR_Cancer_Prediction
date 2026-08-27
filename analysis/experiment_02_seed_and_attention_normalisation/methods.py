from __future__ import annotations

import torch
from entmax import entmax15 as library_entmax15
from entmax import sparsemax as library_sparsemax


def normalise_attention(scores: torch.Tensor, normalizer: str) -> torch.Tensor:
    if normalizer == "softmax":
        return torch.softmax(scores, dim=1)
    if normalizer == "entmax15":
        return library_entmax15(scores, dim=1)
    if normalizer == "sparsemax":
        return library_sparsemax(scores, dim=1)
    raise ValueError(normalizer)


class AttentionMIL(torch.nn.Module):
    def __init__(self, normalizer: str):
        super().__init__()
        self.attention_score = torch.nn.Linear(64, 1)
        self.classifier = torch.nn.Linear(64, 1)
        self.normalizer = normalizer

    def attention_weights(self, embeddings: torch.Tensor) -> torch.Tensor:
        return normalise_attention(self.attention_score(embeddings).T, self.normalizer).T

    def patient_vector(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.attention_weights(embeddings) * embeddings, dim=0, keepdim=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.patient_vector(embeddings))


class SparsemaxMIL(torch.nn.Module):
    """Canonical architecture for the frozen layer-seed factorisation."""

    def __init__(self):
        super().__init__()
        self.attention_score = torch.nn.Linear(64, 1)
        self.classifier = torch.nn.Linear(64, 1)

    def attention_weights(self, embeddings: torch.Tensor) -> torch.Tensor:
        return library_sparsemax(self.attention_score(embeddings).T, dim=1).T

    def patient_vector(self, embeddings: torch.Tensor) -> torch.Tensor:
        weights = self.attention_weights(embeddings)
        return torch.sum(weights * embeddings, dim=0, keepdim=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.patient_vector(embeddings))
