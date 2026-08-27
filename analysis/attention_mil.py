"""Attention-based multiple-instance model shared by Experiments 1--3."""

from __future__ import annotations

import torch
from entmax import entmax15, sparsemax


def normalise_attention(scores: torch.Tensor, normalizer: str) -> torch.Tensor:
    if normalizer == "softmax":
        return torch.softmax(scores, dim=1)
    if normalizer == "entmax15":
        return entmax15(scores, dim=1)
    if normalizer == "sparsemax":
        return sparsemax(scores, dim=1)
    raise ValueError(f"Unknown attention normalizer: {normalizer}")


class AttentionMIL(torch.nn.Module):
    def __init__(self, normalizer: str, embedding_dim: int = 64):
        super().__init__()
        self.attention_score = torch.nn.Linear(embedding_dim, 1)
        self.classifier = torch.nn.Linear(embedding_dim, 1)
        self.normalizer = normalizer

    def attention_weights(self, embeddings: torch.Tensor) -> torch.Tensor:
        scores = self.attention_score(embeddings).T
        return normalise_attention(scores, self.normalizer).T

    def patient_vector(self, embeddings: torch.Tensor) -> torch.Tensor:
        weights = self.attention_weights(embeddings)
        return torch.sum(weights * embeddings, dim=0, keepdim=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.patient_vector(embeddings))
