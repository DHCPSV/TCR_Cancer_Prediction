from __future__ import annotations

import torch
from entmax import sparsemax


# The registry is deliberately next to the canonical model: these are the five
# representations compared by Experiment 01, not generic pipeline options.
METHODS = (
    ("sceptr_sparsemax", "SCEPTR", "sceptr", 64),
    ("atchley_sparsemax", "Atchley factors", "atchley", 5),
    ("kidera_sparsemax", "Kidera factors", "kidera", 10),
    ("aa_property_sparsemax", "Amino-acid properties", "aa_property", 14),
    ("random_sparsemax", "Random descriptor control", "random", 5),
)


class SparsemaxAttentionMIL(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention_score = torch.nn.Linear(embedding_dim, 1)
        self.classifier = torch.nn.Linear(embedding_dim, 1)

    def attention_weights(self, embeddings: torch.Tensor) -> torch.Tensor:
        return sparsemax(self.attention_score(embeddings).T, dim=1).T

    def patient_vector(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.attention_weights(embeddings) * embeddings, dim=0, keepdim=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.patient_vector(embeddings))
