from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis import protocol
from analysis.experiment_04_alice_model import core, report as reporting, study
from analysis.experiment_04_alice_model.methods import AliceOnly, alice_pool


EXPERIMENT_ID = "experiment_04_alice_model_no_hard_gate"
OUTPUT = core.OUTPUT / "no_hard_gate"
CHECKPOINTS = protocol.REPO / "artifacts" / "checkpoints" / EXPERIMENT_ID
RUN_ARTIFACTS = protocol.REPO / "artifacts" / "runs" / EXPERIMENT_ID


class NoHardGateModel(AliceOnly):
    def __init__(self, dimensions: int, *, norm: str, head: str):
        super().__init__(dimensions, norm=norm, head=head, threshold=None)


def no_hard_gate_pool(
    embeddings: torch.Tensor,
    evidence: np.ndarray | torch.Tensor,
    *,
    norm: str,
) -> torch.Tensor:
    return alice_pool(embeddings, evidence, norm=norm, threshold=None)


METHODS = (
    core.Method("no_gate_l1_linear", "No gate + L1 + Linear", core.CHAINS, core.ALICE_SEEDS, 64, "l1", "linear", None),
    core.Method("no_gate_l1_mlp", "No gate + L1 + MLP", core.CHAINS, core.ALICE_SEEDS, 64, "l1", "mlp", None),
    core.Method("no_gate_l2_linear", "No gate + L2 + Linear", core.CHAINS, core.ALICE_SEEDS, 64, "l2", "linear", None),
    core.Method("no_gate_l2_mlp", "No gate + L2 + MLP", core.CHAINS, core.ALICE_SEEDS, 64, "l2", "mlp", None),
)

STUDY = study.StudySpec(
    experiment_id=EXPERIMENT_ID,
    cache_consumer=EXPERIMENT_ID,
    methods=METHODS,
    checkpoints=CHECKPOINTS,
    run_artifacts=RUN_ARTIFACTS,
    output=OUTPUT,
    checkpoint_schema="no_gate",
    progress_label="Experiment 04 no-gate models",
)


def checkpoint(method: core.Method, chain: str, seed_id: str, fold: int) -> Path:
    return study.checkpoint(STUDY, method, chain, seed_id, fold)


def metadata(method: core.Method, chain: str, seed_id: str, fold: int) -> dict:
    return study.checkpoint_metadata(STUDY, method, chain, seed_id, fold)


def internal(device: torch.device) -> pd.DataFrame:
    return study.internal(STUDY, device)


def validation(device: torch.device) -> pd.DataFrame:
    return study.validation(STUDY, device)


def report() -> pd.DataFrame:
    return reporting.no_gate_report(STUDY)


def self_test() -> None:
    embedding = torch.tensor([[1.0, 0.0], [3.0, 2.0], [20.0, 20.0]])
    evidence = np.array([0.6, 0.4, 0.2], dtype=np.float32)
    l1 = no_hard_gate_pool(embedding, evidence, norm="l1")
    l2 = no_hard_gate_pool(embedding, evidence, norm="l2")
    torch.testing.assert_close(l1, torch.tensor([[4.8333335, 4.0]]))
    torch.testing.assert_close(
        l2,
        torch.tensor([[5.8, 4.8]]) / torch.sqrt(torch.tensor(0.56)),
    )
    for method in METHODS:
        model_embedding = torch.randn(len(evidence), method.dimensions)
        output = method.build()(
            {"embedding": model_embedding, "evidence": evidence}
        )
        assert output.shape == (1, 1)
        assert method.threshold is None
    assert {method.method_id for method in METHODS} == {
        "no_gate_l1_linear",
        "no_gate_l1_mlp",
        "no_gate_l2_linear",
        "no_gate_l2_mlp",
    }
    print("no-hard-gate ALICE self-test passed")


def run_frozen(device: torch.device, internal_states: tuple[Path, ...]) -> None:
    study.guard_internal_cache(STUDY, internal_states)
    internal(device)
    validation(device)
    report()
