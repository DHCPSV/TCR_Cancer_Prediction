from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from analysis import protocol
from analysis.experiment_04_alice_model.methods import AliceOnly, HARD_THRESHOLD


EXPERIMENT_ID = "experiment_04_alice_model"
OUTPUT = protocol.REPO / "results" / EXPERIMENT_ID
CHECKPOINTS = protocol.REPO / "artifacts" / "checkpoints" / EXPERIMENT_ID
RUN_ARTIFACTS = protocol.REPO / "artifacts" / "runs" / EXPERIMENT_ID
ALICE = protocol.REPO / "artifacts" / "alice"

CHAINS = ("alpha", "beta")
EPOCHS = 50
LEARNING_RATE = 1e-3
ACCUMULATION = 4
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

    def build(self) -> AliceOnly:
        return AliceOnly(
            self.dimensions,
            norm=self.norm,
            head=self.head,
            threshold=self.threshold,
        )


# This is the complete registry for the confirmatory hard-gate study.
METHODS = (
    Method("alice_l1_linear", "L1 + Linear", CHAINS, ALICE_SEEDS, 64, "l1", "linear", HARD_THRESHOLD),
    Method("alice_l1_mlp", "L1 + MLP", CHAINS, ALICE_SEEDS, 64, "l1", "mlp", HARD_THRESHOLD),
    Method("alice_l2_linear", "L2 + Linear", CHAINS, ALICE_SEEDS, 64, "l2", "linear", HARD_THRESHOLD),
    Method("alice_l2_mlp", "L2 + MLP", CHAINS, ALICE_SEEDS, 64, "l2", "mlp", HARD_THRESHOLD),
)
METHOD_IDS = frozenset(method.method_id for method in METHODS)


def checkpoint(
    method: Method,
    chain: str,
    seed_id: str,
    fold: int,
    root: Path = CHECKPOINTS,
) -> Path:
    return root / method.method_id / chain / seed_id / f"fold_{fold}.pt"


def self_test() -> None:
    if len(METHODS) != 4 or len(METHOD_IDS) != 4:
        raise AssertionError("The hard-gate registry must contain exactly four methods")
    if any(method.threshold != HARD_THRESHOLD for method in METHODS):
        raise AssertionError("Every confirmatory method must use the 0.30 hard gate")
    if any(method.representation != "sceptr" for method in METHODS):
        raise AssertionError("The confirmatory registry is restricted to SCEPTR")
    for method in METHODS:
        state_keys = set(method.build().state_dict())
        if not all(key.startswith("classifier.") for key in state_keys):
            raise AssertionError(f"Unexpected checkpoint state keys for {method.method_id}")
