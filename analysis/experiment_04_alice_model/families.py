from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch

from analysis import protocol, reporting
from analysis.experiment_04_alice_model import methods, study
from analysis.experiment_04_alice_model.methods import (
    HARD_THRESHOLD as ALICE_THRESHOLD,
    LinearHead,
)


REFERENCE_METHOD_ID = "alice_l2_linear"
REFERENCE_NORMALISATION = "l2"
REFERENCE_SEED_ID = "S01"
REFERENCE_SEED_VALUE = int(protocol.SEED_REGISTRY[REFERENCE_SEED_ID]["value"])
EXPERIMENT_ID = methods.EXPERIMENT_ID
METHOD = next(
    method for method in methods.MAIN_METHODS if method.method_id == REFERENCE_METHOD_ID
)
METHOD_ID = METHOD.method_id


def checkpoint_metadata(chain: str, seed_id: str, fold: int) -> dict:
    return study.main_checkpoint_metadata(METHOD, chain, seed_id, fold)


def dataset(chain: str) -> pd.DataFrame:
    frame = study.dataset("internal", chain, "sceptr")
    alice = pd.read_csv(ALICE / "internal_manifest.csv", dtype=str, keep_default_na=False)
    alice = alice.loc[(alice["split"] == "internal") & (alice["chain"] == chain)]
    return frame.merge(
        alice[["subject_id", "sequence_map_file"]],
        on="subject_id",
        validate="one_to_one",
    )


FAMILY_ARTIFACTS = methods.RUN_ARTIFACTS / "families"
FAMILY_RANKS = FAMILY_ARTIFACTS / "ranked"
ALICE = methods.ALICE


def fold_heads() -> dict[int, tuple[torch.Tensor, float]]:
    heads: dict[int, tuple[torch.Tensor, float]] = {}
    for fold in range(5):
        path = methods.CHECKPOINTS / METHOD_ID / "alpha" / REFERENCE_SEED_ID / f"fold_{fold}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = checkpoint_metadata("alpha", REFERENCE_SEED_ID, fold)
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Checkpoint metadata mismatch: {path}")
        model = LinearHead()
        model.load_state_dict(payload["model_state"])
        heads[fold] = (
            model.classifier.weight.detach().reshape(-1),
            float(model.classifier.bias.detach()),
        )
    return heads


def l2_logit_contributions(
    projected_embeddings: torch.Tensor,
    evidence: torch.Tensor,
) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(evidence, ord=2).clamp_min(
        torch.finfo(evidence.dtype).eps
    )
    return projected_embeddings * evidence / denominator


def sequence_contributions() -> pd.DataFrame:
    patients = dataset("alpha")
    heads = fold_heads()
    frames = []
    for patient in patients.itertuples(index=False):
        embedding = torch.load(
            protocol.repo_path(patient.embedding_file), map_location="cpu", weights_only=True
        ).float()
        evidence = np.load(protocol.repo_path(patient.evidence_file), allow_pickle=False)
        mapping = np.load(protocol.repo_path(patient.sequence_map_file), allow_pickle=False)
        selected = np.flatnonzero(evidence > ALICE_THRESHOLD)
        if not len(selected):
            continue
        classifier_weight, classifier_bias = heads[patient.fold]
        weights = torch.from_numpy(evidence[selected]).float()
        contribution = l2_logit_contributions(
            embedding[selected] @ classifier_weight,
            weights,
        ).numpy()
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": patient.subject_id,
                    "source_patient_id": patient.source_patient_id,
                    "true_label": patient.label,
                    "fold": patient.fold,
                    "sample_index": mapping["sample_index"][selected].astype(int),
                    "clone_key": mapping["clone_index"][selected].astype(int),
                    "alice_evidence": evidence[selected],
                    "logit_contribution": contribution,
                    "classifier_bias": classifier_bias,
                }
            ).query("clone_key >= 0")
        )
    if not frames:
        raise RuntimeError("No selected internal Alpha clonotypes")
    positions = pd.concat(frames, ignore_index=True)
    keys = ["subject_id", "source_patient_id", "true_label", "fold", "sample_index", "clone_key"]
    positions = positions.groupby(keys, as_index=False).agg(
        alice_evidence=("alice_evidence", "max"),
        logit_contribution=("logit_contribution", "sum"),
        classifier_bias=("classifier_bias", "first"),
        embedding_multiplicity=("logit_contribution", "size"),
    )
    positions["reference_method_id"] = METHOD_ID
    positions["reference_method_label"] = METHOD.label
    positions["pooling_normalisation"] = REFERENCE_NORMALISATION
    positions["reference_seed_id"] = REFERENCE_SEED_ID
    positions["reference_seed_value"] = REFERENCE_SEED_VALUE
    return positions


def add_sequences(positions: pd.DataFrame) -> pd.DataFrame:
    samples = pd.read_csv(ALICE / "internal_samples.csv", dtype=str)
    samples = samples.loc[samples["chain"] == "alpha"].copy()
    samples["sample_index"] = samples["sample_index"].astype(int)
    grouped = {key: part for key, part in positions.groupby(["subject_id", "sample_index"], sort=False)}
    frames = []
    for sample in samples.itertuples(index=False):
        key = sample.subject_id, sample.sample_index
        if key not in grouped:
            continue
        source = pd.read_csv(protocol.repo_path(sample.input_path), sep="\t")
        alice = pd.read_csv(protocol.repo_path(sample.output_path), sep="\t")
        details = source.merge(alice, on="clone_key", validate="one_to_one")[[
            "clone_key", "cdr3nt", "cdr3aa", "bestVGene", "bestJGene", "D", "p", "q"
        ]]
        part = grouped[key].merge(details, on="clone_key", validate="one_to_one")
        part["sample_id"] = sample.sample_id
        frames.append(part)
    if not frames:
        raise RuntimeError("No ALICE sequence records matched selected clonotypes")
    return pd.concat(frames, ignore_index=True)


def connected_components(sequences: list[str]) -> list[int]:
    parent = list(range(len(sequences)))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    wildcard: dict[str, int] = {}
    for index, sequence in enumerate(sequences):
        for position in range(len(sequence)):
            key = sequence[:position] + "*" + sequence[position + 1 :]
            if key in wildcard:
                join(index, wildcard[key])
            else:
                wildcard[key] = index
    return [root(index) for index in range(len(sequences))]


def build_families(clonotypes: pd.DataFrame) -> pd.DataFrame:
    exact = clonotypes.groupby(
        [
            "subject_id", "source_patient_id", "true_label", "fold", "sample_index",
            "sample_id", "cdr3aa", "bestVGene", "bestJGene",
        ],
        as_index=False,
    ).agg(
        clone_keys=("clone_key", lambda values: ";".join(map(str, values))),
        cdr3nt=("cdr3nt", lambda values: ";".join(pd.unique(values))),
        min_q=("q", "min"),
        min_p=("p", "min"),
        max_D=("D", "max"),
        max_alice_evidence=("alice_evidence", "max"),
        embedding_multiplicity=("embedding_multiplicity", "sum"),
        clonotype_logit_contribution=("logit_contribution", "sum"),
    )
    exact["cdr3_length"] = exact["cdr3aa"].str.len()
    keys = [
        "subject_id", "source_patient_id", "true_label", "fold", "sample_index",
        "sample_id", "bestVGene", "bestJGene", "cdr3_length",
    ]
    rows = []
    for group_key, group in exact.groupby(keys, sort=False):
        group = group.reset_index(drop=True)
        group["family"] = connected_components(group["cdr3aa"].tolist())
        for _, members in group.groupby("family", sort=False):
            center = members.loc[members["clonotype_logit_contribution"].abs().idxmax()]
            rows.append(
                dict(zip(keys, group_key))
                | {
                    "source_label": "cancer_patient" if group_key[2] else "control_patient",
                    "center_cdr3aa": center.cdr3aa,
                    "center_cdr3nt": center.cdr3nt,
                    "member_count": len(members),
                    "member_cdr3aa": ";".join(members["cdr3aa"]),
                    "member_cdr3nt": ";".join(members["cdr3nt"]),
                    "clone_keys": ";".join(members["clone_keys"]),
                    "min_q": members["min_q"].min(),
                    "min_p": members["min_p"].min(),
                    "max_D": members["max_D"].max(),
                    "max_alice_evidence": members["max_alice_evidence"].max(),
                    "embedding_multiplicity": members["embedding_multiplicity"].sum(),
                    "family_logit_contribution": members["clonotype_logit_contribution"].sum(),
                }
            )
    families = pd.DataFrame(rows)
    families["reference_method_id"] = METHOD_ID
    families["reference_method_label"] = METHOD.label
    families["pooling_normalisation"] = REFERENCE_NORMALISATION
    families["reference_seed_id"] = REFERENCE_SEED_ID
    families["reference_seed_value"] = REFERENCE_SEED_VALUE
    return families


def rank_families(families: pd.DataFrame) -> pd.DataFrame:
    cancer = families.nlargest(50, "family_logit_contribution").copy()
    cancer[["rank_type", "model_direction"]] = ["pushes_cancer", "toward_cancer"]
    control = families.nsmallest(50, "family_logit_contribution").copy()
    control[["rank_type", "model_direction"]] = ["pushes_control", "toward_control"]
    q_value = families.nsmallest(100, "min_q").copy()
    q_value[["rank_type", "model_direction"]] = ["smallest_alice_q", "not_ranked_by_direction"]
    ranked = pd.concat([cancer, control, q_value], ignore_index=True)
    ranked["rank_within_type"] = ranked.groupby("rank_type").cumcount() + 1
    return ranked


def internal() -> pd.DataFrame:
    positions = sequence_contributions()
    clonotypes = add_sequences(positions)
    families = build_families(clonotypes)
    protocol.atomic_csv(positions, FAMILY_ARTIFACTS / "sequence_contributions.csv")
    protocol.atomic_csv(clonotypes, FAMILY_ARTIFACTS / "significant_clonotypes.csv")
    protocol.atomic_csv(families, FAMILY_ARTIFACTS / "patient_local_families.csv")
    return families


def report(families: pd.DataFrame | None = None) -> pd.DataFrame:
    reporting.configure_plot()
    if families is None:
        families = pd.read_csv(FAMILY_ARTIFACTS / "patient_local_families.csv")
    ranked = rank_families(families)
    protocol.atomic_csv(
        ranked.loc[ranked["rank_type"] == "pushes_cancer"],
        FAMILY_RANKS / "top_50_pushes_cancer.csv",
    )
    protocol.atomic_csv(
        ranked.loc[ranked["rank_type"] == "pushes_control"],
        FAMILY_RANKS / "top_50_pushes_control.csv",
    )
    protocol.atomic_csv(
        ranked.loc[ranked["rank_type"] == "smallest_alice_q"],
        FAMILY_RANKS / "top_100_smallest_alice_q.csv",
    )
    extremes = pd.concat(
        [families.nsmallest(15, "family_logit_contribution"), families.nlargest(15, "family_logit_contribution")]
    ).sort_values("family_logit_contribution")
    labels = extremes["center_cdr3aa"] + " | " + extremes["subject_id"]
    colors = extremes["true_label"].map({0: "#2563eb", 1: "#dc2626"})
    figure, axis = plt.subplots(figsize=(10.2, 10.8), constrained_layout=True)
    axis.barh(labels, extremes["family_logit_contribution"], color=colors)
    axis.axvline(0, color="#52637a", linewidth=1)
    axis.set(
        xlabel="Held-out L2 + Linear cancer-logit contribution (seed 777)",
    )
    axis.legend(
        handles=[
            Patch(color="#2563eb", label="Control patient"),
            Patch(color="#dc2626", label="Cancer patient"),
        ],
        loc="lower right",
    )
    path = methods.RESULTS / "figures" / "family_contributions.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)

    return ranked


def self_test() -> None:
    components = connected_components(["CAVAAA", "CAVAAB", "CAVBBB"])
    assert components[0] == components[1] != components[2]
    assert len(set(connected_components(["AAAA", "AAAB", "AABB"]))) == 1
    projected = torch.tensor([2.0, 2.0])
    evidence = torch.tensor([0.5, 0.5])
    observed = l2_logit_contributions(projected, evidence)
    expected = projected.new_tensor([np.sqrt(2.0), np.sqrt(2.0)])
    assert torch.allclose(observed, expected)
    assert METHOD_ID == "alice_l2_linear"
    assert REFERENCE_SEED_VALUE == 777
    print(f"{EXPERIMENT_ID} families self-test passed")
