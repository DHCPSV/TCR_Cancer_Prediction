from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from analysis import protocol
from analysis.experiment_03_external_generalisation import study


def internal_fit_pca(
    vectors: np.ndarray,
    internal_mask: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Fit PCA on internal patients and transform both data roles without refitting."""
    component_count = min(
        64,
        int(internal_mask.sum()),
        vectors.shape[1],
    )
    pca = PCA(n_components=component_count).fit(vectors[internal_mask])
    scores = pca.transform(vectors)
    cancer = internal_mask & (labels == 1)
    control = internal_mask & (labels == 0)
    sign_flipped = bool(scores[cancer, 0].mean() < scores[control, 0].mean())
    if sign_flipped:
        scores[:, 0] *= -1
    standard_deviations = np.sqrt(
        np.maximum(pca.explained_variance_[:2], 1e-12)
    )
    return (
        scores[:, :2],
        standard_deviations,
        pca.explained_variance_ratio_[:2],
        sign_flipped,
    )


def _prediction_lookup(
    predictions: pd.DataFrame,
    method_id: str,
    chain: str,
    seed_id: str,
) -> dict[tuple[str, str], float]:
    frame = predictions.loc[
        (predictions["method_id"] == method_id)
        & (predictions["chain"] == chain)
        & (predictions["seed_id"] == seed_id)
    ]
    expected = 169 + (109 if chain == "alpha" else 110)
    if len(frame) != expected:
        raise ValueError(
            f"Incomplete seed predictions: {method_id}/{chain}/{seed_id}"
        )
    keys = ["split", "subject_id"]
    if frame.duplicated(keys).any():
        raise ValueError(
            f"Duplicate seed predictions: {method_id}/{chain}/{seed_id}"
        )
    return frame.set_index(keys)["score"].astype(float).to_dict()


def _patient_records(
    method_id: str,
    model_name: str,
    chain: str,
    seed_id: str,
    internal_rows: pd.DataFrame,
    external_rows: pd.DataFrame,
    folds: dict[str, int],
    internal_vectors: dict[str, np.ndarray],
    external_vectors: dict[str, np.ndarray],
    scores: dict[tuple[str, str], float],
) -> pd.DataFrame:
    seed_value, seed_group = study.seed_metadata(seed_id)
    records: list[dict] = []
    for row in internal_rows.itertuples(index=False):
        records.append(
            {
                "method_id": method_id,
                "model": model_name,
                "chain": chain,
                "seed_id": seed_id,
                "seed_value": str(seed_value),
                "seed_group": seed_group,
                "subject_id": row.subject_id,
                "dataset": "internal",
                "split": "internal_oof",
                "fold": int(folds[row.subject_id]),
                "group": (
                    "internal_cancer" if int(row.label) else "internal_control"
                ),
                "label": int(row.label),
                "tcr_count": int(row.tcr_count),
                "probability": scores[("internal_oof", row.subject_id)],
                "vector": internal_vectors[row.subject_id],
            }
        )
    for row in external_rows.itertuples(index=False):
        records.append(
            {
                "method_id": method_id,
                "model": model_name,
                "chain": chain,
                "seed_id": seed_id,
                "seed_value": str(seed_value),
                "seed_group": seed_group,
                "subject_id": row.subject_id,
                "dataset": "external",
                "split": "locked_external",
                "fold": -1,
                "group": row.cohort,
                "label": int(row.label),
                "tcr_count": int(row.tcr_count),
                "probability": scores[("locked_external", row.subject_id)],
                "vector": external_vectors[row.subject_id] / study.N_FOLDS,
            }
        )
    return pd.DataFrame(records)


def prepare(
    device: torch.device,
    internal_predictions: pd.DataFrame,
    external_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create the seed-resolved patient-vector PCA used in the report."""
    predictions = pd.concat(
        [internal_predictions, external_predictions],
        ignore_index=True,
    )
    point_parts: list[pd.DataFrame] = []
    for method in study.METHODS:
        method_id, model_name, _ = method
        for chain in study.CHAINS:
            internal_rows = study.representation_rows("internal", chain)
            external_rows = study.representation_rows("external", chain)
            folds = protocol.read_fixed_folds(
                chain,
                internal_rows["subject_id"],
            )
            internal_tensors = protocol.load_tensors(internal_rows, device)
            external_tensors = protocol.load_tensors(external_rows, device)

            with torch.no_grad():
                for seed_id in study.SEED_IDS:
                    internal_vectors = {
                        row.subject_id: np.zeros(64)
                        for row in internal_rows.itertuples(index=False)
                    }
                    external_vectors = {
                        row.subject_id: np.zeros(64)
                        for row in external_rows.itertuples(index=False)
                    }
                    internal_seen = {subject_id: 0 for subject_id in internal_vectors}
                    for fold in range(study.N_FOLDS):
                        model = study.load_attention_model(
                            method,
                            chain,
                            seed_id,
                            fold,
                            device,
                        )
                        for row in internal_rows.itertuples(index=False):
                            if folds[row.subject_id] == fold:
                                vector = (
                                    model.patient_vector(
                                        internal_tensors[row.subject_id]
                                    )
                                    .squeeze(0)
                                    .cpu()
                                    .numpy()
                                )
                                internal_vectors[row.subject_id] = vector
                                internal_seen[row.subject_id] += 1
                        for row in external_rows.itertuples(index=False):
                            vector = (
                                model.patient_vector(
                                    external_tensors[row.subject_id]
                                )
                                .squeeze(0)
                                .cpu()
                                .numpy()
                            )
                            external_vectors[row.subject_id] += vector

                    if set(internal_seen.values()) != {1}:
                        raise ValueError(
                            "Internal PCA input is not one-vector-per-patient OOF: "
                            f"{method_id}/{chain}/{seed_id}"
                        )
                    scores = _prediction_lookup(
                        predictions,
                        method_id,
                        chain,
                        seed_id,
                    )
                    seed_frame = _patient_records(
                        method_id,
                        model_name,
                        chain,
                        seed_id,
                        internal_rows,
                        external_rows,
                        folds,
                        internal_vectors,
                        external_vectors,
                        scores,
                    )
                    vectors = np.stack(seed_frame.pop("vector"))
                    internal_mask = seed_frame["dataset"].eq("internal").to_numpy()
                    pc_scores, pc_sds, variance_ratios, sign_flipped = (
                        internal_fit_pca(
                            vectors,
                            internal_mask,
                            seed_frame["label"].to_numpy(),
                        )
                    )
                    probability = (
                        seed_frame["probability"]
                        .clip(1e-6, 1 - 1e-6)
                        .to_numpy()
                    )
                    seed_frame["logit"] = np.log(
                        probability / (1 - probability)
                    )
                    seed_frame["pc1_raw"] = pc_scores[:, 0]
                    seed_frame["pc1_internal_sd"] = pc_scores[:, 0] / pc_sds[0]
                    seed_frame["pc1_sd"] = pc_sds[0]
                    seed_frame["pc1_variance_ratio"] = variance_ratios[0]
                    seed_frame["pc2_raw"] = pc_scores[:, 1]
                    seed_frame["pc2_internal_sd"] = pc_scores[:, 1] / pc_sds[1]
                    seed_frame["pc2_sd"] = pc_sds[1]
                    seed_frame["pc2_variance_ratio"] = variance_ratios[1]
                    seed_frame["pc1_sign_flipped"] = sign_flipped
                    point_parts.append(seed_frame)

            print(f"PCA: {method_id}, {chain}", flush=True)
            del internal_tensors, external_tensors
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = pd.concat(point_parts, ignore_index=True)
    protocol.atomic_csv(
        result,
        study.PCA / "per_seed_patient_coordinates.csv",
    )
    return result


def self_test() -> None:
    assert study.N_FOLDS == 5
    internal = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.2], [1.0, -0.1], [2.0, 0.1]]
    )
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    mask = np.asarray([True, True, True, True, False, False])
    first = internal_fit_pca(
        np.vstack([internal, [[10.0, 10.0], [12.0, 8.0]]]),
        mask,
        labels,
    )
    second = internal_fit_pca(
        np.vstack([internal, [[-100.0, 50.0], [80.0, -90.0]]]),
        mask,
        labels,
    )
    np.testing.assert_allclose(first[0][mask], second[0][mask], atol=1e-12)
    np.testing.assert_allclose(first[1], second[1], atol=1e-12)
    np.testing.assert_allclose(first[2], second[2], atol=1e-12)
    assert first[3] == second[3]
    print("experiment_03_external_generalisation PCA self-test passed")
