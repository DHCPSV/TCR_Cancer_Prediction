from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analysis import protocol, reporting
from analysis.experiment_04_alice_model import methods, study
from analysis.experiment_04_alice_model.methods import alice_pool


CHAINS = ("alpha", "beta")
NORMS = ("l1", "l2")
OUTPUT = methods.RESULTS / "future_work" / "gate_free_pca"
RUN_ARTIFACTS = methods.RUN_ARTIFACTS / "future_work" / "gate_free_pca"
FIGURE = OUTPUT / "figures" / "gate_free_alice_patient_vector_pca.png"


def pooled_vectors(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    features = study.load_features(frame)
    matrices: dict[str, np.ndarray] = {}
    for norm in NORMS:
        matrices[norm] = np.vstack(
            [
                alice_pool(
                    features[subject_id]["embedding"],
                    features[subject_id]["evidence"],
                    norm=norm,
                    threshold=None,
                )
                .reshape(-1)
                .numpy()
                for subject_id in frame["subject_id"]
            ]
        )
    return matrices


def orient_components(
    pca: PCA,
    internal_scores: np.ndarray,
    internal_labels: np.ndarray,
) -> np.ndarray:
    signs = np.ones(2, dtype=np.float64)
    cancer_mean = internal_scores[internal_labels == 1, 0].mean()
    control_mean = internal_scores[internal_labels == 0, 0].mean()
    if cancer_mean < control_mean:
        signs[0] = -1.0
    second = pca.components_[1]
    if second[np.argmax(np.abs(second))] < 0:
        signs[1] = -1.0
    return signs


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    coordinate_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for chain in CHAINS:
        internal = study.dataset("internal", chain, "sceptr")
        external = study.dataset("external", chain, "sceptr")
        internal_vectors = pooled_vectors(internal)
        external_vectors = pooled_vectors(external)
        labels = internal["label"].to_numpy(dtype=int)
        for norm in NORMS:
            pca = PCA(n_components=2).fit(internal_vectors[norm])
            internal_scores = pca.transform(internal_vectors[norm])
            external_scores = pca.transform(external_vectors[norm])
            signs = orient_components(pca, internal_scores, labels)
            scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
            for split, frame, scores in (
                ("internal", internal, internal_scores),
                ("external", external, external_scores),
            ):
                oriented = scores * signs[None, :]
                part = frame[["subject_id", "cohort", "label"]].copy()
                part.insert(0, "split", split)
                part.insert(0, "pooling_normalisation", norm)
                part.insert(0, "chain", chain)
                part["pc1"] = oriented[:, 0]
                part["pc2"] = oriented[:, 1]
                part["pc1_internal_sd"] = oriented[:, 0] / scale[0]
                part["pc2_internal_sd"] = oriented[:, 1] / scale[1]
                coordinate_parts.append(part)
            summary_rows.append(
                {
                    "chain": chain,
                    "pooling_normalisation": norm,
                    "pca_fit_scope": "internal_only",
                    "internal_patients": len(internal),
                    "external_patients": len(external),
                    "pc1_variance_ratio": pca.explained_variance_ratio_[0],
                    "pc2_variance_ratio": pca.explained_variance_ratio_[1],
                    "pc1_sign_rule": "internal_cancer_mean_ge_control",
                    "pc1_sign_flipped": bool(signs[0] < 0),
                    "pc2_sign_rule": "largest_absolute_loading_positive",
                    "pc2_sign_flipped": bool(signs[1] < 0),
                }
            )
    coordinates = pd.concat(coordinate_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    protocol.atomic_csv(coordinates, RUN_ARTIFACTS / "patient_coordinates.csv")
    protocol.atomic_csv(summary, RUN_ARTIFACTS / "summary.csv")
    return coordinates, summary


def plot(coordinates: pd.DataFrame, summary: pd.DataFrame, path: Path = FIGURE) -> None:
    reporting.configure_plot()
    styles = {
        "internal_control": ("Internal control", "#2563eb", "o"),
        "tx100_cancer": ("Internal cancer", "#dc2626", "o"),
        "bcg_control": ("External control", "#60a5fa", "^"),
        "additional_tracerx_cancer": ("External cancer", "#f59e0b", "^"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.4), constrained_layout=True)
    for row_index, chain in enumerate(CHAINS):
        for column_index, norm in enumerate(NORMS):
            axis = axes[row_index, column_index]
            part = coordinates.loc[
                (coordinates["chain"] == chain)
                & (coordinates["pooling_normalisation"] == norm)
            ]
            for cohort, (label, colour, marker) in styles.items():
                cohort_part = part.loc[part["cohort"] == cohort]
                axis.scatter(
                    cohort_part["pc1_internal_sd"],
                    cohort_part["pc2_internal_sd"],
                    s=34,
                    alpha=0.68,
                    color=colour,
                    marker=marker,
                    linewidths=0,
                    label=label,
                )
            information = summary.loc[
                (summary["chain"] == chain)
                & (summary["pooling_normalisation"] == norm)
            ].iloc[0]
            axis.axhline(0, color="#cbd5e1", linewidth=0.8, zorder=0)
            axis.axvline(0, color="#cbd5e1", linewidth=0.8, zorder=0)
            axis.set_title(f"{chain.capitalize()} — {norm.upper()} pooling")
            axis.set_xlabel(
                f"PC1 score ({100 * information.pc1_variance_ratio:.1f}% variance; SD units)"
            )
            axis.set_ylabel(
                f"PC2 score ({100 * information.pc2_variance_ratio:.1f}% variance; SD units)"
            )
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def report() -> None:
    coordinates_path = RUN_ARTIFACTS / "patient_coordinates.csv"
    summary_path = RUN_ARTIFACTS / "summary.csv"
    if coordinates_path.exists() and summary_path.exists():
        coordinates = pd.read_csv(coordinates_path)
        summary = pd.read_csv(summary_path)
    else:
        coordinates, summary = prepare()
    summary = summary.drop(columns=["scope", "status"], errors="ignore")
    protocol.atomic_csv(summary, OUTPUT / "gate_free_pca_summary.csv")
    plot(coordinates, summary)


def self_test() -> None:
    rng = np.random.default_rng(913271)
    controls = rng.normal(loc=-1.0, scale=0.2, size=(10, 4))
    cancers = rng.normal(loc=1.0, scale=0.2, size=(10, 4))
    matrix = np.vstack([controls, cancers])
    labels = np.array([0] * 10 + [1] * 10)
    pca = PCA(n_components=2).fit(matrix)
    scores = pca.transform(matrix)
    signs = orient_components(pca, scores, labels)
    oriented = scores * signs[None, :]
    assert oriented[labels == 1, 0].mean() >= oriented[labels == 0, 0].mean()
    assert signs.shape == (2,)
    print("preliminary gate-free ALICE PCA self-test passed")
