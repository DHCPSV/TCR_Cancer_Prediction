from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis import protocol, reporting
from analysis.experiment_03_external_generalisation import study


MODELS = ("Sparsemax", "Softmax", "Entmax-1.5")
METHOD_IDS = {
    "Sparsemax": "sceptr_sparsemax",
    "Softmax": "sceptr_softmax",
    "Entmax-1.5": "sceptr_entmax15",
}
GROUP_STYLE = {
    "internal_control": ("Internal control", "#4C78A8", "o"),
    "internal_cancer": ("Internal cancer", "#E45756", "o"),
    "bcg_control": ("BCG control", "#72B7B2", "^"),
    "additional_tracerx_cancer": ("Additional TRACERx PBMC cancer", "#F2A104", "^"),
}
TRANSFER_EXAMPLES = (
    ("Internal separation,\nexternal collapse", "Softmax", "S03"),
    ("Reversed external\nranking", "Entmax-1.5", "S04"),
    ("Weak internal separation,\nexternal alignment", "Sparsemax", "S09"),
)

FIGURES = study.RESULTS / "figures"
SEED_ATLAS = study.RESULTS / "supplementary" / "per_seed_pca_atlas"


def load_transfer_predictions() -> pd.DataFrame:
    frame = pd.read_csv(
        study.RUNS / "transfer_predictions.csv",
        dtype={"seed_value": str},
    )
    if set(frame["method_id"]) != set(study.TRANSFER_METHODS):
        raise ValueError("Transfer predictions contain unexpected methods")
    if set(frame["split"]) != {"internal_oof", "locked_external"}:
        raise ValueError("Transfer predictions contain unexpected splits")
    return frame


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(
        ["experiment_id", "method_id", "method_label", "chain", "split"],
        as_index=False,
    ).agg(
        seed_count=("seed_id", "nunique"),
        auc_median=("auc", "median"),
        auc_q1=("auc", lambda values: values.quantile(0.25)),
        auc_q3=("auc", lambda values: values.quantile(0.75)),
        balanced_accuracy_median=("balanced_accuracy", "median"),
    )


def plot_frozen_transfer(metrics: pd.DataFrame) -> None:
    inside = metrics.loc[metrics["split"] == "internal_oof"]
    outside = metrics.loc[metrics["split"] == "locked_external"]
    points = inside.merge(
        outside,
        on=[
            "experiment_id",
            "method_id",
            "method_label",
            "chain",
            "seed_id",
            "seed_value",
            "seed_group",
        ],
        suffixes=("_internal", "_external"),
        validate="one_to_one",
    )
    colors = {
        "sceptr_softmax": "#dc2626",
        "sceptr_entmax15": "#7c3aed",
        "sceptr_sparsemax": "#2563eb",
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.8),
        constrained_layout=True,
    )
    for axis, chain in zip(axes, study.CHAINS):
        for method_id in study.TRANSFER_METHODS:
            part = points.loc[
                (points["chain"] == chain)
                & (points["method_id"] == method_id)
            ]
            axis.scatter(
                part["auc_internal"],
                part["auc_external"],
                s=52,
                alpha=0.82,
                color=colors[method_id],
                label=study.METHOD_LABELS[method_id],
            )
        reporting.add_auc_references(axis)
        axis.set(
            title=chain.capitalize(),
            xlabel="Frozen internal pooled OOF AUC",
            ylabel="External AUC",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].set_ylabel("")
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURES / "frozen_internal_external_auc.png",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


def generate_transfer() -> pd.DataFrame:
    reporting.configure_plot()
    transfer = load_transfer_predictions()
    metrics = reporting.metric_table(transfer)
    if len(metrics) != (
        len(study.METHODS) * len(study.CHAINS) * len(study.SEED_IDS) * 2
    ):
        raise ValueError("Transfer metric table is incomplete")
    summary = _metric_summary(metrics)
    protocol.atomic_csv(metrics, study.RESULTS / "metrics_by_seed.csv")
    protocol.atomic_csv(summary, study.RESULTS / "metrics_summary.csv")
    plot_frozen_transfer(metrics)
    return metrics


def load_seed_pca() -> pd.DataFrame:
    points = pd.read_csv(
        study.PCA / "per_seed_patient_coordinates.csv",
        dtype={"seed_value": str},
    )
    expected = {
        (model, chain, seed_id)
        for model in MODELS
        for chain in study.CHAINS
        for seed_id in study.SEED_IDS
    }
    observed = set(
        map(
            tuple,
            points[["model", "chain", "seed_id"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    if observed != expected:
        raise ValueError(f"Expected 54 seed-resolved combinations, got {len(observed)}")
    for (model, chain, seed_id, dataset), part in points.groupby(
        ["model", "chain", "seed_id", "dataset"]
    ):
        expected_count = 169 if dataset == "internal" else (
            109 if chain == "alpha" else 110
        )
        if len(part) != expected_count or part["subject_id"].nunique() != expected_count:
            raise ValueError(
                f"Incomplete seed PCA: {model}/{chain}/{seed_id}/{dataset}"
            )
    required = {
        "logit",
        "pc1_raw",
        "pc1_internal_sd",
        "pc1_variance_ratio",
        "pc2_raw",
        "pc2_internal_sd",
        "pc2_variance_ratio",
    }
    if not required.issubset(points.columns):
        raise ValueError(
            f"Seed PCA is missing {sorted(required - set(points.columns))}"
        )
    return points


def summarise_seed_pca(points: pd.DataFrame) -> pd.DataFrame:
    predictions = load_transfer_predictions()
    merged = points.merge(
        predictions[
            ["method_id", "chain", "seed_id", "split", "subject_id", "score"]
        ],
        on=["method_id", "chain", "seed_id", "split", "subject_id"],
        validate="one_to_one",
    )
    maximum_delta = float(
        (merged["probability"] - merged["score"]).abs().max()
    )
    if maximum_delta > 1e-7:
        raise ValueError(
            "Seed-resolved PCA differs from frozen predictions: "
            f"maximum delta {maximum_delta}"
        )

    rows: list[dict] = []
    grouping = [
        "method_id",
        "model",
        "chain",
        "seed_id",
        "seed_value",
        "seed_group",
    ]
    for keys, part in points.groupby(grouping):
        internal = part.loc[part["dataset"] == "internal"]
        external = part.loc[part["dataset"] == "external"]
        rows.append(
            {
                "method_id": keys[0],
                "model": keys[1],
                "chain": keys[2],
                "seed_id": keys[3],
                "seed_value": keys[4],
                "seed_group": keys[5],
                "internal_patients": len(internal),
                "external_patients": len(external),
                "internal_auc": roc_auc_score(
                    internal["label"], internal["probability"]
                ),
                "external_auc": roc_auc_score(
                    external["label"], external["probability"]
                ),
                "internal_pc1_shift_sd": (
                    internal.loc[
                        internal["label"] == 1,
                        "pc1_internal_sd",
                    ].mean()
                    - internal.loc[
                        internal["label"] == 0,
                        "pc1_internal_sd",
                    ].mean()
                ),
                "external_pc1_shift_sd": (
                    external.loc[
                        external["label"] == 1,
                        "pc1_internal_sd",
                    ].mean()
                    - external.loc[
                        external["label"] == 0,
                        "pc1_internal_sd",
                    ].mean()
                ),
                "pc1_variance_ratio": internal["pc1_variance_ratio"].iloc[0],
                "pc2_variance_ratio": internal["pc2_variance_ratio"].iloc[0],
                "pc1_sign_flipped": bool(
                    internal["pc1_sign_flipped"].iloc[0]
                ),
            }
        )
    summary = pd.DataFrame(rows)

    reported = pd.read_csv(
        study.RESULTS / "metrics_by_seed.csv",
        dtype={"seed_value": str},
    )
    reported = reported.loc[
        reported["method_id"].isin(METHOD_IDS.values())
        & reported["split"].isin(["internal_oof", "locked_external"])
    ][["method_id", "chain", "seed_id", "split", "auc"]]
    reported = reported.pivot(
        index=["method_id", "chain", "seed_id"],
        columns="split",
        values="auc",
    ).reset_index()
    reported = reported.rename(
        columns={
            "internal_oof": "internal_auc",
            "locked_external": "external_auc",
        }
    )
    comparison = summary.merge(
        reported,
        on=["method_id", "chain", "seed_id"],
        suffixes=("_pca", "_metrics"),
        validate="one_to_one",
    )
    differences = np.column_stack(
        [
            comparison["internal_auc_pca"] - comparison["internal_auc_metrics"],
            comparison["external_auc_pca"] - comparison["external_auc_metrics"],
        ]
    )
    if np.abs(differences).max() > 1e-12:
        raise ValueError(
            "Seed-resolved PCA AUC annotations differ from the metric table"
        )
    protocol.atomic_csv(summary, study.RESULTS / "per_seed_pca_summary.csv")
    return summary


def group_label(value: str) -> str:
    return value.capitalize()


def wrapped_seed(value: str) -> str:
    return value if len(value) <= 11 else f"{value[:10]}\n{value[10:]}"


def panel_symmetric_limit(
    frame: pd.DataFrame,
    column: str,
    minimum: float,
) -> float:
    return max(minimum, float(np.abs(frame[column]).max()) * 1.08)


def seed_y_limit(points: pd.DataFrame, chain: str) -> float:
    part = points.loc[points["chain"] == chain]
    return max(2.5, float(np.abs(part["pc1_internal_sd"]).max()) * 1.06)


def plot_seed_panel(
    axis,
    frame: pd.DataFrame,
    internal_auc: float,
    external_auc: float,
    pc1_variance_ratio: float,
    pc2_variance_ratio: float,
    axis_limit: float,
    marker_size: float,
    annotation_size: float,
) -> None:
    axis.axhline(0, color="#777777", linestyle=":", linewidth=0.8)
    axis.axvline(0, color="#777777", linestyle=":", linewidth=0.8)
    for group, (_, color, marker) in GROUP_STYLE.items():
        part = frame.loc[frame["group"] == group]
        axis.scatter(
            part["pc2_internal_sd"],
            part["pc1_internal_sd"],
            s=marker_size,
            c=color,
            marker=marker,
            alpha=0.80,
            edgecolors="white",
            linewidths=0.28,
        )
    axis.text(
        0.025,
        0.975,
        f"Internal AUC {internal_auc:.3f} · External AUC {external_auc:.3f}\n"
        f"PC1 {pc1_variance_ratio * 100:.1f}% · PC2 {pc2_variance_ratio * 100:.1f}%",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=annotation_size,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.76,
            "pad": 2.0,
        },
    )
    axis.set(
        xlim=(-axis_limit, axis_limit),
        ylim=(-axis_limit, axis_limit),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(alpha=0.16)


def summary_lookup(
    summary: pd.DataFrame,
) -> dict[tuple[str, str, str], tuple[float, float, float, float]]:
    return {
        (row.model, row.chain, row.seed_id): (
            row.internal_auc,
            row.external_auc,
            row.pc1_variance_ratio,
            row.pc2_variance_ratio,
        )
        for row in summary.itertuples(index=False)
    }


def plot_seed_grid(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    chain: str,
    seeds: tuple[str, ...],
    path: Path,
    dpi: int = 220,
) -> None:
    figure, axes = plt.subplots(
        len(seeds),
        len(MODELS),
        figsize=(13.8, 3.1 * len(seeds) + 1.4),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.115,
        top=0.965,
        hspace=0.38,
        wspace=0.25,
    )
    auc = summary_lookup(summary)
    for row_index, seed_id in enumerate(seeds):
        seed_frame = points.loc[
            (points["chain"] == chain) & (points["seed_id"] == seed_id)
        ]
        seed_value = str(seed_frame["seed_value"].iloc[0])
        group = group_label(str(seed_frame["seed_group"].iloc[0]))
        for column_index, model in enumerate(MODELS):
            axis = axes[row_index, column_index]
            part = seed_frame.loc[seed_frame["model"] == model]
            (
                internal_auc,
                external_auc,
                pc1_variance_ratio,
                pc2_variance_ratio,
            ) = auc[(model, chain, seed_id)]
            axis_limit = max(
                panel_symmetric_limit(part, "pc1_internal_sd", 1.0),
                panel_symmetric_limit(part, "pc2_internal_sd", 1.0),
            )
            plot_seed_panel(
                axis,
                part,
                internal_auc,
                external_auc,
                pc1_variance_ratio,
                pc2_variance_ratio,
                axis_limit,
                marker_size=24,
                annotation_size=9.2,
            )
            if row_index == 0:
                axis.set_title(model, fontsize=12)
            if row_index == len(seeds) - 1:
                axis.set_xlabel("PC2 / internal SD")
            axis.set_ylabel("PC1 / internal SD" if column_index == 0 else "")
        position = axes[row_index, 0].get_position()
        figure.text(
            0.025,
            (position.y0 + position.y1) / 2,
            f"{group} · {seed_id}\n{wrapped_seed(seed_value)}",
            rotation=90,
            va="center",
            ha="center",
            fontsize=10.2,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor="#777777",
            label=label,
            markersize=7,
        )
        for label, color, marker in GROUP_STYLE.values()
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.56, 0.02),
        ncol=4,
        frameon=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def plot_transfer_examples(
    points: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    figure = plt.figure(figsize=(17.2, 13.0))
    grid = figure.add_gridspec(
        3,
        3,
        width_ratios=(0.43, 1.0, 1.0),
        left=0.035,
        right=0.985,
        bottom=0.105,
        top=0.97,
        hspace=0.40,
        wspace=0.30,
    )
    y_limit = seed_y_limit(points, "alpha")
    selected_rows: list[dict] = []
    for row_index, (phenomenon, model, seed_id) in enumerate(TRANSFER_EXAMPLES):
        part = points.loc[
            (points["chain"] == "alpha")
            & (points["model"] == model)
            & (points["seed_id"] == seed_id)
        ]
        metric = summary.loc[
            (summary["chain"] == "alpha")
            & (summary["model"] == model)
            & (summary["seed_id"] == seed_id)
        ].iloc[0]
        seed_value = str(part["seed_value"].iloc[0])
        selected_rows.append(
            {
                "phenomenon": phenomenon.replace("\n", " "),
                "chain": "alpha",
                "model": model,
                "method_id": part["method_id"].iloc[0],
                "seed_id": seed_id,
                "seed_value": seed_value,
                "internal_auc": metric["internal_auc"],
                "external_auc": metric["external_auc"],
                "used_for_model_selection": False,
            }
        )

        label_axis = figure.add_subplot(grid[row_index, 0])
        label_axis.axis("off")
        label_axis.text(0.0, 0.78, phenomenon, fontsize=18, va="center")
        label_axis.text(
            0.0,
            0.32,
            f"{model}\n{seed_id}: {wrapped_seed(seed_value)}\n"
            f"Internal AUC {metric['internal_auc']:.3f}\n"
            f"External AUC {metric['external_auc']:.3f}",
            fontsize=15,
            va="center",
            linespacing=1.28,
        )

        logit_axis = figure.add_subplot(grid[row_index, 1])
        x_limit = panel_symmetric_limit(part, "logit", 0.02)
        logit_axis.axvspan(-x_limit, 0, color="#4C78A8", alpha=0.055)
        logit_axis.axvspan(0, x_limit, color="#E45756", alpha=0.055)
        logit_axis.axvline(0, color="#333333", linestyle="--", linewidth=1.0)
        pc_axis = figure.add_subplot(grid[row_index, 2])
        pc2_limit = panel_symmetric_limit(part, "pc2_internal_sd", 1.0)
        pc_axis.axhline(0, color="#777777", linestyle=":", linewidth=0.8)
        pc_axis.axvline(0, color="#777777", linestyle=":", linewidth=0.8)
        for group, (_, color, marker) in GROUP_STYLE.items():
            group_part = part.loc[part["group"] == group]
            logit_axis.scatter(
                group_part["logit"],
                group_part["pc1_internal_sd"],
                s=25,
                c=color,
                marker=marker,
                alpha=0.80,
                edgecolors="white",
                linewidths=0.30,
            )
            pc_axis.scatter(
                group_part["pc2_internal_sd"],
                group_part["pc1_internal_sd"],
                s=25,
                c=color,
                marker=marker,
                alpha=0.80,
                edgecolors="white",
                linewidths=0.30,
            )
        logit_axis.set(
            xlim=(-x_limit, x_limit),
            ylim=(-y_limit, y_limit),
            xlabel="Logit of frozen per-seed cancer probability",
            ylabel="PC1 / internal SD",
        )
        pc_axis.set(
            xlim=(-pc2_limit, pc2_limit),
            ylim=(-y_limit, y_limit),
            xlabel="PC2 / internal SD",
            ylabel="PC1 / internal SD",
        )
        logit_axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        logit_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        pc_axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        pc_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        pc_axis.text(
            0.025,
            0.975,
            f"PC1 {metric['pc1_variance_ratio'] * 100:.1f}%\n"
            f"PC2 {metric['pc2_variance_ratio'] * 100:.1f}%",
            transform=pc_axis.transAxes,
            va="top",
            ha="left",
            fontsize=14,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
                "pad": 2.0,
            },
        )
        for axis in (logit_axis, pc_axis):
            axis.tick_params(labelsize=14)
            axis.xaxis.label.set_size(16)
            axis.yaxis.label.set_size(16)
            axis.grid(alpha=0.16)
        if row_index == 0:
            logit_axis.set_title("Exact classifier score vs PC1", fontsize=20)
            pc_axis.set_title("Patient-vector PC2 vs PC1", fontsize=20)

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor="#777777",
            label=label,
            markersize=10,
        )
        for label, color, marker in GROUP_STYLE.values()
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=15,
        handletextpad=0.6,
        columnspacing=1.5,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURES / "seed_resolved_pca_transfer_examples.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)
    protocol.atomic_csv(
        pd.DataFrame(selected_rows),
        study.RESULTS / "pca_transfer_example_manifest.csv",
    )


def plot_seed_atlas_pages(
    points: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    SEED_ATLAS.mkdir(parents=True, exist_ok=True)
    for chain in study.CHAINS:
        for page, start in enumerate(
            range(0, len(study.SEED_IDS), 3),
            start=1,
        ):
            seeds = study.SEED_IDS[start : start + 3]
            plot_seed_grid(
                points,
                summary,
                chain,
                seeds,
                SEED_ATLAS / f"{chain}_page_{page}.png",
                dpi=300,
            )


def generate_pca() -> pd.DataFrame:
    reporting.configure_plot()
    points = load_seed_pca()
    summary = summarise_seed_pca(points)
    plot_transfer_examples(points, summary)
    plot_seed_atlas_pages(points, summary)
    return summary


def self_test() -> None:
    combinations = [
        (model, chain, seed)
        for model in MODELS
        for chain in study.CHAINS
        for seed in study.SEED_IDS
    ]
    assert len(combinations) == 54
    assert len(study.CHAINS) * len(range(0, len(study.SEED_IDS), 3)) == 6
    assert len(TRANSFER_EXAMPLES) == 3
    assert len({(model, seed) for _, model, seed in TRANSFER_EXAMPLES}) == 3
    assert "DeepRC" not in MODELS
    print("experiment_03_external_generalisation report self-test passed")
