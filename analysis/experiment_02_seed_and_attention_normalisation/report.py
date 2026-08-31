from __future__ import annotations

import importlib.metadata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch

from analysis import protocol, reporting, seed_display
from analysis.experiment_02_seed_and_attention_normalisation import (
    normalizer_study as normalizers,
    layer_seed_study as seeds,
)


EXPERIMENT_ID = seeds.EXPERIMENT_ID
RUN_ARTIFACTS = protocol.repo_path(f"artifacts/runs/{EXPERIMENT_ID}")
RESULTS = protocol.repo_path(f"results/{EXPERIMENT_ID}")
SUPPLEMENTARY = RESULTS / "supplementary"
DEVELOPMENT_SEED_SCREEN = (
    protocol.MANIFESTS / "experiment_02_development_seed_screen.csv"
)
DEVELOPMENT_SEED_SCREEN_SHA256 = (
    "1fb66a8bd890c5fe78be49164de0ded7d4d5b57445d9d45a7c3e1411c588b610"
)

SEED_SCREEN_COLUMNS = [
    "screen_rank",
    "seed_value",
    "auc_mean",
    "auc_min",
    "auc_max",
    "balanced_accuracy_mean",
    "selected_seed_id",
    "seed_group",
    "chain",
    "method_id",
    "epochs",
    "fold_count",
    "split_seed",
    "shuffle_seed",
]

FACTORIZATION_COLORS = {
    "S03": "#b91c1c",
    "S04": "#d97706",
    "S06": "#f59e0b",
    "S08": "#2563eb",
    "S09": "#64748b",
}


def load_development_seed_screen() -> pd.DataFrame:
    """Read and validate the fixed-protocol screen used to choose S01--S09."""
    if protocol.sha256(DEVELOPMENT_SEED_SCREEN) != DEVELOPMENT_SEED_SCREEN_SHA256:
        raise ValueError("Development seed screen checksum mismatch")
    frame = pd.read_csv(
        DEVELOPMENT_SEED_SCREEN,
        dtype={
            "seed_value": str,
            "selected_seed_id": str,
        },
        keep_default_na=False,
    )
    if list(frame.columns) != SEED_SCREEN_COLUMNS:
        raise ValueError("Unexpected columns in the development seed screen")
    if len(frame) != 28 or frame["seed_value"].nunique() != 28:
        raise ValueError("The development seed screen must contain 28 unique seeds")
    if frame["screen_rank"].tolist() != list(range(1, 29)):
        raise ValueError("Development seed ranks must run from 1 to 28")
    if not frame["auc_mean"].is_monotonic_decreasing:
        raise ValueError("Development seed ranks are not ordered by mean AUC")

    metrics = frame[
        ["auc_mean", "auc_min", "auc_max", "balanced_accuracy_mean"]
    ]
    if metrics.isna().any().any() or not metrics.ge(0).all().all():
        raise ValueError("Development seed metrics must be finite and non-negative")
    if not metrics.le(1).all().all():
        raise ValueError("Development seed metrics cannot exceed one")
    if not (
        frame["auc_min"].le(frame["auc_mean"])
        & frame["auc_mean"].le(frame["auc_max"])
    ).all():
        raise ValueError("Mean AUC must lie within the five-fold range")

    constants = {
        "chain": "alpha",
        "method_id": "sceptr_sparsemax",
        "epochs": 50,
        "fold_count": 5,
        "split_seed": protocol.FOLD_SEED,
        "shuffle_seed": protocol.FOLD_SEED,
    }
    for column, expected in constants.items():
        if set(frame[column]) != {expected}:
            raise ValueError(f"Unexpected {column} in the development seed screen")

    expected_selected = {
        str(protocol.SEED_REGISTRY[seed_id]["value"]): (
            seed_id,
            protocol.SEED_REGISTRY[seed_id]["seed_group"],
        )
        for seed_id in normalizers.SEED_IDS
    }
    selected = frame.loc[frame["selected_seed_id"] != ""]
    if len(selected) != 9 or set(selected["seed_value"]) != set(expected_selected):
        raise ValueError("Development screen does not identify exactly S01--S09")
    for row in selected.itertuples(index=False):
        seed_id, seed_group = expected_selected[row.seed_value]
        if (row.selected_seed_id, row.seed_group) != (seed_id, seed_group):
            raise ValueError(f"Incorrect selected-seed metadata for {row.seed_value}")
    if (frame.loc[frame["selected_seed_id"] == "", "seed_group"] != "").any():
        raise ValueError("Unselected development seeds must not have a seed group")
    return frame


def seed_screen_self_test() -> None:
    """Check the small, tracked development-screen manifest."""
    load_development_seed_screen()


def plot_development_seed_screen(frame: pd.DataFrame, path: Path) -> None:
    """Plot all 28 fixed-protocol seeds and highlight the selected panel."""
    reporting.configure_plot()
    figure, axis = plt.subplots(figsize=(11.2, 5.2))
    ranks = frame["screen_rank"].to_numpy(dtype=float)
    axis.vlines(
        ranks,
        frame["auc_min"],
        frame["auc_max"],
        color="#94a3b8",
        linewidth=1.5,
        alpha=0.72,
        zorder=1,
    )
    axis.scatter(
        ranks,
        frame["auc_mean"],
        color="#64748b",
        edgecolor="white",
        linewidth=0.7,
        s=47,
        zorder=2,
    )

    group_handles = []
    for group in ("lucky", "moderate", "ordinary"):
        rows = frame.loc[frame["seed_group"] == group]
        label = seed_display.label(group)
        color = seed_display.GROUP_COLORS[label]
        axis.scatter(
            rows["screen_rank"],
            rows["auc_mean"],
            color=color,
            edgecolor="white",
            linewidth=0.9,
            s=84,
            zorder=3,
        )
        for row in rows.itertuples(index=False):
            axis.annotate(
                row.selected_seed_id,
                (row.screen_rank, row.auc_max),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=color,
                fontsize=9,
                fontweight="bold",
            )
        group_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="white",
                label=f"Selected: {label}",
            )
        )

    axis.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1.2)
    axis.set(
        xlabel="Rank by mean held-out-fold AUC",
        ylabel="Mean held-out-fold AUC (range across 5 folds)",
        xlim=(0.2, 28.8),
        ylim=(0.30, 1.075),
        xticks=[1, 5, 10, 15, 20, 25, 28],
        title="Fixed-protocol alpha-chain seed screen",
    )
    axis.grid(axis="y", alpha=0.18)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor="#64748b",
            markeredgecolor="white",
            label="Other candidate",
        ),
        *group_handles,
    ]
    axis.legend(handles=handles, loc="lower left", frameon=False, ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(figure)


def numbered_seed_group_label(
    seed_id: str,
    seed_ids: tuple[str, ...],
    *,
    multiline: bool = False,
) -> str:
    number = seed_ids.index(seed_id) + 1
    value = str(seeds.seed_value(seed_id))
    if multiline and len(value) > 10:
        value = f"{value[:10]}\n{value[10:]}"
    separator = "\n" if multiline else ": "
    group = seed_display.label(protocol.SEED_REGISTRY[seed_id]["seed_group"])
    base = f"Seed {number}{separator}{value}"
    return f"{base}\n{group}" if multiline else f"{base} — {group}"


def plot_factorization(
    metrics: pd.DataFrame,
    category_summary: pd.DataFrame,
    path: Path,
) -> None:
    order = [seed_id for seed_id, _ in seeds.FACTORIZATION_SEEDS]
    category_by_seed = dict(seeds.FACTORIZATION_SEEDS)
    matrix = metrics.pivot(
        index="attention_seed_id",
        columns="classifier_seed_id",
        values="auc",
    ).loc[order, order]
    category_matrix = category_summary.pivot(
        index="attention_seed_group",
        columns="classifier_seed_group",
        values="pooled_oof_auc_median",
    ).loc[list(seeds.BEHAVIOUR_ORDER), list(seeds.BEHAVIOUR_ORDER)]
    count_matrix = category_summary.pivot(
        index="attention_seed_group",
        columns="classifier_seed_group",
        values="combination_count",
    ).loc[list(seeds.BEHAVIOUR_ORDER), list(seeds.BEHAVIOUR_ORDER)]

    reporting.configure_plot()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.8, 6.6),
        gridspec_kw={"width_ratios": [1.38, 0.82]},
    )
    detailed_axis, summary_axis = axes
    detailed_axis.imshow(matrix, vmin=0.5, vmax=1.0, cmap="viridis", aspect="equal")
    for row in range(len(order)):
        for column in range(len(order)):
            value = matrix.iloc[row, column]
            detailed_axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.72 else "black",
                fontsize=9,
            )
    labels = [
        numbered_seed_group_label(seed_id, tuple(order), multiline=True)
        for seed_id in order
    ]
    detailed_axis.set_xticks(range(len(order)), labels, fontsize=8.5)
    detailed_axis.set_yticks(range(len(order)), labels, fontsize=8.5)
    colours = seed_display.GROUP_COLORS
    for seed_id, label in zip(order, detailed_axis.get_xticklabels()):
        category = category_by_seed[seed_id]
        label.set_bbox(
            {
                "facecolor": colours[seed_display.label(category)],
                "edgecolor": colours[seed_display.label(category)],
                "alpha": 0.16,
                "pad": 2.5,
            }
        )
    for seed_id, label in zip(order, detailed_axis.get_yticklabels()):
        category = category_by_seed[seed_id]
        label.set_bbox(
            {
                "facecolor": colours[seed_display.label(category)],
                "edgecolor": colours[seed_display.label(category)],
                "alpha": 0.16,
                "pad": 2.5,
            }
        )
    detailed_axis.set_xlabel("Classifier initialization seed")
    detailed_axis.set_ylabel("Attention initialization seed")
    detailed_axis.set_title("All 25 seed combinations")

    summary_image = summary_axis.imshow(
        category_matrix,
        vmin=0.5,
        vmax=1.0,
        cmap="viridis",
        aspect="equal",
    )
    for row in range(len(seeds.BEHAVIOUR_ORDER)):
        for column in range(len(seeds.BEHAVIOUR_ORDER)):
            value = category_matrix.iloc[row, column]
            count = int(count_matrix.iloc[row, column])
            summary_axis.text(
                column,
                row,
                f"{value:.3f}\n$n={count}$",
                ha="center",
                va="center",
                color="white" if value < 0.72 else "black",
                fontsize=10,
            )
    categories = list(seeds.BEHAVIOUR_ORDER)
    category_labels = [seed_display.label(category) for category in categories]
    summary_axis.set_xticks(range(len(categories)), category_labels, fontsize=9.5)
    summary_axis.set_yticks(range(len(categories)), category_labels, fontsize=9.5)
    for category, label in zip(categories, summary_axis.get_xticklabels()):
        label.set_color(colours[seed_display.label(category)])
        label.set_fontweight("bold")
    for category, label in zip(categories, summary_axis.get_yticklabels()):
        label.set_color(colours[seed_display.label(category)])
        label.set_fontweight("bold")
    summary_axis.set_xlabel("Classifier seed group")
    summary_axis.set_ylabel("Attention seed group")
    summary_axis.set_title("Median AUC by seed group")

    legend = [
        Patch(
            facecolor=colours[seed_display.label(category)],
            edgecolor=colours[seed_display.label(category)],
            alpha=0.24,
            label=seed_display.label(category),
        )
        for category in categories
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    color_axis = figure.add_axes([0.925, 0.20, 0.018, 0.64])
    figure.colorbar(summary_image, cax=color_axis, label="Patient-level pooled OOF AUC")
    figure.subplots_adjust(
        left=0.12,
        right=0.89,
        top=0.95,
        bottom=0.18,
        wspace=0.34,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def factorization_learning_summary(history: pd.DataFrame, layer: str) -> pd.DataFrame:
    if layer not in {"attention", "classifier"}:
        raise ValueError(f"Unknown factorization layer: {layer}")
    prefix = f"{layer}_"
    keys = [
        f"{prefix}seed_id",
        f"{prefix}seed_value",
        f"{prefix}seed_group",
        "epoch",
    ]
    return history.groupby(keys, as_index=False).agg(
        training_auc_median=("training_auc", "median"),
        held_out_auc_median=("held_out_auc", "median"),
        held_out_auc_q1=("held_out_auc", lambda values: values.quantile(0.25)),
        held_out_auc_q3=("held_out_auc", lambda values: values.quantile(0.75)),
        training_bce_median=("training_bce", "median"),
        held_out_bce_median=("held_out_bce", "median"),
        held_out_bce_q1=("held_out_bce", lambda values: values.quantile(0.25)),
        held_out_bce_q3=("held_out_bce", lambda values: values.quantile(0.75)),
        contributing_runs=("fold", "size"),
    )


def plot_factorization_learning_curves(
    history: pd.DataFrame,
    path: Path,
) -> None:
    reporting.configure_plot()
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), sharex=True)
    seed_ids = tuple(seed_id for seed_id, _ in seeds.FACTORIZATION_SEEDS)
    for column, layer in enumerate(("attention", "classifier")):
        summary = factorization_learning_summary(history, layer)
        id_column = f"{layer}_seed_id"
        for row, metric in enumerate(("auc", "bce")):
            axis = axes[row, column]
            for seed_id in seed_ids:
                group = summary.loc[summary[id_column] == seed_id].sort_values("epoch")
                color = FACTORIZATION_COLORS[seed_id]
                axis.plot(
                    group["epoch"],
                    group[f"training_{metric}_median"],
                    color=color,
                    linestyle="--",
                    linewidth=1.35,
                    alpha=0.55,
                )
                axis.plot(
                    group["epoch"],
                    group[f"held_out_{metric}_median"],
                    color=color,
                    linewidth=2.1,
                )
                axis.fill_between(
                    group["epoch"].to_numpy(dtype=float),
                    group[f"held_out_{metric}_q1"].to_numpy(dtype=float),
                    group[f"held_out_{metric}_q3"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            axis.grid(alpha=0.18)
            if row == 0:
                axis.set_title(f"Grouped by {layer}-layer initialisation")
                axis.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1)
                axis.set_ylim(0, 1)
            else:
                axis.set_xlabel("Training epoch")
    axes[0, 0].set_ylabel("AUC")
    axes[1, 0].set_ylabel("Class-balanced BCE")
    tier_by_seed = dict(seeds.FACTORIZATION_SEEDS)
    seed_handles = [
        Line2D(
            [0],
            [0],
            color=FACTORIZATION_COLORS[seed_id],
            linewidth=2.2,
            label=f"{seeds.seed_value(seed_id)} ({tier_by_seed[seed_id]})",
        )
        for seed_id in seed_ids
    ]
    split_handles = [
        Line2D(
            [0],
            [0],
            color="#111827",
            linewidth=2.1,
            label="Held-out fold median (band: interquartile range)",
        ),
        Line2D(
            [0],
            [0],
            color="#111827",
            linewidth=1.35,
            linestyle="--",
            alpha=0.55,
            label="Training-fold median",
        ),
    ]
    figure.legend(
        handles=seed_handles + split_handles,
        title="Each curve: median across 5 counterpart seeds × 5 folds",
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.96,
        bottom=0.18,
        wspace=0.12,
        hspace=0.20,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def curve_summary(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    return rows.groupby("epoch", as_index=False)[column].agg(mean="mean", sd="std")


def plot_normalizer_learning_curves(
    history: pd.DataFrame,
    path: Path,
) -> None:
    reporting.configure_plot()
    figure, axes = plt.subplots(
        3,
        4,
        figsize=(13.2, 9.2),
        sharex=True,
        constrained_layout=True,
    )
    colors = {"Training": "#2563eb", "Held-out": "#dc2626"}
    columns = (
        ("alpha", "auc", "Alpha — AUC"),
        ("alpha", "bce", "Alpha — BCE"),
        ("beta", "auc", "Beta — AUC"),
        ("beta", "bce", "Beta — BCE"),
    )
    for row_index, (method_id, method_label, _) in enumerate(normalizers.METHODS):
        for column_index, (chain, metric, title) in enumerate(columns):
            axis = axes[row_index, column_index]
            rows = history.loc[
                history["method_id"].eq(method_id) & history["chain"].eq(chain)
            ]
            for value_column, label in (
                (f"training_{metric}", "Training"),
                (f"held_out_{metric}", "Held-out"),
            ):
                summary = curve_summary(rows, value_column)
                x = summary["epoch"].to_numpy(dtype=float)
                mean = summary["mean"].to_numpy(dtype=float)
                sd = summary["sd"].fillna(0).to_numpy(dtype=float)
                axis.plot(x, mean, color=colors[label], linewidth=1.8, label=label)
                axis.fill_between(
                    x,
                    mean - sd,
                    mean + sd,
                    color=colors[label],
                    alpha=0.14,
                    linewidth=0,
                )
            if metric == "auc":
                axis.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1)
                axis.set_ylim(0, 1)
            if row_index == 0:
                axis.set_title(title, fontsize=11)
            if column_index == 0:
                axis.set_ylabel(method_label)
            axis.set_xlim(1, normalizers.EPOCHS)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Mean across seed-fold runs; ribbon: ±1 SD",
        loc="outside upper center",
        ncol=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def plot_normalizers(metrics: pd.DataFrame, path: Path) -> None:
    reporting.configure_plot()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.6),
        sharey=True,
        constrained_layout=True,
    )
    group_markers = {
        "lucky": "o",
        "moderate": "s",
        "ordinary": "^",
    }
    for axis, chain in zip(axes, normalizers.CHAINS):
        chain_rows = metrics.loc[metrics["chain"] == chain]
        for index, (method_id, method_label, _) in enumerate(normalizers.METHODS):
            rows = chain_rows.loc[chain_rows["method_id"] == method_id]
            for group, group_rows in rows.groupby("seed_group"):
                offsets = (
                    (group_rows["seed_id"].str[1:].astype(int) - 5) / 35
                ).to_numpy()
                axis.scatter(
                    index + offsets,
                    group_rows["auc"],
                    marker=group_markers[group],
                    s=54,
                    color=seed_display.GROUP_COLORS[seed_display.label(group)],
                )
            axis.hlines(
                rows["auc"].median(),
                index - 0.22,
                index + 0.22,
                color="#111827",
                linewidth=2,
            )
        axis.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1.2)
        axis.set(
            xticks=range(3),
            xticklabels=[method[1] for method in normalizers.METHODS],
            xlim=(-0.5, 2.5),
            ylim=(0, 1),
        )
        axis.set_title(chain.capitalize())
        axis.set_ylabel("Patient-level pooled OOF AUC")
    axes[1].set_ylabel("")
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            color=seed_display.GROUP_COLORS[seed_display.label(group)],
            label=seed_display.label(group),
        )
        for group, marker in group_markers.items()
    ]
    figure.legend(
        handles=handles,
        title="Seed group",
        loc="outside lower center",
        ncol=3,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def plot_seed_separation(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    sparse = predictions.loc[
        (predictions["method_id"] == "sceptr_sparsemax")
        & (predictions["chain"] == "alpha")
    ].copy()
    auc_by_seed = metrics.loc[
        (metrics["method_id"] == "sceptr_sparsemax")
        & (metrics["chain"] == "alpha"),
        ["seed_id", "auc"],
    ].set_index("seed_id")["auc"]
    seed_order = auc_by_seed.sort_values(ascending=False).index.tolist()
    reporting.configure_plot()
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(11.4, 9.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    randomiser = np.random.default_rng(913271)
    for seed_number, (axis, seed_id) in enumerate(
        zip(axes.ravel(), seed_order),
        start=1,
    ):
        rows = sparse.loc[sparse["seed_id"] == seed_id]
        seed_value, group = normalizers.seed_metadata(seed_id)
        for label, x, color in ((0, 0, "#64748b"), (1, 1, "#dc2626")):
            values = rows.loc[rows["label"] == label, "score"].to_numpy(dtype=float)
            jitter = randomiser.normal(0, 0.045, len(values))
            axis.scatter(
                x + jitter,
                values,
                color=color,
                s=19,
                alpha=0.58,
                linewidths=0,
            )
            axis.hlines(
                np.median(values),
                x - 0.18,
                x + 0.18,
                color="#111827",
                linewidth=2,
            )
        group_label = seed_display.label(group)
        axis.set_title(
            f"Seed {seed_number}: {seed_value} — {group_label}\n"
            f"AUC {auc_by_seed[seed_id]:.3f}",
            fontsize=10,
            color=seed_display.GROUP_COLORS[group_label],
        )
        axis.set_xticks((0, 1), ("Control", "Cancer"))
        axis.set_ylim(-0.03, 1.03)
        axis.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1)
    for axis in axes[:, 0]:
        axis.set_ylabel("Predicted cancer probability")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def factorization_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [
        "attention_seed_id",
        "attention_seed_value",
        "attention_seed_group",
        "classifier_seed_id",
        "classifier_seed_value",
        "classifier_seed_group",
    ]
    for values, group in predictions.groupby(keys, sort=False):
        if len(group) != 169 or group["subject_id"].nunique() != 169:
            raise ValueError(f"Incomplete factorization pair: {values}")
        rows.append(
            dict(zip(keys, values))
            | {
                "split": "internal_oof",
                **reporting.binary_metrics(group["label"], group["score"]),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 25:
        raise ValueError(f"Expected 25 factorization cells, got {len(result)}")
    return result


def factorization_category_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attention_category in seeds.BEHAVIOUR_ORDER:
        for classifier_category in seeds.BEHAVIOUR_ORDER:
            group = metrics.loc[
                (metrics["attention_seed_group"] == attention_category)
                & (metrics["classifier_seed_group"] == classifier_category)
            ]
            if group.empty:
                raise ValueError(
                    "Missing factorization category: "
                    f"{attention_category}, {classifier_category}"
                )
            rows.append(
                {
                    "attention_seed_group": attention_category,
                    "classifier_seed_group": classifier_category,
                    "combination_count": len(group),
                    "pooled_oof_auc_median": group["auc"].median(),
                    "pooled_oof_auc_min": group["auc"].min(),
                    "pooled_oof_auc_max": group["auc"].max(),
                }
            )
    return pd.DataFrame(rows)


def factorization_report() -> None:
    factor_predictions = pd.read_csv(
        RUN_ARTIFACTS / "factorization_predictions.csv",
        dtype={"attention_seed_value": str, "classifier_seed_value": str},
    )
    paired_histories = []
    for attention_id, _ in seeds.FACTORIZATION_SEEDS:
        for classifier_id, _ in seeds.FACTORIZATION_SEEDS:
            for fold in seeds.FOLDS:
                paired_histories.append(
                    seeds.read_factor_history(
                        attention_id,
                        classifier_id,
                        fold,
                        strict_pair=True,
                    )
                )
    paired_history = seeds.validate_factor_history_collection(
        pd.concat(paired_histories, ignore_index=True)
    )
    factor_history = seeds.validate_factor_history_collection(
        pd.read_csv(
            RUN_ARTIFACTS / "factorization_training_metrics.csv",
            dtype={"attention_seed_value": str, "classifier_seed_value": str},
        )
    )
    history_order = ["attention_seed_id", "classifier_seed_id", "fold", "epoch"]
    pd.testing.assert_frame_equal(
        factor_history.sort_values(history_order).reset_index(drop=True),
        paired_history.sort_values(history_order).reset_index(drop=True),
        check_dtype=False,
        obj="Experiment 02 aggregate and paired fold factorization histories",
    )

    category_by_seed = dict(seeds.FACTORIZATION_SEEDS)
    factor_predictions["attention_seed_group"] = factor_predictions[
        "attention_seed_id"
    ].map(category_by_seed)
    factor_predictions["classifier_seed_group"] = factor_predictions[
        "classifier_seed_id"
    ].map(category_by_seed)
    group_columns = ["attention_seed_group", "classifier_seed_group"]
    if factor_predictions[group_columns].isna().any().any():
        raise ValueError("Unknown seed in factorization predictions")

    factor_metrics = factorization_metrics(factor_predictions)
    category_summary = factorization_category_summary(factor_metrics)

    protocol.atomic_csv(factor_metrics, RESULTS / "factorization_pooled_oof.csv")

    plot_factorization(
        factor_metrics,
        category_summary,
        RESULTS / "figures" / "attention_classifier_factorization.png",
    )
    plot_factorization_learning_curves(
        factor_history,
        SUPPLEMENTARY / "figures" / "factorization_learning_curves.png",
    )


def normalizer_report() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(
        RUN_ARTIFACTS / "normalizer_predictions.csv",
        dtype={"seed_value": str},
    )
    aggregate_history_path = RUN_ARTIFACTS / "normalizer_training_metrics.csv"
    if not aggregate_history_path.exists():
        raise FileNotFoundError(
            "Normalizer training histories are missing; rerun the internal stage "
            "before reporting"
        )
    normalizers.validate_predictions(predictions)
    paired_histories = []
    for method in normalizers.METHODS:
        for chain in normalizers.CHAINS:
            for seed_id in normalizers.SEED_IDS:
                for fold in normalizers.FOLDS:
                    paired_histories.append(
                        normalizers.read_history(
                            normalizers.history_path(method[0], chain, seed_id, fold),
                            method,
                            chain,
                            seed_id,
                            fold,
                            strict_pair=True,
                        )
                    )
    paired_history = normalizers.validate_history(
        pd.concat(paired_histories, ignore_index=True)
    )
    history = normalizers.validate_history(
        pd.read_csv(aggregate_history_path, dtype={"seed_value": str})
    )
    history_order = list(normalizers.HISTORY_KEY)
    pd.testing.assert_frame_equal(
        history.sort_values(history_order).reset_index(drop=True),
        paired_history.sort_values(history_order).reset_index(drop=True),
        check_dtype=False,
        obj="Experiment 02 aggregate and paired fold normalizer histories",
    )

    metrics = reporting.metric_table(predictions)
    summary = metrics.groupby(
        ["experiment_id", "method_id", "method_label", "chain", "split"],
        as_index=False,
    ).agg(
        seed_count=("seed_id", "nunique"),
        auc_median=("auc", "median"),
        auc_q1=("auc", lambda values: values.quantile(0.25)),
        auc_q3=("auc", lambda values: values.quantile(0.75)),
        balanced_accuracy_median=("balanced_accuracy", "median"),
    )
    protocol.atomic_csv(
        pd.DataFrame(
            [
                {
                    "torch_version": torch.__version__,
                    "entmax_version": importlib.metadata.version("entmax"),
                    "embedding": "SCEPTR 64D",
                    "epochs": normalizers.EPOCHS,
                    "fold_seed": protocol.FOLD_SEED,
                    "patient_shuffle_seed": normalizers.SHUFFLE_SEED,
                }
            ]
        ),
        RESULTS / "runtime_manifest.csv",
    )
    plot_normalizers(
        metrics,
        RESULTS / "figures" / "normalizer_internal_auc.png",
    )
    plot_seed_separation(
        predictions,
        metrics,
        RESULTS / "figures" / "selected_seed_prediction_separation.png",
    )
    plot_normalizer_learning_curves(
        history,
        SUPPLEMENTARY / "figures" / "normalizer_learning_curves.png",
    )
    protocol.atomic_csv(metrics[reporting.METRIC_COLUMNS], RESULTS / "metrics_by_seed.csv")
    protocol.atomic_csv(summary, RESULTS / "metrics_summary.csv")
    return predictions, metrics, summary


def milestone_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["seed_id", "seed_value", "checkpoint_epoch", "split"]
    for values, group in predictions.groupby(keys, sort=False):
        if len(group) != 169 or group["subject_id"].nunique() != 169:
            raise ValueError(f"Incomplete 300-epoch diagnostic OOF block: {values}")
        rows.append(
            dict(zip(keys, values))
            | reporting.binary_metrics(group["label"], group["score"])
        )
    result = pd.DataFrame(rows)
    expected = len(seeds.TRAJECTORY_SEEDS) * len(seeds.CHECKPOINT_EPOCHS)
    if len(result) != expected:
        raise ValueError(
            f"300-epoch diagnostic has {len(result)} pooled rows; expected {expected}"
        )
    return result


def plot_selected_seed_diagnostics(
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    path: Path,
) -> None:
    reporting.configure_plot()
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    linestyles = {"S01": "-", "S02": "--", "S03": ":"}
    markers = {"S01": "o", "S02": "s", "S03": "^"}
    color = seed_display.GROUP_COLORS["Lucky"]
    for seed_id in seeds.TRAJECTORY_SEEDS:
        pooled = pooled_metrics.loc[pooled_metrics["seed_id"] == seed_id].sort_values(
            "checkpoint_epoch"
        )
        label = numbered_seed_group_label(seed_id, seeds.TRAJECTORY_SEEDS)
        axes[0].plot(
            pooled["checkpoint_epoch"],
            pooled["auc"],
            marker=markers[seed_id],
            linestyle=linestyles[seed_id],
            linewidth=2,
            color=color,
            label=label,
        )
        rows = fold_metrics.loc[fold_metrics["seed_id"] == seed_id]
        summary = rows.groupby("checkpoint_epoch", as_index=False).agg(
            train=("train_balanced_bce", "mean"),
            held_out=("test_balanced_bce", "mean"),
        )
        axes[1].plot(
            summary["checkpoint_epoch"],
            summary["train"],
            color=color,
            linestyle=linestyles[seed_id],
            alpha=0.42,
            linewidth=1.5,
        )
        axes[1].plot(
            summary["checkpoint_epoch"],
            summary["held_out"],
            color=color,
            linestyle=linestyles[seed_id],
            alpha=0.95,
            linewidth=2,
        )
    axes[0].axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1)
    for axis in axes:
        axis.axvline(
            seeds.EPOCHS,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            alpha=0.72,
            label="Main endpoint: 50 epochs" if axis is axes[0] else None,
        )
    axes[0].set(
        xlabel="Checkpoint epoch",
        ylabel="Pooled held-out OOF AUC",
        ylim=(0, 1),
        title="Selected-seed OOF AUC",
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(
        xlabel="Checkpoint epoch",
        ylabel="Class-balanced BCE",
        title="Training-fold (faint) and held-out-fold BCE",
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.16,
        wspace=0.24,
    )
    figure.suptitle(
        "Preliminary 300-epoch diagnostic — three selected-lucky seeds; "
        "main endpoint = 50 epochs",
        y=1.02,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def diagnostics() -> None:
    """Report the optional 300-epoch AUC/BCE diagnostic, without parameters."""
    fold_metrics = pd.read_csv(
        RUN_ARTIFACTS / "trajectory_fold_metrics.csv",
        dtype={"seed_value": str},
    )
    predictions = pd.read_csv(
        RUN_ARTIFACTS / "trajectory_oof.csv",
        dtype={"seed_value": str},
    )
    expected_seeds = set(seeds.TRAJECTORY_SEEDS)
    expected_epochs = set(seeds.CHECKPOINT_EPOCHS)
    if set(fold_metrics["seed_id"]) != expected_seeds:
        raise ValueError("300-epoch diagnostic fold metrics have the wrong seed IDs")
    if set(predictions["seed_id"]) != expected_seeds:
        raise ValueError("300-epoch diagnostic predictions have the wrong seed IDs")
    if set(fold_metrics["checkpoint_epoch"].astype(int)) != expected_epochs:
        raise ValueError("300-epoch diagnostic fold metrics have the wrong checkpoints")
    if set(predictions["checkpoint_epoch"].astype(int)) != expected_epochs:
        raise ValueError("300-epoch diagnostic predictions have the wrong checkpoints")
    if fold_metrics.duplicated(["seed_id", "fold", "checkpoint_epoch"]).any():
        raise ValueError("Duplicate fold metric in 300-epoch diagnostic")
    if predictions.duplicated(
        ["seed_id", "checkpoint_epoch", "subject_id"]
    ).any():
        raise ValueError("Duplicate patient prediction in 300-epoch diagnostic")

    pooled_metrics = milestone_metrics(predictions)
    figures = SUPPLEMENTARY / "figures"
    plot_selected_seed_diagnostics(
        fold_metrics,
        pooled_metrics,
        figures / "selected_seed_300epoch_learning_curves.png",
    )


def generate() -> None:
    """Generate the Experiment 02 tables and figures."""
    factorization_report()
    normalizer_report()
    plot_development_seed_screen(
        load_development_seed_screen(),
        SUPPLEMENTARY / "figures" / "seed_panel_screen.png",
    )
