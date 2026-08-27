"""Manuscript and diagnostic outputs for Experiment 01."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import protocol, reporting, seed_display
from analysis.experiment_01_internal_baseline.methods import METHODS
from analysis.experiment_01_internal_baseline import study


def _predictions() -> pd.DataFrame:
    frame = pd.read_csv(
        study.RUN_ARTIFACTS / "internal_predictions.csv",
        dtype={"seed_value": str},
    )
    study.validate_predictions(frame)
    return frame


def _paired_history() -> pd.DataFrame:
    histories = []
    for method in METHODS:
        for chain in study.CHAINS:
            for seed_id in study.SEED_IDS:
                for fold in study.FOLDS:
                    histories.append(
                        study.read_history(
                            study.history_path(method[0], chain, seed_id, fold),
                            method,
                            chain,
                            seed_id,
                            fold,
                            strict_pair=True,
                        )
                    )
    paired = study.validate_history(pd.concat(histories, ignore_index=True))
    aggregate = study.validate_history(
        pd.read_csv(
            study.RUN_ARTIFACTS / "training_metrics.csv",
            dtype={"seed_value": str},
        )
    )
    order = list(study.HISTORY_KEY)
    pd.testing.assert_frame_equal(
        aggregate.sort_values(order).reset_index(drop=True),
        paired.sort_values(order).reset_index(drop=True),
        check_dtype=False,
        obj="Experiment 01 aggregate and paired fold training histories",
    )
    return aggregate


def _plot_representation_auc(metrics: pd.DataFrame) -> None:
    reporting.configure_plot()
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True, constrained_layout=True)
    colors = {
        seed_id: seed_display.GROUP_COLORS[seed_display.label(seed["seed_group"])]
        for seed_id, seed in study.SEEDS.items()
    }
    labels = [method[1] for method in METHODS]
    for axis, chain in zip(axes, study.CHAINS):
        for index, (method_id, _, _, _) in enumerate(METHODS):
            rows = metrics.loc[(metrics["chain"] == chain) & (metrics["method_id"] == method_id)]
            for offset, seed_id in zip((-0.13, 0.0, 0.13), study.SEED_IDS):
                value = rows.loc[rows["seed_id"] == seed_id, "auc"]
                axis.scatter(
                    index + offset,
                    value,
                    s=58,
                    color=colors[seed_id],
                    label=(
                        f"{study.SEEDS[seed_id]['label']} — "
                        f"{seed_display.label(study.SEEDS[seed_id]['seed_group'])}"
                    )
                    if index == 0
                    else None,
                )
        axis.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1.2)
        axis.set(xlim=(-0.5, len(METHODS) - 0.5), ylim=(0, 1))
        axis.set_xticks(range(len(METHODS)), labels, rotation=24, ha="right")
        axis.set_title(chain.capitalize())
        axis.set_ylabel("Patient-level pooled OOF AUC")
    axes[1].set_ylabel("")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    tier_order = ("Lucky", "Moderate", "Ordinary")
    ordered = sorted(
        zip(handles, legend_labels),
        key=lambda item: tier_order.index(item[1].rsplit(" — ", 1)[-1]),
    )
    figure.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        title="Model initialization",
        loc="outside lower center",
        ncol=3,
    )
    output = study.RESULTS / "figures" / "representation_internal_auc.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def _plot_learning_curves(history: pd.DataFrame) -> None:
    """One Appendix figure combining the conventional AUC and BCE diagnostics."""

    reporting.configure_plot()
    figure, axes = plt.subplots(
        len(METHODS),
        4,
        figsize=(13.2, 11.2),
        sharex=True,
        constrained_layout=True,
    )
    columns = (
        ("alpha", "auc", "Alpha — AUC"),
        ("alpha", "bce", "Alpha — BCE"),
        ("beta", "auc", "Beta — AUC"),
        ("beta", "bce", "Beta — BCE"),
    )
    for row_index, (method_id, method_label, _, _) in enumerate(METHODS):
        for column_index, (chain, metric, title) in enumerate(columns):
            axis = axes[row_index, column_index]
            rows = history.loc[
                (history["method_id"] == method_id) & (history["chain"] == chain)
            ]
            for prefix, label, color in (
                ("training", "Training", "#2563eb"),
                ("held_out", "Held-out", "#dc2626"),
            ):
                summary = rows.groupby("epoch")[f"{prefix}_{metric}"].agg(["mean", "std"])
                x = summary.index.to_numpy(dtype=float)
                mean = summary["mean"].to_numpy(dtype=float)
                spread = summary["std"].fillna(0).to_numpy(dtype=float)
                axis.plot(x, mean, color=color, linewidth=1.5, label=label)
                axis.fill_between(x, mean - spread, mean + spread, color=color, alpha=0.13)
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(method_label)
            axis.set_xlim(1, study.EPOCHS)
            if metric == "auc":
                axis.axhline(0.5, color="#94a3b8", linestyle=":", linewidth=1)
                axis.set_ylim(0, 1)
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
    output = study.RESULTS / "supplementary" / "figures" / "representation_learning_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> pd.DataFrame:
    predictions = _predictions()
    metrics = reporting.metric_table(predictions)
    summary = (
        metrics.groupby(
            ["experiment_id", "method_id", "method_label", "chain", "split"],
            as_index=False,
        )
        .agg(
            seed_count=("seed_id", "nunique"),
            auc_median=("auc", "median"),
            auc_min=("auc", "min"),
            auc_max=("auc", "max"),
            balanced_accuracy_median=("balanced_accuracy", "median"),
        )
    )
    protocol.atomic_csv(metrics, study.RESULTS / "metrics_by_seed.csv")
    protocol.atomic_csv(summary, study.RESULTS / "metrics_summary.csv")
    protocol.atomic_csv(
        pd.DataFrame(
            [
                {
                    "seed_id": seed_id,
                    "figure_label": study.SEEDS[seed_id]["label"],
                    "model_seed": study.SEEDS[seed_id]["value"],
                    "seed_group": study.SEEDS[seed_id]["seed_group"],
                    "fold_seed": protocol.FOLD_SEED,
                    "shuffle_seed": study.SHUFFLE_SEED,
                }
                for seed_id in study.SEED_IDS
            ]
        ),
        study.RESULTS / "seed_manifest.csv",
    )
    _plot_representation_auc(metrics)
    return metrics


def diagnostics() -> None:
    _plot_learning_curves(_paired_history())
