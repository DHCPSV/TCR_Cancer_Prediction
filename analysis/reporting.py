"""Shared metrics, plotting style, and lightweight reporting checks."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score


METRIC_COLUMNS = [
    "experiment_id",
    "method_id",
    "method_label",
    "chain",
    "seed_id",
    "seed_value",
    "seed_group",
    "split",
    "n",
    "auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
]


def binary_metrics(labels: Iterable[int], scores: Iterable[float]) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=int)
    score = np.asarray(list(scores), dtype=float)
    prediction = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def bootstrap_auc(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    repeats: int = 2000,
    seed: int = 913271,
) -> tuple[float, float]:
    y = np.asarray(list(labels), dtype=int)
    score = np.asarray(list(scores), dtype=float)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Both classes are required for stratified bootstrap")
    generator = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = np.concatenate(
            [generator.choice(indices, len(indices), replace=True) for indices in class_indices]
        )
        values[index] = roc_auc_score(y[sample], score[sample])
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = METRIC_COLUMNS[:8]
    rows = []
    for keys, group in predictions.groupby(group_columns, sort=False):
        rows.append(dict(zip(group_columns, keys)) | binary_metrics(group["label"], group["score"]))
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 8,
            "axes.labelpad": 6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_auc_references(axis: plt.Axes) -> None:
    axis.plot([0, 1], [0, 1], "--", color="#52637a", linewidth=1.2)
    axis.axhline(0.5, color="#9aa8ba", linestyle=":", linewidth=1)
    axis.axvline(0.5, color="#9aa8ba", linestyle=":", linewidth=1)
    axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
