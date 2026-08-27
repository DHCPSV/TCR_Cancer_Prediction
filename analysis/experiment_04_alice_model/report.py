from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import protocol, reporting
from analysis.experiment_04_alice_model import learning_curves, methods, study


def read_predictions(
    run_artifacts: Path,
    method_registry: tuple[methods.Method, ...],
) -> pd.DataFrame:
    method_ids = {method.method_id for method in method_registry}
    parts = []
    for filename in ("internal_predictions.csv", "external_predictions.csv"):
        path = run_artifacts / filename
        frame = pd.read_csv(path, dtype={"seed_value": str})
        frame = frame.loc[frame["method_id"].isin(method_ids)].copy()
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = reporting.metric_table(predictions)
    summary = (
        metrics.groupby(
            ["experiment_id", "method_id", "method_label", "chain", "split"],
            as_index=False,
        )
        .agg(
            seed_count=("seed_id", "nunique"),
            auc_median=("auc", "median"),
            auc_q1=("auc", lambda values: values.quantile(0.25)),
            auc_q3=("auc", lambda values: values.quantile(0.75)),
            balanced_accuracy_median=("balanced_accuracy", "median"),
        )
    )
    return metrics, summary


def paired_auc(metrics: pd.DataFrame) -> pd.DataFrame:
    identity = ["method_id", "method_label", "chain", "seed_id"]
    internal = metrics.loc[metrics["split"] == "internal_oof"].set_index(identity)
    external = metrics.loc[metrics["split"] == "locked_external"].set_index(identity)
    return (
        internal[["auc"]]
        .join(
            external[["auc"]],
            lsuffix="_internal",
            rsuffix="_external",
            how="inner",
        )
        .reset_index()
    )


def auc_figure(
    paired: pd.DataFrame,
    method_registry: tuple[methods.Method, ...],
    path: Path,
    *,
    title: str | None = None,
    chains: tuple[str, ...] = methods.CHAINS,
) -> None:
    reporting.configure_plot()
    palette = ("#059669", "#10b981", "#ea580c", "#f59e0b", "#7c3aed", "#2563eb")
    figure, axes_grid = plt.subplots(
        1,
        len(chains),
        figsize=((9.2, 5.2) if len(chains) == 1 else (12.8, 5.2)),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes_grid.ravel()
    for axis, chain in zip(axes, chains):
        for index, method in enumerate(method_registry):
            part = paired.loc[
                (paired["chain"] == chain)
                & (paired["method_id"] == method.method_id)
            ]
            if part.empty:
                continue
            axis.scatter(
                part["auc_internal"],
                part["auc_external"],
                s=48,
                alpha=0.78,
                color=palette[index % len(palette)],
                label=method.label.replace("No gate + ", ""),
            )
        reporting.add_auc_references(axis)
        axis.set(
            title=chain.capitalize(),
            xlabel="Internal pooled OOF AUC",
            ylabel="Reused external stress-test AUC",
        )
    for axis in axes[1:]:
        axis.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2)
    if title:
        figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def selection_coverage(selection: str, threshold: float | None) -> pd.DataFrame:
    records: list[dict] = []
    for chain in methods.CHAINS:
        frame = study.dataset("internal", chain)
        for row in frame.itertuples(index=False):
            evidence = np.load(protocol.repo_path(row.evidence_file), allow_pickle=False)
            active = evidence > 0 if threshold is None else evidence > threshold
            active_count = int(np.count_nonzero(active))
            records.append(
                {
                    "selection": selection,
                    "subject_id": row.subject_id,
                    "chain": chain,
                    "label": int(row.label),
                    "tcr_count": len(evidence),
                    "active_tcr_count": active_count,
                    "active_tcr_fraction": active_count / len(evidence),
                    "zero_hit": active_count == 0,
                }
            )
    return pd.DataFrame(records)


def retention_report() -> pd.DataFrame:
    selections = (("hard_gate_0.30", methods.HARD_THRESHOLD), ("no_hard_gate", None))
    coverage_path = methods.RUN_ARTIFACTS / "retention_by_subject.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
    else:
        coverage = pd.concat(
            [selection_coverage(selection, threshold) for selection, threshold in selections],
            ignore_index=True,
        )
        protocol.atomic_csv(coverage, coverage_path)
    summary = (
        coverage.groupby(["selection", "chain", "label"], as_index=False)
        .agg(
            subjects=("subject_id", "nunique"),
            active_tcr_fraction_median=("active_tcr_fraction", "median"),
            active_tcr_fraction_q1=("active_tcr_fraction", lambda values: values.quantile(0.25)),
            active_tcr_fraction_q3=("active_tcr_fraction", lambda values: values.quantile(0.75)),
            zero_hit_subjects=("zero_hit", "sum"),
        )
    )
    summary["zero_hit_fraction"] = summary["zero_hit_subjects"] / summary["subjects"]
    protocol.atomic_csv(summary, methods.RESULTS / "retention_summary.csv")

    reporting.configure_plot()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharey="row", constrained_layout=True)
    for row_index, (selection, _) in enumerate(selections):
        for column_index, chain in enumerate(methods.CHAINS):
            axis = axes[row_index, column_index]
            part = coverage.loc[
                (coverage["selection"] == selection) & (coverage["chain"] == chain)
            ]
            values = [
                part.loc[part["label"] == label, "active_tcr_fraction"]
                for label in (0, 1)
            ]
            zero_hits = [
                float(part.loc[part["label"] == label, "zero_hit"].mean())
                for label in (0, 1)
            ]
            axis.boxplot(
                values,
                tick_labels=(
                    f"Control\nZero-hit: {zero_hits[0]:.1%}",
                    f"Cancer\nZero-hit: {zero_hits[1]:.1%}",
                ),
                showfliers=False,
            )
            axis.set_title(f"{chain.capitalize()} — {selection.replace('_', ' ')}")
            axis.set_ylim(bottom=0)
            if column_index == 0:
                axis.set_ylabel("Fraction of TCRs with non-zero pooling weight")
    path = methods.RESULTS / "figures" / "retention_and_zero_hits.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return summary


def learning_curve_report(
    history: pd.DataFrame,
    method_registry: tuple[methods.Method, ...],
    target: Path,
    variant_label: str,
) -> None:
    labels = tuple(
        (method.method_id, method.label.replace("No gate + ", ""))
        for method in method_registry
    )
    learning_curves.plot_learning_curves(
        history,
        labels,
        methods.CHAINS,
        target,
        variant_label,
    )


def hard_gate_report() -> pd.DataFrame:
    predictions = read_predictions(methods.RUN_ARTIFACTS, methods.MAIN_METHODS)
    metrics, summary = metric_tables(predictions)
    protocol.atomic_csv(metrics, methods.RESULTS / "metrics_by_seed.csv")
    protocol.atomic_csv(summary, methods.RESULTS / "metrics_summary.csv")
    paired = paired_auc(metrics)
    protocol.atomic_csv(
        pd.DataFrame(
            [
                {
                    "method_id": method.method_id,
                    "method_label": method.label,
                    "scope": "main_hard_gate",
                    "chains": ",".join(method.chains),
                    "seed_ids": ",".join(method.seed_ids),
                    "representation": method.representation,
                    "pooling_normalisation": method.norm,
                    "classifier_head": method.head,
                    "hard_threshold": method.threshold,
                }
                for method in methods.MAIN_METHODS
            ]
        ),
        methods.RESULTS / "method_manifest.csv",
    )
    auc_figure(
        paired,
        methods.MAIN_METHODS,
        methods.RESULTS / "figures" / "pooling_and_heads.png",
    )
    retention_report()
    learning_curve_report(
        study.main_learning_history(),
        methods.MAIN_METHODS,
        methods.RESULTS / "supplementary" / "figures" / "hard_gate_learning_curves.png",
        "ALICE evidence > 0.30",
    )
    return metrics


def hard_gate_comparison_figure(
    comparison: pd.DataFrame,
    method_registry: tuple[methods.Method, ...],
    path: Path,
) -> None:
    reporting.configure_plot()
    colours = ("#059669", "#10b981", "#ea580c", "#f59e0b")
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), constrained_layout=True)
    for row_index, chain in enumerate(methods.CHAINS):
        for column_index, split in enumerate(("internal_oof", "locked_external")):
            axis = axes[row_index, column_index]
            for method, colour in zip(method_registry, colours):
                part = comparison.loc[
                    (comparison["chain"] == chain)
                    & (comparison["split"] == split)
                    & (comparison["method_id"] == method.method_id)
                ]
                axis.scatter(
                    part["auc_hard_030"],
                    part["auc_no_gate"],
                    s=44,
                    alpha=0.78,
                    color=colour,
                    label=method.label.replace("No gate + ", ""),
                )
            reporting.add_auc_references(axis)
            axis.set(
                title=f"{chain.capitalize()} — {'Internal OOF' if split == 'internal_oof' else 'External stress test'}",
                xlabel="AUC with hard gate (evidence > 0.30)",
                ylabel="AUC without hard gate",
            )
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def no_hard_gate_report() -> pd.DataFrame:
    predictions = read_predictions(
        methods.NO_HARD_GATE_RUN_ARTIFACTS,
        methods.NO_HARD_GATE_METHODS,
    )
    metrics, summary = metric_tables(predictions)
    protocol.atomic_csv(metrics, methods.NO_HARD_GATE_RESULTS / "metrics_by_seed.csv")
    protocol.atomic_csv(summary, methods.NO_HARD_GATE_RESULTS / "metrics_summary.csv")

    hard = pd.read_csv(methods.RESULTS / "metrics_by_seed.csv", dtype={"seed_value": str})
    hard_ids = {
        hard_method.method_id: no_gate_method.method_id
        for hard_method, no_gate_method in zip(
            methods.MAIN_METHODS,
            methods.NO_HARD_GATE_METHODS,
        )
    }
    hard = hard.loc[hard["method_id"].isin(hard_ids)].copy()
    hard["method_id"] = hard["method_id"].map(hard_ids)
    comparison = metrics.merge(
        hard[["method_id", "chain", "seed_id", "split", "auc"]],
        on=["method_id", "chain", "seed_id", "split"],
        suffixes=("_no_gate", "_hard_030"),
        validate="one_to_one",
    )
    comparison["auc_delta_no_gate_minus_hard_030"] = (
        comparison["auc_no_gate"] - comparison["auc_hard_030"]
    )
    protocol.atomic_csv(comparison, methods.NO_HARD_GATE_RESULTS / "paired_vs_hard_030.csv")
    hard_gate_comparison_figure(
        comparison,
        methods.NO_HARD_GATE_METHODS,
        methods.NO_HARD_GATE_RESULTS / "figures" / "paired_no_gate_vs_hard_030.png",
    )
    learning_curve_report(
        study.no_hard_gate_learning_history(),
        methods.NO_HARD_GATE_METHODS,
        methods.NO_HARD_GATE_RESULTS / "supplementary" / "figures" / "no_gate_learning_curves.png",
        "ALICE evidence > 0 (no 0.30 threshold)",
    )
    return metrics


def future_work_report() -> pd.DataFrame:
    predictions = read_predictions(
        methods.RUN_ARTIFACTS / "future_work",
        methods.REPRESENTATION_METHODS,
    )
    metrics, summary = metric_tables(predictions)
    output = methods.RESULTS / "future_work"
    protocol.atomic_csv(metrics, output / "metrics_by_seed.csv")
    protocol.atomic_csv(summary, output / "metrics_summary.csv")
    paired = paired_auc(metrics)
    auc_figure(
        paired,
        methods.REPRESENTATION_METHODS,
        output / "figures" / "representation_dependence.png",
        title="Preliminary representation dependence (partial Alpha-chain study)",
        chains=("alpha",),
    )
    return metrics


def self_test() -> None:
    toy = pd.DataFrame(
        {
            "method_id": ["toy"] * 2,
            "method_label": ["Toy"] * 2,
            "chain": ["alpha"] * 2,
            "seed_id": ["S01"] * 2,
            "split": ["internal_oof", "locked_external"],
            "auc": [0.7, 0.6],
        }
    )
    paired = paired_auc(toy)
    assert len(paired) == 1
    assert paired.iloc[0]["auc_internal"] == 0.7
    assert paired.iloc[0]["auc_external"] == 0.6
