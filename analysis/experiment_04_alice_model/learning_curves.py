from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import protocol, reporting


METRIC_INTERVAL = 5
HISTORY_COLUMNS = (
    "experiment_id",
    "method_id",
    "method_label",
    "chain",
    "seed_id",
    "seed_value",
    "fold",
    "epoch",
    "training_auc",
    "held_out_auc",
    "training_bce",
    "held_out_bce",
)


def recorded_epochs(epochs: int) -> tuple[int, ...]:
    return tuple(sorted({1, epochs, *range(METRIC_INTERVAL, epochs + 1, METRIC_INTERVAL)}))


def fold_history_path(
    run_artifacts: Path,
    method_id: str,
    chain: str,
    seed_id: str,
    fold: int,
) -> Path:
    return (
        run_artifacts
        / "training_metrics"
        / method_id
        / chain
        / seed_id
        / f"fold_{fold}.csv"
    )


def checkpoint_binding_path(path: Path) -> Path:
    return Path(f"{path}.checkpoint.json")


def contract_fingerprint(contract: dict) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def history_is_complete(
    path: Path,
    epochs: int,
    identity: dict,
    checkpoint: Path | None = None,
    checkpoint_contract: dict | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, dtype={"seed_value": str})
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    if tuple(frame.columns) != HISTORY_COLUMNS:
        return False
    epoch = pd.to_numeric(frame["epoch"], errors="coerce")
    if (
        not np.isfinite(epoch).all()
        or not np.equal(epoch, np.floor(epoch)).all()
        or epoch.astype(int).tolist() != list(recorded_epochs(epochs))
    ):
        return False
    values = frame[["training_auc", "held_out_auc", "training_bce", "held_out_bce"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        return False
    if not values[["training_auc", "held_out_auc"]].apply(lambda column: column.between(0, 1)).all().all():
        return False
    if (values[["training_bce", "held_out_bce"]] < 0).any().any():
        return False
    identity_matches = all(
        len(frame.loc[frame[key].astype(str) == str(value)]) == len(frame)
        for key, value in identity.items()
    )
    if not identity_matches:
        return False
    if checkpoint is not None:
        binding = checkpoint_binding_path(path)
        if not checkpoint.exists() or not binding.exists():
            return False
        try:
            recorded = json.loads(binding.read_text(encoding="utf-8"))
            if not isinstance(recorded, dict):
                return False
            return (
                recorded.get("checkpoint_sha256") == protocol.sha256(checkpoint)
                and recorded.get("history_sha256") == protocol.sha256(path)
                and checkpoint_contract is not None
                and recorded.get("contract_fingerprint")
                == contract_fingerprint(checkpoint_contract)
            )
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            return False
    return True


def write_fold_history(
    path: Path,
    records: list[dict],
    identity: dict,
    checkpoint: Path,
    checkpoint_contract: dict,
) -> None:
    frame = pd.DataFrame(
        [identity | record for record in records],
        columns=HISTORY_COLUMNS,
    )
    protocol.atomic_csv(frame, path)
    binding = checkpoint_binding_path(path)
    binding.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{binding}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "checkpoint_sha256": protocol.sha256(checkpoint),
                "history_sha256": protocol.sha256(path),
                "contract_fingerprint": contract_fingerprint(checkpoint_contract),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(binding)


def collect_fold_histories(
    run_artifacts: Path,
    jobs: Iterable[dict],
    epochs: int,
) -> pd.DataFrame:
    frames = []
    missing = []
    for job in jobs:
        path = fold_history_path(
            run_artifacts,
            job["method_id"],
            job["chain"],
            job["seed_id"],
            int(job["fold"]),
        )
        checkpoint = Path(job["checkpoint_path"])
        checkpoint_contract = job["checkpoint_contract"]
        identity = {
            key: value
            for key, value in job.items()
            if key not in {"checkpoint_path", "checkpoint_contract"}
        }
        if not history_is_complete(
            path,
            epochs,
            identity,
            checkpoint,
            checkpoint_contract,
        ):
            missing.append(path)
            continue
        frames.append(pd.read_csv(path, dtype={"seed_value": str}))
    if missing:
        raise FileNotFoundError(
            "Training histories are incomplete. Run the internal stage to retrain "
            f"the {len(missing)} affected fold model(s). First missing file: {missing[0]}"
        )
    return pd.concat(frames, ignore_index=True)


def plot_learning_curves(
    history: pd.DataFrame,
    methods: Iterable[tuple[str, str]],
    chains: tuple[str, ...],
    path: Path,
    variant_label: str,
) -> None:
    reporting.configure_plot()
    methods = tuple(methods)
    figure, axes = plt.subplots(
        len(methods),
        len(chains) * 2,
        figsize=(13.2, 10.8),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    figure.suptitle(variant_label, fontsize=13)
    colours = {"training": "#2563eb", "held_out": "#dc2626"}
    for row_index, (method_id, method_label) in enumerate(methods):
        columns = tuple(
            (chain, metric)
            for chain in chains
            for metric in ("auc", "bce")
        )
        for column_index, (chain, metric) in enumerate(columns):
            axis = axes[row_index, column_index]
            part = history.loc[
                (history["method_id"] == method_id) & (history["chain"] == chain)
            ]
            for split, label in (("training", "Fold training"), ("held_out", "Held-out fold")):
                column = f"{split}_{metric}"
                summary = (
                    part.groupby("epoch")[column]
                    .agg(
                        median="median",
                        lower=lambda values: values.quantile(0.25),
                        upper=lambda values: values.quantile(0.75),
                    )
                    .reset_index()
                )
                epoch = summary["epoch"].to_numpy(dtype=float)
                median = summary["median"].to_numpy(dtype=float)
                lower = summary["lower"].to_numpy(dtype=float)
                upper = summary["upper"].to_numpy(dtype=float)
                axis.plot(
                    epoch,
                    median,
                    color=colours[split],
                    linewidth=2.0,
                    label=label,
                )
                axis.fill_between(
                    epoch,
                    lower,
                    upper,
                    color=colours[split],
                    alpha=0.14,
                    linewidth=0,
                )
            if row_index == 0:
                axis.set_title(
                    f"{chain.capitalize()} — {'AUC' if metric == 'auc' else 'BCE'}",
                    fontsize=12,
                )
            axis.grid(color="#e5e7eb", linewidth=0.7)
            if metric == "auc":
                axis.set_ylim(0.0, 1.0)
                ylabel = "AUC"
            else:
                axis.set_ylim(bottom=0.0)
                ylabel = "Class-balanced BCE"
            if column_index == 0:
                axis.set_ylabel(method_label)
            if row_index == len(methods) - 1:
                axis.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Median across seed-fold runs; ribbon: interquartile range",
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
