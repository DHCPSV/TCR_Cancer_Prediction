from __future__ import annotations

import json
import os
import unittest
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis import protocol, reporting
from analysis.attention_mil import AttentionMIL
from analysis.experiment_03_external_generalisation import pca as exp3_pca
from analysis.experiment_03_external_generalisation import study as exp3_study
from analysis.experiment_04_alice_model import methods as exp4_methods
from analysis.experiment_04_alice_model.methods import HARD_THRESHOLD, alice_pool
from pipeline import reproduction_cache


REPO = Path(__file__).resolve().parents[1]
REQUIRE_CACHE = os.environ.get("TCR_REQUIRE_REPRODUCTION_CACHE") == "1"
if not reproduction_cache.CONTENTS_MANIFEST.exists():
    if REQUIRE_CACHE:
        raise RuntimeError(
            "TCR_REQUIRE_REPRODUCTION_CACHE=1 but the reproduction cache is absent"
        )
    raise unittest.SkipTest("optional reproduction cache is not installed")


PREDICTION_METRIC_PAIRS = {
    "experiment_01_internal_baseline": (
        ("artifacts/runs/experiment_01_internal_baseline/internal_predictions.csv",),
        "results/experiment_01_internal_baseline/metrics_by_seed.csv",
    ),
    "experiment_02_seed_and_attention_normalisation": (
        (
            "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
            "normalizer_predictions.csv",
        ),
        "results/experiment_02_seed_and_attention_normalisation/metrics_by_seed.csv",
    ),
    "experiment_03_external_generalisation": (
        (
            "artifacts/runs/experiment_03_external_generalisation/"
            "transfer_predictions.csv",
        ),
        "results/experiment_03_external_generalisation/metrics_by_seed.csv",
    ),
    "experiment_04_alice_model": (
        (
            "artifacts/runs/experiment_04_alice_model/internal_predictions.csv",
            "artifacts/runs/experiment_04_alice_model/external_predictions.csv",
        ),
        "results/experiment_04_alice_model/metrics_by_seed.csv",
    ),
    "experiment_04_alice_model_no_hard_gate": (
        (
            "artifacts/runs/experiment_04_alice_model_no_hard_gate/"
            "internal_predictions.csv",
            "artifacts/runs/experiment_04_alice_model_no_hard_gate/"
            "external_predictions.csv",
        ),
        "results/experiment_04_alice_model/no_hard_gate/metrics_by_seed.csv",
    ),
    "experiment_04_alice_model_future_work": (
        (
            "artifacts/runs/experiment_04_alice_model/future_work/"
            "internal_predictions.csv",
            "artifacts/runs/experiment_04_alice_model/future_work/"
            "external_predictions.csv",
        ),
        "results/experiment_04_alice_model/future_work/metrics_by_seed.csv",
    ),
}


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(
        REPO / relative,
        dtype={"seed_value": str},
        keep_default_na=False,
    )


def checkpoint_grid(root: Path) -> set[tuple[str, str, str, int]]:
    records: set[tuple[str, str, str, int]] = set()
    for path in root.rglob("fold_*.pt"):
        relative = path.relative_to(root)
        if len(relative.parts) != 4:
            continue
        method_id, chain, seed_id, filename = relative.parts
        records.add((method_id, chain, seed_id, int(Path(filename).stem[5:])))
    return records


def test_cache_manifest_and_checkpoint_payloads() -> None:
    summary = reproduction_cache.verify(check_checkpoints=True)
    assert summary["file_count"] > 0


def test_cache_contains_reusable_pipeline_inputs() -> None:
    selected = {
        path.relative_to(REPO).as_posix()
        for path in reproduction_cache.selected_files()
    }
    for root in reproduction_cache.REUSABLE_INPUT_DIRS:
        prefix = root.rstrip("/") + "/"
        assert any(path.startswith(prefix) for path in selected), root
    assert not any(path.startswith("data/raw/") for path in selected)
    assert not any(path.startswith("results/") for path in selected)


def test_all_checkpoints_strict_load() -> None:
    exp4_registry = {
        method.method_id: method
        for method in (
            exp4_methods.MAIN_METHODS
            + exp4_methods.NO_HARD_GATE_METHODS
            + exp4_methods.REPRESENTATION_METHODS
        )
    }
    for root_name in reproduction_cache.CHECKPOINT_DIRS:
        for path in (REPO / root_name).rglob("*.pt"):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            state = payload["model_state"]
            if "attention_score.weight" in state:
                dimensions = int(state["attention_score.weight"].shape[1])
                normalizer = str(payload.get("normalizer", "sparsemax"))
                model = AttentionMIL(normalizer, dimensions)
            else:
                method_id = str(payload["method_id"])
                model = exp4_registry[method_id].build()
            model.load_state_dict(state, strict=True)


def test_metrics_recalculate_from_predictions() -> None:
    for name, (prediction_paths, metric_path) in PREDICTION_METRIC_PAIRS.items():
        predictions = pd.concat([read_csv(path) for path in prediction_paths], ignore_index=True)
        reported = read_csv(metric_path)
        assert list(predictions) == protocol.PREDICTION_COLUMNS, name
        assert list(reported) == reporting.METRIC_COLUMNS, name
        recalculated = reporting.metric_table(predictions)
        keys = reporting.METRIC_COLUMNS[:8]
        joined = recalculated.merge(
            reported,
            on=keys,
            suffixes=("_new", "_reported"),
            validate="one_to_one",
        )
        assert len(joined) == len(recalculated) == len(reported), name
        for column in reporting.METRIC_COLUMNS[8:]:
            np.testing.assert_allclose(
                joined[f"{column}_new"].astype(float),
                joined[f"{column}_reported"].astype(float),
                rtol=0.0,
                atol=1e-7,
                equal_nan=True,
            )


def test_factorisation_keeps_both_layer_seeds() -> None:
    frame = read_csv(
        "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
        "factorization_predictions.csv"
    )
    required = {
        "attention_seed_id",
        "attention_seed_value",
        "attention_seed_group",
        "classifier_seed_id",
        "classifier_seed_value",
        "classifier_seed_group",
    }
    assert required.issubset(frame.columns)
    assert "seed_id" not in frame.columns
    assert not frame.duplicated(
        ["attention_seed_id", "classifier_seed_id", "fold", "subject_id"]
    ).any()


def test_exp3_uses_exp2_checkpoints_without_copies() -> None:
    models = read_csv(
        "artifacts/manifests/experiment_03_external_generalisation_models.csv"
    )
    assert len(models) == 270
    assert set(models["source_experiment"]) == {
        "experiment_02_seed_and_attention_normalisation"
    }
    assert models["checkpoint_path"].str.startswith(
        "artifacts/checkpoints/experiment_02_seed_and_attention_normalisation/normalizers/"
    ).all()
    expected = set(
        product(
            exp3_study.TRANSFER_METHODS,
            exp3_study.CHAINS,
            exp3_study.SEED_IDS,
            range(exp3_study.N_FOLDS),
        )
    )
    observed = {
        (row.method_id, row.chain, row.seed_id, int(row.fold))
        for row in models.itertuples(index=False)
    }
    assert observed == expected
    assert not any(exp3_study.CHECKPOINTS.rglob("*.pt"))


def test_exp3_transfer_and_pca_contract() -> None:
    predictions = read_csv(
        "artifacts/runs/experiment_03_external_generalisation/transfer_predictions.csv"
    )
    points = read_csv(
        "artifacts/runs/experiment_03_external_generalisation/pca/"
        "per_seed_patient_coordinates.csv"
    )
    keys = ["method_id", "chain", "seed_id", "split", "subject_id"]
    assert not predictions.duplicated(keys).any()
    assert not points.duplicated(keys).any()
    merged = points.merge(
        predictions[keys + ["score"]],
        on=keys,
        validate="one_to_one",
    )
    np.testing.assert_allclose(
        merged["probability"].astype(float),
        merged["score"].astype(float),
        rtol=0.0,
        atol=1e-7,
    )

    rng = np.random.default_rng(913271)
    internal = np.vstack([rng.normal(-1, 0.2, (8, 4)), rng.normal(1, 0.2, (8, 4))])
    mask = np.asarray([True] * 16 + [False, False])
    labels = np.asarray([0] * 8 + [1] * 8 + [0, 1])
    first = exp3_pca.internal_fit_pca(
        np.vstack([internal, [[10, 10, 10, 10], [12, 8, 9, 11]]]),
        mask,
        labels,
    )
    second = exp3_pca.internal_fit_pca(
        np.vstack([internal, [[-100, 50, 70, 1], [80, -90, 3, -40]]]),
        mask,
        labels,
    )
    np.testing.assert_allclose(first[0][mask], second[0][mask], atol=1e-12)
    np.testing.assert_allclose(first[1], second[1], atol=1e-12)


def test_exp4_checkpoint_grids_and_pooling() -> None:
    main_expected = {
        (method.method_id, chain, seed_id, fold)
        for method in exp4_methods.MAIN_METHODS + exp4_methods.REPRESENTATION_METHODS
        for chain in method.chains
        for seed_id in method.seed_ids
        for fold in range(5)
    }
    no_hard_gate_expected = {
        (method.method_id, chain, seed_id, fold)
        for method in exp4_methods.NO_HARD_GATE_METHODS
        for chain in method.chains
        for seed_id in method.seed_ids
        for fold in range(5)
    }
    assert len(main_expected) == 520
    assert len(no_hard_gate_expected) == 400
    assert checkpoint_grid(exp4_methods.CHECKPOINTS) == main_expected
    assert checkpoint_grid(exp4_methods.NO_HARD_GATE_CHECKPOINTS) == no_hard_gate_expected

    embeddings = torch.tensor([[100.0, 100.0], [3.0, 5.0], [7.0, 11.0]])
    evidence = np.asarray([HARD_THRESHOLD, HARD_THRESHOLD + 1e-4, 0.0])
    for norm in ("l1", "l2"):
        pooled = alice_pool(embeddings, evidence, norm=norm, threshold=HARD_THRESHOLD)
        torch.testing.assert_close(pooled, embeddings[1:2])
        zero = alice_pool(
            embeddings,
            np.asarray([0.0, HARD_THRESHOLD, 0.2]),
            norm=norm,
            threshold=HARD_THRESHOLD,
        )
        torch.testing.assert_close(zero, torch.zeros((1, 2)))


def test_freeze_contracts() -> None:
    exp3 = json.loads(exp3_study.FREEZE.read_text(encoding="utf-8"))
    assert exp3["checkpoint_count"] == 270
    assert exp3["fold_aggregation"] == "mean"
    assert exp3["pca_fit_split"] == "internal_oof"
    assert exp3["external_projection"] == "transform_only"
    atlas_paths = {
        f"results/experiment_03_external_generalisation/"
        f"supplementary/per_seed_pca_atlas/{chain}_page_{page}.png"
        for chain in exp3_study.CHAINS
        for page in range(1, 4)
    }
    assert atlas_paths.issubset(exp3["files"])
    assert not any("per_seed_pc1_atlas" in path for path in exp3["files"])
    exp4 = json.loads(
        (REPO / "artifacts/manifests/alice_runtime_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    assert exp4["experiments"]["experiment_04_alice_model"][
        "checkpoint_count"
    ] == 400
    assert exp4["experiments"]["experiment_04_alice_model_no_hard_gate"][
        "checkpoint_count"
    ] == 400
    assert exp4["future_work"]["representation_dependence"][
        "checkpoint_count"
    ] == 120


class CachedResultTests(unittest.TestCase):
    test_cache_manifest_and_checkpoint_payloads = staticmethod(
        test_cache_manifest_and_checkpoint_payloads
    )
    test_cache_contains_reusable_pipeline_inputs = staticmethod(
        test_cache_contains_reusable_pipeline_inputs
    )
    test_all_checkpoints_strict_load = staticmethod(test_all_checkpoints_strict_load)
    test_metrics_recalculate_from_predictions = staticmethod(
        test_metrics_recalculate_from_predictions
    )
    test_factorisation_keeps_both_layer_seeds = staticmethod(
        test_factorisation_keeps_both_layer_seeds
    )
    test_exp3_uses_exp2_checkpoints_without_copies = staticmethod(
        test_exp3_uses_exp2_checkpoints_without_copies
    )
    test_exp3_transfer_and_pca_contract = staticmethod(
        test_exp3_transfer_and_pca_contract
    )
    test_exp4_checkpoint_grids_and_pooling = staticmethod(
        test_exp4_checkpoint_grids_and_pooling
    )
    test_freeze_contracts = staticmethod(test_freeze_contracts)
