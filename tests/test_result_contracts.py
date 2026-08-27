from __future__ import annotations

import ast
import json
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from analysis import protocol, reporting
from analysis.experiment_01_internal_baseline import run as exp1_run
from analysis.experiment_02_seed_and_attention_normalisation import run as exp2_run
from analysis.experiment_03_external_generalisation import geometry as exp3_geometry
from analysis.experiment_03_external_generalisation import study as exp3_study
from analysis.experiment_04_alice_model import core as exp4_core
from analysis.experiment_04_alice_model import run as exp4_run
from analysis.experiment_04_alice_model import no_hard_gate, study as exp4_study
from analysis.experiment_04_alice_model.methods import HARD_THRESHOLD, alice_pool


REPO = Path(__file__).resolve().parents[1]

PREDICTION_METRIC_PAIRS = {
    "experiment_01_internal_baseline": (
        ("artifacts/runs/experiment_01_internal_baseline/internal_predictions.csv",),
        "results/experiment_01_internal_baseline/metrics_by_seed.csv",
    ),
    "experiment_02_seed_and_attention_normalisation": (
        (
            "artifacts/runs/experiment_02_seed_and_attention_normalisation/"
            "internal_predictions.csv",
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

RESULT_WHITELIST = {
    "experiment_01_internal_baseline": {
        "metrics_by_seed.csv",
        "metrics_summary.csv",
        "seed_manifest.csv",
        "figures/representation_internal_auc.png",
        "supplementary/figures/representation_learning_curves.png",
    },
    "experiment_02_seed_and_attention_normalisation": {
        "factorization_pooled_oof.csv",
        "metrics_by_seed.csv",
        "metrics_summary.csv",
        "runtime_manifest.csv",
        "figures/attention_classifier_factorization.png",
        "figures/normalizer_internal_auc.png",
        "figures/selected_seed_prediction_separation.png",
        "supplementary/figures/factorization_learning_curves.png",
        "supplementary/figures/normalizer_learning_curves.png",
        "supplementary/figures/selected_seed_300epoch_learning_curves.png",
    },
    "experiment_03_external_generalisation": {
        "metrics_by_seed.csv",
        "metrics_summary.csv",
        "pca_transfer_example_manifest.csv",
        "per_seed_pca_summary.csv",
        "figures/frozen_internal_external_auc.png",
        "figures/seed_resolved_pca_transfer_examples.png",
        *{
            f"supplementary/per_seed_pc1_atlas/{chain}_page_{page}.png"
            for chain in ("alpha", "beta")
            for page in range(1, 4)
        },
    },
    "experiment_04_alice_model": {
        "metrics_by_seed.csv",
        "metrics_summary.csv",
        "method_manifest.csv",
        "retention_summary.csv",
        "figures/pooling_and_heads.png",
        "figures/retention_and_zero_hits.png",
        "figures/family_contributions.png",
        "figures/vdjdb_validation.png",
        "tables/family_analysis_manifest.csv",
        "tables/vdjdb_validation_summary.csv",
        "supplementary/figures/hard_gate_learning_curves.png",
        "no_hard_gate/metrics_by_seed.csv",
        "no_hard_gate/metrics_summary.csv",
        "no_hard_gate/paired_vs_hard_030.csv",
        "no_hard_gate/figures/paired_no_gate_vs_hard_030.png",
        "no_hard_gate/supplementary/figures/no_gate_learning_curves.png",
        "future_work/metrics_by_seed.csv",
        "future_work/metrics_summary.csv",
        "future_work/figures/representation_dependence.png",
        "future_work/gate_free_pca/gate_free_pca_summary.csv",
        "future_work/gate_free_pca/figures/gate_free_alice_patient_vector_pca.png",
    },
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
            raise AssertionError(f"Unexpected checkpoint layout: {path}")
        method_id, chain, seed_id, filename = relative.parts
        records.add((method_id, chain, seed_id, int(Path(filename).stem[5:])))
    return records


class ResultSchemaAndMetricParityTest(unittest.TestCase):
    def test_prediction_and_metric_schemas_and_recalculation(self) -> None:
        metric_keys = reporting.METRIC_COLUMNS[:8]
        value_columns = reporting.METRIC_COLUMNS[8:]
        for name, (prediction_paths, metric_path) in PREDICTION_METRIC_PAIRS.items():
            with self.subTest(name=name):
                predictions = pd.concat(
                    [read_csv(path) for path in prediction_paths],
                    ignore_index=True,
                )
                reported = read_csv(metric_path)
                self.assertEqual(list(predictions), protocol.PREDICTION_COLUMNS)
                self.assertEqual(list(reported), reporting.METRIC_COLUMNS)
                self.assertFalse(predictions.empty)
                self.assertFalse(reported.empty)
                self.assertFalse(
                    predictions.duplicated([*metric_keys, "subject_id"]).any()
                )
                self.assertEqual(set(predictions["label"].astype(int)), {0, 1})
                self.assertTrue(
                    np.isfinite(predictions[["score", "tcr_count"]].astype(float)).all().all()
                )
                self.assertTrue(predictions["score"].astype(float).between(0, 1).all())
                self.assertTrue((predictions["tcr_count"].astype(int) > 0).all())

                recalculated = reporting.metric_table(predictions)
                self.assertFalse(recalculated.duplicated(metric_keys).any())
                self.assertFalse(reported.duplicated(metric_keys).any())
                joined = recalculated.merge(
                    reported,
                    on=metric_keys,
                    suffixes=("_recalculated", "_reported"),
                    validate="one_to_one",
                )
                self.assertEqual(len(joined), len(recalculated))
                self.assertEqual(len(joined), len(reported))
                for column in value_columns:
                    actual = joined[f"{column}_recalculated"].astype(float)
                    expected = joined[f"{column}_reported"].astype(float)
                    difference = np.abs(actual - expected)
                    self.assertTrue(
                        np.allclose(actual, expected, rtol=0.0, atol=1e-7, equal_nan=True),
                        f"{name}/{column}: maximum difference {difference.max()}",
                    )

    def test_experiments_01_and_02_are_internal_only(self) -> None:
        internal_state = SimpleNamespace(path=Path("internal.state"))
        with (
            patch.object(exp1_run.workflow, "ensure_internal", return_value=internal_state) as internal,
            patch.object(exp1_run.workflow, "ensure_external") as external,
            patch.object(exp1_run.workflow, "guard_consumer"),
            patch.object(exp1_run.prepare, "ensure", return_value=Path("descriptors.state")),
            patch.object(exp1_run.training, "device_from", return_value=torch.device("cpu")),
            patch.object(exp1_run.study, "internal"),
        ):
            exp1_run.main(["--stage", "internal", "--device", "cpu"])
            internal.assert_called_once_with()
            external.assert_not_called()

        with (
            patch.object(exp2_run.workflow, "ensure_internal", return_value=internal_state) as internal,
            patch.object(exp2_run.workflow, "ensure_external") as external,
            patch.object(exp2_run.workflow, "guard_consumer"),
            patch.object(exp2_run.training, "device_from", return_value=torch.device("cpu")),
            patch.object(exp2_run.seed_mechanism, "internal"),
            patch.object(exp2_run.normalizer_study, "internal"),
            patch.object(exp2_run.normalizer_study, "write_internal_predictions"),
        ):
            exp2_run.main(["--stage", "internal", "--device", "cpu"])
            internal.assert_called_once_with()
            external.assert_not_called()

        for experiment_id in (
            "experiment_01_internal_baseline",
            "experiment_02_seed_and_attention_normalisation",
        ):
            predictions = read_csv(PREDICTION_METRIC_PAIRS[experiment_id][0][0])
            self.assertEqual(set(predictions["split"]), {"internal_oof"})


class RuntimeDiscoveryTest(unittest.TestCase):
    def test_exp4_discovers_supported_windows_rscript(self) -> None:
        expected = str(exp4_run.WINDOWS_RSCRIPT)
        with (
            patch.object(exp4_run.shutil, "which", return_value=None),
            patch.object(exp4_run.sys, "platform", "win32"),
            patch.object(exp4_run.Path, "is_file", return_value=True),
        ):
            self.assertEqual(exp4_run.discover_rscript(), expected)

    def test_exp4_prefers_rscript_on_path(self) -> None:
        expected = r"D:\R\bin\Rscript.exe"
        with patch.object(exp4_run.shutil, "which", return_value=expected):
            self.assertEqual(exp4_run.discover_rscript(), expected)


class Experiment03FrozenTransferContractTest(unittest.TestCase):
    def test_two_inputs_and_complete_270_model_freeze(self) -> None:
        inputs = read_csv(
            "artifacts/manifests/experiment_03_external_generalisation_inputs.csv"
        )
        models = read_csv(
            "artifacts/manifests/experiment_03_external_generalisation_models.csv"
        )
        self.assertEqual(
            set(inputs["artifact_id"]),
            {"baseline_internal_predictions", "baseline_external_predictions"},
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(models), 270)
        expected = set(
            product(
                exp3_study.TRANSFER_METHODS,
                exp3_study.CHAINS,
                exp3_study.SEED_IDS,
                range(exp3_study.N_FOLDS),
            )
        )
        actual = {
            (row.method_id, row.chain, row.seed_id, int(row.fold))
            for row in models.itertuples(index=False)
        }
        self.assertEqual(actual, expected)
        self.assertEqual(
            set(models["source_experiment"]),
            {"experiment_02_seed_and_attention_normalisation"},
        )
        self.assertTrue(
            models["frozen_path"].str.startswith(
                "artifacts/checkpoints/experiment_03_external_generalisation/"
                "frozen_predecessors/experiment_02_seed_and_attention_normalisation/"
            ).all()
        )
        for row in pd.concat(
            [
                inputs[["frozen_path", "bytes", "sha256"]],
                models[["frozen_path", "bytes", "sha256"]],
            ],
            ignore_index=True,
        ).itertuples(index=False):
            path = REPO / row.frozen_path
            self.assertEqual(path.stat().st_size, int(row.bytes))
            self.assertEqual(protocol.sha256(path), row.sha256)

        freeze = json.loads(exp3_study.FREEZE.read_text(encoding="utf-8"))
        self.assertEqual(freeze["schema_version"], 2)
        self.assertFalse(freeze["external_selection"])
        self.assertEqual(freeze["checkpoint_count"], 270)
        self.assertEqual(freeze["fold_count"], 5)
        self.assertEqual(freeze["fold_aggregation"], "mean")
        self.assertEqual(freeze["pca_fit_split"], "internal_oof")
        self.assertEqual(freeze["external_projection"], "transform_only")
        for relative, expected_hash in freeze["files"].items():
            self.assertEqual(protocol.sha256(REPO / relative), expected_hash)

    def test_transfer_and_seed_resolved_pca_contract(self) -> None:
        predictions = read_csv(
            "artifacts/runs/experiment_03_external_generalisation/"
            "transfer_predictions.csv"
        )
        points = read_csv(
            "artifacts/runs/experiment_03_external_generalisation/geometry/"
            "per_seed_patient_coordinates.csv"
        )
        keys = ["method_id", "chain", "seed_id", "split", "subject_id"]
        self.assertFalse(predictions.duplicated(keys).any())
        self.assertFalse(points.duplicated(keys).any())
        self.assertEqual(set(predictions["split"]), {"internal_oof", "locked_external"})
        self.assertEqual(
            set(zip(points["dataset"], points["split"])),
            {("internal", "internal_oof"), ("external", "locked_external")},
        )
        self.assertTrue(
            points.loc[points["dataset"] == "internal", "fold"].astype(int).between(0, 4).all()
        )
        self.assertEqual(
            set(points.loc[points["dataset"] == "external", "fold"].astype(int)),
            {-1},
        )

        expected_sizes = {"alpha": (169, 109), "beta": (169, 110)}
        for (method_id, chain, seed_id), group in points.groupby(
            ["method_id", "chain", "seed_id"]
        ):
            internal_n, external_n = expected_sizes[chain]
            counts = group.groupby("dataset")["subject_id"].nunique().to_dict()
            self.assertEqual(counts, {"external": external_n, "internal": internal_n})
            internal = group.loc[group["dataset"] == "internal"]
            cancer_pc1 = internal.loc[internal["label"].astype(int) == 1, "pc1_raw"].astype(float)
            control_pc1 = internal.loc[internal["label"].astype(int) == 0, "pc1_raw"].astype(float)
            self.assertGreaterEqual(cancer_pc1.mean() + 1e-12, control_pc1.mean())

        merged = points.merge(
            predictions[keys + ["score"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        self.assertEqual(len(merged), len(points))
        np.testing.assert_allclose(
            merged["probability"].astype(float),
            merged["score"].astype(float),
            rtol=0.0,
            atol=1e-7,
        )
        probability = merged["probability"].astype(float).clip(1e-6, 1 - 1e-6)
        np.testing.assert_allclose(
            merged["logit"].astype(float),
            np.log(probability / (1 - probability)),
            rtol=0.0,
            atol=1e-7,
        )

        rng = np.random.default_rng(913271)
        internal = np.vstack(
            [rng.normal(-1, 0.2, (8, 4)), rng.normal(1, 0.2, (8, 4))]
        )
        labels = np.asarray([0] * 8 + [1] * 8 + [0, 1])
        mask = np.asarray([True] * 16 + [False, False])
        first = exp3_geometry.internal_fit_pca(
            np.vstack([internal, [[10, 10, 10, 10], [12, 8, 9, 11]]]),
            mask,
            labels,
        )
        second = exp3_geometry.internal_fit_pca(
            np.vstack([internal, [[-100, 50, 70, 1], [80, -90, 3, -40]]]),
            mask,
            labels,
        )
        for first_value, second_value in zip(first[:3], second[:3]):
            first_internal = first_value[mask] if first_value.ndim == 2 else first_value
            second_internal = second_value[mask] if second_value.ndim == 2 else second_value
            np.testing.assert_allclose(first_internal, second_internal, atol=1e-12)
        self.assertEqual(first[3], second[3])

        summary = read_csv(
            "results/experiment_03_external_generalisation/per_seed_pca_summary.csv"
        )
        self.assertEqual(len(summary), 3 * 2 * 9)
        self.assertTrue((summary["internal_patients"].astype(int) == 169).all())
        for chain, external_n in (("alpha", 109), ("beta", 110)):
            chain_rows = summary.loc[summary["chain"] == chain]
            self.assertTrue((chain_rows["external_patients"].astype(int) == external_n).all())
        self.assertLessEqual(summary["internal_auc_delta"].astype(float).abs().max(), 1e-7)
        self.assertLessEqual(summary["external_auc_delta"].astype(float).abs().max(), 1e-7)
        examples = read_csv(
            "results/experiment_03_external_generalisation/"
            "pca_transfer_example_manifest.csv"
        )
        self.assertEqual(
            set(examples["used_for_model_selection"].astype(str).str.lower()),
            {"false"},
        )


class Experiment04RegistryAndFreezeContractTest(unittest.TestCase):
    def test_core_no_gate_and_future_work_checkpoint_grids(self) -> None:
        core_ids = {method.method_id for method in exp4_core.METHODS}
        no_gate_ids = {method.method_id for method in no_hard_gate.METHODS}
        future_ids = {method.method_id for method in exp4_study.FUTURE_WORK_METHODS}
        self.assertEqual(
            core_ids,
            {"alice_l1_linear", "alice_l1_mlp", "alice_l2_linear", "alice_l2_mlp"},
        )
        self.assertEqual(
            no_gate_ids,
            {
                "no_gate_l1_linear",
                "no_gate_l1_mlp",
                "no_gate_l2_linear",
                "no_gate_l2_mlp",
            },
        )
        self.assertEqual(len(future_ids), 8)

        core_expected = {
            (method.method_id, chain, seed_id, fold)
            for method in exp4_core.METHODS
            for chain in method.chains
            for seed_id in method.seed_ids
            for fold in range(5)
        }
        future_expected = {
            (method.method_id, chain, seed_id, fold)
            for method in exp4_study.FUTURE_WORK_METHODS
            for chain in method.chains
            for seed_id in method.seed_ids
            for fold in range(5)
        }
        no_gate_expected = {
            (method.method_id, chain, seed_id, fold)
            for method in no_hard_gate.METHODS
            for chain in method.chains
            for seed_id in method.seed_ids
            for fold in range(5)
        }
        main_grid = checkpoint_grid(exp4_core.CHECKPOINTS)
        no_gate_grid = checkpoint_grid(no_hard_gate.CHECKPOINTS)
        self.assertEqual(len(core_expected), 400)
        self.assertEqual(len(future_expected), 120)
        self.assertEqual(len(no_gate_expected), 400)
        self.assertEqual(main_grid, core_expected | future_expected)
        self.assertEqual(no_gate_grid, no_gate_expected)
        self.assertEqual(
            {path.name for path in exp4_core.CHECKPOINTS.iterdir() if path.is_dir()},
            core_ids | future_ids,
        )
        self.assertFalse(
            any(
                "control" in method_id or "attention" in method_id
                for method_id in core_ids | future_ids | no_gate_ids
            )
        )

    def test_frozen_alice_assets_and_checkpoint_digests(self) -> None:
        freeze = json.loads(protocol.ALICE_FREEZE.read_text(encoding="utf-8"))
        self.assertEqual(freeze["schema_version"], 2)
        experiment = freeze["experiments"][exp4_core.EXPERIMENT_ID]
        self.assertNotIn("code_files", experiment)
        for group in ("model_code_files", "analysis_code_files"):
            self.assertTrue(experiment[group])
            for relative, expected_hash in experiment[group].items():
                self.assertEqual(protocol.sha256(REPO / relative), expected_hash)
        protocol.verify_alice_freeze("internal", exp4_core.EXPERIMENT_ID)
        protocol.verify_alice_freeze("external", exp4_core.EXPERIMENT_ID)
        exp4_study.verify_future_work_checkpoint_freeze()

    def test_hard_gate_is_strict_and_zero_hits_produce_zero_vector(self) -> None:
        embeddings = torch.tensor(
            [[100.0, 100.0], [3.0, 5.0], [7.0, 11.0]],
            dtype=torch.float32,
        )
        evidence = np.asarray([HARD_THRESHOLD, HARD_THRESHOLD + 1e-4, 0.0])
        for norm in ("l1", "l2"):
            pooled = alice_pool(
                embeddings,
                evidence,
                norm=norm,
                threshold=HARD_THRESHOLD,
            )
            torch.testing.assert_close(pooled, embeddings[1:2])
            zero = alice_pool(
                embeddings,
                np.asarray([0.0, HARD_THRESHOLD, 0.2]),
                norm=norm,
                threshold=HARD_THRESHOLD,
            )
            torch.testing.assert_close(zero, torch.zeros((1, 2)))


class RepositoryArchitectureContractTest(unittest.TestCase):
    def test_each_experiment_has_run_py_as_its_only_cli(self) -> None:
        for directory in sorted((REPO / "analysis").glob("experiment_*")):
            cli_files = []
            for path in directory.rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                imports_argparse = any(
                    (
                        isinstance(node, ast.Import)
                        and any(alias.name == "argparse" for alias in node.names)
                    )
                    or (
                        isinstance(node, ast.ImportFrom)
                        and node.module == "argparse"
                    )
                    for node in ast.walk(tree)
                )
                has_main_guard = 'if __name__ == "__main__":' in source
                if imports_argparse or has_main_guard:
                    cli_files.append(path.relative_to(directory).as_posix())
            self.assertEqual(cli_files, ["run.py"], directory.name)

    def test_pipeline_does_not_import_experiments(self) -> None:
        forbidden = []
        for path in (REPO / "pipeline").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                if any(name.startswith("analysis.experiment_") for name in names):
                    forbidden.append(path.relative_to(REPO).as_posix())
        self.assertEqual(forbidden, [])

    def test_formal_results_match_the_explicit_whitelist(self) -> None:
        self.assertEqual(
            {path.name for path in (REPO / "results").iterdir() if path.is_dir()},
            set(RESULT_WHITELIST),
        )
        for experiment_id, expected in RESULT_WHITELIST.items():
            root = REPO / "results" / experiment_id
            actual = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(actual, expected, experiment_id)


if __name__ == "__main__":
    unittest.main()
