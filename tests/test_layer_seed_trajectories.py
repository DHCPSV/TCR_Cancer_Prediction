from __future__ import annotations

import contextlib
import hashlib
import io
import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from analysis.experiment_02_seed_and_attention_normalisation import (
    layer_seed_study as study,
    report,
    run,
)


def file_state(root: Path) -> dict[str, tuple[str, int, int]]:
    return {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def write_checkpoint_grid() -> None:
    """Use signed, non-unit-length weights so accidental scaling is visible."""
    model = study.reference_model(0)
    for seed_index, seed_id in enumerate(study.TRAJECTORY_SEEDS):
        for fold, epoch in product(study.FOLDS, study.CHECKPOINT_EPOCHS):
            values = (
                torch.arange(64, dtype=torch.float32) * 0.125
                + seed_index * 4.0
                + epoch * 0.0625
                + fold * 0.5
            )
            with torch.no_grad():
                model.attention_score.weight.copy_(values.reshape(1, -1))
                model.classifier.weight.copy_((-2.0 * values - 1.0).reshape(1, -1))
                model.attention_score.bias.fill_(987.0)
                model.classifier.bias.fill_(-456.0)
            study.save_trajectory_checkpoint(
                study.trajectory_checkpoint(seed_id, fold, epoch),
                model,
                seed_id,
                fold,
                epoch,
            )


class LayerSeedTrajectoryTests(unittest.TestCase):
    def test_weights_are_raw_fivefold_means_and_inputs_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(study, "CHECKPOINTS", root):
                write_checkpoint_grid()
                before = file_state(root)
                with patch.object(
                    study,
                    "load_trajectory_checkpoint",
                    wraps=study.load_trajectory_checkpoint,
                ) as loader:
                    frame = study.trajectory_weights()
                self.assertEqual(before, file_state(root))

                expected_grid = set(
                    product(study.TRAJECTORY_SEEDS, study.FOLDS, study.CHECKPOINT_EPOCHS)
                )
                actual_grid = {
                    (call.args[1], call.args[2], call.args[3])
                    for call in loader.call_args_list
                }
                self.assertEqual(actual_grid, expected_grid)
                self.assertEqual(loader.call_count, len(expected_grid))
                for call in loader.call_args_list:
                    self.assertEqual(torch.device(call.args[4]).type, "cpu")

                self.assertEqual(
                    list(frame.columns),
                    ["seed_id", "checkpoint_epoch", "layer", "dimension", "weight"],
                )
                self.assertFalse(
                    frame.duplicated(
                        ["seed_id", "checkpoint_epoch", "layer", "dimension"]
                    ).any()
                )
                expected = []
                mean_fold = float(np.mean(list(study.FOLDS)))
                for seed_index, seed_id in enumerate(study.TRAJECTORY_SEEDS):
                    for epoch, layer, dimension in product(
                        study.CHECKPOINT_EPOCHS, ("attention", "classifier"), range(64)
                    ):
                        value = (
                            dimension * 0.125
                            + seed_index * 4.0
                            + epoch * 0.0625
                            + mean_fold * 0.5
                        )
                        expected.append(
                            {
                                "seed_id": seed_id,
                                "checkpoint_epoch": epoch,
                                "layer": layer,
                                "dimension": dimension,
                                "weight": (
                                    value if layer == "attention" else -2.0 * value - 1.0
                                ),
                            }
                        )
                keys = ["seed_id", "checkpoint_epoch", "layer", "dimension"]
                actual = frame.sort_values(keys).reset_index(drop=True)
                wanted = pd.DataFrame(expected).sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(
                    actual, wanted, check_dtype=False, check_exact=True
                )

    def test_incomplete_or_mismatched_checkpoints_are_not_plotted(self) -> None:
        for failure in ("missing", "wrong_epoch", "wrong_state_shape"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(study, "CHECKPOINTS", root):
                    write_checkpoint_grid()
                    target = study.trajectory_checkpoint("S01", 0, 0)
                    if failure == "missing":
                        target.unlink()
                        error = FileNotFoundError
                    else:
                        payload = torch.load(target, map_location="cpu", weights_only=True)
                        if failure == "wrong_epoch":
                            payload["checkpoint_epoch"] = 999
                            error = ValueError
                        else:
                            payload["model_state"]["attention_score.weight"] = (
                                torch.zeros(1, 63)
                            )
                            error = RuntimeError
                        torch.save(payload, target)
                    before = file_state(root)
                    with self.assertRaises(error):
                        study.trajectory_weights()
                    self.assertEqual(before, file_state(root))

    def test_complete_trajectory_cache_skips_patient_inputs_and_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            metrics = pd.DataFrame({"checkpoint_epoch": [0]})
            predictions = pd.DataFrame({"score": [0.5]})
            metrics.to_csv(runs / "trajectory_fold_metrics.csv", index=False)
            predictions.to_csv(runs / "trajectory_oof.csv", index=False)
            with (
                patch.object(study, "CHECKPOINTS", root / "checkpoints"),
                patch.object(study, "RUN_ARTIFACTS", runs),
            ):
                write_checkpoint_grid()
                before = file_state(root)
                with (
                    patch.object(
                        study, "load_trajectory_results", return_value=(metrics, predictions)
                    ) as load,
                    patch.object(
                        study, "internal_rows", side_effect=AssertionError("patient data read")
                    ),
                    patch.object(
                        study.protocol,
                        "load_tensors",
                        side_effect=AssertionError("embeddings read"),
                    ),
                    patch.object(
                        study,
                        "train_trajectory_fold",
                        side_effect=AssertionError("training called"),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = study.trajectories(torch.device("cpu"))
                load.assert_called_once_with()
                pd.testing.assert_frame_equal(result[0], metrics)
                pd.testing.assert_frame_equal(result[1], predictions)
                self.assertEqual(before, file_state(root))

    def test_small_training_run_saves_every_epoch_and_reuses_strict_checkpoints(self) -> None:
        generator = torch.Generator().manual_seed(321)
        rows = [
            {
                "subject_id": f"P{index}",
                "label": index % 2,
                "cohort": "internal_control" if index % 2 == 0 else "cancer",
                "tcr_count": 3,
            }
            for index in range(6)
        ]
        tensors = {
            row["subject_id"]: torch.randn(3, 64, generator=generator)
            for row in rows
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(study, "CHECKPOINTS", root),
                patch.object(study, "CHECKPOINT_EPOCHS", (0, 1, 2)),
            ):
                first = study.train_trajectory_fold(
                    "S01", 0, rows[:4], rows[4:], tensors, torch.device("cpu")
                )
                self.assertEqual(
                    {item["checkpoint_epoch"] for item in first[0]}, {0, 1, 2}
                )
                self.assertEqual(len(first[1]), 6)
                self.assertEqual(len(list(root.rglob("*.pt"))), 3)
                before = file_state(root)
                with patch.object(
                    torch.optim, "Adam", side_effect=AssertionError("cache retrained")
                ):
                    second = study.train_trajectory_fold(
                        "S01", 0, rows[:4], rows[4:], tensors, torch.device("cpu")
                    )
                self.assertEqual(first, second)
                self.assertEqual(before, file_state(root))

    def test_internal_study_includes_factorisation_and_trajectories(self) -> None:
        events = []
        frame = pd.DataFrame({"result": [1]})
        device = torch.device("cpu")
        with (
            patch.object(
                study,
                "factorization_internal",
                side_effect=lambda value: events.append(("factorisation", value)) or frame,
            ),
            patch.object(
                study,
                "trajectories",
                side_effect=lambda value: events.append(("trajectories", value)),
            ),
        ):
            result = study.internal(device)
        self.assertEqual(events, [("factorisation", device), ("trajectories", device)])
        self.assertIs(result, frame)

    def test_weight_report_reads_once_and_writes_both_paper_figures(self) -> None:
        frame = pd.DataFrame({"weight": [1.0]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(report, "RESULTS", root),
                patch.object(study, "trajectory_weights", return_value=frame) as load,
                patch.object(report, "plot_weight_trajectory") as plot,
            ):
                report.weight_trajectory_report()
            load.assert_called_once_with()
            self.assertEqual(plot.call_count, 2)
            observed = set()
            for call in plot.call_args_list:
                self.assertIs(call.args[0], frame)
                observed.add((call.args[1], call.args[2]))
            self.assertEqual(
                observed,
                {
                    ("attention", root / "figures" / "attention_weight_trajectory.png"),
                    ("classifier", root / "figures" / "classifier_weight_trajectory.png"),
                },
            )

    def test_main_report_includes_weight_trajectories(self) -> None:
        with (
            patch.object(report, "factorization_report"),
            patch.object(report, "normalizer_report"),
            patch.object(report, "load_development_seed_screen"),
            patch.object(report, "plot_development_seed_screen"),
            patch.object(report, "weight_trajectory_report") as weights,
        ):
            report.generate()
        weights.assert_called_once_with()

    def test_raw_and_cached_all_route_to_the_required_stages(self) -> None:
        for mode in ("raw", "cached"):
            with self.subTest(mode=mode), contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                for module, name in (
                    (study, "self_test"),
                    (run.normalizer_study, "self_test"),
                    (report, "seed_screen_self_test"),
                ):
                    stack.enter_context(patch.object(module, name))
                ready = SimpleNamespace(path=Path("internal.state"))
                prepare = stack.enter_context(
                    patch.object(run.workflow, "ensure_internal", return_value=ready)
                )
                external = stack.enter_context(patch.object(run.workflow, "ensure_external"))
                guard = stack.enter_context(patch.object(run.workflow, "guard_consumer"))
                cache = stack.enter_context(
                    patch.object(run.reproduction_cache, "verify_layout")
                )
                device = torch.device("cpu")
                device_from = stack.enter_context(
                    patch.object(run.training, "device_from", return_value=device)
                )
                internal = stack.enter_context(patch.object(study, "internal"))
                trajectories = stack.enter_context(patch.object(study, "trajectories"))
                normalizers = stack.enter_context(
                    patch.object(run.normalizer_study, "internal")
                )
                generate = stack.enter_context(patch.object(report, "generate"))
                diagnostics = stack.enter_context(patch.object(report, "diagnostics"))
                run.main(["--input-mode", mode, "--stage", "all", "--device", "cpu"])
                generate.assert_called_once_with()
                external.assert_not_called()
                if mode == "raw":
                    prepare.assert_called_once_with()
                    guard.assert_called_once()
                    internal.assert_called_once_with(device)
                    normalizers.assert_called_once_with(device)
                    cache.assert_not_called()
                else:
                    cache.assert_called_once_with()
                    diagnostics.assert_called_once_with()
                    prepare.assert_not_called()
                    guard.assert_not_called()
                    device_from.assert_not_called()
                    internal.assert_not_called()
                    trajectories.assert_not_called()
                    normalizers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
