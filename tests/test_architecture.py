from __future__ import annotations

import ast
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from analysis.experiment_01_internal_baseline import run as exp1_run
from analysis.experiment_02_seed_and_attention_normalisation import run as exp2_run
from analysis.experiment_03_external_generalisation import run as exp3_run
from analysis.experiment_04_alice_model import run as exp4_run
from analysis.experiment_04_alice_model import methods as exp4_methods
from analysis.experiment_04_alice_model import prepare as exp4_prepare


REPO = Path(__file__).resolve().parents[1]


def test_each_experiment_has_one_command_line_entry() -> None:
    for directory in sorted((REPO / "analysis").glob("experiment_*")):
        command_files = []
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            imports_argparse = any(
                isinstance(node, ast.Import)
                and any(alias.name == "argparse" for alias in node.names)
                for node in ast.walk(tree)
            )
            if imports_argparse or 'if __name__ == "__main__":' in source:
                command_files.append(path.relative_to(directory).as_posix())
        assert command_files == ["run.py"], directory.name


def test_pipeline_does_not_import_experiments() -> None:
    for path in (REPO / "pipeline").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("analysis.experiment_")
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("analysis.experiment_")
                    for alias in node.names
                )


def test_experiments_01_and_02_do_not_prepare_external_data() -> None:
    state = SimpleNamespace(path=Path("internal.state"))
    with (
        patch.object(exp1_run.workflow, "ensure_internal", return_value=state),
        patch.object(exp1_run.workflow, "ensure_external") as external,
        patch.object(exp1_run.workflow, "guard_consumer"),
        patch.object(exp1_run.prepare, "ensure", return_value=Path("descriptors.state")),
        patch.object(exp1_run.training, "device_from", return_value=torch.device("cpu")),
        patch.object(exp1_run.study, "internal"),
    ):
        exp1_run.main(["--stage", "internal", "--device", "cpu"])
        external.assert_not_called()

    with (
        patch.object(exp2_run.workflow, "ensure_internal", return_value=state),
        patch.object(exp2_run.workflow, "ensure_external") as external,
        patch.object(exp2_run.workflow, "guard_consumer"),
        patch.object(exp2_run.training, "device_from", return_value=torch.device("cpu")),
        patch.object(exp2_run.layer_seed_study, "internal"),
        patch.object(exp2_run.normalizer_study, "internal"),
    ):
        exp2_run.main(["--stage", "internal", "--device", "cpu"])
        external.assert_not_called()


def test_cached_mode_rejects_training_stages() -> None:
    cases = (
        (exp1_run.main, ["--input-mode", "cached", "--stage", "internal"]),
        (exp2_run.main, ["--input-mode", "cached", "--stage", "internal"]),
        (exp3_run.main, ["--input-mode", "cached", "--stage", "prepare"]),
        (exp4_run.main, ["--input-mode", "cached", "--stage", "validation"]),
    )
    for main, argv in cases:
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                main(argv)
            except SystemExit:
                pass
            else:
                raise AssertionError(
                    f"cached training stage was accepted: {main.__module__}"
                )


def test_empty_package_initializers_are_kept() -> None:
    initializers = list((REPO / "analysis").rglob("__init__.py")) + list(
        (REPO / "pipeline").rglob("__init__.py")
    )
    assert initializers
    assert all(path.stat().st_size < 512 for path in initializers)


def test_results_match_the_public_allowlist() -> None:
    allowed = {
        line[1:]
        for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.startswith("!results/") and not line.endswith("/")
    }
    present = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "results").rglob("*")
        if path.is_file()
    }
    assert present == allowed


def test_public_result_tables_are_aggregated() -> None:
    patient_fields = {
        "subject_id",
        "source_patient_id",
        "sample_id",
        "cdr3",
        "cdr3aa",
        "center_cdr3aa",
        "member_cdr3aa",
        "score",
        "logit",
    }
    for path in (REPO / "results").rglob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = set(next(csv.reader(handle)))
        assert not fields & patient_fields, path


def test_no_hard_gate_freeze_requirement_is_independent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (
            patch.object(exp4_prepare, "FREEZE", root / "freeze.json"),
            patch.object(exp4_methods, "MAIN_METHODS", ()),
            patch.object(exp4_methods, "REPRESENTATION_METHODS", ()),
            patch.object(
                exp4_methods,
                "NO_HARD_GATE_CHECKPOINTS",
                root / "no_hard_gate",
            ),
        ):
            exp4_prepare.write_runtime_freeze(
                (),
                require_core_checkpoints=True,
                require_future_checkpoints=True,
            )
            with unittest.TestCase().assertRaises(ValueError):
                exp4_prepare.write_runtime_freeze(
                    (),
                    require_core_checkpoints=True,
                    require_future_checkpoints=True,
                    require_no_hard_gate_checkpoints=True,
                )


class ArchitectureTests(unittest.TestCase):
    test_each_experiment_has_one_command_line_entry = staticmethod(
        test_each_experiment_has_one_command_line_entry
    )
    test_pipeline_does_not_import_experiments = staticmethod(
        test_pipeline_does_not_import_experiments
    )
    test_experiments_01_and_02_do_not_prepare_external_data = staticmethod(
        test_experiments_01_and_02_do_not_prepare_external_data
    )
    test_cached_mode_rejects_training_stages = staticmethod(
        test_cached_mode_rejects_training_stages
    )
    test_empty_package_initializers_are_kept = staticmethod(
        test_empty_package_initializers_are_kept
    )
    test_results_match_the_public_allowlist = staticmethod(
        test_results_match_the_public_allowlist
    )
    test_public_result_tables_are_aggregated = staticmethod(
        test_public_result_tables_are_aggregated
    )
    test_no_hard_gate_freeze_requirement_is_independent = staticmethod(
        test_no_hard_gate_freeze_requirement_is_independent
    )
