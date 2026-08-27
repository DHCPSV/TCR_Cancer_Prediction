from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

for module_name in (
    "pipeline.build_manifests",
    "pipeline.embed_sceptr",
    "pipeline.prepare_data",
):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

from pipeline import workflow


class CacheGuardTest(unittest.TestCase):
    def test_changed_upstream_archives_generated_roots_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            state = repo / "artifacts" / "provenance" / "data" / "internal.json"
            checkpoints = repo / "artifacts" / "checkpoints" / "experiment_test"
            runs = repo / "artifacts" / "runs" / "experiment_test"
            results = repo / "results" / "experiment_test"
            state.parent.mkdir(parents=True)
            state.write_text("upstream-v1", encoding="utf-8")
            with patch.multiple(
                workflow,
                REPO=repo,
                MANIFESTS=repo / "artifacts" / "manifests",
                STATE_ROOT=repo / "artifacts" / "provenance",
                STALE_CACHE_ROOT=repo / "backup" / "stale_cache",
            ), redirect_stdout(StringIO()):
                workflow.guard_consumer(
                    "experiment_test",
                    (checkpoints, runs, results),
                    (state,),
                )
                for root in (checkpoints, runs, results):
                    root.mkdir(parents=True)
                    (root / "generated.txt").write_text("old", encoding="utf-8")
                state.write_text("upstream-v2", encoding="utf-8")
                workflow.guard_consumer(
                    "experiment_test",
                    (checkpoints, runs, results),
                    (state,),
                )
                for root in (checkpoints, runs, results):
                    self.assertFalse(root.exists())
                events = list(
                    (repo / "backup" / "stale_cache" / "experiment_test").iterdir()
                )
                self.assertEqual(len(events), 1)
                record = json.loads((events[0] / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(len(record["moved"]), 3)
                checkpoints.mkdir(parents=True)
                marker = checkpoints / "new.txt"
                marker.write_text("new", encoding="utf-8")
                workflow.guard_consumer(
                    "experiment_test",
                    (checkpoints, runs, results),
                    (state,),
                )
                self.assertTrue(marker.exists())

    def test_rejects_cache_roots_outside_generated_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            state = repo / "state.json"
            state.parent.mkdir(parents=True)
            state.write_text("upstream", encoding="utf-8")
            with patch.multiple(
                workflow,
                REPO=repo,
                MANIFESTS=repo / "artifacts" / "manifests",
                STATE_ROOT=repo / "artifacts" / "provenance",
                STALE_CACHE_ROOT=repo / "backup" / "stale_cache",
            ):
                with self.assertRaises(ValueError):
                    workflow.guard_consumer("experiment_test", (repo / "data" / "raw",), (state,))

    def test_unbound_cache_is_archived_before_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            state = repo / "artifacts" / "provenance" / "data" / "internal.json"
            checkpoints = repo / "artifacts" / "checkpoints" / "experiment_test"
            state.parent.mkdir(parents=True)
            state.write_text("upstream", encoding="utf-8")
            checkpoints.mkdir(parents=True)
            (checkpoints / "unbound.pt").write_text("unbound", encoding="utf-8")
            with patch.multiple(
                workflow,
                REPO=repo,
                MANIFESTS=repo / "artifacts" / "manifests",
                STATE_ROOT=repo / "artifacts" / "provenance",
                STALE_CACHE_ROOT=repo / "backup" / "stale_cache",
            ), redirect_stdout(StringIO()):
                workflow.guard_consumer(
                    "experiment_test",
                    (checkpoints,),
                    (state,),
                )
                self.assertFalse(checkpoints.exists())
                events = list(
                    (repo / "backup" / "stale_cache" / "experiment_test").iterdir()
                )
                self.assertEqual(len(events), 1)
                self.assertTrue(
                    (
                        events[0]
                        / "artifacts"
                        / "checkpoints"
                        / "experiment_test"
                        / "unbound.pt"
                    ).exists()
                )
                binding = repo / "artifacts" / "provenance" / "consumers" / "experiment_test.json"
                self.assertTrue(binding.exists())


if __name__ == "__main__":
    unittest.main()
