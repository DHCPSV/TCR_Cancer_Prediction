from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis import training


class TrainingPairBindingTest(unittest.TestCase):
    def test_checkpoint_history_and_contract_are_bound_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            checkpoint = repo / "artifacts" / "checkpoints" / "model.pt"
            history = repo / "artifacts" / "runs" / "history.csv"
            checkpoint.parent.mkdir(parents=True)
            history.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint-v1")
            history.write_text("epoch,auc\n1,0.5\n", encoding="utf-8")
            contract = {"epochs": 50, "seed": "777", "fold": 0}

            sidecar = training.bind_checkpoint_history(
                checkpoint, history, contract, repo=repo
            )
            training.validate_checkpoint_history_pair(
                checkpoint, history, contract, repo=repo
            )
            self.assertTrue(
                training.checkpoint_history_pair_is_valid(
                    checkpoint, history, contract, repo=repo
                )
            )
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertIn("checkpoint_sha256", payload)
            self.assertIn("history_sha256", payload)
            self.assertIn("contract_sha256", payload)

            history.write_text("epoch,auc\n1,0.6\n", encoding="utf-8")
            self.assertFalse(
                training.checkpoint_history_pair_is_valid(
                    checkpoint, history, contract, repo=repo
                )
            )

            training.bind_checkpoint_history(
                checkpoint, history, contract, repo=repo
            )
            self.assertFalse(
                training.checkpoint_history_pair_is_valid(
                    checkpoint,
                    history,
                    contract | {"epochs": 100},
                    repo=repo,
                )
            )

            checkpoint.write_bytes(b"checkpoint-v2")
            self.assertFalse(
                training.checkpoint_history_pair_is_valid(
                    checkpoint, history, contract, repo=repo
                )
            )


if __name__ == "__main__":
    unittest.main()
