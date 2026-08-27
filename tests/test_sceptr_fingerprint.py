from __future__ import annotations

import unittest

import torch

from pipeline.embed_sceptr import model_fingerprint


class Wrapper:
    def __init__(self) -> None:
        self._bert = torch.nn.Linear(3, 2)


class SceptrFingerprintTests(unittest.TestCase):
    def test_sceptr_wrapper_uses_inner_bert_state(self) -> None:
        torch.manual_seed(17)
        wrapper = Wrapper()
        self.assertEqual(model_fingerprint(wrapper), model_fingerprint(wrapper._bert))

    def test_fingerprint_changes_with_model_weights(self) -> None:
        torch.manual_seed(17)
        wrapper = Wrapper()
        before = model_fingerprint(wrapper)
        with torch.no_grad():
            wrapper._bert.weight[0, 0] += 1
        self.assertNotEqual(before, model_fingerprint(wrapper))


if __name__ == "__main__":
    unittest.main()
