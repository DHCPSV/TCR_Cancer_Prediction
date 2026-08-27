from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from pipeline import sceptr_adapter


FIXTURE = pd.DataFrame(
    [
        {
            "TRAV": "TRAV8-3",
            "TRAJ": "TRAJ45",
            "CDR3A": "CAVGDAGTGGGADGLTF",
            "TRBV": None,
            "TRBJ": None,
            "CDR3B": None,
        },
        {
            "TRAV": None,
            "TRAJ": None,
            "CDR3A": None,
            "TRBV": "TRBV5-4",
            "TRBJ": "TRBJ2-7",
            "CDR3B": "CASSFEGGGYEQYF",
        },
    ]
)


def test_adapter_matches_public_sceptr_tokens_and_vectors() -> None:
    import sceptr

    model = sceptr.variant.default()
    model.set_batch_size(2)
    normalised = sceptr_adapter.normalise(FIXTURE)
    sceptr_adapter.validate_tokens(normalised, model)
    public = model.calc_vector_representations(FIXTURE)
    adapted = sceptr_adapter.calc_vector_representations(FIXTURE, model).numpy()
    np.testing.assert_allclose(adapted, public, rtol=1e-6, atol=1e-7)


class SceptrAdapterTests(unittest.TestCase):
    test_adapter_matches_public_sceptr_tokens_and_vectors = staticmethod(
        test_adapter_matches_public_sceptr_tokens_and_vectors
    )
