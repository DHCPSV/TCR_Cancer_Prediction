from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import protocol, training
from analysis.experiment_02_seed_and_attention_normalisation import (
    normalizer_study,
    report,
    seed_mechanism,
)
from pipeline import workflow


EXPERIMENT_ID = "experiment_02_seed_and_attention_normalisation"
CHECKPOINTS = protocol.repo_path(f"artifacts/checkpoints/{EXPERIMENT_ID}")
RUN_ARTIFACTS = protocol.repo_path(f"artifacts/runs/{EXPERIMENT_ID}")
RESULTS = protocol.repo_path(f"results/{EXPERIMENT_ID}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("self-test", "internal", "diagnostics", "report", "all"),
        default="all",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)

    if args.stage in ("self-test", "all"):
        seed_mechanism.self_test()
        normalizer_study.self_test()
    if args.stage in ("internal", "diagnostics", "all"):
        upstream = workflow.ensure_internal()
        workflow.guard_consumer(
            EXPERIMENT_ID,
            (CHECKPOINTS, RUN_ARTIFACTS, RESULTS),
            (upstream.path,),
        )
    if args.stage in ("internal", "all"):
        device = training.device_from(args.device)
        factorization = seed_mechanism.internal(device)
        normalizers = normalizer_study.internal(device)
        normalizer_study.write_internal_predictions(factorization, normalizers)
    if args.stage == "diagnostics":
        device = training.device_from(args.device)
        seed_mechanism.diagnostics(device)
        report.diagnostics()
    if args.stage in ("report", "all"):
        report.generate()


if __name__ == "__main__":
    main()
