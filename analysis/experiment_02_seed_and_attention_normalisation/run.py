from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import protocol, training
from analysis.experiment_02_seed_and_attention_normalisation import (
    layer_seed_study,
    normalizer_study,
    report,
)
from pipeline import reproduction_cache, workflow


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
    parser.add_argument("--input-mode", choices=("raw", "cached"), default="raw")
    args = parser.parse_args(argv)

    if args.input_mode == "cached":
        if args.stage == "internal":
            parser.error("cached mode contains report inputs, not training inputs")
        reproduction_cache.verify_layout()
        if args.stage in ("self-test", "all"):
            layer_seed_study.self_test()
            normalizer_study.self_test()
        if args.stage in ("report", "all"):
            report.generate()
        if args.stage in ("diagnostics", "all"):
            report.diagnostics()
        return

    if args.stage in ("self-test", "all"):
        layer_seed_study.self_test()
        normalizer_study.self_test()
    if args.stage in ("internal", "diagnostics", "all"):
        upstream = workflow.ensure_internal()
        workflow.guard_consumer(
            EXPERIMENT_ID,
            (CHECKPOINTS, RUN_ARTIFACTS),
            (upstream.path,),
        )
    if args.stage in ("internal", "all"):
        device = training.device_from(args.device)
        layer_seed_study.internal(device)
        normalizer_study.internal(device)
    if args.stage == "diagnostics":
        device = training.device_from(args.device)
        layer_seed_study.diagnostics(device)
        report.diagnostics()
    if args.stage in ("report", "all"):
        report.generate()


if __name__ == "__main__":
    main()
