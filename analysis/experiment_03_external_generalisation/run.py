from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import protocol, training
from analysis.experiment_03_external_generalisation import geometry, report, study
from pipeline import workflow


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "self-test",
            "prepare",
            "assemble",
            "report",
            "geometry",
            "freeze",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    args = parser.parse_args(argv)

    if args.stage == "self-test":
        study.self_test()
        geometry.self_test()
        report.self_test()
        return

    if args.stage in ("prepare", "all"):
        internal_state = workflow.ensure_internal()
        external_state = workflow.ensure_external()
        workflow.guard_consumer(
            study.EXPERIMENT_ID,
            (study.CHECKPOINTS, study.RUNS, study.RESULTS),
            (internal_state.path, external_state.path),
        )
        device = training.device_from(args.device)
        internal, external, model_records = study.prepare_transfer(device)
        geometry.prepare(device, internal, external)
        study.write_manifests(model_records)

    if args.stage in ("assemble", "all"):
        study.assemble()
    if args.stage in ("report", "all"):
        report.generate_transfer()
    if args.stage in ("geometry", "all"):
        report.generate_geometry()
    if args.stage in ("freeze", "all"):
        study.freeze()


if __name__ == "__main__":
    main()
