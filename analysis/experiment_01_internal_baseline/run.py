"""Single command-line entry point for Experiment 01."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import protocol, training
from analysis.experiment_01_internal_baseline import prepare, report, study
from pipeline import report_cache, workflow


def _ready() -> None:
    upstream = workflow.ensure_internal()
    descriptors = prepare.ensure("internal", upstream)
    workflow.guard_consumer(
        study.EXPERIMENT_ID,
        (study.CHECKPOINTS, study.RUN_ARTIFACTS),
        (upstream.path, descriptors),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("self-test", "internal", "report", "diagnostics", "all"),
        default="all",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--input-mode", choices=("raw", "cached"), default="raw")
    args = parser.parse_args(argv)

    if args.input_mode == "cached":
        if args.stage == "internal":
            parser.error("cached mode contains report inputs, not training inputs")
        report_cache.verify()
        if args.stage in {"self-test", "all"}:
            prepare.self_test()
            study.self_test()
        if args.stage in {"report", "all"}:
            report.main()
        if args.stage in {"diagnostics", "all"}:
            report.diagnostics()
        return

    if args.stage in {"internal", "all"}:
        _ready()
    if args.stage in {"self-test", "all"}:
        prepare.self_test()
        study.self_test()
    if args.stage in {"internal", "all"}:
        study.internal(training.device_from(args.device))
    if args.stage in {"report", "all"}:
        report.main()
    if args.stage == "diagnostics":
        report.diagnostics()


if __name__ == "__main__":
    main()
