from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import protocol, training
from analysis.experiment_03_external_generalisation import pca, report, study
from pipeline import report_cache, workflow


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "self-test",
            "prepare",
            "report",
            "pca",
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
    parser.add_argument("--input-mode", choices=("raw", "cached"), default="raw")
    args = parser.parse_args(argv)

    if args.input_mode == "cached":
        if args.stage in {"prepare", "freeze"}:
            parser.error("cached mode contains report inputs, not preparation inputs")
        report_cache.verify()
        if args.stage in {"self-test", "all"}:
            study.self_test()
            pca.self_test()
            report.self_test()
        if args.stage in {"report", "all"}:
            report.generate_transfer()
        if args.stage in {"pca", "all"}:
            report.generate_pca()
        return

    if args.stage == "self-test":
        study.self_test()
        pca.self_test()
        report.self_test()
        return

    if args.stage in ("prepare", "all"):
        internal_state = workflow.ensure_internal()
        external_state = workflow.ensure_external()
        workflow.guard_consumer(
            study.EXPERIMENT_ID,
            (study.CHECKPOINTS, study.RUNS),
            (internal_state.path, external_state.path),
        )
        device = training.device_from(args.device)
        internal, external, model_records = study.prepare_transfer(device)
        pca.prepare(device, internal, external)
        study.write_manifests(model_records)

    if args.stage in ("report", "all"):
        report.generate_transfer()
    if args.stage in ("pca", "all"):
        report.generate_pca()
    if args.stage in ("freeze", "all"):
        study.freeze()


if __name__ == "__main__":
    main()
