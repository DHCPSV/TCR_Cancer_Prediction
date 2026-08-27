from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import protocol, training
from analysis.experiment_04_alice_model import core, report as reporting, study


WINDOWS_RSCRIPT = Path(r"C:\Program Files\R\R-4.3.3\bin\Rscript.exe")


def discover_rscript() -> str:
    discovered = shutil.which("Rscript")
    if discovered:
        return discovered
    if sys.platform == "win32" and WINDOWS_RSCRIPT.is_file():
        return str(WINDOWS_RSCRIPT)
    return "Rscript"


def self_test() -> None:
    from analysis.experiment_04_alice_model import (
        families,
        methods,
        no_hard_gate,
        prepare,
        vdjdb_lookup,
    )

    methods.self_test()
    core.self_test()
    study.self_test()
    reporting.self_test()
    prepare.self_test()
    no_hard_gate.self_test()
    families.self_test()
    vdjdb_lookup.self_test()
    print("experiment_04_alice_model self-test passed")


def report_formal_results() -> None:
    from analysis.experiment_04_alice_model import families, no_hard_gate, vdjdb_lookup

    reporting.hard_gate_report()
    families.report()
    vdjdb_lookup.report()
    no_hard_gate.report()


def run_future_work(args: argparse.Namespace) -> None:
    from analysis.experiment_04_alice_model import gate_free_pca

    device = training.device_from(args.device)
    internal_states = study.prepare_future_work_inputs("internal", args)
    study.guard_internal_cache(study.FUTURE_WORK_STUDY, internal_states)
    study.internal(study.FUTURE_WORK_STUDY, device)
    study.prepare_future_work_inputs("external", args)
    study.validation(study.FUTURE_WORK_STUDY, device)
    reporting.future_work_report()
    gate_free_pca.report()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "self-test",
            "internal",
            "validation",
            "report",
            "future-work",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    scripts = Path(sys.executable).resolve().parent
    default_olga = shutil.which("olga-compute_pgen") or scripts / "olga-compute_pgen.exe"
    parser.add_argument("--rscript", default=discover_rscript())
    parser.add_argument("--olga", default=str(default_olga))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage == "self-test":
        self_test()
        return
    if args.stage == "future-work":
        run_future_work(args)
        return

    if args.stage == "all":
        self_test()
        device = training.device_from(args.device)
        internal_states = study.prepare_alice_inputs("internal", args)
        study.guard_internal_cache(study.HARD_GATE_STUDY, internal_states)
        study.internal(study.HARD_GATE_STUDY, device)
        study.prepare_alice_inputs("external", args)
        study.validation(study.HARD_GATE_STUDY, device)
        reporting.hard_gate_report()

        from analysis.experiment_04_alice_model import families, no_hard_gate, vdjdb_lookup

        families.report(families.internal())
        vdjdb_lookup.report()
        no_hard_gate.run_frozen(device, study.alice_input_states("internal"))
        return

    if args.stage == "internal":
        states = study.prepare_alice_inputs("internal", args)
        study.guard_internal_cache(study.HARD_GATE_STUDY, states)
        study.internal(study.HARD_GATE_STUDY, training.device_from(args.device))
    elif args.stage == "validation":
        study.prepare_alice_inputs("external", args)
        study.validation(study.HARD_GATE_STUDY, training.device_from(args.device))
    else:
        report_formal_results()


if __name__ == "__main__":
    main()
