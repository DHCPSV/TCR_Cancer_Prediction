from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import training
from analysis.experiment_04_alice_model import report as reporting, study
from pipeline import reproduction_cache


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
        gate_free_pca,
        methods,
        prepare,
        vdjdb_lookup,
    )

    methods.self_test()
    study.self_test()
    reporting.self_test()
    prepare.self_test()
    families.self_test()
    vdjdb_lookup.self_test()
    gate_free_pca.self_test()
    print("experiment_04_alice_model self-test passed")


def report_main_results() -> None:
    from analysis.experiment_04_alice_model import families, vdjdb_lookup

    reporting.hard_gate_report()
    families.report()
    vdjdb_lookup.report()
    reporting.no_hard_gate_report()


def run_future_work(args: argparse.Namespace) -> None:
    from analysis.experiment_04_alice_model import gate_free_pca

    device = training.device_from(args.device)
    internal_states = study.prepare_future_work_inputs("internal", args)
    study.guard_representation_cache(internal_states)
    study.run_representation_comparison(device)
    study.prepare_future_work_inputs("external", args)
    study.validate_representation_comparison(device)
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
    parser.add_argument("--input-mode", choices=("raw", "cached"), default="raw")
    scripts = Path(sys.executable).resolve().parent
    default_olga = shutil.which("olga-compute_pgen") or scripts / "olga-compute_pgen.exe"
    parser.add_argument("--rscript", default=discover_rscript())
    parser.add_argument("--olga", default=str(default_olga))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.input_mode == "cached":
        if args.stage in {"internal", "validation"}:
            raise SystemExit(
                "cached mode contains report inputs, not ALICE preparation or training inputs"
            )
        reproduction_cache.verify_layout()
        if args.stage in {"self-test", "all"}:
            self_test()
        if args.stage in {"report", "all"}:
            report_main_results()
        if args.stage in {"future-work", "all"}:
            from analysis.experiment_04_alice_model import gate_free_pca

            reporting.future_work_report()
            gate_free_pca.report()
        return
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
        study.guard_main_cache(internal_states)
        study.run_hard_gate(device)
        study.prepare_alice_inputs("external", args)
        study.validate_hard_gate(device)
        reporting.hard_gate_report()

        from analysis.experiment_04_alice_model import families, vdjdb_lookup

        families.report(families.internal())
        vdjdb_lookup.report()
        study.guard_no_hard_gate_cache(study.alice_input_states("internal"))
        study.run_no_hard_gate(device)
        study.validate_no_hard_gate(device)
        reporting.no_hard_gate_report()
        return

    if args.stage == "internal":
        states = study.prepare_alice_inputs("internal", args)
        study.guard_main_cache(states)
        study.run_hard_gate(training.device_from(args.device))
    elif args.stage == "validation":
        study.prepare_alice_inputs("external", args)
        study.validate_hard_gate(training.device_from(args.device))
    else:
        report_main_results()


if __name__ == "__main__":
    main()
