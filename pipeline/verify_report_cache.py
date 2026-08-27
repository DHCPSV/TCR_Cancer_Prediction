"""Verify an extracted report cache from the command line."""

from __future__ import annotations

import argparse

from pipeline import report_cache


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-checkpoint-read",
        action="store_true",
        help="Check file hashes without opening every checkpoint.",
    )
    args = parser.parse_args(argv)
    summary = report_cache.verify(check_checkpoints=not args.skip_checkpoint_read)
    print(
        f"Report cache OK: {summary['file_count']} files, "
        f"{summary['bytes'] / (1024 ** 2):.1f} MiB"
    )


if __name__ == "__main__":
    main()
