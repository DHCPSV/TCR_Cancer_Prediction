"""Verify an extracted reproduction cache from the command line."""

from __future__ import annotations

import argparse

from pipeline import reproduction_cache


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-checkpoint-read",
        action="store_true",
        help="Check every file hash without opening every checkpoint.",
    )
    args = parser.parse_args(argv)
    summary = reproduction_cache.verify(
        check_checkpoints=not args.skip_checkpoint_read
    )
    print(
        f"Reproduction cache OK: {summary['file_count']} files, "
        f"{summary['bytes'] / (1024 ** 3):.2f} GiB"
    )


if __name__ == "__main__":
    main()
