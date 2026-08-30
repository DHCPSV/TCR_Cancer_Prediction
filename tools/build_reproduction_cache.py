"""Build the optional reproduction-cache ZIP from a completed project."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from analysis import protocol
from pipeline import reproduction_cache


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="ZIP path outside the repository")
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--commit", default="")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        output.relative_to(protocol.REPO.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Write the reproduction-cache ZIP outside the repository")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    files = reproduction_cache.selected_files()
    manifest = reproduction_cache.write_contents_manifest(files)
    contents_path = reproduction_cache.CONTENTS_MANIFEST
    unpacked_bytes = sum(path.stat().st_size for path in files) + contents_path.stat().st_size
    output.parent.mkdir(parents=True, exist_ok=True)
    completed_bytes = 0
    next_update = 0.1
    with zipfile.ZipFile(
        output,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in files + [contents_path]:
            archive.write(path, path.relative_to(protocol.REPO).as_posix())
            completed_bytes += path.stat().st_size
            fraction = completed_bytes / max(1, unpacked_bytes)
            if fraction >= next_update:
                print(f"[reproduction-cache] {fraction:.0%}", flush=True)
                next_update += 0.1

    release = {
        "schema_version": 2,
        "package_type": "reproduction_cache",
        "version": args.version,
        "compatible_commit": args.commit,
        "download_url": "",
        "zip_name": output.name,
        "zip_bytes": output.stat().st_size,
        "zip_sha256": sha256(output),
        "unpacked_bytes": unpacked_bytes,
        "contents_manifest": contents_path.relative_to(protocol.REPO).as_posix(),
        "contents_manifest_sha256": protocol.sha256(contents_path),
        "file_count": len(manifest["files"]),
    }
    temporary = reproduction_cache.RELEASE_MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(reproduction_cache.RELEASE_MANIFEST)
    print(output)
    print(release["zip_sha256"])


if __name__ == "__main__":
    main()
