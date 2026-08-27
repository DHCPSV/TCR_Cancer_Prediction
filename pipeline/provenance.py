from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def input_record(
    path: Path,
    root: Path,
    *,
    digest: str | None = None,
    **metadata,
) -> dict:
    record = {
        "path": relative(path, root),
        "sha256": digest or sha256(path),
    }
    record.update(metadata)
    return record


def installed_versions(packages: Iterable[str]) -> dict[str, str]:
    values = {}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "not-installed"
    return values


def _signature_payload(record: Mapping) -> dict:
    return {
        "schema_version": record["schema_version"],
        "artifact_type": record["artifact_type"],
        "inputs": record["inputs"],
        "generator": record["generator"],
        "parameters": record["parameters"],
    }


def content_fingerprint(payload: Mapping) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_specification(
    artifact_type: str,
    inputs: list[dict],
    parameters: Mapping,
    code_files: Iterable[Path],
    packages: Iterable[str],
    root: Path,
) -> dict:
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "inputs": inputs,
        "generator": {
            "files": [
                input_record(path, root)
                for path in sorted(code_files, key=lambda item: relative(item, root))
            ],
            "packages": installed_versions(packages),
        },
        "parameters": dict(parameters),
    }
    record["fingerprint"] = content_fingerprint(_signature_payload(record))
    return record


def sidecar_path(output: Path) -> Path:
    return Path(f"{output}.provenance.json")


def _atomic_json(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class CacheStatus:
    state: str
    reason: str


def read_sidecar(output: Path) -> dict | None:
    path = sidecar_path(output)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def inspect_artifact(output: Path, expected: Mapping) -> CacheStatus:
    if not output.exists():
        return CacheStatus("missing", "output is absent")
    recorded = read_sidecar(output)
    if recorded is None:
        return CacheStatus("untracked", "provenance sidecar is absent")
    if not recorded:
        return CacheStatus("corrupt", "provenance sidecar is unreadable")
    if recorded.get("schema_version") != SCHEMA_VERSION:
        return CacheStatus("stale", "provenance schema changed")
    if recorded.get("fingerprint") != expected.get("fingerprint"):
        return CacheStatus("stale", "input, code, package or parameter fingerprint changed")
    output_record = recorded.get("output", {})
    if output_record.get("sha256") != sha256(output):
        return CacheStatus("corrupt", "output checksum does not match its sidecar")
    return CacheStatus("valid", "fingerprint and output checksum match")


def validate_recorded_artifact(
    output: Path,
    root: Path,
    digests: dict[Path, str] | None = None,
    artifact_type: str | None = None,
) -> CacheStatus:
    if not output.exists():
        return CacheStatus("missing", "output is absent")
    recorded = read_sidecar(output)
    if recorded is None:
        return CacheStatus("untracked", "provenance sidecar is absent")
    if not recorded:
        return CacheStatus("corrupt", "provenance sidecar is unreadable")
    if recorded.get("schema_version") != SCHEMA_VERSION:
        return CacheStatus("stale", "provenance schema changed")
    if artifact_type is not None and recorded.get("artifact_type") != artifact_type:
        return CacheStatus("corrupt", "provenance artifact type is incorrect")
    try:
        expected_fingerprint = content_fingerprint(_signature_payload(recorded))
        if recorded.get("fingerprint") != expected_fingerprint:
            return CacheStatus("corrupt", "sidecar fingerprint is internally inconsistent")
        cache = digests if digests is not None else {}

        def current_digest(path: Path) -> str:
            resolved = path.resolve()
            if resolved not in cache:
                cache[resolved] = sha256(resolved)
            return cache[resolved]

        for item in recorded["inputs"]:
            path = Path(item["path"])
            path = path if path.is_absolute() else root / path
            if not path.exists() or current_digest(path) != item["sha256"]:
                return CacheStatus("stale", f"input changed: {item['path']}")
        for item in recorded["generator"]["files"]:
            path = Path(item["path"])
            path = path if path.is_absolute() else root / path
            if not path.exists() or current_digest(path) != item["sha256"]:
                return CacheStatus("stale", f"generator changed: {item['path']}")
        packages = recorded["generator"].get("packages", {})
        if installed_versions(packages) != packages:
            return CacheStatus("stale", "package versions changed")
        if recorded.get("output", {}).get("sha256") != current_digest(output):
            return CacheStatus("corrupt", "output checksum does not match its sidecar")
    except (KeyError, TypeError, OSError):
        return CacheStatus("corrupt", "provenance sidecar has an invalid schema")
    return CacheStatus("valid", "recorded lineage and output checksum match")


def write_sidecar(
    output: Path,
    specification: Mapping,
    root: Path,
    **output_metadata,
) -> None:
    record = dict(specification)
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    record["output"] = {
        "path": relative(output, root),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        **output_metadata,
    }
    _atomic_json(record, sidecar_path(output))


def write_state(path: Path, specification: Mapping) -> bool:
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("fingerprint") == specification.get("fingerprint"):
            return False
    record = dict(specification)
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(record, path)
    return True


def inspect_collection(
    state_path: Path,
    expected: Mapping,
    root: Path,
    digests: dict[Path, str] | None = None,
) -> CacheStatus:
    if not state_path.exists():
        return CacheStatus("untracked", "collection provenance is absent")
    try:
        record = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CacheStatus("corrupt", "collection provenance is unreadable")
    if record.get("fingerprint") != expected.get("fingerprint"):
        return CacheStatus("stale", "collection inputs, code, packages or parameters changed")
    cache = digests if digests is not None else {}
    try:
        for item in record["outputs"]:
            path = Path(item["path"])
            path = path if path.is_absolute() else root / path
            resolved = path.resolve()
            if not path.exists():
                return CacheStatus("missing", f"collection output is absent: {item['path']}")
            if resolved not in cache:
                cache[resolved] = sha256(resolved)
            if cache[resolved] != item["sha256"]:
                return CacheStatus("corrupt", f"collection output changed: {item['path']}")
    except (KeyError, TypeError, OSError):
        return CacheStatus("corrupt", "collection provenance has an invalid schema")
    return CacheStatus("valid", "collection fingerprint and output checksums match")


def write_collection(
    state_path: Path,
    specification: Mapping,
    outputs: Iterable[Path],
    root: Path,
    digests: dict[Path, str] | None = None,
) -> None:
    cache = digests if digests is not None else {}
    records = []
    for output in sorted(outputs, key=lambda item: relative(item, root)):
        resolved = output.resolve()
        if resolved not in cache:
            cache[resolved] = sha256(resolved)
        records.append(
            {
                "path": relative(output, root),
                "sha256": cache[resolved],
                "bytes": output.stat().st_size,
            }
        )
    record = dict(specification)
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    record["outputs"] = records
    _atomic_json(record, state_path)


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minute:02d}m"
    if minutes:
        return f"{minutes:d}m{second:02d}s"
    return f"{second:d}s"


class Progress:
    def __init__(self, stage: str, total: int, updates: int = 4):
        self.stage = stage
        self.total = total
        self.completed = 0
        self.started = time.monotonic()
        self.interval = max(1, total // max(1, updates))
        print(f"[stage] {stage}: 0/{total}", flush=True)

    def advance(self) -> None:
        self.completed += 1
        if self.completed != self.total and self.completed % self.interval:
            return
        elapsed = time.monotonic() - self.started
        eta = elapsed * (self.total - self.completed) / max(1, self.completed)
        print(
            f"[stage] {self.stage}: {self.completed}/{self.total} "
            f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}",
            flush=True,
        )

    def summary(self, **counts: int) -> None:
        elapsed = time.monotonic() - self.started
        details = " ".join(f"{key}={value}" for key, value in counts.items())
        print(f"[cache] {self.stage}: {details} elapsed={format_duration(elapsed)}", flush=True)
