from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline import build_manifests, embed_sceptr, prepare_data, provenance


REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "artifacts" / "manifests"
STATE_ROOT = REPO / "artifacts" / "provenance"
STALE_CACHE_ROOT = REPO / "backup" / "stale_cache"


@dataclass(frozen=True)
class ReadyState:
    role: str
    path: Path


def _role_is_bootstrapped(role: str) -> bool:
    paths = (MANIFESTS / "samples.csv", MANIFESTS / "representations.csv")
    if not all(path.exists() for path in paths):
        return False
    samples = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    representations = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    return role in set(samples.get("role", ())) and role in set(representations.get("role", ()))


def _bootstrap_role(role: str) -> None:
    if not _role_is_bootstrapped(role):
        print(f"[workflow] {role}: building manifests from raw data", flush=True)
        build_manifests.bootstrap(role)
    else:
        print(f"[workflow] {role}: using existing raw-data manifest", flush=True)
    build_manifests.rebuild_views()


def _role_state(role: str) -> ReadyState:
    samples = pd.read_csv(MANIFESTS / f"{role}_samples.csv", dtype=str, keep_default_na=False)
    representations = pd.read_csv(
        MANIFESTS / f"{role}_representations.csv", dtype=str, keep_default_na=False
    )
    files = [
        MANIFESTS / f"{role}_subjects.csv",
        MANIFESTS / f"{role}_samples.csv",
        MANIFESTS / f"{role}_representations.csv",
    ]
    if role == "internal":
        files.append(MANIFESTS / f"fivefold_seed{build_manifests.SEED}.csv")
    for value in samples["tcr_table"].drop_duplicates():
        files.append(provenance.sidecar_path(REPO / value))
    for value in representations["embedding_file"].drop_duplicates():
        files.append(provenance.sidecar_path(REPO / value))
    specification = provenance.artifact_specification(
        "ready_patient_representations",
        [provenance.input_record(path, REPO, ordinal=index) for index, path in enumerate(files)],
        {
            "role": role,
            "sample_rows": len(samples),
            "representation_rows": len(representations),
            "fold_seed": build_manifests.SEED if role == "internal" else None,
        },
        (),
        ("pandas",),
        REPO,
    )
    state_path = STATE_ROOT / "data" / f"{role}.json"
    provenance.write_state(state_path, specification)
    return ReadyState(role, state_path)


def ensure(role: str, *, batch_size: int = 4096) -> ReadyState:
    started = time.monotonic()
    _bootstrap_role(role)
    for chain in ("alpha", "beta"):
        prepare_data.run(role, chain, overwrite=False)
    for chain in ("alpha", "beta"):
        embed_sceptr.run(role, chain, batch_size, overwrite=False)
    digests = build_manifests.refresh(role)
    if role == "internal":
        build_manifests.rebuild_folds()
    build_manifests.verify(role, digests)
    state = _role_state(role)
    print(
        f"[workflow] {role}: ready elapsed={provenance.format_duration(time.monotonic() - started)}",
        flush=True,
    )
    return state


def ensure_internal(*, batch_size: int = 4096) -> ReadyState:
    return ensure("internal", batch_size=batch_size)


def ensure_external(*, batch_size: int = 4096) -> ReadyState:
    return ensure("external", batch_size=batch_size)


def _validated_cache_roots(cache_roots: Iterable[Path]) -> list[Path]:
    generated_parents = tuple(
        path.resolve()
        for path in (
            REPO / "artifacts" / "checkpoints",
            REPO / "artifacts" / "runs",
        )
    )
    roots = []
    for value in cache_roots:
        path = Path(value)
        path = path if path.is_absolute() else REPO / path
        resolved = path.resolve()
        if not any(
            resolved != parent and resolved.is_relative_to(parent)
            for parent in generated_parents
        ):
            raise ValueError(
                "Experiment cache root must be below artifacts/checkpoints "
                f"or artifacts/runs: {path}"
            )
        roots.append(resolved)
    if len(set(roots)) != len(roots):
        raise ValueError("Experiment cache roots contain duplicates")
    for index, root in enumerate(roots):
        if any(
            index != other_index
            and (root.is_relative_to(other) or other.is_relative_to(root))
            for other_index, other in enumerate(roots)
        ):
            raise ValueError("Experiment cache roots must not overlap")
    return roots


def _populated_cache_roots(roots: Iterable[Path]) -> list[Path]:
    return [
        root
        for root in roots
        if root.is_file() or (root.is_dir() and any(path.is_file() for path in root.rglob("*")))
    ]


def _archive_stale_cache(
    name: str,
    roots: Iterable[Path],
    binding: Path,
    previous: dict,
    current: dict,
) -> Path:
    expected_archive_root = (REPO / "backup" / "stale_cache").resolve()
    if STALE_CACHE_ROOT.resolve() != expected_archive_root:
        raise ValueError(f"Stale-cache archive must remain inside the repository: {STALE_CACHE_ROOT}")
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "experiment"
    event = STALE_CACHE_ROOT / safe_name / token
    moved: list[tuple[Path, Path]] = []
    try:
        for source in roots:
            destination = event / source.relative_to(REPO.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
        if binding.exists():
            destination = event / "previous_binding.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(binding, destination)
        record = {
            "consumer": name,
            "previous_fingerprint": previous.get("fingerprint"),
            "current_fingerprint": current.get("fingerprint"),
            "moved": [
                {
                    "source": provenance.relative(source, REPO),
                    "destination": provenance.relative(destination, REPO),
                }
                for source, destination in moved
            ],
        }
        (event / "manifest.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        rollback_failures = []
        for source, destination in reversed(moved):
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
            except Exception as error:
                rollback_failures.append(f"{destination} -> {source}: {error}")
        if rollback_failures:
            raise RuntimeError(
                "Could not archive stale caches or completely restore them: "
                + "; ".join(rollback_failures)
            )
        raise
    return event


def guard_consumer(name: str, cache_roots: Iterable[Path], states: Iterable[Path]) -> None:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
        raise ValueError(f"Invalid experiment cache consumer name: {name!r}")
    inputs = [provenance.input_record(path, REPO) for path in states]
    specification = provenance.artifact_specification(
        "experiment_cache_binding",
        inputs,
        {"consumer": name},
        (),
        (),
        REPO,
    )
    binding = STATE_ROOT / "consumers" / f"{name}.json"
    previous = {}
    binding_exists = binding.exists()
    if binding_exists:
        try:
            previous = json.loads(binding.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    roots = _validated_cache_roots(cache_roots)
    if previous.get("fingerprint") == specification["fingerprint"]:
        print(f"[cache] {name}: upstream binding unchanged", flush=True)
        return
    populated = _populated_cache_roots(roots)
    if populated:
        archived = _archive_stale_cache(name, populated, binding, previous, specification)
        reason = "upstream changed" if binding_exists else "unbound cache has no upstream binding"
        print(
            f"[cache] {name}: {reason}; archived {len(populated)} stale roots to "
            f"{provenance.relative(archived, REPO)}",
            flush=True,
        )
    provenance.write_state(binding, specification)
    print(f"[cache] {name}: bound new cache", flush=True)
