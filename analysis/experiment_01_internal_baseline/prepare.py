from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch

from analysis import protocol
from pipeline import provenance


RESOURCES = Path(__file__).resolve().parent / "resources"
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
DESCRIPTORS = {
    "atchley": ("atchley.tsv", 5),
    "kidera": ("kidera.tsv", 10),
    "aa_property": ("aa_properties.tsv", 14),
    "random": ("random.tsv", 5),
}
RANDOM_DESCRIPTOR_SEED = 913271
CDR3_COLUMN = {"alpha": "CDR3A", "beta": "CDR3B"}
STATE_ROOT = protocol.repo_path("artifacts/provenance/representations")


class UpstreamState(Protocol):
    role: str
    path: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def descriptor_table(method: str) -> pd.DataFrame:
    filename, dimension = DESCRIPTORS[method]
    table = pd.read_csv(RESOURCES / filename, sep="\t").set_index("amino.acid").astype(np.float32)
    if set(table.index) != STANDARD_AA or table.shape != (20, dimension):
        raise ValueError(f"Invalid descriptor resource: {filename}")
    span = table.max() - table.min()
    if (span == 0).any():
        raise ValueError(f"Constant descriptor column: {filename}")
    return ((table.max() - table) / span).sort_index()


@lru_cache(maxsize=None)
def descriptor_lookup(method: str) -> np.ndarray:
    table = descriptor_table(method)
    lookup = np.full((256, table.shape[1]), np.nan, dtype=np.float32)
    for amino_acid, values in table.iterrows():
        lookup[ord(amino_acid)] = values.to_numpy(dtype=np.float32)
    return lookup


def encode_sequences(sequences: pd.Series, method: str) -> torch.Tensor:
    values = sequences.astype(str).str.strip().to_numpy(dtype=object)
    if not len(values) or np.any(values == ""):
        raise ValueError("Empty CDR3 sequence")
    unique, inverse = np.unique(values, return_inverse=True)
    lengths = np.fromiter((len(sequence) for sequence in unique), dtype=np.int16)
    codes = np.full((len(unique), int(lengths.max())), ord("A"), dtype=np.uint8)
    mask = np.arange(codes.shape[1])[None, :] < lengths[:, None]
    allowed = np.zeros(256, dtype=bool)
    allowed[np.fromiter((ord(amino_acid) for amino_acid in STANDARD_AA), dtype=np.uint8)] = True
    for index, sequence in enumerate(unique):
        encoded = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        if not allowed[encoded].all():
            raise ValueError(f"Unsupported CDR3 residue: {sequence}")
        codes[index, : len(encoded)] = encoded
    lookup = descriptor_lookup(method)
    vectors = np.empty((len(unique), lookup.shape[1]), dtype=np.float32)
    for start in range(0, len(unique), 50_000):
        stop = min(start + 50_000, len(unique))
        selected = lookup[codes[start:stop]] * mask[start:stop, :, None]
        vectors[start:stop] = selected.sum(axis=1) / mask[start:stop].sum(axis=1, keepdims=True)
    return torch.from_numpy(np.ascontiguousarray(vectors[inverse]))


def prepare(role: str) -> pd.DataFrame:
    canonical = protocol.read_manifest(f"{role}_representations")
    samples = protocol.read_manifest(f"{role}_samples")
    records = []
    for source in canonical.sort_values(["chain", "subject_id"]).to_dict("records"):
        records.append(dict(source))
        sample = samples.loc[
            (samples["subject_id"] == source["subject_id"])
            & (samples["chain"] == source["chain"])
        ]
        if len(sample) != 1:
            raise ValueError(f"Expected one sample row for {source['subject_id']}/{source['chain']}")
        table = pd.read_csv(
            protocol.repo_path(sample.iloc[0]["tcr_table"]),
            sep="\t",
            usecols=[CDR3_COLUMN[source["chain"]]],
            keep_default_na=False,
        )
        sequences = table[CDR3_COLUMN[source["chain"]]]
        if len(sequences) != int(source["tcr_count"]):
            raise ValueError(f"TCR count mismatch: {source['subject_id']}/{source['chain']}")
        for method, (filename, dimension) in DESCRIPTORS.items():
            tensor = encode_sequences(sequences, method)
            output = protocol.repo_path(
                f"artifacts/representations/{role}/{source['chain']}/{method}/{source['subject_id']}.pt"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".pt.tmp")
            torch.save(tensor, temporary)
            temporary.replace(output)
            record = dict(source)
            record.update(
                embedding_file=output.relative_to(protocol.REPO).as_posix(),
                embedding_dim=str(dimension),
                representation_method=method,
                representation_version=f"{method}-{file_sha256(RESOURCES / filename)[:16]}-reverse-minmax-mean-v1",
                sha256=file_sha256(output),
            )
            records.append(record)
    result = pd.DataFrame(records).sort_values(["chain", "subject_id", "representation_method"])
    expected = {"sceptr", *DESCRIPTORS}
    for key, group in result.groupby(["chain", "subject_id"]):
        if set(group["representation_method"]) != expected:
            raise ValueError(f"Incomplete representation set: {key}")
    protocol.atomic_csv(result, protocol.MANIFESTS / f"representation_comparison_{role}.csv")
    return result


def _descriptor_outputs(role: str, digests: dict[Path, str]) -> list[Path] | None:
    manifest = protocol.MANIFESTS / f"representation_comparison_{role}.csv"
    if not manifest.exists():
        return None
    rows = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    expected_methods = {"sceptr", *DESCRIPTORS}
    if rows.empty:
        return None
    for _, group in rows.groupby(["chain", "subject_id"]):
        if set(group["representation_method"]) != expected_methods:
            return None
    canonical = pd.read_csv(
        protocol.MANIFESTS / f"{role}_representations.csv",
        dtype=str,
        keep_default_na=False,
    ).set_index(["chain", "subject_id"])
    sceptr_rows = rows.loc[rows["representation_method"] == "sceptr"].set_index(
        ["chain", "subject_id"]
    )
    if set(canonical.index) != set(sceptr_rows.index):
        return None
    for key in canonical.index:
        for column in ("embedding_file", "tcr_count", "embedding_dim", "sha256"):
            if canonical.at[key, column] != sceptr_rows.at[key, column]:
                return None
    outputs = [manifest]
    for row in rows.loc[rows["representation_method"] != "sceptr"].itertuples(index=False):
        output = protocol.repo_path(row.embedding_file)
        if not output.exists():
            return None
        resolved = output.resolve()
        if resolved not in digests:
            digests[resolved] = provenance.sha256(resolved)
        if not row.sha256 or digests[resolved] != row.sha256:
            return None
        outputs.append(output)
    return outputs


def ensure(role: str, upstream: UpstreamState) -> Path:
    """Prepare and provenance-bind descriptors without coupling the pipeline to Exp1."""

    if role not in {"internal", "external"} or upstream.role != role:
        raise ValueError(f"Descriptor role mismatch: requested={role!r}, upstream={upstream.role!r}")
    resources = [RESOURCES / value[0] for value in DESCRIPTORS.values()]
    specification = provenance.artifact_specification(
        "representation_comparison",
        [
            provenance.input_record(upstream.path, protocol.REPO),
            *[provenance.input_record(path, protocol.REPO) for path in resources],
        ],
        {
            "role": role,
            "methods": sorted(DESCRIPTORS),
            "transform": "reverse-minmax-mean-v1",
        },
        (Path(__file__),),
        ("numpy", "pandas", "torch"),
        protocol.REPO,
    )
    state_path = STATE_ROOT / f"{role}_descriptors.json"
    digests: dict[Path, str] = {}
    status = provenance.inspect_collection(state_path, specification, protocol.REPO, digests)
    if status.state == "valid":
        print(f"[cache] descriptors {role}: reused", flush=True)
        return state_path

    print(f"[stage] descriptors {role}: rebuilding", flush=True)
    prepare(role)
    digests.clear()
    outputs = _descriptor_outputs(role, digests)
    if outputs is None:
        raise RuntimeError(f"Descriptor preparation did not produce a valid {role} collection")
    provenance.write_collection(
        state_path,
        specification,
        outputs,
        protocol.REPO,
        digests,
    )
    return state_path


def self_test() -> None:
    for method, (_, dimension) in DESCRIPTORS.items():
        table = descriptor_table(method)
        assert table.shape == (20, dimension)
        first = encode_sequences(pd.Series(["AC", "DCA", "AC"]), method)
        assert first.shape == (3, dimension) and torch.equal(first[0], first[2])
        second = encode_sequences(pd.Series(["ACD"]), method)
        torch.testing.assert_close(first[1], second[0])
    stored_random = pd.read_csv(RESOURCES / "random.tsv", sep="\t").set_index("amino.acid")
    expected_random = np.round(
        np.random.default_rng(RANDOM_DESCRIPTOR_SEED).uniform(0.0, 1.0, (20, 5)), 9
    )
    assert list(stored_random.index) == list("ACDEFGHIKLMNPQRSTVWY")
    np.testing.assert_allclose(stored_random.to_numpy(), expected_random, rtol=0.0, atol=5e-10)
    print("experiment_01 representation preparation self-test passed")
