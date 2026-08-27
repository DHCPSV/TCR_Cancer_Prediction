from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from analysis import protocol


ALICE = protocol.REPO / "artifacts" / "alice"
WORKER = protocol.REPO / "third_party" / "alice" / "alice_worker.R"
Q_HIT = 0.001
ALICE_PARAMETERS = {
    "Q_val": 27,
    "thres_counts": 0,
    "N_neighbors_thres": 2,
    "p_adjust_method": "BH",
    "count_mode": "presence_only",
    "olga_version": "1.2.4",
}

FREEZE = protocol.MANIFESTS / "alice_runtime_freeze.json"
EXPERIMENT_ID = "experiment_04_alice_model"
MODEL_CODE_FILES = (
    "analysis/training.py",
    "analysis/experiment_04_alice_model/core.py",
    "analysis/experiment_04_alice_model/methods.py",
    "analysis/experiment_04_alice_model/prepare.py",
    "analysis/experiment_04_alice_model/study.py",
    "pipeline/prepare_data.py",
    "third_party/alice/alice_worker.R",
)
ANALYSIS_CODE_FILES = (
    "analysis/reporting.py",
    "analysis/experiment_04_alice_model/report.py",
    "analysis/experiment_04_alice_model/learning_curves.py",
    "analysis/experiment_04_alice_model/run.py",
    "analysis/experiment_04_alice_model/families.py",
    "analysis/experiment_04_alice_model/vdjdb_lookup.py",
    "analysis/experiment_04_alice_model/no_hard_gate.py",
    "analysis/experiment_04_alice_model/gate_free_pca.py",
)


class RuntimeOptions(Protocol):
    rscript: str
    olga: str
    workers: int


def sample_key(chain: str, subject_id: str, sample_id: str) -> str:
    token = hashlib.sha256(
        f"{chain}|{subject_id}|{sample_id}".encode()
    ).hexdigest()[:12]
    return f"{chain}_{subject_id}_{token}".replace(" ", "_")


def _gzip_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_path(output: Path) -> Path:
    return Path(f"{output}.cache.json")


def _cache_signature(row: dict, olga: str) -> dict:
    input_path = protocol.repo_path(row["input_path"])
    return {
        "schema_version": 1,
        "input_sha256": _gzip_content_sha256(input_path),
        "worker_sha256": protocol.sha256(WORKER),
        "chain": row["chain"],
        "olga_command": olga,
        "parameters": ALICE_PARAMETERS,
    }


def _cache_valid(row: dict, olga: str) -> bool:
    output = protocol.repo_path(row["output_path"])
    metadata = Path(f"{output}.meta.tsv")
    sidecar = _cache_path(output)
    if not output.exists() or not metadata.exists() or not sidecar.exists():
        return False
    try:
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    signature = _cache_signature(row, olga)
    return (
        all(cached.get(key) == value for key, value in signature.items())
        and cached.get("output_sha256") == protocol.sha256(output)
        and cached.get("metadata_sha256") == protocol.sha256(metadata)
    )


def _write_cache(row: dict, olga: str) -> None:
    output = protocol.repo_path(row["output_path"])
    metadata = Path(f"{output}.meta.tsv")
    sidecar = _cache_path(output)
    payload = _cache_signature(row, olga) | {
        "output_sha256": protocol.sha256(output),
        "metadata_sha256": protocol.sha256(metadata),
    }
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(sidecar)


def clonotypes(
    path: Path,
    chain: str,
    selector_column: str = "",
    selector_value: str = "",
) -> pd.DataFrame:
    from pipeline.prepare_data import clean_airr_rows

    cleaned = clean_airr_rows(path, selector_column, selector_value, chain)
    return cleaned.rename(
        columns={
            "v_gene": "V",
            "j_gene": "J",
            "cdr3_aa": "cdr3aa",
            "cdr3_nt": "cdr3nt",
        }
    )


def assert_tcr_table_alignment(
    frame: pd.DataFrame, tcr_table: Path, chain: str
) -> None:
    expected = pd.read_csv(tcr_table, sep="\t", dtype=str, keep_default_na=False)
    suffix = "A" if chain == "alpha" else "B"
    observed = frame[["V", "J", "cdr3aa"]].reset_index(drop=True)
    expected = expected[[f"TR{suffix}V", f"TR{suffix}J", f"CDR3{suffix}"]]
    expected.columns = ["V", "J", "cdr3aa"]
    expected = expected.reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError(f"{tcr_table}: raw cleaning does not match frozen TCR rows")


def alice_clone_mapping(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    nucleotide = frame["cdr3nt"].astype(str).str.strip()
    testable = nucleotide.str.len().gt(0) & nucleotide.str.len().mod(3).eq(0)
    clone_index = np.full(len(frame), -1, dtype=np.int32)
    tested = frame.loc[testable].copy()
    keys = pd.MultiIndex.from_frame(tested[["cdr3nt", "V", "J"]])
    codes, _ = pd.factorize(keys, sort=False)
    clone_index[np.flatnonzero(testable.to_numpy())] = codes.astype(np.int32)
    unique = tested.loc[~pd.Series(codes).duplicated().to_numpy()].reset_index(drop=True)
    return unique, clone_index


def build_inputs(split: str) -> pd.DataFrame:
    samples = protocol.read_manifest(f"{split}_samples").sort_values(
        ["chain", "subject_id", "sample_id"]
    )
    representations = protocol.read_manifest(f"{split}_representations").set_index(
        ["subject_id", "chain"]
    )
    rows = []
    for (chain, subject_id), patient in samples.groupby(
        ["chain", "subject_id"], sort=True
    ):
        sequence_sample, sequence_clone, sample_names = [], [], []
        for index, sample in enumerate(patient.itertuples(index=False)):
            source = protocol.repo_path(sample.raw_file or sample.tcr_table)
            frame = clonotypes(
                source,
                chain,
                sample.raw_subject_column,
                sample.raw_subject_value,
            )
            assert_tcr_table_alignment(
                frame, protocol.repo_path(sample.tcr_table), chain
            )
            unique, clone_index = alice_clone_mapping(frame)
            key = sample_key(chain, subject_id, sample.sample_id)
            input_path = ALICE / "inputs" / split / chain / f"{key}.tsv.gz"
            output_path = ALICE / "outputs" / split / chain / f"{key}.tsv"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "clone_key": np.arange(len(unique), dtype=np.int32),
                    "cdr3nt": unique["cdr3nt"],
                    "cdr3aa": unique["cdr3aa"],
                    "bestVGene": unique["V"],
                    "bestJGene": unique["J"],
                }
            ).to_csv(input_path, sep="\t", index=False, compression="gzip")
            sequence_sample.extend([index] * len(frame))
            sequence_clone.extend(clone_index.tolist())
            sample_names.append(str(sample.sample_id))
            representation = representations.loc[(subject_id, chain)]
            rows.append(
                {
                    "split": split,
                    "role": split,
                    "chain": chain,
                    "subject_id": subject_id,
                    "legacy_patient_id": representation["legacy_patient_id"],
                    "label": representation["label"],
                    "cohort": representation["cohort"],
                    "sample_index": index,
                    "sample_id": sample.sample_id,
                    "sample_key": key,
                    "input_path": input_path.relative_to(protocol.REPO).as_posix(),
                    "output_path": output_path.relative_to(protocol.REPO).as_posix(),
                    "unique_clonotypes": len(unique),
                    "embedding_rows": len(frame),
                }
            )
        mapping = ALICE / "sequence_maps" / split / chain / f"{subject_id}.npz"
        mapping.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mapping,
            sample_index=np.asarray(sequence_sample, dtype=np.int16),
            clone_index=np.asarray(sequence_clone, dtype=np.int32),
            sample_names=np.asarray(sample_names, dtype=str),
        )
    registry = pd.DataFrame(rows)
    protocol.atomic_csv(registry, ALICE / f"{split}_samples.csv")
    return registry


def run_alice(split: str, rscript: str, olga: str, workers: int) -> pd.DataFrame:
    registry = pd.read_csv(ALICE / f"{split}_samples.csv")
    pending = [
        row
        for row in registry.to_dict("records")
        if not _cache_valid(row, olga)
    ]
    print(
        f"{split}: {len(registry) - len(pending)} cached, "
        f"{len(pending)} ALICE sample jobs pending",
        flush=True,
    )
    completed = 0
    lock = threading.Lock()

    environment = os.environ.copy()
    executable = Path(rscript).resolve()
    if executable.exists():
        environment_root = executable.parent.parent
        runtime_paths = (
            environment_root / "Library" / "bin",
            environment_root / "Scripts",
            environment_root / "Lib" / "R" / "bin",
            environment_root / "Lib" / "R" / "bin" / "x64",
        )
        environment["PATH"] = os.pathsep.join(map(str, runtime_paths)) + os.pathsep + environment.get("PATH", "")

    def run(row: dict) -> None:
        nonlocal completed
        process = subprocess.run(
            [
                rscript,
                str(WORKER),
                str(protocol.repo_path(row["input_path"])),
                str(protocol.repo_path(row["output_path"])),
                row["chain"],
                olga,
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        if process.returncode:
            raise RuntimeError(
                f"ALICE failed for {row['sample_key']}:\n{process.stdout}\n{process.stderr}"
            )
        _write_cache(row, olga)
        with lock:
            completed += 1
            if completed == len(pending) or completed % 20 == 0:
                print(f"{split}: ALICE {completed}/{len(pending)}", flush=True)

    if workers == 1:
        for row in pending:
            run(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, pending))
    return registry


def build_evidence(split: str) -> pd.DataFrame:
    registry = pd.read_csv(ALICE / f"{split}_samples.csv")
    representations = protocol.read_manifest(f"{split}_representations")
    records = []
    for (chain, subject_id), samples in registry.groupby(
        ["chain", "subject_id"], sort=True
    ):
        mapping_path = ALICE / "sequence_maps" / split / chain / f"{subject_id}.npz"
        mapping = np.load(mapping_path, allow_pickle=False)
        sequence_sample = mapping["sample_index"]
        sequence_clone = mapping["clone_index"]
        evidence = np.zeros(len(sequence_clone), dtype=np.float32)
        burdens, hits = [], 0
        for sample in samples.itertuples(index=False):
            q = np.ones(sample.unique_clonotypes, dtype=np.float64)
            result = pd.read_csv(protocol.repo_path(sample.output_path), sep="\t")
            if len(result):
                result_keys = result["clone_key"].astype(int).to_numpy()
                if result_keys.min() < 0 or result_keys.max() >= len(q):
                    raise ValueError(f"{sample.output_path}: invalid clone_key")
                q[result_keys] = result["q"].astype(float)
            significant = q < Q_HIT
            hits += int(significant.sum())
            burdens.append(10000.0 * significant.mean() if len(q) else 0.0)
            positions = np.flatnonzero(sequence_sample == sample.sample_index)
            positions = positions[sequence_clone[positions] >= 0]
            evidence[positions] = np.minimum(
                10.0,
                -np.log10(np.maximum(q[sequence_clone[positions]], 1e-10)),
            ) / 10.0
        expected = representations.loc[
            (representations["chain"] == chain)
            & (representations["subject_id"] == subject_id)
        ]
        if len(expected) != 1 or int(expected.iloc[0]["tcr_count"]) != len(evidence):
            raise ValueError(f"{split}/{chain}/{subject_id}: SCEPTR row mismatch")
        evidence_path = ALICE / "evidence" / split / chain / f"{subject_id}.npy"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(evidence_path, evidence, allow_pickle=False)
        subject = expected.iloc[0]
        records.append(
            {
                "split": split,
                "role": split,
                "chain": chain,
                "subject_id": subject_id,
                "legacy_patient_id": subject["legacy_patient_id"],
                "label": int(subject["label"]),
                "cohort": subject["cohort"],
                "evidence_file": evidence_path.relative_to(protocol.REPO).as_posix(),
                "sequence_map_file": mapping_path.relative_to(protocol.REPO).as_posix(),
                "sample_count": len(samples),
                "alice_burden": float(np.mean(burdens)) if burdens else 0.0,
                "alice_hits": hits,
                "alice_tested_clonotypes": int(samples["unique_clonotypes"].sum()),
                "alice_testable": int(samples["unique_clonotypes"].sum() > 0),
                "tcr_count": len(evidence),
                "evidence_nonzero_fraction": float(np.mean(evidence > 0)),
            }
        )
    manifest = pd.DataFrame(records)
    protocol.atomic_csv(manifest, ALICE / f"{split}_manifest.csv")
    return manifest


def audit(split: str) -> None:
    manifest = pd.read_csv(ALICE / f"{split}_manifest.csv")
    representations = protocol.read_manifest(f"{split}_representations")
    merged = manifest.merge(
        representations[["subject_id", "chain", "tcr_count"]],
        on=["subject_id", "chain"],
        suffixes=("_alice", "_sceptr"),
        validate="one_to_one",
    )
    if len(merged) != len(representations):
        raise RuntimeError(f"{split}: incomplete ALICE/representation join")
    if not merged["tcr_count_alice"].astype(int).eq(
        merged["tcr_count_sceptr"].astype(int)
    ).all():
        raise RuntimeError(f"{split}: ALICE/SCEPTR row-count mismatch")
    for row in manifest.itertuples(index=False):
        evidence = np.load(protocol.repo_path(row.evidence_file), allow_pickle=False)
        if len(evidence) != int(row.tcr_count):
            raise RuntimeError(f"{row.subject_id}/{row.chain}: invalid evidence length")
    print(f"{split}: {len(merged)} patient-chain representations verified")


def write_file_manifest(split: str) -> pd.DataFrame:
    manifest = pd.read_csv(ALICE / f"{split}_manifest.csv", dtype=str, keep_default_na=False)
    records = []
    for column in ("evidence_file", "sequence_map_file"):
        for value in manifest[column]:
            path = protocol.repo_path(value)
            records.append(
                {
                    "path": path.relative_to(protocol.REPO).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": protocol.sha256(path),
                }
            )
    result = pd.DataFrame(records).sort_values("path").reset_index(drop=True)
    if result["path"].duplicated().any() or len(result) != 2 * len(manifest):
        raise RuntimeError(f"{split}: invalid ALICE asset manifest")
    protocol.atomic_csv(result, ALICE / f"{split}_file_manifest.csv")
    return result


def write_runtime_freeze(
    splits: tuple[str, ...],
    *,
    include_descriptors: bool = False,
) -> dict:
    from analysis.experiment_04_alice_model import core, study

    payload = {"schema_version": 2, "experiments": {}, "files": {}}
    if FREEZE.exists():
        payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    checkpoint_root = protocol.REPO / "artifacts" / "checkpoints" / EXPERIMENT_ID
    checkpoint_method_ids = [method.method_id for method in core.METHODS]
    checkpoints = [
        path
        for method_id in checkpoint_method_ids
        for path in (checkpoint_root / method_id).rglob("*.pt")
    ]
    expected_core_count = sum(
        len(method.chains) * len(method.seed_ids) * 5
        for method in core.METHODS
    )
    if len(checkpoints) != expected_core_count:
        raise ValueError(
            f"Hard-gate checkpoint coverage is {len(checkpoints)}, expected {expected_core_count}"
        )
    payload.setdefault("experiments", {})[EXPERIMENT_ID] = {
        "checkpoint_method_ids": checkpoint_method_ids,
        "expected_checkpoint_count": expected_core_count,
        "checkpoint_count": len(checkpoints),
        "checkpoint_digest": protocol.file_set_digest(checkpoints),
        "model_code_files": {
            relative: protocol.sha256(protocol.REPO / relative)
            for relative in MODEL_CODE_FILES
        },
        "analysis_code_files": {
            relative: protocol.sha256(protocol.REPO / relative)
            for relative in ANALYSIS_CODE_FILES
        },
    }
    future_method_ids = [method.method_id for method in study.FUTURE_WORK_METHODS]
    future_checkpoints = [
        path
        for method_id in future_method_ids
        for path in (checkpoint_root / method_id).rglob("*.pt")
    ]
    expected_future_count = sum(
        len(method.chains) * len(method.seed_ids) * 5
        for method in study.FUTURE_WORK_METHODS
    )
    if len(future_checkpoints) != expected_future_count:
        raise ValueError(
            "Representation future-work checkpoint coverage is "
            f"{len(future_checkpoints)}, expected {expected_future_count}"
        )
    payload.setdefault("future_work", {})["representation_dependence"] = {
        "status": "preliminary_partial",
        "checkpoint_root": checkpoint_root.relative_to(protocol.REPO).as_posix(),
        "checkpoint_method_ids": future_method_ids,
        "expected_checkpoint_count": expected_future_count,
        "checkpoint_count": len(future_checkpoints),
        "checkpoint_digest": protocol.file_set_digest(future_checkpoints),
    }
    files = payload.setdefault("files", {})
    shared = ("artifacts/manifests/fivefold_seed913271.csv",)
    for relative in shared:
        files[relative] = protocol.sha256(protocol.REPO / relative)
    for split in splits:
        required = [
            f"artifacts/manifests/{split}_subjects.csv",
            f"artifacts/manifests/{split}_samples.csv",
            f"artifacts/manifests/{split}_representations.csv",
            f"artifacts/alice/{split}_manifest.csv",
            f"artifacts/alice/{split}_file_manifest.csv",
        ]
        if include_descriptors:
            required.append(
                f"artifacts/manifests/representation_comparison_{split}.csv"
            )
        for relative in required:
            files[relative] = protocol.sha256(protocol.REPO / relative)
    temporary = FREEZE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(FREEZE)
    protocol._VERIFIED_CHECKPOINTS.discard(EXPERIMENT_ID)
    for split in splits:
        protocol._VERIFIED_ALICE_ASSETS.discard(split)
    print(
        f"ALICE runtime frozen for {','.join(splits)} with {len(checkpoints)} checkpoints",
        flush=True,
    )
    return payload


def self_test() -> None:
    toy = pd.DataFrame(
        {
            "V": ["TRAV1", "TRAV1", "TRAV2", "TRAV3"],
            "J": ["TRAJ1", "TRAJ1", "TRAJ2", "TRAJ3"],
            "cdr3aa": ["CAC", "CAC", "CAS", "CAT"],
            "cdr3nt": ["TGTGCTTGT", "TGTGCTTGT", "", "TGTA"],
        }
    )
    unique, clone_index = alice_clone_mapping(toy)
    assert len(unique) == 1
    assert clone_index.tolist() == [0, 0, -1, -1]
    assert sample_key("alpha", "P1", "S1") == sample_key("alpha", "P1", "S1")
    assert WORKER.exists()
    print("experiment_04_alice_model preparation self-test passed")


def prepare(split: str, args: RuntimeOptions) -> None:
    build_inputs(split)
    run_alice(split, args.rscript, args.olga, args.workers)
    build_evidence(split)
    audit(split)
    write_file_manifest(split)
