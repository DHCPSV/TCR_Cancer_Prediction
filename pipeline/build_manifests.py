"""Rebuild derived canonical manifests and the frozen five-fold assignment."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pipeline import provenance


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts" / "manifests"
SEED = 913271


def write(frame: pd.DataFrame, name: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / name
    temporary = Path(f"{target}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)


def relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_from_name(path: Path, prefix: str) -> str:
    match = re.search(rf"{prefix}_?0*(\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot recover {prefix} subject from {path.name}")
    return f"{prefix}{int(match.group(1)):04d}" if prefix == "LTX" else f"{prefix}_{int(match.group(1)):04d}"


def sample_record(
    raw_file: Path,
    subject_id: str,
    chain: str,
    cohort: str,
    label: int,
    role: str,
    *,
    selector_column: str = "",
    selector_value: str = "",
) -> dict:
    if cohort == "internal_control":
        legacy = raw_file.stem
    elif cohort == "tx100_cancer":
        legacy = f"{subject_id}_positive_{chain}"
    elif cohort == "bcg_control":
        legacy = re.sub(r"_(alpha|beta)$", "", raw_file.name.removesuffix(".tsv.gz"), flags=re.IGNORECASE)
    else:
        legacy = subject_id
    return {
        "sample_id": f"{subject_id}_{chain}",
        "subject_id": subject_id,
        "legacy_patient_id": legacy,
        "timepoint": "pooled" if role == "internal" else "external",
        "chain": chain,
        "cohort": cohort,
        "label": label,
        "role": role,
        "raw_file": relative(raw_file),
        "raw_subject_column": selector_column,
        "raw_subject_value": selector_value,
        "tcr_table": f"artifacts/tcr_tables/{role}/{chain}/{subject_id}_{chain}.tsv",
        "raw_sha256": sha256(raw_file),
        "tcr_table_sha256": "",
    }


def internal_samples() -> pd.DataFrame:
    records = []
    for chain in ("alpha", "beta"):
        controls = sorted((REPO / "data" / "raw" / "internal" / "control" / chain).glob("*.tsv"))
        for raw_file in controls:
            match = re.search(r"UP_\d+", raw_file.name)
            if not match:
                raise ValueError(f"Cannot recover control subject from {raw_file.name}")
            records.append(
                sample_record(
                    raw_file,
                    match.group(),
                    chain,
                    "internal_control",
                    0,
                    "internal",
                )
            )

        cancer_files = list((REPO / "data" / "raw" / "internal" / "cancer" / chain).glob("*.tsv"))
        if len(cancer_files) != 1:
            raise ValueError(f"Expected one pooled TRACERx100 {chain} table, found {len(cancer_files)}")
        raw_file = cancer_files[0]
        ids = pd.read_csv(raw_file, sep="\t", usecols=["LTX_ID"], dtype=str)["LTX_ID"].str.strip()
        for subject_id in sorted(ids.dropna().unique()):
            if not re.fullmatch(r"LTX\d{4}", subject_id):
                raise ValueError(f"Unexpected TRACERx100 subject: {subject_id}")
            records.append(
                sample_record(
                    raw_file,
                    subject_id,
                    chain,
                    "tx100_cancer",
                    1,
                    "internal",
                    selector_column="LTX_ID",
                    selector_value=subject_id,
                )
            )
    frame = pd.DataFrame(records)
    counts = frame.groupby(["chain", "label"])["subject_id"].nunique().to_dict()
    if counts != {("alpha", 0): 111, ("alpha", 1): 58, ("beta", 0): 111, ("beta", 1): 58}:
        raise ValueError(f"Unexpected internal cohort counts: {counts}")
    return frame


def external_samples() -> pd.DataFrame:
    records = []
    for chain in ("alpha", "beta"):
        bcg = sorted((REPO / "data" / "raw" / "external" / "bcg_control" / chain).glob("*.tsv.gz"))
        for raw_file in bcg:
            records.append(
                sample_record(raw_file, subject_from_name(raw_file, "TCV"), chain, "bcg_control", 0, "external")
            )
        cancer = sorted((REPO / "data" / "raw" / "external" / "tx421_cancer" / chain).glob("*.tsv.gz"))
        for raw_file in cancer:
            records.append(
                sample_record(raw_file, subject_from_name(raw_file, "LTX"), chain, "tx421_cancer", 1, "external")
            )
    frame = pd.DataFrame(records)
    counts = frame.groupby(["chain", "label"])["subject_id"].nunique().to_dict()
    if counts != {("alpha", 0): 73, ("alpha", 1): 36, ("beta", 0): 74, ("beta", 1): 36}:
        raise ValueError(f"Unexpected external cohort counts: {counts}")
    return frame


def bootstrap(role: str) -> None:
    incoming = internal_samples() if role == "internal" else external_samples()
    sample_path = ROOT / "samples.csv"
    if sample_path.exists():
        previous = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        incoming = pd.concat([previous.loc[previous["role"] != role], incoming], ignore_index=True)
    incoming = incoming.sort_values(
        ["role", "chain", "label", "subject_id"], kind="stable"
    ).reset_index(drop=True)
    write(incoming, "samples.csv")

    representations = incoming.loc[incoming["role"] == role, [
        "subject_id", "legacy_patient_id", "chain", "cohort", "label", "role"
    ]].drop_duplicates().copy()
    representations["embedding_file"] = representations.apply(
        lambda row: f"artifacts/representations/{row.role}/{row.chain}/sceptr/{row.subject_id}.pt", axis=1
    )
    representations["tcr_count"] = ""
    representations["embedding_dim"] = "64"
    representations["representation_method"] = "sceptr"
    representations["representation_version"] = "sceptr-fast-tokeniser-v1"
    representations["sha256"] = ""
    representation_path = ROOT / "representations.csv"
    if representation_path.exists():
        previous = pd.read_csv(representation_path, dtype=str, keep_default_na=False)
        representations = pd.concat(
            [previous.loc[previous["role"] != role], representations],
            ignore_index=True,
        )
    write(
        representations.sort_values(
            ["role", "chain", "label", "subject_id"], kind="stable"
        ),
        "representations.csv",
    )
    print(f"bootstrapped {role}: {incoming.loc[incoming['role'] == role, 'subject_id'].nunique()} subjects")


def rebuild_views() -> None:
    samples = pd.read_csv(ROOT / "samples.csv", dtype=str, keep_default_na=False)
    representations = pd.read_csv(
        ROOT / "representations.csv", dtype=str, keep_default_na=False
    )
    identity = ["subject_id", "cohort", "label", "role"]
    conflicts = samples.groupby("subject_id")[identity[1:]].nunique()
    if (conflicts > 1).any().any():
        raise RuntimeError("A subject has inconsistent cohort, label or role")
    subjects = samples[identity].drop_duplicates("subject_id").sort_values("subject_id")
    subject_legacy = (
        samples.assign(
            subject_legacy_patient_id=samples["legacy_patient_id"].str.replace(
                r"_(alpha|beta)$", "", regex=True
            )
        )
        .groupby("subject_id")["subject_legacy_patient_id"]
        .first()
    )
    subjects.insert(
        1,
        "legacy_patient_id",
        subjects["subject_id"].map(subject_legacy),
    )
    chains = samples.assign(value=True).pivot_table(
        index="subject_id", columns="chain", values="value", aggfunc="any", fill_value=False
    )
    subjects["has_alpha"] = subjects["subject_id"].map(chains.get("alpha", False)).astype(int)
    subjects["has_beta"] = subjects["subject_id"].map(chains.get("beta", False)).astype(int)
    write(subjects, "subjects.csv")
    for role in sorted(subjects["role"].unique()):
        write(subjects.loc[subjects["role"] == role], f"{role}_subjects.csv")
        write(samples.loc[samples["role"] == role], f"{role}_samples.csv")
        write(
            representations.loc[representations["role"] == role],
            f"{role}_representations.csv",
        )


def refresh(role: str | None = None) -> dict[Path, str]:
    import torch

    samples = pd.read_csv(ROOT / "samples.csv", dtype=str, keep_default_na=False)
    sample_mask = pd.Series(True, index=samples.index) if role is None else samples["role"].eq(role)
    samples.loc[sample_mask, "tcr_table_sha256"] = ""
    digests: dict[Path, str] = {}
    for index, row in samples.loc[sample_mask].iterrows():
        table = REPO / row["tcr_table"]
        if provenance.validate_recorded_artifact(
            table, REPO, digests, "canonical_tcr_table"
        ).state == "valid":
            samples.at[index, "tcr_table_sha256"] = digests[table.resolve()]
            raw_file = (REPO / row["raw_file"]).resolve()
            samples.at[index, "raw_sha256"] = digests[raw_file]
    write(samples, "samples.csv")

    representations = pd.read_csv(ROOT / "representations.csv", dtype=str, keep_default_na=False)
    representation_mask = (
        pd.Series(True, index=representations.index)
        if role is None
        else representations["role"].eq(role)
    )
    representations.loc[representation_mask, ["tcr_count", "sha256"]] = ""
    for index, row in representations.loc[representation_mask].iterrows():
        embedding = REPO / row["embedding_file"]
        if provenance.validate_recorded_artifact(
            embedding, REPO, digests, "sceptr_embedding"
        ).state != "valid":
            continue
        tensor = torch.load(embedding, map_location="cpu", weights_only=True)
        if tensor.ndim != 2 or tensor.shape[1] != 64:
            raise ValueError(f"Unexpected SCEPTR tensor: {row['embedding_file']} {tuple(tensor.shape)}")
        representations.at[index, "tcr_count"] = str(tensor.shape[0])
        representations.at[index, "sha256"] = digests[embedding.resolve()]
    write(representations, "representations.csv")
    rebuild_views()
    return digests


def rebuild_folds() -> None:
    rows = pd.read_csv(
        ROOT / "internal_representations.csv", dtype=str, keep_default_na=False
    )
    records = []
    for chain in ("alpha", "beta"):
        cohort = rows.loc[rows["chain"] == chain].sort_values("subject_id").reset_index(drop=True)
        splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for fold, (_, held_out) in enumerate(splitter.split(cohort, cohort["label"])):
            for index in held_out:
                row = cohort.iloc[index]
                records.append(
                    {
                        "split_id": f"fivefold_seed{SEED}",
                        "seed": SEED,
                        "fold": fold,
                        "subject_id": row.subject_id,
                        "legacy_patient_id": row.legacy_patient_id,
                        "chain": chain,
                        "label": row.label,
                        "role": "internal",
                    }
                )
    write(pd.DataFrame(records), f"fivefold_seed{SEED}.csv")


def verify(
    role: str | None = None,
    digests: dict[Path, str] | None = None,
) -> None:
    subjects = pd.read_csv(ROOT / "subjects.csv", dtype=str)
    representations = pd.read_csv(ROOT / "representations.csv", dtype=str)
    if subjects["subject_id"].duplicated().any():
        raise RuntimeError("Duplicate subject_id")
    if role in (None, "internal"):
        folds = pd.read_csv(ROOT / f"fivefold_seed{SEED}.csv", dtype=str)
        internal = representations.loc[representations["role"] == "internal"]
        expected = internal[["subject_id", "chain"]].drop_duplicates()
        observed = folds[["subject_id", "chain"]].drop_duplicates()
        if set(map(tuple, expected.values)) != set(map(tuple, observed.values)):
            raise RuntimeError("Frozen folds do not cover every internal subject-chain exactly once")
        if folds[["subject_id", "chain"]].duplicated().any():
            raise RuntimeError("An internal subject-chain is held out more than once")
    selected = representations if role is None else representations.loc[representations["role"] == role]
    cache = digests if digests is not None else {}
    for row in selected.itertuples(index=False):
        if not row.tcr_count or not row.sha256:
            raise RuntimeError(f"Incomplete representation metadata: {row.subject_id}/{row.chain}")
        path = REPO / row.embedding_file
        status = provenance.validate_recorded_artifact(
            path, REPO, cache, "sceptr_embedding"
        )
        if status.state != "valid" or cache.get(path.resolve()) != row.sha256:
            raise RuntimeError(f"Representation missing or changed: {row.embedding_file}")
    print(
        f"canonical manifests OK ({role or 'all'}): {len(selected)} representations"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("bootstrap-internal", "bootstrap-external", "views", "folds", "refresh", "verify", "all"),
        default="verify",
    )
    stage = parser.parse_args().stage
    if stage == "bootstrap-internal":
        bootstrap("internal")
    if stage == "bootstrap-external":
        bootstrap("external")
    if stage in {"views", "all"}:
        rebuild_views()
    if stage in {"folds", "all"}:
        rebuild_folds()
    if stage in {"refresh", "all"}:
        refresh()
    if stage in {"verify", "all"}:
        verify()


if __name__ == "__main__":
    main()
