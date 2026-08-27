from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tidytcells as tt

from pipeline import provenance


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artifacts" / "manifests" / "samples.csv"
SCEPTR_COLUMNS = ["TRAV", "TRAJ", "CDR3A", "TRBV", "TRBJ", "CDR3B"]
CHAIN = {
    "alpha": ("TRAV", "TRAJ", "CDR3A", "TRA"),
    "beta": ("TRBV", "TRBJ", "CDR3B", "TRB"),
}


def path(text: str) -> Path:
    value = Path(text)
    return value if value.is_absolute() else REPO / value


def canonical_gene(value: str, prefix: str) -> str | None:
    value = str(value).strip()
    if not value.startswith(prefix):
        return None
    try:
        value = str(
            tt.tr.standardize(value, enforce_functional=True, log_failures=False)
        ).strip()
    except Exception:
        return None
    return value if value.startswith(prefix) else None


def select_subject(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if not column:
        return frame
    if column not in frame:
        raise ValueError(f"Raw table is missing subject selector column {column!r}")
    return frame.loc[frame[column].astype(str).str.strip() == value]


def clean_airr_rows(
    raw_file: Path,
    selector_column: str,
    selector_value: str,
    chain: str,
) -> pd.DataFrame:
    frame = pd.read_csv(raw_file, sep="\t", dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame]
    frame = select_subject(frame, selector_column, selector_value)
    required = {"v_call", "j_call", "junction_aa"}
    if missing := required - set(frame):
        raise ValueError(f"{raw_file} is missing {sorted(missing)}")
    if "productive" in frame:
        productive = frame["productive"].str.strip().str.upper()
        frame = frame.loc[productive.isin({"T", "TRUE", "1"})]

    _, _, _, prefix = CHAIN[chain]
    v = frame["v_call"].map(lambda value: canonical_gene(value, prefix + "V"))
    j = frame["j_call"].map(lambda value: canonical_gene(value, prefix + "J"))
    cdr3 = frame["junction_aa"].astype(str).str.strip().replace("", pd.NA)
    keep = v.notna() & j.notna() & cdr3.notna()
    nucleotide = (
        frame["junction"].astype(str).str.strip()
        if "junction" in frame
        else pd.Series("", index=frame.index, dtype=str)
    )
    return pd.DataFrame(
        {
            "v_gene": v.loc[keep].tolist(),
            "j_gene": j.loc[keep].tolist(),
            "cdr3_aa": cdr3.loc[keep].tolist(),
            "cdr3_nt": nucleotide.loc[keep].tolist(),
        }
    )


def convert(
    raw_file: Path,
    selector_column: str,
    selector_value: str,
    chain: str,
) -> pd.DataFrame:
    cleaned = clean_airr_rows(raw_file, selector_column, selector_value, chain)
    v_column, j_column, cdr3_column, _ = CHAIN[chain]
    keep = len(cleaned)
    output = pd.DataFrame("", index=range(keep), columns=SCEPTR_COLUMNS)
    output[v_column] = cleaned["v_gene"]
    output[j_column] = cleaned["j_gene"]
    output[cdr3_column] = cleaned["cdr3_aa"]
    return output


def _source_contract(
    group: pd.DataFrame,
    chain: str,
    digests: dict[Path, str],
) -> tuple[list[tuple], dict]:
    sources = []
    seen = set()
    inputs = []
    for row in group.itertuples(index=False):
        selector_column = getattr(row, "raw_subject_column", "")
        selector_value = getattr(row, "raw_subject_value", "")
        key = (row.raw_file, selector_column, selector_value)
        if key in seen:
            continue
        seen.add(key)
        raw_file = path(row.raw_file)
        if raw_file not in digests:
            digests[raw_file] = provenance.sha256(raw_file)
        actual = digests[raw_file]
        sources.append((raw_file, selector_column, selector_value))
        inputs.append(
            provenance.input_record(
                raw_file,
                REPO,
                digest=actual,
                ordinal=len(inputs),
                selector_column=selector_column,
                selector_value=selector_value,
            )
        )
    specification = provenance.artifact_specification(
        "canonical_tcr_table",
        inputs,
        {
            "chain": chain,
            "output_columns": SCEPTR_COLUMNS,
            "productive_filter": "T|TRUE|1 when productive is present",
            "functional_gene_filter": True,
        },
        (Path(__file__),),
        ("pandas", "tidytcells"),
        REPO,
    )
    return sources, specification


def run(role: str, chain: str, overwrite: bool) -> dict[str, int]:
    rows = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    rows = rows.loc[(rows["role"] == role) & (rows["chain"] == chain)]
    groups = list(rows.groupby("tcr_table", sort=True))
    progress = provenance.Progress(f"prepare {role}/{chain}", len(groups))
    counts = {"reused": 0, "rebuilt": 0}
    digests: dict[Path, str] = {}
    for target_text, group in groups:
        target = path(target_text)
        sources, specification = _source_contract(group, chain, digests)
        status = provenance.inspect_artifact(target, specification)
        if not overwrite and status.state == "valid":
            counts["reused"] += 1
            progress.advance()
            continue
        pieces = []
        for raw_file, selector_column, selector_value in sources:
            pieces.append(
                convert(raw_file, selector_column, selector_value, chain)
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        result = pd.concat(pieces, ignore_index=True)
        temporary = Path(f"{target}.tmp")
        result.to_csv(temporary, sep="\t", index=False)
        temporary.replace(target)
        provenance.write_sidecar(target, specification, REPO, rows=len(result))
        counts["rebuilt"] += 1
        progress.advance()
    progress.summary(**counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("internal", "external"), required=True)
    parser.add_argument("--chain", choices=("alpha", "beta"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.role, args.chain, args.overwrite)


if __name__ == "__main__":
    main()
