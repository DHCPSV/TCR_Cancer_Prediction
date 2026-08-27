"""Create frozen N x 64 SCEPTR tensors from canonical patient TCR tables."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch

from pipeline import provenance

if __package__:
    from .sceptr_adapter import calc_vector_representations
else:
    from sceptr_adapter import calc_vector_representations


REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "artifacts" / "manifests"


def path(text: str) -> Path:
    value = Path(text)
    return value if value.is_absolute() else REPO / value


def model_fingerprint(model) -> str:
    module = getattr(model, "_bert", model)
    if not hasattr(module, "state_dict"):
        raise TypeError(f"Cannot fingerprint SCEPTR model of type {type(model).__name__}")
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(name.encode("utf-8"))
        value.update(str(tensor.dtype).encode("ascii"))
        value.update(str(tuple(tensor.shape)).encode("ascii"))
        value.update(tensor.view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def _table_contract(
    tables: list[str],
    samples: pd.DataFrame,
    digests: dict[Path, str],
) -> list[dict]:
    inputs = []
    for ordinal, table_text in enumerate(tables):
        table = path(table_text)
        status = provenance.validate_recorded_artifact(
            table, REPO, digests, "canonical_tcr_table"
        )
        if status.state != "valid":
            raise RuntimeError(f"Invalid TCR-table lineage for {table_text}: {status.reason}")
        actual = digests[table.resolve()]
        inputs.append(
            provenance.input_record(
                table,
                REPO,
                digest=actual,
                ordinal=ordinal,
            )
        )
    return inputs


def run(role: str, chain: str, batch_size: int, overwrite: bool) -> dict[str, int]:
    import sceptr

    samples = pd.read_csv(
        MANIFESTS / f"{role}_samples.csv", dtype=str, keep_default_na=False
    )
    representations = pd.read_csv(
        MANIFESTS / f"{role}_representations.csv", dtype=str, keep_default_na=False
    )
    samples = samples.loc[samples["chain"] == chain]
    representations = representations.loc[representations["chain"] == chain]
    model = sceptr.variant.default()
    model.set_batch_size(batch_size)
    model_sha256 = model_fingerprint(model)
    rows = list(representations.itertuples(index=False))
    progress = provenance.Progress(f"embed SCEPTR {role}/{chain}", len(rows))
    counts = {"reused": 0, "rebuilt": 0}
    digests: dict[Path, str] = {}

    for row in rows:
        output = path(row.embedding_file)
        tables = (
            samples.loc[samples["subject_id"] == row.subject_id, "tcr_table"]
            .drop_duplicates()
            .tolist()
        )
        if not tables:
            raise RuntimeError(f"No TCR table for {row.subject_id} {chain}")
        inputs = _table_contract(tables, samples, digests)
        specification = provenance.artifact_specification(
            "sceptr_embedding",
            inputs,
            {
                "role": role,
                "chain": chain,
                "subject_id": row.subject_id,
                "batch_size": batch_size,
                "embedding_dim": 64,
                "representation_method": row.representation_method,
                "representation_version": row.representation_version,
                "model_variant": "sceptr.variant.default",
                "model_state_sha256": model_sha256,
            },
            (Path(__file__), Path(calc_vector_representations.__code__.co_filename)),
            ("pandas", "torch", "sceptr", "libtcrlm"),
            REPO,
        )
        status = provenance.inspect_artifact(output, specification)
        if not overwrite and status.state == "valid":
            counts["reused"] += 1
            progress.advance()
            continue
        tcr = pd.concat(
            [pd.read_csv(path(table), sep="\t", dtype=str, keep_default_na=False) for table in tables],
            ignore_index=True,
        )
        embedding = calc_vector_representations(tcr, model)
        if tuple(embedding.shape) != (len(tcr), 64):
            raise RuntimeError(f"Unexpected embedding shape for {row.subject_id}: {embedding.shape}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{output}.tmp")
        torch.save(embedding.cpu(), temporary)
        temporary.replace(output)
        provenance.write_sidecar(
            output,
            specification,
            REPO,
            rows=len(tcr),
            shape=[len(tcr), 64],
        )
        counts["rebuilt"] += 1
        progress.advance()
    progress.summary(**counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("internal", "external"), required=True)
    parser.add_argument("--chain", choices=("alpha", "beta"), required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.role, args.chain, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
