from __future__ import annotations

from functools import lru_cache

import pandas as pd
import torch
from libtcrlm import schema
from libtcrlm.tokeniser.token_indices import (
    AminoAcidTokenIndex,
    CdrCompartmentIndex,
    DefaultTokenIndex,
)
from torch.nn import utils as torch_utils
import tidytcells as tt


REQUIRED_COLUMNS = ["TRAV", "TRAJ", "CDR3A", "TRBV", "TRBJ", "CDR3B"]
VALID_AMINO_ACIDS = {
    name: int(member)
    for name, member in AminoAcidTokenIndex.__members__.items()
    if name not in {"NULL", "MASK", "CLS"}
}


def calc_vector_representations(instances: pd.DataFrame, model) -> torch.Tensor:
    normalised = normalise(instances)
    validate(normalised)
    validate_tokens(normalised.head(64), model)
    representations = []
    model._bert.eval()
    with torch.no_grad():
        for start in range(0, len(normalised), model._batch_size):
            batch = normalised.iloc[start : start + model._batch_size]
            tokens = [tokenise(row) for row in batch.itertuples(index=False)]
            padded = torch_utils.rnn.pad_sequence(
                tokens,
                batch_first=True,
                padding_value=int(DefaultTokenIndex.NULL),
            )
            representations.append(
                model._bert.get_vector_representations_of(padded.to(model._device)).cpu()
            )
    if not representations:
        raise ValueError("No TCR rows available for SCEPTR embedding")
    result = torch.concatenate(representations).float().cpu()
    if result.shape != (len(normalised), 64) or not torch.isfinite(result).all():
        raise ValueError(f"Invalid SCEPTR output: {tuple(result.shape)}")
    return result


def normalise(instances: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in instances]
    if missing:
        raise ValueError(f"Missing SCEPTR columns: {missing}")
    result = instances[REQUIRED_COLUMNS].copy()
    return result.map(
        lambda value: None
        if pd.isna(value) or str(value).strip() == ""
        else str(value).strip()
    )


def validate(instances: pd.DataFrame) -> None:
    for symbol in sorted(set(instances["TRAV"].dropna())):
        v_cdrs(symbol, "A")
    for symbol in sorted(set(instances["TRBV"].dropna())):
        v_cdrs(symbol, "B")
    validate_junctions(instances, "CDR3A", "TRAJ")
    validate_junctions(instances, "CDR3B", "TRBJ")


def validate_junctions(instances: pd.DataFrame, cdr3: str, j_gene: str) -> None:
    present = instances[cdr3].notna()
    for index, sequence in instances.loc[present, cdr3].items():
        if (
            not sequence
            or sequence[0] != "C"
            or sequence[-1] not in "FWC"
            or not set(sequence) <= set(VALID_AMINO_ACIDS)
        ):
            raise ValueError(f"Invalid {cdr3} at row {index}: {sequence}")
    checks = (
        instances.loc[present, [j_gene, cdr3]]
        .assign(_terminal=instances.loc[present, cdr3].str[-1])
        .drop_duplicates([j_gene, "_terminal"])
    )
    for row in checks.itertuples(index=False):
        j_symbol, sequence, _ = row
        standardised = tt.junction.standardize(
            sequence,
            j_symbol=j_symbol,
            fix_missing_conserved=True,
            log_failures=False,
        )
        if standardised != sequence:
            raise ValueError(f"{cdr3} is not standardised for {j_symbol}: {sequence}")


def validate_tokens(instances: pd.DataFrame, model) -> None:
    official = schema.generate_tcr_series(instances)
    for row, tcr in zip(instances.itertuples(index=False), official):
        if not torch.equal(model._tokeniser.tokenise(tcr), tokenise(row)):
            raise RuntimeError("Direct SCEPTR tokens disagree with the official tokeniser")


def tokenise(row) -> torch.Tensor:
    cdr1a, cdr2a = v_cdrs(row.TRAV, "A") if row.TRAV else ("", "")
    cdr1b, cdr2b = v_cdrs(row.TRBV, "B") if row.TRBV else ("", "")
    tokens = [(int(AminoAcidTokenIndex.CLS), 0, 0, int(CdrCompartmentIndex.NULL))]
    tokens += sequence_tokens(cdr1a, int(CdrCompartmentIndex.CDR1A))
    tokens += sequence_tokens(cdr2a, int(CdrCompartmentIndex.CDR2A))
    tokens += sequence_tokens(row.CDR3A, int(CdrCompartmentIndex.CDR3A))
    tokens += sequence_tokens(cdr1b, int(CdrCompartmentIndex.CDR1B))
    tokens += sequence_tokens(cdr2b, int(CdrCompartmentIndex.CDR2B))
    tokens += sequence_tokens(row.CDR3B, int(CdrCompartmentIndex.CDR3B))
    return torch.tensor(tokens, dtype=torch.long)


@lru_cache(maxsize=None)
def v_cdrs(symbol: str, chain: str) -> tuple[str, str]:
    if chain == "A":
        tcr = schema.make_tcr_from_components(symbol, "CAAAAF", None, None)
        return tcr.cdr1a_sequence or "", tcr.cdr2a_sequence or ""
    tcr = schema.make_tcr_from_components(None, None, symbol, "CAAAAF")
    return tcr.cdr1b_sequence or "", tcr.cdr2b_sequence or ""


def sequence_tokens(sequence: str | None, compartment: int) -> list[tuple[int, int, int, int]]:
    if not sequence:
        return []
    return [
        (VALID_AMINO_ACIDS[amino_acid], position, len(sequence), compartment)
        for position, amino_acid in enumerate(sequence, start=1)
    ]
