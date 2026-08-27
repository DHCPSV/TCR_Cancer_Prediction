from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis import protocol, reporting
from analysis.experiment_04_alice_model.families import (
    FAMILY_ARTIFACTS,
    FAMILY_RANKS,
    REFERENCE_SEED_VALUE,
)
from analysis.experiment_04_alice_model.methods import RESULTS


VDJDB_RELEASE = "2026-05-16"
VDJDB_ROOT = protocol.REPO / "third_party" / "vdjdb"
VDJDB_SOURCE = VDJDB_ROOT / "source.csv"
VDJDB_TABLE = (
    VDJDB_ROOT
    / VDJDB_RELEASE
    / "release"
    / f"vdjdb-{VDJDB_RELEASE}"
    / "vdjdb.slim.txt"
)

LOOKUP_TABLE = FAMILY_ARTIFACTS / "vdjdb_lookup.csv"
INTERSECTION_TABLE = FAMILY_ARTIFACTS / "family_set_intersections.csv"
SUMMARY_TABLE = RESULTS / "tables" / "vdjdb_validation_summary.csv"
FIGURE = RESULTS / "figures" / "vdjdb_validation.png"

FAMILY_ID = [
    "subject_id",
    "sample_index",
    "bestVGene",
    "bestJGene",
    "center_cdr3aa",
]


def gene_names(value: object) -> set[str]:
    return {
        item.strip().split("*")[0].rstrip("_")
        for item in str(value).split(",")
        if item.strip()
    }


def family_sequences(row: object) -> list[str]:
    sequences = [str(row.center_cdr3aa), *str(row.member_cdr3aa).split(";")]
    return list(dict.fromkeys(sequence for sequence in sequences if sequence))


def load_ranked(name: str, expected_rows: int) -> pd.DataFrame:
    path = FAMILY_RANKS / name
    if not path.exists():
        raise FileNotFoundError(
            "Generate the Experiment 04 family analysis with "
            f"`python -m analysis.experiment_04_alice_model.run` first: {path}"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(frame) != expected_rows or frame[FAMILY_ID].duplicated().any():
        raise ValueError(
            f"Expected {expected_rows} unique patient-local families in {path}"
        )
    return frame


def load_vdjdb() -> pd.DataFrame:
    if not VDJDB_SOURCE.exists() or not VDJDB_TABLE.exists():
        raise FileNotFoundError(
            f"The fixed VDJdb {VDJDB_RELEASE} snapshot is required under {VDJDB_ROOT}"
        )
    source = pd.read_csv(VDJDB_SOURCE, dtype=str, keep_default_na=False)
    record = source.loc[source["release"] == VDJDB_RELEASE]
    if len(record) != 1:
        raise ValueError(f"Missing unique VDJdb provenance row for {VDJDB_RELEASE}")
    expected_sha = record.iloc[0]["derived_table_sha256"]
    observed_sha = protocol.sha256(VDJDB_TABLE)
    if observed_sha != expected_sha:
        raise ValueError(
            f"VDJdb SHA-256 mismatch: expected {expected_sha}, observed {observed_sha}"
        )
    frame = pd.read_csv(VDJDB_TABLE, sep="\t", dtype=str, keep_default_na=False)
    return frame.loc[
        (frame["gene"] == "TRA")
        & (frame["species"] == "HomoSapiens")
        & frame["cdr3"].ne("")
    ].copy()


def vdjdb_index(database: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, record in database.iterrows():
        index[record["cdr3"]].append(record.to_dict())
    return dict(index)


def lookup_family(row: object, index: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    target_v = str(row.bestVGene).split("*")[0]
    target_j = str(row.bestJGene).split("*")[0]
    center = str(row.center_cdr3aa)
    matches: list[tuple[str, dict[str, str]]] = []
    for sequence in family_sequences(row):
        for record in index.get(sequence, []):
            if (
                target_v in gene_names(record["v.segm"])
                and target_j in gene_names(record["j.segm"])
            ):
                matches.append((sequence, record))
    matched_sequences = list(dict.fromkeys(sequence for sequence, _ in matches))
    references = sorted(
        {
            record["reference.id"].strip()
            for _, record in matches
            if record["reference.id"].strip()
        }
    )
    if not matches:
        basis = "no strict match"
    elif center in matched_sequences:
        basis = "representative CDR3A + V/J"
    else:
        basis = "family-member CDR3A + V/J"
    return {
        "found_in_vdjdb": bool(matches),
        "match_basis": basis,
        "matched_cdr3a": ";".join(matched_sequences),
        "vdjdb_record_count": len(matches),
        "linked_study_record": bool(references),
    }


def family_keys(frame: pd.DataFrame) -> pd.Series:
    return frame[FAMILY_ID].astype(str).agg("|".join, axis=1)


def build_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    smallest = load_ranked("top_100_smallest_alice_q.csv", 100).copy()
    cancer = load_ranked("top_50_pushes_cancer.csv", 50).copy()
    control = load_ranked("top_50_pushes_control.csv", 50).copy()

    smallest["family_key"] = family_keys(smallest)
    cancer["family_key"] = family_keys(cancer)
    control["family_key"] = family_keys(control)
    cancer_ranks = cancer.set_index("family_key")["rank_within_type"].to_dict()
    control_ranks = control.set_index("family_key")["rank_within_type"].to_dict()

    index = vdjdb_index(load_vdjdb())
    smallest_ranks = smallest.set_index("family_key")["rank_within_type"].to_dict()
    union = (
        pd.concat([smallest, cancer, control], ignore_index=True)
        .drop_duplicates("family_key", keep="first")
        .reset_index(drop=True)
    )
    matches = pd.DataFrame(
        [lookup_family(row, index) for row in union.itertuples(index=False)]
    )
    union = pd.concat([union, matches], axis=1)
    union["in_alice_q_top100"] = union["family_key"].isin(smallest_ranks)
    union["alice_q_rank"] = union["family_key"].map(smallest_ranks).fillna("")
    union["in_cancer_directed_top50"] = union["family_key"].isin(cancer_ranks)
    union["cancer_contribution_rank"] = union["family_key"].map(cancer_ranks).fillna("")
    union["in_control_directed_top50"] = union["family_key"].isin(control_ranks)
    union["control_contribution_rank"] = union["family_key"].map(control_ranks).fillna("")
    if (union["in_cancer_directed_top50"] & union["in_control_directed_top50"]).any():
        raise ValueError("Cancer- and control-directed Top-50 sets unexpectedly overlap")

    q_group = union["in_alice_q_top100"]
    matched = union["found_in_vdjdb"]
    cancer_group = union["in_cancer_directed_top50"]
    control_group = union["in_control_directed_top50"]
    neither_group = q_group & ~(cancer_group | control_group)
    summary_rows = [
        ("ALICE q Top100", "VDJdb match", q_group & matched, 100),
        ("ALICE q Top100", "No VDJdb match", q_group & ~matched, 100),
        ("Cancer-directed Top50", "In q Top100 + VDJdb match", cancer_group & q_group & matched, 50),
        ("Cancer-directed Top50", "In q Top100 + no VDJdb match", cancer_group & q_group & ~matched, 50),
        ("Cancer-directed Top50", "Outside q Top100 + VDJdb match", cancer_group & ~q_group & matched, 50),
        ("Cancer-directed Top50", "Outside q Top100 + no VDJdb match", cancer_group & ~q_group & ~matched, 50),
        ("Control-directed Top50", "In q Top100 + VDJdb match", control_group & q_group & matched, 50),
        ("Control-directed Top50", "In q Top100 + no VDJdb match", control_group & q_group & ~matched, 50),
        ("Control-directed Top50", "Outside q Top100 + VDJdb match", control_group & ~q_group & matched, 50),
        ("Control-directed Top50", "Outside q Top100 + no VDJdb match", control_group & ~q_group & ~matched, 50),
        ("ALICE q Top100: neither direction", "VDJdb match", neither_group & matched, 100),
        ("ALICE q Top100: neither direction", "No VDJdb match", neither_group & ~matched, 100),
    ]

    union_output = union.rename(
        columns={
            "center_cdr3aa": "representative_cdr3a",
            "subject_id": "patient_id",
            "bestVGene": "v_gene",
            "bestJGene": "j_gene",
            "min_q": "minimum_q",
            "linked_study_record": "has_linked_study",
        }
    )
    union_output = union_output[
        [
            "representative_cdr3a",
            "patient_id",
            "v_gene",
            "j_gene",
            "minimum_q",
            "in_alice_q_top100",
            "alice_q_rank",
            "in_cancer_directed_top50",
            "cancer_contribution_rank",
            "in_control_directed_top50",
            "control_contribution_rank",
            "found_in_vdjdb",
            "match_basis",
            "matched_cdr3a",
            "has_linked_study",
        ]
    ]

    output = union.loc[q_group].copy()
    output["model_overlap"] = "neither contribution Top50"
    output.loc[output["in_cancer_directed_top50"], "model_overlap"] = "cancer-directed Top50"
    output.loc[output["in_control_directed_top50"], "model_overlap"] = "control-directed Top50"
    output = output.rename(columns={"rank_within_type": "q_rank"})
    output = output.rename(
        columns={
            "center_cdr3aa": "representative_cdr3a",
            "subject_id": "patient_id",
            "bestVGene": "v_gene",
            "bestJGene": "j_gene",
            "min_q": "minimum_q",
            "linked_study_record": "has_linked_study",
        }
    )
    output = output[
        [
            "q_rank",
            "representative_cdr3a",
            "patient_id",
            "v_gene",
            "j_gene",
            "minimum_q",
            "found_in_vdjdb",
            "match_basis",
            "matched_cdr3a",
            "has_linked_study",
            "model_overlap",
        ]
    ]

    summary = pd.DataFrame(
        [
            {
                "set": set_name,
                "intersection": intersection,
                "count": int(mask.sum()),
                "denominator": denominator,
            }
            for set_name, intersection, mask, denominator in summary_rows
        ]
    )
    summary["fraction"] = summary["count"] / summary["denominator"]
    return output, union_output, summary


def plot_summary(summary: pd.DataFrame) -> None:
    def count(set_name: str, intersection: str) -> int:
        row = summary.loc[
            (summary["set"] == set_name)
            & (summary["intersection"] == intersection),
            "count",
        ]
        if len(row) != 1:
            raise ValueError(f"Missing unique summary row: {set_name} / {intersection}")
        return int(row.iloc[0])

    reporting.configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
    axes = axes.ravel()
    panels = [
        (
            axes[0],
            [
                count("ALICE q Top100", "VDJdb match"),
                count("ALICE q Top100", "No VDJdb match"),
            ],
            ["Strict VDJdb match", "No strict match"],
            ["#2a9d8f", "#d9dee7"],
            "ALICE q-value Top 100: VDJdb coverage",
            100,
        ),
        (
            axes[1],
            [
                count("Cancer-directed Top50", "In q Top100 + VDJdb match"),
                count("Cancer-directed Top50", "In q Top100 + no VDJdb match"),
                count("Control-directed Top50", "In q Top100 + VDJdb match"),
                count("Control-directed Top50", "In q Top100 + no VDJdb match"),
                count("ALICE q Top100: neither direction", "VDJdb match"),
                count("ALICE q Top100: neither direction", "No VDJdb match"),
            ],
            [
                "Cancer Top50 + VDJdb match",
                "Cancer Top50 + no match",
                "Control Top50 + VDJdb match",
                "Control Top50 + no match",
                "Neither Top50 + VDJdb match",
                "Neither Top50 + no match",
            ],
            ["#991b1b", "#fca5a5", "#1e3a8a", "#93c5fd", "#4b5563", "#d9dee7"],
            "ALICE q-value Top 100: directional overlap",
            100,
        ),
        (
            axes[2],
            [
                count("Cancer-directed Top50", "In q Top100 + VDJdb match"),
                count("Cancer-directed Top50", "In q Top100 + no VDJdb match"),
                count("Cancer-directed Top50", "Outside q Top100 + VDJdb match"),
                count("Cancer-directed Top50", "Outside q Top100 + no VDJdb match"),
            ],
            [
                "In q Top100 + VDJdb match",
                "In q Top100 + no VDJdb match",
                "Outside q Top100 + VDJdb match",
                "Outside q Top100 + no VDJdb match",
            ],
            ["#991b1b", "#dc2626", "#f87171", "#fecaca"],
            "Cancer-directed Top 50",
            50,
        ),
        (
            axes[3],
            [
                count("Control-directed Top50", "In q Top100 + VDJdb match"),
                count("Control-directed Top50", "In q Top100 + no VDJdb match"),
                count("Control-directed Top50", "Outside q Top100 + VDJdb match"),
                count("Control-directed Top50", "Outside q Top100 + no VDJdb match"),
            ],
            [
                "In q Top100 + VDJdb match",
                "In q Top100 + no VDJdb match",
                "Outside q Top100 + VDJdb match",
                "Outside q Top100 + no VDJdb match",
            ],
            ["#1e3a8a", "#2563eb", "#60a5fa", "#bfdbfe"],
            "Control-directed Top 50",
            50,
        ),
    ]
    for panel_index, (axis, values, labels, colors, title, total) in enumerate(panels):
        left = 0
        for value, label, color in zip(values, labels, colors):
            axis.barh([0], [value], left=left, height=0.42, color=color, label=label)
            if value:
                label_y = 0.29 if value <= 3 else 0
                axis.text(
                    left + value / 2,
                    label_y,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    clip_on=False,
                )
            left += value
        axis.set(
            xlim=(0, total),
            ylim=(-0.35, 0.45),
            yticks=[],
            xlabel="Number of families",
            title=title,
        )
        axis.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=1 if panel_index == 0 else 2,
            fontsize=9,
        )
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.97,
        bottom=0.12,
        hspace=0.92,
        wspace=0.25,
    )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def report() -> None:
    output, intersections, summary = build_results()
    protocol.atomic_csv(output, LOOKUP_TABLE)
    protocol.atomic_csv(intersections, INTERSECTION_TABLE)
    protocol.atomic_csv(summary, SUMMARY_TABLE)
    plot_summary(summary)
    q_rows = summary.loc[summary["set"] == "ALICE q Top100"]
    cancer_rows = summary.loc[summary["set"] == "Cancer-directed Top50"]
    control_rows = summary.loc[summary["set"] == "Control-directed Top50"]
    print(f"VDJdb strict matches in ALICE q Top100: {q_rows.loc[q_rows['intersection'] == 'VDJdb match', 'count'].iloc[0]}/100")
    print(f"Cancer-directed Top50 partition: {cancer_rows['count'].tolist()}")
    print(f"Control-directed Top50 partition: {control_rows['count'].tolist()}")


def self_test() -> None:
    assert gene_names("TRAV1-2*01,TRAV1-2*02") == {"TRAV1-2"}
    example = type(
        "Family",
        (),
        {"center_cdr3aa": "CAVA", "member_cdr3aa": "CAVA;CAVB;CAVB"},
    )()
    assert family_sequences(example) == ["CAVA", "CAVB"]
    print("VDJdb lookup self-test passed")
