# TCR repertoire classification with ALICE-informed pooling

This repository contains four experiments for patient-level cancer
classification from T-cell receptor (TCR) repertoires.

| Experiment | Data boundary | Question |
|---|---|---|
| `experiment_01_internal_baseline` | Internal cohorts | Which per-sequence representation should be carried forward? |
| `experiment_02_seed_and_attention_normalisation` | Internal cohorts | Where does seed sensitivity arise, and how do attention normalisers differ? |
| `experiment_03_external_generalisation` | Frozen external transfer | Do internally successful models transfer to cohorts from different sources? |
| `experiment_04_alice_model` | Internal training and frozen external transfer | Can ALICE-informed pooling improve transfer? |

The common analysis path is:

```text
raw AIRR repertoires
  -> standardised patient-chain TCR tables
  -> frozen per-sequence representations
  -> attention or ALICE-informed patient pooling
  -> Linear / MLP classifier
  -> patient-level five-fold OOF evaluation
  -> frozen external transfer stress test
```

## Scientific boundary

- Internal evaluation uses 111 non-cancer controls and 58 lung-cancer
  patients. Each patient has one held-out prediction under
  `fivefold_seed913271`.
- The external stress test contains 73 Alpha and 74 Beta BCG/TCV controls and
  36 additional TRACERx PBMC cancer patients. Subject sets do not overlap.
- Each external score is the mean prediction from five frozen fold models.
  External data are not used to select methods, seeds, epochs or thresholds.
- Source and label are confounded in the external cohorts. External results
  measure cohort transfer and are not independent evidence of a cancer-causal
  biomarker.
- `tx421_cancer` is a stable code alias for the 36 additional TRACERx PBMC
  patients outside the internal TRACERx100 subset; it is not a claim that the
  files belong to the published TRACERx421 cohort.
- ALICE L1 and L2 are pooling normalisations, not regularisers. L2 retains
  information about effective selected-TCR count through vector magnitude.

## Repository layout

```text
data/raw/                    local source repertoires (not committed)
artifacts/manifests/         subject, sample, representation and fold identities
artifacts/tcr_tables/        standardised patient-chain TCR tables
artifacts/representations/   per-sequence tensors
artifacts/alice/             ALICE p/q/D outputs, evidence and sequence maps
artifacts/checkpoints/       fold models and training histories
artifacts/runs/              patient predictions and detailed run artifacts
third_party/alice/           pinned TCRgrapher, OLGA models and Windows adapter
third_party/vdjdb/           fixed VDJdb snapshot used by family lookup
pipeline/                    preparation, embedding, manifests and provenance
analysis/                    experiment methods, studies, runners and reports
results/                     aggregate tables and publication figures
```

Shared analysis code has three responsibilities:

- `analysis/protocol.py`: paths, schemas, seeds, manifests, hashes and atomic
  writes;
- `analysis/training.py`: training, prediction, checkpoints and histories;
- `analysis/reporting.py`: metrics, plotting conventions and result checks.

Each experiment has one command-line entry point, `run.py`. Model definitions
remain in `methods.py`, experiment computations in `study.py` or a named study
module, and figure/table generation in `report.py`. The full module and stage
map is in `analysis/README.md`.

## Environment

The supported environment is Windows 11, 64-bit CPython 3.12, an NVIDIA GPU
compatible with CUDA 12.8, and native 64-bit R 4.3.3.

Create the Python environment from the repository root:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "setuptools>=68,<82"
python -m pip install -r requirements-cuda.txt
python -m pip install -e . --no-deps
```

`requirements-cuda.txt` pins the CUDA 12.8 PyTorch wheel. Verify the runtime
before a full run:

```powershell
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); assert sys.version_info[:2] == (3, 12) and torch.cuda.is_available()"
```

Install R 4.3.3 at `C:\Program Files\R\R-4.3.3` and Rtools43 at
`C:\rtools43`. Then install the ALICE runtime dependencies and the pinned
TCRgrapher source:

```powershell
Rscript -e "install.packages('remotes', repos='https://cloud.r-project.org')"
Rscript -e "remotes::install_version('data.table', version='1.17.8', repos='https://cloud.r-project.org'); remotes::install_version('stringdist', version='0.9.15', repos='https://cloud.r-project.org'); remotes::install_version('iterators', version='1.0.14', repos='https://cloud.r-project.org'); remotes::install_version('foreach', version='1.5.2', repos='https://cloud.r-project.org'); remotes::install_version('doParallel', version='1.0.17', repos='https://cloud.r-project.org')"
R.exe CMD INSTALL third_party\alice\tcrgrapher
Rscript -e "library(tcrgrapher); library(data.table); cat('ALICE R runtime ready\n')"
```

## Data placement

Raw patient data are excluded from Git. Extract approved cohorts without
renaming subject or chain files:

```text
data/raw/
  internal/control/{alpha,beta}/
  internal/cancer/{alpha,beta}/
  external/bcg_control/{alpha,beta}/
  external/tx421_cancer/{alpha,beta}/
```

`data/README.md` documents the expected files. The preparation pipeline writes
patient identities and repository-relative source paths to
`artifacts/manifests/`; training does not infer labels from filenames.

## Running the experiments

Install the project in editable mode, run from the repository root, and keep
the scientific order:

```powershell
python -m analysis.experiment_01_internal_baseline.run
python -m analysis.experiment_02_seed_and_attention_normalisation.run
python -m analysis.experiment_03_external_generalisation.run
python -m analysis.experiment_04_alice_model.run
```

`--stage all` is the default. Experiments 01 and 02 never prepare or read the
external cohorts. Experiment 03 requires the frozen Experiment 02 models.

| Stage boundary | Meaning |
|---|---|
| `all` | Results used by the four main experiments |
| `diagnostics` | Standard training diagnostics; Experiment 02 includes the optional 300-epoch selected-seed trajectory |
| `future-work` | Preliminary Experiment 04 representation-dependence and gate-free PCA analyses |
| `self-test` | Fast installation and contract checks without training |

Focused recovery stages such as `prepare`, `internal`, `report`, `assemble`,
`geometry`, `freeze` and `validation` are available through the same four
runners. Implementation modules are not standalone commands.

Typical full runtimes on the development workstation (Windows 11, RTX 3060)
are planning estimates:

| Experiment | From required raw inputs | Valid-cache `all` |
|---|---:|---:|
| Experiment 1 | 1.7--2.3 h | 5--15 min |
| Experiment 2 | 2.4--3.3 h | 15--35 min |
| Experiment 3 | 0.5--1.0 h after Experiment 2 | 10--30 min |
| Experiment 4 | 13--18 h | 30--90 min |

## Artifacts and provenance

`artifacts/` contains reproducibility state, including prepared patient-level
data, representations, checkpoints, predictions and provenance bindings.
`results/` contains aggregate tables and publication figures. Large or
patient-level artifacts should remain local even when publishing the code.

Runners verify recorded input hashes, parameters and upstream bindings before
reusing cached products. A mismatched dependency is rebuilt from the nearest
valid upstream artifact; incompatible downstream products are moved to
`backup/stale_cache/`. Transfer an artifact together with its matching records
under `artifacts/provenance/`.

## References

- Pogorelyy MV, Minervina AA, Shugay M, et al. Detecting T cell receptors
  involved in immune responses from single repertoire snapshots. *PLOS
  Biology*. 2019;17:e3000314. https://doi.org/10.1371/journal.pbio.3000314
- Nagano Y, Pyo AGT, Milighetti M, et al. Contrastive learning of T cell
  receptor representations. *Cell Systems*. 2025;16:101165.
  https://doi.org/10.1016/j.cels.2024.12.006
- Sethna Z, Elhanati Y, Callan CG Jr, et al. OLGA: fast computation of
  generation probabilities of B- and T-cell receptor amino acid sequences and
  motifs. *Bioinformatics*. 2019;35:2974--2981.
  https://doi.org/10.1093/bioinformatics/btz035
- Shugay M, Bagaev DV, Zvyagin IV, et al. VDJdb: a curated database of T-cell
  receptor sequences with known antigen specificity. *Nucleic Acids
  Research*. 2018;46:D419--D427. https://doi.org/10.1093/nar/gkx760

## License

Project-owned code is released under the MIT License. Components in
`third_party/` retain their original licenses.
