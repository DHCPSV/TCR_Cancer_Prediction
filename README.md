# Identifying a T cell receptor sequence signal in the blood of individuals with lung cancer

This repository contains four patient-level cancer-classification experiments
using T-cell receptor (TCR) repertoires.

| Experiment | Main question |
|---|---|
| 1. Internal baseline | Which TCR representation should be used? |
| 2. Seeds and attention | How do layer seeds and attention normalisers affect performance? |
| 3. External transfer | Do the frozen Experiment 2 models transfer to different cohorts? |
| 4. ALICE pooling | Does ALICE-informed pooling improve transfer? |

The main data path is:

```text
raw AIRR tables -> patient TCR tables -> TCR embeddings
-> patient vectors -> fold models -> patient predictions -> reports
```

## Two ways to run the project

Every experiment runner supports `--input-mode raw` and
`--input-mode cached`. The default is `raw`.

### Full run from raw data

This path performs preparation, embedding, training, validation and report
generation. Run the experiments in order:

```powershell
python -m analysis.experiment_01_internal_baseline.run --input-mode raw
python -m analysis.experiment_02_seed_and_attention_normalisation.run --input-mode raw
python -m analysis.experiment_03_external_generalisation.run --input-mode raw
python -m analysis.experiment_04_alice_model.run --input-mode raw
```

Experiments 1 and 2 use internal data only. Experiment 3 reuses the frozen
Experiment 2 checkpoints. Experiment 4 prepares ALICE evidence before fitting
its patient-level classifiers.

### Use the published reproduction cache

The optional reproduction cache contains the prepared patient TCR tables,
TCR representations, ALICE inputs and outputs, provenance records,
checkpoints, training histories, patient-level predictions and the
intermediate tables used by the reports. Raw repertoires and generated
`results/` are not included.

1. Download the ZIP listed in [ARTIFACTS.md](ARTIFACTS.md).
2. Extract it into the repository root without changing its directory layout.
   On Windows PowerShell, run this command from the repository root:

```powershell
Expand-Archive -LiteralPath .\TCR_Cancer_Prediction_V1_reproduction_cache.zip -DestinationPath . -Force
```

3. Verify it:

```powershell
python -m pipeline.verify_reproduction_cache
```

4. Rebuild the reports:

```powershell
python -m analysis.experiment_01_internal_baseline.run --input-mode cached
python -m analysis.experiment_02_seed_and_attention_normalisation.run --input-mode cached
python -m analysis.experiment_03_external_generalisation.run --input-mode cached
python -m analysis.experiment_04_alice_model.run --input-mode cached
```

Cached mode uses the stored predictions and other report inputs. It never
prepares raw data or trains models, and training stages such as `internal`,
`prepare` and `validation` are rejected.

The same extracted cache also supports the raw workflow. When the matching raw
repertoires are present under `data/raw/`, provenance checks reuse unchanged
patient TCR tables, representations and ALICE intermediate files instead of
recomputing them. Existing compatible checkpoints are reused as well.

## Environment

The tested setup is 64-bit Python 3.12 on Windows 11. The full raw workflow
uses an NVIDIA CUDA 12.8 build of PyTorch and R 4.3.3 for ALICE.

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
python -m pip install -e . --no-deps
```

Install the ALICE R dependencies and the included TCRgrapher source:

```powershell
$rscript = 'C:\Program Files\R\R-4.3.3\bin\Rscript.exe'
$r = 'C:\Program Files\R\R-4.3.3\bin\R.exe'
& $rscript -e "install.packages('remotes', repos='https://cloud.r-project.org')"
& $rscript -e "remotes::install_version('data.table', version='1.17.8', repos='https://cloud.r-project.org'); remotes::install_version('stringdist', version='0.9.15', repos='https://cloud.r-project.org'); remotes::install_version('iterators', version='1.0.14', repos='https://cloud.r-project.org'); remotes::install_version('foreach', version='1.5.2', repos='https://cloud.r-project.org'); remotes::install_version('doParallel', version='1.0.17', repos='https://cloud.r-project.org')"
& $r CMD INSTALL third_party\alice\tcrgrapher
```

Use `--rscript` if R is installed elsewhere.

## Raw data

Raw repertoires are not included in Git or in the reproduction cache. Place
approved inputs under:

```text
data/raw/
  internal/control/{alpha,beta}/
  internal/cancer/{alpha,beta}/
  external/bcg_control/{alpha,beta}/
  external/additional_tracerx_cancer/{alpha,beta}/
```

See [data/README.md](data/README.md) for the expected cohorts. Subject IDs and
CDR3 sequences in the published manifests and reproduction cache come from
approved public data sources; they are study identifiers rather than direct
personal identifiers.

## Repository layout

```text
analysis/       models, experiment computations and reports
pipeline/       manifest building, data preparation, embeddings and cache checks
tests/          portable tests and optional cached-result checks
results/        aggregate CSV files and publication figures
third_party/    pinned ALICE/OLGA code and the derived VDJdb snapshot
tools/          release utility for building the optional reproduction cache
```

Inside each experiment, `methods.py` contains the model definitions,
`study.py` or a clearly named study module contains the computation, and
`report.py` creates figures and aggregate tables. `run.py` is the only command
line entry point. See [analysis/README.md](analysis/README.md) for the stage
map.

Run portable tests with:

```powershell
python -m unittest discover -s tests -v
```

Set `TCR_REQUIRE_REPRODUCTION_CACHE=1` to make absence of the optional cache a
test failure instead of a skip.

## Code provenance and third-party material

Project-specific code is kept in `analysis/`, `pipeline/`, `tests/` and
`tools/`. The following external lineage and directory boundaries apply:

- The baseline developed for Experiment 1 and the non-ALICE representation
  and training path build on Jan Pytel's
  [TCR-Cancer-Prediction](https://github.com/pytelj/TCR-Cancer-Prediction)
  project. The workflow has since been reorganised for fixed folds, explicit
  seed studies, external transfer and the reproduction-cache interface used
  here.
- `third_party/alice/tcrgrapher/` is a pinned snapshot of the external
  [TCRgrapher](https://github.com/KseniaMIPT/tcrgrapher) source at commit
  `e2e4347f7689dd21304f2e032c1fa74833367af2`.
- `third_party/alice/olga/` contains the external OLGA 1.2.4 source and the
  human TRA/TRB generation models used by Experiment 4. Its upstream source is
  [OLGA](https://github.com/statbiophys/OLGA).
- `third_party/alice/alice_worker.R` is project-specific integration code. It
  connects the pinned TCRgrapher implementation to OLGA and the supported
  Windows workflow; it is not an upstream ALICE file.
- `third_party/vdjdb/` contains a fixed, derived VDJdb data snapshot used by
  the family lookup. It is external data rather than project-owned code.
- SCEPTR is installed as a Python dependency. Its source is not copied into
  this repository; `pipeline/sceptr_adapter.py` is project-specific adapter
  code.

## Evaluation boundary

- Internal evaluation uses one held-out prediction per patient from the fixed
  five-fold split `fivefold_seed913271`.
- Each external score is the mean of five frozen fold predictions.
- External data are not used to select methods, seeds, epochs or thresholds.
- Source and label are confounded in the external cohorts. External results
  therefore measure cohort transfer, not a cancer-causal biomarker.
- ALICE L1 and L2 refer to pooling normalisation, not weight regularisation.
- Training sums four class-weighted patient losses before each optimiser
  update; it does not average those four losses.

## References

- Pytel J. [TCR-Cancer-Prediction](https://github.com/pytelj/TCR-Cancer-Prediction).
- Pogorelyy MV, Minervina AA, Shugay M, et al. ALICE. *PLOS Biology* 2019.
  [doi:10.1371/journal.pbio.3000314](https://doi.org/10.1371/journal.pbio.3000314).
- Lupyr K. [TCRgrapher](https://github.com/KseniaMIPT/tcrgrapher).
- Nagano Y, Pyo AGT, Milighetti M, et al. SCEPTR. *Cell Systems* 2025.
  [doi:10.1016/j.cels.2024.12.006](https://doi.org/10.1016/j.cels.2024.12.006).
- Sethna Z, Elhanati Y, Callan CG Jr, et al. OLGA. *Bioinformatics* 2019.
  [doi:10.1093/bioinformatics/btz035](https://doi.org/10.1093/bioinformatics/btz035).
- Shugay M, Bagaev DV, Zvyagin IV, et al. VDJdb. *Nucleic Acids Research*
  2018. [doi:10.1093/nar/gkx760](https://doi.org/10.1093/nar/gkx760).

## License

Project-owned code is released under the MIT License. The bundled TCRgrapher
and OLGA sources retain their GPL licences, and the VDJdb snapshot retains its
included upstream licence. The corresponding source and licence files are
kept inside each `third_party/` directory.
