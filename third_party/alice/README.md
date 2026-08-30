# ALICE paper implementation

This directory is the complete, isolated third-party boundary for ALICE.
Project training code reads only derived files under `artifacts/alice/`.

Pinned sources:

- TCRgrapher `First_release`, commit
  `e2e4347f7689dd21304f2e032c1fa74833367af2`
  ([source](https://github.com/KseniaMIPT/tcrgrapher)).
- OLGA `1.2.4`
  ([source](https://github.com/statbiophys/OLGA)).

`tcrgrapher/` is the pinned paper-code snapshot. `olga/` retains the frozen
upstream source and only the human TRA/TRB generation models used by this
project; the worker calls `olga-compute_pgen` from `olga==1.2.4` in the root
`.venv`. Both upstream licences are preserved.
`alice_worker.R` is the only project-owned adapter. It preserves the paper
algorithm and fixed parameters while making the OLGA subprocess call work on
Windows:

```text
Q_val                 27
thres_counts          0
N_neighbors_thres     2
p_adjust_method       BH
significant hit       q < 0.001
models                humanTRA / humanTRB
```

ALICE uses the repository's supported Windows environment: Python 3.12 in
the root `.venv`, `olga==1.2.4` from `requirements.txt`, and system R 4.3.3
installed at `C:\Program Files\R\R-4.3.3`. Activate the local Python
environment before installing or running the worker. The Experiment 4 runner
uses `Rscript` from `PATH`, then checks the standard Windows installation; use
`--rscript` for another location.

The R runtime requires only `data.table`, `stringdist`, `foreach`,
`doParallel` and `iterators`; the remaining packages listed under `Suggests`
in TCRgrapher are not used by this project's ALICE path. Install the pinned
runtime versions and then the vendored package from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
$rscript = 'C:\Program Files\R\R-4.3.3\bin\Rscript.exe'
$r = 'C:\Program Files\R\R-4.3.3\bin\R.exe'
& $rscript -e "install.packages('remotes', repos='https://cloud.r-project.org')"
& $rscript -e "remotes::install_version('data.table', version='1.17.8', repos='https://cloud.r-project.org'); remotes::install_version('stringdist', version='0.9.15', repos='https://cloud.r-project.org'); remotes::install_version('iterators', version='1.0.14', repos='https://cloud.r-project.org'); remotes::install_version('foreach', version='1.5.2', repos='https://cloud.r-project.org'); remotes::install_version('doParallel', version='1.0.17', repos='https://cloud.r-project.org')"
& $r CMD INSTALL third_party\alice\tcrgrapher
```

Conda metadata is intentionally not part of the supported project interface.
If archived R packages must be compiled, use Rtools43 at its default
`C:\rtools43` location.

The worker interface is:

```text
Rscript alice_worker.R INPUT.tsv OUTPUT.tsv alpha|beta olga-compute_pgen
```
