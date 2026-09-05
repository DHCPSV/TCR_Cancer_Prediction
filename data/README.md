# Raw data boundary

Patient-level source repertoires are excluded from Git but included in the
[full reproduction cache](../ARTIFACTS.md). The current package contains all
443 raw files selected by `artifacts/manifests/samples.csv`; no separate raw
download is needed after extraction. It also supplies the prepared tables,
embeddings and other intermediate files under `artifacts/`.

Extract the ZIP into the repository root to populate the paths below. If you
do not use the package, place the approved source files at these same paths:

```text
data/raw/
  internal/
    control/{alpha,beta}/
    cancer/{alpha,beta}/
  external/
    bcg_control/{alpha,beta}/
    additional_tracerx_cancer/{alpha,beta}/
```

`internal/control` contains the age-matched non-cancer cohort and
`internal/cancer` contains the TRACERx100 PBMC cohort. `external/bcg_control`
contains the pre-treatment BCG/TCV controls.
`external/additional_tracerx_cancer` contains the 36 additional TRACERx PBMC
patients that are absent from the internal cohort.

Do not rename subjects or chain files after placement. The pipeline records
their repository-relative paths and identities in `artifacts/manifests/` and
uses manifest and checksum records to check the data boundary. Run
`python -m pipeline.verify_reproduction_cache` after extracting the full ZIP;
it checks raw-file hashes against the sample manifest as well as the remaining
cache contents. Raw and derived patient data remain outside Git.

The experiment runners use these raw inputs only in `--input-mode raw`.
`--input-mode cached --stage all` uses stored report inputs instead, without
regenerating tables or embeddings. Raw mode can rebuild caches whose provenance
does not match the current code; see [ARTIFACTS.md](../ARTIFACTS.md) for the
current release's reuse limitation.
