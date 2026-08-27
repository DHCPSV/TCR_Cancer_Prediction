# Raw data boundary

Patient-level source repertoires are intentionally excluded from Git. Place
the approved, extracted inputs under the following repository-relative paths:

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
uses manifest and checksum records to check the data boundary. Raw archives,
extracted repertoires and derived patient data remain outside version control.
