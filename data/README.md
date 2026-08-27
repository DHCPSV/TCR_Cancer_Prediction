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
    tx421_cancer/{alpha,beta}/
```

`internal/control` contains the age-matched non-cancer cohort and
`internal/cancer` contains the TRACERx100 PBMC cohort. `external/bcg_control`
contains the pre-treatment BCG/TCV controls. `external/tx421_cancer` contains
only the 36 additional TRACERx PBMC patients absent from the internal cohort;
`tx421_cancer` is a code alias rather than a claim about published cohort
provenance.

Do not rename subjects or chain files after placement. The pipeline records
their repository-relative paths and identities in `artifacts/manifests/` and
uses the checked-in manifest and checksum records to audit the frozen data
boundary. Raw archives, extracted repertoires and derived patient data must
remain outside version control.
