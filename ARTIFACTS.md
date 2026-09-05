# Full reproduction cache

This separate download contains the original raw repertoires and the completed
preprocessing, embedding, ALICE and training outputs. No separate repertoire
download is needed. It supports two uses:

- `--input-mode cached --stage all` regenerates each experiment's main,
  Appendix and existing Future Work reports from stored inputs, without
  preparation or training;
- `--input-mode raw` runs the full computation, reusing prepared TCR tables,
  representations, ALICE files and checkpoints when their provenance matches.

## Release details

- Download: https://1drv.ms/u/c/c2c6d65a2073ac72/IQAEVFanW9TGQo2kYDN06_rTATWyG13lQwjCexb4kcsa0h8?e=ZKonUU
- File: `TCR_Cancer_Prediction_V1_reproduction_cache.zip`
- Compatible code: the raw-inclusive cache update; exact cache-code hashes are
  recorded in [the release manifest](artifacts/reproduction_cache_release.json).
- Compressed size: 8,379,845,059 bytes (7.80 GiB)
- Extracted size: 14,450,032,167 bytes (13.46 GiB), including the contents manifest.
- Contents: 12,569 input files plus `artifacts/reproduction_cache_contents.json`.
- Raw repertoires: 443 files (5,780,061,037 bytes), included in the sizes above.
- SHA-256: `49832c22759e04a192ade4f3108e21b3514a2c9ade2c8c3c9dd720dbd249c06f`

This replaces the smaller packages without raw repertoires. The filename is
unchanged, so use the size and SHA-256 to identify the correct download. The
matching code update must be published with this package: older cache
validators reject the added raw files. A compatible commit and download URL
are not yet recorded in the release manifest.

## Download and use

Clone the matching code version and install the Python environment described
in [README.md](README.md#environment). Keep at least 25 GiB free for the ZIP and
its extracted contents, in addition to space for the environment and reports.

From the repository root, verify the downloaded ZIP against the release
manifest before extracting it:

```powershell
$release = Get-Content -Raw .\artifacts\reproduction_cache_release.json | ConvertFrom-Json
$cacheZip = Join-Path (Get-Location).Path $release.zip_name
if ((Get-Item -LiteralPath $cacheZip).Length -ne $release.zip_bytes) { throw 'Cache ZIP size mismatch' }
if ((Get-FileHash -LiteralPath $cacheZip -Algorithm SHA256).Hash -ne $release.zip_sha256) { throw 'Cache ZIP checksum mismatch' }
```

Use a fresh checkout for extraction. `-Force` replaces existing files at
matching paths; do not run it over local data or model outputs you need to keep.

```powershell
Expand-Archive -LiteralPath .\TCR_Cancer_Prediction_V1_reproduction_cache.zip -DestinationPath . -Force
python -m pipeline.verify_reproduction_cache
```

The repository root should then contain `data/raw/` and `artifacts/`, not a
second nested project folder. The verifier checks the exact input list, file
sizes and SHA-256 hashes, raw-file hashes against the sample manifest, and
checkpoint loading. Run the four `--input-mode cached --stage all` commands in
[README.md](README.md#full-reproduction-cache-recommended) to generate the CSV
files and PNG figures under `results/`.

## Contents

The full ZIP contains:

- original internal and external raw repertoires named in the sample manifest;
- prepared patient TCR tables and their provenance sidecars;
- internal and external TCR representations, including SCEPTR embeddings;
- ALICE inputs, outputs, evidence, sequence maps and cache metadata;
- retained checkpoints and training histories;
- patient-level predictions, Experiment 3 PCA coordinates, and the
  Experiment 4 family, retention and gate-free PCA inputs used by the reports;
- the ALICE-derived tables needed to repeat the VDJdb lookup against the fixed
  snapshot already included in `third_party/vdjdb/`;
- manifests and provenance records needed to reuse compatible stages.

It does not contain repository code, `.git/`, a Python/R environment, generated
`results/`, manuscript files or local working material. The fixed third-party
code and VDJdb snapshot come from the Git repository. Results are regenerated
from the extracted cache.

Subject IDs and complete CDR3 sequences are retained because the source
datasets permit their public scientific use and they are not direct personal
identifiers. Dataset and third-party licence terms still apply; the project's
MIT licence does not replace them.

## Verification and raw-mode limitation

The 2026-09-05 package was extracted into a separate checkout and checked with
Python 3.12.10. All four cached workflows completed: the 28 PNG figures and
22 CSV files matched the repository results byte for byte, and all 38
unit/cached-result tests passed. No model retraining was performed.

Use `cached` mode for quick report reproduction. Some bundled TCR-table
provenance records refer to an earlier preprocessing-code hash. An additional
raw preparation check completed but rebuilt all 169 internal alpha tables,
so automatic reuse of the entire raw pipeline is not established. Raw mode
may rebuild affected tables and invalidate downstream caches. The package
retains the original cache bytes and provenance records unchanged.
