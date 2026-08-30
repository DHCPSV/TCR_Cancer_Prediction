# Optional reproduction cache

The reproduction cache is a separate download for reusing the completed
preprocessing, embedding, ALICE and training outputs. It supports two uses:

- `--input-mode cached` verifies the stored report inputs and regenerates all
  published tables and figures without training;
- `--input-mode raw` can reuse unchanged prepared TCR tables,
  representations, ALICE files and checkpoints after the matching raw data
  have been placed under `data/raw/`.

Release details:

- Download: https://1drv.ms/u/c/c2c6d65a2073ac72/IQADw2U6f7EdQ5lh1PHqXku7AV1APGtWD79rmTAvTJe8MtQ?e=bbTfJ5
- File: `TCR_Cancer_Prediction_V1_reproduction_cache.zip`
- Compressed size: 6,953,036,402 bytes (6.48 GiB)
- Extracted size: 8,669,878,235 bytes (8.07 GiB)
- SHA-256: `57aa08d04d856f94055a18c82f36ebd7fb5b13d98fcf4f273a19e5877bbf784d`

After downloading, extract the ZIP into the repository root and run:

```powershell
Expand-Archive -LiteralPath .\TCR_Cancer_Prediction_V1_reproduction_cache.zip -DestinationPath . -Force
python -m pipeline.verify_reproduction_cache
```

The ZIP contains:

- prepared patient TCR tables and their provenance sidecars;
- internal and external TCR representations, including SCEPTR embeddings;
- ALICE inputs, outputs, evidence, sequence maps and cache metadata;
- retained checkpoints and training histories;
- patient-level predictions, Experiment 3 PCA coordinates, and the
  Experiment 4 family, retention and gate-free PCA inputs used by the reports;
- the ALICE-derived tables needed to repeat the VDJdb lookup against the fixed
  snapshot already included in `third_party/vdjdb/`;
- manifests and provenance records needed to reuse compatible stages.

It does not contain raw repertoires or `results/`. Results are regenerated from
the extracted cache. Subject IDs and complete CDR3 sequences are retained
because the source datasets permit their public scientific use and they are
not direct personal identifiers.
