# Optional report cache

The report cache is a separate download for reproducing the published tables
and figures without repeating embedding or training.

Release details will be added here after the ZIP is uploaded:

- Download: pending
- File: `tcr-cancer-prediction-v1.0-report-cache.zip`
- Size: generated with the release package
- SHA-256: generated with the release package

After downloading, extract the ZIP into the repository root and run:

```powershell
python -m pipeline.verify_report_cache
```

The ZIP contains the retained checkpoints and training histories,
patient-level predictions, Experiment 3 PCA coordinates, and the small
Experiment 4 family, retention, VDJdb and gate-free PCA tables used by the
reports. It does not contain raw repertoires, prepared TCR tables, embeddings,
per-TCR ALICE evidence or `results/`.

Subject IDs and CDR3 sequences are retained because the source datasets permit
their public scientific use and they are not direct personal identifiers.
