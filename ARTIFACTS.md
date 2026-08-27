# Optional report cache

The report cache is a separate download for reproducing the published tables
and figures without repeating embedding or training.

Release details:

- Download: pending
- File: `TCR_Cancer_Prediction_V1_report_cache.zip`
- Size: 12,715,770 bytes
- SHA-256: `6ae0a0b74b586fea11449bfce4c7198fc1921266842e2e3b656687e0a5ad7b20`

After downloading, extract the ZIP into the repository root and run:

```powershell
Expand-Archive -LiteralPath .\TCR_Cancer_Prediction_V1_report_cache.zip -DestinationPath . -Force
python -m pipeline.verify_report_cache
```

The ZIP contains the retained checkpoints and training histories,
patient-level predictions, Experiment 3 PCA coordinates, and the small
Experiment 4 family, retention, VDJdb and gate-free PCA tables used by the
reports. It does not contain raw repertoires, prepared TCR tables, embeddings,
per-TCR ALICE evidence or `results/`.

Subject IDs and CDR3 sequences are retained because the source datasets permit
their public scientific use and they are not direct personal identifiers.
