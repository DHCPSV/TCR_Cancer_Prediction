# Fixed VDJdb lookup

The formal family lookup uses the fixed VDJdb `2026-05-16` release. Git retains
only the files required to reproduce and license that lookup:

```text
2026-05-16/release/vdjdb-2026-05-16/
  LICENSE
  vdjdb.slim.meta.txt
  vdjdb.slim.txt
```

`source.csv` records the release URL and SHA-256 hashes. The lookup verifies
the slim table hash before reading it. The release archive, full tables, HTML,
motifs and clustering files are reproducible upstream assets and remain
ignored because they are not read by the formal analysis.

Changing the snapshot requires updating the release constant in
`analysis/experiment_04_alice_model/vdjdb_lookup.py` and the corresponding
provenance row in `source.csv` together.
