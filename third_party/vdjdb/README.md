# Fixed VDJdb lookup

The family lookup uses the fixed VDJdb `2026-05-16` release. Git retains a
derived table containing only the TRA/HomoSapiens rows and columns that can
match the patient-local families analysed in this project:

```text
2026-05-16/release/vdjdb-2026-05-16/
  LICENSE
  vdjdb.slim.meta.txt
  vdjdb.slim.txt
```

`source.csv` records the release URL, the source and derived SHA-256 hashes,
and the derived row count. The lookup verifies the derived table before
reading it. The release archive and full tables remain available upstream and
are not required to reproduce this lookup.

The [full reproduction cache](../../ARTIFACTS.md) supplies the patient-family
tables used as lookup inputs. Experiment 4's `--input-mode cached --stage all`
repeats the lookup and generates its reports using those tables and the fixed
snapshot here; no additional VDJdb download is needed.

Changing the snapshot requires updating the release constant in
`analysis/experiment_04_alice_model/vdjdb_lookup.py` and the corresponding
provenance row in `source.csv` together.
