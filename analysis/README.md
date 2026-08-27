# Analysis modules and stages

Each experiment has one command-line entry point, `run.py`.

| Experiment | Methods and computation | Reporting |
|---|---|---|
| `experiment_01_internal_baseline` | `methods.py`, `prepare.py`, `study.py` | `report.py` |
| `experiment_02_seed_and_attention_normalisation` | `methods.py`, `seed_mechanism.py`, `normalizer_study.py` | `report.py` |
| `experiment_03_external_generalisation` | `study.py`, `geometry.py` | `report.py` |
| `experiment_04_alice_model` | `methods.py`, `prepare.py`, `core.py`, `study.py`, `no_hard_gate.py`, `families.py`, `vdjdb_lookup.py`, `gate_free_pca.py` | `report.py`, `learning_curves.py` |

Shared modules:

- `protocol.py`: paths, schemas, fixed folds, manifests, hashes and atomic
  writes;
- `training.py`: shared training, prediction, checkpoint and history logic;
- `reporting.py`: metrics, bootstrap AUC and plot styling;
- `seed_display.py`: fixed seed-group display labels.

Run from the repository root:

```powershell
python -m analysis.experiment_01_internal_baseline.run
python -m analysis.experiment_02_seed_and_attention_normalisation.run
python -m analysis.experiment_03_external_generalisation.run
python -m analysis.experiment_04_alice_model.run
```

| Experiment | `all` | Additional stage |
|---|---|---|
| 01 | preparation, internal study and report | `diagnostics`: learning curves from existing histories |
| 02 | 50-epoch seed factorisation, normaliser study and report | `diagnostics`: optional 300-epoch selected-seed study |
| 03 | frozen transfer, report, seed-resolved geometry and freeze | `geometry` is also available separately |
| 04 | ALICE preparation, hard-gate study, locked validation, family/VDJdb analysis and no-gate comparison | `future-work`: representation dependence and gate-free PCA |

All four runners also support `--stage self-test`. Focused stages such as
`prepare`, `internal`, `report`, `assemble`, `geometry`, `freeze` and
`validation` are intended for recovery or report-only reruns.
