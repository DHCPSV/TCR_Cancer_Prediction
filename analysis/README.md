# Analysis modules

Each experiment has one command-line entry point: `run.py`.

| Experiment | Model and computation | Reporting |
|---|---|---|
| 1 | `methods.py`, `prepare.py`, `study.py` | `report.py` |
| 2 | `methods.py`, `layer_seed_study.py`, `normalizer_study.py` | `report.py` |
| 3 | `study.py`, `pca.py` | `report.py` |
| 4 | `methods.py`, `prepare.py`, `study.py`, `families.py`, `vdjdb_lookup.py`, `gate_free_pca.py` | `report.py`, `learning_curves.py` |

Shared modules are deliberately small:

- `attention_mil.py`: the shared Attention MIL model;
- `protocol.py`: paths, schemas, fixed seeds, manifests and atomic writes;
- `training.py`: training, prediction and checkpoint/history binding;
- `reporting.py`: metrics and plot styling.

## Stages

| Experiment | Main stages | Optional stage |
|---|---|---|
| 1 | `internal`, `report`, `all` | `diagnostics` |
| 2 | `internal`, `report`, `all` | `diagnostics` |
| 3 | `prepare`, `report`, `pca`, `freeze`, `all` | — |
| 4 | `internal`, `validation`, `report`, `all` | `future-work` |

All runners also support `self-test` and `--input-mode {raw,cached}`. Cached
mode reads the extracted reproduction cache, accepts report stages only and
does not prepare data or train models.

Experiment 2 also reports a frozen aggregate of the 28-seed development
screen used to choose S01--S09. Those pre-V1 runs used the same fixed
alpha-chain SCEPTR--Sparsemax screening protocol; the default workflow trains
the selected formal grid and does not repeat the exploratory screen.
