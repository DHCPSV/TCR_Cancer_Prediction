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

| Experiment | Raw-mode stages | Separate raw-mode stage |
|---|---|---|
| 1 | `internal`, `report`, `all` | `diagnostics` |
| 2 | `internal`, `report`, `all` | `diagnostics` |
| 3 | `prepare`, `report`, `pca`, `freeze`, `all` | — |
| 4 | `internal`, `validation`, `report`, `all` | `future-work` |

All runners also support `self-test` and `--input-mode {raw,cached}`. Both
`--input-mode raw` and `--stage all` are defaults. In raw mode, `all` runs the
main stages; the separate stages above must be requested explicitly.

For quick reproduction, extract and verify the
[full cache](../ARTIFACTS.md), then use `--input-mode cached --stage all`.
The package includes raw repertoires, embeddings and ALICE intermediate files,
but cached mode reads only the stored report inputs and checkpoints. It does
not prepare data, compute embeddings, run ALICE or train models.

| Experiment | Stages included in cached `all` |
|---|---|
| 1 | `self-test`, `report`, `diagnostics` |
| 2 | `self-test`, `report`, `diagnostics` |
| 3 | `self-test`, `report`, `pca` |
| 4 | `self-test`, `report`, `future-work` |

Each listed cached stage can also run on its own. Other stages are rejected
in cached mode; use raw mode for preparation, training or freezing. Reports
write to `results/experiment_*/`; cached `all` includes the Appendix figures
and the existing Future Work results, without extending the experiments.

Experiment 2 `internal` trains the 50-epoch layer-seed and normaliser
comparisons, plus trajectories for three selected alpha-chain SCEPTR--Sparsemax
seeds (S01--S03). The trajectories save epochs 0, 5, 10, 25, 50, 100, 200 and
300. `layer_seed_study.py` reads these checkpoints; `report.py` produces
`attention_weight_trajectory.png` and `classifier_weight_trajectory.png` in
the main `figures/` directory. Both figures are included in `report` and `all`.
The full cache contains all 120 trajectory checkpoints.

Each heatmap shows raw weights averaged across five folds, not biological
feature importance. Each layer uses a separate symmetric colour scale at
the 98th percentile of absolute mean weights. Epoch rows are discrete saved
checkpoints, not equally spaced training intervals. These selected-seed
trajectories do not change the main 50-epoch evaluation endpoint. The optional
`diagnostics` stage plots their AUC/BCE learning curves using the same files.

Experiment 2 also reports a frozen aggregate of the 28-seed development
screen used to choose S01--S09. Those pre-V1 runs used the same fixed
alpha-chain SCEPTR--Sparsemax screening protocol; the default workflow trains
the selected formal grid and does not repeat the exploratory screen.
