# Paired module/newline seed replication

This study retains the original seed-42 Kaggle results and adds seeds **7 and 13**.
Each seed controls both the 70/15/15 split and model training. Within a seed, all
four variants use the same source-row assignments. Source CSVs and module names
stay fixed; no comment-removal treatment is included.

Models: CodeBERT LoRA (6 epochs, FP32), TF-IDF + LinearSVC, and balanced LinearSVC.
Datasets: `control_unmodified`, `only_modules`, `only_newlines`, `modules_newlines`.
All use 3,133 rows: 2,193 train, 470 validation, 470 test.

## Files

- `seed_<seed>/<dataset>/results/`: original metrics, confusion matrices, and
  row-level predictions for the new runs. Seed 42 did not save predictions.
- `seed_<seed>/<dataset>/logs/`: preparation, training and evaluation logs.
- `seed_<seed>/<dataset>/split_manifest.json`: exact zero-based CSV row IDs
  in each split, plus dataset SHA-256. Seed-42 IDs were recovered from its archived splits.
- `seed_<seed>/<dataset>/provenance.json`: source commit, runtime, seed and archive hash.
- `seed_<seed>/<dataset>/pip_freeze.txt`: exact package versions.
- `metrics_by_seed.csv`: accuracy and macro-F1 on the 0–1 scale, one row per model/dataset/seed.
- `paired_deltas.csv`: treatment minus control for the same seed/model, in percentage points.
  Negative means treatment performed worse.
- `aggregate.csv`: mean and sample standard deviation across available seeds (0–1 scale).
  `n_seeds` gives the sample count; SD is blank when only one seed is available.
- `status.json`: available seeds, completeness and paired-split validation status.

Prepared `data/` splits are retained locally but git-ignored; checkpoints are not stored.
The checked-in manifests and unchanged source CSVs reproduce the splits.

## Provenance and import

Seed 42 comes from Kaggle Versions [2](https://www.kaggle.com/code/alexsmyroglou/thesis-experiments-codebert?scriptVersionId=366298337)
and [3](https://www.kaggle.com/code/alexsmyroglou/thesis-experiments-codebert?scriptVersionId=366307931).
Its original metric files remain unmodified; seed inference is explicitly recorded in provenance.

Import a new sweep archive from the repository root:

```sh
python scripts/collect_seed_results.py /path/to/results.zip
```

This refuses conflicting existing results, checks paired row assignments, and
regenerates the three summary CSVs and `status.json`. For the old seed-42 archives,
pass `--legacy-seed 42` explicitly.

Run the new sweep on Kaggle using `notebooks/kaggle_codebert.ipynb`, or directly:

```sh
python scripts/run_seed_sweep.py --seeds 7 13 --out /path/to/fresh-output
```

Use `--quick` for a local SVC/split validation only. Never mix those local results
with the Kaggle study: dependency and hardware differences can affect scores.

Three seeds are an exploratory replication, not strong statistical evidence.
Report all seeds and paired effects; the sample SD is not a confidence interval.

## Completed results: seeds 7, 13 and 42

Seeds 7 and 13 completed in [Kaggle Version 4](https://www.kaggle.com/code/alexsmyroglou/thesis-experiments-codebert?scriptVersionId=366315179), runtime 40m 11s. Source commit: `79ff9db2c42e88bda7ed3537eb84db9e03a4a35a`. Hardware and recorded package versions match seed 42. All 36 model/dataset/seed metrics are present. The 24 new prediction files reproduce their confusion matrices, accuracy and macro-F1; all split manifests passed pairing checks.

Mean accuracy / mean macro-F1 (%), with each seed weighted equally:

| Dataset | CodeBERT LoRA | TF-IDF + LinearSVC | LinearSVC balanced |
|---|---:|---:|---:|
| Control | 99.15 / 99.31 | 95.60 / 94.62 | 95.60 / 94.69 |
| Modules altered | 97.80 / 96.67 | 89.72 / 83.95 | 90.21 / 84.75 |
| Newlines reduced | 96.88 / 95.26 | 95.60 / 94.62 | 95.60 / 94.69 |
| Modules + newlines | 94.47 / 91.57 | 89.72 / 83.95 | 90.21 / 84.75 |

Macro-F1 mean ± sample SD (%; SD expressed in percentage points):

| Dataset | CodeBERT LoRA | TF-IDF + LinearSVC | LinearSVC balanced |
|---|---:|---:|---:|
| Control | 99.31 ± 0.37 | 94.62 ± 1.06 | 94.69 ± 1.63 |
| Modules altered | 96.67 ± 0.72 | 83.95 ± 2.79 | 84.75 ± 3.39 |
| Newlines reduced | 95.26 ± 1.31 | 94.62 ± 1.06 | 94.69 ± 1.63 |
| Modules + newlines | 91.57 ± 0.68 | 83.95 ± 2.79 | 84.75 ± 3.39 |

For CodeBERT, every seed shows lower macro-F1 for each treatment than its paired control. The combined-treatment drop is 7.02–8.21 percentage points (mean 7.74). Module-only and combined-treatment LinearSVC metrics match exactly within every seed; newline-only matches control. These results support repeatability on these three seeds, not statistical certainty or generalization to unseen projects.
