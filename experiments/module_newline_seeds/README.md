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
