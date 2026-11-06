# Datasets

`raw/sui_move_dataset_v1.csv` is the untouched original (3,337 rows). `legacy/` holds the first, module-stripping\nattempt (`no_module`, `no_module_flat`), kept only for the TF-IDF results in `experiments/no_module*`.

## v2: shortcut-removal variants (built by `dataset_prep/build_neutral_datasets.py`)

All four files hold the SAME 3,133 rows in the same order (see `v2/row_map.csv` for the original row of
each), so `src/prepare_data.py` gives the identical train/val/test split for each. Scores can be
compared pair-wise. Rows were dropped only if they were duplicates once all fixes are applied (204 rows,
mostly SemanticError templates that differed only by module name).

| File | Module names | `//` comments | Newlines / indentation |
|---|---|---|---|
| `v2/only_modules.csv` | random names, all rows wrapped | kept | kept |
| `v2/only_comments.csv` | kept | removed | kept |
| `v2/only_newlines.csv` | kept | kept | flattened |
| `v2/neutral_all.csv` | random names, all rows wrapped | removed | flattened |

Exemptions where the label itself is the thing being removed: StyleError/DeadCode and
StyleError/MissingDocumentation keep comments; StyleError/Spacing and StyleError/IncorrectIndentation
keep whitespace. `v2/BUILD_REPORT.txt` has the full report and leak audit.

## Running an experiment without touching the original results

    THESIS_CSV=$PWD/data/v2/<name>.csv THESIS_OUT=$PWD/experiments/<name> sh scripts/run_all.sh

Results go to `experiments/<name>/{data,results,logs,checkpoints}`. `data/` and `checkpoints/` are git-ignored.
