# Datasets

`raw/sui_move_dataset_v1.csv` is the untouched original (3,337 rows). `legacy/` holds the first, module-stripping\nattempt (`no_module`, `no_module_flat`), kept only for the TF-IDF results in `experiments/no_module*`.

## v2: shortcut-removal variants (built by `dataset_prep/build_neutral_datasets.py`)

All v2 variants and `control_unmodified.csv` hold the SAME 3,133 rows in the same order (see `v2/row_map.csv` for the original row of
each), so `src/prepare_data.py` gives the identical train/val/test split for each. Scores can be
compared pair-wise. Rows were dropped only if they were duplicates once all fixes are applied (204 rows,
mostly SemanticError templates that differed only by module name).

| File | Module names | `//` comments | Newlines / indentation |
|---|---|---|---|
| `v2/only_modules.csv` | random names, wrappers added where eligible | kept | kept |
| `v2/only_comments.csv` | kept | removed | kept |
| `v2/only_newlines.csv` | kept | kept | flattened |
| `v2/neutral_all.csv` | random names, wrappers added where eligible | removed | flattened |
| `v2/modules_newlines.csv` | exact names/wrappers from `only_modules` | kept | flattened |

Exemptions where the label itself is the thing being removed: StyleError/DeadCode and
StyleError/MissingDocumentation keep comments; StyleError/Spacing and StyleError/IncorrectIndentation
keep whitespace. `v2/BUILD_REPORT.txt` has the full report and leak audit.

The module rule anonymizes declarations and references; it does not delete module blocks.
Unwrapped SyntaxError samples remain unwrapped. The newline rule retains line breaks after
`//` comments so comments do not swallow subsequent code; comment text remains, although
its internal whitespace is normalized by the existing rule.

Build the combined module/newline variant with:

    python dataset_prep/build_modules_newlines.py

This starts from the existing `only_modules.csv`, reuses its randomized names, and applies
the same newline rule as `only_newlines.csv`. It never applies comment removal or chooses
a new row set. If regenerating all datasets, run it after `build_neutral_datasets.py`.
`v2/modules_newlines_build.json` records validation counts and input/output SHA-256 hashes.

## Running an experiment without touching the original results

    THESIS_CSV=$PWD/data/v2/<name>.csv THESIS_OUT=$PWD/experiments/<name> sh scripts/run_all.sh

Results go to `experiments/<name>/{data,results,logs,checkpoints}`. `data/` and `checkpoints/` are git-ignored.
