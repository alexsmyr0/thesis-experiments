# thesis-experiments

Code, datasets and results for my thesis on evaluating the
[Move Issue Classifier](https://github.com/Vbakalis/move-issue-classifier) (TF-IDF baselines, CodeBERT full
fine-tuning and LoRA, Qwen-7B). The question: do the reported ~99% scores reflect real issue recognition, or
non-semantic shortcuts in the dataset (module names, `//` comments, newlines)?

## Layout

```
data/
  raw/          sui_move_dataset_v1.csv    the untouched original (3,337 rows)
  v2/           shortcut-removal variants  (see data/README.md)
  legacy/       first module-stripping attempt, kept for old results
dataset_prep/   scripts that build data/v2 and data/legacy from data/raw
src/            the experiments: prepare_data, tfidf_baselines, codebert_finetune, qwen_*, summarize
scripts/        run_all.sh  (runs the pipeline for one dataset)
notebooks/      kaggle_codebert.ipynb  (runs CodeBERT on a Kaggle GPU)
analysis/       exploration and inspection scripts (read-only)
experiments/    one folder per run: results/ and logs/ are committed; data/ and checkpoints/ are git-ignored
docs/           notes on how the reproduction differs from the thesis text
```

## Run locally

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt

sh scripts/run_all.sh --quick        # split + six TF-IDF baselines (seconds)
sh scripts/run_all.sh                # + CodeBERT full and LoRA
sh scripts/run_all.sh --qwen         # + Qwen-7B (Apple Silicon / MLX only, hours)

# any other dataset, without touching existing results:
THESIS_CSV=$PWD/data/v2/neutral_all.csv THESIS_OUT=$PWD/experiments/neutral_all sh scripts/run_all.sh
```

`codebert_finetune.py` picks CUDA, then Apple MPS, then CPU. Set `CODEBERT_ARGS=--fp16` to use mixed precision on CUDA.

## Run on Kaggle

1. Push this repo to GitHub (public, or private plus a `GITHUB_TOKEN` Kaggle secret, see the notebook).
2. On kaggle.com: **Create -> New Notebook -> File -> Import Notebook** and upload `notebooks/kaggle_codebert.ipynb`.
3. Session options: **Accelerator = GPU T4 x2**, **Internet = On** (needs a phone-verified account).
4. **Save Version -> Save & Run All (Commit)**, then close the tab.
5. When it finishes, download `results.zip` from the notebook's **Output** tab and unzip it into `experiments/`.
   Commit the new `results/` and `logs/` folders.

The notebook clones this repo, installs `transformers>=5` and `peft`, runs `scripts/run_all.sh --codebert=both` for
each dataset in `DATASETS`, deletes the checkpoints, and zips the results. Qwen/MLX steps do not run on Kaggle.

Notes on comparing runs:
- On multi-GPU machines the script divides the per-GPU batch by the GPU count, so the effective batch stays 16.
- fp16 is off by default. Different hardware and library versions still give slightly different numbers even with
  seed 42. Each `*_metrics.json` records the device, `n_gpu`, fp16, and torch/transformers versions. State in the
  thesis which hardware produced each table, and compare variants only against a control run on the same hardware.

## Versions used for the local (M2 Air, Python 3.12) results

torch 2.14.1, transformers 5.20.2, peft 0.21.2, datasets 5.1.0, accelerate 1.15.0, scikit-learn 1.9.1,
pandas 3.0.6, numpy 2.5.4, mlx 0.32.4, mlx-lm 0.32.2.
