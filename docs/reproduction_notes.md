# Reproduction notes: Bakalis (2026), *LLM-Based Classification of Move Smart-Contract Issues*

Re-runs the thesis experiments on `data/raw/sui_move_dataset_v1.csv`, using the settings in the thesis
PDF and in the upstream code (github.com/Vbakalis/move-issue-classifier, MIT licence, commit
`b3f63a8`). The scripts here are adapted from that code to read the CSV instead of the five
original Excel files.

## Run

See the repository README for the commands. Outputs of the original run are in
`experiments/reproduction_v1/` (`results/`, `logs/`; `logs/summary.txt` compares the thesis numbers with ours).

| Script | Thesis section |
|---|---|
| `src/prepare_data.py` | §3.4 cleaning, deduplication, stratified 70/15/15 split, seed 42 |
| `src/tfidf_baselines.py` | §4.2, Table 2 |
| `src/codebert_finetune.py --mode full/lora` | §4.4, Table 4 |
| `src/qwen_prompting.py --mode zeroshot/fewshot` | §4.3, Table 3 |
| `src/qwen_lora.py prepare/train/evaluate` | §4.5, Table 4 |
| `src/summarize.py` | compares against the thesis tables |

## Where the thesis text and the upstream code disagree

We followed the evidence of what was actually run (the code and the configs saved next to its
results), and kept the thesis text where the code does not settle it.

| Item | Thesis text | Upstream code / saved config | Used here |
|---|---|---|---|
| Split | 70/15/15 (Table 1, §5.2); "80/10/10" in §3.8 | 70/15/15 | 70/15/15 — matches Table 1 exactly |
| CodeBERT max length | 256 | script default 512 | 256 |
| CodeBERT LoRA lr / epochs | 5e-4 / 6 | default 2e-5 / 4; saved metrics record 5e-4 / 6 | 5e-4 / 6 |
| Few-shot example choice | shortest training snippet per class | first per class after seed-42 shuffle | upstream code |
| Qwen LoRA rank / alpha | r=16, α=32 | saved `adapter_config.json`: rank 8, scale 20, dropout 0 | saved config |
| Qwen LoRA optimiser | AdamW | Adam | Adam |
| Qwen best checkpoint | lowest validation loss (iter 400) | not automated | lowest reported val loss among saved checkpoints |

The thesis also states "Python 3.13"; this run used Python 3.12 with current library versions
(scikit-learn 1.9, torch 2.14, transformers 5.20, peft 0.21, mlx-lm). transformers 5 renamed
`warmup_ratio`; `warmup_steps=0.1` gives the same 10 % warmup.

## Data check

3,337 rows → 0 leaked headers → 16 whitespace-normalised duplicates removed → 3,321 samples.
"Syntax Error" is normalised to "SyntaxError" as upstream does. The resulting split
(2,323 / 499 / 499) is identical per class to thesis Table 1.

## Note: Optuna for hyperparameter tuning

[Optuna](https://optuna.org/) is a Python library that automatically searches for good
hyperparameters (learning rate, epochs, LoRA rank, `C`, ...) by running repeated trials of an
`objective(trial)` function. It integrates with scikit-learn and Hugging Face
`Trainer.hyperparameter_search(backend="optuna")`. If used here, optimise on the validation
split only, never test, and fix seeds. It is not used by these scripts, which keep the thesis
settings above; it could later measure how sensitive the CodeBERT LoRA scores are to the
5e-4 / 6 vs. 2e-5 / 4 discrepancy.

## Note: matplotlib for thesis figures

The thesis diagrams and graphs will be made with [matplotlib](https://matplotlib.org/). Generate
them from scripts that read the saved `results/*_metrics.json` files and save with
`fig.savefig(...)`, so figures can be regenerated without retraining.

## Not reproduced

Claude Sonnet 4.6 zero-shot (§5.7) needs an API key and paid calls and was not run.
