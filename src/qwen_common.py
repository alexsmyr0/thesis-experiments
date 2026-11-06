"""Prompt, label parsing and test loop shared by the Qwen prompting and MLX-LoRA runs.

Copied from upstream src/prompting_baseline.py, src/mlx_prepare_data.py and src/mlx_evaluate.py.
"""

import json
import re
import time

import pandas as pd
from mlx_lm import generate

from common import DATA_DIR, RESULTS_DIR, compute_metrics, save_run

MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
LABELS = ["Perfect", "SecurityError", "SemanticError", "StyleError", "SyntaxError"]
SYSTEM = (
    "You are a classifier for Move smart-contract code. "
    f"Classify each code snippet into exactly one of: {', '.join(LABELS)}. "
    "Reply with only the class name, nothing else."
)
LABEL_PATTERN = re.compile(r"\b(" + "|".join(LABELS) + r")\b")


def snippet_turn(code):
    return f"Code:\n```move\n{code.strip()}\n```\n\nClass:"


def parse_label(raw):
    """First label name in the output; upstream falls back to Perfect when none appears."""
    m = LABEL_PATTERN.search(raw)
    return m.group(1) if m else "Perfect"


def evaluate(tag, model, tokenizer, build_messages, max_tokens=12, extra=None):
    """Greedy-generate a label for every test snippet and save metrics plus raw outputs."""
    test = pd.read_csv(DATA_DIR / "test.csv")
    y_true, y_pred, raws = [], [], []
    t0 = time.time()
    for i, row in test.iterrows():
        prompt = tokenizer.apply_chat_template(build_messages(str(row["code"])),
                                               add_generation_prompt=True, tokenize=False)
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        y_true.append(row["primary_label"])
        y_pred.append(parse_label(raw))
        raws.append(raw)
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(test)}  {rate:.2f}/s  ~{(len(test) - i - 1) / rate:.0f}s left", flush=True)
    seconds = time.time() - t0
    unparsed = sum(1 for r in raws if not LABEL_PATTERN.search(r))
    save_run(tag, compute_metrics(y_true, y_pred, LABELS), y_true, y_pred, LABELS,
             extra={"model": MODEL, "eval_seconds": seconds,
                    "unparsed_outputs_defaulted_to_Perfect": unparsed, **(extra or {})})
    with (RESULTS_DIR / f"{tag}_predictions.jsonl").open("w") as f:
        for t, p, r in zip(y_true, y_pred, raws):
            f.write(json.dumps({"true": t, "pred": p, "raw": r}) + "\n")
