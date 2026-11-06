"""Qwen2.5-Coder-7B-Instruct (4-bit MLX) zero-shot and few-shot prompting, no training (thesis §4.3, Table 3).

Few-shot picks the first example of each class after a seed-42 shuffle of the
training set, as upstream code does (the thesis text says "shortest"; see README).

Usage: python qwen_prompting.py --mode zeroshot|fewshot
"""

import argparse

import pandas as pd
from mlx_lm import load

from common import DATA_DIR, SEED
from qwen_common import LABELS, MODEL, SYSTEM, evaluate, snippet_turn


def few_shot_examples(k_per_class=1):
    train = pd.read_csv(DATA_DIR / "train.csv")
    order = pd.Series(range(len(train))).sample(frac=1, random_state=SEED).tolist()
    train = train.iloc[order].reset_index(drop=True)
    return [(str(r["code"]), label) for label in LABELS
            for _, r in train[train["primary_label"] == label].head(k_per_class).iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zeroshot", "fewshot"], required=True)
    args = ap.parse_args()

    examples = few_shot_examples() if args.mode == "fewshot" else []
    demo = []
    for code, label in examples:
        demo += [{"role": "user", "content": snippet_turn(code)}, {"role": "assistant", "content": label}]

    def build_messages(code):
        return [{"role": "system", "content": SYSTEM}, *demo, {"role": "user", "content": snippet_turn(code)}]

    model, tokenizer = load(MODEL)
    tag = "prompting_zeroshot" if args.mode == "zeroshot" else "prompting_fewshot_k1"
    evaluate(tag, model, tokenizer, build_messages, extra={"few_shot_examples": len(examples)})


if __name__ == "__main__":
    main()
