"""Qwen2.5-Coder-7B-Instruct 4-bit MLX LoRA: prepare data, train, evaluate (thesis §4.5, Table 4).

Training settings come from the adapter_config.json upstream saved with its results
(rank 8, scale 20, dropout 0, Adam, 16 layers, batch 2, lr 1e-4, 800 iters,
max length 1024, prompt masking, seed 42); the thesis text says r=16/alpha=32/AdamW.
Like the thesis, the checkpoint with the lowest validation loss is evaluated.

Usage: python qwen_lora.py prepare|train|evaluate
"""

import json
import re
import shutil
import subprocess
import sys

import pandas as pd
from mlx_lm import load

from common import CHECKPOINTS_DIR, DATA_DIR, LOGS_DIR, RESULTS_DIR, SEED
from qwen_common import MODEL, SYSTEM, evaluate, snippet_turn

MLX_DATA = DATA_DIR / "mlx"
ADAPTERS = CHECKPOINTS_DIR / "qwen_lora"
TRAIN_LOG = LOGS_DIR / "qwen_lora_train.log"
CONFIG = {
    "model": MODEL, "train": True, "data": str(MLX_DATA), "adapter_path": str(ADAPTERS),
    "fine_tune_type": "lora", "num_layers": 16, "batch_size": 2, "iters": 800,
    "learning_rate": 1e-4, "optimizer": "adam", "max_seq_length": 1024, "mask_prompt": True,
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
    "steps_per_report": 50, "steps_per_eval": 200, "val_batches": 20, "save_every": 200, "seed": SEED,
}


def user_turn(code):
    # Upstream puts the instructions in the user turn here (no system message), unlike prompting.
    return f"{SYSTEM}\n\n{snippet_turn(code)}"


def prepare():
    MLX_DATA.mkdir(parents=True, exist_ok=True)
    for src, dst in (("train", "train"), ("val", "valid"), ("test", "test")):
        df = pd.read_csv(DATA_DIR / f"{src}.csv")
        with (MLX_DATA / f"{dst}.jsonl").open("w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": user_turn(str(r["code"]))},
                    {"role": "assistant", "content": r["primary_label"]}]}, ensure_ascii=False) + "\n")
        print(f"{dst}.jsonl: {len(df)}")


def train():
    import yaml
    cfg_path = CHECKPOINTS_DIR / "qwen_lora_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    TRAIN_LOG.parent.mkdir(exist_ok=True)
    with TRAIN_LOG.open("w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "mlx_lm", "lora", "--config", str(cfg_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
    if proc.wait():
        sys.exit(f"mlx_lm lora failed with exit code {proc.returncode}")


def best_iteration():
    """Iteration with the lowest reported validation loss among saved checkpoints."""
    losses = {int(m[1]): float(m[2]) for m in
              re.finditer(r"Iter (\d+): Val loss ([\d.]+)", TRAIN_LOG.read_text())}
    saved = {it: loss for it, loss in losses.items()
             if (ADAPTERS / f"{it:07d}_adapters.safetensors").exists()}
    if not saved:
        sys.exit(f"No validation losses matched to saved checkpoints in {TRAIN_LOG}")
    it = min(saved, key=saved.get)
    return it, saved[it], losses


def evaluate_best():
    it, loss, losses = best_iteration()
    best_dir = CHECKPOINTS_DIR / "qwen_lora_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ADAPTERS / "adapter_config.json", best_dir)
    shutil.copy(ADAPTERS / f"{it:07d}_adapters.safetensors", best_dir / "adapters.safetensors")
    print(f"Evaluating iteration {it} (val loss {loss})")
    model, tokenizer = load(MODEL, adapter_path=str(best_dir))
    evaluate("qwen_lora", model, tokenizer, lambda code: [{"role": "user", "content": user_turn(code)}],
             extra={"best_iteration": it, "best_val_loss": loss, "val_loss_by_iter": losses,
                    "train_config": CONFIG})


if __name__ == "__main__":
    {"prepare": prepare, "train": train, "evaluate": evaluate_best}[sys.argv[1]]()
