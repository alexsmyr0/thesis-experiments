"""CodeBERT-base full fine-tuning or LoRA (thesis §4.4, Table 4).

Thesis settings (which differ from the upstream script defaults, see README):
  full: lr 2e-5, 4 epochs     lora: r=16, alpha=32, dropout 0.05 on query/value, lr 5e-4, 6 epochs
  both: batch 16, weight decay 0.01, 10% warmup, max length 256, seed 42,
        keep the epoch with the best validation macro-F1.

Runs on Apple MPS, NVIDIA CUDA (e.g. Kaggle) or CPU, picked automatically. The effective batch size stays 16
on multi-GPU machines (the per-GPU batch is divided by the GPU count), so results stay comparable.
--fp16 is off by default because mixed precision can shift the scores slightly; use it for speed on CUDA.

Usage: python codebert_finetune.py --mode full|lora [--fp16]
"""

import argparse

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from common import CHECKPOINTS_DIR, DATA_DIR, SEED, compute_metrics, save_run

MODEL = "microsoft/codebert-base"
BATCH_SIZE = 16  # effective batch size (thesis setting)
SETTINGS = {"full": {"lr": 2e-5, "epochs": 4}, "lora": {"lr": 5e-4, "epochs": 6}}


def hf_metrics(eval_pred):
    logits, y = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(SETTINGS), required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--fp16", action="store_true", help="mixed precision (CUDA only; may shift scores slightly)")
    args = ap.parse_args()
    cfg = SETTINGS[args.mode]
    set_seed(SEED)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    n_gpu = torch.cuda.device_count() if device == "cuda" else 1
    batch_per_device = max(1, BATCH_SIZE // n_gpu)  # Trainer multiplies by the GPU count
    fp16 = args.fp16 and device == "cuda"
    if args.fp16 and not fp16:
        print("[warn] --fp16 ignored: only supported on CUDA")
    print(f"[device] {device} x{n_gpu}  [mode] {args.mode}  [fp16] {fp16}  {cfg}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    frames = {s: pd.read_csv(DATA_DIR / f"{s}.csv") for s in ("train", "val", "test")}
    labels = sorted(frames["train"]["primary_label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}

    def to_ds(df):
        ds = Dataset.from_dict({"text": df["code"].tolist(),
                                "label": [label2id[l] for l in df["primary_label"]]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_length),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = (to_ds(frames[s]) for s in ("train", "val", "test"))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels), label2id=label2id, id2label=dict(enumerate(labels)))
    if args.mode == "lora":
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["query", "value"], bias="none"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable={trainable:,} total={total:,} ({100 * trainable / total:.2f}%)")

    targs = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR / f"codebert_{args.mode}"),
        num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"],
        per_device_train_batch_size=batch_per_device, per_device_eval_batch_size=32,
        weight_decay=0.01, warmup_steps=0.1,  # transformers 5: a float < 1 is a ratio of total steps
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        logging_steps=25, report_to="none", seed=SEED, use_cpu=(device == "cpu"), fp16=fp16,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=hf_metrics)
    trainer.train()

    out = trainer.predict(test_ds)
    y_true = [labels[i] for i in out.label_ids]
    y_pred = [labels[i] for i in np.argmax(out.predictions, axis=-1)]
    epoch_log = [{k: e[k] for k in ("epoch", "eval_accuracy", "eval_macro_f1") if k in e}
                 for e in trainer.state.log_history if "eval_macro_f1" in e]
    save_run(f"codebert_{args.mode}", compute_metrics(y_true, y_pred, labels), y_true, y_pred, labels,
             extra={"trainable_params": trainable, "total_params": total, **cfg,
                    "max_length": args.max_length, "device": device, "n_gpu": n_gpu, "fp16": fp16,
                    "torch": torch.__version__, "transformers": transformers.__version__, "val_by_epoch": epoch_log,
                    "best_checkpoint": trainer.state.best_model_checkpoint})


if __name__ == "__main__":
    main()
