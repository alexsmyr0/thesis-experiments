"""Shared paths and evaluation helpers (same metrics as upstream src/evaluate.py)."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parent.parent
# Optional overrides so an experiment on a modified CSV never overwrites the original results:
#   THESIS_CSV=/abs/path/to.csv  THESIS_OUT=/abs/path/to/output_folder
# Defaults reproduce the original run (original CSV -> experiments/reproduction_v1/).
RAW_CSV = Path(os.environ.get("THESIS_CSV", REPO / "data" / "raw" / "sui_move_dataset_v1.csv"))
OUT_DIR = Path(os.environ.get("THESIS_OUT", REPO / "experiments" / "reproduction_v1"))
DATA_DIR = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"
CHECKPOINTS_DIR = OUT_DIR / "checkpoints"
LOGS_DIR = OUT_DIR / "logs"
SEED = 42


def compute_metrics(y_true, y_pred, labels):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def save_run(tag, metrics, y_true, y_pred, labels, extra=None):
    """Write metrics JSON and a confusion-matrix PNG for one run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"run": tag, **(extra or {}), **metrics}
    (RESULTS_DIR / f"{tag}_metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(tag)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"cm_{tag}.png", dpi=150)
    plt.close(fig)

    print(f"[{tag}] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    for c in labels:
        print(f"    {c:<14} F1={metrics['per_class'][c]['f1-score']:.3f}")
