"""Figure: accuracy lost vs the unmodified control when each shortcut is removed.

Reads experiments/<run>/results/*_metrics.json (nothing is hard-coded) and writes the figure as a
600-dpi PNG plus vector PDF and SVG.

Usage: python analysis/plot_shortcut_cost.py [OUT_DIR]      (default OUT_DIR: ~/Downloads)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Downloads"

CONDITIONS = [  # (run folder, label)
    ("only_modules", "Module names"),
    ("only_comments", "Comments"),
    ("only_newlines", "Newlines & indentation"),
    ("neutral_all", "All three"),
]
MODELS = [  # (metrics file stem, legend label, colour)
    ("codebert_lora", "CodeBERT (LoRA)", "#2a78d6"),
    ("tfidf_svm_balanced", "TF-IDF + LinearSVC", "#eb6834"),
]
NOISE = 1.5  # approx. run-to-run noise in points on a 470-row test set


def errors(run, stem):
    m = json.load(open(REPO / "experiments" / run / "results" / f"{stem}_metrics.json"))
    cm = m["confusion_matrix"]
    return sum(map(sum, cm)) - sum(cm[i][i] for i in range(len(cm))), sum(map(sum, cm))


plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.2, 4.4))
fig.patch.set_facecolor("white")

bar_h, gap = 0.34, 0.03
for gi, (run, label) in enumerate(CONDITIONS):
    for mi, (stem, _, colour) in enumerate(MODELS):
        e, n = errors(run, stem)
        c, _ = errors("control_unmodified", stem)
        drop = 100 * (e - c) / n
        y = gi + (mi - 0.5) * (bar_h + gap)
        ax.barh(y, drop, height=bar_h, color=colour, zorder=3)
        ax.text(drop + 0.1, y, f"{drop:.1f} pts  ({c} → {e} errors)", va="center", ha="left",
                fontsize=8.5, color="#52514e", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

ax.axvline(NOISE, color="#8a8984", lw=0.9, ls=(0, (3, 3)), zorder=2)
ax.text(NOISE + 0.06, -0.62, "≈ seed noise", fontsize=8, color="#8a8984", va="bottom")
ax.set_yticks(range(len(CONDITIONS)))
ax.set_yticklabels([c[1] for c in CONDITIONS], fontweight="bold")
ax.set_ylim(len(CONDITIONS) - 0.4, -0.65)  # first condition on top
ax.set_xlim(0, 8.3)
ax.set_xticks(range(0, 7))
ax.set_xlabel("Accuracy lost vs unmodified control (percentage points)", color="#52514e")
ax.grid(axis="x", color="#e7e6e2", lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=0, colors="#52514e")
ax.tick_params(axis="y", colors="#0b0b0b", pad=8)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c2")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MODELS]
ax.legend(handles, [m[1] for m in MODELS], loc="lower left", bbox_to_anchor=(-0.02, 1.0),
          ncol=2, frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.5)
ax.set_title("What does each shortcut cost?", loc="left", fontsize=13, fontweight="bold", pad=30)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.text(0.01, 0.012, "Same 470 test rows; one run (seed 42) per bar. Dashed line: approximate run-to-run noise (\u00b11.5 points).",
         fontsize=8, color="#8a8984", va="bottom")

OUT.mkdir(parents=True, exist_ok=True)
stem = OUT / "shortcut_cost"
fig.savefig(f"{stem}.png", dpi=600, facecolor="white")
fig.savefig(f"{stem}.pdf", facecolor="white")
fig.savefig(f"{stem}.svg", facecolor="white")
print("wrote", *(f"{stem}.{ext}" for ext in ("png", "pdf", "svg")), sep="\n  ")
