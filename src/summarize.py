"""Print reproduced test scores next to the numbers reported in the thesis (Tables 2 and 4)."""

import json

from common import RESULTS_DIR

THESIS = {  # run tag: (accuracy, macro-F1) as printed in the thesis
    "tfidf_logreg": (0.950, 0.947),
    "tfidf_logreg_balanced": (0.940, 0.928),
    "tfidf_svm": (0.972, 0.965),
    "tfidf_svm_balanced": (0.972, 0.965),
    "tfidf_rf": (0.952, 0.947),
    "tfidf_rf_balanced": (0.948, 0.940),
    "codebert_full": (0.988, 0.992),
    "codebert_lora": (0.990, 0.993),
    "prompting_zeroshot": (0.499, 0.369),
    "prompting_fewshot_k1": (0.591, 0.512),
    "qwen_lora": (0.972, 0.978),
}


def main():
    head = f"{'Run':<24}{'Thesis acc':>11}{'Ours acc':>10}{'Δ':>8}{'Thesis F1':>11}{'Ours F1':>9}{'Δ':>8}"
    print(head + "\n" + "-" * len(head))
    for tag, (t_acc, t_f1) in THESIS.items():
        path = RESULTS_DIR / f"{tag}_metrics.json"
        if not path.exists():
            print(f"{tag:<24}{t_acc:>11.3f}{'not run':>10}")
            continue
        m = json.loads(path.read_text())
        acc, f1 = m["accuracy"], m["macro_f1"]
        print(f"{tag:<24}{t_acc:>11.3f}{acc:>10.3f}{acc - t_acc:>+8.3f}{t_f1:>11.3f}{f1:>9.3f}{f1 - t_f1:>+8.3f}")


if __name__ == "__main__":
    main()
