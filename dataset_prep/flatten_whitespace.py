"""Make a whitespace-flattened copy of the no-module dataset.

Why: SecurityError / SemanticError samples are hand-made multi-line templates (100% contain newlines),
while Perfect samples are mostly single-line (92%). Move ignores whitespace, so newlines and
indentation carry no meaning but give CodeBERT a way to guess the class.

What it does to every row's Code: collapse every run of whitespace to one space, EXCEPT that a `//` line
comment keeps its line break (otherwise the comment would swallow the code after it).
Labels and Output/Fix are untouched. Comment text itself is NOT removed.

    python dataset_prep/flatten_whitespace.py
Reads data/legacy/sui_move_dataset_v1_no_module.csv, writes data/legacy/..._no_module_flat.csv.
"""
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "legacy" / "sui_move_dataset_v1_no_module.csv"
DST = REPO / "data" / "legacy" / "sui_move_dataset_v1_no_module_flat.csv"


def flatten(code):
    out = []
    for line in code.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        out.append(line + ("\n" if "//" in line else " "))
    return "".join(out).strip()


def main():
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df["Code"] = df["Code"].astype(str).map(flatten)
    primary = df["Label"].str.split("/").str[0].str.strip().str.replace(" ", "")
    print("Share of rows still containing a newline, by class (only rows with // comments should):")
    print(df["Code"].str.contains("\n").groupby(primary).mean().round(2).to_string())
    df.to_csv(DST, index=False, encoding="utf-8-sig")
    print(f"Wrote {DST.name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
