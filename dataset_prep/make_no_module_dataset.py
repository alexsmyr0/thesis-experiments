"""Make a copy of the dataset with the `module security::` / `module example::` shortcut removed.

Finding (see inspect_tfidf_weights.py): every SecurityError snippet starts with
`module security::<name> {` and every SemanticError / (almost every) SyntaxError snippet starts with
`module example::<name> {`. Perfect and StyleError snippets have no such wrapper, so a model can get
the label from the wrapper alone.

What this script does to each snippet (the original CSV is NOT modified):
  1. Removes each `module security|example::<name> {` header (also a dangling `module ` in front of it).
  2. Removes ONE closing `}` per removed header: the matching one if there is one, else the last `}`.
     This keeps the brace balance, so a SyntaxError "missing brace" sample stays broken.
  3. Removes any leftover `example::` / `security::` qualifier in the body (e.g. `example::test::secret()`).
  4. Strips the leftover common indentation and surrounding whitespace, and collapses blank-line runs.
  5. Drops any row whose code becomes empty (only the `module module example::test {}` "duplicate
     module keyword" sample, which has nothing left once the wrapper is gone).

Writes data/legacy/sui_move_dataset_v1_no_module.csv (same columns and encoding as the original) and prints a report.

Run from the Thesis folder:
    python dataset_prep/make_no_module_dataset.py
"""
import re
import textwrap
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "raw" / "sui_move_dataset_v1.csv"
DST = REPO / "data" / "legacy" / "sui_move_dataset_v1_no_module.csv"

HEADER = re.compile(r"(?:module\s+)*module\s+(?:security|example)\s*::\s*\w+\s*\{")
QUALIFIER = re.compile(r"\b(?:security|example)::")


def strip_wrapper(code):
    """Return (new_code, n_headers_removed, n_headers_without_matching_close)."""
    spans, taken, unmatched = [], set(), 0
    headers = list(HEADER.finditer(code))
    for m in headers:
        depth, close = 1, None
        for i in range(m.end(), len(code)):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close is None:  # module never closed (a "missing closing brace" syntax sample)
            unmatched += 1
            close = code.rfind("}")
        if close is not None and close >= 0 and close not in taken:
            taken.add(close)
            spans.append((close, close + 1))
        spans.append((m.start(), m.end()))
    for a, b in sorted(spans, reverse=True):
        code = code[:a] + code[b:]
    code = QUALIFIER.sub("", code)
    code = textwrap.dedent("\n".join(line.rstrip() for line in code.splitlines()))
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip(), len(headers), unmatched


def main():
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    out = df.copy()
    primary = df["Label"].str.split("/").str[0].str.strip().str.replace(" ", "")

    results = df["Code"].astype(str).map(strip_wrapper)
    out["Code"] = results.map(lambda r: r[0])
    n_hdr = results.map(lambda r: r[1])
    n_unm = results.map(lambda r: r[2])

    changed = out["Code"] != df["Code"].astype(str).str.strip()
    print(f"Rows: {len(df)}; rows changed: {changed.sum()}; rows with a header removed: {(n_hdr > 0).sum()}")
    print("Rows changed by class:")
    print(pd.crosstab(primary, changed).rename(columns={False: "unchanged", True: "changed"}).to_string())
    print(f"Rows with 2+ headers: {(n_hdr > 1).sum()}; headers with no matching close brace: {n_unm.sum()}")

    # Snippets that became empty or identical to another snippet
    empty = out["Code"].str.len() == 0
    print(f"Rows that became empty: {empty.sum()}")
    norm = out["Code"].map(lambda s: " ".join(s.split()))
    dup = norm.duplicated(keep=False)
    conflict = out.assign(norm=norm, p=primary)[dup].groupby("norm")["p"].nunique()
    print(f"Rows that now duplicate another row: {dup.sum()} "
          f"(extra copies prepare_data.py would drop: {norm.duplicated().sum()}); "
          f"identical code with DIFFERENT labels: {(conflict > 1).sum()} snippets")

    dropped = out[empty]
    for _, r in df[empty].iterrows():
        print(f"Dropping empty row: {r['Label']!r} / {r['Output/Fix'][:60]!r}")
    out = out[~empty]
    print(f"Rows written: {len(out)} (original {len(df)}, dropped {len(dropped)})")
    out.to_csv(DST, index=False, encoding="utf-8-sig")
    print(f"\nWrote {DST.name}")


if __name__ == "__main__":
    main()
