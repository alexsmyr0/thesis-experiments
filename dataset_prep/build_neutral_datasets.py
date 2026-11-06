"""Build the "v2" datasets that remove three non-semantic shortcuts found in sui_move_dataset_v1.csv.

Shortcuts (see inspect_tfidf_weights.py and the shortcut-baseline analysis):
  MODULES  Every Security/Semantic (and almost every Syntax) sample starts with `module security::x {` or
           `module example::x {`; Perfect/Style samples have no wrapper. The name is the label itself.
  COMMENTS ~32% of SecurityError samples contain `//` comments that describe the bug
           ("// No access control"); Perfect has them in ~1%.
  NEWLINES Security/Semantic samples are hand-made multi-line templates (100% have newlines);
           Perfect samples are mostly one line (92%).

Fixes (each applied from the ORIGINAL csv, which is never modified):
  modules   Rename every `module A::B` declaration to a random nonsense name from one shared pool, consistently
            inside the row (header and later references such as `example::test::secret()`). Samples that have
            no module wrapper (Perfect, Style, ...) are wrapped in a random module as well, so "has a module"
            no longer says anything about the class. SyntaxError samples without a wrapper stay as they are
            (their label can be "declaration outside module"). Newline layout of the row is not changed.
  comments  Remove `//` line comments (not `///` doc comments). Exempt: StyleError/DeadCode (the label IS
            commented-out code) and StyleError/MissingDocumentation. Single-line rows can't delimit a
            comment, so their comments are kept (counted in the report).
  newlines  Collapse all whitespace to single spaces, keeping a line break after a `//` line so the comment
            can't swallow code. Exempt: StyleError/Spacing and StyleError/IncorrectIndentation (the label IS
            whitespace).

Outputs in datasets/v2/:
  neutral_all.csv      all three fixes
  only_modules.csv     modules fix only
  only_comments.csv    comments fix only
  only_newlines.csv    newlines fix only
  control_unmodified.csv  the same rows with the ORIGINAL text, no fixes (paired control)
  row_map.csv          new row number -> original row number
  BUILD_REPORT.txt     this script's printed report

All four files contain exactly the SAME rows in the same order. Rows are dropped if (a) they are a duplicate
once ALL fixes are applied (module names masked) or (b) they become empty. Because the rows and labels are
identical, reproduction/prepare_data.py (seed 42) gives the identical train/val/test partition for every
variant, so before/after scores are paired.

Run from the Thesis folder:
    python dataset_prep/build_neutral_datasets.py
"""
import io
import random
import re
import sys
import textwrap
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "raw" / "sui_move_dataset_v1.csv"
OUT = REPO / "data" / "v2"
SEED = 42

KEEP_COMMENTS = {"DeadCode", "MissingDocumentation"}
KEEP_WHITESPACE = {"Spacing", "IncorrectIndentation"}

LINE_COMMENT = re.compile(r"(?<![/:])//(?!/).*$")
PAIR = re.compile(r"\bmodule\s+(?:module\s+)*(\w+)::(\w+)")
HAS_MODULE_DECL = re.compile(r"^\s*module\b|\bmodule\s+\w+\s*\{", re.M)
STRAY = re.compile(r"(?<![\w:])(?:security|example)::")


# ---------------------------------------------------------------- transforms
def strip_comments(code):
    if "\n" not in code:  # single-line row: a comment can't be told apart from the code after it
        return code
    out = []
    for line in code.split("\n"):
        new = LINE_COMMENT.sub("", line)
        if new != line:
            if not new.strip():
                continue
            new = new.rstrip()
        out.append(new)
    return "\n".join(out)


def flatten(code):
    out = []
    for line in code.splitlines():
        line = " ".join(line.split())
        if line:
            out.append(line + ("\n" if "//" in line else " "))
    return "".join(out).strip()


def rename_and_wrap(code, primary, gen):
    """gen("ns") / gen("mod") return random names. Same original namespace -> same new namespace in a row."""
    pairs = []
    for m in PAIR.finditer(code):
        p = (m.group(1), m.group(2))
        if p not in pairs:
            pairs.append(p)
    ns_map, mapping = {}, {}
    for a, b in pairs:
        ns_map.setdefault(a, gen("ns"))
        mapping[f"{a}::{b}"] = f"{ns_map[a]}::{gen('mod')}"
    if mapping:
        alt = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
        code = re.sub(rf"(?<![\w:])(?:{alt})(?!\w)", lambda m: mapping[m.group(0)], code)
    if STRAY.search(code):
        ns = next(iter(ns_map.values()), None) or gen("ns")
        code = STRAY.sub(ns + "::", code)
    if not pairs and primary != "SyntaxError" and not HAS_MODULE_DECL.search(code):
        name = f"{gen('ns')}::{gen('mod')}"
        if "\n" in code or LINE_COMMENT.search(code.rsplit("\n", 1)[-1]):
            code = f"module {name} {{\n{code}\n}}"
        else:
            code = f"module {name} {{ {code} }}"
    return code


def transform(code, primary, sub, modules, comments, newlines, gen):
    if comments and sub not in KEEP_COMMENTS:
        code = strip_comments(code)
    if newlines and sub not in KEEP_WHITESPACE:
        code = flatten(code)
    elif comments and sub not in KEEP_COMMENTS:
        code = textwrap.dedent(code).strip()
    if modules:
        code = rename_and_wrap(code, primary, gen)
    return code.strip()


# ---------------------------------------------------------------- audit
def audit(name, codes, prim):
    d = pd.DataFrame({"code": codes.values, "p": prim.values})
    cv = CountVectorizer(binary=True, lowercase=False, ngram_range=(1, 2),
                         token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]")
    X = cv.fit_transform(d["code"])
    words = np.array(cv.get_feature_names_out())
    print(f"\n--- audit: {name} ({len(d)} rows)")
    tab = pd.DataFrame({
        "has_module": d.code.str.contains(r"\bmodule\b").groupby(d.p).mean(),
        "has_newline": d.code.str.contains("\n").groupby(d.p).mean(),
        "has_//comment": d.code.str.contains(r"(?<![/:])//(?!/)", regex=True).groupby(d.p).mean(),
        "median_chars": d.code.str.len().groupby(d.p).median(),
    }).round(2)
    print(tab.to_string())
    for c in sorted(d.p.unique()):
        m = (d.p == c).values
        inn = np.asarray(X[m].sum(0)).ravel() / m.sum()
        out = np.asarray(X[~m].sum(0)).ravel()
        ok = (inn >= 0.95) & (out == 0)
        if ok.sum():
            print(f"  LEAK {c}: {ok.sum()} tokens on >=95% of the class and in no other class, e.g. "
                  f"{list(words[ok][:5])}")
    print("  (no 'LEAK' lines = no word/word-pair that is on >=95% of a class and absent elsewhere)")


def braces(s):
    return s.count("{") - s.count("}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    code = df["Code"].astype(str)
    primary = df["Label"].str.split("/").str[0].str.strip().str.replace(" ", "", regex=False)
    sub = df["Label"].str.split("/", n=1).str[1].fillna("").str.strip()

    # Pool of pronounceable nonsense names shared by every class.
    rng = random.Random(SEED)
    syl = ["ba", "ko", "ri", "tu", "mel", "dax", "vo", "zen", "pa", "lu", "fi", "gor", "hu", "sa", "ne", "qui"]
    pool = []
    while len(pool) < 60:
        w = "".join(rng.choice(syl) for _ in range(3))
        if w not in pool:
            pool.append(w)
    ns_pool, mod_pool = pool[:30], pool[30:]

    def make_gen():
        return lambda kind: rng.choice(ns_pool if kind == "ns" else mod_pool)

    mask = lambda kind: "N" if kind == "ns" else "M"  # noqa: E731  (names masked to find duplicates)

    # 1. Decide the shared row set from the fully neutral version.
    key = [" ".join(transform(c, p, s, True, True, True, mask).split()) for c, p, s in zip(code, primary, sub)]
    key = pd.Series(key)
    empty = key.str.len() == 0
    dup = key.duplicated() & ~empty
    # label conflicts among duplicates
    conflicts = (pd.DataFrame({"k": key[dup | key.duplicated(keep=False)], "p": primary})
                 .dropna().groupby("k")["p"].nunique() > 1).sum()
    keep = ~(empty | dup)
    print(f"Source rows: {len(df)}")
    print(f"Dropped as empty after fixes: {empty.sum()}")
    print(f"Dropped as duplicates once all fixes are applied: {dup.sum()} "
          f"(duplicate groups with conflicting labels: {conflicts})")
    print(f"Rows in every v2 dataset: {keep.sum()}")
    print("Class counts kept (original -> v2):")
    print(pd.DataFrame({"original": primary.value_counts(), "v2": primary[keep].value_counts()}).to_string())

    variants = {
        "neutral_all": (True, True, True),
        "only_modules": (True, False, False),
        "only_comments": (False, True, False),
        "only_newlines": (False, False, True),
    }
    idx = np.flatnonzero(keep.values)
    pd.DataFrame({"new_row": range(len(idx)), "original_row": idx}).to_csv(OUT / "row_map.csv", index=False)

    # Paired control: same rows, same order, original text untouched.
    df.iloc[idx].to_csv(OUT / "control_unmodified.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote control_unmodified.csv ({len(idx)} rows, original text)")

    built = {}
    for name, (mo, co, nl) in variants.items():
        gen = make_gen()
        new_code = [transform(code[i], primary[i], sub[i], mo, co, nl, gen) for i in idx]
        out = df.iloc[idx].copy().reset_index(drop=True)
        out["Code"] = new_code
        out.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
        built[name] = out
        changed = (out["Code"] != code.iloc[idx].str.strip().values).sum()
        print(f"\nWrote {name}.csv  ({len(out)} rows, {changed} rows differ from the original)")

    # 2. Sanity checks
    prim_k = primary.iloc[idx].reset_index(drop=True)
    print("\n=== Sanity checks ===")
    for name, out in built.items():
        same_labels = (out["Label"].values == df["Label"].iloc[idx].values).all()
        syn = prim_k == "SyntaxError"
        orig_b = code.iloc[idx].reset_index(drop=True)[syn].map(braces)
        new_b = out["Code"][syn].map(braces)
        wrapped = out["Code"].str.contains(r"\bmodule\b")
        print(f"{name}: labels unchanged={same_labels}; empty rows={(out['Code'].str.len() == 0).sum()}; "
              f"SyntaxError rows whose brace balance changed: {(orig_b.values != new_b.values).sum()}"
              + (f"; rows containing 'module' by class: "
                 f"{wrapped.groupby(prim_k).mean().round(2).to_dict()}" if name in ('neutral_all', 'only_modules') else ""))
    exempt_c = built["only_comments"]["Code"].str.contains(r"(?<![/:])//(?!/)", regex=True)
    single_line_left = (exempt_c & ~built["only_comments"]["Code"].str.contains("\n")).sum()
    print(f"Rows with a // comment that could NOT be stripped because the row is a single line: {single_line_left}")

    # 3. Examples
    print("\n=== One example per variant (Security row, then Perfect row) ===")
    for cls in ("SecurityError", "Perfect"):
        j = prim_k[(prim_k == cls) & df["Code"].iloc[idx].reset_index(drop=True).str.contains("//|\n")].index[3]
        print(f"\n##### original {cls}:\n{code.iloc[idx[j]]}")
        for name, out in built.items():
            print(f"\n----- {name}:\n{out['Code'][j]}")

    # 4. Leak audit: original vs each variant
    audit("ORIGINAL (same rows)", code.iloc[idx].str.strip().reset_index(drop=True), prim_k)
    for name, out in built.items():
        audit(name, out["Code"], prim_k)


if __name__ == "__main__":
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        main()
    (OUT / "BUILD_REPORT.txt").write_text(buf.getvalue())
