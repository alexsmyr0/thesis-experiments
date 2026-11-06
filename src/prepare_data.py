"""Clean the CSV and make the stratified 70/15/15 split (thesis §3.4, upstream src/consolidate.py).

Same steps as upstream: strip whitespace, drop leaked header rows, split
Primary/SubLabel, remove spaces from the primary label ("Syntax Error" ->
"SyntaxError"), dedupe on SHA-256 of whitespace-normalised code, then two
stratified train_test_split calls with seed 42.
"""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_DIR, RAW_CSV, SEED


def fingerprint(code):
    return hashlib.sha256(" ".join(code.split()).encode("utf-8")).hexdigest()


def main():
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"output/fix": "output"})
    print(f"Loaded {len(df)} rows")

    df = df.dropna(subset=["code", "label"]).copy()
    for col in ("code", "label", "output"):
        df[col] = df[col].astype(str).str.strip()

    headers = (df["code"] == "Code") & (df["label"] == "Label")
    print(f"Removed {headers.sum()} leaked header rows")
    df = df[~headers]

    df["primary_label"] = df["label"].str.split("/").str[0].str.strip().str.replace(" ", "")
    df["sublabel"] = df["label"].apply(lambda x: x.split("/", 1)[1].strip() if "/" in x else None)

    before = len(df)
    df = df.loc[~df["code"].map(fingerprint).duplicated()].reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate snippets -> {len(df)} samples")

    train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["primary_label"])
    train, val = train_test_split(train, test_size=0.15 / 0.85, random_state=SEED,
                                  stratify=train["primary_label"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "full_clean.csv", index=False)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.reset_index(drop=True).to_csv(DATA_DIR / f"{name}.csv", index=False)

    table = pd.DataFrame({n: p["primary_label"].value_counts()
                          for n, p in (("train", train), ("val", val), ("test", test))})
    table["total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum()
    print("\nSplit sizes (compare with thesis Table 1):")
    print(table.to_string())


if __name__ == "__main__":
    main()
