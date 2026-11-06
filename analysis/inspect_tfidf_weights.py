"""Show which words the TF-IDF + LinearSVC model relies on most, per category.

Read-only: trains the same model as src/tfidf_baselines.py (svm, no class weights)
on experiments/reproduction_v1/data/train.csv (made by prepare_data.py) and prints the weights. Changes no files.

Run from the repo root:
    python analysis/inspect_tfidf_weights.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

TOP = 10

train = pd.read_csv(Path(__file__).resolve().parent.parent / "experiments" / "reproduction_v1" / "data" / "train.csv")

# Step 1: turn each snippet into 10,000 numbers (the representation).
vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                      sublinear_tf=True, strip_accents="unicode")
X = vec.fit_transform(train["code"])

# Step 2: learn the weights (the classifier).
clf = LinearSVC(max_iter=2000, C=1.0, random_state=42, dual="auto")
clf.fit(X, train["primary_label"])

words = np.array(vec.get_feature_names_out())

print(f"Snippets used for training: {X.shape[0]}")
print(f"Numbers per snippet (words and word pairs): {X.shape[1]}")
print(f"Weights table shape (categories x words): {clf.coef_.shape}")
print(f"Bias per category: {dict(zip(clf.classes_, clf.intercept_.round(2)))}")

for i, label in enumerate(clf.classes_):
    w = clf.coef_[i]
    order = np.argsort(w)
    print(f"\n=== {label} ===")
    print("Words pushing TOWARD this category (largest positive weights):")
    for j in order[::-1][:TOP]:
        print(f"  {w[j]:+.2f}  {words[j]!r}")
    print("Words pushing AWAY from it (most negative weights):")
    for j in order[:TOP]:
        print(f"  {w[j]:+.2f}  {words[j]!r}")
