"""TF-IDF + LogReg / LinearSVC / RandomForest, each with and without balanced weights (thesis §4.2, Table 2).

Settings copied from upstream src/simple_baseline.py.
"""

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import DATA_DIR, SEED, compute_metrics, save_run

CLASSIFIERS = {
    "logreg": lambda cw: LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, class_weight=cw),
    "svm": lambda cw: LinearSVC(max_iter=2000, C=1.0, random_state=SEED, dual="auto", class_weight=cw),
    "rf": lambda cw: RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight=cw),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=["all", *CLASSIFIERS], default="all",
                        help="run one classifier with standard and balanced class weights")
    args = parser.parse_args()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    labels = sorted(train["primary_label"].unique())

    for name, make in CLASSIFIERS.items():
        if args.classifier != "all" and name != args.classifier:
            continue
        for cw in (None, "balanced"):
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                                          sublinear_tf=True, strip_accents="unicode")),
                ("clf", make(cw)),
            ])
            pipe.fit(train["code"], train["primary_label"])
            pred = pipe.predict(test["code"]).tolist()
            tag = f"tfidf_{name}" + ("_balanced" if cw else "")
            y_true = test["primary_label"].tolist()
            save_run(tag, compute_metrics(y_true, pred, labels), y_true, pred, labels)


if __name__ == "__main__":
    main()
