"""
Train two XGBoost cloud-mask models on the same feature table.

Baseline (cf > 0.5) reproduces the course training target on the augmented
data, soft-label (cf > 0.2) is the proposed retrain that targets the
thin-tropical-cirrus-over-land blind spot identified in the microwave land-surface emissivity diagnostic.

Both models share the saved-model hyperparameters (n_estimators=900,
max_depth=7, learning_rate=0.01, objective=binary:logistic) so any difference
in behaviour is attributable to the label threshold.

Run:
    .venv/bin/python scripts/train_models.py \
        --table data/training_jan_jul_2000.parquet \
        --out-baseline model/xgboost_v3_cf_gt_0p5.pkl \
        --out-softlabel model/xgboost_v3_cf_gt_0p2.pkl
"""

import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


FEATURE_ORDER = ["t1", "t2", "t3", "tclr", "sobel", "snoice", "sfc", "t21", "t23", "dt"]
HYPERPARAMS = dict(
    n_estimators=900,
    max_depth=7,
    learning_rate=0.01,
    objective="binary:logistic",
    n_jobs=-1,
    tree_method="hist",
)


def train_one(name, X_train, y_train, X_test, y_test, seed=0):
    print(f"\n=== {name} (positive fraction train={y_train.mean():.3f}, test={y_test.mean():.3f}) ===")
    model = XGBClassifier(**HYPERPARAMS, random_state=seed)
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  fit time: {time.time() - t0:.1f} s")
    for label, X_, y_ in (("train", X_train, y_train), ("test", X_test, y_test)):
        y_hat = model.predict(X_)
        acc = accuracy_score(y_, y_hat)
        cm = confusion_matrix(y_, y_hat, normalize="all")
        # cm[0,0]=clear (TN), cm[0,1]=false cloud (FP), cm[1,0]=false clear (FN), cm[1,1]=cloud (TP)
        print(f"  {label}: accuracy={acc:.4f}")
        print(f"    clear (TN)  = {cm[0,0]:.4f}    false cloud (FP) = {cm[0,1]:.4f}")
        print(f"    false clear (FN) = {cm[1,0]:.4f}    cloud (TP)  = {cm[1,1]:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/training_jan_jul_2000.parquet")
    ap.add_argument("--out-baseline", default="model/xgboost_v3_cf_gt_0p5.pkl")
    ap.add_argument("--out-softlabel", default="model/xgboost_v3_cf_gt_0p2.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    print(f"loaded {len(df):,} rows from {args.table}")
    X = df[FEATURE_ORDER]
    cf = df["cf"]

    # Same 20/80 split for both models, so the test-set is identical.
    X_train, X_test, cf_train, cf_test = train_test_split(
        X, cf, test_size=0.8, random_state=args.seed, shuffle=True,
    )

    # Baseline cf > 0.5
    y05_train = (cf_train > 0.5).astype(int)
    y05_test = (cf_test > 0.5).astype(int)
    m05 = train_one("baseline cf>0.5", X_train, y05_train, X_test, y05_test, seed=args.seed)
    with open(args.out_baseline, "wb") as f:
        pickle.dump(m05, f)
    print(f"  saved {args.out_baseline}")

    # Soft-label cf > 0.2
    y02_train = (cf_train > 0.2).astype(int)
    y02_test = (cf_test > 0.2).astype(int)
    m02 = train_one("soft-label cf>0.2", X_train, y02_train, X_test, y02_test, seed=args.seed)
    with open(args.out_softlabel, "wb") as f:
        pickle.dump(m02, f)
    print(f"  saved {args.out_softlabel}")


if __name__ == "__main__":
    main()
