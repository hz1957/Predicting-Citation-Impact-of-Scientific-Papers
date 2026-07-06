"""Compare supervised baselines with unsupervised cluster features added.

This script does not define or fit a new clustering inside supervised code.
Instead, it loads the text-only K-Means artifacts exported by
unsupervised/run_clustering.py, applies that train-fit cluster model to the
c1 train/test papers, and appends:

  1. cluster id as one-hot features
  2. distances to every cluster centroid

The original run_baselines.py remains the base phase-1 baseline.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge

from run_baselines import classification_row, ranking_row, regression_row
from run_features import make_text, read_records

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
UNSUP_ART = ROOT / "unsupervised" / "artifacts"
TABLES = HERE / "outputs" / "tables"
SEED = 42


def require(path, hint):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")
    return path


def load_split(data_path):
    df = read_records(data_path)
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Expected non-empty c1 train/test split labels.")
    return train, test


def unsup_lsa(train, test):
    tfidf = joblib.load(
        require(
            UNSUP_ART / "tfidf_vectorizer.joblib",
            "Run unsupervised/run_features.py first.",
        )
    )
    lsa = joblib.load(
        require(
            UNSUP_ART / "text_lsa_transformer.joblib",
            "Run unsupervised/run_clustering.py first.",
        )
    )
    train_tfidf = tfidf.transform(make_text(train))
    test_tfidf = tfidf.transform(make_text(test))
    train_lsa = lsa["normalizer"].transform(lsa["svd"].transform(train_tfidf))
    test_lsa = lsa["normalizer"].transform(lsa["svd"].transform(test_tfidf))
    return train_lsa.astype(np.float32), test_lsa.astype(np.float32)


def cluster_block(kmeans, X):
    labels = kmeans.predict(X)
    one_hot = np.eye(kmeans.n_clusters, dtype=np.float32)[labels]
    distances = kmeans.transform(X).astype(np.float32)
    return labels.astype(np.int32), sparse.csr_matrix(np.hstack([one_hot, distances]))


def feature_names(k):
    return [f"unsup_text_cluster_{i}" for i in range(k)] + [
        f"unsup_text_dist_{i}" for i in range(k)
    ]


def evaluate_feature_set(name, x_train, x_test, y_train, y_test, cites_train, cites_test, args):
    rows_reg = []
    rows_rank = []
    rows_cls = []

    ridge = Ridge(alpha=args.ridge_alpha)
    ridge.fit(x_train, y_train)
    pred_train = ridge.predict(x_train)
    pred_test = ridge.predict(x_test)

    for row in [
        regression_row("ridge", "train", y_train, pred_train),
        regression_row("ridge", "test", y_test, pred_test),
    ]:
        row["feature_set"] = name
        rows_reg.append(row)

    threshold = float(np.quantile(cites_train, 0.9))
    for row in [
        ranking_row("ridge", "train", y_train, cites_train, pred_train, threshold),
        ranking_row("ridge", "test", y_test, cites_test, pred_test, threshold),
    ]:
        row["feature_set"] = name
        rows_rank.append(row)

    high_train = (cites_train >= threshold).astype(int)
    high_test = (cites_test >= threshold).astype(int)
    logistic = LogisticRegression(
        C=args.logistic_c,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    logistic.fit(x_train, high_train)
    prob_train = logistic.predict_proba(x_train)[:, 1]
    prob_test = logistic.predict_proba(x_test)[:, 1]
    pred_train_cls = (prob_train >= 0.5).astype(int)
    pred_test_cls = (prob_test >= 0.5).astype(int)

    for row in [
        classification_row(
            "logistic_regression",
            "train",
            high_train,
            prob_train,
            pred_train_cls,
            threshold,
        ),
        classification_row(
            "logistic_regression",
            "test",
            high_test,
            prob_test,
            pred_test_cls,
            threshold,
        ),
    ]:
        row["feature_set"] = name
        rows_cls.append(row)

    return rows_reg, rows_rank, rows_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="c1")
    parser.add_argument("--data", default=None, help="Override path to data file")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data) if args.data else ROOT / "data" / f"{args.cohort}.jsonl.gz"

    t0 = time.time()
    print("Loading base supervised features ...", flush=True)
    x_train = sparse.load_npz(require(ART / "X_train.npz", "Run supervised/run_features.py."))
    x_test = sparse.load_npz(require(ART / "X_test.npz", "Run supervised/run_features.py."))
    y_train = np.load(require(ART / "y_train.npy", "Run supervised/run_features.py."))
    y_test = np.load(require(ART / "y_test.npy", "Run supervised/run_features.py."))
    cites_train = np.load(
        require(ART / "target_citations_train.npy", "Run supervised/run_features.py.")
    )
    cites_test = np.load(
        require(ART / "target_citations_test.npy", "Run supervised/run_features.py.")
    )

    print("Loading raw split to apply unsupervised text transformers ...", flush=True)
    train, test = load_split(data_path)
    train_lsa, test_lsa = unsup_lsa(train, test)
    kmeans = joblib.load(
        require(UNSUP_ART / "kmeans_text.joblib", "Run unsupervised/run_clustering.py first.")
    )
    labels_train, cluster_train = cluster_block(kmeans, train_lsa)
    labels_test, cluster_test = cluster_block(kmeans, test_lsa)
    x_train_aug = sparse.hstack([x_train, cluster_train], format="csr")
    x_test_aug = sparse.hstack([x_test, cluster_test], format="csr")

    sparse.save_npz(ART / "X_train_plus_unsup_text_clusters.npz", x_train_aug)
    sparse.save_npz(ART / "X_test_plus_unsup_text_clusters.npz", x_test_aug)
    np.save(ART / "unsup_text_cluster_labels_train.npy", labels_train)
    np.save(ART / "unsup_text_cluster_labels_test.npy", labels_test)
    with open(ART / "unsup_text_cluster_feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(feature_names(kmeans.n_clusters), fh, indent=2)

    print(
        f"  base train/test: {x_train.shape} / {x_test.shape}; "
        f"augmented: {x_train_aug.shape} / {x_test_aug.shape}",
        flush=True,
    )

    all_reg, all_rank, all_cls = [], [], []
    for name, tr, te in [
        ("base", x_train, x_test),
        ("base_plus_unsup_text_clusters", x_train_aug, x_test_aug),
    ]:
        print(f"Fitting {name} Ridge + Logistic Regression ...", flush=True)
        reg, rank, cls = evaluate_feature_set(
            name, tr, te, y_train, y_test, cites_train, cites_test, args
        )
        all_reg.extend(reg)
        all_rank.extend(rank)
        all_cls.extend(cls)

    pd.DataFrame(all_reg).to_csv(
        TABLES / "regression_metrics_cluster_compare.csv", index=False
    )
    pd.DataFrame(all_rank).to_csv(
        TABLES / "ranking_metrics_cluster_compare.csv", index=False
    )
    pd.DataFrame(all_cls).to_csv(
        TABLES / "classification_metrics_cluster_compare.csv", index=False
    )
    with open(ART / "cluster_augmented_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "cluster_source": str(UNSUP_ART / "kmeans_text.joblib"),
                "cluster_space": "unsupervised text LSA",
                "n_clusters": int(kmeans.n_clusters),
                "ridge_alpha": args.ridge_alpha,
                "logistic_c": args.logistic_c,
            },
            fh,
            indent=2,
        )
    print(f"Done in {time.time() - t0:.0f}s -> {TABLES}", flush=True)


if __name__ == "__main__":
    main()
