import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from run_random_forest_baselines import (
    classification_row,
    ranking_row,
    regression_row,
    tune_classifier,
    tune_regressor,
)
from run_features import make_text, read_records

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
UNSUP_ART = ART
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


def evaluate_feature_set(
    name, x_train, x_test, y_train, y_test, cites_train, cites_test, reg_params, clf_params
):
    rows_reg = []
    rows_rank = []
    rows_cls = []

    rf_reg = RandomForestRegressor(random_state=SEED, n_jobs=-1, **reg_params)
    rf_reg.fit(x_train, y_train)
    pred_train = rf_reg.predict(x_train)
    pred_test = rf_reg.predict(x_test)

    for row in [
        regression_row("random_forest", "train", y_train, pred_train),
        regression_row("random_forest", "test", y_test, pred_test),
    ]:
        row["feature_set"] = name
        rows_reg.append(row)

    threshold = float(np.quantile(cites_train, 0.9))
    for row in [
        ranking_row("random_forest", "train", y_train, cites_train, pred_train, threshold),
        ranking_row("random_forest", "test", y_test, cites_test, pred_test, threshold),
    ]:
        row["feature_set"] = name
        rows_rank.append(row)

    high_train = (cites_train >= threshold).astype(int)
    high_test = (cites_test >= threshold).astype(int)
    rf_clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **clf_params)
    rf_clf.fit(x_train, high_train)
    prob_train = rf_clf.predict_proba(x_train)[:, 1]
    prob_test = rf_clf.predict_proba(x_test)[:, 1]
    pred_train_cls = (prob_train >= 0.5).astype(int)
    pred_test_cls = (prob_test >= 0.5).astype(int)

    for row in [
        classification_row(
            "random_forest",
            "train",
            high_train,
            prob_train,
            pred_train_cls,
            threshold,
        ),
        classification_row(
            "random_forest",
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
    parser.add_argument("--tune", action="store_true", help="run RandomizedSearchCV before fitting")
    parser.add_argument("--n-iter", type=int, default=30, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500, help="used when --tune is not set")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument(
        "--reuse-baseline-params",
        action="store_true",
        help="reuse the tuned hyperparameters saved in artifacts/rf_config.json by "
        "run_random_forest_baselines.py instead of tuning again here",
    )
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

    reg_params = dict(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, max_features=args.max_features,
    )
    clf_params = dict(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, max_features=args.max_features,
        class_weight="balanced",
    )
    if args.reuse_baseline_params:
        cfg_path = require(
            ART / "rf_config.json", "Run run_random_forest_baselines.py --tune first."
        )
        with open(cfg_path) as fh:
            saved = json.load(fh)
        reg_params = saved["regressor_params"]
        clf_params = saved["classifier_params"]
        print(f"Reusing tuned hyperparameters from {cfg_path}", flush=True)
    elif args.tune:
        print("Tuning Random Forest regressor on base features ...", flush=True)
        reg_params = tune_regressor(x_train, y_train, args.n_iter, args.cv_folds)
        threshold = float(np.quantile(cites_train, 0.9))
        high_train = (cites_train >= threshold).astype(int)
        print("Tuning Random Forest classifier on base features ...", flush=True)
        clf_params = tune_classifier(x_train, high_train, args.n_iter, args.cv_folds)

    all_reg, all_rank, all_cls = [], [], []
    for name, tr, te in [
        ("base", x_train, x_test),
        ("base_plus_unsup_text_clusters", x_train_aug, x_test_aug),
    ]:
        print(f"Fitting {name} Random Forest (regression + classification) ...", flush=True)
        reg, rank, cls = evaluate_feature_set(
            name, tr, te, y_train, y_test, cites_train, cites_test, reg_params, clf_params
        )
        all_reg.extend(reg)
        all_rank.extend(rank)
        all_cls.extend(cls)

    pd.DataFrame(all_reg).to_csv(
        TABLES / "rf_regression_metrics_cluster_compare.csv", index=False
    )
    pd.DataFrame(all_rank).to_csv(
        TABLES / "rf_ranking_metrics_cluster_compare.csv", index=False
    )
    pd.DataFrame(all_cls).to_csv(
        TABLES / "rf_classification_metrics_cluster_compare.csv", index=False
    )
    with open(ART / "rf_cluster_augmented_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "cluster_source": str(UNSUP_ART / "kmeans_text.joblib"),
                "cluster_space": "unsupervised text LSA",
                "n_clusters": int(kmeans.n_clusters),
                "regressor_params": reg_params,
                "classifier_params": clf_params,
            },
            fh,
            indent=2,
        )
    print(f"Done in {time.time() - t0:.0f}s -> {TABLES}", flush=True)


if __name__ == "__main__":
    main()
