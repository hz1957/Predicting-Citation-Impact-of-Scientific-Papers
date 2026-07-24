import matplotlib
matplotlib.use("Agg")  # MUST be at the very top before any other imports!

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.parallel import Parallel, delayed

# Silence sklearn parallel config warnings cleanly
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TABLES = HERE / "outputs" / "tables"
FIGS = HERE / "outputs" / "figures"
SEED = 42


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#c3c2b7",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.labelcolor": "#52514e",
            "xtick.color": "#898781",
            "ytick.color": "#898781",
        }
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_row(model_name, split, y_true, y_pred):
    rho = spearmanr(y_true, y_pred).statistic
    return {
        "model": model_name,
        "split": split,
        "rmse_log": rmse(y_true, y_pred),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
        "r2_log": float(r2_score(y_true, y_pred)),
        "spearman_log": float(rho) if np.isfinite(rho) else np.nan,
    }


def classification_row(model_name, split, y_true, prob, pred, threshold):
    row = {
        "model": model_name,
        "split": split,
        "positive_rate": float(np.mean(y_true)),
        "threshold_citations": float(threshold),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }
    if len(np.unique(y_true)) == 2:
        row["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        row["roc_auc"] = np.nan
    return row


def dcg_at_k(relevance, order, k):
    rel = np.asarray(relevance, dtype=float)[order[:k]]
    discounts = np.log2(np.arange(2, k + 2))
    return float(np.sum(rel / discounts))


def ranking_row(model_name, split, y_log, citations, scores, threshold):
    n = len(scores)
    k = max(1, int(np.ceil(0.10 * n)))
    order = np.argsort(scores)[::-1]
    relevant = citations >= threshold
    selected = relevant[order[:k]]
    n_relevant = int(relevant.sum())
    precision = float(selected.mean())
    recall = float(selected.sum() / n_relevant) if n_relevant else np.nan
    ideal_order = np.argsort(y_log)[::-1]
    ideal_dcg = dcg_at_k(y_log, ideal_order, k)
    ndcg = dcg_at_k(y_log, order, k) / ideal_dcg if ideal_dcg > 0 else np.nan
    prevalence = float(relevant.mean())
    return {
        "model": model_name,
        "split": split,
        "k_fraction": 0.10,
        "k_papers": int(k),
        "threshold_citations": float(threshold),
        "positive_rate": prevalence,
        "precision_at_10pct": precision,
        "recall_at_10pct": recall,
        "ndcg_at_10pct": float(ndcg),
        "lift_at_10pct": float(precision / prevalence) if prevalence > 0 else np.nan,
    }


def save_regression_figures(y_true, y_pred, model_label, file_prefix):
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    n = len(y_true)
    idx = rng.choice(n, size=min(6000, n), replace=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(y_true[idx], y_pred[idx], s=8, alpha=0.22, color="#2a78d6", linewidths=0)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1.2, ls="--")
    ax.set_xlabel("true log(1 + citations)")
    ax.set_ylabel("predicted log(1 + citations)")
    ax.set_title(f"{model_label}: predicted vs true")
    fig.tight_layout()
    fig.savefig(FIGS / f"{file_prefix}_pred_vs_true.png", dpi=200)
    plt.close(fig)

    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(residuals, bins=60, color="#2a78d6", alpha=0.85)
    ax.axvline(0, color="#52514e", lw=1.2)
    ax.set_xlabel("prediction error on log scale")
    ax.set_ylabel("papers")
    ax.set_title(f"{model_label} residuals")
    fig.tight_layout()
    fig.savefig(FIGS / f"{file_prefix}_residuals.png", dpi=200)
    plt.close(fig)


def save_pr_curve(y_true, prob, ap, model_label, file_prefix):
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, prob)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(recall, precision, color="#1baf7a", lw=2)
    ax.axhline(np.mean(y_true), color="#52514e", lw=1.0, ls="--", label="test prevalence")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"{model_label} PR curve (AP = {ap:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / f"{file_prefix}_pr_curve.png", dpi=200)
    plt.close(fig)


def save_roc_curve(y_true, prob, auc, model_label, file_prefix):
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, prob)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, color="#2a78d6", lw=2)
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1.0, ls="--", label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{model_label} ROC curve (AUC = {auc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / f"{file_prefix}_roc_curve.png", dpi=200)
    plt.close(fig)


def to_dense_if_needed(X):
    return X.toarray() if sparse.issparse(X) else X


def tune_regressor(x_train, y_train, n_iter, cv_folds):
    param_dist = {
        "n_estimators": [200, 300, 500, 700, 900],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=1),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    print(f"  best params: {search.best_params_}", flush=True)
    print(f"  best CV RMSE: {-search.best_score_:.4f}", flush=True)
    return search.best_params_


def tune_classifier(x_train, y_train, n_iter, cv_folds):
    param_dist = {
        "n_estimators": [200, 300, 500, 700, 900],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
        scoring="roc_auc",
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    print(f"  best params: {search.best_params_}", flush=True)
    print(f"  best CV ROC-AUC: {search.best_score_:.4f}", flush=True)
    return search.best_params_


def feature_importance_table(model, x_test, y_test, feature_names, scoring, top_n):
    impurity = model.feature_importances_
    perm = permutation_importance(
        model, to_dense_if_needed(x_test), y_test, n_repeats=10,
        random_state=SEED, n_jobs=-1, scoring=scoring,
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "impurity_importance": impurity,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    return df.head(top_n)


def get_feature_names(n_features):
    try:
        transformers = joblib.load(ART / "transformers.joblib")
    except FileNotFoundError:
        return [f"feature_{i}" for i in range(n_features)]

    names = []
    encoder = transformers.get("encoder") or transformers.get("cat_encoder")
    if encoder is not None and hasattr(encoder, "get_feature_names_out"):
        names.extend(list(encoder.get_feature_names_out()))
    scaler = transformers.get("scaler") or transformers.get("numeric_scaler")
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        names.extend(list(scaler.feature_names_in_))
    svd = transformers.get("svd") or transformers.get("truncated_svd")
    if svd is not None:
        names.extend([f"LSA_{i}" for i in range(svd.n_components)])

    if len(names) != n_features:
        names = [f"feature_{i}" for i in range(n_features)]
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="run RandomizedSearchCV before fitting")
    parser.add_argument("--n-iter", type=int, default=30, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500, help="used when --tune is not set")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--top-n-features", type=int, default=30)
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading supervised features ...", flush=True)
    x_train = sparse.load_npz(ART / "X_train.npz")
    x_test = sparse.load_npz(ART / "X_test.npz")
    y_train = np.load(ART / "y_train.npy")
    y_test = np.load(ART / "y_test.npy")
    cites_train = np.load(ART / "target_citations_train.npy")
    cites_test = np.load(ART / "target_citations_test.npy")
    print(f"  train/test: {x_train.shape} / {x_test.shape}", flush=True)

    reg_params = dict(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, max_features=args.max_features,
    )
    if args.tune:
        print("Tuning Random Forest regressor ...", flush=True)
        reg_params = tune_regressor(x_train, y_train, args.n_iter, args.cv_folds)

    print("Fitting Random Forest regression ...", flush=True)
    rf_reg = RandomForestRegressor(random_state=SEED, n_jobs=-1, **reg_params)
    rf_reg.fit(x_train, y_train)
    pred_train = rf_reg.predict(x_train)
    pred_test = rf_reg.predict(x_test)
    reg_rows = [
        regression_row("random_forest", "train", y_train, pred_train),
        regression_row("random_forest", "test", y_test, pred_test),
    ]
    pd.DataFrame(reg_rows).to_csv(TABLES / "rf_regression_metrics.csv", index=False)
    np.save(ART / "rf_pred_train.npy", pred_train.astype(np.float32))
    np.save(ART / "rf_pred_test.npy", pred_test.astype(np.float32))
    save_regression_figures(y_test, pred_test, "Random Forest baseline", "rf")
    joblib.dump(rf_reg, ART / "rf_regressor.joblib")

    threshold = float(np.quantile(cites_train, 0.9))
    rank_rows = [
        ranking_row("random_forest", "train", y_train, cites_train, pred_train, threshold),
        ranking_row("random_forest", "test", y_test, cites_test, pred_test, threshold),
    ]
    pd.DataFrame(rank_rows).to_csv(TABLES / "rf_ranking_metrics.csv", index=False)

    high_train = (cites_train >= threshold).astype(int)
    high_test = (cites_test >= threshold).astype(int)

    clf_params = dict(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, max_features=args.max_features,
        class_weight="balanced",
    )
    if args.tune:
        print("Tuning Random Forest classifier ...", flush=True)
        clf_params = tune_classifier(x_train, high_train, args.n_iter, args.cv_folds)

    print(
        f"Fitting Random Forest classification (top-decile threshold={threshold:g}) ...",
        flush=True,
    )
    rf_clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **clf_params)
    rf_clf.fit(x_train, high_train)
    prob_train = rf_clf.predict_proba(x_train)[:, 1]
    prob_test = rf_clf.predict_proba(x_test)[:, 1]
    pred_train_cls = (prob_train >= 0.5).astype(int)
    pred_test_cls = (prob_test >= 0.5).astype(int)

    cls_rows = [
        classification_row("random_forest", "train", high_train, prob_train, pred_train_cls, threshold),
        classification_row("random_forest", "test", high_test, prob_test, pred_test_cls, threshold),
        classification_row(
            "prevalence_baseline",
            "test",
            high_test,
            np.full_like(high_test, fill_value=np.mean(high_train), dtype=float),
            np.zeros_like(high_test),
            threshold,
        ),
    ]
    pd.DataFrame(cls_rows).to_csv(TABLES / "rf_classification_metrics.csv", index=False)
    save_pr_curve(
        high_test, prob_test, average_precision_score(high_test, prob_test),
        "Random Forest baseline", "rf",
    )
    save_roc_curve(
        high_test, prob_test, roc_auc_score(high_test, prob_test),
        "Random Forest baseline", "rf",
    )
    joblib.dump(rf_clf, ART / "rf_classifier.joblib")

    print("Computing feature importance ...", flush=True)
    feature_names = get_feature_names(x_train.shape[1])
    reg_importance = feature_importance_table(
        rf_reg, x_test, y_test, feature_names,
        scoring="neg_root_mean_squared_error", top_n=args.top_n_features,
    )
    reg_importance.to_csv(TABLES / "rf_regression_feature_importance.csv", index=False)
    clf_importance = feature_importance_table(
        rf_clf, x_test, high_test, feature_names,
        scoring="roc_auc", top_n=args.top_n_features,
    )
    clf_importance.to_csv(TABLES / "rf_classification_feature_importance.csv", index=False)

    with open(ART / "rf_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "regressor_params": reg_params,
                "classifier_params": clf_params,
                "highly_cited_threshold": threshold,
                "tuned": args.tune,
            },
            fh,
            indent=2,
        )
    
    # Close figures and force cleanup prior to process exit
    plt.close('all')
    gc.collect()

    print(f"Done in {time.time() - t0:.0f}s -> {TABLES} and {FIGS}", flush=True)


if __name__ == "__main__":
    main()