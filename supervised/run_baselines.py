"""Run phase-1 supervised baselines: Ridge and Logistic Regression."""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
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
    roc_curve,
    roc_auc_score,
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
    """Evaluate the model as a ranker over one split.

    Precision/recall use the train-derived top-decile citation threshold as
    binary relevance. NDCG uses log-citations as graded relevance.
    """
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


def save_regression_figures(y_true, y_pred):
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
    ax.set_title("Ridge baseline: predicted vs true")
    fig.tight_layout()
    fig.savefig(FIGS / "ridge_pred_vs_true.png", dpi=200)
    plt.close(fig)

    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(residuals, bins=60, color="#2a78d6", alpha=0.85)
    ax.axvline(0, color="#52514e", lw=1.2)
    ax.set_xlabel("prediction error on log scale")
    ax.set_ylabel("papers")
    ax.set_title("Ridge baseline residuals")
    fig.tight_layout()
    fig.savefig(FIGS / "ridge_residuals.png", dpi=200)
    plt.close(fig)


def save_pr_curve(y_true, prob, ap):
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, prob)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(recall, precision, color="#1baf7a", lw=2)
    ax.axhline(np.mean(y_true), color="#52514e", lw=1.0, ls="--", label="test prevalence")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Logistic baseline PR curve (AP = {ap:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "logistic_pr_curve.png", dpi=200)
    plt.close(fig)


def save_roc_curve(y_true, prob, auc):
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, prob)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, color="#2a78d6", lw=2)
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1.0, ls="--", label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"Logistic baseline ROC curve (AUC = {auc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "logistic_roc_curve.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
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

    print("Fitting Ridge regression ...", flush=True)
    ridge = Ridge(alpha=args.ridge_alpha)
    ridge.fit(x_train, y_train)
    pred_train = ridge.predict(x_train)
    pred_test = ridge.predict(x_test)
    reg_rows = [
        regression_row("ridge", "train", y_train, pred_train),
        regression_row("ridge", "test", y_test, pred_test),
    ]
    pd.DataFrame(reg_rows).to_csv(TABLES / "regression_metrics.csv", index=False)
    np.save(ART / "ridge_pred_train.npy", pred_train.astype(np.float32))
    np.save(ART / "ridge_pred_test.npy", pred_test.astype(np.float32))
    save_regression_figures(y_test, pred_test)
    joblib.dump(ridge, ART / "ridge_model.joblib")

    threshold = float(np.quantile(cites_train, 0.9))
    rank_rows = [
        ranking_row("ridge", "train", y_train, cites_train, pred_train, threshold),
        ranking_row("ridge", "test", y_test, cites_test, pred_test, threshold),
    ]
    pd.DataFrame(rank_rows).to_csv(TABLES / "ranking_metrics.csv", index=False)

    high_train = (cites_train >= threshold).astype(int)
    high_test = (cites_test >= threshold).astype(int)
    print(
        f"Fitting Logistic Regression (top-decile threshold={threshold:g}) ...",
        flush=True,
    )
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

    cls_rows = [
        classification_row(
            "logistic_regression", "train", high_train, prob_train, pred_train_cls, threshold
        ),
        classification_row(
            "logistic_regression", "test", high_test, prob_test, pred_test_cls, threshold
        ),
        classification_row(
            "prevalence_baseline",
            "test",
            high_test,
            np.full_like(high_test, fill_value=np.mean(high_train), dtype=float),
            np.zeros_like(high_test),
            threshold,
        ),
    ]
    pd.DataFrame(cls_rows).to_csv(TABLES / "classification_metrics.csv", index=False)
    save_pr_curve(
        high_test,
        prob_test,
        average_precision_score(high_test, prob_test),
    )
    save_roc_curve(high_test, prob_test, roc_auc_score(high_test, prob_test))
    joblib.dump(logistic, ART / "logistic_model.joblib")

    with open(ART / "baseline_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ridge_alpha": args.ridge_alpha,
                "logistic_c": args.logistic_c,
                "highly_cited_threshold": threshold,
            },
            fh,
            indent=2,
        )
    print(f"Done in {time.time() - t0:.0f}s -> {TABLES} and {FIGS}", flush=True)


if __name__ == "__main__":
    main()
