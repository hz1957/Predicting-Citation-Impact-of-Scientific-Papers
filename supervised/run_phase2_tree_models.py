"""Run phase-2 nonlinear models and compare against phase-1 baselines.

Design:
  * XGBoost Classifier predicts the train-defined top-decile label and
    is compared with Logistic Regression.
  * XGBoost Regressor predicts log(1 + citations) and is compared with Ridge.

The script reuses the leakage-safe feature matrices created by
supervised/run_features.py. It writes separate phase-2 comparison outputs so
the original phase-1 baseline tables remain unchanged.
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
MPLCONFIGDIR = HERE / "artifacts" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from run_baselines import (
    apply_style,
    classification_row,
    ranking_row,
    regression_row,
)
from run_features import NUMERIC_META

ART = HERE / "artifacts"
TABLES = HERE / "outputs" / "tables"
FIGS = HERE / "outputs" / "figures"
SEED = 42
MODEL_LABELS = {
    "ridge": "Ridge",
    "xgboost_regressor": "XGBoost",
    "logistic_regression": "Logistic",
    "logistic_regression_val_f1_threshold": "Logistic tuned",
    "xgboost_classifier": "XGB Classifier",
    "xgboost_classifier_val_f1_threshold": "XGB tuned",
}


def require(path, hint):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")
    return path


def load_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing optional dependency 'xgboost'. Install it in the project "
            "environment, then rerun:\n"
            "D:\\conda\\Scripts\\conda.exe install -n cs7641-team7 -c conda-forge "
            "xgboost -y"
        ) from exc
    return XGBClassifier, XGBRegressor


def validation_indices(train_meta):
    """Use the latest training year as a temporal validation fold."""
    years = pd.to_numeric(train_meta["year"], errors="coerce")
    max_year = years.max()
    val_mask = years == max_year
    fit_mask = ~val_mask
    if val_mask.sum() == 0 or fit_mask.sum() == 0:
        raise ValueError("Could not build temporal validation split from train_meta.csv.gz.")
    return np.flatnonzero(fit_mask.to_numpy()), np.flatnonzero(val_mask.to_numpy()), int(max_year)


def xgb_params(args, n_estimators):
    return {
        "n_estimators": int(n_estimators),
        "learning_rate": args.xgb_learning_rate,
        "max_depth": args.xgb_max_depth,
        "min_child_weight": args.xgb_min_child_weight,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
        "reg_alpha": args.xgb_reg_alpha,
        "reg_lambda": args.xgb_reg_lambda,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": args.xgb_tree_method,
        "random_state": SEED,
        "n_jobs": args.n_jobs,
    }


def xgb_classifier_params(args, n_estimators, scale_pos_weight):
    return {
        "n_estimators": int(n_estimators),
        "learning_rate": args.xgb_cls_learning_rate,
        "max_depth": args.xgb_cls_max_depth,
        "min_child_weight": args.xgb_cls_min_child_weight,
        "subsample": args.xgb_cls_subsample,
        "colsample_bytree": args.xgb_cls_colsample_bytree,
        "reg_alpha": args.xgb_cls_reg_alpha,
        "reg_lambda": args.xgb_cls_reg_lambda,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": args.xgb_tree_method,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": SEED,
        "n_jobs": args.n_jobs,
    }


def best_f1_threshold(y_true, prob):
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5, 0.0
    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def add_threshold_info(row, decision_threshold, threshold_source):
    row["decision_threshold"] = float(decision_threshold)
    row["threshold_source"] = threshold_source
    return row


def load_baseline_config():
    path = ART / "baseline_config.json"
    if not path.exists():
        return {"logistic_c": 1.0}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def class_balance_weight(y):
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    return negatives / max(1, positives)


def fit_xgboost_classifier(x_train, high_train, train_meta, args):
    XGBClassifier, _ = load_xgboost()
    fit_idx, val_idx, val_year = validation_indices(train_meta)
    print(
        f"Fitting XGBoost classifier validation model: fit={len(fit_idx):,}, "
        f"validation={len(val_idx):,} (year={val_year}) ...",
        flush=True,
    )

    params = xgb_classifier_params(
        args,
        args.xgb_cls_n_estimators,
        class_balance_weight(high_train[fit_idx]),
    )
    tune_model = XGBClassifier(**params)
    try:
        tune_model.fit(
            x_train[fit_idx],
            high_train[fit_idx],
            eval_set=[(x_train[val_idx], high_train[val_idx])],
            early_stopping_rounds=args.xgb_cls_early_stopping_rounds,
            verbose=False,
        )
    except TypeError:
        params["early_stopping_rounds"] = args.xgb_cls_early_stopping_rounds
        tune_model = XGBClassifier(**params)
        tune_model.fit(
            x_train[fit_idx],
            high_train[fit_idx],
            eval_set=[(x_train[val_idx], high_train[val_idx])],
            verbose=False,
        )

    best_iteration = getattr(tune_model, "best_iteration", None)
    if best_iteration is None:
        best_n_estimators = args.xgb_cls_n_estimators
    else:
        best_n_estimators = int(best_iteration) + 1

    val_prob = tune_model.predict_proba(x_train[val_idx])[:, 1]
    tuned_threshold, val_f1 = best_f1_threshold(high_train[val_idx], val_prob)
    best_score = getattr(tune_model, "best_score", None)

    print(
        f"Refitting XGBoost classifier on full train with "
        f"n_estimators={best_n_estimators} ...",
        flush=True,
    )
    final_params = xgb_classifier_params(
        args,
        best_n_estimators,
        class_balance_weight(high_train),
    )
    final_model = XGBClassifier(**final_params)
    final_model.fit(x_train, high_train, verbose=False)
    config = {
        "validation": "latest train year",
        "validation_year": val_year,
        "validation_rows": int(len(val_idx)),
        "early_stopping_rounds": args.xgb_cls_early_stopping_rounds,
        "best_n_estimators": best_n_estimators,
        "best_validation_aucpr": None if best_score is None else float(best_score),
        "validation_f1_threshold": tuned_threshold,
        "validation_f1": val_f1,
        "params": final_params,
    }
    return final_model, tuned_threshold, config


def fit_xgboost_regressor(x_train, y_train, train_meta, args):
    _, XGBRegressor = load_xgboost()
    fit_idx, val_idx, val_year = validation_indices(train_meta)
    print(
        f"Fitting XGBoost validation model: fit={len(fit_idx):,}, "
        f"validation={len(val_idx):,} (year={val_year}) ...",
        flush=True,
    )

    params = xgb_params(args, args.xgb_n_estimators)
    tune_model = XGBRegressor(**params)
    try:
        tune_model.fit(
            x_train[fit_idx],
            y_train[fit_idx],
            eval_set=[(x_train[val_idx], y_train[val_idx])],
            early_stopping_rounds=args.xgb_early_stopping_rounds,
            verbose=False,
        )
    except TypeError:
        params["early_stopping_rounds"] = args.xgb_early_stopping_rounds
        tune_model = XGBRegressor(**params)
        tune_model.fit(
            x_train[fit_idx],
            y_train[fit_idx],
            eval_set=[(x_train[val_idx], y_train[val_idx])],
            verbose=False,
        )

    best_iteration = getattr(tune_model, "best_iteration", None)
    if best_iteration is None:
        best_n_estimators = args.xgb_n_estimators
    else:
        best_n_estimators = int(best_iteration) + 1

    best_score = getattr(tune_model, "best_score", None)
    print(
        f"Refitting XGBoost on full train with n_estimators={best_n_estimators} ...",
        flush=True,
    )
    final_params = xgb_params(args, best_n_estimators)
    final_model = XGBRegressor(**final_params)
    final_model.fit(x_train, y_train, verbose=False)
    config = {
        "validation": "latest train year",
        "validation_year": val_year,
        "validation_rows": int(len(val_idx)),
        "early_stopping_rounds": args.xgb_early_stopping_rounds,
        "best_n_estimators": best_n_estimators,
        "best_validation_rmse": None if best_score is None else float(best_score),
        "params": final_params,
    }
    return final_model, config


def model_feature_importance(model_name, feature_names, importances):
    rows = []
    for rank, idx in enumerate(np.argsort(importances)[::-1], start=1):
        rows.append(
            {
                "model": model_name,
                "rank": rank,
                "feature": feature_names[idx],
                "importance": float(importances[idx]),
                "feature_group": feature_group(feature_names[idx]),
            }
        )
    return rows


def feature_group(name):
    if name.startswith("hist_author_") or name == "hist_n_authors_enriched":
        return "author_history"
    if name.startswith("hist_ref_") or name == "hist_n_references_enriched":
        return "reference_history"
    if name.startswith("full_text_lsa_") or name.startswith("full_text_meta_"):
        return "full_text"
    if name.startswith("venue_identity_"):
        return "venue_identity"
    if name.startswith("venue_prestige_"):
        return "venue_prestige"
    if name.startswith("field_topic_"):
        return "field_topic"
    if name.startswith("affiliation_"):
        return "affiliation"
    if name.startswith("country_"):
        return "country"
    if name.startswith("lsa_"):
        return "text_lsa"
    if name in NUMERIC_META:
        return "numeric_metadata"
    if name.startswith("primary_category_"):
        return "primary_category"
    return "other"


def grouped_feature_importance(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grouped = (
        df.groupby(["model", "feature_group"], as_index=False)["importance"]
        .sum()
        .sort_values(["model", "importance"], ascending=[True, False])
    )
    return grouped


def read_baseline_metrics(path, models):
    df = pd.read_csv(require(path, "Run supervised/run_baselines.py first."))
    return df[df["model"].isin(models)].copy()


def save_regression_compare_figure(regression_df):
    apply_style()
    test_df = regression_df[regression_df["split"] == "test"].copy()
    metrics = ["rmse_log", "mae_log"]
    labels = ["RMSE", "MAE"]
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    colors = ["#2a78d6", "#d66b2a"]
    n_models = len(test_df)
    width = min(0.8 / max(1, n_models), 0.34)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * width
    for i, (_, row) in enumerate(test_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=MODEL_LABELS.get(row["model"], row["model"]),
            color=colors[i % len(colors)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("log-scale error")
    ax.set_title("Regression test comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "phase2_regression_test_compare.png", dpi=200)
    plt.close(fig)


def save_classification_compare_figure(classification_df):
    apply_style()
    test_df = classification_df[
        (classification_df["split"] == "test")
        & (classification_df["model"] != "prevalence_baseline")
    ].copy()
    metrics = ["pr_auc", "roc_auc", "f1"]
    labels = ["PR-AUC", "ROC-AUC", "F1"]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    colors = ["#1baf7a", "#4e8bd6", "#d66b2a", "#9a5cc7"]
    n_models = len(test_df)
    width = min(0.8 / max(1, n_models), 0.22)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * width
    for i, (_, row) in enumerate(test_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=MODEL_LABELS.get(row["model"], row["model"]),
            color=colors[i % len(colors)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Classification test comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "phase2_classification_test_compare.png", dpi=200)
    plt.close(fig)


def save_xgb_regression_figures(y_true, y_pred):
    apply_style()
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    idx = rng.choice(n, size=min(6000, n), replace=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(y_true[idx], y_pred[idx], s=8, alpha=0.22, color="#d66b2a", linewidths=0)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1.2, ls="--")
    ax.set_xlabel("true log(1 + citations)")
    ax.set_ylabel("predicted log(1 + citations)")
    ax.set_title("XGBoost regression: predicted vs true")
    fig.tight_layout()
    fig.savefig(FIGS / "xgboost_pred_vs_true.png", dpi=200)
    plt.close(fig)

    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(residuals, bins=60, color="#d66b2a", alpha=0.85)
    ax.axvline(0, color="#52514e", lw=1.2)
    ax.set_xlabel("prediction error on log scale")
    ax.set_ylabel("papers")
    ax.set_title("XGBoost regression residuals")
    fig.tight_layout()
    fig.savefig(FIGS / "xgboost_residuals.png", dpi=200)
    plt.close(fig)


def save_xgb_classification_figures(y_true, prob):
    apply_style()
    ap = average_precision_score(y_true, prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(recall, precision, color="#d66b2a", lw=2)
    ax.axhline(np.mean(y_true), color="#52514e", lw=1.0, ls="--", label="test prevalence")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"XGBoost classifier PR curve (AP = {ap:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "xgboost_classifier_pr_curve.png", dpi=200)
    plt.close(fig)

    auc = roc_auc_score(y_true, prob)
    fpr, tpr, _ = roc_curve(y_true, prob)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, color="#d66b2a", lw=2)
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1.0, ls="--", label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"XGBoost classifier ROC curve (AUC = {auc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "xgboost_classifier_roc_curve.png", dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--prob-threshold", type=float, default=0.5)

    parser.add_argument("--xgb-cls-n-estimators", type=int, default=1000)
    parser.add_argument("--xgb-cls-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-cls-max-depth", type=int, default=3)
    parser.add_argument("--xgb-cls-min-child-weight", type=float, default=5.0)
    parser.add_argument("--xgb-cls-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-cls-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--xgb-cls-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-cls-reg-lambda", type=float, default=8.0)
    parser.add_argument("--xgb-cls-early-stopping-rounds", type=int, default=50)

    parser.add_argument("--xgb-n-estimators", type=int, default=1000)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-min-child-weight", type=float, default=3.0)
    parser.add_argument("--xgb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=5.0)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--xgb-tree-method", default="hist")
    return parser.parse_args()


def main():
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("Loading supervised feature artifacts ...", flush=True)
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
    train_meta = pd.read_csv(require(ART / "train_meta.csv.gz", "Run supervised/run_features.py."))
    with open(require(ART / "feature_names.json", "Run supervised/run_features.py."), encoding="utf-8") as fh:
        feature_names = json.load(fh)
    print(f"  train/test: {x_train.shape} / {x_test.shape}", flush=True)

    threshold = float(np.quantile(cites_train, 0.9))
    high_train = (cites_train >= threshold).astype(int)
    high_test = (cites_test >= threshold).astype(int)
    fit_idx, val_idx, val_year = validation_indices(train_meta)

    print(f"Tuning classification thresholds on validation year {val_year} ...", flush=True)
    baseline_config = load_baseline_config()
    logistic_tune = LogisticRegression(
        C=baseline_config.get("logistic_c", 1.0),
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    logistic_tune.fit(x_train[fit_idx], high_train[fit_idx])
    logistic_val_prob = logistic_tune.predict_proba(x_train[val_idx])[:, 1]
    logistic_tuned_threshold, logistic_val_f1 = best_f1_threshold(
        high_train[val_idx], logistic_val_prob
    )

    logistic = joblib.load(
        require(ART / "logistic_model.joblib", "Run supervised/run_baselines.py first.")
    )
    logistic_prob_train = logistic.predict_proba(x_train)[:, 1]
    logistic_prob_test = logistic.predict_proba(x_test)[:, 1]
    logistic_tuned_rows = [
        add_threshold_info(
            classification_row(
                "logistic_regression_val_f1_threshold",
                "train",
                high_train,
                logistic_prob_train,
                (logistic_prob_train >= logistic_tuned_threshold).astype(int),
                threshold,
            ),
            logistic_tuned_threshold,
            f"validation_year_{val_year}_max_f1",
        ),
        add_threshold_info(
            classification_row(
                "logistic_regression_val_f1_threshold",
                "test",
                high_test,
                logistic_prob_test,
                (logistic_prob_test >= logistic_tuned_threshold).astype(int),
                threshold,
            ),
            logistic_tuned_threshold,
            f"validation_year_{val_year}_max_f1",
        ),
    ]

    xgb_cls, xgb_cls_tuned_threshold, xgb_cls_config = fit_xgboost_classifier(
        x_train, high_train, train_meta, args
    )
    xgb_cls_prob_train = xgb_cls.predict_proba(x_train)[:, 1]
    xgb_cls_prob_test = xgb_cls.predict_proba(x_test)[:, 1]
    xgb_cls_pred_train = (xgb_cls_prob_train >= args.prob_threshold).astype(int)
    xgb_cls_pred_test = (xgb_cls_prob_test >= args.prob_threshold).astype(int)
    cls_rows = [
        add_threshold_info(
            classification_row(
                "xgboost_classifier",
                "train",
                high_train,
                xgb_cls_prob_train,
                xgb_cls_pred_train,
                threshold,
            ),
            args.prob_threshold,
            "default_probability",
        ),
        add_threshold_info(
            classification_row(
                "xgboost_classifier",
                "test",
                high_test,
                xgb_cls_prob_test,
                xgb_cls_pred_test,
                threshold,
            ),
            args.prob_threshold,
            "default_probability",
        ),
        add_threshold_info(
            classification_row(
                "xgboost_classifier_val_f1_threshold",
                "train",
                high_train,
                xgb_cls_prob_train,
                (xgb_cls_prob_train >= xgb_cls_tuned_threshold).astype(int),
                threshold,
            ),
            xgb_cls_tuned_threshold,
            f"validation_year_{val_year}_max_f1",
        ),
        add_threshold_info(
            classification_row(
                "xgboost_classifier_val_f1_threshold",
                "test",
                high_test,
                xgb_cls_prob_test,
                (xgb_cls_prob_test >= xgb_cls_tuned_threshold).astype(int),
                threshold,
            ),
            xgb_cls_tuned_threshold,
            f"validation_year_{val_year}_max_f1",
        ),
    ]
    np.save(ART / "xgboost_classifier_prob_train.npy", xgb_cls_prob_train.astype(np.float32))
    np.save(ART / "xgboost_classifier_prob_test.npy", xgb_cls_prob_test.astype(np.float32))
    joblib.dump(xgb_cls, ART / "xgboost_classifier.joblib")
    save_xgb_classification_figures(high_test, xgb_cls_prob_test)

    xgb, xgb_config = fit_xgboost_regressor(x_train, y_train, train_meta, args)
    pred_train = xgb.predict(x_train)
    pred_test = xgb.predict(x_test)
    reg_rows = [
        regression_row("xgboost_regressor", "train", y_train, pred_train),
        regression_row("xgboost_regressor", "test", y_test, pred_test),
    ]
    rank_rows = [
        ranking_row("xgboost_regressor", "train", y_train, cites_train, pred_train, threshold),
        ranking_row("xgboost_regressor", "test", y_test, cites_test, pred_test, threshold),
    ]
    np.save(ART / "xgboost_pred_train.npy", pred_train.astype(np.float32))
    np.save(ART / "xgboost_pred_test.npy", pred_test.astype(np.float32))
    joblib.dump(xgb, ART / "xgboost_regressor.joblib")
    save_xgb_regression_figures(y_test, pred_test)

    baseline_cls = read_baseline_metrics(
        TABLES / "classification_metrics.csv",
        ["logistic_regression", "prevalence_baseline"],
    )
    baseline_cls["decision_threshold"] = args.prob_threshold
    baseline_cls["threshold_source"] = "default_probability"
    classification_df = pd.concat(
        [baseline_cls, pd.DataFrame(logistic_tuned_rows + cls_rows)],
        ignore_index=True,
    )
    classification_df.to_csv(TABLES / "phase2_classification_comparison.csv", index=False)

    baseline_reg = read_baseline_metrics(TABLES / "regression_metrics.csv", ["ridge"])
    regression_df = pd.concat([baseline_reg, pd.DataFrame(reg_rows)], ignore_index=True)
    regression_df.to_csv(TABLES / "phase2_regression_comparison.csv", index=False)

    baseline_rank = read_baseline_metrics(TABLES / "ranking_metrics.csv", ["ridge"])
    ranking_df = pd.concat([baseline_rank, pd.DataFrame(rank_rows)], ignore_index=True)
    ranking_df.to_csv(TABLES / "phase2_ranking_comparison.csv", index=False)

    importance_rows = []
    importance_rows.extend(
        model_feature_importance("xgboost_classifier", feature_names, xgb_cls.feature_importances_)
    )
    importance_rows.extend(
        model_feature_importance("xgboost_regressor", feature_names, xgb.feature_importances_)
    )
    pd.DataFrame(importance_rows).to_csv(TABLES / "phase2_feature_importance.csv", index=False)
    grouped_feature_importance(importance_rows).to_csv(
        TABLES / "phase2_feature_group_importance.csv", index=False
    )

    save_regression_compare_figure(regression_df)
    save_classification_compare_figure(classification_df)

    with open(ART / "phase2_tree_model_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "highly_cited_threshold": threshold,
                "prob_threshold": args.prob_threshold,
                "xgboost_classifier": xgb_cls_config,
                "logistic_regression_threshold_tuning": {
                    "validation": "latest train year",
                    "validation_year": val_year,
                    "validation_f1_threshold": logistic_tuned_threshold,
                    "validation_f1": logistic_val_f1,
                },
                "xgboost_regressor": xgb_config,
            },
            fh,
            indent=2,
        )

    print(f"Done in {time.time() - t0:.0f}s -> {TABLES} and {FIGS}", flush=True)


if __name__ == "__main__":
    main()
