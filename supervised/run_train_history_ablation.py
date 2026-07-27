"""Build train-only author/reference history features and evaluate XGBoost.

Inputs:
  * supervised/artifacts/X_train.npz and X_test.npz
  * data/c1.jsonl.gz for split/year/targets/openalex_id
  * OpenAlex enrichment from run_openalex_enrichment.py

The generated features are leakage-safe:
  * train rows use only earlier train years
  * test rows use only train rows earlier than the test paper year
  * no current OpenAlex citation counts or author metrics are used
"""

import argparse
import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TABLES = HERE / "outputs" / "tables"
MPLCONFIGDIR = ART / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from run_baselines import classification_row, ranking_row, regression_row
from run_phase2_tree_models import (
    add_threshold_info,
    fit_xgboost_classifier,
    fit_xgboost_regressor,
    grouped_feature_importance,
    model_feature_importance,
)


FEATURE_NAMES = [
    "hist_n_authors_enriched",
    "hist_author_seen_count",
    "hist_author_seen_fraction",
    "hist_author_prior_papers_sum",
    "hist_author_prior_papers_mean",
    "hist_author_prior_papers_max",
    "hist_author_prior_log_citations_mean",
    "hist_author_prior_log_citations_max",
    "hist_author_prior_high_rate_mean",
    "hist_author_prior_high_rate_max",
    "hist_author_prior_high_papers_sum",
    "hist_author_prior_coauthors_mean",
    "hist_author_prior_coauthors_max",
    "hist_author_has_prior_high_paper",
    "hist_n_references_enriched",
    "hist_ref_seen_train_count",
    "hist_ref_seen_train_fraction",
    "hist_ref_train_log_citations_mean",
    "hist_ref_train_log_citations_max",
    "hist_ref_train_high_count",
    "hist_ref_train_high_rate",
    "hist_ref_train_mean_age",
    "hist_ref_train_min_age",
    "hist_ref_train_authors_count",
    "hist_ref_author_overlap_count",
    "hist_ref_author_overlap_fraction",
]


def require(path, hint):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")
    return path


def normalize_openalex_id(value):
    if not value:
        return None
    value = str(value).strip()
    if not value:
        return None
    if value.startswith("https://openalex.org/"):
        return value
    if value.startswith("W"):
        return f"https://openalex.org/{value}"
    return value


def read_base_records(data_path):
    rows = []
    with gzip.open(data_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            rows.append(
                {
                    "arxiv_id": record.get("arxiv_id"),
                    "openalex_id": normalize_openalex_id(record.get("openalex_id")),
                    "split": record.get("split"),
                    "year": int(record.get("year")),
                    "target_citations": float(record.get("target_citations") or 0.0),
                    "y_log": float(record.get("y_log") or 0.0),
                }
            )
    return pd.DataFrame(rows)


def read_enrichment(path):
    enrichment = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            work_id = normalize_openalex_id(row.get("openalex_id"))
            if not work_id:
                continue
            enrichment[work_id] = {
                "author_ids": list(dict.fromkeys(row.get("author_ids") or [])),
                "institution_ids": list(dict.fromkeys(row.get("institution_ids") or [])),
                "referenced_work_ids": [
                    normalize_openalex_id(ref)
                    for ref in (row.get("referenced_work_ids") or [])
                    if normalize_openalex_id(ref)
                ],
                "source_id": row.get("source_id"),
            }
    return enrichment


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def safe_max(values):
    return float(np.max(values)) if values else 0.0


def safe_min(values):
    return float(np.min(values)) if values else 0.0


def empty_author_stats():
    return {
        "papers": 0,
        "sum_y": 0.0,
        "max_y": 0.0,
        "high": 0,
        "coauthors": set(),
    }


def author_features(author_ids, author_stats):
    n_authors = len(author_ids)
    stats = [author_stats[a] for a in author_ids if a in author_stats]
    seen = len(stats)
    papers = [s["papers"] for s in stats]
    mean_y = [s["sum_y"] / s["papers"] for s in stats if s["papers"] > 0]
    max_y = [s["max_y"] for s in stats]
    high_rates = [s["high"] / s["papers"] for s in stats if s["papers"] > 0]
    high_counts = [s["high"] for s in stats]
    coauthors = [len(s["coauthors"]) for s in stats]
    return [
        n_authors,
        seen,
        seen / n_authors if n_authors else 0.0,
        float(sum(papers)),
        safe_mean(papers),
        safe_max(papers),
        safe_mean(mean_y),
        safe_max(max_y),
        safe_mean(high_rates),
        safe_max(high_rates),
        float(sum(high_counts)),
        safe_mean(coauthors),
        safe_max(coauthors),
        1.0 if any(count > 0 for count in high_counts) else 0.0,
    ]


def reference_features(ref_ids, author_ids, work_stats, paper_year):
    n_refs = len(ref_ids)
    seen_refs = [work_stats[ref] for ref in ref_ids if ref in work_stats]
    seen_count = len(seen_refs)
    y_vals = [ref["y_log"] for ref in seen_refs]
    high_vals = [ref["high"] for ref in seen_refs]
    ages = [max(0, paper_year - ref["year"]) for ref in seen_refs]
    ref_authors = set()
    overlap_refs = 0
    current_authors = set(author_ids)
    for ref in seen_refs:
        authors = set(ref["author_ids"])
        ref_authors.update(authors)
        if current_authors and authors.intersection(current_authors):
            overlap_refs += 1

    return [
        n_refs,
        seen_count,
        seen_count / n_refs if n_refs else 0.0,
        safe_mean(y_vals),
        safe_max(y_vals),
        float(sum(high_vals)),
        float(sum(high_vals) / seen_count) if seen_count else 0.0,
        safe_mean(ages),
        safe_min(ages),
        float(len(ref_authors)),
        float(overlap_refs),
        overlap_refs / seen_count if seen_count else 0.0,
    ]


def add_train_record(row, author_ids, author_stats, work_stats, threshold):
    y_log = float(row["y_log"])
    high = int(float(row["target_citations"]) >= threshold)
    authors = list(dict.fromkeys(author_ids))
    for author in authors:
        stats = author_stats[author]
        stats["papers"] += 1
        stats["sum_y"] += y_log
        stats["max_y"] = max(stats["max_y"], y_log)
        stats["high"] += high
        stats["coauthors"].update(a for a in authors if a != author)
    work_id = row["openalex_id"]
    if work_id:
        work_stats[work_id] = {
            "year": int(row["year"]),
            "y_log": y_log,
            "high": high,
            "author_ids": authors,
        }


def build_history_features(df, enrichment, threshold):
    author_stats = defaultdict(empty_author_stats)
    work_stats = {}
    features = {}
    coverage_rows = []

    all_years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique())
    for year in all_years:
        rows_for_year = df[df["year"] == year]
        for idx, row in rows_for_year.iterrows():
            enriched = enrichment.get(row["openalex_id"], {})
            author_ids = enriched.get("author_ids") or []
            ref_ids = enriched.get("referenced_work_ids") or []
            values = author_features(author_ids, author_stats)
            values.extend(reference_features(ref_ids, author_ids, work_stats, int(year)))
            features[idx] = values
            coverage_rows.append(
                {
                    "split": row["split"],
                    "year": year,
                    "has_enrichment": bool(enriched),
                    "n_authors": len(author_ids),
                    "n_references": len(ref_ids),
                    "n_refs_seen_train": values[15],
                }
            )

        train_rows = rows_for_year[rows_for_year["split"] == "train"]
        for _, row in train_rows.iterrows():
            enriched = enrichment.get(row["openalex_id"], {})
            add_train_record(
                row,
                enriched.get("author_ids") or [],
                author_stats,
                work_stats,
                threshold,
            )

    matrix = np.asarray([features[idx] for idx in df.index], dtype=np.float32)
    return matrix, pd.DataFrame(coverage_rows)


def evaluate(name, x_train, x_test, feature_names, y_train, y_test, cites_train, cites_test, threshold, train_meta, args):
    high_train = (cites_train >= threshold).astype(int)
    high_test = (cites_test >= threshold).astype(int)

    cls_model, tuned_threshold, cls_config = fit_xgboost_classifier(
        x_train, high_train, train_meta, args
    )
    prob_train = cls_model.predict_proba(x_train)[:, 1]
    prob_test = cls_model.predict_proba(x_test)[:, 1]
    cls_rows = []
    for split, y_true, prob in [
        ("train", high_train, prob_train),
        ("test", high_test, prob_test),
    ]:
        for decision_threshold, source in [
            (args.prob_threshold, "default_probability"),
            (tuned_threshold, "validation_year_2020_max_f1"),
        ]:
            row = classification_row(
                "xgboost_classifier",
                split,
                y_true,
                prob,
                (prob >= decision_threshold).astype(int),
                threshold,
            )
            add_threshold_info(row, decision_threshold, source)
            row["feature_set"] = name
            cls_rows.append(row)

    reg_model, reg_config = fit_xgboost_regressor(x_train, y_train, train_meta, args)
    pred_train = reg_model.predict(x_train)
    pred_test = reg_model.predict(x_test)
    reg_rows = []
    rank_rows = []
    for split, y_true, cites, pred in [
        ("train", y_train, cites_train, pred_train),
        ("test", y_test, cites_test, pred_test),
    ]:
        reg_row = regression_row("xgboost_regressor", split, y_true, pred)
        reg_row["feature_set"] = name
        reg_rows.append(reg_row)

        rank_row = ranking_row("xgboost_regressor", split, y_true, cites, pred, threshold)
        rank_row["feature_set"] = name
        rank_rows.append(rank_row)

    importance_rows = []
    importance_rows.extend(
        model_feature_importance(
            f"xgboost_classifier__{name}",
            feature_names,
            cls_model.feature_importances_,
        )
    )
    importance_rows.extend(
        model_feature_importance(
            f"xgboost_regressor__{name}",
            feature_names,
            reg_model.feature_importances_,
        )
    )
    return cls_rows, reg_rows, rank_rows, importance_rows, {
        "feature_set": name,
        "n_features": int(x_train.shape[1]),
        "xgboost_classifier": cls_config,
        "xgboost_regressor": reg_config,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="c1")
    parser.add_argument("--data", default=None)
    parser.add_argument("--enrichment", default=None)
    parser.add_argument("--output-prefix", default="train_history_ablation")
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
    ART.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data) if args.data else ROOT / "data" / f"{args.cohort}.jsonl.gz"
    enrichment_path = (
        Path(args.enrichment)
        if args.enrichment
        else ART / f"openalex_{args.cohort}_author_ref_enrichment.jsonl.gz"
    )

    t0 = time.time()
    base_train = sparse.load_npz(require(ART / "X_train.npz", "Run supervised/run_features.py."))
    base_test = sparse.load_npz(require(ART / "X_test.npz", "Run supervised/run_features.py."))
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
        base_names = json.load(fh)

    df = read_base_records(data_path)
    if len(df[df["split"] == "train"]) != base_train.shape[0]:
        raise ValueError("Train row count does not match supervised feature artifacts.")
    if len(df[df["split"] == "test"]) != base_test.shape[0]:
        raise ValueError("Test row count does not match supervised feature artifacts.")

    enrichment = read_enrichment(
        require(enrichment_path, "Run supervised/run_openalex_enrichment.py first.")
    )
    threshold = float(np.quantile(cites_train, 0.9))
    print(
        f"Building train-history features from {len(enrichment):,} enriched works "
        f"(threshold={threshold:g}) ...",
        flush=True,
    )
    history_matrix, coverage = build_history_features(df, enrichment, threshold)
    coverage.to_csv(TABLES / f"{args.output_prefix}_coverage.csv", index=False)

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    history_train_raw = history_matrix[train_mask.to_numpy()]
    history_test_raw = history_matrix[test_mask.to_numpy()]
    scaler = StandardScaler()
    history_train = scaler.fit_transform(history_train_raw).astype(np.float32)
    history_test = scaler.transform(history_test_raw).astype(np.float32)

    sparse.save_npz(ART / "train_history_features_train.npz", sparse.csr_matrix(history_train))
    sparse.save_npz(ART / "train_history_features_test.npz", sparse.csr_matrix(history_test))
    with open(ART / "train_history_feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(FEATURE_NAMES, fh, indent=2)

    feature_set = "base_plus_train_author_reference_history"
    x_train = sparse.hstack([base_train, sparse.csr_matrix(history_train)], format="csr")
    x_test = sparse.hstack([base_test, sparse.csr_matrix(history_test)], format="csr")
    feature_names = base_names + FEATURE_NAMES
    print(f"Evaluating {feature_set}: {x_train.shape} / {x_test.shape}", flush=True)
    cls, reg, rank, importance, config = evaluate(
        feature_set,
        x_train,
        x_test,
        feature_names,
        y_train,
        y_test,
        cites_train,
        cites_test,
        threshold,
        train_meta,
        args,
    )

    prefix = args.output_prefix
    pd.DataFrame(cls).to_csv(TABLES / f"{prefix}_classification.csv", index=False)
    pd.DataFrame(reg).to_csv(TABLES / f"{prefix}_regression.csv", index=False)
    pd.DataFrame(rank).to_csv(TABLES / f"{prefix}_ranking.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "n_features": int(x_train.shape[1]),
                "n_added_features": len(FEATURE_NAMES),
                "enriched_work_count": len(enrichment),
                "train_enrichment_rate": float(coverage[coverage["split"] == "train"]["has_enrichment"].mean()),
                "test_enrichment_rate": float(coverage[coverage["split"] == "test"]["has_enrichment"].mean()),
            }
        ]
    ).to_csv(TABLES / f"{prefix}_dimensions.csv", index=False)
    pd.DataFrame(importance).to_csv(TABLES / f"{prefix}_importance.csv", index=False)
    grouped_feature_importance(importance).to_csv(
        TABLES / f"{prefix}_group_importance.csv", index=False
    )
    with open(ART / f"{prefix}_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "data": str(data_path),
                "enrichment": str(enrichment_path),
                "highly_cited_threshold": threshold,
                "feature_names": FEATURE_NAMES,
                "feature_set": config,
                "leakage_safety": "Rows use only earlier c1 train rows for author/reference aggregates.",
            },
            fh,
            indent=2,
        )

    print(f"Done in {time.time() - t0:.0f}s -> {prefix}_*.csv in {TABLES}", flush=True)


if __name__ == "__main__":
    main()
