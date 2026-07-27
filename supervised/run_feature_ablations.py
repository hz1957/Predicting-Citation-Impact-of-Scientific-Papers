"""Run feature ablations for phase-2 XGBoost models.

The base feature matrix is the leakage-safe supervised representation from
run_features.py. This script adds available non-leakage predictors one group at
a time and evaluates:

  * XGBoost Classifier for the train-defined high-citation label
  * XGBoost Regressor for log(1 + citations)

Unavailable requested features are intentionally not fabricated:
  * author identity/prestige: the dataset has author_count, but no author IDs
  * reference graph: the dataset has reference_count, but no cited-paper IDs
"""

import argparse
import gzip
import json
import os
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TABLES = HERE / "outputs" / "tables"
MPLCONFIGDIR = ART / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

from run_baselines import classification_row, ranking_row, regression_row
from run_phase2_tree_models import (
    add_threshold_info,
    best_f1_threshold,
    fit_xgboost_classifier,
    fit_xgboost_regressor,
    grouped_feature_importance,
    model_feature_importance,
)

SEED = 42

EXTRA_KEEP = [
    "split",
    "year",
    "venue",
    "field",
    "topic",
    "affiliations",
    "countries",
    "ft_n_chars",
    "ft_n_words",
    "has_fulltext",
    "full_text",
    "target_citations",
    "y_log",
]

FEATURE_SET_SPECS = [
    ("base", []),
    ("base_plus_venue_identity", ["venue_identity"]),
    ("base_plus_venue_prestige", ["venue_prestige"]),
    ("base_plus_field_topic", ["field_topic"]),
    ("base_plus_affiliation_country", ["affiliation_country"]),
    ("base_plus_full_text_lsa", ["full_text_lsa"]),
    ("base_plus_full_text_venue_identity", ["full_text_lsa", "venue_identity"]),
    (
        "base_plus_full_text_affiliation_country",
        ["full_text_lsa", "affiliation_country"],
    ),
    (
        "all_available_extra_features",
        [
            "venue_identity",
            "venue_prestige",
            "field_topic",
            "affiliation_country",
            "full_text_lsa",
        ],
    ),
]


def require(path, hint):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")
    return path


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def read_records(path, full_text_max_chars):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            row = {k: record.get(k) for k in EXTRA_KEEP}
            text = row.get("full_text") or ""
            row["full_text"] = text[:full_text_max_chars]
            rows.append(row)
    return pd.DataFrame(rows)


def load_raw_split(data_path, full_text_max_chars):
    print(f"Reading ablation source fields from {data_path} ...", flush=True)
    df = read_records(data_path, full_text_max_chars)
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Expected non-empty c1 train/test split labels.")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def clean_category_values(values, train_counts, min_count):
    values = values.fillna("__MISSING__").astype(str)
    keep = set(train_counts[train_counts >= min_count].index)
    return values.where(values.isin(keep), "__RARE__")


def categorical_block(train, test, columns, prefix, min_count):
    clean_train = pd.DataFrame(index=train.index)
    clean_test = pd.DataFrame(index=test.index)
    for col in columns:
        train_values = train[col].fillna("__MISSING__").astype(str)
        counts = train_values.value_counts()
        clean_train[col] = clean_category_values(train[col], counts, min_count)
        clean_test[col] = clean_category_values(test[col], counts, min_count)

    encoder = one_hot_encoder()
    x_train = encoder.fit_transform(clean_train)
    x_test = encoder.transform(clean_test)
    names = [f"{prefix}_{name}" for name in encoder.get_feature_names_out(columns)]
    return x_train.astype(np.float32), x_test.astype(np.float32), names


def normalize_list(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def multilabel_block(train_values, test_values, prefix, min_count, max_values):
    train_lists = [normalize_list(v) for v in train_values]
    test_lists = [normalize_list(v) for v in test_values]
    counts = Counter(v for values in train_lists for v in values)
    kept = [
        value
        for value, count in counts.most_common(max_values)
        if count >= min_count
    ]
    index = {value: i for i, value in enumerate(kept)}

    def transform(lists):
        rows, cols = [], []
        for row_idx, values in enumerate(lists):
            seen = set()
            for value in values:
                col_idx = index.get(value)
                if col_idx is not None and col_idx not in seen:
                    rows.append(row_idx)
                    cols.append(col_idx)
                    seen.add(col_idx)
        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(lists), len(kept)))

    names = [f"{prefix}_{value}" for value in kept]
    return transform(train_lists), transform(test_lists), names


def numeric_block(train, test, columns, prefix, log_columns=None):
    log_columns = set(log_columns or [])
    train_num = pd.DataFrame(index=train.index)
    test_num = pd.DataFrame(index=test.index)
    for col in columns:
        tr = pd.to_numeric(train[col], errors="coerce").fillna(0.0)
        te = pd.to_numeric(test[col], errors="coerce").fillna(0.0)
        if col in log_columns:
            tr = np.log1p(tr.clip(lower=0))
            te = np.log1p(te.clip(lower=0))
        train_num[col] = tr
        test_num[col] = te

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_num).astype(np.float32)
    x_test = scaler.transform(test_num).astype(np.float32)
    names = [f"{prefix}_{col}" for col in columns]
    return sparse.csr_matrix(x_train), sparse.csr_matrix(x_test), names


def make_full_text_block(train, test, args):
    print(
        "Building full_text TF-IDF -> LSA block "
        f"(max_chars={args.full_text_max_chars:,}, dims={args.full_text_svd_dims}) ...",
        flush=True,
    )
    vectorizer = TfidfVectorizer(
        max_features=args.full_text_max_features,
        min_df=args.full_text_min_df,
        max_df=args.full_text_max_df,
        stop_words="english",
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    train_tfidf = vectorizer.fit_transform(train["full_text"].fillna(""))
    test_tfidf = vectorizer.transform(test["full_text"].fillna(""))
    max_svd = min(train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1)
    n_svd = min(args.full_text_svd_dims, max_svd)
    svd = TruncatedSVD(
        n_components=n_svd,
        n_iter=args.full_text_svd_iter,
        random_state=SEED,
    )
    normalizer = Normalizer(copy=False)
    train_lsa = normalizer.fit_transform(svd.fit_transform(train_tfidf))
    test_lsa = normalizer.transform(svd.transform(test_tfidf))
    text_train = sparse.csr_matrix(train_lsa.astype(np.float32))
    text_test = sparse.csr_matrix(test_lsa.astype(np.float32))
    text_names = [f"full_text_lsa_{i}" for i in range(n_svd)]

    len_train, len_test, len_names = numeric_block(
        train,
        test,
        ["ft_n_chars", "ft_n_words", "has_fulltext"],
        "full_text_meta",
        log_columns=["ft_n_chars", "ft_n_words"],
    )
    return (
        sparse.hstack([text_train, len_train], format="csr"),
        sparse.hstack([text_test, len_test], format="csr"),
        text_names + len_names,
    )


def smoothed_stats(source, key_col, y_col, positive_col, smooth):
    global_y = float(source[y_col].mean())
    global_pos = float(source[positive_col].mean())
    grouped = source.groupby(key_col).agg(
        count=(y_col, "size"),
        sum_y=(y_col, "sum"),
        sum_pos=(positive_col, "sum"),
    )
    grouped["mean_log"] = (
        grouped["sum_y"] + smooth * global_y
    ) / (grouped["count"] + smooth)
    grouped["positive_rate"] = (
        grouped["sum_pos"] + smooth * global_pos
    ) / (grouped["count"] + smooth)
    return grouped, global_y, global_pos


def apply_prestige_stats(df, stats, global_y, global_pos):
    keys = df["venue"].fillna("__MISSING__").astype(str)
    mean_log = keys.map(stats["mean_log"]).fillna(global_y)
    positive_rate = keys.map(stats["positive_rate"]).fillna(global_pos)
    count = keys.map(stats["count"]).fillna(0.0)
    return pd.DataFrame(
        {
            "venue_prestige_mean_log": mean_log.to_numpy(dtype=float),
            "venue_prestige_positive_rate": positive_rate.to_numpy(dtype=float),
            "venue_prestige_log_count": np.log1p(count.to_numpy(dtype=float)),
        }
    )


def make_venue_prestige_block(train, test, y_train, cites_train, threshold, args):
    print("Building train-only temporal venue prestige block ...", flush=True)
    source = train[["year", "venue"]].copy()
    source["venue"] = source["venue"].fillna("__MISSING__").astype(str)
    source["y_log"] = y_train
    source["highly_cited"] = (cites_train >= threshold).astype(int)

    train_parts = []
    for year in pd.to_numeric(source["year"], errors="coerce").sort_values().unique():
        mask = source["year"] == year
        prior = source[source["year"] < year]
        if prior.empty:
            prior = source
        stats, global_y, global_pos = smoothed_stats(
            prior, "venue", "y_log", "highly_cited", args.prestige_smoothing
        )
        part = apply_prestige_stats(source.loc[mask], stats, global_y, global_pos)
        part.index = source.index[mask]
        train_parts.append(part)
    train_features = pd.concat(train_parts).sort_index()

    stats, global_y, global_pos = smoothed_stats(
        source, "venue", "y_log", "highly_cited", args.prestige_smoothing
    )
    test_features = apply_prestige_stats(test, stats, global_y, global_pos)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features).astype(np.float32)
    x_test = scaler.transform(test_features).astype(np.float32)
    return sparse.csr_matrix(x_train), sparse.csr_matrix(x_test), train_features.columns.tolist()


def build_extra_blocks(train, test, y_train, cites_train, threshold, args, required_blocks):
    blocks = {}
    if "venue_identity" in required_blocks:
        blocks["venue_identity"] = categorical_block(
            train, test, ["venue"], "venue_identity", args.venue_min_count
        )
    if "venue_prestige" in required_blocks:
        blocks["venue_prestige"] = make_venue_prestige_block(
            train, test, y_train, cites_train, threshold, args
        )
    if "field_topic" in required_blocks:
        blocks["field_topic"] = categorical_block(
            train, test, ["field", "topic"], "field_topic", args.topic_min_count
        )

    if "affiliation_country" in required_blocks:
        aff_train, aff_test, aff_names = multilabel_block(
            train["affiliations"],
            test["affiliations"],
            "affiliation",
            args.affiliation_min_count,
            args.max_affiliations,
        )
        country_train, country_test, country_names = multilabel_block(
            train["countries"],
            test["countries"],
            "country",
            1,
            args.max_countries,
        )
        blocks["affiliation_country"] = (
            sparse.hstack([aff_train, country_train], format="csr"),
            sparse.hstack([aff_test, country_test], format="csr"),
            aff_names + country_names,
        )
    if "full_text_lsa" in required_blocks:
        blocks["full_text_lsa"] = make_full_text_block(train, test, args)
    return blocks


def select_feature_set_specs(requested):
    specs_by_name = dict(FEATURE_SET_SPECS)
    if not requested:
        return FEATURE_SET_SPECS
    unknown = [name for name in requested if name not in specs_by_name]
    if unknown:
        valid = ", ".join(specs_by_name)
        raise ValueError(f"Unknown feature set(s): {unknown}. Valid choices: {valid}")
    return [(name, specs_by_name[name]) for name in requested]


def required_blocks_for(specs):
    return {block for _, block_names in specs for block in block_names}


def assemble_feature_sets(base_train, base_test, base_names, blocks, specs):
    for feature_set, block_names in specs:
        train_parts = [base_train]
        test_parts = [base_test]
        names = list(base_names)
        for block_name in block_names:
            x_train, x_test, block_features = blocks[block_name]
            train_parts.append(x_train)
            test_parts.append(x_test)
            names.extend(block_features)
        yield (
            feature_set,
            sparse.hstack(train_parts, format="csr"),
            sparse.hstack(test_parts, format="csr"),
            names,
            block_names,
        )


def evaluate_feature_set(
    feature_set,
    x_train,
    x_test,
    feature_names,
    block_names,
    y_train,
    y_test,
    cites_train,
    cites_test,
    high_train,
    high_test,
    threshold,
    train_meta,
    args,
):
    print(
        f"Evaluating {feature_set}: train/test={x_train.shape}/{x_test.shape} ...",
        flush=True,
    )
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
            row["feature_set"] = feature_set
            row["blocks_added"] = "+".join(block_names) if block_names else "none"
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
        reg_row["feature_set"] = feature_set
        reg_row["blocks_added"] = "+".join(block_names) if block_names else "none"
        reg_rows.append(reg_row)

        rank_row = ranking_row("xgboost_regressor", split, y_true, cites, pred, threshold)
        rank_row["feature_set"] = feature_set
        rank_row["blocks_added"] = "+".join(block_names) if block_names else "none"
        rank_rows.append(rank_row)

    importance_rows = []
    importance_rows.extend(
        model_feature_importance(
            f"xgboost_classifier__{feature_set}",
            feature_names,
            cls_model.feature_importances_,
        )
    )
    importance_rows.extend(
        model_feature_importance(
            f"xgboost_regressor__{feature_set}",
            feature_names,
            reg_model.feature_importances_,
        )
    )

    config = {
        "feature_set": feature_set,
        "blocks_added": block_names,
        "n_features": int(x_train.shape[1]),
        "xgboost_classifier": cls_config,
        "xgboost_regressor": reg_config,
    }
    return cls_rows, reg_rows, rank_rows, importance_rows, config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="c1")
    parser.add_argument("--data", default=None)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=None,
        help="Optional feature-set names to run. By default all ablation specs run.",
    )
    parser.add_argument(
        "--output-prefix",
        default="feature_ablation",
        help="Prefix for output CSV/config files.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--prob-threshold", type=float, default=0.5)

    parser.add_argument("--venue-min-count", type=int, default=5)
    parser.add_argument("--topic-min-count", type=int, default=10)
    parser.add_argument("--affiliation-min-count", type=int, default=20)
    parser.add_argument("--max-affiliations", type=int, default=2000)
    parser.add_argument("--max-countries", type=int, default=250)
    parser.add_argument("--prestige-smoothing", type=float, default=20.0)

    parser.add_argument("--full-text-max-chars", type=int, default=8_000)
    parser.add_argument("--full-text-max-features", type=int, default=30_000)
    parser.add_argument("--full-text-min-df", type=int, default=5)
    parser.add_argument("--full-text-max-df", type=float, default=0.95)
    parser.add_argument("--full-text-svd-dims", type=int, default=50)
    parser.add_argument("--full-text-svd-iter", type=int, default=3)

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

    train, test = load_raw_split(data_path, args.full_text_max_chars)
    if len(train) != base_train.shape[0] or len(test) != base_test.shape[0]:
        raise ValueError("Raw split row counts do not match supervised feature artifacts.")

    threshold = float(np.quantile(cites_train, 0.9))
    high_train = (cites_train >= threshold).astype(int)
    high_test = (cites_test >= threshold).astype(int)

    specs = select_feature_set_specs(args.feature_sets)
    required_blocks = required_blocks_for(specs)
    print(
        "Selected feature sets: " + ", ".join(name for name, _ in specs),
        flush=True,
    )
    print(
        "Required extra blocks: "
        + (", ".join(sorted(required_blocks)) if required_blocks else "none"),
        flush=True,
    )
    blocks = build_extra_blocks(
        train, test, y_train, cites_train, threshold, args, required_blocks
    )
    all_cls, all_reg, all_rank, all_importance, all_config, dim_rows = [], [], [], [], [], []
    for feature_set, x_train, x_test, feature_names, block_names in assemble_feature_sets(
        base_train, base_test, base_names, blocks, specs
    ):
        cls, reg, rank, importance, config = evaluate_feature_set(
            feature_set,
            x_train,
            x_test,
            feature_names,
            block_names,
            y_train,
            y_test,
            cites_train,
            cites_test,
            high_train,
            high_test,
            threshold,
            train_meta,
            args,
        )
        all_cls.extend(cls)
        all_reg.extend(reg)
        all_rank.extend(rank)
        all_importance.extend(importance)
        all_config.append(config)
        dim_rows.append(
            {
                "feature_set": feature_set,
                "blocks_added": "+".join(block_names) if block_names else "none",
                "n_features": int(x_train.shape[1]),
                "n_added_features": int(x_train.shape[1] - base_train.shape[1]),
            }
        )

    prefix = args.output_prefix
    pd.DataFrame(all_cls).to_csv(TABLES / f"{prefix}_classification.csv", index=False)
    pd.DataFrame(all_reg).to_csv(TABLES / f"{prefix}_regression.csv", index=False)
    pd.DataFrame(all_rank).to_csv(TABLES / f"{prefix}_ranking.csv", index=False)
    pd.DataFrame(dim_rows).to_csv(TABLES / f"{prefix}_dimensions.csv", index=False)
    pd.DataFrame(all_importance).to_csv(TABLES / f"{prefix}_importance.csv", index=False)
    grouped_feature_importance(all_importance).to_csv(
        TABLES / f"{prefix}_group_importance.csv", index=False
    )
    with open(ART / f"{prefix}_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "data": str(data_path),
                "highly_cited_threshold": threshold,
                "feature_sets": all_config,
                "unavailable_requested_features": {
                    "author_identity_or_prestige": "No author names/IDs/history in dataset; author_count is already in base.",
                    "reference_graph": "No cited-paper ID list/graph in dataset; reference_count is already in base.",
                    "neural_full_text_embedding": "No precomputed embedding available; this ablation uses full_text TF-IDF -> LSA.",
                },
            },
            fh,
            indent=2,
        )

    print(
        f"Done in {time.time() - t0:.0f}s -> {prefix}_*.csv in {TABLES}",
        flush=True,
    )


if __name__ == "__main__":
    main()
