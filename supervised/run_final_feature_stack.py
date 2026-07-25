"""Run one coherent final feature-stack experiment.

This script fixes the report-level issue where content/identity ablations and
author/reference history were evaluated in separate pipelines. It builds all
candidate blocks once, then evaluates a single ordered set of feature stacks
with the same XGBoost classifier/regressor settings.
"""

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TABLES = HERE / "outputs" / "tables"
FIGS = HERE / "outputs" / "figures"
MPLCONFIGDIR = ART / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from run_feature_ablations import (
    build_extra_blocks,
    evaluate_feature_set,
    load_raw_split,
    require,
)
from run_phase2_tree_models import grouped_feature_importance
from run_baselines import apply_style
from run_train_history_ablation import (
    FEATURE_NAMES as HISTORY_FEATURE_NAMES,
    build_history_features,
    read_base_records,
    read_enrichment,
)


FEATURE_SET_SPECS = [
    ("base", []),
    ("base_plus_full_text_lsa", ["full_text_lsa"]),
    (
        "base_plus_full_text_affiliation_country",
        ["full_text_lsa", "affiliation_country"],
    ),
    (
        "base_plus_full_text_venue_identity",
        ["full_text_lsa", "venue_identity"],
    ),
    ("base_plus_train_author_reference_history", ["train_history"]),
    (
        "base_plus_full_text_history",
        ["full_text_lsa", "train_history"],
    ),
    (
        "base_plus_full_text_affiliation_country_history",
        ["full_text_lsa", "affiliation_country", "train_history"],
    ),
    (
        "base_plus_full_text_venue_identity_history",
        ["full_text_lsa", "venue_identity", "train_history"],
    ),
    (
        "base_plus_full_text_affiliation_venue_history",
        ["full_text_lsa", "affiliation_country", "venue_identity", "train_history"],
    ),
]

FEATURE_SET_LABELS = {
    "base": "Base",
    "base_plus_full_text_lsa": "+ full text",
    "base_plus_full_text_affiliation_country": "+ full text + affil.",
    "base_plus_full_text_venue_identity": "+ full text + venue",
    "base_plus_train_author_reference_history": "+ history",
    "base_plus_full_text_history": "+ full text + history",
    "base_plus_full_text_affiliation_country_history": "+ full text + affil. + history",
    "base_plus_full_text_venue_identity_history": "+ full text + venue + history",
    "base_plus_full_text_affiliation_venue_history": "Full stack",
}


def select_specs(requested):
    specs_by_name = dict(FEATURE_SET_SPECS)
    if not requested:
        return FEATURE_SET_SPECS
    unknown = [name for name in requested if name not in specs_by_name]
    if unknown:
        valid = ", ".join(specs_by_name)
        raise ValueError(f"Unknown feature set(s): {unknown}. Valid choices: {valid}")
    return [(name, specs_by_name[name]) for name in requested]


def required_extra_blocks(specs):
    return {
        block
        for _, block_names in specs
        for block in block_names
        if block != "train_history"
    }


def build_history_block(df, enrichment_path, threshold, train_count, test_count, output_prefix):
    enrichment = read_enrichment(
        require(enrichment_path, "Run supervised/run_openalex_enrichment.py first.")
    )
    print(
        f"Building train-history block from {len(enrichment):,} enriched works ...",
        flush=True,
    )
    history_matrix, coverage = build_history_features(df, enrichment, threshold)
    coverage.to_csv(TABLES / f"{output_prefix}_coverage.csv", index=False)

    train_mask = (df["split"] == "train").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    history_train_raw = history_matrix[train_mask]
    history_test_raw = history_matrix[test_mask]
    if history_train_raw.shape[0] != train_count or history_test_raw.shape[0] != test_count:
        raise ValueError("History block row counts do not match supervised feature artifacts.")

    scaler = StandardScaler()
    history_train = scaler.fit_transform(history_train_raw).astype(np.float32)
    history_test = scaler.transform(history_test_raw).astype(np.float32)

    sparse.save_npz(
        ART / f"{output_prefix}_history_features_train.npz",
        sparse.csr_matrix(history_train),
    )
    sparse.save_npz(
        ART / f"{output_prefix}_history_features_test.npz",
        sparse.csr_matrix(history_test),
    )
    with open(ART / f"{output_prefix}_history_feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(HISTORY_FEATURE_NAMES, fh, indent=2)

    coverage_stats = {
        "enriched_work_count": int(len(enrichment)),
        "train_enrichment_rate": float(
            coverage[coverage["split"] == "train"]["has_enrichment"].mean()
        ),
        "test_enrichment_rate": float(
            coverage[coverage["split"] == "test"]["has_enrichment"].mean()
        ),
    }
    return (
        sparse.csr_matrix(history_train),
        sparse.csr_matrix(history_test),
        HISTORY_FEATURE_NAMES,
        coverage_stats,
    )


def assemble_feature_set(feature_set, block_names, base_train, base_test, base_names, blocks):
    train_parts = [base_train]
    test_parts = [base_test]
    feature_names = list(base_names)
    for block_name in block_names:
        x_train, x_test, names = blocks[block_name]
        train_parts.append(x_train)
        test_parts.append(x_test)
        feature_names.extend(names)
    return (
        feature_set,
        sparse.hstack(train_parts, format="csr"),
        sparse.hstack(test_parts, format="csr"),
        feature_names,
        block_names,
    )


def save_feature_stack_summary_figure(classification_df, ranking_df, output_prefix):
    FIGS.mkdir(parents=True, exist_ok=True)
    apply_style()

    cls = classification_df[
        (classification_df["split"] == "test")
        & (classification_df["threshold_source"] == "validation_year_2020_max_f1")
    ][["feature_set", "pr_auc", "f1"]]
    rank = ranking_df[ranking_df["split"] == "test"][
        ["feature_set", "ndcg_at_10pct"]
    ]
    present = set(cls["feature_set"]) & set(rank["feature_set"])
    order = [name for name, _ in FEATURE_SET_SPECS if name in present]
    if not order:
        return None
    summary = (
        pd.DataFrame({"feature_set": order})
        .merge(cls, on="feature_set", how="left")
        .merge(rank, on="feature_set", how="left")
    )
    summary["label"] = summary["feature_set"].map(FEATURE_SET_LABELS)

    metrics = [
        ("pr_auc", "PR-AUC", "#3b6ea8", 0.28),
        ("f1", "F1", "#2a9d8f", 0.37),
        ("ndcg_at_10pct", "NDCG@10%", "#c17c2a", 0.76),
    ]
    y_pos = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.8), sharey=True)

    for ax, (column, title, color, xmax) in zip(axes, metrics):
        values = summary[column].to_numpy(dtype=float)
        ax.barh(y_pos, values, color=color, alpha=0.86, height=0.64)
        ax.set_title(title, fontsize=11, color="#2f2f2c", pad=8)
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", alpha=0.55)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("test score")
        for y, value in zip(y_pos, values):
            if np.isfinite(value):
                ax.text(
                    min(value + xmax * 0.015, xmax * 0.985),
                    y,
                    f"{value:.3f}",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color="#3f3e3a",
                )

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(summary["label"], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Final XGBoost feature-stack summary", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = FIGS / f"{output_prefix}_summary.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="c1")
    parser.add_argument("--data", default=None)
    parser.add_argument("--enrichment", default=None)
    parser.add_argument("--output-prefix", default="final_feature_stack")
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=None,
        help="Optional subset of final feature-set names to run.",
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
    FIGS.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data) if args.data else ROOT / "data" / f"{args.cohort}.jsonl.gz"
    enrichment_path = (
        Path(args.enrichment)
        if args.enrichment
        else ART / f"openalex_{args.cohort}_author_ref_enrichment.jsonl.gz"
    )
    specs = select_specs(args.feature_sets)

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

    train_raw, test_raw = load_raw_split(data_path, args.full_text_max_chars)
    if len(train_raw) != base_train.shape[0] or len(test_raw) != base_test.shape[0]:
        raise ValueError("Raw split row counts do not match supervised feature artifacts.")

    df_history = read_base_records(data_path)
    if int((df_history["split"] == "train").sum()) != base_train.shape[0]:
        raise ValueError("History train row count does not match supervised feature artifacts.")
    if int((df_history["split"] == "test").sum()) != base_test.shape[0]:
        raise ValueError("History test row count does not match supervised feature artifacts.")

    threshold = float(np.quantile(cites_train, 0.9))
    high_train = (cites_train >= threshold).astype(int)
    high_test = (cites_test >= threshold).astype(int)

    required_blocks = required_extra_blocks(specs)
    print(
        "Unified feature sets: " + ", ".join(name for name, _ in specs),
        flush=True,
    )
    print(
        "Required non-history blocks: "
        + (", ".join(sorted(required_blocks)) if required_blocks else "none"),
        flush=True,
    )
    blocks = build_extra_blocks(
        train_raw, test_raw, y_train, cites_train, threshold, args, required_blocks
    )

    coverage_stats = {}
    if any("train_history" in block_names for _, block_names in specs):
        history_train, history_test, history_names, coverage_stats = build_history_block(
            df_history,
            enrichment_path,
            threshold,
            base_train.shape[0],
            base_test.shape[0],
            args.output_prefix,
        )
        blocks["train_history"] = (history_train, history_test, history_names)

    all_cls, all_reg, all_rank, all_importance, all_config, dim_rows = [], [], [], [], [], []
    for feature_set, block_names in specs:
        _, x_train, x_test, feature_names, assembled_blocks = assemble_feature_set(
            feature_set, block_names, base_train, base_test, base_names, blocks
        )
        cls, reg, rank, importance, config = evaluate_feature_set(
            feature_set,
            x_train,
            x_test,
            feature_names,
            assembled_blocks,
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
                "blocks_added": "+".join(assembled_blocks) if assembled_blocks else "none",
                "n_features": int(x_train.shape[1]),
                "n_added_features": int(x_train.shape[1] - base_train.shape[1]),
                **coverage_stats,
            }
        )

    prefix = args.output_prefix
    classification_df = pd.DataFrame(all_cls)
    regression_df = pd.DataFrame(all_reg)
    ranking_df = pd.DataFrame(all_rank)
    classification_df.to_csv(TABLES / f"{prefix}_classification.csv", index=False)
    regression_df.to_csv(TABLES / f"{prefix}_regression.csv", index=False)
    ranking_df.to_csv(TABLES / f"{prefix}_ranking.csv", index=False)
    pd.DataFrame(dim_rows).to_csv(TABLES / f"{prefix}_dimensions.csv", index=False)
    pd.DataFrame(all_importance).to_csv(TABLES / f"{prefix}_importance.csv", index=False)
    grouped_feature_importance(all_importance).to_csv(
        TABLES / f"{prefix}_group_importance.csv", index=False
    )
    figure_path = save_feature_stack_summary_figure(
        classification_df, ranking_df, prefix
    )
    with open(ART / f"{prefix}_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "data": str(data_path),
                "enrichment": str(enrichment_path),
                "highly_cited_threshold": threshold,
                "feature_sets": all_config,
                "route": "single unified feature-stack experiment; no cross-pipeline table splicing",
                "leakage_safety": (
                    "All preprocessing is train-fit. Train-history rows use only earlier c1 train rows; "
                    "test history rows use only c1 train rows."
                ),
            },
            fh,
            indent=2,
        )
    figure_msg = f"; summary figure at {figure_path}" if figure_path else ""
    print(
        f"Done in {time.time() - t0:.0f}s -> {prefix}_*.csv in {TABLES}"
        f"{figure_msg}",
        flush=True,
    )


if __name__ == "__main__":
    main()
