# Supervised Learning (phase 1) - Ridge + Logistic Regression baselines

This folder will hold the first supervised-learning pass for the project:
simple, reproducible baselines that match the proposal before we add heavier
models such as Random Forests and XGBoost.

Phase 1 uses the **c1 cohort temporal split**:

- train: papers with `split == "train"` (2018-2020)
- test: papers with `split == "test"` (2021-2022)

The target for regression is `y_log = log1p(target_citations)`, exactly as
recommended in the top-level README. Citation-derived leakage fields are never
used as predictors.

## Why not directly reuse `unsupervised/run_features.py`?

We should reuse the **feature design** from the unsupervised pipeline, but not
its saved feature matrix directly.

The unsupervised script builds features for one split at a time and fits
TF-IDF, SVD, and scaling on that input. For supervised evaluation, preprocessing
must be fit on the training split only, then applied unchanged to the test
split:

1. fit `TfidfVectorizer` on train title + abstract
2. fit `TruncatedSVD` on train TF-IDF
3. fit metadata scalers/encoders on train metadata
4. transform the held-out test split with those fitted objects

This avoids preprocessing leakage from the 2021-2022 test papers. The supervised
`run_features.py` should therefore reuse the unsupervised choices and helper
ideas, but write train/test matrices and save the fitted transformers.

## Feature Set

Phase 1 keeps the feature set close to the unsupervised analysis:

- text: `title + abstract`
- text transform: TF-IDF -> TruncatedSVD / LSA
- numeric metadata: `author_count`, `reference_count`, `venue_freq`,
  `n_affiliations`, `n_countries`, `year`, `n_categories`, title/abstract
  length fields
- categorical metadata: `primary_category`

Do not use leakage fields such as `citations_total`, `citations_window`,
`counts_by_year`, `citations_preprint`, `citations_published`, or
`influential_citations`.

Full text is intentionally left out of phase 1. It can be added later as a
heavier feature expansion.

## Models

### Ridge Regression

Task: predict `y_log`.

Metrics:

- RMSE
- MAE
- R2
- Spearman rank correlation
- Precision@10%, Recall@10%, and NDCG@10% when Ridge predictions are treated
  as ranking scores

This is the main interpretable regression baseline for later comparison against
Random Forest and XGBoost.

### Logistic Regression

Task: predict whether a paper is highly cited.

Definition:

- compute the top-decile citation threshold on the c1 train split
- label train/test papers as positive if `target_citations >= threshold`

Metrics:

- PR-AUC
- F1
- ROC-AUC
- positive rate and majority/baseline reference scores

Using the train-derived threshold keeps the binary framing consistent with the
temporal evaluation setup.

## Planned Commands

```bash
D:\conda\envs\cs7641-team7\python.exe supervised/run_features.py
D:\conda\envs\cs7641-team7\python.exe supervised/run_baselines.py
D:\conda\envs\cs7641-team7\python.exe supervised/run_cluster_augmented_baselines.py
```

The scripts expect the full dataset at:

```text
data/c1.jsonl.gz
```

## Outputs

```text
supervised/
  artifacts/
    transformers.joblib
    X_train.npz
    X_test.npz
    y_train.npy
    y_test.npy
    target_citations_train.npy
    target_citations_test.npy
    train_meta.csv.gz
    test_meta.csv.gz
    ridge_model.joblib
    logistic_model.joblib
  outputs/
    tables/
      regression_metrics.csv
      ranking_metrics.csv
      classification_metrics.csv
      regression_metrics_cluster_compare.csv
      ranking_metrics_cluster_compare.csv
      classification_metrics_cluster_compare.csv
    figures/
      ridge_pred_vs_true.png
      ridge_residuals.png
      logistic_pr_curve.png
      logistic_roc_curve.png
  report/
    midterm_supervised_section.html
    midterm_supervised_section.tex
```

`artifacts/` is local and gitignored. Tables and figures are report-ready
outputs that can be committed once generated. The `report/` files are
drop-in supervised-learning sections modeled after
`unsupervised/report/midterm_unsupervised_section.*`. For the HTML report,
copy the generated PNG figures into the website's `assets/img/midterm/`
folder before pasting the section into the midterm page.

`run_cluster_augmented_baselines.py` expects the unsupervised train-fit
cluster artifacts to exist first:

```bash
D:\conda\envs\cs7641-team7\python.exe unsupervised/run_features.py
D:\conda\envs\cs7641-team7\python.exe unsupervised/run_clustering.py
```

It loads the unsupervised text-only `k=8` K-Means model, appends cluster-id
one-hot features plus distances to the eight centroids, and writes comparison
CSV files without overwriting the original baseline metrics.

## Phase 1 Completion Criteria

Phase 1 is complete when:

1. train/test features are generated without leakage
2. Ridge Regression runs on `y_log`
3. Logistic Regression runs on the train-defined top-decile label
4. metrics are written to CSV
5. baseline figures are written for the report

After this, phase 2 can add Random Forest, XGBoost, tuning, ablations, and
feature-importance/SHAP analysis.

## Environment

Use the clean `cs7641-team7` conda environment for this phase:

```bash
conda create -n cs7641-team7 -c conda-forge python=3.12 numpy scipy scikit-learn pandas matplotlib joblib threadpoolctl -y
conda install -n cs7641-team7 --override-channels -c conda-forge "libblas=*=*openblas" "liblapack=*=*openblas" -y
```

The OpenBLAS variant avoids a Windows `0xc06d007f` DLL/threadpool failure seen
with an MKL-backed environment while keeping the default 100-dimensional
randomized SVD used by the feature pipeline.

Fallback only: if randomized SVD still fails in another Windows setup, rerun
with `--svd-algorithm arpack`. That keeps the same 100-dimensional SVD/LSA
feature design, but is not bit-for-bit identical to the default randomized SVD.
