# Supervised Learning - phase 1 baselines + phase 2 nonlinear comparison

This folder holds the supervised-learning pass for the project. Phase 1 uses
simple, reproducible baselines that match the proposal. Phase 2 adds one
nonlinear classification model and one nonlinear regression model for direct
comparison against those baselines.

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

The current supervised feature set keeps close to the unsupervised analysis:

- text: `title + abstract`
- text transform: TF-IDF -> TruncatedSVD / LSA
- numeric metadata: `author_count`, `reference_count`, `venue_freq`,
  `n_affiliations`, `n_countries`, `year`, `n_categories`, title/abstract
  length fields
- categorical metadata: `primary_category`

Do not use leakage fields such as `citations_total`, `citations_window`,
`counts_by_year`, `citations_preprint`, `citations_published`, or
`influential_citations`.

Full text is intentionally left out of phases 1-2. It can be added later as a
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
XGBoost regression.

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

### Phase 2: XGBoost Classification

Task: predict the same train-defined top-decile high-citation label used by
Logistic Regression.

Comparison target:

- Logistic Regression vs XGBoost Classifier

The XGBoost Classifier uses a `binary:logistic` objective, `aucpr` early
stopping, and `scale_pos_weight` for the rare high-citation class. It can model
nonlinear interactions among LSA text dimensions, metadata, and category
indicators while keeping the binary evaluation unchanged. The phase 2
comparison table keeps the default `p >= 0.5` rows and also reports
validation-F1-tuned threshold rows for Logistic Regression and XGBoost
Classifier, with thresholds selected only on the latest training year.

### Phase 2: XGBoost Regression

Task: predict the same `y_log = log1p(target_citations)` target used by Ridge.

Comparison targets:

- Ridge Regression vs XGBoost Regressor on RMSE, MAE, R2, and Spearman
- Ridge Regression vs XGBoost Regressor as ranking scores on Precision@10%,
  Recall@10%, NDCG@10%, and Lift@10%

XGBoost is early-stopped on the latest training year, then refit on the full
train split using the selected number of trees. The temporal test split remains
untouched until final evaluation.

### Feature Ablations

`run_feature_ablations.py` keeps the base supervised matrix unchanged and adds
available non-leakage feature groups one at a time:

- `base`
- `base_plus_venue_identity`
- `base_plus_venue_prestige`
- `base_plus_field_topic`
- `base_plus_affiliation_country`
- `base_plus_full_text_lsa`
- `all_available_extra_features`

Full text is represented as capped `full_text` TF-IDF followed by LSA because
the dataset does not include precomputed neural full-text embeddings. The
default quick ablation uses the first 8,000 full-text characters, 30,000 TF-IDF
terms, and 50 LSA dimensions; these can be increased after the first pass.
Venue prestige is a train-only temporal target-encoding proxy: train rows use
prior training years where available, and test rows use only train-derived venue
statistics. The dataset does not include author names/IDs or cited-paper ID
lists, so author prestige and reference graph features cannot be constructed
without external data; `author_count` and `reference_count` are already in the
base feature matrix.

## Planned Commands

```bash
D:\conda\envs\cs7641-team7\python.exe supervised/run_features.py
D:\conda\envs\cs7641-team7\python.exe supervised/run_baselines.py
D:\conda\envs\cs7641-team7\python.exe supervised/run_cluster_augmented_baselines.py
D:\conda\Scripts\conda.exe install -n cs7641-team7 -c conda-forge xgboost -y
D:\conda\envs\cs7641-team7\python.exe supervised/run_phase2_tree_models.py
D:\conda\envs\cs7641-team7\python.exe supervised/run_feature_ablations.py
D:\conda\envs\cs7641-team7\python.exe supervised/run_feature_ablations.py --feature-sets base_plus_full_text_venue_identity base_plus_full_text_affiliation_country --output-prefix feature_ablation_combo
D:\conda\envs\cs7641-team7\python.exe supervised/run_openalex_enrichment.py --cohort c1
D:\conda\envs\cs7641-team7\python.exe supervised/run_train_history_ablation.py
D:\conda\envs\cs7641-team7\python.exe supervised/run_final_feature_stack.py
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
    xgboost_classifier.joblib
    xgboost_regressor.joblib
  outputs/
    tables/
      regression_metrics.csv
      ranking_metrics.csv
      classification_metrics.csv
      regression_metrics_cluster_compare.csv
      ranking_metrics_cluster_compare.csv
      classification_metrics_cluster_compare.csv
      final_feature_stack_classification.csv
      final_feature_stack_regression.csv
      final_feature_stack_ranking.csv
      final_feature_stack_dimensions.csv
      final_feature_stack_group_importance.csv
      final_feature_stack_importance.csv
    figures/
      ridge_pred_vs_true.png
      logistic_pr_curve.png
      logistic_roc_curve.png
      final_feature_stack_summary.png
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

## Phase 2 Completion Criteria

Phase 2 is complete when:

1. XGBoost Classifier runs on the top-decile citation label
2. XGBoost Regressor runs on `y_log`
3. phase 2 comparison CSVs include the original Ridge/Logistic rows plus the
   new nonlinear-model rows
4. phase 2 figures and feature-importance tables are written for the report
5. the report discusses whether nonlinear models improve classification,
   regression, and ranking performance over phase 1

## Feature Ablation Completion Criteria

Feature ablation is complete when:

1. each feature set is trained with the same XGBoost classification and
   regression settings
2. classification, regression, and ranking ablation tables are written
3. feature dimensions and feature-importance summaries identify which added
   blocks carry signal
4. unavailable requested features are explicitly documented rather than inferred
   from leakage fields or external data

## OpenAlex Train-History Features

`run_openalex_enrichment.py` uses the local `openalex_id` field to fetch
OpenAlex `authorships` and `referenced_works` identifiers. It stores only IDs
and graph edges, not current citation counts or current author metrics.
OpenAlex list requests may require an `OPENALEX_API_KEY` for full c1 enrichment;
without one, use `--limit` for a small smoke test.

`run_train_history_ablation.py` turns the enrichment into time-safe features
using only c1 train rows:

- train rows use only earlier train years
- test rows use only earlier train rows
- author features summarize prior paper count, prior high-impact rate, prior
  log-citation history, and prior coauthor counts
- reference features summarize references that match earlier train works,
  including prior impact, age, and author overlap

This gives us author/reference history signal without using 2026 author profiles
or citation counts from outside the training split.

## Unified Final Feature Stack

`run_final_feature_stack.py` is the report-ready final experiment. It avoids
mixing results from separate ablation pipelines by building full-text LSA,
affiliation/country, venue identity, and train-only author/reference history
blocks in one script, then evaluating all selected feature stacks with the same
XGBClassifier and XGBoost regressor settings. The default route includes base,
content-only, identity/content, history-only, and combined content+identity+
history stacks.

## Environment

Use the clean `cs7641-team7` conda environment for this phase:

```bash
conda create -n cs7641-team7 -c conda-forge python=3.12 numpy scipy scikit-learn pandas matplotlib joblib threadpoolctl -y
conda install -n cs7641-team7 --override-channels -c conda-forge "libblas=*=*openblas" "liblapack=*=*openblas" -y
conda install -n cs7641-team7 -c conda-forge xgboost -y
```

The OpenBLAS variant avoids a Windows `0xc06d007f` DLL/threadpool failure seen
with an MKL-backed environment while keeping the default 100-dimensional
randomized SVD used by the feature pipeline.

Fallback only: if randomized SVD still fails in another Windows setup, rerun
with `--svd-algorithm arpack`. That keeps the same 100-dimensional SVD/LSA
feature design, but is not bit-for-bit identical to the default randomized SVD.
