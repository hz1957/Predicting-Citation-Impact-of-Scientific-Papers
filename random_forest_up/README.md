# Supervised Learning (final) — Random Forest Baselines + Cluster-Augmented Models

Implements the final supervised-learning stage of the project on the **c1 cohort
temporal split** (train: papers published during 2018–2020, test: papers
published during 2021–2022). This stage predicts future citation impact using
Random Forest regression and classification models while reusing the feature
representations developed in the previous supervised and unsupervised learning
pipelines.

The repository contains two complementary experiments:

- **Baseline Random Forest** trained on the supervised feature matrix.
- **Cluster-Augmented Random Forest** that incorporates semantic cluster
  information learned during the unsupervised K-Means stage.

---

# Environment setup (conda)

```bash
conda create -n cs7641-team7 python=3.12 -y
conda activate cs7641-team7
```

This allow to set the given continuity of our team project kernal.


# Pipeline

The supervised pipeline assumes that the preprocessing stage has already been
completed and that the feature matrices are available inside `artifacts/`.

## Step 1 — Build supervised features

```bash
python run_features.py
python run_baselines.py
python run_cluster_augmented_baselines.py
```
Which covered in the previous midterm supervised setting, this generates

- sparse training/testing feature matrices
- target variables
- preprocessing transformers

stored under

```
artifacts/
```

---

## Step 2 — Train baseline Random Forest

First,
```bash
python run_random_forest_basecheck.py
```

and perform hyperparameter tuning in different folder

```bash
python run_random_forest_basecheck.py --n-estimators 300 --max-depth 20 --min-samples-leaf 10 --max-features sqrt
```

This script trains

- Random Forest Regressor
- Random Forest Classifier

and generates all baseline evaluation figures and tables.

# Reusing Previous Stages

This project intentionally reuses artifacts generated during earlier stages of
the machine learning pipeline. Thus, step 3 require supervised stage results and some of unsupervised stage results

## Supervised stage

The following files are loaded directly from the supervised preprocessing stage.

```
X_train.npz
X_test.npz

y_train.npy
y_test.npy

target_citations_train.npy
target_citations_test.npy
```

These contain the final sparse feature matrices and regression/classification
targets used throughout the Random Forest experiments.

---

## Step 3 — Train cluster-augmented Random Forest

Reuse the tuned parameters from the baseline

```bash
python run_cluster_augmented_random_forest.py \
    --reuse-baseline-params
```

and perform hyperparameter tuning in different folder as well

```bash
python run_cluster_augmented_random_forest.py --n-estimators 300 --max-depth 20 --min-samples-leaf 10 --max-features sqrt
```

This stage augments the supervised feature matrix with semantic cluster features
obtained from the unsupervised learning pipeline and compares performance against
the baseline model.

---

# Reusing Previous Stages #2

---

## Unsupervised stage

The cluster-augmented model reuses the semantic representations learned during
the unsupervised learning phase shared in midterm report. 

For simplicity direct copy paste of resulting three artifacts were selected and copy pasted inside the artifact folder for resued in augmented_random_forest.py

```
tfidf_vectorizer.joblib

text_lsa_transformer.joblib

kmeans_text.joblib
```

instead of fitting new clustering models.

Each paper is transformed into the existing LSA space, assigned to the learned
K-Means clusters, and represented using

- cluster membership (one-hot encoding)
- distances to each cluster centroid

These cluster-derived features are appended to the supervised feature matrix
before training the Random Forest models.

This allows the supervised model to incorporate higher-level semantic structure
without introducing information leakage.

---

# What Each Script Does

## run_random_forest_basecheck.py

Implements the baseline supervised learning experiments.

This script

- trains Random Forest regression
- trains Random Forest classification
- optionally performs RandomizedSearchCV
- computes regression metrics
- computes ranking metrics
- computes classification metrics
- saves trained models
- generates publication-ready figures
- computes permutation feature importance

Generated artifacts include Regression

- RMSE
- MAE
- R²
- Spearman correlation

which uses same metric as supervised regression method so that one-to-one comparision available.

---

## run_cluster_augmented_random_forest.py

Builds the cluster-enhanced Random Forest models.

The script

1. Loads the baseline supervised feature matrices.
2. Loads the TF-IDF vectorizer and LSA transformer from the unsupervised stage.
3. Loads the trained K-Means model.
4. Projects train/test papers into the LSA space.
5. Predicts cluster assignments.
6. Computes one-hot cluster features.
7. Computes distances to every cluster centroid.
8. Concatenates these new features with the original supervised feature matrix.
9. Trains Random Forest models on

- baseline features
- baseline + cluster features

10. Compares predictive performance across both feature sets.

---

# Outputs

Generated outputs are organized as

```
outputs/

├── figures/
│   rf_pr_curve.png
    rf_pred_vs_true.png
    rf_residuals.png
    rf_roc_curve.png
├── tables/
    rf_classification_feature_importance.csv
    rf_classification_metrics.csv
    rf_classification_metrics_cluster_compare.csv
    rf_ranking_metrics.csv
    rf_ranking_metrics_cluster_compare.csv
    rf_regression_feature_importance.csv
    rf_regression_metrics.csv
    rf_regression_metrics_cluster_compare.csv


```
For hyperparameter tuning cases

```
outputs_finetune/

├── figures/
│   rf_pr_curve.png
    rf_pred_vs_true.png
    rf_residuals.png
    rf_roc_curve.png
├── tables/
    rf_classification_feature_importance.csv
    rf_classification_metrics.csv
    rf_classification_metrics_cluster_compare.csv
    rf_ranking_metrics.csv
    rf_ranking_metrics_cluster_compare.csv
    rf_regression_feature_importance.csv
    rf_regression_metrics.csv
    rf_regression_metrics_cluster_compare.csv
```

---

# Headline Findings

The Random Forest models extend the supervised learning pipeline by leveraging
both engineered metadata features and latent semantic representations of paper
content.

Key observations include

- Random Forest provides a strong nonlinear baseline for citation prediction.
- Regression, ranking, and classification tasks are evaluated within a unified
  framework.
- Hyperparameter tuning further improves predictive performance through
  RandomizedSearchCV.
- Cluster-derived semantic features provide complementary information learned
  during the unsupervised learning stage.
- Comparing the baseline and cluster-augmented models quantifies the value of
  transferring unsupervised structure into supervised prediction.
- Feature importance analysis identifies the variables that contribute most to
  citation prediction and highly cited paper classification.

The resulting models complete the machine learning pipeline by integrating
feature engineering, unsupervised representation learning, and supervised
prediction into a unified workflow.