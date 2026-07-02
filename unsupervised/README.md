# Unsupervised Learning (midterm) — K-Means impact archetypes + PCA/UMAP/t-SNE

Implements the Unsupervised Learning part of the proposal on the **c1 cohort
train split** (24,691 papers published 2018–2020; the temporal test split is
never touched, so nothing here leaks into the supervised stage).

## Environment setup (conda)

```bash
conda create -n cs7641-team7 python=3.12 -y
conda activate cs7641-team7
pip install scikit-learn pandas numpy scipy matplotlib umap-learn

# optional: only needed to download the data from the team Google Drive
pip install gdown
```

Any recent Python (3.10+) works; the results in this folder were produced
with Python 3.14.5, scikit-learn 1.9.0, pandas 2.3.3, numpy 2.4.6,
scipy 1.17.1, matplotlib 3.10.9, umap-learn 0.5.12. All clustering and
embedding steps are seeded (`random_state=42`), but expect cosmetically
different embeddings across library versions.

To run a script without activating the env each time:

```bash
conda run -n cs7641-team7 python run_features.py
```

## Pipeline

```bash
# once: download data/c1.jsonl.gz (see top-level README), then
python run_features.py            # step 1: TF-IDF -> SVD + scaled metadata
python run_clustering.py          # step 2: K-Means sweeps, profiles, terms
python run_visualize.py           # step 3: PCA / UMAP / t-SNE + figures
python run_k_explorer.py          # step 4 (optional): per-k images for the
                                  # interactive k-slider in report/preview.html
```

Runtime is a few minutes per step on a laptop CPU (the K-Means sweeps and
t-SNE dominate). Steps cache their outputs (`artifacts/`, gitignored), so
each script can be re-run independently; delete
`artifacts/embeddings_*.npz` to recompute the PCA/UMAP/t-SNE embeddings.

## Viewing the HTML report locally

```bash
bash report/make_preview.sh          # rebuild report/preview.html
open report/preview.html             # open in your browser (no server needed)
```

`preview.html` renders the midterm section
(`report/midterm_unsupervised_section.html`) with the website's real
stylesheet and the figures from `outputs/web/`, plus an interactive
"explorer" section at the bottom with a k-slider per feature set. Notes:

- The stylesheet is loaded from the sibling `../CS7641-Web` checkout, so
  that repo must sit next to this one (otherwise the page is unstyled but
  still readable).
- The k-slider needs the images from `python run_k_explorer.py` (step 4);
  without them the explorer section shows a hint instead.
- Rerun `bash report/make_preview.sh` after editing the section snippet or
  regenerating figures. `preview.html` is local-only and gitignored.

## What each step does

1. **`run_features.py`** — streams the gzipped JSONL (skipping `full_text`),
   builds TF-IDF (title + abstract, 13,775 terms after `min_df=5`) compressed
   to 100 SVD components, plus a standardized metadata block (author count,
   reference count, title/abstract length, year, log venue frequency,
   category counts, affiliation/country counts, and cs.LG / cs.CL / cs.CV
   one-hots). Leakage fields (anything derived from citations) are never
   included; `y_log` is kept aside for coloring/profiling only.

2. **`run_clustering.py`** — two K-Means analyses sharing the identical LSA
   text block; output tags name the inputs (`text_meta`, `text`):
   - **A. `text_meta`** (text block + metadata block, block norms
     equalized — the proposal's "assembled feature vectors"): the silhouette
     score has a clear interior maximum at **k = 3**, and the clusters match
     the arXiv primary categories (cs.CV / cs.CL / cs.LG & other) almost
     exactly.
   - **B. `text`** (the same LSA features alone) at **k = 8**: finer topical
     "impact archetypes" (3D and detection, RL, applied ML, optimization
     theory, deep architectures, GANs and adversarial robustness,
     segmentation, NLP). Here the silhouette rises only slowly and
     monotonically with k (typical for text), so k = 8 was fixed as the
     largest k where every cluster stays interpretable and well populated.
   Both sweeps land in `outputs/tables/model_selection.csv`; profiles,
   per-cluster arXiv category shares, metadata z-scores, and top TF-IDF
   terms land in the other `outputs/tables/*_{text_meta,text}.*` files.

3. **`run_visualize.py`** — four figures, one package per analysis:
   - `fig_model_selection` — silhouette vs k for both feature sets
   - `fig_clusters_text_meta` — analysis A: PCA + UMAP + t-SNE colored by the
     k = 3 clusters, median citations, top-decile share
   - `fig_clusters_text` — analysis B, identical layout for the k = 8
     clusters
   - `fig_citation_map` — PCA + UMAP + t-SNE colored by `log1p(3-yr citations)`
   plus `outputs/tables/cluster_summary_text.csv` (display names + sizes +
   citation stats + top terms). Each analysis is embedded in its own feature
   space: `fig_clusters_text_meta` shows PCA/UMAP/t-SNE computed on the
   text + metadata features, while `fig_clusters_text` and
   `fig_citation_map` use PCA/UMAP/t-SNE computed on the text features
   (caches: `artifacts/embeddings_{text,text_meta}.npz`). Every figure is
   written twice:
   - `outputs/web/*.png` — 200 dpi, for the GitHub Pages midterm report
   - `outputs/paper/*.pdf` — vector (scatter layers rasterized at 200 dpi so
     the PDFs stay small), for the Overleaf midterm report

`plot_style.py` holds the shared palette (colorblind-validated categorical
slots; single-hue sequential ramp for citation magnitude) and rcParams.

## Glossary

These are our own labels for derived quantities, not raw dataset fields (for
the dataset fields themselves see the top-level `DATA_DICTIONARY.md`).

- **LSA / text (LSA) features** — the 100-dimensional text representation:
  TF-IDF over title + abstract, reduced with TruncatedSVD and L2-normalized.
  "LSA" (latent semantic analysis) is the standard name for TF-IDF + SVD.
- **Assembled features** — the feature matrix K-Means analysis A runs on: the
  LSA text block concatenated with the standardized metadata block, with the
  two blocks scaled to carry a comparable row norm so neither dominates.
- **(Impact) archetype** — a cluster from the finer **k = 8** K-Means run on the
  text (LSA) features alone (`labels_text.npy`). "Impact" because these
  purely text-defined groups turn out to separate sharply by citation outcome.
  The short names ("3D & object detection", "NLP & speech", …) are assigned by
  hand from each cluster's top TF-IDF terms, not produced by the algorithm.
- **Category split** — the coarse **k = 3** clustering on the assembled
  features; the name describes the outcome we observed (the clusters match
  the arXiv primary categories cs.CV / cs.CL / cs.LG & other), not anything
  the algorithm was told to find.
- **Silhouette / Davies-Bouldin (DB)** — standard internal cluster-quality
  metrics used to pick k (higher silhouette / lower DB = better separation).
- **Top decile / top 10%** — a paper whose 3-yr citation count is in the top
  10% of the whole train split; "top-decile share" is a cluster's fraction of
  such papers.
- **Heavy-tail archetype** — an archetype whose mean citations sit far above its
  median (a few mega-hits dominate), e.g. NLP.
- **`y_log`** — `log1p(citations in first 3 years)`, the log-scaled outcome used
  only for coloring and profiling here (never as a clustering input).
- **Metadata z-score** — a metadata field's per-cluster mean after
  standardization; z = 0 is the global average, so e.g. z = −0.40 authors means
  that cluster averages 0.40 SD below the corpus mean on author count.
- **Distance-to-centroid** — a paper's distance to its assigned cluster center,
  suggested as a cheap engineered feature for the supervised stage.

## Headline findings

- Clusters on the assembled features match the arXiv categories, and
  citation outcomes differ across them: cs.CV papers reach the global
  top decile at ~15%, cs.CL ~11%, the cs.LG & other bulk ~8%.
- Within the text space, 8 topical archetypes separate sharply in outcome:
  3D/object detection (median 7 citations in 3 yrs, 16% top decile) and
  segmentation, NLP, GAN/adversarial clusters sit high, while optimization
  theory papers sit lowest (median 3, 4% top decile, 16% uncited).
- NLP is the heavy-tail archetype: mean 47.6 vs median 6 citations — a few
  transformer-era mega-hits dominate, motivating the log target.
- Topic membership alone is predictive of citation outcome, so the SVD text
  features should carry signal into the supervised stage; cluster id (or
  distance-to-centroid) is a cheap categorical feature to try.
