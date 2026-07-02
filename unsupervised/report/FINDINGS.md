# Unsupervised Learning — findings and visualization recommendations

Everything below comes from the c1 cohort **train split only** (24,691 papers,
2018–2020). Reproduce with the three scripts in `unsupervised/` (see its
README). All numbers cited here are in `unsupervised/outputs/tables/`.

## 0. Glossary

Terms below are our labels for derived quantities, not raw dataset fields
(those are in the top-level `DATA_DICTIONARY.md`).

- **LSA / text (LSA) features** — TF-IDF over title + abstract, reduced to 100
  dims with TruncatedSVD and L2-normalized (LSA = latent semantic analysis).
  Both clustering analyses use this same text block; they differ only in
  whether the metadata block is concatenated to it.
- **Assembled features** — the LSA text block concatenated with the
  standardized metadata block, block norms equalized so neither dominates.
- **(Impact) archetype** — a cluster from the finer k = 8 K-Means on the text
  features alone; "impact" because these text-defined groups separate sharply
  by citation outcome. Names ("NLP & speech", …) are hand-assigned from each
  cluster's top TF-IDF terms.
- **Category split** — the coarse k = 3 clustering on the assembled features;
  named for the observed outcome (clusters match the arXiv primary categories
  cs.CV / cs.CL / cs.LG & other), not for anything the algorithm was told.
- **Top decile / top 10%** — 3-yr citations in the top 10% of the train split.
- **Heavy-tail archetype** — mean citations far above the median (a few
  mega-hits dominate), e.g. NLP.
- **Metadata z-score** — a metadata field's per-cluster mean after
  standardization (z = 0 is the corpus average).

## 1. What was done

1. **Features exactly as proposed**: TF-IDF on title + abstract (13,775 terms)
   compressed to 100 dims with TruncatedSVD (L2-normalized), plus a
   standardized metadata block (author/reference counts, text lengths, year,
   log venue frequency, category/affiliation counts, cs.LG/CL/CV indicators).
   No citation-derived field is ever used as a feature.
2. **K-Means, silhouette-selected k** on the assembled features, plus a finer
   K-Means on the text (LSA) features alone.
3. **PCA, UMAP and t-SNE (all on all 24,691 papers)**, computed separately per
   feature set so each analysis is shown in its own space, colored by
   cluster and by log citations.

## 2. Key findings (the story to tell)

1. **Silhouette selects k = 3 on the assembled features (s = 0.100)** and the
   three clusters match the arXiv category split almost perfectly (99.9%
   cs.CV / 100% cs.CL / cs.LG & other rest). Even at this coarse level
   outcomes differ: cs.CV 14.7% top-decile rate vs 7.6% for the cs.LG &
   other mass. The cs.CL
   cluster has mean 51.4 citations vs median 6 — the transformer-era
   mega-hits.
2. **Eight text-space archetypes separate sharply in citation outcome.**
   Median 3-yr citations / top-decile share / uncited share:
   | C | archetype | n | median | top 10% | uncited |
   |---|-----------|---|--------|---------|---------|
   | 0 | 3D & object detection | 3,496 | 7 | 16.0% | 9.8% |
   | 1 | reinforcement learning | 1,422 | 4.5 | 7.2% | 11.5% |
   | 2 | applied ML & prediction | 4,714 | 4 | 8.7% | 13.6% |
   | 3 | optimization & theory | 3,292 | 3 | 4.0% | 16.2% |
   | 4 | deep architectures | 3,436 | 6 | 10.1% | 11.3% |
   | 5 | GANs & adversarial | 1,452 | 6 | 10.8% | 9.8% |
   | 6 | image segmentation | 2,849 | 6 | 11.8% | 12.2% |
   | 7 | NLP & speech | 4,030 | 6 | 11.3% | 11.1% |
3. **UMAP and t-SNE agree** on the topical islands (robustness check), PCA
   shows the same gradient linearly but with heavy overlap, and the
   citation-colored maps show impact concentrating in specific regions (dense
   dark pockets in detection/segmentation and NLP areas) rather than spreading
   uniformly.
4. **Metadata texture**: theory papers have fewer authors (z = −0.40); vision
   papers more authors and affiliations; NLP abstracts are shorter (z = −0.31).
5. **Hand-off to the supervised stage**: text/topic features carry real signal;
   cluster id or distance-to-centroid are cheap features to add; report
   per-archetype errors, not just global metrics (theory papers are the
   low-signal regime).

## 3. Where the figures are

Every figure exists in two forms, same content:

- `unsupervised/outputs/web/*.png` — 200 dpi PNG on the light surface, for the
  website
- `unsupervised/outputs/paper/*.pdf` — vector PDF (scatter layers rasterized
  at 200 dpi so files stay small), for Overleaf

## 4. Recommendations — website (gh-pages midterm page)

A drop-in section is ready at
`unsupervised/report/midterm_unsupervised_section.html`.

1. Copy `outputs/web/*.png` to `CS7641-Web/assets/img/midterm/`.
2. Paste the section into `progress-documents/midterm/index.html` replacing
   the placeholder; renumber figures if earlier sections add their own.

The snippet uses `fig_model_selection`, `fig_clusters_text`, and
`fig_citation_map`; `fig_clusters_text_meta` is optional. Keep the captions
and the C0..C7 labels on the maps (they are the colorblind fallback).

## 5. Recommendations — paper (Overleaf midterm)

A drop-in section is ready at
`unsupervised/report/midterm_unsupervised_section.tex`.

1. Copy `outputs/paper/fig_model_selection.pdf`, `fig_clusters_text.pdf`,
   and `fig_citation_map.pdf` into the Overleaf project
   (`fig_clusters_text_meta.pdf` optional).
2. `\input` or paste the section; it defines `fig:modelsel`, `fig:clusters`,
   `fig:citemap`.

Use the PDFs (vector text, pre-rasterized point clouds, under ~0.5 MB each);
they are sized for `width=\linewidth` in a single-column layout. If anyone
re-styles the figures, keep one fixed hue per cluster across all figures and
a single-hue ramp for citation magnitude.

## 6. Data issue to flag to the team

`c2.jsonl.gz` and `c3.jsonl.gz` on the shared Google Drive appear to be
**truncated uploads**: downloading them completes at the size Drive reports
(903 MB / 875 MB, vs ~1.0/1.1 GB expected) but both fail `gzip -t` with
"unexpected end of file", twice in a row. `c1.jsonl.gz` is intact (verified),
and c1 is all the midterm needs, but whoever owns the Drive folder should
re-upload c2/c3 before the final report's forecasting experiments.
