# Predicting Citation Impact of Scientific Papers - Dataset (CS 7641, Team 7)

A curated dataset of ~150,000 AI/ML arXiv papers (categories `cs.LG`, `cs.CL`,
`cs.CV`) labeled with citation impact and enriched with metadata, affiliations,
venues, citation time series, and **full paper text**. Built for the project
"Predicting Citation Impact of Scientific Papers".

Each paper is one self-contained JSON record. Everything you need to train and
evaluate a model is in the files under `data/`.

---

## 1. Files

- Download data from [here](https://drive.google.com/drive/folders/1q59-Eyv4w28oXuL33fPhEyQCEsNTKBdR?usp=sharing) to `ML-Project-Team-7/data`

```
ML-Project-Team-7/
├── README.md              # this file
├── DATA_DICTIONARY.md     # every field: type, meaning, coverage, source, leakage flag
├── data/
│   ├── c1.jsonl.gz        # 49,999 papers, 2018-2022 (mature)        ~0.9 GB
│   ├── c2.jsonl.gz        # 50,000 papers, 2023-2024 (semi-mature)   ~1.0 GB
│   └── c3.jsonl.gz        # 49,974 papers, 2025-2026 (fresh)         ~1.1 GB
└── samples/
    ├── c1_sample.json     # 50 records, pretty-printed, full_text truncated
    ├── c2_sample.json
    └── c3_sample.json
```

- Format is **JSONL** (one JSON object per line), **gzip**-compressed. Read it
  directly - no need to unzip first (see Section 4). Open a `samples/*.json`
  file in any editor to see the exact structure.
- Total: **149,973 papers**, ~3.1 GB compressed (~13 GB uncompressed).

---

## 2. Cohorts

Cohorts are mutually exclusive and split by publication year. They differ in how
"mature" their citation counts are, which changes the prediction target.

| Cohort | Years     | Maturity    | Papers  | Target                         | Intended use                       |
|--------|-----------|-------------|---------|--------------------------------|------------------------------------|
| **c1** | 2018-2022 | mature      | 49,999  | citations in first **3 years** | primary training + evaluation      |
| **c2** | 2023-2024 | semi-mature | 50,000  | citations in first **1 year**  | secondary / generalization         |
| **c3** | 2025-2026 | fresh       | 49,974  | citations **to date**          | forecasting / inference (no window)|

**C1 ships a temporal split** (field `split`): train = papers published <= 2020,
test = 2021-2022. This mimics real forecasting (train on the past, predict the
future) and is the recommended setup for your main experiments. C2/C3 have no
predefined split (`split` is null); split them yourself if needed.

---

## 3. The prediction target

- `target_citations` - the citation count to predict (per-cohort, see table above).
- `y_log` - `log1p(target_citations)`. **Train on this** (citation counts are
  extremely right-skewed) and exponentiate predictions back with `expm1`.
- `target_is_window` - `true` if the target is a clean fixed-window count from
  the citation time series; `false` if it fell back to total citations (used when
  the per-year history was unavailable). `citation_window_years` records the
  window length (3 for c1, 1 for c2, null for c3).

Target distribution (note the heavy zero-inflation and skew - this is the core
modeling challenge):

| Cohort | mean | median | max     | % zero |
|--------|------|--------|---------|--------|
| c1     | 18.0 | 4      | 116,427 | 13%    |
| c2     | 2.8  | 0      | 25,346  | 59%    |
| c3     | 3.5  | 0      | 5,340   | 52%    |

### IMPORTANT - do not leak the target
These fields are **outcomes** (computed from citations) and must NOT be used as
input features, or your model will "cheat":

`citations_total`, `citations_window`, `citations_preprint`,
`citations_published`, `influential_citations`, `counts_by_year`,
`citations_source`.

Safe-to-use predictors: title, abstract, full_text, `author_count`,
`reference_count`, `venue`, `venue_freq`, `n_affiliations`, `n_countries`,
`affiliations`, `countries`, `field`, `topic`, `primary_category`,
`n_categories`, `year`, the text-length fields, and `ft_n_chars` / `ft_n_words` /
`has_fulltext`. See `DATA_DICTIONARY.md` for the full per-field breakdown.

---

## 4. How to load

```python
import pandas as pd

# Reads the gzipped JSONL directly (pandas handles .gz). ~0.9-1.1 GB each.
c1 = pd.read_json("data/c1.jsonl.gz", lines=True)

# C1 temporal split is built in:
train = c1[c1["split"] == "train"]
test  = c1[c1["split"] == "test"]

y_train = train["y_log"]                      # train target (log scale)
# pick your own predictor columns (exclude the leakage fields listed above):
feature_cols = ["author_count", "reference_count", "venue_freq",
                "n_affiliations", "n_countries", "abstract", "full_text", "year"]
X_train = train[feature_cols]
```

Memory-light streaming (avoid loading a whole cohort at once):

```python
import gzip, json
with gzip.open("data/c3.jsonl.gz", "rt", encoding="utf-8") as fh:
    for line in fh:
        paper = json.loads(line)
        ...
```

Command-line peek:

```bash
zcat data/c1.jsonl.gz | head -n 1 | python -m json.tool
```

### Prebuilt feature matrices (optional)
If you would rather skip feature engineering, the main repo also ships
ready-to-model matrices (TF-IDF + SVD on text plus scaled/encoded metadata,
321 features) as NumPy arrays in `../data/processed/features/`
(`X_c1_train.npy`, `y_c1_train.npy`, `X_c1_test.npy`, `X_c2.npy`, `X_c3.npy`,
`feature_names.json`, `transformers.joblib`). These are not part of this JSON
package because `.npy` is not JSON-native.

---

## 5. Coverage

Most fields are near-complete. The two lower-coverage fields (`countries`,
`counts_by_year`) are at their natural limit: very recent papers simply have no
citation history yet, and some papers expose no resolvable affiliation country.

| Field             | c1     | c2     | c3     |
|-------------------|--------|--------|--------|
| venue             | 100%   | 100%   | 99.3%  |
| field             | 98.7%  | 99.5%  | 91.5%  |
| affiliations      | 94.8%  | 92.5%  | 87.4%  |
| countries         | 81.0%  | 68.8%  | 55.3%  |
| counts_by_year    | 83.5%  | 67.6%  | 13.3%  |
| full_text         | 99.5%  | 99.6%  | 99.5%  |
| fixed-window target | 83.5% | 67.6% | n/a    |

Citation counts and reference counts are ~100% across all cohorts.

---

## 6. Sources & attribution

- **arXiv metadata** (titles, abstracts, authors, categories, DOIs): the
  `librarian-bots/arxiv-metadata-snapshot` dataset on Hugging Face.
- **Citations, references, venues, affiliations, citation time series, fields,
  topics**: [OpenAlex](https://openalex.org), with
  [Semantic Scholar](https://www.semanticscholar.org/product/api) and
  [Crossref](https://www.crossref.org) as supplementary sources.
- **Affiliations / countries from PDF headers**: parsed with
  [GROBID](https://github.com/kermitt2/grobid).
- **Full text**: extracted from the arXiv PDF of each paper.

Each enriched field carries a `*_source` column recording where its value came
from. Please respect the terms of use of the underlying sources; this dataset is
for the CS 7641 course project.

The full reproducible pipeline (download -> sample -> enrich -> curate ->
featurize -> full text) lives in the parent repository under `src/`.
