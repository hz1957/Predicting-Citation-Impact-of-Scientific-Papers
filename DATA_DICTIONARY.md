# Data Dictionary

Every field present in each JSON record (`data/c{1,2,3}.jsonl.gz`). 42 fields per
paper. The **Role** column tells you how to use each field:

- **predictor** - safe to use as a model input feature.
- **target** - the value to predict (or its components).
- **LEAKAGE** - derived from the citation outcome; do NOT use as an input feature.
- **id / meta** - identifiers and bookkeeping; not a feature.

Coverage is shown as c1 / c2 / c3 (percent of rows with a non-empty value).
Where a field is essentially always present it is marked ~100%.

## Identifiers & cohort

| Field | Type | Role | Coverage | Description |
|-------|------|------|----------|-------------|
| `arxiv_id` | string | id | ~100% | arXiv identifier, e.g. `2009.13303`. Unique key per record. |
| `openalex_id` | string\|null | id | ~95% | OpenAlex work id, when matched. |
| `doi` | string\|null | id | partial | DOI (published or arXiv DOI), when available. |
| `cohort` | string | meta | 100% | `c1`, `c2`, or `c3`. |
| `maturity` | string | meta | 100% | `mature` (c1), `semi-mature` (c2), `fresh` (c3). |
| `split` | string\|null | meta | c1 only | `train` / `test` for c1 (temporal split); null for c2/c3. |

## Text content

| Field | Type | Role | Coverage | Description |
|-------|------|------|----------|-------------|
| `title` | string | predictor | ~100% | Paper title. |
| `abstract` | string | predictor | ~100% | Paper abstract. |
| `full_text` | string\|null | predictor | 99.5 / 99.6 / 99.5% | Full body text extracted from the arXiv PDF. The largest field; placed last in each record. |
| `title_len_chars` | int | predictor | ~100% | Character count of the title. |
| `title_len_words` | int | predictor | ~100% | Word count of the title. |
| `abstract_len_chars` | int | predictor | ~100% | Character count of the abstract. |
| `abstract_len_words` | int | predictor | ~100% | Word count of the abstract. |
| `ft_n_chars` | int | predictor | ~100% | Character count of `full_text` (0 if missing). |
| `ft_n_words` | int | predictor | ~100% | Word count of `full_text` (0 if missing). |
| `has_fulltext` | bool | predictor | ~100% | Whether full text was successfully extracted. |

## Bibliographic / metadata predictors

| Field | Type | Role | Coverage | Description |
|-------|------|------|----------|-------------|
| `year` | int | predictor | ~100% | Publication year. (Also the basis of the c1 temporal split.) |
| `primary_category` | string | predictor | ~100% | Primary arXiv category, e.g. `cs.LG`, `cs.CL`, `cs.CV`. |
| `n_categories` | int | predictor | ~100% | Number of arXiv categories the paper is tagged with. |
| `author_count` | int | predictor | ~100% | Number of authors. |
| `reference_count` | int | predictor | ~100% | Number of references the paper cites (outgoing). |
| `venue` | string | predictor | 100 / 100 / 99.3% | Publication venue / journal / conference (`Unknown` if not resolved). |
| `venue_freq` | int | predictor | ~100% | How many papers in the corpus share this venue (venue popularity). |
| `venue_source` | string | meta | ~100% | Which source provided the venue (provenance). |
| `n_affiliations` | int | predictor | 94.8 / 92.5 / 87.4% | Number of distinct author affiliations. |
| `affiliations` | list[string]\|null | predictor | 94.8 / 92.5 / 87.4% | Raw affiliation strings (institutions). |
| `affiliations_source` | string | meta | ~100% | Which source provided affiliations (provenance). |
| `n_countries` | int | predictor | 81.0 / 68.8 / 55.3% | Number of distinct author countries. |
| `countries` | list[string]\|null | predictor | 81.0 / 68.8 / 55.3% | ISO country codes of author institutions, e.g. `["US","FI"]`. |
| `field` | string | predictor | 98.7 / 99.5 / 91.5% | High-level field of study (e.g. Computer Science). |
| `topic` | string\|null | predictor | partial | Finer-grained topic label. |

## Target (predict these)

| Field | Type | Role | Coverage | Description |
|-------|------|------|----------|-------------|
| `target_citations` | float | **target** | ~100% | The citation count to predict for this cohort (3-yr window for c1, 1-yr for c2, to-date for c3). |
| `y_log` | float | **target** | ~100% | `log1p(target_citations)`. Train on this; invert with `expm1`. |
| `target_is_window` | bool | meta | ~100% | `true` if target is a clean fixed-window count; `false` if it fell back to total citations. |
| `citation_window_years` | int\|null | meta | c1/c2 | Window length used: 3 (c1), 1 (c2), null (c3). |

## Citation outcomes - LEAKAGE (never use as input features)

| Field | Type | Role | Coverage | Description |
|-------|------|------|----------|-------------|
| `citations_total` | float | LEAKAGE | ~100% | Total citations to date (all years). |
| `citations_window` | float\|null | LEAKAGE | 83.5 / 67.6 / 0% | Citations within the fixed window (the clean target source). |
| `counts_by_year` | list[obj]\|null | LEAKAGE | 83.5 / 67.6 / 13.3% | Citation time series: list of `{"year": int, "cited_by_count": int}`. |
| `citations_preprint` | float\|null | LEAKAGE | partial | Citations attributed to the arXiv preprint version. |
| `citations_published` | float\|null | LEAKAGE | partial | Citations attributed to the published version. |
| `influential_citations` | float\|null | LEAKAGE | partial | Semantic Scholar "influential" citation count. |
| `citations_source` | string | meta | ~100% | Which source provided the citation counts (provenance). |

---

## Notes

- **Lists vs scalars**: `affiliations` and `countries` are the raw lists;
  `n_affiliations` and `n_countries` are their lengths (convenient numeric
  features). Use whichever fits your model.
- **`counts_by_year`** is included for analysis/visualization (e.g. citation
  trajectories) but is a citation outcome - keep it out of your feature set.
- **Nulls**: missing values are JSON `null`. Numeric "not applicable" counts
  (e.g. `ft_n_words` when no text) are `0` with `has_fulltext = false`.
- **c3 target**: because c3 papers are too recent for a fixed window,
  `target_citations` = citations-to-date and `target_is_window` is `false`.
  Treat c3 as a forecasting/inference set, not as window-labeled training data.
