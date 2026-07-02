"""Step 1 — build the assembled feature matrix for the unsupervised analysis.

Implements the preprocessing described in the proposal (Methods / Data
Preprocessing) on the c1 cohort train split (papers published <= 2020):

  1. TF-IDF on title + abstract  ->  TruncatedSVD (100 components)
  2. metadata features (author count, reference count, title/abstract length,
     year, frequency-encoded venue, category counts, affiliation counts)
     scaled with StandardScaler
  3. target kept aside as y_log = log1p(target_citations)  (NEVER a feature)

Outputs (unsupervised/artifacts/):
  features.npz   X (standardized, float32), y_log, target_citations, meta arrays
  tfidf.npz      sparse TF-IDF matrix (for per-cluster top-term profiling)
  vocab.json     TF-IDF vocabulary (index -> term)

Usage:  python run_features.py [--cohort c1] [--split train]
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
ART = Path(__file__).resolve().parent / "artifacts"

# Fields we keep while streaming (full_text is deliberately skipped: the
# proposal's unsupervised stage uses title+abstract text features).
KEEP = [
    "arxiv_id", "split", "year", "title", "abstract", "primary_category",
    "n_categories", "author_count", "reference_count", "venue", "venue_freq",
    "n_affiliations", "n_countries", "field",
    "title_len_words", "abstract_len_words",
    "target_citations", "y_log",
]

NUMERIC_META = [
    "author_count", "reference_count", "title_len_words",
    "abstract_len_words", "year", "n_categories", "n_affiliations",
    "n_countries",
]
LOG_META = ["author_count", "reference_count", "venue_freq"]  # heavy tails


def stream_records(path, split=None):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if split and r.get("split") != split:
                continue
            rows.append({k: r.get(k) for k in KEEP})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="c1")
    ap.add_argument("--split", default="train",
                    help="'train', 'test', or 'all'")
    ap.add_argument("--svd-dims", type=int, default=100)
    ap.add_argument("--data", default=None, help="override path to jsonl.gz")
    args = ap.parse_args()

    ART.mkdir(exist_ok=True)
    path = Path(args.data) if args.data else ROOT / "data" / f"{args.cohort}.jsonl.gz"
    split = None if args.split == "all" else args.split

    t0 = time.time()
    print(f"Streaming {path.name} (split={args.split}) ...", flush=True)
    df = stream_records(path, split)
    print(f"  {len(df):,} papers in {time.time()-t0:.0f}s", flush=True)

    # ---- text block: TF-IDF -> SVD --------------------------------------
    text = (df["title"].fillna("") + ". " + df["abstract"].fillna(""))
    tfidf = TfidfVectorizer(
        max_features=50_000, min_df=5, stop_words="english",
        sublinear_tf=True, strip_accents="unicode",
    )
    X_text = tfidf.fit_transform(text)
    print(f"  TF-IDF: {X_text.shape}", flush=True)

    svd = TruncatedSVD(n_components=args.svd_dims, random_state=42)
    X_svd = svd.fit_transform(X_text)
    print(f"  SVD: {X_svd.shape}, explained var "
          f"{svd.explained_variance_ratio_.sum():.3f}", flush=True)

    # ---- metadata block ---------------------------------------------------
    meta = df[NUMERIC_META].copy()
    meta["venue_freq"] = df["venue_freq"]
    for c in meta.columns:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta = meta.fillna(0.0)
    for c in LOG_META:
        meta[c] = np.log1p(meta[c])

    # one-hot of the three primary categories (cs.LG / cs.CL / cs.CV + other)
    cat = df["primary_category"].fillna("other")
    top_cats = ["cs.LG", "cs.CL", "cs.CV"]
    for tc in top_cats:
        meta[f"cat_{tc}"] = (cat == tc).astype(float)

    X_meta = meta.to_numpy(dtype=np.float64)

    # ---- assemble + standardize ------------------------------------------
    X = np.hstack([X_svd, X_meta])
    X = StandardScaler().fit_transform(X).astype(np.float32)
    feat_names = ([f"svd_{i}" for i in range(X_svd.shape[1])]
                  + list(meta.columns))
    print(f"  assembled X: {X.shape}", flush=True)

    np.savez_compressed(
        ART / "features.npz",
        X=X,
        y_log=df["y_log"].to_numpy(dtype=np.float32),
        target_citations=df["target_citations"].to_numpy(dtype=np.float32),
        year=df["year"].to_numpy(dtype=np.int32),
        primary_category=cat.to_numpy(dtype=object),
        arxiv_id=df["arxiv_id"].to_numpy(dtype=object),
        feat_names=np.array(feat_names, dtype=object),
        allow_pickle=True,
    )
    sparse.save_npz(ART / "tfidf.npz", X_text.tocsr())
    with open(ART / "vocab.json", "w") as fh:
        json.dump(tfidf.get_feature_names_out().tolist(), fh)
    df[["arxiv_id", "title", "venue", "field"]].to_csv(
        ART / "meta.csv.gz", index=False, compression="gzip")
    print(f"Done in {time.time()-t0:.0f}s -> {ART}", flush=True)


if __name__ == "__main__":
    main()
