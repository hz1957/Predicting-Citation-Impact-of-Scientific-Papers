"""Build train/test feature matrices for supervised baselines.

The key rule is that every learned preprocessing step is fit on the c1 train
split only, then reused unchanged on the c1 test split. This avoids leakage
from the held-out 2021-2022 papers.

Outputs are written to supervised/artifacts/.
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"

KEEP = [
    "arxiv_id",
    "split",
    "year",
    "title",
    "abstract",
    "primary_category",
    "n_categories",
    "author_count",
    "reference_count",
    "venue_freq",
    "n_affiliations",
    "n_countries",
    "title_len_words",
    "abstract_len_words",
    "target_citations",
    "y_log",
]

NUMERIC_META = [
    "author_count",
    "reference_count",
    "venue_freq",
    "n_affiliations",
    "n_countries",
    "year",
    "n_categories",
    "title_len_words",
    "abstract_len_words",
]

LOG_META = ["author_count", "reference_count", "venue_freq"]
CAT_META = ["primary_category"]
SEED = 42


def read_records(path):
    """Read either the full gzipped JSONL data or the small JSON samples."""
    path = Path(path)
    rows = []
    if path.suffix == ".gz":
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
        with opener(path) as fh:
            for line in fh:
                record = json.loads(line)
                rows.append({k: record.get(k) for k in KEEP})
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                rows.append({k: record.get(k) for k in KEEP})
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            for record in json.load(fh):
                rows.append({k: record.get(k) for k in KEEP})
    else:
        raise ValueError(f"Unsupported data format: {path}")
    return pd.DataFrame(rows)


def make_text(df):
    return df["title"].fillna("") + ". " + df["abstract"].fillna("")


def make_numeric_meta(df):
    meta = df[NUMERIC_META].copy()
    for col in NUMERIC_META:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.fillna(0.0)
    for col in LOG_META:
        meta[col] = np.log1p(meta[col].clip(lower=0))
    return meta


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_features(train, test, args):
    min_df = min(args.min_df, max(1, len(train) // 2))
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        min_df=min_df,
        stop_words="english",
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_tfidf = tfidf.fit_transform(make_text(train))
    test_tfidf = tfidf.transform(make_text(test))
    print(f"  TF-IDF train/test: {train_tfidf.shape} / {test_tfidf.shape}", flush=True)

    max_svd = min(train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1)
    if max_svd < 1:
        raise ValueError("Not enough train rows/terms to fit TruncatedSVD.")
    n_svd = min(args.svd_dims, max_svd)
    if n_svd != args.svd_dims:
        print(f"  reducing SVD dims from {args.svd_dims} to {n_svd}", flush=True)

    print(
        f"  fitting TruncatedSVD: dims={n_svd}, algorithm={args.svd_algorithm}",
        flush=True,
    )
    svd_kwargs = {
        "n_components": n_svd,
        "algorithm": args.svd_algorithm,
        "random_state": SEED,
    }
    if args.svd_algorithm == "randomized":
        svd_kwargs["power_iteration_normalizer"] = "none"
    svd = TruncatedSVD(**svd_kwargs)
    normalizer = Normalizer(copy=False)
    train_lsa = normalizer.fit_transform(svd.fit_transform(train_tfidf))
    test_lsa = normalizer.transform(svd.transform(test_tfidf))
    print(
        f"  LSA train/test: {train_lsa.shape} / {test_lsa.shape}, "
        f"explained var={svd.explained_variance_ratio_.sum():.3f}"
    )

    train_num = make_numeric_meta(train)
    test_num = make_numeric_meta(test)
    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train_num)
    test_num_scaled = scaler.transform(test_num)

    encoder = one_hot_encoder()
    train_cat = train[CAT_META].fillna("Unknown")
    test_cat = test[CAT_META].fillna("Unknown")
    train_cat_enc = encoder.fit_transform(train_cat)
    test_cat_enc = encoder.transform(test_cat)

    cat_names = encoder.get_feature_names_out(CAT_META).tolist()
    feature_names = (
        [f"lsa_{i}" for i in range(n_svd)]
        + NUMERIC_META
        + cat_names
    )

    train_x = sparse.csr_matrix(
        np.hstack([train_lsa, train_num_scaled, train_cat_enc]).astype(np.float32)
    )
    test_x = sparse.csr_matrix(
        np.hstack([test_lsa, test_num_scaled, test_cat_enc]).astype(np.float32)
    )
    transformers = {
        "tfidf": tfidf,
        "svd": svd,
        "normalizer": normalizer,
        "scaler": scaler,
        "one_hot_encoder": encoder,
        "numeric_meta": NUMERIC_META,
        "categorical_meta": CAT_META,
        "feature_names": feature_names,
    }
    return train_x, test_x, transformers


def write_meta(df, out_path):
    cols = [
        "arxiv_id",
        "year",
        "primary_category",
        "target_citations",
        "y_log",
    ]
    df[cols].to_csv(out_path, index=False, compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="c1")
    parser.add_argument("--data", default=None, help="Override path to data file")
    parser.add_argument("--svd-dims", type=int, default=100)
    parser.add_argument(
        "--svd-algorithm",
        choices=["randomized", "arpack"],
        default="randomized",
        help="Keep randomized for parity with unsupervised; use arpack only as a Windows DLL fallback.",
    )
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=5)
    args = parser.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    path = Path(args.data) if args.data else ROOT / "data" / f"{args.cohort}.jsonl.gz"

    t0 = time.time()
    print(f"Reading {path} ...", flush=True)
    df = read_records(path)
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError(
            "Expected c1-style split labels with non-empty train and test rows."
        )
    print(f"  train/test rows: {len(train):,} / {len(test):,}", flush=True)

    train_x, test_x, transformers = build_features(train, test, args)
    sparse.save_npz(ART / "X_train.npz", train_x)
    sparse.save_npz(ART / "X_test.npz", test_x)
    np.save(ART / "y_train.npy", train["y_log"].to_numpy(dtype=np.float32))
    np.save(ART / "y_test.npy", test["y_log"].to_numpy(dtype=np.float32))
    np.save(
        ART / "target_citations_train.npy",
        train["target_citations"].to_numpy(dtype=np.float32),
    )
    np.save(
        ART / "target_citations_test.npy",
        test["target_citations"].to_numpy(dtype=np.float32),
    )
    write_meta(train, ART / "train_meta.csv.gz")
    write_meta(test, ART / "test_meta.csv.gz")
    joblib.dump(transformers, ART / "transformers.joblib")
    with open(ART / "feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(transformers["feature_names"], fh, indent=2)
    with open(ART / "split_info.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "cohort": args.cohort,
                "data": str(path),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_features": int(train_x.shape[1]),
                "svd_dims": int(sum(n.startswith("lsa_") for n in transformers["feature_names"])),
            },
            fh,
            indent=2,
        )
    print(
        f"Done in {time.time() - t0:.0f}s -> {ART} "
        f"({train_x.shape[1]} features)",
        flush=True,
    )


if __name__ == "__main__":
    main()
