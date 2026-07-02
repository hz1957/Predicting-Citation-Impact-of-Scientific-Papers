"""Step 2 — two K-Means clusterings with silhouette-guided model selection.

Both analyses share the identical L2-normalized LSA text block
(TF-IDF -> TruncatedSVD, natural component scale); they differ only in
whether the metadata block is concatenated:

  A. **text_meta** (proposal Methods, "assembled feature vectors"): LSA text
     block + standardized metadata block, the metadata block scaled so both
     blocks carry a comparable row norm. The silhouette score selects k here
     (it has a clear interior maximum).

  B. **text**: the same LSA block alone, at a finer k, run independently on
     the full corpus (not nested inside analysis A's clusters). Here the
     silhouette rises only gradually and monotonically with k (typical for
     text), so k is set to TEXT_K = 8 — the largest k at which every cluster
     remains interpretable and well populated (and the limit of our 8-color
     categorical palette).

Outputs (unsupervised/artifacts/ + outputs/tables/), tags name the INPUTS:
  X_lsa.npy                        LSA text features (for UMAP / t-SNE)
  X_text_meta.npy                  text + metadata features (for UMAP/t-SNE)
  labels_text_meta.npy             analysis-A labels (k chosen by silhouette)
  labels_text.npy                  analysis-B labels (k = TEXT_K)
  labels_sweep_{text_meta,text}.npz  labels for EVERY swept k (k explorer)
  model_selection.csv              k, silhouette, DB, inertia for both
  cluster_profiles_{text_meta,text}.csv
  cluster_meta_zscores_text.csv
  cluster_category_shares_{text_meta,text}.csv
  cluster_terms_{text_meta,text}.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import Normalizer

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TABLES = HERE / "outputs" / "tables"

SEED = 42
K_RANGE_A = range(2, 13)
K_RANGE_B = range(2, 17)
TEXT_K = 8


def sweep(X, k_range, tag):
    rows, labels = [], {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        lab = km.fit_predict(X)
        sil = silhouette_score(X, lab, sample_size=10_000, random_state=SEED)
        db = davies_bouldin_score(X, lab)
        rows.append({"space": tag, "k": k, "silhouette": sil,
                     "davies_bouldin": db, "inertia": km.inertia_})
        labels[k] = lab
        print(f"  [{tag}] k={k:2d}  silhouette={sil:.4f}  DB={db:.3f}",
              flush=True)
    return pd.DataFrame(rows), labels


def profiles(labels, k, cites, y_log, top_cut, tag):
    rows = []
    for c in range(k):
        m = labels == c
        rows.append({
            "cluster": c,
            "n_papers": int(m.sum()),
            "share_of_corpus": float(m.mean()),
            "median_citations_3yr": float(np.median(cites[m])),
            "mean_citations_3yr": float(cites[m].mean()),
            "mean_log_citations": float(y_log[m].mean()),
            "share_uncited": float((cites[m] == 0).mean()),
            "share_top_decile": float((cites[m] >= top_cut).mean()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"cluster_profiles_{tag}.csv", index=False)
    return df


def top_terms(X_text, vocab, labels, k, tag, n=15):
    terms = {}
    for c in range(k):
        mean_tfidf = np.asarray(X_text[labels == c].mean(axis=0)).ravel()
        terms[str(c)] = vocab[np.argsort(mean_tfidf)[::-1][:n]].tolist()
    with open(TABLES / f"cluster_terms_{tag}.json", "w") as fh:
        json.dump(terms, fh, indent=2)
    return terms


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    d = np.load(ART / "features.npz", allow_pickle=True)
    feat_names = list(d["feat_names"])
    n_svd = sum(1 for f in feat_names if f.startswith("svd_"))
    meta_cols = feat_names[n_svd:]
    cites = d["target_citations"]
    y_log = d["y_log"]
    cat = pd.Series(d["primary_category"])
    top_cut = np.quantile(cites, 0.9)

    # ---- build the two spaces --------------------------------------------
    X_text = sparse.load_npz(ART / "tfidf.npz")
    vocab = np.array(json.load(open(ART / "vocab.json")))
    svd = TruncatedSVD(n_components=n_svd, random_state=SEED)
    X_lsa = Normalizer(copy=False).fit_transform(
        svd.fit_transform(X_text)).astype(np.float32)
    np.save(ART / "X_lsa.npy", X_lsa)

    meta_block = d["X"][:, n_svd:]  # standardized metadata from step 1
    # scale so the metadata block's average row norm matches the (unit) text
    # block norm: w * sqrt(n_meta_dims) ~= 1
    w = 1.0 / np.sqrt(meta_block.shape[1])
    X_asm = np.hstack([X_lsa, (meta_block * w).astype(np.float32)])
    np.save(ART / "X_text_meta.npy", X_asm)
    print(f"text {X_lsa.shape} · text+metadata {X_asm.shape}", flush=True)

    # ---- sweeps ------------------------------------------------------------
    sel_a, labels_a = sweep(X_asm, K_RANGE_A, "text_meta")
    sel_b, labels_b = sweep(X_lsa, K_RANGE_B, "text")
    pd.concat([sel_a, sel_b]).to_csv(TABLES / "model_selection.csv",
                                     index=False)

    best_k = int(sel_a.loc[sel_a["silhouette"].idxmax(), "k"])
    print(f"text+metadata: silhouette-optimal k = {best_k}", flush=True)
    lab_asm = labels_a[best_k]
    lab_arch = labels_b[TEXT_K]
    np.save(ART / "labels_text_meta.npy", lab_asm)
    np.save(ART / "labels_text.npy", lab_arch)
    np.savez_compressed(ART / "labels_sweep_text_meta.npz",
                        **{f"k{k}": lab for k, lab in labels_a.items()})
    np.savez_compressed(ART / "labels_sweep_text.npz",
                        **{f"k{k}": lab for k, lab in labels_b.items()})
    with open(ART / "chosen_k.json", "w") as fh:
        json.dump({"text_meta": best_k, "text": TEXT_K}, fh)

    # ---- profiles, category composition, terms -----------------------------
    for tag, lab, k in [("text_meta", lab_asm, best_k),
                        ("text", lab_arch, TEXT_K)]:
        prof = profiles(lab, k, cites, y_log, top_cut, tag)
        comp = pd.crosstab(lab, cat, normalize="index").round(3)
        comp.to_csv(TABLES / f"cluster_category_shares_{tag}.csv")
        terms = top_terms(X_text, vocab, lab, k, tag)
        print(f"\n== {tag} (k={k}) ==", flush=True)
        print(prof.round(3).to_string(index=False), flush=True)
        for c in range(k):
            print(f"  C{c}: {', '.join(terms[str(c)][:8])}", flush=True)

    # standardized-metadata means per text-only cluster
    zprof = pd.DataFrame(
        [meta_block[lab_arch == c].mean(axis=0) for c in range(TEXT_K)],
        columns=meta_cols)
    zprof.insert(0, "cluster", range(TEXT_K))
    zprof.round(3).to_csv(TABLES / "cluster_meta_zscores_text.csv",
                          index=False)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
