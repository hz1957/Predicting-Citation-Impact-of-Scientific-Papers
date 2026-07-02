"""Step 3 — figures and summary tables.

One package figure per clustering analysis (identical layout), one outcome
overlay, one model-selection figure. Filenames name the INPUTS; k and any
interpretation live in titles/captions only. Every figure is written twice:
  outputs/web/*.png    (200 dpi, for the GitHub Pages midterm report)
  outputs/paper/*.pdf  (vector, scatter layers rasterized, for Overleaf)

Figures
  fig_model_selection     silhouette vs k for both feature sets (two panels)
  fig_clusters_text_meta  analysis A (text + metadata, k chosen by
                          silhouette): PCA + UMAP + t-SNE colored by
                          cluster, median citations, top-decile share
  fig_clusters_text       analysis B (text only, k = 8): same layout
  fig_citation_map        PCA + UMAP + t-SNE colored by log 3-yr citations

Extra table
  outputs/tables/cluster_summary_text.csv   display name + sizes + citation
                                            stats + top terms per text-only
                                            cluster (report-ready)

Each analysis is embedded in its own feature space: fig_clusters_text_meta
shows PCA/UMAP/t-SNE of the text + metadata features, fig_clusters_text and
fig_citation_map show PCA/UMAP/t-SNE of the text (LSA) features. Cached in
artifacts/embeddings_{text,text_meta}.npz.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (CATEGORICAL, SEQ_CMAP, MUTED, INK, INK_2, BASELINE,
                        SURFACE, apply_style, save_fig)

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TABLES = HERE / "outputs" / "tables"
WEB = HERE / "outputs" / "web"
PAPER = HERE / "outputs" / "paper"
SEED = 42
TSNE_N = None  # cap for t-SNE; None = all points (fine at ~25k, subsample
               # if the corpus grows by an order of magnitude)

# Display names, assigned BY HAND from each cluster's top TF-IDF terms
# (outputs/tables/cluster_terms_*.json) and category shares — these describe
# observed cluster contents, the algorithm only produced unlabeled groups.
TEXT_CLUSTER_NAMES = {
    0: "3D & object detection",
    1: "reinforcement learning",
    2: "applied ML & prediction",
    3: "optimization & theory",
    4: "deep architectures",
    5: "GANs & adversarial",
    6: "image segmentation",
    7: "NLP & speech",
}
TEXT_META_CLUSTER_NAMES = {
    0: "cs.CV papers (99.9%)",
    1: "cs.LG & other categories",
    2: "cs.CL papers (100%)",
}


# --------------------------------------------------------------------------
def compute_embeddings(X, name="text"):
    """PCA (2D), UMAP, and t-SNE embeddings of feature matrix X (t-SNE
    optionally capped at TSNE_N points). Cached per feature set in
    artifacts/."""
    from sklearn.decomposition import PCA

    cache = ART / f"embeddings_{name}.npz"
    if cache.exists():
        with np.load(cache) as d:
            # materialize before any re-save (np.load is lazy)
            arrays = {key: d[key] for key in d.files}
        if "pca" not in arrays:
            # cache from before PCA was added: compute PCA, keep the rest
            arrays["pca"] = PCA(n_components=2,
                                random_state=SEED).fit_transform(X)
            np.savez_compressed(cache, **arrays)
        return (arrays["pca"], arrays["umap"], arrays["tsne"],
                arrays["tsne_idx"])

    emb_pca = PCA(n_components=2, random_state=SEED).fit_transform(X)

    import umap
    t0 = time.time()
    print(f"UMAP ({name}) ...", flush=True)
    emb_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED,
                         verbose=False).fit_transform(X)
    print(f"  {time.time()-t0:.0f}s", flush=True)

    from sklearn.manifold import TSNE
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(TSNE_N or len(X), len(X)),
                     replace=False)
    t0 = time.time()
    print(f"t-SNE ({name}) ...", flush=True)
    emb_tsne = TSNE(n_components=2, perplexity=30, init="pca",
                    random_state=SEED).fit_transform(X[idx])
    print(f"  {time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(cache, pca=emb_pca, umap=emb_umap, tsne=emb_tsne,
                        tsne_idx=idx)
    return emb_pca, emb_umap, emb_tsne, idx


def cluster_handles(k, names):
    return [plt.Line2D([], [], marker="o", ls="", ms=7, color=CATEGORICAL[c],
                       label=f"C{c} · {names[c]}") for c in range(k)]


def clip_axes(ax, emb, q=0.2):
    """Limit axes to the central quantile range so a handful of extreme
    outliers (mostly in the PCA of the metadata-bearing space) cannot
    squeeze the point mass into a corner."""
    for setter, vals in ((ax.set_xlim, emb[:, 0]), (ax.set_ylim,
                                                    emb[:, 1])):
        lo, hi = np.percentile(vals, [q, 100 - q])
        pad = 0.04 * (hi - lo)
        setter(lo - pad, hi + pad)


def scatter_clusters(ax, emb, labels, k, dot=2.0):
    for c in range(k):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=dot, c=CATEGORICAL[c],
                   alpha=0.35, linewidths=0, rasterized=True)
    # direct labels at cluster medians (secondary encoding for CVD/contrast)
    for c in range(k):
        m = labels == c
        cx, cy = np.median(emb[m, 0]), np.median(emb[m, 1])
        ax.text(cx, cy, f"C{c}", fontsize=10, fontweight="bold", color=INK,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE,
                          ec=BASELINE, lw=0.6, alpha=0.9))
    clip_axes(ax, emb)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def scatter_citations(ax, emb, y_log, dot=2.0, vmax=None):
    order = np.argsort(y_log)  # draw high-citation points on top
    sc = ax.scatter(emb[order, 0], emb[order, 1], s=dot, c=y_log[order],
                    cmap=SEQ_CMAP, alpha=0.55, linewidths=0, rasterized=True,
                    vmin=0, vmax=vmax)
    clip_axes(ax, emb)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return sc


def outcome_bars(ax_med, ax_top, prof, base_rate_x,
                 letters=("d", "e")):
    xt = [f"C{c}" for c in prof["cluster"]]
    colors = [CATEGORICAL[c] for c in prof["cluster"]]

    bars = ax_med.bar(xt, prof["median_citations_3yr"], color=colors,
                      width=0.6)
    for b, v in zip(bars, prof["median_citations_3yr"]):
        ax_med.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:g}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=INK)
    ax_med.set_ylabel("median citations (3 yrs)")
    ax_med.set_title(f"({letters[0]}) Typical impact", loc="left",
                     fontsize=10)
    ax_med.grid(axis="x", visible=False)

    bars = ax_top.bar(xt, 100 * prof["share_top_decile"], color=colors,
                      width=0.6)
    for b, v in zip(bars, 100 * prof["share_top_decile"]):
        ax_top.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=INK)
    ax_top.axhline(10, color=MUTED, lw=0.9)
    ax_top.text(base_rate_x, 10.5, "base rate 10%", ha="center", fontsize=8,
                color=MUTED)
    ax_top.set_ylabel("papers in top decile (%)")
    ax_top.set_title(f"({letters[1]}) Highly cited share", loc="left",
                     fontsize=10)
    ax_top.grid(axis="x", visible=False)


def cluster_package_fig(name, suptitle, embs, tsne_idx,
                        labels, k, names, prof, legend_ncol, base_rate_x,
                        legend_y):
    """The shared layout: PCA + UMAP + t-SNE by cluster, outcome bars below.
    embs = (emb_pca, emb_umap, emb_tsne)."""
    emb_pca, emb_umap, emb_tsne = embs
    fig = plt.figure(figsize=(10.6, 6.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1.0],
                          hspace=0.28, wspace=0.9)
    panels = ((fig.add_subplot(gs[0, 0:2]), emb_pca, labels, "(a) PCA"),
              (fig.add_subplot(gs[0, 2:4]), emb_umap, labels, "(b) UMAP"),
              (fig.add_subplot(gs[0, 4:6]), emb_tsne, labels[tsne_idx],
               "(c) t-SNE"))
    for ax, emb, lab, title in panels:
        scatter_clusters(ax, emb, lab, k, dot=1.4)
        ax.set_title(title, loc="left", fontsize=10)
    outcome_bars(fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6]),
                 prof, base_rate_x)
    fig.legend(handles=cluster_handles(k, names), loc="lower center",
               ncol=legend_ncol, fontsize=8.5,
               bbox_to_anchor=(0.5, legend_y))
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="bold")
    save_fig(fig, name, WEB, PAPER)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    apply_style()
    WEB.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    d = np.load(ART / "features.npz", allow_pickle=True)
    y_log = d["y_log"]
    X_lsa = np.load(ART / "X_lsa.npy")
    ks = json.load(open(ART / "chosen_k.json"))
    k_tm, k_t = ks["text_meta"], ks["text"]
    lab_tm = np.load(ART / "labels_text_meta.npy")
    lab_t = np.load(ART / "labels_text.npy")
    sel = pd.read_csv(TABLES / "model_selection.csv")
    prof_tm = pd.read_csv(TABLES / "cluster_profiles_text_meta.csv")
    prof_t = pd.read_csv(TABLES / "cluster_profiles_text.csv")
    terms_t = json.load(open(TABLES / "cluster_terms_text.json"))

    tm_names = {c: TEXT_META_CLUSTER_NAMES.get(c, f"cluster {c}")
                for c in range(k_tm)}
    t_names = {c: TEXT_CLUSTER_NAMES.get(c, " / ".join(terms_t[str(c)][:2]))
               for c in range(k_t)}

    # each analysis is embedded in ITS OWN feature space; the citation map
    # uses the text embeddings (it accompanies the text-only clustering)
    emb_pca, emb_umap, emb_tsne, tsne_idx = compute_embeddings(X_lsa, "text")
    X_tm = np.load(ART / "X_text_meta.npy")
    emb_pca_tm, emb_umap_tm, emb_tsne_tm, tsne_idx_tm = compute_embeddings(
        X_tm, "text_meta")
    vmax = float(np.quantile(y_log, 0.99))

    # ---- 1. model selection ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for ax, tag, title, chosen in (
            (axes[0], "text_meta", "(a) Text (LSA) + metadata features",
             k_tm),
            (axes[1], "text", "(b) Same text (LSA) features, no metadata",
             k_t)):
        s = sel[sel["space"] == tag]
        ax.plot(s["k"], s["silhouette"], color=CATEGORICAL[0], lw=2,
                marker="o", ms=4.5, mfc=CATEGORICAL[0], mec=SURFACE, mew=1)
        sv = float(s.loc[s["k"] == chosen, "silhouette"].iloc[0])
        ax.scatter([chosen], [sv], s=90, zorder=3, facecolor="none",
                   edgecolor=INK, linewidths=1.2)
        ax.annotate(f"k = {chosen}", (chosen, sv),
                    textcoords="offset points", xytext=(8, -12),
                    fontsize=9, fontweight="bold", color=INK)
        ax.set_xlabel("number of clusters k")
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel("silhouette score")
    fig.suptitle("K-Means model selection (silhouette, 10k subsample)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_fig(fig, "fig_model_selection", WEB, PAPER)
    plt.close(fig)

    # ---- 2/3. one package figure per analysis -------------------------------
    cluster_package_fig(
        "fig_clusters_text_meta",
        f"K-Means (k={k_tm}) on text (LSA) + metadata features",
        (emb_pca_tm, emb_umap_tm, emb_tsne_tm), tsne_idx_tm, lab_tm, k_tm,
        tm_names, prof_tm, legend_ncol=3, base_rate_x=0.5, legend_y=0.0)

    cluster_package_fig(
        "fig_clusters_text",
        f"K-Means (k={k_t}) on text (LSA) features only",
        (emb_pca, emb_umap, emb_tsne), tsne_idx, lab_t, k_t, t_names,
        prof_t, legend_ncol=4, base_rate_x=2.5, legend_y=-0.01)

    # ---- 4. citation overlay -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    scatter_citations(axes[0], emb_pca, y_log, dot=1.4, vmax=vmax)
    axes[0].set_title("(a) PCA", loc="left", fontsize=10)
    scatter_citations(axes[1], emb_umap, y_log, dot=1.4, vmax=vmax)
    axes[1].set_title("(b) UMAP", loc="left", fontsize=10)
    sc = scatter_citations(axes[2], emb_tsne, y_log[tsne_idx], dot=1.4,
                           vmax=vmax)
    axes[2].set_title("(c) t-SNE", loc="left", fontsize=10)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, extend="max")
    cb.set_label("log(1 + citations in first 3 yrs)", fontsize=8.5,
                 color=INK_2)
    cb.ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED)
    cb.outline.set_edgecolor(BASELINE)
    cb.outline.set_linewidth(0.6)
    fig.suptitle("Citation impact over the text (LSA) embeddings",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    save_fig(fig, "fig_citation_map", WEB, PAPER)
    plt.close(fig)

    # ---- 5. report-ready summary table --------------------------------------
    summary = prof_t.copy()
    summary.insert(1, "name", [t_names[c] for c in summary["cluster"]])
    summary["top_terms"] = [", ".join(terms_t[str(c)][:8])
                            for c in summary["cluster"]]
    summary.round(3).to_csv(TABLES / "cluster_summary_text.csv", index=False)

    print(f"Figures written to {WEB} and {PAPER}", flush=True)


if __name__ == "__main__":
    main()
