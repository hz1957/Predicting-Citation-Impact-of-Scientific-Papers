"""Step 4 (optional, diagnostic) — images for the k-explorer in preview.html.

For each feature set (text+metadata, text-only) and each swept k, renders a
PCA + UMAP + t-SNE panel row colored by that k's K-Means labels, into
report/k_explorer_imgs/{space}_k{k}.png. Each feature set is shown in its
own embedding (computed by run_visualize.py / cached in artifacts/), which
does not depend on k — only the coloring changes with the slider.

Also writes report/k_explorer_meta.json (available k values + silhouette per
k) consumed by the slider UI that make_preview.sh appends to preview.html.

Diagnostic tool only: for k > 8 the colors extend beyond our validated
8-color palette (matplotlib tab20), so these images are NOT for the report.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import CATEGORICAL, INK, BASELINE, SURFACE, apply_style
from run_visualize import clip_axes, compute_embeddings

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TABLES = HERE / "outputs" / "tables"
IMGS = HERE / "report" / "k_explorer_imgs"

SPACES = {
    "text_meta": ("Text (LSA) + metadata features", "X_text_meta.npy"),
    "text": ("Text (LSA) features only", "X_lsa.npy"),
}


def colors_for(k):
    if k <= len(CATEGORICAL):
        return CATEGORICAL[:k]
    cmap = plt.colormaps["tab20"]
    return [cmap(i % 20) for i in range(k)]


def render(space, title, embs, tsne_idx, labels, k, sil):
    emb_pca, emb_umap, emb_tsne = embs
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.6))
    palette = colors_for(k)
    for ax, emb, lab, sub in ((axes[0], emb_pca, labels, "PCA"),
                              (axes[1], emb_umap, labels, "UMAP"),
                              (axes[2], emb_tsne, labels[tsne_idx],
                               "t-SNE")):
        for c in range(k):
            m = lab == c
            ax.scatter(emb[m, 0], emb[m, 1], s=1.4, c=[palette[c]],
                       alpha=0.35, linewidths=0, rasterized=True)
        for c in range(k):
            m = lab == c
            ax.text(np.median(emb[m, 0]), np.median(emb[m, 1]), f"C{c}",
                    fontsize=8.5, fontweight="bold", color=INK, ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE,
                              ec=BASELINE, lw=0.5, alpha=0.9))
        clip_axes(ax, emb)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(sub, loc="left", fontsize=10)
    fig.suptitle(f"{title} — k = {k}   (silhouette {sil:.3f})",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(IMGS / f"{space}_k{k}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    apply_style()
    IMGS.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(TABLES / "model_selection.csv")

    meta = {}
    for space, (title, xfile) in SPACES.items():
        X = np.load(ART / xfile)
        emb_pca, emb_umap, emb_tsne, tsne_idx = compute_embeddings(X, space)
        sweep = np.load(ART / f"labels_sweep_{space}.npz")
        sils = dict(zip(sel[sel["space"] == space]["k"],
                        sel[sel["space"] == space]["silhouette"]))
        ks = sorted(int(key[1:]) for key in sweep.files)
        meta[space] = {"title": title, "ks": ks,
                       "silhouette": {str(k): round(float(sils[k]), 4)
                                      for k in ks}}
        for k in ks:
            render(space, title, (emb_pca, emb_umap, emb_tsne), tsne_idx,
                   sweep[f"k{k}"], k, sils[k])
            print(f"{space} k={k} done", flush=True)

    with open(HERE / "report" / "k_explorer_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"Images in {IMGS}", flush=True)


if __name__ == "__main__":
    main()
