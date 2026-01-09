"""
Generate a static grid showing diffusion strength effects.

Creates a 2x4 panel grid:
- Top row: colored by clusterid
- Bottom row: colored by pseudotime
- Columns: alpha ∈ {0.0, 0.2, 0.4, 0.6}

Outputs: alpha_grid.png in the examples directory.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch

import pgdiffusion as pgd


def _get_cluster_colors(adata, cluster_col: str):
    cats = adata.obs[cluster_col].astype("category")
    adata.obs[cluster_col] = cats
    palette = sc.plotting.palettes.default_20
    # Extend palette if needed
    while len(palette) < len(cats.cat.categories):
        palette = palette + palette
    color_map = {cat: palette[i] for i, cat in enumerate(cats.cat.categories)}
    colors = cats.map(color_map)
    return colors


def main():
    data_dir = Path(__file__).parent / "data"
    adata = sc.read_h5ad(data_dir / "adata_minimal.h5ad")
    with open(data_dir / "trajectories.json") as f:
        trajectories = json.load(f)

    X = torch.tensor(adata.obsm["X_pca"], dtype=torch.float32)
    edge_index = pgd.build_graph(adata, trajectories, neighbors_per_side=50)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    alphas = [0.0, 0.2, 0.4, 0.6]
    cluster_col = "clusterid" if "clusterid" in adata.obs.columns else None
    pseudotime_col = "pseudotime" if "pseudotime" in adata.obs.columns else None

    fig, axes = plt.subplots(2, len(alphas), figsize=(16, 7))

    for col_idx, alpha in enumerate(alphas):
        # Diffuse features
        X_diffused = pgd.diffuse(
            X,
            edge_index,
            alpha=float(alpha),
            n_steps=1,
            add_self_loops=True,
        )

        # UMAP on diffused features
        adata_tmp = adata.copy()
        adata_tmp.obsm["X_pseudotime"] = X_diffused.cpu().numpy()
        sc.pp.neighbors(adata_tmp, use_rep="X_pseudotime", n_neighbors=20)
        sc.tl.umap(adata_tmp, random_state=42, min_dist=0.3)
        coords = adata_tmp.obsm["X_umap"]
        x, y = coords[:, 0], coords[:, 1]

        # Top row: clusterid
        ax_top = axes[0, col_idx]
        if cluster_col:
            colors = _get_cluster_colors(adata_tmp, cluster_col)
            ax_top.scatter(x, y, c=colors, s=8, alpha=0.8, linewidths=0)
        else:
            ax_top.scatter(x, y, s=8, alpha=0.8, linewidths=0, c="steelblue")
        ax_top.set_title(f"alpha = {alpha:.1f}", fontsize=12, fontweight="bold")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_xlabel("")
        ax_top.set_ylabel("clusterid" if col_idx == 0 else "")
        ax_top.grid(False)

        # Bottom row: pseudotime
        ax_bot = axes[1, col_idx]
        if pseudotime_col:
            pts = adata_tmp.obs[pseudotime_col].to_numpy()
            sc_plot = ax_bot.scatter(
                x,
                y,
                c=pts,
                cmap="viridis",
                s=8,
                alpha=0.8,
                linewidths=0,
            )
        else:
            sc_plot = ax_bot.scatter(x, y, s=8, alpha=0.8, linewidths=0, c="steelblue")
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])
        ax_bot.set_xlabel("" if col_idx else "UMAP 1")
        ax_bot.set_ylabel("pseudotime" if col_idx == 0 else "")
        ax_bot.grid(False)

    # Add shared colorbar for pseudotime if present
    if pseudotime_col:
        cbar = fig.colorbar(
            sc_plot,
            ax=axes[1, :].tolist(),
            orientation="horizontal",
            fraction=0.05,
            pad=0.08,
        )
        cbar.set_label("pseudotime")

    plt.tight_layout()
    out_path = Path(__file__).parent / "alpha_grid.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
