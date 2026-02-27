"""
2D UMAP projections of the feature space, colored by CKD outcome.

Matches manuscript Section A6: StandardScaler → UMAP with
n_components=2, min_dist=0.1, n_neighbors=15, random_state=1202.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from src.plot_style import get_outcome_colors, style_axis


def generate_umap_projection(features, labels, output_dir, experiment_name):
    """Generate a 2D UMAP scatter plot colored by CKD outcome.

    Parameters
    ----------
    features : np.ndarray, shape (n_patients, n_features)
    labels : np.ndarray, shape (n_patients,)  — 1=CKD positive, 0=non-CKD
    output_dir : str — directory to save the plot and embedding
    experiment_name : str — used in the plot title
    """
    print(f"\n[UMAP] Generating projection for {experiment_name}...")

    # Scale and handle NaN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # UMAP embedding (manuscript parameters)
    reducer = UMAP(
        n_components=2,
        min_dist=0.1,
        n_neighbors=15,
        random_state=config.RANDOM_SEED,
    )
    embedding = reducer.fit_transform(X_scaled)

    # Save embedding
    np.save(os.path.join(output_dir, "umap_embedding.npy"), embedding)

    # Plot
    colors = get_outcome_colors()
    labels_bool = labels.astype(bool)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Non-CKD first (background), then CKD on top
    ax.scatter(
        embedding[~labels_bool, 0], embedding[~labels_bool, 1],
        c=colors["negative"], label="Non-CKD", alpha=0.5, s=10, edgecolors="none",
    )
    ax.scatter(
        embedding[labels_bool, 0], embedding[labels_bool, 1],
        c=colors["positive"], label="CKD Stage 4-5", alpha=0.5, s=10, edgecolors="none",
    )

    style_axis(ax, title=f"UMAP Projection — {experiment_name}",
               xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "umap_projection.png"),
        dpi=config.PLOT_CONFIG["figure_dpi"],
    )
    plt.close(fig)

    n_pos = labels_bool.sum()
    n_neg = (~labels_bool).sum()
    print(f"[UMAP] Done. {n_pos} CKD+ (blue), {n_neg} non-CKD (red). Saved to {output_dir}")
