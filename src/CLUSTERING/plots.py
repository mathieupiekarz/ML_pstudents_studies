"""
plots.py — Visualisations communes du module clustering.

Méthodes exposées
-----------------
plot_cluster_scatter(X, labels, out_path, ...)
    Scatter 2D (Dim1 vs Dim2) coloré par étiquettes de cluster.
    Si X a plus de 2 dimensions, seules les 2 premières sont projetées.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Palette compatible daltoniens (tab10 est robuste jusqu'à 10 clusters)
_CMAP = "tab10"


def plot_cluster_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str = "Carte des clusters",
    method_name: str = "",
    source_label: str = "",
    dim_names: tuple[str, str] = ("Dim 1", "Dim 2"),
) -> None:
    """
    Scatter plot 2D coloré par étiquette de cluster.

    Paramètres
    ----------
    X          : coordonnées (n, ≥2) — seules les 2 premières colonnes sont utilisées
    labels     : entiers 0..k-1 (k-means) ou 1..k (CAH) — normalisés en 0..k-1
    out_path   : chemin du fichier PNG
    title      : titre principal
    method_name: sous-titre (ex. "CAH Ward k=4")
    source_label: identifiant de la réduction (ex. "ACP — no_G1_G2_G3_avg")
    dim_names  : noms des deux axes
    """
    x = X[:, 0]
    y = X[:, 1]

    # Normaliser les labels en 0..k-1 pour la colormap
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels_norm = np.array([remap[l] for l in labels])
    k = len(unique)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        x, y,
        c=labels_norm,
        cmap=_CMAP,
        s=25,
        alpha=0.75,
        linewidths=0.3,
        edgecolors="white",
        vmin=0,
        vmax=max(k - 1, 1),
    )

    # Centroïdes
    for c in range(k):
        mask = labels_norm == c
        cx, cy = x[mask].mean(), y[mask].mean()
        ax.scatter(cx, cy, marker="X", s=120, c="black", zorder=5, linewidths=0)
        ax.annotate(
            f"C{unique[c]}",
            (cx, cy),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            fontweight="bold",
        )

    cbar = fig.colorbar(scatter, ax=ax, ticks=range(k))
    cbar.set_label("Cluster", fontsize=10)
    cbar.set_ticklabels([str(u) for u in unique])

    full_title = title
    if method_name:
        full_title += f"\n{method_name}"
    if source_label:
        full_title += f"  [{source_label}]"
    ax.set_title(full_title, fontsize=12)
    ax.set_xlabel(dim_names[0], fontsize=11)
    ax.set_ylabel(dim_names[1], fontsize=11)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
