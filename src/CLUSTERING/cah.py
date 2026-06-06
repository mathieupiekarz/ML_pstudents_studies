"""
cah.py — Classification Ascendante Hiérarchique (CAH).

Méthodes exposées
-----------------
run_cah(X, k, linkage_method)      → labels (np.ndarray), Z (linkage matrix)
plot_dendrogram(Z, k, out_path)    → dendrogramme avec coupure à k clusters
compare_linkages(X, out_path)      → 3 dendrogrammes + tableau cophénétique
cophenetic_quality(X, Z)           → coefficient cophénétique (float)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import (
    cophenet,
    dendrogram,
    fcluster,
    linkage,
)
from scipy.spatial.distance import pdist

LINKAGE_METHODS = ["ward", "complete", "average"]
LINKAGE_LABELS = {
    "ward": "Ward (D²)",
    "complete": "Lien complet",
    "average": "Lien moyen (UPGMA)",
}


# ── Calcul CAH ───────────────────────────────────────────────────────────────

def run_cah(
    X: np.ndarray,
    k: int,
    linkage_method: str = "ward",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lance une CAH et découpe en k clusters.

    Retourne
    --------
    labels : np.ndarray (n,) — numéros de clusters 1..k
    Z      : np.ndarray      — matrice de liaisons scipy
    """
    metric = "euclidean" if linkage_method == "ward" else "euclidean"
    Z = linkage(X, method=linkage_method, metric=metric)
    labels = fcluster(Z, k, criterion="maxclust")
    return labels, Z


def cophenetic_quality(X: np.ndarray, Z: np.ndarray) -> float:
    """Coefficient cophénétique (corrélation entre distances originales et ultramétriques)."""
    c, _ = cophenet(Z, pdist(X))
    return float(c)


# ── Dendrogramme ─────────────────────────────────────────────────────────────

def plot_dendrogram(
    Z: np.ndarray,
    k: int,
    out_path: Path,
    title: str = "CAH — Dendrogramme (Ward)",
    source_label: str = "",
) -> None:
    """
    Trace le dendrogramme avec la coupure horizontale correspondant à k clusters.
    Affiche uniquement les dernières fusions (p=30) pour lisibilité.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ddata = dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        no_labels=False,
    )

    # Ligne de coupure : hauteur entre la (k)-ème et (k+1)-ème fusion depuis le bas
    heights = sorted(Z[:, 2])
    if k <= len(heights):
        cut_height = (heights[-k] + heights[-(k + 1)]) / 2 if k < len(heights) else heights[-k] * 0.95
        ax.axhline(y=cut_height, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Coupure → {k} clusters")
        ax.legend(fontsize=9)

    ax.set_title(f"{title}{' — ' + source_label if source_label else ''}", fontsize=12)
    ax.set_xlabel("Individus (groupes contractés)", fontsize=10)
    ax.set_ylabel("Distance de fusion", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Comparaison des critères d'agrégation ────────────────────────────────────

def compare_linkages(
    X: np.ndarray,
    out_path: Path,
    source_label: str = "",
) -> dict[str, float]:
    """
    Trace 3 dendrogrammes (Ward, complet, moyen) en sous-graphes et affiche
    le coefficient cophénétique de chacun.

    Retourne un dict {method: coeff_coph}.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    coeff_dict: dict[str, float] = {}

    for ax, method in zip(axes, LINKAGE_METHODS):
        Z = linkage(X, method=method, metric="euclidean")
        c = cophenetic_quality(X, Z)
        coeff_dict[method] = c

        dendrogram(
            Z,
            ax=ax,
            truncate_mode="lastp",
            p=20,
            leaf_rotation=90,
            leaf_font_size=8,
            show_contracted=True,
            no_labels=True,
        )
        ax.set_title(
            f"{LINKAGE_LABELS[method]}\ncoph. = {c:.3f}",
            fontsize=11,
        )
        ax.set_xlabel("Individus", fontsize=9)
        ax.set_ylabel("Distance de fusion", fontsize=9)

    suptitle = "Comparaison des critères d'agrégation (Lance-Williams)"
    if source_label:
        suptitle += f" — {source_label}"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return coeff_dict
