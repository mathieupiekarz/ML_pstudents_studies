"""
kmeans.py — K-means et méthode des nuées dynamiques.

Méthodes exposées
-----------------
run_kmeans(X, k, ...)              → labels (np.ndarray)
run_nuees_dynamiques(X, k, Z)     → labels — k-means initialisé sur centroïdes CAH
plot_elbow_silhouette(X, k_range) → graphe double-axe inertie / silhouette
select_k_silhouette(X, k_range)   → k optimal selon score silhouette
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ── K-means standard ─────────────────────────────────────────────────────────

def run_kmeans(
    X: np.ndarray,
    k: int,
    init: str = "k-means++",
    n_init: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """K-means classique. Retourne labels 0..k-1."""
    km = KMeans(k, init=init, n_init=n_init, random_state=random_state)
    return km.fit_predict(X)


# ── Nuées dynamiques (k-means initialisé sur centroïdes CAH) ─────────────────

def run_nuees_dynamiques(
    X: np.ndarray,
    k: int,
    Z: np.ndarray,
    n_init: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """
    Méthode des nuées dynamiques :
    1. Découpe la CAH (matrice Z) en k classes.
    2. Calcule les centroïdes de ces classes.
    3. Utilise ces centroïdes comme initialisation du k-means.

    Le k-means affine ainsi la partition CAH sans dépendre d'une initialisation aléatoire.
    """
    cah_labels = fcluster(Z, k, criterion="maxclust")  # 1..k
    # Centroïdes des k classes CAH
    centroids = np.array([
        X[cah_labels == c].mean(axis=0) for c in range(1, k + 1)
    ])
    km = KMeans(k, init=centroids, n_init=n_init, random_state=random_state)
    return km.fit_predict(X)


# ── Sélection automatique de k ───────────────────────────────────────────────

def select_k_silhouette(X: np.ndarray, k_range: range | list[int]) -> int:
    """Retourne le k maximisant le score silhouette moyen."""
    scores = {k: silhouette_score(X, run_kmeans(X, k)) for k in k_range}
    return max(scores, key=scores.__getitem__)


# ── Graphe elbow + silhouette ─────────────────────────────────────────────────

def plot_elbow_silhouette(
    X: np.ndarray,
    k_range: range | list[int],
    out_path: Path,
    source_label: str = "",
) -> None:
    """
    Double axe :
    - gauche  : inertie intra-classe (courbe en coude)
    - droite  : score silhouette moyen (plus élevé = meilleur)

    Le k optimal (silhouette) est mis en évidence.
    """
    ks = list(k_range)
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in ks:
        km = KMeans(k, init="k-means++", n_init=20, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels) if k > 1 else float("nan"))

    best_k = ks[int(np.nanargmax(silhouettes))]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color_inertia = "steelblue"
    color_sil = "darkorange"

    ax1.plot(ks, inertias, "o-", color=color_inertia, label="Inertie intra-classe")
    ax1.set_xlabel("Nombre de clusters k", fontsize=11)
    ax1.set_ylabel("Inertie", color=color_inertia, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_inertia)

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, "s--", color=color_sil, label="Silhouette moyen")
    ax2.axvline(best_k, color="crimson", linestyle=":", linewidth=1.5,
                label=f"k optimal = {best_k}")
    ax2.set_ylabel("Score silhouette", color=color_sil, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_sil)

    # Légende unifiée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    title = "Recherche de partitions : coude + silhouette"
    if source_label:
        title += f" — {source_label}"
    ax1.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
