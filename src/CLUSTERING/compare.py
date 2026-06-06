"""
compare.py — Comparaison de partitions.

Méthodes exposées
-----------------
compare_partitions(labels_dict, X, out_path)
    → heatmap ARI + tableau silhouette
    → clustering_comparison.csv

L'Adjusted Rand Index (ARI) mesure l'accord entre deux partitions indépendamment
du nombre de clusters et de l'étiquetage. ARI = 1 : partitions identiques ;
ARI ≈ 0 : accord aléatoire.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score


def compare_partitions(
    labels_dict: dict[str, np.ndarray],
    X: np.ndarray,
    out_path: Path,
    source_label: str = "",
) -> pd.DataFrame:
    """
    Compare toutes les partitions dans `labels_dict`.

    Paramètres
    ----------
    labels_dict : {nom_méthode: labels_array}
    X           : coordonnées (n, d) utilisées pour le score silhouette
    out_path    : chemin du fichier PNG de la heatmap ARI

    Retourne
    --------
    DataFrame avec colonnes : methode, n_clusters, silhouette
    et la matrice ARI en heatmap.
    """
    methods = list(labels_dict)
    n = len(methods)

    # ── Matrice ARI ──────────────────────────────────────────────────────────
    ari_matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        ari = adjusted_rand_score(labels_dict[methods[i]], labels_dict[methods[j]])
        ari_matrix[i, j] = ari
        ari_matrix[j, i] = ari
    np.fill_diagonal(ari_matrix, 1.0)

    # ── Tableau silhouette ────────────────────────────────────────────────────
    rows = []
    for name, labels in labels_dict.items():
        n_clusters = len(np.unique(labels))
        sil = silhouette_score(X, labels) if n_clusters > 1 else float("nan")
        rows.append({"méthode": name, "k": n_clusters, "silhouette_moyen": round(sil, 4)})
    summary_df = pd.DataFrame(rows)

    # ── Figure : heatmap ARI ─────────────────────────────────────────────────
    fig, (ax_heat, ax_table) = plt.subplots(
        1, 2, figsize=(7 + 3 * n, 4 + n),
        gridspec_kw={"width_ratios": [n, 2]},
    )

    im = ax_heat.imshow(ari_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
    ax_heat.set_yticklabels(methods, fontsize=10)

    for i in range(n):
        for j in range(n):
            ax_heat.text(j, i, f"{ari_matrix[i, j]:.2f}",
                         ha="center", va="center", fontsize=9,
                         color="black" if ari_matrix[i, j] > 0.4 else "white")

    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="ARI")
    title = "Comparaison de partitions (ARI)"
    if source_label:
        title += f"\n{source_label}"
    ax_heat.set_title(title, fontsize=11)

    # Tableau silhouette à droite
    ax_table.axis("off")
    col_labels = ["Méthode", "k", "Silhouette"]
    table_data = [
        [r["méthode"], r["k"], f"{r['silhouette_moyen']:.3f}"]
        for _, r in summary_df.iterrows()
    ]
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    ax_table.set_title("Qualité des partitions", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return summary_df
