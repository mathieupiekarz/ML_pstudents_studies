#!/usr/bin/env python3
"""
ACP (PCA) — variables quantitatives de data_global.csv.

Exploration globale : toutes les colonnes sont typées automatiquement ;
seules les quantitatives (non binaires / peu de modalités) entrent dans l'ACP.
Les données sont centrées-réduites avant PCA.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _utils import (
    ACP_RESULTS_DIR as RESULTS_DIR,
    build_typology,
    compute_pca_contributions,
    ensure_results_dir,
    get_acp_columns,
    impute_quantitative,
    load_data,
    n_axes_for_threshold,
    print_exploration_footer,
    top_contributors,
    write_exploration_guide_section,
    write_summary_section,
)

sns.set_theme(style="whitegrid")


def _axis_labels(pct: np.ndarray, n: int = 2) -> tuple[str, str]:
    return (
        f"Axe 1 ({pct[0]:.1f} %)",
        f"Axe 2 ({pct[1]:.1f} %)" if len(pct) > 1 else "Axe 2",
    )


def plot_scree(eigenvalues: np.ndarray, pct: np.ndarray, cum: np.ndarray) -> None:
    axes_n = np.arange(1, len(eigenvalues) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(axes_n, pct, alpha=0.7, label="Inertie (%)")
    ax1.set_xlabel("Axe")
    ax1.set_ylabel("Inertie expliquée (%)")
    ax1.set_title("ACP — Scree plot")
    ax2 = ax1.twinx()
    ax2.plot(axes_n, cum * 100, "o-", color="darkred", label="Cumul (%)")
    ax2.set_ylabel("Inertie cumulée (%)")
    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pca_screeplot.png", dpi=150)
    plt.close(fig)


def plot_cumulative(cum: np.ndarray, n90: int, n95: int) -> None:
    axes_n = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(axes_n, cum * 100, "o-", linewidth=2)
    ax.axhline(90, linestyle="--", color="gray", label="90 %")
    ax.axhline(95, linestyle="--", color="darkgray", label="95 %")
    ax.axvline(n90, linestyle=":", color="green", label=f"Axes 90 % → {n90}")
    ax.axvline(n95, linestyle=":", color="orange", label=f"Axes 95 % → {n95}")
    ax.set_xlabel("Nombre d'axes")
    ax.set_ylabel("Inertie cumulée (%)")
    ax.set_title("ACP — Inertie cumulée (choix du nombre d'axes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pca_cumulative_inertia.png", dpi=150)
    plt.close(fig)


def plot_variable_contributions_bar(
    contrib: pd.DataFrame, dim: int, path: Path
) -> None:
    col = f"Dim{dim + 1}"
    top = contrib[col].sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Contribution (%)")
    ax.set_title(f"ACP — Top contributions variables — {col}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_variables_circle(
    loadings: pd.DataFrame, pct: np.ndarray, variables: list[str]
) -> None:
    xlabel, ylabel = _axis_labels(pct)
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
    ax.add_patch(circle)
    for var in variables:
        ax.arrow(
            0,
            0,
            loadings.loc[var, "Dim1"],
            loadings.loc[var, "Dim2"],
            head_width=0.02,
            length_includes_head=True,
            color="darkblue",
        )
        ax.text(
            loadings.loc[var, "Dim1"] * 1.08,
            loadings.loc[var, "Dim2"] * 1.08,
            var,
            fontsize=8,
        )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="lightgray", linewidth=0.5)
    ax.axvline(0, color="lightgray", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("ACP — Cercle des corrélations (axes 1–2)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pca_variables.png", dpi=150)
    plt.close(fig)


def plot_individuals(coords: pd.DataFrame, pct: np.ndarray) -> None:
    xlabel, ylabel = _axis_labels(pct)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords["Dim1"], coords["Dim2"], alpha=0.5, s=25, c="steelblue")
    ax.axhline(0, color="lightgray", linewidth=0.5)
    ax.axvline(0, color="lightgray", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("ACP — Nuage des individus (axes 1–2)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pca_individuals.png", dpi=150)
    plt.close(fig)


def plot_biplot(
    coords: pd.DataFrame,
    loadings: pd.DataFrame,
    pct: np.ndarray,
    variables: list[str],
) -> None:
    xlabel, ylabel = _axis_labels(pct)
    fig, ax = plt.subplots(figsize=(9, 8))
    scale = 3.0
    ax.scatter(coords["Dim1"], coords["Dim2"], alpha=0.35, s=20, c="lightgray")
    for var in variables:
        lx = loadings.loc[var, "Dim1"] * scale
        ly = loadings.loc[var, "Dim2"] * scale
        ax.arrow(0, 0, lx, ly, head_width=0.05, color="darkred", alpha=0.8)
        ax.text(lx * 1.05, ly * 1.05, var, fontsize=8, color="darkred")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("ACP — Biplot (individus + variables, axes 1–2)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pca_biplot.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_results_dir("ACP")
    df = load_data()
    typology = build_typology(df)
    acp_cols = get_acp_columns(typology)

    if not acp_cols:
        raise ValueError("Aucune variable quantitative détectée pour l'ACP.")

    X = impute_quantitative(df, acp_cols)
    n_obs = len(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=None, random_state=0)
    coords = pca.fit_transform(X_scaled)

    pct = pca.explained_variance_ratio_
    cum = np.cumsum(pct)
    n90 = n_axes_for_threshold(cum, 0.90)
    n95 = n_axes_for_threshold(cum, 0.95)

    contrib = compute_pca_contributions(pca.components_, pca.explained_variance_)
    contrib.index = acp_cols
    contrib.index.name = "variable"

    loadings = pd.DataFrame(
        pca.components_.T[:, :2] * np.sqrt(pca.explained_variance_[:2]),
        index=acp_cols,
        columns=["Dim1", "Dim2"],
    )

    coords_df = pd.DataFrame(
        coords,
        columns=[f"Dim{i + 1}" for i in range(coords.shape[1])],
    )
    coords_df.insert(0, "individu", np.arange(1, n_obs + 1))

    eigen_df = pd.DataFrame(
        {
            "axe": np.arange(1, len(pct) + 1),
            "eigenvalue": pca.explained_variance_,
            "inertia_pct": pct * 100,
            "inertia_cum_pct": cum * 100,
        }
    )

    eigen_df.to_csv(RESULTS_DIR / "pca_eigenvalues.csv", index=False)
    contrib.to_csv(RESULTS_DIR / "pca_variable_contributions.csv")
    coords_df.to_csv(RESULTS_DIR / "pca_individual_coordinates.csv", index=False)

    plot_scree(pca.explained_variance_, pct, cum)
    plot_cumulative(cum, n90, n95)
    plot_variable_contributions_bar(
        contrib, 0, RESULTS_DIR / "pca_variable_contributions_dim1.png"
    )
    plot_variable_contributions_bar(
        contrib, 1, RESULTS_DIR / "pca_variable_contributions_dim2.png"
    )
    plot_variables_circle(loadings, pct * 100, acp_cols)
    plot_individuals(coords_df, pct * 100)
    plot_biplot(coords_df, loadings, pct * 100, acp_cols)

    top1 = top_contributors(contrib, 0)
    top2 = top_contributors(contrib, 1)

    write_summary_section(
        "ACP",
        [
            f"Observations totales : {len(df)}",
            f"Observations utilisées (ACP) : {n_obs}",
            f"Variables ACP : {', '.join(acp_cols)}",
            f"Nombre d'axes pour 90 % d'inertie : {n90}",
            f"Nombre d'axes pour 95 % d'inertie : {n95}",
        ],
    )

    write_exploration_guide_section(
        "ACP",
        "pca_screeplot.png : inertie concentrée ou diffuse ?\n"
        "pca_cumulative_inertia.png : combien d'axes pour 90 % / 95 % ?\n"
        "pca_variable_contributions_dim1/2.png : quelles notes ou absences pilotent chaque axe ?\n"
        "pca_biplot.png : quels profils d'élèves sont liés à quelles variables ?\n"
        f"Axes conseillés : {n90} (90 %), {n95} (95 %).\n"
        "Pistes : si G1/G2/G3 dominent l'axe 1, analyser la progression des notes ; "
        "si absences dominent, croiser avec variables ACM.",
    )

    print_exploration_footer("ACP", len(acp_cols), n90, n95, top1, top2)


if __name__ == "__main__":
    main()
