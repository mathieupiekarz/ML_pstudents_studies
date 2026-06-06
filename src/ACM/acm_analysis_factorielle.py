#!/usr/bin/env python3
"""
ACM (MCA) — variables qualitatives et binaires de data_global.csv.

Complète l'ACP : comportements, socio-démographie, échelles à peu de modalités.
Les binaires sont détectées même si codées en nombres (≤ 2 modalités).
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
import prince
import seaborn as sns

from _utils import (
    ACM_RESULTS_DIR as RESULTS_DIR,
    build_typology,
    compute_mca_modality_contributions,
    ensure_results_dir,
    get_acm_columns,
    get_run_name,
    impute_qualitative,
    load_data,
    modality_variable_label,
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


def plot_scree_mca(pct: np.ndarray, cum: np.ndarray) -> None:
    axes_n = np.arange(1, len(pct) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(axes_n, pct, alpha=0.7, color="teal", label="Inertie (%)")
    ax1.set_xlabel("Axe")
    ax1.set_ylabel("Inertie expliquée (%)")
    ax1.set_title("ACM — Scree plot")
    ax2 = ax1.twinx()
    ax2.plot(axes_n, cum, "o-", color="darkred", label="Cumul (%)")
    ax2.set_ylabel("Inertie cumulée (%)")
    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mca_screeplot.png", dpi=150)
    plt.close(fig)


def plot_cumulative_mca(cum_pct: np.ndarray, n90: int, n95: int) -> None:
    axes_n = np.arange(1, len(cum_pct) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(axes_n, cum_pct, "o-", linewidth=2, color="teal")
    ax.axhline(90, linestyle="--", color="gray", label="90 %")
    ax.axhline(95, linestyle="--", color="darkgray", label="95 %")
    ax.axvline(n90, linestyle=":", color="green", label=f"Axes 90 % → {n90}")
    ax.axvline(n95, linestyle=":", color="orange", label=f"Axes 95 % → {n95}")
    ax.set_xlabel("Nombre d'axes")
    ax.set_ylabel("Inertie cumulée (%)")
    ax.set_title("ACM — Inertie cumulée (choix du nombre d'axes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mca_cumulative_inertia.png", dpi=150)
    plt.close(fig)


def plot_modality_contributions_bar(
    contrib: pd.DataFrame, dim: int, path: Path
) -> None:
    col = f"Dim{dim + 1}"
    top = contrib[col].sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 7))
    top.plot(kind="barh", ax=ax, color="teal")
    ax.set_xlabel("Contribution (%)")
    ax.set_title(f"ACM — Top contributions modalités — {col}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_modalities(
    col_coords: pd.DataFrame, pct: np.ndarray
) -> None:
    xlabel, ylabel = _axis_labels(pct)
    x = col_coords.iloc[:, 0].values
    y = col_coords.iloc[:, 1].values
    labels = col_coords.index.astype(str)
    colors = [
        hash(modality_variable_label(m)) % 10 for m in labels
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x, y, c=colors, cmap="tab10", s=80, alpha=0.85)
    for xi, yi, lab in zip(x, y, labels):
        short = lab.replace("__", "\n") if len(lab) > 20 else lab
        ax.annotate(short, (xi, yi), fontsize=6, alpha=0.9)
    ax.axhline(0, color="lightgray", linewidth=0.5)
    ax.axvline(0, color="lightgray", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("ACM — Nuage des modalités (axes 1–2)")
    fig.colorbar(scatter, ax=ax, label="Groupe (variable)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mca_modalities.png", dpi=150)
    plt.close(fig)


def plot_individuals_mca(row_coords: pd.DataFrame, pct: np.ndarray) -> None:
    xlabel, ylabel = _axis_labels(pct)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        row_coords.iloc[:, 0],
        row_coords.iloc[:, 1],
        alpha=0.5,
        s=25,
        c="teal",
    )
    ax.axhline(0, color="lightgray", linewidth=0.5)
    ax.axvline(0, color="lightgray", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("ACM — Nuage des individus (axes 1–2)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mca_individuals.png", dpi=150)
    plt.close(fig)


def plot_asymmetric_map(
    row_coords: pd.DataFrame,
    col_coords: pd.DataFrame,
    pct: np.ndarray,
) -> None:
    """Carte asymétrique : individus (points) et modalités (triangles)."""
    xlabel, ylabel = _axis_labels(pct)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(
        row_coords.iloc[:, 0],
        row_coords.iloc[:, 1],
        alpha=0.25,
        s=15,
        c="gray",
        label="Individus",
    )
    cx = col_coords.iloc[:, 0].values
    cy = col_coords.iloc[:, 1].values
    ax.scatter(cx, cy, marker="^", s=60, c="crimson", label="Modalités", zorder=5)
    for xi, yi, lab in zip(cx, cy, col_coords.index.astype(str)):
        if (xi**2 + yi**2) > 0.01:
            ax.annotate(
                lab.split("__")[-1][:12],
                (xi, yi),
                fontsize=5,
                color="darkred",
            )
    ax.axhline(0, color="lightgray", linewidth=0.5)
    ax.axvline(0, color="lightgray", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("ACM — Carte asymétrique (individus ↔ modalités)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mca_asymmetric_map.png", dpi=150)
    plt.close(fig)


def main() -> None:
    global RESULTS_DIR
    run_name = get_run_name()
    RESULTS_DIR = ensure_results_dir("ACM", run_name)
    df = load_data()
    typology = build_typology(df)
    acm_cols = get_acm_columns(typology)

    if not acm_cols:
        raise ValueError("Aucune variable qualitative détectée pour l'ACM.")

    X = impute_qualitative(df, acm_cols)
    n_obs = len(X)

    n_categories = sum(X[c].nunique() for c in acm_cols)
    n_comp = min(n_categories - len(acm_cols), n_obs - 1)
    mca = prince.MCA(n_components=n_comp, random_state=0).fit(X)

    pct = np.asarray(mca.percentage_of_variance_)
    cum = np.asarray(mca.cumulative_percentage_of_variance_)
    n90 = n_axes_for_threshold(cum / 100.0, 0.90)
    n95 = n_axes_for_threshold(cum / 100.0, 0.95)

    row_coords = mca.row_coordinates(X)
    col_coords = mca.column_coordinates(X)
    row_coords.columns = [f"Dim{i + 1}" for i in range(row_coords.shape[1])]
    col_coords.columns = [f"Dim{i + 1}" for i in range(col_coords.shape[1])]

    contrib = compute_mca_modality_contributions(col_coords)

    row_out = row_coords.copy()
    row_out.insert(0, "individu", np.arange(1, n_obs + 1))

    eigen_df = pd.DataFrame(
        {
            "axe": np.arange(1, len(pct) + 1),
            "eigenvalue": mca.eigenvalues_,
            "inertia_pct": pct,
            "inertia_cum_pct": cum,
        }
    )

    eigen_df.to_csv(RESULTS_DIR / "mca_eigenvalues.csv", index=False)
    contrib.to_csv(RESULTS_DIR / "mca_modality_contributions.csv")
    row_out.to_csv(RESULTS_DIR / "mca_individual_coordinates.csv", index=False)

    pct_display = pct

    plot_scree_mca(pct_display, cum)
    plot_cumulative_mca(cum, n90, n95)
    plot_modality_contributions_bar(
        contrib, 0, RESULTS_DIR / "mca_modality_contributions_dim1.png"
    )
    plot_modality_contributions_bar(
        contrib, 1, RESULTS_DIR / "mca_modality_contributions_dim2.png"
    )
    plot_modalities(col_coords.iloc[:, :2], pct_display)
    plot_individuals_mca(row_coords.iloc[:, :2], pct_display)
    plot_asymmetric_map(
        row_coords.iloc[:, :2],
        col_coords.iloc[:, :2],
        pct_display,
    )

    top1 = top_contributors(contrib, 0)
    top2 = top_contributors(contrib, 1)

    write_summary_section(
        "ACM",
        [
            f"Observations totales : {len(df)}",
            f"Observations utilisées (ACM) : {n_obs}",
            f"Variables ACM ({len(acm_cols)}) : {', '.join(acm_cols)}",
            f"Nombre d'axes pour 90 % d'inertie : {n90}",
            f"Nombre d'axes pour 95 % d'inertie : {n95}",
        ],
    )

    write_exploration_guide_section(
        "ACM",
        "mca_screeplot.png : structure des variables qualitatives.\n"
        "mca_cumulative_inertia.png : nombre d'axes MCA à retenir.\n"
        "mca_modality_contributions_dim1/2.png : modalités qui structurent chaque axe.\n"
        "mca_asymmetric_map.png : liens profils élèves ↔ catégories.\n"
        f"Axes conseillés : {n90} (90 %), {n95} (95 %).\n"
        "Pistes : modalités éloignées sur un axe → croisements / tableaux ; "
        "compléter avec l'ACP pour relier comportement et notes.",
    )

    print_exploration_footer(
        "ACM", len(acm_cols), n90, n95, top1, top2, results_dir=RESULTS_DIR
    )


if __name__ == "__main__":
    main()
