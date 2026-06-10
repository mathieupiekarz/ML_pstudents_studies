#!/usr/bin/env python3
"""
ACM (MCA) — variables qualitatives et binaires de data_global.csv.

Complète l'ACP : comportements, socio-démographie, échelles à peu de modalités.

Paramètres dans config.py :
  REMOVE_G1_G2    — retire G1.m, G2.m, G1.p, G2.p avant l'analyse
  AVERAGE_MAT_POR — moyenne les paires .m/.p en une seule variable
  → Le dossier results s'appelle results_full, results_no_g1g2, results_avg, etc.
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
    build_typology,
    compute_mca_modality_contributions,
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
from config import apply_preprocessing, results_suffix

sns.set_theme(style="whitegrid")

_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPT_DIR / "results"


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


def plot_modalities(col_coords: pd.DataFrame, pct: np.ndarray) -> None:
    xlabel, ylabel = _axis_labels(pct)
    x = col_coords.iloc[:, 0].values
    y = col_coords.iloc[:, 1].values
    labels = col_coords.index.astype(str)
    colors = [hash(modality_variable_label(m)) % 10 for m in labels]

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
    ax.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1], alpha=0.5, s=25, c="teal")
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
    xlabel, ylabel = _axis_labels(pct)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(
        row_coords.iloc[:, 0], row_coords.iloc[:, 1],
        alpha=0.25, s=15, c="gray", label="Individus",
    )
    cx = col_coords.iloc[:, 0].values
    cy = col_coords.iloc[:, 1].values
    ax.scatter(cx, cy, marker="^", s=60, c="crimson", label="Modalités", zorder=5)
    for xi, yi, lab in zip(cx, cy, col_coords.index.astype(str)):
        if (xi**2 + yi**2) > 0.01:
            ax.annotate(lab.split("__")[-1][:12], (xi, yi), fontsize=5, color="darkred")
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
    RESULTS_DIR = RESULTS_DIR / run_name

    df_raw = load_data()
    df = apply_preprocessing(df_raw)
    from factor_analysis.runners import run_acm
    run_acm(df, RESULTS_DIR, save=True)


if __name__ == "__main__":
    main()