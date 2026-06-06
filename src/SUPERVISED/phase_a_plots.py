"""
phase_a_plots.py — Visualisations de la Phase A.

  - distribution de la cible par cluster (boxplot quant/ordinal, barplot empilé nominal/binaire)
  - projection plan factoriel (Dim1-Dim2) colorée par cluster, puis par cible
  - heatmap : moyenne (ou proportion) de la cible par cluster
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_CMAP = "tab10"


def plot_target_distribution(
    df: pd.DataFrame, target: str, kind: str, out_path: Path
) -> None:
    """Boxplot (quant/ordinal) ou barplot empilé des proportions (nominal/binaire)."""
    clusters = sorted(df["cluster_id"].unique())
    fig, ax = plt.subplots(figsize=(9, 6))

    if kind in ("quantitative", "ordinale"):
        data = [df.loc[df["cluster_id"] == c, target].dropna().to_numpy() for c in clusters]
        ax.boxplot(data, labels=[f"C{c}" for c in clusters], showmeans=True)
        ax.set_ylabel(target, fontsize=11)
        ax.set_title(f"Distribution de {target} par cluster", fontsize=12)
    else:
        ct = pd.crosstab(df["cluster_id"], df[target], normalize="index")
        bottom = np.zeros(len(ct))
        x = np.arange(len(ct))
        cmap = plt.get_cmap(_CMAP)
        for i, modality in enumerate(ct.columns):
            ax.bar(x, ct[modality].to_numpy(), bottom=bottom,
                   label=str(modality), color=cmap(i % 10))
            bottom += ct[modality].to_numpy()
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in ct.index])
        ax.set_ylabel("Proportion", fontsize=11)
        ax.set_title(f"Répartition de {target} par cluster", fontsize=12)
        ax.legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    ax.set_xlabel("Cluster", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scatter(ax, x, y, values, categorical: bool, title: str, label: str):
    if categorical:
        codes, uniques = pd.factorize(pd.Series(values))
        cmap = plt.get_cmap(_CMAP)
        sc = ax.scatter(x, y, c=codes, cmap=_CMAP, s=22, alpha=0.75,
                        vmin=0, vmax=max(len(uniques) - 1, 1))
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                       color=cmap(i % 10), label=str(u))
            for i, u in enumerate(uniques)
        ]
        ax.legend(handles=handles, title=label, fontsize=7,
                  bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        sc = ax.scatter(x, y, c=np.asarray(values, dtype=float),
                        cmap="viridis", s=22, alpha=0.8)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Dim 1", fontsize=10)
    ax.set_ylabel("Dim 2", fontsize=10)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":")


def plot_factorial_by_cluster(
    df: pd.DataFrame, dim_cols: list[str], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    _scatter(ax, df[dim_cols[0]], df[dim_cols[1]], df["cluster_id"],
             categorical=True, title="Plan factoriel coloré par cluster",
             label="Cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_factorial_by_target(
    df: pd.DataFrame, dim_cols: list[str], target: str, kind: str, out_path: Path
) -> None:
    categorical = kind in ("binaire", "nominale", "ordinale")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    _scatter(ax, df[dim_cols[0]], df[dim_cols[1]], df[target],
             categorical=categorical,
             title=f"Plan factoriel coloré par {target}", label=target)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_heatmap(
    df: pd.DataFrame, target: str, kind: str, out_path: Path
) -> None:
    """Heatmap moyenne (quant/ordinal) ou table de proportions (nominal/binaire)."""
    if kind in ("quantitative", "ordinale"):
        mat = df.groupby("cluster_id")[target].mean().to_frame("moyenne")
    else:
        mat = pd.crosstab(df["cluster_id"], df[target], normalize="index")

    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * mat.shape[1], 1 + 0.6 * len(mat)))
    im = ax.imshow(mat.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([str(c) for c in mat.columns], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([f"C{c}" for c in mat.index], fontsize=9)
    for i in range(len(mat)):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.to_numpy()[i, j]:.2f}", ha="center", va="center",
                    fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{target} par cluster", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
