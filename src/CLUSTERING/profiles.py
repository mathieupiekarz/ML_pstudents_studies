"""
profiles.py — Interprétation des clusters post k-means.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DROPOUT_LABELS


def plot_cluster_projection(
    coords: np.ndarray,
    labels: np.ndarray,
    method: str,
    *,
    dim_names: tuple[str, str] = ("Dim 1", "Dim 2"),
    save: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    x, y = coords[:, 0], coords[:, 1]
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels_norm = np.array([remap[l] for l in labels])
    k = len(unique)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(x, y, c=labels_norm, cmap="tab10", s=25, alpha=0.75)
    for c in range(k):
        mask = labels_norm == c
        cx, cy = x[mask].mean(), y[mask].mean()
        ax.scatter(cx, cy, marker="X", s=120, c="black", zorder=5)
        ax.annotate(f"C{unique[c]}", (cx, cy), xytext=(6, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel(dim_names[0])
    ax.set_ylabel(dim_names[1])
    ax.set_title(f"{method} — Projection clusters PC1×PC2")
    fig.tight_layout()
    if save and out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig


def plot_variable_by_cluster(
    df: pd.DataFrame,
    var: str,
    labels: np.ndarray,
    var_type: str,
    *,
    save: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    tmp = pd.DataFrame({"cluster": labels, "var": df[var].values})

    if var_type == "quantitative":
        clusters = sorted(tmp["cluster"].unique())
        data = [tmp.loc[tmp["cluster"] == c, "var"].values for c in clusters]
        ax.boxplot(data, labels=[str(c) for c in clusters])
        ax.set_ylabel(var)
        ax.set_title(f"Distribution de {var} par cluster")
    else:
        ct = pd.crosstab(tmp["cluster"], tmp["var"], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_ylabel("% du cluster")
        ax.set_xlabel("Cluster")
        ax.set_title(f"Répartition de {var} par cluster")
        ax.legend(title=var, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    if save and out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def compute_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    typology: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for var in df.columns:
        row = typology.loc[typology["variable"] == var]
        vtype = row["type_retenu"].iloc[0] if len(row) else "quantitative"
        for cluster in np.unique(labels):
            mask = labels == cluster
            if vtype == "quantitative":
                val = df.loc[mask, var].astype(float).mean()
            else:
                counts = df.loc[mask, var].astype(str).value_counts(normalize=True)
                val = counts.max() if len(counts) else 0.0
            records.append({"variable": var, "cluster": cluster, "value": val, "type": vtype})
    return pd.DataFrame(records)


def plot_cluster_zscore_heatmap(
    profiles: pd.DataFrame,
    method: str,
    *,
    dropout_rates: pd.Series | None = None,
    save: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    pivot = profiles.pivot(index="variable", columns="cluster", values="value")
    z = pivot.sub(pivot.mean(axis=1), axis=0).div(pivot.std(axis=1).replace(0, 1), axis=0)

    if dropout_rates is not None:
        z.loc["taux_decrochage"] = [
            (dropout_rates.get(c, 0) - dropout_rates.mean()) / max(dropout_rates.std(), 1e-6)
            for c in z.columns
        ]

    fig, ax = plt.subplots(figsize=(max(8, len(z.columns) * 1.5), max(6, len(z) * 0.3)))
    sns.heatmap(z, cmap="RdBu_r", center=0, ax=ax, linewidths=0.3)
    ax.set_title(f"{method} — Heatmap z-score par cluster")
    fig.tight_layout()
    if save and out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_stacked_repartition(
    df: pd.DataFrame,
    var: str,
    labels: np.ndarray,
    *,
    save: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    tmp = pd.DataFrame({"cluster": labels, "var": df[var].astype(str).values})
    ct = pd.crosstab(tmp["cluster"], tmp["var"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_ylabel("% du cluster")
    ax.set_xlabel("Cluster")
    ax.set_title(var)
    ax.legend(title=var, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    if save and out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_dropout_by_cluster(
    labels: np.ndarray,
    dropout_mask: pd.Series,
    method: str,
    *,
    save: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    tmp = pd.DataFrame({
        "cluster": labels,
        "statut": ["Décrochage" if d else "Non-décrochage" for d in dropout_mask.values],
    })
    ct = pd.crosstab(tmp["cluster"], tmp["statut"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=["steelblue", "crimson"])
    ax.set_ylabel("% du cluster")
    ax.set_xlabel("Cluster")
    ax.set_title(f"{method} — Décrochage (G3=0) par cluster")
    ax.legend(title="Statut")
    fig.tight_layout()
    if save and out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_all_variable_repartitions(
    df: pd.DataFrame,
    labels: np.ndarray,
    typology: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    max_cols: int = 3,
    max_plots: int | None = None,
) -> list[plt.Figure]:
    figures: list[plt.Figure] = []
    vars_list = list(df.columns)
    if max_plots:
        vars_list = vars_list[:max_plots]

    for var in vars_list:
        row = typology.loc[typology["variable"] == var]
        vtype = row["type_retenu"].iloc[0] if len(row) else "quantitative"
        out_path = out_dir / f"repartition_{var}.png" if out_dir else None
        if vtype == "quantitative":
            fig = plot_variable_by_cluster(df, var, labels, "quantitative", save=out_dir is not None, out_path=out_path)
        else:
            fig = plot_stacked_repartition(df, var, labels, save=out_dir is not None, out_path=out_path)
        figures.append(fig)
    return figures


def plot_variable_repartitions_grid(
    df: pd.DataFrame,
    labels: np.ndarray,
    typology: pd.DataFrame,
    method: str,
    k: int,
    *,
    n_cols: int = 4,
    save: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    """Grille compacte de répartitions par variable (style exemple §6.4)."""
    vars_list = list(df.columns)
    n_vars = len(vars_list)
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes_grid]
    elif n_rows == 1:
        axes_flat = list(axes_grid)
    else:
        axes_flat = axes_grid.flat

    tmp_base = pd.DataFrame({"cluster": labels})

    for ax, var in zip(axes_flat, vars_list):
        row = typology.loc[typology["variable"] == var]
        vtype = row["type_retenu"].iloc[0] if len(row) else "quantitative"
        tmp = tmp_base.copy()
        tmp["var"] = df[var].values

        if vtype == "quantitative":
            clusters = sorted(tmp["cluster"].unique())
            data = [tmp.loc[tmp["cluster"] == c, "var"].values for c in clusters]
            ax.boxplot(data, labels=[str(c) for c in clusters])
            ax.set_ylabel(var)
        else:
            ct = pd.crosstab(tmp["cluster"], tmp["var"].astype(str), normalize="index") * 100
            ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", legend=False)
            ax.set_ylabel("% du cluster")
            if len(ct.columns) <= 6:
                ax.legend(fontsize=7, loc="upper right", title=var)
        ax.set_title(var)
        ax.set_xlabel("cluster")
        ax.tick_params(axis="x", rotation=0)

    for ax in list(axes_flat)[n_vars:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Répartition des variables par cluster — {method}, k={k}",
        fontsize=13,
        y=1.00,
    )
    fig.tight_layout()
    if save and out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
