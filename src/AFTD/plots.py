"""Visualization helpers for AFTD (Gower distance + classical MDS)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gower_mds import MDSResult

FIG_DPI = 150

sns.set_theme(style="whitegrid")


def variable_axis_correlations(
    df: pd.DataFrame,
    coordinates: np.ndarray,
    quantitative: list[str],
    binary: list[str],
    nominal: list[str],
    ordinal: list[str],
) -> pd.DataFrame:
    """Links between variables and MDS axes 1–2.

    Quantitative, binary and ordinal variables use Pearson correlation
    (binary coded 0/1, ordinal as ranks). Nominal variables use the
    correlation ratio (eta).
    """
    if coordinates.shape[1] < 2:
        raise ValueError("At least two MDS dimensions are required for contribution plots.")

    axis1 = coordinates[:, 0]
    axis2 = coordinates[:, 1]
    rows: list[dict[str, object]] = []

    binary_set = set(binary)
    ordinal_set = set(ordinal)
    for col in quantitative + binary + ordinal:
        if col in binary_set:
            values = pd.Categorical(df[col]).codes.astype(np.float64)
        elif col in ordinal_set:
            values = df[col].rank(method="dense").to_numpy(dtype=np.float64)
        else:
            values = df[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "variable": col,
                "axis_1": float(np.corrcoef(values, axis1)[0, 1]),
                "axis_2": float(np.corrcoef(values, axis2)[0, 1]),
            }
        )

    for col in nominal:
        groups = pd.Categorical(df[col])
        rows.append(
            {
                "variable": col,
                "axis_1": _correlation_ratio(axis1, groups),
                "axis_2": _correlation_ratio(axis2, groups),
            }
        )

    result = pd.DataFrame(rows).set_index("variable")
    return result


def _correlation_ratio(axis_values: np.ndarray, categories: pd.Categorical) -> float:
    """Square root of eta-squared (correlation ratio) for a categorical variable."""
    y = np.asarray(axis_values, dtype=np.float64)
    codes = categories.codes
    valid = codes >= 0
    y = y[valid]
    codes = codes[valid]
    if y.size == 0:
        return 0.0

    grand_mean = y.mean()
    ss_total = float(((y - grand_mean) ** 2).sum())
    if ss_total == 0.0:
        return 0.0

    ss_between = 0.0
    for code in np.unique(codes):
        group = y[codes == code]
        ss_between += group.size * (group.mean() - grand_mean) ** 2

    eta_squared = ss_between / ss_total
    return float(np.sqrt(np.clip(eta_squared, 0.0, 1.0)))


def plot_scree(result: MDSResult, path: Path) -> None:
    """Scree plot: explained variance per component and cumulative line."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(1, len(result.explained_variance) + 1)
    ax1.bar(
        x,
        result.explained_variance,
        color="steelblue",
        alpha=0.85,
        label="Variance expliquée (%)",
    )
    ax1.set_xlabel("Composante MDS")
    ax1.set_ylabel("Variance expliquée (%)")
    ax1.set_title("AFTD — Éboulis des composantes (variance expliquée)")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        result.cumulative_variance,
        color="darkorange",
        marker="o",
        linewidth=2,
        label="Variance cumulée (%)",
    )
    ax2.set_ylabel("Variance cumulée (%)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_individuals_categorical(
    coordinates: np.ndarray,
    color_series: pd.Series,
    path: Path,
    title: str,
) -> None:
    """Scatter plot of individuals on axes 1–2, colored by a qualitative variable."""
    fig, ax = plt.subplots(figsize=(10, 8))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    palette = sns.color_palette("tab10", n_colors=color_series.nunique())

    for i, (label, group_idx) in enumerate(color_series.groupby(color_series, observed=False).groups.items()):
        idx = list(group_idx)
        ax.scatter(
            x[idx],
            y[idx],
            label=str(label),
            alpha=0.7,
            s=40,
            color=palette[i % len(palette)],
        )

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Axe MDS 1")
    ax.set_ylabel("Axe MDS 2")
    ax.set_title(title)
    ax.legend(title=color_series.name, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_individuals_continuous(
    coordinates: np.ndarray,
    values: pd.Series,
    path: Path,
    title: str,
    cmap: str = "viridis",
) -> None:
    """Scatter plot colored by a continuous quantitative variable."""
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=values.to_numpy(dtype=np.float64),
        cmap=cmap,
        alpha=0.75,
        s=40,
    )
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Axe MDS 1")
    ax.set_ylabel("Axe MDS 2")
    ax.set_title(title)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(values.name)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_gower_heatmap(
    distance_matrix: np.ndarray,
    sort_labels: pd.Series,
    path: Path,
) -> None:
    """Heatmap of Gower distances with individuals sorted by a qualitative variable."""
    order = sort_labels.sort_values(kind="stable").index
    sorted_d = distance_matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sorted_d,
        cmap="mako",
        square=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws={"label": "Distance de Gower"},
    )
    ax.set_title(f"AFTD — Matrice des distances de Gower (tri par {sort_labels.name})")
    ax.set_xlabel("Individus (triés)")
    ax.set_ylabel("Individus (triés)")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_variable_contributions(correlations: pd.DataFrame, path: Path) -> None:
    """Horizontal bar chart of variable links to MDS axes 1 and 2."""
    n_vars = len(correlations)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_vars * 0.35)))

    for ax, col, axis_label in zip(
        axes,
        ["axis_1", "axis_2"],
        ["Axe MDS 1", "Axe MDS 2"],
    ):
        series = correlations[col].sort_values(key=np.abs)
        colors = ["teal" if v >= 0 else "coral" for v in series.values]
        ax.barh(series.index.astype(str), series.values, color=colors, alpha=0.85)
        ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_xlabel("Corrélation / rapport de corrélation")
        ax.set_title(f"Liaison variables — {axis_label}")
        ax.set_xlim(-1.05, 1.05)

    fig.suptitle("AFTD — Contributions des variables aux deux premiers axes", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
