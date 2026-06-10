"""Visualisation du décrochage (G3=0) sur PC1×PC2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import DROPOUT_LABELS


def plot_pc1_pc2_dropout(
    coords: pd.DataFrame,
    dropout_mask: pd.Series,
    method: str,
    g3_col: str,
    *,
    save: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    x = coords["Dim1"].values
    y = coords["Dim2"].values
    mask = dropout_mask.values.astype(bool)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {True: "crimson", False: "steelblue"}
    markers = {True: "x", False: "o"}

    for is_dropout in (False, True):
        m = mask == is_dropout
        n = m.sum()
        pct = 100 * n / len(mask) if len(mask) else 0
        label = f"{DROPOUT_LABELS[is_dropout]} : {n} ({pct:.1f}%)"
        ax.scatter(
            x[m], y[m],
            c=colors[is_dropout],
            marker=markers[is_dropout],
            s=40 if not is_dropout else 55,
            alpha=0.7,
            label=label,
            linewidths=0.8 if is_dropout else 0,
        )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(f"{method} — PC1 × PC2 — Décrochage ({g3_col}=0)")
    ax.legend(loc="best")
    fig.tight_layout()

    if save and out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
