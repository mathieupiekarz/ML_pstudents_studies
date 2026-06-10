"""Visualisation comparée de l'inertie (style exemple/acp_alternative.ipynb)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from factor_analysis.models import AnalysisResult

LINE_MARKER = "s"


def plot_inertia_comparison(
    results: dict[str, AnalysisResult],
    *,
    title: str,
    n_show: int = 20,
) -> plt.Figure:
    """Figure 1×2 : inertie par axe | inertie cumulée, tronquée à n_show axes."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax_per, ax_cum = axes

    for method, res in results.items():
        pct = res.explained_variance_ratio
        n = min(n_show, len(pct))
        x = np.arange(1, n + 1)
        label = f"{method} ({res.n_vars} vars)" if res.n_vars else method
        ax_per.plot(x, pct[:n] * 100, marker=LINE_MARKER, label=label)
        ax_cum.plot(x, np.cumsum(pct[:n]) * 100, marker=LINE_MARKER, label=label)

    ax_per.set_xlabel("Axe")
    ax_per.set_ylabel("% d'inertie")
    ax_per.set_title("Inertie par axe")
    ax_per.legend()
    ax_per.grid(True, alpha=0.4)

    ax_cum.axhline(60, color="grey", ls="--", alpha=0.5, label="60%")
    ax_cum.axhline(80, color="grey", ls=":", alpha=0.5, label="80%")
    ax_cum.set_xlabel("Nombre d'axes")
    ax_cum.set_ylabel("% cumulé")
    ax_cum.set_title("Inertie cumulée")
    ax_cum.set_ylim(0, 105)
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.4)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig
