"""
kmeans.py — K-means, coude, silhouette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from factor_analysis.inertia_viz import LINE_MARKER


def run_kmeans(
    X: np.ndarray,
    k: int,
    init: str = "k-means++",
    n_init: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    km = KMeans(k, init=init, n_init=n_init, random_state=random_state)
    return km.fit_predict(X)


def run_nuees_dynamiques(
    X: np.ndarray,
    k: int,
    Z: np.ndarray,
    n_init: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    cah_labels = fcluster(Z, k, criterion="maxclust")
    centroids = np.array([
        X[cah_labels == c].mean(axis=0) for c in range(1, k + 1)
    ])
    km = KMeans(k, init=centroids, n_init=n_init, random_state=random_state)
    return km.fit_predict(X)


def compute_k_metrics(X: np.ndarray, k_range: range | list[int]) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(k, init="k-means++", n_init=20, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else float("nan")
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def select_k_silhouette(X: np.ndarray, k_range: range | list[int]) -> int:
    metrics = compute_k_metrics(X, k_range)
    return suggest_k_silhouette(metrics)


def suggest_k_silhouette(metrics: pd.DataFrame) -> int:
    idx = metrics["silhouette"].idxmax()
    return int(metrics.loc[idx, "k"])


def suggest_k_elbow(inertias: pd.Series) -> int:
    """Point de coude via distance maximale à la corde."""
    ks = inertias.index.values.astype(float)
    ys = inertias.values.astype(float)
    if len(ks) < 3:
        return int(ks[0])
    p1 = np.array([ks[0], ys[0]])
    p2 = np.array([ks[-1], ys[-1]])
    line = p2 - p1
    line_norm = line / (np.linalg.norm(line) + 1e-12)
    dists = []
    for i in range(len(ks)):
        p = np.array([ks[i], ys[i]])
        dist = np.abs(np.cross(line_norm, p - p1))
        dists.append(dist)
    return int(ks[int(np.argmax(dists))])


def plot_elbow(
    metrics_by_method: dict[str, pd.DataFrame],
    *,
    save: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, metrics in metrics_by_method.items():
        ax.plot(metrics["k"], metrics["inertia"], f"{LINE_MARKER}-", label=method)
    ax.set_xlabel("k")
    ax.set_ylabel("Inertie intra-classe")
    ax.set_title("Méthode du coude")
    ax.legend()
    fig.tight_layout()
    if save and out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig


def plot_silhouette(
    metrics_by_method: dict[str, pd.DataFrame],
    *,
    save: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, metrics in metrics_by_method.items():
        ax.plot(metrics["k"], metrics["silhouette"], f"{LINE_MARKER}--", label=method)
    ax.set_xlabel("k")
    ax.set_ylabel("Score silhouette")
    ax.set_title("Score silhouette moyen")
    ax.legend()
    fig.tight_layout()
    if save and out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig


def plot_elbow_silhouette(
    X: np.ndarray,
    k_range: range | list[int],
    out_path: Path,
    source_label: str = "",
) -> None:
    metrics = compute_k_metrics(X, k_range)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(metrics["k"], metrics["inertia"], f"{LINE_MARKER}-", color="steelblue", label="Inertie")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertie")
    ax2 = ax1.twinx()
    ax2.plot(metrics["k"], metrics["silhouette"], f"{LINE_MARKER}--", color="darkorange", label="Silhouette")
    ax2.set_ylabel("Silhouette")
    title = "Coude + silhouette"
    if source_label:
        title += f" — {source_label}"
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
