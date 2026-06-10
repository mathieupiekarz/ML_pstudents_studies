"""Pipeline clustering k-means pour le notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CLUSTERING.kmeans import (
    compute_k_metrics,
    plot_elbow,
    plot_silhouette,
    run_kmeans,
    suggest_k_elbow,
    suggest_k_silhouette,
)
from CLUSTERING.profiles import (
    compute_cluster_profiles,
    plot_cluster_projection,
    plot_cluster_zscore_heatmap,
    plot_dropout_by_cluster,
    plot_variable_by_cluster,
    plot_variable_repartitions_grid,
)
from config import extract_dropout_mask
from factor_analysis.models import AnalysisResult, ClusteringResult


def run_clustering_pipeline(
    factor_results: dict[str, AnalysisResult],
    df_preprocessed: pd.DataFrame,
    df_raw: pd.DataFrame,
    typology: pd.DataFrame,
    k_range: range,
    variance_threshold: float,
    k_final: int,
    g3_col: str,
    *,
    methods: list[str] | None = None,
    save: bool = False,
    out_dir: Path | None = None,
) -> tuple[dict[str, ClusteringResult], object, object]:
    dropout_mask = extract_dropout_mask(df_raw, g3_col)
    selected = methods or list(factor_results.keys())
    metrics_by_method: dict[str, pd.DataFrame] = {}
    results: dict[str, ClusteringResult] = {}

    for method in selected:
        res = factor_results[method]
        X = res.select_coords(threshold=variance_threshold)
        metrics_by_method[method] = compute_k_metrics(X, k_range)

    elbow_fig = plot_elbow(metrics_by_method, save=False)
    sil_fig = plot_silhouette(metrics_by_method, save=False)

    for method in selected:
        res = factor_results[method]
        X = res.select_coords(threshold=variance_threshold)
        metrics = metrics_by_method[method]
        labels = run_kmeans(X, k_final)
        k_elbow = suggest_k_elbow(metrics.set_index("k")["inertia"])
        k_sil = suggest_k_silhouette(metrics)

        figs: dict = {}
        prof_figs: dict = {}

        figs["cluster_projection"] = plot_cluster_projection(X, labels, method, save=False)
        profiles = compute_cluster_profiles(df_preprocessed, labels, typology)
        dropout_rates = pd.Series({
            c: dropout_mask[labels == c].mean() * 100
            for c in np.unique(labels)
        })
        prof_figs["zscore_heatmap"] = plot_cluster_zscore_heatmap(
            profiles, method, dropout_rates=dropout_rates, save=False,
        )
        prof_figs["dropout_by_cluster"] = plot_dropout_by_cluster(
            labels, dropout_mask, method, save=False,
        )
        prof_figs["variable_repartitions_grid"] = plot_variable_repartitions_grid(
            df_preprocessed, labels, typology, method, k_final, save=False,
        )

        if save and out_dir:
            method_dir = out_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            for name, fig in {**figs, **prof_figs}.items():
                fig.savefig(method_dir / f"{name}.png", dpi=150, bbox_inches="tight")

        results[method] = ClusteringResult(
            method_label=method,
            labels=labels,
            coords=X,
            k=k_final,
            metrics=metrics,
            figures=figs,
            profile_figures=prof_figs,
        )

    return results, elbow_fig, sil_fig


def plot_variable_for_method(
    df: pd.DataFrame,
    typology: pd.DataFrame,
    labels: np.ndarray,
    var: str,
):
    row = typology.loc[typology["variable"] == var]
    var_type = row["type_retenu"].iloc[0] if len(row) else "quantitative"
    return plot_variable_by_cluster(df, var, labels, var_type, save=False)


def plot_all_repartitions(
    df: pd.DataFrame,
    typology: pd.DataFrame,
    labels: np.ndarray,
    method: str,
    k: int,
) -> plt.Figure:
    return plot_variable_repartitions_grid(df, labels, typology, method, k)
