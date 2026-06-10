"""Types de résultats pour l'analyse factorielle et le clustering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.figure
import numpy as np
import pandas as pd


@dataclass
class AnalysisResult:
    method_label: str
    eigen_df: pd.DataFrame
    explained_variance_ratio: np.ndarray
    coords_df: pd.DataFrame
    contributions: pd.DataFrame | None = None
    figures: dict[str, matplotlib.figure.Figure] = field(default_factory=dict)
    results_dir: Path | None = None
    n_vars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cum_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    def select_coords(self, n_axes: int | None = None, threshold: float = 0.80) -> np.ndarray:
        from _utils import n_axes_for_threshold

        dim_cols = [c for c in self.coords_df.columns if str(c).startswith("Dim")]
        X = self.coords_df[dim_cols].values.astype(float)
        if n_axes is not None:
            return X[:, :n_axes]
        n = n_axes_for_threshold(self.cum_variance, threshold)
        return X[:, :n]


@dataclass
class ClusteringResult:
    method_label: str
    labels: np.ndarray
    coords: np.ndarray
    k: int
    metrics: pd.DataFrame
    figures: dict[str, matplotlib.figure.Figure] = field(default_factory=dict)
    profile_figures: dict[str, matplotlib.figure.Figure] = field(default_factory=dict)
