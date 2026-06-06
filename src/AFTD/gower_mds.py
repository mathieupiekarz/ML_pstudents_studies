"""Gower distance and classical multidimensional scaling (AFTD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

EIGENVALUE_TOL = 1e-10

Correction = Literal["cailliez", "lingoes", "none"]


@dataclass(frozen=True)
class MDSResult:
    """Output of classical MDS on a distance matrix."""

    coordinates: np.ndarray
    eigenvalues: np.ndarray
    explained_variance: np.ndarray
    cumulative_variance: np.ndarray
    n_negative_before: int
    correction: str
    correction_constant: float


def _pairwise_mean_dissimilarity(blocks: list[np.ndarray]) -> np.ndarray:
    """Average variable-wise dissimilarities over all blocks (n x n each)."""
    if not blocks:
        raise ValueError("At least one variable block is required.")
    stacked = np.stack(blocks, axis=0)
    return np.mean(stacked, axis=0)


def _normalized_manhattan(values: np.ndarray) -> np.ndarray | None:
    """Range-normalized absolute differences for a numeric/ordinal column."""
    col_range = float(values.max() - values.min())
    if col_range == 0.0:
        return None
    return np.abs(values[:, np.newaxis] - values[np.newaxis, :]) / col_range


def gower_distance_matrix(
    df: pd.DataFrame,
    quantitative: list[str],
    binary: list[str],
    nominal: list[str],
    ordinal: list[str],
) -> np.ndarray:
    """Compute the Gower distance matrix from scratch.

    Quantitative variables use normalized Manhattan distance (|xi - xj| / range).
    Ordinal variables are converted to ranks, then use the same normalized
    Manhattan distance so the order of categories is respected.
    Binary and nominal variables use 0/1 mismatch indicators.
    The final distance is the mean dissimilarity across all variables.
    """
    blocks: list[np.ndarray] = []

    for col in quantitative:
        values = df[col].to_numpy(dtype=np.float64)
        block = _normalized_manhattan(values)
        if block is not None:
            blocks.append(block)

    for col in ordinal:
        ranks = df[col].rank(method="dense").to_numpy(dtype=np.float64)
        block = _normalized_manhattan(ranks)
        if block is not None:
            blocks.append(block)

    for col in binary:
        values = df[col].to_numpy()
        mismatch = values[:, np.newaxis] != values[np.newaxis, :]
        blocks.append(mismatch.astype(np.float64))

    for col in nominal:
        codes = pd.Categorical(df[col]).codes.astype(np.int32)
        mismatch = codes[:, np.newaxis] != codes[np.newaxis, :]
        blocks.append(mismatch.astype(np.float64))

    distance = _pairwise_mean_dissimilarity(blocks)
    np.fill_diagonal(distance, 0.0)
    distance = np.maximum(distance, distance.T)
    return distance


def _double_centering(squared_or_raw: np.ndarray) -> np.ndarray:
    """Return B = -0.5 * H M H with H the centering matrix."""
    n = squared_or_raw.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * h @ squared_or_raw @ h


def _cailliez_constant(d: np.ndarray) -> float:
    """Additive constant c (Cailliez, 1983) making distances Euclidean.

    c is the largest real eigenvalue of the 2n x 2n block matrix
    [[0, 2*B2], [-I, -4*B1]], with B1 = double-centering of D and
    B2 = double-centering of D^2.
    """
    n = d.shape[0]
    b1 = _double_centering(d)
    b2 = _double_centering(d**2)
    zero = np.zeros((n, n))
    identity = np.eye(n)
    top = np.hstack([zero, 2.0 * b2])
    bottom = np.hstack([-identity, -4.0 * b1])
    block = np.vstack([top, bottom])
    eigvals = np.linalg.eigvals(block)
    c = float(np.max(eigvals.real))
    return max(c, 0.0)


def _lingoes_constant(b2: np.ndarray) -> float:
    """Additive constant for Lingoes (1971): c = -2 * lambda_min(B2)."""
    min_eig = float(np.linalg.eigvalsh(b2).min())
    return max(-2.0 * min_eig, 0.0)


def classical_mds(
    distance_matrix: np.ndarray,
    correction: Correction = "cailliez",
    eigenvalue_tol: float = EIGENVALUE_TOL,
) -> MDSResult:
    """Apply classical MDS (metric AFTD) on a symmetric distance matrix.

    Steps: square distances, double-center, optionally apply an additive
    correction (Cailliez or Lingoes) to remove negative eigenvalues, then
    eigendecompose and build coordinates as V * sqrt(lambda).
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    if d.shape != (n, n):
        raise ValueError("distance_matrix must be square.")

    b = _double_centering(d**2)
    eigenvalues_raw = np.linalg.eigvalsh(b)
    n_negative_before = int(np.sum(eigenvalues_raw < -eigenvalue_tol))

    constant = 0.0
    if correction == "cailliez":
        constant = _cailliez_constant(d)
        if constant > 0.0:
            off_diag = ~np.eye(n, dtype=bool)
            d_corr = d.copy()
            d_corr[off_diag] = d_corr[off_diag] + constant
            b = _double_centering(d_corr**2)
    elif correction == "lingoes":
        constant = _lingoes_constant(b)
        if constant > 0.0:
            off_diag = ~np.eye(n, dtype=bool)
            d_corr_squared = d**2
            d_corr_squared[off_diag] = d_corr_squared[off_diag] + 2.0 * constant
            b = _double_centering(d_corr_squared)
    elif correction != "none":
        raise ValueError(f"Unknown correction: {correction!r}")

    eigenvalues, eigenvectors = np.linalg.eigh(b)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > eigenvalue_tol
    if not np.any(positive):
        raise ValueError("No positive eigenvalues found for MDS.")

    evals_pos = eigenvalues[positive]
    evecs_pos = eigenvectors[:, positive]
    coordinates = evecs_pos * np.sqrt(evals_pos)

    total = float(evals_pos.sum())
    explained = (evals_pos / total) * 100.0
    cumulative = np.cumsum(explained)

    return MDSResult(
        coordinates=coordinates,
        eigenvalues=evals_pos,
        explained_variance=explained,
        cumulative_variance=cumulative,
        n_negative_before=n_negative_before,
        correction=correction,
        correction_constant=constant,
    )
