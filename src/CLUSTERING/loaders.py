"""
loaders.py — Chargement et normalisation des coordonnées issues des réductions.

Chaque source (acp / acm / famd / aftd) produit un CSV différent ;
ce module les normalise en un DataFrame uniforme :
  - index entier (0..n-1)
  - colonnes Dim1, Dim2, …, DimN (float)

Usage :
    X, n_axes_used = load_coordinates("acp", "no_G1_G2_G3_avg",
                                      n_axes=5, variance_threshold=0.80)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent

# ── Chemins vers les CSV de coordonnées ──────────────────────────────────────

_COORD_PATHS: dict[str, str] = {
    "acp":  "ACP/results/{run}/pca_individual_coordinates.csv",
    "acm":  "ACM/results/{run}/mca_individual_coordinates.csv",
    "famd": "FAMD/outputs/{run}/individual_coordinates.csv",
    "aftd": "AFTD/results/{run}/mds_coordinates.csv",
}

_EIGEN_PATHS: dict[str, str | None] = {
    "acp":  "ACP/results/{run}/pca_eigenvalues.csv",
    "acm":  "ACM/results/{run}/mca_eigenvalues.csv",
    "famd": "FAMD/outputs/{run}/eigenvalues.csv",
    "aftd": "AFTD/results/{run}/mds_eigenvalues.csv",
}


def _resolve(template: str, run: str) -> Path:
    return SRC_DIR / template.format(run=run)


# ── Chargement brut ──────────────────────────────────────────────────────────

def _load_raw(source: str, run: str) -> pd.DataFrame:
    source = source.lower()
    if source not in _COORD_PATHS:
        raise ValueError(f"Source inconnue : {source!r}. Valeurs acceptées : {list(_COORD_PATHS)}")

    path = _resolve(_COORD_PATHS[source], run)
    if not path.exists():
        raise FileNotFoundError(
            f"Coordonnées introuvables : {path}\n"
            f"Lancez d'abord : uv run python src/{source.upper()}/... {run}"
        )
    return pd.read_csv(path)


def _normalize_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Retourne un DataFrame avec uniquement les colonnes DimN (float)."""
    if source == "famd":
        # Les colonnes de coordonnées sont nommées "0", "1", …, "9"
        dim_cols = [c for c in df.columns if str(c).lstrip("-").isdigit()]
        df = df[dim_cols].copy()
        df.columns = [f"Dim{i + 1}" for i in range(len(df.columns))]
    else:
        # ACP / ACM / AFTD : colonnes "individu" + "Dim1", "Dim2", …
        dim_cols = [c for c in df.columns if str(c).startswith("Dim")]
        df = df[dim_cols].copy()

    return df.astype(float).reset_index(drop=True)


# ── Sélection du nombre d'axes ───────────────────────────────────────────────

def _parse_variance_series(series: pd.Series) -> np.ndarray:
    """Convertit une colonne de variance (float, 0–100 ou '5.27%') en proportions 0–1."""
    raw = series.astype(str).str.strip().str.rstrip("%")
    values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    if np.nanmax(values) > 1.5:
        values = values / 100.0
    return values


def _n_axes_from_variance(source: str, run: str, threshold: float = 0.80) -> int | None:
    """Retourne le nombre minimal d'axes couvrant `threshold` de la variance."""
    eigen_tpl = _EIGEN_PATHS.get(source)
    if eigen_tpl is None:
        return None
    path = _resolve(eigen_tpl, run)
    if not path.exists():
        return None

    ev = pd.read_csv(path)
    # Nom de la colonne cumulée selon la source
    cum_col = next(
        (c for c in ev.columns if "cum" in c.lower() or "cumulative" in c.lower()),
        None,
    )
    if cum_col is None:
        return None

    cum = _parse_variance_series(ev[cum_col])
    if np.all(np.isnan(cum)):
        return None

    hits = np.where(cum >= threshold)[0]
    return int(hits[0]) + 1 if len(hits) else len(cum)


# ── Point d'entrée public ────────────────────────────────────────────────────

def load_coordinates(
    source: str,
    run: str,
    n_axes: int | None = None,
    variance_threshold: float = 0.80,
) -> tuple[np.ndarray, int]:
    """
    Charge les coordonnées individuelles d'un run de réduction.

    Paramètres
    ----------
    source : "acp" | "acm" | "famd" | "aftd"
    run    : nom du run (ex. "no_G1_G2_G3_avg")
    n_axes : nombre de dimensions à conserver. Si None, déterminé
             automatiquement d'après `variance_threshold`.
    variance_threshold : part de variance cible si n_axes non spécifié
                         (défaut 0.80 = 80 %).

    Retourne
    --------
    X           : np.ndarray (n_individuals, n_axes_used)
    n_axes_used : int
    """
    source = source.lower()
    raw = _load_raw(source, run)
    coords = _normalize_columns(raw, source)

    max_axes = coords.shape[1]

    if n_axes is None:
        n_axes = _n_axes_from_variance(source, run, variance_threshold) or max_axes

    n_axes = min(n_axes, max_axes)
    X = coords.iloc[:, :n_axes].to_numpy(dtype=float)
    return X, n_axes
