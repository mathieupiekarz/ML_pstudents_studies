"""
loaders.py — Chargement et jointure pour l'analyse supervisée.

Assemble trois sources alignées par POSITION de ligne (et non par la colonne
`individu`, dont la numérotation diffère selon la méthode) :
  1. les coordonnées factorielles de la réduction (ACP/ACM/FAMD/AFTD) ;
  2. les labels de cluster d'un run de clustering ;
  3. les variables brutes (explicatives + cibles) de data_global.csv.

Toutes les méthodes imputent les valeurs manquantes (aucune ligne supprimée),
donc l'ordre des individus est identique dans les trois fichiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent

import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from _utils import classify_column, load_data  # noqa: E402

# ── Chemins ──────────────────────────────────────────────────────────────────

_COORD_PATHS: dict[str, str] = {
    "acp":  "ACP/results/{run}/pca_individual_coordinates.csv",
    "acm":  "ACM/results/{run}/mca_individual_coordinates.csv",
    "famd": "FAMD/outputs/{run}/individual_coordinates.csv",
    "aftd": "AFTD/results/{run}/mds_coordinates.csv",
}

_CLUSTER_DIR = "CLUSTERING/results/{run}"
_VALID_METHODS = ("cah", "kmeans", "nuees")


# ── Coordonnées factorielles ──────────────────────────────────────────────────

def load_reduced(source: str, source_run: str) -> pd.DataFrame:
    """Charge les coordonnées d'une réduction, colonnes normalisées Dim1..DimN."""
    source = source.lower()
    if source not in _COORD_PATHS:
        raise ValueError(
            f"Source inconnue : {source!r}. Valeurs : {list(_COORD_PATHS)}"
        )
    path = SRC_DIR / _COORD_PATHS[source].format(run=source_run)
    if not path.exists():
        raise FileNotFoundError(
            f"Coordonnées introuvables : {path}\n"
            f"Lancez d'abord la réduction {source.upper()} sur le run {source_run!r}."
        )
    raw = pd.read_csv(path)

    if source == "famd":
        dim_cols = [c for c in raw.columns if str(c).lstrip("-").isdigit()]
        coords = raw[dim_cols].copy()
        coords.columns = [f"Dim{i + 1}" for i in range(len(dim_cols))]
    else:
        dim_cols = [c for c in raw.columns if str(c).startswith("Dim")]
        coords = raw[dim_cols].copy()

    return coords.astype(float).reset_index(drop=True)


# ── Labels de cluster ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClusterLabels:
    labels: np.ndarray
    method: str
    k: int
    path: Path


def load_labels(cluster_run: str, method: str = "cah") -> ClusterLabels:
    """Charge les labels `<method>_labels_k{k}.csv` d'un run de clustering."""
    method = method.lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Méthode inconnue : {method!r}. Valeurs : {list(_VALID_METHODS)}"
        )
    run_dir = SRC_DIR / _CLUSTER_DIR.format(run=cluster_run)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run de clustering introuvable : {run_dir}")

    matches = sorted(run_dir.glob(f"{method}_labels_k*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Aucun fichier {method}_labels_k*.csv dans {run_dir}.\n"
            f"Lancez d'abord le clustering sur ce run."
        )
    path = matches[0]
    m = re.search(r"_k(\d+)\.csv$", path.name)
    k = int(m.group(1)) if m else -1

    df = pd.read_csv(path)
    labels = df["cluster"].to_numpy()
    return ClusterLabels(labels=labels, method=method, k=k, path=path)


# ── Cibles + typage ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str  # "binaire" | "nominale" | "ordinale" | "quantitative"


def _resolve_target_type(series: pd.Series, name: str) -> str:
    override = config.TARGET_TYPES.get(name)
    if override:
        return override
    return str(classify_column(series, name)["type_retenu"])


def get_target_specs(df_global: pd.DataFrame) -> list[TargetSpec]:
    """Construit la liste des cibles configurées avec leur type."""
    specs: list[TargetSpec] = []
    target_names = config.resolve_target_vars()
    for name in target_names:
        if name not in df_global.columns:
            raise KeyError(
                f"Cible {name!r} absente du dataset. "
                f"Corrigez config.TARGET_VARS / DATASET."
            )
        specs.append(TargetSpec(name=name, kind=_resolve_target_type(df_global[name], name)))
    return specs


# ── Assemblage ────────────────────────────────────────────────────────────────

@dataclass
class JoinedData:
    frame: pd.DataFrame          # coords Dim* + cluster_id + cibles + explicatives
    dim_cols: list[str]
    feature_cols: list[str]      # explicatives prétraitées (hors cibles, hors cluster_id)
    target_specs: list[TargetSpec]
    cluster_in_reduction: list[str]  # cibles présentes dans la réduction (fuite)
    n_dropped_na: dict[str, int]
    excluded_vars: list[str] = field(default_factory=list)


def _reduction_source_columns(source_run_used: bool = True) -> set[str]:
    """Colonnes effectivement entrées dans la réduction (après prétraitement)."""
    df = config.apply_preprocessing(load_data())
    return set(df.columns)


def build_joined(
    source: str,
    source_run: str,
    cluster_run: str,
    method: str = "cah",
) -> tuple[JoinedData, ClusterLabels]:
    """
    Assemble les trois sources par position et applique les contrôles A.1.
    """
    coords = load_reduced(source, source_run)
    cl = load_labels(cluster_run, method)
    df_raw = load_data().reset_index(drop=True)

    n = len(df_raw)
    if not (len(coords) == n == len(cl.labels)):
        raise ValueError(
            "Tailles incohérentes entre les sources : "
            f"coords={len(coords)}, labels={len(cl.labels)}, global={n}. "
            "Les fichiers doivent provenir du même jeu de 382 individus."
        )

    target_specs = get_target_specs(df_raw)
    target_names = [t.name for t in target_specs]

    # Explicatives : même prétraitement que ACP/ACM/FAMD/AFTD (INCLUDE_VARS + AVERAGE_MAT_POR).
    df_features = config.apply_preprocessing(df_raw)
    feature_cols = list(df_features.columns)
    targets_df = df_raw[target_names].reset_index(drop=True)

    # Variables brutes exclues (INCLUDE_VARS=False), hors cibles conservées pour prédiction.
    excluded_vars = sorted(
        c for c, keep in config.resolve_columns(config.DATASET, config.INCLUDE_VARS).items()
        if not keep and c in df_raw.columns and c not in target_names
    )

    # Détection de fuite : cible présente dans la réduction.
    reduction_cols = _reduction_source_columns()
    cluster_in_reduction = [t for t in target_names if t in reduction_cols]

    # Assemblage positionnel.
    frame = pd.concat(
        [
            coords.reset_index(drop=True),
            pd.Series(cl.labels, name="cluster_id"),
            df_features.reset_index(drop=True),
            targets_df,
        ],
        axis=1,
    )

    # Contrôles : doublons d'index, NA sur cibles.
    n_dropped_na: dict[str, int] = {}
    for t in target_names:
        na = int(frame[t].isna().sum())
        n_dropped_na[t] = na

    dim_cols = list(coords.columns)
    joined = JoinedData(
        frame=frame,
        dim_cols=dim_cols,
        feature_cols=feature_cols,
        target_specs=target_specs,
        cluster_in_reduction=cluster_in_reduction,
        n_dropped_na=n_dropped_na,
        excluded_vars=excluded_vars,
    )
    return joined, cl


def target_frame(joined: JoinedData, target: TargetSpec) -> pd.DataFrame:
    """Sous-DataFrame sans NA sur la cible (drop + comptage déjà fait en amont)."""
    cols = joined.dim_cols + ["cluster_id"] + joined.feature_cols + [target.name]
    sub = joined.frame[cols].copy()
    sub = sub.dropna(subset=[target.name]).reset_index(drop=True)
    return sub
