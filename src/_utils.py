"""
Utilitaires partagés pour l'exploration factorielle ACP / ACM.

Exploration globale : aucune colonne exclue (G3 incluse selon la typologie).
L'ACP porte sur les variables quantitatives ; l'ACM sur qualitatives et binaires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Seuil : au-delà, une variable numérique est traitée comme quantitative (ACP).
MAX_MODALITIES = 6

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "data_global.csv"

ACP_RESULTS_DIR = SRC_DIR / "ACP" / "results"
ACM_RESULTS_DIR = SRC_DIR / "ACM" / "results"
SHARED_RESULTS_DIR = SRC_DIR / "shared" / "results"

SUMMARY_PATH = SHARED_RESULTS_DIR / "factor_analysis_summary.txt"
TYPOLOGY_PATH = SHARED_RESULTS_DIR / "variable_typology.csv"
EXPLORATION_GUIDE_PATH = SHARED_RESULTS_DIR / "exploration_guide.txt"


def results_dir_for(method: Literal["ACP", "ACM"]) -> Path:
    return ACP_RESULTS_DIR if method == "ACP" else ACM_RESULTS_DIR


def ensure_results_dir(method: Literal["ACP", "ACM"]) -> Path:
    SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = results_dir_for(method)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def _initial_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "text"


def classify_column(series: pd.Series, name: str) -> dict[str, str | int]:
    """
    Détermine le type retenu et l'analyse cible (ACP ou ACM).

    Les numériques à peu de modalités (≤ MAX_MODALITIES) vont en ACM
    (échelles ordinales, binaires codées en nombres).
    """
    type_initial = _initial_type(series)
    n_mod = int(series.nunique(dropna=True))

    if type_initial == "text" or pd.api.types.is_object_dtype(series):
        if n_mod <= 2:
            type_retenu = "binaire"
            raison = f"texte, {n_mod} modalité(s) → binaire qualitative"
        else:
            type_retenu = "qualitative"
            raison = f"texte, {n_mod} modalités → qualitative"
        utilisee = "ACM"
    elif n_mod <= 2:
        type_retenu = "binaire"
        raison = f"numérique, {n_mod} modalité(s) → binaire qualitative"
        utilisee = "ACM"
    elif n_mod <= MAX_MODALITIES:
        type_retenu = "qualitative"
        raison = (
            f"numérique, {n_mod} modalités (≤ {MAX_MODALITIES}) "
            "→ qualitative ordinale"
        )
        utilisee = "ACM"
    else:
        type_retenu = "quantitative"
        raison = (
            f"numérique, {n_mod} modalités (> {MAX_MODALITIES}) "
            "→ quantitative"
        )
        utilisee = "ACP"

    return {
        "variable": name,
        "type_initial": type_initial,
        "nombre_modalites": n_mod,
        "type_retenu": type_retenu,
        "utilisee_dans": utilisee,
        "raison": raison,
    }


def build_typology(df: pd.DataFrame) -> pd.DataFrame:
    """Parcourt toutes les colonnes ; aucune exclusion métier."""
    SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [classify_column(df[col], col) for col in df.columns]
    typology = pd.DataFrame(rows)
    typology.to_csv(TYPOLOGY_PATH, index=False)
    return typology


def get_acp_columns(typology: pd.DataFrame) -> list[str]:
    return typology.loc[typology["utilisee_dans"] == "ACP", "variable"].tolist()


def get_acm_columns(typology: pd.DataFrame) -> list[str]:
    return typology.loc[typology["utilisee_dans"] == "ACM", "variable"].tolist()


def impute_quantitative(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Imputation par la médiane pour l'ACP (données centrées-réduites ensuite)."""
    out = df[columns].copy()
    imputer = SimpleImputer(strategy="median")
    out.iloc[:, :] = imputer.fit_transform(out)
    return out


def impute_qualitative(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Imputation par la modalité la plus fréquente pour l'ACM."""
    out = df[columns].copy().astype(str)
    imputer = SimpleImputer(strategy="most_frequent")
    out.iloc[:, :] = imputer.fit_transform(out)
    return out.astype("category")


def n_axes_for_threshold(cum_inertia: np.ndarray, threshold: float) -> int:
    """
    Premier nombre d'axes dont l'inertie cumulée atteint threshold (ex. 0.90).

    Les seuils 90 % / 95 % aident à choisir combien d'axes regarder ;
    ce n'est pas une règle absolue pour l'interprétation.
    """
    idx = np.where(cum_inertia >= threshold)[0]
    return int(idx[0] + 1) if len(idx) else len(cum_inertia)


def compute_pca_contributions(
    components: np.ndarray, explained_variance: np.ndarray
) -> pd.DataFrame:
    """Contributions des variables (%) par axe, style ACP sur données standardisées."""
    loadings = components.T * np.sqrt(explained_variance)
    n_axes = loadings.shape[1]
    contrib = np.zeros_like(loadings)
    for k in range(n_axes):
        col_sq = loadings[:, k] ** 2
        total = col_sq.sum()
        contrib[:, k] = (col_sq / total * 100) if total > 0 else 0
    return pd.DataFrame(
        contrib,
        columns=[f"Dim{i + 1}" for i in range(n_axes)],
    )


def compute_mca_modality_contributions(
    column_coords: pd.DataFrame,
) -> pd.DataFrame:
    """Contributions des modalités (%) par axe à partir des coordonnées colonnes."""
    arr = column_coords.values.astype(float)
    n_axes = arr.shape[1]
    contrib = np.zeros_like(arr)
    for k in range(n_axes):
        col_sq = arr[:, k] ** 2
        total = col_sq.sum()
        contrib[:, k] = (col_sq / total * 100) if total > 0 else 0
    out = pd.DataFrame(
        contrib,
        index=column_coords.index,
        columns=[f"Dim{i + 1}" for i in range(n_axes)],
    )
    out.index.name = "modalite"
    return out


def top_contributors(
    contrib: pd.DataFrame, dim: int = 0, n: int = 5
) -> list[str]:
    col = f"Dim{dim + 1}"
    if col not in contrib.columns:
        return []
    s = contrib[col].sort_values(ascending=False).head(n)
    index = s.index.astype(str) if hasattr(s.index, "astype") else s.index
    return [f"{idx} ({val:.1f}%)" for idx, val in zip(index, s.values)]


def modality_variable_label(modality: str) -> str:
    """Extrait le nom de variable depuis le libellé prince (var__modalité)."""
    if "__" in modality:
        return modality.split("__", 1)[0]
    return modality


def write_summary_section(title: str, lines: list[str]) -> None:
    """Ajoute ou remplace une section dans factor_analysis_summary.txt."""
    SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    marker_start = f"=== {title} ==="
    marker_end = f"=== fin {title} ==="
    block = marker_start + "\n" + "\n".join(lines) + "\n" + marker_end + "\n"

    if SUMMARY_PATH.exists():
        text = SUMMARY_PATH.read_text(encoding="utf-8")
        if marker_start in text:
            before = text.split(marker_start)[0]
            after_parts = text.split(marker_end, 1)
            after = after_parts[1] if len(after_parts) > 1 else ""
            text = before + block + after
        else:
            text = text + "\n" + block
    else:
        header = (
            "Résumé — analyse factorielle exploratoire (data_global.csv)\n"
            "Variables exclues : aucune (exploration globale)\n\n"
        )
        text = header + block

    SUMMARY_PATH.write_text(text, encoding="utf-8")


def write_exploration_guide_section(title: str, content: str) -> None:
    import re

    SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    block = f"--- {title} ---\n{content.strip()}\n\n"
    intro = (
        "Guide de lecture — exploration factorielle\n"
        "Objectif : choisir quels axes et quelles analyses poursuivre.\n\n"
    )

    if EXPLORATION_GUIDE_PATH.exists():
        text = EXPLORATION_GUIDE_PATH.read_text(encoding="utf-8")
        pattern = re.compile(
            rf"--- {re.escape(title)} ---.*?(?=\n--- |\Z)",
            re.DOTALL,
        )
        if pattern.search(text):
            text = pattern.sub(block, text)
        else:
            text = text.rstrip() + "\n\n" + block
    else:
        text = intro + block

    EXPLORATION_GUIDE_PATH.write_text(text, encoding="utf-8")


def print_exploration_footer(
    method: Literal["ACP", "ACM"],
    n_vars: int,
    n_axes_90: int,
    n_axes_95: int,
    top_dim1: list[str],
    top_dim2: list[str],
) -> None:
    results = results_dir_for(method)
    print(f"\n=== {method} — exploration ===")
    print(f"Variables : {n_vars}")
    print(f"Axes conseillés (90 % / 95 % inertie cumulée) : {n_axes_90} / {n_axes_95}")
    print(f"Plus contributifs axe 1 : {', '.join(top_dim1) or '—'}")
    print(f"Plus contributifs axe 2 : {', '.join(top_dim2) or '—'}")
    print(
        "Figures prioritaires : scree, cumulative, contributions, "
        + ("biplot" if method == "ACP" else "asymmetric map")
    )
    print(f"Résultats {method} : {results.resolve()}")
    print(f"Fichiers partagés : {SHARED_RESULTS_DIR.resolve()}")
