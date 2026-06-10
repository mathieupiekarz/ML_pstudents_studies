"""
Utilitaires partagés pour l'exploration factorielle ACP / ACM / FAMD / ACP_mixte / AFTD.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

MAX_MODALITIES = 6

BINARY_BASE = {
    "school", "sex", "address", "famsize", "Pstatus", "nursery", "internet",
    "schoolsup", "famsup", "paid", "activities", "higher", "romantic",
}
NOMINAL_BASE = {"Mjob", "Fjob", "reason", "guardian"}
ORDINAL_BASE = {
    "Medu", "Fedu", "traveltime", "studytime",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
}
QUANTITATIVE_BASE = {"age", "failures", "absences", "G1", "G2", "G3"}


def _base_name(name: str) -> str:
    return re.sub(r"\.(m|p)$", "", name)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
SHARED_RESULTS_DIR = SRC_DIR / "shared" / "results"
SUMMARY_PATH = SHARED_RESULTS_DIR / "factor_analysis_summary.txt"
TYPOLOGY_PATH = SHARED_RESULTS_DIR / "variable_typology.csv"
EXPLORATION_GUIDE_PATH = SHARED_RESULTS_DIR / "exploration_guide.txt"


def get_data_path(dataset: str | None = None) -> Path:
    import config
    ds = dataset or config.DATASET
    if ds not in config.DATASET_PATHS:
        raise ValueError(f"Dataset inconnu : {ds!r}")
    return PROJECT_ROOT / "data" / config.DATASET_PATHS[ds]


def get_run_name(argv: list[str] | None = None) -> str:
    args = sys.argv[1:] if argv is None else argv
    if not args or not args[0].strip():
        raise SystemExit(
            "Erreur : nom de sauvegarde requis.\nUsage : <script> <nom_du_run>"
        )
    name = args[0].strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise SystemExit(
            "Nom invalide : lettres, chiffres, '.', '_', '-' uniquement."
        )
    return name


def load_data(dataset: str | None = None) -> pd.DataFrame:
    import config
    path = get_data_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    config.check_include_vars(df, dataset=dataset)
    return df


def _initial_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "text"


def _classify_by_heuristic(n_mod: int, type_initial: str) -> tuple[str, str]:
    if type_initial == "text":
        if n_mod <= 2:
            return "binaire", f"texte, {n_mod} modalité(s) → binaire (repli)"
        return "nominale", f"texte, {n_mod} modalités → nominale (repli)"
    if n_mod <= 2:
        return "binaire", f"numérique, {n_mod} modalité(s) → binaire (repli)"
    if n_mod <= MAX_MODALITIES:
        return "ordinale", f"numérique, {n_mod} modalités → ordinale (repli)"
    return "quantitative", f"numérique, {n_mod} modalités → quantitative (repli)"


def classify_column(series: pd.Series, name: str) -> dict[str, str | int]:
    type_initial = _initial_type(series)
    n_mod = int(series.nunique(dropna=True))
    base = _base_name(name)

    if base in BINARY_BASE:
        type_retenu = "binaire"
        raison = f"règle explicite : {base} → binaire"
    elif base in NOMINAL_BASE:
        type_retenu = "nominale"
        raison = f"règle explicite : {base} → qualitative nominale"
    elif base in ORDINAL_BASE:
        type_retenu = "ordinale"
        raison = f"règle explicite : {base} → qualitative ordinale"
    elif base in QUANTITATIVE_BASE:
        type_retenu = "quantitative"
        raison = f"règle explicite : {base} → quantitative"
    else:
        type_retenu, raison = _classify_by_heuristic(n_mod, type_initial)

    utilisee = "ACP" if type_retenu == "quantitative" else "ACM"

    return {
        "variable": name,
        "type_initial": type_initial,
        "nombre_modalites": n_mod,
        "type_retenu": type_retenu,
        "utilisee_dans": utilisee,
        "raison": raison,
    }


def build_typology(df: pd.DataFrame) -> pd.DataFrame:
    SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [classify_column(df[col], col) for col in df.columns]
    typology = pd.DataFrame(rows)
    typology.to_csv(TYPOLOGY_PATH, index=False)
    return typology


def get_acp_columns(typology: pd.DataFrame) -> list[str]:
    return typology.loc[typology["utilisee_dans"] == "ACP", "variable"].tolist()


def get_acm_columns(typology: pd.DataFrame) -> list[str]:
    return typology.loc[typology["utilisee_dans"] == "ACM", "variable"].tolist()


def get_acp_mixte_columns(typology: pd.DataFrame) -> tuple[list[str], list[str]]:
    quant = typology.loc[typology["type_retenu"] == "quantitative", "variable"].tolist()
    qual = typology.loc[typology["type_retenu"] != "quantitative", "variable"].tolist()
    return quant, qual


def _columns_of_type(typology: pd.DataFrame, type_retenu: str) -> list[str]:
    return typology.loc[typology["type_retenu"] == type_retenu, "variable"].tolist()


def get_aftd_groups(typology: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "quantitative": _columns_of_type(typology, "quantitative"),
        "binary": _columns_of_type(typology, "binaire"),
        "nominal": _columns_of_type(typology, "nominale"),
        "ordinal": _columns_of_type(typology, "ordinale"),
    }


def get_famd_groups(typology: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = _columns_of_type(typology, "quantitative")
    binary = _columns_of_type(typology, "binaire")
    categorical = _columns_of_type(typology, "nominale") + _columns_of_type(
        typology, "ordinale"
    )
    return numeric, categorical, binary


def detect_method_set(typology: pd.DataFrame) -> list[str]:
    has_quant = bool(_columns_of_type(typology, "quantitative"))
    has_qual = bool(
        _columns_of_type(typology, "binaire")
        + _columns_of_type(typology, "nominale")
        + _columns_of_type(typology, "ordinale")
    )
    if has_quant and has_qual:
        return ["AFTD", "FAMD", "ACP_mixte"]
    if has_quant:
        return ["ACP", "AFTD", "FAMD"]
    if has_qual:
        return ["ACM", "AFTD", "FAMD"]
    raise ValueError("Aucune variable active pour l'analyse factorielle.")


# Méthodes exclues du notebook (AFTD : trop d'axes, graphiques illisibles).
NOTEBOOK_EXCLUDED_METHODS: frozenset[str] = frozenset({"AFTD"})


def detect_notebook_method_set(typology: pd.DataFrame) -> list[str]:
    """Comme detect_method_set, sans les méthodes exclues du notebook."""
    return [m for m in detect_method_set(typology) if m not in NOTEBOOK_EXCLUDED_METHODS]


def n_axes_for_threshold(cum_inertia: np.ndarray, threshold: float) -> int:
    idx = np.where(cum_inertia >= threshold)[0]
    return int(idx[0] + 1) if len(idx) else len(cum_inertia)


def n_axes_for_thresholds(
    cum: np.ndarray, thresholds: tuple[float, ...] = (0.60, 0.80, 0.90)
) -> dict[str, int]:
    return {f"{int(t * 100)}%": n_axes_for_threshold(cum, t) for t in thresholds}


def impute_quantitative(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].copy()
    imputer = SimpleImputer(strategy="median")
    out.iloc[:, :] = imputer.fit_transform(out)
    return out


def impute_qualitative(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].copy().astype(str)
    imputer = SimpleImputer(strategy="most_frequent")
    out.iloc[:, :] = imputer.fit_transform(out)
    return out.astype("category")


def compute_pca_contributions(
    components: np.ndarray, explained_variance: np.ndarray
) -> pd.DataFrame:
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


def compute_mca_modality_contributions(column_coords: pd.DataFrame) -> pd.DataFrame:
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


def top_contributors(contrib: pd.DataFrame, dim: int = 0, n: int = 5) -> list[str]:
    col = f"Dim{dim + 1}"
    if col not in contrib.columns:
        return []
    s = contrib[col].sort_values(ascending=False).head(n)
    index = s.index.astype(str) if hasattr(s.index, "astype") else s.index
    return [f"{idx} ({val:.1f}%)" for idx, val in zip(index, s.values)]


def modality_variable_label(modality: str) -> str:
    if "__" in modality:
        return modality.split("__", 1)[0]
    return modality


def write_summary_section(title: str, lines: list[str]) -> None:
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
            "Résumé — analyse factorielle exploratoire\n"
            "Variables exclues : selon INCLUDE_VARS\n\n"
        )
        text = header + block

    SUMMARY_PATH.write_text(text, encoding="utf-8")


def write_exploration_guide_section(title: str, content: str) -> None:
    import re as _re

    SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    block = f"--- {title} ---\n{content.strip()}\n\n"
    intro = (
        "Guide de lecture — exploration factorielle\n"
        "Objectif : choisir quels axes et quelles analyses poursuivre.\n\n"
    )

    if EXPLORATION_GUIDE_PATH.exists():
        text = EXPLORATION_GUIDE_PATH.read_text(encoding="utf-8")
        pattern = _re.compile(
            rf"--- {_re.escape(title)} ---.*?(?=\n--- |\Z)",
            _re.DOTALL,
        )
        if pattern.search(text):
            text = pattern.sub(block, text)
        else:
            text = text.rstrip() + "\n\n" + block
    else:
        text = intro + block

    EXPLORATION_GUIDE_PATH.write_text(text, encoding="utf-8")


def print_exploration_footer(
    method: Literal["ACP", "ACM", "ACP_mixte"],
    n_vars: int,
    n_axes_90: int,
    n_axes_95: int,
    top_dim1: list[str],
    top_dim2: list[str],
    results_dir: Path,
) -> None:
    print(f"\n=== {method} — exploration ===")
    print(f"Variables : {n_vars}")
    print(f"Axes conseillés (90 % / 95 % inertie cumulée) : {n_axes_90} / {n_axes_95}")
    print(f"Plus contributifs axe 1 : {', '.join(top_dim1) or '—'}")
    print(f"Plus contributifs axe 2 : {', '.join(top_dim2) or '—'}")
    print(
        "Figures prioritaires : scree, cumulative, contributions, "
        + ("biplot" if method in ("ACP", "ACP_mixte") else "asymmetric map")
    )
    if results_dir:
        print(f"Résultats {method} : {results_dir.resolve()}")
    print(f"Fichiers partagés : {SHARED_RESULTS_DIR.resolve()}")
