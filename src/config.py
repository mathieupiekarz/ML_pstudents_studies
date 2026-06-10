"""
config.py — Paramètres partagés pour ACP, ACM, FAMD, AFTD et ACP_mixte.

Leviers de prétraitement (apply_preprocessing) :
  1. DATASET        — student-mat | student-por | data_global
  2. INCLUDE_VARS   — flags canoniques (33 concepts) ; expansion .m/.p pour data_global
  3. AVERAGE_MAT_POR — moyenne des paires .m / .p (data_global uniquement)
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

# ── Dataset ──────────────────────────────────────────────────────────────────

DATASET: Literal["student-mat", "student-por", "data_global"] = "data_global"

DATASET_PATHS: dict[str, str] = {
    "student-mat": "student-mat.csv",
    "student-por": "student-por.csv",
    "data_global": "data_global.csv",
}

DATASET_LABELS: dict[str, str] = {
    "student-mat": "Mathématiques",
    "student-por": "Portugais",
    "data_global": "Global (mat + por)",
}

# ── Colonnes partagées / matière (data_global) ───────────────────────────────

SHARED_COLUMNS: list[str] = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet",
]

SUBJECT_COLUMNS: list[str] = [
    "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "higher", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2", "G3",
]

# ── Inclusion par variable (noms canoniques) ─────────────────────────────────

INCLUDE_VARS: dict[str, bool] = {
    "school": True, "sex": True, "age": True, "address": True,
    "famsize": True, "Pstatus": True, "Medu": True, "Fedu": True,
    "Mjob": True, "Fjob": True, "reason": True, "nursery": True,
    "internet": True,
    "guardian": True, "traveltime": True, "studytime": True,
    "failures": True, "schoolsup": True, "famsup": True,
    "paid": True, "activities": True, "higher": True,
    "romantic": True, "famrel": True, "freetime": True,
    "goout": True, "Dalc": True, "Walc": True, "health": True,
    "absences": True, "G1": False, "G2": False, "G3": False,
}

# ── Décrochage scolaire (G3 = 0) ─────────────────────────────────────────────

FOCUS_VAR: str = "G3"
DROPOUT_VALUE: int = 0
DROPOUT_LABELS: dict[bool, str] = {
    True: "Décrochage (G3=0)",
    False: "Non-décrochage (G3>0)",
}

# ── Cibles supervisées ────────────────────────────────────────────────────────

TARGET_VARS: list[str] = ["G3"]
TARGET_TYPES: dict[str, str] = {}

ACTIONABLE_VARS: list[str] = [
    "studytime", "schoolsup", "paid", "activities",
    "goout", "Dalc", "Walc", "absences", "internet", "higher",
]

# ── Moyennage .m / .p ────────────────────────────────────────────────────────

AVERAGE_MAT_POR: bool = False

PAIRS_TO_AVERAGE: list[tuple[str, str, str]] = [
    ("Dalc.m", "Dalc.p", "Dalc"),
    ("Walc.m", "Walc.p", "Walc"),
    ("goout.m", "goout.p", "goout"),
    ("freetime.m", "freetime.p", "freetime"),
    ("studytime.m", "studytime.p", "studytime"),
    ("failures.m", "failures.p", "failures"),
    ("absences.m", "absences.p", "absences"),
    ("traveltime.m", "traveltime.p", "traveltime"),
    ("famrel.m", "famrel.p", "famrel"),
    ("health.m", "health.p", "health"),
    ("G1.m", "G1.p", "G1"),
    ("G2.m", "G2.p", "G2"),
    ("G3.m", "G3.p", "G3"),
]

# ── Migration ancien format 53 clés ──────────────────────────────────────────


def _is_legacy_include_vars(include_vars: dict[str, bool]) -> bool:
    return any(k.endswith(".m") or k.endswith(".p") for k in include_vars)


def migrate_include_vars(old: dict[str, bool]) -> dict[str, bool]:
    """Convertit l'ancien format .m/.p vers noms canoniques."""
    if not _is_legacy_include_vars(old):
        return dict(old)
    canonical = {k: True for k in SHARED_COLUMNS + SUBJECT_COLUMNS}
    for col, keep in old.items():
        base = col
        if col.endswith(".m") or col.endswith(".p"):
            base = col.rsplit(".", 1)[0]
        if base in canonical:
            canonical[base] = canonical[base] and keep
    return canonical


# ── Résolution colonnes ──────────────────────────────────────────────────────


def resolve_columns(
    dataset: str,
    include_vars: dict[str, bool] | None = None,
) -> dict[str, bool]:
    """Mappe les flags canoniques vers les noms de colonnes réels du CSV."""
    flags = migrate_include_vars(include_vars or INCLUDE_VARS)

    if dataset in ("student-mat", "student-por"):
        return {k: flags.get(k, True) for k in SHARED_COLUMNS + SUBJECT_COLUMNS
                if k in flags}

    resolved: dict[str, bool] = {}
    for col in SHARED_COLUMNS:
        resolved[col] = flags.get(col, True)
    for base in SUBJECT_COLUMNS:
        keep = flags.get(base, True)
        resolved[f"{base}.m"] = keep
        resolved[f"{base}.p"] = keep
    return resolved


def resolve_target_vars(
    dataset: str | None = None,
    average_mat_por: bool | None = None,
) -> list[str]:
    ds = dataset or DATASET
    avg = AVERAGE_MAT_POR if average_mat_por is None else average_mat_por
    if ds == "data_global" and not avg:
        return ["G3.m", "G3.p"]
    return ["G3"]


def resolve_g3_column(
    dataset: str | None = None,
    average_mat_por: bool | None = None,
) -> str | list[str]:
    ds = dataset or DATASET
    avg = AVERAGE_MAT_POR if average_mat_por is None else average_mat_por
    if ds in ("student-mat", "student-por"):
        return "G3"
    if avg:
        return "G3"
    return ["G3.m", "G3.p"]


def extract_dropout_mask(df_raw: pd.DataFrame, g3_col: str) -> pd.Series:
    return df_raw[g3_col] == DROPOUT_VALUE


def results_suffix(
    dataset: str | None = None,
    include_vars: dict[str, bool] | None = None,
    average_mat_por: bool | None = None,
) -> str:
    ds = dataset or DATASET
    flags = migrate_include_vars(include_vars or INCLUDE_VARS)
    avg = AVERAGE_MAT_POR if average_mat_por is None else average_mat_por
    parts: list[str] = []
    ds_tag = {"student-mat": "mat", "student-por": "por", "data_global": "global"}
    parts.append(ds_tag.get(ds, ds))
    if avg and ds == "data_global":
        parts.append("avg")
    n_excl = sum(1 for v in flags.values() if not v)
    if n_excl:
        parts.append(f"excl{n_excl}")
    return "_" + "_".join(parts)


def check_include_vars(
    df: pd.DataFrame,
    dataset: str | None = None,
    include_vars: dict[str, bool] | None = None,
) -> None:
    ds = dataset or DATASET
    resolved = resolve_columns(ds, include_vars)
    inconnus = set(resolved) - set(df.columns)
    manquants = set(df.columns) - set(resolved)
    if inconnus or manquants:
        print(
            f"[config] clés inconnues : {sorted(inconnus)} ; "
            f"colonnes sans flag : {sorted(manquants)}"
        )


def apply_preprocessing(
    df: pd.DataFrame,
    *,
    dataset: str | None = None,
    include_vars: dict[str, bool] | None = None,
    average_mat_por: bool | None = None,
) -> pd.DataFrame:
    ds = dataset or DATASET
    flags = resolve_columns(ds, include_vars)
    avg = (AVERAGE_MAT_POR if average_mat_por is None else average_mat_por)
    if ds != "data_global":
        avg = False

    df = df.copy()
    to_drop = [c for c, keep in flags.items() if not keep and c in df.columns]
    df = df.drop(columns=to_drop)

    if avg:
        for col_m, col_p, col_out in PAIRS_TO_AVERAGE:
            if col_m in df.columns and col_p in df.columns:
                df[col_out] = df[[col_m, col_p]].mean(axis=1)
                df = df.drop(columns=[col_m, col_p])
    return df
