"""
config.py — Paramètres partagés pour ACP, ACM, FAMD et AFTD.

Deux leviers de prétraitement, appliqués après load_data() via apply_preprocessing :
  1. INCLUDE_VARS  — un flag True/False par colonne brute (garder / retirer).
  2. AVERAGE_MAT_POR — moyenne des paires .m / .p en une seule variable.

Les noms de dossiers de résultats s'adaptent automatiquement (results_suffix).
"""

# ── Inclusion par variable ───────────────────────────────────────────────────
# True = variable gardée, False = retirée avant l'analyse.
INCLUDE_VARS: dict[str, bool] = {
    # Identité / socio-démographie
    "school": True, "sex": True, "age": True, "address": True,
    "famsize": True, "Pstatus": True, "Medu": True, "Fedu": True,
    "Mjob": True, "Fjob": True, "reason": True, "nursery": True,
    "internet": True,
    # Maths (.m)
    "guardian.m": True, "traveltime.m": True, "studytime.m": True,
    "failures.m": True, "schoolsup.m": True, "famsup.m": True,
    "paid.m": True, "activities.m": True, "higher.m": True,
    "romantic.m": True, "famrel.m": True, "freetime.m": True,
    "goout.m": True, "Dalc.m": True, "Walc.m": True, "health.m": True,
    "absences.m": True, "G1.m": False, "G2.m": False, "G3.m": True,
    # Portugais (.p)
    "guardian.p": True, "traveltime.p": True, "studytime.p": True,
    "failures.p": True, "schoolsup.p": True, "famsup.p": True,
    "paid.p": True, "activities.p": True, "higher.p": True,
    "romantic.p": True, "famrel.p": True, "freetime.p": True,
    "goout.p": True, "Dalc.p": True, "Walc.p": True, "health.p": True,
    "absences.p": True, "G1.p": False, "G2.p": False, "G3.p": True,
}

# ── Mode : moyenner les variables dupliquées .m / .p ─────────────────────────
# Ex : Dalc = (Dalc.m + Dalc.p) / 2  →  une seule variable par concept
AVERAGE_MAT_POR: bool = True

# Variables à moyenner quand AVERAGE_MAT_POR = True
# Chaque tuple : (col_mat, col_por, nom_résultant)
PAIRS_TO_AVERAGE = [
    ("Dalc.m",      "Dalc.p",      "Dalc"),
    ("Walc.m",      "Walc.p",      "Walc"),
    ("goout.m",     "goout.p",     "goout"),
    ("freetime.m",  "freetime.p",  "freetime"),
    ("studytime.m", "studytime.p", "studytime"),
    ("failures.m",  "failures.p",  "failures"),
    ("absences.m",  "absences.p",  "absences"),
    ("traveltime.m","traveltime.p","traveltime"),
    ("famrel.m",    "famrel.p",    "famrel"),
    ("health.m",    "health.p",    "health"),
    ("G3.m",        "G3.p",        "G3"),
]

# ─────────────────────────────────────────────────────────────────────────────


def results_suffix() -> str:
    """Suffixe ajouté au dossier results selon le mode actif."""
    parts = []
    if AVERAGE_MAT_POR:
        parts.append("avg")
    n_excl = sum(1 for v in INCLUDE_VARS.values() if not v)
    if n_excl:
        parts.append(f"excl{n_excl}")
    return ("_" + "_".join(parts)) if parts else "_full"


def check_include_vars(df: "pd.DataFrame") -> None:
    """Avertit (sans bloquer) si INCLUDE_VARS ne couvre pas exactement les colonnes."""
    inconnus = set(INCLUDE_VARS) - set(df.columns)
    manquants = set(df.columns) - set(INCLUDE_VARS)
    if inconnus or manquants:
        print(
            f"[config] clés inconnues : {sorted(inconnus)} ; "
            f"colonnes sans flag : {sorted(manquants)}"
        )


def apply_preprocessing(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Applique les transformations configurées au DataFrame.
    À appeler dans chaque script d'analyse après load_data().

    Ordre : 1) retirer les colonnes désactivées, 2) moyenner les paires .m/.p.
    """
    df = df.copy()

    # 1. Retirer les colonnes désactivées
    to_drop = [c for c, keep in INCLUDE_VARS.items() if not keep and c in df.columns]
    df = df.drop(columns=to_drop)

    # 2. Moyenner les paires .m / .p si les DEUX sont encore présentes
    if AVERAGE_MAT_POR:
        for col_m, col_p, col_out in PAIRS_TO_AVERAGE:
            if col_m in df.columns and col_p in df.columns:
                df[col_out] = df[[col_m, col_p]].mean(axis=1)
                df = df.drop(columns=[col_m, col_p])

    return df
