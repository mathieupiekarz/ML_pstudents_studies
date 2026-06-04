"""
config.py — Paramètres partagés pour ACP, ACM et FAMD.

Modifie les deux flags ci-dessous pour changer le mode d'analyse.
Les noms de dossiers de résultats s'adaptent automatiquement.
"""

# ── Mode 1 : retirer G1 et G2 (garder G3 en variable supplémentaire) ─────────
REMOVE_G1_G2: bool = True

# ── Mode 2 : moyenner les variables dupliquées .m / .p ───────────────────────
# Ex : Dalc = (Dalc.m + Dalc.p) / 2  →  une seule variable par concept
AVERAGE_MAT_POR: bool = True

# ─────────────────────────────────────────────────────────────────────────────
# Variables G1/G2 à exclure quand REMOVE_G1_G2 = True
G1_G2_COLS = [
    "G1.m", "G2.m",   # maths
    "G1.p", "G2.p",   # portugais
]

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
    if REMOVE_G1_G2:
        parts.append("no_g1g2")
    if AVERAGE_MAT_POR:
        parts.append("avg")
    return ("_" + "_".join(parts)) if parts else "_full"


def apply_preprocessing(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Applique les transformations configurées au DataFrame.
    À appeler dans chaque script d'analyse après load_data().
    """
    import pandas as pd

    df = df.copy()

    # 1. Moyenner les paires .m / .p
    if AVERAGE_MAT_POR:
        for col_m, col_p, col_out in PAIRS_TO_AVERAGE:
            if col_m in df.columns and col_p in df.columns:
                df[col_out] = df[[col_m, col_p]].mean(axis=1)
                df = df.drop(columns=[col_m, col_p])

    # 2. Retirer G1 et G2
    if REMOVE_G1_G2:
        cols_to_drop = [c for c in G1_G2_COLS if c in df.columns]
        # Si on a moyenné, G1/G2 n'existent plus sous .m/.p
        # mais on vérifie aussi "G1" et "G2" sans suffixe
        cols_to_drop += [c for c in ["G1", "G2"] if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    return df