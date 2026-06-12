#!/usr/bin/env python3
"""Rebuild 03_cluster_supervise.ipynb from 01_v2 (§1–9) + validation tail (§10–16).

Canonical workflow: edit 03_cluster_supervise.ipynb directly for day-to-day work.
Run this script to regenerate from sources after major structural changes:

    uv run python scripts/build_nb03.py

Sources:
  - 01_reduction_clustering_v2.ipynb  → sections 1–9 (VARIANCE_THRESHOLD, K auto)
  - 03_cluster_supervise.ipynb        → sections 10–16 (validation), kept from cell 35+
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "01_reduction_clustering_v2.ipynb"
NB03_PATH = ROOT / "03_cluster_supervise.ipynb"
SECTION10_START_MARKER = "## 10. Validation statistique"


def _cell_src(c: dict) -> str:
    return "".join(c.get("source", []))


def _set_src(c: dict, text: str) -> None:
    c["source"] = [text]
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None


def _fresh(c: dict) -> dict:
    nc = copy.deepcopy(c)
    if nc["cell_type"] == "code":
        nc["outputs"] = []
        nc["execution_count"] = None
    return nc


GOWER_HELPERS = """
# Helpers Gower (réutilisés sections 14–15)
ranges_num = {col: (mat_full[col].min(), mat_full[col].max()) for col in VARS_NUM}


def encode_row(row):
    \"\"\"Encode une ligne mat_full en vecteur numérique (8 variables).\"\"\"
    vals = [float(row[col]) for col in VARS_NUM]
    vals += [float(row[col] == 'yes') for col in VARS_CAT]
    return np.array(vals, dtype=float)


def gower_distance(a, b, ranges_num):
    \"\"\"Distance mixte de Gower entre deux vecteurs encodés (8 variables).\"\"\"
    d = 0.0
    for i, col in enumerate(VARS_NUM):
        r_min, r_max = ranges_num[col]
        r = r_max - r_min
        d += abs(a[i] - b[i]) / r if r > 0 else 0.0
    offset = len(VARS_NUM)
    for j, _ in enumerate(VARS_CAT):
        d += float(a[offset + j] != b[offset + j])
    return d / (len(VARS_NUM) + len(VARS_CAT))
"""

INTRO = """# Étape 3 — Clustering & validation des profils (V2 — seuil de variance AFTD)

**Projet SY09 — Décrochage scolaire en mathématiques**  
Notebook 3/3 · Distance mixte (§2.2) · AFTD (Ch. 6) · CAH-Ward + K-means (Ch. 7) · Validation statistique & externe

---

## Contexte

Ce notebook identifie des **profils d'élèves** à partir de 8 variables socio-comportementales
disponibles en début d'année, puis **valide** statistiquement et extérieurement ces profils.
Le décrochage (G3=0, G1>0) touche ~9,6 % des élèves en maths.

Le clustering porte sur l'**intégralité** du jeu (`mat_full`, 395 élèves). Le nombre d'axes AFTD
est fixé par `VARIANCE_THRESHOLD` ; le nombre de clusters K est choisi automatiquement
(silhouette, K ≥ 3).

**Plan :**
1. Imports et chargement
2. Sélection de variables — rappel et justification
3. Distance mixte (cours §2.2)
4. AFTD — seuil de variance, scree plot, Shepard, contributions (cours Ch. 6)
5. CAH-Ward — dendrogramme, sauts d'inertie (cours Ch. 7 §5.5)
6. K-means — coude + silhouette, choix automatique de K (cours Ch. 7 §6.1)
7. Comparaison ARI (cours §7)
8. Caractérisation descriptive des profils
9. Synthèse de l'étape non supervisée
10. Validation statistique — les profils existent-ils vraiment ?
11. Stabilité — les clusters résistent-ils à des sous-échantillons ?
12. Validation externe par les trajectoires de notes
13. Cohérence élargie — variables non utilisées dans le clustering
14. Membres typiques et atypiques
15. Usage opérationnel — score de proximité
16. Synthèse de la validation des profils
"""

IMPORTS = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal, fisher_exact
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
DATA_PATH = 'Data/student-mat.csv'

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11
})
BLEU   = '#2196F3'
ROUGE  = '#E53935'
VERT   = '#43A047'
ORANGE = '#FB8C00'
PAL    = [BLEU, ROUGE, VERT, ORANGE]
"""

SECTION9_BILAN = """### 9.2 Bilan et enchaînement vers la validation

**Acquis :** des profils structurés émergent des seules variables
socio-comportementales disponibles en début d'année. Le nombre d'axes (`n_axes`) dépend du
seuil `VARIANCE_THRESHOLD` ; K est choisi automatiquement (silhouette, K ≥ 3).
La convergence CAH / K-means est évaluée via l'ARI.

**Limites de l'étape descriptive :** un taux élevé dans un cluster ne suffit pas à
conclure. Les sections **10 à 16** approfondissent la validation :
- **§10** : significativité statistique des différences inter-clusters
- **§11** : stabilité par bootstrap (AFTD recalculée avec le même seuil)
- **§12–13** : validité externe (notes, variables non utilisées)
- **§14–15** : typicité des membres et assignation opérationnelle

Le **notebook 02** traite quant à lui la **prédiction supervisée du décrochage**
à partir de G1, G2, G3 — question distincte de l'identification de profils.
"""


def _adapt_part1(cells: list[dict]) -> list[dict]:
    part1 = [_fresh(c) for c in cells[:35]]
    _set_src(part1[0], INTRO)
    _set_src(part1[2], IMPORTS)

    s3 = _cell_src(part1[3])
    s3 = s3.replace("mat = pd.read_csv", "mat_full = pd.read_csv")
    s3 = re.sub(r"\bmat\b", "mat_full", s3)
    _set_src(part1[3], s3)

    s6 = re.sub(r"\bmat\b", "mat_full", _cell_src(part1[6]))
    if "gower_distance" not in s6:
        s6 = s6.rstrip() + "\n" + GOWER_HELPERS
    _set_src(part1[6], s6)

    for i, c in enumerate(part1):
        if i in (0, 2, 3, 6):
            continue
        _set_src(c, re.sub(r"\bmat\b", "mat_full", _cell_src(c)))

    for c in part1:
        if "### 9.2 Bilan" in _cell_src(c):
            _set_src(c, SECTION9_BILAN)
            break
    return part1


def _find_section10_index(cells: list[dict]) -> int:
    for i, c in enumerate(cells):
        if SECTION10_START_MARKER in _cell_src(c):
            return i
    raise ValueError(f"Section 10 not found ({SECTION10_START_MARKER})")


def rebuild() -> None:
    with V2_PATH.open() as f:
        v2 = json.load(f)
    with NB03_PATH.open() as f:
        nb03 = json.load(f)

    part1 = _adapt_part1(v2["cells"])
    start = _find_section10_index(nb03["cells"])
    part2 = [_fresh(c) for c in nb03["cells"][start:]]

    out = {
        "nbformat": nb03["nbformat"],
        "nbformat_minor": nb03["nbformat_minor"],
        "metadata": nb03["metadata"],
        "cells": part1 + part2,
    }
    with NB03_PATH.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Wrote {NB03_PATH.name}: {len(part1)} + {len(part2)} = {len(out['cells'])} cells")


if __name__ == "__main__":
    rebuild()
