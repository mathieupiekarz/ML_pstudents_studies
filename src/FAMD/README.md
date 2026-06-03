# Analyse FAMD — exploration globale

Ce dossier contient une **Factor Analysis of Mixed Data (FAMD)** sur le tableau fusionné [`data/data_global.csv`](../../data/data_global.csv). L’objectif n’est pas la prédiction : c’est une **vue d’ensemble** pour décider quelles analyses approfondir ensuite (ACP sur le bloc quantitatif, ACM sur le qualitatif, profils par sous-groupes, etc.).

**Toutes les 53 variables** sont utilisées, y compris les notes **G1, G2 et G3** (maths et portugais).

## Prérequis

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (recommandé) ou `pip`

## Installation des dépendances

Depuis la **racine du projet** :

```bash
uv sync
```

Paquets utilisés : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `prince`.

Alternative avec pip :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn prince
```

## Lancement

Depuis la racine du projet :

```bash
uv run python src/FAMD/famd_analysis.py
```

Ou depuis ce dossier :

```bash
cd src/FAMD
uv run python famd_analysis.py
```

Le script crée automatiquement le dossier `outputs/` et affiche dans le terminal le chemin de chaque fichier généré.

## Fichiers générés

| Fichier | Description |
|---------|-------------|
| `outputs/01_scree_plot.png` | Éboulis des valeurs propres et inertie cumulée |
| `outputs/02_individuals_map_dim1_dim2.png` | Nuage des individus sur les axes 1 et 2 |
| `outputs/03_individuals_colored.png` | Même nuage, coloré par `sex`, `school`, `address` ou `internet` (première disponible) |
| `outputs/04_variable_contributions.png` | Contributions des variables aux axes 1 et 2 |
| `outputs/05_categories_map.png` | Position moyenne des modalités qualitatives |
| `outputs/06_quantitative_variables_map.png` | Cercle des corrélations (variables numériques) |
| `outputs/07_numeric_correlation_heatmap.png` | Heatmap des corrélations entre variables numériques |
| `outputs/08_famd_biplot.png` | Biplot simplifié : individus + variables les plus contributives |
| `outputs/individual_coordinates.csv` | Coordonnées des individus sur les 10 composantes |
| `outputs/variable_coordinates.csv` | Coordonnées / cosinus / η² des variables |
| `outputs/eigenvalues.csv` | Valeurs propres et pourcentages d’inertie |
| `outputs/famd_summary.txt` | Résumé textuel et pistes d’analyses ultérieures |

Le dossier `outputs/` est ignoré par Git (artefacts reproductibles).

## Interprétation rapide des graphiques

### 01 — Éboulis

Montre combien d’**axes** résument l’essentiel de l’information. Un coude net après 2 ou 3 axes suggère de se concentrer sur ces dimensions. **Piste** : si peu d’inertie sur les premiers axes, envisager des analyses séparées (ACP / ACM) plutôt qu’une lecture uniquement globale.

### 02 — Carte des individus

Chaque point est un élève ; la proximité indique des **profils proches** sur l’ensemble des variables (sociales, notes, comportements). **Piste** : nuages séparés → comparer des sous-groupes ou une ACM sur des variables qualitatives.

### 03 — Individus colorés

Même carte, avec une couleur selon une variable qualitative (`sex`, `school`, etc.). **Piste** : si les couleurs forment des zones distinctes, creuser cette variable (tableaux de profils, ACM).

### 04 — Contributions des variables

Indique quelles **variables** structurent chaque axe (sociales, notes G3, absences, etc.). **Piste** : variables dominantes en quanti → ACP ciblée ; en quali → ACM.

### 05 — Carte des modalités

Position moyenne des modalités (ex. `school=GP`, `sex=F`). **Piste** : modalités éloignées → profils différents ; utile pour préparer une ACM détaillée.

### 06 — Cercle des variables quantitatives

Corrélations des variables **numériques** avec les axes (flèches dans le cercle unité). **Piste** : notes ou comportements qui varient ensemble → ACP sur le bloc numérique (éventuellement maths vs portugais).

### 07 — Heatmap des corrélations numériques

Redondances entre variables quantitatives **avant** FAMD (données imputées). **Piste** : fortes corrélations entre G1/G2/G3 ou `.m`/`.p` — information pour simplifier une future ACP, **sans** obligation d’exclure G3 en exploration.

### 08 — Biplot

Synthèse : individus (points) et variables les plus **contributives** (flèches). **Piste** : prioriser les variables à analyser dans des études séparées.

## Rôle dans le projet

La FAMD est une **étape 0** exploratoire, alignée avec les analyses **ACP** (`src/ACP/`) et **ACM** (`src/ACM/`) sur le même jeu de données complet.

Consultez `outputs/famd_summary.txt` après chaque exécution pour les listes de variables, l’inertie des 5 premiers axes et des **pistes d’analyses ultérieures** générées automatiquement.
