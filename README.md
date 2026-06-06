## Colonnes CSV

| Colonne | Nom complet |
|---------|-------------|
| `school` | École |
| `sex` | Sexe |
| `age` | Âge |
| `address` | Type de domicile |
| `famsize` | Taille de la famille |
| `Pstatus` | Cohabitation des parents |
| `Medu` | Niveau d’éducation de la mère |
| `Fedu` | Niveau d’éducation du père |
| `Mjob` | Profession de la mère |
| `Fjob` | Profession du père |
| `reason` | Motif de choix de l’école |
| `guardian` | Tuteur légal |
| `traveltime` | Temps de trajet domicile–école |
| `studytime` | Temps d’étude hebdomadaire |
| `failures` | Nombre d’échecs passés |
| `schoolsup` | Soutien scolaire extra |
| `famsup` | Soutien familial à l’éducation |
| `paid` | Cours payants dans la matière |
| `activities` | Activités extra-scolaires |
| `nursery` | École maternelle |
| `higher` | Poursuite d’études supérieures |
| `internet` | Accès Internet à domicile |
| `romantic` | Relation amoureuse |
| `famrel` | Qualité des relations familiales |
| `freetime` | Temps libre |
| `goout` | Sorties entre amis |
| `Dalc` | Consommation d’alcool en semaine |
| `Walc` | Consommation d’alcool le week-end |
| `health` | État de santé |
| `absences` | Nombre d’absences |
| `G1` | Note de la 1ʳᵉ période |
| `G2` | Note de la 2ᵉ période |
| `G3` | Note finale |

Dans `data_global.csv`, les colonnes dupliquées portent `.m` (mathématiques) ou `.p` (portugais), par ex. `G3.m` = note finale en maths. Dans `data_global_R.csv`, les suffixes sont `.x` (maths) et `.y` (portugais).

## Analyse factorielle (ACP / ACM)

Source unique : `data/data_global.csv`.

### Configuration commune (`src/config.py`)

Le prétraitement est centralisé dans [`src/config.py`](src/config.py) et appliqué à
**ACP, ACM, FAMD et AFTD** juste après le chargement des données :

- `INCLUDE_VARS` : un flag `True`/`False` par variable brute (53 colonnes). Mettre
  une variable à `False` la retire de toutes les analyses. Par défaut, seules
  `G1.m/G1.p` et `G2.m/G2.p` sont exclues (`False`).
- `AVERAGE_MAT_POR` : si `True` (défaut), les paires mathématiques/portugais (`.m`/`.p`)
  d'un même concept sont moyennées en une seule variable (ex. `G3.m` + `G3.p` → `G3`).
  La moyenne n'a lieu que si les deux moitiés de la paire sont conservées.

Le mode de prétraitement actif est rappelé dans les sorties console et le résumé,
mais les résultats sont toujours écrits dans le dossier `results/` (ou `outputs/`
pour FAMD) ; encoder le mode dans le nom d'exécution permet de les distinguer.

Chaque script attend un **nom de sauvegarde** obligatoire en argument. Les sorties
sont écrites dans un sous-dossier portant ce nom (`results/<nom>/`), sans
écraser les exécutions précédentes :

```bash
uv sync
uv run python src/ACP/acp_analysis_factorielle.py run1
uv run python src/ACM/acm_analysis_factorielle.py run1
uv run python src/FAMD/famd_analysis.py run1
uv run python src/AFTD/main.py run1
```

Sans argument, le script s'arrête avec un message d'usage. Le nom n'accepte que
lettres, chiffres, `.`, `_` et `-`.

Résultats :

- **ACP** (`src/ACP/results/<nom>/`) : variables quantitatives — CSV `pca_*`, figures scree, biplot, contributions…
- **ACM** (`src/ACM/results/<nom>/`) : variables qualitatives et binaires — CSV `mca_*`, figures scree, carte asymétrique…
- **Partagés** (`src/shared/results/`) : `variable_typology.csv`, `factor_analysis_summary.txt`, `exploration_guide.txt` (toujours à plat, communs à toutes les exécutions)
- **FAMD** (`src/FAMD/outputs/<nom>/`) : vue globale mixte — voir `src/FAMD/README.md`
- **AFTD** (`src/AFTD/results/<nom>/`) : MDS classique sur la distance de Gower de `data_global.csv` (382 individus, variables mixtes ; ordinales traitées en rangs ; valeurs propres négatives corrigées par la méthode de Cailliez) — `01_scree_plot.png`, cartes individus, heatmap Gower, liaisons variables/axes

## Clustering (`src/CLUSTERING/`)

Le module applique CAH (Ward/Complete/Average), k-means et nuées dynamiques sur les
coordonnées d'une réduction factorielle :

```bash
uv run python src/CLUSTERING/clustering.py <source> <source_run> <cluster_run> [--k K] [--axes N]
uv run python src/CLUSTERING/clustering.py --help
```

- `source` : `acp | acm | famd | aftd` ; `source_run` : le run de la réduction.
- `cluster_run` : nom libre, résultats dans `src/CLUSTERING/results/<cluster_run>/`.
- `--k` force le nombre de clusters (sinon sélection automatique par silhouette) ;
  `--axes` fixe le nombre d'axes retenus (sinon seuil de variance).

## Analyse supervisée (`src/SUPERVISED/`)

Pipeline en trois phases qui confronte les clusters à une (ou plusieurs) variable(s)
cible(s) définie(s) dans `src/config.py` (`TARGET_VARS`, type auto-déduit de la
typologie, surcharge possible via `TARGET_TYPES`) :

- **Phase A — validation exploratoire** : test adapté au type de la cible (ANOVA ou
  Kruskal-Wallis + Dunn, Chi² ou Fisher), tailles d'effet (η², V de Cramér),
  boxplots/barplots, projections factorielles, heatmap.
- **Phase B — modélisation confirmatoire** : panel de modèles (centroïdes, k-NN,
  bayésien naïf, LDA/QDA/ADR, logistique binaire/multinomiale/ordinale + test de
  Brant, logistique L1, arbre élagué, forêt aléatoire, gradient boosting ; OLS /
  Ridge / Lasso pour les cibles quantitatives), règles de décision (Bayes/coûts,
  Neyman-Pearson), régression linéaire inférentielle (ANOVA, tests t/F, diagnostics,
  VIF, sélection Lasso/stepwise), estimation du risque (resubstitution, holdout,
  validation croisée, bootstrap .632), importances Gini/permutation et rôle de
  `cluster_id` (avec vs sans).
- **Phase C — confrontation** : tableau de triangulation A vs B, conclusion
  automatique (4 scénarios) et recommandations.

```bash
uv run python src/SUPERVISED/supervised.py <source> <source_run> <cluster_run> <analysis_run> [--method cah|kmeans|nuees]
uv run python src/SUPERVISED/supervised.py --help
```

Les variables explicatives respectent `INCLUDE_VARS` et `AVERAGE_MAT_POR` comme
ACP/ACM/FAMD/AFTD ; seules les cibles (`TARGET_VARS`) sont lues depuis le CSV brut.

Sorties par cible dans `src/SUPERVISED/results/<analysis_run>/<cible>/` : rapports
`report_phaseA/B/C.txt`, `synthese.txt`, tables CSV (comparaison de modèles,
importances, coefficients, triangulation) et figures PNG (distribution, plans
factoriels, heatmap, diagnostics de régression, arbre élagué).

## Commandes CLI

Référence de tous les scripts Python exécutables du projet. Prérequis commun :
`uv sync` depuis la racine du dépôt.

### Préparation des données

```bash
uv run python src/student-merge.py
```

Fusionne `data/student-mat.csv` et `data/student-por.csv` → `data/data_global.csv`
(382 individus). Aucun argument.

### Réductions factorielles

Chaque script attend un **nom de run** obligatoire (`run_name`) : lettres, chiffres,
`.`, `_`, `-` uniquement. Sorties dans un sous-dossier portant ce nom.

```bash
uv run python src/ACP/acp_analysis_factorielle.py <run_name>
uv run python src/ACM/acm_analysis_factorielle.py <run_name>
uv run python src/FAMD/famd_analysis.py <run_name>
uv run python src/AFTD/main.py <run_name>
```

| Script | Dossier de sortie |
|--------|-------------------|
| ACP | `src/ACP/results/<run_name>/` |
| ACM | `src/ACM/results/<run_name>/` |
| FAMD | `src/FAMD/outputs/<run_name>/` |
| AFTD | `src/AFTD/results/<run_name>/` |

Sans argument : message d'usage et arrêt. Prétraitement (`INCLUDE_VARS`,
`AVERAGE_MAT_POR`) lu depuis `src/config.py`.

Exemple :

```bash
uv run python src/FAMD/famd_analysis.py no_G1_G2_G3
```

### Clustering

```bash
uv run python src/CLUSTERING/clustering.py <source> <source_run> <cluster_run> [--k K] [--axes N]
uv run python src/CLUSTERING/clustering.py --help
```

| Argument | Description |
|----------|-------------|
| `source` | `acp` \| `acm` \| `famd` \| `aftd` |
| `source_run` | nom du run de réduction (ex. `no_G1_G2_G3`) |
| `cluster_run` | nom libre ; sorties dans `src/CLUSTERING/results/<cluster_run>/` |
| `--k K` | nombre de clusters imposé (sinon sélection auto par silhouette, k ∈ [2, 8]) |
| `--axes N` | nombre d'axes retenus (sinon seuil de variance 80 %) |

Exemple :

```bash
uv run python src/CLUSTERING/clustering.py famd no_G1_G2_G3 famd_no_G1_G2_G3
uv run python src/CLUSTERING/clustering.py famd no_G1_G2_G3 famd_k4 --k 4 --axes 5
```

### Analyse supervisée (Phases A / B / C)

```bash
uv run python src/SUPERVISED/supervised.py <source> <source_run> <cluster_run> <analysis_run> [--method cah|kmeans|nuees]
uv run python src/SUPERVISED/supervised.py --help
```

| Argument | Description |
|----------|-------------|
| `source` | `acp` \| `acm` \| `famd` \| `aftd` |
| `source_run` | nom du run de réduction |
| `cluster_run` | nom du run de clustering (`src/CLUSTERING/results/`) |
| `analysis_run` | nom libre ; sorties dans `src/SUPERVISED/results/<analysis_run>/` |
| `--method` | partition utilisée : `cah` (défaut), `kmeans` ou `nuees` ; le `k` est lu depuis le fichier `<method>_labels_k*.csv` |

Cibles définies dans `src/config.py` (`TARGET_VARS`). Explicatives soumises à
`INCLUDE_VARS` et `AVERAGE_MAT_POR`.

Exemple (chaîne complète) :

```bash
uv run python src/FAMD/famd_analysis.py no_G1_G2_G3
uv run python src/CLUSTERING/clustering.py famd no_G1_G2_G3 famd_no_G1_G2_G3
uv run python src/SUPERVISED/supervised.py famd no_G1_G2_G3 famd_no_G1_G2_G3 famd_sup
```

================================================================================
