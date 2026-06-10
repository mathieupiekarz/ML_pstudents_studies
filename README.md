## Colonnes CSV

Typologie issue des règles explicites de [`src/_utils.py`](src/_utils.py) (`classify_column`).
Les colonnes dupliquées `.m` / `.p` héritent du même type que le nom de base.

| Colonne | Nom complet | Type |
|---------|-------------|------|
| `school` | École | Binaire |
| `sex` | Sexe | Binaire |
| `age` | Âge | Quantitative |
| `address` | Type de domicile | Binaire |
| `famsize` | Taille de la famille | Binaire |
| `Pstatus` | Cohabitation des parents | Binaire |
| `Medu` | Niveau d’éducation de la mère | Qualitative ordinale |
| `Fedu` | Niveau d’éducation du père | Qualitative ordinale |
| `Mjob` | Profession de la mère | Qualitative nominale |
| `Fjob` | Profession du père | Qualitative nominale |
| `reason` | Motif de choix de l’école | Qualitative nominale |
| `guardian` | Tuteur légal | Qualitative nominale |
| `traveltime` | Temps de trajet domicile–école | Qualitative ordinale |
| `studytime` | Temps d’étude hebdomadaire | Qualitative ordinale |
| `failures` | Nombre d’échecs passés | Quantitative |
| `schoolsup` | Soutien scolaire extra | Binaire |
| `famsup` | Soutien familial à l’éducation | Binaire |
| `paid` | Cours payants dans la matière | Binaire |
| `activities` | Activités extra-scolaires | Binaire |
| `nursery` | École maternelle | Binaire |
| `higher` | Poursuite d’études supérieures | Binaire |
| `internet` | Accès Internet à domicile | Binaire |
| `romantic` | Relation amoureuse | Binaire |
| `famrel` | Qualité des relations familiales | Qualitative ordinale |
| `freetime` | Temps libre | Qualitative ordinale |
| `goout` | Sorties entre amis | Qualitative ordinale |
| `Dalc` | Consommation d’alcool en semaine | Qualitative ordinale |
| `Walc` | Consommation d’alcool le week-end | Qualitative ordinale |
| `health` | État de santé | Qualitative ordinale |
| `absences` | Nombre d’absences | Quantitative |
| `G1` | Note de la 1ʳᵉ période | Quantitative |
| `G2` | Note de la 2ᵉ période | Quantitative |
| `G3` | Note finale | Quantitative |

Dans `data_global.csv`, les colonnes dupliquées portent `.m` (mathématiques) ou `.p` (portugais), par ex. `G3.m` = note finale en maths (type : quantitative). Dans `data_global_R.csv`, les suffixes sont `.x` (maths) et `.y` (portugais).

## Analyse factorielle (ACP / ACM / FAMD / AFTD / ACP_mixte)

### Choix du dataset (`src/config.py`)

```python
DATASET = "data_global"  # "student-mat" | "student-por" | "data_global"
```

- `student-mat.csv` — mathématiques uniquement (395 individus)
- `student-por.csv` — portugais uniquement (649 individus)
- `data_global.csv` — fusion des deux (382 individus, colonnes `.m` / `.p`)

### Configuration commune (`src/config.py`)

Le prétraitement est centralisé dans [`src/config.py`](src/config.py) et appliqué à
**ACP, ACM, FAMD, AFTD et ACP_mixte** juste après le chargement :

- `INCLUDE_VARS` : flags **canoniques** (33 concepts : `school`, `G3`, `studytime`…).
  Pour `data_global`, chaque concept matière est expandu en `.m` / `.p`.
  Par défaut `G1`, `G2`, `G3` sont à `False` (réductions sans notes).
- `AVERAGE_MAT_POR` : si `True`, moyenne les paires `.m`/`.p` (uniquement sur `data_global`).
- `FOCUS_VAR` / `DROPOUT_VALUE` : étude du décrochage (`G3 == 0`).

Template versionné : [`src/config.example.py`](src/config.example.py) → copier vers `src/config.py`.

### Dispatch automatique des méthodes

| Variables actives | Méthodes |
|---|---|
| quantitatives **et** qualitatives | AFTD, FAMD, ACP_mixte *(notebook : FAMD, ACP_mixte)* |
| quantitatives seulement | ACP, AFTD, FAMD *(notebook : ACP, FAMD)* |
| qualitatives seulement | ACM, AFTD, FAMD *(notebook : ACM, FAMD)* |

### Notebook récapitulatif (flux principal)

```bash
uv sync --extra notebook
uv run jupyter notebook analyse_factorielle_recap.ipynb
```

Le notebook [`analyse_factorielle_recap.ipynb`](analyse_factorielle_recap.ipynb) suit un **pipeline en 3 boutons** (chaque étape dépend de la précédente). **L'AFTD n'y est pas incluse** (trop d'axes MDS, graphiques illisibles) ; utilisez le script CLI si besoin.

| Étape | Action | Paramètres saisis |
|-------|--------|-------------------|
| **1** | Réductions factorielles | dataset, variables, run name |
| **2** | Coude + silhouette + décrochage G3 | seuil variance (`FloatText`), k min/max (`IntText`) |
| **3** | k-means + profils clusters | méthode (`Dropdown`) + k final (`IntText`) |

**Étape 1** : graphique comparatif inertie 1×2 (par axe + cumulée, 20 axes max), contributions Dim1, tableau récap (60 % / 80 % / 90 %, PC1, PC2).

**Étape 2** : après lecture du tableau, choix du seuil → nombre d'axes retenus, courbes coude/silhouette partagées, métriques détaillées, suggestions k, projection G3=0.

**Étape 3** : choix de la méthode et du k → 3.1 projection clusters, 3.2 décrochage par cluster, 3.3 heatmap z-score, 3.4 grille compacte de **toutes** les variables par cluster (4 colonnes).

### Scripts CLI

```bash
uv sync
uv run python src/ACP/acp_analysis_factorielle.py run1
uv run python src/ACM/acm_analysis_factorielle.py run1
uv run python src/FAMD/famd_analysis.py run1
uv run python src/AFTD/main.py run1
uv run python src/ACP_mixte/acp_mixte_analysis.py run1
```

Résultats :

- **ACP** (`src/ACP/results/<nom>/`) : variables quantitatives
- **ACM** (`src/ACM/results/<nom>/`) : variables qualitatives et binaires
- **ACP_mixte** (`src/ACP_mixte/results/<nom>/`) : quantitatives + qualitatives one-hot
- **FAMD** (`src/FAMD/outputs/<nom>/`) : vue globale mixte
- **AFTD** (`src/AFTD/results/<nom>/`) : MDS sur distance de Gower

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
uv run python src/ACP_mixte/acp_mixte_analysis.py <run_name>
```

| Script | Dossier de sortie |
|--------|-------------------|
| ACP | `src/ACP/results/<run_name>/` |
| ACM | `src/ACM/results/<run_name>/` |
| ACP_mixte | `src/ACP_mixte/results/<run_name>/` |
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
