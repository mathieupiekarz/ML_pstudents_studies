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

Le dossier de résultats reçoit un suffixe reflétant le mode actif (ex.
`results_avg_excl4/`, `outputs_avg_excl4/`), puis le sous-dossier du nom d'exécution.

Chaque script attend un **nom de sauvegarde** obligatoire en argument. Les sorties
sont écrites dans un sous-dossier portant ce nom (`results{suffixe}/<nom>/`), sans
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

- **ACP** (`src/ACP/results{suffixe}/<nom>/`) : variables quantitatives — CSV `pca_*`, figures scree, biplot, contributions…
- **ACM** (`src/ACM/results{suffixe}/<nom>/`) : variables qualitatives et binaires — CSV `mca_*`, figures scree, carte asymétrique…
- **Partagés** (`src/shared/results/`) : `variable_typology.csv`, `factor_analysis_summary.txt`, `exploration_guide.txt` (toujours à plat, communs à toutes les exécutions)
- **FAMD** (`src/FAMD/outputs{suffixe}/<nom>/`) : vue globale mixte — voir `src/FAMD/README.md`
- **AFTD** (`src/AFTD/results{suffixe}/<nom>/`) : MDS classique sur la distance de Gower de `data_global.csv` (382 individus, variables mixtes ; ordinales traitées en rangs ; valeurs propres négatives corrigées par la méthode de Cailliez) — `01_scree_plot.png`, cartes individus, heatmap Gower, liaisons variables/axes
