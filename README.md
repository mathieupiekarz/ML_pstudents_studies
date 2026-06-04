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

Source unique : `data/data_global.csv`. Exploration globale (aucune variable exclue).

```bash
uv sync
uv run python src/ACP/acp_analysis_factorielle.py
uv run python src/ACM/acm_analysis_factorielle.py
uv run python src/FAMD/famd_analysis.py
```

Résultats :

- **ACP** (`src/ACP/results/`) : variables quantitatives — CSV `pca_*`, figures scree, biplot, contributions…
- **ACM** (`src/ACM/results/`) : variables qualitatives et binaires — CSV `mca_*`, figures scree, carte asymétrique…
- **Partagés** (`src/shared/results/`) : `variable_typology.csv`, `factor_analysis_summary.txt`, `exploration_guide.txt`
- **FAMD** (`src/FAMD/outputs/`) : vue globale mixte — voir `src/FAMD/README.md`
