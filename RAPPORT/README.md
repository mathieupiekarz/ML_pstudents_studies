# Rapport SY09 — décrochage en mathématiques

## Commandes

```bash
./scripts/export_rapport_figures.sh
make -C RAPPORT
```

## Figures — corps du rapport

| Fichier | Section |
|---------|---------|
| `00_g3_distribution.png` | §2.1 |
| `03_aftd_scree_shepard.png` | §3.1 |
| `03_plan_factoriel_clusters.png` | §3.3 — plan factoriel coloré par cluster |
| `03_profils_clusters.png` | §3.3 — barres de profils |

## Figures — annexes (référencées dans le texte)

| Fichier | Annexe | Référence dans le corps |
|---------|--------|-------------------------|
| `00_discriminants.png` | A | §2.2 |
| `03_cah_dendrogramme.png` | B | §3.2 |
| `03_kmeans_coude_silhouette.png` | C | §3.2 |
| `03_contributions_variables.png` | D | §3.3 |
| `02_phase_b_roc_np.png` | E | §4.3, conclusion |

Les renvois `\hyperref[ann:...]{...}` dans `rapport.tex` sont cliquables dans le PDF.
