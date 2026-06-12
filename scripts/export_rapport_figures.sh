#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FIG_DIR="$ROOT/RAPPORT/figures"
mkdir -p "$FIG_DIR"

NOTEBOOKS=(
  "00_analyse_preliminaire.ipynb"
  "03_cluster_supervise.ipynb"
  "02_apprentissage_supervise.ipynb"
)

echo "=== Exécution des notebooks (export figures) ==="
./scripts/jupyter.sh nbconvert --execute --to notebook --inplace "${NOTEBOOKS[@]}"

EXPECTED=(
  "00_g3_distribution.png"
  "00_trajectoires_decrochage.png"
  "00_discriminants.png"
  "03_aftd_scree_shepard.png"
  "03_plan_factoriel.png"
  "03_cah_dendrogramme.png"
  "03_contributions_variables.png"
  "03_plan_factoriel_clusters.png"
  "03_kmeans_coude_silhouette.png"
  "03_profils_clusters.png"
  "03_trajectoires_notes.png"
  "02_phase_a_decrochage_cluster.png"
  "02_phase_b_risque.png"
  "02_phase_b_roc_np.png"
  "02_phase_c_triangulation.png"
)

echo ""
echo "=== Vérification des figures ==="
missing=0
for f in "${EXPECTED[@]}"; do
  if [[ -f "$FIG_DIR/$f" ]]; then
    echo "  OK  $f"
  else
    echo "  MANQUANT  $f"
    missing=$((missing + 1))
  fi
done

if [[ $missing -gt 0 ]]; then
  echo ""
  echo "Erreur : $missing figure(s) manquante(s) dans $FIG_DIR" >&2
  exit 1
fi

echo ""
echo "Toutes les figures sont prêtes dans $FIG_DIR"
