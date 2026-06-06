#!/usr/bin/env python3
"""AFTD: classical MDS on the global student data using Gower distance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
for path in (str(SCRIPT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

RESULTS_DIR = SCRIPT_DIR / "results"

# Colonnes de coloration / tri pour les cartes (faciles à changer).
COLOR_CATEGORICAL = "Mjob"
COLOR_CONTINUOUS = "G3.m"
HEATMAP_SORT = "Mjob"
CORRECTION = "cailliez"

from _utils import build_typology, get_aftd_groups, get_run_name, load_data
from gower_mds import classical_mds, gower_distance_matrix
from plots import (
    plot_gower_heatmap,
    plot_individuals_categorical,
    plot_individuals_continuous,
    plot_scree,
    plot_variable_contributions,
    variable_axis_correlations,
)


def print_summary_table(result) -> None:
    """Print eigenvalues, explained variance, and cumulative variance."""
    rows = []
    for i, (ev, pct, cum) in enumerate(
        zip(
            result.eigenvalues,
            result.explained_variance,
            result.cumulative_variance,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "Axe": i,
                "Valeur propre": round(float(ev), 4),
                "Var. expliquée (%)": round(float(pct), 2),
                "Cumul (%)": round(float(cum), 2),
            }
        )
    summary = pd.DataFrame(rows)
    print("\n=== Résumé AFTD (MDS classique sur distance de Gower) ===\n")
    print(summary.head(15).to_string(index=False))
    if len(summary) > 15:
        print(f"... ({len(summary)} composantes au total)")
    print()
    print(f"Correction appliquée        : {result.correction}")
    print(f"Valeurs propres négatives   : {result.n_negative_before} (avant correction)")
    print(f"Constante de correction     : {result.correction_constant:.6f}")
    print()


def main() -> int:
    """Load global data, run Gower + MDS, produce plots and console summary."""
    global RESULTS_DIR
    run_name = get_run_name()
    RESULTS_DIR = RESULTS_DIR / run_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    typology = build_typology(df)
    groups = get_aftd_groups(typology)

    print("Groupes de variables (AFTD) :")
    for key, cols in groups.items():
        print(f"  {key} ({len(cols)}) : {cols}")

    distance = gower_distance_matrix(
        df,
        groups["quantitative"],
        groups["binary"],
        groups["nominal"],
        groups["ordinal"],
    )
    mds_result = classical_mds(distance, correction=CORRECTION)

    if mds_result.coordinates.shape[1] < 2:
        print("Warning: fewer than 2 positive eigenvalues; some plots may fail.")

    plot_paths = [
        RESULTS_DIR / "01_scree_plot.png",
        RESULTS_DIR / f"02_individuals_by_{COLOR_CATEGORICAL}.png",
        RESULTS_DIR / f"03_individuals_by_{COLOR_CONTINUOUS.replace('.', '')}.png",
        RESULTS_DIR / "04_gower_heatmap.png",
        RESULTS_DIR / "05_variable_contributions.png",
    ]

    plot_scree(mds_result, plot_paths[0])
    plot_individuals_categorical(
        mds_result.coordinates,
        df[COLOR_CATEGORICAL],
        plot_paths[1],
        title=f"AFTD — Carte des individus (axes 1–2), colorée par {COLOR_CATEGORICAL}",
    )
    plot_individuals_continuous(
        mds_result.coordinates,
        df[COLOR_CONTINUOUS],
        plot_paths[2],
        title=f"AFTD — Carte des individus (axes 1–2), colorée par {COLOR_CONTINUOUS}",
    )
    plot_gower_heatmap(distance, df[HEATMAP_SORT], plot_paths[3])

    correlations = variable_axis_correlations(
        df,
        mds_result.coordinates,
        groups["quantitative"],
        groups["binary"],
        groups["nominal"],
        groups["ordinal"],
    )
    plot_variable_contributions(correlations, plot_paths[4])

    print_summary_table(mds_result)

    print("Figures générées :")
    for path in plot_paths:
        print(f"  - {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
