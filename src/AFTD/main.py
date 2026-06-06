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

CORRECTION = "cailliez"

from _utils import build_typology, get_aftd_groups, get_run_name, load_data
from config import apply_preprocessing, results_suffix

RESULTS_DIR = SCRIPT_DIR / "results"
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

    df_raw = load_data()
    df = apply_preprocessing(df_raw)
    print(f"Prétraitement : {results_suffix()} -> {df.shape[1]} colonnes")

    typology = build_typology(df)
    groups = get_aftd_groups(typology)

    print("Groupes de variables (AFTD) :")
    for key, cols in groups.items():
        print(f"  {key} ({len(cols)}) : {cols}")

    # Colonnes de coloration / tri dérivées des groupes (robustes aux exclusions
    # et au moyennage .m/.p, ex. G3.m -> G3).
    color_categorical = (
        groups["nominal"][0]
        if groups["nominal"]
        else (groups["binary"][0] if groups["binary"] else None)
    )
    color_continuous = next(
        (c for c in ("G3", "G3.m", "G3.p") if c in df.columns),
        groups["quantitative"][0] if groups["quantitative"] else None,
    )
    heatmap_sort = color_categorical

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

    plot_paths: list[Path] = []

    scree_path = RESULTS_DIR / "01_scree_plot.png"
    plot_scree(mds_result, scree_path)
    plot_paths.append(scree_path)

    if color_categorical is not None:
        cat_path = RESULTS_DIR / f"02_individuals_by_{color_categorical}.png"
        plot_individuals_categorical(
            mds_result.coordinates,
            df[color_categorical],
            cat_path,
            title=f"AFTD — Carte des individus (axes 1–2), colorée par {color_categorical}",
        )
        plot_paths.append(cat_path)

    if color_continuous is not None:
        cont_path = RESULTS_DIR / f"03_individuals_by_{color_continuous.replace('.', '')}.png"
        plot_individuals_continuous(
            mds_result.coordinates,
            df[color_continuous],
            cont_path,
            title=f"AFTD — Carte des individus (axes 1–2), colorée par {color_continuous}",
        )
        plot_paths.append(cont_path)

    if heatmap_sort is not None:
        heatmap_path = RESULTS_DIR / "04_gower_heatmap.png"
        plot_gower_heatmap(distance, df[heatmap_sort], heatmap_path)
        plot_paths.append(heatmap_path)

    correlations = variable_axis_correlations(
        df,
        mds_result.coordinates,
        groups["quantitative"],
        groups["binary"],
        groups["nominal"],
        groups["ordinal"],
    )
    contrib_path = RESULTS_DIR / "05_variable_contributions.png"
    plot_variable_contributions(correlations, contrib_path)
    plot_paths.append(contrib_path)

    print_summary_table(mds_result)

    print("Figures générées :")
    for path in plot_paths:
        print(f"  - {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
