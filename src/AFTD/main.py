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

    from factor_analysis.runners import run_aftd
    run_aftd(df, RESULTS_DIR, save=True)
    print(f"Résultats AFTD : {RESULTS_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
