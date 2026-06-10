#!/usr/bin/env python3
"""ACP_mixte — PCA sur variables quantitatives + qualitatives one-hot encodées."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from _utils import get_run_name, load_data
from config import apply_preprocessing, results_suffix
from factor_analysis.runners import run_acp_mixte

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    run_name = get_run_name()
    out = RESULTS_DIR / run_name
    df_raw = load_data()
    df = apply_preprocessing(df_raw)
    print(f"Prétraitement : {results_suffix()}")
    run_acp_mixte(df, out, save=True)
    print(f"Résultats ACP_mixte : {out.resolve()}")


if __name__ == "__main__":
    main()
