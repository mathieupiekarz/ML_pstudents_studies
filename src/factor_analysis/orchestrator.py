"""Orchestrateur des analyses factorielles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from _utils import build_typology, detect_method_set, load_data, n_axes_for_thresholds
from config import apply_preprocessing
from factor_analysis.models import AnalysisResult
from factor_analysis.runners import RUNNERS

SRC_DIR = Path(__file__).resolve().parent.parent
OUTPUT_BASE = SRC_DIR / "notebook_results"

METHOD_DIRS = {
    "ACP": SRC_DIR / "ACP" / "results",
    "ACM": SRC_DIR / "ACM" / "results",
    "FAMD": SRC_DIR / "FAMD" / "outputs",
    "AFTD": SRC_DIR / "AFTD" / "results",
    "ACP_mixte": SRC_DIR / "ACP_mixte" / "results",
}


def build_variance_summary_table(
    results: dict[str, AnalysisResult],
    thresholds: tuple[float, ...] = (0.60, 0.80, 0.90),
) -> pd.DataFrame:
    rows = []
    for label, res in results.items():
        cum = res.cum_variance
        thresh = n_axes_for_thresholds(cum, thresholds)
        pct = res.explained_variance_ratio
        rows.append({
            "approche": label,
            "60%": thresh.get("60%", len(cum)),
            "80%": thresh.get("80%", len(cum)),
            "90%": thresh.get("90%", len(cum)),
            "PC1 (%)": round(float(pct[0]) * 100, 2) if len(pct) > 0 else 0,
            "PC2 (%)": round(float(pct[1]) * 100, 2) if len(pct) > 1 else 0,
        })
    return pd.DataFrame(rows)


def run_factor_analyses(
    dataset: str,
    include_vars: dict[str, bool],
    average_mat_por: bool = False,
    run_name: str = "notebook",
    methods: list[str] | None = None,
    save: bool = False,
    output_dir: Path | None = None,
) -> tuple[dict[str, AnalysisResult], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_raw = load_data(dataset)
    df = apply_preprocessing(
        df_raw, dataset=dataset, include_vars=include_vars, average_mat_por=average_mat_por,
    )
    typology = build_typology(df)
    auto_methods = detect_method_set(typology)
    selected = methods or auto_methods

    out_base = output_dir or (OUTPUT_BASE / run_name)
    if save:
        out_base.mkdir(parents=True, exist_ok=True)

    results: dict[str, AnalysisResult] = {}
    for method in selected:
        if method not in RUNNERS:
            continue
        res_dir = (METHOD_DIRS[method] / run_name) if save else None
        try:
            results[method] = RUNNERS[method](df, res_dir, save=save)
        except ValueError as exc:
            print(f"[skip] {method} : {exc}")

    summary = build_variance_summary_table(results)
    if save and results:
        summary.to_csv(out_base / "variance_summary.csv", index=False)

    return results, typology, df_raw, df
