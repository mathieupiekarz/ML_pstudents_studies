"""Runners factoriels : ACP, ACM, FAMD, AFTD, ACP_mixte."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prince
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _utils import (
    build_typology,
    compute_mca_modality_contributions,
    compute_pca_contributions,
    get_acm_columns,
    get_acp_columns,
    get_acp_mixte_columns,
    get_aftd_groups,
    get_famd_groups,
    impute_qualitative,
    impute_quantitative,
    n_axes_for_threshold,
)
from factor_analysis.inertia_viz import LINE_MARKER
from factor_analysis.models import AnalysisResult

sns.set_theme(style="whitegrid")


def _save_or_return(fig: plt.Figure, path: Path | None, save: bool, key: str, figs: dict):
    if save and path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        figs[key] = fig
    return figs


def run_acp(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    *,
    save: bool = True,
) -> AnalysisResult:
    typology = build_typology(df)
    acp_cols = get_acp_columns(typology)
    if not acp_cols:
        raise ValueError("Aucune variable quantitative pour l'ACP.")

    X = impute_quantitative(df, acp_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=None, random_state=0)
    coords = pca.fit_transform(X_scaled)

    pct = pca.explained_variance_ratio_
    cum = np.cumsum(pct)
    contrib = compute_pca_contributions(pca.components_, pca.explained_variance_)
    contrib.index = acp_cols

    coords_df = pd.DataFrame(coords, columns=[f"Dim{i + 1}" for i in range(coords.shape[1])])
    coords_df.insert(0, "individu", np.arange(1, len(coords_df) + 1))

    eigen_df = pd.DataFrame({
        "axe": np.arange(1, len(pct) + 1),
        "eigenvalue": pca.explained_variance_,
        "inertia_pct": pct * 100,
        "inertia_cum_pct": cum * 100,
    })

    figs: dict = {}
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        eigen_df.to_csv(results_dir / "pca_eigenvalues.csv", index=False)
        contrib.to_csv(results_dir / "pca_variable_contributions.csv")
        coords_df.to_csv(results_dir / "pca_individual_coordinates.csv", index=False)

    # Scree
    fig, ax1 = plt.subplots(figsize=(9, 5))
    axes_n = np.arange(1, len(pct) + 1)
    ax1.bar(axes_n, pct * 100, alpha=0.7, label="Inertie (%)")
    ax1.set_xlabel("Axe")
    ax1.set_ylabel("Inertie expliquée (%)")
    ax1.set_title("ACP — Scree plot")
    ax2 = ax1.twinx()
    ax2.plot(axes_n, cum * 100, f"{LINE_MARKER}-", color="darkred", label="Cumul (%)")
    ax2.set_ylabel("Inertie cumulée (%)")
    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
    fig.tight_layout()
    _save_or_return(fig, results_dir / "pca_screeplot.png" if results_dir else None, save, "scree", figs)

    # Contributions dim1
    col = "Dim1"
    top = contrib[col].sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"ACP — Top contributions — {col}")
    fig.tight_layout()
    _save_or_return(
        fig,
        results_dir / "pca_variable_contributions_dim1.png" if results_dir else None,
        save, "contrib_dim1", figs,
    )

    return AnalysisResult(
        method_label="ACP",
        eigen_df=eigen_df,
        explained_variance_ratio=pct,
        coords_df=coords_df,
        contributions=contrib,
        figures=figs,
        results_dir=results_dir,
        n_vars=len(acp_cols),
    )


def run_acm(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    *,
    save: bool = True,
) -> AnalysisResult:
    typology = build_typology(df)
    acm_cols = get_acm_columns(typology)
    if not acm_cols:
        raise ValueError("Aucune variable qualitative pour l'ACM.")

    X = impute_qualitative(df, acm_cols)
    n_categories = sum(X[c].nunique() for c in acm_cols)
    n_comp = min(n_categories - len(acm_cols), len(X) - 1)
    mca = prince.MCA(n_components=n_comp, random_state=0).fit(X)

    pct = np.asarray(mca.percentage_of_variance_) / 100.0
    cum_pct = np.asarray(mca.cumulative_percentage_of_variance_)
    cum = cum_pct / 100.0

    row_coords = mca.row_coordinates(X)
    col_coords = mca.column_coordinates(X)
    row_coords.columns = [f"Dim{i + 1}" for i in range(row_coords.shape[1])]
    col_coords.columns = [f"Dim{i + 1}" for i in range(col_coords.shape[1])]
    contrib = compute_mca_modality_contributions(col_coords)

    coords_df = row_coords.copy()
    coords_df.insert(0, "individu", np.arange(1, len(coords_df) + 1))

    eigen_df = pd.DataFrame({
        "axe": np.arange(1, len(pct) + 1),
        "eigenvalue": mca.eigenvalues_,
        "inertia_pct": pct * 100,
        "inertia_cum_pct": cum_pct,
    })

    figs: dict = {}
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        eigen_df.to_csv(results_dir / "mca_eigenvalues.csv", index=False)
        contrib.to_csv(results_dir / "mca_modality_contributions.csv")
        coords_df.to_csv(results_dir / "mca_individual_coordinates.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    axes_n = np.arange(1, len(pct) + 1)
    ax1.bar(axes_n, pct * 100, alpha=0.7, color="teal")
    ax1.set_title("ACM — Scree plot")
    ax2 = ax1.twinx()
    ax2.plot(axes_n, cum_pct, f"{LINE_MARKER}-", color="darkred")
    ax2.set_ylabel("Inertie cumulée (%)")
    fig.tight_layout()
    _save_or_return(fig, results_dir / "mca_screeplot.png" if results_dir else None, save, "scree", figs)

    return AnalysisResult(
        method_label="ACM",
        eigen_df=eigen_df,
        explained_variance_ratio=pct,
        coords_df=coords_df,
        contributions=contrib,
        figures=figs,
        results_dir=results_dir,
        n_vars=len(acm_cols),
    )


def run_famd(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    *,
    save: bool = True,
    n_components: int = 10,
) -> AnalysisResult:
    from sklearn.impute import SimpleImputer

    typology = build_typology(df)
    numeric_cols, categorical_cols, binary_cols = get_famd_groups(typology)
    qual_cols = categorical_cols + binary_cols

    out = df.copy()
    if numeric_cols:
        out[numeric_cols] = SimpleImputer(strategy="median").fit_transform(out[numeric_cols])
    for col in qual_cols:
        mode_val = out[col].mode(dropna=True)
        fill = mode_val.iloc[0] if len(mode_val) else "missing"
        out[col] = out[col].fillna(fill).astype(str)
    for col in qual_cols:
        out[col] = out[col].astype("category")
    for col in numeric_cols:
        out[col] = out[col].astype(np.float64)

    famd = None
    for engine in ("sklearn", "scipy"):
        try:
            famd = prince.FAMD(
                n_components=n_components, n_iter=3, random_state=42,
                engine=engine, rescale_with_mean=True, rescale_with_std=False,
            ).fit(out)
            break
        except (TypeError, ValueError):
            continue
    if famd is None:
        raise RuntimeError("Échec FAMD.")

    if hasattr(famd, "eigenvalues_summary"):
        eigen_df = famd.eigenvalues_summary.copy()
    else:
        eig = np.asarray(famd.eigenvalues_)
        pct_arr = 100 * eig / eig.sum()
        eigen_df = pd.DataFrame({
            "eigenvalue": eig,
            "% of variance": pct_arr,
            "% of variance (cumulative)": np.cumsum(pct_arr),
        })

    pct_raw = pd.to_numeric(
        eigen_df["% of variance"].astype(str).str.rstrip("%"), errors="coerce"
    )
    pct = pct_raw.values / (100.0 if pct_raw.max() > 1.5 else 1.0)
    cum_col = "% of variance (cumulative)"
    if cum_col in eigen_df.columns:
        cum_raw = pd.to_numeric(
            eigen_df[cum_col].astype(str).str.rstrip("%"), errors="coerce"
        )
        cum_pct = cum_raw.values
    else:
        cum_pct = np.cumsum(pct) * 100

    row_coords = famd.row_coordinates(out) if hasattr(famd, "row_coordinates") else famd.transform(out)
    n_dims = row_coords.shape[1]
    coords_df = row_coords.copy()
    coords_df.columns = [f"Dim{i + 1}" for i in range(n_dims)]
    coords_df.insert(0, "individu", np.arange(1, len(coords_df) + 1))

    eigen_out = pd.DataFrame({
        "axe": np.arange(1, len(pct) + 1),
        "eigenvalue": eigen_df["eigenvalue"].values,
        "inertia_pct": pct * 100,
        "inertia_cum_pct": cum_pct,
    })

    figs: dict = {}
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        eigen_out.to_csv(results_dir / "eigenvalues.csv", index=False)
        coords_df.to_csv(results_dir / "individual_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(pct)), pct * 100, color="steelblue", alpha=0.8)
    ax.set_title("FAMD — Scree plot")
    ax.set_xlabel("Axe")
    ax.set_ylabel("Inertie (%)")
    fig.tight_layout()
    _save_or_return(fig, results_dir / "01_scree_plot.png" if results_dir else None, save, "scree", figs)

    return AnalysisResult(
        method_label="FAMD",
        eigen_df=eigen_out,
        explained_variance_ratio=pct,
        coords_df=coords_df,
        figures=figs,
        results_dir=results_dir,
        n_vars=len(numeric_cols) + len(qual_cols),
    )


def run_aftd(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    *,
    save: bool = True,
) -> AnalysisResult:
    import sys
    aftd_dir = Path(__file__).resolve().parent.parent / "AFTD"
    if str(aftd_dir) not in sys.path:
        sys.path.insert(0, str(aftd_dir))
    from gower_mds import classical_mds, gower_distance_matrix

    typology = build_typology(df)
    groups = get_aftd_groups(typology)
    distance = gower_distance_matrix(
        df, groups["quantitative"], groups["binary"],
        groups["nominal"], groups["ordinal"],
    )
    mds_result = classical_mds(distance, correction="cailliez")

    n_dims = mds_result.coordinates.shape[1]
    pct = mds_result.explained_variance[:n_dims] / 100.0
    cum = mds_result.cumulative_variance[:n_dims] / 100.0

    coords_df = pd.DataFrame(
        mds_result.coordinates,
        columns=[f"Dim{i + 1}" for i in range(n_dims)],
    )
    coords_df.insert(0, "individu", np.arange(1, len(coords_df) + 1))

    eigen_df = pd.DataFrame({
        "axe": np.arange(1, n_dims + 1),
        "eigenvalue": mds_result.eigenvalues[:n_dims],
        "inertia_pct": mds_result.explained_variance[:n_dims],
        "inertia_cum_pct": cum * 100,
    })

    figs: dict = {}
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        eigen_df.to_csv(results_dir / "mds_eigenvalues.csv", index=False)
        coords_df.to_csv(results_dir / "mds_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(1, n_dims + 1), pct * 100, alpha=0.7, color="purple")
    ax.plot(range(1, n_dims + 1), cum * 100, f"{LINE_MARKER}-", color="darkred")
    ax.set_title("AFTD — Scree plot")
    ax.set_xlabel("Axe")
    fig.tight_layout()
    _save_or_return(fig, results_dir / "01_scree_plot.png" if results_dir else None, save, "scree", figs)

    return AnalysisResult(
        method_label="AFTD",
        eigen_df=eigen_df,
        explained_variance_ratio=pct,
        coords_df=coords_df,
        figures=figs,
        results_dir=results_dir,
        n_vars=sum(len(v) for v in groups.values()),
    )


def run_acp_mixte(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    *,
    save: bool = True,
) -> AnalysisResult:
    typology = build_typology(df)
    quant_cols, qual_cols = get_acp_mixte_columns(typology)
    if not quant_cols and not qual_cols:
        raise ValueError("Aucune variable pour ACP_mixte.")

    parts: list[pd.DataFrame] = []
    col_names: list[str] = []
    parent_map: dict[str, str] = {}

    if quant_cols:
        Xq = impute_quantitative(df, quant_cols)
        parts.append(Xq)
        col_names.extend(quant_cols)
        for c in quant_cols:
            parent_map[c] = c

    if qual_cols:
        Xqual = impute_qualitative(df, qual_cols)
        for col in qual_cols:
            dummies = pd.get_dummies(Xqual[col].astype(str), prefix=col, dtype=float)
            parts.append(dummies)
            for dc in dummies.columns:
                col_names.append(dc)
                parent_map[dc] = col

    X_all = pd.concat(parts, axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=None, random_state=0)
    coords = pca.fit_transform(X_scaled)

    pct = pca.explained_variance_ratio_
    cum = np.cumsum(pct)
    contrib = compute_pca_contributions(pca.components_, pca.explained_variance_)
    contrib.index = col_names

    # Agrégation par variable parente
    parent_contrib = pd.DataFrame(0.0, index=sorted(set(parent_map.values())),
                                  columns=contrib.columns)
    for col in contrib.index:
        parent = parent_map[col]
        parent_contrib.loc[parent] += contrib.loc[col]

    coords_df = pd.DataFrame(coords, columns=[f"Dim{i + 1}" for i in range(coords.shape[1])])
    coords_df.insert(0, "individu", np.arange(1, len(coords_df) + 1))

    eigen_df = pd.DataFrame({
        "axe": np.arange(1, len(pct) + 1),
        "eigenvalue": pca.explained_variance_,
        "inertia_pct": pct * 100,
        "inertia_cum_pct": cum * 100,
    })

    figs: dict = {}
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        eigen_df.to_csv(results_dir / "pca_eigenvalues.csv", index=False)
        parent_contrib.to_csv(results_dir / "pca_variable_contributions.csv")
        coords_df.to_csv(results_dir / "pca_individual_coordinates.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    axes_n = np.arange(1, len(pct) + 1)
    ax1.bar(axes_n, pct * 100, alpha=0.7, color="darkgreen")
    ax1.set_title("ACP_mixte — Scree plot")
    ax2 = ax1.twinx()
    ax2.plot(axes_n, cum * 100, f"{LINE_MARKER}-", color="darkred")
    fig.tight_layout()
    _save_or_return(fig, results_dir / "pca_screeplot.png" if results_dir else None, save, "scree", figs)

    col = "Dim1"
    top = parent_contrib[col].sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax, color="darkgreen")
    ax.set_title(f"ACP_mixte — Top contributions — {col}")
    fig.tight_layout()
    _save_or_return(
        fig,
        results_dir / "pca_variable_contributions_dim1.png" if results_dir else None,
        save, "contrib_dim1", figs,
    )

    return AnalysisResult(
        method_label="ACP_mixte",
        eigen_df=eigen_df,
        explained_variance_ratio=pct,
        coords_df=coords_df,
        contributions=parent_contrib,
        figures=figs,
        results_dir=results_dir,
        n_vars=len(quant_cols) + len(qual_cols),
        metadata={"quant_cols": quant_cols, "qual_cols": qual_cols},
    )


RUNNERS = {
    "ACP": run_acp,
    "ACM": run_acm,
    "FAMD": run_famd,
    "AFTD": run_aftd,
    "ACP_mixte": run_acp_mixte,
}
