#!/usr/bin/env python3
"""
Analyse Factorielle de Données Mixtes (FAMD) — exploration globale.

Mode exploration : toutes les colonnes du CSV sont actives (G3 inclus).
Les graphiques servent à orienter des analyses ultérieures (ACP, ACM, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prince
import seaborn as sns
from matplotlib.patches import Circle
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPLORATORY_MODE = True  # Pas d'exclusion de variables « cible »

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data_global.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Classification partagée (mêmes règles que ACP/ACM/AFTD).
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _utils import build_typology, get_famd_groups  # noqa: E402

N_COMPONENTS = 10
TOP_N_CONTRIB = 15
COLOR_PRIORITY = ["sex", "school", "address", "internet"]
FIG_DPI = 150
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Utilitaires Prince (compatibilité entre versions)
# ---------------------------------------------------------------------------
def safe_call(obj, *method_names, default=None, **kwargs):
    """Appelle la première méthode existante sur obj."""
    for name in method_names:
        if hasattr(obj, name) and callable(getattr(obj, name)):
            return getattr(obj, name)(**kwargs)
    return default


def get_eigenvalues_summary(famd) -> pd.DataFrame:
    if hasattr(famd, "eigenvalues_summary"):
        return famd.eigenvalues_summary.copy()
    if hasattr(famd, "_eigenvalues_summary"):
        return famd._eigenvalues_summary.copy()
    eig = np.asarray(getattr(famd, "eigenvalues_", []))
    if len(eig) == 0:
        raise RuntimeError("Impossible de récupérer les valeurs propres.")
    pct = 100 * eig / eig.sum()
    return pd.DataFrame(
        {
            "eigenvalue": eig,
            "% of variance": pct,
            "% of variance (cumulative)": np.cumsum(pct),
        },
        index=pd.RangeIndex(0, len(eig), name="component"),
    )


def get_row_coordinates(famd, X: pd.DataFrame) -> pd.DataFrame:
    if hasattr(famd, "row_coordinates"):
        return famd.row_coordinates(X)
    return famd.transform(X)


def get_column_coordinates(famd) -> pd.DataFrame:
    for attr in ("column_coordinates_", "column_coordinates"):
        if hasattr(famd, attr):
            return getattr(famd, attr).copy()
    raise RuntimeError("Coordonnées des variables indisponibles.")


def get_column_contributions(famd) -> pd.DataFrame:
    if hasattr(famd, "column_contributions_"):
        return famd.column_contributions_.copy()
    coords = get_column_coordinates(famd)
    eig = np.asarray(famd.eigenvalues_)
    return coords.div(eig, axis="columns")


def get_column_correlations(famd, numeric_cols: list[str]) -> pd.DataFrame | None:
    if hasattr(famd, "column_correlations"):
        corr = famd.column_correlations.copy()
        if numeric_cols:
            available = [c for c in numeric_cols if c in corr.index]
            if available:
                return corr.loc[available]
        return corr
    return None


def fit_famd(df: pd.DataFrame) -> prince.FAMD:
    """Ajuste FAMD avec repli engine si nécessaire."""
    params = dict(
        n_components=N_COMPONENTS,
        n_iter=3,
        random_state=RANDOM_STATE,
        rescale_with_mean=True,
        rescale_with_std=False,
        copy=True,
        check_input=True,
    )
    for engine in ("sklearn", "scipy"):
        try:
            model = prince.FAMD(engine=engine, **params)
            model.fit(df)
            print(f"FAMD ajustée (engine={engine}).")
            return model
        except (TypeError, ValueError) as exc:
            print(f"Engine '{engine}' indisponible : {exc}")
    raise RuntimeError("Échec de l'ajustement FAMD.")


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------
def impute_and_prepare(
    df: pd.DataFrame,
    numeric_cols: list[str],
    qual_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute (médiane / mode), convertit quali en category et quanti en float64.
    Retourne (df_prepared, df_imputed_raw) pour la heatmap sur données imputées.
    """
    out = df.copy()
    raw_imputed = df.copy()

    if numeric_cols:
        num_imp = SimpleImputer(strategy="median")
        out[numeric_cols] = num_imp.fit_transform(out[numeric_cols])
        raw_imputed[numeric_cols] = out[numeric_cols].copy()

    for col in qual_cols:
        mode_val = out[col].mode(dropna=True)
        fill = mode_val.iloc[0] if len(mode_val) else "missing"
        out[col] = out[col].fillna(fill).astype(str)
        raw_imputed[col] = out[col]

    for col in qual_cols:
        out[col] = out[col].astype("category")

    # Prince ne traite comme quantitatives que les colonnes float ;
    # la standardisation des quantitatives est faite par Prince (num_scaler_).
    for col in numeric_cols:
        out[col] = out[col].astype(np.float64)
        raw_imputed[col] = raw_imputed[col].astype(np.float64)

    return out, raw_imputed


def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df


def pick_color_column(df: pd.DataFrame) -> str | None:
    for col in COLOR_PRIORITY:
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------
def plot_scree(eigen_df: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = eigen_df.index.astype(int)
    ax1.bar(x, eigen_df["eigenvalue"], color="steelblue", alpha=0.8, label="Valeur propre")
    ax1.set_xlabel("Axe (composante)")
    ax1.set_ylabel("Valeur propre")
    ax1.set_title("Éboulis des valeurs propres / inertie")

    if "% of variance (cumulative)" in eigen_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            eigen_df["% of variance (cumulative)"],
            color="darkorange",
            marker="o",
            linewidth=2,
            label="Inertie cumulée (%)",
        )
        ax2.set_ylabel("Inertie cumulée (%)")
        ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_individuals(
    row_coords: pd.DataFrame,
    path: Path,
    color_series: pd.Series | None = None,
    title: str = "Projection des individus (axes 1 et 2)",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    x = row_coords.iloc[:, 0]
    y = row_coords.iloc[:, 1]

    if color_series is not None:
        palette = sns.color_palette("tab10", n_colors=color_series.nunique())
        for i, (label, group_idx) in enumerate(color_series.groupby(color_series).groups.items()):
            ax.scatter(
                x.loc[group_idx],
                y.loc[group_idx],
                label=str(label),
                alpha=0.65,
                s=35,
                color=palette[i % len(palette)],
            )
        ax.legend(title=color_series.name, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.scatter(x, y, alpha=0.5, s=25, c="steelblue")

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"Axe 1 — composante 0")
    ax.set_ylabel(f"Axe 2 — composante 1")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_variable_contributions(contrib: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.25)))
    for ax, dim in zip(axes, [0, 1]):
        col = contrib.columns[dim]
        top = contrib[col].nlargest(top_n).sort_values()
        ax.barh(top.index.astype(str), top.values, color="teal", alpha=0.85)
        ax.set_title(f"Contributions des variables — axe {dim + 1}")
        ax.set_xlabel("Contribution")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def compute_category_coordinates(
    row_coords: pd.DataFrame,
    df_qual: pd.DataFrame,
    qual_cols: list[str],
) -> pd.DataFrame:
    """Coordonnées des modalités = moyenne des coordonnées des individus par modalité."""
    records = []
    for col in qual_cols:
        for modality in df_qual[col].astype(str).unique():
            mask = df_qual[col].astype(str) == modality
            if mask.sum() == 0:
                continue
            mean_coord = row_coords.loc[mask, [0, 1]].mean()
            records.append(
                {
                    "variable": col,
                    "modality": modality,
                    "label": f"{col}={modality}",
                    "dim1": mean_coord.iloc[0],
                    "dim2": mean_coord.iloc[1],
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(records)


def plot_categories_map(cat_coords: pd.DataFrame, path: Path) -> None:
    if cat_coords.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Aucune variable qualitative disponible.", ha="center", va="center")
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(cat_coords["dim1"], cat_coords["dim2"], alpha=0.6, s=40, c="purple")
    for _, row in cat_coords.iterrows():
        ax.annotate(
            row["label"],
            (row["dim1"], row["dim2"]),
            fontsize=7,
            alpha=0.85,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Axe 1")
    ax.set_ylabel("Axe 2")
    ax.set_title("Coordonnées des modalités qualitatives (axes 1 et 2)")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_circle(
    corr: pd.DataFrame,
    path: Path,
    numeric_cols: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    circle = Circle((0, 0), 1, fill=False, color="grey", linestyle="--", linewidth=1)
    ax.add_patch(circle)

    if corr is not None and not corr.empty:
        for var in corr.index:
            if var not in numeric_cols:
                continue
            x1 = corr.loc[var, corr.columns[0]]
            x2 = corr.loc[var, corr.columns[1]]
            ax.arrow(0, 0, x1, x2, head_width=0.03, length_includes_head=True, alpha=0.8)
            ax.text(x1 * 1.08, x2 * 1.08, str(var), fontsize=8)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axhline(0, color="lightgrey", linewidth=0.5)
    ax.axvline(0, color="lightgrey", linewidth=0.5)
    ax.set_xlabel("Axe 1 (corrélation)")
    ax.set_ylabel("Axe 2 (corrélation)")
    ax.set_title("Cercle des corrélations — variables quantitatives")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_numeric_heatmap(df_num: pd.DataFrame, path: Path) -> None:
    if df_num.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Pas assez de variables numériques.", ha="center", va="center")
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    corr = df_num.corr(numeric_only=True)
    height = max(8, 0.35 * len(corr))
    fig, ax = plt.subplots(figsize=(12, height))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax, linewidths=0.2)
    ax.set_title("Corrélations entre variables quantitatives (données imputées)")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_biplot(
    row_coords: pd.DataFrame,
    col_coords: pd.DataFrame,
    contrib: pd.DataFrame,
    path: Path,
    top_n: int = TOP_N_CONTRIB,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))

    rx = row_coords.iloc[:, 0]
    ry = row_coords.iloc[:, 1]
    row_scale = max(rx.abs().max(), ry.abs().max(), 1e-6)
    ax.scatter(rx, ry, alpha=0.25, s=12, c="steelblue", label="Individus")

    # Variables les plus contributives sur les deux premiers axes
    c0 = contrib.iloc[:, 0].nlargest(top_n).index
    c1 = contrib.iloc[:, 1].nlargest(top_n).index
    top_vars = list(dict.fromkeys(list(c0) + list(c1)))
    top_vars = [v for v in top_vars if v in col_coords.index][:top_n]
    if top_vars:
        vx = col_coords.loc[top_vars, col_coords.columns[0]]
        vy = col_coords.loc[top_vars, col_coords.columns[1]]
        col_scale = max(vx.abs().max(), vy.abs().max(), 1e-6)
        scale_factor = (0.8 * row_scale) / col_scale
        vx_s = vx * scale_factor
        vy_s = vy * scale_factor
        for var in top_vars:
            ax.arrow(
                0,
                0,
                vx_s[var],
                vy_s[var],
                head_width=0.04 * row_scale,
                length_includes_head=True,
                color="darkgreen",
                alpha=0.85,
            )
            ax.text(vx_s[var] * 1.05, vy_s[var] * 1.05, str(var), fontsize=8, color="darkgreen")

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Axe 1")
    ax.set_ylabel("Axe 2")
    ax.set_title("Biplot simplifié — individus et variables contributives")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Résumé textuel
# ---------------------------------------------------------------------------
def build_analysis_hints(
    eigen_df: pd.DataFrame,
    contrib: pd.DataFrame,
    numeric_cols: list[str],
    qual_cols: list[str],
    color_col: str | None,
) -> list[str]:
    """Pistes d'analyses ultérieures à partir des résultats."""
    hints: list[str] = []

    def _pct_series(col: str) -> pd.Series:
        s = eigen_df[col]
        return s.apply(lambda v: float(str(v).strip().rstrip("%")) if not isinstance(v, (int, float)) else float(v))

    if len(eigen_df) >= 2:
        cum2 = _pct_series("% of variance (cumulative)").iloc[1]
        if cum2 >= 20:
            hints.append(
                f"L'inertie cumulée sur 2 axes ({cum2:.1f}%) suggère une structure exploitable : "
                "envisager une lecture détaillée sur 2–3 axes ou une ACP/ACM ciblée."
            )
        else:
            hints.append(
                f"Inertie cumulée faible sur 2 axes ({cum2:.1f}%) : structure diffuse ; "
                "compléter par des analyses par blocs (quanti vs quali)."
            )

    if numeric_cols:
        hints.append(
            f"Bloc quantitatif ({len(numeric_cols)} variables) : une ACP dédiée peut clarifier "
            "les redondances (notes G1/G2/G3, paires .m/.p) — voir heatmap."
        )
    if qual_cols:
        hints.append(
            f"Bloc qualitatif/binaire ({len(qual_cols)} variables) : une ACM peut approfondir "
            "les modalités qui structurent les profils (voir carte des modalités)."
        )
    if color_col:
        hints.append(
            f"Le nuage coloré par « {color_col} » permet de comparer des sous-groupes ; "
            "si des séparations visibles, croiser avec des tableaux de profils."
        )

    for dim, label in [(0, "axe 1"), (1, "axe 2")]:
        top = contrib.iloc[:, dim].nlargest(5)
        vars_str = ", ".join(f"{idx} ({val:.2%})" if val <= 1 else f"{idx}" for idx, val in top.items())
        hints.append(f"Principaux contributeurs {label} : {vars_str}.")

    return hints


def write_summary(
    path: Path,
    n_obs: int,
    n_vars: int,
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
    eigen_df: pd.DataFrame,
    contrib: pd.DataFrame,
    color_col: str | None,
) -> None:
    qual_cols = categorical_cols + binary_cols
    lines = [
        "=" * 60,
        "RÉSUMÉ FAMD — exploration globale",
        "=" * 60,
        "",
        f"Mode exploration : {EXPLORATORY_MODE}",
        "Toutes les variables du CSV sont actives (G3.m et G3.p incluses).",
        "",
        f"Nombre d'observations : {n_obs}",
        f"Nombre de variables   : {n_vars}",
        "",
        "Variables numériques :",
        "  " + ", ".join(numeric_cols) if numeric_cols else "  (aucune)",
        "",
        "Variables catégorielles :",
        "  " + ", ".join(categorical_cols) if categorical_cols else "  (aucune)",
        "",
        "Variables binaires :",
        "  " + ", ".join(binary_cols) if binary_cols else "  (aucune)",
        "",
        "Inertie expliquée (% de variance) — 5 premiers axes :",
    ]

    def _to_float(val) -> float:
        if isinstance(val, (int, float, np.floating)):
            return float(val)
        s = str(val).strip().rstrip("%")
        return float(s)

    for i in range(min(5, len(eigen_df))):
        row = eigen_df.iloc[i]
        ev = _to_float(row.get("eigenvalue", np.nan))
        pct = _to_float(row.get("% of variance", np.nan))
        cum = _to_float(row.get("% of variance (cumulative)", np.nan))
        lines.append(f"  Axe {i + 1} : eigenvalue={ev:.4f}, variance={pct:.2f}%, cumul={cum:.2f}%")

    lines.extend(["", "Variables les plus contributives :", ""])
    for dim, name in [(0, "Axe 1"), (1, "Axe 2")]:
        col = contrib.columns[dim]
        top = contrib[col].nlargest(10)
        lines.append(f"  {name} :")
        for var, val in top.items():
            fmt = f"{val:.2%}" if val <= 1 else f"{val:.4f}"
            lines.append(f"    - {var} : {fmt}")

    lines.extend(["", "Pistes d'analyses ultérieures", "-" * 40])
    for hint in build_analysis_hints(eigen_df, contrib, numeric_cols, qual_cols, color_col):
        lines.append(f"  • {hint}")

    lines.extend(
        [
            "",
            "Recommandations d'interprétation",
            "-" * 40,
            "  • Éboulis : retenir les axes avant le « coude » pour les graphiques suivants.",
            "  • Nuage des individus : proximité = profils similaires sur l'ensemble des variables.",
            "  • Contributions : variables qui structurent chaque axe (pas causalité).",
            "  • Paires .m / .p : redondance possible entre maths et portugais — information, pas exclusion.",
            "  • Cette FAMD est une étape 0 avant ACP (quanti) ou ACM (quali) plus ciblées.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    df = load_data()
    if not EXPLORATORY_MODE:
        print("Attention : EXPLORATORY_MODE désactivé.", file=sys.stderr)

    typology = build_typology(df)
    numeric_cols, categorical_cols, binary_cols = get_famd_groups(typology)
    qual_cols = categorical_cols + binary_cols

    print(f"Variables numériques ({len(numeric_cols)}) : {numeric_cols}")
    print(f"Variables catégorielles ({len(categorical_cols)}) : {categorical_cols}")
    print(f"Variables binaires ({len(binary_cols)}) : {binary_cols}")

    df_prepared, df_imputed = impute_and_prepare(df, numeric_cols, qual_cols)

    famd = fit_famd(df_prepared)

    eigen_df = get_eigenvalues_summary(famd)
    row_coords = get_row_coordinates(famd, df_prepared)
    col_coords = get_column_coordinates(famd)
    contrib = get_column_contributions(famd)
    corr_quanti = get_column_correlations(famd, numeric_cols)

    # Exports CSV
    eigen_path = OUTPUT_DIR / "eigenvalues.csv"
    eigen_df.reset_index().to_csv(eigen_path, index=False)
    generated.append(eigen_path)

    ind_path = OUTPUT_DIR / "individual_coordinates.csv"
    row_coords.to_csv(ind_path)
    generated.append(ind_path)

    var_path = OUTPUT_DIR / "variable_coordinates.csv"
    col_coords.to_csv(var_path)
    generated.append(var_path)

    color_col = pick_color_column(df_prepared)

    # Graphiques
    plots = [
        ("01_scree_plot.png", lambda: plot_scree(eigen_df, OUTPUT_DIR / "01_scree_plot.png")),
        (
            "02_individuals_map_dim1_dim2.png",
            lambda: plot_individuals(row_coords, OUTPUT_DIR / "02_individuals_map_dim1_dim2.png"),
        ),
        (
            "03_individuals_colored.png",
            lambda: plot_individuals(
                row_coords,
                OUTPUT_DIR / "03_individuals_colored.png",
                color_series=df_prepared[color_col].astype(str) if color_col else None,
                title=f"Individus colorés par {color_col}" if color_col else "Individus",
            ),
        ),
        (
            "04_variable_contributions.png",
            lambda: plot_variable_contributions(contrib, OUTPUT_DIR / "04_variable_contributions.png"),
        ),
    ]

    cat_coords = compute_category_coordinates(row_coords, df_prepared, qual_cols)
    plots.extend(
        [
            (
                "05_categories_map.png",
                lambda: plot_categories_map(cat_coords, OUTPUT_DIR / "05_categories_map.png"),
            ),
            (
                "06_quantitative_variables_map.png",
                lambda: plot_correlation_circle(
                    corr_quanti if corr_quanti is not None else pd.DataFrame(),
                    OUTPUT_DIR / "06_quantitative_variables_map.png",
                    numeric_cols,
                ),
            ),
            (
                "07_numeric_correlation_heatmap.png",
                lambda: plot_numeric_heatmap(
                    df_imputed[numeric_cols] if numeric_cols else df_imputed.iloc[:, :0],
                    OUTPUT_DIR / "07_numeric_correlation_heatmap.png",
                ),
            ),
            (
                "08_famd_biplot.png",
                lambda: plot_biplot(row_coords, col_coords, contrib, OUTPUT_DIR / "08_famd_biplot.png"),
            ),
        ]
    )

    for name, plot_fn in plots:
        plot_fn()
        generated.append(OUTPUT_DIR / name)

    summary_path = OUTPUT_DIR / "famd_summary.txt"
    write_summary(
        summary_path,
        n_obs=len(df_prepared),
        n_vars=len(df_prepared.columns),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        eigen_df=eigen_df,
        contrib=contrib,
        color_col=color_col,
    )
    generated.append(summary_path)

    print("\n" + "=" * 60)
    print("Fichiers générés :")
    for p in generated:
        print(f"  {p.resolve()}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
