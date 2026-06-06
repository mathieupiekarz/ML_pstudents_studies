"""
regression_inference.py — Régression linéaire inférentielle (cible quantitative).

Fournit, à partir d'une matrice de conception encodée :
  - l'ajustement par moindres carrés (OLS) ;
  - la table d'ANOVA de régression et le test F global (significativité du R²) ;
  - les tests t de significativité des coefficients ;
  - des tests d'hypothèses linéaires (dont la nullité conjointe de cluster_id) ;
  - des diagnostics (résidus, QQ, distance de Cook, VIF, Durbin-Watson) ;
  - une sélection de variables (Lasso + pas-à-pas par AIC).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


@dataclass
class RegressionInference:
    summary_text: str
    r2: float
    r2_adj: float
    f_stat: float
    f_pvalue: float
    coefficients: pd.DataFrame
    anova: pd.DataFrame
    cluster_test: dict | None
    durbin_watson: float
    vif: pd.DataFrame
    lasso_selected: list[str]
    stepwise_selected: list[str]
    notes: list[str] = field(default_factory=list)


def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X, has_constant="add")
    arr = Xc.to_numpy(dtype=float)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, col in enumerate(Xc.columns):
            if col == "const":
                continue
            try:
                v = float(variance_inflation_factor(arr, i))
                if not np.isfinite(v):
                    v = np.inf
            except Exception:
                v = np.nan
            rows.append({"variable": col, "VIF": v})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False, na_position="last")


def _stepwise_aic(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """Sélection ascendante par AIC."""
    remaining = list(X.columns)
    selected: list[str] = []
    current_aic = np.inf
    improved = True
    while remaining and improved:
        improved = False
        scores = []
        for cand in remaining:
            cols = selected + [cand]
            model = sm.OLS(y, sm.add_constant(X[cols], has_constant="add")).fit()
            scores.append((model.aic, cand))
        scores.sort()
        best_aic, best_cand = scores[0]
        if best_aic < current_aic - 1e-6:
            current_aic = best_aic
            selected.append(best_cand)
            remaining.remove(best_cand)
            improved = True
    return selected


def run_regression_inference(
    X: pd.DataFrame,
    y: pd.Series,
    cluster_columns: list[str],
) -> RegressionInference:
    """Ajuste l'OLS et calcule l'ensemble des diagnostics inférentiels."""
    notes: list[str] = []
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(np.asarray(y, dtype=float), Xc).fit()

    coeff = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p_value": model.pvalues,
        "ci_low": model.conf_int()[0],
        "ci_high": model.conf_int()[1],
    })
    coeff.index.name = "variable"

    # ANOVA de régression : décomposition SS modèle / résidu / total
    ss_model = float(model.ess)
    ss_resid = float(model.ssr)
    ss_total = ss_model + ss_resid
    df_model = int(model.df_model)
    df_resid = int(model.df_resid)
    anova = pd.DataFrame({
        "source": ["regression", "residus", "total"],
        "somme_carres": [ss_model, ss_resid, ss_total],
        "ddl": [df_model, df_resid, df_model + df_resid],
        "carre_moyen": [ss_model / df_model if df_model else np.nan,
                        ss_resid / df_resid if df_resid else np.nan, np.nan],
        "F": [model.fvalue, np.nan, np.nan],
        "p_value": [model.f_pvalue, np.nan, np.nan],
    })

    # Test d'hypothèse linéaire : nullité conjointe des coefficients de cluster_id
    cluster_test = None
    present = [c for c in cluster_columns if c in Xc.columns]
    if present:
        hypotheses = ", ".join(f"{c} = 0" for c in present)
        try:
            ftest = model.f_test(hypotheses)
            cluster_test = {
                "hypothese": hypotheses,
                "F": float(np.ravel(ftest.fvalue)[0]),
                "p_value": float(ftest.pvalue),
                "ddl_num": int(np.ravel(ftest.df_num)[0]),
                "ddl_den": int(ftest.df_denom),
            }
        except Exception as exc:  # pragma: no cover
            notes.append(f"Test d'hypothèse cluster_id échoué : {exc}")

    dw = float(durbin_watson(model.resid))
    vif = _vif_table(X)

    # Sélection de variables
    lasso = LassoCV(max_iter=5000, random_state=42)
    lasso.fit(X.to_numpy(dtype=float), np.asarray(y, dtype=float))
    lasso_selected = [c for c, co in zip(X.columns, lasso.coef_) if abs(co) > 1e-8]

    stepwise_selected = _stepwise_aic(X, pd.Series(np.asarray(y, dtype=float)))

    return RegressionInference(
        summary_text=str(model.summary()),
        r2=float(model.rsquared),
        r2_adj=float(model.rsquared_adj),
        f_stat=float(model.fvalue),
        f_pvalue=float(model.f_pvalue),
        coefficients=coeff,
        anova=anova,
        cluster_test=cluster_test,
        durbin_watson=dw,
        vif=vif,
        lasso_selected=lasso_selected,
        stepwise_selected=stepwise_selected,
        notes=notes,
    )


def plot_diagnostics(X: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    """4 graphiques : résidus vs ajustés, QQ-plot, distance de Cook, VIF."""
    Xc = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = sm.OLS(np.asarray(y, dtype=float), Xc).fit()
        influence = model.get_influence()
        cooks = np.nan_to_num(influence.cooks_distance[0], nan=0.0, posinf=0.0)
    fitted = model.fittedvalues
    resid = model.resid

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.scatter(fitted, resid, s=15, alpha=0.6)
    ax.axhline(0, color="crimson", ls="--", lw=1)
    ax.set_title("Résidus vs valeurs ajustées")
    ax.set_xlabel("Valeurs ajustées")
    ax.set_ylabel("Résidus")

    ax = axes[0, 1]
    sm.qqplot(resid, line="45", fit=True, ax=ax)
    ax.set_title("QQ-plot des résidus")

    ax = axes[1, 0]
    ax.stem(np.arange(len(cooks)), cooks, markerfmt=",")
    ax.axhline(4 / len(cooks), color="crimson", ls="--", lw=1,
               label="seuil 4/n")
    ax.set_title("Distance de Cook")
    ax.set_xlabel("Individu")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    vif = _vif_table(X).head(15)
    vif = vif[np.isfinite(vif["VIF"])]
    if len(vif):
        ax.barh(vif["variable"], vif["VIF"], color="steelblue")
        ax.axvline(5, color="orange", ls="--", lw=1, label="VIF=5")
        ax.axvline(10, color="crimson", ls="--", lw=1, label="VIF=10")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "VIF non calculables\n(colinéarité parfaite)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("VIF (multicolinéarité, top 15)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
