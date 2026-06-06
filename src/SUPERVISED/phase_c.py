"""
phase_c.py — Confrontation des Phases A et B et conclusion automatique.

Construit pour chaque cible :
  C.1 un tableau de triangulation (variables discriminantes, sous-populations,
      force de la relation) avec une colonne de cohérence ;
  C.2 une conclusion automatique selon 4 scénarios ;
  C.3 des recommandations (variables actionnables, limites, pistes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from phase_a import PhaseAResult
from phase_b import PhaseBResult

OK = "OK"
WARN = "DIVERGENCE"


@dataclass
class PhaseCResult:
    target: str
    triangulation: pd.DataFrame
    conclusion: str
    scenario: str
    recommendations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _top_features(pairs: list[tuple[str, float]], n: int = 5) -> list[str]:
    return [name for name, _ in pairs[:n]]


def _b_relation_strength(b: PhaseBResult) -> tuple[str, float]:
    """Force de la relation côté B : R² (quant) ou meilleure métrique de classif."""
    metric = "RMSE" if b.kind == "quantitative" else (
        "MAE_rangs" if b.kind == "ordinale" else "F1_macro")
    best_row = b.comparison.iloc[0] if len(b.comparison) else None
    if b.kind == "quantitative" and b.regression is not None:
        return "R2", float(b.regression.r2)
    if best_row is not None:
        col = f"{metric}_cv"
        return metric, float(best_row.get(col, np.nan))
    return metric, float("nan")


def run_phase_c(a: PhaseAResult, b: PhaseBResult) -> PhaseCResult:
    # ── Données croisées ──────────────────────────────────────────────────
    a_features = _top_features(a.discriminant_features)
    b_features = b.importances["variable"].head(8).tolist()
    # rapprochement par nom de base (cluster_x ignoré)
    a_set = {f for f in a_features}
    b_set = {f.split("_")[0] for f in b_features}
    shared = a_set & {f for f in a_features if any(f in bf or bf in f for bf in b_features)}
    overlap = bool(a_set & b_set) or bool(shared)

    cluster_useful = b.cluster_id_role.get("cluster_id_ameliore", False)
    a_significant = a.significant

    b_metric_name, b_metric_val = _b_relation_strength(b)
    # prédiction "bonne" : R²>0.1 ou métrique de classif décente
    if b.kind == "quantitative":
        b_predicts = np.isfinite(b_metric_val) and b_metric_val > 0.10
    elif b.kind == "ordinale":
        b_predicts = np.isfinite(b_metric_val)  # MAE rangs : interprété en C
        # heuristique : MAE rangs < 1 = prédiction utile
        b_predicts = np.isfinite(b_metric_val) and b_metric_val < 1.0
    else:
        b_predicts = np.isfinite(b_metric_val) and b_metric_val > 0.5

    # ── C.1 triangulation ─────────────────────────────────────────────────
    rows = [
        {
            "critere": "Variables discriminantes",
            "phase_A": ", ".join(a_features) or "—",
            "phase_B": ", ".join(b_features) or "—",
            "coherence": OK if overlap else WARN,
        },
        {
            "critere": "Sous-populations détectées",
            "phase_A": "clusters significatifs" if a_significant else "clusters non significatifs",
            "phase_B": "cluster_id utile" if cluster_useful else "cluster_id redondant",
            "coherence": OK if (a_significant == cluster_useful) else WARN,
        },
        {
            "critere": "Force de la relation",
            "phase_A": f"{a.effect_name}={a.effect_value:.3f}",
            "phase_B": f"{b_metric_name}={b_metric_val:.3f}",
            "coherence": OK if (a_significant == b_predicts) else WARN,
        },
    ]
    triangulation = pd.DataFrame(rows)

    # ── C.2 scénario de conclusion ────────────────────────────────────────
    if a_significant and b_predicts and cluster_useful:
        scenario = "convergence"
        conclusion = (
            "L'analyse est robuste et validée. Les profils identifiés sont "
            "cohérents avec la variable cible : la Phase A et la Phase B convergent."
        )
    elif a_significant and b_predicts and not cluster_useful:
        scenario = "clusters_redondants"
        conclusion = (
            "Les variables brutes suffisent à prédire la cible. Les clusters sont "
            "utiles à l'interprétation mais redondants pour la prédiction "
            "(cluster_id n'améliore pas le modèle)."
        )
    elif (not a_significant) and b_predicts:
        scenario = "structure_non_alignee"
        conclusion = (
            "La structure latente ne capture pas la variable cible alors que les "
            "variables brutes la prédisent. Envisager une analyse non supervisée "
            "ciblée (en excluant explicitement la cible)."
        )
    elif a_significant and not b_predicts:
        scenario = "divergence"
        conclusion = (
            "La structure des données et la prédiction reposent sur des mécanismes "
            "différents : les clusters séparent la cible mais les modèles peinent à "
            "la prédire. Résultat à discuter."
        )
    else:
        scenario = "relation_faible"
        conclusion = (
            "Ni les clusters (Phase A) ni les modèles (Phase B) ne mettent en "
            "évidence de relation forte avec la cible. La cible semble peu liée "
            "aux variables disponibles."
        )

    # ── C.3 recommandations ───────────────────────────────────────────────
    actionable = set(config.ACTIONABLE_VARS)
    actionable_important = [
        f for f in b_features if f.split(".")[0].split("_")[0] in
        {a.split(".")[0] for a in actionable} or f in actionable
    ]
    recommendations = []
    if actionable_important:
        recommendations.append(
            "Variables actionnables influentes : " + ", ".join(actionable_important[:6])
        )
    else:
        recommendations.append(
            "Aucune variable actionnable ne ressort nettement parmi les plus importantes."
        )
    recommendations.append(
        "Limites : relations corrélationnelles (pas causales) ; valeurs manquantes "
        "imputées ; échantillon n=382 (prudence sur les sous-groupes)."
    )
    if scenario in ("structure_non_alignee", "divergence", "relation_faible"):
        recommendations.append(
            "Piste : relancer la réduction et le clustering en excluant la cible, "
            "ou tester d'autres valeurs de k / d'autres sources de réduction."
        )
    if b.kind == "quantitative" and b.regression is not None:
        high_vif = b.regression.vif[b.regression.vif["VIF"] > 10]["variable"].tolist()
        if high_vif:
            recommendations.append(
                "Multicolinéarité élevée (VIF>10) : " + ", ".join(high_vif[:6])
            )

    return PhaseCResult(
        target=a.target,
        triangulation=triangulation,
        conclusion=conclusion,
        scenario=scenario,
        recommendations=recommendations,
    )
