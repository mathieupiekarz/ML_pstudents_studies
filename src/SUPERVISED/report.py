"""
report.py — Écriture des rapports textuels des Phases A / B / C.
"""

from __future__ import annotations

from pathlib import Path

from phase_a import ALPHA, PhaseAResult
from phase_b import PhaseBResult
from phase_c import PhaseCResult


def _line(title: str) -> str:
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}\n"


def write_phase_a(res: PhaseAResult, out_dir: Path) -> None:
    lines = [_line(f"PHASE A — {res.target} ({res.kind})")]
    lines.append(f"Test utilisé      : {res.test_name}")
    lines.append(f"Statistique       : {res.statistic:.4f}")
    lines.append(f"p-valeur          : {res.p_value:.4g}")
    lines.append(f"Taille d'effet    : {res.effect_name} = {res.effect_value:.4f}")
    verdict = "OUI" if res.significant else "NON"
    lines.append(f"Significatif (α={ALPHA}) ? {verdict}")
    if res.notes:
        lines.append("Notes             : " + " ; ".join(res.notes))
    lines.append("\nConclusion Phase A :")
    if res.significant:
        lines.append(
            f"  Les clusters capturent des différences significatives sur {res.target} "
            f"(taille d'effet {res.effect_name}={res.effect_value:.3f})."
        )
    else:
        lines.append(
            f"  Les clusters ne capturent pas de différence significative sur {res.target}."
        )

    lines.append("\nVariables explicatives les plus discriminantes entre clusters :")
    for name, score in res.discriminant_features:
        lines.append(f"  {name:25s} : {score:.4f}")

    if res.posthoc is not None:
        lines.append("\nPost-hoc de Dunn (p-valeurs, Bonferroni) :")
        lines.append(res.posthoc.round(4).to_string())

    (out_dir / "report_phaseA.txt").write_text("\n".join(lines), encoding="utf-8")
    # Table associée
    res.table.to_csv(out_dir / "A_contingency_or_means.csv", index=False)


def write_phase_b(res: PhaseBResult, out_dir: Path) -> None:
    lines = [_line(f"PHASE B — {res.target} ({res.kind})")]
    lines.append("Comparaison des modèles (estimation du risque) :")
    lines.append(res.comparison.round(4).to_string(index=False))
    lines.append(f"\nMeilleur modèle (CV) : {res.best_model}")

    role = res.cluster_id_role
    lines.append("\nRôle de cluster_id :")
    lines.append(f"  {role['metric']} avec cluster_id  : {role['cv_avec_cluster_id']:.4f}")
    lines.append(f"  {role['metric']} sans cluster_id  : {role['cv_sans_cluster_id']:.4f}")
    lines.append(f"  cluster_id améliore le modèle ? {'OUI' if role['cluster_id_ameliore'] else 'NON'}")

    if res.tree_info:
        lines.append(
            f"\nArbre élagué : ccp_alpha={res.tree_info['ccp_alpha']:.4g}, "
            f"profondeur={res.tree_info['profondeur']}, "
            f"feuilles={res.tree_info['n_feuilles']}"
        )

    if res.brant is not None:
        b = res.brant
        lines.append("\nTest de Brant (odds proportionnels) :")
        if b.note:
            lines.append(f"  {b.note}")
        else:
            verdict = "respectée" if (b.omnibus_p > ALPHA) else "rejetée"
            lines.append(
                f"  χ² global = {b.omnibus_chi2:.3f} (df={b.omnibus_df}), "
                f"p = {b.omnibus_p:.4g} → hypothèse PO {verdict}"
            )

    if res.decision_rules:
        dr = res.decision_rules
        lines.append("\nRègles de décision (cible binaire) :")
        lines.append(f"  Classe positive            : {dr['classe_positive']}")
        lines.append(f"  Règle de Bayes (coûts) acc.: {dr['bayes_accuracy']:.4f}")
        npr = dr["neyman_pearson"]
        lines.append(
            f"  Neyman-Pearson (FPR<= {npr['target_fpr']}) : seuil={npr['seuil']:.3f}, "
            f"FPR={npr['fpr_obtenu']:.3f}, TPR={npr['tpr_obtenu']:.3f}"
        )

    if res.notes:
        lines.append("\nNotes : " + " ; ".join(res.notes))

    (out_dir / "report_phaseB.txt").write_text("\n".join(lines), encoding="utf-8")

    # CSV annexes
    res.comparison.to_csv(out_dir / "B_model_comparison.csv", index=False)
    res.importances.to_csv(out_dir / "B_importances.csv", index=False)
    import pandas as pd
    pd.DataFrame([res.cluster_id_role]).to_csv(out_dir / "B_cluster_id_role.csv", index=False)
    if res.coefficients is not None:
        res.coefficients.to_csv(out_dir / "B_reference_coefficients.csv", index=False)
    if res.regression is not None:
        (out_dir / "B_ols_summary.txt").write_text(
            _ols_text(res.regression), encoding="utf-8"
        )
    if res.decision_rules is not None:
        import json
        (out_dir / "B_decision_rules.txt").write_text(
            json.dumps(res.decision_rules, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _ols_text(reg) -> str:
    parts = [reg.summary_text, "\n\nANOVA de régression :", reg.anova.round(4).to_string(index=False)]
    parts.append(f"\nR² = {reg.r2:.4f} | R² ajusté = {reg.r2_adj:.4f}")
    parts.append(f"Test F global : F = {reg.f_stat:.3f}, p = {reg.f_pvalue:.4g}")
    parts.append(f"Durbin-Watson : {reg.durbin_watson:.3f}")
    if reg.cluster_test:
        ct = reg.cluster_test
        parts.append(
            f"\nTest d'hypothèse linéaire (nullité de cluster_id) : "
            f"F = {ct['F']:.3f}, p = {ct['p_value']:.4g} "
            f"(ddl {ct['ddl_num']}, {ct['ddl_den']})"
        )
    parts.append("\nSélection Lasso : " + ", ".join(reg.lasso_selected) or "(aucune)")
    parts.append("Sélection stepwise (AIC) : " + ", ".join(reg.stepwise_selected))
    parts.append("\nVIF (top) :\n" + reg.vif.head(10).round(3).to_string(index=False))
    if reg.notes:
        parts.append("\nNotes : " + " ; ".join(reg.notes))
    return "\n".join(parts)


def write_phase_c(res: PhaseCResult, out_dir: Path) -> None:
    lines = [_line(f"PHASE C — {res.target} : triangulation et conclusion")]
    lines.append("Tableau de triangulation :")
    lines.append(res.triangulation.to_string(index=False))
    lines.append(f"\nScénario détecté : {res.scenario}")
    lines.append("\nConclusion automatique :")
    lines.append(f"  {res.conclusion}")
    lines.append("\nRecommandations :")
    for r in res.recommendations:
        lines.append(f"  - {r}")
    (out_dir / "report_phaseC.txt").write_text("\n".join(lines), encoding="utf-8")
    res.triangulation.to_csv(out_dir / "C_triangulation.csv", index=False)


def write_synthese(
    target: str, a: PhaseAResult, b: PhaseBResult, c: PhaseCResult, out_dir: Path
) -> None:
    lines = [_line(f"SYNTHÈSE — {target}")]
    lines.append(f"Phase A : {a.test_name}, p={a.p_value:.4g}, "
                 f"{a.effect_name}={a.effect_value:.3f} → "
                 f"{'significatif' if a.significant else 'non significatif'}")
    metric, *_ = (b.comparison.columns[1].rsplit("_", 1)[0],) if len(b.comparison.columns) > 1 else ("?",)
    lines.append(f"Phase B : meilleur modèle = {b.best_model} ; "
                 f"cluster_id utile = {'oui' if b.cluster_id_role['cluster_id_ameliore'] else 'non'}")
    lines.append(f"Phase C : scénario = {c.scenario}")
    lines.append(f"\n{c.conclusion}")
    (out_dir / "synthese.txt").write_text("\n".join(lines), encoding="utf-8")
