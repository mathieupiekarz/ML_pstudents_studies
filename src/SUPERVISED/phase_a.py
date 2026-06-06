"""
phase_a.py — Validation exploratoire des clusters vis-à-vis de la cible.

Pour chaque variable cible, on teste si les clusters capturent des différences
réelles, avec un test adapté au type de la cible et une taille d'effet :
  - quantitative : ANOVA (si normalité + homoscédasticité) sinon Kruskal-Wallis ; η²
  - ordinale     : Kruskal-Wallis + post-hoc de Dunn (Bonferroni) ; η² sur rangs
  - nominale/bin : Chi² (ou Fisher exact si 2x2 à faibles effectifs) ; V de Cramér
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

ALPHA = 0.05


@dataclass
class PhaseAResult:
    target: str
    kind: str
    test_name: str
    statistic: float
    p_value: float
    effect_name: str
    effect_value: float
    significant: bool
    table: pd.DataFrame              # contingence ou moyennes par cluster
    posthoc: pd.DataFrame | None = None
    discriminant_features: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ── Tailles d'effet ───────────────────────────────────────────────────────────

def _eta_squared_from_groups(groups: list[np.ndarray]) -> float:
    """η² = SS_inter / SS_total (sur valeurs ou rangs)."""
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_total = ((all_values - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0
    )
    return float(ss_between / ss_total)


def _cramers_v(contingency: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.sum()
    if n == 0:
        return 0.0
    r, k = contingency.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


# ── Tests par type ────────────────────────────────────────────────────────────

def _groups_by_cluster(df: pd.DataFrame, target: str) -> list[np.ndarray]:
    return [
        df.loc[df["cluster_id"] == c, target].to_numpy(dtype=float)
        for c in sorted(df["cluster_id"].unique())
    ]


def _test_quantitative(df: pd.DataFrame, target: str, on_ranks: bool) -> PhaseAResult:
    work = df.copy()
    kind = "ordinale" if on_ranks else "quantitative"
    if on_ranks:
        work[target] = work[target].rank()

    groups = _groups_by_cluster(work, target)
    groups = [g for g in groups if len(g) > 0]
    notes: list[str] = []

    use_anova = False
    if not on_ranks:
        # Conditions ANOVA : normalité intra-groupe (Shapiro) + homoscédasticité (Levene)
        normal = True
        for g in groups:
            if len(g) >= 3:
                try:
                    if stats.shapiro(g)[1] < ALPHA:
                        normal = False
                        break
                except ValueError:
                    normal = False
                    break
        homosced = True
        if len(groups) >= 2:
            try:
                homosced = stats.levene(*groups)[1] >= ALPHA
            except ValueError:
                homosced = False
        use_anova = normal and homosced
        notes.append(f"normalité={'ok' if normal else 'non'}, "
                     f"homoscédasticité={'ok' if homosced else 'non'}")

    if use_anova:
        stat, p = stats.f_oneway(*groups)
        test_name = "ANOVA (F)"
    else:
        stat, p = stats.kruskal(*groups)
        test_name = "Kruskal-Wallis (H)"

    effect = _eta_squared_from_groups(groups)

    posthoc = None
    if on_ranks and p < ALPHA:
        posthoc = sp.posthoc_dunn(
            work, val_col=target, group_col="cluster_id", p_adjust="bonferroni"
        )

    # Table : moyennes / écarts-types par cluster (sur valeurs d'origine)
    table = (
        df.groupby("cluster_id")[target]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )

    return PhaseAResult(
        target=target, kind=kind, test_name=test_name,
        statistic=float(stat), p_value=float(p),
        effect_name="eta2", effect_value=effect,
        significant=bool(p < ALPHA), table=table, posthoc=posthoc, notes=notes,
    )


def _test_categorical(df: pd.DataFrame, target: str, kind: str) -> PhaseAResult:
    contingency = pd.crosstab(df["cluster_id"], df[target])
    arr = contingency.to_numpy()

    notes: list[str] = []
    chi2, p, dof, expected = stats.chi2_contingency(arr)
    test_name = "Chi²"

    # Fisher exact si tableau 2x2 avec effectifs attendus faibles.
    if arr.shape == (2, 2) and (expected < 5).any():
        _, p = stats.fisher_exact(arr)
        test_name = "Fisher exact"
        notes.append("effectifs attendus < 5 → Fisher exact (2x2)")

    effect = _cramers_v(arr)

    return PhaseAResult(
        target=target, kind=kind, test_name=test_name,
        statistic=float(chi2), p_value=float(p),
        effect_name="cramers_v", effect_value=effect,
        significant=bool(p < ALPHA), table=contingency.reset_index(), notes=notes,
    )


# ── Variables discriminantes entre clusters ───────────────────────────────────

def _discriminant_features(
    df: pd.DataFrame, feature_cols: list[str], target: str
) -> list[tuple[str, float]]:
    """
    Force de liaison de chaque explicative avec le cluster :
    η² (numérique) ou V de Cramér (catégorielle). Exclut la cible.
    """
    scores: list[tuple[str, float]] = []
    clusters = df["cluster_id"]
    for col in feature_cols:
        if col == target:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 6:
            groups = [
                s[clusters == c].dropna().to_numpy(dtype=float)
                for c in sorted(clusters.unique())
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                scores.append((col, _eta_squared_from_groups(groups)))
        else:
            cont = pd.crosstab(clusters, s).to_numpy()
            if cont.shape[0] >= 2 and cont.shape[1] >= 2:
                scores.append((col, _cramers_v(cont)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def run_phase_a(
    df: pd.DataFrame,
    target: str,
    kind: str,
    feature_cols: list[str],
) -> PhaseAResult:
    """Exécute le test approprié + tailles d'effet + variables discriminantes."""
    if kind == "quantitative":
        result = _test_quantitative(df, target, on_ranks=False)
    elif kind == "ordinale":
        result = _test_quantitative(df, target, on_ranks=True)
    else:  # binaire / nominale
        result = _test_categorical(df, target, kind)

    result.discriminant_features = _discriminant_features(df, feature_cols, target)[:10]
    return result
