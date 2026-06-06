"""
phase_b.py — Modélisation supervisée confirmatoire.

Orchestre :
  B.1 préparation (encodage par type + split stratifié 80/20) ;
  B.2 panel de modèles (models.py) ;
  B.3 régression linéaire inférentielle si cible quantitative ;
  B.4 post-élagage d'arbre ;
  B.5 importances (Gini + permutation) ;
  B.6 estimation du risque (evaluation.py) + sélection de modèle ;
  B.7 rôle de cluster_id (avec vs sans) ;
  règles de décision (Bayes / Neyman-Pearson) si cible binaire.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

import evaluation as ev
import models as mdl
import regression_inference as reginf

RANDOM_STATE = 42


@dataclass
class PhaseBResult:
    target: str
    kind: str
    comparison: pd.DataFrame
    best_model: str
    importances: pd.DataFrame
    cluster_id_role: dict
    coefficients: pd.DataFrame | None = None
    regression: reginf.RegressionInference | None = None
    brant: mdl.BrantResult | None = None
    decision_rules: dict | None = None
    tree_info: dict | None = None
    notes: list[str] = field(default_factory=list)


# ── Encodage ──────────────────────────────────────────────────────────────────

def _encode_features(
    df: pd.DataFrame, typology: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode les explicatives + cluster_id en DataFrame numérique nommé
    (non standardisé). Retourne (X_encoded, cluster_columns).
    """
    type_map = dict(zip(typology["variable"], typology["type_retenu"]))
    blocks: list[pd.DataFrame] = []

    nominal_cols, ordinal_cols, binary_cols, quant_cols = [], [], [], []
    for col in feature_cols:
        kind = type_map.get(col, "quantitative")
        if kind == "nominale":
            nominal_cols.append(col)
        elif kind == "ordinale":
            ordinal_cols.append(col)
        elif kind == "binaire":
            binary_cols.append(col)
        else:
            quant_cols.append(col)

    # Quantitatives + ordinales : numériques (ordinal encoding = codes ordonnés)
    for col in quant_cols + ordinal_cols:
        blocks.append(pd.to_numeric(df[col], errors="coerce").to_frame(col))

    # Binaires : code 0/1
    for col in binary_cols:
        codes, _ = pd.factorize(df[col].astype(str))
        blocks.append(pd.Series(codes, name=col).to_frame())

    # Nominales : one-hot (drop_first pour éviter le piège des variables muettes)
    if nominal_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        arr = ohe.fit_transform(df[nominal_cols].astype(str))
        names = ohe.get_feature_names_out(nominal_cols)
        blocks.append(pd.DataFrame(arr, columns=names, index=df.index))

    # cluster_id : one-hot (drop_first : k-1 muettes, référence = 1er cluster)
    cluster_dummies = pd.get_dummies(
        df["cluster_id"].astype(int), prefix="cluster", drop_first=True
    ).astype(float)
    cluster_columns = list(cluster_dummies.columns)
    blocks.append(cluster_dummies)

    X = pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True))
    return X, cluster_columns


def _encode_target(y: pd.Series, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (y_encodé, classes). Quantitative : float ; sinon labels triés."""
    if kind == "quantitative":
        return y.to_numpy(dtype=float), np.array([])
    classes = np.array(sorted(y.dropna().unique(), key=lambda v: (isinstance(v, str), v)))
    return y.to_numpy(), classes


def _stratify_vector(y: np.ndarray, kind: str) -> np.ndarray | None:
    if kind == "quantitative":
        try:
            return pd.qcut(y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop")
        except Exception:
            return None
    return y


# ── Importances ───────────────────────────────────────────────────────────────

def _importances(kind, X_std, y, feature_names) -> pd.DataFrame:
    if kind == "quantitative":
        rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X_std, y)
        perm = permutation_importance(rf, X_std, y, n_repeats=10,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    out = pd.DataFrame({
        "variable": feature_names,
        "gini": rf.feature_importances_,
        "permutation": perm.importances_mean,
    }).sort_values("permutation", ascending=False).reset_index(drop=True)
    return out


# ── Rôle de cluster_id ────────────────────────────────────────────────────────

def _cluster_id_role(kind, X_df, y, classes, cluster_columns) -> dict:
    """Compare le meilleur modèle flexible (RF) avec vs sans cluster_id."""
    metric = ev._primary_metric_name(kind)
    if kind == "quantitative":
        model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

    scaler = StandardScaler()
    X_with = scaler.fit_transform(X_df.to_numpy(dtype=float))
    X_wo_df = X_df.drop(columns=cluster_columns)
    X_without = StandardScaler().fit_transform(X_wo_df.to_numpy(dtype=float))

    cv_with = ev._cv_metrics(model, X_with, y, kind, classes).get(metric, np.nan)
    cv_without = ev._cv_metrics(model, X_without, y, kind, classes).get(metric, np.nan)

    better = (cv_with > cv_without) if ev._greater_is_better(kind) else (cv_with < cv_without)
    return {
        "metric": metric,
        "cv_avec_cluster_id": float(cv_with),
        "cv_sans_cluster_id": float(cv_without),
        "cluster_id_ameliore": bool(better),
    }


# ── Arbre élagué ──────────────────────────────────────────────────────────────

def _pruned_tree(kind, X_std, y, classes, out_path) -> dict:
    tree_cls = DecisionTreeRegressor if kind == "quantitative" else DecisionTreeClassifier
    scoring = "neg_root_mean_squared_error" if kind == "quantitative" else "accuracy"
    pruned, alpha, depth, leaves, curve = mdl.prune_tree(tree_cls, X_std, y, scoring)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    alphas = [c[0] for c in curve]
    scores = [c[1] for c in curve]
    axes[0].plot(alphas, scores, "o-")
    axes[0].axvline(alpha, color="crimson", ls="--", label=f"alpha={alpha:.4g}")
    axes[0].set_xlabel("ccp_alpha")
    axes[0].set_ylabel(f"score CV ({scoring})")
    axes[0].set_title("Élagage par coût-complexité")
    axes[0].legend(fontsize=8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_tree(pruned, max_depth=3, filled=True, fontsize=7,
                  class_names=None if kind == "quantitative" else [str(c) for c in classes],
                  ax=axes[1])
    axes[1].set_title(f"Arbre élagué (prof={depth}, feuilles={leaves})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"ccp_alpha": alpha, "profondeur": depth, "n_feuilles": leaves}


# ── Règles de décision (binaire) ──────────────────────────────────────────────

def _decision_rules(X_tr, y_tr, X_te, y_te, classes) -> dict:
    from sklearn.linear_model import LogisticRegression

    pos_label = classes[1]
    y_tr_bin = (y_tr == pos_label).astype(int)
    y_te_bin = (y_te == pos_label).astype(int)

    clf = LogisticRegression(max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr_bin)
    proba_te = clf.predict_proba(X_te)

    # Règle de Bayes avec coûts asymétriques (ex. coût FN double du FP)
    cost = np.array([[0.0, 1.0], [2.0, 0.0]])
    bayes_pred = mdl.bayes_decision(proba_te, cost)
    bayes_acc = float((bayes_pred == y_te_bin).mean())

    # Neyman-Pearson : FPR <= 10%
    np_rule = mdl.neyman_pearson_threshold(y_te_bin, proba_te[:, 1], target_fpr=0.10)
    return {
        "classe_positive": str(pos_label),
        "bayes_cout_matrice": cost.tolist(),
        "bayes_accuracy": bayes_acc,
        "neyman_pearson": {
            "target_fpr": np_rule.target_fpr,
            "seuil": np_rule.threshold,
            "fpr_obtenu": np_rule.achieved_fpr,
            "tpr_obtenu": np_rule.achieved_tpr,
        },
    }


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_phase_b(
    df: pd.DataFrame,
    typology: pd.DataFrame,
    target: str,
    kind: str,
    feature_cols: list[str],
    out_dir: Path,
) -> PhaseBResult:
    notes: list[str] = []
    X_df, cluster_columns = _encode_features(df, typology, feature_cols)
    feature_names = list(X_df.columns)
    y, classes = _encode_target(df[target], kind)

    # B.1 split holdout stratifié
    strat = _stratify_vector(y, kind)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_df.to_numpy(dtype=float))
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_std, y, test_size=0.20, random_state=RANDOM_STATE, stratify=strat)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_std, y, test_size=0.20, random_state=RANDOM_STATE)
        notes.append("Stratification impossible (effectifs faibles) : split simple.")

    # B.2 + B.6 panel + estimation du risque
    panel = mdl.build_regressors() if kind == "quantitative" else mdl.build_classifiers(kind)
    evaluations = [
        ev.evaluate_model(name, model, X_tr, y_tr, X_te, y_te, X_std, y, kind, classes)
        for name, model in panel.items()
    ]
    comparison = ev.build_comparison_table(evaluations, kind)
    best = ev.select_best(evaluations, kind)

    # B.5 importances
    importances = _importances(kind, X_std, y, feature_names)

    # B.7 rôle de cluster_id
    cluster_role = _cluster_id_role(kind, X_df, y, classes, cluster_columns)

    # B.4 arbre élagué
    tree_info = _pruned_tree(kind, X_std, y, classes, out_dir / "B_tree_pruned.png")

    result = PhaseBResult(
        target=target, kind=kind, comparison=comparison, best_model=best,
        importances=importances, cluster_id_role=cluster_role,
        tree_info=tree_info, notes=notes,
    )

    # B.3 régression linéaire inférentielle (quantitative)
    if kind == "quantitative":
        reg = reginf.run_regression_inference(X_df, pd.Series(y), cluster_columns)
        reginf.plot_diagnostics(X_df, pd.Series(y), out_dir / "B_regression_diagnostics.png")
        result.regression = reg
        result.coefficients = reg.coefficients

    # Logistique ordinale : coefficients + test de Brant
    if kind == "ordinale":
        try:
            ord_model = mdl.OrdinalLogistic().fit(X_std, y)
            coef = pd.DataFrame({
                "variable": feature_names + [f"seuil_{i}" for i in
                                             range(len(ord_model.result_.params) - len(feature_names))],
                "coef": ord_model.result_.params,
            })
            result.coefficients = coef
        except Exception as exc:
            notes.append(f"Logistique ordinale non ajustée : {exc}")
        result.brant = mdl.brant_test(X_std, y, feature_names)

    # Coefficients logistique (binaire / nominale)
    if kind in ("binaire", "nominale"):
        from sklearn.linear_model import LogisticRegression
        name, interp = mdl.interpretable_classifier(kind)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interp.fit(X_std, y)
        coef_arr = np.atleast_2d(interp.coef_)
        coef = pd.DataFrame(coef_arr.T, index=feature_names,
                            columns=[f"classe_{c}" for c in interp.classes_[:coef_arr.shape[0]]]
                            if coef_arr.shape[0] > 1 else ["coef"])
        coef.index.name = "variable"
        result.coefficients = coef.reset_index()

    # Règles de décision (binaire)
    if kind == "binaire" and len(classes) == 2:
        result.decision_rules = _decision_rules(X_tr, y_tr, X_te, y_te, classes)

    return result
