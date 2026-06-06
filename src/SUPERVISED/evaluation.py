"""
evaluation.py — Estimation du risque et sélection de modèle.

Quatre stratégies d'estimation du risque sont fournies pour chaque modèle :
  - resubstitution (erreur d'apprentissage, biais optimiste) ;
  - ensemble de validation (holdout train/test) ;
  - validation croisée k-fold ;
  - bootstrap .632.

Les métriques sont choisies selon le type de la cible. La sélection de modèle
classe le panel sur le score de validation croisée.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold

RANDOM_STATE = 42


# ── Métriques par type ────────────────────────────────────────────────────────

def _primary_metric_name(kind: str) -> str:
    if kind == "quantitative":
        return "RMSE"
    if kind == "ordinale":
        return "MAE_rangs"
    return "F1_macro"


def _greater_is_better(kind: str) -> bool:
    return kind != "quantitative"  # RMSE/MAE : plus bas = meilleur


def compute_metrics(kind: str, y_true, y_pred, proba=None, classes=None) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if kind == "quantitative":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "RMSE": rmse,
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        }
    if kind == "ordinale":
        # MAE sur les rangs (codes ordonnés)
        order = {c: i for i, c in enumerate(classes)} if classes is not None else None
        if order is not None:
            yt = np.array([order.get(v, v) for v in y_true])
            yp = np.array([order.get(v, v) for v in y_pred])
        else:
            yt, yp = y_true, y_pred
        return {
            "MAE_rangs": float(mean_absolute_error(yt, yp)),
            "kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
    # nominale / binaire
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    if proba is not None:
        try:
            if proba.shape[1] == 2:
                out["AUC"] = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                out["AUC"] = float(roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro"))
        except Exception:
            out["AUC"] = float("nan")
    return out


def _score_for_selection(kind: str, metrics: dict[str, float]) -> float:
    return metrics.get(_primary_metric_name(kind), float("nan"))


# ── Estimation du risque ──────────────────────────────────────────────────────

@dataclass
class ModelEvaluation:
    name: str
    resubstitution: dict[str, float]
    holdout: dict[str, float]
    cross_val: dict[str, float]
    bootstrap_632: dict[str, float]
    failed: bool = False
    error: str = ""


def _safe_fit_predict(model, X_tr, y_tr, X_te):
    m = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        proba = None
        if hasattr(m, "predict_proba"):
            try:
                proba = np.asarray(m.predict_proba(X_te))
            except Exception:
                proba = None
    return pred, proba


def _cv_metrics(model, X, y, kind, classes) -> dict[str, float]:
    n_splits = 5
    if kind == "quantitative":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y)

    fold_metrics: list[dict[str, float]] = []
    for tr, te in split_iter:
        pred, proba = _safe_fit_predict(model, X[tr], y[tr], X[te])
        fold_metrics.append(compute_metrics(kind, y[te], pred, proba, classes))
    keys = fold_metrics[0].keys()
    return {k: float(np.nanmean([fm[k] for fm in fold_metrics])) for k in keys}


def _bootstrap_632(model, X, y, kind, classes, n_boot=25) -> dict[str, float]:
    n = len(y)
    rng = np.random.default_rng(RANDOM_STATE)
    # erreur de resubstitution (apprentissage complet)
    pred_in, proba_in = _safe_fit_predict(model, X, y, X)
    resub = compute_metrics(kind, y, pred_in, proba_in, classes)

    oob_scores: list[dict[str, float]] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(idx)] = False
        if oob_mask.sum() < 3:
            continue
        if kind != "quantitative" and len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        pred, proba = _safe_fit_predict(model, X[idx], y[idx], X[oob_mask])
        oob_scores.append(compute_metrics(kind, y[oob_mask], pred, proba, classes))

    if not oob_scores:
        return {k: float("nan") for k in resub}
    keys = resub.keys()
    out = {}
    for k in keys:
        oob = float(np.nanmean([s[k] for s in oob_scores if k in s]))
        # estimateur .632
        out[k] = float(0.368 * resub.get(k, np.nan) + 0.632 * oob)
    return out


def evaluate_model(
    name, model, X_tr, y_tr, X_te, y_te, X_all, y_all, kind, classes,
) -> ModelEvaluation:
    """Calcule les 4 estimations du risque pour un modèle."""
    try:
        # resubstitution
        pred_in, proba_in = _safe_fit_predict(model, X_tr, y_tr, X_tr)
        resub = compute_metrics(kind, y_tr, pred_in, proba_in, classes)
        # holdout
        pred_te, proba_te = _safe_fit_predict(model, X_tr, y_tr, X_te)
        holdout = compute_metrics(kind, y_te, pred_te, proba_te, classes)
        # CV + bootstrap sur l'ensemble complet
        cv = _cv_metrics(model, X_all, y_all, kind, classes)
        boot = _bootstrap_632(model, X_all, y_all, kind, classes)
        return ModelEvaluation(name, resub, holdout, cv, boot)
    except Exception as exc:
        return ModelEvaluation(name, {}, {}, {}, {}, failed=True, error=str(exc))


def build_comparison_table(
    evaluations: list[ModelEvaluation], kind: str
) -> pd.DataFrame:
    """Tableau récapitulatif : métrique principale par stratégie d'estimation."""
    metric = _primary_metric_name(kind)
    rows = []
    for ev in evaluations:
        rows.append({
            "modele": ev.name,
            f"{metric}_resub": ev.resubstitution.get(metric, float("nan")),
            f"{metric}_holdout": ev.holdout.get(metric, float("nan")),
            f"{metric}_cv": ev.cross_val.get(metric, float("nan")),
            f"{metric}_boot632": ev.bootstrap_632.get(metric, float("nan")),
            "echec": ev.failed,
        })
    df = pd.DataFrame(rows)
    sort_col = f"{metric}_cv"
    df = df.sort_values(sort_col, ascending=not _greater_is_better(kind),
                        na_position="last").reset_index(drop=True)
    return df


def select_best(evaluations: list[ModelEvaluation], kind: str) -> str:
    """Nom du meilleur modèle selon le score de validation croisée."""
    metric = _primary_metric_name(kind)
    valid = [ev for ev in evaluations if not ev.failed and metric in ev.cross_val]
    if not valid:
        return ""
    key = lambda ev: ev.cross_val[metric]
    return (max if _greater_is_better(kind) else min)(valid, key=key).name
