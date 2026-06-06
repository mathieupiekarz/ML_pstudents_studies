"""
models.py — Panel de modèles supervisés + règles de décision.

Classification (binaire / nominale / ordinale) :
  classifieur euclidien (centroïdes), k-NN, bayésien naïf (indépendance
  conditionnelle), LDA, QDA, analyse discriminante régularisée (shrinkage /
  reg_param), régression logistique (binaire/multinomiale/L1), logistique
  ordinale (proportional odds) + test de Brant, arbre élagué, forêt aléatoire,
  gradient boosting.

Régression (quantitative) :
  moindres carrés, Ridge, Lasso, k-NN, arbre, forêt, gradient boosting.

Règles de décision (cible binaire) :
  règle de Bayes / minimisation du risque (matrice de coûts) ;
  règle de Neyman-Pearson (seuil à taux de faux positifs fixé).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LassoCV,
    LinearRegression,
    LogisticRegression,
    RidgeCV,
)
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestCentroid,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42


# ── Logistique ordinale (proportional odds) via statsmodels ──────────────────

class OrdinalLogistic(BaseEstimator, ClassifierMixin):
    """Wrapper sklearn de statsmodels OrderedModel (logit, proportional odds)."""

    def __init__(self, method: str = "bfgs", maxiter: int = 200):
        self.method = method
        self.maxiter = maxiter

    def fit(self, X, y):
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        mapping = {c: i for i, c in enumerate(self.classes_)}
        y_codes = np.array([mapping[v] for v in y])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = OrderedModel(y_codes, X, distr="logit")
            self.result_ = self.model_.fit(method=self.method,
                                           maxiter=self.maxiter, disp=False)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray(self.result_.model.predict(self.result_.params, exog=X))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


# ── Test de Brant (hypothèse des odds proportionnels) ────────────────────────

@dataclass
class BrantResult:
    omnibus_chi2: float
    omnibus_df: int
    omnibus_p: float
    per_variable: dict[str, tuple[float, float]]  # var -> (chi2, p) df=J-2
    note: str = ""


def brant_test(X: np.ndarray, y: np.ndarray, var_names: list[str]) -> BrantResult:
    """
    Test de Brant : compare les pentes de J-1 régressions logistiques binaires
    cumulées P(y > j). Sous l'hypothèse des odds proportionnels, ces pentes sont
    égales. Statistique de Wald globale (df = p*(J-2)) et par variable (df = J-2).
    """
    from scipy import stats as _st

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    cats = np.sort(np.unique(y))
    J = len(cats)
    n, p = X.shape

    if J < 3:
        return BrantResult(np.nan, 0, np.nan, {}, "J<3 : test de Brant non applicable.")

    Xc = np.column_stack([np.ones(n), X])  # intercept
    betas, pis = [], []
    for j in range(J - 1):
        z = (y > cats[j]).astype(float)
        try:
            lr = LogisticRegression(penalty=None, max_iter=1000)
            lr.fit(X, z)
        except Exception:
            return BrantResult(np.nan, 0, np.nan, {},
                               "Échec d'ajustement d'un logit binaire (Brant).")
        beta = np.concatenate([lr.intercept_, lr.coef_.ravel()])
        betas.append(beta)
        pis.append(lr.predict_proba(X)[:, 1])

    betas = np.array(betas)            # (J-1, p+1)
    slopes = betas[:, 1:]              # (J-1, p)

    # Variances/covariances (formules de Brant 1990)
    var_blocks: dict[tuple[int, int], np.ndarray] = {}
    for l in range(J - 1):
        Wl = pis[l] * (1 - pis[l])
        var_blocks[(l, l)] = np.linalg.pinv(Xc.T @ (Wl[:, None] * Xc))
    for l in range(J - 1):
        for m in range(l + 1, J - 1):
            Wlm = pis[l] * (1 - pis[m])  # l<m => π_l>=π_m (cumul décroissant)
            mid = Xc.T @ (Wlm[:, None] * Xc)
            var_blocks[(l, m)] = var_blocks[(l, l)] @ mid @ var_blocks[(m, m)]

    # Contrastes : différences successives des pentes entre modèles
    pp = p + 1
    rows = []
    diffs = []
    for l in range(J - 2):
        for k in range(p):
            row = np.zeros((J - 1) * pp)
            row[l * pp + (k + 1)] = 1.0
            row[(l + 1) * pp + (k + 1)] = -1.0
            rows.append(row)
            diffs.append(slopes[l, k] - slopes[l + 1, k])
    D = np.array(rows)
    delta = np.array(diffs)

    # Matrice de covariance complète empilée
    big = np.zeros(((J - 1) * pp, (J - 1) * pp))
    for l in range(J - 1):
        for m in range(J - 1):
            block = var_blocks[(l, m)] if l <= m else var_blocks[(m, l)].T
            big[l * pp:(l + 1) * pp, m * pp:(m + 1) * pp] = block

    omega = D @ big @ D.T
    try:
        chi2 = float(delta @ np.linalg.pinv(omega) @ delta)
    except np.linalg.LinAlgError:
        chi2 = np.nan
    df = p * (J - 2)
    p_global = float(_st.chi2.sf(chi2, df)) if np.isfinite(chi2) else np.nan

    # Par variable : contraste restreint à la variable k
    per_var: dict[str, tuple[float, float]] = {}
    for k in range(p):
        # contraste restreint à la variable k sur les cutpoints successifs
        rows_k = []
        diffs_k = []
        for l in range(J - 2):
            row = np.zeros((J - 1) * pp)
            row[l * pp + (k + 1)] = 1.0
            row[(l + 1) * pp + (k + 1)] = -1.0
            rows_k.append(row)
            diffs_k.append(slopes[l, k] - slopes[l + 1, k])
        Dk = np.array(rows_k)
        dk = np.array(diffs_k)
        omk = Dk @ big @ Dk.T
        try:
            c_k = float(dk @ np.linalg.pinv(omk) @ dk)
            pk = float(_st.chi2.sf(c_k, J - 2))
        except np.linalg.LinAlgError:
            c_k, pk = np.nan, np.nan
        name = var_names[k] if k < len(var_names) else f"x{k}"
        per_var[name] = (c_k, pk)

    return BrantResult(chi2, df, p_global, per_var)


# ── Panels de modèles ────────────────────────────────────────────────────────

def build_classifiers(kind: str) -> dict[str, BaseEstimator]:
    """Dictionnaire {nom: estimateur} pour une cible de classification."""
    multinomial = kind == "nominale"
    _ = multinomial  # le solveur gère le multinomial par défaut
    models: dict[str, BaseEstimator] = {
        "euclidien (centroides)": NearestCentroid(),
        "euclidien parcimonieux": NearestCentroid(shrink_threshold=0.2),
        "k-NN": KNeighborsClassifier(n_neighbors=15),
        "bayesien naif": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.0),
        "ADR (LDA shrinkage)": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "ADR (QDA reg)": QuadraticDiscriminantAnalysis(reg_param=0.5),
        "logistique": LogisticRegression(max_iter=2000),
        "logistique L1 (parcimonieuse)": LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000,
        ) if not multinomial else LogisticRegression(
            penalty="l1", solver="saga", max_iter=3000,
        ),
        "arbre (elague)": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "foret aleatoire": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if kind == "ordinale":
        models["logistique ordinale (PO)"] = OrdinalLogistic()
    return models


def build_regressors() -> dict[str, BaseEstimator]:
    """Dictionnaire {nom: estimateur} pour une cible quantitative."""
    return {
        "moindres carres (OLS)": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "Lasso (parcimonieux)": LassoCV(max_iter=5000, random_state=RANDOM_STATE),
        "k-NN": KNeighborsRegressor(n_neighbors=15),
        "arbre (elague)": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "foret aleatoire": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def interpretable_classifier(kind: str) -> tuple[str, BaseEstimator]:
    """Modèle interprétable de référence (coefficients) selon le type."""
    if kind == "ordinale":
        return "logistique ordinale (PO)", OrdinalLogistic()
    return "logistique", LogisticRegression(max_iter=2000)


# ── Post-élagage d'arbre par coût-complexité ─────────────────────────────────

def prune_tree(tree_cls, X, y, scoring_cv, cv=5):
    """
    Sélectionne ccp_alpha par validation croisée (contrôle de complexité).
    Retourne (meilleur_arbre_non_entraine, alpha, profondeur, n_feuilles, courbe).
    """
    from sklearn.model_selection import cross_val_score

    base = tree_cls(random_state=RANDOM_STATE)
    path = base.cost_complexity_pruning_path(X, y)
    alphas = path.ccp_alphas
    # éviter le dernier alpha (arbre réduit à la racine)
    alphas = alphas[:-1] if len(alphas) > 1 else alphas

    best_alpha, best_score, curve = 0.0, -np.inf, []
    for a in alphas:
        clf = tree_cls(random_state=RANDOM_STATE, ccp_alpha=a)
        score = cross_val_score(clf, X, y, cv=cv, scoring=scoring_cv).mean()
        curve.append((float(a), float(score)))
        if score > best_score:
            best_score, best_alpha = score, a

    pruned = tree_cls(random_state=RANDOM_STATE, ccp_alpha=best_alpha)
    pruned.fit(X, y)
    return pruned, float(best_alpha), int(pruned.get_depth()), int(pruned.get_n_leaves()), curve


# ── Règles de décision (cible binaire) ───────────────────────────────────────

def bayes_decision(proba: np.ndarray, cost_matrix: np.ndarray | None = None) -> np.ndarray:
    """
    Règle de Bayes / minimisation du risque.
    proba : (n, K) probabilités a posteriori.
    cost_matrix : (K, K), cost[i, j] = coût de prédire j quand la vraie classe est i.
    Défaut : coûts 0/1 -> argmax de la probabilité a posteriori.
    """
    proba = np.asarray(proba, dtype=float)
    K = proba.shape[1]
    if cost_matrix is None:
        return proba.argmax(axis=1)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    # risque attendu de prédire j = sum_i proba_i * cost[i, j]
    expected = proba @ cost_matrix  # (n, K)
    return expected.argmin(axis=1)


@dataclass
class NeymanPearson:
    threshold: float
    achieved_fpr: float
    achieved_tpr: float
    target_fpr: float


def neyman_pearson_threshold(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.10
) -> NeymanPearson:
    """
    Règle de Neyman-Pearson : seuil sur le score de la classe positive
    maximisant le taux de vrais positifs sous contrainte FPR <= target_fpr.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    feasible = fpr <= target_fpr
    if not feasible.any():
        idx = int(np.argmin(fpr))
    else:
        idx = int(np.argmax(np.where(feasible, tpr, -np.inf)))
    return NeymanPearson(
        threshold=float(thr[idx]),
        achieved_fpr=float(fpr[idx]),
        achieved_tpr=float(tpr[idx]),
        target_fpr=target_fpr,
    )
