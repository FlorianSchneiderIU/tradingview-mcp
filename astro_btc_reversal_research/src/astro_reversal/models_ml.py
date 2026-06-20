"""Model factory for M3 (proposal section 16).

Logistic regression (L1 / L2 / elastic-net) and a tree model
(HistGradientBoosting) - both from scikit-learn, so no new dependency. Logistic
models are wrapped in a StandardScaler pipeline; the tree model is scale-free.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name: str):
    if name == "logistic_l2":
        clf = LogisticRegression(C=0.05, penalty="l2", solver="lbfgs",
                                 max_iter=2000, class_weight="balanced")
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if name == "logistic_l1":
        clf = LogisticRegression(C=0.05, penalty="l1", solver="liblinear",
                                 max_iter=2000, class_weight="balanced")
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if name == "logistic_elasticnet":
        clf = LogisticRegression(C=0.05, penalty="elasticnet", solver="saga",
                                 l1_ratio=0.5, max_iter=5000, class_weight="balanced")
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=50, random_state=42,
        )
    raise ValueError(f"unknown model {name!r}")


DEFAULT_MODELS = ["logistic_l2", "hgb"]
