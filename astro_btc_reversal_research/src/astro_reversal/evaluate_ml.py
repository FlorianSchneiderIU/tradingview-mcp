"""Evaluation for M3 pivot-window models (proposal section 21).

Because pivots are rare, the headline metrics are PR-AUC, precision@K and lift@K,
alongside ROC-AUC and the Brier score (calibration). Everything is computed on
pooled out-of-sample predictions from walk-forward folds, plus the final holdout.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss

from . import reuse


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> dict:
    n = len(y_true)
    base = float(np.mean(y_true)) if n else float("nan")
    if n == 0:
        return {"k_frac": k_frac, "precision": float("nan"), "lift": float("nan"), "n_selected": 0}
    k = max(1, int(round(n * k_frac)))
    top = np.argsort(y_score)[::-1][:k]
    prec = float(np.mean(y_true[top]))
    lift = float(prec / base) if base > 0 else float("nan")
    return {"k_frac": k_frac, "precision": prec, "lift": lift, "n_selected": int(k)}


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray,
                           k_fracs=(0.01, 0.05, 0.10)) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    base = float(np.mean(y_true)) if y_true.size else float("nan")
    metrics = {
        "n": int(y_true.size),
        "base_rate": base,
        "pr_auc": reuse.safe_average_precision(y_true, y_score),
        "roc_auc": reuse.safe_roc_auc(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "precision_at": {f"top_{kf:.2f}": precision_at_k(y_true, y_score, kf) for kf in k_fracs},
    }
    return metrics


def walk_forward_predictions(model_name, build_model, X, y, folds):
    """Fit per fold, return pooled OOS (y, score) plus per-fold PR-AUC."""
    pooled_y: list[int] = []
    pooled_s: list[float] = []
    fold_pr: list[float] = []
    for train_idx, test_idx in folds:
        model = build_model(model_name)
        if len(np.unique(y[train_idx])) < 2:
            continue
        model.fit(X.iloc[train_idx], y[train_idx])
        score = model.predict_proba(X.iloc[test_idx])[:, 1]
        pooled_y.extend(y[test_idx].tolist())
        pooled_s.extend(score.tolist())
        fold_pr.append(reuse.safe_average_precision(y[test_idx], score))
    return np.array(pooled_y, dtype=int), np.array(pooled_s, dtype=float), fold_pr


def holdout_predictions(model_name, build_model, X, y, dev_idx, holdout_idx, embargo):
    """Train once on dev (minus embargo tail), score the holdout once."""
    train = np.asarray(dev_idx, dtype=int)
    if embargo > 0 and train.size > embargo:
        train = train[:-embargo]
    if len(np.unique(y[train])) < 2 or holdout_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    model = build_model(model_name)
    model.fit(X.iloc[train], y[train])
    score = model.predict_proba(X.iloc[holdout_idx])[:, 1]
    return y[holdout_idx], score
