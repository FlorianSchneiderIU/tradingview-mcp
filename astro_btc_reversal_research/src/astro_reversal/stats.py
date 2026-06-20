"""Statistical helpers: rates, lift, binomial test, bootstrap CIs, FDR.

Kept deliberately small and dependency-light (numpy + scipy.stats).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def hit_rate(successes: np.ndarray | list) -> float:
    arr = np.asarray(successes, dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def lift(rate: float, baseline: float) -> float:
    if baseline is None or not math.isfinite(baseline) or baseline <= 0:
        return float("nan")
    return float(rate / baseline)


def binomial_test(k: int, n: int, p: float, alternative: str = "greater") -> float:
    """P-value that an observed k/n exceeds baseline rate p (one-sided by default)."""
    if n <= 0 or not math.isfinite(p):
        return float("nan")
    p = min(max(p, 1e-9), 1 - 1e-9)
    return float(stats.binomtest(int(k), int(n), p, alternative=alternative).pvalue)


def bootstrap_rate_ci(
    successes: np.ndarray | list,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    arr = np.asarray(successes, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return (lo, hi)


def bootstrap_diff_ci(
    successes_a: np.ndarray | list,
    successes_b: np.ndarray | list,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """CI for rate(A) - rate(B) via independent resampling."""
    a = np.asarray(successes_a, dtype=float)
    b = np.asarray(successes_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_a = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    boot_b = rng.choice(b, size=(n_boot, b.size), replace=True).mean(axis=1)
    diff = boot_a - boot_b
    return (float(np.quantile(diff, alpha / 2)), float(np.quantile(diff, 1 - alpha / 2)))


def benjamini_hochberg(pvalues: np.ndarray | list, alpha: float = 0.05) -> np.ndarray:
    """Return a boolean array of rejections under BH false-discovery-rate control."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    reject = np.zeros(n, dtype=bool)
    if n == 0:
        return reject
    finite = np.isfinite(p)
    order = np.argsort(np.where(finite, p, np.inf))
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = finite[order] & (ranked <= thresh)
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))
        reject[order[: cutoff + 1]] = finite[order[: cutoff + 1]]
    return reject


def empirical_pvalue(observed: float, null_samples: np.ndarray | list) -> float:
    """One-sided p-value: fraction of null >= observed (with +1 smoothing)."""
    null = np.asarray(null_samples, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not math.isfinite(observed):
        return float("nan")
    return float((np.sum(null >= observed) + 1) / (null.size + 1))
