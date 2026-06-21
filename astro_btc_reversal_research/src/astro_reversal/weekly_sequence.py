"""Does prior weeks' extreme timing predict this week's reversal window?

Builds a per-week sequence with lagged features (prior weeks' low/high timing,
range, retest, and this week's open vs prior levels) and a target for this week's
low timing. Provides transition matrices, autocorrelation, and inter-low spacing
so a model (or a rule) can be tested for a learnable weekly-timing cycle.

Leakage: targets are this-week's extremes (known only at week end -> labels). All
features use ONLY prior weeks plus this week's OPEN, so the model could run at the
Monday open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EARLY_FRAC = 0.40   # 'early' low = within the first 40% of the week (Mon-Wed)


def bucket3(frac: float) -> int:
    return 0 if frac < EARLY_FRAC else (1 if frac < 0.70 else 2)


def build_sequence(records: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    r = records.reset_index(drop=True).copy()
    r["low_early"] = (r["low_frac"] < EARLY_FRAC).astype(int)
    r["high_early"] = (r["high_frac"] < EARLY_FRAC).astype(int)
    r["low_bucket"] = r["low_frac"].map(bucket3)

    feat = pd.DataFrame(index=r.index)
    for L in range(1, n_lags + 1):
        feat[f"low_frac_l{L}"] = r["low_frac"].shift(L)
        feat[f"low_dow_l{L}"] = r["low_dow"].shift(L)
        feat[f"low_early_l{L}"] = r["low_early"].shift(L)
        feat[f"high_frac_l{L}"] = r["high_frac"].shift(L)
        feat[f"range_l{L}"] = r["weekly_range_pct"].shift(L)
        feat[f"low_first_l{L}"] = r["low_first"].shift(L).astype(float)
        feat[f"retest_low_l{L}"] = r["retest_low"].shift(L).astype(float)
    span = (r["high_price"].shift(1) - r["low_price"].shift(1))
    feat["open_vs_prevlow"] = (r["open"] - r["low_price"].shift(1)) / span
    feat["open_vs_prevhigh"] = (r["open"] - r["high_price"].shift(1)) / span
    feat["ret_prev_week"] = (r["close"].shift(1) / r["open"].shift(1) - 1.0)

    keep = ["week_start", "low_frac", "low_dow", "low_early", "low_bucket",
            "high_frac", "high_early", "low_time"]
    out = pd.concat([r[keep], feat], axis=1).dropna().reset_index(drop=True)
    return out


def transition_matrix(prev: np.ndarray, cur: np.ndarray, k: int):
    """Counts M[a,b] = #(prev=a, cur=b), plus chi-square independence test."""
    M = np.zeros((k, k), dtype=float)
    for a, b in zip(prev.astype(int), cur.astype(int)):
        M[a, b] += 1
    nz = M[M.sum(axis=1) > 0][:, M.sum(axis=0) > 0]
    chi2 = p = float("nan")
    if nz.shape[0] > 1 and nz.shape[1] > 1:
        chi2, p, _, _ = stats.chi2_contingency(nz)
    return M, float(chi2), float(p)


def autocorr(x: np.ndarray, lags) -> dict:
    x = np.asarray(x, float)
    out = {}
    for L in lags:
        if len(x) > L + 2:
            out[L] = float(np.corrcoef(x[:-L], x[L:])[0, 1])
    return out


def low_spacing_days(records: pd.DataFrame) -> np.ndarray:
    t = pd.to_datetime(records["low_time"], utc=True).sort_values().to_numpy()
    if len(t) < 2:
        return np.array([])
    return np.diff(t).astype("timedelta64[h]").astype(float) / 24.0
