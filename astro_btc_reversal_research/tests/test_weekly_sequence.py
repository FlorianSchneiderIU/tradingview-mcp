"""Weekly-sequence primitives: feature build (no leakage), transition, autocorr."""

import numpy as np
import pandas as pd

from astro_reversal import weekly_sequence as wseq


def _records(n=40, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    low_frac = rng.uniform(0, 1, n)
    return pd.DataFrame({
        "week": [f"w{i}" for i in range(n)],
        "week_start": [start + pd.Timedelta(weeks=i) for i in range(n)],
        "weekly_range_pct": rng.uniform(0.03, 0.2, n),
        "open": 100 + rng.normal(0, 5, n), "close": 100 + rng.normal(0, 5, n),
        "low_price": 95 + rng.normal(0, 2, n), "high_price": 105 + rng.normal(0, 2, n),
        "low_time": [start + pd.Timedelta(weeks=i, days=float(low_frac[i] * 7)) for i in range(n)],
        "low_frac": low_frac, "low_dow": (low_frac * 7).astype(int),
        "high_frac": rng.uniform(0, 1, n), "low_first": rng.integers(0, 2, n).astype(bool),
        "retest_low": rng.integers(0, 2, n).astype(bool),
    })


def test_build_sequence_excludes_target_from_features():
    seq = wseq.build_sequence(_records(), n_lags=3)
    feat_cols = [c for c in seq.columns if c not in
                 ("week_start", "low_frac", "low_dow", "low_early", "low_bucket",
                  "high_frac", "high_early", "low_time")]
    # No feature is the current-week target; all lagged features exist.
    assert "low_early" not in feat_cols
    assert "low_frac_l1" in feat_cols and "open_vs_prevlow" in feat_cols
    assert len(seq) == 40 - 3  # n_lags rows dropped


def test_transition_detects_dependence():
    # Strongly persistent series (long runs) -> strong dependence (small p).
    s = np.array(([0] * 10 + [1] * 10) * 8)
    _, chi2, p = wseq.transition_matrix(s[:-1], s[1:], 2)
    assert p < 0.01
    # Independent coin flips -> no dependence (large p).
    rng = np.random.default_rng(1)
    r = rng.integers(0, 2, 400)
    _, _, p2 = wseq.transition_matrix(r[:-1], r[1:], 2)
    assert p2 > 0.05


def test_autocorr_known_signal():
    x = np.sin(np.arange(200) * 0.3)
    ac = wseq.autocorr(x, [1, 2])
    assert ac[1] > 0.8  # smooth signal is highly autocorrelated at lag 1
