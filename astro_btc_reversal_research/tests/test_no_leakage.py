"""Anti-leakage guards (proposal section 32).

Features (dump flags, aspect membership) must depend only on past/current data.
Labels (expansion) are allowed to look forward - we assert they actually do, so
they can never be mistaken for features.
"""

import numpy as np
import pandas as pd

from astro_reversal import event_labels


def _frame(n=60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "close_time": pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC"),
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "atr": np.full(n, 2.0),
    })


def test_dump_flags_ignore_the_future():
    frame = _frame()
    base = event_labels.dump_flags(frame, lookback=2, threshold_atr=1.0)
    t = 30
    perturbed = frame.copy()
    perturbed.loc[t + 1:, "close"] = perturbed.loc[t + 1:, "close"] + 50.0  # change the future
    after = event_labels.dump_flags(perturbed, lookback=2, threshold_atr=1.0)
    # Flags at and before bar t are a function of past/current bars only.
    assert np.array_equal(base[: t + 1], after[: t + 1])


def test_atr_normalized_move_warmup_is_nan():
    frame = _frame()
    move = event_labels.atr_normalized_move(frame, lookback=3)
    assert np.all(np.isnan(move[:3]))
    assert np.isfinite(move[3:]).all()


def test_expansion_label_uses_the_future():
    frame = _frame()
    t = 20
    base = event_labels.evaluate_expansion(frame, t, "bull", horizon=3, target_atr=1.0, buffer_atr=0.1)
    perturbed = frame.copy()
    # Force a big up-move inside the forward window -> outcome must change.
    perturbed.loc[t + 1: t + 3, "high"] = perturbed.loc[t + 1: t + 3, "high"] + 100.0
    after = event_labels.evaluate_expansion(perturbed, t, "bull", horizon=3, target_atr=1.0, buffer_atr=0.1)
    assert base["mfe_atr"] != after["mfe_atr"], "expansion label must depend on forward bars"
