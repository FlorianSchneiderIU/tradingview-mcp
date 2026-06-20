"""Walk-forward split correctness + embargo, and price-feature leakage guard."""

import numpy as np
import pandas as pd

from astro_reversal import features_ml, walk_forward


def test_dev_holdout_split():
    times = pd.Series(pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC"))
    dev, hold = walk_forward.split_dev_holdout(times, pd.Timestamp("2023-03-01", tz="UTC"))
    assert dev.max() < hold.min()
    assert dev.size + hold.size == 100


def test_folds_are_causal_with_embargo():
    dev = np.arange(1000)
    embargo = 5
    folds = walk_forward.walk_forward_folds(dev, n_folds=4, embargo=embargo)
    assert len(folds) == 4
    for train, test in folds:
        # Train strictly precedes test, with at least `embargo` purged candles between.
        assert train.max() < test.min()
        assert test.min() - train.max() > embargo


def test_folds_expanding():
    dev = np.arange(1000)
    folds = walk_forward.walk_forward_folds(dev, n_folds=4, embargo=3)
    train_sizes = [t.size for t, _ in folds]
    assert train_sizes == sorted(train_sizes)  # expanding training window


def _frame(n=120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "close_time": pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC"),
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": rng.uniform(10, 20, n),
        "atr": np.full(n, 2.0),
    })


def test_price_features_ignore_the_future():
    frame = _frame()
    base = features_ml.price_features(frame)
    t = 60
    perturbed = frame.copy()
    perturbed.loc[t + 1:, ["open", "high", "low", "close", "volume"]] *= 3.0
    after = features_ml.price_features(perturbed)
    # Every trailing feature up to bar t must be unchanged by future perturbation.
    pd.testing.assert_frame_equal(base.iloc[: t + 1], after.iloc[: t + 1])
