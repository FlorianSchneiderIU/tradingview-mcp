"""Dark Pivot replication primitives."""

import numpy as np
import pandas as pd

from astro_reversal import dark_pivot_replica as dpr


def _frame(close, atr=2.0):
    close = np.asarray(close, float)
    n = len(close)
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "close_time": pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC"),
        "open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
        "atr": np.full(n, atr),
    })


def test_declined_into():
    f = _frame([100, 99, 98, 101, 97])
    d = dpr.declined_into(f, 1)
    assert list(d) == [False, True, True, False, True]


def test_bullish_expansion_any_higher_high():
    # rises after index 2 -> expansion true before the peak, false at the top.
    f = _frame([100, 99, 98, 105, 104])
    exp = dpr.bullish_expansion_within(f, horizon=2, x_atr=0.0)
    assert exp[2]            # 98 -> later high 106 within 2 days
    assert not exp[3]        # 105 is the peak; nothing higher after within horizon... 104 high=105 < 106
    # (index 3 high=106, index4 high=105 -> not higher) -> False


def test_local_extreme_low_and_high():
    f = _frame([100, 95, 100, 105, 100])
    low = dpr.local_extreme(f, window=1, kind="low")
    high = dpr.local_extreme(f, window=1, kind="high")
    assert low[1]            # 95 is the local low
    assert high[3]           # 105 is the local high


def test_midpoints():
    eb = np.array([10, 20, 30])
    assert list(dpr.midpoints(eb)) == [15, 25]
