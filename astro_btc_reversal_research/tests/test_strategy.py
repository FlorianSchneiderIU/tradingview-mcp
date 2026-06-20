"""Strategy primitive checks (gate, signal, backtest plumbing)."""

import numpy as np
import pandas as pd

from astro_reversal import strategy


def _ltf(close, high=None, low=None, open_=None, atr=2.0):
    close = np.asarray(close, float)
    n = len(close)
    high = close + 1.0 if high is None else np.asarray(high, float)
    low = close - 1.0 if low is None else np.asarray(low, float)
    open_ = close if open_ is None else np.asarray(open_, float)
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "close_time": pd.date_range("2024-01-01 01:00", periods=n, freq="h", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close, "atr": np.full(n, atr),
    })


def test_sweep_reclaim_signal():
    # Flat at 100, then a bar wicks below the prior-low (to 96) but closes back at 101.
    close = [100] * 15 + [101]
    low = [99] * 15 + [96]
    high = [101] * 15 + [101.5]
    sig = strategy.ltf_long_signals(_ltf(close, high=high, low=low), lookback=12)
    assert sig[-1]
    assert not sig[:-1].any()


def test_displacement_filter_rejects_weak_reclaim():
    close = [100] * 15 + [99.6]   # closes only barely above the swept low, tiny body
    low = [99] * 15 + [96]
    high = [101] * 15 + [101]
    openp = [100] * 15 + [99.5]
    sig = strategy.ltf_long_signals(_ltf(close, high=high, low=low, open_=openp),
                                    lookback=12, require_displacement=True,
                                    disp_body_atr=0.5, disp_close_frac=0.5)
    assert not sig.any()


def test_daily_dump_gate_active_window():
    close = [100, 100, 94, 100, 100, 100, 100]  # dump at index 2
    daily = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=len(close), freq="D", tz="UTC"),
        "close_time": pd.date_range("2024-01-02", periods=len(close), freq="D", tz="UTC"),
        "open": close, "high": np.array(close) + 1, "low": np.array(close) - 1,
        "close": close, "atr": np.full(len(close), 2.0),
    })
    dump, active = strategy.daily_dump_gate(daily, lookback=1, threshold_atr=1.0, hold_days=3)
    assert dump[2]
    # Active for the dump day and the following two days (hold_days=3 inclusive window).
    assert active[2] and active[3] and active[4]
    assert not active[5]
