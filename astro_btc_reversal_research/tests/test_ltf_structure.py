"""LTF structure primitives: Wyckoff spring detection + HTF pivots."""

import numpy as np
import pandas as pd

from astro_reversal import ltf_structure as lts


def _frame(rows):
    # rows: list of (open, high, low, close)
    arr = np.array(rows, float)
    n = len(arr)
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "close_time": pd.date_range("2024-01-01 00:05", periods=n, freq="5min", tz="UTC"),
        "open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2], "close": arr[:, 3],
        "atr": np.full(n, 1.0),
    })


def test_spring_detects_sweep_with_rejection():
    # 12 flat bars with low=99, then a bar that wicks to 96 and closes back at 100.7.
    rows = [(100, 101, 99, 100)] * 12 + [(100, 100.8, 96, 100.7)]
    sig = lts.spring_long_signals(_frame(rows), lookback=12, wick_frac=0.5, close_pos=0.5)
    assert sig[-1]
    assert not sig[:-1].any()


def test_spring_rejects_weak_reclaim_no_wick():
    # Sweeps below but closes near the low (no rejection) -> not a spring.
    rows = [(100, 101, 99, 100)] * 12 + [(100, 100.2, 96, 98.6)]
    sig = lts.spring_long_signals(_frame(rows), lookback=12, wick_frac=0.5, close_pos=0.5)
    assert not sig.any()


def test_deeper_lookback_is_subset():
    # A spring that sweeps the 12-bar low need not sweep a deeper low; the deeper
    # filter yields fewer or equal signals.
    rows = [(100, 101, 99, 100)] * 30 + [(100, 100.8, 98.5, 100.7)]
    shallow = lts.spring_long_signals(_frame(rows), lookback=12).sum()
    deep = lts.spring_long_signals(_frame(rows), lookback=24).sum()
    assert deep <= shallow


def test_near_times_mask_window():
    f = _frame([(100, 101, 99, 100)] * 50)
    t = pd.to_datetime(f["open_time"].iloc[25], utc=True)
    mask = lts.near_times_mask(f, [t], window_bars=3)
    assert mask[22:29].all()
    assert not mask[:22].any() and not mask[29:].any()
