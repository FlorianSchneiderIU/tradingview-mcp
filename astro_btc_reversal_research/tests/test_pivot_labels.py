"""Pivot labelling on a deterministic synthetic series."""

import numpy as np
import pandas as pd

from astro_reversal import pivots


def _zigzag_frame() -> pd.DataFrame:
    # Up 100->120, down 120->100, up 100->120 : clear pivots with ATR=1.
    path = list(range(100, 121)) + list(range(119, 99, -1)) + list(range(101, 121))
    close = np.array(path, dtype=float)
    n = len(close)
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "close_time": pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC"),
        "open": close,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "atr": np.ones(n),
    })


def test_atr_pivots_find_peak_and_trough():
    frame = _zigzag_frame()
    piv = pivots.atr_directional_pivots(frame, threshold_atr=3.0)
    kinds = [p.kind for p in piv]
    assert "high" in kinds and "low" in kinds
    # Highs should sit near the 120 peaks, lows near the 100 trough.
    highs = [p.index for p in piv if p.kind == "high"]
    lows = [p.index for p in piv if p.kind == "low"]
    assert any(frame["high"].iloc[i] >= 120.0 for i in highs)
    assert any(frame["low"].iloc[i] <= 100.5 for i in lows)


def test_fractal_pivots_basic():
    frame = _zigzag_frame()
    piv = pivots.fractal_pivots(frame, left=3, right=3)
    assert len(piv) > 0
    # The global max high index must be flagged as a fractal high.
    peak_idx = int(frame["high"].idxmax())
    assert any(p.index == peak_idx and p.kind == "high" for p in piv)


def test_pivot_window_mask_shape_and_coverage():
    frame = _zigzag_frame()
    piv = pivots.atr_directional_pivots(frame, threshold_atr=3.0)
    mask = pivots.pivot_within_window_mask(piv, len(frame), half_window=2)
    assert mask.shape[0] == len(frame)
    assert mask.dtype == bool
    assert mask.sum() >= len(piv)  # each pivot lights up its own +/- window
