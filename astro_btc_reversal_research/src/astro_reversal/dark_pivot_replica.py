"""Faithful replication of the public 'Dark Pivot' claim.

Claim (paraphrased from the source tweets):
  * Dark Pivot dates == Moon-Pluto hard aspects (~weekly cadence; next = 2026-06-24).
  * "Each time we dump on the activation day, at least 1 day should give a bullish
    expansion" -> the day is counted as having marked a local bottom.
  * Reported ~22 signals, 17 bottoms = 77.27% hit rate.
  * A '50% window' (midpoint between consecutive Dark Pivots) marks the opposite extreme.

The success rule is deliberately loose, so the headline hit rate means nothing
without a baseline: after almost any down day BTC makes a higher high within a few
days. This module reproduces their hit rate AND scores the identical rule on
ordinary dump days and random days, so lift (not the raw %) is the verdict.

All inputs are past/current except the forward 'bullish expansion' check, which is
a label by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def declined_into(frame: pd.DataFrame, lookback: int) -> np.ndarray:
    """'Dumped on the activation day': close below the close `lookback` bars earlier."""
    close = frame["close"].to_numpy(float)
    out = np.zeros(len(frame), dtype=bool)
    out[lookback:] = close[lookback:] < close[:-lookback]
    return out


def bullish_expansion_within(frame: pd.DataFrame, horizon: int, x_atr: float = 0.0) -> np.ndarray:
    """For each bar d: within d+1..d+horizon, does price expand up past high[d] (by x_atr*ATR)?

    x_atr = 0 -> any higher high (the loosest reading of 'a bullish expansion').
    x_atr > 0 -> requires a real expansion of that size beyond the activation high.
    Forward-looking -> LABEL.
    """
    high = frame["high"].to_numpy(float)
    atr = frame["atr"].to_numpy(float)
    n = len(frame)
    out = np.zeros(n, dtype=bool)
    for d in range(n):
        end = min(n, d + horizon + 1)
        if end <= d + 1:
            continue
        thresh = high[d] + x_atr * (atr[d] if np.isfinite(atr[d]) else 0.0)
        out[d] = bool(np.nanmax(high[d + 1:end]) >= thresh)
    return out


def local_extreme(frame: pd.DataFrame, window: int, kind: str) -> np.ndarray:
    """Bar is the lowest low (kind='low') / highest high (kind='high') in +/- window.

    Forward-looking -> LABEL. This is the literal 'marked a local bottom/top' reading.
    """
    lows = frame["low"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    n = len(frame)
    out = np.zeros(n, dtype=bool)
    for d in range(n):
        lo = max(0, d - window)
        hi = min(n - 1, d + window)
        if kind == "low":
            out[d] = lows[d] <= np.nanmin(lows[lo:hi + 1])
        else:
            out[d] = highs[d] >= np.nanmax(highs[lo:hi + 1])
    return out


def midpoints(event_bars: np.ndarray) -> np.ndarray:
    """The '50% window' bars: midpoint between consecutive Dark Pivot firings."""
    eb = np.sort(np.unique(event_bars))
    if eb.size < 2:
        return np.array([], dtype=int)
    return ((eb[:-1] + eb[1:]) // 2).astype(int)
