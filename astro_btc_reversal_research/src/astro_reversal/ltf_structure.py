"""Multi-timeframe structure study around weekly pivots (the 1:20-1:30 thesis).

The trader times weekly swings and confirms on 1m/5m with a Wyckoff spring, taking
1:20-1:30 RR. That only works because the LTF stop is tiny relative to the weekly
target. This module:

  * marks HTF (daily zigzag) weekly pivot lows/highs,
  * detects 5m Wyckoff springs (sweep of a significant low + reclaim with rejection),
  * lets the backtest measure how often a spring runs to 20R/30R with a tight stop,
  * and compares springs NEAR a true weekly low vs elsewhere - i.e. whether the
    asymmetric opportunity concentrates at weekly lows (a learnable structural edge).

Spring detection uses past/current bars only; 'near a weekly low' uses the HTF pivot
(a hindsight label) and is only for the opportunity-concentration study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import reuse


def htf_pivots(daily: pd.DataFrame, threshold_atr: float):
    """Daily zigzag pivots + the list of low pivots with their following high."""
    piv = reuse.zigzag_pivots(daily, threshold_atr)
    lows = []
    highs = []
    for i, p in enumerate(piv):
        nxt_opp = next((q for q in piv[i + 1:] if q.kind != p.kind), None)
        rec = {"time": p.time, "price": p.price,
               "next_opp_time": nxt_opp.time if nxt_opp else None,
               "next_opp_price": nxt_opp.price if nxt_opp else None}
        (lows if p.kind == "low" else highs).append(rec)
    return piv, lows, highs


def spring_long_signals(
    frame: pd.DataFrame,
    lookback: int,
    wick_frac: float = 0.5,
    close_pos: float = 0.5,
) -> np.ndarray:
    """Wyckoff spring (long): bar sweeps below the lowest low of the prior `lookback`
    bars, then reclaims it (close back above) with a clear lower-wick rejection."""
    low = frame["low"]
    high = frame["high"]
    close = frame["close"]
    open_ = frame["open"]
    prev_low = low.shift(1).rolling(lookback, min_periods=lookback).min()
    sweep = (low < prev_low) & (close > prev_low)
    rng = (high - low).replace(0.0, np.nan)
    lower_wick = (np.minimum(open_, close) - low) / rng
    closed_high = (close - low) / rng
    spring = sweep & (lower_wick >= wick_frac) & (closed_high >= close_pos)
    return spring.fillna(False).to_numpy(dtype=bool)


def near_times_mask(frame: pd.DataFrame, times, window_bars: int) -> np.ndarray:
    """Mark bars within +/- window_bars of any timestamp in `times`."""
    n = len(frame)
    mask = np.zeros(n, dtype=bool)
    ot = pd.to_datetime(frame["open_time"], utc=True).to_numpy()
    valid = [t for t in times if t is not None]
    if not valid:
        return mask
    idx = np.searchsorted(ot, pd.to_datetime(pd.Series(valid), utc=True).to_numpy())
    for i in idx:
        i = int(min(max(i, 0), n - 1))
        mask[max(0, i - window_bars): min(n - 1, i + window_bars) + 1] = True
    return mask
