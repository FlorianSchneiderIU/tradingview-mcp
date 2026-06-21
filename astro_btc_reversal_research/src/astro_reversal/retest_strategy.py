"""Retest-entry strategy: enter on the mid-week retest of the established weekly low.

Real-time, no hindsight. Within each week (Mon 00:00 UTC) we track the running
weekly low; once price rallies away from it by ``move_away_pct`` the low 'leg' is
established and its level frozen. A later 5m Wyckoff spring that retests that frozen
level (its low within ``retest_tol_pct``) is the entry - tiny stop below the spring,
high-RR target. A clean break below the level resets to track the next leg.

This is the natural follow-up to the weekly-timing finding: weekly lows cluster
early-week and ~30% get retested mid-week, so the retest is a higher-conviction,
tight-stop entry into the same window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ltf_structure as lts, weekly_timing as wt


def week_change_mask(frame: pd.DataFrame) -> np.ndarray:
    wk = pd.to_datetime(frame["open_time"], utc=True).dt.to_period("W").astype(str).to_numpy()
    chg = np.ones(len(wk), dtype=bool)
    chg[1:] = wk[1:] != wk[:-1]
    return chg


def retest_entries(
    frame: pd.DataFrame,
    spring_mask: np.ndarray,
    move_away_pct: float,
    retest_tol_pct: float,
    early_week_only: bool = True,
    est_max_frac: float = 0.40,
) -> np.ndarray:
    """Boolean entry mask: spring retests the established weekly-low leg."""
    low = frame["low"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    chg = week_change_mask(frame)
    _, frac = wt.time_of_week(frame)
    n = len(frame)
    entries = np.zeros(n, dtype=bool)

    wl = np.inf          # running weekly low
    rh = -np.inf         # running high since the weekly low
    wl_frac = 1.0        # fraction-into-week when the running low was set
    established = False
    est_low = np.nan

    for t in range(n):
        if chg[t]:
            wl, rh, wl_frac, established, est_low = low[t], high[t], frac[t], False, np.nan
            continue
        rh = max(rh, high[t])
        if not established:
            if low[t] < wl:
                wl, rh, wl_frac = low[t], high[t], frac[t]
            elif (rh - wl) >= move_away_pct * wl:
                established, est_low = True, wl
        else:
            tol = retest_tol_pct * est_low
            if spring_mask[t] and (est_low - tol) <= low[t] <= (est_low + tol):
                if (not early_week_only) or (wl_frac < est_max_frac):
                    entries[t] = True
            if low[t] < est_low - tol:           # clean breakdown -> track a new leg
                wl, rh, wl_frac, established, est_low = low[t], high[t], frac[t], False, np.nan
    return entries


def spring_mask(frame: pd.DataFrame, lookback: int, wick_frac: float, close_pos: float) -> np.ndarray:
    return lts.spring_long_signals(frame, lookback, wick_frac, close_pos)
