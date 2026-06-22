"""Fibonacci in TIME (not price): project Fibonacci time levels from prior swings and
from a Fibonacci-number bar grid, so we can test whether pivots (and spring reversals)
concentrate at Fibonacci *time* points.

The decisive control is the non-Fibonacci placebo: if pivots cluster at Fib ratios AND
equally at ordinary (non-Fib) ratios of the same prior swing, the effect is just swing-
duration persistence, not Fibonacci. A real Fib-time effect needs Fib > non-Fib > random.

All levels are projected from CONFIRMED prior pivots and lie strictly in the future of
their anchor, so the level *times* are known in advance (no lookahead); the pivot landing
on a level is the forward event being tested.
"""

from __future__ import annotations

import numpy as np

# Canonical Fibonacci time ratios (of a prior swing's duration) and the placebo set.
FIB_RATIOS = (0.382, 0.618, 1.0, 1.618, 2.618)
NONFIB_RATIOS = (0.5, 0.75, 0.9, 1.1, 1.3, 1.5, 2.0)
# Fibonacci-number time zones (bar offsets from an anchor) and a non-Fib placebo grid.
FIB_ZONES = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377)
NONFIB_ZONES = (4, 6, 7, 9, 10, 11, 12, 15, 18, 25, 40, 70, 110)


def _pivot_indices(pivots) -> list[int]:
    return [int(p.index) for p in pivots]


def swing_ratio_levels(pivots, n_bars: int, ratios=FIB_RATIOS, from_start: bool = True) -> np.ndarray:
    """Project ratio x (prior swing duration) forward from each swing.

    For consecutive opposite pivots A(t0) -> B(t1), duration D = t1 - t0; levels at
    t1 + r*D (and, with from_start, t0 + r*D for r > 1). Returns unique bar indices that
    fall strictly after the swing end and inside the series.
    """
    idx = _pivot_indices(pivots)
    levels: list[int] = []
    for k in range(1, len(idx)):
        t0, t1 = idx[k - 1], idx[k]
        d = t1 - t0
        if d <= 0:
            continue
        for r in ratios:
            lvl = int(round(t1 + r * d))
            if t1 < lvl < n_bars:
                levels.append(lvl)
            if from_start and r > 1.0:
                lvl2 = int(round(t0 + r * d))
                if t1 < lvl2 < n_bars:
                    levels.append(lvl2)
    return np.unique(np.array(levels, dtype=int))


def fib_zone_levels(pivots, n_bars: int, zones=FIB_ZONES) -> np.ndarray:
    """Vertical time lines at Fibonacci-number bar offsets from each pivot anchor."""
    idx = _pivot_indices(pivots)
    levels: list[int] = []
    for a in idx:
        for f in zones:
            lvl = a + int(f)
            if a < lvl < n_bars:
                levels.append(lvl)
    return np.unique(np.array(levels, dtype=int))


def levels_to_event_bars(levels: np.ndarray, n_bars: int) -> np.ndarray:
    """Clamp/clean a set of level bar-indices to a valid unique event-bar array."""
    if levels is None or len(levels) == 0:
        return np.array([], dtype=int)
    lv = np.asarray(levels, dtype=int)
    return np.unique(lv[(lv >= 0) & (lv < n_bars)])
