"""Random and shifted event-calendar baselines (proposal section 15).

A real aspect calendar is only interesting if it beats:
  * random calendars   (same event count + similar spacing, placed at random), and
  * shifted calendars   (real events nudged by a fixed offset).

Both return calendars as arrays of candle bar-indices so callers can score them
with the same machinery used for the real calendar.
"""

from __future__ import annotations

import numpy as np


def random_calendars(
    event_count: int,
    n_bars: int,
    n_draws: int,
    min_spacing: int = 1,
    seed: int = 42,
    eligible: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Draw ``n_draws`` random calendars, each with ~``event_count`` bar-indices.

    Sampling is uniform without replacement over ``eligible`` bars (default: all),
    rejecting picks closer than ``min_spacing`` to keep spacing comparable to the
    real (roughly periodic) aspect calendar.
    """
    rng = np.random.default_rng(seed)
    pool = np.arange(n_bars) if eligible is None else np.asarray(eligible, dtype=int)
    pool = pool[(pool >= 0) & (pool < n_bars)]
    calendars: list[np.ndarray] = []
    if event_count <= 0 or pool.size == 0:
        return [np.array([], dtype=int) for _ in range(n_draws)]

    # Fast path: no meaningful spacing constraint -> plain sample without replacement.
    if min_spacing <= 1 or pool.size <= event_count:
        size = min(event_count, pool.size)
        for _ in range(n_draws):
            calendars.append(np.sort(rng.choice(pool, size=size, replace=False)))
        return calendars

    # Spacing-constrained: greedily accept from a single shuffled draw per calendar.
    for _ in range(n_draws):
        candidates = rng.permutation(pool)
        chosen = np.empty(event_count, dtype=int)
        m = 0
        for cand in candidates:
            if m == 0 or np.min(np.abs(chosen[:m] - cand)) >= min_spacing:
                chosen[m] = cand
                m += 1
                if m == event_count:
                    break
        calendars.append(np.sort(chosen[:m]))
    return calendars


def shifted_calendars(
    event_bars: np.ndarray | list,
    offsets_bars: list[int],
    n_bars: int,
) -> dict[int, np.ndarray]:
    """Shift the real event bar-indices by each offset (in candles), clipped to range."""
    base = np.asarray(event_bars, dtype=int)
    out: dict[int, np.ndarray] = {}
    for off in offsets_bars:
        shifted = base + int(off)
        shifted = shifted[(shifted >= 0) & (shifted < n_bars)]
        out[int(off)] = np.unique(shifted)
    return out


def bars_per_day(frame) -> float:
    """Average candles per day for converting day-offsets to bar-offsets."""
    import pandas as pd

    opens = pd.to_datetime(frame["open_time"], utc=True)
    span_days = (opens.iloc[-1] - opens.iloc[0]).total_seconds() / 86_400.0
    return float((len(frame) - 1) / span_days) if span_days > 0 else 1.0
