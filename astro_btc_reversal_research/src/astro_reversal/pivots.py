"""Pivot labelling.

Two definitions (proposal section 12):
  * ATR directional-change pivots  -> reuse the existing ``zigzag_pivots``.
  * Fractal pivots                 -> simple highest-high / lowest-low in [t-L, t+R].

Both use future candles, so their outputs are LABELS ONLY and must never be fed
back in as features (proposal section 32).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import reuse

PivotEvent = reuse.PivotEvent


def atr_directional_pivots(frame: pd.DataFrame, threshold_atr: float) -> list[PivotEvent]:
    """ATR-based directional-change pivots (reuses the existing zigzag)."""
    return reuse.zigzag_pivots(frame, threshold_atr)


def fractal_pivots(frame: pd.DataFrame, left: int, right: int) -> list[PivotEvent]:
    """Fractal pivots: high[t] is the max high in [t-left, t+right] (and symmetric for lows)."""
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    times = pd.to_datetime(frame["close_time"], utc=True)
    n = len(frame)
    pivots: list[PivotEvent] = []
    for t in range(left, n - right):
        window = slice(t - left, t + right + 1)
        if highs[t] >= highs[window].max():
            pivots.append(PivotEvent(index=t, time=pd.Timestamp(times.iloc[t]).tz_convert("UTC"),
                                     kind="high", price=float(highs[t]), threshold_atr=float("nan")))
        if lows[t] <= lows[window].min():
            pivots.append(PivotEvent(index=t, time=pd.Timestamp(times.iloc[t]).tz_convert("UTC"),
                                     kind="low", price=float(lows[t]), threshold_atr=float("nan")))
    return sorted({(p.index, p.kind): p for p in pivots}.values(), key=lambda item: item.index)


def pivot_indices(pivots: list[PivotEvent], kind: str | None = None) -> list[int]:
    return [int(p.index) for p in pivots if kind is None or p.kind == kind]


def pivot_stats(frame: pd.DataFrame, pivots: list[PivotEvent], horizon: int) -> dict:
    """Summary used to judge whether a pivot definition yields *rare, significant*
    reversal windows rather than ordinary swing churn."""
    from . import labels_ml  # local import to avoid a cycle

    n = len(frame)
    idx = np.array([p.index for p in pivots], dtype=int)
    span_days = (pd.to_datetime(frame["close_time"], utc=True).iloc[-1]
                 - pd.to_datetime(frame["open_time"], utc=True).iloc[0]).total_seconds() / 86_400.0
    bpd = (n - 1) / span_days if span_days > 0 else float("nan")
    gaps = np.diff(idx) if idx.size > 1 else np.array([np.nan])
    lab = labels_ml.forward_labels(frame, pivots, horizon)
    return {
        "n_pivots": len(pivots),
        "median_gap_bars": float(np.median(gaps)),
        "median_gap_days": float(np.median(gaps) / bpd) if np.isfinite(bpd) else float("nan"),
        "pivots_per_day": float(len(pivots) / span_days) if span_days > 0 else float("nan"),
        "base_rate_any": float(lab["y_any"].mean()),
        "base_rate_low": float(lab["y_low"].mean()),
        "base_rate_high": float(lab["y_high"].mean()),
    }


def pivot_within_window_mask(pivots: list[PivotEvent], n_bars: int, half_window: int,
                             kind: str | None = None) -> np.ndarray:
    """Boolean array: True for candle c if a pivot of ``kind`` occurs in [c-half, c+half].

    Used for window-based scoring (proposal section 22). Forward-looking -> label only.
    """
    mask = np.zeros(n_bars, dtype=bool)
    for idx in pivot_indices(pivots, kind):
        lo = max(0, idx - half_window)
        hi = min(n_bars - 1, idx + half_window)
        mask[lo: hi + 1] = True
    return mask
