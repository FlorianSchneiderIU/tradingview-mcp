"""Capitulation Spring — pure signal logic (importable + testable, no I/O).

The validated edge (see astro_btc_reversal_research): a deep-liquidity-sweep 5m
Wyckoff spring (sweep of a ~15-day low + reclaim with rejection), early in the week
(Mon-Wed), filtered to unusually negative funding (funding_z <= threshold = shorts
crowded after a flush -> contrarian-bullish capitulation). Long only. Scaled exit
25% @ 4R / 50% @ 12R / 25% @ 30R, stop -> breakeven after the first partial.

This module is deliberately dependency-light (numpy only) so it can be unit-tested
offline and reused by the live sidecar bot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WEEK_MS = 7 * 24 * 3600 * 1000


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> float:
    """Last Wilder ATR value (RMA of true range)."""
    n = len(close)
    if n < length + 1:
        return float("nan")
    prev_close = close[:-1]
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:] - prev_close),
    ])
    atr = tr[:length].mean()
    for x in tr[length:]:
        atr = (atr * (length - 1) + x) / length
    return float(atr)


def week_fraction(ts_ms: int) -> float:
    """Fraction into the Mon 00:00 UTC -> Sun 23:59 week (0=Mon open, ~1=Sun close)."""
    # Unix epoch (1970-01-01) was a Thursday; Monday 00:00 is offset by 4 days.
    monday_offset = 4 * 24 * 3600 * 1000
    return ((ts_ms - monday_offset) % WEEK_MS) / WEEK_MS


@dataclass(frozen=True)
class SpringConfig:
    sweep_lookback: int = 4320       # 15 days of 5m bars
    wick_frac: float = 0.5           # lower-wick rejection fraction
    close_pos: float = 0.5           # close in the top half of the bar's range
    week_frac_max: float = 0.40      # only the first 40% of the week (Mon-Wed)
    funding_z_thr: float = -1.0      # funding_z must be <= this
    atr_len: int = 14
    stop_buffer_atr: float = 0.05    # stop below the spring low, in ATR
    tp_r: tuple = (4.0, 12.0, 30.0)  # scaled take-profit R multiples
    tp_qty_pct: tuple = (25.0, 50.0, 25.0)


def detect_spring(bars: list[dict], funding_z: float | None, cfg: SpringConfig) -> dict | None:
    """Return a long setup dict if the just-closed bar (bars[-1]) is a capitulation
    spring meeting all gates, else None.

    bars: chronological list of {ts(ms), open, high, low, close}; bars[-1] is the
    just-closed 5m candle. funding_z: latest funding z-score for this symbol.
    """
    n = len(bars)
    if n < cfg.sweep_lookback + cfg.atr_len + 2:
        return None
    if funding_z is None or not math.isfinite(funding_z) or funding_z > cfg.funding_z_thr:
        return None

    last = bars[-1]
    if week_fraction(int(last["ts"])) >= cfg.week_frac_max:
        return None

    high = np.fromiter((b["high"] for b in bars), float, n)
    low = np.fromiter((b["low"] for b in bars), float, n)
    close = np.fromiter((b["close"] for b in bars), float, n)
    open_ = np.fromiter((b["open"] for b in bars), float, n)

    prev_low = float(np.min(low[n - 1 - cfg.sweep_lookback: n - 1]))  # prior `lookback` lows
    last_low, last_high, last_close, last_open = low[-1], high[-1], close[-1], open_[-1]
    if not (last_low < prev_low and last_close > prev_low):
        return None  # not a sweep + reclaim of the deep low
    rng = last_high - last_low
    if rng <= 0:
        return None
    lower_wick = (min(last_open, last_close) - last_low) / rng
    closed_high = (last_close - last_low) / rng
    if lower_wick < cfg.wick_frac or closed_high < cfg.close_pos:
        return None  # no clean rejection

    atr = wilder_atr(high, low, close, cfg.atr_len)
    if not math.isfinite(atr) or atr <= 0:
        return None

    entry = float(last_close)
    stop = float(last_low - cfg.stop_buffer_atr * atr)
    risk = entry - stop
    if risk <= 0 or risk / entry < 1e-5:
        return None
    targets = [entry + r * risk for r in cfg.tp_r]
    return {
        "direction": "long",
        "entry": entry,
        "stop": stop,
        "risk": risk,
        "atr": atr,
        "spring_low": float(last_low),
        "prev_deep_low": prev_low,
        "funding_z": float(funding_z),
        "week_fraction": week_fraction(int(last["ts"])),
        "lower_wick_frac": float(lower_wick),
        "close_pos": float(closed_high),
        "targets": targets,
        "tp_qty_pct": list(cfg.tp_qty_pct),
        "tp_r": list(cfg.tp_r),
        "target": float(targets[-1]),  # convenience: final target
    }
