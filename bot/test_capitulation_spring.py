"""Offline tests for the Capitulation Spring signal logic (pure, no I/O)."""

import numpy as np

from capitulation_spring import SpringConfig, detect_spring, week_fraction, wilder_atr

FIVE_MIN_MS = 5 * 60 * 1000
# A Monday 00:00 UTC anchor (2024-01-01 was a Monday).
MONDAY_MS = 1704067200000


def _bars(n, base=100.0, monday_anchor=True):
    start = MONDAY_MS if monday_anchor else MONDAY_MS + 5 * 24 * 3600 * 1000  # Mon vs Sat
    bars = []
    for i in range(n):
        bars.append({"ts": start + i * FIVE_MIN_MS, "open": base, "high": base + 0.5,
                     "low": base - 0.5, "close": base, "volume": 1.0})
    return bars


def _make_spring(funding_anchor_monday=True, with_wick=True):
    cfg = SpringConfig(sweep_lookback=300, atr_len=14)  # small lookback for the test
    n = 360
    bars = _bars(n, base=100.0, monday_anchor=funding_anchor_monday)
    # Establish a clear prior-low band at ~99 over the lookback window; then the last
    # bar sweeps to 96 and reclaims to 100.4 with a long lower wick.
    last = bars[-1]
    last["open"] = 99.6
    last["high"] = 100.5
    last["low"] = 96.0 if with_wick else 99.0
    last["close"] = 100.4 if with_wick else 99.05
    return cfg, bars


def test_week_fraction_monday_is_zero():
    assert abs(week_fraction(MONDAY_MS)) < 1e-6
    # Saturday ~ 5/7 into the week.
    sat = MONDAY_MS + 5 * 24 * 3600 * 1000
    assert 0.70 < week_fraction(sat) < 0.74


def test_wilder_atr_constant_range():
    n = 50
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    assert abs(wilder_atr(high, low, close, 14) - 2.0) < 1e-9


def test_detect_spring_accepts_clean_setup():
    cfg, bars = _make_spring()
    out = detect_spring(bars, funding_z=-1.5, cfg=cfg)
    assert out is not None
    assert out["direction"] == "long"
    assert out["entry"] > out["stop"]
    assert len(out["targets"]) == 3
    # Targets are 4R/12R/30R above entry.
    risk = out["entry"] - out["stop"]
    assert abs(out["targets"][0] - (out["entry"] + 4 * risk)) < 1e-6
    assert abs(out["targets"][2] - (out["entry"] + 30 * risk)) < 1e-6


def test_reject_when_funding_not_negative_enough():
    cfg, bars = _make_spring()
    assert detect_spring(bars, funding_z=-0.5, cfg=cfg) is None   # above threshold -1.0
    assert detect_spring(bars, funding_z=None, cfg=cfg) is None


def test_reject_when_late_in_week():
    cfg, bars = _make_spring(funding_anchor_monday=False)  # Saturday -> week_frac ~0.71
    assert detect_spring(bars, funding_z=-1.5, cfg=cfg) is None


def test_reject_when_no_wick_rejection():
    cfg, bars = _make_spring(with_wick=False)  # closes near the swept low, tiny wick
    assert detect_spring(bars, funding_z=-1.5, cfg=cfg) is None


def test_reject_when_no_sweep():
    cfg, bars = _make_spring()
    bars[-1]["low"] = 99.2   # never trades below the ~99 prior low
    assert detect_spring(bars, funding_z=-1.5, cfg=cfg) is None
