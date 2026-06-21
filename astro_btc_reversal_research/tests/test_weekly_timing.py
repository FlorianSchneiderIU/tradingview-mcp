"""Weekly-timing primitives: extreme location, retest detection, time-of-week."""

import numpy as np
import pandas as pd

from astro_reversal import weekly_timing as wt


def _week_frame(lows, highs):
    """Build a 2-week 15m frame; week 1 gets the given low/high paths, week 2 flat."""
    n = len(lows)
    close = (np.array(lows) + np.array(highs)) / 2
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),  # Mon
        "close_time": pd.date_range("2024-01-01 00:15", periods=n, freq="15min", tz="UTC"),
        "open": close, "high": np.array(highs, float), "low": np.array(lows, float), "close": close,
    })


def test_extreme_location_and_low_first():
    # 700 bars (> min_week_bars within the first week). Low at bar 10, high at bar 300.
    n = 700
    lows = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows[10] = 90.0
    highs[300] = 120.0
    recs = wt.weekly_records(_week_frame(lows, highs), min_week_bars=400)
    assert len(recs) >= 1
    w0 = recs.iloc[0]
    assert w0["low_first"]            # low (bar 10) before high (bar 300)
    assert w0["low_dow"] == 0          # Monday


def test_retest_low_detection():
    n = 700
    lows = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows[10] = 90.0          # weekly low
    highs[10] = 90.5
    highs[100:150] = 110.0   # price leaves the zone (range 90..120)
    highs[300] = 120.0
    lows[400] = 91.0         # returns near the low -> retest
    recs = wt.weekly_records(_week_frame(lows, highs), retest_tol_frac=0.1,
                             move_away_frac=0.25, min_week_bars=400)
    assert bool(recs.iloc[0]["retest_low"])


def test_time_of_week_monotone_and_dow():
    f = _week_frame(np.full(700, 100.0), np.full(700, 101.0))
    dow, frac = wt.time_of_week(f)
    assert dow[0] == 0                 # starts Monday
    # Within one week (672 bars) frac increases; it resets at the next week boundary.
    assert frac[0] < frac[100] < frac[400]
    assert 0.0 <= frac[0] < 0.01
