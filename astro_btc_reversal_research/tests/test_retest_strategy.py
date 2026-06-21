"""Retest-entry state machine: establish weekly low -> move away -> spring retest."""

import numpy as np
import pandas as pd

from astro_reversal import retest_strategy as rts


def _frame(low, high):
    n = len(low)
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),  # one Monday
        "close_time": pd.date_range("2024-01-01 00:05", periods=n, freq="5min", tz="UTC"),
        "open": np.array(low, float), "high": np.array(high, float),
        "low": np.array(low, float), "close": np.array(high, float),
        "atr": np.full(n, 1.0),
    })


def test_retest_entry_fires_after_establish_and_return():
    n = 40
    low = [100.0] * n
    high = [101.0] * n
    low[3], high[3] = 90.0, 90.5          # weekly low
    high[10] = 96.0                        # rally away (>1.5% of 90) -> established at 90
    low[20] = 90.2                         # return within 0.5% of the low
    spring = np.zeros(n, dtype=bool)
    spring[20] = True                      # spring confirms the retest
    entries = rts.retest_entries(_frame(low, high), spring, move_away_pct=0.015,
                                 retest_tol_pct=0.005, early_week_only=False)
    assert entries[20]
    assert entries.sum() == 1


def test_no_entry_without_move_away():
    n = 30
    low = [90.5] * n                       # price stays within ~1% of the low
    high = [91.0] * n                      # high only ~1.1% above the low (< 1.5% move-away)
    low[3] = 90.0
    low[15] = 90.1
    spring = np.zeros(n, dtype=bool)
    spring[15] = True
    entries = rts.retest_entries(_frame(low, high), spring, move_away_pct=0.015,
                                 retest_tol_pct=0.005, early_week_only=False)
    assert not entries.any()               # leg never established -> no retest entry


def test_breakdown_below_level_blocks_old_level_entry():
    n = 40
    low = [100.0] * n
    high = [101.0] * n
    low[3], high[3] = 90.0, 90.5
    high[8] = 96.0                         # established at 90
    low[15] = 88.0                         # clean breakdown below 90 -> reset
    low[25] = 90.1                         # back near the OLD level, but it was invalidated
    spring = np.zeros(n, dtype=bool)
    spring[25] = True
    entries = rts.retest_entries(_frame(low, high), spring, move_away_pct=0.015,
                                 retest_tol_pct=0.005, early_week_only=False)
    assert not entries[25]                  # no move-away re-established after the breakdown
