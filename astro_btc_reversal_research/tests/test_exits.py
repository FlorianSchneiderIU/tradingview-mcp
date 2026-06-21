"""Scaled-exit simulator + short upthrust signal."""

import numpy as np
import pandas as pd

from astro_reversal import exits, ltf_structure as lts


def _frame(highs, lows, entry_open=100.0):
    n = len(highs)
    o = np.full(n, entry_open)
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "close_time": pd.date_range("2024-01-01 00:15", periods=n, freq="15min", tz="UTC"),
        "open": o, "high": np.array(highs, float), "low": np.array(lows, float),
        "close": o, "atr": np.full(n, 1.0),
    })


def _signal_frame(path_high, path_low):
    # bar0 = signal (low 90 -> stop ~89.95, risk ~10.05 at entry 100); bars 1.. = path
    highs = [91.0] + path_high
    lows = [90.0] + path_low
    f = _frame(highs, lows)
    f.loc[0, "open"] = 90.5
    return f


def test_scaled_tp1_then_breakeven_nets_about_1r():
    # Hit TP1 (4R) then come back to breakeven on the remaining 75% -> ~+1R.
    f = _signal_frame([141.0, 110.0], [139.0, 99.0])  # bar1 hits TP1; bar2 low<=entry -> BE
    tr = exits.simulate_scaled_trade(f, 0, "long", tps=(4, 12, 30), fracs=(0.25, 0.5, 0.25),
                                     stop_buffer_atr=0.05, cost_bps_round_trip=0.0)
    assert abs(tr["result_r"] - 1.0) < 0.05
    assert tr["tp_filled"] == 1


def test_scaled_all_targets_blended_r():
    f = _signal_frame([141.0, 221.0, 402.0], [150.0, 300.0, 390.0])  # TP1, TP2, TP3
    tr = exits.simulate_scaled_trade(f, 0, "long", tps=(4, 12, 30), fracs=(0.25, 0.5, 0.25),
                                     stop_buffer_atr=0.05, cost_bps_round_trip=0.0)
    assert abs(tr["result_r"] - (0.25 * 4 + 0.5 * 12 + 0.25 * 30)) < 0.1  # 14.5R
    assert tr["tp_filled"] == 3


def test_scaled_full_stop_before_tp1():
    f = _signal_frame([105.0], [85.0])  # stop (89.95) hit before any target
    tr = exits.simulate_scaled_trade(f, 0, "long", tps=(4, 12, 30), fracs=(0.25, 0.5, 0.25),
                                     stop_buffer_atr=0.05, cost_bps_round_trip=0.0)
    assert abs(tr["result_r"] + 1.0) < 0.05
    assert tr["tp_filled"] == 0


def test_upthrust_short_signal():
    rows_high = [101.0] * 12 + [104.0]   # sweeps above prior high
    rows_low = [99.0] * 12 + [99.5]
    close = [100.0] * 12 + [99.6]        # reclaims below prior high (101) with upper wick
    openp = [100.0] * 12 + [100.0]
    f = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=13, freq="15min", tz="UTC"),
        "close_time": pd.date_range("2024-01-01 00:15", periods=13, freq="15min", tz="UTC"),
        "open": openp, "high": rows_high, "low": rows_low, "close": close, "atr": np.full(13, 1.0),
    })
    sig = lts.upthrust_short_signals(f, lookback=12, wick_frac=0.5, close_pos=0.5)
    assert sig[-1] and not sig[:-1].any()
