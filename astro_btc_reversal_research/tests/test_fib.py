"""Fibonacci time + price level generation."""

import pandas as pd

from astro_reversal import fib_price, fib_time, reuse


def _piv(index, price, kind="low"):
    return reuse.PivotEvent(index=index, time=pd.Timestamp("2024-01-01", tz="UTC"),
                            kind=kind, price=float(price), threshold_atr=float("nan"))


def test_swing_ratio_levels_repeat():
    pivots = [_piv(0, 100, "high"), _piv(10, 90, "low")]   # swing duration 10
    levels = fib_time.swing_ratio_levels(pivots, n_bars=100, ratios=(1.0,), from_start=False)
    assert 20 in levels                                    # t1 + 1.0*D = 10 + 10
    levels2 = fib_time.swing_ratio_levels(pivots, 100, ratios=(0.618, 1.618))
    assert 16 in levels2 and 26 in levels2                 # 10+6.18~16 ; 10+16.18~26


def test_fib_zone_levels():
    pivots = [_piv(5, 100)]
    z = fib_time.fib_zone_levels(pivots, n_bars=100, zones=(1, 2, 3, 5, 8))
    assert set(z.tolist()) == {6, 7, 8, 10, 13}            # 5 + {1,2,3,5,8}


def test_fib_price_levels_and_proximity():
    leg = (90.0, 100.0)                                    # range 10
    # 0.618 retracement from the high: 100 - 6.18 = 93.82
    near, dist = fib_price.price_at_fib(93.82, leg, fib_price.FIB_RETRACE, fib_price.FIB_EXT, tol_frac=0.02)
    assert near and dist < 0.02
    # A price clearly off any fib level is not near.
    off, _ = fib_price.price_at_fib(91.0, leg, fib_price.FIB_RETRACE, fib_price.FIB_EXT, tol_frac=0.03)
    assert not off


def test_active_leg_needs_two_prior_pivots():
    pivots = [_piv(0, 100, "high"), _piv(10, 90, "low"), _piv(20, 110, "high")]
    assert fib_price.active_leg(pivots, bar_idx=5) is None        # only one prior pivot
    leg = fib_price.active_leg(pivots, bar_idx=25)                # pivots at 10 and 20
    assert leg == (90.0, 110.0)
