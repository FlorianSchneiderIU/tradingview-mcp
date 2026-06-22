"""Fibonacci in PRICE: retracement (internal) and extension levels of the most recent
completed swing, evaluated at a given bar/price. Pairs with fib_time for the confluence
test ("a reversal at a Fibonacci time AND price").

A non-Fibonacci placebo set mirrors the construction so we can tell a real Fib-price
effect from "price simply reacts at round fractions of the prior swing".
"""

from __future__ import annotations

import math

FIB_RETRACE = (0.382, 0.5, 0.618, 0.786)
FIB_EXT = (1.272, 1.414, 1.618, 2.0, 2.618)
NONFIB_RETRACE = (0.30, 0.45, 0.55, 0.70)
NONFIB_EXT = (1.10, 1.35, 1.50, 1.80, 2.30)


def active_leg(pivots, bar_idx: int):
    """Most recent completed swing leg (two pivots strictly before bar_idx)."""
    prior = [p for p in pivots if int(p.index) < bar_idx]
    if len(prior) < 2:
        return None
    a, b = prior[-2], prior[-1]
    lo, hi = (a.price, b.price) if a.price <= b.price else (b.price, a.price)
    return float(lo), float(hi)


def fib_levels(lo: float, hi: float, retr=FIB_RETRACE, ext=FIB_EXT) -> list[float]:
    rng = hi - lo
    out: list[float] = []
    for r in retr:                       # internal retracements (both directions)
        out.append(hi - r * rng)
        out.append(lo + r * rng)
    for e in ext:                        # extensions beyond the leg
        out.append(hi + (e - 1.0) * rng)
        out.append(lo - (e - 1.0) * rng)
    return out


def price_at_fib(price: float, leg, retr=FIB_RETRACE, ext=FIB_EXT, tol_frac: float = 0.05):
    """Is ``price`` within tol_frac of the swing range of any fib level of ``leg``?

    Returns (is_near: bool, nearest_distance_frac: float).
    """
    if leg is None:
        return False, float("nan")
    lo, hi = leg
    rng = hi - lo
    if rng <= 0:
        return False, float("nan")
    levels = fib_levels(lo, hi, retr, ext)
    d = min(abs(price - lv) for lv in levels) / rng
    return (d <= tol_frac and math.isfinite(d)), float(d)
