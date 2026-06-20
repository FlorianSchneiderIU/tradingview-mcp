"""Event-study / calendar-search correctness."""

import numpy as np

from astro_reversal import calendar_search


def _near_mask(n, pivot_bars, window):
    mask = np.zeros(n, dtype=bool)
    for p in pivot_bars:
        mask[max(0, p - window): min(n - 1, p + window) + 1] = True
    return mask


def test_perfect_calendar_hits_every_firing():
    n = 200
    pivots = list(range(10, 200, 10))
    window = 1
    near = _near_mask(n, pivots, window)
    holdout = np.zeros(n, dtype=bool)
    # Calendar fires exactly on pivots -> hit rate 1.0 and lift > 1.
    res = calendar_search.event_study(np.array(pivots), near, window, holdout, n_draws=200)
    assert res["hit_rate"] == 1.0
    assert res["lift"] > 1.0
    assert res["binomial_p"] < 0.01


def test_offset_calendar_misses():
    n = 200
    pivots = list(range(10, 200, 10))
    window = 1
    near = _near_mask(n, pivots, window)
    holdout = np.zeros(n, dtype=bool)
    # Fire 4 bars away from every pivot (outside the +/-1 window) -> hit rate 0.
    res = calendar_search.event_study(np.array([p + 4 for p in pivots]), near, window, holdout, n_draws=200)
    assert res["hit_rate"] == 0.0


def test_aspect_event_one_per_passage():
    # Separation dips to 0 twice; expect exactly two firings at the minima.
    n = 100
    sep = np.full(n, 50.0)
    sep[20:25] = [3, 1, 0.2, 1, 3]     # passage 1, min at 22
    sep[60:65] = [3, 1, 0.1, 1, 3]     # passage 2, min at 62
    events = calendar_search.aspect_event_bars(sep, aspect=0.0, orb_deg=2.0)
    assert list(events) == [22, 62]


def test_conditional_isolates_astro_vs_context():
    n = 300
    pivots = list(range(20, 300, 20))
    window = 1
    near = _near_mask(n, pivots, window)
    holdout = np.zeros(n, dtype=bool)
    cond = np.zeros(n, dtype=bool)
    cond[5::5] = True  # "dump" days every 5 bars (some near pivots, most not)
    # Calendar fires only on dump days that ARE pivots -> hit 1.0, baseline < 1.
    firings = np.array(pivots)
    res = calendar_search.conditional_event_study(firings, cond, near, window, holdout, n_draws=200)
    assert res is not None
    assert res["hit_rate"] == 1.0
    assert res["baseline_hit"] < 1.0
    assert res["lift"] > 1.0
    assert res["n_context_bars"] == int(cond.sum())


def test_too_few_firings_returns_none():
    n = 100
    near = _near_mask(n, [50], 2)
    holdout = np.zeros(n, dtype=bool)
    assert calendar_search.event_study(np.array([10, 20]), near, 2, holdout, min_event_count=5) is None
