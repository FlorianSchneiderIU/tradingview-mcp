"""Calendar event-study search (the *precision* framing).

Reframes the question away from "classify every candle" toward:

    Is there an astro calendar whose firings reliably contain a pivot within a
    tolerance window - at a hit rate well above random windows of the same width
    and count? Low recall is acceptable; high precision (hit rate + lift) is the goal.

A "calendar" is a sparse set of firing bars (one per aspect passage, or per
confluence run). For each calendar we measure:

    hit_rate   = fraction of firings with a pivot within +/- window
    baseline   = fraction of ALL bars within +/- window of a pivot (= a random
                 window's expected hit rate)
    lift       = hit_rate / baseline
    coverage   = fraction of time the calendar's windows occupy (sparsity)

plus a binomial p (vs baseline), an empirical p from random calendars matched on
count + spacing, a shifted-calendar control, and a holdout (2025+) breakdown.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import baselines, pivots, reuse, stats


def near_pivot_mask(frame: pd.DataFrame, pivot_threshold_atr: float, window_bars: int):
    """Boolean array: bar is within +/- window of an ATR directional-change pivot."""
    piv = pivots.atr_directional_pivots(frame, pivot_threshold_atr)
    mask = pivots.pivot_within_window_mask(piv, len(frame), window_bars, kind=None)
    return mask, piv


def directional_pivot_mask(frame: pd.DataFrame, pivot_threshold_atr: float, window_bars: int,
                           kind: str):
    """Boolean array: bar is within +/- window of a pivot of ``kind`` ('low' or 'high')."""
    piv = pivots.atr_directional_pivots(frame, pivot_threshold_atr)
    mask = pivots.pivot_within_window_mask(piv, len(frame), window_bars, kind=kind)
    return mask, piv


def conditional_event_study(
    event_bars: np.ndarray,
    cond_mask: np.ndarray,
    near_mask: np.ndarray,
    window_bars: int,
    holdout_mask: np.ndarray,
    n_draws: int = 500,
    seed: int = 42,
    min_event_count: int = 5,
) -> dict | None:
    """Event study *within a price context* (e.g. dump days).

    The calendar is the aspect firings that also satisfy ``cond_mask`` (e.g. a
    dump into the event). The baseline is random bars drawn from the SAME context,
    so lift isolates the astro contribution beyond the price context itself.
    """
    n = len(near_mask)
    eb = np.unique(event_bars[(event_bars >= 0) & (event_bars < n)])
    eb_cond = eb[cond_mask[eb]]
    ne = int(eb_cond.size)
    cond_idx = np.flatnonzero(cond_mask)
    if ne < min_event_count or cond_idx.size == 0:
        return None

    near_cond = near_mask[cond_idx].astype(float)
    baseline = float(near_cond.mean())
    hit = float(near_mask[eb_cond].mean())
    k = int(near_mask[eb_cond].sum())
    binom_p = stats.binomial_test(k, ne, baseline)

    rng = np.random.default_rng(seed)
    null = rng.choice(near_cond, size=(n_draws, ne), replace=True).mean(axis=1)
    rand_p = stats.empirical_pvalue(hit, null)

    hold_cond = cond_idx[holdout_mask[cond_idx]]
    base_h = float(near_mask[hold_cond].mean()) if hold_cond.size else float("nan")
    eb_h = eb_cond[holdout_mask[eb_cond]]
    hit_h = float(near_mask[eb_h].mean()) if eb_h.size else float("nan")

    return {
        "n_context_bars": int(cond_idx.size),
        "n_events": ne,
        "hit_rate": hit,
        "baseline_hit": baseline,
        "lift": stats.lift(hit, baseline),
        "binomial_p": binom_p,
        "random_p": rand_p,
        "null_mean": float(np.mean(null)) if null.size else float("nan"),
        "holdout_n_events": int(eb_h.size),
        "holdout_hit_rate": hit_h,
        "holdout_baseline": base_h,
        "holdout_lift": stats.lift(hit_h, base_h),
    }


def aspect_event_bars(sep: np.ndarray, aspect: float, orb_deg: float) -> np.ndarray:
    """One firing per aspect passage: the min-orb bar of each contiguous in-orb run."""
    orb = reuse.angular_distance_to_aspect(sep, float(aspect))
    in_orb = orb <= orb_deg
    n = len(sep)
    events: list[int] = []
    i = 0
    while i < n:
        if in_orb[i]:
            j = i
            while j + 1 < n and in_orb[j + 1]:
                j += 1
            seg = orb[i:j + 1]
            events.append(i + int(np.argmin(seg)))
            i = j + 1
        else:
            i += 1
    return np.array(events, dtype=int)


def confluence_event_bars(
    raw: pd.DataFrame, bodies: list[str], aspects: list[float], orb_deg: float, min_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Firings where >= ``min_count`` pair-aspects are simultaneously within orb.

    Returns (event_bars, count_per_bar). One firing per contiguous high-confluence run
    (the peak-count bar of the run).
    """
    available = [b for b in bodies if f"{b}_lon" in raw.columns]
    n = len(raw)
    count = np.zeros(n, dtype=int)
    for left, right in combinations(available, 2):
        sep = (raw[f"{left}_lon"].to_numpy(float) - raw[f"{right}_lon"].to_numpy(float)) % 360.0
        min_orb = np.min(np.vstack([
            reuse.angular_distance_to_aspect(sep, a) for a in aspects]), axis=0)
        count += (min_orb <= orb_deg).astype(int)
    active = count >= min_count
    events: list[int] = []
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            seg = count[i:j + 1]
            events.append(i + int(np.argmax(seg)))
            i = j + 1
        else:
            i += 1
    return np.array(events, dtype=int), count


def event_study(
    event_bars: np.ndarray,
    near_mask: np.ndarray,
    window_bars: int,
    holdout_mask: np.ndarray,
    n_draws: int = 500,
    seed: int = 42,
    min_event_count: int = 5,
) -> dict | None:
    """Evaluate one calendar. Returns None if too few firings."""
    n = len(near_mask)
    eb = np.unique(event_bars[(event_bars >= 0) & (event_bars < n)])
    ne = int(eb.size)
    if ne < min_event_count:
        return None

    hit = float(near_mask[eb].mean())
    k = int(near_mask[eb].sum())
    baseline = float(near_mask.mean())
    lift = stats.lift(hit, baseline)
    binom_p = stats.binomial_test(k, ne, baseline)

    # Coverage = union of the calendar's +/- window firings.
    cov = np.zeros(n, dtype=bool)
    for b in eb:
        cov[max(0, b - window_bars): min(n - 1, b + window_bars) + 1] = True
    coverage = float(cov.mean())

    # Random-calendar empirical p: uniform windows of the same count (fast, standard).
    cals = baselines.random_calendars(ne, n, n_draws, min_spacing=1, seed=seed)
    null = [float(near_mask[c].mean()) for c in cals if c.size]
    rand_p = stats.empirical_pvalue(hit, null)

    # Holdout breakdown.
    eb_h = eb[holdout_mask[eb]]
    base_h = float(near_mask[holdout_mask].mean()) if holdout_mask.any() else float("nan")
    hit_h = float(near_mask[eb_h].mean()) if eb_h.size else float("nan")

    return {
        "n_events": ne,
        "hit_rate": hit,
        "baseline_hit": baseline,
        "lift": lift,
        "coverage": coverage,
        "binomial_p": binom_p,
        "random_p": rand_p,
        "null_mean": float(np.mean(null)) if null else float("nan"),
        "holdout_n_events": int(eb_h.size),
        "holdout_hit_rate": hit_h,
        "holdout_baseline": base_h,
        "holdout_lift": stats.lift(hit_h, base_h),
    }


def shifted_hit_rates(
    event_bars: np.ndarray, near_mask: np.ndarray, offsets_bars: list[int]
) -> list[dict]:
    """Hit rate of the calendar shifted by each offset (should beat these if real)."""
    n = len(near_mask)
    out = []
    for off in offsets_bars:
        sh = event_bars + off
        sh = sh[(sh >= 0) & (sh < n)]
        hit = float(near_mask[sh].mean()) if sh.size else float("nan")
        out.append({"offset_bars": int(off), "n": int(sh.size), "hit_rate": hit})
    return out
