"""Milestone 2 - full aspect library discovery.

For each (body pair x aspect angle) builds a per-candle "in aspect window"
membership (angular separation within an orb), then measures pivot-window LIFT:

    lift = P(pivot within +/-W | in aspect window) / P(pivot within +/-W)   [baseline]

Controls per hypothesis: a binomial p-value vs the baseline rate, an empirical
random-subset p-value, and a holdout (2025+) lift. Benjamini-Hochberg FDR is
applied across all (pair x aspect) hypotheses to guard the large search space.

Uses candle-aligned longitudes (per-bar orb membership) rather than exact events
- faster for hundreds of hypotheses and explicitly permitted by the plan.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import pivots, reuse, stats


def _candle_longitudes(frame: pd.DataFrame, timeframe: str, cache_dir: Path) -> pd.DataFrame:
    raw = reuse.compute_skyfield_positions(frame["open_time"], cache_dir, timeframe)
    if raw.empty:
        raise RuntimeError("Skyfield ephemeris unavailable; cannot run discovery.")
    return raw.reset_index(drop=True)


def run_discovery(
    frame: pd.DataFrame,
    timeframe: str,
    bodies: list[str],
    aspects: list[float],
    orb_deg: float,
    window_bars: int,
    window_kind: str,
    pivot_threshold_atr: float,
    holdout_start: pd.Timestamp,
    min_in_window_bars: int,
    random_draws: int,
    fdr_alpha: float,
    seed: int,
    cache_dir: Path | None = None,
    symbol: str = "BTCUSDT",
    start: str = "",
    end: str = "",
) -> dict:
    cache_dir = cache_dir or reuse.DEFAULT_CACHE_DIR
    n = len(frame)
    open_times = pd.to_datetime(frame["open_time"], utc=True)
    raw = _candle_longitudes(frame, timeframe, cache_dir)

    piv = pivots.atr_directional_pivots(frame, pivot_threshold_atr)
    pivot_window = pivots.pivot_within_window_mask(piv, n, window_bars, kind=None).astype(float)
    baseline_rate = float(np.mean(pivot_window))

    holdout_mask = (open_times >= holdout_start).to_numpy()
    baseline_rate_holdout = float(np.mean(pivot_window[holdout_mask])) if holdout_mask.any() else float("nan")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    available = [b for b in bodies if f"{b}_lon" in raw.columns]
    for left, right in combinations(available, 2):
        sep = (raw[f"{left}_lon"].to_numpy(dtype=float) - raw[f"{right}_lon"].to_numpy(dtype=float)) % 360.0
        for aspect in aspects:
            orb = reuse.angular_distance_to_aspect(sep, float(aspect))
            in_aspect = orb <= orb_deg
            n_in = int(np.sum(in_aspect))
            if n_in < min_in_window_bars:
                continue
            k = int(np.sum(pivot_window[in_aspect]))
            rate_in = k / n_in
            lift = stats.lift(rate_in, baseline_rate)
            binom_p = stats.binomial_test(k, n_in, baseline_rate)

            # Random-subset null: rate over a random size-n_in subset of all candles.
            null = rng.choice(pivot_window, size=(random_draws, n_in), replace=True).mean(axis=1)
            random_p = stats.empirical_pvalue(rate_in, null)

            # Holdout lift (reported, never used to rank).
            in_aspect_h = in_aspect & holdout_mask
            n_in_h = int(np.sum(in_aspect_h))
            rate_in_h = float(np.mean(pivot_window[in_aspect_h])) if n_in_h > 0 else float("nan")
            holdout_lift = stats.lift(rate_in_h, baseline_rate_holdout)

            rows.append({
                "pair": f"{left}-{right}",
                "aspect": float(aspect),
                "in_window_bars": n_in,
                "rate_in": rate_in,
                "lift": lift,
                "binomial_p": binom_p,
                "random_p": random_p,
                "holdout_in_bars": n_in_h,
                "holdout_lift": holdout_lift,
            })

    table = pd.DataFrame.from_records(rows)
    if not table.empty:
        reject = stats.benjamini_hochberg(table["binomial_p"].to_numpy(), alpha=fdr_alpha)
        table["bh_significant"] = reject
        table = table.sort_values("lift", ascending=False).reset_index(drop=True)
    else:
        table["bh_significant"] = pd.Series(dtype=bool)

    top = table.head(25).to_dict(orient="records")
    results = {
        "config": {
            "symbol": symbol, "timeframe": timeframe, "start": start, "end": end,
            "orb_deg": orb_deg, "window_bars": window_bars, "window_kind": window_kind,
            "pivot_threshold_atr": pivot_threshold_atr, "fdr_alpha": fdr_alpha,
            "holdout_start": str(holdout_start), "min_in_window_bars": min_in_window_bars,
            "random_draws": random_draws,
        },
        "data": {"bars": n, "n_pivots": len(piv)},
        "baseline_rate": baseline_rate,
        "baseline_rate_holdout": baseline_rate_holdout,
        "n_hypotheses": int(len(table)),
        "n_significant": int(table["bh_significant"].sum()) if not table.empty else 0,
        "top": top,
    }
    return {"results": results, "table": table}
