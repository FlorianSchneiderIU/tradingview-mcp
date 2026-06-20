"""Milestone 1 - Dark Pivot candidate test (CLI).

Question: when BTC dumps into or on a Moon-Pluto hard-aspect day, do the next
N candles show bullish expansion more often than on ordinary dump days?

Outputs (reports/): dark_pivot_events.csv, dark_pivot_results.json, dark_pivot_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG_SRC = Path(__file__).resolve().parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from astro_reversal import (  # noqa: E402
    baselines,
    data,
    ephemeris_events,
    event_labels,
    pivots,
    report,
    stats,
)


def _window_mask(event_bars: np.ndarray, n_bars: int, half: int) -> np.ndarray:
    mask = np.zeros(n_bars, dtype=bool)
    for b in event_bars:
        if b < 0:
            continue
        mask[max(0, b - half): min(n_bars - 1, b + half) + 1] = True
    return mask


def _rate_for_window(window_mask, dump_bars, success_map) -> tuple[float, int, list]:
    sel = [b for b in dump_bars if window_mask[b] and b in success_map]
    succ = [success_map[b] for b in sel]
    rate = float(np.mean(succ)) if succ else float("nan")
    return rate, len(sel), succ


def main() -> int:
    p = argparse.ArgumentParser(description="Dark Pivot candidate test (Moon-Pluto).")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--pivot-threshold-atr", type=float, default=2.0)
    p.add_argument("--dump-lookback", type=int, default=2)
    p.add_argument("--dump-threshold-atr", type=float, default=1.0)
    p.add_argument("--expansion-horizon", type=int, default=3)
    p.add_argument("--expansion-target-atr", type=float, default=1.0)
    p.add_argument("--expansion-buffer-atr", type=float, default=0.1)
    p.add_argument("--event-window-bars", type=int, default=1)
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    base_interval = cfg["data"]["base_interval"]
    seed = int(cfg.get("random_seed", 42))
    n_draws = int(cfg.get("random_calendar_draws", 1000))
    offsets_days = list(cfg.get("shifted_calendar_offsets_days", [3, 7, 13, 21, 37, 83]))
    dark = cfg["special_candidates"]["dark_pivot"]
    holdout_start = pd.Timestamp(cfg["holdout_start"]).tz_convert("UTC") if "+" in str(cfg["holdout_start"]) \
        else pd.Timestamp(cfg["holdout_start"], tz="UTC")

    frame = data.load_ohlcv(symbol, args.timeframe, start, end, base_interval=base_interval)
    n = len(frame)
    open_times = pd.to_datetime(frame["open_time"], utc=True)

    # 1) Moon-Pluto hard-aspect events -> candle bars.
    events = ephemeris_events.compute_aspect_events(
        dark["body_1"], dark["body_2"], dark["aspects"], start, end,
    )
    events = ephemeris_events.map_events_to_candles(events, frame)
    event_bars = np.unique(events.loc[events["bar_index"] >= 0, "bar_index"].to_numpy(dtype=int))
    half = args.event_window_bars
    event_window = _window_mask(event_bars, n, half)

    # 2) Dump days + precomputed bullish expansion outcome for every dump day.
    dumps = event_labels.dump_flags(frame, args.dump_lookback, args.dump_threshold_atr)
    dump_bars = [int(b) for b in np.flatnonzero(dumps)]
    success_map: dict[int, bool] = {}
    expansion_map: dict[int, dict] = {}
    for b in dump_bars:
        res = event_labels.evaluate_expansion(
            frame, b, "bull", args.expansion_horizon, args.expansion_target_atr, args.expansion_buffer_atr,
        )
        if res is not None:
            success_map[b] = bool(res["success"])
            expansion_map[b] = res

    # 3) Set A (event-window dumps) vs Set B (ordinary dumps).
    a_bars = [b for b in dump_bars if event_window[b] and b in success_map]
    b_bars = [b for b in dump_bars if not event_window[b] and b in success_map]
    succ_a = [success_map[b] for b in a_bars]
    succ_b = [success_map[b] for b in b_bars]
    rate_a = stats.hit_rate(succ_a)
    rate_b = stats.hit_rate(succ_b)

    expansion_test = {
        "n_event": len(a_bars),
        "n_baseline": len(b_bars),
        "rate_event": rate_a,
        "rate_baseline": rate_b,
        "lift": stats.lift(rate_a, rate_b),
        "rate_diff": float(rate_a - rate_b) if np.isfinite(rate_a) and np.isfinite(rate_b) else float("nan"),
        "ci_event": stats.bootstrap_rate_ci(succ_a, seed=seed),
        "ci_baseline": stats.bootstrap_rate_ci(succ_b, seed=seed),
        "diff_ci": stats.bootstrap_diff_ci(succ_a, succ_b, seed=seed),
        "binomial_p": stats.binomial_test(int(np.sum(succ_a)), len(succ_a), rate_b),
        "mfe_mae": _mfe_mae_summary([expansion_map[b] for b in a_bars]),
    }

    # 4) Random-calendar null: same #events, similar spacing, scored identically.
    spacing = int(np.median(np.diff(event_bars))) if event_bars.size > 1 else 1
    rand_cals = baselines.random_calendars(
        event_count=int(event_bars.size), n_bars=n, n_draws=n_draws,
        min_spacing=max(1, spacing), seed=seed,
    )
    null_rates = []
    for cal in rand_cals:
        wmask = _window_mask(cal, n, half)
        r_rate, r_n, _ = _rate_for_window(wmask, dump_bars, success_map)
        if r_n >= 5:
            null_rates.append(r_rate)
    random_calendar = {
        "n_draws": n_draws,
        "n_valid_draws": len(null_rates),
        "min_spacing_bars": max(1, spacing),
        "null_mean": float(np.mean(null_rates)) if null_rates else float("nan"),
        "empirical_p": stats.empirical_pvalue(rate_a, null_rates),
    }

    # 5) Shifted-calendar baseline.
    bpd = baselines.bars_per_day(frame)
    offsets_bars = [int(round(d * bpd)) for d in offsets_days]
    shifted = baselines.shifted_calendars(event_bars, offsets_bars, n)
    shifted_calendar = []
    for off_days, off_bars in zip(offsets_days, offsets_bars):
        wmask = _window_mask(shifted.get(off_bars, np.array([], dtype=int)), n, half)
        s_rate, s_n, _ = _rate_for_window(wmask, dump_bars, success_map)
        shifted_calendar.append({"offset_days": off_days, "offset_bars": off_bars, "n": s_n, "rate": s_rate})

    # 6) Holdout (out-of-sample) check on set A.
    dev_a = [b for b in a_bars if open_times.iloc[b] < holdout_start]
    hold_a = [b for b in a_bars if open_times.iloc[b] >= holdout_start]
    holdout = {
        "n_event_dev": len(dev_a),
        "rate_event_dev": stats.hit_rate([success_map[b] for b in dev_a]),
        "n_event_holdout": len(hold_a),
        "rate_event_holdout": stats.hit_rate([success_map[b] for b in hold_a]),
    }

    # 7) Pivot proximity (secondary).
    piv = pivots.atr_directional_pivots(frame, args.pivot_threshold_atr)
    pivot_low_mask = pivots.pivot_within_window_mask(piv, n, half, kind="low")
    event_pivot_share = float(np.mean(pivot_low_mask[event_bars])) if event_bars.size else float("nan")
    baseline_pivot_share = float(np.mean(pivot_low_mask))
    pivot_proximity = {
        "n_pivots": len(piv),
        "event_pivot_share": event_pivot_share,
        "baseline_pivot_share": baseline_pivot_share,
        "lift": stats.lift(event_pivot_share, baseline_pivot_share),
    }

    results = {
        "config": {
            "symbol": symbol, "timeframe": args.timeframe, "start": start, "end": end,
            "body_1": dark["body_1"], "body_2": dark["body_2"], "aspects": dark["aspects"],
            "dump_lookback": args.dump_lookback, "dump_threshold_atr": args.dump_threshold_atr,
            "expansion_horizon": args.expansion_horizon, "expansion_target_atr": args.expansion_target_atr,
            "expansion_buffer_atr": args.expansion_buffer_atr, "event_window_bars": args.event_window_bars,
            "pivot_threshold_atr": args.pivot_threshold_atr, "holdout_start": str(holdout_start),
            "random_seed": seed,
        },
        "data": {
            "bars": n,
            "first_open": open_times.iloc[0],
            "last_close": pd.Timestamp(frame["close_time"].iloc[-1]),
            "n_aspect_events": int(event_bars.size),
            "n_dump_days": len(dump_bars),
        },
        "expansion_test": expansion_test,
        "random_calendar": random_calendar,
        "shifted_calendar": shifted_calendar,
        "holdout": holdout,
        "pivot_proximity": pivot_proximity,
    }

    outdir = args.outdir
    report.write_csv(outdir / "dark_pivot_events.csv", events)
    report.write_json(outdir / "dark_pivot_results.json", results)
    (outdir / "dark_pivot_report.md").write_text(report.dark_pivot_markdown(results), encoding="utf-8")
    print(report.dark_pivot_markdown(results))
    return 0


def _mfe_mae_summary(rows: list[dict]) -> dict:
    if not rows:
        return {k: float("nan") for k in
                ["mean_mfe_r", "mean_mae_r", "median_max_r", "share_ge_1r", "share_ge_2r"]}
    mfe = np.array([r["mfe_r"] for r in rows], dtype=float)
    mae = np.array([r["mae_r"] for r in rows], dtype=float)
    maxr = np.array([r["max_r_available"] for r in rows], dtype=float)
    return {
        "mean_mfe_r": float(np.mean(mfe)),
        "mean_mae_r": float(np.mean(mae)),
        "median_max_r": float(np.median(maxr)),
        "share_ge_1r": float(np.mean(maxr >= 1.0)),
        "share_ge_2r": float(np.mean(maxr >= 2.0)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
