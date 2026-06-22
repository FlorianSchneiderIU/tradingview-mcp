"""Do pivots concentrate at Fibonacci TIME levels? (event study + placebo control)

For HTF (zigzag) pivots, build several "time calendars" and ask, for each, whether a
pivot lands within +/- window of a firing more than random windows of the same
count/width (calendar_search.event_study):

  fib_ratio      Fib ratios (0.382/0.618/1.0/1.618/2.618) x prior swing duration
  fib_ratio_ex1  same, EXCLUDING 1.0 (removes the "swing simply repeats" confound)
  repeat_1x      ONLY 1.0x (pure swing-duration persistence benchmark)
  nonfib_ratio   ordinary non-Fib ratios (0.5/0.75/0.9/1.1/1.3/1.5/2.0) -> placebo
  fib_zone       Fibonacci-number bar offsets (1,2,3,5,8,13,...) from each pivot
  nonfib_zone    non-Fib bar offsets of similar density -> placebo
  random         random time calendar (same count) -> null

Verdict: a real Fibonacci-TIME effect needs fib_ratio_ex1 to beat BOTH nonfib_ratio and
random (and be more than just repeat_1x). Outputs reports/fib_time_<tf>.{json,md}.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PKG_SRC = Path(__file__).resolve().parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from astro_reversal import (  # noqa: E402
    baselines, calendar_search, data, fib_time, ltf_structure as lts, pivots, report,
)


def _two_prop_p(k1, n1, k2, n2) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se <= 0:
        return float("nan")
    z = (k1 / n1 - k2 / n2) / se
    return float(2 * (1 - scipy_stats.norm.cdf(abs(z))))


def main() -> int:
    p = argparse.ArgumentParser(description="Fibonacci-time event study with non-Fib placebo control.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--timeframe", default="1d", choices=["1h", "4h", "1d"])
    p.add_argument("--pivot-threshold-atr", type=float, default=3.0)
    p.add_argument("--window-bars", type=int, default=None, help="time tolerance; default windows.medium")
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    symbol = cfg["data"]["symbol"]
    tf = args.timeframe
    window_bars = args.window_bars or int(cfg["windows"][tf]["medium"])
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")
    n_draws = int(cfg.get("random_calendar_draws", 1000))
    seed = int(cfg.get("random_seed", 42))

    frame = data.load_ohlcv(symbol, tf, cfg["data"]["start"], cfg["data"]["end"],
                            base_interval=cfg["data"]["base_interval"])
    n = len(frame)
    times = pd.to_datetime(frame["open_time"], utc=True)
    holdout_mask = (times >= holdout_start).to_numpy()

    piv = lts.htf_pivots(frame, args.pivot_threshold_atr)[0]
    near_mask = pivots.pivot_within_window_mask(piv, n, window_bars, kind=None)

    fib_ex1 = tuple(r for r in fib_time.FIB_RATIOS if r != 1.0)
    calendars = {
        "fib_ratio": fib_time.swing_ratio_levels(piv, n, fib_time.FIB_RATIOS),
        "fib_ratio_ex1": fib_time.swing_ratio_levels(piv, n, fib_ex1),
        "repeat_1x": fib_time.swing_ratio_levels(piv, n, (1.0,)),
        "nonfib_ratio": fib_time.swing_ratio_levels(piv, n, fib_time.NONFIB_RATIOS),
        "fib_zone": fib_time.fib_zone_levels(piv, n, fib_time.FIB_ZONES),
        "nonfib_zone": fib_time.fib_zone_levels(piv, n, fib_time.NONFIB_ZONES),
    }
    # Random-time null matched to the fib_ratio firing count.
    ref_count = max(1, len(calendars["fib_ratio"]))
    calendars["random"] = baselines.random_calendars(ref_count, n, 1, min_spacing=1, seed=seed)[0]

    books = {}
    for name, eb in calendars.items():
        res = calendar_search.event_study(eb, near_mask, window_bars, holdout_mask,
                                          n_draws=n_draws, seed=seed, min_event_count=5)
        if res is None:
            books[name] = {"n_events": int(len(eb)), "skipped": True}
            continue
        res["hits"] = int(round(res["hit_rate"] * res["n_events"]))
        books[name] = res

    # Fib vs placebo / persistence two-proportion tests (on fib_ratio_ex1).
    base = books.get("fib_ratio_ex1", {})
    comparisons = {}
    if not base.get("skipped"):
        for other in ("nonfib_ratio", "repeat_1x", "random"):
            o = books.get(other, {})
            if not o.get("skipped"):
                comparisons[f"fib_ex1_vs_{other}"] = _two_prop_p(
                    base["hits"], base["n_events"], o["hits"], o["n_events"])

    results = {
        "config": {"symbol": symbol, "timeframe": tf, "window_bars": window_bars,
                   "pivot_threshold_atr": args.pivot_threshold_atr, "n_pivots": len(piv),
                   "baseline_hit": float(np.mean(near_mask)), "holdout_start": str(holdout_start)},
        "data": {"bars": n},
        "books": books,
        "fib_vs_placebo_pvalues": comparisons,
    }
    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / f"fib_time_{tf}.json", results)
    (args.outdir / f"fib_time_{tf}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r):
    c = r["config"]
    L = [
        f"# Fibonacci-TIME event study - {c['timeframe']}",
        "",
        f"**{c['symbol']}** | {r['data']['bars']} bars | {c['n_pivots']} pivots (ATR {c['pivot_threshold_atr']}) | "
        f"tolerance +/-{c['window_bars']} bars | baseline pivot-window rate {_f(c['baseline_hit'])}.",
        "",
        "Question: do pivots land at Fibonacci TIME projections more than at ordinary (non-Fib) ratios "
        "of the same prior swing, and more than random? **Decisive = fib_ratio_ex1 beats nonfib_ratio AND "
        "random** (and is more than the repeat_1x persistence benchmark).",
        "",
        "| Calendar | Firings | Hit rate | Baseline | Lift | Binom p | Rand p | Holdout lift |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = ["fib_ratio", "fib_ratio_ex1", "repeat_1x", "nonfib_ratio", "fib_zone", "nonfib_zone", "random"]
    for name in order:
        b = r["books"].get(name, {})
        if b.get("skipped"):
            L.append(f"| {name} | {b.get('n_events', 0)} | (too few) | | | | | |")
            continue
        L.append(f"| {name} | {b['n_events']} | {_f(b['hit_rate'])} | {_f(b['baseline_hit'])} | "
                 f"{_f(b['lift'])} | {_f(b['binomial_p'], 4)} | {_f(b['random_p'], 4)} | {_f(b['holdout_lift'])} |")
    L += ["", "## Fib(ex-1.0) vs placebo / persistence (two-proportion p-values)", ""]
    for k, v in r["fib_vs_placebo_pvalues"].items():
        L.append(f"- {k}: p = {_f(v, 4)}")
    L += [
        "",
        "## Reading guide",
        "",
        "If `fib_ratio_ex1` lift ~= `nonfib_ratio` lift, the clustering is just swing-duration "
        "persistence, not Fibonacci. If `fib_ratio_ex1` ~ `random` (lift ~1), there is no time effect at "
        "all. A real Fib-time edge shows fib_ratio_ex1 lift clearly > nonfib and > random with small "
        "two-proportion p-values and a holdout lift > 1.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
