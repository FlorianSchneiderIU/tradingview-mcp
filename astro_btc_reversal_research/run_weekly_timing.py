"""When do BTC weekly lows/highs print within the week, and are they retested?

Outputs (reports/): weekly_timing.json, weekly_timing.md, weekly_records.csv
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

from astro_reversal import data, report, weekly_timing as wt  # noqa: E402


def _heatmap(records: pd.DataFrame, which: str) -> dict:
    """Day-of-week x session fraction-of-weeks grid for the extreme `which`."""
    sess_names = [s[0] for s in wt.SESSIONS]
    grid = {d: {s: 0 for s in sess_names} for d in wt.DOW_NAMES}
    for _, r in records.iterrows():
        grid[wt.DOW_NAMES[int(r[f"{which}_dow"])]][wt.session_of(int(r[f"{which}_hour"]))] += 1
    n = len(records)
    return {d: {s: grid[d][s] / n for s in sess_names} for d in wt.DOW_NAMES}


def main() -> int:
    p = argparse.ArgumentParser(description="Weekly low/high timing + intra-week retest study.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    w = cfg["weekly_timing"]
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    interval = w["interval"]

    frame = data.load_ohlcv(symbol, interval, start, end, base_interval=interval)
    recs = wt.weekly_records(frame, w["retest_tol_frac"], w["move_away_frac"], w["min_week_bars"])

    low_dist = wt.distribution(recs, "low")
    high_dist = wt.distribution(recs, "high")
    n = len(recs)
    retest = {
        "p_retest_low": float(recs["retest_low"].mean()),
        "p_retest_high": float(recs["retest_high"].mean()),
        "low_retest_dow": {wt.DOW_NAMES[int(k)]: float(v) for k, v in
                           recs.loc[recs["retest_low"], "retest_low_dow"].value_counts(normalize=True).sort_index().items()},
        "high_retest_dow": {wt.DOW_NAMES[int(k)]: float(v) for k, v in
                            recs.loc[recs["retest_high"], "retest_high_dow"].value_counts(normalize=True).sort_index().items()},
        "low_retest_median_hours_after": float(recs.loc[recs["retest_low"], "retest_low_bars_after"].median() * 0.25)
        if recs["retest_low"].any() else float("nan"),
        "high_retest_median_hours_after": float(recs.loc[recs["retest_high"], "retest_high_bars_after"].median() * 0.25)
        if recs["retest_high"].any() else float("nan"),
    }

    results = {
        "config": {"symbol": symbol, "interval": interval, "start": start, "end": end,
                   "retest_tol_frac": w["retest_tol_frac"], "move_away_frac": w["move_away_frac"]},
        "n_weeks": n,
        "low_first_fraction": float(recs["low_first"].mean()),
        "low_timing": low_dist,
        "high_timing": high_dist,
        "low_heatmap": _heatmap(recs, "low"),
        "high_heatmap": _heatmap(recs, "high"),
        "retest": retest,
    }

    report.write_csv(args.outdir / "weekly_records.csv", recs)
    report.write_json(args.outdir / "weekly_timing.json", results)
    (args.outdir / "weekly_timing.md").write_text(_markdown(results), encoding="utf-8")
    print(_markdown(results))
    return 0


def _bar(frac, width=20):
    return "#" * int(round(frac * width))


def _dow_table(dist, title):
    L = [f"### {title} - day of week", "", "| Day | % of weeks |", "|---|---|"]
    for day, v in dist["dow"].items():
        L.append(f"| {day} | {v*100:5.1f}%  {_bar(v)} |")
    L += ["", "| Session (UTC) | % of weeks |", "|---|---|"]
    for sess, v in sorted(dist["session"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {sess} | {v*100:5.1f}% |")
    L.append(f"\nMean position in week: {dist['mean_frac_into_week']*100:.0f}% "
             f"(median {dist['median_frac_into_week']*100:.0f}%).")
    return L


def _heat_table(grid, title):
    sess = [s[0] for s in wt.SESSIONS]
    L = [f"### {title} - day x session (% of weeks)", "",
         "| Day | " + " | ".join(sess) + " |", "|---|" + "---|" * len(sess)]
    for day, row in grid.items():
        L.append(f"| {day} | " + " | ".join(f"{row[s]*100:.1f}" for s in sess) + " |")
    return L


def _markdown(r: dict) -> str:
    c = r["config"]
    rt = r["retest"]
    L = [
        "# Weekly Low/High Timing + Intra-week Retest",
        "",
        f"**{c['symbol']}** {c['interval']} | {c['start']} -> {c['end']} | {r['n_weeks']} complete weeks "
        "(Mon 00:00 -> Sun 23:59 UTC).",
        f"Weekly LOW prints before the weekly HIGH in **{r['low_first_fraction']*100:.0f}%** of weeks "
        "(i.e. that share of weeks closed up from their low-to-high sequence).",
        "",
        "## Weekly LOW timing",
        "",
        *_dow_table(r["low_timing"], "Weekly low"),
        "",
        *_heat_table(r["low_heatmap"], "Weekly low"),
        "",
        "## Weekly HIGH timing",
        "",
        *_dow_table(r["high_timing"], "Weekly high"),
        "",
        *_heat_table(r["high_heatmap"], "Weekly high"),
        "",
        "## Intra-week retest of the extreme",
        "",
        f"- Weekly LOW retested later the same week: **{rt['p_retest_low']*100:.0f}%** of weeks "
        f"(median {rt['low_retest_median_hours_after']:.0f}h after the low).",
        f"- Weekly HIGH retested later the same week: **{rt['p_retest_high']*100:.0f}%** of weeks "
        f"(median {rt['high_retest_median_hours_after']:.0f}h after the high).",
        f"- Retest = price leaves by {c['move_away_frac']*100:.0f}% of the weekly range, then returns "
        f"within {c['retest_tol_frac']*100:.0f}% of the extreme.",
        "",
        "Low-retest day-of-week: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in rt["low_retest_dow"].items()),
        "",
        "High-retest day-of-week: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in rt["high_retest_dow"].items()),
        "",
        "## How to use",
        "",
        "Concentrate spring-long hunting in the day/session where weekly LOWS cluster, and look for the "
        "low's retest window as a second entry. Treat clustering as a mild prior, not a guarantee - check "
        "the bars and whether the modal cell is much above the uniform 1/7 (14.3%) per day baseline.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
