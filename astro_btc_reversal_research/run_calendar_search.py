"""Astro calendar search (precision framing) - CLI.

Finds sparse astro calendars whose firings reliably contain a pivot within a
tolerance window, beating random windows of the same count/width. Searches every
single pair x aspect, the Moon-Pluto Dark Pivot, and aspect-confluence calendars.

Outputs (reports/): calendar_<tf>.csv, calendar_<tf>.json, calendar_<tf>.md
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PKG_SRC = Path(__file__).resolve().parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from astro_reversal import (  # noqa: E402
    calendar_search,
    data,
    report,
    reuse,
    stats,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Astro calendar search (precision framing).")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--timeframe", default="1d", choices=["1h", "4h", "1d"])
    p.add_argument("--symbol", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--window-kind", default=None, choices=["tight", "medium", "wide"])
    p.add_argument("--pivot-threshold-atr", type=float, default=None)
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    cc = cfg["calendar"]
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    tf = args.timeframe
    window_kind = args.window_kind or cc["window_kind"]
    window_bars = int(cfg["windows"][tf][window_kind])
    orb_deg = float(cfg["orb"]["discovery_orb_degrees"][tf])
    pivot_thr = args.pivot_threshold_atr if args.pivot_threshold_atr is not None \
        else float(cc["pivot_threshold_atr"][tf])
    seed = int(cfg.get("random_seed", 42))
    n_draws = int(cc["random_draws"])
    min_events = int(cc["min_event_count"])
    fdr_alpha = float(cc["fdr_alpha"])
    bodies = cfg["bodies"]
    aspects = cfg["aspects"]["discovery"]
    offsets_days = list(cfg.get("shifted_calendar_offsets_days", [3, 7, 13, 21, 37, 83]))

    frame = data.load_ohlcv(symbol, tf, start, end, base_interval=cfg["data"]["base_interval"])
    n = len(frame)
    times = pd.to_datetime(frame["open_time"], utc=True)
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")
    holdout_mask = (times >= holdout_start).to_numpy()

    near_mask, piv = calendar_search.near_pivot_mask(frame, pivot_thr, window_bars)
    baseline_hit = float(near_mask.mean())
    raw = reuse.compute_skyfield_positions(frame["open_time"], reuse.DEFAULT_CACHE_DIR, tf).reset_index(drop=True)
    available = [b for b in bodies if f"{b}_lon" in raw.columns]

    def study(event_bars):
        return calendar_search.event_study(event_bars, near_mask, window_bars, holdout_mask,
                                           n_draws=n_draws, seed=seed, min_event_count=min_events)

    # 1) Single pair x aspect calendars.
    rows = []
    for left, right in combinations(available, 2):
        sep = (raw[f"{left}_lon"].to_numpy(float) - raw[f"{right}_lon"].to_numpy(float)) % 360.0
        for aspect in aspects:
            eb = calendar_search.aspect_event_bars(sep, aspect, orb_deg)
            res = study(eb)
            if res is None:
                continue
            res.update({"pair": f"{left}-{right}", "aspect": float(aspect)})
            rows.append(res)
    table = pd.DataFrame.from_records(rows)
    if not table.empty:
        table["bh_significant"] = stats.benjamini_hochberg(table["binomial_p"].to_numpy(), fdr_alpha)
        table = table.sort_values("hit_rate", ascending=False).reset_index(drop=True)
    else:
        table["bh_significant"] = pd.Series(dtype=bool)

    # 2) Dark Pivot calendar (union of Moon-Pluto hard aspects).
    if {"moon", "pluto"}.issubset(available):
        sep_mp = (raw["moon_lon"].to_numpy(float) - raw["pluto_lon"].to_numpy(float)) % 360.0
        dp_events = np.unique(np.concatenate([
            calendar_search.aspect_event_bars(sep_mp, a, orb_deg) for a in (0, 90, 180, 270)]))
    else:
        dp_events = np.array([], dtype=int)
    dark_pivot = study(dp_events) or {"n_events": 0, "hit_rate": float("nan"), "baseline_hit": baseline_hit,
                                      "lift": float("nan"), "coverage": float("nan"),
                                      "binomial_p": float("nan"), "random_p": float("nan"),
                                      "holdout_hit_rate": float("nan"), "holdout_baseline": float("nan"),
                                      "holdout_lift": float("nan")}
    bpd = (n - 1) / max(1e-9, (times.iloc[-1] - times.iloc[0]).total_seconds() / 86400)
    dp_shifted = calendar_search.shifted_hit_rates(
        dp_events, near_mask, [int(round(d * bpd)) for d in offsets_days])

    # 3) Confluence calendars.
    confluence = []
    for k in cc["confluence_min_counts"]:
        ev, _ = calendar_search.confluence_event_bars(raw, bodies, cc["confluence_aspects"], orb_deg, int(k))
        res = study(ev)
        if res is None:
            res = {"n_events": int(ev.size), "hit_rate": float("nan"), "baseline_hit": baseline_hit,
                   "lift": float("nan"), "binomial_p": float("nan"), "holdout_hit_rate": float("nan")}
        res["min_count"] = int(k)
        confluence.append(res)

    results = {
        "config": {"symbol": symbol, "timeframe": tf, "start": start, "end": end,
                   "window_bars": window_bars, "window_kind": window_kind, "orb_deg": orb_deg,
                   "pivot_threshold_atr": pivot_thr, "fdr_alpha": fdr_alpha,
                   "holdout_start": str(holdout_start)},
        "data": {"bars": n, "n_pivots": len(piv), "baseline_hit": baseline_hit},
        "n_hypotheses": int(len(table)),
        "n_significant": int(table["bh_significant"].sum()) if not table.empty else 0,
        "dark_pivot": dark_pivot,
        "dark_pivot_shifted": dp_shifted,
        "confluence": confluence,
        "top": table.head(30).to_dict(orient="records"),
    }

    outdir = args.outdir
    report.write_csv(outdir / f"calendar_{tf}.csv", table)
    report.write_json(outdir / f"calendar_{tf}.json", results)
    (outdir / f"calendar_{tf}.md").write_text(report.calendar_markdown(results), encoding="utf-8")
    print(report.calendar_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
