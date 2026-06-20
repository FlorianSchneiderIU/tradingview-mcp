"""Direction-conditional calendar search (astro + price context) - CLI.

Tests the surviving hypothesis: an aspect firing may only matter when price
dumped (expect a bottom) or pumped (expect a top) into it. The baseline is random
bars from the SAME context, so lift isolates the astro contribution beyond the
move itself.

Outputs (reports/): conditional_<tf>.csv, conditional_<tf>.json, conditional_<tf>.md
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
    event_labels,
    report,
    reuse,
    stats,
)


def _search(direction, cond_mask, near_mask, raw, available, aspects, orb_deg, window_bars,
            holdout_mask, n_draws, seed, min_events, fdr_alpha):
    rows = []
    for left, right in combinations(available, 2):
        sep = (raw[f"{left}_lon"].to_numpy(float) - raw[f"{right}_lon"].to_numpy(float)) % 360.0
        for aspect in aspects:
            eb = calendar_search.aspect_event_bars(sep, aspect, orb_deg)
            res = calendar_search.conditional_event_study(
                eb, cond_mask, near_mask, window_bars, holdout_mask,
                n_draws=n_draws, seed=seed, min_event_count=min_events)
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

    # Dark Pivot in this context.
    dark = None
    if {"moon", "pluto"}.issubset(available):
        sep_mp = (raw["moon_lon"].to_numpy(float) - raw["pluto_lon"].to_numpy(float)) % 360.0
        dp_events = np.unique(np.concatenate([
            calendar_search.aspect_event_bars(sep_mp, a, orb_deg) for a in (0, 90, 180, 270)]))
        dark = calendar_search.conditional_event_study(
            dp_events, cond_mask, near_mask, window_bars, holdout_mask,
            n_draws=n_draws, seed=seed, min_event_count=min_events)

    base = float(near_mask[np.flatnonzero(cond_mask)].mean()) if cond_mask.any() else float("nan")
    return {
        "n_context_bars": int(cond_mask.sum()),
        "baseline_hit": base,
        "n_hypotheses": int(len(table)),
        "n_significant": int(table["bh_significant"].sum()) if not table.empty else 0,
        "dark_pivot": dark,
        "top": table.head(25).to_dict(orient="records"),
    }, table


def main() -> int:
    p = argparse.ArgumentParser(description="Direction-conditional calendar search.")
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
    cc, cal = cfg["conditional"], cfg["calendar"]
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    tf = args.timeframe
    window_kind = args.window_kind or cc["window_kind"]
    window_bars = int(cfg["windows"][tf][window_kind])
    orb_deg = float(cfg["orb"]["discovery_orb_degrees"][tf])
    pivot_thr = args.pivot_threshold_atr if args.pivot_threshold_atr is not None \
        else float(cal["pivot_threshold_atr"][tf])
    seed = int(cfg.get("random_seed", 42))
    n_draws = int(cc["random_draws"])
    min_events = int(cc["min_event_count"])
    fdr_alpha = float(cc["fdr_alpha"])
    lookback = int(cc["dump_lookback"][tf])
    move_thr = float(cc["dump_threshold_atr"])
    bodies, aspects = cfg["bodies"], cfg["aspects"]["discovery"]

    frame = data.load_ohlcv(symbol, tf, start, end, base_interval=cfg["data"]["base_interval"])
    n = len(frame)
    times = pd.to_datetime(frame["open_time"], utc=True)
    holdout_mask = (times >= pd.Timestamp(cfg["holdout_start"], tz="UTC")).to_numpy()

    low_mask, piv = calendar_search.directional_pivot_mask(frame, pivot_thr, window_bars, "low")
    high_mask, _ = calendar_search.directional_pivot_mask(frame, pivot_thr, window_bars, "high")
    dumps = event_labels.dump_flags(frame, lookback, move_thr)
    pumps = event_labels.pump_flags(frame, lookback, move_thr)

    raw = reuse.compute_skyfield_positions(frame["open_time"], reuse.DEFAULT_CACHE_DIR, tf).reset_index(drop=True)
    available = [b for b in bodies if f"{b}_lon" in raw.columns]

    common = dict(raw=raw, available=available, aspects=aspects, orb_deg=orb_deg,
                  window_bars=window_bars, holdout_mask=holdout_mask, n_draws=n_draws,
                  seed=seed, min_events=min_events, fdr_alpha=fdr_alpha)
    dump_block, dump_tab = _search("dump_bottom", dumps, low_mask, **common)
    pump_block, pump_tab = _search("pump_top", pumps, high_mask, **common)

    results = {
        "config": {"symbol": symbol, "timeframe": tf, "start": start, "end": end,
                   "window_bars": window_bars, "window_kind": window_kind, "orb_deg": orb_deg,
                   "pivot_threshold_atr": pivot_thr, "dump_threshold_atr": move_thr,
                   "dump_lookback": lookback, "fdr_alpha": fdr_alpha,
                   "holdout_start": str(cfg["holdout_start"])},
        "data": {"bars": n, "n_pivots": len(piv), "n_dump_bars": int(dumps.sum()),
                 "n_pump_bars": int(pumps.sum())},
        "dump_bottom": dump_block,
        "pump_top": pump_block,
    }

    outdir = args.outdir
    combined = pd.concat([
        dump_tab.assign(context="dump_bottom"), pump_tab.assign(context="pump_top")], ignore_index=True)
    report.write_csv(outdir / f"conditional_{tf}.csv", combined)
    report.write_json(outdir / f"conditional_{tf}.json", results)
    (outdir / f"conditional_{tf}.md").write_text(report.conditional_calendar_markdown(results), encoding="utf-8")
    print(report.conditional_calendar_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
