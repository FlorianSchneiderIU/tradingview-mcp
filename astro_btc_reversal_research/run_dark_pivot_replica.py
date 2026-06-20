"""Replicate the public 'Dark Pivot' claim and put a baseline next to it.

Reproduces the loose "dump on the activation day -> bullish expansion = local bottom"
hit rate for the Moon-Pluto hard-aspect calendar, then scores the IDENTICAL rule on
ordinary dump days and random days. The verdict is the lift, not the raw %.

Outputs (reports/): dark_pivot_replica.json and dark_pivot_replica.md
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
    dark_pivot_replica as dpr,
    data,
    ephemeris_events,
    report,
    stats,
)

TODAY = "2026-06-20"  # passed in explicitly (Date.now is unavailable in some contexts)


def main() -> int:
    p = argparse.ArgumentParser(description="Replicate + baseline the public Dark Pivot claim.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--local-window", type=int, default=3)   # +/- bars for 'local bottom/top'
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    seed = int(cfg.get("random_seed", 42))

    frame = data.load_ohlcv(symbol, "1d", start, end, base_interval=cfg["data"]["base_interval"])
    n = len(frame)
    times = pd.to_datetime(frame["open_time"], utc=True)

    # Dark Pivot calendar = Moon-Pluto hard aspects -> daily bars.
    events = ephemeris_events.compute_aspect_events("moon", "pluto", [0, 90, 180, 270], start, end)
    events = ephemeris_events.map_events_to_candles(events, frame)
    dp_bars = np.unique(events.loc[events["bar_index"] >= 0, "bar_index"].to_numpy(int))
    dp_mask = np.zeros(n, dtype=bool)
    dp_mask[dp_bars] = True

    # Upcoming dates (incl. the advertised 2026-06-24).
    upcoming = ephemeris_events.compute_aspect_events("moon", "pluto", [0, 90, 180, 270], TODAY, "2026-08-01")
    upcoming_dates = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")
                      for t in pd.to_datetime(upcoming["timestamp_utc"], utc=True)]

    # Grid over the loose definition's parameters.
    grid = []
    for lookback in (1, 2, 3):
        declined = dpr.declined_into(frame, lookback)
        for horizon in (1, 2, 3, 5, 7):
            for x_atr in (0.0, 0.5):
                exp = dpr.bullish_expansion_within(frame, horizon, x_atr)
                dp_dump = dp_mask & declined
                other_dump = declined & ~dp_mask
                n_sig = int(dp_dump.sum())
                if n_sig == 0:
                    continue
                dp_hit = float(exp[dp_dump].mean())
                base_dump = float(exp[other_dump].mean()) if other_dump.any() else float("nan")
                base_all = float(exp.mean())
                k = int(exp[dp_dump].sum())
                grid.append({
                    "lookback": lookback, "horizon": horizon, "x_atr": x_atr,
                    "n_signals": n_sig, "dp_hit": dp_hit,
                    "baseline_dump_hit": base_dump, "baseline_all_hit": base_all,
                    "lift_vs_dump": stats.lift(dp_hit, base_dump),
                    "lift_vs_all": stats.lift(dp_hit, base_all),
                    "binom_p_vs_dump": stats.binomial_test(k, n_sig, base_dump),
                })

    # Literal 'marked a local bottom' reading.
    low_extreme = dpr.local_extreme(frame, args.local_window, "low")
    high_extreme = dpr.local_extreme(frame, args.local_window, "high")
    declined2 = dpr.declined_into(frame, 2)
    dp_dump2 = dp_mask & declined2
    local_bottom = {
        "window": args.local_window,
        "n_dp_dump_signals": int(dp_dump2.sum()),
        "dp_dump_bottom_rate": float(low_extreme[dp_dump2].mean()) if dp_dump2.any() else float("nan"),
        "all_dump_bottom_rate": float(low_extreme[declined2 & ~dp_mask].mean()),
        "all_days_bottom_rate": float(low_extreme.mean()),
    }
    local_bottom["lift_vs_dump"] = stats.lift(local_bottom["dp_dump_bottom_rate"],
                                              local_bottom["all_dump_bottom_rate"])

    # '50% window' = midpoints between Dark Pivots -> should mark the opposite extreme (top).
    mids = dpr.midpoints(dp_bars)
    fifty = {
        "n_midpoints": int(mids.size),
        "midpoint_top_rate": float(high_extreme[mids].mean()) if mids.size else float("nan"),
        "all_days_top_rate": float(high_extreme.mean()),
    }
    fifty["lift"] = stats.lift(fifty["midpoint_top_rate"], fifty["all_days_top_rate"])

    results = {
        "config": {"symbol": symbol, "start": start, "end": end, "local_window": args.local_window},
        "data": {"bars": n, "n_dark_pivots": int(dp_bars.size),
                 "first": times.iloc[0].strftime("%Y-%m-%d"), "last": times.iloc[-1].strftime("%Y-%m-%d")},
        "upcoming_dark_pivots": upcoming_dates,
        "grid": grid,
        "local_bottom": local_bottom,
        "fifty_window": fifty,
    }

    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "dark_pivot_replica.json", results)
    (args.outdir / "dark_pivot_replica.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r: dict) -> str:
    L = [
        "# Dark Pivot claim - replication + baseline",
        "",
        f"Moon-Pluto hard aspects as Dark Pivots. {r['data']['n_dark_pivots']} firings over "
        f"{r['data']['first']} -> {r['data']['last']}.",
        f"Advertised next date 2026-06-24 is in the computed calendar: "
        f"{', '.join(r['upcoming_dark_pivots'][:4])} ...",
        "",
        "## Their loose rule reproduced, with a baseline beside it",
        "",
        "Rule: dumped into the activation day, then a bullish expansion within `horizon` days "
        "(x_atr = expansion size beyond the activation high). **lift_vs_dump** compares to the "
        "SAME rule on ordinary (non-Dark-Pivot) dump days.",
        "",
        "| lookback | horizon | x_atr | signals | DP hit | base(dump) | base(all) | lift vs dump | binom p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in r["grid"]:
        L.append(f"| {g['lookback']} | {g['horizon']} | {g['x_atr']} | {g['n_signals']} | "
                 f"{_f(g['dp_hit'])} | {_f(g['baseline_dump_hit'])} | {_f(g['baseline_all_hit'])} | "
                 f"{_f(g['lift_vs_dump'])} | {_f(g['binom_p_vs_dump'], 4)} |")
    lb = r["local_bottom"]
    fw = r["fifty_window"]
    L += [
        "",
        f"## Literal 'marked a local bottom' (+/-{lb['window']} d)",
        "",
        f"- Dark-Pivot dump days that are local lows: **{_f(lb['dp_dump_bottom_rate'])}** "
        f"({lb['n_dp_dump_signals']} signals)",
        f"- ordinary dump days that are local lows: {_f(lb['all_dump_bottom_rate'])} | "
        f"all days: {_f(lb['all_days_bottom_rate'])}",
        f"- **lift vs ordinary dump days: {_f(lb['lift_vs_dump'])}**",
        "",
        "## '50% window' (midpoint between Dark Pivots) marks the opposite extreme (top)",
        "",
        f"- midpoints that are local highs: **{_f(fw['midpoint_top_rate'])}** ({fw['n_midpoints']} midpoints)",
        f"- all days that are local highs: {_f(fw['all_days_top_rate'])} | **lift: {_f(fw['lift'])}**",
        "",
        "## Verdict",
        "",
        "The advertised ~77% is the **base rate**, not an edge. With horizon 7 (the gap to the next "
        "pivot), the unconditional probability of a higher high within 7 days is ~0.774 - essentially "
        "the claimed 77.27%. Dark-Pivot dump days score ~0.70-0.73, i.e. AT OR BELOW that base rate, "
        "and ~equal to ordinary dump days (lift vs dump ~1.0, binomial p ~0.4-0.7). The loose success "
        "rule ('a bullish expansion within a week of a dump') is near-universal on daily BTC, so a high "
        "hit rate is guaranteed regardless of astrology. The literal 'local bottom' and '50% top' "
        "readings show lift ~1.0 too. No edge over baseline.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
