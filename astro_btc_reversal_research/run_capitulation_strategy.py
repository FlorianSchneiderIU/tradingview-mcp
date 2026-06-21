"""Full Capitulation strategy: LONGS + SHORTS, fixed-RR vs scaled exit, across the basket.

Long  = deep-sweep spring   + funding_z <= -thr  (shorts crowded -> contrarian long).
Short = deep-sweep upthrust  + funding_z >= +thr  (longs crowded  -> contrarian short).
Exits = fixed 30R  vs  scaled 25%@4R / 50%@12R / 25%@30R (stop->BE after first partial).

Outputs (reports/): capitulation_strategy.json and capitulation_strategy.md
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
    basket, capitulation as cap, data, exits, ltf_structure as lts, reuse, report, weekly_timing as wt,
)


def _agg(trades, n_weeks, holdout_start):
    s = reuse.trade_summary(trades)
    s["trades_per_week"] = float(s["trades"] / n_weeks) if n_weeks else float("nan")
    hold = [t for t in trades if pd.Timestamp(t["entry_time"]) >= holdout_start]
    dev = [t for t in trades if pd.Timestamp(t["entry_time"]) < holdout_start]
    s["holdout_avg_r"] = float(np.mean([t["result_r"] for t in hold])) if hold else float("nan")
    s["holdout_trades"] = len(hold)
    s["dev_avg_r"] = float(np.mean([t["result_r"] for t in dev])) if dev else float("nan")
    by = {}
    for t in trades:
        by.setdefault(pd.Timestamp(t["entry_time"]).year, []).append(t["result_r"])
    s["by_year"] = {str(y): [len(v), round(float(np.mean(v)), 2)] for y, v in sorted(by.items())}
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="Capitulation long+short, fixed vs scaled exit.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--fz", type=float, default=1.0, help="|funding_z| threshold")
    p.add_argument("--rr", type=int, default=30)
    p.add_argument("--interval", default=None, help="override LTF interval (e.g. 5m, 15m)")
    p.add_argument("--spring-lookback", type=int, default=None, help="deep-sweep bars (15d: 4320@5m, 1440@15m)")
    p.add_argument("--max-hold", type=int, default=None, help="max hold bars (14d: 4032@5m, 1344@15m)")
    p.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "reports")
    args = p.parse_args()

    cfg = data.load_config(args.config)
    b = dict(cfg["basket_spring"])
    if args.interval:
        b["interval"] = args.interval
    if args.spring_lookback:
        b["spring_lookback"] = args.spring_lookback
    if args.max_hold:
        b["max_hold_bars"] = args.max_hold
    lookback = b["spring_lookback"]
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")
    span_weeks = (reuse.parse_utc_datetime(b["end"]) - reuse.parse_utc_datetime(b["start"])).days / 7.0
    tps, fracs = (4.0, 12.0, float(args.rr)), (0.25, 0.50, 0.25)

    books = {k: [] for k in ("long_fixed", "long_scaled", "short_fixed", "short_scaled")}
    for sym in basket.BASKET:
        try:
            frame = basket.load_symbol(sym, b["interval"], b["start"], b["end"])
            funding = cap.fetch_funding(sym, b["start"], None)
            oi = cap.fetch_oi(sym, "4h", b["start"], None)
        except Exception:
            continue
        if len(frame) < lookback + 100:
            continue
        _, frac = wt.time_of_week(frame)
        early = frac < b["week_frac_max"]
        feats = cap.capitulation_features(frame, funding, oi)
        fz = feats.get("funding_z", pd.Series(np.nan, index=frame.index)).to_numpy()
        if "funding_z" not in feats:
            continue

        spring = lts.spring_long_signals(frame, lookback, b["wick_frac"], b["close_pos"]) & early
        upthr = lts.upthrust_short_signals(frame, lookback, b["wick_frac"], b["close_pos"]) & early
        long_mask = spring & np.where(np.isfinite(fz), fz <= -args.fz, False)
        short_mask = upthr & np.where(np.isfinite(fz), fz >= args.fz, False)

        kw = dict(max_hold_bars=b["max_hold_bars"], stop_buffer_atr=b["stop_buffer_atr"],
                  cost_bps_round_trip=b["cost_bps_round_trip"])
        for tr in exits.backtest_fixed(frame, long_mask, "long", args.rr, **kw):
            tr["symbol"] = sym; books["long_fixed"].append(tr)
        for tr in exits.backtest_scaled(frame, long_mask, "long", tps, fracs, **kw):
            tr["symbol"] = sym; books["long_scaled"].append(tr)
        for tr in exits.backtest_fixed(frame, short_mask, "short", args.rr, **kw):
            tr["symbol"] = sym; books["short_fixed"].append(tr)
        for tr in exits.backtest_scaled(frame, short_mask, "short", tps, fracs, **kw):
            tr["symbol"] = sym; books["short_scaled"].append(tr)

    agg = {name: _agg(tr, span_weeks, holdout_start) for name, tr in books.items()}
    # Combined books (long+short) for total cadence + blended expectancy.
    agg["combined_fixed"] = _agg(books["long_fixed"] + books["short_fixed"], span_weeks, holdout_start)
    agg["combined_scaled"] = _agg(books["long_scaled"] + books["short_scaled"], span_weeks, holdout_start)

    results = {
        "config": {"interval": b["interval"], "start": b["start"], "end": b["end"],
                   "spring_lookback": lookback, "week_frac_max": b["week_frac_max"],
                   "funding_z_thr": args.fz, "rr": args.rr, "tps": list(tps), "fracs": list(fracs),
                   "cost_bps_round_trip": b["cost_bps_round_trip"], "span_weeks": round(span_weeks, 1),
                   "holdout_start": str(holdout_start)},
        "aggregate": agg,
    }
    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "capitulation_strategy.json", results)
    (args.outdir / "capitulation_strategy.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r: dict) -> str:
    c = r["config"]
    L = [
        "# Capitulation Strategy: Longs + Shorts, Fixed vs Scaled Exit",
        "",
        f"{c['interval']} | {c['start']} -> {c['end']} (~{c['span_weeks']} wk) | sweep {c['spring_lookback']} "
        f"bars | early-week | |funding_z|>={c['funding_z_thr']} | costs {c['cost_bps_round_trip']} bps RT.",
        f"Long = spring + funding<=-{c['funding_z_thr']}; Short = upthrust + funding>=+{c['funding_z_thr']}. "
        f"Fixed = {c['rr']}R; Scaled = 25%@{c['tps'][0]:g}R / 50%@{c['tps'][1]:g}R / 25%@{c['tps'][2]:g}R, "
        "stop->BE after first partial.",
        "",
        "| Book | Trades | Trades/wk | Win % | Avg R | Net R | PF | MaxDD R | Dev avg R | Holdout avg R (n) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = ["long_fixed", "long_scaled", "short_fixed", "short_scaled", "combined_fixed", "combined_scaled"]
    for name in order:
        a = r["aggregate"][name]
        L.append(f"| {name} | {a['trades']} | {_f(a['trades_per_week'],2)} | {_f(a['win_rate']*100,1)} | "
                 f"{_f(a['avg_r'])} | {_f(a['net_r'],1)} | {_f(a['profit_factor'])} | {_f(a['max_drawdown_r'],1)} | "
                 f"{_f(a['dev_avg_r'])} | {_f(a['holdout_avg_r'])} ({a['holdout_trades']}) |")
    L += ["", "## Per-year (combined_scaled)", "",
          ", ".join(f"{y}: {v[1]}R/{v[0]}t" for y, v in r["aggregate"]["combined_scaled"]["by_year"].items()),
          "", "## Reading guide", "",
          "Compare scaled vs fixed: scaled should LIFT win rate and SHRINK MaxDD (banking 4R partials, "
          "BE after TP1), trading some avg R for consistency. Do SHORTS add positive, holdout-positive "
          "trades? Combined cadence ~ longs + shorts per week. Watch dev vs holdout and the bear year(s).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
