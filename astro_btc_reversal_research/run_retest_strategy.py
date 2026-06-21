"""Retest-entry strategy backtest: long on the mid-week retest of the weekly low.

Compares the retest entry (with/without the early-week gate) against the prior
'first deep-sweep spring' entry and ungated springs, at 20R/30R, net of costs,
with per-year + holdout breakdown and the R-distribution.

Outputs (reports/): retest_strategy.json and retest_strategy.md
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
    data, ltf_structure as lts, retest_strategy as rts, reuse, strategy, report, weekly_timing as wt,
)


def _summ(trades):
    return reuse.trade_summary(trades)


def main() -> int:
    p = argparse.ArgumentParser(description="Weekly-low retest entry backtest.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--ltf", default=None, help="override LTF interval (e.g. 1m, 5m)")
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    s = cfg["retest_strategy"]
    ls = cfg["ltf_structure"]
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    ltf_int = args.ltf or s["ltf_interval"]
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")

    ltf = data.load_ohlcv(symbol, ltf_int, start, end, base_interval=ltf_int)
    n = len(ltf)

    # Local spring (for the retest) and the deep-sweep spring (prior best entry).
    sp_local = rts.spring_mask(ltf, s["spring_lookback"], s["wick_frac"], s["close_pos"])
    sp_deep = lts.spring_long_signals(ltf, ls["spring_lookback"], ls["wick_frac"], ls["close_pos"])
    _, frac = wt.time_of_week(ltf)
    early = frac < s["est_max_frac"]

    masks = {
        "retest_earlywk": rts.retest_entries(ltf, sp_local, s["move_away_pct"], s["retest_tol_pct"],
                                             early_week_only=True, est_max_frac=s["est_max_frac"]),
        "retest_anytime": rts.retest_entries(ltf, sp_local, s["move_away_pct"], s["retest_tol_pct"],
                                             early_week_only=False),
        "first_deep_spring_earlywk": sp_deep & early,
        "ungated_local_spring": sp_local,
    }

    def bt(mask, rr):
        return strategy.backtest_long(ltf, mask, rr, s["max_hold_bars"],
                                      s["stop_buffer_atr"], s["cost_bps_round_trip"])

    books = {}
    for name, mask in masks.items():
        exc = bt(mask, s["excursion_rr"])
        rr_rows = {}
        for rr in s["rr_values"]:
            trades = bt(mask, rr)
            hold = [t for t in trades if pd.Timestamp(t["entry_time"]) >= holdout_start]
            rr_rows[f"rr_{rr:g}"] = {
                "all": _summ(trades),
                "holdout": _summ(hold),
                "by_year": {str(y): _summ(ts) for y, ts in strategy.split_trades_by_year(trades).items()},
            }
        books[name] = {
            "n_signals": int(mask.sum()),
            "r_distribution": strategy.r_distribution(exc, levels=(5, 10, 20, 30)),
            "avg_mfe_r": float(np.mean([t["mfe_r"] for t in exc])) if exc else float("nan"),
            "rr": rr_rows,
        }

    results = {
        "config": {"symbol": symbol, "start": start, "end": end, "ltf": ltf_int,
                   "spring_lookback": s["spring_lookback"], "move_away_pct": s["move_away_pct"],
                   "retest_tol_pct": s["retest_tol_pct"], "est_max_frac": s["est_max_frac"],
                   "stop_buffer_atr": s["stop_buffer_atr"], "max_hold_bars": s["max_hold_bars"],
                   "cost_bps_round_trip": s["cost_bps_round_trip"], "rr_values": s["rr_values"],
                   "holdout_start": str(holdout_start)},
        "data": {"ltf_bars": n},
        "books": books,
    }
    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "retest_strategy.json", results)
    (args.outdir / "retest_strategy.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r: dict) -> str:
    c = r["config"]
    L = [
        "# Weekly-Low Retest Entry Strategy",
        "",
        f"**{c['symbol']}** LTF {c['ltf']} | {c['start']} -> {c['end']} ({r['data']['ltf_bars']} bars)",
        f"Establish weekly low after a {c['move_away_pct']*100:.1f}% rally; enter on a spring retesting "
        f"it within {c['retest_tol_pct']*100:.1f}%; stop {c['stop_buffer_atr']} ATR below; costs "
        f"{c['cost_bps_round_trip']} bps RT.",
        "",
        "## R available after entry (excursion pass)",
        "",
        "| Book | signals | avg MFE R | 5R | 10R | 20R | 30R |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, b in r["books"].items():
        rd = b["r_distribution"]
        L.append(f"| {name} | {b['n_signals']} | {_f(b['avg_mfe_r'])} | {_f(rd['reach_5r'])} | "
                 f"{_f(rd['reach_10r'])} | {_f(rd['reach_20r'])} | {_f(rd['reach_30r'])} |")
    for rr in c["rr_values"]:
        key = f"rr_{rr:g}"
        L += ["", f"## Target {rr}R (net of costs)", "",
              "| Book | Trades | Win % | Avg R | Net R | PF | MaxDD R | Holdout avg R (n) |",
              "|---|---|---|---|---|---|---|---|"]
        for name, b in r["books"].items():
            e = b["rr"][key]
            a, h = e["all"], e["holdout"]
            L.append(f"| {name} | {a['trades']} | {_f(a['win_rate']*100,1)} | {_f(a['avg_r'])} | "
                     f"{_f(a['net_r'],1)} | {_f(a['profit_factor'])} | {_f(a['max_drawdown_r'],1)} | "
                     f"{_f(h['avg_r'])} ({h['trades']}) |")
        # Per-year for the headline retest_earlywk book.
        by = r["books"]["retest_earlywk"]["rr"][key]["by_year"]
        L += ["", "retest_earlywk per-year: " + ", ".join(
            f"{y} {_f(v['avg_r'],2)}R/{v['trades']}t" for y, v in sorted(by.items()))]
    L += [
        "",
        "## Reading guide",
        "",
        "Does **retest_earlywk** beat **first_deep_spring_earlywk** on avg R and holdout? The retest "
        "should give a tighter stop (entry at a tested level) and thus higher R per win. Watch trade "
        "count - the retest is rarer; weigh expectancy against sample size and MaxDD.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
