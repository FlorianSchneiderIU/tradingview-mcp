"""Multi-TF structure edge: 5m Wyckoff springs around weekly lows (1:20-1:30 RR).

Tests whether a 5m spring at/near a weekly low runs to 20-30R often enough to be
net profitable (tiny LTF stop, weekly-swing target), and whether that opportunity
concentrates at weekly lows (learnable) vs random springs.

Outputs (reports/): ltf_structure.json and ltf_structure.md
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

from astro_reversal import data, ltf_structure as lts, report, strategy  # noqa: E402


def _summ(trades):
    return reuse_summary(trades)


def reuse_summary(trades):
    from astro_reversal import reuse
    s = reuse.trade_summary(trades)
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="5m Wyckoff spring structure edge around weekly lows.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--spring-lookback", type=int, default=None,
                   help="override: bars defining the swept low (e.g. 2016=7d, 8640=30d on 5m)")
    p.add_argument("--week-frac-max", type=float, default=None,
                   help="gate springs to the first fraction of the week (e.g. 0.4 = Mon-Wed)")
    p.add_argument("--dows", default=None,
                   help="gate springs to these days of week, e.g. '0,4' for Mon+Fri")
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    s = dict(cfg["ltf_structure"])
    if args.spring_lookback is not None:
        s["spring_lookback"] = args.spring_lookback
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")

    daily = data.load_ohlcv(symbol, s["htf_interval"], start, end, base_interval=cfg["data"]["base_interval"])
    # 5m loads directly from its own cache (cannot resample up from 15m).
    ltf = data.load_ohlcv(symbol, s["ltf_interval"], start, end, base_interval=s["ltf_interval"])

    _, lows, _ = lts.htf_pivots(daily, s["htf_threshold_atr"])
    spring = lts.spring_long_signals(ltf, s["spring_lookback"], s["wick_frac"], s["close_pos"])

    # Optional time-of-week gate (the seasonal window from run_weekly_timing.py).
    from astro_reversal import weekly_timing as wt
    dow_arr, frac_arr = wt.time_of_week(ltf)
    week_gate = np.ones(len(ltf), dtype=bool)
    if args.week_frac_max is not None:
        week_gate &= frac_arr < args.week_frac_max
    if args.dows is not None:
        keep = {int(x) for x in args.dows.split(",") if x.strip() != ""}
        week_gate &= np.isin(dow_arr, list(keep))
    spring = spring & week_gate

    near_low = lts.near_times_mask(ltf, [d["time"] for d in lows], s["near_low_window_bars"])

    masks = {
        "all_springs": spring,
        "near_weekly_low": spring & near_low,
        "away_from_low": spring & ~near_low,
    }

    def bt(mask, rr):
        return strategy.backtest_long(ltf, mask, rr, s["max_hold_bars"],
                                      s["stop_buffer_atr"], s["cost_bps_round_trip"])

    out_books = {}
    for name, mask in masks.items():
        exc = bt(mask, s["excursion_rr"])
        rr_rows = {}
        for rr in s["rr_values"]:
            trades = bt(mask, rr)
            summ = reuse_summary(trades)
            hold = [t for t in trades if pd.Timestamp(t["entry_time"]) >= holdout_start]
            dev = [t for t in trades if pd.Timestamp(t["entry_time"]) < holdout_start]
            rr_rows[f"rr_{rr:g}"] = {
                "all": summ,
                "dev": reuse_summary(dev),
                "holdout": reuse_summary(hold),
                "by_year": {str(y): reuse_summary(ts)
                            for y, ts in strategy.split_trades_by_year(trades).items()},
            }
        out_books[name] = {
            "n_signals": int(mask.sum()),
            "r_distribution": strategy.r_distribution(exc, levels=(5, 10, 20, 30)),
            "avg_mfe_r": float(np.mean([t["mfe_r"] for t in exc])) if exc else float("nan"),
            "rr": rr_rows,
        }

    results = {
        "config": {"symbol": symbol, "start": start, "end": end, "ltf": s["ltf_interval"],
                   "htf": s["htf_interval"], "htf_threshold_atr": s["htf_threshold_atr"],
                   "spring_lookback": s["spring_lookback"], "stop_buffer_atr": s["stop_buffer_atr"],
                   "max_hold_bars": s["max_hold_bars"], "cost_bps_round_trip": s["cost_bps_round_trip"],
                   "rr_values": s["rr_values"], "near_low_window_bars": s["near_low_window_bars"],
                   "spring_lookback": s["spring_lookback"],
                   "week_frac_max": args.week_frac_max, "dows": args.dows,
                   "holdout_start": str(holdout_start)},
        "data": {"ltf_bars": len(ltf), "daily_bars": len(daily), "n_weekly_lows": len(lows),
                 "n_springs": int(spring.sum())},
        "books": out_books,
    }

    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "ltf_structure.json", results)
    (args.outdir / "ltf_structure.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r: dict) -> str:
    c = r["config"]
    d = r["data"]
    L = [
        "# Multi-TF Structure Edge: 5m Wyckoff Springs at Weekly Lows",
        "",
        f"**{c['symbol']}** LTF {c['ltf']} / HTF {c['htf']} | {c['start']} -> {c['end']}",
        f"Weekly lows (daily zigzag >= {c['htf_threshold_atr']} ATR): {d['n_weekly_lows']} | "
        f"5m springs: {d['n_springs']} | stop {c['stop_buffer_atr']} ATR below spring | "
        f"costs {c['cost_bps_round_trip']} bps RT | 'near low' = +/-{c['near_low_window_bars']} 5m bars.",
        "",
        "## Max-R available after a spring (excursion pass)",
        "",
        "| Book | signals | avg MFE R | reach 5R | 10R | 20R | 30R |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, b in r["books"].items():
        rd = b["r_distribution"]
        L.append(f"| {name} | {b['n_signals']} | {_f(b['avg_mfe_r'])} | {_f(rd['reach_5r'])} | "
                 f"{_f(rd['reach_10r'])} | {_f(rd['reach_20r'])} | {_f(rd['reach_30r'])} |")
    L += ["", "## Fixed-RR P&L net of costs (expectancy = avg R)", ""]
    for rr in c["rr_values"]:
        key = f"rr_{rr:g}"
        L += [f"### Target {rr}R", "",
              "| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R | Holdout avg R |",
              "|---|---|---|---|---|---|---|---|"]
        for name, b in r["books"].items():
            e = b["rr"][key]
            a = e["all"]
            L.append(f"| {name} | {a['trades']} | {_f(a['win_rate'])} | {_f(a['avg_r'])} | "
                     f"{_f(a['net_r'], 1)} | {_f(a['profit_factor'])} | {_f(a['max_drawdown_r'], 1)} | "
                     f"{_f(e['holdout']['avg_r'])} |")
        L.append("")
    L += [
        "## Reading guide",
        "",
        "At 20R breakeven win rate is ~1/(20+1) ~ 4.8% (before costs); the tight 5m stop makes costs a "
        "real fraction of risk, so read **avg R** (net). The decisive question: does **near_weekly_low** "
        "beat **away_from_low** on reach-20R and avg R? If yes, the asymmetric opportunity concentrates "
        "at weekly lows -> a learnable gate (predict the weekly-low zone on the HTF, confirm with the 5m "
        "spring). If near and away look the same, the spring alone has no structural edge.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
