"""Multi-symbol deep-sweep-spring basket backtest (breadth -> ~1 setup/week?).

Applies the validated edge (15d/30d deep-sweep 5m/15m Wyckoff spring, tiny stop,
high-RR target) across a basket of liquid perps and aggregates: trade frequency
(trades/week), expectancy net of costs, holdout, per-year, per-symbol, and the
time-clustering of setups.

Outputs (reports/): basket_spring.json and basket_spring.md
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
    basket, data, ltf_structure as lts, reuse, strategy, report, weekly_timing as wt,
)


def _summ(trades):
    return reuse.trade_summary(trades)


def _agg(trades, n_weeks, holdout_start):
    s = _summ(trades)
    s["trades_per_week"] = float(s["trades"] / n_weeks) if n_weeks else float("nan")
    hold = [t for t in trades if pd.Timestamp(t["entry_time"]) >= holdout_start]
    s["holdout_avg_r"] = float(np.mean([t["result_r"] for t in hold])) if hold else float("nan")
    s["holdout_trades"] = len(hold)
    s["reach_20r"] = float(np.mean([t["mfe_r"] >= 20 for t in trades])) if trades else float("nan")
    s["reach_30r"] = float(np.mean([t["mfe_r"] >= 30 for t in trades])) if trades else float("nan")
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-symbol deep-sweep-spring basket.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--spring-lookback", type=int, default=None)
    p.add_argument("--week-frac-max", type=float, default=None)
    p.add_argument("--no-week-gate", action="store_true")
    p.add_argument("--symbols", default=None, help="comma list override")
    p.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "reports")
    args = p.parse_args()

    cfg = data.load_config(args.config)
    b = cfg["basket_spring"]
    lookback = args.spring_lookback or b["spring_lookback"]
    week_frac_max = None if args.no_week_gate else (args.week_frac_max or b.get("week_frac_max"))
    symbols = args.symbols.split(",") if args.symbols else basket.BASKET
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")
    span_weeks = (reuse.parse_utc_datetime(b["end"]) - reuse.parse_utc_datetime(b["start"])).days / 7.0

    all_trades = {rr: [] for rr in b["rr_values"]}
    exc_trades = []
    per_symbol = {}
    loaded = 0
    for sym in symbols:
        try:
            frame = basket.load_symbol(sym, b["interval"], b["start"], b["end"])
        except Exception as ex:
            per_symbol[sym] = {"error": str(ex)}
            continue
        if len(frame) < lookback + 100:
            per_symbol[sym] = {"skipped_bars": len(frame)}
            continue
        loaded += 1
        spring = lts.spring_long_signals(frame, lookback, b["wick_frac"], b["close_pos"])
        if week_frac_max is not None:
            _, frac = wt.time_of_week(frame)
            spring = spring & (frac < week_frac_max)
        # Excursion + fixed-RR (tag trades with symbol).
        exc = strategy.backtest_long(frame, spring, b["excursion_rr"], b["max_hold_bars"],
                                     b["stop_buffer_atr"], b["cost_bps_round_trip"])
        for t in exc:
            t["symbol"] = sym
        exc_trades.extend(exc)
        sym_rr = {}
        for rr in b["rr_values"]:
            tr = strategy.backtest_long(frame, spring, rr, b["max_hold_bars"],
                                        b["stop_buffer_atr"], b["cost_bps_round_trip"])
            for t in tr:
                t["symbol"] = sym
            all_trades[rr].extend(tr)
            sym_rr[f"rr_{rr:g}"] = _summ(tr)
        per_symbol[sym] = {"bars": len(frame), "n_signals": int(spring.sum()), **sym_rr}

    # Aggregate (use excursion trades for the R-distribution / frequency).
    agg = {f"rr_{rr:g}": _agg(all_trades[rr], span_weeks, holdout_start) for rr in b["rr_values"]}
    # Setup clustering: trades per calendar quarter (from excursion book).
    if exc_trades:
        q = pd.Series([pd.Timestamp(t["entry_time"]).to_period("Q").strftime("%YQ%q") for t in exc_trades])
        per_quarter = {k: int(v) for k, v in q.value_counts().sort_index().items()}
    else:
        per_quarter = {}

    results = {
        "config": {"interval": b["interval"], "start": b["start"], "end": b["end"],
                   "spring_lookback": lookback, "week_frac_max": week_frac_max,
                   "stop_buffer_atr": b["stop_buffer_atr"], "max_hold_bars": b["max_hold_bars"],
                   "cost_bps_round_trip": b["cost_bps_round_trip"], "rr_values": b["rr_values"],
                   "holdout_start": str(holdout_start), "span_weeks": round(span_weeks, 1)},
        "n_symbols_loaded": loaded, "n_symbols_requested": len(symbols),
        "aggregate": agg, "per_symbol": per_symbol, "trades_per_quarter": per_quarter,
        "n_excursion_signals": len(exc_trades),
    }
    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "basket_spring.json", results)
    (args.outdir / "basket_spring.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r: dict) -> str:
    c = r["config"]
    L = [
        "# Multi-Symbol Deep-Sweep-Spring Basket",
        "",
        f"{r['n_symbols_loaded']}/{r['n_symbols_requested']} symbols | {c['interval']} | "
        f"{c['start']} -> {c['end']} (~{c['span_weeks']} weeks) | sweep lookback {c['spring_lookback']} "
        f"bars | week gate {c['week_frac_max']} | costs {c['cost_bps_round_trip']} bps RT.",
        "",
        "## Aggregate (all symbols pooled)",
        "",
        "| Target | Trades | Trades/week | Win % | Avg R | Net R | PF | MaxDD R | reach20R | Holdout avg R (n) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rr in c["rr_values"]:
        a = r["aggregate"][f"rr_{rr:g}"]
        L.append(f"| {rr}R | {a['trades']} | {_f(a['trades_per_week'],2)} | {_f(a['win_rate']*100,1)} | "
                 f"{_f(a['avg_r'])} | {_f(a['net_r'],1)} | {_f(a['profit_factor'])} | "
                 f"{_f(a['max_drawdown_r'],1)} | {_f(a['reach_20r'])} | "
                 f"{_f(a['holdout_avg_r'])} ({a['holdout_trades']}) |")
    L += ["", "## Per-symbol (target 20R)", "",
          "| Symbol | bars | signals | trades | win % | avg R | net R | PF |",
          "|---|---|---|---|---|---|---|---|"]
    for sym, d in r["per_symbol"].items():
        if "rr_20" not in d:
            L.append(f"| {sym} | - | - | (skipped/err) | | | | |")
            continue
        s = d["rr_20"]
        L.append(f"| {sym} | {d['bars']} | {d['n_signals']} | {s['trades']} | {_f(s['win_rate']*100,1)} | "
                 f"{_f(s['avg_r'])} | {_f(s['net_r'],1)} | {_f(s['profit_factor'])} |")
    L += ["", "## Setup clustering (excursion signals per quarter)", "",
          ", ".join(f"{k}:{v}" for k, v in r["trades_per_quarter"].items()), "",
          "## Reading guide", "",
          "Goal: ~1 trade/week (trades/week ~ 1) with avg R > 0 net of costs AND positive holdout. "
          "Watch the per-quarter clustering - basket deep-sweeps bunch during market-wide crashes, so "
          "'trades/week' is an average, not an even cadence. A few symbols may carry the edge; check "
          "the per-symbol table for breadth vs concentration.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
