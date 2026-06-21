"""Capitulation-filtered basket: keep only deep-sweep springs that coincide with a
funding flush and/or an open-interest drop (forced liquidations).

Goal: turn the ~4 raw setups/week into ~1/week of high-conviction capitulation
bottoms with materially better win rate / expectancy, holding on the holdout.

Outputs (reports/): basket_capitulation.json and basket_capitulation.md
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
    basket, capitulation as cap, data, ltf_structure as lts, reuse, strategy, report, weekly_timing as wt,
)


def _agg(trades, n_weeks, holdout_start):
    s = reuse.trade_summary(trades)
    s["trades_per_week"] = float(s["trades"] / n_weeks) if n_weeks else float("nan")
    hold = [t for t in trades if pd.Timestamp(t["entry_time"]) >= holdout_start]
    s["holdout_avg_r"] = float(np.mean([t["result_r"] for t in hold])) if hold else float("nan")
    s["holdout_trades"] = len(hold)
    s["reach_20r"] = float(np.mean([t["mfe_r"] >= 20 for t in trades])) if trades else float("nan")
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="Capitulation-filtered basket spring.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--spring-lookback", type=int, default=None)
    p.add_argument("--rr", type=int, default=20)
    p.add_argument("--fz", type=float, default=-1.0, help="funding_z threshold (<=)")
    p.add_argument("--oiz", type=float, default=-1.0, help="oi_z threshold (<=, OI flush)")
    p.add_argument("--no-week-gate", action="store_true")
    p.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "reports")
    args = p.parse_args()

    cfg = data.load_config(args.config)
    b = cfg["basket_spring"]
    lookback = args.spring_lookback or b["spring_lookback"]
    rr = args.rr
    week_frac_max = None if args.no_week_gate else b.get("week_frac_max")
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")
    span_weeks = (reuse.parse_utc_datetime(b["end"]) - reuse.parse_utc_datetime(b["start"])).days / 7.0

    books = {k: [] for k in ("all", "cap_funding", "cap_oiflush", "cap_either", "cap_both")}
    coverage = {"with_funding": 0, "with_oi": 0}
    for sym in basket.BASKET:
        try:
            frame = basket.load_symbol(sym, b["interval"], b["start"], b["end"])
            funding = cap.fetch_funding(sym, b["start"], None)
            oi = cap.fetch_oi(sym, "4h", b["start"], None)
        except Exception:
            continue
        if len(frame) < lookback + 100:
            continue
        spring = lts.spring_long_signals(frame, lookback, b["wick_frac"], b["close_pos"])
        if week_frac_max is not None:
            _, frac = wt.time_of_week(frame)
            spring = spring & (frac < week_frac_max)
        feats = cap.capitulation_features(frame, funding, oi)
        fz = feats.get("funding_z", pd.Series(np.nan, index=frame.index)).to_numpy()
        oiz = feats.get("oi_z", pd.Series(np.nan, index=frame.index)).to_numpy()
        if "funding_z" in feats:
            coverage["with_funding"] += 1
        if "oi_z" in feats:
            coverage["with_oi"] += 1

        gate_f = np.where(np.isfinite(fz), fz <= args.fz, False)
        gate_o = np.where(np.isfinite(oiz), oiz <= args.oiz, False)
        masks = {
            "all": spring,
            "cap_funding": spring & gate_f,
            "cap_oiflush": spring & gate_o,
            "cap_either": spring & (gate_f | gate_o),
            "cap_both": spring & gate_f & gate_o,
        }
        for name, m in masks.items():
            tr = strategy.backtest_long(frame, m, rr, b["max_hold_bars"],
                                        b["stop_buffer_atr"], b["cost_bps_round_trip"])
            for t in tr:
                t["symbol"] = sym
            books[name].extend(tr)

    agg = {name: _agg(tr, span_weeks, holdout_start) for name, tr in books.items()}
    results = {
        "config": {"interval": b["interval"], "start": b["start"], "end": b["end"], "rr": rr,
                   "spring_lookback": lookback, "week_frac_max": week_frac_max,
                   "funding_z_thr": args.fz, "oi_z_thr": args.oiz,
                   "cost_bps_round_trip": b["cost_bps_round_trip"], "span_weeks": round(span_weeks, 1),
                   "holdout_start": str(holdout_start)},
        "coverage": coverage, "aggregate": agg,
    }
    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "basket_capitulation.json", results)
    (args.outdir / "basket_capitulation.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r: dict) -> str:
    c = r["config"]
    L = [
        f"# Capitulation-Filtered Basket Spring (target {c['rr']}R)",
        "",
        f"{c['interval']} | {c['start']} -> {c['end']} (~{c['span_weeks']} wk) | sweep {c['spring_lookback']} "
        f"bars | week gate {c['week_frac_max']} | funding_z<={c['funding_z_thr']} | oi_z<={c['oi_z_thr']} | "
        f"coverage funding {r['coverage']['with_funding']} / OI {r['coverage']['with_oi']} symbols.",
        "",
        "Capitulation = deep-sweep spring coinciding with unusually negative funding (shorts crowded) "
        "and/or an OI flush (forced long liquidations).",
        "",
        "| Book | Trades | Trades/wk | Win % | Avg R | Net R | PF | MaxDD R | reach20R | Holdout avg R (n) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, a in r["aggregate"].items():
        L.append(f"| {name} | {a['trades']} | {_f(a['trades_per_week'],2)} | {_f(a['win_rate']*100,1)} | "
                 f"{_f(a['avg_r'])} | {_f(a['net_r'],1)} | {_f(a['profit_factor'])} | {_f(a['max_drawdown_r'],1)} | "
                 f"{_f(a['reach_20r'])} | {_f(a['holdout_avg_r'])} ({a['holdout_trades']}) |")
    L += [
        "",
        "## Reading guide",
        "",
        "Does a capitulation book beat **all** on avg R / reach20R / holdout while keeping trades/week "
        "near 1? That would be the high-conviction weekly setup: deep-sweep spring + funding/OI flush. "
        "If capitulation books are no better, the flush adds nothing.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
