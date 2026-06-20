"""High-RR reversal strategy backtest (Milestone 4) - CLI.

HTF daily dump -> bottom zone; LTF (1h) sweep+reclaim -> long with a tight stop
below the swept low and a high-RR target. Reports the full R-distribution, fixed-RR
P&L (net of costs), gated-vs-ungated contribution, per-year + holdout stability, and
an optional Moon-Pluto overlay (expected to add nothing).

Outputs (reports/): strategy.json and strategy.md
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
    data,
    ephemeris_events,
    report,
    reuse,
    strategy,
)


def _load_ltf(symbol, interval, start, end, base_interval):
    ltf = data.load_ohlcv(symbol, interval, start, end, base_interval=base_interval)
    ltf["prev_range_high"] = ltf["high"].shift(1).rolling(12, min_periods=12).max()
    ltf["prev_range_low"] = ltf["low"].shift(1).rolling(12, min_periods=12).min()
    return ltf


def _summ(trades):
    s = reuse.trade_summary(trades)
    s["expectancy_r"] = s.get("avg_r")
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="High-RR reversal strategy backtest.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--astro-overlay", action="store_true",
                   help="further restrict longs to within +/-1 day of a Moon-Pluto hard aspect")
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    sc = cfg["strategy"]
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    base_interval = cfg["data"]["base_interval"]
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")

    daily = data.load_ohlcv(symbol, sc["htf_interval"], start, end, base_interval=base_interval)
    ltf = _load_ltf(symbol, sc["ltf_interval"], start, end, base_interval)

    _, active_day = strategy.daily_dump_gate(
        daily, sc["htf_dump_lookback"], sc["htf_dump_threshold_atr"], sc["hold_days"])
    active = strategy.ltf_active_mask(daily, ltf, active_day)
    signal = strategy.ltf_long_signals(
        ltf,
        lookback=int(sc.get("sweep_lookback", 12)),
        require_displacement=bool(sc.get("require_displacement", False)),
        disp_body_atr=float(sc.get("disp_body_atr", 0.5)),
        disp_close_frac=float(sc.get("disp_close_frac", 0.5)),
    )

    # Optional Moon-Pluto overlay on the active window.
    astro_active = None
    if args.astro_overlay:
        ev = ephemeris_events.compute_aspect_events("moon", "pluto", [0, 90, 180, 270], start, end)
        ev = ephemeris_events.map_events_to_candles(ev, ltf)
        ebars = ev.loc[ev["bar_index"] >= 0, "bar_index"].to_numpy(int)
        amask = np.zeros(len(ltf), dtype=bool)
        win = 24  # +/-1 day on 1h
        for b in ebars:
            amask[max(0, b - win): min(len(ltf) - 1, b + win) + 1] = True
        astro_active = active & amask

    gated = active & signal
    ungated = signal

    def run(rr, mask):
        return strategy.backtest_long(ltf, mask, rr, sc["max_hold_bars"],
                                      sc["stop_buffer_atr"], sc["cost_bps_round_trip"])

    # Excursion pass (big target) -> R-distribution available after the trigger.
    exc_gated = run(sc["excursion_rr"], gated)
    exc_ungated = run(sc["excursion_rr"], ungated)

    # Fixed-RR P&L.
    rr_results = {}
    for rr in sc["rr_values"]:
        g = run(rr, gated)
        u = run(rr, ungated)
        entry = {"gated": _summ(g), "ungated": _summ(u)}
        # Per-year + holdout for the gated book.
        by_year = {str(y): _summ(ts) for y, ts in strategy.split_trades_by_year(g).items()}
        hold = [t for t in g if pd.Timestamp(t["entry_time"]) >= holdout_start]
        dev = [t for t in g if pd.Timestamp(t["entry_time"]) < holdout_start]
        entry["gated_by_year"] = by_year
        entry["gated_dev"] = _summ(dev)
        entry["gated_holdout"] = _summ(hold)
        rr_results[f"rr_{rr:g}"] = entry

    astro_results = None
    if astro_active is not None:
        astro_results = {f"rr_{rr:g}": _summ(run(rr, astro_active & signal)) for rr in sc["rr_values"]}

    results = {
        "config": {"symbol": symbol, "start": start, "end": end,
                   "ltf": sc["ltf_interval"], "htf": sc["htf_interval"],
                   "htf_dump_lookback": sc["htf_dump_lookback"],
                   "htf_dump_threshold_atr": sc["htf_dump_threshold_atr"],
                   "hold_days": sc["hold_days"], "stop_buffer_atr": sc["stop_buffer_atr"],
                   "max_hold_bars": sc["max_hold_bars"], "cost_bps_round_trip": sc["cost_bps_round_trip"],
                   "rr_values": sc["rr_values"], "holdout_start": str(holdout_start)},
        "data": {"ltf_bars": len(ltf), "daily_bars": len(daily),
                 "active_ltf_fraction": float(active.mean()),
                 "n_sweep_signals": int(signal.sum()),
                 "n_gated_signals": int(gated.sum())},
        "r_distribution": {"gated": strategy.r_distribution(exc_gated),
                           "ungated": strategy.r_distribution(exc_ungated),
                           "gated_avg_mfe_r": float(np.mean([t["mfe_r"] for t in exc_gated])) if exc_gated else float("nan"),
                           "gated_avg_mae_r": float(np.mean([t["mae_r"] for t in exc_gated])) if exc_gated else float("nan")},
        "rr_results": rr_results,
        "astro_overlay": astro_results,
    }

    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "strategy.json", results)
    (args.outdir / "strategy.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _row(name, s):
    return (f"| {name} | {s['trades']} | {_f(s['win_rate'])} | {_f(s['avg_r'])} | {_f(s['net_r'],1)} | "
            f"{_f(s['profit_factor'])} | {_f(s['max_drawdown_r'],1)} |")


def _markdown(r: dict) -> str:
    c = r["config"]
    d = r["data"]
    rd = r["r_distribution"]
    L = [
        "# High-RR Reversal Strategy (Milestone 4)",
        "",
        f"**{c['symbol']}** LTF {c['ltf']} / HTF {c['htf']} | {c['start']} -> {c['end']}",
        f"Setup: daily dump (>= {c['htf_dump_threshold_atr']} ATR over {c['htf_dump_lookback']}d) opens a "
        f"{c['hold_days']}-day long-alert window; enter on a {c['ltf']} sweep+reclaim of the prior-range "
        f"low; stop {c['stop_buffer_atr']} ATR below the sweep; costs {c['cost_bps_round_trip']} bps RT.",
        f"LTF bars {d['ltf_bars']} | sweep signals {d['n_sweep_signals']} | gated signals "
        f"{d['n_gated_signals']} | active window {_f(d['active_ltf_fraction'])} of time.",
        "",
        "## R available after the trigger (excursion pass, big target)",
        "",
        f"Gated avg MFE {_f(rd['gated_avg_mfe_r'])} R | avg MAE {_f(rd['gated_avg_mae_r'])} R",
        "",
        "| Book | reach 1R | 2R | 3R | 5R | 10R |",
        "|---|---|---|---|---|---|",
        f"| gated | {_f(rd['gated']['reach_1r'])} | {_f(rd['gated']['reach_2r'])} | "
        f"{_f(rd['gated']['reach_3r'])} | {_f(rd['gated']['reach_5r'])} | {_f(rd['gated']['reach_10r'])} |",
        f"| ungated | {_f(rd['ungated']['reach_1r'])} | {_f(rd['ungated']['reach_2r'])} | "
        f"{_f(rd['ungated']['reach_3r'])} | {_f(rd['ungated']['reach_5r'])} | {_f(rd['ungated']['reach_10r'])} |",
        "",
        "## Fixed-RR P&L (net of costs)",
        "",
    ]
    for rr_key, e in r["rr_results"].items():
        L += [
            f"### {rr_key.replace('rr_', 'target ')}R",
            "",
            "| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R |",
            "|---|---|---|---|---|---|---|",
            _row("gated (dump+sweep)", e["gated"]),
            _row("ungated (sweep only)", e["ungated"]),
            _row("gated DEV (<holdout)", e["gated_dev"]),
            _row("gated HOLDOUT (2025+)", e["gated_holdout"]),
            "",
            "Per-year (gated): " + ", ".join(
                f"{y} {_f(s['net_r'],1)}R ({s['trades']}t, {_f(s['win_rate'])}wr)"
                for y, s in sorted(e["gated_by_year"].items())),
            "",
        ]
    if r["astro_overlay"]:
        L += ["## Moon-Pluto overlay (longs only near a Dark Pivot)", "",
              "| Target | Trades | Win rate | Avg R | Net R | PF | MaxDD R |",
              "|---|---|---|---|---|---|---|"]
        for rr_key, s in r["astro_overlay"].items():
            L.append(_row(rr_key.replace("rr_", "") + "R", s))
        L.append("")
    L += [
        "## Reading guide",
        "",
        "Profitable only if avg R (expectancy) > 0 net of costs and it holds on DEV *and* HOLDOUT. "
        "Win rate needed ~ 1/(RR+1) (e.g. >33% at 2R, >17% at 5R). Compare gated vs ungated to see if "
        "the dump gate helps; compare the astro overlay to confirm it does not. High RR with thin win "
        "rate => fat-tailed equity; weight MaxDD and per-year stability, not just net R.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
