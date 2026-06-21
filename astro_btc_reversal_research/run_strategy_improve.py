"""Address the two real objections to the 5m Capitulation Spring:
  (1) fees vs tight stops  -> decompose gross/net, model realistic maker/taker fills;
  (2) -26R drawdown        -> sweep stop width, and cap concurrent positions (portfolio).

Outputs (reports/5m): strategy_improve.json and strategy_improve.md
"""

from __future__ import annotations

import argparse
import heapq
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

INTERVAL, SWEEP_LB, MAX_HOLD = "5m", 4320, 4032
TPS, FRACS = (4.0, 12.0, 30.0), (0.25, 0.50, 0.25)


def max_dd_r(trades) -> float:
    if not trades:
        return 0.0
    rs = [t["result_r"] for t in sorted(trades, key=lambda t: t["exit_time"])]
    eq = np.cumsum(rs)
    return float(np.min(eq - np.maximum.accumulate(eq)))


def summarize(trades, span_weeks, holdout_start):
    if not trades:
        return {"trades": 0}
    r = np.array([t["result_r"] for t in trades])
    hold = [t for t in trades if pd.Timestamp(t["entry_time"]) >= holdout_start]
    return {
        "trades": len(trades), "trades_per_week": round(len(trades) / span_weeks, 2),
        "win_rate": round(float(np.mean(r > 0)) * 100, 1), "avg_r": round(float(r.mean()), 3),
        "net_r": round(float(r.sum()), 1), "max_dd_r": round(max_dd_r(trades), 1),
        "holdout_avg_r": round(float(np.mean([t["result_r"] for t in hold])), 3) if hold else float("nan"),
        "holdout_n": len(hold),
    }


def portfolio_cap(trades, k):
    """Take signals in entry-time order; skip when k positions already open."""
    ts = sorted(trades, key=lambda t: t["entry_time"])
    open_exits: list = []
    taken = []
    for t in ts:
        et = pd.Timestamp(t["entry_time"])
        while open_exits and open_exits[0] <= et:
            heapq.heappop(open_exits)
        if k is None or len(open_exits) < k:
            taken.append(t)
            heapq.heappush(open_exits, pd.Timestamp(t["exit_time"]))
    return taken


def build_trades(stop_buffer, fz_thr, fee_kind, masks_cache, frames, span_weeks):
    """Backtest capitulation-long scaled trades across the basket for one config."""
    fee = {"gross": dict(maker_bps=0.0, taker_bps=0.0),
           "taker11": dict(maker_bps=5.5, taker_bps=5.5),
           "realistic": dict(maker_bps=2.0, taker_bps=5.5)}[fee_kind]
    out = []
    for sym, frame in frames.items():
        mask = masks_cache[(sym, fz_thr)]
        tr = exits.backtest_scaled(frame, mask, "long", TPS, FRACS, max_hold_bars=MAX_HOLD,
                                   stop_buffer_atr=stop_buffer, **fee)
        for t in tr:
            t["symbol"] = sym
        out.extend(tr)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Fee + drawdown improvements for the 5m capitulation long.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "reports" / "5m")
    args = p.parse_args()
    cfg = data.load_config(args.config)
    b = cfg["basket_spring"]
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")
    span_weeks = (reuse.parse_utc_datetime(b["end"]) - reuse.parse_utc_datetime(b["start"])).days / 7.0

    # Load frames + capitulation masks once (fz in {-1.0, -1.5}).
    frames, masks = {}, {}
    for sym in basket.BASKET:
        try:
            fr = basket.load_symbol(sym, INTERVAL, b["start"], b["end"])
            fu = cap.fetch_funding(sym, b["start"], None)
            oi = cap.fetch_oi(sym, "4h", b["start"], None)
        except Exception:
            continue
        if len(fr) < SWEEP_LB + 100:
            continue
        _, frac = wt.time_of_week(fr)
        sp = lts.spring_long_signals(fr, SWEEP_LB, b["wick_frac"], b["close_pos"]) & (frac < b["week_frac_max"])
        fz = cap.capitulation_features(fr, fu, oi).get("funding_z")
        if fz is None:
            continue
        fzv = fz.to_numpy()
        frames[sym] = fr
        for thr in (-1.0, -1.5):
            masks[(sym, thr)] = sp & np.where(np.isfinite(fzv), fzv <= thr, False)

    results = {"config": {"interval": INTERVAL, "span_weeks": round(span_weeks, 1),
                          "tps": list(TPS), "fracs": list(FRACS), "holdout_start": str(holdout_start)}}

    # 1) Fee decomposition (buffer 0.05, fz<=-1).
    base = {k: build_trades(0.05, -1.0, k, masks, frames, span_weeks) for k in ("gross", "taker11", "realistic")}
    rp = np.array([t["risk_pct"] for t in base["realistic"]])
    results["fee_decomposition"] = {
        "median_stop_pct": round(float(np.median(rp)) * 100, 3),
        "mean_stop_pct": round(float(np.mean(rp)) * 100, 3),
        "gross": summarize(base["gross"], span_weeks, holdout_start),
        "net_taker_11bps": summarize(base["taker11"], span_weeks, holdout_start),
        "net_realistic_makerTP": summarize(base["realistic"], span_weeks, holdout_start),
    }

    # 2) Stop-width x funding sweep (realistic fees).
    sweep = {}
    for buf in (0.05, 0.15, 0.30, 0.50):
        for thr in (-1.0, -1.5):
            tr = build_trades(buf, thr, "realistic", masks, frames, span_weeks)
            sweep[f"buf{buf}_fz{thr}"] = summarize(tr, span_weeks, holdout_start)
    results["stop_funding_sweep"] = sweep

    # 3) Portfolio concurrency cap (realistic fees, buffer 0.30, fz<=-1 -> a robust config).
    cap_trades = build_trades(0.30, -1.0, "realistic", masks, frames, span_weeks)
    results["portfolio_cap"] = {
        (f"K={k}" if k else "K=inf"): summarize(portfolio_cap(cap_trades, k), span_weeks, holdout_start)
        for k in (None, 8, 6, 4, 2)
    }

    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "strategy_improve.json", results)
    (args.outdir / "strategy_improve.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _row(name, s):
    if not s or s.get("trades", 0) == 0:
        return f"| {name} | 0 | | | | | | |"
    return (f"| {name} | {s['trades']} | {s['trades_per_week']} | {s['win_rate']} | {s['avg_r']} | "
            f"{s['net_r']} | {s['max_dd_r']} | {s['holdout_avg_r']} ({s['holdout_n']}) |")


def _markdown(r):
    fd = r["fee_decomposition"]
    L = [
        "# 5m Capitulation Long — Fee & Drawdown Improvements",
        "",
        f"Median stop distance: **{fd['median_stop_pct']}% of price** (mean {fd['mean_stop_pct']}%).",
        "",
        "## 1) Fee decomposition (buffer 0.05 ATR, funding_z<=-1)",
        "",
        "| Fee model | Trades | /wk | Win % | Avg R | Net R | MaxDD R | Holdout avg R (n) |",
        "|---|---|---|---|---|---|---|---|",
        _row("gross (0 fees)", fd["gross"]),
        _row("net taker 11bps RT", fd["net_taker_11bps"]),
        _row("net realistic (taker entry/stop, maker TP)", fd["net_realistic_makerTP"]),
        "",
        "## 2) Stop-width x funding sweep (realistic fees)",
        "",
        "| Config | Trades | /wk | Win % | Avg R | Net R | MaxDD R | Holdout avg R (n) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, s in r["stop_funding_sweep"].items():
        L.append(_row(k, s))
    L += ["", "## 3) Portfolio concurrency cap (realistic fees, buffer 0.30, fz<=-1)", "",
          "| Max concurrent | Trades | /wk | Win % | Avg R | Net R | MaxDD R | Holdout avg R (n) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in r["portfolio_cap"].items():
        L.append(_row(k, s))
    L += ["", "## Reading guide", "",
          "Wider stops cut the fee fraction (fewer R but lower MaxDD); the concurrency cap is the real "
          "drawdown lever — capping correlated knife-catches during market-wide crashes shrinks MaxDD "
          "while barely touching avg R. Net account DD% = MaxDD_R x risk-per-trade; with a cap of K and "
          "total heat H%, risk-per-trade = H/K.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
