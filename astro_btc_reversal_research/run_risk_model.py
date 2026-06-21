"""Production-ish risk model for the 5m Capitulation Spring long.

Builds the trades, fetches BTC.D (Binance BTCDOMUSDT) + BTC daily for regime, and
reports real account metrics (CAGR / MaxDD% / MAR) under: regime gates (BTC freefall,
BTC.D ripping) and position-sizing / concurrency choices.

Outputs (reports/5m): risk_model.json and risk_model.md
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
    basket, capitulation as cap, data, exits, ltf_structure as lts, portfolio, regime,
    reuse, report, weekly_timing as wt,
)

INTERVAL, SWEEP_LB, MAX_HOLD = "5m", 4320, 4032
TPS, FRACS = (4.0, 12.0, 30.0), (0.25, 0.50, 0.25)


def build_long_trades(b):
    trades = []
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
        mask = sp & np.where(np.isfinite(fzv), fzv <= -1.0, False)
        tr = exits.backtest_scaled(fr, mask, "long", TPS, FRACS, max_hold_bars=MAX_HOLD,
                                   stop_buffer_atr=0.05, maker_bps=2.0, taker_bps=5.5)
        for t in tr:
            t["symbol"] = sym
        trades.extend(tr)
    return trades


def main() -> int:
    p = argparse.ArgumentParser(description="Risk model + BTC.D regime gates for the capitulation long.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "reports" / "5m")
    args = p.parse_args()
    cfg = data.load_config(args.config)
    b = cfg["basket_spring"]

    trades = build_long_trades(b)
    btcdom = regime.fetch_btcdom("1d", "2021-01-01", None)
    btc_daily = data.load_ohlcv("BTCUSDT", "1d", b["start"], b["end"], base_interval="15m")
    reg = regime.build_daily_regime(btc_daily, btcdom)
    reg_rows = regime.regime_for_trades(trades, reg)

    gates = {
        "none": dict(freefall_thr=None, btcd_up_thr=None),
        "freefall<-0.15": dict(freefall_thr=-0.15, btcd_up_thr=None),
        "freefall<-0.25": dict(freefall_thr=-0.25, btcd_up_thr=None),
        "btcd_14d>+0.05": dict(freefall_thr=None, btcd_up_thr=0.05),
        "btcd_14d>+0.08": dict(freefall_thr=None, btcd_up_thr=0.08),
        "freefall<-0.15 & btcd>+0.05": dict(freefall_thr=-0.15, btcd_up_thr=0.05),
    }
    gate_results = {}
    for name, rules in gates.items():
        skip = regime.skip_mask(reg_rows, **rules)
        kept = [t for t, s in zip(trades, skip) if not s]
        gate_results[name] = {"n_skipped": int(skip.sum()),
                              **portfolio.simulate_equity(kept, risk_pct=0.005, max_concurrent=4)}

    # Sizing / concurrency grid on the no-gate book.
    sizing = {}
    for risk in (0.005, 0.0075, 0.01):
        for K in (2, 4, 6):
            sizing[f"risk{risk*100:g}%_K{K}"] = portfolio.simulate_equity(trades, risk_pct=risk, max_concurrent=K)

    btcd_cov = reg_rows["btcd_14d_chg"].notna().mean() if len(reg_rows) else 0.0
    results = {
        "config": {"interval": INTERVAL, "fz_thr": -1.0, "stop_buffer_atr": 0.05,
                   "fees": "taker entry/stop 5.5bps, maker TP 2bps", "tps": list(TPS), "fracs": list(FRACS)},
        "n_trades_raw": len(trades),
        "btcd_coverage": round(float(btcd_cov), 2),
        "regime_gates": gate_results,
        "sizing_grid": sizing,
    }
    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "risk_model.json", results)
    (args.outdir / "risk_model.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _g(s, k):
    v = s.get(k)
    return "n/a" if v is None or (isinstance(v, float) and v != v) else v


def _markdown(r):
    L = [
        "# 5m Capitulation Long — Risk Model + BTC.D Regime Gate",
        "",
        f"{r['n_trades_raw']} raw trades | fees: {r['config']['fees']} | risk 0.5%/trade, max 4 concurrent "
        f"unless noted | BTC.D coverage {r['btcd_coverage']:.0%} of trades.",
        "",
        "## Regime gates (account metrics, risk 0.5%, K=4)",
        "",
        "| Gate | Skipped | Trades | /wk | Win % | CAGR % | MaxDD % | MAR | Total % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in r["regime_gates"].items():
        L.append(f"| {name} | {s.get('n_skipped',0)} | {_g(s,'trades_taken')} | {_g(s,'trades_per_week')} | "
                 f"{_g(s,'win_rate')} | {_g(s,'cagr_pct')} | {_g(s,'max_dd_pct')} | {_g(s,'mar')} | "
                 f"{_g(s,'total_return_pct')} |")
    L += ["", "## Sizing / concurrency grid (no gate)", "",
          "| Config | Trades | /wk | CAGR % | MaxDD % | MAR | Total % |",
          "|---|---|---|---|---|---|---|"]
    for name, s in r["sizing_grid"].items():
        L.append(f"| {name} | {_g(s,'trades_taken')} | {_g(s,'trades_per_week')} | {_g(s,'cagr_pct')} | "
                 f"{_g(s,'max_dd_pct')} | {_g(s,'mar')} | {_g(s,'total_return_pct')} |")
    L += ["", "## Reading guide", "",
          "MAR = CAGR / MaxDD% (higher is better; >0.5 is decent, >1 is strong). The regime gate is "
          "worth keeping only if it raises MAR (cuts MaxDD more than it cuts CAGR). BTC.D rising = alts "
          "bleeding; BTC freefall = cascade. But capitulation bottoms often occur DURING those, so a gate "
          "can also remove the best entries - let the MAR column decide.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
