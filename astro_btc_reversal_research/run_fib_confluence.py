"""Fibonacci TIME + PRICE confluence on the spring reversal.

For each deep-sweep 5m Wyckoff spring on BTC, flag whether it occurs at a Fibonacci
TIME projection (swing-ratio level within a tolerance window) and/or at a Fibonacci
PRICE level (retracement/extension of the prior swing). Compare the spring's outcome
(reach-20R, mean max-R) across books vs non-Fib placebos:

  all, fib_time, fib_price, fib_both, nonfib_time, nonfib_price, neither

Verdict: the user's "Fib time AND price" idea works only if `fib_both` clearly beats
`all` and the non-Fib placebos (and holds out). Outputs reports/fib_confluence.{json,md}.
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
    basket, data, exits, fib_price, fib_time, ltf_structure as lts, pivots as piv_mod, report,
)


def _time_near_mask(levels, n, window):
    mask = np.zeros(n, dtype=bool)
    for b in levels:
        mask[max(0, b - window): min(n - 1, b + window) + 1] = True
    return mask


def main() -> int:
    p = argparse.ArgumentParser(description="Fib time+price confluence on the 5m spring.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--basket", action="store_true", help="pool springs across the 18-symbol basket")
    p.add_argument("--sweep-lookback", type=int, default=4320)
    p.add_argument("--pivot-threshold-atr", type=float, default=5.0)
    p.add_argument("--time-window", type=int, default=12, help="fib-time tolerance in 5m bars")
    p.add_argument("--price-tol", type=float, default=0.05, help="fib-price tolerance (frac of swing range)")
    p.add_argument("--max-hold", type=int, default=4032)
    p.add_argument("--stop-buffer-atr", type=float, default=0.05)
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")
    fib_ex1 = tuple(r for r in fib_time.FIB_RATIOS if r != 1.0)
    symbols = basket.BASKET if args.basket else [args.symbol]
    totals = {"bars": 0, "springs": 0, "pivots": 0}

    def build_rows(symbol):
        frame = basket.load_symbol(symbol, "5m", cfg["data"]["start"], cfg["data"]["end"])
        n = len(frame)
        if n < args.sweep_lookback + 100:
            return []
        high = frame["high"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        close = frame["close"].to_numpy(float)
        atr = frame["atr"].to_numpy(float)
        holdout = (pd.to_datetime(frame["open_time"], utc=True) >= holdout_start).to_numpy()
        pivots = piv_mod.atr_directional_pivots(frame, args.pivot_threshold_atr)
        spring = lts.spring_long_signals(frame, args.sweep_lookback, 0.5, 0.5)
        spring_idx = np.flatnonzero(spring)
        fib_tm = _time_near_mask(fib_time.swing_ratio_levels(pivots, n, fib_ex1), n, args.time_window)
        nonfib_tm = _time_near_mask(fib_time.swing_ratio_levels(pivots, n, fib_time.NONFIB_RATIOS), n, args.time_window)
        totals["bars"] += n
        totals["springs"] += int(spring_idx.size)
        totals["pivots"] += len(pivots)
        out = []
        for b in spring_idx:
            b = int(b)
            risk = close[b] - (low[b] - args.stop_buffer_atr * atr[b])
            end = min(n - 1, b + args.max_hold)
            if risk <= 0 or not np.isfinite(risk) or end <= b:
                continue
            mfe = (np.nanmax(high[b + 1:end + 1]) - close[b]) / risk
            tr = exits.simulate_scaled_trade(frame, b, "long", (4.0, 12.0, 30.0), (0.25, 0.5, 0.25),
                                             max_hold_bars=args.max_hold, stop_buffer_atr=args.stop_buffer_atr,
                                             maker_bps=2.0, taker_bps=5.5)
            leg = fib_price.active_leg(pivots, b)
            fp, _ = fib_price.price_at_fib(low[b], leg, fib_price.FIB_RETRACE, fib_price.FIB_EXT, args.price_tol)
            nfp, _ = fib_price.price_at_fib(low[b], leg, fib_price.NONFIB_RETRACE, fib_price.NONFIB_EXT, args.price_tol)
            out.append({"symbol": symbol, "mfe": float(mfe), "reach20": mfe >= 20,
                        "result_r": float(tr["result_r"]) if tr else float("nan"),
                        "fib_time": bool(fib_tm[b]), "nonfib_time": bool(nonfib_tm[b]),
                        "fib_price": bool(fp), "nonfib_price": bool(nfp), "holdout": bool(holdout[b])})
        return out

    rows = []
    for sym in symbols:
        try:
            rows.extend(build_rows(sym))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {sym}: {exc}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no springs")
        return 0
    n, pivots = totals["bars"], [None] * totals["pivots"]

    def summ(mask):
        sub = df[mask]
        if len(sub) == 0:
            return {"n": 0}
        h = sub[sub["holdout"]]
        r = sub["result_r"].dropna()
        gains = float(r[r > 0].sum())
        losses = float(-r[r < 0].sum())
        return {"n": int(len(sub)), "reach20": float(sub["reach20"].mean()),
                "mean_mfe": float(sub["mfe"].mean()),
                "avg_r": float(r.mean()) if len(r) else float("nan"),
                "win_rate": float((r > 0).mean()) if len(r) else float("nan"),
                "pf": float(gains / losses) if losses > 0 else float("inf"),
                "net_r": float(r.sum()),
                "holdout_n": int(len(h)),
                "holdout_avg_r": float(h["result_r"].dropna().mean()) if len(h) else float("nan")}

    books = {
        "all": summ(np.ones(len(df), bool)),
        "fib_time": summ(df["fib_time"]),
        "fib_price": summ(df["fib_price"]),
        "fib_both": summ(df["fib_time"] & df["fib_price"]),
        "nonfib_time": summ(df["nonfib_time"]),
        "nonfib_price": summ(df["nonfib_price"]),
        "nonfib_both": summ(df["nonfib_time"] & df["nonfib_price"]),
        "neither": summ(~df["fib_time"] & ~df["fib_price"]),
    }
    results = {
        "config": {"symbol": ("basket(%d)" % len(symbols)) if args.basket else args.symbol,
                   "sweep_lookback": args.sweep_lookback,
                   "pivot_threshold_atr": args.pivot_threshold_atr, "time_window": args.time_window,
                   "price_tol": args.price_tol, "max_hold": args.max_hold, "bars": n,
                   "n_springs": int(len(df)), "n_pivots": len(pivots),
                   "all_reach20": books["all"]["reach20"], "holdout_start": str(holdout_start)},
        "books": books,
    }
    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "fib_confluence.json", results)
    (args.outdir / "fib_confluence.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r):
    c = r["config"]
    L = [
        "# Fibonacci TIME + PRICE confluence on the 5m spring",
        "",
        f"**{c['symbol']}** | {c['bars']} bars | {c['n_springs']} deep-sweep springs (lookback "
        f"{c['sweep_lookback']}) | {c['n_pivots']} pivots | fib-time +/-{c['time_window']} bars | "
        f"fib-price tol {c['price_tol']} of swing range.",
        f"Baseline (all springs) reach-20R: **{_f(c['all_reach20'])}**. Profitability = scaled exit "
        "4/12/30R, stop->BE, realistic fees (taker entry/stop, maker TP).",
        "",
        "| Book | Springs | Avg R | Win % | PF | Net R | reach-20R | Holdout avg R (n) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ["all", "fib_time", "fib_price", "fib_both", "nonfib_both", "neither"]:
        b = r["books"][name]
        if b.get("n", 0) == 0:
            L.append(f"| {name} | 0 | | | | | | |")
            continue
        L.append(f"| {name} | {b['n']} | {_f(b['avg_r'])} | {_f(b['win_rate'])} | {_f(b['pf'],2)} | "
                 f"{_f(b['net_r'],1)} | {_f(b['reach20'])} | {_f(b['holdout_avg_r'])} ({b['holdout_n']}) |")
    L += [
        "",
        "## Reading guide (trader's framing)",
        "",
        "Useful as a filter only if **fib_both** (Fib time AND price) has a clearly higher **avg R** "
        "(net of costs) than **all** springs (the no-filter baseline) AND than **neither** (springs that "
        "fail the filter), and it holds on the holdout. We are NOT asking whether it beats a random/non-Fib "
        "set - only whether the confluence selects more profitable entries than trading every spring. "
        "Weight the sample size: the confluence is a deep filter, so n is small.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
