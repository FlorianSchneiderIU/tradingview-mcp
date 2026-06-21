#!/usr/bin/env python3
"""Entry-timing probe: what happens AT the stop-loss level?

For each (gated) Wolfe signal we simulate a SHIFTED trade that only fills if price
reaches the original SL (E -> SL), with the new stop a further R away (SL -> SL-R).
This deliberately misses every original winner; it isolates the population that got
stopped and asks: from the SL, does price mean-revert (bounce) or continue?

If the bounce expectancy is positive, a DCA second entry at the SL is justified.

Three target choices for the shifted entry at E2=SL (R = |E-SL|):
  sym1R   : target SL+R (=original entry E),  stop SL-R   -> symmetric +/-1R bounce test
  origRR  : target SL+RR*R (original geometry shifted down by R), stop SL-R
  dca_T   : target = original target T (the real DCA second leg), stop SL-R

All R-multiples are net of cost and in units of the original R. Discovery on
train+val; held-out (>=2025-06) reported separately.

Usage: python scripts/wolfe_entry_timing_research.py
"""
from __future__ import annotations
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, importlib
bw = importlib.import_module("backtest_wolfe_wave")

CFG = "bot/configs/wolfe_wave_shared_v1_configs.json"
TRAIN_END = pd.Timestamp("2024-12-01", tz="UTC")
VAL_END = pd.Timestamp("2025-06-01", tz="UTC")
HOLD = 144  # exec bars to wait for fill, and again for the shifted bracket


def metrics(rs):
    rs = np.asarray(rs, float); n = len(rs)
    if n == 0: return dict(n=0, win=0.0, pf=0.0, avg=0.0, net=0.0)
    gw = rs[rs > 0].sum(); gl = -rs[rs < 0].sum()
    return dict(n=n, win=float((rs > 0).mean()), pf=(gw/gl) if gl > 1e-9 else 99.0,
                avg=float(rs.mean()), net=float(rs.sum()))


def sim_shifted(frame, sig, cfg, mode):
    """Returns (status, r_multiple). status in {'nofill','win','loss','timeout'}."""
    i = int(sig.entry_index)
    E = float(sig.entry_price); SL = float(sig.stop_price); T = float(sig.target_price)
    R = abs(E - SL)
    if R <= 0:
        return ("nofill", 0.0)
    long = sig.direction == "long"
    rr = abs(T - E) / R
    E2 = SL
    SL2 = SL - R if long else SL + R
    if mode == "sym1R":
        T2 = SL + R if long else SL - R
    elif mode == "origRR":
        T2 = SL + rr * R if long else SL - rr * R
    else:  # dca_T
        T2 = T
    n = len(frame)
    high = frame["high"].to_numpy(); low = frame["low"].to_numpy()
    open_ = frame["open"].to_numpy(); close = frame["close"].to_numpy()
    # 1) wait for fill at E2 (price reaches the old SL)
    fill_i = None
    for j in range(i + 1, min(n - 1, i + HOLD) + 1):
        if (long and low[j] <= E2) or (not long and high[j] >= E2):
            fill_i = j; break
    if fill_i is None:
        return ("nofill", 0.0)
    cost = bw._cost_r(E2, R, cfg)  # round-trip cost in R units
    # 2) bracket from fill bar onward
    for j in range(fill_i, min(n - 1, fill_i + HOLD) + 1):
        hi = high[j]; lo = low[j]
        if long:
            tgt = hi >= T2; stp = lo <= SL2
            if tgt and stp:
                return ("win", rr - cost) if bw.high_before_low(open_[j], hi, lo) else ("loss", -1.0 - cost)
            if stp: return ("loss", -1.0 - cost)
            if tgt: return ("win", (T2 - E2) / R - cost)
        else:
            tgt = lo <= T2; stp = hi >= SL2
            if tgt and stp:
                return ("loss", -1.0 - cost) if bw.high_before_low(open_[j], hi, lo) else ("win", rr - cost)
            if stp: return ("loss", -1.0 - cost)
            if tgt: return ("win", (E2 - T2) / R - cost)
    # timeout: mark to last close
    last = close[min(n - 1, fill_i + HOLD)]
    r = ((last - E2) if long else (E2 - last)) / R - cost
    return ("timeout", r)


def main():
    cfg_dict = json.load(open(CFG))
    universe = [k for k in cfg_dict if not k.startswith("_")]
    base = {k: v for k, v in cfg_dict[universe[0]].items() if not k.startswith("_")}
    modes = ["sym1R", "origRR", "dca_T"]
    recs = []   # one row per signal: entry_time, original outcome, shifted outcomes per mode
    for sym in universe:
        p = f"scripts/data/{sym.lower()}_5m_bybit.csv"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        cfgmap = dict(base); cfgmap["mintick"] = cfg_dict[sym].get("mintick", 0.01)
        cfg = bw.WolfeConfig.from_mapping(cfgmap)
        frame = bw.add_indicators(bw.ensure_ohlcv_frame(df), cfg.atr_length, cfg.ema_length, cfg.rsi_length)
        try:
            signals = bw.find_wolfe_signals(df, cfg, symbol=sym)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym} ERR {exc}", flush=True); continue
        for sig in signals:
            row = {"entry_time": pd.Timestamp(sig.entry_time)}
            for mode in modes:
                st, r = sim_shifted(frame, sig, cfg, mode)
                row[f"{mode}_status"] = st
                row[f"{mode}_r"] = r
            recs.append(row)
    d = pd.DataFrame(recs)
    d["entry_time"] = pd.to_datetime(d["entry_time"], utc=True)
    dev = d[d.entry_time < VAL_END]
    oos = d[d.entry_time >= VAL_END]
    print(f"total gated signals: {len(d)} | dev {len(dev)} | oos {len(oos)}")

    for label, seg in [("DEV(train+val)", dev), ("OOS held-out(>=25-06)", oos)]:
        print(f"\n=== {label} — shifted entry at SL (fills only when original would be stopped) ===")
        for mode in modes:
            filled = seg[seg[f"{mode}_status"] != "nofill"]
            fill_rate = len(filled) / len(seg) if len(seg) else 0.0
            m = metrics(filled[f"{mode}_r"])
            print(f"  {mode:7} fill_rate={fill_rate:.0%}  n={m['n']:5d} win={m['win']:.1%} "
                  f"PF={m['pf']:.2f} avg_r={m['avg']:+.3f} netR={m['net']:+.1f}")
    print("\nNote: net R is in units of the ORIGINAL R. A profitable shifted trade means")
    print("price tends to bounce from the SL — i.e. a DCA second entry at the SL adds value.")


if __name__ == "__main__":
    main()
