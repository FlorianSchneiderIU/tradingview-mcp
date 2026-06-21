#!/usr/bin/env python3
"""Proposed DCA strategy vs current gated Wolfe, sweeping the DCA stop tightness.

Per gated signal (long; short mirrored): E entry, SL=E-R, target T (R=|E-SL|).
  * Path A — T before SL: leg1 full TP (+RR). DCA never fills.
  * Path B — SL first: DCA leg2 fills at E2=SL. Combined target = first entry E;
    combined stop = SL - k*R (tweakable). On bounce-to-E: 0 + 1R = +1R.
    On stop: leg1 -(1+k)R + leg2 -k*R = -(1+2k)R.

Tighter k -> higher RR on the DCA leg (reward R vs risk k*R = 1/k) and smaller
tail, but the bounce must happen before dipping k*R below SL.

Headline metric is ret/DD (net R / max drawdown R) — scale-invariant, so it
strips out the leverage that made raw avg_r/PF look good. Compare to ORIGINAL.

Usage: python scripts/wolfe_dca_strategy_research.py
"""
from __future__ import annotations
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, importlib
bw = importlib.import_module("backtest_wolfe_wave")

CFG = "bot/configs/wolfe_wave_shared_v1_configs.json"
TRAIN_END = pd.Timestamp("2024-12-01", tz="UTC")
VAL_END = pd.Timestamp("2025-06-01", tz="UTC")
HOLD = 144
K_VALUES = [0.15, 0.2, 0.25, 0.3, 0.4]
MONTHLY_K = 0.25   # report month-by-month regime stability for this k


def maxdd(arr):
    eq = np.cumsum(np.asarray(arr, float)); peak = np.maximum.accumulate(eq)
    return float((peak - eq).max()) if len(eq) else 0.0


def metrics(rs):
    rs = np.asarray(rs, float); n = len(rs)
    if n == 0: return dict(n=0, win=0.0, pf=0.0, avg=0.0, net=0.0, worst=0.0, dd=0.0, rdd=0.0)
    gw = rs[rs > 0].sum(); gl = -rs[rs < 0].sum()
    dd = maxdd(rs)
    return dict(n=n, win=float((rs > 0).mean()), pf=(gw/gl) if gl > 1e-9 else 99.0,
                avg=float(rs.mean()), net=float(rs.sum()), worst=float(rs.min()),
                dd=dd, rdd=float(rs.sum()/dd) if dd > 1e-9 else 0.0)


def sim(frame, sig, cfg):
    """Return (orig_r, {k: prop_r}, path) — leg1 walked once, DCA branch per k."""
    i = int(sig.entry_index)
    E = float(sig.entry_price); SL = float(sig.stop_price); T = float(sig.target_price)
    R = abs(E - SL)
    if R <= 0:
        return None
    long = sig.direction == "long"
    rr = abs(T - E) / R
    cost = bw._cost_r(E, R, cfg)
    n = len(frame)
    high = frame["high"].to_numpy(); low = frame["low"].to_numpy()
    op = frame["open"].to_numpy(); close = frame["close"].to_numpy()

    sl_bar = None
    for j in range(i + 1, min(n - 1, i + HOLD) + 1):
        hi = high[j]; lo = low[j]
        tgt = (hi >= T) if long else (lo <= T)
        stp = (lo <= SL) if long else (hi >= SL)
        if tgt and stp:
            if bw.high_before_low(op[j], hi, lo) == long:
                tgt, stp = True, False
            else:
                tgt, stp = False, True
        if tgt:
            return (rr - cost, {k: rr - cost for k in K_VALUES}, "A_target", E / R)
        if stp:
            sl_bar = j; break
    if sl_bar is None:
        cl = close[min(n - 1, i + HOLD)]
        r = ((cl - E) if long else (E - cl)) / R - cost
        return (r, {k: r for k in K_VALUES}, "timeout", E / R)

    orig = -1.0 - cost
    props = {}
    for k in K_VALUES:
        SL2 = SL - k * R if long else SL + k * R
        out = None
        for m in range(sl_bar, min(n - 1, sl_bar + HOLD) + 1):
            hm = high[m]; lm = low[m]
            btgt = (hm >= E) if long else (lm <= E)
            bstp = (lm <= SL2) if long else (hm >= SL2)
            if btgt and bstp:
                if bw.high_before_low(op[m], hm, lm) == long:
                    btgt, bstp = True, False
                else:
                    btgt, bstp = False, True
            if btgt:
                out = 1.0 - 2.0 * cost; break          # leg1 0 + leg2 +1R
            if bstp:
                out = -(1.0 + 2.0 * k) - 2.0 * cost; break
        if out is None:
            cl = close[min(n - 1, sl_bar + HOLD)]
            leg1 = ((cl - E) if long else (E - cl)) / R
            leg2 = ((cl - SL) if long else (SL - cl)) / R
            out = leg1 + leg2 - 2.0 * cost
        props[k] = out
    return (orig, props, "B", E / R)


def main():
    cfg_dict = json.load(open(CFG))
    universe = [k for k in cfg_dict if not k.startswith("_")]
    base = {k: v for k, v in cfg_dict[universe[0]].items() if not k.startswith("_")}
    rows = []
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
            r = sim(frame, sig, cfg)
            if r is None:
                continue
            row = {"entry_time": pd.Timestamp(sig.entry_time), "orig_r": r[0], "path": r[2], "e_over_r": r[3]}
            for k in K_VALUES:
                row[f"prop_{k}"] = r[1][k]
            rows.append(row)
    d = pd.DataFrame(rows); d["entry_time"] = pd.to_datetime(d["entry_time"], utc=True)
    d.to_csv("scripts/output/wolfe_dca_ksweep.csv", index=False)

    for label, seg in [("DEV(train+val)", d[d.entry_time < VAL_END].sort_values("entry_time")),
                       ("OOS held-out(>=25-06)", d[d.entry_time >= VAL_END].sort_values("entry_time"))]:
        mo = metrics(seg["orig_r"])
        print(f"\n=== {label}  (n={len(seg)}) ===")
        print(f"  ORIGINAL        WR={mo['win']:.0%} PF={mo['pf']:.2f} avg={mo['avg']:+.3f} net={mo['net']:+.0f} maxDD={mo['dd']:.0f} worst={mo['worst']:+.2f}  ret/DD={mo['rdd']:.2f}")
        for k in K_VALUES:
            m = metrics(seg[f"prop_{k}"])
            tail = -(1.0 + 2.0 * k)
            print(f"  DCA k={k:<4} (tail {tail:+.1f}R)  WR={m['win']:.0%} PF={m['pf']:.2f} avg={m['avg']:+.3f} net={m['net']:+.0f} maxDD={m['dd']:.0f} worst={m['worst']:+.2f}  ret/DD={m['rdd']:.2f}")
    print("\nret/DD is scale-invariant (risk-adjusted). DCA only wins if ret/DD beats ORIGINAL.")

    # Monthly regime stability for the chosen k (does it beat original every month?)
    oos = d[d.entry_time >= VAL_END].copy()
    oos["m"] = oos.entry_time.dt.to_period("M")
    col = f"prop_{MONTHLY_K}"
    print(f"\nOOS monthly (k={MONTHLY_K}) net R — regime stability:")
    wins = 0; tot = 0
    for m, g in oos.groupby("m"):
        o = g["orig_r"].sum(); p = g[col].sum(); tot += 1; wins += int(p >= o)
        flag = "" if p >= o else "  <-- DCA worse"
        print(f"  {m}  n={len(g):3d}  orig={o:+6.1f}  dca={p:+6.1f}{flag}")
    print(f"  months DCA>=orig: {wins}/{tot}")

    # --- Slippage stress test (k=MONTHLY_K) ---
    # Extra adverse slippage S bps applied per unit-fill. Original = 2 unit-fills
    # (entry+exit). DCA Path A/timeout = 2; Path B = 4 (leg1 entry + leg2 entry +
    # 2x-size combined exit) -> DCA carries ~2x the slip drag, the realistic concern.
    print(f"\n=== SLIPPAGE STRESS (k={MONTHLY_K}) — extra bps/unit-fill on top of base cost ===")
    kk = f"prop_{MONTHLY_K}"
    seg = d[d.entry_time >= VAL_END].sort_values("entry_time").copy()
    fills_orig = 2.0
    fills_dca = seg["path"].map(lambda p: 4.0 if p == "B" else 2.0).to_numpy()
    eor = seg["e_over_r"].to_numpy()
    print(f"  {'S(bps)':>7} | {'ORIG PF/retDD':>16} | {'DCA PF/retDD':>16}")
    for S in (0, 3, 7, 15):
        x = (S / 1e4) * eor                       # adverse R per unit-fill, per trade
        o_adj = seg["orig_r"].to_numpy() - fills_orig * x
        d_adj = seg[kk].to_numpy() - fills_dca * x
        mo = metrics(pd.Series(o_adj)); md = metrics(pd.Series(d_adj))
        print(f"  {S:>7} | {mo['pf']:.2f} / {mo['rdd']:>6.2f}   | {md['pf']:.2f} / {md['rdd']:>6.2f}")
    print("  (DCA edge is robust if it still beats ORIG PF & ret/DD at 7-15 extra bps.)")


if __name__ == "__main__":
    main()
