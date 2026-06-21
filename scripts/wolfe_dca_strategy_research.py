#!/usr/bin/env python3
"""Evaluate the proposed DCA strategy switch vs the current gated Wolfe.

Per gated signal (long; short mirrored):
  * Leg1 enters at E with original target T and original stop SL (R = |E-SL|).
  * Path A — price reaches T before SL: leg1 takes full TP (+RR). DCA never fills.
  * Path B — price reaches SL first: a DCA leg2 fills at SL. The COMBINED target
    becomes the first entry E; the combined stop is SL-R (one more R).
      - bounce to E:  leg1 0  + leg2 +1R  = +1R   (the "0.5R average win")
      - continue to SL-R: leg1 -2R + leg2 -1R = -3R

Outcomes are summed across legs in units of the per-leg risk R, net of cost
(1 round-trip on Path A, 2 on Path B). We compare to the ORIGINAL gated trade
(leg1 only: +RR or -1R). Discovery on train+val; held-out (>=2025-06) reported.

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


def metrics(rs):
    rs = np.asarray(rs, float); n = len(rs)
    if n == 0: return dict(n=0, win=0.0, pf=0.0, avg=0.0, net=0.0, worst=0.0)
    gw = rs[rs > 0].sum(); gl = -rs[rs < 0].sum()
    return dict(n=n, win=float((rs > 0).mean()), pf=(gw/gl) if gl > 1e-9 else 99.0,
                avg=float(rs.mean()), net=float(rs.sum()), worst=float(rs.min()))


def sim(frame, sig, cfg):
    """Return (orig_r, prop_r, path)."""
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
    SL2 = SL - R if long else SL + R

    for j in range(i + 1, min(n - 1, i + HOLD) + 1):
        hi = high[j]; lo = low[j]
        tgt = (hi >= T) if long else (lo <= T)
        stp = (lo <= SL) if long else (hi >= SL)
        if tgt and stp:
            # original ambiguous bar: which first?
            if (bw.high_before_low(op[j], hi, lo)) == long:
                tgt, stp = True, False   # target first for long / stop first handled below
            else:
                tgt, stp = False, True
        if tgt:
            return (rr - cost, rr - cost, "A_target")
        if stp:
            # DCA leg2 fills at SL on bar j; run combined target E / stop SL2 from j..
            orig = -1.0 - cost
            for k in range(j, min(n - 1, j + HOLD) + 1):
                hk = high[k]; lk = low[k]
                btgt = (hk >= E) if long else (lk <= E)        # bounce back to entry
                bstp = (lk <= SL2) if long else (hk >= SL2)    # one more R against
                if btgt and bstp:
                    if (bw.high_before_low(op[k], hk, lk)) == long:
                        btgt, bstp = True, False
                    else:
                        btgt, bstp = False, True
                if btgt:
                    return (orig, 1.0 - 2.0 * cost, "B_bounce")
                if bstp:
                    return (orig, -3.0 - 2.0 * cost, "B_cont")
            # combined timeout: mark both legs to last close
            cl = close[min(n - 1, j + HOLD)]
            leg1 = ((cl - E) if long else (E - cl)) / R
            leg2 = ((cl - SL) if long else (SL - cl)) / R
            return (orig, leg1 + leg2 - 2.0 * cost, "B_timeout")
    # leg1 timeout (never hit T or SL): DCA never fills -> identical
    cl = close[min(n - 1, i + HOLD)]
    r = ((cl - E) if long else (E - cl)) / R - cost
    return (r, r, "timeout")


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
            rows.append({"entry_time": pd.Timestamp(sig.entry_time),
                         "orig_r": r[0], "prop_r": r[1], "path": r[2]})
    d = pd.DataFrame(rows)
    d["entry_time"] = pd.to_datetime(d["entry_time"], utc=True)
    # Risk-normalized variant: cap each SIGNAL's max loss at the original 1R by
    # sizing both legs at 1/3 unit (Path B worst = -3R*(1/3) = -1R). Leg1 winners
    # are then 1/3-size too -> tests whether the edge survives constant risk.
    d["prop_rn"] = d["prop_r"] / 3.0
    d["orig_full"] = d["orig_r"]  # original already 1R risk

    def maxdd(series):  # max drawdown of a cumulative R curve
        eq = np.cumsum(series.to_numpy(dtype=float))
        peak = np.maximum.accumulate(eq)
        return float((peak - eq).max()) if len(eq) else 0.0

    d.to_csv("scripts/output/wolfe_dca_trades.csv", index=False)
    for label, seg in [("DEV(train+val)", d[d.entry_time < VAL_END]),
                       ("OOS held-out(>=25-06)", d[d.entry_time >= VAL_END])]:
        seg = seg.sort_values("entry_time")
        mo = metrics(seg["orig_r"]); mp = metrics(seg["prop_r"]); mrn = metrics(seg["prop_rn"])
        print(f"\n=== {label}  (n={len(seg)}) ===")
        print(f"  ORIGINAL        win={mo['win']:.1%} PF={mo['pf']:.2f} avg_r={mo['avg']:+.3f} netR={mo['net']:+.1f} worst={mo['worst']:+.2f} maxDD={maxdd(seg['orig_r']):.1f}R  ret/DD={mo['net']/max(maxdd(seg['orig_r']),1e-9):.2f}")
        print(f"  PROPOSED(2leg)  win={mp['win']:.1%} PF={mp['pf']:.2f} avg_r={mp['avg']:+.3f} netR={mp['net']:+.1f} worst={mp['worst']:+.2f} maxDD={maxdd(seg['prop_r']):.1f}R  ret/DD={mp['net']/max(maxdd(seg['prop_r']),1e-9):.2f}")
        print(f"  PROPOSED(rn 1/3)win={mrn['win']:.1%} PF={mrn['pf']:.2f} avg_r={mrn['avg']:+.3f} netR={mrn['net']:+.1f} worst={mrn['worst']:+.2f} maxDD={maxdd(seg['prop_rn']):.1f}R  ret/DD={mrn['net']/max(maxdd(seg['prop_rn']),1e-9):.2f}")
        pc = seg["path"].value_counts(normalize=True)
        print("  path mix:", {k: f"{v:.0%}" for k, v in pc.items()})
    # Monthly OOS regime stability (proposed 2-leg net R per month)
    oos = d[d.entry_time >= VAL_END].copy()
    oos["m"] = oos.entry_time.dt.to_period("M")
    print("\nOOS monthly net R (regime stability):")
    for m, g in oos.groupby("m"):
        print(f"  {m}  n={len(g):3d}  orig={g['orig_r'].sum():+6.1f}  prop2leg={g['prop_r'].sum():+6.1f}  bounce/cont={ (g.path=='B_bounce').sum() }/{ (g.path=='B_cont').sum() }")
    print("\nNote: R = per-leg initial risk. PROPOSED(2leg) risks up to 3R/signal; rn=1/3 caps risk ~1R.")


if __name__ == "__main__":
    main()
