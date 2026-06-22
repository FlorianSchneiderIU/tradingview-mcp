#!/usr/bin/env python3
"""Scan VWAP- and funding-mean-reversion forward returns across the ALT universe (53
symbols with both 5m + funding data). BTC/ETH intraday MR was sub-cost; the hypothesis
is that less-efficient alts have a LARGER reversion that clears the ~16bps round-trip.

Exit-agnostic: mean FADE forward return (bps), POOLED across the universe (selection-free)
dev vs untouched holdout, then per-symbol dispersion. Gross + net-of-16bps reported."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.research_meanrev_funding import load_5m, vwap_signals, funding_signals, HOLDOUT_DAYS  # noqa: E402

COST_BPS = 16.0
HORIZONS = (6, 12, 24, 48, 96)
UNI = ("1000BONKUSDT,1000PEPEUSDT,AAVEUSDT,ADAUSDT,ALGOUSDT,APEUSDT,APTUSDT,ARBUSDT,ARKMUSDT,"
       "ATOMUSDT,AVAXUSDT,BCHUSDT,BNBUSDT,BTCUSDT,DASHUSDT,DOGEUSDT,DOTUSDT,DYDXUSDT,EIGENUSDT,"
       "ENAUSDT,ETHUSDT,FILUSDT,GALAUSDT,HBARUSDT,HNTUSDT,ICPUSDT,IDUSDT,INJUSDT,JTOUSDT,LINKUSDT,"
       "LTCUSDT,MEMEUSDT,NEARUSDT,ONDOUSDT,OPUSDT,ORDIUSDT,PENDLEUSDT,PENGUUSDT,POLUSDT,PORTALUSDT,"
       "SEIUSDT,SOLUSDT,STGUSDT,STRKUSDT,STXUSDT,SUIUSDT,TRUMPUSDT,UNIUSDT,WUSDT,XLMUSDT,XMRUSDT,"
       "XRPUSDT,ZECUSDT").split(",")


def fwd_rows(df, sig, hstart):
    """Return list of (is_holdout, dir, {h: gross_bps})."""
    close = df.close.to_numpy(); n = len(df); ot = df.open_time
    rows = []
    for e in sig:
        i = e["idx"]
        rec = {}
        for h in HORIZONS:
            if i + h < n:
                rec[h] = e["dir"] * (close[i + h] - close[i]) / close[i] * 1e4
        if rec:
            rows.append((ot.iloc[i] >= hstart, rec))
    return rows


def summarize(pool, title):
    print(f"\n=== {title}: POOLED across universe (gross fade bps / net of {COST_BPS:.0f}bps) ===")
    print(f"{'horizon':<9}{'dev_n':>8}{'dev_gross':>10}{'dev_net':>9}{'dev_hit':>8}"
          f"{'hold_n':>8}{'hold_gross':>11}{'hold_net':>9}{'hold_hit':>9}")
    for h in HORIZONS:
        dv = np.array([r[1][h] for r in pool if not r[0] and h in r[1]])
        ho = np.array([r[1][h] for r in pool if r[0] and h in r[1]])
        if len(dv) == 0 or len(ho) == 0:
            continue
        print(f"+{h:<8}{len(dv):>8}{dv.mean():>10.1f}{dv.mean()-COST_BPS:>9.1f}{(dv>0).mean()*100:>7.0f}%"
              f"{len(ho):>8}{ho.mean():>11.1f}{ho.mean()-COST_BPS:>9.1f}{(ho>0).mean()*100:>8.0f}%")


def main():
    vwap_pool, fund_pool = [], []
    per_sym = []   # (sym, vwap_hold_net@24, fund_hold_net@24)
    H = 24
    for k, sym in enumerate(UNI, 1):
        try:
            df = load_5m(sym)
        except Exception as exc:
            print(f"  skip {sym}: {exc}", flush=True); continue
        hstart = df.open_time.max() - pd.Timedelta(days=HOLDOUT_DAYS)
        vsig = vwap_signals(df, W=576, k=2.5, stop_atr=2.0, max_hold=288)
        fsig = funding_signals(df, sym, N=30, k=2.0, stop_atr=2.0, rr=1.0, max_hold=288)
        vr, fr = fwd_rows(df, vsig, hstart), fwd_rows(df, fsig, hstart)
        vwap_pool += vr; fund_pool += fr

        def hold_net(rows):
            x = np.array([r[1][H] for r in rows if r[0] and H in r[1]])
            return (x.mean() - COST_BPS, len(x)) if len(x) else (np.nan, 0)
        def dev_net(rows):
            x = np.array([r[1][H] for r in rows if not r[0] and H in r[1]])
            return (x.mean() - COST_BPS) if len(x) else np.nan
        vhn, vhc = hold_net(vr); fhn, fhc = hold_net(fr)
        per_sym.append((sym, dev_net(vr), vhn, vhc, dev_net(fr), fhn, fhc))
        print(f"  [{k}/{len(UNI)}] {sym}: vwap_sig={len(vsig)} fund_sig={len(fsig)}", flush=True)

    summarize(vwap_pool, "A) VWAP z>2.5 fade")
    summarize(fund_pool, "B) funding z>2.0 fade")

    print(f"\n=== per-symbol net fade @+{H} bars (net of {COST_BPS:.0f}bps), ranked by VWAP holdout ===")
    print(f"{'symbol':<14}{'vwap_dev':>10}{'vwap_hold':>11}{'n':>5}   {'fund_dev':>10}{'fund_hold':>11}{'n':>5}")
    for s, vd, vh, vc, fd, fh, fc in sorted(per_sym, key=lambda x: (-(x[2] if np.isfinite(x[2]) else -1e9))):
        print(f"{s:<14}{vd:>10.1f}{vh:>11.1f}{vc:>5}   {fd:>10.1f}{fh:>11.1f}{fc:>5}")
    vpos = sum(1 for r in per_sym if np.isfinite(r[1]) and np.isfinite(r[2]) and r[1] > 0 and r[2] > 0)
    fpos = sum(1 for r in per_sym if np.isfinite(r[4]) and np.isfinite(r[5]) and r[4] > 0 and r[5] > 0)
    print(f"\nVWAP net-positive in BOTH dev & holdout: {vpos}/{len(per_sym)}")
    print(f"funding net-positive in BOTH dev & holdout: {fpos}/{len(per_sym)}")


if __name__ == "__main__":
    main()
