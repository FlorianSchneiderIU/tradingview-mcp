#!/usr/bin/env python3
"""Two fresh BTC/ETH concepts with a real microstructure rationale on 24/7 crypto,
tested the disciplined way (no ML, small param grid, dev -> untouched holdout + a
random-time null to prove the SIGNAL adds value over entering anytime).

  A) VWAP mean-reversion: fade extreme rolling-VWAP deviations back to VWAP.
  B) Funding mean-reversion: fade crowded funding (longs pay -> short; shorts pay -> long).

Cost = fees+slippage (8 bps/side). Offline: scripts/data/<sym>_5m_bybit.csv (+ funding/).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)

FEE_BPS_SIDE = 8.0
HOLDOUT_DAYS = 365
DATA = "scripts/data/{}_5m_bybit.csv"
FUND = "scripts/data/funding/{}.csv"
BARS_PER_DAY = 288   # 5m


def load_5m(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(DATA.format(symbol.lower()))
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("open_time").reset_index(drop=True)
    # ATR(14) on 5m
    pc = df["close"].shift(1)
    tr = pd.concat([(df.high - df.low), (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return df


def simulate(df: pd.DataFrame, entries: list[dict]) -> pd.DataFrame:
    """entries: {idx, dir(+1/-1), stop, target, max_hold}. Entry at NEXT bar open. Cost-aware."""
    o, h, l, c = df.open.to_numpy(), df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    n = len(df)
    rows = []
    for e in entries:
        i = e["idx"] + 1
        if i >= n:
            continue
        d, stop, tgt, mh = e["dir"], e["stop"], e["target"], e["max_hold"]
        entry = o[i]
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        exit_px, end = c[min(i + mh, n - 1)], min(i + mh, n - 1)
        for j in range(i, min(i + mh, n - 1) + 1):
            if d > 0:
                if l[j] <= stop:
                    exit_px, end = stop, j; break
                if h[j] >= tgt:
                    exit_px, end = tgt, j; break
            else:
                if h[j] >= stop:
                    exit_px, end = stop, j; break
                if l[j] <= tgt:
                    exit_px, end = tgt, j; break
        gross = (exit_px - entry) / risk if d > 0 else (entry - exit_px) / risk
        cost_r = (entry * 2 * FEE_BPS_SIDE / 1e4) / risk
        rows.append({"entry_time": df.open_time.iloc[i], "dir": d, "r": gross - cost_r, "bars": end - i})
    return pd.DataFrame(rows)


def pooled(t: pd.DataFrame) -> dict:
    if t.empty:
        return dict(trades=0, pf=0.0, net_r=0.0, avg_r=0.0, win=0.0)
    r = t["r"].to_numpy(float)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(trades=len(r), pf=(gp / gl if gl > 0 else np.inf), net_r=r.sum(), avg_r=r.mean(), win=(r > 0).mean())


# ---------- concept A: VWAP mean-reversion ----------
def vwap_signals(df: pd.DataFrame, *, W: int, k: float, stop_atr: float, max_hold: int) -> list[dict]:
    pv = (df.close * df.volume).rolling(W).sum()
    vv = df.volume.rolling(W).sum()
    vwap = (pv / vv).to_numpy()
    dev = df.close.to_numpy() - vwap
    sd = pd.Series(dev).rolling(W).std().to_numpy()
    z = np.where(sd > 0, dev / sd, np.nan)
    atr = df.atr.to_numpy(); close = df.close.to_numpy()
    out, block_until = [], -1
    for i in range(W, len(df) - 1):
        if i <= block_until or not np.isfinite(z[i]) or not np.isfinite(atr[i]):
            continue
        if z[i] > k:        # stretched above -> short back to vwap
            d = -1
        elif z[i] < -k:     # stretched below -> long back to vwap
            d = +1
        else:
            continue
        entry = close[i]
        stop = entry + d * (-stop_atr) * atr[i] if False else (entry - d * stop_atr * atr[i])
        # stop is AGAINST the trade: long stop below, short stop above
        stop = entry - stop_atr * atr[i] if d > 0 else entry + stop_atr * atr[i]
        out.append({"idx": i, "dir": d, "stop": stop, "target": vwap[i], "max_hold": max_hold})
        block_until = i + max_hold
    return out


# ---------- concept B: funding mean-reversion ----------
def funding_signals(df: pd.DataFrame, symbol: str, *, N: int, k: float, stop_atr: float,
                    rr: float, max_hold: int) -> list[dict]:
    try:
        f = pd.read_csv(FUND.format(symbol.lower()))
    except FileNotFoundError:
        return []
    f["t"] = pd.to_datetime(f["ts_ms"].astype("int64"), unit="ms", utc=True)
    f = f.sort_values("t").reset_index(drop=True)
    f["z"] = (f.funding_rate - f.funding_rate.rolling(N).mean()) / f.funding_rate.rolling(N).std()
    f = f.dropna(subset=["z"])
    # map each funding settlement to the first 5m bar at/after it
    atr = df.atr.to_numpy(); close = df.close.to_numpy()
    out = []
    for _, row in f.iterrows():
        if abs(row["z"]) < k:
            continue
        i = int(df.open_time.searchsorted(row["t"]))
        if i <= 0 or i >= len(df) - 1 or not np.isfinite(atr[i]):
            continue
        d = -1 if row["z"] > 0 else +1     # crowded longs (high funding) -> short
        entry = close[i]
        stop = entry - stop_atr * atr[i] if d > 0 else entry + stop_atr * atr[i]
        tgt = entry + d * rr * stop_atr * atr[i]
        out.append({"idx": i, "dir": d, "stop": stop, "target": tgt, "max_hold": max_hold})
    return out


def null_random_time(df: pd.DataFrame, real: list[dict], n_iter: int, hstart) -> float:
    """Same count + same direction-rule + same stop/target geometry, but at RANDOM bars.
    Returns 95th pctile of holdout net_r under random timing."""
    if not real:
        return np.inf
    n_hold = sum(1 for e in real if df.open_time.iloc[min(e["idx"] + 1, len(df) - 1)] >= hstart)
    if n_hold == 0:
        return np.inf
    sample = real[0]
    mh = sample["max_hold"]
    atr = df.atr.to_numpy(); close = df.close.to_numpy()
    # template: fade toward a rolling-24h vwap sign so random entries use the SAME rule shape
    pv = (df.close * df.volume).rolling(BARS_PER_DAY).sum(); vv = df.volume.rolling(BARS_PER_DAY).sum()
    vwap = (pv / vv).to_numpy()
    hold_idx = np.where((df.open_time >= hstart).to_numpy())[0]
    hold_idx = hold_idx[(hold_idx > BARS_PER_DAY) & (hold_idx < len(df) - mh - 1)]
    nets = []
    for _ in range(n_iter):
        pick = rng.choice(hold_idx, size=min(n_hold, len(hold_idx)), replace=False)
        ents = []
        for i in pick:
            if not np.isfinite(atr[i]) or not np.isfinite(vwap[i]):
                continue
            d = -1 if close[i] > vwap[i] else +1
            stop = close[i] - sample.get("_stop_atr", 1.5) * atr[i] if d > 0 else close[i] + sample.get("_stop_atr", 1.5) * atr[i]
            ents.append({"idx": int(i), "dir": d, "stop": stop, "target": vwap[i], "max_hold": mh})
        nets.append(pooled(simulate(df, ents))["net_r"])
    return float(np.percentile(nets, 95))


def evaluate(df, signals, hstart, label):
    t = simulate(df, signals)
    if t.empty:
        return None
    dev = pooled(t[t.entry_time < hstart]); hold = pooled(t[t.entry_time >= hstart])
    return {"label": label, "dev": dev, "hold": hold, "trades": t}


def forward_probe(df, sig, hstart, label):
    """Exit-agnostic edge test: mean FADE forward return (bps, net of 16bps round-trip)
    at several horizons, dev vs holdout. Positive => reversion edge; negative => extremes
    EXTEND (momentum). sig entries carry 'dir' = the fade direction we'd take."""
    close = df.close.to_numpy(); n = len(df)
    ot = df.open_time
    print(f"\n  [forward-return probe] {label}  (mean fade return, bps, net of 16bps)")
    print(f"   {'horizon':<10}{'dev_n':>7}{'dev_bps':>9}{'dev_hit':>9}{'hold_n':>8}{'hold_bps':>10}{'hold_hit':>9}")
    for h in (6, 12, 24, 48, 96):
        rows = []
        for e in sig:
            i = e["idx"]
            if i + h >= n:
                continue
            ret = (close[i + h] - close[i]) / close[i]
            fade = e["dir"] * ret * 1e4 - 16.0     # bps, net of round-trip cost
            rows.append((ot.iloc[i], fade))
        if not rows:
            continue
        r = pd.DataFrame(rows, columns=["t", "bps"])
        d = r[r.t < hstart]; ho = r[r.t >= hstart]
        print(f"   +{h:<9}{len(d):>7}{d.bps.mean():>9.1f}{(d.bps>0).mean()*100:>8.0f}%"
              f"{len(ho):>8}{ho.bps.mean():>10.1f}{(ho.bps>0).mean()*100:>8.0f}%")


def main():
    for symbol in ("BTCUSDT", "ETHUSDT"):
        print(f"\n{'='*78}\n{symbol}\n{'='*78}", flush=True)
        df = load_5m(symbol)
        hstart = df.open_time.max() - pd.Timedelta(days=HOLDOUT_DAYS)

        # exit-agnostic edge probes (does the extreme revert or extend?)
        forward_probe(df, vwap_signals(df, W=576, k=2.5, stop_atr=2.0, max_hold=288),
                      hstart, "VWAP z>2.5 fade")
        forward_probe(df, funding_signals(df, symbol, N=30, k=2.0, stop_atr=2.0, rr=1.0, max_hold=288),
                      hstart, "funding z>2.0 fade")

        # ---- Concept A: VWAP MR grid ----
        print("\n[A] VWAP mean-reversion  (dev -> holdout)")
        print(f"{'W/k/stopATR/hold':<26}{'dev_n':>6}{'dev_PF':>7}{'dev_R':>8}{'h_n':>5}{'h_PF':>6}{'h_R':>8}{'h_win':>7}")
        bestA = None
        for W in (288, 576):
            for k in (2.0, 2.5, 3.0):
                for s in (1.0, 1.5, 2.0):
                    for H in (96, 288):
                        sig = vwap_signals(df, W=W, k=k, stop_atr=s, max_hold=H)
                        for e in sig:
                            e["_stop_atr"] = s
                        r = evaluate(df, sig, hstart, f"{W}/{k}/{s}/{H}")
                        if r and r["dev"]["trades"] >= 80:
                            if bestA is None or r["dev"]["avg_r"] > bestA["dev"]["avg_r"]:
                                bestA = r
        if bestA:
            for r in [bestA]:
                d, h = r["dev"], r["hold"]
                print(f"{r['label']:<26}{d['trades']:>6}{d['pf']:>7.2f}{d['net_r']:>8.1f}"
                      f"{h['trades']:>5}{h['pf']:>6.2f}{h['net_r']:>8.1f}{h['win']*100:>6.0f}%  <- best-by-dev")
            # null on the best config's holdout
            W, k, s, H = bestA["label"].split("/");
            sig = vwap_signals(df, W=int(W), k=float(k), stop_atr=float(s), max_hold=int(H))
            for e in sig: e["_stop_atr"] = float(s)
            p95 = null_random_time(df, sig, 100, hstart)
            print(f"   holdout net_r={bestA['hold']['net_r']:.1f}  random-time p95={p95:.1f}  "
                  f"beats_null={bestA['hold']['net_r'] > p95}")

        # ---- Concept B: funding MR grid ----
        print("\n[B] Funding mean-reversion  (dev -> holdout)")
        print(f"{'N/k/stopATR/rr/hold':<26}{'dev_n':>6}{'dev_PF':>7}{'dev_R':>8}{'h_n':>5}{'h_PF':>6}{'h_R':>8}{'h_win':>7}")
        bestB = None
        for N in (30, 90):
            for k in (1.0, 1.5, 2.0):
                for s in (1.5, 2.5):
                    for rr in (1.0, 1.5, 2.0):
                        for H in (288, 576):
                            sig = funding_signals(df, symbol, N=N, k=k, stop_atr=s, rr=rr, max_hold=H)
                            r = evaluate(df, sig, hstart, f"{N}/{k}/{s}/{rr}/{H}")
                            if r and r["dev"]["trades"] >= 40:
                                if bestB is None or r["dev"]["avg_r"] > bestB["dev"]["avg_r"]:
                                    bestB = r
        if bestB:
            d, h = bestB["dev"], bestB["hold"]
            print(f"{bestB['label']:<26}{d['trades']:>6}{d['pf']:>7.2f}{d['net_r']:>8.1f}"
                  f"{h['trades']:>5}{h['pf']:>6.2f}{h['net_r']:>8.1f}{h['win']*100:>6.0f}%  <- best-by-dev")
        else:
            print("  (no funding config met min trade count)")


if __name__ == "__main__":
    main()
