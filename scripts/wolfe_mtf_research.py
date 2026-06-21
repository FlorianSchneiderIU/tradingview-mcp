#!/usr/bin/env python3
"""Wolfe MTF context research: can RSI + fib-position on 15m/1h/4h/1d cluster out
losing Wolfe trades?

Hypothesis (user): Wolfe reversals work better when price is already STRETCHED in
the reversal direction (overbought RSI / near the extreme of the swing range), and
worse near mid-range (~50% fib).

Method (overfit-safe):
  * Generate all Wolfe signals across the validated universe with the shipped
    shared config; label each by net R (win = net R > 0).
  * Attach, using ONLY completed bars at/before entry (no look-ahead):
      rsi_{tf}      RSI(14) on tf in {15m,1h,4h,1d}
      pos_{tf}      position in recent swing range: (close-lo)/(hi-lo), 0..1
      favor_rsi_{tf}   direction-aware overbought/oversold-in-favor (short: rsi; long: 100-rsi)
      favor_pos_{tf}   direction-aware stretch toward reversal (short: pos; long: 1-pos)
      mid_dist_{tf}    |pos-0.5| (0 = mid/50% fib, 0.5 = at an extreme)
  * DISCOVER on train+val (entry < 2025-06); VALIDATE on the held-out year (>=2025-06).

Usage:
  python scripts/wolfe_mtf_research.py            # build + analyze
  python scripts/wolfe_mtf_research.py --reuse    # reuse cached feature table
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, importlib
bw = importlib.import_module("backtest_wolfe_wave")

CFG_PATH = "bot/configs/wolfe_wave_shared_v1_configs.json"
CACHE = "scripts/output/wolfe_mtf_signals.csv"
FUNDING_DIR = "scripts/data/funding"
TRAIN_END = pd.Timestamp("2024-12-01", tz="UTC")
VAL_END   = pd.Timestamp("2025-06-01", tz="UTC")   # held-out year starts here
TFS = ["15m", "1h", "4h", "1d"]
LOOKBACK = {"15m": 96, "1h": 168, "4h": 180, "1d": 60}  # recent-swing window per TF


def metrics(d):
    r = np.asarray(d, float); n = len(r)
    if n == 0: return dict(n=0, win=0.0, pf=0.0, avg=0.0, net=0.0)
    gw = r[r > 0].sum(); gl = -r[r < 0].sum()
    return dict(n=n, win=float((r > 0).mean()), pf=(gw/gl) if gl > 1e-9 else 99.0,
                avg=float(r.mean()), net=float(r.sum()))


def build_table(cfg_dict):
    universe = [k for k in cfg_dict if not k.startswith("_")]
    base = {k: v for k, v in cfg_dict[universe[0]].items() if not k.startswith("_")}
    rows = []
    for sym in universe:
        p = f"scripts/data/{sym.lower()}_5m_bybit.csv"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        cfgmap = dict(base); cfgmap["mintick"] = cfg_dict[sym].get("mintick", 0.01)
        try:
            tr = bw.run_backtest(df, bw.WolfeConfig.from_mapping(cfgmap), symbol=sym)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym} backtest ERR {exc}", flush=True); continue
        if not len(tr):
            continue
        # Funding history (8h posts): direction-aware "crowdedness" toward reversal.
        fr_ct = fr_v = None
        fpath = os.path.join(FUNDING_DIR, f"{sym.lower()}.csv")
        if os.path.exists(fpath):
            fdf = pd.read_csv(fpath)
            fr_ct = fdf["ts_ms"].to_numpy() * 1_000_000  # ms -> ns to match entry_ns
            fr_v = fdf["funding_rate"].to_numpy(dtype=float)
        # Precompute MTF rsi + swing range per tf
        tf_arrays = {}
        for tf in TFS:
            rs = bw.add_indicators(bw.resample_ohlc(df, tf), 14, 200, 14)
            L = LOOKBACK[tf]
            hi = rs["high"].rolling(L, min_periods=max(10, L // 3)).max()
            lo = rs["low"].rolling(L, min_periods=max(10, L // 3)).min()
            ct = pd.to_datetime(rs["close_time"], utc=True).astype("int64").to_numpy()
            tf_arrays[tf] = (ct, rs["rsi"].to_numpy(), rs["close"].to_numpy(),
                             hi.to_numpy(), lo.to_numpy())
        et = pd.to_datetime(tr["entry_time"], utc=True)
        for i in range(len(tr)):
            entry_ns = et.iloc[i].value
            direction = tr["direction"].iloc[i]
            row = {"symbol": sym, "entry_time": et.iloc[i], "direction": direction,
                   "r": float(tr["r_multiple_net"].iloc[i]), "score": float(tr["score"].iloc[i])}
            row["win"] = 1 if row["r"] > 0 else 0
            ok = True
            for tf in TFS:
                ct, rsi_a, close_a, hi_a, lo_a = tf_arrays[tf]
                idx = int(np.searchsorted(ct, entry_ns, "right")) - 1
                if idx < 0:
                    ok = False; break
                rsi = float(rsi_a[idx]); c = float(close_a[idx]); h = float(hi_a[idx]); l = float(lo_a[idx])
                pos = (c - l) / (h - l) if (h - l) > 1e-12 else 0.5
                pos = min(max(pos, 0.0), 1.0)
                if not np.isfinite(rsi):
                    ok = False; break
                row[f"rsi_{tf}"] = rsi
                row[f"pos_{tf}"] = pos
                row[f"favor_rsi_{tf}"] = rsi if direction == "short" else (100.0 - rsi)
                row[f"favor_pos_{tf}"] = pos if direction == "short" else (1.0 - pos)
                row[f"mid_dist_{tf}"] = abs(pos - 0.5)
            if not ok:
                continue
            # Funding: last posted rate at/before entry, direction-aware (short
            # benefits from positive funding = crowded longs; long from negative).
            row["funding"] = np.nan
            row["favor_funding"] = np.nan
            row["favor_funding_cum3"] = np.nan   # ~24h
            row["favor_funding_cum9"] = np.nan   # ~3d
            if fr_ct is not None and len(fr_ct):
                k = int(np.searchsorted(fr_ct, entry_ns, "right")) - 1
                if k >= 0:
                    fr = float(fr_v[k])
                    sgn = 1.0 if direction == "short" else -1.0
                    row["funding"] = fr
                    row["favor_funding"] = sgn * fr
                    row["favor_funding_cum3"] = sgn * float(fr_v[max(0, k - 2):k + 1].sum())
                    row["favor_funding_cum9"] = sgn * float(fr_v[max(0, k - 8):k + 1].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def quartile_table(df, feat, label):
    try:
        q = pd.qcut(df[feat], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
    except Exception:
        return
    print(f"\n  [{label}] {feat} quartiles (train+val):")
    for b in q.cat.categories:
        m = metrics(df.loc[q == b, "r"])
        rng = (df.loc[q == b, feat].min(), df.loc[q == b, feat].max())
        print(f"    {b:9} [{rng[0]:.1f}..{rng[1]:.1f}]  n={m['n']:4d} win={m['win']:.1%} PF={m['pf']:.2f} avg_r={m['avg']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(CFG_PATH))

    if args.reuse and os.path.exists(CACHE):
        df = pd.read_csv(CACHE, parse_dates=["entry_time"])
        print(f"reused {CACHE}: {len(df)} signals")
    else:
        os.makedirs("scripts/output", exist_ok=True)
        print("building MTF feature table across universe ...", flush=True)
        df = build_table(cfg)
        df.to_csv(CACHE, index=False)
        print(f"wrote {CACHE}: {len(df)} signals")

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    dev = df[df.entry_time < VAL_END].copy()       # train+val (discovery)
    oos = df[df.entry_time >= VAL_END].copy()       # held-out (validation)
    print(f"\nbaseline  dev(train+val): {metrics(dev['r'])}")
    print(f"baseline  oos(held-out):  {metrics(oos['r'])}")

    # 1) Does stretch help? quartile WR/PF on dev for each direction-aware feature.
    print("\n=== STRETCH HYPOTHESIS (dev only) — higher favor_* should = better ===")
    for tf in TFS:
        quartile_table(dev, f"favor_rsi_{tf}", "RSI-in-favor")
        quartile_table(dev, f"favor_pos_{tf}", "fib-pos-in-favor")
    print("\n=== MID-RANGE HYPOTHESIS (dev only) — higher mid_dist should = better ===")
    for tf in TFS:
        quartile_table(dev, f"mid_dist_{tf}", "dist-from-50%")

    # FUNDING (incremental — table already has the HTF gate applied).
    print("\n=== FUNDING HYPOTHESIS (dev only) — higher favor_funding (crowded against the prior move) should = better ===")
    devf = dev.dropna(subset=["favor_funding"])
    print(f"  (rows with funding: {len(devf)}/{len(dev)})")
    for feat in ["favor_funding", "favor_funding_cum3", "favor_funding_cum9"]:
        quartile_table(devf, feat, "funding")

    # 2) Simple candidate filters discovered on dev, validated on oos.
    print("\n=== CANDIDATE FILTERS (discovered on dev, VALIDATED on held-out) ===")
    cands = {
        # EXCLUSION filters: drop over-stretched HTF setups (worst dev buckets), keep volume.
        "drop favor_pos_4h>=0.8":          lambda d: d["favor_pos_4h"] < 0.8,
        "drop favor_pos_1h>=0.9":          lambda d: d["favor_pos_1h"] < 0.9,
        "drop favor_rsi_4h>=62":           lambda d: d["favor_rsi_4h"] < 62,
        "drop favor_rsi_1d>=58":           lambda d: d["favor_rsi_1d"] < 58,
        "drop pos_4h>=0.8 OR rsi_1d>=58":  lambda d: (d["favor_pos_4h"] < 0.8) & (d["favor_rsi_1d"] < 58),
        "drop pos_4h>=0.8 OR rsi_4h>=62 OR rsi_1d>=58":
            lambda d: (d["favor_pos_4h"] < 0.8) & (d["favor_rsi_4h"] < 62) & (d["favor_rsi_1d"] < 58),
        "keep favor_pos_4h in [0.0,0.8) & favor_pos_1h<0.9":
            lambda d: (d["favor_pos_4h"] < 0.8) & (d["favor_pos_1h"] < 0.9),
        # FUNDING filters (NaN funding treated as pass so no-funding rows aren't dropped).
        "drop favor_funding<0":        lambda d: d["favor_funding"].fillna(0) >= 0,
        "drop favor_funding_cum9<0":   lambda d: d["favor_funding_cum9"].fillna(0) >= 0,
        "favor_funding>=1e-4":         lambda d: d["favor_funding"].fillna(1e-4) >= 1e-4,
        "drop favor_funding_cum3<0":   lambda d: d["favor_funding_cum3"].fillna(0) >= 0,
    }
    base_oos = metrics(oos["r"])
    print(f"  {'filter':42} | {'dev WR/PF/n':>20} | {'OOS WR/PF/n (kept%)':>26}")
    print(f"  {'(no filter)':42} | {dev['win'].mean()*100:5.1f}% {metrics(dev['r'])['pf']:.2f} {len(dev):5d} | "
          f"{base_oos['win']*100:5.1f}% {base_oos['pf']:.2f} {base_oos['n']:5d} (100%)")
    for name, fn in cands.items():
        dsel = dev[fn(dev)]; osel = oos[fn(oos)]
        dm = metrics(dsel["r"]); om = metrics(osel["r"])
        kept = 100.0 * om["n"] / max(base_oos["n"], 1)
        print(f"  {name:42} | {dm['win']*100:5.1f}% {dm['pf']:.2f} {dm['n']:5d} | "
              f"{om['win']*100:5.1f}% {om['pf']:.2f} {om['n']:5d} ({kept:.0f}%)")


if __name__ == "__main__":
    main()
