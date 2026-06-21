#!/usr/bin/env python3
"""Fetch the full Wolfe universe and re-validate the SHARED config.

1. Fetch 5m klines (public Bybit, no auth) for every symbol in
   wolfe_wave_configs.json that lacks a cached CSV; write to scripts/data/.
2. Run the shared, quality-gated config across the FULL universe, pooled, with a
   held-out year (>=2025-06, incl. the live-failure window) used only to report.
3. Rewrite bot/configs/wolfe_wave_shared_v1_configs.json with provenance + a
   per-symbol held-out breakdown.

Run from repo root:  python scripts/revalidate_wolfe_universe.py [--since 2022-01-01]
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import pandas as pd, numpy as np, importlib
bw = importlib.import_module("backtest_wolfe_wave")

DATA = os.path.join(REPO, "scripts", "data")
DEP_PATH = os.path.join(REPO, "bot", "configs", "wolfe_wave_configs.json")
OUT_PATH = os.path.join(REPO, "bot", "configs", "wolfe_wave_shared_v1_configs.json")

SHARED = {"allow_longs":True,"allow_shorts":True,"atr_length":14,"ema_length":200,
"exec_tf":"5m","pattern_tf":"15m","fee_aware_stop":True,"fee_bps_side":5.5,"slippage_bps_side":1.0,
"long_max_rsi":58.0,"short_min_rsi":42.0,"max_entry_risk_pct":3.5,"min_entry_risk_pct":0.05,
"max_entry_wait_bars":36,"max_epa_slope_atr":0.65,"max_fee_to_price_risk":0.25,"max_hold_bars":144,
"max_p4_retrace":0.95,"min_p4_retrace":0.25,"max_p5_break_atr":2.2,"min_p5_break_atr":0.05,
"max_pattern_bars":220,"min_pattern_bars":12,"max_rr":5.0,"min_rr":1.5,"max_stop_atr":4.0,
"min_stop_atr":0.35,"max_time_ratio":3.8,"min_score":66.0,"min_volume_ratio":0.0,
"one_trade_at_a_time":True,"pivot_confirm_window":3,"pivot_method":"fractal","pivot_source":"close",
"pivot_window":12,"regime_filter":"none","require_reclaim":True,"require_reclaim_vs_p5":True,
"risk_fraction":0.01,"rsi_length":14,"stop_atr_buffer":0.3,"target_projection_bars":30,
"trend_filter":"rsi","zigzag_atr_mult":1.0,
# HTF over-extension gate (validated): skip reversals already stretched >=80% of the
# recent 4h swing range in the trade direction. Lifts OOS WR/PF, keeps ~77% of signals.
"max_favor_pos_htf":0.8,"htf_extension_tf":"4h","htf_extension_lookback":180}

TRAIN_END = pd.Timestamp("2024-12-01", tz="UTC")
VAL_END   = pd.Timestamp("2025-06-01", tz="UTC")
MIN_HOLDOUT_TRADES = 10   # (reporting only) held-out trade count threshold
MIN_TOTAL_TRADES   = 30   # per-symbol inclusion floor on TOTAL trades (whole period)


def met(r):
    r = np.asarray(r, float); n = len(r)
    if n == 0: return dict(n=0, avg=0.0, win=0.0, pf=0.0, net=0.0)
    gw = r[r > 0].sum(); gl = -r[r < 0].sum()
    return dict(n=n, avg=float(r.mean()), win=float((r > 0).mean()),
                pf=(gw/gl) if gl > 1e-9 else 99.0, net=float(r.sum()))


def fetch_missing(symbols, since):
    start = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    fetched, failed = [], []
    for i, sym in enumerate(symbols, 1):
        path = os.path.join(DATA, f"{sym.lower()}_5m_bybit.csv")
        if os.path.exists(path):
            continue
        try:
            t0 = time.time()
            df = bw.fetch_bybit_klines(sym, "5m", start, end)
            df.to_csv(path, index=False)
            fetched.append(sym)
            print(f"  [{i}/{len(symbols)}] {sym}: {len(df)} bars "
                  f"{str(df['open_time'].iloc[0])[:10]}..{str(df['open_time'].iloc[-1])[:10]} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append((sym, str(exc)))
            print(f"  [{i}/{len(symbols)}] {sym}: FETCH FAILED {exc}", flush=True)
    return fetched, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2022-01-01")
    ap.add_argument("--dca-k", type=float, default=0.0,
                    help="If >0, enable DCA second leg with combined stop SL-(k*R).")
    ap.add_argument("--out", default=OUT_PATH, help="Output config path.")
    args = ap.parse_args()
    out_path = args.out
    if args.dca_k > 0:
        SHARED["dca_enabled"] = True
        SHARED["dca_stop_frac_k"] = float(args.dca_k)
        print(f"DCA ENABLED: k={args.dca_k} -> combined stop SL-{args.dca_k}R, target=first entry")

    dep = json.load(open(DEP_PATH))
    universe = [k for k in dep if not k.startswith("_")]
    print(f"Universe: {len(universe)} symbols. Fetching missing since {args.since} ...", flush=True)
    fetched, failed = fetch_missing(universe, args.since)
    print(f"Fetched {len(fetched)} new, {len(failed)} failed.\n", flush=True)

    # Validate shared config across every universe symbol that now has data.
    allt, per_sym, per_sym_total = [], {}, {}
    used = []
    for sym in universe:
        path = os.path.join(DATA, f"{sym.lower()}_5m_bybit.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            cfgmap = dict(SHARED); cfgmap["mintick"] = dep[sym].get("mintick", 0.01)
            tr = bw.run_backtest(df, bw.WolfeConfig.from_mapping(cfgmap), symbol=sym)
            used.append(sym)
            if len(tr):
                t = tr.assign(symbol=sym)[["entry_time", "r_multiple_net", "symbol"]].copy()
                t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
                allt.append(t)
                oos = t[t.entry_time >= VAL_END]["r_multiple_net"]
                per_sym[sym] = met(oos)
                per_sym_total[sym] = len(t)
        except Exception as exc:  # noqa: BLE001
            print(f"  validate {sym} ERR {exc}", flush=True)

    pooled = pd.concat(allt, ignore_index=True)
    tr_m = met(pooled[pooled.entry_time < TRAIN_END]["r_multiple_net"])
    va_m = met(pooled[(pooled.entry_time >= TRAIN_END) & (pooled.entry_time < VAL_END)]["r_multiple_net"])
    oo_m = met(pooled[pooled.entry_time >= VAL_END]["r_multiple_net"])

    print(f"\n=== SHARED config across {len(used)} symbols, pooled {len(pooled)} trades ===")
    for lbl, m in [("train(<24-12)", tr_m), ("val(24-12..25-06)", va_m), ("OOS held-out(>=25-06)", oo_m)]:
        print(f"  {lbl:24} n={m['n']:5d} win={m['win']:.1%} PF={m['pf']:.2f} avg_r={m['avg']:+.3f} netR={m['net']:+.1f}")

    # Include by TOTAL trade count (a symbol participates if it has enough trades
    # over the whole period). We deliberately do NOT gate on the held-out window:
    # filtering by held-out count or sign contaminates the holdout (curve-fitting),
    # and would wrongly drop liquid majors (e.g. BTC) whose held-out frequency is
    # low after the over-extension gate. Negative-on-holdout symbols are kept (noise).
    include = [s for s in used if per_sym_total.get(s, 0) >= MIN_TOTAL_TRADES]
    drop_thin = [s for s in used if per_sym_total.get(s, 0) < MIN_TOTAL_TRADES]
    drop_neg = [s for s in include if per_sym.get(s, {}).get("net", 0) <= 0]  # reported only
    print(f"\nper-symbol inclusion: include={len(include)} (by TOTAL trades>= {MIN_TOTAL_TRADES})  "
          f"drop(thin)={len(drop_thin)}  [of included, {len(drop_neg)} net-neg on holdout, kept]")
    print("  worst held-out symbols:", sorted(((per_sym[s]['net'], s) for s in include if s in per_sym))[:8])

    # Recompute pooled OOS restricted to the INCLUDE set (the deployable portfolio).
    inc_oos = pooled[(pooled.entry_time >= VAL_END) & (pooled.symbol.isin(include))]["r_multiple_net"]
    inc_m = met(inc_oos)
    print(f"\nINCLUDE-set held-out OOS: n={inc_m['n']} win={inc_m['win']:.1%} PF={inc_m['pf']:.2f} avg_r={inc_m['avg']:+.3f} netR={inc_m['net']:+.1f}")

    out = {"_quality_profile": {"mode": "shadow", "name": "wolfe_mtf_v1"},
           "_validation": {
               "method": "pooled walk-forward; held-out year >=2025-06 (incl. live-failure window)",
               "selected_on": "shared params (not per-symbol); symbols included if TOTAL trades>=%d (held-out window not used for inclusion)" % MIN_TOTAL_TRADES,
               "universe_symbols": len(used),
               "included_symbols": len(include),
               "pooled_all": {"train": {k: round(v,3) for k,v in tr_m.items()},
                              "validation": {k: round(v,3) for k,v in va_m.items()},
                              "oos_holdout": {k: round(v,3) for k,v in oo_m.items()}},
               "oos_holdout_included": {k: round(v,3) for k,v in inc_m.items()},
               "dca_enabled": bool(SHARED.get("dca_enabled", False)),
               "dca_stop_frac_k": SHARED.get("dca_stop_frac_k"),
               "fetch_failed": [s for s, _ in failed],
           }}
    for sym in include:
        e = dict(SHARED); e["mintick"] = dep[sym].get("mintick", 0.01)
        e["_oos_holdout"] = {k: round(v, 3) for k, v in per_sym[sym].items()}
        out[sym] = e
    json.dump(out, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"\nwrote {out_path}: {len(include)} included symbols (of {len(used)} validated)")


if __name__ == "__main__":
    main()
