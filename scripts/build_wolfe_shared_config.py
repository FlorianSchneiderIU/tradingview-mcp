#!/usr/bin/env python3
"""Build + validate a single SHARED Wolfe config across the universe.

Rationale: the deployed config tuned min_score (and 40+ params) PER symbol via
best-of-500 selection -> overfit, and early-deployed loose thresholds (min_score
48) traded low-quality signals that lost live. A single shared, quality-gated
config validated on POOLED trades with a held-out year is far more robust.

Selection was done on pooled train+val; the held-out year (>=2025-06, which
includes the live-failure window) is reported, NOT used to pick params.
Writes bot/configs/wolfe_wave_shared_v1_configs.json with provenance.
"""
from __future__ import annotations
import sys, glob, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import pandas as pd, numpy as np, importlib
bw=importlib.import_module("backtest_wolfe_wave")

# Shared params (validated variant: regime off, min_score 66). Pattern/exec/risk
# match the deployed strategy; only mintick is taken per-symbol for live rounding.
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
"trend_filter":"rsi","zigzag_atr_mult":1.0}

dep=json.load(open("bot/configs/wolfe_wave_configs.json"))
files=sorted(glob.glob("scripts/data/*_5m_bybit.csv"))
TRAIN_END=pd.Timestamp("2024-12-01",tz="UTC"); VAL_END=pd.Timestamp("2025-06-01",tz="UTC")
def met(r):
    r=np.asarray(r,float); n=len(r)
    if n==0: return dict(n=0,avg=0,win=0,pf=0,net=0)
    gw=r[r>0].sum(); gl=-r[r<0].sum()
    return dict(n=n,avg=float(r.mean()),win=float((r>0).mean()),pf=(gw/gl) if gl>1e-9 else 99.0,net=float(r.sum()))

allt=[]; symbols=[]
for f in files:
    sym=os.path.basename(f)[:-len("_5m_bybit.csv")].upper()
    if sym not in dep: continue   # only universe symbols
    mintick=dep[sym].get("mintick",0.01)
    cfgmap=dict(SHARED); cfgmap["mintick"]=mintick
    try:
        df=pd.read_csv(f); cfg=bw.WolfeConfig.from_mapping(cfgmap)
        tr=bw.run_backtest(df,cfg,symbol=sym)
        symbols.append(sym)
        if len(tr): allt.append(tr.assign(symbol=sym)[["entry_time","r_multiple_net","symbol"]])
    except Exception as e: print("ERR",sym,e)
t=pd.concat(allt,ignore_index=True); t["entry_time"]=pd.to_datetime(t["entry_time"],utc=True)
tr_m=met(t[t.entry_time<TRAIN_END]["r_multiple_net"])
va_m=met(t[(t.entry_time>=TRAIN_END)&(t.entry_time<VAL_END)]["r_multiple_net"])
oo_m=met(t[t.entry_time>=VAL_END]["r_multiple_net"])
print(f"SHARED config across {len(symbols)} symbols, pooled {len(t)} trades:")
for lbl,m in [("train(<24-12)",tr_m),("val(24-12..25-06)",va_m),("OOS(>=25-06, held-out)",oo_m)]:
    print(f"  {lbl:24} n={m['n']:5d} win={m['win']:.1%} PF={m['pf']:.2f} avg_r={m['avg']:+.3f} netR={m['net']:+.1f}")

out={"_quality_profile":{"mode":"shadow","name":"wolfe_mtf_v1"},
     "_validation":{"method":"pooled walk-forward, held-out year >=2025-06 (incl. live-failure window)",
        "selected_on":"pooled train+val","shared_params":True,
        "train":{k:round(v,3) for k,v in tr_m.items()},
        "validation":{k:round(v,3) for k,v in va_m.items()},
        "oos_holdout":{k:round(v,3) for k,v in oo_m.items()},
        "note":"Single shared config (not per-symbol) to avoid overfitting. min_score=66 quality gate."}}
for sym in symbols:
    e=dict(SHARED); e["mintick"]=dep[sym].get("mintick",0.01); out[sym]=e
json.dump(out,open("bot/configs/wolfe_wave_shared_v1_configs.json","w"),indent=2,sort_keys=True)
print(f"\nwrote bot/configs/wolfe_wave_shared_v1_configs.json ({len(symbols)} symbols)")
