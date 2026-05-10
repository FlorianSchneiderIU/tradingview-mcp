import sys, math, importlib.util
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')

# Import without triggering __main__ block
spec = importlib.util.spec_from_file_location('al', 'scripts/million_moves_v43_adaptive_lag.py')
al = importlib.util.module_from_spec(spec)
spec.loader.exec_module(al)

from sklearn.tree import DecisionTreeClassifier

print("Fetching XRP data...")
df = al.fetch_ohlcv('XRP/USDT:USDT', '2023-01-01')
print(f"Got {len(df)} bars: {df.index[0]} -> {df.index[-1]}")

close  = df['close'].values.astype(float)
high   = df['high'].values.astype(float)
low    = df['low'].values.astype(float)
open_  = df['open'].values.astype(float)
volume = df['volume'].values.astype(float)

ema200      = al.compute_ema(close, al.EMA_LEN)
atr14       = al.compute_atr(high, low, close, 14)
atr_pctile  = al.compute_atr_pctile(atr14, al.ATR_WIN)
vol_ratio   = al.compute_vol_ratio(volume, al.VOL_WIN)
rsi14       = al.compute_rsi(close, 14)
adx14       = al.compute_adx(high, low, close, 14)
combo_sigs  = al.precompute_combo_signals(close, open_, high, low, ema200)

# XRP fold 7: IS 2024-07-01→2025-07-01, OOS 2025-07-01→2025-10-01
i0 = int(np.searchsorted(df.index, pd.Timestamp('2025-07-01', tz='UTC')))
i1 = int(np.searchsorted(df.index, pd.Timestamp('2025-10-01', tz='UTC')))
t0 = int(np.searchsorted(df.index, pd.Timestamp('2024-07-01', tz='UTC')))
t1 = i0
print(f"\nIS: bars {t0}-{t1}  ({df.index[t0].date()} -> {df.index[t1].date()})")
print(f"OOS: bars {i0}-{i1}  ({df.index[i0].date()} -> {df.index[i1].date()})")

# Build training data and train DTs (mirroring run_coin)
feat_rows, labels, cidxs = al.build_is_training_data(
    t0, t1, close, high, low, open_, ema200,
    atr14, atr_pctile, vol_ratio, rsi14, adx14, combo_sigs
)
dts = []
for c_idx in range(al.N_COMBOS):
    mask = [k for k, ci in enumerate(cidxs) if ci == c_idx]
    if len(mask) < al.DT_MIN_SIGS:
        dts.append(None); continue
    X = np.array([feat_rows[k] for k in mask])
    y = np.array([labels[k]    for k in mask])
    if len(np.unique(y)) < 2:
        dts.append(None); continue
    dt = DecisionTreeClassifier(max_depth=al.DT_MAX_DEPTH, min_samples_leaf=al.DT_MIN_LEAF,
                                 class_weight='balanced', random_state=42)
    dt.fit(X, y)
    dts.append(dt)
print(f"DTs trained: {sum(1 for d in dts if d is not None)}/{al.N_COMBOS}")

# Run OOS simulation
rs, trades, usage = al.sim_adaptive_oos(
    i0, i1, close, high, low, open_, ema200,
    atr14, atr_pctile, vol_ratio, rsi14, adx14,
    combo_sigs, dts
)

total_r = float(np.sum(rs))
print(f"\nTrades: {len(trades)}  TotalR: {total_r:.2f}  WinRate: {np.mean(rs>0):.0%}")
print(f"Combo usage: {usage}")
print()

for t in trades:
    r = t['r']
    ei = t['entry_i']
    xi = t['exit_i']
    is_long = t['direction'] == 'long'
    entry_c  = close[ei]
    sl_val   = (low[ei]  - al.SL_MULT * atr14[ei]) if is_long else (high[ei] + al.SL_MULT * atr14[ei])
    risk_v   = abs(entry_c - sl_val)
    exit_c   = close[xi]

    # Check for SL breaches between entry and exit that should have fired
    sl_breached_at = None
    for k in range(ei + 1, xi + 1):
        if is_long  and low[k]  <= sl_val: sl_breached_at = k; break
        if not is_long and high[k] >= sl_val: sl_breached_at = k; break

    flags = []
    if r < -1.05:
        flags.append(f'LARGE_LOSS(R={r:.2f})')
    if sl_breached_at is not None and t['reason'] != 'SL':
        dt_sl = df.index[sl_breached_at]
        flags.append(f'SL_BREACH_MISSED@bar{sl_breached_at}({dt_sl.strftime("%m/%d %H:%M")})')

    flag_str = '  *** ' + ' | '.join(flags) + ' ***' if flags else ''
    dt1 = df.index[ei].strftime('%m/%d %H:%M')
    dt2 = df.index[xi].strftime('%m/%d %H:%M')
    d = 'L' if is_long else 'S'
    print(f"  {d} [{dt1}->{dt2}] {t['reason']:<5} R={r:>+7.3f} spd{t['combo']}  "
          f"entry={entry_c:.5f} sl={sl_val:.5f} risk={risk_v:.6f} exit={exit_c:.5f}{flag_str}")
