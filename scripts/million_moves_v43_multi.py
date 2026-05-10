"""
Million Moves V4.3 - Multi-Coin Leaderboard
============================================
For each symbol in the top-50 list:
  1. Sweep trail exit params (80 combos x 6 walk-forward folds)
  2. Pick best trail config by OOS Sharpe
  3. Train per-fold DecisionTree filter (depth=2, min_leaf=15) with
     IS Sharpe gate — skip filter if tree found no signal
  4. Report Trail + Trail+DT Sharpe across all 6 OOS folds

Output: sorted leaderboard printed to console + saved to CSV.

Usage:
  python scripts/million_moves_v43_multi.py
  python scripts/million_moves_v43_multi.py --since 2024-01-01
  python scripts/million_moves_v43_multi.py --symbols BTC/USDT,SOL/USDT,XRP/USDT
"""
from __future__ import annotations

import argparse, json, math, os, time, warnings
import numpy as np
import pandas as pd
import ccxt

warnings.filterwarnings("ignore")

try:
    from sklearn.tree import DecisionTreeClassifier
    ML_OK = True
except ImportError:
    ML_OK = False

# ---------------------------------------------------------------------------
# Symbols — top 50 by market cap (ex stablecoins/wrapped)
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "TON/USDT", "SHIB/USDT", "AVAX/USDT",
    "TRX/USDT", "LINK/USDT", "SUI/USDT", "DOT/USDT", "BCH/USDT",
    "HBAR/USDT", "LTC/USDT", "UNI/USDT", "NEAR/USDT", "APT/USDT",
    "ETC/USDT", "PEPE/USDT", "ICP/USDT", "OP/USDT", "RENDER/USDT",
    "STX/USDT", "WIF/USDT", "ATOM/USDT", "FIL/USDT", "ARB/USDT",
    "INJ/USDT", "GRT/USDT", "MKR/USDT", "AAVE/USDT", "SEI/USDT",
    "JUP/USDT", "VET/USDT", "FLOKI/USDT", "POL/USDT", "XLM/USDT",
    "WLD/USDT", "BONK/USDT", "TIA/USDT", "HYPE/USDT", "TAO/USDT",
    "S/USDT", "CAKE/USDT", "PYTH/USDT", "ALGO/USDT", "FARTCOIN/USDT",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SINCE_DATE = "2024-01-01"
TIMEFRAME  = "15m"
MIN_BARS   = 5000     # skip coins without enough data

# Supertrend
ST_MULT    = 3.5;  ST_ATR_LEN = 11
EMA_LEN    = 200;  SMA_LEN = 13;  ATR_LEN = 14
VOL_WIN    = 20;   ATR_PCTILE_WIN = 100
ATR_LO = 10;  ATR_HI = 90;  VOL_THR = 1.05

# Walk-forward
TRAIN_M = 12;  OOS_M = 3;  STEP_M = 3

# Trail sweep grid — 5 x 4 x 4 = 80 combos
SL_MULTS    = [1.5, 2.0, 2.5, 3.0, 3.5]
TP1_RS      = [0.5, 0.75, 1.0, 1.5]
TRAIL_MULTS = [0.5, 1.0, 1.5, 2.0]

# DT config (conservative for cross-coin generalization)
DT_DEPTH     = 2
DT_MIN_LEAF  = 15
DT_MIN_IS_SH = 0.5
DT_MIN_SIGS  = 20

FEATURE_NAMES = [
    "atr_pctile", "vol_ratio", "ema200_dist", "ema200_slope",
    "body_ratio", "sma13_dist", "hour_utc", "direction",
    "mom5", "atr_norm", "day_of_week", "rsi14", "rsi4h",
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------
def to_bybit_symbol(symbol: str) -> str:
    """Convert 'BTC/USDT' -> 'BTC/USDT:USDT' for Bybit linear perps."""
    if ":" not in symbol:
        quote = symbol.split("/")[1]
        return f"{symbol}:{quote}"
    return symbol

def fetch_ohlcv(symbol, timeframe, since_date, exchange):
    since_ms = exchange.parse8601(f"{since_date}T00:00:00Z")
    bars = []
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not chunk:
            break
        bars.extend(chunk)
        if len(chunk) < 1000:
            break
        since_ms = chunk[-1][0] + 1
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def _rma(vals, length):
    alpha = 1.0 / length
    out = np.full(len(vals), np.nan)
    s = 0
    while s < len(vals) and np.isnan(vals[s]):
        s += 1
    se = s + length
    if se > len(vals):
        return out
    out[se - 1] = float(np.nanmean(vals[s:se]))
    for i in range(se, len(vals)):
        v = vals[i]
        out[i] = alpha * v + (1.0 - alpha) * out[i - 1] if not np.isnan(v) else out[i - 1]
    return out

def compute_atr(high, low, close, length):
    pc = np.empty_like(close); pc[0] = np.nan; pc[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return _rma(tr, length)

def compute_ema(close, length):
    alpha = 2.0 / (length + 1)
    out = np.full(len(close), np.nan)
    for i, v in enumerate(close):
        if not np.isnan(v):
            out[i] = v
            for j in range(i + 1, len(close)):
                out[j] = alpha * close[j] + (1.0 - alpha) * out[j - 1]
            break
    return out

def compute_sma(arr, length):
    return pd.Series(arr).rolling(length).mean().values

def compute_supertrend(open_, close, atr_st, mult):
    n = len(open_)
    ur = open_ + mult * atr_st
    lr = open_ - mult * atr_st
    upper, lower = ur.copy(), lr.copy()
    direction = np.full(n, np.nan)
    st = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(atr_st[i - 1]):
            direction[i] = 2.0; upper[i] = ur[i]; lower[i] = lr[i]
        else:
            lower[i] = lr[i] if (lr[i] > lower[i-1] or close[i-1] < lower[i-1]) else lower[i-1]
            upper[i] = ur[i] if (ur[i] < upper[i-1] or close[i-1] > upper[i-1]) else upper[i-1]
            ps = st[i-1]
            if np.isnan(ps):
                ps = upper[i-1] if not np.isnan(upper[i-1]) else lower[i-1]
            if ps == upper[i-1]:
                direction[i] = -1.0 if close[i] > upper[i] else 1.0
            else:
                direction[i] =  1.0 if close[i] < lower[i] else -1.0
        st[i] = lower[i] if direction[i] == -1.0 else upper[i]
    return st

def build_raw_signals(close, open_, sma13, ema200, atr_st):
    st = compute_supertrend(open_, close, atr_st, ST_MULT)
    n = len(close)
    pc = np.empty(n); pc[0] = np.nan; pc[1:] = close[:-1]
    ps = np.empty(n); ps[0] = np.nan; ps[1:] = st[:-1]
    pe = np.empty(n); pe[0] = np.nan; pe[1:] = ema200[:-1]
    co = (~np.isnan(pc)) & (~np.isnan(ps)) & (~np.isnan(st)) & (pc < ps) & (close > st)
    cu = (~np.isnan(pc)) & (~np.isnan(ps)) & (~np.isnan(st)) & (pc > ps) & (close < st)
    above = (~np.isnan(pe)) & (~np.isnan(ema200)) & (pc > pe) & (close > ema200)
    sbull = co & (~np.isnan(sma13)) & (close >= sma13) &  above
    sbear = cu & (~np.isnan(sma13)) & (close <= sma13) & (~above)
    return sbull.astype(bool), sbear.astype(bool)

def compute_rolling_atr_pctile(atr14, window=ATR_PCTILE_WIN):
    n = len(atr14)
    out = np.full(n, 50.0)
    for i in range(window, n):
        w = atr14[i - window:i]
        if not np.isnan(atr14[i]) and not np.all(np.isnan(w)):
            valid = w[~np.isnan(w)]
            out[i] = float(np.sum(valid < atr14[i])) / len(valid) * 100.0
    return out

def compute_vol_ratio(volume, win=VOL_WIN):
    vs = compute_sma(volume, win)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = volume / vs
    return np.where(np.isnan(ratio) | np.isinf(ratio), 1.0, ratio)

def compute_rsi(close, period=14):
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    d = np.diff(close)
    gain = np.maximum(d, 0.0)
    loss = np.maximum(-d, 0.0)
    avg_g = gain[:period].mean()
    avg_l = loss[:period].mean()
    rs = avg_g / avg_l if avg_l > 0 else np.inf
    out[period] = 100.0 - 100.0 / (1.0 + rs)
    for j in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gain[j]) / period
        avg_l = (avg_l * (period - 1) + loss[j]) / period
        rs = avg_g / avg_l if avg_l > 0 else np.inf
        out[j + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out

def compute_rsi_htf(ts_index, close_15m, htf="4h", period=14):
    s = pd.Series(close_15m, index=ts_index)
    s_htf = s.resample(htf, closed="right", label="right").last().dropna()
    rsi_vals = compute_rsi(s_htf.values, period)
    s_htf_rsi = pd.Series(rsi_vals, index=s_htf.index)
    return s_htf_rsi.reindex(ts_index, method="ffill").fillna(50.0).values

def generate_wf_folds(index):
    folds = []; fold_id = 1; fold_start = index[0]; data_end = index[-1]
    while True:
        train_end = fold_start + pd.DateOffset(months=TRAIN_M)
        oos_start = train_end
        oos_end   = min(oos_start + pd.DateOffset(months=OOS_M),
                        data_end + pd.Timedelta(seconds=1))
        if oos_start > data_end:
            break
        tr_il  = np.where((index >= fold_start) & (index < train_end))[0]
        oos_il = np.where((index >= oos_start)  & (index < oos_end))[0]
        if len(tr_il) > 50 and len(oos_il) > 0:
            folds.append(dict(
                fold_id=fold_id,
                i0t=int(tr_il[0]),  i1t=int(tr_il[-1]) + 1,
                i0o=int(oos_il[0]), i1o=int(oos_il[-1]) + 1,
            ))
        fold_start = fold_start + pd.DateOffset(months=STEP_M)
        fold_id += 1
    return folds

# ---------------------------------------------------------------------------
# Trail simulation
# ---------------------------------------------------------------------------
def _sim_trail(close, high, low, sbull_raw, sbear_raw,
               atr14, atr_pct, vol_rat,
               sl_mult, tp1_r, trail_mult, signal_mask=None):
    n = len(close)
    r_list = []; trades = []
    active = False; is_long = False; entry = 0.0
    sl_ = tp1 = 0.0; risk = 1.0
    tp1_hit = False; trail_sl = 0.0; acc_r = 0.0; entry_i = 0

    for i in range(1, n):
        h = high[i]; l = low[i]; c = close[i]; atr = atr14[i]

        if active:
            if is_long:
                if not tp1_hit and h >= tp1:
                    acc_r += 0.5 * tp1_r
                    trail_sl = entry; tp1_hit = True
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = h - trail_mult * atr
                        if cand > trail_sl:
                            trail_sl = cand
                    if l <= trail_sl:
                        total_r = acc_r + 0.5 * max(0.0, (trail_sl - entry) / risk)
                        r_list.append(total_r)
                        trades.append({
                            "entry_i": entry_i, "exit_i": i, "r": total_r, "reason": "Trail",
                            "direction": "long" if is_long else "short", "entry_price": entry,
                            "exit_price": trail_sl, "stop_price": sl_, "tp1_price": tp1,
                        })
                        active = False; continue
                else:
                    if l <= sl_:
                        r_list.append(-1.0)
                        trades.append({
                            "entry_i": entry_i, "exit_i": i, "r": -1.0, "reason": "SL",
                            "direction": "long", "entry_price": entry,
                            "exit_price": sl_, "stop_price": sl_, "tp1_price": tp1,
                        })
                        active = False
            else:
                if not tp1_hit and l <= tp1:
                    acc_r += 0.5 * tp1_r
                    trail_sl = entry; tp1_hit = True
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = l + trail_mult * atr
                        if cand < trail_sl:
                            trail_sl = cand
                    if h >= trail_sl:
                        total_r = acc_r + 0.5 * max(0.0, (entry - trail_sl) / risk)
                        r_list.append(total_r)
                        trades.append({
                            "entry_i": entry_i, "exit_i": i, "r": total_r, "reason": "Trail",
                            "direction": "long" if is_long else "short", "entry_price": entry,
                            "exit_price": trail_sl, "stop_price": sl_, "tp1_price": tp1,
                        })
                        active = False; continue
                else:
                    if h >= sl_:
                        r_list.append(-1.0)
                        trades.append({
                            "entry_i": entry_i, "exit_i": i, "r": -1.0, "reason": "SL",
                            "direction": "short", "entry_price": entry,
                            "exit_price": sl_, "stop_price": sl_, "tp1_price": tp1,
                        })
                        active = False

        if active and is_long and sbear_raw[i]:
            rem = 0.5 if tp1_hit else 1.0
            total_r = acc_r + rem * (c - entry) / risk
            r_list.append(total_r)
            trades.append({
                "entry_i": entry_i, "exit_i": i, "r": total_r, "reason": "Rev",
                "direction": "long", "entry_price": entry,
                "exit_price": c, "stop_price": sl_, "tp1_price": tp1,
            })
            active = False
        if active and not is_long and sbull_raw[i]:
            rem = 0.5 if tp1_hit else 1.0
            total_r = acc_r + rem * (entry - c) / risk
            r_list.append(total_r)
            trades.append({
                "entry_i": entry_i, "exit_i": i, "r": total_r, "reason": "Rev",
                "direction": "short", "entry_price": entry,
                "exit_price": c, "stop_price": sl_, "tp1_price": tp1,
            })
            active = False

        if not active and not math.isnan(atr):
            ap = atr_pct[i]
            if not (ATR_LO < ap < ATR_HI):
                continue
            if vol_rat[i] < VOL_THR:
                continue
            if sbull_raw[i]:
                if signal_mask is not None and not signal_mask[i]:
                    continue
                sl_ = l - atr * sl_mult; risk = max(c - sl_, 1e-10)
                entry = c; is_long = True; tp1 = c + tp1_r * risk
                tp1_hit = False; trail_sl = sl_; acc_r = 0.0; active = True; entry_i = i
            elif sbear_raw[i]:
                if signal_mask is not None and not signal_mask[i]:
                    continue
                sl_ = h + atr * sl_mult; risk = max(sl_ - c, 1e-10)
                entry = c; is_long = False; tp1 = c - tp1_r * risk
                tp1_hit = False; trail_sl = sl_; acc_r = 0.0; active = True; entry_i = i

    if active:
        cl = close[-1]; rem = 0.5 if tp1_hit else 1.0
        total_r = acc_r + rem * ((cl - entry) if is_long else (entry - cl)) / risk
        r_list.append(total_r)
        trades.append({
            "entry_i": entry_i, "exit_i": n - 1, "r": total_r, "reason": "Open",
            "direction": "long" if is_long else "short", "entry_price": entry,
            "exit_price": cl, "stop_price": sl_, "tp1_price": tp1,
        })

    return np.array(r_list, dtype=np.float64), trades

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def extract_features(signal_indices, close, high, low, open_, volume,
                     atr14, atr_pct, vol_rat, ema200, sma13,
                     sbull, sbear, timestamps, rsi14, rsi4h):
    rows = []
    for i in signal_indices:
        if i < 20:
            rows.append(None); continue
        f1 = atr_pct[i] / 100.0
        f2 = float(vol_rat[i])
        f3 = (close[i] - ema200[i]) / close[i] if not np.isnan(ema200[i]) else 0.0
        e10 = ema200[max(0, i - 10)]
        f4 = (ema200[i] - e10) / e10 if not np.isnan(e10) and e10 > 0 else 0.0
        rng = high[i] - low[i]
        f5 = abs(close[i] - open_[i]) / rng if rng > 0 else 0.0
        f6 = abs(close[i] - sma13[i]) / close[i] if not np.isnan(sma13[i]) else 0.0
        f7 = timestamps[i].hour / 23.0 if hasattr(timestamps[i], "hour") else 0.5
        f8 = 1.0 if sbull[i] else 0.0
        f9 = (close[i] - close[max(0, i - 5)]) / close[max(0, i - 5)]
        f10 = atr14[i] / close[i] if not np.isnan(atr14[i]) else 0.0
        f11 = timestamps[i].weekday() / 6.0 if hasattr(timestamps[i], "weekday") else 0.5
        f12 = float(rsi14[i]) / 100.0 if not np.isnan(rsi14[i]) else 0.5
        f13 = float(rsi4h[i]) / 100.0 if not np.isnan(rsi4h[i]) else 0.5
        rows.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13])
    return rows

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(r_arr, min_n=1):
    n = len(r_arr)
    if n < min_n:
        return {"n": n, "sharpe": -99.0, "total_r": 0.0, "win_rate": 0.0, "pf": 0.0}
    std = float(np.std(r_arr, ddof=1))
    mean = float(np.mean(r_arr))
    wins = r_arr[r_arr > 0]; losses = r_arr[r_arr < 0]
    gw = float(wins.sum()) if len(wins) > 0 else 0.0
    gl = float(-losses.sum()) if len(losses) > 0 else 0.0
    return {
        "n":        n,
        "sharpe":   round(mean / std if std > 1e-12 else 0.0, 4),
        "total_r":  round(float(r_arr.sum()), 2),
        "win_rate": round(len(wins) / n, 3),
        "pf":       round(gw / gl if gl > 0 else 0.0, 3),
    }

# ---------------------------------------------------------------------------
# Per-coin analysis
# ---------------------------------------------------------------------------
def _trade_rows(symbol, variant, fold_id, fold_start_i, ts, trades, sl, tp1, tr):
    rows = []
    for trade in trades:
        entry_i = fold_start_i + int(trade["entry_i"])
        exit_i = fold_start_i + int(trade["exit_i"])
        rows.append({
            "symbol": symbol,
            "variant": variant,
            "fold_id": fold_id,
            "entry_i": entry_i,
            "exit_i": exit_i,
            "entry_time": ts[entry_i],
            "exit_time": ts[exit_i],
            "direction": trade.get("direction"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "stop_price": trade.get("stop_price"),
            "tp1_price": trade.get("tp1_price"),
            "exit_reason": trade.get("reason"),
            "r": float(trade.get("r", np.nan)),
            "sl_mult": sl,
            "tp1_r": tp1,
            "trail_mult": tr,
        })
    return rows


def analyze_coin(symbol, df, verbose=False, collect_trades=False):
    """Sweep + DT for one coin. Returns result dict."""
    if len(df) < MIN_BARS:
        return {"symbol": symbol, "error": f"only {len(df)} bars"}

    close  = df["close"].values.astype(np.float64)
    open_  = df["open"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    ts     = df.index

    # Indicators
    atr_st  = compute_atr(high, low, close, ST_ATR_LEN)
    atr14   = compute_atr(high, low, close, ATR_LEN)
    ema200  = compute_ema(close, EMA_LEN)
    sma13   = compute_sma(close, SMA_LEN)
    atr_pct = compute_rolling_atr_pctile(atr14, ATR_PCTILE_WIN)
    vol_rat = compute_vol_ratio(volume, VOL_WIN)
    rsi14   = compute_rsi(close, 14)
    rsi4h   = compute_rsi_htf(ts, close, "4h", 14)

    sbull, sbear = build_raw_signals(close, open_, sma13, ema200, atr_st)
    n_bull = int(sbull.sum()); n_bear = int(sbear.sum())

    if n_bull + n_bear < 20:
        return {"symbol": symbol, "error": "too few signals"}

    folds = generate_wf_folds(ts)
    if len(folds) < 2:
        return {"symbol": symbol, "error": "not enough folds"}

    # ------------------------------------------------------------------
    # STEP 1 — Trail parameter sweep
    # ------------------------------------------------------------------
    combos = [(sl, tp1, tr)
              for sl in SL_MULTS
              for tp1 in TP1_RS
              for tr in TRAIL_MULTS]

    combo_results = {}
    for (sl, tp1, tr) in combos:
        oos_r_all = []
        for fold in folds:
            i0o, i1o = fold["i0o"], fold["i1o"]
            r_oos, _ = _sim_trail(
                close[i0o:i1o], high[i0o:i1o], low[i0o:i1o],
                sbull[i0o:i1o], sbear[i0o:i1o],
                atr14[i0o:i1o], atr_pct[i0o:i1o], vol_rat[i0o:i1o],
                sl_mult=sl, tp1_r=tp1, trail_mult=tr,
            )
            oos_r_all.extend(r_oos.tolist())
        r_arr = np.array(oos_r_all)
        m = metrics(r_arr, min_n=5)
        combo_results[(sl, tp1, tr)] = m

    # Pick winner by OOS Sharpe
    best_combo = max(combo_results, key=lambda k: combo_results[k]["sharpe"])
    best_sl, best_tp1, best_tr = best_combo
    best_trail_m = combo_results[best_combo]

    # Count positive folds for best trail config
    trail_pos_folds = 0
    trail_oos_r = []
    trade_rows = []
    for fold in folds:
        i0o, i1o = fold["i0o"], fold["i1o"]
        r_oos, td_oos = _sim_trail(
            close[i0o:i1o], high[i0o:i1o], low[i0o:i1o],
            sbull[i0o:i1o], sbear[i0o:i1o],
            atr14[i0o:i1o], atr_pct[i0o:i1o], vol_rat[i0o:i1o],
            sl_mult=best_sl, tp1_r=best_tp1, trail_mult=best_tr,
        )
        trail_oos_r.extend(r_oos.tolist())
        if collect_trades:
            trade_rows.extend(_trade_rows(symbol, "trail", fold["fold_id"], i0o, ts, td_oos, best_sl, best_tp1, best_tr))
        if len(r_oos) > 0 and metrics(r_oos)["sharpe"] > 0:
            trail_pos_folds += 1

    trail_total_m = metrics(np.array(trail_oos_r), min_n=5)

    if not ML_OK:
        return {
            "symbol": symbol,
            "n_bull": n_bull, "n_bear": n_bear,
            "best_sl": best_sl, "best_tp1": best_tp1, "best_tr": best_tr,
            "trail_sh": trail_total_m["sharpe"],
            "trail_r":  trail_total_m["total_r"],
            "trail_win": trail_total_m["win_rate"],
            "trail_pf":  trail_total_m["pf"],
            "trail_n":   trail_total_m["n"],
            "trail_pos": f"{trail_pos_folds}/{len(folds)}",
            "dt_sh": None, "dt_r": None, "dt_win": None, "dt_pf": None,
            "dt_n": None, "dt_pos": None,
        }

    # ------------------------------------------------------------------
    # STEP 2 — Per-fold DT filter with best trail params
    # ------------------------------------------------------------------
    dt_oos_r = []; dt_pos_folds = 0

    for fold in folds:
        i0t, i1t = fold["i0t"], fold["i1t"]
        i0o, i1o = fold["i0o"], fold["i1o"]

        # IS trail trades (for labels)
        r_is, td_is = _sim_trail(
            close[i0t:i1t], high[i0t:i1t], low[i0t:i1t],
            sbull[i0t:i1t], sbear[i0t:i1t],
            atr14[i0t:i1t], atr_pct[i0t:i1t], vol_rat[i0t:i1t],
            sl_mult=best_sl, tp1_r=best_tp1, trail_mult=best_tr,
        )

        sig_idx_is = [i for i in range(i0t, i1t)
                      if (sbull[i] or sbear[i])
                      and ATR_LO < atr_pct[i] < ATR_HI
                      and vol_rat[i] >= VOL_THR]

        if len(sig_idx_is) < DT_MIN_SIGS or len(td_is) < 5:
            # Not enough IS data — pass all OOS signals through
            r_oos, td_oos = _sim_trail(
                close[i0o:i1o], high[i0o:i1o], low[i0o:i1o],
                sbull[i0o:i1o], sbear[i0o:i1o],
                atr14[i0o:i1o], atr_pct[i0o:i1o], vol_rat[i0o:i1o],
                sl_mult=best_sl, tp1_r=best_tp1, trail_mult=best_tr,
            )
            dt_oos_r.extend(r_oos.tolist())
            if collect_trades:
                trade_rows.extend(_trade_rows(symbol, "dt", fold["fold_id"], i0o, ts, td_oos, best_sl, best_tp1, best_tr))
            if len(r_oos) > 0 and metrics(r_oos)["sharpe"] > 0:
                dt_pos_folds += 1
            continue

        # Build IS feature matrix and labels
        is_label_map = {(i0t + t["entry_i"]): (1 if t["r"] > 0 else 0) for t in td_is}
        feats_is, labels_is, valid_is = [], [], []
        for gi in sig_idx_is:
            row = extract_features(
                [gi], close, high, low, open_, volume,
                atr14, atr_pct, vol_rat, ema200, sma13,
                sbull, sbear, ts, rsi14, rsi4h,
            )[0]
            if row is None:
                continue
            lbl = is_label_map.get(gi)
            if lbl is None:
                continue
            feats_is.append(row); labels_is.append(lbl); valid_is.append(gi)

        if len(feats_is) < DT_MIN_SIGS or len(set(labels_is)) < 2:
            r_oos, td_oos = _sim_trail(
                close[i0o:i1o], high[i0o:i1o], low[i0o:i1o],
                sbull[i0o:i1o], sbear[i0o:i1o],
                atr14[i0o:i1o], atr_pct[i0o:i1o], vol_rat[i0o:i1o],
                sl_mult=best_sl, tp1_r=best_tp1, trail_mult=best_tr,
            )
            dt_oos_r.extend(r_oos.tolist())
            if collect_trades:
                trade_rows.extend(_trade_rows(symbol, "dt", fold["fold_id"], i0o, ts, td_oos, best_sl, best_tp1, best_tr))
            if len(r_oos) > 0 and metrics(r_oos)["sharpe"] > 0:
                dt_pos_folds += 1
            continue

        # Train DT
        X_is = np.array(feats_is, dtype=np.float64)
        y_is = np.array(labels_is, dtype=int)
        clf = DecisionTreeClassifier(
            max_depth=DT_DEPTH, min_samples_leaf=DT_MIN_LEAF,
            class_weight="balanced", random_state=42,
        )
        clf.fit(X_is, y_is)

        # Tune threshold on IS
        is_probas = clf.predict_proba(X_is)[:, 1]
        is_r_map = {(i0t + t["entry_i"]): t["r"] for t in td_is}
        is_gp = [(gi, float(is_probas[k])) for k, gi in enumerate(valid_is)]
        best_thr = 0.55; best_is_sh = -99.0
        for thr in np.arange(0.40, 0.90, 0.05):
            sel_r = [is_r_map[gi] for gi, p in is_gp if p >= thr and gi in is_r_map]
            if len(sel_r) < 8:
                continue
            s = metrics(np.array(sel_r), min_n=5)["sharpe"]
            if s > best_is_sh:
                best_is_sh = s; best_thr = thr

        # Build OOS mask
        use_dt = best_is_sh >= DT_MIN_IS_SH
        sig_idx_oos = [i for i in range(i0o, i1o)
                       if (sbull[i] or sbear[i])
                       and ATR_LO < atr_pct[i] < ATR_HI
                       and vol_rat[i] >= VOL_THR]

        oos_mask = np.zeros(i1o - i0o, dtype=bool)
        if sig_idx_oos and use_dt:
            feats_oos, valid_oos = [], []
            for gi in sig_idx_oos:
                row = extract_features(
                    [gi], close, high, low, open_, volume,
                    atr14, atr_pct, vol_rat, ema200, sma13,
                    sbull, sbear, ts, rsi14, rsi4h,
                )[0]
                if row is not None:
                    feats_oos.append(row); valid_oos.append(gi)
            if feats_oos:
                probas = clf.predict_proba(np.array(feats_oos, dtype=np.float64))[:, 1]
                for k, gi in enumerate(valid_oos):
                    if probas[k] >= best_thr:
                        oos_mask[gi - i0o] = True
        elif sig_idx_oos:
            for gi in sig_idx_oos:
                oos_mask[gi - i0o] = True

        r_oos, td_oos = _sim_trail(
            close[i0o:i1o], high[i0o:i1o], low[i0o:i1o],
            sbull[i0o:i1o], sbear[i0o:i1o],
            atr14[i0o:i1o], atr_pct[i0o:i1o], vol_rat[i0o:i1o],
            sl_mult=best_sl, tp1_r=best_tp1, trail_mult=best_tr,
            signal_mask=oos_mask,
        )
        dt_oos_r.extend(r_oos.tolist())
        if collect_trades:
            trade_rows.extend(_trade_rows(symbol, "dt", fold["fold_id"], i0o, ts, td_oos, best_sl, best_tp1, best_tr))
        if len(r_oos) > 0 and metrics(r_oos)["sharpe"] > 0:
            dt_pos_folds += 1

    dt_total_m = metrics(np.array(dt_oos_r), min_n=3)

    result = {
        "symbol":   symbol,
        "n_bull":   n_bull,
        "n_bear":   n_bear,
        "best_sl":  best_sl,
        "best_tp1": best_tp1,
        "best_tr":  best_tr,
        "trail_sh": trail_total_m["sharpe"],
        "trail_r":  trail_total_m["total_r"],
        "trail_win": trail_total_m["win_rate"],
        "trail_pf":  trail_total_m["pf"],
        "trail_n":   trail_total_m["n"],
        "trail_pos": f"{trail_pos_folds}/{len(folds)}",
        "dt_sh":    dt_total_m["sharpe"],
        "dt_r":     dt_total_m["total_r"],
        "dt_win":   dt_total_m["win_rate"],
        "dt_pf":    dt_total_m["pf"],
        "dt_n":     dt_total_m["n"],
        "dt_pos":   f"{dt_pos_folds}/{len(folds)}",
    }
    if collect_trades:
        result["_trade_rows"] = trade_rows
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MM V4.3 Multi-Coin Leaderboard")
    parser.add_argument("--since",   default=SINCE_DATE)
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated list of symbols (default: top-50 list)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--write-trades", action="store_true",
                        help="Write per-trade OOS rows for best trail and DT variants.")
    parser.add_argument("--output-prefix", default=os.path.join(OUT_DIR, "million_moves_v43_multi"),
                        help="Output prefix for result CSVs.")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_SYMBOLS

    print(f"Multi-Coin Leaderboard  |  {len(symbols)} symbols  |  since {args.since}")
    print(f"Sweep: {len(SL_MULTS)}x{len(TP1_RS)}x{len(TRAIL_MULTS)}={len(SL_MULTS)*len(TP1_RS)*len(TRAIL_MULTS)} combos  |  DT depth={DT_DEPTH} min_leaf={DT_MIN_LEAF}\n")

    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "linear"},  # USDT-margined perpetual futures
    })

    results = []
    all_trade_rows = []
    t0_total = time.time()

    for sym in symbols:
        t0 = time.time()
        print(f"  {sym:20s} ...", end="", flush=True)
        try:
            df = fetch_ohlcv(to_bybit_symbol(sym), TIMEFRAME, args.since, exchange)
            res = analyze_coin(sym, df, verbose=args.verbose, collect_trades=args.write_trades)
        except Exception as e:
            res = {"symbol": sym, "error": str(e)[:60]}

        elapsed = time.time() - t0
        if "error" in res:
            print(f" SKIP ({res['error']})  [{elapsed:.0f}s]")
        else:
            print(f" Trail {res['trail_sh']:+.3f} ({res['trail_pos']})  "
                  f"DT {res['dt_sh']:+.3f} ({res['dt_pos']})  "
                  f"params sl={res['best_sl']} tp1={res['best_tp1']} tr={res['best_tr']}  "
                  f"[{elapsed:.0f}s]")
        if args.write_trades and "_trade_rows" in res:
            all_trade_rows.extend(res.pop("_trade_rows"))
        results.append(res)

    # Filter out errors
    ok = [r for r in results if "error" not in r]

    # Sort by DT Sharpe descending
    ok.sort(key=lambda r: r["dt_sh"], reverse=True)

    total_elapsed = time.time() - t0_total
    print(f"\n{'='*100}")
    print(f"  LEADERBOARD  |  sorted by Trail+DT Sharpe  |  {len(ok)}/{len(results)} coins ok  |  {total_elapsed/60:.1f}min total")
    print(f"{'='*100}")
    hdr = f"  {'Symbol':<16}  {'Trail_Sh':>8}  {'TPos':>5}  {'Trail_R':>8}  {'TR_Win':>6}  {'TR_PF':>5}  ||  {'DT_Sh':>8}  {'DPos':>5}  {'DT_R':>8}  {'DT_Win':>6}  {'DT_PF':>5}  {'DT_n':>5}  {'Params'}"
    print(hdr)
    print(f"  {'-'*len(hdr.strip())}")
    for r in ok:
        params = f"sl={r['best_sl']} tp1={r['best_tp1']} tr={r['best_tr']}"
        print(f"  {r['symbol']:<16}  {r['trail_sh']:>+8.3f}  {r['trail_pos']:>5}  "
              f"{r['trail_r']:>+8.1f}  {r['trail_win']:>6.1%}  {r['trail_pf']:>5.2f}  ||  "
              f"{r['dt_sh']:>+8.3f}  {r['dt_pos']:>5}  "
              f"{r['dt_r']:>+8.1f}  {r['dt_win']:>6.1%}  {r['dt_pf']:>5.2f}  "
              f"{r['dt_n']:>5}  {params}")
    print(f"{'='*100}")

    # Save CSV
    out_path = f"{args.output_prefix}_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved -> {out_path}")
    if args.write_trades:
        trades_path = f"{args.output_prefix}_trades.csv"
        pd.DataFrame(all_trade_rows).to_csv(trades_path, index=False)
        print(f"Trades saved -> {trades_path}")

    # Save best configs per coin as JSON
    configs = {}
    for r in ok:
        configs[r["symbol"]] = {
            "sl":        r["best_sl"],
            "tp1":       r["best_tp1"],
            "trail":     r["best_tr"],
            "trail_sh":  r["trail_sh"],
            "trail_pos": r["trail_pos"],
            "dt_sh":     r["dt_sh"],
            "dt_pos":    r["dt_pos"],
        }
    cfg_path = f"{args.output_prefix}_best_configs.json"
    with open(cfg_path, "w") as fh:
        json.dump(configs, fh, indent=2)
    print(f"Best configs saved -> {cfg_path}")
    print(f"Total elapsed: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
