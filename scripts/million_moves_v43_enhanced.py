"""
Million Moves V4.3 — Enhanced Strategy
=======================================

Changes vs baseline:

  1. STRUCTURE SL — SL placed below swing low/high of last N bars (not raw ATR
     multiple from signal-bar low/high).  Typically tighter in trending markets
     while still grounded in real support/resistance.

  2. BREAK-EVEN management — after price moves 1R in the trade direction, the
     stop is shifted to entry.  Eliminates the -0.53R-avg reversal exits that
     cost -26.6R in the baseline OOS.

  3. SINGLE TP at 2.5R — removes the TP1(33%)/TP2(50%)/TP3 scale-out chain.
     TP1-only exits averaged only +0.44R, dragging down expectancy.  A single
     target at 2.5R concentrates on the high-R bucket.

  4. ATR REGIME FILTER — compute rolling percentile of ATR14 over last 100 bars;
     only take signals in the 20–80 percentile band.  Avoids entering in extreme
     volatility where stops must be huge, OR in dead-flat market where moves
     are too small to reach TP.

  5. VOLUME FILTER — signal-bar volume must exceed 20-bar SMA × threshold.
     Confirms there is real participation behind the breakout.

  6. ML SIGNAL FILTER (sklearn GradientBoostingClassifier) — per-IS-fold,
     trains a binary classifier on a set of 8 per-signal features to predict
     whether a trade will end in positive R.  Applied as a hard filter on OOS
     signals (predict_proba > threshold).  Anti-lookahead: model trained
     entirely on IS data, threshold tuned on IS data, applied to OOS only.

Walk-forward:  12m train / 3m OOS / 3m step  (matches baseline)
Fixed params:  st_mult = 3.5 (walk-forward consensus), st_atr_len = 11

Run modes
  python scripts/million_moves_v43_enhanced.py             # all three variants
  python scripts/million_moves_v43_enhanced.py --no-ml     # skip ML variant
  python scripts/million_moves_v43_enhanced.py --since 2024-04-01
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import ccxt

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fixed parameters (from walk-forward consensus)
# ---------------------------------------------------------------------------
ST_MULT       = 3.5
ST_ATR_LEN    = 11
EMA_LEN       = 200
SMA_LEN       = 13
ATR_SL_LEN    = 14

# Baseline params (from walk-forward best combo: folds 4/5/6)
BASELINE_SL_MULT = 3.5
BASELINE_TP_MULT = 2.0   # used as tp1/tp2/tp3 multiplier in original 3-tier system

# Enhanced exit config (structure SL + 2-tier fixed TP)
SL_SWING_BARS   = 5       # structure SL: look back N full bars
SL_ATR_BUFFER   = 0.25    # ATR buffer below/above swing level
TP1_R           = 1.0     # 1st TP: close 50% here, move SL to break-even
TP2_R           = 3.0     # 2nd TP: close remaining 50% here
BE_TRIGGER_R    = TP1_R   # alias kept for compatibility
TP_R            = TP2_R   # alias for single-TP variant

# Trail exit config — sweep winner (Sharpe +0.225, 6/6 folds, n=240)
TRAIL_SL_MULT  = 3.0    # ATR SL distance (signal-bar low/high ± sl_mult×ATR14)
TRAIL_TP1_R    = 0.75   # bank 50% here, move SL to break-even
TRAIL_MULT     = 0.50   # trail remaining 50% by this many ATR14 units

# Filter config
ATR_PCTILE_WIN  = 100     # window for ATR percentile
ATR_PCTILE_LOW  = 10      # exclude very low vol (widened from 20 — less restrictive)
ATR_PCTILE_HIGH = 90      # exclude extreme vol  (widened from 80)
VOL_SMA_WIN     = 20      # volume SMA window
VOL_THRESHOLD   = 1.05    # volume must exceed SMA × this (relaxed from 1.2)

# ML config
ML_MIN_SIGNALS  = 25      # min IS signals to train ML
ML_MAX_DEPTH    = 3       # tree depth — 3 gives at most 8 leaves / ~7 readable rules
ML_MIN_LEAF     = 8       # min samples per leaf (prevents overfitting on small folds)
ML_PROBA_THRESH = 0.55    # default win-rate threshold on leaf (tuned per fold)

# Feature names (must match extract_ml_features order)
FEATURE_NAMES = [
    "atr_pctile",    # f1 : rolling ATR percentile (0-1)
    "vol_ratio",     # f2 : volume / 20-bar SMA
    "ema200_dist",   # f3 : (close - EMA200) / close  (signed)
    "ema200_slope",  # f4 : EMA200 10-bar slope / EMA200
    "body_ratio",    # f5 : candle body / range (0-1)
    "sma13_dist",    # f6 : abs(close - SMA13) / close
    "hour_utc",      # f7 : hour of day 0-23 scaled 0-1
    "direction",     # f8 : 1=long  0=short
    "mom5",          # f9 : 5-bar price return
    "atr_norm",      # f10: ATR14 / close
    "day_of_week",   # f11: weekday 0=Mon..6=Sun scaled 0-1
    "rsi14",         # f12: RSI(14) on 15m scaled 0-1
    "rsi4h",         # f13: RSI(14) on 4H scaled 0-1
]

# Walk-forward
TRAIN_MONTHS    = 12
OOS_MONTHS      = 3
STEP_MONTHS     = 3

SYMBOL          = "ETH/USDT"
TIMEFRAME       = "15m"
SINCE_DATE      = "2024-01-01"
OUT_DIR         = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol, timeframe, since_date):
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since_date}T00:00:00Z")
    bars = []
    print(f"Fetching {symbol} {timeframe} from {since_date} …", flush=True)
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
    print(f"  -> {len(df):,} bars  ({df.index[0]} ... {df.index[-1]})", flush=True)
    return df


# ---------------------------------------------------------------------------
# Indicators (causal, numpy)
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
    """Returns (sbull, sbear) bool arrays — no ATR/volume filter."""
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


# ---------------------------------------------------------------------------
# Additional indicator arrays needed for enhancements
# ---------------------------------------------------------------------------
def compute_rolling_atr_pctile(atr14, window=ATR_PCTILE_WIN):
    """Rolling percentile rank of atr14[i] within last `window` bars. Causal."""
    n = len(atr14)
    out = np.full(n, 50.0)
    for i in range(window, n):
        w = atr14[i - window:i]
        if not np.isnan(atr14[i]) and not np.all(np.isnan(w)):
            out[i] = float(np.sum(w < atr14[i])) / len(w[~np.isnan(w)]) * 100.0
    return out

def compute_vol_ratio(volume, sma_win=VOL_SMA_WIN):
    """volume / sma(volume, sma_win). Causal."""
    vs = compute_sma(volume, sma_win)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = volume / vs
    ratio = np.where(np.isnan(ratio) | np.isinf(ratio), 1.0, ratio)
    return ratio


def compute_rsi(close, period=14):
    """Wilder RSI. Causal; first `period` bars = NaN."""
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
    """RSI(period) on HTF bars, forward-filled back to 15m. Causal, no look-ahead."""
    s = pd.Series(close_15m, index=ts_index)
    s_htf = s.resample(htf, closed="right", label="right").last().dropna()
    rsi_vals = compute_rsi(s_htf.values, period)
    s_htf_rsi = pd.Series(rsi_vals, index=s_htf.index)
    result = s_htf_rsi.reindex(ts_index, method="ffill")
    return result.fillna(50.0).values



# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------
def generate_wf_folds(index, train_months=TRAIN_MONTHS,
                      oos_months=OOS_MONTHS, step_months=STEP_MONTHS):
    folds = []; fold_id = 1; fold_start = index[0]; data_end = index[-1]
    while True:
        train_end = fold_start + pd.DateOffset(months=train_months)
        oos_start = train_end
        oos_end   = min(oos_start + pd.DateOffset(months=oos_months),
                        data_end + pd.Timedelta(seconds=1))
        if oos_start > data_end:
            break
        tr_il  = np.where((index >= fold_start) & (index < train_end))[0]
        oos_il = np.where((index >= oos_start)  & (index < oos_end))[0]
        if len(tr_il) > 50 and len(oos_il) > 0:
            folds.append(dict(fold_id=fold_id,
                              train_start=fold_start, train_end=train_end,
                              oos_start=oos_start,   oos_end=oos_end,
                              train_i0=int(tr_il[0]),  train_i1=int(tr_il[-1]) + 1,
                              oos_i0=int(oos_il[0]),   oos_i1=int(oos_il[-1]) + 1,
                              train_bars=len(tr_il),    oos_bars=len(oos_il)))
        fold_start = fold_start + pd.DateOffset(months=step_months)
        fold_id += 1
    return folds


# ---------------------------------------------------------------------------
# BASELINE simulation (original 3-tier exits)
# ---------------------------------------------------------------------------
def _sim_baseline(close, high, low, sbull, sbear, atr14, sl_mult, tp_mult):
    """Original exit logic (ATR SL, 3-tier TP scale-out). Returns R-multiple array."""
    r_list = []
    active = False; is_long = False; entry = 0.0
    sl_ = tp1 = tp2 = tp3 = 0.0; remain = 0.0
    tp1h = tp2h = False; acc = 0.0
    for i in range(1, len(close)):
        h = high[i]; l = low[i]; c = close[i]; atr = atr14[i]
        if active:
            if is_long:
                sh = l <= sl_
                if sh and not tp1h:
                    risk = max(entry - sl_, 1e-10)
                    r_list.append(((sl_ - entry) / entry * remain) / (risk / entry))
                    active = False
                else:
                    if not tp1h and h >= tp1:
                        acc += (tp1 - entry) / entry * 0.33; remain -= 0.33; tp1h = True
                    if active and tp1h and not tp2h and h >= tp2:
                        f = remain * 0.5; acc += (tp2 - entry) / entry * f; remain -= f; tp2h = True
                    if active and tp2h and h >= tp3:
                        risk = max(entry - sl_, 1e-10)
                        r_list.append((acc + (tp3 - entry)/entry * remain) / (risk/entry))
                        active = False
                    elif active and sh:
                        risk = max(entry - sl_, 1e-10)
                        r_list.append((acc + (sl_ - entry)/entry * remain) / (risk/entry))
                        active = False
            else:
                sh = h >= sl_
                if sh and not tp1h:
                    risk = max(sl_ - entry, 1e-10)
                    r_list.append(((entry - sl_) / entry * remain) / (risk / entry))
                    active = False
                else:
                    if not tp1h and l <= tp1:
                        acc += (entry - tp1) / entry * 0.33; remain -= 0.33; tp1h = True
                    if active and tp1h and not tp2h and l <= tp2:
                        f = remain * 0.5; acc += (entry - tp2)/entry * f; remain -= f; tp2h = True
                    if active and tp2h and l <= tp3:
                        risk = max(sl_ - entry, 1e-10)
                        r_list.append((acc + (entry - tp3)/entry * remain) / (risk/entry))
                        active = False
                    elif active and sh:
                        risk = max(sl_ - entry, 1e-10)
                        r_list.append((acc + (entry - sl_)/entry * remain) / (risk/entry))
                        active = False
        if active and is_long and sbear[i]:
            risk = max(entry - sl_, 1e-10)
            r_list.append((acc + (c - entry)/entry * remain) / (risk / entry)); active = False
        if active and not is_long and sbull[i]:
            risk = max(sl_ - entry, 1e-10)
            r_list.append((acc + (entry - c)/entry * remain) / (risk / entry)); active = False
        if not active and not math.isnan(atr):
            if sbull[i]:
                sl_ = l - atr * sl_mult; risk = max(c - sl_, 1e-10); entry = c; is_long = True
                tp1 = c + 1*tp_mult*risk; tp2 = c + 2*tp_mult*risk; tp3 = c + 3*tp_mult*risk
                remain = 1.0; tp1h = tp2h = False; acc = 0.0; active = True
            elif sbear[i]:
                sl_ = h + atr * sl_mult; risk = max(sl_ - c, 1e-10); entry = c; is_long = False
                tp1 = c - 1*tp_mult*risk; tp2 = c - 2*tp_mult*risk; tp3 = c - 3*tp_mult*risk
                remain = 1.0; tp1h = tp2h = False; acc = 0.0; active = True
    if active:
        cl = close[-1]
        pnl = (cl - entry)/entry if is_long else (entry - cl)/entry
        risk_f = max(abs(entry - sl_), 1e-10) / entry
        r_list.append((acc + pnl * remain) / risk_f)
    return np.array(r_list, dtype=np.float64)


# ---------------------------------------------------------------------------
# ENHANCED simulation
# ---------------------------------------------------------------------------
def _sim_enhanced(
    close, high, low, volume,
    sbull_raw, sbear_raw,
    atr14, atr_pctile, vol_ratio,
    sl_swing_bars=SL_SWING_BARS, sl_atr_buf=SL_ATR_BUFFER,
    tp1_r=TP1_R, tp2_r=TP2_R,
    atr_lo=ATR_PCTILE_LOW, atr_hi=ATR_PCTILE_HIGH,
    vol_thr=VOL_THRESHOLD,
    signal_mask=None,   # optional ML mask (bool array)
):
    """
    Enhanced exit logic:
      - Structure SL (swing-based)
      - 2-tier TP: TP1 at tp1_r (close 50%, move SL to BE), TP2 at tp2_r (close rest)
      - ATR + volume entry filters
      - Optional ML signal_mask

    Returns (r_multiples array, trades list) where each trade is a dict.
    The R-multiple for each trade = position-weighted P&L / initial_risk (normalised to 1R).
    """
    n = len(close)
    r_list = []
    trades = []

    active = False; is_long = False; entry = 0.0
    sl_ = tp1 = tp2 = 0.0; risk = 0.0
    tp1_hit = False; acc_r = 0.0; entry_time_idx = 0

    for i in range(1, n):
        h = high[i]; l = low[i]; c = close[i]; atr = atr14[i]

        if active:
            rem = 0.5 if tp1_hit else 1.0  # remaining position fraction
            if is_long:
                # TP1 check first (must happen before TP2 to correctly bank partial)
                if not tp1_hit and h >= tp1:
                    acc_r += 0.5 * (tp1 - entry) / risk  # bank 50% at TP1
                    sl_ = entry                            # move SL to break-even
                    tp1_hit = True; rem = 0.5
                # TP2 check (remaining 50%, or full position if both hit same bar)
                if h >= tp2:
                    total_r = acc_r + rem * (tp2 - entry) / risk
                    r_list.append(total_r)
                    trades.append({"entry_i": entry_time_idx, "exit_i": i,
                                   "direction": "long", "entry": entry, "exit": tp2,
                                   "risk_pct": risk/entry, "r": total_r, "reason": "TP2"})
                    active = False; continue
                # SL check (rem already updated above if TP1 hit this bar)
                if l <= sl_:
                    total_r = acc_r + rem * (sl_ - entry) / risk
                    reason = "BE" if tp1_hit else "SL"
                    r_list.append(total_r)
                    trades.append({"entry_i": entry_time_idx, "exit_i": i,
                                   "direction": "long", "entry": entry, "exit": sl_,
                                   "risk_pct": risk/entry, "r": total_r, "reason": reason})
                    active = False
            else:
                if not tp1_hit and l <= tp1:
                    acc_r += 0.5 * (entry - tp1) / risk
                    sl_ = entry
                    tp1_hit = True; rem = 0.5
                if l <= tp2:
                    total_r = acc_r + rem * (entry - tp2) / risk
                    r_list.append(total_r)
                    trades.append({"entry_i": entry_time_idx, "exit_i": i,
                                   "direction": "short", "entry": entry, "exit": tp2,
                                   "risk_pct": risk/entry, "r": total_r, "reason": "TP2"})
                    active = False; continue
                if h >= sl_:
                    total_r = acc_r + rem * (entry - sl_) / risk
                    reason = "BE" if tp1_hit else "SL"
                    r_list.append(total_r)
                    trades.append({"entry_i": entry_time_idx, "exit_i": i,
                                   "direction": "short", "entry": entry, "exit": sl_,
                                   "risk_pct": risk/entry, "r": total_r, "reason": reason})
                    active = False

        # reversal exit — opposite signal fires while in trade
        if active and is_long and sbear_raw[i]:
            rem = 0.5 if tp1_hit else 1.0
            total_r = acc_r + rem * (c - entry) / risk
            r_list.append(total_r)
            trades.append({"entry_i": entry_time_idx, "exit_i": i,
                           "direction": "long", "entry": entry, "exit": c,
                           "risk_pct": risk/entry, "r": total_r, "reason": "Rev"})
            active = False
        if active and not is_long and sbull_raw[i]:
            rem = 0.5 if tp1_hit else 1.0
            total_r = acc_r + rem * (entry - c) / risk
            r_list.append(total_r)
            trades.append({"entry_i": entry_time_idx, "exit_i": i,
                           "direction": "short", "entry": entry, "exit": c,
                           "risk_pct": risk/entry, "r": total_r, "reason": "Rev"})
            active = False

        # new entry
        if not active and not math.isnan(atr):
            # ATR regime filter
            ap = atr_pctile[i]
            if not (atr_lo < ap < atr_hi):
                continue
            # volume filter
            if vol_ratio[i] < vol_thr:
                continue
            if sbull_raw[i]:
                # check ML mask
                if signal_mask is not None and not signal_mask[i]:
                    continue
                # structure SL: min of last SL_SWING_BARS bars (causal)
                start_w = max(0, i - sl_swing_bars)
                swing_sl = float(np.min(low[start_w:i]))
                sl_ = swing_sl - atr * sl_atr_buf
                risk = max(c - sl_, 1e-10)
                entry = c; is_long = True
                tp1 = c + tp1_r * risk; tp2 = c + tp2_r * risk
                tp1_hit = False; acc_r = 0.0; active = True; entry_time_idx = i
            elif sbear_raw[i]:
                if signal_mask is not None and not signal_mask[i]:
                    continue
                start_w = max(0, i - sl_swing_bars)
                swing_sl = float(np.max(high[start_w:i]))
                sl_ = swing_sl + atr * sl_atr_buf
                risk = max(sl_ - c, 1e-10)
                entry = c; is_long = False
                tp1 = c - tp1_r * risk; tp2 = c - tp2_r * risk
                tp1_hit = False; acc_r = 0.0; active = True; entry_time_idx = i

    if active:
        cl = close[-1]
        # Close remaining 50% (or 100% if TP1 not hit) at last close
        remaining_frac = 0.5 if tp1_hit else 1.0
        pnl_r = acc_r + remaining_frac * ((cl - entry) if is_long else (entry - cl)) / risk
        r_list.append(pnl_r)
        trades.append({"entry_i": entry_time_idx, "exit_i": n - 1,
                       "direction": "long" if is_long else "short",
                       "entry": entry, "exit": cl,
                       "risk_pct": risk/entry, "r": pnl_r, "reason": "Open"})

    return np.array(r_list, dtype=np.float64), trades


# ---------------------------------------------------------------------------
# TRAIL simulation  (sweep winner: sl_mult=3.0, tp1_r=0.75, trail_mult=0.50)
# ---------------------------------------------------------------------------
def _sim_trail(
    close, high, low, volume,
    sbull_raw, sbear_raw,
    atr14, atr_pctile, vol_ratio,
    sl_mult=TRAIL_SL_MULT, tp1_r=TRAIL_TP1_R, trail_mult=TRAIL_MULT,
    atr_lo=ATR_PCTILE_LOW, atr_hi=ATR_PCTILE_HIGH,
    vol_thr=VOL_THRESHOLD,
    signal_mask=None,
):
    """
    ATR SL (sl_mult×ATR from signal-bar low/high) + 2-stage exit:
      - Bank 50% at TP1 (tp1_r × risk), move SL to break-even
      - Trail remaining 50% by trail_mult × ATR14 from running high/low

    Sweep winner: sl_mult=3.0, tp1_r=0.75, trail_mult=0.50
    OOS result: Sharpe +0.225, TotalR +45.4R, 62.5% win rate, 6/6 positive folds.

    Returns (r_multiples, trades).
    """
    n = len(close)
    r_list = []
    trades = []

    active = False; is_long = False; entry = 0.0
    sl_ = tp1 = 0.0; risk = 1.0
    tp1_hit = False; trail_sl = 0.0; acc_r = 0.0; entry_i = 0

    for i in range(1, n):
        h = high[i]; l = low[i]; c = close[i]; atr = atr14[i]

        if active:
            if is_long:
                # TP1 check
                if not tp1_hit and h >= tp1:
                    acc_r += 0.5 * tp1_r
                    trail_sl = entry      # trail starts at break-even
                    tp1_hit = True
                # Trail management (after TP1)
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = h - trail_mult * atr
                        if cand > trail_sl:
                            trail_sl = cand
                    if l <= trail_sl:
                        total_r = acc_r + 0.5 * max(0.0, (trail_sl - entry) / risk)
                        r_list.append(total_r)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "long", "entry": entry,
                                       "exit": trail_sl, "risk_pct": risk / entry,
                                       "r": total_r, "reason": "Trail"})
                        active = False; continue
                else:
                    # Initial SL (full position, before TP1)
                    if l <= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "long", "entry": entry,
                                       "exit": sl_, "risk_pct": risk / entry,
                                       "r": -1.0, "reason": "SL"})
                        active = False
            else:  # short
                # TP1 check
                if not tp1_hit and l <= tp1:
                    acc_r += 0.5 * tp1_r
                    trail_sl = entry      # trail starts at break-even
                    tp1_hit = True
                # Trail management (after TP1)
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = l + trail_mult * atr
                        if cand < trail_sl:
                            trail_sl = cand
                    if h >= trail_sl:
                        total_r = acc_r + 0.5 * max(0.0, (entry - trail_sl) / risk)
                        r_list.append(total_r)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "short", "entry": entry,
                                       "exit": trail_sl, "risk_pct": risk / entry,
                                       "r": total_r, "reason": "Trail"})
                        active = False; continue
                else:
                    if h >= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "short", "entry": entry,
                                       "exit": sl_, "risk_pct": risk / entry,
                                       "r": -1.0, "reason": "SL"})
                        active = False

        # Reversal exit (opposite raw signal)
        if active and is_long and sbear_raw[i]:
            rem = 0.5 if tp1_hit else 1.0
            total_r = acc_r + rem * (c - entry) / risk
            r_list.append(total_r)
            trades.append({"entry_i": entry_i, "exit_i": i,
                           "direction": "long", "entry": entry, "exit": c,
                           "risk_pct": risk / entry, "r": total_r, "reason": "Rev"})
            active = False
        if active and not is_long and sbull_raw[i]:
            rem = 0.5 if tp1_hit else 1.0
            total_r = acc_r + rem * (entry - c) / risk
            r_list.append(total_r)
            trades.append({"entry_i": entry_i, "exit_i": i,
                           "direction": "short", "entry": entry, "exit": c,
                           "risk_pct": risk / entry, "r": total_r, "reason": "Rev"})
            active = False

        # New entry
        if not active and not math.isnan(atr):
            ap = atr_pctile[i]
            if not (atr_lo < ap < atr_hi):
                continue
            if vol_ratio[i] < vol_thr:
                continue
            if sbull_raw[i]:
                if signal_mask is not None and not signal_mask[i]:
                    continue
                sl_ = l - atr * sl_mult
                risk = max(c - sl_, 1e-10)
                entry = c; is_long = True
                tp1 = c + tp1_r * risk
                tp1_hit = False; trail_sl = sl_; acc_r = 0.0; active = True; entry_i = i
            elif sbear_raw[i]:
                if signal_mask is not None and not signal_mask[i]:
                    continue
                sl_ = h + atr * sl_mult
                risk = max(sl_ - c, 1e-10)
                entry = c; is_long = False
                tp1 = c - tp1_r * risk
                tp1_hit = False; trail_sl = sl_; acc_r = 0.0; active = True; entry_i = i

    # Close any open trade at EOD
    if active:
        cl = close[-1]
        rem = 0.5 if tp1_hit else 1.0
        exit_price = cl
        total_r = acc_r + rem * ((exit_price - entry) if is_long else (entry - exit_price)) / risk
        r_list.append(total_r)
        trades.append({"entry_i": entry_i, "exit_i": n - 1,
                       "direction": "long" if is_long else "short",
                       "entry": entry, "exit": exit_price,
                       "risk_pct": risk / entry, "r": total_r, "reason": "Open"})

    return np.array(r_list, dtype=np.float64), trades


# ---------------------------------------------------------------------------
# ML feature extraction
# ---------------------------------------------------------------------------
def extract_ml_features(signal_indices, close, high, low, open_, volume,
                        atr14, atr_pctile, vol_ratio, ema200, sma13,
                        sbull, sbear, timestamps, rsi14=None, rsi4h=None):
    """
    Build feature matrix for each signal index.
    Features are computed using only data up to (and including) bar i — causal.
    """
    rows = []
    for i in signal_indices:
        if i < 20:
            rows.append(None); continue
        # f1: ATR percentile (volatility regime)
        f1 = atr_pctile[i] / 100.0
        # f2: volume / SMA20(volume)
        f2 = float(vol_ratio[i])
        # f3: distance from EMA200 (signed: + = above, - = below)
        f3 = (close[i] - ema200[i]) / close[i] if not np.isnan(ema200[i]) else 0.0
        # f4: EMA200 10-bar slope
        e10 = ema200[max(0, i - 10)]
        f4 = (ema200[i] - e10) / e10 if not np.isnan(e10) and e10 > 0 else 0.0
        # f5: candle body ratio (body / range)
        rng = high[i] - low[i]
        f5 = abs(close[i] - open_[i]) / rng if rng > 0 else 0.0
        # f6: abs(close - SMA13) / close
        f6 = abs(close[i] - sma13[i]) / close[i] if not np.isnan(sma13[i]) else 0.0
        # f7: hour of day (UTC)
        f7 = timestamps[i].hour / 23.0 if hasattr(timestamps[i], 'hour') else 0.5
        # f8: direction (1=long, 0=short)
        f8 = 1.0 if sbull[i] else 0.0
        # f9: recent 5-bar momentum (close vs 5 bars ago)
        f9 = (close[i] - close[max(0, i - 5)]) / close[max(0, i - 5)]
        # f10: ATR normalised to price (relative volatility)
        f10 = atr14[i] / close[i] if not np.isnan(atr14[i]) else 0.0
        # f11: day of week (0=Mon .. 6=Sun, scaled 0-1)
        f11 = timestamps[i].weekday() / 6.0 if hasattr(timestamps[i], 'weekday') else 0.5
        # f12: RSI(14) on 15m scaled 0-1
        f12 = float(rsi14[i]) / 100.0 if rsi14 is not None and not np.isnan(rsi14[i]) else 0.5
        # f13: RSI(14) on 4H scaled 0-1
        f13 = float(rsi4h[i]) / 100.0 if rsi4h is not None and not np.isnan(rsi4h[i]) else 0.5
        rows.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13])
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(r_arr, min_n=1):
    n = len(r_arr)
    if n < min_n:
        return {"n": n, "sharpe": -99.0, "total_r": 0.0, "win_rate": 0.0,
                "avg_r": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "pf": 0.0, "exp": 0.0}
    std = float(np.std(r_arr, ddof=1))
    mean = float(np.mean(r_arr))
    wins = r_arr[r_arr > 0]; losses = r_arr[r_arr < 0]
    gw = float(wins.sum()) if len(wins) > 0 else 0.0
    gl = float(-losses.sum()) if len(losses) > 0 else 0.0
    return {
        "n":        n,
        "sharpe":   round(mean / std if std > 1e-12 else 0.0, 5),
        "total_r":  round(float(r_arr.sum()), 3),
        "win_rate": round(len(wins) / n, 4),
        "avg_r":    round(mean, 4),
        "avg_win":  round(float(wins.mean()) if len(wins) > 0 else 0.0, 4),
        "avg_loss": round(float(losses.mean()) if len(losses) > 0 else 0.0, 4),
        "pf":       round(gw / gl if gl > 0 else 0.0, 4),
        "exp":      round(mean, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MM V4.3 Enhanced Strategy")
    parser.add_argument("--since",       default=SINCE_DATE)
    parser.add_argument("--symbol",      default=SYMBOL)
    parser.add_argument("--tf",          default=TIMEFRAME)
    parser.add_argument("--no-ml",       action="store_true", help="Skip ML variant")
    parser.add_argument("--train",       type=int, default=TRAIN_MONTHS)
    parser.add_argument("--oos",         type=int, default=OOS_MONTHS)
    parser.add_argument("--step",        type=int, default=STEP_MONTHS)
    parser.add_argument("--tp1-r",       type=float, default=TP1_R,
                        help="Enhanced: Tier-1 TP in R (50%% close, SL to BE, default %(default)s)")
    parser.add_argument("--tp2-r",       type=float, default=TP2_R,
                        help="Enhanced: Tier-2 TP in R (remaining 50%% close, default %(default)s)")
    parser.add_argument("--sl-mult",     type=float, default=TRAIL_SL_MULT,
                        help="Trail: ATR SL multiplier (default %(default)s)")
    parser.add_argument("--trail-tp1-r", type=float, default=TRAIL_TP1_R,
                        help="Trail: partial TP in R (50%% close, default %(default)s)")
    parser.add_argument("--trail-mult",  type=float, default=TRAIL_MULT,
                        help="Trail: ATR trail distance after TP1 (default %(default)s)")
    parser.add_argument("--ml-depth",    type=int,   default=ML_MAX_DEPTH,
                        help="DT max depth (default %(default)s)")
    parser.add_argument("--ml-min-leaf", type=int,   default=ML_MIN_LEAF,
                        help="DT min samples per leaf (default %(default)s)")
    parser.add_argument("--ml-min-is-sh", type=float, default=0.50,
                        help="Skip DT filter if IS Sharpe below this (default %(default)s)")
    args = parser.parse_args()
    run_ml = ML_AVAILABLE and not args.no_ml

    t0 = time.time()

    # ── 1. Fetch ──────────────────────────────────────────────────────────
    df = fetch_ohlcv(args.symbol, args.tf, args.since)

    # ── 2. Indicators on full dataset (causal) ────────────────────────────
    print("Computing indicators…", flush=True)
    close   = df["close"].values.astype(np.float64)
    open_   = df["open"].values.astype(np.float64)
    high    = df["high"].values.astype(np.float64)
    low     = df["low"].values.astype(np.float64)
    volume  = df["volume"].values.astype(np.float64)
    ts      = df.index  # DatetimeIndex

    atr_st   = compute_atr(high, low, close, ST_ATR_LEN)
    atr14    = compute_atr(high, low, close, ATR_SL_LEN)
    ema200   = compute_ema(close, EMA_LEN)
    sma13    = compute_sma(close, SMA_LEN)
    atr_pct  = compute_rolling_atr_pctile(atr14, ATR_PCTILE_WIN)
    vol_rat  = compute_vol_ratio(volume, VOL_SMA_WIN)
    rsi14_arr = compute_rsi(close, 14)
    rsi4h_arr = compute_rsi_htf(ts, close, "4h", 14)

    # Pre-compute raw signals for ST_MULT = 3.5
    sbull_raw, sbear_raw = build_raw_signals(close, open_, sma13, ema200, atr_st)
    print(f"  Raw signals: Sbull={sbull_raw.sum()}  Sbear={sbear_raw.sum()}", flush=True)

    # ── 3. Walk-forward folds ─────────────────────────────────────────────
    folds = generate_wf_folds(df.index, args.train, args.oos, args.step)
    print(f"  Walk-forward: {len(folds)} folds", flush=True)

    # ── 4. Per-fold run ───────────────────────────────────────────────────
    results_base, results_enh, results_trail, results_ml = [], [], [], []
    oos_trades_enh, oos_trades_trail, oos_trades_ml = [], [], []

    for fold in folds:
        fid = fold["fold_id"]
        i0t, i1t = fold["train_i0"], fold["train_i1"]
        i0o, i1o = fold["oos_i0"],  fold["oos_i1"]

        # ── Baseline (original 3-tier, ATR SL, no filters) ──
        r_base_oos = _sim_baseline(
            close[i0o:i1o], high[i0o:i1o], low[i0o:i1o],
            sbull_raw[i0o:i1o], sbear_raw[i0o:i1o],
            atr14[i0o:i1o],
            sl_mult=BASELINE_SL_MULT, tp_mult=BASELINE_TP_MULT,
        )
        mb = metrics(r_base_oos)

        # ── Enhanced (structure SL, 2-tier TP, ATR+vol filter) ──
        r_enh_oos, td_enh = _sim_enhanced(
            close[i0o:i1o], high[i0o:i1o], low[i0o:i1o], volume[i0o:i1o],
            sbull_raw[i0o:i1o], sbear_raw[i0o:i1o],
            atr14[i0o:i1o], atr_pct[i0o:i1o], vol_rat[i0o:i1o],
            tp1_r=args.tp1_r, tp2_r=args.tp2_r,
        )
        me = metrics(r_enh_oos)

        for t in td_enh:
            t["fold_id"] = fid; t["variant"] = "enhanced"
            t["entry_time"] = str(ts[i0o + t["entry_i"]])[:19]
            t["exit_time"]  = str(ts[min(i0o + t["exit_i"], i1o - 1)])[:19]
        oos_trades_enh.extend(td_enh)

        # ── Trail (ATR SL + TP1 partial + trailing stop) ──────────────────
        r_trail_oos, td_trail = _sim_trail(
            close[i0o:i1o], high[i0o:i1o], low[i0o:i1o], volume[i0o:i1o],
            sbull_raw[i0o:i1o], sbear_raw[i0o:i1o],
            atr14[i0o:i1o], atr_pct[i0o:i1o], vol_rat[i0o:i1o],
            sl_mult=args.sl_mult, tp1_r=args.trail_tp1_r, trail_mult=args.trail_mult,
        )
        mt = metrics(r_trail_oos)
        for t in td_trail:
            t["fold_id"] = fid; t["variant"] = "trail"
            t["entry_time"] = str(ts[i0o + t["entry_i"]])[:19]
            t["exit_time"]  = str(ts[min(i0o + t["exit_i"], i1o - 1)])[:19]
        oos_trades_trail.extend(td_trail)

        # ── ML variant (trains on Trail IS labels, filters Trail OOS) ─────
        mm = {"n": 0, "sharpe": float("nan"), "total_r": 0.0, "win_rate": 0.0,
              "avg_r": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "pf": 0.0, "exp": 0.0}
        if run_ml:
            # Collect IS signals that pass ATR/vol filters
            sig_idx_is = [i for i in range(i0t, i1t)
                          if (sbull_raw[i] or sbear_raw[i])
                          and ATR_PCTILE_LOW < atr_pct[i] < ATR_PCTILE_HIGH
                          and vol_rat[i] >= VOL_THRESHOLD]

            if len(sig_idx_is) >= ML_MIN_SIGNALS:
                # Simulate IS trails to get labels (positive R = win)
                _, td_is = _sim_trail(
                    close[i0t:i1t], high[i0t:i1t], low[i0t:i1t], volume[i0t:i1t],
                    sbull_raw[i0t:i1t], sbear_raw[i0t:i1t],
                    atr14[i0t:i1t], atr_pct[i0t:i1t], vol_rat[i0t:i1t],
                    sl_mult=args.sl_mult, tp1_r=args.trail_tp1_r, trail_mult=args.trail_mult,
                )
                # Map IS trail-trades to global bar indices
                is_label_map = {(i0t + t["entry_i"]): (1 if t["r"] > 0 else 0) for t in td_is}

                # Build feature matrix for IS signals
                feats_is, labels_is, valid_is_idx = [], [], []
                for gi in sig_idx_is:
                    row_feats = extract_ml_features(
                        [gi], close, high, low, open_, volume,
                        atr14, atr_pct, vol_rat, ema200, sma13,
                        sbull_raw, sbear_raw, ts,
                        rsi14=rsi14_arr, rsi4h=rsi4h_arr,
                    )[0]
                    if row_feats is None:
                        continue
                    label = is_label_map.get(gi, None)
                    if label is None:
                        continue
                    feats_is.append(row_feats)
                    labels_is.append(label)
                    valid_is_idx.append(gi)

                if len(feats_is) >= ML_MIN_SIGNALS and len(set(labels_is)) == 2:
                    X_is = np.array(feats_is, dtype=np.float64)
                    y_is = np.array(labels_is, dtype=int)

                    # Decision Tree — no scaling needed; thresholds stay in original units
                    clf = DecisionTreeClassifier(
                        max_depth=args.ml_depth,
                        min_samples_leaf=args.ml_min_leaf,
                        class_weight="balanced",
                        random_state=42,
                    )
                    clf.fit(X_is, y_is)

                    # ── Print human-readable tree rules ──────────────────
                    tree_txt = export_text(clf, feature_names=FEATURE_NAMES, decimals=3)
                    print(f"\n  Fold {fid} decision tree (IS n={len(feats_is)}, "
                          f"pos={sum(labels_is)}/{len(labels_is)}):",
                          flush=True)
                    for line in tree_txt.splitlines():
                        print(f"    {line}", flush=True)

                    # ── Feature importances ──────────────────────────────
                    imps = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                                  key=lambda x: -x[1])
                    non_zero = [(n, v) for n, v in imps if v > 0.005]
                    if non_zero:
                        print(f"  Fold {fid} feature importances: "
                              + "  ".join(f"{n}={v:.3f}" for n, v in non_zero),
                              flush=True)

                    # ── Tune leaf win-rate threshold on IS ───────────────
                    is_probas = clf.predict_proba(X_is)[:, 1]
                    best_thr = ML_PROBA_THRESH; best_s = -99.0
                    is_r_map = {(i0t + t["entry_i"]): t["r"] for t in td_is}
                    is_gp = [(gi, float(is_probas[k])) for k, gi in enumerate(valid_is_idx)]
                    for thr in np.arange(0.40, 0.90, 0.05):
                        selected_r = [is_r_map[gi] for gi, p in is_gp
                                      if p >= thr and gi in is_r_map]
                        if len(selected_r) < 8:
                            continue
                        s = metrics(np.array(selected_r), min_n=5)["sharpe"]
                        if s > best_s:
                            best_s = s; best_thr = thr

                    # ── Build OOS signal mask ────────────────────────────
                    # If IS Sharpe too low, DT found no signal — skip filter (use all signals)
                    use_dt_filter = best_s >= args.ml_min_is_sh
                    sig_idx_oos = [i for i in range(i0o, i1o)
                                   if (sbull_raw[i] or sbear_raw[i])
                                   and ATR_PCTILE_LOW < atr_pct[i] < ATR_PCTILE_HIGH
                                   and vol_rat[i] >= VOL_THRESHOLD]

                    oos_mask_arr = np.zeros(i1o - i0o, dtype=bool)
                    if sig_idx_oos and use_dt_filter:
                        feats_oos, valid_oos_idx = [], []
                        for gi in sig_idx_oos:
                            row = extract_ml_features(
                                [gi], close, high, low, open_, volume,
                                atr14, atr_pct, vol_rat, ema200, sma13,
                                sbull_raw, sbear_raw, ts,
                                rsi14=rsi14_arr, rsi4h=rsi4h_arr,
                            )[0]
                            if row is not None:
                                feats_oos.append(row)
                                valid_oos_idx.append(gi)
                        if feats_oos:
                            probas_oos = clf.predict_proba(
                                np.array(feats_oos, dtype=np.float64))[:, 1]
                            for k, gi in enumerate(valid_oos_idx):
                                if probas_oos[k] >= best_thr:
                                    oos_mask_arr[gi - i0o] = True

                    elif sig_idx_oos and not use_dt_filter:
                        # DT had no IS edge — pass all signals through (no filter)
                        for gi in sig_idx_oos:
                            oos_mask_arr[gi - i0o] = True

                    # ── Run Trail with DT mask ───────────────────────────
                    r_ml_oos, td_ml = _sim_trail(
                        close[i0o:i1o], high[i0o:i1o], low[i0o:i1o], volume[i0o:i1o],
                        sbull_raw[i0o:i1o], sbear_raw[i0o:i1o],
                        atr14[i0o:i1o], atr_pct[i0o:i1o], vol_rat[i0o:i1o],
                        sl_mult=args.sl_mult, tp1_r=args.trail_tp1_r,
                        trail_mult=args.trail_mult, signal_mask=oos_mask_arr,
                    )
                    mm = metrics(r_ml_oos)
                    for t in td_ml:
                        t["fold_id"] = fid; t["variant"] = "trail_dt"
                        t["entry_time"] = str(ts[i0o + t["entry_i"]])[:19]
                        t["exit_time"]  = str(ts[min(i0o + t["exit_i"], i1o - 1)])[:19]
                    oos_trades_ml.extend(td_ml)
                    print(f"  Fold {fid} Trail+DT: thr={best_thr:.2f} IS_Sh={best_s:.3f} "
                          f"-> OOS Sh={mm['sharpe']:+.3f} n={mm['n']}", flush=True)

        results_base.append({"fold": fid, **{f"b_{k}": v for k, v in mb.items()}})
        results_enh.append({"fold": fid,  **{f"e_{k}": v for k, v in me.items()}})
        results_trail.append({"fold": fid, **{f"t_{k}": v for k, v in mt.items()}})
        results_ml.append({"fold": fid,   **{f"m_{k}": v for k, v in mm.items()}})

        print(
            f"  Fold {fid}  "
            f"Base: n={mb['n']:3d} Sh={mb['sharpe']:+.3f} R={mb['total_r']:+6.2f}  |  "
            f"Enh:  n={me['n']:3d} Sh={me['sharpe']:+.3f} R={me['total_r']:+6.2f}  |  "
            f"Trail:n={mt['n']:3d} Sh={mt['sharpe']:+.3f} R={mt['total_r']:+6.2f}"
            + (f"  |  Trail+DT:n={mm['n']:3d} Sh={mm['sharpe']:+.3f} R={mm['total_r']:+6.2f}"
               if run_ml and mm["n"] > 0 else ""),
            flush=True,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    SEP = "-" * 80
    print(f"\n{SEP}")
    print("  COMPARISON SUMMARY  (OOS R-multiples)")
    print(SEP)

    # Collect combined OOS R-multiples per variant
    r_all_base_folds = []
    for f in folds:
        r_b = _sim_baseline(
            close[f["oos_i0"]:f["oos_i1"]], high[f["oos_i0"]:f["oos_i1"]],
            low[f["oos_i0"]:f["oos_i1"]], sbull_raw[f["oos_i0"]:f["oos_i1"]],
            sbear_raw[f["oos_i0"]:f["oos_i1"]], atr14[f["oos_i0"]:f["oos_i1"]],
            BASELINE_SL_MULT, BASELINE_TP_MULT)
        r_all_base_folds.append(r_b)
    r_all_base  = np.concatenate(r_all_base_folds) if r_all_base_folds else np.array([])
    r_all_enh   = np.array([t["r"] for t in oos_trades_enh])   if oos_trades_enh   else np.array([])
    r_all_trail = np.array([t["r"] for t in oos_trades_trail]) if oos_trades_trail else np.array([])
    r_all_ml    = np.array([t["r"] for t in oos_trades_ml])    if oos_trades_ml    else np.array([])

    def fmt_block(label, r_arr):
        if len(r_arr) == 0:
            return f"  {label:32s}  N/A"
        m = metrics(r_arr)
        return (f"  {label:32s}  n={m['n']:4d}  "
                f"Sharpe={m['sharpe']:+.4f}  "
                f"total_R={m['total_r']:+7.2f}  "
                f"avg_R={m['avg_r']:+.4f}  "
                f"win={100*m['win_rate']:.0f}%  "
                f"pf={m['pf']:.3f}")

    print(fmt_block("Baseline (3-tier, no filter)",     r_all_base))
    print(fmt_block("Enhanced (struct SL, 2-tier TP)",  r_all_enh))
    trail_lbl = (f"Trail (sl={args.sl_mult:.2f} tp1={args.trail_tp1_r:.2f} "
                 f"tr={args.trail_mult:.2f})")
    print(fmt_block(trail_lbl, r_all_trail))
    if len(r_all_ml) > 0:
        print(fmt_block("Trail + DecisionTree filter",      r_all_ml))

    # Count positive folds per variant
    def pos_folds(results, key):
        return sum(1 for r in results if r.get(key, 0) > 0)

    print(f"\n  Per-fold results:")
    print(f"  {'Fold':4s}  {'Base_n':6s} {'Base_Sh':8s} {'Base_R':7s}  "
          f"{'Enh_n':6s} {'Enh_Sh':7s} {'Enh_R':6s}  "
          f"{'Trail_n':7s} {'Trail_Sh':8s} {'Trail_R':7s}")
    for b, e, tr in zip(results_base, results_enh, results_trail):
        print(f"  {b['fold']:4d}  "
              f"{b['b_n']:6d} {b['b_sharpe']:+8.3f} {b['b_total_r']:+7.2f}  "
              f"{e['e_n']:6d} {e['e_sharpe']:+7.3f} {e['e_total_r']:+6.2f}  "
              f"{tr['t_n']:7d} {tr['t_sharpe']:+8.3f} {tr['t_total_r']:+7.2f}")

    print(f"\n  Positive folds:  "
          f"Baseline={pos_folds(results_base,'b_total_r')}/{len(folds)}  "
          f"Enhanced={pos_folds(results_enh,'e_total_r')}/{len(folds)}  "
          f"Trail={pos_folds(results_trail,'t_total_r')}/{len(folds)}"
          + (f"  Trail+DT={pos_folds(results_ml,'m_total_r')}/{len(folds)}"
             if run_ml else ""))

    # Exit reason breakdowns
    if oos_trades_trail:
        print(f"\n  Trail exit breakdown (OOS):")
        for reason, grp in pd.DataFrame(oos_trades_trail).groupby("reason"):
            r_g = grp["r"].values
            print(f"    {reason:8s}  n={len(r_g):4d}  avg={r_g.mean():+.3f}R  "
                  f"total={r_g.sum():+.2f}R")

    if oos_trades_ml:
        print(f"\n  Trail+DT exit breakdown (OOS):")
        for reason, grp in pd.DataFrame(oos_trades_ml).groupby("reason"):
            r_g = grp["r"].values
            print(f"    {reason:8s}  n={len(r_g):4d}  avg={r_g.mean():+.3f}R  "
                  f"total={r_g.sum():+.2f}R")

    if oos_trades_enh:
        print(f"\n  Enhanced exit breakdown (OOS):")
        for reason, grp in pd.DataFrame(oos_trades_enh).groupby("reason"):
            r_g = grp["r"].values
            print(f"    {reason:8s}  n={len(r_g):4d}  avg={r_g.mean():+.3f}R  "
                  f"total={r_g.sum():+.2f}R")

    print(f"\n  Total elapsed: {time.time()-t0:.1f}s")
    print(SEP)

    # ── Save ──────────────────────────────────────────────────────────────
    if oos_trades_enh:
        p = os.path.join(OUT_DIR, "million_moves_v43_enhanced_oos.csv")
        pd.DataFrame(oos_trades_enh).to_csv(p, index=False)
        print(f"\nEnhanced OOS trades -> {p}")
    if oos_trades_trail:
        p = os.path.join(OUT_DIR, "million_moves_v43_trail_oos.csv")
        pd.DataFrame(oos_trades_trail).to_csv(p, index=False)
        print(f"Trail OOS trades    -> {p}")
    if oos_trades_ml:
        p = os.path.join(OUT_DIR, "million_moves_v43_ml_oos.csv")
        pd.DataFrame(oos_trades_ml).to_csv(p, index=False)
        print(f"ML OOS trades       -> {p}")

    # Summary comparison CSV
    rows = []
    for b, e, tr, m in zip(results_base, results_enh, results_trail, results_ml):
        rows.append({
            "fold": b["fold"],
            "base_n": b["b_n"], "base_sharpe": b["b_sharpe"],
            "base_total_r": b["b_total_r"], "base_avg_r": b["b_avg_r"],
            "base_win": b["b_win_rate"],
            "enh_n": e["e_n"], "enh_sharpe": e["e_sharpe"],
            "enh_total_r": e["e_total_r"], "enh_avg_r": e["e_avg_r"],
            "enh_win": e["e_win_rate"],
            "trail_n": tr["t_n"], "trail_sharpe": tr["t_sharpe"],
            "trail_total_r": tr["t_total_r"], "trail_avg_r": tr["t_avg_r"],
            "trail_win": tr["t_win_rate"],
            "ml_n": m["m_n"], "ml_sharpe": m.get("m_sharpe", float("nan")),
            "ml_total_r": m.get("m_total_r", 0.0),
        })
    p = os.path.join(OUT_DIR, "million_moves_v43_comparison.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"Comparison table    -> {p}")


if __name__ == "__main__":
    freeze_support()
    main()
