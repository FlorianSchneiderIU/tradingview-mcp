"""
indicators.py — Shared indicator logic for bot.py and train_dt.py.
All functions are pure transforms on numpy arrays.  No I/O, no network calls.

Strategy constants must match million_moves_v43_multi.py exactly.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ── Strategy constants ─────────────────────────────────────────────────────────
ST_MULT       = 3.5;  ST_ATR_LEN = 11
EMA_LEN       = 200;  SMA_LEN = 13;  ATR_LEN = 14
VOL_WIN       = 20;   ATR_PCTILE_WIN = 100
ATR_LO        = 10;   ATR_HI = 90;   VOL_THR = 1.05

FEATURE_NAMES = [
    "atr_pctile", "vol_ratio",  "ema200_dist", "ema200_slope",
    "body_ratio",  "sma13_dist", "hour_utc",    "direction",
    "mom5",        "atr_norm",   "day_of_week", "rsi14",        "rsi4h",
]
N_FEATURES = len(FEATURE_NAMES)  # 13

# ── Low-level indicators ───────────────────────────────────────────────────────

def _rma(vals: np.ndarray, length: int) -> np.ndarray:
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


def ind_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            length: int) -> np.ndarray:
    pc = np.empty_like(close); pc[0] = np.nan; pc[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return _rma(tr, length)


def ind_ema(close: np.ndarray, length: int) -> np.ndarray:
    alpha = 2.0 / (length + 1)
    out = np.full(len(close), np.nan)
    for i, v in enumerate(close):
        if not np.isnan(v):
            out[i] = v
            for j in range(i + 1, len(close)):
                out[j] = alpha * close[j] + (1.0 - alpha) * out[j - 1]
            break
    return out


def ind_sma(arr: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(length - 1, len(arr)):
        out[i] = float(np.mean(arr[i - length + 1:i + 1]))
    return out


def ind_supertrend(open_: np.ndarray, close: np.ndarray,
                   atr_arr: np.ndarray, mult: float) -> np.ndarray:
    n = len(open_)
    ur = open_ + mult * atr_arr
    lr = open_ - mult * atr_arr
    upper, lower = ur.copy(), lr.copy()
    direction = np.full(n, 2.0)
    st = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(atr_arr[i - 1]):
            direction[i] = 2.0; upper[i] = ur[i]; lower[i] = lr[i]
        else:
            lower[i] = lr[i] if (lr[i] > lower[i-1] or close[i-1] < lower[i-1]) else lower[i-1]
            upper[i] = ur[i] if (ur[i] < upper[i-1] or close[i-1] > upper[i-1]) else upper[i-1]
            ps = st[i - 1]
            if np.isnan(ps):
                ps = upper[i - 1] if not np.isnan(upper[i - 1]) else lower[i - 1]
            if ps == upper[i - 1]:
                direction[i] = -1.0 if close[i] > upper[i] else 1.0
            else:
                direction[i] =  1.0 if close[i] < lower[i] else -1.0
        st[i] = lower[i] if direction[i] == -1.0 else upper[i]
    return st


def build_signals(close: np.ndarray, open_: np.ndarray,
                  sma13: np.ndarray, ema200: np.ndarray,
                  atr_st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    st = ind_supertrend(open_, close, atr_st, ST_MULT)
    n  = len(close)
    pc = np.empty(n); pc[0] = np.nan; pc[1:] = close[:-1]
    ps = np.empty(n); ps[0] = np.nan; ps[1:] = st[:-1]
    pe = np.empty(n); pe[0] = np.nan; pe[1:] = ema200[:-1]
    co    = (~np.isnan(pc)) & (~np.isnan(ps)) & (~np.isnan(st)) & (pc < ps) & (close > st)
    cu    = (~np.isnan(pc)) & (~np.isnan(ps)) & (~np.isnan(st)) & (pc > ps) & (close < st)
    above = (~np.isnan(pe)) & (~np.isnan(ema200)) & (pc > pe) & (close > ema200)
    sbull = co & (~np.isnan(sma13)) & (close >= sma13) &  above
    sbear = cu & (~np.isnan(sma13)) & (close <= sma13) & (~above)
    return sbull.astype(bool), sbear.astype(bool)


# ── ATR percentile ─────────────────────────────────────────────────────────────

def rolling_atr_pctile(atr14: np.ndarray, window: int = ATR_PCTILE_WIN) -> np.ndarray:
    """Full-array version used in training."""
    n = len(atr14)
    out = np.full(n, 50.0)
    for i in range(window, n):
        w = atr14[i - window:i]           # exclude current bar (same as backtest)
        if not np.isnan(atr14[i]) and not np.all(np.isnan(w)):
            valid = w[~np.isnan(w)]
            out[i] = float(np.sum(valid < atr14[i])) / len(valid) * 100.0
    return out


def atr_pctile_last(atr14: np.ndarray, window: int = ATR_PCTILE_WIN) -> float:
    """Scalar version for live inference (equivalent at the last bar)."""
    n = len(atr14)
    if n < window + 1 or np.isnan(atr14[-1]):
        return 50.0
    w = atr14[-(window + 1):-1]
    valid = w[~np.isnan(w)]
    if len(valid) == 0:
        return 50.0
    return float(np.sum(valid < atr14[-1])) / len(valid) * 100.0


# ── Volume ratio ───────────────────────────────────────────────────────────────

def rolling_vol_ratio(volume: np.ndarray, win: int = VOL_WIN) -> np.ndarray:
    """Full-array version used in training.  SMA includes current bar."""
    out = np.full(len(volume), 1.0)
    for i in range(win - 1, len(volume)):
        avg = float(np.mean(volume[i - win + 1:i + 1]))
        if avg > 0:
            out[i] = float(volume[i]) / avg
    return out


def vol_ratio_last(volume: np.ndarray, win: int = VOL_WIN) -> float:
    """Scalar version.  Matches rolling_vol_ratio at the last index."""
    if len(volume) < win:
        return 1.0
    avg = float(np.mean(volume[-win:]))   # includes current bar
    return float(volume[-1]) / avg if avg > 0 else 1.0


# ── RSI ────────────────────────────────────────────────────────────────────────

def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    d     = np.diff(close)
    gain  = np.maximum(d, 0.0); loss = np.maximum(-d, 0.0)
    avg_g = gain[:period].mean(); avg_l = loss[:period].mean()
    rs    = avg_g / avg_l if avg_l > 0 else np.inf
    out[period] = 100.0 - 100.0 / (1.0 + rs)
    for j in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gain[j]) / period
        avg_l = (avg_l * (period - 1) + loss[j]) / period
        rs    = avg_g / avg_l if avg_l > 0 else np.inf
        out[j + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def compute_rsi_htf(ts_index: pd.DatetimeIndex, close_15m: np.ndarray,
                    htf: str = "4h", period: int = 14) -> np.ndarray:
    """Resample close to HTF, compute RSI, forward-fill back to 15m index.
    Used in training (batch, has pandas index available).
    """
    s      = pd.Series(close_15m, index=ts_index)
    s_htf  = s.resample(htf, closed="right", label="right").last().dropna()
    rsi_v  = compute_rsi(s_htf.values, period)
    s_rsi  = pd.Series(rsi_v, index=s_htf.index)
    return s_rsi.reindex(ts_index, method="ffill").fillna(50.0).values


def rsi4h_last_from_bars(bars: list[dict], period: int = 14) -> float:
    """Compute 4H RSI from a list of 15m bar dicts without pandas.
    Groups bars into 4H buckets by UTC hour//4, returns last RSI value.
    """
    four_h_closes: list[float] = []
    cur_bucket = None
    cur_close  = 0.0
    for bar in bars:
        dt_b   = datetime.fromtimestamp(bar["ts"] / 1000, tz=timezone.utc)
        bucket = (dt_b.date(), dt_b.hour // 4)
        if bucket != cur_bucket:
            if cur_bucket is not None:
                four_h_closes.append(cur_close)
            cur_bucket = bucket
        cur_close = bar["close"]
    if cur_bucket is not None:
        four_h_closes.append(cur_close)

    if len(four_h_closes) < period + 2:
        return 50.0
    arr = np.array(four_h_closes, dtype=np.float64)
    rsi = compute_rsi(arr, period)
    last = rsi[-1]
    return float(last) if not np.isnan(last) else 50.0


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features_batch(signal_indices, close, high, low, open_, volume,
                            atr14, atr_pct, vol_rat, ema200, sma13,
                            sbull, sbear, timestamps, rsi14, rsi4h):
    """Batch extraction for training — mirrors multi.py extract_features exactly."""
    rows = []
    for i in signal_indices:
        if i < 20:
            rows.append(None); continue
        f1  = atr_pct[i] / 100.0
        f2  = float(vol_rat[i])
        f3  = (close[i] - ema200[i]) / close[i] if not np.isnan(ema200[i]) else 0.0
        e10 = ema200[max(0, i - 10)]
        f4  = (ema200[i] - e10) / e10 if not np.isnan(e10) and e10 > 0 else 0.0
        rng = high[i] - low[i]
        f5  = abs(close[i] - open_[i]) / rng if rng > 0 else 0.0
        f6  = abs(close[i] - sma13[i]) / close[i] if not np.isnan(sma13[i]) else 0.0
        f7  = timestamps[i].hour    / 23.0 if hasattr(timestamps[i], "hour")    else 0.5
        f8  = 1.0 if sbull[i] else 0.0
        f9  = (close[i] - close[max(0, i - 5)]) / close[max(0, i - 5)]
        f10 = atr14[i] / close[i] if not np.isnan(atr14[i]) else 0.0
        f11 = timestamps[i].weekday() / 6.0 if hasattr(timestamps[i], "weekday") else 0.5
        f12 = float(rsi14[i]) / 100.0 if not np.isnan(rsi14[i]) else 0.5
        f13 = float(rsi4h[i]) / 100.0 if not np.isnan(rsi4h[i]) else 0.5
        rows.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13])
    return rows


def extract_live_feature_vector(bars: list[dict],
                                 is_long: bool) -> Optional[np.ndarray]:
    """
    Compute the 13-element feature vector from the last bar in `bars`.
    bars: list of OHLCV dicts {ts(ms), open, high, low, close, volume}, oldest→newest.
    Returns None if insufficient data for any indicator.
    """
    n = len(bars)
    min_need = max(EMA_LEN + 20, ATR_PCTILE_WIN + 1, SMA_LEN + 5, 28)
    if n < min_need:
        return None

    o  = np.array([b["open"]   for b in bars], dtype=np.float64)
    h  = np.array([b["high"]   for b in bars], dtype=np.float64)
    l  = np.array([b["low"]    for b in bars], dtype=np.float64)
    c  = np.array([b["close"]  for b in bars], dtype=np.float64)
    v  = np.array([b["volume"] for b in bars], dtype=np.float64)

    atr14_arr  = ind_atr(h, l, c, ATR_LEN)
    ema200_arr = ind_ema(c, EMA_LEN)
    sma13_arr  = ind_sma(c, SMA_LEN)

    i = n - 1
    if any(np.isnan(x) for x in (atr14_arr[i], ema200_arr[i], sma13_arr[i])):
        return None

    ap = atr_pctile_last(atr14_arr, ATR_PCTILE_WIN)
    vr = vol_ratio_last(v, VOL_WIN)

    rsi14_arr     = compute_rsi(c, 14)
    rsi4h_val     = rsi4h_last_from_bars(bars, 14)

    dt_bar = datetime.fromtimestamp(bars[i]["ts"] / 1000, tz=timezone.utc)
    e10    = ema200_arr[max(0, i - 10)]
    rng    = h[i] - l[i]

    f1  = ap / 100.0
    f2  = vr
    f3  = (c[i] - ema200_arr[i]) / c[i]
    f4  = (ema200_arr[i] - e10) / e10 if not np.isnan(e10) and e10 > 0 else 0.0
    f5  = abs(c[i] - o[i]) / rng if rng > 0 else 0.0
    f6  = abs(c[i] - sma13_arr[i]) / c[i]
    f7  = dt_bar.hour    / 23.0
    f8  = 1.0 if is_long else 0.0
    f9  = (c[i] - c[max(0, i - 5)]) / c[max(0, i - 5)]
    f10 = atr14_arr[i] / c[i]
    f11 = dt_bar.weekday() / 6.0
    f12 = float(rsi14_arr[i]) / 100.0 if not np.isnan(rsi14_arr[i]) else 0.5
    f13 = rsi4h_val / 100.0

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13],
                    dtype=np.float64)


# ── Trail simulation + metrics (used in train_dt.py) ──────────────────────────

def sim_trail(close, high, low, sbull_raw, sbear_raw,
              atr14, atr_pct, vol_rat,
              sl_mult, tp1_r, trail_mult, signal_mask=None):
    """Identical to _sim_trail in million_moves_v43_multi.py."""
    n = len(close)
    r_list = []; trades = []
    active = False; is_long = False; entry = 0.0
    sl_ = tp1 = 0.0; risk = 1.0
    tp1_hit = False; trail_sl = 0.0; acc_r = 0.0; entry_i = 0

    for i in range(1, n):
        h_ = high[i]; l_ = low[i]; c_ = close[i]; atr_ = atr14[i]

        if active:
            if is_long:
                if not tp1_hit and h_ >= tp1:
                    acc_r += 0.5 * tp1_r; trail_sl = entry; tp1_hit = True
                if tp1_hit:
                    if not math.isnan(atr_):
                        cand = h_ - trail_mult * atr_
                        if cand > trail_sl: trail_sl = cand
                    if l_ <= trail_sl:
                        tot = acc_r + 0.5 * max(0.0, (trail_sl - entry) / risk)
                        r_list.append(tot)
                        trades.append({"entry_i": entry_i, "exit_i": i, "r": tot, "reason": "Trail"})
                        active = False; continue
                else:
                    if l_ <= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i, "r": -1.0, "reason": "SL"})
                        active = False
            else:
                if not tp1_hit and l_ <= tp1:
                    acc_r += 0.5 * tp1_r; trail_sl = entry; tp1_hit = True
                if tp1_hit:
                    if not math.isnan(atr_):
                        cand = l_ + trail_mult * atr_
                        if cand < trail_sl: trail_sl = cand
                    if h_ >= trail_sl:
                        tot = acc_r + 0.5 * max(0.0, (entry - trail_sl) / risk)
                        r_list.append(tot)
                        trades.append({"entry_i": entry_i, "exit_i": i, "r": tot, "reason": "Trail"})
                        active = False; continue
                else:
                    if h_ >= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i, "r": -1.0, "reason": "SL"})
                        active = False

        if active and is_long and sbear_raw[i]:
            rem = 0.5 if tp1_hit else 1.0
            tot = acc_r + rem * (c_ - entry) / risk
            r_list.append(tot); trades.append({"entry_i": entry_i, "exit_i": i, "r": tot, "reason": "Rev"})
            active = False
        if active and not is_long and sbull_raw[i]:
            rem = 0.5 if tp1_hit else 1.0
            tot = acc_r + rem * (entry - c_) / risk
            r_list.append(tot); trades.append({"entry_i": entry_i, "exit_i": i, "r": tot, "reason": "Rev"})
            active = False

        if not active and not math.isnan(atr_):
            ap = atr_pct[i]
            if not (ATR_LO < ap < ATR_HI): continue
            if vol_rat[i] < VOL_THR: continue
            if sbull_raw[i]:
                if signal_mask is not None and not signal_mask[i]: continue
                sl_ = l_ - atr_ * sl_mult; risk = max(c_ - sl_, 1e-10)
                entry = c_; is_long = True; tp1 = c_ + tp1_r * risk
                tp1_hit = False; trail_sl = sl_; acc_r = 0.0; active = True; entry_i = i
            elif sbear_raw[i]:
                if signal_mask is not None and not signal_mask[i]: continue
                sl_ = h_ + atr_ * sl_mult; risk = max(sl_ - c_, 1e-10)
                entry = c_; is_long = False; tp1 = c_ - tp1_r * risk
                tp1_hit = False; trail_sl = sl_; acc_r = 0.0; active = True; entry_i = i

    if active:
        cl  = close[-1]; rem = 0.5 if tp1_hit else 1.0
        tot = acc_r + rem * ((cl - entry) if is_long else (entry - cl)) / risk
        r_list.append(tot)
        trades.append({"entry_i": entry_i, "exit_i": n - 1, "r": tot, "reason": "Open"})

    return np.array(r_list, dtype=np.float64), trades


def metrics(r_arr: np.ndarray, min_n: int = 1) -> dict:
    n = len(r_arr)
    if n < min_n:
        return {"n": n, "sharpe": -99.0, "total_r": 0.0, "win_rate": 0.0, "pf": 0.0}
    std  = float(np.std(r_arr, ddof=1))
    mean = float(np.mean(r_arr))
    wins = r_arr[r_arr > 0]; losses = r_arr[r_arr < 0]
    gw   = float(wins.sum())   if len(wins)   > 0 else 0.0
    gl   = float(-losses.sum()) if len(losses) > 0 else 0.0
    return {
        "n":        n,
        "sharpe":   round(mean / std if std > 1e-12 else 0.0, 4),
        "total_r":  round(float(r_arr.sum()), 2),
        "win_rate": round(len(wins) / n, 3),
        "pf":       round(gw / gl if gl > 0 else 0.0, 3),
    }
