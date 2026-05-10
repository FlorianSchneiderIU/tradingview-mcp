"""
Million Moves V4.3 — Adaptive Lag via Decision Tree
====================================================

Per-coin walk-forward experiment. For each of 5 lag-speed combos
(supertrend ATR len × SMA filter len), trains a decision tree on IS
data to predict whether a trade will be a winner given the regime at
entry. In OOS, picks the highest-confidence combo that fires a signal.

Lag-speed combos:
  speed_0  atr_len= 7  sma_len= 8   very fast / reactive
  speed_1  atr_len= 9  sma_len=10   fast
  speed_2  atr_len=11  sma_len=13   BASELINE (walk-forward consensus)
  speed_3  atr_len=14  sma_len=16   slow
  speed_4  atr_len=21  sma_len=21   very slow / trend-following

Regime features (all causal at the signal bar):
  adx14        ADX(14)          — trend strength  (0-1 scaled)
  atr_pctile   100-bar ATR pct  — volatility regime (0-1)
  rsi14        RSI(14)          — momentum (0-1)
  ema200_dist  (close-EMA200)/close signed — trend distance
  ema200_slope EMA200 10-bar slope / EMA200 — trend acceleration
  vol_ratio    volume / SMA20(vol) — participation
  mom5         5-bar close return — short momentum
  body_ratio   |open-close| / (high-low) — candle conviction
  direction    1=long / 0=short

Training per combo (strict anti-lookahead):
  - Run sequential IS backtest → list of (entry_bar, R)
  - Build features at each entry_bar (causal)
  - Label: 1 if R > 0, else 0
  - Train DecisionTreeClassifier(max_depth=3, min_samples_leaf=6)

OOS inference:
  - For each OOS bar, compute regime features
  - For every combo that fires a signal at this bar, predict P(win)
  - Pick the combo with highest P(win) (ties: prefer baseline)
  - If best P(win) < threshold (default 0.50), skip the bar
  - Simulate that combo's trade with trail exit

Exit (same for all combos — trail sweep winner):
  SL   = signal_bar low  - 3.0 × ATR14   (long)
         signal_bar high + 3.0 × ATR14   (short)
  TP1 @ 0.75R  →  bank 50%,  move SL to break-even
  Trail remaining 50% by 0.5 × ATR14

ATR filter: ATR percentile 10–90  (same as enhanced)
Volume filter: volume > SMA20 × 1.05  (same as enhanced)

Walk-forward: 12m train / 3m OOS / 3m step
Data source:  Bybit linear perpetuals via ccxt (same as live bot)

Usage
-----
  python scripts/million_moves_v43_adaptive_lag.py
  python scripts/million_moves_v43_adaptive_lag.py --coins ETH BTC ARB GRT
  python scripts/million_moves_v43_adaptive_lag.py --since 2024-01-01
  python scripts/million_moves_v43_adaptive_lag.py --max-depth 4 --threshold 0.55
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
import ccxt

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    SKLEARN_OK = True
except ImportError:
    print("ERROR: scikit-learn not installed.  pip install scikit-learn")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ─── Global config ─────────────────────────────────────────────────────────
SINCE_DATE   = "2024-01-01"
TIMEFRAME    = "15m"
TRAIN_M, OOS_M, STEP_M = 12, 3, 3

ST_MULT      = 3.5      # supertrend multiplier — fixed at WF consensus
EMA_LEN      = 200

# Trail exit (sweep winner)
SL_MULT      = 3.0
TP1_R        = 0.75
TRAIL_MULT   = 0.50

# Entry filters
ATR_WIN      = 100
ATR_LO       = 10
ATR_HI       = 90
VOL_WIN      = 20
VOL_THR      = 1.05

# Decision tree
DT_MAX_DEPTH  = 3
DT_MIN_LEAF   = 6
DT_MIN_SIGS   = 20     # min IS trades to train DT; else use baseline
DT_THRESHOLD  = 0.50   # min P(win) to enter any trade in OOS; use 0 to always trade

# Lag-speed combos  (supertrend_atr_len, sma_filter_len)
COMBOS = [
    (7,  8),   # speed_0 — very fast
    (9,  10),  # speed_1 — fast
    (11, 13),  # speed_2 — BASELINE
    (14, 16),  # speed_3 — slow
    (21, 21),  # speed_4 — very slow
]
N_COMBOS    = len(COMBOS)
COMBO_NAMES = [f"spd{i}(atr{a},sma{s})" for i, (a, s) in enumerate(COMBOS)]
BASELINE_IDX = 2   # (11, 13) is the baseline

FEATURE_NAMES = [
    "adx14", "atr_pctile", "rsi14",
    "ema200_dist", "ema200_slope",
    "vol_ratio", "mom5", "body_ratio",
    "direction",
]

# Default coin list (Bybit linear symbols)
DEFAULT_COINS = [
    ("ETH",  "ETH/USDT:USDT"),
    ("BTC",  "BTC/USDT:USDT"),
    ("ARB",  "ARB/USDT:USDT"),
    ("GRT",  "GRT/USDT:USDT"),
    ("ADA",  "ADA/USDT:USDT"),
    ("NEAR", "NEAR/USDT:USDT"),
    ("XRP",  "XRP/USDT:USDT"),
    ("ETC",  "ETC/USDT:USDT"),
]

OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_OUT     = os.path.join(OUT_DIR, "adaptive_lag_results.csv")
RULES_OUT   = os.path.join(OUT_DIR, "adaptive_lag_dt_rules.txt")


# ─── Data fetch (Bybit) ─────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, since_date: str) -> pd.DataFrame:
    ex = ccxt.bybit({
        "enableRateLimit": True,
        "timeout": 30_000,
        "options": {"defaultType": "linear"},
    })
    since_ms = ex.parse8601(f"{since_date}T00:00:00Z")
    now_ms   = ex.milliseconds()
    limit_ms = 15 * 60 * 1000   # one 15m bar
    bars: list = []
    while True:
        chunk = None
        for attempt in range(5):
            try:
                chunk = ex.fetch_ohlcv(symbol, TIMEFRAME, since=since_ms, limit=1000)
                break
            except Exception as err:
                if attempt == 4:
                    raise RuntimeError(
                        f"fetch_ohlcv failed for {symbol} at since_ms={since_ms}: {err}"
                    ) from err
                # Exponential backoff keeps long history pulls resilient to transient API/network issues.
                time.sleep(1.5 * (attempt + 1))

        if not chunk:
            break
        bars.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts >= now_ms - limit_ms:
            break
        if last_ts < since_ms:
            raise RuntimeError(
                f"Non-monotonic OHLCV timestamps for {symbol}: last_ts={last_ts}, since_ms={since_ms}"
            )
        since_ms = last_ts + 1

    if not bars:
        raise RuntimeError(f"No data returned for {symbol}")
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ─── Core indicator helpers ───────────────────────────────────────────────────
def _rma(vals: np.ndarray, length: int) -> np.ndarray:
    """Wilder's smoothed MA — matches Pine ta.rma."""
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
        out[i] = (alpha * v + (1.0 - alpha) * out[i - 1]) if not np.isnan(v) else out[i - 1]
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    pc = np.empty_like(close)
    pc[0] = np.nan
    pc[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return _rma(tr, length)


def compute_ema(close: np.ndarray, length: int) -> np.ndarray:
    alpha = 2.0 / (length + 1)
    out = np.full(len(close), np.nan)
    for i, v in enumerate(close):
        if not np.isnan(v):
            out[i] = v
            for j in range(i + 1, len(close)):
                out[j] = alpha * close[j] + (1.0 - alpha) * out[j - 1]
            break
    return out


def compute_sma(arr: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(arr).rolling(length).mean().values


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
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


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX(period) — matches Pine Script ta.adx logic."""
    n = len(close)
    pc = np.empty(n)
    ph = np.empty(n)
    pl = np.empty(n)
    pc[0] = ph[0] = pl[0] = np.nan
    pc[1:] = close[:-1]
    ph[1:] = high[:-1]
    pl[1:]  = low[:-1]

    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))

    up   = high - ph
    down = pl - low
    plus_dm  = np.where((up > down) & (up > 0),  up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0]  = np.nan
    minus_dm[0] = np.nan

    tr_rma    = _rma(tr,        period)
    pdm_rma   = _rma(plus_dm,  period)
    mdm_rma   = _rma(minus_dm, period)

    with np.errstate(invalid="ignore", divide="ignore"):
        pdi  = 100.0 * (pdm_rma  / tr_rma)
        mdi  = 100.0 * (mdm_rma  / tr_rma)
        dsum = pdi + mdi
        dx   = np.where(dsum > 0, np.abs(pdi - mdi) / dsum * 100.0, 0.0)

    adx = _rma(dx, period)
    return adx


def compute_atr_pctile(atr14: np.ndarray, window: int = ATR_WIN) -> np.ndarray:
    n = len(atr14)
    out = np.full(n, 50.0)
    for i in range(window, n):
        w = atr14[i - window:i]
        if not np.isnan(atr14[i]) and not np.all(np.isnan(w)):
            valid = w[~np.isnan(w)]
            out[i] = float(np.sum(valid < atr14[i])) / len(valid) * 100.0
    return out


def compute_vol_ratio(volume: np.ndarray, sma_win: int = VOL_WIN) -> np.ndarray:
    vs = compute_sma(volume, sma_win)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = volume / vs
    return np.where(np.isnan(ratio) | np.isinf(ratio), 1.0, ratio)


# ─── Supertrend ──────────────────────────────────────────────────────────────
def compute_supertrend(
    open_: np.ndarray,
    close:  np.ndarray,
    atr_st: np.ndarray,
    mult:   float,
) -> np.ndarray:
    """Returns ST array. Source = open, direction checks = close (matches Pine)."""
    n = len(open_)
    ur = open_ + mult * atr_st
    lr = open_ - mult * atr_st
    upper = ur.copy()
    lower = lr.copy()
    direction = np.full(n, np.nan)
    st = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(atr_st[i - 1]):
            direction[i] = 2.0
            upper[i] = ur[i]
            lower[i] = lr[i]
        else:
            lower[i] = lr[i] if (lr[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]
            upper[i] = ur[i] if (ur[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
            ps = st[i - 1]
            if np.isnan(ps):
                ps = upper[i - 1] if not np.isnan(upper[i - 1]) else lower[i - 1]
            if ps == upper[i - 1]:
                direction[i] = -1.0 if close[i] > upper[i] else 1.0
            else:
                direction[i] =  1.0 if close[i] < lower[i] else -1.0
        st[i] = lower[i] if direction[i] == -1.0 else upper[i]

    return st


def build_signals(
    close:   np.ndarray,
    open_:   np.ndarray,
    high:    np.ndarray,
    low:     np.ndarray,
    ema200:  np.ndarray,
    atr_len: int,
    sma_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smart Signals for a given (atr_len, sma_len) combo.
    Returns (sbull, sbear) boolean arrays.
    """
    atr_st = compute_atr(high, low, close, atr_len)
    st     = compute_supertrend(open_, close, atr_st, ST_MULT)
    sma    = compute_sma(close, sma_len)

    n  = len(close)
    pc = np.empty(n);  pc[0]  = np.nan;  pc[1:]  = close[:-1]
    ps = np.empty(n);  ps[0]  = np.nan;  ps[1:]  = st[:-1]
    pe = np.empty(n);  pe[0]  = np.nan;  pe[1:]  = ema200[:-1]

    co = (~np.isnan(pc)) & (~np.isnan(ps)) & (~np.isnan(st)) & (pc < ps) & (close > st)
    cu = (~np.isnan(pc)) & (~np.isnan(ps)) & (~np.isnan(st)) & (pc > ps) & (close < st)
    above = (~np.isnan(pe)) & (~np.isnan(ema200)) & (pc > pe) & (close > ema200)

    sbull = co & (~np.isnan(sma)) & (close >= sma) &  above
    sbear = cu & (~np.isnan(sma)) & (close <= sma) & (~above)
    return sbull.astype(bool), sbear.astype(bool)


# ─── Walk-forward folds ──────────────────────────────────────────────────────
def generate_folds(index: pd.DatetimeIndex) -> list[dict]:
    folds = []
    fold_id    = 1
    fold_start = index[0]
    data_end   = index[-1]
    while True:
        train_end  = fold_start + pd.DateOffset(months=TRAIN_M)
        oos_start  = train_end
        oos_end    = min(oos_start + pd.DateOffset(months=OOS_M),
                         data_end  + pd.Timedelta(seconds=1))
        if oos_start > data_end:
            break
        tr_il  = np.where((index >= fold_start) & (index < train_end))[0]
        oos_il = np.where((index >= oos_start)  & (index <  oos_end))[0]
        if len(tr_il) > 50 and len(oos_il) > 0:
            folds.append(dict(
                fold_id=fold_id,
                train_start=fold_start,  train_end=train_end,
                oos_start=oos_start,     oos_end=oos_end,
                train_i0=int(tr_il[0]),  train_i1=int(tr_il[-1]) + 1,
                oos_i0=int(oos_il[0]),   oos_i1=int(oos_il[-1]) + 1,
            ))
        fold_start = fold_start + pd.DateOffset(months=STEP_M)
        fold_id   += 1
    return folds


# ─── Trail simulation ─────────────────────────────────────────────────────────
def sim_trail(
    close:    np.ndarray,
    high:     np.ndarray,
    low:      np.ndarray,
    sbull:    np.ndarray,
    sbear:    np.ndarray,
    atr14:    np.ndarray,
    atr_pctile: np.ndarray,
    vol_ratio:  np.ndarray,
    atr_lo: float = ATR_LO,
    atr_hi: float = ATR_HI,
    vol_thr: float = VOL_THR,
    sl_mult: float = SL_MULT,
    tp1_r:   float = TP1_R,
    trail_m: float = TRAIL_MULT,
    signal_mask: np.ndarray | None = None,
    max_i: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Trail exit sequential backtest.
    Returns (r_multiples, trades_list).
    Each trade dict includes entry_i for feature lookup.
    """
    n = max_i if max_i is not None else len(close)
    r_list: list[float] = []
    trades: list[dict] = []

    active = False
    is_long = False
    entry = sl_ = tp1 = risk = trail_sl_ = 0.0
    tp1_hit = False
    acc_r = 0.0
    entry_i = 0

    for i in range(1, n):
        h = high[i]; l = low[i]; c = close[i]; atr = atr14[i]

        if active:
            if is_long:
                if not tp1_hit and h >= tp1:
                    acc_r   += 0.5 * tp1_r
                    trail_sl_ = entry
                    tp1_hit   = True
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = h - trail_m * atr
                        if cand > trail_sl_:
                            trail_sl_ = cand
                    if l <= trail_sl_:
                        total_r = acc_r + 0.5 * max(0.0, (trail_sl_ - entry) / risk)
                        r_list.append(total_r)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "long", "r": total_r, "reason": "Trail"})
                        active = False
                        continue
                else:
                    if l <= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "long", "r": -1.0, "reason": "SL"})
                        active = False
            else:
                if not tp1_hit and l <= tp1:
                    acc_r   += 0.5 * tp1_r
                    trail_sl_ = entry
                    tp1_hit   = True
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = l + trail_m * atr
                        if cand < trail_sl_:
                            trail_sl_ = cand
                    if h >= trail_sl_:
                        total_r = acc_r + 0.5 * max(0.0, (entry - trail_sl_) / risk)
                        r_list.append(total_r)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "short", "r": total_r, "reason": "Trail"})
                        active = False
                        continue
                else:
                    if h >= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "short", "r": -1.0, "reason": "SL"})
                        active = False

        # Reversal exits
        if active and is_long and sbear[i]:
            rem     = 0.5 if tp1_hit else 1.0
            total_r = acc_r + rem * (c - entry) / risk
            r_list.append(total_r)
            trades.append({"entry_i": entry_i, "exit_i": i,
                           "direction": "long", "r": total_r, "reason": "Rev"})
            active = False
        if active and not is_long and sbull[i]:
            rem     = 0.5 if tp1_hit else 1.0
            total_r = acc_r + rem * (entry - c) / risk
            r_list.append(total_r)
            trades.append({"entry_i": entry_i, "exit_i": i,
                           "direction": "short", "r": total_r, "reason": "Rev"})
            active = False

        # New entry
        if not active and not math.isnan(atr):
            if not (atr_lo < atr_pctile[i] < atr_hi):
                continue
            if vol_ratio[i] < vol_thr:
                continue
            if sbull[i]:
                if signal_mask is not None and not signal_mask[i]:
                    continue
                sl_    = l - sl_mult * atr
                risk   = max(c - sl_, 1e-10)
                entry  = c
                is_long = True
                tp1    = c + tp1_r * risk
                tp1_hit = False
                trail_sl_ = sl_
                acc_r   = 0.0
                active  = True
                entry_i = i
            elif sbear[i]:
                if signal_mask is not None and not signal_mask[i]:
                    continue
                sl_    = h + sl_mult * atr
                risk   = max(sl_ - c, 1e-10)
                entry  = c
                is_long = False
                tp1    = c - tp1_r * risk
                tp1_hit = False
                trail_sl_ = sl_
                acc_r   = 0.0
                active  = True
                entry_i = i

    # Close any open trade at end of simulation window
    if active:
        cl  = close[n - 1]
        rem = 0.5 if tp1_hit else 1.0
        total_r = acc_r + rem * ((cl - entry) if is_long else (entry - cl)) / risk
        r_list.append(total_r)
        trades.append({"entry_i": entry_i, "exit_i": n - 1,
                       "direction": "long" if is_long else "short",
                       "r": total_r, "reason": "Open"})

    return np.array(r_list, dtype=np.float64), trades


# ─── Precompute all combo signals ────────────────────────────────────────────
def precompute_combo_signals(
    close: np.ndarray, open_: np.ndarray,
    high:  np.ndarray, low:   np.ndarray,
    ema200: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns list of (sbull, sbear) for each combo in COMBOS."""
    result = []
    for atr_len, sma_len in COMBOS:
        sb, ss = build_signals(close, open_, high, low, ema200, atr_len, sma_len)
        result.append((sb, ss))
    return result


# ─── Regime features ─────────────────────────────────────────────────────────
def extract_features(
    i:          int,
    close:      np.ndarray,
    high:       np.ndarray,
    low:        np.ndarray,
    open_:      np.ndarray,
    ema200:     np.ndarray,
    atr14:      np.ndarray,
    atr_pctile: np.ndarray,
    vol_ratio:  np.ndarray,
    rsi14:      np.ndarray,
    adx14:      np.ndarray,
    direction:  int,   # 1=long, 0=short
) -> np.ndarray | None:
    """
    Compute the 9-element feature vector at bar i (all causal).
    Returns None if any required indicator is still warming up.
    """
    if i < max(ATR_WIN + 14, EMA_LEN + 10):
        return None

    adx   = adx14[i]
    ap    = atr_pctile[i]
    rsi   = rsi14[i]
    em200 = ema200[i]
    e10   = ema200[max(0, i - 10)]
    vr    = vol_ratio[i]
    c     = close[i]
    o     = open_[i]
    h     = high[i]
    l_    = low[i]

    if any(np.isnan(x) for x in [adx, ap, rsi, em200, e10, vr, c]):
        return None

    ema200_dist  = (c - em200) / c
    ema200_slope = (em200 - e10) / e10 if e10 > 0 else 0.0
    mom5 = (c - close[max(0, i - 5)]) / close[max(0, i - 5)] if close[max(0, i - 5)] > 0 else 0.0
    rng  = h - l_
    body = abs(c - o)
    body_ratio = body / rng if rng > 0 else 0.0

    return np.array([
        adx / 100.0,       # adx14      (scaled 0-1)
        ap  / 100.0,       # atr_pctile (scaled 0-1)
        rsi / 100.0,       # rsi14      (scaled 0-1)
        ema200_dist,       # signed (approx -0.2 to +0.2)
        ema200_slope,      # slope
        vr,                # volume ratio
        mom5,              # 5-bar momentum
        body_ratio,        # candle conviction
        float(direction),  # 1=long / 0=short
    ], dtype=np.float64)


# ─── IS training data builder ────────────────────────────────────────────────
def build_is_training_data(
    i0: int,
    i1: int,
    close: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    open_: np.ndarray,
    ema200: np.ndarray,
    atr14:  np.ndarray,
    atr_pctile: np.ndarray,
    vol_ratio:  np.ndarray,
    rsi14: np.ndarray,
    adx14: np.ndarray,
    combo_sigs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """
    For each combo c, run IS sequential backtest on [i0:i1].
    For each trade, build regime features at entry and label (1=win, 0=loss).

    Returns: (feat_rows, labels, combo_idxs)
    """
    feat_rows: list[np.ndarray] = []
    labels:    list[int] = []
    combo_idxs: list[int] = []

    sl_clip = np.full(len(close), True)   # we run full array but only label IS trades
    sl_clip[:i0] = False
    sl_clip[i1:] = False

    for c_idx, (sb_full, ss_full) in enumerate(combo_sigs):
        # Mask signals to IS window (entries only allowed in [i0, i1))
        sb = sb_full.copy(); sb[:i0] = False; sb[i1:] = False
        ss = ss_full.copy(); ss[:i0] = False; ss[i1:] = False

        # Pass max_i=i1 so IS trades cannot exit using OOS price data (no look-ahead)
        _, trades = sim_trail(
            close, high, low, sb, ss, atr14, atr_pctile, vol_ratio, max_i=i1
        )

        for t in trades:
            ei = t["entry_i"]
            if not (i0 <= ei < i1):
                continue
            direc = 1 if t["direction"] == "long" else 0
            feat = extract_features(
                ei, close, high, low, open_, ema200,
                atr14, atr_pctile, vol_ratio, rsi14, adx14, direc
            )
            if feat is None:
                continue
            feat_rows.append(feat)
            labels.append(1 if t["r"] > 0 else 0)
            combo_idxs.append(c_idx)

    return feat_rows, labels, combo_idxs


# ─── OOS adaptive simulation ─────────────────────────────────────────────────
def sim_adaptive_oos(
    i0: int,
    i1: int,
    close: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    open_: np.ndarray,
    ema200: np.ndarray,
    atr14:  np.ndarray,
    atr_pctile: np.ndarray,
    vol_ratio:  np.ndarray,
    rsi14: np.ndarray,
    adx14: np.ndarray,
    combo_sigs: list[tuple[np.ndarray, np.ndarray]],
    dts:   list[DecisionTreeClassifier | None],  # one per combo
    threshold: float = DT_THRESHOLD,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """
    OOS simulation using DT-selected lag speed.

    At each bar i in [i0, i1):
      - For each combo c: if signal fires AND DT_c predicts P(win) >= threshold,
        record (c, P(win))
      - Pick combo with highest P(win); ties broken in favour of baseline
      - Enter trade with trail exit

    Returns (r_multiples, trades, combo_usage[N_COMBOS])
    combo_usage counts how many times each speed was chosen in OOS.
    """
    n = len(close)
    r_list: list[float] = []
    trades: list[dict]  = []
    combo_usage = np.zeros(N_COMBOS, dtype=int)

    active   = False
    is_long  = False
    entry    = sl_ = tp1 = risk = trail_sl_ = 0.0
    tp1_hit  = False
    acc_r    = 0.0
    entry_i  = 0
    sel_combo = BASELINE_IDX    # which combo is currently managing the open trade

    for i in range(max(1, i0), i1):
        h = high[i]; l = low[i]; c = close[i]; atr = atr14[i]; ap = atr_pctile[i]

        # ── Manage open trade ──
        if active:
            sb_rev, ss_rev = combo_sigs[sel_combo]

            if is_long:
                if not tp1_hit and h >= tp1:
                    acc_r    += 0.5 * TP1_R
                    trail_sl_ = entry
                    tp1_hit   = True
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = h - TRAIL_MULT * atr
                        if cand > trail_sl_:
                            trail_sl_ = cand
                    if l <= trail_sl_:
                        total_r = acc_r + 0.5 * max(0.0, (trail_sl_ - entry) / risk)
                        r_list.append(total_r)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "long", "r": total_r,
                                       "reason": "Trail", "combo": sel_combo})
                        active = False; continue
                else:
                    if l <= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "long", "r": -1.0,
                                       "reason": "SL", "combo": sel_combo})
                        active = False
            else:
                if not tp1_hit and l <= tp1:
                    acc_r    += 0.5 * TP1_R
                    trail_sl_ = entry
                    tp1_hit   = True
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = l + TRAIL_MULT * atr
                        if cand < trail_sl_:
                            trail_sl_ = cand
                    if h >= trail_sl_:
                        total_r = acc_r + 0.5 * max(0.0, (entry - trail_sl_) / risk)
                        r_list.append(total_r)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "short", "r": total_r,
                                       "reason": "Trail", "combo": sel_combo})
                        active = False; continue
                else:
                    if h >= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i,
                                       "direction": "short", "r": -1.0,
                                       "reason": "SL", "combo": sel_combo})
                        active = False

            # Reversal from the selected combo's signals
            if active and is_long and ss_rev[i]:
                rem     = 0.5 if tp1_hit else 1.0
                total_r = acc_r + rem * (c - entry) / risk
                r_list.append(total_r)
                trades.append({"entry_i": entry_i, "exit_i": i,
                               "direction": "long", "r": total_r,
                               "reason": "Rev", "combo": sel_combo})
                active = False
            if active and not is_long and sb_rev[i]:
                rem     = 0.5 if tp1_hit else 1.0
                total_r = acc_r + rem * (entry - c) / risk
                r_list.append(total_r)
                trades.append({"entry_i": entry_i, "exit_i": i,
                               "direction": "short", "r": total_r,
                               "reason": "Rev", "combo": sel_combo})
                active = False

        # ── Entry selection ──
        if not active and not math.isnan(atr):
            if not (ATR_LO < ap < ATR_HI):
                continue
            if vol_ratio[i] < VOL_THR:
                continue

            # Check each combo for a signal at bar i
            best_prob  = -1.0
            best_cidx  = -1
            best_direc = False   # True = long

            for c_idx, (sb, ss) in enumerate(combo_sigs):
                for is_bull, sig in [(True, sb[i]), (False, ss[i])]:
                    if not sig:
                        continue
                    direc = 1 if is_bull else 0
                    feat  = extract_features(
                        i, close, high, low, open_, ema200,
                        atr14, atr_pctile, vol_ratio, rsi14, adx14, direc
                    )
                    if feat is None:
                        continue

                    dt = dts[c_idx]
                    if dt is None:
                        # No DT trained — use default 0.5 for baseline, 0 for others
                        prob = 0.5 if c_idx == BASELINE_IDX else 0.0
                    else:
                        prob = float(dt.predict_proba(feat.reshape(1, -1))[0, 1])

                    if prob > best_prob:
                        best_prob  = prob
                        best_cidx  = c_idx
                        best_direc = is_bull

            if best_cidx < 0 or best_prob < threshold:
                continue

            # Enter with the winning combo
            sel_combo = best_cidx
            combo_usage[sel_combo] += 1

            if best_direc:
                sl_   = l - SL_MULT * atr
                risk  = max(c - sl_, 1e-10)
                entry = c; is_long = True
                tp1   = c + TP1_R * risk
            else:
                sl_   = h + SL_MULT * atr
                risk  = max(sl_ - c, 1e-10)
                entry = c; is_long = False
                tp1   = c - TP1_R * risk

            tp1_hit   = False
            trail_sl_ = sl_
            acc_r     = 0.0
            active    = True
            entry_i   = i

    # Close open trade at end of OOS (use last bar of OOS window, not end of dataset)
    if active:
        cl  = close[i1 - 1]
        rem = 0.5 if tp1_hit else 1.0
        total_r = acc_r + rem * ((cl - entry) if is_long else (entry - cl)) / risk
        r_list.append(total_r)
        trades.append({"entry_i": entry_i, "exit_i": i1 - 1,
                       "direction": "long" if is_long else "short",
                       "r": total_r, "reason": "Open", "combo": sel_combo})

    return np.array(r_list, dtype=np.float64), trades, combo_usage


# ─── Baseline OOS simulation (fixed combo_2 = 11,13) ────────────────────────
def sim_baseline_oos(
    i0: int, i1: int,
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    atr14: np.ndarray, atr_pctile: np.ndarray, vol_ratio: np.ndarray,
    combo_sigs: list[tuple[np.ndarray, np.ndarray]],
    combo_idx: int = BASELINE_IDX,
) -> tuple[np.ndarray, list[dict]]:
    sb_base, ss_base = combo_sigs[combo_idx]
    sb = sb_base.copy(); sb[:i0] = False; sb[i1:] = False
    ss = ss_base.copy(); ss[:i0] = False; ss[i1:] = False
    return sim_trail(close, high, low, sb, ss, atr14, atr_pctile, vol_ratio, max_i=i1)


# ─── IS combo performance (for best-IS fold-level selection) ─────────────────
def compute_is_combo_totals(
    i0: int, i1: int,
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    atr14: np.ndarray, atr_pctile: np.ndarray, vol_ratio: np.ndarray,
    combo_sigs: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Returns total_R for each combo over IS window [i0:i1] (anti-lookahead)."""
    totals = np.zeros(N_COMBOS, dtype=float)
    for c_idx, (sb, ss) in enumerate(combo_sigs):
        sb_ = sb.copy(); sb_[:i0] = False; sb_[i1:] = False
        ss_ = ss.copy(); ss_[:i0] = False; ss_[i1:] = False
        rs, _ = sim_trail(
            close, high, low, sb_, ss_, atr14, atr_pctile, vol_ratio, max_i=i1
        )
        totals[c_idx] = float(rs.sum()) if len(rs) > 0 else 0.0
    return totals


# ─── Consensus OOS simulation (≥K combos must agree direction) ───────────────
def sim_consensus_oos(
    i0: int, i1: int,
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    atr14: np.ndarray, atr_pctile: np.ndarray, vol_ratio: np.ndarray,
    combo_sigs: list[tuple[np.ndarray, np.ndarray]],
    min_agree: int = 3,
) -> tuple[np.ndarray, list[dict]]:
    """
    Only enter when min_agree or more combos fire in the same direction.
    Builds a consensus signal mask, then runs standard trail exit — no ML.
    """
    n = len(close)
    sb_con = np.zeros(n, dtype=bool)
    ss_con = np.zeros(n, dtype=bool)
    for i in range(i0, i1):
        bull = sum(1 for sb, _ss in combo_sigs if sb[i])
        bear = sum(1 for _sb, ss in combo_sigs if ss[i])
        if bull >= min_agree:
            sb_con[i] = True
        if bear >= min_agree:
            ss_con[i] = True
    return sim_trail(
        close, high, low, sb_con, ss_con, atr14, atr_pctile, vol_ratio, max_i=i1
    )


# ─── Summary stats ────────────────────────────────────────────────────────────
def stats(rs: np.ndarray, label: str = "") -> dict:
    if len(rs) == 0:
        return {"label": label, "n": 0, "total_r": 0.0, "win_rate": 0.0,
                "avg_r": 0.0, "sharpe": 0.0}
    n  = len(rs)
    tr = float(rs.sum())
    wr = float((rs > 0).mean())
    avg = float(rs.mean())
    sh  = float(rs.mean() / rs.std()) if rs.std() > 0 else 0.0
    return {"label": label, "n": n, "total_r": round(tr, 2),
            "win_rate": round(wr, 3), "avg_r": round(avg, 3),
            "sharpe": round(sh, 3)}


# ─── Main walk-forward per coin ─────────────────────────────────────────────
def run_coin(
    name:      str,
    symbol:    str,
    since:     str,
    max_depth: int,
    threshold: float,
    rules_fh,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  {name}  ({symbol})")
    print(f"{'='*60}")

    # Fetch data
    try:
        df = fetch_ohlcv(symbol, since)
    except Exception as e:
        print(f"  [ERROR] fetch failed: {e}")
        return []

    print(f"  {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    # Core arrays
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    open_  = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)

    # Global indicators (ETR = entry filter; always use ATR14 for exit)
    ema200     = compute_ema(close, EMA_LEN)
    atr14      = compute_atr(high, low, close, 14)
    atr_pctile = compute_atr_pctile(atr14, ATR_WIN)
    vol_ratio  = compute_vol_ratio(volume, VOL_WIN)
    rsi14      = compute_rsi(close, 14)
    adx14      = compute_adx(high, low, close, 14)

    # Pre-compute signals for all combos (full array)
    print("  Pre-computing signals for all combos …", flush=True)
    combo_sigs = precompute_combo_signals(close, open_, high, low, ema200)
    n_signals_per_combo = [int(sb.sum() + ss.sum()) for sb, ss in combo_sigs]
    print("  Signals per combo: " +
          "  ".join(f"{COMBO_NAMES[i]}: {n_signals_per_combo[i]}" for i in range(N_COMBOS)))

    # Walk-forward folds
    folds = generate_folds(df.index)
    print(f"  {len(folds)} walk-forward folds  (train {TRAIN_M}m / OOS {OOS_M}m / step {STEP_M}m)")

    rows: list[dict] = []

    for fold in folds:
        fid   = fold["fold_id"]
        t0, t1 = fold["train_i0"], fold["train_i1"]
        o0, o1 = fold["oos_i0"],   fold["oos_i1"]

        # ── IS: compute per-combo total_R (for best-IS selection) ──
        is_combo_totals = compute_is_combo_totals(
            t0, t1, close, high, low, atr14, atr_pctile, vol_ratio, combo_sigs
        )
        best_is_idx = int(np.argmax(is_combo_totals))
        # Tie-break: prefer BASELINE_IDX if within 0.01R
        if abs(is_combo_totals[best_is_idx] - is_combo_totals[BASELINE_IDX]) < 0.01:
            best_is_idx = BASELINE_IDX

        # ── IS: build training data per combo ──
        feat_rows, labels, cidxs = build_is_training_data(
            t0, t1, close, high, low, open_, ema200,
            atr14, atr_pctile, vol_ratio, rsi14, adx14, combo_sigs
        )

        # ── Train one DT per combo ──
        dts: list[DecisionTreeClassifier | None] = []

        for c_idx in range(N_COMBOS):
            mask = [k for k, ci in enumerate(cidxs) if ci == c_idx]
            if len(mask) < DT_MIN_SIGS:
                dts.append(None)
                continue
            X = np.array([feat_rows[k] for k in mask])
            y = np.array([labels[k]    for k in mask])
            if len(np.unique(y)) < 2:
                dts.append(None)
                continue
            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=DT_MIN_LEAF,
                class_weight="balanced",
                random_state=42,
            )
            dt.fit(X, y)
            dts.append(dt)

        trained = sum(1 for d in dts if d is not None)
        is_wins = sum(1 for l_ in labels if l_ == 1)
        is_totals_str = "  ".join(
            f"{COMBO_NAMES[c]}:{is_combo_totals[c]:>+.1f}R" for c in range(N_COMBOS)
        )
        print(
            f"\n  Fold {fid}  IS {fold['train_start'].date()}→{fold['train_end'].date()}"
            f"  OOS {fold['oos_start'].date()}→{fold['oos_end'].date()}"
            f"  |  IS trades={len(labels)}  wins={is_wins}  DTs trained={trained}/{N_COMBOS}"
        )
        print(f"    IS combo totals:  {is_totals_str}  → best={COMBO_NAMES[best_is_idx]}")

        # Write DT rules for this fold
        rules_fh.write(f"\n{'─'*70}\n")
        rules_fh.write(f"Coin: {name}  Fold {fid}  OOS {fold['oos_start'].date()}→{fold['oos_end'].date()}\n")
        for c_idx, dt in enumerate(dts):
            if dt is None:
                rules_fh.write(f"  {COMBO_NAMES[c_idx]}: [no DT — too few IS trades]\n")
                continue
            text = export_text(dt, feature_names=FEATURE_NAMES, show_weights=True, decimals=3)
            rules_fh.write(f"\n  {COMBO_NAMES[c_idx]}  (IS win-rate: "
                           f"{sum(labels[k]==1 for k,ci in enumerate(cidxs) if ci==c_idx)}"
                           f"/{sum(1 for ci in cidxs if ci==c_idx)} trades)\n")
            for line in text.split("\n"):
                rules_fh.write("    " + line + "\n")

        # ── OOS: baseline ──
        r_base, _ = sim_baseline_oos(
            o0, o1, close, high, low, atr14, atr_pctile, vol_ratio, combo_sigs
        )
        s_base = stats(r_base, "baseline")

        # ── OOS: best-IS combo (fold-level selection, no ML) ──
        r_best_is, _ = sim_baseline_oos(
            o0, o1, close, high, low, atr14, atr_pctile, vol_ratio,
            combo_sigs, combo_idx=best_is_idx
        )
        s_best_is = stats(r_best_is, "best_is")

        # ── OOS: consensus ≥3/5 combos agree ──
        r_consensus, _ = sim_consensus_oos(
            o0, o1, close, high, low, atr14, atr_pctile, vol_ratio,
            combo_sigs, min_agree=3
        )
        s_consensus = stats(r_consensus, "consensus")

        # ── OOS: DT adaptive ──
        r_adapt, _, combo_usage = sim_adaptive_oos(
            o0, o1, close, high, low, open_, ema200,
            atr14, atr_pctile, vol_ratio, rsi14, adx14,
            combo_sigs, dts, threshold=threshold
        )
        s_adapt = stats(r_adapt, "adaptive")

        usage_str = "  ".join(
            f"{COMBO_NAMES[c]}:{combo_usage[c]}" for c in range(N_COMBOS)
        )
        print(f"    OOS combo usage (DT): {usage_str}")
        print(
            f"    BASELINE   n={s_base['n']:>3}  total_R={s_base['total_r']:>+7.2f}"
            f"  win={s_base['win_rate']:.0%}  sharpe={s_base['sharpe']:>+6.3f}"
        )
        print(
            f"    BEST_IS    n={s_best_is['n']:>3}  total_R={s_best_is['total_r']:>+7.2f}"
            f"  win={s_best_is['win_rate']:.0%}  sharpe={s_best_is['sharpe']:>+6.3f}"
            f"   Δ={s_best_is['total_r']-s_base['total_r']:>+.2f}R"
        )
        print(
            f"    CONSENSUS  n={s_consensus['n']:>3}  total_R={s_consensus['total_r']:>+7.2f}"
            f"  win={s_consensus['win_rate']:.0%}  sharpe={s_consensus['sharpe']:>+6.3f}"
            f"   Δ={s_consensus['total_r']-s_base['total_r']:>+.2f}R"
        )
        print(
            f"    DT_ADAPT   n={s_adapt['n']:>3}  total_R={s_adapt['total_r']:>+7.2f}"
            f"  win={s_adapt['win_rate']:.0%}  sharpe={s_adapt['sharpe']:>+6.3f}"
            f"   Δ={s_adapt['total_r']-s_base['total_r']:>+.2f}R"
        )

        rows.append({
            "coin": name, "fold": fid,
            "oos_start": str(fold["oos_start"].date()),
            "oos_end":   str(fold["oos_end"].date()),
            **{f"base_{k}": v for k, v in s_base.items() if k != "label"},
            **{f"bis_{k}":  v for k, v in s_best_is.items() if k != "label"},
            "bis_combo": best_is_idx,
            **{f"con_{k}":  v for k, v in s_consensus.items() if k != "label"},
            **{f"adpt_{k}": v for k, v in s_adapt.items() if k != "label"},
            "delta_bis_r": round(s_best_is["total_r"] - s_base["total_r"], 2),
            "delta_con_r": round(s_consensus["total_r"] - s_base["total_r"], 2),
            "delta_r":     round(s_adapt["total_r"] - s_base["total_r"], 2),
            **{f"usage_{COMBO_NAMES[c]}": int(combo_usage[c]) for c in range(N_COMBOS)},
        })

    # Per-coin summary
    if rows:
        dfr = pd.DataFrame(rows)
        br_total  = dfr["base_total_r"].sum()
        bis_total = dfr["bis_total_r"].sum()
        con_total = dfr["con_total_r"].sum()
        ar_total  = dfr["adpt_total_r"].sum()
        n_pos_base = (dfr["base_total_r"] > 0).sum()
        n_pos_bis  = (dfr["bis_total_r"]  > 0).sum()
        n_pos_con  = (dfr["con_total_r"]  > 0).sum()
        n_pos_adpt = (dfr["adpt_total_r"] > 0).sum()
        n_folds    = len(rows)
        print(f"\n  ── {name} SUMMARY  ({n_folds} folds) ──")
        print(f"     BASELINE:  total={br_total:>+7.2f}R  pos_folds={n_pos_base}/{n_folds}")
        print(f"     BEST_IS:   total={bis_total:>+7.2f}R  pos_folds={n_pos_bis}/{n_folds}  Δ={bis_total-br_total:>+.2f}R")
        print(f"     CONSENSUS: total={con_total:>+7.2f}R  pos_folds={n_pos_con}/{n_folds}  Δ={con_total-br_total:>+.2f}R")
        print(f"     DT_ADAPT:  total={ar_total:>+7.2f}R  pos_folds={n_pos_adpt}/{n_folds}  Δ={ar_total-br_total:>+.2f}R")

    return rows


# ─── Entry point ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Adaptive Lag DT per coin")
    ap.add_argument("--coins",     nargs="*",  default=None,
                    help="Coin tickers to run (e.g. ETH BTC ARB). Default: all.")
    ap.add_argument("--since",     default=SINCE_DATE, help="Start date YYYY-MM-DD")
    ap.add_argument("--max-depth", type=int, default=DT_MAX_DEPTH,
                    help="DecisionTree max_depth (default 3)")
    ap.add_argument("--threshold", type=float, default=DT_THRESHOLD,
                    help="Min P(win) required to enter trade (default 0.50)")
    args = ap.parse_args()

    # Resolve coin list
    if args.coins:
        ticker_to_sym = {name: sym for name, sym in DEFAULT_COINS}
        coin_list = []
        for t in args.coins:
            t_up = t.upper()
            if t_up in ticker_to_sym:
                coin_list.append((t_up, ticker_to_sym[t_up]))
            else:
                # Try as raw symbol
                coin_list.append((t_up, f"{t_up}/USDT:USDT"))
    else:
        coin_list = DEFAULT_COINS

    print(f"\nAdaptive Lag — Decision Tree Signal Speed Selection")
    print(f"Combos: {N_COMBOS}  ({', '.join(COMBO_NAMES)})")
    print(f"Features: {FEATURE_NAMES}")
    print(f"DT: max_depth={args.max_depth}  min_leaf={DT_MIN_LEAF}  threshold={args.threshold}")
    print(f"Exit: trail(sl_mult={SL_MULT}, tp1_r={TP1_R}, trail={TRAIL_MULT})")
    print(f"Coins: {[n for n,_ in coin_list]}")
    print(f"Since: {args.since}")

    all_rows: list[dict] = []

    rules_path = RULES_OUT
    with open(rules_path, "w", encoding="utf-8") as rules_fh:
        rules_fh.write("Adaptive Lag — Decision Tree Rules per Coin per Fold\n")
        rules_fh.write(f"Generated: {pd.Timestamp.now()}\n")
        rules_fh.write(f"Combos: {COMBO_NAMES}\n")
        rules_fh.write(f"Features: {FEATURE_NAMES}\n\n")

        for name, symbol in coin_list:
            rows = run_coin(
                name, symbol, args.since,
                args.max_depth, args.threshold, rules_fh
            )
            all_rows.extend(rows)

    # ── Overall summary ──
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(CSV_OUT, index=False)
        print(f"\n{'='*60}")
        print("  OVERALL SUMMARY ACROSS ALL COINS")
        print(f"{'='*60}")
        for coin in df_all["coin"].unique():
            sub = df_all[df_all["coin"] == coin]
            n_folds = len(sub)
            br  = sub["base_total_r"].sum()
            bisr = sub["bis_total_r"].sum()
            conr = sub["con_total_r"].sum()
            ar  = sub["adpt_total_r"].sum()
            nb  = (sub["base_total_r"] > 0).sum()
            nbis = (sub["bis_total_r"]  > 0).sum()
            ncon = (sub["con_total_r"]  > 0).sum()
            na  = (sub["adpt_total_r"] > 0).sum()
            print(
                f"  {coin:<6}  base={br:>+7.2f}R({nb}/{n_folds})"
                f"  best_IS={bisr:>+7.2f}R({nbis}/{n_folds}) Δ={bisr-br:>+5.2f}R"
                f"  consensus={conr:>+7.2f}R({ncon}/{n_folds}) Δ={conr-br:>+5.2f}R"
                f"  DT={ar:>+7.2f}R({na}/{n_folds}) Δ={ar-br:>+5.2f}R"
            )

        tot_b   = df_all["base_total_r"].sum()
        tot_bis = df_all["bis_total_r"].sum()
        tot_con = df_all["con_total_r"].sum()
        tot_a   = df_all["adpt_total_r"].sum()
        n_total = len(df_all)
        np_b   = (df_all["base_total_r"] > 0).sum()
        np_bis = (df_all["bis_total_r"]  > 0).sum()
        np_con = (df_all["con_total_r"]  > 0).sum()
        np_a   = (df_all["adpt_total_r"] > 0).sum()
        print(
            f"\n  TOTAL   base={tot_b:>+8.2f}R({np_b}/{n_total})"
            f"  best_IS={tot_bis:>+8.2f}R({np_bis}/{n_total}) Δ={tot_bis-tot_b:>+.2f}R"
            f"  consensus={tot_con:>+8.2f}R({np_con}/{n_total}) Δ={tot_con-tot_b:>+.2f}R"
            f"  DT={tot_a:>+8.2f}R({np_a}/{n_total}) Δ={tot_a-tot_b:>+.2f}R"
        )
        print(f"\n  Results CSV : {CSV_OUT}")
        print(f"  DT rules TXT: {rules_path}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
