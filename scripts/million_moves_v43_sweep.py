"""
Million Moves V4.3 — Exit Parameter Sweep
==========================================

Sweeps ATR-SL multiplier × TP params across 6 walk-forward folds.
All variants use ATR-based SL anchored to the signal-bar's low/high.

Exit modes tested
-----------------
  fixed   : Close 50% at TP1 → SL to break-even → Close 50% at TP2 (hard R)
  trail   : Close 50% at TP1 → SL to break-even → Trail remainder by ATR
  single  : 100% position, single TP at tp_r, no partial, no BE (clean baseline)
  3tier   : Original 33%/50%/100% scale-out with no BE (mimics original algo)

Grids
-----
  sl_mult  : 0.5 0.75 1.0 1.5 2.0 2.5 3.0 3.5   (ATR SL distance)
  tp1_r    : 0.5 0.75 1.0 1.5                    (fixed / trail: first partial exit)
  tp2_r    : 1.0 1.5 2.0 2.5 3.0 4.0             (fixed: second exit)
  trail_m  : 0.5 1.0 1.5 2.0 3.0                 (trail: ATR multiplier for trail)
  tp_r     : 1.0 1.5 2.0 2.5 3.0 4.0             (single mode: full-position TP)
  tp_mult  : 1.0 1.5 2.0 2.5 3.0                 (3tier: sl/tp scale factor)

Filters (ATR pctile 10-90, volume > SMA20 × 1.05) are ON by default.
Use --no-filter to disable, --mode to restrict to one mode.

Usage
-----
  python scripts/million_moves_v43_sweep.py
  python scripts/million_moves_v43_sweep.py --no-filter
  python scripts/million_moves_v43_sweep.py --mode fixed
  python scripts/million_moves_v43_sweep.py --mode trail
  python scripts/million_moves_v43_sweep.py --top 40
"""

from __future__ import annotations

import argparse, math, os, time
import numpy as np
import pandas as pd
import ccxt
from numpy.lib.stride_tricks import sliding_window_view

# ── Config ───────────────────────────────────────────────────────────────────
SYMBOL     = "ETH/USDT"
TIMEFRAME  = "15m"
SINCE_DATE = "2024-01-01"
ST_MULT    = 3.5
ST_ATR_LEN = 11
EMA_LEN    = 200
SMA_LEN    = 13
ATR14_LEN  = 14
TRAIN_M, OOS_M, STEP_M = 12, 3, 3
ATR_WIN    = 100
VOL_WIN    = 20
DEF_LO, DEF_HI, DEF_VOL = 10, 90, 1.05
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── Sweep Grids ───────────────────────────────────────────────────────────────
SL_MULTS    = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
TP1_RS      = [0.5, 0.75, 1.0, 1.5]
TP2_RS      = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
TRAIL_MULTS = [0.5, 1.0, 1.5, 2.0, 3.0]
SINGLE_TPS  = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
TIER3_TMULTS = [1.0, 1.5, 2.0, 2.5, 3.0]


# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol, timeframe, since_date):
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since_date}T00:00:00Z")
    bars = []
    print(f"Fetching {symbol} {timeframe} from {since_date} …", flush=True)
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not chunk: break
        bars.extend(chunk)
        if len(chunk) < 1000: break
        since_ms = chunk[-1][0] + 1
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    print(f"  -> {len(df):,} bars  ({df.index[0]} ... {df.index[-1]})", flush=True)
    return df


# ── Indicators ────────────────────────────────────────────────────────────────
def _rma(vals, length):
    alpha = 1.0 / length
    out = np.full(len(vals), np.nan)
    s = 0
    while s < len(vals) and np.isnan(vals[s]):
        s += 1
    se = s + length
    if se > len(vals): return out
    out[se - 1] = float(np.nanmean(vals[s:se]))
    for i in range(se, len(vals)):
        v = vals[i]
        out[i] = alpha * v + (1 - alpha) * out[i-1] if not np.isnan(v) else out[i-1]
    return out

def compute_atr(high, low, close, length):
    pc = np.empty_like(close); pc[0] = np.nan; pc[1:] = close[:-1]
    tr = np.maximum(high-low, np.maximum(np.abs(high-pc), np.abs(low-pc)))
    return _rma(tr, length)

def compute_ema(close, length):
    alpha = 2.0 / (length + 1)
    out = np.full(len(close), np.nan)
    for i, v in enumerate(close):
        if not np.isnan(v):
            out[i] = v
            for j in range(i+1, len(close)):
                out[j] = alpha * close[j] + (1-alpha) * out[j-1]
            break
    return out

def compute_sma(arr, length):
    return pd.Series(arr).rolling(length).mean().values

def compute_supertrend(open_, close, atr_st, mult):
    n = len(open_)
    ur = open_ + mult * atr_st; lr = open_ - mult * atr_st
    upper, lower = ur.copy(), lr.copy()
    direction = np.full(n, np.nan); st = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(atr_st[i-1]):
            direction[i] = 2.0; upper[i] = ur[i]; lower[i] = lr[i]
        else:
            lower[i] = lr[i] if lr[i] > lower[i-1] or close[i-1] < lower[i-1] else lower[i-1]
            upper[i] = ur[i] if ur[i] < upper[i-1] or close[i-1] > upper[i-1] else upper[i-1]
            ps = st[i-1]
            if np.isnan(ps): ps = upper[i-1] if not np.isnan(upper[i-1]) else lower[i-1]
            direction[i] = (-1.0 if close[i] > upper[i] else 1.0) if ps == upper[i-1] else \
                           (1.0 if close[i] < lower[i] else -1.0)
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

def compute_atr_pctile_fast(atr14, window=ATR_WIN):
    """Vectorized rolling percentile via sliding_window_view — O(n*w) but fast numpy."""
    n = len(atr14)
    out = np.full(n, 50.0)
    if n < window: return out
    wins = sliding_window_view(atr14, window)   # (n-w+1, w)
    cur  = wins[:, -1]                           # current bar value
    hist = wins[:, :-1]                          # preceding w-1 bars
    ranks = (hist < cur[:, None]).sum(axis=1) / (window - 1) * 100.0
    out[window-1:] = ranks
    return out

def compute_vol_ratio(volume, win=VOL_WIN):
    vs = compute_sma(volume, win)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = volume / vs
    return np.where(np.isnan(r) | np.isinf(r), 1.0, r)


# ── WF folds ──────────────────────────────────────────────────────────────────
def generate_wf_folds(index, train_months=TRAIN_M, oos_months=OOS_M, step_months=STEP_M):
    folds = []; fold_id = 1; fold_start = index[0]; data_end = index[-1]
    while True:
        train_end = fold_start + pd.DateOffset(months=train_months)
        oos_start = train_end
        oos_end   = min(oos_start + pd.DateOffset(months=oos_months),
                        data_end + pd.Timedelta(seconds=1))
        if oos_start > data_end: break
        tr_il  = np.where((index >= fold_start) & (index < train_end))[0]
        oos_il = np.where((index >= oos_start)  & (index < oos_end))[0]
        if len(tr_il) > 50 and len(oos_il) > 0:
            folds.append(dict(fold_id=fold_id,
                              train_i0=int(tr_il[0]),  train_i1=int(tr_il[-1])+1,
                              oos_i0=int(oos_il[0]),   oos_i1=int(oos_il[-1])+1))
        fold_start += pd.DateOffset(months=step_months)
        fold_id += 1
    return folds


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(r_arr, min_n=5):
    n = len(r_arr)
    if n < min_n:
        return dict(n=n, sharpe=-99.0, total_r=0.0, avg_r=0.0,
                    win_rate=0.0, pf=0.0, pos_folds=0)
    std = float(np.std(r_arr, ddof=1))
    mean = float(np.mean(r_arr))
    wins = r_arr[r_arr > 0]; losses = r_arr[r_arr < 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    return dict(
        n=n,
        sharpe=round(mean / std if std > 1e-12 else 0.0, 4),
        total_r=round(float(r_arr.sum()), 2),
        avg_r=round(mean, 4),
        win_rate=round(len(wins)/n, 4),
        pf=round(gw/gl if gl > 0 else 0.0, 3),
    )


# ── Simulation: 2-tier fixed ──────────────────────────────────────────────────
def _sim_fixed(close, high, low, sbull_raw, sbear_raw, entry_sbull, entry_sbear,
               atr14, sl_mult, tp1_r, tp2_r):
    """
    ATR SL + 2-tier TP.
      - 50% closed at TP1 → SL moved to break-even
      - 50% closed at TP2 (hard R target)
    Reversal exits use raw signals; entries use filtered arrays.
    """
    n = len(close); r_list = []
    active = False; is_long = False
    entry = sl_ = tp1 = tp2 = 0.0; risk = 1.0
    tp1_hit = False; acc_r = 0.0

    for i in range(1, n):
        h, l, c, atr = high[i], low[i], close[i], atr14[i]

        if active:
            rem = 0.5 if tp1_hit else 1.0
            if is_long:
                if not tp1_hit and h >= tp1:
                    acc_r += 0.5 * tp1_r; sl_ = entry; tp1_hit = True; rem = 0.5
                if h >= tp2:
                    r_list.append(acc_r + rem * tp2_r); active = False; continue
                if l <= sl_:
                    r_list.append(acc_r + rem * (sl_ - entry) / risk); active = False
            else:
                if not tp1_hit and l <= tp1:
                    acc_r += 0.5 * tp1_r; sl_ = entry; tp1_hit = True; rem = 0.5
                if l <= tp2:
                    r_list.append(acc_r + rem * tp2_r); active = False; continue
                if h >= sl_:
                    r_list.append(acc_r + rem * (entry - sl_) / risk); active = False

        if active:  # reversal (must re-check active after possible close above)
            rem = 0.5 if tp1_hit else 1.0
            if is_long and sbear_raw[i]:
                r_list.append(acc_r + rem * (c - entry) / risk); active = False
            elif not is_long and sbull_raw[i]:
                r_list.append(acc_r + rem * (entry - c) / risk); active = False

        if not active and not math.isnan(atr):
            if entry_sbull[i]:
                sl_ = l - atr * sl_mult; risk = max(c - sl_, 1e-10)
                entry = c; is_long = True
                tp1 = c + tp1_r * risk; tp2 = c + tp2_r * risk
                tp1_hit = False; acc_r = 0.0; active = True
            elif entry_sbear[i]:
                sl_ = h + atr * sl_mult; risk = max(sl_ - c, 1e-10)
                entry = c; is_long = False
                tp1 = c - tp1_r * risk; tp2 = c - tp2_r * risk
                tp1_hit = False; acc_r = 0.0; active = True

    if active:
        cl = close[-1]; rem = 0.5 if tp1_hit else 1.0
        r_list.append(acc_r + rem * ((cl - entry) if is_long else (entry - cl)) / risk)
    return np.array(r_list, dtype=np.float64)


# ── Simulation: trail after TP1 ───────────────────────────────────────────────
def _sim_trail(close, high, low, sbull_raw, sbear_raw, entry_sbull, entry_sbear,
               atr14, sl_mult, tp1_r, trail_mult):
    """
    ATR SL + trailing remainder.
      - 50% closed at TP1 → trail SL starts at break-even
      - Remainder exits when trailing SL is hit (or reversal)
    Trail: long → trail_sl = max(BE, high[i] - trail_mult * ATR)
           short → trail_sl = min(BE, low[i]  + trail_mult * ATR)
    """
    n = len(close); r_list = []
    active = False; is_long = False
    entry = sl_ = tp1 = 0.0; risk = 1.0
    tp1_hit = False; trail_sl = 0.0; acc_r = 0.0

    for i in range(1, n):
        h, l, c, atr = high[i], low[i], close[i], atr14[i]

        if active:
            if is_long:
                # TP1 check
                if not tp1_hit and h >= tp1:
                    acc_r += 0.5 * tp1_r
                    trail_sl = entry          # start trail at BE
                    tp1_hit = True
                    # Immediately ratchet trail based on this bar's high
                    cand = h - trail_mult * atr
                    if cand > trail_sl: trail_sl = cand

                if tp1_hit:
                    # Update trail
                    cand = h - trail_mult * atr
                    if cand > trail_sl: trail_sl = cand
                    # Trail exit
                    if l <= trail_sl:
                        r_trail = max(0.0, (trail_sl - entry) / risk)
                        r_list.append(acc_r + 0.5 * r_trail); active = False; continue
                else:
                    # Initial SL
                    if l <= sl_:
                        r_list.append(-1.0); active = False
            else:
                if not tp1_hit and l <= tp1:
                    acc_r += 0.5 * tp1_r
                    trail_sl = entry
                    tp1_hit = True
                    cand = l + trail_mult * atr
                    if cand < trail_sl: trail_sl = cand

                if tp1_hit:
                    cand = l + trail_mult * atr
                    if cand < trail_sl: trail_sl = cand
                    if h >= trail_sl:
                        r_trail = max(0.0, (entry - trail_sl) / risk)
                        r_list.append(acc_r + 0.5 * r_trail); active = False; continue
                else:
                    if h >= sl_:
                        r_list.append(-1.0); active = False

        if active:
            rem = 0.5 if tp1_hit else 1.0
            if is_long and sbear_raw[i]:
                r_list.append(acc_r + rem * (c - entry) / risk); active = False
            elif not is_long and sbull_raw[i]:
                r_list.append(acc_r + rem * (entry - c) / risk); active = False

        if not active and not math.isnan(atr):
            if entry_sbull[i]:
                sl_ = l - atr * sl_mult; risk = max(c - sl_, 1e-10)
                entry = c; is_long = True; tp1 = c + tp1_r * risk
                tp1_hit = False; trail_sl = entry; acc_r = 0.0; active = True
            elif entry_sbear[i]:
                sl_ = h + atr * sl_mult; risk = max(sl_ - c, 1e-10)
                entry = c; is_long = False; tp1 = c - tp1_r * risk
                tp1_hit = False; trail_sl = entry; acc_r = 0.0; active = True

    if active:
        cl = close[-1]; rem = 0.5 if tp1_hit else 1.0
        r_list.append(acc_r + rem * ((cl - entry) if is_long else (entry - cl)) / risk)
    return np.array(r_list, dtype=np.float64)


# ── Simulation: single TP (no partial) ───────────────────────────────────────
def _sim_single(close, high, low, sbull_raw, sbear_raw, entry_sbull, entry_sbear,
                atr14, sl_mult, tp_r):
    """Full position, single TP, no break-even management."""
    n = len(close); r_list = []
    active = False; is_long = False
    entry = sl_ = tp = 0.0; risk = 1.0

    for i in range(1, n):
        h, l, c, atr = high[i], low[i], close[i], atr14[i]

        if active:
            if is_long:
                if h >= tp:  r_list.append(tp_r);             active = False; continue
                if l <= sl_: r_list.append(-1.0);             active = False
            else:
                if l <= tp:  r_list.append(tp_r);             active = False; continue
                if h >= sl_: r_list.append(-1.0);             active = False

        if active:
            if is_long and sbear_raw[i]:
                r_list.append((c - entry) / risk); active = False
            elif not is_long and sbull_raw[i]:
                r_list.append((entry - c) / risk); active = False

        if not active and not math.isnan(atr):
            if entry_sbull[i]:
                sl_ = l - atr * sl_mult; risk = max(c - sl_, 1e-10)
                entry = c; is_long = True; tp = c + tp_r * risk; active = True
            elif entry_sbear[i]:
                sl_ = h + atr * sl_mult; risk = max(sl_ - c, 1e-10)
                entry = c; is_long = False; tp = c - tp_r * risk; active = True

    if active:
        cl = close[-1]
        r_list.append(((cl - entry) if is_long else (entry - cl)) / risk)
    return np.array(r_list, dtype=np.float64)


# ── Simulation: original 3-tier (baseline reference) ─────────────────────────
def _sim_3tier(close, high, low, sbull_raw, sbear_raw, entry_sbull, entry_sbear,
               atr14, sl_mult, tp_mult):
    """
    Original 3-tier scale-out:
      33% at 1×tp_mult×R, 50% of remaining at 2×, 100% at 3×.
    No break-even management (as per original algo design).
    """
    n = len(close); r_list = []
    active = False; is_long = False
    entry = sl_ = 0.0; risk = 1.0
    tp1 = tp2 = tp3 = 0.0; remain = 1.0; tp1h = tp2h = False; acc = 0.0

    for i in range(1, n):
        h, l, c, atr = high[i], low[i], close[i], atr14[i]

        if active:
            if is_long:
                hit_sl = l <= sl_
                if hit_sl and not tp1h:
                    r_list.append(-1.0); active = False
                else:
                    if not tp1h and h >= tp1:
                        acc += remain * 0.33 * tp_mult; remain *= 0.67; tp1h = True
                    if active and tp1h and not tp2h and h >= tp2:
                        f = remain * 0.5
                        acc += f * 2 * tp_mult; remain -= f; tp2h = True
                    if active and tp2h and h >= tp3:
                        r_list.append(acc + remain * 3 * tp_mult); active = False
                    elif active and hit_sl:
                        lost = remain * (sl_ - entry) / risk
                        r_list.append(acc + lost); active = False
            else:
                hit_sl = h >= sl_
                if hit_sl and not tp1h:
                    r_list.append(-1.0); active = False
                else:
                    if not tp1h and l <= tp1:
                        acc += remain * 0.33 * tp_mult; remain *= 0.67; tp1h = True
                    if active and tp1h and not tp2h and l <= tp2:
                        f = remain * 0.5
                        acc += f * 2 * tp_mult; remain -= f; tp2h = True
                    if active and tp2h and l <= tp3:
                        r_list.append(acc + remain * 3 * tp_mult); active = False
                    elif active and hit_sl:
                        lost = remain * (entry - sl_) / risk
                        r_list.append(acc + lost); active = False

        if active:
            if is_long and sbear_raw[i]:
                lost = remain * (c - entry) / risk
                r_list.append(acc + lost); active = False
            elif not is_long and sbull_raw[i]:
                lost = remain * (entry - c) / risk
                r_list.append(acc + lost); active = False

        if not active and not math.isnan(atr):
            if entry_sbull[i]:
                sl_ = l - atr * sl_mult; risk = max(c - sl_, 1e-10); entry = c; is_long = True
                tp1 = c + 1*tp_mult*risk; tp2 = c + 2*tp_mult*risk; tp3 = c + 3*tp_mult*risk
                remain = 1.0; tp1h = tp2h = False; acc = 0.0; active = True
            elif entry_sbear[i]:
                sl_ = h + atr * sl_mult; risk = max(sl_ - c, 1e-10); entry = c; is_long = False
                tp1 = c - 1*tp_mult*risk; tp2 = c - 2*tp_mult*risk; tp3 = c - 3*tp_mult*risk
                remain = 1.0; tp1h = tp2h = False; acc = 0.0; active = True

    if active:
        cl = close[-1]
        lost = remain * ((cl - entry) if is_long else (entry - cl)) / risk
        r_list.append(acc + lost)
    return np.array(r_list, dtype=np.float64)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since",      default=SINCE_DATE)
    parser.add_argument("--symbol",     default=SYMBOL)
    parser.add_argument("--tf",         default=TIMEFRAME)
    parser.add_argument("--no-filter",  action="store_true")
    parser.add_argument("--mode",       default="all",
                        choices=["all", "fixed", "trail", "single", "3tier"])
    parser.add_argument("--top",        type=int, default=30)
    args = parser.parse_args()

    t0 = time.time()

    # ── Data ──────────────────────────────────────────────────────────────────
    df = fetch_ohlcv(args.symbol, args.tf, args.since)
    close  = df["close"].values.astype(np.float64)
    open_  = df["open"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # ── Indicators ────────────────────────────────────────────────────────────
    print("Computing indicators…", flush=True)
    atr_st  = compute_atr(high, low, close, ST_ATR_LEN)
    atr14   = compute_atr(high, low, close, ATR14_LEN)
    ema200  = compute_ema(close, EMA_LEN)
    sma13   = compute_sma(close, SMA_LEN)
    atr_pct = compute_atr_pctile_fast(atr14, ATR_WIN)
    vol_rat = compute_vol_ratio(volume, VOL_WIN)

    sbull_raw, sbear_raw = build_raw_signals(close, open_, sma13, ema200, atr_st)
    print(f"  Raw signals  Sbull={sbull_raw.sum()}  Sbear={sbear_raw.sum()}", flush=True)

    if args.no_filter:
        filter_mask = np.ones(len(close), dtype=bool)
        print("  Filters: OFF", flush=True)
    else:
        filter_mask = (atr_pct > DEF_LO) & (atr_pct < DEF_HI) & (vol_rat >= DEF_VOL)
        print(f"  Filters: ON  (ATR pctile {DEF_LO}-{DEF_HI}, vol>{DEF_VOL})", flush=True)

    entry_sbull = (sbull_raw & filter_mask).astype(bool)
    entry_sbear = (sbear_raw & filter_mask).astype(bool)
    print(f"  Filtered entry signals  long={entry_sbull.sum()}  short={entry_sbear.sum()}", flush=True)

    # ── Folds ─────────────────────────────────────────────────────────────────
    folds = generate_wf_folds(df.index)
    print(f"  Walk-forward: {len(folds)} folds\n", flush=True)

    # Pre-slice arrays for each OOS fold (avoids repeated indexing)
    oos_slices = [
        dict(
            close=close[f["oos_i0"]:f["oos_i1"]],
            high=high[f["oos_i0"]:f["oos_i1"]],
            low=low[f["oos_i0"]:f["oos_i1"]],
            sbull_raw=sbull_raw[f["oos_i0"]:f["oos_i1"]],
            sbear_raw=sbear_raw[f["oos_i0"]:f["oos_i1"]],
            entry_sbull=entry_sbull[f["oos_i0"]:f["oos_i1"]],
            entry_sbear=entry_sbear[f["oos_i0"]:f["oos_i1"]],
            atr14=atr14[f["oos_i0"]:f["oos_i1"]],
        )
        for f in folds
    ]

    # ── Build combos ──────────────────────────────────────────────────────────
    combos = []
    run_modes = (["fixed","trail","single","3tier"] if args.mode == "all"
                 else [args.mode])

    if "fixed" in run_modes:
        for sl in SL_MULTS:
            for t1 in TP1_RS:
                for t2 in TP2_RS:
                    if t2 > t1:
                        combos.append(("fixed", sl, t1, t2))

    if "trail" in run_modes:
        for sl in SL_MULTS:
            for t1 in TP1_RS:
                for tm in TRAIL_MULTS:
                    combos.append(("trail", sl, t1, tm))

    if "single" in run_modes:
        for sl in SL_MULTS:
            for tp in SINGLE_TPS:
                combos.append(("single", sl, tp, None))

    if "3tier" in run_modes:
        for sl in SL_MULTS:
            for tm in TIER3_TMULTS:
                combos.append(("3tier", sl, tm, None))

    print(f"Running {len(combos)} combos × {len(folds)} folds …", flush=True)

    # ── Run sweep ─────────────────────────────────────────────────────────────
    rows = []
    t_sim = time.time()
    for mode, p1, p2, p3 in combos:
        oos_r_per_fold = []
        for sl_ in oos_slices:
            if mode == "fixed":
                r = _sim_fixed(sl_["close"], sl_["high"], sl_["low"],
                               sl_["sbull_raw"], sl_["sbear_raw"],
                               sl_["entry_sbull"], sl_["entry_sbear"],
                               sl_["atr14"], p1, p2, p3)
            elif mode == "trail":
                r = _sim_trail(sl_["close"], sl_["high"], sl_["low"],
                               sl_["sbull_raw"], sl_["sbear_raw"],
                               sl_["entry_sbull"], sl_["entry_sbear"],
                               sl_["atr14"], p1, p2, p3)
            elif mode == "single":
                r = _sim_single(sl_["close"], sl_["high"], sl_["low"],
                                sl_["sbull_raw"], sl_["sbear_raw"],
                                sl_["entry_sbull"], sl_["entry_sbear"],
                                sl_["atr14"], p1, p2)
            else:  # 3tier
                r = _sim_3tier(sl_["close"], sl_["high"], sl_["low"],
                               sl_["sbull_raw"], sl_["sbear_raw"],
                               sl_["entry_sbull"], sl_["entry_sbear"],
                               sl_["atr14"], p1, p2)
            oos_r_per_fold.append(r)

        # Combined OOS metrics
        all_r = np.concatenate([r for r in oos_r_per_fold if len(r) > 0])
        m = metrics(all_r)
        pos_f = sum(1 for r in oos_r_per_fold if len(r) > 0 and r.sum() > 0)

        if mode == "fixed":
            label = f"fix  sl={p1:.2f} tp1={p2:.2f} tp2={p3:.2f}"
        elif mode == "trail":
            label = f"trail sl={p1:.2f} tp1={p2:.2f} tr={p3:.2f}"
        elif mode == "single":
            label = f"sngl sl={p1:.2f} tp={p2:.2f}"
        else:
            label = f"3t   sl={p1:.2f} tm={p2:.2f}"

        rows.append({
            "label": label, "mode": mode,
            "sl_mult": p1, "p2": p2, "p3": p3,
            **m,
            "pos_folds": pos_f,
        })

    sim_elapsed = time.time() - t_sim

    # ── Sort & print ──────────────────────────────────────────────────────────
    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values("sharpe", ascending=False).reset_index(drop=True)

    SEP = "─" * 100
    print(f"\n{SEP}")
    print(f"  EXIT PARAMETER SWEEP  |  OOS R-multiples  |  {len(combos)} combos × {len(folds)} folds")
    print(SEP)
    hdr = (f"  {'Rank':4s}  {'Label':38s}  {'n':5s}  {'Sharpe':7s}  "
           f"{'TotalR':7s}  {'AvgR':6s}  {'Win%':5s}  {'PF':5s}  {'PosF'}  ")
    print(hdr)
    print(f"  {'':4s}  {'-'*38}  {'-----':5s}  {'-'*7}  {'-'*7}  {'-'*6}  {'-----':5s}  {'-----':5s}  {'-'*5}")

    # Print top N
    top = min(args.top, len(df_res))
    for rank, row in df_res.head(top).iterrows():
        m_ = row
        print(f"  {rank+1:4d}  {m_['label']:38s}  {m_['n']:5d}  "
              f"{m_['sharpe']:+7.4f}  {m_['total_r']:+7.2f}  "
              f"{m_['avg_r']:+6.4f}  {100*m_['win_rate']:5.1f}%  "
              f"{m_['pf']:5.3f}  {m_['pos_folds']}/{len(folds)}")

    # Print baseline reference (3tier, no filter = what we know from prior run)
    ref_idx = df_res[
        (df_res["mode"] == "3tier") & (df_res["sl_mult"] == 3.5) & (df_res["p2"] == 2.0)
    ].index
    if len(ref_idx) > 0:
        r_ = df_res.loc[ref_idx[0]]
        ridx = df_res.index.get_loc(ref_idx[0])
        print(f"\n  REF   {r_['label']:38s}  {r_['n']:5d}  "
              f"{r_['sharpe']:+7.4f}  {r_['total_r']:+7.2f}  "
              f"{r_['avg_r']:+6.4f}  {100*r_['win_rate']:5.1f}%  "
              f"{r_['pf']:5.3f}  {r_['pos_folds']}/{len(folds)}"
              f"  (rank #{ridx+1})")

    print(f"\n  Sim time: {sim_elapsed:.1f}s  |  Total: {time.time()-t0:.1f}s")
    print(SEP)

    # ── Per-mode best ─────────────────────────────────────────────────────────
    print("\n  BEST PER MODE:")
    for mode in ["fixed", "trail", "single", "3tier"]:
        sub = df_res[df_res["mode"] == mode]
        if sub.empty: continue
        best = sub.iloc[0]
        print(f"    {mode:6s}  {best['label']:38s}  Sharpe={best['sharpe']:+.4f}  "
              f"TotalR={best['total_r']:+.2f}  n={best['n']}  {best['pos_folds']}/{len(folds)} folds")

    # ── Heatmap: fixed mode (sl_mult vs tp2_r, best tp1_r per cell) ──────────
    if "fixed" in run_modes:
        print("\n  FIXED MODE — Sharpe heatmap (sl_mult vs tp2_r, best tp1_r):")
        fixed_df = df_res[df_res["mode"] == "fixed"].copy()
        heat = fixed_df.pivot_table(index="sl_mult", columns="p3", values="sharpe",
                                    aggfunc="max")
        print("  sl\\tp2   " + "  ".join(f"{c:4.1f}" for c in heat.columns))
        for sl_m in heat.index:
            vals = "  ".join(f"{heat.loc[sl_m, c]:+.3f}" if not pd.isna(heat.loc[sl_m, c])
                             else "  N/A" for c in heat.columns)
            print(f"  {sl_m:4.2f}    {vals}")

    if "trail" in run_modes:
        print("\n  TRAIL MODE — Sharpe heatmap (sl_mult vs trail_mult, best tp1_r):")
        trail_df = df_res[df_res["mode"] == "trail"].copy()
        heat = trail_df.pivot_table(index="sl_mult", columns="p3", values="sharpe",
                                    aggfunc="max")
        print("  sl\\trail  " + "  ".join(f"{c:4.1f}" for c in heat.columns))
        for sl_m in heat.index:
            vals = "  ".join(f"{heat.loc[sl_m, c]:+.3f}" if not pd.isna(heat.loc[sl_m, c])
                             else "  N/A" for c in heat.columns)
            print(f"  {sl_m:4.2f}    {vals}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "million_moves_v43_sweep_results.csv")
    df_res[["label","mode","sl_mult","p2","p3","n","sharpe","total_r",
            "avg_r","win_rate","pf","pos_folds"]].to_csv(out_path, index=False)
    print(f"\n  Full results -> {out_path}")


if __name__ == "__main__":
    main()
