"""
Million Moves Algo V4.3 — Walk-Forward Parameter Optimizer
===========================================================

Anti-lookahead guarantees
  - All indicators (EMA, ATR, Supertrend) computed CAUSALLY on the full
    dataset — each bar only uses data up to and including that bar.
  - Walk-forward: IS window strictly before OOS window (train_i1 <= oos_i0).
  - Best params selected on IS data only, then frozen for OOS evaluation.
  - Warmup bars handled naturally: indicators computed from bar 0 with no
    future information.

Parameter grid
  supertrend_mult : 1.0 to 5.0, step 0.5  =>  9 values
  sl_atr_mult     : 1.0 to 4.0, step 0.5  =>  7 values
  tp_mult         : 0.5 to 3.0, step 0.5  =>  6 values
  Total per fold  : 9 x 7 x 6 = 378 combos

Walk-forward schedule (default)
  Train  : 12 months rolling
  OOS    :  3 months
  Step   :  3 months
  ~6 folds on BINANCE ETHUSDT 15m (2024-01-01 to present)

Objective : per-trade Sharpe = mean(trade_pnl%) / std(trade_pnl%, ddof=1)
            (minimum 5 IS trades required)

Outputs
  million_moves_v43_wf_folds.csv       -- per-fold summary
  million_moves_v43_wf_oos_trades.csv  -- all OOS trades across folds
  million_moves_v43_wf_grid.csv        -- full IS grid (--save-grid flag)
"""

from __future__ import annotations

import math
import os
import sys
import argparse
import time
import warnings
from multiprocessing import Pool, cpu_count, freeze_support

import numpy as np
import pandas as pd
import ccxt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ST_MULT_GRID  = [round(x * 0.5, 1) for x in range(2, 11)]   # 1.0 … 5.0
SL_MULT_GRID  = [round(x * 0.5, 1) for x in range(2, 9)]    # 1.0 … 4.0
TP_MULT_GRID  = [round(x * 0.5, 1) for x in range(1, 7)]    # 0.5 … 3.0

TRAIN_MONTHS  = 12
OOS_MONTHS    = 3
STEP_MONTHS   = 3
MIN_TRADES_IS = 5

ST_ATR_LEN    = 11
EMA_LEN       = 200
SMA_LEN       = 13
ATR_SL_LEN    = 14

SYMBOL        = "ETH/USDT"
TIMEFRAME     = "15m"
SINCE_DATE    = "2024-01-01"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, timeframe: str, since_date: str) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since_date}T00:00:00Z")
    bars: list = []
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
# Indicator math — causal, numpy arrays
# ---------------------------------------------------------------------------
def _rma(vals: np.ndarray, length: int) -> np.ndarray:
    """Wilder's smoothed moving average matching Pine's ta.rma."""
    alpha = 1.0 / length
    out = np.full(len(vals), np.nan)
    # find first non-nan
    start = 0
    while start < len(vals) and np.isnan(vals[start]):
        start += 1
    seed_end = start + length
    if seed_end > len(vals):
        return out
    out[seed_end - 1] = float(np.nanmean(vals[start:seed_end]))
    for i in range(seed_end, len(vals)):
        prev = out[i - 1]
        v = vals[i]
        out[i] = alpha * v + (1.0 - alpha) * prev if not np.isnan(v) else prev
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                length: int) -> np.ndarray:
    prev_c = np.empty_like(close)
    prev_c[0] = np.nan
    prev_c[1:] = close[:-1]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)),
    )
    return _rma(tr, length)


def compute_ema(close: np.ndarray, length: int) -> np.ndarray:
    alpha = 2.0 / (length + 1)
    out = np.full(len(close), np.nan)
    # find first non-nan and seed
    for i, v in enumerate(close):
        if not np.isnan(v):
            out[i] = v
            for j in range(i + 1, len(close)):
                out[j] = alpha * close[j] + (1.0 - alpha) * out[j - 1]
            break
    return out


def compute_sma(close: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(close).rolling(length).mean().values


def compute_supertrend(open_: np.ndarray, close: np.ndarray,
                       atr_st: np.ndarray, mult: float) -> np.ndarray:
    """
    Pine Script supertrend with source=open_.
    Direction/ratchet checks use close (matching the original algo).
    Returns the supertrend line array.
    """
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
            # ratchet lower band
            lower[i] = lr[i] if (lr[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) \
                              else lower[i - 1]
            # ratchet upper band
            upper[i] = ur[i] if (ur[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) \
                              else upper[i - 1]
            # direction
            prev_st = st[i - 1]
            if np.isnan(prev_st):
                prev_st = upper[i - 1] if not np.isnan(upper[i - 1]) else lower[i - 1]
            if prev_st == upper[i - 1]:
                direction[i] = -1.0 if close[i] > upper[i] else 1.0
            else:
                direction[i] =  1.0 if close[i] < lower[i] else -1.0
        st[i] = lower[i] if direction[i] == -1.0 else upper[i]

    return st


# ---------------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------------
def build_signals(close: np.ndarray, open_: np.ndarray,
                  sma13: np.ndarray, ema200: np.ndarray,
                  atr_st: np.ndarray, st_mult: float,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (sbull, sbear) boolean arrays.
    All logic is causal: prev_* uses a 1-bar shift (np.roll with NaN guard).
    """
    st = compute_supertrend(open_, close, atr_st, st_mult)
    n = len(close)

    prev_c = np.empty(n); prev_c[0] = np.nan; prev_c[1:] = close[:-1]
    prev_s = np.empty(n); prev_s[0] = np.nan; prev_s[1:] = st[:-1]
    prev_e = np.empty(n); prev_e[0] = np.nan; prev_e[1:] = ema200[:-1]

    # crossover / crossunder (causal)
    co = (~np.isnan(prev_c)) & (~np.isnan(prev_s)) & \
         (~np.isnan(st)) & (prev_c < prev_s) & (close > st)
    cu = (~np.isnan(prev_c)) & (~np.isnan(prev_s)) & \
         (~np.isnan(st)) & (prev_c > prev_s) & (close < st)

    above_ema = (~np.isnan(prev_e)) & (~np.isnan(ema200)) & \
                (prev_c > prev_e) & (close > ema200)

    sbull = co & (~np.isnan(sma13)) & (close >= sma13) & above_ema
    sbear = cu & (~np.isnan(sma13)) & (close <= sma13) & (~above_ema)

    return sbull.astype(bool), sbear.astype(bool)


# ---------------------------------------------------------------------------
# Walk-forward fold generator
# ---------------------------------------------------------------------------
def generate_wf_folds(
    index: pd.DatetimeIndex,
    train_months: int = TRAIN_MONTHS,
    oos_months:   int = OOS_MONTHS,
    step_months:  int = STEP_MONTHS,
) -> list[dict]:
    folds = []
    fold_id = 1
    data_end = index[-1]
    fold_start = index[0]

    while True:
        train_end  = fold_start + pd.DateOffset(months=train_months)
        oos_start  = train_end
        oos_end    = oos_start + pd.DateOffset(months=oos_months)

        if oos_start > data_end:
            break
        # clip OOS to data end
        oos_end_clipped = min(oos_end, data_end + pd.Timedelta(seconds=1))

        tr_il  = np.where((index >= fold_start) & (index < train_end))[0]
        oos_il = np.where((index >= oos_start) & (index < oos_end_clipped))[0]

        if len(tr_il) > 50 and len(oos_il) > 0:
            folds.append({
                "fold_id":     fold_id,
                "train_start": fold_start,
                "train_end":   train_end,
                "oos_start":   oos_start,
                "oos_end":     oos_end_clipped,
                "train_i0":    int(tr_il[0]),
                "train_i1":    int(tr_il[-1]) + 1,
                "oos_i0":      int(oos_il[0]),
                "oos_i1":      int(oos_il[-1]) + 1,
                "train_bars":  len(tr_il),
                "oos_bars":    len(oos_il),
            })

        fold_start = fold_start + pd.DateOffset(months=step_months)
        fold_id += 1

    return folds


# ---------------------------------------------------------------------------
# Fast simulation — returns per-signal PnL% array (IS hot path)
# ---------------------------------------------------------------------------
def _simulate_pnl(
    close:    np.ndarray,
    high:     np.ndarray,
    low:      np.ndarray,
    sbull:    np.ndarray,
    sbear:    np.ndarray,
    atr14:    np.ndarray,
    sl_mult:  float,
    tp_mult:  float,
) -> np.ndarray:
    pnl_list: list[float] = []

    active = False
    is_long = False
    entry = 0.0
    sl_ = tp1 = tp2 = tp3 = 0.0
    remain = 0.0
    tp1_hit = tp2_hit = False
    acc = 0.0

    for i in range(1, len(close)):
        h = high[i]; l = low[i]; c = close[i]
        atr = atr14[i]

        if active:
            if is_long:
                sl_hit = l <= sl_
                if sl_hit and not tp1_hit:
                    pnl_list.append(acc + (sl_ - entry) / entry * remain)
                    active = False
                else:
                    if not tp1_hit and h >= tp1:
                        acc += (tp1 - entry) / entry * 0.33
                        remain -= 0.33
                        tp1_hit = True
                    if active and tp1_hit and not tp2_hit and h >= tp2:
                        f = remain * 0.5
                        acc += (tp2 - entry) / entry * f
                        remain -= f
                        tp2_hit = True
                    if active and tp2_hit and h >= tp3:
                        pnl_list.append(acc + (tp3 - entry) / entry * remain)
                        active = False
                    elif active and sl_hit:
                        pnl_list.append(acc + (sl_ - entry) / entry * remain)
                        active = False
            else:  # short
                sl_hit = h >= sl_
                if sl_hit and not tp1_hit:
                    pnl_list.append(acc + (entry - sl_) / entry * remain)
                    active = False
                else:
                    if not tp1_hit and l <= tp1:
                        acc += (entry - tp1) / entry * 0.33
                        remain -= 0.33
                        tp1_hit = True
                    if active and tp1_hit and not tp2_hit and l <= tp2:
                        f = remain * 0.5
                        acc += (entry - tp2) / entry * f
                        remain -= f
                        tp2_hit = True
                    if active and tp2_hit and l <= tp3:
                        pnl_list.append(acc + (entry - tp3) / entry * remain)
                        active = False
                    elif active and sl_hit:
                        pnl_list.append(acc + (entry - sl_) / entry * remain)
                        active = False

        # reversal
        if active and is_long and sbear[i]:
            pnl_list.append(acc + (c - entry) / entry * remain)
            active = False
        if active and not is_long and sbull[i]:
            pnl_list.append(acc + (entry - c) / entry * remain)
            active = False

        # new entry
        if not active and not math.isnan(atr):
            if sbull[i]:
                sl_  = l - atr * sl_mult
                risk = max(c - sl_, 1e-10)
                entry = c; is_long = True
                tp1 = c + tp_mult * risk
                tp2 = c + 2 * tp_mult * risk
                tp3 = c + 3 * tp_mult * risk
                remain = 1.0; tp1_hit = tp2_hit = False; acc = 0.0; active = True
            elif sbear[i]:
                sl_  = h + atr * sl_mult
                risk = max(sl_ - c, 1e-10)
                entry = c; is_long = False
                tp1 = c - tp_mult * risk
                tp2 = c - 2 * tp_mult * risk
                tp3 = c - 3 * tp_mult * risk
                remain = 1.0; tp1_hit = tp2_hit = False; acc = 0.0; active = True

    # close any open trade at end of period
    if active:
        cl = close[-1]
        pnl = (cl - entry) / entry if is_long else (entry - cl) / entry
        pnl_list.append(acc + pnl * remain)

    return np.array(pnl_list, dtype=np.float64)


# ---------------------------------------------------------------------------
# Detailed simulation — for OOS trade logging
# ---------------------------------------------------------------------------
def _simulate_detailed(
    close:      np.ndarray,
    high:       np.ndarray,
    low:        np.ndarray,
    sbull:      np.ndarray,
    sbear:      np.ndarray,
    atr14:      np.ndarray,
    sl_mult:    float,
    tp_mult:    float,
    timestamps: pd.DatetimeIndex,
) -> list[dict]:
    trades: list[dict] = []

    active = False
    is_long = False
    entry = 0.0
    sl_ = tp1 = tp2 = tp3 = 0.0
    remain = 0.0
    tp1_hit = tp2_hit = False
    acc = 0.0
    entry_time = None
    reasons: list[str] = []
    entry_h = entry_l = 0.0  # high/low at signal bar (for logging)

    def _save(ts, exit_price: float) -> None:
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   ts,
            "direction":   "long" if is_long else "short",
            "entry_price": round(float(entry), 6),
            "exit_price":  round(float(exit_price), 6),
            "sl":          round(float(sl_), 6),
            "tp1":         round(float(tp1), 6),
            "tp2":         round(float(tp2), 6),
            "tp3":         round(float(tp3), 6),
            "exit_reasons": "|".join(reasons),
            "pnl_pct":     round(float(acc) * 100, 4),
        })

    for i in range(1, len(close)):
        h = high[i]; l = low[i]; c = close[i]
        atr = atr14[i]; ts = timestamps[i]

        if active:
            if is_long:
                sl_hit = l <= sl_
                if sl_hit and not tp1_hit:
                    acc += (sl_ - entry) / entry * remain
                    reasons.append("SL"); _save(ts, sl_); active = False
                else:
                    if not tp1_hit and h >= tp1:
                        acc += (tp1 - entry) / entry * 0.33
                        remain -= 0.33; tp1_hit = True; reasons.append("TP1")
                    if active and tp1_hit and not tp2_hit and h >= tp2:
                        f = remain * 0.5
                        acc += (tp2 - entry) / entry * f
                        remain -= f; tp2_hit = True; reasons.append("TP2")
                    if active and tp2_hit and h >= tp3:
                        acc += (tp3 - entry) / entry * remain
                        reasons.append("TP3"); _save(ts, tp3); active = False
                    elif active and sl_hit:
                        acc += (sl_ - entry) / entry * remain
                        reasons.append("SL"); _save(ts, sl_); active = False
            else:
                sl_hit = h >= sl_
                if sl_hit and not tp1_hit:
                    acc += (entry - sl_) / entry * remain
                    reasons.append("SL"); _save(ts, sl_); active = False
                else:
                    if not tp1_hit and l <= tp1:
                        acc += (entry - tp1) / entry * 0.33
                        remain -= 0.33; tp1_hit = True; reasons.append("TP1")
                    if active and tp1_hit and not tp2_hit and l <= tp2:
                        f = remain * 0.5
                        acc += (entry - tp2) / entry * f
                        remain -= f; tp2_hit = True; reasons.append("TP2")
                    if active and tp2_hit and l <= tp3:
                        acc += (entry - tp3) / entry * remain
                        reasons.append("TP3"); _save(ts, tp3); active = False
                    elif active and sl_hit:
                        acc += (entry - sl_) / entry * remain
                        reasons.append("SL"); _save(ts, sl_); active = False

        # reversal exits
        if active and is_long and sbear[i]:
            acc += (c - entry) / entry * remain
            reasons.append("Rev"); _save(ts, c); active = False
        if active and not is_long and sbull[i]:
            acc += (entry - c) / entry * remain
            reasons.append("Rev"); _save(ts, c); active = False

        # new entry
        if not active and not math.isnan(atr):
            if sbull[i]:
                sl_  = l - atr * sl_mult
                risk = max(c - sl_, 1e-10)
                entry = c; is_long = True
                tp1 = c + tp_mult * risk
                tp2 = c + 2 * tp_mult * risk
                tp3 = c + 3 * tp_mult * risk
                remain = 1.0; tp1_hit = tp2_hit = False
                acc = 0.0; active = True; entry_time = ts; reasons = []
                entry_h = h; entry_l = l
            elif sbear[i]:
                sl_  = h + atr * sl_mult
                risk = max(sl_ - c, 1e-10)
                entry = c; is_long = False
                tp1 = c - tp_mult * risk
                tp2 = c - 2 * tp_mult * risk
                tp3 = c - 3 * tp_mult * risk
                remain = 1.0; tp1_hit = tp2_hit = False
                acc = 0.0; active = True; entry_time = ts; reasons = []
                entry_h = h; entry_l = l

    # close open trade at end of period
    if active:
        cl = close[-1]
        pnl = (cl - entry) / entry if is_long else (entry - cl) / entry
        acc += pnl * remain
        reasons.append("Open"); _save(timestamps[-1], cl)

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pnls: np.ndarray, min_trades: int = MIN_TRADES_IS) -> dict:
    n = len(pnls)
    if n < min_trades:
        return {"sharpe": -99.0, "n_trades": n, "pnl_pct": 0.0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "profit_factor": 0.0}
    std  = float(np.std(pnls, ddof=1))
    mean = float(np.mean(pnls))
    sharpe = mean / std if std > 1e-12 else 0.0
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gw = float(wins.sum())   if len(wins)   > 0 else 0.0
    gl = float(-losses.sum()) if len(losses) > 0 else 0.0
    return {
        "sharpe":        round(sharpe, 5),
        "n_trades":      n,
        "pnl_pct":       round(float(pnls.sum()) * 100, 4),
        "win_rate":      round(float(len(wins)) / n, 4),
        "avg_win":       round(float(wins.mean())   * 100 if len(wins)   > 0 else 0.0, 4),
        "avg_loss":      round(float(losses.mean()) * 100 if len(losses) > 0 else 0.0, 4),
        "profit_factor": round(gw / gl if gl > 0 else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker — module-level for Windows pickling
# ---------------------------------------------------------------------------
def _worker_run_fold_stmult(args: tuple) -> list[dict]:
    """
    Runs all SL_MULT x TP_MULT combos for one (fold_id, st_mult) pair.
    Takes pre-sliced training arrays to avoid sending full dataset via IPC.
    """
    (fold_id, st_mult,
     close_tr, high_tr, low_tr,
     sbull_tr, sbear_tr, atr14_tr) = args

    results = []
    for sl_mult in SL_MULT_GRID:
        for tp_mult in TP_MULT_GRID:
            pnls = _simulate_pnl(
                close_tr, high_tr, low_tr,
                sbull_tr, sbear_tr, atr14_tr,
                sl_mult, tp_mult,
            )
            m = compute_metrics(pnls)
            results.append({
                "fold_id": fold_id,
                "st_mult": st_mult,
                "sl_mult": sl_mult,
                "tp_mult": tp_mult,
                **m,
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MM V4.3 Walk-Forward Optimizer")
    parser.add_argument("--since",     default=SINCE_DATE,
                        help="Start date for data fetch (YYYY-MM-DD)")
    parser.add_argument("--symbol",    default=SYMBOL)
    parser.add_argument("--tf",        default=TIMEFRAME)
    parser.add_argument("--workers",   type=int,
                        default=max(1, cpu_count() - 1),
                        help="Worker processes (0 = sequential, no multiprocessing)")
    parser.add_argument("--save-grid", action="store_true",
                        help="Save full IS grid to CSV")
    parser.add_argument("--train",     type=int, default=TRAIN_MONTHS,
                        help="Training window in months")
    parser.add_argument("--oos",       type=int, default=OOS_MONTHS,
                        help="OOS window in months")
    parser.add_argument("--step",      type=int, default=STEP_MONTHS,
                        help="Step size in months")
    args = parser.parse_args()
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    df = fetch_ohlcv(args.symbol, args.tf, args.since)

    # ------------------------------------------------------------------
    # 2. Compute indicators on full dataset (causal — no lookahead)
    # ------------------------------------------------------------------
    print("Computing indicators…", flush=True)
    open_  = df["open"].values.astype(np.float64)
    close  = df["close"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)

    atr_st = compute_atr(high, low, close, ST_ATR_LEN)
    atr14  = compute_atr(high, low, close, ATR_SL_LEN)
    ema200 = compute_ema(close, EMA_LEN)
    sma13  = compute_sma(close, SMA_LEN)

    # ------------------------------------------------------------------
    # 3. Pre-compute signals for every st_mult (causal, full dataset)
    # ------------------------------------------------------------------
    print(f"Pre-computing signals for {len(ST_MULT_GRID)} st_mult values…", flush=True)
    all_signals: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for st_mult in ST_MULT_GRID:
        sb, sr = build_signals(close, open_, sma13, ema200, atr_st, st_mult)
        all_signals[st_mult] = (sb, sr)
        print(f"  st_mult={st_mult:.1f}:  Sbull={int(sb.sum()):4d}  Sbear={int(sr.sum()):4d}",
              flush=True)

    # ------------------------------------------------------------------
    # 4. Walk-forward fold generation
    # ------------------------------------------------------------------
    folds = generate_wf_folds(df.index, args.train, args.oos, args.step)
    print(f"\nGenerated {len(folds)} walk-forward folds:")
    for f in folds:
        print(f"  Fold {f['fold_id']:2d}: "
              f"train {str(f['train_start'])[:10]} -> {str(f['train_end'])[:10]}  |  "
              f"OOS   {str(f['oos_start'])[:10]} -> {str(f['oos_end'])[:10]}  "
              f"(train={f['train_bars']:,} bars, oos={f['oos_bars']:,} bars)")
    if not folds:
        print("ERROR: No folds generated. Check date range / train-months setting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Build worker tasks (fold x st_mult)
    # ------------------------------------------------------------------
    tasks = []
    for fold in folds:
        i0, i1 = fold["train_i0"], fold["train_i1"]
        for st_mult in ST_MULT_GRID:
            sb, sr = all_signals[st_mult]
            tasks.append((
                fold["fold_id"], st_mult,
                close[i0:i1].copy(),
                high[i0:i1].copy(),
                low[i0:i1].copy(),
                sb[i0:i1].copy(),
                sr[i0:i1].copy(),
                atr14[i0:i1].copy(),
            ))

    n_tasks  = len(tasks)
    n_combos = n_tasks * len(SL_MULT_GRID) * len(TP_MULT_GRID)
    print(f"\nIS grid: {n_tasks} tasks × {len(SL_MULT_GRID) * len(TP_MULT_GRID)} combos = "
          f"{n_combos:,} backtests …", flush=True)

    # ------------------------------------------------------------------
    # 6. Run IS grid
    # ------------------------------------------------------------------
    all_is: list[dict] = []
    t_start = time.time()
    report_every = max(1, n_tasks // 8)

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for k, chunk in enumerate(pool.imap_unordered(
                    _worker_run_fold_stmult, tasks, chunksize=2)):
                all_is.extend(chunk)
                if (k + 1) % report_every == 0 or (k + 1) == n_tasks:
                    print(f"  {k+1}/{n_tasks} tasks done  "
                          f"({time.time()-t_start:.0f}s elapsed)", flush=True)
    else:
        for k, task in enumerate(tasks):
            all_is.extend(_worker_run_fold_stmult(task))
            if (k + 1) % report_every == 0 or (k + 1) == n_tasks:
                print(f"  {k+1}/{n_tasks} tasks done  "
                      f"({time.time()-t_start:.0f}s elapsed)", flush=True)

    is_df = pd.DataFrame(all_is)
    print(f"IS grid complete in {time.time()-t_start:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 7. Per-fold: select best IS params -> evaluate OOS
    # ------------------------------------------------------------------
    print("\nEvaluating OOS with best IS params…", flush=True)
    fold_results: list[dict] = []
    oos_trades_all: list[dict] = []

    for fold in folds:
        fid = fold["fold_id"]
        fold_is = is_df[is_df["fold_id"] == fid]
        if fold_is.empty or fold_is["sharpe"].max() <= -98.0:
            print(f"  Fold {fid}: no valid IS combos — skipping.")
            continue

        # Best IS params (selected on IS Sharpe only — OOS never seen)
        best_row = fold_is.loc[fold_is["sharpe"].idxmax()]
        best = {
            "st": float(best_row["st_mult"]),
            "sl": float(best_row["sl_mult"]),
            "tp": float(best_row["tp_mult"]),
        }

        # IS metrics with best params
        i0_tr, i1_tr = fold["train_i0"], fold["train_i1"]
        is_pnls = _simulate_pnl(
            close[i0_tr:i1_tr], high[i0_tr:i1_tr], low[i0_tr:i1_tr],
            all_signals[best["st"]][0][i0_tr:i1_tr],
            all_signals[best["st"]][1][i0_tr:i1_tr],
            atr14[i0_tr:i1_tr],
            best["sl"], best["tp"],
        )
        im = compute_metrics(is_pnls, min_trades=0)

        # OOS evaluation — params frozen, OOS slice never influenced selection
        i0_oo, i1_oo = fold["oos_i0"], fold["oos_i1"]
        oos_pnls = _simulate_pnl(
            close[i0_oo:i1_oo], high[i0_oo:i1_oo], low[i0_oo:i1_oo],
            all_signals[best["st"]][0][i0_oo:i1_oo],
            all_signals[best["st"]][1][i0_oo:i1_oo],
            atr14[i0_oo:i1_oo],
            best["sl"], best["tp"],
        )
        om = compute_metrics(oos_pnls, min_trades=0)

        # Detailed OOS trades for export
        oos_td = _simulate_detailed(
            close[i0_oo:i1_oo], high[i0_oo:i1_oo], low[i0_oo:i1_oo],
            all_signals[best["st"]][0][i0_oo:i1_oo],
            all_signals[best["st"]][1][i0_oo:i1_oo],
            atr14[i0_oo:i1_oo],
            best["sl"], best["tp"],
            df.index[i0_oo:i1_oo],
        )
        for t in oos_td:
            t["fold_id"]  = fid
            t["st_mult"]  = best["st"]
            t["sl_mult"]  = best["sl"]
            t["tp_mult"]  = best["tp"]
        oos_trades_all.extend(oos_td)

        fold_results.append({
            "fold_id":          fid,
            "train_start":      str(fold["train_start"])[:10],
            "train_end":        str(fold["train_end"])[:10],
            "oos_start":        str(fold["oos_start"])[:10],
            "oos_end":          str(fold["oos_end"])[:10],
            "best_st_mult":     best["st"],
            "best_sl_mult":     best["sl"],
            "best_tp_mult":     best["tp"],
            "is_sharpe":        im["sharpe"],
            "is_n_trades":      im["n_trades"],
            "is_pnl_pct":       im["pnl_pct"],
            "is_win_rate":      im["win_rate"],
            "is_profit_factor": im["profit_factor"],
            "oos_sharpe":       om["sharpe"],
            "oos_n_trades":     om["n_trades"],
            "oos_pnl_pct":      om["pnl_pct"],
            "oos_win_rate":     om["win_rate"],
            "oos_profit_factor": om["profit_factor"],
        })

        print(
            f"  Fold {fid}  st={best['st']:.1f} sl={best['sl']:.1f} tp={best['tp']:.1f}"
            f"  IS  Sharpe={im['sharpe']:+.3f} n={im['n_trades']:3d}"
            f"  OOS Sharpe={om['sharpe']:+.3f} n={om['n_trades']:3d}"
            f"  OOS pnl={om['pnl_pct']:+.2f}%",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    SEP = "─" * 72
    print(f"\n{SEP}")
    print("  WALK-FORWARD OPTIMIZATION SUMMARY")
    print(SEP)

    if fold_results:
        folds_df = pd.DataFrame(fold_results)
        print(folds_df[[
            "fold_id", "best_st_mult", "best_sl_mult", "best_tp_mult",
            "is_sharpe", "oos_sharpe",
            "is_n_trades", "oos_n_trades",
            "is_pnl_pct", "oos_pnl_pct",
        ]].to_string(index=False))

        avg_is  = folds_df["is_sharpe"].mean()
        avg_oos = folds_df["oos_sharpe"].mean()
        ratio   = avg_oos / avg_is if abs(avg_is) > 1e-6 else float("nan")

        print(f"\n  Avg IS  Sharpe : {avg_is:+.4f}")
        print(f"  Avg OOS Sharpe : {avg_oos:+.4f}")
        print(f"  OOS/IS ratio   : {ratio:.3f}  (>0.5 = good robustness)")

        pc = folds_df[["best_st_mult", "best_sl_mult", "best_tp_mult"]].value_counts()
        print(f"\n  Most frequent best-param combos across folds:")
        print(pc.head(6).to_string())
    else:
        folds_df = pd.DataFrame()

    oos_df = pd.DataFrame(oos_trades_all) if oos_trades_all else pd.DataFrame()
    if not oos_df.empty:
        ap = oos_df["pnl_pct"].values
        print(f"\n  Combined OOS: {len(oos_df)} trades  "
              f"win={100 * (ap > 0).mean():.1f}%  "
              f"avg={ap.mean():+.3f}%  total={ap.sum():+.2f}%")

    print(f"\n  Total elapsed: {time.time() - t0:.1f}s")
    print(SEP)

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------
    if not folds_df.empty:
        p = os.path.join(OUT_DIR, "million_moves_v43_wf_folds.csv")
        folds_df.to_csv(p, index=False)
        print(f"\nFold summary  -> {p}")

    if not oos_df.empty:
        p = os.path.join(OUT_DIR, "million_moves_v43_wf_oos_trades.csv")
        oos_df.to_csv(p, index=False)
        print(f"OOS trades    -> {p}")

    if args.save_grid and not is_df.empty:
        p = os.path.join(OUT_DIR, "million_moves_v43_wf_grid.csv")
        is_df.to_csv(p, index=False)
        print(f"IS full grid  -> {p}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    freeze_support()
    main()
