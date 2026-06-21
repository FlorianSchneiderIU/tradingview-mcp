"""Can a model identify, in real time, which 5m springs are at a weekly low?

The hindsight study showed springs near a weekly low run to 20-30R far more often
than springs elsewhere. To trade it we must predict the weekly-low zone from
structure available AT the spring - no future data. This module builds:

  * per-spring outcomes (max-R available, and net R at a fixed 20R target), and
  * a real-time feature matrix (5m structure + the *previous completed* daily bar's
    HTF context, shifted to avoid lookahead).

The label is reach_20R = (max favorable excursion >= 20R). A walk-forward classifier
(see run_spring_model.py) then ranks springs; if its top slice is net-positive at 20R
out-of-sample, the weekly-low timing is learnable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ltf_structure as lts, reuse


def spring_outcomes(ltf, spring_idx, excursion_rr, fixed_rr, max_hold, buffer, cost) -> pd.DataFrame:
    """Per-spring max-R (excursion) and net R at a fixed target. Overlap allowed -
    each spring is evaluated independently for labelling."""
    rows = []
    for idx in spring_idx:
        exc = reuse.simulate_sweep_trade(ltf, int(idx), "long", excursion_rr, max_hold, buffer, cost)
        if exc is None:
            continue
        fixed = reuse.simulate_sweep_trade(ltf, int(idx), "long", fixed_rr, max_hold, buffer, cost)
        rows.append({
            "idx": int(idx),
            "time": exc["signal_time"],
            "mfe_r": float(exc["mfe_r"]),
            "fixed_r": float(fixed["result_r"]) if fixed else np.nan,
        })
    return pd.DataFrame(rows)


def _daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    close = daily["close"].astype(float)
    high = daily["high"].astype(float)
    atr = daily["atr"].astype(float)
    out = pd.DataFrame({"open_time": pd.to_datetime(daily["open_time"], utc=True)})
    roll_high = high.rolling(30, min_periods=10).max()
    out["d_dd30"] = (close - roll_high) / roll_high
    out["d_since_high30"] = (high.rolling(30, min_periods=10)
                            .apply(lambda w: len(w) - 1 - int(np.argmax(w)), raw=True))
    ema50 = close.ewm(span=50, adjust=False).mean()
    out["d_ema50_dist"] = (close - ema50) / close
    out["d_atr_norm"] = atr / close
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["d_rsi14"] = (100 - 100 / (1 + gain / loss.replace(0.0, np.nan))) / 100.0
    # Shift so a 5m bar only ever sees the PREVIOUS completed daily bar (no lookahead).
    feat_cols = [c for c in out.columns if c != "open_time"]
    out[feat_cols] = out[feat_cols].shift(1)
    return out


def spring_features(ltf: pd.DataFrame, daily: pd.DataFrame, spring_lookback: int) -> pd.DataFrame:
    """Real-time structural features on every 5m bar (select spring rows later)."""
    close = ltf["close"].astype(float)
    high = ltf["high"].astype(float)
    low = ltf["low"].astype(float)
    open_ = ltf["open"].astype(float)
    atr = ltf["atr"].astype(float)
    logc = np.log(close)

    f = pd.DataFrame(index=ltf.index)
    for k in (12, 48, 288):
        f[f"l_ret{k}"] = logc.diff(k)
    ema200 = close.ewm(span=200, adjust=False).mean()
    f["l_ema200_dist"] = (close - ema200) / close
    f["l_atr_norm"] = atr / close
    f["l_vol48"] = logc.diff().rolling(48, min_periods=10).std()
    prev_low = low.shift(1).rolling(spring_lookback, min_periods=spring_lookback).min()
    f["l_sweep_depth"] = (prev_low - low) / atr            # how far below the swept low
    rng = (high - low).replace(0.0, np.nan)
    f["l_wick"] = (np.minimum(open_, close) - low) / rng
    roll_min = low.rolling(2016, min_periods=288).min()    # ~7d low on 5m
    f["l_above_7dlow"] = (close - roll_min) / close
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["l_rsi14"] = (100 - 100 / (1 + gain / loss.replace(0.0, np.nan))) / 100.0

    # Merge previous-day HTF context.
    dfeat = _daily_features(daily)
    left = pd.DataFrame({"open_time": pd.to_datetime(ltf["open_time"], utc=True)})
    merged = pd.merge_asof(left.sort_values("open_time"), dfeat.sort_values("open_time"),
                           on="open_time", direction="backward")
    for c in [c for c in dfeat.columns if c != "open_time"]:
        f[c] = merged[c].to_numpy()
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
