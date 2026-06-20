"""Direction-conditional and expansion labels (proposal sections 13-14).

  * dump_into_event / pump_into_event : ATR-normalised pre-event move (uses PAST bars only).
  * expansion success + MFE/MAE/max-R : forward window outcome (LABEL ONLY).

All R-multiples use a local invalidation (stop just beyond the event candle's
extreme), matching the proposal's "max_R_available using local invalidation".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def atr_normalized_move(frame: pd.DataFrame, lookback: int) -> np.ndarray:
    """(close[t] - close[t-lookback]) / ATR[t]. NaN for the first ``lookback`` bars."""
    close = frame["close"].to_numpy(dtype=float)
    atr = frame["atr"].to_numpy(dtype=float)
    out = np.full(len(frame), np.nan, dtype=float)
    out[lookback:] = (close[lookback:] - close[:-lookback]) / atr[lookback:]
    out[~np.isfinite(out)] = np.nan
    return out


def dump_flags(frame: pd.DataFrame, lookback: int, threshold_atr: float) -> np.ndarray:
    move = atr_normalized_move(frame, lookback)
    return np.where(np.isfinite(move), move <= -abs(threshold_atr), False)


def pump_flags(frame: pd.DataFrame, lookback: int, threshold_atr: float) -> np.ndarray:
    move = atr_normalized_move(frame, lookback)
    return np.where(np.isfinite(move), move >= abs(threshold_atr), False)


def evaluate_expansion(
    frame: pd.DataFrame,
    t: int,
    direction: str,
    horizon: int,
    target_atr: float,
    buffer_atr: float,
) -> dict | None:
    """Evaluate post-event expansion over the next ``horizon`` candles from event candle ``t``.

    direction = 'bull' looks for an up-expansion (local bottom thesis); 'bear' the reverse.
    Returns success flag, MFE/MAE in ATR and R units, max_R_available, and timing.
    """
    n = len(frame)
    if t < 0 or t + 1 >= n:
        return None
    close = float(frame["close"].iloc[t])
    atr = float(frame["atr"].iloc[t])
    if not math.isfinite(atr) or atr <= 0:
        return None

    if direction == "bull":
        stop = float(frame["low"].iloc[t]) - buffer_atr * atr
        target = close + target_atr * atr
        risk = close - stop
    else:
        stop = float(frame["high"].iloc[t]) + buffer_atr * atr
        target = close - target_atr * atr
        risk = stop - close
    if not math.isfinite(risk) or risk <= 0:
        return None

    end = min(n - 1, t + horizon)
    mfe_atr = 0.0
    mae_atr = 0.0
    time_to_mfe = 0
    time_to_mae = 0
    outcome = "none"
    hit_bar = end
    for cursor in range(t + 1, end + 1):
        high = float(frame["high"].iloc[cursor])
        low = float(frame["low"].iloc[cursor])
        if direction == "bull":
            fav = (high - close) / atr
            adv = (close - low) / atr
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            fav = (close - low) / atr
            adv = (high - close) / atr
            hit_stop = high >= stop
            hit_target = low <= target
        if fav > mfe_atr:
            mfe_atr = fav
            time_to_mfe = cursor - t
        if adv > mae_atr:
            mae_atr = adv
            time_to_mae = cursor - t
        # Conservative: if both touched in one bar, assume the stop filled first.
        if hit_stop:
            outcome = "stop"
            hit_bar = cursor
            break
        if hit_target:
            outcome = "target"
            hit_bar = cursor
            break

    return {
        "event_bar": int(t),
        "direction": direction,
        "success": outcome == "target",
        "outcome": outcome,
        "hit_bar_offset": int(hit_bar - t),
        "entry_ref": close,
        "stop": stop,
        "target": target,
        "risk_atr": risk / atr,
        "mfe_atr": float(mfe_atr),
        "mae_atr": float(mae_atr),
        "mfe_r": float(mfe_atr * atr / risk),
        "mae_r": float(mae_atr * atr / risk),
        "max_r_available": float(mfe_atr * atr / risk),
        "time_to_mfe": int(time_to_mfe),
        "time_to_mae": int(time_to_mae),
    }


def expansion_table(
    frame: pd.DataFrame,
    event_bars: list[int],
    direction: str,
    horizon: int,
    target_atr: float,
    buffer_atr: float,
) -> pd.DataFrame:
    """Expansion metrics for each event candle. One row per evaluable event."""
    rows = []
    times = pd.to_datetime(frame["open_time"], utc=True)
    for t in sorted(set(int(b) for b in event_bars)):
        res = evaluate_expansion(frame, t, direction, horizon, target_atr, buffer_atr)
        if res is None:
            continue
        res["event_time"] = pd.Timestamp(times.iloc[t])
        rows.append(res)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)
