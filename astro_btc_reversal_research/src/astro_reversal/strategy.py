"""High-RR reversal strategy (proposal Milestone 4).

Thesis built only on what survived testing:
  * HTF (daily) dump -> a *bottom zone* where mean reversion is likely (the 77% base
    rate). This is the timing/context, NOT a prediction.
  * LTF (e.g. 1h) **sweep + reclaim** of a recent low -> the entry trigger ("the
    right reaction"). Stop just below the swept low gives a tight risk -> high RR.

The edge, if any, is RR asymmetry, so we measure the full R-distribution and the
gated-vs-ungated contribution honestly. Entry/stop/target use the reused
``simulate_sweep_trade``; the LTF frame must come from ``add_ltf_indicators``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import event_labels, reuse


def daily_dump_gate(daily: pd.DataFrame, lookback: int, threshold_atr: float, hold_days: int):
    """Return (dump_flags, active_day) where active_day is True for `hold_days`
    after each daily dump (the long-alert window)."""
    dump = event_labels.dump_flags(daily, lookback, threshold_atr)
    active = pd.Series(dump.astype(int)).rolling(hold_days, min_periods=1).max().fillna(0).astype(bool)
    return dump, active.to_numpy()


def ltf_active_mask(daily: pd.DataFrame, ltf: pd.DataFrame, active_day: np.ndarray) -> np.ndarray:
    """Broadcast the daily long-alert window onto LTF bars (most-recent daily bar)."""
    dser = pd.DataFrame({
        "open_time": pd.to_datetime(daily["open_time"], utc=True),
        "active": active_day,
    }).sort_values("open_time")
    left = pd.DataFrame({"open_time": pd.to_datetime(ltf["open_time"], utc=True)}).sort_values("open_time")
    merged = pd.merge_asof(left, dser, on="open_time", direction="backward")
    return merged["active"].fillna(False).to_numpy(dtype=bool)


def ltf_long_signals(
    ltf: pd.DataFrame,
    lookback: int = 12,
    require_displacement: bool = False,
    disp_body_atr: float = 0.5,
    disp_close_frac: float = 0.5,
) -> np.ndarray:
    """Bullish sweep+reclaim of a significant prior low.

    Sweep: the bar trades below the lowest low of the prior ``lookback`` bars but
    closes back above it. With ``require_displacement`` the reclaim must also be a
    strong bullish candle (body >= disp_body_atr*ATR and close in the top
    (1-disp_close_frac) of the range) - i.e. an actual reaction, not a weak poke.
    """
    prev_low = ltf["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    sweep = (ltf["low"] < prev_low) & (ltf["close"] > prev_low)
    if require_displacement:
        rng = (ltf["high"] - ltf["low"]).replace(0.0, np.nan)
        body = ltf["close"] - ltf["open"]
        disp = (body > 0) & (body >= disp_body_atr * ltf["atr"]) & \
               (((ltf["close"] - ltf["low"]) / rng) >= disp_close_frac)
        sweep = sweep & disp
    return sweep.fillna(False).to_numpy(dtype=bool)


def backtest_long(
    ltf: pd.DataFrame,
    entry_mask: np.ndarray,
    rr: float,
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> list[dict]:
    """Walk forward bar by bar, taking non-overlapping long sweep entries where
    entry_mask is True."""
    trades: list[dict] = []
    cursor = 20
    n = len(ltf)
    while cursor < n - 2:
        if not entry_mask[cursor]:
            cursor += 1
            continue
        trade = reuse.simulate_sweep_trade(
            ltf, cursor, "long", rr, max_hold_bars, stop_buffer_atr, cost_bps_round_trip)
        if trade is None:
            cursor += 1
            continue
        trades.append(trade)
        cursor = max(cursor + 1, int(trade["exit_idx"]) + 1)
    return trades


def r_distribution(trades: list[dict], levels=(1, 2, 3, 5, 10)) -> dict:
    """P(max favorable excursion reaches kR) from the excursion pass."""
    if not trades:
        return {f"reach_{k}r": float("nan") for k in levels}
    mfe = np.array([t["mfe_r"] for t in trades], dtype=float)
    return {f"reach_{k}r": float(np.mean(mfe >= k)) for k in levels}


def split_trades_by_year(trades: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for t in trades:
        yr = pd.Timestamp(t["entry_time"]).year
        out.setdefault(yr, []).append(t)
    return out
