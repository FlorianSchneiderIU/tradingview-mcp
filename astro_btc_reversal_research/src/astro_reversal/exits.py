"""Scaled-exit simulator (partial take-profits) for high-RR trades.

Models the discretionary management described for the 30R approach: take 25% at ~4R,
50% at ~10-15R, the last 25% at the full target, and move the stop to breakeven after
the first partial. This trades peak-R on the bulk of the position for a much higher
hit rate and a far smaller left tail.

Returns the blended R per trade (net of round-trip costs applied per fill), so it
drops into the same aggregation as ``simulate_sweep_trade``.
"""

from __future__ import annotations

import math

import pandas as pd


def simulate_scaled_trade(
    frame: pd.DataFrame,
    signal_idx: int,
    direction: str,
    tps: tuple[float, ...] = (4.0, 12.0, 30.0),
    fracs: tuple[float, ...] = (0.25, 0.50, 0.25),
    max_hold_bars: int = 1344,
    stop_buffer_atr: float = 0.05,
    cost_bps_round_trip: float = 11.0,
    move_stop_be_after: int = 1,
    maker_bps: float | None = None,
    taker_bps: float | None = None,
) -> dict | None:
    """Enter next bar; scale out at the R-targets; stop -> breakeven after partial #N.

    Realistic fees: entry is taker (market in on confirmation), TP fills are maker
    (resting limits), stop/timeout is taker. Pass maker_bps/taker_bps (per side) to use
    the split; otherwise both default to cost_bps_round_trip/2 (all-taker, backward compat).
    """
    if maker_bps is None:
        maker_bps = cost_bps_round_trip / 2.0
    if taker_bps is None:
        taker_bps = cost_bps_round_trip / 2.0
    n = len(frame)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None
    sig = frame.iloc[signal_idx]
    entry = float(frame["open"].iloc[entry_idx])
    atr = float(sig["atr"])
    if not math.isfinite(atr) or atr <= 0:
        return None
    if direction == "long":
        stop = float(sig["low"]) - stop_buffer_atr * atr
        risk = entry - stop
        targets = [entry + r * risk for r in tps]
    else:
        stop = float(sig["high"]) + stop_buffer_atr * atr
        risk = stop - entry
        targets = [entry - r * risk for r in tps]
    if not math.isfinite(risk) or risk <= 0:
        return None

    maker_r = (maker_bps / 10_000.0) * entry / risk     # resting-limit TP fills
    taker_r = (taker_bps / 10_000.0) * entry / risk     # market entry / stop / timeout
    end = min(n - 1, entry_idx + max_hold_bars)
    remaining = 1.0
    realized = -taker_r  # entry cost on the full position (taker)
    cur_stop = stop
    filled = 0
    be_moved = False
    exit_reason = "timeout"
    exit_idx = end
    mfe_r = 0.0

    for c in range(entry_idx, end + 1):
        hi = float(frame["high"].iloc[c])
        lo = float(frame["low"].iloc[c])
        fav = (hi - entry) / risk if direction == "long" else (entry - lo) / risk
        mfe_r = max(mfe_r, fav)
        stop_hit = (lo <= cur_stop) if direction == "long" else (hi >= cur_stop)
        if stop_hit:  # conservative: stop checked before targets each bar
            r_at_stop = (cur_stop - entry) / risk if direction == "long" else (entry - cur_stop) / risk
            realized += remaining * r_at_stop - taker_r * remaining
            remaining = 0.0
            exit_reason = "stop" if not be_moved else "breakeven"
            exit_idx = c
            break
        while filled < len(tps):
            hit = (hi >= targets[filled]) if direction == "long" else (lo <= targets[filled])
            if not hit:
                break
            f = fracs[filled]
            realized += f * tps[filled] - maker_r * f
            remaining -= f
            filled += 1
            if filled >= move_stop_be_after and not be_moved:
                cur_stop = entry
                be_moved = True
            if remaining <= 1e-9:
                break
        if remaining <= 1e-9:
            exit_reason = "target"
            exit_idx = c
            break
    else:
        px = float(frame["close"].iloc[end])
        r_mark = (px - entry) / risk if direction == "long" else (entry - px) / risk
        realized += remaining * r_mark - taker_r * remaining

    return {
        "result_r": float(realized),
        "mfe_r": float(mfe_r),
        "tp_filled": int(filled),
        "exit_reason": exit_reason,
        "exit_idx": int(exit_idx),
        "entry_idx": int(entry_idx),
        "risk_pct": float(risk / entry),     # stop distance as a fraction of price
        "entry_time": pd.Timestamp(frame["open_time"].iloc[entry_idx]),
        "exit_time": pd.Timestamp(frame["close_time"].iloc[exit_idx]),
        "direction": direction,
    }


def backtest_fixed(
    frame: pd.DataFrame,
    entry_mask,
    direction: str,
    rr: float,
    max_hold_bars: int = 1344,
    stop_buffer_atr: float = 0.05,
    cost_bps_round_trip: float = 11.0,
) -> list[dict]:
    """Non-overlapping fixed-RR trades (long or short) via the reused sweep simulator."""
    from . import reuse
    trades: list[dict] = []
    cursor = 20
    n = len(frame)
    while cursor < n - 2:
        if not entry_mask[cursor]:
            cursor += 1
            continue
        tr = reuse.simulate_sweep_trade(frame, cursor, direction, rr, max_hold_bars,
                                        stop_buffer_atr, cost_bps_round_trip)
        if tr is None:
            cursor += 1
            continue
        trades.append(tr)
        cursor = max(cursor + 1, int(tr["exit_idx"]) + 1)
    return trades


def backtest_scaled(
    frame: pd.DataFrame,
    entry_mask,
    direction: str,
    tps=(4.0, 12.0, 30.0),
    fracs=(0.25, 0.50, 0.25),
    max_hold_bars: int = 1344,
    stop_buffer_atr: float = 0.05,
    cost_bps_round_trip: float = 11.0,
    maker_bps: float | None = None,
    taker_bps: float | None = None,
) -> list[dict]:
    """Non-overlapping scaled-exit trades where entry_mask is True."""
    trades: list[dict] = []
    cursor = 20
    n = len(frame)
    while cursor < n - 2:
        if not entry_mask[cursor]:
            cursor += 1
            continue
        tr = simulate_scaled_trade(frame, cursor, direction, tps, fracs, max_hold_bars,
                                   stop_buffer_atr, cost_bps_round_trip,
                                   maker_bps=maker_bps, taker_bps=taker_bps)
        if tr is None:
            cursor += 1
            continue
        trades.append(tr)
        cursor = max(cursor + 1, int(tr["exit_idx"]) + 1)
    return trades
