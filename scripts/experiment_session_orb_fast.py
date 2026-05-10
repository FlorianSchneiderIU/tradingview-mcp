from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import parse_utc_datetime  # noqa: E402
from scripts.experiment_session_orb import (  # noqa: E402
    SESSION_SPECS,
    OrbConfig,
    add_htf_context,
    build_grid,
    load_cached_symbol,
    metrics,
    parse_thresholds,
    train_ml_ranker,
)


@dataclass(frozen=True)
class SessionContext:
    session_date: pd.Timestamp
    session: str
    or_minutes: int
    session_id: str
    or_start_idx: int
    or_end_idx: int
    trade_start_idx: int
    trade_end_idx: int
    or_high: float
    or_low: float
    or_width_atr: float
    first_break_idx: int
    first_break_side: int


@dataclass
class Arrays:
    open_time: np.ndarray
    close_time: np.ndarray
    minute: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    atr: np.ndarray
    vol_sma20: np.ndarray
    ema20: np.ndarray
    ema200: np.ndarray
    daily_vwap: np.ndarray
    ret_1h: np.ndarray
    ret_4h: np.ndarray
    ret_24h: np.ndarray
    h1_trend: np.ndarray
    h4_trend: np.ndarray
    d1_trend: np.ndarray
    prev_day_high: np.ndarray
    prev_day_low: np.ndarray
    bull_fvg_mid: np.ndarray
    bear_fvg_mid: np.ndarray
    bull_fvg_width_atr: np.ndarray
    bear_fvg_width_atr: np.ndarray
    bull_fvg_age: np.ndarray
    bear_fvg_age: np.ndarray
    bull_fvg_bottom: np.ndarray
    bull_fvg_top: np.ndarray
    bear_fvg_bottom: np.ndarray
    bear_fvg_top: np.ndarray


def nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def to_arrays(df: pd.DataFrame) -> Arrays:
    n = len(df)
    bull_bottom = nan_array(n)
    bull_top = nan_array(n)
    bear_bottom = nan_array(n)
    bear_top = nan_array(n)
    highs = df["high"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()
    for i in range(2, n):
        if lows[i] > highs[i - 2]:
            bull_bottom[i] = highs[i - 2]
            bull_top[i] = lows[i]
        if highs[i] < lows[i - 2]:
            bear_bottom[i] = highs[i]
            bear_top[i] = lows[i - 2]
    return Arrays(
        open_time=df["open_time"].to_numpy(),
        close_time=df["close_time"].to_numpy(),
        minute=df["minute_of_day"].astype(int).to_numpy(),
        open=df["open"].astype(float).to_numpy(),
        high=highs,
        low=lows,
        close=df["close"].astype(float).to_numpy(),
        volume=df["volume"].astype(float).to_numpy(),
        atr=df["atr"].bfill().ffill().astype(float).to_numpy(),
        vol_sma20=df["vol_sma20"].bfill().ffill().astype(float).to_numpy(),
        ema20=df["ema20"].bfill().ffill().astype(float).to_numpy(),
        ema200=df["ema200"].bfill().ffill().astype(float).to_numpy(),
        daily_vwap=df["daily_vwap"].bfill().ffill().astype(float).to_numpy(),
        ret_1h=df["ret_1h"].fillna(0.0).astype(float).to_numpy(),
        ret_4h=df["ret_4h"].fillna(0.0).astype(float).to_numpy(),
        ret_24h=df["ret_24h"].fillna(0.0).astype(float).to_numpy(),
        h1_trend=df.get("h1_close_vs_ema50_pct", pd.Series(np.nan, index=df.index)).fillna(0.0).astype(float).to_numpy(),
        h4_trend=df.get("h4_close_vs_ema50_pct", pd.Series(np.nan, index=df.index)).fillna(0.0).astype(float).to_numpy(),
        d1_trend=df.get("d1_close_vs_ema50_pct", pd.Series(np.nan, index=df.index)).fillna(0.0).astype(float).to_numpy(),
        prev_day_high=df["prev_day_high"].astype(float).to_numpy(),
        prev_day_low=df["prev_day_low"].astype(float).to_numpy(),
        bull_fvg_mid=df["bull_fvg_mid"].astype(float).to_numpy(),
        bear_fvg_mid=df["bear_fvg_mid"].astype(float).to_numpy(),
        bull_fvg_width_atr=df["bull_fvg_width_atr"].astype(float).to_numpy(),
        bear_fvg_width_atr=df["bear_fvg_width_atr"].astype(float).to_numpy(),
        bull_fvg_age=df["bull_fvg_age"].astype(float).to_numpy(),
        bear_fvg_age=df["bear_fvg_age"].astype(float).to_numpy(),
        bull_fvg_bottom=bull_bottom,
        bull_fvg_top=bull_top,
        bear_fvg_bottom=bear_bottom,
        bear_fvg_top=bear_top,
    )


def first_true(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def build_contexts(df: pd.DataFrame, arrays: Arrays, *, sessions: list[str], or_minutes: list[int]) -> list[SessionContext]:
    contexts: list[SessionContext] = []
    indices_by_day = df.groupby("session_date", sort=True).indices
    for session_date, raw_indices in indices_by_day.items():
        idxs = np.asarray(raw_indices, dtype=int)
        mins = arrays.minute[idxs]
        for session in sessions:
            spec = SESSION_SPECS[session]
            start = spec["start_min"]
            end = spec["end_min"]
            for or_min in or_minutes:
                or_end = start + or_min
                or_idxs = idxs[(mins >= start) & (mins < or_end)]
                trade_idxs = idxs[(mins >= or_end) & (mins < end)]
                if len(or_idxs) < max(2, or_min // 5 - 1) or len(trade_idxs) < 2:
                    continue
                or_high = float(np.max(arrays.high[or_idxs]))
                or_low = float(np.min(arrays.low[or_idxs]))
                atr = float(arrays.atr[or_idxs[-1]])
                if not math.isfinite(atr) or atr <= 0:
                    continue
                up = arrays.high[trade_idxs] > or_high
                down = arrays.low[trade_idxs] < or_low
                up_first = first_true(up)
                down_first = first_true(down)
                first_idx = -1
                first_side = 0
                if up_first is not None and (down_first is None or up_first <= down_first):
                    first_idx = int(trade_idxs[up_first])
                    first_side = 1
                    if down_first is not None and down_first == up_first:
                        first_side = 1 if arrays.close[first_idx] >= arrays.open[first_idx] else -1
                elif down_first is not None:
                    first_idx = int(trade_idxs[down_first])
                    first_side = -1
                contexts.append(
                    SessionContext(
                        session_date=pd.Timestamp(session_date),
                        session=session,
                        or_minutes=or_min,
                        session_id=f"{pd.Timestamp(session_date).date()}_{session}",
                        or_start_idx=int(or_idxs[0]),
                        or_end_idx=int(or_idxs[-1]),
                        trade_start_idx=int(trade_idxs[0]),
                        trade_end_idx=int(trade_idxs[-1]),
                        or_high=or_high,
                        or_low=or_low,
                        or_width_atr=(or_high - or_low) / atr,
                        first_break_idx=first_idx,
                        first_break_side=first_side,
                    )
                )
    return contexts


def simulate_exit(
    a: Arrays,
    *,
    entry_idx: int,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    max_hold_bars: int,
    session_end_idx: int,
    fee_bps_per_side: float,
) -> dict[str, Any] | None:
    risk = entry - stop if direction == "long" else stop - entry
    if risk <= 0 or not math.isfinite(risk):
        return None
    end_idx = min(len(a.close) - 1, session_end_idx, entry_idx + max_hold_bars)
    exit_idx = end_idx
    exit_price = float(a.close[end_idx])
    exit_reason = "time"
    for i in range(entry_idx, end_idx + 1):
        if direction == "long":
            stop_hit = a.low[i] <= stop
            target_hit = a.high[i] >= target
        else:
            stop_hit = a.high[i] >= stop
            target_hit = a.low[i] <= target
        if stop_hit and target_hit:
            exit_idx, exit_price, exit_reason = i, stop, "stop_and_target_same_bar"
            break
        if stop_hit:
            exit_idx, exit_price, exit_reason = i, stop, "stop"
            break
        if target_hit:
            exit_idx, exit_price, exit_reason = i, target, "target"
            break
    gross = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
    fee_r = ((entry + exit_price) * fee_bps_per_side / 10000.0) / risk
    return {
        "exit_index": exit_idx,
        "exit_time": a.close_time[exit_idx],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "r_multiple_gross": gross,
        "r_multiple": gross - fee_r,
    }


def stop_target(
    a: Arrays,
    ctx: SessionContext,
    cfg: OrbConfig,
    *,
    direction: str,
    signal_idx: int,
    entry: float,
    sweep_low: float | None = None,
    sweep_high: float | None = None,
) -> tuple[float, float] | None:
    atr = float(a.atr[signal_idx])
    buffer = cfg.stop_buffer_atr * atr
    or_mid = (ctx.or_high + ctx.or_low) / 2.0
    if direction == "long":
        if cfg.stop_mode == "or_mid":
            stop = or_mid - buffer
        elif cfg.stop_mode == "or_opposite":
            stop = ctx.or_low - buffer
        elif cfg.stop_mode == "sweep":
            stop = (sweep_low if sweep_low is not None else a.low[signal_idx]) - buffer
        else:
            stop = a.low[signal_idx] - buffer
        risk = entry - stop
        target = entry + cfg.rr * risk
    else:
        if cfg.stop_mode == "or_mid":
            stop = or_mid + buffer
        elif cfg.stop_mode == "or_opposite":
            stop = ctx.or_high + buffer
        elif cfg.stop_mode == "sweep":
            stop = (sweep_high if sweep_high is not None else a.high[signal_idx]) + buffer
        else:
            stop = a.high[signal_idx] + buffer
        risk = stop - entry
        target = entry - cfg.rr * risk
    if risk <= 0 or risk / entry > 0.08:
        return None
    return stop, target


def make_row(
    a: Arrays,
    *,
    symbol: str,
    cfg: OrbConfig,
    ctx: SessionContext,
    signal_idx: int,
    entry_idx: int,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    sweep_side: int,
    sweep_depth_atr: float,
    fee_bps_per_side: float,
) -> dict[str, Any] | None:
    outcome = simulate_exit(
        a,
        entry_idx=entry_idx,
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        max_hold_bars=cfg.max_hold_bars,
        session_end_idx=ctx.trade_end_idx,
        fee_bps_per_side=fee_bps_per_side,
    )
    if outcome is None:
        return None
    atr = float(a.atr[signal_idx])
    sign = 1.0 if direction == "long" else -1.0
    or_width = ctx.or_high - ctx.or_low
    or_mid = (ctx.or_high + ctx.or_low) / 2.0
    signal_range = a.high[signal_idx] - a.low[signal_idx]
    signal_body = a.close[signal_idx] - a.open[signal_idx]
    same_fvg_mid = a.bull_fvg_mid[signal_idx] if direction == "long" else a.bear_fvg_mid[signal_idx]
    opp_fvg_mid = a.bear_fvg_mid[signal_idx] if direction == "long" else a.bull_fvg_mid[signal_idx]
    same_fvg_width = a.bull_fvg_width_atr[signal_idx] if direction == "long" else a.bear_fvg_width_atr[signal_idx]
    same_fvg_age = a.bull_fvg_age[signal_idx] if direction == "long" else a.bear_fvg_age[signal_idx]
    prev_same = a.prev_day_low[signal_idx] if direction == "long" else a.prev_day_high[signal_idx]
    prev_opp = a.prev_day_high[signal_idx] if direction == "long" else a.prev_day_low[signal_idx]
    return {
        "symbol": symbol.upper(),
        "variant": cfg.variant,
        "family": cfg.family,
        "session": cfg.session,
        "session_date": ctx.session_date,
        "session_id": ctx.session_id,
        "direction": direction,
        "direction_long": 1.0 if direction == "long" else 0.0,
        "signal_index": signal_idx,
        "entry_index": entry_idx,
        "signal_time": a.close_time[signal_idx],
        "entry_time": a.open_time[entry_idx],
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "win_label": 1 if outcome["r_multiple"] > 0 else 0,
        "or_minutes": float(cfg.or_minutes),
        "rr": float(cfg.rr),
        "max_hold_bars": float(cfg.max_hold_bars),
        "min_break_atr": float(cfg.min_break_atr),
        "retest_tolerance_atr": float(cfg.retest_tolerance_atr),
        "min_sweep_atr": float(cfg.min_sweep_atr),
        "or_width_atr": ctx.or_width_atr,
        "or_width_pct": (or_width / a.close[signal_idx]) * 100.0,
        "or_close_pos": (a.close[signal_idx] - ctx.or_low) / or_width if or_width > 0 else math.nan,
        "entry_vs_or_mid_atr": sign * (entry - or_mid) / atr,
        "entry_risk_atr": abs(entry - stop) / atr,
        "target_distance_atr": abs(target - entry) / atr,
        "minutes_after_or": float((signal_idx - ctx.or_end_idx + 1) * 5),
        "first_break_side_aligned": 1.0 if (ctx.first_break_side > 0 and direction == "long") or (ctx.first_break_side < 0 and direction == "short") else 0.0,
        "sweep_side_aligned": 1.0 if (sweep_side < 0 and direction == "long") or (sweep_side > 0 and direction == "short") else 0.0,
        "sweep_depth_atr": sweep_depth_atr,
        "signal_range_atr": signal_range / atr if atr > 0 else math.nan,
        "signal_body_atr_dir": sign * signal_body / atr if atr > 0 else math.nan,
        "signal_vol_mult": a.volume[signal_idx] / a.vol_sma20[signal_idx] if a.vol_sma20[signal_idx] > 0 else math.nan,
        "ret_1h_dir": sign * a.ret_1h[signal_idx],
        "ret_4h_dir": sign * a.ret_4h[signal_idx],
        "ret_24h_dir": sign * a.ret_24h[signal_idx],
        "h1_trend_aligned": sign * a.h1_trend[signal_idx],
        "h4_trend_aligned": sign * a.h4_trend[signal_idx],
        "d1_trend_aligned": sign * a.d1_trend[signal_idx],
        "close_vs_ema20_atr_dir": sign * (a.close[signal_idx] - a.ema20[signal_idx]) / atr,
        "close_vs_ema200_atr_dir": sign * (a.close[signal_idx] - a.ema200[signal_idx]) / atr,
        "close_vs_daily_vwap_atr_dir": sign * (a.close[signal_idx] - a.daily_vwap[signal_idx]) / atr,
        "prev_day_same_gap_atr": abs(prev_same - entry) / atr if math.isfinite(prev_same) else math.nan,
        "prev_day_opp_gap_atr": abs(prev_opp - entry) / atr if math.isfinite(prev_opp) else math.nan,
        "same_side_fvg_dist_atr": sign * (a.close[signal_idx] - same_fvg_mid) / atr if math.isfinite(same_fvg_mid) else math.nan,
        "opp_side_fvg_dist_atr": -sign * (a.close[signal_idx] - opp_fvg_mid) / atr if math.isfinite(opp_fvg_mid) else math.nan,
        "same_side_fvg_width_atr": same_fvg_width,
        "same_side_fvg_age": same_fvg_age,
        "session_asia": 1.0 if cfg.session == "asia" else 0.0,
        "session_london": 1.0 if cfg.session == "london" else 0.0,
        "session_ny": 1.0 if cfg.session == "ny" else 0.0,
        "family_breakout": 1.0 if cfg.family == "breakout" else 0.0,
        "family_retest": 1.0 if cfg.family == "retest" else 0.0,
        "family_judas": 1.0 if cfg.family == "judas" else 0.0,
        "mode_fvg_retest": 1.0 if cfg.entry_mode == "fvg_retest" else 0.0,
        "mode_level_retest": 1.0 if cfg.entry_mode == "level_retest" else 0.0,
        **outcome,
    }


def trade_for_context(a: Arrays, *, symbol: str, cfg: OrbConfig, ctx: SessionContext, fee_bps_per_side: float) -> dict[str, Any] | None:
    if ctx.or_width_atr < cfg.min_or_width_atr or ctx.or_width_atr > cfg.max_or_width_atr:
        return None
    lo = ctx.trade_start_idx
    hi = ctx.trade_end_idx
    idxs = np.arange(lo, hi + 1)

    if cfg.family == "breakout":
        long_mask = a.close[idxs] > ctx.or_high + cfg.min_break_atr * a.atr[idxs]
        short_mask = a.close[idxs] < ctx.or_low - cfg.min_break_atr * a.atr[idxs]
        long_first = first_true(long_mask)
        short_first = first_true(short_mask)
        if long_first is None and short_first is None:
            return None
        if long_first is not None and (short_first is None or long_first <= short_first):
            signal_idx = int(idxs[long_first])
            direction = "long"
        else:
            signal_idx = int(idxs[short_first])
            direction = "short"
        entry_idx = signal_idx + 1
        if entry_idx >= len(a.close):
            return None
        entry = float(a.open[entry_idx])
        prices = stop_target(a, ctx, cfg, direction=direction, signal_idx=signal_idx, entry=entry)
        if prices is None:
            return None
        stop, target = prices
        return make_row(a, symbol=symbol, cfg=cfg, ctx=ctx, signal_idx=signal_idx, entry_idx=entry_idx, direction=direction, entry=entry, stop=stop, target=target, sweep_side=0, sweep_depth_atr=0.0, fee_bps_per_side=fee_bps_per_side)

    if cfg.family == "retest":
        if ctx.first_break_idx < 0 or ctx.first_break_side == 0:
            return None
        direction = "long" if ctx.first_break_side > 0 else "short"
        boundary = ctx.or_high if direction == "long" else ctx.or_low
        start = ctx.first_break_idx + 1
        end = min(ctx.trade_end_idx, ctx.first_break_idx + cfg.retest_wait_bars)
        for signal_idx in range(start, end + 1):
            if direction == "long":
                accepted = a.low[signal_idx] <= boundary + cfg.retest_tolerance_atr * a.atr[signal_idx] and a.close[signal_idx] > boundary
            else:
                accepted = a.high[signal_idx] >= boundary - cfg.retest_tolerance_atr * a.atr[signal_idx] and a.close[signal_idx] < boundary
            if not accepted:
                continue
            entry_idx = signal_idx + 1
            if entry_idx >= len(a.close):
                return None
            entry = float(a.open[entry_idx])
            prices = stop_target(a, ctx, cfg, direction=direction, signal_idx=signal_idx, entry=entry)
            if prices is None:
                return None
            stop, target = prices
            return make_row(a, symbol=symbol, cfg=cfg, ctx=ctx, signal_idx=signal_idx, entry_idx=entry_idx, direction=direction, entry=entry, stop=stop, target=target, sweep_side=0, sweep_depth_atr=0.0, fee_bps_per_side=fee_bps_per_side)
        return None

    if cfg.family == "judas":
        long_mask = (a.low[idxs] < ctx.or_low - cfg.min_sweep_atr * a.atr[idxs]) & (a.close[idxs] > ctx.or_low)
        short_mask = (a.high[idxs] > ctx.or_high + cfg.min_sweep_atr * a.atr[idxs]) & (a.close[idxs] < ctx.or_high)
        long_first = first_true(long_mask)
        short_first = first_true(short_mask)
        if long_first is None and short_first is None:
            return None
        if long_first is not None and (short_first is None or long_first <= short_first):
            sweep_idx = int(idxs[long_first])
            direction = "long"
            sweep_side = -1
            sweep_depth = (ctx.or_low - a.low[sweep_idx]) / a.atr[sweep_idx]
        else:
            sweep_idx = int(idxs[short_first])
            direction = "short"
            sweep_side = 1
            sweep_depth = (a.high[sweep_idx] - ctx.or_high) / a.atr[sweep_idx]

        signal_idx = sweep_idx
        if cfg.entry_mode == "level_retest":
            boundary = ctx.or_low if direction == "long" else ctx.or_high
            found = False
            for j in range(sweep_idx + 1, min(ctx.trade_end_idx, sweep_idx + cfg.retest_wait_bars) + 1):
                if direction == "long":
                    accepted = a.low[j] <= boundary + cfg.retest_tolerance_atr * a.atr[j] and a.close[j] > boundary
                else:
                    accepted = a.high[j] >= boundary - cfg.retest_tolerance_atr * a.atr[j] and a.close[j] < boundary
                if accepted:
                    signal_idx = j
                    found = True
                    break
            if not found:
                return None
        elif cfg.entry_mode == "fvg_retest":
            found = False
            for j in range(sweep_idx + 1, min(ctx.trade_end_idx, sweep_idx + cfg.retest_wait_bars) + 1):
                if direction == "long" and math.isfinite(a.bull_fvg_top[j]):
                    gap_mid = (a.bull_fvg_top[j] + a.bull_fvg_bottom[j]) / 2.0
                    for k in range(j + 1, min(ctx.trade_end_idx, j + cfg.retest_wait_bars) + 1):
                        if a.low[k] <= a.bull_fvg_top[j] and a.close[k] >= gap_mid:
                            signal_idx = k
                            found = True
                            break
                elif direction == "short" and math.isfinite(a.bear_fvg_bottom[j]):
                    gap_mid = (a.bear_fvg_top[j] + a.bear_fvg_bottom[j]) / 2.0
                    for k in range(j + 1, min(ctx.trade_end_idx, j + cfg.retest_wait_bars) + 1):
                        if a.high[k] >= a.bear_fvg_bottom[j] and a.close[k] <= gap_mid:
                            signal_idx = k
                            found = True
                            break
                if found:
                    break
            if not found:
                return None

        entry_idx = signal_idx + 1
        if entry_idx >= len(a.close):
            return None
        entry = float(a.open[entry_idx])
        prices = stop_target(
            a,
            ctx,
            cfg,
            direction=direction,
            signal_idx=signal_idx,
            entry=entry,
            sweep_low=float(a.low[sweep_idx]),
            sweep_high=float(a.high[sweep_idx]),
        )
        if prices is None:
            return None
        stop, target = prices
        return make_row(a, symbol=symbol, cfg=cfg, ctx=ctx, signal_idx=signal_idx, entry_idx=entry_idx, direction=direction, entry=entry, stop=stop, target=target, sweep_side=sweep_side, sweep_depth_atr=sweep_depth, fee_bps_per_side=fee_bps_per_side)
    return None


def generate_trades(a: Arrays, *, symbol: str, cfg: OrbConfig, contexts: list[SessionContext], fee_bps_per_side: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ctx in contexts:
        if ctx.session != cfg.session or ctx.or_minutes != cfg.or_minutes:
            continue
        row = trade_for_context(a, symbol=symbol, cfg=cfg, ctx=ctx, fee_bps_per_side=fee_bps_per_side)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def score_configs(a: Arrays, *, symbol: str, configs: list[OrbConfig], contexts: list[SessionContext], split: pd.Timestamp, fee_bps_per_side: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pos, cfg in enumerate(configs, 1):
        trades = generate_trades(a, symbol=symbol, cfg=cfg, contexts=contexts, fee_bps_per_side=fee_bps_per_side)
        if trades.empty:
            train = trades
            oos = trades
        else:
            trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
            train = trades[trades["entry_time"] < split]
            oos = trades[trades["entry_time"] >= split]
        rows.append({**asdict(cfg), "variant": cfg.variant, **{f"train_{k}": v for k, v in metrics(train).items()}, **{f"oos_{k}": v for k, v in metrics(oos).items()}})
        if pos % 25 == 0:
            print(f"  scored {pos}/{len(configs)} configs", flush=True)
    out = pd.DataFrame(rows)
    pf = out["train_profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    out["train_score"] = out["train_net_r"].astype(float) + 8.0 * np.log1p(pf) + 0.25 * out["train_max_dd_r"].astype(float)
    return out


def apply_candidate_filter(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if frame.empty or name == "none":
        return frame
    out = frame.copy()
    mask = pd.Series(True, index=out.index)
    if name in {"judas_fvg_risk2", "judas_fvg_risk25", "asia_ny_judas_fvg_risk25"}:
        mask &= out["family"].eq("judas")
        mask &= out["mode_fvg_retest"].astype(float).eq(1.0)
    if name == "judas_fvg_risk2":
        mask &= out["entry_risk_atr"].astype(float) >= 2.0
    elif name == "judas_fvg_risk25":
        mask &= out["entry_risk_atr"].astype(float) >= 2.5
    elif name == "asia_ny_judas_fvg_risk25":
        mask &= out["entry_risk_atr"].astype(float) >= 2.5
        mask &= out["session"].isin(["asia", "ny"])
    else:
        raise ValueError(f"Unknown --ml-candidate-filter {name}")
    return out[mask].copy()


def select_ranked_trades(scored: pd.DataFrame, *, threshold: float, split: pd.Timestamp) -> pd.DataFrame:
    if scored.empty or "ml_prob" not in scored.columns:
        return pd.DataFrame(columns=scored.columns)
    selected_parts = []
    eligible = scored[scored["ml_prob"] >= threshold].copy()
    for _, group in eligible.groupby("session_id", sort=True):
        selected_parts.append(group.sort_values("ml_prob", ascending=False).head(1))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=scored.columns)
    selected["sample"] = np.where(pd.to_datetime(selected["entry_time"], utc=True) < split, "train", "oos")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast session ORB/Judas/FVG research pass.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/bybit_linear"))
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--train-start", default="2022-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--sessions", default="asia,london,ny")
    parser.add_argument("--or-minutes", default="30,60,90")
    parser.add_argument("--grid-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--top-train-variants", type=int, default=24)
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument(
        "--ml-candidate-filter",
        choices=["none", "judas_fvg_risk2", "judas_fvg_risk25", "asia_ny_judas_fvg_risk25"],
        default="none",
    )
    parser.add_argument("--selected-threshold", type=float, default=0.50)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/session_orb_btc_fast"))
    args = parser.parse_args()

    symbol = args.symbol.upper()
    split = pd.Timestamp(parse_utc_datetime(args.split))
    train_start = pd.Timestamp(parse_utc_datetime(args.train_start))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    sessions = [x.strip() for x in args.sessions.split(",") if x.strip()]
    or_minutes = [int(x.strip()) for x in args.or_minutes.split(",") if x.strip()]

    print(f"{symbol}: loading cached 5m data", flush=True)
    raw = load_cached_symbol(symbol, args.cache_dir)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    raw["close_time"] = pd.to_datetime(raw["close_time"], utc=True)
    raw = raw[(raw["open_time"] >= train_start - pd.Timedelta(days=90)) & (raw["open_time"] < end)].copy()
    print(f"{symbol}: preparing indicators on {len(raw):,} bars", flush=True)
    df = add_htf_context(raw)
    df = df[(df["open_time"] >= train_start) & (df["open_time"] < end)].reset_index(drop=True)
    a = to_arrays(df)
    contexts = build_contexts(df, a, sessions=sessions, or_minutes=or_minutes)
    configs = build_grid(args)
    print(f"{symbol}: {len(contexts):,} session contexts, {len(configs)} configs", flush=True)
    summary = score_configs(a, symbol=symbol, configs=configs, contexts=contexts, split=split, fee_bps_per_side=args.fee_bps_per_side)

    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary.sort_values(["train_score", "train_profit_factor", "train_net_r"], ascending=[False, False, False]).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}", flush=True)
    eligible = summary[(summary["train_trades"] >= 80) & (summary["oos_trades"] >= 15)].copy()
    if eligible.empty:
        eligible = summary[summary["train_trades"] >= 40].copy()
    top = eligible.sort_values(["train_score", "train_profit_factor", "train_net_r"], ascending=[False, False, False]).head(args.top_train_variants)
    cols = ["variant", "train_trades", "train_profit_factor", "train_net_r", "train_max_dd_r", "oos_trades", "oos_profit_factor", "oos_net_r", "oos_max_dd_r"]
    print("\nTop train-selected variants with OOS validation:", flush=True)
    print(top[cols].head(12).to_string(index=False), flush=True)

    variant_to_cfg = {cfg.variant: cfg for cfg in configs}
    top_frames = [
        generate_trades(a, symbol=symbol, cfg=variant_to_cfg[v], contexts=contexts, fee_bps_per_side=args.fee_bps_per_side)
        for v in top["variant"].tolist()
    ]
    top_trades = pd.concat([f for f in top_frames if not f.empty], ignore_index=True) if top_frames else pd.DataFrame()
    trades_path = args.output_prefix.with_name(args.output_prefix.name + "_top_variant_trades.csv")
    top_trades.to_csv(trades_path, index=False)
    print(f"Wrote {trades_path}", flush=True)

    ml_candidates = apply_candidate_filter(top_trades, args.ml_candidate_filter)
    candidates_path = args.output_prefix.with_name(args.output_prefix.name + "_ml_candidates.csv")
    ml_candidates.to_csv(candidates_path, index=False)
    print(f"Wrote {candidates_path} ({len(ml_candidates)} rows after filter={args.ml_candidate_filter})", flush=True)

    ml_table, scored, ml_cols = train_ml_ranker(ml_candidates, split=split, thresholds=parse_thresholds(args.thresholds))
    ml_path = args.output_prefix.with_name(args.output_prefix.name + "_ml_thresholds.csv")
    scored_path = args.output_prefix.with_name(args.output_prefix.name + "_ml_scored_candidates.csv")
    selected_path = args.output_prefix.with_name(args.output_prefix.name + f"_ml_selected_t{args.selected_threshold:.2f}.csv")
    ml_table.to_csv(ml_path, index=False)
    scored.to_csv(scored_path, index=False)
    selected = select_ranked_trades(scored, threshold=args.selected_threshold, split=split)
    selected.to_csv(selected_path, index=False)
    print(f"Wrote {ml_path}", flush=True)
    print(f"Wrote {scored_path}", flush=True)
    print(f"Wrote {selected_path}", flush=True)
    if not ml_table.empty:
        print("\nML ranker threshold table:", flush=True)
        print(ml_table.to_string(index=False), flush=True)
        train_choices = ml_table[ml_table["train_trades"] >= 80].copy()
        if train_choices.empty:
            train_choices = ml_table.copy()
        chosen = train_choices.sort_values(["train_profit_factor", "train_net_r", "train_trades"], ascending=[False, False, False]).iloc[0]
        print("\nTrain-chosen ML threshold:", flush=True)
        print(chosen.to_string(), flush=True)
        print(f"\nML feature count: {len(ml_cols)}", flush=True)


if __name__ == "__main__":
    main()
