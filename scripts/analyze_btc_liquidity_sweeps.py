from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import add_indicators, ensure_ohlcv_frame, load_ohlcv_csv, resample_ohlc


@dataclass(frozen=True)
class Level:
    level_id: str
    side: str
    value: float
    pivot_idx: int
    confirm_idx: int
    pivot_time: pd.Timestamp
    confirm_time: pd.Timestamp


@dataclass(frozen=True)
class PathData:
    timeframe: str
    open_time_ns: np.ndarray
    close_time_ns: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray


TF_SETTINGS = {
    "15m": {"pivot_window": 6, "max_age": 384, "horizon": 96, "min_events": 25},
    "1h": {"pivot_window": 4, "max_age": 192, "horizon": 72, "min_events": 18},
    "4h": {"pivot_window": 3, "max_age": 96, "horizon": 48, "min_events": 10},
}

TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
IDEAL_TP_R_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
STOP_BUFFER_ATR_VALUES = [0.0, 0.05, 0.10, 0.15, 0.25, 0.35]
STOP_MODES = ["hard_5m", "close_5m", "close_15m", "close_1h"]
STOP_ANCHOR_MODES = ["wick", "half_wick_body", "body"]


def session_name(ts: pd.Timestamp) -> str:
    hour = int(pd.Timestamp(ts).hour)
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 10:
        return "london_open"
    if 10 <= hour < 13:
        return "london_late"
    if 13 <= hour < 16:
        return "ny_open"
    if 16 <= hour < 21:
        return "ny_late"
    return "late"


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(ensure_ohlcv_frame(frame), 14, 200, 14)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float)
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()
    out["ema200_slope_atr"] = (out["ema200"] - out["ema200"].shift(20)) / out["atr"].replace(0.0, np.nan)
    out["range_atr"] = (high - low) / out["atr"].replace(0.0, np.nan)
    out["body_atr"] = (close - open_).abs() / out["atr"].replace(0.0, np.nan)
    out["volume_ratio"] = volume / volume.rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    out["atr_ratio"] = out["atr"] / out["atr"].rolling(200, min_periods=50).median().replace(0.0, np.nan)
    out["pre_return_4_atr"] = (close - close.shift(4)) / out["atr"].replace(0.0, np.nan)
    out["pre_return_12_atr"] = (close - close.shift(12)) / out["atr"].replace(0.0, np.nan)
    out["pre_range_12_atr"] = (high.rolling(12, min_periods=6).max() - low.rolling(12, min_periods=6).min()) / out["atr"].replace(0.0, np.nan)
    out["dist_ema200_atr"] = (close - out["ema200"]) / out["atr"].replace(0.0, np.nan)
    return out


def pivot_levels(frame: pd.DataFrame, timeframe: str, window: int) -> dict[int, list[Level]]:
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    close_times = pd.to_datetime(frame["close_time"], utc=True)
    out: dict[int, list[Level]] = {}
    if len(frame) <= 2 * window + 1:
        return out
    window_size = 2 * window + 1
    high_windows = np.lib.stride_tricks.sliding_window_view(highs, window_size)
    low_windows = np.lib.stride_tricks.sliding_window_view(lows, window_size)
    centers = np.arange(window, len(frame) - window)
    high_centers = highs[centers]
    low_centers = lows[centers]
    is_high = (
        np.isfinite(high_centers)
        & (high_centers == np.nanmax(high_windows, axis=1))
        & (np.nanargmax(high_windows, axis=1) == window)
    )
    is_low = (
        np.isfinite(low_centers)
        & (low_centers == np.nanmin(low_windows, axis=1))
        & (np.nanargmin(low_windows, axis=1) == window)
    )
    for idx in centers[np.flatnonzero(is_high)]:
        idx = int(idx)
        confirm_idx = idx + window
        out.setdefault(confirm_idx, []).append(
            Level(
                level_id=f"{timeframe}|H|{idx}",
                side="resistance",
                value=float(highs[idx]),
                pivot_idx=idx,
                confirm_idx=confirm_idx,
                pivot_time=pd.Timestamp(close_times.iloc[idx]).tz_convert("UTC"),
                confirm_time=pd.Timestamp(close_times.iloc[confirm_idx]).tz_convert("UTC"),
            )
        )
    for idx in centers[np.flatnonzero(is_low)]:
        idx = int(idx)
        confirm_idx = idx + window
        out.setdefault(confirm_idx, []).append(
            Level(
                level_id=f"{timeframe}|L|{idx}",
                side="support",
                value=float(lows[idx]),
                pivot_idx=idx,
                confirm_idx=confirm_idx,
                pivot_time=pd.Timestamp(close_times.iloc[idx]).tz_convert("UTC"),
                confirm_time=pd.Timestamp(close_times.iloc[confirm_idx]).tz_convert("UTC"),
            )
        )
    return out


def forward_outcome(
    frame: pd.DataFrame,
    *,
    index: int,
    direction: str,
    stop_anchor: float,
    atr: float,
    horizon: int,
    stop_buffer_atr: float,
    target_rr: float,
) -> dict[str, Any] | None:
    entry_idx = index + 1
    if entry_idx >= len(frame) - 1:
        return None
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    open_times = pd.to_datetime(frame["open_time"], utc=True)
    close_times = pd.to_datetime(frame["close_time"], utc=True)

    entry = float(opens[entry_idx])
    stop = float(stop_anchor - stop_buffer_atr * atr if direction == "long" else stop_anchor + stop_buffer_atr * atr)
    risk = entry - stop if direction == "long" else stop - entry
    if not math.isfinite(risk) or risk <= 0.0:
        return None
    target = entry + target_rr * risk if direction == "long" else entry - target_rr * risk
    exit_idx = min(len(frame) - 1, entry_idx + horizon)
    hit_stop_idx: int | None = None
    hit_target_idx: int | None = None
    mfe_r = 0.0
    mae_r = 0.0
    for cursor in range(entry_idx, exit_idx + 1):
        if direction == "long":
            mfe_r = max(mfe_r, (highs[cursor] - entry) / risk)
            mae_r = max(mae_r, (entry - lows[cursor]) / risk)
            if hit_stop_idx is None and lows[cursor] <= stop:
                hit_stop_idx = cursor
            if hit_target_idx is None and highs[cursor] >= target:
                hit_target_idx = cursor
        else:
            mfe_r = max(mfe_r, (entry - lows[cursor]) / risk)
            mae_r = max(mae_r, (highs[cursor] - entry) / risk)
            if hit_stop_idx is None and highs[cursor] >= stop:
                hit_stop_idx = cursor
            if hit_target_idx is None and lows[cursor] <= target:
                hit_target_idx = cursor
        if hit_stop_idx is not None or hit_target_idx is not None:
            break

    target_before_stop = hit_target_idx is not None and (hit_stop_idx is None or hit_target_idx <= hit_stop_idx)
    stop_before_target = hit_stop_idx is not None and (hit_target_idx is None or hit_stop_idx < hit_target_idx)
    if target_before_stop:
        result_r = target_rr
        result = "target"
        outcome_idx = int(hit_target_idx)
    elif stop_before_target:
        result_r = -1.0
        result = "stop"
        outcome_idx = int(hit_stop_idx)
    else:
        exit_price = float(closes[exit_idx])
        result_r = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
        result = "timeout"
        outcome_idx = int(exit_idx)
    return {
        "entry_idx": int(entry_idx),
        "entry_time": pd.Timestamp(open_times.iloc[entry_idx]).tz_convert("UTC"),
        "entry_price": entry,
        "stop_price": stop,
        "risk_pct": risk / entry,
        "target_price": float(target),
        "target_before_stop": bool(target_before_stop),
        "stop_before_target": bool(stop_before_target),
        "outcome": result,
        "outcome_idx": outcome_idx,
        "outcome_time": pd.Timestamp(close_times.iloc[outcome_idx]).tz_convert("UTC"),
        "outcome_r": float(result_r),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "bars_to_outcome": int(outcome_idx - entry_idx),
    }


def detect_sweeps(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    pivot_window: int,
    max_age: int,
    horizon: int,
    min_reclaim_pos: float,
    min_sweep_depth_atr: float,
    stop_buffer_atr: float,
    target_rr: float,
    max_scan: int,
) -> pd.DataFrame:
    levels_by_confirm = pivot_levels(frame, timeframe, pivot_window)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    opens = frame["open"].to_numpy(dtype=float)
    atrs = frame["atr"].to_numpy(dtype=float)
    times = pd.to_datetime(frame["close_time"], utc=True)

    supports: list[Level] = []
    resistances: list[Level] = []
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for idx in range(len(frame) - 2):
        for level in levels_by_confirm.get(idx - 1, []):
            if level.side == "support":
                supports.append(level)
            else:
                resistances.append(level)
        if idx % 1000 == 0:
            supports = [level for level in supports if level.level_id not in used and 0 < idx - level.confirm_idx <= max_age]
            resistances = [level for level in resistances if level.level_id not in used and 0 < idx - level.confirm_idx <= max_age]
        atr = float(atrs[idx])
        if not math.isfinite(atr) or atr <= 0.0:
            continue

        def candidate_levels(levels: list[Level]) -> list[Level]:
            live = [level for level in levels if level.level_id not in used and 0 < idx - level.confirm_idx <= max_age]
            return live[-max_scan:]

        for level in candidate_levels(supports):
            if closes[idx] < level.value:
                used.add(level.level_id)
                continue
            if lows[idx] < level.value and closes[idx] > level.value:
                bar_range = max(float(highs[idx] - lows[idx]), 1e-12)
                reclaim_pos = float((closes[idx] - lows[idx]) / bar_range)
                sweep_depth = float((level.value - lows[idx]) / atr)
                if reclaim_pos < min_reclaim_pos or sweep_depth < min_sweep_depth_atr:
                    continue
                outcome = forward_outcome(
                    frame,
                    index=idx,
                    direction="long",
                    stop_anchor=float(lows[idx]),
                    atr=atr,
                    horizon=horizon,
                    stop_buffer_atr=stop_buffer_atr,
                    target_rr=target_rr,
                )
                if outcome is None:
                    continue
                rows.append(event_row(frame, timeframe, idx, level, "long", reclaim_pos, sweep_depth, outcome))
                used.add(level.level_id)
                break

        for level in candidate_levels(resistances):
            if closes[idx] > level.value:
                used.add(level.level_id)
                continue
            if highs[idx] > level.value and closes[idx] < level.value:
                bar_range = max(float(highs[idx] - lows[idx]), 1e-12)
                reclaim_pos = float((highs[idx] - closes[idx]) / bar_range)
                sweep_depth = float((highs[idx] - level.value) / atr)
                if reclaim_pos < min_reclaim_pos or sweep_depth < min_sweep_depth_atr:
                    continue
                outcome = forward_outcome(
                    frame,
                    index=idx,
                    direction="short",
                    stop_anchor=float(highs[idx]),
                    atr=atr,
                    horizon=horizon,
                    stop_buffer_atr=stop_buffer_atr,
                    target_rr=target_rr,
                )
                if outcome is None:
                    continue
                rows.append(event_row(frame, timeframe, idx, level, "short", reclaim_pos, sweep_depth, outcome))
                used.add(level.level_id)
                break
    return pd.DataFrame(rows)


def event_row(
    frame: pd.DataFrame,
    timeframe: str,
    idx: int,
    level: Level,
    direction: str,
    reclaim_pos: float,
    sweep_depth_atr: float,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    row = frame.iloc[idx]
    close_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
    sweep_open = float(row["open"])
    sweep_high = float(row["high"])
    sweep_low = float(row["low"])
    sweep_close = float(row["close"])
    sweep_body_edge = min(sweep_open, sweep_close) if direction == "long" else max(sweep_open, sweep_close)
    sweep_stop_anchor = sweep_low if direction == "long" else sweep_high
    sweep_atr = float(row["atr"])
    return {
        "timeframe": timeframe,
        "event_idx": int(idx),
        "event_time": close_time,
        "direction": direction,
        "session": session_name(close_time),
        "hour": int(close_time.hour),
        "weekday": int(close_time.weekday()),
        "level_id": level.level_id,
        "level_side": level.side,
        "level": float(level.value),
        "level_age_bars": int(idx - level.confirm_idx),
        "pivot_time": level.pivot_time,
        "confirm_time": level.confirm_time,
        "sweep_open": sweep_open,
        "sweep_high": sweep_high,
        "sweep_low": sweep_low,
        "sweep_close": sweep_close,
        "sweep_stop_anchor": sweep_stop_anchor,
        "sweep_body_edge": sweep_body_edge,
        "sweep_wick_to_body_atr": abs(sweep_body_edge - sweep_stop_anchor) / sweep_atr if sweep_atr > 0.0 else math.nan,
        "sweep_atr": sweep_atr,
        "sweep_depth_atr": float(sweep_depth_atr),
        "reclaim_pos": float(reclaim_pos),
        "sweep_range_atr": float(row["range_atr"]),
        "sweep_body_atr": float(row["body_atr"]),
        "volume_ratio": float(row["volume_ratio"]),
        "atr_ratio": float(row["atr_ratio"]),
        "rsi": float(row["rsi"]),
        "dist_ema200_atr": float(row["dist_ema200_atr"]),
        "ema200_slope_atr": float(row["ema200_slope_atr"]),
        "pre_return_4_atr": float(row["pre_return_4_atr"]),
        "pre_return_12_atr": float(row["pre_return_12_atr"]),
        "pre_range_12_atr": float(row["pre_range_12_atr"]),
        "trend_side": trend_side(float(row["close"]), float(row["ema200"]), float(row["ema200_slope_atr"])),
        **outcome,
    }


def trend_side(close: float, ema: float, slope: float) -> str:
    if not all(math.isfinite(value) for value in [close, ema, slope]):
        return "unknown"
    if close >= ema and slope >= 0.0:
        return "up"
    if close <= ema and slope <= 0.0:
        return "down"
    return "mixed"


def summarize_group(events: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    out = (
        events.groupby(columns, dropna=False)
        .agg(
            count=("target_before_stop", "size"),
            hit_rate=("target_before_stop", "mean"),
            avg_r=("outcome_r", "mean"),
            median_mfe_r=("mfe_r", "median"),
            median_mae_r=("mae_r", "median"),
            avg_sweep_depth=("sweep_depth_atr", "mean"),
            avg_reclaim_pos=("reclaim_pos", "mean"),
        )
        .reset_index()
    )
    return out[out["count"] >= min_count].sort_values(["hit_rate", "avg_r", "count"], ascending=[False, False, False])


def numeric_bucket_summary(events: pd.DataFrame, column: str, min_count: int) -> pd.DataFrame:
    values = pd.to_numeric(events[column], errors="coerce")
    valid = events[np.isfinite(values)].copy()
    if valid.empty or valid[column].nunique() < 4:
        return pd.DataFrame()
    valid[f"{column}_bucket"] = pd.qcut(valid[column], q=4, duplicates="drop")
    out = summarize_group(valid, [f"{column}_bucket"], min_count)
    out.insert(0, "feature", column)
    return out


def feature_contrast(events: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "sweep_depth_atr",
        "reclaim_pos",
        "sweep_range_atr",
        "sweep_body_atr",
        "volume_ratio",
        "atr_ratio",
        "rsi",
        "dist_ema200_atr",
        "ema200_slope_atr",
        "pre_return_4_atr",
        "pre_return_12_atr",
        "pre_range_12_atr",
        "level_age_bars",
        "risk_pct",
    ]
    rows: list[dict[str, Any]] = []
    good = events[events["target_before_stop"]].copy()
    bad = events[~events["target_before_stop"]].copy()
    for column in numeric:
        g = pd.to_numeric(good[column], errors="coerce")
        b = pd.to_numeric(bad[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "good_median": float(g.median()) if len(g) else math.nan,
                "bad_median": float(b.median()) if len(b) else math.nan,
                "delta": float(g.median() - b.median()) if len(g) and len(b) else math.nan,
                "good_mean": float(g.mean()) if len(g) else math.nan,
                "bad_mean": float(b.mean()) if len(b) else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("delta", key=lambda s: s.abs(), ascending=False)


def top_patterns(events: pd.DataFrame, min_count: int) -> pd.DataFrame:
    frame = events.copy()
    frame["deep_sweep"] = pd.to_numeric(frame["sweep_depth_atr"], errors="coerce") >= frame["sweep_depth_atr"].median()
    frame["strong_reclaim"] = pd.to_numeric(frame["reclaim_pos"], errors="coerce") >= frame["reclaim_pos"].median()
    frame["high_volume"] = pd.to_numeric(frame["volume_ratio"], errors="coerce") >= frame["volume_ratio"].median()
    frame["trend_extension"] = pd.to_numeric(frame["dist_ema200_atr"], errors="coerce").abs() >= frame["dist_ema200_atr"].abs().median()
    groups = [
        ["timeframe", "direction", "session"],
        ["timeframe", "direction", "trend_side"],
        ["timeframe", "direction", "deep_sweep", "strong_reclaim"],
        ["timeframe", "direction", "session", "trend_side"],
        ["timeframe", "direction", "high_volume", "trend_extension"],
    ]
    parts = [summarize_group(frame, group, min_count) for group in groups]
    out = pd.concat([part.assign(pattern=" + ".join([col for col in part.columns if col in set(sum(groups, []))])) for part in parts if not part.empty], ignore_index=True)
    return out.sort_values(["hit_rate", "avg_r", "count"], ascending=[False, False, False])


def add_candidate_flags(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = events.copy()
    out["candidate_4h_session_trend"] = False
    out["candidate_4h_session_trend_bounded"] = False
    if out.empty:
        return out, {"pre_range_12_atr_q75_4h": math.nan, "atr_ratio_q75_4h": math.nan}

    four_h = out["timeframe"] == "4h"
    four_h_events = out[four_h].copy()
    pre_range_q75 = float(pd.to_numeric(four_h_events["pre_range_12_atr"], errors="coerce").quantile(0.75)) if not four_h_events.empty else math.nan
    atr_ratio_q75 = float(pd.to_numeric(four_h_events["atr_ratio"], errors="coerce").quantile(0.75)) if not four_h_events.empty else math.nan

    long_ok = (
        (out["direction"] == "long")
        & out["session"].isin(["late", "ny_open", "london_late"])
        & (out["trend_side"] == "up")
    )
    short_ok = (
        (out["direction"] == "short")
        & out["session"].isin(["asia", "late", "ny_late"])
        & (out["trend_side"] == "down")
    )
    session_trend = four_h & (long_ok | short_ok)
    bounded = (
        session_trend
        & (pd.to_numeric(out["pre_range_12_atr"], errors="coerce") <= pre_range_q75)
        & (pd.to_numeric(out["atr_ratio"], errors="coerce") <= atr_ratio_q75)
    )
    out.loc[session_trend, "candidate_4h_session_trend"] = True
    out.loc[bounded, "candidate_4h_session_trend_bounded"] = True
    return out, {"pre_range_12_atr_q75_4h": pre_range_q75, "atr_ratio_q75_4h": atr_ratio_q75}


def ideal_excursion_before_stop(
    open_time_ns: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    entry_time: pd.Timestamp,
    timeframe: str,
    horizon: int,
    direction: str,
    entry: float,
    stop: float,
) -> dict[str, Any]:
    risk = entry - stop if direction == "long" else stop - entry
    if not math.isfinite(risk) or risk <= 0.0 or timeframe not in TF_MINUTES:
        return {
            "mfe_before_stop_r_5m": math.nan,
            "mfe_until_stop_bar_r_5m": math.nan,
            "mae_before_stop_r_5m": math.nan,
            "stop_hit_5m": False,
            "bars_5m_to_stop": math.nan,
            "hours_to_stop": math.nan,
            "horizon_close_r_5m": math.nan,
        }

    entry_time = pd.Timestamp(entry_time).tz_convert("UTC")
    start_idx = int(np.searchsorted(open_time_ns, entry_time.value, side="left"))
    if start_idx >= len(open_time_ns):
        return {
            "mfe_before_stop_r_5m": math.nan,
            "mfe_until_stop_bar_r_5m": math.nan,
            "mae_before_stop_r_5m": math.nan,
            "stop_hit_5m": False,
            "bars_5m_to_stop": math.nan,
            "hours_to_stop": math.nan,
            "horizon_close_r_5m": math.nan,
        }
    end_exclusive = entry_time + pd.Timedelta(minutes=TF_MINUTES[timeframe] * horizon)
    end_idx = int(np.searchsorted(open_time_ns, end_exclusive.value, side="left") - 1)
    end_idx = min(max(end_idx, start_idx), len(open_time_ns) - 1)

    mfe_before_stop = 0.0
    mfe_until_stop_bar = 0.0
    mae_before_stop = 0.0
    stop_idx: int | None = None
    last_idx = end_idx
    for cursor in range(start_idx, end_idx + 1):
        if direction == "long":
            stop_hit = lows[cursor] <= stop
            current_mfe = (highs[cursor] - entry) / risk
            current_mae = (entry - lows[cursor]) / risk
        else:
            stop_hit = highs[cursor] >= stop
            current_mfe = (entry - lows[cursor]) / risk
            current_mae = (highs[cursor] - entry) / risk

        if stop_hit:
            stop_idx = cursor
            mfe_until_stop_bar = max(mfe_before_stop, float(current_mfe))
            last_idx = cursor
            break

        mfe_before_stop = max(mfe_before_stop, float(current_mfe))
        mfe_until_stop_bar = max(mfe_until_stop_bar, float(current_mfe))
        mae_before_stop = max(mae_before_stop, float(current_mae))
        last_idx = cursor

    horizon_close = float(closes[last_idx])
    horizon_close_r = (horizon_close - entry) / risk if direction == "long" else (entry - horizon_close) / risk
    bars_to_stop = int(stop_idx - start_idx) if stop_idx is not None else math.nan
    hours_to_stop = float(bars_to_stop * 5.0 / 60.0) if stop_idx is not None else math.nan
    return {
        "mfe_before_stop_r_5m": float(mfe_before_stop),
        "mfe_until_stop_bar_r_5m": float(mfe_until_stop_bar),
        "mae_before_stop_r_5m": float(mae_before_stop),
        "stop_hit_5m": bool(stop_idx is not None),
        "bars_5m_to_stop": bars_to_stop,
        "hours_to_stop": hours_to_stop,
        "horizon_close_r_5m": float(horizon_close_r),
    }


def add_ideal_excursions(events: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    rows: list[dict[str, Any]] = []
    base = ensure_ohlcv_frame(base)
    open_time_ns = pd.to_datetime(base["open_time"], utc=True).to_numpy(dtype="datetime64[ns]").astype("int64")
    highs = base["high"].to_numpy(dtype=float)
    lows = base["low"].to_numpy(dtype=float)
    closes = base["close"].to_numpy(dtype=float)
    for _, event in events.iterrows():
        timeframe = str(event["timeframe"])
        settings = TF_SETTINGS.get(timeframe)
        if settings is None:
            rows.append({})
            continue
        rows.append(
            ideal_excursion_before_stop(
                open_time_ns,
                highs,
                lows,
                closes,
                entry_time=pd.Timestamp(event["entry_time"]),
                timeframe=timeframe,
                horizon=int(settings["horizon"]),
                direction=str(event["direction"]),
                entry=float(event["entry_price"]),
                stop=float(event["stop_price"]),
            )
        )
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def named_event_groups(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = {
        "all_sweeps": events,
        "4h_all": events[events["timeframe"] == "4h"],
        "4h_session_trend": events[events["candidate_4h_session_trend"]],
        "4h_session_trend_bounded": events[events["candidate_4h_session_trend_bounded"]],
    }
    return {name: group.copy() for name, group in groups.items() if not group.empty}


def ideal_tp_distribution(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, group in named_event_groups(events).items():
        values = pd.to_numeric(group["mfe_before_stop_r_5m"], errors="coerce").dropna()
        winners_1r = values[values >= 1.0]
        winners_2r = values[values >= 2.0]
        if values.empty:
            continue
        rows.append(
            {
                "group": name,
                "trades": int(len(values)),
                "stop_hit_rate_5m": float(group["stop_hit_5m"].mean()),
                "median_mfe_before_stop_r": float(values.quantile(0.50)),
                "p75_mfe_before_stop_r": float(values.quantile(0.75)),
                "p90_mfe_before_stop_r": float(values.quantile(0.90)),
                "p95_mfe_before_stop_r": float(values.quantile(0.95)),
                "p99_mfe_before_stop_r": float(values.quantile(0.99)),
                "winners_1r": int(len(winners_1r)),
                "winners_1r_median_ideal_r": float(winners_1r.quantile(0.50)) if not winners_1r.empty else math.nan,
                "winners_1r_p75_ideal_r": float(winners_1r.quantile(0.75)) if not winners_1r.empty else math.nan,
                "winners_1r_p90_ideal_r": float(winners_1r.quantile(0.90)) if not winners_1r.empty else math.nan,
                "winners_2r": int(len(winners_2r)),
                "winners_2r_median_ideal_r": float(winners_2r.quantile(0.50)) if not winners_2r.empty else math.nan,
                "winners_2r_p75_ideal_r": float(winners_2r.quantile(0.75)) if not winners_2r.empty else math.nan,
                "winners_2r_p90_ideal_r": float(winners_2r.quantile(0.90)) if not winners_2r.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)


def ideal_tp_curve(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, group in named_event_groups(events).items():
        values = pd.to_numeric(group["mfe_before_stop_r_5m"], errors="coerce").dropna()
        if values.empty:
            continue
        for rr in IDEAL_TP_R_VALUES:
            fill_rate = float((values >= rr).mean())
            rows.append(
                {
                    "group": name,
                    "target_r": float(rr),
                    "trades": int(len(values)),
                    "fill_count_before_stop": int((values >= rr).sum()),
                    "fill_rate_before_stop": fill_rate,
                    "breakeven_win_rate": float(1.0 / (rr + 1.0)),
                    "stop_assumed_expectancy_r": float(fill_rate * rr - (1.0 - fill_rate)),
                }
            )
    return pd.DataFrame(rows)


def make_path_data(frame: pd.DataFrame, timeframe: str) -> PathData:
    ordered = ensure_ohlcv_frame(frame)
    return PathData(
        timeframe=timeframe,
        open_time_ns=pd.to_datetime(ordered["open_time"], utc=True).to_numpy(dtype="datetime64[ns]").astype("int64"),
        close_time_ns=pd.to_datetime(ordered["close_time"], utc=True).to_numpy(dtype="datetime64[ns]").astype("int64"),
        highs=ordered["high"].to_numpy(dtype=float),
        lows=ordered["low"].to_numpy(dtype=float),
        closes=ordered["close"].to_numpy(dtype=float),
    )


def build_replay_paths(base: pd.DataFrame) -> dict[str, PathData]:
    base = ensure_ohlcv_frame(base)
    return {
        "5m": make_path_data(base, "5m"),
        "15m": make_path_data(resample_ohlc(base, "15m"), "15m"),
        "1h": make_path_data(resample_ohlc(base, "1h"), "1h"),
    }


def empty_stop_variant_result() -> dict[str, Any]:
    return {
        "valid": False,
        "stop_price": math.nan,
        "risk_pct": math.nan,
        "mfe_before_stop_r": math.nan,
        "mae_before_stop_r": math.nan,
        "stop_hit": False,
        "stop_exit_r": math.nan,
        "bars_5m_to_stop": math.nan,
        "hours_to_stop": math.nan,
        "horizon_close_r": math.nan,
    }


def stop_anchor_for_event(event: pd.Series, direction: str, stop_anchor_mode: str) -> float:
    wick_anchor = float(event["sweep_stop_anchor"])
    body_edge = float(event["sweep_body_edge"])
    if stop_anchor_mode == "wick":
        return wick_anchor
    if stop_anchor_mode == "half_wick_body":
        return (wick_anchor + body_edge) / 2.0
    if stop_anchor_mode == "body":
        return body_edge
    raise ValueError(f"Unknown stop anchor mode: {stop_anchor_mode}")


def scan_5m_excursion(
    path: PathData,
    *,
    start_ns: int,
    end_ns: int,
    direction: str,
    entry: float,
    risk: float,
) -> tuple[float, float, float]:
    start_idx = int(np.searchsorted(path.open_time_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(path.open_time_ns, end_ns, side="right") - 1)
    end_idx = min(max(end_idx, start_idx), len(path.open_time_ns) - 1)
    if start_idx >= len(path.open_time_ns):
        return math.nan, math.nan, math.nan

    mfe = 0.0
    mae = 0.0
    for cursor in range(start_idx, end_idx + 1):
        if direction == "long":
            mfe = max(mfe, float((path.highs[cursor] - entry) / risk))
            mae = max(mae, float((entry - path.lows[cursor]) / risk))
        else:
            mfe = max(mfe, float((entry - path.lows[cursor]) / risk))
            mae = max(mae, float((path.highs[cursor] - entry) / risk))
    close_price = float(path.closes[end_idx])
    close_r = (close_price - entry) / risk if direction == "long" else (entry - close_price) / risk
    return float(mfe), float(mae), float(close_r)


def replay_stop_variant(
    paths: dict[str, PathData],
    event: pd.Series,
    *,
    stop_anchor_mode: str,
    stop_buffer_atr: float,
    stop_mode: str,
) -> dict[str, Any]:
    timeframe = str(event["timeframe"])
    settings = TF_SETTINGS.get(timeframe)
    if settings is None or stop_mode not in STOP_MODES or stop_anchor_mode not in STOP_ANCHOR_MODES:
        return empty_stop_variant_result()

    direction = str(event["direction"])
    entry = float(event["entry_price"])
    anchor = stop_anchor_for_event(event, direction, stop_anchor_mode)
    atr = float(event["sweep_atr"])
    if not all(math.isfinite(value) for value in [entry, anchor, atr]) or atr <= 0.0:
        return empty_stop_variant_result()

    stop = anchor - stop_buffer_atr * atr if direction == "long" else anchor + stop_buffer_atr * atr
    risk = entry - stop if direction == "long" else stop - entry
    if not math.isfinite(risk) or risk <= 0.0:
        return empty_stop_variant_result()

    entry_time = pd.Timestamp(event["entry_time"]).tz_convert("UTC")
    start_ns = int(entry_time.value)
    end_ns = int((entry_time + pd.Timedelta(minutes=TF_MINUTES[timeframe] * int(settings["horizon"]))).value)
    base = paths["5m"]

    stop_hit = False
    stop_exit_r = math.nan
    stop_end_ns = end_ns
    bars_5m_to_stop = math.nan

    if stop_mode == "hard_5m":
        start_idx = int(np.searchsorted(base.open_time_ns, start_ns, side="left"))
        end_idx = int(np.searchsorted(base.open_time_ns, end_ns, side="right") - 1)
        end_idx = min(max(end_idx, start_idx), len(base.open_time_ns) - 1)
        if start_idx >= len(base.open_time_ns):
            return empty_stop_variant_result()
        stop_idx: int | None = None
        mfe = 0.0
        mae = 0.0
        last_idx = end_idx
        for cursor in range(start_idx, end_idx + 1):
            if direction == "long":
                hit = bool(base.lows[cursor] <= stop)
                current_mfe = float((base.highs[cursor] - entry) / risk)
                current_mae = float((entry - base.lows[cursor]) / risk)
            else:
                hit = bool(base.highs[cursor] >= stop)
                current_mfe = float((entry - base.lows[cursor]) / risk)
                current_mae = float((base.highs[cursor] - entry) / risk)
            if hit:
                stop_idx = cursor
                last_idx = max(start_idx, cursor - 1)
                break
            mfe = max(mfe, current_mfe)
            mae = max(mae, current_mae)
            last_idx = cursor
        stop_hit = stop_idx is not None
        if stop_hit:
            stop_exit_r = -1.0
            stop_end_ns = int(base.open_time_ns[max(start_idx, stop_idx - 1)])
            bars_5m_to_stop = int(stop_idx - start_idx)
        else:
            close_price = float(base.closes[last_idx])
            stop_exit_r = math.nan
        horizon_close = float(base.closes[last_idx])
        horizon_close_r = (horizon_close - entry) / risk if direction == "long" else (entry - horizon_close) / risk
        return {
            "valid": True,
            "stop_price": float(stop),
            "risk_pct": float(risk / entry),
            "mfe_before_stop_r": float(mfe),
            "mae_before_stop_r": float(mae),
            "stop_hit": bool(stop_hit),
            "stop_exit_r": float(stop_exit_r) if math.isfinite(stop_exit_r) else math.nan,
            "bars_5m_to_stop": bars_5m_to_stop,
            "hours_to_stop": float(bars_5m_to_stop * 5.0 / 60.0) if math.isfinite(bars_5m_to_stop) else math.nan,
            "horizon_close_r": float(horizon_close_r),
        }

    confirm_tf = stop_mode.removeprefix("close_")
    confirm = paths[confirm_tf]
    start_idx = int(np.searchsorted(confirm.close_time_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(confirm.close_time_ns, end_ns, side="right") - 1)
    end_idx = min(max(end_idx, start_idx), len(confirm.close_time_ns) - 1)
    if start_idx < len(confirm.close_time_ns):
        for cursor in range(start_idx, end_idx + 1):
            close_price = float(confirm.closes[cursor])
            hit = close_price <= stop if direction == "long" else close_price >= stop
            if hit:
                stop_hit = True
                stop_end_ns = int(confirm.close_time_ns[cursor])
                stop_exit_r = (close_price - entry) / risk if direction == "long" else (entry - close_price) / risk
                break

    mfe, mae, horizon_close_r = scan_5m_excursion(
        base,
        start_ns=start_ns,
        end_ns=stop_end_ns,
        direction=direction,
        entry=entry,
        risk=risk,
    )
    if stop_hit:
        start_5m = int(np.searchsorted(base.open_time_ns, start_ns, side="left"))
        stop_5m = int(np.searchsorted(base.close_time_ns, stop_end_ns, side="left"))
        bars_5m_to_stop = int(max(0, stop_5m - start_5m))
    return {
        "valid": bool(math.isfinite(mfe)),
        "stop_price": float(stop),
        "risk_pct": float(risk / entry),
        "mfe_before_stop_r": float(mfe),
        "mae_before_stop_r": float(mae),
        "stop_hit": bool(stop_hit),
        "stop_exit_r": float(stop_exit_r) if math.isfinite(stop_exit_r) else math.nan,
        "bars_5m_to_stop": bars_5m_to_stop,
        "hours_to_stop": float(bars_5m_to_stop * 5.0 / 60.0) if math.isfinite(bars_5m_to_stop) else math.nan,
        "horizon_close_r": float(horizon_close_r),
    }


def stop_variant_event_results(events: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    paths = build_replay_paths(base)
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        for stop_anchor_mode in STOP_ANCHOR_MODES:
            for stop_mode in STOP_MODES:
                for buffer_atr in STOP_BUFFER_ATR_VALUES:
                    result = replay_stop_variant(
                        paths,
                        event,
                        stop_anchor_mode=stop_anchor_mode,
                        stop_buffer_atr=buffer_atr,
                        stop_mode=stop_mode,
                    )
                    rows.append(
                        {
                            "event_time": event["event_time"],
                            "entry_time": event["entry_time"],
                            "timeframe": event["timeframe"],
                            "direction": event["direction"],
                            "session": event["session"],
                            "trend_side": event["trend_side"],
                            "candidate_4h_session_trend": event["candidate_4h_session_trend"],
                            "candidate_4h_session_trend_bounded": event["candidate_4h_session_trend_bounded"],
                            "stop_anchor_mode": stop_anchor_mode,
                            "stop_mode": stop_mode,
                            "stop_buffer_atr": float(buffer_atr),
                            **result,
                        }
                    )
    out = pd.DataFrame(rows)
    return out[out["valid"]].reset_index(drop=True)


def stop_variant_groups(stop_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if stop_results.empty:
        return {}
    groups = {
        "all_sweeps": stop_results,
        "4h_all": stop_results[stop_results["timeframe"] == "4h"],
        "4h_session_trend": stop_results[stop_results["candidate_4h_session_trend"]],
        "4h_session_trend_bounded": stop_results[stop_results["candidate_4h_session_trend_bounded"]],
    }
    return {name: group.copy() for name, group in groups.items() if not group.empty}


def stop_variant_summary(stop_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group in stop_variant_groups(stop_results).items():
        for (stop_anchor_mode, stop_mode, buffer_atr), variant in group.groupby(["stop_anchor_mode", "stop_mode", "stop_buffer_atr"], dropna=False):
            mfe = pd.to_numeric(variant["mfe_before_stop_r"], errors="coerce")
            mae = pd.to_numeric(variant["mae_before_stop_r"], errors="coerce")
            stop_exit = pd.to_numeric(variant["stop_exit_r"], errors="coerce")
            rows.append(
                {
                    "group": group_name,
                    "stop_anchor_mode": stop_anchor_mode,
                    "stop_mode": stop_mode,
                    "stop_buffer_atr": float(buffer_atr),
                    "trades": int(len(variant)),
                    "median_risk_pct": float(pd.to_numeric(variant["risk_pct"], errors="coerce").median()),
                    "stop_hit_rate": float(variant["stop_hit"].mean()),
                    "avg_stop_exit_r": float(stop_exit.dropna().mean()) if not stop_exit.dropna().empty else math.nan,
                    "median_mfe_before_stop_r": float(mfe.median()),
                    "p75_mfe_before_stop_r": float(mfe.quantile(0.75)),
                    "p90_mfe_before_stop_r": float(mfe.quantile(0.90)),
                    "median_mae_before_stop_r": float(mae.median()),
                    "p75_mae_before_stop_r": float(mae.quantile(0.75)),
                    "p90_mae_before_stop_r": float(mae.quantile(0.90)),
                    "fill_1r": float((mfe >= 1.0).mean()),
                    "fill_2r": float((mfe >= 2.0).mean()),
                    "fill_2_5r": float((mfe >= 2.5).mean()),
                    "fill_3r": float((mfe >= 3.0).mean()),
                    "fill_4r": float((mfe >= 4.0).mean()),
                    "fill_5r": float((mfe >= 5.0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "stop_anchor_mode", "stop_mode", "stop_buffer_atr"])


def stop_variant_tp_curve(stop_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group in stop_variant_groups(stop_results).items():
        for (stop_anchor_mode, stop_mode, buffer_atr), variant in group.groupby(["stop_anchor_mode", "stop_mode", "stop_buffer_atr"], dropna=False):
            mfe = pd.to_numeric(variant["mfe_before_stop_r"], errors="coerce")
            stop_hit = variant["stop_hit"].astype(bool)
            stop_exit = pd.to_numeric(variant["stop_exit_r"], errors="coerce")
            timeout_r = pd.to_numeric(variant["horizon_close_r"], errors="coerce")
            for rr in IDEAL_TP_R_VALUES:
                target_hit = mfe >= rr
                result_r = pd.Series(np.where(target_hit, rr, np.where(stop_hit, stop_exit, timeout_r)), index=variant.index)
                rows.append(
                    {
                        "group": group_name,
                        "stop_anchor_mode": stop_anchor_mode,
                        "stop_mode": stop_mode,
                        "stop_buffer_atr": float(buffer_atr),
                        "target_r": float(rr),
                        "trades": int(len(variant)),
                        "target_fill_rate": float(target_hit.mean()),
                        "stop_hit_rate_without_target": float((~target_hit & stop_hit).mean()),
                        "timeout_rate_without_target": float((~target_hit & ~stop_hit).mean()),
                        "avg_result_r": float(result_r.mean()),
                        "median_result_r": float(result_r.median()),
                        "win_rate": float((result_r > 0.0).mean()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["group", "stop_anchor_mode", "stop_mode", "stop_buffer_atr", "target_r"])


def compact_stop_variant_leaders(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    idx = curve.groupby(["group", "stop_anchor_mode", "stop_mode", "stop_buffer_atr"], dropna=False)["avg_result_r"].idxmax()
    leaders = curve.loc[idx].copy()
    return leaders.sort_values(["group", "avg_result_r", "target_fill_rate"], ascending=[True, False, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC liquidity sweep event study across 15m, 1h, and 4h.")
    parser.add_argument("--input", type=Path, default=Path("scripts/data/btcusdt_5m_bybit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/btc_liquidity_sweep_study"))
    parser.add_argument("--timeframes", default="15m,1h,4h")
    parser.add_argument("--min-reclaim-pos", type=float, default=0.55)
    parser.add_argument("--min-sweep-depth-atr", type=float, default=0.02)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.15)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--max-scan", type=int, default=12)
    parser.add_argument("--skip-stop-variants", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_ohlcv_csv(args.input)
    base = ensure_ohlcv_frame(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_events: list[pd.DataFrame] = []
    for timeframe in [item.strip() for item in args.timeframes.split(",") if item.strip()]:
        if timeframe == "5m":
            frame = enrich(base)
        else:
            frame = enrich(resample_ohlc(base, timeframe))
        settings = TF_SETTINGS[timeframe]
        events = detect_sweeps(
            frame,
            timeframe=timeframe,
            pivot_window=int(settings["pivot_window"]),
            max_age=int(settings["max_age"]),
            horizon=int(settings["horizon"]),
            min_reclaim_pos=args.min_reclaim_pos,
            min_sweep_depth_atr=args.min_sweep_depth_atr,
            stop_buffer_atr=args.stop_buffer_atr,
            target_rr=args.target_rr,
            max_scan=args.max_scan,
        )
        all_events.append(events)
        print(
            f"{timeframe}: events={len(events)} hit_rate={events['target_before_stop'].mean() if not events.empty else 0.0:.3f} "
            f"avg_r={events['outcome_r'].mean() if not events.empty else 0.0:.3f}",
            flush=True,
        )
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events = add_ideal_excursions(events, base)
    events, candidate_thresholds = add_candidate_flags(events)
    events.to_csv(args.output_dir / "btc_liquidity_sweep_events.csv", index=False)
    events[events["candidate_4h_session_trend"]].to_csv(args.output_dir / "filtered_4h_session_trend_candidates.csv", index=False)
    events[events["candidate_4h_session_trend_bounded"]].to_csv(args.output_dir / "filtered_4h_session_trend_bounded_candidates.csv", index=False)
    pd.DataFrame([candidate_thresholds]).to_csv(args.output_dir / "candidate_filter_thresholds.csv", index=False)

    stop_results = pd.DataFrame()
    stop_curve = pd.DataFrame()
    if not args.skip_stop_variants:
        stop_results = stop_variant_event_results(events, base)
        stop_curve = stop_variant_tp_curve(stop_results)
        stop_results.to_csv(args.output_dir / "stop_variant_event_results.csv", index=False)

    min_count = 10
    summaries = {
        "summary_by_tf_direction_session.csv": summarize_group(events, ["timeframe", "direction", "session"], min_count),
        "summary_by_tf_direction_trend.csv": summarize_group(events, ["timeframe", "direction", "trend_side"], min_count),
        "summary_by_tf_direction.csv": summarize_group(events, ["timeframe", "direction"], min_count),
        "feature_contrast_good_vs_bad.csv": feature_contrast(events),
        "top_patterns.csv": top_patterns(events, min_count),
        "ideal_tp_distribution.csv": ideal_tp_distribution(events),
        "ideal_tp_rr_curve.csv": ideal_tp_curve(events),
    }
    if not stop_results.empty:
        summaries["stop_variant_summary.csv"] = stop_variant_summary(stop_results)
        summaries["stop_variant_tp_curve.csv"] = stop_curve
        summaries["stop_variant_leaders.csv"] = compact_stop_variant_leaders(stop_curve)
    bucket_parts = [
        numeric_bucket_summary(events, column, min_count)
        for column in [
            "sweep_depth_atr",
            "reclaim_pos",
            "volume_ratio",
            "atr_ratio",
            "dist_ema200_atr",
            "pre_return_12_atr",
            "pre_range_12_atr",
            "level_age_bars",
            "risk_pct",
        ]
    ]
    summaries["numeric_bucket_summary.csv"] = pd.concat([part for part in bucket_parts if not part.empty], ignore_index=True)
    for filename, table in summaries.items():
        table.to_csv(args.output_dir / filename, index=False)

    overview = {
        "events": int(len(events)),
        "target_rr": float(args.target_rr),
        "hit_rate": float(events["target_before_stop"].mean()) if not events.empty else 0.0,
        "avg_r": float(events["outcome_r"].mean()) if not events.empty else 0.0,
        "median_mfe_r": float(events["mfe_r"].median()) if not events.empty else 0.0,
        "median_mae_r": float(events["mae_r"].median()) if not events.empty else 0.0,
    }
    pd.DataFrame([overview]).to_csv(args.output_dir / "overview.csv", index=False)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
