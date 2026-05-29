from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import (
    add_indicators,
    bybit_symbol,
    ensure_ohlcv_frame,
    fetch_bybit_klines,
    load_ohlcv_csv,
    metric_row,
    normalize_timeframe,
    parse_utc_datetime,
    resample_ohlc,
    split_trades,
    strategy_metrics,
)


STRATEGIES = {
    "vol_breakout": "Volatility contraction breakout",
    "sweep_reclaim": "Liquidity sweep + reclaim",
    "trend_pullback": "Trend pullback continuation",
    "bb_regime": "Regime-switched Bollinger/Keltner",
    "intraday_timing": "Intraday momentum/reversal timing",
}

METRIC_COLUMNS = [
    "robust_score",
    "train_net_r",
    "validation_net_r",
    "oos_net_r",
    "all_net_r",
    "train_avg_r",
    "validation_avg_r",
    "oos_avg_r",
    "all_avg_r",
    "train_profit_factor",
    "validation_profit_factor",
    "oos_profit_factor",
    "all_profit_factor",
    "train_trades",
    "validation_trades",
    "oos_trades",
    "all_trades",
]

INTERVAL_TO_BARS_5M = {
    "5m": 1,
    "15m": 3,
    "1h": 12,
    "4h": 48,
    "1d": 288,
    "1w": 2016,
}

SIGNAL_FRAME_CACHE: dict[tuple[int, str], pd.DataFrame] = {}
ENTRY_AFTER_CACHE: dict[tuple[int, str], pd.Series] = {}
SWING_LEVEL_CACHE: dict[tuple[int, str, int], pd.DataFrame] = {}
LEVEL_FRAME_CACHE: dict[tuple[int, str], pd.DataFrame] = {}
PROJECTED_LEVEL_CACHE: dict[tuple[int, int, str, str, int], pd.DataFrame] = {}


@dataclass(frozen=True)
class Event:
    direction: str
    signal_time: pd.Timestamp
    entry_after: pd.Timestamp
    stop_anchor: float
    target_anchor: float | None
    context: str


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def comma_values(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def session_name(hour: int) -> str:
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "ny"
    return "late"


def allowed_session(ts: pd.Timestamp, session: str) -> bool:
    session = str(session)
    if session in {"all", ""}:
        return True
    hour = int(pd.Timestamp(ts).hour)
    if session == "london_ny":
        return 7 <= hour < 21
    if session == "london_open":
        return 7 <= hour < 10
    if session == "ny_open":
        return 13 <= hour < 16
    if session == "killzones":
        return (7 <= hour < 10) or (13 <= hour < 16)
    return session_name(hour) == session


def load_symbol_frame(symbol: str, *, interval: str, days: int, cache_dir: Path, end: str | None) -> pd.DataFrame:
    symbol = bybit_symbol(symbol)
    interval = normalize_timeframe(interval)
    cache_path = cache_dir / f"{symbol.lower()}_{interval}_bybit.csv"
    if cache_path.exists():
        frame = load_ohlcv_csv(cache_path)
    else:
        end_dt = parse_utc_datetime(end) if end else datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        frame = fetch_bybit_klines(symbol, interval, start_dt, end_dt)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
    out = ensure_ohlcv_frame(frame)
    if out.empty:
        return out
    end_ts = pd.Timestamp(parse_utc_datetime(end)).tz_convert("UTC") if end else pd.Timestamp(out["open_time"].iloc[-1]).tz_convert("UTC")
    start_ts = end_ts - pd.Timedelta(days=days)
    times = pd.to_datetime(out["open_time"], utc=True)
    return out[(times >= start_ts) & (times <= end_ts)].reset_index(drop=True)


def with_common_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(frame, 14, 200, 14).copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema100"] = close.ewm(span=100, adjust=False).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()
    out["ema50_slope_atr"] = (out["ema50"] - out["ema50"].shift(10)) / out["atr"].replace(0.0, np.nan)
    out["ema200_slope_atr"] = (out["ema200"] - out["ema200"].shift(20)) / out["atr"].replace(0.0, np.nan)
    out["range_atr"] = (high - low) / out["atr"].replace(0.0, np.nan)
    out["volume_ratio"] = volume / volume.rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    out["bb_mid20"] = close.rolling(20, min_periods=20).mean()
    out["bb_std20"] = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_width20"] = (4.0 * out["bb_std20"]) / close.replace(0.0, np.nan)
    out["keltner_width20"] = (2.0 * out["atr"]) / close.replace(0.0, np.nan)
    return out


def prepared_signal_frame(exec_frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf = normalize_timeframe(timeframe)
    key = (id(exec_frame), tf)
    cached = SIGNAL_FRAME_CACHE.get(key)
    if cached is not None:
        return cached
    out = with_common_indicators(resample_ohlc(exec_frame, tf))
    out.attrs["timeframe"] = tf
    SIGNAL_FRAME_CACHE[key] = out
    return out


def normalize_research_timeframe(timeframe: str) -> str:
    raw = str(timeframe).strip().lower()
    if raw in {"1w", "1wk", "w", "week", "weekly"}:
        return "1w"
    if raw in {"1mo", "1mth", "month", "monthly"}:
        return "1mo"
    return normalize_timeframe(raw)


def calendar_bucket_start(times: pd.Series, period: str) -> pd.Series:
    ts = pd.to_datetime(times, utc=True)
    period = str(period)
    if period in {"daily", "1d", "d"}:
        return ts.dt.floor("D")
    if period in {"weekly", "1w", "w"}:
        day = ts.dt.floor("D")
        return day - pd.to_timedelta(day.dt.weekday, unit="D")
    if period in {"monthly", "1mo", "m"}:
        return ts.dt.tz_convert(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    raise ValueError(f"Unknown calendar period: {period}")


def resample_ohlc_research(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf = normalize_research_timeframe(timeframe)
    if tf != "1w":
        return resample_ohlc(df, tf)

    ordered = ensure_ohlcv_frame(df)
    out = ordered.copy()
    out["bucket"] = calendar_bucket_start(out["open_time"], "weekly")
    resampled = (
        out.groupby("bucket", sort=True)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
        .rename(columns={"bucket": "open_time"})
    )
    resampled["close_time"] = resampled["open_time"] + pd.Timedelta(days=7) - pd.Timedelta(milliseconds=1)
    return resampled[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def prepared_level_frame(exec_frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf = normalize_research_timeframe(timeframe)
    key = (id(exec_frame), tf)
    cached = LEVEL_FRAME_CACHE.get(key)
    if cached is not None:
        return cached
    out = with_common_indicators(resample_ohlc_research(exec_frame, tf))
    out.attrs["timeframe"] = tf
    LEVEL_FRAME_CACHE[key] = out
    return out


def width_percentile(series: pd.Series, lookback: int) -> pd.Series:
    def pct(values: np.ndarray) -> float:
        if len(values) <= 1 or not np.isfinite(values[-1]):
            return math.nan
        prior = values[:-1]
        prior = prior[np.isfinite(prior)]
        if prior.size == 0:
            return math.nan
        return float((prior <= values[-1]).mean())

    return series.rolling(lookback + 1, min_periods=max(10, lookback // 3)).apply(pct, raw=True)


def swing_level_frame(signal_frame: pd.DataFrame, pivot_window: int) -> pd.DataFrame:
    key = (id(signal_frame), normalize_research_timeframe(str(signal_frame.attrs.get("timeframe", "5m"))), int(pivot_window))
    cached = SWING_LEVEL_CACHE.get(key)
    if cached is not None:
        return cached

    w = max(int(pivot_window), 1)
    n = len(signal_frame)
    highs = signal_frame["high"].to_numpy(dtype=float)
    lows = signal_frame["low"].to_numpy(dtype=float)
    swing_high = np.full(n, np.nan, dtype=float)
    swing_low = np.full(n, np.nan, dtype=float)
    swing_high_age = np.full(n, np.nan, dtype=float)
    swing_low_age = np.full(n, np.nan, dtype=float)
    swing_high_confirm = np.full(n, np.nan, dtype=float)
    swing_low_confirm = np.full(n, np.nan, dtype=float)

    if n > 2 * w + 1:
        window = 2 * w + 1
        high_windows = np.lib.stride_tricks.sliding_window_view(highs, window)
        low_windows = np.lib.stride_tricks.sliding_window_view(lows, window)
        centers = np.arange(w, n - w)
        high_centers = highs[centers]
        low_centers = lows[centers]
        is_high = (
            np.isfinite(high_centers)
            & (high_centers == np.nanmax(high_windows, axis=1))
            & (np.nanargmax(high_windows, axis=1) == w)
        )
        is_low = (
            np.isfinite(low_centers)
            & (low_centers == np.nanmin(low_windows, axis=1))
            & (np.nanargmin(low_windows, axis=1) == w)
        )

        high_by_confirm: dict[int, float] = {}
        low_by_confirm: dict[int, float] = {}
        for center, keep in zip(centers, is_high, strict=False):
            if bool(keep):
                high_by_confirm[int(center + w)] = float(highs[int(center)])
        for center, keep in zip(centers, is_low, strict=False):
            if bool(keep):
                low_by_confirm[int(center + w)] = float(lows[int(center)])

        last_high = math.nan
        last_low = math.nan
        last_high_confirm = math.nan
        last_low_confirm = math.nan
        for idx in range(n):
            usable_idx = idx - 1
            if usable_idx in high_by_confirm:
                last_high = high_by_confirm[usable_idx]
                last_high_confirm = float(usable_idx)
            if usable_idx in low_by_confirm:
                last_low = low_by_confirm[usable_idx]
                last_low_confirm = float(usable_idx)
            swing_high[idx] = last_high
            swing_low[idx] = last_low
            swing_high_confirm[idx] = last_high_confirm
            swing_low_confirm[idx] = last_low_confirm
            if _finite(last_high_confirm):
                swing_high_age[idx] = float(idx) - last_high_confirm
            if _finite(last_low_confirm):
                swing_low_age[idx] = float(idx) - last_low_confirm

    out = pd.DataFrame(
        {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "swing_high_age": swing_high_age,
            "swing_low_age": swing_low_age,
            "swing_high_confirm_idx": swing_high_confirm,
            "swing_low_confirm_idx": swing_low_confirm,
        },
        index=signal_frame.index,
    )
    SWING_LEVEL_CACHE[key] = out
    return out


def project_htf_swing_levels(
    exec_frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
    level_tf: str,
    pivot_window: int,
) -> pd.DataFrame:
    tf = normalize_research_timeframe(level_tf)
    key = (id(exec_frame), id(signal_frame), "htf_swing", tf, int(pivot_window))
    cached = PROJECTED_LEVEL_CACHE.get(key)
    if cached is not None:
        return cached

    htf = prepared_level_frame(exec_frame, tf)
    levels = swing_level_frame(htf, pivot_window)
    n = len(signal_frame)
    out = pd.DataFrame(index=signal_frame.index)
    for column in [
        "level_high",
        "level_low",
        "level_high_age",
        "level_low_age",
        "level_high_id",
        "level_low_id",
    ]:
        out[column] = np.nan
    if htf.empty:
        PROJECTED_LEVEL_CACHE[key] = out
        return out

    signal_times = pd.to_datetime(signal_frame["open_time"], utc=True).to_numpy(dtype="datetime64[ns]")
    htf_close_times = pd.to_datetime(htf["close_time"], utc=True).to_numpy(dtype="datetime64[ns]")
    htf_open_times = pd.to_datetime(htf["open_time"], utc=True).to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(htf_close_times, signal_times, side="right") - 1
    valid = (idx >= 0) & (idx < len(htf))
    valid_idx = idx[valid]
    htf_ms = max(float((pd.Timestamp(htf_close_times[min(len(htf_close_times) - 1, 1)]) - pd.Timestamp(htf_open_times[0])).total_seconds() * 1000.0), 1.0)
    if tf in {"4h", "1d", "1w"}:
        htf_ms = float({"4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000}[tf])

    elapsed = np.zeros(n, dtype=float)
    elapsed[valid] = np.maximum(
        (signal_times[valid] - htf_open_times[valid_idx]).astype("timedelta64[ms]").astype(float) / htf_ms,
        0.0,
    )
    high_age = levels["swing_high_age"].to_numpy(dtype=float)
    low_age = levels["swing_low_age"].to_numpy(dtype=float)
    out.loc[valid, "level_high"] = levels["swing_high"].to_numpy(dtype=float)[valid_idx]
    out.loc[valid, "level_low"] = levels["swing_low"].to_numpy(dtype=float)[valid_idx]
    out.loc[valid, "level_high_age"] = high_age[valid_idx] + elapsed[valid]
    out.loc[valid, "level_low_age"] = low_age[valid_idx] + elapsed[valid]
    out.loc[valid, "level_high_id"] = levels["swing_high_confirm_idx"].to_numpy(dtype=float)[valid_idx]
    out.loc[valid, "level_low_id"] = levels["swing_low_confirm_idx"].to_numpy(dtype=float)[valid_idx]

    PROJECTED_LEVEL_CACHE[key] = out
    return out


def previous_period_levels(signal_frame: pd.DataFrame, period: str) -> pd.DataFrame:
    period = str(period)
    key = (id(signal_frame), id(signal_frame), "period", period, 0)
    cached = PROJECTED_LEVEL_CACHE.get(key)
    if cached is not None:
        return cached

    out = pd.DataFrame(index=signal_frame.index)
    times = pd.to_datetime(signal_frame["open_time"], utc=True)
    buckets = calendar_bucket_start(times, period)
    frame = signal_frame.copy()
    frame["bucket"] = buckets
    grouped = frame.groupby("bucket", sort=True).agg(level_high=("high", "max"), level_low=("low", "min"))
    grouped["prev_high"] = grouped["level_high"].shift(1)
    grouped["prev_low"] = grouped["level_low"].shift(1)
    grouped["period_id"] = np.arange(len(grouped), dtype=float)
    mapped = pd.DataFrame({"bucket": buckets}, index=signal_frame.index).join(
        grouped[["prev_high", "prev_low", "period_id"]],
        on="bucket",
    )
    out["level_high"] = mapped["prev_high"].astype(float)
    out["level_low"] = mapped["prev_low"].astype(float)
    out["level_high_age"] = 0.0
    out["level_low_age"] = 0.0
    out["level_high_id"] = mapped["period_id"].astype(float)
    out["level_low_id"] = mapped["period_id"].astype(float)
    PROJECTED_LEVEL_CACHE[key] = out
    return out


def regime_ok(row: pd.Series, direction: str, regime: str) -> bool:
    regime = str(regime)
    close = float(row.get("close", math.nan))
    ema = float(row.get("ema200", row.get("ema", math.nan)))
    slope = float(row.get("ema200_slope_atr", row.get("ema_slope_atr", math.nan)))
    atr_ratio = float(row.get("atr_ratio", math.nan))
    if regime in {"none", ""}:
        return True
    if regime == "high_vol":
        return _finite(atr_ratio) and atr_ratio >= 1.0
    if regime == "low_vol":
        return _finite(atr_ratio) and atr_ratio <= 1.0
    if regime == "trend_aligned":
        if not (_finite(close) and _finite(ema) and _finite(slope)):
            return False
        return (close >= ema and slope >= 0.0) if direction == "long" else (close <= ema and slope <= 0.0)
    if regime == "mean_reversion":
        if not (_finite(close) and _finite(ema) and _finite(slope)):
            return False
        return (close <= ema and slope <= 0.0) if direction == "long" else (close >= ema and slope >= 0.0)
    raise ValueError(f"Unknown regime: {regime}")


def trend_ok(row: pd.Series, direction: str, trend_filter: str) -> bool:
    trend_filter = str(trend_filter)
    close = float(row.get("close", math.nan))
    ema = float(row.get("ema200", row.get("ema", math.nan)))
    slope = float(row.get("ema200_slope_atr", row.get("ema_slope_atr", math.nan)))
    rsi = float(row.get("rsi", math.nan))
    if trend_filter in {"none", ""}:
        return True
    if trend_filter == "ema":
        if not (_finite(close) and _finite(ema)):
            return False
        return close >= ema if direction == "long" else close <= ema
    if trend_filter == "ema_slope":
        if not (_finite(close) and _finite(ema) and _finite(slope)):
            return False
        return (close >= ema and slope >= 0.0) if direction == "long" else (close <= ema and slope <= 0.0)
    if trend_filter == "rsi":
        if not _finite(rsi):
            return False
        return rsi <= 62.0 if direction == "long" else rsi >= 38.0
    if trend_filter == "counter_ema":
        if not (_finite(close) and _finite(ema)):
            return False
        return close <= ema if direction == "long" else close >= ema
    raise ValueError(f"Unknown trend_filter: {trend_filter}")


def entry_after_times(exec_frame: pd.DataFrame, signal_frame: pd.DataFrame) -> pd.Series:
    tf = normalize_timeframe(str(signal_frame.attrs.get("timeframe", ""))) if signal_frame.attrs.get("timeframe") else ""
    if tf:
        key = (id(exec_frame), tf)
        cached = ENTRY_AFTER_CACHE.get(key)
        if cached is not None:
            return cached
    # Entry can occur on the next 5m bar after a completed signal candle.
    exec_times = pd.to_datetime(exec_frame["open_time"], utc=True).to_numpy(dtype="datetime64[ns]")
    signal_close = pd.to_datetime(signal_frame["close_time"], utc=True).to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(exec_times, signal_close + np.timedelta64(1, "ns"), side="left")
    safe_idx = np.minimum(idx, len(exec_times) - 1)
    out = pd.Series(pd.to_datetime(exec_times[safe_idx], utc=True), index=signal_frame.index)
    out[idx >= len(exec_times)] = pd.NaT
    if tf:
        ENTRY_AFTER_CACHE[key] = out
    return out


def build_vol_breakout_events(exec_frame: pd.DataFrame, params: dict[str, Any]) -> list[Event]:
    tf = normalize_timeframe(params["signal_tf"])
    sig = prepared_signal_frame(exec_frame, tf)
    lookback = int(params["lookback"])
    squeeze_lookback = int(params["squeeze_lookback"])
    width_col = "bb_width20" if params["width_source"] == "bb" else "keltner_width20"
    sig["width_pctile"] = width_percentile(sig[width_col], squeeze_lookback)
    prior_high = sig["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    prior_low = sig["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    entry_after = entry_after_times(exec_frame, sig)
    events: list[Event] = []
    atr_values = sig["atr"].to_numpy(dtype=float)
    close_values = sig["close"].to_numpy(dtype=float)
    prior_high_values = prior_high.to_numpy(dtype=float)
    prior_low_values = prior_low.to_numpy(dtype=float)
    width_values = sig["width_pctile"].to_numpy(dtype=float)
    valid = (
        np.isfinite(atr_values)
        & (atr_values > 0.0)
        & np.isfinite(width_values)
        & (width_values <= float(params["max_width_pctile"]))
        & np.isfinite(prior_high_values)
        & np.isfinite(prior_low_values)
        & (prior_high_values > prior_low_values)
    )
    break_atr = float(params["break_atr"])
    long_idx = np.flatnonzero(valid & (close_values > prior_high_values + break_atr * atr_values))
    short_idx = np.flatnonzero(valid & (close_values < prior_low_values - break_atr * atr_values))
    long_set = set(int(i) for i in long_idx)
    for i in np.r_[long_idx, short_idx]:
        i = int(i)
        row = sig.iloc[int(i)]
        atr = float(atr_values[int(i)])
        hi = float(prior_high_values[int(i)])
        lo = float(prior_low_values[int(i)])
        signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        after = entry_after.iloc[int(i)]
        if pd.isna(after):
            continue
        if i in long_set:
            direction = "long"
            if trend_ok(row, direction, params["trend_filter"]) and regime_ok(row, direction, params["regime"]):
                events.append(
                    Event(direction, signal_time, after, lo - float(params["stop_buffer_atr"]) * atr, None, "vol_breakout")
                )
        else:
            direction = "short"
            if trend_ok(row, direction, params["trend_filter"]) and regime_ok(row, direction, params["regime"]):
                events.append(
                    Event(direction, signal_time, after, hi + float(params["stop_buffer_atr"]) * atr, None, "vol_breakout")
                )
    return events


def build_sweep_reclaim_events(exec_frame: pd.DataFrame, params: dict[str, Any]) -> list[Event]:
    tf = normalize_timeframe(params["signal_tf"])
    sig = prepared_signal_frame(exec_frame, tf)
    lookback = int(params["lookback"])
    level_source = str(params.get("level_source", "rolling"))
    if level_source == "rolling":
        level_high = sig["high"].rolling(lookback, min_periods=lookback).max().shift(1)
        level_low = sig["low"].rolling(lookback, min_periods=lookback).min().shift(1)
        level_high_age = pd.Series(float(lookback), index=sig.index)
        level_low_age = pd.Series(float(lookback), index=sig.index)
        level_high_id = pd.Series(np.arange(len(sig)), index=sig.index)
        level_low_id = pd.Series(np.arange(len(sig)), index=sig.index)
    elif level_source == "swing":
        levels = swing_level_frame(sig, int(params.get("pivot_window", 3)))
        level_high = levels["swing_high"]
        level_low = levels["swing_low"]
        level_high_age = levels["swing_high_age"]
        level_low_age = levels["swing_low_age"]
        level_high_id = levels["swing_high_confirm_idx"]
        level_low_id = levels["swing_low_confirm_idx"]
    elif level_source == "htf_swing":
        levels = project_htf_swing_levels(exec_frame, sig, str(params.get("level_tf", "4h")), int(params.get("pivot_window", 3)))
        level_high = levels["level_high"]
        level_low = levels["level_low"]
        level_high_age = levels["level_high_age"]
        level_low_age = levels["level_low_age"]
        level_high_id = levels["level_high_id"]
        level_low_id = levels["level_low_id"]
    elif level_source == "period":
        levels = previous_period_levels(sig, str(params.get("period_level", "daily")))
        level_high = levels["level_high"]
        level_low = levels["level_low"]
        level_high_age = levels["level_high_age"]
        level_low_age = levels["level_low_age"]
        level_high_id = levels["level_high_id"]
        level_low_id = levels["level_low_id"]
    else:
        raise ValueError(f"Unknown sweep level_source: {level_source}")
    entry_after = entry_after_times(exec_frame, sig)
    events: list[Event] = []
    atr_values = sig["atr"].to_numpy(dtype=float)
    close_values = sig["close"].to_numpy(dtype=float)
    high_values = sig["high"].to_numpy(dtype=float)
    low_values = sig["low"].to_numpy(dtype=float)
    level_high_values = level_high.to_numpy(dtype=float)
    level_low_values = level_low.to_numpy(dtype=float)
    level_high_age_values = level_high_age.to_numpy(dtype=float)
    level_low_age_values = level_low_age.to_numpy(dtype=float)
    level_high_id_values = level_high_id.to_numpy(dtype=float)
    level_low_id_values = level_low_id.to_numpy(dtype=float)
    valid_base = np.isfinite(atr_values) & (atr_values > 0.0)
    min_sweep = float(params["min_sweep_atr"]) * atr_values
    reclaim = float(params["reclaim_atr"]) * atr_values
    max_level_age = int(params.get("level_lookback", lookback))
    reclaim_bars = int(params.get("reclaim_bars", 0))
    min_close_location = float(params.get("min_close_location", 0.0))
    direction_mode = str(params.get("direction", "both"))
    session = str(params.get("session", "all"))
    min_stop_buffer_pct = float(params.get("min_stop_buffer_pct", 0.0)) / 100.0
    allow_longs = direction_mode in {"both", "long"}
    allow_shorts = direction_mode in {"both", "short"}

    long_sweeps = np.flatnonzero(
        valid_base
        & allow_longs
        & np.isfinite(level_low_values)
        & np.isfinite(level_low_age_values)
        & (level_low_age_values <= max_level_age)
        & (low_values < level_low_values - min_sweep)
    )
    short_sweeps = np.flatnonzero(
        valid_base
        & allow_shorts
        & np.isfinite(level_high_values)
        & np.isfinite(level_high_age_values)
        & (level_high_age_values <= max_level_age)
        & (high_values > level_high_values + min_sweep)
    )
    used_levels: set[tuple[str, int]] = set()

    def close_location(idx: int) -> float:
        bar_range = float(high_values[idx] - low_values[idx])
        if not _finite(bar_range) or bar_range <= 0.0:
            return math.nan
        return float((close_values[idx] - low_values[idx]) / bar_range)

    for i in long_sweeps:
        i = int(i)
        level_id = int(level_low_id_values[i]) if _finite(level_low_id_values[i]) else i
        key = ("long", level_id)
        if key in used_levels:
            continue
        sweep_low = float(low_values[i])
        end_idx = min(len(sig) - 1, i + reclaim_bars)
        for j in range(i, end_idx + 1):
            if not (np.isfinite(close_values[j]) and np.isfinite(atr_values[j]) and atr_values[j] > 0.0):
                continue
            location = close_location(j)
            if close_values[j] <= level_low_values[i] + float(params["reclaim_atr"]) * atr_values[j]:
                continue
            if _finite(location) and location < min_close_location:
                continue
            row = sig.iloc[j]
            after = entry_after.iloc[j]
            if pd.isna(after):
                break
            if not allowed_session(pd.Timestamp(row["close_time"]), session):
                continue
            direction = "long"
            if trend_ok(row, direction, params["trend_filter"]) and regime_ok(row, direction, params["regime"]):
                signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
                stop_buffer = max(float(params["stop_buffer_atr"]) * float(atr_values[j]), float(close_values[j]) * min_stop_buffer_pct)
                stop = sweep_low - stop_buffer
                events.append(Event(direction, signal_time, after, stop, None, "sweep_reclaim"))
                used_levels.add(key)
            break

    for i in short_sweeps:
        i = int(i)
        level_id = int(level_high_id_values[i]) if _finite(level_high_id_values[i]) else i
        key = ("short", level_id)
        if key in used_levels:
            continue
        sweep_high = float(high_values[i])
        end_idx = min(len(sig) - 1, i + reclaim_bars)
        for j in range(i, end_idx + 1):
            if not (np.isfinite(close_values[j]) and np.isfinite(atr_values[j]) and atr_values[j] > 0.0):
                continue
            location = close_location(j)
            if close_values[j] >= level_high_values[i] - float(params["reclaim_atr"]) * atr_values[j]:
                continue
            if _finite(location) and location > 1.0 - min_close_location:
                continue
            row = sig.iloc[j]
            after = entry_after.iloc[j]
            if pd.isna(after):
                break
            if not allowed_session(pd.Timestamp(row["close_time"]), session):
                continue
            direction = "short"
            if trend_ok(row, direction, params["trend_filter"]) and regime_ok(row, direction, params["regime"]):
                signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
                stop_buffer = max(float(params["stop_buffer_atr"]) * float(atr_values[j]), float(close_values[j]) * min_stop_buffer_pct)
                stop = sweep_high + stop_buffer
                events.append(Event(direction, signal_time, after, stop, None, "sweep_reclaim"))
                used_levels.add(key)
            break
    return events


def build_trend_pullback_events(exec_frame: pd.DataFrame, params: dict[str, Any]) -> list[Event]:
    tf = normalize_timeframe(params["signal_tf"])
    sig = prepared_signal_frame(exec_frame, tf)
    trend_col = f"ema{int(params['trend_ema'])}"
    pullback_col = f"ema{int(params['pullback_ema'])}"
    stop_lookback = int(params["stop_lookback"])
    prior_high = sig["high"].shift(1)
    prior_low = sig["low"].shift(1)
    swing_low = sig["low"].rolling(stop_lookback, min_periods=stop_lookback).min()
    swing_high = sig["high"].rolling(stop_lookback, min_periods=stop_lookback).max()
    entry_after = entry_after_times(exec_frame, sig)
    events: list[Event] = []
    atr_values = sig["atr"].to_numpy(dtype=float)
    close_values = sig["close"].to_numpy(dtype=float)
    high_values = sig["high"].to_numpy(dtype=float)
    low_values = sig["low"].to_numpy(dtype=float)
    trend_values = sig[trend_col].to_numpy(dtype=float)
    pull_values = sig[pullback_col].to_numpy(dtype=float)
    slope_values = sig["ema200_slope_atr"].to_numpy(dtype=float)
    prior_high_values = prior_high.to_numpy(dtype=float)
    prior_low_values = prior_low.to_numpy(dtype=float)
    valid = (
        np.isfinite(atr_values)
        & (atr_values > 0.0)
        & np.isfinite(close_values)
        & np.isfinite(trend_values)
        & np.isfinite(pull_values)
        & np.isfinite(slope_values)
        & (np.abs(slope_values) >= float(params["min_slope_atr"]))
    )
    if str(params["trigger"]) == "close_reclaim":
        long_trigger = close_values > pull_values
        short_trigger = close_values < pull_values
    else:
        long_trigger = np.isfinite(prior_high_values) & (close_values > prior_high_values)
        short_trigger = np.isfinite(prior_low_values) & (close_values < prior_low_values)
    long_idx = np.flatnonzero(valid & (close_values > trend_values) & (slope_values > 0.0) & (low_values <= pull_values) & long_trigger)
    short_idx = np.flatnonzero(valid & (close_values < trend_values) & (slope_values < 0.0) & (high_values >= pull_values) & short_trigger)
    for i in long_idx:
        i = int(i)
        row = sig.iloc[i]
        after = entry_after.iloc[i]
        if pd.isna(after):
            continue
        signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        stop = float(swing_low.iloc[i]) - float(params["stop_buffer_atr"]) * float(atr_values[i])
        if _finite(stop):
            events.append(Event("long", signal_time, after, stop, None, "trend_pullback"))
    for i in short_idx:
        i = int(i)
        row = sig.iloc[i]
        after = entry_after.iloc[i]
        if pd.isna(after):
            continue
        signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        stop = float(swing_high.iloc[i]) + float(params["stop_buffer_atr"]) * float(atr_values[i])
        if _finite(stop):
            events.append(Event("short", signal_time, after, stop, None, "trend_pullback"))
    return events


def build_bb_regime_events(exec_frame: pd.DataFrame, params: dict[str, Any]) -> list[Event]:
    tf = normalize_timeframe(params["signal_tf"])
    sig = prepared_signal_frame(exec_frame, tf)
    length = int(params["length"])
    close = sig["close"].astype(float)
    mid = close.rolling(length, min_periods=length).mean()
    std = close.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + float(params["band_mult"]) * std
    lower = mid - float(params["band_mult"]) * std
    width = (upper - lower) / close.replace(0.0, np.nan)
    sig["band_width_pctile"] = width_percentile(width, int(params["width_lookback"]))
    entry_after = entry_after_times(exec_frame, sig)
    mode = str(params["mode"])
    events: list[Event] = []
    atr_values = sig["atr"].to_numpy(dtype=float)
    close_values = sig["close"].to_numpy(dtype=float)
    high_values = sig["high"].to_numpy(dtype=float)
    low_values = sig["low"].to_numpy(dtype=float)
    mid_values = mid.to_numpy(dtype=float)
    upper_values = upper.to_numpy(dtype=float)
    lower_values = lower.to_numpy(dtype=float)
    width_pct_values = sig["band_width_pctile"].to_numpy(dtype=float)
    valid = (
        np.isfinite(atr_values)
        & (atr_values > 0.0)
        & np.isfinite(mid_values)
        & np.isfinite(upper_values)
        & np.isfinite(lower_values)
        & np.isfinite(width_pct_values)
    )
    if mode == "breakout":
        breakout_mask = valid
        revert_mask = np.zeros(len(sig), dtype=bool)
    elif mode == "mean_reversion":
        breakout_mask = np.zeros(len(sig), dtype=bool)
        revert_mask = valid
    else:
        breakout_mask = valid & (width_pct_values >= float(params["breakout_width_pctile"]))
        revert_mask = valid & (width_pct_values <= float(params["reversion_width_pctile"]))
    long_break = np.flatnonzero(breakout_mask & (close_values > upper_values))
    short_break = np.flatnonzero(breakout_mask & (close_values < lower_values))
    long_revert = np.flatnonzero(revert_mask & (low_values < lower_values) & (close_values > lower_values))
    short_revert = np.flatnonzero(revert_mask & (high_values > upper_values) & (close_values < upper_values))
    for i in long_break:
        i = int(i)
        row = sig.iloc[i]
        after = entry_after.iloc[i]
        if pd.isna(after):
            continue
        signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        direction = "long"
        if trend_ok(row, direction, params["trend_filter"]) and regime_ok(row, direction, params["regime"]):
            stop = float(mid_values[i]) - float(params["stop_buffer_atr"]) * float(atr_values[i])
            events.append(Event(direction, signal_time, after, stop, None, "bb_breakout"))
    for i in short_break:
        i = int(i)
        row = sig.iloc[i]
        after = entry_after.iloc[i]
        if pd.isna(after):
            continue
        signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        direction = "short"
        if trend_ok(row, direction, params["trend_filter"]) and regime_ok(row, direction, params["regime"]):
            stop = float(mid_values[i]) + float(params["stop_buffer_atr"]) * float(atr_values[i])
            events.append(Event(direction, signal_time, after, stop, None, "bb_breakout"))
    for i in long_revert:
        i = int(i)
        row = sig.iloc[i]
        after = entry_after.iloc[i]
        if pd.isna(after):
            continue
        signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        direction = "long"
        if trend_ok(row, direction, "counter_ema") and regime_ok(row, direction, params["regime"]):
            stop = float(low_values[i]) - float(params["stop_buffer_atr"]) * float(atr_values[i])
            events.append(Event(direction, signal_time, after, stop, float(mid_values[i]), "bb_mean_reversion"))
    for i in short_revert:
        i = int(i)
        row = sig.iloc[i]
        after = entry_after.iloc[i]
        if pd.isna(after):
            continue
        signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        direction = "short"
        if trend_ok(row, direction, "counter_ema") and regime_ok(row, direction, params["regime"]):
            stop = float(high_values[i]) + float(params["stop_buffer_atr"]) * float(atr_values[i])
            events.append(Event(direction, signal_time, after, stop, float(mid_values[i]), "bb_mean_reversion"))
    return events


def build_intraday_timing_events(exec_frame: pd.DataFrame, params: dict[str, Any]) -> list[Event]:
    tf = normalize_timeframe(params["signal_tf"])
    sig = prepared_signal_frame(exec_frame, tf)
    lookback = int(params["lookback"])
    ret_atr = (sig["close"] - sig["close"].shift(lookback)) / sig["atr"].replace(0.0, np.nan)
    entry_after = entry_after_times(exec_frame, sig)
    mode = str(params["mode"])
    events: list[Event] = []
    atr_values = sig["atr"].to_numpy(dtype=float)
    impulse_values = ret_atr.to_numpy(dtype=float)
    valid = np.isfinite(atr_values) & (atr_values > 0.0) & np.isfinite(impulse_values)
    idxs = np.flatnonzero(valid & (np.abs(impulse_values) >= float(params["min_impulse_atr"])))
    for i in idxs:
        i = int(i)
        row = sig.iloc[i]
        impulse = float(impulse_values[i])
        signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        if not allowed_session(signal_time, str(params["session"])):
            continue
        after = entry_after.iloc[i]
        if pd.isna(after):
            continue
        if mode == "momentum":
            direction = "long" if impulse > 0 else "short"
        else:
            direction = "short" if impulse > 0 else "long"
        if not trend_ok(row, direction, params["trend_filter"]):
            continue
        stop_dist = float(params["stop_atr"]) * float(atr_values[i])
        stop = float(row["close"]) - stop_dist if direction == "long" else float(row["close"]) + stop_dist
        events.append(Event(direction, signal_time, after, stop, None, f"intraday_{mode}"))
    return events


EVENT_BUILDERS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], list[Event]]] = {
    "vol_breakout": build_vol_breakout_events,
    "sweep_reclaim": build_sweep_reclaim_events,
    "trend_pullback": build_trend_pullback_events,
    "bb_regime": build_bb_regime_events,
    "intraday_timing": build_intraday_timing_events,
}


def simulate_events(exec_frame: pd.DataFrame, events: list[Event], params: dict[str, Any], symbol: str) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    frame = ensure_ohlcv_frame(exec_frame)
    open_times = pd.to_datetime(frame["open_time"], utc=True).to_numpy(dtype="datetime64[ns]")
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    max_hold = int(params["max_hold_bars"])
    rr = float(params["rr"])
    fee_bps = float(params.get("fee_bps_side", 5.5))
    slippage_bps = float(params.get("slippage_bps_side", 1.0))
    min_risk_pct = float(params.get("min_entry_risk_pct", 0.05)) / 100.0
    max_risk_pct = float(params.get("max_entry_risk_pct", 4.0)) / 100.0
    one_at_a_time = bool(params.get("one_trade_at_a_time", True))

    rows: list[dict[str, Any]] = []
    next_available = 0
    seen: set[tuple[int, str]] = set()
    for event in sorted(events, key=lambda item: item.entry_after):
        entry_idx = int(np.searchsorted(open_times, np.datetime64(event.entry_after.to_datetime64()), side="left"))
        if entry_idx >= len(frame) - 2:
            continue
        key = (entry_idx, event.direction)
        if key in seen:
            continue
        seen.add(key)
        if one_at_a_time and entry_idx < next_available:
            continue
        entry = float(opens[entry_idx])
        stop = float(event.stop_anchor)
        if not (_finite(entry) and _finite(stop)):
            continue
        risk = abs(entry - stop)
        if risk <= 0.0:
            continue
        risk_pct = risk / entry
        if risk_pct < min_risk_pct or risk_pct > max_risk_pct:
            continue
        if event.direction == "long" and stop >= entry:
            continue
        if event.direction == "short" and stop <= entry:
            continue
        if event.target_anchor is not None and _finite(event.target_anchor):
            target = float(event.target_anchor)
            planned_rr = abs(target - entry) / risk
            if planned_rr < rr:
                continue
        else:
            planned_rr = rr
            target = entry + rr * risk if event.direction == "long" else entry - rr * risk
        if event.direction == "long" and target <= entry:
            continue
        if event.direction == "short" and target >= entry:
            continue
        exit_idx = min(entry_idx + max_hold, len(frame) - 1)
        exit_price = float(closes[exit_idx])
        exit_reason = "timeout"
        for j in range(entry_idx, exit_idx + 1):
            if event.direction == "long":
                if lows[j] <= stop:
                    exit_idx = j
                    exit_price = stop
                    exit_reason = "stop"
                    break
                if highs[j] >= target:
                    exit_idx = j
                    exit_price = target
                    exit_reason = "target"
                    break
            else:
                if highs[j] >= stop:
                    exit_idx = j
                    exit_price = stop
                    exit_reason = "stop"
                    break
                if lows[j] <= target:
                    exit_idx = j
                    exit_price = target
                    exit_reason = "target"
                    break
        gross_r = (exit_price - entry) / risk if event.direction == "long" else (entry - exit_price) / risk
        cost_r = ((2.0 * fee_bps) + (2.0 * slippage_bps)) / 10_000.0 * entry / risk
        net_r = gross_r - cost_r
        ret = (exit_price - entry) / entry if event.direction == "long" else (entry - exit_price) / entry
        ret -= ((2.0 * fee_bps) + (2.0 * slippage_bps)) / 10_000.0
        rows.append(
            {
                "symbol": symbol,
                "strategy": params["strategy"],
                "direction": event.direction,
                "context": event.context,
                "signal_time": event.signal_time,
                "entry_time": pd.Timestamp(open_times[entry_idx]).tz_localize("UTC"),
                "exit_time": pd.Timestamp(open_times[exit_idx]).tz_localize("UTC"),
                "entry_price": entry,
                "exit_price": exit_price,
                "stop_price": stop,
                "target_price": target,
                "target_rr_planned": planned_rr,
                "r_multiple_gross": gross_r,
                "r_multiple_net": net_r,
                "return_pct": ret,
                "hold_bars": exit_idx - entry_idx,
                "exit_reason": exit_reason,
            }
        )
        if one_at_a_time:
            next_available = exit_idx + 1
    return pd.DataFrame(rows)


def product_grid(options: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    keys = list(options)
    return [dict(zip(keys, values)) for values in itertools.product(*(options[key] for key in keys))]


def strategy_options(strategy: str) -> dict[str, tuple[Any, ...]]:
    common = {
        "fee_bps_side": (5.5,),
        "slippage_bps_side": (1.0,),
        "one_trade_at_a_time": (True,),
        "min_entry_risk_pct": (0.05,),
        "max_entry_risk_pct": (4.0,),
        "max_events_per_config": (900,),
    }
    if strategy == "vol_breakout":
        options = {
            "strategy": (strategy,),
            "signal_tf": ("15m", "1h"),
            "lookback": (12, 20, 36),
            "squeeze_lookback": (80, 120),
            "width_source": ("bb", "keltner"),
            "max_width_pctile": (0.20, 0.35),
            "break_atr": (0.0, 0.10, 0.20),
            "stop_buffer_atr": (0.10, 0.30),
            "rr": (1.5, 2.0, 3.0),
            "max_hold_bars": (48, 96, 192),
            "trend_filter": ("none", "ema", "ema_slope"),
            "regime": ("none", "high_vol", "trend_aligned"),
            **common,
        }
    elif strategy == "sweep_reclaim":
        sweep_common = {
            **common,
            "min_entry_risk_pct": (0.10, 0.30, 0.60),
            "max_events_per_config": (1800,),
        }
        options = {
            "strategy": (strategy,),
            "signal_tf": ("5m",),
            "level_source": ("htf_swing", "period"),
            "level_tf": ("4h", "1d", "1w"),
            "period_level": ("daily", "weekly", "monthly"),
            "pivot_window": (2, 3, 5),
            "lookback": (12,),
            "min_sweep_atr": (0.0, 0.15, 0.30, 0.50),
            "reclaim_atr": (0.0, 0.05, 0.15),
            "reclaim_bars": (0, 3, 6, 12),
            "level_lookback": (6, 12, 24, 48),
            "min_close_location": (0.50, 0.60, 0.70),
            "stop_buffer_atr": (0.15, 0.30, 0.60),
            "min_stop_buffer_pct": (0.02, 0.05),
            "rr": (1.2, 1.5, 2.0, 3.0),
            "max_hold_bars": (24, 48, 96, 192),
            "trend_filter": ("none", "rsi", "counter_ema"),
            "regime": ("none", "high_vol", "low_vol", "mean_reversion"),
            "session": ("all", "london_open", "ny_open", "killzones", "london_ny"),
            "direction": ("both", "long", "short"),
            **sweep_common,
        }
    elif strategy == "trend_pullback":
        options = {
            "strategy": (strategy,),
            "signal_tf": ("15m", "1h"),
            "trend_ema": (100, 200),
            "pullback_ema": (20, 50),
            "trigger": ("close_reclaim", "break_prev"),
            "min_slope_atr": (0.0, 0.05, 0.10),
            "stop_lookback": (3, 6, 12),
            "stop_buffer_atr": (0.10, 0.30),
            "rr": (1.5, 2.0, 3.0),
            "max_hold_bars": (48, 96, 192),
            **common,
        }
    elif strategy == "bb_regime":
        options = {
            "strategy": (strategy,),
            "signal_tf": ("15m", "1h"),
            "mode": ("breakout", "mean_reversion", "auto"),
            "length": (20, 40),
            "band_mult": (1.8, 2.0, 2.5),
            "width_lookback": (80, 120),
            "breakout_width_pctile": (0.50, 0.70),
            "reversion_width_pctile": (0.25, 0.40),
            "stop_buffer_atr": (0.10, 0.30),
            "rr": (1.2, 1.5, 2.0, 3.0),
            "max_hold_bars": (48, 96, 192),
            "trend_filter": ("none", "ema", "ema_slope"),
            "regime": ("none", "high_vol", "low_vol", "trend_aligned", "mean_reversion"),
            **common,
        }
    elif strategy == "intraday_timing":
        options = {
            "strategy": (strategy,),
            "signal_tf": ("15m", "1h"),
            "mode": ("momentum", "reversal"),
            "lookback": (1, 3, 6, 12),
            "min_impulse_atr": (0.20, 0.50, 1.00),
            "session": ("all", "asia", "london", "ny"),
            "trend_filter": ("none", "ema", "ema_slope", "counter_ema"),
            "stop_atr": (0.80, 1.20, 1.80),
            "rr": (1.2, 1.5, 2.0),
            "max_hold_bars": (48, 96, 192),
            **common,
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return options


def strategy_grid(strategy: str) -> list[dict[str, Any]]:
    return product_grid(strategy_options(strategy))


def sample_grid(rows: list[dict[str, Any]], max_configs: int, seed: int) -> list[dict[str, Any]]:
    if max_configs <= 0 or len(rows) <= max_configs:
        return rows
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=max_configs, replace=False)
    return [rows[int(i)] for i in indices]


def sample_strategy_grid(
    strategy: str,
    max_configs: int,
    seed: int,
    fixed_params: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    options = strategy_options(strategy)
    if fixed_params:
        options = dict(options)
        for key, values in fixed_params.items():
            options[key] = values
    keys = list(options)
    lengths = [len(options[key]) for key in keys]
    total = math.prod(lengths)
    if max_configs <= 0 or total <= max_configs:
        return product_grid(options)

    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    while len(selected) < max_configs:
        selected.add(int(rng.integers(0, total)))

    rows: list[dict[str, Any]] = []
    for flat_idx in sorted(selected):
        cursor = flat_idx
        row: dict[str, Any] = {}
        for key, length in zip(reversed(keys), reversed(lengths), strict=False):
            choice_idx = cursor % length
            cursor //= length
            row[key] = options[key][choice_idx]
        rows.append({key: row[key] for key in keys})
    return rows


def normalized_params_for_scoring(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    if out.get("strategy") == "sweep_reclaim":
        level_source = str(out.get("level_source", ""))
        if level_source == "period":
            out["level_tf"] = ""
            out["pivot_window"] = 0
            out["level_lookback"] = 0
            out["lookback"] = 0
        elif level_source == "htf_swing":
            out["period_level"] = ""
            out["lookback"] = 0
    return out


def evaluate_config(frame: pd.DataFrame, symbol: str, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = normalized_params_for_scoring(params)
    events = EVENT_BUILDERS[params["strategy"]](frame, params)
    max_events = int(params.get("max_events_per_config", 900))
    if len(events) > max_events:
        empty = pd.DataFrame()
        empty_m = strategy_metrics(empty)
        penalty = -1_000.0 - float(len(events) - max_events) * 0.10
        row = {
            **params,
            "symbol": symbol,
            "event_count": float(len(events)),
            "overactive": True,
            "selection_pass": False,
            "oos_pass": False,
            "robust_score": penalty,
            **{f"train_{key}": value for key, value in empty_m.items()},
            **{f"validation_{key}": value for key, value in empty_m.items()},
            **{f"oos_{key}": value for key, value in empty_m.items()},
            **{f"all_{key}": value for key, value in empty_m.items()},
        }
        return empty, row
    trades = simulate_events(frame, events, params, symbol)
    start = pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC")
    end = pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC")
    train_end = start + (end - start) * 0.60
    validation_end = start + (end - start) * 0.80
    buckets = split_trades(trades, train_end=train_end, validation_end=validation_end)
    train_m = strategy_metrics(buckets["train"])
    val_m = strategy_metrics(buckets["validation"])
    oos_m = strategy_metrics(buckets["oos"])
    all_m = strategy_metrics(trades)

    def capped_pf(metric: dict[str, float]) -> float:
        value = float(metric["profit_factor"])
        if not math.isfinite(value):
            return 5.0
        return min(value, 5.0)

    selection_pass = (
        train_m["trades"] >= 20
        and val_m["trades"] >= 8
        and train_m["net_r"] > 0.0
        and val_m["net_r"] > 0.0
        and train_m["profit_factor"] >= 1.10
        and val_m["profit_factor"] >= 1.10
        and min(train_m["avg_r"], val_m["avg_r"]) > 0.02
    )
    oos_pass = (
        oos_m["trades"] >= 8
        and oos_m["net_r"] > 0.0
        and oos_m["profit_factor"] >= 1.05
        and oos_m["avg_r"] > 0.0
    )
    robust_score = (
        train_m["net_r"]
        + val_m["net_r"]
        + min(capped_pf(train_m), capped_pf(val_m)) * 4.0
        + min(train_m["avg_r"], val_m["avg_r"]) * 20.0
        + math.sqrt(max(min(train_m["trades"], val_m["trades"]), 0.0))
        - abs(min(train_m["max_dd_r"], val_m["max_dd_r"])) * 0.25
    )
    if all_m["trades"] <= 0:
        robust_score = -500.0
    row = {
        **params,
        "symbol": symbol,
        "event_count": float(len(events)),
        "overactive": False,
        "selection_pass": bool(selection_pass),
        "oos_pass": bool(oos_pass),
        "robust_score": float(robust_score),
        **metric_row(buckets["train"], "train"),
        **metric_row(buckets["validation"], "validation"),
        **metric_row(buckets["oos"], "oos"),
        **metric_row(trades, "all"),
    }
    return trades, row


def finite_array(values: pd.Series, cap: float | None = None) -> np.ndarray:
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = out[np.isfinite(out)]
    fill = float(np.nanmedian(finite)) if finite.size else 0.0
    out = np.where(np.isfinite(out), out, fill)
    if cap is not None:
        out = np.clip(out, -cap, cap)
    return out


def lowpass_scores(scores: pd.DataFrame, *, radius: float, min_neighbors: int, outlier_penalty: float) -> pd.DataFrame:
    if scores.empty:
        return scores
    out = scores.copy().reset_index(drop=True)
    param_cols = [
        col
        for col in out.columns
        if col
        not in {
            "symbol",
            "event_count",
            "overactive",
            "selection_pass",
            "oos_pass",
            *METRIC_COLUMNS,
        }
        and not col.endswith("_net_r")
        and not col.endswith("_avg_r")
        and not col.endswith("_trades")
        and not col.endswith("_profit_factor")
        and not col.endswith("_max_dd_r")
        and not col.endswith("_total_return")
        and not col.endswith("_sharpe")
        and not col.endswith("_win_rate")
    ]
    n = len(out)
    d2 = np.zeros((n, n), dtype=float)
    for col in param_cols:
        values = out[col]
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if len(finite) and values.astype(str).nunique() > 2:
            arr = numeric.to_numpy(dtype=float)
            fill = float(np.nanmedian(finite))
            arr = np.where(np.isfinite(arr), arr, fill)
            span = float(np.nanmax(arr) - np.nanmin(arr))
            if span > 0.0:
                delta = (arr[:, None] - arr[None, :]) / span
                d2 += delta * delta
        else:
            arr = values.astype(str).to_numpy()
            d2 += 0.75 * (arr[:, None] != arr[None, :]).astype(float)

    evaluated = pd.to_numeric(out["robust_score"], errors="coerce").notna().to_numpy()
    if not evaluated.any():
        return out
    evaluated_idx = np.flatnonzero(evaluated)
    min_neighbors = min(max(int(min_neighbors), 1), int(evaluated.sum()))
    radius2 = float(radius) ** 2
    pass_mask = out["selection_pass"].fillna(False).astype(bool).to_numpy()
    metric_values = {
        metric: finite_array(out[metric], cap=10.0 if metric.endswith("profit_factor") else None)
        for metric in METRIC_COLUMNS
        if metric in out.columns
    }
    neighbor_counts = np.zeros(n, dtype=int)
    local_pass_rates = np.full(n, np.nan, dtype=float)
    lowpass_values = {metric: np.full(n, np.nan, dtype=float) for metric in metric_values}
    for row_idx in range(n):
        if not evaluated[row_idx]:
            continue
        local = (d2[row_idx] <= radius2) & evaluated
        if int(local.sum()) < min_neighbors:
            nearest = evaluated_idx[np.argsort(d2[row_idx, evaluated_idx])[:min_neighbors]]
            local[nearest] = True
        neighbor_counts[row_idx] = int(local.sum())
        local_pass_rates[row_idx] = float(pass_mask[local].mean()) if local.any() else math.nan
        for metric, values in metric_values.items():
            lowpass_values[metric][row_idx] = float(np.median(values[local]))
    for metric, values in lowpass_values.items():
        out[f"lowpass_{metric}"] = values
    raw = out["robust_score"].to_numpy(dtype=float)
    low = out["lowpass_robust_score"].to_numpy(dtype=float)
    out["local_neighbor_count"] = neighbor_counts
    out["local_pass_rate"] = local_pass_rates
    out["lowpass_outlier_gap"] = np.maximum(raw - low, 0.0)
    out["optimization_score"] = low
    out["stability_score"] = low - float(outlier_penalty) * out["lowpass_outlier_gap"].to_numpy(dtype=float)
    return out.sort_values(
        ["selection_pass", "stability_score", "optimization_score", "local_pass_rate", "robust_score"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def evaluate_strategy_for_symbol(
    frame: pd.DataFrame,
    *,
    symbol: str,
    strategy: str,
    max_configs: int,
    seed: int,
    lowpass_radius: float,
    lowpass_min_neighbors: int,
    lowpass_outlier_penalty: float,
    max_events_per_config: int,
    fixed_params: dict[str, tuple[str, ...]] | None = None,
    trades_dir: Path | None = None,
) -> pd.DataFrame:
    grid = sample_strategy_grid(strategy, max_configs=max_configs, seed=seed, fixed_params=fixed_params)
    rows: list[dict[str, Any]] = []
    best_trades: pd.DataFrame | None = None
    best_score = -math.inf
    for idx, params in enumerate(grid, start=1):
        params = {**params, "max_events_per_config": int(max_events_per_config)}
        try:
            trades, row = evaluate_config(frame, symbol, params)
        except Exception as exc:  # noqa: BLE001 - research sweeps should continue through bad configs.
            row = {**params, "symbol": symbol, "grid_index": idx, "error": str(exc), "robust_score": math.nan}
            trades = pd.DataFrame()
        row["grid_index"] = idx
        rows.append(row)
        score = float(row.get("robust_score", -math.inf))
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_trades = trades
    table = pd.DataFrame(rows)
    table = lowpass_scores(
        table,
        radius=lowpass_radius,
        min_neighbors=lowpass_min_neighbors,
        outlier_penalty=lowpass_outlier_penalty,
    )
    if trades_dir is not None and best_trades is not None:
        trades_dir.mkdir(parents=True, exist_ok=True)
        best_trades.to_csv(trades_dir / f"{symbol.lower()}_{strategy}_best_raw_trades.csv", index=False)
    return table


def summarize_table(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {}
    selection = table[table["selection_pass"].fillna(False)].copy() if "selection_pass" in table.columns else pd.DataFrame()
    both = selection[selection["oos_pass"].fillna(False)].copy() if not selection.empty and "oos_pass" in selection.columns else pd.DataFrame()
    winner = (both if not both.empty else selection if not selection.empty else table).iloc[0]
    return {
        "symbol": winner.get("symbol"),
        "strategy": winner.get("strategy"),
        "candidate_source": "selection+oos" if not both.empty else "selection" if not selection.empty else "best_raw",
        "selection_pass_count": int(selection.shape[0]),
        "oos_pass_count": int(both.shape[0]),
        **{key: winner[key] for key in winner.index if key not in {"error"}},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research five structural crypto strategies with low-pass stability scoring.")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,LTCUSDT,SOLUSDT,XRPUSDT,UNIUSDT,LINKUSDT",
        help="Comma-separated Bybit symbols.",
    )
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--end")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/structural_strategy_research"))
    parser.add_argument("--max-configs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=940)
    parser.add_argument("--lowpass-radius", type=float, default=0.85)
    parser.add_argument("--lowpass-min-neighbors", type=int, default=9)
    parser.add_argument("--lowpass-outlier-penalty", type=float, default=0.65)
    parser.add_argument("--max-events-per-config", type=int, default=900)
    parser.add_argument(
        "--fixed-param",
        action="append",
        default=[],
        help="Restrict grid values, e.g. --fixed-param level_source=htf_swing --fixed-param level_tf=4h. Use commas for multiple values.",
    )
    parser.add_argument("--save-best-trades", action="store_true")
    return parser.parse_args()


def parse_fixed_params(items: list[str]) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--fixed-param expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        values = tuple(value.strip() for value in raw.split(",") if value.strip())
        if not key.strip() or not values:
            raise ValueError(f"--fixed-param expects KEY=VALUE, got {item!r}")
        out[key.strip()] = values
    return out


def main() -> None:
    args = parse_args()
    symbols = [bybit_symbol(symbol) for symbol in comma_values(args.symbols, [])]
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    fixed_params = parse_fixed_params(args.fixed_param)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []
    run_meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "strategies": strategies,
        "days": args.days,
        "max_configs": args.max_configs,
        "lowpass_radius": args.lowpass_radius,
        "lowpass_min_neighbors": args.lowpass_min_neighbors,
        "fixed_params": fixed_params,
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    for strategy in strategies:
        print(f"\n=== {strategy}: {STRATEGIES[strategy]} ===", flush=True)
        strategy_tables: list[pd.DataFrame] = []
        strategy_summaries: list[dict[str, Any]] = []
        for symbol_idx, symbol in enumerate(symbols, start=1):
            frame = load_symbol_frame(symbol, interval=args.interval, days=args.days, cache_dir=args.cache_dir, end=args.end)
            if frame.empty:
                print(f"{symbol}: no data", flush=True)
                continue
            table = evaluate_strategy_for_symbol(
                frame,
                symbol=symbol,
                strategy=strategy,
                max_configs=args.max_configs,
                seed=args.seed + symbol_idx * 17,
                lowpass_radius=args.lowpass_radius,
                lowpass_min_neighbors=args.lowpass_min_neighbors,
                lowpass_outlier_penalty=args.lowpass_outlier_penalty,
                max_events_per_config=args.max_events_per_config,
                fixed_params=fixed_params,
                trades_dir=args.output_dir / "trades" if args.save_best_trades else None,
            )
            strategy_tables.append(table)
            summary = summarize_table(table)
            strategy_summaries.append(summary)
            all_summaries.append(summary)
            out_path = args.output_dir / f"{strategy}_{symbol.lower()}.tuning.csv"
            table.to_csv(out_path, index=False)
            if summary:
                print(
                    f"{symbol}: {summary.get('candidate_source')} "
                    f"opt={float(summary.get('optimization_score', 0.0)):+.2f} "
                    f"all={float(summary.get('all_net_r', 0.0)):+.2f}R/{float(summary.get('all_trades', 0.0)):.0f} "
                    f"oos={float(summary.get('oos_net_r', 0.0)):+.2f}R/{float(summary.get('oos_trades', 0.0)):.0f} "
                    f"pass={summary.get('selection_pass_count')}/{summary.get('oos_pass_count')}",
                    flush=True,
                )
        if strategy_tables:
            combined = pd.concat(strategy_tables, ignore_index=True)
            combined = combined.sort_values(
                ["selection_pass", "stability_score", "optimization_score", "oos_net_r", "all_net_r"],
                ascending=[False, False, False, False, False],
                na_position="last",
            )
            combined.to_csv(args.output_dir / f"{strategy}_combined.tuning.csv", index=False)
        pd.DataFrame(strategy_summaries).to_csv(args.output_dir / f"{strategy}_summary.csv", index=False)
    summary_frame = pd.DataFrame(all_summaries)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(
            ["candidate_source", "stability_score", "optimization_score", "oos_net_r"],
            ascending=[True, False, False, False],
            na_position="last",
        )
    summary_frame.to_csv(args.output_dir / "all_strategy_summary.csv", index=False)
    print(f"\nWrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
