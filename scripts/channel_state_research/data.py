from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest_turtle_soup import (
    add_atr,
    fetch_klines,
    normalize_binance_spot_symbol,
    normalize_timeframe,
    parse_utc_datetime,
    resample_ohlc,
)


@dataclass(frozen=True)
class MarketDataset:
    symbol: str
    source_interval: str
    base_frame: pd.DataFrame
    bars_by_timeframe: dict[str, pd.DataFrame]


def _to_utc_datetime(value: str | datetime | pd.Timestamp, *, is_end: bool = False) -> datetime:
    if isinstance(value, str):
        dt = parse_utc_datetime(value)
        if is_end and len(value.strip()) == 10:
            dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
        return dt
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
        return ts.to_pydatetime()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def ensure_cache(
    symbol: str,
    interval: str,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    requested_symbol = normalize_binance_spot_symbol(symbol).lower()
    requested_interval = normalize_timeframe(interval)
    start_dt = _to_utc_datetime(start)
    end_dt = _to_utc_datetime(end, is_end=True)

    for candidate in sorted(cache_dir.glob(f"{requested_symbol}_{requested_interval}_*.pkl")):
        try:
            frame = pd.read_pickle(candidate)
        except Exception:
            continue
        if frame.empty:
            continue
        first_open = pd.Timestamp(frame["open_time"].iloc[0]).to_pydatetime()
        last_close = pd.Timestamp(frame["close_time"].iloc[-1]).to_pydatetime()
        if first_open <= start_dt and last_close >= end_dt:
            return candidate

    cache_path = cache_dir / f"{requested_symbol}_{requested_interval}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.pkl"
    if cache_path.exists():
        return cache_path

    frame = fetch_klines(symbol, requested_interval, _to_ms(start_dt), _to_ms(end_dt))
    frame.to_pickle(cache_path)
    return cache_path


def load_base_candles(
    symbol: str,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    cache_dir: Path = Path("scripts/.cache"),
    interval: str = "5m",
) -> pd.DataFrame:
    cache_path = ensure_cache(symbol, interval, start, end, cache_dir)
    frame = pd.read_pickle(cache_path).sort_values("open_time").reset_index(drop=True).copy()
    start_ts = pd.Timestamp(_to_utc_datetime(start)).tz_convert("UTC")
    end_ts = pd.Timestamp(_to_utc_datetime(end, is_end=True)).tz_convert("UTC")
    mask = (frame["open_time"] >= start_ts) & (frame["close_time"] <= end_ts)
    out = frame.loc[mask].reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No candles available for {symbol} {interval} between {start_ts} and {end_ts}.")
    return out


def prepare_timeframe_bars(base_frame: pd.DataFrame, timeframe: str, atr_length: int = 14) -> pd.DataFrame:
    normalized = normalize_timeframe(timeframe)
    bars = resample_ohlc(base_frame, normalized).sort_values("open_time").reset_index(drop=True).copy()
    bars = add_atr(bars, atr_length)
    bars["body_high"] = bars[["open", "close"]].max(axis=1)
    bars["body_low"] = bars[["open", "close"]].min(axis=1)
    bars["return_1"] = bars["close"].pct_change()
    bars["log_return_1"] = np.log(bars["close"]).diff()
    bars["bar_index"] = np.arange(len(bars), dtype=float)
    bars["timeframe"] = normalized
    return bars


def build_market_dataset(
    symbol: str,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    timeframes: list[str],
    cache_dir: Path = Path("scripts/.cache"),
    base_interval: str = "5m",
    atr_length: int = 14,
) -> MarketDataset:
    base = load_base_candles(symbol, start, end, cache_dir=cache_dir, interval=base_interval)
    bars_by_timeframe = {
        normalize_timeframe(timeframe): prepare_timeframe_bars(base, timeframe, atr_length=atr_length)
        for timeframe in timeframes
    }
    return MarketDataset(
        symbol=normalize_binance_spot_symbol(symbol),
        source_interval=normalize_timeframe(base_interval),
        base_frame=base,
        bars_by_timeframe=bars_by_timeframe,
    )


def merge_asof_timeframe_state(
    decision_frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    suffix = f"_{normalize_timeframe(timeframe)}"
    renamed = state_frame.copy()
    rename_map = {column: f"{column}{suffix}" for column in renamed.columns if column != "close_time"}
    renamed = renamed.rename(columns=rename_map)
    return pd.merge_asof(
        decision_frame.sort_values("close_time"),
        renamed.sort_values("close_time"),
        on="close_time",
        direction="backward",
    )
