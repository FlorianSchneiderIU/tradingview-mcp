#!/usr/bin/env python3
"""Download Bybit BTCUSDT candles and score hourly ATR RR trades by daily regime.

The script fetches public Bybit V5 klines, classifies each day into one of four
daily regimes, then simulates independent hourly long/short 1:1 RR trades with
a stop distance of 2 * hourly ATR. Results are grouped into hour-of-day
histograms per regime, entry mode, and direction.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


BYBIT_URL = "https://api.bybit.com"
INTERVAL_MS = {
    "1h": 3_600_000,
    "1d": 86_400_000,
}
BYBIT_INTERVALS = {
    "1h": "60",
    "1d": "D",
}
REGIMES = ("long_trending", "short_trending", "sideways", "chop")
ENTRY_MODES = ("open", "close")
DIRECTIONS = ("long", "short")
TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "entry_hour_utc",
    "entry_mode",
    "direction",
    "entry_price",
    "stop",
    "target",
    "exit_price",
    "risk",
    "atr",
    "outcome",
    "r_multiple",
    "holding_hours",
]


@dataclass(frozen=True)
class Period:
    start: pd.Timestamp
    end: pd.Timestamp


def parse_utc_datetime(value: str) -> pd.Timestamp:
    text = value.strip().replace("Z", "+00:00")
    if len(text) == 10:
        text = f"{text}T00:00:00+00:00"
    ts = pd.Timestamp(datetime.fromisoformat(text))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def resolve_period(args: argparse.Namespace) -> Period:
    if args.end:
        end = parse_utc_datetime(args.end)
    else:
        end = pd.Timestamp.now(tz="UTC").floor("h")

    if args.start:
        start = parse_utc_datetime(args.start)
    else:
        start = end - pd.DateOffset(years=args.years)

    if start >= end:
        raise ValueError(f"Start must be before end, got start={start} end={end}")
    return Period(start=start, end=end)


def bybit_symbol(raw: str) -> str:
    symbol = raw.strip().upper()
    if ":" in symbol:
        symbol = symbol.split(":", 1)[-1]
    return symbol.replace("/", "").replace("-", "").replace(".P", "")


def normalize_interval(raw: str) -> str:
    value = raw.strip().lower()
    aliases = {"60": "1h", "d": "1d", "1d": "1d", "1h": "1h"}
    if value not in aliases:
        raise ValueError(f"Unsupported interval {raw!r}; expected 1h or 1d")
    return aliases[value]


def _cache_path(cache_dir: Path, symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    safe_symbol = bybit_symbol(symbol).lower()
    safe_start = start.strftime("%Y%m%d%H%M")
    safe_end = end.strftime("%Y%m%d%H%M")
    return cache_dir / f"{safe_symbol}_{interval}_{safe_start}_{safe_end}.csv"


def fetch_bybit_klines(
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    category: str = "linear",
    base_url: str = BYBIT_URL,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    sleep_seconds: float = 0.08,
) -> pd.DataFrame:
    """Fetch closed Bybit V5 klines, oldest to newest."""
    symbol = bybit_symbol(symbol)
    interval = normalize_interval(interval)
    interval_ms = INTERVAL_MS[interval]
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, symbol, interval, start, end)
        if path.exists() and not force_refresh:
            return read_ohlcv_csv(path)

    start_ms = int(start.timestamp() * 1000)
    cursor_end_ms = int(end.timestamp() * 1000)
    rows: dict[int, list[Any]] = {}
    session = requests.Session()

    while cursor_end_ms >= start_ms:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": BYBIT_INTERVALS[interval],
            "start": start_ms,
            "end": cursor_end_ms,
            "limit": 1000,
        }
        batch = None
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = session.get(f"{base_url.rstrip('/')}/v5/market/kline", params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                ret_code = payload.get("retCode", 0)
                if ret_code not in (0, "0"):
                    raise RuntimeError(f"Bybit retCode={ret_code} retMsg={payload.get('retMsg')}")
                batch = payload.get("result", {}).get("list", [])
                break
            except Exception as exc:  # noqa: BLE001 - retry public market data fetches.
                last_error = exc
                time.sleep(0.5 * (attempt + 1))

        if batch is None:
            raise last_error if last_error is not None else RuntimeError("Failed to fetch Bybit klines")
        if not batch:
            break

        oldest = min(int(row[0]) for row in batch)
        for row in batch:
            rows[int(row[0])] = row
        if len(batch) < 1000 or oldest <= start_ms:
            break
        cursor_end_ms = oldest - 1
        time.sleep(sleep_seconds)

    if not rows:
        raise RuntimeError(f"No Bybit klines returned for {symbol} {interval}")

    frame = pd.DataFrame(
        [rows[key] for key in sorted(rows)],
        columns=["open_time_ms", "open", "high", "low", "close", "volume", "turnover"],
    )
    frame["open_time"] = pd.to_datetime(frame["open_time_ms"].astype("int64"), unit="ms", utc=True)
    frame["close_time"] = frame["open_time"] + pd.Timedelta(milliseconds=interval_ms - 1)
    for column in ("open", "high", "low", "close", "volume", "turnover"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)

    now = pd.Timestamp.now(tz="UTC")
    candle_end = frame["open_time"] + pd.Timedelta(milliseconds=interval_ms)
    frame = frame[(frame["open_time"] >= start) & (frame["open_time"] < end) & (candle_end <= now)].copy()
    out = frame[["open_time", "close_time", "open", "high", "low", "close", "volume", "turnover"]]
    out = out.dropna().sort_values("open_time").reset_index(drop=True)

    if cache_dir is not None:
        out.to_csv(path, index=False)
    return out


def read_ohlcv_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    for column in ("open", "high", "low", "close", "volume", "turnover"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame.dropna(subset=["open_time", "close_time", "open", "high", "low", "close"]).reset_index(drop=True)


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    parts = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def add_atr(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    out = frame.copy()
    out[f"atr_{length}"] = rma(true_range(out), length)
    return out


def add_adx(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    out = frame.copy()
    up_move = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    atr = rma(true_range(out), length)
    plus_di = 100.0 * rma(plus_dm, length) / atr.replace(0, np.nan)
    minus_di = 100.0 * rma(minus_dm, length) / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out[f"plus_di_{length}"] = plus_di
    out[f"minus_di_{length}"] = minus_di
    out[f"adx_{length}"] = rma(dx, length)
    return out


def rolling_last_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def percentile(values: np.ndarray) -> float:
        last = values[-1]
        if not np.isfinite(last):
            return np.nan
        clean = values[np.isfinite(values)]
        if clean.size == 0:
            return np.nan
        return float(np.mean(clean <= last))

    return series.rolling(window, min_periods=min_periods).apply(percentile, raw=True)


def choppiness_index(frame: pd.DataFrame, length: int) -> pd.Series:
    tr_sum = true_range(frame).rolling(length, min_periods=length).sum()
    high_max = frame["high"].rolling(length, min_periods=length).max()
    low_min = frame["low"].rolling(length, min_periods=length).min()
    denom = (high_max - low_min).replace(0, np.nan)
    ratio = tr_sum / denom
    return 100.0 * np.log10(ratio.replace(0, np.nan)) / math.log10(length)


def classify_daily_regimes(
    daily: pd.DataFrame,
    *,
    adx_length: int,
    atr_length: int,
    trend_adx: float,
    sideways_adx: float,
    sideways_bbw_pctile: float,
    sideways_atr_pctile: float,
    chop_threshold: float,
) -> pd.DataFrame:
    out = daily.copy().sort_values("open_time").reset_index(drop=True)
    out = add_adx(out, adx_length)
    out = add_atr(out, atr_length)
    out["ema_50"] = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema_200"] = out["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    out["ema_50_slope_10d"] = out["ema_50"].pct_change(10)
    mid = out["close"].rolling(20, min_periods=20).mean()
    stdev = out["close"].rolling(20, min_periods=20).std(ddof=0)
    out["bb_width"] = (4.0 * stdev) / mid.replace(0, np.nan)
    out["bb_width_pctile_100d"] = rolling_last_percentile(out["bb_width"], 100, 30)
    out["atr_pctile_100d"] = rolling_last_percentile(out[f"atr_{atr_length}"], 100, 30)
    out["chop_14"] = choppiness_index(out, 14)

    adx = out[f"adx_{adx_length}"]
    long_trend = (
        (adx >= trend_adx)
        & (out["close"] > out["ema_50"])
        & (out["ema_50"] > out["ema_200"])
        & (out["ema_50_slope_10d"] > 0)
    )
    short_trend = (
        (adx >= trend_adx)
        & (out["close"] < out["ema_50"])
        & (out["ema_50"] < out["ema_200"])
        & (out["ema_50_slope_10d"] < 0)
    )
    sideways = (
        ~(long_trend | short_trend)
        & (adx <= sideways_adx)
        & (out["bb_width_pctile_100d"] <= sideways_bbw_pctile)
        & (out["atr_pctile_100d"] <= sideways_atr_pctile)
        & (out["chop_14"] < chop_threshold)
    )
    chop = ~(long_trend | short_trend | sideways)

    out["regime"] = np.select(
        [long_trend, short_trend, sideways, chop],
        ["long_trending", "short_trending", "sideways", "chop"],
        default="chop",
    )
    out["regime_available_time"] = out["open_time"] + pd.Timedelta(days=1)
    return out


def trim_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, time_column: str = "open_time") -> pd.DataFrame:
    if frame.empty or time_column not in frame.columns:
        return frame.copy()
    mask = (frame[time_column] >= start) & (frame[time_column] < end)
    return frame.loc[mask].reset_index(drop=True).copy()


def build_hourly_features(hourly: pd.DataFrame, atr_length: int) -> pd.DataFrame:
    out = add_atr(hourly.sort_values("open_time").reset_index(drop=True), atr_length)
    out["atr_for_open_entry"] = out[f"atr_{atr_length}"].shift(1)
    out["atr_for_close_entry"] = out[f"atr_{atr_length}"]
    return out


def _trade_levels(direction: str, entry: float, risk: float, rr: float) -> tuple[float, float]:
    if direction == "long":
        return entry - risk, entry + risk * rr
    if direction == "short":
        return entry + risk, entry - risk * rr
    raise ValueError(f"Unknown direction {direction!r}")


def _bar_hits(direction: str, high: float, low: float, stop: float, target: float) -> tuple[bool, bool]:
    if direction == "long":
        return low <= stop, high >= target
    return high >= stop, low <= target


def _r_multiple(direction: str, entry: float, exit_price: float, risk: float) -> float:
    side = 1.0 if direction == "long" else -1.0
    return side * (exit_price - entry) / risk


def simulate_trades(
    hourly: pd.DataFrame,
    *,
    entry_mode: str,
    direction: str,
    atr_length: int,
    sl_atr_multiple: float,
    rr: float,
    tie_policy: str,
    max_hold_hours: int,
) -> pd.DataFrame:
    if entry_mode not in ENTRY_MODES:
        raise ValueError(f"Unsupported entry mode {entry_mode}")
    if direction not in DIRECTIONS:
        raise ValueError(f"Unsupported direction {direction}")

    frame = hourly.sort_values("open_time").reset_index(drop=True)
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    open_times = frame["open_time"].to_numpy()
    close_times = frame["open_time"] + pd.Timedelta(hours=1)
    atr_col = "atr_for_open_entry" if entry_mode == "open" else "atr_for_close_entry"
    atrs = frame[atr_col].to_numpy(float)
    rows: list[dict[str, Any]] = []
    n = len(frame)

    for i in range(n):
        if entry_mode == "open":
            entry_price = opens[i]
            entry_time = utc_timestamp(open_times[i])
            first_check = i
        else:
            if i + 1 >= n:
                continue
            entry_price = closes[i]
            entry_time = close_times.iloc[i]
            first_check = i + 1

        atr = atrs[i]
        risk = float(atr) * sl_atr_multiple
        if not np.isfinite(entry_price) or not np.isfinite(risk) or risk <= 0:
            continue

        stop, target = _trade_levels(direction, entry_price, risk, rr)
        max_j = n - 1
        if max_hold_hours > 0:
            max_j = min(max_j, first_check + max_hold_hours - 1)

        outcome = "unresolved"
        exit_idx = max_j
        exit_price = closes[max_j]
        result_r = _r_multiple(direction, entry_price, exit_price, risk)

        for j in range(first_check, max_j + 1):
            hit_stop, hit_target = _bar_hits(direction, highs[j], lows[j], stop, target)
            if hit_stop and hit_target:
                exit_idx = j
                if tie_policy == "target":
                    outcome = "target"
                    exit_price = target
                    result_r = rr
                elif tie_policy == "skip":
                    outcome = "ambiguous"
                    exit_price = np.nan
                    result_r = np.nan
                else:
                    outcome = "stop"
                    exit_price = stop
                    result_r = -1.0
                break
            if hit_target:
                outcome = "target"
                exit_idx = j
                exit_price = target
                result_r = rr
                break
            if hit_stop:
                outcome = "stop"
                exit_idx = j
                exit_price = stop
                result_r = -1.0
                break

        if outcome == "unresolved" and max_hold_hours > 0:
            outcome = "timeout"

        exit_time = frame["open_time"].iloc[exit_idx] + pd.Timedelta(hours=1)
        holding_hours = max(0.0, (exit_time - entry_time).total_seconds() / 3600.0)
        rows.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_hour_utc": int(entry_time.hour),
                "entry_mode": entry_mode,
                "direction": direction,
                "entry_price": entry_price,
                "stop": stop,
                "target": target,
                "exit_price": exit_price,
                "risk": risk,
                "atr": atr,
                "outcome": outcome,
                "r_multiple": result_r,
                "holding_hours": holding_hours,
            }
        )

    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def assign_regimes(trades: pd.DataFrame, daily_regimes: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    fixed_cols = {
        "regime_available_time",
        "regime",
        "ema_50",
        "ema_200",
        "ema_50_slope_10d",
        "bb_width",
        "bb_width_pctile_100d",
        "atr_pctile_100d",
        "chop_14",
    }
    indicator_prefixes = ("adx_", "plus_di_", "minus_di_", "atr_")
    available_cols = [
        column
        for column in daily_regimes.columns
        if column in fixed_cols or column.startswith(indicator_prefixes)
    ]
    state = daily_regimes[available_cols].dropna(subset=["regime_available_time", "regime"]).sort_values("regime_available_time")
    merged = pd.merge_asof(
        trades.sort_values("entry_time"),
        state,
        left_on="entry_time",
        right_on="regime_available_time",
        direction="backward",
    )
    return merged.dropna(subset=["regime"]).reset_index(drop=True)


def summarize_histogram(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        by_key: dict[tuple[str, str, str, int], pd.DataFrame] = {}
    else:
        grouped = trades.groupby(["regime", "entry_mode", "direction", "entry_hour_utc"], dropna=False)
        by_key = {key: group for key, group in grouped}

    for regime in REGIMES:
        for entry_mode in ENTRY_MODES:
            for direction in DIRECTIONS:
                for hour in range(24):
                    group = by_key.get((regime, entry_mode, direction, hour))
                    if group is None or group.empty:
                        rows.append(
                            {
                                "regime": regime,
                                "entry_mode": entry_mode,
                                "direction": direction,
                                "entry_hour_utc": hour,
                                "trades": 0,
                                "wins": 0,
                                "losses": 0,
                                "ambiguous": 0,
                                "timeouts": 0,
                                "unresolved": 0,
                                "win_rate": np.nan,
                                "loss_rate": np.nan,
                                "avg_r": np.nan,
                                "median_r": np.nan,
                                "total_r": 0.0,
                                "avg_holding_hours": np.nan,
                            }
                        )
                        continue

                    clean_r = pd.to_numeric(group["r_multiple"], errors="coerce")
                    trades_count = int(len(group))
                    wins = int((group["outcome"] == "target").sum())
                    losses = int((group["outcome"] == "stop").sum())
                    rows.append(
                        {
                            "regime": regime,
                            "entry_mode": entry_mode,
                            "direction": direction,
                            "entry_hour_utc": hour,
                            "trades": trades_count,
                            "wins": wins,
                            "losses": losses,
                            "ambiguous": int((group["outcome"] == "ambiguous").sum()),
                            "timeouts": int((group["outcome"] == "timeout").sum()),
                            "unresolved": int((group["outcome"] == "unresolved").sum()),
                            "win_rate": wins / trades_count,
                            "loss_rate": losses / trades_count,
                            "avg_r": float(clean_r.mean()) if clean_r.notna().any() else np.nan,
                            "median_r": float(clean_r.median()) if clean_r.notna().any() else np.nan,
                            "total_r": float(clean_r.sum(skipna=True)),
                            "avg_holding_hours": float(group["holding_hours"].mean()),
                        }
                    )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Bybit BTCUSDT 1h and 1d candles, classify daily regimes, and "
            "build hourly histograms for ATR-based 1:1 RR long/short trades."
        )
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Bybit symbol, e.g. BTCUSDT")
    parser.add_argument("--category", default="linear", help="Bybit product category, default: linear")
    parser.add_argument("--base-url", default=BYBIT_URL, help="Bybit REST base URL")
    parser.add_argument("--years", type=int, default=5, help="Lookback years when --start is omitted")
    parser.add_argument("--start", help="UTC start date/time, e.g. 2021-05-25 or 2021-05-25T00:00:00Z")
    parser.add_argument("--end", help="UTC end date/time, defaults to current UTC hour")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/bybit_regime_histogram"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/output/bybit_regime_histogram"))
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached kline CSVs")
    parser.add_argument("--daily-warmup-days", type=int, default=300, help="Extra daily candles for indicators")
    parser.add_argument("--hourly-warmup-hours", type=int, default=250, help="Extra hourly candles for ATR")
    parser.add_argument("--atr-length", type=int, default=14, help="Hourly ATR length for trade stops")
    parser.add_argument("--daily-adx-length", type=int, default=14, help="Daily ADX length for regime classification")
    parser.add_argument("--daily-atr-length", type=int, default=14, help="Daily ATR length for regime classification")
    parser.add_argument("--trend-adx", type=float, default=22.0, help="ADX threshold for trend regimes")
    parser.add_argument("--sideways-adx", type=float, default=18.0, help="Max ADX for sideways regime")
    parser.add_argument("--sideways-bbw-pctile", type=float, default=0.45, help="Max 100d BB width percentile for sideways")
    parser.add_argument("--sideways-atr-pctile", type=float, default=0.55, help="Max 100d ATR percentile for sideways")
    parser.add_argument("--chop-threshold", type=float, default=55.0, help="Choppiness index reference threshold")
    parser.add_argument("--sl-atr-multiple", type=float, default=2.0, help="Stop distance in hourly ATR multiples")
    parser.add_argument("--rr", type=float, default=1.0, help="Reward:risk multiple. 1.0 means 1:1 RR")
    parser.add_argument("--entry-mode", choices=("open", "close", "both"), default="both")
    parser.add_argument(
        "--tie-policy",
        choices=("stop", "target", "skip"),
        default="stop",
        help="When TP and SL both hit inside one hourly candle, default is conservative stop",
    )
    parser.add_argument(
        "--max-hold-hours",
        type=int,
        default=0,
        help="0 means hold until TP/SL or data end; otherwise force close after this many hours",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = bybit_symbol(args.symbol)
    period = resolve_period(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hourly_fetch_start = period.start - pd.Timedelta(hours=args.hourly_warmup_hours)
    daily_fetch_start = period.start - pd.Timedelta(days=args.daily_warmup_days)

    print(f"Fetching {symbol} 1h candles from {hourly_fetch_start} to {period.end} ...")
    hourly_raw = fetch_bybit_klines(
        symbol,
        "1h",
        hourly_fetch_start,
        period.end,
        category=args.category,
        base_url=args.base_url,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
    )
    print(f"Fetching {symbol} 1d candles from {daily_fetch_start} to {period.end} ...")
    daily_raw = fetch_bybit_klines(
        symbol,
        "1d",
        daily_fetch_start,
        period.end,
        category=args.category,
        base_url=args.base_url,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
    )

    hourly = build_hourly_features(hourly_raw, args.atr_length)
    daily_regimes = classify_daily_regimes(
        daily_raw,
        adx_length=args.daily_adx_length,
        atr_length=args.daily_atr_length,
        trend_adx=args.trend_adx,
        sideways_adx=args.sideways_adx,
        sideways_bbw_pctile=args.sideways_bbw_pctile,
        sideways_atr_pctile=args.sideways_atr_pctile,
        chop_threshold=args.chop_threshold,
    )

    entry_modes = ENTRY_MODES if args.entry_mode == "both" else (args.entry_mode,)
    trade_frames: list[pd.DataFrame] = []
    for entry_mode in entry_modes:
        for direction in DIRECTIONS:
            simulated = simulate_trades(
                hourly,
                entry_mode=entry_mode,
                direction=direction,
                atr_length=args.atr_length,
                sl_atr_multiple=args.sl_atr_multiple,
                rr=args.rr,
                tie_policy=args.tie_policy,
                max_hold_hours=args.max_hold_hours,
            )
            simulated = trim_frame(simulated, period.start, period.end, time_column="entry_time")
            trade_frames.append(simulated)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    trades = assign_regimes(trades, daily_regimes)
    if args.tie_policy == "skip" and "outcome" in trades.columns:
        trades = trades[trades["outcome"] != "ambiguous"].reset_index(drop=True)
    histogram = summarize_histogram(trades)

    stamp = f"{symbol.lower()}_{period.start:%Y%m%d}_{period.end:%Y%m%d}"
    daily_path = args.output_dir / f"{stamp}_daily_regimes.csv"
    trades_path = args.output_dir / f"{stamp}_hourly_trades.csv"
    histogram_path = args.output_dir / f"{stamp}_hourly_histogram.csv"
    params_path = args.output_dir / f"{stamp}_run_summary.txt"

    daily_to_save = trim_frame(daily_regimes, period.start, period.end)
    daily_to_save.to_csv(daily_path, index=False)
    trades.to_csv(trades_path, index=False)
    histogram.to_csv(histogram_path, index=False)

    summary_lines = [
        f"symbol={symbol}",
        f"category={args.category}",
        f"period_start={period.start}",
        f"period_end={period.end}",
        f"hourly_candles={len(trim_frame(hourly_raw, period.start, period.end))}",
        f"daily_candles={len(trim_frame(daily_raw, period.start, period.end))}",
        f"trades={len(trades)}",
        f"sl_atr_multiple={args.sl_atr_multiple}",
        f"rr={args.rr}",
        f"tie_policy={args.tie_policy}",
        f"max_hold_hours={args.max_hold_hours}",
        f"daily_regimes_csv={daily_path}",
        f"hourly_trades_csv={trades_path}",
        f"hourly_histogram_csv={histogram_path}",
    ]
    params_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if "regime" in trades.columns:
        regime_counts = trades["regime"].value_counts().reindex(REGIMES, fill_value=0)
    else:
        regime_counts = pd.Series(0, index=REGIMES, dtype=int)
    print("\nTrade rows by daily regime:")
    print(regime_counts.to_string())
    print("\nOutput files:")
    print(f"  {daily_path}")
    print(f"  {trades_path}")
    print(f"  {histogram_path}")
    print(f"  {params_path}")

    best = (
        histogram[histogram["trades"] > 0]
        .sort_values(["avg_r", "trades"], ascending=[False, False])
        .head(12)
        [["regime", "entry_mode", "direction", "entry_hour_utc", "trades", "win_rate", "avg_r", "total_r"]]
    )
    if not best.empty:
        print("\nTop histogram cells by avg_r:")
        print(best.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
