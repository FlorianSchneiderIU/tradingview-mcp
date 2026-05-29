from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


BYBIT_URL = "https://api.bybit.com"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

LOWPASS_NUMERIC_WEIGHTS = {
    "pivot_window": 0.45,
    "pivot_confirm_window": 0.35,
    "zigzag_atr_mult": 0.80,
    "max_time_ratio": 0.70,
    "max_p5_break_atr": 0.65,
    "stop_atr_buffer": 0.80,
    "min_rr": 0.90,
    "min_score": 1.15,
    "target_projection_bars": 0.65,
    "max_hold_bars": 0.55,
}
LOWPASS_CATEGORICAL_WEIGHTS = {
    "pattern_tf": 1.25,
    "pivot_method": 0.85,
    "pivot_source": 0.45,
    "trend_filter": 0.65,
    "regime_filter": 0.70,
}
LOWPASS_MEDIAN_METRICS = [
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
RESAMPLE_RULE = {
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}
TIMEFRAME_ALIASES = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "60": "1h",
    "240": "4h",
    "D": "1d",
}
BYBIT_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1d": "D",
}


def normalize_timeframe(timeframe: str) -> str:
    tf = timeframe.strip()
    if tf in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[tf]
    tf = tf.lower()
    if tf in INTERVAL_MS:
        return tf
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


def bybit_symbol(raw: str) -> str:
    symbol = raw.strip().upper()
    if ":" in symbol:
        symbol = symbol.split(":", 1)[1]
    return symbol.replace("/", "").replace("-", "").replace(".P", "")


def parse_utc_datetime(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    if len(text) == 10:
        text = f"{text}T00:00:00+00:00"
    out = datetime.fromisoformat(text)
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def timeframe_seconds(timeframe: str) -> float:
    return INTERVAL_MS[normalize_timeframe(timeframe)] / 1000.0


def timestamp_seconds(value: Any) -> float:
    return pd.Timestamp(value).tz_convert("UTC").timestamp()


def round_to_mintick(value: float, mintick: float) -> float:
    if mintick <= 0:
        return float(value)
    return round(round(float(value) / mintick) * mintick, 10)


def fetch_bybit_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    *,
    category: str = "linear",
    base_url: str = BYBIT_URL,
) -> pd.DataFrame:
    symbol = bybit_symbol(symbol)
    interval = normalize_timeframe(interval)
    interval_ms = INTERVAL_MS[interval]
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
                    raise RuntimeError(f"Bybit returned retCode={ret_code} retMsg={payload.get('retMsg')}")
                batch = payload.get("result", {}).get("list", [])
                break
            except Exception as exc:  # noqa: BLE001 - keep fetch retry simple for research scripts.
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        if batch is None:
            raise last_error if last_error is not None else RuntimeError("Failed to fetch Bybit klines.")
        if not batch:
            break
        oldest = min(int(row[0]) for row in batch)
        for row in batch:
            rows[int(row[0])] = row
        if len(batch) < 1000 or oldest <= start_ms:
            break
        cursor_end_ms = oldest - 1
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"No Bybit klines returned for {symbol} {interval}.")

    frame = pd.DataFrame(
        [rows[key] for key in sorted(rows)],
        columns=["open_time_ms", "open", "high", "low", "close", "volume", "turnover"],
    )
    frame["open_time"] = pd.to_datetime(frame["open_time_ms"].astype("int64"), unit="ms", utc=True)
    frame["close_time"] = frame["open_time"] + pd.Timedelta(milliseconds=interval_ms - 1)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    now = pd.Timestamp.now(tz="UTC")
    frame = frame[frame["open_time"] + pd.Timedelta(milliseconds=interval_ms) <= now].copy()
    return frame[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def fetch_bybit_mintick(symbol: str, *, category: str = "linear", base_url: str = BYBIT_URL) -> float:
    response = requests.get(
        f"{base_url.rstrip('/')}/v5/market/instruments-info",
        params={"category": category, "symbol": bybit_symbol(symbol)},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    ret_code = payload.get("retCode", 0)
    if ret_code not in (0, "0"):
        raise RuntimeError(f"Bybit returned retCode={ret_code} retMsg={payload.get('retMsg')}")
    rows = payload.get("result", {}).get("list", [])
    if not rows:
        raise RuntimeError(f"No Bybit {category} instrument found for {symbol}")
    return float(rows[0].get("priceFilter", {}).get("tickSize", "0.01"))


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    if timeframe not in RESAMPLE_RULE:
        raise ValueError(f"Cannot resample to {timeframe}")
    ordered = ensure_ohlcv_frame(df)
    resampled = (
        ordered.set_index("open_time")
        .resample(RESAMPLE_RULE[timeframe], label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    resampled["close_time"] = resampled["open_time"] + pd.Timedelta(milliseconds=INTERVAL_MS[timeframe] - 1)
    return resampled[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def ensure_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out = df.copy()
    if "open_time" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or "index": "open_time"})
        else:
            raise ValueError("DataFrame needs open_time or a DatetimeIndex.")
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
    if "close_time" not in out.columns:
        inferred = infer_timeframe_from_frame(out)
        out["close_time"] = out["open_time"] + pd.Timedelta(milliseconds=INTERVAL_MS[inferred] - 1)
    else:
        out["close_time"] = pd.to_datetime(out["close_time"], utc=True, errors="coerce")
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    return out.dropna(subset=["open_time", "close_time", *required]).sort_values("open_time").reset_index(drop=True)


def infer_timeframe_from_frame(df: pd.DataFrame) -> str:
    times = pd.to_datetime(df["open_time"], utc=True, errors="coerce").dropna().sort_values()
    if len(times) < 2:
        return "5m"
    seconds = times.diff().dropna().dt.total_seconds()
    seconds = seconds[seconds > 0]
    if seconds.empty:
        return "5m"
    median = float(seconds.median())
    return min(INTERVAL_MS, key=lambda tf: abs(INTERVAL_MS[tf] / 1000.0 - median))


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def add_indicators(df: pd.DataFrame, atr_length: int, ema_length: int, rsi_length: int) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = rma(tr, atr_length)
    out["ema"] = out["close"].ewm(span=ema_length, adjust=False).mean()
    out["ema_slope_atr"] = (out["ema"] - out["ema"].shift(20)) / out["atr"].replace(0.0, np.nan)
    atr_baseline = out["atr"].rolling(200, min_periods=50).median()
    out["atr_ratio"] = out["atr"] / atr_baseline.replace(0.0, np.nan)
    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rs = rma(gain, rsi_length) / rma(loss, rsi_length).replace(0.0, np.nan)
    out["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    out["volume_sma"] = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"].replace(0.0, np.nan)
    return out


def high_before_low(open_value: float, high_value: float, low_value: float) -> bool:
    return abs(open_value - high_value) < abs(open_value - low_value)


@dataclass(frozen=True)
class Pivot:
    idx: int
    confirm_idx: int
    time: pd.Timestamp
    confirm_time: pd.Timestamp
    kind: str
    price: float
    atr: float


@dataclass(frozen=True)
class WolfeConfig:
    exec_tf: str = "5m"
    pattern_tf: str = "1h"
    pivot_method: str = "zigzag"
    pivot_source: str = "wick"
    pivot_window: int = 5
    pivot_confirm_window: int = 0
    zigzag_atr_mult: float = 1.5
    atr_length: int = 14
    ema_length: int = 200
    rsi_length: int = 14
    allow_longs: bool = True
    allow_shorts: bool = True
    max_time_ratio: float = 2.8
    min_pattern_bars: int = 12
    max_pattern_bars: int = 220
    min_p5_break_atr: float = 0.05
    max_p5_break_atr: float = 2.5
    min_p4_retrace: float = 0.25
    max_p4_retrace: float = 0.95
    max_epa_slope_atr: float = 0.65
    min_score: float = 62.0
    stop_atr_buffer: float = 0.45
    min_stop_atr: float = 0.35
    max_stop_atr: float = 4.0
    target_projection_bars: int = 18
    min_rr: float = 1.35
    max_rr: float = 5.0
    max_entry_wait_bars: int = 36
    require_reclaim: bool = True
    require_reclaim_vs_p5: bool = True
    min_volume_ratio: float = 0.0
    trend_filter: str = "none"
    regime_filter: str = "none"
    long_max_rsi: float = 58.0
    short_min_rsi: float = 42.0
    max_hold_bars: int = 288
    one_trade_at_a_time: bool = True
    mintick: float = 0.1
    fee_bps_side: float = 5.5
    slippage_bps_side: float = 1.0
    risk_fraction: float = 0.01
    min_entry_risk_pct: float = 0.05
    max_entry_risk_pct: float = 3.5

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "WolfeConfig":
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        clean = {key: values[key] for key in values if key in fields}
        return cls(**clean)


@dataclass(frozen=True)
class WolfeSignal:
    symbol: str
    direction: str
    event_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_index: int
    entry_price: float
    stop_price: float
    target_price: float
    target_rr_planned: float
    score: float
    p5_break_atr: float
    symmetry_ratio: float
    epa_slope_atr: float
    volume_ratio: float
    rsi: float
    trend_context: str
    pattern_tf: str
    exec_tf: str
    pivot_method: str
    pivots: tuple[Pivot, Pivot, Pivot, Pivot, Pivot]

    @property
    def event_key(self) -> str:
        pivot_blob = "-".join(f"{p.kind}{p.idx}" for p in self.pivots)
        return f"{self.symbol}|{self.pattern_tf}|{self.direction}|{self.entry_time.isoformat()}|{pivot_blob}"


@dataclass(frozen=True)
class WolfeTrade:
    symbol: str
    direction: str
    event_key: str
    event_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    target_rr_planned: float
    r_multiple_gross: float
    r_multiple_net: float
    return_pct: float
    hold_bars: int
    exit_reason: str
    score: float
    p5_break_atr: float
    symmetry_ratio: float
    epa_slope_atr: float
    volume_ratio: float
    rsi: float
    pattern_tf: str
    exec_tf: str
    pivot_method: str


def line_params_idx(p1: Pivot, p2: Pivot) -> tuple[float, float]:
    dx = p2.idx - p1.idx
    if dx == 0:
        return 0.0, p1.price
    m = (p2.price - p1.price) / dx
    b = p1.price - m * p1.idx
    return float(m), float(b)


def line_params_time(p1: Pivot, p2: Pivot) -> tuple[float, float]:
    x1 = timestamp_seconds(p1.time)
    x2 = timestamp_seconds(p2.time)
    dx = x2 - x1
    if dx == 0:
        return 0.0, p1.price
    m = (p2.price - p1.price) / dx
    b = p1.price - m * x1
    return float(m), float(b)


def line_value(m: float, b: float, x: float) -> float:
    return float(m * x + b)


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _score_closeness(value: float, ideal: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(value - ideal) / tolerance)


def dedupe_alternating(pivots: list[Pivot]) -> list[Pivot]:
    out: list[Pivot] = []
    for pivot in sorted(pivots, key=lambda p: (p.idx, p.confirm_idx)):
        if pivot.confirm_idx < pivot.idx:
            continue
        if not out:
            out.append(pivot)
            continue
        last = out[-1]
        if pivot.kind != last.kind:
            out.append(pivot)
            continue
        replace = (pivot.kind == "H" and pivot.price > last.price) or (pivot.kind == "L" and pivot.price < last.price)
        if replace:
            out[-1] = pivot
    return out


def pivot_price_arrays(pattern: pd.DataFrame, cfg: WolfeConfig) -> tuple[np.ndarray, np.ndarray]:
    source = str(cfg.pivot_source).strip().lower()
    if source in {"wick", "wicks", "high_low", "hl"}:
        return pattern["high"].to_numpy(dtype=float), pattern["low"].to_numpy(dtype=float)
    if source in {"close", "closes"}:
        close = pattern["close"].to_numpy(dtype=float)
        return close, close
    if source in {"body", "bodies", "open_close", "oc"}:
        open_values = pattern["open"].to_numpy(dtype=float)
        close = pattern["close"].to_numpy(dtype=float)
        return np.maximum(open_values, close), np.minimum(open_values, close)
    raise ValueError(f"Unknown pivot_source: {cfg.pivot_source!r}")


def fractal_pivots(pattern: pd.DataFrame, cfg: WolfeConfig) -> list[Pivot]:
    w = int(cfg.pivot_window)
    right = int(cfg.pivot_confirm_window) if int(cfg.pivot_confirm_window) > 0 else w
    if w <= 0 or right <= 0 or len(pattern) <= w + right:
        return []
    highs, lows = pivot_price_arrays(pattern, cfg)
    atrs = pattern["atr"].to_numpy(dtype=float)
    close_times = pd.to_datetime(pattern["close_time"], utc=True).to_numpy()
    window_size = w + right + 1
    high_windows = np.lib.stride_tricks.sliding_window_view(highs, window_size)
    low_windows = np.lib.stride_tricks.sliding_window_view(lows, window_size)
    center_positions = np.arange(w, len(pattern) - right)
    high_centers = highs[center_positions]
    low_centers = lows[center_positions]
    is_high = (high_centers == np.max(high_windows, axis=1)) & (np.argmax(high_windows, axis=1) == w)
    is_low = (low_centers == np.min(low_windows, axis=1)) & (np.argmin(low_windows, axis=1) == w)
    keep = np.flatnonzero(is_high ^ is_low)
    pivots: list[Pivot] = []
    for pos in keep:
        idx = int(center_positions[pos])
        confirm_idx = idx + right
        if bool(is_high[pos]):
            pivots.append(
                Pivot(
                    idx=idx,
                    confirm_idx=confirm_idx,
                    time=pd.Timestamp(close_times[idx]),
                    confirm_time=pd.Timestamp(close_times[confirm_idx]),
                    kind="H",
                    price=float(highs[idx]),
                    atr=float(atrs[idx]),
                )
            )
        else:
            pivots.append(
                Pivot(
                    idx=idx,
                    confirm_idx=confirm_idx,
                    time=pd.Timestamp(close_times[idx]),
                    confirm_time=pd.Timestamp(close_times[confirm_idx]),
                    kind="L",
                    price=float(lows[idx]),
                    atr=float(atrs[idx]),
                )
            )
    return dedupe_alternating(pivots)


def zigzag_pivots(pattern: pd.DataFrame, cfg: WolfeConfig) -> list[Pivot]:
    if len(pattern) < max(10, cfg.atr_length + 2):
        return []
    highs, lows = pivot_price_arrays(pattern, cfg)
    atrs = pattern["atr"].to_numpy(dtype=float)
    pivots: list[Pivot] = []
    direction = 0
    candidate_high_idx = 0
    candidate_low_idx = 0

    for idx in range(1, len(pattern)):
        atr_value = float(atrs[idx])
        if not _finite(atr_value) or atr_value <= 0.0:
            continue
        threshold = cfg.zigzag_atr_mult * atr_value
        if direction >= 0 and highs[idx] >= highs[candidate_high_idx]:
            candidate_high_idx = idx
        if direction <= 0 and lows[idx] <= lows[candidate_low_idx]:
            candidate_low_idx = idx

        if direction == 0:
            up_move = highs[idx] - lows[candidate_low_idx]
            down_move = highs[candidate_high_idx] - lows[idx]
            if up_move >= threshold:
                pivots.append(_pivot_from_row(pattern, candidate_low_idx, idx, "L", lows[candidate_low_idx]))
                direction = 1
                candidate_high_idx = idx
            elif down_move >= threshold:
                pivots.append(_pivot_from_row(pattern, candidate_high_idx, idx, "H", highs[candidate_high_idx]))
                direction = -1
                candidate_low_idx = idx
            continue

        if direction == 1:
            if highs[idx] >= highs[candidate_high_idx]:
                candidate_high_idx = idx
            if highs[candidate_high_idx] - lows[idx] >= threshold:
                pivots.append(_pivot_from_row(pattern, candidate_high_idx, idx, "H", highs[candidate_high_idx]))
                direction = -1
                candidate_low_idx = idx
        else:
            if lows[idx] <= lows[candidate_low_idx]:
                candidate_low_idx = idx
            if highs[idx] - lows[candidate_low_idx] >= threshold:
                pivots.append(_pivot_from_row(pattern, candidate_low_idx, idx, "L", lows[candidate_low_idx]))
                direction = 1
                candidate_high_idx = idx

    return dedupe_alternating(pivots)


def _pivot_from_row(pattern: pd.DataFrame, idx: int, confirm_idx: int, kind: str, price: float) -> Pivot:
    return Pivot(
        idx=int(idx),
        confirm_idx=int(confirm_idx),
        time=pd.Timestamp(pattern["close_time"].iloc[idx]),
        confirm_time=pd.Timestamp(pattern["close_time"].iloc[confirm_idx]),
        kind=kind,
        price=float(price),
        atr=float(pattern["atr"].iloc[idx]),
    )


def find_pivots(pattern: pd.DataFrame, cfg: WolfeConfig) -> list[Pivot]:
    method = cfg.pivot_method.strip().lower()
    if method == "fractal":
        return fractal_pivots(pattern, cfg)
    if method == "zigzag":
        return zigzag_pivots(pattern, cfg)
    raise ValueError(f"Unknown pivot_method: {cfg.pivot_method!r}")


def score_pattern(
    pivots: tuple[Pivot, Pivot, Pivot, Pivot, Pivot],
    *,
    direction: str,
    p5_break_atr: float,
    symmetry_ratio: float,
    p4_retrace: float,
    epa_slope_atr: float,
    target_rr: float,
    cfg: WolfeConfig,
) -> float:
    symmetry = _score_closeness(symmetry_ratio, 1.0, max(cfg.max_time_ratio - 1.0, 0.1))
    false_break_mid = (cfg.min_p5_break_atr + cfg.max_p5_break_atr) / 2.0
    false_break_tol = max((cfg.max_p5_break_atr - cfg.min_p5_break_atr) / 2.0, 0.1)
    false_break = _score_closeness(p5_break_atr, false_break_mid, false_break_tol)
    retrace_mid = (cfg.min_p4_retrace + cfg.max_p4_retrace) / 2.0
    retrace_tol = max((cfg.max_p4_retrace - cfg.min_p4_retrace) / 2.0, 0.1)
    retrace = _score_closeness(p4_retrace, retrace_mid, retrace_tol)
    slope = max(0.0, 1.0 - epa_slope_atr / max(cfg.max_epa_slope_atr, 1e-9))
    rr_mid = min(max((cfg.min_rr + min(cfg.max_rr, 4.0)) / 2.0, cfg.min_rr), cfg.max_rr)
    rr_tol = max(rr_mid - cfg.min_rr, cfg.max_rr - rr_mid, 0.5)
    rr_score = _score_closeness(target_rr, rr_mid, rr_tol)

    p1, p2, p3, p4, p5 = pivots
    if direction == "long":
        depth_13 = abs(p1.price - p3.price) / max(p5.atr, 1e-9)
        depth_35 = abs(p3.price - p5.price) / max(p5.atr, 1e-9)
    else:
        depth_13 = abs(p3.price - p1.price) / max(p5.atr, 1e-9)
        depth_35 = abs(p5.price - p3.price) / max(p5.atr, 1e-9)
    depth_balance = _score_closeness(depth_35 / max(depth_13, 1e-9), 1.0, 1.25)

    raw = (
        20.0 * symmetry
        + 22.0 * false_break
        + 16.0 * retrace
        + 14.0 * slope
        + 18.0 * rr_score
        + 10.0 * depth_balance
    )
    return round(float(max(0.0, min(100.0, raw))), 3)


def _pattern_context_ok(pattern_row: pd.Series, direction: str, cfg: WolfeConfig) -> tuple[bool, str]:
    trend_filter = cfg.trend_filter.strip().lower()
    regime_filter = cfg.regime_filter.strip().lower()
    rsi = float(pattern_row.get("rsi", math.nan))
    close = float(pattern_row.get("close", math.nan))
    ema = float(pattern_row.get("ema", math.nan))
    atr = float(pattern_row.get("atr", math.nan))
    ema_slope_atr = float(pattern_row.get("ema_slope_atr", math.nan))
    atr_ratio = float(pattern_row.get("atr_ratio", math.nan))

    contexts: list[str] = []
    if trend_filter in {"none", ""}:
        contexts.append("trend:none")
    elif trend_filter == "rsi":
        if direction == "long" and _finite(rsi) and rsi > cfg.long_max_rsi:
            return False, f"rsi>{cfg.long_max_rsi:g}"
        if direction == "short" and _finite(rsi) and rsi < cfg.short_min_rsi:
            return False, f"rsi<{cfg.short_min_rsi:g}"
        contexts.append("trend:rsi")
    elif trend_filter == "counter_ema":
        if not (_finite(close) and _finite(ema) and _finite(atr) and atr > 0):
            return False, "trend_unavailable"
        dist_atr = (close - ema) / atr
        if direction == "long" and dist_atr < -6.0:
            return False, "deep_below_ema"
        if direction == "short" and dist_atr > 6.0:
            return False, "deep_above_ema"
        contexts.append("trend:counter_ema")
    else:
        raise ValueError(f"Unknown trend_filter: {cfg.trend_filter!r}")

    if regime_filter in {"none", ""}:
        contexts.append("regime:none")
    elif regime_filter == "high_vol":
        if not (_finite(atr_ratio) and atr_ratio >= 1.0):
            return False, "regime_not_high_vol"
        contexts.append("regime:high_vol")
    elif regime_filter == "low_vol":
        if not (_finite(atr_ratio) and atr_ratio <= 1.0):
            return False, "regime_not_low_vol"
        contexts.append("regime:low_vol")
    elif regime_filter == "trend_aligned":
        if not (_finite(close) and _finite(ema) and _finite(ema_slope_atr)):
            return False, "regime_unavailable"
        if direction == "long" and not (close >= ema and ema_slope_atr >= 0.0):
            return False, "regime_not_bull_trend"
        if direction == "short" and not (close <= ema and ema_slope_atr <= 0.0):
            return False, "regime_not_bear_trend"
        contexts.append("regime:trend_aligned")
    elif regime_filter == "mean_reversion":
        if not (_finite(close) and _finite(ema) and _finite(ema_slope_atr)):
            return False, "regime_unavailable"
        if direction == "long" and not (close <= ema and ema_slope_atr <= 0.0):
            return False, "regime_not_down_reversion"
        if direction == "short" and not (close >= ema and ema_slope_atr >= 0.0):
            return False, "regime_not_up_reversion"
        contexts.append("regime:mean_reversion")
    else:
        raise ValueError(f"Unknown regime_filter: {cfg.regime_filter!r}")
    return True, "|".join(contexts)


def validate_pivot_five(
    pivots: tuple[Pivot, Pivot, Pivot, Pivot, Pivot],
    pattern: pd.DataFrame,
    cfg: WolfeConfig,
) -> dict[str, Any] | None:
    p1, p2, p3, p4, p5 = pivots
    kinds = "".join(p.kind for p in pivots)
    direction = "long" if kinds == "LHLHL" else "short" if kinds == "HLHLH" else ""
    if not direction:
        return None
    if direction == "long" and not cfg.allow_longs:
        return None
    if direction == "short" and not cfg.allow_shorts:
        return None

    span = p5.idx - p1.idx
    if span < cfg.min_pattern_bars or span > cfg.max_pattern_bars:
        return None
    a = p3.idx - p1.idx
    b = p5.idx - p3.idx
    if a <= 0 or b <= 0:
        return None
    symmetry_ratio = max(a, b) / min(a, b)
    if symmetry_ratio > cfg.max_time_ratio:
        return None

    m13_i, b13_i = line_params_idx(p1, p3)
    m24_i, b24_i = line_params_idx(p2, p4)
    m14_t, b14_t = line_params_time(p1, p4)
    l13_at_5 = line_value(m13_i, b13_i, p5.idx)
    l24_at_5 = line_value(m24_i, b24_i, p5.idx)
    atr = max(float(pattern["atr"].iloc[p5.idx]), float(p5.atr), 1e-9)

    if direction == "long":
        if not (p3.price < p1.price and p5.price < p3.price and p4.price < p2.price and p4.price > p3.price):
            return None
        p4_retrace = (p4.price - p3.price) / max(p2.price - p3.price, 1e-9)
        breakout = l13_at_5 - p5.price
        if l24_at_5 <= l13_at_5:
            return None
    else:
        if not (p3.price > p1.price and p5.price > p3.price and p4.price > p2.price and p4.price < p3.price):
            return None
        p4_retrace = (p3.price - p4.price) / max(p3.price - p2.price, 1e-9)
        breakout = p5.price - l13_at_5
        if l13_at_5 <= l24_at_5:
            return None

    p5_break_atr = breakout / atr
    if p5_break_atr < cfg.min_p5_break_atr or p5_break_atr > cfg.max_p5_break_atr:
        return None
    if p4_retrace < cfg.min_p4_retrace or p4_retrace > cfg.max_p4_retrace:
        return None
    epa_slope_atr = abs((m14_t * timeframe_seconds(cfg.pattern_tf)) / atr)
    if epa_slope_atr > cfg.max_epa_slope_atr:
        return None
    row = pattern.iloc[min(max(p5.confirm_idx, 0), len(pattern) - 1)]
    context_ok, trend_context = _pattern_context_ok(row, direction, cfg)
    if not context_ok:
        return None
    return {
        "direction": direction,
        "m13_i": m13_i,
        "b13_i": b13_i,
        "m14_t": m14_t,
        "b14_t": b14_t,
        "p5_break_atr": float(p5_break_atr),
        "symmetry_ratio": float(symmetry_ratio),
        "p4_retrace": float(p4_retrace),
        "epa_slope_atr": float(epa_slope_atr),
        "trend_context": trend_context,
    }


def _exec_time_index(exec_df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(exec_df["close_time"], utc=True, errors="coerce"))


def _first_exec_after(
    exec_df: pd.DataFrame,
    when: pd.Timestamp,
    exec_times: pd.DatetimeIndex | None = None,
) -> int:
    times = exec_times if exec_times is not None else _exec_time_index(exec_df)
    return int(times.searchsorted(pd.Timestamp(when).tz_convert("UTC"), side="right"))


def _target_from_epa(
    *,
    direction: str,
    entry_price: float,
    entry_time: pd.Timestamp,
    stop_price: float,
    valid: dict[str, Any],
    cfg: WolfeConfig,
) -> tuple[float, float] | None:
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return None
    target_time = pd.Timestamp(entry_time).tz_convert("UTC") + pd.Timedelta(
        seconds=timeframe_seconds(cfg.pattern_tf) * max(cfg.target_projection_bars, 1)
    )
    target_raw = line_value(float(valid["m14_t"]), float(valid["b14_t"]), timestamp_seconds(target_time))
    if direction == "long":
        if target_raw <= entry_price:
            return None
        rr = (target_raw - entry_price) / risk
    else:
        if target_raw >= entry_price:
            return None
        rr = (entry_price - target_raw) / risk
    if rr < cfg.min_rr or rr > cfg.max_rr:
        return None
    return round_to_mintick(target_raw, cfg.mintick), float(rr)


def _find_entry(
    *,
    symbol: str,
    pivots: tuple[Pivot, Pivot, Pivot, Pivot, Pivot],
    valid: dict[str, Any],
    pattern: pd.DataFrame,
    exec_df: pd.DataFrame,
    exec_times: pd.DatetimeIndex | None,
    cfg: WolfeConfig,
) -> WolfeSignal | None:
    p5 = pivots[-1]
    if p5.confirm_idx >= len(pattern):
        return None
    start = _first_exec_after(exec_df, pd.Timestamp(pattern["close_time"].iloc[p5.confirm_idx]), exec_times)
    end = min(len(exec_df) - 1, start + max(int(cfg.max_entry_wait_bars), 1))
    if start >= len(exec_df):
        return None

    direction = str(valid["direction"])
    line13_m, line13_b = line_params_time(pivots[0], pivots[2])
    for entry_idx in range(start, end + 1):
        row = exec_df.iloc[entry_idx]
        entry_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        close = float(row["close"])
        x_seconds = timestamp_seconds(entry_time)
        line13 = line_value(line13_m, line13_b, x_seconds)
        if cfg.require_reclaim:
            if direction == "long" and close <= line13:
                continue
            if direction == "short" and close >= line13:
                continue
        if cfg.require_reclaim_vs_p5:
            if direction == "long" and close <= p5.price:
                continue
            if direction == "short" and close >= p5.price:
                continue
        volume_ratio = float(row.get("volume_ratio", math.nan))
        if cfg.min_volume_ratio > 0 and _finite(volume_ratio) and volume_ratio < cfg.min_volume_ratio:
            continue
        atr = max(float(row.get("atr", math.nan)), 1e-9)
        stop_buffer = max(cfg.stop_atr_buffer * atr, cfg.min_stop_atr * atr)
        if direction == "long":
            stop = round_to_mintick(p5.price - stop_buffer, cfg.mintick)
        else:
            stop = round_to_mintick(p5.price + stop_buffer, cfg.mintick)
        entry = round_to_mintick(close, cfg.mintick)
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        risk_atr = risk / atr
        risk_pct = risk / entry * 100.0 if entry > 0 else math.inf
        if risk_atr > cfg.max_stop_atr:
            continue
        if risk_pct < cfg.min_entry_risk_pct or risk_pct > cfg.max_entry_risk_pct:
            continue
        target = _target_from_epa(
            direction=direction,
            entry_price=entry,
            entry_time=entry_time,
            stop_price=stop,
            valid=valid,
            cfg=cfg,
        )
        if target is None:
            continue
        target_price, target_rr = target
        score = score_pattern(
            pivots,
            direction=direction,
            p5_break_atr=float(valid["p5_break_atr"]),
            symmetry_ratio=float(valid["symmetry_ratio"]),
            p4_retrace=float(valid["p4_retrace"]),
            epa_slope_atr=float(valid["epa_slope_atr"]),
            target_rr=target_rr,
            cfg=cfg,
        )
        if score < cfg.min_score:
            continue
        return WolfeSignal(
            symbol=symbol,
            direction=direction,
            event_time=pd.Timestamp(pattern["close_time"].iloc[p5.confirm_idx]).tz_convert("UTC"),
            entry_time=entry_time,
            entry_index=int(entry_idx),
            entry_price=float(entry),
            stop_price=float(stop),
            target_price=float(target_price),
            target_rr_planned=float(target_rr),
            score=float(score),
            p5_break_atr=float(valid["p5_break_atr"]),
            symmetry_ratio=float(valid["symmetry_ratio"]),
            epa_slope_atr=float(valid["epa_slope_atr"]),
            volume_ratio=volume_ratio if _finite(volume_ratio) else math.nan,
            rsi=float(row.get("rsi", math.nan)),
            trend_context=str(valid["trend_context"]),
            pattern_tf=cfg.pattern_tf,
            exec_tf=cfg.exec_tf,
            pivot_method=cfg.pivot_method,
            pivots=pivots,
        )
    return None


def find_wolfe_signals(
    exec_df: pd.DataFrame,
    cfg: WolfeConfig,
    *,
    symbol: str = "BTCUSDT",
) -> list[WolfeSignal]:
    cfg = WolfeConfig.from_mapping({**asdict(cfg), "exec_tf": normalize_timeframe(cfg.exec_tf), "pattern_tf": normalize_timeframe(cfg.pattern_tf)})
    exec_frame = add_indicators(ensure_ohlcv_frame(exec_df), cfg.atr_length, cfg.ema_length, cfg.rsi_length)
    pattern = resample_ohlc(exec_frame, cfg.pattern_tf)
    pattern = add_indicators(pattern, cfg.atr_length, cfg.ema_length, cfg.rsi_length)
    pivots = find_pivots(pattern, cfg)
    exec_times = _exec_time_index(exec_frame)
    signals: list[WolfeSignal] = []
    seen_entries: set[tuple[int, str]] = set()

    for idx in range(len(pivots) - 4):
        five = tuple(pivots[idx : idx + 5])
        valid = validate_pivot_five(five, pattern, cfg)
        if valid is None:
            continue
        signal = _find_entry(
            symbol=symbol,
            pivots=five,
            valid=valid,
            pattern=pattern,
            exec_df=exec_frame,
            exec_times=exec_times,
            cfg=cfg,
        )
        if signal is None:
            continue
        key = (signal.entry_index, signal.direction)
        if key in seen_entries:
            continue
        seen_entries.add(key)
        signals.append(signal)
    return sorted(signals, key=lambda sig: sig.entry_index)


def signal_rows(signals: list[WolfeSignal]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sig in signals:
        row = {
            "symbol": sig.symbol,
            "event_key": sig.event_key,
            "direction": sig.direction,
            "event_time": sig.event_time,
            "entry_time": sig.entry_time,
            "entry_index": sig.entry_index,
            "entry_price": sig.entry_price,
            "stop_price": sig.stop_price,
            "target_price": sig.target_price,
            "target_rr_planned": sig.target_rr_planned,
            "score": sig.score,
            "p5_break_atr": sig.p5_break_atr,
            "symmetry_ratio": sig.symmetry_ratio,
            "epa_slope_atr": sig.epa_slope_atr,
            "volume_ratio": sig.volume_ratio,
            "rsi": sig.rsi,
            "trend_context": sig.trend_context,
            "pattern_tf": sig.pattern_tf,
            "exec_tf": sig.exec_tf,
            "pivot_method": sig.pivot_method,
        }
        for i, pivot in enumerate(sig.pivots, start=1):
            row[f"p{i}_kind"] = pivot.kind
            row[f"p{i}_time"] = pivot.time
            row[f"p{i}_confirm_time"] = pivot.confirm_time
            row[f"p{i}_price"] = pivot.price
            row[f"p{i}_idx"] = pivot.idx
            row[f"p{i}_confirm_idx"] = pivot.confirm_idx
        rows.append(row)
    return pd.DataFrame(rows)


def _cost_r(entry_price: float, risk: float, cfg: WolfeConfig) -> float:
    if risk <= 0:
        return 0.0
    return ((2.0 * cfg.fee_bps_side) + (2.0 * cfg.slippage_bps_side)) / 10_000.0 * entry_price / risk


def run_backtest(
    exec_df: pd.DataFrame,
    cfg: WolfeConfig,
    *,
    symbol: str = "BTCUSDT",
    precomputed_signals: list[WolfeSignal] | None = None,
) -> pd.DataFrame:
    frame = add_indicators(ensure_ohlcv_frame(exec_df), cfg.atr_length, cfg.ema_length, cfg.rsi_length)
    signals = precomputed_signals if precomputed_signals is not None else find_wolfe_signals(frame, cfg, symbol=symbol)
    trades: list[WolfeTrade] = []
    next_available_idx = 0

    for sig in signals:
        if cfg.one_trade_at_a_time and sig.entry_index < next_available_idx:
            continue
        if sig.entry_index >= len(frame) - 1:
            continue
        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        exit_idx = min(len(frame) - 1, sig.entry_index + max(1, int(cfg.max_hold_bars)))
        exit_price = float(frame["close"].iloc[exit_idx])
        exit_reason = "timeout"

        for idx in range(sig.entry_index + 1, exit_idx + 1):
            row = frame.iloc[idx]
            open_value = float(row["open"])
            high_value = float(row["high"])
            low_value = float(row["low"])
            if sig.direction == "long":
                target_hit = high_value >= target
                stop_hit = low_value <= stop
                if target_hit and stop_hit:
                    if high_before_low(open_value, high_value, low_value):
                        exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                    else:
                        exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                    break
                if stop_hit:
                    exit_idx, exit_price, exit_reason = idx, stop, "stop"
                    break
                if target_hit:
                    exit_idx, exit_price, exit_reason = idx, target, "target"
                    break
            else:
                target_hit = low_value <= target
                stop_hit = high_value >= stop
                if target_hit and stop_hit:
                    if high_before_low(open_value, high_value, low_value):
                        exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                    else:
                        exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                    break
                if stop_hit:
                    exit_idx, exit_price, exit_reason = idx, stop, "stop"
                    break
                if target_hit:
                    exit_idx, exit_price, exit_reason = idx, target, "target"
                    break

        gross_r = (exit_price - entry) / risk if sig.direction == "long" else (entry - exit_price) / risk
        net_r = gross_r - _cost_r(entry, risk, cfg)
        trades.append(
            WolfeTrade(
                symbol=symbol,
                direction=sig.direction,
                event_key=sig.event_key,
                event_time=sig.event_time,
                entry_time=sig.entry_time,
                exit_time=pd.Timestamp(frame["close_time"].iloc[exit_idx]).tz_convert("UTC"),
                entry_price=float(entry),
                exit_price=float(exit_price),
                stop_price=float(stop),
                target_price=float(target),
                target_rr_planned=float(sig.target_rr_planned),
                r_multiple_gross=float(gross_r),
                r_multiple_net=float(net_r),
                return_pct=float(cfg.risk_fraction * net_r),
                hold_bars=int(exit_idx - sig.entry_index),
                exit_reason=exit_reason,
                score=float(sig.score),
                p5_break_atr=float(sig.p5_break_atr),
                symmetry_ratio=float(sig.symmetry_ratio),
                epa_slope_atr=float(sig.epa_slope_atr),
                volume_ratio=float(sig.volume_ratio),
                rsi=float(sig.rsi),
                pattern_tf=sig.pattern_tf,
                exec_tf=sig.exec_tf,
                pivot_method=sig.pivot_method,
            )
        )
        next_available_idx = exit_idx + 1

    return pd.DataFrame([asdict(trade) for trade in trades])


def strategy_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
        }
    r = pd.to_numeric(trades["r_multiple_net"], errors="coerce").fillna(0.0)
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    equity_r = r.cumsum()
    dd = equity_r - equity_r.cummax()
    returns = pd.to_numeric(trades["return_pct"], errors="coerce").fillna(0.0)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    return {
        "trades": float(len(trades)),
        "net_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "max_dd_r": float(dd.min()),
        "total_return": float((1.0 + returns).prod() - 1.0),
        "sharpe": float(returns.mean() / std * math.sqrt(len(returns))) if std > 0.0 else 0.0,
    }


def split_trades(trades: pd.DataFrame, *, train_end: pd.Timestamp, validation_end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {"train": trades.copy(), "validation": trades.copy(), "oos": trades.copy()}
    times = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    return {
        "train": trades[times < train_end].copy(),
        "validation": trades[(times >= train_end) & (times < validation_end)].copy(),
        "oos": trades[times >= validation_end].copy(),
    }


def metric_row(trades: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in strategy_metrics(trades).items()}


def finite_metric(values: pd.Series, cap: float | None = None) -> np.ndarray:
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = out[np.isfinite(out)]
    fill = float(np.nanmedian(finite)) if finite.size else 0.0
    out = np.where(np.isfinite(out), out, fill)
    if cap is not None:
        out = np.clip(out, -cap, cap)
    return out


def parameter_distance2(rows: pd.DataFrame) -> np.ndarray:
    n = len(rows)
    d2 = np.zeros((n, n), dtype=float)
    for col, weight in LOWPASS_NUMERIC_WEIGHTS.items():
        if col not in rows.columns:
            continue
        values = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        values = np.where(np.isfinite(values), values, np.nanmedian(finite))
        span = float(np.nanmax(values) - np.nanmin(values))
        if span <= 0.0:
            continue
        delta = (values[:, None] - values[None, :]) / span
        d2 += weight * delta * delta

    for col, weight in LOWPASS_CATEGORICAL_WEIGHTS.items():
        if col not in rows.columns:
            continue
        values = rows[col].astype(str).to_numpy()
        d2 += weight * (values[:, None] != values[None, :]).astype(float)
    return d2


def median_lowpass_scores(
    scores: pd.DataFrame,
    *,
    radius: float,
    min_neighbors: int,
    outlier_penalty: float,
) -> pd.DataFrame:
    if scores.empty or "robust_score" not in scores.columns:
        return scores

    out = scores.copy().reset_index(drop=True)
    n = len(out)
    error_mask = out["error"].notna().to_numpy() if "error" in out.columns else np.zeros(n, dtype=bool)
    evaluated_mask = ~error_mask & pd.to_numeric(out["robust_score"], errors="coerce").notna().to_numpy()
    if not evaluated_mask.any():
        return out

    d2 = parameter_distance2(out)
    radius2 = max(float(radius), 0.0) ** 2
    min_neighbors = min(max(int(min_neighbors), 1), int(evaluated_mask.sum()))
    metric_values = {
        metric: finite_metric(out[metric], cap=10.0 if metric.endswith("profit_factor") else None)
        for metric in LOWPASS_MEDIAN_METRICS
        if metric in out.columns
    }
    pass_mask = (
        out["selection_pass"].fillna(False).astype(bool).to_numpy()
        if "selection_pass" in out.columns
        else np.ones(n, dtype=bool)
    )

    neighbor_counts = np.zeros(n, dtype=int)
    local_pass_rates = np.full(n, np.nan, dtype=float)
    lowpass_values: dict[str, np.ndarray] = {
        metric: np.full(n, np.nan, dtype=float) for metric in metric_values
    }

    evaluated_indices = np.flatnonzero(evaluated_mask)
    for row_idx in range(n):
        if not evaluated_mask[row_idx]:
            continue
        local_mask = (d2[row_idx] <= radius2) & evaluated_mask
        if int(local_mask.sum()) < min_neighbors:
            nearest = evaluated_indices[np.argsort(d2[row_idx, evaluated_indices])[:min_neighbors]]
            local_mask[nearest] = True

        neighbor_counts[row_idx] = int(local_mask.sum())
        local_pass_rates[row_idx] = float(pass_mask[local_mask].mean()) if local_mask.any() else np.nan
        for metric, values in metric_values.items():
            lowpass_values[metric][row_idx] = float(np.median(values[local_mask]))

    for metric, values in lowpass_values.items():
        out[f"lowpass_{metric}"] = values

    raw_score = pd.to_numeric(out["robust_score"], errors="coerce").to_numpy(dtype=float)
    lowpass_score = out["lowpass_robust_score"].to_numpy(dtype=float)
    out["local_neighbor_count"] = neighbor_counts
    out["local_pass_rate"] = local_pass_rates
    out["lowpass_outlier_gap"] = np.maximum(raw_score - lowpass_score, 0.0)
    out["optimization_score"] = lowpass_score
    out["stability_score"] = lowpass_score - float(outlier_penalty) * out["lowpass_outlier_gap"].to_numpy(dtype=float)
    return out


BTC_TUNE_KEYS = [
    "pivot_method",
    "pattern_tf",
    "pivot_source",
    "pivot_window",
    "pivot_confirm_window",
    "zigzag_atr_mult",
    "max_time_ratio",
    "max_p5_break_atr",
    "stop_atr_buffer",
    "min_rr",
    "min_score",
    "target_projection_bars",
    "max_hold_bars",
    "trend_filter",
    "regime_filter",
]
BTC_TUNE_VALUES = {
    "pattern_tf": ("5m", "15m", "1h", "4h"),
    "pivot_source": ("wick", "body", "close"),
    "pivot_window": (3, 5, 8, 12),
    "zigzag_atr_mult": (1.0, 1.4, 1.8, 2.2),
    "max_time_ratio": (2.2, 3.0, 3.8),
    "max_p5_break_atr": (1.4, 2.2, 3.0),
    "stop_atr_buffer": (0.30, 0.50, 0.75),
    "min_rr": (1.2, 1.5, 2.0),
    "min_score": (48.0, 58.0, 64.0, 70.0),
    "target_projection_bars": (8, 12, 18, 30),
    "max_hold_bars": (96, 144, 288, 432, 576),
    "trend_filter": ("none", "rsi"),
    "regime_filter": ("none", "high_vol", "low_vol", "trend_aligned", "mean_reversion"),
}
BTC_CONFIRM_VALUES = {
    "zigzag": (0,),
    "fractal": (0, 3, 5, 8, 12),
}


def btc_parameter_grid(max_configs: int, seed: int = 42) -> list[dict[str, Any]]:
    methods = ("zigzag", "fractal")

    def rows_in_order():
        for pivot_method in methods:
            for values in itertools.product(
                BTC_TUNE_VALUES["pattern_tf"],
                BTC_TUNE_VALUES["pivot_source"],
                BTC_TUNE_VALUES["pivot_window"],
                BTC_CONFIRM_VALUES[pivot_method],
                BTC_TUNE_VALUES["zigzag_atr_mult"],
                BTC_TUNE_VALUES["max_time_ratio"],
                BTC_TUNE_VALUES["max_p5_break_atr"],
                BTC_TUNE_VALUES["stop_atr_buffer"],
                BTC_TUNE_VALUES["min_rr"],
                BTC_TUNE_VALUES["min_score"],
                BTC_TUNE_VALUES["target_projection_bars"],
                BTC_TUNE_VALUES["max_hold_bars"],
                BTC_TUNE_VALUES["trend_filter"],
                BTC_TUNE_VALUES["regime_filter"],
            ):
                yield dict(zip(BTC_TUNE_KEYS, (pivot_method, *values)))

    if max_configs <= 0:
        return list(rows_in_order())

    total = sum(
        math.prod(
            [
                len(BTC_TUNE_VALUES["pattern_tf"]),
                len(BTC_TUNE_VALUES["pivot_source"]),
                len(BTC_TUNE_VALUES["pivot_window"]),
                len(BTC_CONFIRM_VALUES[method]),
                len(BTC_TUNE_VALUES["zigzag_atr_mult"]),
                len(BTC_TUNE_VALUES["max_time_ratio"]),
                len(BTC_TUNE_VALUES["max_p5_break_atr"]),
                len(BTC_TUNE_VALUES["stop_atr_buffer"]),
                len(BTC_TUNE_VALUES["min_rr"]),
                len(BTC_TUNE_VALUES["min_score"]),
                len(BTC_TUNE_VALUES["target_projection_bars"]),
                len(BTC_TUNE_VALUES["max_hold_bars"]),
                len(BTC_TUNE_VALUES["trend_filter"]),
                len(BTC_TUNE_VALUES["regime_filter"]),
            ]
        )
        for method in methods
    )
    target = min(int(max_configs), total)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def pick(values: tuple[Any, ...]) -> Any:
        return values[int(rng.integers(len(values)))]

    attempts = 0
    while len(rows) < target and attempts < target * 100:
        attempts += 1
        pivot_method = str(pick(methods))
        row = {
            "pivot_method": pivot_method,
            "pattern_tf": pick(BTC_TUNE_VALUES["pattern_tf"]),
            "pivot_source": pick(BTC_TUNE_VALUES["pivot_source"]),
            "pivot_window": pick(BTC_TUNE_VALUES["pivot_window"]),
            "pivot_confirm_window": pick(BTC_CONFIRM_VALUES[pivot_method]),
            "zigzag_atr_mult": pick(BTC_TUNE_VALUES["zigzag_atr_mult"]),
            "max_time_ratio": pick(BTC_TUNE_VALUES["max_time_ratio"]),
            "max_p5_break_atr": pick(BTC_TUNE_VALUES["max_p5_break_atr"]),
            "stop_atr_buffer": pick(BTC_TUNE_VALUES["stop_atr_buffer"]),
            "min_rr": pick(BTC_TUNE_VALUES["min_rr"]),
            "min_score": pick(BTC_TUNE_VALUES["min_score"]),
            "target_projection_bars": pick(BTC_TUNE_VALUES["target_projection_bars"]),
            "max_hold_bars": pick(BTC_TUNE_VALUES["max_hold_bars"]),
            "trend_filter": pick(BTC_TUNE_VALUES["trend_filter"]),
            "regime_filter": pick(BTC_TUNE_VALUES["regime_filter"]),
        }
        key = tuple(row[key] for key in BTC_TUNE_KEYS)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    if len(rows) < target:
        for row in rows_in_order():
            key = tuple(row[key] for key in BTC_TUNE_KEYS)
            if key in seen:
                continue
            rows.append(row)
            if len(rows) >= target:
                break
    return rows


def plain_value(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def coerce_tune_param(key: str, value: Any) -> Any:
    if key in {"pivot_window", "pivot_confirm_window", "target_projection_bars", "max_hold_bars"}:
        return int(float(value))
    if key in {
        "zigzag_atr_mult",
        "max_time_ratio",
        "max_p5_break_atr",
        "stop_atr_buffer",
        "min_rr",
        "min_score",
    }:
        return float(value)
    return str(value)


def btc_params_from_row(row: pd.Series) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in BTC_TUNE_KEYS:
        if key not in row.index or pd.isna(row[key]):
            continue
        params[key] = coerce_tune_param(key, plain_value(row[key]))
    if "pivot_method" in params:
        method = str(params["pivot_method"])
        if method in BTC_CONFIRM_VALUES and params.get("pivot_confirm_window") not in BTC_CONFIRM_VALUES[method]:
            params["pivot_confirm_window"] = BTC_CONFIRM_VALUES[method][0]
    return params


def config_key(values: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(values.get(key) for key in BTC_TUNE_KEYS)


def adjacent_values(values: tuple[Any, ...], current: Any, width: int) -> tuple[Any, ...]:
    if not values:
        return (current,)
    if isinstance(values[0], str):
        try:
            idx = values.index(str(current))
        except ValueError:
            return (current, *values)
    else:
        numeric = [float(value) for value in values]
        target = float(current)
        idx = int(np.argmin(np.abs(np.asarray(numeric) - target)))
    lo = max(0, idx - max(int(width), 0))
    hi = min(len(values), idx + max(int(width), 0) + 1)
    out = list(values[lo:hi])
    if current not in out:
        out.insert(0, current)
    return tuple(dict.fromkeys(out))


def local_refinement_options(seed_params: dict[str, Any], width: int) -> dict[str, tuple[Any, ...]]:
    method = str(seed_params.get("pivot_method", "fractal"))
    confirm_values = BTC_CONFIRM_VALUES.get(method, (0,))
    return {
        "pivot_method": (method,),
        "pattern_tf": adjacent_values(BTC_TUNE_VALUES["pattern_tf"], seed_params.get("pattern_tf", "15m"), width),
        "pivot_source": tuple(dict.fromkeys((seed_params.get("pivot_source", "wick"), *BTC_TUNE_VALUES["pivot_source"]))),
        "pivot_window": adjacent_values(BTC_TUNE_VALUES["pivot_window"], seed_params.get("pivot_window", 8), width),
        "pivot_confirm_window": adjacent_values(confirm_values, seed_params.get("pivot_confirm_window", 0), width),
        "zigzag_atr_mult": adjacent_values(BTC_TUNE_VALUES["zigzag_atr_mult"], seed_params.get("zigzag_atr_mult", 1.4), width),
        "max_time_ratio": adjacent_values(BTC_TUNE_VALUES["max_time_ratio"], seed_params.get("max_time_ratio", 3.0), width),
        "max_p5_break_atr": adjacent_values(BTC_TUNE_VALUES["max_p5_break_atr"], seed_params.get("max_p5_break_atr", 2.2), width),
        "stop_atr_buffer": adjacent_values(BTC_TUNE_VALUES["stop_atr_buffer"], seed_params.get("stop_atr_buffer", 0.5), width),
        "min_rr": adjacent_values(BTC_TUNE_VALUES["min_rr"], seed_params.get("min_rr", 1.5), width),
        "min_score": adjacent_values(BTC_TUNE_VALUES["min_score"], seed_params.get("min_score", 58.0), width),
        "target_projection_bars": adjacent_values(
            BTC_TUNE_VALUES["target_projection_bars"],
            seed_params.get("target_projection_bars", 18),
            width,
        ),
        "max_hold_bars": adjacent_values(BTC_TUNE_VALUES["max_hold_bars"], seed_params.get("max_hold_bars", 288), width),
        "trend_filter": tuple(dict.fromkeys((seed_params.get("trend_filter", "none"), *BTC_TUNE_VALUES["trend_filter"]))),
        "regime_filter": tuple(
            dict.fromkeys((seed_params.get("regime_filter", "none"), *BTC_TUNE_VALUES["regime_filter"]))
        ),
    }


def sample_local_configs(
    seed_params: dict[str, Any],
    *,
    samples: int,
    width: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    options = local_refinement_options(seed_params, width)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(row: dict[str, Any]) -> None:
        normalized = {key: coerce_tune_param(key, row[key]) for key in BTC_TUNE_KEYS if key in row}
        key = config_key(normalized)
        if key in seen:
            return
        seen.add(key)
        rows.append(normalized)

    add(seed_params)
    for key, values in options.items():
        for value in values:
            row = {**seed_params, key: value}
            add(row)

    attempts = 0
    target = max(int(samples), len(rows))
    while len(rows) < target and attempts < target * 100:
        attempts += 1
        row = {key: values[int(rng.integers(len(values)))] for key, values in options.items()}
        add(row)
    return rows[:target]


def select_refinement_seeds(
    scores: pd.DataFrame,
    *,
    max_seeds: int,
    seed: int,
) -> pd.DataFrame:
    if scores.empty:
        return scores
    rows = scores.copy()
    if "error" in rows.columns:
        rows = rows[rows["error"].isna()].copy()
    rows = rows.dropna(subset=[key for key in BTC_TUNE_KEYS if key in rows.columns])
    if rows.empty:
        return rows

    rng = np.random.default_rng(seed)
    picks: list[pd.Series] = []
    seen: set[tuple[Any, ...]] = set()

    def add(frame: pd.DataFrame, reason: str, limit: int) -> None:
        nonlocal picks
        if limit <= 0 or frame.empty:
            return
        for _, row in frame.iterrows():
            params = btc_params_from_row(row)
            key = config_key(params)
            if key in seen:
                continue
            seen.add(key)
            item = row.copy()
            item["refine_seed_reason"] = reason
            picks.append(item)
            if len(picks) >= max_seeds or sum(1 for pick in picks if pick.get("refine_seed_reason") == reason) >= limit:
                break

    lowpass_cols = [col for col in ["optimization_score", "stability_score", "local_pass_rate", "robust_score"] if col in rows.columns]
    if lowpass_cols:
        add(rows.sort_values(lowpass_cols, ascending=[False] * len(lowpass_cols), na_position="last"), "top_lowpass", max(1, max_seeds // 4))

    selection = rows[rows.get("selection_pass", pd.Series(False, index=rows.index)).fillna(False)].copy()
    both = selection[selection.get("oos_pass", pd.Series(False, index=selection.index)).fillna(False)].copy()
    raw_cols = [col for col in ["robust_score", "oos_net_r", "validation_net_r", "train_net_r"] if col in rows.columns]
    if raw_cols:
        add(both.sort_values(raw_cols, ascending=[False] * len(raw_cols), na_position="last"), "both_pass_raw", max(1, max_seeds // 4))
        add(selection.sort_values(raw_cols, ascending=[False] * len(raw_cols), na_position="last"), "selection_pass_raw", max(1, max_seeds // 4))

    if len(picks) < max_seeds:
        okay = rows.copy()
        mask = pd.Series(False, index=okay.index)
        if "selection_pass" in okay.columns:
            mask |= okay["selection_pass"].fillna(False).astype(bool)
        robust_values = pd.to_numeric(okay.get("robust_score", pd.Series(np.nan, index=okay.index)), errors="coerce")
        robust_finite = robust_values[np.isfinite(robust_values)]
        if "oos_pass" in okay.columns:
            mask |= okay["oos_pass"].fillna(False).astype(bool) & (robust_values >= 0.0)
        for col in ["optimization_score", "robust_score", "lowpass_robust_score"]:
            if col in okay.columns:
                values = pd.to_numeric(okay[col], errors="coerce")
                finite = values[np.isfinite(values)]
                if not finite.empty:
                    floor = 0.0 if col == "robust_score" else -50.0
                    mask |= values >= max(float(finite.quantile(0.65)), floor)
        okay = okay[mask].copy()
        if not okay.empty:
            order = rng.permutation(len(okay))
            add(okay.iloc[order], "random_okayish", max_seeds - len(picks))

    if len(picks) < max_seeds and raw_cols:
        add(rows.sort_values(raw_cols, ascending=[False] * len(raw_cols), na_position="last"), "fallback_raw", max_seeds - len(picks))

    out = pd.DataFrame([row.to_dict() for row in picks[:max_seeds]])
    if not out.empty:
        if "refine_seed_id" in out.columns:
            out = out.drop(columns=["refine_seed_id"])
        out.insert(0, "refine_seed_id", range(1, len(out) + 1))
    return out


def refinement_grid_from_scores(
    scores: pd.DataFrame,
    *,
    max_seeds: int,
    samples_per_seed: int,
    neighbor_width: int,
    seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    seeds = select_refinement_seeds(scores, max_seeds=max_seeds, seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for _, seed_row in seeds.iterrows():
        seed_params = btc_params_from_row(seed_row)
        local_rows = sample_local_configs(
            seed_params,
            samples=samples_per_seed,
            width=neighbor_width,
            rng=rng,
        )
        for local in local_rows:
            key = config_key(local)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    **local,
                    "refine_seed_id": int(seed_row["refine_seed_id"]),
                    "refine_seed_reason": str(seed_row.get("refine_seed_reason", "")),
                    "refine_seed_optimization_score": float(seed_row.get("optimization_score", np.nan)),
                    "refine_seed_robust_score": float(seed_row.get("robust_score", np.nan)),
                }
            )
    return rows, seeds


def evaluate_btc_grid(
    df: pd.DataFrame,
    base_cfg: WolfeConfig,
    *,
    symbol: str,
    grid: list[dict[str, Any]],
    min_train_trades: int = 8,
    min_validation_trades: int = 3,
    lowpass: bool = True,
    lowpass_radius: float = 0.45,
    lowpass_min_neighbors: int = 9,
    lowpass_outlier_penalty: float = 0.65,
) -> pd.DataFrame:
    frame = ensure_ohlcv_frame(df)
    start = pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC")
    end = pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC")
    train_end = start + (end - start) * 0.60
    validation_end = start + (end - start) * 0.80
    rows: list[dict[str, Any]] = []

    config_fields = set(WolfeConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    for idx, params in enumerate(grid, start=1):
        config_params = {key: params[key] for key in params if key in config_fields}
        meta = {key: params[key] for key in params if key not in config_fields}
        cfg = WolfeConfig.from_mapping({**asdict(base_cfg), **config_params})
        try:
            trades = run_backtest(frame, cfg, symbol=symbol)
        except Exception as exc:  # noqa: BLE001 - bad research configs should not kill the sweep.
            rows.append({"grid_index": idx, "error": str(exc), **meta, **asdict(cfg)})
            continue
        buckets = split_trades(trades, train_end=train_end, validation_end=validation_end)
        train_m = strategy_metrics(buckets["train"])
        val_m = strategy_metrics(buckets["validation"])
        oos_m = strategy_metrics(buckets["oos"])
        selection_pass = (
            train_m["trades"] >= min_train_trades
            and val_m["trades"] >= min_validation_trades
            and train_m["net_r"] > 0.0
            and val_m["net_r"] > 0.0
            and train_m["profit_factor"] >= 1.05
            and val_m["profit_factor"] >= 1.05
        )
        oos_pass = (
            oos_m["trades"] >= min_validation_trades
            and oos_m["net_r"] > 0.0
            and oos_m["profit_factor"] >= 1.05
        )
        robust_score = (
            min(train_m["avg_r"], val_m["avg_r"]) * 120.0
            + min(train_m["profit_factor"], val_m["profit_factor"], 5.0) * 5.0
            + min(train_m["net_r"], val_m["net_r"]) * 0.5
            - abs(min(train_m["max_dd_r"], val_m["max_dd_r"])) * 0.25
        )
        if train_m["trades"] < min_train_trades:
            robust_score -= (min_train_trades - train_m["trades"]) * 15.0
        if val_m["trades"] < min_validation_trades:
            robust_score -= (min_validation_trades - val_m["trades"]) * 20.0
        rows.append(
            {
                "grid_index": idx,
                "robust_score": float(robust_score),
                "train_start": start,
                "train_end": train_end,
                "validation_end": validation_end,
                "data_end": end,
                "selection_pass": selection_pass,
                "oos_pass": oos_pass,
                **meta,
                **asdict(cfg),
                **metric_row(buckets["train"], "train"),
                **metric_row(buckets["validation"], "validation"),
                **metric_row(buckets["oos"], "oos"),
                **metric_row(trades, "all"),
            }
        )
    out = pd.DataFrame(rows)
    if lowpass and "robust_score" in out.columns:
        out = median_lowpass_scores(
            out,
            radius=lowpass_radius,
            min_neighbors=lowpass_min_neighbors,
            outlier_penalty=lowpass_outlier_penalty,
        )
    sort_cols = (
        [
            "optimization_score",
            "stability_score",
            "local_pass_rate",
            "robust_score",
            "oos_net_r",
            "validation_avg_r",
            "train_avg_r",
        ]
        if lowpass and "optimization_score" in out.columns
        else ["robust_score", "oos_net_r", "validation_avg_r", "train_avg_r"]
    )
    sort_cols = [col for col in sort_cols if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return out.reset_index(drop=True)


def tune_btc(
    df: pd.DataFrame,
    base_cfg: WolfeConfig,
    *,
    symbol: str,
    max_configs: int = 0,
    min_train_trades: int = 8,
    min_validation_trades: int = 3,
    lowpass: bool = True,
    lowpass_radius: float = 0.45,
    lowpass_min_neighbors: int = 9,
    lowpass_outlier_penalty: float = 0.65,
) -> pd.DataFrame:
    grid = btc_parameter_grid(max_configs, seed=42)
    return evaluate_btc_grid(
        df,
        base_cfg,
        symbol=symbol,
        grid=grid,
        min_train_trades=min_train_trades,
        min_validation_trades=min_validation_trades,
        lowpass=lowpass,
        lowpass_radius=lowpass_radius,
        lowpass_min_neighbors=lowpass_min_neighbors,
        lowpass_outlier_penalty=lowpass_outlier_penalty,
    )


def refine_btc(
    df: pd.DataFrame,
    base_cfg: WolfeConfig,
    *,
    symbol: str,
    scores: pd.DataFrame,
    max_seeds: int,
    samples_per_seed: int,
    neighbor_width: int,
    seed: int,
    min_train_trades: int = 8,
    min_validation_trades: int = 3,
    lowpass: bool = True,
    lowpass_radius: float = 0.45,
    lowpass_min_neighbors: int = 9,
    lowpass_outlier_penalty: float = 0.65,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid, seeds = refinement_grid_from_scores(
        scores,
        max_seeds=max_seeds,
        samples_per_seed=samples_per_seed,
        neighbor_width=neighbor_width,
        seed=seed,
    )
    result = evaluate_btc_grid(
        df,
        base_cfg,
        symbol=symbol,
        grid=grid,
        min_train_trades=min_train_trades,
        min_validation_trades=min_validation_trades,
        lowpass=lowpass,
        lowpass_radius=lowpass_radius,
        lowpass_min_neighbors=lowpass_min_neighbors,
        lowpass_outlier_penalty=lowpass_outlier_penalty,
    )
    return result, seeds


def save_best_config(path: Path, symbol: str, row: pd.Series) -> None:
    fields = tuple(WolfeConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    payload: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            payload = existing
    payload[symbol.upper()] = {
        key: row[key].item() if hasattr(row[key], "item") else row[key]
        for key in fields
        if key in row.index and pd.notna(row[key])
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return ensure_ohlcv_frame(frame)


def build_synthetic() -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=600, freq="5min", tz="UTC")
    base = np.linspace(100.0, 108.0, len(times)) + np.sin(np.linspace(0, 20, len(times))) * 2.0
    noise = np.sin(np.linspace(0, 80, len(times))) * 0.35
    close = base + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.45
    low = np.minimum(open_, close) - 0.45
    volume = np.full(len(times), 100.0)
    return pd.DataFrame(
        {
            "open_time": times,
            "close_time": times + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def print_metrics(label: str, trades: pd.DataFrame) -> None:
    metrics = strategy_metrics(trades)
    print(
        f"{label}: trades={metrics['trades']:.0f} net_r={metrics['net_r']:.2f} "
        f"avg_r={metrics['avg_r']:.3f} wr={metrics['win_rate']:.1%} "
        f"pf={metrics['profit_factor']:.3f} max_dd_r={metrics['max_dd_r']:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research/backtest BTC Wolfe Wave setups on Bybit OHLCV.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=900)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--cache-csv", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--mintick", type=float)
    parser.add_argument("--no-auto-mintick", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--max-configs", type=int, default=500)
    parser.add_argument("--refine-from-tuning", type=Path)
    parser.add_argument("--refine-seeds", type=int, default=12)
    parser.add_argument("--refine-samples-per-seed", type=int, default=80)
    parser.add_argument("--refine-neighbor-width", type=int, default=1)
    parser.add_argument("--refine-random-seed", type=int, default=725)
    parser.add_argument("--pattern-tfs", help="Comma-separated pattern timeframes to include in tuning/refinement.")
    parser.add_argument("--regime-filters", help="Comma-separated regime filters to include in tuning/refinement.")
    parser.add_argument("--disable-lowpass", action="store_true")
    parser.add_argument("--lowpass-radius", type=float, default=0.45)
    parser.add_argument("--lowpass-min-neighbors", type=int, default=9)
    parser.add_argument("--lowpass-outlier-penalty", type=float, default=0.65)
    parser.add_argument("--allow-unstable-lowpass-save", action="store_true")
    parser.add_argument("--require-oos-pass-to-save", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/wolfe_wave_btc"))
    parser.add_argument("--save-best-config", type=Path, default=Path("bot/configs/wolfe_wave_configs.json"))
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def split_cli_values(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    return values or None


def apply_tune_value_filters(args: argparse.Namespace) -> None:
    pattern_tfs = split_cli_values(args.pattern_tfs)
    if pattern_tfs is not None:
        normalized = tuple(normalize_timeframe(value) for value in pattern_tfs)
        unknown = [value for value in normalized if value not in BTC_TUNE_VALUES["pattern_tf"]]
        if unknown:
            raise ValueError(f"Unknown pattern timeframe(s): {', '.join(unknown)}")
        BTC_TUNE_VALUES["pattern_tf"] = normalized

    regime_filters = split_cli_values(args.regime_filters)
    if regime_filters is not None:
        normalized_filters = tuple(value.lower() for value in regime_filters)
        unknown_filters = [value for value in normalized_filters if value not in BTC_TUNE_VALUES["regime_filter"]]
        if unknown_filters:
            raise ValueError(f"Unknown regime filter(s): {', '.join(unknown_filters)}")
        BTC_TUNE_VALUES["regime_filter"] = normalized_filters


def config_from_file(path: Path | None, symbol: str, interval: str) -> WolfeConfig:
    base = WolfeConfig(exec_tf=normalize_timeframe(interval))
    if path is None:
        return base
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and symbol.upper() in payload:
        payload = payload[symbol.upper()]
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Wolfe config payload in {path}")
    return WolfeConfig.from_mapping({**asdict(base), **payload, "exec_tf": normalize_timeframe(interval)})


def load_or_fetch_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.synthetic_smoke:
        return build_synthetic()
    if args.csv:
        return window_frame(load_ohlcv_csv(args.csv), args)
    cache_csv = args.cache_csv or Path("scripts/data") / f"{bybit_symbol(args.symbol).lower()}_{normalize_timeframe(args.interval)}_bybit.csv"
    if cache_csv.exists():
        frame = load_ohlcv_csv(cache_csv)
        if not frame.empty:
            return window_frame(frame, args)
    end = parse_utc_datetime(args.end) if args.end else datetime.now(timezone.utc)
    start = parse_utc_datetime(args.start) if args.start else end - timedelta(days=args.days)
    frame = fetch_bybit_klines(args.symbol, args.interval, start, end)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_csv, index=False)
    return window_frame(frame, args)


def window_frame(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = ensure_ohlcv_frame(frame)
    if out.empty:
        return out
    end = parse_utc_datetime(args.end) if args.end else pd.Timestamp(out["open_time"].iloc[-1]).to_pydatetime()
    start = parse_utc_datetime(args.start) if args.start else end - timedelta(days=args.days)
    mask = (pd.to_datetime(out["open_time"], utc=True) >= pd.Timestamp(start)) & (
        pd.to_datetime(out["open_time"], utc=True) <= pd.Timestamp(end)
    )
    return out.loc[mask].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.symbol = bybit_symbol(args.symbol)
    args.interval = normalize_timeframe(args.interval)
    apply_tune_value_filters(args)
    frame = load_or_fetch_data(args)
    cfg = config_from_file(args.config_json, args.symbol, args.interval)
    if args.mintick is not None:
        cfg = WolfeConfig.from_mapping({**asdict(cfg), "mintick": float(args.mintick)})
    elif not args.synthetic_smoke and not args.no_auto_mintick:
        try:
            cfg = WolfeConfig.from_mapping({**asdict(cfg), "mintick": fetch_bybit_mintick(args.symbol)})
        except Exception as exc:  # noqa: BLE001 - research can still run with config/default mintick.
            print(f"{args.symbol}: mintick lookup failed ({exc}); using mintick={cfg.mintick:g}", flush=True)
    print(
        f"{args.symbol}: loaded {len(frame)} {args.interval} bars "
        f"{pd.Timestamp(frame['open_time'].iloc[0]).date()} -> {pd.Timestamp(frame['open_time'].iloc[-1]).date()} "
        f"mintick={cfg.mintick:g}"
    )

    if args.tune:
        seed_table = pd.DataFrame()
        if args.refine_from_tuning is not None:
            if args.output_prefix == Path("scripts/wolfe_wave_btc"):
                args.output_prefix = Path("scripts/wolfe_wave_btc_refined")
            source_scores = pd.read_csv(args.refine_from_tuning)
            result, seed_table = refine_btc(
                frame,
                cfg,
                symbol=args.symbol,
                scores=source_scores,
                max_seeds=args.refine_seeds,
                samples_per_seed=args.refine_samples_per_seed,
                neighbor_width=args.refine_neighbor_width,
                seed=args.refine_random_seed,
                lowpass=not args.disable_lowpass,
                lowpass_radius=args.lowpass_radius,
                lowpass_min_neighbors=args.lowpass_min_neighbors,
                lowpass_outlier_penalty=args.lowpass_outlier_penalty,
            )
        else:
            result = tune_btc(
                frame,
                cfg,
                symbol=args.symbol,
                max_configs=args.max_configs,
                lowpass=not args.disable_lowpass,
                lowpass_radius=args.lowpass_radius,
                lowpass_min_neighbors=args.lowpass_min_neighbors,
                lowpass_outlier_penalty=args.lowpass_outlier_penalty,
            )
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        result_path = args.output_prefix.with_suffix(".tuning.csv")
        result.to_csv(result_path, index=False)
        print(f"Saved tuning table: {result_path}")
        if not seed_table.empty:
            seeds_path = args.output_prefix.with_suffix(".seeds.csv")
            seed_table.to_csv(seeds_path, index=False)
            print(f"Saved refinement seeds: {seeds_path}")
        if not result.empty and "robust_score" in result.columns:
            print(result.head(20).to_string(index=False))
            save_candidates = result.copy()
            if not args.allow_unstable_lowpass_save and "selection_pass" in result.columns:
                save_candidates = save_candidates[save_candidates["selection_pass"].fillna(False)].copy()
            if args.require_oos_pass_to_save and "oos_pass" in save_candidates.columns:
                save_candidates = save_candidates[save_candidates["oos_pass"].fillna(False)].copy()
            if save_candidates.empty:
                print("No candidate passed the requested save gates; best config left unchanged.")
            else:
                best = save_candidates.iloc[0]
                if "selection_pass" in best.index and not bool(best["selection_pass"]):
                    print(
                        "Best low-pass candidate did not pass the raw train/validation gate; "
                        "saving because --allow-unstable-lowpass-save was set."
                    )
                save_best_config(args.save_best_config, args.symbol, best)
                print(f"Saved best config: {args.save_best_config}")
        return

    signals = find_wolfe_signals(frame, cfg, symbol=args.symbol)
    trades = run_backtest(frame, cfg, symbol=args.symbol, precomputed_signals=signals)
    signals_path = args.output_prefix.with_suffix(".signals.csv")
    trades_path = args.output_prefix.with_suffix(".trades.csv")
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    signal_rows(signals).to_csv(signals_path, index=False)
    trades.to_csv(trades_path, index=False)
    print(f"signals={len(signals)} trades={len(trades)}")
    print_metrics("all", trades)
    if not trades.empty:
        frame_start = pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC")
        frame_end = pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC")
        train_end = frame_start + (frame_end - frame_start) * 0.60
        validation_end = frame_start + (frame_end - frame_start) * 0.80
        for label, bucket in split_trades(trades, train_end=train_end, validation_end=validation_end).items():
            print_metrics(label, bucket)
    print(f"Saved signals: {signals_path}")
    print(f"Saved trades: {trades_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
