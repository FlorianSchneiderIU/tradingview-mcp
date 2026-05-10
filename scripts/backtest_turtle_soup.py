from __future__ import annotations

import argparse
import math
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests


BINANCE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}
RESAMPLE_RULE = {
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "W-MON",
}
TIMEFRAME_ALIASES = {
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "60": "1h",
    "240": "4h",
    "1D": "1d",
    "1W": "1w",
}

BFM_LINE_FEATURE_COLUMNS = [
    "bfm_same_entry_gap_atr",
    "bfm_same_mid_gap_atr",
    "bfm_same_wick_gap_atr",
    "bfm_same_signed_entry_gap_atr",
    "bfm_same_inside_zone",
    "bfm_same_near_0_5atr",
    "bfm_same_near_1_0atr",
    "bfm_same_tf_minutes",
    "bfm_same_set",
    "bfm_opp_entry_dist_atr",
    "bfm_opp_close_dist_atr",
    "bfm_opp_tf_minutes",
    "bfm_opp_set",
]
BFM_CHANNEL_FEATURE_TIMEFRAMES = ("1h", "4h", "1d")
BFM_CHANNEL_FEATURE_NAMES = (
    "channel_available",
    "channel_pos",
    "channel_pos_clipped",
    "channel_above",
    "channel_below",
    "channel_width_atr",
    "upper_slope_atr",
    "lower_slope_atr",
    "width_slope_atr",
    "trend_slope_atr",
    "widening",
    "closing",
    "parallel",
    "trend_up",
    "trend_down",
    "trend_aligned",
)
BFM_CHANNEL_FEATURE_COLUMNS = [
    f"bfm_{timeframe}_{name}"
    for timeframe in BFM_CHANNEL_FEATURE_TIMEFRAMES
    for name in BFM_CHANNEL_FEATURE_NAMES
]
BFM_ZONE_FEATURE_COLUMNS = BFM_LINE_FEATURE_COLUMNS + BFM_CHANNEL_FEATURE_COLUMNS
DEFAULT_BFM_ZONE_TIMEFRAMES = "1h,4h,1d"
DEFAULT_BFM_ZONE_TF_SETS = (
    "1h=330:220,264:176,211:141,169:112;"
    "4h=180:120,144:96,115:77,92:61;"
    "1d=105:70,84:56,67:45,54:36"
)


def normalize_binance_spot_symbol(symbol: str) -> str:
    raw = symbol.strip().upper()
    if ":" in raw:
        exchange, raw = raw.split(":", 1)
        if exchange != "BINANCE":
            raise ValueError(f"Only BINANCE spot symbols are supported by this data loader, got {symbol!r}.")
    if raw.endswith(".P"):
        raise ValueError(
            f"{symbol!r} looks like a futures/perpetual TradingView symbol. "
            "This backtest fetches Binance spot klines; use a spot chart such as BINANCE:ETHUSDT for parity."
        )
    return raw.replace("/", "").replace("-", "")


def normalize_timeframe(timeframe: str) -> str:
    tf = timeframe.strip()
    if tf in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[tf]
    tf = tf.lower()
    if tf in INTERVAL_MS:
        return tf
    raise ValueError(f"Unsupported timeframe: {timeframe!r}. Supported: {', '.join(INTERVAL_MS)}")


def parse_utc_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    if len(normalized) == 10:
        normalized = f"{normalized}T00:00:00+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def round_to_mintick(value: float, mintick: float) -> float:
    if mintick <= 0:
        return value
    return round(round(value / mintick) * mintick, 10)


@dataclass
class Trade:
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    r_multiple: float
    exit_reason: str
    hold_bars: int
    exec_tf: str
    structure_tf: str
    entry_mode: str
    zone_tf: str
    zone_top: float
    zone_bottom: float
    sweep_time: pd.Timestamp
    choch_time: pd.Timestamp
    signal_time: pd.Timestamp
    ob_top: float
    ob_bottom: float
    sweep_index: int
    choch_index: int
    signal_index: int
    entry_index: int
    exit_index: int
    zone_hold_prob: float
    zone_source: str = "ob_break"
    liquidity_level: float = math.nan
    liquidity_pivot_time: pd.Timestamp | None = None
    liquidity_confirm_time: pd.Timestamp | None = None
    liquidity_sfp_strict: bool = False


@dataclass
class Config:
    exec_tf: str
    structure_tf: str
    entry_mode: str
    tf1: str = "4h"
    tf2: str = "1d"
    use_tf1: bool = True
    use_tf2: bool = True
    allow_longs: bool = True
    allow_shorts: bool = True
    prioritize_higher_tf: bool = True
    htf_zone_width_atr: float = 0.25
    zone_penetration_frac: float = 0.50
    htf_left: int = 5
    htf_right: int = 5
    htf_ob_search_bars: int = 50
    structure_left: int = 2
    structure_right: int = 2
    max_structure_bars_to_choch: int = 32
    ob_search_exec_bars: int = 60
    retest_valid_exec_bars: int = 60
    retest_close_pos: float = 0.50
    stop_buffer_atr: float = 0.10
    target_rr: float = 2.0
    max_hold_exec_bars: int = 120
    limit_entry_pos: float = 0.50
    pre_entry_invalidation_mode: str = "OB Or Stop Wick"
    ob_use_body: bool = False
    invalidate_on_close: bool = True
    block_dead_zone: bool = True
    dead_zone_start_hour: int = 5
    dead_zone_end_hour: int = 11
    min_sweep_reclaim_pos: float = 0.70
    htf_bias_mode: str = "none"
    htf_bias_len: int = 20
    min_bias_score: int = 1
    use_first4_return_bias: bool = False
    first4_return_threshold: float = 0.5
    use_first4_range_bias: bool = False
    first4_range_lower: float = 0.30
    first4_range_upper: float = 0.70
    use_prev_day_reversion_bias: bool = False
    prev_day_reversion_threshold: float = 1.0
    use_thursday_bearish_bias: bool = False
    min_sweep_volume_mult: float = 0.0
    max_sweep_volume_mult: float = 0.0
    require_structure_fvg: bool = False
    mintick: float = 0.01
    slippage_ticks: int = 2
    min_entry_risk_pct: float = 0.0
    max_entry_risk_pct: float = math.inf
    max_zone_scan: int = 0
    zone_hold_model_path: str | None = None
    zone_hold_min_prob: float = 0.0
    zone_hold_filter_tf: str = "4h"
    zone_hold_reject_unscored: bool = False
    zone_hold_label_rr: float = 1.0
    zone_hold_label_horizon_bars: int = 288
    zone_hold_mbq_ob_lookback_bars: int = 200
    zone_hold_mbq_confluence_atr: float = 0.50
    zone_hold_bfm_timeframes: str = DEFAULT_BFM_ZONE_TIMEFRAMES
    zone_hold_bfm_tf_sets: str = DEFAULT_BFM_ZONE_TF_SETS
    zone_hold_bfm_invalidation: str = "wick"
    zone_hold_bfm_max_extension_bars: int = 300
    use_sfp_liquidity_zones: bool = False
    sfp_timeframes: str = "15m,1h,4h"
    sfp_left: int = 15
    sfp_right: int = 10
    sfp_level_width_atr: float = 0.15
    sfp_strict: bool = True
    sfp_require_open_reclaim: bool = True
    sfp_max_active_levels: int = 80


_ZONE_HOLD_MODEL_CACHE: dict[str, dict] = {}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    symbol = normalize_binance_spot_symbol(symbol)
    interval = normalize_timeframe(interval)
    rows: list[list] = []
    cursor = start_ms
    interval_ms = INTERVAL_MS[interval]

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        batch = None
        last_error = None
        for attempt in range(5):
            try:
                response = requests.get(BINANCE_URL, params=params, timeout=30)
                response.raise_for_status()
                batch = response.json()
                break
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        if batch is None:
            raise last_error if last_error is not None else RuntimeError("Failed to fetch klines.")
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + interval_ms
        if len(batch) < 1000:
            break
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"No Binance spot klines returned for {symbol} {interval}.")

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    resampled = (
        df.set_index("open_time")
        .resample(RESAMPLE_RULE[timeframe], label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    duration = pd.Timedelta(milliseconds=INTERVAL_MS[timeframe])
    resampled["close_time"] = resampled["open_time"] + duration - pd.Timedelta(milliseconds=1)
    return resampled


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out = df.copy()
    out["atr"] = rma(tr, length)
    return out


def build_confirmed_pivots(values: pd.Series, left: int, right: int, mode: str) -> list[dict]:
    arr = values.to_list()
    pivots: list[dict] = []
    for i in range(left, len(arr) - right):
        window = arr[i - left:i + right + 1]
        center = arr[i]
        if mode == "high":
            ok = center == max(window) and window.index(center) == left
        else:
            ok = center == min(window) and window.index(center) == left
        if ok:
            pivots.append({"pivot_index": i, "value": float(center)})
    return pivots


def find_last_opposite_candle_in_df(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str, use_body: bool) -> tuple[float, float] | None:
    if end_idx < start_idx:
        return None
    opens = df["open"].to_list()
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    closes = df["close"].to_list()
    return find_last_opposite_candle(opens, highs, lows, closes, start_idx, end_idx, direction, use_body)


def find_smc_order_block_in_df(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str, use_body: bool) -> tuple[float, float] | None:
    if end_idx < start_idx:
        return None
    window = df.iloc[start_idx:end_idx + 1]
    if window.empty:
        return None
    if direction == "long":
        idx = int(window["low"].astype(float).idxmin())
    else:
        idx = int(window["high"].astype(float).idxmax())
    row = df.iloc[idx]
    if use_body:
        top = max(float(row["open"]), float(row["close"]))
        bottom = min(float(row["open"]), float(row["close"]))
    else:
        top = float(row["high"])
        bottom = float(row["low"])
    return top, bottom


def build_htf_zone_events(
    exec_df: pd.DataFrame,
    timeframe: str,
    left: int,
    right: int,
    zone_width_atr: float,
    ob_search_bars: int = 50,
    use_body: bool = False,
) -> tuple[list[dict], list[dict]]:
    del zone_width_atr
    htf = resample_ohlc(exec_df, timeframe)
    ph_conf: list[float | None] = [None] * len(htf)
    pl_conf: list[float | None] = [None] * len(htf)
    ph_idx_conf: list[int | None] = [None] * len(htf)
    pl_idx_conf: list[int | None] = [None] * len(htf)

    for item in build_confirmed_pivots(htf["high"], left, right, "high"):
        confirm_idx = item["pivot_index"] + right
        if confirm_idx < len(htf):
            ph_conf[confirm_idx] = item["value"]
            ph_idx_conf[confirm_idx] = item["pivot_index"]

    for item in build_confirmed_pivots(htf["low"], left, right, "low"):
        confirm_idx = item["pivot_index"] + right
        if confirm_idx < len(htf):
            pl_conf[confirm_idx] = item["value"]
            pl_idx_conf[confirm_idx] = item["pivot_index"]

    high_events: list[dict] = []
    low_events: list[dict] = []
    active_high = math.nan
    active_low = math.nan
    active_high_idx: int | None = None
    active_low_idx: int | None = None
    high_crossed = False
    low_crossed = False
    closes = htf["close"].to_list()

    for i in range(len(htf)):
        if ph_conf[i] is not None:
            active_high = float(ph_conf[i])
            active_high_idx = ph_idx_conf[i]
            high_crossed = False
        if pl_conf[i] is not None:
            active_low = float(pl_conf[i])
            active_low_idx = pl_idx_conf[i]
            low_crossed = False

        prev_close = closes[i - 1] if i > 0 else closes[i]
        bull_break = not math.isnan(active_high) and not high_crossed and closes[i] > active_high and prev_close <= active_high
        bear_break = not math.isnan(active_low) and not low_crossed and closes[i] < active_low and prev_close >= active_low

        if bull_break and active_high_idx is not None:
            start_idx = max(active_high_idx, i - ob_search_bars)
            ob = find_smc_order_block_in_df(htf, start_idx, i - 1, "long", use_body)
            if ob is not None:
                top, bottom = ob
                low_events.append({
                    "time": htf.iloc[i]["close_time"],
                    "top": top,
                    "bottom": bottom,
                    "width": top - bottom,
                })
            high_crossed = True

        if bear_break and active_low_idx is not None:
            start_idx = max(active_low_idx, i - ob_search_bars)
            ob = find_smc_order_block_in_df(htf, start_idx, i - 1, "short", use_body)
            if ob is not None:
                top, bottom = ob
                high_events.append({
                    "time": htf.iloc[i]["close_time"],
                    "top": top,
                    "bottom": bottom,
                    "width": top - bottom,
                })
            low_crossed = True

    return high_events, low_events


def parse_timeframe_list(raw: str | None, default: str = "15m,1h,4h") -> list[str]:
    text = raw if raw is not None and str(raw).strip() else default
    out: list[str] = []
    for chunk in str(text).split(","):
        timeframe = normalize_timeframe(chunk.strip())
        if timeframe not in out:
            out.append(timeframe)
    return out


def build_sfp_liquidity_zone_events(
    exec_df: pd.DataFrame,
    *,
    timeframes: list[str],
    left: int,
    right: int,
    width_atr: float,
) -> tuple[list[dict], list[dict]]:
    """Confirmed pivot liquidity levels used by SFP/Turtle Soup style sweeps."""
    high_events: list[dict] = []
    low_events: list[dict] = []
    left = max(1, int(left))
    right = max(1, int(right))
    width_atr = max(0.01, float(width_atr))

    for timeframe in timeframes:
        htf = add_atr(resample_ohlc(exec_df, timeframe))
        if len(htf) <= left + right:
            continue
        atrs = htf["atr"].bfill().ffill().to_list()
        for item in build_confirmed_pivots(htf["high"], left, right, "high"):
            pivot_idx = int(item["pivot_index"])
            confirm_idx = pivot_idx + right
            if confirm_idx >= len(htf):
                continue
            level = float(item["value"])
            width = max(float(atrs[confirm_idx]) * width_atr, abs(level) * 0.00001)
            high_events.append({
                "time": htf.iloc[confirm_idx]["close_time"],
                "pivot_time": htf.iloc[pivot_idx]["close_time"],
                "confirm_time": htf.iloc[confirm_idx]["close_time"],
                "top": level + width,
                "bottom": level,
                "width": width,
                "level": level,
                "source": "sfp_pivot",
                "tf": timeframe,
            })
        for item in build_confirmed_pivots(htf["low"], left, right, "low"):
            pivot_idx = int(item["pivot_index"])
            confirm_idx = pivot_idx + right
            if confirm_idx >= len(htf):
                continue
            level = float(item["value"])
            width = max(float(atrs[confirm_idx]) * width_atr, abs(level) * 0.00001)
            low_events.append({
                "time": htf.iloc[confirm_idx]["close_time"],
                "pivot_time": htf.iloc[pivot_idx]["close_time"],
                "confirm_time": htf.iloc[confirm_idx]["close_time"],
                "top": level,
                "bottom": level - width,
                "width": width,
                "level": level,
                "source": "sfp_pivot",
                "tf": timeframe,
            })

    high_events.sort(key=lambda item: item["time"])
    low_events.sort(key=lambda item: item["time"])
    return high_events, low_events


def build_structure_choch_events(exec_df: pd.DataFrame, timeframe: str, left: int, right: int) -> list[dict]:
    structure = resample_ohlc(exec_df, timeframe)
    ph_conf: list[float | None] = [None] * len(structure)
    pl_conf: list[float | None] = [None] * len(structure)

    for item in build_confirmed_pivots(structure["high"], left, right, "high"):
        confirm_idx = item["pivot_index"] + right
        if confirm_idx < len(structure):
            ph_conf[confirm_idx] = item["value"]

    for item in build_confirmed_pivots(structure["low"], left, right, "low"):
        confirm_idx = item["pivot_index"] + right
        if confirm_idx < len(structure):
            pl_conf[confirm_idx] = item["value"]

    active_high = math.nan
    active_low = math.nan
    high_crossed = False
    low_crossed = False
    trend = 0
    events: list[dict] = []
    closes = structure["close"].to_list()

    for i in range(len(structure)):
        if ph_conf[i] is not None:
            active_high = float(ph_conf[i])
            high_crossed = False
        if pl_conf[i] is not None:
            active_low = float(pl_conf[i])
            low_crossed = False

        prev_close = closes[i - 1] if i > 0 else closes[i]
        bull_break = not math.isnan(active_high) and not high_crossed and closes[i] > active_high and prev_close <= active_high
        bear_break = not math.isnan(active_low) and not low_crossed and closes[i] < active_low and prev_close >= active_low

        if bull_break and trend == -1:
            events.append({
                "time": structure.iloc[i]["close_time"],
                "direction": "bull",
                "break_level": active_high,
                "has_fvg": bool(structure.iloc[i]["low"] > structure.iloc[i - 2]["high"]) if i >= 2 else False,
            })
        if bear_break and trend == 1:
            events.append({
                "time": structure.iloc[i]["close_time"],
                "direction": "bear",
                "break_level": active_low,
                "has_fvg": bool(structure.iloc[i]["high"] < structure.iloc[i - 2]["low"]) if i >= 2 else False,
            })

        if bull_break:
            high_crossed = True
            trend = 1
        if bear_break:
            low_crossed = True
            trend = -1

    return events


def build_htf_bias_events(exec_df: pd.DataFrame, timeframe: str, length: int) -> list[dict]:
    htf = resample_ohlc(exec_df, timeframe)
    htf["ema"] = htf["close"].ewm(span=length, adjust=False).mean()
    htf["ema_prev"] = htf["ema"].shift(1)
    events: list[dict] = []
    for _, row in htf.iterrows():
        bias = 0
        if pd.notna(row["ema_prev"]):
            if row["close"] > row["ema"] and row["ema"] >= row["ema_prev"]:
                bias = 1
            elif row["close"] < row["ema"] and row["ema"] <= row["ema_prev"]:
                bias = -1
        events.append({
            "time": row["close_time"],
            "bias": bias,
        })
    return events


def build_htf_sma_bias_events(exec_df: pd.DataFrame, timeframe: str, length: int) -> list[dict]:
    htf = resample_ohlc(exec_df, timeframe)
    htf["sma"] = htf["close"].rolling(length).mean()
    events: list[dict] = []
    for _, row in htf.iterrows():
        bias = 0
        if pd.notna(row["sma"]):
            if row["close"] > row["sma"]:
                bias = 1
            elif row["close"] < row["sma"]:
                bias = -1
        events.append({
            "time": row["close_time"],
            "bias": bias,
        })
    return events


def build_daily_context(exec_df: pd.DataFrame) -> dict[pd.Timestamp, dict]:
    h1 = resample_ohlc(exec_df, "1h")
    day = resample_ohlc(exec_df, "1d")
    context: dict[pd.Timestamp, dict] = {}

    for i, row in day.iterrows():
        day_start = row["open_time"]
        first4 = h1[(h1["open_time"] >= day_start) & (h1["open_time"] < day_start + pd.Timedelta(hours=4))]
        first4_ret = math.nan
        first4_range_pos = math.nan
        if len(first4) >= 4:
            first4_open = float(first4.iloc[0]["open"])
            first4_close = float(first4.iloc[3]["close"])
            first4_high = float(first4["high"].max())
            first4_low = float(first4["low"].min())
            first4_ret = (first4_close / first4_open - 1.0) * 100.0
            rng = first4_high - first4_low
            first4_range_pos = (first4_close - first4_low) / rng if rng > 0 else math.nan

        prev_day_ret = math.nan
        if i > 0:
            prev_open = float(day.iloc[i - 1]["open"])
            prev_close = float(day.iloc[i - 1]["close"])
            prev_day_ret = (prev_close / prev_open - 1.0) * 100.0

        context[day_start] = {
            "weekday": day_start.day_name(),
            "first4_ret": first4_ret,
            "first4_range_pos": first4_range_pos,
            "prev_day_ret": prev_day_ret,
        }

    return context


def current_day_context(day_context: dict[pd.Timestamp, dict], now: pd.Timestamp) -> dict:
    day_start = pd.Timestamp(now).floor("D")
    ctx = day_context.get(day_start)
    if ctx is None:
        return {"first4_ret": math.nan, "first4_range_pos": math.nan, "prev_day_ret": math.nan}
    return ctx


def zone_mid(zone: dict) -> float:
    return (float(zone["top"]) + float(zone["bottom"])) / 2.0


def confluence_counts(zone: dict, same_zones: list[dict], opp_zones: list[dict], threshold: float) -> tuple[int, int]:
    mid = zone_mid(zone)
    same_count = sum(
        1
        for other in same_zones
        if other.get("id") != zone.get("id") and abs(mid - zone_mid(other)) <= threshold
    )
    opp_count = sum(1 for other in opp_zones if abs(mid - zone_mid(other)) <= threshold)
    return same_count + opp_count, same_count


def load_zone_hold_model(path: str) -> dict:
    cached = _ZONE_HOLD_MODEL_CACHE.get(path)
    if cached is not None:
        return cached

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "zone_hold_model_path requires joblib/sklearn. Run with the project .venv or install joblib."
        ) from exc

    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload or "feature_columns" not in payload:
        raise ValueError(f"Unsupported zone-hold model payload: {path}")
    _ZONE_HOLD_MODEL_CACHE[path] = payload
    return payload


def predict_zone_hold_probability(model_payload: dict, feature_row: dict) -> float:
    columns = list(model_payload["feature_columns"])
    data = {column: [float(feature_row.get(column, math.nan))] for column in columns}
    frame = pd.DataFrame(data)
    prob = model_payload["model"].predict_proba(frame)[:, 1][0]
    return float(prob)


@dataclass(frozen=True)
class BfmLineRecord:
    side: str
    timeframe: str
    tf_minutes: float
    set_number: int
    start_exec_index: int
    end_exec_index: int
    exec_slope: float
    exec_intercept: float

    def evaluate(self, index: int) -> float:
        return float(self.exec_slope * index + self.exec_intercept)


@dataclass(frozen=True)
class BfmFeatureProjection:
    timeframes: tuple[str, ...]
    records: tuple[BfmLineRecord, ...]


def zone_hold_model_requires_bfm(model_payload: dict | None) -> bool:
    if model_payload is None:
        return False
    columns = set(model_payload.get("feature_columns", []))
    return any(column in columns for column in BFM_ZONE_FEATURE_COLUMNS)


def _parse_bfm_sets(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in str(raw).split(","):
        text = chunk.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Invalid BFM pivot set {text!r}; expected left:right.")
        left_raw, right_raw = text.split(":", 1)
        left = int(left_raw)
        right = int(right_raw)
        if left <= 0 or right <= 0:
            raise ValueError(f"Invalid BFM pivot set {text!r}; left/right must be positive.")
        out.append((left, right))
    if not out:
        raise ValueError("At least one BFM pivot set is required.")
    return out


def parse_bfm_feature_timeframes(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        raw = DEFAULT_BFM_ZONE_TIMEFRAMES
    out: list[str] = []
    for chunk in str(raw).split(","):
        text = chunk.strip()
        if not text:
            continue
        timeframe = normalize_timeframe(text)
        if timeframe not in out:
            out.append(timeframe)
    return out or [normalize_timeframe("1h")]


def parse_bfm_feature_tf_sets(raw: str | None, timeframes: list[str]) -> dict[str, list[tuple[int, int]]]:
    default_sets = _parse_bfm_sets("300:200,240:160,192:128,154:102")
    out = {timeframe: list(default_sets) for timeframe in timeframes}
    text = str(raw or DEFAULT_BFM_ZONE_TF_SETS).strip()
    if not text:
        return out
    if "=" not in text:
        sets = _parse_bfm_sets(text)
        return {timeframe: list(sets) for timeframe in timeframes}
    for chunk in text.split(";"):
        piece = chunk.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Invalid BFM timeframe set {piece!r}; expected timeframe=left:right,...")
        timeframe_raw, sets_raw = piece.split("=", 1)
        timeframe = normalize_timeframe(timeframe_raw.strip())
        if timeframe in out:
            out[timeframe] = _parse_bfm_sets(sets_raw)
    return out


def parse_bfm_feature_groups(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        raw = "line,channel"
    groups: list[str] = []
    for chunk in str(raw).split(","):
        group = chunk.strip().lower()
        if not group:
            continue
        if group == "all":
            group = "line,channel"
            for item in group.split(","):
                if item not in groups:
                    groups.append(item)
            continue
        if group not in {"line", "channel"}:
            raise ValueError("BFM feature groups must be line, channel, or all.")
        if group not in groups:
            groups.append(group)
    return groups or ["line", "channel"]


def bfm_feature_columns_for_groups(raw: str | None) -> list[str]:
    columns: list[str] = []
    groups = parse_bfm_feature_groups(raw)
    if "line" in groups:
        columns.extend(BFM_LINE_FEATURE_COLUMNS)
    if "channel" in groups:
        columns.extend(BFM_CHANNEL_FEATURE_COLUMNS)
    return columns


def prepare_bfm_feature_bars(exec_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    bars = resample_ohlc(exec_df, timeframe)
    bars = add_atr(bars)
    bars["body_high"] = bars[["open", "close"]].max(axis=1)
    bars["body_low"] = bars[["open", "close"]].min(axis=1)
    bars["bar_index"] = range(len(bars))
    return bars.reset_index(drop=True)


def _timestamp_ns(values: pd.Series) -> np.ndarray:
    return pd.to_datetime(values, utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")


def build_bfm_feature_projection(
    exec_df: pd.DataFrame,
    *,
    timeframes: list[str],
    tf_sets: dict[str, list[tuple[int, int]]],
    invalidation: str,
    max_extension_bars: int,
) -> BfmFeatureProjection:
    from scripts.plot_zone_channel_history import build_bfm_magic_lines

    exec_times_ns = _timestamp_ns(exec_df["close_time"])
    records: list[BfmLineRecord] = []
    if len(exec_times_ns) == 0:
        return BfmFeatureProjection(tuple(timeframes), tuple())

    for timeframe in timeframes:
        bars = prepare_bfm_feature_bars(exec_df, timeframe)
        if len(bars) < 2:
            continue
        source_times_ns = _timestamp_ns(bars["close_time"])
        source_indices = np.arange(len(bars), dtype=float)
        lines, _ = build_bfm_magic_lines(
            bars,
            tf_sets.get(timeframe) or tf_sets.get(normalize_timeframe(timeframe)) or _parse_bfm_sets("300:200"),
            invalidation=invalidation,
            max_extension_bars=max_extension_bars,
        )
        for line in lines:
            if line.line_end_index < 0 or line.line_end_index >= len(source_times_ns):
                continue
            active_start_ns = int(pd.Timestamp(line.end_pivot.confirm_time).tz_convert("UTC").value)
            active_end_ns = int(source_times_ns[line.line_end_index])
            if active_end_ns < active_start_ns:
                continue
            lo = int(np.searchsorted(exec_times_ns, active_start_ns, side="left"))
            hi = int(np.searchsorted(exec_times_ns, active_end_ns, side="right"))
            if lo >= hi:
                continue
            start_x = float(np.interp(float(exec_times_ns[lo]), source_times_ns.astype(float), source_indices))
            start_y = float(line.slope * start_x + line.intercept)
            end_exec_index = hi - 1
            if end_exec_index > lo:
                end_x = float(np.interp(float(exec_times_ns[end_exec_index]), source_times_ns.astype(float), source_indices))
                end_y = float(line.slope * end_x + line.intercept)
                exec_slope = (end_y - start_y) / float(end_exec_index - lo)
            else:
                exec_slope = 0.0
            records.append(
                BfmLineRecord(
                    side=str(line.side),
                    timeframe=timeframe,
                    tf_minutes=INTERVAL_MS[timeframe] / 60_000.0,
                    set_number=int(line.set_number),
                    start_exec_index=lo,
                    end_exec_index=end_exec_index,
                    exec_slope=float(exec_slope),
                    exec_intercept=float(start_y - exec_slope * lo),
                )
            )

    return BfmFeatureProjection(tuple(timeframes), tuple(records))


def _missing_bfm_channel_values() -> dict[str, float]:
    out: dict[str, float] = {}
    for timeframe in BFM_CHANNEL_FEATURE_TIMEFRAMES:
        prefix = f"bfm_{timeframe}_"
        out.update({
            f"{prefix}channel_available": 0.0,
            f"{prefix}channel_pos": math.nan,
            f"{prefix}channel_pos_clipped": math.nan,
            f"{prefix}channel_above": 0.0,
            f"{prefix}channel_below": 0.0,
            f"{prefix}channel_width_atr": math.nan,
            f"{prefix}upper_slope_atr": math.nan,
            f"{prefix}lower_slope_atr": math.nan,
            f"{prefix}width_slope_atr": math.nan,
            f"{prefix}trend_slope_atr": math.nan,
            f"{prefix}widening": 0.0,
            f"{prefix}closing": 0.0,
            f"{prefix}parallel": 0.0,
            f"{prefix}trend_up": 0.0,
            f"{prefix}trend_down": 0.0,
            f"{prefix}trend_aligned": 0.0,
        })
    return out


def _bfm_channel_values(
    projection: BfmFeatureProjection | None,
    *,
    direction: str,
    index: int,
    atr: float,
    close: float,
) -> dict[str, float]:
    out = _missing_bfm_channel_values()
    if projection is None or atr <= 0 or not math.isfinite(atr) or not math.isfinite(close):
        return out

    sign = 1.0 if direction == "long" else -1.0
    parallel_threshold = 0.001
    for timeframe in BFM_CHANNEL_FEATURE_TIMEFRAMES:
        supports: list[tuple[BfmLineRecord, float]] = []
        resistances: list[tuple[BfmLineRecord, float]] = []
        for record in projection.records:
            if record.timeframe != timeframe or index < record.start_exec_index or index > record.end_exec_index:
                continue
            value = record.evaluate(index)
            if not math.isfinite(value):
                continue
            if record.side == "support":
                supports.append((record, value))
            elif record.side == "resistance":
                resistances.append((record, value))

        best: tuple[tuple[float, float], BfmLineRecord, float, BfmLineRecord, float, float] | None = None
        for lower_record, lower_value in supports:
            for upper_record, upper_value in resistances:
                width = upper_value - lower_value
                if width <= 0 or not math.isfinite(width):
                    continue
                position = (close - lower_value) / width
                if 0.0 <= position <= 1.0:
                    outside = 0.0
                else:
                    outside = min(abs(position), abs(position - 1.0))
                score = (outside, width)
                if best is None or score < best[0]:
                    best = (score, lower_record, lower_value, upper_record, upper_value, position)
        if best is None:
            continue

        _, lower_record, lower_value, upper_record, upper_value, position = best
        upper_slope_atr = upper_record.exec_slope / atr
        lower_slope_atr = lower_record.exec_slope / atr
        width_slope_atr = upper_slope_atr - lower_slope_atr
        trend_slope_atr = (upper_slope_atr + lower_slope_atr) / 2.0
        trend_dir = 1.0 if trend_slope_atr > parallel_threshold else -1.0 if trend_slope_atr < -parallel_threshold else 0.0
        prefix = f"bfm_{timeframe}_"
        out.update({
            f"{prefix}channel_available": 1.0,
            f"{prefix}channel_pos": float(position),
            f"{prefix}channel_pos_clipped": float(min(1.0, max(0.0, position))),
            f"{prefix}channel_above": 1.0 if position > 1.0 else 0.0,
            f"{prefix}channel_below": 1.0 if position < 0.0 else 0.0,
            f"{prefix}channel_width_atr": float((upper_value - lower_value) / atr),
            f"{prefix}upper_slope_atr": float(upper_slope_atr),
            f"{prefix}lower_slope_atr": float(lower_slope_atr),
            f"{prefix}width_slope_atr": float(width_slope_atr),
            f"{prefix}trend_slope_atr": float(trend_slope_atr),
            f"{prefix}widening": 1.0 if width_slope_atr > parallel_threshold else 0.0,
            f"{prefix}closing": 1.0 if width_slope_atr < -parallel_threshold else 0.0,
            f"{prefix}parallel": 1.0 if abs(width_slope_atr) <= parallel_threshold else 0.0,
            f"{prefix}trend_up": 1.0 if trend_dir > 0.0 else 0.0,
            f"{prefix}trend_down": 1.0 if trend_dir < 0.0 else 0.0,
            f"{prefix}trend_aligned": float(sign * trend_dir),
        })

    return out


def bfm_zone_feature_values(
    *,
    projection: BfmFeatureProjection | None,
    direction: str,
    zone: dict,
    index: int,
    atr: float,
    close: float,
    high: float,
    low: float,
) -> dict[str, float]:
    missing = {
        "bfm_same_entry_gap_atr": 999.0,
        "bfm_same_mid_gap_atr": 999.0,
        "bfm_same_wick_gap_atr": 999.0,
        "bfm_same_signed_entry_gap_atr": 999.0,
        "bfm_same_inside_zone": 0.0,
        "bfm_same_near_0_5atr": 0.0,
        "bfm_same_near_1_0atr": 0.0,
        "bfm_same_tf_minutes": -1.0,
        "bfm_same_set": -1.0,
        "bfm_opp_entry_dist_atr": 999.0,
        "bfm_opp_close_dist_atr": 999.0,
        "bfm_opp_tf_minutes": -1.0,
        "bfm_opp_set": -1.0,
    }
    missing.update(_missing_bfm_channel_values())
    if projection is None or atr <= 0 or not math.isfinite(atr):
        return missing

    zone_top = float(zone["top"])
    zone_bottom = float(zone["bottom"])
    zone_mid_value = (zone_top + zone_bottom) / 2.0
    if direction == "long":
        same_side = "support"
        opp_side = "resistance"
        entry_price = zone_top
        wick_price = low
    else:
        same_side = "resistance"
        opp_side = "support"
        entry_price = zone_bottom
        wick_price = high

    best_same: tuple[float, BfmLineRecord, float] | None = None
    best_opp: tuple[float, BfmLineRecord, float] | None = None
    for record in projection.records:
        if index < record.start_exec_index or index > record.end_exec_index:
            continue
        value = record.evaluate(index)
        if not math.isfinite(value):
            continue
        if record.side == same_side:
            score = min(abs(value - entry_price), abs(value - zone_mid_value), abs(value - wick_price))
            if best_same is None or score < best_same[0]:
                best_same = (score, record, value)
        elif record.side == opp_side:
            if direction == "long":
                distance = value - entry_price
                close_distance = value - close
            else:
                distance = entry_price - value
                close_distance = close - value
            if distance > 0 and close_distance > 0 and (best_opp is None or distance < best_opp[0]):
                best_opp = (distance, record, value)

    out = dict(missing)
    if best_same is not None:
        _, record, value = best_same
        entry_gap_atr = abs(value - entry_price) / atr
        mid_gap_atr = abs(value - zone_mid_value) / atr
        wick_gap_atr = abs(value - wick_price) / atr
        if direction == "long":
            signed_gap_atr = (entry_price - value) / atr
        else:
            signed_gap_atr = (value - entry_price) / atr
        out.update({
            "bfm_same_entry_gap_atr": float(entry_gap_atr),
            "bfm_same_mid_gap_atr": float(mid_gap_atr),
            "bfm_same_wick_gap_atr": float(wick_gap_atr),
            "bfm_same_signed_entry_gap_atr": float(signed_gap_atr),
            "bfm_same_inside_zone": 1.0 if zone_bottom <= value <= zone_top else 0.0,
            "bfm_same_near_0_5atr": 1.0 if entry_gap_atr <= 0.5 else 0.0,
            "bfm_same_near_1_0atr": 1.0 if entry_gap_atr <= 1.0 else 0.0,
            "bfm_same_tf_minutes": float(record.tf_minutes),
            "bfm_same_set": float(record.set_number),
        })
    if best_opp is not None:
        distance, record, value = best_opp
        if direction == "long":
            close_dist = value - close
        else:
            close_dist = close - value
        out.update({
            "bfm_opp_entry_dist_atr": float(distance / atr),
            "bfm_opp_close_dist_atr": float(close_dist / atr) if close_dist > 0 else 0.0,
            "bfm_opp_tf_minutes": float(record.tf_minutes),
            "bfm_opp_set": float(record.set_number),
        })
    out.update(_bfm_channel_values(projection, direction=direction, index=index, atr=atr, close=close))
    return out


def label_zone_hold_outcome(
    df: pd.DataFrame,
    start_idx: int,
    direction: str,
    entry_price: float,
    fail_price: float,
    target_rr: float,
    horizon_bars: int,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
    closes: list[float] | None = None,
) -> dict | None:
    risk = abs(entry_price - fail_price)
    if risk <= 0:
        return None

    end_idx = min(len(df), start_idx + horizon_bars)
    if start_idx >= end_idx:
        return None

    sign = 1.0 if direction == "long" else -1.0
    target_price = entry_price + sign * risk * target_rr
    highs = highs if highs is not None else df["high"].to_list()
    lows = lows if lows is not None else df["low"].to_list()
    opens = opens if opens is not None else df["open"].to_list()
    closes = closes if closes is not None else df["close"].to_list()
    last_close_r = 0.0

    for j in range(start_idx, end_idx):
        if direction == "long":
            target_hit = highs[j] >= target_price
            fail_hit = lows[j] <= fail_price
            if target_hit and fail_hit:
                return {
                    "hold_label": 1 if tradingview_high_before_low(opens[j], highs[j], lows[j]) else 0,
                    "bars_to_outcome": j - start_idx + 1,
                }
            if target_hit:
                return {"hold_label": 1, "bars_to_outcome": j - start_idx + 1}
            if fail_hit:
                return {"hold_label": 0, "bars_to_outcome": j - start_idx + 1}
            last_close_r = (closes[j] - entry_price) / risk
        else:
            target_hit = lows[j] <= target_price
            fail_hit = highs[j] >= fail_price
            if target_hit and fail_hit:
                return {
                    "hold_label": 1 if not tradingview_high_before_low(opens[j], highs[j], lows[j]) else 0,
                    "bars_to_outcome": j - start_idx + 1,
                }
            if target_hit:
                return {"hold_label": 1, "bars_to_outcome": j - start_idx + 1}
            if fail_hit:
                return {"hold_label": 0, "bars_to_outcome": j - start_idx + 1}
            last_close_r = (entry_price - closes[j]) / risk

    return {
        "hold_label": 1 if last_close_r > 0 else 0,
        "bars_to_outcome": end_idx - start_idx,
    }


def zone_hold_feature_row(
    exec_df: pd.DataFrame,
    cfg: Config,
    direction: str,
    zone: dict,
    index: int,
    rank: int,
    active_same: int,
    active_opp: int,
    same_zones: list[dict],
    opp_zones: list[dict],
    current_bias_4h: int,
    current_bias_1d: int,
    current_htf_sma50_bias: int,
    day_context: dict[pd.Timestamp, dict],
    last20_known_outcomes: list[int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    atrs: list[float],
    vol_sma20: list[float],
    times: list[pd.Timestamp],
    bfm_projection: BfmFeatureProjection | None = None,
) -> dict | None:
    width = float(zone["width"])
    if width <= 0 or atrs[index] <= 0 or closes[index] <= 0:
        return None

    sign = 1.0 if direction == "long" else -1.0
    now = pd.Timestamp(times[index])
    exec_tf = normalize_timeframe(cfg.exec_tf)
    age_bars = max(0.0, (now - pd.Timestamp(zone["time"])).total_seconds() / (INTERVAL_MS[exec_tf] / 1000.0))
    age_pct = min(100.0, 100.0 * age_bars / max(1.0, float(cfg.zone_hold_mbq_ob_lookback_bars)))
    touch_count = float(zone.get("touch_count", 0))
    overhit = 20.0 if touch_count > 2 else 10.0 if touch_count > 1 else 0.0
    health = max(0.0, 100.0 - age_pct - overhit)
    confluence_total, confluence_same = confluence_counts(
        zone,
        same_zones,
        opp_zones,
        cfg.zone_hold_mbq_confluence_atr * atrs[index],
    )

    if direction == "long":
        entry_price = float(zone["top"])
        penetration_frac = (float(zone["top"]) - lows[index]) / width
        close_distance_pct = (closes[index] - float(zone["top"])) / closes[index] * 100.0
        reclaim_pos = (closes[index] - lows[index]) / (highs[index] - lows[index]) if highs[index] > lows[index] else 0.0
        same_bar_reaction_atr = max(0.0, (highs[index] - entry_price) / atrs[index])
        same_bar_close_reaction_atr = (closes[index] - entry_price) / atrs[index]
        same_bar_adverse_atr = max(0.0, (entry_price - lows[index]) / atrs[index])
    else:
        entry_price = float(zone["bottom"])
        penetration_frac = (highs[index] - float(zone["bottom"])) / width
        close_distance_pct = (float(zone["bottom"]) - closes[index]) / closes[index] * 100.0
        reclaim_pos = (highs[index] - closes[index]) / (highs[index] - lows[index]) if highs[index] > lows[index] else 0.0
        same_bar_reaction_atr = max(0.0, (entry_price - lows[index]) / atrs[index])
        same_bar_close_reaction_atr = (entry_price - closes[index]) / atrs[index]
        same_bar_adverse_atr = max(0.0, (highs[index] - entry_price) / atrs[index])

    ctx = current_day_context(day_context, now)
    hour = now.hour + now.minute / 60.0
    dow = now.dayofweek
    vol_mult = volumes[index] / vol_sma20[index] if vol_sma20[index] > 0 else math.nan

    def signed_feature(column: str) -> float:
        value = exec_df.iloc[index][column]
        return sign * float(value) if pd.notna(value) else math.nan

    row = {
        "direction_long": 1.0 if direction == "long" else 0.0,
        "zone_age_hours": (now - pd.Timestamp(zone["time"])).total_seconds() / 3600.0,
        "zone_width_pct": width / closes[index] * 100.0,
        "zone_width_atr": width / atrs[index],
        "penetration_frac": penetration_frac,
        "close_distance_pct": close_distance_pct,
        "reclaim_pos": reclaim_pos,
        "sweep_range_atr": (highs[index] - lows[index]) / atrs[index],
        "vol_mult": vol_mult,
        "ret_1h_dir": signed_feature("ret_1h"),
        "ret_4h_dir": signed_feature("ret_4h"),
        "ret_24h_dir": signed_feature("ret_24h"),
        "range_1h_pct": float(exec_df.iloc[index]["range_1h_pct"]) if pd.notna(exec_df.iloc[index]["range_1h_pct"]) else math.nan,
        "range_4h_pct": float(exec_df.iloc[index]["range_4h_pct"]) if pd.notna(exec_df.iloc[index]["range_4h_pct"]) else math.nan,
        "bias_4h_aligned": sign * current_bias_4h,
        "bias_1d_aligned": sign * current_bias_1d,
        "first4_ret_dir": sign * float(ctx["first4_ret"]) if pd.notna(ctx["first4_ret"]) else math.nan,
        "first4_range_pos": float(ctx["first4_range_pos"]) if pd.notna(ctx["first4_range_pos"]) else math.nan,
        "prev_day_ret_dir": sign * float(ctx["prev_day_ret"]) if pd.notna(ctx["prev_day_ret"]) else math.nan,
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "dow_sin": math.sin(2.0 * math.pi * dow / 7.0),
        "dow_cos": math.cos(2.0 * math.pi * dow / 7.0),
        "active_same_dir_zones": float(active_same),
        "active_opp_zones": float(active_opp),
        "zone_rank": float(rank),
        "prior_zone_touches": touch_count,
        "same_bar_reaction_atr": same_bar_reaction_atr,
        "same_bar_close_reaction_atr": same_bar_close_reaction_atr,
        "same_bar_adverse_atr": same_bar_adverse_atr,
        "reclaim_body_atr": sign * (closes[index] - opens[index]) / atrs[index],
        "mbq_zone_health": health,
        "confluence_count_0_5atr": float(confluence_total),
        "confluence_same_count_0_5atr": float(confluence_same),
        "htf_sma50_aligned": sign * current_htf_sma50_bias,
        "last20_known_hold_rate": float(sum(last20_known_outcomes) / len(last20_known_outcomes)) if last20_known_outcomes else math.nan,
    }
    if bfm_projection is not None:
        row.update(
            bfm_zone_feature_values(
                projection=bfm_projection,
                direction=direction,
                zone=zone,
                index=index,
                atr=atrs[index],
                close=closes[index],
                high=highs[index],
                low=lows[index],
            )
        )
    return row


def bias_filters_enabled(cfg: Config) -> bool:
    return any([
        cfg.htf_bias_mode != "none",
        cfg.use_first4_return_bias,
        cfg.use_first4_range_bias,
        cfg.use_prev_day_reversion_bias,
        cfg.use_thursday_bearish_bias,
    ])


def compute_bias_scores(
    cfg: Config,
    now: pd.Timestamp,
    day_context: dict[pd.Timestamp, dict],
    htf_bias_4h: int,
    htf_bias_1d: int,
) -> tuple[int, int]:
    bull_score = 0
    bear_score = 0

    if cfg.htf_bias_mode in {"4h_ema", "4h_1d_ema"}:
        if htf_bias_4h > 0:
            bull_score += 1
        elif htf_bias_4h < 0:
            bear_score += 1
    if cfg.htf_bias_mode == "4h_1d_ema":
        if htf_bias_1d > 0:
            bull_score += 1
        elif htf_bias_1d < 0:
            bear_score += 1

    ctx = day_context.get(now.floor("1D"))
    if ctx is None:
        return bull_score, bear_score

    if now.hour >= 4 and cfg.use_first4_return_bias and not math.isnan(ctx["first4_ret"]):
        if ctx["first4_ret"] >= cfg.first4_return_threshold:
            bull_score += 1
        elif ctx["first4_ret"] <= -cfg.first4_return_threshold:
            bear_score += 1

    if now.hour >= 4 and cfg.use_first4_range_bias and not math.isnan(ctx["first4_range_pos"]):
        if ctx["first4_range_pos"] >= cfg.first4_range_upper:
            bull_score += 1
        elif ctx["first4_range_pos"] <= cfg.first4_range_lower:
            bear_score += 1

    if cfg.use_prev_day_reversion_bias and not math.isnan(ctx["prev_day_ret"]):
        if ctx["prev_day_ret"] >= cfg.prev_day_reversion_threshold:
            bear_score += 1
        elif ctx["prev_day_ret"] <= -cfg.prev_day_reversion_threshold:
            bull_score += 1

    if cfg.use_thursday_bearish_bias and ctx["weekday"] == "Thursday":
        bear_score += 1

    return bull_score, bear_score


def direction_allowed(cfg: Config, direction: str, bull_score: int, bear_score: int) -> bool:
    if not bias_filters_enabled(cfg):
        return True
    if direction == "long":
        return bull_score - bear_score >= cfg.min_bias_score
    return bear_score - bull_score >= cfg.min_bias_score


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
        }

    rs = [t.r_multiple for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(trades),
        "win_rate": round(100 * len(wins) / len(trades), 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf"),
        "net_r": round(sum(rs), 3),
        "avg_r": round(sum(rs) / len(rs), 3),
        "avg_win_r": round(sum(wins) / len(wins), 3) if wins else 0.0,
        "avg_loss_r": round(sum(losses) / len(losses), 3) if losses else 0.0,
    }


def find_last_opposite_candle(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    start_idx: int,
    end_idx: int,
    direction: str,
    use_body: bool,
) -> tuple[float, float] | None:
    if end_idx < start_idx:
        return None
    for idx in range(end_idx, start_idx - 1, -1):
        if direction == "long" and closes[idx] < opens[idx]:
            return (opens[idx], closes[idx]) if use_body else (highs[idx], lows[idx])
        if direction == "short" and closes[idx] > opens[idx]:
            return (closes[idx], opens[idx]) if use_body else (highs[idx], lows[idx])
    return None


def long_pre_entry_invalid(
    mode: str,
    retry_armed: bool,
    invalidate_on_close: bool,
    close_val: float,
    low_val: float,
    ob_bottom_val: float,
    zone_bottom_val: float,
    stop_val: float,
) -> bool:
    effective_mode = "Stop Sweep" if mode == "Blocked Setup Retry" and retry_armed else ("OB Or Stop Wick" if mode == "Blocked Setup Retry" else mode)
    if effective_mode == "OB Or Stop Wick":
        return (close_val < ob_bottom_val if invalidate_on_close else low_val < ob_bottom_val) or low_val <= stop_val
    if effective_mode == "OB Or Stop Close":
        return (close_val < ob_bottom_val if invalidate_on_close else low_val < ob_bottom_val) or close_val <= stop_val
    if effective_mode == "Stop Sweep":
        return low_val <= stop_val
    if effective_mode == "Zone Boundary":
        return close_val < zone_bottom_val if invalidate_on_close else low_val < zone_bottom_val
    raise ValueError(f"Unknown pre-entry invalidation mode: {mode}")


def short_pre_entry_invalid(
    mode: str,
    retry_armed: bool,
    invalidate_on_close: bool,
    close_val: float,
    high_val: float,
    ob_top_val: float,
    zone_top_val: float,
    stop_val: float,
) -> bool:
    effective_mode = "Stop Sweep" if mode == "Blocked Setup Retry" and retry_armed else ("OB Or Stop Wick" if mode == "Blocked Setup Retry" else mode)
    if effective_mode == "OB Or Stop Wick":
        return (close_val > ob_top_val if invalidate_on_close else high_val > ob_top_val) or high_val >= stop_val
    if effective_mode == "OB Or Stop Close":
        return (close_val > ob_top_val if invalidate_on_close else high_val > ob_top_val) or close_val >= stop_val
    if effective_mode == "Stop Sweep":
        return high_val >= stop_val
    if effective_mode == "Zone Boundary":
        return close_val > zone_top_val if invalidate_on_close else high_val > zone_top_val
    raise ValueError(f"Unknown pre-entry invalidation mode: {mode}")


def tradingview_high_before_low(open_val: float, high_val: float, low_val: float) -> bool:
    return abs(open_val - high_val) < abs(open_val - low_val)


def limit_entry_fill_price(direction: str, limit_price: float, open_val: float, high_val: float, low_val: float, mintick: float) -> float | None:
    if direction == "long":
        if open_val <= limit_price:
            return round_to_mintick(open_val, mintick)
        if low_val <= limit_price <= high_val:
            return round_to_mintick(limit_price, mintick)
    else:
        if open_val >= limit_price:
            return round_to_mintick(open_val, mintick)
        if low_val <= limit_price <= high_val:
            return round_to_mintick(limit_price, mintick)
    return None


def market_fill_price(direction: str, open_val: float, cfg: Config) -> float:
    slip = cfg.slippage_ticks * cfg.mintick
    price = open_val + slip if direction == "long" else open_val - slip
    return round_to_mintick(price, cfg.mintick)


def market_exit_price(direction: str, open_val: float, cfg: Config) -> float:
    slip = cfg.slippage_ticks * cfg.mintick
    price = open_val - slip if direction == "long" else open_val + slip
    return round_to_mintick(price, cfg.mintick)


def price_exit_for_bar(position: dict, open_val: float, high_val: float, low_val: float, cfg: Config) -> tuple[float, str] | None:
    direction = position["direction"]
    stop = position["stop"]
    target = position["target"]
    slip = cfg.slippage_ticks * cfg.mintick
    high_first = tradingview_high_before_low(open_val, high_val, low_val)

    if direction == "long":
        if open_val <= stop:
            return round_to_mintick(open_val - slip, cfg.mintick), "stop"
        if open_val >= target:
            return round_to_mintick(open_val, cfg.mintick), "target"
        stop_hit = low_val <= stop
        target_hit = high_val >= target
        if stop_hit and target_hit:
            if high_first:
                return round_to_mintick(target, cfg.mintick), "target_same_bar"
            return round_to_mintick(stop - slip, cfg.mintick), "stop_same_bar"
        if stop_hit:
            return round_to_mintick(stop - slip, cfg.mintick), "stop"
        if target_hit:
            return round_to_mintick(target, cfg.mintick), "target"
    else:
        if open_val >= stop:
            return round_to_mintick(open_val + slip, cfg.mintick), "stop"
        if open_val <= target:
            return round_to_mintick(open_val, cfg.mintick), "target"
        stop_hit = high_val >= stop
        target_hit = low_val <= target
        if stop_hit and target_hit:
            if high_first:
                return round_to_mintick(stop + slip, cfg.mintick), "stop_same_bar"
            return round_to_mintick(target, cfg.mintick), "target_same_bar"
        if stop_hit:
            return round_to_mintick(stop + slip, cfg.mintick), "stop"
        if target_hit:
            return round_to_mintick(target, cfg.mintick), "target"

    return None


def side_metrics(trades: list[Trade], direction: str) -> dict:
    side = [t for t in trades if t.direction == direction]
    summary = summarize(side)
    summary["count"] = len(side)
    return summary


def build_position_from_entry(order: dict, entry_index: int, entry_time: pd.Timestamp, entry_price: float, cfg: Config) -> dict | None:
    direction = order["direction"]
    if direction == "long":
        risk = entry_price - order["stop"]
        target = entry_price + risk * cfg.target_rr
    else:
        risk = order["stop"] - entry_price
        target = entry_price - risk * cfg.target_rr

    if risk <= cfg.mintick:
        return None
    risk_pct = risk / entry_price * 100
    if risk_pct < cfg.min_entry_risk_pct or risk_pct > cfg.max_entry_risk_pct:
        return None

    return {
        "direction": direction,
        "entry_index": entry_index,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "stop": order["stop"],
        "target": target,
        "risk": risk,
        "signal_index": order["signal_index"],
        "signal_time": order["signal_time"],
        "setup": order["setup"],
        "exit_orders_active_from": entry_index + 1,
    }


def trade_from_position(position: dict, exit_index: int, exit_time: pd.Timestamp, exit_price: float, exit_reason: str, cfg: Config) -> Trade:
    direction = position["direction"]
    if direction == "long":
        r_multiple = (exit_price - position["entry_price"]) / position["risk"]
    else:
        r_multiple = (position["entry_price"] - exit_price) / position["risk"]

    return Trade(
        direction=direction,
        entry_time=position["entry_time"],
        exit_time=exit_time,
        entry_price=position["entry_price"],
        exit_price=exit_price,
        stop_price=position["stop"],
        target_price=position["target"],
        r_multiple=r_multiple,
        exit_reason=exit_reason,
        hold_bars=exit_index - position["entry_index"] + 1,
        exec_tf=cfg.exec_tf,
        structure_tf=cfg.structure_tf,
        entry_mode=cfg.entry_mode,
        zone_tf=position["setup"]["zone_tf"],
        zone_top=position["setup"]["zone_top"],
        zone_bottom=position["setup"]["zone_bottom"],
        sweep_time=position["setup"]["sweep_time"],
        choch_time=position["setup"]["choch_time"],
        signal_time=position["signal_time"],
        ob_top=position["setup"]["ob_top"],
        ob_bottom=position["setup"]["ob_bottom"],
        sweep_index=position["setup"]["sweep_idx"],
        choch_index=position["setup"]["choch_exec_idx"],
        signal_index=position["signal_index"],
        entry_index=position["entry_index"],
        exit_index=exit_index,
        zone_hold_prob=position["setup"].get("zone_hold_prob", math.nan),
        zone_source=position["setup"].get("zone_source", "ob_break"),
        liquidity_level=float(position["setup"].get("liquidity_level", math.nan)),
        liquidity_pivot_time=position["setup"].get("liquidity_pivot_time"),
        liquidity_confirm_time=position["setup"].get("liquidity_confirm_time"),
        liquidity_sfp_strict=bool(position["setup"].get("liquidity_sfp_strict", False)),
    )


def run_backtest(exec_df: pd.DataFrame, cfg: Config, return_state: bool = False) -> list[Trade] | tuple[list[Trade], dict]:
    exec_tf = normalize_timeframe(cfg.exec_tf)
    structure_tf = normalize_timeframe(cfg.structure_tf)
    tf1 = normalize_timeframe(cfg.tf1)
    tf2 = normalize_timeframe(cfg.tf2)
    exec_df = add_atr(exec_df)
    exec_df["vol_sma20"] = exec_df["volume"].rolling(20).mean()
    exec_df["ret_1h"] = exec_df["close"].pct_change(12) * 100.0
    exec_df["ret_4h"] = exec_df["close"].pct_change(48) * 100.0
    exec_df["ret_24h"] = exec_df["close"].pct_change(288) * 100.0
    exec_df["range_1h_pct"] = (exec_df["high"].rolling(12).max() - exec_df["low"].rolling(12).min()) / exec_df["close"] * 100.0
    exec_df["range_4h_pct"] = (exec_df["high"].rolling(48).max() - exec_df["low"].rolling(48).min()) / exec_df["close"] * 100.0
    zone_hold_model = load_zone_hold_model(cfg.zone_hold_model_path) if cfg.zone_hold_model_path and cfg.zone_hold_min_prob > 0 else None
    zone_hold_filter_tf = normalize_timeframe(cfg.zone_hold_filter_tf) if cfg.zone_hold_filter_tf else ""
    bfm_projection: BfmFeatureProjection | None = None
    if zone_hold_model_requires_bfm(zone_hold_model):
        bfm_config = dict(zone_hold_model.get("bfm_feature_config") or {}) if zone_hold_model is not None else {}
        bfm_timeframes = parse_bfm_feature_timeframes(bfm_config.get("timeframes", cfg.zone_hold_bfm_timeframes))
        bfm_tf_sets = parse_bfm_feature_tf_sets(bfm_config.get("tf_sets", cfg.zone_hold_bfm_tf_sets), bfm_timeframes)
        bfm_projection = build_bfm_feature_projection(
            exec_df,
            timeframes=bfm_timeframes,
            tf_sets=bfm_tf_sets,
            invalidation=str(bfm_config.get("invalidation", cfg.zone_hold_bfm_invalidation)),
            max_extension_bars=int(bfm_config.get("max_extension_bars", cfg.zone_hold_bfm_max_extension_bars)),
        )
    tf1_high, tf1_low = build_htf_zone_events(
        exec_df,
        tf1,
        cfg.htf_left,
        cfg.htf_right,
        cfg.htf_zone_width_atr,
        cfg.htf_ob_search_bars,
        cfg.ob_use_body,
    )
    tf2_high, tf2_low = build_htf_zone_events(
        exec_df,
        tf2,
        cfg.htf_left,
        cfg.htf_right,
        cfg.htf_zone_width_atr,
        cfg.htf_ob_search_bars,
        cfg.ob_use_body,
    )
    sfp_high: list[dict] = []
    sfp_low: list[dict] = []
    if cfg.use_sfp_liquidity_zones:
        sfp_high, sfp_low = build_sfp_liquidity_zone_events(
            exec_df,
            timeframes=parse_timeframe_list(cfg.sfp_timeframes),
            left=cfg.sfp_left,
            right=cfg.sfp_right,
            width_atr=cfg.sfp_level_width_atr,
        )
    structure_events = build_structure_choch_events(exec_df, structure_tf, cfg.structure_left, cfg.structure_right)
    structure_event_times = [event["time"] for event in structure_events]
    bias_4h_events = build_htf_bias_events(exec_df, "4h", cfg.htf_bias_len)
    bias_1d_events = build_htf_bias_events(exec_df, "1d", cfg.htf_bias_len)
    zone_hold_sma_tf = zone_hold_filter_tf or tf1
    htf_sma50_events = build_htf_sma_bias_events(exec_df, zone_hold_sma_tf, 50) if zone_hold_model is not None else []
    day_context = build_daily_context(exec_df)

    tf1_hi_ptr = tf1_lo_ptr = tf2_hi_ptr = tf2_lo_ptr = 0
    sfp_hi_ptr = sfp_lo_ptr = 0
    bias_4h_ptr = bias_1d_ptr = 0
    htf_sma50_ptr = 0
    tf1_res_zones: list[dict] = []
    tf1_sup_zones: list[dict] = []
    tf2_res_zones: list[dict] = []
    tf2_sup_zones: list[dict] = []
    sfp_res_zones: list[dict] = []
    sfp_sup_zones: list[dict] = []
    current_bias_4h = 0
    current_bias_1d = 0
    current_htf_sma50_bias = 0
    pending_known_outcomes: list[tuple[int, int]] = []
    last20_known_outcomes: list[int] = []

    opens = exec_df["open"].to_list()
    highs = exec_df["high"].to_list()
    lows = exec_df["low"].to_list()
    closes = exec_df["close"].to_list()
    volumes = exec_df["volume"].to_list()
    atrs = exec_df["atr"].bfill().ffill().to_list()
    vol_sma20 = exec_df["vol_sma20"].bfill().ffill().to_list()
    times = exec_df["open_time"].to_list()
    close_times = exec_df["close_time"].to_list()

    long_setup = None
    short_setup = None
    pending_entry = None
    pending_close = None
    position = None
    latest_zone_hold_candidate = None
    trades: list[Trade] = []
    max_choch_wait_exec_bars = cfg.max_structure_bars_to_choch * max(1.0, INTERVAL_MS[structure_tf] / INTERVAL_MS[exec_tf])

    for i in range(len(exec_df)):
        now = times[i]
        visible_time = close_times[i]

        if pending_known_outcomes:
            still_pending: list[tuple[int, int]] = []
            for outcome_idx, label in pending_known_outcomes:
                if outcome_idx <= i:
                    last20_known_outcomes.append(label)
                    if len(last20_known_outcomes) > 20:
                        last20_known_outcomes.pop(0)
                else:
                    still_pending.append((outcome_idx, label))
            pending_known_outcomes = still_pending

        if pending_entry and pending_entry["submitted_index"] < i and position is None:
            if pending_entry["order_type"] == "market":
                entry_price = market_fill_price(pending_entry["direction"], opens[i], cfg)
            else:
                entry_price = limit_entry_fill_price(
                    pending_entry["direction"],
                    pending_entry["entry_price"],
                    opens[i],
                    highs[i],
                    lows[i],
                    cfg.mintick,
                )
            if entry_price is not None:
                new_position = build_position_from_entry(pending_entry, i, now, entry_price, cfg)
                if new_position is not None:
                    position = new_position
                    if position["direction"] == "long":
                        long_setup = None
                    else:
                        short_setup = None
                pending_entry = None

        if position is not None and pending_close and pending_close["submitted_index"] < i:
            exit_price = market_exit_price(position["direction"], opens[i], cfg)
            trades.append(trade_from_position(position, i, now, exit_price, pending_close["reason"], cfg))
            position = None
            pending_close = None

        if position is not None and i >= position["exit_orders_active_from"]:
            price_exit = price_exit_for_bar(position, opens[i], highs[i], lows[i], cfg)
            if price_exit is not None:
                exit_price, exit_reason = price_exit
                trades.append(trade_from_position(position, i, now, exit_price, exit_reason, cfg))
                position = None
                pending_close = None

        while tf1_hi_ptr < len(tf1_high) and tf1_high[tf1_hi_ptr]["time"] <= visible_time:
            tf1_res_zones.append(dict(tf1_high[tf1_hi_ptr], used=False, touch_count=0, tf=tf1, id=f"{tf1}-res-{tf1_hi_ptr}"))
            tf1_hi_ptr += 1
        while tf1_lo_ptr < len(tf1_low) and tf1_low[tf1_lo_ptr]["time"] <= visible_time:
            tf1_sup_zones.append(dict(tf1_low[tf1_lo_ptr], used=False, touch_count=0, tf=tf1, id=f"{tf1}-sup-{tf1_lo_ptr}"))
            tf1_lo_ptr += 1
        while tf2_hi_ptr < len(tf2_high) and tf2_high[tf2_hi_ptr]["time"] <= visible_time:
            tf2_res_zones.append(dict(tf2_high[tf2_hi_ptr], used=False, touch_count=0, tf=tf2, id=f"{tf2}-res-{tf2_hi_ptr}"))
            tf2_hi_ptr += 1
        while tf2_lo_ptr < len(tf2_low) and tf2_low[tf2_lo_ptr]["time"] <= visible_time:
            tf2_sup_zones.append(dict(tf2_low[tf2_lo_ptr], used=False, touch_count=0, tf=tf2, id=f"{tf2}-sup-{tf2_lo_ptr}"))
            tf2_lo_ptr += 1
        while sfp_hi_ptr < len(sfp_high) and sfp_high[sfp_hi_ptr]["time"] <= visible_time:
            zone = dict(sfp_high[sfp_hi_ptr], used=False, touch_count=0, id=f"sfp-res-{sfp_hi_ptr}")
            sfp_res_zones.append(zone)
            if cfg.sfp_max_active_levels > 0 and len(sfp_res_zones) > cfg.sfp_max_active_levels:
                sfp_res_zones = sfp_res_zones[-cfg.sfp_max_active_levels :]
            sfp_hi_ptr += 1
        while sfp_lo_ptr < len(sfp_low) and sfp_low[sfp_lo_ptr]["time"] <= visible_time:
            zone = dict(sfp_low[sfp_lo_ptr], used=False, touch_count=0, id=f"sfp-sup-{sfp_lo_ptr}")
            sfp_sup_zones.append(zone)
            if cfg.sfp_max_active_levels > 0 and len(sfp_sup_zones) > cfg.sfp_max_active_levels:
                sfp_sup_zones = sfp_sup_zones[-cfg.sfp_max_active_levels :]
            sfp_lo_ptr += 1
        while bias_4h_ptr < len(bias_4h_events) and bias_4h_events[bias_4h_ptr]["time"] <= visible_time:
            current_bias_4h = bias_4h_events[bias_4h_ptr]["bias"]
            bias_4h_ptr += 1
        while bias_1d_ptr < len(bias_1d_events) and bias_1d_events[bias_1d_ptr]["time"] <= visible_time:
            current_bias_1d = bias_1d_events[bias_1d_ptr]["bias"]
            bias_1d_ptr += 1
        while htf_sma50_ptr < len(htf_sma50_events) and htf_sma50_events[htf_sma50_ptr]["time"] <= visible_time:
            current_htf_sma50_bias = htf_sma50_events[htf_sma50_ptr]["bias"]
            htf_sma50_ptr += 1

        bull_zone = None
        bear_zone = None
        bull_score, bear_score = compute_bias_scores(cfg, now, day_context, current_bias_4h, current_bias_1d)
        in_dead_zone = cfg.block_dead_zone and cfg.dead_zone_start_hour <= now.hour <= cfg.dead_zone_end_hour
        tf1_sup_zones = [zone for zone in tf1_sup_zones if not zone["used"] and lows[i] >= zone["bottom"]]
        tf2_sup_zones = [zone for zone in tf2_sup_zones if not zone["used"] and lows[i] >= zone["bottom"]]
        tf1_res_zones = [zone for zone in tf1_res_zones if not zone["used"] and highs[i] <= zone["top"]]
        tf2_res_zones = [zone for zone in tf2_res_zones if not zone["used"] and highs[i] <= zone["top"]]
        sfp_sup_zones = [zone for zone in sfp_sup_zones if not zone["used"] and lows[i] >= zone["bottom"]]
        sfp_res_zones = [zone for zone in sfp_res_zones if not zone["used"] and highs[i] <= zone["top"]]

        def sweep_candidates(zones: list[dict]) -> list[dict]:
            candidates = [zone for zone in reversed(zones) if not zone["used"]]
            if cfg.max_zone_scan > 0:
                return candidates[: cfg.max_zone_scan]
            return candidates

        if cfg.prioritize_higher_tf:
            support_zones = (sweep_candidates(tf2_sup_zones) if cfg.use_tf2 else []) + (sweep_candidates(tf1_sup_zones) if cfg.use_tf1 else []) + sweep_candidates(sfp_sup_zones)
            resistance_zones = (sweep_candidates(tf2_res_zones) if cfg.use_tf2 else []) + (sweep_candidates(tf1_res_zones) if cfg.use_tf1 else []) + sweep_candidates(sfp_res_zones)
        else:
            support_zones = (sweep_candidates(tf1_sup_zones) if cfg.use_tf1 else []) + (sweep_candidates(tf2_sup_zones) if cfg.use_tf2 else []) + sweep_candidates(sfp_sup_zones)
            resistance_zones = (sweep_candidates(tf1_res_zones) if cfg.use_tf1 else []) + (sweep_candidates(tf2_res_zones) if cfg.use_tf2 else []) + sweep_candidates(sfp_res_zones)

        active_support_zones = tf1_sup_zones + tf2_sup_zones + sfp_sup_zones
        active_resistance_zones = tf1_res_zones + tf2_res_zones + sfp_res_zones

        def sfp_filter_passes(zone: dict, direction: str, index: int) -> bool:
            if zone.get("source") != "sfp_pivot" or not cfg.sfp_strict:
                return True
            level = float(zone.get("level", zone["top"] if direction == "long" else zone["bottom"]))
            start = max(0, index - max(1, int(cfg.sfp_left)) + 1)
            eps = max(abs(level) * 1e-10, 1e-12)
            if direction == "long":
                open_ok = (not cfg.sfp_require_open_reclaim) or opens[index] > level
                return (
                    lows[index] < level
                    and closes[index] > level
                    and open_ok
                    and lows[index] <= min(lows[start:index + 1]) + eps
                    and min(closes[start:index + 1]) >= level - eps
                )
            open_ok = (not cfg.sfp_require_open_reclaim) or opens[index] < level
            return (
                highs[index] > level
                and closes[index] < level
                and open_ok
                and highs[index] >= max(highs[start:index + 1]) - eps
                and max(closes[start:index + 1]) <= level + eps
            )

        def zone_hold_filter_passes(
            direction: str,
            zone: dict,
            rank: int,
            active_same: int,
            active_opp: int,
            same_zones: list[dict],
            opp_zones: list[dict],
        ) -> tuple[bool, float, dict]:
            if zone_hold_model is None:
                return True, math.nan, {
                    "zone_hold_applied": False,
                    "zone_hold_reason": "no_model",
                    "zone_hold_threshold": cfg.zone_hold_min_prob,
                }
            if zone_hold_filter_tf and zone.get("tf") != zone_hold_filter_tf:
                accepted = not cfg.zone_hold_reject_unscored
                return accepted, math.nan, {
                    "zone_hold_applied": False,
                    "zone_hold_reason": "filter_tf_mismatch",
                    "zone_hold_threshold": cfg.zone_hold_min_prob,
                    "zone_hold_filter_tf": zone_hold_filter_tf,
                    "zone_hold_zone_tf": zone.get("tf"),
                }

            features = zone_hold_feature_row(
                exec_df,
                cfg,
                direction,
                zone,
                i,
                rank,
                active_same,
                active_opp,
                same_zones,
                opp_zones,
                current_bias_4h,
                current_bias_1d,
                current_htf_sma50_bias,
                day_context,
                last20_known_outcomes,
                opens,
                highs,
                lows,
                closes,
                volumes,
                atrs,
                vol_sma20,
                times,
                bfm_projection,
            )
            if features is None:
                accepted = not cfg.zone_hold_reject_unscored
                return accepted, math.nan, {
                    "zone_hold_applied": False,
                    "zone_hold_reason": "features_unavailable",
                    "zone_hold_threshold": cfg.zone_hold_min_prob,
                }
            prob = predict_zone_hold_probability(zone_hold_model, features)

            entry_price = float(zone["top"]) if direction == "long" else float(zone["bottom"])
            fail_price = float(zone["bottom"]) if direction == "long" else float(zone["top"])
            outcome = label_zone_hold_outcome(
                exec_df,
                i + 1,
                direction,
                entry_price,
                fail_price,
                cfg.zone_hold_label_rr,
                cfg.zone_hold_label_horizon_bars,
                highs,
                lows,
                opens,
                closes,
            )
            if outcome is not None:
                pending_known_outcomes.append((i + int(outcome["bars_to_outcome"]), int(outcome["hold_label"])))

            accepted = prob >= cfg.zone_hold_min_prob
            return accepted, prob, {
                "zone_hold_applied": True,
                "zone_hold_reason": "prob_above_threshold" if accepted else "prob_below_threshold",
                "zone_hold_threshold": cfg.zone_hold_min_prob,
            }

        for rank, zone in enumerate(support_zones):
            if not zone or zone["used"]:
                continue
            penetration_limit = zone["bottom"] - zone["width"] * cfg.zone_penetration_frac
            sweep_range = highs[i] - lows[i]
            reclaim_pos = (closes[i] - lows[i]) / sweep_range if sweep_range > 0 else 0.0
            vol_mult = volumes[i] / vol_sma20[i] if vol_sma20[i] > 0 else 0.0
            vol_ok = (
                vol_mult >= cfg.min_sweep_volume_mult
                and (cfg.max_sweep_volume_mult <= 0 or vol_mult <= cfg.max_sweep_volume_mult)
            )
            if (
                lows[i] <= zone["top"]
                and lows[i] >= penetration_limit
                and closes[i] > zone["top"]
                and reclaim_pos >= cfg.min_sweep_reclaim_pos
                and vol_ok
                and sfp_filter_passes(zone, "long", i)
            ):
                ok, prob, zone_hold_meta = zone_hold_filter_passes(
                    "long",
                    zone,
                    rank,
                    len(support_zones),
                    len(resistance_zones),
                    active_support_zones,
                    active_resistance_zones,
                )
                zone["zone_hold_prob"] = prob
                latest_zone_hold_candidate = {
                    "direction": "long",
                    "sweep_time": now,
                    "zone_tf": zone.get("tf"),
                    "zone_top": float(zone["top"]),
                    "zone_bottom": float(zone["bottom"]),
                    "zone_hold_prob": prob,
                    "zone_rank": rank,
                    "reclaim_pos": reclaim_pos,
                    "volume_mult": vol_mult,
                    **zone_hold_meta,
                }
                if ok:
                    bull_zone = zone
                elif cfg.allow_longs:
                    zone["used"] = True
                break
            if lows[i] <= zone["top"] and lows[i] >= penetration_limit:
                zone["touch_count"] = zone.get("touch_count", 0) + 1

        for rank, zone in enumerate(resistance_zones):
            if not zone or zone["used"]:
                continue
            penetration_limit = zone["top"] + zone["width"] * cfg.zone_penetration_frac
            sweep_range = highs[i] - lows[i]
            reclaim_pos = (highs[i] - closes[i]) / sweep_range if sweep_range > 0 else 0.0
            vol_mult = volumes[i] / vol_sma20[i] if vol_sma20[i] > 0 else 0.0
            vol_ok = (
                vol_mult >= cfg.min_sweep_volume_mult
                and (cfg.max_sweep_volume_mult <= 0 or vol_mult <= cfg.max_sweep_volume_mult)
            )
            if (
                highs[i] >= zone["bottom"]
                and highs[i] <= penetration_limit
                and closes[i] < zone["bottom"]
                and reclaim_pos >= cfg.min_sweep_reclaim_pos
                and vol_ok
                and sfp_filter_passes(zone, "short", i)
            ):
                ok, prob, zone_hold_meta = zone_hold_filter_passes(
                    "short",
                    zone,
                    rank,
                    len(resistance_zones),
                    len(support_zones),
                    active_resistance_zones,
                    active_support_zones,
                )
                zone["zone_hold_prob"] = prob
                latest_zone_hold_candidate = {
                    "direction": "short",
                    "sweep_time": now,
                    "zone_tf": zone.get("tf"),
                    "zone_top": float(zone["top"]),
                    "zone_bottom": float(zone["bottom"]),
                    "zone_hold_prob": prob,
                    "zone_rank": rank,
                    "reclaim_pos": reclaim_pos,
                    "volume_mult": vol_mult,
                    **zone_hold_meta,
                }
                if ok:
                    bear_zone = zone
                elif cfg.allow_shorts:
                    zone["used"] = True
                break
            if highs[i] >= zone["bottom"] and highs[i] <= penetration_limit:
                zone["touch_count"] = zone.get("touch_count", 0) + 1

        if bull_zone and cfg.allow_longs:
            bull_zone["used"] = True
            if pending_entry and pending_entry["direction"] == "long":
                pending_entry = None
            long_setup = {
                "zone_tf": bull_zone["tf"],
                "zone_top": bull_zone["top"],
                "zone_bottom": bull_zone["bottom"],
                "sweep_idx": i,
                "sweep_time": now,
                "sweep_extreme": lows[i],
                "event_search_idx": bisect_right(structure_event_times, now),
                "choch_found": False,
                "retry_armed": False,
                "choch_exec_idx": None,
                "choch_break_level": None,
                "ob_top": None,
                "ob_bottom": None,
                "limit_price": None,
                "planned_stop": None,
                "zone_hold_prob": bull_zone.get("zone_hold_prob", math.nan),
                "zone_source": bull_zone.get("source", "ob_break"),
                "liquidity_level": float(bull_zone.get("level", bull_zone["top"])),
                "liquidity_pivot_time": bull_zone.get("pivot_time"),
                "liquidity_confirm_time": bull_zone.get("confirm_time", bull_zone.get("time")),
                "liquidity_sfp_strict": bool(bull_zone.get("source") == "sfp_pivot"),
            }

        if bear_zone and cfg.allow_shorts:
            bear_zone["used"] = True
            if pending_entry and pending_entry["direction"] == "short":
                pending_entry = None
            short_setup = {
                "zone_tf": bear_zone["tf"],
                "zone_top": bear_zone["top"],
                "zone_bottom": bear_zone["bottom"],
                "sweep_idx": i,
                "sweep_time": now,
                "sweep_extreme": highs[i],
                "event_search_idx": bisect_right(structure_event_times, now),
                "choch_found": False,
                "retry_armed": False,
                "choch_exec_idx": None,
                "choch_break_level": None,
                "ob_top": None,
                "ob_bottom": None,
                "limit_price": None,
                "planned_stop": None,
                "zone_hold_prob": bear_zone.get("zone_hold_prob", math.nan),
                "zone_source": bear_zone.get("source", "ob_break"),
                "liquidity_level": float(bear_zone.get("level", bear_zone["bottom"])),
                "liquidity_pivot_time": bear_zone.get("pivot_time"),
                "liquidity_confirm_time": bear_zone.get("confirm_time", bear_zone.get("time")),
                "liquidity_sfp_strict": bool(bear_zone.get("source") == "sfp_pivot"),
            }

        if long_setup and not long_setup["choch_found"]:
            pre_choch_invalid = closes[i] < long_setup["zone_bottom"] if cfg.invalidate_on_close else lows[i] < long_setup["zone_bottom"]
            if i - long_setup["sweep_idx"] > max_choch_wait_exec_bars or pre_choch_invalid:
                long_setup = None
            else:
                long_setup["sweep_extreme"] = min(long_setup["sweep_extreme"], lows[i])
                while long_setup["event_search_idx"] < len(structure_events) and structure_events[long_setup["event_search_idx"]]["time"] <= close_times[i]:
                    event = structure_events[long_setup["event_search_idx"]]
                    long_setup["event_search_idx"] += 1
                    if event["direction"] != "bull" or event["time"] <= long_setup["sweep_time"] or i <= long_setup["sweep_idx"]:
                        continue
                    if cfg.require_structure_fvg and not event.get("has_fvg", False):
                        continue
                    start_idx = max(long_setup["sweep_idx"] + 1, i - cfg.ob_search_exec_bars)
                    ob = find_last_opposite_candle(opens, highs, lows, closes, start_idx, i - 1, "long", cfg.ob_use_body)
                    if ob is None:
                        long_setup = None
                    else:
                        long_setup["choch_found"] = True
                        long_setup["choch_exec_idx"] = i
                        long_setup["choch_time"] = event["time"]
                        long_setup["choch_break_level"] = event["break_level"]
                        long_setup["ob_top"], long_setup["ob_bottom"] = ob
                        long_setup["planned_stop"] = long_setup["sweep_extreme"] - atrs[i] * cfg.stop_buffer_atr
                        ob_range = long_setup["ob_top"] - long_setup["ob_bottom"]
                        long_setup["limit_price"] = (
                            long_setup["zone_top"]
                            if cfg.entry_mode == "zone_retest"
                            else long_setup["ob_bottom"] + ob_range * cfg.limit_entry_pos
                        )
                    break

        if short_setup and not short_setup["choch_found"]:
            pre_choch_invalid = closes[i] > short_setup["zone_top"] if cfg.invalidate_on_close else highs[i] > short_setup["zone_top"]
            if i - short_setup["sweep_idx"] > max_choch_wait_exec_bars or pre_choch_invalid:
                short_setup = None
            else:
                short_setup["sweep_extreme"] = max(short_setup["sweep_extreme"], highs[i])
                while short_setup["event_search_idx"] < len(structure_events) and structure_events[short_setup["event_search_idx"]]["time"] <= close_times[i]:
                    event = structure_events[short_setup["event_search_idx"]]
                    short_setup["event_search_idx"] += 1
                    if event["direction"] != "bear" or event["time"] <= short_setup["sweep_time"] or i <= short_setup["sweep_idx"]:
                        continue
                    if cfg.require_structure_fvg and not event.get("has_fvg", False):
                        continue
                    start_idx = max(short_setup["sweep_idx"] + 1, i - cfg.ob_search_exec_bars)
                    ob = find_last_opposite_candle(opens, highs, lows, closes, start_idx, i - 1, "short", cfg.ob_use_body)
                    if ob is None:
                        short_setup = None
                    else:
                        short_setup["choch_found"] = True
                        short_setup["choch_exec_idx"] = i
                        short_setup["choch_time"] = event["time"]
                        short_setup["choch_break_level"] = event["break_level"]
                        short_setup["ob_top"], short_setup["ob_bottom"] = ob
                        short_setup["planned_stop"] = short_setup["sweep_extreme"] + atrs[i] * cfg.stop_buffer_atr
                        ob_range = short_setup["ob_top"] - short_setup["ob_bottom"]
                        short_setup["limit_price"] = (
                            short_setup["zone_bottom"]
                            if cfg.entry_mode == "zone_retest"
                            else short_setup["ob_top"] - ob_range * cfg.limit_entry_pos
                        )
                    break

        if long_setup and long_setup["choch_found"] and position is None and (pending_entry is None or pending_entry["direction"] == "long"):
            while long_setup["event_search_idx"] < len(structure_events) and structure_events[long_setup["event_search_idx"]]["time"] <= close_times[i]:
                event = structure_events[long_setup["event_search_idx"]]
                long_setup["event_search_idx"] += 1
                if event["direction"] == "bull" and event["time"] > long_setup["choch_time"]:
                    long_setup["latest_bull_break_level"] = event["break_level"]
            expired = i - long_setup["choch_exec_idx"] > cfg.retest_valid_exec_bars
            long_invalid_boundary = long_setup["zone_bottom"] if cfg.entry_mode == "zone_retest" else long_setup["ob_bottom"]
            invalid = long_pre_entry_invalid(
                cfg.pre_entry_invalidation_mode,
                long_setup["retry_armed"],
                cfg.invalidate_on_close,
                closes[i],
                lows[i],
                long_invalid_boundary,
                long_setup["zone_bottom"],
                long_setup["planned_stop"],
            )
            retry_trend_invalid = cfg.pre_entry_invalidation_mode == "Blocked Setup Retry" and long_setup["retry_armed"] and not direction_allowed(cfg, "long", bull_score, bear_score)
            retry_structure_invalid = (
                cfg.pre_entry_invalidation_mode == "Blocked Setup Retry"
                and long_setup["retry_armed"]
                and long_setup.get("latest_bull_break_level") is not None
                and long_setup["choch_break_level"] is not None
                and long_setup["latest_bull_break_level"] > long_setup["choch_break_level"]
            )
            if expired or invalid or retry_trend_invalid or retry_structure_invalid:
                long_setup = None
                if pending_entry and pending_entry["direction"] == "long":
                    pending_entry = None
            elif not in_dead_zone and direction_allowed(cfg, "long", bull_score, bear_score):
                if cfg.entry_mode == "retest_close":
                    ob_range = long_setup["ob_top"] - long_setup["ob_bottom"]
                    retest_close = long_setup["ob_bottom"] + ob_range * cfg.retest_close_pos
                    touched = lows[i] <= long_setup["ob_top"] and highs[i] >= long_setup["ob_bottom"]
                    if touched and closes[i] >= retest_close and i + 1 < len(exec_df) and long_setup["planned_stop"] < closes[i]:
                        pending_entry = {
                            "direction": "long",
                            "order_type": "market",
                            "submitted_index": i,
                            "entry_price": None,
                            "stop": long_setup["planned_stop"],
                            "signal_index": i,
                            "signal_time": close_times[i],
                            "setup": long_setup.copy(),
                        }
                else:
                    if long_setup["planned_stop"] < long_setup["limit_price"]:
                        pending_entry = {
                            "direction": "long",
                            "order_type": "limit",
                            "submitted_index": i,
                            "entry_price": long_setup["limit_price"],
                            "stop": long_setup["planned_stop"],
                            "signal_index": i,
                            "signal_time": close_times[i],
                            "setup": long_setup.copy(),
                        }
            else:
                if pending_entry and pending_entry["direction"] == "long":
                    pending_entry = None
                if cfg.entry_mode == "retest_close":
                    ob_range = long_setup["ob_top"] - long_setup["ob_bottom"]
                    retest_close = long_setup["ob_bottom"] + ob_range * cfg.retest_close_pos
                    touched = lows[i] <= long_setup["ob_top"] and highs[i] >= long_setup["ob_bottom"]
                    would_enter = touched and closes[i] >= retest_close and long_setup["planned_stop"] < closes[i]
                else:
                    would_enter = lows[i] <= long_setup["limit_price"] <= highs[i] and long_setup["planned_stop"] < long_setup["limit_price"]
                if (
                    cfg.pre_entry_invalidation_mode == "Blocked Setup Retry"
                    and in_dead_zone
                    and direction_allowed(cfg, "long", bull_score, bear_score)
                    and would_enter
                ):
                    long_setup["retry_armed"] = True

        if short_setup and short_setup["choch_found"] and position is None and (pending_entry is None or pending_entry["direction"] == "short"):
            while short_setup["event_search_idx"] < len(structure_events) and structure_events[short_setup["event_search_idx"]]["time"] <= close_times[i]:
                event = structure_events[short_setup["event_search_idx"]]
                short_setup["event_search_idx"] += 1
                if event["direction"] == "bear" and event["time"] > short_setup["choch_time"]:
                    short_setup["latest_bear_break_level"] = event["break_level"]
            expired = i - short_setup["choch_exec_idx"] > cfg.retest_valid_exec_bars
            short_invalid_boundary = short_setup["zone_top"] if cfg.entry_mode == "zone_retest" else short_setup["ob_top"]
            invalid = short_pre_entry_invalid(
                cfg.pre_entry_invalidation_mode,
                short_setup["retry_armed"],
                cfg.invalidate_on_close,
                closes[i],
                highs[i],
                short_invalid_boundary,
                short_setup["zone_top"],
                short_setup["planned_stop"],
            )
            retry_trend_invalid = cfg.pre_entry_invalidation_mode == "Blocked Setup Retry" and short_setup["retry_armed"] and not direction_allowed(cfg, "short", bull_score, bear_score)
            retry_structure_invalid = (
                cfg.pre_entry_invalidation_mode == "Blocked Setup Retry"
                and short_setup["retry_armed"]
                and short_setup.get("latest_bear_break_level") is not None
                and short_setup["choch_break_level"] is not None
                and short_setup["latest_bear_break_level"] < short_setup["choch_break_level"]
            )
            if expired or invalid or retry_trend_invalid or retry_structure_invalid:
                short_setup = None
                if pending_entry and pending_entry["direction"] == "short":
                    pending_entry = None
            elif not in_dead_zone and direction_allowed(cfg, "short", bull_score, bear_score):
                if cfg.entry_mode == "retest_close":
                    ob_range = short_setup["ob_top"] - short_setup["ob_bottom"]
                    retest_close = short_setup["ob_top"] - ob_range * cfg.retest_close_pos
                    touched = highs[i] >= short_setup["ob_bottom"] and lows[i] <= short_setup["ob_top"]
                    if touched and closes[i] <= retest_close and i + 1 < len(exec_df) and short_setup["planned_stop"] > closes[i]:
                        pending_entry = {
                            "direction": "short",
                            "order_type": "market",
                            "submitted_index": i,
                            "entry_price": None,
                            "stop": short_setup["planned_stop"],
                            "signal_index": i,
                            "signal_time": close_times[i],
                            "setup": short_setup.copy(),
                        }
                else:
                    if short_setup["planned_stop"] > short_setup["limit_price"]:
                        pending_entry = {
                            "direction": "short",
                            "order_type": "limit",
                            "submitted_index": i,
                            "entry_price": short_setup["limit_price"],
                            "stop": short_setup["planned_stop"],
                            "signal_index": i,
                            "signal_time": close_times[i],
                            "setup": short_setup.copy(),
                        }
            else:
                if pending_entry and pending_entry["direction"] == "short":
                    pending_entry = None
                if cfg.entry_mode == "retest_close":
                    ob_range = short_setup["ob_top"] - short_setup["ob_bottom"]
                    retest_close = short_setup["ob_top"] - ob_range * cfg.retest_close_pos
                    touched = highs[i] >= short_setup["ob_bottom"] and lows[i] <= short_setup["ob_top"]
                    would_enter = touched and closes[i] <= retest_close and short_setup["planned_stop"] > closes[i]
                else:
                    would_enter = lows[i] <= short_setup["limit_price"] <= highs[i] and short_setup["planned_stop"] > short_setup["limit_price"]
                if (
                    cfg.pre_entry_invalidation_mode == "Blocked Setup Retry"
                    and in_dead_zone
                    and direction_allowed(cfg, "short", bull_score, bear_score)
                    and would_enter
                ):
                    short_setup["retry_armed"] = True

        if (
            position is not None
            and pending_close is None
            and i - position["entry_index"] >= cfg.max_hold_exec_bars
            and i + 1 < len(exec_df)
        ):
            pending_close = {
                "submitted_index": i,
                "reason": "time",
            }

    if return_state:
        return trades, {
            "pending_entry": pending_entry,
            "pending_close": pending_close,
            "position": position,
            "long_setup": long_setup,
            "short_setup": short_setup,
            "latest_zone_hold_candidate": latest_zone_hold_candidate,
            "last_open_time": times[-1] if times else None,
            "last_close_time": close_times[-1] if close_times else None,
        }
    return trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BINANCE:ETHUSDT")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--start", help="UTC start date/datetime, e.g. 2024-01-01 or 2024-01-01T00:00:00Z")
    parser.add_argument("--end", help="UTC end date/datetime. Defaults to now.")
    parser.add_argument("--tf1", default="4h", help="First HTF zone timeframe, e.g. 4h, 1d, 1w.")
    parser.add_argument("--tf2", default="1d", help="Second HTF zone timeframe, e.g. 1d or 1w.")
    parser.add_argument("--htf-left", type=int, default=5)
    parser.add_argument("--htf-right", type=int, default=5)
    parser.add_argument("--htf-ob-search-bars", type=int, default=50)
    parser.add_argument("--max-structure-bars-to-choch", type=int, default=32)
    parser.add_argument("--no-dead-zone-filter", action="store_true")
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.0)
    parser.add_argument("--max-entry-risk-pct", type=float, default=math.inf)
    parser.add_argument("--max-zone-scan", type=int, default=0)
    parser.add_argument("--zone-hold-model", help="Optional sklearn/joblib model from ml_zone_hold_filter.py.")
    parser.add_argument("--zone-hold-min-prob", type=float, default=0.0)
    parser.add_argument("--zone-hold-filter-tf", default="4h")
    parser.add_argument("--reject-unscored-zone-hold", action="store_true")
    parser.add_argument("--longs-only", action="store_true")
    parser.add_argument("--shorts-only", action="store_true")
    args = parser.parse_args()

    end_dt = parse_utc_datetime(args.end) if args.end else datetime.now(timezone.utc)
    start_dt = parse_utc_datetime(args.start) if args.start else end_dt - timedelta(days=args.days)
    if start_dt >= end_dt:
        raise ValueError("--start must be before --end.")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    data_symbol = normalize_binance_spot_symbol(args.symbol)
    tf1 = normalize_timeframe(args.tf1)
    tf2 = normalize_timeframe(args.tf2)

    results = []
    for exec_tf in ["3m", "5m"]:
        exec_df = fetch_klines(args.symbol, exec_tf, start_ms, end_ms)
        from_label = exec_df["open_time"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")
        to_label = exec_df["close_time"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
        for structure_tf in ["15m", "1h"]:
            for entry_mode in ["zone_retest", "retest_close", "limit_mid"]:
                cfg = Config(
                    exec_tf=exec_tf,
                    structure_tf=structure_tf,
                    entry_mode=entry_mode,
                    tf1=tf1,
                    tf2=tf2,
                    htf_left=args.htf_left,
                    htf_right=args.htf_right,
                    htf_ob_search_bars=args.htf_ob_search_bars,
                    max_structure_bars_to_choch=args.max_structure_bars_to_choch,
                    block_dead_zone=not args.no_dead_zone_filter,
                    allow_longs=not args.shorts_only,
                    allow_shorts=not args.longs_only,
                    min_entry_risk_pct=args.min_entry_risk_pct,
                    max_entry_risk_pct=args.max_entry_risk_pct,
                    max_zone_scan=args.max_zone_scan,
                    zone_hold_model_path=args.zone_hold_model,
                    zone_hold_min_prob=args.zone_hold_min_prob,
                    zone_hold_filter_tf=args.zone_hold_filter_tf,
                    zone_hold_reject_unscored=args.reject_unscored_zone_hold,
                )
                trades = run_backtest(exec_df, cfg)
                summary = summarize(trades)
                long_side = side_metrics(trades, "long")
                short_side = side_metrics(trades, "short")
                results.append({
                    "source": f"BINANCE spot {data_symbol}",
                    "tf1": tf1,
                    "tf2": tf2,
                    "exec_tf": exec_tf,
                    "structure_tf": structure_tf,
                    "entry_mode": entry_mode,
                    "from": from_label,
                    "to": to_label,
                    "trades": summary["trades"],
                    "win_rate": summary["win_rate"],
                    "profit_factor": summary["profit_factor"],
                    "net_r": summary["net_r"],
                    "avg_r": summary["avg_r"],
                    "long_net_r": round(long_side["net_r"], 3),
                    "short_net_r": round(short_side["net_r"], 3),
                    "zone_hold_min_prob": args.zone_hold_min_prob if args.zone_hold_model else 0.0,
                    "min_entry_risk_pct": args.min_entry_risk_pct,
                })

    out = pd.DataFrame(results).sort_values(["net_r", "profit_factor"], ascending=[False, False])
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
