from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pyharmonics.search import HarmonicSearch
    from pyharmonics.technicals import OHLCTechnicals
except ImportError as exc:  # pragma: no cover - depends on optional research package.
    raise SystemExit(
        "pyharmonics is required for this research script. Install it with: "
        "python -m pip install pyharmonics"
    ) from exc

from scripts.backtest_wolfe_wave import (  # noqa: E402
    INTERVAL_MS,
    add_indicators,
    ensure_ohlcv_frame,
    fetch_bybit_klines,
    high_before_low,
    load_ohlcv_csv,
    normalize_timeframe,
    parse_utc_datetime,
    resample_ohlc,
    strategy_metrics,
)


LOWPASS_NUMERIC_WEIGHTS = {
    "peak_spacing": 0.80,
    "fib_tolerance": 1.00,
    "forming_percent_c_to_d": 0.45,
    "confirm_bars": 0.70,
    "entry_window_bars": 0.55,
    "htf_stretch_atr": 0.65,
    "htf_rsi_extreme": 0.50,
    "prz_atr_buffer": 0.60,
    "stop_atr_buffer": 0.85,
    "rr": 1.00,
    "breakeven_trigger_r": 0.70,
    "min_harmonic_quality_score": 0.80,
    "max_hold_bars": 0.45,
}
LOWPASS_CATEGORICAL_WEIGHTS = {
    "pattern_tf": 1.35,
    "family": 1.10,
    "pattern_mode": 1.10,
    "pattern_name_filter": 0.95,
    "direction_filter": 0.75,
    "entry_mode": 0.75,
    "time_filter": 0.65,
    "htf_filter": 0.85,
    "candle_filter": 0.85,
    "trigger_candle_filter": 0.70,
    "trend_filter": 0.80,
}
LOWPASS_METRICS = [
    "robust_score",
    "train_net_r",
    "validation_net_r",
    "oos_net_r",
    "all_net_r",
    "train_avg_r",
    "validation_avg_r",
    "oos_avg_r",
    "all_avg_r",
    "train_trades",
    "validation_trades",
    "oos_trades",
    "all_trades",
]


@dataclass(frozen=True)
class HarmonicConfig:
    pattern_tf: str = "1h"
    family: str = "ABC"
    peak_spacing: int = 20
    fib_tolerance: float = 0.03
    pattern_mode: str = "formed"
    forming_percent_c_to_d: float = 0.80
    pattern_lookback_bars: int = 1200
    pattern_step_bars: int = 200
    search_limit_to: int = -1
    confirm_bars: int = 20
    entry_window_bars: int = 48
    entry_mode: str = "next_open"
    prz_atr_buffer: float = 0.15
    candle_filter: str = "none"
    trigger_candle_filter: str = "all"
    pattern_name_filter: str = "all"
    direction_filter: str = "both"
    stop_atr_buffer: float = 0.35
    rr: float = 2.0
    breakeven_trigger_r: float = 0.0
    min_harmonic_quality_score: float = 0.0
    max_hold_bars: int = 96
    trend_filter: str = "none"
    time_filter: str = "all"
    htf_filter: str = "none"
    htf_stretch_atr: float = 0.75
    htf_rsi_extreme: float = 55.0
    atr_length: int = 14
    ema_length: int = 200
    rsi_length: int = 14
    fee_bps_side: float = 5.5
    slippage_bps_side: float = 1.0
    max_fee_to_price_risk: float = 0.25
    min_entry_risk_pct: float = 0.001
    risk_fraction: float = 0.01
    one_trade_at_a_time: bool = True


@dataclass(frozen=True)
class HarmonicPatternEvent:
    symbol: str
    family: str
    name: str
    direction: str
    completion_time: pd.Timestamp
    detection_time: pd.Timestamp
    pattern_mode: str
    completion_price: float
    completion_min_price: float
    completion_max_price: float
    x: tuple[pd.Timestamp, ...]
    y: tuple[float, ...]
    retraces: dict[str, float]
    formed: bool
    pattern_tf: str
    peak_spacing: int
    fib_tolerance: float
    event_key: str


def parse_csv_values(value: str, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for chunk in str(value or "").split(","):
        text = chunk.strip()
        if text:
            out.append(cast(text))
    return out


def filter_tokens(value: str, *, normalize: str = "lower", wildcards: set[str] | None = None) -> set[str]:
    wildcards = wildcards or {"all", "any", "*"}
    if normalize == "upper":
        wildcards = {item.upper() for item in wildcards}
    elif normalize == "lower":
        wildcards = {item.lower() for item in wildcards}
    tokens: set[str] = set()
    for chunk in str(value or "").replace("|", ",").replace("+", ",").split(","):
        text = chunk.strip()
        if not text:
            continue
        if normalize == "upper":
            text = text.upper()
        elif normalize == "lower":
            text = text.lower()
        tokens.add(text)
    if not tokens or tokens <= wildcards:
        return set()
    return tokens - wildcards


def family_tokens(value: str) -> set[str]:
    return filter_tokens(value, normalize="upper")


def event_filter_matches(event: "HarmonicPatternEvent", cfg: "HarmonicConfig") -> bool:
    families = family_tokens(cfg.family)
    if families and event.family.upper() not in families:
        return False
    directions = filter_tokens(
        cfg.direction_filter,
        normalize="lower",
        wildcards={"all", "any", "*", "both"},
    )
    if directions and event.direction.lower() not in directions:
        return False
    names = filter_tokens(cfg.pattern_name_filter, normalize="upper")
    return not names or event.name.upper() in names


def trigger_candle_filter_matches(trigger_info: dict[str, Any], cfg: "HarmonicConfig") -> bool:
    triggers = filter_tokens(cfg.trigger_candle_filter, normalize="lower")
    if not triggers:
        return True
    primary = str(trigger_info.get("trigger_candle_primary", "none") or "none").lower()
    tags = {
        token.strip().lower()
        for token in str(trigger_info.get("trigger_candle_tags", "none") or "none").split("|")
        if token.strip()
    }
    return primary in triggers or bool(tags & triggers)


def timeframe_delta(timeframe: str) -> timedelta:
    return timedelta(milliseconds=INTERVAL_MS[normalize_timeframe(timeframe)])


def round_trip_cost_rate(cfg: HarmonicConfig) -> float:
    return max((2.0 * cfg.fee_bps_side + 2.0 * cfg.slippage_bps_side) / 10_000.0, 0.0)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def log_ratio_score(value: float, *, target: float = 1.0, width: float = math.log(2.2)) -> float:
    value = finite_float(value, 0.0)
    target = max(finite_float(target, 1.0), 1e-12)
    if value <= 0.0:
        return 0.0
    offset = math.log(value / target) / max(width, 1e-12)
    return float(math.exp(-(offset * offset)))


def range_score(value: float, low: float, high: float, *, softness: float) -> float:
    value = finite_float(value, math.nan)
    if not math.isfinite(value):
        return 0.5
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return float(math.exp(-((distance / max(softness, 1e-12)) ** 2)))


def harmonic_geometry_features(event: HarmonicPatternEvent) -> dict[str, float]:
    points = len(event.x)
    out: dict[str, float] = {
        "harmonic_point_count": float(points),
        "harmonic_quality_score": 0.0,
        "harmonic_time_score": 0.0,
        "harmonic_slope_score": 0.0,
        "harmonic_compactness_score": 0.0,
        "harmonic_bc_time_score": 0.0,
        "harmonic_fib_score": 0.0,
        "harmonic_ab_bars": math.nan,
        "harmonic_bc_bars": math.nan,
        "harmonic_cd_bars": math.nan,
        "harmonic_ab_move": math.nan,
        "harmonic_bc_move": math.nan,
        "harmonic_cd_move": math.nan,
        "harmonic_cd_ab_time_ratio": math.nan,
        "harmonic_ab_cd_time_balance": math.nan,
        "harmonic_bc_ab_time_ratio": math.nan,
        "harmonic_cd_ab_price_ratio": math.nan,
        "harmonic_ab_cd_slope_balance": math.nan,
        "harmonic_cd_prior_time_ratio": math.nan,
        "harmonic_abc_retrace": finite_float(event.retraces.get("ABC"), math.nan),
        "harmonic_bcd_extension": finite_float(event.retraces.get("BCD"), math.nan),
    }
    if points < 4:
        return out

    seconds_per_bar = max(timeframe_delta(event.pattern_tf).total_seconds(), 1.0)
    times = [pd.Timestamp(value).tz_convert("UTC") for value in event.x]
    prices = [float(value) for value in event.y]
    leg_bars = [
        max((times[idx + 1] - times[idx]).total_seconds() / seconds_per_bar, 1e-9)
        for idx in range(points - 1)
    ]
    leg_moves = [abs(prices[idx + 1] - prices[idx]) for idx in range(points - 1)]

    ab_bars = leg_bars[-3]
    bc_bars = leg_bars[-2]
    cd_bars = leg_bars[-1]
    ab_move = leg_moves[-3]
    bc_move = leg_moves[-2]
    cd_move = leg_moves[-1]
    ab_slope = ab_move / max(ab_bars, 1e-9)
    cd_slope = cd_move / max(cd_bars, 1e-9)
    cd_ab_time_ratio = cd_bars / max(ab_bars, 1e-9)
    bc_ab_time_ratio = bc_bars / max(ab_bars, 1e-9)
    cd_ab_price_ratio = cd_move / max(ab_move, 1e-12)
    ab_cd_time_balance = min(ab_bars, cd_bars) / max(ab_bars, cd_bars)
    ab_cd_slope_balance = min(ab_slope, cd_slope) / max(ab_slope, cd_slope, 1e-12)
    cd_prior_time_ratio = cd_bars / max(ab_bars + bc_bars, 1e-9)

    time_score = log_ratio_score(cd_ab_time_ratio, target=1.0, width=math.log(2.3))
    slope_score = max(min(ab_cd_slope_balance, 1.0), 0.0) ** 0.75
    compactness_score = 1.0 if cd_prior_time_ratio <= 1.0 else log_ratio_score(cd_prior_time_ratio, target=1.0, width=math.log(2.0))
    bc_time_score = log_ratio_score(bc_ab_time_ratio, target=0.8, width=math.log(2.5))
    abc_score = range_score(out["harmonic_abc_retrace"], 0.35, 0.95, softness=0.18)
    bcd_score = range_score(out["harmonic_bcd_extension"], 0.90, 2.60, softness=0.45)
    fib_score = 0.5 * abc_score + 0.5 * bcd_score
    quality = 100.0 * (
        0.35 * time_score
        + 0.25 * slope_score
        + 0.20 * compactness_score
        + 0.10 * bc_time_score
        + 0.10 * fib_score
    )

    out.update(
        {
            "harmonic_quality_score": float(quality),
            "harmonic_time_score": float(100.0 * time_score),
            "harmonic_slope_score": float(100.0 * slope_score),
            "harmonic_compactness_score": float(100.0 * compactness_score),
            "harmonic_bc_time_score": float(100.0 * bc_time_score),
            "harmonic_fib_score": float(100.0 * fib_score),
            "harmonic_ab_bars": float(ab_bars),
            "harmonic_bc_bars": float(bc_bars),
            "harmonic_cd_bars": float(cd_bars),
            "harmonic_ab_move": float(ab_move),
            "harmonic_bc_move": float(bc_move),
            "harmonic_cd_move": float(cd_move),
            "harmonic_cd_ab_time_ratio": float(cd_ab_time_ratio),
            "harmonic_ab_cd_time_balance": float(ab_cd_time_balance),
            "harmonic_bc_ab_time_ratio": float(bc_ab_time_ratio),
            "harmonic_cd_ab_price_ratio": float(cd_ab_price_ratio),
            "harmonic_ab_cd_slope_balance": float(ab_cd_slope_balance),
            "harmonic_cd_prior_time_ratio": float(cd_prior_time_ratio),
        }
    )
    return out


def add_htf_context(frame: pd.DataFrame, cfg: HarmonicConfig) -> pd.DataFrame:
    out = ensure_ohlcv_frame(frame).copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True).astype("datetime64[ns, UTC]")
    out["close_time"] = pd.to_datetime(out["close_time"], utc=True).astype("datetime64[ns, UTC]")
    base_cols = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
    for timeframe in ("1h", "4h"):
        prefix = f"htf_{timeframe}"
        if f"{prefix}_rsi" in out.columns and f"{prefix}_ema_dist_atr" in out.columns:
            continue
        htf = resample_ohlc(out[base_cols], timeframe)
        htf = add_indicators(htf, cfg.atr_length, cfg.ema_length, cfg.rsi_length)
        htf["close_time"] = pd.to_datetime(htf["close_time"], utc=True).astype("datetime64[ns, UTC]")
        htf[f"{prefix}_ema_dist_atr"] = (htf["close"] - htf["ema"]) / htf["atr"].replace(0.0, np.nan)
        context = htf[
            ["close_time", "rsi", "ema_slope_atr", "atr_ratio", f"{prefix}_ema_dist_atr"]
        ].rename(
            columns={
                "close_time": f"{prefix}_close_time",
                "rsi": f"{prefix}_rsi",
                "ema_slope_atr": f"{prefix}_ema_slope_atr",
                "atr_ratio": f"{prefix}_atr_ratio",
            }
        )
        out = pd.merge_asof(
            out.sort_values("open_time"),
            context.sort_values(f"{prefix}_close_time"),
            left_on="open_time",
            right_on=f"{prefix}_close_time",
            direction="backward",
        )
    return out.reset_index(drop=True)


def prepare_harmonic_frame(exec_df: pd.DataFrame, cfg: HarmonicConfig) -> pd.DataFrame:
    frame = ensure_ohlcv_frame(exec_df)
    indicator_columns = {"atr", "ema", "ema_slope_atr", "atr_ratio", "rsi", "volume_ratio"}
    if not indicator_columns.issubset(frame.columns):
        frame = add_indicators(frame, cfg.atr_length, cfg.ema_length, cfg.rsi_length)
    if "htf_1h_rsi" not in frame.columns or "htf_4h_rsi" not in frame.columns:
        frame = add_htf_context(frame, cfg)
    return frame


def to_pyharmonics_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_ohlcv_frame(df)
    out = frame.copy()
    out["dts"] = out["close_time"]
    out["close_time"] = pd.to_datetime(out["close_time"], utc=True).map(lambda ts: int(ts.timestamp()))
    return (
        out.set_index("open_time")[["open", "high", "low", "close", "volume", "close_time", "dts"]]
        .sort_index()
    )


def pattern_to_event(
    pattern: Any,
    *,
    symbol: str,
    family: str,
    pattern_tf: str,
    peak_spacing: int,
    fib_tolerance: float,
) -> HarmonicPatternEvent | None:
    raw = pattern.to_dict() if hasattr(pattern, "to_dict") else dict(pattern)
    x_raw = raw.get("x") or []
    y_raw = raw.get("y") or []
    if len(x_raw) < 3 or len(y_raw) != len(x_raw):
        return None
    x = tuple(pd.Timestamp(value).tz_convert("UTC") for value in x_raw)
    y = tuple(float(value) for value in y_raw)
    bullish = bool(raw.get("bullish"))
    direction = "long" if bullish else "short"
    formed = bool(raw.get("formed", True))
    completion_time = x[-1]
    if formed:
        detection_time = completion_time + timeframe_delta(pattern_tf) * int(peak_spacing)
        pattern_mode = "formed"
    else:
        detection_time = completion_time + timeframe_delta(pattern_tf)
        pattern_mode = "forming"
    observed_d_price = float(y[-1])
    completion_min = float(raw.get("completion_min_price", observed_d_price))
    completion_max = float(raw.get("completion_max_price", observed_d_price))
    completion_price = (completion_min + completion_max) / 2.0 if not formed else observed_d_price
    retraces = {
        str(key): float(value)
        for key, value in (raw.get("retraces") or {}).items()
        if value is not None and math.isfinite(float(value))
    }
    name = str(raw.get("name") or "unknown")
    event_key = (
        f"{symbol}:{pattern_tf}:{pattern_mode}:{family}:{name}:{direction}:"
        f"{completion_time.isoformat()}:{completion_price:.10g}"
    )
    return HarmonicPatternEvent(
        symbol=symbol,
        family=family,
        name=name,
        direction=direction,
        completion_time=completion_time,
        detection_time=detection_time,
        pattern_mode=pattern_mode,
        completion_price=completion_price,
        completion_min_price=completion_min,
        completion_max_price=completion_max,
        x=x,
        y=y,
        retraces=retraces,
        formed=formed,
        pattern_tf=pattern_tf,
        peak_spacing=int(peak_spacing),
        fib_tolerance=float(fib_tolerance),
        event_key=event_key,
    )


def find_harmonic_events(
    pattern_df: pd.DataFrame,
    cfg: HarmonicConfig,
    *,
    symbol: str,
    progress_label: str | None = None,
    progress_every_chunks: int = 0,
) -> list[HarmonicPatternEvent]:
    frame = ensure_ohlcv_frame(pattern_df)
    lookback = int(cfg.pattern_lookback_bars)
    step = max(1, int(cfg.pattern_step_bars))
    if lookback > 0 and len(frame) > lookback:
        events_by_key: dict[str, HarmonicPatternEvent] = {}
        total_chunks = max(1, math.ceil(max(0, len(frame) - lookback) / step) + 1)
        started = datetime.now(timezone.utc)
        for chunk_number, end in enumerate(range(lookback, len(frame) + step, step), start=1):
            end = min(end, len(frame))
            start = max(0, end - lookback)
            chunk = frame.iloc[start:end].copy()
            if len(chunk) < max(50, cfg.peak_spacing * 4):
                continue
            keep_start_idx = max(start, end - step)
            keep_start_time = pd.Timestamp(frame["open_time"].iloc[keep_start_idx]).tz_convert("UTC")
            keep_end_time = pd.Timestamp(frame["open_time"].iloc[end - 1]).tz_convert("UTC") + timeframe_delta(cfg.pattern_tf)
            for event in _find_harmonic_events_full(chunk, cfg, symbol=symbol):
                if event.pattern_mode == "formed":
                    detection_time = max(
                        event.detection_time,
                        event.completion_time + timeframe_delta(cfg.pattern_tf) * cfg.peak_spacing,
                    )
                else:
                    detection_time = max(event.detection_time, event.completion_time + timeframe_delta(cfg.pattern_tf))
                if keep_start_time <= detection_time <= keep_end_time:
                    events_by_key[event.event_key] = replace(event, detection_time=detection_time)
            if progress_label and progress_every_chunks > 0 and (
                chunk_number == 1 or chunk_number % progress_every_chunks == 0 or end >= len(frame)
            ):
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                print(
                    f"{progress_label} event_chunks={chunk_number}/{total_chunks} "
                    f"events={len(events_by_key)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if end >= len(frame):
                break
        return sorted(events_by_key.values(), key=lambda item: (item.completion_time, item.event_key))
    return _find_harmonic_events_full(frame, cfg, symbol=symbol)


def _find_harmonic_events_full(
    pattern_df: pd.DataFrame,
    cfg: HarmonicConfig,
    *,
    symbol: str,
) -> list[HarmonicPatternEvent]:
    ph_frame = to_pyharmonics_frame(pattern_df)
    technicals = OHLCTechnicals(ph_frame, symbol, cfg.pattern_tf, peak_spacing=int(cfg.peak_spacing))
    search = HarmonicSearch(technicals, fib_tolerance=float(cfg.fib_tolerance))
    requested = {item.strip().upper() for item in cfg.family.split("+") if item.strip()}
    events: list[HarmonicPatternEvent] = []
    modes = {item.strip().lower() for item in str(cfg.pattern_mode or "formed").split("+") if item.strip()}
    if "both" in modes:
        modes.update({"formed", "forming"})
    if not modes:
        modes = {"formed"}

    pattern_sets: list[dict[str, list[Any]]] = []
    if "formed" in modes:
        search.search(limit_to=int(cfg.search_limit_to))
        pattern_sets.append(search.get_patterns(formed=True))
    if "forming" in modes:
        search.forming(limit_to=int(cfg.search_limit_to), percent_c_to_d=float(cfg.forming_percent_c_to_d))
        pattern_sets.append(search.get_patterns(formed=False))

    for patterns in pattern_sets:
        for family, rows in patterns.items():
            family_name = str(family).upper()
            if requested and family_name not in requested:
                continue
            for pattern in rows:
                event = pattern_to_event(
                    pattern,
                    symbol=symbol,
                    family=family_name,
                    pattern_tf=cfg.pattern_tf,
                    peak_spacing=cfg.peak_spacing,
                    fib_tolerance=cfg.fib_tolerance,
                )
                if event is not None:
                    events.append(event)
    events.sort(key=lambda item: (item.completion_time, item.event_key))
    seen: set[str] = set()
    unique: list[HarmonicPatternEvent] = []
    for event in events:
        if event.event_key in seen:
            continue
        seen.add(event.event_key)
        unique.append(event)
    return unique


def split_trades(
    trades: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {"train": trades.copy(), "validation": trades.copy(), "oos": trades.copy()}
    entry_time = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    return {
        "train": trades[entry_time < train_end].copy(),
        "validation": trades[(entry_time >= train_end) & (entry_time < validation_end)].copy(),
        "oos": trades[entry_time >= validation_end].copy(),
    }


def trend_allows(row: pd.Series, direction: str, trend_filter: str) -> bool:
    mode = str(trend_filter or "none").strip().lower()
    if mode in {"", "none"}:
        return True
    close = float(row["close"])
    ema = float(row.get("ema", np.nan))
    slope = float(row.get("ema_slope_atr", np.nan))
    if not math.isfinite(ema) or not math.isfinite(slope):
        return False
    if mode == "with_ema":
        return close >= ema and slope >= 0.0 if direction == "long" else close <= ema and slope <= 0.0
    if mode == "counter_ema":
        return close <= ema and slope <= 0.0 if direction == "long" else close >= ema and slope >= 0.0
    if mode == "above_below_ema":
        return close >= ema if direction == "long" else close <= ema
    raise ValueError(f"Unsupported trend_filter={trend_filter!r}")


def time_filter_allows(row: pd.Series, cfg: HarmonicConfig) -> bool:
    mode = str(cfg.time_filter or "all").strip().lower()
    if mode in {"", "all", "none", "*"}:
        return True
    open_time = pd.Timestamp(row["open_time"]).tz_convert("UTC")
    hour = int(open_time.hour)
    weekday = int(open_time.weekday())
    is_weekday = weekday < 5
    is_asia = 0 <= hour <= 6
    is_eu_us = 7 <= hour <= 18
    if mode in {"weekday", "no_weekend"}:
        return is_weekday
    if mode == "no_asia":
        return not is_asia
    if mode == "eu_us":
        return is_eu_us
    if mode == "weekday_no_asia":
        return is_weekday and not is_asia
    if mode == "weekday_eu_us":
        return is_weekday and is_eu_us
    if mode == "no_sunday":
        return weekday != 6
    raise ValueError(f"Unsupported time_filter={cfg.time_filter!r}")


def htf_stretched(row: pd.Series, direction: str, cfg: HarmonicConfig, timeframe: str) -> bool:
    value = finite_float(row.get(f"htf_{timeframe}_ema_dist_atr"), math.nan)
    if not math.isfinite(value):
        return False
    threshold = max(float(cfg.htf_stretch_atr), 0.0)
    return value <= -threshold if direction == "long" else value >= threshold


def htf_rsi_extreme(row: pd.Series, direction: str, cfg: HarmonicConfig, timeframe: str) -> bool:
    value = finite_float(row.get(f"htf_{timeframe}_rsi"), math.nan)
    if not math.isfinite(value):
        return False
    upper = min(max(float(cfg.htf_rsi_extreme), 50.0), 99.0)
    lower = 100.0 - upper
    return value <= lower if direction == "long" else value >= upper


def htf_mean_reversion(row: pd.Series, direction: str, cfg: HarmonicConfig, timeframe: str) -> bool:
    dist = finite_float(row.get(f"htf_{timeframe}_ema_dist_atr"), math.nan)
    slope = finite_float(row.get(f"htf_{timeframe}_ema_slope_atr"), math.nan)
    if not math.isfinite(dist) or not math.isfinite(slope):
        return False
    threshold = max(float(cfg.htf_stretch_atr) * 0.5, 0.10)
    if direction == "long":
        return dist <= -threshold and slope <= 0.0
    return dist >= threshold and slope >= 0.0


def htf_filter_allows(row: pd.Series, direction: str, cfg: HarmonicConfig) -> bool:
    mode = str(cfg.htf_filter or "none").strip().lower()
    if mode in {"", "all", "none", "*"}:
        return True
    if mode == "htf_stretch_1h":
        return htf_stretched(row, direction, cfg, "1h")
    if mode == "htf_stretch_4h":
        return htf_stretched(row, direction, cfg, "4h")
    if mode == "htf_stretch_any":
        return htf_stretched(row, direction, cfg, "1h") or htf_stretched(row, direction, cfg, "4h")
    if mode == "htf_rsi_1h":
        return htf_rsi_extreme(row, direction, cfg, "1h")
    if mode == "htf_rsi_4h":
        return htf_rsi_extreme(row, direction, cfg, "4h")
    if mode == "htf_rsi_any":
        return htf_rsi_extreme(row, direction, cfg, "1h") or htf_rsi_extreme(row, direction, cfg, "4h")
    if mode == "htf_mean_reversion_1h":
        return htf_mean_reversion(row, direction, cfg, "1h")
    if mode == "htf_mean_reversion_4h":
        return htf_mean_reversion(row, direction, cfg, "4h")
    if mode == "htf_combo_1h":
        return htf_stretched(row, direction, cfg, "1h") and htf_rsi_extreme(row, direction, cfg, "1h")
    if mode == "htf_combo_4h":
        return htf_stretched(row, direction, cfg, "4h") and htf_rsi_extreme(row, direction, cfg, "4h")
    if mode == "htf_combo_any":
        return (
            htf_stretched(row, direction, cfg, "1h") and htf_rsi_extreme(row, direction, cfg, "1h")
        ) or (
            htf_stretched(row, direction, cfg, "4h") and htf_rsi_extreme(row, direction, cfg, "4h")
        )
    raise ValueError(f"Unsupported htf_filter={cfg.htf_filter!r}")


def context_allows(row: pd.Series, direction: str, cfg: HarmonicConfig) -> bool:
    return time_filter_allows(row, cfg) and htf_filter_allows(row, direction, cfg)


def candle_reversal_tags(frame: pd.DataFrame, idx: int, direction: str) -> set[str]:
    if idx < 0 or idx >= len(frame):
        return {"none"}
    row = frame.iloc[idx]
    prev = frame.iloc[idx - 1] if idx > 0 else row
    open_value = float(row["open"])
    high_value = float(row["high"])
    low_value = float(row["low"])
    close_value = float(row["close"])
    prev_open = float(prev["open"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])
    prev_close = float(prev["close"])
    candle_range = max(high_value - low_value, 1e-12)
    body = abs(close_value - open_value)
    upper_wick = high_value - max(open_value, close_value)
    lower_wick = min(open_value, close_value) - low_value
    close_pos = (close_value - low_value) / candle_range
    tags: set[str] = set()

    if body / candle_range <= 0.15:
        tags.add("small_body")

    if direction == "long":
        if (
            close_value > open_value
            and prev_close < prev_open
            and open_value <= prev_close
            and close_value >= prev_open
        ):
            tags.add("engulfing")
        if lower_wick >= max(body * 2.0, candle_range * 0.35) and close_pos >= 0.55:
            tags.add("pinbar")
        if high_value > prev_high and low_value < prev_low and close_pos >= 0.65:
            tags.add("outside_reversal")
        if close_pos >= 0.75 and close_value > open_value:
            tags.add("strong_close")
        if low_value < prev_low and close_value > prev_close:
            tags.add("reclaim")
    else:
        if (
            close_value < open_value
            and prev_close > prev_open
            and open_value >= prev_close
            and close_value <= prev_open
        ):
            tags.add("engulfing")
        if upper_wick >= max(body * 2.0, candle_range * 0.35) and close_pos <= 0.45:
            tags.add("pinbar")
        if high_value > prev_high and low_value < prev_low and close_pos <= 0.35:
            tags.add("outside_reversal")
        if close_pos <= 0.25 and close_value < open_value:
            tags.add("strong_close")
        if high_value > prev_high and close_value < prev_close:
            tags.add("reclaim")

    if not tags:
        tags.add("none")
    return tags


def primary_candle_pattern(tags: set[str]) -> str:
    for tag in ("engulfing", "pinbar", "outside_reversal", "reclaim", "strong_close", "small_body"):
        if tag in tags:
            return tag
    return "none"


def candle_filter_matches(tags: set[str], candle_filter: str) -> bool:
    mode = str(candle_filter or "none").strip().lower()
    if mode in {"", "none"}:
        return True
    if mode == "any_reversal":
        return bool(tags & {"engulfing", "pinbar", "outside_reversal", "reclaim", "strong_close"})
    return mode in tags


def completion_zone_bounds(event: HarmonicPatternEvent, atr: float, cfg: HarmonicConfig) -> tuple[float, float]:
    zone_low = min(event.completion_min_price, event.completion_max_price, event.completion_price)
    zone_high = max(event.completion_min_price, event.completion_max_price, event.completion_price)
    buffer = max(float(cfg.prz_atr_buffer) * max(float(atr), 0.0), 0.0)
    return zone_low - buffer, zone_high + buffer


def candle_touches_zone(row: pd.Series, event: HarmonicPatternEvent, atr: float, cfg: HarmonicConfig) -> bool:
    zone_low, zone_high = completion_zone_bounds(event, atr, cfg)
    return float(row["high"]) >= zone_low and float(row["low"]) <= zone_high


def entry_mode_value(cfg: HarmonicConfig) -> str:
    mode = str(cfg.entry_mode or "next_open").strip().lower()
    return "next_open" if mode in {"", "formed", "default"} else mode


def locate_trigger_break_entry(
    frame: pd.DataFrame,
    *,
    event: HarmonicPatternEvent,
    trigger_idx: int,
    start_idx: int,
    cfg: HarmonicConfig,
) -> tuple[int | None, dict[str, Any]]:
    if trigger_idx < 0 or trigger_idx >= len(frame) - 1:
        return None, {}
    mode = entry_mode_value(cfg)
    if mode not in {"trigger_break", "trigger_close_break"}:
        return start_idx, {}
    trigger_row = frame.iloc[trigger_idx]
    level = float(trigger_row["high"] if event.direction == "long" else trigger_row["low"])
    scan_start = max(start_idx, trigger_idx + 1)
    scan_end = min(len(frame) - 2, start_idx + max(0, int(cfg.entry_window_bars)))
    for break_idx in range(scan_start, scan_end + 1):
        row = frame.iloc[break_idx]
        if event.direction == "long":
            broke = float(row["close" if mode == "trigger_close_break" else "high"]) > level
        else:
            broke = float(row["close" if mode == "trigger_close_break" else "low"]) < level
        if not broke:
            continue
        return break_idx + 1, {
            "trigger_break_index": int(break_idx),
            "trigger_break_time": pd.Timestamp(row["close_time"]).tz_convert("UTC"),
            "trigger_break_level": float(level),
            "entry_trigger": mode,
        }
    return None, {}


def locate_entry(
    frame: pd.DataFrame,
    *,
    event: HarmonicPatternEvent,
    cfg: HarmonicConfig,
    start_idx: int,
) -> tuple[int | None, dict[str, Any]]:
    if start_idx >= len(frame) - 1:
        return None, {}

    scan_for_trigger = event.pattern_mode == "forming" or str(cfg.candle_filter or "none").lower() != "none"
    if not scan_for_trigger:
        trigger_idx = max(0, start_idx - 1)
        trigger_row = frame.iloc[trigger_idx]
        trigger_atr = float(trigger_row.get("atr", np.nan))
        tags = candle_reversal_tags(frame, trigger_idx, event.direction)
        entry_idx, break_info = locate_trigger_break_entry(
            frame,
            event=event,
            trigger_idx=trigger_idx,
            start_idx=start_idx,
            cfg=cfg,
        )
        if entry_idx is None:
            return None, {}
        return entry_idx, {
            "trigger_index": int(trigger_idx),
            "trigger_time": pd.Timestamp(trigger_row["close_time"]).tz_convert("UTC"),
            "trigger_candle_tags": "|".join(sorted(tags)),
            "trigger_candle_primary": primary_candle_pattern(tags),
            "trigger_touched_prz": bool(
                math.isfinite(trigger_atr) and candle_touches_zone(trigger_row, event, trigger_atr, cfg)
            ),
            "entry_trigger": "delayed_formed_entry",
            **break_info,
        }

    end_idx = min(len(frame) - 2, start_idx + max(0, int(cfg.entry_window_bars)))
    for trigger_idx in range(start_idx, end_idx + 1):
        trigger_row = frame.iloc[trigger_idx]
        trigger_atr = float(trigger_row.get("atr", np.nan))
        if not math.isfinite(trigger_atr) or trigger_atr <= 0.0:
            continue
        if not candle_touches_zone(trigger_row, event, trigger_atr, cfg):
            continue
        tags = candle_reversal_tags(frame, trigger_idx, event.direction)
        if not candle_filter_matches(tags, cfg.candle_filter):
            continue
        zone_low, zone_high = completion_zone_bounds(event, trigger_atr, cfg)
        default_entry_idx = trigger_idx + 1
        entry_idx, break_info = locate_trigger_break_entry(
            frame,
            event=event,
            trigger_idx=trigger_idx,
            start_idx=default_entry_idx,
            cfg=cfg,
        )
        if entry_idx is None:
            continue
        return entry_idx, {
            "trigger_index": int(trigger_idx),
            "trigger_time": pd.Timestamp(trigger_row["close_time"]).tz_convert("UTC"),
            "trigger_candle_tags": "|".join(sorted(tags)),
            "trigger_candle_primary": primary_candle_pattern(tags),
            "trigger_touched_prz": True,
            "trigger_zone_low": zone_low,
            "trigger_zone_high": zone_high,
            "entry_trigger": "prz_touch_reversal" if str(cfg.candle_filter or "none").lower() != "none" else "prz_touch",
            **break_info,
        }
    return None, {}


def simulate_trade(
    frame: pd.DataFrame,
    *,
    event: HarmonicPatternEvent,
    entry_idx: int,
    cfg: HarmonicConfig,
    symbol: str,
    trigger_info: dict[str, Any] | None = None,
    geometry: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    trigger_info = trigger_info or {}
    geometry = geometry or harmonic_geometry_features(event)
    entry_row = frame.iloc[entry_idx]
    if not trend_allows(entry_row, event.direction, cfg.trend_filter):
        return None
    if not context_allows(entry_row, event.direction, cfg):
        return None
    entry = float(entry_row["open"])
    atr = float(entry_row.get("atr", np.nan))
    if not math.isfinite(atr) or atr <= 0.0:
        return None

    if event.direction == "long":
        structural_stop = min(event.completion_min_price, event.completion_price)
        stop = structural_stop - cfg.stop_atr_buffer * atr
        if stop >= entry:
            return None
        risk = entry - stop
        target = entry + cfg.rr * risk
    else:
        structural_stop = max(event.completion_max_price, event.completion_price)
        stop = structural_stop + cfg.stop_atr_buffer * atr
        if stop <= entry:
            return None
        risk = stop - entry
        target = entry - cfg.rr * risk

    if risk <= 0.0:
        return None
    entry_risk_pct = risk / entry if entry > 0.0 else math.inf
    if entry_risk_pct < cfg.min_entry_risk_pct:
        return None
    cost_r = round_trip_cost_rate(cfg) * entry / risk
    if cfg.max_fee_to_price_risk > 0.0 and cost_r > cfg.max_fee_to_price_risk:
        return None

    exit_limit = min(len(frame) - 1, entry_idx + max(1, int(cfg.max_hold_bars)))
    exit_idx = exit_limit
    exit_price = float(frame["close"].iloc[exit_idx])
    exit_reason = "timeout"
    breakeven_trigger_r = max(finite_float(cfg.breakeven_trigger_r, 0.0), 0.0)
    breakeven_enabled = breakeven_trigger_r > 0.0 and breakeven_trigger_r < float(cfg.rr)
    breakeven_active = False
    breakeven_activation_idx: int | None = None
    breakeven_activation_time: pd.Timestamp | None = None
    breakeven_price = entry
    breakeven_trigger_price = (
        entry + breakeven_trigger_r * risk if event.direction == "long" else entry - breakeven_trigger_r * risk
    )

    def activate_breakeven(idx: int) -> None:
        nonlocal breakeven_active, breakeven_activation_idx, breakeven_activation_time
        if breakeven_active:
            return
        breakeven_active = True
        breakeven_activation_idx = int(idx)
        breakeven_activation_time = pd.Timestamp(frame["close_time"].iloc[idx]).tz_convert("UTC")

    for idx in range(entry_idx + 1, exit_limit + 1):
        row = frame.iloc[idx]
        open_value = float(row["open"])
        high_value = float(row["high"])
        low_value = float(row["low"])
        if event.direction == "long":
            target_hit = high_value >= target
            initial_stop_hit = low_value <= stop
            breakeven_trigger_hit = breakeven_enabled and high_value >= breakeven_trigger_price
            breakeven_stop_hit = breakeven_active and low_value <= breakeven_price
            if breakeven_active:
                if target_hit and breakeven_stop_hit:
                    if high_before_low(open_value, high_value, low_value):
                        exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                    else:
                        exit_idx, exit_price, exit_reason = idx, breakeven_price, "breakeven_same_bar"
                    break
                if breakeven_stop_hit:
                    exit_idx, exit_price, exit_reason = idx, breakeven_price, "breakeven"
                    break
                if target_hit:
                    exit_idx, exit_price, exit_reason = idx, target, "target"
                    break
                continue
            if target_hit and initial_stop_hit:
                if high_before_low(open_value, high_value, low_value):
                    if breakeven_trigger_hit:
                        activate_breakeven(idx)
                    exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                else:
                    exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                break
            if breakeven_trigger_hit and initial_stop_hit:
                if high_before_low(open_value, high_value, low_value):
                    activate_breakeven(idx)
                    if low_value <= breakeven_price:
                        exit_idx, exit_price, exit_reason = idx, breakeven_price, "breakeven_same_bar"
                        break
                else:
                    exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                    break
            if initial_stop_hit:
                exit_idx, exit_price, exit_reason = idx, stop, "stop"
                break
            if target_hit:
                if breakeven_trigger_hit:
                    activate_breakeven(idx)
                exit_idx, exit_price, exit_reason = idx, target, "target"
                break
            if breakeven_trigger_hit:
                activate_breakeven(idx)
        else:
            target_hit = low_value <= target
            initial_stop_hit = high_value >= stop
            breakeven_trigger_hit = breakeven_enabled and low_value <= breakeven_trigger_price
            breakeven_stop_hit = breakeven_active and high_value >= breakeven_price
            if breakeven_active:
                if target_hit and breakeven_stop_hit:
                    if high_before_low(open_value, high_value, low_value):
                        exit_idx, exit_price, exit_reason = idx, breakeven_price, "breakeven_same_bar"
                    else:
                        exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                    break
                if breakeven_stop_hit:
                    exit_idx, exit_price, exit_reason = idx, breakeven_price, "breakeven"
                    break
                if target_hit:
                    exit_idx, exit_price, exit_reason = idx, target, "target"
                    break
                continue
            if target_hit and initial_stop_hit:
                if high_before_low(open_value, high_value, low_value):
                    exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                else:
                    if breakeven_trigger_hit:
                        activate_breakeven(idx)
                    exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                break
            if breakeven_trigger_hit and initial_stop_hit:
                if high_before_low(open_value, high_value, low_value):
                    exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                    break
                activate_breakeven(idx)
                if high_value >= breakeven_price:
                    exit_idx, exit_price, exit_reason = idx, breakeven_price, "breakeven_same_bar"
                    break
            if initial_stop_hit:
                exit_idx, exit_price, exit_reason = idx, stop, "stop"
                break
            if target_hit:
                if breakeven_trigger_hit:
                    activate_breakeven(idx)
                exit_idx, exit_price, exit_reason = idx, target, "target"
                break
            if breakeven_trigger_hit:
                activate_breakeven(idx)

    gross_r = (exit_price - entry) / risk if event.direction == "long" else (entry - exit_price) / risk
    net_r = gross_r - cost_r
    return {
        "symbol": symbol,
        "strategy": "pyharmonics",
        "event_key": event.event_key,
        "family": event.family,
        "pattern_name": event.name,
        "pattern_mode": event.pattern_mode,
        "direction": event.direction,
        "completion_time": event.completion_time,
        "detection_time": event.detection_time,
        "trigger_time": trigger_info.get("trigger_time"),
        "trigger_index": trigger_info.get("trigger_index"),
        "trigger_candle_primary": trigger_info.get("trigger_candle_primary", "none"),
        "trigger_candle_tags": trigger_info.get("trigger_candle_tags", "none"),
        "trigger_touched_prz": bool(trigger_info.get("trigger_touched_prz", False)),
        "entry_trigger": trigger_info.get("entry_trigger", ""),
        "entry_mode": cfg.entry_mode,
        "trigger_break_index": trigger_info.get("trigger_break_index"),
        "trigger_break_time": trigger_info.get("trigger_break_time"),
        "trigger_break_level": trigger_info.get("trigger_break_level"),
        "trigger_zone_low": trigger_info.get("trigger_zone_low"),
        "trigger_zone_high": trigger_info.get("trigger_zone_high"),
        "entry_time": pd.Timestamp(frame["open_time"].iloc[entry_idx]).tz_convert("UTC"),
        "exit_time": pd.Timestamp(frame["close_time"].iloc[exit_idx]).tz_convert("UTC"),
        "entry_index": int(entry_idx),
        "exit_index": int(exit_idx),
        "entry_price": entry,
        "exit_price": float(exit_price),
        "stop_price": float(stop),
        "structural_stop_price": float(structural_stop),
        "target_price": float(target),
        "target_rr_planned": float(cfg.rr),
        "breakeven_trigger_r": float(breakeven_trigger_r),
        "breakeven_trigger_price": float(breakeven_trigger_price) if breakeven_enabled else math.nan,
        "breakeven_price": float(breakeven_price) if breakeven_enabled else math.nan,
        "breakeven_activated": bool(breakeven_active),
        "breakeven_activation_index": breakeven_activation_idx,
        "breakeven_activation_time": breakeven_activation_time,
        "entry_risk_pct": float(entry_risk_pct),
        "fee_to_price_risk": float(cost_r),
        "r_multiple_gross": float(gross_r),
        "r_multiple_net": float(net_r),
        "return_pct": float(cfg.risk_fraction * net_r),
        "hold_bars": int(exit_idx - entry_idx),
        "exit_reason": exit_reason,
        "pattern_tf": cfg.pattern_tf,
        "exec_tf": normalize_timeframe_from_frame(frame),
        "peak_spacing": int(cfg.peak_spacing),
        "fib_tolerance": float(cfg.fib_tolerance),
        "forming_percent_c_to_d": float(cfg.forming_percent_c_to_d),
        "confirm_bars": int(cfg.confirm_bars),
        "entry_window_bars": int(cfg.entry_window_bars),
        "prz_atr_buffer": float(cfg.prz_atr_buffer),
        "candle_filter": cfg.candle_filter,
        "trigger_candle_filter": cfg.trigger_candle_filter,
        "pattern_name_filter": cfg.pattern_name_filter,
        "direction_filter": cfg.direction_filter,
        "stop_atr_buffer": float(cfg.stop_atr_buffer),
        "min_harmonic_quality_score": float(cfg.min_harmonic_quality_score),
        "trend_filter": cfg.trend_filter,
        "time_filter": cfg.time_filter,
        "htf_filter": cfg.htf_filter,
        "htf_stretch_atr": float(cfg.htf_stretch_atr),
        "htf_rsi_extreme": float(cfg.htf_rsi_extreme),
        "entry_hour_utc": int(pd.Timestamp(frame["open_time"].iloc[entry_idx]).tz_convert("UTC").hour),
        "entry_weekday": int(pd.Timestamp(frame["open_time"].iloc[entry_idx]).tz_convert("UTC").weekday()),
        "htf_1h_rsi": finite_float(entry_row.get("htf_1h_rsi"), math.nan),
        "htf_1h_ema_dist_atr": finite_float(entry_row.get("htf_1h_ema_dist_atr"), math.nan),
        "htf_1h_ema_slope_atr": finite_float(entry_row.get("htf_1h_ema_slope_atr"), math.nan),
        "htf_1h_atr_ratio": finite_float(entry_row.get("htf_1h_atr_ratio"), math.nan),
        "htf_4h_rsi": finite_float(entry_row.get("htf_4h_rsi"), math.nan),
        "htf_4h_ema_dist_atr": finite_float(entry_row.get("htf_4h_ema_dist_atr"), math.nan),
        "htf_4h_ema_slope_atr": finite_float(entry_row.get("htf_4h_ema_slope_atr"), math.nan),
        "htf_4h_atr_ratio": finite_float(entry_row.get("htf_4h_atr_ratio"), math.nan),
        "completion_price": float(event.completion_price),
        "completion_min_price": float(event.completion_min_price),
        "completion_max_price": float(event.completion_max_price),
        "retraces": event.retraces,
        **geometry,
    }


def normalize_timeframe_from_frame(frame: pd.DataFrame) -> str:
    times = pd.to_datetime(frame["open_time"], utc=True, errors="coerce").dropna().sort_values()
    if len(times) < 2:
        return "unknown"
    seconds = float(times.diff().dropna().dt.total_seconds().median())
    return min(INTERVAL_MS, key=lambda tf: abs(INTERVAL_MS[tf] / 1000.0 - seconds))


def run_backtest(
    exec_df: pd.DataFrame,
    cfg: HarmonicConfig,
    *,
    symbol: str,
    precomputed_events: list[HarmonicPatternEvent] | None = None,
) -> pd.DataFrame:
    frame = prepare_harmonic_frame(exec_df, cfg)
    pattern_df = resample_ohlc(frame, cfg.pattern_tf)
    events = precomputed_events if precomputed_events is not None else find_harmonic_events(pattern_df, cfg, symbol=symbol)
    open_times = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    open_ns = np.array([pd.Timestamp(value).value for value in open_times], dtype=np.int64)
    confirm_delta = timeframe_delta(cfg.pattern_tf) * max(0, int(cfg.confirm_bars))
    trades: list[dict[str, Any]] = []
    next_available_idx = 0

    for event in events:
        if not event_filter_matches(event, cfg):
            continue
        geometry = harmonic_geometry_features(event)
        if finite_float(geometry.get("harmonic_quality_score"), 0.0) < float(cfg.min_harmonic_quality_score):
            continue
        if event.pattern_mode == "forming":
            available_time = event.detection_time
        else:
            available_time = max(event.detection_time, event.completion_time + confirm_delta)
        entry_ns = pd.Timestamp(available_time).value
        start_idx = int(np.searchsorted(open_ns, entry_ns, side="left"))
        if start_idx >= len(frame) - 1:
            continue
        if cfg.one_trade_at_a_time and start_idx < next_available_idx:
            continue
        entry_idx, trigger_info = locate_entry(frame, event=event, cfg=cfg, start_idx=start_idx)
        if entry_idx is None:
            continue
        if not trigger_candle_filter_matches(trigger_info, cfg):
            continue
        if cfg.one_trade_at_a_time and entry_idx < next_available_idx:
            continue
        trade = simulate_trade(
            frame,
            event=event,
            entry_idx=entry_idx,
            cfg=cfg,
            symbol=symbol,
            trigger_info=trigger_info,
            geometry=geometry,
        )
        if trade is None:
            continue
        trades.append(trade)
        next_available_idx = int(trade["exit_index"]) + 1

    return pd.DataFrame(trades)


def robust_score(metrics: dict[str, float]) -> float:
    train_trades = metrics.get("train_trades", 0.0)
    validation_trades = metrics.get("validation_trades", 0.0)
    oos_trades = metrics.get("oos_trades", 0.0)
    all_trades = metrics.get("all_trades", 0.0)
    if validation_trades < 2 or oos_trades < 2 or all_trades < 8:
        return -1000.0 + all_trades
    val_avg = metrics.get("validation_avg_r", 0.0)
    oos_avg = metrics.get("oos_avg_r", 0.0)
    train_avg = metrics.get("train_avg_r", 0.0)
    stability_penalty = abs(train_avg - oos_avg) + 0.5 * abs(val_avg - oos_avg)
    dd_penalty = abs(min(metrics.get("all_max_dd_r", 0.0), 0.0)) * 0.20
    trade_bonus = min(math.log1p(all_trades), 4.0) * 0.15
    return (
        metrics.get("oos_net_r", 0.0)
        + 0.65 * metrics.get("validation_net_r", 0.0)
        + 0.20 * metrics.get("train_net_r", 0.0)
        + 4.0 * min(val_avg, oos_avg)
        - 2.0 * stability_penalty
        - dd_penalty
        + trade_bonus
    )


def config_metrics(
    trades: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> dict[str, float]:
    buckets = split_trades(trades, train_end=train_end, validation_end=validation_end)
    out: dict[str, float] = {}
    for name, bucket in buckets.items():
        for key, value in strategy_metrics(bucket).items():
            out[f"{name}_{key}"] = value
    for key, value in strategy_metrics(trades).items():
        out[f"all_{key}"] = value
    out["robust_score"] = robust_score(out)
    return out


def candle_pattern_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "trigger_candle_primary" not in trades.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [
        ["trigger_candle_primary"],
        ["direction"],
        ["pattern_name"],
        ["direction", "trigger_candle_primary"],
        ["pattern_name", "trigger_candle_primary"],
        ["pattern_mode", "trigger_candle_primary"],
        ["family", "trigger_candle_primary"],
        ["pattern_mode", "family", "trigger_candle_primary"],
    ]
    for cols in group_cols:
        missing = [col for col in cols if col not in trades.columns]
        if missing:
            continue
        for keys, group in trades.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            metrics = strategy_metrics(group)
            row: dict[str, Any] = {
                "group": "+".join(cols),
                **{col: key for col, key in zip(cols, keys)},
                **metrics,
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["avg_r", "net_r", "trades"], ascending=[False, False, False]).reset_index(drop=True)


def lowpass_distance(row: pd.Series, other: pd.Series, table: pd.DataFrame) -> float:
    distance = 0.0
    for column, weight in LOWPASS_NUMERIC_WEIGHTS.items():
        values = pd.to_numeric(table[column], errors="coerce")
        scale = float(values.max() - values.min())
        if not math.isfinite(scale) or scale <= 0.0:
            scale = max(abs(float(row[column])) if pd.notna(row[column]) else 1.0, 1.0)
        distance += weight * abs(float(row[column]) - float(other[column])) / scale
    for column, weight in LOWPASS_CATEGORICAL_WEIGHTS.items():
        if str(row[column]) != str(other[column]):
            distance += weight
    return float(distance)


def add_lowpass_scores(
    table: pd.DataFrame,
    *,
    radius: float = 1.60,
    min_neighbors: int = 5,
) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    out = table.copy()
    size = len(out)
    distances = np.zeros((size, size), dtype=float)

    for column, weight in LOWPASS_NUMERIC_WEIGHTS.items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            scale = float(finite_values.max() - finite_values.min())
            fallback = max(abs(float(finite_values[0])), 1.0)
        else:
            scale = 0.0
            fallback = 1.0
        if not math.isfinite(scale) or scale <= 0.0:
            scale = fallback
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        distances += weight * np.abs(clean[:, None] - clean[None, :]) / scale

    for column, weight in LOWPASS_CATEGORICAL_WEIGHTS.items():
        if column not in out.columns:
            continue
        values = out[column].astype(str).to_numpy()
        distances += weight * (values[:, None] != values[None, :])

    metric_arrays = {
        metric: pd.to_numeric(out[metric], errors="coerce").to_numpy(dtype=float)
        for metric in LOWPASS_METRICS
        if metric in out.columns
    }
    metric_values = {metric: [] for metric in metric_arrays}
    neighbor_counts: list[int] = []
    fallback_neighbors = max(1, min(int(min_neighbors), size))
    for idx in range(size):
        neighbor_idx = np.flatnonzero(distances[idx] <= radius)
        if len(neighbor_idx) < fallback_neighbors:
            neighbor_idx = np.argpartition(distances[idx], fallback_neighbors - 1)[:fallback_neighbors]
        neighbor_counts.append(int(len(neighbor_idx)))
        for metric, values in metric_arrays.items():
            with np.errstate(all="ignore"):
                median = float(np.nanmedian(values[neighbor_idx]))
            metric_values[metric].append(median if math.isfinite(median) else 0.0)
    out["lowpass_neighbors"] = neighbor_counts
    for metric, values in metric_values.items():
        out[f"lowpass_{metric}"] = values
    if "lowpass_robust_score" not in out.columns and "robust_score" in out.columns:
        out["lowpass_robust_score"] = out["robust_score"]
    return out


def build_grid(args: argparse.Namespace) -> list[HarmonicConfig]:
    configs: list[HarmonicConfig] = []
    for (
        pattern_tf,
        family,
        pattern_mode,
        peak_spacing,
        fib_tolerance,
        forming_percent,
        confirm_bars,
        entry_window,
        entry_mode,
        prz_buffer,
        candle_filter,
        trigger_candle_filter,
        pattern_name_filter,
        direction_filter,
        stop_buffer,
        rr,
        breakeven_trigger,
        min_harmonic_quality_score,
        max_hold,
        trend_filter,
        time_filter,
        htf_filter,
        htf_stretch_atr,
        htf_rsi_extreme,
    ) in itertools.product(
        parse_csv_values(args.pattern_tfs, str),
        parse_csv_values(args.families, str),
        parse_csv_values(args.pattern_modes, str),
        parse_csv_values(args.peak_spacings, int),
        parse_csv_values(args.fib_tolerances, float),
        parse_csv_values(args.forming_percents, float),
        parse_csv_values(args.confirm_bars, int),
        parse_csv_values(args.entry_window_bars, int),
        parse_csv_values(args.entry_modes, str),
        parse_csv_values(args.prz_buffers, float),
        parse_csv_values(args.candle_filters, str),
        parse_csv_values(args.trigger_candle_filters, str),
        parse_csv_values(args.pattern_name_filters, str),
        parse_csv_values(args.direction_filters, str),
        parse_csv_values(args.stop_buffers, float),
        parse_csv_values(args.rrs, float),
        parse_csv_values(args.breakeven_triggers, float),
        parse_csv_values(args.min_harmonic_quality_scores, float),
        parse_csv_values(args.max_hold_bars, int),
        parse_csv_values(args.trend_filters, str),
        parse_csv_values(args.time_filters, str),
        parse_csv_values(args.htf_filters, str),
        parse_csv_values(args.htf_stretch_atrs, float),
        parse_csv_values(args.htf_rsi_extremes, float),
    ):
        configs.append(
            HarmonicConfig(
                pattern_tf=normalize_timeframe(pattern_tf),
                family=family.upper(),
                peak_spacing=peak_spacing,
                fib_tolerance=fib_tolerance,
                pattern_mode=pattern_mode.strip().lower(),
                forming_percent_c_to_d=forming_percent,
                pattern_lookback_bars=args.pattern_lookback_bars,
                pattern_step_bars=args.pattern_step_bars,
                search_limit_to=args.search_limit_to,
                confirm_bars=confirm_bars,
                entry_window_bars=entry_window,
                entry_mode=entry_mode.strip().lower(),
                prz_atr_buffer=prz_buffer,
                candle_filter=candle_filter.strip().lower(),
                trigger_candle_filter=trigger_candle_filter.strip().lower(),
                pattern_name_filter=pattern_name_filter.strip().upper(),
                direction_filter=direction_filter.strip().lower(),
                stop_atr_buffer=stop_buffer,
                rr=rr,
                breakeven_trigger_r=breakeven_trigger,
                min_harmonic_quality_score=min_harmonic_quality_score,
                max_hold_bars=max_hold,
                trend_filter=trend_filter,
                time_filter=time_filter.strip().lower(),
                htf_filter=htf_filter.strip().lower(),
                htf_stretch_atr=htf_stretch_atr,
                htf_rsi_extreme=htf_rsi_extreme,
                fee_bps_side=args.fee_bps_side,
                slippage_bps_side=args.slippage_bps_side,
                max_fee_to_price_risk=args.max_fee_to_price_risk,
                min_entry_risk_pct=args.min_entry_risk_pct,
                risk_fraction=args.risk_fraction,
                one_trade_at_a_time=not args.allow_overlap,
            )
        )
    if args.limit_configs and args.limit_configs > 0:
        configs = configs[: args.limit_configs]
    return configs


def load_or_fetch_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv:
        return load_ohlcv_csv(Path(args.input_csv))
    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end) if args.end else datetime.now(timezone.utc)
    return fetch_bybit_klines(args.symbol, args.exec_tf, start, end)


def run_tuning(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol = args.symbol.upper()
    base = ensure_ohlcv_frame(load_or_fetch_frame(args))
    configs = build_grid(args)
    train_end = pd.Timestamp(parse_utc_datetime(args.train_end))
    validation_end = pd.Timestamp(parse_utc_datetime(args.validation_end))
    event_cache: dict[tuple[Any, ...], list[HarmonicPatternEvent]] = {}
    rows: list[dict[str, Any]] = []
    best_trades = pd.DataFrame()
    best_score = -math.inf

    for number, cfg in enumerate(configs, start=1):
        key = (
            cfg.pattern_tf,
            cfg.family,
            cfg.pattern_mode,
            cfg.peak_spacing,
            cfg.fib_tolerance,
            cfg.forming_percent_c_to_d,
            cfg.pattern_lookback_bars,
            cfg.pattern_step_bars,
            cfg.search_limit_to,
        )
        if key not in event_cache:
            pattern_df = resample_ohlc(base, cfg.pattern_tf)
            event_cache[key] = find_harmonic_events(pattern_df, cfg, symbol=symbol)
        events = [
            event for event in event_cache[key]
            if event.family in {item.strip().upper() for item in cfg.family.split("+") if item.strip()}
        ]
        filtered_events = [event for event in events if event_filter_matches(event, cfg)]
        trades = run_backtest(base, cfg, symbol=symbol, precomputed_events=filtered_events)
        metrics = config_metrics(trades, train_end=train_end, validation_end=validation_end)
        row = {
            "symbol": symbol,
            "config_number": number,
            **asdict(cfg),
            "pattern_events": len(filtered_events),
            **metrics,
        }
        rows.append(row)
        score = float(metrics.get("robust_score", -math.inf))
        if score > best_score:
            best_score = score
            best_trades = trades.copy()
        if args.progress_every and number % args.progress_every == 0:
            print(f"evaluated {number}/{len(configs)} configs")

    table = pd.DataFrame(rows)
    table = add_lowpass_scores(
        table,
        radius=float(args.lowpass_radius),
        min_neighbors=int(args.lowpass_min_neighbors),
    )
    sort_columns = ["lowpass_robust_score", "robust_score", "oos_net_r", "validation_net_r"]
    table = table.sort_values(sort_columns, ascending=[False, False, False, False]).reset_index(drop=True)
    return table, best_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research pyharmonics-based crypto reversal strategies.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--exec-tf", default="15m")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--validation-end", default="2025-09-01")
    parser.add_argument("--pattern-tfs", default="1h,4h")
    parser.add_argument("--families", default="ABC,ABCD,XABCD")
    parser.add_argument("--pattern-modes", default="formed")
    parser.add_argument("--peak-spacings", default="10,14,20,28")
    parser.add_argument("--fib-tolerances", default="0.02,0.03,0.05")
    parser.add_argument("--forming-percents", default="0.80")
    parser.add_argument("--pattern-lookback-bars", type=int, default=1200)
    parser.add_argument("--pattern-step-bars", type=int, default=200)
    parser.add_argument("--search-limit-to", type=int, default=-1)
    parser.add_argument("--confirm-bars", default="5,10,20,28")
    parser.add_argument("--entry-window-bars", default="48")
    parser.add_argument("--entry-modes", default="next_open")
    parser.add_argument("--prz-buffers", default="0.15")
    parser.add_argument("--candle-filters", default="none")
    parser.add_argument("--trigger-candle-filters", default="all")
    parser.add_argument("--pattern-name-filters", default="all")
    parser.add_argument("--direction-filters", default="both")
    parser.add_argument("--stop-buffers", default="0.20,0.35,0.55")
    parser.add_argument("--rrs", default="1.25,1.5,2.0,2.5")
    parser.add_argument("--breakeven-triggers", default="0")
    parser.add_argument("--min-harmonic-quality-scores", default="0")
    parser.add_argument("--max-hold-bars", default="48,96,192")
    parser.add_argument("--trend-filters", default="none,with_ema,counter_ema")
    parser.add_argument("--time-filters", default="all")
    parser.add_argument("--htf-filters", default="none")
    parser.add_argument("--htf-stretch-atrs", default="0.75")
    parser.add_argument("--htf-rsi-extremes", default="55")
    parser.add_argument("--fee-bps-side", type=float, default=5.5)
    parser.add_argument("--slippage-bps-side", type=float, default=1.0)
    parser.add_argument("--max-fee-to-price-risk", type=float, default=0.25)
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.001)
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--lowpass-radius", type=float, default=1.60)
    parser.add_argument("--lowpass-min-neighbors", type=int, default=5)
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-prefix", default="scripts/pyharmonics_btc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, best_trades = run_tuning(args)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tuning_path = prefix.with_suffix(".tuning.csv")
    trades_path = prefix.with_suffix(".best_trades.csv")
    candle_path = prefix.with_suffix(".candle_patterns.csv")
    table.to_csv(tuning_path, index=False)
    best_trades.to_csv(trades_path, index=False)
    candle_pattern_summary(best_trades).to_csv(candle_path, index=False)
    print(f"Saved tuning table: {tuning_path}")
    print(f"Saved best trades: {trades_path}")
    print(f"Saved candle summary: {candle_path}")
    if not table.empty:
        cols = [
            "symbol",
            "pattern_tf",
            "family",
            "pattern_mode",
            "peak_spacing",
            "fib_tolerance",
            "forming_percent_c_to_d",
            "confirm_bars",
            "entry_window_bars",
            "entry_mode",
            "prz_atr_buffer",
            "candle_filter",
            "trigger_candle_filter",
            "pattern_name_filter",
            "direction_filter",
            "stop_atr_buffer",
            "rr",
            "breakeven_trigger_r",
            "min_harmonic_quality_score",
            "max_hold_bars",
            "trend_filter",
            "time_filter",
            "htf_filter",
            "htf_stretch_atr",
            "htf_rsi_extreme",
            "lowpass_robust_score",
            "robust_score",
            "all_trades",
            "validation_trades",
            "oos_trades",
            "validation_net_r",
            "oos_net_r",
            "all_net_r",
            "all_avg_r",
            "all_max_dd_r",
            "lowpass_neighbors",
        ]
        print(table[[col for col in cols if col in table.columns]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
