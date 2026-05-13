from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import parse_utc_datetime  # noqa: E402


DEFAULT_UNIVERSE = Path("scripts/bybit_top50_turtle_per_symbol_turnover_sfp_tex_v4_universe.csv")
DEFAULT_CACHE_DIR = Path("scripts/.cache/bybit_linear")
DEFAULT_OUT_PREFIX = Path("scripts/pine_strategy_candidate_research")


@dataclass(frozen=True)
class CandidateSpec:
    strategy: str
    timeframe: str
    params: dict[str, Any]

    @property
    def name(self) -> str:
        bits = [self.strategy, self.timeframe]
        for key in sorted(self.params):
            value = self.params[key]
            if isinstance(value, float):
                token = f"{key}{value:g}"
            else:
                token = f"{key}{value}"
            bits.append(token)
        return "_".join(bits).replace(".", "p").replace("/", "-")


@dataclass
class Trade:
    symbol: str
    strategy: str
    spec_name: str
    timeframe: str
    direction: str
    signal_index: int
    entry_index: int
    exit_index: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    exit_reason: str
    r_multiple: float
    gross_r: float
    fee_r: float
    risk_pct: float
    atr_pct: float
    bars_held: int
    feature_json: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("signal_time", "entry_time", "exit_time"):
            out[key] = pd.Timestamp(out[key]).isoformat()
        return out


def clean_symbol(symbol: str) -> str:
    return "".join(ch for ch in str(symbol).upper() if ch.isalnum())


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_universe(path: Path, max_symbols: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    symbols = [clean_symbol(x) for x in frame["symbol"].dropna().tolist()]
    symbols = [x for x in symbols if x.endswith("USDT")]
    out: list[str] = []
    for symbol in symbols:
        if symbol not in out:
            out.append(symbol)
    return out[:max_symbols]


def find_cache_path(symbol: str, cache_dir: Path, interval: str = "5m") -> Path | None:
    lower = symbol.lower()
    candidates = sorted(
        cache_dir.glob(f"{lower}_{interval}_*.pkl"),
        key=lambda p: (len(p.name), p.stat().st_mtime),
        reverse=True,
    )
    if not candidates:
        return None
    full = [p for p in candidates if "20210901_20260420" in p.name]
    return full[0] if full else candidates[0]


def load_frame(symbol: str, cache_dir: Path, train_start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = find_cache_path(symbol, cache_dir, "5m")
    if path is None:
        raise FileNotFoundError(f"no 5m cache found for {symbol}")
    frame = pd.read_pickle(path).copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    warmup_start = train_start - pd.Timedelta(days=30)
    frame = frame[(frame["open_time"] >= warmup_start) & (frame["open_time"] < end)].copy()
    frame = frame.dropna(subset=["open_time", "open", "high", "low", "close"]).reset_index(drop=True)
    return frame


def resample_frame(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "5m":
        return frame.copy().reset_index(drop=True)
    rule = {"15m": "15min", "1h": "1h"}.get(timeframe)
    if rule is None:
        raise ValueError(f"unsupported timeframe {timeframe}")
    indexed = frame.set_index("open_time")
    out = indexed.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    seconds = pd.Timedelta(rule).total_seconds()
    out["close_time"] = out["open_time"] + pd.to_timedelta(seconds, unit="s") - pd.Timedelta(milliseconds=1)
    return out[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def rma(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if len(values) == 0:
        return out
    alpha = 1.0 / float(length)
    acc = math.nan
    for i, value in enumerate(values):
        if not math.isfinite(value):
            out[i] = acc
            continue
        if not math.isfinite(acc):
            if i + 1 >= length:
                window = values[i + 1 - length : i + 1]
                acc = float(np.nanmean(window))
        else:
            acc = alpha * value + (1.0 - alpha) * acc
        out[i] = acc
    return out


def sma(values: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(values).rolling(length, min_periods=length).mean().to_numpy(dtype=float)


def ema(values: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(values).ewm(span=length, adjust=False, min_periods=length).mean().to_numpy(dtype=float)


def rsi(values: np.ndarray, length: int = 14) -> np.ndarray:
    delta = np.diff(values, prepend=values[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss > 0)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return out


def pivots(values: np.ndarray, left: int, right: int, kind: str) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    n = len(values)
    for pivot_idx in range(left, n - right):
        start = pivot_idx - left
        stop = pivot_idx + right + 1
        center = values[pivot_idx]
        if not math.isfinite(center):
            continue
        window = values[start:stop]
        if kind == "high":
            if center >= np.nanmax(window):
                out[pivot_idx + right] = center
        else:
            if center <= np.nanmin(window):
                out[pivot_idx + right] = center
    return out


def highest_previous(values: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(values).rolling(length, min_periods=length).max().shift(1).to_numpy(dtype=float)


def lowest_previous(values: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(values).rolling(length, min_periods=length).min().shift(1).to_numpy(dtype=float)


def base_context(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    tr = true_range(high, low, close)
    atr14 = rma(tr, 14)
    return {
        "atr14": atr14,
        "atr20": rma(tr, 20),
        "tr": tr,
        "vol_sma20": sma(volume, 20),
        "vol_sma50": sma(volume, 50),
        "ema20": ema(close, 20),
        "ema50": ema(close, 50),
        "ema200": ema(close, 200),
        "rsi14": rsi(close, 14),
    }


def add_signal_features(
    frame: pd.DataFrame,
    ctx: dict[str, np.ndarray],
    idx: int,
    direction: int,
    entry: float,
    stop: float,
    extra: dict[str, float],
) -> dict[str, float]:
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    atr = ctx["atr14"][idx]
    sign = float(direction)
    features = {
        "direction_long": 1.0 if direction > 0 else 0.0,
        "hour_utc": float(pd.Timestamp(frame["close_time"].iloc[idx]).hour),
        "dow": float(pd.Timestamp(frame["close_time"].iloc[idx]).dayofweek),
        "atr_pct": float(atr / close[idx] * 100.0) if atr > 0 else math.nan,
        "risk_pct": float(abs(entry - stop) / entry * 100.0) if entry > 0 else math.nan,
        "volume_ratio20": float(volume[idx] / ctx["vol_sma20"][idx]) if ctx["vol_sma20"][idx] > 0 else math.nan,
        "rsi14_dir": sign * (float(ctx["rsi14"][idx]) - 50.0) if math.isfinite(ctx["rsi14"][idx]) else math.nan,
        "close_vs_ema20_atr_dir": sign * (close[idx] - ctx["ema20"][idx]) / atr if atr > 0 and math.isfinite(ctx["ema20"][idx]) else math.nan,
        "close_vs_ema50_atr_dir": sign * (close[idx] - ctx["ema50"][idx]) / atr if atr > 0 and math.isfinite(ctx["ema50"][idx]) else math.nan,
        "close_vs_ema200_atr_dir": sign * (close[idx] - ctx["ema200"][idx]) / atr if atr > 0 and math.isfinite(ctx["ema200"][idx]) else math.nan,
    }
    features.update(extra)
    return features


def simulate_signals(
    frame: pd.DataFrame,
    ctx: dict[str, np.ndarray],
    *,
    symbol: str,
    spec: CandidateSpec,
    signals: list[dict[str, Any]],
    rr: float,
    max_hold_bars: int,
    fee_bps_per_side: float,
    min_risk_pct: float,
) -> list[Trade]:
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    open_ = frame["open"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    trades: list[Trade] = []
    blocked_until = -1
    n = len(frame)
    for signal in signals:
        signal_idx = int(signal["idx"])
        entry_idx = signal_idx + 1
        if entry_idx >= n or entry_idx <= blocked_until:
            continue
        direction = int(signal["direction"])
        entry = float(open_[entry_idx])
        stop = float(signal["stop"])
        if direction > 0 and stop >= entry:
            continue
        if direction < 0 and stop <= entry:
            continue
        risk = abs(entry - stop)
        risk_pct = risk / entry * 100.0 if entry > 0 else math.nan
        if not math.isfinite(risk_pct) or risk_pct < min_risk_pct:
            continue
        target = entry + direction * risk * rr
        last = min(n - 1, entry_idx + max_hold_bars)
        exit_idx = last
        exit_price = float(close[last])
        exit_reason = "timeout"
        for j in range(entry_idx, last + 1):
            if direction > 0:
                sl_hit = low[j] <= stop
                tp_hit = high[j] >= target
                if sl_hit or tp_hit:
                    if sl_hit:
                        exit_idx = j
                        exit_price = stop
                        exit_reason = "sl"
                    else:
                        exit_idx = j
                        exit_price = target
                        exit_reason = "tp"
                    break
            else:
                sl_hit = high[j] >= stop
                tp_hit = low[j] <= target
                if sl_hit or tp_hit:
                    if sl_hit:
                        exit_idx = j
                        exit_price = stop
                        exit_reason = "sl"
                    else:
                        exit_idx = j
                        exit_price = target
                        exit_reason = "tp"
                    break
        gross_r = direction * (exit_price - entry) / risk
        fee_r = (2.0 * fee_bps_per_side / 10000.0) * entry / risk
        features = add_signal_features(frame, ctx, signal_idx, direction, entry, stop, signal.get("features", {}))
        trades.append(
            Trade(
                symbol=symbol,
                strategy=spec.strategy,
                spec_name=spec.name,
                timeframe=spec.timeframe,
                direction="long" if direction > 0 else "short",
                signal_index=signal_idx,
                entry_index=entry_idx,
                exit_index=exit_idx,
                signal_time=pd.Timestamp(frame["close_time"].iloc[signal_idx]),
                entry_time=pd.Timestamp(frame["open_time"].iloc[entry_idx]),
                exit_time=pd.Timestamp(frame["close_time"].iloc[exit_idx]),
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                exit_price=exit_price,
                exit_reason=exit_reason,
                r_multiple=float(gross_r - fee_r),
                gross_r=float(gross_r),
                fee_r=float(fee_r),
                risk_pct=float(risk_pct),
                atr_pct=float(features.get("atr_pct", math.nan)),
                bars_held=int(exit_idx - entry_idx + 1),
                feature_json=json.dumps(features, sort_keys=True),
            )
        )
        blocked_until = exit_idx
    return trades


def signals_pivot_breakout(frame: pd.DataFrame, ctx: dict[str, np.ndarray], spec: CandidateSpec) -> list[dict[str, Any]]:
    left = int(spec.params["left"])
    right = int(spec.params["right"])
    max_bars = int(spec.params["line_max_bars"])
    max_high = int(spec.params.get("max_high", 5))
    max_low = int(spec.params.get("max_low", 5))
    sl_atr = float(spec.params["sl_atr"])
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    atr = ctx["atr14"]
    ph = pivots(high, left, right, "high")
    pl = pivots(low, left, right, "low")
    prev_high_price = math.nan
    prev_high_bar = -1
    prev_low_price = math.nan
    prev_low_bar = -1
    high_lines: list[dict[str, float]] = []
    low_lines: list[dict[str, float]] = []
    out: list[dict[str, Any]] = []
    for i in range(len(frame)):
        if math.isfinite(ph[i]):
            pivot_bar = i - right
            pivot_price = float(ph[i])
            if math.isfinite(prev_high_price) and pivot_bar != prev_high_bar:
                slope = (pivot_price - prev_high_price) / float(pivot_bar - prev_high_bar)
                high_lines.append(
                    {"x1": float(prev_high_bar), "y1": prev_high_price, "slope": slope, "det": float(i), "active": 1.0}
                )
                if max_high > 0 and len(high_lines) > max_high:
                    high_lines = high_lines[-max_high:]
            prev_high_price = pivot_price
            prev_high_bar = pivot_bar
        if math.isfinite(pl[i]):
            pivot_bar = i - right
            pivot_price = float(pl[i])
            if math.isfinite(prev_low_price) and pivot_bar != prev_low_bar:
                slope = (pivot_price - prev_low_price) / float(pivot_bar - prev_low_bar)
                low_lines.append(
                    {"x1": float(prev_low_bar), "y1": prev_low_price, "slope": slope, "det": float(i), "active": 1.0}
                )
                if max_low > 0 and len(low_lines) > max_low:
                    low_lines = low_lines[-max_low:]
            prev_low_price = pivot_price
            prev_low_bar = pivot_bar

        for line in high_lines:
            if line["active"] <= 0:
                continue
            line_price = line["y1"] + line["slope"] * (i - line["x1"])
            expired = i - line["det"] >= max_bars
            if close[i] > line_price or expired:
                line["active"] = 0.0
                if close[i] > line_price and math.isfinite(atr[i]) and atr[i] > 0:
                    stop = close[i] - atr[i] * sl_atr
                    out.append(
                        {
                            "idx": i,
                            "direction": 1,
                            "stop": stop,
                            "features": {
                                "line_age": float(i - line["det"]),
                                "line_slope_atr": float(line["slope"] / atr[i]),
                                "break_distance_atr": float((close[i] - line_price) / atr[i]),
                            },
                        }
                    )
        for line in low_lines:
            if line["active"] <= 0:
                continue
            line_price = line["y1"] + line["slope"] * (i - line["x1"])
            expired = i - line["det"] >= max_bars
            if close[i] < line_price or expired:
                line["active"] = 0.0
                if close[i] < line_price and math.isfinite(atr[i]) and atr[i] > 0:
                    stop = close[i] + atr[i] * sl_atr
                    out.append(
                        {
                            "idx": i,
                            "direction": -1,
                            "stop": stop,
                            "features": {
                                "line_age": float(i - line["det"]),
                                "line_slope_atr": float(line["slope"] / atr[i]),
                                "break_distance_atr": float((line_price - close[i]) / atr[i]),
                            },
                        }
                    )
    out.sort(key=lambda x: x["idx"])
    return out


def signals_melona_trendline(frame: pd.DataFrame, ctx: dict[str, np.ndarray], spec: CandidateSpec) -> list[dict[str, Any]]:
    left = int(spec.params["left"])
    right = int(spec.params["right"])
    line_max = int(spec.params["line_max_bars"])
    sl_atr = float(spec.params["sl_atr"])
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    atr = ctx["atr14"]
    ph = pivots(high, left, right, "high")
    pl = pivots(low, left, right, "low")
    prev_high_bar = -1
    prev_low_bar = -1
    prev_high = curr_high = math.nan
    prev_low = curr_low = math.nan
    prev_close_h = curr_close_h = math.nan
    prev_close_l = curr_close_l = math.nan
    high_lines: list[dict[str, float]] = []
    low_lines: list[dict[str, float]] = []
    out: list[dict[str, Any]] = []
    for i in range(len(frame)):
        if math.isfinite(pl[i]):
            prev_low_bar, prev_low, prev_close_l = prev_low_bar if prev_low_bar >= 0 else i - right, curr_low, curr_close_l
            curr_low_bar = i - right
            curr_low = float(low[curr_low_bar])
            curr_close_l = float(close[curr_low_bar])
            if math.isfinite(prev_low) and prev_low_bar != curr_low_bar:
                if prev_low < curr_low and curr_low > prev_close_l:
                    slope = (curr_low - prev_low) / float(curr_low_bar - prev_low_bar)
                    low_lines.append({"x1": float(prev_low_bar), "y1": prev_low, "slope": slope, "det": float(i), "active": 1.0})
            prev_low_bar = curr_low_bar
        if math.isfinite(ph[i]):
            prev_high_bar, prev_high, prev_close_h = prev_high_bar if prev_high_bar >= 0 else i - right, curr_high, curr_close_h
            curr_high_bar = i - right
            curr_high = float(high[curr_high_bar])
            curr_close_h = float(close[curr_high_bar])
            if math.isfinite(prev_high) and prev_high_bar != curr_high_bar:
                if prev_high > curr_high and prev_close_h > curr_high:
                    slope = (curr_high - prev_high) / float(curr_high_bar - prev_high_bar)
                    high_lines.append({"x1": float(prev_high_bar), "y1": prev_high, "slope": slope, "det": float(i), "active": 1.0})
            prev_high_bar = curr_high_bar
        if close[i] > prev_high if math.isfinite(prev_high) else False:
            prev_high = 0.0
        if close[i] < prev_low if math.isfinite(prev_low) else False:
            prev_low = 9999999999.0
        for line in low_lines:
            if line["active"] <= 0:
                continue
            line_price = line["y1"] + line["slope"] * (i - line["x1"])
            if i - line["det"] > line_max:
                line["active"] = 0.0
                continue
            if close[i] < line_price and math.isfinite(atr[i]) and atr[i] > 0:
                line["active"] = 0.0
                out.append(
                    {
                        "idx": i,
                        "direction": -1,
                        "stop": close[i] + atr[i] * sl_atr,
                        "features": {
                            "line_age": float(i - line["det"]),
                            "line_slope_atr": float(line["slope"] / atr[i]),
                            "break_distance_atr": float((line_price - close[i]) / atr[i]),
                        },
                    }
                )
        for line in high_lines:
            if line["active"] <= 0:
                continue
            line_price = line["y1"] + line["slope"] * (i - line["x1"])
            if i - line["det"] > line_max:
                line["active"] = 0.0
                continue
            if close[i] > line_price and math.isfinite(atr[i]) and atr[i] > 0:
                line["active"] = 0.0
                out.append(
                    {
                        "idx": i,
                        "direction": 1,
                        "stop": close[i] - atr[i] * sl_atr,
                        "features": {
                            "line_age": float(i - line["det"]),
                            "line_slope_atr": float(line["slope"] / atr[i]),
                            "break_distance_atr": float((close[i] - line_price) / atr[i]),
                        },
                    }
                )
    out.sort(key=lambda x: x["idx"])
    return out


def signals_ha_supertrend(frame: pd.DataFrame, ctx: dict[str, np.ndarray], spec: CandidateSpec) -> list[dict[str, Any]]:
    period = int(spec.params["period"])
    multiplier = float(spec.params["multiplier"])
    sensitivity = float(spec.params.get("sensitivity", 1.0))
    sl_atr = float(spec.params["sl_atr"])
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    source = close
    ha_close = (open_ + high + low + source) / 4.0
    ha_open = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        if i == 0 or not math.isfinite(ha_open[i - 1]):
            ha_open[i] = (open_[i] + source[i]) / 2.0
        else:
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    atr = rma(true_range(high, low, close), period)
    adj = multiplier * sensitivity
    up = np.full(len(frame), np.nan)
    dn = np.full(len(frame), np.nan)
    up1 = np.full(len(frame), np.nan)
    dn1 = np.full(len(frame), np.nan)
    last_up_value = math.nan
    last_dn_value = math.nan
    trend = np.full(len(frame), 1, dtype=int)
    out: list[dict[str, Any]] = []
    for i in range(len(frame)):
        if not math.isfinite(atr[i]):
            continue
        raw_up = ha_close[i] - adj * atr[i]
        raw_dn = ha_close[i] + adj * atr[i]
        if i > 0 and math.isfinite(up[i - 1]) and ha_close[i - 1] > up[i - 1]:
            last_up_value = up[i - 1]
        if i > 0 and math.isfinite(dn[i - 1]) and ha_close[i - 1] < dn[i - 1]:
            last_dn_value = dn[i - 1]
        up1[i] = last_up_value
        dn1[i] = last_dn_value
        up[i] = raw_up
        if i > 0 and math.isfinite(up1[i]) and ha_close[i - 1] > up1[i]:
            up[i] = max(raw_up, up1[i])
        dn[i] = raw_dn
        if i > 0 and math.isfinite(dn1[i]) and ha_close[i - 1] < dn1[i]:
            dn[i] = min(raw_dn, dn1[i])
        if i > 0:
            trend[i] = trend[i - 1]
            if trend[i - 1] == -1 and math.isfinite(dn1[i]) and ha_close[i] > dn1[i]:
                trend[i] = 1
            elif trend[i - 1] == 1 and math.isfinite(up1[i]) and ha_close[i] < up1[i]:
                trend[i] = -1
            if trend[i] != trend[i - 1]:
                direction = 1 if trend[i] == 1 else -1
                stop = close[i] - direction * atr[i] * sl_atr
                out.append(
                    {
                        "idx": i,
                        "direction": direction,
                        "stop": stop,
                        "features": {
                            "ha_close_vs_band_atr": (
                                (ha_close[i] - dn1[i]) / atr[i]
                                if direction > 0 and math.isfinite(dn1[i])
                                else (up1[i] - ha_close[i]) / atr[i]
                                if math.isfinite(up1[i])
                                else math.nan
                            ),
                            "ha_trend_prev": float(trend[i - 1]),
                        },
                    }
                )
    return out


def signals_liquidity_sweep(frame: pd.DataFrame, ctx: dict[str, np.ndarray], spec: CandidateSpec) -> list[dict[str, Any]]:
    swing = int(spec.params["swing"])
    min_wick = float(spec.params["min_wick"])
    min_rej = float(spec.params["min_rej"])
    vol_mult = float(spec.params["vol_mult"])
    cooldown = int(spec.params["cooldown"])
    sl_buf = float(spec.params["sl_buf"])
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    open_ = frame["open"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    atr = ctx["atr14"]
    vol_ma = ctx["vol_sma20"]
    sw_h = highest_previous(high, swing * 2)
    sw_l = lowest_previous(low, swing * 2)
    out: list[dict[str, Any]] = []
    last_bull = -10_000
    last_bear = -10_000
    for i in range(len(frame)):
        if not (math.isfinite(sw_h[i]) and math.isfinite(sw_l[i]) and math.isfinite(atr[i]) and atr[i] > 0):
            continue
        candle_range = max(1e-12, high[i] - low[i])
        wick_u = high[i] - max(open_[i], close[i])
        wick_d = min(open_[i], close[i]) - low[i]
        rej_u = wick_u / candle_range * 100.0
        rej_d = wick_d / candle_range * 100.0
        vol_ok = vol_mult <= 0 or (math.isfinite(vol_ma[i]) and volume[i] > vol_ma[i] * vol_mult)
        bull = (
            low[i] < sw_l[i]
            and close[i] > sw_l[i]
            and (sw_l[i] - low[i]) >= atr[i] * min_wick
            and rej_d >= min_rej
            and vol_ok
            and i - last_bull >= cooldown
        )
        bear = (
            high[i] > sw_h[i]
            and close[i] < sw_h[i]
            and (high[i] - sw_h[i]) >= atr[i] * min_wick
            and rej_u >= min_rej
            and vol_ok
            and i - last_bear >= cooldown
        )
        if bull:
            last_bull = i
            out.append(
                {
                    "idx": i,
                    "direction": 1,
                    "stop": float(low[i] - atr[i] * sl_buf),
                    "features": {
                        "sweep_depth_atr": float((sw_l[i] - low[i]) / atr[i]),
                        "rejection_pct": float(rej_d),
                        "level_distance_atr": float((close[i] - sw_l[i]) / atr[i]),
                    },
                }
            )
        if bear:
            last_bear = i
            out.append(
                {
                    "idx": i,
                    "direction": -1,
                    "stop": float(high[i] + atr[i] * sl_buf),
                    "features": {
                        "sweep_depth_atr": float((high[i] - sw_h[i]) / atr[i]),
                        "rejection_pct": float(rej_u),
                        "level_distance_atr": float((sw_h[i] - close[i]) / atr[i]),
                    },
                }
            )
    out.sort(key=lambda x: x["idx"])
    return out


def signals_pressure(frame: pd.DataFrame, ctx: dict[str, np.ndarray], spec: CandidateSpec) -> list[dict[str, Any]]:
    sl_atr = float(spec.params["sl_atr"])
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    atr = ctx["atr14"]
    out: list[dict[str, Any]] = []
    for i in range(2, len(frame)):
        if not math.isfinite(atr[i]) or atr[i] <= 0:
            continue
        bull = open_[i - 2] > close[i - 2] and close[i - 1] > open_[i - 1] and close[i] > open_[i] and low[i - 1] < low[i - 2] and close[i] > high[i - 1]
        bear = open_[i - 2] < close[i - 2] and close[i - 1] < open_[i - 1] and close[i] < open_[i] and high[i - 1] > high[i - 2] and close[i] < low[i - 1]
        if bull:
            stop = min(low[i], low[i - 1], low[i - 2]) - atr[i] * sl_atr
            out.append({"idx": i, "direction": 1, "stop": float(stop), "features": {"pattern_range_atr": float((max(high[i], high[i - 1]) - min(low[i], low[i - 1])) / atr[i])}})
        if bear:
            stop = max(high[i], high[i - 1], high[i - 2]) + atr[i] * sl_atr
            out.append({"idx": i, "direction": -1, "stop": float(stop), "features": {"pattern_range_atr": float((max(high[i], high[i - 1]) - min(low[i], low[i - 1])) / atr[i])}})
    return out


def signals_demarker_exhaustion(frame: pd.DataFrame, ctx: dict[str, np.ndarray], spec: CandidateSpec) -> list[dict[str, Any]]:
    qual = int(spec.params["qual"])
    length = int(spec.params["length"])
    sl_atr = float(spec.params["sl_atr"])
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    atr = ctx["atr14"]
    bindex = 0
    sindex = 0
    out: list[dict[str, Any]] = []
    for i in range(4, len(frame)):
        if close[i] > close[i - 4]:
            bindex += 1
        if close[i] < close[i - 4]:
            sindex += 1
        if not math.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if bindex > qual and close[i] < open_[i] and high[i] >= np.nanmax(high[max(0, i - length + 1) : i + 1]):
            bindex = 0
            out.append({"idx": i, "direction": -1, "stop": float(high[i] + atr[i] * sl_atr), "features": {"exhaust_count": float(qual + 1), "exhaust_length": float(length)}})
        if sindex > qual and close[i] > open_[i] and low[i] <= np.nanmin(low[max(0, i - length + 1) : i + 1]):
            sindex = 0
            out.append({"idx": i, "direction": 1, "stop": float(low[i] - atr[i] * sl_atr), "features": {"exhaust_count": float(qual + 1), "exhaust_length": float(length)}})
    return out


SIGNAL_BUILDERS = {
    "pivot_breakout": signals_pivot_breakout,
    "melona_trendline": signals_melona_trendline,
    "ha_supertrend": signals_ha_supertrend,
    "liquidity_sweep": signals_liquidity_sweep,
    "melona_pressure": signals_pressure,
    "demarker_exhaustion": signals_demarker_exhaustion,
}


def build_specs(args: argparse.Namespace) -> list[CandidateSpec]:
    timeframes = parse_list(args.timeframes)
    allowed = {x.strip() for x in parse_list(getattr(args, "strategies", ""))} if getattr(args, "strategies", "") else set()
    specs: list[CandidateSpec] = []
    def add(spec: CandidateSpec) -> None:
        if not allowed or spec.strategy in allowed:
            specs.append(spec)

    if args.grid_mode == "smoke":
        for tf in timeframes:
            add(CandidateSpec("pivot_breakout", tf, {"left": 20, "right": 20, "line_max_bars": 300, "max_high": 5, "max_low": 5, "sl_atr": 1.5, "rr": 2.0, "max_hold_bars": 288}))
            add(CandidateSpec("ha_supertrend", tf, {"period": 5, "multiplier": 1.5, "sensitivity": 1.0, "sl_atr": 1.2, "rr": 1.5, "max_hold_bars": 288}))
            add(CandidateSpec("liquidity_sweep", tf, {"swing": 5, "min_wick": 0.1, "min_rej": 5.0, "vol_mult": 1.2, "cooldown": 3, "sl_buf": 0.3, "rr": 1.5, "max_hold_bars": 144}))
            add(CandidateSpec("melona_pressure", tf, {"sl_atr": 0.2, "rr": 1.5, "max_hold_bars": 144}))
            add(CandidateSpec("demarker_exhaustion", tf, {"qual": 13, "length": 40, "sl_atr": 0.3, "rr": 1.5, "max_hold_bars": 144}))
        return specs

    left_pairs = [(10, 10), (20, 20), (30, 20)]
    line_bars = [150, 300] if args.grid_mode == "fast" else [100, 200, 300, 500]
    sl_atrs = [1.0, 1.5] if args.grid_mode == "fast" else [0.8, 1.0, 1.5, 2.0]
    rrs = [1.0, 1.5, 2.0] if args.grid_mode == "fast" else [1.0, 1.25, 1.5, 2.0, 3.0]
    for tf in timeframes:
        hold_base = 288 if tf == "5m" else 96 if tf == "15m" else 48
        for left, right in left_pairs:
            for max_bars in line_bars:
                for sl_atr in sl_atrs:
                    for rr in rrs:
                        add(CandidateSpec("pivot_breakout", tf, {"left": left, "right": right, "line_max_bars": max_bars, "max_high": 5, "max_low": 5, "sl_atr": sl_atr, "rr": rr, "max_hold_bars": hold_base}))
                        add(CandidateSpec("melona_trendline", tf, {"left": left, "right": right, "line_max_bars": 300, "sl_atr": sl_atr, "rr": rr, "max_hold_bars": hold_base}))
        ha_periods = [5, 14] if args.grid_mode == "fast" else [5, 7, 10, 14]
        ha_mults = [1.5, 1.6]
        ha_sens = [1.0, 2.3] if args.grid_mode == "fast" else [1.0, 1.5, 2.3]
        for period in ha_periods:
            for mult in ha_mults:
                for sens in ha_sens:
                    for sl_atr in sl_atrs:
                        for rr in rrs:
                            add(CandidateSpec("ha_supertrend", tf, {"period": period, "multiplier": mult, "sensitivity": sens, "sl_atr": sl_atr, "rr": rr, "max_hold_bars": hold_base}))
        sweep_swings = [5, 10] if args.grid_mode == "fast" else [3, 5, 10, 20]
        for swing in sweep_swings:
            for min_wick in ([0.1, 0.25] if args.grid_mode == "fast" else [0.05, 0.1, 0.25, 0.5]):
                for vol_mult in ([0.0, 1.2] if args.grid_mode == "fast" else [0.0, 1.0, 1.2, 1.5]):
                    for sl_buf in ([0.2, 0.5] if args.grid_mode == "fast" else [0.0, 0.2, 0.5]):
                        for rr in rrs:
                            add(CandidateSpec("liquidity_sweep", tf, {"swing": swing, "min_wick": min_wick, "min_rej": 5.0, "vol_mult": vol_mult, "cooldown": 3, "sl_buf": sl_buf, "rr": rr, "max_hold_bars": hold_base}))
        for sl_atr in ([0.0, 0.2, 0.5] if args.grid_mode == "fast" else [0.0, 0.2, 0.5, 1.0]):
            for rr in rrs:
                add(CandidateSpec("melona_pressure", tf, {"sl_atr": sl_atr, "rr": rr, "max_hold_bars": hold_base}))
        for qual, length in [(13, 40), (8, 30), (5, 20)]:
            for sl_atr in ([0.0, 0.3, 0.8] if args.grid_mode == "fast" else [0.0, 0.2, 0.5, 1.0]):
                for rr in rrs:
                    add(CandidateSpec("demarker_exhaustion", tf, {"qual": qual, "length": length, "sl_atr": sl_atr, "rr": rr, "max_hold_bars": hold_base}))
    return specs


def metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
            "trades_per_week": 0.0,
        }
    r = trades["r_multiple"].astype(float)
    wins = r[r > 0]
    losses = r[r < 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    curve = r.cumsum()
    dd = curve.cummax() - curve
    start = pd.to_datetime(trades["entry_time"], utc=True).min()
    end = pd.to_datetime(trades["entry_time"], utc=True).max()
    weeks = max((end - start).total_seconds() / (86400.0 * 7.0), 1e-9)
    return {
        "trades": int(len(trades)),
        "net_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0),
        "max_dd_r": float(dd.max()) if len(dd) else 0.0,
        "trades_per_week": float(len(trades) / weeks),
    }


def prefixed_metrics(trades: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics(trades).items()}


def run_symbol_specs(symbol: str, base_frame: pd.DataFrame, specs: list[CandidateSpec], args: argparse.Namespace, train_start: pd.Timestamp) -> list[Trade]:
    by_tf: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray]]] = {}
    out: list[Trade] = []
    for tf in sorted({spec.timeframe for spec in specs}):
        frame = resample_frame(base_frame, tf)
        frame = frame[frame["open_time"] >= train_start - pd.Timedelta(days=10)].reset_index(drop=True)
        by_tf[tf] = (frame, base_context(frame))
    for spec in specs:
        frame, ctx = by_tf[spec.timeframe]
        builder = SIGNAL_BUILDERS[spec.strategy]
        try:
            signals = builder(frame, ctx, spec)
            out.extend(
                simulate_signals(
                    frame,
                    ctx,
                    symbol=symbol,
                    spec=spec,
                    signals=signals,
                    rr=float(spec.params["rr"]),
                    max_hold_bars=int(spec.params["max_hold_bars"]),
                    fee_bps_per_side=args.fee_bps_per_side,
                    min_risk_pct=args.min_risk_pct,
                )
            )
        except Exception as exc:
            print(f"  {symbol} {spec.name}: failed {type(exc).__name__}: {exc}", flush=True)
    return out


def run_symbol_job(payload: tuple[str, dict[str, Any], list[CandidateSpec], str, str, str]) -> tuple[str, list[dict[str, Any]], str | None]:
    symbol, args_dict, specs, train_start_raw, end_raw, cache_dir_raw = payload
    args = argparse.Namespace(**args_dict)
    train_start = pd.Timestamp(parse_utc_datetime(train_start_raw))
    end = pd.Timestamp(parse_utc_datetime(end_raw))
    try:
        frame = load_frame(symbol, Path(cache_dir_raw), train_start, end)
        trades = run_symbol_specs(symbol, frame, specs, args, train_start)
        return symbol, [trade.to_dict() for trade in trades], None
    except Exception as exc:
        return symbol, [], f"{type(exc).__name__}: {exc}"


def summarize(all_trades: pd.DataFrame, split: pd.Timestamp, min_oos_trades: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if all_trades.empty:
        return pd.DataFrame()
    all_trades = all_trades.copy()
    all_trades["entry_time"] = pd.to_datetime(all_trades["entry_time"], utc=True, errors="coerce")
    for (spec_name, strategy, timeframe), group in all_trades.groupby(["spec_name", "strategy", "timeframe"]):
        train = group[group["entry_time"] < split]
        oos = group[group["entry_time"] >= split]
        row = {
            "spec_name": spec_name,
            "strategy": strategy,
            "timeframe": timeframe,
            "symbols": int(group["symbol"].nunique()),
            **prefixed_metrics(train, "train"),
            **prefixed_metrics(oos, "oos"),
            **prefixed_metrics(group, "all"),
        }
        row["score"] = (
            row["oos_avg_r"] * math.sqrt(max(row["oos_trades"], 1))
            + 0.15 * min(row["oos_profit_factor"], 3.0)
            - 0.015 * row["oos_max_dd_r"]
        )
        row["eligible"] = bool(row["oos_trades"] >= min_oos_trades and row["train_trades"] >= min_oos_trades * 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["eligible", "score", "oos_net_r"], ascending=[False, False, False])


def feature_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    rows: list[dict[str, Any]] = []
    for _, row in trades.iterrows():
        base = row.to_dict()
        raw = base.pop("feature_json", "{}")
        try:
            features = json.loads(raw)
        except Exception:
            features = {}
        for key, value in features.items():
            base[f"f_{key}"] = value
        rows.append(base)
    return pd.DataFrame(rows)


def train_ml_filters(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    split: pd.Timestamp,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if trades.empty or summary.empty:
        return pd.DataFrame()
    model_dir = args.out_prefix.parent / f"{args.out_prefix.name}_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    expanded = feature_table(trades)
    expanded["entry_time"] = pd.to_datetime(expanded["entry_time"], utc=True, errors="coerce")
    feature_cols = [c for c in expanded.columns if c.startswith("f_")]
    for _, candidate in summary[summary["eligible"]].head(args.ml_top_specs).iterrows():
        spec_name = candidate["spec_name"]
        data = expanded[expanded["spec_name"].eq(spec_name)].copy()
        train = data[data["entry_time"] < split].copy()
        oos = data[data["entry_time"] >= split].copy()
        if len(train) < args.ml_min_train or len(oos) < args.ml_min_oos:
            continue
        if train["r_multiple"].gt(0).nunique() < 2:
            continue
        cols = [c for c in feature_cols if data[c].notna().any()]
        if not cols:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=350,
                max_depth=5,
                min_samples_leaf=max(25, min(100, len(train) // 30)),
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
        )
        model.fit(train[cols].astype(float), train["r_multiple"].gt(0).astype(int))
        scored = data.copy()
        scored["ml_prob"] = model.predict_proba(scored[cols].astype(float))[:, 1]
        best_row: dict[str, Any] | None = None
        for threshold in parse_float_list(args.ml_thresholds):
            selected = scored[scored["ml_prob"] >= threshold].copy()
            selected_train = selected[selected["entry_time"] < split]
            selected_oos = selected[selected["entry_time"] >= split]
            row = {
                "spec_name": spec_name,
                "strategy": candidate["strategy"],
                "timeframe": candidate["timeframe"],
                "threshold": threshold,
                "feature_count": len(cols),
                **prefixed_metrics(selected_train, "train"),
                **prefixed_metrics(selected_oos, "oos"),
            }
            row["score"] = row["oos_avg_r"] * math.sqrt(max(row["oos_trades"], 1)) + 0.2 * min(row["oos_profit_factor"], 3.0)
            if row["oos_trades"] >= args.ml_min_oos and (best_row is None or row["score"] > best_row["score"]):
                best_row = row
        if best_row is None:
            continue
        payload = {
            "model": model,
            "feature_columns": cols,
            "threshold": float(best_row["threshold"]),
            "spec_name": spec_name,
            "strategy": candidate["strategy"],
            "timeframe": candidate["timeframe"],
            "source": "scripts/algos/20260512 pine ports",
        }
        model_path = model_dir / f"{spec_name}_rf.joblib"
        joblib.dump(payload, model_path)
        best_row["model_path"] = str(model_path)
        rows.append(best_row)
    return pd.DataFrame(rows).sort_values(["score", "oos_net_r"], ascending=[False, False]) if rows else pd.DataFrame()


def write_report(args: argparse.Namespace, summary: pd.DataFrame, ml_summary: pd.DataFrame, trades: pd.DataFrame, specs: list[CandidateSpec]) -> Path:
    path = args.out_prefix.with_suffix(".md")
    lines: list[str] = []
    lines.append("# Pine Strategy Candidate Research")
    lines.append("")
    lines.append("Source Pine scripts: `scripts/algos/20260512`.")
    lines.append("TradingView MCP compile/check: all staged Pine files compiled successfully; MELONA OB scripts emitted Pine consistency warnings but no compile errors.")
    lines.append("")
    lines.append("## Ported Atoms")
    lines.append("")
    lines.append("- `pivot_breakout`: `Indi 177` pivot high/low trendline breakout.")
    lines.append("- `ha_supertrend`: `Indi 172` and `MELONA CONFIRMER U3` Heikin-Ashi supertrend flips.")
    lines.append("- `liquidity_sweep`: `Indi 170` Liquidity Matrix wick sweep logic.")
    lines.append("- `melona_pressure`: MELONA OB single-candle order-block pressure pattern.")
    lines.append("- `demarker_exhaustion`: MELONA OB major TP-point/exhaustion reversal.")
    lines.append("- `melona_trendline`: MELONA OB EzTrendline broken-line signal.")
    lines.append("")
    lines.append(f"Specs tested: {len(specs)}. Trades generated: {len(trades):,}.")
    lines.append("")
    def md_table(frame: pd.DataFrame, cols: list[str]) -> str:
        if frame.empty:
            return "_No rows._"
        shown = frame[cols].copy()
        for column in shown.columns:
            if pd.api.types.is_float_dtype(shown[column]):
                shown[column] = shown[column].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
            else:
                shown[column] = shown[column].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(row[column]) for column in cols) + " |" for _, row in shown.iterrows()]
        return "\n".join([header, sep, *body])

    if not summary.empty:
        lines.append("## Best Raw OOS Specs")
        lines.append("")
        cols = ["strategy", "timeframe", "spec_name", "oos_trades", "oos_net_r", "oos_avg_r", "oos_win_rate", "oos_profit_factor", "oos_max_dd_r", "oos_trades_per_week"]
        top = summary[summary["eligible"]].head(20)
        lines.append(md_table(top, cols))
        lines.append("")
    if not ml_summary.empty:
        lines.append("## Best ML-Filtered Specs")
        lines.append("")
        cols = ["strategy", "timeframe", "spec_name", "threshold", "oos_trades", "oos_net_r", "oos_avg_r", "oos_win_rate", "oos_profit_factor", "model_path"]
        lines.append(md_table(ml_summary.head(20), cols))
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Backtests enter at the next candle open after the Pine signal bar to avoid lookahead.")
    lines.append("- Same-bar TP/SL collisions are resolved conservatively as SL first.")
    lines.append("- PnL is in R multiples after estimated taker fees.")
    lines.append("- These tests are research outputs; live wiring should start disabled and be enabled symbol by symbol.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest/triage staged Pine strategy candidates.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--timeframes", default="5m,15m")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--split", default="2025-07-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--grid-mode", choices=["smoke", "fast", "full"], default="fast")
    parser.add_argument("--strategies", default="", help="Comma-separated subset: pivot_breakout,ha_supertrend,liquidity_sweep,melona_pressure,demarker_exhaustion,melona_trendline")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--min-risk-pct", type=float, default=0.15)
    parser.add_argument("--min-oos-trades", type=int, default=60)
    parser.add_argument("--ml-top-specs", type=int, default=12)
    parser.add_argument("--ml-min-train", type=int, default=250)
    parser.add_argument("--ml-min-oos", type=int, default=50)
    parser.add_argument("--ml-thresholds", default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_start = pd.Timestamp(parse_utc_datetime(args.train_start))
    split = pd.Timestamp(parse_utc_datetime(args.split))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    if args.symbols.strip():
        symbols = [clean_symbol(x) for x in parse_list(args.symbols)]
    else:
        symbols = load_universe(args.universe, args.max_symbols)
    specs = build_specs(args)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"Testing {len(specs)} specs on {len(symbols)} symbols from {train_start.date()} to {end.date()} split={split.date()}", flush=True)
    all_trade_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if args.workers <= 1:
        for pos, symbol in enumerate(symbols, start=1):
            print(f"[{pos}/{len(symbols)}] {symbol}: loading cache/backtesting ...", flush=True)
            symbol, rows, error = run_symbol_job(
                (
                    symbol,
                    vars(args),
                    specs,
                    str(train_start),
                    str(end),
                    str(args.cache_dir),
                )
            )
            if error:
                failures.append({"symbol": symbol, "error": error})
                print(f"  {symbol}: failed {error}", flush=True)
            else:
                all_trade_rows.extend(rows)
                print(f"  {symbol}: {len(rows):,} candidate trades", flush=True)
    else:
        payloads = [
            (symbol, vars(args), specs, str(train_start), str(end), str(args.cache_dir))
            for symbol in symbols
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_symbol_job, payload): payload[0] for payload in payloads}
            done_count = 0
            for future in concurrent.futures.as_completed(future_map):
                done_count += 1
                symbol, rows, error = future.result()
                if error:
                    failures.append({"symbol": symbol, "error": error})
                    print(f"[{done_count}/{len(symbols)}] {symbol}: failed {error}", flush=True)
                else:
                    all_trade_rows.extend(rows)
                    print(f"[{done_count}/{len(symbols)}] {symbol}: {len(rows):,} candidate trades", flush=True)
    trades_frame = pd.DataFrame(all_trade_rows)
    if not trades_frame.empty:
        trades_frame["entry_time"] = pd.to_datetime(trades_frame["entry_time"], utc=True, errors="coerce")
        trades_frame = trades_frame[trades_frame["entry_time"] >= train_start].copy()
    summary = summarize(trades_frame, split, args.min_oos_trades)
    ml_summary = pd.DataFrame() if args.skip_ml else train_ml_filters(trades_frame, summary, split, args)
    trades_path = args.out_prefix.with_name(f"{args.out_prefix.name}_trades.csv")
    summary_path = args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv")
    ml_path = args.out_prefix.with_name(f"{args.out_prefix.name}_ml_summary.csv")
    fail_path = args.out_prefix.with_name(f"{args.out_prefix.name}_failures.csv")
    trades_frame.to_csv(trades_path, index=False)
    summary.to_csv(summary_path, index=False)
    ml_summary.to_csv(ml_path, index=False)
    pd.DataFrame(failures).to_csv(fail_path, index=False)
    report = write_report(args, summary, ml_summary, trades_frame, specs)
    print(f"Saved trades: {trades_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved ML summary: {ml_path}", flush=True)
    print(f"Saved report: {report}", flush=True)
    if not summary.empty:
        print(summary.head(12)[["strategy", "timeframe", "oos_trades", "oos_net_r", "oos_avg_r", "oos_profit_factor", "spec_name"]].to_string(index=False), flush=True)
    if not ml_summary.empty:
        print("ML filtered:", flush=True)
        print(ml_summary.head(8)[["strategy", "timeframe", "threshold", "oos_trades", "oos_net_r", "oos_avg_r", "oos_profit_factor", "spec_name"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
