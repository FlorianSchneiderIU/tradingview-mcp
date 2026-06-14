from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import (  # noqa: E402
    add_indicators,
    ensure_ohlcv_frame,
    fetch_bybit_klines,
    normalize_timeframe,
    resample_ohlc,
)

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


UTC = timezone.utc
DEFAULT_CACHE_DIR = Path("scripts/.cache/astro_cycle")
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class PivotEvent:
    index: int
    time: pd.Timestamp
    kind: str
    price: float
    threshold_atr: float


@dataclass(frozen=True)
class SplitSpec:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def parse_utc_datetime(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    if len(text) == 10:
        text = f"{text}T00:00:00+00:00"
    out = datetime.fromisoformat(text)
    if out.tzinfo is None:
        out = out.replace(tzinfo=UTC)
    return out.astimezone(UTC)


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def safe_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(":", "_").replace("/", "").replace("-", "").replace(".", "")


def load_bybit_cached(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
) -> pd.DataFrame:
    interval = normalize_timeframe(interval)
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbol_key = safe_symbol(symbol)
    exact = cache_dir / f"bybit_{symbol_key}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
    if exact.exists():
        frame = pd.read_pickle(exact)
        return ensure_ohlcv_frame(frame)

    # Reuse a broader cache if one already covers the request.
    for candidate in sorted(cache_dir.glob(f"bybit_{symbol_key}_{interval}_*.pkl")):
        try:
            frame = ensure_ohlcv_frame(pd.read_pickle(candidate))
        except Exception:
            continue
        if frame.empty:
            continue
        first = pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC")
        last = pd.Timestamp(frame["close_time"].iloc[-1]).tz_convert("UTC")
        if first <= pd.Timestamp(start) and last >= pd.Timestamp(end):
            mask = (frame["open_time"] >= pd.Timestamp(start)) & (frame["close_time"] <= pd.Timestamp(end))
            return frame.loc[mask].reset_index(drop=True)

    frame = fetch_bybit_klines(symbol, interval, start, end)
    frame.to_pickle(exact)
    return ensure_ohlcv_frame(frame)


def prepare_frame(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    normalized = normalize_timeframe(timeframe)
    if normalized != normalize_timeframe(infer_interval(frame)):
        frame = resample_ohlc(frame, normalized)
    out = add_indicators(ensure_ohlcv_frame(frame), 14, 200, 14)
    out["bar_index"] = np.arange(len(out), dtype=int)
    return out.reset_index(drop=True)


def infer_interval(frame: pd.DataFrame) -> str:
    times = pd.to_datetime(frame["open_time"], utc=True).sort_values()
    if len(times) < 2:
        return "15m"
    seconds = times.diff().dropna().dt.total_seconds()
    median = float(seconds[seconds > 0].median())
    known = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    return min(known, key=lambda item: abs(known[item] - median))


def zigzag_pivots(frame: pd.DataFrame, threshold_atr: float) -> list[PivotEvent]:
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    atr = frame["atr"].to_numpy(dtype=float)
    times = pd.to_datetime(frame["close_time"], utc=True)
    pivots: list[PivotEvent] = []

    valid = np.flatnonzero(np.isfinite(atr) & (atr > 0))
    if len(valid) < 20:
        return pivots
    start = int(valid[0])

    seeking = "high"
    high_idx: int | None = start
    low_idx: int | None = start
    high_price = float(highs[start])
    low_price = float(lows[start])
    last_pivot_idx = -1

    for idx in range(start + 1, len(frame)):
        current_atr = float(atr[idx])
        if not math.isfinite(current_atr) or current_atr <= 0:
            continue

        if seeking == "high":
            if high_idx is None:
                high_idx = idx
                high_price = float(highs[idx])
            if highs[idx] >= high_price:
                high_price = float(highs[idx])
                high_idx = idx
            reversal_size = high_price - float(lows[idx])
            if high_idx > last_pivot_idx and reversal_size >= threshold_atr * current_atr:
                pivots.append(
                    PivotEvent(
                        index=int(high_idx),
                        time=pd.Timestamp(times.iloc[high_idx]).tz_convert("UTC"),
                        kind="high",
                        price=float(high_price),
                        threshold_atr=float(reversal_size / current_atr),
                    )
                )
                last_pivot_idx = int(high_idx)
                seeking = "low"
                low_idx = None
                low_price = math.inf
                continue

        if seeking == "low":
            if low_idx is None:
                low_idx = idx
                low_price = float(lows[idx])
            if lows[idx] <= low_price:
                low_price = float(lows[idx])
                low_idx = idx
            reversal_size = float(highs[idx]) - low_price
            if low_idx > last_pivot_idx and reversal_size >= threshold_atr * current_atr:
                pivots.append(
                    PivotEvent(
                        index=int(low_idx),
                        time=pd.Timestamp(times.iloc[low_idx]).tz_convert("UTC"),
                        kind="low",
                        price=float(low_price),
                        threshold_atr=float(reversal_size / current_atr),
                    )
                )
                last_pivot_idx = int(low_idx)
                seeking = "high"
                high_idx = None
                high_price = -math.inf
                continue

    pivots = sorted({(p.index, p.kind): p for p in pivots}.values(), key=lambda item: item.index)
    return pivots


def make_forward_labels(
    frame: pd.DataFrame,
    pivots: list[PivotEvent],
    horizon_bars: int,
) -> pd.DataFrame:
    labels = pd.DataFrame(
        {
            "open_time": pd.to_datetime(frame["open_time"], utc=True),
            "close_time": pd.to_datetime(frame["close_time"], utc=True),
            "y_any": np.zeros(len(frame), dtype=int),
            "y_high": np.zeros(len(frame), dtype=int),
            "y_low": np.zeros(len(frame), dtype=int),
            "pivot_here": np.zeros(len(frame), dtype=int),
        }
    )
    for pivot in pivots:
        pivot_idx = int(pivot.index)
        start = max(0, pivot_idx - horizon_bars + 1)
        stop = min(len(labels), pivot_idx + 1)
        labels.loc[start: stop - 1, "y_any"] = 1
        labels.loc[start: stop - 1, f"y_{pivot.kind}"] = 1
        labels.loc[pivot_idx, "pivot_here"] = 1
    return labels


def calendar_features(times: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(times, utc=True)
    hour = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0
    dow = ts.dt.dayofweek.to_numpy(dtype=float)
    dom = (ts.dt.day.to_numpy(dtype=float) - 1.0) / 31.0
    month = (ts.dt.month.to_numpy(dtype=float) - 1.0)
    out = pd.DataFrame(index=ts.index)
    for name, value, period in [
        ("tod", hour, 24.0),
        ("dow", dow, 7.0),
        ("month", month, 12.0),
        ("dom", dom, 1.0),
    ]:
        angle = 2.0 * np.pi * value / period
        out[f"cal_{name}_sin"] = np.sin(angle)
        out[f"cal_{name}_cos"] = np.cos(angle)
    out["cal_is_weekend"] = (dow >= 5).astype(float)
    return out


def fixed_cycle_features(times: pd.Series, harmonics: int = 4) -> pd.DataFrame:
    ts = pd.to_datetime(times, utc=True)
    epoch = pd.Timestamp("2000-01-06T18:14:00Z")
    days = (ts - epoch).dt.total_seconds().to_numpy(dtype=float) / SECONDS_PER_DAY
    periods = {
        "moon_synodic": 29.530588,
        "moon_draconic": 27.212221,
        "moon_anomalistic": 27.55455,
        "mercury_synodic": 115.88,
        "venus_synodic": 583.92,
        "mars_synodic": 779.94,
        "jupiter_synodic": 398.88,
        "saturn_synodic": 378.09,
        "jupiter_saturn": 7253.46,
        "btc_halving_like": 1458.0,
    }
    out = pd.DataFrame(index=ts.index)
    for name, period in periods.items():
        for k in range(1, harmonics + 1):
            angle = 2.0 * np.pi * k * days / period
            out[f"cyc_{name}_h{k}_sin"] = np.sin(angle)
            out[f"cyc_{name}_h{k}_cos"] = np.cos(angle)
    return out


def _skyfield_available() -> bool:
    try:
        import skyfield.api  # noqa: F401

        return True
    except Exception:
        return False


def astro_cache_path(times: pd.Series, cache_dir: Path, timeframe: str) -> Path:
    ts = pd.to_datetime(times, utc=True)
    start = pd.Timestamp(ts.iloc[0]).strftime("%Y%m%d%H%M")
    end = pd.Timestamp(ts.iloc[-1]).strftime("%Y%m%d%H%M")
    return cache_dir / f"skyfield_geo_{normalize_timeframe(timeframe)}_{start}_{end}.pkl"


def compute_skyfield_positions(times: pd.Series, cache_dir: Path, timeframe: str) -> pd.DataFrame:
    if not _skyfield_available():
        return pd.DataFrame(index=times.index)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = astro_cache_path(times, cache_dir, timeframe)
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if len(cached) == len(times):
            return cached.reset_index(drop=True)

    from skyfield.api import Loader

    ts = pd.to_datetime(times, utc=True).reset_index(drop=True)
    py_datetimes = [item.to_pydatetime() for item in ts]
    loader = Loader(str(cache_dir / "skyfield"))
    eph = loader("de421.bsp")
    timescale = loader.timescale()
    earth = eph["earth"]
    bodies = {
        "sun": "sun",
        "moon": "moon",
        "mercury": "mercury",
        "venus": "venus",
        "mars": "mars",
        "jupiter": "jupiter barycenter",
        "saturn": "saturn barycenter",
        "uranus": "uranus barycenter",
        "neptune": "neptune barycenter",
        "pluto": "pluto barycenter",
    }

    rows: dict[str, np.ndarray] = {}
    chunk_size = 20_000
    for body_key in bodies:
        rows[f"{body_key}_lon"] = np.full(len(ts), np.nan, dtype=float)
        rows[f"{body_key}_lat"] = np.full(len(ts), np.nan, dtype=float)
        rows[f"{body_key}_dec"] = np.full(len(ts), np.nan, dtype=float)
        rows[f"{body_key}_dist"] = np.full(len(ts), np.nan, dtype=float)

    for start in range(0, len(ts), chunk_size):
        stop = min(len(ts), start + chunk_size)
        sky_t = timescale.from_datetimes(py_datetimes[start:stop])
        observer = earth.at(sky_t)
        for body_key, body_name in bodies.items():
            apparent = observer.observe(eph[body_name]).apparent()
            lat, lon, distance = apparent.ecliptic_latlon()
            _, dec, _ = apparent.radec()
            rows[f"{body_key}_lon"][start:stop] = lon.degrees % 360.0
            rows[f"{body_key}_lat"][start:stop] = lat.degrees
            rows[f"{body_key}_dec"][start:stop] = dec.degrees
            rows[f"{body_key}_dist"][start:stop] = distance.au

    raw = pd.DataFrame(rows)
    raw.to_pickle(cache_path)
    return raw


def angular_distance_to_aspect(angle: np.ndarray, aspect: float) -> np.ndarray:
    diff = (angle - aspect + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def astro_features(
    times: pd.Series,
    cache_dir: Path,
    timeframe: str,
    body_harmonics: int = 4,
    pair_harmonics: int = 4,
) -> pd.DataFrame:
    raw = compute_skyfield_positions(times, cache_dir, timeframe)
    if raw.empty:
        return pd.DataFrame(index=times.index)

    out = pd.DataFrame(index=times.index)
    bodies = [column[:-4] for column in raw.columns if column.endswith("_lon")]
    day_values = (
        (pd.to_datetime(times, utc=True) - pd.Timestamp("2000-01-01T00:00:00Z"))
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / SECONDS_PER_DAY
    )

    for body in bodies:
        lon = raw[f"{body}_lon"].to_numpy(dtype=float)
        lon_rad = np.deg2rad(lon)
        for k in range(1, body_harmonics + 1):
            out[f"astro_{body}_lon_h{k}_sin"] = np.sin(k * lon_rad)
            out[f"astro_{body}_lon_h{k}_cos"] = np.cos(k * lon_rad)
        for coord in ["lat", "dec"]:
            values = raw[f"{body}_{coord}"].to_numpy(dtype=float)
            out[f"astro_{body}_{coord}_sin"] = np.sin(np.deg2rad(values))
            out[f"astro_{body}_{coord}_cos"] = np.cos(np.deg2rad(values))
        distance = raw[f"{body}_dist"].to_numpy(dtype=float)
        denom = np.nanstd(distance)
        if math.isfinite(denom) and denom > 0:
            out[f"astro_{body}_dist_z"] = (distance - np.nanmean(distance)) / denom
        unwrapped = np.unwrap(lon_rad)
        speed = np.gradient(unwrapped, day_values) * 180.0 / np.pi
        out[f"astro_{body}_speed"] = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
        if body not in {"sun", "moon"}:
            out[f"astro_{body}_retrograde"] = (speed < 0.0).astype(float)

    if {"moon", "sun"}.issubset(set(bodies)):
        phase = (raw["moon_lon"].to_numpy(dtype=float) - raw["sun_lon"].to_numpy(dtype=float)) % 360.0
        phase_rad = np.deg2rad(phase)
        for k in range(1, 9):
            out[f"astro_moon_phase_h{k}_sin"] = np.sin(k * phase_rad)
            out[f"astro_moon_phase_h{k}_cos"] = np.cos(k * phase_rad)
        for aspect in [0.0, 90.0, 180.0, 270.0]:
            out[f"astro_moon_phase_orb_{int(aspect)}"] = angular_distance_to_aspect(phase, aspect)

    major_aspects = np.array([0.0, 60.0, 90.0, 120.0, 180.0], dtype=float)
    cluster_3 = np.zeros(len(raw), dtype=float)
    cluster_5 = np.zeros(len(raw), dtype=float)
    for left, right in combinations(bodies, 2):
        angle = (raw[f"{left}_lon"].to_numpy(dtype=float) - raw[f"{right}_lon"].to_numpy(dtype=float)) % 360.0
        angle_rad = np.deg2rad(angle)
        for k in range(1, pair_harmonics + 1):
            out[f"astro_pair_{left}_{right}_h{k}_sin"] = np.sin(k * angle_rad)
            out[f"astro_pair_{left}_{right}_h{k}_cos"] = np.cos(k * angle_rad)
        min_orb = np.min(np.vstack([angular_distance_to_aspect(angle, aspect) for aspect in major_aspects]), axis=0)
        cluster_3 += (min_orb <= 3.0).astype(float)
        cluster_5 += (min_orb <= 5.0).astype(float)

    out["astro_major_aspect_cluster_orb3"] = cluster_3
    out["astro_major_aspect_cluster_orb5"] = cluster_5
    return out


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.replace([np.inf, -np.inf], np.nan).copy()
    medians = out.median(numeric_only=True)
    out = out.fillna(medians).fillna(0.0)
    return out.astype(float)


def build_feature_matrix(times: pd.Series, cache_dir: Path, timeframe: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    calendar = calendar_features(times).reset_index(drop=True)
    cycles = fixed_cycle_features(times).reset_index(drop=True)
    astro = astro_features(times, cache_dir, timeframe).reset_index(drop=True)

    groups["calendar"] = list(calendar.columns)
    groups["cycles"] = list(cycles.columns)
    groups["astro"] = list(astro.columns)
    features = pd.concat([calendar, cycles, astro], axis=1)
    features = clean_features(features)
    groups["astro_cycle"] = groups["cycles"] + groups["astro"]
    groups["all"] = list(features.columns)
    return features, groups


def split_masks(times: pd.Series, spec: SplitSpec) -> dict[str, np.ndarray]:
    ts = pd.to_datetime(times, utc=True)
    return {
        "train": (ts <= spec.train_end).to_numpy(),
        "validation": ((ts > spec.train_end) & (ts <= spec.validation_end)).to_numpy(),
        "test": (ts > spec.validation_end).to_numpy(),
    }


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=0.03,
                    max_iter=2_000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def event_recall_for_active(
    active: np.ndarray,
    pivot_indices: list[int],
    eligible_mask: np.ndarray,
    horizon_bars: int,
) -> float:
    eligible_positions = np.flatnonzero(eligible_mask)
    eligible_set = set(int(item) for item in eligible_positions)
    active_set = set(int(item) for item in np.flatnonzero(active))
    relevant = [idx for idx in pivot_indices if idx in eligible_set]
    if not relevant:
        return float("nan")
    hits = 0
    for pivot_idx in relevant:
        start = max(0, pivot_idx - horizon_bars + 1)
        if any(item in active_set for item in range(start, pivot_idx + 1)):
            hits += 1
    return float(hits / len(relevant))


def precision_at_coverages(
    y_true: np.ndarray,
    y_score: np.ndarray,
    full_active_base: np.ndarray,
    full_indices: np.ndarray,
    pivot_indices: list[int],
    eligible_mask: np.ndarray,
    horizon_bars: int,
    coverages: list[float],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    base_rate = float(np.mean(y_true)) if len(y_true) else float("nan")
    for coverage in coverages:
        if len(y_score) == 0:
            continue
        cutoff = np.quantile(y_score, 1.0 - coverage)
        active_local = y_score >= cutoff
        full_active = full_active_base.copy()
        full_active[full_indices] = active_local
        precision = float(np.mean(y_true[active_local])) if np.any(active_local) else float("nan")
        out[f"top_{coverage:.3f}"] = {
            "coverage": float(np.mean(active_local)),
            "threshold": float(cutoff),
            "active_bars": int(np.sum(active_local)),
            "precision": precision,
            "lift": float(precision / base_rate) if base_rate > 0 and math.isfinite(precision) else float("nan"),
            "event_recall": event_recall_for_active(
                full_active,
                pivot_indices,
                eligible_mask,
                horizon_bars,
            ),
        }
    return out


def evaluate_feature_set(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    y_column: str,
    columns: list[str],
    masks: dict[str, np.ndarray],
    pivots: list[PivotEvent],
    horizon_bars: int,
    coverages: list[float],
) -> tuple[dict[str, Any], np.ndarray]:
    y = labels[y_column].to_numpy(dtype=int)
    train_mask = masks["train"]
    validation_mask = masks["validation"]
    test_mask = masks["test"]
    X = features.loc[:, columns]
    model = make_model()
    model.fit(X.loc[train_mask], y[train_mask])

    scores = np.full(len(labels), np.nan, dtype=float)
    for name, mask in masks.items():
        if np.any(mask):
            scores[mask] = model.predict_proba(X.loc[mask])[:, 1]

    pivot_indices = [int(p.index) for p in pivots]
    metrics: dict[str, Any] = {
        "feature_count": len(columns),
        "target": y_column,
    }
    for split_name, mask in masks.items():
        split_y = y[mask]
        split_score = scores[mask]
        split_indices = np.flatnonzero(mask)
        full_active_base = np.zeros(len(labels), dtype=bool)
        metrics[split_name] = {
            "bars": int(np.sum(mask)),
            "base_rate": float(np.mean(split_y)) if len(split_y) else float("nan"),
            "average_precision": safe_average_precision(split_y, split_score),
            "roc_auc": safe_roc_auc(split_y, split_score),
            "brier": float(brier_score_loss(split_y, split_score)) if len(np.unique(split_y)) > 1 else float("nan"),
            "precision_at": precision_at_coverages(
                split_y,
                split_score,
                full_active_base,
                split_indices,
                pivot_indices,
                mask,
                horizon_bars,
                coverages,
            ),
        }
    return metrics, scores


def shifted_placebo(features: pd.DataFrame, shift_columns: list[str], shift_bars: int) -> pd.DataFrame:
    out = features.copy()
    if not shift_columns:
        return out
    shifted = out.loc[:, shift_columns].shift(shift_bars)
    out.loc[:, shift_columns] = shifted.fillna(0.0)
    return out


def summarize_pivots(frame: pd.DataFrame, pivots: list[PivotEvent], horizon_bars: int) -> dict[str, Any]:
    if len(frame) == 0:
        return {}
    kinds = pd.Series([pivot.kind for pivot in pivots], dtype="string")
    span_days = (
        pd.Timestamp(frame["close_time"].iloc[-1]).tz_convert("UTC")
        - pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC")
    ).total_seconds() / SECONDS_PER_DAY
    return {
        "pivots": len(pivots),
        "pivot_highs": int((kinds == "high").sum()) if not kinds.empty else 0,
        "pivot_lows": int((kinds == "low").sum()) if not kinds.empty else 0,
        "pivots_per_day": float(len(pivots) / span_days) if span_days > 0 else float("nan"),
        "horizon_bars": int(horizon_bars),
    }


def select_threshold_from_validation(
    scores: np.ndarray,
    validation_mask: np.ndarray,
    coverage: float,
) -> float:
    validation_scores = scores[validation_mask]
    validation_scores = validation_scores[np.isfinite(validation_scores)]
    if len(validation_scores) == 0:
        return float("nan")
    return float(np.quantile(validation_scores, 1.0 - coverage))


def add_ltf_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(ensure_ohlcv_frame(frame), 14, 200, 14)
    out["prev_range_high"] = out["high"].shift(1).rolling(12, min_periods=12).max()
    out["prev_range_low"] = out["low"].shift(1).rolling(12, min_periods=12).min()
    return out.reset_index(drop=True)


def max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    equity = np.cumsum(np.asarray(values, dtype=float))
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity - peaks))


def profit_factor(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    gains = float(arr[arr > 0].sum())
    losses = float(arr[arr < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / abs(losses))


def simulate_sweep_trade(
    frame: pd.DataFrame,
    signal_idx: int,
    direction: str,
    rr: float,
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> dict[str, Any] | None:
    entry_idx = signal_idx + 1
    if entry_idx >= len(frame):
        return None
    signal = frame.iloc[signal_idx]
    entry = float(frame["open"].iloc[entry_idx])
    atr = float(signal["atr"])
    if not math.isfinite(atr) or atr <= 0:
        return None
    if direction == "long":
        stop = float(signal["low"] - stop_buffer_atr * atr)
        risk = entry - stop
        target = entry + rr * risk
    else:
        stop = float(signal["high"] + stop_buffer_atr * atr)
        risk = stop - entry
        target = entry - rr * risk
    if not math.isfinite(risk) or risk <= 0:
        return None

    cost_r = (cost_bps_round_trip / 10_000.0) * entry / risk
    end_idx = min(len(frame) - 1, entry_idx + max_hold_bars)
    result_r = 0.0
    exit_reason = "timeout"
    exit_idx = end_idx
    exit_price = float(frame["close"].iloc[end_idx])
    mfe_r = 0.0
    mae_r = 0.0

    for cursor in range(entry_idx, end_idx + 1):
        high = float(frame["high"].iloc[cursor])
        low = float(frame["low"].iloc[cursor])
        if direction == "long":
            mfe_r = max(mfe_r, (high - entry) / risk)
            mae_r = max(mae_r, (entry - low) / risk)
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            mfe_r = max(mfe_r, (entry - low) / risk)
            mae_r = max(mae_r, (high - entry) / risk)
            hit_stop = high >= stop
            hit_target = low <= target
        if hit_stop:
            result_r = -1.0 - cost_r
            exit_reason = "stop"
            exit_idx = cursor
            exit_price = stop
            break
        if hit_target:
            result_r = rr - cost_r
            exit_reason = "target"
            exit_idx = cursor
            exit_price = target
            break

    if exit_reason == "timeout":
        if direction == "long":
            result_r = (exit_price - entry) / risk - cost_r
        else:
            result_r = (entry - exit_price) / risk - cost_r

    return {
        "exit_idx": int(exit_idx),
        "signal_time": pd.Timestamp(frame["close_time"].iloc[signal_idx]).tz_convert("UTC"),
        "entry_time": pd.Timestamp(frame["open_time"].iloc[entry_idx]).tz_convert("UTC"),
        "exit_time": pd.Timestamp(frame["close_time"].iloc[exit_idx]).tz_convert("UTC"),
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk_pct": risk / entry,
        "result_r": float(result_r),
        "exit_reason": exit_reason,
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "hold_bars": int(exit_idx - entry_idx + 1),
    }


def trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(trade["result_r"]) for trade in trades]
    if not values:
        return {
            "trades": 0,
            "win_rate": float("nan"),
            "avg_r": float("nan"),
            "net_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "targets": 0,
            "stops": 0,
            "timeouts": 0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "trades": len(values),
        "win_rate": float(np.mean(arr > 0)),
        "avg_r": float(np.mean(arr)),
        "median_r": float(np.median(arr)),
        "net_r": float(np.sum(arr)),
        "profit_factor": profit_factor(values),
        "max_drawdown_r": max_drawdown(values),
        "targets": int(sum(trade["exit_reason"] == "target" for trade in trades)),
        "stops": int(sum(trade["exit_reason"] == "stop" for trade in trades)),
        "timeouts": int(sum(trade["exit_reason"].startswith("timeout") for trade in trades)),
        "avg_mfe_r": float(np.mean([trade["mfe_r"] for trade in trades])),
    }


def backtest_ltf_sweeps(
    ltf: pd.DataFrame,
    htf_labels: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    test_start: pd.Timestamp,
    rr_values: list[float],
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> dict[str, Any]:
    ltf = add_ltf_indicators(ltf)
    score_frame = htf_labels[["open_time", "close_time"]].copy()
    score_frame["score"] = scores
    score_frame["active"] = score_frame["score"] >= threshold
    aligned = pd.merge_asof(
        ltf.sort_values("open_time"),
        score_frame[["open_time", "score", "active"]].rename(columns={"open_time": "htf_open_time"}).sort_values("htf_open_time"),
        left_on="open_time",
        right_on="htf_open_time",
        direction="backward",
    )
    aligned["in_test"] = pd.to_datetime(aligned["open_time"], utc=True) > test_start
    aligned["active"] = aligned["active"].fillna(False).astype(bool)

    high_sweep = (aligned["high"] > aligned["prev_range_high"]) & (aligned["close"] < aligned["prev_range_high"])
    low_sweep = (aligned["low"] < aligned["prev_range_low"]) & (aligned["close"] > aligned["prev_range_low"])
    both = high_sweep & low_sweep
    short_signal = high_sweep & ~both
    long_signal = low_sweep & ~both

    output: dict[str, Any] = {}
    for gate_name, gate_mask in {
        "ungated_test": aligned["in_test"].to_numpy(dtype=bool),
        "astro_gated_test": (aligned["in_test"] & aligned["active"]).to_numpy(dtype=bool),
    }.items():
        output[gate_name] = {}
        for rr in rr_values:
            trades: list[dict[str, Any]] = []
            cursor = 20
            while cursor < len(aligned) - 2:
                if not gate_mask[cursor]:
                    cursor += 1
                    continue
                direction = None
                if bool(long_signal.iloc[cursor]):
                    direction = "long"
                elif bool(short_signal.iloc[cursor]):
                    direction = "short"
                if direction is None:
                    cursor += 1
                    continue
                trade = simulate_sweep_trade(
                    aligned,
                    cursor,
                    direction,
                    rr,
                    max_hold_bars,
                    stop_buffer_atr,
                    cost_bps_round_trip,
                )
                if trade is None:
                    cursor += 1
                    continue
                trades.append(trade)
                cursor = max(cursor + 1, int(trade["exit_idx"]) + 1)
            output[gate_name][f"rr_{rr:g}"] = trade_summary(trades)
    return output


def write_top_windows(
    labels: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    windows = labels[["open_time", "close_time", "y_any", "y_high", "y_low", "pivot_here"]].copy()
    windows["score"] = scores
    windows = windows[windows["score"] >= threshold].sort_values("score", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Research BTC astrology/cycle reversal-window timing.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--pivot-threshold-atr", type=float, default=8.0)
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--train-end", default="2023-12-31T23:59:59Z")
    parser.add_argument("--validation-end", default="2024-12-31T23:59:59Z")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=Path("scripts/astro_cycle_timing_results.json"))
    parser.add_argument("--top-windows-csv", type=Path, default=Path("scripts/astro_cycle_top_windows.csv"))
    parser.add_argument("--active-coverage", type=float, default=0.02)
    parser.add_argument("--include-ltf", action="store_true")
    parser.add_argument("--ltf-start", default=None)
    parser.add_argument("--ltf-interval", default="5m")
    parser.add_argument("--rr-values", default="10,20,30")
    parser.add_argument("--max-hold-bars", type=int, default=288)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--cost-bps-round-trip", type=float, default=8.0)
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    split_spec = SplitSpec(
        train_end=pd.Timestamp(parse_utc_datetime(args.train_end)),
        validation_end=pd.Timestamp(parse_utc_datetime(args.validation_end)),
    )
    coverages = [0.005, 0.01, 0.02, 0.05]

    base = load_bybit_cached(args.symbol, args.timeframe, start, end, args.cache_dir)
    frame = prepare_frame(base, args.timeframe)
    pivots = zigzag_pivots(frame, args.pivot_threshold_atr)
    labels = make_forward_labels(frame, pivots, args.horizon_bars)
    features, groups = build_feature_matrix(labels["open_time"], args.cache_dir, args.timeframe)
    masks = split_masks(labels["open_time"], split_spec)

    results: dict[str, Any] = {
        "config": {
            "symbol": args.symbol,
            "start": start,
            "end": end,
            "timeframe": args.timeframe,
            "pivot_threshold_atr": args.pivot_threshold_atr,
            "horizon_bars": args.horizon_bars,
            "train_end": split_spec.train_end,
            "validation_end": split_spec.validation_end,
            "active_coverage": args.active_coverage,
            "skyfield_available": _skyfield_available(),
        },
        "data": {
            "bars": len(frame),
            "first_open": pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC"),
            "last_close": pd.Timestamp(frame["close_time"].iloc[-1]).tz_convert("UTC"),
            "feature_counts": {name: len(columns) for name, columns in groups.items()},
            "pivot_summary": summarize_pivots(frame, pivots, args.horizon_bars),
        },
        "models": {},
    }

    model_specs = {
        "calendar_only": groups["calendar"],
        "cycle_only": groups["cycles"],
        "astro_only": groups["astro"] if groups["astro"] else groups["cycles"],
        "astro_cycle": groups["astro_cycle"],
        "all_real": groups["all"],
    }
    scores_by_model: dict[str, np.ndarray] = {}
    for model_name, columns in model_specs.items():
        metrics, scores = evaluate_feature_set(
            features,
            labels,
            "y_any",
            columns,
            masks,
            pivots,
            args.horizon_bars,
            coverages,
        )
        results["models"][model_name] = metrics
        scores_by_model[model_name] = scores

    interval_seconds = (pd.Timestamp(frame["open_time"].iloc[1]) - pd.Timestamp(frame["open_time"].iloc[0])).total_seconds()
    shift_bars = max(1, int(round(37.0 * SECONDS_PER_DAY / interval_seconds)))
    placebo = shifted_placebo(features, groups["astro_cycle"], shift_bars)
    placebo_metrics, placebo_scores = evaluate_feature_set(
        placebo,
        labels,
        "y_any",
        groups["all"],
        masks,
        pivots,
        args.horizon_bars,
        coverages,
    )
    results["models"]["placebo_shift_37d"] = placebo_metrics
    scores_by_model["placebo_shift_37d"] = placebo_scores

    strategy_model_name = "all_real"
    strategy_scores = scores_by_model[strategy_model_name]
    threshold = select_threshold_from_validation(strategy_scores, masks["validation"], args.active_coverage)
    results["strategy_threshold"] = {
        "model": strategy_model_name,
        "threshold": threshold,
        "selected_on": "validation",
        "coverage": args.active_coverage,
    }
    write_top_windows(labels, strategy_scores, threshold, args.top_windows_csv)

    if args.include_ltf:
        ltf_start = parse_utc_datetime(args.ltf_start) if args.ltf_start else max(start, split_spec.validation_end.to_pydatetime() - timedelta(days=90))
        ltf = load_bybit_cached(args.symbol, args.ltf_interval, ltf_start, end, args.cache_dir)
        rr_values = [float(item.strip()) for item in args.rr_values.split(",") if item.strip()]
        results["ltf_sweep_backtest"] = {
            "interval": args.ltf_interval,
            "start": ltf_start,
            "end": end,
            "rr_values": rr_values,
            "max_hold_bars": args.max_hold_bars,
            "stop_buffer_atr": args.stop_buffer_atr,
            "cost_bps_round_trip": args.cost_bps_round_trip,
            "gates": {},
        }
        for gate_model_name in ["calendar_only", "all_real", "placebo_shift_37d"]:
            gate_scores = scores_by_model[gate_model_name]
            gate_threshold = select_threshold_from_validation(gate_scores, masks["validation"], args.active_coverage)
            results["ltf_sweep_backtest"]["gates"][gate_model_name] = {
                "threshold": gate_threshold,
                "results": backtest_ltf_sweeps(
                    ltf,
                    labels,
                    gate_scores,
                    gate_threshold,
                    split_spec.validation_end,
                    rr_values,
                    args.max_hold_bars,
                    args.stop_buffer_atr,
                    args.cost_bps_round_trip,
                ),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(results, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
