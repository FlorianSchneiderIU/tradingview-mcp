from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

UTC = timezone.utc
SECONDS_PER_DAY = 86_400.0
DEFAULT_CACHE_DIR = Path("scripts/.cache/astro_cycle")


@dataclass(frozen=True)
class PivotSet:
    threshold_atr: float
    indices: np.ndarray
    kinds: np.ndarray


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


def ensure_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out = frame.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    if "close_time" in out.columns:
        out["close_time"] = pd.to_datetime(out["close_time"], utc=True)
    else:
        out["close_time"] = out["open_time"]
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    return out.dropna(subset=["open_time", *required]).sort_values("open_time").reset_index(drop=True)


def load_ohlcv_cached(symbol: str, timeframe: str, start: datetime, end: datetime, cache_dir: Path) -> pd.DataFrame:
    symbol_key = safe_symbol(symbol)
    exact = cache_dir / f"bybit_{symbol_key}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
    if exact.exists():
        return ensure_ohlcv_frame(pd.read_pickle(exact))

    for candidate in sorted(cache_dir.glob(f"bybit_{symbol_key}_{timeframe}_*.pkl")):
        frame = ensure_ohlcv_frame(pd.read_pickle(candidate))
        if frame.empty:
            continue
        first = pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC")
        last = pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC")
        if first <= pd.Timestamp(start) and last >= pd.Timestamp(end):
            mask = (frame["open_time"] >= pd.Timestamp(start)) & (frame["open_time"] <= pd.Timestamp(end))
            return frame.loc[mask].reset_index(drop=True)
    raise FileNotFoundError(f"No cached {symbol} {timeframe} frame covering {start} to {end} in {cache_dir}")


def add_atr(frame: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    out = ensure_ohlcv_frame(frame)
    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = true_range.ewm(alpha=1 / length, adjust=False).mean()
    return out


def zigzag_pivots(frame: pd.DataFrame, threshold_atr: float) -> PivotSet:
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    atr = frame["atr"].to_numpy(dtype=float)
    valid = np.flatnonzero(np.isfinite(atr) & (atr > 0))
    if len(valid) < 20:
        return PivotSet(threshold_atr, np.array([], dtype=np.int32), np.array([], dtype=object))

    start = int(valid[0])
    seeking = "high"
    high_idx: int | None = start
    low_idx: int | None = start
    high_price = float(highs[start])
    low_price = float(lows[start])
    last_pivot_idx = -1
    pivots: list[tuple[int, str]] = []

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
                pivots.append((int(high_idx), "high"))
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
                pivots.append((int(low_idx), "low"))
                last_pivot_idx = int(low_idx)
                seeking = "high"
                high_idx = None
                high_price = -math.inf
                continue

    deduped = sorted({(idx, kind): (idx, kind) for idx, kind in pivots}.values(), key=lambda item: item[0])
    return PivotSet(
        threshold_atr=float(threshold_atr),
        indices=np.array([idx for idx, _ in deduped], dtype=np.int32),
        kinds=np.array([kind for _, kind in deduped], dtype=object),
    )


def forward_label(n_rows: int, pivot_indices: np.ndarray, horizon_bars: int) -> np.ndarray:
    diff = np.zeros(n_rows + 1, dtype=np.int32)
    starts = np.maximum(0, pivot_indices.astype(np.int64) - horizon_bars + 1)
    stops = np.minimum(n_rows, pivot_indices.astype(np.int64) + 1)
    np.add.at(diff, starts, 1)
    np.add.at(diff, stops, -1)
    return (np.cumsum(diff[:-1]) > 0).astype(np.int8)


def calendar_features(times: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(times, utc=True)
    hour = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0
    minute = ts.dt.minute.to_numpy(dtype=float)
    dow = ts.dt.dayofweek.to_numpy(dtype=float)
    dom = (ts.dt.day.to_numpy(dtype=float) - 1.0) / 31.0
    month = ts.dt.month.to_numpy(dtype=float) - 1.0
    doy = ts.dt.dayofyear.to_numpy(dtype=float) - 1.0

    out = pd.DataFrame(index=ts.index)
    cyclic_inputs = [
        ("tod", hour, 24.0),
        ("minute", minute, 60.0),
        ("dow", dow, 7.0),
        ("dom", dom, 1.0),
        ("month", month, 12.0),
        ("doy", doy, 365.2425),
    ]
    for name, value, period in cyclic_inputs:
        for harmonic in [1, 2]:
            angle = 2.0 * np.pi * harmonic * value / period
            out[f"cal_{name}_h{harmonic}_sin"] = np.sin(angle)
            out[f"cal_{name}_h{harmonic}_cos"] = np.cos(angle)

    out["cal_is_weekend"] = (dow >= 5).astype(float)
    out["cal_asia_session"] = ((hour >= 0.0) & (hour < 8.0)).astype(float)
    out["cal_europe_session"] = ((hour >= 7.0) & (hour < 16.0)).astype(float)
    out["cal_us_session"] = ((hour >= 13.5) & (hour < 21.0)).astype(float)
    out["cal_eu_us_overlap"] = ((hour >= 13.5) & (hour < 16.0)).astype(float)
    out["cal_daily_open_2h"] = ((hour < 2.0) | (hour >= 22.0)).astype(float)
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
        for harmonic in range(1, harmonics + 1):
            angle = 2.0 * np.pi * harmonic * days / period
            out[f"cyc_{name}_h{harmonic}_sin"] = np.sin(angle)
            out[f"cyc_{name}_h{harmonic}_cos"] = np.cos(angle)
    return out


def angular_distance_to_aspect(angle: np.ndarray, aspect: float) -> np.ndarray:
    diff = (angle - aspect + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def astro_cache_path(times: pd.Series, cache_dir: Path, timeframe: str) -> Path:
    ts = pd.to_datetime(times, utc=True)
    start = pd.Timestamp(ts.iloc[0]).strftime("%Y%m%d%H%M")
    end = pd.Timestamp(ts.iloc[-1]).strftime("%Y%m%d%H%M")
    return cache_dir / f"skyfield_geo_{timeframe}_{start}_{end}.pkl"


def astro_core_features(raw: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    bodies = [column[:-4] for column in raw.columns if column.endswith("_lon")]
    day_values = (
        (pd.to_datetime(times, utc=True).reset_index(drop=True) - pd.Timestamp("2000-01-01T00:00:00Z"))
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / SECONDS_PER_DAY
    )

    for body in bodies:
        lon = raw[f"{body}_lon"].to_numpy(dtype=float)
        lon_rad = np.deg2rad(lon)
        for harmonic in [1, 2]:
            out[f"astro_{body}_lon_h{harmonic}_sin"] = np.sin(harmonic * lon_rad)
            out[f"astro_{body}_lon_h{harmonic}_cos"] = np.cos(harmonic * lon_rad)
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
        for harmonic in range(1, 9):
            out[f"astro_moon_phase_h{harmonic}_sin"] = np.sin(harmonic * phase_rad)
            out[f"astro_moon_phase_h{harmonic}_cos"] = np.cos(harmonic * phase_rad)
        for aspect in [0.0, 90.0, 180.0, 270.0]:
            out[f"astro_moon_phase_orb_{int(aspect)}"] = angular_distance_to_aspect(phase, aspect) / 180.0

    major_aspects = np.array([0.0, 60.0, 90.0, 120.0, 180.0], dtype=float)
    cluster_3 = np.zeros(len(raw), dtype=float)
    cluster_5 = np.zeros(len(raw), dtype=float)
    for left, right in combinations(bodies, 2):
        angle = (raw[f"{left}_lon"].to_numpy(dtype=float) - raw[f"{right}_lon"].to_numpy(dtype=float)) % 360.0
        angle_rad = np.deg2rad(angle)
        out[f"astro_pair_{left}_{right}_h1_sin"] = np.sin(angle_rad)
        out[f"astro_pair_{left}_{right}_h1_cos"] = np.cos(angle_rad)
        min_orb = np.min(np.vstack([angular_distance_to_aspect(angle, aspect) for aspect in major_aspects]), axis=0)
        out[f"astro_pair_{left}_{right}_major_orb"] = min_orb / 180.0
        cluster_3 += (min_orb <= 3.0).astype(float)
        cluster_5 += (min_orb <= 5.0).astype(float)

    out["astro_major_aspect_cluster_orb3"] = cluster_3
    out["astro_major_aspect_cluster_orb5"] = cluster_5
    return out


def load_astro_core_for_5m(
    five_minute_times: pd.Series,
    symbol: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
) -> pd.DataFrame:
    frame_15m = load_ohlcv_cached(symbol, "15m", start, end, cache_dir)
    raw_path = astro_cache_path(frame_15m["open_time"], cache_dir, "15m")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing cached Skyfield positions: {raw_path}")
    raw = pd.read_pickle(raw_path).reset_index(drop=True)
    if len(raw) != len(frame_15m):
        raise ValueError(f"Skyfield cache length mismatch: {len(raw)} rows for {len(frame_15m)} 15m candles")
    astro_15m = astro_core_features(raw, frame_15m["open_time"]).replace([np.inf, -np.inf], np.nan)
    astro_15m = astro_15m.fillna(astro_15m.median(numeric_only=True)).fillna(0.0)
    astro_15m.insert(0, "open_time", pd.to_datetime(frame_15m["open_time"], utc=True))
    target = pd.DataFrame({"open_time": pd.to_datetime(five_minute_times, utc=True)})
    merged = pd.merge_asof(target.sort_values("open_time"), astro_15m.sort_values("open_time"), on="open_time", direction="backward")
    return merged.drop(columns=["open_time"]).reset_index(drop=True)


def build_feature_matrix(
    times: pd.Series,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    include_astro: bool,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    civil = calendar_features(times).reset_index(drop=True)
    cycles = fixed_cycle_features(times).reset_index(drop=True)
    pieces = [civil, cycles]
    groups["civil"] = list(civil.columns)
    groups["cycles"] = list(cycles.columns)
    groups["civil_cycle"] = groups["civil"] + groups["cycles"]

    if include_astro:
        astro = load_astro_core_for_5m(times, symbol, start, end, cache_dir)
        pieces.append(astro)
        groups["astro_core"] = list(astro.columns)
    else:
        groups["astro_core"] = []

    features = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0).astype(np.float32)
    groups["all_time"] = list(features.columns)
    return features, groups


def split_masks(times: pd.Series, train_end: datetime, validation_end: datetime) -> dict[str, np.ndarray]:
    ts = pd.to_datetime(times, utc=True)
    train_cut = pd.Timestamp(train_end)
    validation_cut = pd.Timestamp(validation_end)
    return {
        "train": (ts < train_cut).to_numpy(),
        "validation": ((ts >= train_cut) & (ts < validation_cut)).to_numpy(),
        "test": (ts >= validation_cut).to_numpy(),
    }


def balanced_weights(y: np.ndarray) -> np.ndarray:
    y = y.astype(int)
    pos_rate = float(np.mean(y))
    if pos_rate <= 0.0 or pos_rate >= 1.0:
        return np.ones(len(y), dtype=np.float32)
    out = np.empty(len(y), dtype=np.float32)
    out[y == 1] = 0.5 / pos_rate
    out[y == 0] = 0.5 / (1.0 - pos_rate)
    return out


def make_model(name: str, seed: int) -> Any:
    if name == "logit":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=0.08,
                        max_iter=500,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=100,
            max_leaf_nodes=17,
            min_samples_leaf=300,
            l2_regularization=0.03,
            random_state=seed,
        )
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=350,
            max_depth=8,
            min_samples_leaf=500,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def fit_model(model: Any, x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> Any:
    if isinstance(model, Pipeline):
        model.fit(x_train, y_train, logit__sample_weight=weights)
    elif isinstance(model, HistGradientBoostingClassifier):
        model.fit(x_train, y_train, sample_weight=weights)
    else:
        model.fit(x_train, y_train)
    return model


def predict_probability(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(float)
    decision = model.decision_function(x)
    return (1.0 / (1.0 + np.exp(-decision))).astype(float)


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def coverage_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    coverages: list[float],
    *,
    pivot_indices: np.ndarray | None = None,
    split_mask: np.ndarray | None = None,
    horizon_bars: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base_rate = float(np.mean(y_true))
    n = len(y_true)
    order = np.argsort(-y_score)
    for coverage in coverages:
        active_count = max(1, int(round(n * coverage)))
        active = np.zeros(n, dtype=bool)
        active[order[:active_count]] = True
        precision = float(np.mean(y_true[active])) if active.any() else float("nan")
        row: dict[str, Any] = {
            "coverage": float(active.mean()),
            "active_bars": int(active.sum()),
            "precision": precision,
            "lift": float(precision / base_rate) if base_rate > 0 else float("nan"),
        }
        if pivot_indices is not None and split_mask is not None and horizon_bars is not None:
            row["event_recall"] = event_recall(active, pivot_indices, split_mask, horizon_bars)
        out.append(row)
    return out


def threshold_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: dict[float, float],
    *,
    pivot_indices: np.ndarray | None = None,
    split_mask: np.ndarray | None = None,
    horizon_bars: int | None = None,
) -> list[dict[str, Any]]:
    base_rate = float(np.mean(y_true))
    out = []
    for intended_coverage, threshold in thresholds.items():
        active = y_score >= threshold
        precision = float(np.mean(y_true[active])) if active.any() else float("nan")
        row: dict[str, Any] = {
            "intended_validation_coverage": float(intended_coverage),
            "threshold": float(threshold),
            "coverage": float(active.mean()),
            "active_bars": int(active.sum()),
            "precision": precision,
            "lift": float(precision / base_rate) if active.any() and base_rate > 0 else float("nan"),
        }
        if pivot_indices is not None and split_mask is not None and horizon_bars is not None:
            row["event_recall"] = event_recall(active, pivot_indices, split_mask, horizon_bars)
        out.append(row)
    return out


def validation_thresholds(y_score: np.ndarray, coverages: list[float]) -> dict[float, float]:
    thresholds = {}
    for coverage in coverages:
        active_count = max(1, int(round(len(y_score) * coverage)))
        thresholds[coverage] = float(np.sort(y_score)[-active_count])
    return thresholds


def event_recall(active_in_split: np.ndarray, pivot_indices: np.ndarray, split_mask: np.ndarray, horizon_bars: int) -> float:
    split_positions = np.flatnonzero(split_mask)
    if len(split_positions) == 0:
        return float("nan")
    first = int(split_positions[0])
    last = int(split_positions[-1])
    active_full = np.zeros(len(split_mask), dtype=bool)
    active_full[split_positions] = active_in_split
    relevant = pivot_indices[(pivot_indices >= first) & (pivot_indices <= last)]
    if len(relevant) == 0:
        return float("nan")
    hits = 0
    for pivot_idx in relevant:
        start = max(first, int(pivot_idx) - horizon_bars + 1)
        if bool(active_full[start : int(pivot_idx) + 1].any()):
            hits += 1
    return float(hits / len(relevant))


def calibration_bins(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"y": y_true.astype(float), "score": y_score.astype(float)})
    frame["bin"] = pd.qcut(frame["score"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    out = []
    for bin_id, group in frame.groupby("bin", sort=True):
        out.append(
            {
                "bin": int(bin_id),
                "count": int(len(group)),
                "score_mean": float(group["score"].mean()),
                "actual_rate": float(group["y"].mean()),
                "score_min": float(group["score"].min()),
                "score_max": float(group["score"].max()),
            }
        )
    return out


def evaluate_one(
    *,
    features: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    model_name: str,
    seed: int,
    coverages: list[float],
    pivot_indices: np.ndarray,
    horizon_bars: int,
) -> dict[str, Any]:
    if not columns:
        raise ValueError("Feature set has no columns.")
    x = features.loc[:, columns].to_numpy(dtype=np.float32, copy=False)
    x_train = x[masks["train"]]
    x_val = x[masks["validation"]]
    x_test = x[masks["test"]]
    y_train = y[masks["train"]].astype(int)
    y_val = y[masks["validation"]].astype(int)
    y_test = y[masks["test"]].astype(int)
    model = make_model(model_name, seed)
    fit_model(model, x_train, y_train, balanced_weights(y_train))
    score_val_raw = predict_probability(model, x_val)
    score_test_raw = predict_probability(model, x_test)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(score_val_raw, y_val)
    score_val = calibrator.transform(score_val_raw)
    score_test = calibrator.transform(score_test_raw)
    thresholds = validation_thresholds(score_val, coverages)

    return {
        "model": model_name,
        "feature_count": len(columns),
        "train_base_rate": float(np.mean(y_train)),
        "validation_base_rate": float(np.mean(y_val)),
        "test_base_rate": float(np.mean(y_test)),
        "validation": {
            "average_precision": safe_average_precision(y_val, score_val),
            "roc_auc": safe_roc_auc(y_val, score_val),
            "brier": float(brier_score_loss(y_val, np.clip(score_val, 0.0, 1.0))),
            "top_coverage": coverage_table(y_val, score_val, coverages),
        },
        "test": {
            "average_precision": safe_average_precision(y_test, score_test),
            "roc_auc": safe_roc_auc(y_test, score_test),
            "brier": float(brier_score_loss(y_test, np.clip(score_test, 0.0, 1.0))),
            "top_coverage": coverage_table(
                y_test,
                score_test,
                coverages,
                pivot_indices=pivot_indices,
                split_mask=masks["test"],
                horizon_bars=horizon_bars,
            ),
            "validation_thresholds": threshold_table(
                y_test,
                score_test,
                thresholds,
                pivot_indices=pivot_indices,
                split_mask=masks["test"],
                horizon_bars=horizon_bars,
            ),
            "calibration": calibration_bins(y_test, score_test),
        },
    }


def flatten_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        for result in record["results"]:
            best_top = max(result["test"]["top_coverage"], key=lambda item: item["lift"])
            best_threshold = max(
                result["test"]["validation_thresholds"],
                key=lambda item: -math.inf if math.isnan(item["precision"]) else item["lift"],
            )
            rows.append(
                {
                    "threshold_atr": record["threshold_atr"],
                    "horizon_bars": record["horizon_bars"],
                    "feature_set": result["feature_set"],
                    "model": result["model"],
                    "feature_count": result["feature_count"],
                    "test_base_rate": result["test_base_rate"],
                    "test_ap": result["test"]["average_precision"],
                    "test_roc_auc": result["test"]["roc_auc"],
                    "test_brier": result["test"]["brier"],
                    "best_ranked_coverage": best_top["coverage"],
                    "best_ranked_precision": best_top["precision"],
                    "best_ranked_lift": best_top["lift"],
                    "best_ranked_event_recall": best_top.get("event_recall", float("nan")),
                    "best_val_threshold_target_coverage": best_threshold["intended_validation_coverage"],
                    "best_val_threshold_test_coverage": best_threshold["coverage"],
                    "best_val_threshold_precision": best_threshold["precision"],
                    "best_val_threshold_lift": best_threshold["lift"],
                    "best_val_threshold_event_recall": best_threshold.get("event_recall", float("nan")),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["threshold_atr", "horizon_bars", "best_ranked_lift", "test_ap"],
        ascending=[True, True, False, False],
    )


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict 5m BTC reversal-window probability from time-only features.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-01-01")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--thresholds", default="6,8", help="ATR directional-change thresholds.")
    parser.add_argument("--horizons", default="3,6,12,24", help="Forward 5m candle horizons.")
    parser.add_argument("--feature-sets", default="civil,civil_cycle,astro_core,all_time")
    parser.add_argument("--models", default="hgb,logit")
    parser.add_argument("--coverages", default="0.005,0.01,0.02,0.05,0.10")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-astro", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("scripts/ltf_calendar_probability_results.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("scripts/ltf_calendar_probability_summary.csv"))
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    train_end = parse_utc_datetime(args.train_end)
    validation_end = parse_utc_datetime(args.validation_end)
    thresholds = parse_float_list(args.thresholds)
    horizons = parse_int_list(args.horizons)
    requested_feature_sets = parse_str_list(args.feature_sets)
    model_names = parse_str_list(args.models)
    coverages = parse_float_list(args.coverages)

    print(f"Loading {args.symbol} 5m from cache...")
    frame = add_atr(load_ohlcv_cached(args.symbol, "5m", start, end, args.cache_dir))
    masks = split_masks(frame["open_time"], train_end, validation_end)
    print(
        f"Rows={len(frame):,} train={int(masks['train'].sum()):,} "
        f"validation={int(masks['validation'].sum()):,} test={int(masks['test'].sum()):,}"
    )

    print("Building time-only features...")
    include_astro = not args.no_astro and any(item in {"astro_core", "all_time"} for item in requested_feature_sets)
    features, groups = build_feature_matrix(
        frame["open_time"],
        symbol=args.symbol,
        start=start,
        end=end,
        cache_dir=args.cache_dir,
        include_astro=include_astro,
    )
    print(f"Feature matrix: {features.shape[0]:,} rows x {features.shape[1]:,} columns")
    feature_sets = {name: groups[name] for name in requested_feature_sets if name in groups and groups[name]}
    missing_sets = sorted(set(requested_feature_sets) - set(feature_sets))
    if missing_sets:
        print(f"Skipping unavailable/empty feature sets: {', '.join(missing_sets)}")

    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        pivots = zigzag_pivots(frame, threshold)
        print(f"Threshold {threshold:g} ATR: {len(pivots.indices):,} pivots")
        for horizon in horizons:
            y = forward_label(len(frame), pivots.indices, horizon)
            record = {
                "threshold_atr": float(threshold),
                "horizon_bars": int(horizon),
                "horizon_minutes": int(horizon * 5),
                "pivots": int(len(pivots.indices)),
                "base_rates": {
                    split: float(np.mean(y[mask])) for split, mask in masks.items()
                },
                "results": [],
            }
            print(
                f"  Horizon {horizon:>2} bars ({horizon * 5:>3}m): "
                f"test base={record['base_rates']['test']:.3%}"
            )
            for feature_name, columns in feature_sets.items():
                for model_name in model_names:
                    print(f"    fitting {model_name:<5} {feature_name:<12} ({len(columns)} cols)")
                    result = evaluate_one(
                        features=features,
                        columns=columns,
                        y=y,
                        masks=masks,
                        model_name=model_name,
                        seed=args.seed,
                        coverages=coverages,
                        pivot_indices=pivots.indices,
                        horizon_bars=horizon,
                    )
                    result["feature_set"] = feature_name
                    record["results"].append(result)
            records.append(record)

    output = {
        "symbol": args.symbol,
        "timeframe": "5m",
        "start": start,
        "end": end,
        "train_end_exclusive": train_end,
        "validation_end_exclusive": validation_end,
        "feature_sets": {name: len(columns) for name, columns in feature_sets.items()},
        "models": model_names,
        "coverages": coverages,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=json_default), encoding="utf-8")
    summary = flatten_summary(records)
    summary.to_csv(args.summary_csv, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_csv}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
