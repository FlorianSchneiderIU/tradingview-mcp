from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_ltf_calendar_probability import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    add_atr,
    calendar_features,
    ensure_ohlcv_frame,
    fixed_cycle_features,
    json_default,
    load_ohlcv_cached,
    parse_float_list,
    parse_int_list,
    parse_utc_datetime,
)

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TIMEFRAME_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1h": 3_600,
    "4h": 14_400,
    "1d": 86_400,
}
RESAMPLE_RULE = {
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}
FLOOR_RULE = {
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "D",
}


@dataclass(frozen=True)
class LayerSpec:
    name: str
    child_tf: str
    parent_tf: str
    horizons: tuple[int, ...]

    @property
    def children_per_parent(self) -> int:
        return TIMEFRAME_SECONDS[self.parent_tf] // TIMEFRAME_SECONDS[self.child_tf]


def default_layers() -> list[LayerSpec]:
    return [
        LayerSpec("4h_to_1d", "4h", "1d", (1,)),
        LayerSpec("1h_to_4h", "1h", "4h", (1, 4, 8)),
        LayerSpec("15m_to_1h", "15m", "1h", (1, 4)),
        LayerSpec("5m_to_15m", "5m", "15m", (1, 3)),
    ]


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.replace([np.inf, -np.inf], np.nan).copy()
    medians = out.median(numeric_only=True)
    return out.fillna(medians).fillna(0.0).astype(float)


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_atr(frame)
    atr = out["atr"].replace(0.0, np.nan)
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_100"] = out["close"].ewm(span=100, adjust=False).mean()
    out["ema_20_slope_atr"] = (out["ema_20"] - out["ema_20"].shift(5)) / atr
    out["ema_100_slope_atr"] = (out["ema_100"] - out["ema_100"].shift(10)) / atr
    out["close_ema20_atr"] = (out["close"] - out["ema_20"]) / atr
    out["close_ema100_atr"] = (out["close"] - out["ema_100"]) / atr
    atr_baseline = out["atr"].rolling(100, min_periods=20).median()
    out["atr_ratio"] = out["atr"] / atr_baseline.replace(0.0, np.nan)
    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rs = rma(gain, 14) / rma(loss, 14).replace(0.0, np.nan)
    out["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    volume_sma = out["volume"].rolling(20, min_periods=5).mean()
    out["volume_ratio"] = out["volume"] / volume_sma.replace(0.0, np.nan)
    return out


def resample_ohlc(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "5m":
        return ensure_ohlcv_frame(frame)
    if timeframe not in RESAMPLE_RULE:
        raise ValueError(f"Unsupported resample timeframe: {timeframe}")
    ordered = ensure_ohlcv_frame(frame)
    resampled = (
        ordered.set_index("open_time")
        .resample(RESAMPLE_RULE[timeframe], label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    resampled["close_time"] = resampled["open_time"] + pd.to_timedelta(TIMEFRAME_SECONDS[timeframe], unit="s") - pd.Timedelta(
        milliseconds=1
    )
    return resampled[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def parent_keys(times: pd.Series, parent_tf: str) -> pd.Series:
    return pd.to_datetime(times, utc=True).dt.floor(FLOOR_RULE[parent_tf])


def completed_parent_mask(frame: pd.DataFrame, parent_tf: str, expected_children: int) -> np.ndarray:
    keys = parent_keys(frame["open_time"], parent_tf)
    sizes = keys.groupby(keys).transform("size").to_numpy(dtype=int)
    return sizes == expected_children


def parent_extreme_indices(frame: pd.DataFrame, parent_tf: str, expected_children: int) -> dict[str, np.ndarray]:
    work = frame[["open_time", "high", "low"]].copy()
    work["parent_key"] = parent_keys(work["open_time"], parent_tf)
    sizes = work.groupby("parent_key")["high"].transform("size")
    work = work.loc[sizes == expected_children]
    grouped = work.groupby("parent_key", sort=True)
    high_idx = grouped["high"].idxmax().to_numpy(dtype=np.int32)
    low_idx = grouped["low"].idxmin().to_numpy(dtype=np.int32)
    return {"high": np.sort(high_idx), "low": np.sort(low_idx)}


def future_extreme_label(n_rows: int, extreme_indices: np.ndarray, horizon_bars: int) -> np.ndarray:
    diff = np.zeros(n_rows + 1, dtype=np.int32)
    for idx in extreme_indices.astype(np.int64):
        start = max(0, int(idx) - horizon_bars)
        stop = int(idx)
        if stop <= start:
            continue
        diff[start] += 1
        diff[stop] -= 1
    return (np.cumsum(diff[:-1]) > 0).astype(np.int8)


def horizon_masks(times: pd.Series, child_tf: str, horizon_bars: int, train_end: datetime, validation_end: datetime, end: datetime) -> dict[str, np.ndarray]:
    ts = pd.to_datetime(times, utc=True)
    horizon_delta = pd.to_timedelta(TIMEFRAME_SECONDS[child_tf] * horizon_bars, unit="s")
    horizon_end = ts + horizon_delta
    train_cut = pd.Timestamp(train_end)
    validation_cut = pd.Timestamp(validation_end)
    end_cut = pd.Timestamp(end)
    return {
        "train": ((ts < train_cut) & (horizon_end < train_cut)).to_numpy(),
        "validation": ((ts >= train_cut) & (ts < validation_cut) & (horizon_end < validation_cut)).to_numpy(),
        "test": ((ts >= validation_cut) & (horizon_end < end_cut)).to_numpy(),
    }


def event_recall_future(active_in_split: np.ndarray, extreme_indices: np.ndarray, split_mask: np.ndarray, horizon_bars: int) -> float:
    split_positions = np.flatnonzero(split_mask)
    if len(split_positions) == 0:
        return float("nan")
    first = int(split_positions[0])
    last = int(split_positions[-1])
    active_full = np.zeros(len(split_mask), dtype=bool)
    active_full[split_positions] = active_in_split
    relevant = extreme_indices[(extreme_indices >= first) & (extreme_indices <= last)]
    if len(relevant) == 0:
        return float("nan")
    hits = 0
    for idx in relevant:
        start = max(first, int(idx) - horizon_bars)
        stop = int(idx)
        if stop > start and bool(active_full[start:stop].any()):
            hits += 1
    return float(hits / len(relevant))


def coverage_table_future(
    y_true: np.ndarray,
    y_score: np.ndarray,
    coverages: list[float],
    *,
    extreme_indices: np.ndarray | None = None,
    split_mask: np.ndarray | None = None,
    horizon_bars: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base_rate = float(np.mean(y_true))
    order = np.argsort(-y_score)
    for coverage in coverages:
        active_count = max(1, int(round(len(y_true) * coverage)))
        active = np.zeros(len(y_true), dtype=bool)
        active[order[:active_count]] = True
        precision = float(np.mean(y_true[active])) if active.any() else float("nan")
        row: dict[str, Any] = {
            "coverage": float(active.mean()),
            "active_bars": int(active.sum()),
            "precision": precision,
            "lift": float(precision / base_rate) if base_rate > 0 else float("nan"),
        }
        if extreme_indices is not None and split_mask is not None and horizon_bars is not None:
            row["event_recall"] = event_recall_future(active, extreme_indices, split_mask, horizon_bars)
        out.append(row)
    return out


def validation_thresholds(y_score: np.ndarray, coverages: list[float]) -> dict[float, float]:
    thresholds = {}
    for coverage in coverages:
        active_count = max(1, int(round(len(y_score) * coverage)))
        thresholds[coverage] = float(np.sort(y_score)[-active_count])
    return thresholds


def threshold_table_future(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: dict[float, float],
    *,
    extreme_indices: np.ndarray,
    split_mask: np.ndarray,
    horizon_bars: int,
) -> list[dict[str, Any]]:
    base_rate = float(np.mean(y_true))
    rows = []
    for intended_coverage, threshold in thresholds.items():
        active = y_score >= threshold
        precision = float(np.mean(y_true[active])) if active.any() else float("nan")
        rows.append(
            {
                "intended_validation_coverage": float(intended_coverage),
                "threshold": float(threshold),
                "coverage": float(active.mean()),
                "active_bars": int(active.sum()),
                "precision": precision,
                "lift": float(precision / base_rate) if active.any() and base_rate > 0 else float("nan"),
                "event_recall": event_recall_future(active, extreme_indices, split_mask, horizon_bars),
            }
        )
    return rows


def calibration_bins(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"y": y_true.astype(float), "score": y_score.astype(float)})
    frame["bin"] = pd.qcut(frame["score"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in frame.groupby("bin", sort=True):
        rows.append(
            {
                "bin": int(bin_id),
                "count": int(len(group)),
                "score_mean": float(group["score"].mean()),
                "actual_rate": float(group["y"].mean()),
                "score_min": float(group["score"].min()),
                "score_max": float(group["score"].max()),
            }
        )
    return rows


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
                        max_iter=600,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=80,
            max_leaf_nodes=17,
            min_samples_leaf=250,
            l2_regularization=0.04,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def fit_model(model: Any, x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> Any:
    if isinstance(model, Pipeline):
        model.fit(x_train, y_train, logit__sample_weight=weights)
    else:
        model.fit(x_train, y_train, sample_weight=weights)
    return model


def predict_probability(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(float)


def price_context_features(frame: pd.DataFrame, spec: LayerSpec) -> pd.DataFrame:
    ts = pd.to_datetime(frame["open_time"], utc=True)
    parent_key = parent_keys(frame["open_time"], spec.parent_tf)
    atr = frame["atr"].replace(0.0, np.nan)
    out = pd.DataFrame(index=frame.index)

    out["range_atr"] = (frame["high"] - frame["low"]) / atr
    out["body_atr"] = (frame["close"] - frame["open"]) / atr
    out["body_abs_atr"] = (frame["close"] - frame["open"]).abs() / atr
    candle_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    out["close_pos"] = (frame["close"] - frame["low"]) / candle_range
    out["upper_wick_frac"] = (frame["high"] - frame[["open", "close"]].max(axis=1)) / candle_range
    out["lower_wick_frac"] = (frame[["open", "close"]].min(axis=1) - frame["low"]) / candle_range
    out["atr_ratio"] = frame["atr_ratio"]
    out["rsi"] = frame["rsi"] / 100.0
    out["volume_ratio"] = frame["volume_ratio"]
    out["close_ema20_atr"] = frame["close_ema20_atr"]
    out["close_ema100_atr"] = frame["close_ema100_atr"]
    out["ema_20_slope_atr"] = frame["ema_20_slope_atr"]
    out["ema_100_slope_atr"] = frame["ema_100_slope_atr"]

    for lag in [1, 2, 3, 6, 12]:
        out[f"return_{lag}_atr"] = (frame["close"] - frame["close"].shift(lag)) / atr

    grouped = frame.groupby(parent_key, sort=False)
    running_high = grouped["high"].cummax()
    running_low = grouped["low"].cummin()
    parent_open = grouped["open"].transform("first")
    parent_high_prev = running_high.groupby(parent_key).shift(1)
    parent_low_prev = running_low.groupby(parent_key).shift(1)

    out["parent_return_atr"] = (frame["close"] - parent_open) / atr
    out["running_parent_range_atr"] = (running_high - running_low) / atr
    out["dist_to_running_high_atr"] = (running_high - frame["close"]) / atr
    out["dist_to_running_low_atr"] = (frame["close"] - running_low) / atr
    out["made_parent_running_high"] = (frame["high"] >= parent_high_prev.fillna(-np.inf)).astype(float)
    out["made_parent_running_low"] = (frame["low"] <= parent_low_prev.fillna(np.inf)).astype(float)
    out["close_pos_in_running_parent"] = (frame["close"] - running_low) / (running_high - running_low).replace(0.0, np.nan)

    elapsed = (ts - parent_key).dt.total_seconds().to_numpy(dtype=float)
    slot = np.floor(elapsed / TIMEFRAME_SECONDS[spec.child_tf]).astype(int)
    denom = max(spec.children_per_parent - 1, 1)
    out["parent_slot_norm"] = slot / denom
    out["parent_slots_remaining"] = (spec.children_per_parent - slot - 1) / denom
    angle = 2.0 * np.pi * slot / max(spec.children_per_parent, 1)
    out["parent_slot_sin"] = np.sin(angle)
    out["parent_slot_cos"] = np.cos(angle)
    out["parent_is_first_child"] = (slot == 0).astype(float)
    out["parent_is_last_child"] = (slot == spec.children_per_parent - 1).astype(float)

    complete = completed_parent_mask(frame, spec.parent_tf, spec.children_per_parent)
    out["parent_complete_group"] = complete.astype(float)
    return out


def build_features(frame: pd.DataFrame, spec: LayerSpec) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    price = price_context_features(frame, spec).reset_index(drop=True)
    calendar = calendar_features(frame["open_time"]).reset_index(drop=True)
    cycles = fixed_cycle_features(frame["open_time"], harmonics=2).reset_index(drop=True)
    features = pd.concat([price, calendar, cycles], axis=1)
    features = clean_features(features).astype(np.float32)
    groups = {
        "price": list(price.columns),
        "time": list(calendar.columns) + list(cycles.columns),
        "price_time": list(features.columns),
    }
    return features, groups


def evaluate_model(
    *,
    features: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    model_name: str,
    seed: int,
    coverages: list[float],
    extreme_indices: np.ndarray,
    horizon_bars: int,
) -> dict[str, Any]:
    x = features.loc[:, columns].to_numpy(dtype=np.float32, copy=False)
    y_train = y[masks["train"]].astype(int)
    y_val = y[masks["validation"]].astype(int)
    y_test = y[masks["test"]].astype(int)
    x_train = x[masks["train"]]
    x_val = x[masks["validation"]]
    x_test = x[masks["test"]]

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
            "top_coverage": coverage_table_future(y_val, score_val, coverages),
        },
        "test": {
            "average_precision": safe_average_precision(y_test, score_test),
            "roc_auc": safe_roc_auc(y_test, score_test),
            "brier": float(brier_score_loss(y_test, np.clip(score_test, 0.0, 1.0))),
            "top_coverage": coverage_table_future(
                y_test,
                score_test,
                coverages,
                extreme_indices=extreme_indices,
                split_mask=masks["test"],
                horizon_bars=horizon_bars,
            ),
            "validation_thresholds": threshold_table_future(
                y_test,
                score_test,
                thresholds,
                extreme_indices=extreme_indices,
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
            valid_thresholds = [
                item for item in result["test"]["validation_thresholds"] if not math.isnan(item["precision"])
            ]
            best_threshold = max(valid_thresholds, key=lambda item: item["lift"]) if valid_thresholds else {}
            rows.append(
                {
                    "layer": record["layer"],
                    "child_tf": record["child_tf"],
                    "parent_tf": record["parent_tf"],
                    "direction": record["direction"],
                    "horizon_bars": record["horizon_bars"],
                    "horizon_minutes": record["horizon_minutes"],
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
                    "best_val_threshold_target_coverage": best_threshold.get(
                        "intended_validation_coverage", float("nan")
                    ),
                    "best_val_threshold_test_coverage": best_threshold.get("coverage", float("nan")),
                    "best_val_threshold_precision": best_threshold.get("precision", float("nan")),
                    "best_val_threshold_lift": best_threshold.get("lift", float("nan")),
                    "best_val_threshold_event_recall": best_threshold.get("event_recall", float("nan")),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["layer", "direction", "horizon_bars", "test_ap"],
        ascending=[True, True, True, False],
    )


def parse_layer_horizons(value: str | None, layers: list[LayerSpec]) -> list[LayerSpec]:
    if not value:
        return layers
    mapping: dict[str, tuple[int, ...]] = {}
    for part in value.split(";"):
        if not part.strip():
            continue
        name, raw_horizons = part.split(":", 1)
        mapping[name.strip()] = tuple(parse_int_list(raw_horizons))
    out = []
    for layer in layers:
        if layer.name in mapping:
            out.append(LayerSpec(layer.name, layer.child_tf, layer.parent_tf, mapping[layer.name]))
        else:
            out.append(layer)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical BTC parent-extreme probability research.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-01-01")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--models", default="hgb")
    parser.add_argument("--feature-sets", default="price_time")
    parser.add_argument("--directions", default="low,high")
    parser.add_argument("--coverages", default="0.005,0.01,0.02,0.05,0.10,0.20")
    parser.add_argument(
        "--layer-horizons",
        default=None,
        help="Optional semicolon map, e.g. '4h_to_1d:1;1h_to_4h:1,4;15m_to_1h:1;5m_to_15m:1'.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=Path("scripts/hierarchical_reversal_results.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("scripts/hierarchical_reversal_summary.csv"))
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    train_end = parse_utc_datetime(args.train_end)
    validation_end = parse_utc_datetime(args.validation_end)
    model_names = parse_str_list(args.models)
    requested_feature_sets = parse_str_list(args.feature_sets)
    directions = parse_str_list(args.directions)
    coverages = parse_float_list(args.coverages)
    layers = parse_layer_horizons(args.layer_horizons, default_layers())

    print(f"Loading {args.symbol} 5m from cache...")
    base_5m = load_ohlcv_cached(args.symbol, "5m", start, end, args.cache_dir)
    base_5m = base_5m.loc[pd.to_datetime(base_5m["open_time"], utc=True) < pd.Timestamp(end)].reset_index(drop=True)
    frames = {"5m": add_indicators(base_5m)}
    for timeframe in ["15m", "1h", "4h", "1d"]:
        frames[timeframe] = add_indicators(resample_ohlc(base_5m, timeframe))
    for timeframe, frame in frames.items():
        print(f"  {timeframe}: {len(frame):,} rows {frame['open_time'].min()} -> {frame['open_time'].max()}")

    records: list[dict[str, Any]] = []
    for layer in layers:
        child = frames[layer.child_tf]
        print(f"\nLayer {layer.name}: {layer.child_tf} predicts {layer.parent_tf} extremes")
        features, groups = build_features(child, layer)
        feature_sets = {name: groups[name] for name in requested_feature_sets if name in groups}
        if not feature_sets:
            raise ValueError(f"No valid feature sets requested. Available: {sorted(groups)}")
        extremes = parent_extreme_indices(child, layer.parent_tf, layer.children_per_parent)
        print(
            f"  complete parents={len(extremes['low']):,}; "
            f"features={features.shape[1]} columns; horizons={layer.horizons}"
        )
        for horizon in layer.horizons:
            masks = horizon_masks(child["open_time"], layer.child_tf, horizon, train_end, validation_end, end)
            horizon_minutes = horizon * TIMEFRAME_SECONDS[layer.child_tf] // 60
            for direction in directions:
                if direction not in extremes:
                    raise ValueError(f"Unknown direction: {direction}")
                y = future_extreme_label(len(child), extremes[direction], horizon)
                record = {
                    "layer": layer.name,
                    "child_tf": layer.child_tf,
                    "parent_tf": layer.parent_tf,
                    "direction": direction,
                    "horizon_bars": int(horizon),
                    "horizon_minutes": int(horizon_minutes),
                    "children_per_parent": int(layer.children_per_parent),
                    "extreme_events": int(len(extremes[direction])),
                    "base_rates": {split: float(np.mean(y[mask])) for split, mask in masks.items()},
                    "rows": {split: int(mask.sum()) for split, mask in masks.items()},
                    "results": [],
                }
                print(
                    f"  h={horizon:<2} {direction:<4} test base={record['base_rates']['test']:.3%} "
                    f"rows test={record['rows']['test']:,}"
                )
                for feature_name, columns in feature_sets.items():
                    for model_name in model_names:
                        print(f"    fitting {model_name:<5} {feature_name:<10} ({len(columns)} cols)")
                        result = evaluate_model(
                            features=features,
                            columns=columns,
                            y=y,
                            masks=masks,
                            model_name=model_name,
                            seed=args.seed,
                            coverages=coverages,
                            extreme_indices=extremes[direction],
                            horizon_bars=horizon,
                        )
                        result["feature_set"] = feature_name
                        record["results"].append(result)
                records.append(record)

    output = {
        "symbol": args.symbol,
        "start": start,
        "end": end,
        "train_end_exclusive": train_end,
        "validation_end_exclusive": validation_end,
        "models": model_names,
        "feature_sets": requested_feature_sets,
        "coverages": coverages,
        "layers": [layer.__dict__ | {"children_per_parent": layer.children_per_parent} for layer in layers],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=json_default), encoding="utf-8")
    summary = flatten_summary(records)
    summary.to_csv(args.summary_csv, index=False)
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.summary_csv}")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
