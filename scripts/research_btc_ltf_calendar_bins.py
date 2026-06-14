from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_ltf_calendar_probability import (
    DEFAULT_CACHE_DIR,
    add_atr,
    calibration_bins,
    coverage_table,
    forward_label,
    json_default,
    load_ohlcv_cached,
    parse_float_list,
    parse_int_list,
    parse_utc_datetime,
    split_masks,
    threshold_table,
    validation_thresholds,
    zigzag_pivots,
)

SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class KeySpec:
    name: str
    keys: np.ndarray
    family: str


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def logit(values: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    clipped = np.clip(values.astype(float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def phase_bin(times: pd.Series, period_days: float, bins: int, epoch: str = "2000-01-06T18:14:00Z") -> np.ndarray:
    ts = pd.to_datetime(times, utc=True)
    days = (ts - pd.Timestamp(epoch)).dt.total_seconds().to_numpy(dtype=float) / SECONDS_PER_DAY
    phase = np.mod(days / period_days, 1.0)
    return np.floor(phase * bins).astype(np.int64).clip(0, bins - 1)


def make_keys(times: pd.Series) -> list[KeySpec]:
    ts = pd.to_datetime(times, utc=True)
    hour = ts.dt.hour.to_numpy(dtype=np.int64)
    minute = ts.dt.minute.to_numpy(dtype=np.int64)
    dow = ts.dt.dayofweek.to_numpy(dtype=np.int64)
    month = (ts.dt.month.to_numpy(dtype=np.int64) - 1).clip(0, 11)
    slot_5m = (hour * 12 + minute // 5).clip(0, 287)
    slot_15m = (hour * 4 + minute // 15).clip(0, 95)
    slot_30m = (hour * 2 + minute // 30).clip(0, 47)
    hour_week = dow * 24 + hour
    slot_week = dow * 288 + slot_5m
    session = np.select(
        [
            (hour >= 0) & (hour < 7),
            (hour >= 7) & (hour < 13),
            (hour >= 13) & (hour < 17),
            (hour >= 17) & (hour < 22),
        ],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int64)

    moon8 = phase_bin(times, 29.530588, 8)
    moon16 = phase_bin(times, 29.530588, 16)
    draconic8 = phase_bin(times, 27.212221, 8)
    anomalistic8 = phase_bin(times, 27.55455, 8)
    mercury8 = phase_bin(times, 115.88, 8)
    venus8 = phase_bin(times, 583.92, 8)
    mars8 = phase_bin(times, 779.94, 8)
    halving8 = phase_bin(times, 1458.0, 8, epoch="2012-11-28T00:00:00Z")

    return [
        KeySpec("tod_5m", slot_5m, "civil"),
        KeySpec("tod_15m", slot_15m, "civil"),
        KeySpec("hour_week", hour_week, "civil"),
        KeySpec("slot_week_30m", dow * 48 + slot_30m, "civil"),
        KeySpec("slot_week_15m", dow * 96 + slot_15m, "civil"),
        KeySpec("slot_week_5m", slot_week, "civil"),
        KeySpec("month_hour", month * 24 + hour, "civil"),
        KeySpec("month_slot_30m", month * 48 + slot_30m, "civil"),
        KeySpec("dow_session", dow * 5 + session, "civil"),
        KeySpec("moon8", moon8, "cycle"),
        KeySpec("moon16", moon16, "cycle"),
        KeySpec("moon_draconic_combo", moon8 * 8 + draconic8, "cycle"),
        KeySpec("moon_anomalistic_combo", moon8 * 8 + anomalistic8, "cycle"),
        KeySpec("moon_three_cycle_combo", (moon8 * 8 + draconic8) * 8 + anomalistic8, "cycle"),
        KeySpec("inner_planet_combo", (mercury8 * 8 + venus8) * 8 + mars8, "cycle"),
        KeySpec("moon_halving_combo", moon8 * 8 + halving8, "cycle"),
        KeySpec("dow_session_moon8", (dow * 5 + session) * 8 + moon8, "mixed"),
        KeySpec("hour_week_moon8", hour_week * 8 + moon8, "mixed"),
        KeySpec("hour_week_moon16", hour_week * 16 + moon16, "mixed"),
        KeySpec("slot_week_30m_moon8", (dow * 48 + slot_30m) * 8 + moon8, "mixed"),
        KeySpec("slot_week_15m_moon8", (dow * 96 + slot_15m) * 8 + moon8, "mixed"),
        KeySpec("slot_week_5m_moon4", slot_week * 4 + phase_bin(times, 29.530588, 4), "mixed"),
        KeySpec("slot_day_15m_moon8", slot_15m * 8 + moon8, "mixed"),
        KeySpec("slot_day_5m_moon8", slot_5m * 8 + moon8, "mixed"),
    ]


def smoothed_probabilities(keys: np.ndarray, y: np.ndarray, train_mask: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    train_keys = keys[train_mask].astype(np.int64)
    y_train = y[train_mask].astype(float)
    global_rate = float(np.mean(y_train))
    size = int(max(int(keys.max()), int(train_keys.max())) + 1)
    counts = np.bincount(train_keys, minlength=size).astype(float)
    positives = np.bincount(train_keys, weights=y_train, minlength=size).astype(float)
    rates = (positives + alpha * global_rate) / (counts + alpha)
    return rates[keys.astype(np.int64)], global_rate


def calibrate_from_validation(raw_val: np.ndarray, y_val: np.ndarray, raw_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, y_val)
    return calibrator.transform(raw_val), calibrator.transform(raw_test)


def evaluate_scores(
    *,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    raw_scores: np.ndarray,
    coverages: list[float],
    pivot_indices: np.ndarray,
    horizon_bars: int,
) -> dict[str, Any]:
    y_val = y[masks["validation"]].astype(int)
    y_test = y[masks["test"]].astype(int)
    raw_val = raw_scores[masks["validation"]]
    raw_test = raw_scores[masks["test"]]
    score_val, score_test = calibrate_from_validation(raw_val, y_val, raw_test)
    thresholds = validation_thresholds(score_val, coverages)
    raw_thresholds = validation_thresholds(raw_val, coverages)
    return {
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
            "raw_validation_thresholds": threshold_table(
                y_test,
                raw_test,
                raw_thresholds,
                pivot_indices=pivot_indices,
                split_mask=masks["test"],
                horizon_bars=horizon_bars,
            ),
            "calibration": calibration_bins(y_test, score_test),
        },
    }


def choose_alpha_for_key(
    *,
    key: KeySpec,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    alphas: list[float],
) -> dict[str, Any]:
    y_val = y[masks["validation"]].astype(int)
    best: dict[str, Any] | None = None
    for alpha in alphas:
        raw_scores, global_rate = smoothed_probabilities(key.keys, y, masks["train"], alpha)
        score_val, _ = calibrate_from_validation(raw_scores[masks["validation"]], y_val, raw_scores[masks["validation"]])
        ap = safe_average_precision(y_val, score_val)
        auc = safe_roc_auc(y_val, score_val)
        candidate = {
            "key": key,
            "alpha": float(alpha),
            "global_rate": global_rate,
            "raw_scores": raw_scores,
            "validation_ap": ap,
            "validation_roc_auc": auc,
        }
        if best is None or (ap, auc) > (best["validation_ap"], best["validation_roc_auc"]):
            best = candidate
    if best is None:
        raise RuntimeError(f"No alpha result for {key.name}")
    return best


def build_ensembles(selected: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    by_name = {item["key"].name: item["raw_scores"] for item in selected}
    by_family: dict[str, list[np.ndarray]] = {}
    for item in selected:
        by_family.setdefault(item["key"].family, []).append(item["raw_scores"])

    ensembles: dict[str, np.ndarray] = {}
    for family, scores in by_family.items():
        if len(scores) >= 2:
            ensembles[f"ensemble_{family}"] = sigmoid(np.mean(np.vstack([logit(score) for score in scores]), axis=0))

    ranked = sorted(selected, key=lambda item: item["validation_ap"], reverse=True)
    for size in [3, 5, 8]:
        if len(ranked) >= size:
            ensembles[f"ensemble_top{size}_val_ap"] = sigmoid(
                np.mean(np.vstack([logit(item["raw_scores"]) for item in ranked[:size]]), axis=0)
            )

    preferred = [
        "slot_week_5m",
        "slot_week_15m",
        "hour_week",
        "dow_session_moon8",
        "hour_week_moon8",
        "slot_week_30m_moon8",
        "moon_three_cycle_combo",
        "inner_planet_combo",
    ]
    present = [by_name[name] for name in preferred if name in by_name]
    if len(present) >= 2:
        ensembles["ensemble_structured_mixed"] = sigmoid(np.mean(np.vstack([logit(score) for score in present]), axis=0))
    return ensembles


def flatten(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        for result in record["results"]:
            best_top = max(result["test"]["top_coverage"], key=lambda item: item["lift"])
            valid_thresholds = [
                item for item in result["test"]["validation_thresholds"] if not math.isnan(item["precision"])
            ]
            valid_raw_thresholds = [
                item for item in result["test"]["raw_validation_thresholds"] if not math.isnan(item["precision"])
            ]
            best_threshold = max(valid_thresholds, key=lambda item: item["lift"]) if valid_thresholds else {}
            best_raw_threshold = (
                max(valid_raw_thresholds, key=lambda item: item["lift"]) if valid_raw_thresholds else {}
            )
            rows.append(
                {
                    "threshold_atr": record["threshold_atr"],
                    "horizon_bars": record["horizon_bars"],
                    "model": result["name"],
                    "family": result["family"],
                    "alpha": result.get("alpha", float("nan")),
                    "test_base_rate": record["base_rates"]["test"],
                    "validation_ap": result["validation"]["average_precision"],
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
                    "best_raw_val_threshold_target_coverage": best_raw_threshold.get(
                        "intended_validation_coverage", float("nan")
                    ),
                    "best_raw_val_threshold_test_coverage": best_raw_threshold.get("coverage", float("nan")),
                    "best_raw_val_threshold_precision": best_raw_threshold.get("precision", float("nan")),
                    "best_raw_val_threshold_lift": best_raw_threshold.get("lift", float("nan")),
                    "best_raw_val_threshold_event_recall": best_raw_threshold.get("event_recall", float("nan")),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["threshold_atr", "horizon_bars", "best_ranked_lift", "test_ap"],
        ascending=[True, True, False, False],
    )


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoothed calendar-bin probability models for BTC 5m reversal windows.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-01-01")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--thresholds", default="6,8")
    parser.add_argument("--horizons", default="3,6,12,24")
    parser.add_argument("--alphas", default="10,25,50,100,250,500,1000,2500")
    parser.add_argument("--coverages", default="0.005,0.01,0.02,0.05,0.10")
    parser.add_argument("--families", default="civil,cycle,mixed")
    parser.add_argument("--output", type=Path, default=Path("scripts/ltf_calendar_bin_results.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("scripts/ltf_calendar_bin_summary.csv"))
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    train_end = parse_utc_datetime(args.train_end)
    validation_end = parse_utc_datetime(args.validation_end)
    thresholds = parse_float_list(args.thresholds)
    horizons = parse_int_list(args.horizons)
    alphas = parse_float_list(args.alphas)
    coverages = parse_float_list(args.coverages)
    families = set(parse_str_list(args.families))

    print(f"Loading {args.symbol} 5m from cache...")
    frame = add_atr(load_ohlcv_cached(args.symbol, "5m", start, end, args.cache_dir))
    masks = split_masks(frame["open_time"], train_end, validation_end)
    print(
        f"Rows={len(frame):,} train={int(masks['train'].sum()):,} "
        f"validation={int(masks['validation'].sum()):,} test={int(masks['test'].sum()):,}"
    )
    keys = [key for key in make_keys(frame["open_time"]) if key.family in families]
    print(f"Calendar keys={len(keys)}")

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
                "base_rates": {split: float(np.mean(y[mask])) for split, mask in masks.items()},
                "results": [],
            }
            print(
                f"  Horizon {horizon:>2} bars ({horizon * 5:>3}m): "
                f"test base={record['base_rates']['test']:.3%}"
            )
            selected = []
            for key in keys:
                choice = choose_alpha_for_key(key=key, y=y, masks=masks, alphas=alphas)
                selected.append(choice)
                metrics = evaluate_scores(
                    y=y,
                    masks=masks,
                    raw_scores=choice["raw_scores"],
                    coverages=coverages,
                    pivot_indices=pivots.indices,
                    horizon_bars=horizon,
                )
                metrics.update({"name": key.name, "family": key.family, "alpha": choice["alpha"]})
                record["results"].append(metrics)

            for name, raw_scores in build_ensembles(selected).items():
                metrics = evaluate_scores(
                    y=y,
                    masks=masks,
                    raw_scores=raw_scores,
                    coverages=coverages,
                    pivot_indices=pivots.indices,
                    horizon_bars=horizon,
                )
                metrics.update({"name": name, "family": "ensemble"})
                record["results"].append(metrics)
            records.append(record)

    output = {
        "symbol": args.symbol,
        "timeframe": "5m",
        "start": start,
        "end": end,
        "train_end_exclusive": train_end,
        "validation_end_exclusive": validation_end,
        "alphas": alphas,
        "coverages": coverages,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=json_default), encoding="utf-8")
    summary = flatten(records)
    summary.to_csv(args.summary_csv, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_csv}")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
