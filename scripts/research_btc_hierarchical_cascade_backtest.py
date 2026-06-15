from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_hierarchical_reversal import (  # noqa: E402
    FLOOR_RULE,
    TIMEFRAME_SECONDS,
    LayerSpec,
    add_indicators,
    balanced_weights,
    build_features,
    clean_features_from_train,
    default_layers,
    fit_model,
    future_extreme_label,
    horizon_masks,
    json_default,
    load_ohlcv_cached,
    make_model,
    parent_extreme_indices,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    parse_utc_datetime,
    predict_probability,
    resample_ohlc,
)
from scripts.research_btc_ltf_calendar_probability import DEFAULT_CACHE_DIR  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(reference[np.isfinite(reference)].astype(float))
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=float)
    return np.searchsorted(ref, values, side="right").astype(float) / float(len(ref))


def fit_layer_score_frame(
    frame: pd.DataFrame,
    spec: LayerSpec,
    *,
    train_end: Any,
    validation_end: Any,
    end: Any,
    model_name: str,
    feature_set: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, set[int]]]:
    features, groups = build_features(frame, spec)
    if feature_set not in groups:
        raise ValueError(f"Unknown feature set {feature_set!r}; available={sorted(groups)}")
    columns = groups[feature_set]
    masks = horizon_masks(frame["open_time"], spec.child_tf, 1, train_end, validation_end, end)
    clean = clean_features_from_train(features.loc[:, columns], masks["train"])
    x = clean.to_numpy(dtype=np.float32, copy=False)
    extremes = parent_extreme_indices(frame, spec.parent_tf, spec.children_per_parent)

    score_frame = pd.DataFrame(
        {
            "predictor_time": pd.to_datetime(frame["open_time"], utc=True),
            "target_time": pd.to_datetime(frame["open_time"], utc=True).shift(-1),
        }
    )
    diagnostics: dict[str, Any] = {
        "layer": spec.name,
        "child_tf": spec.child_tf,
        "parent_tf": spec.parent_tf,
        "feature_set": feature_set,
        "feature_count": len(columns),
        "rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "directions": {},
    }
    extreme_time_sets: dict[str, set[int]] = {}

    for direction in ["low", "high"]:
        y = future_extreme_label(len(frame), extremes[direction], 1)
        model = make_model(model_name, seed)
        fit_model(model, x[masks["train"]], y[masks["train"]], balanced_weights(y[masks["train"]]))
        raw = predict_probability(model, x)
        train_raw = raw[masks["train"]]
        pct = percentile_against(train_raw, raw)
        score_frame[f"{direction}_raw"] = raw
        score_frame[f"{direction}_pct"] = pct
        diagnostics["directions"][direction] = {
            "train_base_rate": float(np.mean(y[masks["train"]])),
            "validation_base_rate": float(np.mean(y[masks["validation"]])),
            "test_base_rate": float(np.mean(y[masks["test"]])),
        }
        extreme_times = pd.to_datetime(frame.loc[extremes[direction], "open_time"], utc=True)
        extreme_time_sets[direction] = set(extreme_times.astype("int64").tolist())

    score_frame = score_frame.dropna(subset=["target_time"]).reset_index(drop=True)
    return score_frame, diagnostics, extreme_time_sets


def lookup_scores(
    target_times: pd.Series,
    score_frame: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    lookup = score_frame[
        ["target_time", "low_raw", "low_pct", "high_raw", "high_pct"]
    ].rename(
        columns={
            "target_time": f"{prefix}_target_time",
            "low_raw": f"{prefix}_low_raw",
            "low_pct": f"{prefix}_low_pct",
            "high_raw": f"{prefix}_high_raw",
            "high_pct": f"{prefix}_high_pct",
        }
    )
    left = pd.DataFrame({f"{prefix}_target_time": pd.to_datetime(target_times, utc=True)})
    return left.merge(lookup, on=f"{prefix}_target_time", how="left")


def build_cascade_frame(
    frame_5m: pd.DataFrame,
    score_frames: dict[str, pd.DataFrame],
    extreme_sets: dict[str, dict[str, set[int]]],
    *,
    train_end: Any,
    validation_end: Any,
    end: Any,
) -> pd.DataFrame:
    target_5m = pd.to_datetime(frame_5m["open_time"], utc=True).shift(-1)
    out = pd.DataFrame(
        {
            "predictor_idx": np.arange(len(frame_5m), dtype=np.int32),
            "predictor_time": pd.to_datetime(frame_5m["open_time"], utc=True),
            "target_idx": np.arange(len(frame_5m), dtype=np.int32) + 1,
            "target_5m": target_5m,
        }
    ).dropna(subset=["target_5m"])
    out = out[out["target_5m"] < pd.Timestamp(end)].reset_index(drop=True)
    out["target_15m"] = out["target_5m"].dt.floor(FLOOR_RULE["15m"])
    out["target_1h"] = out["target_5m"].dt.floor(FLOOR_RULE["1h"])
    out["target_4h"] = out["target_5m"].dt.floor(FLOOR_RULE["4h"])

    mappings = [
        ("l5", "target_5m", score_frames["5m_to_15m"]),
        ("l15", "target_15m", score_frames["15m_to_1h"]),
        ("l1h", "target_1h", score_frames["1h_to_4h"]),
        ("l4h", "target_4h", score_frames["4h_to_1d"]),
    ]
    for prefix, time_column, score_frame in mappings:
        aligned = lookup_scores(out[time_column], score_frame, prefix)
        for column in aligned.columns:
            if column.endswith("_target_time"):
                continue
            out[column] = aligned[column].to_numpy()

    target_ns = {
        "l5": out["target_5m"].astype("int64"),
        "l15": out["target_15m"].astype("int64"),
        "l1h": out["target_1h"].astype("int64"),
        "l4h": out["target_4h"].astype("int64"),
    }
    layer_names = {
        "l5": "5m_to_15m",
        "l15": "15m_to_1h",
        "l1h": "1h_to_4h",
        "l4h": "4h_to_1d",
    }
    for direction in ["low", "high"]:
        truth_columns = []
        for prefix in ["l4h", "l1h", "l15", "l5"]:
            column = f"{prefix}_{direction}_truth"
            out[column] = target_ns[prefix].isin(extreme_sets[layer_names[prefix]][direction]).to_numpy()
            truth_columns.append(column)
        out[f"nested_{direction}_truth"] = out[truth_columns].all(axis=1)

        score_columns = [
            f"l4h_{direction}_pct",
            f"l1h_{direction}_pct",
            f"l15_{direction}_pct",
            f"l5_{direction}_pct",
        ]
        score_values = out[score_columns].to_numpy(dtype=float)
        out[f"{direction}_cascade_min"] = np.nanmin(score_values, axis=1)
        out[f"{direction}_cascade_geom"] = np.exp(
            np.nanmean(np.log(np.clip(score_values, 1e-6, 1.0)), axis=1)
        )
        opposite = "high" if direction == "low" else "low"
        out[f"{direction}_all_directional"] = np.ones(len(out), dtype=bool)
        for prefix in ["l4h", "l1h", "l15", "l5"]:
            out[f"{direction}_all_directional"] &= (
                out[f"{prefix}_{direction}_pct"] >= out[f"{prefix}_{opposite}_pct"]
            )

    target_time = pd.to_datetime(out["target_5m"], utc=True)
    out["split"] = np.select(
        [
            target_time < pd.Timestamp(train_end),
            (target_time >= pd.Timestamp(train_end)) & (target_time < pd.Timestamp(validation_end)),
            (target_time >= pd.Timestamp(validation_end)) & (target_time < pd.Timestamp(end)),
        ],
        ["train", "validation", "test"],
        default="outside",
    )
    required = [
        f"{prefix}_{direction}_pct"
        for prefix in ["l4h", "l1h", "l15", "l5"]
        for direction in ["low", "high"]
    ]
    out = out.dropna(subset=required).reset_index(drop=True)
    return out


def cascade_localization_table(
    cascade: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stages = {
        "4h": ["l4h"],
        "4h_1h": ["l4h", "l1h"],
        "4h_1h_15m": ["l4h", "l1h", "l15"],
        "full": ["l4h", "l1h", "l15", "l5"],
    }
    for split in ["validation", "test"]:
        part = cascade[cascade["split"] == split]
        for direction in ["low", "high"]:
            for stage_name, prefixes in stages.items():
                truth_columns = [f"{prefix}_{direction}_truth" for prefix in prefixes]
                truth = part[truth_columns].all(axis=1).to_numpy(dtype=bool)
                event_count = int(truth.sum())
                for threshold in thresholds:
                    active = np.ones(len(part), dtype=bool)
                    for prefix in prefixes:
                        active &= part[f"{prefix}_{direction}_pct"].to_numpy(dtype=float) >= threshold
                    precision = float(truth[active].mean()) if active.any() else float("nan")
                    rows.append(
                        {
                            "split": split,
                            "stage": stage_name,
                            "direction": direction,
                            "threshold": threshold,
                            "signals": int(active.sum()),
                            "coverage": float(active.mean()),
                            "events": event_count,
                            "precision": precision,
                            "base_rate": float(truth.mean()),
                            "lift": float(precision / truth.mean()) if active.any() and truth.mean() > 0 else float("nan"),
                            "event_recall": float((truth & active).sum() / event_count) if event_count > 0 else float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


def add_entry_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for lookback in [6, 12]:
        out[f"prev_high_{lookback}"] = out["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        out[f"prev_low_{lookback}"] = out["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    candle_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["close_pos"] = (out["close"] - out["low"]) / candle_range
    out["upper_wick_frac"] = (out["high"] - out[["open", "close"]].max(axis=1)) / candle_range
    out["lower_wick_frac"] = (out[["open", "close"]].min(axis=1) - out["low"]) / candle_range
    return out


def entry_setup(frame: pd.DataFrame, anchor_idx: int, direction: str, style: str) -> tuple[int, int] | None:
    if anchor_idx < 12 or anchor_idx + 2 >= len(frame):
        return None
    row = frame.iloc[anchor_idx]
    if style == "direct":
        return anchor_idx, anchor_idx + 1

    if direction == "long":
        rejection = float(row["close_pos"]) >= 0.60 and float(row["lower_wick_frac"]) >= 0.20
        sweep6 = float(row["low"]) < float(row["prev_low_6"]) and float(row["close"]) > float(row["prev_low_6"])
        sweep12 = float(row["low"]) < float(row["prev_low_12"]) and float(row["close"]) > float(row["prev_low_12"])
    else:
        rejection = float(row["close_pos"]) <= 0.40 and float(row["upper_wick_frac"]) >= 0.20
        sweep6 = float(row["high"]) > float(row["prev_high_6"]) and float(row["close"]) < float(row["prev_high_6"])
        sweep12 = float(row["high"]) > float(row["prev_high_12"]) and float(row["close"]) < float(row["prev_high_12"])

    if style == "rejection":
        return (anchor_idx, anchor_idx + 1) if rejection else None
    if style == "sweep6":
        return (anchor_idx, anchor_idx + 1) if sweep6 else None
    if style == "sweep12":
        return (anchor_idx, anchor_idx + 1) if sweep12 else None
    if style == "displacement":
        confirm_idx = anchor_idx + 1
        confirm = frame.iloc[confirm_idx]
        atr = float(row["atr"])
        if not math.isfinite(atr) or atr <= 0:
            return None
        if direction == "long":
            valid = (
                rejection
                and float(confirm["close"]) > float(confirm["open"])
                and float(confirm["close"]) > float(row["high"])
                and (float(confirm["close"]) - float(confirm["open"])) / atr >= 0.25
            )
        else:
            valid = (
                rejection
                and float(confirm["close"]) < float(confirm["open"])
                and float(confirm["close"]) < float(row["low"])
                and (float(confirm["open"]) - float(confirm["close"])) / atr >= 0.25
            )
        return (confirm_idx, confirm_idx + 1) if valid else None
    raise ValueError(f"Unknown entry style: {style}")


def simulate_trade(
    frame: pd.DataFrame,
    anchor_idx: int,
    signal_idx: int,
    entry_idx: int,
    direction: str,
    rr: float,
    max_hold_bars: int,
    stop_buffer_atr: float,
    min_risk_pct: float,
    cost_bps_round_trip: float,
) -> dict[str, Any] | None:
    if entry_idx >= len(frame):
        return None
    anchor = frame.iloc[anchor_idx]
    signal = frame.iloc[signal_idx]
    entry = float(frame["open"].iloc[entry_idx])
    atr = float(anchor["atr"])
    if not math.isfinite(atr) or atr <= 0:
        return None
    if direction == "long":
        stop = min(float(anchor["low"]), float(signal["low"])) - stop_buffer_atr * atr
        stop = min(stop, entry * (1.0 - min_risk_pct))
        risk = entry - stop
        target = entry + rr * risk
    else:
        stop = max(float(anchor["high"]), float(signal["high"])) + stop_buffer_atr * atr
        stop = max(stop, entry * (1.0 + min_risk_pct))
        risk = stop - entry
        target = entry - rr * risk
    if not math.isfinite(risk) or risk <= 0:
        return None

    cost_r = (cost_bps_round_trip / 10_000.0) * entry / risk
    end_idx = min(len(frame) - 1, entry_idx + max_hold_bars - 1)
    exit_idx = end_idx
    exit_reason = "timeout"
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
            exit_idx = cursor
            exit_reason = "stop"
            exit_price = stop
            break
        if hit_target:
            result_r = rr - cost_r
            exit_idx = cursor
            exit_reason = "target"
            exit_price = target
            break
    else:
        if direction == "long":
            result_r = (exit_price - entry) / risk - cost_r
        else:
            result_r = (entry - exit_price) / risk - cost_r

    return {
        "anchor_idx": int(anchor_idx),
        "signal_idx": int(signal_idx),
        "entry_idx": int(entry_idx),
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
        "cost_r": float(cost_r),
    }


def max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    equity = np.cumsum(np.asarray(values, dtype=float))
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity - peaks))


def trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": float("nan"),
            "avg_r": float("nan"),
            "median_r": float("nan"),
            "net_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "targets": 0,
            "stops": 0,
            "timeouts": 0,
            "avg_mfe_r": float("nan"),
            "avg_cost_r": float("nan"),
        }
    values = np.asarray([float(item["result_r"]) for item in trades], dtype=float)
    gains = float(values[values > 0].sum())
    losses = float(values[values < 0].sum())
    return {
        "trades": len(trades),
        "win_rate": float(np.mean(values > 0)),
        "avg_r": float(np.mean(values)),
        "median_r": float(np.median(values)),
        "net_r": float(values.sum()),
        "profit_factor": float(gains / abs(losses)) if losses < 0 else (float("inf") if gains > 0 else 0.0),
        "max_drawdown_r": max_drawdown(values.tolist()),
        "targets": int(sum(item["exit_reason"] == "target" for item in trades)),
        "stops": int(sum(item["exit_reason"] == "stop" for item in trades)),
        "timeouts": int(sum(item["exit_reason"] == "timeout" for item in trades)),
        "avg_mfe_r": float(np.mean([item["mfe_r"] for item in trades])),
        "avg_cost_r": float(np.mean([item["cost_r"] for item in trades])),
    }


def build_trade_candidates(
    cascade: pd.DataFrame,
    frame_5m: pd.DataFrame,
    *,
    min_threshold: float,
    entry_styles: list[str],
    rr_values: list[float],
    min_risk_pcts: list[float],
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, trade_direction in [("low", "long"), ("high", "short")]:
        selected = cascade[
            (cascade["split"].isin(["validation", "test"]))
            & (cascade[f"{direction}_cascade_min"] >= min_threshold)
        ]
        for item in selected.itertuples(index=False):
            anchor_idx = int(item.target_idx)
            base = {
                "split": str(item.split),
                "target_time": item.target_5m,
                "target_idx": anchor_idx,
                "direction": trade_direction,
                "cascade_direction": direction,
                "cascade_min": float(getattr(item, f"{direction}_cascade_min")),
                "cascade_geom": float(getattr(item, f"{direction}_cascade_geom")),
                "all_directional": bool(getattr(item, f"{direction}_all_directional")),
                "nested_truth": bool(getattr(item, f"nested_{direction}_truth")),
            }
            for prefix in ["l4h", "l1h", "l15", "l5"]:
                base[f"{prefix}_pct"] = float(getattr(item, f"{prefix}_{direction}_pct"))
            for style in entry_styles:
                setup = entry_setup(frame_5m, anchor_idx, trade_direction, style)
                if setup is None:
                    continue
                signal_idx, entry_idx = setup
                for min_risk_pct in min_risk_pcts:
                    row = {
                        **base,
                        "entry_style": style,
                        "signal_idx": signal_idx,
                        "entry_idx": entry_idx,
                        "min_risk_pct": min_risk_pct,
                    }
                    for rr in rr_values:
                        trade = simulate_trade(
                            frame_5m,
                            anchor_idx,
                            signal_idx,
                            entry_idx,
                            trade_direction,
                            rr,
                            max_hold_bars,
                            stop_buffer_atr,
                            min_risk_pct,
                            cost_bps_round_trip,
                        )
                        key = f"{rr:g}"
                        if trade is None:
                            row[f"result_r_{key}"] = np.nan
                            row[f"exit_idx_{key}"] = np.nan
                            row[f"exit_reason_{key}"] = ""
                            row[f"mfe_r_{key}"] = np.nan
                            row[f"cost_r_{key}"] = np.nan
                        else:
                            row[f"result_r_{key}"] = trade["result_r"]
                            row[f"exit_idx_{key}"] = trade["exit_idx"]
                            row[f"exit_reason_{key}"] = trade["exit_reason"]
                            row[f"mfe_r_{key}"] = trade["mfe_r"]
                            row[f"cost_r_{key}"] = trade["cost_r"]
                    rows.append(row)
    return pd.DataFrame(rows)


def selected_trade_rows(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    rr: float,
    require_directional: bool,
    direction_scope: str = "both",
) -> list[dict[str, Any]]:
    key = f"{rr:g}"
    selected = candidates[candidates["cascade_min"] >= threshold].copy()
    if direction_scope != "both":
        selected = selected[selected["direction"] == direction_scope]
    if require_directional:
        selected = selected[selected["all_directional"]]
    selected = selected.sort_values(["signal_idx", "cascade_min", "cascade_geom"], ascending=[True, False, False])
    rows: list[dict[str, Any]] = []
    blocked_until = -1
    for _, row in selected.iterrows():
        signal_idx = int(row["signal_idx"])
        if signal_idx <= blocked_until:
            continue
        result_r = float(row[f"result_r_{key}"])
        exit_idx = int(row[f"exit_idx_{key}"])
        if not math.isfinite(result_r) or not math.isfinite(exit_idx):
            continue
        rows.append(
            {
                "signal_time": row["target_time"],
                "signal_idx": signal_idx,
                "exit_idx": exit_idx,
                "direction": row["direction"],
                "entry_style": row["entry_style"],
                "cascade_min": float(row["cascade_min"]),
                "cascade_geom": float(row["cascade_geom"]),
                "nested_truth": bool(row["nested_truth"]),
                "result_r": result_r,
                "exit_reason": str(row[f"exit_reason_{key}"]),
                "mfe_r": float(row[f"mfe_r_{key}"]),
                "cost_r": float(row[f"cost_r_{key}"]),
            }
        )
        blocked_until = max(blocked_until, exit_idx)
    return rows


def tune_trade_variants(
    candidates: pd.DataFrame,
    thresholds: list[float],
    rr_values: list[float],
    entry_styles: list[str],
    min_risk_pcts: list[float],
    min_validation_trades: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    validation = candidates[candidates["split"] == "validation"]
    test = candidates[candidates["split"] == "test"]
    for style in entry_styles:
        validation_style = validation[validation["entry_style"] == style]
        test_style = test[test["entry_style"] == style]
        for min_risk_pct in min_risk_pcts:
            validation_risk = validation_style[validation_style["min_risk_pct"] == min_risk_pct]
            test_risk = test_style[test_style["min_risk_pct"] == min_risk_pct]
            for rr in rr_values:
                for direction_scope in ["both", "long", "short"]:
                    for require_directional in [False, True]:
                        for threshold in thresholds:
                            val_trades = selected_trade_rows(
                                validation_risk,
                                threshold=threshold,
                                rr=rr,
                                require_directional=require_directional,
                                direction_scope=direction_scope,
                            )
                            test_trades = selected_trade_rows(
                                test_risk,
                                threshold=threshold,
                                rr=rr,
                                require_directional=require_directional,
                                direction_scope=direction_scope,
                            )
                            val_summary = trade_summary(val_trades)
                            test_summary = trade_summary(test_trades)
                            dd = abs(float(val_summary["max_drawdown_r"]))
                            objective = (
                                float(val_summary["net_r"]) / max(dd, 10.0)
                                + 0.25 * (0.0 if math.isnan(float(val_summary["avg_r"])) else float(val_summary["avg_r"]))
                            )
                            if int(val_summary["trades"]) < min_validation_trades:
                                objective = -math.inf
                            rows.append(
                                {
                                    "entry_style": style,
                                    "min_risk_pct": min_risk_pct,
                                    "rr": rr,
                                    "threshold": threshold,
                                    "direction_scope": direction_scope,
                                    "require_directional": require_directional,
                                    "objective": objective,
                                    **{f"validation_{key}": value for key, value in val_summary.items()},
                                    **{f"test_{key}": value for key, value in test_summary.items()},
                                }
                            )
    table = pd.DataFrame(rows).sort_values(
        [
            "objective",
            "validation_net_r",
            "validation_profit_factor",
            "entry_style",
            "min_risk_pct",
            "rr",
            "threshold",
            "direction_scope",
            "require_directional",
        ],
        ascending=[False, False, False, True, True, True, True, True, True],
    )
    if table.empty or not np.isfinite(float(table.iloc[0]["objective"])):
        return table, {}, []
    best = table.iloc[0].to_dict()
    selected_test = selected_trade_rows(
        test[
            (test["entry_style"] == best["entry_style"])
            & (test["min_risk_pct"] == best["min_risk_pct"])
        ],
        threshold=float(best["threshold"]),
        rr=float(best["rr"]),
        require_directional=bool(best["require_directional"]),
        direction_scope=str(best["direction_scope"]),
    )
    return table, best, selected_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a 4h -> 1h -> 15m -> 5m BTC reversal cascade.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-01-01")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", choices=["hgb", "logit"], default="hgb")
    parser.add_argument("--feature-set", choices=["price", "time", "price_time"], default="price_time")
    parser.add_argument("--thresholds", default="0.60,0.70,0.75,0.80,0.85,0.90,0.93,0.95")
    parser.add_argument("--entry-styles", default="direct,rejection,sweep6,sweep12,displacement")
    parser.add_argument("--rr-values", default="1.5,2,3,5,10")
    parser.add_argument("--min-risk-pcts", default="0.0025,0.005,0.0075,0.01")
    parser.add_argument("--max-hold-bars", type=int, default=288)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--cost-bps-round-trip", type=float, default=8.0)
    parser.add_argument("--min-validation-trades", type=int, default=15)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--cascade-cache",
        type=Path,
        default=Path("scripts/.cache/astro_cycle/hierarchical_cascade_hgb_price_time.pkl"),
    )
    parser.add_argument("--refresh-cascade", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("scripts/hierarchical_cascade_backtest.json"))
    parser.add_argument("--localization-csv", type=Path, default=Path("scripts/hierarchical_cascade_localization.csv"))
    parser.add_argument("--variants-csv", type=Path, default=Path("scripts/hierarchical_cascade_trade_variants.csv"))
    parser.add_argument("--trades-csv", type=Path, default=Path("scripts/hierarchical_cascade_best_test_trades.csv"))
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    train_end = parse_utc_datetime(args.train_end)
    validation_end = parse_utc_datetime(args.validation_end)
    thresholds = parse_float_list(args.thresholds)
    entry_styles = parse_str_list(args.entry_styles)
    rr_values = parse_float_list(args.rr_values)
    min_risk_pcts = parse_float_list(args.min_risk_pcts)

    print(f"Loading {args.symbol} and building timeframes...")
    base_5m = load_ohlcv_cached(args.symbol, "5m", start, end, args.cache_dir)
    base_5m = base_5m[pd.to_datetime(base_5m["open_time"], utc=True) < pd.Timestamp(end)].reset_index(drop=True)
    frames = {"5m": add_indicators(base_5m)}
    for timeframe in ["15m", "1h", "4h"]:
        frames[timeframe] = add_indicators(resample_ohlc(base_5m, timeframe))
    frames["5m"] = add_entry_features(frames["5m"])

    layer_diagnostics: dict[str, Any] = {}
    if args.cascade_cache.exists() and not args.refresh_cascade:
        print(f"Loading cascade cache {args.cascade_cache}...")
        cached = pd.read_pickle(args.cascade_cache)
        cascade = cached["cascade"]
        layer_diagnostics = cached.get("layer_diagnostics", {})
    else:
        score_frames: dict[str, pd.DataFrame] = {}
        extreme_sets: dict[str, dict[str, set[int]]] = {}
        layer_by_child = {layer.child_tf: layer for layer in default_layers()}
        for child_tf in ["4h", "1h", "15m", "5m"]:
            spec = layer_by_child[child_tf]
            print(f"Fitting {spec.name} {args.model} {args.feature_set}...")
            score_frame, diagnostics, layer_extremes = fit_layer_score_frame(
                frames[child_tf],
                spec,
                train_end=train_end,
                validation_end=validation_end,
                end=end,
                model_name=args.model,
                feature_set=args.feature_set,
                seed=args.seed,
            )
            score_frames[spec.name] = score_frame
            layer_diagnostics[spec.name] = diagnostics
            extreme_sets[spec.name] = layer_extremes

        print("Aligning strict layer predictions into final 5m candidates...")
        cascade = build_cascade_frame(
            frames["5m"],
            score_frames,
            extreme_sets,
            train_end=train_end,
            validation_end=validation_end,
            end=end,
        )
        args.cascade_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            {"cascade": cascade, "layer_diagnostics": layer_diagnostics},
            args.cascade_cache,
        )
    localization = cascade_localization_table(cascade, thresholds)
    localization.to_csv(args.localization_csv, index=False)
    print(
        f"Cascade rows validation={(cascade['split'] == 'validation').sum():,} "
        f"test={(cascade['split'] == 'test').sum():,}"
    )

    print("Precomputing cascade entry variants and trade outcomes...")
    candidates = build_trade_candidates(
        cascade,
        frames["5m"],
        min_threshold=min(thresholds),
        entry_styles=entry_styles,
        rr_values=rr_values,
        min_risk_pcts=min_risk_pcts,
        max_hold_bars=args.max_hold_bars,
        stop_buffer_atr=args.stop_buffer_atr,
        cost_bps_round_trip=args.cost_bps_round_trip,
    )
    print(
        f"Trade candidates validation={(candidates['split'] == 'validation').sum() if not candidates.empty else 0:,} "
        f"test={(candidates['split'] == 'test').sum() if not candidates.empty else 0:,}"
    )
    variants, best, best_test_trades = tune_trade_variants(
        candidates,
        thresholds,
        rr_values,
        entry_styles,
        min_risk_pcts,
        args.min_validation_trades,
    )
    variants.to_csv(args.variants_csv, index=False)
    pd.DataFrame(best_test_trades).to_csv(args.trades_csv, index=False)

    full_localization = localization[localization["stage"] == "full"].sort_values(
        ["split", "direction", "threshold"]
    )
    result = {
        "config": {
            "symbol": args.symbol,
            "start": start,
            "end": end,
            "train_end": train_end,
            "validation_end": validation_end,
            "model": args.model,
            "feature_set": args.feature_set,
            "thresholds": thresholds,
            "entry_styles": entry_styles,
            "rr_values": rr_values,
            "min_risk_pcts": min_risk_pcts,
            "max_hold_bars": args.max_hold_bars,
            "stop_buffer_atr": args.stop_buffer_atr,
            "cost_bps_round_trip": args.cost_bps_round_trip,
        },
        "layer_diagnostics": layer_diagnostics,
        "cascade_rows": {
            "validation": int((cascade["split"] == "validation").sum()),
            "test": int((cascade["split"] == "test").sum()),
        },
        "full_localization": full_localization.to_dict(orient="records"),
        "best_validation_selected_variant": best,
        "best_test_summary": trade_summary(best_test_trades),
    }
    args.output_json.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.localization_csv}")
    print(f"Wrote {args.variants_csv}")
    print(f"Wrote {args.trades_csv}")
    print("\nBest validation-selected variant:")
    print(json.dumps(best, indent=2, default=json_default))
    print("\nBest variant test summary:")
    print(json.dumps(trade_summary(best_test_trades), indent=2, default=json_default))


if __name__ == "__main__":
    main()
