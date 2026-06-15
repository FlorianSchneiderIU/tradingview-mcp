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
from pandas.tseries.offsets import DateOffset
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_hierarchical_cascade_backtest import (  # noqa: E402
    add_entry_features,
    trade_summary,
)
from scripts.research_btc_hierarchical_reversal import (  # noqa: E402
    add_indicators,
    json_default,
    load_ohlcv_cached,
    parse_float_list,
    parse_str_list,
    parse_utc_datetime,
)
from scripts.research_btc_ltf_calendar_probability import DEFAULT_CACHE_DIR  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


CASCADE_FEATURES = [
    "cascade_min",
    "cascade_geom",
    "cascade_dispersion",
    "cascade_bottleneck_layer",
    "l4h_pct",
    "l1h_pct",
    "l15_pct",
    "l5_pct",
    "opp_cascade_min",
    "opp_cascade_geom",
    "cascade_min_edge",
    "l4h_edge",
    "l1h_edge",
    "l15_edge",
    "l5_edge",
    "directional_layer_count",
    "all_directional",
    "is_long",
]

PRICE_FEATURES = [
    "range_atr",
    "body_atr",
    "body_abs_atr",
    "close_pos",
    "upper_wick_frac",
    "lower_wick_frac",
    "rejection_wick_frac",
    "opposite_wick_frac",
    "directional_body_atr",
    "directional_close_pos",
    "atr_ratio",
    "rsi_norm",
    "directional_rsi",
    "volume_ratio",
    "close_ema20_atr",
    "close_ema100_atr",
    "directional_close_ema20_atr",
    "directional_close_ema100_atr",
    "ema_20_slope_atr",
    "ema_100_slope_atr",
    "directional_ema20_slope_atr",
    "directional_ema100_slope_atr",
    "return_1_atr",
    "return_3_atr",
    "return_6_atr",
    "return_12_atr",
    "directional_return_1_atr",
    "directional_return_3_atr",
    "directional_return_6_atr",
    "directional_return_12_atr",
    "sweep_6_atr",
    "sweep_12_atr",
    "sweep_24_atr",
    "reclaim_6_atr",
    "reclaim_12_atr",
    "reclaim_24_atr",
]

TIME_FEATURES = [
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "slot_15m_sin",
    "slot_15m_cos",
]

WYCKOFF_FEATURES = [
    "wyckoff_style_spring",
    "wyckoff_style_sos",
    "wyckoff_style_test",
    "window_offset_minutes",
    "spring_sweep_atr",
    "spring_reclaim_atr",
    "spring_range_width_atr",
    "spring_rejection_wick",
    "spring_directional_close_pos",
    "spring_directional_body_atr",
    "spring_volume_ratio",
    "confirmation_bars",
    "confirmation_displacement_atr",
    "confirmation_directional_body_atr",
    "confirmation_volume_ratio",
    "test_depth_atr",
    "test_volume_ratio_to_spring",
]

FEATURE_SETS = {
    "price_only": PRICE_FEATURES + TIME_FEATURES + ["is_long"],
    "cascade_only": CASCADE_FEATURES + TIME_FEATURES,
    "cascade_price": CASCADE_FEATURES + PRICE_FEATURES + TIME_FEATURES,
    "wyckoff_only": PRICE_FEATURES + WYCKOFF_FEATURES + TIME_FEATURES + ["is_long"],
    "cascade_wyckoff": CASCADE_FEATURES + PRICE_FEATURES + WYCKOFF_FEATURES + TIME_FEATURES,
}


def period_start(period: pd.Period) -> pd.Timestamp:
    return pd.Timestamp(period.to_timestamp()).tz_localize("UTC")


def safe_float(value: Any, default: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else default


def add_path_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_entry_features(frame)
    atr = out["atr"].replace(0.0, np.nan)
    candle_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["range_atr"] = (out["high"] - out["low"]) / atr
    out["body_atr"] = (out["close"] - out["open"]) / atr
    out["body_abs_atr"] = out["body_atr"].abs()
    out["rsi_norm"] = out["rsi"] / 100.0
    for lag in [1, 3, 6, 12]:
        out[f"return_{lag}_atr"] = (out["close"] - out["close"].shift(lag)) / atr
    for lookback in [24]:
        out[f"prev_high_{lookback}"] = out["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        out[f"prev_low_{lookback}"] = out["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    out["close_pos"] = (out["close"] - out["low"]) / candle_range
    return out.replace([np.inf, -np.inf], np.nan)


def directional_price_features(row: pd.Series, direction: str) -> dict[str, float]:
    is_long = direction == "long"
    sign = 1.0 if is_long else -1.0
    atr = safe_float(row["atr"], float("nan"))
    if not math.isfinite(atr) or atr <= 0.0:
        atr = 1.0
    close_pos = safe_float(row["close_pos"], 0.5)
    upper_wick = safe_float(row["upper_wick_frac"])
    lower_wick = safe_float(row["lower_wick_frac"])
    features: dict[str, float] = {
        "range_atr": safe_float(row["range_atr"]),
        "body_atr": safe_float(row["body_atr"]),
        "body_abs_atr": safe_float(row["body_abs_atr"]),
        "close_pos": close_pos,
        "upper_wick_frac": upper_wick,
        "lower_wick_frac": lower_wick,
        "rejection_wick_frac": lower_wick if is_long else upper_wick,
        "opposite_wick_frac": upper_wick if is_long else lower_wick,
        "directional_body_atr": sign * safe_float(row["body_atr"]),
        "directional_close_pos": close_pos if is_long else 1.0 - close_pos,
        "atr_ratio": safe_float(row["atr_ratio"], 1.0),
        "rsi_norm": safe_float(row["rsi_norm"], 0.5),
        "directional_rsi": (50.0 - safe_float(row["rsi"], 50.0)) / 50.0 if is_long else (
            safe_float(row["rsi"], 50.0) - 50.0
        ) / 50.0,
        "volume_ratio": safe_float(row["volume_ratio"], 1.0),
        "close_ema20_atr": safe_float(row["close_ema20_atr"]),
        "close_ema100_atr": safe_float(row["close_ema100_atr"]),
        "directional_close_ema20_atr": sign * safe_float(row["close_ema20_atr"]),
        "directional_close_ema100_atr": sign * safe_float(row["close_ema100_atr"]),
        "ema_20_slope_atr": safe_float(row["ema_20_slope_atr"]),
        "ema_100_slope_atr": safe_float(row["ema_100_slope_atr"]),
        "directional_ema20_slope_atr": sign * safe_float(row["ema_20_slope_atr"]),
        "directional_ema100_slope_atr": sign * safe_float(row["ema_100_slope_atr"]),
    }
    for lag in [1, 3, 6, 12]:
        value = safe_float(row[f"return_{lag}_atr"])
        features[f"return_{lag}_atr"] = value
        features[f"directional_return_{lag}_atr"] = sign * value
    for lookback in [6, 12, 24]:
        if is_long:
            sweep = (safe_float(row[f"prev_low_{lookback}"], safe_float(row["low"])) - safe_float(row["low"])) / atr
            reclaim = (safe_float(row["close"]) - safe_float(row[f"prev_low_{lookback}"], safe_float(row["close"]))) / atr
        else:
            sweep = (safe_float(row["high"]) - safe_float(row[f"prev_high_{lookback}"], safe_float(row["high"]))) / atr
            reclaim = (safe_float(row[f"prev_high_{lookback}"], safe_float(row["close"])) - safe_float(row["close"])) / atr
        features[f"sweep_{lookback}_atr"] = sweep
        features[f"reclaim_{lookback}_atr"] = reclaim
    return features


def time_features(timestamp: pd.Timestamp) -> dict[str, float]:
    ts = pd.Timestamp(timestamp)
    minute_of_day = ts.hour * 60 + ts.minute
    tod = 2.0 * np.pi * minute_of_day / 1440.0
    dow = 2.0 * np.pi * ts.dayofweek / 7.0
    month = 2.0 * np.pi * (ts.month - 1) / 12.0
    slot = (ts.minute % 15) // 5
    slot_angle = 2.0 * np.pi * slot / 3.0
    return {
        "tod_sin": float(np.sin(tod)),
        "tod_cos": float(np.cos(tod)),
        "dow_sin": float(np.sin(dow)),
        "dow_cos": float(np.cos(dow)),
        "month_sin": float(np.sin(month)),
        "month_cos": float(np.cos(month)),
        "slot_15m_sin": float(np.sin(slot_angle)),
        "slot_15m_cos": float(np.cos(slot_angle)),
    }


def cascade_features(item: Any, cascade_direction: str) -> dict[str, float]:
    opposite = "high" if cascade_direction == "low" else "low"
    layer_values = np.asarray(
        [float(getattr(item, f"{prefix}_{cascade_direction}_pct")) for prefix in ["l4h", "l1h", "l15", "l5"]],
        dtype=float,
    )
    opposite_values = np.asarray(
        [float(getattr(item, f"{prefix}_{opposite}_pct")) for prefix in ["l4h", "l1h", "l15", "l5"]],
        dtype=float,
    )
    edges = layer_values - opposite_values
    return {
        "cascade_min": float(getattr(item, f"{cascade_direction}_cascade_min")),
        "cascade_geom": float(getattr(item, f"{cascade_direction}_cascade_geom")),
        "cascade_dispersion": float(np.std(layer_values)),
        "cascade_bottleneck_layer": float(np.argmin(layer_values)) / 3.0,
        "l4h_pct": float(layer_values[0]),
        "l1h_pct": float(layer_values[1]),
        "l15_pct": float(layer_values[2]),
        "l5_pct": float(layer_values[3]),
        "opp_cascade_min": float(getattr(item, f"{opposite}_cascade_min")),
        "opp_cascade_geom": float(getattr(item, f"{opposite}_cascade_geom")),
        "cascade_min_edge": float(
            getattr(item, f"{cascade_direction}_cascade_min") - getattr(item, f"{opposite}_cascade_min")
        ),
        "l4h_edge": float(edges[0]),
        "l1h_edge": float(edges[1]),
        "l15_edge": float(edges[2]),
        "l5_edge": float(edges[3]),
        "directional_layer_count": float(np.sum(edges >= 0.0)) / 4.0,
        "all_directional": float(bool(getattr(item, f"{cascade_direction}_all_directional"))),
        "is_long": float(cascade_direction == "low"),
    }


def simulate_trade_multi_rr(
    frame: pd.DataFrame,
    *,
    anchor_idx: int,
    entry_idx: int,
    direction: str,
    rr_values: list[float],
    max_hold_bars: int,
    stop_buffer_atr: float,
    min_risk_pct: float,
    cost_bps_round_trip: float,
) -> dict[float, dict[str, Any]] | None:
    if entry_idx >= len(frame):
        return None
    anchor = frame.iloc[anchor_idx]
    entry = float(frame["open"].iloc[entry_idx])
    atr = float(anchor["atr"])
    if not math.isfinite(atr) or atr <= 0.0:
        return None
    if direction == "long":
        stop = float(anchor["low"]) - stop_buffer_atr * atr
        stop = min(stop, entry * (1.0 - min_risk_pct))
        risk = entry - stop
        targets = {rr: entry + rr * risk for rr in rr_values}
    else:
        stop = float(anchor["high"]) + stop_buffer_atr * atr
        stop = max(stop, entry * (1.0 + min_risk_pct))
        risk = stop - entry
        targets = {rr: entry - rr * risk for rr in rr_values}
    if not math.isfinite(risk) or risk <= 0.0:
        return None

    cost_r = (cost_bps_round_trip / 10_000.0) * entry / risk
    end_idx = min(len(frame) - 1, entry_idx + max_hold_bars - 1)
    unresolved = set(rr_values)
    results: dict[float, dict[str, Any]] = {}
    mfe_r = 0.0
    mae_r = 0.0
    for cursor in range(entry_idx, end_idx + 1):
        high = float(frame["high"].iloc[cursor])
        low = float(frame["low"].iloc[cursor])
        if direction == "long":
            mfe_r = max(mfe_r, (high - entry) / risk)
            mae_r = max(mae_r, (entry - low) / risk)
            hit_stop = low <= stop
        else:
            mfe_r = max(mfe_r, (entry - low) / risk)
            mae_r = max(mae_r, (high - entry) / risk)
            hit_stop = high >= stop
        if hit_stop:
            for rr in unresolved:
                results[rr] = {
                    "result_r": -1.0 - cost_r,
                    "exit_idx": cursor,
                    "exit_time": pd.Timestamp(frame["close_time"].iloc[cursor]).tz_convert("UTC"),
                    "exit_reason": "stop",
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "cost_r": cost_r,
                }
            unresolved.clear()
            break
        for rr in list(unresolved):
            hit_target = high >= targets[rr] if direction == "long" else low <= targets[rr]
            if hit_target:
                results[rr] = {
                    "result_r": rr - cost_r,
                    "exit_idx": cursor,
                    "exit_time": pd.Timestamp(frame["close_time"].iloc[cursor]).tz_convert("UTC"),
                    "exit_reason": "target",
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "cost_r": cost_r,
                }
                unresolved.remove(rr)
        if not unresolved:
            break

    if unresolved:
        exit_price = float(frame["close"].iloc[end_idx])
        timeout_r = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
        for rr in unresolved:
            results[rr] = {
                "result_r": timeout_r - cost_r,
                "exit_idx": end_idx,
                "exit_time": pd.Timestamp(frame["close_time"].iloc[end_idx]).tz_convert("UTC"),
                "exit_reason": "timeout",
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "cost_r": cost_r,
            }
    return results


def build_path_candidates(
    cascade: pd.DataFrame,
    frame_5m: pd.DataFrame,
    *,
    candidate_start: pd.Timestamp,
    minimum_cascade: float,
    min_risk_pcts: list[float],
    rr_values: list[float],
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = cascade[pd.to_datetime(cascade["target_5m"], utc=True) >= candidate_start]
    for cascade_direction, direction in [("low", "long"), ("high", "short")]:
        selected = eligible[eligible[f"{cascade_direction}_cascade_min"] >= minimum_cascade]
        print(f"  {direction}: {len(selected):,} cascade windows")
        for item in selected.itertuples(index=False):
            anchor_idx = int(item.target_idx)
            entry_idx = anchor_idx + 1
            if anchor_idx < 24 or entry_idx >= len(frame_5m):
                continue
            anchor = frame_5m.iloc[anchor_idx]
            entry_time = pd.Timestamp(frame_5m["open_time"].iloc[entry_idx]).tz_convert("UTC")
            base = {
                "signal_idx": anchor_idx,
                "entry_idx": entry_idx,
                "target_time": pd.Timestamp(item.target_5m).tz_convert("UTC"),
                "decision_time": entry_time,
                "direction": direction,
                "cascade_direction": cascade_direction,
                "nested_truth": bool(getattr(item, f"nested_{cascade_direction}_truth")),
                **cascade_features(item, cascade_direction),
                **directional_price_features(anchor, direction),
                **time_features(entry_time),
            }
            for min_risk_pct in min_risk_pcts:
                candidate = {**base, "min_risk_pct": float(min_risk_pct)}
                outcomes = simulate_trade_multi_rr(
                    frame_5m,
                    anchor_idx=anchor_idx,
                    entry_idx=entry_idx,
                    direction=direction,
                    rr_values=rr_values,
                    max_hold_bars=max_hold_bars,
                    stop_buffer_atr=stop_buffer_atr,
                    min_risk_pct=min_risk_pct,
                    cost_bps_round_trip=cost_bps_round_trip,
                )
                for rr in rr_values:
                    key = f"{rr:g}"
                    trade = outcomes.get(rr) if outcomes is not None else None
                    if trade is None:
                        candidate[f"result_r_{key}"] = np.nan
                        candidate[f"exit_idx_{key}"] = np.nan
                        candidate[f"exit_time_{key}"] = pd.NaT
                        candidate[f"exit_reason_{key}"] = ""
                        candidate[f"mfe_r_{key}"] = np.nan
                        candidate[f"mae_r_{key}"] = np.nan
                        candidate[f"cost_r_{key}"] = np.nan
                    else:
                        candidate[f"result_r_{key}"] = float(trade["result_r"])
                        candidate[f"exit_idx_{key}"] = int(trade["exit_idx"])
                        candidate[f"exit_time_{key}"] = trade["exit_time"]
                        candidate[f"exit_reason_{key}"] = str(trade["exit_reason"])
                        candidate[f"mfe_r_{key}"] = float(trade["mfe_r"])
                        candidate[f"mae_r_{key}"] = float(trade["mae_r"])
                        candidate[f"cost_r_{key}"] = float(trade["cost_r"])
                rows.append(candidate)
    out = pd.DataFrame(rows)
    out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True)
    for rr in rr_values:
        key = f"{rr:g}"
        out[f"exit_time_{key}"] = pd.to_datetime(out[f"exit_time_{key}"], utc=True)
    return out.sort_values(["decision_time", "cascade_min"], ascending=[True, False]).reset_index(drop=True)


def clean_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.loc[:, columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def make_execution_model(kind: str, seed: int) -> Any:
    if kind == "logit":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.15,
                        class_weight="balanced",
                        max_iter=1_500,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=240,
            max_depth=7,
            min_samples_leaf=50,
            max_features=0.7,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {kind}")


def undo_balanced_prior(probability: np.ndarray, positive_rate: float) -> np.ndarray:
    q = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    prior = float(np.clip(positive_rate, 1e-6, 1.0 - 1e-6))
    odds = (q / (1.0 - q)) * (prior / (1.0 - prior))
    return odds / (1.0 + odds)


def rows_to_summary(rows: pd.DataFrame, rr: float) -> dict[str, Any]:
    if rows.empty:
        return trade_summary([])
    key = f"{rr:g}"
    trades = [
        {
            "result_r": float(row[f"result_r_{key}"]),
            "exit_reason": str(row[f"exit_reason_{key}"]),
            "mfe_r": float(row[f"mfe_r_{key}"]),
            "cost_r": float(row[f"cost_r_{key}"]),
        }
        for _, row in rows.iterrows()
    ]
    return trade_summary(trades)


def nonoverlapping_rows(frame: pd.DataFrame, rr: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    key = f"{rr:g}"
    ordered = frame.sort_values(["signal_idx", "score", "cascade_min"], ascending=[True, False, False])
    keep: list[int] = []
    blocked_until = -1
    for idx, row in ordered.iterrows():
        signal_idx = int(row["signal_idx"])
        exit_idx = safe_float(row[f"exit_idx_{key}"], float("nan"))
        if signal_idx <= blocked_until or not math.isfinite(exit_idx):
            continue
        keep.append(idx)
        blocked_until = max(blocked_until, int(exit_idx))
    return ordered.loc[keep].sort_values("decision_time").copy()


def validation_month_fraction(rows: pd.DataFrame, rr: float) -> float:
    if rows.empty:
        return 0.0
    key = f"{rr:g}"
    monthly = rows.assign(month=rows["decision_time"].dt.to_period("M")).groupby("month")[f"result_r_{key}"].sum()
    return float((monthly > 0.0).mean()) if len(monthly) else 0.0


def select_threshold(
    validation: pd.DataFrame,
    *,
    rr: float,
    coverages: list[float],
    min_trades: int,
    min_positive_month_fraction: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for coverage in coverages:
        threshold = float(np.quantile(validation["score"], 1.0 - coverage))
        selected = nonoverlapping_rows(validation[validation["score"] >= threshold], rr)
        summary = rows_to_summary(selected, rr)
        positive_fraction = validation_month_fraction(selected, rr)
        if int(summary["trades"]) < min_trades:
            continue
        if float(summary["net_r"]) <= 0.0 or float(summary["profit_factor"]) <= 1.0:
            continue
        if positive_fraction < min_positive_month_fraction:
            continue
        drawdown = abs(float(summary["max_drawdown_r"]))
        objective = (
            float(summary["net_r"]) / max(drawdown, 5.0)
            + 0.20 * float(summary["avg_r"])
            + 0.10 * positive_fraction
        )
        record = {
            "active": True,
            "threshold": threshold,
            "coverage": coverage,
            "positive_month_fraction": positive_fraction,
            "objective": objective,
            **summary,
        }
        if best is None or objective > float(best["objective"]):
            best = record
    if best is None:
        return {
            "active": False,
            "threshold": float("inf"),
            "coverage": 0.0,
            "positive_month_fraction": 0.0,
            "objective": 0.0,
            **trade_summary([]),
        }
    return best


def resolved_before(frame: pd.DataFrame, rr: float, cutoff: pd.Timestamp) -> pd.Series:
    key = f"{rr:g}"
    return pd.to_datetime(frame[f"exit_time_{key}"], utc=True) < cutoff


def materialize_dynamic_outcome(frame: pd.DataFrame, rr_values: list[float]) -> pd.DataFrame:
    out = frame.copy()
    selected_rr = out["selected_rr"].to_numpy(dtype=float)
    out["selected_result_r"] = np.nan
    out["selected_exit_idx"] = np.nan
    out["selected_exit_time"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["selected_exit_reason"] = ""
    out["selected_mfe_r"] = np.nan
    out["selected_cost_r"] = np.nan
    for rr in rr_values:
        key = f"{rr:g}"
        mask = np.isclose(selected_rr, rr)
        out.loc[mask, "selected_result_r"] = out.loc[mask, f"result_r_{key}"]
        out.loc[mask, "selected_exit_idx"] = out.loc[mask, f"exit_idx_{key}"]
        out.loc[mask, "selected_exit_time"] = out.loc[mask, f"exit_time_{key}"]
        out.loc[mask, "selected_exit_reason"] = out.loc[mask, f"exit_reason_{key}"]
        out.loc[mask, "selected_mfe_r"] = out.loc[mask, f"mfe_r_{key}"]
        out.loc[mask, "selected_cost_r"] = out.loc[mask, f"cost_r_{key}"]
    out["selected_exit_time"] = pd.to_datetime(out["selected_exit_time"], utc=True)
    return out


def nonoverlapping_dynamic_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(["signal_idx", "selected_ev", "cascade_min"], ascending=[True, False, False])
    keep: list[int] = []
    blocked_until = -1
    for idx, row in ordered.iterrows():
        signal_idx = int(row["signal_idx"])
        exit_idx = safe_float(row["selected_exit_idx"], float("nan"))
        if signal_idx <= blocked_until or not math.isfinite(exit_idx):
            continue
        keep.append(idx)
        blocked_until = max(blocked_until, int(exit_idx))
    return ordered.loc[keep].sort_values("decision_time").copy()


def dynamic_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return trade_summary([])
    trades = [
        {
            "result_r": float(row["selected_result_r"]),
            "exit_reason": str(row["selected_exit_reason"]),
            "mfe_r": float(row["selected_mfe_r"]),
            "cost_r": float(row["selected_cost_r"]),
        }
        for _, row in rows.iterrows()
    ]
    return trade_summary(trades)


def dynamic_positive_month_fraction(rows: pd.DataFrame) -> float:
    if rows.empty:
        return 0.0
    monthly = rows.assign(month=rows["decision_time"].dt.to_period("M")).groupby("month")["selected_result_r"].sum()
    return float((monthly > 0.0).mean()) if len(monthly) else 0.0


def select_dynamic_threshold(
    validation: pd.DataFrame,
    *,
    coverages: list[float],
    min_trades: int,
    min_positive_month_fraction: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for coverage in coverages:
        threshold = float(np.quantile(validation["selected_ev"], 1.0 - coverage))
        selected = nonoverlapping_dynamic_rows(validation[validation["selected_ev"] >= threshold])
        summary = dynamic_summary(selected)
        positive_fraction = dynamic_positive_month_fraction(selected)
        if int(summary["trades"]) < min_trades:
            continue
        if float(summary["net_r"]) <= 0.0 or float(summary["profit_factor"]) <= 1.0:
            continue
        if positive_fraction < min_positive_month_fraction:
            continue
        drawdown = abs(float(summary["max_drawdown_r"]))
        objective = (
            float(summary["net_r"]) / max(drawdown, 5.0)
            + 0.20 * float(summary["avg_r"])
            + 0.10 * positive_fraction
        )
        record = {
            "active": True,
            "threshold": threshold,
            "coverage": coverage,
            "positive_month_fraction": positive_fraction,
            "objective": objective,
            **summary,
        }
        if best is None or objective > float(best["objective"]):
            best = record
    if best is None:
        return {
            "active": False,
            "threshold": float("inf"),
            "coverage": 0.0,
            "positive_month_fraction": 0.0,
            "objective": 0.0,
            **trade_summary([]),
        }
    return best


def run_dynamic_walkforward(
    candidates: pd.DataFrame,
    *,
    rr_values: list[float],
    min_risk_pct: float,
    direction_scope: str,
    feature_set: str,
    model_name: str,
    validation_months: int,
    start_month: str,
    end_month: str,
    coverages: list[float],
    min_validation_trades: int,
    min_positive_month_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    columns = FEATURE_SETS[feature_set]
    risk_candidates = candidates[
        np.isclose(candidates["min_risk_pct"].to_numpy(dtype=float), min_risk_pct)
    ].copy()
    if direction_scope != "both":
        risk_candidates = risk_candidates[risk_candidates["direction"] == direction_scope].copy()
    months = pd.period_range(start_month, end_month, freq="M")
    selected_test_parts: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, Any]] = []

    for period in months:
        test_start = period_start(period)
        test_end = period_start(period + 1)
        validation_start = test_start - DateOffset(months=validation_months)
        train_resolved = np.ones(len(risk_candidates), dtype=bool)
        validation_resolved = np.ones(len(risk_candidates), dtype=bool)
        for rr in rr_values:
            train_resolved &= resolved_before(risk_candidates, rr, validation_start).to_numpy()
            validation_resolved &= resolved_before(risk_candidates, rr, test_start).to_numpy()
        train = risk_candidates[
            (risk_candidates["decision_time"] < validation_start) & train_resolved
        ].copy()
        validation = risk_candidates[
            (risk_candidates["decision_time"] >= validation_start)
            & (risk_candidates["decision_time"] < test_start)
            & validation_resolved
        ].copy()
        test = risk_candidates[
            (risk_candidates["decision_time"] >= test_start)
            & (risk_candidates["decision_time"] < test_end)
        ].copy()
        if len(train) < 250 or len(validation) < 50 or test.empty:
            monthly_rows.append(
                {
                    "month": str(period),
                    "active": False,
                    "skip_reason": "insufficient_data",
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                }
            )
            continue

        train_x = clean_matrix(train, columns)
        validation_x = clean_matrix(validation, columns)
        test_x = clean_matrix(test, columns)
        valid_heads = True
        for rr in rr_values:
            key = f"{rr:g}"
            y_train = (train[f"exit_reason_{key}"] == "target").astype(np.int8)
            if y_train.nunique() < 2 or int(y_train.sum()) < 15:
                valid_heads = False
                break
            model = make_execution_model(model_name, seed)
            model.fit(train_x, y_train)
            target_rate = float(y_train.mean())
            validation[f"p_target_{key}"] = undo_balanced_prior(
                model.predict_proba(validation_x)[:, 1],
                target_rate,
            )
            test[f"p_target_{key}"] = undo_balanced_prior(
                model.predict_proba(test_x)[:, 1],
                target_rate,
            )
            validation[f"ev_{key}"] = (
                validation[f"p_target_{key}"] * (rr + 1.0)
                - 1.0
                - validation[f"cost_r_{key}"].to_numpy(dtype=float)
            )
            test[f"ev_{key}"] = (
                test[f"p_target_{key}"] * (rr + 1.0)
                - 1.0
                - test[f"cost_r_{key}"].to_numpy(dtype=float)
            )
        if not valid_heads:
            monthly_rows.append(
                {
                    "month": str(period),
                    "active": False,
                    "skip_reason": "insufficient_target_labels",
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                }
            )
            continue

        ev_columns = [f"ev_{rr:g}" for rr in rr_values]
        rr_array = np.asarray(rr_values, dtype=float)
        validation_choice = np.argmax(validation[ev_columns].to_numpy(dtype=float), axis=1)
        test_choice = np.argmax(test[ev_columns].to_numpy(dtype=float), axis=1)
        validation["selected_rr"] = rr_array[validation_choice]
        test["selected_rr"] = rr_array[test_choice]
        validation["selected_ev"] = validation[ev_columns].to_numpy(dtype=float)[
            np.arange(len(validation)), validation_choice
        ]
        test["selected_ev"] = test[ev_columns].to_numpy(dtype=float)[np.arange(len(test)), test_choice]
        validation = materialize_dynamic_outcome(validation, rr_values)
        test = materialize_dynamic_outcome(test, rr_values)
        selection = select_dynamic_threshold(
            validation,
            coverages=coverages,
            min_trades=min_validation_trades,
            min_positive_month_fraction=min_positive_month_fraction,
        )
        if bool(selection["active"]):
            selected_test = test[test["selected_ev"] >= float(selection["threshold"])].copy()
            selected_test["month"] = str(period)
            selected_test["selected_threshold"] = float(selection["threshold"])
            selected_test["selected_coverage"] = float(selection["coverage"])
            selected_test_parts.append(selected_test)
        else:
            selected_test = test.iloc[:0].copy()
        test_summary = dynamic_summary(nonoverlapping_dynamic_rows(selected_test))
        monthly_rows.append(
            {
                "month": str(period),
                "active": bool(selection["active"]),
                "skip_reason": "" if bool(selection["active"]) else "validation_no_edge",
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "threshold": selection["threshold"],
                "coverage": selection["coverage"],
                "validation_positive_month_fraction": selection["positive_month_fraction"],
                **{f"validation_{name}": value for name, value in selection.items() if name != "active"},
                **{f"test_{name}": value for name, value in test_summary.items()},
            }
        )

    selected = (
        pd.concat(selected_test_parts, ignore_index=True)
        if selected_test_parts
        else risk_candidates.iloc[:0].copy()
    )
    trades = nonoverlapping_dynamic_rows(selected)
    summary = dynamic_summary(trades)
    positive_months = 0
    rr_counts: dict[str, int] = {}
    if not trades.empty:
        month_net = trades.groupby("month")["selected_result_r"].sum()
        positive_months = int((month_net > 0.0).sum())
        rr_counts = {
            f"{float(rr):g}": int(count)
            for rr, count in trades["selected_rr"].value_counts().sort_index().items()
        }
    result = {
        "rr": "dynamic",
        "rr_choices": ",".join(f"{rr:g}" for rr in rr_values),
        "rr_counts": rr_counts,
        "min_risk_pct": min_risk_pct,
        "direction_scope": direction_scope,
        "feature_set": feature_set,
        "model": model_name,
        "validation_months": validation_months,
        "months": len(months),
        "active_months": int(sum(bool(row.get("active")) for row in monthly_rows)),
        "positive_months": positive_months,
        **summary,
    }
    return result, pd.DataFrame(monthly_rows), trades


def run_walkforward(
    candidates: pd.DataFrame,
    *,
    rr: float,
    min_risk_pct: float,
    direction_scope: str,
    feature_set: str,
    model_name: str,
    validation_months: int,
    start_month: str,
    end_month: str,
    coverages: list[float],
    min_validation_trades: int,
    min_positive_month_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    columns = FEATURE_SETS[feature_set]
    key = f"{rr:g}"
    risk_candidates = candidates[
        np.isclose(candidates["min_risk_pct"].to_numpy(dtype=float), min_risk_pct)
        & np.isfinite(candidates[f"result_r_{key}"])
    ].copy()
    if direction_scope != "both":
        risk_candidates = risk_candidates[risk_candidates["direction"] == direction_scope].copy()
    months = pd.period_range(start_month, end_month, freq="M")
    scored_test_parts: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, Any]] = []

    for period in months:
        test_start = period_start(period)
        test_end = period_start(period + 1)
        validation_start = test_start - DateOffset(months=validation_months)
        train = risk_candidates[
            (risk_candidates["decision_time"] < validation_start)
            & resolved_before(risk_candidates, rr, validation_start)
        ].copy()
        validation = risk_candidates[
            (risk_candidates["decision_time"] >= validation_start)
            & (risk_candidates["decision_time"] < test_start)
            & resolved_before(risk_candidates, rr, test_start)
        ].copy()
        test = risk_candidates[
            (risk_candidates["decision_time"] >= test_start)
            & (risk_candidates["decision_time"] < test_end)
        ].copy()
        y_train = (train[f"exit_reason_{key}"] == "target").astype(np.int8)
        if (
            len(train) < 250
            or len(validation) < 50
            or test.empty
            or y_train.nunique() < 2
            or int(y_train.sum()) < 20
        ):
            monthly_rows.append(
                {
                    "month": str(period),
                    "active": False,
                    "skip_reason": "insufficient_data",
                    "train_rows": len(train),
                    "train_targets": int(y_train.sum()),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                }
            )
            continue

        model = make_execution_model(model_name, seed)
        model.fit(clean_matrix(train, columns), y_train)
        validation["score"] = model.predict_proba(clean_matrix(validation, columns))[:, 1]
        test["score"] = model.predict_proba(clean_matrix(test, columns))[:, 1]
        selection = select_threshold(
            validation,
            rr=rr,
            coverages=coverages,
            min_trades=min_validation_trades,
            min_positive_month_fraction=min_positive_month_fraction,
        )
        if bool(selection["active"]):
            selected_test = test[test["score"] >= float(selection["threshold"])].copy()
            selected_test["month"] = str(period)
            selected_test["selected_threshold"] = float(selection["threshold"])
            selected_test["selected_coverage"] = float(selection["coverage"])
            scored_test_parts.append(selected_test)
        else:
            selected_test = test.iloc[:0].copy()
        test_selected_nonoverlap = nonoverlapping_rows(selected_test, rr)
        test_summary = rows_to_summary(test_selected_nonoverlap, rr)
        monthly_rows.append(
            {
                "month": str(period),
                "active": bool(selection["active"]),
                "skip_reason": "" if bool(selection["active"]) else "validation_no_edge",
                "train_rows": len(train),
                "train_targets": int(y_train.sum()),
                "train_target_rate": float(y_train.mean()),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "threshold": selection["threshold"],
                "coverage": selection["coverage"],
                "validation_positive_month_fraction": selection["positive_month_fraction"],
                **{f"validation_{name}": value for name, value in selection.items() if name not in {"active"}},
                **{f"test_{name}": value for name, value in test_summary.items()},
            }
        )

    scored = pd.concat(scored_test_parts, ignore_index=True) if scored_test_parts else risk_candidates.iloc[:0].copy()
    trades = nonoverlapping_rows(scored, rr)
    summary = rows_to_summary(trades, rr)
    if not trades.empty:
        month_net = trades.groupby("month")[f"result_r_{key}"].sum()
        positive_months = int((month_net > 0.0).sum())
    else:
        positive_months = 0
    result = {
        "rr": rr,
        "min_risk_pct": min_risk_pct,
        "direction_scope": direction_scope,
        "feature_set": feature_set,
        "model": model_name,
        "validation_months": validation_months,
        "months": len(months),
        "active_months": int(sum(bool(row.get("active")) for row in monthly_rows)),
        "positive_months": positive_months,
        **summary,
    }
    return result, pd.DataFrame(monthly_rows), trades


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-safe monthly walk-forward path model for the BTC hierarchical reversal cascade."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--candidate-start", default="2024-01-01")
    parser.add_argument("--start-month", default="2025-01")
    parser.add_argument("--end-month", default="2026-05")
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--models", default="hgb")
    parser.add_argument("--feature-sets", default="cascade_price,cascade_only,price_only")
    parser.add_argument("--rr-values", default="1.5,2,3,5,10")
    parser.add_argument("--min-risk-pcts", default="0.005,0.01")
    parser.add_argument("--direction-scopes", default="both")
    parser.add_argument("--minimum-cascade", type=float, default=0.60)
    parser.add_argument("--coverages", default="0.02,0.035,0.05,0.075,0.10,0.15,0.20")
    parser.add_argument("--min-validation-trades", type=int, default=8)
    parser.add_argument("--min-positive-validation-month-fraction", type=float, default=0.50)
    parser.add_argument("--max-hold-bars", type=int, default=288)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--cost-bps-round-trip", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dynamic-rr", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--cascade-cache",
        type=Path,
        default=Path("scripts/.cache/astro_cycle/hierarchical_cascade_hgb_price_time.pkl"),
    )
    parser.add_argument(
        "--prior-cascade-cache",
        type=Path,
        default=None,
        help="Optional earlier OOS cascade cache used before --prior-cascade-end.",
    )
    parser.add_argument("--prior-cascade-end", default="2024-01-01")
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=Path("scripts/.cache/astro_cycle/hierarchical_path_candidates.pkl"),
    )
    parser.add_argument("--refresh-candidates", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("scripts/hierarchical_path_walkforward.json"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("scripts/hierarchical_path_walkforward_summary.csv"),
    )
    parser.add_argument(
        "--monthly-csv",
        type=Path,
        default=Path("scripts/hierarchical_path_walkforward_monthly.csv"),
    )
    parser.add_argument(
        "--trades-csv",
        type=Path,
        default=Path("scripts/hierarchical_path_walkforward_trades.csv"),
    )
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    candidate_start = pd.Timestamp(parse_utc_datetime(args.candidate_start))
    models = parse_str_list(args.models)
    feature_sets = parse_str_list(args.feature_sets)
    rr_values = parse_float_list(args.rr_values)
    min_risk_pcts = parse_float_list(args.min_risk_pcts)
    direction_scopes = parse_str_list(args.direction_scopes)
    coverages = parse_float_list(args.coverages)
    for feature_set in feature_sets:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set {feature_set!r}; available={sorted(FEATURE_SETS)}")
    if any(scope not in {"both", "long", "short"} for scope in direction_scopes):
        raise ValueError("Direction scopes must be drawn from: both,long,short")

    if args.candidate_cache.exists() and not args.refresh_candidates:
        print(f"Loading path candidate cache {args.candidate_cache}...")
        cached = pd.read_pickle(args.candidate_cache)
        candidates = cached["candidates"]
        cached_config = cached["config"]
        cached_rrs = {float(value) for value in cached["config"]["rr_values"]}
        cached_risks = {float(value) for value in cached["config"]["min_risk_pcts"]}
        if not set(rr_values).issubset(cached_rrs) or not set(min_risk_pcts).issubset(cached_risks):
            raise ValueError("Candidate cache does not contain every requested RR/min-risk value; use --refresh-candidates.")
        if (
            float(cached_config["minimum_cascade"]) != float(args.minimum_cascade)
            or int(cached_config["max_hold_bars"]) != int(args.max_hold_bars)
            or float(cached_config["cost_bps_round_trip"]) != float(args.cost_bps_round_trip)
        ):
            raise ValueError("Candidate cache configuration differs from this run; use --refresh-candidates.")
    else:
        print("Loading 5m data and hierarchical cascade...")
        base_5m = load_ohlcv_cached(args.symbol, "5m", start, end, args.cache_dir)
        base_5m = base_5m[pd.to_datetime(base_5m["open_time"], utc=True) < pd.Timestamp(end)].reset_index(drop=True)
        frame_5m = add_path_features(add_indicators(base_5m))
        cached_cascade = pd.read_pickle(args.cascade_cache)
        cascade = cached_cascade["cascade"]
        if args.prior_cascade_cache is not None:
            prior_end = pd.Timestamp(parse_utc_datetime(args.prior_cascade_end))
            prior = pd.read_pickle(args.prior_cascade_cache)["cascade"]
            prior_time = pd.to_datetime(prior["target_5m"], utc=True)
            main_time = pd.to_datetime(cascade["target_5m"], utc=True)
            cascade = pd.concat(
                [
                    prior[(prior_time >= candidate_start) & (prior_time < prior_end)],
                    cascade[main_time >= prior_end],
                ],
                ignore_index=True,
            ).sort_values("target_5m").reset_index(drop=True)
            print(
                f"Combined prequential cascades at {prior_end}: "
                f"{len(cascade):,} aligned 5m rows"
            )
        print("Building direct-entry path outcomes...")
        candidates = build_path_candidates(
            cascade,
            frame_5m,
            candidate_start=candidate_start,
            minimum_cascade=args.minimum_cascade,
            min_risk_pcts=min_risk_pcts,
            rr_values=rr_values,
            max_hold_bars=args.max_hold_bars,
            stop_buffer_atr=args.stop_buffer_atr,
            cost_bps_round_trip=args.cost_bps_round_trip,
        )
        args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            {
                "candidates": candidates,
                "config": {
                    "candidate_start": candidate_start,
                    "minimum_cascade": args.minimum_cascade,
                    "rr_values": rr_values,
                    "min_risk_pcts": min_risk_pcts,
                    "max_hold_bars": args.max_hold_bars,
                    "stop_buffer_atr": args.stop_buffer_atr,
                    "cost_bps_round_trip": args.cost_bps_round_trip,
                    "cascade_cache": str(args.cascade_cache),
                    "prior_cascade_cache": str(args.prior_cascade_cache) if args.prior_cascade_cache else None,
                    "prior_cascade_end": args.prior_cascade_end,
                },
            },
            args.candidate_cache,
        )
        print(f"Wrote {args.candidate_cache}")

    candidates["decision_time"] = pd.to_datetime(candidates["decision_time"], utc=True)
    print(f"Path candidates: {len(candidates):,}")
    summary_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for model_name in models:
        for feature_set in feature_sets:
            for min_risk_pct in min_risk_pcts:
                for direction_scope in direction_scopes:
                    if args.dynamic_rr:
                        label = (
                            f"{model_name}/{feature_set}/risk={min_risk_pct:g}/"
                            f"direction={direction_scope}/rr=dynamic"
                        )
                        print(f"Running {label}...")
                        summary, monthly, trades = run_dynamic_walkforward(
                            candidates,
                            rr_values=rr_values,
                            min_risk_pct=min_risk_pct,
                            direction_scope=direction_scope,
                            feature_set=feature_set,
                            model_name=model_name,
                            validation_months=args.validation_months,
                            start_month=args.start_month,
                            end_month=args.end_month,
                            coverages=coverages,
                            min_validation_trades=args.min_validation_trades,
                            min_positive_month_fraction=args.min_positive_validation_month_fraction,
                            seed=args.seed,
                        )
                        summary_rows.append(summary)
                        monthly.insert(0, "experiment", label)
                        trades.insert(0, "experiment", label)
                        monthly_frames.append(monthly)
                        trade_frames.append(trades)
                    else:
                        for rr in rr_values:
                            label = (
                                f"{model_name}/{feature_set}/risk={min_risk_pct:g}/"
                                f"direction={direction_scope}/rr={rr:g}"
                            )
                            print(f"Running {label}...")
                            summary, monthly, trades = run_walkforward(
                                candidates,
                                rr=rr,
                                min_risk_pct=min_risk_pct,
                                direction_scope=direction_scope,
                                feature_set=feature_set,
                                model_name=model_name,
                                validation_months=args.validation_months,
                                start_month=args.start_month,
                                end_month=args.end_month,
                                coverages=coverages,
                                min_validation_trades=args.min_validation_trades,
                                min_positive_month_fraction=args.min_positive_validation_month_fraction,
                                seed=args.seed,
                            )
                            summary_rows.append(summary)
                            monthly.insert(0, "experiment", label)
                            trades.insert(0, "experiment", label)
                            monthly_frames.append(monthly)
                            trade_frames.append(trades)

    summary_table = pd.DataFrame(summary_rows).sort_values(
        ["net_r", "profit_factor", "trades"],
        ascending=[False, False, False],
    )
    monthly_table = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    trades_table = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    args.summary_csv.write_text(summary_table.to_csv(index=False), encoding="utf-8")
    args.monthly_csv.write_text(monthly_table.to_csv(index=False), encoding="utf-8")
    args.trades_csv.write_text(trades_table.to_csv(index=False), encoding="utf-8")

    result = {
        "config": {
            "symbol": args.symbol,
            "candidate_start": candidate_start,
            "start_month": args.start_month,
            "end_month": args.end_month,
            "validation_months": args.validation_months,
            "models": models,
            "feature_sets": feature_sets,
            "rr_values": rr_values,
            "min_risk_pcts": min_risk_pcts,
            "direction_scopes": direction_scopes,
            "minimum_cascade": args.minimum_cascade,
            "coverages": coverages,
            "min_validation_trades": args.min_validation_trades,
            "min_positive_validation_month_fraction": args.min_positive_validation_month_fraction,
            "max_hold_bars": args.max_hold_bars,
            "cost_bps_round_trip": args.cost_bps_round_trip,
            "dynamic_rr": args.dynamic_rr,
        },
        "candidate_rows": len(candidates),
        "experiments": summary_table.to_dict(orient="records"),
    }
    args.output_json.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.monthly_csv}")
    print(f"Wrote {args.trades_csv}")
    print("\nTop experiments:")
    print(summary_table.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
