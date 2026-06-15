from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_astro_cycle_timing import load_bybit_cached  # noqa: E402
from scripts.research_btc_hierarchical_path_walkforward import (  # noqa: E402
    CASCADE_FEATURES,
    PRICE_FEATURES,
    TIME_FEATURES,
    WYCKOFF_FEATURES,
    clean_matrix,
    make_execution_model,
    period_start,
    trade_summary,
    undo_balanced_prior,
)
from scripts.research_btc_hierarchical_cascade_backtest import max_drawdown  # noqa: E402
from scripts.research_btc_hierarchical_reversal import (  # noqa: E402
    json_default,
    parse_float_list,
    parse_str_list,
    parse_utc_datetime,
)
from scripts.research_btc_hierarchical_wyckoff_1m import prepare_1m_features  # noqa: E402
from scripts.research_btc_ltf_calendar_probability import DEFAULT_CACHE_DIR  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


EXCURSION_THRESHOLDS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: str
    partial_rr: float
    runner_rr: float
    partial_fraction: float


POLICIES = [
    PolicySpec("fixed_1p5", "fixed", 1.5, 1.5, 1.0),
    PolicySpec("fixed_2", "fixed", 2.0, 2.0, 1.0),
    PolicySpec("partial_1p5_75_runner_5", "runner", 1.5, 5.0, 0.75),
    PolicySpec("partial_1p5_50_runner_5", "runner", 1.5, 5.0, 0.50),
    PolicySpec("partial_1p5_75_runner_10", "runner", 1.5, 10.0, 0.75),
    PolicySpec("partial_2_75_runner_5", "runner", 2.0, 5.0, 0.75),
    PolicySpec("partial_2_50_runner_10", "runner", 2.0, 10.0, 0.50),
]
POLICY_BY_NAME = {spec.name: spec for spec in POLICIES}


CONTEXT_FEATURES = [
    "daily_range_atr",
    "daily_directional_extreme_proximity",
    "directional_close_vs_daily_vwap_atr",
    "prev_day_same_gap_atr",
    "prev_day_opp_gap_atr",
    "prev_day_same_swept",
    "prev_week_same_gap_atr",
    "prev_week_opp_gap_atr",
    "prev_week_same_swept",
    "session_asia",
    "session_london",
    "session_ny",
    "session_late",
    "active_session_range_atr",
    "active_session_directional_extreme_proximity",
    "directional_close_vs_session_vwap_atr",
    "session_same_gap_atr",
    "session_opp_gap_atr",
]
for window in [15, 60, 240, 1440]:
    CONTEXT_FEATURES.extend(
        [
            f"rolling_range_{window}_atr",
            f"rolling_directional_position_{window}",
            f"directional_return_{window}_atr",
            f"realized_vol_{window}",
            f"trend_efficiency_{window}",
        ]
    )
CONTEXT_FEATURES.extend(
    [
        "range_compression_15_240",
        "range_compression_60_1440",
        "atr_compression_15_240",
        "volume_impulse_5_60",
    ]
)

BASE_WYCKOFF_FEATURES = PRICE_FEATURES + WYCKOFF_FEATURES + TIME_FEATURES + ["is_long"]
FEATURE_SETS = {
    "base_wyckoff": BASE_WYCKOFF_FEATURES,
    "base_combined": CASCADE_FEATURES + BASE_WYCKOFF_FEATURES,
    "rich_price": BASE_WYCKOFF_FEATURES + CONTEXT_FEATURES,
    "rich_combined": CASCADE_FEATURES + BASE_WYCKOFF_FEATURES + CONTEXT_FEATURES,
}


def add_context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["open_time"], utc=True)
    day = ts.dt.floor("D")
    week = (day - pd.to_timedelta(ts.dt.dayofweek, unit="D")).dt.floor("D")
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    volume = out["volume"].replace(0.0, np.nan)
    atr = out["atr"].replace(0.0, np.nan)

    out["day_key"] = day
    out["week_key"] = week
    out["daily_running_high"] = out.groupby(day, sort=False)["high"].cummax()
    out["daily_running_low"] = out.groupby(day, sort=False)["low"].cummin()
    out["daily_vwap"] = (
        (typical * out["volume"]).groupby(day, sort=False).cumsum()
        / out["volume"].groupby(day, sort=False).cumsum().replace(0.0, np.nan)
    )

    daily = (
        out.groupby(day, sort=True)
        .agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"))
        .shift(1)
    )
    out["prev_day_high"] = day.map(daily["day_high"])
    out["prev_day_low"] = day.map(daily["day_low"])
    out["prev_day_close"] = day.map(daily["day_close"])

    weekly = (
        out.groupby(week, sort=True)
        .agg(week_high=("high", "max"), week_low=("low", "min"), week_close=("close", "last"))
        .shift(1)
    )
    out["prev_week_high"] = week.map(weekly["week_high"])
    out["prev_week_low"] = week.map(weekly["week_low"])
    out["prev_week_close"] = week.map(weekly["week_close"])

    hours = ts.dt.hour.to_numpy()
    session_specs = {
        "asia": (0, 8),
        "london": (7, 16),
        "ny": (13, 22),
    }
    for name, (start_hour, end_hour) in session_specs.items():
        active = (hours >= start_hour) & (hours < end_hour)
        high = out["high"].where(active)
        low = out["low"].where(active)
        session_volume = out["volume"].where(active, 0.0)
        session_pv = (typical * out["volume"]).where(active, 0.0)
        out[f"{name}_running_high"] = high.groupby(day, sort=False).cummax().groupby(day, sort=False).ffill()
        out[f"{name}_running_low"] = low.groupby(day, sort=False).cummin().groupby(day, sort=False).ffill()
        out[f"{name}_vwap"] = (
            session_pv.groupby(day, sort=False).cumsum()
            / session_volume.groupby(day, sort=False).cumsum().replace(0.0, np.nan)
        )
        out[f"session_{name}"] = active.astype(float)
    out["session_late"] = ((hours >= 22) | (hours < 0)).astype(float)

    log_return = np.log(out["close"]).diff()
    absolute_move = out["close"].diff().abs()
    for window in [15, 60, 240, 1440]:
        rolling_high = out["high"].rolling(window, min_periods=max(5, window // 4)).max()
        rolling_low = out["low"].rolling(window, min_periods=max(5, window // 4)).min()
        rolling_range = rolling_high - rolling_low
        out[f"rolling_high_{window}"] = rolling_high
        out[f"rolling_low_{window}"] = rolling_low
        out[f"rolling_range_{window}_atr_raw"] = rolling_range / atr
        out[f"rolling_position_{window}"] = (out["close"] - rolling_low) / rolling_range.replace(0.0, np.nan)
        out[f"return_{window}_atr_raw"] = (out["close"] - out["close"].shift(window)) / atr
        out[f"realized_vol_{window}_raw"] = log_return.rolling(
            window,
            min_periods=max(5, window // 4),
        ).std() * math.sqrt(window)
        path_length = absolute_move.rolling(window, min_periods=max(5, window // 4)).sum()
        out[f"trend_efficiency_{window}_raw"] = (
            (out["close"] - out["close"].shift(window)).abs() / path_length.replace(0.0, np.nan)
        )

    out["atr_15"] = out["atr"].rolling(15, min_periods=5).mean()
    out["atr_240"] = out["atr"].rolling(240, min_periods=60).mean()
    out["volume_5"] = volume.rolling(5, min_periods=2).mean()
    out["volume_60"] = volume.rolling(60, min_periods=15).mean()
    return out.replace([np.inf, -np.inf], np.nan)


def context_features_at(frame: pd.DataFrame, idx: int, direction: str) -> dict[str, float]:
    row = frame.iloc[idx]
    sign = 1.0 if direction == "long" else -1.0
    is_long = direction == "long"
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    atr = float(row["atr"])
    if not math.isfinite(atr) or atr <= 0.0:
        atr = 1.0

    daily_high = float(row["daily_running_high"])
    daily_low = float(row["daily_running_low"])
    daily_range = daily_high - daily_low
    daily_position = (close - daily_low) / daily_range if daily_range > 0.0 else 0.5
    prev_same = float(row["prev_day_low"] if is_long else row["prev_day_high"])
    prev_opp = float(row["prev_day_high"] if is_long else row["prev_day_low"])
    week_same = float(row["prev_week_low"] if is_long else row["prev_week_high"])
    week_opp = float(row["prev_week_high"] if is_long else row["prev_week_low"])

    hour = pd.Timestamp(row["open_time"]).hour
    if 0 <= hour < 8:
        session = "asia"
    elif 8 <= hour < 13:
        session = "london"
    elif 13 <= hour < 22:
        session = "ny"
    else:
        session = "ny"
    session_high = float(row[f"{session}_running_high"])
    session_low = float(row[f"{session}_running_low"])
    session_vwap = float(row[f"{session}_vwap"])
    session_range = session_high - session_low
    session_position = (close - session_low) / session_range if session_range > 0.0 else 0.5
    session_same = session_low if is_long else session_high
    session_opp = session_high if is_long else session_low

    features: dict[str, float] = {
        "daily_range_atr": daily_range / atr,
        "daily_directional_extreme_proximity": 1.0 - daily_position if is_long else daily_position,
        "directional_close_vs_daily_vwap_atr": sign * (close - float(row["daily_vwap"])) / atr,
        "prev_day_same_gap_atr": abs(close - prev_same) / atr if math.isfinite(prev_same) else 0.0,
        "prev_day_opp_gap_atr": abs(close - prev_opp) / atr if math.isfinite(prev_opp) else 0.0,
        "prev_day_same_swept": float(
            math.isfinite(prev_same)
            and ((low < prev_same < close) if is_long else (high > prev_same > close))
        ),
        "prev_week_same_gap_atr": abs(close - week_same) / atr if math.isfinite(week_same) else 0.0,
        "prev_week_opp_gap_atr": abs(close - week_opp) / atr if math.isfinite(week_opp) else 0.0,
        "prev_week_same_swept": float(
            math.isfinite(week_same)
            and ((low < week_same < close) if is_long else (high > week_same > close))
        ),
        "session_asia": float(0 <= hour < 8),
        "session_london": float(8 <= hour < 13),
        "session_ny": float(13 <= hour < 22),
        "session_late": float(hour >= 22),
        "active_session_range_atr": session_range / atr if math.isfinite(session_range) else 0.0,
        "active_session_directional_extreme_proximity": (
            (1.0 - session_position if is_long else session_position)
            if math.isfinite(session_position)
            else 0.5
        ),
        "directional_close_vs_session_vwap_atr": (
            sign * (close - session_vwap) / atr if math.isfinite(session_vwap) else 0.0
        ),
        "session_same_gap_atr": abs(close - session_same) / atr if math.isfinite(session_same) else 0.0,
        "session_opp_gap_atr": abs(close - session_opp) / atr if math.isfinite(session_opp) else 0.0,
    }
    for window in [15, 60, 240, 1440]:
        position = float(row[f"rolling_position_{window}"])
        features[f"rolling_range_{window}_atr"] = float(row[f"rolling_range_{window}_atr_raw"])
        features[f"rolling_directional_position_{window}"] = (
            1.0 - position if is_long else position
        ) if math.isfinite(position) else 0.5
        features[f"directional_return_{window}_atr"] = sign * float(row[f"return_{window}_atr_raw"])
        features[f"realized_vol_{window}"] = float(row[f"realized_vol_{window}_raw"])
        features[f"trend_efficiency_{window}"] = float(row[f"trend_efficiency_{window}_raw"])
    range_15 = float(row["rolling_range_15_atr_raw"])
    range_60 = float(row["rolling_range_60_atr_raw"])
    range_240 = float(row["rolling_range_240_atr_raw"])
    range_1440 = float(row["rolling_range_1440_atr_raw"])
    features["range_compression_15_240"] = range_15 / range_240 if range_240 > 0.0 else 0.0
    features["range_compression_60_1440"] = range_60 / range_1440 if range_1440 > 0.0 else 0.0
    atr_15 = float(row["atr_15"])
    atr_240 = float(row["atr_240"])
    features["atr_compression_15_240"] = atr_15 / atr_240 if atr_240 > 0.0 else 0.0
    volume_5 = float(row["volume_5"])
    volume_60 = float(row["volume_60"])
    features["volume_impulse_5_60"] = volume_5 / volume_60 if volume_60 > 0.0 else 0.0
    return {
        key: (float(value) if math.isfinite(float(value)) else 0.0)
        for key, value in features.items()
    }


def policy_column(name: str, field: str) -> str:
    return f"policy_{name}_{field}"


def simulate_excursion_and_policies(
    frame: pd.DataFrame,
    row: pd.Series,
    *,
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> dict[str, Any] | None:
    spring_idx = int(row["spring_idx"])
    entry_idx = int(row["entry_idx"])
    if spring_idx < 0 or entry_idx >= len(frame):
        return None
    direction = str(row["direction"])
    spring = frame.iloc[spring_idx]
    entry = float(frame["open"].iloc[entry_idx])
    atr = float(spring["atr"])
    min_risk_pct = float(row["min_risk_pct"])
    if not math.isfinite(atr) or atr <= 0.0:
        return None
    if direction == "long":
        stop = min(float(spring["low"]) - stop_buffer_atr * atr, entry * (1.0 - min_risk_pct))
        risk = entry - stop
    else:
        stop = max(float(spring["high"]) + stop_buffer_atr * atr, entry * (1.0 + min_risk_pct))
        risk = stop - entry
    if not math.isfinite(risk) or risk <= 0.0:
        return None
    cost_r = (cost_bps_round_trip / 10_000.0) * entry / risk
    end_idx = min(len(frame) - 1, entry_idx + max_hold_bars - 1)
    targets = {
        threshold: entry + threshold * risk if direction == "long" else entry - threshold * risk
        for threshold in EXCURSION_THRESHOLDS
    }
    first_hit: dict[float, int | None] = {threshold: None for threshold in EXCURSION_THRESHOLDS}
    policy_states: dict[str, dict[str, Any]] = {}
    for spec in POLICIES:
        if spec.kind == "runner":
            policy_states[spec.name] = {
                "partial_hit": False,
                "done": False,
                "result_r": math.nan,
                "exit_idx": math.nan,
                "reason": "",
            }

    mfe_r = 0.0
    mae_r = 0.0
    stop_idx: int | None = None
    for cursor in range(entry_idx, end_idx + 1):
        high = float(frame["high"].iloc[cursor])
        low = float(frame["low"].iloc[cursor])
        hit_stop = low <= stop if direction == "long" else high >= stop
        favorable = (high - entry) / risk if direction == "long" else (entry - low) / risk
        adverse = (entry - low) / risk if direction == "long" else (high - entry) / risk
        if hit_stop:
            stop_idx = cursor
            mae_r = max(mae_r, adverse)
            for spec in POLICIES:
                if spec.kind != "runner":
                    continue
                state = policy_states[spec.name]
                if state["done"]:
                    continue
                if state["partial_hit"]:
                    runner_r = 0.0
                    state["result_r"] = (
                        spec.partial_fraction * spec.partial_rr
                        + (1.0 - spec.partial_fraction) * runner_r
                        - cost_r
                    )
                    state["reason"] = "partial_then_be"
                else:
                    state["result_r"] = -1.0 - cost_r
                    state["reason"] = "stop"
                state["exit_idx"] = cursor
                state["done"] = True
            break

        mfe_r = max(mfe_r, favorable)
        mae_r = max(mae_r, adverse)
        for threshold, target in targets.items():
            if first_hit[threshold] is None:
                touched = high >= target if direction == "long" else low <= target
                if touched:
                    first_hit[threshold] = cursor

        for spec in POLICIES:
            if spec.kind != "runner":
                continue
            state = policy_states[spec.name]
            if state["done"]:
                continue
            if not state["partial_hit"]:
                partial_target = targets[spec.partial_rr]
                partial_touched = high >= partial_target if direction == "long" else low <= partial_target
                if partial_touched:
                    state["partial_hit"] = True
                    breakeven_touched = low <= entry if direction == "long" else high >= entry
                    runner_target = targets[spec.runner_rr]
                    runner_touched = high >= runner_target if direction == "long" else low <= runner_target
                    if breakeven_touched:
                        runner_r = 0.0
                        state["reason"] = "partial_then_be"
                    elif runner_touched:
                        runner_r = spec.runner_rr
                        state["reason"] = "runner_target"
                    else:
                        continue
                    state["result_r"] = (
                        spec.partial_fraction * spec.partial_rr
                        + (1.0 - spec.partial_fraction) * runner_r
                        - cost_r
                    )
                    state["exit_idx"] = cursor
                    state["done"] = True
            else:
                breakeven_touched = low <= entry if direction == "long" else high >= entry
                runner_target = targets[spec.runner_rr]
                runner_touched = high >= runner_target if direction == "long" else low <= runner_target
                if breakeven_touched:
                    runner_r = 0.0
                    state["reason"] = "partial_then_be"
                elif runner_touched:
                    runner_r = spec.runner_rr
                    state["reason"] = "runner_target"
                else:
                    continue
                state["result_r"] = (
                    spec.partial_fraction * spec.partial_rr
                    + (1.0 - spec.partial_fraction) * runner_r
                    - cost_r
                )
                state["exit_idx"] = cursor
                state["done"] = True

    label_end_idx = stop_idx if stop_idx is not None else end_idx
    exit_price = float(frame["close"].iloc[label_end_idx])
    timeout_r = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
    for spec in POLICIES:
        if spec.kind != "runner":
            continue
        state = policy_states[spec.name]
        if state["done"]:
            continue
        if state["partial_hit"]:
            runner_r = timeout_r
            state["result_r"] = (
                spec.partial_fraction * spec.partial_rr
                + (1.0 - spec.partial_fraction) * runner_r
                - cost_r
            )
            state["reason"] = "partial_then_timeout"
        else:
            state["result_r"] = timeout_r - cost_r
            state["reason"] = "timeout"
        state["exit_idx"] = label_end_idx
        state["done"] = True

    output: dict[str, Any] = {
        "risk_pct": risk / entry,
        "excursion_cost_r": cost_r,
        "mfe_before_stop_r": min(mfe_r, 20.0),
        "mae_before_stop_r": min(mae_r, 20.0),
        "structural_stop_hit": float(stop_idx is not None),
        "label_end_idx": label_end_idx,
        "label_end_time": pd.Timestamp(frame["close_time"].iloc[label_end_idx]).tz_convert("UTC"),
        "bars_to_label_end": label_end_idx - entry_idx + 1,
    }
    for threshold in EXCURSION_THRESHOLDS:
        key = f"{threshold:g}"
        output[f"hit_{key}r"] = float(first_hit[threshold] is not None)
        output[f"bars_to_{key}r"] = (
            int(first_hit[threshold] - entry_idx + 1) if first_hit[threshold] is not None else math.nan
        )
    for spec in POLICIES:
        if spec.kind == "fixed":
            key = f"{spec.partial_rr:g}"
            output[policy_column(spec.name, "result_r")] = float(row[f"result_r_{key}"])
            output[policy_column(spec.name, "exit_idx")] = int(row[f"exit_idx_{key}"])
            output[policy_column(spec.name, "exit_time")] = pd.Timestamp(row[f"exit_time_{key}"])
            output[policy_column(spec.name, "reason")] = str(row[f"exit_reason_{key}"])
        else:
            state = policy_states[spec.name]
            exit_idx = int(state["exit_idx"])
            output[policy_column(spec.name, "result_r")] = float(state["result_r"])
            output[policy_column(spec.name, "exit_idx")] = exit_idx
            output[policy_column(spec.name, "exit_time")] = pd.Timestamp(
                frame["close_time"].iloc[exit_idx]
            ).tz_convert("UTC")
            output[policy_column(spec.name, "reason")] = str(state["reason"])
    return output


def build_excursion_candidates(
    candidates: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    min_risk_pcts: list[float],
    styles: list[str],
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> pd.DataFrame:
    selected = candidates[
        candidates["wyckoff_style"].isin(styles)
        & candidates["min_risk_pct"].isin(min_risk_pcts)
    ].copy()
    rows: list[dict[str, Any]] = []
    for count, (_, candidate) in enumerate(selected.iterrows(), start=1):
        signal_idx = int(candidate["signal_idx"])
        if signal_idx >= len(frame):
            continue
        path = simulate_excursion_and_policies(
            frame,
            candidate,
            max_hold_bars=max_hold_bars,
            stop_buffer_atr=stop_buffer_atr,
            cost_bps_round_trip=cost_bps_round_trip,
        )
        if path is None:
            continue
        rows.append(
            {
                **candidate.to_dict(),
                **context_features_at(frame, signal_idx, str(candidate["direction"])),
                **path,
            }
        )
        if count % 2_000 == 0:
            print(f"  enriched {count:,}/{len(selected):,} candidates")
    out = pd.DataFrame(rows)
    out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True)
    out["label_end_time"] = pd.to_datetime(out["label_end_time"], utc=True)
    for spec in POLICIES:
        out[policy_column(spec.name, "exit_time")] = pd.to_datetime(
            out[policy_column(spec.name, "exit_time")],
            utc=True,
        )
    return out.sort_values(["decision_time", "cascade_min"], ascending=[True, False]).reset_index(drop=True)


def enforce_monotone_probabilities(frame: pd.DataFrame) -> None:
    previous: np.ndarray | None = None
    for threshold in EXCURSION_THRESHOLDS:
        column = f"p_hit_{threshold:g}r"
        current = frame[column].to_numpy(dtype=float)
        if previous is not None:
            current = np.minimum(previous, current)
            frame[column] = current
        previous = current


def add_policy_expectations(frame: pd.DataFrame, policy_scope: str) -> None:
    cost = frame["excursion_cost_r"].to_numpy(dtype=float)
    for spec in POLICIES:
        p_partial = frame[f"p_hit_{spec.partial_rr:g}r"].to_numpy(dtype=float)
        if spec.kind == "fixed":
            expected = p_partial * spec.partial_rr - (1.0 - p_partial) - cost
        else:
            p_runner = frame[f"p_hit_{spec.runner_rr:g}r"].to_numpy(dtype=float)
            p_runner = np.minimum(p_partial, p_runner)
            expected = (
                -(1.0 - p_partial)
                + (p_partial - p_runner) * (spec.partial_fraction * spec.partial_rr)
                + p_runner
                * (
                    spec.partial_fraction * spec.partial_rr
                    + (1.0 - spec.partial_fraction) * spec.runner_rr
                )
                - cost
            )
        frame[policy_column(spec.name, "expected_r")] = expected
    allowed = POLICIES if policy_scope == "adaptive" else [POLICY_BY_NAME[policy_scope]]
    expectation_columns = [policy_column(spec.name, "expected_r") for spec in allowed]
    values = frame[expectation_columns].to_numpy(dtype=float)
    choice = np.argmax(values, axis=1)
    frame["selected_policy"] = np.asarray([spec.name for spec in allowed], dtype=object)[choice]
    frame["selected_expected_r"] = values[np.arange(len(frame)), choice]
    frame["selected_result_r"] = np.nan
    frame["selected_exit_idx"] = np.nan
    frame["selected_exit_time"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    frame["selected_reason"] = ""
    for spec in POLICIES:
        mask = frame["selected_policy"] == spec.name
        frame.loc[mask, "selected_result_r"] = frame.loc[
            mask, policy_column(spec.name, "result_r")
        ]
        frame.loc[mask, "selected_exit_idx"] = frame.loc[
            mask, policy_column(spec.name, "exit_idx")
        ]
        frame.loc[mask, "selected_exit_time"] = frame.loc[
            mask, policy_column(spec.name, "exit_time")
        ]
        frame.loc[mask, "selected_reason"] = frame.loc[
            mask, policy_column(spec.name, "reason")
        ]
    frame["selected_exit_time"] = pd.to_datetime(frame["selected_exit_time"], utc=True)


def nonoverlapping_policy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(
        ["signal_idx", "selected_expected_r", "cascade_min"],
        ascending=[True, False, False],
    )
    keep: list[int] = []
    blocked_until = -1
    for idx, row in ordered.iterrows():
        signal_idx = int(row["signal_idx"])
        exit_idx = float(row["selected_exit_idx"])
        if signal_idx <= blocked_until or not math.isfinite(exit_idx):
            continue
        keep.append(idx)
        blocked_until = max(blocked_until, int(exit_idx))
    return ordered.loc[keep].sort_values("decision_time").copy()


def policy_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return trade_summary([])
    values = rows["selected_result_r"].to_numpy(dtype=float)
    gains = float(values[values > 0.0].sum())
    losses = float(values[values < 0.0].sum())
    return {
        "trades": len(rows),
        "win_rate": float(np.mean(values > 0.0)),
        "avg_r": float(np.mean(values)),
        "median_r": float(np.median(values)),
        "net_r": float(values.sum()),
        "profit_factor": float(gains / abs(losses)) if losses < 0.0 else (float("inf") if gains > 0 else 0.0),
        "max_drawdown_r": max_drawdown(values.tolist()),
        "targets": int(rows["selected_reason"].isin(["target", "runner_target"]).sum()),
        "stops": int((rows["selected_reason"] == "stop").sum()),
        "timeouts": int(rows["selected_reason"].str.contains("timeout", na=False).sum()),
        "avg_mfe_r": float(rows["mfe_before_stop_r"].mean()),
        "avg_cost_r": float(rows["excursion_cost_r"].mean()),
    }


def positive_month_fraction(rows: pd.DataFrame) -> float:
    if rows.empty:
        return 0.0
    monthly = rows.assign(month=rows["decision_time"].dt.to_period("M")).groupby("month")[
        "selected_result_r"
    ].sum()
    return float((monthly > 0.0).mean()) if len(monthly) else 0.0


def select_policy_threshold(
    validation: pd.DataFrame,
    *,
    coverages: list[float],
    min_trades: int,
    min_positive_month_fraction: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    eligible = validation[validation["selected_expected_r"] > 0.0]
    if eligible.empty:
        return {"active": False, "threshold": float("inf"), "coverage": 0.0, **trade_summary([])}
    for coverage in coverages:
        threshold = float(np.quantile(validation["selected_expected_r"], 1.0 - coverage))
        threshold = max(0.0, threshold)
        selected = nonoverlapping_policy_rows(
            validation[validation["selected_expected_r"] >= threshold]
        )
        summary = policy_summary(selected)
        positive_fraction = positive_month_fraction(selected)
        if int(summary["trades"]) < min_trades:
            continue
        if float(summary["net_r"]) <= 0.0 or float(summary["profit_factor"]) <= 1.0:
            continue
        if positive_fraction < min_positive_month_fraction:
            continue
        objective = (
            float(summary["net_r"]) / max(abs(float(summary["max_drawdown_r"])), 5.0)
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
        return {"active": False, "threshold": float("inf"), "coverage": 0.0, **trade_summary([])}
    return best


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else float("nan")


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")


def run_excursion_walkforward(
    candidates: pd.DataFrame,
    *,
    feature_set: str,
    model_name: str,
    min_risk_pct: float,
    direction_scope: str,
    policy_scope: str,
    validation_months: int,
    start_month: str,
    end_month: str,
    coverages: list[float],
    min_validation_trades: int,
    min_positive_month_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = FEATURE_SETS[feature_set]
    universe = candidates[
        np.isclose(candidates["min_risk_pct"].to_numpy(dtype=float), min_risk_pct)
    ].copy()
    if direction_scope != "both":
        universe = universe[universe["direction"] == direction_scope].copy()

    months = pd.period_range(start_month, end_month, freq="M")
    selected_parts: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for period in months:
        test_start = period_start(period)
        test_end = period_start(period + 1)
        validation_start = test_start - DateOffset(months=validation_months)
        train = universe[
            (universe["decision_time"] < validation_start)
            & (universe["label_end_time"] < validation_start)
        ].copy()
        validation = universe[
            (universe["decision_time"] >= validation_start)
            & (universe["decision_time"] < test_start)
            & (universe["label_end_time"] < test_start)
        ].copy()
        test = universe[
            (universe["decision_time"] >= test_start)
            & (universe["decision_time"] < test_end)
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
        valid = True
        for threshold in EXCURSION_THRESHOLDS:
            key = f"{threshold:g}"
            y_train = train[f"hit_{key}r"].astype(np.int8)
            if y_train.nunique() < 2 or int(y_train.sum()) < 15:
                valid = False
                break
            model = make_execution_model(model_name, seed)
            model.fit(train_x, y_train)
            prior = float(y_train.mean())
            validation[f"p_hit_{key}r"] = undo_balanced_prior(
                model.predict_proba(validation_x)[:, 1],
                prior,
            )
            test[f"p_hit_{key}r"] = undo_balanced_prior(
                model.predict_proba(test_x)[:, 1],
                prior,
            )
            diagnostic_rows.append(
                {
                    "month": str(period),
                    "threshold_r": threshold,
                    "train_rate": prior,
                    "validation_ap": safe_ap(
                        validation[f"hit_{key}r"].to_numpy(dtype=int),
                        validation[f"p_hit_{key}r"].to_numpy(dtype=float),
                    ),
                    "validation_auc": safe_auc(
                        validation[f"hit_{key}r"].to_numpy(dtype=int),
                        validation[f"p_hit_{key}r"].to_numpy(dtype=float),
                    ),
                    "test_ap": safe_ap(
                        test[f"hit_{key}r"].to_numpy(dtype=int),
                        test[f"p_hit_{key}r"].to_numpy(dtype=float),
                    ),
                    "test_auc": safe_auc(
                        test[f"hit_{key}r"].to_numpy(dtype=int),
                        test[f"p_hit_{key}r"].to_numpy(dtype=float),
                    ),
                }
            )
        if not valid:
            monthly_rows.append(
                {
                    "month": str(period),
                    "active": False,
                    "skip_reason": "insufficient_excursion_labels",
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                }
            )
            continue

        enforce_monotone_probabilities(validation)
        enforce_monotone_probabilities(test)
        add_policy_expectations(validation, policy_scope)
        add_policy_expectations(test, policy_scope)
        selection = select_policy_threshold(
            validation,
            coverages=coverages,
            min_trades=min_validation_trades,
            min_positive_month_fraction=min_positive_month_fraction,
        )
        if bool(selection["active"]):
            selected_test = test[
                test["selected_expected_r"] >= float(selection["threshold"])
            ].copy()
            selected_test["month"] = str(period)
            selected_test["selected_threshold"] = float(selection["threshold"])
            selected_test["selected_coverage"] = float(selection["coverage"])
            selected_parts.append(selected_test)
        else:
            selected_test = test.iloc[:0].copy()
        test_summary = policy_summary(nonoverlapping_policy_rows(selected_test))
        monthly_rows.append(
            {
                "month": str(period),
                "active": bool(selection["active"]),
                "skip_reason": "" if bool(selection["active"]) else "validation_no_edge",
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "threshold": selection.get("threshold"),
                "coverage": selection.get("coverage"),
                **{f"validation_{key}": value for key, value in selection.items() if key != "active"},
                **{f"test_{key}": value for key, value in test_summary.items()},
            }
        )

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else universe.iloc[:0].copy()
    trades = nonoverlapping_policy_rows(selected)
    summary = policy_summary(trades)
    policy_counts = (
        {str(name): int(count) for name, count in trades["selected_policy"].value_counts().items()}
        if not trades.empty
        else {}
    )
    positive_months = 0
    if not trades.empty:
        month_net = trades.groupby("month")["selected_result_r"].sum()
        positive_months = int((month_net > 0.0).sum())
    result = {
        "feature_set": feature_set,
        "model": model_name,
        "min_risk_pct": min_risk_pct,
        "direction_scope": direction_scope,
        "policy_scope": policy_scope,
        "validation_months": validation_months,
        "months": len(months),
        "active_months": int(sum(bool(row.get("active")) for row in monthly_rows)),
        "positive_months": positive_months,
        "policy_counts": policy_counts,
        **summary,
    }
    return result, pd.DataFrame(monthly_rows), trades, pd.DataFrame(diagnostic_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Excursion-distribution and partial-runner research for hierarchical 1m Wyckoff entries."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--start-month", default="2025-01")
    parser.add_argument("--end-month", default="2026-05")
    parser.add_argument("--validation-months", type=int, default=12)
    parser.add_argument("--models", default="logit")
    parser.add_argument("--feature-sets", default="rich_combined,rich_price,base_combined")
    parser.add_argument("--direction-scopes", default="both,long,short")
    parser.add_argument("--policy-scopes", default="adaptive")
    parser.add_argument("--styles", default="spring,sos,test")
    parser.add_argument("--min-risk-pcts", default="0.005")
    parser.add_argument("--max-hold-bars", type=int, default=1440)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--cost-bps-round-trip", type=float, default=8.0)
    parser.add_argument("--coverages", default="0.02,0.035,0.05,0.075,0.10,0.15,0.20")
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--min-positive-validation-month-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--wyckoff-candidate-cache",
        type=Path,
        default=Path("scripts/.cache/astro_cycle/hierarchical_wyckoff_1m_candidates.pkl"),
    )
    parser.add_argument(
        "--excursion-cache",
        type=Path,
        default=Path("scripts/.cache/astro_cycle/hierarchical_wyckoff_excursion_candidates.pkl"),
    )
    parser.add_argument("--refresh-excursions", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("scripts/hierarchical_excursion_runner.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("scripts/hierarchical_excursion_runner_summary.csv"))
    parser.add_argument("--monthly-csv", type=Path, default=Path("scripts/hierarchical_excursion_runner_monthly.csv"))
    parser.add_argument("--trades-csv", type=Path, default=Path("scripts/hierarchical_excursion_runner_trades.csv"))
    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        default=Path("scripts/hierarchical_excursion_runner_diagnostics.csv"),
    )
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    models = parse_str_list(args.models)
    feature_sets = parse_str_list(args.feature_sets)
    direction_scopes = parse_str_list(args.direction_scopes)
    policy_scopes = parse_str_list(args.policy_scopes)
    styles = parse_str_list(args.styles)
    min_risk_pcts = parse_float_list(args.min_risk_pcts)
    coverages = parse_float_list(args.coverages)
    for feature_set in feature_sets:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set {feature_set!r}; available={sorted(FEATURE_SETS)}")
    for policy_scope in policy_scopes:
        if policy_scope != "adaptive" and policy_scope not in POLICY_BY_NAME:
            raise ValueError(
                f"Unknown policy scope {policy_scope!r}; "
                f"available={['adaptive', *sorted(POLICY_BY_NAME)]}"
            )

    if args.excursion_cache.exists() and not args.refresh_excursions:
        print(f"Loading excursion cache {args.excursion_cache}...")
        cached = pd.read_pickle(args.excursion_cache)
        candidates = cached["candidates"]
        config = cached["config"]
        if (
            pd.Timestamp(config["start"]) != start
            or pd.Timestamp(config["end"]) != end
            or int(config["max_hold_bars"]) != args.max_hold_bars
            or float(config["stop_buffer_atr"]) != args.stop_buffer_atr
            or float(config["cost_bps_round_trip"]) != args.cost_bps_round_trip
            or not set(styles).issubset(set(config["styles"]))
            or not set(min_risk_pcts).issubset({float(x) for x in config["min_risk_pcts"]})
        ):
            raise ValueError("Excursion cache configuration differs; use --refresh-excursions.")
    else:
        print("Loading 1m history and Wyckoff candidates...")
        raw = load_bybit_cached(args.symbol, "1m", start, end, args.cache_dir)
        frame = add_context_frame(prepare_1m_features(raw, 20))
        source = pd.read_pickle(args.wyckoff_candidate_cache)["candidates"]
        print("Building rich context, excursion labels, and partial-runner outcomes...")
        candidates = build_excursion_candidates(
            source,
            frame,
            min_risk_pcts=min_risk_pcts,
            styles=styles,
            max_hold_bars=args.max_hold_bars,
            stop_buffer_atr=args.stop_buffer_atr,
            cost_bps_round_trip=args.cost_bps_round_trip,
        )
        args.excursion_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            {
                "candidates": candidates,
                "config": {
                    "start": start,
                    "end": end,
                    "styles": styles,
                    "min_risk_pcts": min_risk_pcts,
                    "max_hold_bars": args.max_hold_bars,
                    "stop_buffer_atr": args.stop_buffer_atr,
                    "cost_bps_round_trip": args.cost_bps_round_trip,
                    "context_features": CONTEXT_FEATURES,
                    "policies": [spec.name for spec in POLICIES],
                },
            },
            args.excursion_cache,
        )
        print(f"Wrote {args.excursion_cache}")

    candidates = candidates[
        candidates["wyckoff_style"].isin(styles)
        & candidates["min_risk_pct"].isin(min_risk_pcts)
    ].copy()
    candidates["decision_time"] = pd.to_datetime(candidates["decision_time"], utc=True)
    candidates["label_end_time"] = pd.to_datetime(candidates["label_end_time"], utc=True)
    print(f"Excursion candidates: {len(candidates):,}")

    summaries: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for model_name in models:
        for feature_set in feature_sets:
            for min_risk_pct in min_risk_pcts:
                for direction_scope in direction_scopes:
                    for policy_scope in policy_scopes:
                        label = (
                            f"{model_name}/{feature_set}/risk={min_risk_pct:g}/"
                            f"direction={direction_scope}/policy={policy_scope}"
                        )
                        print(f"Running {label}...")
                        summary, monthly, trades, diagnostics = run_excursion_walkforward(
                            candidates,
                            feature_set=feature_set,
                            model_name=model_name,
                            min_risk_pct=min_risk_pct,
                            direction_scope=direction_scope,
                            policy_scope=policy_scope,
                            validation_months=args.validation_months,
                            start_month=args.start_month,
                            end_month=args.end_month,
                            coverages=coverages,
                            min_validation_trades=args.min_validation_trades,
                            min_positive_month_fraction=args.min_positive_validation_month_fraction,
                            seed=args.seed,
                        )
                        summary["experiment"] = label
                        summaries.append(summary)
                        monthly.insert(0, "experiment", label)
                        trades.insert(0, "experiment", label)
                        diagnostics.insert(0, "experiment", label)
                        monthly_frames.append(monthly)
                        trade_frames.append(trades)
                        diagnostic_frames.append(diagnostics)

    summary_table = pd.DataFrame(summaries).sort_values(
        ["net_r", "profit_factor", "trades"],
        ascending=[False, False, False],
    )
    monthly_table = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    trades_table = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    diagnostics_table = (
        pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    )
    args.summary_csv.write_text(summary_table.to_csv(index=False), encoding="utf-8")
    args.monthly_csv.write_text(monthly_table.to_csv(index=False), encoding="utf-8")
    args.trades_csv.write_text(trades_table.to_csv(index=False), encoding="utf-8")
    args.diagnostics_csv.write_text(diagnostics_table.to_csv(index=False), encoding="utf-8")
    result = {
        "config": {
            "symbol": args.symbol,
            "start": start,
            "end": end,
            "start_month": args.start_month,
            "end_month": args.end_month,
            "validation_months": args.validation_months,
            "models": models,
            "feature_sets": feature_sets,
            "direction_scopes": direction_scopes,
            "policy_scopes": policy_scopes,
            "styles": styles,
            "min_risk_pcts": min_risk_pcts,
            "policies": [spec.name for spec in POLICIES],
            "excursion_thresholds": EXCURSION_THRESHOLDS,
        },
        "candidate_rows": len(candidates),
        "experiments": summary_table.to_dict(orient="records"),
    }
    args.output_json.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.monthly_csv}")
    print(f"Wrote {args.trades_csv}")
    print(f"Wrote {args.diagnostics_csv}")
    print("\nTop experiments:")
    print(summary_table.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
