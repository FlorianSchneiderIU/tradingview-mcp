from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from scripts.channel_state_research.channels import (
    BoundaryPoint,
    LineFit,
    build_body_envelope_points,
    fit_boundary_line,
    pivot_points,
    snapshot_from_lines,
)
from scripts.channel_state_research.data import merge_asof_timeframe_state
from scripts.channel_state_research.swings import Pivot, extract_causal_swings


@dataclass(frozen=True)
class TimeframeFeatureSpec:
    timeframe: str
    reversal_mult: float
    estimator: str = "theil_sen"
    structural_point_count: int = 5
    min_points: int = 3
    body_envelope_lookback: int = 12
    body_envelope_min_separation: int = 2
    body_envelope_min_move_atr: float = 0.1
    touch_epsilon_atr: float = 0.2
    touch_lookback_bars: int = 20
    persistence_lookback_bars: int = 20
    swing_lookback_pivots: int = 8
    ransac_residual_atr: float = 0.75


def build_timeframe_state_frame(
    frame: pd.DataFrame,
    spec: TimeframeFeatureSpec,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    close_times = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    open_values = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float, copy=False)
    high_values = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float, copy=False)
    low_values = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float, copy=False)
    close_values = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    volume_values = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=float, copy=False)
    body_high_values = pd.to_numeric(frame["body_high"], errors="coerce").to_numpy(dtype=float, copy=False)
    body_low_values = pd.to_numeric(frame["body_low"], errors="coerce").to_numpy(dtype=float, copy=False)
    atr_values = pd.to_numeric(frame["atr"], errors="coerce").to_numpy(dtype=float, copy=False)
    return_values = pd.to_numeric(frame["return_1"], errors="coerce").to_numpy(dtype=float, copy=False)
    log_return_values = pd.to_numeric(frame["log_return_1"], errors="coerce").to_numpy(dtype=float, copy=False)
    bar_index_values = pd.to_numeric(frame["bar_index"], errors="coerce").to_numpy(dtype=float, copy=False)

    pivots = extract_causal_swings(frame, spec.reversal_mult)
    upper_wick_points = pivot_points(pivots, "high")
    lower_wick_points = pivot_points(pivots, "low")
    upper_body_points = build_body_envelope_points(
        frame,
        side="upper",
        lookback=spec.body_envelope_lookback,
        min_separation=spec.body_envelope_min_separation,
        min_move_atr=spec.body_envelope_min_move_atr,
    )
    lower_body_points = build_body_envelope_points(
        frame,
        side="lower",
        lookback=spec.body_envelope_lookback,
        min_separation=spec.body_envelope_min_separation,
        min_move_atr=spec.body_envelope_min_move_atr,
    )

    pivot_events = sorted(pivots, key=lambda item: item.confirm_index)
    upper_wick_events = sorted(upper_wick_points, key=lambda item: item.confirm_index)
    lower_wick_events = sorted(lower_wick_points, key=lambda item: item.confirm_index)
    upper_body_events = sorted(upper_body_points, key=lambda item: item.confirm_index)
    lower_body_events = sorted(lower_body_points, key=lambda item: item.confirm_index)

    active_pivots: list[Pivot] = []
    active_upper_wick: list[BoundaryPoint] = []
    active_lower_wick: list[BoundaryPoint] = []
    active_upper_body: list[BoundaryPoint] = []
    active_lower_body: list[BoundaryPoint] = []

    pivot_ptr = 0
    upper_wick_ptr = 0
    lower_wick_ptr = 0
    upper_body_ptr = 0
    lower_body_ptr = 0

    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    previous_valid_state: dict[str, float] | None = None

    for index in range(len(frame)):
        while pivot_ptr < len(pivot_events) and pivot_events[pivot_ptr].confirm_index <= index:
            active_pivots.append(pivot_events[pivot_ptr])
            pivot_ptr += 1
        while upper_wick_ptr < len(upper_wick_events) and upper_wick_events[upper_wick_ptr].confirm_index <= index:
            active_upper_wick.append(upper_wick_events[upper_wick_ptr])
            upper_wick_ptr += 1
        while lower_wick_ptr < len(lower_wick_events) and lower_wick_events[lower_wick_ptr].confirm_index <= index:
            active_lower_wick.append(lower_wick_events[lower_wick_ptr])
            lower_wick_ptr += 1
        while upper_body_ptr < len(upper_body_events) and upper_body_events[upper_body_ptr].confirm_index <= index:
            active_upper_body.append(upper_body_events[upper_body_ptr])
            upper_body_ptr += 1
        while lower_body_ptr < len(lower_body_events) and lower_body_events[lower_body_ptr].confirm_index <= index:
            active_lower_body.append(lower_body_events[lower_body_ptr])
            lower_body_ptr += 1

        atr_value = float(atr_values[index]) if np.isfinite(atr_values[index]) else np.nan
        bar_index_value = float(bar_index_values[index]) if np.isfinite(bar_index_values[index]) else float(index)
        upper_wick_fit = _fit_active_points(active_upper_wick, bar_index_value, atr_value, spec)
        lower_wick_fit = _fit_active_points(active_lower_wick, bar_index_value, atr_value, spec)
        upper_body_fit = _fit_active_points(active_upper_body, bar_index_value, atr_value, spec)
        lower_body_fit = _fit_active_points(active_lower_body, bar_index_value, atr_value, spec)

        wick = snapshot_from_lines(upper_wick_fit, lower_wick_fit, bar_index_value)
        body = snapshot_from_lines(upper_body_fit, lower_body_fit, bar_index_value)

        feature_row: dict[str, float | int | str | pd.Timestamp] = {
            "close_time": pd.Timestamp(close_times.iloc[index]).tz_convert("UTC"),
            "open_tf": float(open_values[index]),
            "high_tf": float(high_values[index]),
            "low_tf": float(low_values[index]),
            "close_tf": float(close_values[index]),
            "volume_tf": float(volume_values[index]),
            "body_high_tf": float(body_high_values[index]),
            "body_low_tf": float(body_low_values[index]),
            "atr_tf": atr_value,
            "return_1_tf": float(return_values[index]) if np.isfinite(return_values[index]) else np.nan,
            "log_return_1_tf": float(log_return_values[index]) if np.isfinite(log_return_values[index]) else np.nan,
            "bar_index_tf": bar_index_value,
            "wick_valid_flag": float(wick.valid),
            "body_valid_flag": float(body.valid),
            "wick_upper_points_used": float(upper_wick_fit.points_used) if upper_wick_fit is not None else 0.0,
            "wick_lower_points_used": float(lower_wick_fit.points_used) if lower_wick_fit is not None else 0.0,
            "body_upper_points_used": float(upper_body_fit.points_used) if upper_body_fit is not None else 0.0,
            "body_lower_points_used": float(lower_body_fit.points_used) if lower_body_fit is not None else 0.0,
        }
        feature_row.update(_family_structure_features("wick", wick, atr_value))
        feature_row.update(_family_structure_features("body", body, atr_value))
        feature_row.update(
            {
                "body_to_wick_width_ratio": _safe_div(body.width, wick.width),
                "midline_gap": body.midline - wick.midline if body.valid and wick.valid else np.nan,
                "slope_difference_body_vs_wick_upper": (
                    float(upper_body_fit.slope) - float(upper_wick_fit.slope)
                    if upper_body_fit is not None and upper_wick_fit is not None
                    else np.nan
                ),
                "slope_difference_body_vs_wick_lower": (
                    float(lower_body_fit.slope) - float(lower_wick_fit.slope)
                    if lower_body_fit is not None and lower_wick_fit is not None
                    else np.nan
                ),
                "upper_wick_boundary": wick.upper_value,
                "lower_wick_boundary": wick.lower_value,
                "upper_body_boundary": body.upper_value,
                "lower_body_boundary": body.lower_value,
                "wick_midline": wick.midline,
                "body_midline": body.midline,
            }
        )
        feature_row.update(
            _excursion_features(
                high_value=float(high_values[index]),
                low_value=float(low_values[index]),
                close_value=float(close_values[index]),
                body_high_value=float(body_high_values[index]),
                body_low_value=float(body_low_values[index]),
                wick=wick,
                body=body,
            )
        )
        feature_row.update(
            _touch_features(
                index=index,
                wick=wick,
                body=body,
                epsilon_atr=spec.touch_epsilon_atr,
                lookback=spec.touch_lookback_bars,
                atr_values=atr_values,
                bar_index_values=bar_index_values,
                high_values=high_values,
                low_values=low_values,
                body_high_values=body_high_values,
                body_low_values=body_low_values,
            )
        )
        feature_row.update(
            _persistence_features(
                index=index,
                wick=wick,
                body=body,
                lookback=spec.persistence_lookback_bars,
                bar_index_values=bar_index_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                body_high_values=body_high_values,
                body_low_values=body_low_values,
            )
        )
        feature_row.update(_swing_features(active_pivots, index, float(close_values[index]), atr_value, spec))
        feature_row.update(_evolution_features(previous_valid_state, wick, body))

        if wick.valid or body.valid:
            previous_valid_state = {
                "wick_width": wick.width,
                "body_width": body.width,
                "upper_body_slope": float(upper_body_fit.slope) if upper_body_fit is not None else np.nan,
                "lower_body_slope": float(lower_body_fit.slope) if lower_body_fit is not None else np.nan,
                "upper_wick_slope": float(upper_wick_fit.slope) if upper_wick_fit is not None else np.nan,
                "lower_wick_slope": float(lower_wick_fit.slope) if lower_wick_fit is not None else np.nan,
                "body_to_wick_width_ratio": feature_row["body_to_wick_width_ratio"],  # type: ignore[index]
            }

        rows.append(feature_row)

    output = pd.DataFrame(rows)
    groups = {
        "state_base": [
            "open_tf",
            "high_tf",
            "low_tf",
            "close_tf",
            "volume_tf",
            "body_high_tf",
            "body_low_tf",
            "atr_tf",
            "return_1_tf",
            "log_return_1_tf",
            "bar_index_tf",
        ],
        "structural": [
            "wick_valid_flag",
            "body_valid_flag",
            "wick_upper_points_used",
            "wick_lower_points_used",
            "body_upper_points_used",
            "body_lower_points_used",
            "upper_wick_slope",
            "lower_wick_slope",
            "wick_slope_gap",
            "wick_width",
            "wick_width_over_atr",
            "wick_mid_slope",
            "upper_body_slope",
            "lower_body_slope",
            "body_slope_gap",
            "body_width",
            "body_width_over_atr",
            "body_mid_slope",
            "body_to_wick_width_ratio",
            "midline_gap",
            "slope_difference_body_vs_wick_upper",
            "slope_difference_body_vs_wick_lower",
            "upper_wick_boundary",
            "lower_wick_boundary",
            "upper_body_boundary",
            "lower_body_boundary",
            "wick_midline",
            "body_midline",
        ],
        "excursion_acceptance": [
            "high_above_upper_wick",
            "close_above_upper_wick",
            "close_above_upper_body",
            "body_above_upper_body",
            "upper_wick_overshoot_size",
            "upper_body_break_size",
            "low_below_lower_wick",
            "close_below_lower_wick",
            "close_below_lower_body",
            "body_below_lower_body",
            "lower_wick_overshoot_size",
            "lower_body_break_size",
        ],
        "touch_interaction": [
            "count_touches_upper_wick",
            "count_touches_lower_wick",
            "count_touches_upper_body",
            "count_touches_lower_body",
            "time_since_last_upper_touch_wick",
            "time_since_last_lower_touch_wick",
            "time_since_last_upper_touch_body",
            "time_since_last_lower_touch_body",
            "consecutive_closes_above_upper_body",
            "consecutive_closes_below_lower_body",
            "consecutive_rejections_upper",
            "consecutive_rejections_lower",
        ],
        "swing_state": [
            "last_pivot_type",
            "bars_since_last_pivot",
            "bars_since_last_pivot_confirm",
            "price_distance_to_last_pivot",
            "price_distance_to_last_pivot_over_atr",
            "size_last_upswing",
            "size_last_downswing",
            "higher_high_flag",
            "lower_low_flag",
            "higher_low_flag",
            "lower_high_flag",
            "pivot_frequency",
            "mean_recent_swing_size",
            "swing_size_std",
        ],
        "channel_evolution": [
            "change_in_wick_width",
            "change_in_body_width",
            "slope_acceleration_upper",
            "slope_acceleration_lower",
            "body_wick_ratio_change",
        ],
    }
    return output, groups


def build_decision_dataset(
    state_frames: dict[str, pd.DataFrame],
    state_groups: dict[str, dict[str, list[str]]],
    *,
    decision_timeframe: str = "1h",
    context_timeframes: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    decision_tf = decision_timeframe
    contexts = context_timeframes or [tf for tf in state_frames if tf != decision_tf]

    decision = _rename_timeframe_frame(state_frames[decision_tf], decision_tf)
    groups: dict[str, list[str]] = {}
    _extend_groups_with_timeframe(groups, state_groups[decision_tf], decision_tf)

    for timeframe in contexts:
        decision = merge_asof_timeframe_state(decision, state_frames[timeframe], timeframe)
        _extend_groups_with_timeframe(groups, state_groups[timeframe], timeframe)

    decision["decision_time"] = decision["close_time"]
    decision["decision_close"] = decision[f"close_tf_{decision_tf}"]

    position_columns: list[str] = []
    for timeframe in [decision_tf, *contexts]:
        position_columns.extend(_add_position_features(decision, timeframe))
    groups["position"] = position_columns

    regime_columns = _add_regime_features(decision, decision_tf)
    groups["regime"] = regime_columns

    confluence_columns = _add_confluence_features(decision, [decision_tf, *contexts])
    groups["confluence"] = confluence_columns

    baseline_columns = [
        f"return_1_tf_{decision_tf}",
        "realized_vol_1h",
        "realized_vol_24h",
        "return_std_7d",
        "volume_zscore_1h",
        "rolling_trend_strength_1h",
        "rolling_autocorrelation_1h",
        "recent_gapless_momentum_1h",
        f"atr_tf_{decision_tf}",
    ]
    groups["baseline_price"] = [column for column in baseline_columns if column in decision.columns]

    model_feature_columns: list[str] = []
    for group_name in [
        "structural",
        "position",
        "excursion_acceptance",
        "touch_interaction",
        "swing_state",
        "channel_evolution",
        "confluence",
        "regime",
    ]:
        model_feature_columns.extend(groups.get(group_name, []))
    groups["all_features"] = list(dict.fromkeys(model_feature_columns))
    return decision, groups


def _rename_timeframe_frame(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    suffix = f"_{timeframe}"
    rename_map = {column: f"{column}{suffix}" for column in frame.columns if column != "close_time"}
    return frame.rename(columns=rename_map)


def _extend_groups_with_timeframe(groups: dict[str, list[str]], timeframe_groups: dict[str, list[str]], timeframe: str) -> None:
    suffix = f"_{timeframe}"
    for group_name, columns in timeframe_groups.items():
        groups.setdefault(group_name, [])
        groups[group_name].extend(f"{column}{suffix}" for column in columns)


def _fit_active_points(
    points: list[BoundaryPoint],
    bar_index_value: float,
    atr_value: float,
    spec: TimeframeFeatureSpec,
) -> LineFit | None:
    recent = points[-spec.structural_point_count :]
    if len(recent) < spec.min_points:
        return None
    residual_threshold = spec.ransac_residual_atr * atr_value if np.isfinite(atr_value) else None
    return fit_boundary_line(recent, method=spec.estimator, residual_threshold=residual_threshold)


def _family_structure_features(prefix: str, snapshot, atr_value: float) -> dict[str, float]:
    upper = snapshot.upper
    lower = snapshot.lower
    return {
        f"upper_{prefix}_slope": float(upper.slope) if upper is not None else np.nan,
        f"lower_{prefix}_slope": float(lower.slope) if lower is not None else np.nan,
        f"{prefix}_slope_gap": (
            float(upper.slope - lower.slope) if upper is not None and lower is not None else np.nan
        ),
        f"{prefix}_width": snapshot.width if snapshot.valid else np.nan,
        f"{prefix}_width_over_atr": _safe_div(snapshot.width, atr_value) if snapshot.valid else np.nan,
        f"{prefix}_mid_slope": snapshot.mid_slope if snapshot.valid else np.nan,
    }


def _excursion_features(
    *,
    high_value: float,
    low_value: float,
    close_value: float,
    body_high_value: float,
    body_low_value: float,
    wick,
    body,
) -> dict[str, float]:
    upper_wick = wick.upper_value if wick.valid else np.nan
    lower_wick = wick.lower_value if wick.valid else np.nan
    upper_body = body.upper_value if body.valid else np.nan
    lower_body = body.lower_value if body.valid else np.nan
    return {
        "high_above_upper_wick": float(high_value > upper_wick) if np.isfinite(upper_wick) else np.nan,
        "close_above_upper_wick": float(close_value > upper_wick) if np.isfinite(upper_wick) else np.nan,
        "close_above_upper_body": float(close_value > upper_body) if np.isfinite(upper_body) else np.nan,
        "body_above_upper_body": float(body_low_value > upper_body) if np.isfinite(upper_body) else np.nan,
        "upper_wick_overshoot_size": max(0.0, high_value - upper_wick) if np.isfinite(upper_wick) else np.nan,
        "upper_body_break_size": max(0.0, body_low_value - upper_body) if np.isfinite(upper_body) else np.nan,
        "low_below_lower_wick": float(low_value < lower_wick) if np.isfinite(lower_wick) else np.nan,
        "close_below_lower_wick": float(close_value < lower_wick) if np.isfinite(lower_wick) else np.nan,
        "close_below_lower_body": float(close_value < lower_body) if np.isfinite(lower_body) else np.nan,
        "body_below_lower_body": float(body_high_value < lower_body) if np.isfinite(lower_body) else np.nan,
        "lower_wick_overshoot_size": max(0.0, lower_wick - low_value) if np.isfinite(lower_wick) else np.nan,
        "lower_body_break_size": max(0.0, lower_body - body_high_value) if np.isfinite(lower_body) else np.nan,
    }


def _touch_features(
    *,
    index: int,
    wick,
    body,
    epsilon_atr: float,
    lookback: int,
    atr_values: np.ndarray,
    bar_index_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    body_high_values: np.ndarray,
    body_low_values: np.ndarray,
) -> dict[str, float]:
    return {
        "count_touches_upper_wick": _touch_count(index, wick.upper, "upper", "wick", epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values),
        "count_touches_lower_wick": _touch_count(index, wick.lower, "lower", "wick", epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values),
        "count_touches_upper_body": _touch_count(index, body.upper, "upper", "body", epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values),
        "count_touches_lower_body": _touch_count(index, body.lower, "lower", "body", epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values),
        "time_since_last_upper_touch_wick": _time_since_last_touch(index, wick.upper, "upper", "wick", epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values),
        "time_since_last_lower_touch_wick": _time_since_last_touch(index, wick.lower, "lower", "wick", epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values),
        "time_since_last_upper_touch_body": _time_since_last_touch(index, body.upper, "upper", "body", epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values),
        "time_since_last_lower_touch_body": _time_since_last_touch(index, body.lower, "lower", "body", epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values),
    }


def _touch_count(
    index: int,
    line: LineFit | None,
    side: str,
    family: str,
    epsilon_atr: float,
    lookback: int,
    atr_values: np.ndarray,
    bar_index_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    body_high_values: np.ndarray,
    body_low_values: np.ndarray,
) -> float:
    if line is None:
        return np.nan
    matches = _touch_matches(index, line, side, family, epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values)
    return float(sum(matches))


def _time_since_last_touch(
    index: int,
    line: LineFit | None,
    side: str,
    family: str,
    epsilon_atr: float,
    lookback: int,
    atr_values: np.ndarray,
    bar_index_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    body_high_values: np.ndarray,
    body_low_values: np.ndarray,
) -> float:
    if line is None:
        return np.nan
    matches = _touch_matches(index, line, side, family, epsilon_atr, lookback, atr_values, bar_index_values, high_values, low_values, body_high_values, body_low_values)
    for offset, matched in enumerate(reversed(matches)):
        if matched:
            return float(offset)
    return np.nan


def _touch_matches(
    index: int,
    line: LineFit,
    side: str,
    family: str,
    epsilon_atr: float,
    lookback: int,
    atr_values: np.ndarray,
    bar_index_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    body_high_values: np.ndarray,
    body_low_values: np.ndarray,
) -> list[bool]:
    start = max(0, index - lookback + 1)
    matches: list[bool] = []
    for cursor in range(start, index + 1):
        atr_value = float(atr_values[cursor]) if np.isfinite(atr_values[cursor]) else np.nan
        if not np.isfinite(atr_value) or atr_value <= 0.0:
            matches.append(False)
            continue
        epsilon = float(epsilon_atr) * atr_value
        boundary = line.evaluate(float(bar_index_values[cursor]))
        if family == "wick" and side == "upper":
            anchor = float(high_values[cursor])
        elif family == "wick" and side == "lower":
            anchor = float(low_values[cursor])
        elif family == "body" and side == "upper":
            anchor = float(body_high_values[cursor])
        else:
            anchor = float(body_low_values[cursor])
        matches.append(abs(anchor - boundary) <= epsilon)
    return matches


def _persistence_features(
    *,
    index: int,
    wick,
    body,
    lookback: int,
    bar_index_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    close_values: np.ndarray,
    body_high_values: np.ndarray,
    body_low_values: np.ndarray,
) -> dict[str, float]:
    def consecutive(predicate: Callable[[int], bool]) -> float:
        count = 0
        for cursor in range(index, max(-1, index - lookback), -1):
            if predicate(cursor):
                count += 1
            else:
                break
        return float(count)

    def close_above_upper_body(cursor: int) -> bool:
        if body.upper is None:
            return False
        return float(close_values[cursor]) > body.upper.evaluate(float(bar_index_values[cursor]))

    def close_below_lower_body(cursor: int) -> bool:
        if body.lower is None:
            return False
        return float(close_values[cursor]) < body.lower.evaluate(float(bar_index_values[cursor]))

    def upper_rejection(cursor: int) -> bool:
        if wick.upper is None or body.upper is None:
            return False
        bar_index = float(bar_index_values[cursor])
        return float(high_values[cursor]) > wick.upper.evaluate(bar_index) and float(body_high_values[cursor]) <= body.upper.evaluate(bar_index)

    def lower_rejection(cursor: int) -> bool:
        if wick.lower is None or body.lower is None:
            return False
        bar_index = float(bar_index_values[cursor])
        return float(low_values[cursor]) < wick.lower.evaluate(bar_index) and float(body_low_values[cursor]) >= body.lower.evaluate(bar_index)

    return {
        "consecutive_closes_above_upper_body": consecutive(close_above_upper_body),
        "consecutive_closes_below_lower_body": consecutive(close_below_lower_body),
        "consecutive_rejections_upper": consecutive(upper_rejection),
        "consecutive_rejections_lower": consecutive(lower_rejection),
    }


def _swing_features(
    pivots: list[Pivot],
    index: int,
    close_value: float,
    atr_value: float,
    spec: TimeframeFeatureSpec,
) -> dict[str, float]:
    if not pivots:
        return {
            "last_pivot_type": 0.0,
            "bars_since_last_pivot": np.nan,
            "bars_since_last_pivot_confirm": np.nan,
            "price_distance_to_last_pivot": np.nan,
            "price_distance_to_last_pivot_over_atr": np.nan,
            "size_last_upswing": np.nan,
            "size_last_downswing": np.nan,
            "higher_high_flag": np.nan,
            "lower_low_flag": np.nan,
            "higher_low_flag": np.nan,
            "lower_high_flag": np.nan,
            "pivot_frequency": np.nan,
            "mean_recent_swing_size": np.nan,
            "swing_size_std": np.nan,
        }

    last_pivot = pivots[-1]
    highs = [pivot for pivot in pivots if pivot.pivot_type == "high"]
    lows = [pivot for pivot in pivots if pivot.pivot_type == "low"]
    recent_swings = [abs(pivots[idx].pivot_price - pivots[idx - 1].pivot_price) for idx in range(1, len(pivots))]
    recent_window = pivots[-spec.swing_lookback_pivots :]
    bars_window = max(1, index - recent_window[0].confirm_index + 1)

    return {
        "last_pivot_type": 1.0 if last_pivot.pivot_type == "high" else -1.0,
        "bars_since_last_pivot": float(index - last_pivot.pivot_index),
        "bars_since_last_pivot_confirm": float(index - last_pivot.confirm_index),
        "price_distance_to_last_pivot": close_value - float(last_pivot.pivot_price),
        "price_distance_to_last_pivot_over_atr": _safe_div(close_value - float(last_pivot.pivot_price), atr_value),
        "size_last_upswing": _last_swing_size(pivots, "low", "high"),
        "size_last_downswing": _last_swing_size(pivots, "high", "low"),
        "higher_high_flag": _comparison_flag(highs, lambda newer, older: newer > older),
        "lower_low_flag": _comparison_flag(lows, lambda newer, older: newer < older),
        "higher_low_flag": _comparison_flag(lows, lambda newer, older: newer > older),
        "lower_high_flag": _comparison_flag(highs, lambda newer, older: newer < older),
        "pivot_frequency": float(len(recent_window) / bars_window),
        "mean_recent_swing_size": float(np.mean(recent_swings[-spec.swing_lookback_pivots :])) if recent_swings else np.nan,
        "swing_size_std": float(np.std(recent_swings[-spec.swing_lookback_pivots :])) if recent_swings else np.nan,
    }


def _last_swing_size(pivots: list[Pivot], from_type: str, to_type: str) -> float:
    for cursor in range(len(pivots) - 1, 0, -1):
        current = pivots[cursor]
        previous = pivots[cursor - 1]
        if previous.pivot_type == from_type and current.pivot_type == to_type:
            return abs(float(current.pivot_price) - float(previous.pivot_price))
    return np.nan


def _comparison_flag(points: list[Pivot], comparator: Callable[[float, float], bool]) -> float:
    if len(points) < 2:
        return np.nan
    newer = float(points[-1].pivot_price)
    older = float(points[-2].pivot_price)
    return float(comparator(newer, older))


def _evolution_features(previous_valid_state: dict[str, float] | None, wick, body) -> dict[str, float]:
    body_to_wick_ratio = _safe_div(body.width, wick.width) if body.valid and wick.valid else np.nan
    if previous_valid_state is None:
        return {
            "change_in_wick_width": np.nan,
            "change_in_body_width": np.nan,
            "slope_acceleration_upper": np.nan,
            "slope_acceleration_lower": np.nan,
            "body_wick_ratio_change": np.nan,
        }
    return {
        "change_in_wick_width": wick.width - previous_valid_state.get("wick_width", np.nan) if wick.valid else np.nan,
        "change_in_body_width": body.width - previous_valid_state.get("body_width", np.nan) if body.valid else np.nan,
        "slope_acceleration_upper": (
            float(body.upper.slope if body.upper is not None else np.nan) - previous_valid_state.get("upper_body_slope", np.nan)
        ),
        "slope_acceleration_lower": (
            float(body.lower.slope if body.lower is not None else np.nan) - previous_valid_state.get("lower_body_slope", np.nan)
        ),
        "body_wick_ratio_change": body_to_wick_ratio - previous_valid_state.get("body_to_wick_width_ratio", np.nan),
    }


def _add_position_features(frame: pd.DataFrame, timeframe: str) -> list[str]:
    close_value = frame["decision_close"].astype(float)
    upper_wick = frame.get(f"upper_wick_boundary_{timeframe}")
    lower_wick = frame.get(f"lower_wick_boundary_{timeframe}")
    upper_body = frame.get(f"upper_body_boundary_{timeframe}")
    lower_body = frame.get(f"lower_body_boundary_{timeframe}")
    wick_width = frame.get(f"wick_width_{timeframe}")
    body_width = frame.get(f"body_width_{timeframe}")
    created: list[str] = []

    def add(name: str, series: pd.Series) -> None:
        frame[name] = series
        created.append(name)

    if upper_wick is not None and lower_wick is not None and wick_width is not None:
        add(f"dist_close_to_upper_wick_{timeframe}", upper_wick - close_value)
        add(f"dist_close_to_lower_wick_{timeframe}", close_value - lower_wick)
        add(f"dist_close_to_upper_wick_over_wickwidth_{timeframe}", _safe_series_div(upper_wick - close_value, wick_width))
        add(f"dist_close_to_lower_wick_over_wickwidth_{timeframe}", _safe_series_div(close_value - lower_wick, wick_width))
        add(f"pos_in_wick_{timeframe}", _safe_series_div(close_value - lower_wick, wick_width))
    if upper_body is not None and lower_body is not None and body_width is not None:
        add(f"dist_close_to_upper_body_{timeframe}", upper_body - close_value)
        add(f"dist_close_to_lower_body_{timeframe}", close_value - lower_body)
        add(f"dist_close_to_upper_body_over_bodywidth_{timeframe}", _safe_series_div(upper_body - close_value, body_width))
        add(f"dist_close_to_lower_body_over_bodywidth_{timeframe}", _safe_series_div(close_value - lower_body, body_width))
        add(f"pos_in_body_{timeframe}", _safe_series_div(close_value - lower_body, body_width))
    return created


def _add_regime_features(frame: pd.DataFrame, decision_tf: str) -> list[str]:
    returns = frame[f"return_1_tf_{decision_tf}"].astype(float)
    log_returns = frame[f"log_return_1_tf_{decision_tf}"].astype(float)
    volume = frame[f"volume_tf_{decision_tf}"].astype(float)
    close_value = frame[f"close_tf_{decision_tf}"].astype(float)
    created = []

    frame["ATR_1h"] = frame[f"atr_tf_{decision_tf}"].astype(float)
    created.append("ATR_1h")

    frame["realized_vol_1h"] = log_returns.rolling(24).std()
    frame["realized_vol_24h"] = log_returns.rolling(24).std() * np.sqrt(24.0)
    frame["return_std_7d"] = returns.rolling(24 * 7).std()
    volume_mean = volume.rolling(24 * 7).mean()
    volume_std = volume.rolling(24 * 7).std()
    frame["volume_zscore_1h"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)
    momentum_24h = close_value.pct_change(24)
    frame["rolling_trend_strength_1h"] = momentum_24h.abs() / frame["realized_vol_24h"].replace(0.0, np.nan)
    frame["rolling_autocorrelation_1h"] = returns.rolling(48).corr(returns.shift(1))
    frame["recent_gapless_momentum_1h"] = returns.rolling(24).sum()
    created.extend(
        [
            "realized_vol_1h",
            "realized_vol_24h",
            "return_std_7d",
            "volume_zscore_1h",
            "rolling_trend_strength_1h",
            "rolling_autocorrelation_1h",
            "recent_gapless_momentum_1h",
        ]
    )
    return created


def _add_confluence_features(frame: pd.DataFrame, timeframes: list[str]) -> list[str]:
    created: list[str] = []
    wick_mid_slopes = [f"wick_mid_slope_{timeframe}" for timeframe in timeframes if f"wick_mid_slope_{timeframe}" in frame.columns]
    body_mid_slopes = [f"body_mid_slope_{timeframe}" for timeframe in timeframes if f"body_mid_slope_{timeframe}" in frame.columns]

    def sign_agreement(columns: list[str]) -> pd.Series:
        signs = frame[columns].apply(np.sign)
        count = signs.notna().sum(axis=1).replace(0, np.nan)
        return signs.sum(axis=1).abs() / count

    if wick_mid_slopes:
        frame["sign_agreement_wick_mid_slope"] = sign_agreement(wick_mid_slopes)
        created.append("sign_agreement_wick_mid_slope")
    if body_mid_slopes:
        frame["sign_agreement_body_mid_slope"] = sign_agreement(body_mid_slopes)
        created.append("sign_agreement_body_mid_slope")

    if body_mid_slopes:
        body_slope_values = frame[body_mid_slopes]
        frame["num_bullish_timeframes"] = (body_slope_values > 0.0).sum(axis=1).astype(float)
        frame["num_bearish_timeframes"] = (body_slope_values < 0.0).sum(axis=1).astype(float)
        created.extend(["num_bullish_timeframes", "num_bearish_timeframes"])

    if {"pos_in_body_1h", "body_mid_slope_4h"}.issubset(frame.columns):
        frame["1h_near_lower_body_and_4h_uptrend"] = (
            (frame["pos_in_body_1h"] <= 0.15) & (frame["body_mid_slope_4h"] > 0.0)
        ).astype(float)
        created.append("1h_near_lower_body_and_4h_uptrend")
    if {"close_above_upper_body_1h", "body_mid_slope_1d"}.issubset(frame.columns):
        frame["1h_breakout_above_body_and_1d_uptrend"] = (
            (frame["close_above_upper_body_1h"] > 0.0) & (frame["body_mid_slope_1d"] > 0.0)
        ).astype(float)
        created.append("1h_breakout_above_body_and_1d_uptrend")
    if {"pos_in_body_1h", "body_mid_slope_4h", "body_mid_slope_1d", "body_mid_slope_1w"}.issubset(frame.columns):
        frame["1h_below_body_mid_while_4h_1d_1w_all_positive"] = (
            (frame["pos_in_body_1h"] < 0.5)
            & (frame["body_mid_slope_4h"] > 0.0)
            & (frame["body_mid_slope_1d"] > 0.0)
            & (frame["body_mid_slope_1w"] > 0.0)
        ).astype(float)
        created.append("1h_below_body_mid_while_4h_1d_1w_all_positive")
    if {"consecutive_rejections_upper_1h", "body_mid_slope_4h", "body_mid_slope_1d"}.issubset(frame.columns):
        frame["1h_rejection_at_upper_wick_while_4h_1d_down"] = (
            (frame["consecutive_rejections_upper_1h"] > 0.0)
            & (frame["body_mid_slope_4h"] < 0.0)
            & (frame["body_mid_slope_1d"] < 0.0)
        ).astype(float)
        created.append("1h_rejection_at_upper_wick_while_4h_1d_down")

    return created


def _safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return np.nan
    return float(numerator / denominator)


def _safe_series_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)
