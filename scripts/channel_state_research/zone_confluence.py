from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scripts.backtest_turtle_soup import build_htf_zone_events, normalize_binance_spot_symbol
from scripts.channel_state_research.labels import high_before_low


@dataclass(frozen=True)
class ZoneChannelEventSpec:
    zone_timeframes: tuple[str, ...] = ("4h", "1d")
    zone_left: int = 5
    zone_right: int = 5
    zone_ob_search_bars: int = 50
    zone_use_body: bool = False
    zone_penetration_frac: float = 0.50
    min_reclaim_pos: float = 0.60
    max_zone_scan: int = 50
    confluence_epsilon_atr: float = 0.50
    entry_mode: str = "market_reclaim"
    passive_entry_window_bars: int = 6
    passive_entry_buffer_atr: float = 0.0
    stop_mode: str = "channel_anchor"
    stop_buffer_atr: float = 0.20
    target_buffer_atr: float = 0.20
    label_horizon_bars: int = 24
    fee_bps_side: float = 5.0
    slippage_bps_side: float = 2.0
    channel_timeframes: tuple[str, ...] = ("1h", "4h", "1d")
    execution_timeframe: str = "1h"


def build_zone_channel_event_dataset(
    *,
    symbol: str,
    exec_frame: pd.DataFrame,
    decision_frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    spec: ZoneChannelEventSpec,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    return _build_zone_channel_dataset(
        symbol=symbol,
        exec_frame=exec_frame,
        decision_frame=decision_frame,
        feature_groups=feature_groups,
        spec=spec,
        include_labels=True,
    )


def build_zone_channel_signal_dataset(
    *,
    symbol: str,
    exec_frame: pd.DataFrame,
    decision_frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    spec: ZoneChannelEventSpec,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    return _build_zone_channel_dataset(
        symbol=symbol,
        exec_frame=exec_frame,
        decision_frame=decision_frame,
        feature_groups=feature_groups,
        spec=spec,
        include_labels=False,
    )


def _build_zone_channel_dataset(
    *,
    symbol: str,
    exec_frame: pd.DataFrame,
    decision_frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    spec: ZoneChannelEventSpec,
    include_labels: bool,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    normalized_symbol = normalize_binance_spot_symbol(symbol)
    exec_bars = exec_frame.sort_values("close_time").reset_index(drop=True).copy()
    decisions = decision_frame.sort_values("decision_time").reset_index(drop=True).copy()
    if len(exec_bars) != len(decisions):
        raise ValueError("Execution frame and decision frame must align one-to-one on the execution timeframe.")
    exec_close_time = pd.to_datetime(exec_bars["close_time"], utc=True, errors="coerce")
    decision_time = pd.to_datetime(decisions["decision_time"], utc=True, errors="coerce")
    if not exec_close_time.equals(decision_time):
        raise ValueError("Execution frame close times and decision frame decision_time must match.")

    zone_streams = _zone_streams(exec_bars, spec)
    active_long_zones: list[dict[str, Any]] = []
    active_short_zones: list[dict[str, Any]] = []
    zone_ptr: dict[tuple[str, str], int] = {(timeframe, side): 0 for timeframe in spec.zone_timeframes for side in ("long", "short")}

    opens = exec_bars["open"].astype(float).to_list()
    highs = exec_bars["high"].astype(float).to_list()
    lows = exec_bars["low"].astype(float).to_list()
    closes = exec_bars["close"].astype(float).to_list()
    atrs = exec_bars["atr"].astype(float).to_list()
    close_times = pd.to_datetime(exec_bars["close_time"], utc=True, errors="coerce").to_list()

    base_feature_columns = _model_feature_columns(feature_groups)
    event_groups = _event_feature_groups(feature_groups)
    rows: list[dict[str, Any]] = []

    for index, (_, decision_row) in enumerate(decisions.iterrows()):
        visible_time = pd.Timestamp(decision_row["decision_time"]).tz_convert("UTC")
        _update_visible_zones(
            zone_streams=zone_streams,
            zone_ptr=zone_ptr,
            visible_time=visible_time,
            active_long_zones=active_long_zones,
            active_short_zones=active_short_zones,
        )

        active_long_zones = [
            zone
            for zone in active_long_zones
            if _zone_is_still_active(zone, "long", highs[index], lows[index], spec.zone_penetration_frac)
        ]
        active_short_zones = [
            zone
            for zone in active_short_zones
            if _zone_is_still_active(zone, "short", highs[index], lows[index], spec.zone_penetration_frac)
        ]

        long_candidates = [zone for zone in reversed(active_long_zones) if not zone["used"]]
        short_candidates = [zone for zone in reversed(active_short_zones) if not zone["used"]]
        if spec.max_zone_scan > 0:
            long_candidates = long_candidates[: spec.max_zone_scan]
            short_candidates = short_candidates[: spec.max_zone_scan]

        active_long_count = len(long_candidates)
        active_short_count = len(short_candidates)

        for rank, zone in enumerate(long_candidates):
            touched, reclaim_pos = _long_zone_touch_state(zone, lows[index], highs[index], closes[index], spec.zone_penetration_frac)
            if touched:
                zone["touch_count"] += 1
            if not touched or closes[index] <= float(zone["top"]) or reclaim_pos < spec.min_reclaim_pos:
                continue
            confluence = _zone_channel_confluence(decision_row, zone, "long", spec)
            if not confluence["is_confluent"]:
                continue
            row_builder = _build_event_row if include_labels else _build_signal_row
            row = row_builder(
                symbol=normalized_symbol,
                decision_row=decision_row,
                base_feature_columns=base_feature_columns,
                direction="long",
                zone=zone,
                event_index=index,
                event_rank=rank,
                active_same=active_long_count,
                active_opp=active_short_count,
                confluence=confluence,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                close_times=close_times,
                spec=spec,
            )
            if row is not None:
                rows.append(row)
                zone["used"] = True
                break

        for rank, zone in enumerate(short_candidates):
            touched, reclaim_pos = _short_zone_touch_state(zone, highs[index], lows[index], closes[index], spec.zone_penetration_frac)
            if touched:
                zone["touch_count"] += 1
            if not touched or closes[index] >= float(zone["bottom"]) or reclaim_pos < spec.min_reclaim_pos:
                continue
            confluence = _zone_channel_confluence(decision_row, zone, "short", spec)
            if not confluence["is_confluent"]:
                continue
            row_builder = _build_event_row if include_labels else _build_signal_row
            row = row_builder(
                symbol=normalized_symbol,
                decision_row=decision_row,
                base_feature_columns=base_feature_columns,
                direction="short",
                zone=zone,
                event_index=index,
                event_rank=rank,
                active_same=active_short_count,
                active_opp=active_long_count,
                confluence=confluence,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                close_times=close_times,
                spec=spec,
            )
            if row is not None:
                rows.append(row)
                zone["used"] = True
                break

    events = pd.DataFrame(rows)
    if events.empty:
        empty_columns = _event_columns(base_feature_columns) if include_labels else _signal_columns(base_feature_columns)
        return pd.DataFrame(columns=empty_columns), event_groups

    for column in ["event_time", "zone_time", "entry_time", "exit_time"]:
        if column in events.columns:
            events[column] = pd.to_datetime(events[column], utc=True, errors="coerce")
    return events, event_groups


def label_channel_trade_outcome(
    *,
    direction: str,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    close_times: list[pd.Timestamp],
    start_index: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    horizon_bars: int,
) -> dict[str, Any] | None:
    risk = abs(entry_price - stop_price)
    if not math.isfinite(risk) or risk <= 0.0:
        return None
    target_r = abs(target_price - entry_price) / risk
    if not math.isfinite(target_r) or target_r <= 0.0:
        return None

    final_index = min(len(closes) - 1, start_index + int(horizon_bars))
    if start_index > final_index:
        return None
    entry_time = pd.Timestamp(close_times[start_index]).tz_convert("UTC")

    mfe_r = 0.0
    mae_r = 0.0
    last_close_r = 0.0

    for cursor in range(start_index, final_index + 1):
        if direction == "long":
            mfe_r = max(mfe_r, (highs[cursor] - entry_price) / risk)
            mae_r = max(mae_r, (entry_price - lows[cursor]) / risk)
            target_hit = highs[cursor] >= target_price
            stop_hit = lows[cursor] <= stop_price
            if target_hit and stop_hit:
                target_first = high_before_low(opens[cursor], highs[cursor], lows[cursor])
                realized_r = target_r if target_first else -1.0
                return {
                    "hold_label": 1.0 if target_first else 0.0,
                    "future_r": realized_r,
                    "outcome": "target_same_bar" if target_first else "stop_same_bar",
                    "entry_time": entry_time,
                    "bars_to_outcome": float(cursor - start_index + 1),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            if target_hit:
                return {
                    "hold_label": 1.0,
                    "future_r": float(target_r),
                    "outcome": "target",
                    "entry_time": entry_time,
                    "bars_to_outcome": float(cursor - start_index + 1),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            if stop_hit:
                return {
                    "hold_label": 0.0,
                    "future_r": -1.0,
                    "outcome": "stop",
                    "entry_time": entry_time,
                    "bars_to_outcome": float(cursor - start_index + 1),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            last_close_r = (closes[cursor] - entry_price) / risk
        else:
            mfe_r = max(mfe_r, (entry_price - lows[cursor]) / risk)
            mae_r = max(mae_r, (highs[cursor] - entry_price) / risk)
            target_hit = lows[cursor] <= target_price
            stop_hit = highs[cursor] >= stop_price
            if target_hit and stop_hit:
                target_first = not high_before_low(opens[cursor], highs[cursor], lows[cursor])
                realized_r = target_r if target_first else -1.0
                return {
                    "hold_label": 1.0 if target_first else 0.0,
                    "future_r": realized_r,
                    "outcome": "target_same_bar" if target_first else "stop_same_bar",
                    "entry_time": entry_time,
                    "bars_to_outcome": float(cursor - start_index + 1),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            if target_hit:
                return {
                    "hold_label": 1.0,
                    "future_r": float(target_r),
                    "outcome": "target",
                    "entry_time": entry_time,
                    "bars_to_outcome": float(cursor - start_index + 1),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            if stop_hit:
                return {
                    "hold_label": 0.0,
                    "future_r": -1.0,
                    "outcome": "stop",
                    "entry_time": entry_time,
                    "bars_to_outcome": float(cursor - start_index + 1),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            last_close_r = (entry_price - closes[cursor]) / risk

    clipped_r = max(-1.0, min(float(target_r), float(last_close_r)))
    return {
        "hold_label": 1.0 if clipped_r > 0.0 else 0.0,
        "future_r": float(clipped_r),
        "outcome": "timeout",
        "entry_time": entry_time,
        "bars_to_outcome": float(final_index - start_index + 1),
        "exit_time": pd.Timestamp(close_times[final_index]).tz_convert("UTC"),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
    }


def label_passive_retest_trade_outcome(
    *,
    direction: str,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    close_times: list[pd.Timestamp],
    signal_index: int,
    limit_entry_price: float,
    stop_price: float,
    target_price: float,
    entry_window_bars: int,
    horizon_bars: int,
) -> dict[str, Any] | None:
    risk = abs(limit_entry_price - stop_price)
    if not math.isfinite(risk) or risk <= 0.0:
        return None
    target_r = abs(target_price - limit_entry_price) / risk
    if not math.isfinite(target_r) or target_r <= 0.0:
        return None

    fill_deadline = min(len(closes) - 1, signal_index + max(int(entry_window_bars), 1))
    if signal_index + 1 > fill_deadline:
        return None

    fill_index: int | None = None
    for cursor in range(signal_index + 1, fill_deadline + 1):
        if direction == "long" and lows[cursor] <= limit_entry_price:
            fill_index = cursor
            break
        if direction == "short" and highs[cursor] >= limit_entry_price:
            fill_index = cursor
            break
    if fill_index is None:
        return None

    final_index = min(len(closes) - 1, fill_index + int(horizon_bars))
    mfe_r = 0.0
    mae_r = 0.0
    last_close_r = 0.0

    for cursor in range(fill_index, final_index + 1):
        if direction == "long":
            stop_hit = lows[cursor] <= stop_price
            target_hit = highs[cursor] >= target_price
            low_before_high = not high_before_low(opens[cursor], highs[cursor], lows[cursor])
            if cursor == fill_index:
                mae_r = max(mae_r, (limit_entry_price - lows[cursor]) / risk)
                if low_before_high:
                    mfe_r = max(mfe_r, (highs[cursor] - limit_entry_price) / risk)
            else:
                mfe_r = max(mfe_r, (highs[cursor] - limit_entry_price) / risk)
                mae_r = max(mae_r, (limit_entry_price - lows[cursor]) / risk)

            if stop_hit:
                return {
                    "hold_label": 0.0,
                    "future_r": -1.0,
                    "outcome": "stop_on_fill_bar" if cursor == fill_index else "stop",
                    "bars_to_outcome": float(cursor - fill_index + 1),
                    "entry_delay_bars": float(fill_index - signal_index),
                    "entry_time": pd.Timestamp(close_times[fill_index]).tz_convert("UTC"),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            if target_hit and (cursor > fill_index or low_before_high):
                return {
                    "hold_label": 1.0,
                    "future_r": float(target_r),
                    "outcome": "target_on_fill_bar" if cursor == fill_index else "target",
                    "bars_to_outcome": float(cursor - fill_index + 1),
                    "entry_delay_bars": float(fill_index - signal_index),
                    "entry_time": pd.Timestamp(close_times[fill_index]).tz_convert("UTC"),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            last_close_r = (closes[cursor] - limit_entry_price) / risk
        else:
            stop_hit = highs[cursor] >= stop_price
            target_hit = lows[cursor] <= target_price
            high_before_low_bar = high_before_low(opens[cursor], highs[cursor], lows[cursor])
            if cursor == fill_index:
                mae_r = max(mae_r, (highs[cursor] - limit_entry_price) / risk)
                if high_before_low_bar:
                    mfe_r = max(mfe_r, (limit_entry_price - lows[cursor]) / risk)
            else:
                mfe_r = max(mfe_r, (limit_entry_price - lows[cursor]) / risk)
                mae_r = max(mae_r, (highs[cursor] - limit_entry_price) / risk)

            if stop_hit:
                return {
                    "hold_label": 0.0,
                    "future_r": -1.0,
                    "outcome": "stop_on_fill_bar" if cursor == fill_index else "stop",
                    "bars_to_outcome": float(cursor - fill_index + 1),
                    "entry_delay_bars": float(fill_index - signal_index),
                    "entry_time": pd.Timestamp(close_times[fill_index]).tz_convert("UTC"),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            if target_hit and (cursor > fill_index or high_before_low_bar):
                return {
                    "hold_label": 1.0,
                    "future_r": float(target_r),
                    "outcome": "target_on_fill_bar" if cursor == fill_index else "target",
                    "bars_to_outcome": float(cursor - fill_index + 1),
                    "entry_delay_bars": float(fill_index - signal_index),
                    "entry_time": pd.Timestamp(close_times[fill_index]).tz_convert("UTC"),
                    "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
                    "mfe_r": float(mfe_r),
                    "mae_r": float(mae_r),
                }
            last_close_r = (limit_entry_price - closes[cursor]) / risk

    clipped_r = max(-1.0, min(float(target_r), float(last_close_r)))
    return {
        "hold_label": 1.0 if clipped_r > 0.0 else 0.0,
        "future_r": float(clipped_r),
        "outcome": "timeout",
        "bars_to_outcome": float(final_index - fill_index + 1),
        "entry_delay_bars": float(fill_index - signal_index),
        "entry_time": pd.Timestamp(close_times[fill_index]).tz_convert("UTC"),
        "exit_time": pd.Timestamp(close_times[final_index]).tz_convert("UTC"),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
    }


def _zone_streams(exec_bars: pd.DataFrame, spec: ZoneChannelEventSpec) -> dict[tuple[str, str], list[dict[str, Any]]]:
    streams: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for timeframe in spec.zone_timeframes:
        supply_events, demand_events = build_htf_zone_events(
            exec_bars,
            timeframe,
            spec.zone_left,
            spec.zone_right,
            0.25,
            spec.zone_ob_search_bars,
            spec.zone_use_body,
        )
        streams[(timeframe, "long")] = [_add_zone_metadata(event, timeframe, "long", index) for index, event in enumerate(demand_events)]
        streams[(timeframe, "short")] = [_add_zone_metadata(event, timeframe, "short", index) for index, event in enumerate(supply_events)]
    return streams


def _add_zone_metadata(event: dict[str, Any], timeframe: str, direction: str, index: int) -> dict[str, Any]:
    return {
        **event,
        "zone_tf": timeframe,
        "direction": direction,
        "touch_count": 0,
        "used": False,
        "id": f"{timeframe}-{direction}-{index}-{pd.Timestamp(event['time']).isoformat()}-{float(event['top']):.8f}-{float(event['bottom']):.8f}",
    }


def _update_visible_zones(
    *,
    zone_streams: dict[tuple[str, str], list[dict[str, Any]]],
    zone_ptr: dict[tuple[str, str], int],
    visible_time: pd.Timestamp,
    active_long_zones: list[dict[str, Any]],
    active_short_zones: list[dict[str, Any]],
) -> None:
    for timeframe in {key[0] for key in zone_streams}:
        long_key = (timeframe, "long")
        short_key = (timeframe, "short")
        while zone_ptr[long_key] < len(zone_streams[long_key]) and pd.Timestamp(zone_streams[long_key][zone_ptr[long_key]]["time"]).tz_convert("UTC") < visible_time:
            active_long_zones.append(zone_streams[long_key][zone_ptr[long_key]])
            zone_ptr[long_key] += 1
        while zone_ptr[short_key] < len(zone_streams[short_key]) and pd.Timestamp(zone_streams[short_key][zone_ptr[short_key]]["time"]).tz_convert("UTC") < visible_time:
            active_short_zones.append(zone_streams[short_key][zone_ptr[short_key]])
            zone_ptr[short_key] += 1


def _zone_is_still_active(
    zone: dict[str, Any],
    direction: str,
    high_value: float,
    low_value: float,
    penetration_frac: float,
) -> bool:
    if zone["used"]:
        return False
    width = max(0.0, float(zone["width"]))
    if direction == "long":
        invalidation_limit = float(zone["bottom"]) - width * penetration_frac
        return low_value >= invalidation_limit
    invalidation_limit = float(zone["top"]) + width * penetration_frac
    return high_value <= invalidation_limit


def _long_zone_touch_state(zone: dict[str, Any], low_value: float, high_value: float, close_value: float, penetration_frac: float) -> tuple[bool, float]:
    width = float(zone["width"])
    sweep_range = high_value - low_value
    reclaim_pos = (close_value - low_value) / sweep_range if sweep_range > 0 else 0.0
    penetration_limit = float(zone["bottom"]) - width * penetration_frac
    touched = low_value <= float(zone["top"]) and low_value >= penetration_limit
    return touched, reclaim_pos


def _short_zone_touch_state(zone: dict[str, Any], high_value: float, low_value: float, close_value: float, penetration_frac: float) -> tuple[bool, float]:
    width = float(zone["width"])
    sweep_range = high_value - low_value
    reclaim_pos = (high_value - close_value) / sweep_range if sweep_range > 0 else 0.0
    penetration_limit = float(zone["top"]) + width * penetration_frac
    touched = high_value >= float(zone["bottom"]) and high_value <= penetration_limit
    return touched, reclaim_pos


def _zone_channel_confluence(
    decision_row: pd.Series,
    zone: dict[str, Any],
    direction: str,
    spec: ZoneChannelEventSpec,
) -> dict[str, Any]:
    atr_value = _finite_float(decision_row.get(f"atr_tf_{spec.execution_timeframe}", np.nan))
    if not math.isfinite(atr_value) or atr_value <= 0.0:
        return {"is_confluent": False}

    prefix = "lower" if direction == "long" else "upper"
    matches: list[dict[str, Any]] = []
    for timeframe in spec.channel_timeframes:
        for family in ("wick", "body"):
            boundary_value = _finite_float(decision_row.get(f"{prefix}_{family}_boundary_{timeframe}", np.nan))
            if not math.isfinite(boundary_value):
                continue
            gap_abs = _zone_interval_gap(float(zone["top"]), float(zone["bottom"]), boundary_value)
            gap_atr = gap_abs / atr_value
            matches.append(
                {
                    "timeframe": timeframe,
                    "family": family,
                    "boundary_value": boundary_value,
                    "gap_abs": gap_abs,
                    "gap_atr": gap_atr,
                }
            )

    if not matches:
        return {"is_confluent": False}

    matches.sort(key=lambda item: (item["gap_atr"], item["timeframe"], item["family"]))
    best = matches[0]
    epsilon = float(spec.confluence_epsilon_atr)
    zone_mid = (float(zone["top"]) + float(zone["bottom"])) / 2.0
    out = {
        "is_confluent": bool(best["gap_atr"] <= epsilon),
        "matched_boundary_gap_abs": float(best["gap_abs"]),
        "matched_boundary_gap_atr": float(best["gap_atr"]),
        "matched_boundary_value": float(best["boundary_value"]),
        "matched_boundary_is_wick": 1.0 if best["family"] == "wick" else 0.0,
        "matched_boundary_is_body": 1.0 if best["family"] == "body" else 0.0,
        "confluence_boundary_count": float(sum(1 for item in matches if item["gap_atr"] <= epsilon)),
        "confluence_boundary_count_wide": float(sum(1 for item in matches if item["gap_atr"] <= 2.0 * epsilon)),
        "zone_mid_to_decision_close_atr": (
            (float(decision_row["decision_close"]) - zone_mid) / atr_value if direction == "long" else (zone_mid - float(decision_row["decision_close"])) / atr_value
        ),
    }
    for timeframe in spec.channel_timeframes:
        out[f"matched_boundary_tf_{timeframe}"] = 1.0 if best["timeframe"] == timeframe else 0.0
    return out


def _build_signal_row(
    *,
    symbol: str,
    decision_row: pd.Series,
    base_feature_columns: list[str],
    direction: str,
    zone: dict[str, Any],
    event_index: int,
    event_rank: int,
    active_same: int,
    active_opp: int,
    confluence: dict[str, Any],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    atrs: list[float],
    close_times: list[pd.Timestamp] | None = None,
    spec: ZoneChannelEventSpec,
) -> dict[str, Any] | None:
    atr_value = _finite_float(atrs[event_index])
    signal_price = _finite_float(decision_row["decision_close"])
    if not math.isfinite(signal_price) or signal_price <= 0.0 or not math.isfinite(atr_value) or atr_value <= 0.0:
        return None

    support_anchor = np.nanmin(
        np.array(
            [
                float(zone["bottom"]),
                _finite_float(decision_row.get(f"lower_wick_boundary_{spec.execution_timeframe}", np.nan)),
                _finite_float(decision_row.get(f"lower_body_boundary_{spec.execution_timeframe}", np.nan)),
            ],
            dtype=float,
        )
    )
    resistance_anchor = np.nanmax(
        np.array(
            [
                float(zone["top"]),
                _finite_float(decision_row.get(f"upper_wick_boundary_{spec.execution_timeframe}", np.nan)),
                _finite_float(decision_row.get(f"upper_body_boundary_{spec.execution_timeframe}", np.nan)),
            ],
            dtype=float,
        )
    )
    event_low = float(lows[event_index])
    event_high = float(highs[event_index])
    matched_boundary_value = _finite_float(confluence.get("matched_boundary_value", np.nan))
    entry_price = _resolve_entry_price(
        direction=direction,
        entry_mode=spec.entry_mode,
        signal_price=float(signal_price),
        zone_top=float(zone["top"]),
        zone_bottom=float(zone["bottom"]),
        matched_boundary_value=matched_boundary_value,
        atr_value=atr_value,
        passive_entry_buffer_atr=spec.passive_entry_buffer_atr,
    )
    if not math.isfinite(entry_price) or entry_price <= 0.0:
        return None
    if direction == "long":
        target_anchor = _first_finite(
            [
                decision_row.get(f"upper_body_boundary_{spec.execution_timeframe}", np.nan),
                decision_row.get(f"upper_wick_boundary_{spec.execution_timeframe}", np.nan),
            ]
        )
        stop_price = _resolve_stop_price(
            direction=direction,
            stop_mode=spec.stop_mode,
            channel_anchor=float(support_anchor),
            zone_top=float(zone["top"]),
            zone_bottom=float(zone["bottom"]),
            event_low=event_low,
            event_high=event_high,
            atr_value=atr_value,
            stop_buffer_atr=spec.stop_buffer_atr,
        )
        target_price = float(target_anchor) - spec.target_buffer_atr * atr_value if math.isfinite(target_anchor) else np.nan
    else:
        target_anchor = _first_finite(
            [
                decision_row.get(f"lower_body_boundary_{spec.execution_timeframe}", np.nan),
                decision_row.get(f"lower_wick_boundary_{spec.execution_timeframe}", np.nan),
            ]
        )
        stop_price = _resolve_stop_price(
            direction=direction,
            stop_mode=spec.stop_mode,
            channel_anchor=float(resistance_anchor),
            zone_top=float(zone["top"]),
            zone_bottom=float(zone["bottom"]),
            event_low=event_low,
            event_high=event_high,
            atr_value=atr_value,
            stop_buffer_atr=spec.stop_buffer_atr,
        )
        target_price = float(target_anchor) + spec.target_buffer_atr * atr_value if math.isfinite(target_anchor) else np.nan

    if not math.isfinite(stop_price) or not math.isfinite(target_price):
        return None
    risk_abs = abs(entry_price - stop_price)
    if not math.isfinite(risk_abs) or risk_abs <= 0.0:
        return None
    if direction == "long" and target_price <= entry_price:
        return None
    if direction == "short" and target_price >= entry_price:
        return None
    target_distance_atr = abs(target_price - entry_price) / atr_value
    stop_distance_atr = abs(entry_price - stop_price) / atr_value
    target_rr_planned = abs(target_price - entry_price) / risk_abs
    if not math.isfinite(target_rr_planned) or target_rr_planned <= 0.0:
        return None

    cost_r = ((2.0 * spec.fee_bps_side) + (2.0 * spec.slippage_bps_side)) / 10_000.0 * entry_price / risk_abs
    row: dict[str, Any] = {
        "symbol": symbol,
        "event_time": pd.Timestamp(decision_row["decision_time"]).tz_convert("UTC"),
        "direction": direction,
        "zone_time": pd.Timestamp(zone["time"]).tz_convert("UTC"),
        "zone_tf_5m": 1.0 if str(zone["zone_tf"]) == "5m" else 0.0,
        "zone_tf_15m": 1.0 if str(zone["zone_tf"]) == "15m" else 0.0,
        "zone_tf_1h": 1.0 if str(zone["zone_tf"]) == "1h" else 0.0,
        "zone_tf_4h": 1.0 if str(zone["zone_tf"]) == "4h" else 0.0,
        "zone_tf_1d": 1.0 if str(zone["zone_tf"]) == "1d" else 0.0,
        "zone_tf_1w": 1.0 if str(zone["zone_tf"]) == "1w" else 0.0,
        "zone_tf": str(zone["zone_tf"]),
        "zone_top": float(zone["top"]),
        "zone_bottom": float(zone["bottom"]),
        "signal_price": float(signal_price),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "entry_mode_market_reclaim": 1.0 if spec.entry_mode == "market_reclaim" else 0.0,
        "entry_mode_passive_retest": 1.0 if spec.entry_mode == "passive_retest" else 0.0,
        "stop_mode_channel_anchor": 1.0 if spec.stop_mode == "channel_anchor" else 0.0,
        "stop_mode_zone": 1.0 if spec.stop_mode == "zone" else 0.0,
        "stop_mode_reaction_extreme": 1.0 if spec.stop_mode == "reaction_extreme" else 0.0,
        "entry_distance_from_signal_atr": abs(float(signal_price) - float(entry_price)) / atr_value,
        "risk_abs": float(risk_abs),
        "cost_r": float(cost_r),
        "target_rr_planned": float(target_rr_planned),
        "target_distance_atr": float(target_distance_atr),
        "stop_distance_atr": float(stop_distance_atr),
        "zone_width_pct": abs(float(zone["top"]) - float(zone["bottom"])) / entry_price * 100.0,
        "zone_width_atr": abs(float(zone["top"]) - float(zone["bottom"])) / atr_value,
        "zone_age_hours": (pd.Timestamp(decision_row["decision_time"]).tz_convert("UTC") - pd.Timestamp(zone["time"]).tz_convert("UTC")).total_seconds() / 3600.0,
        "zone_rank": float(event_rank),
        "prior_zone_touches": float(zone["touch_count"]),
        "active_same_dir_zones": float(active_same),
        "active_opp_zones": float(active_opp),
        "penetration_frac": _penetration_frac(direction, zone, highs[event_index], lows[event_index]),
        "reclaim_pos": _reclaim_pos(direction, highs[event_index], lows[event_index], closes[event_index]),
        "same_bar_reaction_atr": _same_bar_reaction_atr(direction, signal_price, highs[event_index], lows[event_index], atr_value),
        "same_bar_close_reaction_atr": _same_bar_close_reaction_atr(direction, signal_price, closes[event_index], atr_value),
        "same_bar_adverse_atr": _same_bar_adverse_atr(direction, signal_price, highs[event_index], lows[event_index], atr_value),
        "reaction_range_atr": (highs[event_index] - lows[event_index]) / atr_value,
        "reaction_body_atr": (closes[event_index] - opens[event_index]) / atr_value if direction == "long" else (opens[event_index] - closes[event_index]) / atr_value,
        "event_key": f"{symbol}|{direction}|{spec.entry_mode}|{pd.Timestamp(decision_row['decision_time']).isoformat()}|{zone['id']}",
    }
    row.update({column: decision_row[column] for column in base_feature_columns if column in decision_row.index})
    row.update(confluence)
    return row


def _build_event_row(
    *,
    symbol: str,
    decision_row: pd.Series,
    base_feature_columns: list[str],
    direction: str,
    zone: dict[str, Any],
    event_index: int,
    event_rank: int,
    active_same: int,
    active_opp: int,
    confluence: dict[str, Any],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    atrs: list[float],
    close_times: list[pd.Timestamp],
    spec: ZoneChannelEventSpec,
) -> dict[str, Any] | None:
    row = _build_signal_row(
        symbol=symbol,
        decision_row=decision_row,
        base_feature_columns=base_feature_columns,
        direction=direction,
        zone=zone,
        event_index=event_index,
        event_rank=event_rank,
        active_same=active_same,
        active_opp=active_opp,
        confluence=confluence,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        atrs=atrs,
        spec=spec,
    )
    if row is None:
        return None

    if spec.entry_mode == "passive_retest":
        outcome = label_passive_retest_trade_outcome(
            direction=direction,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            close_times=close_times,
            signal_index=event_index,
            limit_entry_price=float(row["entry_price"]),
            stop_price=float(row["stop_price"]),
            target_price=float(row["target_price"]),
            entry_window_bars=spec.passive_entry_window_bars,
            horizon_bars=spec.label_horizon_bars,
        )
    else:
        outcome = label_channel_trade_outcome(
            direction=direction,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            close_times=close_times,
            start_index=event_index + 1,
            entry_price=float(row["entry_price"]),
            stop_price=float(row["stop_price"]),
            target_price=float(row["target_price"]),
            horizon_bars=spec.label_horizon_bars,
        )
    if outcome is None:
        return None

    out = row.copy()
    out["entry_time"] = pd.Timestamp(outcome.get("entry_time", decision_row["decision_time"])).tz_convert("UTC")
    out["entry_delay_bars"] = float(outcome.get("entry_delay_bars", 0.0))
    out["future_r_net"] = float(outcome["future_r"]) - float(out["cost_r"])
    out.update(outcome)
    return out


def _model_feature_columns(feature_groups: dict[str, list[str]]) -> list[str]:
    ordered_groups = [
        "structural",
        "position",
        "excursion_acceptance",
        "touch_interaction",
        "swing_state",
        "channel_evolution",
        "confluence",
        "regime",
    ]
    columns: list[str] = []
    for group_name in ordered_groups:
        columns.extend(feature_groups.get(group_name, []))
    return list(dict.fromkeys(column for column in columns if column != "close_time"))


def _event_feature_groups(base_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    groups = {name: list(columns) for name, columns in base_groups.items()}
    groups["zone_context"] = [
        "zone_tf_5m",
        "zone_tf_15m",
        "zone_tf_1h",
        "zone_tf_4h",
        "zone_tf_1d",
        "zone_tf_1w",
        "zone_age_hours",
        "zone_width_pct",
        "zone_width_atr",
        "zone_rank",
        "prior_zone_touches",
        "active_same_dir_zones",
        "active_opp_zones",
        "entry_mode_market_reclaim",
        "entry_mode_passive_retest",
        "entry_delay_bars",
        "entry_distance_from_signal_atr",
    ]
    groups["zone_reaction"] = [
        "penetration_frac",
        "reclaim_pos",
        "same_bar_reaction_atr",
        "same_bar_close_reaction_atr",
        "same_bar_adverse_atr",
        "reaction_range_atr",
        "reaction_body_atr",
    ]
    groups["zone_confluence"] = [
        "stop_mode_channel_anchor",
        "stop_mode_zone",
        "stop_mode_reaction_extreme",
        "matched_boundary_gap_abs",
        "matched_boundary_gap_atr",
        "matched_boundary_value",
        "matched_boundary_is_wick",
        "matched_boundary_is_body",
        "matched_boundary_tf_5m",
        "matched_boundary_tf_15m",
        "matched_boundary_tf_1h",
        "matched_boundary_tf_4h",
        "matched_boundary_tf_1d",
        "matched_boundary_tf_1w",
        "confluence_boundary_count",
        "confluence_boundary_count_wide",
        "zone_mid_to_decision_close_atr",
        "target_rr_planned",
        "target_distance_atr",
        "stop_distance_atr",
    ]
    return groups


def _signal_columns(base_feature_columns: list[str]) -> list[str]:
    columns = [
        "symbol",
        "event_time",
        "direction",
        "zone_time",
        "zone_tf_5m",
        "zone_tf_15m",
        "zone_tf_1h",
        "zone_tf_4h",
        "zone_tf_1d",
        "zone_tf_1w",
        "zone_tf",
        "zone_top",
        "zone_bottom",
        "signal_price",
        "entry_price",
        "stop_price",
        "target_price",
        "entry_mode_market_reclaim",
        "entry_mode_passive_retest",
        "stop_mode_channel_anchor",
        "stop_mode_zone",
        "stop_mode_reaction_extreme",
        "entry_distance_from_signal_atr",
        "risk_abs",
        "cost_r",
        "target_rr_planned",
        "target_distance_atr",
        "stop_distance_atr",
        "zone_width_pct",
        "zone_width_atr",
        "zone_age_hours",
        "zone_rank",
        "prior_zone_touches",
        "active_same_dir_zones",
        "active_opp_zones",
        "penetration_frac",
        "reclaim_pos",
        "same_bar_reaction_atr",
        "same_bar_close_reaction_atr",
        "same_bar_adverse_atr",
        "reaction_range_atr",
        "reaction_body_atr",
        "event_key",
        "matched_boundary_gap_abs",
        "matched_boundary_gap_atr",
        "matched_boundary_value",
        "matched_boundary_is_wick",
        "matched_boundary_is_body",
        "matched_boundary_tf_5m",
        "matched_boundary_tf_15m",
        "matched_boundary_tf_1h",
        "matched_boundary_tf_4h",
        "matched_boundary_tf_1d",
        "matched_boundary_tf_1w",
        "confluence_boundary_count",
        "confluence_boundary_count_wide",
        "zone_mid_to_decision_close_atr",
    ]
    return list(dict.fromkeys(columns + list(base_feature_columns)))


def _event_columns(base_feature_columns: list[str]) -> list[str]:
    columns = [
        "symbol",
        "event_time",
        "entry_time",
        "direction",
        "zone_time",
        "zone_tf_5m",
        "zone_tf_15m",
        "zone_tf_1h",
        "zone_tf_4h",
        "zone_tf_1d",
        "zone_tf_1w",
        "zone_tf",
        "zone_top",
        "zone_bottom",
        "signal_price",
        "entry_price",
        "stop_price",
        "target_price",
        "entry_mode_market_reclaim",
        "entry_mode_passive_retest",
        "stop_mode_channel_anchor",
        "stop_mode_zone",
        "stop_mode_reaction_extreme",
        "entry_delay_bars",
        "entry_distance_from_signal_atr",
        "risk_abs",
        "cost_r",
        "future_r_net",
        "target_rr_planned",
        "target_distance_atr",
        "stop_distance_atr",
        "zone_width_pct",
        "zone_width_atr",
        "zone_age_hours",
        "zone_rank",
        "prior_zone_touches",
        "active_same_dir_zones",
        "active_opp_zones",
        "penetration_frac",
        "reclaim_pos",
        "same_bar_reaction_atr",
        "same_bar_close_reaction_atr",
        "same_bar_adverse_atr",
        "reaction_range_atr",
        "reaction_body_atr",
        "event_key",
        "matched_boundary_gap_abs",
        "matched_boundary_gap_atr",
        "matched_boundary_value",
        "matched_boundary_is_wick",
        "matched_boundary_is_body",
        "matched_boundary_tf_5m",
        "matched_boundary_tf_15m",
        "matched_boundary_tf_1h",
        "matched_boundary_tf_4h",
        "matched_boundary_tf_1d",
        "matched_boundary_tf_1w",
        "confluence_boundary_count",
        "confluence_boundary_count_wide",
        "zone_mid_to_decision_close_atr",
        "hold_label",
        "future_r",
        "outcome",
        "bars_to_outcome",
        "exit_time",
        "mfe_r",
        "mae_r",
    ]
    return list(dict.fromkeys(columns + list(base_feature_columns)))


def _resolve_entry_price(
    *,
    direction: str,
    entry_mode: str,
    signal_price: float,
    zone_top: float,
    zone_bottom: float,
    matched_boundary_value: float,
    atr_value: float,
    passive_entry_buffer_atr: float,
) -> float:
    if entry_mode == "passive_retest":
        buffer_abs = float(passive_entry_buffer_atr) * float(atr_value)
        if direction == "long":
            anchor = float(zone_top)
            if math.isfinite(matched_boundary_value):
                anchor = min(float(signal_price), max(anchor, float(matched_boundary_value)))
            return min(float(signal_price), anchor + buffer_abs)
        anchor = float(zone_bottom)
        if math.isfinite(matched_boundary_value):
            anchor = max(float(signal_price), min(anchor, float(matched_boundary_value)))
        return max(float(signal_price), anchor - buffer_abs)
    return float(signal_price)


def _resolve_stop_price(
    *,
    direction: str,
    stop_mode: str,
    channel_anchor: float,
    zone_top: float,
    zone_bottom: float,
    event_low: float,
    event_high: float,
    atr_value: float,
    stop_buffer_atr: float,
) -> float:
    buffer_abs = float(stop_buffer_atr) * float(atr_value)
    if direction == "long":
        if stop_mode == "zone":
            return float(zone_bottom) - buffer_abs
        if stop_mode == "reaction_extreme":
            return float(event_low) - buffer_abs
        return float(channel_anchor) - buffer_abs
    if stop_mode == "zone":
        return float(zone_top) + buffer_abs
    if stop_mode == "reaction_extreme":
        return float(event_high) + buffer_abs
    return float(channel_anchor) + buffer_abs


def _zone_interval_gap(zone_top: float, zone_bottom: float, boundary: float) -> float:
    upper = max(zone_top, zone_bottom)
    lower = min(zone_top, zone_bottom)
    if lower <= boundary <= upper:
        return 0.0
    return min(abs(boundary - upper), abs(boundary - lower))


def _finite_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value)


def _first_finite(values: list[Any]) -> float:
    for value in values:
        numeric = _finite_float(value)
        if math.isfinite(numeric):
            return numeric
    return np.nan


def _penetration_frac(direction: str, zone: dict[str, Any], high_value: float, low_value: float) -> float:
    width = abs(float(zone["top"]) - float(zone["bottom"]))
    if width <= 0.0:
        return np.nan
    if direction == "long":
        return (float(zone["top"]) - low_value) / width
    return (high_value - float(zone["bottom"])) / width


def _reclaim_pos(direction: str, high_value: float, low_value: float, close_value: float) -> float:
    sweep_range = high_value - low_value
    if sweep_range <= 0.0:
        return 0.0
    if direction == "long":
        return (close_value - low_value) / sweep_range
    return (high_value - close_value) / sweep_range


def _same_bar_reaction_atr(direction: str, entry_price: float, high_value: float, low_value: float, atr_value: float) -> float:
    if atr_value <= 0.0:
        return np.nan
    if direction == "long":
        return max(0.0, (high_value - entry_price) / atr_value)
    return max(0.0, (entry_price - low_value) / atr_value)


def _same_bar_close_reaction_atr(direction: str, entry_price: float, close_value: float, atr_value: float) -> float:
    if atr_value <= 0.0:
        return np.nan
    if direction == "long":
        return (close_value - entry_price) / atr_value
    return (entry_price - close_value) / atr_value


def _same_bar_adverse_atr(direction: str, entry_price: float, high_value: float, low_value: float, atr_value: float) -> float:
    if atr_value <= 0.0:
        return np.nan
    if direction == "long":
        return max(0.0, (entry_price - low_value) / atr_value)
    return max(0.0, (high_value - entry_price) / atr_value)
