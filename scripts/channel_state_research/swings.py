from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Pivot:
    pivot_type: str
    pivot_index: int
    confirm_index: int
    pivot_time: pd.Timestamp
    confirm_time: pd.Timestamp
    pivot_price: float
    excursion_size: float
    bars_since_previous_pivot: int


def extract_causal_swings(
    frame: pd.DataFrame,
    reversal_mult: float,
    *,
    atr_column: str = "atr",
    high_column: str = "high",
    low_column: str = "low",
    time_column: str = "close_time",
) -> list[Pivot]:
    if frame.empty:
        return []

    highs = frame[high_column].astype(float).to_list()
    lows = frame[low_column].astype(float).to_list()
    atrs = frame[atr_column].astype(float).to_list()
    times = pd.to_datetime(frame[time_column], utc=True).to_list()

    first_valid = next((idx for idx, value in enumerate(atrs) if pd.notna(value) and float(value) > 0.0), None)
    if first_valid is None:
        return []

    pivots: list[Pivot] = []
    direction = 0

    candidate_high_idx = first_valid
    candidate_low_idx = first_valid
    candidate_high = highs[first_valid]
    candidate_low = lows[first_valid]

    for index in range(first_valid + 1, len(frame)):
        atr_value = atrs[index]
        if pd.isna(atr_value) or float(atr_value) <= 0.0:
            continue
        reversal_amount = float(reversal_mult) * float(atr_value)

        if highs[index] >= candidate_high:
            candidate_high = highs[index]
            candidate_high_idx = index
        if lows[index] <= candidate_low:
            candidate_low = lows[index]
            candidate_low_idx = index

        if direction == 0:
            if candidate_low_idx < index and highs[index] - candidate_low >= reversal_amount:
                pivots.append(
                    _build_pivot(
                        pivots,
                        pivot_type="low",
                        pivot_index=candidate_low_idx,
                        confirm_index=index,
                        pivot_price=candidate_low,
                        times=times,
                    )
                )
                direction = 1
                candidate_high_idx = index
                candidate_high = highs[index]
                candidate_low_idx = index
                candidate_low = lows[index]
                continue
            if candidate_high_idx < index and candidate_high - lows[index] >= reversal_amount:
                pivots.append(
                    _build_pivot(
                        pivots,
                        pivot_type="high",
                        pivot_index=candidate_high_idx,
                        confirm_index=index,
                        pivot_price=candidate_high,
                        times=times,
                    )
                )
                direction = -1
                candidate_high_idx = index
                candidate_high = highs[index]
                candidate_low_idx = index
                candidate_low = lows[index]
                continue

        if direction >= 0:
            if highs[index] >= candidate_high:
                candidate_high = highs[index]
                candidate_high_idx = index
            if candidate_high_idx < index and candidate_high - lows[index] >= reversal_amount:
                pivots.append(
                    _build_pivot(
                        pivots,
                        pivot_type="high",
                        pivot_index=candidate_high_idx,
                        confirm_index=index,
                        pivot_price=candidate_high,
                        times=times,
                    )
                )
                direction = -1
                candidate_low_idx = index
                candidate_low = lows[index]
                candidate_high_idx = index
                candidate_high = highs[index]
                continue

        if direction <= 0:
            if lows[index] <= candidate_low:
                candidate_low = lows[index]
                candidate_low_idx = index
            if candidate_low_idx < index and highs[index] - candidate_low >= reversal_amount:
                pivots.append(
                    _build_pivot(
                        pivots,
                        pivot_type="low",
                        pivot_index=candidate_low_idx,
                        confirm_index=index,
                        pivot_price=candidate_low,
                        times=times,
                    )
                )
                direction = 1
                candidate_high_idx = index
                candidate_high = highs[index]
                candidate_low_idx = index
                candidate_low = lows[index]

    return pivots


def _build_pivot(
    pivots: list[Pivot],
    *,
    pivot_type: str,
    pivot_index: int,
    confirm_index: int,
    pivot_price: float,
    times: list[pd.Timestamp],
) -> Pivot:
    previous = pivots[-1] if pivots else None
    excursion_size = abs(float(pivot_price) - float(previous.pivot_price)) if previous is not None else 0.0
    bars_since_previous_pivot = int(pivot_index - previous.pivot_index) if previous is not None else 0
    pivot_time = pd.Timestamp(times[pivot_index]).tz_convert("UTC")
    confirm_time = pd.Timestamp(times[confirm_index]).tz_convert("UTC")
    return Pivot(
        pivot_type=pivot_type,
        pivot_index=int(pivot_index),
        confirm_index=int(confirm_index),
        pivot_time=pivot_time,
        confirm_time=confirm_time,
        pivot_price=float(pivot_price),
        excursion_size=excursion_size,
        bars_since_previous_pivot=bars_since_previous_pivot,
    )
