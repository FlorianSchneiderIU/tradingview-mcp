from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.channel_state_research.swings import Pivot

try:
    from sklearn.linear_model import LinearRegression, RANSACRegressor

    SKLEARN_CHANNELS = True
except ImportError:
    LinearRegression = None
    RANSACRegressor = None
    SKLEARN_CHANNELS = False


@dataclass(frozen=True)
class BoundaryPoint:
    index: int
    confirm_index: int
    time: pd.Timestamp
    confirm_time: pd.Timestamp
    price: float


@dataclass(frozen=True)
class LineFit:
    method: str
    slope: float
    intercept: float
    points_used: int
    median_abs_residual: float

    def evaluate(self, x_value: float) -> float:
        return float(self.slope * x_value + self.intercept)


@dataclass(frozen=True)
class ChannelSnapshot:
    upper: LineFit | None
    lower: LineFit | None
    upper_value: float
    lower_value: float
    width: float
    midline: float
    mid_slope: float
    valid: bool


def pivot_points(pivots: list[Pivot], pivot_type: str) -> list[BoundaryPoint]:
    return [
        BoundaryPoint(
            index=pivot.pivot_index,
            confirm_index=pivot.confirm_index,
            time=pivot.pivot_time,
            confirm_time=pivot.confirm_time,
            price=pivot.pivot_price,
        )
        for pivot in pivots
        if pivot.pivot_type == pivot_type
    ]


def build_body_envelope_points(
    frame: pd.DataFrame,
    *,
    side: str,
    lookback: int,
    min_separation: int,
    min_move_atr: float,
) -> list[BoundaryPoint]:
    prices = frame["body_high"].astype(float).to_numpy() if side == "upper" else frame["body_low"].astype(float).to_numpy()
    atrs = frame["atr"].astype(float).to_numpy()
    times = pd.to_datetime(frame["close_time"], utc=True).to_list()
    points: list[BoundaryPoint] = []
    last_index = -10_000
    last_price = np.nan

    for index in range(len(frame)):
        start = max(0, index - lookback + 1)
        window = prices[start : index + 1]
        price = float(prices[index])
        is_envelope = price >= float(np.nanmax(window)) if side == "upper" else price <= float(np.nanmin(window))
        if not is_envelope:
            continue
        if index - last_index < min_separation:
            continue
        if points and np.isfinite(last_price):
            minimum_move = float(atrs[index]) * float(min_move_atr) if np.isfinite(atrs[index]) else 0.0
            if abs(price - last_price) < minimum_move:
                continue
        timestamp = pd.Timestamp(times[index]).tz_convert("UTC")
        points.append(
            BoundaryPoint(
                index=index,
                confirm_index=index,
                time=timestamp,
                confirm_time=timestamp,
                price=price,
            )
        )
        last_index = index
        last_price = price

    return points


def fit_boundary_line(
    points: list[BoundaryPoint],
    *,
    method: str,
    residual_threshold: float | None = None,
) -> LineFit | None:
    if len(points) < 2:
        return None
    method_name = method.strip().lower()
    x = np.asarray([point.index for point in points], dtype=float)
    y = np.asarray([point.price for point in points], dtype=float)

    if method_name in {"last2", "last_two", "two_point"}:
        slope, intercept = _fit_two_point(x, y)
    elif method_name == "ols":
        slope, intercept = _fit_ols(x, y)
    elif method_name in {"theil_sen", "theilsen", "theil-sen"}:
        slope, intercept = _fit_theil_sen(x, y)
        method_name = "theil_sen"
    elif method_name == "ransac":
        slope, intercept = _fit_ransac(x, y, residual_threshold)
    else:
        raise ValueError(f"Unsupported boundary fit method: {method}")

    prediction = slope * x + intercept
    residual = np.abs(y - prediction)
    return LineFit(
        method=method_name,
        slope=float(slope),
        intercept=float(intercept),
        points_used=len(points),
        median_abs_residual=float(np.nanmedian(residual)) if len(residual) else 0.0,
    )


def snapshot_from_lines(upper: LineFit | None, lower: LineFit | None, x_value: float) -> ChannelSnapshot:
    if upper is None or lower is None:
        return ChannelSnapshot(
            upper=upper,
            lower=lower,
            upper_value=np.nan,
            lower_value=np.nan,
            width=np.nan,
            midline=np.nan,
            mid_slope=np.nan,
            valid=False,
        )

    upper_value = upper.evaluate(x_value)
    lower_value = lower.evaluate(x_value)
    width = upper_value - lower_value
    valid = bool(np.isfinite(width) and width > 0.0)
    return ChannelSnapshot(
        upper=upper,
        lower=lower,
        upper_value=upper_value,
        lower_value=lower_value,
        width=width,
        midline=(upper_value + lower_value) / 2.0 if valid else np.nan,
        mid_slope=(upper.slope + lower.slope) / 2.0 if valid else np.nan,
        valid=valid,
    )


def _fit_two_point(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_pair = x[-2:]
    y_pair = y[-2:]
    dx = x_pair[1] - x_pair[0]
    if abs(dx) <= 1e-12:
        return 0.0, float(y_pair[-1])
    slope = (y_pair[1] - y_pair[0]) / dx
    intercept = y_pair[1] - slope * x_pair[1]
    return float(slope), float(intercept)


def _fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, float(y[-1])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _fit_theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slopes: list[float] = []
    for left in range(len(x) - 1):
        for right in range(left + 1, len(x)):
            dx = x[right] - x[left]
            if abs(dx) <= 1e-12:
                continue
            slopes.append((y[right] - y[left]) / dx)
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def _fit_ransac(x: np.ndarray, y: np.ndarray, residual_threshold: float | None) -> tuple[float, float]:
    if not SKLEARN_CHANNELS:
        return _fit_theil_sen(x, y)

    estimator = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=max(2, min(3, len(x))),
        residual_threshold=residual_threshold if residual_threshold is not None and residual_threshold > 0.0 else None,
        random_state=7,
    )
    estimator.fit(x.reshape(-1, 1), y)
    line = estimator.estimator_
    if line is None:
        return _fit_theil_sen(x, y)
    return float(line.coef_[0]), float(line.intercept_)
